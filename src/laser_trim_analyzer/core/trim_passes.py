"""Locate and read the intermediate trim-pass sweeps.

The parser keeps only the untrimmed sweep and the final one. Each pass in
between is a full sweep of the same shape, and together with the per-pass
recipe they are the record of what each cut did. Pass index 0 IS the
untrimmed sweep and is already stored on the track row, so it never appears
here.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from laser_trim_analyzer.core.models import SystemType
from laser_trim_analyzer.utils.constants import SYSTEM_A_COLUMNS, SYSTEM_B_COLUMNS

logger = logging.getLogger(__name__)

# "SEC1 TRK1 2 TRM2" / "SEC1 TRK1 1" / "SEC1 TRK1 TRM"
_A = re.compile(r"^(?P<sec>\S+)\s+(?P<trk>TRK\d)\s+(?P<rest>.+)$", re.I)


def pass_sheets(sheet_names: List[str], system: SystemType,
                track_id: str) -> List[Tuple[int, str]]:
    """Ordered (pass_index, sheet_name), first cut first, untrimmed excluded."""
    found: List[Tuple[int, str]] = []
    if system == SystemType.A:
        trailing_trm: Optional[str] = None
        for name in sheet_names:
            m = _A.match(name.strip())
            if not m or m.group("trk").upper() != track_id.upper():
                continue
            parts = m.group("rest").split()
            if parts[0].isdigit():
                idx = int(parts[0])
                if idx > 0:                      # 0 is the untrimmed sweep
                    found.append((idx, name))
            elif parts[0].upper().startswith("TRM"):
                trailing_trm = name              # older naming: final pass
        found.sort()
        if trailing_trm is not None:
            found.append(((found[-1][0] + 1) if found else 1, trailing_trm))
        return found

    # Systems B and C share one layout.
    lin_error = None
    for name in sheet_names:
        low = name.strip().lower()
        if low == "lin error":
            lin_error = name
        elif low.startswith("trim ") and low[5:].strip().isdigit():
            found.append((int(low[5:].strip()), name))
    found.sort()
    if lin_error is not None:
        found.append(((found[-1][0] + 1) if found else 1, lin_error))
    return found


def _col(df: pd.DataFrame, idx: int, start_row: int) -> List[Optional[float]]:
    """One column as floats, preserving blanks as None.

    Blanks matter: an ignored point carries no limit, and turning that into
    0.0 would grade every point as failing. That exact bug cost a week on
    the final-test side in September 2026.

    This only strips blanks trailing the very end of the column itself; it
    has no idea where the sweep actually ends versus where a legitimately
    ignored point sits. `read_pass` is what re-aligns every column to the
    position column's length, which is the real end-of-sweep signal.
    """
    out: List[Optional[float]] = []
    if idx >= df.shape[1]:
        return out
    for r in range(start_row, df.shape[0]):
        v = df.iat[r, idx]
        if v is None or (isinstance(v, float) and pd.isna(v)) or not isinstance(v, (int, float)):
            out.append(None)
        else:
            out.append(float(v))
    while out and out[-1] is None:      # trailing blank rows are not data
        out.pop()
    return out


# System A pass sheets are 21 columns wide and carry per-POINT process data
# that System B does not record at all. Column indices verified against
# `SEC1 TRK1 2 TRM2` of the 8232-1 shop 243 fixture, whose header row reads:
#   11 trim target · 12 initial trim value · 13 final trim value ·
#   15 Pred. Deltas · 16 Used Deltas · 18 Cut Lengths · 19 Trim Currents
# "Cut Lengths" is the cut applied AT EACH POSITION, not one figure for the
# pass. Together with the before/after sweeps it is the whole input to a
# cut-length model, which is why it is captured now rather than later.
_A_PER_POINT: Dict[str, int] = {
    "trim_target": 11,
    "initial_trim_value": 12,
    "final_trim_value": 13,
    "pred_deltas": 15,
    "used_deltas": 16,
    "cut_lengths": 18,
    "trim_currents": 19,
}


def _aligned(df: pd.DataFrame, idx: int, start_row: int, n: int, key: str) -> List[Optional[float]]:
    """Read column `idx` and force it to length `n` (positions' length).

    `_col` only knows how to strip blanks trailing ITS OWN column; it has no
    idea where the real sweep ends. `n` — positions' length — is the real
    answer, so every other column is forced to match it here:

    - Shorter than `n`: padded with None. This is the common, legitimate
      case — an ignored point's error/limit is blank while position keeps
      going (confirmed on both fixtures this module is tested against: DLTS
      `SEC1 TRK1 2 TRM2` opens with 6 ignored points before the graded
      window starts; LTS `Trim 1` closes with 6, positions 23..27.5 real,
      ungraded).
    - Longer than `n`: truncated, but LOUDLY. This should not normally
      happen for a key `read_pass` already reads — but System B's own
      `measured_volts` column (`SYSTEM_B_COLUMNS["measured_volts"]`, not
      currently read here) runs 2 rows past where `positions` goes blank on
      every System-B pass sheet checked, with genuine non-blank data in
      those extra rows (e.g. `lts_8232-1_194.xls` `Trim 1`: 5.021913 and
      5.021891 past position's last real row). A future key added through
      this helper would silently reproduce the exact truncation-without-a-
      trace bug this module was fixed for, so the overflow case logs
      instead of vanishing quietly.
    """
    vals = _col(df, idx, start_row)
    if len(vals) > n:
        logger.warning(
            "trim_passes.read_pass: column %r (index %d) returned %d values, "
            "more than positions' %d; dropping the last %d row(s) rather than "
            "let the sweep misalign. This may mean that column legitimately "
            "outruns positions (as SYSTEM_B_COLUMNS['measured_volts'] does on "
            "every System-B pass sheet checked) and should not be read through "
            "this helper without a length policy of its own.",
            key, idx, len(vals), n, len(vals) - n,
        )
    return (vals + [None] * n)[:n]


def read_pass(df: pd.DataFrame, system: SystemType, start_row: int) -> Dict[str, Any]:
    """One pass sheet as a sweep. Keys mirror the track dict the parser builds.

    System A adds the per-point process columns above. On systems B and C
    those keys are ABSENT rather than empty, so a consumer can distinguish
    "this machine does not record it" from "it recorded nothing".

    Every returned list is aligned to the length of `positions` — see
    `_aligned` for why, and for the overflow case it guards against.
    """
    cols = SYSTEM_A_COLUMNS if system == SystemType.A else SYSTEM_B_COLUMNS
    positions = _col(df, cols["position"], start_row)
    n = len(positions)

    out: Dict[str, Any] = {
        "positions": positions,
        "errors": _aligned(df, cols["error"], start_row, n, "errors"),
        "upper_limits": _aligned(df, cols["upper_limit"], start_row, n, "upper_limits"),
        "lower_limits": _aligned(df, cols["lower_limit"], start_row, n, "lower_limits"),
    }
    if system == SystemType.A:
        for key, idx in _A_PER_POINT.items():
            out[key] = _aligned(df, idx, start_row, n, key)
    return out
