"""Locate and read the intermediate trim-pass sweeps.

The parser keeps only the untrimmed sweep and the final one. Each pass in
between is a full sweep of the same shape, and together with the per-pass
recipe they are the record of what each cut did. Pass index 0 IS the
untrimmed sweep and is already stored on the track row, so it never appears
here.
"""
import logging
import numbers
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
        # numbers.Real, NOT isinstance(v, (int, float)): a column pandas types
        # as int64 hands back np.int64, which is neither -- so every cell read
        # as None and the column vanished with no error. Real instance:
        # `8340_147_TA_Test Data_4-2-2026_3-01 PMTrimmed Correct.xls`, `Lin
        # Error`, 61 integer-zero cells silently dropped. Rare only because a
        # header string usually forces object dtype; a model writing
        # whole-number cut lengths would lose the lot. bool is an Integral, so
        # it is excluded explicitly -- a TRUE cell is not a measurement, and
        # the OLD test turned True into 1.0 and False into 0.0, fabricating
        # measurements out of flags.
        #
        # np.timedelta64 is excluded for the opposite reason: it subclasses
        # np.signedinteger, so `numbers.Real` says yes and `float()` then
        # raises TypeError -- widening the test made it raise where it used
        # to return None. A timedelta64[ns] COLUMN hands back pd.Timedelta,
        # which is not Real and lands on None safely; only a raw
        # np.timedelta64 sitting in an object column reaches here, so this is
        # a guard against a shape nothing produces today rather than an
        # observed failure. A duration is not a measurement either way.
        if (v is None or isinstance(v, (bool, np.timedelta64))
                or not isinstance(v, numbers.Real)):
            out.append(None)
        else:
            f = float(v)
            out.append(None if f != f else f)    # NaN is a blank, not a value
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


# ---------------------------------------------------------------------------
# Laser 1 (LTS, System B format) records a second thing per pass that laser 2
# does not: `TrimVolts N`, beside each `Trim N`. One COLUMN per engaged
# position, one ROW per laser increment, each cell the output voltage after
# that increment -- the material's response curve to being cut, which is what
# a cut-length model learns from. Measured on 4,972 real laser-1 files
# (2026-09-23): `TrimVolts N` exists iff `Trim N` does (the only exceptions
# are no-cut templates and one touch-up file); no header row; row 0 non-zero
# in every engaged column; zeros are end padding only (no zero between two
# readings in any of 107 passes checked).
#
# Stored as `increment_volts`, NEVER `trim_volts`: that key already carries
# the Trim Parameters sheet's "Trim Volts" SETTING into `trim_voltage`.
#
# The last reading of a column is NOT `Trim N`'s measured value at that
# position (ratio 0.94-1.23 across the corpus): this is the live reading
# during the cut, `Trim N` is the verification sweep after it. Never use one
# for the other.
# ---------------------------------------------------------------------------

XLS_MAX_COLUMNS = 256      # BIFF8 (.xls) stops at column IV

_TRIMVOLTS = re.compile(r"^trimvolts\s*(\d+)$", re.I)


def trimvolts_sheets(sheet_names: List[str]) -> Dict[int, str]:
    """{N: sheet name} for every `TrimVolts N` sheet in the workbook.

    A pass number claimed by two sheets ("TrimVolts3" and "TrimVolts 3") is
    left out rather than guessed: which one belongs to `Trim 3` is not
    something the names can say.
    """
    found: Dict[int, List[str]] = {}
    for name in sheet_names:
        m = _TRIMVOLTS.match(str(name).strip())
        if m:
            found.setdefault(int(m.group(1)), []).append(name)
    return {n: names[0] for n, names in found.items() if len(names) == 1}


def _whole_count(v: Any) -> Optional[int]:
    """A non-negative whole number, or None -- never a guess.

    Same reading as the database's `_as_int_or_none`, which stores these very
    fields as `trim_setup.points_ignored_start/end` (numeric text "7" -> 7,
    7.0 -> 7, 7.5 -> None), so the two can never disagree about a file. Two
    differences, both refusals: a TRUE cell is a flag, not a count (float(True)
    is 1.0), and a negative count is not a count.
    """
    if v is None or isinstance(v, (bool, np.bool_)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        # OverflowError: float() of an int past 1e308. It runs outside any try
        # in parse_file, so it must never raise -- a raise here would fail the
        # whole file, not just this capture.
        return None
    if f != f or not f.is_integer() or f < 0:     # NaN, inf, fractional, negative
        return None
    return int(f)


def increment_volts_frame(setup: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int]]:
    """(first_row, window) for a laser-1 file's `TrimVolts` sheets.

    From the file's own `Model Parameters`, as `trim_setup.normalise_key`
    spells them: `first_row` is `Initial Points Ignored` -- the position index
    of column 0 -- and `window` is how many columns the sheet SHOULD have,
    `Number of Readings (Lin)` - initial - ending ignored + 1. Measured on all
    6,264 local laser-1 TrimVolts sheets (2026-09-24): exact on 6,198; the
    other 66 are all NARROWER -- 11 at the .xls column cap, 55 that logged
    fewer columns (mostly 1x1 sheets, one off-by-one). `first_row` needs only
    its own field; `window` needs all three, and is None rather than a
    nonsense count when they disagree.
    """
    setup = setup or {}
    first_row = _whole_count(setup.get("initial_points_ignored"))
    ending = _whole_count(setup.get("ending_points_ignored"))
    readings = _whole_count(setup.get("number_of_readings_lin"))
    window = None
    if first_row is not None and ending is not None and readings is not None:
        span = readings - first_row - ending + 1
        window = span if span >= 1 else None
    return first_row, window


def read_increment_volts(df: pd.DataFrame, first_row: Optional[int],
                         window: Optional[int]) -> Dict[str, Any]:
    """Laser 1's `TrimVolts N` sheet: one column per engaged position, one row per laser
    increment, each cell the output voltage after it. Zeros are end padding (a position
    that converged early), never a reading; a blank is never a reading either, and nor is
    text or a TRUE/FALSE flag (the same refusals as `_col`).

    Curve k belongs to the pass's `positions[first_row + k]`. `first_row` is the position
    index of column 0 (the file's Initial Points Ignored); `window` is how many columns the
    file SHOULD have. A sheet at the .xls column limit, or narrower than its window, has
    lost positions and says so.

    A column with no reading at all -- blank at the top, with only zero padding or nothing
    below -- is a position the pass never reached, and stays in the list as [] so every
    curve keeps its index. Measured 2026-09-24: 1,897 such columns in 574 of the 6,264
    local sheets, always one trailing block (573 of the 574 are 8232-1), and not one
    reading anywhere below a leading blank.
    """
    curves: List[List[float]] = []
    for col in range(df.shape[1]):
        readings: List[float] = []
        for v in df.iloc[:, col].tolist():
            if (isinstance(v, (bool, np.bool_, np.timedelta64))
                    or not isinstance(v, numbers.Real)):
                break                 # blank, text or a flag: this position's run is over
            f = float(v)
            if f != f or f == 0.0:
                break                 # NaN blank or zero padding
            readings.append(f)
        curves.append(readings)
    n = df.shape[1]
    truncated = n >= XLS_MAX_COLUMNS or (window is not None and n < window)
    return {"increment_volts": curves, "increment_volts_first_row": first_row,
            "increment_volts_truncated": bool(truncated)}
