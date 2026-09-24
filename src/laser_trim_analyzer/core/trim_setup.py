"""Read the laser parameter sheets that sit beside the trim sweeps.

Three layouts exist and none of them are discoverable from the sheet name:
System B `Model Parameters` puts the VALUE first and the label second;
System A `Model Parameters` and `Track Parameters` put the label first; and
both formats' `Trim Parameters` are label-first with one column per trim
pass. Callers state the layout; this module does no guessing.

Pure: a DataFrame in, plain data out. No Excel I/O, no state, no logging of
file paths — which is what makes it testable without a workbook.
"""
import re
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Match, Optional

import numpy as np
import pandas as pd

# Keys promoted to real, indexed columns because the engine queries them
# across models and over time. Everything else stays in the stored block.
PROMOTED: Dict[str, str] = {
    "initial_resistance_lower_limit": "initial_resistance_low",
    "initial_resistance_upper_limit": "initial_resistance_high",
    "final_resistance_lower_limit": "final_resistance_low",
    "final_resistance_upper_limit": "final_resistance_high",
    "min_resistance": "final_resistance_low",
    "max_resistance": "final_resistance_high",
    "theo_resistance": "theoretical_resistance",
    "theoretical_resistance": "theoretical_resistance",
    "test_voltage": "test_voltage",
    "laser_power": "laser_power",
    "indexing_method": "indexing_method",
    "initial_points_ignored": "points_ignored_start",
    "ending_points_ignored": "points_ignored_end",
}

_PAREN = re.compile(r"\s*\(([^)]*)\)")
_NONWORD = re.compile(r"[^a-z0-9]+")


def _paren_sub(m: "Match[str]") -> str:
    """Drop a numeric-range parenthetical; fold a unit parenthetical in.

    '(0-255)' is a calibration range that drifts across firmware versions,
    so it is dropped. '(mm)' is a unit: 'Laser Cut Length (mm)' (System A,
    millimetres) and the unitless 'Laser Cut Length' (System B, raw laser
    counts) are different quantities that happen to share a label, and
    dropping the unit would collapse them onto the same key.
    """
    inner = m.group(1)
    return "" if any(ch.isdigit() for ch in inner) else " " + inner


def normalise_key(label: str) -> str:
    """'Laser Power (0-255)' -> 'laser_power'; 'Laser Cut Length (mm)' ->
    'laser_cut_length_mm'. Stable across wording drift."""
    s = _PAREN.sub(_paren_sub, str(label)).strip().lower()
    s = s.replace("#", "num").replace("%", "pct").replace(".", "")
    return _NONWORD.sub("_", s).strip("_")


def _clean(v: Any) -> Any:
    """One parameter VALUE, reduced to something JSON can hold.

    The whole dict is stored as one `SafeJSON` blob. `SafeJSON` test-serialises
    before binding and, on `TypeError`, logs and substitutes None -- so ONE
    unserialisable cell does not lose that cell, it loses THE ENTIRE
    PARAMETER BLOCK for that file, written as the literal string "null" and
    read back as `[]`. A consumer calling `.get()` on that gets an
    AttributeError, because it is a list where a dict was promised.

    Not hypothetical: `Template Updated` arrives from pandas as a `datetime`,
    and it cost the stored block on 44 of 522 real DLTS files across 23
    models (6828, 8251-1, 8706-3, 8747A-E, 8867-*, 8878, 8886, 8888, 8889,
    8914, 8922...). The promoted columns looked correct the whole time, which
    is why it went unnoticed.

    The coercion happens HERE, where the stored shape is decided, rather than
    by leaning on a `json.dumps(default=str)` in the writer: a reader should
    be able to predict what a parameter value looks like without knowing
    which serializer it went through.
    """
    if v is None:
        return None
    # numpy scalar -> plain Python scalar first, so every branch below sees a
    # builtin type and the stored JSON never carries a numpy repr.
    if isinstance(v, np.generic):
        v = v.item()
    try:
        if pd.isna(v):          # NaN, NaT, pd.NA
            return None
    except (TypeError, ValueError):
        pass                    # not a scalar pandas understands; carry on
    if isinstance(v, str):
        v = v.strip()
        return v or None
    if isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, (datetime, date, time)):
        return v.isoformat()
    if isinstance(v, timedelta):
        return str(v)
    # Anything else at all: keep the information as text rather than let it
    # take the whole block down with it.
    return str(v)


def _label(v: Any) -> Any:
    """A LABEL cell, which must be genuine text or nothing.

    Deliberately not `_clean`: `_clean` coerces a datetime to an ISO string so
    the VALUE survives JSON, and running a label through it would make a date
    sitting in the label column of a value-first sheet into a key. It does --
    `8251-1` has `2026-01-06 00:00:00 | Template Updated:` in the value-first
    `Model Parameters` layout, which briefly produced the key
    `2026_01_06t00_00_00`. A label is text or it is not a label.
    """
    if isinstance(v, np.generic):
        v = v.item()
    if not isinstance(v, str):
        return None
    v = v.strip()
    return v or None


def read_keyvalue(df: pd.DataFrame, *, label_col: int, value_col: int) -> Dict[str, Any]:
    """Normalised {key: value} from a two-column parameter sheet."""
    out: Dict[str, Any] = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        if len(row) <= max(label_col, value_col):
            continue
        label = _label(row.iloc[label_col])
        if label is None:
            continue
        key = normalise_key(label)
        if not key or key in out:          # first occurrence wins
            continue
        out[key] = _clean(row.iloc[value_col])
    return out


# Column C of a two-track System A 'Track Parameters' sheet carries Track 2's
# own block, under the SAME labels column B uses for Track 1 -- see
# read_track2_keyvalue. Corpus-verified (4,871 local DLTS files, 2026-09-24):
# a real block always shows 32 or 34 non-null values in that column; nothing
# else in the whole corpus comes close -- the next-highest count anywhere
# else is 4 (a stray hand-typed comment spanning 1-4 rows, e.g. "degrees
# from end"). No file anywhere in between. The threshold sits in the middle
# of that gap, not at its edge, so it has margin either direction.
_TRACK2_MIN_VALUES = 10


def read_track2_keyvalue(df: pd.DataFrame, *, label_col: int = 0,
                          value_col: int = 2) -> Dict[str, Any]:
    """Track 2's own block from column C of a two-track System A sheet.

    Same shape as `read_keyvalue`'s output (normalised {key: value}, first
    occurrence wins) -- but returns {} unless column C is judged a REAL
    Track 2 block, not a template artifact.

    A System A "Track Parameters" sheet always reserves a 3rd column for a
    second track, whether or not the part actually has one. Two verified
    failure modes rule out reading it unconditionally:
    - A genuinely single-track part (e.g. model 2475-10): column C is
      entirely blank, including its own header cell.
    - A single-track FILE of a genuinely dual-track part -- each track cut
      in its own file, e.g. shop 8530-112 (TRK1 only) / 8530-112A (TRK2
      only), both carrying the SAME "Track Parameters" sheet -- has column C
      fully populated with a real, distinct Track 2 block (own Alias, edge
      positions, laser height). A few OTHER single-track files instead carry
      just 1-4 stray hand-typed comments in column C (e.g. "degrees from
      end") that are not a parameter block at all.

    So "real" is decided from the block itself, never from which sheets this
    FILE happens to have (this module reads no Excel and is told no sheet
    names -- see the module docstring): the header cell (row 0) names TRK2,
    OR at least `_TRACK2_MIN_VALUES` rows carry a value under the same
    labels column B uses. The OR matters: verified on the corpus, a few real
    blocks are mislabelled -- column C's own header wrongly repeats
    "SEC1-TRK1", a copy-paste in the source workbook (e.g. shop 8532-8A) --
    while still carrying a full, genuinely different set of values. The
    header check alone would miss those; the value-count check alone would
    also flag the 1-4-cell comment noise, which the threshold excludes.
    """
    if df is None or df.empty or df.shape[1] <= value_col:
        return {}
    header = df.iloc[0, value_col] if df.shape[0] else None
    header_names_trk2 = isinstance(header, str) and "TRK2" in header.upper()
    non_null = int(df.iloc[1:, value_col].notna().sum()) if df.shape[0] > 1 else 0
    if not header_names_trk2 and non_null < _TRACK2_MIN_VALUES:
        return {}
    return read_keyvalue(df, label_col=label_col, value_col=value_col)


# The four PROMOTED resistance-limit columns findings/data.py needs out of a
# stored parameters block (Track 1's `trim_setup.parameters`, or Track 2's
# `trim_setup.track2_parameters`) with no DB row to hand it one. Same source
# of truth `_write_trim_setup` promotes onto TrimSetup, so a Track 2 block
# resolves an alias label (e.g. System B/C's "min/max resistance") exactly
# the way a Track 1 block would -- even though every Track 2 block seen in
# the corpus is System A layout and only ever carries the four canonical
# "...Resistance ... Limit" labels.
_RESISTANCE_COLUMNS = ("initial_resistance_low", "initial_resistance_high",
                       "final_resistance_low", "final_resistance_high")


def resistance_limits(parameters: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """{initial/final_resistance_low/high} out of a raw parameters block.

    For each of the four columns, the first PROMOTED key (in PROMOTED's own
    declared order) that is present in `parameters` wins -- one definition,
    shared with `_write_trim_setup`, never re-typed. `parameters` itself may
    be None or {} (no block captured); every column reads None then.
    """
    out: Dict[str, Optional[float]] = {c: None for c in _RESISTANCE_COLUMNS}
    if not parameters:
        return out
    for key, column in PROMOTED.items():
        if column not in out or out[column] is not None:
            continue
        if key not in parameters or parameters[key] is None:
            continue
        try:
            out[column] = float(parameters[key])
        except (TypeError, ValueError):
            continue
    return out


def read_per_pass(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """One dict per trim pass from a label-first, column-per-pass sheet.

    Row 0 holds the pass labels from column 1 onward. A column whose label
    and every value are blank is not a pass and is dropped — System B pads
    its sheet out to five columns whether or not five passes ran.
    """
    if df is None or df.empty or df.shape[1] < 2:
        return []
    passes: List[Dict[str, Any]] = []
    for col in range(1, df.shape[1]):
        label = _clean(df.iloc[0, col]) if df.shape[0] else None
        body = {}
        for r in range(1, df.shape[0]):
            key = _label(df.iloc[r, 0])    # a label, not a value: see _label
            if key is None:
                continue
            val = _clean(df.iloc[r, col])
            if val is not None:
                body[normalise_key(key)] = val
        if not body:
            continue
        body["label"] = str(label) if label is not None else f"pass{col}"
        passes.append(body)
    return passes
