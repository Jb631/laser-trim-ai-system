# Trim Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the per-pass trim sweeps and the laser parameter blocks that the parser currently discards, so a single reprocess produces the full dataset the recommendation engine needs.

**Architecture:** The parser gains two additive readers — one for the parameter block, one for the intermediate pass sheets. Their output rides on new optional fields of the existing data models, so the processor only passes it through. The manager writes two new tables. Nothing the parser returns today changes, and that is proved by diffing every stored field against the current tree.

**Tech Stack:** Python 3.11+ (proven on 3.14), pandas + xlrd for `.xls`, SQLAlchemy 2.0 ORM, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-process-recommendations-design.md`

## Global Constraints

- **`data/analysis.db` is the owner's production work database (3.7 GB).** Open it read-only, always: `sqlite3.connect('file:data/analysis.db?mode=ro', uri=True)`. Never open it read-write. Both QA harnesses require a copy path and refuse the real path by name.
- **Never `git add -A`.** Add files by name.
- Commit trailer: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`
- **`core/processor.py` is protected.** Only the pass-through edit in Task 7 may touch it. No changes to the trim analysis path, the incremental scan, `_classify_scan`, or `_load_processed_hashes`.
- **SQLAlchemy 2.0 syntax** (`case()` not `func.case()`). Type hints where practical. Logging via the `logging` module.
- New ORM tables are created automatically by `Base.metadata.create_all(self._engine, checkfirst=True)` at `database/manager.py:276`. New **columns on existing tables** need the idempotent `ALTER TABLE` pattern; new **tables** do not.
- **Weak assertions are forbidden in the QA sweeps.** A check that can pass on an ERROR result is a bug. Every new check must be demonstrated failing against current code before it is committed.
- Three laser systems exist: **A** (DLTS), **B** (LTS), **C** (LTS3). C is format-identical to B and is identified by an `LTS3` path segment. Anything that reads System B sheets automatically covers C.

## File Structure

| File | Responsibility |
|---|---|
| `src/laser_trim_analyzer/core/trim_setup.py` | **New.** Pure functions that turn a parameter-sheet DataFrame into a normalised dict. No Excel I/O, no state. |
| `src/laser_trim_analyzer/core/trim_passes.py` | **New.** Pure functions that enumerate pass sheets in order and read one pass sweep. |
| `src/laser_trim_analyzer/core/parser.py` | Modify. Calls the two new modules and adds keys to what it already returns. |
| `src/laser_trim_analyzer/core/models.py` | Modify. Optional fields on `TrackData` and `AnalysisResult`. |
| `src/laser_trim_analyzer/core/processor.py` | Modify, pass-through only. |
| `src/laser_trim_analyzer/database/models.py` | Modify. Two new tables. |
| `src/laser_trim_analyzer/database/manager.py` | Modify. Writes the new rows; fixes the final-test early return. |
| `tests/fixtures/trim/` | **New.** Four real sample workbooks, two per format. |
| `tests/test_trim_setup.py`, `tests/test_trim_passes.py`, `tests/test_trim_capture_noop.py` | **New.** |
| `scripts/app_qa_sweep.py` | Modify. New checks. |

Splitting the two readers out of `parser.py` is deliberate: that file is already over 1,200 lines, and these are pure, table-driven functions that are far easier to test outside it.

---

### Task 1: Fixtures

**Files:**
- Create: `tests/fixtures/trim/lts_8232-1_194.xls`, `tests/fixtures/trim/lts_8232-1_193.xls`, `tests/fixtures/trim/dlts_8232-1_243.xls`, `tests/fixtures/trim/dlts_8232-1_242.xls`

**Interfaces:**
- Consumes: nothing.
- Produces: fixture paths used by every later task.

- [ ] **Step 1: Copy four real workbooks**

```bash
mkdir -p tests/fixtures/trim
cp "Work Files/Sample_Base_2026-04-10/LTS/8232-1/8232-1_194_TA_Test Data_4-9-2026_7-41 AM_ResTrimmed Correct.xls" tests/fixtures/trim/lts_8232-1_194.xls
cp "Work Files/Sample_Base_2026-04-10/LTS/8232-1/8232-1_193_TA_Test Data_4-9-2026_7-15 AM_ResTrimmed Correct.xls" tests/fixtures/trim/lts_8232-1_193.xls
cp "Work Files/Sample_Base_2026-04-10/DLTS/8232-1/8232-1_243_TEST DATA_7-13-2022_4-58 PM.xls" tests/fixtures/trim/dlts_8232-1_243.xls
cp "Work Files/Sample_Base_2026-04-10/DLTS/8232-1/8232-1_242_TEST DATA_7-13-2022_4-34 PM.xls" tests/fixtures/trim/dlts_8232-1_242.xls
ls -la tests/fixtures/trim/
```

If a source filename differs, list the directory and take the two newest `.xls` files from each. This follows the precedent of `tests/fixtures/final_test/`.

- [ ] **Step 2: Verify the sheets these tests depend on are present**

```bash
.venv/bin/python - <<'EOF'
import pandas as pd
for f in ("tests/fixtures/trim/lts_8232-1_194.xls", "tests/fixtures/trim/dlts_8232-1_243.xls"):
    print(f, pd.ExcelFile(f).sheet_names)
EOF
```

Expected: the LTS file lists `Model Parameters`, `Trim Parameters`, `Lin Error`, `test`, `Trim 1`, `Trim 2`. The DLTS file lists `Model Parameters`, `Track Parameters`, `Trim Parameters`, and sheets matching `SEC1 TRK1 *`.

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/trim
git commit -m "test: real trim workbooks for both laser formats

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Read the parameter block

**Files:**
- Create: `src/laser_trim_analyzer/core/trim_setup.py`
- Test: `tests/test_trim_setup.py`

**Interfaces:**
- Consumes: fixtures from Task 1.
- Produces:
  - `read_keyvalue(df: pd.DataFrame, *, label_col: int, value_col: int) -> Dict[str, Any]`
  - `read_per_pass(df: pd.DataFrame) -> List[Dict[str, Any]]`
  - `PROMOTED: Dict[str, str]` — maps a normalised key to the DB column name.
  - `normalise_key(label: str) -> str`

**Domain context the implementer needs.** These sheets are laid out three different ways and the layout is not discoverable from the sheet name:

- System B `Model Parameters`: **value first, label second** — row 0 is `['8232-1', 'Model number', nan, ...]`. 62 rows.
- System A `Model Parameters` and `Track Parameters`: **label first, value second** — `['Initial Resistance Upper Limit', 4600]`.
- Both formats' `Trim Parameters`: **label first, then one column per trim pass.** System A row 0 is `['TRIM PARAMETERS', 'SEC1-TRK1-TRM1', 'SEC1-TRK1-TRM2']`; System B row 0 is `['Pass', 1, 2, 3, 4, 5]`. Row 1 onward is `[label, value_for_pass_1, value_for_pass_2, ...]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trim_setup.py
import pandas as pd
import pytest
from laser_trim_analyzer.core.trim_setup import (
    normalise_key, read_keyvalue, read_per_pass, PROMOTED)

LTS = "tests/fixtures/trim/lts_8232-1_194.xls"
DLTS = "tests/fixtures/trim/dlts_8232-1_243.xls"


def test_normalise_key_is_stable_across_wording():
    assert normalise_key("Initial Resistance Upper Limit") == "initial_resistance_upper_limit"
    assert normalise_key("Laser Power (0-255)") == "laser_power"
    assert normalise_key("  Min Resistance ") == "min_resistance"


def test_system_b_model_parameters_are_value_first():
    df = pd.read_excel(LTS, sheet_name="Model Parameters", header=None)
    kv = read_keyvalue(df, label_col=1, value_col=0)
    assert kv["model_number"] == "8232-1"
    assert kv["max_resistance"] == 5500
    assert kv["min_resistance"] == 5000
    assert kv["theo_resistance"] == 5250
    assert kv["initial_points_ignored"] == 2
    assert kv["ending_points_ignored"] == 7


def test_system_a_track_parameters_are_label_first():
    df = pd.read_excel(DLTS, sheet_name="Track Parameters", header=None)
    kv = read_keyvalue(df, label_col=0, value_col=1)
    assert kv["initial_resistance_upper_limit"] == 4600
    assert kv["initial_resistance_lower_limit"] == 4200
    assert kv["final_resistance_upper_limit"] == 5500
    assert kv["final_resistance_lower_limit"] == 5000
    assert kv["test_voltage"] == 10
    assert kv["indexing_method"] == "ERROR-SPLIT"


def test_system_a_trim_parameters_give_one_dict_per_pass():
    df = pd.read_excel(DLTS, sheet_name="Trim Parameters", header=None)
    passes = read_per_pass(df)
    assert len(passes) == 2
    assert passes[0]["label"] == "SEC1-TRK1-TRM1"
    assert passes[0]["laser_cut_length_mm"] == 0.75
    assert passes[1]["laser_cut_length_mm"] == 0.88
    assert passes[0]["laser_speed_high"] == 0.35
    assert passes[1]["laser_speed_high"] == 0.25


def test_promoted_keys_all_exist_in_a_real_sheet():
    """Every key we promote to a real column must be readable from a real file,
    or the column is dead weight."""
    a = read_keyvalue(pd.read_excel(DLTS, sheet_name="Track Parameters", header=None),
                      label_col=0, value_col=1)
    b = read_keyvalue(pd.read_excel(LTS, sheet_name="Model Parameters", header=None),
                      label_col=1, value_col=0)
    seen = set(a) | set(b)
    missing = [k for k in PROMOTED if k not in seen]
    assert not missing, f"promoted but never seen in a real file: {missing}"


def test_unreadable_sheet_yields_empty_not_an_exception():
    assert read_keyvalue(pd.DataFrame(), label_col=0, value_col=1) == {}
    assert read_per_pass(pd.DataFrame()) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_trim_setup.py -v`
Expected: all fail with `ModuleNotFoundError: No module named 'laser_trim_analyzer.core.trim_setup'`.

- [ ] **Step 3: Implement**

```python
# src/laser_trim_analyzer/core/trim_setup.py
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
from typing import Any, Dict, List

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

_PAREN = re.compile(r"\s*\([^)]*\)")
_NONWORD = re.compile(r"[^a-z0-9]+")


def normalise_key(label: str) -> str:
    """'Laser Power (0-255)' -> 'laser_power'. Stable across wording drift."""
    s = _PAREN.sub("", str(label)).strip().lower()
    s = s.replace("#", "num").replace("%", "pct").replace(".", "")
    return _NONWORD.sub("_", s).strip("_")


def _clean(v: Any) -> Any:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return v


def read_keyvalue(df: pd.DataFrame, *, label_col: int, value_col: int) -> Dict[str, Any]:
    """Normalised {key: value} from a two-column parameter sheet."""
    out: Dict[str, Any] = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        if len(row) <= max(label_col, value_col):
            continue
        label = _clean(row.iloc[label_col])
        if not isinstance(label, str):
            continue
        key = normalise_key(label)
        if not key or key in out:          # first occurrence wins
            continue
        out[key] = _clean(row.iloc[value_col])
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
            key = _clean(df.iloc[r, 0])
            if not isinstance(key, str):
                continue
            val = _clean(df.iloc[r, col])
            if val is not None:
                body[normalise_key(key)] = val
        if not body:
            continue
        body["label"] = str(label) if label is not None else f"pass{col}"
        passes.append(body)
    return passes
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_trim_setup.py -v`
Expected: 6 passed.

If `test_promoted_keys_all_exist_in_a_real_sheet` fails, the fix is to **remove the unseen key from `PROMOTED`**, not to weaken the test. A promoted column nothing writes to is dead weight.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/core/trim_setup.py tests/test_trim_setup.py
git commit -m "feat: read the laser parameter sheets

Three layouts, none discoverable from the sheet name, so callers state
which one they have rather than the reader guessing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Enumerate and read the pass sheets

**Files:**
- Create: `src/laser_trim_analyzer/core/trim_passes.py`
- Test: `tests/test_trim_passes.py`

**Interfaces:**
- Consumes: `SYSTEM_A_COLUMNS`, `SYSTEM_B_COLUMNS` from `laser_trim_analyzer.utils.constants`; `SystemType` from `laser_trim_analyzer.core.models`.
- Produces:
  - `pass_sheets(sheet_names: List[str], system: SystemType, track_id: str) -> List[Tuple[int, str]]` — ordered `(pass_index, sheet_name)`, first cut first, untrimmed excluded.
  - `read_pass(df: pd.DataFrame, system: SystemType, start_row: int) -> Dict[str, Any]` — keys `positions`, `errors`, `upper_limits`, `lower_limits`.

**Domain context.** Sheet naming differs by format and by vintage:

- System A / C, older: `SEC1 TRK1 0`, `SEC1 TRK1 1`, … then `SEC1 TRK1 TRM` for the final pass.
- System A / C, newer (e.g. 8895): `SEC1 TRK1 0`, `SEC1 TRK1 1 TRM1`, `SEC1 TRK1 2 TRM2`, … — every pass carries a `TRM` suffix and **the numeric third part is the pass index**.
- System B: `test` is untrimmed, `Trim 1` / `Trim 2` are the passes, `Lin Error` is the final state.
- **Index 0 is the untrimmed sweep and is already stored on the track row.** It must not appear in the returned list.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trim_passes.py
import pandas as pd
import pytest
from laser_trim_analyzer.core.models import SystemType
from laser_trim_analyzer.core.trim_passes import pass_sheets, read_pass

DLTS = "tests/fixtures/trim/dlts_8232-1_243.xls"
LTS = "tests/fixtures/trim/lts_8232-1_194.xls"


def test_system_a_orders_passes_and_drops_the_untrimmed_sweep():
    names = ["Model Parameters", "SEC1 TRK1 0", "SEC1 TRK1 1 TRM1",
             "SEC1 TRK1 2 TRM2", "SEC1 TRK1 3 TRM2"]
    assert pass_sheets(names, SystemType.A, "TRK1") == [
        (1, "SEC1 TRK1 1 TRM1"), (2, "SEC1 TRK1 2 TRM2"), (3, "SEC1 TRK1 3 TRM2")]


def test_system_a_older_naming_puts_trm_last():
    names = ["SEC1 TRK1 0", "SEC1 TRK1 1", "SEC1 TRK1 TRM"]
    got = pass_sheets(names, SystemType.A, "TRK1")
    assert [s for _, s in got] == ["SEC1 TRK1 1", "SEC1 TRK1 TRM"]


def test_system_a_ignores_the_other_track():
    names = ["SEC1 TRK1 1 TRM1", "SEC1 TRK2 1 TRM1"]
    assert pass_sheets(names, SystemType.A, "TRK2") == [(1, "SEC1 TRK2 1 TRM1")]


def test_system_b_orders_trim_sheets_then_lin_error():
    names = ["test", "Trim 1", "Trim 2", "Lin Error", "Notes"]
    assert pass_sheets(names, SystemType.B, "default") == [
        (1, "Trim 1"), (2, "Trim 2"), (3, "Lin Error")]


def test_system_c_reads_as_system_b():
    names = ["test", "Trim 1", "Lin Error"]
    assert pass_sheets(names, SystemType.C, "default") == [(1, "Trim 1"), (2, "Lin Error")]


def test_no_pass_sheets_is_empty_not_an_error():
    assert pass_sheets(["Notes", "hold"], SystemType.B, "default") == []


def test_read_pass_returns_a_full_sweep():
    df = pd.read_excel(DLTS, sheet_name="SEC1 TRK1 2 TRM2", header=None)
    sweep = read_pass(df, SystemType.A, start_row=1)
    assert len(sweep["positions"]) == len(sweep["errors"]) > 50
    assert max(abs(e) for e in sweep["errors"] if e is not None) == pytest.approx(0.06262, abs=1e-4)


def test_read_pass_keeps_blank_limits_as_none():
    """Ignored points carry no limit. They must stay None, never 0.0 —
    a 0.0 limit grades every point as failing."""
    df = pd.read_excel(DLTS, sheet_name="SEC1 TRK1 2 TRM2", header=None)
    sweep = read_pass(df, SystemType.A, start_row=1)
    assert any(v is None for v in sweep["upper_limits"])
    assert 0.0 not in [v for v in sweep["upper_limits"] if v is not None]


def test_system_a_pass_sheets_carry_per_point_process_data():
    """The richest thing in these files: what was cut at EVERY position, not
    one cut length for the whole pass. This is what a cut-length model learns
    from, so losing it would gut the deferred work."""
    df = pd.read_excel(DLTS, sheet_name="SEC1 TRK1 2 TRM2", header=None)
    sweep = read_pass(df, SystemType.A, start_row=1)
    assert len(sweep["cut_lengths"]) == len(sweep["positions"])
    assert any(v is not None for v in sweep["cut_lengths"])
    assert any(v is not None for v in sweep["trim_currents"])
    # Predicted vs actually-applied correction: the machine's own accuracy.
    assert "pred_deltas" in sweep and "used_deltas" in sweep


def test_system_b_has_no_per_point_process_data_and_says_so():
    """System B sheets are the plain sweep layout. The extra keys must be
    absent rather than present-and-empty, so a consumer can tell the
    difference between 'not recorded' and 'recorded as nothing'."""
    df = pd.read_excel(LTS, sheet_name="Trim 1", header=None)
    sweep = read_pass(df, SystemType.B, start_row=0)
    assert "cut_lengths" not in sweep
    assert len(sweep["positions"]) > 50
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_trim_passes.py -v`
Expected: all fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/laser_trim_analyzer/core/trim_passes.py
"""Locate and read the intermediate trim-pass sweeps.

The parser keeps only the untrimmed sweep and the final one. Each pass in
between is a full sweep of the same shape, and together with the per-pass
recipe they are the record of what each cut did. Pass index 0 IS the
untrimmed sweep and is already stored on the track row, so it never appears
here.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from laser_trim_analyzer.core.models import SystemType
from laser_trim_analyzer.utils.constants import SYSTEM_A_COLUMNS, SYSTEM_B_COLUMNS

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


def read_pass(df: pd.DataFrame, system: SystemType, start_row: int) -> Dict[str, Any]:
    """One pass sheet as a sweep. Keys mirror the track dict the parser builds.

    System A adds the per-point process columns above. On systems B and C
    those keys are ABSENT rather than empty, so a consumer can distinguish
    "this machine does not record it" from "it recorded nothing".
    """
    cols = SYSTEM_A_COLUMNS if system == SystemType.A else SYSTEM_B_COLUMNS
    out: Dict[str, Any] = {
        "positions": _col(df, cols["position"], start_row),
        "errors": _col(df, cols["error"], start_row),
        "upper_limits": _col(df, cols["upper_limit"], start_row),
        "lower_limits": _col(df, cols["lower_limit"], start_row),
    }
    if system == SystemType.A:
        n = len(out["positions"])
        for key, idx in _A_PER_POINT.items():
            vals = _col(df, idx, start_row)
            # Pad or trim to the sweep length so every per-point list lines up
            # index-for-index with positions; a short column means the sheet
            # stopped recording, not that the sweep was shorter.
            out[key] = (vals + [None] * n)[:n]
    return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_trim_passes.py -v`
Expected: 8 passed. If the fixture's sheet names differ, print `pd.ExcelFile(DLTS).sheet_names` and correct the test's literals — never loosen the assertion.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/core/trim_passes.py tests/test_trim_passes.py
git commit -m "feat: locate and read the intermediate trim passes

Blank limits stay None. A 0.0 limit grades every point as failing, which
is the bug that cost a week on the final-test side.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Optional fields on the data models

**Files:**
- Modify: `src/laser_trim_analyzer/core/models.py` (TrackData ~line 135-190, AnalysisResult ~line 280-303)
- Test: `tests/test_trim_capture_models.py`

**Interfaces:**
- Produces: `TrackData.trim_passes: List[Dict[str, Any]]`, `AnalysisResult.trim_setup: Optional[Dict[str, Any]]`.

Both default to empty so every existing construction site keeps working untouched.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trim_capture_models.py
from laser_trim_analyzer.core.models import AnalysisStatus, TrackData


def _track(**kw):
    base = dict(track_id="TRK1", status=AnalysisStatus.PASS,
                travel_length=55.0, linearity_spec=0.01)
    base.update(kw)
    return TrackData(**base)


def test_trim_passes_defaults_to_empty_so_existing_callers_are_unaffected():
    assert _track().trim_passes == []


def test_trim_passes_round_trips():
    t = _track(trim_passes=[{"pass_index": 1, "label": "TRM1",
                             "positions": [0.0], "errors": [0.1]}])
    assert t.trim_passes[0]["label"] == "TRM1"


def test_analysis_result_setup_defaults_to_none():
    from laser_trim_analyzer.core.models import AnalysisResult, FileMetadata, SystemType
    from datetime import datetime
    r = AnalysisResult(
        metadata=FileMetadata(filename="x.xls", file_path="x.xls", model="8232-1",
                              serial="1", system=SystemType.B, file_date=datetime.now()),
        overall_status=AnalysisStatus.PASS, processing_time=0.1, tracks=[_track()])
    assert r.trim_setup is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_trim_capture_models.py -v`
Expected: FAIL — pydantic rejects the unknown field `trim_passes`.

- [ ] **Step 3: Implement**

In `TrackData`, after the `measured_electrical_angle` field:

```python
    trim_passes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="One entry per laser trim pass: the sweep at that stage plus "
                    "the recipe used. Empty when the file records no intermediate "
                    "passes, or when read from a database written before capture.")
```

In `AnalysisResult`, after `data_quality_issues`:

```python
    trim_setup: Optional[Dict[str, Any]] = Field(
        None,
        description="The laser parameter block for this file: resistance limits, "
                    "laser settings, ignored-point counts. None for files that "
                    "carry no parameter sheet.")
```

Ensure `Any` and `Dict` are in the `typing` import at the top of the file.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_trim_capture_models.py -v`
Expected: 3 passed.

- [ ] **Step 5: Confirm nothing else broke**

Run: `.venv/bin/python -m pytest tests/test_models.py tests/test_processor.py -q --junitxml=/tmp/j.xml; python -c "import xml.etree.ElementTree as E;r=E.parse('/tmp/j.xml').getroot();print(r.attrib)"`
Expected: `failures='0' errors='0'`. Read the counts from the XML — `pytest -q` omits the summary line in 9.1.1.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/core/models.py tests/test_trim_capture_models.py
git commit -m "feat: data models carry trim passes and setup

Both default to empty so the processor can pass them through without
inspecting them and every existing construction site is unaffected.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Wire the readers into the parser

**Files:**
- Modify: `src/laser_trim_analyzer/core/parser.py` — `_extract_track_data` return block at ~line 1013-1035, `_extract_system_a_tracks` ~line 528, `_extract_system_b_tracks` ~line 615, `parse_file` ~line 55
- Test: `tests/test_trim_passes.py` (append)

**Interfaces:**
- Consumes: `trim_setup.read_keyvalue`, `trim_setup.read_per_pass`, `trim_passes.pass_sheets`, `trim_passes.read_pass`.
- Produces: track dict gains `"trim_passes"`; the `parse_file` result dict gains `"trim_setup"`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_trim_passes.py
from pathlib import Path
from laser_trim_analyzer.core.parser import ExcelParser


def test_parser_attaches_passes_to_the_track():
    result = ExcelParser().parse_file(Path(DLTS))
    track = result["tracks"][0]
    assert len(track["trim_passes"]) >= 2
    first = track["trim_passes"][0]
    assert first["pass_index"] == 1
    assert len(first["errors"]) > 50
    assert first["laser_cut_length_mm"] == 0.75   # recipe joined to sweep


def test_parser_attaches_setup_to_the_file():
    result = ExcelParser().parse_file(Path(DLTS))
    setup = result["trim_setup"]
    assert setup["initial_resistance_lower_limit"] == 4200
    assert setup["final_resistance_upper_limit"] == 5500


def test_system_b_setup_carries_the_final_resistance_spec():
    result = ExcelParser().parse_file(Path(LTS))
    setup = result["trim_setup"]
    assert setup["min_resistance"] == 5000
    assert setup["max_resistance"] == 5500


def test_everything_the_parser_returned_before_is_unchanged():
    """The keys that existed before capture must still be present."""
    track = ExcelParser().parse_file(Path(DLTS))["tracks"][0]
    for key in ("track_id", "positions", "errors", "upper_limits", "lower_limits",
                "untrimmed_positions", "untrimmed_errors", "travel_length",
                "linearity_spec", "untrimmed_resistance", "trimmed_resistance"):
        assert key in track, f"capture dropped the pre-existing key {key}"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_trim_passes.py -k "parser or setup or unchanged" -v`
Expected: the first three fail with `KeyError`; the fourth passes already and is a guard against regression.

- [ ] **Step 3: Implement**

Add near the top of `parser.py`:

```python
from laser_trim_analyzer.core import trim_passes as _tp
from laser_trim_analyzer.core import trim_setup as _ts
```

Add this helper as a method on the parser class:

```python
    def _read_trim_passes(self, xl, system_type, track_id, start_row, recipes):
        """Sweeps for every pass after the untrimmed one, each joined to its recipe.

        Never raises: a file whose pass sheets are malformed must still parse
        for everything else. Returns [] when there is nothing to read.
        """
        out = []
        try:
            for idx, sheet in _tp.pass_sheets(xl.sheet_names, system_type, track_id):
                try:
                    df = pd.read_excel(xl, sheet_name=sheet, header=None)
                except Exception:
                    continue
                entry = {"pass_index": idx, "sheet": sheet}
                entry.update(_tp.read_pass(df, system_type, start_row))
                if 0 < idx <= len(recipes):
                    entry.update({k: v for k, v in recipes[idx - 1].items()
                                  if k not in entry})
                out.append(entry)
        except Exception:
            logger.warning("Could not read trim passes for %s", track_id, exc_info=True)
        return out
```

In `_extract_system_a_tracks`, before the per-track loop:

```python
        try:
            recipes = _ts.read_per_pass(
                pd.read_excel(xl, sheet_name="Trim Parameters", header=None))
        except Exception:
            recipes = []
```

and where each track dict is finished, add:

```python
            track["trim_passes"] = self._read_trim_passes(
                xl, SystemType.A, track_id, start_row, recipes)
```

Do the same in `_extract_system_b_tracks` with `SystemType.B` and `track_id="default"`. `start_row` is the value that function already computed for the trimmed sheet; reuse it rather than recomputing.

In `parse_file`, after tracks are extracted and before the return, add:

```python
        setup = {}
        for sheet, label_col, value_col in (
                ("Track Parameters", 0, 1),     # System A
                ("Model Parameters", 0, 1),     # System A
                ("Model Parameters", 1, 0)):    # System B/C: value first
            if sheet not in xl.sheet_names:
                continue
            try:
                got = _ts.read_keyvalue(
                    pd.read_excel(xl, sheet_name=sheet, header=None),
                    label_col=label_col, value_col=value_col)
            except Exception:
                continue
            # A layout that produced nothing was the wrong layout for this file.
            if len(got) >= 3:
                for k, v in got.items():
                    setup.setdefault(k, v)
        result["trim_setup"] = setup or None
```

Adapt the last line to however `parse_file` names the dict it returns.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_trim_passes.py tests/test_trim_setup.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/core/parser.py tests/test_trim_passes.py
git commit -m "feat: parser keeps the passes it used to throw away

Additive only. Every key the track dict carried before is still there, and
a malformed pass sheet degrades to an empty list rather than failing the file.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The two new tables

**Files:**
- Modify: `src/laser_trim_analyzer/database/models.py` (after `class TrackResult`, ~line 337)
- Test: `tests/test_trim_capture_db.py`

**Interfaces:**
- Produces: ORM classes `TrimPass` and `TrimSetup`.

New tables are created by `Base.metadata.create_all(..., checkfirst=True)` at `manager.py:276`. No `ALTER TABLE` migration is needed — that pattern is only for new columns on existing tables.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trim_capture_db.py
from laser_trim_analyzer.database.manager import DatabaseManager


def test_new_tables_are_created_on_a_fresh_database(tmp_path):
    db = DatabaseManager(tmp_path / "t.db")
    with db.session() as s:
        names = {r[0] for r in s.execute(
            __import__("sqlalchemy").text(
                "SELECT name FROM sqlite_master WHERE type='table'"))}
    assert "trim_passes" in names
    assert "trim_setup" in names


def test_trim_pass_columns_are_what_the_engine_needs(tmp_path):
    db = DatabaseManager(tmp_path / "t.db")
    import sqlalchemy as sa
    with db.session() as s:
        cols = {r[1] for r in s.execute(sa.text("PRAGMA table_info(trim_passes)"))}
    for c in ("track_result_id", "pass_index", "sheet", "positions", "errors",
              "upper_limits", "lower_limits", "laser_cut_length", "laser_speed_high",
              "trim_voltage", "cut_lengths", "trim_currents", "used_deltas"):
        assert c in cols, f"missing {c}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_trim_capture_db.py -v`
Expected: FAIL — `trim_passes` not in the table list.

- [ ] **Step 3: Implement**

```python
class TrimPass(Base):
    """One laser cut: the sweep measured after it, and the recipe that made it.

    The parser used to keep only the untrimmed sweep and the final one. This
    is everything in between — the record of what each cut actually did to
    the curve, which is what a cut-length model would learn from.
    """
    __tablename__ = 'trim_passes'

    id = Column(Integer, primary_key=True)
    track_result_id = Column(Integer, ForeignKey('track_results.id', ondelete='CASCADE'),
                             nullable=False)
    pass_index = Column(Integer, nullable=False)   # 1 = first cut; 0 is the untrimmed sweep
    sheet = Column(String(64))
    label = Column(String(64))

    positions = Column(SafeJSON, nullable=True)
    errors = Column(SafeJSON, nullable=True)
    upper_limits = Column(SafeJSON, nullable=True)
    lower_limits = Column(SafeJSON, nullable=True)

    # Per-POINT process data. System A only; NULL on B and C, which do not
    # record it. `cut_lengths` is the cut applied at each position — the
    # single most valuable column in these files.
    cut_lengths = Column(SafeJSON, nullable=True)
    trim_currents = Column(SafeJSON, nullable=True)
    pred_deltas = Column(SafeJSON, nullable=True)
    used_deltas = Column(SafeJSON, nullable=True)
    trim_target = Column(SafeJSON, nullable=True)
    final_trim_value = Column(SafeJSON, nullable=True)

    # Per-PASS recipe, from the Trim Parameters sheet.
    laser_cut_length = Column(Float)
    laser_speed_high = Column(Float)
    laser_speed_low = Column(Float)
    trim_voltage = Column(Float)
    trim_upper_tolerance = Column(Float)
    trim_lower_tolerance = Column(Float)
    recipe = Column(SafeJSON, nullable=True)       # everything else from the sheet

    created_date = Column(DateTime, default=utc_now, nullable=False)

    __table_args__ = (
        Index('idx_trimpass_track', 'track_result_id'),
        Index('idx_trimpass_track_idx', 'track_result_id', 'pass_index', unique=True),
    )


class TrimSetup(Base):
    """The laser parameter block for one analysed file.

    Promoted columns are the ones the engine queries across models and over
    time; `parameters` keeps the whole block so nothing is lost to a schema
    decision made today.
    """
    __tablename__ = 'trim_setup'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_results.id', ondelete='CASCADE'),
                         nullable=False, unique=True)

    initial_resistance_low = Column(Float)
    initial_resistance_high = Column(Float)
    final_resistance_low = Column(Float)
    final_resistance_high = Column(Float)
    theoretical_resistance = Column(Float)
    test_voltage = Column(Float)
    laser_power = Column(Float)
    indexing_method = Column(String(32))
    points_ignored_start = Column(Integer)
    points_ignored_end = Column(Integer)

    parameters = Column(SafeJSON, nullable=True)
    created_date = Column(DateTime, default=utc_now, nullable=False)

    __table_args__ = (
        Index('idx_trimsetup_analysis', 'analysis_id'),
        Index('idx_trimsetup_initial_low', 'initial_resistance_low'),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_trim_capture_db.py -v`
Expected: 2 passed.

- [ ] **Step 5: Confirm an existing database picks them up too**

```bash
cp data/analysis.db /tmp/schema_check.db
.venv/bin/python -c "
from laser_trim_analyzer.database.manager import DatabaseManager
import sqlalchemy as sa
db = DatabaseManager('/tmp/schema_check.db')
with db.session() as s:
    n = {r[0] for r in s.execute(sa.text(\"SELECT name FROM sqlite_master WHERE type='table'\"))}
print('trim_passes' in n, 'trim_setup' in n)"
rm -f /tmp/schema_check.db*
```
Expected: `True True`. **The copy, never `data/analysis.db` itself.**

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/database/models.py tests/test_trim_capture_db.py
git commit -m "feat: tables for the trim passes and the laser setup

Promoted columns for what the engine queries; the whole parameter block
kept alongside so nothing is lost to a schema decision made today.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Carry it through the processor and write it

**Files:**
- Modify: `src/laser_trim_analyzer/core/processor.py` — the trim branch that builds `TrackData` (~line 287) and `AnalysisResult` (~line 439)
- Modify: `src/laser_trim_analyzer/database/manager.py` — `_map_track_to_db` ~line 3397, `save_analysis` ~line 3391, `_update_existing_analysis` ~line 3751
- Test: `tests/test_trim_capture_db.py` (append)

**Interfaces:**
- Consumes: `TrackData.trim_passes`, `AnalysisResult.trim_setup` from Task 4; `TrimPass`, `TrimSetup` from Task 6.
- Produces: `DatabaseManager._write_trim_passes(session, db_track, track)` and `_write_trim_setup(session, analysis_id, setup)`.

**This is the only edit permitted to `processor.py`.** Two lines that copy a value from the parser dict into the model. Nothing else in that file changes.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_trim_capture_db.py
from pathlib import Path
import sqlalchemy as sa


def test_pipeline_writes_passes_and_setup(tmp_path, monkeypatch):
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.core.processor import Processor
    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    import laser_trim_analyzer.database as dbpkg
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)

    proc = Processor(use_ml=False)
    result = proc.process_file(Path("tests/fixtures/trim/dlts_8232-1_243.xls"))
    db.save_analysis(result)

    with db.session() as s:
        passes = s.execute(sa.text(
            "SELECT pass_index, laser_cut_length FROM trim_passes ORDER BY pass_index")).all()
        setup = s.execute(sa.text(
            "SELECT initial_resistance_low, final_resistance_high FROM trim_setup")).first()
    assert [p[0] for p in passes] == [1, 2, 3]
    assert passes[0][1] == 0.75
    assert setup == (4200.0, 5500.0)


def test_saving_twice_does_not_duplicate_passes(tmp_path, monkeypatch):
    """A reprocess must refresh, not accumulate."""
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.core.processor import Processor
    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    proc = Processor(use_ml=False)
    p = Path("tests/fixtures/trim/dlts_8232-1_243.xls")
    db.save_analysis(proc.process_file(p))
    db.save_analysis(proc.process_file(p))
    with db.session() as s:
        n = s.execute(sa.text("SELECT COUNT(*) FROM trim_passes")).scalar()
    assert n == 3, f"expected 3 pass rows after two saves, got {n}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_trim_capture_db.py -k pipeline -v`
Expected: FAIL — no rows in `trim_passes`.

- [ ] **Step 3: Implement the processor pass-through**

Where the trim branch builds each `TrackData`, add one argument:

```python
                        trim_passes=track_data.get("trim_passes") or [],
```

Where it builds the `AnalysisResult`, add one argument:

```python
                trim_setup=parsed.get("trim_setup"),
```

Use whatever local name holds the parser's result dict. **No other change to this file.**

- [ ] **Step 4: Implement the writes**

In `manager.py`, add two helpers and call them where tracks are persisted (both `save_analysis`'s insert path near line 3391 and `_update_existing_analysis` near line 3751):

```python
    def _write_trim_passes(self, session, db_track, track) -> None:
        """Replace this track's pass rows. Idempotent: a reprocess refreshes."""
        from laser_trim_analyzer.database.models import TrimPass
        passes = getattr(track, "trim_passes", None) or []
        session.query(TrimPass).filter(
            TrimPass.track_result_id == db_track.id).delete(synchronize_session=False)
        per_point = ("cut_lengths", "trim_currents", "pred_deltas",
                     "used_deltas", "trim_target", "final_trim_value")
        for p in passes:
            recipe = {k: v for k, v in p.items()
                      if k not in ("positions", "errors", "upper_limits",
                                   "lower_limits", "pass_index", "sheet")
                      and k not in per_point}
            session.add(TrimPass(
                track_result_id=db_track.id,
                pass_index=p.get("pass_index"),
                sheet=p.get("sheet"),
                label=str(p.get("label"))[:64] if p.get("label") else None,
                positions=p.get("positions"), errors=p.get("errors"),
                upper_limits=p.get("upper_limits"), lower_limits=p.get("lower_limits"),
                **{k: p.get(k) for k in per_point},
                laser_cut_length=_as_float(p.get("laser_cut_length_mm")
                                           or p.get("laser_cut_length")),
                laser_speed_high=_as_float(p.get("laser_speed_high")),
                laser_speed_low=_as_float(p.get("laser_speed_low")),
                trim_voltage=_as_float(p.get("trim_volts") or p.get("trim_voltage")),
                trim_upper_tolerance=_as_float(p.get("trim_upper_tolerance")),
                trim_lower_tolerance=_as_float(p.get("trim_lower_tolerance")),
                recipe=recipe or None,
            ))

    def _write_trim_setup(self, session, analysis_id: int, setup) -> None:
        """Replace this analysis's setup row. Idempotent."""
        from laser_trim_analyzer.core.trim_setup import PROMOTED
        from laser_trim_analyzer.database.models import TrimSetup
        if not setup:
            return
        session.query(TrimSetup).filter(
            TrimSetup.analysis_id == analysis_id).delete(synchronize_session=False)
        row = TrimSetup(analysis_id=analysis_id, parameters=setup)
        for key, column in PROMOTED.items():
            if key in setup and getattr(row, column, None) is None:
                value = setup[key]
                setattr(row, column, value if column == "indexing_method"
                        else _as_float(value))
        session.add(row)
```

And a module-level helper beside the other small utilities:

```python
def _as_float(v):
    """Float or None. Parameter sheets carry text in numeric cells."""
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None
```

`indexing_method` is a string; everything else promoted is numeric, which is why it is the one exception above.

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_trim_capture_db.py -v`
Expected: 4 passed.

- [ ] **Step 6: Confirm the processor diff is two lines**

```bash
git diff --stat src/laser_trim_analyzer/core/processor.py
git diff src/laser_trim_analyzer/core/processor.py | grep '^+' | grep -v '^+++'
```
Expected: exactly the two added arguments. If anything else appears, revert it.

- [ ] **Step 7: Commit**

```bash
git add src/laser_trim_analyzer/core/processor.py src/laser_trim_analyzer/database/manager.py tests/test_trim_capture_db.py
git commit -m "feat: store the passes and the setup

Writes replace rather than append, so a reprocess refreshes instead of
accumulating. The processor change is two arguments and nothing else.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Make a reprocess actually refresh final-test rows

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py` — `save_final_test`, the content-hash early return roughly 35 lines into the method
- Test: `tests/test_ft_reprocess_refresh.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: no new callable; changes the behaviour of `save_final_test` on a content match.

**Why this is here.** Today, a re-read final-test file whose content already matches a stored row returns that row's id and updates nothing. Trim rows update in place; final-test rows do not. A reprocess therefore refreshes half of what you expect and silently skips the rest. The spec's fresh-database route works around it; this fixes it so future reprocesses are repeatable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ft_reprocess_refresh.py
import sqlalchemy as sa
from laser_trim_analyzer.database.manager import DatabaseManager


def _meta(name="x.xls"):
    from datetime import datetime
    return {"filename": name, "file_path": f"/tmp/{name}", "model": "8232-1",
            "serial": "1", "file_date": datetime(2026, 1, 1)}


def test_resaving_the_same_content_refreshes_the_verdict(tmp_path):
    db = DatabaseManager(tmp_path / "t.db")
    tracks = [{"track_id": "default", "linearity_pass": False,
               "linearity_fail_points": 4, "linearity_error": 0.02}]
    first = db.save_final_test(metadata=_meta(), tracks=tracks,
                               test_results={"linearity_pass": False},
                               file_hash="abc123", file_size=10,
                               file_modified_date=None)
    tracks[0].update(linearity_pass=True, linearity_fail_points=0, linearity_error=0.004)
    second = db.save_final_test(metadata=_meta(), tracks=tracks,
                                test_results={"linearity_pass": True},
                                file_hash="abc123", file_size=10,
                                file_modified_date=None)
    assert second == first, "must update the same row, not create another"
    with db.session() as s:
        verdict = s.execute(sa.text(
            "SELECT linearity_pass FROM final_test_results WHERE id=:i"),
            {"i": first}).scalar()
        rows = s.execute(sa.text("SELECT COUNT(*) FROM final_test_results")).scalar()
    assert bool(verdict) is True, "the stored verdict was not refreshed"
    assert rows == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ft_reprocess_refresh.py -v`
Expected: FAIL on `the stored verdict was not refreshed` — the early return leaves the old verdict in place.

- [ ] **Step 3: Implement**

Find the content-hash early return near the top of `save_final_test` — it currently reads roughly:

```python
                    existing = (
                        session.query(DBFinalTestResult)
                        .filter(DBFinalTestResult.file_hash == file_hash)
                        .first()
                    )
                    if existing:
                        if file_size is not None and existing.file_size is None:
                            existing.file_size = file_size
                            existing.file_modified_date = file_modified_date
                            session.commit()
                        logger.debug(f"Final test already exists: {metadata.get('filename')}")
                        return existing.id
```

Replace the `if existing:` body with this, keeping the query above it unchanged:

```python
                    if existing:
                        dup_id = existing.id
                        dup_path_on_record = existing.file_path
                        # Stamp the stat while we have it; a legacy row may
                        # carry none, and the scan's fast path needs it.
                        if file_size is not None and existing.file_size is None:
                            existing.file_size = file_size
                            existing.file_modified_date = file_modified_date
                            session.commit()
```

then, after that `with self.session()` block closes (so no write nests inside a read transaction):

```python
            this_path = str(metadata.get("file_path") or "")
            if this_path and this_path != (dup_path_on_record or ""):
                # Same content under a second path: remember THIS path so the
                # scan stops re-offering it every run.
                self._mark_ft_duplicate_path(
                    metadata, this_path, file_hash, file_size, file_modified_date,
                    f"same content as final_test_results id {dup_id}")
            # Refresh the stored verdict from this parse. Trim rows have always
            # updated in place; final-test rows returned early and did not, so a
            # reprocess refreshed half of what anyone would expect.
            try:
                self.apply_final_test_regrade(dup_id, tracks, test_results)
            except Exception:
                logger.warning("Could not refresh final test %s on reprocess",
                               metadata.get("filename"), exc_info=True)
            return dup_id
```

`apply_final_test_regrade` already replaces a result's verdict and its track rows and is used by the re-grade pass, so reuse it rather than writing a second updater. Restructure the surrounding `try`/`with` nesting as needed so `dup_id` and `dup_path_on_record` are in scope; the rule is only that the write happens outside the read session. The genuinely-new-file path below is untouched.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ft_reprocess_refresh.py tests/test_ft_graded_window.py tests/test_failed_file_markers.py -v`
Expected: all pass. The graded-window and marker suites are included because they exercise the same save path.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py tests/test_ft_reprocess_refresh.py
git commit -m "fix: a reprocess refreshes final-test rows instead of skipping them

Trim rows updated in place and final-test rows did not, so a reprocess
quietly refreshed half of what anyone would expect.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Prove it changed nothing, and guard it in the sweep

**Files:**
- Create: `tests/test_trim_capture_noop.py`
- Modify: `scripts/app_qa_sweep.py` — the `--only ingest` section
- Test: both of the above

**Interfaces:**
- Consumes: everything above.
- Produces: `check_trim_capture(db_path)` in the sweep.

- [ ] **Step 1: Write the no-op proof**

```python
# tests/test_trim_capture_noop.py
"""Every field that existed before capture must be byte-identical after it.

The method: run the same real files through the tree as it was before this
work and through the tree as it is now, into two throwaway databases, and
diff every stored column. This caught two real bugs in September 2026 and is
the only reason a parser change of this size is safe.
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "tests" / "fixtures" / "parse_baseline.json"
FIXTURES = sorted((REPO / "tests" / "fixtures" / "trim").glob("*.xls"))

PRE_EXISTING = [
    "track_id", "status", "linearity_spec", "linearity_pass", "linearity_fail_points",
    "sigma_gradient", "sigma_threshold", "sigma_pass", "optimal_offset", "optimal_slope",
    "untrimmed_resistance", "trimmed_resistance", "resistance_change",
    "resistance_change_percent", "travel_length", "unit_length", "trim_pass_count",
    "untrimmed_rms_error", "untrimmed_error_max", "trimmed_rms_error",
    "position_data", "error_data", "upper_limits", "lower_limits",
    "untrimmed_positions", "untrimmed_errors",
]


@pytest.mark.skipif(not FIXTURES, reason="trim fixtures absent")
def test_no_pre_existing_field_changed():
    """Compares against a baseline captured from the pre-capture tree.

    Regenerate ONLY from a commit before this work:
        git stash && python tests/test_trim_capture_noop.py --write-baseline && git stash pop
    """
    if not BASELINE.exists():
        pytest.skip("baseline not captured; see the docstring")
    expected = json.loads(BASELINE.read_text())
    actual = _dump()
    assert set(actual) == set(expected), "a fixture file appeared or vanished"
    diffs = []
    for name, fields in expected.items():
        for key in PRE_EXISTING:
            if fields.get(key) != actual[name].get(key):
                diffs.append(f"{name}.{key}: {fields.get(key)!r} -> {actual[name].get(key)!r}")
    assert not diffs, "capture changed pre-existing values:\n" + "\n".join(diffs[:20])


def _dump():
    """{filename: {field: value}} for every fixture, via the real pipeline."""
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    out = {}
    with tempfile.TemporaryDirectory() as d:
        db = mgr.DatabaseManager(Path(d) / "noop.db")
        mgr._db_manager = db
        proc = Processor(use_ml=False)
        for f in FIXTURES:
            result = proc.process_file(f)
            db.save_analysis(result)
        import sqlalchemy as sa
        with db.session() as s:
            cols = ", ".join(f"t.{c}" for c in PRE_EXISTING)
            for row in s.execute(sa.text(
                    f"SELECT a.filename, {cols} FROM track_results t "
                    "JOIN analysis_results a ON a.id = t.analysis_id")):
                out[f"{row[0]}::{row[1]}"] = dict(zip(PRE_EXISTING, row[1:]))
    return out


if __name__ == "__main__":
    if "--write-baseline" in sys.argv:
        BASELINE.write_text(json.dumps(_dump(), indent=1, default=str))
        print(f"baseline written: {BASELINE}")
```

- [ ] **Step 2: Capture the baseline from the pre-capture tree**

```bash
git stash
PYTHONPATH=$PWD/src .venv/bin/python tests/test_trim_capture_noop.py --write-baseline
git stash pop
git add tests/fixtures/parse_baseline.json
```

If `tests/fixtures/parse_baseline.json` already exists from earlier work, write to `parse_baseline_trim.json` instead and update the constant.

- [ ] **Step 3: Run the proof**

Run: `.venv/bin/python -m pytest tests/test_trim_capture_noop.py -v`
Expected: PASS with no diffs. **A failure here is a real regression — investigate it, never regenerate the baseline to make it green.**

- [ ] **Step 4: Add the sweep check, and make it fail first**

```python
def check_trim_capture(db_path):
    """Passes and setup are captured, and blank limits never become 0.0."""
    from pathlib import Path
    from laser_trim_analyzer.core.parser import ExcelParser
    fixtures = sorted(Path("tests/fixtures/trim").glob("*.xls"))
    if not fixtures:
        return [("trim capture: fixtures present", "SKIP", "tests/fixtures/trim is empty")]
    out, parser = [], ExcelParser()
    with_passes = with_setup = 0
    for f in fixtures:
        parsed = parser.parse_file(f)
        if parsed.get("trim_setup"):
            with_setup += 1
        for track in parsed.get("tracks") or []:
            passes = track.get("trim_passes") or []
            if passes:
                with_passes += 1
            for p in passes:
                ups = p.get("upper_limits") or []
                if any(v == 0.0 for v in ups if v is not None):
                    out.append(("trim capture: a blank limit became 0.0", "FAIL",
                                f"{f.name} pass {p.get('pass_index')}"))
    out.append(("trim capture: passes read from every fixture",
                "PASS" if with_passes >= len(fixtures) else "FAIL",
                f"{with_passes} tracks with passes from {len(fixtures)} files"))
    out.append(("trim capture: setup read from every fixture",
                "PASS" if with_setup == len(fixtures) else "FAIL",
                f"{with_setup} of {len(fixtures)}"))
    return out
```

Register it in the `--only ingest` group alongside the existing ingest checks.

To prove it can fail, temporarily make `trim_passes.read_pass` return `0.0` where it returns `None`, run the sweep, confirm the blank-limit check goes FAIL, then revert:

```bash
cp data/analysis.db /tmp/qa_copy.db
.venv/bin/python scripts/app_qa_sweep.py /tmp/qa_copy.db --only ingest
rm -f /tmp/qa_copy.db*
```

- [ ] **Step 5: Run the full suites**

```bash
cp data/analysis.db /tmp/qa_copy.db
.venv/bin/python -m pytest tests -x -q --junitxml=/tmp/junit.xml -p no:cacheprovider
python -c "import xml.etree.ElementTree as E;print(E.parse('/tmp/junit.xml').getroot().attrib)"
.venv/bin/python scripts/app_qa_sweep.py /tmp/qa_copy.db
.venv/bin/python scripts/chart_qa_render_all.py qa_output /tmp/qa_copy.db
rm -f /tmp/qa_copy.db*
```

Expected: junit `failures='0' errors='0'`. The sweep's only FAILs are the three known home-calibrated ones: NaN-bearing fail-point count, DRAWN marker count, and the limit-column scan row threshold. Any fourth FAIL is yours. If the full suite is too slow to finish, run at minimum `tests/test_trim_setup.py tests/test_trim_passes.py tests/test_trim_capture_db.py tests/test_trim_capture_noop.py tests/test_ft_reprocess_refresh.py tests/test_ft_graded_window.py tests/test_failed_file_markers.py tests/test_discover_paths.py` and say in the report that the full run was not completed.

- [ ] **Step 6: Commit and push**

```bash
git add tests/test_trim_capture_noop.py tests/fixtures/parse_baseline*.json scripts/app_qa_sweep.py
git commit -m "test: prove capture changed nothing, and guard it in the sweep

Every pre-existing stored field diffed against a baseline taken from the
tree before this work. The sweep check was made to fail first.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin V6 && git push origin V6:main
```

---

## Discovered while planning: the files are richer than the spec assumed

The spec described the per-pass table as holding the sweep plus "the recipe used for that pass" — one cut length per pass. Opening a System A pass sheet during planning showed it is 21 columns wide and records, **at every position**: the cut length applied, the trim current, the predicted delta, the delta actually used, the trim target and the final trim value.

That is a different order of information. One cut length per pass tells you how hard the machine cut; a cut length per position tells you *where* it cut and how much, alongside what it predicted the correction would be and what it actually got. Predicted versus used is the machine's own accuracy, recorded for free on every cut ever made.

This plan captures those columns (Tasks 3, 6 and 7) rather than deferring them, because they cost nothing extra to read now and re-running an overnight reprocess to collect them later would be expensive. Systems B and C do not record them, and the keys are absent rather than empty on those files so the difference stays visible.

The deferred cut-length model is correspondingly stronger than the spec's section on it assumed. Worth re-reading that section against this before it is picked up.

## After this plan

The next step is James's reprocess into a fresh database, per the spec's section 3. Only after that does the engine get planned, because the trajectory data has to exist before the analyzers that read it can be designed honestly.
