# Slope Optimization Fix — Theory-Based Rotation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the slope/angle optimization to match what the trim equipment actually does — rotate the error curve by adjusting the theoretical reference (like changing B8 in the spreadsheet), instead of incorrectly scaling error values.

**Architecture:** Read the theory_volts column from trim Excel files (already defined in column constants but never extracted), store it alongside position/error data, and use it in the optimizer with the correct formula: `adjusted_error = error + theory * k + offset` where `k` represents the B8 adjustment bounded by angle tolerance. Apply the same optimization to Final Test data when the model spec allows angle tolerance. The FT parser already reads theory_values — just needs DB storage and optimizer wiring.

**Tech Stack:** Python, SQLAlchemy 2.0 (SQLite), pandas, scipy.optimize, numpy

---

## Background

### What the equipment does
When adjusting B8 (travel length) in the trim sheet:
1. B8 value changes within the angle tolerance
2. Theory Volts column recalculates: `Theory = (pos - min_pos) / B8 * test_volts`
3. Errors recalculate: `Error = Actual - Theory`
4. Effect: the error curve rotates/swings
5. Spec limits stay fixed
6. Operator finds the B8 that makes the curve flattest within spec

### What the app currently does (WRONG)
```python
adjusted = error * slope + offset  # scales the error values
```
This is wrong because the correction should be proportional to the THEORY value at each position (which can be 0-10V), not to the ERROR value (which is ~0.001V). The difference is 1000x.

### What the app should do (CORRECT)
```python
adjusted = error + theory * k + offset
# where k = 1 - B8_old / B8_new, bounded by angle tolerance
```

### Data analysis results (230 DLTS + 31 LTS models)
- 248 models (95%) have LINEAR theory: `theory = (pos - min_pos) / B8 * test_volts`
- 13 models (5%) have NON-LINEAR theory (function pots with custom output curves)
- Theory column exists in every file (col F for System A, col C for System B)
- Theory column is defined in constants.py but never read by the parser

---

## Task 1: Read theory_volts column in parser

**Files:**
- Modify: `src/laser_trim_analyzer/core/parser.py:507-647` (_extract_track_data method)

- [ ] **Step 1: Extract theory_volts after errors**

In `_extract_track_data`, after extracting errors (line ~541) and before the length alignment block, add:

```python
# Theory volts column — needed for correct slope/angle optimization
# Theory represents the ideal output at each position; when B8 changes,
# theory recalculates and the error curve rotates
theory_volts = self._get_column_data(df, columns["theory_volts"], data_start, allow_nan=True)
```

- [ ] **Step 2: Align theory_volts to same length as positions**

After the existing position/error alignment block, add:

```python
# Align theory to match positions length
if len(theory_volts) > len(positions):
    theory_volts = theory_volts[:len(positions)]
elif len(theory_volts) < len(positions):
    # Pad with linear interpolation if shorter (shouldn't normally happen)
    if len(theory_volts) >= 2:
        step = theory_volts[-1] - theory_volts[-2] if len(theory_volts) >= 2 else 0
        while len(theory_volts) < len(positions):
            theory_volts.append(theory_volts[-1] + step)
    else:
        theory_volts = theory_volts + [0.0] * (len(positions) - len(theory_volts))
```

- [ ] **Step 3: Include theory_volts in the return dict**

Add `"theory_volts": theory_volts,` to the return dict (line ~632-647).

- [ ] **Step 4: Also read test_volts (reference voltage) for bounds calculation**

For System A, test_volts is in B13 (row 12, col 1). Read it:

```python
test_volts = self._get_cell_from_df(df, "B13") if system_type == SystemType.A else None
```

For System B, test_volts isn't in a fixed cell — derive it from the theory column:

```python
if test_volts is None and theory_volts:
    test_volts = max(theory_volts) if theory_volts else None
```

Add `"test_volts": test_volts,` to the return dict.

- [ ] **Step 5: Syntax check**

Run: `python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/parser.py').read()); print('OK')"`

- [ ] **Step 6: Run parser on sample files to verify theory_volts is read correctly**

```python
python3 -c "
from laser_trim_analyzer.core.parser import ExcelParser
from pathlib import Path
p = ExcelParser()

# Test DLTS (System A)
r = p.parse_file(Path('Work Files/Sample_Base_2026-04-10/DLTS/6828/6828_117_TEST DATA_3-13-2026_9-40 AM.xls'))
t = r['tracks'][0]
print(f'DLTS 6828: {len(t[\"theory_volts\"])} theory pts, first={t[\"theory_volts\"][0]:.4f}, last={t[\"theory_volts\"][-1]:.4f}, test_volts={t[\"test_volts\"]}')

# Test non-linear DLTS
r = p.parse_file(Path('Work Files/Sample_Base_2026-04-10/DLTS/8444/8444_67_TEST DATA_10-16-2023_3-51 PM.xls'))
t = r['tracks'][0]
print(f'DLTS 8444 (non-linear): {len(t[\"theory_volts\"])} theory pts, first={t[\"theory_volts\"][0]:.4f}, last={t[\"theory_volts\"][-1]:.4f}')

# Test LTS (System B)  
import os
lts_files = os.listdir('Work Files/Sample_Base_2026-04-10/LTS/8340-1')
r = p.parse_file(Path(f'Work Files/Sample_Base_2026-04-10/LTS/8340-1/{lts_files[0]}'))
t = r['tracks'][0]
print(f'LTS 8340-1: {len(t[\"theory_volts\"])} theory pts, first={t[\"theory_volts\"][0]:.4f}, last={t[\"theory_volts\"][-1]:.4f}')
print('All good!')
"
```

- [ ] **Step 7: Commit**

```bash
git add src/laser_trim_analyzer/core/parser.py
git commit -m "feat(parser): read theory_volts column for correct slope optimization"
```

---

## Task 2: Add theory_data column to database

**Files:**
- Modify: `src/laser_trim_analyzer/database/models.py` (TrackResult class)
- Modify: `src/laser_trim_analyzer/database/manager.py` (_run_migrations, _map_analysis_to_db, _map_db_to_analysis)

- [ ] **Step 1: Add theory_data and test_volts columns to TrackResult**

In `src/laser_trim_analyzer/database/models.py`, find the TrackResult class. Near the existing `position_data` and `error_data` SafeJSON columns (line ~355-362), add:

```python
theory_data = Column(SafeJSON, nullable=True)   # Array of theoretical output values
test_volts = Column(Float, nullable=True)        # Reference voltage (for slope bounds)
```

- [ ] **Step 2: Add migration to create columns**

In `src/laser_trim_analyzer/database/manager.py`, in `_run_migrations`, add:

```python
# Migration: Add theory_data and test_volts columns for slope optimization
try:
    session.execute(text("ALTER TABLE track_results ADD COLUMN theory_data TEXT"))
    session.commit()
    logger.info("Migration: Added theory_data column to track_results")
except Exception as e:
    if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
        logger.warning(f"theory_data migration warning: {e}")
    session.rollback()

try:
    session.execute(text("ALTER TABLE track_results ADD COLUMN test_volts FLOAT"))
    session.commit()
    logger.info("Migration: Added test_volts column to track_results")
except Exception as e:
    if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
        logger.warning(f"test_volts migration warning: {e}")
    session.rollback()
```

- [ ] **Step 3: Update _map_analysis_to_db to include theory_data and test_volts**

Find where `position_data` and `error_data` are mapped from the analysis result to the DB track. Add `theory_data` and `test_volts` alongside them. Search for where track data is mapped (look for `position_data=` assignment in the track mapping code).

- [ ] **Step 4: Update _map_db_to_analysis to read theory_data back**

Find where DB tracks are mapped back to analysis results. Add `theory_data` and `test_volts` to the mapping.

- [ ] **Step 5: Syntax check**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/models.py').read()); print('models OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('manager OK')"
```

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/database/models.py src/laser_trim_analyzer/database/manager.py
git commit -m "feat(db): add theory_data and test_volts columns for slope optimization"
```

---

## Task 3: Pass theory data through the processor to the analyzer

**Files:**
- Modify: `src/laser_trim_analyzer/core/processor.py` (where tracks are built from parsed data)
- Modify: `src/laser_trim_analyzer/core/models.py` (TrackData model, if theory_volts field needed)

- [ ] **Step 1: Find where parsed track data becomes TrackData**

In `processor.py`, find where the parser's track dict is converted to `TrackData` objects. The parser returns `{"theory_volts": [...], "test_volts": 10, ...}` and this needs to flow through to the analyzer.

- [ ] **Step 2: Add theory_volts and test_volts to TrackData model**

In `models.py`, find the `TrackData` class and add:

```python
theory_volts: Optional[List[float]] = Field(None, description="Theoretical output values")
test_volts: Optional[float] = Field(None, description="Reference voltage for slope bounds")
```

- [ ] **Step 3: Map parsed theory_volts into TrackData in processor**

Where the processor builds TrackData from parsed data, add:

```python
theory_volts=track_dict.get("theory_volts"),
test_volts=track_dict.get("test_volts"),
```

- [ ] **Step 4: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/processor.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/models.py').read()); print('OK')"
git add src/laser_trim_analyzer/core/processor.py src/laser_trim_analyzer/core/models.py
git commit -m "feat(processor): pass theory_volts through to analyzer"
```

---

## Task 4: Fix the optimizer to use theory-based rotation

**Files:**
- Modify: `src/laser_trim_analyzer/core/analyzer.py`

This is the core fix. The optimizer needs to change from `error * slope + offset` to `error + theory * k + offset`.

- [ ] **Step 1: Update _calculate_linearity to accept and use theory_volts**

Add `theory_volts: Optional[List[float]] = None` parameter. Pass it to `_calculate_optimal_adjustment`. Update the adjustment application:

Change from:
```python
shifted_errors = [e * optimal_slope + optimal_offset for e in errors]
```
To:
```python
if theory_volts and optimal_k != 0:
    shifted_errors = [e + theory_volts[i] * optimal_k + optimal_offset 
                      for i, e in enumerate(errors)]
else:
    shifted_errors = [e + optimal_offset for e in errors]
```

Note: the return tuple changes — `optimal_slope` becomes `optimal_k` (the theory scaling factor). `optimal_k = 0` means no rotation (equivalent to `slope = 1.0`).

- [ ] **Step 2: Update _calculate_optimal_adjustment**

Change the method to work with theory_volts and return `(offset, k)` instead of `(offset, slope)`:

When slope is locked (no angle tolerance): return `(offset, 0.0)` — k=0 means no rotation.

When slope has headroom: call a new `_optimize_offset_and_k` that optimizes:
```python
def objective(params):
    offset, k = params
    if k < k_lo or k > k_hi:
        return 1e12
    violations = 0
    max_err = 0.0
    for i in range(n):
        if exclude_indices and i in exclude_indices:
            continue
        adjusted = errors[i] + theory_volts[i] * k + offset
        ul = upper_limits[i]
        ll = lower_limits[i]
        if ul is not None and ll is not None:
            if not (np.isnan(ul) or np.isnan(ll)):
                if adjusted > ul or adjusted < ll:
                    violations += 1
                max_err = max(max_err, abs(adjusted))
    return violations * 1e6 + max_err
```

- [ ] **Step 3: Compute k bounds from angle tolerance**

Replace `_slope_bounds_from_angle_tol` with `_k_bounds_from_angle_tol`:

```python
def _k_bounds_from_angle_tol(self, angle_spec, angle_tol, angle_tol_type):
    """
    Compute bounds on k (theory scaling factor) from angle tolerance.
    
    k = 1 - B8_old / B8_new
    When B8 increases (longer travel), k > 0 (adds theory to error, rotates one way)
    When B8 decreases (shorter travel), k < 0 (subtracts theory, rotates other way)
    """
    if angle_spec is None or angle_spec == 0:
        return 0.0, 0.0  # No adjustment allowed
    
    tol_type = (angle_tol_type or "").strip().lower()
    ONE_SIDED_HEADROOM = 0.05
    
    if tol_type in ("symmetric", "range", "bilateral"):
        if angle_tol is None or angle_tol <= 0:
            return 0.0, 0.0
        # k = 1 - A/(A+T) to 1 - A/(A-T)
        k_lo = 1 - angle_spec / (angle_spec - angle_tol)  # negative
        k_hi = 1 - angle_spec / (angle_spec + angle_tol)  # positive
        return k_lo, k_hi
    
    if tol_type == "min":
        # Part can be longer -> B8_new > B8_old -> k > 0 only
        return 0.0, ONE_SIDED_HEADROOM
    
    if tol_type == "max":
        # Part can be shorter -> B8_new < B8_old -> k < 0 only
        return -ONE_SIDED_HEADROOM, 0.0
    
    return 0.0, 0.0  # No tolerance type -> no adjustment
```

- [ ] **Step 4: Update the offset-only optimizer**

`_calculate_optimal_offset` stays the same — it doesn't involve theory. No changes needed.

- [ ] **Step 5: Update all callers of _calculate_linearity**

Search for all calls to `_calculate_linearity` and pass `theory_volts`. Also update how `optimal_slope` is used downstream — it's now `optimal_k`.

The TrackData result stores `optimal_offset` and potentially a new field for the k factor. Check where `optimal_slope` is referenced in the codebase and update accordingly.

- [ ] **Step 6: Update the chart display**

In `analyze.py`, `_display_track_chart` currently applies `shifted_errors = [e * slope + offset for e in errors]`. This needs to use theory_data:

```python
if track.theory_data and hasattr(track, 'optimal_k') and track.optimal_k:
    shifted_errors = [e + track.theory_data[i] * track.optimal_k + track.optimal_offset 
                      for i, e in enumerate(track.error_data)]
else:
    shifted_errors = [e + track.optimal_offset for e in track.error_data]
```

- [ ] **Step 7: Syntax check**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/analyzer.py').read()); print('OK')"
```

- [ ] **Step 8: Test with sample files**

Process a few sample files and verify:
1. Linear model (6828): optimization finds similar offset, k ≈ 0
2. Non-linear model (8444): optimization uses theory-based rotation
3. Model with no angle tolerance: k stays at 0, only offset used
4. Compare pass/fail results with the old approach

```python
python3 -c "
from laser_trim_analyzer.core.processor import Processor
from laser_trim_analyzer.config import get_config
from pathlib import Path

config = get_config()
proc = Processor(config)

# Test files
test_files = [
    'Work Files/Sample_Base_2026-04-10/DLTS/6828/6828_117_TEST DATA_3-13-2026_9-40 AM.xls',
    'Work Files/Sample_Base_2026-04-10/DLTS/8444/8444_67_TEST DATA_10-16-2023_3-51 PM.xls',
]
for f in test_files:
    result = proc.process_file(Path(f))
    for t in result.tracks:
        print(f'{f.split(\"/\")[-1]}: offset={t.optimal_offset:.6f}, k={getattr(t, \"optimal_k\", \"N/A\")}, fail_pts={t.linearity_fail_points}')
"
```

- [ ] **Step 9: Commit**

```bash
git add src/laser_trim_analyzer/core/analyzer.py
git commit -m "fix(analyzer): use theory-based rotation instead of error scaling for slope optimization"
```

---

## Task 5: Database field mapping and backwards compatibility

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py` (mapping functions)
- Modify: `src/laser_trim_analyzer/database/models.py` (TrackResult — add optimal_k column)

- [ ] **Step 1: Add optimal_k column to TrackResult**

In `models.py`, add to TrackResult near the existing `optimal_offset`:

```python
optimal_k = Column(Float, default=0.0)  # Theory rotation factor (replaces slope)
```

- [ ] **Step 2: Add migration for optimal_k**

In `_run_migrations`:

```python
try:
    session.execute(text("ALTER TABLE track_results ADD COLUMN optimal_k FLOAT DEFAULT 0.0"))
    session.commit()
except Exception as e:
    if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
        logger.warning(f"optimal_k migration warning: {e}")
    session.rollback()
```

- [ ] **Step 3: Update the DB mapping to save/load theory_data, test_volts, optimal_k**

Find the mapping functions and ensure theory_data, test_volts, and optimal_k are saved and loaded.

- [ ] **Step 4: Handle backwards compatibility**

Existing records have `optimal_slope` but no `optimal_k` or `theory_data`. When loading old records:
- If `theory_data` is None, the chart should fall back to `error + offset` only (no rotation)
- If `optimal_k` is None, treat as 0 (no rotation)

- [ ] **Step 5: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/models.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
git add src/laser_trim_analyzer/database/models.py src/laser_trim_analyzer/database/manager.py
git commit -m "feat(db): add optimal_k and theory_data fields with backwards compatibility"
```

---

## Task 6: Apply slope optimization to Final Test data

**Why:** The same angle tolerance applies to Final Test data. The FT parser already reads `theory_values` but doesn't store them or run the optimizer.

**Files:**
- Modify: `src/laser_trim_analyzer/database/models.py` (FinalTestTrack — add theory_data column)
- Modify: `src/laser_trim_analyzer/database/manager.py` (FT migration + mapping)
- Modify: `src/laser_trim_analyzer/core/processor.py` (apply optimizer to FT tracks)

- [ ] **Step 1: Add theory_data and optimal_k to FinalTestTrack model**

In `models.py`, find the `FinalTestTrack` class and add:

```python
theory_data = Column(SafeJSON, nullable=True)   # Array of theoretical output values
optimal_k = Column(Float, default=0.0)          # Theory rotation factor
```

- [ ] **Step 2: Add migration for FT columns**

In `_run_migrations`:

```python
try:
    session.execute(text("ALTER TABLE final_test_tracks ADD COLUMN theory_data TEXT"))
    session.commit()
except Exception as e:
    if "duplicate column" not in str(e).lower():
        logger.warning(f"ft theory_data migration: {e}")
    session.rollback()

try:
    session.execute(text("ALTER TABLE final_test_tracks ADD COLUMN optimal_k FLOAT DEFAULT 0.0"))
    session.commit()
except Exception as e:
    if "duplicate column" not in str(e).lower():
        logger.warning(f"ft optimal_k migration: {e}")
    session.rollback()
```

- [ ] **Step 3: Store theory_values from FT parser**

In the processor's Final Test handling, the parser already returns `theory_values` in the track dict. Map it to `theory_data` when saving FinalTestTrack to the database.

- [ ] **Step 4: Apply optimizer to FT tracks when angle tolerance exists**

In the processor's FT processing flow, after parsing, look up the model spec for angle tolerance. If tolerance exists, run the same `_calculate_optimal_adjustment` (with theory-based rotation) on the FT errors. Store the resulting `optimal_k` and adjusted pass/fail results.

- [ ] **Step 5: Update Compare page chart to use theory-based adjustment**

In `compare.py`, when displaying FT data overlaid on trim data, apply the same theory-based correction if `optimal_k` is stored.

- [ ] **Step 6: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/models.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/processor.py').read()); print('OK')"
git add src/laser_trim_analyzer/database/models.py src/laser_trim_analyzer/database/manager.py src/laser_trim_analyzer/core/processor.py
git commit -m "feat(ft): apply theory-based slope optimization to Final Test data"
```

---

## Task 7: Full integration test with sample files

**Files:**
- No code changes — verification only

- [ ] **Step 1: Process a batch of DLTS files and verify results**

Process files from several models (linear and non-linear) and verify:
- theory_data is populated in the database
- optimal_k is computed correctly
- Pass/fail results are reasonable
- Charts display correctly

- [ ] **Step 2: Process a batch of LTS files and verify**

Same checks for System B files.

- [ ] **Step 3: Verify existing data backwards compatibility**

Load an old record (without theory_data) and verify:
- Chart displays correctly (falls back to offset-only)
- No crashes or errors
- Analyze page works

- [ ] **Step 4: Compare results before/after**

For a few key models, compare:
- Old approach: error * slope + offset
- New approach: error + theory * k + offset
- Are there cases where pass/fail changes? Document them.

- [ ] **Step 5: Commit any final fixes**

```bash
git add -A
git commit -m "test: verify slope optimization fix across all sample files"
```
