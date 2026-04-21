# Point Exclusion Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users specify per-model data point indices to exclude from linearity fail-point counting, so known mechanical artifacts (endpoint settling) don't cause false failures.

**Architecture:** Add `exclude_points` TEXT column to `model_specs`. Parse a human-friendly string ("0-2, 48-50") into a JSON blob on save. The Analyzer's `_count_fail_points` and `_calculate_failure_margins` skip excluded indices. The Analyze page chart renders excluded fail points as gray markers.

**Tech Stack:** SQLAlchemy (migration), JSON storage, customtkinter (Specs edit panel), matplotlib (gray markers on Analyze chart)

---

### File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/laser_trim_analyzer/database/models.py` | Modify | Add `exclude_points` column to ModelSpec |
| `src/laser_trim_analyzer/database/manager.py` | Modify | Migration for new column |
| `src/laser_trim_analyzer/core/analyzer.py` | Modify | Skip excluded indices in `_count_fail_points`, `_calculate_linearity`, `_calculate_failure_margins` |
| `src/laser_trim_analyzer/core/processor.py` | Modify | Pass `exclude_indices` from spec to analyzer |
| `src/laser_trim_analyzer/gui/pages/specs.py` | Modify | Add Exclude Points field to edit panel |
| `src/laser_trim_analyzer/gui/pages/analyze.py` | Modify | Render excluded points as gray markers, show exclusion count |
| `src/laser_trim_analyzer/gui/widgets/chart.py` | Modify | Accept `excluded_points` param in `plot_linearity` |

---

### Task 1: Add `exclude_points` column to ModelSpec + migration

**Files:**
- Modify: `src/laser_trim_analyzer/database/models.py:1137` (after `aliases`)
- Modify: `src/laser_trim_analyzer/database/manager.py` (migration section)

- [ ] **Step 1: Add column to ORM model**

In `src/laser_trim_analyzer/database/models.py`, add after the `aliases` line (line 1137):

```python
    # JSON list of point indices/ranges to exclude from linearity fail counting.
    # Format: {"exclude": [0, 1, [48, 50]]}  — ints and [start, end] ranges.
    # Set via Specs page as human-friendly "0-1, 48-50" notation.
    exclude_points = Column(Text, nullable=True)
```

- [ ] **Step 2: Add migration in manager.py**

In `src/laser_trim_analyzer/database/manager.py`, in the `_run_migrations` method, add after the `aliases` migration block (around line 561):

```python
            # Migration: Add exclude_points column to model_specs
            try:
                session.execute(text("SELECT exclude_points FROM model_specs LIMIT 1"))
            except OperationalError:
                try:
                    session.execute(text("ALTER TABLE model_specs ADD COLUMN exclude_points TEXT"))
                    session.commit()
                    logger.info("Migration: Added exclude_points column to model_specs")
                except Exception as e:
                    if "duplicate column" not in str(e).lower():
                        logger.warning(f"exclude_points migration warning: {e}")
                    session.rollback()
```

- [ ] **Step 3: Verify migration runs**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.database.manager import DatabaseManager
db = DatabaseManager('data/analysis.db')
specs = db.get_all_model_specs()
print('exclude_points' in specs[0] if specs else 'No specs')
db.close()
"
```
Expected: `True` (column exists and is returned in spec dicts)

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/database/models.py src/laser_trim_analyzer/database/manager.py
git commit -m "feat(db): add exclude_points column to model_specs"
```

---

### Task 2: Add exclusion parsing helpers to Analyzer

**Files:**
- Modify: `src/laser_trim_analyzer/core/analyzer.py` (top of file, near imports)

- [ ] **Step 1: Add parse helper function**

Add after the existing imports in `src/laser_trim_analyzer/core/analyzer.py` (around line 20, before the class definition):

```python
def parse_exclude_points(raw: Optional[str]) -> Set[int]:
    """Parse exclude_points JSON string into a set of integer indices.
    
    Accepts JSON like: {"exclude": [0, 1, [48, 50]]}
    Individual ints become single indices. [start, end] ranges are inclusive.
    Returns empty set for None/empty/invalid input.
    """
    if not raw:
        return set()
    try:
        import json
        data = json.loads(raw)
        items = data.get("exclude", []) if isinstance(data, dict) else []
        indices = set()
        for item in items:
            if isinstance(item, int):
                indices.add(item)
            elif isinstance(item, list) and len(item) == 2:
                start, end = int(item[0]), int(item[1])
                indices.update(range(start, end + 1))
        return indices
    except (json.JSONDecodeError, TypeError, ValueError):
        return set()


def format_exclude_points(indices: Set[int]) -> str:
    """Convert a set of indices back to human-friendly string like '0-2, 48-50'.
    
    Groups consecutive indices into ranges.
    """
    if not indices:
        return ""
    sorted_idx = sorted(indices)
    ranges = []
    start = prev = sorted_idx[0]
    for idx in sorted_idx[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = idx
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def human_to_exclude_json(text: str) -> Optional[str]:
    """Parse human-friendly '0-2, 48-50' into JSON storage format.
    
    Returns JSON string or None if input is empty.
    """
    if not text or not text.strip():
        return None
    import json
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            pieces = part.split("-", 1)
            try:
                start, end = int(pieces[0].strip()), int(pieces[1].strip())
                items.append([start, end])
            except ValueError:
                continue
        else:
            try:
                items.append(int(part))
            except ValueError:
                continue
    if not items:
        return None
    return json.dumps({"exclude": items})
```

Also add `Set` to the typing imports at the top of the file if not present:

```python
from typing import Dict, List, Optional, Any, Tuple, Set
```

- [ ] **Step 2: Verify parsing works**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.core.analyzer import parse_exclude_points, format_exclude_points, human_to_exclude_json

# Parse JSON -> set
assert parse_exclude_points('{\"exclude\": [0, 1, [48, 50]]}') == {0, 1, 48, 49, 50}
assert parse_exclude_points(None) == set()
assert parse_exclude_points('') == set()
assert parse_exclude_points('garbage') == set()

# Set -> human string
assert format_exclude_points({0, 1, 2, 48, 49, 50}) == '0-2, 48-50'
assert format_exclude_points({5}) == '5'
assert format_exclude_points(set()) == ''

# Human string -> JSON
j = human_to_exclude_json('0-2, 48-50')
assert parse_exclude_points(j) == {0, 1, 2, 48, 49, 50}
assert human_to_exclude_json('') is None

print('All parsing tests passed')
"
```
Expected: `All parsing tests passed`

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/core/analyzer.py
git commit -m "feat(analyzer): add exclude_points parsing helpers"
```

---

### Task 3: Wire exclusion into Analyzer methods

**Files:**
- Modify: `src/laser_trim_analyzer/core/analyzer.py`

- [ ] **Step 1: Update `_count_fail_points` to accept `exclude_indices`**

Change the method signature and loop (around line 678):

```python
    def _count_fail_points(
        self,
        errors: List[float],
        upper_limits: List[float],
        lower_limits: List[float],
        exclude_indices: Optional[Set[int]] = None,
    ) -> int:
        """Count points outside specification limits, skipping excluded indices."""
        if not upper_limits or not lower_limits:
            return 0

        n = min(len(errors), len(upper_limits), len(lower_limits))
        count = 0

        for i in range(n):
            if exclude_indices and i in exclude_indices:
                continue
            if upper_limits[i] is not None and lower_limits[i] is not None:
                if not (np.isnan(upper_limits[i]) or np.isnan(lower_limits[i])):
                    if errors[i] > upper_limits[i] or errors[i] < lower_limits[i]:
                        count += 1

        return count
```

- [ ] **Step 2: Update `_calculate_linearity` to pass `exclude_indices` through**

Change signature (around line 331) — add `exclude_indices` param:

```python
    def _calculate_linearity(
        self,
        positions: List[float],
        errors: List[float],
        upper_limits: List[float],
        lower_limits: List[float],
        linearity_spec: float,
        linearity_type: Optional[str] = None,
        angle_spec: Optional[float] = None,
        angle_tol: Optional[float] = None,
        angle_tol_type: Optional[str] = None,
        exclude_indices: Optional[Set[int]] = None,
    ) -> Tuple[float, float, float, bool, int, float, int]:
```

Then update the two `_count_fail_points` calls inside this method to pass `exclude_indices`:

```python
        raw_fail_points = self._count_fail_points(errors, upper_limits, lower_limits, exclude_indices)
```

```python
        fail_points = self._count_fail_points(shifted_errors, upper_limits, lower_limits, exclude_indices)
```

- [ ] **Step 3: Update `_calculate_failure_margins` to skip excluded indices**

Change signature (around line 699) — add `exclude_indices` param:

```python
    def _calculate_failure_margins(
        self,
        shifted_errors: List[float],
        upper_limits: List[float],
        lower_limits: List[float],
        exclude_indices: Optional[Set[int]] = None,
    ) -> Dict[str, Optional[float]]:
```

Add skip logic at the start of the loop body (inside `for i in range(n):`), right after `ll = lower_limits[i]`:

```python
            if exclude_indices and i in exclude_indices:
                continue
```

- [ ] **Step 4: Update `analyze_track` to parse and pass exclusions**

In `analyze_track` (around line 79), after reading `measured_electrical_angle`, add:

```python
        exclude_indices = parse_exclude_points(track_data.get("exclude_points"))
```

Then pass it to `_calculate_linearity` (around line 108):

```python
        (optimal_offset, optimal_slope, linearity_error, linearity_pass,
         fail_points, raw_linearity_error, raw_fail_points) = self._calculate_linearity(
            positions, errors, upper_limits, lower_limits, linearity_spec,
            linearity_type=linearity_type,
            angle_spec=angle_spec,
            angle_tol=angle_tol,
            angle_tol_type=angle_tol_type,
            exclude_indices=exclude_indices,
        )
```

And pass it to `_calculate_failure_margins` (around line 144):

```python
        margin_metrics = self._calculate_failure_margins(
            shifted_errors, upper_limits, lower_limits, exclude_indices
        )
```

- [ ] **Step 5: Verify analyzer compiles and processes files**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.core.processor import Processor
from laser_trim_analyzer.config import get_config
import glob
processor = Processor(get_config())
files = glob.glob('Work Files/Sample_Base_2026-04-10/DLTS/**/*.xls', recursive=True)
result = processor.process_file(files[0])
print(f'OK: {result.metadata.model} status={result.overall_status}')
"
```
Expected: `OK: <model> status=<status>` — no crash.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/core/analyzer.py
git commit -m "feat(analyzer): skip excluded point indices in linearity analysis"
```

---

### Task 4: Pass exclusion spec from Processor to Analyzer

**Files:**
- Modify: `src/laser_trim_analyzer/core/processor.py`

- [ ] **Step 1: Add `exclude_points` to `_get_spec_for_analysis` return dict**

In `_get_spec_for_analysis` (around line 930), change the return dict:

```python
            return {
                "linearity_type": spec.get("linearity_type"),
                "angle_spec": spec.get("electrical_angle"),
                "angle_tol": spec.get("electrical_angle_tol"),
                "angle_tol_type": spec.get("electrical_angle_tol_type"),
                "exclude_points": spec.get("exclude_points"),
            }
```

Also add `"exclude_points": None` to the `empty` dict at the start of the method:

```python
        empty = {
            "linearity_type": None,
            "angle_spec": None,
            "angle_tol": None,
            "angle_tol_type": None,
            "exclude_points": None,
        }
```

- [ ] **Step 2: Pass `exclude_points` into track_data before calling analyzer**

In the trim file processing path (around line 188), where `analyzer.analyze_track` is called, the track_data dict comes from the parser. We need to inject `exclude_points` into it before calling the analyzer. Add this line before the `analyzer.analyze_track` call:

```python
                    track_data["exclude_points"] = spec["exclude_points"]
```

Find the block around line 186-194 that looks like:

```python
                result = self.analyzer.analyze_track(
                    track_data,
                    model=metadata.model,
                    linearity_type=linearity_type,
```

Add the injection line just before this call.

Do the same for the Final Test processing path (around line 370) — add `track_dict["exclude_points"] = ft_spec["exclude_points"]` before the analyzer call there.

- [ ] **Step 3: Verify processing still works**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.core.processor import Processor
from laser_trim_analyzer.config import get_config
import glob
processor = Processor(get_config())
files = glob.glob('Work Files/Sample_Base_2026-04-10/DLTS/**/*.xls', recursive=True)
for f in files[:5]:
    result = processor.process_file(f)
    if result:
        print(f'OK: {result.metadata.model} status={result.overall_status} tracks={len(result.tracks)}')
"
```
Expected: 5 files process successfully.

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/core/processor.py
git commit -m "feat(processor): pass exclude_points from model spec to analyzer"
```

---

### Task 5: Add Exclude Points field to Specs page

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/specs.py`

- [ ] **Step 1: Add field to edit panel**

In the `fields` list inside `_create_edit_panel` (around line 179), add before the `("notes", ...)` entry:

```python
            ("exclude_points", "Exclude Points", "entry"),
```

- [ ] **Step 2: Add placeholder text for the exclude_points entry**

After the edit fields loop (after line 207), add:

```python
        # Set placeholder for exclude_points field
        if "exclude_points" in self._edit_fields:
            _, ep_widget = self._edit_fields["exclude_points"]
            ep_widget.configure(placeholder_text="e.g. 0-2, 48-50")
```

- [ ] **Step 3: Handle JSON↔human conversion on load**

In `_on_select` (around line 368), after the general field population loop, add special handling for `exclude_points`:

```python
        # Special handling: convert exclude_points JSON to human-friendly display
        if "exclude_points" in self._edit_fields:
            _, ep_widget = self._edit_fields["exclude_points"]
            ep_widget.delete(0, "end")
            raw_json = spec.get("exclude_points", "")
            if raw_json:
                from laser_trim_analyzer.core.analyzer import parse_exclude_points, format_exclude_points
                indices = parse_exclude_points(raw_json)
                ep_widget.insert(0, format_exclude_points(indices))
```

- [ ] **Step 4: Handle JSON conversion on save**

In `_get_edit_values` (around line 448), after the general value collection loop, add conversion for exclude_points:

```python
        # Special handling: convert human-friendly exclude_points to JSON
        if "exclude_points" in values and values["exclude_points"]:
            from laser_trim_analyzer.core.analyzer import human_to_exclude_json
            values["exclude_points"] = human_to_exclude_json(values["exclude_points"])
        elif "exclude_points" in values:
            values["exclude_points"] = None
```

- [ ] **Step 5: Verify Specs page compiles**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
import laser_trim_analyzer.gui.pages.specs
print('OK')
"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/specs.py
git commit -m "feat(specs): add Exclude Points field with human-friendly notation"
```

---

### Task 6: Render excluded points as gray markers on Analyze chart

**Files:**
- Modify: `src/laser_trim_analyzer/gui/widgets/chart.py`
- Modify: `src/laser_trim_analyzer/gui/pages/analyze.py`

- [ ] **Step 1: Add `excluded_points` parameter to `plot_linearity` in chart.py**

In `src/laser_trim_analyzer/gui/widgets/chart.py`, update the `plot_linearity` method signature (around line 215) to add the parameter:

```python
        excluded_points: Optional[List[int]] = None,
```

Then after the existing fail_points scatter block (after line 375), add:

```python
        if excluded_points:
            excl_x = [positions[i] for i in excluded_points if i < len(positions)]
            excl_y = [shifted_errors[i] for i in excluded_points if i < len(shifted_errors)]
            ax.scatter(
                excl_x, excl_y,
                color='gray',
                marker='o',
                s=40,
                facecolors='none',
                linewidths=1.5,
                label='Excluded Points',
                zorder=4,
                alpha=0.6,
            )
```

- [ ] **Step 2: Look up model exclusions in Analyze page and pass to chart**

In `src/laser_trim_analyzer/gui/pages/analyze.py`, in the method that builds fail_indices and calls the chart (around line 910), add exclusion lookup after the fail_indices loop:

After computing `fail_indices` (around line 921), add:

```python
        # Look up excluded points for this model
        excluded_indices = set()
        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.core.analyzer import parse_exclude_points
            db = get_database()
            spec = db.get_model_spec(self.current_result.metadata.model)
            if spec:
                excluded_indices = parse_exclude_points(spec.get("exclude_points"))
        except Exception:
            pass

        # Separate excluded fail points from real fail points
        excluded_fail_indices = [i for i in fail_indices if i in excluded_indices]
        real_fail_indices = [i for i in fail_indices if i not in excluded_indices]
```

Then update the `plot_linearity` call (around line 966) to use `real_fail_indices` instead of `fail_indices`, and pass `excluded_fail_indices`:

Replace `fail_points=fail_indices` with:
```python
            fail_points=real_fail_indices,
            excluded_points=excluded_fail_indices,
```

- [ ] **Step 3: Add exclusion note to metrics panel**

In the metrics display method on the Analyze page, after the existing metrics text, add:

```python
        if excluded_indices:
            lines.append(f"  Points excluded: {len(excluded_indices)} (per model spec)")
```

Find where metrics lines are built (the method that populates track info) and add this after the fail_points line.

- [ ] **Step 4: Verify both files compile**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
import laser_trim_analyzer.gui.widgets.chart
import laser_trim_analyzer.gui.pages.analyze
print('OK')
"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/widgets/chart.py src/laser_trim_analyzer/gui/pages/analyze.py
git commit -m "feat(analyze): render excluded points as gray markers on chart"
```

---

### Task 7: End-to-end verification

- [ ] **Step 1: Set an exclusion rule and verify analysis changes**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.core.analyzer import Analyzer, parse_exclude_points, human_to_exclude_json
from laser_trim_analyzer.core.processor import Processor
from laser_trim_analyzer.config import get_config
import glob

# Process a file without exclusions
processor = Processor(get_config())
files = glob.glob('Work Files/Sample_Base_2026-04-10/DLTS/**/*.xls', recursive=True)
result = processor.process_file(files[0])
if result:
    track = result.tracks[0]
    print(f'Without exclusions: fail_points={track.linearity_fail_points}, pass={track.linearity_pass}')

    # Now test the analyzer directly with exclusions
    analyzer = Analyzer()
    # Re-parse the file to get raw track_data
    from laser_trim_analyzer.core.parser import ExcelParser
    parser = ExcelParser()
    parsed = parser.parse_file(files[0])
    track_data = parsed['tracks'][0]
    
    # Add exclusion for first and last point
    n_points = len(track_data['positions'])
    track_data['exclude_points'] = human_to_exclude_json(f'0, {n_points - 1}')
    
    result2 = analyzer.analyze_track(track_data, model=parsed['metadata'].model)
    print(f'With exclusions (first+last): fail_points={result2.linearity_fail_points}, pass={result2.linearity_pass}')
    print(f'Exclusion applied: {result2.linearity_fail_points <= track.linearity_fail_points}')
"
```
Expected: The excluded version should have equal or fewer fail points.

- [ ] **Step 2: Verify round-trip parsing**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.core.analyzer import parse_exclude_points, format_exclude_points, human_to_exclude_json

# Full round trip: human -> JSON -> set -> human
original = '0-2, 25, 48-50'
json_str = human_to_exclude_json(original)
indices = parse_exclude_points(json_str)
back_to_human = format_exclude_points(indices)
print(f'Input:  {original}')
print(f'JSON:   {json_str}')
print(f'Set:    {sorted(indices)}')
print(f'Output: {back_to_human}')
assert indices == {0, 1, 2, 25, 48, 49, 50}
print('Round-trip OK')
"
```
Expected: `Round-trip OK`

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: point exclusion rules — per-model ignore list for linearity analysis

Adds ability to exclude specific data point indices from linearity
fail-point counting on a per-model basis. Excluded points still render
on charts (as gray hollow markers) but don't affect pass/fail.

- model_specs.exclude_points column (JSON storage)
- Human-friendly notation on Specs page: '0-2, 48-50'
- Analyzer skips excluded indices in _count_fail_points and margins
- Analyze page shows gray markers for excluded fail points"
```
