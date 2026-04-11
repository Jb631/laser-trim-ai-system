# Phase 2: Spec-Aware Analysis Engine --- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the analyzer from a single offset-only optimization to a full linearity-type-aware adjustment engine that supports offset + slope, parses station compensation values from file cells, stores both raw and optimized results, and applies to both trim and Final Test analysis.

**Prerequisite:** Phase 1 (Model Reference System) must be complete. The `model_specs` table with `linearity_type` field must exist and be populated.

**Architecture:**
- `_calculate_optimal_offset` becomes `_calculate_optimal_adjustment` returning `(offset, slope)` with linearity-type-aware constraints
- Compensation values parsed from three sources: FT Sheet1 cell M4, LTS "Lin Chart" cell O1, DLTS trim sheet cell B7
- New `compensation` field on `TrackData` model and `TrackResult` DB column
- New `optimized_offset` + `optimized_slope` + `raw_linearity_error` + `optimized_linearity_error` fields stored on TrackResult
- Analyzer receives `linearity_type` from processor (looked up from model_specs)
- UI shows station vs. optimal comparison when both are available

**Linearity Type Behaviors:**
| Type | Offset | Slope | Description |
|------|--------|-------|-------------|
| Absolute | Fixed (0) | Fixed (1) | No adjustment allowed --- error band is absolute |
| Independent | Free | Free | Full offset + slope optimization (most common) |
| Term Base | Fixed (0) | Fixed (1) | Same as Absolute --- terminals define the reference |
| Zero-Based | Fixed (0) | Free | Slope optimization only, offset locked at zero |

**Tech Stack:** SQLAlchemy 2.0, scipy.optimize, numpy, existing parser/analyzer patterns

**Dev Environment Notes:**
- No pytest installed --- use `python3 -c "import ast; ast.parse(open('file').read())"` for syntax checks
- No pydantic --- use dataclasses or plain dicts (note: core/models.py does use pydantic, follow existing patterns)
- Runtime: `python3` (not `python`)
- Working directory: `/Users/jb631/projects/laser-trim-ai-system-v5/`
- Branch: `v5-upgrade`

---

### Task 1: Add Compensation and Optimization Fields to Data Models

**Files:**
- Modify: `src/laser_trim_analyzer/core/models.py`
- Modify: `src/laser_trim_analyzer/database/models.py`

- [ ] **Step 1: Add new fields to TrackData (pydantic model)**

In `src/laser_trim_analyzer/core/models.py`, add the following fields to the `TrackData` class, after the existing `optimal_offset` field (around line 104):

```python
    # Spec-aware optimization results (Phase 2)
    optimal_slope: float = Field(default=1.0, description="Optimal slope adjustment (1.0 = no change)")
    station_compensation: Optional[float] = Field(None, description="Compensation value from station file (offset applied by machine)")
    linearity_type: Optional[str] = Field(None, description="Linearity type from model specs (Absolute, Independent, Term Base, Zero-Based)")
    raw_linearity_error: Optional[float] = Field(None, ge=0, description="Max error before any optimization")
    optimized_linearity_error: Optional[float] = Field(None, ge=0, description="Max error after optimal offset+slope adjustment")
    raw_fail_points: Optional[int] = Field(None, ge=0, description="Fail points before optimization")
```

- [ ] **Step 2: Add new columns to TrackResult (SQLAlchemy model)**

In `src/laser_trim_analyzer/database/models.py`, add columns to the `TrackResult` class, after `optimal_offset` (around line 328):

```python
    # Spec-aware optimization (Phase 2)
    optimal_slope = Column(Float, default=1.0)
    station_compensation = Column(Float)
    linearity_type = Column(String(30))
    raw_linearity_error = Column(Float)
    optimized_linearity_error = Column(Float)
    raw_fail_points = Column(Integer)
```

- [ ] **Step 3: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/models.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/models.py').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/core/models.py src/laser_trim_analyzer/database/models.py
git commit -m "feat: add compensation and spec-aware optimization fields to data models"
```

---

### Task 2: Upgrade Analyzer with Linearity-Type-Aware Optimization

**Files:**
- Modify: `src/laser_trim_analyzer/core/analyzer.py`

This is the core of Phase 2. The existing `_calculate_optimal_offset` (lines 339-394) is replaced by `_calculate_optimal_adjustment` which returns both offset and slope, constrained by linearity type.

- [ ] **Step 1: Add `_calculate_optimal_adjustment` method**

Replace `_calculate_optimal_offset` with a new method. Keep the old method as a private fallback alias. Add this after the existing `_calculate_linearity` method:

```python
    def _calculate_optimal_adjustment(
        self,
        positions: List[float],
        errors: List[float],
        upper_limits: List[float],
        lower_limits: List[float],
        linearity_type: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Calculate optimal offset and slope adjustment, constrained by linearity type.

        Linearity types control which degrees of freedom are available:
        - Absolute / Term Base: No adjustment (offset=0, slope=1.0)
        - Independent: Free offset + slope optimization (most common)
        - Zero-Based: Slope optimization only (offset=0)
        - None/unknown: Falls back to offset-only (legacy behavior)

        The slope adjustment rescales errors: adjusted_error[i] = error[i] * slope + offset
        This models the real-world compensation the test station can apply.

        Args:
            positions: Position values (X-axis)
            errors: Error values at each position
            upper_limits: Upper spec limits
            lower_limits: Lower spec limits
            linearity_type: From model_specs table

        Returns:
            (optimal_offset, optimal_slope) tuple
        """
        lin_type = (linearity_type or "").strip().lower()

        # Absolute and Term Base: no adjustment allowed
        if lin_type in ("absolute", "term base"):
            return 0.0, 1.0

        # Zero-Based: slope only (offset locked at 0)
        if lin_type == "zero-based":
            slope = self._optimize_slope_only(positions, errors, upper_limits, lower_limits)
            return 0.0, slope

        # Independent: full offset + slope optimization
        if lin_type == "independent":
            return self._optimize_offset_and_slope(positions, errors, upper_limits, lower_limits)

        # Unknown/None: legacy offset-only behavior for backwards compatibility
        offset = self._calculate_optimal_offset(errors, upper_limits, lower_limits)
        return offset, 1.0

    def _optimize_slope_only(
        self,
        positions: List[float],
        errors: List[float],
        upper_limits: List[float],
        lower_limits: List[float],
    ) -> float:
        """
        Optimize slope with offset fixed at zero.

        Adjusted error = error * slope
        Minimize fail points, then minimize max error as tiebreaker.
        """
        n = min(len(errors), len(upper_limits), len(lower_limits))
        if n == 0:
            return 1.0

        def objective(slope_val: float) -> float:
            """Combined objective: violation count * 1e6 + max_abs_error."""
            s = slope_val
            violations = 0
            max_err = 0.0
            for i in range(n):
                adjusted = errors[i] * s
                ul = upper_limits[i]
                ll = lower_limits[i]
                if ul is not None and ll is not None:
                    if not (np.isnan(ul) or np.isnan(ll)):
                        if adjusted > ul or adjusted < ll:
                            violations += 1
                        max_err = max(max_err, abs(adjusted))
            return violations * 1e6 + max_err

        try:
            # Search around slope=1.0 (no change)
            result = optimize.minimize_scalar(
                objective,
                bounds=(0.8, 1.2),
                method='bounded',
                options={'xatol': 1e-6}
            )
            return float(result.x)
        except Exception:
            return 1.0

    def _optimize_offset_and_slope(
        self,
        positions: List[float],
        errors: List[float],
        upper_limits: List[float],
        lower_limits: List[float],
    ) -> Tuple[float, float]:
        """
        Optimize both offset and slope for Independent linearity.

        Adjusted error = error * slope + offset
        Uses two-stage optimization:
        1. Grid search for rough (offset, slope) candidates
        2. Nelder-Mead refinement of best candidate

        Returns:
            (offset, slope) tuple
        """
        n = min(len(errors), len(upper_limits), len(lower_limits))
        if n == 0:
            return 0.0, 1.0

        # Calculate band center differences for initial offset guess
        differences = []
        for i in range(n):
            ul = upper_limits[i]
            ll = lower_limits[i]
            if ul is not None and ll is not None:
                if not (np.isnan(ul) or np.isnan(ll)):
                    midpoint = (ul + ll) / 2
                    differences.append(midpoint - errors[i])
        initial_offset = float(np.median(differences)) if differences else 0.0

        def objective(params):
            """Combined objective: violation count * 1e6 + max_abs_error."""
            offset, slope = params
            violations = 0
            max_err = 0.0
            for i in range(n):
                adjusted = errors[i] * slope + offset
                ul = upper_limits[i]
                ll = lower_limits[i]
                if ul is not None and ll is not None:
                    if not (np.isnan(ul) or np.isnan(ll)):
                        if adjusted > ul or adjusted < ll:
                            violations += 1
                        max_err = max(max_err, abs(adjusted))
            return violations * 1e6 + max_err

        try:
            # Stage 1: coarse grid search
            best_params = (initial_offset, 1.0)
            best_cost = objective(best_params)
            offset_range = abs(initial_offset) + 0.02
            for slope_candidate in [0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10]:
                for offset_factor in np.linspace(-offset_range, offset_range, 11):
                    cost = objective((offset_factor, slope_candidate))
                    if cost < best_cost:
                        best_cost = cost
                        best_params = (offset_factor, slope_candidate)

            # Stage 2: Nelder-Mead refinement
            result = optimize.minimize(
                objective,
                x0=best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 500}
            )
            return float(result.x[0]), float(result.x[1])
        except Exception:
            return initial_offset, 1.0
```

- [ ] **Step 2: Update `_calculate_linearity` to use type-aware adjustment**

Modify the `_calculate_linearity` method (around line 306) to accept linearity_type and positions, and use the new `_calculate_optimal_adjustment`:

```python
    def _calculate_linearity(
        self,
        positions: List[float],
        errors: List[float],
        upper_limits: List[float],
        lower_limits: List[float],
        linearity_spec: float,
        linearity_type: Optional[str] = None,
    ) -> Tuple[float, float, float, bool, int, float, int]:
        """
        Calculate linearity metrics with spec-aware optimal adjustment.

        Returns:
            (optimal_offset, optimal_slope, linearity_error, linearity_pass,
             fail_points, raw_linearity_error, raw_fail_points)
        """
        # Calculate raw results (no adjustment)
        raw_fail_points = self._count_fail_points(errors, upper_limits, lower_limits)
        raw_linearity_error = max(abs(e) for e in errors) if errors else 0.0

        # Calculate optimal adjustment (constrained by linearity type)
        optimal_offset, optimal_slope = self._calculate_optimal_adjustment(
            positions, errors, upper_limits, lower_limits, linearity_type
        )

        # Apply adjustment: adjusted = error * slope + offset
        shifted_errors = [e * optimal_slope + optimal_offset for e in errors]

        # Calculate optimized max error
        linearity_error = max(abs(e) for e in shifted_errors) if shifted_errors else 0.0

        # Count fail points after adjustment
        fail_points = self._count_fail_points(shifted_errors, upper_limits, lower_limits)

        # Linearity passes only if ALL points are within limits (zero tolerance)
        linearity_pass = fail_points == 0

        logger.debug(
            f"Linearity: type={linearity_type}, offset={optimal_offset:.6f}, "
            f"slope={optimal_slope:.6f}, error={linearity_error:.6f}, "
            f"fail_points={fail_points} (raw={raw_fail_points}), pass={linearity_pass}"
        )

        return (optimal_offset, optimal_slope, linearity_error, linearity_pass,
                fail_points, raw_linearity_error, raw_fail_points)
```

- [ ] **Step 3: Update `analyze_track` to pass linearity_type and store new fields**

Update the `analyze_track` method (around line 60) to:
1. Accept `linearity_type` parameter
2. Accept `station_compensation` parameter
3. Pass them through to `_calculate_linearity`
4. Store the new fields in the returned `TrackData`

Add parameters to the method signature:

```python
    def analyze_track(
        self,
        track_data: Dict[str, Any],
        model: Optional[str] = None,
        linearity_type: Optional[str] = None,
        station_compensation: Optional[float] = None,
    ) -> TrackData:
```

Update the linearity analysis call (around line 100-103) to use the new method:

```python
        # Linearity analysis (spec-aware)
        (optimal_offset, optimal_slope, linearity_error, linearity_pass,
         fail_points, raw_linearity_error, raw_fail_points) = self._calculate_linearity(
            positions, errors, upper_limits, lower_limits, linearity_spec,
            linearity_type=linearity_type
        )
```

Update the shifted_errors calculation (around line 131) to apply both offset and slope:

```python
        # Apply adjustment for margin calculation
        shifted_errors = [e * optimal_slope + optimal_offset for e in errors]
```

Add the new fields to the `TrackData` constructor return (around line 151-191):

```python
            optimal_slope=optimal_slope,
            station_compensation=station_compensation,
            linearity_type=linearity_type,
            raw_linearity_error=raw_linearity_error,
            optimized_linearity_error=linearity_error,
            raw_fail_points=raw_fail_points,
```

- [ ] **Step 4: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/analyzer.py').read()); print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/core/analyzer.py
git commit -m "feat: upgrade analyzer with linearity-type-aware offset+slope optimization"
```

---

### Task 3: Parse Compensation Values from File Cells

**Files:**
- Modify: `src/laser_trim_analyzer/core/parser.py`
- Modify: `src/laser_trim_analyzer/core/final_test_parser.py`
- Modify: `src/laser_trim_analyzer/utils/constants.py`

Compensation values are the offset the station actually applied during testing. Three sources:
- **FT files (Format 1):** Cell M4 on Sheet1 (column 12, row 3)
- **LTS (System B) trim files:** Cell O1 on "Lin Chart" sheet (column 14, row 0)
- **DLTS (System A) trim files:** Cell B7 on the trimmed track sheet (column 1, row 6)

The constants for System A (B7) and System B (O1) already exist in `SYSTEM_A_CELLS["compensation"]` and `SYSTEM_B_CELLS["compensation"]`. We need to add the FT constant and implement parsing.

- [ ] **Step 1: Add FT compensation constant**

In `src/laser_trim_analyzer/utils/constants.py`, add after the existing `FINAL_TEST_FORMAT1_METADATA` dict (around line 127):

```python
# Final Test compensation cell location (Format 1)
# Cell M4 on Sheet1 contains the compensation/offset value applied by the test station
FINAL_TEST_FORMAT1_COMPENSATION: Final[Dict[str, Any]] = {
    "cell": "M4",       # Column M (12), Row 4 (3 zero-indexed)
    "col": 12,
    "row": 3,
}
```

- [ ] **Step 2: Parse compensation from trim files (parser.py)**

In `src/laser_trim_analyzer/core/parser.py`, in the `_extract_track_data` method (around line 505), after extracting `measured_electrical_angle` (around line 567) and before extracting untrimmed data (around line 571), add compensation extraction:

```python
            # Extract station compensation value
            # System A (DLTS): B7 on trimmed track sheet
            # System B (LTS): O1 on "Lin Chart" sheet (separate sheet, not the data sheet)
            station_compensation = None
            if "compensation" in cells:
                if system_type == SystemType.B:
                    # System B compensation is on the "Lin Chart" sheet, not the data sheet
                    from laser_trim_analyzer.utils.constants import SYSTEM_B_COMPENSATION_SHEET
                    try:
                        comp_sheet = SYSTEM_B_COMPENSATION_SHEET
                        if comp_sheet in xl.sheet_names:
                            df_comp = pd.read_excel(xl, sheet_name=comp_sheet, header=None, nrows=5)
                            station_compensation = self._get_cell_from_df(df_comp, cells["compensation"])
                            del df_comp
                    except Exception as e:
                        logger.debug(f"Could not read compensation from {comp_sheet}: {e}")
                else:
                    # System A compensation is on the same trimmed sheet (already loaded as df)
                    # But df was already deleted above. Re-read just the header rows.
                    try:
                        df_meta = pd.read_excel(xl, sheet_name=trimmed_sheet, header=None, nrows=10)
                        station_compensation = self._get_cell_from_df(df_meta, cells["compensation"])
                        del df_meta
                    except Exception as e:
                        logger.debug(f"Could not read compensation from {trimmed_sheet}: {e}")
```

**Important:** The `df` variable is deleted earlier (line 593: `del df`), so we need to move the compensation extraction before that deletion, or re-read just the header rows. The cleanest approach is to extract it from `df` before `del df`. Insert the compensation extraction right after the `measured_electrical_angle` extraction and before `del df`:

Actually, looking at the code flow more carefully: the `df` variable is available until line 593 (`del df`). The compensation cell for System A is B7 which is in the first few rows, so we can read it from `df` directly. For System B, it's on a different sheet ("Lin Chart") so we need a separate read. Insert the extraction before `del df` (around line 593):

```python
            # Extract compensation before freeing df
            station_compensation = None
            if "compensation" in cells:
                if system_type == SystemType.B:
                    from laser_trim_analyzer.utils.constants import SYSTEM_B_COMPENSATION_SHEET
                    try:
                        comp_sheet = SYSTEM_B_COMPENSATION_SHEET
                        if comp_sheet in xl.sheet_names:
                            df_comp = pd.read_excel(xl, sheet_name=comp_sheet, header=None, nrows=5)
                            station_compensation = self._get_cell_from_df(df_comp, cells["compensation"])
                            del df_comp
                    except Exception as e:
                        logger.debug(f"Could not read LTS compensation from Lin Chart: {e}")
                else:
                    # System A: compensation cell is on the same sheet we already have loaded
                    station_compensation = self._get_cell_from_df(df, cells["compensation"])
```

Then add `station_compensation` to the returned dict (around line 623):

```python
                "station_compensation": station_compensation,
```

- [ ] **Step 3: Parse compensation from FT files (final_test_parser.py)**

In `src/laser_trim_analyzer/core/final_test_parser.py`, in the `_parse_format1` method, after extracting metadata from file content (around line 168), add compensation extraction:

```python
            # Extract compensation from cell M4 (column 12, row 3)
            from laser_trim_analyzer.utils.constants import FINAL_TEST_FORMAT1_COMPENSATION
            comp_info = FINAL_TEST_FORMAT1_COMPENSATION
            if df.shape[0] > comp_info["row"] and df.shape[1] > comp_info["col"]:
                comp_val = df.iloc[comp_info["row"], comp_info["col"]]
                if pd.notna(comp_val):
                    try:
                        metadata["station_compensation"] = float(comp_val)
                    except (ValueError, TypeError):
                        pass
```

Note: The `df` for Sheet1 is available in the try block at this point. The compensation should be added to `metadata` dict and then propagated through to the track data.

Also add `station_compensation` to the track dicts returned by `_extract_format1_tracks`. In the track dict construction (around line 580), add:

```python
                    "station_compensation": None,  # Set from metadata in processor
```

- [ ] **Step 4: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/parser.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/final_test_parser.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/utils/constants.py').read()); print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/core/parser.py src/laser_trim_analyzer/core/final_test_parser.py src/laser_trim_analyzer/utils/constants.py
git commit -m "feat: parse station compensation from trim and FT file cells"
```

---

### Task 4: Wire Processor to Pass Linearity Type and Compensation to Analyzer

**Files:**
- Modify: `src/laser_trim_analyzer/core/processor.py`

The processor is the bridge between parsing and analysis. It needs to:
1. Look up linearity_type from `model_specs` for the parsed model
2. Pass compensation from parsed track data through to the analyzer
3. Pass linearity_type to `analyzer.analyze_track()`

- [ ] **Step 1: Add model spec lookup to processor**

Add a method to `Processor` for looking up model specs:

```python
    def _get_linearity_type(self, model: str) -> Optional[str]:
        """Look up linearity type from model_specs table."""
        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            spec = db.get_model_spec(model)
            if spec:
                return spec.get("linearity_type")
        except Exception as e:
            logger.debug(f"Could not look up model spec for {model}: {e}")
        return None
```

- [ ] **Step 2: Pass linearity_type and compensation to analyze_track**

Find the location in the processor where `self.analyzer.analyze_track()` is called (search for `analyze_track`). Update each call site to include the new parameters:

```python
        # Look up linearity type for this model
        linearity_type = self._get_linearity_type(metadata.model)

        # Get compensation from parsed track data
        station_compensation = track_data.get("station_compensation")
```

Then pass them to the analyzer call:

```python
        track_result = self.analyzer.analyze_track(
            track_data,
            model=metadata.model,
            linearity_type=linearity_type,
            station_compensation=station_compensation,
        )
```

For FT processing, check if compensation was in the FT metadata and propagate it to tracks:

```python
        # For FT files, compensation may be in metadata rather than track data
        ft_compensation = ft_parsed.get("metadata", {}).get("station_compensation")
        for track in ft_parsed.get("tracks", []):
            if track.get("station_compensation") is None and ft_compensation is not None:
                track["station_compensation"] = ft_compensation
```

- [ ] **Step 3: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/processor.py').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/core/processor.py
git commit -m "feat: wire processor to pass linearity type and compensation to analyzer"
```

---

### Task 5: Store New Fields in Database

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`

The database manager saves `TrackData` results into `TrackResult` rows. We need to ensure the new fields are persisted.

- [ ] **Step 1: Add migration for new columns**

In `_ensure_tables()` or the init migration section, add column creation with `checkfirst`-style logic. Since SQLAlchemy's `create_all` only creates missing tables (not columns), use an explicit ALTER TABLE approach:

```python
        # Phase 2: Add spec-aware optimization columns to track_results
        phase2_columns = {
            "optimal_slope": "FLOAT DEFAULT 1.0",
            "station_compensation": "FLOAT",
            "linearity_type": "VARCHAR(30)",
            "raw_linearity_error": "FLOAT",
            "optimized_linearity_error": "FLOAT",
            "raw_fail_points": "INTEGER",
        }
        for col_name, col_type in phase2_columns.items():
            try:
                self.engine.execute(text(
                    f"ALTER TABLE track_results ADD COLUMN {col_name} {col_type}"
                ))
                logger.info(f"Added column track_results.{col_name}")
            except Exception:
                pass  # Column already exists
```

Note: Use the existing migration pattern in the file. Check whether the codebase uses `engine.execute()` or `with engine.connect() as conn: conn.execute()`. Use whichever pattern already exists.

- [ ] **Step 2: Persist new fields when saving track results**

Find the method that saves TrackData/AnalysisResult to the database (likely `save_result` or similar). Add the new fields to the track result mapping:

```python
            # Spec-aware optimization fields (Phase 2)
            optimal_slope=track.optimal_slope if hasattr(track, 'optimal_slope') else 1.0,
            station_compensation=getattr(track, 'station_compensation', None),
            linearity_type=getattr(track, 'linearity_type', None),
            raw_linearity_error=getattr(track, 'raw_linearity_error', None),
            optimized_linearity_error=getattr(track, 'optimized_linearity_error', None),
            raw_fail_points=getattr(track, 'raw_fail_points', None),
```

- [ ] **Step 3: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py
git commit -m "feat: persist spec-aware optimization fields in database"
```

---

### Task 6: Update Analyze Page with Station vs. Optimal Comparison

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/analyze.py`

When the user views a file in the Analyze page, show:
- The linearity type (from model specs)
- Station compensation value (what the machine used)
- Optimal adjustment (what the optimizer found)
- Raw vs. optimized results comparison

- [ ] **Step 1: Add optimization comparison to the File Info display**

Find the `_update_file_info()` or equivalent method. After the existing Model Specifications section (added in Phase 1), add an optimization comparison section:

```python
        # Show spec-aware optimization results
        track = analysis.primary_track if hasattr(analysis, 'primary_track') else None
        if track:
            lines.append("")
            lines.append("-" * 40)
            lines.append("  LINEARITY OPTIMIZATION")
            lines.append("-" * 40)

            if track.linearity_type:
                lines.append(f"  Linearity Type:    {track.linearity_type}")

            if track.station_compensation is not None:
                lines.append(f"  Station Comp:      {track.station_compensation:.6f}")

            lines.append(f"  Optimal Offset:    {track.optimal_offset:.6f}")

            if hasattr(track, 'optimal_slope') and track.optimal_slope != 1.0:
                lines.append(f"  Optimal Slope:     {track.optimal_slope:.6f}")

            if track.raw_linearity_error is not None and track.optimized_linearity_error is not None:
                lines.append(f"  Raw Error:         {track.raw_linearity_error:.6f}")
                lines.append(f"  Optimized Error:   {track.optimized_linearity_error:.6f}")
                if track.raw_linearity_error > 0:
                    improvement = ((track.raw_linearity_error - track.optimized_linearity_error)
                                   / track.raw_linearity_error * 100)
                    lines.append(f"  Error Reduction:   {improvement:.1f}%")

            if track.raw_fail_points is not None:
                lines.append(f"  Raw Fail Points:   {track.raw_fail_points}")
                lines.append(f"  Opt. Fail Points:  {track.linearity_fail_points}")

            # Station vs optimal comparison
            if track.station_compensation is not None:
                station_offset = track.station_compensation
                optimal_offset = track.optimal_offset
                delta = optimal_offset - station_offset
                if abs(delta) > 0.0001:
                    lines.append(f"  Station-Optimal:   {delta:+.6f} (station {'under' if delta > 0 else 'over'}-compensating)")
                else:
                    lines.append(f"  Station-Optimal:   ~0 (well-matched)")
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/analyze.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/analyze.py
git commit -m "feat: show station vs optimal comparison in Analyze page"
```

---

### Task 7: Apply Spec-Aware Analysis to Final Test Processing

**Files:**
- Modify: `src/laser_trim_analyzer/core/processor.py`

The analyzer must apply the same linearity-type-aware optimization when processing FT files, not just trim files. This ensures consistent comparison between trim and FT results.

- [ ] **Step 1: Verify FT processing path uses analyze_track with linearity_type**

Find the FT file processing method in `processor.py` (likely `_process_final_test_file` or similar). Ensure it:

1. Looks up `linearity_type` the same way as trim processing
2. Passes `station_compensation` from the FT parser metadata
3. Calls `self.analyzer.analyze_track()` with both parameters

The FT processing path may construct track data differently or use a separate code path. The key requirement is that this code:

```python
        # For FT files: same spec-aware analysis as trim files
        linearity_type = self._get_linearity_type(model)
        ft_compensation = parsed_data.get("metadata", {}).get("station_compensation")

        for track_data in parsed_data.get("tracks", []):
            # Set compensation from metadata if not already on track
            if track_data.get("station_compensation") is None:
                track_data["station_compensation"] = ft_compensation

            track_result = self.analyzer.analyze_track(
                track_data,
                model=model,
                linearity_type=linearity_type,
                station_compensation=track_data.get("station_compensation"),
            )
```

is applied consistently. If the FT processing constructs AnalysisResult differently (e.g., skipping the analyzer), update it to use the analyzer.

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/processor.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/core/processor.py
git commit -m "feat: apply spec-aware optimization to Final Test processing"
```

---

### Task 8: Update Chart Widget for Slope Visualization

**Files:**
- Modify: `src/laser_trim_analyzer/gui/widgets/chart.py`

When the analyzer applies a slope adjustment (not just offset), the linearity chart should show both the raw error curve and the optimized (offset+slope adjusted) error curve, so the user can see the effect.

- [ ] **Step 1: Add optimized curve to linearity chart**

Find the linearity chart plotting method. When `optimal_slope != 1.0`, add a second line showing the optimized errors:

```python
        # If slope adjustment was applied, show both raw and optimized curves
        if hasattr(track, 'optimal_slope') and track.optimal_slope is not None and abs(track.optimal_slope - 1.0) > 0.001:
            # Plot raw errors (lighter, dashed)
            ax.plot(positions, errors, '--', color='gray', alpha=0.5,
                    linewidth=0.8, label='Raw errors')

            # Plot optimized errors (main line)
            optimized = [e * track.optimal_slope + track.optimal_offset for e in errors]
            ax.plot(positions, optimized, '-', color='#2196F3',
                    linewidth=1.0, label=f'Optimized (slope={track.optimal_slope:.3f})')

            ax.legend(loc='best', fontsize=7)
        else:
            # Standard: just show offset-shifted errors
            shifted = [e + track.optimal_offset for e in errors]
            ax.plot(positions, shifted, '-', color='#2196F3', linewidth=1.0)
```

This requires finding the exact location in the chart code where error data is plotted. The chart widget likely receives a TrackData object and plots `error_data` with `position_data`.

- [ ] **Step 2: Add compensation annotation**

If `station_compensation` is available, add a small text annotation showing it:

```python
        if hasattr(track, 'station_compensation') and track.station_compensation is not None:
            ax.annotate(
                f'Station comp: {track.station_compensation:.4f}',
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=7, color='gray', alpha=0.7,
            )
```

- [ ] **Step 3: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/widgets/chart.py').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/gui/widgets/chart.py
git commit -m "feat: show raw vs optimized curves when slope adjustment applied"
```

---

### Task 9: Update Excel Export with New Fields

**Files:**
- Modify: `src/laser_trim_analyzer/export/excel.py`

The Excel export should include the new spec-aware fields so users can see them in reports.

- [ ] **Step 1: Add new columns to export**

Find the export method that writes track-level data. Add columns for:

```python
        # Spec-aware optimization columns
        "Linearity Type",
        "Station Compensation",
        "Optimal Offset",
        "Optimal Slope",
        "Raw Error",
        "Optimized Error",
        "Raw Fail Points",
        "Opt. Fail Points",
```

And populate them from the track data:

```python
        linearity_type=getattr(track, 'linearity_type', '') or '',
        station_comp=getattr(track, 'station_compensation', ''),
        optimal_offset=track.optimal_offset,
        optimal_slope=getattr(track, 'optimal_slope', 1.0),
        raw_error=getattr(track, 'raw_linearity_error', ''),
        optimized_error=getattr(track, 'optimized_linearity_error', ''),
        raw_fail_points=getattr(track, 'raw_fail_points', ''),
        opt_fail_points=track.linearity_fail_points,
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/export/excel.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/export/excel.py
git commit -m "feat: add spec-aware optimization fields to Excel export"
```

---

### Task 10: Final Verification and Integration Test

**Files:**
- All modified files

- [ ] **Step 1: Full syntax check on all modified files**

```bash
python3 -c "
import ast, os
files = [
    'src/laser_trim_analyzer/core/models.py',
    'src/laser_trim_analyzer/core/analyzer.py',
    'src/laser_trim_analyzer/core/parser.py',
    'src/laser_trim_analyzer/core/final_test_parser.py',
    'src/laser_trim_analyzer/core/processor.py',
    'src/laser_trim_analyzer/database/models.py',
    'src/laser_trim_analyzer/database/manager.py',
    'src/laser_trim_analyzer/gui/pages/analyze.py',
    'src/laser_trim_analyzer/gui/widgets/chart.py',
    'src/laser_trim_analyzer/export/excel.py',
    'src/laser_trim_analyzer/utils/constants.py',
]
for f in files:
    ast.parse(open(f).read())
    print(f'OK: {f}')
print('All files pass syntax check')
"
```

- [ ] **Step 2: Verify analyzer logic with inline test**

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.core.analyzer import Analyzer
import numpy as np

a = Analyzer()

# Generate test data: 100 points, linearity spec 0.01
positions = list(np.linspace(0, 100, 100))
errors = list(np.random.normal(0.002, 0.001, 100))
upper = [0.01] * 100
lower = [-0.01] * 100

# Test 1: Absolute - no adjustment
off, slope = a._calculate_optimal_adjustment(positions, errors, upper, lower, 'Absolute')
assert off == 0.0 and slope == 1.0, f'Absolute failed: off={off}, slope={slope}'
print('PASS: Absolute returns (0, 1.0)')

# Test 2: Independent - free offset+slope
off, slope = a._calculate_optimal_adjustment(positions, errors, upper, lower, 'Independent')
print(f'PASS: Independent returns off={off:.6f}, slope={slope:.6f}')

# Test 3: Zero-Based - slope only
off, slope = a._calculate_optimal_adjustment(positions, errors, upper, lower, 'Zero-Based')
assert off == 0.0, f'Zero-Based offset should be 0, got {off}'
print(f'PASS: Zero-Based returns off=0, slope={slope:.6f}')

# Test 4: Term Base - same as Absolute
off, slope = a._calculate_optimal_adjustment(positions, errors, upper, lower, 'Term Base')
assert off == 0.0 and slope == 1.0, f'Term Base failed: off={off}, slope={slope}'
print('PASS: Term Base returns (0, 1.0)')

# Test 5: None/unknown - legacy offset-only
off, slope = a._calculate_optimal_adjustment(positions, errors, upper, lower, None)
assert slope == 1.0, f'Legacy should have slope=1.0, got {slope}'
print(f'PASS: Legacy (None) returns off={off:.6f}, slope=1.0')

# Test 6: Full analyze_track with linearity_type
track_data = {
    'track_id': 'TRK1',
    'positions': positions,
    'errors': errors,
    'upper_limits': upper,
    'lower_limits': lower,
    'travel_length': 100.0,
    'linearity_spec': 0.01,
}
result = a.analyze_track(track_data, model='8340-1', linearity_type='Independent')
print(f'PASS: analyze_track with Independent: offset={result.optimal_offset:.6f}, slope={result.optimal_slope:.6f}')
print(f'  raw_error={result.raw_linearity_error:.6f}, opt_error={result.optimized_linearity_error:.6f}')
assert result.linearity_type == 'Independent'
assert result.raw_linearity_error is not None
assert result.optimized_linearity_error is not None
print('All analyzer tests passed')
"
```

- [ ] **Step 3: Verify compensation parsing compiles**

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from laser_trim_analyzer.utils.constants import (
    SYSTEM_A_CELLS, SYSTEM_B_CELLS, SYSTEM_B_COMPENSATION_SHEET,
    FINAL_TEST_FORMAT1_COMPENSATION
)
print(f'System A comp cell: {SYSTEM_A_CELLS[\"compensation\"]}')
print(f'System B comp cell: {SYSTEM_B_CELLS[\"compensation\"]} (sheet: {SYSTEM_B_COMPENSATION_SHEET})')
print(f'FT comp cell: M{FINAL_TEST_FORMAT1_COMPENSATION[\"row\"]+1} (col {FINAL_TEST_FORMAT1_COMPENSATION[\"col\"]})')
print('All compensation constants OK')
"
```

- [ ] **Step 4: Commit and tag**

```bash
git add -A
git commit -m "feat: Phase 2 complete - Spec-Aware Analysis Engine"
git push origin v5-upgrade
```
