# Code Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all Critical and High issues from the 6-reviewer code audit -- crashes, data corruption, wrong results, UI freezes, and visual bugs.

**Architecture:** Fixes are grouped by priority. Tier 1 (crashes/data loss) first, then Tier 2 (wrong results), Tier 3 (UI/visual), Tier 4 (cleanup). Each task is self-contained and can be committed independently.

**Tech Stack:** Python, SQLAlchemy 2.0, SQLite, matplotlib, customtkinter, scikit-learn, scipy

**No test suite exists** -- verify each fix with `python3 -c "import ast; ast.parse(open('<file>').read())"` syntax checks. Manual testing at work with real data.

---

## Tier 1: Crash & Data Corruption Fixes

### Task 1: Fix nested session crash in apply_to_database

**Why:** `apply_to_database` opens a session, then calls `_save_state_to_db()` which opens a SECOND session on the same StaticPool connection. Inner rollback corrupts outer session state. This is the most likely cause of the user's reported crashes.

**Files:**
- Modify: `src/laser_trim_analyzer/ml/manager.py:765,997-1002`

- [ ] **Step 1: Refactor `_save_state_to_db` to accept an optional session**

In `src/laser_trim_analyzer/ml/manager.py`, change the method signature and body to reuse an existing session when provided:

```python
def _save_state_to_db(self, existing_session=None) -> None:
    """Save ML state to database model_ml_state table."""
    try:
        from laser_trim_analyzer.database.models import ModelMLState

        def _do_save(session):
            for model_name in self.trained_models:
                # Get or create state record
                state = session.query(ModelMLState).filter(
                    ModelMLState.model == model_name
                ).first()

                if not state:
                    state = ModelMLState(model=model_name)
                    session.add(state)

                # Update from threshold optimizer
                optimizer = self.threshold_optimizers.get(model_name)
                if optimizer and optimizer.is_calculated:
                    state.is_trained = True
                    state.sigma_threshold = optimizer.threshold
                    state.threshold_confidence = optimizer.confidence
                    state.threshold_method = optimizer.method
                    state.n_pass = optimizer.n_pass
                    state.n_fail = optimizer.n_fail
                    state.pass_sigma_mean = optimizer.pass_sigma_mean
                    state.pass_sigma_std = optimizer.pass_sigma_std
                    state.pass_sigma_max = optimizer.pass_sigma_max
                    state.fail_sigma_min = optimizer.fail_sigma_min
                    state.fail_sigma_mean = optimizer.fail_sigma_mean
                    state.avg_fail_severity = optimizer.avg_fail_severity
                    state.training_samples = optimizer.n_samples
                    state.training_date = optimizer.calculated_date

                # Update from predictor
                predictor = self.predictors.get(model_name)
                if predictor and predictor.is_trained:
                    state.predictor_trained = True
                    if predictor.metrics:
                        state.predictor_accuracy = predictor.metrics.accuracy
                        state.predictor_precision = predictor.metrics.precision
                        state.predictor_recall = predictor.metrics.recall
                        state.predictor_f1 = predictor.metrics.f1
                        state.predictor_auc = predictor.metrics.auc_roc
                    state.feature_importance = predictor.feature_importance

                # Update from profiler
                profiler = self.profilers.get(model_name)
                if profiler and profiler.profile:
                    p = profiler.profile
                    if p.sigma:
                        state.sigma_mean = p.sigma.mean
                        state.sigma_std = p.sigma.std
                        state.sigma_p5 = p.sigma.p5
                        state.sigma_p50 = p.sigma.p50
                        state.sigma_p95 = p.sigma.p95
                    if p.linearity_error:
                        state.error_mean = p.linearity_error.mean
                        state.error_std = p.linearity_error.std
                    state.pass_rate = p.pass_rate
                    state.fail_rate = p.fail_rate
                    state.linearity_pass_rate = p.linearity_pass_rate
                    state.avg_fail_points = p.avg_fail_points
                    state.track_correlation = p.track_correlation
                    state.spec_margin_percent = p.spec_margin_percent
                    state.difficulty_score = p.difficulty_score
                    state.quality_percentile = p.quality_percentile
                    state.linearity_spec = p.linearity_spec

                # Update from drift detector
                detector = self.drift_detectors.get(model_name)
                if detector and detector.has_baseline:
                    state.drift_has_baseline = True
                    state.drift_baseline_mean = detector.baseline_mean
                    state.drift_baseline_std = detector.baseline_std
                    state.drift_baseline_p5 = detector.baseline_p5
                    state.drift_baseline_p50 = detector.baseline_p50
                    state.drift_baseline_p95 = detector.baseline_p95
                    state.drift_baseline_samples = detector.baseline_samples
                    state.drift_baseline_cutoff_date = detector.baseline_cutoff_date
                    state.cusum_pos = detector.cusum_pos
                    state.cusum_neg = detector.cusum_neg
                    state.peak_cusum = detector._peak_cusum
                    state.ewma_value = detector.ewma_value
                    state.is_drifting = detector.is_drifting
                    state.drift_direction = detector.drift_direction.value if detector.drift_direction else None
                    state.drift_start_date = detector.drift_start_date
                    state.samples_since_baseline = detector.samples_since_baseline

            if not existing_session:
                session.commit()
            logger.info(f"Saved ML state for {len(self.trained_models)} models to database")

        if existing_session:
            _do_save(existing_session)
        else:
            with self.db.session() as session:
                _do_save(session)

    except Exception as e:
        logger.error(f"Error saving ML state to database: {e}")
```

- [ ] **Step 2: Pass session from apply_to_database**

Change line 765 in `apply_to_database` from:
```python
# Save updated drift state
self._save_state_to_db()
```
to:
```python
# Save updated drift state (reuse outer session to avoid nested session crash)
self._save_state_to_db(existing_session=session)
```

- [ ] **Step 3: Syntax check**

Run: `python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/ml/manager.py').read()); print('OK')"`

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/ml/manager.py
git commit -m "fix(ml): eliminate nested session crash in apply_to_database"
```

---

### Task 2: Add write lock to apply_to_database

**Why:** `apply_to_database` performs bulk writes without acquiring `_write_lock`. If user processes files while apply is running, concurrent writes cause "database is locked" crashes. Note: `_write_lock` is `threading.Lock()` (not RLock), so we must not hold it while calling methods that also acquire it.

**Files:**
- Modify: `src/laser_trim_analyzer/ml/manager.py:440`

- [ ] **Step 1: Wrap the session block with the write lock**

In `apply_to_database`, change the session block (around line 440) from:
```python
with self.db.session() as session:
```
to:
```python
with self.db._write_lock:
  with self.db.session() as session:
```

And indent the entire body of the session block by one level.

- [ ] **Step 2: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/ml/manager.py').read()); print('OK')"
git add src/laser_trim_analyzer/ml/manager.py
git commit -m "fix(ml): acquire write lock during apply_to_database"
```

---

### Task 3: Fix filename-only dedup causing silent data overwrites

**Why:** `save_analysis` matches existing records by filename only. Files from different folders with the same name silently overwrite each other. Fix: match by filename AND file_path.

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py:702-704,2397-2402`

- [ ] **Step 1: Update save_analysis filename match to include file_path**

Change the existing query (line 702-704) from:
```python
existing = session.query(DBAnalysisResult).filter(
    DBAnalysisResult.filename == analysis.metadata.filename
).first()
```
to:
```python
existing = session.query(DBAnalysisResult).filter(
    DBAnalysisResult.filename == analysis.metadata.filename,
    DBAnalysisResult.file_path == str(analysis.metadata.file_path),
).first()
```

- [ ] **Step 2: Update _update_existing_analysis similarly**

Change the query in `_update_existing_analysis` (line 2397-2402) from:
```python
existing = (
    session.query(DBAnalysisResult)
    .filter(DBAnalysisResult.filename == analysis.metadata.filename)
    .first()
)
```
to:
```python
existing = (
    session.query(DBAnalysisResult)
    .filter(
        DBAnalysisResult.filename == analysis.metadata.filename,
        DBAnalysisResult.file_path == str(analysis.metadata.file_path),
    )
    .first()
)
```

- [ ] **Step 3: Also update file_path when updating existing analysis**

In `_update_existing_analysis`, after the existing fields are updated, add:
```python
existing.file_path = str(analysis.metadata.file_path)
```

- [ ] **Step 4: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
git add src/laser_trim_analyzer/database/manager.py
git commit -m "fix(db): use file_path + filename for dedup to prevent overwrites"
```

---

### Task 4: Add missing winfo_exists() guards in process.py

**Why:** Background thread callbacks crash with Tcl errors if the user closes the app during folder scanning.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/process.py`

- [ ] **Step 1: Add guards to all unprotected callbacks**

Add `if not self.winfo_exists(): return` as the first line of each of these methods:
- `_on_folder_scanned` (~line 403)
- `_on_folder_scan_error` (~line 465)
- `_on_folder_scan_cancelled` (~line 474)
- `_on_incremental_count_ready` (~line 437)
- `_update_file_count_label` (~line 453)
- `_on_progress_update` (~line 695) -- replace the `self.is_processing` check with `if not self.winfo_exists(): return`

- [ ] **Step 2: Fix the folder scan progress lambda**

Find the lambda around line 386-388 that directly calls `self.file_count_label.configure()`. Wrap it in a method or add a guard:
```python
def _update_scan_progress(self, count, remaining):
    if not self.winfo_exists():
        return
    self.file_count_label.configure(
        text=f"Scanning... {count:,} files found ({remaining:,} folders queued)")
```
Then replace the lambda with:
```python
self.after(0, lambda c=count, r=remaining: self._update_scan_progress(c, r))
```

- [ ] **Step 3: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/process.py').read()); print('OK')"
git add src/laser_trim_analyzer/gui/pages/process.py
git commit -m "fix(gui): add winfo_exists guards to prevent crash on close during scanning"
```

---

## Tier 2: Data Correctness Fixes

### Task 5: Fix excluded points influencing offset optimization

**Why:** When user excludes outlier points, they should be excluded from BOTH pass/fail counting AND offset optimization. Currently they're only excluded from counting, so the correction is still pulled toward accommodating outliers.

**Files:**
- Modify: `src/laser_trim_analyzer/core/analyzer.py:434,462-472`

- [ ] **Step 1: Add exclude_indices parameter to _calculate_optimal_adjustment**

Change the method signature (line 462-472) to accept `exclude_indices`:
```python
def _calculate_optimal_adjustment(
    self,
    positions: List[float],
    errors: List[float],
    upper_limits: List[float],
    lower_limits: List[float],
    linearity_type: Optional[str] = None,
    angle_spec: Optional[float] = None,
    angle_tol: Optional[float] = None,
    angle_tol_type: Optional[str] = None,
    exclude_indices: Optional[set] = None,
) -> Tuple[float, float]:
```

- [ ] **Step 2: Filter excluded points in the optimization objective**

Inside `_calculate_optimal_adjustment`, find where the violation count / cost function is computed. In `_calculate_optimal_offset` and `_optimize_offset_and_slope`, modify the objective to skip excluded indices when counting violations:

In `_calculate_optimal_offset` (~line 699), change the `violation_count` inner function to skip excluded points:
```python
def violation_count(offset):
    count = 0
    for i, (e, u, l) in enumerate(zip(errors, upper_limits, lower_limits)):
        if exclude_indices and i in exclude_indices:
            continue
        shifted = e + offset
        if u is not None and shifted > u:
            count += 1
        if l is not None and shifted < l:
            count += 1
    return count
```

Similarly update `_optimize_offset_and_slope` (~line 655) to pass `exclude_indices` through.

- [ ] **Step 3: Pass exclude_indices from _calculate_linearity**

At line 434, change the call to pass `exclude_indices`:
```python
optimal_offset, optimal_slope = self._calculate_optimal_adjustment(
    positions, errors, upper_limits, lower_limits, linearity_type,
    angle_spec=angle_spec,
    angle_tol=angle_tol,
    angle_tol_type=angle_tol_type,
    exclude_indices=exclude_indices,
)
```

- [ ] **Step 4: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/analyzer.py').read()); print('OK')"
git add src/laser_trim_analyzer/core/analyzer.py
git commit -m "fix(analyzer): exclude user-excluded points from offset optimization"
```

---

### Task 6: Fix NaN error padding fabricating perfect data

**Why:** When error column is shorter than positions, padding with 0.0 fabricates "perfect" measurements that dilute sigma and improve linearity artificially. Fix: truncate positions to match errors, and log a warning.

**Files:**
- Modify: `src/laser_trim_analyzer/core/parser.py:543-548`

- [ ] **Step 1: Replace zero-padding with position truncation**

Change lines 543-548 from:
```python
# Trim errors to match positions length (or pad if needed)
if len(errors) > len(positions):
    errors = errors[:len(positions)]
elif len(errors) < len(positions):
    # Pad with zeros if error column is shorter
    errors = errors + [0.0] * (len(positions) - len(errors))
```
to:
```python
# Align positions and errors arrays — never fabricate data
if len(errors) > len(positions):
    errors = errors[:len(positions)]
elif len(errors) < len(positions):
    # Truncate positions to match errors — don't pad with fake 0.0 values
    logger.warning(
        f"SANITY: {file_path.name} [{trimmed_sheet}] "
        f"error column ({len(errors)} pts) shorter than positions ({len(positions)} pts) "
        f"— truncating positions to match"
    )
    positions = positions[:len(errors)]
```

- [ ] **Step 2: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/core/parser.py').read()); print('OK')"
git add src/laser_trim_analyzer/core/parser.py
git commit -m "fix(parser): truncate positions instead of padding errors with fake zeros"
```

---

### Task 7: Add threshold zero guard

**Why:** `_weighted_threshold`, `_separation_threshold`, and `_percentile_threshold` can produce 0.0 for models with very low sigma values. The DB constraint `CHECK(sigma_threshold > 0)` then causes `apply_to_database` to fail for that model.

**Files:**
- Modify: `src/laser_trim_analyzer/ml/threshold_optimizer.py:188-189`

- [ ] **Step 1: Add minimum guard after strategy selection**

After line 189 where `self.threshold = result.threshold`, add a guard:
```python
# Store results — enforce minimum to satisfy DB constraint
self.threshold = max(0.00005, result.threshold)
result.threshold = self.threshold
```

- [ ] **Step 2: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/ml/threshold_optimizer.py').read()); print('OK')"
git add src/laser_trim_analyzer/ml/threshold_optimizer.py
git commit -m "fix(ml): enforce minimum threshold to prevent DB constraint violation"
```

---

### Task 8: Fix SafeJSON shared mutable default

**Why:** `SafeJSON._get_none_value()` returns the same list object for all rows. If any code mutates the returned list, the mutation is visible to all rows.

**Files:**
- Modify: `src/laser_trim_analyzer/database/models.py:50-52`

- [ ] **Step 1: Return a fresh copy**

Change `_get_none_value` from:
```python
def _get_none_value(self):
    """Return the none_as default value."""
    return self.none_as
```
to:
```python
def _get_none_value(self):
    """Return a fresh copy of the none_as default value."""
    if isinstance(self.none_as, list):
        return list(self.none_as)
    if isinstance(self.none_as, dict):
        return dict(self.none_as)
    return self.none_as
```

- [ ] **Step 2: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/models.py').read()); print('OK')"
git add src/laser_trim_analyzer/database/models.py
git commit -m "fix(db): return fresh copy from SafeJSON to prevent cross-row contamination"
```

---

### Task 9: Fix status validators rejecting uppercase NAME strings

**Why:** The DB stores status as uppercase NAMEs ("PASS", "FAIL") after migration. But the validators try `StatusType(status)` which looks up by VALUE ("Pass", "Fail") — uppercase names silently become ERROR.

**Files:**
- Modify: `src/laser_trim_analyzer/database/models.py` — all three validators (~lines 268-283, 472-487, 921-933)

- [ ] **Step 1: Add NAME lookup fallback to all three validators**

In each of the three `@validates` methods (AnalysisResult, TrackResult, FinalTestTrack), change the try/except block from:
```python
try:
    return StatusType(status)
except ValueError:
    return StatusType.ERROR
```
to:
```python
try:
    return StatusType(status)
except ValueError:
    try:
        return StatusType[status]  # Try NAME lookup (e.g., "PASS")
    except KeyError:
        return StatusType.ERROR
```

Apply this change to all three validators.

- [ ] **Step 2: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/models.py').read()); print('OK')"
git add src/laser_trim_analyzer/database/models.py
git commit -m "fix(db): status validators now accept uppercase NAME strings"
```

---

## Tier 3: UI Freezes & Visual Fixes

### Task 10: Move trends tab DB queries to background threads

**Why:** Comparative, Cpk, Yield, and Drift views all run DB queries on the main thread, freezing the UI.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py` — methods `_show_comparative_trends`, `_show_cpk_trend`, `_show_yield_trend`, `_show_drift_timeline` (~lines 2319-2596)

- [ ] **Step 1: Wrap each view's DB query in a background thread**

For each of the four methods, extract the DB query into a background thread and update the UI via `self.after(0, ...)`. Follow the existing pattern from `_load_data`:

```python
def _show_comparative_trends(self):
    """Show comparative trends view."""
    # Show loading state
    # ...existing UI setup code...

    def _load():
        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            # ...existing DB query code...
            self.after(0, lambda: self._display_comparative_trends(data))
        except Exception as e:
            logger.error(f"Error loading comparative trends: {e}")

    get_thread_manager().start_thread(target=_load, name="comparative-trends")
```

Apply the same pattern to all four methods.

- [ ] **Step 2: Move `_get_ml_summary_insights` to background thread**

The `_get_ml_summary_insights` method (~line 1977) loads MLManager on the main thread. Move it into `_load_summary_data` (the existing background loader).

- [ ] **Step 3: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/trends.py').read()); print('OK')"
git add src/laser_trim_analyzer/gui/pages/trends.py
git commit -m "fix(trends): move DB queries to background threads to prevent UI freeze"
```

---

### Task 11: Fix chart export missing excluded points

**Why:** Exported chart doesn't show excluded points as gray circles, so the export can show different pass/fail than the on-screen chart.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/analyze.py` — `_plot_error_vs_position_export` (~line 1706)

- [ ] **Step 1: Pass excluded_indices to the export method and render gray circles**

In `_plot_error_vs_position_export`, add the same excluded point rendering logic from `_display_track_chart`. After plotting fail points, add:
```python
# Render excluded points as gray circles (matching on-screen display)
if excluded_indices:
    excluded_positions = [positions[i] for i in excluded_indices if i < len(positions)]
    excluded_errors = [shifted_errors[i] for i in excluded_indices if i < len(shifted_errors)]
    ax.scatter(excluded_positions, excluded_errors,
              color='gray', marker='o', s=30, alpha=0.5, zorder=5,
              label='Excluded')
```

- [ ] **Step 2: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/analyze.py').read()); print('OK')"
git add src/laser_trim_analyzer/gui/pages/analyze.py
git commit -m "fix(analyze): show excluded points in exported charts"
```

---

### Task 12: Fix heatmap figure height never resetting

**Why:** After rendering heatmap with dynamic height, the figure stays oversized for any subsequent chart on the same widget.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/widgets/chart.py` — `plot_heatmap` (~line 1909)

- [ ] **Step 1: Save and restore original figure size**

At the start of `plot_heatmap`, save the original size. At the end, schedule a restore after drawing:
```python
# Save original figure size so it can be restored when chart is cleared
self._heatmap_original_size = self.style.figure_size
```

Then in the `clear()` method, add a restore check:
```python
if hasattr(self, '_heatmap_original_size') and self._heatmap_original_size:
    self.figure.set_size_inches(*self._heatmap_original_size)
    self._heatmap_original_size = None
```

- [ ] **Step 2: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/widgets/chart.py').read()); print('OK')"
git add src/laser_trim_analyzer/gui/widgets/chart.py
git commit -m "fix(chart): restore figure size after heatmap rendering"
```

---

### Task 13: Fix Compare page NaN spec limit interpolation

**Why:** `np.interp` doesn't handle NaN in the y-values array correctly, causing the Compare page to show fewer fail points than the Analyze page for the same data.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/compare.py` (~line 988-991)

- [ ] **Step 1: Filter NaN values before interpolation**

Replace the `np.interp` calls with NaN-safe versions. Before interpolation, replace None/NaN with the nearest valid value:
```python
# Build clean limit arrays — replace None with nearest valid value for interp
def _clean_limits_for_interp(limits):
    """Replace None with nearest valid value so np.interp works correctly."""
    clean = [v for v in limits]
    # Forward fill
    last_valid = None
    for i, v in enumerate(clean):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            last_valid = v
        elif last_valid is not None:
            clean[i] = last_valid
    # Backward fill remaining
    last_valid = None
    for i in range(len(clean) - 1, -1, -1):
        if clean[i] is not None and not (isinstance(clean[i], float) and np.isnan(clean[i])):
            last_valid = clean[i]
        elif last_valid is not None:
            clean[i] = last_valid
    return [v if v is not None else 0 for v in clean]
```

Then use `_clean_limits_for_interp(upper_limits)` and `_clean_limits_for_interp(lower_limits)` before the `np.interp` calls.

- [ ] **Step 2: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/compare.py').read()); print('OK')"
git add src/laser_trim_analyzer/gui/pages/compare.py
git commit -m "fix(compare): handle NaN spec limits in interpolation"
```

---

### Task 14: Fix histogram stats vs shape mismatch

**Why:** Histogram title shows mean/std from ALL data (including outliers) but the bars only show the filtered 98th percentile subset, which is misleading.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/widgets/chart.py` — `plot_histogram` (~line 580)

- [ ] **Step 1: Compute stats from filtered data**

Find where the title stats are computed (before the 98th percentile filtering) and move them to AFTER filtering. Or add a note in the title like "(N outliers excluded)":

```python
# Compute stats from display values (excluding outliers) for consistency
display_mean = np.mean(display_values)
display_std = np.std(display_values)
n_outliers = len(values) - len(display_values)
outlier_note = f" ({n_outliers} outliers excluded)" if n_outliers > 0 else ""
title_text = f"{title}\nMean: {display_mean:.6f}, Std: {display_std:.6f}{outlier_note}"
```

- [ ] **Step 2: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/widgets/chart.py').read()); print('OK')"
git add src/laser_trim_analyzer/gui/widgets/chart.py
git commit -m "fix(chart): compute histogram stats from displayed data, not full dataset"
```

---

## Tier 4: Cleanup

### Task 15: Wire up dead ML config options

**Why:** `min_samples_for_training` defaults to 100 in config but Settings hardcodes 20. The other ML config flags (`use_threshold_optimizer`, `use_drift_detector`) are never read.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/settings.py` — `_run_training` (~line 1128)

- [ ] **Step 1: Use config value for min_samples**

Change the hardcoded `min_samples=20` in `_run_training` to read from config:
```python
results = ml_manager.train_all_models(
    min_samples=self.app.config.ml.min_samples_for_training,
    progress_callback=on_progress
)
```

- [ ] **Step 2: Set sensible default**

In config.py, verify `min_samples_for_training` defaults to 20 (not 100) since 20 is the tested value. Update if needed.

- [ ] **Step 3: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/settings.py').read()); print('OK')"
git add src/laser_trim_analyzer/gui/pages/settings.py src/laser_trim_analyzer/config.py
git commit -m "fix(settings): use config value for ML min_samples instead of hardcoded 20"
```

---

### Task 16: Persist drift recovery counter

**Why:** `_consecutive_recovered` resets to 0 on app restart, changing drift detection behavior.

**Files:**
- Modify: `src/laser_trim_analyzer/ml/manager.py` — `_save_state_to_db` and `_load_state_from_db`

- [ ] **Step 1: Add migration for consecutive_recovered column**

In `src/laser_trim_analyzer/database/manager.py`, add a migration in `_run_migrations`:
```python
try:
    session.execute(text("ALTER TABLE model_ml_state ADD COLUMN consecutive_recovered INTEGER DEFAULT 0"))
    session.commit()
except Exception:
    session.rollback()
```

- [ ] **Step 2: Save/restore in ML manager**

In `_save_state_to_db`, add:
```python
if detector and detector.has_baseline:
    # ...existing fields...
    state.consecutive_recovered = detector._consecutive_recovered
```

In `_load_state_from_db`, add:
```python
if state.drift_has_baseline:
    # ...existing fields...
    detector._consecutive_recovered = getattr(state, 'consecutive_recovered', 0) or 0
```

- [ ] **Step 3: Add column to ModelMLState model**

In `src/laser_trim_analyzer/database/models.py`, add to the `ModelMLState` class:
```python
consecutive_recovered = Column(Integer, default=0)
```

- [ ] **Step 4: Syntax check and commit**

```bash
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/ml/manager.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/models.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/database/manager.py').read()); print('OK')"
git add src/laser_trim_analyzer/ml/manager.py src/laser_trim_analyzer/database/models.py src/laser_trim_analyzer/database/manager.py
git commit -m "fix(ml): persist drift recovery counter across restarts"
```
