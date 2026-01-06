# ML Redesign Progress Tracker

> **Last Updated**: 2025-12-27
> **Status**: COMPLETE

---

## Quick Status

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Core Infrastructure | **Complete** | 6/6 |
| Phase 2: Training Pipeline | **Complete** | 6/6 |
| Phase 3: Apply to Database | **Complete** | 4/4 |
| Phase 4: UI Integration | **Complete** | 5/5 |
| Phase 5: Cleanup | **Complete** | 5/5 |

---

## Phase 1: Core Infrastructure

### Tasks
- [x] 1.1 Create `ModelPredictor` class in `ml/predictor.py`
- [x] 1.2 Create `ModelThresholdOptimizer` class in `ml/threshold_optimizer.py`
- [x] 1.3 Create `ModelDriftDetector` class in `ml/drift_detector.py`
- [x] 1.4 Create `ModelProfiler` class in `ml/profiler.py`
- [x] 1.5 Create `MLManager` class in `ml/manager.py`
- [x] 1.6 Add `model_ml_state` table to database

### Files to Create/Modify
- `src/laser_trim_analyzer/ml/predictor.py` (NEW)
- `src/laser_trim_analyzer/ml/threshold_optimizer.py` (NEW)
- `src/laser_trim_analyzer/ml/drift_detector.py` (NEW)
- `src/laser_trim_analyzer/ml/profiler.py` (NEW)
- `src/laser_trim_analyzer/ml/manager.py` (NEW)
- `src/laser_trim_analyzer/database/models.py` (ADD table)

### Notes
- Keep old `threshold.py` and `drift.py` until Phase 5
- New classes are independent, don't modify existing code yet

---

## Phase 2: Training Pipeline

### Tasks
- [x] 2.1 Feature extraction from DB records (TrackResult → feature dict)
- [x] 2.2 Training data gathering (Trim + Final Test, with severity weighting)
- [x] 2.3 Per-model RandomForest training in ModelPredictor
- [x] 2.4 Severity-weighted threshold calculation in ModelThresholdOptimizer
- [x] 2.5 Profile statistics calculation in ModelProfiler
- [x] 2.6 Drift baseline calculation in ModelDriftDetector

### Files to Modify
- `src/laser_trim_analyzer/ml/predictor.py`
- `src/laser_trim_analyzer/ml/threshold_optimizer.py`
- `src/laser_trim_analyzer/ml/drift_detector.py`
- `src/laser_trim_analyzer/ml/profiler.py`
- `src/laser_trim_analyzer/ml/manager.py`
- `src/laser_trim_analyzer/database/manager.py` (query methods)

### Notes
- Use BOTH Trim and Final Test data for learning
- Final Test has priority when linked, but Trim data always contributes
- Severity = fail_points count, influences threshold calculation
- Minimum 50 samples per model for full training

---

## Phase 3: Apply to Database

### Tasks
- [x] 3.1 Batch update sigma_pass using learned thresholds
- [x] 3.2 Add failure_probability to track records (column already exists)
- [x] 3.3 Run drift detection and store alerts (QAAlert with DRIFT_DETECTED)
- [x] 3.4 Progress reporting for UI (ApplyProgress callback)

### Files to Modify
- `src/laser_trim_analyzer/ml/manager.py`
- `src/laser_trim_analyzer/database/manager.py`
- `src/laser_trim_analyzer/database/models.py` (if adding failure_probability column)

### Notes
- This is the "Apply to DB" button functionality
- Must handle 70k+ records efficiently (batch updates)
- Progress callback for UI feedback

---

## Phase 4: UI Integration

### Tasks
- [x] 4.1 Settings page ML section with "Train Models" and "Apply to DB" buttons
- [x] 4.2 Per-model training status table in Settings (shows first 10 models)
- [x] 4.3 Trends page with ML insights (drift alerts, threshold recommendations)
- [x] 4.4 Dashboard drift alerts
- [x] 4.5 Model comparison views (difficulty ranking, spec analysis)

### Files to Modify
- `src/laser_trim_analyzer/gui/pages/settings.py`
- `src/laser_trim_analyzer/gui/pages/trends.py`
- `src/laser_trim_analyzer/gui/pages/dashboard.py`

### Notes
- Settings: Two buttons + status table
- Trends: Drift alerts, per-model insights
- Dashboard: Active drift warnings

---

## Phase 5: Cleanup

### Tasks
- [x] 5.1 Update Processor to use MLManager instead of legacy ThresholdOptimizer
- [x] 5.2 Update Analyzer to use per-model threshold dictionary
- [x] 5.3 Update Settings page to use MLManager for re-analysis
- [x] 5.4 Remove legacy threshold.py and drift.py files
- [x] 5.5 Clean up __init__.py exports

### Files Modified
- `src/laser_trim_analyzer/core/processor.py` (uses MLManager for thresholds)
- `src/laser_trim_analyzer/core/analyzer.py` (accepts threshold dictionary)
- `src/laser_trim_analyzer/gui/pages/settings.py` (uses MLManager for re-analysis)
- `src/laser_trim_analyzer/ml/__init__.py` (removed legacy exports)
- `src/laser_trim_analyzer/ml/threshold.py` (DELETED)
- `src/laser_trim_analyzer/ml/drift.py` (DELETED)

---

## Session Log

### Session: 2025-12-27
- Updated ML_REDESIGN_PLAN.md with expanded scope
- Added ModelProfiler for statistical insights
- Added ModelThresholdOptimizer (separate from predictor)
- Clarified: Training uses BOTH Trim and Final Test data
- Clarified: Severity weighting (fail_points) influences thresholds
- Clarified: Processing is completely separate from ML
- Clarified: Two-button workflow (Train Models → Apply to DB)
- **Completed Phase 1**: All 6 core classes created
  - ModelPredictor: RandomForest failure probability prediction
  - ModelThresholdOptimizer: Severity-weighted threshold calculation
  - ModelDriftDetector: CUSUM/EWMA drift detection with baselines
  - ModelProfiler: Statistical profiling with insights
  - MLManager: Orchestrates training, persistence, and application
  - ModelMLState: Database table for persisting ML state
- **Completed Phase 2**: Training pipeline fully implemented
  - MLManager._get_training_data(): Gathers Trim + Final Test data
  - MLManager._extract_features_from_data(): Feature extraction
  - MLManager.train_model(): Orchestrates all component training
  - MLManager._save_state_to_db(): Persists to ModelMLState table
  - MLManager._load_state_from_db(): Loads from ModelMLState table
- **Completed Phase 3**: Apply to Database
  - apply_to_database() updates sigma_pass, failure_probability
  - Drift detection runs on each track, creates QAAlerts
  - get_drift_status() provides current drift state per model
- **Completed Phase 4**: UI Integration
  - Settings page: Train Models and Apply to DB buttons with progress
  - Settings page: Per-model training status table (top 10 models)
  - Trends page: _get_ml_recommendations() uses MLManager for thresholds
  - Trends page: _update_detail_ml() shows drift status, difficulty, insights
  - Trends page: _update_ml_summary() shows difficulty ranking, drift summary
  - Dashboard: Drift alerts section shows models with active drift
- **Completed Phase 5**: Full Cleanup
  - Updated Processor to load thresholds from MLManager (database)
  - Updated Analyzer to accept threshold dictionary (no legacy imports)
  - Updated Settings page _run_reanalysis to use MLManager
  - Updated Settings page on_show to check MLManager for status
  - DELETED legacy threshold.py and drift.py files
  - Cleaned up __init__.py exports (no legacy imports)
  - Processing now uses per-model ML thresholds when available
- **ML REDESIGN COMPLETE - NO LEGACY CODE REMAINING**

### Session: 2025-12-26
- Created ML_REDESIGN_PLAN.md design document
- Created ML_PROGRESS.md progress tracker
- Reverted messy ML changes from earlier
- Kept bug fixes: xlrd, incremental error retry, Final Test None handling

---

## Bug Fixes Applied (Keep These)

These were fixed during the ML exploration session and should NOT be reverted:

1. **pyproject.toml**: Added `xlrd>=2.0.1` for .xls file support
2. **core/models.py**: Made `file_date` Optional for Final Test files
3. **core/processor.py**: Handle None values in Final Test processing
4. **database/manager.py**: `is_file_processed()` checks `success == True` (retry errors)

---

## Design Reference

See `docs/ML_REDESIGN_PLAN.md` for full architecture details.

Key classes:
- **ModelPredictor**: Per-model RandomForest for failure probability
- **ModelThresholdOptimizer**: Per-model threshold from pass/fail + severity
- **ModelDriftDetector**: Per-model CUSUM/EWMA drift detection
- **ModelProfiler**: Per-model statistical profile for insights
- **MLManager**: Orchestrates all per-model ML, handles persistence

Key principles:
- **No ML during processing** - Process 70k+ files fast
- **Train from DB** - After processing, train ML from stored data
- **Two buttons** - "Train Models" and "Apply to DB"
- **Use ALL data** - Trim + Final Test, pass/fail + severity
