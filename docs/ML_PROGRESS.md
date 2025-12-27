# ML Redesign Progress Tracker

> **Last Updated**: 2025-12-26
> **Status**: Phase 1 - Not Started

---

## Quick Status

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Core Infrastructure | Not Started | 0/5 |
| Phase 2: Training Pipeline | Not Started | 0/4 |
| Phase 3: Integration | Not Started | 0/4 |
| Phase 4: UI Updates | Not Started | 0/4 |
| Phase 5: Cleanup | Not Started | 0/3 |

---

## Phase 1: Core ML Infrastructure

### Tasks
- [ ] 1.1 Create `ModelPredictor` class in `ml/predictor.py`
- [ ] 1.2 Create `ModelDriftDetector` class in `ml/drift_detector.py`
- [ ] 1.3 Create `MLManager` class in `ml/manager.py`
- [ ] 1.4 Add `model_ml_state` table to database
- [ ] 1.5 Implement save/load for per-model state

### Files to Create/Modify
- `src/laser_trim_analyzer/ml/predictor.py` (NEW)
- `src/laser_trim_analyzer/ml/drift_detector.py` (NEW)
- `src/laser_trim_analyzer/ml/manager.py` (NEW)
- `src/laser_trim_analyzer/database/models.py` (ADD table)
- `src/laser_trim_analyzer/database/manager.py` (ADD methods)

### Notes
- Keep old `threshold.py` and `drift.py` as fallback until Phase 5
- New classes should be independent, not modify existing code yet

---

## Phase 2: Training Pipeline

### Tasks
- [ ] 2.1 Feature extraction from TrackResult
- [ ] 2.2 Training data gathering (linked Final Test + unlinked trim)
- [ ] 2.3 Per-model RandomForest training
- [ ] 2.4 Optimal threshold calculation + drift baseline

### Files to Modify
- `src/laser_trim_analyzer/ml/predictor.py`
- `src/laser_trim_analyzer/ml/manager.py`
- `src/laser_trim_analyzer/database/manager.py` (query methods)

### Notes
- Use Final Test pass/fail as primary ground truth
- Fall back to linearity pass/fail when no Final Test link
- Minimum 50 samples per model for training

---

## Phase 3: Integration

### Tasks
- [ ] 3.1 Update Analyzer to accept optional MLManager
- [ ] 3.2 Use per-model threshold in `_get_threshold()`
- [ ] 3.3 Add failure probability to TrackData
- [ ] 3.4 Integrate drift detection during analysis

### Files to Modify
- `src/laser_trim_analyzer/core/analyzer.py`
- `src/laser_trim_analyzer/core/processor.py`
- `src/laser_trim_analyzer/core/models.py` (if needed)

### Notes
- Analyzer should work with or without MLManager (graceful fallback)
- Don't break existing functionality

---

## Phase 4: UI Updates

### Tasks
- [ ] 4.1 Update Settings page with ML training section
- [ ] 4.2 Add per-model training status display
- [ ] 4.3 Update Trends page with ML insights
- [ ] 4.4 Add drift alerts to Dashboard

### Files to Modify
- `src/laser_trim_analyzer/gui/pages/settings.py`
- `src/laser_trim_analyzer/gui/pages/trends.py`
- `src/laser_trim_analyzer/gui/pages/dashboard.py`

### Notes
- Settings: "Train All Models" button, status per model
- Trends: Drift alerts, threshold recommendations

---

## Phase 5: Cleanup

### Tasks
- [ ] 5.1 Decide: remove old ML or keep as fallback
- [ ] 5.2 Update documentation
- [ ] 5.3 Test with real production data

### Files to Modify
- `src/laser_trim_analyzer/ml/threshold.py` (maybe remove)
- `src/laser_trim_analyzer/ml/drift.py` (maybe remove)
- `docs/ML_REDESIGN_PLAN.md` (mark complete)
- `CHANGELOG.md`

---

## Session Log

### Session: 2025-12-26
- Created ML_REDESIGN_PLAN.md design document
- Created ML_PROGRESS.md progress tracker
- Reverted messy ML changes from earlier
- Kept bug fixes: xlrd, incremental error retry, Final Test None handling
- **Next**: Start Phase 1.1 - Create ModelPredictor class

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
- **ModelPredictor**: Per-model RandomForest for failure prediction + threshold
- **ModelDriftDetector**: Per-model CUSUM/EWMA drift detection
- **MLManager**: Orchestrates all per-model ML, handles persistence
