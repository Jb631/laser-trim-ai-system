# Phase 3: ML Integration - Checklist

**Duration**: 5 days
**Goal**: Wire ML models (FailurePredictor, DriftDetector) to processing pipeline
**Status**: 🔄 In Progress
**Progress**: 80% (Day 4 complete)

---

## Overview

Per ADR-005, ML models exist but need to be wired to the processing pipeline:
- ✅ **ThresholdOptimizer**: Already wired to sigma_analyzer (lines 294-305)
- ✅ **FailurePredictor**: Wired to UnifiedProcessor (Day 2)
- ✅ **DriftDetector**: Wired to historical analysis (Day 3)

**Priority Pattern** (following ThresholdOptimizer):
1. Check feature flag first
2. Try ML prediction if model is trained
3. Fall back to formula if ML not available
4. Log which method used for debugging

---

## Day 1: ML Infrastructure Analysis ✅ COMPLETE

**Goal**: Understand current ML architecture and plan integration points
**Status**: ✅ Complete

### Tasks

- [x] **1.1** Analyze existing ML components
  - ✅ MLPredictor class in predictors.py (1,175 lines)
  - ✅ FailurePredictor in models.py:233-466 (RandomForestClassifier)
  - ✅ DriftDetector in models.py:468-886 (IsolationForest)
  - ✅ ThresholdOptimizer in models.py:28-231 (RandomForestRegressor)

- [x] **1.2** Identify integration points
  - ✅ UnifiedProcessor._process_file_internal() → FailurePredictor
  - ✅ Historical page batch analysis → DriftDetector
  - ✅ Documented in ADR-005

- [x] **1.3** Review existing ML wiring pattern
  - ✅ ThresholdOptimizer wired in sigma_analyzer.py:273-372
  - ✅ Pattern: ML-first with formula fallback, logging which method used
  - ✅ Documented in ADR-005 as reference implementation

- [x] **1.4** Create ML integration design
  - ✅ Interface documented in ADR-005
  - ✅ Fallback behavior defined
  - ✅ Logging strategy established

- [x] **1.5** Add feature flags for ML integration
  - ✅ `use_ml_failure_predictor: false` (config.py:275-278)
  - ✅ `use_ml_drift_detector: false` (config.py:279-282)

**Key Findings**:
- MLPredictor._impl_predictor is always None (bug to fix)
- Models are registered but not automatically trained
- No training trigger from GUI

**Completion Criteria**: ✅ All met
- ✅ ML architecture documented (session-7.md)
- ✅ Integration points identified (ADR-005)
- ✅ Design pattern established (following ThresholdOptimizer)
- ✅ Feature flags added (config.py)

**Actual Time**: ~2 hours

---

## Day 2: FailurePredictor Integration ✅ COMPLETE

**Goal**: Wire FailurePredictor to UnifiedProcessor
**Status**: ✅ Complete

### Tasks

- [x] **2.1** Create ML prediction interface in UnifiedProcessor
  - ✅ Added `predict_failure()` method (unified_processor.py:1000-1048)
  - ✅ ML-first with formula fallback pattern
  - ✅ Proper logging of which method used
  - ✅ Added helper methods:
    - `_can_use_ml_failure_predictor()` - checks if ML model is available
    - `_predict_failure_ml()` - ML-based prediction
    - `_calculate_formula_failure()` - formula fallback
    - `_extract_failure_features()` - feature extraction for ML
    - `_risk_from_probability()` - probability to risk category
    - `_get_contributing_factors()` - feature importance

- [x] **2.2** Integrate FailurePredictor with analysis flow
  - ✅ Follows ThresholdOptimizer pattern (ADR-005)
  - ✅ Feature flag check first (`use_ml_failure_predictor`)
  - ✅ Returns FailurePrediction model with all required fields

- [x] **2.3** FailurePrediction model compatibility
  - ✅ Uses existing FailurePrediction model (models.py:314-319)
  - ✅ Returns: failure_probability, risk_category, gradient_margin, contributing_factors

- [x] **2.4** Test FailurePredictor integration
  - ✅ Tested formula fallback (3 test cases passed)
  - ✅ Both pass: LOW risk (0.10 prob)
  - ✅ Sigma fails: MEDIUM risk (0.40 prob)
  - ✅ Both fail: HIGH risk (0.70 prob)
  - ⏸️ ML path needs trained model to test

- [x] **2.5** Bug fix: WindowsPath in predictors.py
  - ✅ Fixed `model_path.startswith()` error (predictors.py:365-373)
  - ✅ Convert Path to string before string operations

**Completion Criteria**: ✅ All met
- ✅ FailurePredictor wired to UnifiedProcessor
- ✅ Formula fallback working correctly
- ✅ Tests passing
- ⏸️ GUI updates deferred (already displays existing predictions)

**Actual Time**: ~1 hour

---

## Day 3: DriftDetector Integration ✅ COMPLETE

**Goal**: Wire DriftDetector to historical analysis
**Status**: ✅ Complete

### Tasks

- [x] **3.1** Create drift detection interface
  - ✅ Added `detect_drift()` method to UnifiedProcessor (unified_processor.py:1277-1325)
  - ✅ ML-first with formula fallback pattern
  - ✅ Added helper methods:
    - `_can_use_ml_drift_detector()` - checks if ML model is available
    - `_detect_drift_ml()` - ML-based detection using DriftDetector model
    - `_detect_drift_formula()` - CUSUM statistical fallback
    - `_extract_drift_features()` - feature extraction for ML
    - `_classify_drift_severity_formula()` - severity classification
    - `_generate_drift_recommendations_formula()` - actionable recommendations

- [x] **3.2** Integrate DriftDetector with historical page
  - ✅ Updated `_detect_process_drift()` in historical_page.py
  - ✅ Uses UnifiedProcessor.detect_drift() with fallback
  - ✅ Added `_run_drift_detection()` wrapper method
  - ✅ Added `_detect_drift_inline_fallback()` for graceful degradation

- [x] **3.3** Drift report structure
  - ✅ Returns comprehensive report with:
    - drift_detected: bool
    - drift_severity: (negligible, low, moderate, high, critical)
    - drift_rate: float (0.0 to 1.0)
    - drift_trend: (stable, increasing, decreasing)
    - drift_points: List of detected drift indices
    - recommendations: List of actionable recommendations
    - feature_drift: Per-feature drift analysis
    - method_used: 'ml' or 'formula'

- [x] **3.4** Test DriftDetector integration
  - ✅ Tested with drifting data: Correctly detected as "critical" (28% drift rate)
  - ✅ Tested with stable data: Correctly detected as "negligible" (0% drift rate)
  - ✅ Formula fallback working correctly

- [x] **3.5** Update GUI for drift visualization
  - ✅ Historical page shows method used (ML or Statistical)
  - ✅ Displays severity, drift rate, trend, and recommendations
  - ✅ Drift chart updates with detection results
  - ✅ Drift alert card updates based on results

**Completion Criteria**: ✅ All met
- ✅ DriftDetector wired to historical analysis
- ✅ Formula fallback working correctly
- ✅ GUI displays drift information with recommendations
- ✅ Tests passing

**Actual Time**: ~1 hour

---

## Day 4: ML Pipeline Optimization ✅ COMPLETE

**Goal**: Optimize ML predictions for batch processing
**Status**: ✅ Complete

### Tasks

- [x] **4.1** Implement batch predictions
  - ✅ Added `predict_failures_batch()` method (unified_processor.py:1186-1243)
  - ✅ Added `_predict_failures_batch_ml()` for batch ML inference
  - ✅ 3.65x speedup vs individual predictions

- [x] **4.2** Add ML model caching
  - ✅ Lazy model loading already exists in predictors.py:357-432
  - ✅ Model version tracking in place (version attribute)
  - ✅ Models loaded on-demand from disk

- [x] **4.3** Add prediction caching
  - ✅ Added `get_cached_prediction()` method (unified_processor.py:1306-1322)
  - ✅ Added `cache_prediction()` method (unified_processor.py:1324-1346)
  - ✅ LRU eviction strategy (max 1000 entries)
  - ✅ Added `prediction_cache_stats` property

- [x] **4.4** Performance benchmarks
  - ✅ Individual prediction: 0.023ms/sample
  - ✅ Batch prediction: 0.006ms/sample (3.65x faster)
  - ✅ Cache read: 0.0004ms/sample (63x faster than prediction)
  - ✅ Safe prediction overhead: negligible

- [x] **4.5** Handle ML errors gracefully
  - ✅ Added `_run_ml_with_timeout()` method (unified_processor.py:1369-1419)
  - ✅ Added `_check_memory_available()` method (unified_processor.py:1421-1452)
  - ✅ Added `predict_failure_safe()` method with timeout/memory check
  - ✅ Added `detect_drift_safe()` method with timeout/memory check
  - ✅ Added `ml_health_stats` property for monitoring

**Completion Criteria**: ✅ All met
- ✅ Batch predictions working (3.65x speedup)
- ✅ ML caching implemented (lazy loading + prediction cache)
- ✅ Performance acceptable (0.006ms/sample, well under 10% overhead)
- ✅ Error handling robust (timeout, memory, graceful degradation)

**Actual Time**: ~1 hour

---

## Day 5: Testing, Documentation & Cleanup

**Goal**: Complete testing and documentation
**Status**: ⏸️ Not Started

### Tasks

- [ ] **5.1** Run full test suite
  - Unit tests for ML integration
  - Integration tests
  - Performance tests

- [ ] **5.2** Test ML fallback behavior
  - Test with no models
  - Test with failed models
  - Verify formula fallback

- [ ] **5.3** Update documentation
  - ARCHITECTURE.md with ML flow
  - Update ADR-005 status
  - PROGRESS.md updates

- [ ] **5.4** Add ML configuration docs
  - Document feature flags
  - Document model training
  - Document prediction interpretation

- [ ] **5.5** Commit and tag Phase 3 completion
  - Commit: `[PHASE-3.5] COMPLETE: ML Integration`
  - Tag: `phase-3-complete`

**Completion Criteria**:
- All tests passing
- Documentation complete
- Feature flags working
- Phase 3 complete

**Estimated Time**: 4-6 hours

---

## Phase 3 Summary

**Total Tasks**: 25
**Estimated Total Time**: 26-36 hours (5 days)

**Deliverables**:
1. FailurePredictor wired to processing pipeline
2. DriftDetector wired to historical analysis
3. Batch prediction optimization
4. Prediction caching
5. Updated documentation

**Success Metrics**:
- All tests passing: 100%
- ML overhead: <10% per file
- Fallback working: 100% coverage
- Feature flags: All working

---

## Notes

- Feature flags default to OFF (ADR-001)
- Formula fallback always available
- Log which prediction method used
- Test thoroughly before enabling by default

**Phase 3 starts when Day 1, Task 1.1 is checked**
