# Phase 4: File Splitting & Modularization

**Duration**: 5 days
**Goal**: Split mega-files (>3000 lines) into focused modules (<800 lines each)
**Target**: -3,500 lines through deduplication and modularization

---

## Mega-Files to Split

| Priority | File | Lines | Target Modules |
|----------|------|-------|----------------|
| 1 | chart_widget.py | 4,266 | Base, Sigma, Linearity, Resistance, Multi |
| 2 | historical_page.py | 4,411 | Base, Filters, Charts, Drift, Export |
| 3 | batch_processing_page.py | 3,587 | Base, Queue, Progress, Results |
| 4 | multi_track_page.py | 3,082 | Base, Charts, Comparison, Export |
| 5 | manager.py (database) | 2,900 | Base, CRUD, Queries, ML, Processed |
| 6 | processor.py | 2,687 | DEPRECATED - keep as-is (unified_processor replaces) |

---

## Day 1: Chart Widget Splitting

### Task 1.1: Analyze chart_widget.py
- [ ] Read entire file, identify class structure
- [ ] Map chart types (sigma, linearity, resistance, multi-track)
- [ ] Identify shared utilities and base components
- [ ] Document splitting strategy

### Task 1.2: Create chart widgets module structure
- [ ] Create `gui/widgets/charts/` directory
- [ ] Create `__init__.py` with public exports
- [ ] Create `base.py` with shared chart utilities

### Task 1.3: Extract sigma charts
- [ ] Create `sigma_charts.py`
- [ ] Move sigma-related chart code
- [ ] Update imports in chart_widget.py
- [ ] Test sigma charts still work

### Task 1.4: Extract linearity charts
- [ ] Create `linearity_charts.py`
- [ ] Move linearity-related chart code
- [ ] Update imports
- [ ] Test linearity charts

### Task 1.5: Extract resistance charts
- [ ] Create `resistance_charts.py`
- [ ] Move resistance-related chart code
- [ ] Update imports
- [ ] Test resistance charts

**Day 1 Success Criteria**:
- [ ] chart_widget.py reduced from 4,266 to <1,500 lines
- [ ] All chart types still functional
- [ ] All tests passing (57/57)

---

## Day 2: Historical Page Splitting

### Task 2.1: Analyze historical_page.py
- [ ] Read entire file, identify components
- [ ] Map filters, charts, drift analysis, export sections
- [ ] Identify shared utilities
- [ ] Document splitting strategy

### Task 2.2: Create historical page module structure
- [ ] Create `gui/pages/historical/` directory
- [ ] Create `__init__.py` with HistoricalPage export
- [ ] Create `base.py` with page infrastructure

### Task 2.3: Extract filter components
- [ ] Create `filters.py`
- [ ] Move date/model/system filter code
- [ ] Update imports
- [ ] Test filtering still works

### Task 2.4: Extract drift analysis
- [ ] Create `drift_analysis.py`
- [ ] Move DriftDetector integration
- [ ] Update imports
- [ ] Test drift detection still works

### Task 2.5: Extract export functionality
- [ ] Create `export.py`
- [ ] Move export-related code
- [ ] Update imports
- [ ] Test exports still work

**Day 2 Success Criteria**:
- [ ] historical_page.py reduced from 4,411 to <1,200 lines
- [ ] All historical features functional
- [ ] All tests passing (57/57)

---

## Day 3: Batch Processing Page Splitting

### Task 3.1: Analyze batch_processing_page.py
- [ ] Read entire file, identify components
- [ ] Map queue, progress, results sections
- [ ] Identify threading/async patterns
- [ ] Document splitting strategy

### Task 3.2: Create batch processing module structure
- [ ] Create `gui/pages/batch/` directory
- [ ] Create `__init__.py` with BatchProcessingPage export
- [ ] Create `base.py` with page infrastructure

### Task 3.3: Extract queue management
- [ ] Create `queue_manager.py`
- [ ] Move file queue, drag-drop, validation code
- [ ] Update imports
- [ ] Test queue functionality

### Task 3.4: Extract progress tracking
- [ ] Create `progress_tracker.py`
- [ ] Move progress bar, status updates, threading code
- [ ] Update imports
- [ ] Test progress tracking

### Task 3.5: Extract results display
- [ ] Create `results_view.py`
- [ ] Move results tree, summary, error handling
- [ ] Update imports
- [ ] Test results display

**Day 3 Success Criteria**:
- [ ] batch_processing_page.py reduced from 3,587 to <1,000 lines
- [ ] Batch processing fully functional
- [ ] All tests passing (57/57)

---

## Day 4: Database Manager Splitting

### Task 4.1: Analyze manager.py
- [ ] Read entire file, identify method groups
- [ ] Map CRUD, queries, ML, processed files methods
- [ ] Identify shared utilities
- [ ] Document splitting strategy

### Task 4.2: Create database module structure
- [ ] Create `database/managers/` directory (or use mixins)
- [ ] Decide: Mixins vs separate managers
- [ ] Create base infrastructure

### Task 4.3: Extract ML-related methods
- [ ] Create `ml_manager.py` or mixin
- [ ] Move ML threshold, model training methods
- [ ] Update imports
- [ ] Test ML methods

### Task 4.4: Extract processed files methods
- [ ] Create `processed_files.py` or mixin
- [ ] Move incremental processing methods
- [ ] Update imports
- [ ] Test processed files tracking

### Task 4.5: Extract query methods
- [ ] Create `queries.py` or mixin
- [ ] Move complex query builders
- [ ] Update imports
- [ ] Test queries

**Day 4 Success Criteria**:
- [ ] manager.py reduced from 2,900 to <1,200 lines
- [ ] All database operations functional
- [ ] All tests passing (57/57)

---

## Day 5: Multi-Track Page + Cleanup

### Task 5.1: Analyze multi_track_page.py
- [ ] Read entire file, identify components
- [ ] Map chart, comparison, export sections
- [ ] Document splitting strategy

### Task 5.2: Split multi_track_page.py
- [ ] Create `gui/pages/multi_track/` directory
- [ ] Extract chart components
- [ ] Extract comparison logic
- [ ] Test all features

### Task 5.3: Cleanup and consolidation
- [ ] Remove any dead code discovered
- [ ] Update all imports throughout codebase
- [ ] Verify no circular imports
- [ ] Run full test suite

### Task 5.4: Documentation
- [ ] Update ARCHITECTURE.md with new structure
- [ ] Update PROGRESS.md with final metrics
- [ ] Calculate total line reduction

### Task 5.5: Phase completion
- [ ] All mega-files under 1,500 lines
- [ ] All tests passing (57/57)
- [ ] Performance benchmark (no regression)
- [ ] Final commit: [PHASE-4.5] COMPLETE

**Day 5 Success Criteria**:
- [ ] multi_track_page.py reduced from 3,082 to <1,000 lines
- [ ] All pages and widgets functional
- [ ] All tests passing (57/57)
- [ ] **Phase 4 complete**

---

## Phase 4 Success Metrics

| Metric | Before | Target | Actual |
|--------|--------|--------|--------|
| chart_widget.py | 4,266 | <1,500 | |
| historical_page.py | 4,411 | <1,200 | |
| batch_processing_page.py | 3,587 | <1,000 | |
| multi_track_page.py | 3,082 | <1,000 | |
| manager.py | 2,900 | <1,200 | |
| Total Lines Removed | - | -3,500 | |
| Tests | 57/57 | 57/57 | |

---

## Notes

- processor.py (2,687 lines) is DEPRECATED - will be removed after Phase 6
- unified_processor.py (2,161 lines) is the replacement - splitting not needed yet
- Focus on GUI files first (most duplication)
- Use feature flags if architecture changes significantly
