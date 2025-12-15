# Laser Trim Analyzer v3 - Complete Redesign Plan

**Created**: 2025-12-14
**Status**: Planning Phase
**Author**: Claude Code + User Collaboration

---

## Executive Summary

This document defines the complete redesign plan for Laser Trim Analyzer v3. The goal is to create a **simpler, cleaner, industry-standard application** focused on:

1. **Data Analysis & Insights** - Clear, actionable analysis of trim data
2. **Issue Detection** - Early warning of problems before they become failures
3. **ML-Driven Process Improvement** - Predictive analytics and optimization
4. **Simplicity** - Easy to use, easy to maintain, easy to understand

The new app will be built in `src/laser_trim_v3/` while keeping v2 intact for rollback.

---

## Analysis of Current State (v2)

### What Works Well (Keep)
| Component | Lines | Status | Keep? |
|-----------|-------|--------|-------|
| Database Schema | 949 | Solid, production-ready | ✅ Reuse |
| Pydantic Models | 623 | Well-designed | ✅ Simplify & reuse |
| ML Models (ThresholdOptimizer, DriftDetector) | ~1,300 | Working algorithms | ✅ Port cleanly |
| Excel Parsing (core logic) | ~500 | Functional | ✅ Extract & clean |
| Sigma Analysis | 389 | Core algorithm | ✅ Keep |
| Incremental Processing | ~200 | Already implemented | ✅ Integrate properly |

### What's Broken/Overly Complex (Fix or Remove)
| Component | Lines | Problem | Solution |
|-----------|-------|---------|----------|
| 4 Processor Classes | 5,753 | 36% duplication, confusing | → 1 UnifiedProcessor |
| Chart System (5 files) | 3,165 | Over-engineered, buggy | → 1 SimpleChart |
| 11 GUI Pages | 19,481 | Too many, overlap | → 5 focused pages |
| Validation (4 systems) | 2,500+ | Scattered, duplicate | → 1 validation module |
| Excel Utils (4 files) | 3,300 | Redundant | → 1 excel module |
| Database Manager | 2,900 | Monolith | → Split by concern |

### Key Metrics from v2
- **Total Files**: 110 Python files
- **Total Lines**: ~144,000 lines
- **Avg File Size**: 1,307 lines (should be ~300-500)
- **Files > 2,000 lines**: 5 (bad)
- **Files > 1,500 lines**: 12 (concerning)
- **Estimated Duplicate Code**: 10-15%

---

## v3 Design Philosophy

### Core Principles

1. **Simple > Complex**
   - Fewer files, fewer lines, clearer purpose
   - One way to do each thing
   - No optional features that don't work

2. **Data-First**
   - Process once, analyze many times
   - Database is the source of truth
   - Incremental updates only

3. **ML-Integrated**
   - ML is not bolted on, it's built-in
   - Every analysis includes ML insights
   - Thresholds are learned, not hardcoded

4. **Industry-Standard**
   - SPC control charts (I/MR charts)
   - Cpk/Ppk capability metrics
   - Standard quality terminology

5. **User-Focused**
   - Clear, actionable information
   - No jargon without explanation
   - Export what matters

### Target Metrics for v3
- **Total Files**: ~30 Python files (73% reduction)
- **Total Lines**: ~15,000 lines (90% reduction)
- **Avg File Size**: ~500 lines
- **Max File Size**: 800 lines
- **Test Coverage**: 80%+

---

## Feature Requirements (Prioritized)

### Must Have (P0) - Core Functionality
| Feature | Description | v2 Source |
|---------|-------------|-----------|
| Single File Analysis | Analyze one Excel file, show results | single_file_page.py |
| Batch Processing | Process hundreds of files efficiently | batch_processing_page.py |
| Incremental Mode | Only process new/modified files | ProcessedFile table |
| Track Comparison | Compare TA vs TB tracks on same unit | multi_track_page.py |
| Model Trends | See trends over time per model number | historical_page.py |
| Excel Export | Export results to Excel | enhanced_excel_export.py |
| System A/B Support | Handle both system types | excel_utils.py |

### Should Have (P1) - Important
| Feature | Description | v2 Source |
|---------|-------------|-----------|
| Final Line Comparison | Compare trim data to final test data | final_test_comparison_page.py |
| ML Threshold Optimization | Learn optimal pass/fail thresholds | ThresholdOptimizer |
| Drift Detection | Detect process shifts over time | DriftDetector |
| SPC Control Charts | Industry-standard I/MR charts | SimpleChartWidget |
| Dashboard Overview | Health scores, alerts, quick stats | home_page.py |

### Could Have (P2) - Nice to Have
| Feature | Description | v2 Source |
|---------|-------------|-----------|
| ML Training UI | Train models from GUI | ml_tools_page.py |
| Failure Prediction | Predict which units might fail | FailurePredictor |
| Alert Management | Track and resolve quality alerts | QAAlert table |

### Won't Have (Removed from v3)
| Feature | Reason |
|---------|--------|
| CSV/HTML Export | Complexity for minimal value |
| Multiple chart libraries | One chart system is enough |
| 4 processor strategies | Auto-select is sufficient |
| AI Insights Page | Not adding real value currently |
| Animated widgets | Unnecessary UI complexity |

---

## Architecture

### Folder Structure

```
src/laser_trim_v3/
├── __init__.py
├── __main__.py                 # Entry point
├── app.py                      # Main application window
│
├── core/                       # Core processing logic
│   ├── __init__.py
│   ├── parser.py              # Excel file parsing (~300 lines)
│   ├── analyzer.py            # Sigma, linearity, quality analysis (~400 lines)
│   ├── processor.py           # Single processor with auto-strategy (~500 lines)
│   └── models.py              # Pydantic data models (~400 lines)
│
├── ml/                         # Machine learning
│   ├── __init__.py
│   ├── threshold.py           # Threshold optimization (~300 lines)
│   └── drift.py               # Drift detection (~300 lines)
│
├── database/                   # Data persistence
│   ├── __init__.py
│   ├── models.py              # SQLAlchemy models (reuse from v2)
│   └── manager.py             # Simplified DB operations (~600 lines)
│
├── gui/                        # User interface
│   ├── __init__.py
│   ├── main_window.py         # Main application window (~400 lines)
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── dashboard.py       # Overview, alerts, quick stats (~400 lines)
│   │   ├── process.py         # Import files, batch processing (~500 lines)
│   │   ├── analyze.py         # Single file, track compare, results (~600 lines)
│   │   ├── trends.py          # Model trends, drift, ML insights (~500 lines)
│   │   └── settings.py        # Configuration (~300 lines)
│   └── widgets/
│       ├── __init__.py
│       ├── chart.py           # Single chart widget (~500 lines)
│       ├── table.py           # Data tables (~200 lines)
│       └── cards.py           # Metric cards, alerts (~200 lines)
│
├── export/                     # Export functionality
│   ├── __init__.py
│   └── excel.py               # Excel-only export (~400 lines)
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── constants.py           # Shared constants (~100 lines)
│   └── validators.py          # Input validation (~200 lines)
│
├── config.py                   # Configuration management (~200 lines)
│
└── tests/                      # Unit tests
    ├── __init__.py
    ├── test_parser.py
    ├── test_analyzer.py
    ├── test_processor.py
    ├── test_ml.py
    └── test_database.py
```

**Estimated Total**: ~6,500 lines of application code + tests

### Module Dependencies

```
                    ┌─────────────┐
                    │   app.py    │
                    │ (entry)     │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │  gui/pages  │ │   config    │ │  database   │
    │  (5 pages)  │ │             │ │  /manager   │
    └──────┬──────┘ └─────────────┘ └──────┬──────┘
           │                               │
    ┌──────▼──────┐                 ┌──────▼──────┐
    │ gui/widgets │                 │  database   │
    │ (3 widgets) │                 │  /models    │
    └──────┬──────┘                 └─────────────┘
           │
    ┌──────▼───────────────────────────────┐
    │              core/                    │
    │  parser → analyzer → processor       │
    └──────┬───────────────────────────────┘
           │
    ┌──────▼──────┐
    │    ml/      │
    │ threshold   │
    │ drift       │
    └─────────────┘
```

---

## Database Schema (Reuse from v2)

The v2 database schema is well-designed and production-ready. We will reuse these tables:

### Core Tables
| Table | Purpose | Rows |
|-------|---------|------|
| `analysis_results` | File-level results | ~70,000 expected |
| `track_results` | Track-level data | ~100,000 expected |
| `processed_files` | Incremental processing tracking | ~70,000 expected |
| `ml_predictions` | ML model outputs | ~70,000 expected |
| `qa_alerts` | Quality alerts | Variable |
| `batch_info` | Batch metadata | ~1,000 expected |

### Key Fields for Analysis
- `sigma_gradient` - Primary quality metric
- `sigma_threshold` - Pass/fail threshold (ML-learned)
- `sigma_pass` - Boolean pass/fail
- `failure_probability` - ML-predicted risk
- `risk_category` - High/Medium/Low
- `drift_detected` - Process drift flag

---

## ML Strategy

### Models to Keep

#### 1. ThresholdOptimizer (RandomForestRegressor)
**Purpose**: Learn optimal sigma thresholds per model type
**Input Features**:
- Model number
- Unit length
- Linearity spec
- Historical pass rate
**Output**: Recommended threshold value
**Training**: Requires ~100+ samples per model

#### 2. DriftDetector (Statistical + IsolationForest)
**Purpose**: Detect when process is shifting over time
**Methods**:
- CUSUM statistical control
- Moving average trend detection
- IsolationForest anomaly detection
**Output**: Drift alerts with direction and magnitude

### Removed ML (Simplification)
- **FailurePredictor**: Merged into DriftDetector (similar purpose)
- **MLManager**: Over-abstraction, not needed
- **MLEngine**: Simplified into individual model files

### ML Integration Points
1. **During Processing**: Calculate sigma → check against ML threshold
2. **After Batch**: Run drift detection on model groups
3. **On Dashboard**: Show ML health, alert counts

---

## GUI Design

### 5 Pages (down from 11)

#### 1. Dashboard (Home)
**Purpose**: At-a-glance health overview
**Components**:
- Health score card (overall pass rate)
- Recent alerts (top 5)
- Processing stats (files today, this week)
- Quick action buttons (Process New, View Trends)

#### 2. Process
**Purpose**: Import and process files
**Components**:
- File drop zone (drag & drop)
- Folder selector (batch processing)
- Incremental mode toggle
- Progress bar with file count
- Recent results summary

#### 3. Analyze
**Purpose**: View and compare results
**Sub-modes**:
- **Single File**: Detailed analysis with charts
- **Track Compare**: Side-by-side TA vs TB
- **Final Line**: Compare to final test data
**Components**:
- File selector
- Results table
- Analysis chart
- Export button

#### 4. Trends
**Purpose**: Historical analysis and ML insights
**Components**:
- Model selector dropdown
- Date range picker
- SPC control chart (I/MR)
- Trend statistics
- Drift alerts panel
- Threshold recommendations

#### 5. Settings
**Purpose**: Configuration
**Components**:
- Database path selector
- Export location
- ML training trigger
- Theme toggle (dark/light)

### Single Chart Widget
Replace the 5-file, 3,000+ line chart system with one clean widget:

```python
class SimpleChart:
    """Single chart widget for all visualizations."""

    def plot_control(self, data, value_col, date_col, ucl, lcl, spec_limit):
        """SPC control chart with limits."""

    def plot_trend(self, data, value_col, date_col):
        """Simple trend line with optional regression."""

    def plot_histogram(self, data, value_col, bins=30):
        """Distribution histogram."""

    def plot_bar(self, data, category_col, value_col):
        """Bar chart for comparisons."""

    def show_placeholder(self, message):
        """Show message when no data."""

    def show_error(self, title, message):
        """Show error state."""
```

---

## Implementation Phases

### Phase 1: Foundation (Days 1-3)
**Goal**: Core infrastructure working

- [ ] Create folder structure
- [ ] Set up `__main__.py` and `app.py`
- [ ] Port database models (copy from v2)
- [ ] Create simplified `config.py`
- [ ] Set up basic main window shell

**Deliverable**: App launches, connects to database

### Phase 2: Core Processing (Days 4-7)
**Goal**: Can process files

- [ ] Build `parser.py` (extract from v2 excel_utils)
- [ ] Build `analyzer.py` (extract from v2 sigma_analyzer)
- [ ] Build `processor.py` (simplified from v2 unified_processor)
- [ ] Build `models.py` (simplified from v2 core/models)
- [ ] Implement incremental processing

**Deliverable**: Can process single file from CLI

### Phase 3: ML Integration (Days 8-10)
**Goal**: ML models working

- [ ] Port `threshold.py` (from ThresholdOptimizer)
- [ ] Port `drift.py` (from DriftDetector)
- [ ] Wire ML to processor
- [ ] Implement ML fallback (formula when ML unavailable)

**Deliverable**: Processing includes ML predictions

### Phase 4: GUI - Pages (Days 11-16)
**Goal**: All 5 pages functional

- [ ] Build `chart.py` widget
- [ ] Build `cards.py` and `table.py` widgets
- [ ] Build Dashboard page
- [ ] Build Process page
- [ ] Build Analyze page
- [ ] Build Trends page
- [ ] Build Settings page

**Deliverable**: Full GUI working

### Phase 5: Export & Polish (Days 17-19)
**Goal**: Export working, UI polished

- [ ] Build Excel export
- [ ] Add proper error handling
- [ ] Add loading states
- [ ] Polish layouts
- [ ] Add tooltips/help text

**Deliverable**: Production-ready UI

### Phase 6: Testing & Documentation (Days 20-22)
**Goal**: Tested and documented

- [ ] Write unit tests (80% coverage target)
- [ ] Manual testing with real files
- [ ] Write user documentation
- [ ] Update CHANGELOG

**Deliverable**: Release candidate

---

## Progress Tracking

### Session Protocol

**Every session MUST**:
1. Read this plan file first
2. Check current phase status below
3. Update progress after completing tasks
4. Document any decisions/changes

### Current Status

```
Phase 1: Foundation      [=====] COMPLETE
  - [x] Folder structure created
  - [x] __init__.py files created
  - [x] config.py created (simplified, ~200 lines)
  - [x] __main__.py entry point created
  - [x] app.py main window (~180 lines)
  - [x] All 5 GUI pages created (placeholder implementations):
        - dashboard.py (~120 lines)
        - process.py (~150 lines)
        - analyze.py (~130 lines)
        - trends.py (~170 lines)
        - settings.py (~190 lines)
  - [x] database/models.py (ported from v2, ~950 lines)
  - [x] core/models.py (simplified Pydantic models, ~270 lines)

Phase 2: Core Processing [=====] COMPLETE
  - [x] utils/constants.py (~80 lines) - shared constants
  - [x] core/parser.py (~380 lines) - Excel file parsing, System A/B detection
  - [x] core/analyzer.py (~350 lines) - Sigma, linearity, risk analysis
  - [x] core/processor.py (~470 lines) - Memory-safe unified processor
  - [x] database/manager.py (~620 lines) - Simplified database operations

  Memory-safe design for 8GB systems:
  - Limits concurrent workers based on available RAM
  - Uses generators to avoid memory accumulation
  - Explicit GC between batches
  - Falls back to sequential if memory critical

Phase 3: ML Integration  [=====] COMPLETE
  - [x] ml/threshold.py (~310 lines) - RandomForest threshold optimization
  - [x] ml/drift.py (~340 lines) - Hybrid drift detection (CUSUM + EWMA + IsolationForest)
  - [x] Wired ML to analyzer with automatic formula fallback
  - [x] Processor loads ML models on startup (if available)

  ML Approach:
  - ThresholdOptimizer: RandomForest for model-specific thresholds
  - DriftDetector: CUSUM (gradual shifts) + EWMA (trends) + IsolationForest (anomalies)
  - Automatic fallback to formula-based calculation when ML unavailable

Phase 4: GUI Pages       [=====] COMPLETE
Phase 5: Export & Polish [ ] Not Started
Phase 6: Testing & Docs  [ ] Not Started
```

### Session Log

| Date | Session | Work Done | Next Steps |
|------|---------|-----------|------------|
| 2025-12-14 | Planning | Created this plan | Start Phase 1 |
| 2025-12-15 | Phase 1 Start | Created folder structure, __init__.py files, config.py, __main__.py. Updated CLAUDE.md to clearly indicate v3 is in separate folder. | Continue with app.py, port database models |
| 2025-12-15 | Phase 1 Cont. | Created app.py with sidebar navigation and 5 placeholder pages (Dashboard, Process, Analyze, Trends, Settings). App structure is complete - launches with placeholder UI. Total v3 code: ~940 lines across 12 files. | Port database models, then implement core processing |
| 2025-12-15 | Phase 1 Complete | Ported database models from v2 (well-designed, reused as-is). Created simplified Pydantic models (~270 lines vs v2's 600+). Phase 1 COMPLETE. Total v3 code: ~2,200 lines across 15 files. | Start Phase 2: Core Processing |
| 2025-12-15 | Phase 2 Complete | Created core processing modules: parser.py (~380 lines), analyzer.py (~350 lines), processor.py (~470 lines with memory-safe design), database/manager.py (~620 lines). Total Phase 2: ~1,900 lines. Total v3 code: ~4,100 lines across 19 files. | Start Phase 3: ML Integration |
| 2025-12-15 | Phase 3 Complete | Created ML modules: threshold.py (~310 lines), drift.py (~340 lines). Wired ML to analyzer/processor with automatic fallback. Total Phase 3: ~650 lines. Total v3 code: ~4,750 lines across 21 files. | Phase 4: Wire GUI pages to processing |
| 2025-12-15 | Phase 4 Complete | Wired all GUI pages to real processing pipeline: chart.py (~400 lines), process.py rewritten (~375 lines), analyze.py rewritten (~555 lines), dashboard.py rewritten (~430 lines), trends.py rewritten (~480 lines). Added database methods for dashboard stats, alerts, model stats, and trend data. Phase 4 COMPLETE. Total v3 code: ~7,000 lines across 22 files. | Phase 5: Export functionality and polish |

---

## Migration Path

### For Users
1. v3 will be in separate folder - v2 remains untouched
2. v3 can read existing v2 database (same schema)
3. If v3 has issues, users can continue using v2
4. After v3 is stable, v2 folder can be archived

### For Development
1. Don't modify v2 code during v3 development
2. Copy/port code, don't move it
3. Test v3 independently from v2
4. Only after v3 is complete, consider archiving v2

---

## Success Criteria

### v3 is complete when:
- [ ] All 5 pages functional
- [ ] Can process 1000+ files without errors
- [ ] Incremental mode works correctly
- [ ] ML predictions populate
- [ ] Charts render without errors
- [ ] Excel export includes all key fields
- [ ] 80%+ test coverage
- [ ] Documentation complete
- [ ] No known bugs

### v3 is better than v2 when:
- [ ] Faster to process large batches
- [ ] Easier to understand code (smaller files)
- [ ] Fewer UI bugs
- [ ] Clearer error messages
- [ ] More consistent results

---

## Questions to Resolve

1. **Database location**: Keep same paths as v2 or allow configuration?
   - Recommendation: Same default, but configurable in Settings

2. **Chart library**: Keep matplotlib or try something lighter?
   - Recommendation: Keep matplotlib (proven, no new dependencies)

3. **Incremental default**: On or off by default?
   - Recommendation: On by default (90% use case)

4. **ML training**: Automatic or manual trigger?
   - Recommendation: Manual trigger in Settings (clear user control)

---

## Appendix A: What Works Well in v2 (PRESERVE)

After reviewing the current single file analysis implementation, these components work well and should be preserved in v3:

### 1. Single File Analysis Flow (single_file_page.py)
**What works:**
- File browser with last directory memory
- Pre-validation option (checks file readability before processing)
- Options: Generate Plots, Save to Database, Comprehensive Validation
- Clear progress dialog with status updates
- Emergency reset button for frozen states
- Thread-based processing (non-blocking UI)

**Keep in v3:**
- Same workflow: Browse → Pre-validate (optional) → Analyze → View Results → Export
- Settings persistence (last directory, preferences)
- Background processing with progress feedback

### 2. Sigma Analysis Algorithm (sigma_analyzer.py)
**What works:**
- MATLAB-compatible gradient calculation
- Butterworth filter application
- Endpoint removal (configurable)
- ML-threshold lookup (formula fallback if ML unavailable)
- Comprehensive logging for debugging
- Proper handling of edge cases (division by zero, NaN values)

**Key algorithm (preserve exactly):**
```python
# Calculate gradients with step size
for i in range(len(positions) - step_size):
    dx = positions[i + step_size] - positions[i]
    dy = errors[i + step_size] - errors[i]
    if abs(dx) > 1e-6:
        gradient = dy / dx
        gradients.append(gradient)

# Sigma is standard deviation of gradients
sigma_gradient = np.std(gradients, ddof=1)
```

### 3. Linearity Analysis with Offset Adjustment (linearity_analyzer.py)
**What works:**
- Optimal offset calculation using multiple methods:
  1. Median of differences from band center
  2. Scipy optimization to minimize limit violations
  3. Simple centering fallback
- Zero-tolerance fail point counting (any point outside limits = fail)
- NaN-safe limit interpolation

**Key feature (preserve exactly):**
```python
# Calculate optimal offset to center errors within limits
def _calculate_optimal_offset(errors, upper_limits, lower_limits):
    # For each point, calculate how far error is from band center
    for i in range(len(errors)):
        midpoint = (upper_limits[i] + lower_limits[i]) / 2
        differences.append(midpoint - errors[i])

    median_offset = np.median(differences)

    # Optimize to minimize violations
    result = optimize.minimize_scalar(
        violation_count,
        bounds=(median_offset - search_range, median_offset + search_range),
        method='bounded'
    )
```

### 4. Error vs Position Plot (plotting_utils.py)
**What works:**
- Multi-panel layout: Main plot + Range analysis + Metrics + Status
- Untrimmed data shown as dashed blue line
- Trimmed data with offset applied shown as solid green
- Varying spec limits plotted correctly (not just horizontal lines)
- Fail points highlighted with red X markers
- Auto-scaling with 5% padding
- Professional tick formatting based on data scale
- Offset value shown in plot title

**Chart features to preserve:**
```
┌─────────────────────────────────────────────────┐
│  Error vs Position Analysis (Offset: 0.000123V) │
│  ┌──────────────────────────────────────────┐   │
│  │  -- Untrimmed Data (blue dashed)         │   │
│  │  —— Trimmed Data (green solid)           │   │
│  │  -- Upper/Lower Spec Limits (red dashed) │   │
│  │  X  Fail Points (if any)                 │   │
│  │  [Shaded spec limit area]                │   │
│  └──────────────────────────────────────────┘   │
├─────────────────┬──────────────┬────────────────┤
│ Error Range     │ Metrics      │ Status         │
│ (by segment)    │ Summary      │ PASS/FAIL      │
└─────────────────┴──────────────┴────────────────┘
```

### 5. Analysis Display Widget (analysis_display.py)
**What works:**
- MetricCards for key values (color-coded by status)
- Track selector dropdown (for multi-track files)
- Clear pass/fail indicators
- Validation status and grade display

**Cards to preserve:**
- Overall Status (PASS/FAIL with color)
- Sigma Gradient (value + threshold)
- Sigma Pass (Yes/No)
- Linearity Error (raw and shifted)
- Linearity Pass (Yes/No)
- Processing Time

### 6. Theme-Aware Colors (plotting_utils.py)
**Color scheme to preserve:**
```python
QA_COLORS = {
    'pass': '#27ae60',      # Green
    'fail': '#e74c3c',      # Red
    'warning': '#f39c12',   # Orange
    'info': '#3498db',      # Blue
    'untrimmed': '#3498db', # Blue
    'trimmed': '#27ae60',   # Green
    'spec_limit': '#e74c3c',# Red
    'threshold': '#f39c12', # Orange
}
```

---

## Appendix B: Charts to Improve/Remove

### Current Charts (v2 has too many)
1. **Error vs Position** ✅ KEEP - core visualization
2. **Error Range Analysis** ✅ KEEP - useful for segment analysis
3. **Metrics Summary** ✅ KEEP - text summary panel
4. **Status Display** ✅ KEEP - pass/fail indicator
5. **SPC Control Chart** ✅ KEEP - industry standard for trends
6. **Histogram** ⚠️ SIMPLIFY - only show when useful
7. **Pareto** ❌ REMOVE - rarely used
8. **Correlation** ❌ REMOVE - rarely used
9. **PCA** ❌ REMOVE - over-engineered
10. **Anomaly Detection** ⚠️ MERGE into drift detection

### v3 Chart Widget - Simplified
```python
class ChartWidget:
    """Simplified chart widget for v3."""

    # Core plots (always available)
    def plot_error_vs_position(self, track_data):
        """Main analysis plot with offset adjustment."""

    def plot_spc_control(self, data, model):
        """I-MR control chart for trends."""

    # Optional plots (shown when relevant)
    def plot_track_comparison(self, track_a, track_b):
        """Side-by-side track comparison."""

    def plot_histogram(self, values, title):
        """Simple distribution view."""

    # Utilities
    def clear(self):
        """Clear the chart."""

    def show_placeholder(self, message):
        """Show empty state."""
```

---

## Appendix C: Files to Port from v2

### Direct Copy (minimal changes)
- `database/models.py` → `database/models.py`

### Extract & Simplify
- `core/models.py` → `core/models.py` (remove unused fields)
- `analysis/sigma_analyzer.py` → `core/analyzer.py` (preserve algorithm)
- `analysis/linearity_analyzer.py` → `core/analyzer.py` (preserve offset calculation)
- `ml/models.py` (ThresholdOptimizer, DriftDetector) → `ml/threshold.py`, `ml/drift.py`
- `utils/excel_utils.py` (core parsing) → `core/parser.py`
- `utils/plotting_utils.py` (error vs position plot) → `gui/widgets/chart.py`

### Rewrite (new, simpler)
- All GUI pages (new, simpler structure)
- Processor (unified, simplified)
- Config (simplified)

### Don't Port (Remove)
- processor.py (deprecated)
- fast_processor.py (deprecated)
- large_scale_processor.py (deprecated)
- cached_processor.py (not needed)
- All mixin files (simplify into pages)
- analytics_engine.py (dead code)
- Multiple chart files (consolidate into one)
- API client (not used)
- CLI commands (focus on GUI)
