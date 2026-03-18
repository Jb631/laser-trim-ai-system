# Laser Trim Analyzer V4 Upgrade Plan

**Created:** 2026-03-17
**Author:** James Beresford
**Purpose:** Transform the app from a measurement recording tool into an operational root cause identification and cost impact analysis platform.

---

## Background

Analysis of the current database (81,328 records, 311 models) revealed several issues that limit the app's value:

- **Data contamination**: ~33% of records are from legacy/inactive models or non-laser-trim files (output smoothness, scrap investigations, noise tests)
- **No cost visibility**: Failures aren't connected to dollar impact or throughput loss
- **Trends page noise**: Too much data without focus — hard to identify where to pay attention
- **Dashboard charts are hard to read**: Pareto chart has confusing dual-axis layout, P-chart is crammed into tiny space, model focus text is a raw data dump with no visual hierarchy
- **Near-miss blindness**: The app treats all failures equally, but ~46% of failures have only 1-3 points out of spec (fixable with small process improvements)
- **Active model management is basic**: MPS list exists but doesn't include pricing, and can't be imported from a spreadsheet

Key data findings from the analysis:
- 59.2% overall pass rate at laser trim, 57.6% at Final Test
- ~46% of linearity failures are near-misses (1-3 fail points)
- Failure rate improving: 21% in 2023 down to 14% in early 2026
- Monday failure rate elevated (20.3% vs ~16% other days)
- Estimated scrap cost: $107K+/year (conservative, limited pricing data)

---

## Design Principles

1. **Clean data first** — No analysis is trustworthy if the data is contaminated
2. **Fix existing code** — Don't create unnecessary new files or over-engineer
3. **Every feature must work end-to-end** — No partial implementations
4. **Explain the "why"** — Code changes should be educational (James is learning Python)
5. **Ops-level focus** — Features should answer "what's it costing us?" and "where do we focus?"
6. **Respect existing architecture** — Build on what's already there (active models, FT matching, ML system)

---

## Phase 1: Data Foundation

**Goal:** Make the data trustworthy so every analysis built on top of it is credible.

### 1.1 Parser File Type Filtering

**File:** `src/laser_trim_analyzer/core/parser.py`

**Problem:** `detect_file_type()` only distinguishes "trim" vs "final_test". Output smoothness, scrap investigation, and noise/vibration files are silently classified as trim data.

**Solution:** Add filename and sheet-structure checks to identify and reject (or flag) non-laser-trim files. Detection patterns:
- "Scrap" in filename → reject or flag as `scrap_investigation`
- "Noise" or "noise" in filename → reject or flag as `noise_test`
- "Smoothness" or "Output Smooth" in filename or sheet names → reject or flag as `output_smoothness`
- Sheet names that don't match System A (SEC1 TRK) or System B (test/Lin Error) patterns → reject

**Behavior:** Return a new file type (e.g., `non_trim`) that the processor skips with a log message. Don't silently process these files.

**Testing:** Process a folder containing a mix of trim, final test, scrap, and smoothness files. Verify only trim and final test files create database records.

### 1.2 Database Cleanup Capability

**File:** `src/laser_trim_analyzer/gui/pages/settings.py`, `src/laser_trim_analyzer/database/manager.py`

**Problem:** No way to remove legacy/contaminated records without direct SQL access. 27K+ legacy records are polluting analyses.

**Solution:** Add to Settings page:
- "Database Cleanup" section with options:
  - Delete all records for models NOT in the active MPS list
  - Delete all records older than a specified date
  - Delete records identified as non-trim files (if flagged during Phase 1.1)
  - Preview mode: show what WOULD be deleted before committing
- Confirmation dialog with record count before deletion
- Backup reminder before cleanup

**Important:** This should be a deliberate manual action, not automatic. The preview is critical — James needs to verify before deleting.

### 1.3 Database Indexing

**File:** `src/laser_trim_analyzer/database/manager.py` (migration on startup)

**Problem:** No indexes on frequently queried columns. Every page load queries the full table.

**Solution:** Add indexes on startup migration:
- `analysis_results`: index on `model`, `file_date`, `overall_status`
- `track_results`: index on `analysis_id`, `linearity_pass`
- `final_test_results`: index on `model`, `linked_trim_id`

**Expected impact:** Significant speed improvement on Analyze, Trends, and Dashboard pages.

### 1.4 Ingest Validation

**File:** `src/laser_trim_analyzer/core/processor.py`

**Problem:** Processor accepts any result from the parser without checking if the numbers make sense.

**Solution:** Add post-parse validation checks:
- Sigma gradient < 0 → flag as data error
- All-zero error data → flag as parse failure (element wasn't measured)
- Position array length < 10 → flag as incomplete data
- Error data length doesn't match position data length → flag as parse error

**Note on duplicate serials:** Same serial number appearing multiple times is VALID — this happens when a unit goes through the laser for additional trim passes. These should NOT be rejected. They should be kept as separate records with sequential timestamps.

**Behavior:** Flag suspect records with a new `data_quality` field rather than rejecting them. This preserves the data while letting downstream analyses filter it out.

---

## Phase 1.5: Dashboard & Chart Fixes

**Goal:** Make the dashboard charts readable and useful. Charts that don't make sense undermine trust in the whole app.

### 1.5.1 Fix Pareto Chart Layout

**File:** `src/laser_trim_analyzer/gui/widgets/chart.py` — `plot_pareto()`

**Problems:**
- Cumulative % line is on a secondary X-axis at the TOP (`ax.twiny()`), which is confusing on a horizontal bar chart
- Standard Pareto charts are vertical (bars going up, cumulative line overlaid)
- Value labels overlap when there are many models

**Fix:** Rewrite `plot_pareto()` as a vertical bar chart (standard Pareto orientation):
- Vertical bars: models on X-axis, failure count on left Y-axis
- Cumulative % line on right Y-axis (`ax.twinx()`) — this is the standard layout
- Rotate X-axis model labels 45 degrees for readability
- Limit to top 10 models (not 15) for readability
- Add 80% cumulative line as horizontal dashed line
- Color bars by severity: darkest red for #1, gradient to orange for lower ranks

### 1.5.2 Improve Dashboard Layout

**File:** `src/laser_trim_analyzer/gui/pages/dashboard.py`

**Problems:**
- P-chart trend is crammed into center column of row 2 at `figure_size=(4, 2.5)` — too small to read
- "Where to Focus" text panel is a raw textbox dump competing with the Pareto chart
- 3-column grid forces charts into narrow panels

**Fix:** Rework the dashboard layout:
- Row 0: Three metric cards (Linearity Quality, Database Total, Sigma Health) — keep as-is, these work
- Row 1: System comparison info — keep as-is
- Row 2: P-chart trend expanded to full width (3 columns). Increase figure size to `(8, 3)` minimum
- Row 3: Pareto chart (2 columns) + "Where to Focus" panel (1 column)
- Move alerts and drift status into a collapsible or scrollable section, or into a dedicated alerts row above the charts

### 1.5.3 Fix P-Chart Readability

**File:** `src/laser_trim_analyzer/gui/widgets/chart.py` — `plot_pchart()`

**Problems:**
- At `(4, 2.5)` figure size, date labels overlap and are unreadable
- Control limit lines are thin and hard to distinguish from data line
- Out-of-control markers (small X) are hard to spot

**Fix:**
- Increase minimum figure size for P-chart (handled by layout fix in 1.5.2)
- Make control limit lines more visible: increase linewidth to 1.5, use distinct color from data line
- Make out-of-control markers larger and more prominent (red circles with border, size 120+)
- Show fewer date labels (max 8) with clearer formatting (month-day only, no year)
- Add a subtle fill between the data line and the center line to show deviation visually

### 1.5.4 Clean Up "Where to Focus" Panel

**File:** `src/laser_trim_analyzer/gui/pages/dashboard.py` — `_update_model_display()`

**Problem:** The model focus text is a monospace dump that looks like a terminal log. Hard to scan, no visual priority.

**Fix:** Rework as a structured list with visual hierarchy:
- Each model gets a mini card or formatted block: model name (bold), pass rate (color-coded), failure count, recommendation
- Top 5 only (not 10) — focus attention
- Color-code each entry: red for high-impact, orange for medium, green for improving
- Add a "DECLINING" tag that's visually prominent (red background or icon), not just text
- Consider replacing the textbox with actual CTkFrame widgets stacked vertically for better formatting

---

## Phase 2: Operational Analytics

**Goal:** Add the analyses that answer "why are units failing?" and "what's it costing us?"

### 2.1 Model Management with Pricing

**Files:** `src/laser_trim_analyzer/config.py`, `src/laser_trim_analyzer/gui/pages/settings.py`, `src/laser_trim_analyzer/database/models.py`

**Problem:** Active model list exists but doesn't include unit pricing. Pricing is needed for cost impact calculations. Currently requires manual entry one model at a time.

**Solution:** Enhance the Settings active models section:
- Add a "Unit Price" column next to each model in the MPS list
- Add "Import from Excel/CSV" button that reads a spreadsheet with Model and Price columns (like the backlog export James already has)
- Allow filtering/pasting from a backlog spreadsheet
- Store pricing in config or a new `model_metadata` database table
- Make pricing available to Dashboard and Trends pages for cost calculations

**UX flow:** User exports backlog from ERP → opens in Excel → filters to production models → copies Model + Price columns → pastes or imports into app → saves

### 2.2 Near-Miss Analysis View

**Files:** `src/laser_trim_analyzer/gui/pages/dashboard.py` or new section in `analyze.py`

**Problem:** All failures are treated equally. A unit that misses spec by 0.01% at one point is reported the same as a unit that's wildly out of spec at 50 points.

**Solution:** Add a near-miss analysis section showing:
- Distribution histogram: how many fail points per failed track (1-3, 4-10, 11-50, 50+)
- Percentage of failures that are "near-miss" (1-3 fail points) vs "hard fail" (10+ points)
- Per-model near-miss rates — which models have the most recoverable failures?
- Sigma gradient comparison: near-miss tracks vs hard-fail tracks vs passing tracks

**Why this matters:** If 46% of failures are near-misses, that's the strongest argument for process improvement investment. Small gains = big recovery.

**Note:** The zero-tolerance linearity spec is a customer requirement and cannot change. The goal is to identify which units are close enough that upstream process improvements (better elements, tighter environmental controls) would push them over the line.

### 2.3 Cost Impact Dashboard

**Files:** `src/laser_trim_analyzer/gui/pages/dashboard.py`

**Problem:** Dashboard shows pass/fail rates but doesn't connect failures to dollars or throughput.

**Solution:** Add cost impact section to Dashboard (requires pricing from 2.1):
- **Pareto chart by cost**: Models ranked by (failure_count × unit_price × cost_ratio). Top 5-10 models labeled.
- **Monthly scrap cost trend**: Estimated cost of failures per month over time
- **Recovery opportunity**: "If top 5 models improved to X% failure rate, estimated monthly recovery = $Y"
- **Throughput impact**: Estimated hours lost to failed units (if we can estimate processing time per unit)

Cost ratio default: 50% of selling price (configurable in Settings). This is a conservative estimate of labor + material invested by the time a unit reaches final test.

### 2.4 Temporal Pattern Analysis

**Files:** `src/laser_trim_analyzer/gui/pages/trends.py`

**Problem:** Trends page shows per-model SPC charts but doesn't surface time-based patterns across all models.

**Solution:** Add to Trends page "All Models" summary view:
- **Day-of-week chart**: Bar chart showing failure rate by day (Monday-Friday/Saturday)
- **Monthly trend**: Failure rate by calendar month (shows seasonal patterns)
- **Year-over-year comparison**: Same months across different years

**Noise reduction for Trends page:**
- Default to showing only MPS/active models (use existing active models config)
- Add "minimum sample size" filter — don't show models with <20 records in the selected time range
- Add severity filter — show only models above a failure rate threshold (e.g., >10%)
- Sort models by "needs attention" score (high failure rate + high volume + trending worse)

### 2.5 Failure Margin Tracking

**Files:** `src/laser_trim_analyzer/core/analyzer.py`, `src/laser_trim_analyzer/database/models.py`

**Problem:** The app records pass/fail and fail point count, but not HOW FAR out of spec failed points were.

**Solution:** During analysis, calculate and store:
- `max_violation`: The maximum amount any single point exceeded its spec limit (absolute value)
- `avg_violation`: Average violation across all fail points
- `margin_to_spec`: For passing tracks, how close the worst point was to the spec limit (positive = margin remaining)

These metrics let you distinguish:
- Comfortable pass (margin_to_spec > 20%) — healthy process
- Tight pass (margin_to_spec < 5%) — at risk, could fail with slight variation
- Near-miss fail (max_violation < X) — almost passed, recoverable
- Hard fail (max_violation >> spec) — fundamental issue

---

## Phase 3: Predictive Improvements

**Goal:** Make the ML system more useful and the Final Test correlation more reliable.

### 3.1 Final Test Matching Improvements

**File:** `src/laser_trim_analyzer/database/manager.py`

**Problem:** Only 34% of Final Test results are linked to trim data. The existing matching (model + serial + 60-day window) is correct in design but the low match rate suggests data quality issues — serial numbers may not be consistent between trim files and FT files.

**Solution:**
- Add diagnostic report: for unmatched FT records, show WHY they didn't match (no matching model? no matching serial? date out of range?)
- Add manual linking capability in Compare page: user can search and link records that automated matching missed
- Add fuzzy serial matching option: strip leading zeros, ignore case, handle common formatting differences
- Log match attempts so mismatches can be investigated

### 3.2 ML Retraining Triggers

**File:** `src/laser_trim_analyzer/ml/manager.py`

**Problem:** ML retraining is manual only. Models go stale as new data accumulates.

**Solution:** Add automatic retraining triggers:
- When a model accumulates 50+ new records since last training
- When drift is detected on a model
- When new Final Test data is linked (new ground truth)
- Add "staleness" indicator on Settings page showing days since each model was last trained

### 3.3 Process Capability (Cpk) Calculation

**File:** `src/laser_trim_analyzer/core/analyzer.py`

**Problem:** No standard SPC capability metrics. Cpk is the universal language for process capability in aerospace/AS9100.

**Solution:** Calculate per-model Cpk based on:
- Sigma gradient distribution vs threshold (USL)
- Linearity error distribution vs spec limits
- Display on Trends page detail view alongside SPC chart
- Color code: Cpk < 1.0 (red, incapable), 1.0-1.33 (yellow, marginal), >1.33 (green, capable)

---

## Phase 4: Operational Integration

**Goal:** Make the app a tool for operational decision-making, not just quality reporting.

### 4.1 Executive Summary Export

**File:** `src/laser_trim_analyzer/export/excel.py` (or new export module)

**Problem:** Export only dumps raw data to Excel. No formatted report for management.

**Solution:** One-click export that generates a formatted report including:
- Overall health summary (pass rates, trends, Cpk)
- Top 10 cost-impact models with Pareto chart
- Near-miss analysis summary
- Recommended focus areas
- Monthly trend charts

Format: Excel workbook with formatted sheets and embedded charts, or PDF.

### 4.2 Element Screening Recommendations

**Based on Phase 2 data:**
- Flag models where near-miss rate > 40% as candidates for in-process testing
- Flag models with batch-clustered failures as candidates for incoming element inspection
- Flag models with consistently high failure rates (>50%) as candidates for design review

This is a recommendations display, not automated action — the app surfaces the data, James and engineering make the decisions.

### 4.3 Incoming Element Quality Tracking

**Future consideration:** If Betatronix starts testing elements before assembly, the app could track incoming element quality and correlate it with final linearity outcomes. This would close the loop between element production and finished unit quality.

---

## Implementation Notes

### Session Protocol for Claude Code

Each coding session should:
1. Read this plan and `docs/UPGRADE_TRACKER.md` to check current phase and pending tasks
2. Work on the next incomplete task in the tracker
3. Test changes before marking complete
4. Update the tracker with progress and any notes
5. Commit with descriptive message referencing the phase/task number

### Dependencies Between Tasks

- Phase 2.1 (pricing) must be done before 2.3 (cost dashboard)
- Phase 1.1 (parser filtering) should be done before 1.2 (cleanup) so new files don't re-contaminate
- Phase 2.5 (failure margin) feeds into 2.2 (near-miss analysis)
- Phase 1.3 (indexes) can be done anytime and should be early for performance

### Suggested Order Within Phases

**Phase 1:** 1.3 (indexes, quick win) → 1.1 (parser) → 1.4 (validation) → 1.2 (cleanup)
**Phase 2:** 2.1 (pricing) → 2.5 (margin tracking) → 2.2 (near-miss) → 2.3 (cost dashboard) → 2.4 (temporal)
**Phase 3:** 3.1 (FT matching) → 3.3 (Cpk) → 3.2 (ML triggers)
**Phase 4:** 4.1 (export) → 4.2 (recommendations) → 4.3 (future)
