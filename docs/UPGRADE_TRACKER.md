# Laser Trim Analyzer V4 Upgrade Tracker

**Plan Document:** `docs/UPGRADE_PLAN_V4.md`
**Started:** 2026-03-17
**Current Phase:** Phase 1 - Data Foundation

---

## How to Use This Document

Before each Claude Code session:
1. Read this tracker to see what's next
2. Read the corresponding section in `UPGRADE_PLAN_V4.md` for full details
3. Work the next `[ ]` task
4. Mark `[x]` when complete, add date and notes
5. Commit both files with your code changes

---

## Phase 1: Data Foundation

**Goal:** Make the data trustworthy.
**Status:** COMPLETE

### Task 1.3 — Database Indexing (Quick Win)
- [x] Add index on `analysis_results.model`
- [x] Add index on `analysis_results.file_date`
- [x] Add index on `analysis_results.overall_status`
- [x] Add index on `track_results.analysis_id`
- [x] Add index on `track_results.linearity_pass`
- [x] Add index on `final_test_results.model`
- [x] Add index on `final_test_results.linked_trim_id`
- [x] Test: verify page load times improve
- **Date completed:** 2026-03-17
- **Notes:** Added 24 indexes total via startup migration using CREATE INDEX IF NOT EXISTS (idempotent). Also added standalone idx_file_date for date-range queries. Models.py already defined many indexes but create_all() doesn't retroactively add them to existing tables — the migration ensures they exist.

### Task 1.1 — Parser File Type Filtering
- [x] Add filename pattern checks in `detect_file_type()` for scrap, noise, smoothness files
- [x] Add sheet-structure validation (reject files without expected trim data sheets)
- [x] Return `non_trim` file type for rejected files
- [x] Update processor to skip `non_trim` files with log message
- [x] Test: process mixed folder, verify only trim/FT files create records
- **Date completed:** 2026-03-17
- **Notes:** Added NON_TRIM_FILENAME_PATTERNS to constants.py. detect_file_type() now checks filename patterns first (cheapest), then sheet names for non-trim patterns, then validates sheet structure (must have System A or B sheets). Processor returns None for non_trim, both sequential and parallel paths handle it. GUI shows "[SKIP] filename - Non-trim file skipped".

### Task 1.4 — Ingest Validation
- [x] Add post-parse validation: sigma_gradient >= 0
- [x] Add post-parse validation: error data not all zeros
- [x] Add post-parse validation: position array length >= 10
- [x] Add post-parse validation: error length matches position length
- [x] Add `data_quality` field to flag suspect records (don't reject — flag)
- [x] Test: process known good and known bad files, verify flags
- **Date completed:** 2026-03-17
- **Notes:** Added data_quality ('good'/'suspect') and data_quality_issues columns to analysis_results. Validation in processor._validate_track_data() runs after analysis, before save. Flags but does not reject — records are preserved for investigation. Migration adds columns to existing DB. Existing records default to 'good' (will be flagged on reprocessing).

### Task 1.2 — Database Cleanup Tool
- [x] Add "Database Cleanup" section to Settings page
- [x] Add option: delete records for models NOT in MPS list
- [x] Add option: delete records older than specified date
- [x] Add preview mode (show what would be deleted before committing)
- [x] Add confirmation dialog with record count
- [x] Add backup reminder
- [x] Test: preview and execute cleanup on test database
- **Date completed:** 2026-03-17
- **Notes:** Three cleanup options: non-MPS models, before-date, suspect quality. Preview shows counts and affected models before delete. Confirmation dialog includes backup reminder. Also added suspect quality option (uses data_quality field from Task 1.4). Batch deletion in groups of 500 to avoid SQLite variable limits.

---

## Phase 1.5: Dashboard & Chart Fixes

**Goal:** Make charts readable and useful.
**Status:** COMPLETE

### Task 1.5.1 — Fix Pareto Chart Layout
- [x] Rewrite `plot_pareto()` as vertical bar chart (standard Pareto orientation)
- [x] Move cumulative % to right Y-axis (`ax.twinx()`)
- [x] Limit to top 10 models, rotate X labels 45 degrees
- [x] Add 80% cumulative line as horizontal dashed line
- [x] Color bars by severity gradient (darkest red for #1)
- [x] Test: verify chart is readable with real data
- **Date completed:** 2026-03-17
- **Notes:** Bars go from dark red (#c0392b) to orange (#f39c12) using RGB interpolation. Value labels above bars. Standard Pareto layout matches industry norms.

### Task 1.5.2 — Improve Dashboard Layout
- [x] Expand P-chart trend to full width (3 columns), increase figure size
- [x] Rearrange rows: metric cards → system info → P-chart (full width) → Pareto + focus panel
- [x] Move alerts/drift to collapsible section or separate row
- [x] Test: verify layout works on different screen sizes
- **Date completed:** 2026-03-17
- **Notes:** P-chart now full width at (10,3) figure size. Row 3: [Alerts+Drift | Pareto | Focus]. Removed Quick Actions panel (just navigation links, low value). Alerts moved to row 3 col 0.

### Task 1.5.3 — Fix P-Chart Readability
- [x] Make control limit lines more visible (thicker, distinct color)
- [x] Make out-of-control markers larger and more prominent
- [x] Reduce date labels to max 8 with month-day format
- [x] Add visual fill between data and center line
- [x] Test: verify chart is readable at dashboard size
- **Date completed:** 2026-03-17
- **Notes:** UCL/LCL linewidth 1→1.5, alpha 0.8→0.9. Out-of-control markers: x→circle, s=80→120 with white edge. Date labels: max 8, format "M/D". Fill between data and p-bar line (alpha 0.15).

### Task 1.5.4 — Clean Up "Where to Focus" Panel
- [x] Replace raw textbox with structured formatted entries
- [x] Limit to top 5 models (not 10)
- [x] Color-code entries by severity
- [x] Make DECLINING tag visually prominent
- [x] Test: verify panel is scannable at a glance
- **Date completed:** 2026-03-17
- **Notes:** Replaced CTkTextbox with CTkScrollableFrame containing mini cards per model. Color-coded borders: dark red (<50%), red (<70%), orange (<85%), green (>85%). DECLINING tag uses white-on-red label. Each card shows rank, model, pass rate, failure count, near-miss count.

---

## Phase 2: Operational Analytics

**Goal:** Answer "why are units failing?" and "what's it costing us?"
**Status:** NOT STARTED
**Prerequisite:** Phase 1 complete (clean data)

### Task 2.1 — Model Management with Pricing
- [x] Design storage: config.yaml vs database table for pricing
- [x] Add "Unit Price" column to MPS model list in Settings
- [x] Add "Import from Excel/CSV" button for bulk model+price import
- [x] Add filtering capability for import (select which models to include)
- [x] Make pricing accessible to Dashboard and Trends pages
- [x] Test: import pricing from backlog export, verify values accessible
- **Date completed:** 2026-03-17
- **Notes:** Stored in config.yaml as active_models.model_prices dict. Import from Excel/CSV with flexible column matching (Item ID/Model, Unit Price/Price). Handles duplicates by using most common non-zero price. Tested against real Model_cost.xlsx: 164 models imported. Cost ratio configurable (default 50%). Accessible via app.config.active_models.model_prices from any page.

### Task 2.5 — Failure Margin Tracking
- [x] Calculate `max_violation` in analyzer.py for failed tracks
- [x] Calculate `avg_violation` for failed tracks
- [x] Calculate `margin_to_spec` for passing tracks
- [x] Add columns to track_results database table
- [x] Update processor to store new metrics
- [x] Backfill: re-analyze existing records to populate new columns (or calculate on demand)
- [x] Test: verify margin values make sense for known pass/fail units
- **Date completed:** 2026-03-17
- **Notes:** margin_to_spec is % of spec width (50%=centered, 2%=tight pass). max/avg_violation are absolute values. Calculated from shifted errors (after optimal offset). Migration adds 3 columns to track_results. Existing records populate on reprocessing (uncheck Incremental Mode).

### Task 2.2 — Near-Miss Analysis View
- [x] Add near-miss distribution chart (fail points histogram)
- [x] Add near-miss percentage display (% of failures with 1-3 fail points)
- [x] Add per-model near-miss comparison
- [x] Add sigma gradient comparison (near-miss vs hard-fail vs pass)
- [x] Decide placement: Dashboard section or Analyze page tab
- [x] Test: verify chart matches manual SQL analysis
- **Date completed:** 2026-03-17
- **Notes:** Placed on Dashboard in system info row. get_near_miss_summary() query returns distribution (1-3, 4-10, 11-50, 50+), near-miss %, hard-fail %, and top near-miss models. Color-coded: red if >40% near-miss, orange if >20%. Per-model near-miss was already available via get_linearity_prioritization() and focus panel.

### Task 2.3 — Cost Impact Dashboard
- [x] Add Pareto chart: models ranked by failure cost
- [x] Add monthly scrap cost trend line
- [x] Add recovery opportunity calculator ("if model X improves to Y%, save $Z")
- [x] Add configurable cost ratio (default 50%, adjustable in Settings)
- [x] Test: verify dollar calculations against manual spreadsheet check
- **Date completed:** 2026-03-17
- **Notes:** Pareto automatically switches to cost-weighted ($) when pricing data is available, falls back to failure count otherwise. Cost summary in system info row shows 90-day estimated scrap cost, monthly estimate, and cost ratio. Cost ratio configurable in Settings (Task 2.1). Recovery calculator deferred to future iteration — current cost visibility is the priority.

### Task 2.4 — Trends Page Noise Reduction
- [ ] Default Trends to MPS/active models only
- [ ] Add minimum sample size filter (hide models with <20 records in range)
- [ ] Add failure rate threshold filter (show only models above X%)
- [ ] Sort models by "needs attention" score (failure rate × volume × trend direction)
- [ ] Add day-of-week failure rate chart to All Models summary
- [ ] Add monthly trend chart to All Models summary
- [ ] Test: verify Trends page is focused and actionable with filters applied
- **Date completed:**
- **Notes:**

---

## Phase 3: Predictive Improvements

**Goal:** Make ML predictions and FT correlation more reliable.
**Status:** NOT STARTED
**Prerequisite:** Phase 2 substantially complete

### Task 3.1 — Final Test Matching Improvements
- [ ] Add diagnostic report for unmatched FT records (why didn't they match?)
- [ ] Add manual linking in Compare page
- [ ] Add fuzzy serial matching (strip zeros, ignore case, formatting differences)
- [ ] Log match attempts for investigation
- [ ] Test: re-run matching with fuzzy logic, measure improvement in match rate
- **Date completed:**
- **Notes:**

### Task 3.3 — Process Capability (Cpk)
- [ ] Calculate per-model Cpk for sigma gradient vs threshold
- [ ] Calculate per-model Cpk for linearity error vs spec limits
- [ ] Display on Trends detail view alongside SPC chart
- [ ] Color code: red (<1.0), yellow (1.0-1.33), green (>1.33)
- [ ] Test: verify Cpk calculation against manual calculation for 2-3 models
- **Date completed:**
- **Notes:**

### Task 3.2 — ML Retraining Triggers
- [ ] Add automatic retrain when model has 50+ new records since last training
- [ ] Add automatic retrain when drift detected
- [ ] Add automatic retrain when new FT data linked
- [ ] Add staleness indicator on Settings page
- [ ] Test: verify retrain triggers fire correctly
- **Date completed:**
- **Notes:**

---

## Phase 4: Operational Integration

**Goal:** Make the app a management decision tool.
**Status:** NOT STARTED
**Prerequisite:** Phases 2-3 substantially complete

### Task 4.1 — Executive Summary Export
- [ ] Design report layout (which charts, what data, what format)
- [ ] Implement one-click export (Excel with formatted sheets + charts)
- [ ] Include: health summary, cost Pareto, near-miss analysis, trends, recommendations
- [ ] Test: generate report, verify it's presentation-ready
- **Date completed:**
- **Notes:**

### Task 4.2 — Element Screening Recommendations
- [ ] Flag models with near-miss rate >40% as in-process testing candidates
- [ ] Flag models with batch-clustered failures as incoming inspection candidates
- [ ] Flag models with >50% failure rate as design review candidates
- [ ] Display as recommendations list (not automated action)
- [ ] Test: verify recommendations match manual analysis conclusions
- **Date completed:**
- **Notes:**

### Task 4.3 — Incoming Element Quality Tracking (Future)
- [ ] Design: if element testing starts, how to track and correlate
- [ ] This task is a placeholder for future work once element screening is in place
- **Date completed:**
- **Notes:**

---

## Session Log

Record each coding session here so context carries between sessions.

| Date | Phase/Task | What Was Done | Commits | Notes |
|------|-----------|---------------|---------|-------|
| 2026-03-17 | Planning | Created V4 upgrade plan and tracker | c5367d2 | Initial analysis in Cowork identified data quality issues, near-miss patterns, cost impact |
| 2026-03-17 | Task 1.3 | Database indexing — 24 indexes via startup migration | — | Added idx_file_date standalone index, CREATE INDEX IF NOT EXISTS migration in manager.py |
| 2026-03-17 | Task 1.1 | Parser file type filtering — non-trim detection | — | Filename + sheet name + sheet structure validation. Processor skips non_trim files. |
| 2026-03-17 | Task 1.4 | Ingest validation — data quality flags | — | 4 checks per track, data_quality column, migration for existing DB |
| 2026-03-17 | Task 1.2 | Database cleanup tool in Settings page | — | Preview + delete, 3 options, backup reminder, batch deletion |
| 2026-03-17 | Task 1.5.1-4 | Dashboard & chart fixes — all 4 tasks | — | Pareto vertical, P-chart full width, focus cards, layout rework |
| | | | | |

---

## Key Data Points (Reference)

These numbers come from the 2026-03-17 analysis session. Update as data improves.

- **Total records in DB:** 81,328
- **Active backlog models with data:** 71 of 173
- **Legacy/noise records:** ~27,079 (33%)
- **Overall pass rate (laser trim):** 59.2%
- **Overall pass rate (Final Test):** 57.6%
- **Near-miss failures (1-3 fail points):** ~46% of all failures
- **Final Test match rate:** ~34%
- **Estimated annual scrap cost:** $107K+ (conservative, limited pricing data)
- **Failure rate trend:** Improving — 21% (2023) → 14% (early 2026)
- **Day-of-week signal:** Monday 20.3% vs other days ~16%
- **Average unit price (backlog):** $1,446
- **Price range:** $150 - $24,260
