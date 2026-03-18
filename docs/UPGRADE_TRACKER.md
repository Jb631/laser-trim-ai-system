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
**Status:** NOT STARTED

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
- [ ] Add "Database Cleanup" section to Settings page
- [ ] Add option: delete records for models NOT in MPS list
- [ ] Add option: delete records older than specified date
- [ ] Add preview mode (show what would be deleted before committing)
- [ ] Add confirmation dialog with record count
- [ ] Add backup reminder
- [ ] Test: preview and execute cleanup on test database
- **Date completed:**
- **Notes:**

---

## Phase 1.5: Dashboard & Chart Fixes

**Goal:** Make charts readable and useful.
**Status:** NOT STARTED

### Task 1.5.1 — Fix Pareto Chart Layout
- [ ] Rewrite `plot_pareto()` as vertical bar chart (standard Pareto orientation)
- [ ] Move cumulative % to right Y-axis (`ax.twinx()`)
- [ ] Limit to top 10 models, rotate X labels 45 degrees
- [ ] Add 80% cumulative line as horizontal dashed line
- [ ] Color bars by severity gradient (darkest red for #1)
- [ ] Test: verify chart is readable with real data
- **Date completed:**
- **Notes:**

### Task 1.5.2 — Improve Dashboard Layout
- [ ] Expand P-chart trend to full width (3 columns), increase figure size
- [ ] Rearrange rows: metric cards → system info → P-chart (full width) → Pareto + focus panel
- [ ] Move alerts/drift to collapsible section or separate row
- [ ] Test: verify layout works on different screen sizes
- **Date completed:**
- **Notes:**

### Task 1.5.3 — Fix P-Chart Readability
- [ ] Make control limit lines more visible (thicker, distinct color)
- [ ] Make out-of-control markers larger and more prominent
- [ ] Reduce date labels to max 8 with month-day format
- [ ] Add visual fill between data and center line
- [ ] Test: verify chart is readable at dashboard size
- **Date completed:**
- **Notes:**

### Task 1.5.4 — Clean Up "Where to Focus" Panel
- [ ] Replace raw textbox with structured formatted entries
- [ ] Limit to top 5 models (not 10)
- [ ] Color-code entries by severity
- [ ] Make DECLINING tag visually prominent
- [ ] Test: verify panel is scannable at a glance
- **Date completed:**
- **Notes:**

---

## Phase 2: Operational Analytics

**Goal:** Answer "why are units failing?" and "what's it costing us?"
**Status:** NOT STARTED
**Prerequisite:** Phase 1 complete (clean data)

### Task 2.1 — Model Management with Pricing
- [ ] Design storage: config.yaml vs database table for pricing
- [ ] Add "Unit Price" column to MPS model list in Settings
- [ ] Add "Import from Excel/CSV" button for bulk model+price import
- [ ] Add filtering capability for import (select which models to include)
- [ ] Make pricing accessible to Dashboard and Trends pages
- [ ] Test: import pricing from backlog export, verify values accessible
- **Date completed:**
- **Notes:**

### Task 2.5 — Failure Margin Tracking
- [ ] Calculate `max_violation` in analyzer.py for failed tracks
- [ ] Calculate `avg_violation` for failed tracks
- [ ] Calculate `margin_to_spec` for passing tracks
- [ ] Add columns to track_results database table
- [ ] Update processor to store new metrics
- [ ] Backfill: re-analyze existing records to populate new columns (or calculate on demand)
- [ ] Test: verify margin values make sense for known pass/fail units
- **Date completed:**
- **Notes:**

### Task 2.2 — Near-Miss Analysis View
- [ ] Add near-miss distribution chart (fail points histogram)
- [ ] Add near-miss percentage display (% of failures with 1-3 fail points)
- [ ] Add per-model near-miss comparison
- [ ] Add sigma gradient comparison (near-miss vs hard-fail vs pass)
- [ ] Decide placement: Dashboard section or Analyze page tab
- [ ] Test: verify chart matches manual SQL analysis
- **Date completed:**
- **Notes:**

### Task 2.3 — Cost Impact Dashboard
- [ ] Add Pareto chart: models ranked by failure cost
- [ ] Add monthly scrap cost trend line
- [ ] Add recovery opportunity calculator ("if model X improves to Y%, save $Z")
- [ ] Add configurable cost ratio (default 50%, adjustable in Settings)
- [ ] Test: verify dollar calculations against manual spreadsheet check
- **Date completed:**
- **Notes:**

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
