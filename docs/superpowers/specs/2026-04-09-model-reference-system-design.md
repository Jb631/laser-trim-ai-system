# Laser Trim Analyzer v5 — Accuracy & Analytics Upgrade

**Date:** 2026-04-09
**Status:** Draft
**Branch:** To be created from main (tag v4.0.0 as safety net first)

## Vision

Transform the app into a definitive manufacturing quality platform where **anyone** can open it and immediately understand: what are the quality issues, where should we focus, and what's getting better or worse. The app should surface insights that drive action — not just record measurements.

### Design Principles

1. **Accuracy first** — Pass/fail determinations must match what a skilled operator would conclude after applying all allowed adjustments. Inaccurate data undermines trust in everything else.
2. **Actionable at a glance** — Dashboard shows the 3-5 things that need attention right now, ranked by impact. No digging required for the critical stuff.
3. **Layered depth** — Executive summary → model-level detail → unit-level detail. Each layer answers different questions.
4. **Keep everything, enhance everything** — Sigma analysis, ML thresholds, drift detection, ML insights, recommendations all stay and get better. Nothing is removed.
5. **No customer data** — Technical specs and quality metrics only.

### Existing Features to Preserve and Enhance

| Feature | Current State | Enhancement Direction |
|---------|--------------|----------------------|
| Sigma analysis | Per-track sigma gradient | Keep as leading indicator, contextualize with spec-aware thresholds |
| ML per-model thresholds | Trained from historical data | Add spec-aware baseline comparison, improve with accurate pass/fail |
| Drift detection (CUSUM/EWMA) | Detects process shifts | Add element-type and product-class drift groupings |
| ML failure prediction | Per-model classifiers | Improve accuracy with spec-aware features |
| Risk categories | Low/Medium/High | Tie to cost impact and production volume |
| Trim effectiveness | Linearity improvement % | Show before/after optimization comparison |
| Near-miss analysis | Units close to threshold | Enhance with spec-aware margin calculation |
| Pareto charts | Top failing models | Add element type and product class breakdowns |
| P-charts | Process control | Add control limits per element type |
| Executive export | Excel summary | Add spec-aware pass/fail comparison |

### New Analytics to Add

- **Spec-aware pass rate** — "Station says 60% pass, but with proper adjustment 78% actually pass" — this is the key metric that shows the real quality picture
- **Adjustment impact analysis** — Which models benefit most from offset/slope optimization? Where is the gap between station and optimal?
- **Element type performance comparison** — Ceramic vs Winding vs Black element failure rates and trends
- **Product class performance** — Runner vs Low Vol vs Space quality levels
- **Model-level scorecards** — At-a-glance health for each model: pass rate trend, Cpk, drift status, ML confidence, spec margins
- **Resistance/angle tracking** — Are measured values within spec? Trends over time.
- **Cpk/Ppk analysis** — Process capability indices per model using spec-aware limits
- **First-pass yield vs adjusted yield** — How many pass on first trim vs require re-trim?
- **Cost-weighted prioritization** — Combine failure rate × production volume × product class to rank where to focus

---

## Phased Implementation

### Constraints
- Keep current app (v4.0.0) as safety net — tag before starting
- Each phase delivers working value independently
- No breaking changes to existing data or functionality

---

## Phase 1: Model Reference System

### Goal
Add model engineering specs to the app so it knows what each model is, how it should be evaluated, and how to categorize it for analysis.

---

## 1. Database: `model_specs` Table

New SQLAlchemy model `ModelSpec` in `database/models.py`:

| Column | Type | Notes |
|--------|------|-------|
| `id` | Integer, PK | Auto-increment |
| `model` | String(50), unique, not null | Model number (e.g., "8340-1", "1844205") |
| `element_type` | String(30), nullable | Ceramic, Winding, Black, Blue, G11, Infinatron |
| `product_class` | String(20), nullable | Runner, Low Vol, Space, Panel, Rotary |
| `linearity_type` | String(30), nullable | Absolute, Independent, Term Base, Zero-Based, VR Max, Custom |
| `linearity_spec_text` | String(100), nullable | Raw spec text from reference (e.g., "± 0.5% (Absolute)") |
| `linearity_spec_pct` | Float, nullable | Parsed numeric spec as percentage (e.g., 0.5) where parseable |
| `total_resistance_min` | Float, nullable | Minimum resistance in Ohms |
| `total_resistance_max` | Float, nullable | Maximum resistance in Ohms |
| `electrical_angle` | Float, nullable | Nominal electrical angle (inches or degrees) |
| `electrical_angle_tol` | Float, nullable | Angle tolerance ± value |
| `electrical_angle_unit` | String(10), nullable | "in" or "deg" |
| `output_smoothness` | String(50), nullable | Output smoothness spec if applicable |
| `circuit_type` | String(10), nullable | "Open" or "Closed" |
| `notes` | Text, nullable | Free-text notes |
| `created_at` | DateTime | Auto-set |
| `updated_at` | DateTime | Auto-updated |

**Migration:** Add table via existing migration pattern in `DatabaseManager.__init__`. No schema changes to existing tables.

**Relationship to existing data:** The `model_specs.model` field matches `analysis_results.model`. No foreign key — joined by model string at query time. This keeps the existing schema untouched and handles models that exist in one table but not the other.

---

## 2. Excel Import

### One-time seed from `model_reference_table_cleaned.xlsx`

Import logic in `database/manager.py`:

- Read all three sheets (Model Reference, Element Type, Product Class)
- Model Reference sheet is primary — has the most complete data
- Element Type and Product Class sheets supplement with broader coverage (216 and 309 models respectively)
- For models that appear only in the supplementary sheets, create a record with just element_type or product_class
- Parse `linearity_spec_text` to extract `linearity_type` and `linearity_spec_pct` where possible:
  - "± 0.5% (Absolute)" → type=Absolute, pct=0.5
  - "± 1.0% (Independent)" → type=Independent, pct=1.0
  - "± 1.0% (Term Base)" → type=Term Base, pct=1.0
  - "± 0.5% (VR Max)" → type=VR Max, pct=0.5
  - "See Chart" / "See Table" / complex specs → type=Custom, pct=None
- Parse resistance range text "950 - 1,050 Ω" → min=950, max=1050
- Parse electrical angle "1.31" ± .005"" → angle=1.31, tol=0.005, unit="in"
- Skip Customer column entirely
- Skip Operator column

### Re-import capability

- Settings page button: "Import Model Specs from Excel"
- File picker to select the Excel file
- Merge logic: update existing models, add new ones, never delete (user might have added custom models)
- Show summary: "Imported 199 models (185 updated, 14 new)"

---

## 3. Model Specs Page (New — 7th Page)

A dedicated page in the app navigation (Dashboard, Process, Analyze, Compare, Trends, **Specs**, Settings).

### Model List View
- Scrollable table showing all models with key columns: model, element type, product class, linearity type, resistance range, electrical angle
- Search/filter box to find models quickly
- Sort by any column
- Color-coded completeness: highlight models missing key fields (linearity type, element type)

### Edit Model (Inline or Detail Panel)
- Click a model row to edit its fields
- Element type: dropdown with known values + free text option
- Product class: dropdown with known values
- Linearity type: dropdown (Absolute, Independent, Term Base, Zero-Based, VR Max, Custom)
- Linearity spec text: free text for the raw spec description
- Resistance range: two number fields (min/max Ohms)
- Electrical angle: number + tolerance + unit dropdown (in/deg)
- Output smoothness: free text
- Circuit type: Open/Closed dropdown
- Notes: free text
- Save/Cancel buttons

### Add New Model
- "Add Model" button at top
- Same form as edit, with model number field editable

### Delete Model
- Delete with confirmation

### Initial Seed
- One-time "Import from Excel" button in Settings page to seed the database from the reference spreadsheet
- After initial import, all management happens on the Specs page
- Re-import option available but merges only (updates existing, adds new, never deletes)

---

## 4. Analyze Page Enrichment

When viewing an analysis result, if a matching `model_spec` exists:

### Info Panel Enhancement
Add a "Model Specs" section to the File Info tab showing:
- Element Type
- Product Class
- Linearity Type
- Resistance Range
- Electrical Angle ± tolerance
- Output Smoothness (if set)
- Circuit Type (if set)

### Cross-Check Indicators
Where the file's parsed data overlaps with the reference spec:
- Compare file's `linearity_spec` (from upper/lower limits) against reference `linearity_spec_pct`
- If they differ significantly, show a warning indicator: "File spec (±0.025) differs from reference (±0.5%)"
- This helps catch test station misconfiguration

---

## 5. Dashboard & Trends: Filtering and Grouping

### Dashboard Page
- Add filter dropdowns: Element Type, Product Class
- When filtered, all dashboard cards (pass rate, linearity quality, etc.) reflect only matching models
- Add a breakdown card: "Performance by Element Type" or "Performance by Product Class" showing pass rates per category

### Trends Page
- Add the same filter dropdowns
- Trend charts can be filtered by element type or product class
- Add comparison view: overlay trends for different element types or product classes on the same chart

### Implementation
- New query methods in `database/manager.py`:
  - `get_dashboard_stats(days_back, element_type=None, product_class=None)`
  - Similar filter params added to existing trends queries
- Queries join `analysis_results` to `model_specs` on model name
- Models without a spec entry are shown as "Unspecified" in groupings

---

## 6. Data Visualization Improvements

### Analyze Page Charts
- Show electrical angle and resistance data when available from model specs
- Display element type and product class in chart title/subtitle for context

### Dashboard
- Performance by Element Type bar chart (pass rate per type)
- Performance by Product Class bar chart
- These use the model_specs join to categorize

### Trends
- Filter trend lines by element type or product class
- Compare categories side-by-side (e.g., Ceramic vs Winding failure trends over time)

---

## 7. Technical Notes

### Model Name Matching
The `model_specs.model` must match the parsed model from trim/FT files. The existing parser normalization applies (e.g., "8340-1" stays "8340-1"). The editor should validate model names match the format used in the database.

### Performance
- `model_specs` table will have ~300-500 rows — trivially small
- Joins with `analysis_results` (82K+ rows) should use the existing model index
- Cache model specs in memory at app startup for fast lookup during processing

### Backward Compatibility
- No changes to existing tables or columns
- App works fine with an empty model_specs table (no specs = no enrichment, no filtering)
- Existing data and functionality fully preserved

---

## Phase 2: Spec-Aware Analysis Engine

**Goal:** Make pass/fail determination as accurate as possible by doing what operators do manually — optimizing offset and slope within allowed tolerances per linearity type.

### 2.1 Linearity Type-Aware Optimization

Using the `linearity_type` from the model_specs table, apply the correct calculation:

| Linearity Type | Offset | Slope | What the engine does |
|---------------|--------|-------|---------------------|
| Absolute | Fixed (index point) | Fixed | No optimization — raw error vs fixed reference line |
| Absolute + angle tol | Fixed | Within tolerance | Optimize slope within `electrical_angle_tol` bounds |
| Term Base | Fixed (endpoints) | Fixed | Same as Absolute (reference pinned at 0% and 100%) |
| Zero-Based | Fixed at zero | Free | Optimize slope only, offset pinned at zero |
| Independent | Free | Free | Optimize both slope + offset (best-fit line, minimax) |
| VR Max | TBD | TBD | Verify meaning — likely same as Absolute expressed as voltage ratio |
| Custom (bowtie etc.) | Per model | Per model | Use per-point limits from file, optimize offset within those |

### 2.2 Offset + Slope Optimization (Trim AND Final Test)

Upgrade `_calculate_optimal_offset` in `analyzer.py` to `_calculate_optimal_adjustment`:

- **Applies to both trim file analysis AND Final Test analysis** — the same optimization logic runs on any linearity error data regardless of source
- **Current:** Optimizes offset only (1 DOF) — shifts error curve up/down
- **New:** Optimizes offset + slope (2 DOF for Independent) or slope within bounds (for Absolute + angle tolerance)
- Slope adjustment = rotating the reference line, mathematically equivalent to adjusting electrical angle
- For each model, the allowed DOF comes from `linearity_type` and `electrical_angle_tol`
- Optimization goal: minimize fail points (same as current), then minimize max error as tiebreaker
- For non-uniform (bowtie) limits: constrained optimization against per-point upper/lower limits
- Store both raw and optimized results so users can see the improvement

### 2.3 FT Compensation Parsing

Parse the compensation value from Final Test files:
- Format 1 (Sheet1-based): Cell M4 labeled "Compensation" — the offset the test station applied
- Other formats: investigate and document where compensation lives in each format variant
- Store parsed compensation as `ft_compensation` on the FinalTestResult record

### 2.4 Station vs Optimal Comparison

New analysis output fields:
- `station_fail_points` — fail count using the FT station's compensation (what the station reported)
- `optimal_fail_points` — fail count after app's optimal offset+slope adjustment
- `optimal_offset` — the app's calculated best offset
- `optimal_slope_adj` — the app's calculated best slope adjustment (if allowed by linearity type)
- `could_pass` — boolean: could this unit pass with optimal adjustments?

### 2.5 UI Changes

- Analyze page: show "Station: 14 fails → Optimized: 0 fails (PASS)" comparison
- Compare page: show station vs optimized side-by-side for trim-to-FT pairs
- Dashboard: option to show pass rates using "optimized" determination vs "station" determination
- Chart: overlay the optimized reference line against the station reference line

---

## Phase 3: Format Audit, Parsing Fixes & Output Smoothness

**Goal:** Ensure every file format variant is correctly parsed with accurate data extraction. Add output smoothness analysis as a new data type.

### 3.1 Format Documentation

- Catalog all file format variants across System A, System B, and Final Test
- Document column layouts, metadata cell locations, and sheet structures for each variant
- Bring home example files for each format and build a test suite

### 3.2 Pre-Trim Chart Fix

- Investigate the "straight line down" issue on pre-trim linearity charts
- Verify column mapping for untrimmed sheets across all System B variants
- Fix `allow_nan` inconsistency between trimmed and untrimmed error extraction
- Handle scale mismatch when pre-trim error range overwhelms post-trim (separate Y axis or auto-scale)

### 3.3 FT Matching Improvements

- Audit matching accuracy using work database
- Fix serial normalization collision risk (trailing letter stripping)
- Consider closest-in-time matching vs most-recent-before
- Add match quality indicators visible in Compare page

### 3.4 Cross-Format Spec Verification

- When processing a file, look up its model in `model_specs`
- Compare file-parsed linearity limits against reference spec
- Flag mismatches (possible test station misconfiguration)
- Compare file's electrical angle against reference spec angle + tolerance

### 3.5 Output Smoothness Integration

Add output smoothness as a new data type in the app. OS files live alongside linearity files in the Test Station folder (in an "Output Smoothness" or "OS" subfolder per model).

**File format (consistent across all models):**
- Excel (.xlsx) with sheets: Test Data, Report, Rev History
- Test Data sheet layout:
  - Metadata (rows 0-7, cols A-C): Model, Serial, Pot Type (Rotary/Linear), Gear Ratio, Input Voltage, Deviation Spec %, Electrical Travel (deg or in), Sample Rate (Hz)
  - Result (row 1, cols G-I): Max Deviation (V), Max Spec Deviation (V), PASSED/FAILED
  - Time-series (rows 1+, cols D-E): Time (s), Filtered Volts (V) — 1000 Hz sampled data
- Filename pattern: `model-snSerial_OS_date_time.xlsx` (the `_OS_` identifies it as output smoothness)

**New database table: `smoothness_results`**
- model, serial, test_date, pot_type, gear_ratio, input_voltage
- deviation_spec_pct, electrical_travel, electrical_travel_unit
- max_deviation, max_spec_deviation, pass/fail
- Time-series data stored as JSON array
- Linked to trim analysis by model+serial (same as FT matching)

**New database table: `smoothness_data`** (optional — or store time-series inline)
- Foreign key to smoothness_results
- Time-series arrays: time_data, voltage_data

**Parser:** New `SmoothnessParser` in `core/smoothness_parser.py`
- Detect OS files by `_OS_` in filename
- Extract metadata from rows 0-7
- Extract result from row 1
- Extract time-series from cols D-E
- Simple, consistent format — no variant handling needed

**UI — New Smoothness page (8th page):**
- List view of OS test results by model
- Chart: Filtered voltage vs time with spec deviation band overlay
- Pass/fail summary per model
- Trends: OS pass rate over time, per model
- Link to corresponding trim/FT data for the same serial

**Dashboard integration:**
- OS pass rate card alongside linearity pass rate
- Models failing OS highlighted in the attention list
- OS trends in the Trends page filters

---

## Phase 4: Advanced Analytics & Visualization

**Goal:** Make the app the single source of truth for manufacturing quality. Anyone opens it and knows exactly what's wrong, what's improving, and where to focus — without needing to ask anyone.

### 4.1 Dashboard Overhaul — "Open and Know"

The dashboard should answer these questions at a glance:

**Top section — What needs attention right now?**
- Top 3-5 models ranked by impact (failure rate × volume × product class priority)
- Each card shows: model, pass rate trend arrow, fail count this week, element type badge
- Color-coded: red = getting worse, yellow = stable-bad, green = improving

**Middle section — How are we doing overall?**
- Spec-aware pass rate: "Station: 62% → Optimized: 74%" showing the gap between reported and actual
- First-pass yield trend (pass on first trim vs re-trim needed)
- Pass rate by element type (bar chart: Ceramic vs Winding vs Black etc.)
- Pass rate by product class (Runner vs Low Vol vs Space)

**Bottom section — Process health**
- Drift alerts: models where CUSUM/EWMA detected a shift (existing, enhanced with element type grouping)
- ML confidence summary: how many models have reliable predictions vs need more data
- Sigma process health (existing, preserved)

### 4.2 Model Scorecards

Accessible from Dashboard (click a model) or Analyze page — a single-page summary for any model:

- **Pass rate:** Current, 30-day trend, 90-day trend, all-time
- **Spec-aware pass rate:** Station vs optimized (Phase 2 data)
- **Cpk/Ppk:** Process capability index calculated from linearity error distribution vs spec limits
- **Sigma health:** Current sigma gradient distribution, ML threshold, margin
- **Drift status:** CUSUM/EWMA indicators, when last shift detected
- **ML confidence:** Predictor accuracy, training samples, staleness
- **Failure pattern:** Where on the travel do failures cluster? (position-based fail heatmap)
- **Trim effectiveness:** Average linearity improvement from trimming
- **Volume:** Units processed per week/month

### 4.3 Enhanced Trends Page

- **Comparative trends:** Overlay multiple models, element types, or product classes on one chart
- **Cpk trend:** Process capability over time per model
- **Yield trend:** First-pass yield vs adjusted yield over time
- **Drift timeline:** Visual timeline showing when drift events occurred per model
- **Spec-aware vs station trend:** Gap between station-reported and optimized pass rates over time

### 4.4 Cpk/Ppk Process Capability

Add process capability analysis:

- **Cpk** (within-subgroup): How well the process performs relative to spec limits, accounting for centering
- **Ppk** (overall): Long-term process performance
- Calculated from linearity error distribution vs upper/lower spec limits
- For bowtie specs: use the tightest spec point (minimum band width) as the capability reference
- Display on model scorecards and trends
- Color-coded: Cpk < 1.0 = red (not capable), 1.0-1.33 = yellow (marginal), > 1.33 = green (capable)

### 4.5 Failure Pattern Analysis

- **Position-based fail heatmap:** For each model, show where on the travel linearity failures cluster. Do failures happen at the ends, center, or specific positions? This tells the engineer WHERE the trim process needs improvement.
- **Failure mode categorization:** Offset issue (all points shifted) vs shape issue (specific region fails) vs overall (scattered failures)
- **Before/after trim comparison:** Show the improvement distribution — which positions improved most from trimming?

### 4.6 ML Enhancements

- **ML recommendations preserved and enhanced:** Keep existing failure predictions and risk categories
- **Spec-aware ML features:** Add linearity type, element type, product class as features to the predictor
- **Adjustment recommendation:** "This model typically benefits from +0.003 offset adjustment" based on historical optimal adjustments
- **Drift detection per category:** Detect drift across all Ceramic models or all Runners, not just per-model
- **Anomaly detection context:** Flag anomalies with context — "This unit's sigma is 3x the model average, and it's a Ceramic element which typically has tighter distributions"

### 4.7 Reporting & Export Enhancements

- **Executive summary export:** One-page PDF/Excel showing overall quality health, top issues, trend direction
- **Model scorecard export:** Detailed per-model report for engineering review
- **Spec-aware pass/fail export:** Export with both station and optimized determinations
- **Scheduled reports:** Weekly/monthly auto-generated summaries (if feasible with current architecture)
