# Phase 1: Model Reference System — Design Spec

**Date:** 2026-04-09
**Status:** Draft
**Branch:** To be created from main (tag v4.0.0 as safety net first)

## Goal

Add a model reference system so the app knows the engineering specs for each model — element type, product class, linearity type, resistance range, electrical angle, etc. This data enriches the Analyze page, enables filtering/grouping on Dashboard and Trends, and provides the foundation for Phase 2's spec-aware analysis engine.

## Non-Goals

- No customer data anywhere in the app
- No changes to the linearity calculation engine (Phase 2)
- No FT compensation parsing (Phase 2)
- No slope/angle optimization (Phase 2)

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

## Phase 2 Preview (Not In Scope)

For context on why Phase 1 matters, Phase 2 will use the `linearity_type` field to:
- Apply the correct optimization per model (offset-only for Absolute, offset+slope for Independent)
- Parse FT compensation values and compare against optimal
- Optimize within angle tolerance bounds using `electrical_angle_tol`
- Report true pass/fail after all allowed adjustments
- Show "station result vs optimized result" comparisons
