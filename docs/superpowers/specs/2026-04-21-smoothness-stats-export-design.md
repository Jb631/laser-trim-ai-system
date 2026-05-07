# Output Smoothness — Per-Model Stats & Excel Export

**Date:** 2026-04-21
**Status:** Approved

## Problem

The Output Smoothness page shows a flat list of results with global stats (total, pass rate, link rate) that don't update when filtering by model. There's no way to compare models against each other or export smoothness data for offline analysis. With OS testing now running on all parts at ATP, per-model visibility and export are needed.

## Design

### 1. Per-Model Stats Panel

Replace the current single-line stats label with a stats panel between the top bar and the results/detail area.

**When "All Models" is selected:** Show a comparison table (scrollable) with one row per model, sorted by pass rate ascending (worst first). Columns:

| Column | Description |
|--------|-------------|
| Model | Model number |
| Count | Total OS tests |
| Pass Rate | X/Y (Z%) |
| Avg Max Smoothness | Mean of `max_smoothness_value` across tests |
| Worst Case | Highest `max_smoothness_value` seen |
| Spec Limit | From `smoothness_spec` |
| Margin | `spec_limit - avg_max_smoothness` — how much room before failures |

Color-code the pass rate and margin cells: red for failing/negative margin, yellow for tight margin (<20% of spec), green for healthy.

**When a specific model is selected:** Show a single-model summary card with the same stats laid out horizontally (not a table). The results list below continues to show individual test results for that model.

**Clicking a model row in the comparison table** sets the model dropdown to that model, switching to the single-model view.

### 2. Excel Export

Add an "Export to Excel" button in the top bar (next to Refresh).

**Behavior:**
- Respects the current model filter — if "All Models", exports everything; if a model is selected, exports just that model
- One worksheet per model, worksheet tab named with the model number
- Each worksheet has a header row then one row per OS test, sorted by date descending

**Columns per worksheet:**

| Column | Source |
|--------|--------|
| Model | `SmoothnessResult.model` |
| Serial | `SmoothnessResult.serial` |
| Element Label | `SmoothnessResult.element_label` |
| Date | `SmoothnessResult.file_date` |
| Status | `SmoothnessResult.overall_status` (Pass/Fail) |
| Spec Limit | `SmoothnessResult.smoothness_spec` |
| Max Smoothness | `SmoothnessResult.max_smoothness_value` |
| Avg Smoothness | `SmoothnessResult.avg_smoothness_value` |
| Linked Trim ID | `SmoothnessResult.linked_trim_id` |
| Match Confidence | `SmoothnessResult.match_confidence` |

**Summary row at bottom of each worksheet:** Count, Pass Rate, Avg Max Smoothness, Worst Case, Margin — mirroring the on-screen stats.

File save dialog defaults to `Output_Smoothness_YYYY-MM-DD.xlsx`.

### 3. Database Changes

Add one new query method to `DatabaseManager`:

- `get_smoothness_stats_by_model(model=None, days_back=90)` — Returns a list of per-model stat dicts. When `model` is specified, returns a single-element list for that model. Each dict contains: `model`, `count`, `passed`, `pass_rate`, `avg_max_smoothness`, `worst_case`, `spec_limit`, `margin`.

The existing `search_smoothness_results(model=None)` already handles per-model filtering for the export data rows. No changes needed there.

### 4. Layout Change

Current layout (top to bottom):
1. Top bar (title, model dropdown, refresh, stats label)
2. Content (results list + detail panel side by side)

New layout:
1. Top bar (title, model dropdown, refresh, **export button**)
2. **Stats panel** (comparison table or single-model card)
3. Content (results list + detail panel side by side)

The stats panel gets a fixed height (~200px for the table, ~60px for single-model card) so the results list remains the main scrollable area.

## Files to Modify

| File | Change |
|------|--------|
| `src/laser_trim_analyzer/gui/pages/smoothness.py` | Stats panel, export button, layout restructure |
| `src/laser_trim_analyzer/database/manager.py` | Add `get_smoothness_stats_by_model()` |

## Out of Scope

- Smoothness trend charts (time series per model) — future work
- Integration with the existing Export page — smoothness export lives on the smoothness page itself
- Smoothness spec lookup from model_specs — currently stored per-result from the test file
