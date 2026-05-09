# Drift Tab UX Redesign — Design Spec

**Date:** 2026-05-08
**Status:** Approved (brainstorm), pending implementation plan
**Source:** Issue #3 in `Work Files/5-8-2026/app errors.txt` — "resistance drift is hard to understand, no model specific view for drift."

---

## Problem

The Drift tab on the Trends page has two sub-views (ML Drift and Process Drift) toggled by a segmented button. Both have usability issues:

- The **ML Drift timeline** is a sparse scatter chart — sigma drift events are infrequent, so the chart is mostly empty space dotted with tiny up/down triangles. The user reports "the chart is large with little tiny arrows, the design doesn't seem correct."
- The **Process Drift** view is three stacked horizontal bar-chart panels (Untrimmed Resistance, Electrical Angle, Trim Passes), each ranking the top 10 models by z-score. The information is readable ("I can easily tell if the resistance is changing from the norm"), but the layout feels off as a design.
- There is **no per-model view** anywhere in the Drift tab. When a user spots a drifting model, they cannot drill in to see the model's actual behavior over time.

Dead code at `trends.py:3046–3258` (`_load_drift_data`, `_update_drift_display`, `_on_drift_model_select`) implements a per-model sidebar + chart that was never wired up after the segmented-button refactor. The chart widget method `ChartWidget.plot_drift_chart()` (chart.py:1492) it relied on still exists.

## Goals

1. Replace both sub-views' "All models" rendering with denser, more scannable table layouts.
2. Add a global model filter at the top of the Drift tab. Selecting a model switches the entire tab into a single-model "investigation" dashboard.
3. Reuse the existing data plumbing where possible — the design must not require new ML/statistical concepts.

## Non-goals

- Changing the underlying drift detection algorithms (CUSUM/EWMA/z-score thresholds stay as-is).
- Replacing the dashboard's Drift Alerts card or any other entry point — only the Drift tab on the Trends page is in scope.
- Per-track drift (drift is per-model only, as today).

---

## Design

### 1. Tab header row

A new model filter dropdown appears in the Drift tab's header alongside the existing day-range selector:

```
[ Day range ▼ ]   [ Model: All models ▼ ]   [ ML Drift | Process Drift ]
```

- The dropdown is populated by `db.get_models_with_sigma_data()` — only models that have at least one analysis row inside the active day-range window appear, so dead/never-processed models don't clutter the list.
- The default value is `All models`.
- Selecting a specific model:
  - Hides the `[ ML Drift | Process Drift ]` segmented button.
  - Replaces the content area with the single-model dashboard (Section 3).
  - A small `✕ clear filter` link or button next to the dropdown returns to All models.
- The day-range filter still applies in both modes (it bounds the data window for tables and charts alike).

### 2. All-models mode

The existing `[ ML Drift | Process Drift ]` segmented toggle stays. Each sub-tab now renders a sortable model table.

#### 2a. ML Drift sub-tab

A single sortable table, one row per model:

| Model | Status | Drift score | Last event | Days drifting | Sigma trend |
|-------|--------|-------------|------------|---------------|-------------|
| 7965  | ↑ DRIFTING (red badge) | 12.4 / 5.0 | 2026-04-22 | 16d | sparkline (red) |
| 8492  | ↑ DRIFTING (red badge) | 8.7 / 5.0  | 2026-05-01 | 7d  | sparkline (red) |
| 6607  | ↓ DRIFTING (orange)    | 6.1 / 5.0  | 2026-04-29 | 9d  | sparkline (orange) |
| 8508-A| ✓ stable (green)       | 2.1 / 5.0  | —         | —   | sparkline (green) |
| 8232-1| ○ no baseline (gray)   | —          | —         | —   | "need 30+ samples" |

- **Status** badge: ↑ DRIFTING / ↓ DRIFTING / ✓ stable / ○ no baseline. Color follows current direction-color scheme (red = up = degrading, orange = down = improving, green = stable, gray = no baseline).
- **Drift score**: `cusum_value / cusum_h` from the per-model `DriftDetector`. Renders as `"X.X / Y.Y"`. Colored to match status.
- **Last event**: Date the model entered its CURRENT drifting run — i.e., when CUSUM first crossed `cusum_h` in the active spell. If the model is presently stable, this is blank (a previously-drifted-but-recovered model shows `✓ stable` with no Last event). Pulled from the `drift_events` table or computed from cusum history — confirmed during planning.
- **Days drifting**: Days between Last event and today. Blank for stable models. Counts contiguous drifting days only; recovery resets the counter.
- **Sigma trend**: Inline SVG sparkline of the model's sigma_gradient over the day-range window. Color follows status.
- Default sort: **Status** (drifting first), then **Drift score** descending. All columns sortable by click.
- Click a row → sets the global model filter to that model (i.e., drills into single-model view).

#### 2b. Process Drift sub-tab

A secondary tab strip selects which physical metric is shown:

```
[ Untrimmed Resistance | Electrical Angle | Trim Passes ]
```

Below the strip, a sortable table with one row per model:

| Model | Baseline | Recent | Δ% | z | Trend |
|-------|----------|--------|----|----|-------|
| 8492  | 10.45 kΩ | 11.21 kΩ | +7.3% | +4.8 | sparkline (red) |
| 7965  | 3.20 kΩ  | 3.42 kΩ  | +6.9% | +3.5 | sparkline (red) |
| 8508-A| 8.10 kΩ  | 7.62 kΩ  | -5.9% | -2.6 | sparkline (orange) |
| 2475-8| 5.05 kΩ  | 5.10 kΩ  | +1.0% | +0.5 | sparkline (green) |

- The metric tab strip replaces the three stacked bar panels — only one metric is shown at a time, eliminating vertical scroll.
- **Baseline** / **Recent**: numeric values formatted with the metric's unit (`fmt` and `unit` from `_PROCESS_DRIFT_METRICS`).
- **Δ%**: `(recent − baseline) / baseline × 100`, signed.
- **z**: existing z-score, color-coded by severity (same green/orange/red/purple scale as the current bar chart).
- **Trend**: sparkline of the metric over the day-range window for that model.
- Default sort: |z| descending. All columns sortable.
- Click a row → drills into single-model view (same as ML Drift table).

### 3. Single-model mode

Triggered when a specific model is selected in the global filter (either from the dropdown or by clicking a row in either All-models table).

The segmented `[ ML Drift | Process Drift ]` toggle is hidden. The content area becomes a single dashboard for the selected model:

#### 3a. Header bar

```
[7965]  [↑ DRIFTING]  [16d drifting]  [score 12.4 / 5.0]  [1,101 units]
```

- Model name (large, bold).
- Status badge — same style as the All-models tables.
- Pills for: days drifting (or "stable for Xd" if not drifting), drift score, and unit count in the day-range window.

#### 3b. 2×2 chart grid

Four panels, each a time-series chart:

| Top-left: **Sigma drift** | Top-right: **Untrimmed Resistance** |
| Bottom-left: **Electrical Angle** | Bottom-right: **Trim Passes** |

- **Sigma drift** panel uses the existing `ChartWidget.plot_drift_chart()` method (chart.py:1492). Plots sigma_gradient vs date with control limits (UCL, LCL, center) and a vertical line at the baseline cutoff date. Bottom of panel: small caption with sigma values and CUSUM threshold (e.g., `0.0142 → 0.0181 (+27%) · UCL 0.0175`).
- **Process metric panels** plot the metric over time with a horizontal baseline-mean reference line. Bottom caption shows baseline → recent values, % change, and z-score (e.g., `3.20 kΩ → 3.42 kΩ (+6.9%) · z=+3.5`).
- Each panel header includes a small status indicator (↑ red / ↓ orange / · green) for that specific signal.

If a metric has insufficient data for the model, that panel shows an empty state: `"Need ≥20 baseline + ≥5 recent samples; have N/M."`

If the model has no baseline at all (sigma drift detector not trained), the sigma panel reads `"No baseline yet — train this model in Settings, or wait for ≥30 samples."` The other three panels still render their time series.

---

## Data layer changes

The design implies the following new or extended queries on `DatabaseManager`. Exact signatures to be settled in the implementation plan; this section is a checklist to size the work.

1. **`get_drift_status_table(days_back: int) -> List[Dict]`**
   Per-model row data for the ML Drift table: status, drift score, last event date, days drifting. Builds on existing `ml_manager.get_drift_status()` plus a query for last drift event date and a recent-sigma series for the sparkline.

2. **`get_process_drift_table(metric, days_back) -> List[Dict]`**
   Replaces / extends `get_process_drift_by_model()` to also return Δ% and a recent-history series for the sparkline.

3. **`get_model_drift_dashboard(model, days_back) -> Dict`**
   One round-trip for the single-model view: sigma series with control limits, plus three metric series (untrimmed resistance, electrical angle, trim passes) with their baseline/recent stats.

4. **`get_models_with_sigma_data() -> List[str]`**
   Populates the model dropdown (skip models with zero analysis rows in the day-range window).

The dead `_load_drift_data` / `_on_drift_model_select` block (trends.py:3046–3258) and its sidebar widget references should be deleted as part of this work.

---

## Out of scope

- Comparing two models side-by-side.
- Exporting drift snapshots (separate Reports work).
- Editing per-model drift thresholds (lives in Settings).
- Per-track drift (drift remains per-model).

---

## Pointers to existing code

- Drift tab structure: `src/laser_trim_analyzer/gui/pages/trends.py`
  - `_create_drift_view`, `_on_drift_subtab_changed`, `_show_drift_timeline`, `_render_drift_timeline`, `_show_process_drift`, `_render_process_drift`
  - Dead code to remove: `_refresh_drift_data`, `_load_drift_data`, `_update_drift_display`, `_on_drift_model_select`, `_show_drift_error` (3046–3258), plus the unassigned `_drift_model_list` / `_drift_chart_frame` / `_drift_details_label` / `_drift_chart_placeholder` references.
- Sigma drift chart method (already exists): `src/laser_trim_analyzer/gui/widgets/chart.py:1492` — `plot_drift_chart`.
- Drift detection state per model: `ml_manager.drift_detectors[model]` provides `cusum_pos`, `cusum_neg`, `cusum_h`, `ewma_value`, `baseline_cutoff_date`, `is_drifting`, `drift_direction`, `get_control_limits()`.
- Process drift query: `DatabaseManager.get_process_drift_by_model(metric, baseline_days, recent_days)`.
- Metric metadata (units, fmt strings): `DatabaseManager._PROCESS_DRIFT_METRICS`.
