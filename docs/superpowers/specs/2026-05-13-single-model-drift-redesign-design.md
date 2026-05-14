# Single-Model Drift Dashboard — Redesign Spec

**Date:** 2026-05-13
**Status:** Approved (brainstorm), pending implementation plan
**Source:** User report on 2026-05-13 — "the charts are not helpful in the current design they have lines that go all over the place, and the trim pass chart is blank."
**Supersedes (single-model section only):** `docs/superpowers/specs/2026-05-08-drift-ux-design.md` §3 (single-model mode). All-models tables in that spec are unchanged.

---

## Problem

The single-model investigation view (Trends → Drift → select a model) renders a 2×2 grid of time-series charts (Sigma / Untrimmed Resistance / Electrical Angle / Trim Passes). It is currently unusable as a decision-support view:

1. **Chart noise.** Each panel plots raw per-track measurements as a connected line. A model running 1,000+ units over the day-range window produces a dense noise carpet, not a trend. The eye cannot extract direction, rate, or "is this in control today" from the rendering.
2. **Trim Passes panel is blank.** Two reasons compound:
   - The `trim_pass_count` column was added by parser commit `26894cb` (2026-05-07). Every row ingested before that date carries `NULL` for the column. The current panel filters `None` out, leaving most models with no points.
   - Even after re-parse, `trim_pass_count` is an integer (mostly 1, occasionally 2). Plotted as a raw line, it's a flat line at 1.0 with occasional integer jumps — not a meaningful trend signal.

The 2026-05-08 spec defined the *layout* of this view (2×2 grid, header pills) correctly. The bug is that every panel uses the wrong chart type for the data it carries.

## Goals

1. Replace the per-panel chart rendering so the trend is visible at a glance from the panel header alone, with the chart confirming detail.
2. Reframe the Trim Passes panel as **Retrim Rate** — a smoothed daily percentage of units that needed more than 1 pass. Same visual language as the other process panels; meaningful even at low retrim counts.
3. Make the Sigma panel a hybrid SPC chart: individual unit dots (green = in-control, red = out-of-control) plus a smoothed mean overlay. Preserves the Shewhart "single-point violation" signal that the other metrics don't have.
4. Bucket data adaptively by unit count, not by fixed time intervals, so the same chart renders sensibly for a 5-unit-per-week model and a 500-unit-per-day model.

## Non-goals

- Changing the all-models tables, the day-range filter, the model dropdown, or any other Drift-tab plumbing. Layout from the 2026-05-08 spec stays.
- Changing what counts as "drifting" — the underlying `DriftDetector` (CUSUM/EWMA), the `z_score` math, and the metric set stay as-is.
- Adding new metrics. Sigma / Untrimmed Resistance / Electrical Angle / Retrim Rate are the four panels.
- Per-track drift detection. Drift is per-model.
- Western Electric rules 2–8 (e.g., "8 consecutive on one side of center"). Rule 1 only — single point beyond UCL/LCL is the violation criterion for v1.

---

## Design

### 1. Layout

The existing 2×2 grid and the existing header bar above it are unchanged from the 2026-05-08 spec. Only the *contents of each panel* are reworked.

```
┌──────────────────────────────────────────────────────────────┐
│ 7965 · ↑ DRIFTING · 16d · score 12.4 / 5.0 · 1,101 units    │
├──────────────────────────────────┬───────────────────────────┤
│ Sigma Drift                      │ Untrimmed Resistance      │
│ (hybrid SPC: dots + smoothed)    │ (smoothed + band)         │
├──────────────────────────────────┼───────────────────────────┤
│ Electrical Angle                 │ Retrim Rate               │
│ (smoothed + band)                │ (smoothed + band)         │
└──────────────────────────────────┴───────────────────────────┘
```

### 2. Per-panel anatomy

Every panel has:

- **Title row:** metric name + status pill.
- **Subtitle row:** `baseline → recent (Δ%) · z=±X.X`, one short line. For the Retrim Rate panel: `baseline → recent (Δ pp) · baseline N%` (percentage points; z-score is replaced by absolute baseline rate for plain-language reading).
- **Chart area:** ~140px tall, x-axis = bucket index (with N labels at corners), y-axis = metric value. No tick clutter; emphasize the trend.

**Status pill** in the header summarizes the panel's state. Pill color follows the line color so the chart and pill agree.

| Pill | Color | Condition |
| --- | --- | --- |
| `✓ stable` | green | `|z_score| < 2.0` (or for sigma, no UCL/LCL crossings in window) |
| `↑ DRIFT` / `↓ DRIFT` | orange | `2.0 ≤ |z_score| < 3.0` |
| `↑ OOC` / `↓ OOC` | red | `|z_score| ≥ 3.0` OR (sigma only) any unit crosses UCL/LCL |
| `↑ rising` / `↓ falling` | orange | retrim rate only — recent rate ≥ 2× baseline AND recent ≥ 10% |
| `○ no baseline` | gray | insufficient data; see Empty states |

**Reference lines** (matplotlib `axhline` / `axvline`):

- Gray horizontal at `baseline_mean` (all panels) — "this is where the metric used to live."
- Orange vertical at `baseline_cutoff_date` (all panels) — "everything left of this is baseline; everything right is recent."
- Red dashed horizontal at UCL and LCL (sigma panel only) — from `detector.get_control_limits()`.
- Solid horizontal at center (sigma panel only) — from the detector's center value.

### 3. Process panels (Resistance, Angle, Retrim Rate)

Each renders the same shape:

- **Smoothed mean line** — one point per adaptive bucket (see §5). Color = pill color.
- **Confidence band** (filled, alpha ≈ 0.15) — `mean ± SE`, where `SE = bucket_stddev / sqrt(bucket_n)`. High-N buckets get tight bands; low-N get loose bands. This visually communicates statistical confidence without a separate widget.

**Retrim Rate panel — derivation:**

```
per bucket: retrim_rate = (count of tracks where trim_pass_count > 1) / bucket_n × 100
baseline retrim rate = same calc over the baseline window
recent retrim rate = same calc over the recent window
```

The retrim rate panel has a fixed y-axis lower bound of 0%. Upper bound is `max(20%, 1.5 × peak observed)` so the panel doesn't auto-zoom into noise when the rate is consistently low.

### 4. Sigma panel (hybrid SPC)

Same axes and reference lines as the process panels, plus:

- **Individual unit dots** — one dot per track in window, x = bucket index it falls in (jittered slightly to avoid stacking), y = sigma_gradient value.
- **Dot color** — green if `LCL ≤ value ≤ UCL`, red if outside. (Western Electric rule 1 only.)
- **Smoothed mean overlay** — same smoothed line as the process panels, drawn on top of the dots in white (or off-white) for contrast.
- **Header pill** — `↑ OOC · N violations` where N = count of dots outside UCL/LCL in the entire window. If N = 0, the normal pill scheme applies (`✓ stable` / `↑ DRIFT` from z-score on the smoothed mean).

If the detector has no baseline yet, control limits and the smoothed line are both absent; only raw dots render, all gray. The empty-state caption (§6) takes precedence over the dot rendering in that case.

### 5. Adaptive bucketing

Bucketing happens **in the GUI layer**, not the DB. The query (`get_model_drift_dashboard`) already returns rows ordered by file_date.

Algorithm:

```
N_PER_BUCKET = 50   # default; tunable constant, not user-facing
sort rows by file_date ascending
walk rows; emit a new bucket every N_PER_BUCKET rows
final bucket may contain < N_PER_BUCKET rows; render anyway if ≥ 5, else fold into the previous bucket
```

For each bucket, compute and store:

- `bucket_index` (0..K-1)
- `n` (number of rows; shown as label at the bucket's x-coordinate corner)
- `mean`, `stddev`, `se = stddev / sqrt(n)` (for the smoothed line + band)
- `min_date`, `max_date` (for hover tooltips, future enhancement)
- `retrim_count` (panel-specific; for retrim rate calc)

`N_PER_BUCKET = 50` is the default; we'll surface it later as a Settings constant if real-world usage shows it needs per-model tuning.

### 6. Empty states

Each panel handles missing data independently. Drawing rules:

| Condition | Panel renders |
| --- | --- |
| No rows in window | Text only: `"No data in window"` — no chart. |
| Rows in window but only one bucket forms (n < 5 total) | A single large dot at the metric value, plus the numeric value. No line, no band. |
| Sigma panel, no baseline (`detector.has_baseline == False`) | Text only: `"Need ≥30 baseline samples to enable SPC; have N. Train this model in Settings."` Other three panels still render normally. |
| Retrim Rate panel, all rows in window have `trim_pass_count is None` (pre-feature data) | Text only: `"trim_pass_count not captured for this window's rows. Re-parse to populate."` Other three panels still render normally. |
| Any panel, `baseline_mean is None` (i.e., baseline window empty) | Render the smoothed line for the recent window; suppress the gray baseline reference line and the subtitle's `baseline →` half (read `— → recent (—)`). |

Empty states are silent failures — they never crash the parent dashboard, never raise, never leave a half-rendered chart.

---

## Data layer

The existing `DatabaseManager.get_model_drift_dashboard(model, days_back)` (manager.py:7757) already returns:

```python
{
    "model": str,
    "unit_count": int,
    "sigma_series": List[Tuple[iso_date, float]],
    "process": {
        "untrimmed_resistance": {
            "series": List[Tuple[iso_date, float]],
            "baseline_mean": Optional[float],
            "recent_mean": Optional[float],
            "delta_pct": Optional[float],
            "z_score": Optional[float],
        },
        "measured_electrical_angle": { ... same shape ... },
        "trim_pass_count": { ... same shape ... },
    },
}
```

**Required additions, NOT breaking changes:**

1. Each entry in `series` keeps its `(iso_date, float)` shape. Bucketing happens in the GUI; the query stays per-track.
2. **Add `baseline_cutoff_date: Optional[str]`** to the top-level dict — needed for the orange vertical line on all four panels. Definition: `now - recent_days` (the existing `recent_cutoff` value already computed at manager.py:7770). This is also what the `DriftDetector.baseline_cutoff_date` returns for trained models; using the query's recent-window cutoff guarantees all four panels show the same vertical line even when the detector has no baseline.
3. **Add `process.retrim_rate_series`** to the dict (alongside `process.trim_pass_count`). Shape: `List[Tuple[iso_date, int_0_or_1]]` — one tuple per non-NULL row, where the int is `1` if that row had `trim_pass_count > 1` else `0`. The GUI buckets these and computes `mean × 100` to get the retrim rate %. Keeps the existing `trim_pass_count` series untouched (still used by the All-Models Process Drift table).

No new tables, no schema migration. The `trim_pass_count` column already exists; the data layer just exposes its `> 1` counts.

---

## Renderer changes

`src/laser_trim_analyzer/gui/pages/trends.py:_render_single_model_drift` (lines ~2723–2880) is rewritten. Roughly:

- A helper `_compute_buckets(series, n_per_bucket=50)` returns a list of `Bucket(mean, stddev, se, n, bucket_index, min_date, max_date)` dicts.
- A helper `_draw_smoothed_panel(ax, buckets, baseline_mean, baseline_cutoff, status, color)` draws the process-panel rendering.
- A helper `_draw_sigma_panel(ax, sigma_rows, detector, baseline_cutoff)` draws the hybrid SPC chart with dots + smoothed overlay.
- A helper `_draw_empty_state(ax, message)` replaces the existing inline empty-state code.

`get_model_drift_dashboard` gains the additions in §Data layer.

No changes to `ChartWidget` (`chart.py`). The renderer uses matplotlib axes directly, as it already does.

---

## Out of scope

- Hover tooltips on the panels (future polish — bucket date range, exact value).
- Drill-down from a single bucket to the units inside it (future feature).
- Per-model `N_PER_BUCKET` tuning UI (default constant for now).
- Western Electric SPC rules 2–8 (only rule 1 — single point beyond UCL/LCL — flags out-of-control in v1).
- Smoothness metric on the dashboard (separate concern; lives in the Smoothness page).
- Backfilling `trim_pass_count` on existing DB rows — that's a user re-parse, not a code change.

---

## Pointers to existing code

- Renderer to replace: `src/laser_trim_analyzer/gui/pages/trends.py:_render_single_model_drift` (lines 2723–2880)
- Data query to extend: `src/laser_trim_analyzer/database/manager.py:get_model_drift_dashboard` (lines 7757–7820)
- Drift detector control limits: `src/laser_trim_analyzer/ml/drift_detector.py` — `DriftDetector.get_control_limits()` returns `(lower, center, upper)`; `has_baseline`, `baseline_cutoff_date`, `cusum_pos`, `cusum_neg`, `cusum_h`, `drift_direction`, `is_drifting` attrs all already used by the current renderer.
- Empty-state helper: `_draw_empty_state` already exists in trends.py (reused as-is for the panel-level "no data in window" case).
- Metric metadata (units, format strings): `DatabaseManager._PROCESS_DRIFT_METRICS` — the Retrim Rate panel adds its own entry rather than reusing the `trim_pass_count` one (different unit, different format).

---

## Validation

After implementation, the redesign is correct iff:

1. Opening the drift dashboard for any model with ≥50 units in the day-range window renders all four panels with visible smoothed trend lines — no noise carpets.
2. The Retrim Rate panel renders a meaningful curve for any model with retrims in the window, and a clean empty-state message when `trim_pass_count` is uniformly NULL.
3. The Sigma panel shows red dots wherever a real unit crossed UCL or LCL, and the count in the header pill matches the count of red dots.
4. Switching the day-range filter rebuckets and re-renders within ≤500ms for a typical 1000-unit model.
5. Switching between two different models (rapid clicking) does not leave stale axes from the prior model on screen (matches the existing generation-counter guard at trends.py:2699).
