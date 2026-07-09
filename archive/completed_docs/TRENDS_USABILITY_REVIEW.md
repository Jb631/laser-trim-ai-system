# Trends Page — Visual Usability Review

**Reviewer:** Claude (automated visual audit)
**Date:** 2026-05-07
**Branch:** fix/critical-26
**Database:** ~78,850 records, 232 active models, data spanning 2015–2026

---

## Executive Summary

Three charts are broken at production data scale and need immediate fixes: **Cpk by Model** has completely illegible y-axis labels with 60+ overlapping model names; **Drift Detection Timeline** overplots every marker at a single x-position because dates are treated as strings; and the **Heat Map** is unreadable with 232 rows of tiny cells and overlapping numbers. The remaining charts — Standard P-chart, Comparative, and Yield — are functional but have moderate issues around data density at longer time ranges and silent model caps. Trim Difficulty has no data, which is expected (requires reprocessing with `trim_pass_count`).

---

## Charts by Severity

### 1. Cpk by Model — BROKEN (Critical)

**Can I read this in 2 seconds?** No. The chart is visually impenetrable.

**What it should answer:** "Which models have the worst process capability, and which are below the Cpk thresholds?"

**Specific problems:**

- **Y-axis labels are a wall of overlapping text.** With 60+ models displayed, model names stack on top of each other and become completely illegible even when zoomed in. You cannot determine which bar belongs to which model.
- **Extreme outlier crushes the x-axis scale.** One model has a Cpk near 130, which compresses all other bars (most are 0–10) into a narrow sliver at the left edge. The Capable (1.33) and Minimum (1.0) reference lines are invisible — they sit at x ≈ 1 but the axis runs to 130.
- **No value labels on bars.** Even if you could read the model names, you can't read the Cpk values.
- **Date range doesn't help.** Switching from "All Time" to "Last 90 Days" produces the exact same illegibility — the problem is row count, not date filtering.

**Recommended fix:** Cap display to the bottom 15–20 models (worst Cpk), clip x-axis at 10 or use log scale, and add value annotations on each bar. Offer "Show all" as an opt-in.

---

### 2. Drift Detection Timeline — BROKEN (Critical)

**Can I read this in 2 seconds?** No. Every marker sits at a single x-position.

**What it should answer:** "When did each model start drifting, and is drift getting better or worse over time?"

**Specific problems:**

- **All markers overplotted at one date (2026-04-15).** The x-axis shows a single date string, and every model's drift markers (red up-triangles for degrading, yellow down-triangles for improving) are stacked vertically at that one position. The chart looks like a single column of triangles instead of a timeline.
- **Root cause (confirmed from code review):** Dates are stored/plotted as strings, so matplotlib treats them as a single categorical value rather than a temporal axis. Even if there were multiple detection dates, they'd be ordinal categories with no proportional spacing.
- **No magnitude information.** The markers show direction (up/down) but not drift severity. A model drifting 0.1σ looks the same as one drifting 5σ.

**Recommended fix:** Parse detection dates as `datetime` objects so matplotlib creates a proper time axis. Add marker size or color intensity to encode drift magnitude.

---

### 3. Heat Map (All Models, All Time) — BROKEN (Critical)

**Can I read this in 2 seconds?** No. It's a solid wall of tiny colored cells.

**What it should answer:** "Which models are failing on which metrics, and where are the hot spots?"

**Specific problems:**

- **232 model rows make individual cells microscopic.** Each cell is roughly 3 pixels tall, far too small to distinguish colors or read annotations.
- **Overlapping cell annotations.** The numbers printed inside each cell overlap with adjacent rows, creating an unreadable mess of digits.
- **No scroll or pagination.** The entire 232-row heatmap is crammed into a single chart viewport.

**Recommended fix:** Show only the top/bottom 20 models by a selectable metric (e.g., fail rate). Add a model search/filter. Remove cell annotations at high row counts or switch to a larger font with fewer rows.

---

### 4. Yield Trend (All Models, All Time) — Moderate

**Can I read this in 2 seconds?** Partially. The overall shape is visible but individual data points blur together.

**What it should answer:** "Is our overall yield improving or declining over time?"

**Specific problems:**

- **Too many weekly data points over 10+ years.** The x-axis has hundreds of weekly labels (2015-W01 through 2026-W15) that overlap heavily. The data becomes a noisy scatter rather than a clear trend.
- **At 90 days, only a single data point appears.** The weekly aggregation produces just one point (2026-W15) at ~0% yield, which isn't useful. This suggests the date filter interacts poorly with the weekly bucketing — possibly only one week has data meeting the "Min samples: 20" threshold in the last 90 days.
- **No smoothing or trend line.** A moving average or LOESS curve would help extract the signal from weekly noise.
- **Y-axis goes 0–100% regardless of data range.** If actual yield is 50–65%, the chart wastes half the vertical space.

**Recommended fix:** Add a rolling average trend line (e.g., 12-week MA). Auto-scale y-axis to data range with padding. At long time ranges, switch from weekly to monthly aggregation.

---

### 5. Comparative (All Models) — Minor

**Can I read this in 2 seconds?** Yes, for the models shown.

**What it should answer:** "How do selected models compare on key metrics side-by-side?"

**Specific problems:**

- **Silent 4-model cap.** The "All Models" dropdown implies all 232 models would be compared, but only 4 are shown. There's no indication that a limit was applied or how the 4 were selected.
- **Radar chart can become cluttered.** At the 4-model cap it works fine, but if the cap were raised the polygons would overlap.

**Recommended fix:** Show a clear message like "Showing top 4 models by sample count (of 232)" or allow user selection. The cap itself is a reasonable design choice — just communicate it.

---

### 6. Standard P-Chart — Single Model (e.g., 8340-1) — Good

**Can I read this in 2 seconds?** Yes.

**What it should answer:** "Is this model's pass rate stable or is it trending out of control?"

**Specific problems (minor):**

- Control limits (UCL/LCL) and trend line are present and readable.
- Date labels can overlap at "All Time" but remain functional.
- Good use of color coding (green for pass, red for fail points).

**Recommended fix:** None critical. Could auto-thin x-axis labels at long date ranges.

---

### 7. Standard — All Models Summary Cards — Good

**Can I read this in 2 seconds?** Yes.

The "Models Requiring Attention" table and "Top Performing / Recent Issues" summary cards are clean and scannable. Sigma values, model names, and trend indicators are clearly visible.

**Recommended fix:** None.

---

### 8. Trim Difficulty — No Data (Not Scorable)

**Can I read this in 2 seconds?** N/A — shows "No trim difficulty data yet."

**What it should answer:** "Which models are hardest to trim successfully?"

**Specific problems:**

- The empty state message is accurate ("trim_pass_count is captured at parse time — process new files or reanalyze existing ones to populate this view") but doesn't give a one-click path to reprocess.
- This is expected behavior, not a bug — the feature requires data reprocessing.

**Recommended fix:** Add a "Reanalyze All" button or link to the Process page from the empty state.

---

## Cross-Reference: Code Review Findings

| Code Review Finding | Visually Confirmed? | Notes |
|---|---|---|
| Heat Map illegible at scale | **Yes** | 232 rows, cells ~3px tall, overlapping annotations |
| Sigma scatter uses ordinal index | Partially | Not directly tested (Standard scatter not isolated), but the P-chart x-axis is date-based and works |
| Comparative 5-model cap | **Modified** — 4-model cap observed | Only 4 models shown, not 5. Silent cap confirmed. |
| Cpk missing labels / x-limit | **Yes** | Labels overlap completely, x-axis runs to 130 with no clipping |
| Drift string dates overplotting | **Yes** | All markers at single x-position (2026-04-15) |

---

## Priority Fix Order

1. **Cpk by Model** — Cap rows to 15–20, clip x-axis, add value labels
2. **Drift Timeline** — Convert date strings to datetime objects
3. **Heat Map** — Cap rows to 20, add filter/search
4. **Yield Trend** — Add rolling average, auto-scale y-axis
5. **Comparative** — Add "showing N of M" label
