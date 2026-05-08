# Trends Page — 3-Tab Consolidation

**Date:** 2026-05-07
**Branch:** `main` (post-merge of `fix/critical-26`)
**Status:** Design — awaiting user review

User feedback after running the redesigned Trends page: "the priorities tab is all messed up" and "i'm still not sure we need all those tabs" (currently 7). This spec consolidates to 3 tabs and rebuilds the Priorities content as native widgets to fix the visual breakage.

## Context

The current Trends segmented button has 7 entries — Priorities, Standard, Comparative, Cpk Trend, Drift, Process Drift, Trim Difficulty — accumulated across this branch's many feature commits. The audit committed at `docs/superpowers/specs/2026-05-07-trends-cleanup-final-design.md` already addressed redundant *content* but didn't reduce the tab count.

Two things broke when the user actually opened the app:

1. **Priorities Section 1 ("Focus This Week")** is rendered as `ax.text(...)` blob on a borderless matplotlib axis. At GUI window sizes the monospace text wraps and clips badly, looking unfinished.
2. **Cognitive load.** Seven tabs is too many for a page where each individual user spends most of their time on one or two views.

## Out of scope

- DB-layer changes. All consolidation is GUI-only. The DB methods that back the removed tabs (`get_cpk_by_model`, `get_comparative_model_trends`) stay because the executive Excel export still calls them.
- Renaming tabs. Keeping "Standard" / "Drift" / "Trim Difficulty" verbatim — renames create churn for no functional benefit and break operator muscle memory.
- Standard *detail* view (when a specific model is selected). The sigma scatter + P-chart + sigma distribution histogram trio is the user's confirmed primary tool. Untouched.

## Final tab structure

3 tabs in the segmented button: **Standard** · **Drift** · **Trim Difficulty**

### Standard tab — All Models view

Top to bottom in the scrollable content frame:

1. **Active Models Summary** stats row — unchanged. Active models / total samples / pass rates / Top Anomaly Model.
2. **Focus This Week** — top 5 priority models from `get_linearity_prioritization`. Rebuilt as a `CTkFrame` containing 5 `CTkLabel` rows (NOT matplotlib `ax.text`). Each row: rank · model · fail rate (color-coded) · `→ recommendation` text. This is the fix for "all messed up".
3. **Failure Severity** — bar chart, four buckets (1-3 / 4-10 / 11-50 / 50+ fail points). Same `_render_priorities` Section 2 chart, lifted onto Standard summary as its own `ChartWidget`.
4. **Cost Impact** — horizontal bar chart, top 15 models by estimated scrap cost. Same `_render_priorities` Section 3 chart, lifted onto Standard summary.
5. **ML Status** — one-line text: `134 trained · 3 drifting (see Drift tab)`. Same as today.

Removed from Standard summary:
- Heatmap (`alerts_chart` for "Models Requiring Attention" — Focus This Week replaces it)
- Heatmap chart (the model × week pass-rate matrix — user dislikes it)
- Comparative pass-rate trends chart (was on a separate tab; not migrated)
- "Where to Focus →" pointer button (the content is now native, no need to point elsewhere)
- Cpk-by-model bar chart (was on Cpk Trend tab; not migrated)

### Standard tab — single-model detail view

Unchanged. Sigma scatter, P-chart, sigma distribution histogram. No Cpk trend added (per user choice C — drop Cpk live view entirely).

### Drift tab

Two-button toggle at the top of the content frame: `[ ML Drift | Process Drift ]`. Default to ML Drift on first show.

- **ML Drift panel** — current Drift timeline content (CUSUM/EWMA chart per model, model list on the left). Implementation lifted unchanged from current `_show_drift_timeline` / `_render_drift_timeline`.
- **Process Drift panel** — current Process Drift three-stack content (untrimmed resistance / electrical angle / trim_pass_count z-score bar charts). Implementation lifted unchanged from current `_show_process_drift` / `_render_process_drift`.

Toggling between buttons calls the appropriate `_show_*` method. Each render method's "still on this tab?" guard (added in commit `4231a9b`) needs to be updated to also check the toggle state, not just `self._trend_type.get() == "Drift"`.

### Trim Difficulty tab

Unchanged. Bar chart with avg-error-reduction annotation per model.

## Removals

Methods to delete from `gui/pages/trends.py`:

- `_show_priorities`, `_render_priorities` — content moves into Standard summary directly
- `_show_comparative_trends`, `_render_comparative_trends` — chart not migrated
- `_show_cpk_trend`, `_render_cpk_trend` — chart not migrated
- `_show_process_drift` — kept but called from the new toggle, not from `_on_trend_type_changed`
- `_render_process_drift` — kept, signature unchanged
- The "Where to Focus" pointer-button frame (`_impact_frame`) and its label — replaced by the native Focus This Week widget rows
- The model heatmap section (`_heatmap_frame`, `heatmap_chart`, `_update_heatmap`) — deleted; chart widget's `plot_heatmap` method itself stays in `widgets/chart.py` (no other callers, but harmless and `chart.py` is shared infrastructure)

Segmented button `values=` shrinks from 7 to 3.

`_on_trend_type_changed` simplifies — only Standard / Drift / Trim Difficulty branches plus the no-op safety net for legacy stored selections.

## Drift toggle implementation

New widget on Drift tab:

```
┌────────────────────────────────┐
│ Drift Detection                │
│                                │
│ View: [ ML Drift  ] [ Process ]│  ← CTkSegmentedButton
│       (active     ) (inactive )│
├────────────────────────────────┤
│                                │
│  (panel renders below — one    │
│   of the two views at a time)  │
│                                │
└────────────────────────────────┘
```

State:
- `self._drift_subtab: str` — `"ML Drift"` or `"Process Drift"`. Default `"ML Drift"`.
- `_on_drift_subtab_changed(value)` — sets `self._drift_subtab`, then calls `_show_drift_timeline()` or `_show_process_drift()`.

The "still on this tab?" guard inside `_render_drift_timeline` and `_render_process_drift` needs to check `self._trend_type.get() == "Drift" and self._drift_subtab == <expected>`. Otherwise a stale render fires after the toggle switches and destroys the new view's chart.

## Focus This Week widget rebuild

Currently:
```python
ax1.text(0.01, 0.98, "\n".join(lines), transform=ax1.transAxes,
         ha="left", va="top", fontsize=10, color="#dddddd",
         family="monospace")
```

This packs all 5 priority models into a single matplotlib text annotation on a hidden axis. At GUI sizes the wrapping is unpredictable.

New approach — actual GUI widgets:

```python
focus_frame = ctk.CTkFrame(self.content)
ctk.CTkLabel(focus_frame, text="Focus This Week", ...).pack(...)

for i, p in enumerate(priorities[:5], start=1):
    row = ctk.CTkFrame(focus_frame, fg_color="transparent")
    row.pack(fill="x", padx=15, pady=4)
    # rank, model, fail rate (color-coded), recommendation as separate
    # CTkLabels packed side="left" — proper text rendering, proper
    # wrapping, proper truncation if needed.
```

Color coding: fail rate ≥30% red, ≥15% orange, ≥5% amber, else default. Recommendation text always renders on a second line beneath the main row (consistent alignment regardless of recommendation length). When the model has no recommendation, the second line shows "—" in gray rather than collapsing — keeps row heights uniform across the 5 entries so the visual rhythm is stable.

The `priority_models` data comes from the existing `_load_summary_data` background fetch, no new DB query needed. The widget rebuild lives inside `_update_summary_display`, called the same way the existing alert / best / worst widgets are.

## Files affected

| File | Change |
|---|---|
| `src/laser_trim_analyzer/gui/pages/trends.py` | Major. ~700-line restructure: deletions, toggle widget, Focus This Week rebuild, segmented button shrink. |
| `docs/superpowers/specs/2026-05-07-trends-3-tab-consolidation-design.md` | This spec. |

No backend, no schema, no test changes.

## Testing

Manual verification — pure UI restructure with no logic changes. Smoke-import all GUI pages after the change. Click each of the 3 tabs and confirm they render without `TclError`. Click the Drift toggle in both directions and confirm both sub-views render correctly (the stale-render guard from commit `4231a9b` needs updating per the Drift toggle section above).

No new automated tests. Existing tests (`test_anomaly_rate_by_model.py`, `test_trim_difficulty_avg_error_reduction.py`) cover the DB methods that aren't being touched.

## Success criteria

- Trends segmented button shows exactly 3 entries: Standard, Drift, Trim Difficulty.
- Standard summary view shows: stats row → Focus This Week (5 widget rows) → Failure Severity bar chart → Cost Impact horizontal bars → ML status one-liner. No heatmap, no Comparative chart, no Cpk-by-model bars, no pointer button to a deleted Priorities tab.
- "Focus This Week" rows render as proper text (no clipping, no wrapping artifacts) at typical window sizes.
- Drift tab has a sub-toggle. Clicking each toggle button renders the corresponding view (ML Drift timeline OR Process Drift z-score panels) without leaving stale chart state from the other view.
- Trim Difficulty tab unchanged.
- All 11 existing tests still pass.
- All 6 GUI page imports succeed.
- No `TclError` when navigating between tabs in any order, including switching while a background load is in flight.
