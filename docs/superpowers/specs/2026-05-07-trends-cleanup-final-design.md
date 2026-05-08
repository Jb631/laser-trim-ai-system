# Trends/Dashboard Cleanup — Final Pass

**Date:** 2026-05-07
**Branch:** fix/critical-26
**Status:** Design — awaiting user review

This spec covers the four remaining items from the app-wide UX audit that closed out the Trends-page redesign work on PR #3. ML capability work is tracked separately (see the parallel ML audit, in flight).

## Context

Earlier in this branch we already addressed the high-impact audit items:

- New Priorities tab (focus list, near-miss split, cost impact)
- New Process Drift tab (untrimmed resistance, electrical angle, trim passes)
- Failure Margin block, ML probability badge, Worst Zone in Analyze
- Yield tab merged into Dashboard P-chart with date range
- Removed duplicate Where-to-Focus and Drift textboxes

These four remaining items are the long tail: smaller surface-level fixes, all backed by signals already stored on every track but underused.

## Out of scope

- Anything ML-related (covered by the parallel ML audit's recommendations)
- Compare and Process page audits (deferred — the user explicitly chose to focus on remaining audit items first)
- Aggregate `max_error_reduction_percent` chart was originally proposed in option B but was rolled into this scope at user request (option A)

## Items

### 1. Per-model anomaly rate

**Problem.** `is_anomaly` is flagged per-track and shown via a `!` indicator on Analyze list items, but never rolled up. A model with 12 anomalies in 30 days is a setup-issue red flag and is currently invisible at the model level.

**Change.**

- New DB method `DatabaseManager.get_anomaly_rate_by_model(days_back=90, min_samples=10)`. Returns `[{model, total_tracks, anomaly_count, anomaly_rate, last_anomaly_date}]` joined from `analysis_results` × `track_results` filtered by `file_date` window. Sorted by `anomaly_rate` descending.
- Surface on Trends > Standard summary view's "Active Models Summary" stats. Add an `anomaly_count` column to the existing per-model table, not a new chart.
- Color-code the cell: amber when `anomaly_rate >= 5%`, red when `>= 15%`. Below 5% renders default style.

**Files touched.** `database/manager.py`, `gui/pages/trends.py`.

### 2. Dashboard Pareto cleanup

**Problem.** Dashboard row 3 has both a "Failure Pareto" chart (column 1) and a "Where to Focus" panel of cards (column 2). Both rank models by failure impact; cards already include count plus a recommendation string. The Pareto chart adds no information the cards don't already convey.

**Change.**

- Delete the Pareto frame and its rendering helpers. Specifically `_pareto_frame`, `_pareto_placeholder`, `pareto_chart` widget references and any `_update_pareto_*` calls.
- Re-grid the Where-to-Focus panel to span columns 1+2, giving the cards roughly twice the width they have today. Card text was cramped before.
- The DB call `get_priority_models` (or equivalent) that fed the Pareto stays — the cards still need that data.

**Files touched.** `gui/pages/dashboard.py` only.

### 3. `deviation_uniformity` in the Failure Margin block

**Problem.** `deviation_uniformity` (coefficient of variation of absolute errors) is computed and stored on every track but never displayed. Low uniformity on a *passing* track means errors are concentrated in one region of the element — a heads-up that the unit is mechanically asymmetric and likely to fail soon.

**Change.**

- Add one line to the Failure Margin block in `analyze.py:_display_metrics`, between the margin/violation lines and the ML failure probability line.
- Format: `Error uniformity: {dev_unif:.2f}  (1.0 = uniform across track, ≥1.5 = concentrated in one region)`
- Hidden if value is `None`.

**Files touched.** `gui/pages/analyze.py` only.

### 4. `max_error_reduction_percent` aggregate on Trim Difficulty

**Problem.** The Trim Difficulty tab annotates each model bar with `{count} units · max {max_passes} · retrim {retrim_rate}%`. Adding average `max_error_reduction_percent` lets the operator distinguish two failure patterns: high passes + high error reduction = process is succeeding at hard trims (working as intended); high passes + low error reduction = retrimming isn't actually helping (process root-cause issue).

**Change.**

- Extend `DatabaseManager.get_trim_difficulty_by_model()` to also pull `func.avg(DBTrackResult.max_error_reduction_percent)` per model. Append `avg_error_reduction` to each row dict (may be `None` if no data).
- Update the bar annotation in `_render_trim_difficulty` to: `{count} units · max {max_passes} · retrim {retrim_rate}% · avg Δ {avg_error_reduction:.0f}%`. Skip the trailing field if value is `None`.
- No layout change beyond annotation text length.

**Files touched.** `database/manager.py`, `gui/pages/trends.py`.

## Implementation order

1. Item 3 — smallest, isolated, validates the spec format
2. Item 1 — DB method + UI surface
3. Item 4 — DB method extension + chart annotation tweak
4. Item 2 — UI restructure, biggest visual change, lowest risk if done last

Each item ships as its own commit so the PR diff stays reviewable.

## Testing

All four items are GUI-side and rely on data already in the schema. Verification approach:

- Item 1: spot-check DB method output in a Python REPL against a known model with anomalies.
- Item 2: visual diff of Dashboard before/after.
- Item 3: open Analyze on a unit with known `deviation_uniformity` and confirm the line appears with the expected value.
- Item 4: open Trim Difficulty tab and confirm the new annotation field renders without breaking the layout.

No new test files needed for this scope.

## Success criteria

- Trends Standard summary shows anomaly counts/rates per model with color coding.
- Dashboard "Where to Focus" panel has visibly more horizontal space.
- Analyze metrics tab shows `Error uniformity: X.XX` line for tracks where the value is computed.
- Trim Difficulty bar annotations include `avg Δ N%` field where data exists.
- All four changes ship in distinct, reviewable commits on `fix/critical-26`.
