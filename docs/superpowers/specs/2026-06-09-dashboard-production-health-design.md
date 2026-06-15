# Spec — Dashboard "Production Health" + Model-page fixes (V6)

**Date:** 2026-06-09
**Status:** Design approved by James 2026-06-09. Drives an implementation plan next.
**Branch:** `V6` (long-lived; not merged to main — Graduation still gated).
**Builds on:** `docs/superpowers/plans/2026-06-01-spec3-rewrite-foundations.md` (the V6 contracts: PageBase, V6App, theme tokens, threading §2.4, QA rules §3). All of those apply here verbatim.

This work came out of James's V6 smoke test. It adds a **Dashboard** landing page and fixes three **Model-page** defects he found. Both ship together.

---

## 0. Mission fit (why these widgets earn their place)

The mission (foundations §0): *which models' element production is drifting, how bad, what evidence.* Three jobs: find drift, show evidence, feed the system.

- **Triage** already answers "what's drifting today?".
- **Dashboard** answers the broader **"how is the line doing?"** — production yield — and **routes the worst models into the Model evidence view**. The clickable worst-yield list is the bridge that keeps the Dashboard serving *find-drift → show-evidence* rather than being a dead stats wall. A yield panel with no path to action would not ship.
- The Model fixes restore the **show-evidence** quality James expects (right metric on open, real baseline-vs-recent numbers, a readable table).

---

## Part 1 — Dashboard page ("Production Health")

### 1.1 Placement & landing
New page `"dashboard"`. Sidebar order becomes **Dashboard · Triage · Process · Model · Settings**. `V6App` opens on Dashboard (`show_page("dashboard")` replaces the current `show_page("triage")` at startup). Triage stays one click away. `DashboardPage(PageBase)`, `page_title = "Dashboard"`.

### 1.2 Header
A time-window selector (`CTkOptionMenu`): **30d / 90d / 365d / All**, default **90d** — same `_WINDOW_DAYS` map and pattern the Model page already uses. Changing it reloads.

### 1.3 Two yield panels (side by side) — `YieldPanel` widget
A reusable `YieldPanel` rendered from a stats dict:
- large **pass-rate %** headline,
- **Pass / Warning / Fail** counts,
- a total line (units for trim; "N matched records" for FT, which discloses partial coverage),
- a compact **pass-rate-over-time trend** line (**per-day buckets** across the window, the shape `get_dashboard_stats` already produces — "is the line getting worse?").

**Left = Trim analysis yield** (from `analysis_results.overall_status`).
**Right = Final-test yield** (from `final_test_results.overall_status`).

**Pass-rate definition (explicit, transparent, adjustable):** `pass_rate = Pass / (Pass + Warning + Fail)`. A **Warning is not a clean pass** (it failed sigma or linearity), so it counts against. **Error and Untrimmed are excluded from the denominator** (Untrimmed = test-sweep, ungradeable; Error = processing failure). The raw breakdown is always shown next to the %, so the number is never a black box. If the denominator is 0 → empty state, never a fabricated `0%`/`—` masquerading as measured (Q9).

### 1.4 Lowest-yield models — `WorstModelsList` widget
A table below the panels: **model · units · trim pass-rate · FT pass-rate** (FT shows `—` when no matched FT records). Sorted **worst trim pass-rate first**. Only models with **≥ 5 units in the window** are ranked (so a 1-unit 0% model can't top the list); the min-sample guard is stated in the UI. Capped at **top 10**, and if more qualify the cap is disclosed ("showing 10 of N", Q10). **Row click → `set_model_route(model)` + `show_page("model")`** — the actionable bridge to evidence.

### 1.5 Data layer (grounded, reused where it exists)
All reads run **off the Tk thread** and apply via `safe_after` (foundations §2.4); a **generation token** guards against rapid window changes overwriting a newer load (§2.4 rule 2). All rows **materialized to plain dicts inside the session** (I8).

- **Trim yield + trend:** reuse the existing `DatabaseManager.get_dashboard_stats(days_back, ...)` (manager.py:1674) — it already returns pass/fail counts, `pass_rate`, and per-day `pass_rate` buckets with UNTRIMMED excluded. Map `days_back` from the window (All → a large `days_back`, e.g. ~36500 days).
- **Final-test yield + trend:** a new parallel helper computing the same shape over `final_test_results` (group `overall_status`, bucket by `file_date`), plus the matched-record count for the coverage line.
- **Worst models:** a new per-model rollup — counts by status per model over the window → trim pass-rate (and FT pass-rate joined where present) → sorted ascending, min-units filtered, top-10.

The FT-yield and worst-models helpers are **pure functions** (db + cutoff in, dict/list out), unit-tested directly against a tmp DB like `list_known_models` — no Tk required. Exact module/method names are pinned in the implementation plan; they live beside the existing yield code (DatabaseManager or a small `core` analytics helper), reusing `get_dashboard_stats` rather than duplicating its logic.

### 1.6 Empty / edge states
- No data in window → "No production data in the last {window} — process files or widen the window." (Q9)
- FT table empty but trim present → trim panel populated, FT panel shows "No matched final-test records in this window."
- Worst-models list empty (everything ≥ threshold passing) → "All models above {threshold}% — nothing to flag."

---

## Part 2 — Model-page fixes (folded into this spec)

### 2.1 Open on the worst/flagged metric
**Problem:** opening a model (via selector, or a Triage row with no focus metric) shows a *static default* metric, so a model whose Trim-pass-count is +6.6σ Out-of-Control still opens on a calm "Sigma gradient" chart.
**Fix:** when there is **no routing focus**, default the selected metric to the model's **`status.worst_metric`** (if the model has any tier > Stable); otherwise fall back to the existing static default (`untrimmed_sigma_gradient`). A routing focus (Triage card deep-link) still wins. Track whether the user has **explicitly clicked a pill** for the current model so auto-selection never overrides a manual choice; reset that flag when the model changes.

### 2.2 Real "Recent" values (computed UI-side)
**Problem:** the Drift Metrics **Recent** column is empty for every metric. `model_metric_state` persists the baseline + CUSUM/EWMA accumulators but **no recent-window mean**, and `get_model_drift_status` feeds the hydrated detector no new data, so `MetricStatus.recent_mean` is always `None`.
**Fix:** the Model page **computes a recent-window mean per metric from the actual windowed data** (the same series the focus chart already loads) and fills the **Recent** column with it; the focus chart also draws a **recent-mean marker/line** so baseline-vs-recent is visible. **Δσ is left unchanged** — it is the detector's magnitude (σ over the tier threshold, foundations Q6); a Stable metric legitimately reads `+0.00`. We are filling the genuinely-missing Recent number, not re-defining Δσ.

### 2.3 Drift Metrics table alignment
**Problem:** header and each row are separate frames using `pack(expand=True)`, which sizes every cell to its own text + a share of slack, so columns drift apart.
**Fix:** rebuild `DriftMetricsTab` as **one grid** (header row + metric rows sharing the same parent and the same weighted `grid_columnconfigure`), cells **left-aligned**, so columns line up exactly. Row-click-selects-metric behavior is preserved.

---

## 3. Components & boundaries

| Unit | Purpose | Depends on |
|------|---------|-----------|
| `YieldPanel` (widget) | render one yield dict (%, counts, total, trend) | theme, FocusChart-style mini chart or a small line |
| `WorstModelsList` (widget) | render ranked rows, emit `on_row_click(model)` | theme |
| `DashboardPage` (page) | window selector + 2 panels + list; threaded reload | the two widgets, the data helpers, `app` |
| FT-yield helper (pure) | windowed yield dict over `final_test_results` | db |
| worst-models helper (pure) | per-model windowed rollup, sorted | db |
| Model-page edits | worst-metric default, UI recent mean, grid table | existing Model page + `DriftMetricsTab` |

Each widget is constructed under `tk_root` and driven by a `set_*` method (testable in isolation, like the existing 3b/3c widgets).

---

## 4. Testing

- **Pure helpers:** FT-yield (counts, pass-rate definition, windowing, empty); worst-models (sorting, min-units guard, cap, FT join). Against tmp DBs.
- **Widgets:** `YieldPanel.set_yield`, `WorstModelsList.set_rows` + row-click emits model — construction + render under `tk_root`.
- **DashboardPage:** `make_app` — page exists, is the landing, `reload_now()` populates from seeded data, worst-model row-click routes to Model (`current_page == "model"`, route set).
- **Model fixes:** opening a seeded model with a flagged metric selects `worst_metric`; recent mean computed for a known series; `DriftMetricsTab` builds on a grid (rows/header share columns).
- Full regression sweep (foundations §7) green, plus a new `tests/test_dashboard.py`.

---

## 5. Decisions (baked in, approved) & out of scope

**Decided:** yield only (not throughput/cost — those were declined); both trim + FT panels; trend lines included; clickable worst-models list included; Dashboard is the default landing.

**Out of scope:** throughput and cost/impact dashboards; unit-level dedup for the "units" count (count records; dedup is a later refinement); changing the detector to persist a recent mean (UI computes it instead); any Final-Test matching changes; anything touching V5 GUI files (Graduation stays gated).
