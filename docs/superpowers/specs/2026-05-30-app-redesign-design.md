# V6 App Redesign — Design Spec

**Date:** 2026-05-30
**Status:** Approved, ready for implementation specs
**Parent:** `docs/superpowers/specs/2026-05-14-app-mission-realignment-design.md` (anchor)
**Target branch:** `V6` (long-lived; main stays at V5 until V6 graduates)
**Source:** Brainstorming session with James 2026-05-30. Captures the umbrella design that ties Specs 1/2/3 together.

This spec describes the **whole-app redesign**: shell, navigation, page set, and the cross-cutting decisions that determine what each downstream feature spec must build. It does **not** specify per-feature implementation details — those live in Spec 1 (already drafted), Spec 2 (to be written), and Spec 3 (to be written).

---

## Mission (recap)

> *Which product models' element production is drifting (sudden lot shift or slow trend), how bad, and what evidence do I take to engineers and MEs?*

Every top-level item in the new app must directly serve one of:
1. **Find drift** — which models need attention right now?
2. **Show evidence** — how bad, what to hand engineers?
3. **Feed the system** — get new data in so questions (1) and (2) can be answered.

Anything that doesn't pass this test is either folded into a page that does pass, demoted to a tab/section, or removed.

## What changed from the anchor spec

The anchor doc's "Important Constraints — Things NOT to change" section listed Smoothness page and Specs page as constraints-to-preserve. This spec **revises both of those constraints** because applying the strict mission test to the top-level nav exposed them as parallel/admin workflows rather than mission-direct.

| Anchor doc constraint | V6 redesign decision |
|---|---|
| "Smoothness page is critical and stays. Not a candidate for the page audit." | Smoothness *functionality* is critical and stays. The standalone Smoothness page is folded into **Model** as a tab. Per-model smoothness investigation lives next to per-model drift investigation — same context, one place. |
| "Specs page" listed as kept | Specs *functionality* (per-model spec limits, exclude_points) stays. Specs is folded into **Settings** as a section. Admin work, not daily flow. |
| "Existing alert tier system stays. Cure for 'flags everything' is to tighten thresholds." | Unchanged. Tier taxonomy stays; Spec 2 tightens. |
| "Excel export and per-unit chart drill-down stay exactly as they are." | Unchanged. Triggered from Model row clicks. |
| "Per-unit failure predictor demoted to 'available but not headline.'" | Unchanged. Opt-in expandable panel on Model page. |
| "File ingest behavior (Process page) not in scope." | Unchanged. Process page ports verbatim into the new shell with restyled chrome. |

The anchor spec's resolved Q1 (V6 branch), Q2 (7 watched metrics), and Q3 (existing tiers + tighten thresholds) are unchanged. Q4 (page audit) — the resolution recorded in the anchor said "defer until new mission views exist" — is now resolved as part of *this* spec: 10 V5 pages → 4 V6 pages.

## Deployment model

- **`main`** continues to be the V5 deployment. Receives bugfixes and Spec 1 (`untrimmed_sigma_gradient` column). Existing pages and behavior unchanged for daily use at work.
- **`V6`** is the long-lived redesign branch. Receives Specs 2 and 3 plus this design doc. Not deployed until it graduates.
- **Sync rule:** `main` → `V6` merge-forward after every bugfix/Spec 1 work lands on main. Avoid `V6` → `main` until V6 is ready to replace V5.
- **Graduation:** when V6 demonstrably serves the mission better than V5, James decides when to swap. V5 in `main` history is preserved.

## Architecture

### Shell

- **Left sidebar navigation.** 4 fixed items + branding header. Active item highlighted with accent stripe.
- **Greenfield CSS / theme.** Existing widgets get restyled to match. No emoji used in production UI (the mockups used some for clarity during brainstorming); use Material-style icons or none.
- **Density:** medium. Charts get vertical real estate (sidebar nav doesn't eat header space).
- **Color palette / theming:** carry over the existing tier color semantics so Stable/Warning/Drift/Out-of-Control read consistently with V5 history; refresh the chrome (primary nav color, surface backgrounds) to feel like one app.

### Pages

#### 1. Triage — landing

**Purpose:** Answer "anything to look at today?" in under 10 seconds.

**Layout:** Two zones.
- **Top — "Needs attention" cards.** 3–5 model cards visible without scrolling, color-coded by tier (existing taxonomy: Stable / Warning / Drift / Out-of-Control). Each card shows: model, alert type (step change / slow drift), worst-metric name, magnitude (σ count or trend slope). Empty state when nothing is flagged: a brief affirmative ("All models within tolerance — last processing 2 hours ago").
- **Bottom — "All models" browse.** Search box + scrollable list of every model in the database. Tier color on each row but desaturated relative to the cards above. Click any model → Model page for that model. Search-as-you-type filters the list.

**Triage also surfaces smoothness alerts.** Because Smoothness lives on Model now, a model whose smoothness signal has drifted appears in the "Needs attention" zone with a "smoothness" alert-type label. Spec 2 owns the detector mechanism; Triage just consumes the alerts.

**Entry mode handling:**
- Page load on app open → Triage is the default landing.
- Returning from Model page → preserves the scroll position and any search filter.

#### 2. Process — file ingest

**Purpose:** Get trim, Final Test, and Smoothness data into the system.

**Layout:** Existing V5 Process page logic ports verbatim. Same incremental-mode toggle, same scan progress, same per-file status. Only the chrome (sidebar, header) is new. No feature changes.

#### 3. Model — per-model drift investigation

**Purpose:** Deliver "how bad, what evidence?" for one model.

**Layout:** Three regions stacked.

1. **Glance row (top).** Pill bar showing all 7 watched metrics + a Smoothness pill (8 pills total). Each pill is color-coded (Stable/Warning/Drift/Out-of-Control) and shows the metric name + a one-line summary (e.g., "+4.2σ", "OK 30d", "drift 14d"). Click any pill to swap the focus chart.
2. **Focus chart (middle, largest).** One large time-series chart for the selected metric. Defaults to the triggering metric when entered from Triage; defaults to `sigma_gradient` (or last-selected) when entered from search. Chart overlays baseline vs recent visually.
3. **Tabs (bottom).**
   - **Drift Metrics** (default) — the 8-pill grid with each metric's baseline-vs-recent stats and a small sparkline. Click any to swap into the focus chart slot above.
   - **Smoothness** — existing Smoothness page logic for this model only. Per-model smoothness trend, recent test result list, per-unit smoothness charts.
   - **Units** — list of units in the recent batch with row-click → existing per-unit pre/post-trim charts. Excel export button at top.

**Evidence export.** Sticky button in the header: "Copy summary" (text block with model + alert + baseline-vs-recent numbers, paste-ready for chat or email) and "Export evidence pack" (Excel with the 7 metrics' history + flagged-batch unit list).

**Per-unit predictor (demoted).** Collapsible "Predictor" section below the tabs, default collapsed. When expanded, shows the existing per-unit predictor output as a sortable table. Marked "diagnostic — not part of daily flow."

#### 4. Settings — admin / config

**Purpose:** Hold everything that isn't daily flow.

**Layout:** Single page with vertically stacked accordion sections. Each section collapsible; remembers expanded state per user.

Sections in order:
1. **Alert thresholds (NEW)** — per-tier, per-metric sliders. Live preview against the last 90 days of historical data: as you drag a threshold, the count of "would-have-flagged" models updates in real time. This is the primary tool for fixing "flags everything."
2. **Per-model specs** (folded in from V5 Specs page) — spec limit overrides per model, exclude_points management. Existing UI ported with restyled chrome.
3. **ML training** — per-model threshold optimizer, drift detector retrain, profiler refresh, predictor retrain. Existing Settings ML controls.
4. **Pricing / config** — existing.
5. **Database cleanup** — existing.

### Data layer additions (informational — owned by Specs 1 and 2)

Spec 3 (the UI shell + Triage + Model + restyle) depends on data the existing analyzer doesn't yet produce. Those are owned by:

- **Spec 1** (drafted on `main`, will arrive in V6 via merge-forward) — adds `TrackResult.untrimmed_sigma_gradient`. The Model page's glance pill for "untrim_sigma" reads this column.
- **Spec 2** (to be drafted on V6) — refactors `ml/drift_detector.py` from one-metric to N-metric. Adds the step-change-vs-slow-drift classifier. Tightens per-tier thresholds. Emits alerts for all 7 watched metrics plus smoothness. The Triage "Needs attention" zone reads these alerts.

Spec 3 must wait on Spec 2 — Triage's content is the detector's output. The new Settings "Alert thresholds" section is also Spec 2's UI surface (Spec 3 just renders sliders against Spec 2's per-tier-per-metric config).

## Cross-cutting decisions

### Navigation pattern
**Left sidebar** with icon + label per item. 4 items: Triage (top, default), Process, Model, Settings. Active item highlighted with an accent stripe. Sidebar is fixed-width (~140 px) and visible on every page. No top tabs, no breadcrumbs.

### URL / state model
Each page owns its own state. **Model page** specifically supports a model-id + focus-metric route (e.g. internal route `/model/8340-1?focus=untrimmed_resistance`) so Triage can deep-link with the triggering metric pre-selected.

### Alert tier visual identity
Existing tier taxonomy and color semantics (Stable / Warning / Drift / Out-of-Control) carry forward unchanged. The refresh is chrome (sidebar color, surface backgrounds) — the tier colors themselves stay so V5 history-readers don't get reorientation cost.

### Theme
Single light theme for v1. Dark theme deferred. Color palette and typography decisions for the refreshed chrome are left to Spec 3's implementation plan — not locked at design-spec level.

### Empty states
Every page has an explicit empty state.
- **Triage with no alerts:** affirmative summary, "All models within tolerance."
- **Triage with no models in DB:** "Process some files to get started" with a link to Process.
- **Model when no metric data exists:** "No measurements for this model yet."
- **Settings sections:** sensible defaults shown, no special empty state needed.

### Smoothness alert surfacing
Even though Smoothness is now a tab on Model, smoothness-derived drift alerts still appear in Triage's "Needs attention" zone with a `smoothness` alert-type label. Click-through opens the Model page with the Smoothness tab pre-selected.

## Implementation sequence

Three downstream specs follow this one, in order:

| Spec | What it builds | Branch | Status |
|---|---|---|---|
| **Spec 1** | `untrimmed_sigma_gradient` column + analyzer change + migration | `main` (V6 inherits) | Drafted 2026-05-30 (`2026-05-30-spec1-upstream-signals-capture.md`); ready for implementation plan |
| **Spec 2** | Multi-metric drift detector. Step-change vs slow-drift classifier. Per-tier per-metric threshold tightening with config persisted to disk. Smoothness signal feeding alerts. | `V6` | To draft after Spec 1 lands |
| **Spec 3** | New shell, sidebar nav, Triage page, Model page (incl. Drift Metrics / Smoothness / Units tabs and demoted Predictor panel), Settings restructure (incl. threshold-tuning UI, Specs section, ports of ML / pricing / cleanup), Process page port-with-restyle. Old Dashboard / Quality Health / Scorecard / Compare / Trends / Analyze / Specs / Smoothness pages retired in the same Spec. | `V6` | To draft after Spec 2 lands |

Each spec is drafted, reviewed, planned, and executed via the established skill chain (writing-plans → subagent-driven-development) before the next is started. Specs 2 and 3 are large enough to warrant their own brainstorming passes; their design specs will likely follow the same depth as this one.

## Open questions

None that block Spec 2 drafting. Spec 2 will surface its own decisions (algorithm details for step-change, baseline-window size, smoothness-signal definition, etc.). Spec 3 will surface its own (exact theme palette, icon library, layout densities for monitor sizes).

## Non-goals

- Real-time streaming. Batch flow is preserved.
- Cross-product-model views (e.g., "compare model X to model Y" as a primary view). Mission is per-model. Compare-style needs are absorbed into Model's Drift Metrics tab (which already shows baseline-vs-recent).
- Mobile / tablet layouts. Manufacturing PC monitors are the target.
- Auth, multi-user, role permissions. Single-operator workstation tool.
- Replacing the analyzer, parser, DB schema (beyond Spec 1's column add), or ML modules. Those are stable foundations.
- Western Electric rules 2–8 (anchor spec already deferred). Rule 1 only in Spec 2.
- Per-unit failure prediction as a headline view. Stays demoted per anchor.

## What this spec does NOT decide

- Spec 2's algorithm details (step-change window, EWMA decay, smoothness-signal calculation).
- Spec 3's exact CSS palette, icon set, font choices, default layout densities.
- Whether the existing Trends page's specific chart configurations carry into the Model page's Focus chart (Spec 3's implementer's call when porting).
- Migration UX for users on V5 when V6 ships (the "first run on V6" experience). Deferred to graduation time.
