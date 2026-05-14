# App Mission Realignment — Working Spec

**Date:** 2026-05-14
**Status:** 🟡 Working draft — captures discussion through 2026-05-14. Several decisions still pending (see Open Questions). Do not start implementation from this doc; it needs another review pass once the deployment-strategy question is answered.
**Source:** Conversation with James on 2026-05-14. James reported the app "is a little all over the place and it kinda flags everything as a problem" and asked to rethink the app to better align with his actual workflow.

This is the **anchor spec** for the V5 → V6 (or V5 → new app — TBD) realignment. Once approved, three downstream feature specs derive from it.

---

## Mission

**When James opens this app, he is trying to answer:**

> *Which product models' element production is drifting (sudden lot shift or slow trend), how bad, and what evidence do I take to the engineers and MEs?*

Everything else is supporting infrastructure for that question. Anything that does not contribute to answering it is either a forensic drill-down (kept, possibly demoted) or dead weight (later candidate for removal).

## Why now

James' description of his actual workflow on 2026-05-14:

- **When drift is detected**, he needs three pieces of evidence to hand to engineers/MEs:
  1. How has untrimmed resistance changed?
  2. How bad is linearity now vs. before?
  3. Is electrical angle the same or different?
- **Daily use**: opens the app to process new files, then "analyzes models trying to see if any model is getting worse, or looks into specific models we think were having issues with." Complaint: "the app currently is a little all over the place and it kinda flags everything as a problem."
- **Per-unit interest**: only via charts (lin pre/post trim) and the Excel export. Forensic drill-down after a model is flagged — never the primary view.
- **Ah-ha moment**: would be either (a) detecting that a recent lot of elements is materially different from previous lots, or (b) detecting slow drift over time getting better or worse.

The two key implications are:

1. **The per-unit failure predictor does not serve this workflow.** Units that pass laser linearity always proceed to FT; units that fail are always reworked. There is no per-unit intervention point for the predictor to act on.
2. **The drift detector watches the wrong signal.** It currently monitors only `sigma_gradient`, which is calculated on **post-trim** data. By the time post-trim sigma drifts, the upstream element-production process has been off for a while. Untrimmed data — already stored in the DB — is the proper upstream signal.

## Success criteria

The app succeeds when, on a routine daily visit:

1. The landing view answers "anything to look at today?" in under 10 seconds with low false-positive rate. The current "flags everything" behavior is the chief complaint and the chief thing to fix — but via **threshold tuning**, not by removing alert tiers (see Important Constraints below).
2. When a model *is* flagged, the next click surfaces — without further digging — the baseline-vs-recent snapshot for: **untrimmed resistance**, **linearity error**, **electrical angle**, plus the metric that triggered the alert. James copies/exports that to hand to engineers.
3. James can navigate to *any* specific model (not just the flagged ones) and see its drift profile, because he sometimes investigates models on third-party tips.
4. Per-unit detail remains accessible as drill-down (charts pre/post trim, Excel export of all units for a model) for forensic deep-dive after triage.

## Important Constraints — Things NOT to change

These are explicit guardrails based on James' feedback. The realignment must preserve them.

- **Existing alert tier system stays.** James noted "im nervouse about losing the current teirs because it took a long time to get the app where it is today." The cure for "flags everything" is to **tighten the thresholds** that gate the existing tiers, not to delete a tier. The current Stable / Warning / Drift / Out-of-Control taxonomy (or whatever the in-code names are) is retained.
- **Smoothness page is critical** and stays. It is not a candidate for the page audit.
- **Excel export and per-unit chart drill-down** stay exactly as they are.
- **File ingest behavior** (Process page) is not in scope.
- **Per-unit failure predictor** is demoted to "available but not headline." Not deleted in this work. May be revisited later but James will not lose access to existing trained models.

## Open Questions (must resolve before implementation)

### Q1: Deployment strategy — A vs B vs C

James said: *"i dont know the best plan, maybe this should be a completely new build so i dont lose the current app."* The realignment can be done in one of three ways:

**A. New repo entirely.** Spin up a separate codebase that reads from the same SQLite DB. V5 (this repo) stays exactly as-is, deployed to work. The new app is a parallel codebase for the new mission views. Pros: highest isolation, zero risk to V5. Cons: two codebases to maintain, parser/analyzer/DB-models code duplicated or extracted to a shared library.

**B. Additive within current repo.** New mission views added as *additional* pages alongside everything that exists today. Nothing existing changes behavior. After the new views prove themselves, you can choose to consolidate (or not — they can coexist forever). Pros: one codebase, low risk because additive. Cons: page count grows, some sprawl.

**C. Long-lived V6 branch.** Fork into a development branch; V5 main stays stable for deployment. Pros: lets V6 diverge cleanly. Cons: long-lived branches drift hard, deploy/sync pain.

**Recommendation:** B. It preserves the working app (zero risk to V5 behavior) without duplicating the codebase. New views coexist with old. If the new views turn out to be the better lens, James naturally migrates to them; if not, no harm.

**Decision required before implementation specs are drafted.** Defer to James.

### Q2: Watched metrics — final list

Working list of metrics the multi-metric drift detector should watch per model:

1. `sigma_gradient` (current, kept — post-trim curve smoothness)
2. `untrimmed_sigma_gradient` (new — same calculation fed pre-trim arrays; pure upstream element-quality signal)
3. `untrimmed_resistance` (starting resistance — direct element-fab signal)
4. `linearity_error` (post-trim, what gets handed to engineers)
5. `measured_electrical_angle` (vs. spec — element angle drift)
6. `trim_pass_count` (number of laser passes needed — proxy for how marginal raw elements are)
7. `resistance_change_percent` (how much material the laser removed — proxy for how far raw elements are from target)

That's 7 metrics. James confirmed all are useful. Confirm this is the final list before drafting the drift detector spec.

### Q3: Alert taxonomy — keep current, just tighten

The existing tiers stay. We need to enumerate them precisely from the current code (`StatusType` enum and however alerts are categorized in `ml/drift_detector.py`) and document each tier's current trigger condition. Then propose per-tier **threshold tightening** to reduce false positives.

The two new alert *types* this realignment surfaces are orthogonal to the tiers — they are:

| Alert type | Trigger | What it tells James |
| ---------- | ------- | ------------------- |
| **Step change** | Recent lot mean differs from baseline lot mean by `> N` SE on at least one watched metric | "A batch of elements is materially different from what came before" |
| **Slow drift** | Linear regression slope over `M`-day window is statistically non-zero and worsening | "This model is gradually getting worse (or better) over time" |

Both can fire at any of the existing tier severities. Step change is usually higher-severity because it's actionable today; slow drift is medium because it's an emerging concern, not an emergency.

### Q4: Page audit — defer

Don't audit pages now. Defer until the new mission views are built and James can compare side-by-side. The current 10 pages (per CLAUDE.md): Dashboard, Process, Analyze, Compare, Trends, Quality Health, Scorecard, Smoothness, Specs, Settings.

**Confirmed kept**: Process, Analyze, Smoothness, Specs, Settings, Excel export from any page.
**Possibly redundant after new views exist**: Quality Health, Scorecard, Compare, Dashboard (may be reframed rather than removed).
**Trends**: stays — it's the per-model historical-lookback view, which the mission actively requires.

Page audit becomes its own (later) spec when James has the new views in hand and wants to consolidate.

---

## What needs to change architecturally

This spec does not enumerate every code change — that's what the downstream feature specs are for. Architecture-level:

### Drift detector — multi-metric

Currently in `src/laser_trim_analyzer/ml/drift_detector.py`. Watches one metric: `sigma_gradient`. Holds CUSUM + EWMA per model.

Needs to become a per-model, **per-metric** state — i.e., a `ModelDriftDetector` holds one CUSUM/EWMA pair per watched metric. The orchestration in `ml/manager.py` (around lines 246, 280, 288, 511, etc., where it currently passes `sigma_gradient` arrays) loops over the metric list instead.

Alert output includes which metric drifted, so James knows where to look upstream.

Step-change detection is a new check: compare the most-recent N samples' mean against the baseline distribution, flag if delta exceeds a configurable threshold.

### Upstream signals capture

Add `untrimmed_sigma_gradient` column to `TrackResult`. Calculate by calling the existing `_calculate_sigma` (in `core/analyzer.py:338`) with the untrimmed arrays (`untrimmed_positions`, `untrimmed_errors`) — already loaded into local vars in `analyze_track` at line 193 but currently unused for sigma.

Migration: alembic-style add-column. Old rows backfilled by the natural file-reprocessing flow (James already routinely reprocesses; CLAUDE.md treats reprocess as the normal mechanism). Until backfilled, drift detector treats `NULL` as missing data for that metric.

### Landing + investigation views

Per the deployment strategy decision (Q1), this is either new pages in the existing app (Option B) or new app entirely (Option A). Either way:

**Landing view** — "Models needing attention today." Sorted list. Each row shows: model, alert type (step change / slow drift), worst-metric, magnitude. Default to under-flagging.

**Per-model investigation view** — Drilled-down view for one model. Side-by-side baseline-vs-recent for the seven watched metrics. Trend chart over time. Quick access to the unit-level chart drill-downs and Excel export that already exist.

---

## Downstream feature specs (sequence)

Once this anchor spec is approved (after Q1–Q3 resolved), three feature specs follow in order:

1. **Upstream signals capture** — `untrimmed_sigma_gradient` column + calculation + reprocess-backfill behavior. Smallest, cleanest, foundation for everything else. Independent — can land first regardless of Q1 deployment-strategy answer (the column is useful to both Option A and Option B).
2. **Multi-metric drift detector** — refactor `ModelDriftDetector` to N-metric. Add step-change vs. slow-drift classification. Tighten existing tier thresholds.
3. **Landing + investigation view** — new mission-aligned UI. Form depends on Q1 (new pages in V5 vs. new app).

A fourth spec — **page audit and consolidation** — comes *only if* James decides he wants it, after Spec 3 lands.

---

## Conversation notes (context for picking this back up)

Captured here so a future session has the same context this one does without re-reading chat.

### Earlier in the same session: exclude_points ML semantics

James asked how excluding points via the Specs page affects ML training. Confirmed (and saved to user-level memory) that:

- Excluded points reduce `fail_points`, change `optimal_offset`, and consequently flip the `linearity_pass` label from FAIL → PASS for any unit whose only failing positions were excluded.
- This is desired behavior: excluded positions represent fixture artifacts, not hidden defects.
- Workflow: reprocess all files (rewrites `fail_points` and labels) → retrain per-model ML in Settings.
- Intentional minor inconsistency: `linearity_error` (analyzer.py:508) does NOT filter excluded indices (uses full max abs error), while `fail_points` does. Leave alone.

This is orthogonal to the realignment but should not be regressed.

### James' clarification of sigma_gradient calculation

James thought sigma_gradient was calculated on pre-trim data. **It is not.** It's calculated on `errors` (post-trim) — see `core/analyzer.py:213-216, 338-394`. The untrimmed arrays are loaded into local vars (line 193) but not passed to `_calculate_sigma`. This is the gap Spec 1 fills.

### James' core complaint

"The app currently is a little all over the place and it kinda flags everything as a problem." This is the lived-experience signal that drives the realignment. The fix surface is two-pronged:

1. **Selectivity in alerting** — tighten thresholds on existing tiers (Q3).
2. **Focus in UI** — landing page shows only models that need attention, with everything else accessible but not in front of you.

### Why per-unit predictor is being demoted

The predictor was originally designed to predict which units would fail FT so they could be diverted. But in the actual manufacturing flow:

- Units that pass laser linearity always go to FT.
- Units that fail laser linearity always get reworked.
- There is no decision point between laser and FT where a per-unit failure prediction could act.

So the predictor has been answering a question the workflow never asks. Demoted, not deleted (James may use it as background context). Reconsider deletion in a later cycle.

### Lookback need (separate from drift)

James also said: *"its also for data lookback so i can see historically what resistances worked vs what didnt and if linearity error is getting better or worse compared to history."*

This is a **reporting/UI need**, not a new ML feature. The data is already in the DB (`untrimmed_resistance`, `trimmed_resistance`, `linearity_error`, `file_date`, per model). The Trends page partly does this; the per-model investigation view in Spec 3 should expand on it.

---

## Next steps (when picking this up)

1. **Decide Q1** (deployment strategy: new repo vs. additive vs. branch). Default recommendation is **B (additive)**, but James asked to defer.
2. **Confirm Q2** (the seven watched metrics are final, or some get dropped/added).
3. **Confirm Q3** (existing tiers stay; we'll enumerate and propose tightened thresholds in the multi-metric-detector spec).
4. **Start Spec 1** (untrimmed_sigma_gradient capture) — safe to start *independent of Q1* because it's a pure analyzer/DB addition.
5. Iterate Spec 2 and Spec 3 after Spec 1 lands.

---

## Non-goals

- Real-time intervention between laser and FT (no such process exists).
- Per-unit failure prediction as the headline.
- Cross-product-model analysis (e.g., "compare model X to model Y"). Mission is per-model.
- Real-time streaming. The app remains batch.
- Adding more measurement metrics that aren't already captured by the existing parser. Use what's in the DB.
- Western Electric rules 2–8. Rule 1 only for v1.
- Deleting working features. Demotion before deletion. James worked hard to get the app where it is.
