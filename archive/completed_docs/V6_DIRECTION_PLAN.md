# Laser Trim Analyzer — Direction & Feature Plan

**Decision: continue V6, evolve it to fully serve the mission. Not V5, not a rebuild.**
The hard-won drift engine + the validated knowledge of the real signal live in `ml/` and
are portable/kept regardless. V6 already nails Job 1 (drift). The work is finishing Job 2
(investigation/history), which V6 under-built — plus the 3rd-laser support and data fixes.

---

## What the app is FOR (James' mission — the test every feature must pass)

**Job 1 — Proactive drift watch.** Tell me which potentiometer models' element production
is drifting (a sudden lot shift or a slow trend), how bad, and give me the evidence to hand
the engineers and MEs — so we catch the problem *upstream* instead of at final test, where
it's most expensive.

**Job 2 — Reactive investigation.** When there's an issue at laser trim or final test:
- **2a:** pull the linearity chart/data for the *specific problem units* in front of me now.
- **2b:** pull a model's *full history* — measured angle, resistance, linearity pass-rate,
  **test-pass vs trim-pass**, **trim-pass count** — over the whole record.

Audience: **me, engineers, and MEs.** *Not* leadership reporting.

---

## Feature set (derived from the mission)

### Keep / already done
- **Triage** — what's drifting, sorted by real baseline shift, scoped to active models. *(Job 1)*
- **Model drift evidence** — baseline/recent/shift table, copy-summary, evidence pack for
  engineers/MEs. *(Job 1)*
- **Per-unit linearity chart** — pre/post-trim, spec band, fail points, PNG/PDF export. *(Job 2a)*
- **Spec import** from the master reference sheet (done this session). *(infrastructure)*

### Build / finish — these serve the mission and are weak today
- **Trim↔FT overlay + Escape/Overkill (test-pass vs trim-pass)** — "are we catching it upstream
  or escaping to final test." Serves Job 1's whole point AND Job 2a investigation. **HIGH.**
- **Per-model history view** — angle, resistance, linearity pass-rate, trim-pass count trended
  over the full record, browsable. *(Job 2b)* **HIGH.**
- **Fast unit lookup** — pull up the specific units you're working on now, by serial / lot /
  recent. V6's Units tab is a thin recent-list. *(Job 2a)* **HIGH.**
- **Fix Missing Tracks** — re-parse so problem units actually have a chart (the ~7% of records
  with empty arrays / the "warning unit shows no data" issue). *(Job 2a)*
- **Smoothness view** — you investigate it, so keep a per-model smoothness view. *(Job 2b)*
- **Spec discrepancy check + fuller spec editor** — file-spec vs reference, AS9100 integrity.
- **Active-models (MPS) input** — let you pin the active model list explicitly.

### NEW requirement
- **3rd laser support** — a third trim system was just added. Add the enum value, teach the
  parser to detect its file format, and extend A-vs-B comparisons to A/B/C. ~116 code sites
  assume two systems. **Needs one sample file from the 3rd laser.** **HIGH / time-sensitive
  (new production is flowing now).**

### Cut — not in the mission
- Executive / leadership Excel report; cost-impact / scrap-$ view (engineer/ME-only app).
- Recent-alerts feed (Triage replaces it); category breakdown; trim-difficulty ranking;
  classic SPC histograms; Trends PDF; single/batch Excel export (evidence pack covers it).

---

## Shared-core data-integrity fixes (apply no matter what — affect both V5 and V6)
- **Warning-status logic** — some Warning units are linearity-FAIL (should they be Fail?),
  some are both-pass (should be Pass), some are un-evaluated. Trace `_determine_overall_status`
  and make the rule consistent and explainable.
- **Empty measurement arrays** — ~5,500 track rows (7%) have no sweep. Recover via Fix
  Missing Tracks; until then, the modal explains why (done).
- **Smoothness avg vs max** — the two columns are different quantities (`avg` ≈ 7× `max`);
  clarify or relabel.

---

## Bugs (fix first — done unless noted)
- App **freeze** — couldn't reproduce (app not visible when reported); WAL + 30s timeout rule
  out a permanent SQLite lock; data loads are already threaded. Hardening to confirm no
  main-thread heavy op. **Open — needs to be caught live.**
- Jagged trend charts → scatter. **Fixed.**
- Smoothness footer 18-decimal formatting → `.4g`. **Fixed.**
- Warning-unit modal blank message → explains the empty case. **Fixed.**
- Triage couldn't scroll → bounded scrollable card area. **Fixed.**

---

## Proposed build sequence
1. **3rd laser support** (unblocks current production) + confirm/fix the freeze.
2. **Investigation/history**: fast unit lookup + per-model history view (angle, resistance,
   pass-rate, trim-pass count).
3. **Trim↔FT overlay + Escape/Overkill** (test-pass vs trim-pass).
4. **Fix Missing Tracks**, spec discrepancy check, active-models input, smoothness view.
5. **Shared-core data-integrity fixes** (warning-status logic, smoothness columns).

## What I need from you
- **One sample data file from the 3rd laser** to build parser support.
