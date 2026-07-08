# V6 Finish-Line Plan — 2026-07-06

Full-code review of V6 branch (all of `gui/v6`, `core`, `database`, `ml`) + docs/git history + test run + live DB inspection. Supersedes nothing — this sequences the *remaining* work after V6_DIRECTION_PLAN.md (2026-06-24).

## Verdict

V6 is feature-complete against the direction plan except 3rd-laser support and Fix Missing Tracks. All specs (2, 3a–3e, dashboard) are implemented and the non-GUI test suite passes (180+ tests). What stands between "works on my machine" and "trustworthy production tool" is a small set of wiring/correctness bugs — several of which directly explain "keeps running into issues":

1. **Triage is frozen in time** — new data never advances drift state (P0-1)
2. **A likely mechanism for the intermittent freeze** — main-thread DB calls + a global DB lock + cross-thread Tk calls (P1-1/P1-2)
3. **A crash that silently blanks Model-page tabs** (P1-3)
4. **Settings sensitivity preset writes wrong thresholds** (P0-2)

Everything below has file:line references. Items marked ~ are one-sitting fixes.

## Current state (verified)

- Branch V6, 88 commits past main, last commit 2026-06-25. Working tree clean vs HEAD except a git index quirk (see Hygiene) and 3 untracked items.
- Tests: all non-GUI tests pass (spec1/2/3b data-layer, composite, audit fixes, drift live-advance, yield, predictor). GUI tests need tkinter (not runnable in this sandbox — run locally).
- DB (data/analysis.db, 3.2 GB): 78,435 analyses (2013 → 2026-05-28), 80,292 tracks. model_metric_state trained 2026-06-24 for 279 models (2,511 rows). Composite deployed: 38,215 scored tracks. trim_pass_count filled 97.5%. Status mix: PASS 36,824 / **WARNING 32,912 (42%)** / FAIL 8,590 / UNTRIMMED 109.

## P0 — Drift-engine correctness (the app's core promise)

**P0-1. `advance_drift_state` has zero callers.** ~
The live-advance half of the drift lifecycle was built and tested (`tests/test_drift_live_advance.py` passes) but never wired. Nothing calls it — not the processor, not the GUI. Triage state is frozen at training time (2026-06-24); new batches move nothing until a full manual retrain.
- Fix: call it per affected model after batch save (Process page worker, `gui/v6/pages/process_page.py` after save loop) and/or on app start.
- Also fix its boundary bug: `file_date > after` filter (`ml/drift_training.py:266,280`) permanently skips samples sharing the last processed day (file_date is day-granularity). Needs `>=` + dedup, or track processed IDs.

**P0-2. Bonferroni mismatch between training and Settings preset path.** ~
Training divides tier FP targets by 9 (`ml/drift_training.py:152-156`); `preview_alert_count` (`ml/manager.py:~1735`) and `apply_sensitivity_preset` (`ml/manager.py:~1825`) don't. "Save preset" writes thresholds ~9× looser than a retrain at the same preset; preview counts don't match trained reality.
- Fix: share one threshold-computation helper that applies the correction.

**P0-3. `recent_window` not persisted → step-change detection dead at read time.** 
Hydrated detectors (`ml/manager.py:1655-1679`) start with empty windows, so `AlertType.STEP_CHANGE` is unreachable in Triage and the `_sigma_shift` worst-metric tiebreaker (`multi_metric_drift_detector.py:341-344`) is always 0 → arbitrary same-tier metric pick. Fix alongside P0-1 (persist window in model_metric_state, or rebuild from last N rows at hydrate).

**P0-4. Composite demotion keyed on `is_trained`, not deployment** (`multi_metric_drift_detector.py:350-356`). Once ≥30 composite scores exist, the trim-effort family (incl. `untrimmed_error_max`) stays demoted even if a retrain un-deploys the composite or its baseline goes degenerate — the family can be silently muted. ~

**P0-5. Degenerate-baseline guard hole for zero-mean constants.** Training floors std to 1e-9 (`drift_training.py:147-148`); the guard cutoff (`multi_metric_drift_detector.py:58-63`) then never fires for all-zero baselines (e.g. resistance_change_percent) → collapsing limits, garbage flags. ~

## P1 — Stability: the freeze and silent blanking

**P1-1. Freeze mechanism (most probable): global DB RLock + main-thread DB calls.**
`DatabaseManager.session()` holds a process-wide RLock for the whole session (`database/manager.py:216,1130`; StaticPool = single SQLite connection, so WAL concurrency is nullified). Main-thread DB call sites that block behind any worker holding the lock:
- `UnitChartModal.__init__` → `load_unit_track` (`gui/v6/widgets/unit_chart_modal.py:61`)
- Per-model specs section load/save/delete (`gui/v6/sections/per_model_specs.py:75,144,166` — :75 runs at startup)
- Copy-summary (`gui/v6/pages/model_page.py:320-322` — two sessions, 9 agg queries)
- Startup auto-train gate (`gui/v6/app.py:71`)
- PredictorPanel "Show" → `load_all()` unpickles every model on the UI thread (`widgets/predictor_panel.py:61-69`)
Long lock holders: yield computations fetch full tables (`core/yield_stats.py:33-36`), drift training, batch save loop.
- Fix: move these 5 call sites to workers (pattern already exists everywhere else).

**P1-2. `safe_after` makes Tk calls from worker threads** (`gui/v6/page_base.py:44-49`; cloned in training_modal + all six settings sections). `winfo_exists()`/`after()` from a non-main thread is not thread-safe Tkinter; combined with P1-1 it's a deadlock recipe (worker waits on main loop, main thread waits on DB lock).
- Fix: single `queue.Queue` drained by a main-thread `after(50, ...)` poller in PageBase; sections use it too.

**P1-3. Smoothness tab crash silently blanks later tabs.** ~
`sorted(records, key=lambda r: r.get("file_date") or 0)` (`gui/v6/widgets/smoothness_tab.py:37`) raises TypeError (datetime vs int) when any file_date is None — None is expected (display code handles it at :40). Model page applies smoothness before trim-FT/history (`model_page.py:182-184`) inside one broad `except` (:167-170), so one bad record = stale/blank tabs with no error. Fix: `key=... or datetime.min`, and split the broad except per-tab + surface errors.

**P1-4. Triage refresh N+1.** `get_triage_alerts` hydrates per model, + `compute_recent_means` (9 agg joins) per flagged model; `list_known_models` re-runs hydration (`ml/manager.py:1495-1551,1795`). Seconds per refresh at 279 models, all under the global lock — feels like a hang. Batch the queries.

## P1.5 — Data trust (AS9100-relevant)

**P2-1. WARNING status recompute.** 42% of analyses are WARNING with three mixed meanings (old rule labeled linearity-FAIL-but-sigma-pass as Warning; untrimmed-sigma gate change 5da21e8; frozen historical thresholds). For a zero-tolerance linearity requirement, linearity-FAIL rows presenting as "Warning" is a misclassification problem, not cosmetic. Rows were never re-graded (no migration exists).
- Fix: status-recompute backfill from stored `linearity_pass`/`sigma_pass` (rule at `core/analyzer.py:279-286` is now correct; `processor.py:1077-1103` rollup fine). Runbook it like trim_pass_count. Also update stale docstring `core/models.py:363`.

**P2-2. Fix Missing Tracks can't see the actual problem rows.** Finders only detect analyses with *zero* track rows (`database/manager.py:5657,5772`); the ~7% with NULL arrays (SafeJSON maps []→NULL, `database/models.py:60-61`) are invisible to it and to `scan_database_health`. Also `update_trim_tracks_from_final_test` fabricates `sigma_gradient=0.0, sigma_pass=True` (`manager.py:5922-5924`) — contaminates sigma stats and ML training data.
- Fix: extend finders to NULL-array tracks; write NULLs (not fake values) for FT-derived rows and exclude them from training queries.

**P2-3. Parser data-loss edge:** `_get_column_data` stops at ≥10 consecutive NaN (`core/parser.py:1125-1128`) — an interior dropout discards all subsequent real data. Worth a bounded look-ahead.

## P2 — Remaining features

**P3-1. 3rd laser support.** Still blocked on a sample file. Detection is sheet-name based (`core/parser.py:106-124`) and **unknown formats silently default to System B** (:123) — so 3rd-laser files are likely being mis-parsed under B right now, not rejected. Interim guard (~): make unknown-format files ERROR loudly instead of defaulting. Full support touches: `core/models.py:21` + `database/models.py:131` enums, `utils/constants.py` column maps, ~10 A/B branches in parser.py (431,605,752,764,836,419,563,714), manager comparison queries, dashboard A/B charts (~150 grep hits total).

**P3-2. Ingest-time spec discrepancy check.** `get_spec_discrepancies` exists (retrospective, `manager.py:8332`, N+1 loop at :8362); processor never warns at ingest when file spec ≠ reference. Add the check in processor + fix N+1.

**P3-3. Consistency polish:** evidence pack labels CUSUM magnitude "Δσ" while UI shows honest (recent−baseline)/std (`export/evidence.py:71,92`) — exported numbers contradict the screen. History tab pass-rate is per-track while Trim-vs-FT is per-unit — tabs can disagree; pick per-unit or label them.

## P3 — Hygiene

- Git index quirk: `active_models.py` + `history_tab.py` show staged-deleted + untracked but content == HEAD. `git restore --staged` both (or `git add`), then commit `launch_v6.command` + `scripts/resummarize_smoothness.py`; add `work db 6-23-2026/` to .gitignore.
- Old single-metric drift detector is write-only in v6 (trained, never read — readers are v5 pages v6 never imports). Skip its training in v6 or retire with v5 pages at graduation.
- Full main→V6 merge audit (Batches A–H landed on main post-split; D-SIGMA was cherry-picked; verify nothing else is missing).
- Stale: repo CLAUDE.md still says "Current focus: BUGFIX_PLAN_V5"; "8 metrics" comments vs 9 WATCHED_METRICS (`drift_types.py:146`).

## Suggested sequence

1. **Stability first** (P1-1/2/3): they're what you *feel* daily, and the P1-3 fix is minutes.
2. **Drift-engine wiring** (P0-1..5): Triage can't be trusted until state advances and presets are consistent. One focused session.
3. **3rd-laser interim guard** now; full support the day a sample file exists.
4. **WARNING recompute + Fix Missing Tracks** (runbook-style, backfill on work DB).
5. Spec-check at ingest, consistency polish, hygiene, then v5-page retirement + main merge-back.

Steps 1+2 are the "finish line" for daily use; 3–5 are the finish line for trust and AS9100 posture.
