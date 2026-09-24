# Findings Catalogue Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the process-findings catalogue — station setup mismatch, where the loss is made, rework load, configured-vs-optimal target, machine comparison and the other laser settings — each honest about what it cannot say.

**Architecture:** Each finding is one analyzer module in `findings/analyzers/` (pure over `TrackView`s, or given a read-only DB handle when it needs final test), run and independently guarded by `findings/engine.compute_for_model`, mapped to a Findings-page group in `findings/presentation.py`. Two catalogue items extend existing code instead (#8 inside `ink_target`; #2 reuses `core/spec_alignment`'s comparison).

**Tech Stack:** Python 3, SQLAlchemy 2.0 on SQLite; no GUI work beyond presentation mapping.

**Spec:** `docs/superpowers/specs/2026-09-24-findings-catalogue-completion-design.md` (rulings 1–6).

## Global Constraints

- **Never open `data/analysis.db` read-write.** Real-data checks use a COPY (`cp data/analysis.db /tmp/f.db`, deleted afterwards) through `scripts/refresh_findings.py`.
- Any test that builds a `Processor` or `DatabaseManager` injects BOTH `laser_trim_analyzer.database.manager._db_manager` AND `laser_trim_analyzer.database._db_manager`.
- The only levers are `findings/model.py`'s `LEVERS` (`laser_settings`, `laser_limit_table`, `ink`, `deposition`); `Finding()` raises on any other.
- A finding that claims a gain carries `gain_definition` and `scope_annual_tracks` for ITS population; every analyzer in this plan claims NO gain (`expected_gain_points=None`).
- An analyzer can say nothing, and a test proves it on data with nothing to say (made to fail first). A crash is named in `facts["errors"]` by the engine's guard — never swallowed inside the analyzer.
- A record that failed processing is not a measurement (`core/model_stats._FAILED_PROCESSING`); a pass rate is a verdict against a test — never compare across limit tables.
- Lasers are named with `core.models.laser_label()` — "Laser 1 (LTS)", never the A/B/C letters. Laser 1 = LTS = "B"; Laser 2 = DLTS = "A"; Laser 3 = LTS3 = "C".
- Every new analyzer module is mapped in `presentation.ANALYZER_GROUP` (a test enumerates `findings/analyzers/` and fails on an unmapped module).
- `strftime("%-d")` is forbidden. Bind datetimes into `text()` queries as `f"{dt:%Y-%m-%d %H:%M:%S.%f}"` (the stored format), never a raw `datetime`.
- The gate is the whole suite: `.venv/bin/python scripts/run_test_gate.py` (read its summary line). Every threshold and guard is mutation-checked.
- Commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Stage by explicit path; never `git add -A`; never stage `.claude/settings.local.json`. Example data is INVENTED (no backlog names, POs or prices).

## File structure

| file | change | responsibility |
|---|---|---|
| `findings/analyzers/station_setup.py` | create | #2: laser vs final-test limits |
| `findings/analyzers/loss_origin.py` | create | #3: does incoming measurement predict the laser verdict |
| `findings/analyzers/rework_load.py` | create | #5: laser FAIL → final-test PASS unit-days, confirmed |
| `findings/analyzers/ink_target.py` | modify | #8: say when the configured window disagrees |
| `findings/analyzers/machine_compare.py` | create | #10: same model, same table, same months, two lasers |
| `findings/analyzers/setup_change.py` | create | #9/B1c: other laser settings changed; did the pass rate follow |
| `findings/engine.py` | modify | run and guard the new analyzers; facts keys |
| `findings/presentation.py` | modify | groups, readouts, statements for the new analyzers |
| tests | create | one test file per analyzer + presentation additions |

---

### Task 1: `machine_compare` (#10) — the template the others follow

**Files:** Create `src/laser_trim_analyzer/findings/analyzers/machine_compare.py`, `tests/test_findings_machine_compare.py`; modify `findings/engine.py`, `findings/presentation.py`.

**Interfaces:** `analyze(model, tracks, laser_label) -> Tuple[Dict[str, Any], List[Finding]]`; constants `MIN_TRACKS_PER_LASER = 100`, `MIN_GAP_POINTS = 10.0`. Facts: `{f"{table_key}": {"months": [...], "by_laser": {"<laser label>": {"n": int, "pass_pct": float}}}}`. Evidence: `{"table": key, "months": [first, last], "by_laser": {...}, "best_laser": code, "worst_laser": code}`.

Measured on the rebuild (read-only, 2026-09-24): 6126 — same limit table, same months — laser 2 (DLTS) 99% of 488, laser 1 (LTS) 76% of 610; 6952 laser 2 96% of 156, laser 1 84% of 160, laser 3 85% of 62. So on the real data this analyzer produces exactly two findings.

- [ ] **Step 1: Failing tests** — build `TrackView`s the way `tests/test_findings_cut_setting.py` does (read its `track()` helper and reuse the pattern) and assert:
  1. two lasers, same table, same months, 200 tracks each, 95% vs 75% → ONE finding, lever `laser_settings`, `expected_gain_points is None`, title names both lasers via `laser_label` ("Laser 2 (DLTS) passes 95% … Laser 1 (LTS) 75%"), `systems` holds both codes.
  2. the same two lasers on DIFFERENT limit tables → nothing (a pass rate is a verdict against a test).
  3. the same table but DIFFERENT months (laser 2 in 2024, laser 1 in 2025) → nothing.
  4. a 9-point gap → nothing; 99 tracks on one side → nothing (both thresholds).
  5. failed-processing / ungraded tracks (`linearity_pass is None`) are not counted.
  6. one laser only → `({}, [])`.
- [ ] **Step 2:** run → FAIL (module missing).
- [ ] **Step 3: Implement.** Group graded tracks (`linearity_pass is not None`, `limit_table` not None) by `(limit_table.key, "YYYY-MM")`; keep only the cells where ≥ 2 lasers ran; pool per `(table key, laser)` over those shared months; for each table with ≥ 2 lasers of ≥ `MIN_TRACKS_PER_LASER`, compare the best and worst laser; emit when the gap ≥ `MIN_GAP_POINTS`. Summary says the months, the table's graded point count, and the caveat: "Settings that work on one laser may not transfer; this compares what each laser achieved on the same test in the same months, not why." Category `"Machine comparison"`.
- [ ] **Step 4: Engine + presentation.** In `compute_for_model`, add `"machine_compare": None` to the facts skeleton and a guarded call (`facts["machine_compare"], found = machine_compare.analyze(model, tracks, _laser_label)`). In `presentation.py`: `ANALYZER_GROUP["machine_compare"] = "yield"`; `statement()` returns the title (no special case needed).
- [ ] **Step 5:** tests PASS; mutation-check `MIN_GAP_POINTS` (→ 0: test 4 red) and the table rule (group by month only: test 2 red); restore.
- [ ] **Step 6: Real data.** `cp data/analysis.db /tmp/f.db && .venv/bin/python scripts/refresh_findings.py /tmp/f.db 6126 6952 8821` (check the script's argument syntax) — record in the commit what 6126 and 6952 now say; 8821 must say nothing. Delete the copy.
- [ ] **Step 7:** gate → GREEN; commit `feat(findings): machine comparison -- same model, same test, same months, two lasers`.

---

### Task 2: `loss_origin` (#3) — a fact always, a finding only when strong

**Files:** Create `findings/analyzers/loss_origin.py`, `tests/test_findings_loss_origin.py`; modify `engine.py`, `presentation.py`.

**Interfaces:** `analyze(model, tracks, laser_label)`; `MIN_TRACKS = 300`, `MIN_PER_OUTCOME = 50`, `STRONG_AUC = 0.70`, `LOOKBACK_DAYS = 365`. `auc(fails: List[float], passes: List[float]) -> Optional[float]` (Mann–Whitney with half-credit ties; the probability a FAIL scores above a PASS). Facts per laser label: `{"n", "fails", "auc_error", "auc_resistance"}` (either AUC may be None).

Measured (latest year, 2026-09-24): 6607 on laser 1 (LTS) AUC 0.74 over 1,290 tracks, 465 failing — the only one of eight models checked that qualifies; 8232-1 0.57; 8340-1 0.57; 8202-1 0.71 with only 27 failures (excluded by `MIN_PER_OUTCOME`).

- [ ] **Step 1: Failing tests:** `auc` on hand-made lists (perfect separation 1.0, identical 0.5, reversed 0.0, ties half-credit); a strong synthetic laser (fails' incoming max|error| drawn from a higher range) → one finding, lever `deposition`, title "Laser 1 (LTS): incoming linearity predicts the laser verdict (AUC 0.9x)"; a weak one (same distribution) → facts only, no finding; 40 failures → no finding even with AUC 0.9 (`MIN_PER_OUTCOME`); tracks without an untrimmed sweep are skipped, never scored 0.
- [ ] **Step 2:** FAIL. **Step 3: Implement** — per laser, latest year, score = `max(abs(e) for e in t.untrimmed_errors if e is not None)`; `auc_resistance` from `t.untrimmed_resistance` (skip None and the `1e9+` junk readings — `0 < r < 1e9`). Finding only on `auc_error`. Summary: "Tracks that fail at the laser already arrived with worse linearity: the loss starts before the laser, so deposition is the lever to look at first. This routes the question; it never grades a part." Category `"Where the loss is made"`.
- [ ] **Step 4:** engine facts key `"loss_origin"`; `ANALYZER_GROUP["loss_origin"] = "yield"`.
- [ ] **Step 5:** PASS; mutation-check `STRONG_AUC` and `MIN_PER_OUTCOME`; **Step 6:** real data on a copy for 6607, 8232-1, 8202-1 — record them; **Step 7:** gate, commit `feat(findings): where the loss is made -- incoming linearity against the laser verdict`.

---

### Task 3: `station_setup` (#2) — the laser and final test grading to different limits

**Files:** Create `findings/analyzers/station_setup.py`, `tests/test_findings_station_setup.py`; modify `engine.py`, `presentation.py`.

**Interfaces:** `analyze(model, db, tracks, laser_label)` — takes the db because final-test limits live in `final_test_tracks`. It uses `core/spec_alignment`'s `_linked_pairs`, `_trim_arrays`, `_ft_arrays`, `compare_arrays`, `MIN_MATCHED`, `DIFFER_SHARE` and `_band_text` DIRECTLY — not `compare_station_specs`, which turns a read failure into "insufficient" for the banner's sake; here an exception must propagate to the engine's guard and be named in `facts["errors"]`. (If importing the underscore names is awkward, add a small public `sample_and_compare(db, model, sample_per_side) -> SpecComparison` in `spec_alignment.py` that raises, and make `compare_station_specs` call it inside its existing try — one comparison, two callers.)

- [ ] **Step 1: Failing tests** with an in-memory/tmp DB (both globals injected), seeding one linked trim/FT pair set per case (read `tests/` for an existing spec-alignment test and reuse its seeding helper): stations whose limits differ over 40% of matched positions → one finding in the check group, lever `laser_limit_table`, no gain, summary carrying the `note`-style sentence (share + both bands); aligned → nothing; fewer than `MIN_MATCHED` positions → nothing; a query that raises (monkeypatch the sampler to raise) → the ENGINE records `facts["errors"]["station_setup"]` (test through `compute_for_model`).
- [ ] **Step 2:** FAIL. **Step 3: Implement.** Readout = graded tracks in the model's latest year (`n_units`). Title: "The laser and final test grade to different limits over N% of the travel". Category `"Station setup"`. `systems` = the lasers among the model's latest-year tracks.
- [ ] **Step 4:** engine: `station_setup.analyze(model, db, tracks, _laser_label)` guarded; `ANALYZER_GROUP["station_setup"] = "check"`; the check group's `meaning` becomes "Graded against more than one limit table, or to different limits than final test, so pass rates across the change don't compare" and its `empty` text mentions both.
- [ ] **Step 5:** PASS; mutation-check `DIFFER_SHARE` use; **Step 6:** real data on a copy for 8232-1 (TRACKER D1's case) and 8340-1 — record; **Step 7:** gate, commit `feat(findings): station setup -- when the laser and final test grade to different limits`.

---

### Task 4: `rework_load` (#5) — laser FAIL, final-test PASS, confirmed as hand trim

**Files:** Create `findings/analyzers/rework_load.py`, `tests/test_findings_rework_load.py`; modify `engine.py`, `presentation.py`.

**Interfaces:** `analyze(model, db, tracks, laser_label)`. Counts unit-days through `DatabaseManager.get_model_trim_ft_agreement(model, cutoff_date=latest - 365 days)` — its `overkills` are exactly "failed trim but passed final test" by THE unit-day rule (per track the day's last attempt; every track must pass). Confirmation: for those units vs pass/pass units, the ratio final-test linearity error ÷ laser final linearity error (`final_test_results.linearity_error` vs the linked analysis' tracks' `final_linearity_error_shifted`, max over tracks); `CONFIRM_RATIO = 0.6` — the rework median must be ≤ 0.6 × the pass/pass median (their error fell between the stations; a unit that merely passed a laxer final test would not show it). `MIN_UNIT_DAYS = 30`, `MIN_CONTROL = 30`.

- [ ] **Step 1: Failing tests** (tmp DB, both globals): 40 rework unit-days whose FT error is a third of their laser error + 40 pass/pass whose ratio ≈ 1 → one finding in laser_time, readout 40; the same 40 but FT error ≈ laser error (no improvement) → nothing, because the signature is not confirmed; 20 unit-days → nothing; no linked final tests → `({...}, [])` with facts saying so; failed-processing tracks excluded.
- [ ] **Step 2:** FAIL. **Step 3: Implement.** Title "{laser}: N units a year fail here and pass final test after rework". Summary names the hand-trim reading and the ratio evidence; lever `laser_settings`; category `"Rework load"`; `evidence["facts"] = {"rework_unit_days": N, "control_n": ..., "median_ratio_rework": ..., "median_ratio_control": ...}`.
- [ ] **Step 4:** engine facts key `"rework_load"`; `ANALYZER_GROUP["rework_load"] = "laser_time"`; `_LASER_TIME_FIELD["Rework load"] = "rework_unit_days"`; the laser_time group's meaning becomes "Units cut that didn't need it, given more cuts than planned, or hand-trimmed after failing at the laser".
- [ ] **Step 5:** PASS; mutation-check `CONFIRM_RATIO` (→ 10: the no-improvement test red); **Step 6:** real data for 8340-1, 6607, 8232-1, 8397-2, 8275 (≈0 expected) — record; **Step 7:** gate, commit `feat(findings): rework load -- laser fails that pass final test after hand trim`.

---

### Task 5: #8 inside `ink_target` — the configured window disagrees

**Files:** Modify `findings/analyzers/ink_target.py`, `tests/test_findings_ink_target.py`, `findings/presentation.py`.

- [ ] **Step 1: Failing tests:** a finding whose recommended window lies wholly outside the configured incoming window gets `evidence["configured_disagrees"] is True`, one sentence in the summary ("The station is set to accept A to B Ω, and the window that did best lies outside it."), and presentation tag "outside the configured window"; an overlapping window → False, no tag; no configured window → key absent/None, no tag.
- [ ] **Step 2:** FAIL. **Step 3:** implement (overlap test on the two closed intervals); in `presentation._tags`, add the tag when any member's evidence has `configured_disagrees`.
- [ ] **Step 4:** PASS; mutation-check the overlap test; gate; commit `feat(findings): say when the station's configured incoming window disagrees with the data`.

---

### Task 6: `setup_change` (#9 remainder, B1c) — the other laser settings

**Files:** Create `findings/analyzers/setup_change.py`, `tests/test_findings_setup_change.py`; modify `findings/data.py` (load `trim_setup.parameters` onto each `TrackView` as `setup: Optional[Dict[str, Any]]`), `engine.py`, `presentation.py`.

**Interfaces:** `ALIASES: Dict[str, str]` (`{"response_linear_or_function": "response"}` — laser 1's `Response` and laser 2's `Response (Linear or Function)`, ruled one setting on 2026-09-23; check the exact normalised keys in `core/trim_setup.normalise_key` output on the fixtures); `EXCLUDED: FrozenSet[str]` (identity-like keys: serial, dates, file names, operator, comments — build it from the keys actually present in the fixtures' `parameters`, and list them in the module docstring); `MIN_RUN_DAYS = 60`; `MIN_TRACKS_SIDE = 100`.

- [ ] **Step 1: Failing tests:** a numeric parameter that changes once for a (model, laser) with ≥ 60 days and ≥ 100 tracks on each side, SAME limit table → one history finding with `evidence["before"]["trim_pass_pct"]`/`["after"]["trim_pass_pct"]` (the same shape `recipe_change` uses, so `presentation.readout` reads it unchanged) and `evidence["after"]["first"]` (ISO date); a change that coincides with a limit-table change → nothing (the test moved too); a 30-day run → nothing; an excluded key changing → nothing; `Response` on laser 1 and `Response (Linear or Function)` on laser 2 are one key after aliasing; non-numeric values are compared as strings.
- [ ] **Step 2:** FAIL. **Step 3: Implement** — per (model, laser, track name), order tracks by date, find runs of constant value per key, and report each boundary that passes the rules; title "{laser}: {setting} changed from A to B"; lever `laser_settings`; category `"Setting change"`; no gain.
- [ ] **Step 4:** `data.py` loads `parameters` (one extra LEFT JOIN column); engine facts key `"setup_change"`; `ANALYZER_GROUP["setup_change"] = "history"`; the history group's meaning becomes "Recipe and setting changes, newest first".
- [ ] **Step 5:** PASS; mutation-check `MIN_RUN_DAYS` and the table rule; **Step 6:** real data on a copy for three models with known setting changes (TRACKER memory: 10 of 135 local models showed a changed Laser Power/Duration/Indexing; 6828's incoming target 1700 → 1800) — record what fires; **Step 7:** gate, commit `feat(findings): setting changes beyond the cut recipe -- did the pass rate follow`.

---

### Task 7: Close-out

- [ ] `scripts/refresh_findings.py` on a COPY for ALL models; record the count per group before/after and the run time; delete the copy.
- [ ] `scripts/app_qa_sweep.py` on a copy: its "every cached finding's analyzer maps to a group" entry covers the new analyzers.
- [ ] `TRACKER.md`: B6 ticked with what each analyzer found on the real database; B1c ticked (setup_change); the Response alias recorded.
- [ ] `python scripts/check_no_customer_values.py` → 0 problems.
