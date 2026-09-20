# Limit Tables: One Test Per Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **How this plan carries its code.** Every task's code already exists as a pair of patches — tests first, then source — written and validated by the plan's author in an isolated worktree against real data before this document was written. They live in the plan's SDD workspace, `.superpowers/sdd/2026-09-20-limit-tables/task-N-{tests,src}.patch`, and apply cleanly to the branch. A task is: apply the tests patch, SEE it fail for the stated reason, apply the source patch, see it pass, prove the tests have teeth with the named breaks, commit. The patches are the code; this document is the reasoning, the expectations, and the order.

**Goal:** Make the limit table — the per-point (upper, lower) pairs a laser grades a track against — a first-class fact: report when a model is graded against more than one, and stop any other analyzer comparing pass rates across two.

**Architecture:** `findings/data.py` gives every `TrackView` its track name, its positions and a `LimitTable` (a content fingerprint plus the band by position). A new analyzer, `findings/analyzers/limit_tables.py`, reports tables in service at once or a table that changed, comparing bands by interpolation over shared travel. `ink_target` adds the table to the things it holds constant; `recipe_change` discloses a table change beside a recipe change and gives the like-for-like move where one table spans it. The engine runs the new analyzer under the same per-analyzer guard and caches its history; the Model page's Findings tab shows the history when there is more than one table.

**Tech Stack:** Python 3.14, SQLAlchemy 2.0 (one raw SQL read, unchanged in shape), customtkinter (one text section), pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-process-recommendations-design.md` — finding catalogue #7, "Limit-table drift: more than one limit table in service for the same model at the same station. Lever: laser limit table, same day."

## Why now — what the data said (2026-09-20, work database, read-only)

Grouping every stored final sweep's limits by exact content, per (model, track name, laser), and counting a table only when at least 30 tracks used it:

- Of 109 groups with 100+ tracks, 58 have only ever used one table, 36 two, 15 three or more.
- **17 groups are graded against two or more tables inside their latest 12 months** — 10 at once, 7 as a change.
- **8232-1 on laser 1:** 89 graded points (111 rows, 0.5° steps) until 2025, 45 graded points (57 rows, 1.0°) since — the same band. 34 % against 49 % leave the laser inside limits. The 57-row table is the one laser 2 used for nine years and the one final test grades.
- **8506A and 8506B on laser 2:** every band loosened from ±0.01 V to ±0.0375 V in the first week of July 2025; 83 % → 100 % and 70 % → 100 %.
- **8340 on laser 1:** the later 61-point table is wider at four end positions by up to 0.164 V; 34 % → 82 %.
- It is per track name, so it is not a multi-track artifact.

And the confound is not hypothetical. The plan's author fed the existing `ink_target` analyzer two tables in two eras where resistance does nothing inside either table: pooled, it **manufactured an "aim lower" recommendation** (ρ ≈ −0.6) out of a change of test. With the table held constant it says nothing. On the real development slice, holding the table constant *uncovered* a lever that pooling had diluted below the threshold: 8340-1, 708–958 Ω, 50 % against 44 % over 1,033 tracks; and 8232-1's existing finding strengthens from +3.9 to +5 points.

## Global Constraints

Every task's requirements include these.

- **The app is an analysis tool, never a screening tool.** Findings point at a lever, never at a part.
- **Only four levers exist** (`findings/model.LEVERS`). This analyzer's is `laser_limit_table` (same day). **The ATP spec is never a lever.**
- **It must be able to say nothing.** One table in service, a second table of fewer than 30 tracks or under 10 % of the year, a table retired more than a year ago, two *tracks* of one model with different tables, two *lasers* with different tables — all produce NO finding. Every one has a negative test.
- **Never claim a gain.** A laxer table passing more units is a different test, not more yield: `expected_gain_points=None`, always.
- **Never say density explains a pass-rate gap.** Say the denser table is the stricter test; when the stricter table nevertheless passes more, say something else differs too.
- **Tell a change in time order** (Before / After; earlier / later) — never in order of track count.
- **A blank limit is an ungraded row, never 0.0**, and the pattern of ungraded rows is part of the table.
- **A final sweep of exact zeros is not a measurement** (the no-cut template, 1,182 tracks in a database ingested before 2026-09-20): the loader nulls its verdict and its errors but keeps its limits — they are the table that was in service.
- **Show the shop's laser names** via `laser_label()`; never "System B".
- **A failure is never silence:** the engine names a crashed analyzer in `facts["errors"]`; the new analyzer gets the same guard and the tab the same name mapping.
- **Database safety.** Never open `data/analysis.db` from a test or script. A test that builds a `Processor` sets BOTH `laser_trim_analyzer.database.manager._db_manager` and `laser_trim_analyzer.database._db_manager`.
- **No weak checks.** Every new test is seen to FAIL first; every named break is applied from a `cp` backup and restored with `cp` + `cmp` (never `git checkout --`). Clear `__pycache__` and run with `PYTHONDONTWRITEBYTECODE=1 … -B … -p no:cacheprovider`. Counts come from the junit `<testsuite>` element.
- **The gate is the whole suite**: `python scripts/run_test_gate.py` (one file per process) or one guarded process — about three minutes either way. A task is not done on its own files alone.
- `core/processor.py` and the parsers are not touched. Never `git add -A`. Do not push.
- Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` (an agent whose own attribution instruction names another model follows that instead).

## File Structure

| File | Responsibility |
|---|---|
| `src/laser_trim_analyzer/findings/data.py` | + `LimitTable`, `limit_table_of`, `_is_blank_template`; `TrackView` gains `track_name`, `final_positions`, `limit_table`; the loader reads `track_id` (the name) and `position_data` |
| `src/laser_trim_analyzer/findings/analyzers/limit_tables.py` | NEW — `compare(old, new)`, `analyze(model, tracks, laser_label) -> (history, findings)` |
| `src/laser_trim_analyzer/findings/analyzers/ink_target.py` | group key gains the limit-table fingerprint; the summary and evidence say so |
| `src/laser_trim_analyzer/findings/analyzers/recipe_change.py` | `_side` records the busiest table; a change of table is disclosed; `_like_for_like` |
| `src/laser_trim_analyzer/findings/engine.py` | runs `limit_tables` under the per-analyzer guard; `facts["limit_tables"]` |
| `src/laser_trim_analyzer/gui/v6/widgets/findings_tab.py` | a "LIMIT TABLES" section when the history has more than one row; the analyzer's human name |
| `scripts/app_qa_sweep.py` | the fixtures check asserts the new facts key |
| `tests/findings_helpers.py` | + `table()`, `on_table()`, `table_era()` |
| `tests/test_findings_limit_tables.py` | NEW — 12 tests |
| `tests/test_findings_{data,ink_target,recipe_change,engine_db,tab}.py` | + the tests named in each task |

---

### Task 1: The loader knows which test a track was held to

**Files:** Modify `src/laser_trim_analyzer/findings/data.py`, `tests/findings_helpers.py`, `tests/test_findings_data.py`.

**Interfaces — Produces:**
- `LimitTable(key: str, rows: int, graded: int, band: Tuple[Tuple[float, float], ...])` — frozen; `band` is (position, half-width) for graded rows, sorted by position, empty when the track has no positions.
- `limit_table_of(positions, upper, lower) -> Optional[LimitTable]` — None when there are fewer than 3 usable limit pairs or the arrays do not line up. Limits are rounded to 1e-5 before fingerprinting, so float noise is not a new table; an ungraded row IS part of the fingerprint.
- `TrackView.track_name: str = "default"`, `TrackView.final_positions: Optional[Tuple] = None` (both appended with defaults — every existing keyword construction keeps working), `TrackView.limit_table` (a `cached_property`).
- `_is_blank_template(errors) -> bool` — at least 10 real readings, all exactly 0.0.
- Test helpers: `table(n_points, half, span, end_half)`, `on_table(track, tab, name)`, `table_era(first_id, start, n, tab, good_share, *, system, name, step_days, cuts, r_in)`.

- [ ] **Step 1:** `git apply .superpowers/sdd/2026-09-20-limit-tables/task-1-tests.patch`. Read the two new tests.
- [ ] **Step 2: RED.** `tests/test_findings_data.py` → 10 tests, **2 failures**: `test_the_loader_carries_the_track_name_the_positions_and_so_the_limit_table` (`AttributeError: … 'track_name'`) and `test_a_stored_final_sweep_of_exact_zeros_loses_its_verdict_but_keeps_its_limits` (`ImportError: cannot import name '_is_blank_template'`).
- [ ] **Step 3:** `git apply …/task-1-src.patch`.
- [ ] **Step 4: GREEN.** `tests/test_findings_data.py` 10/0/0; then every other `tests/test_findings_*.py`, each alone — all must stay green (the new `TrackView` fields have defaults; nothing else may move).
- [ ] **Step 5: Teeth** (both validated by the plan's author). (a) In the loader, delete the two-line `if _is_blank_template(final_errors):` branch → exactly `test_a_stored_final_sweep_of_exact_zeros…` fails. (b) In `limit_table_of`, return `band=()` always → exactly `test_the_loader_carries_the_track_name…` fails (`len(tab.band) == tab.graded`). Restore after each, `cmp`.
- [ ] **Step 6: Commit** `src/laser_trim_analyzer/findings/data.py tests/findings_helpers.py tests/test_findings_data.py` — `feat: every track knows which limit table it was graded against`.

### Task 2: The limit-table analyzer

**Files:** Create `src/laser_trim_analyzer/findings/analyzers/limit_tables.py`, `tests/test_findings_limit_tables.py`.

**Interfaces — Consumes:** `TrackView.limit_table`, `.track_name`, `.system`, `.file_date`, `.linearity_pass`; `Finding`; `stats.pct`. **Produces:** `compare(old: LimitTable, new: LimitTable) -> dict` with `kind` ∈ {`same_band_other_density`, `same_band_same_points`, `different_band`, `not_comparable`}, `compared_positions`, `graded_old`, `graded_new`, and (when comparable) `wider`, `narrower`, `max_abs_diff`; `analyze(model, tracks, laser_label) -> (history, findings)`.

Constants: `MIN_TABLE_N = 30`, `MIN_SHARE = 0.10`, `WINDOW_DAYS = 365`, `MIN_OVERLAP_DAYS = 30`, `MIN_COMPARED = 5`, `BAND_TOL = 1e-5`, `BAND_REL_TOL = 0.01`, `MORE_PASSING = 5.0`.

Why interpolation: two tables rarely share a grid (a 0.5° table against a 1° one; two 171-row tables a fraction of a degree apart). Matching position for position left 8 of the 17 real cases "not comparable"; interpolating the older table at the newer table's positions left none — and corrected the author's own first reading of 8340 ("same band") to "wider at four end positions".

- [ ] **Step 1:** apply `task-2-tests.patch`. **RED:** 12 tests, **11 failures** (`ImportError`/`ModuleNotFoundError` on `limit_tables`). The one that passes is `test_a_table_is_its_content_not_its_float_noise`: it exercises only `limit_table_of`, which Task 1 already delivered.
- [ ] **Step 2:** apply `task-2-src.patch`. **GREEN:** 12/0/0.
- [ ] **Step 3: Teeth** (all four validated; each fails exactly ONE test). (a) `MIN_OVERLAP_DAYS = 100000` → `test_two_tables_at_once_same_band_other_density` (it becomes "changed"). (b) `older, newer = live[0], live[1]` instead of sorting by `first_of` → `test_a_change_is_told_in_time_order…`. (c) In `compare`, `dict(old.band).get(p)` instead of `_half_width_at(old.band, p)` → `test_bands_are_compared_by_interpolation…`. (d) Group by `(t.system, 'all')` → `test_two_tracks_of_one_model_may_carry_different_tables`. Restore after each, `cmp`.
- [ ] **Step 4: Commit** both new files — `feat: report a model graded against more than one limit table`.

### Task 3: The ink target holds the limit table constant

**Files:** Modify `findings/analyzers/ink_target.py`, `tests/test_findings_ink_target.py`. (May run in parallel with Task 4 — disjoint files.)

- [ ] **Step 1:** apply `task-3-tests.patch`. **RED:** 7 tests, **1 failure** — `test_a_change_of_limit_table_is_not_credited_to_resistance`: the pooled analyzer returns an "aim lower" finding where there is nothing. This failure IS the evidence that the confound is real; quote it in the report.
- [ ] **Step 2:** apply `task-3-src.patch`. **GREEN:** 7/0/0.
- [ ] **Step 3: Teeth** (validated). Make the third element of the group key `None` for every track → exactly that test fails again. Restore, `cmp`.
- [ ] **Step 4: Commit** — `fix: the ink target no longer credits resistance with a change of limit table`.

### Task 4: A recipe change says when the test changed with it

**Files:** Modify `findings/analyzers/recipe_change.py`, `tests/test_findings_recipe_change.py`.

- [ ] **Step 1:** apply `task-4-tests.patch`. **RED:** 7 tests, **3 failures** — the three new tests (`KeyError: 'limit_table_changed'`).
- [ ] **Step 2:** apply `task-4-src.patch`. **GREEN:** 7/0/0. Note the synthetic eras start on a quarter boundary on purpose: a quarter that mixes two recipes is never "stable", so tracks in it belong to neither side.
- [ ] **Step 3: Teeth** (both validated). (a) `table_changed = False` → the first two new tests fail. (b) In `_like_for_like`, `>= 10**9` in place of `>= MIN_SIDE` → exactly `test_the_like_for_like_move_is_given_when_one_table_spans_the_change` fails. Restore after each, `cmp`.
- [ ] **Step 4: Commit** — `feat: a recipe change discloses a limit-table change, and gives the like-for-like move`.

### Task 5: The engine runs it, the cache keeps its history, the tab shows it

**Files:** Modify `findings/engine.py`, `gui/v6/widgets/findings_tab.py`, `tests/test_findings_engine_db.py`, `tests/test_findings_tab.py`.

**Interfaces — Produces:** `facts["limit_tables"]`: `None` = not computed, `[]` = computed and nothing reached 30 tracks, else one row per table `{system, track, rows, graded, n, first, last, trim_pass_pct}` in time order. The pinned facts key set in `test_every_documented_key_exists…` grows by exactly this key — a deliberate update of a pin, in the same commit.

- [ ] **Step 1:** apply `task-5-tests.patch`. **RED:** `tests/test_findings_engine_db.py` 17 tests, **3 failures** (the pinned key set, and the two new tests); `tests/test_findings_tab.py` 11 tests, **1 failure** (the new one).
- [ ] **Step 2:** apply `task-5-src.patch`. **GREEN:** 17/0/0 and 11/0/0. Then every `tests/test_findings_*.py`, each alone.
- [ ] **Step 3: Teeth** (both validated; one test each). (a) In the engine, take the `limit_tables.analyze` call OUT of its `try` → `test_a_crash_in_the_limit_table_analyzer_is_named_like_any_other` (the exception escapes). (b) In the tab, `len(tables) > 0` in place of `> 1` → `test_more_than_one_limit_table_gets_its_own_section_and_one_does_not`. Restore after each, `cmp`.
- [ ] **Step 4: Real data.** On a COPY of `Work Files/dev_db/slice_v2.db` (never the original): `refresh_findings(db, None, report)` with both globals set. Expected, from the author's run: 8 findings in under 30 s, `failed_models == {}`, `analyzer_errors == {}`; limit-table findings for 8232-1 (89 against 45 graded points, same band, concurrent) and 8340-1 (53 against 99); an ink-target finding for 8340-1 that did not exist before (708–958 Ω). Delete the copy. Report what you got.
- [ ] **Step 5: Commit** — `feat: limit tables in the findings engine and on the Model page`.

### Task 6: Sweep, gate, docs

**Files:** Modify `scripts/app_qa_sweep.py` (the findings fixtures check: assert `facts.get("limit_tables") == []` — four tracks never reach 30 — beside the existing key checks, made to fail first by seeding the key as `None` in the engine), `TRACKER.md`, the spec's status line.

- [ ] **Step 1:** the sweep assertion, with its fail-first break.
- [ ] **Step 2:** `python scripts/run_test_gate.py` — the whole suite; every file OK.
- [ ] **Step 3:** `python scripts/app_qa_sweep.py --only findings` → exit 0.
- [ ] **Step 4:** TRACKER B6: tick "limit tables (#7)"; add the 17-group result and the two questions for James (were the 8506A/B limits loosened by ECN? which 8232-1 table is the intended one?).
- [ ] **Step 5: Commit** — `test: the limit-table analyzer is guarded in the sweep`.

---

## Self-review (done while writing)

- **Spec coverage.** Catalogue #7 asks for "more than one limit table in service for the same model at the same station" → Task 2, per (laser, track name). The spec's lever and lead time → `laser_limit_table`, same day. "It must be able to say nothing" → seven negative tests in Task 2 plus the unchanged-table case in Task 4. Not in this plan, deliberately: comparing the laser's table with FINAL TEST's (catalogue #2, station setup mismatch — needs the rebuilt final-test data), and filtering stale models off the Findings page (belongs to the page, using the backlog's active-model list).
- **Placeholders.** None: every task names its patch pair, its RED count and reason, its GREEN count, and its breaks. Every RED count, every GREEN count and all eleven breaks were RUN by the author before this was committed — the patches applied in order to a fresh worktree of the branch, and each break failed exactly the tests named. An implementer who sees anything else must stop and report, not improvise.
- **Type consistency.** `analyze(model, tracks, laser_label)` as for every analyzer; `limit_tables` returns `(history, findings)` like `recipe_change`. `TrackView.limit_table` is `Optional[LimitTable]` everywhere; analyzers key on `.key`.
- **Proven on real data before writing:** prototype over the whole work database (280 models → 17 findings, all 17 comparable once interpolation replaced exact matching); reference implementation over the rebuilt slice (8 findings, 18 s, no analyzer errors).
