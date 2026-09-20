# Whole-app code review — 2026-09-20

You asked whether the app "may need a full review and refactor". This is the review.
Three reviewers read the whole codebase in parallel (parsers + analysis + ML + exports;
the database layer + ingest path; both UIs + scripts + the test suite) and checked every
claim about data against a **read-only** connection to the work database. 45 findings.
The full notes — code quoted, queries shown, fix and proof for each — are in
`.superpowers/review-2026-09-20/` (three files, ~2,100 lines). This page is the ranked
summary.

**Status legend:** FIXED = committed locally tonight, tested and reviewed · DECIDE = needs a
call from you · LATER = real, not urgent. Commit hashes are local ones on `V6`.
Nothing here is pushed; see the top of `TRACKER.md`.

**Two corrections to the reviewers' own findings, made by testing them.** (1) The test-suite
hang was blamed on the garbage collector calling into Tk from a worker thread. A controlled
2×2 experiment showed that is *not* the cause (§4, #25). (2) "An all-zero error column is
accepted as a flawless part" turned out to be something more specific and worse: files in
which **no cut was made at all** (§1, #3). A review is a list of hypotheses; these two were
wrong in the mechanism and right that something was badly off.

---

## The short answer

**The app does not need a rewrite. It needs three specific diseases treated**, and almost
every finding below is one of them:

1. **The same rule is written in several places, and the copies disagree.** "Is this point
   in spec" exists five times with three different policies for a blank cell. "Yield"
   exists four times and the copies give **12.3 % and 76.0 % on the same data**. The
   production-database refusal existed four times and none of them caught
   `data/Analysis.db`. Every one of tonight's worst findings is a second copy that drifted.
   The cure is not restructuring — it is making each rule live in exactly one function.
2. **A failure is allowed to look like a result.** A crashed query renders as "No data for
   this model". A blank measurement is stored as `0.0` — dead centre of the band, the most
   flattering value a zero-tolerance metric can hold. An ungradeable final test is stored
   as PASS. A `999.999` "error" marker is averaged into a mean. On an analysis tool whose
   honest answer is sometimes "nothing to act on", silence and failure **must** look
   different.
3. **The safety nets could not fail.** The test suite was believed to take 2.5 hours — it
   was **hanging**, so nobody ran it, so gates were hand-picked lists, and twice today a
   broken test reached `main` that way. It now runs **1,770 tests in about 3 minutes**.
   `pytest -q` printed no pass count at all (a run that collected nothing looked green).
   The "645 real files" parse gate compared only a coarse status rank, not the numbers.
   Two sweep checks are tautologies. Five scripts write to the production database by default.

What is genuinely *good*, and worth saying: the ingest pipeline's design (content-hash
scan, per-file guards), the trim-capture work, the V6 page structure, the UI-thread
dispatcher, and the domain rules in `CLAUDE.md` are sound. The rules are right. The code
does not yet obey them everywhere.

---

## 1. Things that change what gets STORED — settle before a rebuild

| # | Finding | What it costs you | Status |
|---|---|---|---|
| 1 | **Final-test errors recomputed from measured-vs-theory are divided by full scale, then graded against VOLT limits.** Fires whenever a sheet's error column is under 90 % populated — which is normal when the station ignores more than 10 % of its rows. | A final test the station FAILED is stored as a PASS. Reproduced on a fixture in this repo (7539-2 sn23): station flags 149 of 161 points, the app stored 0 fail points, PASS. | **FIXED** `0de5035` — evidence in §5 |
| 2 | **The byte cache and hash cache shared by the ingest worker threads had no lock** (the byte cache is *my* code from this morning, and it is on `main`). | A `KeyError` between one worker's two steps is recorded as an ERROR row — a **good file lost, silently**, in an unattended rebuild. Reproduced: 325 exceptions in 6 s under a forced-switch loop. | **FIXED** `99d8937` |
| 3 | **Files in which NO CUT was made are stored as flawless trimmed units.** Reported as "an all-zero error column is accepted"; what it really is: when the operator takes a sweep and makes no cut, laser 1 still writes a `Lin Error` sheet — the blank template, measured = theory to 1e-12. The parser took it for the final sweep. (Corrected 2026-09-20 after James: laser 3 writes laser 2's sheets, not laser 1's, and none of these tracks is laser 3.) | **1,182 tracks** (6126: 455 · 6607: 224 · 8340: 193 · 8232-1: 163 · 1844205: 92), 1,181 of them linearity PASS, nearly all laser 1 since 2023. In your slice the workbook's REAL sweep is **out of limits in 111 of 130**. 8232-1's laser-1 yield is 42.0 % with them and **39.1 % without**. | **FIXED** `e16808e` + a follow-up in review tonight — they become UNTRIMMED |
| 4 | **A final test nothing could grade is PASS at file level and FAIL at track level.** | 289 files with no verdict read `overall_status='PASS'`; 735 such tracks read `FAIL`. Only 0.19 % of files — but per model it bites: **8502 reads 27.6 % FT pass; over the files that were actually graded it is 15.5 %**. 8513: 19.8 % vs 12.1 %. | DECIDE (§6) |
| 5 | **Ingest writes trims and final tests through two different database handles** (`Processor` has no `db` parameter; final-test and smoothness saves go through the `get_database()` global). | In the app both resolve to the same database, so production is fine. In **tests and scripts** they do not — it is the mechanism behind the three accidental opens of the production DB, and it means neither QA gate can prove where rows went. | LATER (M) — the both-globals rule is the workaround |
| 6 | **"Fix Missing Tracks" writes invented numbers** (`sigma_gradient = 0`, `sigma_pass = True`, `linearity_spec = 0.02`) into the trim table. | Every repaired row enters Cpk, drift and the ML thresholds as a measured perfect part, and nothing tags which rows were repaired. | DECIDE — write NULLs instead (S) |
| 7 | **A trim track with no usable per-point limits is graded PASS, not "cannot grade".** | Nothing today (0 rows match). It is a loaded gun: the first template with an empty limit block produces a silent run of passes. | LATER (S) |
| 8 | **Final Test Format 4 reads position from a column that is constant 0.0.** | Every Format 4 final test plots all its points at x = 0 and gets a meaningless sigma. Verified on one file; only four Format 4 files were ever available. | LATER (S) |

## 2. Wrong numbers on a screen or in a workbook

| # | Finding | What it costs you | Status |
|---|---|---|---|
| 9 | **Four yield rules.** The Excel export's "Pass Rate" counts WARNING as a failure; the documented `linearity_yield` counts it as accepted (sigma is never a rejection — your rule). | Same database, **12.3 % vs 76.0 %**. 50,465 of the 50,564 WARNING analyses have every track passing linearity. One of those numbers is on a sheet headed "Batch Analysis Summary". | DECIDE — one definition (PASS + WARNING), one helper (M) |
| 10 | **`999.999` error markers are averaged.** The evidence pack's "Mean sigma gradient" for 8856 reads **432.99** for 2024-07; the true mean is 0.0012. The drift detector's sample loader fed them to the lot medians too. | A 350,000× error in a workbook you hand to engineers. | **FIXED** `3deaeae` — one shared definition, both consumers |
| 11 | **On the V6 Model page a crashed query renders as "No trim / final-test data for this model"**, and if the verdict query fails for model B the header keeps showing **model A's verdict and pills** under B's name. | The only finding in the UI where a wrong number can reach a process decision. | **FIXED** `9222a30`, `46be6a2` — one banner names what failed; the header is always reset; a failed drift-status load no longer reads "NOT TRAINED" |
| 12 | **Triage and Home reload on an unguarded worker.** If the query raises, the thread dies and the page keeps showing the old list with no hint. | A stale "what do I work on today" list. | LATER (S) |
| 13 | **One future-dated final-test file hides 996 of a model's 997 records** behind the 30/90-day window (the window anchors on the newest date). | One model today; one bad cell in one workbook will do it again. | LATER (S) — clamp the anchor to today; make the sweep FAIL on future dates |
| 14 | Dashboard "failed" tile: `failed = count` should be `failed += count` — the ERROR bucket is overwritten by FAIL. Headline and trend also use different windows, so the bars never sum to the tile. | Small today (112 rows), but two tiles disagree by construction, and every cross-check costs an investigation. | LATER (S, one character) |
| 15 | `linearity_spec` is computed two ways; the final-test one is the one the trim parser's own comment explains is wrong (8639-30 reads 58 % too wide). | Feeds the displayed "spec" and two ML features. | LATER (S) |
| 16 | Exported unit charts label a trace "corrected" on exactly the units where the correction failed. | A misleading PDF, not a wrong stored number. | LATER (S) |
| 17 | Cpk and the ML features treat `linearity_spec_pct` (a percent) as volts, and fabricate 0.0 from NULL. **Dormant:** `model_specs` has 0 rows. It switches on with the first spec you load — which is where the ATP audit is heading. | "Incapable" would print as "Excellent" on a 10 V part. | LATER — but BEFORE loading any specs |

## 3. Speed, locking and the database file

| # | Finding | Status |
|---|---|---|
| 18 | One lock is held for the whole of every session, and a **full** `rematch_final_tests` can be triggered *at application startup* by a migration. This is the shape of the 2026-09-14 "broken processing" night. Fix: delete the startup call (the cheap scoped rematch already runs after every batch). | DECIDE (S) |
| 19 | If the processed-file index fails to load once (e.g. "database is locked"), the whole share is silently reprocessed — a WARNING line, nothing else. | LATER (S) |
| 20 | Every ingested file gets its own transaction and fsync; `save_batch` exists and has no callers. | LATER (M) — goes with tracker A3 |
| 21 | The stat-heal issues three UPDATEs per queued file (452,814 statements for the 150,938-entry heal). | LATER (S) |
| 22 | A full rematch can remove existing final-test links and never reports how many; cleanup silently unlinks final tests and the preview does not say so. | LATER (S each) |
| 23 | Startup writes to the production database on every launch (a 416-row backfill that can never finish), so the app cannot be opened read-only. | LATER (S) |
| 24 | 853 lines of `manager.py` are called from nowhere. | LATER (S) — pure deletion, do it first |

## 4. The safety nets

| # | Finding | Status |
|---|---|---|
| 25 | **The test suite hung; it was never slow.** A test fixture kept one Tk window alive for the whole run while individual tests built and destroyed app windows. With two or three live windows in one process, Tk's `update()` never returns: self-rearming timers keep its queue non-empty (~94 callbacks a second, forever, at full CPU — indistinguishable from "slow"). The reviewer blamed the garbage collector calling Tk from a worker thread; a 2×2 experiment (fixture scope × a guard on that) showed the fixture alone cures both reproductions and the guard alone cures neither. **Whole suite now: 1,770 tests, 0 failures, 3 min 5 s, in one process.** The guard landed anyway on its own measured merit (a font dropped on a worker stalls it 1.07 s). | **FIXED** `7d8a91e` |
| 26 | `pytest -q` plus the `-q` already in `pyproject.toml` = verbosity −2, and pytest prints **no pass count at all**. This bit me four times today. | **FIXED** `f260fed` |
| 26b | **The "645 real files" parse gate compared only `error < empty < ok`** — numbers for just five models — yet two documents (one of them mine) called it a full diff. A fix tonight rerouted a whole class of files and it never blinked. Audit of every stored field: 633 of 641 identical, the other 8 all explained and deliberate. | **FIXED** `ff1e62b` — every frozen value is compared now |
| 27 | **Five scripts write to `data/analysis.db` by default** (`backfill_trim_effort.py` with no dry-run at all; `fix_smoothness_tracks.py` cannot even be pointed elsewhere). The QA harnesses refuse that path; these never got the same treatment. | DECIDE (S) — I would do it; say the word |
| 28 | The production-DB refusal compared path *text*: `data/Analysis.db` (same file on a Mac or NTFS volume) or a hard link walked straight past it. Proven with a decoy: the old refusal wrote 434 KB into the decoy "production" file. And the tests of the refusal **named the real database in their command line** — a regression would have opened it before going red. | **FIXED** `cc133c2`, `4cbcfac` — one guard (`scripts/_db_guard.py`) compares the FILE, used by both QA harnesses, the stall probe, the dev-DB builder and the findings tool; no test names the real database any more |
| 29 | Two `app_qa_sweep.py` checks cannot fail, and data-quality problems are WARN, which the exit code ignores. | LATER (S) |
| 30 | Test databases are never closed; handles pile up for the whole run. | LATER (S) |

## 5. The final-test units fix — the evidence

One deletion in two places (stop dividing by full scale), landed test-first, then measured
**through the real pipeline, before and after, on real files**:

| | |
|---|---|
| Home slice, 3,774 final-test files | 3,768 take the "use the sheet's own errors" branch — **zero change on all 3,766 of those tracks, 13 fields each**. 3 take the recompute branch. |
| Sample base, 690 files across ~350 models | 9 more recompute-branch tracks |
| The 12 affected tracks | **6 flip, all PASS → FAIL.** Agreement with the station's own verdict **5 of 12 → 11 of 12**. No station-PASS became a FAIL. |
| Scale | ≈ 1.3 % of final-test files take the branch → on the order of 2,000 files and several hundred verdict changes across your ~151,000 final-test rows |

One thing it overturned: a test in this repo had recorded the app's PASS on 7539-2 sn23 (station:
149 of 161 points flagged) as *"the offset correction rescues it — a real and expected
disagreement"*. It was this bug. In volts the best offset still leaves 11 points out. When the
app disagrees with the station in the **forgiving** direction, check units before crediting the
offset correction.

**Why it matters for the rebuild:** the rebuild re-grades every final-test file through this
branch. If the rebuild runs first, the wrong-unit errors are what gets stored, and the repair is
a re-grade pass that re-parses ~151,000 tracks at about 80 a minute.

## 6. Decisions I did not make for you

- **No-cut files (1,182 tracks) — done, but one question is yours.** They now load as
  UNTRIMMED, the app's existing status for "a test sweep with no laser-trim run", so they leave
  yield and baselines. The open question: such a file IS a real measurement of the unit at that
  moment (109 of the 130 in your slice have other files for the same serial the same day — they
  are check sweeps between trimming sessions, often named "Trimmed Correct" or "Touch-up_Lin
  Out"). Should a no-cut check sweep count for anything?
- **An ungradeable final test.** Proposal: one new status, `UNGRADED`, used by both the
  file and the track writer, excluded from every rate's denominator.
- **Yield.** Proposal: PASS + WARNING everywhere, one helper, and rename the export column
  so nothing can quietly keep the old meaning.
- **"Fix Missing Tracks".** Proposal: write NULL, not invented numbers.
- **The startup rematch.** Proposal: delete the automatic call.
- **Format 2 final tests** also normalise by full scale, but they carry no limits, so it
  only changes the displayed magnitude. Left alone pending your call.

## 7. The refactor, such as it is

`database/manager.py` is 10,139 lines, but it is eight things, and six of them barely touch
the rest. The reviewer's order — one commit each, each proven by both gates plus the sweep:

1. delete the 853 dead lines → 2. `database/migrations.py` (~1,100 lines; fix the startup
rematch while there) → 3. `database/specs.py` → 4. `database/ft_matching.py` (the two
matchers have already drifted apart twice) → 5. `database/maintenance.py` →
6. `database/smoothness.py` → 7. **stop.** What remains — the save paths and the aggregate
readers — is where the numbers are made; the win there is single-sourcing the verdict
rules, not moving files.

**Retiring V5** removes ~15,900 lines and three findings with them (raw "System A/B"
labels, 215 uncached fonts, a second set of yield rules). Six steps, in the full notes:
move the chart widget into `gui/v6/`, extract four pure functions that tests import from V5
pages, flip the default entry point and live with it for a week, then delete.

## 8. Something the review did NOT look for, found on the way

**17 model/laser combinations are graded against two or more limit tables inside their latest
year** (work database, read-only; a table counted only with 30+ tracks; per track name, so it is
not a multi-track effect). 8232-1 on laser 1: 89 graded points until 2025, 45 since — the same
band, and the 45-point table is the one final test uses; 34 % vs 49 % leave the laser inside
limits. 8506A and 8506B on laser 2: every band loosened from ±0.01 V to ±0.0375 V in the first
week of July 2025 (83 % → 100 %, 70 % → 100 %). 8340 on laser 1: the later table is wider at four
end positions by up to 0.164 V (34 % → 82 %).

A pass rate is a verdict against a test. Every trend across such a change compares two tests.
It also confounds the new findings engine: fed two tables in two eras, its ink-target analyzer
**manufactured an "aim lower" recommendation out of a change of test**. A plan to make the limit
table a held-constant fact, with an analyzer that reports these (spec catalogue #7, lever: laser
limit table, same day), is written and validated on your data:
`docs/superpowers/plans/2026-09-20-limit-tables.md`.

## 9. If you only do five things

1. Take the fixes in §1 into the rebuild (they are ready; say "push").
2. Decide the ungraded-final-test rule and the one definition of yield (§6) — they are in every
   number the rebuild will produce.
3. Make the *whole suite* the gate from now on: `python scripts/run_test_gate.py`, three minutes.
4. Tell me whether you knew about the limit-table changes (§8) — especially 8506A/B.
5. Let me make the five backfill scripts refuse the production database unless told otherwise.
