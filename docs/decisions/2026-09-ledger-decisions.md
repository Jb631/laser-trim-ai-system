# Ledger decisions, pulled out of `.superpowers/`

`.superpowers/` is git-ignored — it holds the working ledgers (`progress.md`) and per-task
briefs/reports for five build plans run between 2026-09-17 and 2026-09-20. TRACKER.md item H3
(2026-09-20) audited all five: 130 judgement calls total, 56 already durable somewhere else, 51
pure process (who reviewed what, which model, dispatch order), and 23 that existed **only** in
the ledgers — plus ten tracked files that cite specific ledger files by name, which would break
if the workspaces were deleted.

This document is that inlining, written 2026-09-24. It does two jobs: **records the substance**
of the judgement calls that would otherwise be lost (below), and **records what the four
by-name-cited files actually say**, so the citing code's claim is checked and preserved even
though this document doesn't (yet) replace those citations — see the closing list. Nothing here
is customer data; nothing here changes code.

## Decisions that live only in the ledgers

I read all five `progress.md` files in full — `2026-09-17-trim-capture`, `2026-09-20-limit-tables`,
`2026-09-20-process-findings-engine`, `E1-backlog-upload`, `prebuild-fixes` — checked every
judgement call against the current code with `git grep`, and kept only the ones that are (a) an
actual decision (a choice, a reason, a cost — not just "no test covers this") and (b) not written
down anywhere durable today. That leaves 10, below. Two more are explicitly closed already (see
end of this section), and three of TRACKER's own named examples of "the other 20" — the Response
field, the float-in-an-Integer-column, and the two differently-scoped pass rates — are all in this
list, which is some evidence this pass found the right *kind* of item even where the count differs.
On the count itself: see "Reconciling the count" below.

### From the trim-capture ledger

**1. The "Response" field may be one field recorded under two different keys.**
System B's per-pass label is `"Response"`; System A's is `"Response (Linear or Function)"`
(value `"LINEAR"`). They might be the same underlying field, recorded differently by the two
machines — but the parenthetical-stripping rule added for the mm/count cut-length collision
(`"(mm)"` is a unit, kept; `"(0-255)"` is a calibration range, dropped) also keeps
`"(Linear or Function)"`, so the two machines now produce two separate keys where the old,
buggier rule silently merged them into one. Nobody has determined whether the split is correct
or an accident of a rule built for a different problem.
*Where:* `src/laser_trim_analyzer/core/trim_setup.py` (`normalise_key`, `read_per_pass`) — the
values land in `TrimPass.recipe` via `database/manager.py: DatabaseManager._write_trim_passes`.
*Ledger:* `.superpowers/sdd/2026-09-17-trim-capture/progress.md:70-73`.

**2. A pass's ignored-point counts are stored as floats in Integer-typed columns.**
`_write_trim_setup` runs `_as_float()` over every promoted scalar except `indexing_method` —
including `points_ignored_start`/`points_ignored_end`, which are declared `Integer` columns — so
the database ends up holding `4.0`, not `4`.
*Why kept as-is:* harmless under SQLite's column-affinity typing, and it was the brief's own
reference code rather than an implementer's deviation, so nobody has revisited whether to
special-case the two integer columns.
*Where:* `src/laser_trim_analyzer/database/manager.py: DatabaseManager._write_trim_setup`.
*Ledger:* `.superpowers/sdd/2026-09-17-trim-capture/progress.md:237-239`.

**3. Promoted trim-setup values are stored twice.**
`TrimPass.recipe` and `TrimSetup.parameters` store the *whole* per-pass/per-setup dict as JSON,
including keys that are *also* promoted to their own dedicated columns — so some values exist
both as a column and inside the JSON blob for the same row.
*Why:* Plausibly intentional (a reader of the JSON blob doesn't need to know the promoted-column
mapping to get a complete picture), but this was never confirmed as deliberate, and no test
pins it either way.
*Where:* `src/laser_trim_analyzer/database/manager.py: DatabaseManager._write_trim_passes` and
`_write_trim_setup`.
*Ledger:* `.superpowers/sdd/2026-09-17-trim-capture/progress.md:240-241`.

### From the limit-tables ledger

**4. The limit-table history strip and a limit-table finding show pass rates on different scopes.**
`limit_tables.analyze()` returns two kinds of pass-rate numbers for the same model: each
`history` entry reports a table's **all-time** pass rate (every track ever graded against it),
while a `Finding`'s own `older`/`newer` percentages are **windowed** to the last 365 days. Both
can appear on the same Model-page tab, unlabeled as to scope.
*Why not reconciled:* the history strip's job is "what tables has this model ever used" (all-time
is the natural scope); the finding's job is "did the recent change move the result" (so it scopes
to the window being evaluated). The decision was to ship both, not to pick one scope or label them.
*Where:* `src/laser_trim_analyzer/findings/analyzers/limit_tables.py: analyze` — compare the
`history.append(...)` block against `p_old`/`p_new`.
*Ledger:* `.superpowers/sdd/2026-09-20-limit-tables/progress.md:37` (Minor 11).

**5. A large second limit table can still go unreported on a high-volume model.**
`MIN_SHARE = 0.10` requires a second table to carry at least 10% of a model's last-year tracks
before it counts as "in service" and can produce a finding. On a high-volume model, several
hundred real tracks on a genuine second table can still be under that 10% and never surface.
*Why:* `MIN_SHARE` was tuned to filter one-off straggler files, not scaled for absolute volume;
nobody has revisited it for the high-volume case.
*Where:* `src/laser_trim_analyzer/findings/analyzers/limit_tables.py` — module constant
`MIN_SHARE`, used in `analyze`'s `live = sorted(...)` filter.
*Ledger:* `.superpowers/sdd/2026-09-20-limit-tables/progress.md:37` (Minor 13).

**6. Comparing two limit tables by half-width alone is safe only because band centres hold still.**
`LimitTable.band` stores only `(position, half-width)` pairs, discarding the actual centre of
each `(upper, lower)` pair. So the bracket-test comparison in `compare()` can only ever detect a
table getting wider or narrower — never one whose centre moved while its width stayed the same.
*Why this is safe today:* checked once, empirically: 81,016 of 81,017 real tracks surveyed had a
constant band centre (one junk row on a single model). That number lives only in this ledger
line — if it ever stops holding for some model, this reduction would miss a real change silently.
*Where:* `src/laser_trim_analyzer/findings/data.py: limit_table_of` (the `band` construction) and
`findings/analyzers/limit_tables.py: compare`.
*Ledger:* `.superpowers/sdd/2026-09-20-limit-tables/progress.md:28` (the "STATE at 18:20" note,
last sentence).

### From the process-findings-engine ledger

**7. The findings engine's pass-row query is asymmetric with its track-row query, on purpose.**
`load_model_tracks`'s `pass_rows` SQL has no `a.system IN ('A','B','C')` filter, while its
`track_rows` SQL does. The asymmetry was judged harmless — an unmatched pass row is never looked
up — rather than made consistent.
*Why not fixed:* fixing it means touching a query with no observed defect, for consistency alone;
the call was to leave it and record why, not to spend a review cycle on it.
*Where:* `src/laser_trim_analyzer/findings/data.py: load_model_tracks`.
*Ledger:* `.superpowers/sdd/2026-09-20-process-findings-engine/progress.md:40`.

**8. A warning promised to the code-review backlog never actually arrived there.**
Constructing `Processor(use_ml=False)` — which `tests/test_findings_data.py:17` does, to run
real files through the pipeline for per-pass data — still unpickles sklearn estimators and
prints an `InconsistentVersionWarning`, even though `use_ml=False` correctly skips
`_load_ml_thresholds()`. The reviewer called it pre-existing and out of scope, and named it a
candidate for TRACKER's C1 code-review list — but it was never added to
`docs/CODE_REVIEW_2026-09-20.md` or TRACKER.md, so today nothing outside this ledger records it.
*Where:* `src/laser_trim_analyzer/core/processor.py: Processor.__init__` (the `use_ml` gate) —
the unpickling path itself wasn't localized further than "pre-existing" by the review that found
it, so it isn't pinned to one line here either.
*Ledger:* `.superpowers/sdd/2026-09-20-process-findings-engine/progress.md:41`.

**9. Ink-target's "current incoming window" has two unhandled edge cases.**
The station's configured incoming-resistance window is read as
`max(with_window, key=lambda t: t.file_date)` — the most recent track that carries one. Two
gaps were found and left unhandled: an exact tie on `file_date` silently falls back to whatever
order the underlying list happens to be in (effectively resistance order, since that's how the
list arrives); and `if t.initial_r_low and t.initial_r_high` treats a legitimate `0.0` Ω bound
the same as "no window at all," because `0.0` is falsy in Python.
*Why not fixed:* both are "theoretical" — no real model has hit either case (no real model has a
0 Ω limit) — so a blind fix was deferred rather than made without a test to justify it.
*Where:* `src/laser_trim_analyzer/findings/analyzers/ink_target.py: analyze` (the
`with_window`/`configured` block).
*Ledger:* `.superpowers/sdd/2026-09-20-process-findings-engine/progress.md:68-69`.

**10. Tests patch two different `_db_manager` globals, and only one of them does anything.**
The project's convention for injecting a temporary database into a test is to monkeypatch *both*
`laser_trim_analyzer.database.manager._db_manager` and `laser_trim_analyzer.database._db_manager`
— see `tests/test_trim_capture_db.py`'s setup for an example. `get_database()` only ever reads
the first one; the second patch is currently inert.
*Why kept anyway:* belt-and-braces, in case a future code path resolves the database through the
package-level name instead of calling `get_database()`; patching with `raising=False` makes the
extra, currently-useless patch cost nothing.
*Where:* `src/laser_trim_analyzer/database/manager.py: get_database` (reads only its own module
global) — the dual-patch pattern itself recurs across many test files.
*Ledger:* `.superpowers/sdd/2026-09-20-process-findings-engine/progress.md:43`.

### Already closed (recorded here only for completeness)

TRACKER.md's H3 entry names two of the original 23 as fixed and made durable before this
document existed — both are real code, not ledger text, so they are **not** repeated above:

- **A trim pass with a missing `pass_index` no longer risks the whole file's save.**
  `DatabaseManager._write_trim_passes` (`database/manager.py`) now drops such a pass with a
  logged warning instead of letting it reach `pass_index NOT NULL` at flush, which would have
  rolled back every track's verdict for that file. TRACKER.md "Done recently"; ledger:
  `2026-09-17-trim-capture/progress.md:233-236`.
- **The two copies of the pass-sheet regex can no longer drift apart.**
  `parser.py`'s `_PASS_SHEET_RE` and `trim_passes.py`'s `_A` are pinned identical by
  `tests/test_trim_passes.py:385-402`. TRACKER.md "Done recently"; ledger:
  `2026-09-17-trim-capture/progress.md:177-178`.

### Reconciling the count

TRACKER's H3 says 23 lived only in the ledgers, 2 now closed (above) — 21 remaining by that
arithmetic, though H3's own next sentence calls the remainder "the other 20," a small
inconsistency in the audit note itself. This document records 10. Three likely reasons for the
gap, roughly in order of how much of it I'd attribute to each:

1. **Scope.** The 130-call audit evidently drew on the full ledgers — every `task-N-brief.md`
   and `task-N-report.md`, not just `progress.md`. This document's brief scoped the read to each
   `progress.md` in full (plus the four cited files). A judgement call recorded only in a task
   report's body, and only summarized tersely or not at all in `progress.md`, would not surface
   here. Spot-checks of the task reports found no "Minor" lists of their own — review
   classifications live in `progress.md` — which is why I didn't read all of them in full, but it
   doesn't rule out a call recorded some other way in one.
2. **Three more days of work landed on this branch** between the audit (2026-09-20) and this
   document (2026-09-24). This pass found the live code already documenting far more than
   expected — most "Ruling N" decisions in the ledgers turned out to already have a matching
   docstring or comment (e.g. the entire limit-tables review fix — C1, C2, C3, I4, I5, I6, I7 —
   is written into `limit_tables.py`, `ink_target.py`, and `recipe_change.py`'s own module
   docstrings). Some of the 21 may have been closed as an incidental side effect of other work,
   the same way two items were closed on purpose.
3. **A stricter bar.** I excluded "no test exercises path X" notes that don't carry a stated
   reason and cost — the task's own instruction to exclude "pure process" reads, to me, as
   excluding bare test-coverage gaps too, since they're not a choice between alternatives so much
   as an acknowledged gap. The audit may have counted some of these; this document doesn't.

## What the cited notes said

Four ledger files are cited by path, by name, in eight lines across eight tracked files. Each is
read in full below, condensed to what a reader of the *citing* code needs — not everything in
the brief/report, just the substance the citation is standing in for.

### `H5-report.md` — cited by `utils/threads.py:210`, `tests/conftest.py:128`, `tests/test_tk_thread_safety.py:4`

The project believed the test suite was slow (~2.5 hours). It wasn't slow — it **hung**, at over
100% CPU, which looks identical to slow from outside. The report's headline: a 188-second suite
(1,769 tests, 0 failures, 0 errors, 5 deliberate skips) that used to hang indefinitely.

**The mechanism**, nailed down with a 2×2 experiment (fixture scope × a font-finalizer guard,
each on/off, against two independent reproduction cases): a session-scoped `tk_root` pytest
fixture left two or three live CTk roots alive in one process. Tk's macOS idle handler pumps the
Cocoa event loop from *inside* `Misc.update()`, so CustomTkinter's self-rearming `after()` timers
(`AppearanceModeTracker`, `ScalingTracker`) and the app's own `UiDispatcher` poll keep re-arming
there — `update()` never sees an empty event queue and never returns. **Function-scoping
`tk_root` (F) cures both reproduction cases on its own. A font-finalizer guard (G) cures
neither. F+G equals F exactly.** The originally-suspected mechanism — the garbage collector
calling into Tk from a worker thread — is ruled out directly: every hung process's thread dump
shows exactly one thread (the main one); a census of 28,948 Tk-dispatched callbacks found 100%
on the main thread.

**G was kept anyway, on separate, measured merit, not because it fixes the hang.** Probing what
an unguarded `tkinter.font.Font.__del__` does today: on a worker thread it blocks that worker for
1.07 seconds inside `_tkinter`'s `WaitForMainloop`, then raises `RuntimeError: main thread is not
in main loop` (which `__del__` swallows) — and doesn't even delete the font from Tcl anyway. The
guard removes the stall and the forbidden off-thread Tk call (a hard project rule: workers never
call Tk); the font still leaks either way, so nothing is lost by guarding it. The report is
explicit that no worker-thread font finalization was *observed* in any of its own runs — the
guard is insurance against a cost that is real *if* it happens, not a fix for something seen.

Also landed as a separate commit: pytest's `-q` had been doubled (once in `pyproject.toml`'s
`addopts`, once implicitly), which pushed verbosity to a level that printed **no pass count at
all** — a run that collected zero tests looked exactly like a clean pass. Fixed so a
no-collection run now reports its exit code and a real count.

*Citing lines:* `threads.py:210` cites the measured 1.07 s worker stall as the justification for
`guard_tk_font_finalizer()`. `conftest.py:128` cites the report for the cost of function-scoping
(~0.03 s per fresh root, plus a larger Tk teardown bill in a couple of heavy test files).
`test_tk_thread_safety.py:4` cites the 2×2 experiment for "these are two separate bugs, found
together, and the experiment is what kept them apart."

### `M1-brief.md` — cited by `tests/test_model_page_failures.py:4`

The defect: `ModelPage._reload()` runs about 14 loaders on a worker thread, each independently
`try`/`except`-guarded and falling through to an empty default on failure (`{}`, `[]`, `None`).
`apply()` then hands those defaults to the tabs, which render an empty default as a **statement
of fact** — e.g. "No trim / final-test data for this model." A crashed loader and a model that
genuinely has no data produce the identical sentence; the only trace is a line in the log file.

Worse: two of the applies were *conditional* (`if status: ...`, `if verdict: ...`). Since
`self._verdict` is one persistent label reused across models, a crash loading model B's status or
verdict left **model A's verdict and pills on screen under model B's name** — a plausible number
attributed to the wrong thing, on the screen used to decide what to change about a process.

The brief specifies deliberately the smallest fix that removes the hazard, not a rewrite: (1) a
`failed` list of short, human-readable loader names built on the worker, appended to in each
loader's `except`; (2) **one** load banner (not fourteen separate widget changes), shown when
`failed` is non-empty and cleared automatically on the next successful reload; (3) the verdict
label is **always** configured — a real result, or `"—"` — never left showing a stale value; (4)
the pill row and drift tab each get a `clear()` that resets them to their "just constructed" look
instead of being silently skipped when there's no status. Five new tests pin: a crashed loader is
named, not silently rendered as absence; a healthy page shows no banner; the banner clears on the
next successful load; model B never shows model A's verdict text; model B never shows model A's
pills.

*Citing line:* `test_model_page_failures.py:4` points here for the design points above and for
the seeding/synchronous-open test pattern borrowed from `test_spec3c_model.py` /
`test_findings_tab.py`.

### `task-E1-brief.md` — cited by `config.py:127`, `gui/v6/sections/backlog.py:4` and `:98`, `tests/test_backlog_section.py:3`

James's request (2026-09-20): Settings had two separate inputs — an Active Models (MPS) list and
a Pricing list — that he wanted replaced by one backlog upload producing both. Three design
points were settled with him the same day: **(1)** price is the unit price on the model's
**latest order** by Order Date — not the most common price, not an average. **(2)** Add-on line
items (item IDs like a model number plus a suffix for inspection/test/lot paperwork) are **never**
counted as, or folded into, the base model — only an Item ID that matches a known model *exactly*
counts. **(3)** Each upload **replaces** the backlog's active-model list; manual pins stay, for
exceptions.

*(The brief's own worked examples pair specific model numbers with specific backlog prices from
the owner's real export — real values, omitted here entirely, per this document's own rule.
One such pair was in fact the leak the controller found and scrubbed from local history before
any push; see the process-findings-engine ledger's Ruling 20.)*

**Data model:** `mps_models` stays the one list every other screen reads (dashboard cost
priorities, triage, trends, compare, analyze, database cleanup), maintained as
`sorted(backlog_models | pinned_models)` by one helper, `rebuild_active_list()`, called from both
the upload path and the manual-pin save path so the two can't drift apart. An upload **replaces**
`backlog_models`/`backlog_open_qty` (this week's list) but **merges** `model_prices` — a model
that drops off the backlog keeps its last known price, since cost analytics on a model with no
open orders this week still needs one. A one-time migration: if `mps_models` is non-empty while
both `pinned_models` and `backlog_models` are empty, the existing list predates this feature and
was hand-pinned — it's copied into `pinned_models` (not saved until the next user action).
Customer names and PO numbers are read from the export but never stored in the result — only
model, open quantity, price, and earliest need date leave the parser.

*Citing lines:* `config.py:127` cites the brief for the full design behind the four new
`ActiveModelsConfig` fields. `backlog.py:4` and `:98` cite it for the module's overall design and
for `_migrate_hand_pinned_list`'s one-time-upgrade rationale specifically.
`test_backlog_section.py:3` cites it for "the 7 points" — the seven numbered test requirements
the brief's Step 2 lists for that file.

### `task-7-brief.md` (trim-capture plan) — cited by `tests/test_trim_capture_db.py:69` and `:96`

Task 7 is what wires the already-captured per-pass sweeps and per-file setup block through to the
database: a two-line change to `core/processor.py` (`TrackData` gets `trim_passes=...`,
`AnalysisResult` gets `trim_setup=...`, "no other change to this file") plus two new methods on
`DatabaseManager` — `_write_trim_passes` and `_write_trim_setup` — called from both the
fresh-insert path and the update-existing path, so a reprocess **replaces** (delete, then
re-insert) rather than accumulates duplicate rows.

The two items the citing tests label "`#1`" and "`#2`" aren't a numbered list inside the brief
file itself — they're two concerns carried forward into Task 7's dispatch from the *previous*
task's review, recorded in the ledger rather than as brief text:

- **#1** — the existing schema test proved only that the `(track_result_id, pass_index)` unique
  index *exists*, never that it actually *rejects* a duplicate insert. Task 7 is the first task
  that writes pass rows, so it's the first place a real behavioural test belongs.
- **#2** — `pass_sheets` (the sheet-name parser) has no dedup guard: two differently-named sheets
  that normalize to the same leading number could, in principle, produce two passes with the same
  `pass_index` and collide on that same unique index at write time. Never observed in a real
  fixture, but the decision (this ledger's Ruling 11) was that the write path must **tolerate**
  the collision — keep the first occurrence, drop and log the rest — rather than let an uncaught
  `IntegrityError` roll back the *entire* analysis save (every track's verdict for that file) to
  protect one redundant pass row.

Both concerns are now full behavioural tests (`test_trim_pass_unique_index_rejects_duplicate_insert`,
`test_write_trim_passes_tolerates_duplicate_pass_index`) and the tolerate-logic itself is
documented in `database/manager.py: DatabaseManager._write_trim_passes`.

*Citing lines:* `test_trim_capture_db.py:69` and `:96` are exactly those two tests.

## The limit-tables patches

`docs/superpowers/plans/2026-09-20-limit-tables.md` says its per-task patches under
`.superpowers/sdd/2026-09-20-limit-tables/` "are the code." That plan **was** executed on
2026-09-20, the same day it was written — `git log --oneline --grep "limit"` (case-insensitive)
returns 74 matches, including every task commit below, confirming the work landed as commits on
`V6` and isn't sitting unexecuted as patch files. So the plan's claim is now a historical
description of *how* the change was authored (validate a patch in an isolated worktree, then have
an implementer apply it for review), not a live pointer — the actual code is the commits, not the
`.patch` files beside them. In order: `b6a502d` (the plan doc itself) · `7d7d24c` (Task 1, limit
identity on every track) · `87a1566` (Task 2, the multi-table analyzer) · `d913a0a` (Task 3,
ink_target's window fix) · `679941c` (Task 4, recipe_change's disclosure) · `7183e43` (Task 5,
engine + Model page wiring) · `dea3efa` (the fix for the whole-branch review's C1/C2/C3/I4/I5/I6/I7
findings) · `8ae662c` (Task 6, folded into the findings-engine plan's own Task 11 sweep gate).

One thing worth knowing before you go looking: the ledger itself cites an **earlier** set of
hashes for the first six of those — `ccacbe0`, `0e74462`, `cce7992`, `f36074d`, `70816bb`,
`ce7a859`. Those are real, but they're pre-scrub: the same evening, a history rewrite removed a
leaked real backlog price from local history (the process-findings-engine ledger's Ruling 20),
which changed every commit's hash from that point forward while keeping the same author, the
same timestamps, and the same content. The old hashes are no longer reachable from any branch —
`git show ccacbe0` will fail with "unknown revision"; `git show 7d7d24c` is the same change under
its current name.

## Citations to repoint

**Repointed 2026-09-24:** every line below now names this document instead (and
`tests/test_backfill_increment_volts.py:2`, added later, names the tracked parse-fixes plan its
brief was cut from). No tracked file cites a `.superpowers/` path any more — `git grep
'.superpowers/sdd' -- ':!docs/*' ':!TRACKER.md'` returns nothing — so deleting the workspaces
breaks no citation. The list is kept as the record of what each line said before:

- `src/laser_trim_analyzer/utils/threads.py:210` — `` `.superpowers/sdd/prebuild-fixes/H5-report.md`): dropping a Font on a ``
- `tests/conftest.py:128` — `` `.superpowers/sdd/prebuild-fixes/H5-report.md`. ``
- `tests/test_tk_thread_safety.py:4` — `` the same bug; the 2x2 experiment in `.superpowers/sdd/prebuild-fixes/H5-report.md` ``
- `tests/test_model_page_failures.py:4` — `See .superpowers/sdd/prebuild-fixes/M1-brief.md. Seeding and the synchronous`
- `src/laser_trim_analyzer/config.py:127` — `# pin-save path. See task-E1-brief.md for the full design.`
- `src/laser_trim_analyzer/gui/v6/sections/backlog.py:4` — `and task-E1-brief.md for the full design.`
- `src/laser_trim_analyzer/gui/v6/sections/backlog.py:98` — `` task-E1-brief.md); it is saved with whatever the user does next.""" ``
- `tests/test_backlog_section.py:3` — `Covers the 7 points task-E1-brief.md names for this file, in the same order.`
- `tests/test_trim_capture_db.py:69` — `` """Findings review (task-7-brief.md #1): the schema test only ever proved ``
- `tests/test_trim_capture_db.py:96` — `` """Findings review (task-7-brief.md #2): pass_sheets has no dedup guard, ``

Not in this list, but related: `TRACKER.md`'s H3 entry (around line 629) names all five
workspaces and the four files above as the reason the workspaces can't yet be deleted. It isn't a
"see X for details" citation the way the ten lines above are, so it isn't repointed here — but
once this document's citations are wired in, H3 is the natural place to record that the blocker
is resolved.
