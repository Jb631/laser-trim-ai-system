# Ingest speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a full ingest several times faster by batching the saves into one transaction per 20
files (A4), parsing and analysing in worker processes while one consumer saves (A3), and letting a
dropped worker come back (A5) — without changing a single stored number.

**Architecture:** The worker body becomes pure (`analyse_path` → an `Outcome` value carrying the
parse's own stat and hash); every database write moves onto ONE consumer that applies outcomes
through `write_batch` (BEGIN IMMEDIATE, a SAVEPOINT per file, the processed marker inside the same
savepoint). Worker processes (spawn everywhere, module-level code in `core/ingest_worker.py`) never
open a database: model specs, thresholds and predictors reach them as a `SpecSnapshot`. The thread
pool stays as the fallback and for small runs.

**Tech Stack:** Python 3.14, SQLAlchemy 2.0 on SQLite (WAL), `concurrent.futures.ProcessPoolExecutor`
(spawn), `logging.handlers.QueueListener`, pytest via `scripts/run_test_gate.py`.

**Spec:** `docs/superpowers/specs/2026-09-25-ingest-speed-design.md` — measured facts F1–F13 (§1), the
design (§3 A4, §4 A3, §5 A5), 23 proposed rulings (§7), the work probe (§8), the task list (§9).
Every task below is one §9 item; its requirements are the spec section it names. **Controller
rulings on the spec (2026-09-25):** rulings 2–23 accepted as written. Ruling 1 AMENDED: the probe
(Task 1) is built and pushed FIRST, on its own, so James can run it at work; Tasks 2–12 proceed
without waiting for its output, because the design does not hinge on it — the probe sets K (ruling 5)
and confirms the pragmas (ruling 12) and the expected gain, and both are retuned from its output when
it arrives — cost if wrong: a K retune.

## Global Constraints

- **No stored number changes.** `tests/test_parse_all_models.py` (645 real files, every frozen value)
  must stay green with ZERO refreshed entries in every task; run it first after any change to the
  processor, the analyzer path or a save path, and read its pass/skip counts.
- **Production is read-only.** Never open `data/analysis.db` read-write; the probe copies it with the
  SQLite backup API from a `mode=ro` connection and refuses a copy path inside `data/`. Any script or
  test that builds a Processor/DatabaseManager or can reach `get_database()` injects a manager into
  BOTH `laser_trim_analyzer.database.manager._db_manager` and `laser_trim_analyzer.database._db_manager`
  first; the guard (`DefaultDatabaseRefused`) refuses the default outside the app.
- **Workers never open a database** (ruling 15) and **never call Tk** (UI thread discipline: post via
  `gui/v6/ui_dispatch`).
- **`_set_sqlite_pragma` is never deleted** — it is how foreign keys are enforced; new pragmas go
  beside `foreign_keys` in it (ruling 12).
- **A record that failed processing is not a measurement**; ERROR rows keep their reason; a failure is
  named, never counted as a result (a failed save counts as an error, ruling 22).
- **Windows is the target**: spawn start method on every platform, module-level worker code, picklable
  values, PowerShell commands (`.\.venv\Scripts\python …`) in BRING_TO_WORK. No packaged EXE.
- `strftime("%-d")` is forbidden; datetimes bound into `text()` as `f"{dt:%Y-%m-%d %H:%M:%S.%f}"`.
- The whole suite is the gate: `scripts/run_test_gate.py` (read its summary line) and the app sweep on a
  writable COPY after any change to save paths or queries (`scripts/app_qa_sweep.py <copy>`; the known
  FAIL is D3's same-day links). New behaviour gets a sweep entry in the same commit.
- Commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; stage by explicit path;
  never `git add -A`, `git stash`, `checkout -- <file>`, `reset`. Example data is INVENTED;
  `scripts/check_no_customer_values.py` before any push. Mutation checks: a fresh directory per
  mutation and `python -B`.

---

### Task 1: The work probe (spec §8, ruling 3)
**Files:** Create `scripts/ingest_save_probe.py`, `tests/test_ingest_save_probe.py`; modify
`scripts/pool_probe.py`, `scripts/ingest_speed_probe.py`, `scripts/parse_locality_probe.py` (ruling 3:
copy `model_specs` from a named database into their throwaway ones).
- [ ] Tests first: the source database is opened only `mode=ro` (and through the backup API); a copy
  path inside `data/` is refused; a copy drive with under twice the database free is refused; every
  manager is on the copy and injected into both globals before any `Processor`; a 3-file `--no-loop`
  run on a tmp database built from fixtures prints every SAVE line; the copy and temp files are
  deleted at the end and on Ctrl-C. Then the probe, exactly as §8 prints it (it carries its own copy
  of the batch writer until Task 6 lands).
- [ ] BRING_TO_WORK step (controller writes it): the one PowerShell command and "send the whole output back".

### Task 2: An honest batch line (spec 3.10, ruling 2)
- [ ] The batch line and `Ingest so far` print the save's CPU beside its wall time. Test: a save stub
  that sleeps (wall, no CPU) and one that spins (CPU) are told apart.

### Task 3: `file_hash` indexes (spec 3.7, ruling 11)
- [ ] Index `file_hash` on `final_test_results` and `smoothness_results` via an idempotent migration.
  Tests: `EXPLAIN QUERY PLAN` says SEARCH; running the migration twice is a no-op. QA-sweep entry.

### Task 4: Pragmas (spec 3.8, ruling 12)
- [ ] `synchronous=NORMAL`, `cache_size=-65536` in `_set_sqlite_pragma` beside `foreign_keys`. Test:
  every new connection reports foreign_keys 1, synchronous 1, cache_size −65536. QA-sweep entry on
  the app's own connection.

### Task 5: Session-taking trim body (spec §3, rulings 7, 10)
- [ ] `_save_analysis_in(session, a, stat, hash)`; `save_analysis` wraps it; `_record_processed_file`
  and `_write_failure_marker` take the carried stat and hash. Tests: rows identical to today's for
  every fixture; no `os.stat` or file read when values are given.

### Task 6: `write_batch` (spec 3.4–3.6, rulings 4–8, 22)
- [ ] BEGIN IMMEDIATE, a savepoint per item, outcomes, the nested-session guard (`session()` raises
  mid-batch), `save_batch` rebuilt on it (writes trim_passes and trim_setup). Tests: F6's two cases
  (subprocess, `os._exit` mid-batch → those files are new again); F5 now raises; save_batch rows ==
  save_analysis rows; flush at 20 files or 2 s.

### Task 7: Session-taking final-test, smoothness and skip-marker bodies (spec 3.3, ruling 9)
- [ ] Public methods wrap them. Tests: identical rows to the old paths, each also inside `write_batch`.

### Task 8: `SpecSnapshot` (spec §4, ruling 16)
- [ ] One resolver for the snapshot and `get_model_spec`/`resolve_spec_for_ft`. Test: snapshot and
  database answer identically for every fixture model plus section-letter and alias cases.

### Task 9: Values out of the worker (spec §4, ruling 9)
- [ ] `analyse_path` → `Outcome` carrying stat and hash; the thread pool runs it; `process_batch` hands
  outcomes to a writer (V6 passes one; with none, each write is applied at once — V5's loop keeps
  working). Tests: no database touched for every fixture kind (`get_database` patched to raise); V5's
  loop still saves final tests; the 645-file gate.

### Task 10: `run_folder` on the writer (spec §3, rulings 5, 20, 22)
- [ ] Buckets from committed outcomes, flush on Stop and at the end, two failed commits stop the folder
  with the error named. Update the tests that fake `db.save_analysis`. Tests: the cancel suite
  unchanged; a failed save counts as an error; buckets sum to processed.

### Task 11: A3 — `core/ingest_worker.py` and the process pool (spec §4, rulings 13–19)
- [ ] Context, the database trap, the QueueListener logging, chunked dispatch, the thread fallback,
  close. Tests: spec §4.9. QA-sweep entry: process mode on the sweep's files gives the thread mode's rows.

### Task 12: A5 and close-out (spec §5, ruling 21)
- [ ] `next_cap` and its wiring; the probe switched to the real writer; BRING_TO_WORK's before-and-after
  steps; TRACKER A0's Amdahl paragraph corrected (the controller does the docs).
