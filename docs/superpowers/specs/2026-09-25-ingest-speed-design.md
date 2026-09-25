# Ingest speed — batch the saves (A4), worker processes (A3), workers that come back (A5)

Design spec, 2026-09-25, branch `ingest-speed` (written at 39461a2; `main` is d77afd7 —
`core/processor.py` and `core/ingest_run.py` are identical there, `manager.py` line numbers
below are main's). Design only: no production code.

**How it was measured.** This Mac (M4 Pro, 14 cores, Python 3.14.2, SQLite 3.50.4); APFS clones
of a copy of the rebuilt work database (6.23 GB: 107,603 analyses, 151,793 final tests, 322 model
specs), each configuration from an identical clone warmed into the OS cache; production opened
only `mode=ro`; every manager injected into both `_db_manager` globals first. Sample: the first
240 DLTS, 120 LTS and 240 Test Station files of `Work Files/home_slice` in the ingest's discovery
order, their rows deleted from the clone so every save is a new file's INSERT path. Other agents'
jobs shared the Mac (load 3–5): repeats are ranges, never best-of. Scripts are in the session
scratchpad `speed/` (`PYTHONPATH=<worktree>/src .venv/bin/python <script>`); §8's probe carries
the ones that matter to the work laptop.

## 0. The short version

1. **The save is not the 432 ms.** A realistic save into the 6.2 GB copy costs 4.2–5.0 ms here;
   SQL is 0.3 ms of it, the ORM's unit of work about half. Taking the save out of the real loop
   entirely makes the loop 3.5 ms/file faster.
2. **The batch line's `of which save` is mostly waiting.** In the 4-thread loop the save's wall
   time is 33 ms against 5.3 ms of CPU — the consumer queues for the GIL while parsers run — and
   the app's own line printed `44 ms/file` for it.
3. **A0's 339 ms "parse" was the cheap analysis.** Every probe parses on a throwaway database
   whose `model_specs` table is empty; the spec-driven analysis the ingest really runs costs
   ~48% more (DLTS 67 → 99 ms/file, same files). A third of the "432 ms outside parsing" is
   parsing.
4. **With processes the parse and the save overlap; they do not add.** A prototype — 8 spawned
   workers, one consumer saving 20 files per transaction into the clone — runs DLTS at 13.3
   ms/file against today's loop at 102–116, LTS at 6.1 against ~39, final tests at 4.4 against ~30.
5. **Every final-test save scans the whole `final_test_results` table** (no index on
   `file_hash`): 20 of its 22 ms. An index makes it 3.5 ms.
6. Batching is still right — the durable flush a laptop pays per commit, 2–2.4x fewer WAL bytes,
   and A3 needs every worker-side write on one consumer anyway — but what it buys at work depends
   on the laptop's flush, which §8's probe measures.
7. **About 200 ms/file of A0 does not reproduce headless here** (§2). The probe's LOOP block says
   whether it lives in the code (A3 removes it) or in the app and machine (it doesn't).

## 1. Measured facts

**F1 — the save, split** (`measure_save.py 3`, `flush_split.py`; 356 real trim results, 1.58
trim passes each, median of 3 repeats, ms per file, plain macOS fsync):

| batch | synchronous | cache | total | Python/ORM | SQL | commit | p99 |
|---|---|---|---|---|---|---|---|
| 1 (today) | FULL | 2 MB | 4.81 | 3.44 | 0.19 | 1.16 | 20.7 |
| 1 | NORMAL | 64 MB | 4.47 | 3.22 | 0.20 | 1.05 | 19.7 |
| 10 | FULL | 2 MB | 4.38 | 3.15 | 0.36 | 0.87 | 7.0 |
| 50 | NORMAL | 64 MB | 4.17 | 3.09 | 0.32 | 0.77 | 4.4 |

One level finer: the flush is 2.2 ms (2.1 ORM bookkeeping + 0.1–0.2 executing its INSERTs),
other SQL 0.06, commit 0.8–1.0 (fsync + amortised autocheckpoints — the p99), other Python
0.8–1.7. 10.6 statements per file. Batching and pragmas change almost nothing on this disk.

**F2 — with a durable flush.** `PRAGMA fullfsync=1` makes SQLite use F_FULLFSYNC, which empties
the drive's write cache as Windows' FlushFileBuffers does (plain macOS fsync does not — why the
Mac's commit is nearly free). Per-file commits then cost **9.5, 18 and 33 ms/file** across the
three repeats (FULL, 2 MB; up to 56 at 64 MB) — the flush waits on whatever else the disk holds.
NORMAL per file: 5.3–12. Batches of 10: 5.2–20 (FULL), 4.7–15 (NORMAL). Batches of 50: 4.9–7.7
(FULL), **4.5–8.2 (NORMAL, 64 MB)**. Batching turns a heavy-tailed per-file flush into ~1–2
ms/file in most repeats; NORMAL removes the per-commit WAL flush. **F3 — WAL per file**
(`wal_growth.py`, checkpoints off): 63 pages (252 KB) with per-file commits, 33 at batches of 10,
26 at 50 — hot index pages are written once per transaction.

**F4 — pysqlite's savepoint trap** (`savepoint_semantics.py`). The manager's engine uses
pysqlite's legacy transaction control, which emits no BEGIN before a SAVEPOINT; SQLite then
starts the transaction at the SAVEPOINT, and RELEASE of that outermost savepoint is a COMMIT.
Measured: without an explicit BEGIN, after the first file's RELEASE `in_transaction` is False and
a second connection sees that row mid-batch — a "savepoint per file" batch commits per file.
With `BEGIN IMMEDIATE` first: `in_transaction` stays True, the other connection sees 0 rows until
the end, and a duplicate-key file rolls back alone.

**F5 — a nested session commits the batch** (`nested_session.py`). Inside that batch, a helper
that opens its own `db.session()` (as `mark_file_skipped`, `save_final_test`,
`apply_final_test_regrade` all do) commits everything before it on the shared connection; a batch
that then failed left all three of its rows behind.

**F6 — a crash mid-batch, and one bad file** (`crash_demo.py`, real results, a child process).
Hard exit (`os._exit`, no COMMIT) after the 20th of 30 savepoints: **0 analysis rows, 0
processed_files rows, and the ingest's own scan (`_classify_scan` after `_load_processed_hashes`)
calls all 30 "new"**. A batch of 30 where file 5's save raises: 29 rows + 29 markers committed;
file 5 has neither and the scan calls it "new".

**F7 — the real loop** (`loop_decomp.py`; DLTS, 240 files, specs present, the 281,827-path index
loaded; ms/file excluding the one-off 1.6 s index load):

| run | ms/file |
|---|---|
| parse+analyse, one at a time | 99–101 |
| parse+analyse, 4 threads (`ThreadPoolExecutor.map`) | 95–100 (1.0x: the GIL) |
| `run_folder` as shipped (4 threads, 20-file batches, `gc.collect` per batch, save on the consumer), two runs interleaved with the next row | **101.8, 103.1** |
| the same with `save_analysis` a no-op | **98.3, 99.4** |
| save inside that loop: wall / thread CPU | **33.2–33.9 / 5.0–5.3 per save** |
| the same loop at the 5 ms default switch interval | 117; save wall 98 per save |

Five earlier runs under heavier machine load gave 104–117. The app's own batch line from one
of them: `of which save 10.5s (36%, 44 ms/file)`. The loop costs ~4 ms/file beyond parsing
(barrier, per-batch GC, bookkeeping) plus ~3.5 for the save.

**F8 — the specs** (`spec_effect.py`, `spec_answer.py`). Same 240 DLTS files, tiny database:
66.6–68.4 ms/file without specs, 98.2–100.4 with the clone's 322 rows copied in. Without specs,
156 of 356 trim results store different graded numbers (offset, error, fail points); 0 verdicts
flipped in this sample. A fresh `DatabaseManager` database has 0 spec rows, and neither
`pool_probe.py`, `ingest_speed_probe.py` nor `parse_locality_probe.py` copies any. The work
database's specs were written 2026-09-20 14:02 UTC, before its first analysis row (14:09): A0
and the rebuild ran **with** specs; pool_probe's 339.2 at work ran without.

**F9 — the final-test save.** `EXPLAIN QUERY PLAN SELECT … FROM final_test_results WHERE
file_hash = ?` → `SCAN final_test_results` (read-only on production: 15 ms warm, 1,108 ms cold);
the same for `smoothness_results`. `measure_ft_save.py 2` (239 real final tests, plain fsync):
22–24 ms/file, SQL 20–21 of it in 4 statements; with the index (`--index`, built in 0.14 s):
3.4–3.7 batched, 5.0 per file (FULL), SQL 2.4–2.7. The shipped loop (`loop_decomp.py loop
Test_Station [--ftindex]`): 29.9 → 18.4 ms/file. `update_processed_file_stats` runs the same
scan once per healed entry.

**F10 — processes doing the real analysis** (`procs_specs.py`; spawn; each worker's throwaway
database holds a copy of `model_specs` — the stand-in for §4's snapshot):

| files | one at a time | 1 worker | 4 workers | 8 workers | pool ready | RSS per worker |
|---|---|---|---|---|---|---|
| DLTS 240 | 99.4 | 100.6 | 26.3 (3.8x) | 16.0 (6.2x) | 1.2–1.5 s | 264–291 MB |
| LTS 120 | 33.1 | 31.5 | 8.5 (3.9x) | 5.1 (6.5x) | 1.1–1.4 s | 221–258 MB |

0 of 238 DLTS and 0 of 118 LTS results differ field by field from the in-process result. Their
pickled **bytes** differ every time: pydantic's fields-set is a set of strings, ordered by each
process's hash seed — so equality must be tested on `model_dump()`, never on bytes. **F11 —
pickling** (`prep.py`): a trim `AnalysisResult` round-trips equal; 33 KB median, 59 KB p90, 125
KB max (DLTS mean 45 KB); 0.06 ms to dump, 0.07 ms to load. A final-test payload: 4.5 KB.

**F12 — A3+A4, end to end** (`proto_pipeline.py`, `proto_ft_pipeline.py`): 8 spawned workers
with specs, the parent's one consumer saving 20 per transaction (BEGIN IMMEDIATE + a savepoint
per file, NORMAL, 64 MB) into a clone. DLTS **13.3 ms/file** (consumer save 5.8 wall / 5.5 CPU);
DLTS with 4 workers 25.3; LTS **6.1** (consumer 5.3 — the consumer is now the ceiling); final
tests 24.1 without the index (consumer-bound on the scan), **4.4** with it. Every row landed
(238/238, 118/118, 239/239 plus the one serial-less file refused).

**F13 — the logs** (`data/laser_trim.log*`, read). The rebuild's 168,501-file batch was the
**Final Test folder** (`rematch skipped (no new trims)`; "Final Test … recorded as skipped" lines
throughout), whose saves run inside the worker threads. It ran 465 ms/file = 2.15 files/s; the
VPN's measured ceiling is ~2.4 files/s (memory note share-io-cost-model). The same laptop took 250
new final tests in 42.9 s (172 ms/file) at 07:53 on 09-23. Where B2 ran is not recorded.

**F14 — code facts.** StaticPool (main:341): the app has ONE SQLite connection, shared by every
thread and serialised by `_write_lock`, an RLock held by `session()` (main:1758); the two direct
`_SessionFactory()` uses (main:1775, 8060) hold it too. `_record_processed_file` (main:2081) does
a real `stat_once` and a hash lookup inside the open transaction, `_write_failure_marker` a
second stat. `save_batch` (main:1870) and `_get_linearity_type` (processor.py:1299) have no
callers. Spawn never re-runs a `*.__main__` module in the child (`multiprocessing/spawn.py:256`),
so under `python -m src` a worker has no logging and never calls `allow_default_database()`.

## 2. Where the 432 ms goes

A0 at work: loop 771.7 ms/file = 339.2 (pool_probe) + 432.5 "outside parsing". Against F7–F8:

- **~160 ms of it is parsing.** pool_probe ran the spec-less analysis. At the Mac's DLTS ratio
  (1.48) the real parse+analyse at work is ~500 ms/file, leaving ~270 outside it. The probe's
  LOOP block measures the laptop's own ratio.
- **The save's own cost is ~30 ms.** The laptop measured 28.9 ms/save into an empty database
  (ingest_speed_probe, 2026-09-20) — ~5x the Mac, the same factor as its parsing. F7 shows a
  save costs the thread loop about its CPU, whatever its wall time says.
- **The loop's own machinery is ~20 ms** if the Mac's ~4 ms scales the same way.
- **~200 ms/file is unexplained** and does not reproduce headless here. Unmeasured candidates:
  the V6 window (A0 ran in the app; its Tk thread competes for the same GIL), the laptop's
  durable flush while the save holds `_write_lock` (worker threads block on that lock for the
  per-file spec lookup), Windows scheduling six GIL-contending threads at the 0.5 ms switch
  interval, a different file mix (A0's first 674 files in walk order; pool_probe's first 200
  sorted), and the laptop's Python version (GC differs before 3.14).

So the brief's premise — the save is the serial half — does not hold on the evidence: the save is
~4% of the loop here and ~4% of A0 by the one laptop figure there is. Its conclusion, A4 first,
survives for other reasons (§6).

## 3. A4 — batch the saves

**3.1 Shape.** A `BatchWriter` owned by the ingest run, fed on the consumer thread. Items are a
trim result, a final-test payload, a smoothness payload or a skip marker. It flushes at **20
items or 2 s after the first buffered item**, whichever comes first, and always at generator end,
on Stop and at folder end; the consumer waits on the pool with a 1 s timeout, so a stalled share
still commits what it has. A flush calls `db.write_batch(items)` and gets back one outcome per
item: `saved(id)`, `duplicate` or `failed(reason)`. Two consecutive failed commits end the folder
with the error named, instead of parsing 60,000 files into nothing.

**3.2 One transaction, a savepoint per file.** `write_batch` opens one `session()`, runs
`BEGIN IMMEDIATE` as its first statement (F4: without it, RELEASE commits per file), then per
item `sp = session.begin_nested()` → the item's body → `sp.commit()`, or `sp.rollback()` and a
`failed` outcome; the session's exit commits once. IMMEDIATE takes the write lock up front, so
another process holding it makes BEGIN wait `busy_timeout` (30 s) and fail loudly, instead of the
batch failing half-way on a read-to-write upgrade.

**3.3 The lock and no nested sessions.** `session()` holds `_write_lock` for the whole batch —
load-bearing, since one connection serves every thread (F14) and any other session's commit
mid-batch would commit the batch. UI loaders wait at most one batch (~0.1 s here, ~0.5 s at an
estimated laptop cost). F5 forbids the other half: every body the writer runs takes the session
— `_save_analysis_in`, `_save_final_test_in` (its duplicate, refresh and IntegrityError branches,
`_find_matching_trim`, `_mark_ft_duplicate_path`, `apply_final_test_regrade` and
`_refresh_final_test_identity` all handed the session), `_save_smoothness_in`,
`_mark_file_skipped_in` — and the public methods become wrappers that open a session and call the
body. A thread-local "batch open" flag makes `session()` raise if entered on that thread
mid-batch, so a future helper fails a test instead of committing half a batch.

**3.4 Marked processed iff committed — from the code.** `save_analysis` (main:1790) opens one
`session()` and calls `_record_processed_file(session, …)` inside it; since a6f2085 the
reprocess branch `_update_existing_analysis` (main:3907) does the same. `session()` commits once
at exit and rolls back on any exception. `_load_processed_hashes` (processor.py:1739) reads
processed_files (success=True), final_test_results and smoothness_results — committed rows only;
final-test and smoothness rows are their own markers and a skip marker is a processed_files row.
The unit of "processed" is therefore the unit of commit. In a batch, each file's rows and marker
share a savepoint, so a crash before the COMMIT leaves every file of the batch new and one bad
file leaves only itself new — F6 measured both.

**3.5 `save_batch`** becomes a wrapper over `write_batch` with trim items, so it writes
`trim_passes` and `trim_setup` exactly as `save_analysis` does (the parked C2 finding). It has no
callers, so nothing else moves; its commit-per-file loop is deleted.

**3.6 No file I/O inside a transaction.** Every result carries the parse's own (size, mtime,
sha256) — the parser already has them (`stat_once` in its parse scope; `hash_bytes_for` on the
bytes it read). `_record_processed_file` and `_write_failure_marker` take them and do no I/O:
that removes a share round trip per file (113 ms over the VPN) from inside the lock, and records
the stat of the bytes actually parsed. Today a file rewritten between parse and save is recorded
with the new stat on the old content, and the next scan's fast path skips the new content.

**3.7 The final-test index.** A migration adds `idx_ft_file_hash ON final_test_results(file_hash)`
and `idx_smoothness_file_hash ON smoothness_results(file_hash)` (`IF NOT EXISTS`; 0.14 s on the
copy). F9: the FT save 22 → 3.5 ms, the shipped FT loop 30 → 18 ms/file, and every other
by-hash lookup (`is_file_processed`, both lookups in `save_final_test`, and the stat heal's one
scan per entry) stops scanning.

**3.8 Pragmas.** `_set_sqlite_pragma` (main:364) keeps `foreign_keys=ON` and adds
`synchronous=NORMAL` and `cache_size=-65536` (64 MiB). *Where:* every connection. The app has
exactly one (F14), so "the ingest's only" would mean a second connection and a new locking
story; the listener is the one place every manager passes. *What a power cut can lose:* with WAL
+ NORMAL the WAL is synced before every checkpoint (automatic every 1,000 pages ≈ 30–40 files at
F3's rates), not at each commit, so an OS crash or power cut can roll back the batches committed
since the last checkpoint. It never corrupts the file, and an app crash or kill loses nothing
committed. A rolled-back file has neither rows nor marker, so the next run re-processes it (F6)
— re-runnable ingest is what makes the trade free. The UI's own writes (settings, backlog upload,
spec edits) share the same seconds-wide window, and all can be redone. *Cache:* no effect here
(SQL is 0.3 ms either way), a little at batch 50 (fewer spills); bounded and cheap. The probe
decides both.

**3.9 What Progress and Stop see.** The bar still counts parsed files (credited on "completed",
unchanged). The buckets and `Ingest so far` count **committed** outcomes, so a failed save is an
error, never a pass; they lag the bar by at most 20 files or 2 s. Stop keeps its contract — the
processor stops at the next 20-file chunk, everything parsed goes to the writer and is committed
before `_post_batch`. Closing the window mid-run loses at most the unflushed ≤ 20, which are new
next run.

**3.10 An honest batch line.** `log_phases` and the 200-file `Ingest so far` line print the
save's CPU (`time.thread_time()`) beside its wall time. Windows counts it in ~15.6 ms ticks, so
it is trustworthy only summed over hundreds of saves — which is all the line reports.

## 4. A3 — parse and analyse in worker processes

**4.1 Shape.** A new Tk-free, database-free module, `core/ingest_worker.py`. A `WorkerContext`
(config, the spec snapshot, ML thresholds and predictors, the ML models directory as an absolute
path) goes to each worker once, as `initargs`. `analyse_path(path, disk_stat) -> Outcome` is
today's `process_file` body minus every write. The in-process path runs the same function on
threads, and the consumer applies every Outcome through §3's writer — one analysis body, two
pools. `Processor.process_file` stays for its other callers (V5, `track_repair`, scripts) as
analyse + the old side effects.

**4.2 Every database touch in the worker body today, and what replaces it** (processor.py lines).
TRACKER counts six (the skip markers); there are eleven — the worker also *saves* every final
test and smoothness file, looks up a spec per file, and loads ML state when it is constructed:

| where | today | becomes |
|---|---|---|
| 203–234, `__init__` → `_load_ml_thresholds` | `get_database()` + `get_shared_ml_manager` | loaded once per folder in the parent; thresholds and predictors travel in the context |
| 260, 278 | `_mark_file_skipped` (non-trim; parameter workbook) | `Outcome(skip, marker)` with the parse's hash and stat |
| 488, 649, 655, 1295 | `_mark_file_skipped` (permanent or non-transient failure: trim, FT ×2, smoothness) | `Outcome(error, marker)`; the transient/permanent classifiers stay in the worker |
| 563, 617 | `db.save_final_test` (header-only; full) | `Outcome(final_test, payload)` — the same kwargs |
| 1234 | `db.save_smoothness_result` | `Outcome(smoothness, payload)` |
| 1339–1344, `_get_spec_for_analysis` — per file, twice for 8340 | `get_model_spec` / `resolve_spec_for_ft` | a `SpecSnapshot` built from `get_all_model_specs()` (main:8173) at folder start; `get_model_spec` and `resolve_spec_for_ft` call the same resolver, so there is one rule |
| 1716–1720, `_mark_file_skipped` body | hash + stat + `db.mark_file_skipped` | hash and stat in the worker (it has the bytes), the write on the consumer |
| 1299–1308, `_get_linearity_type` | `get_model_spec` | dead (no callers): delete |

Already on the consumer and unchanged: `_load_processed_hashes`, the `_is_processed` fallback,
the stat heal, the 8-thread verify pool. `ml/predictor.py:777` (`_add_spec_features`) reaches the
database but is off the predict path the processor calls. Two per-process caches stop being
shared, the hash cache and `_BYTES_CACHE`: the parent's `calculate_file_hash` would miss and
re-read the whole file (~600 ms over the share), which is why the Outcome must carry the hash.

**4.3 The guard in a worker: a safety net for the database, a corruption path for the results.**
A spawned worker never calls `allow_default_database()` (F14), so its `get_database()` raises
`DefaultDatabaseRefused` (main 0250d95): the database is safe. The results are not: every worker
touch sits inside `except Exception`. `_get_spec_for_analysis` logs at DEBUG and returns "no
spec", so the analysis quietly runs spec-less (F8: 156 of 356 results store different numbers).
`_mark_file_skipped` logs at DEBUG and the file is re-read every run. The final-test save becomes
an ERROR result. The one ERROR line per process reaches the console, not the log (no handler).
Nothing would crash; the rebuild would come back subtly wrong. So (1) no touch remains (4.2), and
(2) the worker initializer makes `get_database`, `DatabaseManager.__init__` and `sqlite3.connect`
raise `WorkerDatabaseAccess` and record the call site; `analyse_path` returns `Outcome(internal)`
for any file during which a reach was recorded, whatever the analysis produced — never saved,
logged at ERROR by the parent, counted in the batch line, retried next run. The conftest guard
does not reach spawned children; this trap is their guard.

**4.4 Windows `spawn`.** `mp.get_context("spawn")` on every OS, so the Mac's tests run Windows'
start method. Worker function and initializer are module-level in an importable module (no
lambdas, closures or bound methods); paths go as `str` with the walk's (size, mtime); arguments
and results pickle (F10, F11) and tests compare `model_dump()`, never bytes. Under `python -m src`
the child skips `src/__main__` (F14) — no GUI, no logging, no default database — so the
initializer sets up what the worker needs. The composite-risk models load from a CWD-relative
`Path("data/ml_models")` (processor.py:194): the context carries the absolute path. No
`freeze_support` (no EXE, per CLAUDE.md).

**4.5 Start-up, memory, how many.** Here the pool is ready in 1.1–1.5 s at 1–8 workers (they
start in parallel); at work it is unknown and longer — the endpoint scanner reads every imported
module (pool_probe's first run lost seconds to it); the probe prints it. One pool per folder run,
created after the scan and only when **≥ 200 files** are left to process; below that, the
in-process path as today, which is also where tests with stub processors stay. Workers =
min(8, CPUs − 1, ⌊(free GB − 4) / 0.5⌋): 265–290 MB each measured, so 8 is ~2.3 GB of the
laptop's 48. Past 8, DLTS still gains but LTS and final tests are already consumer-bound (F12).

**4.6 Fallback to threads.** The pool is warmed (one no-op per worker, 120 s limit). Failing to
start or warm — `BrokenProcessPool`, `OSError`, a pickling error, the timeout — logs one WARNING
naming the cause, and the run continues on threads. A pool that breaks mid-run (a worker killed)
re-runs its in-flight files, and the rest of the folder, on threads. The batch line names the
mode: `workers: 8 processes`, or `4 threads (processes could not start: …)`.

**4.7 Logging.** The parent makes a `ctx.Queue()` and runs a `QueueListener` on the root
handlers (`respect_handler_level=True`) for the pool's life; the initializer drops any inherited
handlers and installs one `QueueHandler`. One process writes the rotating log — a second
`RotatingFileHandler` cannot rotate on Windows while another process holds the file.

**4.8 Dispatch, Stop, close.** Files go out in 20-file chunks, at most two chunks in flight: no
barrier per batch, and in process mode no `gc.collect` every 20 files (the parent no longer
parses). Stop is checked before each chunk, so test_ingest_cancel's "a multiple of 20" promise
holds. Closing: cancel, the bounded wait the app already has, then `shutdown(cancel_futures=True)`
and terminate workers still busy after 5 s — safe, because a worker holds no database handle.

**4.9 Tests.** No test's worker can reach any database: the initializer installs §4.3's trap
unconditionally, so no test can forget it. The process path is paid for once — one module, a
module-scoped 2-worker spawn pool (~1.5 s here), the committed fixtures (`tests/fixtures/trim`,
`final_test`, `trimvolts`, `trim_setup`) — and pins: worker outcome == in-process outcome field
by field for every fixture kind; a worker reaching `get_database`, `DatabaseManager(tmp)` or
`sqlite3.connect` yields `internal`, never a verdict (made to fail first by removing the trap); a
worker WARNING reaches the parent's handler; a pool factory that raises finishes on threads and
says so; Stop lands on a multiple of 20; close terminates a stuck worker and leaves committed rows
intact. Everything else runs `analyse_path` in-process with `get_database` patched to raise —
the proof that the body touches no database, at no spawn cost. One test pins that fewer than 200
files never spawn. The 645-file parse gate runs on every task that moves the analysis body.

## 5. A5 — a dropped worker comes back

Throttling becomes an **in-flight cap** `k` over a fixed pool (threads or processes), checked at
every chunk: above 90% memory k − 1 (never below 1), above 95% k = 1 (today's sequential
fallback), and below 80% on two consecutive checks k + 1, up to the pool size. The 10-point gap
and two-check wait stop it oscillating on one reading; each change is logged once (`workers 7 → 8:
memory back to 78%`). A pure `next_cap(k, percent, calm_checks)`, unit-tested over a scripted
memory sequence, plus one run on a patched memory probe whose log shows the recovery. What it
cannot do: an idle worker keeps its ~0.27 GB, so if our own workers are the pressure the answer is
fewer at start (4.5's rule), not the cap.

## 6. Order and expected gain

**The model.** Threads share one GIL, so today's per-file time is a sum: L = P + S + O (parse,
save, other). With processes the consumer's work runs beside the workers', so it is a pipeline:
L ≈ max(P / sₙ, C), where C is the consumer's per-file work (unpickle 0.07 ms + save). F12 shows
the pipeline holds. A0's "A3 alone caps at 1.35x" is withdrawn on both counts: it added a time
that overlaps, and it took P from the spec-less probe.

| ms per file | today | A4 alone | A3 alone | both |
|---|---|---|---|---|
| Mac DLTS | 102–116 | ~1 less | ~16 | **13.3** |
| Mac LTS | ~39 | ~1 less | ~6 | **6.1** |
| Mac final test | ~30 | 18.4 (index) | 24.1 (no index) | **4.4** |
| laptop DLTS, if the ~200 ms is thread-coupled | 772 | ~750–760 | ~81–90 | ~81–90 |
| laptop DLTS, if it is a fixed cost in the parent | 772 | ~750–760 | ~270 | ~255–270 |

Bold, "today" and the final-test cells are measured (F7, F9, F12); the other Mac cells follow
from F1 and F10. Laptop rows are projections: P ≈ 500 (339 × 1.48), S ≈ 29, s₈ = 6.2, the flush
share of S 10–20 ms. A4 alone buys the trim folders almost nothing unless the laptop's per-file
flush is large (the probe's batch-1 commit); for final tests the index is worth ~40%. A LAN
rebuild would take ~3 h on the optimistic row (final tests ~1 h, DLTS ~1.5, LTS ~0.25) and ~8 h on
the pessimistic one, against 21.8 h for B2's Final Test folder alone.

**Why A4 first anyway.** A3 cannot start until the worker's writes are values with somewhere to
go, and the writer is that somewhere. The index plus batching is the biggest single win for the
biggest folder. And the probe must measure the laptop's flush before K and the pragmas are fixed.

**What James measures.** Before anything ships: the probe (§8), once. After A4 lands, and again
after A3: the probe, plus a two-minute app run on DLTS and one on Final Test, reading ms/file and
(after A3) the worker mode from the batch line. A4 has worked if final tests speed up by ≥ 1.4x;
A3 has worked if DLTS falls below a third of before. If A3 does not, the ~200 ms is in the parent,
and the next measurement is the app with the window minimised against visible.

**What would make this spec wrong.** (1) At work, parse with specs ≈ without — then the ~200 ms
is ~350 and more likely parent-side, and A3 buys less. (2) The probe's headless loop is as slow as
A0 — the cost is in the code path after all: profile it on the laptop before A3. (3) The headless
loop is fast but the app is slow — it is the window; A3 still helps (the GUI then contends only
with the consumer), but the GUI needs its own look. (4) The batch-1 commit at work is ≥ 100 ms —
the save *is* the serial cost there, and A4 is the big win for trims too. (5) Processes cannot
start on the work laptop — A3 falls back and buys nothing; that is an IT conversation.

**Over the VPN.** A full pass is capped near 2.4 files/s — ~17 round trips at 54 ms per file and
~2.5 MB/s aggregate through the tunnel, whatever its cause (share-io-cost-model) — whatever runs
at either end. Processes add CPU the link cannot feed. A4 takes one round trip per file off the
consumer (the save-time stat, 3.6), and carrying the hash stops a naive A3 adding a whole-file
re-read (4.2); neither lifts the ceiling. B2's Final Test pace (2.15 files/s) sits on it: if B2 ran
over the VPN, its per-file numbers say nothing about the code (question 1).

## 7. Proposed rulings

1. Ruling: build and run §8's probe at work before any production change — the laptop's flush and the unexplained ~200 ms set K, the pragmas and the expected gain — cost if wrong: one six-minute run.
2. Ruling: the batch line and `Ingest so far` print the save's CPU beside its wall time — wall is 85% GIL waiting in the thread loop (F7) — cost if wrong: one more number.
3. Ruling: pool_probe, ingest_speed_probe and parse_locality_probe copy `model_specs` from a named database into their throwaway ones — they measure an analysis a third cheaper than the ingest's (F8) — cost if wrong: none.
4. Ruling: one transaction per batch, opened with `BEGIN IMMEDIATE`, a SAVEPOINT per file — without the explicit BEGIN pysqlite commits at the first RELEASE (F4) — cost if wrong: per-file commits survive, silently.
5. Ruling: flush at 20 files or 2 s, and at generator end, Stop and folder end — the measured knee lies between 10 and 50, 20 is the cancel chunk, and the lock is held ≤ ~0.5 s at laptop speed — cost if wrong: retune K from the probe.
6. Ruling: nothing inside a batch may open its own session; every write is a session-taking body and `session()` raises if entered mid-batch — a nested session commits the batch (F5) — cost if wrong: partial batches no one sees.
7. Ruling: a file's rows and its processed marker share one savepoint — then "processed" is exactly "committed" (3.4, F6) — cost if wrong: files marked done with no data.
8. Ruling: `save_batch` becomes a wrapper over `write_batch` and writes trim_passes and trim_setup; its per-file-commit body goes — it has no callers — cost if wrong: none.
9. Ruling: the worker's writes (final test, smoothness, skip markers) become values applied by the consumer's writer, as part of A4 — they are A3's prerequisite and batch the final-test folder today — cost if wrong: final tests keep a commit per file.
10. Ruling: no file I/O inside a write transaction; saves record the parse's own (size, mtime, hash) — removes a share round trip per file from under the lock and records the bytes actually parsed — cost if wrong: a stat per file, and the parse-to-save race.
11. Ruling: index `file_hash` on final_test_results and smoothness_results — every final-test save scans 151,793 rows (F9) — cost if wrong: a few MB of index.
12. Ruling: `synchronous=NORMAL` and `cache_size=-65536` on every connection, in `_set_sqlite_pragma`, beside `foreign_keys` — the app has one connection, NORMAL is WAL's documented safe setting, a power cut rolls back at most one checkpoint interval and never corrupts — cost if wrong: one line reverts it.
13. Ruling: worker processes only when ≥ 200 files are left after the scan — start-up swamps small runs, and small-run tests keep their stubs — cost if wrong: a threshold to tune.
14. Ruling: workers = min(8, CPUs − 1, ⌊(free GB − 4)/0.5⌋) — 6.2x at 8, ~0.27 GB each, and the consumer is the ceiling past 8 for light files — cost if wrong: speed left on the table.
15. Ruling: workers never open a database; the initializer traps `get_database`, `DatabaseManager` and `sqlite3.connect`, and a trapped file becomes an `internal` outcome — the guard alone would store spec-less numbers without a word (4.3) — cost if wrong: a quietly wrong rebuild.
16. Ruling: specs, ML thresholds and predictors reach workers as a snapshot taken at folder start, through one resolver shared with `get_model_spec`/`resolve_spec_for_ft` — a spec edited mid-run takes effect at the next folder — cost if wrong: that edit waits one folder.
17. Ruling: spawn on every platform, module-level worker code in `core/ingest_worker.py`, field-level equality in tests — the Mac then tests Windows' start method, and pickled bytes differ across processes (F10) — cost if wrong: tests that pass here and fail there.
18. Ruling: worker logs go through one `QueueListener` in the parent — workers have no logging config, and two processes cannot share a rotating file on Windows — cost if wrong: worker warnings on the console only.
19. Ruling: a pool that cannot start, or breaks, hands its work to threads and the batch line says which mode ran and why — cost if wrong: none.
20. Ruling: Stop stays at a 20-file chunk boundary; close terminates workers after a 5 s grace — they hold nothing — cost if wrong: a slower close.
21. Ruling: A5 is an in-flight cap with hysteresis (−1 above 90%, +1 after two checks below 80%, 1 above 95%) — cost if wrong: a slow recovery.
22. Ruling: two consecutive failed batch commits stop the folder with the error named — cost if wrong: one run stops that might have limped on.
23. Ruling: order is probe → honest batch line → A4 (index, pragmas, writer, values out of the worker) → A3 → A5 — the brief's order, but because A3 depends on A4, not because the save is 432 ms — cost if wrong: none; it is the dependency order.

## 8. The probe James runs at work

One command in PowerShell, from the repo:

    cd C:\dev\laser-trim-ai-system
    .\.venv\Scripts\python scripts\ingest_save_probe.py "\\192.168.66.9\BTXData\Departments\System Data\TEST_DATA\DLTS" 120

It copies `data\analysis.db` with SQLite's backup API from a `mode=ro` connection (production
is never opened for writing; the copy is consistent even with the app open) to
`%TEMP%\ingest_save_probe_<pid>.db`, refusing if that drive has under twice the database free.
Every manager it builds is on that copy, injected into both globals before any `Processor`. It
takes the first N files in discovery order, copies them to a local temp folder, parses them
once and deletes their rows from the copy. Windows has no instant clone, so each setting saves
the same results under fresh identities (a per-setting filename prefix and a derived hash) —
every save stays a new file's INSERT into one copy. At the end, on Ctrl-C too, it deletes the
copy and the temp files. `--no-loop` stops after SAVE (~3 min); `--keep` keeps the copy. ~6
minutes for 120 DLTS files. Until A4 lands it carries its own copy of the batch writer, then
calls the real one. Output — `nn.n` stands for a number; the arrows are printed:

    ingest save probe  <date time>  Python <x.y.z>  <n> CPUs  <n> GB RAM (<n> free)  <drive>: <n> GB free
    copy   <path>  <n.nn> GB  made in <n> s from data\analysis.db (read-only)
    files  120 of <n> from <folder>, first in discovery order, copied locally

    SAVE  120 new results per setting, median of 2 rounds, ms per file
    batch  sync    cache   total  python   sql  commit    p99
        1  FULL     2MB     nn.n    nn.n  nn.n    nn.n   nn.n   <- today
        1  FULL    64MB     nn.n    ...                         (then NORMAL 2MB, NORMAL 64MB)
       10  FULL     2MB     nn.n    ...                         (and the other three)
       50  FULL     2MB     nn.n    ...                         (and the other three)
        1  OFF      2MB     nn.n    ...                         <- reference only: no flush at all

    LOOP  the same 120 files, local copies, model specs from the copy, ms per file
    parse+analyse, one at a time, without specs (what pool_probe measured)   nn.n
    parse+analyse, one at a time, with specs (what the ingest does)          nn.n
    parse+analyse, 4 threads, with specs                                     nn.n
    parse+analyse, 8 processes, with specs (pool ready in n.n s)             nn.n
    today's loop, headless (run_folder)                                      nn.n
       of which save: wall nn.n, cpu nn.n | GC pauses nn.n | rest nn.n
    copy and temp files deleted.

The 12 settings run interleaved, two rounds in opposite orders; each prints one line with its
split (Python/ORM, SQL, commit) and p99. `synchronous=OFF` appears only as the floor that shows
how much of `commit` is the flush. Send the whole output back.

## 9. Tasks (smallest reviewable units; each lands tests first, made to fail, then the full
`run_test_gate.py`, `check_no_customer_values.py`, and TRACKER in the same commit)

1. **The probe** (`scripts/ingest_save_probe.py`) and ruling 3 for the old probes. Tests: the
   source is opened only `mode=ro` and a copy path inside `data/` is refused; a 3-file
   `--no-loop` run on a tmp database built from fixtures prints every SAVE line. A BRING_TO_WORK step.
2. **Honest batch line** (3.10). Test: a save stub that sleeps (wall, no CPU) and one that spins
   (CPU) are told apart.
3. **The final-test and smoothness `file_hash` indexes** (3.7). Tests: EXPLAIN says SEARCH; the
   migration is idempotent. QA-sweep entry.
4. **Pragmas** (3.8). Test: every new connection reports foreign_keys 1, synchronous 1,
   cache_size −65536. QA-sweep entry on the app's own connection.
5. **Session-taking trim body** — `_save_analysis_in(session, a, stat, hash)`; `save_analysis`
   wraps it; `_record_processed_file` and `_write_failure_marker` take the carried stat and hash.
   Tests: rows identical to today's for every fixture; no `os.stat` or file read when values are
   given.
6. **`write_batch`** — BEGIN IMMEDIATE, savepoint per item, outcomes, the nested-session guard,
   `save_batch` rebuilt on it. Tests: F6's two cases (subprocess, `os._exit`); F5 now raises;
   save_batch rows == save_analysis rows.
7. **Session-taking final-test, smoothness and skip-marker bodies** (3.3); public methods wrap
   them. Tests: identical rows to the old paths, each also inside `write_batch`.
8. **`SpecSnapshot`**, one resolver for the snapshot and `get_model_spec`/`resolve_spec_for_ft`.
   Test: snapshot and database answer identically for every fixture model plus section-letter and
   alias cases.
9. **Values out of the worker** — `analyse_path` → `Outcome` carrying stat and hash; the thread
   pool runs it; `process_batch` hands outcomes to a writer (V6 passes one; with none, each write
   is applied at once — V5's loop keeps working). Tests: no database touched for every fixture
   kind (`get_database` patched to raise); V5's loop still saves final tests; 645-file gate.
10. **`run_folder` on the writer** — buckets from committed outcomes, flush on Stop and at the
    end, ruling 22. Update the tests that fake `db.save_analysis`. Tests: the cancel suite unchanged;
    a failed save counts as an error; buckets sum to processed.
11. **A3: `core/ingest_worker.py` and the process pool** — context, trap, logging, chunked
    dispatch, fallback, close. Tests: §4.9. QA-sweep entry: process mode on the sweep's files gives
    the thread mode's rows.
12. **A5** — `next_cap` and its wiring (§5). Then the probe switched to the real writer,
    BRING_TO_WORK's before-and-after steps, and TRACKER A0's Amdahl paragraph corrected.

## 10. Questions for James

1. Where did the B2 rebuild run — at work, or over the VPN overnight? Its Final Test pace (2.15
   files/s) sits on the VPN's ceiling (F13).
2. Was the V6 window open and visible during A0's 674-file run? (The probe separates it anyway.)
3. What does `.\.venv\Scripts\python --version` say on the laptop? (GC differs before 3.14.)
