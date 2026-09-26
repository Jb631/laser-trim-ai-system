r"""What a SAVE costs on this disk, and where the ingest LOOP's time goes -- measured on a copy.

    cd C:\dev\laser-trim-ai-system
    .\.venv\Scripts\python scripts\ingest_save_probe.py "\\192.168.66.9\BTXData\Departments\System Data\TEST_DATA\DLTS" 120

Send the WHOLE output back. About 6 minutes for 120 DLTS files. `--no-loop` stops after the SAVE
block (~3 min); `--keep` keeps the copy of the database; `--db PATH` names the database to copy
(default: this checkout's data\analysis.db).

WHY (spec docs/superpowers/specs/2026-09-25-ingest-speed-design.md, sections 2 and 8). On the Mac a
save costs ~5 ms and batching changes almost nothing, because macOS's fsync does not empty the
drive's cache -- Windows' does, on every commit. Only the laptop's own disk can say what batching
the saves buys there: the SAVE block. And ~200 ms of every file in the run at work (A0) does not
reproduce on the Mac: the LOOP block times the same files alone, with and without model specs, on
threads and on processes, and through the ingest's own loop, so the difference says where it lives.

SAFETY. The database is opened ONLY read-only (`mode=ro`) and copied with SQLite's backup API --
consistent even with the app open -- to %TEMP%\ingest_save_probe_<pid>.db. Everything is measured
on that copy: every DatabaseManager this builds is on it, injected into BOTH `_db_manager` globals
before any Processor exists (the processor reaches its database through get_database()). A copy
path inside any `data` folder is refused, and so is a drive with under twice the database free.
At the end -- on Ctrl-C too -- the copy and the local copies of the files are deleted.

THE SAVE LINES. The first N files (in the ingest's own discovery order) are copied to a local temp
folder, parsed ONCE, and their rows deleted from the copy. Each setting then saves those same
results under fresh identities -- a per-setting filename prefix and a content hash derived from the
real one -- so every save is a new file's INSERT, never an update. Batch 1 is `save_analysis`, one
commit per file; 10 and 50 are the app's own batch writer (`write_batch`, spec Task 6 -- what the
ingest has saved with since Task 10): ONE transaction opened with BEGIN IMMEDIATE and a SAVEPOINT
per file -- without that explicit BEGIN pysqlite commits at the first RELEASE, and a "batch" is one
commit per file again (spec F4). The 12 settings run interleaved, two rounds in opposite orders;
each line is
the median of its two rounds, in ms per file: total, then Python/ORM (building the rows, the unit
of work), SQL (the statements themselves) and commit (the WAL write, its flush, and any automatic
checkpoint); p99 is over single saves at batch 1 and over whole batches otherwise. `<- today` marks
whichever row matches the app's OWN pragmas, read fresh off its connection -- `1 FULL 2MB` until
Task 4 landed (ruling 12), `1 NORMAL 64MB` since; `1 OFF` is only the floor -- no flush at all, never
a proposal.

THE LOOP LINES. `rest` is what the ingest's own loop costs per file beyond parsing on 4 threads and
the save's own CPU: batch barriers, GC, bookkeeping -- and whatever the unexplained part is. The
process line runs the ingest's OWN worker processes (core/ingest_worker.py, spec section 4: spawn,
the database trap, the model specs as a SpecSnapshot); the last line is the same loop again with
those worker processes parsing while this process saves (A3) -- the app does that for a folder
with 200 or more new files. The LOOP runs under the app's OWN pragmas (read from the copy's
connection as the app opens it), never under whichever SAVE setting happened to run last.

POINT IT AT A LASER FOLDER (DLTS or LTS). Final-test and smoothness files are saved by the processor
itself the moment they are parsed, so in another folder they would land in the copy untimed and
un-forgotten; the SAVE block times trim saves only.

WHEN SOMETHING GOES WRONG the last line says so in words -- `FAILED: <step>: <error> -- ...` --
with the copy and temp files already deleted (or it says which could not be), and the Python
detail above it for Claude. Exit codes: 0 done, 1 failed, 2 refused, 130 interrupted.
"""
import argparse
import gc
import hashlib
import logging
import ntpath
import os
import platform
import re
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _db_guard import is_production_db  # noqa: E402

BATCHES = (1, 10, 50)
SYNCS = ("FULL", "NORMAL")
CACHES_MB = (2, 64)
# PRAGMA cache_size values, in KiB when negative. "2MB" is SQLite's own default (-2000) -- what the
# app ran before Task 4; "64MB" is ruling 12's value, in _set_sqlite_pragma since.
CACHE_PRAGMA = {2: -2000, 64: -65536}
SYNC_CODE = {"OFF": 0, "NORMAL": 1, "FULL": 2}
SETTINGS = [(b, s, c) for b in BATCHES for s in SYNCS for c in CACHES_MB]
FLOOR = (1, "OFF", 2)
ROUNDS = 2
HEADER = "batch  sync    cache   total  python   sql  commit    p99"
LOOP_LABELS = ("parse+analyse, one at a time, without specs (what pool_probe measured)",
               "parse+analyse, one at a time, with specs (what the ingest does)",
               "parse+analyse, 4 threads, with specs",
               "parse+analyse, {n} processes, with specs (pool ready in {ready:.1f} s)",
               "today's loop, headless (run_folder)",
               "the same loop on worker processes, headless (run_folder)")
LOOP_WIDTH = len(LOOP_LABELS[0])


# ---------------------------------------------------------------------------------------------
# refusals -- every one of them decided before a single byte is written

def _inside_a_data_folder(path) -> bool:
    """True when any folder above `path` is named `data` -- in any case and any spelling Windows
    accepts: back- or forward slashes, a drive or UNC root, relative, with `..` in it.

    Resolved first (relative -> absolute, `..` folded away, symlinks followed); a path written
    with backslashes is then normalised the Windows way whatever machine this runs on, so the
    rule means the same thing on the Mac that tests it as on the laptop that runs it."""
    try:
        text = str(Path(path).resolve())
    except OSError:
        text = os.path.abspath(str(path))
    if "\\" in text:
        text = ntpath.normpath(text)
    parts = [p for p in re.split(r"[\\/]+", text) if p]
    return any(p.casefold() == "data" for p in parts[:-1])


def _refusal(source: Path, copy: Path, files_dir: Path, folder: Path, n: int):
    """Why this run must not start, or None."""
    if n < 1:
        return "N must be at least 1"
    if not folder.is_dir():
        return f"not a folder: {folder}"
    if not source.is_file():
        return f"no such database: {source}"
    if not copy.parent.is_dir():
        return f"no such temp folder: {copy.parent}"
    if _inside_a_data_folder(copy):
        return (f"the copy would land in {copy.parent}, inside a `data` folder -- the probe never "
                "writes near a database it could be mistaken for. Pass --tmp elsewhere.")
    if is_production_db(copy, REPO) or copy.resolve() == source.resolve():
        return f"{copy} cannot be told apart from the database it copies"
    for p in (copy, files_dir):
        if p.exists():
            return f"{p} already exists (left from an earlier run?) -- delete it or pass --tmp"
    need = source.stat().st_size + (Path(str(source) + "-wal").stat().st_size
                                    if Path(str(source) + "-wal").exists() else 0)
    free = shutil.disk_usage(copy.parent).free
    if free < 2 * need:
        return (f"{copy.parent} has {free / 1e9:.1f} GB free; the copy needs twice the database, "
                f"{2 * need / 1e9:.1f} GB. Free some space or pass --tmp on another drive.")
    return None


# ---------------------------------------------------------------------------------------------
# the copy

def _backup(source: Path, copy: Path) -> None:
    """SQLite's backup API from a READ-ONLY connection: the source is never opened for writing,
    and the copy is one consistent file (journal included) even with the app writing."""
    src = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(str(copy))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _open_copy(copy: Path):
    """A manager on the copy, injected into BOTH globals -- before any Processor exists."""
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    db = mgr.DatabaseManager(copy)     # runs the app's own migrations -- on the copy
    mgr._db_manager = db
    dbpkg._db_manager = db
    return db


def _raw(db) -> sqlite3.Connection:
    """The one DBAPI connection behind the manager's StaticPool."""
    with db._engine.connect() as c:
        return c.connection.dbapi_connection


def _app_pragmas(db):
    """(synchronous, cache_size) as the APP's own connection has them -- read before any setting
    runs, so it is whatever the app's connect listener sets today plus SQLite's defaults."""
    rc = _raw(db)
    return (rc.execute("PRAGMA synchronous").fetchone()[0],
            rc.execute("PRAGMA cache_size").fetchone()[0])


def _set_pragmas(db, synchronous, cache_size) -> None:
    rc = _raw(db)
    rc.execute(f"PRAGMA synchronous={int(synchronous)}")
    rc.execute(f"PRAGMA cache_size={int(cache_size)}")


def _today(app):
    """The SAVE setting that IS the app today (batch 1, its own pragmas), or None."""
    sync, cache = app
    for b, s, c in SETTINGS:
        if b == 1 and SYNC_CODE[s] == sync and CACHE_PRAGMA[c] == cache:
            return (b, s, c)
    return None


# ---------------------------------------------------------------------------------------------
# the files

def _pick(folder: Path, n: int):
    from laser_trim_analyzer.core.ingest_run import discover_excel_files
    files, _stats = discover_excel_files(str(folder))
    return files[:n], len(files)


def _copy_locally(files, folder: Path, files_dir: Path):
    """Local copies, keeping each file's own name (the parser reads model and serial from it) and
    its mtime; a file in a subfolder keeps that subfolder, so two same-named files cannot collide."""
    out = []
    for f in files:
        try:
            rel = Path(f).relative_to(folder)
        except ValueError:
            rel = Path(Path(f).name)
        dst = files_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        out.append(dst)
    return out


def _parse_once(paths):
    """The ingest's own Processor (specs and ML thresholds from the copy); trim results only --
    final-test and smoothness files are saved inside the processor and are not a save to time."""
    from laser_trim_analyzer.core.processor import Processor
    proc = Processor()
    results = []
    for p in paths:
        r = proc.process_file(p)
        if r is not None and getattr(r, "file_type", "trim") == "trim":
            results.append(r)
    return results


def _forget(db, results) -> None:
    """Delete these files' rows from the copy, so saving them -- and the loop -- see NEW files."""
    from sqlalchemy import text
    from laser_trim_analyzer.utils.hashing import calculate_file_hash
    with db.session() as s:
        for r in results:
            m = r.metadata
            ids = [row[0] for row in s.execute(text(
                "SELECT id FROM analysis_results WHERE filename = :f AND model = :m AND serial = :s"),
                {"f": m.filename, "m": m.model, "s": m.serial})]
            for aid in ids:
                s.execute(text("DELETE FROM qa_alerts WHERE analysis_id = :a"), {"a": aid})
                s.execute(text("DELETE FROM trim_passes WHERE track_result_id IN "
                               "(SELECT id FROM track_results WHERE analysis_id = :a)"), {"a": aid})
                s.execute(text("DELETE FROM track_results WHERE analysis_id = :a"), {"a": aid})
                s.execute(text("DELETE FROM trim_setup WHERE analysis_id = :a"), {"a": aid})
                s.execute(text("DELETE FROM analysis_results WHERE id = :a"), {"a": aid})
            s.execute(text("DELETE FROM processed_files WHERE file_hash = :h OR filename = :n"),
                      {"h": calculate_file_hash(Path(m.file_path)), "n": Path(m.file_path).name})


# ---------------------------------------------------------------------------------------------
# the SAVE block

def _save_batch_probe(db, analyses):
    """The app's OWN batch writer (`write_batch`, spec 3.2; Task 12 retired the copy this probe
    carried): ONE transaction, opened with BEGIN IMMEDIATE, a SAVEPOINT per file whose body is
    save_analysis's own (`_save_analysis_in`), the processed marker inside it.

    The explicit BEGIN is the whole point (spec F4): pysqlite emits no BEGIN before a SAVEPOINT,
    SQLite then starts the transaction AT the savepoint, and RELEASE of that outermost savepoint is
    a COMMIT -- one commit per file, which is what batching exists to avoid.

    Each file's (size, mtime) and hash are taken before the transaction, as the ingest's writer is
    handed them -- and inside this timed call, so the totals stay comparable with batch 1's."""
    from laser_trim_analyzer.database.manager import BatchCommitError, TrimWrite
    items = [TrimWrite(a, *db._file_identity(a.metadata.file_path)) for a in analyses]
    try:
        outcomes = db.write_batch(items)
    except BatchCommitError as e:
        outcomes = e.outcomes
    return [("saved", o.row_id) if o.status == "saved"
            else (o.status, (o.reason or o.status)[:160]) for o in outcomes]


class _SaveTimer:
    """DBAPI statement time and commit time, from the engine's and the session's own events."""

    def __init__(self, db):
        self.db, self.sql, self.commit = db, 0.0, 0.0
        self._t = self._c = None

    def _before(self, conn, cursor, statement, parameters, context, executemany):
        self._t = time.perf_counter()

    def _after(self, conn, cursor, statement, parameters, context, executemany):
        if self._t is not None:
            self.sql += time.perf_counter() - self._t
            self._t = None

    def _on_commit(self, conn):
        self._c = time.perf_counter()

    def _after_commit(self, session):
        if self._c is not None:
            self.commit += time.perf_counter() - self._c
            self._c = None

    def __enter__(self):
        from sqlalchemy import event
        from sqlalchemy.orm import Session
        self._hooks = [(self.db._engine, "before_cursor_execute", self._before),
                       (self.db._engine, "after_cursor_execute", self._after),
                       (self.db._engine, "commit", self._on_commit),
                       (Session, "after_commit", self._after_commit)]
        for target, name, fn in self._hooks:
            event.listen(target, name, fn)
        return self

    def __exit__(self, *exc):
        from sqlalchemy import event
        for target, name, fn in self._hooks:
            event.remove(target, name, fn)


class _DerivedHashes:
    """While a setting runs, the save's content hash is the REAL one (the same cached lookup the
    app makes) run through sha256 with that setting's tag: a new file every time, at no new cost."""

    def __enter__(self):
        from laser_trim_analyzer.database import manager as mgr
        self._mgr, self._real, self.tag = mgr, mgr.calculate_file_hash, None

        def derived(file_path, use_cache=True, known_stat=None):
            real = self._real(file_path, use_cache=use_cache, known_stat=known_stat)
            if self.tag is None:
                return real
            return hashlib.sha256(f"{real}:{self.tag}".encode()).hexdigest()
        mgr.calculate_file_hash = derived
        return self

    def __exit__(self, *exc):
        self._mgr.calculate_file_hash = self._real


def _p99(samples):
    s = sorted(samples)
    return s[min(len(s) - 1, int(len(s) * 0.99))] if s else 0.0


def _run_setting(db, results, setting, tag, timer, hashes):
    batch, sync, cache_mb = setting
    _set_pragmas(db, SYNC_CODE[sync], CACHE_PRAGMA[cache_mb])
    items = []
    for r in results:
        a = r.model_copy(deep=True)
        a.metadata.filename = f"PROBE{tag}_{a.metadata.filename}"
        items.append(a)
    hashes.tag = tag
    sql0, com0 = timer.sql, timer.commit
    samples, failed = [], []
    t0 = time.perf_counter()
    if batch == 1:
        for a in items:
            t = time.perf_counter()
            try:
                db.save_analysis(a)
            except Exception as e:
                failed.append(f"{type(e).__name__}: {e}"[:160])
            samples.append(time.perf_counter() - t)
    else:
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            t = time.perf_counter()
            outs = _save_batch_probe(db, chunk)
            samples.append((time.perf_counter() - t) / len(chunk))
            failed += [why for what, why in outs if what != "saved"]
    wall = time.perf_counter() - t0
    hashes.tag = None
    n = len(items)
    total, sql, commit = wall / n * 1e3, (timer.sql - sql0) / n * 1e3, (timer.commit - com0) / n * 1e3
    return {"total": total, "python": total - sql - commit, "sql": sql, "commit": commit,
            "p99": _p99(samples) * 1e3, "failed": failed}


def _row(setting, m, today=None) -> str:
    b, s, c = setting
    line = (f"{b:>5}  {s:<7}{c:>3}MB{m['total']:>9.1f}{m['python']:>8.1f}{m['sql']:>6.1f}"
            f"{m['commit']:>8.1f}{m['p99']:>7.1f}")
    if setting == today:
        line += "   <- today"
    elif setting == FLOOR:
        line += "   <- reference only: no flush at all"
    if m["failed"]:
        line += f"   ({len(m['failed'])} saves FAILED: {m['failed'][0]})"
    return line


def _run_save_block(db, results, today=None) -> None:
    n = len(results)
    print(f"SAVE  {n} new results per setting, median of {ROUNDS} rounds, ms per file")
    print(HEADER)
    if not results:
        print("  no trim files among them -- point the probe at a laser folder (DLTS or LTS)")
        return
    order = SETTINGS + [FLOOR]
    runs = {s: [] for s in order}
    with _SaveTimer(db) as timer, _DerivedHashes() as hashes:
        for rnd in range(ROUNDS):
            for k, setting in enumerate(order if rnd % 2 == 0 else order[::-1]):
                _status(f"SAVE  round {rnd + 1} of {ROUNDS}, setting {k + 1} of {len(order)}")
                runs[setting].append(_run_setting(db, results, setting,
                                                  f"{rnd}{order.index(setting):02d}", timer, hashes))
    _status("")
    for setting in order:
        rs = runs[setting]
        m = {k: statistics.median(r[k] for r in rs)
             for k in ("total", "python", "sql", "commit", "p99")}
        m["failed"] = [f for r in rs for f in r["failed"]]
        print(_row(setting, m, today))


# ---------------------------------------------------------------------------------------------
# the LOOP block

def _time_serial(proc, paths) -> float:
    proc.process_file(paths[0])                       # warm the imports, not the measurement
    t = time.perf_counter()
    for p in paths:
        proc.process_file(p)
    return (time.perf_counter() - t) / len(paths) * 1e3


def _time_threads(proc, paths, n) -> float:
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(proc.process_file, paths))
    return (time.perf_counter() - t) / len(paths) * 1e3


def _sendable_predictors(predictors):
    """(what can be sent to a worker, None) -- or ({}, the reason) when the ML predictors cannot
    be pickled. Then the process line runs WITHOUT them and must say so: next to lines that ran
    with them it would otherwise read as the same analysis, and it is not."""
    import pickle
    predictors = dict(predictors)
    if not predictors:
        return {}, None
    try:
        pickle.dumps(predictors)
    except Exception as e:
        return {}, (f"<- WITHOUT the {len(predictors)} ML predictors: they could not be sent to a "
                    f"worker ({type(e).__name__}: {e})")[:240]
    return predictors, None


def _process_line(n, ready, value, note) -> str:
    line = _loop_line(LOOP_LABELS[3].format(n=n, ready=ready), value)
    return f"{line}   {note}" if note else line


def _time_processes(db, proc, paths, n):
    """(ms/file, seconds until every worker was up, a note if the predictors stayed behind -- or if
    any file came back `internal`) on the ingest's OWN worker processes (core/ingest_worker.py:
    spawn, the database trap, one log queue): the Processor's own analysis, with the copy's model
    specs, ML thresholds and -- when they can be sent -- predictors carried as a SpecSnapshot, as
    the ingest carries them. Start-up is NOT charged to throughput."""
    from laser_trim_analyzer.core import ingest_worker
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database.specs import SpecSnapshot
    predictors, note = _sendable_predictors(proc._model_predictors)
    snapshot = SpecSnapshot(specs=tuple(db.get_all_model_specs()),
                            ml_thresholds=dict(proc._model_thresholds), ml_predictors=predictors)
    sender = Processor(config=proc.config, snapshot=snapshot, ml_storage_path=proc.ml_storage_path)
    pool = ingest_worker.WorkerPool.start(ingest_worker.context_for(sender), n)
    try:
        t = time.perf_counter()
        outcomes = [f.result() for f in [pool.submit(p) for p in paths]]
        wall = time.perf_counter() - t
    finally:
        pool.close()
    internal = [o for o in outcomes if o is not None and o.internal]
    if internal:
        why = (f"<- {len(internal)} of {len(paths)} files came back internal "
               f"({internal[0].internal})")[:240]
        note = f"{note} {why}" if note else why
    return wall / len(paths) * 1e3, pool.ready_seconds, note


from laser_trim_analyzer.core.processor import Processor as _Processor  # noqa: E402

_BUILT = []


class _Seen(_Processor):
    """The ingest's own Processor, remembering each one built (the loop's own scan timings live
    on it). At module level, so a worker process can build it too: a worker is handed its parent's
    processor class by reference, and a class it cannot import would send the loop to threads."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _BUILT.append(self)


def _time_run_folder(db, files_dir: Path, paths, processes: int = 0):
    """Today's loop, exactly as the ingest runs it, headless: run_folder over the local copies,
    incremental, on the parallel path (the threshold is lowered so N files take it, as every real
    folder does) -- with `processes`, on that many of the ingest's worker processes, however few
    files (the app starts them from 200). The post-batch work (re-link, drift, findings) is not the
    per-file loop and is skipped. Returns (loop ms/file, save wall, save cpu, gc pauses) per file --
    the save figures the batch writer's own (every save is in its `write_batch` since Task 10) --
    and which workers ran."""
    from sqlalchemy import text
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core import ingest_run, ingest_worker

    # The SAVE block recorded these local paths as processed; the loop must meet them as new.
    prefix = str(files_dir)
    with db.session() as s:
        s.execute(text("DELETE FROM processed_files WHERE substr(file_path, 1, :n) = :p"),
                  {"n": len(prefix), "p": prefix})

    pauses = {"t": 0.0, "at": None}

    def on_gc(phase, info):
        if phase == "start":
            pauses["at"] = time.perf_counter()
        elif pauses["at"] is not None:
            pauses["t"] += time.perf_counter() - pauses["at"]
            pauses["at"] = None

    cfg = Config()
    cfg.processing.turbo_mode_threshold = 1
    real = (ingest_run._post_batch, ingest_run.Processor, ingest_worker.PROCESS_MIN_FILES,
            ingest_worker.MAX_WORKERS)
    ingest_run._post_batch = lambda *a, **k: None
    ingest_run.Processor = _Seen
    if processes:
        ingest_worker.PROCESS_MIN_FILES, ingest_worker.MAX_WORKERS = 1, processes
    del _BUILT[:]
    gc.callbacks.append(on_gc)
    try:
        res = ingest_run.run_folder(str(files_dir), db=db, config=cfg, incremental=True)
    finally:
        gc.callbacks.remove(on_gc)
        (ingest_run._post_batch, ingest_run.Processor, ingest_worker.PROCESS_MIN_FILES,
         ingest_worker.MAX_WORKERS) = real
    if not res.ok:
        raise RuntimeError(f"run_folder failed: {res.error}")
    done = int(getattr(res.summary, "processed", 0) or 0) or len(paths)
    loop_s = _BUILT[-1].last_scan_stats.get("process_seconds", 0.0)   # the consumer: built
    #                                          after the run's own planning processor
    return (loop_s / done * 1e3, res.phases.get("save", 0.0) / done * 1e3,
            res.phases.get("save_cpu", 0.0) / done * 1e3, pauses["t"] / done * 1e3, res.workers)


def _loop_line(label: str, value: float) -> str:
    return f"{label:<{LOOP_WIDTH}}{value:>7.1f}"


def _run_loop_block(db, paths, files_dir: Path, procs: int, results=()) -> None:
    from laser_trim_analyzer.core.processor import Processor
    print(f"LOOP  the same {len(paths)} files, local copies, model specs from the copy, ms per file")

    _status("LOOP  parsing without specs")
    bare = Processor()
    no_spec = bare._get_spec_for_analysis("")     # the processor's own "no spec" answer
    bare._get_spec_for_analysis = lambda *a, **k: dict(no_spec)
    print(_loop_line(LOOP_LABELS[0], _time_serial(bare, paths)), flush=True)

    _status("LOOP  parsing with specs")
    real = Processor()
    print(_loop_line(LOOP_LABELS[1], _time_serial(real, paths)), flush=True)

    _status("LOOP  parsing on 4 threads")
    threads = _time_threads(real, paths, 4)
    print(_loop_line(LOOP_LABELS[2], threads), flush=True)

    _status(f"LOOP  parsing on {procs} processes")
    per, ready, note = _time_processes(db, real, paths, procs)
    print(_process_line(procs, ready, per, note), flush=True)

    _status("LOOP  today's loop")
    loop, save_wall, save_cpu, gc_ms, _ = _time_run_folder(db, files_dir, paths)
    print(_loop_line(LOOP_LABELS[4], loop))
    print(f"   of which save: wall {save_wall:.1f}, cpu {save_cpu:.1f} | GC pauses {gc_ms:.1f} | "
          f"rest {loop - threads - save_cpu:.1f}", flush=True)

    _status(f"LOOP  the loop on {procs} worker processes")
    _forget(db, results)            # the loop above stored them: this one must meet them new too
    loop, save_wall, save_cpu, _, workers = _time_run_folder(db, files_dir, paths, procs)
    _status("")
    print(_loop_line(LOOP_LABELS[5], loop))          # how many ran: the `workers` detail below
    print(f"   of which save: wall {save_wall:.1f}, cpu {save_cpu:.1f} | workers {workers}")


# ---------------------------------------------------------------------------------------------

def _status(msg: str) -> None:
    """Progress on stderr, overwritten in place -- stdout stays exactly the report."""
    if sys.stderr.isatty():
        sys.stderr.write("\r" + msg.ljust(78)[:78] + ("\r" if not msg else ""))
        sys.stderr.flush()


def _machine_line(tmp_root: Path) -> str:
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram = f"{vm.total / 2**30:.1f} GB RAM ({vm.available / 2**30:.1f} free)"
    except Exception:
        ram = "RAM unknown"
    drive = Path(tmp_root).resolve().drive.rstrip(":") or "temp"
    return (f"ingest save probe  {datetime.now():%Y-%m-%d %H:%M}  Python {platform.python_version()}"
            f"  {os.cpu_count()} CPUs  {ram}  {drive}: {shutil.disk_usage(tmp_root).free / 1e9:.0f} GB free")


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="What a save costs on this disk (see the docstring).")
    ap.add_argument("folder", help="a laser folder on the share, e.g. ...\\TEST_DATA\\DLTS")
    ap.add_argument("n", type=int, nargs="?", default=120, help="how many files (default 120)")
    ap.add_argument("--no-loop", action="store_true", help="stop after the SAVE block")
    ap.add_argument("--keep", action="store_true", help="keep the copy of the database")
    ap.add_argument("--db", help="the database to copy (default: data\\analysis.db); read-only")
    ap.add_argument("--tmp", help="where the copy and the file copies go (default: the temp folder)")
    ap.add_argument("--procs", type=int, default=min(8, os.cpu_count() or 1),
                    help="worker processes for the LOOP block (default 8)")
    return ap.parse_args(argv)


FAILED_TAIL = "send Claude this output"


def _robust_stdout() -> None:
    """A console that cannot show a character must never turn the report into a crash."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def _failed(step: str, error: BaseException, detail: str, what_happened: str) -> None:
    """The Python detail first, for Claude -- then ONE plain line, always the last one printed."""
    if detail:
        print(detail.rstrip())
    print(f"FAILED: {step}: {type(error).__name__}: {error} — {what_happened}; {FAILED_TAIL}")


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _robust_stdout()
    source = Path(args.db) if args.db else REPO / "data" / "analysis.db"
    source_label = str(source) if args.db else str(Path("data") / "analysis.db")
    tmp_root = Path(args.tmp) if args.tmp else Path(tempfile.gettempdir())
    folder = Path(args.folder)
    copy = tmp_root / f"ingest_save_probe_{os.getpid()}.db"
    files_dir = tmp_root / f"ingest_save_probe_{os.getpid()}_files"

    # Outside the clean-up below on purpose: a refusal names files that are NOT ours to delete
    # (a leftover from another run, say), and before this passes nothing has been written.
    try:
        problem = _refusal(source, copy, files_dir, folder, args.n)
    except Exception as e:
        _failed("checking the paths", e, traceback.format_exc(), "nothing was written")
        return 1
    if problem:
        print(f"REFUSED: {problem}")
        return 2

    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    before = (mgr._db_manager, getattr(dbpkg, "_db_manager", None))
    quiet = logging.root.manager.disable
    logging.disable(logging.WARNING)       # per-file parser warnings would bury the numbers
    db, interrupted, failure, step = None, False, None, "starting"
    try:
        print(_machine_line(tmp_root), flush=True)
        step = "copying the database"
        _status("copying the database (read-only) ...")
        t = time.perf_counter()
        _backup(source, copy)
        made = time.perf_counter() - t
        print(f"copy   {copy}  {copy.stat().st_size / 1e9:.2f} GB  made in {made:.0f} s from "
              f"{source_label} (read-only)", flush=True)
        step = "opening the copy"
        from laser_trim_analyzer.ml import invalidate_shared_ml_manager
        invalidate_shared_ml_manager()     # a cached ML manager could belong to another database
        db = _open_copy(copy)
        app = _app_pragmas(db)             # the app's own, before any setting touches them
        step = "walking the folder"
        _status("walking the folder ...")
        files, found = _pick(folder, args.n)
        step = "copying the files"
        paths = _copy_locally(files, folder, files_dir)
        print(f"files  {len(paths)} of {found:,} from {folder}, first in discovery order, "
              "copied locally", flush=True)
        step = "parsing the files"
        _status(f"parsing {len(paths)} files once ...")
        results = _parse_once(paths)
        step = "clearing their rows from the copy"
        _forget(db, results)
        print()
        step = "saving the results"
        _run_save_block(db, results, _today(app))
        if not args.no_loop and paths:
            print()
            step = "timing the loop"
            # Explicitly the app's own pragmas: the SAVE block leaves whichever setting ran last.
            _set_pragmas(db, *app)
            _run_loop_block(db, paths, files_dir, max(1, args.procs), results)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as e:
        failure = (step, e, traceback.format_exc())
    finally:
        _status("")
        closing = []
        try:
            if db is not None:
                db.close()
        except Exception as e:             # never allowed to hide what went wrong first
            closing.append(f"the copy would not close ({type(e).__name__}: {e})")
        mgr._db_manager, dbpkg._db_manager = before
        try:
            from laser_trim_analyzer.ml import invalidate_shared_ml_manager
            invalidate_shared_ml_manager()
        except Exception:
            pass
        logging.disable(quiet)
        try:
            left = closing + _cleanup(copy, files_dir, keep=args.keep)
        except Exception as e:
            left = closing + [f"the clean-up itself failed ({type(e).__name__}: {e})"]

    if failure:
        if left:
            what = (f"the copy and temp files could NOT all be deleted ({'; '.join(left)}) — "
                    "delete them by hand")
        elif args.keep and copy.exists():
            what = f"the copy was kept at {copy}; the temp files were deleted"
        else:
            what = "the copy and temp files were deleted"
        _failed(failure[0], failure[1], failure[2], what)
        return 1
    if interrupted:
        print("interrupted")
    if left:
        print("could NOT delete: " + "; ".join(left) + " -- delete by hand")
    elif args.keep and copy.exists():
        print(f"copy kept at {copy} -- delete it when done; temp files deleted.")
    else:
        print("copy and temp files deleted.")
    return 130 if interrupted else 0


def _cleanup(copy: Path, files_dir: Path, *, keep: bool):
    """Delete the copy (unless --keep) and the local file copies. Returns what could not go."""
    left = []
    if not keep:
        for suffix in ("", "-wal", "-shm", "-journal"):
            p = Path(str(copy) + suffix)
            try:
                p.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                left.append(f"{p} ({e})")
    shutil.rmtree(files_dir, ignore_errors=True)
    if files_dir.exists():
        left.append(str(files_dir))
    return left


if __name__ == "__main__":
    raise SystemExit(main())
