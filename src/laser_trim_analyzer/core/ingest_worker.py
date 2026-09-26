"""Worker PROCESSES for the ingest: they parse and analyse; the parent's one consumer saves.

Ingest-speed A3 (spec docs/superpowers/specs/2026-09-25-ingest-speed-design.md, section 4, rulings
13-19). Threads share one GIL, so four parser threads run at 1.0x (F7); spawned processes do not
(F10: 6.2x at 8). Since Task 9 the analysis is a pure function of the file -- `analyse_path`
returns an `Outcome` and makes no write -- so it can run in a child while the parent's consumer
applies every Outcome through the batch writer: the parse and the save overlap instead of adding.

Tk-free, and it never opens a database: every value a worker needs travels in a `WorkerContext`
(the config, the folder's `SpecSnapshot` with its ML thresholds and predictors, the models folder
its PARENT resolved, the processor class), handed to each worker once. What this module owns:

  * the worker side -- `_init_worker` (run once in each child) and `analyse` (per file);
  * the DATABASE TRAP (ruling 15): the initializer makes `get_database`, `DatabaseManager(...)`
    and `sqlite3.connect` raise `WorkerDatabaseAccess` and record where they were reached, and
    `analyse` turns any file during which one was reached into `Outcome(internal=...)` -- never a
    verdict, never a spec-less result (spec 4.3: every such reach in the analysis sits inside an
    `except Exception` that would otherwise let it pass as "no spec"). The conftest guard does not
    reach a spawned child; this trap is its guard, installed unconditionally;
  * logging (ruling 18): each worker drops its handlers and installs ONE QueueHandler; the parent
    runs one listener for the pool's life and hands each record to its OWN logger of that name,
    so one process writes the rotating log (Windows cannot rotate a file another process holds);
  * the pool -- `WorkerPool.start` (spawn on every platform, ruling 17; warmed, 120 s limit;
    `PoolFailed` naming the cause when it cannot start), `submit`, and `close` (cancel, a grace
    for busy workers, then terminate -- safe, a worker holds no database handle; ruling 20);
  * `worker_count` (ruling 14) and `close_worker_pools` (the window's close).

The dispatch -- chunks, Stop, the in-flight cap, the fall back to threads -- is the processor's
(`Processor._process_parallel`): one analysis body, two pools.

Windows is the target: spawn everywhere, module-level worker code, picklable values; Python 3.11+
(no 3.12-only API).
"""
from __future__ import annotations

import logging
import math
import multiprocessing
import os
import pickle
import queue as _queue
import threading
import time
import traceback
import weakref
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Ruling 13: a pool only when this many files are left to process after the scan -- below it the
# start-up (~1.5 s here, unknown and longer at work) is not worth it, and the small runs keep the
# in-process path (where the tests with stub processors live).
PROCESS_MIN_FILES = 200
# Ruling 14: min(MAX_WORKERS, CPUs - 1, floor((free GB - RESERVE_GB) / WORKER_GB)). 265-290 MB per
# worker was measured (F10); past 8 the light folders are already consumer-bound (F12).
MAX_WORKERS = 8
RESERVE_GB = 4.0
WORKER_GB = 0.5
START_TIMEOUT = 120.0     # spec 4.6: a pool that is not up and warm by then is not coming
CLOSE_GRACE = 5.0         # ruling 20: what a busy worker gets before it is terminated

# The folders a worker must REFUSE to load trained models from. Empty in the app. The test suite
# sets it to a checkout's real data/ folders (tests/conftest.py), because its own guard -- no test
# may score with a checkout's real trained models -- cannot reach a spawned child.
REFUSE_MODELS_UNDER: Tuple[str, ...] = ()


class WorkerDatabaseAccess(RuntimeError):
    """A worker process reached for a database. It never may (ruling 15)."""


class PoolFailed(RuntimeError):
    """Worker processes could not be started or warmed; the message says why."""


@dataclass(frozen=True)
class WorkerContext:
    """Everything a worker process needs, sent to each worker once (pickled once, in the parent).

    `snapshot` carries the model specs, the ML thresholds and the trained predictors (ruling 16);
    `ml_models_dir` is where the composite trim-risk models load from -- the PARENT's resolved
    folder, as an absolute path: a spawned child resolving its own would read the real app
    directory's, whatever its parent was told (the Task 9-10 hand-off). `processor_class` is the
    parent's own (sent by reference: a class a child cannot import cannot be sent, and the folder
    then runs on threads). `package_file` is where the parent imported laser_trim_analyzer from --
    a worker that imported another tree refuses to start. `refuse_models_under`: see
    REFUSE_MODELS_UNDER.
    """
    config: Any
    snapshot: Any
    ml_models_dir: str
    processor_class: type
    use_ml: bool = True
    package_file: str = ""
    refuse_models_under: Tuple[str, ...] = ()


def _package_file() -> str:
    import laser_trim_analyzer
    return str(Path(laser_trim_analyzer.__file__).resolve())


def context_for(processor) -> WorkerContext:
    """The context a worker needs to analyse exactly as `processor` does in this process."""
    return WorkerContext(
        config=processor.config,
        snapshot=getattr(processor, "_snapshot", None),
        ml_models_dir=str(Path(processor.ml_storage_path).resolve()),
        processor_class=type(processor),
        use_ml=getattr(processor, "_use_ml", True),
        package_file=_package_file(),
        refuse_models_under=tuple(str(d) for d in REFUSE_MODELS_UNDER),
    )


def worker_count(cpus: Optional[int] = None, free_gb: Optional[float] = None) -> Tuple[int, str]:
    """(how many worker processes, why not more) -- ruling 14:
    min(MAX_WORKERS, CPUs - 1, floor((free GB - RESERVE_GB) / WORKER_GB)). Zero or less means
    none, and the reason says which limit it was."""
    if cpus is None:
        cpus = (getattr(os, "process_cpu_count", None) or os.cpu_count)() or 1
    if free_gb is None:
        try:
            import psutil
            free_gb = psutil.virtual_memory().available / 2 ** 30
        except Exception as e:
            return 0, f"free memory unknown ({type(e).__name__}: {e})"
    by_memory = math.floor((free_gb - RESERVE_GB) / WORKER_GB)
    n = min(MAX_WORKERS, cpus - 1, by_memory)
    if n >= 1:
        return n, ""
    if cpus - 1 < 1:
        return 0, f"{cpus} CPU"
    return 0, f"only {free_gb:.1f} GB free (a worker needs {WORKER_GB} GB beyond {RESERVE_GB:.0f})"


# =============================================================================================
# the worker side -- runs in each spawned child

_PROCESSOR = None
_BARRIER = None
_REACHES: List[str] = []           # what the current file reached for; see analyse()


def _call_site() -> str:
    """The innermost frame outside this module: who reached for the database."""
    for frame in reversed(traceback.extract_stack()[:-2]):
        if Path(frame.filename).name != "ingest_worker.py":
            return f"{Path(frame.filename).name}:{frame.lineno} in {frame.name}"
    return "an unknown caller"


def _refused(what: str):
    def refuse(*args, **kwargs):
        site = _call_site()
        _REACHES.append(f"{what} from {site}")
        raise WorkerDatabaseAccess(
            f"a worker process may not open a database: {what} from {site} (ingest-speed "
            f"ruling 15 -- everything a worker needs travels in its snapshot)")
    return refuse


def _install_trap() -> None:
    """The DATABASE TRAP (ruling 15, spec 4.3), unconditionally: every way this code reaches a
    database -- the module global, a manager of its own, SQLite itself -- records where it was
    reached and raises. `analyse` then refuses the file's result, whatever the analysis made of
    the refusal."""
    import sqlite3
    import sqlite3.dbapi2
    from laser_trim_analyzer.database import manager as _mgr
    import laser_trim_analyzer.database as _dbpkg
    _mgr.get_database = _refused("get_database()")
    _dbpkg.get_database = _refused("get_database()")
    _mgr.DatabaseManager.__init__ = _refused("DatabaseManager(...)")
    sqlite3.connect = _refused("sqlite3.connect")
    sqlite3.dbapi2.connect = _refused("sqlite3.connect")


def _guard_models(folders) -> None:
    """Under the test suite only (REFUSE_MODELS_UNDER): refuse, and record, any load of a trained
    model from these folders -- the conftest guard, carried into the child."""
    from laser_trim_analyzer.ml import composite_risk as _cr
    from laser_trim_analyzer.ml import predictor as _pred
    protected = [Path(f).resolve() for f in folders]

    def check(path):
        target = Path(path).resolve()
        if any(target == d or d in target.parents for d in protected):
            _REACHES.append(f"a checkout's real trained models ({target})")
            raise RuntimeError(f"refusing to load {target}: a test's worker may not read a "
                               "checkout's real trained models")

    real_composite = _cr.CompositeRiskModel.load.__func__
    real_predictor = _pred.ModelPredictor.load

    def composite_load(cls, path):
        check(path)
        return real_composite(cls, path)

    def predictor_load(self, path):
        check(path)
        return real_predictor(self, path)

    _cr.CompositeRiskModel.load = classmethod(composite_load)
    _pred.ModelPredictor.load = predictor_load


def _install_logging(log_queue, log_setup) -> None:
    """ONE QueueHandler, the parent's levels (spec 4.7): nothing inherited, nothing of its own."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(QueueHandler(log_queue))
    levels, disabled = log_setup
    for name, level in levels.items():
        (logging.getLogger(name) if name else root).setLevel(level)
    logging.disable(disabled)


def _init_worker(blob: bytes, log_queue, log_setup, barrier, errors) -> None:
    """Run once in each worker, before it takes a file. Logging first, so a failure below reaches
    the parent's log; then the trap, BEFORE anything is built; then the processor, exactly as the
    parent's -- and a worker whose processor reached for a database while it was built refuses to
    start rather than analyse anything in that state."""
    global _PROCESSOR, _BARRIER
    try:
        _install_logging(log_queue, log_setup)
        _install_trap()
        # Trusted bytes: this worker's own parent pickled them a moment ago (WorkerPool.start),
        # and they arrive over the pipe multiprocessing pickles every task and result across.
        ctx: WorkerContext = pickle.loads(blob)
        here = _package_file()
        if ctx.package_file and here != ctx.package_file:
            raise RuntimeError(f"this worker imported {here}, its parent {ctx.package_file}: "
                               "they would analyse with different code")
        from laser_trim_analyzer import config as _config
        _config._config = ctx.config        # get_config() anywhere in the analysis is the parent's
        if ctx.refuse_models_under:
            _guard_models(ctx.refuse_models_under)
        _PROCESSOR = ctx.processor_class(config=ctx.config, use_ml=ctx.use_ml,
                                         snapshot=ctx.snapshot,
                                         ml_storage_path=Path(ctx.ml_models_dir))
        if _REACHES:
            raise WorkerDatabaseAccess(f"building the worker's processor reached for a "
                                       f"database: {_REACHES[0]}")
        _BARRIER = barrier
    except BaseException as e:
        try:
            errors.put(f"{type(e).__name__}: {e}")
        except Exception:
            pass
        raise


def _warm(timeout: float) -> int:
    """One per worker at start-up: waits until EVERY worker has started, so a warm pool is a
    whole pool (a worker blocked here cannot take a second warm-up)."""
    if _BARRIER is not None:
        _BARRIER.wait(timeout)
    return os.getpid()


def analyse(path: str, disk_stat=None):
    """ONE file, in a worker: the parent processor's own analysis, and nothing it reached for.

    Any database reach during it (or a refused model load, under tests) makes the file
    `Outcome(internal=...)`, naming what was reached -- whatever the analysis returned, and
    whether or not it swallowed the refusal: never saved, new again next run (spec 4.3)."""
    from laser_trim_analyzer.core.processor import Outcome
    del _REACHES[:]
    try:
        outcome = _PROCESSOR.analyse_path(Path(path), disk_stat)
    except BaseException:
        if not _REACHES:
            raise
        outcome = None
    if _REACHES:
        reach = _REACHES[0]
        del _REACHES[:]
        return Outcome(path=str(path), internal=(
            f"it reached for {reach} inside a worker process -- a worker never opens a database, "
            f"so nothing it found was kept"))
    return outcome


# =============================================================================================
# the parent side

class _ToParentLoggers(logging.Handler):
    """A worker's record, handled by the PARENT's logger of the same name: the parent's own
    levels, handlers and propagation decide where it goes -- its one rotating log file among
    them."""

    def emit(self, record):
        try:
            name = record.name
            target = logging.getLogger() if name in ("root", "") else logging.getLogger(name)
            target.handle(record)
        except Exception:                  # never the listener thread's end: one record's loss
            self.handleError(record)


class _LogListener(QueueListener):
    """The pool's one listener. It stops without writing to its queue: a worker terminated while
    it held the queue's write lock (POSIX) would otherwise block the stop sentinel forever."""

    def __init__(self, log_queue):
        super().__init__(log_queue, _ToParentLoggers(), respect_handler_level=True)
        self._halt = threading.Event()

    def dequeue(self, block):
        while True:
            try:
                return self.queue.get(timeout=0.1)
            except _queue.Empty:
                if self._halt.is_set():
                    return self._sentinel
            except Exception:              # a record torn by a terminated worker: skip it
                if self._halt.is_set():
                    return self._sentinel

    def stop(self):
        if self._thread is not None:
            self._halt.set()
            self._thread.join()
            self._thread = None


def _parent_log_setup() -> Tuple[Dict[str, int], int]:
    """The parent's logging levels, for its workers to mirror: a record is created, or not, where
    it is logged -- in the worker."""
    levels = {"": logging.getLogger().level}
    for name, lg in list(logging.root.manager.loggerDict.items()):
        if isinstance(lg, logging.Logger) and lg.level != logging.NOTSET:
            levels[name] = lg.level
    return levels, logging.root.manager.disable


_LIVE: "weakref.WeakSet[WorkerPool]" = weakref.WeakSet()


class WorkerPool:
    """One folder run's worker processes (spec 4.5): started, warmed, fed files, closed.

    `lookahead = 2`: the dispatch keeps up to two 20-file chunks in flight, so the workers never
    wait on the consumer while it saves (spec 4.8); `in_process = False`: the parent parses
    nothing, so it needs no garbage collection between chunks."""

    lookahead = 2
    in_process = False

    def __init__(self, executor, listener, n: int, pids, ready_seconds: float):
        self._executor = executor
        self._listener = listener
        self._lock = threading.Lock()
        self.size = n
        self.pids = sorted(pids)
        self.ready_seconds = ready_seconds
        self.closed = False
        self.mode = f"{n} process{'es' if n != 1 else ''} (ready in {ready_seconds:.1f} s)"

    @classmethod
    def start(cls, ctx: WorkerContext, n: int, *, timeout: float = START_TIMEOUT) -> "WorkerPool":
        """Spawn `n` workers with `ctx` and warm every one of them, within `timeout` seconds.
        Raises PoolFailed with the cause -- the context could not be pickled, a worker's own
        initializer refused (its words), a worker died, or the time ran out -- and leaves no
        process behind."""
        t0 = time.monotonic()
        try:
            blob = pickle.dumps(ctx, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise PoolFailed(f"the worker context could not be sent to a process "
                             f"({type(e).__name__}: {e})") from e
        mp = multiprocessing.get_context("spawn")          # what Windows does, on every platform
        log_queue = mp.Queue()
        errors = mp.SimpleQueue()
        listener = _LogListener(log_queue)
        listener.start()
        executor = None
        try:
            executor = ProcessPoolExecutor(
                max_workers=n, mp_context=mp, initializer=_init_worker,
                initargs=(blob, log_queue, _parent_log_setup(), mp.Barrier(n), errors))
            futures = [executor.submit(_warm, timeout) for _ in range(n)]
            done, not_done = wait(futures, timeout=timeout)
            if not_done:
                raise PoolFailed(f"the {n} worker processes were not all up within {timeout:.0f} s")
            pids = [f.result() for f in futures]
        except BaseException as e:
            why = None
            try:
                if not errors.empty():
                    why = errors.get()
            except Exception:
                pass
            if executor is not None:
                _stop_executor(executor, grace=0.0)
            listener.stop()
            if isinstance(e, PoolFailed) and why is None:
                raise
            if not isinstance(e, Exception):
                raise
            raise PoolFailed(why or f"{type(e).__name__}: {e}") from e
        pool = cls(executor, listener, n, pids, time.monotonic() - t0)
        _LIVE.add(pool)
        return pool

    def submit(self, path, disk_stat=None):
        """One file to the pool; its Future's result is the file's Outcome."""
        return self._executor.submit(analyse, str(path), disk_stat)

    def call(self, fn, *args):
        """Any module-level function, in one worker (for the tests and the probe)."""
        return self._executor.submit(fn, *args)

    def close(self, grace: float = CLOSE_GRACE) -> int:
        """Stop the pool: nothing new starts, a busy worker gets `grace` seconds to finish its
        file, then it is terminated. Returns how many were terminated. Idempotent."""
        with self._lock:
            if self.closed:
                return 0
            self.closed = True
        try:
            return _stop_executor(self._executor, grace)
        finally:
            self._listener.stop()
            _LIVE.discard(self)


def _stop_executor(executor, grace: float) -> int:
    """shutdown(cancel_futures), `grace` seconds for the workers to exit, then terminate (and kill)
    the rest. The executor's own process table, copied first: shutdown drops it."""
    procs = list((getattr(executor, "_processes", None) or {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    deadline = time.monotonic() + max(0.0, grace)
    for p in procs:
        p.join(max(0.0, deadline - time.monotonic()))
    terminated = 0
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
                terminated += 1
        except Exception:
            pass
    for p in procs:
        try:
            p.join(2.0)
            if p.is_alive():
                p.kill()
                p.join(2.0)
        except Exception:
            pass
    return terminated


def close_worker_pools(grace: float = CLOSE_GRACE) -> int:
    """Close every live pool (the window's close, ruling 20; and the exit safety net below).
    Returns how many busy workers had to be terminated."""
    terminated = 0
    for pool in list(_LIVE):
        try:
            terminated += pool.close(grace)
        except Exception:
            logger.exception("Could not close the ingest's worker processes")
    return terminated


def _close_at_exit() -> None:
    if len(_LIVE):
        close_worker_pools()


# Before concurrent.futures' own exit hook: that one JOINS every worker process, so one stuck on a
# file would keep the interpreter from ever exiting. threading's exit hooks run in reverse order
# of registration, and concurrent.futures.process registered its own on import, above.
try:
    threading._register_atexit(_close_at_exit)
except Exception:                                   # pragma: no cover - interpreter shutting down
    import atexit
    atexit.register(_close_at_exit)
