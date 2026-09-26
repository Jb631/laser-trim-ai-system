"""Worker PROCESSES parse and analyse while one consumer saves (ingest-speed Task 11, A3; spec
section 4, rulings 13-19).

Threads share one GIL: four parser threads run at 1.0x (F7). Worker processes do not (F10: 6.2x
at 8), and since Task 9 the analysis is a pure function of the file -- `analyse_path` returns an
Outcome and touches no database -- so it can run in a spawned child while the parent's one
consumer saves every Outcome through the batch writer. What these tests pin:

  * a worker's Outcome is field for field the in-process one, for every kind of file the ingest
    meets (spec 4.9) -- compared as values, never as pickled bytes (F10: pydantic's fields-set
    pickles in each process's own hash order);
  * a worker NEVER opens a database (ruling 15): its initializer traps `get_database`,
    `DatabaseManager(...)` and `sqlite3.connect`, and a file during which any of them was reached
    comes back `internal` -- never a verdict, never a spec-less result -- named, and new again
    next run. The conftest guard does not reach a spawned child; this trap is its guard;
  * a worker's WARNING reaches the parent's log (ruling 18: one QueueListener in the parent);
  * a pool that cannot start, or breaks, finishes the folder on threads and says so (ruling 19);
  * Stop lands on a whole 20-file chunk, and close terminates a stuck worker after its grace and
    leaves every committed row intact (ruling 20);
  * processes only from 200 files left, and only for a Processor that carries its snapshot
    (ruling 13); how many follows ruling 14;
  * a worker scores with the trained models its PARENT resolved -- never its own folder -- and
    the forests predict on one thread (controller ruling: n_jobs = 1 at load).

The process path is paid for a few times per file, not per test: two module-scoped 2-worker
pools. Every stub a worker builds is in tests/worker_stubs.py -- a child cannot be monkeypatched.
"""
import dataclasses
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

import save_rows
import v5_loop
import worker_stubs

REPO = Path(__file__).resolve().parents[1]
TRIMS = save_rows.FIXTURES / "trim"

# Invented specs for the fixtures' own models (values invented), shaped as get_model_spec returns
# them, so the analysis is spec-dependent -- a snapshot that did not reach the worker would show.
_SPEC_KEYS = ("element_type", "product_class", "linearity_type", "linearity_spec_text",
              "linearity_spec_pct", "total_resistance_min", "total_resistance_max",
              "electrical_angle", "electrical_angle_tol", "electrical_angle_tol_type",
              "electrical_angle_unit", "output_smoothness", "circuit_type", "open_closed",
              "aliases", "exclude_points", "exclude_points_ft", "notes")
_INVENTED = [
    {"model": "8232-1", "linearity_type": "independent", "electrical_angle": 340.0,
     "electrical_angle_tol": 5.0, "electrical_angle_tol_type": "bilateral"},
    {"model": "8074", "linearity_type": "absolute", "exclude_points": "1-2"},
    {"model": "7553", "linearity_type": "terminal", "exclude_points_ft": "3"},
    {"model": "7458", "linearity_type": "independent", "aliases": "7458X"},
    {"model": "7539-2", "linearity_type": "absolute"},
    {"model": "8639-30", "linearity_type": "independent"},
    {"model": "8340-1", "linearity_type": "terminal"},
]


def _snapshot():
    from laser_trim_analyzer.database.specs import SpecSnapshot
    rows = []
    for i, spec in enumerate(_INVENTED, start=1):
        row = {"id": i, "model": spec["model"]}
        row.update({k: spec.get(k) for k in _SPEC_KEYS})
        rows.append(row)
    return SpecSnapshot(specs=tuple(rows))


def _parallel_config():
    """A config whose folders take the pool path however few files they hold."""
    from laser_trim_analyzer.config import Config
    cfg = Config()
    cfg.processing.turbo_mode_threshold = 1
    return cfg


def _stat(path):
    st = os.stat(path)
    return (st.st_size, st.st_mtime)


def _plain(x):
    """Plain values: what two processes' objects are compared as (never their pickled bytes)."""
    import datetime as _dt
    import enum
    import numpy as np
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if hasattr(x, "model_dump"):
        return _plain(x.model_dump())
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (_dt.datetime, _dt.date)):
        return x.isoformat()
    if isinstance(x, enum.Enum):
        return f"{type(x).__name__}.{x.name}"
    if isinstance(x, Path):
        return str(x)
    return x


def _comparable(outcome):
    """An Outcome as values, minus what differs by the CLOCK alone: when the analysis started, how
    long it took, and the `datetime.now()` an ERROR result's minimal metadata is dated with."""
    d = _plain(outcome)
    d.pop("started", None)
    result = d.get("result")
    if result:
        result["processing_time"] = 0.0
        if result["metadata"]["model"] == "Unknown":
            result["metadata"]["file_date"] = "<now>"
    return d


class _Collect:
    """A writer that keeps every Outcome it is handed (the dispatch is what is under test)."""

    def __init__(self):
        self.outcomes = []

    def add(self, outcome):
        self.outcomes.append(outcome)
        return outcome.result

    def tick(self):
        pass

    def flush(self):
        pass


@pytest.fixture
def processes(monkeypatch):
    """Worker processes for any folder in this test, however small: two of them."""
    from laser_trim_analyzer.core import ingest_worker
    monkeypatch.setattr(ingest_worker, "PROCESS_MIN_FILES", 1)
    monkeypatch.setattr(ingest_worker, "MAX_WORKERS", 2)
    return ingest_worker


@pytest.fixture
def db(tmp_path, monkeypatch):
    from laser_trim_analyzer.database.manager import DatabaseManager
    d = DatabaseManager(tmp_path / "run.db")
    save_rows.inject(d, monkeypatch)
    yield d
    d.close()


# ---- two pools for the whole module (spec 4.9: the process path is paid for once) -------------

@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    root = tmp_path_factory.mktemp("scenario")
    return root, v5_loop.build_v5_scenario(root)


def _start(proc, n=2, **replace):
    """A pool for `proc`, told (as every test's is, by conftest) which folders hold a checkout's
    real models -- a module-scoped pool is built before any test's own fixtures run."""
    from laser_trim_analyzer.core.ingest_worker import WorkerPool, context_for
    replace.setdefault("refuse_models_under", worker_stubs.checkout_data_dirs())
    return WorkerPool.start(dataclasses.replace(context_for(proc), **replace), n)


@pytest.fixture(scope="module")
def real_pool(tmp_path_factory):
    """The REAL Processor, carrying the invented snapshot, in two spawned workers."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.processor import Processor
    models = tmp_path_factory.mktemp("no_models")
    proc = Processor(config=Config(), use_ml=False, snapshot=_snapshot(), ml_storage_path=models)
    pool = _start(proc)
    yield proc, pool
    pool.close()


@pytest.fixture(scope="module")
def stub_pool():
    """The file-name-driven stub (worker_stubs.StubProcessor) in two spawned workers."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.database.specs import SpecSnapshot
    pool = _start(worker_stubs.StubProcessor(config=Config(), snapshot=SpecSnapshot()))
    yield pool
    pool.close()


# ---- the worker's outcome IS the analysis's --------------------------------------------------

def test_a_worker_returns_exactly_what_the_analysis_returns_in_this_process(scenario, real_pool):
    """Every kind of file the ingest meets -- trims of both lasers, a TrimVolts and a trim-setup
    file, a two-track file, a no-cut template, every final-test format, smoothness, not-test-data,
    and every failure -- analysed in a spawned worker is, field by field, what the same Processor
    returns in this process: the result, every write it asks for, and the (size, mtime) and hash
    it read. And the snapshot was in force there: analysed spec-less, the outcomes differ."""
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database.specs import SpecSnapshot
    root, paths = scenario
    proc, pool = real_pool
    assert len(set(pool.pids)) == 2 and os.getpid() not in pool.pids, pool.pids
    futures = [pool.submit(p, _stat(p)) for p in paths]
    in_workers = [_comparable(f.result(timeout=180)) for f in futures]
    here = [_comparable(proc.analyse_path(p, _stat(p))) for p in paths]
    diffs = save_rows.differences(in_workers, here, exact=True)
    assert not diffs, "\n".join(diffs)
    kinds = {(o["result"] or {}).get("file_type", "none") for o in in_workers}
    assert {"trim", "final_test", "smoothness", "none"} <= kinds, kinds
    assert all(o["internal"] is None for o in in_workers), "a worker reached for a database"
    bare = Processor(config=proc.config, use_ml=False, snapshot=SpecSnapshot(),
                     ml_storage_path=proc.ml_storage_path)
    spec_less = [_comparable(bare.analyse_path(p, _stat(p))) for p in paths]
    assert save_rows.differences(spec_less, in_workers, exact=True), (
        "the invented specs moved nothing -- this test could not see a snapshot lost on the way")


def test_a_worker_runs_the_parents_own_source_tree_and_logs_through_one_queue(real_pool):
    """A spawned child gets its parent's sys.path, but an editable install also puts the MAIN
    checkout's src on it: the worker must import the tree the parent did (a worktree's, under
    PYTHONPATH), or its numbers come from other code. And its logging is ONE QueueHandler to the
    parent -- the inherited handlers dropped (spec 4.7)."""
    _, pool = real_pool
    assert pool.call(worker_stubs.package_file).result(timeout=60) == worker_stubs.package_file()
    assert pool.call(worker_stubs.root_handlers).result(timeout=60) == ["QueueHandler"]


# ---- the database trap (ruling 15, spec 4.3) --------------------------------------------------

@pytest.mark.parametrize("reach, named", [("reach_get_database", "get_database"),
                                          ("reach_manager_get_database", "get_database()"),
                                          ("reach_manager", "DatabaseManager"),
                                          ("reach_sqlite", "sqlite3.connect"),
                                          ("reach_dbapi2", "sqlite3.dbapi2.connect")])
def test_a_worker_that_reaches_for_a_database_returns_internal_never_a_verdict(
        stub_pool, tmp_path, reach, named):
    """The stub reaches for a database and SWALLOWS the refusal, as the real analysis's `except
    Exception` would, then returns its result anyway. The worker must return `internal`, naming
    what was reached and from where -- never that result -- and nothing was opened or created."""
    folder = tmp_path / "files"
    f = worker_stubs.make_files(folder, [f"{reach}_1.xls"])[0]
    outcome = stub_pool.submit(f).result(timeout=60)
    assert outcome.internal is not None, "the worker returned the analysis's verdict"
    assert outcome.result is None and outcome.writes == ()
    assert named in outcome.internal and "worker_stubs.py" in outcome.internal, outcome.internal
    assert sorted(p.name for p in folder.iterdir()) == [f.name], "a database file was created"


def test_the_real_analysis_without_its_snapshot_is_internal_in_a_worker_never_spec_less(tmp_path):
    """The shape the trap exists for (spec 4.3): the real analysis looks a spec up per file, and
    without a snapshot it asks get_database(). Unguarded, a worker's refusal would be swallowed at
    DEBUG and the file analysed spec-less -- different stored numbers, without a word."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.processor import Processor
    proc = Processor(config=Config(), use_ml=False, ml_storage_path=tmp_path / "no_models")
    pool = _start(proc, 1)
    try:
        f = TRIMS / "dlts_8232-1_242.xls"
        outcome = pool.submit(f, _stat(f)).result(timeout=120)
    finally:
        pool.close()
    assert outcome.internal is not None and outcome.result is None
    assert "get_database" in outcome.internal and "_get_spec_for_analysis" in outcome.internal


def test_a_processor_that_reaches_for_a_database_while_it_is_built_never_starts(tmp_path):
    """A worker whose processor opens a database as it is BUILT refuses to start, by name --
    the folder then runs on threads -- rather than analyse anything in that state."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.ingest_worker import PoolFailed
    from laser_trim_analyzer.database.specs import SpecSnapshot
    with pytest.raises(PoolFailed) as refused:
        _start(worker_stubs.LoudInitProcessor(config=Config(), snapshot=SpecSnapshot()), 1)
    assert "get_database" in str(refused.value), refused.value


# ---- the refusal of the real models is in place before any fixture runs (review m-7) ---------

@pytest.fixture(scope="module")
def refusal_at_module_scope():
    """What a module-scoped fixture -- built BEFORE any test's own fixtures run -- would hand a
    worker: the folders to refuse models from, and a context's copy of them."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core import ingest_worker
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database.specs import SpecSnapshot
    ctx = ingest_worker.context_for(Processor(config=Config(), use_ml=False, snapshot=SpecSnapshot(),
                                              ml_storage_path=REPO / "no_models_here"))
    return tuple(ingest_worker.REFUSE_MODELS_UNDER), tuple(ctx.refuse_models_under)


def test_a_module_scoped_pool_carries_the_refusal_of_the_real_models(refusal_at_module_scope):
    """conftest sets it when it is IMPORTED, not only per test: a module-scoped fixture runs before
    the per-test ones, and a pool it built would otherwise carry no refusal at all."""
    module_level, in_a_context = refusal_at_module_scope
    assert set(module_level) == set(in_a_context) == set(worker_stubs.checkout_data_dirs())


# ---- logging (ruling 18) ---------------------------------------------------------------------

def test_a_worker_warning_reaches_the_parents_log(stub_pool, tmp_path, caplog):
    f = worker_stubs.make_files(tmp_path, ["warn_1.xls"])[0]
    with caplog.at_level(logging.WARNING):
        stub_pool.submit(f).result(timeout=60)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not [
                r for r in caplog.records if "invented warning for warn_1.xls" in r.getMessage()]:
            time.sleep(0.05)
    got = [r for r in caplog.records if "invented warning for warn_1.xls" in r.getMessage()]
    assert len(got) == 1, [r.getMessage() for r in caplog.records]
    assert got[0].levelno == logging.WARNING
    assert got[0].name == "laser_trim_analyzer.tests.worker_stubs"
    assert got[0].process != os.getpid() and got[0].process in stub_pool.pids


# ---- which pool, and the fallback (rulings 13, 14, 19) ---------------------------------------

@pytest.mark.parametrize("cpus, free_gb, want", [(14, 20.0, 8), (4, 20.0, 3), (14, 6.2, 4),
                                                 (14, 4.4, 0), (1, 20.0, 0), (16, 5.0, 2)])
def test_how_many_worker_processes(cpus, free_gb, want):
    """Ruling 14: min(8, CPUs - 1, floor((free GB - 4) / 0.5)) -- and none, said why, below one."""
    from laser_trim_analyzer.core.ingest_worker import worker_count
    n, why = worker_count(cpus=cpus, free_gb=free_gb)
    assert n == want
    if n < 1:
        assert why, "no worker processes, and no reason given"


def test_fewer_than_200_files_left_never_start_a_process_and_200_do(tmp_path, monkeypatch):
    """Ruling 13, the real threshold: 199 files left stay on threads without a pool being asked
    for; 200 ask for one."""
    from laser_trim_analyzer.core import ingest_worker
    from laser_trim_analyzer.database.specs import SpecSnapshot
    asked = []

    def recorded(cls, ctx, n, **k):
        asked.append(n)
        raise ingest_worker.PoolFailed("recorded here, never started")

    monkeypatch.setattr(ingest_worker.WorkerPool, "start", classmethod(recorded))
    proc = worker_stubs.StubProcessor(config=_parallel_config(), snapshot=SpecSnapshot())
    few = [tmp_path / f"f{i:03d}.xls" for i in range(199)]
    assert len(list(proc.process_batch(few, incremental=False, writer=_Collect()))) == 199
    assert asked == [] and "threads" in proc.last_workers and "199 files" in proc.last_workers
    enough = few + [tmp_path / "f199.xls"]
    assert len(list(proc.process_batch(enough, incremental=False, writer=_Collect()))) == 200
    assert len(asked) == 1


def test_a_processor_without_its_snapshot_never_starts_a_process(tmp_path, processes,
                                                                 monkeypatch):
    """V5's loop and the scripts build a Processor with no snapshot: its analysis asks the
    database, which a worker may never open -- so it stays on threads, and says why."""
    asked = []
    monkeypatch.setattr(processes.WorkerPool, "start",
                        classmethod(lambda cls, *a, **k: asked.append(1)))
    proc = worker_stubs.StubProcessor(config=_parallel_config())
    files = [tmp_path / f"f{i:03d}.xls" for i in range(30)]
    assert len(list(proc.process_batch(files, incremental=False, writer=_Collect()))) == 30
    assert asked == [] and "no spec snapshot" in proc.last_workers


def test_a_pool_that_cannot_start_runs_the_folder_on_threads_and_says_why(
        tmp_path, processes, monkeypatch, caplog):
    from laser_trim_analyzer.database.specs import SpecSnapshot

    def refused(cls, ctx, n, **k):
        raise processes.PoolFailed("invented: the endpoint scanner said no")

    monkeypatch.setattr(processes.WorkerPool, "start", classmethod(refused))
    proc = worker_stubs.StubProcessor(config=_parallel_config(), snapshot=SpecSnapshot())
    files = worker_stubs.make_files(tmp_path, [f"f{i:03d}.xls" for i in range(30)])
    writer = _Collect()
    with caplog.at_level(logging.WARNING):
        got = list(proc.process_batch(files, incremental=False, writer=writer))
    assert len(got) == 30 and len(writer.outcomes) == 30
    assert "threads (processes could not start: invented: the endpoint scanner said no)" \
        in proc.last_workers, proc.last_workers
    assert any(r.levelno == logging.WARNING and "invented: the endpoint scanner said no"
               in r.getMessage() for r in caplog.records)


def test_a_worker_that_imported_another_source_tree_refuses_to_start_and_names_both(tmp_path):
    """A REAL start failure, in the child's own initializer: the reason it gives is the one the
    batch line will carry, not a generic 'the pool is broken'."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.ingest_worker import PoolFailed
    from laser_trim_analyzer.database.specs import SpecSnapshot
    proc = worker_stubs.StubProcessor(config=Config(), snapshot=SpecSnapshot())
    with pytest.raises(PoolFailed) as refused:
        _start(proc, 1, package_file="/nowhere/else/laser_trim_analyzer/__init__.py")
    assert "/nowhere/else" in str(refused.value), refused.value
    assert worker_stubs.package_file() in str(refused.value), refused.value


def test_a_context_that_cannot_be_sent_to_a_worker_is_refused_before_any_process_starts():
    """Spawn pickles what a worker is handed: a processor class a child cannot import is refused
    at once, by name, in the parent -- the folder then runs on threads."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.ingest_worker import PoolFailed
    from laser_trim_analyzer.database.specs import SpecSnapshot

    class NotImportable(worker_stubs.StubProcessor):
        pass

    t0 = time.monotonic()
    with pytest.raises(PoolFailed) as refused:
        _start(NotImportable(config=Config(), snapshot=SpecSnapshot()), 1)
    assert "could not be sent" in str(refused.value) and time.monotonic() - t0 < 5


def test_a_pool_that_breaks_mid_run_finishes_the_folder_on_threads(tmp_path, processes, caplog):
    """A worker killed mid-run (os._exit) breaks the pool: its in-flight files and the rest of the
    folder run on threads -- every file handed over exactly once -- and the mode says so. The files
    are slow, so the pool breaks with a whole window in flight, not only the killed file."""
    import re
    from laser_trim_analyzer.database.specs import SpecSnapshot
    names = ([f"slow_a{i:03d}.xls" for i in range(25)] + ["die_1.xls"]
             + [f"slow_b{i:03d}.xls" for i in range(34)])
    files = worker_stubs.make_files(tmp_path, names)
    proc = worker_stubs.StubProcessor(config=_parallel_config(), snapshot=SpecSnapshot())
    writer = _Collect()
    with caplog.at_level(logging.WARNING):
        got = list(proc.process_batch(files, incremental=False, writer=writer))
    assert sorted(o.path for o in writer.outcomes) == sorted(str(f) for f in files)
    assert len(got) == len(files)
    assert proc.last_workers.startswith("2 processes"), proc.last_workers
    assert ", then " in proc.last_workers and " (worker processes broke after " in proc.last_workers
    assert "thread" in proc.last_workers.split(", then ")[1], proc.last_workers
    broke_after = int(re.search(r"broke after (\d+) files", proc.last_workers).group(1))
    assert broke_after < 40, ("the pool broke with little else in flight -- this test could not "
                              f"see in-flight files lost ({proc.last_workers})")
    assert any(r.levelno == logging.WARNING and "broke" in r.getMessage() for r in caplog.records)


def _out_and_in(monkeypatch, pool_class, proc, files):
    """The order files went OUT to the pool and came back IN to the writer."""
    events = []
    real_submit = pool_class.submit

    def submit(self, path, disk_stat=None):
        events.append("out")
        return real_submit(self, path, disk_stat)

    monkeypatch.setattr(pool_class, "submit", submit)

    class _Log(_Collect):
        def add(self, outcome):
            events.append("in")
            return super().add(outcome)

    assert len(list(proc.process_batch(files, incremental=False, writer=_Log()))) == len(files)
    outs = [i for i, e in enumerate(events) if e == "out"]
    ins = [i for i, e in enumerate(events) if e == "in"]
    most = max(sum(1 for o in outs if o <= i) - sum(1 for x in ins if x <= i)
               for i in range(len(events)))
    return outs, ins, most


def test_worker_processes_get_the_next_chunk_before_the_last_one_is_back(tmp_path, processes,
                                                                         monkeypatch):
    """Spec 4.8: no barrier per batch in process mode -- up to two 20-file chunks in flight, so a
    worker never waits while the consumer saves -- and never more than two. Threads keep the
    batch boundary they always had: the next chunk only once the last is all back."""
    from laser_trim_analyzer.core.processor import _ThreadPool
    from laser_trim_analyzer.database.specs import SpecSnapshot
    files = worker_stubs.make_files(tmp_path, [f"slow_{i:03d}.xls" for i in range(60)])
    proc = worker_stubs.StubProcessor(config=_parallel_config(), snapshot=SpecSnapshot())
    outs, ins, most = _out_and_in(monkeypatch, processes.WorkerPool, proc, files)
    assert proc.last_workers.startswith("2 processes"), proc.last_workers
    assert outs[20] < ins[19], "the 21st file went out only once the first 20 were back: a barrier"
    assert most == 40, f"at most two chunks in flight, and two: {most}"
    monkeypatch.setattr(processes, "PROCESS_MIN_FILES", 10 ** 9)
    outs, ins, most = _out_and_in(monkeypatch, _ThreadPool, proc, files)
    assert "threads" in proc.last_workers, proc.last_workers
    assert outs[20] > ins[19] and most == 20, (outs[20], ins[19], most)


def test_while_the_worker_processes_start_the_progress_line_says_so(tmp_path, processes):
    """Starting the workers can take a while at work (an endpoint scanner reads every imported
    module, spec 4.5): the progress line says what is happening instead of going quiet."""
    from laser_trim_analyzer.database.specs import SpecSnapshot
    files = worker_stubs.make_files(tmp_path, [f"f{i:03d}.xls" for i in range(30)])
    proc = worker_stubs.StubProcessor(config=_parallel_config(), snapshot=SpecSnapshot())
    said = []
    got = list(proc.process_batch(files, progress_callback=lambda st: said.append(st),
                                  incremental=False, writer=_Collect()))
    assert len(got) == 30 and proc.last_workers.startswith("2 processes")
    starting = [st.message for st in said if st.status == "scanning"
                and "worker processes" in (st.message or "")]
    assert starting == ["Starting 2 worker processes for 30 files…"], starting
    first_done = next(i for i, st in enumerate(said) if st.status == "completed")
    assert said.index(next(st for st in said if st.message in starting)) < first_done


# ---- Stop and close (ruling 20) --------------------------------------------------------------

def test_stop_in_process_mode_lands_on_a_whole_chunk_and_every_file_handed_out_arrives(
        tmp_path, processes):
    from laser_trim_analyzer.database.specs import SpecSnapshot
    files = worker_stubs.make_files(tmp_path, [f"slow_{i:03d}.xls" for i in range(120)])
    proc = worker_stubs.StubProcessor(config=_parallel_config(), snapshot=SpecSnapshot())
    cancel, seen, writer = threading.Event(), [], _Collect()

    def trip(status):
        if status.status in ("completed", "skipped", "failed"):
            seen.append(status.filename)
            if len(seen) >= 3:
                cancel.set()

    got = list(proc.process_batch(files, progress_callback=trip, incremental=False,
                                  cancel=cancel, writer=writer))
    assert "processes" in proc.last_workers, proc.last_workers
    assert len(got) % 20 == 0 and 20 <= len(got) < 120, len(got)
    assert len(writer.outcomes) == len(got) == len(seen)


def test_close_terminates_a_stuck_worker_and_leaves_the_committed_rows_intact(
        tmp_path, processes, db, monkeypatch):
    """The window closes while a worker is stuck on one file: after its grace the worker is
    terminated -- it holds no database handle, so nothing can be torn -- the run ends, and every
    row that had committed is still there, whole. The stuck file stored nothing: new next run."""
    import weakref
    import psutil
    from laser_trim_analyzer.core.ingest_run import BatchWriter
    from laser_trim_analyzer.database.specs import SpecSnapshot
    # A registry of this test's own: the module's shared pools are live too, and closing the
    # window closes EVERY live pool.
    monkeypatch.setattr(processes, "_LIVE", weakref.WeakSet())
    names = ([f"a{i:03d}.xls" for i in range(20)] + ["stuck_1.xls"]
             + [f"b{i:03d}.xls" for i in range(19)])
    files = worker_stubs.make_files(tmp_path / "in", names)
    proc = worker_stubs.StubProcessor(config=_parallel_config(), snapshot=SpecSnapshot())
    committed = []
    writer = BatchWriter(proc, db, committed.append)
    cancel, ended = threading.Event(), threading.Event()

    def run():
        try:
            list(proc.process_batch(files, incremental=False, cancel=cancel, writer=writer))
        finally:
            ended.set()

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and len(committed) < 20:
        time.sleep(0.05)
    assert len(committed) >= 20, "the first chunk never committed"
    pools = list(processes._LIVE)
    assert len(pools) == 1
    pids = pools[0].pids
    before = _rows(db)
    cancel.set()
    t0 = time.monotonic()
    terminated = processes.close_worker_pools(grace=0.5)
    assert ended.wait(30), "the run did not end once its pool was closed"
    assert time.monotonic() - t0 < 15
    assert terminated >= 1, "no busy worker was terminated -- the stuck file was not in flight?"
    for pid in pids:
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    after = _rows(db)
    assert before and set(before) <= set(after), "a committed row went missing"
    assert "stuck_1.xls" not in after
    con = sqlite3.connect(f"file:{db.database_path}?mode=ro", uri=True)
    try:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        con.close()


# ---- a parent that dies, a start that is stopped, and the interpreter's exit ------------------

_PARENT_SCRIPT = """
import json, sys, time
from pathlib import Path

if __name__ == "__main__":
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core import ingest_worker
    from laser_trim_analyzer.core.ingest_worker import WorkerPool, context_for
    from laser_trim_analyzer.database.specs import SpecSnapshot
    import worker_stubs
    ingest_worker.CLOSE_GRACE = 0.5
    proc = worker_stubs.StubProcessor(config=Config(), snapshot=SpecSnapshot(),
                                      ml_storage_path=Path({models!r}))
    pool = WorkerPool.start(context_for(proc), {n})
    files = worker_stubs.make_files(Path({folder!r}), {names!r})
    pool.submit(files[0]).result(timeout=60)          # a file analysed: the workers are working
    for f in files[1:]:
        pool.submit(f)                                # (the stuck one never finishes)
    time.sleep(1.0)
    print(json.dumps(pool.pids), flush=True)
    {then}
"""


def _parent(tmp_path, *, n, names, then):
    """A parent process of its own with a pool of `n` workers that analysed a file; returns it
    and the worker pids it printed."""
    import json
    import subprocess
    import sys
    script = tmp_path / "parent.py"
    script.write_text(_PARENT_SCRIPT.format(models=str(tmp_path / "no_models"), n=n,
                                            folder=str(tmp_path / "files"), names=names,
                                            then=then))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO / "tests")]))
    parent = subprocess.Popen([sys.executable, "-B", str(script)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env=env, cwd=tmp_path)
    got = []
    reader = threading.Thread(target=lambda: got.append(parent.stdout.readline()), daemon=True)
    reader.start()
    reader.join(120)
    if not got or not got[0].strip():
        parent.kill()
        raise AssertionError(f"the parent printed no worker pids: {parent.stderr.read()[-2000:]}")
    return parent, json.loads(got[0])


def _alive(pids):
    import psutil
    out = []
    for pid in pids:
        try:
            if psutil.Process(pid).status() != psutil.STATUS_ZOMBIE:
                out.append(pid)
        except psutil.Error:
            pass
    return out


def _gone_within(pids, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and _alive(pids):
        time.sleep(0.2)
    return _alive(pids)


def _kill(parent, pids):
    import psutil
    if parent.poll() is None:
        parent.kill()
    for pid in _alive(pids):
        try:
            psutil.Process(pid).kill()
        except psutil.Error:
            pass


def test_worker_processes_die_with_a_parent_that_is_killed(tmp_path):
    """Review I-1: the parent is killed outright -- SIGKILL here, "End task" or a crash at work --
    so nothing of its own runs: no close, no exit hook. Its idle workers must not live on
    (7-8 hidden python processes per folder, ~0.3-0.5 GB each, until a reboot): each worker
    watches its parent and exits with it."""
    parent, pids = _parent(tmp_path, n=2, names=["plain_1.xls"], then="time.sleep(600)")
    try:
        assert len(pids) == 2 and len(_alive(pids)) == 2
        parent.kill()
        parent.wait(10)
        left = _gone_within(pids, 15)
        assert not left, f"worker processes outlived their killed parent: {left}"
    finally:
        _kill(parent, pids)


def test_the_interpreter_exits_promptly_with_a_busy_worker(tmp_path):
    """Review m-3: the parent's main code returns while a worker is busy on a file and nothing
    closed the pool. concurrent.futures joins every worker process at interpreter exit, so without
    ingest_worker's own exit hook -- which closes live pools first, a grace and then terminate --
    the process would wait for that file however long it takes."""
    parent, pids = _parent(tmp_path, n=1, names=["plain_1.xls", "stuck_1.xls"], then="pass")
    try:
        t0 = time.monotonic()
        parent.wait(30)
        assert time.monotonic() - t0 < 30
        assert not _gone_within(pids, 10), "the busy worker outlived the exit"
    finally:
        _kill(parent, pids)


def _slow_start_context(tmp_path):
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.ingest_worker import context_for
    from laser_trim_analyzer.database.specs import SpecSnapshot
    proc = worker_stubs.SlowStartProcessor(config=Config(), snapshot=SpecSnapshot(),
                                          ml_storage_path=tmp_path / "no_models")
    return context_for(proc)


def test_stop_while_the_worker_processes_start_is_honoured_at_once(tmp_path, processes):
    """Review m-1: Stop pressed while the workers are still starting (20 s here, as an endpoint
    scanner can make it at work) ends the start then, not when it would have finished -- and no
    worker is left behind."""
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(processes.PoolFailed, match="stopped") as stopped:
        processes.WorkerPool.start(_slow_start_context(tmp_path), 2, cancel=cancel)
    assert time.monotonic() - t0 < 8, "the start ran on after Stop"
    assert isinstance(stopped.value, processes.PoolStopped)


def test_closing_the_window_while_the_worker_processes_start_is_honoured_at_once(
        tmp_path, processes, monkeypatch):
    """Review m-1: the window closes while the workers are still starting -- the pool is already
    registered for closing, so the close reaches it and the start ends then."""
    import weakref
    monkeypatch.setattr(processes, "_LIVE", weakref.WeakSet())
    threading.Timer(0.5, lambda: processes.close_worker_pools(grace=0.2)).start()
    t0 = time.monotonic()
    with pytest.raises(processes.PoolFailed, match="closed"):
        processes.WorkerPool.start(_slow_start_context(tmp_path), 2)
    assert time.monotonic() - t0 < 8, "the start ran on after the window closed"
    assert len(processes._LIVE) == 0


def test_a_folder_stopped_while_its_worker_processes_start_analyses_nothing(tmp_path, processes):
    from laser_trim_analyzer.database.specs import SpecSnapshot
    proc = worker_stubs.SlowStartProcessor(config=_parallel_config(), snapshot=SpecSnapshot(),
                                          ml_storage_path=tmp_path / "no_models")
    files = worker_stubs.make_files(tmp_path / "in", [f"f{i:03d}.xls" for i in range(30)])
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    t0 = time.monotonic()
    got = list(proc.process_batch(files, incremental=False, cancel=cancel, writer=_Collect()))
    assert got == [] and time.monotonic() - t0 < 10
    assert "stopped while the worker processes were starting" in proc.last_workers, \
        proc.last_workers


# ---- a worker pool that keeps reaching for a database (review m-6) ----------------------------

def test_worker_processes_that_keep_reaching_for_a_database_hand_the_folder_to_threads(
        tmp_path, processes, db, caplog):
    """A regression that makes the analysis reach for a database would turn every file of a
    170,000-file run into an `internal` error in worker processes -- the folder parsed into
    nothing. After one chunk's worth of such files in a row the rest of the folder goes to
    threads, where the analysis may open the database, and the batch line says why."""
    from laser_trim_analyzer.database.specs import SpecSnapshot
    files = worker_stubs.make_files(tmp_path / "in", [f"reach_get_database_{i:03d}.xls"
                                                     for i in range(60)])
    proc = worker_stubs.StubProcessor(config=_parallel_config(), snapshot=SpecSnapshot(),
                                      ml_storage_path=tmp_path / "no_models")
    writer = _Collect()
    with caplog.at_level(logging.WARNING):
        got = list(proc.process_batch(files, incremental=False, writer=writer))
    internal = [o for o in writer.outcomes if o.internal]
    assert len(internal) == 20, len(internal)
    assert len(writer.outcomes) == 60 and len(got) == 40
    assert proc.last_workers.startswith("2 processes"), proc.last_workers
    assert "then" in proc.last_workers and "reached for a database" in proc.last_workers, \
        proc.last_workers
    assert any(r.levelno == logging.WARNING and "reached for a database" in r.getMessage()
               for r in caplog.records)


def _rows(db):
    con = sqlite3.connect(f"file:{db.database_path}?mode=ro", uri=True)
    try:
        return sorted(r[0] for r in con.execute("SELECT filename FROM analysis_results"))
    finally:
        con.close()


# ---- the whole ingest, in process mode ---------------------------------------------------------

def test_run_folder_in_process_mode_stores_exactly_what_v5s_loop_stored(db, tmp_path, processes):
    """run_folder with worker processes over the whole V5 scenario stores exactly the rows V5's
    own loop stored (the committed golden the thread mode is held to), and counts exactly the
    thread mode's buckets."""
    from laser_trim_analyzer.core.ingest_run import run_folder
    v5_loop.build_v5_scenario(tmp_path)
    res = run_folder(str(tmp_path / "in"), db=db, config=_parallel_config(), incremental=True)
    assert res.ok, res.error
    assert res.workers.startswith("2 processes"), res.workers
    got = v5_loop.v5_snapshot(db.database_path, tmp_path, [])["tables"]
    want = save_rows.load_golden(v5_loop.GOLDEN)["tables"]
    diffs = save_rows.differences(v5_loop.order_free(got), v5_loop.order_free(want))
    assert not diffs, "\n".join(diffs)
    assert res.buckets == {"failed": 5, "warnings": 3, "passed": 6, "errors": 8}, res.buckets
    assert res.new_trims == 10


def test_internal_outcomes_are_errors_named_counted_and_new_again_next_run(
        db, tmp_path, processes, monkeypatch, caplog):
    """A file whose analysis reached for a database in a worker is logged at ERROR by the parent
    with its reason, counted as an error, stores nothing, and the batch line counts it (spec
    4.3); the rest of the folder is saved."""
    from laser_trim_analyzer.core import ingest_run
    monkeypatch.setattr(ingest_run, "Processor", worker_stubs.StubProcessor)
    monkeypatch.setattr(ingest_run, "_post_batch", lambda *a, **k: None)
    folder = tmp_path / "in"
    worker_stubs.make_files(folder, ["reach_get_database_1.xls", "reach_sqlite_2.xls",
                                     "plain_1.xls", "plain_2.xls"])
    with caplog.at_level(logging.INFO):
        res = ingest_run.run_folder(str(folder), db=db, config=_parallel_config(),
                                    incremental=True)
    assert res.ok, res.error
    assert res.workers.startswith("2 processes"), res.workers
    assert res.unsaved == 2 and res.buckets.get("errors") == 4, (res.unsaved, res.buckets)
    assert _rows(db) == ["plain_1.xls", "plain_2.xls"]
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    for name, what in (("reach_get_database_1.xls", "get_database"),
                       ("reach_sqlite_2.xls", "sqlite3.connect")):
        assert any(name in m and what in m for m in errors), errors
    line = next(r.getMessage() for r in caplog.records
                if r.getMessage().startswith("Batch phases: walk"))
    assert "2 file(s) not analysed" in line and "workers 2 processes" in line, line


# ---- trained models (the Task 9-10 hand-off; controller ruling on n_jobs) ---------------------

def test_with_trained_models_process_mode_stores_what_thread_mode_stores(
        tmp_path, processes, monkeypatch):
    """Trained models PRESENT -- a deployed composite trim-risk model, a failure predictor and a
    sigma threshold (invented, in the redirected app directory's models folder): run_folder in
    worker processes stores exactly what run_folder on threads stores, every ML column included.
    The predictor reached the workers in the snapshot and the composite model loaded there from
    the folder the PARENT resolved -- a spawned child's own resolution would be the real app
    directory, which the conftest redirect never reaches. And the models were in force: without
    the predictor, the stored probabilities differ."""
    import shutil
    from laser_trim_analyzer.config import ml_models_directory
    from laser_trim_analyzer.core.ingest_run import run_folder
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml import invalidate_shared_ml_manager
    from test_batch_writer import _invented_trained_models_for_8232_1
    folder = tmp_path / "laser"
    for n in ("dlts_8232-1_242.xls", "dlts_8232-1_243.xls", "lts_8232-1_193.xls",
              "lts_8232-1_194.xls", "dlts_8074_18.xls"):
        save_rows._pinned_copy(TRIMS / n, folder / n)
    got = {}
    for label in ("processes", "threads", "no_predictor"):
        d = DatabaseManager(tmp_path / f"{label}.db")
        try:
            if label == "processes":
                _invented_trained_models_for_8232_1(d)
            else:
                from laser_trim_analyzer.database.models import ModelMLState
                with d.session() as s:
                    s.add(ModelMLState(model="8232-1", is_trained=True, sigma_threshold=0.0042))
            if label == "no_predictor":
                shutil.rmtree(ml_models_directory() / "predictors")
            monkeypatch.setattr(processes, "PROCESS_MIN_FILES", 1 if label == "processes" else 10 ** 9)
            invalidate_shared_ml_manager()
            save_rows.inject(d, monkeypatch)
            res = run_folder(str(folder), db=d, config=_parallel_config(), incremental=True)
            assert res.ok, res.error
            assert res.workers.startswith("2 processes") == (label == "processes"), res.workers
        finally:
            invalidate_shared_ml_manager()
            d.close()
        got[label] = v5_loop.order_free(save_rows.dump_rows(tmp_path / f"{label}.db", tmp_path))
    diffs = save_rows.differences(got["processes"], got["threads"], exact=True)
    assert not diffs, "\n".join(diffs)
    tracks = got["processes"]["track_results"]
    assert sum(t["composite_trim_risk_score"] is not None for t in tracks) == 4, \
        "the composite model scored every 8232-1 track in the workers, and no other"
    assert sum(t["sigma_threshold"] == 0.0042 for t in tracks) == 4, "the ML threshold ruled"
    assert save_rows.differences(got["no_predictor"], got["threads"], exact=True), \
        "the predictor moved nothing -- this test could not see it lost on the way to a worker"


def test_under_tests_a_worker_refuses_a_checkouts_real_models_and_the_file_says_so(tmp_path):
    """The conftest guard (no test may score with a checkout's REAL trained models) does not
    reach a spawned child, so the worker carries it: a load from a folder it was told to refuse
    makes the file internal -- loud -- never a quiet score."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.processor import Processor
    from test_batch_writer import _invented_trained_models_for_8232_1
    from laser_trim_analyzer.config import ml_models_directory
    from laser_trim_analyzer.database.manager import DatabaseManager
    d = DatabaseManager(tmp_path / "models.db")
    try:
        _invented_trained_models_for_8232_1(d)
    finally:
        d.close()
    proc = Processor(config=Config(), use_ml=False, snapshot=_snapshot(),
                     ml_storage_path=ml_models_directory())
    pool = _start(proc, 1, refuse_models_under=(str(ml_models_directory()),))
    try:
        f = TRIMS / "dlts_8232-1_242.xls"
        outcome = pool.submit(f, _stat(f)).result(timeout=120)
    finally:
        pool.close()
    assert outcome.internal is not None and outcome.result is None
    assert "trained models" in outcome.internal, outcome.internal


def test_the_suite_tells_every_worker_which_folders_hold_real_models():
    """conftest hands ingest_worker the checkout's real data/ folders for each test, so every
    worker context a test builds carries the refusal above."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.ingest_worker import context_for
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database.specs import SpecSnapshot
    ctx = context_for(Processor(config=Config(), use_ml=False, snapshot=SpecSnapshot()))
    assert set(ctx.refuse_models_under) == set(worker_stubs.checkout_data_dirs())


def test_a_loaded_predictor_predicts_on_one_thread(tmp_path):
    """Controller ruling (2026-09-25): every loaded forest runs on ONE thread. A forest built with
    n_jobs=-1 sums its trees across threads in a varying order, so a stored failure probability
    varied in its last bit run to run -- and inside N worker processes it would be nested
    parallelism, all cores per predict call per worker. A freshly trained one runs on one thread
    too: it can reach a snapshot without being saved."""
    import random
    import pandas as pd
    from laser_trim_analyzer.ml.predictor import FEATURE_COLUMNS, ModelPredictor
    random.seed(5)
    trained = ModelPredictor("9990")
    X = pd.DataFrame([{c: random.random() + (0.4 if i % 3 == 0 else 0.0) for c in FEATURE_COLUMNS}
                      for i in range(60)])
    trained.train(X, pd.Series([1 if i % 3 == 0 else 0 for i in range(60)]))
    assert trained.classifier.n_jobs == 1
    trained.classifier.n_jobs = -1                       # saved as the trainer used to leave it
    assert trained.save(tmp_path / "predictors" / "9990.pkl")
    loaded = ModelPredictor("9990")
    assert loaded.load(tmp_path / "predictors" / "9990.pkl")
    assert loaded.classifier.n_jobs == 1
    features = {c: 0.3 for c in FEATURE_COLUMNS}
    assert len({loaded.predict_failure_probability(features) for _ in range(40)}) == 1


# ---- the writer's seam ------------------------------------------------------------------------

def test_the_writer_settles_a_failed_save_through_the_processors_public_rule(
        db, tmp_path, monkeypatch):
    """The batch writer decides what a failed final-test save becomes through ONE public rule on
    the processor (`failed_save`) -- the same one `apply_outcome` uses -- never its private
    helpers."""
    from laser_trim_analyzer.core.ingest_run import run_folder
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database.manager import DatabaseManager
    from side_writes import FT_FIXTURES
    station = tmp_path / "Test Station" / "ft"
    save_rows._pinned_copy(save_rows.FIXTURES / "final_test" / FT_FIXTURES[0],
                           station / FT_FIXTURES[0])

    def refuses(self, session, *a, **k):
        raise ValueError("invented: Serial cannot be empty")

    monkeypatch.setattr(DatabaseManager, "_save_final_test_in", refuses)
    ruled = []
    real = Processor.failed_save

    def spied(self, write, file_path, exc, started):
        ruled.append((Path(file_path).name, type(exc).__name__))
        return real(self, write, file_path, exc, started)

    monkeypatch.setattr(Processor, "failed_save", spied)
    res = run_folder(str(tmp_path / "Test Station"), db=db, config=None, incremental=True)
    assert res.ok, res.error
    assert ruled == [(FT_FIXTURES[0], "ValueError")], ruled
