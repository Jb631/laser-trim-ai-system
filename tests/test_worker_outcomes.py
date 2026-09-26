"""The worker returns what it found; one writer saves it (ingest-speed Task 9; spec 4.1-4.2,
ruling 9).

Before this, the processor's pool threads made four kinds of database write themselves, in the
middle of the analysis: skip markers, final-test saves, smoothness saves -- and, per file, a spec
lookup (Task 8's snapshot answers those now). A worker PROCESS (Task 11) can open no database, so
`analyse_path` now does the whole analysis and touches none: every write comes back as a value in
an Outcome, carrying the (size, mtime) and hash of the bytes it parsed, and the CONSUMER applies
it -- through a writer when one is given (the ingest), at once through the public methods when
none is (V5's loop, `process_file`). The rows must not move: `v5_loop.py`'s golden holds what V5's
loop stored before any write moved.
"""
import pytest

import save_rows
import v5_loop


@pytest.fixture
def db(tmp_path, monkeypatch):
    from laser_trim_analyzer.database.manager import DatabaseManager
    d = DatabaseManager(tmp_path / "v5.db")
    save_rows.inject(d, monkeypatch)
    yield d
    d.close()


def test_v5s_loop_stores_exactly_todays_rows(db, tmp_path):
    """V5's own loop over every kind of file -- `process_batch` with no writer, `save_analysis`
    for each result -- stores exactly the rows it stored at f232528, before the worker's writes
    moved: every column of every row of the nine tables, and the same results yielded in the same
    order. (Committed golden: the only tolerance is the cross-platform float one, see save_rows.)"""
    paths = v5_loop.build_v5_scenario(tmp_path)
    seen = v5_loop.v5_loop(db, paths, tmp_path, parallel=False)
    snap = v5_loop.v5_snapshot(db.database_path, tmp_path, seen)
    if save_rows.REGENERATE:
        save_rows.write_golden(snap, v5_loop.GOLDEN)
    save_rows.assert_matches_golden(snap, v5_loop.GOLDEN)


def _in_submission_order(monkeypatch):
    """The pool's results in the order the files were SUBMITTED (each still waited for): row ids
    then follow the scenario's order in the parallel path too, so rows compare exactly."""
    from laser_trim_analyzer.core import processor
    monkeypatch.setattr(processor, "as_completed", lambda fs, *a, **k: iter(list(fs)))


def _record_threads(monkeypatch):
    """Every write the database is asked for, with the thread that asked."""
    import threading
    from laser_trim_analyzer.database.manager import DatabaseManager
    calls = []
    for name in ("save_analysis", "save_final_test", "save_smoothness_result",
                 "mark_file_skipped", "write_batch"):
        real = getattr(DatabaseManager, name)

        def recorded(self, *a, _real=real, _name=name, **k):
            calls.append((_name, threading.current_thread()))
            return _real(self, *a, **k)
        monkeypatch.setattr(DatabaseManager, name, recorded)
    return calls


def test_v5s_loop_through_the_pool_stores_the_same_rows_and_writes_on_one_thread(
        db, tmp_path, monkeypatch):
    """The parallel path (the one V5 takes for 100 files or more): the pool threads only
    analyse, and every write -- V5's own trim saves AND the final-test, smoothness and marker
    writes the pool threads used to make themselves -- happens on the thread that iterates the
    batch. The rows are the golden's, and V5's loop still saves its final tests."""
    import threading
    _in_submission_order(monkeypatch)
    calls = _record_threads(monkeypatch)
    paths = v5_loop.build_v5_scenario(tmp_path)
    seen = v5_loop.v5_loop(db, paths, tmp_path, parallel=True)
    snap = v5_loop.v5_snapshot(db.database_path, tmp_path, seen)
    save_rows.assert_matches_golden(snap, v5_loop.GOLDEN)
    kinds = {name for name, _ in calls}
    assert {"save_final_test", "save_smoothness_result", "mark_file_skipped"} <= kinds, kinds
    others = sorted({(name, th.name) for name, th in calls if th is not threading.current_thread()})
    assert others == [], f"written from another thread than the consumer's: {others}"
    stored = [r for r in snap["tables"]["final_test_results"]]
    assert len(stored) == 7 and all(r["id"] for r in stored), "V5's loop saved its final tests"


def test_the_analysis_touches_no_database_for_any_kind_of_file(db, tmp_path, monkeypatch):
    """`analyse_path` for every kind of file the ingest meets -- trims, every final-test format,
    smoothness, not-test-data, every failure -- with `get_database`, `DatabaseManager(...)` and
    `sqlite3.connect` all refusing: nothing reaches for a database. And its Outcomes are the
    WHOLE story: applied afterwards, with V5's trim saves, they store exactly the golden's rows."""
    import sqlite3
    from laser_trim_analyzer.core.processor import Processor, take_spec_snapshot
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    paths = v5_loop.build_v5_scenario(tmp_path)
    proc = Processor(use_ml=False, snapshot=take_spec_snapshot(use_ml=False, db=db))
    proc.ml_storage_path = tmp_path / "no_ml_models"
    reached = []

    def refuse(what):
        def refused(*a, **k):
            reached.append(what)                  # recorded: a reach the caller swallows shows
            raise AssertionError(f"the analysis reached for {what}")
        return refused

    with monkeypatch.context() as m:
        m.setattr(mgr, "get_database", refuse("get_database"))
        m.setattr(dbpkg, "get_database", refuse("get_database"))
        m.setattr(mgr.DatabaseManager, "__init__", refuse("DatabaseManager(...)"))
        m.setattr(sqlite3, "connect", refuse("sqlite3.connect"))
        outcomes = [proc.analyse_path(p) for p in paths]
    assert reached == [], reached
    assert len(outcomes) == len(paths)
    seen = []
    for outcome in outcomes:                      # V5's loop, applied afterwards
        result = proc.apply_outcome(outcome)
        if result is not None:
            seen.append((result.metadata.filename, result.file_type, result.overall_status.value))
            db.save_analysis(result)
    save_rows.assert_matches_golden(v5_loop.v5_snapshot(db.database_path, tmp_path, seen),
                                    v5_loop.GOLDEN)


def test_a_writer_gets_every_outcome_on_the_consumers_thread_and_nothing_else_writes(
        db, tmp_path, monkeypatch):
    """With a writer, `process_batch` hands it one Outcome per file -- a file that is not test
    data included (it yields nothing) -- on the thread iterating the batch, and makes no write of
    its own: with the database refused, the batch runs. The writer's Outcomes, applied, store
    the golden's rows."""
    import threading
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.processor import Processor, take_spec_snapshot
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    _in_submission_order(monkeypatch)
    paths = v5_loop.build_v5_scenario(tmp_path)
    config = Config()
    config.processing.turbo_mode_threshold = 1                     # the pool path
    proc = Processor(config=config, use_ml=False, snapshot=take_spec_snapshot(use_ml=False, db=db))
    proc.ml_storage_path = tmp_path / "no_ml_models"

    class Collecting:
        def __init__(self):
            self.outcomes, self.threads = [], set()

        def add(self, outcome):
            self.outcomes.append(outcome)
            self.threads.add(threading.current_thread())
            return outcome.result

    writer, reached = Collecting(), []

    def refused(*a, **k):
        reached.append("get_database")
        raise AssertionError("the batch reached for the database")

    with monkeypatch.context() as m:
        m.setattr(mgr, "get_database", refused)
        m.setattr(dbpkg, "get_database", refused)
        yielded = list(proc.process_batch([p for p in paths], incremental=False, writer=writer))
    assert reached == []
    assert [o.path for o in writer.outcomes] == [str(p) for p in paths], "one Outcome per file"
    assert writer.threads == {threading.current_thread()}, writer.threads
    assert len(yielded) == sum(1 for o in writer.outcomes if o.result is not None)
    seen = []
    for outcome in writer.outcomes:
        result = proc.apply_outcome(outcome)
        if result is not None:
            seen.append((result.metadata.filename, result.file_type, result.overall_status.value))
            db.save_analysis(result)
    save_rows.assert_matches_golden(v5_loop.v5_snapshot(db.database_path, tmp_path, seen),
                                    v5_loop.GOLDEN)


def test_a_trim_outcome_carries_the_parses_own_stat_and_hash(db, tmp_path, monkeypatch):
    """The (size, mtime) and SHA-256 a trim's Outcome carries are those of the bytes that were
    PARSED (spec 3.6, ruling 10) -- not a stat taken afterwards: a file whose mtime moves the
    moment the parse is done still reports the parse's."""
    import hashlib
    import os
    from laser_trim_analyzer.core import parser as parser_mod
    from laser_trim_analyzer.core.processor import Processor, take_spec_snapshot
    src = save_rows._pinned_copy(save_rows.FIXTURES / "trim" / "dlts_8232-1_242.xls",
                                 tmp_path / "laser" / "dlts_8232-1_242.xls")
    parsed_bytes = src.read_bytes()
    real = parser_mod.ExcelParser.parse_file

    def parse_then_touch(self, file_path, *a, **k):
        out = real(self, file_path, *a, **k)
        os.utime(file_path, (save_rows.MTIME + 100, save_rows.MTIME + 100))   # after the read
        return out

    monkeypatch.setattr(parser_mod.ExcelParser, "parse_file", parse_then_touch)
    proc = Processor(use_ml=False, snapshot=take_spec_snapshot(use_ml=False, db=db))
    proc.ml_storage_path = tmp_path / "no_ml_models"
    outcome = proc.analyse_path(src)
    assert outcome.result is not None and outcome.result.file_type == "trim"
    assert outcome.stat == (len(parsed_bytes), save_rows.MTIME), outcome.stat
    assert outcome.file_hash == hashlib.sha256(parsed_bytes).hexdigest()
    assert outcome.writes == () and outcome.identity_error is None


def test_a_final_tests_stat_is_the_walks_converted_as_the_scan_reads_it(db, tmp_path, monkeypatch):
    """The review's Task 9 hand-off: a final test records the WALK's (size, mtime) with the
    mtime converted `datetime.fromtimestamp` (local), exactly as `_disk_stat_for_save` always
    did -- and so the next scan settles it from memory, with no hash and no read. Run in a
    zone off UTC, where a UTC conversion would miss the fast path by hours."""
    import time
    from datetime import datetime
    from laser_trim_analyzer.core.processor import Processor, take_spec_snapshot
    from laser_trim_analyzer.database.manager import FinalTestWrite
    if hasattr(time, "tzset"):
        monkeypatch.setenv("TZ", "America/Chicago")
        time.tzset()
    try:
        ft = save_rows._pinned_copy(
            save_rows.FIXTURES / "final_test" / "7458-sn7_4-2-2026_3-19 PM.xls",
            tmp_path / "Test Station" / "7458-sn7_4-2-2026_3-19 PM.xls")
        walk = (ft.stat().st_size, save_rows.MTIME - 7200.0)       # the walk's own, invented
        proc = Processor(use_ml=False, snapshot=take_spec_snapshot(use_ml=False, db=db))
        outcome = proc.analyse_path(ft, walk)
        (write,) = outcome.writes
        assert isinstance(write, FinalTestWrite)
        assert (write.file_size, write.file_modified_date) == (
            walk[0], datetime.fromtimestamp(walk[1]))
        assert outcome.stat == walk
        result = proc.apply_outcome(outcome)
        assert result.final_test_id is not None
        fresh = Processor(use_ml=False)
        fresh._load_processed_hashes()
        fresh._disk_stats = {str(ft): walk}
        assert fresh._classify_scan(ft) == "processed", "the next scan must know it from memory"
    finally:
        if hasattr(time, "tzset"):
            monkeypatch.delenv("TZ", raising=False)
            time.tzset()


def _integrity(msg):
    from sqlalchemy.exc import IntegrityError
    return IntegrityError("INSERT INTO final_test_tracks (invented)", {}, Exception(msg))


def _operational(msg):
    from sqlalchemy.exc import OperationalError
    return OperationalError("INSERT INTO final_test_results (invented)", {}, Exception(msg))


# (what failed, the exception, the marker it earns: its reason / None = a marker with no reason /
# "no marker"). Every value is invented.
SAVE_FAILURES = [
    ("a content refusal", ValueError("invented: the tracks disagree"),
     "ValueError: invented: the tracks disagree"),
    ("a permanent refusal", ValueError("Serial cannot be empty"), None),
    ("a malformed unit", _integrity("UNIQUE constraint failed: final_test_tracks.track_id"), None),
    ("a database error", _operational("no such column: invented_column"), "no marker"),
    ("a bug in the save", RuntimeError("invented: the save's own bug"), "no marker"),
    ("a locked file", PermissionError("invented: the file is locked"), "no marker"),
]


@pytest.mark.parametrize("what,failure,marker_reason", SAVE_FAILURES,
                         ids=[c[0] for c in SAVE_FAILURES])
@pytest.mark.parametrize("via", ["process_file", "the ingest's batch writer"])
def test_a_final_test_save_that_fails_marks_the_file_only_for_its_own_content(
        db, tmp_path, monkeypatch, what, failure, marker_reason, via):
    """A final-test save that raises -- on the consumer's thread, after the analysis, and in the
    ingest inside a batch -- is an ERROR result ("Final Test error: ..."). It records the file as
    unreadable ONLY when the file's own content caused it (ruling of 2026-09-25): a validation
    refusal is marked with its reason, a permanent one or a malformed unit with no reason (as
    ever); a database or system error -- or anything else the save cannot attribute to the file
    -- marks NOTHING: the file stays new. In a batch it counts as an error (a malformed unit as
    failed), and its marker commits right after the batch."""
    import sqlite3
    from laser_trim_analyzer.core.ingest_run import BatchWriter
    from laser_trim_analyzer.core.processor import Processor, take_spec_snapshot
    from laser_trim_analyzer.database.manager import DatabaseManager
    ft = save_rows._pinned_copy(
        save_rows.FIXTURES / "final_test" / "7458-sn7_4-2-2026_3-19 PM.xls",
        tmp_path / "Test Station" / "7458-sn7_4-2-2026_3-19 PM.xls")

    def fails(self, *a, **kw):
        raise failure

    monkeypatch.setattr(DatabaseManager, "save_final_test" if via == "process_file"
                        else "_save_final_test_in", fails)     # the public save / the batch body
    proc = Processor(use_ml=False, snapshot=take_spec_snapshot(use_ml=False, db=db))
    if via == "process_file":
        result = proc.process_file(ft)
    else:
        settled = []
        writer = BatchWriter(proc, db, settled.append)
        writer.add(proc.analyse_path(ft))
        writer.flush()
        (committed,) = settled
        want_bucket = "failed" if what == "a malformed unit" else "errors"
        assert committed.bucket == want_bucket and not committed.saved, committed
        assert committed.unsaved == (marker_reason == "no marker"), committed
        result = committed.result
    assert result.overall_status.value == "Error" and result.file_type == "final_test"
    assert result.errors == [f"Final Test error: {failure}"]
    con = sqlite3.connect(f"file:{db.database_path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT error_message, success FROM processed_files").fetchall()
    finally:
        con.close()
    if marker_reason == "no marker":
        assert rows == [], f"{what} must never mark the file unreadable: {rows}"
    elif marker_reason is None:
        assert len(rows) == 1 and rows[0][0].startswith("content sha256="), rows
    else:
        assert len(rows) == 1 and rows[0][0].startswith(f"unreadable: {marker_reason}"), rows


@pytest.mark.parametrize("failure,marked", [
    (ValueError("invented: the smoothness save failed"), True),       # the file's own content
    (_operational("disk I/O error (invented)"), False),               # the database's refusal
], ids=["a content refusal", "a database error"])
def test_a_smoothness_save_that_fails_marks_the_file_only_for_its_own_content(
        db, tmp_path, monkeypatch, failure, marked):
    """The smoothness rule, on the save side: an ERROR result ("Smoothness error: ..."), and a
    skip marker with its reason ONLY for the file's own content -- a database error leaves the
    file new (ruling of 2026-09-25)."""
    import sqlite3
    from laser_trim_analyzer.core.processor import Processor, take_spec_snapshot
    from laser_trim_analyzer.database.manager import DatabaseManager
    os_file = v5_loop._betatronix_os(tmp_path / "os" / "9991-sn4_OS_2-3-2026_10-00-00 AM.xlsx",
                                     max_dev=0.0031, spec=0.0050, result="PASSED")

    def fails(self, **kw):
        raise failure

    monkeypatch.setattr(DatabaseManager, "save_smoothness_result", fails)
    proc = Processor(use_ml=False, snapshot=take_spec_snapshot(use_ml=False, db=db))
    result = proc.process_file(os_file)
    assert result.overall_status.value == "Error" and result.file_type == "smoothness"
    assert result.errors == [f"Smoothness error: {failure}"]
    con = sqlite3.connect(f"file:{db.database_path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT error_message FROM processed_files").fetchall()
    finally:
        con.close()
    if marked:
        assert len(rows) == 1 and rows[0][0].startswith(
            "unreadable: ValueError: invented: the smoothness save failed"), rows
    else:
        assert rows == [], rows


def test_a_written_marker_is_remembered_by_the_run(db, tmp_path):
    """As when the analysis wrote markers itself: once written, this run's in-memory caches know
    the path and its content, so the rest of the run treats them as processed."""
    import hashlib
    from laser_trim_analyzer.core.processor import Processor
    noise = tmp_path / "junk" / "9993_noise_capture.xls"
    noise.parent.mkdir(parents=True)
    noise.write_bytes(b"invented: an oscilloscope capture")
    proc = Processor(use_ml=False)
    proc._processed_filenames, proc._processed_hashes = set(), set()
    assert proc.process_file(noise) is None
    assert str(noise) in proc._processed_filenames
    assert hashlib.sha256(noise.read_bytes()).hexdigest() in proc._processed_hashes


def test_run_folder_writes_everything_into_its_own_database(tmp_path, monkeypatch):
    """run_folder's writer writes through the database run_folder was GIVEN -- where its trim
    saves go -- never a second one reached through `get_database()`: a run's final tests,
    smoothness rows and markers land beside its trims."""
    import sqlite3
    from laser_trim_analyzer.core.ingest_run import run_folder
    from laser_trim_analyzer.database.manager import DatabaseManager
    given = DatabaseManager(tmp_path / "given.db")
    elsewhere = DatabaseManager(tmp_path / "elsewhere.db")
    save_rows.inject(elsewhere, monkeypatch)              # what get_database() would hand out
    folder = tmp_path / "run"
    save_rows._pinned_copy(save_rows.FIXTURES / "final_test" / "7458-sn7_4-2-2026_3-19 PM.xls",
                           folder / "Test Station" / "7458-sn7_4-2-2026_3-19 PM.xls")
    save_rows._pinned_copy(save_rows.FIXTURES / "trim" / "dlts_8232-1_242.xls",
                           folder / "laser" / "dlts_8232-1_242.xls")
    noise = folder / "laser" / "9993_noise_capture.xls"
    noise.write_bytes(b"invented: an oscilloscope capture")
    try:
        res = run_folder(str(folder), db=given, config=None, incremental=True)
        assert res.ok, res.error

        def count(db, table):
            con = sqlite3.connect(f"file:{db.database_path}?mode=ro", uri=True)
            try:
                return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            finally:
                con.close()
        assert [count(given, t) for t in ("analysis_results", "final_test_results",
                                          "processed_files")] == [1, 1, 2]
        assert [count(elsewhere, t) for t in ("analysis_results", "final_test_results",
                                              "processed_files")] == [0, 0, 0]
    finally:
        given.close()
        elsewhere.close()
