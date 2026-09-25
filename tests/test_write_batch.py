"""`write_batch`: one transaction per batch, a savepoint per file, processed means committed
(ingest-speed Task 6; spec 3.2-3.6, rulings 4-8 and 22).

The traps it is built around were measured before a line of it existed (spec section 1):
  F4  pysqlite emits no BEGIN before a SAVEPOINT, so SQLite starts the transaction AT the first
      savepoint and RELEASE of it is a COMMIT: without an explicit `BEGIN IMMEDIATE` a "batch" is
      one commit per file, silently.
  F5  one SQLite connection serves every thread (StaticPool), so a helper that opens its own
      `session()` inside a batch commits everything before it.
  F6  a hard crash mid-batch must leave every file of the batch unprocessed (new next run), and
      one bad file must leave only itself unprocessed.
Each test below is the measured trap, now refused -- and each was made to fail by breaking the
rule it pins (see the Task 6 report's mutation table).
"""
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from sqlalchemy import event, text

import save_rows

REPO = Path(__file__).resolve().parents[1]
FOUR = ("dlts_8232-1_242.xls", "dlts_8232-1_243.xls", "lts_8232-1_193.xls", "lts_8232-1_194.xls")
_PARSED = {}


@pytest.fixture
def db(tmp_path, monkeypatch):
    from laser_trim_analyzer.database.manager import DatabaseManager
    d = DatabaseManager(tmp_path / "batch.db")
    save_rows.inject(d, monkeypatch)
    yield d
    d.close()


def _parsed(name):
    """A fixture's parse, once per test process (no test database holds a spec, so every
    database gives the same result), handed out as a deep copy."""
    if name not in _PARSED:
        from laser_trim_analyzer.core.processor import Processor
        proc = Processor(use_ml=False)
        proc.ml_storage_path = REPO / "no_ml_models_here"
        _PARSED[name] = proc.process_file(REPO / "tests" / "fixtures" / "trim" / name)
    return _PARSED[name].model_copy(deep=True)


def _items(db, names=FOUR):
    from laser_trim_analyzer.database.manager import TrimWrite
    out = []
    for n in names:
        r = _parsed(n)
        out.append(TrimWrite(r, *db._file_identity(r.metadata.file_path)))
    return out


def _count(db_path, table):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def _rows_by_file(db_path):
    """{filename: (analysis rows, track rows, pass rows, setup rows, processed rows)}."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = {}
        for (name,) in con.execute("SELECT DISTINCT filename FROM processed_files "
                                   "UNION SELECT DISTINCT filename FROM analysis_results"):
            aids = [r[0] for r in con.execute("SELECT id FROM analysis_results WHERE filename=?",
                                              (name,))]
            q = ",".join("?" * len(aids)) or "NULL"
            out[name] = (
                len(aids),
                con.execute(f"SELECT COUNT(*) FROM track_results WHERE analysis_id IN ({q})",
                            aids).fetchone()[0],
                con.execute("SELECT COUNT(*) FROM trim_passes WHERE track_result_id IN "
                            f"(SELECT id FROM track_results WHERE analysis_id IN ({q}))",
                            aids).fetchone()[0],
                con.execute(f"SELECT COUNT(*) FROM trim_setup WHERE analysis_id IN ({q})",
                            aids).fetchone()[0],
                con.execute("SELECT COUNT(*) FROM processed_files WHERE filename=?",
                            (name,)).fetchone()[0])
        return out
    finally:
        con.close()


def _scan(paths):
    """What the ingest's own incremental scan calls each path: 'new' or 'processed'."""
    from laser_trim_analyzer.core.processor import Processor
    proc = Processor(use_ml=False)
    proc._load_processed_hashes()
    proc._disk_stats = {str(p): (os.stat(p).st_size, os.stat(p).st_mtime) for p in paths}
    return [proc._classify_scan(Path(p)) for p in paths]


class _Watcher:
    """A second connection, asking at every SAVEPOINT the batch opens: can I write, and how many
    rows do I see? Plus a count of the batch's COMMITs."""

    def __init__(self, db):
        self.db = db
        self.other = sqlite3.connect(str(db.database_path), timeout=0)
        self.seen, self.commits = [], 0
        event.listen(db._engine, "savepoint", self._at_savepoint)
        event.listen(db._engine, "commit", self._at_commit)

    def _at_savepoint(self, conn, name):
        try:
            self.other.execute("BEGIN IMMEDIATE")
            self.other.execute("ROLLBACK")
            could_write = True
        except sqlite3.OperationalError as e:
            could_write = str(e)
        rows = self.other.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0]
        self.seen.append((could_write, rows))

    def _at_commit(self, conn):
        self.commits += 1

    def close(self):
        event.remove(self.db._engine, "savepoint", self._at_savepoint)
        event.remove(self.db._engine, "commit", self._at_commit)
        self.other.close()


@pytest.fixture
def watcher():
    """`watcher(db)` -> a _Watcher, released at the end of the test however it ends."""
    made = []
    yield lambda db: made.append(_Watcher(db)) or made[-1]
    for w in made:
        w.close()


# ---- ruling 4 / F4: one transaction, taken with BEGIN IMMEDIATE ---------------------------------

def test_a_batch_is_one_transaction_that_holds_the_write_lock_from_its_first_statement(db, watcher):
    """At EVERY savepoint -- the first one included, before the batch has written anything -- a
    second connection can neither write nor see a single row, and the batch commits exactly once.
    Without BEGIN IMMEDIATE the first RELEASE commits (F4); with a plain deferred BEGIN the lock
    is only taken at the first write, and the second connection gets in before it."""
    items = _items(db)
    w = watcher(db)
    outcomes = db.write_batch(items)
    assert [o.status for o in outcomes] == ["saved"] * 4
    assert w.seen == [("database is locked", 0)] * 4, w.seen
    assert w.commits == 1
    assert _count(db.database_path, "analysis_results") == 4


# ---- ruling 7 / F6: a savepoint per file; the file's rows and its marker share it ---------------

def test_a_file_that_fails_mid_save_rolls_back_alone_and_the_others_commit(db, monkeypatch):
    """The third file fails AFTER its analysis row, tracks and passes were flushed (in its setup
    row): it keeps none of them and no processed marker, so the next scan calls it new; the other
    three commit with their markers."""
    items = _items(db)
    doomed = items[2].analysis.metadata.filename
    real = db._write_trim_setup

    def fails_for_one(session, analysis_id, setup):
        from laser_trim_analyzer.database.models import AnalysisResult as A
        if session.get(A, analysis_id).filename == doomed:
            raise RuntimeError("invented failure in one file's save")
        return real(session, analysis_id, setup)

    monkeypatch.setattr(db, "_write_trim_setup", fails_for_one)
    with save_rows.no_file_io(monkeypatch) as touched:      # the failure path touches no file either
        outcomes = db.write_batch(items)
    assert touched == [], touched
    assert [o.status for o in outcomes] == ["saved", "saved", "failed", "saved"]
    assert "invented failure" in outcomes[2].reason and outcomes[2].row_id is None
    rows = _rows_by_file(db.database_path)
    assert doomed not in rows, f"the failed file left rows behind: {rows.get(doomed)}"
    assert all(rows[n][0] == 1 and rows[n][4] == 1 for n in FOUR if n != doomed), rows
    paths = [i.analysis.metadata.file_path for i in items]
    assert _scan(paths) == ["processed", "processed", "new", "processed"]


def test_a_marker_that_cannot_be_written_takes_its_files_rows_with_it(db, monkeypatch):
    """Ruling 7 from the other side: the failure is in the processed marker itself. If the marker
    lived outside the file's savepoint, the rows would commit without it -- data nobody marked."""
    items = _items(db)
    doomed = items[1].analysis.metadata.filename
    real = db._record_processed_file

    def fails_for_one(session, file_path, *a, **k):
        if Path(file_path).name == doomed:
            raise RuntimeError("invented failure writing one marker")
        return real(session, file_path, *a, **k)

    monkeypatch.setattr(db, "_record_processed_file", fails_for_one)
    outcomes = db.write_batch(items)
    assert [o.status for o in outcomes] == ["saved", "failed", "saved", "saved"]
    rows = _rows_by_file(db.database_path)
    assert doomed not in rows, f"rows committed without their marker: {rows.get(doomed)}"
    assert _count(db.database_path, "processed_files") == 3


_CHILD = r'''
import logging, os, sys
from pathlib import Path
logging.disable(logging.WARNING)
db_path, kill_at, tmp = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
files = [Path(p) for p in sys.argv[4:]]
from sqlalchemy import event
from laser_trim_analyzer.database import manager as mgr
import laser_trim_analyzer.database as dbpkg
db = mgr.DatabaseManager(db_path)            # BOTH globals, before any Processor exists
mgr._db_manager = db
dbpkg._db_manager = db
from laser_trim_analyzer.core.processor import Processor
proc = Processor(use_ml=False)
proc.ml_storage_path = tmp / "no_ml_models"
items = [mgr.TrimWrite(r, *db._file_identity(r.metadata.file_path))
         for r in (proc.process_file(f) for f in files)]
if kill_at == "commit":
    @event.listens_for(db._engine, "commit")
    def _die(conn):
        os._exit(3)                          # every savepoint released; only the COMMIT missing
else:
    released = [0]
    @event.listens_for(db._engine, "release_savepoint")
    def _die(conn, name, context):
        released[0] += 1
        if released[0] == int(kill_at):
            os._exit(3)
db.write_batch(items)
print("NOT KILLED")
'''


@pytest.mark.parametrize("kill_at", ["2", "commit"])
def test_a_crash_mid_batch_leaves_every_file_of_the_batch_new(tmp_path, kill_at):
    """F6, in a real child process killed with os._exit -- no COMMIT, no cleanup -- after the second
    file's RELEASE, or with every savepoint released and only the COMMIT missing. Afterwards no
    table holds a row of the batch, the ingest's own scan calls all four files new, and a rerun
    saves them as new rows."""
    import shutil
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    files = []
    for n in FOUR:
        files.append(tmp_path / "in" / n)
        files[-1].parent.mkdir(exist_ok=True)
        shutil.copyfile(REPO / "tests" / "fixtures" / "trim" / n, files[-1])
    db_path = tmp_path / "crash.db"
    mgr.DatabaseManager(db_path).close()          # the schema exists before the child starts
    child = tmp_path / "child.py"
    child.write_text(_CHILD)
    r = subprocess.run([sys.executable, "-B", str(child), str(db_path), kill_at, str(tmp_path),
                        *map(str, files)], cwd=tmp_path, capture_output=True, text=True,
                       timeout=300, env=dict(os.environ, PYTHONPATH=str(REPO / "src")))
    assert r.returncode == 3 and "NOT KILLED" not in r.stdout, (r.returncode, r.stdout, r.stderr[-2000:])
    for table in save_rows.TABLES:
        assert _count(db_path, table) == 0, f"{table} kept rows of a batch that never committed"
    d = mgr.DatabaseManager(db_path)
    old = (mgr._db_manager, getattr(dbpkg, "_db_manager", None))
    mgr._db_manager = dbpkg._db_manager = d
    try:
        assert _scan(files) == ["new"] * 4
        from laser_trim_analyzer.core.processor import Processor
        proc = Processor(use_ml=False)
        proc.ml_storage_path = tmp_path / "no_ml_models"
        items = [mgr.TrimWrite(x, *d._file_identity(x.metadata.file_path))
                 for x in (proc.process_file(f) for f in files)]
        rerun = d.write_batch(items)
        assert [(o.status, o.row_id) for o in rerun] == [("saved", 1), ("saved", 2),
                                                         ("saved", 3), ("saved", 4)]
        assert _scan(files) == ["processed"] * 4
    finally:
        mgr._db_manager, dbpkg._db_manager = old
        d.close()


# ---- ruling 6 / F5: nothing inside a batch opens its own session ---------------------------------

def test_a_session_opened_inside_a_batch_raises_and_fails_only_its_file(db, monkeypatch):
    """A helper that opens its own session mid-batch (as mark_file_skipped does) is REFUSED: it
    raises, its write never happens, its file fails -- and the other files commit."""
    from laser_trim_analyzer.database.manager import NestedSessionError
    items = _items(db)
    raised = []
    real = db._write_trim_setup
    culprit = items[1].analysis.metadata.filename

    def opens_its_own(session, analysis_id, setup):
        from laser_trim_analyzer.database.models import AnalysisResult as A
        if session.get(A, analysis_id).filename == culprit:
            try:
                db.mark_file_skipped(filename="helper.xls", file_path="/invented/helper.xls",
                                     file_hash="ab" * 32, file_size=1, file_modified_date=None)
            except Exception as e:
                raised.append(e)
                raise
        return real(session, analysis_id, setup)

    monkeypatch.setattr(db, "_write_trim_setup", opens_its_own)
    outcomes = db.write_batch(items)
    assert len(raised) == 1 and isinstance(raised[0], NestedSessionError)
    assert [o.status for o in outcomes] == ["saved", "failed", "saved", "saved"]
    assert "session" in outcomes[1].reason
    rows = _rows_by_file(db.database_path)
    assert "helper.xls" not in rows and culprit not in rows
    assert _count(db.database_path, "analysis_results") == 3


def test_a_nested_session_can_no_longer_commit_half_a_batch(db, monkeypatch, watcher):
    """F5 end to end: the helper runs mid-batch, the batch then fails to commit. Before the guard
    the helper's commit had already stored the first file (and itself); now nothing survives."""
    from laser_trim_analyzer.database.manager import BatchCommitError
    items = _items(db, FOUR[:3])
    real = db._write_trim_setup
    second = items[1].analysis.metadata.filename

    def helper_on_second(session, analysis_id, setup):
        from laser_trim_analyzer.database.models import AnalysisResult as A
        if session.get(A, analysis_id).filename == second:
            try:
                db.mark_file_skipped(filename="helper.xls", file_path="/invented/helper.xls",
                                     file_hash="ab" * 32, file_size=1, file_modified_date=None)
            except Exception:
                pass                                  # a helper that swallows everything
        return real(session, analysis_id, setup)

    monkeypatch.setattr(db, "_write_trim_setup", helper_on_second)
    w = watcher(db)

    # Refuse ONLY the batch's own final commit: armed at the third file's savepoint, after the
    # helper ran. Refusing every commit would also refuse the HELPER's -- and hide the very
    # mid-batch commit F5 is about (that version of this test passed with the guard removed).
    def arm(conn, name):
        if len(w.seen) == 3:
            armed.append(True)
    armed = []
    event.listen(db._engine, "savepoint", arm)

    def refuse_commit(conn):
        if armed:
            raise RuntimeError("invented: the disk refused the commit")
    event.listen(db._engine, "commit", refuse_commit)
    with pytest.raises(BatchCommitError):
        db.write_batch(items)
    assert armed and [rows for _, rows in w.seen] == [0, 0, 0], f"rows visible mid-batch: {w.seen}"
    for table in save_rows.TABLES:
        assert _count(db.database_path, table) == 0, f"{table} kept rows of a failed batch"


def test_a_swallowed_refusal_still_fails_its_file(db, monkeypatch):
    """The refusal is RECORDED as well as raised: a helper that catches it and carries on cannot
    turn a half-done file into a saved one."""
    items = _items(db, FOUR[:2])
    real = db._write_trim_setup

    def swallows(session, analysis_id, setup):
        try:
            with db.session() as s:
                s.execute(text("SELECT 1"))
        except Exception:
            pass
        return real(session, analysis_id, setup)

    monkeypatch.setattr(db, "_write_trim_setup", swallows)
    outcomes = db.write_batch(items)
    assert [o.status for o in outcomes] == ["failed", "failed"]
    assert _count(db.database_path, "analysis_results") == 0


def test_another_thread_waits_for_the_batch_instead_of_being_refused(db, monkeypatch):
    """The guard is per THREAD: a UI loader on another thread is not refused -- it waits on the
    lock for the whole batch (never getting in mid-batch) and then reads all of it."""
    items = _items(db, FOUR[:2])
    started, release, seen = threading.Event(), threading.Event(), {}
    real = db._write_trim_setup

    def pauses_once(session, analysis_id, setup):
        if not started.is_set():
            started.set()
            release.wait(10)
        return real(session, analysis_id, setup)

    monkeypatch.setattr(db, "_write_trim_setup", pauses_once)

    def reader():
        started.wait(10)
        try:
            with db.session() as s:
                seen["rows"] = s.execute(text("SELECT COUNT(*) FROM analysis_results")).scalar()
        except Exception as e:
            seen["error"] = e

    batch = threading.Thread(target=lambda: seen.setdefault("outcomes", db.write_batch(items)))
    other = threading.Thread(target=reader)
    batch.start()
    other.start()
    assert started.wait(10)
    time.sleep(0.3)                                    # the reader is at the lock by now
    got_in_mid_batch = dict(seen)
    release.set()
    batch.join(20)
    other.join(20)
    assert got_in_mid_batch == {}, f"another thread got in mid-batch: {got_in_mid_batch}"
    assert "error" not in seen, f"another thread was refused: {seen.get('error')!r}"
    assert seen["rows"] == 2 and [o.status for o in seen["outcomes"]] == ["saved", "saved"]


# ---- review I-1: a transaction ended UNDER the batch is never reported as a committed batch ------

def _orphans(db_path):
    """Track rows with no analysis row, and pass rows with no track row."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return (con.execute("SELECT COUNT(*) FROM track_results WHERE analysis_id NOT IN "
                            "(SELECT id FROM analysis_results)").fetchone()[0]
                + con.execute("SELECT COUNT(*) FROM trim_passes WHERE track_result_id NOT IN "
                              "(SELECT id FROM track_results)").fetchone()[0])
    finally:
        con.close()


def _assert_what_survives_and_that_a_rerun_completes_it(db, items, ended_by):
    """After a transaction was ended under the batch: only what that foreign COMMIT made durable
    survives -- the first file whole, the second's first rows WITHOUT their marker -- or nothing,
    after a ROLLBACK. So every marker that exists sits on a whole file, the scan re-offers every
    file that is not whole, and running the same batch again completes all four."""
    stored = {n: (r[0], r[4]) for n, r in _rows_by_file(db.database_path).items()}
    paths = [i.analysis.metadata.file_path for i in items]
    if ended_by == "commit":
        assert stored == {FOUR[0]: (1, 1), FOUR[1]: (1, 0)}, stored
        assert _scan(paths) == ["processed", "new", "new", "new"]
    else:
        assert stored == {}, stored
        assert _scan(paths) == ["new"] * 4
    assert [o.status for o in db.write_batch(_items(db))] == ["saved"] * 4
    rows = _rows_by_file(db.database_path)
    assert all(rows[n][0] == 1 and rows[n][4] == 1 for n in FOUR) and _orphans(db.database_path) == 0, rows


@pytest.mark.parametrize("how", ["commit", "rollback"])
def test_a_transaction_ended_under_the_batch_makes_it_raise(db, monkeypatch, how):
    """Review I-1. A COMMIT or ROLLBACK that is not the batch's -- straight on the one shared
    connection -- ends its transaction while the second file is half-written, and the file carries
    on writing. pysqlite silently BEGINs again at the next INSERT, so the per-file tripwire saw a
    transaction, the file's failed RELEASE was booked as that file's own failure, and the batch
    committed the rest: outcomes that contradicted the rows. Now the batch's guard savepoint is gone
    with the transaction it lived in, and the batch raises instead of committing."""
    from laser_trim_analyzer.database.manager import BatchCommitError
    items = _items(db)
    real, armed = db._write_trim_setup, [True]

    def ends_it_once(session, analysis_id, setup):
        from laser_trim_analyzer.database.models import AnalysisResult as A
        if armed and session.get(A, analysis_id).filename == FOUR[1]:
            armed.clear()
            getattr(session.connection().connection.dbapi_connection, how)()
        return real(session, analysis_id, setup)

    monkeypatch.setattr(db, "_write_trim_setup", ends_it_once)
    with pytest.raises(BatchCommitError) as e:
        db.write_batch(items)
    assert [o.status for o in e.value.outcomes] == ["failed"] * 4
    assert "ended" in str(e.value)
    _assert_what_survives_and_that_a_rerun_completes_it(db, items, how)


@pytest.mark.parametrize("action", ["commit", "close"])
def test_a_lockless_session_on_another_thread_cannot_end_the_batch_quietly(db, monkeypatch, action):
    """The reviewer's counterfactual (I-1): while the batch is paused inside its second file,
    another thread makes a Session WITHOUT `_write_lock` -- only the raw sessionmaker can, now that
    `_new_session` refuses -- and commits it, or only closes it (the pool's reset is a ROLLBACK).
    Before the fix write_batch returned normally: file 2 'failed' yet stored whole with its marker,
    or file 1 'saved' yet stored nowhere. Now it raises, with the same invariants as above."""
    from laser_trim_analyzer.database.manager import BatchCommitError
    items = _items(db)
    paused, release, result = threading.Event(), threading.Event(), {}
    real, calls = db._write_trim_setup, [0]

    def pauses_in_the_second(session, analysis_id, setup):
        calls[0] += 1
        if calls[0] == 2:
            paused.set()
            release.wait(20)
        return real(session, analysis_id, setup)

    monkeypatch.setattr(db, "_write_trim_setup", pauses_in_the_second)

    def run():
        try:
            result["outcomes"] = db.write_batch(items)
        except Exception as e:
            result["error"] = e

    def lockless():
        s = db._session_maker()                    # the counterfactual: no lock, no guard
        s.execute(text("SELECT 1"))
        if action == "commit":
            s.commit()
        s.close()

    batch = threading.Thread(target=run)
    batch.start()
    try:
        assert paused.wait(20)
        other = threading.Thread(target=lockless)
        other.start()
        other.join(20)
    finally:
        release.set()
        batch.join(30)
    assert isinstance(result.get("error"), BatchCommitError), result
    calls[0] = 99                                  # the rerun below must not pause
    _assert_what_survives_and_that_a_rerun_completes_it(
        db, items, "commit" if action == "commit" else "rollback")


def test_no_session_can_be_made_without_holding_the_lock(db):
    """I-1's other half: `session()` takes `_write_lock` before it makes a Session, and
    `_new_session` now REFUSES a caller that does not hold it -- on this thread -- so the lockless
    Session above cannot be written by accident."""
    from laser_trim_analyzer.database.manager import DatabaseError
    with pytest.raises(DatabaseError, match="_write_lock"):
        db._new_session()
    with db._write_lock:
        db._new_session().close()
        refused = []
        t = threading.Thread(target=lambda: refused.append(_refusal(db._new_session)))
        t.start()
        t.join(10)
    assert refused == ["DatabaseError"], "the lock held by ANOTHER thread is not this one's"
    with db.session() as s:
        assert s.execute(text("SELECT 1")).scalar() == 1


def _refusal(make):
    try:
        make().close()
        return None
    except Exception as e:
        return type(e).__name__


# ---- outcomes, and ruling 22: a failed batch commit is named and counted -------------------------

def test_each_file_gets_an_outcome_saved_duplicate_or_failed(db, monkeypatch):
    """saved(row id) | duplicate (a UNIQUE constraint the file's OWN rows broke -- here two tracks
    with one track_id; never "already stored", which the body turns into an update -- the bucket
    the ingest has always called 'skipped') | failed(why) -- one per item, in order. Only UNIQUE
    is a duplicate: any other IntegrityError is a failure, never quietly a skip."""
    import sqlite3 as _sqlite3
    from sqlalchemy.exc import IntegrityError
    from laser_trim_analyzer.database.manager import TrimWrite
    items = _items(db, (FOUR[0],))
    twice = _parsed(FOUR[1])
    twice.tracks.append(twice.tracks[0].model_copy())              # two tracks, one track_id
    no_id = _parsed(FOUR[2])
    no_id.tracks[0].track_id = None                                # refused by the ORM
    other = _parsed(FOUR[3])
    for r in (twice, no_id, other):
        items.append(TrimWrite(r, *db._file_identity(r.metadata.file_path)))
    real = db._write_trim_setup

    def not_null_for_the_last(session, analysis_id, setup):
        from laser_trim_analyzer.database.models import AnalysisResult as A
        if session.get(A, analysis_id).filename == other.metadata.filename:
            raise IntegrityError("INSERT INTO trim_setup ...", {},
                                 _sqlite3.IntegrityError("NOT NULL constraint failed: trim_setup.x"))
        return real(session, analysis_id, setup)

    monkeypatch.setattr(db, "_write_trim_setup", not_null_for_the_last)
    outcomes = db.write_batch(items)
    assert [o.status for o in outcomes] == ["saved", "duplicate", "failed", "failed"]
    assert outcomes[0].row_id == 1 and outcomes[0].reason is None
    assert "UNIQUE constraint" in outcomes[1].reason and outcomes[1].row_id is None
    assert "Track ID cannot be empty" in outcomes[2].reason
    assert "NOT NULL constraint" in outcomes[3].reason
    assert _count(db.database_path, "analysis_results") == 1


def test_a_batch_that_cannot_commit_stores_nothing_and_says_how_many_in_a_row(db, monkeypatch):
    """Ruling 22's raw material: a batch whose transaction cannot be committed raises
    BatchCommitError -- every item failed (nothing was stored), the cause named, and how many batch
    commits in a row have now failed; one that commits resets the count."""
    from laser_trim_analyzer.database.manager import BatchCommitError
    refuse = [True]

    def maybe_refuse(conn):
        if refuse[0]:
            raise RuntimeError("invented: the disk refused the commit")
    event.listen(db._engine, "commit", maybe_refuse)
    seen = []
    for _ in range(2):
        with pytest.raises(BatchCommitError) as e:
            db.write_batch(_items(db, FOUR[:2]))
        seen.append((e.value.consecutive, [o.status for o in e.value.outcomes]))
        assert "the disk refused the commit" in str(e.value)
    assert seen == [(1, ["failed", "failed"]), (2, ["failed", "failed"])]
    assert _count(db.database_path, "analysis_results") == 0
    refuse[0] = False
    assert [o.status for o in db.write_batch(_items(db, FOUR[:1]))] == ["saved"]
    refuse[0] = True
    with pytest.raises(BatchCommitError) as e:
        db.write_batch(_items(db, FOUR[1:2]))
    assert e.value.consecutive == 1, "a commit that succeeded must reset the count"


def test_a_batch_that_cannot_take_the_write_lock_fails_loudly_and_stores_nothing(db):
    """BEGIN IMMEDIATE takes the lock up front: another process holding it makes the batch fail
    at its first statement, whole -- never half-way through on a read-to-write upgrade."""
    from laser_trim_analyzer.database.manager import BatchCommitError
    with db.session() as s:
        s.execute(text("PRAGMA busy_timeout=100"))     # not the app's 30 s: this is a test
    holder = sqlite3.connect(str(db.database_path))
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(BatchCommitError) as e:
            db.write_batch(_items(db, FOUR[:2]))
        assert "database is locked" in str(e.value)
        assert [o.status for o in e.value.outcomes] == ["failed", "failed"]
    finally:
        holder.rollback()
        holder.close()
    assert _count(db.database_path, "analysis_results") == 0


def test_write_batch_refuses_what_it_cannot_write(db):
    from laser_trim_analyzer.database.manager import TrimWrite
    with pytest.raises(TypeError):
        db.write_batch([_parsed(FOUR[0])])             # a bare result, not a TrimWrite
    assert db.write_batch([]) == []
    ft = _parsed(FOUR[0])
    ft.file_type = "final_test"
    outcomes = db.write_batch([TrimWrite(ft, None, None)])
    assert outcomes[0].status == "failed" and "final_test" in outcomes[0].reason


def test_a_file_with_no_carried_stat_is_saved_without_a_marker_and_says_so(db, caplog):
    """Review m-4. stat=None keeps its one meaning -- the file could not be statted: its rows are
    saved and no processed marker is, so the next run offers it again (the golden pins this for a
    file gone before its save). WARNED, not refused: an ERROR result for a file that could not be
    statted legitimately has no stat and has always kept its row, and refusing would also split
    save_batch from save_analysis for a gone file. But once the ingest carries the parse's own stat,
    any other None is a caller that dropped it -- a file re-parsed every run -- and must be heard."""
    import logging
    from laser_trim_analyzer.database.manager import TrimWrite
    items = [TrimWrite(_parsed(FOUR[0]), None, None)] + _items(db, FOUR[1:2])
    with caplog.at_level(logging.WARNING, logger="laser_trim_analyzer.database.manager"):
        outcomes = db.write_batch(items)
    assert [o.status for o in outcomes] == ["saved", "saved"]
    warned = [r.getMessage() for r in caplog.records
              if r.name == "laser_trim_analyzer.database.manager" and r.levelno == logging.WARNING]
    assert len(warned) == 1 and FOUR[0] in warned[0] and "marker" in warned[0], warned
    rows = _rows_by_file(db.database_path)
    assert rows[FOUR[0]][0] == 1 and rows[FOUR[0]][4] == 0, "rows saved, no marker"
    assert rows[FOUR[1]][4] == 1


# ---- every stored value: the golden, through write_batch and save_batch --------------------------

@pytest.mark.parametrize("per_batch", [1, 4, 20])
def test_write_batch_stores_exactly_todays_rows(db, tmp_path, monkeypatch, per_batch):
    """The whole golden scenario (every kind, both update paths, two paths with one content hash)
    through write_batch -- one file per batch, four, and all of it in ONE transaction, where a
    later file updates rows an earlier one wrote in the same, still-uncommitted transaction.

    Every write_batch call runs inside the file-I/O trap (ruling 10, review m-2): handed the carried
    values, the whole batch -- savepoints, outcomes, the tripwire, the guard -- touches no file. Its
    rows are compared EXACTLY with save_analysis's in this process, then with the golden."""
    from laser_trim_analyzer.database.manager import TrimWrite
    steps = save_rows.build_scenario(tmp_path)
    reference = save_rows.reference_snapshot(steps, tmp_path)
    items = [TrimWrite(r, *db._file_identity(r.metadata.file_path)) for _, r in steps]
    outcomes = []
    with save_rows.no_file_io(monkeypatch) as touched:
        for i in range(0, len(items), per_batch):
            outcomes += db.write_batch(items[i:i + per_batch])
    assert touched == [], f"write_batch touched the file system: {touched}"
    assert all(o.status == "saved" for o in outcomes), outcomes
    ids = [(label, o.row_id) for (label, _), o in zip(steps, outcomes)]
    snap = save_rows.snapshot(Path(db.database_path), tmp_path, ids)
    save_rows.assert_same_rows(snap, reference, f"write_batch, {per_batch} per batch")
    save_rows.assert_matches_golden(snap)


def test_save_batch_stores_exactly_what_save_analysis_stores(db, tmp_path):
    """Ruling 8: save_batch -- no callers, kept for its signature -- is rebuilt on write_batch, so it
    writes trim_passes and trim_setup as save_analysis does (the parked C2 finding: it never did),
    and a final-test or smoothness result passes through exactly as save_analysis returns it."""
    steps = save_rows.build_scenario(tmp_path)
    reference = save_rows.reference_snapshot(steps, tmp_path)
    ids = db.save_batch([r for _, r in steps])
    assert ids == [rid for _, rid in save_rows.load_golden()["ids"]]
    labelled = [(label, rid) for (label, _), rid in zip(steps, ids)]
    snap = save_rows.snapshot(Path(db.database_path), tmp_path, labelled)
    save_rows.assert_same_rows(snap, reference, "save_batch")
    save_rows.assert_matches_golden(snap)
    ft, sm = _parsed(FOUR[0]), _parsed(FOUR[1])
    ft.file_type, ft.final_test_id = "final_test", 77
    sm.file_type, sm.smoothness_id = "smoothness", None
    assert db.save_batch([ft, sm]) == [db.save_analysis(ft), db.save_analysis(sm)] == [77, -1]
