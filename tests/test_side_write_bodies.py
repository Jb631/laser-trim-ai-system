"""Final-test, smoothness and skip-marker saves write into the session they are given
(ingest-speed Task 7; spec 3.3, ruling 9).

Each write an ingest worker makes today besides the trim save -- `save_final_test`,
`save_smoothness_result`, `mark_file_skipped` -- gets a body that writes into a session its CALLER
owns, with no commit, no transaction-level rollback and no session of its own inside it: a body
that committed would commit the whole batch it runs in (spec F5), and write_batch's guard savepoint
turns any commit that slipped through into BatchCommitError. The public methods keep their own
session and commit, with today's behaviour. Two of them committed inside themselves today
(`save_final_test` three sessions deep, through `apply_final_test_regrade`, `mark_file_skipped`
and `_refresh_final_test_identity`); their never-fatal parts now run in savepoints of their own.

The rows are compared WHOLE with the golden captured from today's public methods before any body
moved, and every path in one process EXACTLY with the public methods run beside it (side_writes.py).
"""
from pathlib import Path

import pytest

import save_rows
import side_writes


@pytest.fixture
def db(tmp_path, monkeypatch):
    from laser_trim_analyzer.database.manager import DatabaseManager
    d = DatabaseManager(tmp_path / "side.db")
    save_rows.inject(d, monkeypatch)
    yield d
    d.close()


def _snap(db, root, ids):
    return save_rows.snapshot(Path(db.database_path), root, ids, side_writes.SIDE_TABLES)


# ---- the reference: today's rows, whole ---------------------------------------------------------

def test_the_public_saves_store_exactly_todays_rows(db, tmp_path):
    """Every final-test format, smoothness and marker branch through the PUBLIC methods, compared
    column by column with the golden captured from the code BEFORE the bodies moved. With
    LTA_REGENERATE_SAVE_ROWS=1 it (re)writes that golden instead."""
    steps = side_writes.build_side_scenario(tmp_path, db)
    snap = _snap(db, tmp_path, side_writes.run_public(db, steps))
    if save_rows.REGENERATE:
        save_rows.write_golden(snap, side_writes.SIDE_GOLDEN)
        pytest.skip(f"regenerated {side_writes.SIDE_GOLDEN}")
    save_rows.assert_matches_golden(snap, side_writes.SIDE_GOLDEN)


# ---- the bodies, inside write_batch -------------------------------------------------------------

def _items(db, steps):
    from laser_trim_analyzer.database.manager import (
        FinalTestWrite, SkipMarkerWrite, SmoothnessWrite, TrimWrite)
    out = []
    for _, kind, kw in steps:
        if kind == "trim":
            a = kw["analysis"]
            out.append(TrimWrite(a, *db._file_identity(a.metadata.file_path)))
        elif kind == "final_test":
            out.append(FinalTestWrite(**kw))
        elif kind == "smoothness":
            out.append(SmoothnessWrite(**kw))
        else:
            out.append(SkipMarkerWrite(**kw))
    return out


def _what_the_public_path_said(public_value):
    """The outcome a batch owes each step, from what its public method returned."""
    if isinstance(public_value, str) and public_value.startswith("raised "):
        return "failed"
    return "duplicate" if public_value == -1 else "saved"


@pytest.mark.parametrize("per_batch", [1, 4, 30])
def test_the_bodies_inside_write_batch_store_exactly_the_public_rows(db, tmp_path, monkeypatch,
                                                                    per_batch):
    """Every step as a write_batch item -- one per batch, four, all in ONE transaction, where later
    files re-grade, re-mark and refresh rows the earlier ones wrote in the same uncommitted
    transaction. Handed the carried values, not one touches a file; not one trips the batch's guard
    (a commit inside a body would raise BatchCommitError here); and the rows are EXACTLY those the
    public methods store beside them in this process, and the golden's."""
    steps = side_writes.build_side_scenario(tmp_path, db)
    public = side_writes.public_snapshot(steps, tmp_path)
    items = _items(db, steps)
    outcomes = []
    with save_rows.no_file_io(monkeypatch) as touched:
        for i in range(0, len(items), per_batch):
            outcomes += db.write_batch(items[i:i + per_batch])
    assert touched == [], f"a body touched the file system: {touched}"
    said = [value for _, value in public["ids"]]
    assert [o.status for o in outcomes] == [_what_the_public_path_said(v) for v in said]
    for o, value in zip(outcomes, said):
        if o.status == "saved" and value is not None:
            assert o.row_id == value, (o, value)
    assert "Serial cannot be empty" in outcomes[6].reason
    ids = [(label, value) for (label, _, _), value in zip(steps, said)]
    snap = _snap(db, tmp_path, ids)
    save_rows.assert_same_rows(snap, public, f"the bodies, {per_batch} per batch",
                               "the public methods")
    save_rows.assert_matches_golden(snap, side_writes.SIDE_GOLDEN)


def test_one_batch_of_every_kind_is_one_transaction(db, tmp_path):
    """At every savepoint the batch opens -- its files' and the bodies' own (the insert that may
    hit the UNIQUE identity, the never-fatal marker, re-grade and identity refresh) -- a second
    connection can neither write nor see a single final test, and the batch commits exactly once."""
    import sqlite3
    from sqlalchemy import event
    steps = side_writes.build_side_scenario(tmp_path, db)
    items = _items(db, steps)
    other = sqlite3.connect(str(db.database_path), timeout=0)
    seen, commits = [], []

    def at_savepoint(conn, name):
        try:
            other.execute("BEGIN IMMEDIATE")
            other.execute("ROLLBACK")
            seen.append("could write")
        except sqlite3.OperationalError as e:
            seen.append(str(e))
        seen.append(other.execute("SELECT COUNT(*) FROM final_test_results").fetchone()[0])

    event.listen(db._engine, "savepoint", at_savepoint)
    event.listen(db._engine, "commit", lambda conn: commits.append(1))
    try:
        db.write_batch(items)
    finally:
        event.remove(db._engine, "savepoint", at_savepoint)
        other.close()
    assert len(seen) > 2 * len(items), "the bodies opened no savepoints of their own"
    assert set(seen) == {"database is locked", 0}, seen
    assert commits == [1]


def test_a_commit_inside_a_never_fatal_part_still_cannot_commit_the_batch(db, tmp_path,
                                                                          monkeypatch):
    """The never-fatal parts of the final-test save (the duplicate path's marker, the re-grade, the
    identity refresh) swallow their own failures, as they always did. So a commit that crept into
    one would not raise there -- and must still not commit the batch: the guard savepoint is gone
    with the transaction, and the batch raises instead."""
    from laser_trim_analyzer.database.manager import BatchCommitError
    steps = side_writes.build_side_scenario(tmp_path, db)
    real = db._mark_file_skipped_in

    def commits_first(session, **kw):
        session.commit()                       # what a body must never do
        return real(session, **kw)

    monkeypatch.setattr(db, "_mark_file_skipped_in", commits_first)
    with pytest.raises(BatchCommitError):
        db.write_batch(_items(db, steps))


def test_a_failed_regrade_costs_only_itself_on_both_paths(db, tmp_path, monkeypatch):
    """Never fatal, as before: when the re-grade of a stored final test fails -- AFTER it has
    written (its rows flushed, then an error) -- the save still returns that row's id, keeps what it
    had already written (the legacy row's stat, the other path's marker) and keeps NOTHING of the
    re-grade: through the public method and inside a batch alike, EXACTLY the same rows."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    steps = side_writes.build_side_scenario(tmp_path, db)
    real = DatabaseManager._write_final_test_regrade

    def writes_then_fails(self, session, final_test_id, tracks, test_results):
        real(self, session, final_test_id, tracks, test_results)
        session.flush()                          # the re-grade's rows are in the transaction...
        raise RuntimeError("invented failure after a re-grade wrote")      # ...then it fails

    monkeypatch.setattr(DatabaseManager, "_write_final_test_regrade", writes_then_fails)
    public = side_writes.public_snapshot(steps, tmp_path)
    outcomes = db.write_batch(_items(db, steps))
    said = [value for _, value in public["ids"]]
    assert said[9] == 6 and said[10] == 2, "the duplicate saves still return the stored row"
    assert [o.status for o in outcomes] == [_what_the_public_path_said(v) for v in said]
    snap = _snap(db, tmp_path, [(label, v) for (label, _, _), v in zip(steps, said)])
    save_rows.assert_same_rows(snap, public, "a batch whose re-grade fails", "the public methods")
    t = snap["tables"]
    assert [r for r in t["final_test_tracks"] if r["final_test_id"] == 6] == [], \
        "the header-only row kept NO tracks: the re-grade that wrote them failed"
    row6 = next(r for r in t["final_test_results"] if r["id"] == 6)
    assert row6["file_size"] == "<size of file>", "the stat stamped before the re-grade stays"
    assert any("same content as final_test_results id 2" in (r["error_message"] or "")
               for r in t["processed_files"]), "the other path's marker stays"


def test_a_failed_duplicate_path_marker_costs_only_the_marker(db, tmp_path, monkeypatch, caplog):
    """The other path's marker is never fatal either: one that fails after writing leaves no
    marker row, and the save goes on to re-grade and return the stored row -- on both paths. And
    it SAYS so, at WARNING, naming the file and the cause (review m-1): unmarked, the path is read
    again on every run."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    steps = side_writes.build_side_scenario(tmp_path, db)
    real = DatabaseManager._mark_file_skipped_in

    def writes_then_fails(self, session, **kw):
        real(self, session, **kw)                # the marker row is in the transaction...
        if "same content as" in (kw.get("error_message") or ""):
            raise RuntimeError("invented failure after the marker wrote")   # ...then it fails

    monkeypatch.setattr(DatabaseManager, "_mark_file_skipped_in", writes_then_fails)
    with caplog.at_level("WARNING"):
        public = side_writes.public_snapshot(steps, tmp_path)
        outcomes = db.write_batch(_items(db, steps))
    warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"
              and "could not mark this path as holding content already on record" in r.getMessage()]
    assert len(warned) == 2 and all("invented failure after the marker wrote" in m
                                    and ".xls" in m for m in warned), warned    # both paths
    said = [value for _, value in public["ids"]]
    assert said[10] == 2 and outcomes[10].status == "saved" and outcomes[10].row_id == 2
    snap = _snap(db, tmp_path, [(label, v) for (label, _, _), v in zip(steps, said)])
    save_rows.assert_same_rows(snap, public, "a batch whose marker fails", "the public methods")
    assert not any("same content as" in (r["error_message"] or "")
                   for r in snap["tables"]["processed_files"]), "the failed marker left a row"
    golden = save_rows.load_golden(side_writes.SIDE_GOLDEN)["tables"]
    assert ([r for r in snap["tables"]["final_test_tracks"] if r["final_test_id"] == 2]
            == [r for r in golden["final_test_tracks"] if r["final_test_id"] == 2]), \
        "the re-grade after the failed marker still ran"


@pytest.mark.parametrize("public", ["save_final_test", "save_smoothness_result",
                                    "mark_file_skipped"])
def test_a_public_save_inside_a_batch_is_refused(db, tmp_path, monkeypatch, public):
    """Ruling 6: inside a batch every write takes the batch's session. The public methods open
    their own, so one called from a batch body is refused (NestedSessionError) -- its file fails,
    it commits nothing."""
    from laser_trim_analyzer.database.manager import NestedSessionError
    steps = side_writes.build_side_scenario(tmp_path, db)
    kind = {"save_final_test": "final_test", "save_smoothness_result": "smoothness",
            "mark_file_skipped": "marker"}[public]
    kw = next(k for _, kd, k in steps if kd == kind)
    items = _items(db, steps[:1])
    raised = []
    real = db._write_trim_setup

    def calls_the_public_method(session, analysis_id, setup):
        try:
            getattr(db, public)(**kw)
        except Exception as e:
            raised.append(e)
            raise
        return real(session, analysis_id, setup)

    monkeypatch.setattr(db, "_write_trim_setup", calls_the_public_method)
    outcomes = db.write_batch(items)
    assert len(raised) == 1 and isinstance(raised[0], NestedSessionError), raised
    assert [o.status for o in outcomes] == ["failed"]


def test_a_failed_identity_refresh_costs_only_the_refresh(db, tmp_path, caplog):
    """The identity refresh of a file re-exported in place is never fatal: when its UPDATE fails
    at the flush, the save still returns the stored row, leaves that row's identity as it was, and
    everything else commits -- on both paths. (Without its savepoint the failed flush would poison
    the session, and the whole save would fail.)"""
    from sqlalchemy import event
    from laser_trim_analyzer.database.models import FinalTestResult
    steps = side_writes.build_side_scenario(tmp_path, db)
    refreshed = side_writes._sha("re-exported in place")

    def refuse_the_refresh(mapper, connection, target):
        if target.file_hash == refreshed:
            raise RuntimeError("invented failure writing the refreshed identity")

    event.listen(FinalTestResult, "before_update", refuse_the_refresh)
    try:
        with caplog.at_level("WARNING"):
            public = side_writes.public_snapshot(steps, tmp_path)
            outcomes = db.write_batch(_items(db, steps))
    finally:
        event.remove(FinalTestResult, "before_update", refuse_the_refresh)
    # ...and it is SAID, at WARNING, naming the file and the cause, on both paths (review m-1).
    warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"
              and "could not refresh its identity" in r.getMessage()]
    assert len(warned) == 2 and all("invented failure writing the refreshed identity" in m
                                    and ".xls" in m for m in warned), warned
    said = [value for _, value in public["ids"]]
    assert said[11] == 3 and outcomes[11].status == "saved" and outcomes[11].row_id == 3
    snap = _snap(db, tmp_path, [(label, v) for (label, _, _), v in zip(steps, said)])
    save_rows.assert_same_rows(snap, public, "a batch whose identity refresh fails",
                               "the public methods")
    row3 = next(r for r in snap["tables"]["final_test_results"] if r["id"] == 3)
    assert row3["file_hash"] == "<sha256 of file>", "the refresh that failed changed the identity"
