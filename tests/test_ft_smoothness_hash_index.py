"""`file_hash` indexes on final_test_results and smoothness_results (spec 3.7, ruling 11).

Every final-test and smoothness save looks up its own content hash before deciding
insert vs. duplicate/upsert (save_final_test, save_smoothness_result) -- and so does
is_file_processed and the stat-heal pass (update_processed_file_stats). Without an
index SQLite SCANs the whole table for every one of those lookups: F9 measured the
final-test save at 22-24 ms/file on a table of 151,793 rows -- 20-21 of it SQL -- and
3.4-3.7 once indexed; the shipped FT loop went 30 -> 18 ms/file.

The migration lives beside the app's other `CREATE INDEX IF NOT EXISTS` statements in
`DatabaseManager._run_migrations` (main:~876), which already runs, idempotently, on
every launch.
"""
import logging
import os
import sqlite3
import stat
from datetime import datetime

import pytest


def _meta(model="6607", serial="s1"):
    return {
        "filename": f"{model}_{serial}_FINAL TEST.xls",
        "file_path": "/x",
        "model": model,
        "serial": serial,
        "file_date": datetime(2026, 8, 1),
        "test_date": datetime(2026, 8, 1),
    }


def _explain(path, table, file_hash="ab" * 32):
    """The EXPLAIN QUERY PLAN for exactly the shape the ORM runs -- `.filter(file_hash
    == ...).first()` is `SELECT * FROM <table> WHERE file_hash = ? LIMIT 1` -- so the
    plan text matches what the app actually asks SQLite to do, not a narrower stand-in.
    """
    raw = sqlite3.connect(str(path))
    try:
        rows = raw.execute(
            f"EXPLAIN QUERY PLAN SELECT * FROM {table} WHERE file_hash = ? LIMIT 1",
            (file_hash,),
        ).fetchall()
    finally:
        raw.close()
    return " | ".join(str(r) for r in rows)


def test_final_test_hash_lookup_uses_the_index_not_a_scan(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager

    path = tmp_path / "x.db"
    db = DatabaseManager(path)
    db.close()
    plan = _explain(path, "final_test_results")
    assert "SCAN" not in plan.upper(), plan
    assert "SEARCH FINAL_TEST_RESULTS USING INDEX IDX_FT_FILE_HASH" in plan.upper(), plan


def test_smoothness_hash_lookup_uses_the_index_not_a_scan(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager

    path = tmp_path / "x.db"
    db = DatabaseManager(path)
    db.close()
    plan = _explain(path, "smoothness_results")
    assert "SCAN" not in plan.upper(), plan
    assert "SEARCH SMOOTHNESS_RESULTS USING INDEX IDX_SMOOTHNESS_FILE_HASH" in plan.upper(), plan


def test_the_index_migration_running_twice_is_a_no_op(tmp_path):
    """A second app launch against the same file re-runs `_run_migrations` -- the
    `CREATE INDEX IF NOT EXISTS` must not error and must not duplicate the index.
    """
    from laser_trim_analyzer.database.manager import DatabaseManager

    path = tmp_path / "x.db"
    db1 = DatabaseManager(path)
    db1.close()
    db2 = DatabaseManager(path)  # second launch
    db2.close()

    raw = sqlite3.connect(str(path))
    try:
        rows = raw.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name IN ('idx_ft_file_hash', 'idx_smoothness_file_hash')"
        ).fetchall()
    finally:
        raw.close()
    assert sorted(r[0] for r in rows) == ["idx_ft_file_hash", "idx_smoothness_file_hash"]


def test_final_test_and_smoothness_saved_rows_are_unchanged_by_the_index(tmp_path):
    """The index changes the QUERY PLAN, never the DATA: same duplicate detection,
    same upsert behaviour, same row counts as before the migration existed.
    """
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import FinalTestResult, SmoothnessResult

    db = DatabaseManager(tmp_path / "x.db")
    try:
        h = "cd" * 32
        id1 = db.save_final_test(metadata=_meta(), tracks=[], test_results={}, file_hash=h)
        id2 = db.save_final_test(metadata=_meta(), tracks=[], test_results={}, file_hash=h)
        assert id1 == id2  # the same content hash is recognised as the same row
        with db.session() as session:
            assert session.query(FinalTestResult).count() == 1
            row = session.query(FinalTestResult).one()
            assert row.file_hash == h and row.model == "6607" and row.serial == "s1"

        sh = "ef" * 32
        sid1 = db.save_smoothness_result(metadata=_meta(), tracks=[], file_hash=sh)
        sid2 = db.save_smoothness_result(metadata=_meta(), tracks=[], file_hash=sh)
        assert sid1 == sid2
        with db.session() as session:
            assert session.query(SmoothnessResult).count() == 1
            row = session.query(SmoothnessResult).one()
            assert row.file_hash == sh and row.model == "6607" and row.serial == "s1"
    finally:
        db.close()


def test_the_index_migration_on_a_readonly_pre_task3_database_is_skipped_and_logged(tmp_path, caplog):
    """James's real first launch: a database that predates this branch -- no idx_ft_file_hash or
    idx_smoothness_file_hash yet -- opened from a file he cannot write (CLAUDE.md: production
    data/analysis.db is chmod'd read-only on purpose; the containing folder stays writable, which
    is what lets SQLite still create WAL-mode's -shm/-wal lock-coordination files).

    The migration's CREATE INDEX statements fail ("attempt to write a readonly database"). The
    except block must roll the session back -- the same idiom as every sibling migration in
    _run_migrations -- log a WARNING (not swallow it, not crash), and leave the manager fully
    usable: later migrations in the same run still get their turn, and the app opens.
    """
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import FinalTestResult

    path = tmp_path / "readonly.db"
    db = DatabaseManager(path)  # first launch: creates the schema, both new indexes included
    db.close()

    # Simulate a database that predates Task 3: drop the two new indexes, matching what a real
    # pre-pull database looks like.
    raw = sqlite3.connect(str(path))
    raw.execute("DROP INDEX IF EXISTS idx_ft_file_hash")
    raw.execute("DROP INDEX IF EXISTS idx_smoothness_file_hash")
    raw.commit()
    raw.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # no pending WAL frames before going read-only
    raw.close()

    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # file read-only, folder untouched
    try:
        with caplog.at_level(logging.WARNING, logger="laser_trim_analyzer.database.manager"):
            db2 = DatabaseManager(path)  # must not raise
        index_warnings = [r.message for r in caplog.records if "Index migration warning" in r.message]
        assert index_warnings, (
            f"the read-only failure must be logged, not swallowed: {[r.message for r in caplog.records]}")
        assert "readonly database" in index_warnings[0]

        # The session is not left unusable -- a later migration in the same run still logs its
        # own attempt rather than silently never running (the "LTS3 retag" migration, further down
        # _run_migrations, also writes unconditionally on every launch).
        assert any("LTS3 retag migration warning" in r.message for r in caplog.records), (
            "a later migration going silent would mean the session was left poisoned by the "
            "unrolled-back index migration failure")

        # And the manager itself is fully usable afterward -- this is the actual promise:
        # a read-only database does not stop the app from opening and reading.
        with db2.session() as session:
            assert session.query(FinalTestResult).count() == 0
        db2.close()
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
