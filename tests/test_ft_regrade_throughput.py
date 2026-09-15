"""The re-grade pass has to finish, and a half-finished one has to be useful.

Work incident, 2026-09-14. The pass started at 14:19 and was still going at
15:35. It walked the 151,375 rows in `id` order — ingest order, which is
OLDEST FIRST — so it spent its first ninety minutes on 2010-era workbooks while
every number the app shows comes from the last year or two. It wrote one
transaction per row against a 3.7 GB database, and measured about 80 rows a
minute: roughly thirty hours.

Three things are pinned here, and each one was made to fail against the code
as it stood that afternoon:

  * ORDER — FAIL rows first, newest first. FAIL is the verdict the window fix
    can actually move, and recent rows are the ones the dashboards read, so
    stopping at 5pm leaves something worth having.
  * BATCHED WRITES — 200 rows per transaction, not one per row, with the SAME
    result in the database either way.
  * PROGRESS — a rate and an ETA, so "is this going to finish tonight?" is a
    question the screen answers.
"""
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import text

from laser_trim_analyzer.core.final_test_parser import FinalTestParser
from laser_trim_analyzer.core.ft_regrade import regrade_final_tests

FIXTURES = Path(__file__).parent / "fixtures" / "final_test"
SN180 = FIXTURES / "8232-1-sn180_3-28-2026_9-37 AM.xls"


@pytest.fixture(scope="module")
def parser():
    return FinalTestParser()


def _manager(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    return DatabaseManager(tmp_path / "ft.db")


def _seed(db, parser, *, n, path=SN180):
    """N stored final tests, all marked legacy, all pointing at one real file.

    Same source file for every row on purpose: what is under test is the
    ORDER, the TRANSACTIONS and the PROGRESS, not the grading — the grading
    has its own suite (test_ft_graded_window.py) and one body grades both.
    """
    parsed = parser.parse_file(path)
    ids = []
    for i in range(n):
        metadata = dict(parsed["metadata"])
        metadata["file_path"] = str(path)
        metadata["serial"] = f"sn{i:04d}"
        ids.append(db.save_final_test(
            metadata=metadata, tracks=parsed["tracks"],
            test_results=parsed["test_results"],
            file_hash=f"{parsed['file_hash']}-{i}"))
    with db.session() as session:
        session.execute(text(
            "UPDATE final_test_results SET graded_window_source = NULL"))
        session.commit()
    return ids


# ---- (1) the order a partial run leaves behind -----------------------------

def test_fail_rows_come_first_and_newest_first_within_each_group(tmp_path,
                                                                 parser):
    """The selection order IS the plan for a run that will be stopped early."""
    db = _manager(tmp_path)
    ids = _seed(db, parser, n=6)
    # id ASC is oldest-ingested first; file_date and verdict are set against
    # the grain of it, so a query that still sorts by id sorts wrong.
    #   id:            0     1     2     3     4     5
    #   year:        2011  2026  2013  2025  2010  2024
    #   verdict:     PASS  PASS  FAIL  FAIL  FAIL  NULL
    plan = [(2011, 1), (2026, 1), (2013, 0), (2025, 0), (2010, 0), (2024, None)]
    with db.session() as session:
        for ft_id, (year, verdict) in zip(ids, plan):
            session.execute(
                text("UPDATE final_test_results SET file_date = :d, "
                     "linearity_pass = :p WHERE id = :i"),
                {"d": datetime(year, 6, 1), "p": verdict, "i": ft_id})
        session.commit()

    rows = db.get_final_tests_for_regrade(only_legacy=True)
    years = [r["file_date"].year for r in rows]
    verdicts = [r["linearity_pass"] for r in rows]

    assert years == [2025, 2013, 2010, 2026, 2024, 2011], (
        f"wrong order: {list(zip(years, verdicts))}")
    assert verdicts[:3] == [False, False, False], "FAIL rows must lead"
    assert years[:3] == sorted(years[:3], reverse=True), "newest FAIL first"
    assert years[3:] == sorted(years[3:], reverse=True), "newest rest first"


def test_a_stopped_run_has_done_the_recent_failures(tmp_path, parser):
    """The point of the order, stated as the outcome it buys.

    Stop after one batch and the rows that got re-graded must be the recent
    FAILs — not a random slice of 2010.
    """
    db = _manager(tmp_path)
    ids = _seed(db, parser, n=6)
    plan = [(2011, 1), (2026, 1), (2013, 0), (2025, 0), (2010, 0), (2024, None)]
    with db.session() as session:
        for ft_id, (year, verdict) in zip(ids, plan):
            session.execute(
                text("UPDATE final_test_results SET file_date = :d, "
                     "linearity_pass = :p WHERE id = :i"),
                {"d": datetime(year, 6, 1), "p": verdict, "i": ft_id})
        session.commit()

    # Stop the moment the first batch has landed — the 5pm case.
    cancel = threading.Event()
    real_write = db.apply_final_test_regrades

    def write_then_stop(batch):
        written = real_write(batch)
        cancel.set()
        return written

    db.apply_final_test_regrades = write_then_stop
    report = regrade_final_tests(db, apply=True, only_legacy=True,
                                 batch_size=2, workers=2, cancel=cancel)
    assert report.cancelled
    with db.session() as session:
        graded = [r[0] for r in session.execute(text(
            "SELECT file_date FROM final_test_results "
            "WHERE graded_window_source IS NOT NULL")).all()]
    years = sorted(str(g)[:4] for g in graded)
    assert years == ["2013", "2025"], (
        f"a stopped run should have done the newest failures, did {years}")
    # And the rest are still marked legacy, so the next run picks them up.
    assert db.count_legacy_ft_verdicts() == 4


# ---- (2) one transaction per batch, same result ----------------------------

def test_a_batch_of_rows_is_written_in_one_transaction(tmp_path, parser,
                                                       monkeypatch):
    """151,375 commits against a 3.7 GB database is where the hours went."""
    from laser_trim_analyzer.database import manager as manager_mod

    db = _manager(tmp_path)
    _seed(db, parser, n=7)

    commits = []
    real_session = db.session

    class _CountingSession:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, item):
            return getattr(self._inner, item)

        def commit(self):
            commits.append(1)
            return self._inner.commit()

    import contextlib

    @contextlib.contextmanager
    def counting(*a, **kw):
        with real_session(*a, **kw) as s:
            yield _CountingSession(s)

    monkeypatch.setattr(db, "session", counting)
    report = regrade_final_tests(db, apply=True, only_legacy=True,
                                 batch_size=200, workers=2)
    assert report.examined == 7
    assert len(commits) == 1, (
        f"7 rows should be ONE transaction, took {len(commits)}")
    assert db.count_legacy_ft_verdicts() == 0


def test_batched_and_per_row_writes_leave_the_same_database(tmp_path, parser):
    """The batch writer must not be a second, subtly different writer.

    Both paths go through `_write_final_test_regrade`; this is the check that
    says so in terms of the stored rows rather than in terms of the code.
    """
    from laser_trim_analyzer.database.manager import DatabaseManager

    def _snapshot(db):
        with db.session() as session:
            rows = session.execute(text(
                "SELECT serial, linearity_pass, overall_status, "
                "       graded_window_source, station_linearity_pass, "
                "       round(linearity_error, 9) "
                "FROM final_test_results ORDER BY serial")).all()
            tracks = session.execute(text(
                "SELECT f.serial, t.track_id, t.linearity_pass, "
                "       t.linearity_fail_points, t.graded_start, t.graded_end, "
                "       round(t.optimal_offset, 9) "
                "FROM final_test_tracks t JOIN final_test_results f "
                "ON f.id = t.final_test_id ORDER BY f.serial, t.track_id")).all()
        return [tuple(r) for r in rows], [tuple(r) for r in tracks]

    batched = DatabaseManager(tmp_path / "batched.db")
    _seed(batched, parser, n=5)
    regrade_final_tests(batched, apply=True, only_legacy=True, batch_size=200)

    one_at_a_time = DatabaseManager(tmp_path / "single.db")
    _seed(one_at_a_time, parser, n=5)
    regrade_final_tests(one_at_a_time, apply=True, only_legacy=True,
                        batch_size=1)

    assert _snapshot(batched) == _snapshot(one_at_a_time)


def test_the_batch_writer_chunks_at_two_hundred(tmp_path):
    """The transaction is bounded even when the caller hands over more.

    Driven straight at the manager with synthetic payloads: the point is the
    chunking rule, and 450 real workbooks is not a unit test.
    """
    from laser_trim_analyzer.database import manager as manager_mod

    db = _manager(tmp_path)
    assert manager_mod.REGRADE_WRITE_CHUNK == 200

    seen = []

    def fake_write(session, ft_id, tracks, results):
        seen.append((id(session), ft_id))
        return True

    db._write_final_test_regrade = fake_write
    written = db.apply_final_test_regrades(
        [(i, [], {}) for i in range(450)])
    assert written == 450
    sessions = []
    for sess, _ in seen:
        if not sessions or sessions[-1] != sess:
            sessions.append(sess)
    assert len(sessions) == 3, f"450 rows should be 3 chunks, was {len(sessions)}"


# ---- (3) the rate and the ETA ----------------------------------------------

def test_progress_reports_a_rate_and_an_eta(tmp_path, parser):
    db = _manager(tmp_path)
    _seed(db, parser, n=4)
    seen = []

    def progress(done, total, name, rate, eta):
        seen.append((done, total, name, rate, eta))

    regrade_final_tests(db, apply=True, only_legacy=True, batch_size=2,
                        workers=2, progress=progress)
    assert seen, "no progress at all"
    done, total, name, rate, eta = seen[-1]
    assert (done, total) == (4, 4)
    assert isinstance(name, str) and name
    assert rate is None or isinstance(rate, float)
    assert isinstance(eta, str)
    # Nothing left to predict, so nothing is claimed.
    assert eta == ""


def test_the_progress_line_says_the_same_thing_to_both_front_ends():
    from laser_trim_analyzer.core.ft_regrade import format_regrade_line
    assert (format_regrade_line(12480, 151375, 2.6, "about 14 h 50 min left")
            == "12,480 of 151,375 · 2.6 files/s · about 14 h 50 min left")
    # Unknown rate and unknown ETA are DROPPED, never printed as zero: "0.0
    # files/s" in the first seconds of a long job reads as a stuck run.
    assert format_regrade_line(3, 151375, None, "") == "3 of 151,375"
    assert (format_regrade_line(3, 151375, None, "estimating…")
            == "3 of 151,375 · estimating…")


def test_progress_is_rate_limited_not_per_file(tmp_path, parser):
    """Eight workers over a share can finish several files a second, and this
    callback repaints a label. It is sent every ~2 s, plus once at the end."""
    db = _manager(tmp_path)
    _seed(db, parser, n=6)
    calls = []
    regrade_final_tests(
        db, apply=True, only_legacy=True, batch_size=3, workers=3,
        progress=lambda *a: calls.append(a))
    assert 1 <= len(calls) <= 2, (
        f"{len(calls)} progress calls for 6 fast files — is it per-file?")
    assert calls[-1][0] == 6, "the final call must report the final count"


# ---- (4) what did not change ----------------------------------------------

def test_workers_default_to_eight():
    import inspect
    from laser_trim_analyzer.core import ft_regrade
    sig = inspect.signature(ft_regrade.regrade_final_tests)
    assert sig.parameters["workers"].default == 8


def test_cancel_before_the_first_row_writes_nothing(tmp_path, parser):
    db = _manager(tmp_path)
    _seed(db, parser, n=4)
    cancel = threading.Event()
    cancel.set()
    report = regrade_final_tests(db, apply=True, only_legacy=True,
                                 cancel=cancel)
    assert report.cancelled
    assert report.examined == 0
    assert db.count_legacy_ft_verdicts() == 4, "a stopped run must be resumable"


# ---- (5) the log, which is what survives the window being closed -----------
#
# 2026-09-15: the re-grade ran 08:18 -> 14:49 on the work machine and left NO
# record of it — not how many rows it did, not how fast, not whether it
# finished or was stopped. Six and a half hours of a 3.7 GB database being
# rewritten, and the only evidence it happened at all was the row counts
# afterwards. These pin the three lines that fix that.

def _regrade_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == "laser_trim_analyzer.core.ft_regrade"
            and r.levelname == "INFO"
            and r.getMessage().startswith("Re-grade")]


def _seed_a_mixed_batch(db, parser, tmp_path):
    """7 rows: 5 gradeable, 1 whose file is gone, 1 the parser chokes on.

    Two of the gradeable rows are given a stored FAIL the file does not
    support, so the run has a real FAIL->PASS count to print.
    """
    ids = _seed(db, parser, n=7)
    broken = tmp_path / "broken.xls"
    broken.write_bytes(b"this is not a workbook")
    with db.session() as session:
        session.execute(text(
            "UPDATE final_test_results SET file_path = :p WHERE id = :i"),
            {"p": str(tmp_path / "gone" / "vanished.xls"), "i": ids[0]})
        session.execute(text(
            "UPDATE final_test_results SET file_path = :p WHERE id = :i"),
            {"p": str(broken), "i": ids[1]})
        session.execute(text(
            "UPDATE final_test_results SET linearity_pass = 0 "
            "WHERE id IN (:a, :b)"), {"a": ids[2], "b": ids[3]})
        session.commit()
    return ids


def test_the_run_logs_its_start_its_progress_and_its_end(tmp_path, parser,
                                                         caplog, monkeypatch):
    import logging
    from laser_trim_analyzer.core import ft_regrade

    db = _manager(tmp_path)
    _seed_a_mixed_batch(db, parser, tmp_path)
    # Every 500 rows in production; 2 here, so seven rows exercise the same
    # line the work run prints every few minutes.
    monkeypatch.setattr(ft_regrade, "REGRADE_LOG_EVERY", 2)

    caplog.set_level(logging.INFO, logger="laser_trim_analyzer.core.ft_regrade")
    report = regrade_final_tests(db, apply=True, only_legacy=True,
                                 batch_size=3, workers=2)
    lines = _regrade_lines(caplog)
    assert len(lines) >= 5, f"expected start + 3 progress + end, got {lines}"

    start = lines[0]
    assert start == ("Re-grade: 7 rows selected (FAIL-first, newest first), "
                     "workers=2, apply=yes"), start

    middle = lines[1:-1]
    assert len(middle) == 3, f"7 rows every 2 should log 3 times: {middle}"
    for line in middle:
        assert "/7" in line, line
        assert "changed PASS→FAIL" in line and "FAIL→PASS" in line, line

    # The end line carries the same totals as the report the caller gets —
    # the log and the screen can never disagree about how it went.
    end = lines[-1]
    moved = report.transitions()
    assert report.examined == 7
    assert moved["fail_to_pass"] == 2, "the two stored FAILs should be rescued"
    assert report.missing == 1 and report.errors == 1
    assert end.startswith("Re-grade finished: 7 of 7 rows"), end
    assert "wall " in end, end
    assert f"FAIL→PASS {moved['fail_to_pass']:,}" in end, end
    assert f"PASS→FAIL {moved['pass_to_fail']:,}" in end, end
    assert f"→NULL {moved['to_null']:,}" in end, end
    assert f"missing {report.missing:,}" in end, end
    assert f"errors {report.errors:,}" in end, end


def test_a_stopped_run_says_so_in_the_log(tmp_path, parser, caplog):
    import logging

    db = _manager(tmp_path)
    _seed(db, parser, n=4)
    cancel = threading.Event()
    real_write = db.apply_final_test_regrades

    def write_then_stop(batch):
        written = real_write(batch)
        cancel.set()
        return written

    db.apply_final_test_regrades = write_then_stop
    caplog.set_level(logging.INFO, logger="laser_trim_analyzer.core.ft_regrade")
    report = regrade_final_tests(db, apply=True, only_legacy=True,
                                 batch_size=2, workers=2, cancel=cancel)
    assert report.cancelled
    end = _regrade_lines(caplog)[-1]
    assert end.startswith("Re-grade cancelled: 2 of 4 rows"), end
    assert "wall " in end, end


def test_a_dry_run_says_it_is_a_dry_run(tmp_path, parser, caplog):
    import logging

    db = _manager(tmp_path)
    _seed(db, parser, n=2)
    caplog.set_level(logging.INFO, logger="laser_trim_analyzer.core.ft_regrade")
    regrade_final_tests(db, apply=False, only_legacy=True, workers=2)
    start = _regrade_lines(caplog)[0]
    assert "apply=no — dry run, nothing is written" in start, start
