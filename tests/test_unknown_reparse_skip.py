"""The Unknown-model re-parse migration stops announcing work it never does.

On the work database this migration re-parsed the same 381 filenames and fixed
0 of them on EVERY launch for months, printing two INFO lines each time. A
migration that has provably finished should cost nothing and say nothing, or
the startup log stops being something anyone reads.

The rule these tests pin down:
  - Unknown population UNCHANGED since the last attempt -> no re-parse work,
    nothing logged above DEBUG.
  - Unknown population CHANGED (rows arrived, or were fixed/deleted) -> one
    attempt, outcome reported at INFO.

A "launch" here is a fresh DatabaseManager on the same file, which is exactly
what starting the app does.
"""
import logging
from datetime import datetime

from laser_trim_analyzer.database.manager import DatabaseManager

MIGRATION_LOGGER = "laser_trim_analyzer.database.manager"

# Captured at import, before any test can patch it. `_launch` installs a
# counting spy that delegates here; without this the second `_launch` in a
# single test would chain onto the first test's spy (monkeypatch only unwinds
# at teardown) and count every call twice.
_REAL_REPARSE = DatabaseManager.__dict__["_reparse_filename"].__func__


def _add_unknown(session, tag):
    """One analysis_results row whose filename cannot yield a model.

    Single-part stem that is not a bare model number, so `_reparse_filename`
    falls through to ("Unknown", "Unknown") — the shape of all 381 real rows.
    """
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, SystemType, StatusType)
    when = datetime.now()
    session.add(DBAR(filename=f"junk{tag}.xls", file_path=f"/f/junk{tag}.xls",
                     file_hash=f"h{tag}", model="Unknown", serial="Unknown",
                     system=SystemType.A, file_date=when, timestamp=when,
                     overall_status=StatusType.PASS, has_multi_tracks=False,
                     processing_time=0.1))


def _seed(tmp_path, n, name="m.db"):
    """A database holding `n` unparseable Unknown rows, with the migration
    having already had its one look at them."""
    path = tmp_path / name
    db = DatabaseManager(path)
    with db.session() as s:
        for i in range(n):
            _add_unknown(s, i)
        s.commit()
    db.close()
    # The launch that first sees the rows: this one is expected to do the work.
    DatabaseManager(path).close()
    return path


def _launch(path, monkeypatch, caplog):
    """Open the database again; return (reparse_calls, INFO+ messages)."""
    calls = {"n": 0}

    def _counted(filename):
        calls["n"] += 1
        return _REAL_REPARSE(filename)

    monkeypatch.setattr(DatabaseManager, "_reparse_filename", staticmethod(_counted))
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=MIGRATION_LOGGER):
        DatabaseManager(path).close()
    loud = [r.getMessage() for r in caplog.records
            if r.name == MIGRATION_LOGGER and r.levelno >= logging.INFO
            and "Unknown" in r.getMessage()]
    return calls["n"], loud


def test_unchanged_unknown_count_does_no_work_and_says_nothing(tmp_path, monkeypatch, caplog):
    """The second launch in a row with the same population is a no-op."""
    path = _seed(tmp_path, 5)
    calls, loud = _launch(path, monkeypatch, caplog)
    assert calls == 0, f"re-parsed {calls} filenames it had already failed on"
    assert loud == [], f"announced itself again: {loud}"


def test_new_unknown_rows_make_it_try_again_and_report(tmp_path, monkeypatch, caplog):
    """Rows it has never seen are exactly what should wake it up."""
    path = _seed(tmp_path, 5)
    db = DatabaseManager(path)
    with db.session() as s:
        _add_unknown(s, "new")
        s.commit()
    db.close()

    calls, loud = _launch(path, monkeypatch, caplog)
    assert calls == 6, f"attempted {calls} of 6 Unknown filenames"
    assert any("Re-parsing 6" in m for m in loud), loud
    assert any("No Unknown records could be re-parsed" in m for m in loud), loud

    # ...and having tried the larger population, it settles again.
    calls2, loud2 = _launch(path, monkeypatch, caplog)
    assert calls2 == 0 and loud2 == [], (calls2, loud2)


def test_rows_leaving_the_pool_also_count_as_changed(tmp_path, monkeypatch, caplog):
    """The marker tracks the population, not merely 'have I ever run'.

    Rows leaving the Unknown pool (repaired by another tool, or deleted) change
    the count too, and the remaining filenames deserve one more attempt.
    """
    from sqlalchemy import text
    path = _seed(tmp_path, 5)
    db = DatabaseManager(path)
    with db.session() as s:
        s.execute(text("DELETE FROM analysis_results WHERE filename = 'junk0.xls'"))
        s.commit()
    db.close()

    calls, loud = _launch(path, monkeypatch, caplog)
    assert calls == 4, f"attempted {calls} of the 4 remaining Unknown filenames"
    assert loud, "a changed population must report at INFO"


def test_a_repaired_row_is_not_re_attempted_next_launch(tmp_path, monkeypatch, caplog):
    """When the re-parse actually fixes rows, the marker follows the REMAINDER.

    Guards the off-by-one that would store the pre-fix count and so re-run
    forever on a database the migration had just improved.
    """
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, SystemType, StatusType)
    path = _seed(tmp_path, 5)
    db = DatabaseManager(path)
    with db.session() as s:
        # A filename the improved parser CAN read: model 8340-1, serial 201.
        when = datetime.now()
        s.add(DBAR(filename="8340-1_201_data.xls", file_path="/f/ok.xls",
                   file_hash="hok", model="Unknown", serial="Unknown",
                   system=SystemType.A, file_date=when, timestamp=when,
                   overall_status=StatusType.PASS, has_multi_tracks=False,
                   processing_time=0.1))
        s.commit()
    db.close()

    calls, loud = _launch(path, monkeypatch, caplog)
    assert calls == 6
    assert any("Fixed 1 of 6" in m for m in loud), loud

    # 5 Unknown rows remain; the marker must already say 5, not 6.
    calls2, loud2 = _launch(path, monkeypatch, caplog)
    assert calls2 == 0, f"re-attempted {calls2} filenames after a successful pass"
    assert loud2 == [], loud2
