"""Design doc 2026-09-23 ruling 5b/5c: 'Unknown' is the parser's sentinel, not a
model name, and the Integer trim_setup columns must hold integers, never floats
(both invented example data -- model "M" and the raw fixtures below)."""
import sqlalchemy as sa

from laser_trim_analyzer.database import manager as mgr
import laser_trim_analyzer.database as dbpkg
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAnalysisResult, SystemType, StatusType, TrimSetup)


def _wired_db(tmp_path, monkeypatch):
    """A DatabaseManager on an isolated tmp file, injected into BOTH module
    globals per the task's global constraints."""
    db = DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)
    return db


def test_unknown_is_not_a_known_model(tmp_path, monkeypatch):
    db = _wired_db(tmp_path, monkeypatch)
    with db.session() as session:
        session.add(DBAnalysisResult(
            filename="unknown_sample.xls", model="Unknown", serial="1",
            system=SystemType.A, overall_status=StatusType.PASS))
        session.add(DBAnalysisResult(
            filename="known_sample.xls", model="M", serial="2",
            system=SystemType.A, overall_status=StatusType.PASS))

    assert db.get_known_models() == {"M"}


def test_integer_columns_get_integers(tmp_path, monkeypatch):
    db = _wired_db(tmp_path, monkeypatch)
    with db.session() as session:
        row = DBAnalysisResult(
            filename="setup_sample.xls", model="M", serial="3",
            system=SystemType.A, overall_status=StatusType.PASS)
        session.add(row)
        session.flush()
        analysis_id = row.id
        db._write_trim_setup(session, analysis_id, {
            "initial_points_ignored": 2.0,
            "ending_points_ignored": 7.5,
        })

    with db.session() as session:
        setup = session.query(TrimSetup).filter_by(analysis_id=analysis_id).one()
        assert setup.points_ignored_start == 2
        assert setup.points_ignored_end is None
        # The raw values are never lost, fractional or not.
        assert setup.parameters == {
            "initial_points_ignored": 2.0,
            "ending_points_ignored": 7.5,
        }
        typeof_start, typeof_end = session.execute(sa.text(
            "SELECT typeof(points_ignored_start), typeof(points_ignored_end) "
            "FROM trim_setup WHERE analysis_id = :aid"),
            {"aid": analysis_id}).one()
    assert typeof_start == "integer"
    assert typeof_end == "null"
