from laser_trim_analyzer.database.manager import DatabaseManager


def test_new_tables_are_created_on_a_fresh_database(tmp_path):
    db = DatabaseManager(tmp_path / "t.db")
    with db.session() as s:
        names = {r[0] for r in s.execute(
            __import__("sqlalchemy").text(
                "SELECT name FROM sqlite_master WHERE type='table'"))}
    assert "trim_passes" in names
    assert "trim_setup" in names


def test_trim_pass_columns_are_what_the_engine_needs(tmp_path):
    db = DatabaseManager(tmp_path / "t.db")
    import sqlalchemy as sa
    with db.session() as s:
        cols = {r[1] for r in s.execute(sa.text("PRAGMA table_info(trim_passes)"))}
    for c in ("track_result_id", "pass_index", "sheet", "positions", "errors",
              "upper_limits", "lower_limits", "laser_cut_length", "laser_speed_high",
              "trim_voltage", "cut_lengths", "trim_currents", "used_deltas"):
        assert c in cols, f"missing {c}"


# append to tests/test_trim_capture_db.py
from pathlib import Path
import sqlalchemy as sa
import pytest


def test_pipeline_writes_passes_and_setup(tmp_path, monkeypatch):
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.core.processor import Processor
    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    import laser_trim_analyzer.database as dbpkg
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)

    proc = Processor(use_ml=False)
    result = proc.process_file(Path("tests/fixtures/trim/dlts_8232-1_243.xls"))
    db.save_analysis(result)

    with db.session() as s:
        passes = s.execute(sa.text(
            "SELECT pass_index, laser_cut_length FROM trim_passes ORDER BY pass_index")).all()
        setup = s.execute(sa.text(
            "SELECT initial_resistance_low, final_resistance_high FROM trim_setup")).first()
    assert [p[0] for p in passes] == [1, 2, 3]
    assert passes[0][1] == 0.75
    assert setup == (4200.0, 5500.0)


def test_saving_twice_does_not_duplicate_passes(tmp_path, monkeypatch):
    """A reprocess must refresh, not accumulate."""
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.core.processor import Processor
    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    proc = Processor(use_ml=False)
    p = Path("tests/fixtures/trim/dlts_8232-1_243.xls")
    db.save_analysis(proc.process_file(p))
    db.save_analysis(proc.process_file(p))
    with db.session() as s:
        n = s.execute(sa.text("SELECT COUNT(*) FROM trim_passes")).scalar()
    assert n == 3, f"expected 3 pass rows after two saves, got {n}"


def test_trim_pass_unique_index_rejects_duplicate_insert(tmp_path, monkeypatch):
    """Findings review (task-7-brief.md #1): the schema test only ever proved
    the (track_result_id, pass_index) index exists, never that it actually
    rejects a duplicate row. This is the first task to write pass rows, so
    prove the DB-level behaviour, not just the DDL.
    """
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.database.models import TrimPass
    from laser_trim_analyzer.core.processor import Processor

    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    proc = Processor(use_ml=False)
    result = proc.process_file(Path("tests/fixtures/trim/dlts_8232-1_243.xls"))
    db.save_analysis(result)

    with db.session() as s:
        track_result_id = s.execute(sa.text(
            "SELECT track_result_id FROM trim_passes WHERE pass_index = 1 LIMIT 1"
        )).scalar()
    assert track_result_id is not None

    with pytest.raises(sa.exc.IntegrityError):
        with db.session() as s:
            s.add(TrimPass(track_result_id=track_result_id, pass_index=1))


def test_write_trim_passes_tolerates_duplicate_pass_index(tmp_path, monkeypatch):
    """Findings review (task-7-brief.md #2): pass_sheets has no dedup guard,
    so two differently-named sheets that normalise to the same leading
    number would produce two passes with the same pass_index and collide on
    the unique index at write time. Never observed in a fixture, but a
    production file that fails to save entirely (verdict and all) would be
    worse than one that loses a single duplicate pass row -- so the write
    path must tolerate the collision, not let it raise.
    """
    from laser_trim_analyzer.database import manager as mgr
    from laser_trim_analyzer.core.processor import Processor

    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    proc = Processor(use_ml=False)
    result = proc.process_file(Path("tests/fixtures/trim/dlts_8232-1_243.xls"))

    track = result.tracks[0]
    assert len(track.trim_passes) >= 2
    # Force the collision the parser is not known to produce but cannot rule
    # out: relabel pass 2 as pass 1's index.
    track.trim_passes[1]["pass_index"] = track.trim_passes[0]["pass_index"]

    db_id = db.save_analysis(result)  # must not raise
    assert db_id > 0

    with db.session() as s:
        n = s.execute(sa.text("SELECT COUNT(*) FROM trim_passes")).scalar()
        indices = [r[0] for r in s.execute(
            sa.text("SELECT pass_index FROM trim_passes ORDER BY pass_index"))]
    assert n == 2, f"expected the colliding pass to be dropped, got {n} rows"
    assert indices == sorted(set(indices)), "no duplicate pass_index made it to storage"
