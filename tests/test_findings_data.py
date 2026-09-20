from pathlib import Path

import pytest

FIXTURES = sorted(Path("tests/fixtures/trim").glob("*.xls"))


@pytest.fixture
def fixture_db(tmp_path, monkeypatch):
    """The four real 8232-1 workbooks through the real pipeline into a throwaway database."""
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    db = mgr.DatabaseManager(tmp_path / "findings.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)      # BOTH globals, or get_database()
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)    # builds one at the configured path
    proc = Processor(use_ml=False)
    for f in FIXTURES:
        db.save_analysis(proc.process_file(f))
    return db


def test_the_four_fixtures_are_present():
    assert [f.name for f in FIXTURES] == ["dlts_8232-1_242.xls", "dlts_8232-1_243.xls",
                                          "lts_8232-1_193.xls", "lts_8232-1_194.xls"]


def test_loader_returns_real_cuts_only(fixture_db):
    from laser_trim_analyzer.findings.data import load_model_tracks
    tracks = load_model_tracks(fixture_db, "8232-1")
    assert len(tracks) == 4
    cuts = sorted((t.system, len(t.passes)) for t in tracks)
    # Laser 1 (LTS = system B) files hold Trim 1, Trim 2 and a duplicate `Lin Error` sheet:
    # TWO real cuts, not three.
    assert cuts == [("A", 2), ("A", 3), ("B", 2), ("B", 2)]
    assert all(not p.sheet.lower().startswith("lin error") for t in tracks for p in t.passes)


def test_recipe_is_what_the_laser_was_told_to_do(fixture_db):
    from laser_trim_analyzer.findings.data import load_model_tracks
    recipes = sorted(t.recipe for t in load_model_tracks(fixture_db, "8232-1"))
    assert recipes == [(2, (0.75, 0.88)), (2, (4100.0, 4100.0)), (2, (4100.0, 4100.0)),
                       (3, (0.75, 0.88, 0.88))]


def test_station_resistance_limits_come_through(fixture_db):
    from laser_trim_analyzer.findings.data import load_model_tracks
    tracks = load_model_tracks(fixture_db, "8232-1")
    assert {(t.final_r_low, t.final_r_high) for t in tracks} == {(5000.0, 5500.0)}
    assert {(t.initial_r_low, t.initial_r_high) for t in tracks if t.system == "A"} == {(4200.0, 4600.0)}
    # Laser 1 files carry no configured incoming window (spec: "Format asymmetry").
    assert {(t.initial_r_low, t.initial_r_high) for t in tracks if t.system == "B"} == {(None, None)}


def test_yardstick_reproduces_the_apps_verdict_on_the_fixtures(fixture_db):
    from laser_trim_analyzer.findings.data import load_model_tracks, yardstick_fidelity
    y = yardstick_fidelity(load_model_tracks(fixture_db, "8232-1"))
    assert y["n"] == 4 and y["agreement"] == 1.0
    assert y["faithful"] is False           # four tracks cannot vouch for a model (needs 30)


def test_an_unknown_model_is_empty_not_an_error(fixture_db):
    from laser_trim_analyzer.findings.data import load_model_tracks
    assert load_model_tracks(fixture_db, "no-such-model") == []
