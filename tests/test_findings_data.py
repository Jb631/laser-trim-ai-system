from pathlib import Path

import pytest

FIXTURES = sorted(Path("tests/fixtures/trim").glob("*.xls"))


@pytest.fixture
def fixture_db(tmp_path, monkeypatch):
    """Every real fixture in tests/fixtures/trim/ (the four 8232-1 workbooks plus the
    two-track DLTS fixtures) through the real pipeline into a throwaway database."""
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


def test_the_expected_fixtures_are_present():
    """The original four 8232-1 workbooks, plus the two-track DLTS fixtures added for
    Task 9 (`dlts_8074_18.xls`, `dlts_7553_10B.xls`) -- different models, so they do
    not change any assertion below that queries model "8232-1" specifically."""
    assert [f.name for f in FIXTURES] == [
        "dlts_7553_10B.xls", "dlts_8074_18.xls",
        "dlts_8232-1_242.xls", "dlts_8232-1_243.xls",
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


def test_the_yardstick_result_carries_the_bar_it_was_held_to():
    from laser_trim_analyzer.findings import data
    y = data.yardstick_fidelity([])
    assert y == {"n": 0, "agreement": None, "faithful": False, "min_n": 30, "min_agreement": 0.99}
    assert (data.YARDSTICK_MIN_N, data.YARDSTICK_MIN_AGREEMENT) == (30, 0.99)


def test_one_corrupt_stored_array_costs_that_track_not_the_whole_model():
    """`_arr` used to let json.JSONDecodeError escape, which aborted load_model_tracks -- so ONE bad
    row among four thousand cost the model every finding. A corrupt array is an ungradeable track."""
    from laser_trim_analyzer.findings.data import _arr
    assert _arr("[1, 2, 3]") == (1, 2, 3)
    assert _arr("[1, 2, ") is None and _arr("not json") is None and _arr(b"\xff\xfe") is None
    assert _arr(None) is None and _arr('{"a": 1}') is None and _arr([4, 5]) == (4, 5)


def test_the_loader_carries_the_track_name_the_positions_and_so_the_limit_table(fixture_db):
    from laser_trim_analyzer.findings.data import load_model_tracks
    tracks = load_model_tracks(fixture_db, "8232-1")
    # laser 2 names its tracks in the sheet ("TRK1"); laser 1 takes the letter from a `_TA_` in the FILE
    # name, which these renamed fixtures do not have -- real files do ("Track A").
    assert {(t.system, t.track_name) for t in tracks} == {("A", "TRK1"), ("B", "default")}
    for t in tracks:
        assert t.final_positions is not None and len(t.final_positions) == len(t.final_upper)
        tab = t.limit_table
        assert tab is not None and tab.rows == len(t.final_upper) and 3 <= tab.graded <= tab.rows
        assert len(tab.band) == tab.graded                      # every graded row has a position
    # the two lasers' fixtures were graded against different tables; each laser against ONE
    assert len({t.limit_table.key for t in tracks if t.system == "A"}) == 1
    assert len({t.limit_table.key for t in tracks if t.system == "B"}) == 1


def test_a_stored_final_sweep_of_exact_zeros_loses_its_verdict_but_keeps_its_limits(fixture_db):
    """A database ingested before 2026-09-20 holds ~1,182 no-cut files whose 'final sweep' is the blank
    template: every reading exactly 0.0, stored as a flawless linearity PASS. The loader must not hand
    that to the analyzers as a pass -- but the limits on it ARE the table that was in service."""
    import json
    from sqlalchemy import text
    from laser_trim_analyzer.findings.data import _is_blank_template, load_model_tracks
    before = {t.track_id: t for t in load_model_tracks(fixture_db, "8232-1")}
    victim = next(t for t in before.values() if t.system == "B")
    zeros = json.dumps([None] * 6 + [0.0] * (len(victim.final_errors) - 6))
    with fixture_db.session() as s:
        s.execute(text("UPDATE track_results SET error_data = :e, linearity_pass = 1 WHERE id = :i"),
                  {"e": zeros, "i": victim.track_id})
    after = {t.track_id: t for t in load_model_tracks(fixture_db, "8232-1")}[victim.track_id]
    assert after.final_errors is None and after.linearity_pass is None
    assert after.limit_table is not None and after.limit_table.key == victim.limit_table.key
    assert _is_blank_template([None] * 6 + [0.0] * 105)
    assert not _is_blank_template([0.0] * 9)                     # too few readings to call
    assert not _is_blank_template([0.0] * 50 + [1e-9]) and not _is_blank_template(None)


def test_a_failed_processing_track_is_not_a_measurement(fixture_db):
    """An ERROR track stored with linearity_pass=0 (every such row on the rebuilt work
    database) must not reach the analyzers, which count every non-None verdict."""
    from sqlalchemy import text
    from laser_trim_analyzer.findings.data import load_model_tracks
    before = load_model_tracks(fixture_db, "8232-1")
    victim = before[0].track_id
    with fixture_db.session() as s:
        s.execute(text("UPDATE track_results SET status='ERROR', linearity_pass=0 WHERE id=:i"),
                  {"i": victim})
        s.commit()
    after = load_model_tracks(fixture_db, "8232-1")
    assert victim not in {t.track_id for t in after}
    assert len(after) == len(before) - 1
