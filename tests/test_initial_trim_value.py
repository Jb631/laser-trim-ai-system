"""Laser 2 (DLTS) and laser 3 (LTS3, read in laser 2's format) pass sheets carry, per
position, the trim target (column L), the INITIAL trim value (M) and the final trim value
(N). `core/trim_passes._A_PER_POINT` has always read column M, but until now
`database/manager.py::_write_trim_passes` stored only six of its seven per-point keys --
`initial_trim_value` fell into the pass row's `recipe` JSON blob instead of getting its own
column (2026-09-24). Existing laser-2/3 rows are NOT migrated at start-up (a heavy UPDATE
across ~83,000 rows at start-up is the shape of the 2026-09-14 night) -- they keep the value
in `recipe` and must read back the same value as a new row through
`trim_passes.initial_trim_values`.
"""
import json
from pathlib import Path

import pytest
import sqlalchemy as sa

from laser_trim_analyzer.core.trim_passes import initial_trim_values

DLTS = Path("tests/fixtures/trim/dlts_8232-1_243.xls")
LTS = Path("tests/fixtures/trim/lts_8232-1_193.xls")


# ---------------------------------------------------------------------------
# The pure helper -- both storage shapes read back the same value.
# ---------------------------------------------------------------------------

def test_the_row_column_wins_when_present():
    assert initial_trim_values([1.0, 2.0], {"initial_trim_value": [9.0, 9.0]}) == [1.0, 2.0]


def test_falls_back_to_the_recipe_blob_when_the_row_column_is_empty():
    # None (a raw-SQL real NULL) and [] (the ORM's read-back of one, per SafeJSON's
    # none_as=[] default) must both fall through.
    assert initial_trim_values(None, {"initial_trim_value": [1.0, None, 2.0]}) == [1.0, None, 2.0]
    assert initial_trim_values([], {"initial_trim_value": [1.0, 2.0]}) == [1.0, 2.0]


def test_a_recipe_as_a_raw_json_string_is_decoded():
    """A raw-SQL caller (a sweep script reading sqlite3 rows directly) gets text, not a
    dict; the ORM gets a dict already. Both must work."""
    assert initial_trim_values(None, '{"initial_trim_value": [3.0, 4.0]}') == [3.0, 4.0]


def test_a_row_value_as_a_raw_json_string_is_decoded():
    assert initial_trim_values('[5.0, 6.0]', {"initial_trim_value": [1.0]}) == [5.0, 6.0]


def test_neither_place_has_it():
    assert initial_trim_values(None, None) is None
    assert initial_trim_values(None, {"trim_velocity": 3.0}) is None   # key simply absent
    assert initial_trim_values(None, []) is None                       # the ORM's empty-recipe shape
    assert initial_trim_values(None, "") is None


def test_a_malformed_json_string_is_refused_not_a_crash():
    assert initial_trim_values(None, "{not valid json") is None
    assert initial_trim_values("{not valid json", {"initial_trim_value": [1.0]}) == [1.0]


# ---------------------------------------------------------------------------
# Step 1 (task-8-brief.md): the real DLTS fixture through the real pipeline.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DLTS.exists(), reason="DLTS fixture")
def test_dlts_trim_passes_store_initial_trim_value_in_their_own_column(tmp_path, monkeypatch):
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    from laser_trim_analyzer.core.processor import Processor

    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)

    proc = Processor(use_ml=False)
    result = proc.process_file(DLTS)
    db.save_analysis(result)

    with db.session() as s:
        rows = s.execute(sa.text(
            "SELECT pass_index, sheet, initial_trim_value, recipe FROM trim_passes "
            "WHERE sheet NOT LIKE 'lin error%' ORDER BY pass_index")).all()
    assert rows, "the DLTS fixture has real Trim N passes"
    for idx, sheet, row_value, recipe in rows:
        assert row_value is not None, f"pass {idx} ({sheet}) has no initial_trim_value column value"
        parsed = json.loads(row_value) if isinstance(row_value, str) else row_value
        assert isinstance(parsed, list) and any(v is not None for v in parsed), (
            f"pass {idx} ({sheet})'s column holds no real value: {parsed!r}")
        recipe_dict = json.loads(recipe) if isinstance(recipe, str) else (recipe or {})
        assert "initial_trim_value" not in (recipe_dict or {}), (
            f"pass {idx} ({sheet}) still carries the key inside recipe")


@pytest.mark.skipif(not LTS.exists(), reason="LTS fixture")
def test_a_laser_1_pass_has_no_initial_trim_value_in_either_place(tmp_path, monkeypatch):
    """Laser 1's sheets have no such column: neither the row nor its recipe carries it."""
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    from laser_trim_analyzer.core.processor import Processor

    db = mgr.DatabaseManager(tmp_path / "t.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)

    proc = Processor(use_ml=False)
    result = proc.process_file(LTS)
    db.save_analysis(result)

    with db.session() as s:
        rows = s.execute(sa.text("SELECT initial_trim_value, recipe FROM trim_passes")).all()
    assert rows, "the LTS fixture has trim_passes rows"
    for row_value, recipe in rows:
        assert row_value is None, f"a laser-1 row unexpectedly has {row_value!r}"
        recipe_dict = json.loads(recipe) if isinstance(recipe, str) else (recipe or {})
        assert "initial_trim_value" not in (recipe_dict or {})


def test_an_old_row_reads_back_through_the_helper(tmp_path):
    """A row written the OLD way: `initial_trim_value` column NULL, `recipe` carries the
    key -- built directly with raw SQL (never through `_write_trim_passes`, the code under
    test), so a regression in the writer cannot mask a regression in the read helper."""
    from laser_trim_analyzer.database import manager as mgr

    db = mgr.DatabaseManager(tmp_path / "old.db")
    with db.session() as s:
        s.execute(sa.text(
            "INSERT INTO analysis_results (filename, file_date, model, serial, system, "
            "has_multi_tracks, overall_status, timestamp, data_quality) VALUES "
            "('x.xls', '2026-01-01 00:00:00', 'M', '1', 'A', 0, 'PASS', "
            "'2026-01-01 00:00:00', 'good')"))
        aid = s.execute(sa.text("SELECT id FROM analysis_results")).scalar()
        s.execute(sa.text(
            "INSERT INTO track_results (analysis_id, track_id, status) "
            "VALUES (:aid, 'TRK1', 'Pass')"), {"aid": aid})
        tid = s.execute(sa.text("SELECT id FROM track_results")).scalar()
        s.execute(sa.text(
            "INSERT INTO trim_passes (track_result_id, pass_index, sheet, "
            "initial_trim_value, recipe, created_date) VALUES "
            "(:tid, 1, 'SEC1 TRK1 1 TRM1', NULL, :recipe, '2026-01-01 00:00:00')"),
            {"tid": tid, "recipe": json.dumps({"initial_trim_value": [1.5, None, 2.5],
                                               "trim_velocity": 3.0})})
        s.commit()

    with db.session() as s:
        row_value, recipe = s.execute(sa.text(
            "SELECT initial_trim_value, recipe FROM trim_passes")).first()
    assert row_value is None, "the column must be a real NULL, not the JSON text 'null'"
    recipe_dict = json.loads(recipe) if isinstance(recipe, str) else recipe
    assert initial_trim_values(row_value, recipe_dict) == [1.5, None, 2.5]


def test_the_migration_adds_the_column_to_an_existing_database(tmp_path):
    """Same pattern as the increment_volts migration: an existing DB missing the column
    gets it added (metadata only), and a database that already has it is left alone."""
    import sqlalchemy as sa2
    from laser_trim_analyzer.database import manager as mgr

    db_path = tmp_path / "pre.db"
    db = mgr.DatabaseManager(db_path)
    with db.session() as s:
        s.execute(sa2.text("ALTER TABLE trim_passes DROP COLUMN initial_trim_value"))
        s.commit()
    db.close()

    db2 = mgr.DatabaseManager(db_path)   # start-up migration must add it back
    with db2.session() as s:
        cols = {r[1] for r in s.execute(sa2.text("PRAGMA table_info(trim_passes)"))}
    assert "initial_trim_value" in cols
    db2.close()
