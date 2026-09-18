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
