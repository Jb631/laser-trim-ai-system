"""Tests for the composite trim-risk early-warning feature (2026-06-01 plan)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_trackdata_has_untrimmed_error_max_field():
    from laser_trim_analyzer.core.models import TrackData
    fields = TrackData.model_fields  # pydantic v2
    assert "untrimmed_error_max" in fields, "TrackData must expose untrimmed_error_max"


def test_trackresult_has_untrimmed_error_max_column():
    from laser_trim_analyzer.database.models import TrackResult
    assert hasattr(TrackResult, "untrimmed_error_max"), \
        "TrackResult must have an untrimmed_error_max column"


def test_analyzer_computes_untrimmed_error_max():
    from laser_trim_analyzer.core.analyzer import Analyzer
    a = Analyzer()  # __init__ args (scaling_factor, model_thresholds) all have defaults
    # untrimmed errors with a clear worst point at -0.05; trimmed much smaller
    untrimmed = [0.01, -0.02, 0.03, -0.05, 0.012]
    trimmed = [0.001, -0.002, 0.0015, -0.001, 0.0008]
    res = a._calculate_trim_effectiveness(
        trimmed_errors=trimmed,
        untrimmed_errors=untrimmed,
        untrimmed_resistance=4486.0,
        trimmed_resistance=5256.0,
    )
    assert abs(res["untrimmed_error_max"] - 0.05) < 1e-9
    assert abs(res["resistance_change"] - 770.0) < 1e-9


def test_untrimmed_error_max_present_for_untrimmed_only_track():
    """The guard exists so untrimmed-only test-sweep tracks (no trim run, so an
    empty trimmed sweep) -- the upstream-drift case -- still get
    untrimmed_error_max. RMS still requires both sweeps and must stay absent."""
    from laser_trim_analyzer.core.analyzer import Analyzer
    a = Analyzer()
    res = a._calculate_trim_effectiveness(
        trimmed_errors=[],                          # untrimmed-only: no trim run
        untrimmed_errors=[0.01, -0.02, 0.064, 0.03],
        untrimmed_resistance=None,
        trimmed_resistance=None,
    )
    assert abs(res["untrimmed_error_max"] - 0.064) < 1e-9
    assert "untrimmed_rms_error" not in res  # both-sweep guard still gates RMS


def test_backfill_fills_only_null_rows(tmp_path):
    import sqlite3, json
    from scripts.backfill_trim_effort import backfill_trim_effort

    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE track_results (
            id INTEGER PRIMARY KEY,
            untrimmed_errors TEXT,
            untrimmed_resistance REAL,
            trimmed_resistance REAL,
            untrimmed_error_max REAL,
            untrimmed_rms_error REAL,
            resistance_change REAL,
            resistance_change_percent REAL
        );
        """
    )
    # row 1: everything NULL -> should be filled
    con.execute(
        "INSERT INTO track_results (id, untrimmed_errors, untrimmed_resistance, trimmed_resistance) "
        "VALUES (1, ?, 4486.0, 5256.0)",
        (json.dumps([0.01, -0.05, 0.03]),),
    )
    # row 2: untrimmed_error_max already set -> must NOT be overwritten
    con.execute(
        "INSERT INTO track_results (id, untrimmed_errors, untrimmed_error_max) VALUES (2, ?, 999.0)",
        (json.dumps([0.01, -0.05, 0.03]),),
    )
    con.commit(); con.close()

    n = backfill_trim_effort(str(db))
    assert n >= 1

    con = sqlite3.connect(db)
    r1 = con.execute(
        "SELECT untrimmed_error_max, untrimmed_rms_error, resistance_change, resistance_change_percent "
        "FROM track_results WHERE id=1"
    ).fetchone()
    assert abs(r1[0] - 0.05) < 1e-9          # error_max
    assert r1[1] > 0                          # rms filled
    assert abs(r1[2] - 770.0) < 1e-9          # resistance_change
    assert abs(r1[3] - (770.0 / 4486.0 * 100)) < 1e-6
    r2 = con.execute("SELECT untrimmed_error_max FROM track_results WHERE id=2").fetchone()
    assert r2[0] == 999.0                      # untouched
    con.close()


def test_trackresult_has_composite_score_column():
    from laser_trim_analyzer.database.models import TrackResult
    assert hasattr(TrackResult, "composite_trim_risk_score")
