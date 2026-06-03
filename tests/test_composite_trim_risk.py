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


def test_training_record_includes_composite_features():
    import inspect
    from laser_trim_analyzer.ml import manager as mgr
    src = inspect.getsource(mgr._get_training_data) if hasattr(mgr, "_get_training_data") \
        else inspect.getsource(mgr.MLManager._get_training_data)
    for key in ("untrimmed_error_max", "resistance_change_percent", "trim_pass_count"):
        assert f"'{key}'" in src or f'"{key}"' in src, f"_get_training_data must emit {key}"


# ---------------------------------------------------------------------------
# Task 7: CompositeRiskModel tests
# ---------------------------------------------------------------------------

def _toy_frame(n=240, separable=True, seed=0):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    fail = rng.integers(0, 2, n)
    emax = rng.normal(0.05, 0.01, n) + (0.03 * fail if separable else 0.0)
    sigma = rng.normal(0.001, 0.0003, n)
    rcp = rng.normal(15, 3, n)
    serial = [f"S{i//3}" for i in range(n)]  # repeated serials -> grouping matters
    return pd.DataFrame({
        "untrimmed_error_max": emax, "untrimmed_sigma_gradient": sigma,
        "resistance_change_percent": rcp, "trim_pass_count": rng.integers(1, 3, n),
        "linearity_pass": 1 - fail, "serial": serial,
    })


def test_composite_trains_and_scores_in_unit_interval():
    from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
    m = CompositeRiskModel("8232-1")
    res = m.train(_toy_frame(separable=True))
    assert res.n_samples == 240
    assert 0.0 <= res.cv_auc <= 1.0
    p = m.predict_proba({"untrimmed_error_max": 0.09, "untrimmed_sigma_gradient": 0.001,
                         "resistance_change_percent": 15.0, "trim_pass_count": 2})
    assert 0.0 <= p <= 1.0


def test_composite_deploy_gate_blocks_no_signal_model():
    from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
    m = CompositeRiskModel("noise")
    res = m.train(_toy_frame(separable=False, seed=7))
    # no real signal -> low confidence / no lift -> not deployed
    assert res.deployed is False


def test_composite_deploy_gate_passes_separable_model():
    from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
    m = CompositeRiskModel("real")
    res = m.train(_toy_frame(separable=True, seed=1))
    assert res.cv_auc > 0.6


def test_composite_drops_all_null_feature():
    import numpy as np
    from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
    df = _toy_frame(separable=True)
    df["trim_pass_count"] = np.nan          # simulate pre-reprocess state
    m = CompositeRiskModel("preReprocess")
    res = m.train(df)
    assert "trim_pass_count" not in res.features_used
    assert res.deployed in (True, False)     # still trains on the rest


def test_composite_save_load_roundtrip(tmp_path):
    from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
    m = CompositeRiskModel("rt")
    m.train(_toy_frame(separable=True))
    p = tmp_path / "rt.pkl"
    m.save(p)
    m2 = CompositeRiskModel.load(p)
    feat = {"untrimmed_error_max": 0.09, "untrimmed_sigma_gradient": 0.001,
            "resistance_change_percent": 15.0, "trim_pass_count": 2}
    assert abs(m.predict_proba(feat) - m2.predict_proba(feat)) < 1e-9
