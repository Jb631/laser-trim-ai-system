"""Tests for the composite trim-risk early-warning feature (2026-06-01 plan)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


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
