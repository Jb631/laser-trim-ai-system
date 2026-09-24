"""A track too short to grade is an ERROR, and an ERROR carries no verdict.

2026-09-23: `_create_failed_track` stored linearity_pass=False, so four Findings analyzers
counted 94 'Insufficient data points' ERRORs as linearity FAILS -- 8856's laser-2 pass
rate read 27.7% when the graded tracks passed 49.0%.
"""
from laser_trim_analyzer.core.analyzer import Analyzer
from laser_trim_analyzer.core.models import AnalysisStatus


def _short_track(n=5):
    return {"track_id": "default", "positions": [float(i) for i in range(n)],
            "errors": [0.001] * n, "upper_limits": [0.01] * n, "lower_limits": [-0.01] * n,
            "travel_length": 1.0, "linearity_spec": 0.01}


def test_a_track_too_short_to_grade_is_an_error_with_no_verdict():
    t = Analyzer().analyze_track(_short_track())
    assert t.status == AnalysisStatus.ERROR
    assert t.anomaly_reason == "Insufficient data points"
    assert t.linearity_pass is None          # not False: nothing was graded
    assert t.sigma_pass is None
