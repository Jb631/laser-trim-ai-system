"""Regression tests for the linearity_spec plausibility guard.

Root cause (measured 2026-08-30 against data/analysis.db): model 8888's
13 corrupt tracks are NOT a parser bug. The parser read the cells
faithfully — the source workbooks really contain, in the Upper Lin Lim
column, 0.03 repeated 23 times and then 1.03, 2.03, 3.03 ... 148.03.
That is Excel's fill-handle "Fill Series" applied to a decimal: dragging
0.03 down increments the integer part instead of copying. The median of
that column is exactly 63.03, which is what got stored and which passes
essentially every unit.

The fixtures below are the REAL cell content, reconstructed exactly from
the upper_limits/lower_limits JSON the parser stored for those files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --- real cell content -------------------------------------------------

def _fill_series_8888():
    """8888_1..13_TEST DATA: 23 correct rows, then Excel fill-series."""
    upper = [0.03] * 23 + [round(n + 0.03, 2) for n in range(1, 149)]
    lower = [-u for u in upper]
    return upper, lower


def _fill_series_8094_2():
    """8094-2_1_TA_Test Data: same artifact, 58 correct rows then a ramp."""
    upper = [0.025] * 58 + [round(n + 0.025, 3) for n in range(1, 60)]
    lower = [-u for u in upper]
    return upper, lower


def _good_8888():
    """8888_14 onward — the template after it was fixed."""
    upper = [0.03] * 171
    return upper, [-0.03] * 171


def _parser():
    from laser_trim_analyzer.core.parser import ExcelParser
    return ExcelParser()


# --- the exact number this bug produced --------------------------------

def test_fill_series_column_still_medians_to_63_03():
    """Pin the observed value: the guard must fire on 63.03, not mask it."""
    import numpy as np
    upper, lower = _fill_series_8888()
    assert len(upper) == 171, "8888 tracks stored 171 limit rows"
    spec = _parser()._calculate_linearity_spec(upper, lower)
    assert abs(spec - 63.03) < 1e-9, f"expected the real 63.03, got {spec}"
    assert abs(float(np.median(upper)) - 63.03) < 1e-9


def test_8888_fill_series_is_rejected():
    upper, lower = _fill_series_8888()
    p = _parser()
    reason = p._validate_limit_columns(upper, lower, p._calculate_linearity_spec(upper, lower))
    assert reason is not None, "63.03 V spec must not be accepted silently"
    assert "consistent band" in reason
    # the reason must name the modal (true) value so the bench can act on it
    assert "0.03" in reason


def test_8094_2_fill_series_is_rejected():
    upper, lower = _fill_series_8094_2()
    p = _parser()
    spec = p._calculate_linearity_spec(upper, lower)
    assert abs(spec - 1.025) < 1e-9, f"expected the real 1.025, got {spec}"
    reason = p._validate_limit_columns(upper, lower, spec)
    assert reason is not None, "1.025 is within the absolute range; shape must catch it"


def test_good_8888_template_is_accepted():
    upper, lower = _good_8888()
    p = _parser()
    spec = p._calculate_linearity_spec(upper, lower)
    assert abs(spec - 0.03) < 1e-9
    assert p._validate_limit_columns(upper, lower, spec) is None, \
        "the 43 clean 8888 tracks must keep grading normally"


# --- the other defect families the same guard covers -------------------

def test_sign_error_collapsing_spec_to_zero_is_rejected():
    """8097: lower limit stored as +0.005, so the spec computes to 0.0."""
    p = _parser()
    upper, lower = [0.005] * 97, [0.005] * 97
    spec = p._calculate_linearity_spec(upper, lower)
    assert spec == 0.0, "this is what 8097 stored"
    reason = p._validate_limit_columns(upper, lower, spec)
    assert reason is not None and "+/- band" in reason


def test_position_column_in_limits_is_rejected():
    """6952: the position column landed in upper_limits."""
    p = _parser()
    upper = [-169.39 + 5.0 * i for i in range(65)]
    lower = [0.025] * 65
    reason = p._validate_limit_columns(upper, lower, p._calculate_linearity_spec(upper, lower))
    assert reason is not None and "+/- band" in reason


def test_ragged_lower_limit_is_rejected():
    """7965: lower limit jumps between -0.05, 0.0, 0.02 and 0.05."""
    p = _parser()
    upper = [0.1] * 60
    lower = ([-0.05] * 4 + [0.02, 0.0, 0.02, 0.02]) * 7 + [0.05] * 4
    reason = p._validate_limit_columns(upper, lower, p._calculate_linearity_spec(upper, lower))
    assert reason is not None


# --- must NOT fire on real, legitimately-shaped specs -------------------

def test_legitimate_wide_spec_is_accepted():
    """Model 8802 ("Every deg") really does run a 1.0 V spec."""
    p = _parser()
    upper, lower = [1.0] * 18, [-1.0] * 18
    spec = p._calculate_linearity_spec(upper, lower)
    assert abs(spec - 1.0) < 1e-9
    assert p._validate_limit_columns(upper, lower, spec) is None, \
        "an absolute ceiling below 1.0 would false-positive on 8802"


def test_endpoint_widened_band_is_accepted():
    """8340-style sheets widen the first/last rows to +/-0.20 on purpose."""
    p = _parser()
    upper = [0.20] + [0.02] * 98 + [0.20]
    lower = [-u for u in upper]
    assert p._validate_limit_columns(upper, lower, p._calculate_linearity_spec(upper, lower)) is None


def test_position_varying_band_is_accepted():
    """8232-1 / 8762 / 8508-x carry genuinely position-varying spec bands."""
    p = _parser()
    upper = [0.0112 + 0.0057 * (i % 45) / 45 for i in range(89)]
    lower = [-u for u in upper]
    assert p._validate_limit_columns(upper, lower, p._calculate_linearity_spec(upper, lower)) is None


def test_missing_limits_are_not_flagged():
    """Absent limits are a different condition, handled upstream."""
    p = _parser()
    assert p._validate_limit_columns([], [], 0.01) is None
    assert p._validate_limit_columns([None] * 5, [None] * 5, 0.01) is None


# --- refuse to grade ----------------------------------------------------

def test_trackdata_exposes_the_warning_field():
    from laser_trim_analyzer.core.models import TrackData
    assert "linearity_spec_warning" in TrackData.model_fields


def test_trackresult_has_the_warning_column():
    from laser_trim_analyzer.database.models import TrackResult
    assert hasattr(TrackResult, "linearity_spec_warning")


def test_corrupt_spec_is_not_graded_as_passing():
    """The whole point: a bad spec must never manufacture a PASS."""
    from laser_trim_analyzer.core.analyzer import Analyzer
    from laser_trim_analyzer.core.models import AnalysisStatus

    upper, lower = _fill_series_8888()
    n = len(upper)
    track = {
        "track_id": "TRK1",
        "positions": [float(i) for i in range(n)],
        # errors well inside 63.03 but OUTSIDE the true 0.03 band
        "errors": [0.0396 if i == 5 else 0.01 for i in range(n)],
        "upper_limits": upper,
        "lower_limits": lower,
        "travel_length": 342.0,
        "linearity_spec": 63.03,
        "linearity_spec_warning": "limit column is not a consistent band",
        "unit_length": 342.0,
    }
    result = Analyzer().analyze_track(track, model="8888")
    assert result.linearity_pass is not True, \
        "a 63 V spec must not be allowed to pass a unit"
    assert result.linearity_pass is None, "verdict must be indeterminate, not a graded FAIL"
    assert result.linearity_spec_warning
    assert result.status == AnalysisStatus.ERROR


def test_clean_track_still_grades_normally():
    from laser_trim_analyzer.core.analyzer import Analyzer
    from laser_trim_analyzer.core.models import AnalysisStatus

    n = 171
    track = {
        "track_id": "TRK1",
        "positions": [float(i) for i in range(n)],
        "errors": [0.005] * n,
        "upper_limits": [0.03] * n,
        "lower_limits": [-0.03] * n,
        "travel_length": 342.0,
        "linearity_spec": 0.03,
        "linearity_spec_warning": None,
        "unit_length": 342.0,
    }
    result = Analyzer().analyze_track(track, model="8888")
    assert result.linearity_pass is True
    assert result.linearity_spec_warning is None
    assert result.status != AnalysisStatus.ERROR


# --- remediation: what is safely recoverable, and what is not ----------

def _recover():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "fix_corrupt_linearity_spec.py"
    spec = importlib.util.spec_from_file_location("fix_corrupt_linearity_spec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.recover_fill_series_band


def test_recovers_true_spec_from_8888_fill_series():
    upper, lower = _fill_series_8888()
    spec, support = _recover()(upper, lower)
    assert spec == 0.03, f"8888's real spec is 0.03, got {spec}"
    assert support == 23, "23 rows survived the fill before it ramped"


def test_recovers_8094_2_despite_its_ramp_restarting():
    """8094-2 goes ...0.025, 1.025, 0.025, 1.025, 2.025... — not monotonic."""
    upper = [0.025] * 57 + [1.025, 0.025] + [round(n + 0.025, 3) for n in range(1, 59)]
    lower = [-u for u in upper]
    spec, support = _recover()(upper, lower)
    assert spec == 0.025, f"8094-2's real spec is 0.025, got {spec}"
    assert support == 58


def test_ragged_limits_are_NOT_recovered():
    """7965 regression: the naive 'smallest half-width wins' rule recovered
    0.025 when the model's real spec is 0.05 — a tighter-than-real band that
    manufactures 9 failures. Ragged columns must stay unrecoverable."""
    upper = [0.1] * 60
    lower = ([-0.05] * 4 + [0.02, 0.0, 0.02, 0.02]) * 7 + [0.05] * 4
    spec, support = _recover()(upper, lower)
    assert spec is None, f"must not invent a spec for ragged limits, got {spec}"


def test_clean_column_is_not_recovered():
    """Nothing to fix on a healthy band."""
    upper, lower = _good_8888()
    assert _recover()(upper, lower)[0] is None


def test_sign_error_is_not_recovered():
    """8097's columns share a sign; there is no band to recover."""
    assert _recover()([0.005] * 97, [0.005] * 97)[0] is None
