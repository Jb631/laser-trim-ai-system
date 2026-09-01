"""Regression tests: the linearity error MAGNITUDE must never vanish silently.

Root cause (measured 2026-08-31 against data/analysis.db): model 8232-1 had
3,108 tracks since 2023 with linearity_pass, linearity_fail_points,
linearity_spec and optimal_offset all populated — and EVERY error-magnitude
column NULL. 8770 had 206 more. Linearity is the zero-tolerance customer
disposition and 8232-1 is #1 on the FOCUS list, so this was the worst
possible model to lose it on.

The mechanism, proven end-to-end:

  1. These files carry a lead-in dead zone: the trimmed sweep records no
     value for the first 6 of 57 points, so error_data starts [NaN]*6 and
     the matching limit columns start [null]*6.
  2. analyzer._calculate_linearity computed the magnitude with
     `max(abs(e) for e in shifted_errors)`. Python's max() seeds with the
     FIRST element and every later comparison against NaN is False, so a
     NaN at index 0 propagates straight through to the result. A NaN
     anywhere else is skipped and the answer comes out right — the bug was
     a pure positional lottery.
  3. BaseAnalysisModel._nan_means_missing (a mode="before" model_validator
     added 2026-07-10 in 413167e) coerces NaN to None on every field before
     the `ge=0` constraint can reject it. That is why the row still saved
     with all its other fields intact instead of raising ValidationError.
  4. manager.py stores that None, and SQLite writes NULL.

The DB confirmed the lottery with zero exceptions across 42,387 tracks since
2023: NaN at index 0 -> magnitude NULL (3,314 tracks); NaN elsewhere ->
magnitude stored (2 tracks); no NaN -> magnitude stored (39,071 tracks).

Downstream this was worse than missing: manager.py reads the column back as
`abs(... or 0.0)`, so a dropped magnitude re-entered the app as 0.0 — a
flawless part — on the metric the customer will not tolerate any error on.

The fixtures below are the REAL cell content, taken verbatim from the
position/error/limit JSON stored for
    8232-1_47_TA_Test Data_8-12-2026_1-12 PMTrimmed Correct.xls
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NAN = float("nan")

# --- real cell content -------------------------------------------------

POSITIONS = [
    -27.5, -27.0, -26.0, -25.0, -24.0, -23.0, -22.0, -21.0, -20.0, -19.0,
    -18.0, -17.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0,
    4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
    27.0, 27.5,
]

# Six unmeasured lead-in points, then the trimmed sweep.
ERRORS = [NAN] * 6 + [
    0.01537999999999995, -0.006673181818181773, 0.001214636363636279,
    0.005812454545454537, 0.014271272727272688, 0.014191090909090764,
    0.010880909090908997, 0.006987727272727007, 0.0017165454545455816,
    0.005608363636363745, 0.010234181818181476, 0.009679999999999911,
    0.005443818181817939, 0.008645636363636022, -0.0005265454545453352,
    0.0036572727272723426, 0.005761090909091049, 0.0038239090909089057,
    -0.0012002727272726332, 0.0027755454545452807, -0.0006066363636367811,
    -0.002763818181818145, -0.007931000000000132, -0.00021918181818136873,
    -0.0003913636363641615, -0.008059545454545791, -0.014219727272727134,
    -0.01148390909090935, -0.01342409090909058, -0.014352272727272464,
    -0.011219454545455143, -0.015209636363636925, -0.01429681818181816,
    -0.014326999999999757, -0.015428181818181841, -0.015440363636364474,
    -0.016756545454545524, -0.01509472727272776, -0.017274909090908785,
    -0.02026109090909145, -0.01308827272727342, -0.0176884545454552,
    -0.02206463636363587, -0.032480818181818805, -0.004486999999999242,
    -0.017492181818180796, -0.017399363636362963, -0.0007215454545441702,
    -0.01035172727272915, 0.00953709090908994, -0.004863000000000284,
]

# The graded band: no limits over the lead-in OR the trailing six points.
_BAND = [
    0.016907407407407413, 0.016388888888888894, 0.015870370370370375,
    0.015351851851851856, 0.014833333333333339, 0.01431481481481482,
    0.013796296296296301, 0.013277777777777782, 0.012759259259259264,
    0.012240740740740745, 0.011722222222222226, 0.011203703703703707,
    0.010685185185185188, 0.01016666666666667, 0.00964814814814815,
    0.009129629629629632, 0.008611111111111113, 0.008092592592592594,
    0.007574074074074075, 0.007055555555555556, 0.006537037037037037,
    0.0060185185185185185, 0.0055, 0.0060185185185185185,
    0.006537037037037037, 0.007055555555555556, 0.007574074074074075,
    0.008092592592592594, 0.008611111111111113, 0.009129629629629632,
    0.00964814814814815, 0.01016666666666667, 0.010685185185185188,
    0.011203703703703707, 0.011722222222222226, 0.012240740740740745,
    0.012759259259259264, 0.013277777777777782, 0.013796296296296301,
    0.01431481481481482, 0.014833333333333339, 0.015351851851851856,
    0.015870370370370375, 0.016388888888888894, 0.016907407407407413,
]
UPPER_LIMITS = [None] * 6 + _BAND + [None] * 6
LOWER_LIMITS = [None] * 6 + [-u for u in _BAND] + [None] * 6

LINEARITY_SPEC = 0.011203703703703707
TRAVEL_LENGTH = 55.0

# The numbers this file actually contains, recomputed over the MEASURED points.
TRUE_RAW_ERROR = 0.032480818181818805     # |worst| before the offset, at pos 21.0
TRUE_SHIFTED_ERROR = 0.027248813131313762  # |worst| after optimal_offset
STORED_OFFSET = 0.005232005050505041
STORED_FAIL_POINTS = 14                    # grading was never affected
STORED_MAX_DEV_POSITION = 21.0


def _track(errors=None, upper=None, lower=None, positions=None):
    return {
        "track_id": "Track A",
        "positions": POSITIONS if positions is None else positions,
        "errors": ERRORS if errors is None else errors,
        "upper_limits": UPPER_LIMITS if upper is None else upper,
        "lower_limits": LOWER_LIMITS if lower is None else lower,
        "travel_length": TRAVEL_LENGTH,
        "linearity_spec": LINEARITY_SPEC,
        "unit_length": 55.0,
        "untrimmed_resistance": 5279.0,
        "trimmed_resistance": 5278.0,
    }


def _analyze(**kw):
    from laser_trim_analyzer.core.analyzer import Analyzer
    return Analyzer().analyze_track(_track(**kw), model="8232-1",
                                    linearity_type="Absolute")


# --- the fixture really is the shape that broke ------------------------

def test_fixture_is_the_leading_nan_shape_that_broke():
    """Guard the fixture itself: this must stay a leading-NaN array."""
    assert len(ERRORS) == 57 and len(POSITIONS) == 57
    assert math.isnan(ERRORS[0]), "index 0 must be NaN — that is the trigger"
    assert sum(1 for e in ERRORS if math.isnan(e)) == 6
    assert all(math.isnan(e) for e in ERRORS[:6])
    assert not any(math.isnan(e) for e in ERRORS[6:])
    # the unmeasured points are exactly the ungraded ones
    assert UPPER_LIMITS[:6] == [None] * 6


def test_python_max_is_the_positional_lottery():
    """Pin the language behaviour this bug rode in on."""
    assert math.isnan(max(abs(e) for e in [NAN, 1.0, 2.0]))
    assert max(abs(e) for e in [1.0, 2.0, NAN]) == 2.0


# --- the magnitude must be recorded ------------------------------------

def test_linearity_error_is_recorded_despite_leading_nan():
    """THE BUG: a real, measured 2.4x-over-spec error came back as None."""
    t = _analyze()
    assert t.linearity_error is not None, (
        "linearity error magnitude was dropped — this is the zero-tolerance "
        "customer metric and the file plainly contains a value")
    assert not math.isnan(t.linearity_error)
    assert t.linearity_error == pytest.approx(TRUE_SHIFTED_ERROR, rel=1e-9)


def test_raw_linearity_error_is_recorded_despite_leading_nan():
    t = _analyze()
    assert t.raw_linearity_error is not None
    assert t.raw_linearity_error == pytest.approx(TRUE_RAW_ERROR, rel=1e-9)


def test_every_magnitude_column_is_populated():
    """All four magnitude fields fell together; all four must come back."""
    t = _analyze()
    for name in ("linearity_error", "raw_linearity_error",
                 "optimized_linearity_error", "max_deviation"):
        val = getattr(t, name)
        assert val is not None, f"{name} silently dropped to None"
        assert not math.isnan(val), f"{name} is NaN"
        assert val > 0, f"{name} came back as a manufactured zero"


def test_magnitude_agrees_with_the_position_that_survived():
    """max_deviation_position never broke; the magnitude must match it."""
    t = _analyze()
    assert t.max_deviation_position == STORED_MAX_DEV_POSITION
    idx = POSITIONS.index(t.max_deviation_position)
    expected = abs(ERRORS[idx] + t.optimal_offset)
    assert t.max_deviation == pytest.approx(expected, rel=1e-9)


def test_magnitude_exceeds_spec_so_the_part_is_not_reported_perfect():
    """The recovered number must be big enough to matter: 2.4x the spec."""
    t = _analyze()
    assert t.linearity_error > LINEARITY_SPEC * 2, (
        f"recovered {t.linearity_error} against spec {LINEARITY_SPEC}")


def test_magnitude_spans_every_measured_point_including_dead_zones():
    """The magnitude covers all MEASURED points, graded or not.

    These files have an ungraded dead zone at BOTH ends (no limits over the
    first six or last six points), so a track can pass the graded band while
    its worst measured point sits outside it. That is the column's existing
    meaning, not a defect of this fix: measured against the real DB on
    2026-08-31, 7,270 of 30,495 passing tracks since 2023 already store a
    magnitude above spec (p95 ratio 3.4x, max 16.5x), and 27 other models
    carry dead-zone tracks on exactly these semantics. Narrowing this to the
    graded band would silently redefine a column with 100k+ rows in it.
    """
    t = _analyze()
    # index 49 (position 21.0) is inside the graded band here...
    assert UPPER_LIMITS[49] is not None
    # ...but the trailing dead zone is measured and must still be in scope
    assert UPPER_LIMITS[-1] is None and not math.isnan(ERRORS[-1])
    shifted = [e + t.optimal_offset for e in ERRORS if not math.isnan(e)]
    assert t.linearity_error == pytest.approx(max(abs(e) for e in shifted))


# --- position must not decide the answer -------------------------------

def test_nan_position_does_not_change_the_magnitude():
    """The lottery is closed: leading, middle and trailing NaN must agree."""
    measured = ERRORS[6:]
    lead = [NAN] * 6 + measured
    trail = measured + [NAN] * 6
    middle = measured[:20] + [NAN] * 6 + measured[20:]

    got = []
    for errs in (lead, trail, middle):
        t = _analyze(errors=errs,
                     upper=[None] * len(errs), lower=[None] * len(errs))
        assert t.raw_linearity_error is not None
        got.append(t.raw_linearity_error)
    assert got[0] == pytest.approx(got[1]) == pytest.approx(got[2])
    assert got[0] == pytest.approx(TRUE_RAW_ERROR, rel=1e-9)


# --- grading must be untouched by the fix ------------------------------

def test_fix_does_not_disturb_the_pass_fail_disposition():
    """Grading always worked (NaN points had no limits). Keep it identical."""
    t = _analyze()
    assert t.linearity_fail_points == STORED_FAIL_POINTS
    assert t.linearity_pass is False
    assert t.optimal_offset == pytest.approx(STORED_OFFSET, rel=1e-9)


def test_clean_track_is_unchanged_by_the_fix():
    """No NaN anywhere: the magnitude must be exactly max(abs(shifted))."""
    errs = ERRORS[6:]
    t = _analyze(errors=errs, upper=[None] * len(errs),
                 lower=[None] * len(errs), positions=POSITIONS[6:])
    assert t.raw_linearity_error == pytest.approx(max(abs(e) for e in errs))


# --- the same NaN leak one layer down: the optimal offset --------------

def _errors_with_nan_inside_the_graded_band():
    """Move one unmeasured point INSIDE the limits, where 8232-1 never has one.

    _calculate_optimal_offset guards the limit columns against NaN but used
    to read errors[i] unguarded, so `lower_limits[i] - errors[i]` produced a
    (NaN, NaN) feasible interval that poisoned the whole boundary search.
    Measured against the real DB on 2026-08-31: all 1,842 tracks with a NULL
    optimal_offset carry exactly this shape, and every one was recorded FAIL
    on the strength of that failed computation.
    """
    errs = list(ERRORS)
    errs[20] = NAN                     # index 20 is inside the graded band
    assert UPPER_LIMITS[20] is not None
    return errs


def test_nan_inside_the_graded_band_still_yields_a_real_offset():
    t = _analyze(errors=_errors_with_nan_inside_the_graded_band())
    assert t.optimal_offset is not None, (
        "one unmeasured point in the band must not destroy the offset "
        "computed from every other point")
    assert not math.isnan(t.optimal_offset)


def test_nan_inside_the_graded_band_still_yields_a_magnitude():
    t = _analyze(errors=_errors_with_nan_inside_the_graded_band())
    assert t.linearity_error is not None
    assert not math.isnan(t.linearity_error)
    assert t.linearity_error > 0


def test_unmeasured_point_in_the_band_is_still_counted_as_a_failure():
    """The guard must not soften the zero-tolerance rule.

    An unmeasured point inside the graded band cannot be shown to be in
    spec, so it still counts as a fail — it is merely no longer allowed to
    pick the offset for every other point too.
    """
    clean = _analyze()
    holed = _analyze(errors=_errors_with_nan_inside_the_graded_band())
    assert holed.linearity_fail_points >= clean.linearity_fail_points
    assert holed.linearity_pass is False


def test_offset_guard_ignores_only_the_unmeasured_point():
    """With the hole at an index that was already failing, the offset must
    stay close to the clean answer — the guard drops one interval, not the
    optimisation."""
    clean = _analyze()
    holed = _analyze(errors=_errors_with_nan_inside_the_graded_band())
    assert abs(holed.optimal_offset - clean.optimal_offset) < abs(clean.optimal_offset) + 0.01


# --- the genuinely uncomputable case must be DISCLOSED, not silent -----

def test_all_nan_track_is_disclosed_rather_than_silently_null():
    """Nothing measured at all -> no magnitude, but it must say so.

    A silent NULL on the zero-tolerance metric is the defect either way,
    so an uncomputable magnitude has to leave a reason behind and must not
    be reported as a passing track.
    """
    errs = [NAN] * len(POSITIONS)
    t = _analyze(errors=errs)
    assert t.linearity_error is None
    assert t.linearity_spec_warning, (
        "an uncomputable magnitude must record WHY, not just store NULL")
    assert "measure" in t.linearity_spec_warning.lower()
    assert t.linearity_pass is not True, "must never read as a passing track"


def test_all_nan_track_does_not_manufacture_a_zero():
    errs = [NAN] * len(POSITIONS)
    t = _analyze(errors=errs)
    for name in ("linearity_error", "raw_linearity_error", "max_deviation"):
        assert getattr(t, name) != 0.0, f"{name} manufactured a perfect 0.0"


# --- the silencer itself must stop being silent ------------------------

def test_nan_coercion_logs_when_it_drops_a_computed_value(caplog):
    """_nan_means_missing turned a computation failure into a missing value
    with no trace. It may still coerce, but it must leave a log line."""
    import logging
    from laser_trim_analyzer.core.models import TrackData, AnalysisStatus

    with caplog.at_level(logging.WARNING):
        t = TrackData(
            track_id="T", status=AnalysisStatus.FAIL, travel_length=1.0,
            linearity_spec=0.01, sigma_gradient=0.001, sigma_threshold=0.01,
            sigma_pass=True, optimal_offset=0.0, linearity_error=NAN,
            linearity_pass=False, linearity_fail_points=0,
        )
    assert t.linearity_error is None
    assert any("linearity_error" in r.message for r in caplog.records), (
        "NaN->None coercion must name the field it dropped")
