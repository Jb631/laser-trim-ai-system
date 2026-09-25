"""The trim-vs-FT overlay, moved off V5's Compare page (2026-08-31).

V5's Compare page (gui/pages/compare.py) drew the only trim-vs-final-test
overlay in the app — the last feature keeping V5 alive as a fallback. These
tests pin the semantics the V6 unit chart has to honour, which are easy to
get subtly wrong:

  * the LINKAGE is final_test_results.linked_trim_id -> analysis_results.id,
    and only at match_confidence >= 0.70;
  * a unit can be final-tested more than once, so "the linked FT" means the
    NEWEST one, and the chart has to say so;
  * an FT sweep carries its OWN offset / slope / limits. Grading it with the
    TRIM correction would draw a curve nobody ever measured or judged.
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest


_POS = [float(i) for i in range(8)]
_TRIM_ERR = [0.001] * 8
_FT_ERR = [0.010] * 8
_LIM_HI = [0.025] * 8
_LIM_LO = [-0.025] * 8
_TRIM_OFFSET = -0.5      # deliberately huge: if it leaks onto the FT trace,
_FT_OFFSET = 0.002       # the FT curve lands nowhere near its own band


def _db(tmp_path, name="ft_overlay.db"):
    from laser_trim_analyzer.database.manager import DatabaseManager

    return DatabaseManager(tmp_path / name)


def _seed(db, *, n_ft=1, confidence=0.99, ft_sweep=True, ft_track_id="default",
          trim_track_id="TRK1"):
    """One trim analysis plus n_ft linked final tests, newest last."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, FinalTestResult as DBFT,
        FinalTestTrack as DBFTT, StatusType, SystemType)

    when = datetime(2026, 3, 1)
    with db.session() as s:
        ar = DBAR(filename="t.xls", file_path="/t/t",
                  file_hash="trim".ljust(64, "0"), model="8340-1", serial="77",
                  system=SystemType.A, file_date=when, timestamp=when,
                  overall_status=StatusType.PASS, has_multi_tracks=False,
                  processing_time=0.1)
        s.add(ar)
        s.flush()
        s.add(DBTR(analysis_id=ar.id, track_id=trim_track_id,
                   status=StatusType.PASS,
                   position_data=_POS, error_data=_TRIM_ERR,
                   upper_limits=_LIM_HI, lower_limits=_LIM_LO,
                   optimal_offset=_TRIM_OFFSET, optimal_slope=0.0,
                   linearity_pass=True, linearity_fail_points=0))
        aid = ar.id
        for i in range(n_ft):
            tested = datetime(2026, 4, 1 + i)
            ft = DBFT(filename=f"ft{i}.xls", model="8340-1", serial="77",
                      test_date=tested, file_date=tested, timestamp=tested,
                      overall_status=StatusType.PASS, linked_trim_id=aid,
                      match_confidence=confidence)
            s.add(ft)
            s.flush()
            s.add(DBFTT(final_test_id=ft.id, track_id=ft_track_id,
                        status=StatusType.PASS,
                        position_data=_POS if ft_sweep else None,
                        error_data=[e + i * 0.001 for e in _FT_ERR] if ft_sweep else None,
                        upper_limits=_LIM_HI if ft_sweep else None,
                        lower_limits=_LIM_LO if ft_sweep else None,
                        optimal_offset=_FT_OFFSET, optimal_slope=0.0,
                        linearity_pass=True))
        s.commit()
    return aid


def test_resolves_the_linked_final_test(tmp_path):
    from laser_trim_analyzer.core.ft_overlay import load_ft_overlay

    db = _db(tmp_path)
    aid = _seed(db)
    ov = load_ft_overlay(db, aid)
    assert ov["available"] is True
    assert ov["confidence"] == pytest.approx(0.99)
    assert ov["n_links"] == 1


def test_no_link_is_a_plain_reason_not_an_empty_chart(tmp_path):
    from laser_trim_analyzer.core.ft_overlay import load_ft_overlay

    db = _db(tmp_path)
    aid = _seed(db, n_ft=0)
    ov = load_ft_overlay(db, aid)
    assert ov["available"] is False
    assert ov["reason"] and "final test" in ov["reason"].lower()


def test_low_confidence_link_is_refused_and_says_why(tmp_path):
    """0.70 is the threshold every corrected trim/FT number in the app uses."""
    from laser_trim_analyzer.core.ft_overlay import load_ft_overlay

    db = _db(tmp_path)
    aid = _seed(db, confidence=0.55)
    ov = load_ft_overlay(db, aid)
    assert ov["available"] is False
    assert "0.55" in ov["reason"] and "0.70" in ov["reason"]


def test_link_without_a_sweep_is_refused_and_says_why(tmp_path):
    from laser_trim_analyzer.core.ft_overlay import load_ft_overlay

    db = _db(tmp_path)
    aid = _seed(db, ft_sweep=False)
    ov = load_ft_overlay(db, aid)
    assert ov["available"] is False
    assert "sweep" in ov["reason"].lower()


def test_a_position_column_that_does_not_sweep_is_refused(tmp_path):
    """Found on real data (the 6828 family, ~1% of linked FT sweeps): the FT
    position column holds an error-like column that oscillates across 0.03
    against a 340-degree trim travel. Mapped onto the trim axis it draws a
    smooth diagonal that looks like a measurement. Refuse it and say why."""
    from laser_trim_analyzer.core.ft_overlay import is_sweep_axis, load_ft_overlay
    from laser_trim_analyzer.database.models import FinalTestTrack as DBFTT

    assert is_sweep_axis([0.0, 1.0, 2.0, 3.0]) is True
    assert is_sweep_axis([0.0, -0.02, 0.03, -0.01, 0.02]) is False

    db = _db(tmp_path)
    aid = _seed(db)
    with db.session() as s:
        t = s.query(DBFTT).first()
        t.position_data = [0.0, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.0]
        s.commit()

    ov = load_ft_overlay(db, aid)
    assert ov["available"] is False
    assert "position" in ov["reason"].lower()


def test_multiple_final_tests_show_the_newest_and_say_so(tmp_path):
    """Same serial can be tested more than once — that is valid data."""
    from laser_trim_analyzer.core.ft_overlay import load_ft_overlay

    db = _db(tmp_path)
    aid = _seed(db, n_ft=3)
    ov = load_ft_overlay(db, aid)
    assert ov["available"] is True
    assert ov["n_links"] == 3
    assert ov["date"] == "2026-04-03", ov["date"]          # the newest
    assert "3" in ov["label"] and "newest" in ov["label"].lower(), ov["label"]
    # ...and it is the newest sweep, not merely the newest header row.
    assert ov["errors"][0] == pytest.approx(_FT_ERR[0] + 0.002)


def test_ft_trace_is_corrected_with_its_own_offset_never_the_trims(tmp_path):
    """The FT station's adjustment is its own. Applying the trim offset would
    draw a curve no instrument ever produced."""
    from laser_trim_analyzer.core.ft_overlay import load_ft_overlay

    db = _db(tmp_path)
    aid = _seed(db)
    ov = load_ft_overlay(db, aid)
    assert ov["offset"] == pytest.approx(_FT_OFFSET)
    assert ov["corrected"][0] == pytest.approx(_FT_ERR[0] + _FT_OFFSET)
    # The trim offset is -0.5; if it leaked in, this would be about -0.49.
    assert ov["corrected"][0] > 0


def test_ft_carries_its_own_spec_limits(tmp_path):
    from laser_trim_analyzer.core.ft_overlay import load_ft_overlay

    db = _db(tmp_path)
    aid = _seed(db)
    ov = load_ft_overlay(db, aid)
    assert ov["upper_limits"] == pytest.approx(_LIM_HI)
    assert ov["lower_limits"] == pytest.approx(_LIM_LO)


def test_ft_track_is_paired_to_the_trim_track_by_designator(tmp_path):
    """TRK1 <-> Track A. Pairing by index mismatches multi-track units."""
    from laser_trim_analyzer.core.ft_overlay import pair_ft_track

    ft_tracks = [{"track_id": "B"}, {"track_id": "A"}]
    assert pair_ft_track("TRK1", ft_tracks)["track_id"] == "A"
    assert pair_ft_track("Track B", ft_tracks)["track_id"] == "B"
    # Legacy single-track data on either side still pairs.
    assert pair_ft_track("default", ft_tracks)["track_id"] == "B"
    assert pair_ft_track("TRK1", [{"track_id": "default"}])["track_id"] == "default"


# ---- positions_on_trim_axis: ONE alignment, both callers (fix round 2, 2026-09-25) --------------
# The final-test sweep placed on the trim sweep's axis, index for index, and how it got there:
# "as measured", "shifted", "rescaled", or (None, None) when there is no honest common axis.
# rework_load compares the stations position by position -- a MEASUREMENT, never rescaled onto a
# guessed axis. The overlay draws a labelled PICTURE, so rescale=True keeps its old mapping of a
# column recorded in other units. Both shift a final test measured from another zero: the overlay
# used to draw those (8340-1, 8397-2 and 41 more models) half off the trim travel.

def test_positions_are_left_alone_when_the_two_sweeps_share_a_scale():
    """No cosmetic rescaling when none is needed — that would distort the FT
    trace against a trim sweep it already lines up with."""
    from laser_trim_analyzer.core.ft_overlay import positions_on_trim_axis

    trim = [float(i) for i in range(100)]
    ft = [float(i) + 0.5 for i in range(98)]
    for picture in (True, False):
        out, how = positions_on_trim_axis(ft, trim, rescale=picture)
        assert how == "as measured"
        assert out == pytest.approx(ft)


def test_a_shorter_ft_sweep_is_not_stretched_to_fill_the_trim_axis():
    """Found on real data (analysis_id 104050): the FT sweep covers +/-14 of a
    +/-20 trim travel. That is a shorter SWEEP, not a different scale — and
    stretching it would move every FT error to a position it was never
    measured at."""
    from laser_trim_analyzer.core.ft_overlay import positions_on_trim_axis

    trim = [-20.0 + i for i in range(41)]
    ft = [-14.0 + i * 0.5 for i in range(57)]
    for picture in (True, False):
        out, how = positions_on_trim_axis(ft, trim, rescale=picture)
        assert how == "as measured"
        assert out[0] == pytest.approx(-14.0) and out[-1] == pytest.approx(14.0)


def test_a_picture_rescales_a_column_recorded_in_other_units_and_says_so():
    """FT files can be recorded in a different unit entirely. The overlay maps
    such a sweep onto the trim's span AND admits that it did -- a measurement
    never does (next test)."""
    from laser_trim_analyzer.core.ft_overlay import positions_on_trim_axis

    trim = [float(i) for i in range(100)]          # 0..99
    ft = [i * 0.01 for i in range(100)]            # 0..0.99 — 100x smaller
    out, how = positions_on_trim_axis(ft, trim, rescale=True)
    assert how == "rescaled"
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(99.0)


def test_the_same_span_from_a_different_zero_is_shifted_onto_the_trim_axis_8340_1_shape():
    """8340-1: the FT file counts 0 to 0.61, the trim file -0.305 to 0.305 -- the same travel,
    another origin. Compared raw, only half the travel would overlap, and the wrong half."""
    from laser_trim_analyzer.core.ft_overlay import positions_on_trim_axis

    ft = [round(0.01 * i, 2) for i in range(62)]                    # 0.00 .. 0.61
    trim = [-0.305 + 0.61 * i / 62 for i in range(63)]              # -0.305 .. 0.305
    for picture in (True, False):
        out, how = positions_on_trim_axis(ft, trim, rescale=picture)
        assert how == "shifted"
        assert len(out) == len(ft)                                  # index for index
        assert out[0] == pytest.approx(-0.305) and out[-1] == pytest.approx(0.305)
        assert out[31] == pytest.approx(0.005)                      # every point moves by the same amount


def test_the_same_span_from_a_different_zero_is_shifted_onto_the_trim_axis_8397_2_shape():
    """8397-2: some FT files count 0.05 to 240.05 against the trim's -120 to 120."""
    from laser_trim_analyzer.core.ft_overlay import positions_on_trim_axis

    ft = [0.05 + 4.0 * i for i in range(61)]                        # 0.05 .. 240.05
    trim = [-120.0 + 4.0 * i for i in range(61)]                    # -120 .. 120
    for picture in (True, False):
        out, how = positions_on_trim_axis(ft, trim, rescale=picture)
        assert how == "shifted"
        assert out == pytest.approx(trim)


def test_a_genuinely_shorter_ft_sweep_is_left_where_it_was_measured_6607_shape():
    """6607: FT sweeps +/-14 of the trim's +/-22 -- a shorter sweep centred on the same zero. Moving
    or stretching it would put every FT error where it was never measured."""
    from laser_trim_analyzer.core.ft_overlay import positions_on_trim_axis

    ft = [-14.0 + 0.5 * i for i in range(57)]
    trim = [-22.0 + i for i in range(45)]
    out, how = positions_on_trim_axis(ft, trim)
    assert how == "as measured"
    assert out == pytest.approx(ft)


def test_a_matching_sweep_is_left_alone():
    from laser_trim_analyzer.core.ft_overlay import positions_on_trim_axis

    ft = [-28.0 + i for i in range(57)]
    trim = [-27.5 + 0.5 * i for i in range(111)]
    out, how = positions_on_trim_axis(ft, trim)
    assert how == "as measured" and out == pytest.approx(ft)


def test_a_column_an_order_of_magnitude_off_the_trim_span_is_no_axis_for_a_measurement():
    """25 real 8397-2 files hold one stray value and then zeros in their position column (span
    0.046 against the trim's 240). A measurement is never rescaled onto a guessed axis: None."""
    from laser_trim_analyzer.core.ft_overlay import positions_on_trim_axis

    ft = [-0.046] + [0.0] * 24
    trim = [-120.0 + 10.0 * i for i in range(25)]
    assert positions_on_trim_axis(ft, trim) == (None, None)
    for picture in (True, False):                                   # fewer than 2 trim positions
        assert positions_on_trim_axis([0.0, 1.0], [5.0], rescale=picture) == (None, None)


def test_a_missing_position_stays_missing_and_keeps_its_index():
    """The FT arrays are index-aligned (errors, limits, graded window): dropping a None, as the
    old align_positions did, would slide every later point onto its neighbour's error."""
    from laser_trim_analyzer.core.ft_overlay import positions_on_trim_axis

    ft = [0.0, None, 0.2, 0.3, 0.4, 0.5, 0.6]
    trim = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    for picture in (True, False):
        out, how = positions_on_trim_axis(ft, trim, rescale=picture)
        assert how == "shifted"
        assert out[1] is None and len(out) == 7
        assert out[0] == pytest.approx(-0.3) and out[6] == pytest.approx(0.3)


def test_spans_within_a_quarter_of_each_other_are_the_same_travel():
    """_SAME_SPAN at its boundary: a final test 1.24x the trim span, counted from another zero, is
    the same travel shifted; at 1.26x it is a different sweep and is left where it was measured."""
    from laser_trim_analyzer.core.ft_overlay import positions_on_trim_axis

    trim = [-10.0 + i for i in range(21)]                            # span 20, centre 0
    for factor, expected in ((1.24, "shifted"), (1.26, "as measured")):
        ft = [100.0 + 20.0 * factor * i / 20 for i in range(21)]    # centre far from 0
        assert positions_on_trim_axis(ft, trim)[1] == expected, factor


def test_centres_more_than_a_tenth_of_the_span_apart_count_from_another_zero():
    """_OTHER_ZERO at its boundary: the same span with its centre 0.09 of the trim span away is
    left alone; 0.11 away it is shifted."""
    from laser_trim_analyzer.core.ft_overlay import positions_on_trim_axis

    trim = [-10.0 + i for i in range(21)]                            # span 20
    for offset, expected in ((0.09 * 20, "as measured"), (0.11 * 20, "shifted")):
        ft = [p + offset for p in trim]
        assert positions_on_trim_axis(ft, trim)[1] == expected, offset


# ---- the overlay itself, on both real shapes -----------------------------------------------------

def _seed_axes(db, trim_positions, ft_positions):
    """One trim analysis on `trim_positions` and one linked final test on `ft_positions`, flat
    errors inside flat limits -- only the axes differ. Returns the analysis id."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, FinalTestResult as DBFT,
        FinalTestTrack as DBFTT, StatusType, SystemType)

    when, tested = datetime(2026, 3, 1), datetime(2026, 4, 1)
    n_t, n_f = len(trim_positions), len(ft_positions)
    with db.session() as s:
        ar = DBAR(filename="axes.xls", file_path="/t/axes", file_hash="axes".ljust(64, "0"),
                  model="8340-1", serial="77", system=SystemType.B, file_date=when,
                  timestamp=when, overall_status=StatusType.PASS, has_multi_tracks=False,
                  processing_time=0.1)
        s.add(ar)
        s.flush()
        s.add(DBTR(analysis_id=ar.id, track_id="Track A", status=StatusType.PASS,
                   position_data=list(trim_positions), error_data=[0.001] * n_t,
                   upper_limits=[0.025] * n_t, lower_limits=[-0.025] * n_t,
                   optimal_offset=0.0, optimal_slope=0.0, linearity_pass=True,
                   linearity_fail_points=0))
        ft = DBFT(filename="axes_ft.xls", model="8340-1", serial="77", test_date=tested,
                  file_date=tested, timestamp=tested, overall_status=StatusType.PASS,
                  linked_trim_id=ar.id, match_confidence=0.99)
        s.add(ft)
        s.flush()
        s.add(DBFTT(final_test_id=ft.id, track_id="default", status=StatusType.PASS,
                    position_data=list(ft_positions), error_data=[0.010] * n_f,
                    upper_limits=[0.025] * n_f, lower_limits=[-0.025] * n_f,
                    optimal_offset=0.0, optimal_slope=0.0, linearity_pass=True))
        s.commit()
        return ar.id


@pytest.mark.parametrize("trim,ft", [
    ([-0.305 + 0.01 * i for i in range(62)], [0.01 * i for i in range(62)]),        # 8340-1
    ([-120.0 + 4.0 * i for i in range(61)], [0.05 + 4.0 * i for i in range(61)]),   # 8397-2
], ids=["8340-1", "8397-2"])
def test_the_overlay_draws_a_final_test_measured_from_another_zero_over_the_trim_travel(
        tmp_path, trim, ft):
    from laser_trim_analyzer.core.ft_overlay import load_ft_overlay

    db = _db(tmp_path)
    aid = _seed_axes(db, trim, ft)
    ov = load_ft_overlay(db, aid, trim_track_id="Track A", trim_positions=trim)
    assert ov["available"] is True
    assert ov["positions"][0] == pytest.approx(trim[0])             # over the trim travel,
    assert ov["positions"][-1] == pytest.approx(trim[-1])           # not half off its right
    assert len(ov["positions"]) == len(ov["errors"]) == len(ft)
    assert ov["shifted"] is True and ov["rescaled"] is False
    assert "shifted to the trim's zero" in ov["label"]


def test_the_overlay_still_rescales_a_column_in_other_units_and_says_so(tmp_path):
    from laser_trim_analyzer.core.ft_overlay import load_ft_overlay

    db = _db(tmp_path)
    trim = [float(i) for i in range(100)]
    aid = _seed_axes(db, trim, [i * 0.01 for i in range(100)])
    ov = load_ft_overlay(db, aid, trim_track_id="Track A", trim_positions=trim)
    assert ov["rescaled"] is True and ov["shifted"] is False
    assert ov["positions"][-1] == pytest.approx(99.0)
    assert "x aligned to trim travel" in ov["label"]


def test_the_overlay_with_no_trim_axis_draws_the_final_test_where_it_was_measured(tmp_path):
    from laser_trim_analyzer.core.ft_overlay import load_ft_overlay

    db = _db(tmp_path)
    ft = [0.01 * i for i in range(62)]
    aid = _seed_axes(db, [-0.305 + 0.01 * i for i in range(62)], ft)
    ov = load_ft_overlay(db, aid, trim_track_id="Track A", trim_positions=None)
    assert ov["available"] is True
    assert ov["shifted"] is False and ov["rescaled"] is False
    assert ov["positions"] == pytest.approx(ft)
    assert "shifted" not in ov["label"] and "aligned" not in ov["label"]


def test_export_document_draws_the_ft_overlay_with_its_own_band(tmp_path):
    """The print document gets the overlay too, styled apart from the trim
    trace and carrying the FT's own limits."""
    from laser_trim_analyzer.core.ft_overlay import load_ft_overlay
    from laser_trim_analyzer.export.unit_chart import build_unit_export_figure

    db = _db(tmp_path)
    aid = _seed(db)
    ov = load_ft_overlay(db, aid)
    data = {"position_data": _POS, "error_data": _TRIM_ERR,
            "upper_limits": _LIM_HI, "lower_limits": _LIM_LO,
            "optimal_offset": 0.0, "linearity_pass": True, "sigma_pass": True}
    fig = build_unit_export_figure({"model": "8340-1", "serial": "77", "n_tracks": 1},
                                   data, fail_points=[], kind="trim", ft_overlay=ov)
    labels = [(l.get_label() or "") for l in fig.axes[0].get_lines()]
    ft_lines = [l for l in labels if l.lower().startswith("final test")]
    assert ft_lines, labels
    # Its own band is drawn, not the trim's reused.
    assert any("limit" in l.lower() and "final test" in l.lower() for l in labels), labels
    # The document names the linked file so the overlay is traceable.
    txt = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("Final test:" in t for t in txt), txt


def test_export_document_without_overlay_is_unchanged(tmp_path):
    from laser_trim_analyzer.export.unit_chart import build_unit_export_figure

    data = {"position_data": _POS, "error_data": _TRIM_ERR,
            "upper_limits": _LIM_HI, "lower_limits": _LIM_LO,
            "optimal_offset": 0.0, "linearity_pass": True, "sigma_pass": True}
    fig = build_unit_export_figure({"model": "M", "serial": "1", "n_tracks": 1},
                                   data, fail_points=[], kind="trim")
    labels = [(l.get_label() or "").lower() for l in fig.axes[0].get_lines()]
    assert not [l for l in labels if l.startswith("final test")]


def test_an_ungraded_run_whose_edge_position_is_missing_is_shaded_over_what_is_known():
    """M7 (final review, 2026-09-25): the overlay keeps a missing position in place (so later
    points never slide onto their neighbour), which handed axvspan a None bound -- a TypeError.
    None of the stored sweeps carries one today; a run is now shaded over its known positions,
    and a run with no known position is not shaded at all."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from laser_trim_analyzer.export.unit_chart import draw_ft_overlay

    fig = Figure()
    ax = fig.add_subplot()
    ov = {"available": True, "label": "Final test", "station_flags": [],
          "positions": [None, 0.5, 1.0, 2.0, 3.0, float("nan"), 4.0, 4.5, None, 6.0, 7.0, None],
          "corrected": [0.0] * 12, "errors": [], "upper_limits": [], "lower_limits": [],
          "ungraded_indices": [0, 1, 2, 5, 6, 7, 8, 10, 11]}
    draw_ft_overlay(ax, ov)                                    # used to raise TypeError
    spans = sorted((round(p.get_x(), 6), round(p.get_x() + p.get_width(), 6)) for p in ax.patches)
    assert spans == [(0.5, 1.0), (4.0, 4.5)]          # the run ending at 7.0 knows one position only
