"""A V5 chart export never calls a track with no error data a linearity PASS
(2026-09-25).

`gui/pages/export.py::_export_single_chart` / `_export_multi_page_pdf`, and
`gui/pages/analyze.py::_export_comprehensive_chart`, each "RECALCULATE fail
points using actual spec limits" with their own copy of the same loop: start
`actual_fail_count = 0`, and only add to it inside
`if upper_limits and lower_limits and track.error_data:`. When a track has no
`error_data` (blank measurement -- nothing was captured, or the sweep is
empty), that `if` is never entered, so the accumulator stays at its initial
0 -- and `corrected_values['linearity_pass'] = actual_fail_count == 0` reads
True. A blank measurement is ungraded, never a pass (CLAUDE.md).

The fix lifts the shared computation into `core.analyzer.
linearity_verdict_from_limits`, which returns
`{'fail_points': None, 'linearity_pass': None}` when there is nothing to
grade, and teaches the two pages' `_plot_metrics_summary`/`_plot_status_display`
(mirroring the UNTRIMMED / failed_processing branches already there, from the
prior "never invents a verdict" fix) to render "NOT GRADED" / "No linearity
data" for that sentinel, instead of feeding None into the PASS/WARNING/FAIL
trichotomy.
"""
from datetime import datetime
from pathlib import Path

import pytest

from laser_trim_analyzer.core.models import (
    AnalysisResult, AnalysisStatus, FileMetadata, RiskCategory, SystemType, TrackData,
)

MODEL = "NODATA-1"


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _no_error_data_track(status=AnalysisStatus.PASS, **overrides) -> TrackData:
    """A track carrying a real (non-UNTRIMMED, non-failed-processing) status,
    real sigma numbers, but NO measured error curve -- the shape the bug
    mishandles. sigma_pass=True is deliberate: pre-fix, this is the setup
    that makes the chart render 'STATUS: PASS' on a track with nothing to
    grade (sigma_pass and (bug's) linearity_pass=True)."""
    fields = dict(
        track_id="default",
        status=status,
        travel_length=1.0,
        linearity_spec=0.01,
        sigma_gradient=0.01,
        sigma_threshold=0.03,
        sigma_pass=True,
        optimal_offset=0.0,
        linearity_error=None,
        linearity_pass=None,
        linearity_fail_points=0,
        risk_category=RiskCategory.LOW,
        # The bug's trigger: no measured curve at all.
        position_data=None,
        error_data=None,
        upper_limits=None,
        lower_limits=None,
    )
    fields.update(overrides)
    return TrackData(**fields)


def _graded_track(status, sigma_pass, linearity_pass, fail_points=0) -> TrackData:
    """A normally-graded track with a real measured curve -- the regression
    guard population (must still show a real verdict after the fix)."""
    n = 12
    return TrackData(
        track_id="default",
        status=status,
        travel_length=1.0,
        linearity_spec=0.01,
        sigma_gradient=0.01,
        sigma_threshold=0.03,
        sigma_pass=sigma_pass,
        optimal_offset=0.0,
        linearity_error=0.005,
        linearity_pass=linearity_pass,
        linearity_fail_points=fail_points,
        position_data=[float(i) for i in range(n)],
        error_data=[0.001] * n,
        upper_limits=[0.01] * n,
        lower_limits=[-0.01] * n,
        risk_category=RiskCategory.LOW,
    )


def _wrap(track: TrackData, status: AnalysisStatus, serial: str) -> AnalysisResult:
    return AnalysisResult(
        metadata=FileMetadata(
            filename=f"{serial}.xls",
            file_path=Path(f"/fake/{serial}.xls"),
            file_date=datetime(2026, 1, 1),
            model=MODEL,
            serial=serial,
            system=SystemType.B,
        ),
        overall_status=status,
        processing_time=0.1,
        tracks=[track],
    )


# ---------------------------------------------------------------------------
# the shared pure helper -- core.analyzer.linearity_verdict_from_limits
# ---------------------------------------------------------------------------

def test_no_error_data_is_ungraded_not_a_pass():
    from laser_trim_analyzer.core.analyzer import linearity_verdict_from_limits
    track = _no_error_data_track()
    result = linearity_verdict_from_limits(track)
    assert result == {"fail_points": None, "linearity_pass": None}


def test_empty_error_data_list_is_ungraded_too():
    """[] is falsy, same as None -- must not slip past an `is None` check."""
    from laser_trim_analyzer.core.analyzer import linearity_verdict_from_limits
    track = _no_error_data_track(error_data=[], position_data=[],
                                 upper_limits=[0.01] * 3, lower_limits=[-0.01] * 3)
    result = linearity_verdict_from_limits(track)
    assert result == {"fail_points": None, "linearity_pass": None}


def test_graded_track_with_all_points_in_band_passes():
    from laser_trim_analyzer.core.analyzer import linearity_verdict_from_limits
    track = _graded_track(AnalysisStatus.PASS, True, True)
    result = linearity_verdict_from_limits(track)
    assert result == {"fail_points": 0, "linearity_pass": True}


def test_graded_track_with_out_of_band_points_fails_and_counts_them():
    from laser_trim_analyzer.core.analyzer import linearity_verdict_from_limits
    track = _graded_track(AnalysisStatus.FAIL, False, False, fail_points=3)
    # Push 3 of the 12 points outside the +/-0.01 band.
    track.error_data = [0.001] * 9 + [0.05, -0.05, 0.05]
    result = linearity_verdict_from_limits(track)
    assert result == {"fail_points": 3, "linearity_pass": False}


def test_falls_back_to_flat_spec_band_when_no_per_point_limits():
    from laser_trim_analyzer.core.analyzer import linearity_verdict_from_limits
    track = _graded_track(AnalysisStatus.FAIL, False, False)
    track.upper_limits = None
    track.lower_limits = None
    track.error_data = [0.001] * 11 + [0.5]      # one point way outside +/-0.01
    result = linearity_verdict_from_limits(track)
    assert result == {"fail_points": 1, "linearity_pass": False}


def test_theory_and_slope_shift_errors_before_grading():
    """Regression pin for the existing (correct) k-rotation behaviour: with
    theory_volts and a nonzero optimal_slope, points are shifted by
    theory[i]*k + offset before comparison, not just by the flat offset."""
    from laser_trim_analyzer.core.analyzer import linearity_verdict_from_limits
    n = 5
    track = TrackData(
        track_id="default", status=AnalysisStatus.PASS, travel_length=1.0,
        linearity_spec=0.01, optimal_offset=0.0, optimal_slope=1.0,
        theory_volts=[0.0, 0.0, 0.0, 0.0, 0.02],   # last point: +0.02*k = +0.02
        position_data=[float(i) for i in range(n)],
        error_data=[0.0, 0.0, 0.0, 0.0, -0.015],   # -0.015 + 0.02 = 0.005: back in band
        upper_limits=[0.01] * n, lower_limits=[-0.01] * n,
    )
    result = linearity_verdict_from_limits(track)
    assert result == {"fail_points": 0, "linearity_pass": True}


# ---------------------------------------------------------------------------
# _plot_metrics_summary / _plot_status_display -- the render-side decision
# ---------------------------------------------------------------------------

UNGRADED = {"fail_points": None, "linearity_pass": None}


@pytest.mark.parametrize("page_module,page_class", [
    ("laser_trim_analyzer.gui.pages.export", "ExportPage"),
    ("laser_trim_analyzer.gui.pages.analyze", "AnalyzePage"),
])
def test_plot_status_display_no_error_data_is_not_graded(page_module, page_class):
    import importlib
    from matplotlib.figure import Figure
    Page = getattr(importlib.import_module(page_module), page_class)

    track = _no_error_data_track()
    fig = Figure()
    ax = fig.add_subplot(111)
    Page._plot_status_display(None, ax, track, result=None, corrected_values=UNGRADED)

    joined = "\n".join(t.get_text() for t in ax.texts)
    assert "STATUS: PASS" not in joined, joined
    assert "STATUS: FAIL" not in joined, joined
    assert "STATUS: WARNING" not in joined, joined
    assert "STATUS: NOT GRADED" in joined.upper(), joined


@pytest.mark.parametrize("page_module,page_class", [
    ("laser_trim_analyzer.gui.pages.export", "ExportPage"),
    ("laser_trim_analyzer.gui.pages.analyze", "AnalyzePage"),
])
def test_plot_metrics_summary_no_error_data_is_not_graded(page_module, page_class):
    import importlib
    from matplotlib.figure import Figure
    Page = getattr(importlib.import_module(page_module), page_class)

    track = _no_error_data_track()
    fig = Figure()
    ax = fig.add_subplot(111)
    Page._plot_metrics_summary(None, ax, track, corrected_values=UNGRADED)  # must not raise

    joined = "\n".join(t.get_text() for t in ax.texts)
    assert "Linearity Pass: " not in joined, joined
    assert "not graded" in joined.lower(), joined


@pytest.mark.parametrize("page_module,page_class", [
    ("laser_trim_analyzer.gui.pages.export", "ExportPage"),
    ("laser_trim_analyzer.gui.pages.analyze", "AnalyzePage"),
])
def test_plot_status_display_graded_tracks_are_unaffected(page_module, page_class):
    """Regression guard either side of the new branch."""
    import importlib
    from matplotlib.figure import Figure
    Page = getattr(importlib.import_module(page_module), page_class)

    fail_cv = {"fail_points": 3, "linearity_pass": False}
    fail_track = _graded_track(AnalysisStatus.FAIL, False, False, fail_points=3)
    fig = Figure()
    ax = fig.add_subplot(111)
    Page._plot_status_display(None, ax, fail_track, result=None, corrected_values=fail_cv)
    assert "STATUS: FAIL" in "\n".join(t.get_text() for t in ax.texts)

    pass_cv = {"fail_points": 0, "linearity_pass": True}
    pass_track = _graded_track(AnalysisStatus.PASS, True, True)
    fig2 = Figure()
    ax2 = fig2.add_subplot(111)
    Page._plot_status_display(None, ax2, pass_track, result=None, corrected_values=pass_cv)
    assert "STATUS: PASS" in "\n".join(t.get_text() for t in ax2.texts)


# ---------------------------------------------------------------------------
# the three named orchestrator functions -- the decision they actually draw
# ---------------------------------------------------------------------------

def _spy_page(Page):
    """A bare instance (skips __init__ -- no Tk) with the two verdict-drawing
    methods replaced by spies that record `corrected_values`, and the other
    two (unrelated to this bug) stubbed to no-ops."""
    seen = {}
    page = object.__new__(Page)

    def _metrics_spy(self, ax, track, corrected_values=None):
        seen["metrics"] = corrected_values

    def _status_spy(self, ax, track, result, corrected_values=None):
        seen["status"] = corrected_values

    Page._plot_metrics_summary = _metrics_spy
    Page._plot_status_display = _status_spy
    Page._plot_error_vs_position_export = lambda self, ax, track: None
    Page._plot_unit_info = lambda self, ax, result, track: None
    return page, seen


def test_export_single_chart_no_error_data_draws_an_ungraded_verdict(tmp_path):
    from laser_trim_analyzer.gui.pages.export import ExportPage
    page, seen = _spy_page(ExportPage)
    result = _wrap(_no_error_data_track(), AnalysisStatus.PASS, "ES-1")

    ExportPage._export_single_chart(page, result, tmp_path / "out.png")

    assert seen["metrics"] == UNGRADED, seen["metrics"]
    assert seen["status"] == UNGRADED, seen["status"]


def test_export_multi_page_pdf_no_error_data_draws_an_ungraded_verdict(tmp_path):
    from laser_trim_analyzer.gui.pages.export import ExportPage
    page, seen = _spy_page(ExportPage)
    result = _wrap(_no_error_data_track(), AnalysisStatus.PASS, "ES-2")

    ExportPage._export_multi_page_pdf(page, [result], tmp_path / "out.pdf")

    assert seen["metrics"] == UNGRADED, seen["metrics"]
    assert seen["status"] == UNGRADED, seen["status"]


def test_analyze_export_comprehensive_chart_no_error_data_draws_an_ungraded_verdict(tmp_path):
    import types
    from laser_trim_analyzer.gui.pages.analyze import AnalyzePage
    page, seen = _spy_page(AnalyzePage)
    result = _wrap(_no_error_data_track(), AnalysisStatus.PASS, "ES-3")
    page.current_result = result
    page.track_selector = types.SimpleNamespace(get=lambda: "")   # no match -> tracks[0]

    AnalyzePage._export_comprehensive_chart(page, str(tmp_path / "out.png"))

    assert seen["metrics"] == UNGRADED, seen["metrics"]
    assert seen["status"] == UNGRADED, seen["status"]
