"""Unit chart / print export must grade linearity the way the ANALYZER does.

Bug (found 2026-08-31 on the print export of 8415-1 SN 26 Track A,
analysis_id=88725): the document showed "Fail Points: 18" next to
"Linearity Pass: YES" and a green PASS stamp — on the ZERO-TOLERANCE
customer disposition.

Root cause: the analyzer's grading of record applies BOTH an offset and a
theory rotation factor k (analyzer._calculate_linearity):

    shifted = error + theory_volts * k + offset

k is persisted as TrackResult.optimal_slope and theory_volts as
theory_data. V5's pages read optimal_slope; V6's load_unit_track never
selected it, so compute_fail_points and export/unit_chart.py re-graded on
`error + offset` alone. Dropping the rotation term invents fail points at
the end of travel, where theory (and so the dropped term) is largest.

These tests pin two separate guarantees:
  1. the renderers apply the SAME formula the analyzer used, and
  2. the fail-point count and the displayed verdict can never contradict
     each other, whatever the arrays say.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest


# A track shaped like the real defect: a constant residual once the rotation
# is applied, so `error + theory*k + offset` is comfortably in spec at every
# point, while `error + offset` alone drifts out at the END of travel (where
# theory is largest) — exactly the signature seen on 88725.
_K = 0.0025
_OFFSET = -0.0058
_THEORY = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
_ERRORS = [-0.008 - _K * t for t in _THEORY]
_UPPER = [0.025] * len(_THEORY)
_LOWER = [-0.025] * len(_THEORY)
# offset-only: -0.0138 - 0.0025*t  -> out of band for t = 5, 6, 7
_OFFSET_ONLY_FAILS = [5, 6, 7]


def _texts(fig):
    """Every string the figure actually renders."""
    out = []
    for ax in fig.axes:
        out.extend(t.get_text() for t in ax.texts)
    return out


def test_compute_fail_points_applies_theory_rotation():
    """With k and theory supplied, no point is out of spec.

    Fails if compute_fail_points grades on `error + offset` alone.
    """
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import compute_fail_points

    # Sanity: the offset-only reading is the WRONG answer we must not produce.
    assert compute_fail_points(_ERRORS, _UPPER, _LOWER,
                               offset=_OFFSET) == _OFFSET_ONLY_FAILS

    assert compute_fail_points(_ERRORS, _UPPER, _LOWER, offset=_OFFSET,
                               k=_K, theory=_THEORY) == []


def test_compute_fail_points_rotation_still_catches_real_violations():
    """Rotation must not become a blanket excuse — a genuinely out-of-spec
    point still fails after the rotation is applied.

    Fails if compute_fail_points ignores the arrays once k is non-zero.
    """
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import compute_fail_points

    errors = list(_ERRORS)
    errors[2] = -0.30  # nothing this offset/k pair can rescue

    assert compute_fail_points(errors, _UPPER, _LOWER, offset=_OFFSET,
                               k=_K, theory=_THEORY) == [2]


def test_compute_fail_points_ignores_rotation_without_theory():
    """k with no theory column must fall back to offset-only, matching the
    analyzer's own `if theory_volts and optimal_k != 0` guard.

    Fails if the renderer applies k against a missing/short theory array.
    """
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import compute_fail_points

    assert compute_fail_points(_ERRORS, _UPPER, _LOWER, offset=_OFFSET,
                               k=_K, theory=None) == _OFFSET_ONLY_FAILS


def test_trim_export_verdict_agrees_with_rendered_fail_points():
    """The print export must not print a fail count beside a passing verdict.

    This is the exact contradiction seen on 88725. Fails while the TRIM
    branch pairs a renderer-computed len(fail_points) with the STORED
    linearity_pass.
    """
    from laser_trim_analyzer.export.unit_chart import build_unit_export_figure

    data = {
        "position_data": list(range(len(_THEORY))),
        "error_data": _ERRORS,
        "upper_limits": _UPPER, "lower_limits": _LOWER,
        "optimal_offset": _OFFSET,
        # The analyzer graded this PASS (it applied the rotation).
        "linearity_pass": True, "linearity_error": 0.0138,
        "sigma_pass": True,
    }
    # Renderer hands in the WRONG (offset-only) fail points, as it does today.
    fig = build_unit_export_figure({"model": "M", "serial": "1", "n_tracks": 1}, data,
                                   fail_points=_OFFSET_ONLY_FAILS, kind="trim")
    txt = _texts(fig)

    fails = [t for t in txt if t.startswith("Fail Points:")]
    verdict = [t for t in txt if t.startswith("Linearity Pass:")]
    assert fails and verdict
    if fails[0] != "Fail Points: 0":
        assert verdict[0] != "Linearity Pass: YES", (
            f"contradiction rendered: {fails[0]!r} beside {verdict[0]!r}")
    assert "PASS" not in _stamp(fig) or fails[0] == "Fail Points: 0", (
        f"green PASS stamp printed beside {fails[0]!r}")


def _stamp(fig):
    """The big status word in the 4th panel."""
    for t in fig.axes[-1].texts:
        s = t.get_text()
        if s in ("PASS", "FAIL", "PASS (WATCH)", "PASS*", "NOT EVALUATED"):
            return s
    return ""


def test_trim_export_corrected_trace_uses_rotation():
    """The green 'corrected' line is the graded trace, so it must carry the
    rotation term too — otherwise the drawn line contradicts the printed
    Linearity Error.

    Fails while build_unit_export_figure computes `errors + offset`.
    """
    from laser_trim_analyzer.export.unit_chart import build_unit_export_figure

    data = {
        "position_data": list(range(len(_THEORY))),
        "error_data": _ERRORS,
        "upper_limits": _UPPER, "lower_limits": _LOWER,
        "optimal_offset": _OFFSET, "optimal_slope": _K, "theory_data": _THEORY,
        "linearity_pass": True, "sigma_pass": True,
    }
    fig = build_unit_export_figure({"model": "M", "serial": "1", "n_tracks": 1}, data,
                                   fail_points=[], kind="trim")
    ax = fig.axes[0]
    corrected = None
    for line in ax.get_lines():
        if (line.get_label() or "").startswith("Trimmed corrected"):
            corrected = list(line.get_ydata())
    assert corrected is not None, "corrected trace not drawn"
    # Rotation applied => flat residual, every point inside the band.
    assert max(abs(v) for v in corrected) < 0.025, corrected


def test_load_unit_track_carries_rotation_terms(tmp_path):
    """load_unit_track must surface optimal_slope and theory_data.

    Without them the renderers CANNOT reproduce the analyzer's verdict.
    Fails while load_unit_track selects neither column.
    """
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType, SystemType)
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import load_unit_track

    db = DatabaseManager(tmp_path / "rot.db")
    when = datetime(2026, 7, 14)
    with db.session() as s:
        ar = DBAR(filename="rot.xls", file_path="/t/rot",
                  file_hash="rot".ljust(64, "0"), model="8415-1", serial="26",
                  system=SystemType.B, file_date=when, timestamp=when,
                  overall_status=StatusType.PASS, has_multi_tracks=False,
                  processing_time=0.1)
        s.add(ar)
        s.flush()
        s.add(DBTR(analysis_id=ar.id, track_id="Track A", status=StatusType.PASS,
                   position_data=list(range(len(_THEORY))), error_data=_ERRORS,
                   upper_limits=_UPPER, lower_limits=_LOWER, theory_data=_THEORY,
                   optimal_offset=_OFFSET, optimal_slope=_K,
                   linearity_pass=True, linearity_fail_points=0,
                   linearity_spec=0.025))
        s.commit()
        aid = ar.id

    data = load_unit_track(db, aid)
    assert data is not None
    assert data.get("optimal_slope") == pytest.approx(_K)
    assert data.get("theory_data") == pytest.approx(_THEORY)


def test_loaded_track_reproduces_stored_analyzer_verdict(tmp_path):
    """End-to-end: what load_unit_track returns, fed to compute_fail_points,
    must reproduce the analyzer's STORED fail-point count.

    This is the invariant the whole fix exists to hold. Fails today because
    the rotation term never survives the load.
    """
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType, SystemType)
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import (
        compute_fail_points, load_unit_track)

    db = DatabaseManager(tmp_path / "rot2.db")
    when = datetime(2026, 7, 14)
    with db.session() as s:
        ar = DBAR(filename="rot2.xls", file_path="/t/rot2",
                  file_hash="rot2".ljust(64, "0"), model="8415-1", serial="26",
                  system=SystemType.B, file_date=when, timestamp=when,
                  overall_status=StatusType.PASS, has_multi_tracks=False,
                  processing_time=0.1)
        s.add(ar)
        s.flush()
        s.add(DBTR(analysis_id=ar.id, track_id="Track A", status=StatusType.PASS,
                   position_data=list(range(len(_THEORY))), error_data=_ERRORS,
                   upper_limits=_UPPER, lower_limits=_LOWER, theory_data=_THEORY,
                   optimal_offset=_OFFSET, optimal_slope=_K,
                   linearity_pass=True, linearity_fail_points=0,
                   linearity_spec=0.025))
        s.commit()
        aid = ar.id

    data = load_unit_track(db, aid)
    fp = compute_fail_points(
        data["error_data"], data["upper_limits"], data["lower_limits"],
        offset=data.get("optimal_offset") or 0.0,
        k=data.get("optimal_slope") or 0.0,
        theory=data.get("theory_data"))
    assert len(fp) == 0, (
        f"renderer says {len(fp)} fail points; analyzer stored 0")
