"""Trim necessity + FT sweep viewer (James, 2026-07-14).

* compute_trim_necessity — "was the laser even needed for linearity?":
  unit counts as already-passing only when EVERY track's raw pre-trim worst
  point is inside spec; only actually-trimmed units enter the denominator.
* load_ft_track — clicking a final-test unit must produce its sweep arrays.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _unit(s, model, serial, when, i, DBAR, SystemType, StatusType,
          status=None):
    ar = DBAR(filename=f"{model}-{serial}-{i}.xls", file_path=f"/t/{i}",
              file_hash=f"tn{model}{serial}{i}".ljust(64, "0"), model=model,
              serial=serial, system=SystemType.A, file_date=when,
              timestamp=when, overall_status=status or StatusType.PASS,
              has_multi_tracks=False, processing_time=0.1)
    s.add(ar)
    s.flush()
    return ar


def test_trim_necessity_counts_prepass_units(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType, SystemType)
    from laser_trim_analyzer.core.yield_stats import compute_trim_necessity

    db = DatabaseManager(tmp_path / "tn.db")
    when = datetime(2025, 6, 2)
    with db.session() as s:
        # Unit A: single track, pre-trim already inside spec -> prepass.
        a = _unit(s, "TN", "1", when, 0, DBAR, SystemType, StatusType)
        s.add(DBTR(analysis_id=a.id, track_id="T1", status=StatusType.PASS,
                   untrimmed_error_max=0.01, linearity_spec=0.05,
                   trim_pass_count=1, resistance_change_percent=12.0))
        # Unit B: pre-trim outside spec -> needed the trim.
        b = _unit(s, "TN", "2", when, 1, DBAR, SystemType, StatusType)
        s.add(DBTR(analysis_id=b.id, track_id="T1", status=StatusType.PASS,
                   untrimmed_error_max=0.08, linearity_spec=0.05,
                   trim_pass_count=1, resistance_change_percent=15.0))
        # Unit C: TWO tracks, one inside one outside -> NOT prepass
        # (zero-tolerance: every track must already pass).
        c = _unit(s, "TN", "3", when, 2, DBAR, SystemType, StatusType)
        s.add(DBTR(analysis_id=c.id, track_id="TA", status=StatusType.PASS,
                   untrimmed_error_max=0.01, linearity_spec=0.05,
                   trim_pass_count=1))
        s.add(DBTR(analysis_id=c.id, track_id="TB", status=StatusType.PASS,
                   untrimmed_error_max=0.09, linearity_spec=0.05,
                   trim_pass_count=1))
        # Unit D: never trimmed (test-sweep only) -> excluded entirely.
        d = _unit(s, "TN", "4", when, 3, DBAR, SystemType, StatusType)
        s.add(DBTR(analysis_id=d.id, track_id="T1", status=StatusType.PASS,
                   untrimmed_error_max=0.01, linearity_spec=0.05,
                   trim_pass_count=0))
        s.commit()

    tn = compute_trim_necessity(db, "TN")
    assert tn["trimmed_units"] == 3
    assert tn["prepass_units"] == 1
    assert abs(tn["prepass_share"] - 100.0 / 3) < 0.01
    assert abs(tn["avg_resistance_change_prepass"] - 12.0) < 0.01


def test_trim_necessity_respects_cutoff_and_empty(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType, SystemType)
    from laser_trim_analyzer.core.yield_stats import compute_trim_necessity

    db = DatabaseManager(tmp_path / "tnc.db")
    with db.session() as s:
        old = _unit(s, "TNC", "1", datetime(2023, 1, 5), 0, DBAR, SystemType, StatusType)
        s.add(DBTR(analysis_id=old.id, track_id="T1", status=StatusType.PASS,
                   untrimmed_error_max=0.01, linearity_spec=0.05, trim_pass_count=1))
        s.commit()
    # Cutoff after the only unit -> nothing in window.
    tn = compute_trim_necessity(db, "TNC", cutoff=datetime(2024, 1, 1))
    assert tn is None or tn["trimmed_units"] == 0
    # No such model -> None (never a bogus dict, never an exception).
    assert compute_trim_necessity(db, "NOPE") is None


def test_offset_feasibility_opposing_points():
    """The 7845 case (James, 2026-07-14): one visible fail point that 'looks
    adjustable', but an opposing point already near the OTHER limit makes any
    clearing offset impossible. The note must name both binding points."""
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import (
        _offset_verdict_note, compute_fail_points, compute_offset_feasibility)

    #                 pt0    pt1(fail low)  pt2(near upper)
    errors = [0.000, -0.070,        0.040]
    upper  = [0.050,  0.050,        0.050]
    lower  = [-0.050, -0.050,      -0.050]
    # pt1 needs o >= 0.02; pt2 allows o <= 0.01 -> infeasible.
    lo_b, hi_b, i_lo, i_hi = compute_offset_feasibility(errors, upper, lower)
    assert round(lo_b, 6) == 0.02 and round(hi_b, 6) == 0.01
    assert (i_lo, i_hi) == (1, 2)
    fp = compute_fail_points(errors, upper, lower, offset=0.0)
    note, binding = _offset_verdict_note(fp, errors, upper, lower)
    assert "can't fix" in note and "#1" in note and "#2" in note
    assert binding == [1, 2]


def test_offset_feasibility_fixable_flags_inconsistency():
    """If a clearing offset EXISTS while fail points are recorded, that is a
    data/verdict inconsistency and must be said loudly, never hidden."""
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import (
        _offset_verdict_note, compute_fail_points)

    errors = [0.000, -0.070, 0.010]        # o = +0.03 clears everything
    upper  = [0.050,  0.050, 0.050]
    lower  = [-0.050, -0.050, -0.050]
    fp = compute_fail_points(errors, upper, lower, offset=0.0)
    assert fp == [1]
    note, binding = _offset_verdict_note(fp, errors, upper, lower)
    assert "inconsistency" in note.lower()


def test_offset_note_absent_on_passing_unit():
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import _offset_verdict_note
    note, binding = _offset_verdict_note([], [0.0], [0.05], [-0.05])
    assert note is None and binding is None


def test_offset_zero_width_window_is_boundary_riding_not_inconsistency():
    """The 7965 SN 367 case: the ONLY clearing offset parks two points
    exactly ON their limits (window width 0). That is a legitimate FAIL and
    must read as boundary-riding — never as 'data inconsistency'."""
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import (
        _offset_verdict_note, compute_fail_points)

    # pt1 needs o >= 0.02 exactly; pt2 allows o <= 0.02 exactly.
    errors = [0.000, -0.070, 0.030]
    upper  = [0.050,  0.050, 0.050]
    lower  = [-0.050, -0.050, -0.050]
    fp = compute_fail_points(errors, upper, lower, offset=0.0)
    assert fp == [1]
    note, binding = _offset_verdict_note(fp, errors, upper, lower)
    assert "boundary" in note.lower() or "zero margin" in note.lower()
    assert "inconsistency" not in note.lower()
    assert binding == [1, 2]


def test_load_ft_track_returns_sweep(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        FinalTestResult as DBFT, FinalTestTrack as DBFTT, StatusType)
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import (
        compute_fail_points, load_ft_track)

    db = DatabaseManager(tmp_path / "ftc.db")
    when = datetime(2025, 6, 2)
    with db.session() as s:
        ft = DBFT(filename="FT-1.xls", model="8340", serial="7",
                  test_date=when, file_date=when, timestamp=when,
                  overall_status=StatusType.FAIL)
        s.add(ft)
        s.flush()
        s.add(DBFTT(final_test_id=ft.id, track_id="T1", status=StatusType.FAIL,
                    linearity_spec=0.05, linearity_error=0.07,
                    position_data=[0.0, 1.0, 2.0],
                    error_data=[0.0, 0.07, -0.01],
                    upper_limits=[0.05, 0.05, 0.05],
                    lower_limits=[-0.05, -0.05, -0.05],
                    optimal_offset=0.0))
        s.commit()
        ft_id = ft.id

    data = load_ft_track(db, ft_id)
    assert data is not None
    assert data["serial"] == "7" and data["result"] == "FAIL"
    assert len(data["position_data"]) == 3
    fp = compute_fail_points(data["error_data"], data["upper_limits"],
                             data["lower_limits"],
                             offset=data.get("optimal_offset") or 0.0)
    assert fp == [1]                     # the 0.07 point breaches +0.05
    assert load_ft_track(db, 999999) is None


def test_ft_reconciled_verdict_keeps_station_fail_visible(tmp_path):
    """Production FT path (James 2026-07-14, 'best-fit but reconciled'): real FT
    records store NO offset, so the chart applies a best-fit. When that best-fit
    clears a sweep the station recorded as FAIL, the verdict must NOT read as a
    silent clean pass — 0 fail points, but a reconciliation note that keeps the
    station FAIL visible and says to reprocess."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        FinalTestResult as DBFT, FinalTestTrack as DBFTT, StatusType)
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import (
        ft_reconciled_verdict, load_ft_track)

    db = DatabaseManager(tmp_path / "ftr.db")
    when = datetime(2025, 6, 2)
    with db.session() as s:
        ft = DBFT(filename="FTR-1.xls", model="8340", serial="9",
                  test_date=when, file_date=when, timestamp=when,
                  overall_status=StatusType.FAIL)
        s.add(ft)
        s.flush()
        # As measured, the 0.07 point breaches +0.05 -> station FAIL. A DC offset
        # of ~-0.03 centers every point in spec. optimal_offset is NULL (as real
        # FT records are), so load_ft_track fabricates the best-fit.
        s.add(DBFTT(final_test_id=ft.id, track_id="T1", status=StatusType.FAIL,
                    linearity_spec=0.05, linearity_error=0.07,
                    position_data=[0.0, 1.0, 2.0],
                    error_data=[0.0, 0.07, -0.01],
                    upper_limits=[0.05, 0.05, 0.05],
                    lower_limits=[-0.05, -0.05, -0.05],
                    optimal_offset=None))
        s.commit()
        ft_id = ft.id

    data = load_ft_track(db, ft_id)
    assert data["offset_source"] == "best_fit"        # NULL stored -> fabricated
    assert abs(data["optimal_offset"] - (-0.03)) < 1e-9
    fp, recomputed_pass, note, binding = ft_reconciled_verdict(data)
    assert fp == [] and recomputed_pass is True        # best-fit clears every point
    assert note is not None and "reprocess" in note.lower()  # station FAIL stays visible
    assert binding is None

    # A genuinely stored 0.0 offset must be RESPECTED, not overwritten by best-fit
    # (the falsy-0.0 bug): grading then happens at offset 0 and the breach shows.
    with db.session() as s:
        ft2 = DBFT(filename="FTR-2.xls", model="8340", serial="10",
                   test_date=when, file_date=when, timestamp=when,
                   overall_status=StatusType.FAIL)
        s.add(ft2)
        s.flush()
        s.add(DBFTT(final_test_id=ft2.id, track_id="T1", status=StatusType.FAIL,
                    linearity_spec=0.05, position_data=[0.0, 1.0, 2.0],
                    error_data=[0.0, 0.07, -0.01],
                    upper_limits=[0.05, 0.05, 0.05],
                    lower_limits=[-0.05, -0.05, -0.05],
                    optimal_offset=0.0))
        s.commit()
        ft2_id = ft2.id
    d2 = load_ft_track(db, ft2_id)
    assert d2["offset_source"] == "stored" and d2["optimal_offset"] == 0.0
    fp2, pass2, _n2, _b2 = ft_reconciled_verdict(d2)
    assert fp2 == [1] and pass2 is False
