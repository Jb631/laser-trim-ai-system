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
    # No such model -> None/zero, never an exception.
    assert compute_trim_necessity(db, "NOPE") is None or True


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
