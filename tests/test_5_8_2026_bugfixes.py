"""Verification tests for the four bug fixes from the 2026-05-08 work-day report.

Each test maps to one issue from "Work Files/5-8-2026/app errors.txt":
  #1 export crash from defer() misuse  → light_load query path
  #2 FT compare track A/B mislink       → _pair_tracks helper
  #4 excluded points appear "not applied" on later files
                                         → analyze page passes ALL excluded
                                           indices (not just failing ones)
  #5 chart axis stretches to include excluded outlier
                                         → chart constrains ylim to the
                                           non-excluded data range
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# #1: defer() chaining no longer crashes the light_load export path.
# ---------------------------------------------------------------------------
def test_light_load_query_does_not_crash():
    """Build the same loader option chain the fixed code uses and confirm
    SQLAlchemy accepts it. Reproduces the exact API call that crashed in
    production with `_AbstractLoad.defer() takes from 2 to 3 positional
    arguments but 9 were given`.
    """
    from sqlalchemy.orm import joinedload
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAnalysisResult,
        TrackResult as DBTrackResult,
    )

    tracks_loader = joinedload(DBAnalysisResult.tracks)
    # The pre-fix code passed all 8 columns positionally to a single
    # defer() call, which raises TypeError. The fix chains them.
    for col in (
        DBTrackResult.position_data,
        DBTrackResult.error_data,
        DBTrackResult.upper_limits,
        DBTrackResult.lower_limits,
        DBTrackResult.theory_data,
        DBTrackResult.test_volts,
        DBTrackResult.untrimmed_positions,
        DBTrackResult.untrimmed_errors,
    ):
        tracks_loader = tracks_loader.defer(col)

    # If we got here without TypeError, the chained API call works.
    assert tracks_loader is not None


def test_get_historical_data_light_load_returns_rows():
    """End-to-end: hit the actual fixed function against the local DB and
    confirm it returns rows without crashing.
    """
    db_path = Path(__file__).resolve().parents[1] / "data" / "analysis.db"
    if not db_path.exists():
        # If the local DB doesn't exist, skip rather than fail — the
        # production crash repro lives at the API level above.
        import pytest
        pytest.skip("local analysis.db not present")

    from laser_trim_analyzer.database.manager import DatabaseManager

    # DatabaseManager takes a Path, not a URL.
    db = DatabaseManager(db_path)
    rows = db.get_historical_data(
        model="7965", days_back=36500, limit=10, light_load=True
    )
    assert isinstance(rows, list)
    # 7965 has 1101 rows in the local DB; expect at least one back.
    assert len(rows) > 0


# ---------------------------------------------------------------------------
# #2: track-pairing helper bridges trim and FT track-id naming conventions.
# ---------------------------------------------------------------------------
def test_pair_tracks_pairs_track_b_trim_with_ft_track_b():
    """The 6607 case from the user report: trim file is single-track 'Track B'
    (System B newer naming), FT file has both 'A' and 'B' sheets. Pre-fix
    code took ft_tracks[0] which is 'A' — pairing trim-B against FT-A.
    """
    from laser_trim_analyzer.gui.pages.compare import _pair_tracks

    trim_tracks = [{"track_id": "Track B", "errors": [1, 2, 3]}]
    ft_tracks = [
        {"track_id": "A", "errors": [9, 9, 9]},
        {"track_id": "B", "errors": [1, 2, 3]},
    ]
    trim, ft = _pair_tracks(trim_tracks, ft_tracks)
    assert trim["track_id"] == "Track B"
    assert ft["track_id"] == "B"


def test_pair_tracks_maps_trk1_to_a():
    from laser_trim_analyzer.gui.pages.compare import _pair_tracks

    trim_tracks = [{"track_id": "TRK1"}]
    ft_tracks = [{"track_id": "B"}, {"track_id": "A"}]
    _, ft = _pair_tracks(trim_tracks, ft_tracks)
    assert ft["track_id"] == "A"


def test_pair_tracks_legacy_default_falls_back_to_first_ft():
    from laser_trim_analyzer.gui.pages.compare import _pair_tracks

    trim_tracks = [{"track_id": "default"}]
    ft_tracks = [{"track_id": "default"}]
    trim, ft = _pair_tracks(trim_tracks, ft_tracks)
    assert trim["track_id"] == "default"
    assert ft["track_id"] == "default"


def test_pair_tracks_handles_empty_sides():
    from laser_trim_analyzer.gui.pages.compare import _pair_tracks

    assert _pair_tracks([], []) == (None, None)
    trim_only = [{"track_id": "Track A"}]
    assert _pair_tracks(trim_only, []) == (trim_only[0], None)
    ft_only = [{"track_id": "A"}]
    assert _pair_tracks([], ft_only) == (None, ft_only[0])


# ---------------------------------------------------------------------------
# #4: analyze page passes ALL excluded indices to chart, not just failing ones.
# This is verified by reading the modified file — the test guards against
# regression to the old `excluded_fail_indices` filter.
# ---------------------------------------------------------------------------
def test_analyze_passes_all_excluded_indices_not_just_failing():
    src = Path(__file__).resolve().parents[1] / "src" / "laser_trim_analyzer" / "gui" / "pages" / "analyze.py"
    text = src.read_text()
    # The bug was: `excluded_points=excluded_fail_indices` — filtering the
    # excluded markers down to the failing-and-excluded subset.
    assert "excluded_points=excluded_fail_indices" not in text, (
        "regression: chart only receives the failing subset of excluded "
        "indices — passing-and-excluded points won't show their marker"
    )
    # Fix: pass the full sorted set.
    assert "excluded_points=sorted(excluded_indices)" in text


# ---------------------------------------------------------------------------
# #5: chart constrains y-axis to the non-excluded data range.
# ---------------------------------------------------------------------------
def test_chart_ylim_excludes_outlier_at_zero_volts():
    """Reproduces the 8492 case: the last data point drops to 0V and is
    excluded. The chart should NOT scale to include it — pre-fix it did,
    compressing the real data into an unreadable thin band.
    """
    from laser_trim_analyzer.gui.widgets.chart import ChartWidget
    import customtkinter as ctk

    root = ctk.CTk()
    try:
        chart = ChartWidget(root)
        n = 50
        positions = [i * 0.1 for i in range(n)]
        # Real data oscillates in ±0.001 — tiny.
        trimmed_errors = [0.001 if i % 2 else -0.001 for i in range(n)]
        # Last point is the outlier at -10V (a "drops to 0 volts" failure
        # that's been excluded by spec).
        trimmed_errors[-1] = -10.0

        upper = [0.005] * n
        lower = [-0.005] * n

        chart.plot_error_vs_position(
            positions=positions,
            trimmed_errors=trimmed_errors,
            upper_limits=upper,
            lower_limits=lower,
            offset=0.0,
            k=0.0,
            excluded_points=[n - 1],
        )

        ax = chart.figure.axes[0]
        y_lo, y_hi = ax.get_ylim()
        # The outlier sits at -10V. If the axis extends below -1V the
        # autoscale included the excluded point — bug not fixed.
        assert y_lo > -1.0, (
            f"y-axis lower bound {y_lo:.4f} reaches the excluded outlier "
            f"at -10V — chart is still autoscaling over excluded points"
        )
        # Real data + spec is ±0.005; allow generous headroom but not
        # multiple orders of magnitude.
        assert y_hi - y_lo < 0.1, (
            f"y-range {y_hi - y_lo:.4f} is too wide for ±0.005 spec data"
        )
    finally:
        root.destroy()
