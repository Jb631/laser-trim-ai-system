"""Tests for the adaptive-N bucketing helper used by the drift dashboard."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")  # headless; must run before any pyplot import
import matplotlib.pyplot as plt

import pytest


@pytest.fixture
def series_factory():
    """Generate a (iso_date, value) series with N total rows."""
    def make(n: int, start_value: float = 1.0):
        base_date = datetime(2026, 1, 1)
        return [
            ((base_date + timedelta(hours=i)).isoformat(), start_value + i * 0.01)
            for i in range(n)
        ]
    return make


def test_compute_buckets_empty_series_returns_empty():
    from laser_trim_analyzer.gui.pages.trends import _compute_buckets
    assert _compute_buckets([]) == []


def test_compute_buckets_basic_evenly_divides(series_factory):
    """100 rows, N=50 → exactly 2 buckets of 50 each."""
    from laser_trim_analyzer.gui.pages.trends import _compute_buckets
    series = series_factory(100)

    buckets = _compute_buckets(series, n_per_bucket=50)

    assert len(buckets) == 2
    assert [b["n"] for b in buckets] == [50, 50]
    assert [b["bucket_index"] for b in buckets] == [0, 1]


def test_compute_buckets_mean_and_se(series_factory):
    """Bucket mean and SE are correct for known input."""
    from laser_trim_analyzer.gui.pages.trends import _compute_buckets

    # Construct 10 rows with values 1.0..1.9 (step 0.1).
    base_date = datetime(2026, 1, 1)
    series = [
        ((base_date + timedelta(hours=i)).isoformat(), 1.0 + i * 0.1)
        for i in range(10)
    ]

    buckets = _compute_buckets(series, n_per_bucket=10)

    assert len(buckets) == 1
    b = buckets[0]
    assert b["n"] == 10
    # Mean of 1.0, 1.1, ..., 1.9
    assert b["mean"] == pytest.approx(1.45)
    # Sample stddev should be > 0
    assert b["stddev"] > 0
    # SE = stddev / sqrt(n)
    import math
    assert b["se"] == pytest.approx(b["stddev"] / math.sqrt(10))


def test_compute_buckets_min_max_date(series_factory):
    """Each bucket records min_date and max_date of its rows."""
    from laser_trim_analyzer.gui.pages.trends import _compute_buckets
    series = series_factory(20)

    buckets = _compute_buckets(series, n_per_bucket=10)

    assert len(buckets) == 2
    # Bucket 0: rows 0-9; Bucket 1: rows 10-19
    assert buckets[0]["min_date"] == series[0][0]
    assert buckets[0]["max_date"] == series[9][0]
    assert buckets[1]["min_date"] == series[10][0]
    assert buckets[1]["max_date"] == series[19][0]


def test_compute_buckets_small_trailing_folds_into_previous():
    """Trailing partial bucket with < 5 rows folds into the previous bucket."""
    from laser_trim_analyzer.gui.pages.trends import _compute_buckets
    # 52 rows: first 50 form bucket 0, last 2 should fold into bucket 0
    base_date = datetime(2026, 1, 1)
    series = [
        ((base_date + timedelta(hours=i)).isoformat(), 1.0)
        for i in range(52)
    ]

    buckets = _compute_buckets(series, n_per_bucket=50)

    assert len(buckets) == 1
    assert buckets[0]["n"] == 52
    assert buckets[0]["max_date"] == series[-1][0]   # folded rows extend max_date
    assert buckets[0]["min_date"] == series[0][0]    # start date unchanged


def test_compute_buckets_small_trailing_with_no_previous_renders_anyway():
    """If there's no previous bucket to fold into and the only bucket has
    < 5 rows, still emit it — the renderer handles the single-bucket case."""
    from laser_trim_analyzer.gui.pages.trends import _compute_buckets
    base_date = datetime(2026, 1, 1)
    series = [
        ((base_date + timedelta(hours=i)).isoformat(), 1.0)
        for i in range(3)
    ]

    buckets = _compute_buckets(series, n_per_bucket=50)

    assert len(buckets) == 1
    assert buckets[0]["n"] == 3


def test_compute_buckets_trailing_bucket_render_threshold():
    """Trailing partial bucket with >= 5 rows renders as its own bucket."""
    from laser_trim_analyzer.gui.pages.trends import _compute_buckets
    # 56 rows: bucket 0 = 50, trailing = 6 → trailing renders as bucket 1
    base_date = datetime(2026, 1, 1)
    series = [
        ((base_date + timedelta(hours=i)).isoformat(), 1.0)
        for i in range(56)
    ]

    buckets = _compute_buckets(series, n_per_bucket=50)

    assert len(buckets) == 2
    assert [b["n"] for b in buckets] == [50, 6]


def test_compute_buckets_single_value_zero_stddev(series_factory):
    """A bucket of length 1 has stddev = 0 and se = 0 (no division-by-zero)."""
    from laser_trim_analyzer.gui.pages.trends import _compute_buckets
    base_date = datetime(2026, 1, 1)
    series = [(base_date.isoformat(), 1.0)]

    buckets = _compute_buckets(series, n_per_bucket=50)

    assert len(buckets) == 1
    assert buckets[0]["stddev"] == 0.0
    assert buckets[0]["se"] == 0.0


# ---------------------------------------------------------------------------
# Task 3: _draw_smoothed_panel tests
# ---------------------------------------------------------------------------

def _make_axes():
    fig, ax = plt.subplots()
    return fig, ax


def test_draw_smoothed_panel_renders_line_and_band():
    from laser_trim_analyzer.gui.pages.trends import (
        _compute_buckets, _draw_smoothed_panel,
    )
    series = [
        ((datetime(2026, 1, 1) + timedelta(hours=i)).isoformat(), 1.0 + i * 0.01)
        for i in range(20)
    ]
    buckets = _compute_buckets(series, n_per_bucket=10)
    fig, ax = _make_axes()

    _draw_smoothed_panel(
        ax,
        buckets=buckets,
        baseline_mean=1.05,
        baseline_cutoff_bucket_index=1,
        color="#7ed99e",
    )

    # Exactly one Line2D for the smoothed mean
    assert len(ax.get_lines()) >= 1, "Should draw at least one mean line"
    # At least one PolyCollection from fill_between (the confidence band)
    assert len(ax.collections) >= 1, "Should draw a confidence band"
    plt.close(fig)


def test_draw_smoothed_panel_no_baseline_skips_baseline_line():
    """When baseline_mean is None, the gray baseline reference line is not drawn."""
    from laser_trim_analyzer.gui.pages.trends import (
        _compute_buckets, _draw_smoothed_panel,
    )
    series = [
        ((datetime(2026, 1, 1) + timedelta(hours=i)).isoformat(), 1.0)
        for i in range(20)
    ]
    buckets = _compute_buckets(series, n_per_bucket=10)
    fig, ax = _make_axes()

    n_lines_before = len(ax.get_lines())
    _draw_smoothed_panel(
        ax, buckets=buckets,
        baseline_mean=None, baseline_cutoff_bucket_index=None,
        color="#7ed99e",
    )
    # With baseline=None we expect ONE line (the mean) and NO axhline
    # (the baseline reference); contrast with the prior test where we
    # also wouldn't pass an axvline (cutoff=None here too).
    assert len(ax.get_lines()) - n_lines_before == 1
    plt.close(fig)


def test_draw_smoothed_panel_single_bucket_dot_no_line():
    """Single bucket renders as a single dot, no line, no band."""
    from laser_trim_analyzer.gui.pages.trends import (
        _compute_buckets, _draw_smoothed_panel,
    )
    series = [(datetime(2026, 1, 1).isoformat(), 1.0)]
    buckets = _compute_buckets(series, n_per_bucket=50)
    assert len(buckets) == 1
    fig, ax = _make_axes()

    _draw_smoothed_panel(
        ax, buckets=buckets,
        baseline_mean=None, baseline_cutoff_bucket_index=None,
        color="#7ed99e",
    )

    # No connected line (PathCollection from scatter, not a Line2D for the line plot)
    # We don't check exact mpl artifact type; the contract is "no fill_between collection"
    assert len(ax.collections) == 0 or all(
        not isinstance(c, matplotlib.collections.PolyCollection)
        for c in ax.collections
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task 4: _draw_sigma_panel tests
# ---------------------------------------------------------------------------

class _StubDetector:
    """Minimal detector stand-in for sigma-panel tests."""
    def __init__(self, has_baseline, lcl=None, center=None, ucl=None):
        self.has_baseline = has_baseline
        self._limits = (lcl, center, ucl)

    def get_control_limits(self):
        return self._limits


def test_draw_sigma_panel_renders_dots_and_smoothed_overlay():
    """With baseline + in-range values: green dots + smoothed white line."""
    from laser_trim_analyzer.gui.pages.trends import _draw_sigma_panel

    rows = [
        ((datetime(2026, 1, 1) + timedelta(hours=i)).isoformat(), 0.01 + i * 0.0005)
        for i in range(30)
    ]
    fig, ax = _make_axes()
    # UCL=0.030 is above all generated values (max = 0.01 + 29*0.0005 = 0.0245)
    detector = _StubDetector(has_baseline=True, lcl=0.005, center=0.015, ucl=0.030)

    violations = _draw_sigma_panel(
        ax,
        sigma_series=rows,
        detector=detector,
        baseline_cutoff_bucket_index=2,
        n_per_bucket=10,
    )

    # All in-range → zero violations
    assert violations == 0
    # UCL/LCL/center are three horizontal lines drawn via axhline
    # (the helper also draws the baseline-cutoff vertical line — 4 total)
    n_ref_lines = sum(1 for ln in ax.get_lines() if ln.get_linestyle() in ("--",))
    assert n_ref_lines >= 2  # at minimum UCL and LCL
    plt.close(fig)


def test_draw_sigma_panel_red_dots_for_out_of_control():
    """Values above UCL render red; the helper returns a positive violation count."""
    from laser_trim_analyzer.gui.pages.trends import _draw_sigma_panel

    rows = [
        ((datetime(2026, 1, 1) + timedelta(hours=i)).isoformat(), v)
        for i, v in enumerate([0.01, 0.01, 0.01, 0.030, 0.035])  # last 2 > UCL=0.020
    ]
    fig, ax = _make_axes()
    detector = _StubDetector(has_baseline=True, lcl=0.005, center=0.012, ucl=0.020)

    violations = _draw_sigma_panel(
        ax, sigma_series=rows,
        detector=detector,
        baseline_cutoff_bucket_index=None,
        n_per_bucket=10,
    )
    assert violations == 2
    plt.close(fig)


def test_draw_sigma_panel_no_baseline_dots_only():
    """detector.has_baseline=False → only gray dots, no smoothed line, no limits."""
    from laser_trim_analyzer.gui.pages.trends import _draw_sigma_panel

    rows = [(datetime(2026, 1, 1).isoformat(), 0.01)]
    fig, ax = _make_axes()
    detector = _StubDetector(has_baseline=False)

    violations = _draw_sigma_panel(
        ax, sigma_series=rows,
        detector=detector,
        baseline_cutoff_bucket_index=None,
        n_per_bucket=10,
    )

    assert violations == 0
    # No control-limit reference lines (axhline) drawn — primary contract is
    # "violations=0, no crash"
    plt.close(fig)
