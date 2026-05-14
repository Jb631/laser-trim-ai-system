"""Tests for the adaptive-N bucketing helper used by the drift dashboard."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
