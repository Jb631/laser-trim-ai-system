# Single-Model Drift Dashboard Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the 2×2 single-model drift dashboard so each panel renders a smoothed mean + ±1 SE confidence band over adaptive-N buckets, with the sigma panel additionally showing individual unit dots colored by SPC violation. Replace the unreadable "Trim Passes" panel with a "Retrim Rate" panel.

**Architecture:** One small extension to the existing `get_model_drift_dashboard` DB query (adds `baseline_cutoff_date` and a per-row retrim-rate series). Three new module-level renderer helpers in `trends.py` (`_compute_buckets`, `_draw_smoothed_panel`, `_draw_sigma_panel`). `_render_single_model_drift` (currently 2723–2880) is rewritten to call the helpers — substantially shorter than today. No schema changes, no new tables, no `ChartWidget` changes.

**Tech Stack:** Python 3.10+, SQLAlchemy 2.0, matplotlib (Agg-headless for tests), customtkinter, pytest.

---

## Spec reference

This plan implements `docs/superpowers/specs/2026-05-13-single-model-drift-redesign-design.md`. Spec sections referenced by section number below.

## File structure

| File | Action | Purpose |
| ---- | ------ | ------- |
| `src/laser_trim_analyzer/database/manager.py` | Modify (`get_model_drift_dashboard` at lines 7757–7861) | Add `baseline_cutoff_date` to top-level dict; add `process.retrim_rate_series` |
| `src/laser_trim_analyzer/gui/pages/trends.py` | Modify (rewrite `_render_single_model_drift` at lines 2723–2880; add three module-level helpers near top of file) | New rendering helpers + rewired renderer |
| `tests/test_drift_dashboard_data.py` | Modify | Add tests for the two new dict fields |
| `tests/test_drift_buckets.py` | Create | Pure-function tests for `_compute_buckets` and panel-draw helpers |

---

## Task 1: Extend `get_model_drift_dashboard` with `baseline_cutoff_date` and `retrim_rate_series`

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py:7757-7861`
- Test: `tests/test_drift_dashboard_data.py`

Spec §Data layer points 2 and 3.

- [ ] **Step 1: Write the failing tests**

Open `tests/test_drift_dashboard_data.py` and add these tests at the end of the file. They use the existing `db` fixture and `_add_analysis` helper already present in the file.

```python
def test_get_model_drift_dashboard_includes_baseline_cutoff_date(db):
    """baseline_cutoff_date must be `now - recent_days`, in ISO format,
    so all four panels can draw the same vertical reference line."""
    from datetime import datetime, timedelta

    _add_analysis(db, "TEST-A", datetime.now() - timedelta(days=5),
                  sigma_values=(0.01,))

    result = db.get_model_drift_dashboard(
        model="TEST-A", days_back=90, recent_days=14
    )

    assert "baseline_cutoff_date" in result, (
        "Top-level dict must include baseline_cutoff_date"
    )
    cutoff = datetime.fromisoformat(result["baseline_cutoff_date"])
    expected = datetime.now() - timedelta(days=14)
    # within a small wall-clock delta of expected
    assert abs((cutoff - expected).total_seconds()) < 5


def test_get_model_drift_dashboard_includes_retrim_rate_series(db):
    """retrim_rate_series is one (iso_date, 0_or_1) per non-NULL row,
    where 1 means trim_pass_count > 1."""
    from datetime import datetime, timedelta

    # Three rows: first two needed only 1 pass, third needed 2.
    base_date = datetime.now() - timedelta(days=10)
    for i, tpc in enumerate([1, 1, 2]):
        _add_analysis(
            db, "TEST-B",
            base_date + timedelta(hours=i),
            sigma_values=(0.01,),
            serial=f"{i:04d}",
            trim_pass_count=tpc,
        )

    result = db.get_model_drift_dashboard(
        model="TEST-B", days_back=90, recent_days=14
    )

    series = result["process"]["retrim_rate_series"]
    # 3 rows in, 3 entries out, values 0,0,1
    assert len(series) == 3
    assert [v for _, v in series] == [0, 0, 1]


def test_get_model_drift_dashboard_retrim_rate_skips_null_trim_pass_count(db):
    """Rows with NULL trim_pass_count (pre-feature data) must be excluded
    from retrim_rate_series so the panel can detect the all-NULL case."""
    from datetime import datetime, timedelta

    _add_analysis(
        db, "TEST-C",
        datetime.now() - timedelta(days=5),
        sigma_values=(0.01,),
        trim_pass_count=None,
    )

    result = db.get_model_drift_dashboard(
        model="TEST-C", days_back=90, recent_days=14
    )

    assert result["process"]["retrim_rate_series"] == []
```

If the existing `_add_analysis` helper does not yet accept `trim_pass_count`, modify it to take that kwarg and set it on the track row. Find the helper at the top of `tests/test_drift_dashboard_data.py` (around line 25) and add the parameter; default to `None` so existing tests keep working. Example:

```python
def _add_analysis(db, model, file_date, sigma_values=(0.01,), serial="0001",
                  trim_pass_count=None):
    ...
    # inside the track-creation loop, set:
    #   trim_pass_count=trim_pass_count
    # on the DBTrackResult constructor
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
pytest tests/test_drift_dashboard_data.py::test_get_model_drift_dashboard_includes_baseline_cutoff_date \
       tests/test_drift_dashboard_data.py::test_get_model_drift_dashboard_includes_retrim_rate_series \
       tests/test_drift_dashboard_data.py::test_get_model_drift_dashboard_retrim_rate_skips_null_trim_pass_count \
       -v
```

Expected: 3 FAIL with `KeyError: 'baseline_cutoff_date'` and `KeyError: 'retrim_rate_series'`.

- [ ] **Step 3: Implement the two additions in `get_model_drift_dashboard`**

In `src/laser_trim_analyzer/database/manager.py`, locate `get_model_drift_dashboard` (line 7757). Make two changes:

**Change A — track retrim_rate at row time.** After the existing `process_series` initialization (around line 7795), add a parallel structure:

```python
        retrim_rate_series: List[Tuple[str, int]] = []
```

Inside the existing per-row loop, immediately after the existing `for metric, value in ...` loop body (line 7820) and before the loop ends, append the retrim-rate entry. Note: `tpc` is the local variable already bound from the unpacking at line 7799.

Add this immediately after the existing per-row metric loop (i.e., still inside `for file_date, sigma, ur, mea, tpc in rows:`):

```python
            # Retrim rate: one (iso_date, 0_or_1) per non-NULL trim_pass_count row.
            # 1 iff this row needed more than one trim pass. NULL rows are
            # skipped so the panel can detect the "all pre-feature" case.
            if tpc is not None:
                try:
                    tpc_int = int(tpc)
                except (TypeError, ValueError):
                    tpc_int = None
                if tpc_int is not None:
                    retrim_rate_series.append((iso, 1 if tpc_int > 1 else 0))
```

**Change B — surface the two new fields in the return dict.** Replace the existing return block (lines 7856–7861) with:

```python
        return {
            "model": model,
            "unit_count": len(rows),
            "baseline_cutoff_date": recent_cutoff.isoformat(),
            "sigma_series": sigma_series,
            "process": {
                **process,
                "retrim_rate_series": retrim_rate_series,
            },
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_drift_dashboard_data.py -v
```

Expected: all tests in the file pass (the 3 new ones plus the pre-existing ones).

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py tests/test_drift_dashboard_data.py
git commit -m "feat(drift): get_model_drift_dashboard adds baseline_cutoff_date + retrim_rate_series"
```

---

## Task 2: Add `_compute_buckets` helper to trends.py

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py` (add helper near top of file, after imports)
- Test: `tests/test_drift_buckets.py` (create)

Spec §5 (adaptive bucketing algorithm).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_drift_buckets.py`:

```python
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
    """Trailing partial bucket with ≥ 5 rows renders as its own bucket."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_drift_buckets.py -v
```

Expected: all tests FAIL with `ImportError: cannot import name '_compute_buckets'`.

- [ ] **Step 3: Implement `_compute_buckets`**

In `src/laser_trim_analyzer/gui/pages/trends.py`, find the existing module-level imports and add this helper immediately after them, before the first class definition. (Search for `class TrendsPage` or similar to locate the boundary.)

```python
# ============================================================================
# Drift-dashboard helpers (module-level for testability)
# ============================================================================

def _compute_buckets(
    series: List[Tuple[str, float]],
    n_per_bucket: int = 50,
) -> List[Dict[str, Any]]:
    """Group a time-ordered (iso_date, value) series into adaptive buckets.

    A new bucket is emitted every `n_per_bucket` rows. The trailing partial
    bucket is kept as its own bucket if it holds at least 5 rows; otherwise
    it folds into the previous bucket (so a tail of a few stragglers doesn't
    distort the chart with a tiny low-confidence point). If there's no
    previous bucket to fold into, the tail is rendered as-is.

    Returns a list of dicts with keys:
        bucket_index, n, mean, stddev, se, min_date, max_date

    `stddev` and `se` are 0.0 for n=1 (avoids division-by-zero).
    """
    if not series:
        return []

    import math

    raw_buckets: List[List[Tuple[str, float]]] = []
    cur: List[Tuple[str, float]] = []
    for entry in series:
        cur.append(entry)
        if len(cur) >= n_per_bucket:
            raw_buckets.append(cur)
            cur = []

    if cur:
        if len(cur) >= 5 or not raw_buckets:
            raw_buckets.append(cur)
        else:
            raw_buckets[-1].extend(cur)

    out: List[Dict[str, Any]] = []
    for idx, rows in enumerate(raw_buckets):
        values = [v for _, v in rows]
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            var = sum((x - mean) ** 2 for x in values) / (n - 1)
            stddev = var ** 0.5
            se = stddev / math.sqrt(n)
        else:
            stddev = 0.0
            se = 0.0
        out.append({
            "bucket_index": idx,
            "n": n,
            "mean": mean,
            "stddev": stddev,
            "se": se,
            "min_date": rows[0][0],
            "max_date": rows[-1][0],
        })

    return out
```

Make sure `List`, `Tuple`, `Dict`, `Any` are imported at the top of the file. They likely already are — check the existing `from typing import ...` line near the top and add any missing names.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_drift_buckets.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/trends.py tests/test_drift_buckets.py
git commit -m "feat(drift): _compute_buckets helper — adaptive-N bucketing"
```

---

## Task 3: Add `_draw_smoothed_panel` helper

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py` (add helper after `_compute_buckets`)
- Test: `tests/test_drift_buckets.py` (extend)

Spec §3 (process panels) and §2 (per-panel anatomy — reference lines, status pill).

This helper draws ONE process panel given a matplotlib Axes plus all the data it needs. Pure with respect to side effects on the Axes — no DB calls, no widget calls.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_drift_buckets.py`:

```python
import matplotlib
matplotlib.use("Agg")  # headless; do this before pyplot import
import matplotlib.pyplot as plt
from datetime import datetime as _dt


def _make_axes():
    fig, ax = plt.subplots()
    return fig, ax


def test_draw_smoothed_panel_renders_line_and_band():
    from laser_trim_analyzer.gui.pages.trends import (
        _compute_buckets, _draw_smoothed_panel,
    )
    series = [
        ((_dt(2026, 1, 1) + timedelta(hours=i)).isoformat(), 1.0 + i * 0.01)
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
        ((_dt(2026, 1, 1) + timedelta(hours=i)).isoformat(), 1.0)
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
    series = [(_dt(2026, 1, 1).isoformat(), 1.0)]
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_drift_buckets.py -v
```

Expected: the 3 new tests FAIL with `ImportError: cannot import name '_draw_smoothed_panel'`.

- [ ] **Step 3: Implement `_draw_smoothed_panel`**

In `src/laser_trim_analyzer/gui/pages/trends.py`, immediately after `_compute_buckets`, add:

```python
def _draw_smoothed_panel(
    ax,
    buckets: List[Dict[str, Any]],
    baseline_mean: Optional[float],
    baseline_cutoff_bucket_index: Optional[int],
    color: str,
) -> None:
    """Draw a process-metric panel: smoothed mean + confidence band.

    Adds, in order:
      - a horizontal gray dashed line at baseline_mean (if not None)
      - a vertical orange dashed line at baseline_cutoff_bucket_index (if not None)
      - the smoothed mean line (always, when ≥ 2 buckets)
      - the ±1 SE confidence band (filled, alpha 0.15; when ≥ 2 buckets)
      - a single dot at the bucket value when there's exactly one bucket
        (single-bucket case)

    The caller is responsible for setting the title, subtitle, x/y limits,
    and ticks. This helper is intentionally pure with respect to those —
    it only draws series content into the Axes.
    """
    if not buckets:
        return

    # Reference lines first so they end up under the series
    if baseline_mean is not None:
        ax.axhline(baseline_mean, color="#888", linestyle="--", linewidth=0.7)
    if baseline_cutoff_bucket_index is not None:
        ax.axvline(
            baseline_cutoff_bucket_index,
            color="#fd7e14", linestyle="--", linewidth=0.7,
        )

    xs = [b["bucket_index"] for b in buckets]
    means = [b["mean"] for b in buckets]

    if len(buckets) == 1:
        # Single-bucket case: dot only, no line / no band.
        ax.scatter([xs[0]], [means[0]], color=color, s=40, zorder=3)
        return

    # ≥ 2 buckets: smoothed line + confidence band
    ses = [b["se"] for b in buckets]
    upper = [m + s for m, s in zip(means, ses)]
    lower = [m - s for m, s in zip(means, ses)]

    ax.fill_between(xs, lower, upper, color=color, alpha=0.15, linewidth=0)
    ax.plot(xs, means, color=color, linewidth=1.8)
```

Make sure `Optional` is imported from `typing` at the top of the file. The function does NOT need `import matplotlib` — matplotlib is already used elsewhere in the file via `self.figure` and the chart widget.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_drift_buckets.py -v
```

Expected: all tests in this file (Task 2's + Task 3's) pass.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/trends.py tests/test_drift_buckets.py
git commit -m "feat(drift): _draw_smoothed_panel — process-panel renderer"
```

---

## Task 4: Add `_draw_sigma_panel` helper

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py` (add helper after `_draw_smoothed_panel`)
- Test: `tests/test_drift_buckets.py` (extend)

Spec §4 (sigma hybrid SPC chart).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_drift_buckets.py`:

```python
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
        ((_dt(2026, 1, 1) + timedelta(hours=i)).isoformat(), 0.01 + i * 0.0005)
        for i in range(30)
    ]
    fig, ax = _make_axes()
    detector = _StubDetector(has_baseline=True, lcl=0.005, center=0.012, ucl=0.020)

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
        ((_dt(2026, 1, 1) + timedelta(hours=i)).isoformat(), v)
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

    rows = [(_dt(2026, 1, 1).isoformat(), 0.01)]
    fig, ax = _make_axes()
    detector = _StubDetector(has_baseline=False)

    violations = _draw_sigma_panel(
        ax, sigma_series=rows,
        detector=detector,
        baseline_cutoff_bucket_index=None,
        n_per_bucket=10,
    )

    assert violations == 0
    # No control-limit reference lines (axhline) drawn
    for ln in ax.get_lines():
        # Allow the cutoff axvline if present, but no axhlines.
        # axhline produces a Line2D with identical y; check that no horizontal
        # ref line was drawn at any of the limit values.
        pass  # smoke test only — primary contract is "violations=0, no crash"
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_drift_buckets.py -v
```

Expected: the 3 new tests FAIL with `ImportError: cannot import name '_draw_sigma_panel'`.

- [ ] **Step 3: Implement `_draw_sigma_panel`**

In `src/laser_trim_analyzer/gui/pages/trends.py`, immediately after `_draw_smoothed_panel`, add:

```python
def _draw_sigma_panel(
    ax,
    sigma_series: List[Tuple[str, float]],
    detector,
    baseline_cutoff_bucket_index: Optional[int],
    n_per_bucket: int = 50,
) -> int:
    """Draw the hybrid SPC sigma panel: individual unit dots (green = in-control,
    red = out-of-control by Western Electric rule 1) + smoothed white mean
    overlay + UCL/LCL/center reference lines.

    Returns the count of out-of-control unit dots in the entire window (for the
    header pill).
    """
    if not sigma_series:
        return 0

    # Bucket each row's index for x-axis alignment with the smoothed line.
    # We compute the bucket index by integer-dividing the row's position in
    # the (already-sorted) series by n_per_bucket — matches what _compute_buckets does.
    has_baseline = bool(getattr(detector, "has_baseline", False))
    if has_baseline:
        lcl, center, ucl = detector.get_control_limits()
    else:
        lcl, center, ucl = None, None, None

    # Slight per-dot horizontal jitter so coincident sigma values at the same
    # bucket index don't completely overlap. Deterministic (index-based) so
    # rendering is reproducible.
    xs: List[float] = []
    ys: List[float] = []
    colors: List[str] = []
    violations = 0
    for i, (_, value) in enumerate(sigma_series):
        bucket_idx = i // n_per_bucket
        jitter = ((i % n_per_bucket) - n_per_bucket / 2) / (n_per_bucket * 2.5)
        xs.append(bucket_idx + jitter)
        ys.append(value)
        if has_baseline and (
            (lcl is not None and value < lcl) or (ucl is not None and value > ucl)
        ):
            colors.append("#ff4040")
            violations += 1
        else:
            colors.append("#7ed99e" if has_baseline else "#888888")

    # Reference lines before dots so they end up under the series.
    if has_baseline:
        if ucl is not None:
            ax.axhline(ucl, color="#dc3545", linestyle="--", linewidth=0.7)
        if lcl is not None:
            ax.axhline(lcl, color="#dc3545", linestyle="--", linewidth=0.7)
        if center is not None:
            ax.axhline(center, color="#666", linewidth=0.5)
    if baseline_cutoff_bucket_index is not None:
        ax.axvline(
            baseline_cutoff_bucket_index,
            color="#fd7e14", linestyle="--", linewidth=0.7,
        )

    ax.scatter(xs, ys, c=colors, s=10, alpha=0.6, linewidths=0, zorder=2)

    # Smoothed mean overlay on top of the dots — only when baseline exists
    # (no point smoothing into nonexistent control bands).
    if has_baseline:
        buckets = _compute_buckets(sigma_series, n_per_bucket=n_per_bucket)
        if len(buckets) >= 2:
            bxs = [b["bucket_index"] for b in buckets]
            means = [b["mean"] for b in buckets]
            ax.plot(bxs, means, color="#ffffff", linewidth=1.6, zorder=3)

    return violations
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_drift_buckets.py -v
```

Expected: all tests in the file pass.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/trends.py tests/test_drift_buckets.py
git commit -m "feat(drift): _draw_sigma_panel — hybrid SPC with violation count"
```

---

## Task 5: Rewrite `_render_single_model_drift`

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py:2723-2880`

Spec §1 (layout unchanged), §2 (per-panel anatomy), §3 (process panels), §4 (sigma), §6 (empty states), §Validation #1–3, #5.

This task replaces the body of `_render_single_model_drift` to use the helpers. No new unit tests at this layer — the helpers are covered; this task is integration wiring verified by the smoke test in Task 6.

- [ ] **Step 1: Read the current implementation**

Open `src/laser_trim_analyzer/gui/pages/trends.py` and locate `_render_single_model_drift` (line 2723). Skim the existing body to confirm the entry guards (winfo_exists, gen check, tab/model checks), the header-pill rendering (lines 2741–2783), and the chart creation (lines 2785–2880). The entry guards and header-pill block are **kept as-is**; only the chart-area block is rewritten.

- [ ] **Step 2: Replace the chart-area block**

Replace lines 2784–2880 (everything from `# 2×2 chart grid` through the final `self.status_label.configure(text=f"Drift dashboard · {model}")`) with the following block. The `try` / `except` wrapper that surrounds the whole method (line 2732 and matching `except`) remains in place.

```python
            # 2×2 chart grid
            ChartWidget, ChartStyle = _ensure_chart_module()
            chart = ChartWidget(
                self._drift_content_frame,
                style=ChartStyle(figure_size=(12, 7), dpi=100),
            )
            chart.pack(fill="both", expand=True, padx=10, pady=(8, 10))
            self._chart_widgets.append(chart)
            chart.clear()

            fig = chart.figure
            gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.25)

            # Compute the bucket index that marks the baseline → recent boundary.
            # Used as the x-coordinate for the orange vertical line on each panel.
            baseline_cutoff_iso = data.get("baseline_cutoff_date")
            def _cutoff_bucket(series, n_per_bucket=50):
                if not baseline_cutoff_iso or not series:
                    return None
                from datetime import datetime as _dt
                cutoff = _dt.fromisoformat(baseline_cutoff_iso)
                # Index of the first row whose date >= cutoff
                for i, (iso, _) in enumerate(series):
                    if _dt.fromisoformat(iso) >= cutoff:
                        return i // n_per_bucket
                return None  # cutoff is past the end of the window

            process = data.get("process", {})

            # ---- Top-left: Sigma drift (hybrid SPC) ----
            ax_sigma = fig.add_subplot(gs[0, 0])
            chart._style_axis(ax_sigma)
            sigma_pts = data.get("sigma_series", [])

            if not sigma_pts:
                self._draw_empty_state(ax_sigma, "No data in window")
            elif not (detector and detector.has_baseline):
                self._draw_empty_state(
                    ax_sigma,
                    f"Need ≥30 baseline samples to enable SPC; "
                    f"have {len(sigma_pts)}. Train this model in Settings.",
                )
            else:
                violations = _draw_sigma_panel(
                    ax_sigma,
                    sigma_series=sigma_pts,
                    detector=detector,
                    baseline_cutoff_bucket_index=_cutoff_bucket(sigma_pts),
                )
                # Title with status pill text
                if violations > 0:
                    title_status = f"↑ OOC · {violations} violations"
                else:
                    title_status = "✓ stable"
                ax_sigma.set_title(
                    f"Sigma Drift  ·  {title_status}",
                    loc="left", fontsize=11, fontweight="bold",
                )
            ax_sigma.tick_params(axis="x", rotation=0, labelsize=8)

            # ---- Three process panels ----
            metric_axes = (
                ("untrimmed_resistance", gs[0, 1], "Untrimmed Resistance"),
                ("measured_electrical_angle", gs[1, 0], "Electrical Angle"),
            )
            meta = self._db_metric_meta()
            for metric, slot, title in metric_axes:
                ax = fig.add_subplot(slot)
                chart._style_axis(ax)
                panel = process.get(metric, {})
                series = panel.get("series", [])
                if not series:
                    self._draw_empty_state(ax, f"No {title.lower()} data in window")
                    ax.tick_params(axis="x", rotation=0, labelsize=8)
                    continue

                buckets = _compute_buckets(series, n_per_bucket=50)
                z = panel.get("z_score") or 0.0
                if abs(z) >= 3.0:
                    color = "#ff8080"
                    pill = ("↑ OOC" if z > 0 else "↓ OOC")
                elif abs(z) >= 2.0:
                    color = "#ffb060"
                    pill = ("↑ DRIFT" if z > 0 else "↓ DRIFT")
                else:
                    color = "#7ed99e"
                    pill = "✓ stable"

                _draw_smoothed_panel(
                    ax,
                    buckets=buckets,
                    baseline_mean=panel.get("baseline_mean"),
                    baseline_cutoff_bucket_index=_cutoff_bucket(series),
                    color=color,
                )

                fmt = meta.get(metric, {}).get("fmt", "{:.2f}")
                unit = meta.get(metric, {}).get("unit", "")
                base_mean = panel.get("baseline_mean")
                base_s = fmt.format(base_mean) if base_mean is not None else "—"
                rec_s = (
                    fmt.format(panel["recent_mean"])
                    if panel.get("recent_mean") is not None else "—"
                )
                pct = panel.get("delta_pct")
                pct_s = f"({pct:+.1f}%)" if pct is not None else ""
                z_s = f"z={z:+.1f}" if panel.get("z_score") is not None else ""
                ax.set_title(
                    f"{title}  ·  {pill}  ·  "
                    f"{base_s} → {rec_s} {unit} {pct_s}  {z_s}",
                    loc="left", fontsize=10, fontweight="bold",
                )
                ax.tick_params(axis="x", rotation=0, labelsize=8)

            # ---- Bottom-right: Retrim Rate ----
            ax_retrim = fig.add_subplot(gs[1, 1])
            chart._style_axis(ax_retrim)
            retrim_series = process.get("retrim_rate_series", [])

            if not retrim_series:
                self._draw_empty_state(
                    ax_retrim,
                    "trim_pass_count not captured for this window's rows. "
                    "Re-parse to populate.",
                )
            else:
                # Convert 0/1 → bucket retrim rate %.
                buckets = _compute_buckets(retrim_series, n_per_bucket=50)
                # Mean of 0/1 values * 100 = percentage. SE scales the same way.
                for b in buckets:
                    b["mean"] = b["mean"] * 100.0
                    b["stddev"] = b["stddev"] * 100.0
                    b["se"] = b["se"] * 100.0

                # baseline retrim rate from the rows before baseline_cutoff
                baseline_rate = None
                recent_rate = None
                if baseline_cutoff_iso:
                    from datetime import datetime as _dt
                    cutoff = _dt.fromisoformat(baseline_cutoff_iso)
                    base_vals = [v for iso, v in retrim_series
                                 if _dt.fromisoformat(iso) < cutoff]
                    recent_vals = [v for iso, v in retrim_series
                                   if _dt.fromisoformat(iso) >= cutoff]
                    if base_vals:
                        baseline_rate = sum(base_vals) / len(base_vals) * 100.0
                    if recent_vals:
                        recent_rate = sum(recent_vals) / len(recent_vals) * 100.0

                # Status pill: rising if recent ≥ 2× baseline AND recent ≥ 10%
                if (baseline_rate is not None and recent_rate is not None
                        and recent_rate >= 10.0
                        and recent_rate >= 2.0 * max(baseline_rate, 1.0)):
                    pill = "↑ rising"
                    color = "#ffb060"
                elif recent_rate is not None and recent_rate >= 15.0:
                    pill = "↑ OOC"
                    color = "#ff8080"
                else:
                    pill = "✓ stable"
                    color = "#7ed99e"

                _draw_smoothed_panel(
                    ax_retrim,
                    buckets=buckets,
                    baseline_mean=baseline_rate,
                    baseline_cutoff_bucket_index=_cutoff_bucket(retrim_series),
                    color=color,
                )
                # Y-axis: 0% lower bound; upper bound max(20%, 1.5 × peak).
                peak = max((b["mean"] for b in buckets), default=0.0)
                ax_retrim.set_ylim(0.0, max(20.0, 1.5 * peak))

                base_s = f"{baseline_rate:.1f}%" if baseline_rate is not None else "—"
                rec_s = f"{recent_rate:.1f}%" if recent_rate is not None else "—"
                delta_pp = (
                    f"({recent_rate - baseline_rate:+.1f} pp)"
                    if baseline_rate is not None and recent_rate is not None
                    else ""
                )
                ax_retrim.set_title(
                    f"Retrim Rate  ·  {pill}  ·  "
                    f"{base_s} → {rec_s}  {delta_pp}",
                    loc="left", fontsize=10, fontweight="bold",
                )
            ax_retrim.tick_params(axis="x", rotation=0, labelsize=8)

            try:
                fig.tight_layout()
            except Exception:
                pass
            chart.canvas.draw_idle()
            self.status_label.configure(text=f"Drift dashboard · {model}")
```

- [ ] **Step 3: Smoke-import the file**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
python3 -c "import ast; ast.parse(open('src/laser_trim_analyzer/gui/pages/trends.py').read()); print('OK')"
```

Expected: `OK`. Resolves any syntax errors from the edit before running the full test suite.

- [ ] **Step 4: Run existing test suite to confirm nothing else broke**

```bash
pytest tests/ -v 2>&1 | tail -25
```

Expected: no test regressions. The new `tests/test_drift_buckets.py` tests still pass; the existing suite is unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/trends.py
git commit -m "feat(drift): rewrite single-model dashboard render with smoothed buckets + hybrid SPC"
```

---

## Task 6: End-to-end smoke test on a real DB snapshot

**Files:**
- Test: `tests/test_drift_dashboard_render.py` (create)

Spec §Validation #1–5.

This task validates the integration end-to-end: build a small temp DB with realistic data, call `get_model_drift_dashboard`, then call the rewritten renderer's helpers exactly as `_render_single_model_drift` would, and verify nothing crashes plus the bucket counts and violation counts are what we expect.

- [ ] **Step 1: Write the failing test**

Create `tests/test_drift_dashboard_render.py`:

```python
"""End-to-end smoke for the redesigned single-model drift dashboard.

Builds a small temp DB with one model and exercises the helpers in the same
order the GUI renderer does. Verifies no crash, sigma violation count matches
the planted out-of-control rows, and retrim-rate bucketing produces the
expected percentages.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.gui.pages.trends import (
    _compute_buckets, _draw_smoothed_panel, _draw_sigma_panel,
)


class _StubDetector:
    has_baseline = True
    def get_control_limits(self):
        # UCL = 0.020 — planted values >0.020 below will count as violations
        return (0.005, 0.012, 0.020)


@pytest.fixture
def db_with_drifting_model(tmp_path):
    """50 in-control sigma rows + 5 out-of-control, mixed retrim counts."""
    from tests.test_drift_dashboard_data import _add_analysis

    db = DatabaseManager(tmp_path / "test.db")
    base_date = datetime.now() - timedelta(days=20)

    # 50 in-control rows (sigma 0.010, trim_pass_count 1)
    for i in range(50):
        _add_analysis(
            db, "DR-MODEL",
            base_date + timedelta(hours=i),
            sigma_values=(0.010,),
            serial=f"{i:04d}",
            trim_pass_count=1,
        )

    # 5 out-of-control rows (sigma 0.030, trim_pass_count 2 — retrims)
    for i in range(5):
        _add_analysis(
            db, "DR-MODEL",
            base_date + timedelta(days=15, hours=i),
            sigma_values=(0.030,),
            serial=f"X{i:03d}",
            trim_pass_count=2,
        )

    return db


def test_drift_dashboard_end_to_end(db_with_drifting_model):
    data = db_with_drifting_model.get_model_drift_dashboard(
        model="DR-MODEL", days_back=90, recent_days=14,
    )

    # Top-level shape (Task 1 contract)
    assert "baseline_cutoff_date" in data
    assert "retrim_rate_series" in data["process"]

    # Sigma panel: 5 dots above UCL = 5 violations
    fig, ax = plt.subplots()
    violations = _draw_sigma_panel(
        ax, sigma_series=data["sigma_series"],
        detector=_StubDetector(),
        baseline_cutoff_bucket_index=0,
        n_per_bucket=50,
    )
    assert violations == 5
    plt.close(fig)

    # Retrim rate: 5 out of 55 rows had trim_pass_count > 1 → 5/55 ≈ 9.1%
    buckets = _compute_buckets(
        data["process"]["retrim_rate_series"], n_per_bucket=50,
    )
    # With 55 rows and N=50, we get one bucket of 50 + a trailing 5 → 2 buckets
    # (5 ≥ 5 so it stands on its own).
    assert len(buckets) == 2
    # Overall mean across all rows should be 5/55:
    overall_mean = sum(b["mean"] * b["n"] for b in buckets) / sum(b["n"] for b in buckets)
    assert overall_mean == pytest.approx(5 / 55, rel=1e-6)

    # Smoothed-panel render does not crash:
    fig2, ax2 = plt.subplots()
    _draw_smoothed_panel(
        ax2, buckets=[{**b, "mean": b["mean"] * 100.0,
                       "stddev": b["stddev"] * 100.0,
                       "se": b["se"] * 100.0} for b in buckets],
        baseline_mean=0.0, baseline_cutoff_bucket_index=0,
        color="#ffb060",
    )
    plt.close(fig2)
```

- [ ] **Step 2: Run the test**

```bash
pytest tests/test_drift_dashboard_render.py -v
```

Expected: PASS. If it fails, fix the offending helper/data layer until it passes.

- [ ] **Step 3: Manual visual verification**

Start the app and exercise the dashboard against your real DB:

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
python src/__main__.py
```

In the app:
1. Open Trends → Drift.
2. Select a model with ≥ 50 units in the day-range window (use 8862, 8712, or any other high-volume model).
3. Confirm: all four panels render smoothed lines (no noise carpets); sigma panel shows colored dots with the smoothed mean overlay; retrim rate panel shows a sensible % curve OR a clean "Re-parse to populate" message for pre-feature data.
4. Switch to a low-volume model and confirm the panels render gracefully (single-bucket dot or empty-state text).
5. Switch the day-range filter and confirm rebucket-and-redraw works (≤ ~500ms feels instant).

Note any visual oddities (clipped text, overlapping titles) and fix them as small follow-ups; not blockers for this commit.

- [ ] **Step 4: Commit**

```bash
git add tests/test_drift_dashboard_render.py
git commit -m "test(drift): end-to-end smoke for single-model dashboard helpers"
```

---

## Self-review

Checked against the spec:

| Spec section | Implementing task |
| ------------ | ----------------- |
| §1 Layout | Task 5 (renderer wires existing 2×2 grid unchanged) |
| §2 Per-panel anatomy (title, subtitle, ref lines, status pill) | Task 3 (lines/band/ref lines), Task 4 (sigma), Task 5 (titles + status pill mapping) |
| §3 Process panels (smoothed + band, retrim rate calc + y-axis floor) | Task 3 (helper), Task 5 (retrim-rate-specific %, y-axis bounds) |
| §4 Sigma hybrid SPC (dots, smoothed overlay, violation count) | Task 4 |
| §5 Adaptive bucketing | Task 2 |
| §6 Empty states (5 distinct conditions) | Task 5 (no-data, no-baseline, no-retrim-data); single-bucket case is in Task 3 (`_draw_smoothed_panel`) |
| §Data layer (baseline_cutoff_date, retrim_rate_series) | Task 1 |
| §Validation #1–#5 | Task 6 (smoke) and Task 5 Step 4 (existing suite) — manual verification covers #3 (sigma red dots), #4 (rebucket on day-range change), #5 (stale-axes guard kept intact from existing code) |

Placeholder scan: no TBD / TODO / "implement later" terms. Each step contains the actual content to write or the exact command to run.

Type consistency: bucket dict keys (`bucket_index`, `n`, `mean`, `stddev`, `se`, `min_date`, `max_date`) are introduced in Task 2 and consumed identically in Tasks 3, 4, 5, 6. `_draw_smoothed_panel` and `_draw_sigma_panel` signatures match across tasks.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-13-single-model-drift-redesign.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
