# Trends/Dashboard Cleanup — Final Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface four underused track-level signals (`deviation_uniformity`, anomaly rate, average error reduction, plus removing the redundant Dashboard Pareto chart) so each lands in the place where it answers a real operator question.

**Architecture:** Three of the four items are pure rendering changes that consume signals already on `TrackResult` rows. Two extend existing aggregation methods on `DatabaseManager` (`get_anomaly_rate_by_model`, `get_trim_difficulty_by_model`). One restructures the Dashboard grid. No schema changes, no migrations, no new ML.

**Tech Stack:** Python 3.x, SQLAlchemy 2.x ORM, CustomTkinter for GUI, matplotlib via the in-house `ChartWidget` wrapper. Existing test suite uses pytest at `tests/`.

**Source spec:** `docs/superpowers/specs/2026-05-07-trends-cleanup-final-design.md` (commit `0c78917`).

---

## File Structure

Tasks 1, 3, and 4 each add or extend one method/widget; Task 2 deletes one panel and re-grids one container. No new files.

| File | Change |
|---|---|
| `src/laser_trim_analyzer/gui/pages/analyze.py` | Modify `_display_metrics` Failure Margin block |
| `src/laser_trim_analyzer/database/manager.py` | Add `get_anomaly_rate_by_model`; extend `get_trim_difficulty_by_model` |
| `src/laser_trim_analyzer/gui/pages/trends.py` | Modify `_render_trim_difficulty` annotation; modify the Active Models Summary table to display anomaly columns |
| `src/laser_trim_analyzer/gui/pages/dashboard.py` | Delete Pareto frame; re-grid Where-to-Focus to span columns 1+2 |
| `tests/test_anomaly_rate_by_model.py` | NEW — covers the new DB method |
| `tests/test_trim_difficulty_avg_error_reduction.py` | NEW — covers the extended DB method |

---

## Task 1: Add `deviation_uniformity` line to Failure Margin block

Pure UI, single-file, smallest change. Validates we have the right block to modify before touching anything else.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/analyze.py` (the FAILURE MARGIN section inside `_display_metrics`)

There is no test layer for this block — `_display_metrics` writes plain text into a CTkTextbox and isn't unit-testable without a Tk root. Verification is visual + syntax check.

- [ ] **Step 1: Locate the FAILURE MARGIN block in `_display_metrics`**

Run: `grep -n "FAILURE MARGIN\|Margin to spec\|Max violation" src/laser_trim_analyzer/gui/pages/analyze.py`

Expected output: matching lines around the recently-added Failure Margin block (commit `a5887e9`). Confirm the block exists between the `Result:` line and the ML failure probability line.

- [ ] **Step 2: Add the `deviation_uniformity` line**

Insert immediately after the `max_viol`/`avg_viol` rendering, BEFORE the `ML failure prob` line.

```python
                # deviation_uniformity is the coefficient of variation of
                # absolute errors across the track. < 1.0 = uniform across
                # length, ≥ 1.5 = concentrated in one region (early warning
                # of element heterogeneity even on a passing track).
                dev_unif = getattr(track, "deviation_uniformity", None)
                if dev_unif is not None:
                    lines.append(
                        f"    Error uniformity: {dev_unif:.2f}  "
                        f"(1.0 = uniform across track, ≥1.5 = concentrated)"
                    )
```

The exact `old_string` to anchor against (use Edit with replace_all=False):

```python
                if max_viol is None and avg_viol is None:
                    lines.append("    Violation data: N/A")
                # Keep the ML prediction here; it's a different signal
                # (predicted FT failure) and useful alongside the margin.
                if track.failure_probability is not None:
```

Replace with:

```python
                if max_viol is None and avg_viol is None:
                    lines.append("    Violation data: N/A")
                # deviation_uniformity is the coefficient of variation of
                # absolute errors across the track. < 1.0 = uniform across
                # length, ≥ 1.5 = concentrated in one region (early warning
                # of element heterogeneity even on a passing track).
                dev_unif = getattr(track, "deviation_uniformity", None)
                if dev_unif is not None:
                    lines.append(
                        f"    Error uniformity: {dev_unif:.2f}  "
                        f"(1.0 = uniform across track, ≥1.5 = concentrated)"
                    )
                # Keep the ML prediction here; it's a different signal
                # (predicted FT failure) and useful alongside the margin.
                if track.failure_probability is not None:
```

- [ ] **Step 3: Syntax check**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m py_compile src/laser_trim_analyzer/gui/pages/analyze.py && echo OK`

Expected output: `OK`.

- [ ] **Step 4: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/gui/pages/analyze.py
git commit -m "feat(analyze): show deviation_uniformity in Failure Margin block

Coefficient of variation of absolute errors across the track. <1.0 =
uniform along length; ≥1.5 = concentrated in one region. Early-warning
signal of element heterogeneity even on a passing track. Stored on
every TrackResult but never displayed before this commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add `get_anomaly_rate_by_model` DB method (test-first)

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`
- Test: `tests/test_anomaly_rate_by_model.py` (NEW)

- [ ] **Step 1: Write the failing test**

Create `tests/test_anomaly_rate_by_model.py`:

```python
"""Tests for DatabaseManager.get_anomaly_rate_by_model."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import (
    AnalysisResult,
    TrackResult,
    StatusType,
    SystemType,
    RiskCategory,
)


@pytest.fixture
def db(tmp_path):
    mgr = DatabaseManager(db_path=tmp_path / "anomaly.db")
    return mgr


def _add_track(db, model: str, file_date: datetime, *, is_anomaly: bool):
    """Helper: insert one analysis row with one track row."""
    with db.session() as session:
        analysis = AnalysisResult(
            filename=f"{model}_sn{is_anomaly}_{file_date.timestamp()}.xls",
            file_path=f"/fake/{model}.xls",
            file_date=file_date,
            model=model,
            serial=f"sn{int(file_date.timestamp())}",
            system=SystemType.A,
            has_multi_tracks=False,
            overall_status=StatusType.PASS,
            timestamp=datetime.utcnow(),
        )
        session.add(analysis)
        session.flush()
        track = TrackResult(
            analysis_id=analysis.id,
            track_id="TRK1",
            status=StatusType.PASS,
            travel_length=10.0,
            linearity_spec=0.01,
            sigma_gradient=0.001,
            sigma_threshold=0.005,
            sigma_pass=True,
            optimal_offset=0.0,
            final_linearity_error_shifted=0.001,
            linearity_pass=True,
            linearity_fail_points=0,
            risk_category=RiskCategory.LOW,
            is_anomaly=is_anomaly,
        )
        session.add(track)


def test_returns_empty_when_no_data(db):
    rows = db.get_anomaly_rate_by_model(days_back=90, min_samples=1)
    assert rows == []


def test_counts_anomalies_per_model(db):
    now = datetime.utcnow()
    # Model A: 8 normal + 2 anomaly -> 20% anomaly rate
    for _ in range(8):
        _add_track(db, "MODEL-A", now - timedelta(days=1), is_anomaly=False)
    for _ in range(2):
        _add_track(db, "MODEL-A", now - timedelta(days=1), is_anomaly=True)
    # Model B: 10 normal -> 0% rate
    for _ in range(10):
        _add_track(db, "MODEL-B", now - timedelta(days=1), is_anomaly=False)

    rows = db.get_anomaly_rate_by_model(days_back=30, min_samples=5)
    by_model = {r["model"]: r for r in rows}
    assert by_model["MODEL-A"]["anomaly_count"] == 2
    assert by_model["MODEL-A"]["total_tracks"] == 10
    assert by_model["MODEL-A"]["anomaly_rate"] == pytest.approx(20.0)
    assert by_model["MODEL-B"]["anomaly_count"] == 0
    assert by_model["MODEL-B"]["anomaly_rate"] == pytest.approx(0.0)


def test_filters_by_min_samples(db):
    now = datetime.utcnow()
    for _ in range(3):
        _add_track(db, "TINY", now - timedelta(days=1), is_anomaly=True)

    rows = db.get_anomaly_rate_by_model(days_back=30, min_samples=10)
    assert all(r["model"] != "TINY" for r in rows)


def test_filters_by_days_back(db):
    now = datetime.utcnow()
    # Older than the window
    for _ in range(20):
        _add_track(db, "OLD", now - timedelta(days=120), is_anomaly=True)

    rows = db.get_anomaly_rate_by_model(days_back=30, min_samples=5)
    assert all(r["model"] != "OLD" for r in rows)


def test_sorted_by_rate_descending(db):
    now = datetime.utcnow()
    # MODEL-LOW: 1/20 = 5%
    for _ in range(19):
        _add_track(db, "MODEL-LOW", now - timedelta(days=1), is_anomaly=False)
    _add_track(db, "MODEL-LOW", now - timedelta(days=1), is_anomaly=True)
    # MODEL-HIGH: 5/10 = 50%
    for _ in range(5):
        _add_track(db, "MODEL-HIGH", now - timedelta(days=1), is_anomaly=False)
    for _ in range(5):
        _add_track(db, "MODEL-HIGH", now - timedelta(days=1), is_anomaly=True)

    rows = db.get_anomaly_rate_by_model(days_back=30, min_samples=5)
    models_in_order = [r["model"] for r in rows]
    assert models_in_order.index("MODEL-HIGH") < models_in_order.index("MODEL-LOW")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest tests/test_anomaly_rate_by_model.py -v`

Expected: every test fails with `AttributeError: 'DatabaseManager' object has no attribute 'get_anomaly_rate_by_model'`.

- [ ] **Step 3: Implement the method**

In `src/laser_trim_analyzer/database/manager.py`, locate `get_trim_difficulty_by_model` and insert the new method directly above it. The implementation:

```python
    def get_anomaly_rate_by_model(
        self,
        days_back: int = 90,
        min_samples: int = 10,
    ) -> List[Dict[str, Any]]:
        """Aggregate anomaly flag rate per model over a window.

        is_anomaly is set per-track when the trim has the linear-slope
        signature of a true trim failure (vs random noise). Rolling it
        up per model surfaces persistent setup issues — e.g. a model
        with 12 anomalies in 30 days likely has a fixture or operator
        problem rather than random material variation.

        Args:
            days_back: window length in days, anchored to file_date.
            min_samples: minimum total tracks per model to include in
                the result so a single anomaly on a low-volume model
                doesn't dominate the ranking.

        Returns:
            List of dicts sorted by anomaly_rate descending. Each dict:
                model, total_tracks, anomaly_count, anomaly_rate
                (percent), last_anomaly_date.
        """
        with self.session() as session:
            cutoff = datetime.now() - timedelta(days=days_back)
            rows = (
                session.query(
                    DBAnalysisResult.model,
                    func.count(DBTrackResult.id).label("total_tracks"),
                    func.sum(
                        case((DBTrackResult.is_anomaly == True, 1), else_=0)
                    ).label("anomaly_count"),
                    func.max(
                        case(
                            (DBTrackResult.is_anomaly == True,
                             DBAnalysisResult.file_date),
                            else_=None,
                        )
                    ).label("last_anomaly_date"),
                )
                .join(DBTrackResult, DBAnalysisResult.id == DBTrackResult.analysis_id)
                .filter(
                    DBAnalysisResult.model.isnot(None),
                    DBAnalysisResult.model != "Unknown",
                    DBAnalysisResult.file_date >= cutoff,
                )
                .group_by(DBAnalysisResult.model)
                .having(func.count(DBTrackResult.id) >= min_samples)
                .all()
            )

            results = []
            for r in rows:
                total = int(r.total_tracks or 0)
                anom = int(r.anomaly_count or 0)
                rate = (anom / total * 100.0) if total else 0.0
                results.append({
                    "model": r.model,
                    "total_tracks": total,
                    "anomaly_count": anom,
                    "anomaly_rate": rate,
                    "last_anomaly_date": r.last_anomaly_date,
                })
            results.sort(key=lambda r: -r["anomaly_rate"])
            return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest tests/test_anomaly_rate_by_model.py -v`

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/database/manager.py tests/test_anomaly_rate_by_model.py
git commit -m "feat(db): get_anomaly_rate_by_model — per-model anomaly rollup

Aggregates is_anomaly flag count per model over a date window.
Rolling per-track anomalies up to the model level surfaces
persistent setup issues (e.g. 12 anomalies in 30 days = fixture or
operator problem, not random variation). Returns list sorted by
anomaly_rate descending; min_samples filter prevents low-volume
models from dominating the ranking.

Tests cover: empty DB, multi-model counting, min_samples filter,
days_back filter, sort ordering.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Surface anomaly rate in Trends Active Models Summary

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py`

The Trends Standard summary view already has an "Active Models Summary" stats display. We add anomaly-rate columns to its rows.

- [ ] **Step 1: Find the Active Models Summary update method**

Run: `grep -n "active_models_data\|Active Models Summary\|_update_summary_display\|_load_summary_data" src/laser_trim_analyzer/gui/pages/trends.py | head -20`

Note the line that loads `active_models` and the function that renders it. The data is fetched in `_load_summary_data` and rendered as part of `_update_summary_display`.

- [ ] **Step 2: Augment `_load_summary_data` to also fetch anomaly rates**

Find the block in `_load_summary_data` that calls `db.get_active_models_summary(...)`. Immediately after, add a parallel fetch:

```python
        # Anomaly rates per model — surfaced as additional columns on the
        # Active Models Summary so persistent setup issues are visible
        # at the model level (per-track is_anomaly was visible per-unit
        # but never rolled up before).
        try:
            anomaly_rows = db.get_anomaly_rate_by_model(
                days_back=self.selected_days, min_samples=10
            )
            anomaly_by_model = {r["model"]: r for r in anomaly_rows}
        except Exception as e:
            logger.debug(f"Could not load anomaly rates: {e}")
            anomaly_by_model = {}
```

Pass `anomaly_by_model` into the `self.after(0, ...)` call by adding it to the kwargs the `_update_summary_display_if_current` callback forwards.

The exact `old_string` to find:

```python
        # Update UI on main thread; capture gen so we can discard stale loads
        self.after(0, lambda g=gen: self._update_summary_display_if_current(
            g, active_models, alert_models, model_names, trending_worse,
            mps_models=mps_models, recent_days=recent_days,
            priority_models=priority_models, heatmap_data=heatmap_data,
            ml_insights=ml_insights
        ))
```

Replace with:

```python
        # Anomaly rates per model — surfaced as additional columns on the
        # Active Models Summary so persistent setup issues are visible
        # at the model level (per-track is_anomaly was visible per-unit
        # but never rolled up before).
        try:
            anomaly_rows = db.get_anomaly_rate_by_model(
                days_back=self.selected_days, min_samples=10
            )
            anomaly_by_model = {r["model"]: r for r in anomaly_rows}
        except Exception as e:
            logger.debug(f"Could not load anomaly rates: {e}")
            anomaly_by_model = {}

        # Update UI on main thread; capture gen so we can discard stale loads
        self.after(0, lambda g=gen: self._update_summary_display_if_current(
            g, active_models, alert_models, model_names, trending_worse,
            mps_models=mps_models, recent_days=recent_days,
            priority_models=priority_models, heatmap_data=heatmap_data,
            ml_insights=ml_insights, anomaly_by_model=anomaly_by_model,
        ))
```

- [ ] **Step 3: Plumb `anomaly_by_model` through the gen-guard wrapper**

Find `_update_summary_display_if_current` (defined just below `_load_summary_data`) and ensure it passes through any kwargs. If it uses `**kwargs`, no change needed. If it lists kwargs explicitly, add `anomaly_by_model=None` to the signature and the forwarding call.

Run: `grep -n "_update_summary_display_if_current\|_update_summary_display(" src/laser_trim_analyzer/gui/pages/trends.py`

Confirm the wrapper uses `*args, **kwargs` (it should — check). If not, update both the wrapper and `_update_summary_display`.

- [ ] **Step 4: Add `anomaly_by_model` to `_update_summary_display` signature**

Find:

```python
    def _update_summary_display(
        self,
        active_models: List[Dict[str, Any]],
        alert_models: List[Dict[str, Any]],
        model_names: List[str],
        trending_worse: Optional[List[Dict[str, Any]]] = None,
        mps_models: Optional[List[str]] = None,
        recent_days: int = 90,
        priority_models: Optional[List[Dict[str, Any]]] = None,
        heatmap_data: Optional[Dict[str, Any]] = None,
        ml_insights: Optional[Dict[str, Any]] = None,
    ):
```

Add the new parameter at the end:

```python
    def _update_summary_display(
        self,
        active_models: List[Dict[str, Any]],
        alert_models: List[Dict[str, Any]],
        model_names: List[str],
        trending_worse: Optional[List[Dict[str, Any]]] = None,
        mps_models: Optional[List[str]] = None,
        recent_days: int = 90,
        priority_models: Optional[List[Dict[str, Any]]] = None,
        heatmap_data: Optional[Dict[str, Any]] = None,
        ml_insights: Optional[Dict[str, Any]] = None,
        anomaly_by_model: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
```

- [ ] **Step 5: Locate where active_models rows are rendered**

The existing summary stats live in `self.summary_stat_labels` populated in `_create_summary_view`. The audit said the row already iterates per-active-model. Search for the iteration:

Run: `grep -n "active_models\b\|active_models_data\b" src/laser_trim_analyzer/gui/pages/trends.py | head -10`

Find where each model's stats get displayed (likely a label-update loop or a row-creation loop in `_update_summary_display`).

- [ ] **Step 6: Append anomaly columns to each rendered row**

Where the per-model display row is built, after the existing pass-rate / sample-count cells, append:

```python
            # Anomaly rate column — color-coded amber/red at thresholds.
            anom = (anomaly_by_model or {}).get(model_name)
            if anom is None:
                anom_text = "—"
                anom_color = "gray"
            else:
                rate = anom["anomaly_rate"]
                count = anom["anomaly_count"]
                anom_text = f"{count}  ({rate:.1f}%)"
                if rate >= 15.0:
                    anom_color = "#dc3545"  # red — persistent setup issue
                elif rate >= 5.0:
                    anom_color = "#fd7e14"  # amber — watch list
                else:
                    anom_color = "white"
```

Apply that text + color to a new label in the same row layout the existing columns use. The exact widget type (CTkLabel) and grid/pack call must match the surrounding columns — open the file and copy the existing pattern.

- [ ] **Step 7: Add a column header for the new field**

Wherever the summary-stats column headers are defined (look for the existing "Model", "Pass Rate", etc. headers), append `"Anomaly (count, rate)"`.

- [ ] **Step 8: Syntax check**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m py_compile src/laser_trim_analyzer/gui/pages/trends.py && echo OK`

Expected: `OK`.

- [ ] **Step 9: Smoke-import test**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -c "import sys; sys.path.insert(0,'src'); from laser_trim_analyzer.gui.pages.trends import TrendsPage; print('import OK')"`

Expected: `import OK`.

- [ ] **Step 10: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/gui/pages/trends.py
git commit -m "feat(trends): show per-model anomaly rate in Active Models Summary

Adds an Anomaly column to the Trends Standard summary table, showing
count and rate per model over the selected date window. Color-coded
amber at ≥5% and red at ≥15% so persistent setup issues are visible
at a glance without drilling into individual files. Backed by the new
get_anomaly_rate_by_model DB method.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Extend `get_trim_difficulty_by_model` with avg error reduction (test-first)

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py` (existing `get_trim_difficulty_by_model`)
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py` (`_render_trim_difficulty` annotation)
- Test: `tests/test_trim_difficulty_avg_error_reduction.py` (NEW)

- [ ] **Step 1: Write the failing test**

Create `tests/test_trim_difficulty_avg_error_reduction.py`:

```python
"""Tests for the avg_error_reduction extension to get_trim_difficulty_by_model."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import (
    AnalysisResult,
    TrackResult,
    StatusType,
    SystemType,
    RiskCategory,
)


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=tmp_path / "trim_difficulty.db")


def _add_track(db, model, file_date, *, trim_passes, error_reduction):
    with db.session() as session:
        analysis = AnalysisResult(
            filename=f"{model}_{file_date.timestamp()}.xls",
            file_path=f"/fake/{model}.xls",
            file_date=file_date,
            model=model,
            serial=f"sn{int(file_date.timestamp())}",
            system=SystemType.A,
            has_multi_tracks=False,
            overall_status=StatusType.PASS,
            timestamp=datetime.utcnow(),
        )
        session.add(analysis)
        session.flush()
        track = TrackResult(
            analysis_id=analysis.id,
            track_id="TRK1",
            status=StatusType.PASS,
            travel_length=10.0,
            linearity_spec=0.01,
            sigma_gradient=0.001,
            sigma_threshold=0.005,
            sigma_pass=True,
            optimal_offset=0.0,
            final_linearity_error_shifted=0.001,
            linearity_pass=True,
            linearity_fail_points=0,
            risk_category=RiskCategory.LOW,
            trim_pass_count=trim_passes,
            max_error_reduction_percent=error_reduction,
        )
        session.add(track)


def test_avg_error_reduction_present_when_data_exists(db):
    now = datetime.utcnow()
    for er in [40.0, 60.0, 50.0, 70.0, 80.0]:
        _add_track(db, "MODEL-X", now - timedelta(days=1),
                    trim_passes=2, error_reduction=er)

    rows = db.get_trim_difficulty_by_model(
        days_back=30, min_units=5, limit=10
    )
    by_model = {r["model"]: r for r in rows}
    assert "MODEL-X" in by_model
    assert by_model["MODEL-X"]["avg_error_reduction"] == pytest.approx(60.0)


def test_avg_error_reduction_none_when_no_data(db):
    now = datetime.utcnow()
    for _ in range(5):
        _add_track(db, "MODEL-Y", now - timedelta(days=1),
                    trim_passes=2, error_reduction=None)

    rows = db.get_trim_difficulty_by_model(
        days_back=30, min_units=5, limit=10
    )
    by_model = {r["model"]: r for r in rows}
    assert by_model["MODEL-Y"]["avg_error_reduction"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest tests/test_trim_difficulty_avg_error_reduction.py -v`

Expected: tests fail with `KeyError: 'avg_error_reduction'` (the key doesn't exist in the existing return shape).

- [ ] **Step 3: Extend `get_trim_difficulty_by_model`**

Find the current method and add the new aggregate column to the query. The exact `old_string`:

```python
        with self.session() as session:
            cutoff = datetime.now() - timedelta(days=days_back)
            results = session.query(
                DBAnalysisResult.model,
                func.count(DBTrackResult.id).label("count"),
                func.avg(DBTrackResult.trim_pass_count).label("avg_passes"),
                func.max(DBTrackResult.trim_pass_count).label("max_passes"),
                func.sum(
                    case((DBTrackResult.trim_pass_count > 1, 1), else_=0)
                ).label("retrims"),
            ).join(DBTrackResult).filter(
                DBAnalysisResult.file_date >= cutoff,
                DBTrackResult.trim_pass_count.isnot(None),
            ).group_by(DBAnalysisResult.model).having(
                func.count(DBTrackResult.id) >= min_units
            ).order_by(desc("avg_passes")).limit(limit).all()
```

Replace with:

```python
        with self.session() as session:
            cutoff = datetime.now() - timedelta(days=days_back)
            results = session.query(
                DBAnalysisResult.model,
                func.count(DBTrackResult.id).label("count"),
                func.avg(DBTrackResult.trim_pass_count).label("avg_passes"),
                func.max(DBTrackResult.trim_pass_count).label("max_passes"),
                func.sum(
                    case((DBTrackResult.trim_pass_count > 1, 1), else_=0)
                ).label("retrims"),
                # Avg max_error_reduction_percent — distinguishes models
                # where retrimming actually helps (high reduction) from
                # those where extra passes don't improve outcomes (low
                # reduction, process root-cause issue).
                func.avg(
                    DBTrackResult.max_error_reduction_percent
                ).label("avg_error_reduction"),
            ).join(DBTrackResult).filter(
                DBAnalysisResult.file_date >= cutoff,
                DBTrackResult.trim_pass_count.isnot(None),
            ).group_by(DBAnalysisResult.model).having(
                func.count(DBTrackResult.id) >= min_units
            ).order_by(desc("avg_passes")).limit(limit).all()
```

Then update the return-list comprehension to include `avg_error_reduction`. Find:

```python
            return [
                {
                    "model": r.model,
                    "count": int(r.count or 0),
                    "avg_passes": float(r.avg_passes or 0.0),
                    "max_passes": int(r.max_passes or 0),
                    "retrim_rate": (float(r.retrims or 0) / float(r.count)) * 100.0
                    if r.count else 0.0,
                }
                for r in results
            ]
```

Replace with:

```python
            return [
                {
                    "model": r.model,
                    "count": int(r.count or 0),
                    "avg_passes": float(r.avg_passes or 0.0),
                    "max_passes": int(r.max_passes or 0),
                    "retrim_rate": (float(r.retrims or 0) / float(r.count)) * 100.0
                    if r.count else 0.0,
                    "avg_error_reduction": (
                        float(r.avg_error_reduction)
                        if r.avg_error_reduction is not None
                        else None
                    ),
                }
                for r in results
            ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest tests/test_trim_difficulty_avg_error_reduction.py -v`

Expected: both tests pass.

- [ ] **Step 5: Update the Trim Difficulty annotation**

In `src/laser_trim_analyzer/gui/pages/trends.py`, find the annotation in `_render_trim_difficulty`. The exact `old_string`:

```python
                for i, r in enumerate(rows_top_first):
                    ax.text(
                        avgs[i] + 0.05,
                        i,
                        f"{r['count']} units · max {r['max_passes']} · "
                        f"retrim {r['retrim_rate']:.0f}%",
                        va="center",
                        fontsize=8,
                        color="#cccccc",
                    )
```

Replace with:

```python
                for i, r in enumerate(rows_top_first):
                    # Append avg error-reduction when available so the
                    # operator can distinguish "many passes that work"
                    # from "many passes that don't help" at a glance.
                    aer = r.get("avg_error_reduction")
                    aer_part = f" · avg Δ {aer:.0f}%" if aer is not None else ""
                    ax.text(
                        avgs[i] + 0.05,
                        i,
                        f"{r['count']} units · max {r['max_passes']} · "
                        f"retrim {r['retrim_rate']:.0f}%{aer_part}",
                        va="center",
                        fontsize=8,
                        color="#cccccc",
                    )
```

- [ ] **Step 6: Syntax check both files**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m py_compile src/laser_trim_analyzer/database/manager.py src/laser_trim_analyzer/gui/pages/trends.py && echo OK`

Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/database/manager.py src/laser_trim_analyzer/gui/pages/trends.py tests/test_trim_difficulty_avg_error_reduction.py
git commit -m "feat(trends): trim-difficulty annotation gets avg error reduction

get_trim_difficulty_by_model now aggregates
max_error_reduction_percent per model. The Trim Difficulty bar
annotation grows from
  '{count} units · max {max} · retrim {retrim}%'
to
  '{count} units · max {max} · retrim {retrim}% · avg Δ {er}%'
when data exists.

Distinguishes models where extra trim passes are genuinely fixing
outcomes (high Δ) from models where extra passes don't help (low Δ
— process root-cause issue). Same chart, sharper diagnostic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Remove the Dashboard Pareto chart

Pure UI restructure. No tests.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/dashboard.py`

- [ ] **Step 1: Identify everything that touches the Pareto frame**

Run: `grep -n "_pareto_frame\|_pareto_placeholder\|pareto_chart\|Pareto" src/laser_trim_analyzer/gui/pages/dashboard.py`

You should see frame creation, placeholder label, chart-widget assignment, and any `_update_*` rendering. Note every line for deletion.

- [ ] **Step 2: Delete the Pareto frame block from `_create_ui`**

The exact `old_string` (verify first by re-reading the surrounding code):

```python
        # Pareto chart (column 1)
        self._pareto_frame = ctk.CTkFrame(content)
        self._pareto_frame.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")
        pareto_label = ctk.CTkLabel(
            self._pareto_frame, text="Failure Pareto",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        pareto_label.pack(padx=15, pady=(15, 5), anchor="w")
        self._pareto_placeholder = ctk.CTkLabel(
            self._pareto_frame, text="Loading...", text_color="gray"
        )
        self._pareto_placeholder.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.pareto_chart = None
```

Delete entirely. Replace with a single comment line so the diff is clear:

```python
        # Failure Pareto chart removed in this commit — the same ranking
        # information now lives in the expanded Where-to-Focus panel.
```

- [ ] **Step 3: Re-grid the Where-to-Focus panel to span columns 1+2**

Find the current Where-to-Focus grid call (column 2) and change it to span both columns. The exact `old_string`:

```python
        self.model_frame.grid(row=3, column=2, padx=10, pady=10, sticky="nsew")
```

Replace with:

```python
        # Where-to-Focus now spans columns 1+2 since the Pareto chart that
        # used to sit in column 1 has been removed (redundant ranking).
        self.model_frame.grid(row=3, column=1, columnspan=2, padx=10, pady=10, sticky="nsew")
```

- [ ] **Step 4: Remove every reference to `pareto_chart` / `_pareto_*`**

Run again: `grep -n "_pareto\|pareto_chart" src/laser_trim_analyzer/gui/pages/dashboard.py`

For each remaining hit, delete the line (or the surrounding block if it's a method that no longer has any other purpose). Common offenders: a `_render_pareto` helper, calls in `_ensure_chart_initialized`, references in `_cleanup_charts` or similar. Each must be removed cleanly.

After removal, re-run the grep. Expected: no matches.

- [ ] **Step 5: Syntax + smoke import**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m py_compile src/laser_trim_analyzer/gui/pages/dashboard.py && python3 -c "import sys; sys.path.insert(0,'src'); from laser_trim_analyzer.gui.pages.dashboard import DashboardPage; print('OK')"`

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/gui/pages/dashboard.py
git commit -m "refactor(dashboard): drop redundant Pareto chart, expand Where-to-Focus

The Failure Pareto bar chart and the Where-to-Focus card panel showed
the same model ranking. Cards already include count plus a
recommendation string; the chart added no information. Per UX audit,
remove the Pareto chart and re-grid the Where-to-Focus panel to span
columns 1+2 — gives the cards roughly twice the horizontal space they
had before.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

After all five tasks land:

- [ ] **Run the full test suite to verify no regressions**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest -q 2>&1 | tail -10`

Expected: all tests pass. The two new test files (`test_anomaly_rate_by_model.py`, `test_trim_difficulty_avg_error_reduction.py`) appear in the count.

- [ ] **Push the branch**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git push
```

PR #3 auto-updates with the five new commits.
