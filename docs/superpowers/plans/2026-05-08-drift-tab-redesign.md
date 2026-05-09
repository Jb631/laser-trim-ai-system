# Drift Tab Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Trends → Drift tab so it has a global model filter, dense sortable tables for the "All models" view in both ML Drift and Process Drift sub-tabs, and a 2×2 chart-grid investigation dashboard when a single model is selected.

**Architecture:** Database layer gains four new query methods that return everything the new UI needs in one round-trip per panel; the GUI gets a small sortable-table helper class plus a tk.Canvas sparkline helper, both lived inside `trends.py`; the single-model view reuses the existing `ChartWidget.plot_drift_chart` for sigma and a 2×2 `gridspec` for the four metric panels. The dead per-model sidebar code from the pre-refactor era is deleted.

**Tech Stack:** Python 3.14 · SQLAlchemy 2.x · CustomTkinter · Matplotlib · pytest

**Source spec:** `docs/superpowers/specs/2026-05-08-drift-ux-design.md`

---

## File map

**Modify**
- `src/laser_trim_analyzer/database/manager.py` — add 4 query methods
- `src/laser_trim_analyzer/gui/pages/trends.py` — replace Drift-tab section, delete dead code

**Create**
- `tests/test_drift_dashboard_data.py` — DB-query tests

No new source files (per project convention, helper class lives inline in `trends.py`).

---

## Task 1: DB query — `get_models_with_sigma_data`

Used to populate the global model filter dropdown. Returns models that have at least one non-null `sigma_gradient` value within the day-range window — so dead/never-processed models don't clutter the list.

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py` (add method near `get_models_list` at line 2756)
- Test: `tests/test_drift_dashboard_data.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_drift_dashboard_data.py`:

```python
"""Tests for the four DB queries that feed the redesigned Drift tab."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAnalysisResult,
    TrackResult as DBTrackResult,
    SystemType as DBSystemType,
    StatusType as DBStatusType,
    RiskCategory as DBRiskCategory,
)


@pytest.fixture
def db(tmp_path):
    """Fresh on-disk SQLite per test."""
    return DatabaseManager(tmp_path / "test.db")


def _add_analysis(db, model, file_date, sigma_values=(0.01,), serial="0001"):
    """Helper: insert one analysis row with N tracks, each with given sigma."""
    with db.session() as s:
        ar = DBAnalysisResult(
            filename=f"{model}_{serial}_{file_date.strftime('%Y%m%d')}.xls",
            file_path=f"/fake/{model}_{serial}.xls",
            file_hash=f"{model}{serial}{file_date.timestamp()}",
            model=model,
            serial=serial,
            system=DBSystemType.B,
            file_date=file_date,
            timestamp=datetime.now(),
            overall_status=DBStatusType.PASS,
            has_multi_tracks=False,
            processing_time=0.1,
        )
        s.add(ar)
        s.flush()
        for i, sigma in enumerate(sigma_values):
            tr = DBTrackResult(
                analysis_id=ar.id,
                track_id=f"TRK{i+1}",
                status=DBStatusType.PASS,
                sigma_gradient=sigma,
                sigma_threshold=0.02,
                sigma_pass=True,
                travel_length=1.0,
                linearity_spec=0.01,
                risk_category=DBRiskCategory.LOW,
            )
            s.add(tr)
        s.commit()


def test_get_models_with_sigma_data_includes_models_with_sigma():
    db = DatabaseManager(Path("/tmp/_drift_t1.db"))
    Path("/tmp/_drift_t1.db").unlink(missing_ok=True)
    db = DatabaseManager(Path("/tmp/_drift_t1.db"))

    today = datetime.now()
    _add_analysis(db, "7965", today - timedelta(days=2), sigma_values=(0.012,))
    _add_analysis(db, "8492", today - timedelta(days=5), sigma_values=(0.015,))

    models = db.get_models_with_sigma_data(days_back=30)
    assert "7965" in models
    assert "8492" in models


def test_get_models_with_sigma_data_excludes_old_models():
    db = DatabaseManager(Path("/tmp/_drift_t2.db"))
    Path("/tmp/_drift_t2.db").unlink(missing_ok=True)
    db = DatabaseManager(Path("/tmp/_drift_t2.db"))

    today = datetime.now()
    _add_analysis(db, "RECENT", today - timedelta(days=2), sigma_values=(0.01,))
    _add_analysis(db, "OLD", today - timedelta(days=400), sigma_values=(0.01,))

    models = db.get_models_with_sigma_data(days_back=30)
    assert "RECENT" in models
    assert "OLD" not in models


def test_get_models_with_sigma_data_excludes_null_sigma():
    db = DatabaseManager(Path("/tmp/_drift_t3.db"))
    Path("/tmp/_drift_t3.db").unlink(missing_ok=True)
    db = DatabaseManager(Path("/tmp/_drift_t3.db"))

    today = datetime.now()
    # Insert a row with sigma=None
    with db.session() as s:
        ar = DBAnalysisResult(
            filename="NULLSIG_0001.xls",
            file_path="/fake/NULLSIG_0001.xls",
            file_hash="nullsig",
            model="NULLSIG",
            serial="0001",
            system=DBSystemType.B,
            file_date=today - timedelta(days=2),
            timestamp=datetime.now(),
            overall_status=DBStatusType.ERROR,
            has_multi_tracks=False,
            processing_time=0.1,
        )
        s.add(ar)
        s.flush()
        tr = DBTrackResult(
            analysis_id=ar.id,
            track_id="TRK1",
            status=DBStatusType.ERROR,
            sigma_gradient=None,
            sigma_threshold=None,
            sigma_pass=False,
            travel_length=1.0,
            linearity_spec=0.01,
            risk_category=DBRiskCategory.UNKNOWN,
        )
        s.add(tr)
        s.commit()

    models = db.get_models_with_sigma_data(days_back=30)
    assert "NULLSIG" not in models
```

- [ ] **Step 2: Run test to verify it fails**

```
python3 -m pytest tests/test_drift_dashboard_data.py::test_get_models_with_sigma_data_includes_models_with_sigma -v
```
Expected: FAIL with `AttributeError: 'DatabaseManager' object has no attribute 'get_models_with_sigma_data'`.

- [ ] **Step 3: Implement the query**

Add this method to `DatabaseManager` in `manager.py`, immediately AFTER the existing `get_models_list` method (~line 2772):

```python
    def get_models_with_sigma_data(self, days_back: int = 30) -> List[str]:
        """Models with at least one non-null sigma_gradient inside the
        day-range window. Used to populate the Drift tab's model dropdown
        — never-processed and dead models don't clutter the list.
        """
        cutoff = datetime.now() - timedelta(days=days_back)
        with self.session() as session:
            rows = (
                session.query(DBAnalysisResult.model)
                .join(DBTrackResult, DBTrackResult.analysis_id == DBAnalysisResult.id)
                .filter(
                    DBAnalysisResult.model.isnot(None),
                    DBAnalysisResult.model != "Unknown",
                    DBAnalysisResult.file_date >= cutoff,
                    DBTrackResult.sigma_gradient.isnot(None),
                )
                .distinct()
                .all()
            )
            return sorted([r[0] for r in rows if r[0]], key=_model_sort_key)
```

- [ ] **Step 4: Run all 3 tests for this method**

```
python3 -m pytest tests/test_drift_dashboard_data.py -v -k "models_with_sigma"
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py tests/test_drift_dashboard_data.py
git commit -m "feat(drift): add get_models_with_sigma_data for drift filter dropdown"
```

---

## Task 2: DB query — `get_process_drift_table`

Extends the existing process-drift snapshot to also return a per-model sparkline series (recent values over time). The current `get_process_drift_by_model` returns enough for the bar chart but not the sparkline. We add a new method that returns the per-row dict the new table needs, including a `delta_pct` field and a `series` list.

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py` (add method after existing `get_process_drift_by_model`)
- Test: `tests/test_drift_dashboard_data.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_drift_dashboard_data.py`:

```python
def test_process_drift_table_returns_delta_pct_and_series():
    db_path = Path("/tmp/_drift_t4.db")
    db_path.unlink(missing_ok=True)
    db = DatabaseManager(db_path)

    today = datetime.now()
    # 7965: baseline ~3.0, recent ~3.3 → +10% drift
    for i in range(25):
        _add_analysis(
            db, "7965",
            today - timedelta(days=60 + i),
            sigma_values=(0.01,),
            serial=f"{i:04d}",
        )
        # Set untrimmed_resistance via separate update — fixture only sets sigma
    # We need untrimmed_resistance set on the track rows. Patch them in:
    with db.session() as s:
        from laser_trim_analyzer.database.models import (
            TrackResult as DBTrackResult,
            AnalysisResult as DBAnalysisResult,
        )
        rows = (
            s.query(DBTrackResult)
            .join(DBAnalysisResult)
            .filter(DBAnalysisResult.model == "7965")
            .all()
        )
        for i, tr in enumerate(rows):
            tr.untrimmed_resistance = 3.0  # baseline
        s.commit()

    # Add 6 recent samples with higher resistance
    for i in range(6):
        _add_analysis(
            db, "7965",
            today - timedelta(days=i + 1),
            sigma_values=(0.01,),
            serial=f"R{i:03d}",
        )
    with db.session() as s:
        from laser_trim_analyzer.database.models import (
            TrackResult as DBTrackResult,
            AnalysisResult as DBAnalysisResult,
        )
        rows = (
            s.query(DBTrackResult)
            .join(DBAnalysisResult)
            .filter(
                DBAnalysisResult.model == "7965",
                DBAnalysisResult.serial.like("R%"),
            )
            .all()
        )
        for tr in rows:
            tr.untrimmed_resistance = 3.3  # recent
        s.commit()

    rows = db.get_process_drift_table(
        metric="untrimmed_resistance",
        baseline_days=90,
        recent_days=14,
    )
    assert rows, "expected at least one model row"
    r = next(x for x in rows if x["model"] == "7965")
    # Δ% should be +10% (3.0 → 3.3)
    assert 8.0 < r["delta_pct"] < 12.0
    # series is a list of (date_iso, value) tuples — at least one point
    assert isinstance(r["series"], list)
    assert len(r["series"]) >= 1
    # First and second elements are date string and float
    pt = r["series"][0]
    assert isinstance(pt[0], str)
    assert isinstance(pt[1], float)
```

- [ ] **Step 2: Run test to verify it fails**

```
python3 -m pytest tests/test_drift_dashboard_data.py::test_process_drift_table_returns_delta_pct_and_series -v
```
Expected: FAIL with `AttributeError: ... 'get_process_drift_table'`.

- [ ] **Step 3: Implement the query**

Add this method to `DatabaseManager` in `manager.py`, immediately AFTER the existing `get_process_drift_by_model` method:

```python
    def get_process_drift_table(
        self,
        metric: str,
        baseline_days: int = 90,
        recent_days: int = 14,
        min_baseline_samples: int = 20,
        min_recent_samples: int = 5,
        z_threshold: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Same shape as get_process_drift_by_model, plus delta_pct and a
        per-model time series (date_iso, value) for sparkline rendering.
        """
        if metric not in self._PROCESS_DRIFT_METRICS:
            raise ValueError(
                f"Unknown drift metric {metric!r}; "
                f"choose from {list(self._PROCESS_DRIFT_METRICS)}"
            )
        if recent_days >= baseline_days:
            raise ValueError("recent_days must be less than baseline_days")

        column = getattr(DBTrackResult, metric)
        now = datetime.now()
        recent_cutoff = now - timedelta(days=recent_days)
        baseline_start = now - timedelta(days=baseline_days)

        with self.session() as session:
            rows = (
                session.query(
                    DBAnalysisResult.model,
                    column,
                    DBAnalysisResult.file_date,
                )
                .join(DBTrackResult, DBAnalysisResult.id == DBTrackResult.analysis_id)
                .filter(
                    DBAnalysisResult.model.isnot(None),
                    DBAnalysisResult.model != "Unknown",
                    DBAnalysisResult.file_date >= baseline_start,
                    column.isnot(None),
                )
                .order_by(DBAnalysisResult.file_date.asc())
                .all()
            )

        # Bucket into baseline/recent and collect a series for the sparkline.
        per_model: Dict[str, Dict[str, Any]] = {}
        for model, value, file_date in rows:
            if value is None or file_date is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            entry = per_model.setdefault(
                model, {"baseline": [], "recent": [], "series": []}
            )
            entry["series"].append((file_date.isoformat(), value))
            bucket = "recent" if file_date >= recent_cutoff else "baseline"
            entry[bucket].append(value)

        results = []
        for model, buckets in per_model.items():
            base = buckets["baseline"]
            recent = buckets["recent"]
            if len(base) < min_baseline_samples:
                continue
            if len(recent) < min_recent_samples:
                continue
            base_mean = sum(base) / len(base)
            if len(base) > 1:
                var = sum((x - base_mean) ** 2 for x in base) / (len(base) - 1)
                base_std = var ** 0.5
            else:
                base_std = 0.0
            recent_mean = sum(recent) / len(recent)
            delta = recent_mean - base_mean
            z = (delta / base_std) if base_std > 0 else 0.0
            delta_pct = (delta / base_mean * 100.0) if base_mean != 0 else 0.0
            direction = "up" if delta > 0 else ("down" if delta < 0 else "stable")
            is_drifting = abs(z) >= z_threshold and base_std > 0

            # Downsample series to ≤80 points to keep sparkline rendering cheap
            series = buckets["series"]
            if len(series) > 80:
                step = len(series) // 80
                series = series[::step]

            results.append({
                "model": model,
                "baseline_mean": base_mean,
                "baseline_std": base_std,
                "baseline_n": len(base),
                "recent_mean": recent_mean,
                "recent_n": len(recent),
                "delta": delta,
                "delta_pct": delta_pct,
                "z_score": z,
                "direction": direction,
                "is_drifting": is_drifting,
                "series": series,
            })

        results.sort(key=lambda r: abs(r["z_score"]), reverse=True)
        return results
```

- [ ] **Step 4: Run the test**

```
python3 -m pytest tests/test_drift_dashboard_data.py::test_process_drift_table_returns_delta_pct_and_series -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py tests/test_drift_dashboard_data.py
git commit -m "feat(drift): get_process_drift_table — adds delta_pct + sparkline series"
```

---

## Task 3: DB query — `get_model_drift_dashboard`

One round-trip query that returns everything the single-model dashboard needs: sigma series with control limits stub, plus the three process metric series with baseline/recent stats. Single trip avoids 4 separate DB queries for one screen.

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`
- Test: `tests/test_drift_dashboard_data.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_drift_dashboard_data.py`:

```python
def test_get_model_drift_dashboard_returns_all_panels():
    db_path = Path("/tmp/_drift_t5.db")
    db_path.unlink(missing_ok=True)
    db = DatabaseManager(db_path)

    today = datetime.now()
    for i in range(15):
        _add_analysis(
            db, "8492",
            today - timedelta(days=30 - i),
            sigma_values=(0.01 + i * 0.0001,),
            serial=f"{i:04d}",
        )

    data = db.get_model_drift_dashboard(model="8492", days_back=60)
    assert data["model"] == "8492"
    # Sigma panel
    assert "sigma_series" in data
    assert isinstance(data["sigma_series"], list)
    assert len(data["sigma_series"]) >= 10
    # Three process panels keyed by metric
    for metric in ("untrimmed_resistance",
                   "measured_electrical_angle",
                   "trim_pass_count"):
        assert metric in data["process"]
        panel = data["process"][metric]
        assert "series" in panel
        assert "baseline_mean" in panel
        assert "recent_mean" in panel


def test_get_model_drift_dashboard_returns_empty_for_unknown_model():
    db_path = Path("/tmp/_drift_t6.db")
    db_path.unlink(missing_ok=True)
    db = DatabaseManager(db_path)
    data = db.get_model_drift_dashboard(model="NOPE", days_back=60)
    assert data["model"] == "NOPE"
    assert data["sigma_series"] == []
    assert data["unit_count"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
python3 -m pytest tests/test_drift_dashboard_data.py::test_get_model_drift_dashboard_returns_all_panels -v
```
Expected: FAIL with AttributeError.

- [ ] **Step 3: Implement the query**

Add this method to `DatabaseManager` in `manager.py`, immediately after `get_process_drift_table`:

```python
    def get_model_drift_dashboard(
        self,
        model: str,
        days_back: int = 90,
        recent_days: int = 14,
    ) -> Dict[str, Any]:
        """Per-model drift dashboard data: sigma series + 3 process metric
        series with baseline/recent stats. Single round-trip per panel.
        """
        if recent_days >= days_back:
            raise ValueError("recent_days must be less than days_back")

        cutoff = datetime.now() - timedelta(days=days_back)
        recent_cutoff = datetime.now() - timedelta(days=recent_days)

        with self.session() as session:
            rows = (
                session.query(
                    DBAnalysisResult.file_date,
                    DBTrackResult.sigma_gradient,
                    DBTrackResult.untrimmed_resistance,
                    DBTrackResult.measured_electrical_angle,
                    DBTrackResult.trim_pass_count,
                )
                .join(DBTrackResult, DBTrackResult.analysis_id == DBAnalysisResult.id)
                .filter(
                    DBAnalysisResult.model == model,
                    DBAnalysisResult.file_date >= cutoff,
                )
                .order_by(DBAnalysisResult.file_date.asc())
                .all()
            )

        sigma_series: List[Tuple[str, float]] = []
        process_series: Dict[str, List[Tuple[str, float]]] = {
            "untrimmed_resistance": [],
            "measured_electrical_angle": [],
            "trim_pass_count": [],
        }
        process_baseline: Dict[str, List[float]] = {k: [] for k in process_series}
        process_recent: Dict[str, List[float]] = {k: [] for k in process_series}

        for file_date, sigma, ur, mea, tpc in rows:
            if file_date is None:
                continue
            iso = file_date.isoformat()
            if sigma is not None:
                sigma_series.append((iso, float(sigma)))
            for metric, value in (
                ("untrimmed_resistance", ur),
                ("measured_electrical_angle", mea),
                ("trim_pass_count", tpc),
            ):
                if value is None:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                process_series[metric].append((iso, value))
                if file_date >= recent_cutoff:
                    process_recent[metric].append(value)
                else:
                    process_baseline[metric].append(value)

        process: Dict[str, Dict[str, Any]] = {}
        for metric in process_series:
            base = process_baseline[metric]
            recent = process_recent[metric]
            base_mean = (sum(base) / len(base)) if base else None
            recent_mean = (sum(recent) / len(recent)) if recent else None
            if base and len(base) > 1:
                var = sum((x - base_mean) ** 2 for x in base) / (len(base) - 1)
                base_std = var ** 0.5
            else:
                base_std = 0.0
            delta = (
                recent_mean - base_mean
                if base_mean is not None and recent_mean is not None
                else None
            )
            z = (delta / base_std) if (delta is not None and base_std > 0) else None
            delta_pct = (
                (delta / base_mean * 100.0)
                if delta is not None and base_mean and base_mean != 0
                else None
            )
            process[metric] = {
                "series": process_series[metric],
                "baseline_mean": base_mean,
                "baseline_n": len(base),
                "recent_mean": recent_mean,
                "recent_n": len(recent),
                "delta": delta,
                "delta_pct": delta_pct,
                "z_score": z,
                "is_drifting": z is not None and abs(z) >= 2.0,
            }

        return {
            "model": model,
            "unit_count": len(rows),
            "sigma_series": sigma_series,
            "process": process,
        }
```

Note: `Tuple` should already be imported at the top of `manager.py` from `typing`. If not, add it.

- [ ] **Step 4: Run the tests**

```
python3 -m pytest tests/test_drift_dashboard_data.py::test_get_model_drift_dashboard_returns_all_panels tests/test_drift_dashboard_data.py::test_get_model_drift_dashboard_returns_empty_for_unknown_model -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py tests/test_drift_dashboard_data.py
git commit -m "feat(drift): get_model_drift_dashboard for single-model investigation view"
```

---

## Task 4: DB helper — `get_drift_state_for_models`

Returns the static-DB portion of each model's drift state (drift_start_date, is_drifting, direction) plus a sigma trend series — the things that come from the DB. The CUSUM score itself is in-memory on `MLManager.drift_detectors`, so the caller (the GUI) joins that side at render time.

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`
- Test: `tests/test_drift_dashboard_data.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_drift_dashboard_data.py`:

```python
def test_get_drift_state_for_models_pulls_drift_start_date():
    db_path = Path("/tmp/_drift_t7.db")
    db_path.unlink(missing_ok=True)
    db = DatabaseManager(db_path)

    today = datetime.now()
    _add_analysis(db, "7965", today - timedelta(days=2), sigma_values=(0.012,))

    # Insert ModelMLState row marking 7965 as drifting
    from laser_trim_analyzer.database.models import ModelMLState
    with db.session() as s:
        s.add(ModelMLState(
            model="7965",
            is_drifting=True,
            drift_direction="up",
            drift_start_date=today - timedelta(days=10),
            updated_date=today - timedelta(days=1),
        ))
        s.commit()

    data = db.get_drift_state_for_models(days_back=30)
    assert "7965" in data
    row = data["7965"]
    assert row["is_drifting"] is True
    assert row["direction"] == "up"
    assert row["drift_start_date"] is not None
    # Sigma trend present
    assert isinstance(row["sigma_series"], list)
    assert len(row["sigma_series"]) >= 1


def test_get_drift_state_for_models_includes_stable_models():
    db_path = Path("/tmp/_drift_t8.db")
    db_path.unlink(missing_ok=True)
    db = DatabaseManager(db_path)

    today = datetime.now()
    _add_analysis(db, "8275", today - timedelta(days=1), sigma_values=(0.01,))
    # No ModelMLState row → treated as no-baseline / stable.

    data = db.get_drift_state_for_models(days_back=30)
    assert "8275" in data
    row = data["8275"]
    assert row["is_drifting"] is False
    assert row["drift_start_date"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```
python3 -m pytest tests/test_drift_dashboard_data.py -k "drift_state_for_models" -v
```
Expected: FAIL with AttributeError.

- [ ] **Step 3: Implement the method**

Add to `DatabaseManager` in `manager.py`, near the other drift queries:

```python
    def get_drift_state_for_models(
        self,
        days_back: int = 30,
        max_series_points: int = 60,
    ) -> Dict[str, Dict[str, Any]]:
        """Per-model drift state from the DB only.

        For each model with sigma data in the window, returns:
            is_drifting, direction, drift_start_date,
            sigma_series (list of (iso_date, sigma) tuples).

        The CUSUM score and threshold live on the in-memory DriftDetector;
        the caller is expected to join those in at render time.
        """
        from laser_trim_analyzer.database.models import ModelMLState

        cutoff = datetime.now() - timedelta(days=days_back)
        with self.session() as session:
            # Pull sigma series for every model in the window
            rows = (
                session.query(
                    DBAnalysisResult.model,
                    DBAnalysisResult.file_date,
                    DBTrackResult.sigma_gradient,
                )
                .join(DBTrackResult, DBTrackResult.analysis_id == DBAnalysisResult.id)
                .filter(
                    DBAnalysisResult.model.isnot(None),
                    DBAnalysisResult.model != "Unknown",
                    DBAnalysisResult.file_date >= cutoff,
                    DBTrackResult.sigma_gradient.isnot(None),
                )
                .order_by(DBAnalysisResult.file_date.asc())
                .all()
            )
            # Pull drift state rows
            ml_states = {
                s.model: s
                for s in session.query(ModelMLState).all()
            }

        per_model: Dict[str, Dict[str, Any]] = {}
        for model, file_date, sigma in rows:
            if not model or file_date is None or sigma is None:
                continue
            entry = per_model.setdefault(
                model, {"sigma_series": []}
            )
            entry["sigma_series"].append((file_date.isoformat(), float(sigma)))

        result: Dict[str, Dict[str, Any]] = {}
        for model, entry in per_model.items():
            series = entry["sigma_series"]
            # Downsample to keep sparkline rendering cheap
            if len(series) > max_series_points:
                step = len(series) // max_series_points
                series = series[::step]
            state = ml_states.get(model)
            result[model] = {
                "model": model,
                "is_drifting": bool(state.is_drifting) if state else False,
                "direction": state.drift_direction if state else None,
                "drift_start_date": state.drift_start_date if state else None,
                "sigma_series": series,
            }
        return result
```

- [ ] **Step 4: Run all DB tests**

```
python3 -m pytest tests/test_drift_dashboard_data.py -v
```
Expected: all tests pass (8 if you wrote all of them).

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py tests/test_drift_dashboard_data.py
git commit -m "feat(drift): get_drift_state_for_models — DB side of ML drift table"
```

---

## Task 5: Sortable table + sparkline helpers (in `trends.py`)

Two small reusable helpers that the new tables need:
1. **Sparkline:** a `tk.Canvas` widget showing a tiny line chart inside a row.
2. **SortableTable:** a CTk-based grid with click-to-sort headers and click-to-select rows.

Both live as private classes inside `trends.py` to avoid adding files.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py` (add classes near the top, after the existing module-level helpers)

- [ ] **Step 1: Add the Sparkline widget**

Find the top of the `TrendsPage` class definition in `trends.py` (search `class TrendsPage`). Just BEFORE that class, add:

```python
class _Sparkline(tk.Canvas):
    """Tiny inline line chart drawn on a tk.Canvas. Used inside table rows.

    Pure tkinter (no matplotlib) so per-row rendering is cheap.
    """

    def __init__(self, parent, width=80, height=14, **kwargs):
        super().__init__(
            parent, width=width, height=height,
            highlightthickness=0, bd=0, **kwargs,
        )
        self._w = width
        self._h = height

    def draw(self, values: list, color: str = "#7ed99e") -> None:
        self.delete("all")
        if not values or len(values) < 2:
            return
        v_min = min(values)
        v_max = max(values)
        span = v_max - v_min if v_max != v_min else 1.0
        n = len(values)
        # Map values to canvas coordinates with 1-px top/bottom margin
        h = self._h - 2
        pts = []
        for i, v in enumerate(values):
            x = i * (self._w - 1) / (n - 1)
            # invert so higher values plot toward the top
            y = 1 + h - (v - v_min) / span * h
            pts.extend((x, y))
        self.create_line(*pts, fill=color, width=1.2, smooth=False)
```

You'll need `import tkinter as tk` at the top of `trends.py` if not already imported. Search for `import tkinter` — if absent, add it under the customtkinter import.

- [ ] **Step 2: Add the SortableTable widget**

Immediately AFTER the `_Sparkline` class:

```python
class _SortableTable(ctk.CTkScrollableFrame):
    """Grid layout with click-to-sort headers and click-to-select rows.

    Columns is a list of (key, label, render) tuples:
      - key: string used for sort ordering (rows are dicts keyed by this)
      - label: header text
      - render: callable(parent, row_dict) -> tk widget (or None to render
                row_dict[key] as plain text via a CTkLabel).

    Rows are dicts. Pass row_click=fn to be notified on row selection;
    the callback receives the row dict.
    """

    def __init__(
        self,
        parent,
        columns,
        rows,
        row_click=None,
        default_sort_key=None,
        default_sort_reverse=False,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self._columns = columns
        self._rows = list(rows)
        self._row_click = row_click
        self._sort_key = default_sort_key
        self._sort_reverse = default_sort_reverse
        self._build()

    def _build(self):
        for w in self.winfo_children():
            w.destroy()
        # Header row
        for col_idx, (key, label, _) in enumerate(self._columns):
            arrow = ""
            if key == self._sort_key:
                arrow = " ↓" if self._sort_reverse else " ↑"
            btn = ctk.CTkButton(
                self,
                text=f"{label}{arrow}",
                anchor="w",
                fg_color="transparent",
                hover_color=("gray85", "gray25"),
                text_color=("gray20", "gray80"),
                font=ctk.CTkFont(size=10, weight="bold"),
                height=22,
                command=lambda k=key: self._on_sort(k),
            )
            btn.grid(row=0, column=col_idx, sticky="ew", padx=2, pady=(2, 4))

        # Data rows
        sorted_rows = self._sorted_rows()
        for row_idx, row in enumerate(sorted_rows, start=1):
            for col_idx, (key, _, render) in enumerate(self._columns):
                if render is None:
                    val = row.get(key)
                    text = "" if val is None else str(val)
                    cell = ctk.CTkLabel(
                        self, text=text, anchor="w",
                        font=ctk.CTkFont(size=10),
                    )
                else:
                    cell = render(self, row)
                cell.grid(row=row_idx, column=col_idx, sticky="ew", padx=4, pady=1)
                if self._row_click is not None:
                    cell.bind(
                        "<Button-1>",
                        lambda _e, r=row: self._row_click(r),
                    )

        # Stretch all columns evenly
        for col_idx in range(len(self._columns)):
            self.grid_columnconfigure(col_idx, weight=1)

    def _sorted_rows(self):
        if self._sort_key is None:
            return self._rows

        def keyfn(r):
            v = r.get(self._sort_key)
            if v is None:
                # Push None to the end regardless of direction
                return (1, 0)
            if isinstance(v, (int, float)):
                return (0, v)
            return (0, str(v))

        return sorted(self._rows, key=keyfn, reverse=self._sort_reverse)

    def _on_sort(self, key):
        if self._sort_key == key:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_key = key
            self._sort_reverse = False
        self._build()

    def update_rows(self, rows):
        """Replace the row data and re-render."""
        self._rows = list(rows)
        self._build()
```

- [ ] **Step 3: Smoke-test the helpers**

Create a one-off script `/tmp/smoke_drift_helpers.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "src"))
sys.path.insert(0, "/Users/jb631/projects/laser-trim-ai-system-v5/src")

import customtkinter as ctk
from laser_trim_analyzer.gui.pages.trends import _Sparkline, _SortableTable

root = ctk.CTk()
root.geometry("600x300")

sp = _Sparkline(root, width=80, height=14)
sp.pack()
sp.draw([1, 3, 2, 5, 4, 6, 7], color="#ff8080")

cols = [
    ("model", "Model", None),
    ("z", "z-score", None),
]
rows = [{"model": "7965", "z": 4.8}, {"model": "8492", "z": -2.1}]
tbl = _SortableTable(root, cols, rows, default_sort_key="z", default_sort_reverse=True)
tbl.pack(fill="both", expand=True)

root.after(2000, root.destroy)
root.mainloop()
print("OK — helpers render without exceptions.")
```

Run:

```
python3 /tmp/smoke_drift_helpers.py
```
Expected: window opens, closes after 2 seconds, prints "OK".

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/trends.py
git commit -m "feat(trends): _Sparkline + _SortableTable helpers for drift redesign"
```

---

## Task 6: Header row — global model filter dropdown

Add a model selector to the Drift tab header. When changed, it stores the chosen model on `self._drift_filter_model` and triggers a re-render. "All models" maps to the existing behavior.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py`

- [ ] **Step 1: Add filter state attribute**

Locate the section in `TrendsPage.__init__` where `self._drift_subtab` is initialized (around line 94, just after `self._drift_subtab: str = "ML Drift"`). Add immediately after:

```python
        # Global model filter for the Drift tab. None / "All models" → all-models view.
        # Set when the user picks a model from the new dropdown in _create_drift_view.
        self._drift_filter_model: Optional[str] = None
```

- [ ] **Step 2: Modify `_create_drift_view`**

Locate `_create_drift_view` (around line 2379). Replace its body — keeping the signature — with:

```python
    def _create_drift_view(self) -> "ChartWidget":
        """Build the Drift tab: header row (model filter + ML/Process toggle)
        on top, chart frame below. Returns the chart widget for All-models
        sub-views; the single-model view replaces the chart frame contents.
        """
        for widget in self.content.winfo_children():
            widget.destroy()
        self._cleanup_charts()

        self.content.grid_rowconfigure(0, weight=0)
        self.content.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.content, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        # Model filter dropdown
        ctk.CTkLabel(
            header, text="Model:",
            font=ctk.CTkFont(size=11),
        ).pack(side="left", padx=(0, 4))
        try:
            db = get_database()
            model_options = ["All models"] + db.get_models_with_sigma_data(
                days_back=self.selected_days
            )
        except Exception as e:
            logger.debug(f"Could not populate drift model filter: {e}")
            model_options = ["All models"]
        self._drift_model_filter = ctk.CTkComboBox(
            header,
            values=model_options,
            command=self._on_drift_model_filter_changed,
            width=160,
        )
        self._drift_model_filter.set(self._drift_filter_model or "All models")
        self._drift_model_filter.pack(side="left", padx=(0, 14))

        # ML/Process sub-tab toggle — hidden when a specific model is selected.
        self._drift_subtab_button = ctk.CTkSegmentedButton(
            header,
            values=["ML Drift", "Process Drift"],
            command=self._on_drift_subtab_changed,
        )
        self._drift_subtab_button.set(self._drift_subtab)
        if self._drift_filter_model:
            self._drift_subtab_button.pack_forget()
        else:
            self._drift_subtab_button.pack(side="left")

        chart_frame = ctk.CTkFrame(self.content)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self._drift_content_frame = chart_frame

        ChartWidget, ChartStyle = _ensure_chart_module()
        chart = ChartWidget(
            chart_frame,
            style=ChartStyle(figure_size=(12, 6), dpi=100),
        )
        chart.pack(fill="both", expand=True, padx=15, pady=15)
        self._chart_widgets.append(chart)
        return chart
```

- [ ] **Step 3: Add the filter-change handler**

Immediately after `_on_drift_subtab_changed`, add:

```python
    def _on_drift_model_filter_changed(self, value: str):
        """Switch between All-models view and single-model dashboard."""
        new = None if value == "All models" else value
        if new == self._drift_filter_model:
            return
        self._drift_filter_model = new
        self._load_generation += 1
        # Render: pickup whichever sub-view is current (ignored when single
        # model; the single-model dashboard is rendered directly)
        if new is None:
            if self._drift_subtab == "ML Drift":
                self._show_drift_timeline()
            else:
                self._show_process_drift()
        else:
            self._show_single_model_drift()
```

`_show_single_model_drift` is implemented in Task 9. For now, add a stub:

```python
    def _show_single_model_drift(self):
        """Single-model investigation dashboard. Implemented in Task 9."""
        if not self._drift_filter_model:
            return
        # Stub — overwritten in Task 9
        chart = self._create_drift_view()
        self._draw_empty_state(
            chart.figure.add_subplot(111),
            f"Single-model view for {self._drift_filter_model} — coming next",
        )
        chart.canvas.draw_idle()
```

- [ ] **Step 4: Manual smoke-test**

Launch the app:
```
python3 src/__main__.py
```
Navigate Trends → Drift. Verify:
1. A "Model:" dropdown appears in the Drift header row.
2. "All models" is the default and the existing ML Drift / Process Drift behavior still works.
3. Selecting a specific model shows the stub message.
4. Selecting "All models" again returns to the existing view.

Close the app.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/trends.py
git commit -m "feat(trends): drift tab — global model filter dropdown + scaffolding"
```

---

## Task 7: ML Drift "All models" sortable table

Replaces the current sparse-timeline render with a sortable table of one row per model. Combines `db.get_drift_state_for_models()` with `ml_manager.drift_detectors` to get the CUSUM score.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py`

- [ ] **Step 1: Replace `_show_drift_timeline` and `_render_drift_timeline`**

Locate `_show_drift_timeline` (around line 2587). Replace BOTH `_show_drift_timeline` and `_render_drift_timeline` with:

```python
    def _show_drift_timeline(self):
        """Show ML Drift sortable model table (All-models view)."""
        if self._drift_filter_model is not None:
            # Single-model view takes over; don't render the table.
            return
        self.status_label.configure(text="Loading drift table...")
        selected_days = self.selected_days

        def _load():
            try:
                from laser_trim_analyzer.ml import get_shared_ml_manager
                db = get_database()
                state_by_model = db.get_drift_state_for_models(
                    days_back=selected_days
                )
                ml_manager = get_shared_ml_manager(db)
                # Combine DB state + in-memory CUSUM score per detector
                rows = []
                for model, state in state_by_model.items():
                    detector = ml_manager.drift_detectors.get(model)
                    has_baseline = detector is not None and detector.has_baseline
                    if detector is not None:
                        cusum_value = max(detector.cusum_pos, detector.cusum_neg)
                        cusum_h = detector.cusum_h
                    else:
                        cusum_value = None
                        cusum_h = None
                    days_drifting = None
                    if state["is_drifting"] and state["drift_start_date"]:
                        days_drifting = (
                            datetime.now() - state["drift_start_date"]
                        ).days
                    rows.append({
                        "model": model,
                        "has_baseline": has_baseline,
                        "is_drifting": state["is_drifting"],
                        "direction": state["direction"],
                        "drift_score": cusum_value,
                        "drift_threshold": cusum_h,
                        "drift_start_date": state["drift_start_date"],
                        "days_drifting": days_drifting,
                        "sigma_series": state["sigma_series"],
                    })
                self.after(0, lambda: self._render_drift_timeline(rows))
            except Exception as e:
                logger.error(f"Drift table error: {e}", exc_info=True)
                self.after(0, lambda: self.status_label.configure(
                    text=f"Drift error: {e}"))

        get_thread_manager().start_thread(target=_load, name="drift-table")

    def _render_drift_timeline(self, rows):
        """Render the ML Drift sortable model table on the main thread."""
        if not self.winfo_exists():
            return
        if (
            self._trend_type.get() != "Drift"
            or self._drift_subtab != "ML Drift"
            or self._drift_filter_model is not None
        ):
            return
        try:
            self._create_drift_view()  # rebuild header + content frame
            for w in self._drift_content_frame.winfo_children():
                w.destroy()

            if not rows:
                lbl = ctk.CTkLabel(
                    self._drift_content_frame,
                    text="No drift data yet — process more files or train ML in Settings.",
                    font=ctk.CTkFont(size=11),
                    text_color="gray",
                )
                lbl.pack(expand=True, padx=20, pady=20)
                self.status_label.configure(text="No drift data")
                return

            def render_status(parent, row):
                if not row["has_baseline"]:
                    return ctk.CTkLabel(
                        parent, text="○ no baseline",
                        text_color="gray",
                        font=ctk.CTkFont(size=10),
                    )
                if row["is_drifting"]:
                    if row["direction"] == "up":
                        return ctk.CTkLabel(
                            parent, text="↑ DRIFTING",
                            text_color="#ff8080",
                            font=ctk.CTkFont(size=10, weight="bold"),
                        )
                    return ctk.CTkLabel(
                        parent, text="↓ DRIFTING",
                        text_color="#ffb060",
                        font=ctk.CTkFont(size=10, weight="bold"),
                    )
                return ctk.CTkLabel(
                    parent, text="✓ stable",
                    text_color="#7ed99e",
                    font=ctk.CTkFont(size=10),
                )

            def render_score(parent, row):
                if row["drift_score"] is None:
                    text = "—"
                    color = "gray"
                else:
                    text = f"{row['drift_score']:.1f} / {row['drift_threshold']:.1f}"
                    color = (
                        "#ff8080" if row["is_drifting"] and row["direction"] == "up"
                        else "#ffb060" if row["is_drifting"]
                        else "#7ed99e"
                    )
                return ctk.CTkLabel(
                    parent, text=text, text_color=color,
                    font=ctk.CTkFont(size=10),
                )

            def render_last_event(parent, row):
                d = row["drift_start_date"]
                text = d.strftime("%Y-%m-%d") if (d and row["is_drifting"]) else "—"
                return ctk.CTkLabel(
                    parent, text=text,
                    font=ctk.CTkFont(size=10),
                )

            def render_days(parent, row):
                d = row["days_drifting"]
                text = f"{d}d" if d is not None else "—"
                return ctk.CTkLabel(
                    parent, text=text,
                    font=ctk.CTkFont(size=10),
                )

            def render_spark(parent, row):
                s = _Sparkline(parent, width=80, height=14)
                values = [v for _, v in row["sigma_series"]]
                color = (
                    "#ff8080" if row["is_drifting"] and row["direction"] == "up"
                    else "#ffb060" if row["is_drifting"]
                    else "#7ed99e"
                )
                s.draw(values, color=color)
                return s

            columns = [
                ("model", "Model", None),
                ("status_sort", "Status", render_status),
                ("score_sort", "Drift score", render_score),
                ("drift_start_date", "Last event", render_last_event),
                ("days_drifting", "Days drifting", render_days),
                ("model", "Sigma trend", render_spark),
            ]
            # Add sort-key fields
            for r in rows:
                # Drifting first (0), stable (1), no-baseline (2)
                if not r["has_baseline"]:
                    r["status_sort"] = 2
                elif r["is_drifting"]:
                    r["status_sort"] = 0
                else:
                    r["status_sort"] = 1
                r["score_sort"] = (
                    r["drift_score"] if r["drift_score"] is not None else -1
                )

            table = _SortableTable(
                self._drift_content_frame,
                columns=columns,
                rows=rows,
                row_click=lambda r: self._on_drift_row_click(r["model"]),
                default_sort_key="status_sort",
                default_sort_reverse=False,
            )
            table.pack(fill="both", expand=True, padx=10, pady=10)
            drifting_count = sum(1 for r in rows if r["is_drifting"])
            self.status_label.configure(
                text=f"Drift: {drifting_count} drifting / {len(rows)} models"
            )
        except Exception as e:
            logger.error(f"Drift table render error: {e}", exc_info=True)
            self.status_label.configure(text=f"Drift render error: {e}")

    def _on_drift_row_click(self, model: str):
        """Drill into a model's drift dashboard from a table row click."""
        self._drift_filter_model = model
        if hasattr(self, "_drift_model_filter"):
            self._drift_model_filter.set(model)
        self._show_single_model_drift()
```

- [ ] **Step 2: Manual smoke-test**

Launch:
```
python3 src/__main__.py
```
Navigate Trends → Drift → ML Drift. Verify:
1. A sortable table appears with rows for models, columns Model | Status | Drift score | Last event | Days drifting | Sigma trend.
2. Drifting models appear at the top by default.
3. Clicking a column header sorts by that column.
4. Clicking a row sets the model filter to that model and shows the Task 6 stub.

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/trends.py
git commit -m "feat(trends): ML Drift all-models — sortable table replaces sparse timeline"
```

---

## Task 8: Process Drift "All models" — metric-tabs + sortable table

Replaces the 3-stacked-panels layout with a metric tab strip (Untrimmed R / Elec Angle / Trim Passes) plus the sortable table for the active metric.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py`

- [ ] **Step 1: Add active-metric state**

In `__init__`, after the `_drift_filter_model` attribute added in Task 6:

```python
        # Which physical metric is shown on the Process Drift sub-tab.
        self._process_drift_metric: str = "untrimmed_resistance"
```

- [ ] **Step 2: Replace `_show_process_drift` and `_render_process_drift`**

Locate `_show_process_drift` (around line 2719). Replace both that and `_render_process_drift` with:

```python
    def _show_process_drift(self):
        """Show Process Drift table for the active metric (All-models view)."""
        if self._drift_filter_model is not None:
            return
        self.status_label.configure(text="Loading process drift...")
        selected_days = self.selected_days
        baseline_days = max(selected_days, 60)
        recent_days = max(7, min(28, baseline_days // 7))
        metric = self._process_drift_metric

        def _load():
            try:
                db = get_database()
                rows = db.get_process_drift_table(
                    metric=metric,
                    baseline_days=baseline_days,
                    recent_days=recent_days,
                )
                self.after(0, lambda: self._render_process_drift_table(rows))
            except Exception as e:
                logger.error(f"Process drift error: {e}", exc_info=True)
                self.after(0, lambda exc=e: self.status_label.configure(
                    text=f"Process drift error: {exc}"))

        get_thread_manager().start_thread(target=_load, name="process-drift-table")

    def _render_process_drift_table(self, rows):
        if not self.winfo_exists():
            return
        if (
            self._trend_type.get() != "Drift"
            or self._drift_subtab != "Process Drift"
            or self._drift_filter_model is not None
        ):
            return
        try:
            self._create_drift_view()
            for w in self._drift_content_frame.winfo_children():
                w.destroy()

            # Metric tab strip
            tabs = ctk.CTkFrame(self._drift_content_frame, fg_color="transparent")
            tabs.pack(fill="x", padx=10, pady=(8, 0))
            metric_options = [
                ("untrimmed_resistance", "Untrimmed Resistance"),
                ("measured_electrical_angle", "Electrical Angle"),
                ("trim_pass_count", "Trim Passes"),
            ]
            for mkey, mlabel in metric_options:
                is_active = mkey == self._process_drift_metric
                btn = ctk.CTkButton(
                    tabs,
                    text=mlabel,
                    width=130,
                    height=24,
                    fg_color=("#0d6efd" if is_active else ("gray85", "gray25")),
                    hover_color=("#3b8eff" if is_active else ("gray75", "gray35")),
                    text_color=("white" if is_active else None),
                    font=ctk.CTkFont(size=10, weight=("bold" if is_active else "normal")),
                    command=lambda k=mkey: self._on_process_metric_changed(k),
                )
                btn.pack(side="left", padx=(0, 4))

            if not rows:
                lbl = ctk.CTkLabel(
                    self._drift_content_frame,
                    text="No models meet the baseline+recent thresholds for this metric.",
                    font=ctk.CTkFont(size=11),
                    text_color="gray",
                )
                lbl.pack(expand=True, padx=20, pady=20)
                self.status_label.configure(text="Process drift: no rows")
                return

            meta = self._db_metric_meta().get(self._process_drift_metric, {})
            unit = meta.get("unit", "")
            fmt = meta.get("fmt", "{:.2f}")

            def render_baseline(parent, row):
                txt = f"{fmt.format(row['baseline_mean'])} {unit}".strip()
                return ctk.CTkLabel(parent, text=txt, font=ctk.CTkFont(size=10))

            def render_recent(parent, row):
                txt = f"{fmt.format(row['recent_mean'])} {unit}".strip()
                return ctk.CTkLabel(parent, text=txt, font=ctk.CTkFont(size=10))

            def render_delta_pct(parent, row):
                v = row["delta_pct"]
                color = (
                    "#ff8080" if abs(row["z_score"]) >= 3.0
                    else "#ffb060" if abs(row["z_score"]) >= 2.0
                    else "#7ed99e"
                )
                return ctk.CTkLabel(
                    parent, text=f"{v:+.1f}%",
                    text_color=color,
                    font=ctk.CTkFont(size=10),
                )

            def render_z(parent, row):
                z = row["z_score"]
                color = (
                    "#ff8080" if abs(z) >= 3.0
                    else "#ffb060" if abs(z) >= 2.0
                    else "#7ed99e"
                )
                return ctk.CTkLabel(
                    parent, text=f"{z:+.1f}",
                    text_color=color,
                    font=ctk.CTkFont(size=10),
                )

            def render_trend(parent, row):
                s = _Sparkline(parent, width=80, height=14)
                values = [v for _, v in row["series"]]
                color = (
                    "#ff8080" if abs(row["z_score"]) >= 3.0
                    else "#ffb060" if abs(row["z_score"]) >= 2.0
                    else "#7ed99e"
                )
                s.draw(values, color=color)
                return s

            columns = [
                ("model", "Model", None),
                ("baseline_mean", "Baseline", render_baseline),
                ("recent_mean", "Recent", render_recent),
                ("delta_pct", "Δ%", render_delta_pct),
                ("z_score", "z", render_z),
                ("model", "Trend", render_trend),
            ]
            # Sort by |z| desc — but SortableTable sorts on raw values, so
            # add an absolute-z field for default sort.
            for r in rows:
                r["abs_z"] = abs(r["z_score"])
            columns.insert(0, ("abs_z", "", None))  # hidden helper column
            # Hide first column visually (zero width). Easiest: just don't
            # add it. Instead use 'z_score' as default sort and reverse for
            # |z| via custom keyfn — but SortableTable doesn't support keyfn.
            # Simplest: drop the helper col and sort by abs_z by setting
            # default_sort_key="abs_z" but include it in columns means it
            # renders. Compromise: leave it visible as a small "|z|" column.
            columns[0] = ("abs_z", "|z|", None)

            table = _SortableTable(
                self._drift_content_frame,
                columns=columns,
                rows=rows,
                row_click=lambda r: self._on_drift_row_click(r["model"]),
                default_sort_key="abs_z",
                default_sort_reverse=True,
            )
            table.pack(fill="both", expand=True, padx=10, pady=10)
            drifting_count = sum(1 for r in rows if r["is_drifting"])
            self.status_label.configure(
                text=f"Process drift ({self._process_drift_metric}): "
                     f"{drifting_count} drifting / {len(rows)} models"
            )
        except Exception as e:
            logger.error(f"Process drift render error: {e}", exc_info=True)
            self.status_label.configure(text=f"Process drift render error: {e}")

    def _on_process_metric_changed(self, metric: str):
        if metric == self._process_drift_metric:
            return
        self._process_drift_metric = metric
        self._show_process_drift()
```

- [ ] **Step 3: Manual smoke-test**

Launch the app, go Trends → Drift → Process Drift. Verify:
1. A 3-button metric strip appears with Untrimmed Resistance highlighted.
2. The table below has columns |z| | Model | Baseline | Recent | Δ% | z | Trend.
3. Clicking another metric button reloads the table for that metric.
4. Clicking a row drills into the single-model stub.

- [ ] **Step 4: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/trends.py
git commit -m "feat(trends): Process Drift all-models — metric tabs + sortable table"
```

---

## Task 9: Single-model dashboard

Replaces the Task 6 stub with the real 2×2 chart grid and header pill bar.

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py`

- [ ] **Step 1: Replace `_show_single_model_drift`**

Find the stub `_show_single_model_drift` added in Task 6. Replace it with:

```python
    def _show_single_model_drift(self):
        """Single-model investigation dashboard.

        Layout:
          Header pills (model · status · days drifting · score · units)
          2×2 chart grid:
            [ Sigma Drift           | Untrimmed Resistance ]
            [ Electrical Angle      | Trim Passes          ]
        """
        if not self._drift_filter_model:
            return
        model = self._drift_filter_model
        self.status_label.configure(text=f"Loading drift dashboard for {model}...")
        selected_days = self.selected_days

        def _load():
            try:
                from laser_trim_analyzer.ml import get_shared_ml_manager
                db = get_database()
                data = db.get_model_drift_dashboard(
                    model=model, days_back=selected_days
                )
                ml_manager = get_shared_ml_manager(db)
                detector = ml_manager.drift_detectors.get(model)
                self.after(0, lambda: self._render_single_model_drift(
                    data, detector
                ))
            except Exception as e:
                logger.error(f"Single-model drift error: {e}", exc_info=True)
                self.after(0, lambda exc=e: self.status_label.configure(
                    text=f"Drift dashboard error: {exc}"))

        get_thread_manager().start_thread(
            target=_load, name="single-model-drift",
        )

    def _render_single_model_drift(self, data: Dict[str, Any], detector):
        if not self.winfo_exists():
            return
        if self._drift_filter_model != data.get("model"):
            return
        try:
            self._create_drift_view()
            for w in self._drift_content_frame.winfo_children():
                w.destroy()

            model = data["model"]
            unit_count = data["unit_count"]
            process = data["process"]

            # Header pill bar
            pills = ctk.CTkFrame(self._drift_content_frame, fg_color="transparent")
            pills.pack(fill="x", padx=10, pady=(8, 4))

            # Model name big
            ctk.CTkLabel(
                pills, text=model,
                font=ctk.CTkFont(size=18, weight="bold"),
            ).pack(side="left", padx=(0, 12))

            # Status badge
            is_drifting = bool(detector and detector.is_drifting)
            direction = (
                detector.drift_direction.value
                if detector and detector.drift_direction
                else None
            )
            if not detector or not detector.has_baseline:
                badge_text, badge_color = "○ no baseline", "gray"
            elif is_drifting and direction == "up":
                badge_text, badge_color = "↑ DRIFTING", "#ff8080"
            elif is_drifting:
                badge_text, badge_color = "↓ DRIFTING", "#ffb060"
            else:
                badge_text, badge_color = "✓ stable", "#7ed99e"
            ctk.CTkLabel(
                pills, text=badge_text, text_color=badge_color,
                font=ctk.CTkFont(size=11, weight="bold"),
            ).pack(side="left", padx=(0, 12))

            # Days drifting / score / units pills
            def _pill(text):
                f = ctk.CTkFrame(pills, fg_color=("gray85", "gray25"), corner_radius=4)
                f.pack(side="left", padx=4)
                ctk.CTkLabel(
                    f, text=text, font=ctk.CTkFont(size=10),
                ).pack(padx=8, pady=2)

            if detector and detector.has_baseline:
                cusum_value = max(detector.cusum_pos, detector.cusum_neg)
                _pill(f"score {cusum_value:.1f} / {detector.cusum_h:.1f}")
            _pill(f"{unit_count:,} units")

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

            # ---- Top-left: Sigma drift with control limits ----
            ax_sigma = fig.add_subplot(gs[0, 0])
            chart._style_axis(ax_sigma)
            sigma_pts = data["sigma_series"]
            if not sigma_pts:
                self._draw_empty_state(
                    ax_sigma,
                    "No sigma data in window",
                )
            else:
                from datetime import datetime as _dt
                dates = [_dt.fromisoformat(d) for d, _ in sigma_pts]
                values = [v for _, v in sigma_pts]
                ax_sigma.plot(dates, values, color="#0d6efd", linewidth=1.2)
                if detector and detector.has_baseline:
                    lower, center, upper = detector.get_control_limits()
                    ax_sigma.axhline(upper, color="#dc3545",
                                     linestyle="--", linewidth=0.6)
                    ax_sigma.axhline(lower, color="#dc3545",
                                     linestyle="--", linewidth=0.6)
                    ax_sigma.axhline(center, color="#666", linewidth=0.5)
                    if detector.baseline_cutoff_date:
                        ax_sigma.axvline(
                            detector.baseline_cutoff_date,
                            color="#fd7e14", linestyle="--", linewidth=0.6,
                        )
            ax_sigma.set_title("Sigma Drift", loc="left",
                              fontsize=11, fontweight="bold")
            ax_sigma.tick_params(axis="x", rotation=30, labelsize=8)

            # ---- Three process panels ----
            metric_axes = (
                ("untrimmed_resistance", gs[0, 1], "Untrimmed Resistance"),
                ("measured_electrical_angle", gs[1, 0], "Electrical Angle"),
                ("trim_pass_count", gs[1, 1], "Trim Passes"),
            )
            meta = self._db_metric_meta()
            for metric, slot, title in metric_axes:
                ax = fig.add_subplot(slot)
                chart._style_axis(ax)
                panel = process.get(metric, {})
                series = panel.get("series", [])
                if not series:
                    self._draw_empty_state(
                        ax, f"No {title.lower()} data in window",
                    )
                    ax.set_title(title, loc="left",
                                 fontsize=11, fontweight="bold")
                    ax.tick_params(axis="x", rotation=30, labelsize=8)
                    continue
                from datetime import datetime as _dt
                dates = [_dt.fromisoformat(d) for d, _ in series]
                values = [v for _, v in series]
                z = panel.get("z_score") or 0.0
                color = (
                    "#ff8080" if abs(z) >= 3.0
                    else "#ffb060" if abs(z) >= 2.0
                    else "#7ed99e"
                )
                ax.plot(dates, values, color=color, linewidth=1.1)
                base_mean = panel.get("baseline_mean")
                if base_mean is not None:
                    ax.axhline(base_mean, color="#666", linewidth=0.5)
                # Subtitle line under the title with values
                fmt = meta.get(metric, {}).get("fmt", "{:.2f}")
                unit = meta.get(metric, {}).get("unit", "")
                base_s = fmt.format(base_mean) if base_mean is not None else "—"
                rec_s = (
                    fmt.format(panel["recent_mean"])
                    if panel.get("recent_mean") is not None else "—"
                )
                pct = panel.get("delta_pct")
                pct_s = f"({pct:+.1f}%)" if pct is not None else ""
                z_s = f"z={z:+.1f}" if panel.get("z_score") is not None else ""
                ax.set_title(
                    f"{title}  ·  {base_s} → {rec_s} {unit} {pct_s}  {z_s}",
                    loc="left", fontsize=10, fontweight="bold",
                )
                ax.tick_params(axis="x", rotation=30, labelsize=8)

            try:
                fig.tight_layout()
            except Exception:
                pass
            chart.canvas.draw_idle()
            self.status_label.configure(text=f"Drift dashboard · {model}")
        except Exception as e:
            logger.error(f"Single-model drift render error: {e}", exc_info=True)
            self.status_label.configure(text=f"Drift dashboard error: {e}")
```

- [ ] **Step 2: Manual smoke-test**

Launch the app. Trends → Drift. Pick a model from the dropdown that has data (e.g. 7965). Verify:
1. Header bar shows model name, status badge, score pill, units pill.
2. 2×2 grid: Sigma Drift (top-left, with control limits if baseline exists), Untrimmed Resistance, Electrical Angle, Trim Passes.
3. Each process panel has a title with baseline → recent values, % change, and z-score.
4. Reset the dropdown to "All models" → returns to the table view.

- [ ] **Step 3: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/trends.py
git commit -m "feat(trends): single-model drift dashboard — 2×2 grid + header pills"
```

---

## Task 10: Delete dead code

Remove the orphaned per-model sidebar code from the pre-segmented-button era (`_refresh_drift_data`, `_load_drift_data`, `_update_drift_display`, `_on_drift_model_select`, `_show_drift_error`, plus references to never-assigned widgets `_drift_model_list`, `_drift_chart_frame`, `_drift_chart_placeholder`, `_drift_details_label`).

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py`

- [ ] **Step 1: Locate and delete the dead block**

In `trends.py`, find `_refresh_drift_data` (around line 3046). Delete from there through the end of `_show_drift_error` (around line 3270). All five methods are unreachable after the segmented-button refactor — none of `_drift_model_list`, `_drift_chart_frame`, `_drift_chart_placeholder`, `_drift_details_label` are ever assigned in the file.

Verify the deletion by searching:

```
grep -n "_refresh_drift_data\|_load_drift_data\|_update_drift_display\|_on_drift_model_select\|_show_drift_error\|_drift_model_list\|_drift_chart_frame\|_drift_chart_placeholder\|_drift_details_label" src/laser_trim_analyzer/gui/pages/trends.py
```
Expected: only `self.drift_chart = None` at line 277 (the chart-widget reference, kept). If any of the others remain, delete them too.

- [ ] **Step 2: Also remove `self.drift_chart = None`**

The single-model dashboard creates its own chart widget per render, so the long-lived `self.drift_chart` is no longer needed. Remove the line `self.drift_chart = None` from `__init__`.

- [ ] **Step 3: Run the full test suite**

```
python3 -m pytest tests/ -v
```
Expected: all tests pass (existing 19 + 8 new from Tasks 1-4).

- [ ] **Step 4: Manual smoke-test (final)**

Launch the app. Walk the full Drift tab:
1. Default view: All models + ML Drift table (Task 7).
2. Switch to Process Drift sub-tab: metric tab strip + table (Task 8).
3. Try each metric tab.
4. Pick a specific model from the global dropdown: single-model dashboard (Task 9).
5. Click a row in either All-models table: drills into that model.
6. Reset to All models: returns to whichever sub-tab was last active.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/pages/trends.py
git commit -m "refactor(trends): drop orphaned drift sidebar code from pre-refactor era"
```

---

## Self-review against the spec

| Spec section | Covered by |
|--------------|------------|
| 1. Tab header row · global model filter | Task 6 |
| 1. Day-range filter still applies | Tasks 7, 8, 9 use `self.selected_days` |
| 1. Filter populated by `get_models_with_sigma_data` | Task 1 implements; Task 6 wires it up |
| 1. "✕ clear filter" affordance | Task 6 — combo-box "All models" entry serves as the clear control |
| 2a. ML Drift sortable table columns | Task 7 |
| 2a. Default sort: Status then drift score desc | Task 7 (`status_sort` default + tie via stable sort by row order; explicit drift-score sort available via header click) |
| 2a. Click row → drill in | Task 7 (`_on_drift_row_click`) |
| 2b. Process Drift metric tabs | Task 8 |
| 2b. Sortable table with baseline / recent / Δ% / z / sparkline | Task 8 |
| 2b. Default sort: |z| desc | Task 8 (`abs_z` column) |
| 3a. Header pills | Task 9 |
| 3b. 2×2 chart grid | Task 9 |
| 3b. Sigma drift uses control limits | Task 9 |
| 3b. Empty states for missing baseline / data | Task 9 |
| Data layer · 4 query methods | Tasks 1-4 |
| Dead code removal | Task 10 |
