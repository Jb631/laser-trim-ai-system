# Spec 2 — Multi-Metric Drift Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the V6 multi-metric drift detector — 8 metrics per model (7 trim + smoothness), ML-learned per-model per-metric sensitivity driven by a global preset, new `model_metric_state` table, three-check detection (CUSUM + EWMA + step-change), public API for Spec 3 consumers.

**Architecture:** New ORM table, new types module, new detector classes (MetricDetector + MultiMetricDriftDetector), training orchestration, public API functions. Old `ModelDriftDetector` class stays in place. Pure additive new-code spec. Decomposed into 7 TDD-structured tasks.

**Tech Stack:** Python 3.x, SQLAlchemy 2.0, scipy.stats (inverse normal CDF), numpy, pytest. SQLite database.

**Target branch:** `V6` only. Verify you are on `V6` before starting Task 1. `main` stays at V5 + Spec 1.

**Spec reference:** `docs/superpowers/specs/2026-05-30-spec2-multi-metric-drift-design.md`

---

## File Structure

**Files created (5):**
- `src/laser_trim_analyzer/ml/drift_types.py` — enums (`DriftTier`, `AlertType`) + dataclasses (`MetricStatus`, `ModelDriftStatus`, `ModelAlertSummary`, `TrainingSummary`) + the preset→FP-rate mapping constant.
- `src/laser_trim_analyzer/ml/multi_metric_drift_detector.py` — `MetricDetector` + `MultiMetricDriftDetector` classes + threshold math helpers.
- `src/laser_trim_analyzer/ml/drift_training.py` — `train_drift_detector` orchestration.
- `tests/test_spec2_multi_metric_drift.py` — all tests for Spec 2.

**Files modified (3):**
- `src/laser_trim_analyzer/database/models.py` — add `ModelMetricState` ORM class.
- `src/laser_trim_analyzer/database/manager.py` — add `model_metric_state` startup migration + first-startup auto-train hook.
- `src/laser_trim_analyzer/ml/manager.py` — add `get_drifting_models`, `get_model_drift_status`, `preview_alert_count` public functions.
- `src/laser_trim_analyzer/config.py` — add `drift_sensitivity: str = "standard"` field to `MLConfig`.

**Files NOT touched** (Spec 2 non-goals):
- `src/laser_trim_analyzer/ml/drift_detector.py` — old `ModelDriftDetector` class stays. Spec 3 retires it.
- Any GUI page. Spec 3 owns UI.

---

## Task 1: ORM table + startup migration

**Files:**
- Modify: `src/laser_trim_analyzer/database/models.py`
- Modify: `src/laser_trim_analyzer/database/manager.py`
- Test: `tests/test_spec2_multi_metric_drift.py` (CREATE)

- [ ] **Step 1: Create the test file with three schema/migration tests**

Create `tests/test_spec2_multi_metric_drift.py` with this content:

```python
"""Spec 2 — Multi-metric drift detector.

Each test maps to one element of the spec at
docs/superpowers/specs/2026-05-30-spec2-multi-metric-drift-design.md.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Task 1: ORM model_metric_state -- schema + idempotent startup migration
# ---------------------------------------------------------------------------


def test_model_metric_state_table_has_expected_columns():
    """ORM declares all spec'd columns with correct types and the
    UNIQUE(model, metric) constraint.
    """
    from laser_trim_analyzer.database.models import ModelMetricState

    cols = {c.name: c for c in ModelMetricState.__table__.columns}

    required = {
        "id", "model", "metric",
        "baseline_cutoff_date", "baseline_mean", "baseline_std", "baseline_count",
        "is_trained",
        "h_warning", "h_drift", "h_oc",
        "L_warning", "L_drift", "L_oc",
        "z_warning", "z_drift", "z_oc",
        "ewma_state", "cusum_pos", "cusum_neg",
        "last_updated",
    }
    missing = required - set(cols)
    assert not missing, f"ModelMetricState missing columns: {sorted(missing)}"

    assert cols["model"].nullable is False
    assert cols["metric"].nullable is False
    assert cols["baseline_count"].nullable is False
    assert cols["is_trained"].nullable is False

    # Unique constraint on (model, metric)
    uq_cols = {
        frozenset(c.name for c in uq.columns)
        for uq in ModelMetricState.__table__.constraints
        if uq.__class__.__name__ == "UniqueConstraint"
    }
    assert frozenset({"model", "metric"}) in uq_cols, (
        f"Missing UNIQUE(model, metric); got: {uq_cols}"
    )


def test_migration_creates_table_on_fresh_db(tmp_path):
    """A fresh DB has the table created by DatabaseManager init."""
    import sqlite3
    from laser_trim_analyzer.database.manager import DatabaseManager

    db_path = tmp_path / "fresh.db"
    DatabaseManager(db_path)

    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    conn.close()
    assert "model_metric_state" in tables


def test_migration_idempotent_on_existing_db(tmp_path):
    """Initializing twice doesn't error and doesn't duplicate the table."""
    import sqlite3
    from laser_trim_analyzer.database.manager import DatabaseManager

    db_path = tmp_path / "twice.db"
    DatabaseManager(db_path)
    cols_first = _table_cols(db_path, "model_metric_state")

    # Second init should be a no-op for this table
    DatabaseManager(db_path)
    cols_second = _table_cols(db_path, "model_metric_state")

    assert cols_first == cols_second


def _table_cols(db_path, table):
    """Helper: return the set of column names for a table."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    conn.close()
    return cols
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v`

Expected: 3 FAILs — `ModelMetricState` import fails or class doesn't exist; migration test fails because table not created.

- [ ] **Step 3: Add the ORM class to `database/models.py`**

In `src/laser_trim_analyzer/database/models.py`, find a good location (after `ModelMLState` at line 995, around line 1130 where its class body ends — read those lines to find the exact end). Append a new class after `ModelMLState`:

```python
class ModelMetricState(Base):
    """
    Per-model per-metric drift detection state and learned thresholds.

    Spec 2 (2026-05-30): replaces the single-metric drift state held by
    the old ModelDriftDetector class.  One row per (model, metric) tuple.
    8 metrics watched per model: 7 trim + max_smoothness_value.

    Thresholds (h/L/z per tier) are recomputed in place whenever the
    sensitivity preset changes — they derive from baseline_std and the
    target FP rate for the (preset, tier) pair.
    """
    __tablename__ = 'model_metric_state'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model = Column(String(50), nullable=False, index=True)
    metric = Column(String(50), nullable=False, index=True)

    # Baseline (learned during training)
    baseline_cutoff_date = Column(DateTime)
    baseline_mean = Column(Float)
    baseline_std = Column(Float)
    baseline_count = Column(Integer, nullable=False, default=0)
    is_trained = Column(Boolean, nullable=False, default=False)

    # Per-tier thresholds (derived from baseline_std + preset target FP rate)
    h_warning = Column(Float)
    h_drift = Column(Float)
    h_oc = Column(Float)
    L_warning = Column(Float)
    L_drift = Column(Float)
    L_oc = Column(Float)
    z_warning = Column(Float)
    z_drift = Column(Float)
    z_oc = Column(Float)

    # Live runtime state (updated on each new sample)
    ewma_state = Column(Float)
    cusum_pos = Column(Float, nullable=False, default=0.0)
    cusum_neg = Column(Float, nullable=False, default=0.0)
    last_updated = Column(DateTime)

    __table_args__ = (
        UniqueConstraint('model', 'metric', name='uq_model_metric_state'),
        Index('idx_model_metric_state_model_metric', 'model', 'metric'),
    )
```

Make sure `UniqueConstraint` and `Index` are imported at the top of `models.py` (they are — many existing tables use them).

- [ ] **Step 4: Add the migration entry to `DatabaseManager`**

In `src/laser_trim_analyzer/database/manager.py`, find the startup migration block (the same area as the Spec 1 migration around line 560-589). After all the existing track_results migrations, add a new section for `model_metric_state` creation:

```python
            # Migration: Create model_metric_state table for Spec 2.
            # This is a CREATE TABLE rather than ALTER TABLE because the
            # table is entirely new in V6.  Use Base.metadata.create_all
            # with checkfirst=True for idempotency.
            try:
                from laser_trim_analyzer.database.models import ModelMetricState
                ModelMetricState.__table__.create(bind=session.bind, checkfirst=True)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.warning(
                    f"Migration warning for model_metric_state (may already exist): {e}"
                )
```

(Indentation must match the surrounding migration block.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v`

Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/database/models.py src/laser_trim_analyzer/database/manager.py tests/test_spec2_multi_metric_drift.py
git commit -m "feat(spec2): add model_metric_state ORM table + startup migration

New table holds per-(model, metric) drift baseline + learned per-tier
thresholds + live CUSUM/EWMA runtime state.  Idempotent CREATE TABLE
runs at DatabaseManager startup.

Spec: docs/superpowers/specs/2026-05-30-spec2-multi-metric-drift-design.md"
```

---

## Task 2: Types module — enums, dataclasses, preset mapping

**Files:**
- Create: `src/laser_trim_analyzer/ml/drift_types.py`
- Test: `tests/test_spec2_multi_metric_drift.py` (APPEND)

- [ ] **Step 1: Append type-module tests**

Append to `tests/test_spec2_multi_metric_drift.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: types -- enums, dataclasses, preset mapping
# ---------------------------------------------------------------------------


def test_drift_tier_ordering():
    """DriftTier values must compare in severity order."""
    from laser_trim_analyzer.ml.drift_types import DriftTier

    assert DriftTier.STABLE < DriftTier.WARNING
    assert DriftTier.WARNING < DriftTier.DRIFT
    assert DriftTier.DRIFT < DriftTier.OUT_OF_CONTROL


def test_alert_type_values():
    """AlertType has the two values from the spec."""
    from laser_trim_analyzer.ml.drift_types import AlertType

    assert AlertType.STEP_CHANGE
    assert AlertType.SLOW_DRIFT


def test_target_fp_rate_matrix_standard_preset():
    """target_fp_for_tier('standard', ...) returns the spec's matrix."""
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, target_fp_for_tier,
    )

    assert target_fp_for_tier("standard", DriftTier.WARNING) == pytest.approx(0.05)
    assert target_fp_for_tier("standard", DriftTier.DRIFT) == pytest.approx(0.01)
    assert target_fp_for_tier("standard", DriftTier.OUT_OF_CONTROL) == pytest.approx(0.001)


def test_target_fp_rate_matrix_all_presets():
    """Verify all four presets produce strictly-monotone-stricter FP rates."""
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, target_fp_for_tier,
    )

    for tier in (DriftTier.WARNING, DriftTier.DRIFT, DriftTier.OUT_OF_CONTROL):
        loose = target_fp_for_tier("loose", tier)
        standard = target_fp_for_tier("standard", tier)
        tight = target_fp_for_tier("tight", tier)
        strict = target_fp_for_tier("strict", tier)
        assert loose > standard > tight > strict, (
            f"Tier {tier}: presets should be strictly stricter; "
            f"got loose={loose}, standard={standard}, tight={tight}, strict={strict}"
        )


def test_metric_status_dataclass_round_trip():
    """MetricStatus dataclass holds all spec fields."""
    from laser_trim_analyzer.ml.drift_types import (
        MetricStatus, DriftTier, AlertType,
    )

    ms = MetricStatus(
        metric="sigma_gradient",
        tier=DriftTier.WARNING,
        alert_type=AlertType.STEP_CHANGE,
        magnitude=2.3,
        baseline_mean=0.010,
        baseline_std=0.002,
        recent_mean=0.012,
        recent_count=5,
        is_trained=True,
    )
    assert ms.metric == "sigma_gradient"
    assert ms.tier == DriftTier.WARNING
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v -k "drift_tier or alert_type or target_fp or metric_status"`

Expected: 5 FAILs — module doesn't exist.

- [ ] **Step 3: Create the types module**

Create `src/laser_trim_analyzer/ml/drift_types.py`:

```python
"""Spec 2 — types shared across the multi-metric drift detector.

Enums, dataclasses, and the sensitivity-preset → target-FP-rate mapping
constant.  Kept separate from the detector logic so consumers (Spec 3 UI,
manager.py API functions) can import types without pulling in scipy.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, Enum
from typing import Dict, List, Optional, Tuple


class DriftTier(IntEnum):
    """Drift severity tiers.  IntEnum so comparisons work directly
    (DriftTier.WARNING < DriftTier.DRIFT) for worst-of aggregation.
    """
    STABLE = 0
    WARNING = 1
    DRIFT = 2
    OUT_OF_CONTROL = 3


class AlertType(Enum):
    """How the drift was detected -- determines the displayed alert
    type when a tier elevation fires.
    """
    STEP_CHANGE = "step_change"
    SLOW_DRIFT = "slow_drift"


# Sensitivity preset -> per-tier target false-positive rate.
# See spec section "Sensitivity learning" for the rationale.
_PRESET_FP_MATRIX: Dict[str, Dict[DriftTier, float]] = {
    "loose":    {DriftTier.WARNING: 0.10,   DriftTier.DRIFT: 0.05,    DriftTier.OUT_OF_CONTROL: 0.01},
    "standard": {DriftTier.WARNING: 0.05,   DriftTier.DRIFT: 0.01,    DriftTier.OUT_OF_CONTROL: 0.001},
    "tight":    {DriftTier.WARNING: 0.01,   DriftTier.DRIFT: 0.001,   DriftTier.OUT_OF_CONTROL: 0.0001},
    "strict":   {DriftTier.WARNING: 0.001,  DriftTier.DRIFT: 0.0001,  DriftTier.OUT_OF_CONTROL: 0.00001},
}


def target_fp_for_tier(preset: str, tier: DriftTier) -> float:
    """Return the target false-positive rate for the (preset, tier) pair.

    Raises KeyError if the preset is unknown.  Spec 3's Settings UI
    must validate the preset string before passing it in.
    """
    return _PRESET_FP_MATRIX[preset][tier]


# Allowed metric names.  Single source of truth -- detector, training,
# and queries import from here.
WATCHED_METRICS: Tuple[str, ...] = (
    "sigma_gradient",
    "untrimmed_sigma_gradient",
    "untrimmed_resistance",
    "linearity_error",
    "measured_electrical_angle",
    "trim_pass_count",
    "resistance_change_percent",
    "max_smoothness_value",
)


@dataclass
class MetricStatus:
    """Current state of one (model, metric) pair.  Returned by
    MetricDetector.get_status() and aggregated into ModelDriftStatus.
    """
    metric: str
    tier: DriftTier
    alert_type: Optional[AlertType]   # None when tier == STABLE
    magnitude: float                   # σ-units over the tier's threshold
    baseline_mean: float
    baseline_std: float
    recent_mean: Optional[float]
    recent_count: int
    is_trained: bool


@dataclass
class ModelDriftStatus:
    """Full per-metric breakdown for one model -- for Spec 3 Model page."""
    model: str
    overall_tier: DriftTier
    worst_metric: Optional[str]
    worst_alert_type: Optional[AlertType]
    per_metric: Dict[str, MetricStatus] = field(default_factory=dict)
    last_processed: Optional[datetime] = None


@dataclass
class ModelAlertSummary:
    """Compact form for Spec 3 Triage list."""
    model: str
    tier: DriftTier
    alert_type: AlertType
    worst_metric: str
    magnitude: float


@dataclass
class TrainingSummary:
    """Returned by train_drift_detector for progress reporting."""
    models_trained: int
    metrics_per_model: int = 8
    skipped_insufficient_data: List[Tuple[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v -k "drift_tier or alert_type or target_fp or metric_status"`

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/ml/drift_types.py tests/test_spec2_multi_metric_drift.py
git commit -m "feat(spec2): types module -- enums, dataclasses, preset mapping

DriftTier (IntEnum so comparisons work), AlertType, four dataclasses,
the WATCHED_METRICS tuple, and the sensitivity-preset target-FP-rate
matrix.  No detector logic, no scipy dependency -- safe for any
consumer to import."
```

---

## Task 3: `MetricDetector` class — threshold math + three checks

**Files:**
- Create: `src/laser_trim_analyzer/ml/multi_metric_drift_detector.py`
- Test: `tests/test_spec2_multi_metric_drift.py` (APPEND)

- [ ] **Step 1: Append MetricDetector tests**

Append to `tests/test_spec2_multi_metric_drift.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: MetricDetector -- threshold math + three checks
# ---------------------------------------------------------------------------


def test_threshold_math_at_known_fp_rates():
    """compute_thresholds returns the inverse-CDF values for known p."""
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        compute_thresholds,
    )

    sigma = 1.0  # makes L and z numeric
    # Per spec: L = phi^-1(1 - p/2), z = phi^-1(1 - p), h via SPC approx
    h_05, L_05, z_05 = compute_thresholds(sigma=sigma, target_fp=0.05)
    assert L_05 == pytest.approx(1.96, abs=0.01)
    assert z_05 == pytest.approx(1.645, abs=0.01)

    h_01, L_01, z_01 = compute_thresholds(sigma=sigma, target_fp=0.01)
    assert L_01 == pytest.approx(2.576, abs=0.01)
    assert z_01 == pytest.approx(2.326, abs=0.01)

    # h should grow as p shrinks (stricter)
    assert h_01 > h_05


def test_metric_detector_flat_baseline_never_flags():
    """A series of samples at baseline_mean never elevates the tier."""
    from laser_trim_analyzer.ml.multi_metric_drift_detector import MetricDetector
    from laser_trim_analyzer.ml.drift_types import DriftTier

    det = _build_trained_detector(baseline_mean=10.0, baseline_std=1.0)

    for _ in range(100):
        status = det.update(10.0)
        assert status.tier == DriftTier.STABLE


def test_metric_detector_step_change_trips_step_check():
    """An abrupt mean shift of 3σ trips step-change at some tier."""
    from laser_trim_analyzer.ml.multi_metric_drift_detector import MetricDetector
    from laser_trim_analyzer.ml.drift_types import DriftTier, AlertType

    det = _build_trained_detector(baseline_mean=0.0, baseline_std=1.0)

    # Feed 5 samples at the new mean to fill the step-change window
    for _ in range(5):
        status = det.update(3.0)

    assert status.tier > DriftTier.STABLE
    assert status.alert_type == AlertType.STEP_CHANGE


def test_metric_detector_slow_ramp_trips_cusum_or_ewma():
    """A linear ramp from 0 to 2σ over 50 samples trips slow-drift."""
    from laser_trim_analyzer.ml.multi_metric_drift_detector import MetricDetector
    from laser_trim_analyzer.ml.drift_types import DriftTier, AlertType

    det = _build_trained_detector(baseline_mean=0.0, baseline_std=1.0)

    for i in range(50):
        value = (i / 50.0) * 2.0  # 0 → 2σ over 50 samples
        status = det.update(value)

    assert status.tier > DriftTier.STABLE
    assert status.alert_type == AlertType.SLOW_DRIFT


def test_metric_detector_handles_nan_input():
    """NaN samples are ignored; no crash, runtime state unchanged."""
    import math
    from laser_trim_analyzer.ml.multi_metric_drift_detector import MetricDetector

    det = _build_trained_detector(baseline_mean=0.0, baseline_std=1.0)
    # Prime with a real sample so we have a non-zero ewma
    det.update(0.0)
    cusum_before = det.cusum_pos

    status = det.update(math.nan)
    assert det.cusum_pos == cusum_before  # unchanged


def test_metric_detector_untrained_never_elevates():
    """is_trained=False means tier stays Stable forever."""
    from laser_trim_analyzer.ml.multi_metric_drift_detector import MetricDetector
    from laser_trim_analyzer.ml.drift_types import DriftTier

    det = _build_trained_detector(baseline_mean=0.0, baseline_std=1.0)
    det.is_trained = False

    for v in (0.0, 5.0, -5.0, 100.0):
        status = det.update(v)
        assert status.tier == DriftTier.STABLE


def _build_trained_detector(*, baseline_mean, baseline_std):
    """Helper: build a MetricDetector with standard-preset thresholds
    pre-computed.  Use this in every detector test below.
    """
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        MetricDetector, compute_thresholds,
    )
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, target_fp_for_tier,
    )

    h = {}
    L = {}
    z = {}
    for tier in (DriftTier.WARNING, DriftTier.DRIFT, DriftTier.OUT_OF_CONTROL):
        p = target_fp_for_tier("standard", tier)
        h[tier], L[tier], z[tier] = compute_thresholds(baseline_std, p)

    return MetricDetector(
        metric="test_metric",
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        baseline_count=100,
        is_trained=True,
        h_per_tier={t.name: v for t, v in h.items()},
        L_per_tier={t.name: v for t, v in L.items()},
        z_per_tier={t.name: v for t, v in z.items()},
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v -k metric_detector`

Expected: 6 FAILs — `MetricDetector` and `compute_thresholds` don't exist.

- [ ] **Step 3: Create the detector module**

Create `src/laser_trim_analyzer/ml/multi_metric_drift_detector.py`:

```python
"""Spec 2 — MetricDetector + MultiMetricDriftDetector.

The detector logic lives here.  Threshold math uses scipy.stats.norm
for inverse normal CDF.  Three independent checks per sample: CUSUM,
EWMA, step-change.

This file replaces the analytical core of the old ml/drift_detector.py
ModelDriftDetector class for V6.  The old class is retained until Spec 3
retires the UI that uses it.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np
from scipy import stats

from laser_trim_analyzer.ml.drift_types import (
    AlertType,
    DriftTier,
    MetricStatus,
    ModelAlertSummary,
    ModelDriftStatus,
    WATCHED_METRICS,
)


# EWMA smoothing constant.  Fixed in v1 per spec.
EWMA_LAMBDA: float = 0.2

# CUSUM allowance (k).  Half the baseline std.  Fixed in v1 per spec.
CUSUM_K_SIGMAS: float = 0.5

# Step-change window size N (number of recent samples averaged).
STEP_CHANGE_WINDOW: int = 5


def compute_thresholds(sigma: float, target_fp: float) -> tuple[float, float, float]:
    """Compute (h, L, z) thresholds for a target false-positive rate.

    - L = phi^-1(1 - p/2)         (EWMA two-sided control-limit width)
    - z = phi^-1(1 - p)           (step-change one-sided cutoff)
    - h = -ln(p) / k_cusum * sigma  (SPC approximation; see spec)

    sigma is the baseline standard deviation; needed only for h.
    """
    if target_fp <= 0 or target_fp >= 1:
        raise ValueError(f"target_fp must be in (0, 1), got {target_fp}")

    L = float(stats.norm.ppf(1.0 - target_fp / 2.0))
    z = float(stats.norm.ppf(1.0 - target_fp))
    h = -math.log(target_fp) / CUSUM_K_SIGMAS * sigma
    return h, L, z


@dataclass
class MetricDetector:
    """One model × one metric detector.

    Holds baseline stats, per-tier thresholds, and live runtime state.
    update() processes a new sample and returns the current MetricStatus.
    """
    metric: str
    baseline_mean: float
    baseline_std: float
    baseline_count: int
    is_trained: bool

    # Thresholds keyed by tier name ("WARNING", "DRIFT", "OUT_OF_CONTROL")
    h_per_tier: Dict[str, float] = field(default_factory=dict)
    L_per_tier: Dict[str, float] = field(default_factory=dict)
    z_per_tier: Dict[str, float] = field(default_factory=dict)

    # Runtime state
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    ewma_state: Optional[float] = None
    recent_window: Deque[float] = field(default_factory=lambda: deque(maxlen=STEP_CHANGE_WINDOW))

    # Last computed status (for get_status without re-processing)
    _last_status: Optional[MetricStatus] = None

    def update(self, value: float) -> MetricStatus:
        """Process a new sample; update state; return current status."""
        # Ignore NaN/Inf -- they shouldn't move state
        if value is None or not math.isfinite(value):
            return self.get_status()

        # If never trained, runtime updates still happen so a later
        # training pass can flip is_trained=True with non-zero state.
        # Use baseline_mean=0 as a no-op if baseline is also None.
        mu = self.baseline_mean if self.baseline_mean is not None else 0.0
        sigma = self.baseline_std if self.baseline_std is not None else 1.0

        # Initialize EWMA to baseline mean on first update
        if self.ewma_state is None:
            self.ewma_state = mu

        # CUSUM update with allowance k = 0.5σ
        k = CUSUM_K_SIGMAS * sigma
        self.cusum_pos = max(0.0, self.cusum_pos + (value - mu) - k)
        self.cusum_neg = min(0.0, self.cusum_neg + (value - mu) + k)

        # EWMA update
        self.ewma_state = EWMA_LAMBDA * value + (1.0 - EWMA_LAMBDA) * self.ewma_state

        # Step-change window
        self.recent_window.append(value)

        status = self._compute_status()
        self._last_status = status
        return status

    def _compute_status(self) -> MetricStatus:
        """Evaluate current state against per-tier thresholds.

        Returns the highest tier any check trips at.  When step-change and
        slow-drift tie at the same tier, step-change is preferred for the
        displayed alert_type.
        """
        if not self.is_trained:
            return MetricStatus(
                metric=self.metric,
                tier=DriftTier.STABLE,
                alert_type=None,
                magnitude=0.0,
                baseline_mean=self.baseline_mean or 0.0,
                baseline_std=self.baseline_std or 0.0,
                recent_mean=None,
                recent_count=len(self.recent_window),
                is_trained=False,
            )

        sigma = self.baseline_std
        mu = self.baseline_mean
        # EWMA control limits use sigma_ewma = sigma * sqrt(lambda / (2-lambda))
        sigma_ewma = sigma * math.sqrt(EWMA_LAMBDA / (2.0 - EWMA_LAMBDA))

        recent_mean = (
            float(np.mean(list(self.recent_window))) if self.recent_window else None
        )
        recent_n = len(self.recent_window)

        # Test each tier from strictest to loosest; pick the highest tier
        # that any check trips at.
        winning_tier = DriftTier.STABLE
        winning_alert_type: Optional[AlertType] = None
        winning_magnitude = 0.0

        for tier in (DriftTier.OUT_OF_CONTROL, DriftTier.DRIFT, DriftTier.WARNING):
            tier_key = tier.name
            h = self.h_per_tier.get(tier_key)
            L = self.L_per_tier.get(tier_key)
            z = self.z_per_tier.get(tier_key)
            if h is None or L is None or z is None:
                continue

            cusum_trip = self.cusum_pos > h or self.cusum_neg < -h
            ewma_trip = (
                abs(self.ewma_state - mu) > L * sigma_ewma
            )
            step_trip = False
            step_magnitude = 0.0
            if recent_mean is not None and recent_n >= STEP_CHANGE_WINDOW:
                step_magnitude = (
                    abs(recent_mean - mu) * math.sqrt(recent_n) / sigma
                    if sigma > 0 else 0.0
                )
                step_trip = step_magnitude > z

            if cusum_trip or ewma_trip or step_trip:
                # First tier we hit (strictest) wins.
                winning_tier = tier
                # Step-change wins the alert_type tiebreaker
                if step_trip:
                    winning_alert_type = AlertType.STEP_CHANGE
                    winning_magnitude = step_magnitude
                else:
                    winning_alert_type = AlertType.SLOW_DRIFT
                    winning_magnitude = max(
                        abs(self.cusum_pos) / h if h > 0 else 0.0,
                        abs(self.ewma_state - mu) / (L * sigma_ewma) if (L * sigma_ewma) > 0 else 0.0,
                    )
                break  # strictest wins; no need to check lower tiers

        return MetricStatus(
            metric=self.metric,
            tier=winning_tier,
            alert_type=winning_alert_type,
            magnitude=winning_magnitude,
            baseline_mean=mu,
            baseline_std=sigma,
            recent_mean=recent_mean,
            recent_count=recent_n,
            is_trained=True,
        )

    def get_status(self) -> MetricStatus:
        """Current status without processing a new sample."""
        if self._last_status is None:
            return self._compute_status()
        return self._last_status

    def reset_runtime(self) -> None:
        """Zero out CUSUM and re-init EWMA to baseline mean.

        Called after a baseline refresh (training) so the detector
        starts fresh against the new baseline.
        """
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.ewma_state = self.baseline_mean
        self.recent_window.clear()
        self._last_status = None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v -k metric_detector`

Expected: 6 PASS.

If any of the synthetic-data tests (step-change, slow-ramp) fail, the most likely cause is the CUSUM h-approximation being mis-tuned. The plan defers calibration of that constant (`k_cusum = 0.5σ`, `h = -ln(p)/k_cusum*σ`) to the implementation phase per the spec — adjust the constants in `multi_metric_drift_detector.py` if the synthetic tests don't hit the expected behavior. Document any adjustments inline.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/ml/multi_metric_drift_detector.py tests/test_spec2_multi_metric_drift.py
git commit -m "feat(spec2): MetricDetector with CUSUM + EWMA + step-change

Threshold math via scipy.stats.norm.ppf for L and z; SPC approximation
for h.  Three independent checks per sample; strictest tier wins;
step-change wins same-tier alert_type tiebreaker.

is_trained=False detectors stay tier=Stable even with extreme samples
so models still warming up don't generate spurious alerts."
```

---

## Task 4: `MultiMetricDriftDetector` container

**Files:**
- Modify: `src/laser_trim_analyzer/ml/multi_metric_drift_detector.py` (extend with new class)
- Test: `tests/test_spec2_multi_metric_drift.py` (APPEND)

- [ ] **Step 1: Append container tests**

Append to `tests/test_spec2_multi_metric_drift.py`:

```python
# ---------------------------------------------------------------------------
# Task 4: MultiMetricDriftDetector -- worst-of aggregation
# ---------------------------------------------------------------------------


def test_multi_metric_detector_worst_of_tier():
    """Model's overall_tier is the max of its metric tiers."""
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        MultiMetricDriftDetector,
    )
    from laser_trim_analyzer.ml.drift_types import DriftTier

    det1 = _build_trained_detector(baseline_mean=0.0, baseline_std=1.0)
    det2 = _build_trained_detector(baseline_mean=10.0, baseline_std=2.0)
    det1.metric = "sigma_gradient"
    det2.metric = "linearity_error"

    mmd = MultiMetricDriftDetector("8340-1", {
        "sigma_gradient": det1,
        "linearity_error": det2,
    })

    # Inject step changes -- det1 sees mild drift, det2 sees big drift
    for _ in range(5):
        mmd.update({"sigma_gradient": 1.5, "linearity_error": 18.0})

    status = mmd.get_status()
    # det2 (linearity_error) saw the bigger deviation -> drives overall
    assert status.overall_tier >= DriftTier.WARNING
    assert status.worst_metric == "linearity_error"


def test_multi_metric_detector_step_change_wins_tier_tie():
    """Within the worst metric, step-change wins alert_type when tied."""
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        MultiMetricDriftDetector,
    )
    from laser_trim_analyzer.ml.drift_types import AlertType, DriftTier

    # Single-metric detector hit with a sharp step
    det = _build_trained_detector(baseline_mean=0.0, baseline_std=1.0)
    det.metric = "sigma_gradient"

    mmd = MultiMetricDriftDetector("test-model", {"sigma_gradient": det})

    for _ in range(5):
        mmd.update({"sigma_gradient": 4.0})

    status = mmd.get_status()
    assert status.worst_alert_type == AlertType.STEP_CHANGE


def test_multi_metric_detector_partial_sample_ok():
    """A sample missing some metrics still works -- absent keys treated
    as 'no new data this tick' for that metric.
    """
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        MultiMetricDriftDetector,
    )

    det1 = _build_trained_detector(baseline_mean=0.0, baseline_std=1.0)
    det2 = _build_trained_detector(baseline_mean=0.0, baseline_std=1.0)
    det1.metric = "sigma_gradient"
    det2.metric = "untrimmed_sigma_gradient"

    mmd = MultiMetricDriftDetector("test", {
        "sigma_gradient": det1,
        "untrimmed_sigma_gradient": det2,
    })

    # Only update one metric
    status = mmd.update({"sigma_gradient": 0.5})
    # Should not crash; the missing metric stays at its prior state
    assert "untrimmed_sigma_gradient" in status.per_metric
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v -k multi_metric_detector`

Expected: 3 FAILs — `MultiMetricDriftDetector` doesn't exist yet.

- [ ] **Step 3: Add the container class**

Append to `src/laser_trim_analyzer/ml/multi_metric_drift_detector.py`:

```python
@dataclass
class MultiMetricDriftDetector:
    """Container for one model's 8 MetricDetector instances.

    Renamed from the old ModelDriftDetector to avoid collision while
    the legacy class still lives in ml/drift_detector.py.  Spec 3 will
    retire the old class and consumers will swap to import this one.
    """
    model: str
    metrics: Dict[str, MetricDetector]
    last_processed: Optional[datetime] = None

    def update(self, sample: Dict[str, float]) -> ModelDriftStatus:
        """Process a per-metric sample dict.  Returns the new model status.

        Missing keys in `sample` mean 'no new data this tick' for that
        metric; its state is unchanged.
        """
        from datetime import datetime as _dt
        for metric_name, detector in self.metrics.items():
            if metric_name in sample:
                value = sample[metric_name]
                if value is not None:
                    detector.update(value)
        self.last_processed = _dt.now()
        return self.get_status()

    def get_status(self) -> ModelDriftStatus:
        """Aggregate per-metric status into a model-level status.

        Overall tier = max of metric tiers.  Worst-metric and alert-type
        come from the metric driving the max tier.
        """
        per_metric = {
            name: det.get_status() for name, det in self.metrics.items()
        }

        # Find the worst metric.  Sort by (tier, magnitude) descending.
        ranked = sorted(
            per_metric.items(),
            key=lambda kv: (int(kv[1].tier), kv[1].magnitude),
            reverse=True,
        )
        if ranked:
            worst_name, worst_status = ranked[0]
        else:
            worst_name, worst_status = None, None

        overall_tier = worst_status.tier if worst_status else DriftTier.STABLE
        worst_metric = worst_name if overall_tier > DriftTier.STABLE else None
        worst_alert_type = (
            worst_status.alert_type
            if (worst_status and overall_tier > DriftTier.STABLE)
            else None
        )

        return ModelDriftStatus(
            model=self.model,
            overall_tier=overall_tier,
            worst_metric=worst_metric,
            worst_alert_type=worst_alert_type,
            per_metric=per_metric,
            last_processed=self.last_processed,
        )
```

Add `from datetime import datetime` to the imports at the top of the file if not already present (you'll need it for `last_processed`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v -k multi_metric_detector`

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/ml/multi_metric_drift_detector.py tests/test_spec2_multi_metric_drift.py
git commit -m "feat(spec2): MultiMetricDriftDetector container

Holds Dict[metric_name, MetricDetector].  update() routes per-metric
samples; get_status() returns ModelDriftStatus with worst-of aggregation
(highest tier wins; ties broken by magnitude).  Missing keys in the
sample dict mean 'no new data this tick' for that metric."
```

---

## Task 5: Training orchestration

**Files:**
- Create: `src/laser_trim_analyzer/ml/drift_training.py`
- Test: `tests/test_spec2_multi_metric_drift.py` (APPEND)

- [ ] **Step 1: Append training tests**

Append to `tests/test_spec2_multi_metric_drift.py`:

```python
# ---------------------------------------------------------------------------
# Task 5: train_drift_detector -- baseline computation + threshold writing
# ---------------------------------------------------------------------------


def test_training_writes_one_row_per_model_per_metric(tmp_path):
    """For each (model, metric) with sufficient history, one row is
    written to model_metric_state.
    """
    from datetime import datetime, timedelta
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR,
        SystemType as DBSystemType, StatusType as DBStatusType,
        ModelMetricState,
    )
    from laser_trim_analyzer.ml.drift_training import train_drift_detector
    from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS

    db = DatabaseManager(tmp_path / "train.db")
    today = datetime.now()

    # Build 50 TrackResults for one model so baseline_count >= 30
    with db.session() as s:
        for i in range(50):
            ar = DBAR(
                filename=f"f{i}.xls",
                file_path=f"/fake/f{i}.xls",
                file_hash=f"hash-{i}",
                model="TEST-MODEL",
                serial=f"sn{i}",
                system=DBSystemType.A,
                file_date=today - timedelta(days=60 - i),
                timestamp=today,
                overall_status=DBStatusType.PASS,
                has_multi_tracks=False,
                processing_time=0.1,
            )
            s.add(ar)
            s.flush()
            tr = DBTR(
                analysis_id=ar.id,
                track_id="TRK1",
                sigma_gradient=0.01 + 0.0005 * i,
                untrimmed_sigma_gradient=0.015 + 0.0003 * i,
                untrimmed_resistance=1000.0 + i,
                linearity_error=0.005 + 0.0001 * i,
                measured_electrical_angle=170.0,
                trim_pass_count=2,
                resistance_change_percent=15.0,
            )
            s.add(tr)
        s.commit()

    summary = train_drift_detector(db, sensitivity_preset="standard")

    assert summary.models_trained >= 1
    with db.session() as s:
        rows = s.query(ModelMetricState).filter(
            ModelMetricState.model == "TEST-MODEL"
        ).all()
        # 7 trim metrics get rows; max_smoothness_value gets skipped
        # because there are no SmoothnessResult rows in this fixture.
        metric_names = {r.metric for r in rows}
        for trim_metric in WATCHED_METRICS:
            if trim_metric != "max_smoothness_value":
                assert trim_metric in metric_names, (
                    f"Missing trained row for metric {trim_metric}"
                )


def test_training_marks_insufficient_history_untrained(tmp_path):
    """Models with fewer than 30 baseline samples get is_trained=False."""
    from datetime import datetime, timedelta
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR,
        SystemType as DBSystemType, StatusType as DBStatusType,
        ModelMetricState,
    )
    from laser_trim_analyzer.ml.drift_training import train_drift_detector

    db = DatabaseManager(tmp_path / "thin.db")
    today = datetime.now()

    with db.session() as s:
        for i in range(10):  # only 10 samples -- below threshold
            ar = DBAR(
                filename=f"f{i}.xls",
                file_path=f"/fake/f{i}.xls",
                file_hash=f"hash-{i}",
                model="THIN-MODEL",
                serial=f"sn{i}",
                system=DBSystemType.A,
                file_date=today - timedelta(days=30 - i),
                timestamp=today,
                overall_status=DBStatusType.PASS,
                has_multi_tracks=False,
                processing_time=0.1,
            )
            s.add(ar)
            s.flush()
            tr = DBTR(analysis_id=ar.id, track_id="TRK1", sigma_gradient=0.01)
            s.add(tr)
        s.commit()

    train_drift_detector(db, sensitivity_preset="standard")

    with db.session() as s:
        rows = s.query(ModelMetricState).filter(
            ModelMetricState.model == "THIN-MODEL"
        ).all()
        for r in rows:
            assert r.is_trained is False, (
                f"Metric {r.metric}: expected is_trained=False with only "
                f"10 samples; got baseline_count={r.baseline_count}"
            )


def test_training_idempotent(tmp_path):
    """Running training twice doesn't double-write rows."""
    from datetime import datetime, timedelta
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR,
        SystemType as DBSystemType, StatusType as DBStatusType,
        ModelMetricState,
    )
    from laser_trim_analyzer.ml.drift_training import train_drift_detector

    db = DatabaseManager(tmp_path / "twice.db")
    today = datetime.now()

    with db.session() as s:
        for i in range(50):
            ar = DBAR(
                filename=f"f{i}.xls",
                file_path=f"/fake/f{i}.xls",
                file_hash=f"hash-{i}",
                model="REPEAT-MODEL",
                serial=f"sn{i}",
                system=DBSystemType.A,
                file_date=today - timedelta(days=60 - i),
                timestamp=today,
                overall_status=DBStatusType.PASS,
                has_multi_tracks=False,
                processing_time=0.1,
            )
            s.add(ar)
            s.flush()
            tr = DBTR(analysis_id=ar.id, track_id="TRK1", sigma_gradient=0.01)
            s.add(tr)
        s.commit()

    train_drift_detector(db, sensitivity_preset="standard")
    with db.session() as s:
        count_after_first = s.query(ModelMetricState).count()

    train_drift_detector(db, sensitivity_preset="standard")
    with db.session() as s:
        count_after_second = s.query(ModelMetricState).count()

    assert count_after_first == count_after_second
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v -k training`

Expected: 3 FAILs — module doesn't exist.

- [ ] **Step 3: Create the training module**

Create `src/laser_trim_analyzer/ml/drift_training.py`:

```python
"""Spec 2 — training orchestration for the multi-metric drift detector.

train_drift_detector reads historical data per model, computes baselines
and per-tier thresholds, and writes/upserts model_metric_state rows.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAR,
    ModelMetricState,
    SmoothnessResult as DBSR,
    TrackResult as DBTR,
)
from laser_trim_analyzer.ml.drift_types import (
    DriftTier,
    TrainingSummary,
    WATCHED_METRICS,
    target_fp_for_tier,
)
from laser_trim_analyzer.ml.multi_metric_drift_detector import compute_thresholds

logger = logging.getLogger(__name__)


# Minimum number of baseline samples required to consider a (model, metric)
# trained.  Below this, the row is still written but is_trained=False so the
# detector reports tier=Stable forever for that metric.
MIN_BASELINE_SAMPLES: int = 30


# Maps metric name -> SQLAlchemy column on TrackResult, except smoothness
# which maps to SmoothnessResult.max_smoothness_value.
_TRACK_METRIC_COLUMNS = {
    "sigma_gradient": DBTR.sigma_gradient,
    "untrimmed_sigma_gradient": DBTR.untrimmed_sigma_gradient,
    "untrimmed_resistance": DBTR.untrimmed_resistance,
    "linearity_error": DBTR.linearity_error,
    "measured_electrical_angle": DBTR.measured_electrical_angle,
    "trim_pass_count": DBTR.trim_pass_count,
    "resistance_change_percent": DBTR.resistance_change_percent,
}


def train_drift_detector(
    db,
    sensitivity_preset: str = "standard",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> TrainingSummary:
    """Train (or retrain) drift detection per model.

    For each model in the DB:
      1. Load historical samples per metric.
      2. Compute baseline mean/std/count.
      3. Compute per-tier thresholds from baseline_std + sensitivity preset.
      4. Upsert into model_metric_state.
    """
    start = time.time()

    # Discover models that have data
    with db.session() as s:
        models = [
            r[0] for r in s.query(DBAR.model).distinct().all()
        ]
        smoothness_models = [
            r[0] for r in s.query(DBSR.model).distinct().all()
        ]
        all_models = sorted(set(models) | set(smoothness_models))

    models_trained = 0
    skipped: list[Tuple[str, str]] = []

    for i, model in enumerate(all_models):
        if progress_callback:
            progress_callback(model, i, len(all_models))

        # Train each metric for this model
        model_had_any_training = False
        for metric in WATCHED_METRICS:
            ok = _train_one_metric(db, model, metric, sensitivity_preset)
            if ok:
                model_had_any_training = True
            else:
                skipped.append((model, metric))

        if model_had_any_training:
            models_trained += 1

    return TrainingSummary(
        models_trained=models_trained,
        metrics_per_model=len(WATCHED_METRICS),
        skipped_insufficient_data=skipped,
        duration_seconds=time.time() - start,
    )


def _train_one_metric(
    db,
    model: str,
    metric: str,
    sensitivity_preset: str,
) -> bool:
    """Compute baseline + thresholds for one (model, metric) and upsert.

    Returns True if the row was written with is_trained=True.
    """
    values = _load_historical_values(db, model, metric)

    if len(values) < MIN_BASELINE_SAMPLES:
        # Write an untrained sentinel row so future passes can detect it
        _upsert_metric_state(
            db, model, metric,
            baseline_mean=None, baseline_std=None,
            baseline_count=len(values),
            is_trained=False,
            thresholds=None,
        )
        return False

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < MIN_BASELINE_SAMPLES:
        _upsert_metric_state(
            db, model, metric,
            baseline_mean=None, baseline_std=None,
            baseline_count=len(arr),
            is_trained=False,
            thresholds=None,
        )
        return False

    baseline_mean = float(np.mean(arr))
    baseline_std = float(np.std(arr, ddof=1))
    # Avoid divide-by-zero in threshold math
    if baseline_std <= 0.0:
        baseline_std = 1e-9

    thresholds: dict[DriftTier, tuple[float, float, float]] = {}
    for tier in (DriftTier.WARNING, DriftTier.DRIFT, DriftTier.OUT_OF_CONTROL):
        p = target_fp_for_tier(sensitivity_preset, tier)
        thresholds[tier] = compute_thresholds(sigma=baseline_std, target_fp=p)

    _upsert_metric_state(
        db, model, metric,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        baseline_count=len(arr),
        is_trained=True,
        thresholds=thresholds,
    )
    return True


def _load_historical_values(db, model: str, metric: str) -> list[float]:
    """Load this model's historical samples for the given metric."""
    if metric == "max_smoothness_value":
        with db.session() as s:
            rows = s.query(DBSR.max_smoothness_value).filter(
                DBSR.model == model,
                DBSR.max_smoothness_value.isnot(None),
            ).all()
            return [r[0] for r in rows if r[0] is not None]

    col = _TRACK_METRIC_COLUMNS.get(metric)
    if col is None:
        return []

    with db.session() as s:
        rows = s.query(col).join(DBAR, DBTR.analysis_id == DBAR.id).filter(
            DBAR.model == model, col.isnot(None),
        ).all()
        return [r[0] for r in rows if r[0] is not None]


def _upsert_metric_state(
    db,
    model: str,
    metric: str,
    *,
    baseline_mean: Optional[float],
    baseline_std: Optional[float],
    baseline_count: int,
    is_trained: bool,
    thresholds: Optional[dict],
) -> None:
    """Insert or update a single model_metric_state row."""
    with db.session() as s:
        row = s.query(ModelMetricState).filter(
            ModelMetricState.model == model,
            ModelMetricState.metric == metric,
        ).first()

        if row is None:
            row = ModelMetricState(model=model, metric=metric)
            s.add(row)

        row.baseline_mean = baseline_mean
        row.baseline_std = baseline_std
        row.baseline_count = baseline_count
        row.is_trained = is_trained
        row.last_updated = datetime.now()
        # Reset runtime state so the detector starts fresh against the new baseline
        row.cusum_pos = 0.0
        row.cusum_neg = 0.0
        row.ewma_state = baseline_mean

        if thresholds is not None:
            row.h_warning, row.L_warning, row.z_warning = thresholds[DriftTier.WARNING]
            row.h_drift,   row.L_drift,   row.z_drift   = thresholds[DriftTier.DRIFT]
            row.h_oc,      row.L_oc,      row.z_oc      = thresholds[DriftTier.OUT_OF_CONTROL]
        else:
            for col in ("h_warning", "L_warning", "z_warning",
                        "h_drift", "L_drift", "z_drift",
                        "h_oc", "L_oc", "z_oc"):
                setattr(row, col, None)

        s.commit()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v -k training`

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/ml/drift_training.py tests/test_spec2_multi_metric_drift.py
git commit -m "feat(spec2): train_drift_detector orchestration

Iterates models × metrics, loads historical samples, computes baseline
mean/std + per-tier thresholds, upserts model_metric_state rows.
Smoothness metric reads from SmoothnessResult; the other 7 from
TrackResult.  Insufficient-history (< 30 samples) writes an is_trained=
False sentinel so future training passes can upgrade once data grows."
```

---

## Task 6: Public API in `ml/manager.py`

**Files:**
- Modify: `src/laser_trim_analyzer/ml/manager.py`
- Test: `tests/test_spec2_multi_metric_drift.py` (APPEND)

- [ ] **Step 1: Append API tests**

Append to `tests/test_spec2_multi_metric_drift.py`:

```python
# ---------------------------------------------------------------------------
# Task 6: Public API in ml/manager.py
# ---------------------------------------------------------------------------


def test_get_drifting_models_empty_when_nothing_flagged(tmp_path):
    """Fresh DB with no training data -> no flagged models."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.manager import get_drifting_models

    db = DatabaseManager(tmp_path / "empty.db")
    result = get_drifting_models(db, sensitivity_preset="standard")
    assert result == []


def test_get_drifting_models_returns_flagged_only(tmp_path):
    """Models with overall_tier > Stable appear in the result list."""
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.drift_types import DriftTier
    from laser_trim_analyzer.ml.manager import get_drifting_models

    db = DatabaseManager(tmp_path / "flagged.db")

    # Hand-write a row that puts model 8340-1 into a flag-tripping state.
    # Trick: set cusum_pos > h_warning so the detector's evaluate path
    # reports Warning when checked.
    with db.session() as s:
        row = ModelMetricState(
            model="8340-1",
            metric="sigma_gradient",
            baseline_mean=0.01,
            baseline_std=0.001,
            baseline_count=100,
            is_trained=True,
            h_warning=1.0, h_drift=5.0, h_oc=10.0,
            L_warning=2.0, L_drift=3.0, L_oc=4.0,
            z_warning=1.6, z_drift=2.3, z_oc=3.0,
            cusum_pos=2.0,  # > h_warning -> trips Warning
            cusum_neg=0.0,
            ewma_state=0.01,
            last_updated=datetime.now(),
        )
        s.add(row)
        s.commit()

    result = get_drifting_models(db, sensitivity_preset="standard")
    flagged_models = [r.model for r in result]
    assert "8340-1" in flagged_models


def test_get_model_drift_status_includes_all_eight_metric_slots(tmp_path):
    """For any known model, get_model_drift_status returns a per_metric
    dict with all 8 metric keys (some may be is_trained=False).
    """
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.manager import get_model_drift_status
    from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS

    db = DatabaseManager(tmp_path / "single.db")
    status = get_model_drift_status(db, "UNKNOWN-MODEL")

    assert set(status.per_metric.keys()) == set(WATCHED_METRICS)


def test_preview_alert_count_returns_per_tier_counts(tmp_path):
    """preview_alert_count returns a dict with the three tier names as keys."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.manager import preview_alert_count

    db = DatabaseManager(tmp_path / "preview.db")
    counts = preview_alert_count(db, sensitivity_preset="standard")

    assert "warning" in counts
    assert "drift" in counts
    assert "out_of_control" in counts
    # Empty DB -> zero counts
    assert counts["warning"] == 0
    assert counts["drift"] == 0
    assert counts["out_of_control"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v -k "get_drifting or get_model_drift_status or preview_alert_count"`

Expected: 4 FAILs — functions don't exist.

- [ ] **Step 3: Add the API functions to `ml/manager.py`**

In `src/laser_trim_analyzer/ml/manager.py`, append these three functions at the bottom of the file (after the existing class definitions):

```python
def get_drifting_models(db, sensitivity_preset: str = "standard"):
    """Return sorted list of currently-flagged models.

    Reads model_metric_state directly (no historical re-scan).  Each
    model's overall_tier is computed by hydrating its 8 MetricDetector
    rows and running the worst-of aggregation.  Sorted by (tier desc,
    magnitude desc).

    Returns empty list when nothing is above Stable.
    """
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, ModelAlertSummary, WATCHED_METRICS,
    )

    summaries: list[ModelAlertSummary] = []
    with db.session() as s:
        # Get unique models that have at least one row
        models = [
            r[0] for r in s.query(ModelMetricState.model).distinct().all()
        ]

    for model in models:
        status = get_model_drift_status(db, model)
        if status.overall_tier > DriftTier.STABLE:
            summaries.append(ModelAlertSummary(
                model=model,
                tier=status.overall_tier,
                alert_type=status.worst_alert_type,
                worst_metric=status.worst_metric or "",
                magnitude=status.per_metric[status.worst_metric].magnitude
                          if status.worst_metric else 0.0,
            ))

    summaries.sort(key=lambda r: (int(r.tier), r.magnitude), reverse=True)
    return summaries


def get_model_drift_status(db, model: str):
    """Return full per-metric breakdown for one model.

    Hydrates MetricDetector instances from model_metric_state rows and
    asks the container for its current status.
    """
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        MetricDetector, MultiMetricDriftDetector,
    )

    metrics = {}
    with db.session() as s:
        rows = s.query(ModelMetricState).filter(
            ModelMetricState.model == model,
        ).all()
        rows_by_metric = {r.metric: r for r in rows}

    for metric_name in WATCHED_METRICS:
        row = rows_by_metric.get(metric_name)
        if row is None:
            # No DB row -> create a placeholder untrained detector
            metrics[metric_name] = MetricDetector(
                metric=metric_name,
                baseline_mean=0.0,
                baseline_std=0.0,
                baseline_count=0,
                is_trained=False,
            )
        else:
            det = MetricDetector(
                metric=metric_name,
                baseline_mean=row.baseline_mean or 0.0,
                baseline_std=row.baseline_std or 0.0,
                baseline_count=row.baseline_count,
                is_trained=row.is_trained,
                h_per_tier={
                    "WARNING": row.h_warning or 0.0,
                    "DRIFT": row.h_drift or 0.0,
                    "OUT_OF_CONTROL": row.h_oc or 0.0,
                },
                L_per_tier={
                    "WARNING": row.L_warning or 0.0,
                    "DRIFT": row.L_drift or 0.0,
                    "OUT_OF_CONTROL": row.L_oc or 0.0,
                },
                z_per_tier={
                    "WARNING": row.z_warning or 0.0,
                    "DRIFT": row.z_drift or 0.0,
                    "OUT_OF_CONTROL": row.z_oc or 0.0,
                },
                cusum_pos=row.cusum_pos or 0.0,
                cusum_neg=row.cusum_neg or 0.0,
                ewma_state=row.ewma_state,
            )
            metrics[metric_name] = det

    container = MultiMetricDriftDetector(model=model, metrics=metrics)
    return container.get_status()


def preview_alert_count(db, sensitivity_preset: str) -> dict:
    """Count models that would flag at each tier under the candidate preset.

    Cheap -- doesn't re-scan history.  Recomputes per-tier thresholds for
    each existing row, then evaluates against cached runtime state.
    """
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, target_fp_for_tier, WATCHED_METRICS,
    )
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        MetricDetector, MultiMetricDriftDetector, compute_thresholds,
    )

    counts = {"warning": 0, "drift": 0, "out_of_control": 0}

    with db.session() as s:
        models = [
            r[0] for r in s.query(ModelMetricState.model).distinct().all()
        ]

    for model in models:
        # Build detectors with candidate-preset thresholds in place of cached ones
        metrics = {}
        with db.session() as s:
            rows = s.query(ModelMetricState).filter(
                ModelMetricState.model == model
            ).all()

        for row in rows:
            if not row.is_trained or row.baseline_std is None:
                continue
            h_per_tier = {}
            L_per_tier = {}
            z_per_tier = {}
            for tier in (DriftTier.WARNING, DriftTier.DRIFT, DriftTier.OUT_OF_CONTROL):
                p = target_fp_for_tier(sensitivity_preset, tier)
                h, L, z = compute_thresholds(row.baseline_std, p)
                h_per_tier[tier.name] = h
                L_per_tier[tier.name] = L
                z_per_tier[tier.name] = z

            metrics[row.metric] = MetricDetector(
                metric=row.metric,
                baseline_mean=row.baseline_mean,
                baseline_std=row.baseline_std,
                baseline_count=row.baseline_count,
                is_trained=True,
                h_per_tier=h_per_tier,
                L_per_tier=L_per_tier,
                z_per_tier=z_per_tier,
                cusum_pos=row.cusum_pos or 0.0,
                cusum_neg=row.cusum_neg or 0.0,
                ewma_state=row.ewma_state,
            )

        container = MultiMetricDriftDetector(model=model, metrics=metrics)
        status = container.get_status()
        if status.overall_tier == DriftTier.WARNING:
            counts["warning"] += 1
        elif status.overall_tier == DriftTier.DRIFT:
            counts["drift"] += 1
        elif status.overall_tier == DriftTier.OUT_OF_CONTROL:
            counts["out_of_control"] += 1

    return counts
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v -k "get_drifting or get_model_drift_status or preview_alert_count"`

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/ml/manager.py tests/test_spec2_multi_metric_drift.py
git commit -m "feat(spec2): public API -- get_drifting_models, status, preview

Three functions Spec 3's Triage / Model / Settings pages will call.
All three hydrate MetricDetector instances from model_metric_state rows
on demand -- no in-memory cache, DB is the source of truth.

preview_alert_count is the cheap-recompute path used by the Settings
sensitivity-preset slider's live preview."
```

---

## Task 7: Config integration + first-startup auto-train

**Files:**
- Modify: `src/laser_trim_analyzer/config.py`
- Modify: `src/laser_trim_analyzer/database/manager.py`
- Test: `tests/test_spec2_multi_metric_drift.py` (APPEND)

- [ ] **Step 1: Append config + startup tests**

Append to `tests/test_spec2_multi_metric_drift.py`:

```python
# ---------------------------------------------------------------------------
# Task 7: Config + first-startup auto-train hook
# ---------------------------------------------------------------------------


def test_ml_config_has_drift_sensitivity_default():
    """MLConfig defaults drift_sensitivity to 'standard'."""
    from laser_trim_analyzer.config import MLConfig

    cfg = MLConfig()
    assert cfg.drift_sensitivity == "standard"


def test_config_load_reads_drift_sensitivity(tmp_path):
    """Loading config.yaml with ml.drift_sensitivity sets the field."""
    import yaml
    from laser_trim_analyzer.config import Config

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "ml": {"drift_sensitivity": "tight"},
    }))

    cfg = Config.load(yaml_path)
    assert cfg.ml.drift_sensitivity == "tight"


def test_first_startup_does_not_crash_on_empty_db(tmp_path):
    """A brand-new DB initializes without raising, even though model_metric_
    state is empty and auto-train would have nothing to do.
    """
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "first.db")
    # No exception = pass
    assert db is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v -k "drift_sensitivity or first_startup"`

Expected: 2 FAILs (config field doesn't exist) + 1 PASS (DatabaseManager already initializes — we just want to be sure adding the Task 7 hook doesn't break that).

- [ ] **Step 3: Add the `drift_sensitivity` field to `MLConfig`**

In `src/laser_trim_analyzer/config.py`, find `MLConfig` at line 73-79:

```python
@dataclass
class MLConfig:
    """ML configuration."""
    enabled: bool = True
    use_threshold_optimizer: bool = True
    use_drift_detector: bool = True
    min_samples_for_training: int = 20
```

Add the new field after `min_samples_for_training`:

```python
@dataclass
class MLConfig:
    """ML configuration."""
    enabled: bool = True
    use_threshold_optimizer: bool = True
    use_drift_detector: bool = True
    min_samples_for_training: int = 20
    drift_sensitivity: str = "standard"  # Spec 2: loose / standard / tight / strict
```

The existing `Config.load()` `for key, value in data["ml"].items()` loop will pick up the new key automatically — no changes to `load()` needed.

- [ ] **Step 4: Optional first-startup auto-train hook**

For v1, **skip the auto-train hook**. The "first-startup auto-train" behaviour described in Spec 2 is a UX nicety; Spec 3's Settings page provides the "Retrain drift detector" button that James clicks manually after V6 starts. Auto-train can be wired in later when Spec 3 lands the Settings UI. The third test (`test_first_startup_does_not_crash_on_empty_db`) confirms the manager works without the hook.

If you want to wire it in proactively, add a check at the end of `DatabaseManager.__init__`:

```python
# Spec 2: first-startup auto-train if model_metric_state is empty.
# Disabled by default; Spec 3's Settings UI calls train_drift_detector
# explicitly via the 'Retrain' button.
```

Just the comment is fine for v1 — the hook itself is deferred to Spec 3 work.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v -k "drift_sensitivity or first_startup"`

Expected: 3 PASS.

- [ ] **Step 6: Run the entire Spec 2 test file**

Run: `pytest tests/test_spec2_multi_metric_drift.py -v`

Expected: all PASS (Task 1: 3 + Task 2: 5 + Task 3: 6 + Task 4: 3 + Task 5: 3 + Task 6: 4 + Task 7: 3 = ~27 tests).

- [ ] **Step 7: Run regression sweep**

Run: `pytest tests/test_spec1_untrimmed_sigma.py tests/test_log_derived_bugfixes_2026_05_30.py tests/test_5_8_2026_bugfixes.py tests/test_spec2_multi_metric_drift.py -v 2>&1 | tail -10`

Expected: ~90+ tests PASS total, zero failures. If any pre-existing test fails, Spec 2 has accidentally disturbed an earlier fix.

- [ ] **Step 8: Commit**

```bash
git add src/laser_trim_analyzer/config.py tests/test_spec2_multi_metric_drift.py
git commit -m "feat(spec2): drift_sensitivity preset in MLConfig

Adds the 'drift_sensitivity' field to MLConfig (default 'standard').
Config.load() picks it up via the existing ml-section loop -- no changes
to the loader.

First-startup auto-train hook deferred to Spec 3 work; manual Retrain
button on the Settings page will be the trigger."
```

---

## Post-implementation verification

After all 7 tasks land:

```
pytest tests/ -v 2>&1 | tail -20
```

Expected: zero failures across the full suite (Spec 2's ~27 + the historical regression set).

Then on V6 with a real local DB:

1. Switch to V6: `git checkout V6`
2. Start the app once so the migration runs: `python -m laser_trim_analyzer.app` (or whatever the entry point is)
3. Open the DB: `sqlite3 ./data/analysis.db`
4. Verify: `SELECT name FROM sqlite_master WHERE type='table' AND name='model_metric_state';` should return one row.
5. Run training in a Python shell:
   ```python
   from laser_trim_analyzer.config import Config
   from laser_trim_analyzer.database.manager import DatabaseManager
   from laser_trim_analyzer.ml.drift_training import train_drift_detector

   cfg = Config.load()
   db = DatabaseManager(cfg.database.path)
   summary = train_drift_detector(db, cfg.ml.drift_sensitivity)
   print(summary)
   ```
6. Verify in the DB: `SELECT model, metric, is_trained, baseline_count FROM model_metric_state ORDER BY model, metric LIMIT 20;` should show populated rows for trim metrics and (if any smoothness data exists) the smoothness metric.

## Out-of-scope reminders (do NOT do these)

- **Do not** modify the old `ml/drift_detector.py` (the legacy `ModelDriftDetector` class). It stays until Spec 3 retires the UI that uses it.
- **Do not** add a UI surface for any of the new functions. Spec 3 owns that.
- **Do not** wire `train_drift_detector` into the file-processing pipeline. Spec 3 adds the Settings "Retrain" button; until then, training is called only from a Python shell or test.
- **Do not** add per-model sensitivity overrides. The preset is global.
- **Do not** export drift alerts to Excel. Future spec.
