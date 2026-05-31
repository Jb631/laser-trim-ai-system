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

    assert len(cols_first) > 0, (
        "Table was not created on first DatabaseManager init; "
        "idempotency test would have passed spuriously"
    )
    assert cols_first == cols_second


def _table_cols(db_path, table):
    """Helper: return the set of column names for a table."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    conn.close()
    return cols


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
