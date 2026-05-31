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
