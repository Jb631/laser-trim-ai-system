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
                status=DBStatusType.PASS,
                sigma_gradient=0.01 + 0.0005 * i,
                untrimmed_sigma_gradient=0.015 + 0.0003 * i,
                untrimmed_resistance=1000.0 + i,
                # TrackResult has no plain `linearity_error`; the canonical
                # spec-shifted column is what predictor.py + the drift
                # detector watch.
                final_linearity_error_shifted=0.005 + 0.0001 * i,
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
            tr = DBTR(analysis_id=ar.id, track_id="TRK1", status=DBStatusType.PASS, sigma_gradient=0.01)
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
            tr = DBTR(analysis_id=ar.id, track_id="TRK1", status=DBStatusType.PASS, sigma_gradient=0.01)
            s.add(tr)
        s.commit()

    train_drift_detector(db, sensitivity_preset="standard")
    with db.session() as s:
        count_after_first = s.query(ModelMetricState).count()

    train_drift_detector(db, sensitivity_preset="standard")
    with db.session() as s:
        count_after_second = s.query(ModelMetricState).count()

    assert count_after_first == count_after_second
