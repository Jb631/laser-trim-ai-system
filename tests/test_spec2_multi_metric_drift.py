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


def test_degenerate_baseline_never_flags_and_no_absurd_magnitude():
    """A near-constant baseline (σ≈0) is non-monitorable: it must stay STABLE
    with magnitude 0, not explode to billions of sigma. Regression for the
    production bug where a constant electrical-angle baseline (std=4.5e-16)
    reported +1.99e11 σ and topped Triage."""
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        MetricDetector, is_degenerate_baseline,
    )
    from laser_trim_analyzer.ml.drift_types import DriftTier

    # std collapsed to float epsilon relative to a ~2.0 mean (real case: 7764)
    det = _build_trained_detector(baseline_mean=1.99, baseline_std=4.5e-16)
    for _ in range(10):
        status = det.update(2.5)          # a value far from the constant baseline
    assert status.tier == DriftTier.STABLE
    assert status.magnitude == 0.0

    # constant integer baseline (real case: 1848803 trim_pass_count, std=1e-9)
    det2 = _build_trained_detector(baseline_mean=1.0, baseline_std=1e-9)
    for _ in range(10):
        status2 = det2.update(2.0)
    assert status2.tier == DriftTier.STABLE
    assert status2.magnitude == 0.0

    # the classifier itself: degenerate cases vs the smallest legitimate σ (~1.5e-4)
    assert is_degenerate_baseline(1.99, 4.5e-16) is True
    assert is_degenerate_baseline(1.0, 1e-9) is True
    assert is_degenerate_baseline(0.0, 0.0) is True
    assert is_degenerate_baseline(0.0015, 1.5e-4) is False   # real untrimmed_sigma_gradient
    assert is_degenerate_baseline(0.0, 0.13) is False        # near-zero mean, real spread


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


def test_composite_active_demotes_family_metrics_from_tier():
    """When the composite is deployed (its metric is trained), the trim-effort family
    features (sigma, resistance_change, trim_pass) are evidence-only and don't raise the
    model tier. Without a deployed composite the same trip DOES flag (worst-of)."""
    from laser_trim_analyzer.ml.multi_metric_drift_detector import MultiMetricDriftDetector
    from laser_trim_analyzer.ml.drift_types import DriftTier

    fam = _build_trained_detector(baseline_mean=0.0, baseline_std=1.0)
    fam.metric = "untrimmed_sigma_gradient"
    for _ in range(5):
        fam.update(5.0)                                 # trip the family metric
    assert fam.get_status().tier > DriftTier.STABLE

    composite = _build_trained_detector(baseline_mean=0.0, baseline_std=1.0)  # trained, stable
    composite.metric = "composite_trim_risk_score"

    det = MultiMetricDriftDetector("M", {
        "untrimmed_sigma_gradient": fam,
        "composite_trim_risk_score": composite,
    })
    status = det.get_status()
    assert status.overall_tier == DriftTier.STABLE                       # family demoted
    assert status.per_metric["untrimmed_sigma_gradient"].tier > DriftTier.STABLE  # still evidence

    # No deployed composite -> family metric drives the flag as before.
    det_no_comp = MultiMetricDriftDetector("M2", {"untrimmed_sigma_gradient": fam})
    assert det_no_comp.get_status().overall_tier > DriftTier.STABLE


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


def test_active_model_set_recent_vs_legacy(tmp_path):
    """Active = data within recent_days of the DATASET's latest date (not wall-clock),
    plus MPS-pinned models. A unit last run years ago is inactive."""
    from datetime import datetime, timedelta
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, SystemType as DBSystemType, StatusType as DBStatusType)
    from laser_trim_analyzer.ml.manager import active_model_set

    db = DatabaseManager(tmp_path / "active.db")
    latest = datetime(2026, 5, 28)
    legacy = latest - timedelta(days=2000)   # ~5.5 years before the latest data
    with db.session() as s:
        s.add(DBAR(filename="r.xls", file_path="/f", file_hash="h1", model="RECENT", serial="s1",
                   system=DBSystemType.A, file_date=latest, timestamp=latest,
                   overall_status=DBStatusType.PASS, has_multi_tracks=False, processing_time=0.1))
        s.add(DBAR(filename="o.xls", file_path="/f", file_hash="h2", model="LEGACY", serial="s2",
                   system=DBSystemType.A, file_date=legacy, timestamp=legacy,
                   overall_status=DBStatusType.PASS, has_multi_tracks=False, processing_time=0.1))
        s.commit()

    active = active_model_set(db, recent_days=90)
    assert "RECENT" in active
    assert "LEGACY" not in active
    # MPS pin keeps a legacy model active regardless of recency.
    assert "LEGACY" in active_model_set(db, recent_days=90, mps_models=["LEGACY"])


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
            # D-SIGMA: drift watches the untrimmed-sweep sigma (post-trim
            # sigma_gradient is no longer a watched drift metric).
            metric="untrimmed_sigma_gradient",
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
