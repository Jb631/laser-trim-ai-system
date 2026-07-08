"""Drift-state fidelity fixes (2026-07-06):

1. Zero-mean degenerate baselines: training substitutes std=1e-9 for constant
   baselines; for zero-MEAN constants the CV guard let that through and the
   collapsing control limits produced garbage flags.
2. recent_window persistence: hydrated detectors woke with an empty window —
   STEP_CHANGE unreachable at read time, σ-shift tiebreaker always 0.
3. Composite demotion: keyed on is_trained alone, a degenerate or stale
   composite silenced its whole input family (untrimmed_error_max etc.).
"""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_drift_live_advance import _seed


# ---- 1. zero-mean degenerate hole ------------------------------------------

def test_zero_mean_constant_baseline_is_degenerate():
    from laser_trim_analyzer.ml.multi_metric_drift_detector import is_degenerate_baseline
    # The exact training sentinel for an all-zero baseline.
    assert is_degenerate_baseline(0.0, 1e-9) is True
    # Legit small sigma still monitorable (smallest real ~1.5e-4).
    assert is_degenerate_baseline(0.0, 1.5e-4) is False
    assert is_degenerate_baseline(0.01, 1.5e-4) is False


def test_zero_mean_constant_model_reads_stable(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as AR, TrackResult as TR, SystemType, StatusType)
    from laser_trim_analyzer.ml.drift_training import train_drift_detector
    from laser_trim_analyzer.ml.manager import get_model_drift_status
    from datetime import datetime, timedelta

    db = DatabaseManager(tmp_path / "z.db")
    start = datetime(2026, 1, 1)
    with db.session() as s:
        for i in range(50):
            ar = AR(filename=f"Z-{i}.xls", file_path=f"/f/Z/{i}", file_hash=f"Z-h{i}",
                    model="ZERO", serial=f"sn{i}", system=SystemType.A,
                    file_date=start + timedelta(days=i), timestamp=start + timedelta(days=i),
                    overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1)
            s.add(ar); s.flush()
            # Constant ZERO-mean metric — the degenerate-hole case.
            s.add(TR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                     resistance_change_percent=0.0))
        s.commit()

    train_drift_detector(db, sensitivity_preset="standard")
    status = get_model_drift_status(db, "ZERO")
    ms = status.per_metric.get("resistance_change_percent")
    assert ms is not None
    assert ms.tier.name == "STABLE", (
        "constant zero-mean baseline must be non-monitorable (STABLE), "
        f"got {ms.tier} magnitude={ms.magnitude}")


# ---- 2. recent_window persistence -------------------------------------------

def test_recent_window_survives_hydration(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.drift_training import train_drift_detector
    from laser_trim_analyzer.ml.manager import get_model_drift_status

    db = DatabaseManager(tmp_path / "w.db")
    _seed(db, "WIN", [0.010 + (i % 3) * 0.0002 for i in range(60)])
    train_drift_detector(db, sensitivity_preset="standard")

    status = get_model_drift_status(db, "WIN")
    ms = status.per_metric.get("untrimmed_resistance")
    assert ms is not None and ms.is_trained
    # Pre-fix: recent_count == 0 and recent_mean is None after hydration.
    assert ms.recent_count > 0, "persisted step window must hydrate"
    assert ms.recent_mean is not None


def test_step_change_flaggable_after_restart(tmp_path):
    """A sharp jump ingested via advance, then a fresh hydration (restart),
    must still expose a non-STABLE tier — the window is part of state now."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.drift_training import (
        train_drift_detector, advance_drift_state)
    from laser_trim_analyzer.ml.manager import get_model_drift_status
    from datetime import datetime

    db = DatabaseManager(tmp_path / "s.db")
    _seed(db, "STEP", [0.010 + (i % 3) * 0.0002 for i in range(50)])
    train_drift_detector(db, sensitivity_preset="standard")

    _seed(db, "STEP", [0.05] * 6, start=datetime(2026, 6, 1))
    advance_drift_state(db, model="STEP")

    # Fresh hydration == app restart.
    status = get_model_drift_status(db, "STEP")
    ms = status.per_metric.get("untrimmed_resistance")
    assert ms.tier.name != "STABLE"
    assert ms.recent_count >= 5  # full window persisted and reloaded


# ---- 3. composite demotion key ----------------------------------------------

def _mk_detector(metric, *, mean=0.01, std=0.001, trained=True, tripped=False,
                 represents_family=True):
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        MetricDetector, STEP_CHANGE_WINDOW)
    det = MetricDetector(
        metric=metric, baseline_mean=mean, baseline_std=std, baseline_count=40,
        is_trained=trained,
        h_per_tier={"WARNING": 0.004, "DRIFT": 0.006, "OUT_OF_CONTROL": 0.009},
        L_per_tier={"WARNING": 2.0, "DRIFT": 2.6, "OUT_OF_CONTROL": 3.3},
        z_per_tier={"WARNING": 2.0, "DRIFT": 2.6, "OUT_OF_CONTROL": 3.3},
        cusum_pos=0.02 if tripped else 0.0,  # far past every h when tripped
        cusum_neg=0.0, ewma_state=mean,
        recent_window=deque(maxlen=STEP_CHANGE_WINDOW),
        represents_family=represents_family,
    )
    return det


def test_degenerate_composite_does_not_silence_family():
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        MultiMetricDriftDetector)

    # Composite trained but DEGENERATE (constant zero-mean score) + a family
    # metric genuinely tripped.
    comp = _mk_detector("composite_trim_risk_score", mean=0.0, std=1e-9)
    fam = _mk_detector("untrimmed_error_max", tripped=True)
    det = MultiMetricDriftDetector(model="M", metrics={
        "composite_trim_risk_score": comp, "untrimmed_error_max": fam})

    status = det.get_status()
    assert status.overall_tier.name != "STABLE", (
        "a degenerate composite must not demote its family")
    assert status.worst_metric == "untrimmed_error_max"


def test_stale_composite_does_not_silence_family():
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        MultiMetricDriftDetector)

    comp = _mk_detector("composite_trim_risk_score", represents_family=False)
    fam = _mk_detector("untrimmed_error_max", tripped=True)
    det = MultiMetricDriftDetector(model="M", metrics={
        "composite_trim_risk_score": comp, "untrimmed_error_max": fam})

    assert det.get_status().overall_tier.name != "STABLE"


def test_healthy_composite_still_demotes_family():
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        MultiMetricDriftDetector)

    comp = _mk_detector("composite_trim_risk_score")           # healthy, quiet
    fam = _mk_detector("untrimmed_error_max", tripped=True)    # tripped
    det = MultiMetricDriftDetector(model="M", metrics={
        "composite_trim_risk_score": comp, "untrimmed_error_max": fam})

    # Family is represented by the composite -> demoted -> model stays stable.
    assert det.get_status().overall_tier.name == "STABLE"
