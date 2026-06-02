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
    # The drift detector watches the spec-shifted linearity error -- this is
    # the same column predictor.py treats as the canonical linearity error
    # (see predictor.py line ~686).  TrackResult has no plain `linearity_error`.
    "linearity_error": DBTR.final_linearity_error_shifted,
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

    # Bonferroni multiplicity correction: a model's tier is the worst-of N
    # independent per-metric detectors, so testing each at the full per-tier FP
    # rate inflates the family-wise false-alarm rate ~N-fold (the "flags
    # everything" complaint). Spend the per-tier budget across the watched
    # metrics so the MODEL-level false-alarm rate matches the preset.
    n_metrics = max(1, len(WATCHED_METRICS))
    thresholds: dict[DriftTier, tuple[float, float, float]] = {}
    for tier in (DriftTier.WARNING, DriftTier.DRIFT, DriftTier.OUT_OF_CONTROL):
        p = target_fp_for_tier(sensitivity_preset, tier) / n_metrics
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
