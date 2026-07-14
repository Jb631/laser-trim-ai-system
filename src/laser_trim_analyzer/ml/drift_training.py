"""Spec 2 — training orchestration for the multi-metric drift detector.

train_drift_detector reads historical data per model, computes baselines
and per-tier thresholds, and writes/upserts model_metric_state rows.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
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
    "untrimmed_error_max": DBTR.untrimmed_error_max,
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
    "composite_trim_risk_score": DBTR.composite_trim_risk_score,
}

# Public alias so the Spec 3 UI charts the SAME column the detector trained on
# (prevents the linearity_error/final_linearity_error_shifted mismatch). Read-only.
TRACK_METRIC_COLUMNS = _TRACK_METRIC_COLUMNS


def train_drift_detector(
    db,
    sensitivity_preset: str = "standard",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    model: Optional[str] = None,
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
    if model is not None:
        # Single-model retrain (baseline requalification path).
        all_models = [m for m in all_models if m == model]

    models_trained = 0
    skipped: list[Tuple[str, str]] = []

    for i, model in enumerate(all_models):
        if progress_callback:
            progress_callback(model, i, len(all_models))

        # Baseline floor: a manual requalification (design change) makes
        # history before its effective date OFF-LIMITS for this model's
        # baselines (2026-07-13, James's policy: per-model manual reset).
        baseline_start = None
        try:
            req = db.get_baseline_requalification(model)
            if req:
                baseline_start = datetime.fromisoformat(str(req[0])[:19])
        except Exception:
            logger.exception("baseline requalification lookup failed for %s", model)

        # Train each metric for this model
        model_had_any_training = False
        for metric in WATCHED_METRICS:
            ok = _train_one_metric(db, model, metric, sensitivity_preset,
                                   baseline_start=baseline_start)
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


def corrected_tier_thresholds(
    sensitivity_preset: str, baseline_std: float
) -> dict:
    """Per-tier (h, L, z) thresholds with the Bonferroni family-wise correction.

    THE single source of threshold math. The worst-of-N aggregation across the
    watched metrics inflates the family-wise false-alarm rate, so each tier's
    target FP is divided by len(WATCHED_METRICS) before computing thresholds.

    Training, preview_alert_count, and apply_sensitivity_preset must ALL go
    through here — before 2026-07-06 the preview/apply paths skipped the
    correction, so "Save preset" wrote thresholds ~9x looser (in FP target)
    than retraining at the same preset, and the preview counts didn't match
    what training would produce.
    """
    n_metrics = max(1, len(WATCHED_METRICS))
    thresholds: dict[DriftTier, tuple[float, float, float]] = {}
    for tier in (DriftTier.WARNING, DriftTier.DRIFT, DriftTier.OUT_OF_CONTROL):
        p = target_fp_for_tier(sensitivity_preset, tier) / n_metrics
        thresholds[tier] = compute_thresholds(sigma=baseline_std, target_fp=p)
    return thresholds


def _train_one_metric(
    db,
    model: str,
    metric: str,
    sensitivity_preset: str,
    baseline_start=None,
) -> bool:
    """Compute baseline + thresholds for one (model, metric) and upsert.

    Returns True if the row was written with is_trained=True.
    """
    # Load dated samples so we can fix the baseline to an EARLY window and then
    # REPLAY the recent window through the detector. Without this, the baseline
    # is computed over all history (laundering any drift into it) and the runtime
    # state is reset, so the detector can never flag (the H8 "always Stable" bug).
    # A requalified baseline (design change) floors the sample window: data
    # from the old design must not launder the new design's baseline.
    samples = [
        (d, v, rid) for (d, v, rid) in _load_samples_with_dates(
            db, model, metric,
            after=(baseline_start - timedelta(seconds=1)) if baseline_start else None)
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]

    if len(samples) < MIN_BASELINE_SAMPLES:
        _upsert_metric_state(
            db, model, metric,
            baseline_mean=None, baseline_std=None,
            baseline_count=len(samples), is_trained=False, thresholds=None,
        )
        return False

    # Baseline = oldest ~70% (>= MIN_BASELINE_SAMPLES); replay = the remainder.
    split = min(len(samples), max(MIN_BASELINE_SAMPLES, int(len(samples) * 0.7)))
    baseline_samples = samples[:split]
    replay_samples = samples[split:]

    arr = np.asarray([v for (_d, v, _r) in baseline_samples], dtype=float)
    baseline_mean = float(np.mean(arr))
    baseline_std = float(np.std(arr, ddof=1))
    if baseline_std <= 0.0:
        baseline_std = 1e-9

    thresholds = corrected_tier_thresholds(sensitivity_preset, baseline_std)

    # Replay the recent window so persisted runtime state reflects drift already
    # present in history. Samples are WINSORIZED (clipped to baseline ± the
    # suspect gate): an isolated scale-corrupt value (e.g. 380σ) contributes
    # at most one bounded push and then decays, while a real sustained shift
    # still accumulates past threshold within a couple of samples.
    from laser_trim_analyzer.ml.drift_types import SUSPECT_SIGMA_GATE
    clip_lo = baseline_mean - SUSPECT_SIGMA_GATE * baseline_std
    clip_hi = baseline_mean + SUSPECT_SIGMA_GATE * baseline_std
    det = _build_detector(metric, baseline_mean, baseline_std, len(baseline_samples),
                          thresholds_dict=thresholds)
    for (_d, v, _r) in replay_samples:
        det.update(min(max(float(v), clip_lo), clip_hi))

    _upsert_metric_state(
        db, model, metric,
        baseline_mean=baseline_mean, baseline_std=baseline_std,
        baseline_count=len(baseline_samples), is_trained=True, thresholds=thresholds,
        cusum_pos=det.cusum_pos, cusum_neg=det.cusum_neg, ewma_state=det.ewma_state,
        baseline_cutoff_date=baseline_samples[-1][0], last_sample_date=samples[-1][0],
        # Watermark = highest source-row id consumed (samples are date-ordered,
        # so the max is NOT necessarily the last element).
        last_row_id=max(rid for (_d, _v, rid) in samples),
        recent_window=list(det.recent_window),
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
    cusum_pos: Optional[float] = None,
    cusum_neg: Optional[float] = None,
    ewma_state: Optional[float] = None,
    baseline_cutoff_date=None,
    last_sample_date=None,
    last_row_id: Optional[int] = None,
    recent_window: Optional[list] = None,
) -> None:
    """Insert or update a single model_metric_state row.

    Runtime state (cusum/ewma) uses the REPLAYED values when provided (so the
    detector reflects drift already in history); otherwise it resets to the
    baseline. last_updated stores the last SAMPLE date (a file_date marker for
    advance_drift_state), not wall-clock time.
    """
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
        row.baseline_cutoff_date = baseline_cutoff_date
        # last_updated = last processed SAMPLE date (advance starts after this).
        row.last_updated = last_sample_date or datetime.now()
        # Advance watermark (source-row id). None on untrained rows.
        row.last_row_id = last_row_id
        # Persist the step-change window so it survives hydration.
        row.recent_window = list(recent_window) if recent_window else None
        # Runtime state: replayed values if given, else reset to baseline.
        row.cusum_pos = cusum_pos if cusum_pos is not None else 0.0
        row.cusum_neg = cusum_neg if cusum_neg is not None else 0.0
        row.ewma_state = ewma_state if ewma_state is not None else baseline_mean

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


def _load_samples_with_dates(db, model: str, metric: str, after=None,
                             after_row_id=None):
    """Return [(file_date, value, row_id)] for a model+metric, oldest first.

    row_id is the source row's autoincrement id (smoothness_results.id for
    max_smoothness_value, track_results.id otherwise) — the advance watermark.

    Filters (advance_drift_state passes exactly one):
      * after_row_id — id strictly greater. Precise: ids always move forward
        on ingest, so same-day samples added after a run are still consumed.
      * after (datetime) — file_date strictly newer. Legacy fallback for state
        rows trained before last_row_id existed; day-granularity file_dates
        make it skip same-day arrivals, so it's used at most once per row.
    """
    out = []
    if metric == "max_smoothness_value":
        with db.session() as s:
            q = s.query(DBSR.file_date, DBSR.max_smoothness_value, DBSR.id).filter(
                DBSR.model == model, DBSR.max_smoothness_value.isnot(None))
            if after_row_id is not None:
                q = q.filter(DBSR.id > after_row_id)
            elif after is not None:
                q = q.filter(DBSR.file_date > after)
            for d, v, rid in q.order_by(DBSR.file_date).all():
                if d is not None and v is not None:
                    out.append((d, v, rid))
        return out

    col = _TRACK_METRIC_COLUMNS.get(metric)
    if col is None:
        return out
    with db.session() as s:
        q = (s.query(DBAR.file_date, col, DBTR.id)
             .join(DBTR, DBTR.analysis_id == DBAR.id)
             .filter(DBAR.model == model, col.isnot(None)))
        if after_row_id is not None:
            q = q.filter(DBTR.id > after_row_id)
        elif after is not None:
            q = q.filter(DBAR.file_date > after)
        for d, v, rid in q.order_by(DBAR.file_date).all():
            if d is not None and v is not None:
                out.append((d, v, rid))
    return out


def _build_detector(metric, baseline_mean, baseline_std, baseline_count, *,
                    thresholds_dict=None, hLz=None,
                    cusum_pos=0.0, cusum_neg=0.0, ewma_state=None,
                    recent_window=None):
    """Construct a MetricDetector from either a {DriftTier:(h,L,z)} dict
    (training) or a triple of per-tier h/L/z dicts (advance, from a DB row)."""
    from collections import deque
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        MetricDetector, STEP_CHANGE_WINDOW)

    if thresholds_dict is not None:
        h = {"WARNING": thresholds_dict[DriftTier.WARNING][0],
             "DRIFT": thresholds_dict[DriftTier.DRIFT][0],
             "OUT_OF_CONTROL": thresholds_dict[DriftTier.OUT_OF_CONTROL][0]}
        L = {"WARNING": thresholds_dict[DriftTier.WARNING][1],
             "DRIFT": thresholds_dict[DriftTier.DRIFT][1],
             "OUT_OF_CONTROL": thresholds_dict[DriftTier.OUT_OF_CONTROL][1]}
        z = {"WARNING": thresholds_dict[DriftTier.WARNING][2],
             "DRIFT": thresholds_dict[DriftTier.DRIFT][2],
             "OUT_OF_CONTROL": thresholds_dict[DriftTier.OUT_OF_CONTROL][2]}
    else:
        h, L, z = hLz

    return MetricDetector(
        metric=metric,
        baseline_mean=baseline_mean or 0.0,
        baseline_std=baseline_std or 0.0,
        baseline_count=baseline_count or 0,
        is_trained=True,
        h_per_tier=h, L_per_tier=L, z_per_tier=z,
        cusum_pos=cusum_pos or 0.0, cusum_neg=cusum_neg or 0.0,
        ewma_state=ewma_state if ewma_state is not None else (baseline_mean or 0.0),
        recent_window=deque(
            [float(v) for v in (recent_window or [])], maxlen=STEP_CHANGE_WINDOW),
    )


def advance_drift_state(db, model: Optional[str] = None) -> int:
    """Advance each trained (model, metric) detector over samples that arrived
    AFTER its last_updated marker, persisting the new cusum/ewma state.

    This is what makes the V6 detector respond to NEW data -- without it the
    runtime state is frozen at training time and get_drifting_models always
    reads Stable. Call it after a processing batch (or from Settings). Returns
    the number of (model, metric) rows actually advanced.
    """
    with db.session() as s:
        q = s.query(ModelMetricState).filter(ModelMetricState.is_trained == True)  # noqa: E712
        if model is not None:
            q = q.filter(ModelMetricState.model == model)
        targets = [(r.model, r.metric) for r in q.all()]

    advanced = 0
    for mdl, metric in targets:
        with db.session() as s:
            row = s.query(ModelMetricState).filter(
                ModelMetricState.model == mdl,
                ModelMetricState.metric == metric,
            ).first()
            if row is None or not row.is_trained or row.baseline_std is None:
                continue
            # Prefer the row-id watermark (exact). Date fallback only for rows
            # trained before last_row_id existed — it skips same-day arrivals
            # once, then the watermark takes over below.
            if row.last_row_id is not None:
                new_samples = _load_samples_with_dates(
                    db, mdl, metric, after_row_id=row.last_row_id)
            else:
                new_samples = _load_samples_with_dates(
                    db, mdl, metric, after=row.last_updated)
            if not new_samples:
                continue
            det = _build_detector(
                metric, row.baseline_mean, row.baseline_std, row.baseline_count,
                hLz=(
                    {"WARNING": row.h_warning or 0.0, "DRIFT": row.h_drift or 0.0,
                     "OUT_OF_CONTROL": row.h_oc or 0.0},
                    {"WARNING": row.L_warning or 0.0, "DRIFT": row.L_drift or 0.0,
                     "OUT_OF_CONTROL": row.L_oc or 0.0},
                    {"WARNING": row.z_warning or 0.0, "DRIFT": row.z_drift or 0.0,
                     "OUT_OF_CONTROL": row.z_oc or 0.0},
                ),
                cusum_pos=row.cusum_pos, cusum_neg=row.cusum_neg, ewma_state=row.ewma_state,
                recent_window=row.recent_window,
            )
            # Winsorize like training replay: suspect-scale values get one
            # bounded push, never ownership of CUSUM (see SUSPECT_SIGMA_GATE).
            from laser_trim_analyzer.ml.drift_types import SUSPECT_SIGMA_GATE
            c_lo = row.baseline_mean - SUSPECT_SIGMA_GATE * row.baseline_std
            c_hi = row.baseline_mean + SUSPECT_SIGMA_GATE * row.baseline_std
            for _d, v, _r in new_samples:
                det.update(min(max(float(v), c_lo), c_hi))
            row.cusum_pos = det.cusum_pos
            row.cusum_neg = det.cusum_neg
            row.ewma_state = det.ewma_state
            row.recent_window = list(det.recent_window) or None
            # Samples are date-ordered; take explicit maxes (a backfill of
            # old-dated files can put the newest id mid-list and vice versa).
            row.last_updated = max(d for (d, _v, _r) in new_samples)
            row.last_row_id = max(
                [rid for (_d, _v, rid) in new_samples] + [row.last_row_id or 0])
            s.commit()
            advanced += 1
    return advanced
