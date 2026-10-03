"""Spec 2 — training orchestration for the multi-metric drift detector.

train_drift_detector reads historical data per model, computes baselines
and per-tier thresholds, and writes/upserts model_metric_state rows.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAR,
    ModelMetricState,
    SmoothnessResult as DBSR,
    TrackResult as DBTR,
)
from laser_trim_analyzer.ml.drift_types import (
    DriftTier,
    RECENT_LOT_DAYS,
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

# Minimum FT→trim match confidence for a linked pair to count in the ESCAPE
# watch. 0.5 ≈ exact-serial links within ~4 months (decay reaches 0.40 at the
# 180-day window edge). The agreement/overkill queries stay at their stricter
# 0.70; escapes tolerate more distance because assembly dwell is months and
# a wrong-unit link at the same model still measures the model's escape rate.
ESCAPE_MIN_CONFIDENCE: float = 0.5


# ---- what may feed drift (2026-10-02) -------------------------------------------------------
# The rules the stored drift state was built under. Bump it whenever a change here would give a
# different state from the same data: `ensure_drift_rules` then retrains every model once, at the
# next start (about 11 s on the home copy), so nobody has to remember to.
DRIFT_RULES_VERSION = "2026-10-02"
DRIFT_RULES_KEY = "drift_rules"

# A reading that cannot be real never feeds drift, and is counted (drift_exclusions). From the
# home copy, 2026-10-02: 37 untrimmed resistances >= 10 MOhm or <= 0 on files that did NOT fail
# processing, and 31 resistance changes computed from such a resistance (up to 5.6e10 %); 4
# electrical angles above 400 degrees. The ceiling is the one the findings use (ink_target MAX_R).
FAILED_READING_MARKER = 999.999     # the analyser's sentinel (core/model_stats.py)
RESISTANCE_CEILING_OHMS = 1e7
ANGLE_CEILING_DEG = 400.0
# NOT a floor at 0 degrees: the 2,416 negative angles on the home copy are -0.1 on the 8340
# family and 7715, whose angles all sit near zero (0.4-3.3 on average), and core/model_stats.py
# protects them for the same reason. The angle is evidence-only (not a trigger) either way.

LEFT_OUT_REASONS = ("suspect", "impossible")


def _impossible_resistance(r) -> bool:
    return r is not None and not (0.0 < r < RESISTANCE_CEILING_OHMS)


def reading_is_impossible(metric: str, value, untrimmed_r=None, trimmed_r=None) -> bool:
    """Can `value` not be a real reading of `metric`? NaN is MISSING, not impossible (the lot
    builder drops it uncounted, as before); +-inf and the 999.999 marker are impossible
    everywhere."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if math.isinf(value) or value == FAILED_READING_MARKER:
        return True
    if metric == "measured_electrical_angle":
        return value > ANGLE_CEILING_DEG
    if metric == "untrimmed_resistance":
        return _impossible_resistance(value)
    if metric == "resistance_change_percent":
        # A change is only as real as the two resistances it is made from.
        return _impossible_resistance(untrimmed_r) or _impossible_resistance(trimmed_r)
    return False


def _left_out_reason(metric: str, value, data_quality, untrimmed_r=None, trimmed_r=None):
    """Why a sample is left out of drift ("suspect", "impossible"), or None to keep it."""
    if data_quality == "suspect":
        return "suspect"
    if reading_is_impossible(metric, value, untrimmed_r, trimmed_r):
        return "impossible"
    return None


def _count(left_out, reason) -> None:
    if left_out is not None:
        left_out[reason] = left_out.get(reason, 0) + 1


def drift_exclusions(db, model: str) -> Dict[str, Dict[str, int]]:
    """How many of `model`'s samples each watched metric left out, and why: {metric:
    {"suspect": n, "impossible": n}} for EVERY watched metric (zeros included). Counted over the
    window drift itself reads (after a baseline requalification, when there is one)."""
    floor = None
    try:
        req = db.get_baseline_requalification(model)
        if req:
            floor = datetime.fromisoformat(str(req[0])[:19])
    except Exception:
        logger.exception("baseline requalification lookup failed for %s", model)
    out: Dict[str, Dict[str, int]] = {}
    for metric in WATCHED_METRICS:
        counts = {reason: 0 for reason in LEFT_OUT_REASONS}
        _load_samples_with_dates(db, model, metric,
                                 after=(floor - timedelta(seconds=1)) if floor else None,
                                 left_out=counts)
        out[metric] = counts
    return out


def _coerce_dt(d):
    """SQLite returns COALESCE(datetime, datetime) as a raw string — the
    expression loses the column's DateTime result processor."""
    if isinstance(d, str):
        return datetime.fromisoformat(d[:19])
    return d


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
    # Captured now: the loop below rebinds `model` to each model's name in turn.
    full_run = model is None

    # Discover models that have data. FINAL-TEST models are included since
    # the FT watch (2026-07-13): a model tested-but-never-trimmed in this DB
    # (e.g. 8864 — 213 recent FT records, zero trim rows) still deserves
    # ft_fail_fraction monitoring; its trim metrics train NOT TRAINED.
    from laser_trim_analyzer.database.models import FinalTestResult as DBFT
    with db.session() as s:
        models = [
            r[0] for r in s.query(DBAR.model).distinct().all()
        ]
        smoothness_models = [
            r[0] for r in s.query(DBSR.model).distinct().all()
        ]
        ft_models = [
            r[0] for r in s.query(DBFT.model).distinct().all()
        ]
        all_models = sorted((set(models) | set(smoothness_models) | set(ft_models))
                            - {None, ""})
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

    if full_run:
        # Every model's state is now built under the current rules.
        _record_drift_rules(db)

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

    LOT MODE (2026-07-13, James's redesign): the observation is a production
    LOT (day-cluster, gap > LOT_GAP_DAYS), value = lot MEDIAN. Baseline
    mean/σ come from historical lot medians, so ordinary lot-to-lot wander is
    the calibrated noise floor and an alarm means "this lot sits outside the
    model's own lot history" — one verdict per lot instead of one per unit
    (a 50-unit shifted lot used to push CUSUM 50 times: the 44-flagged pile).
    Requalification floors the LOT history at the effective date. An OPEN lot
    (still inside the changeover gap) is never fed to detector state.

    Returns True if the row was written with is_trained=True.
    """
    from laser_trim_analyzer.ml.lots import (
        LOT_GAP_DAYS, MIN_LOT_BASELINE_N, MIN_LOTS_TRAIN, REPLAY_LOTS,
        get_model_lots, judged_lots)

    lots = get_model_lots(db, model, metric, after=baseline_start)
    closed = [l for l in lots if not l.is_open()]

    if len(closed) < MIN_LOTS_TRAIN:
        _upsert_metric_state(
            db, model, metric,
            baseline_mean=None, baseline_std=None,
            baseline_count=len(closed), is_trained=False, thresholds=None,
        )
        return False

    # Replayed: the newest REPLAY_LOTS lots the detector JUDGES (a small lot pooled with the one
    # after it, the newest small lot left waiting -- lots.judged_lots), always leaving at least
    # MIN_LOTS_TRAIN - REPLAY_LOTS lots for the baseline. Counted on JUDGED lots (2026-10-02):
    # counted on raw lots, a waiting 2-unit lot took a replay slot and pushed one more real lot
    # of an ongoing drift into the baseline -- 8889's resistance, rising since July, fell just
    # under the line that way.
    from laser_trim_analyzer.ml.lots import MEAN_AGGREGATED_METRICS as _MEAN
    judged = judged_lots(closed, use_mean=metric in _MEAN)
    n_replay = min(REPLAY_LOTS, max(0, len(judged) - (MIN_LOTS_TRAIN - REPLAY_LOTS)))
    replay_lots = judged[len(judged) - n_replay:] if n_replay else []
    # Baseline = every closed lot that ends before the first replayed lot begins; tiny lots
    # (n < MIN_LOT_BASELINE_N) are left out of its statistics -- their medians are too noisy to
    # calibrate sigma -- unless the model has hardly any other kind.
    pool = [l for l in closed if not replay_lots or l.end < replay_lots[0].start]
    baseline_lots = [l for l in pool if l.n >= MIN_LOT_BASELINE_N]
    if len(baseline_lots) < 5:
        baseline_lots = pool                  # tiny-lot model: use what exists
    baseline_cutoff = pool[-1].end if pool else closed[0].end

    arr = np.asarray([l.median for l in baseline_lots], dtype=float)
    baseline_mean = float(np.mean(arr))
    baseline_std = float(np.std(arr, ddof=1))
    # σ floor: a model whose historical lot medians are nearly identical
    # (quantized values, short history) yields σ≈0, and the first ordinary
    # change reads as +250σ — absurd numbers that erode trust. Floor lot-σ
    # at 1% of |mean| (plus a tiny absolute), standard guard against an
    # underestimated σ. Tiers stay correct; displayed shifts stay sane.
    baseline_std = max(baseline_std, 0.01 * abs(baseline_mean), 1e-9)
    from laser_trim_analyzer.ml.lots import MEAN_AGGREGATED_METRICS
    if metric in MEAN_AGGREGATED_METRICS:
        # Fractions live in [0,1]; a historically-clean model has mean≈0 and
        # σ≈0, so floor at 2 percentage points: a 20%-fail lot on a clean
        # model reads +9σ (alarms, sanely) instead of +∞. Applies to all
        # fail/escape-fraction metrics (trim linearity, final test, escapes).
        baseline_std = max(baseline_std, 0.02)

    thresholds = corrected_tier_thresholds(sensitivity_preset, baseline_std)

    # Replay newest closed lots so persisted runtime state reflects drift
    # already in history. Lot medians are robust to corrupt units, but clip
    # to the suspect gate anyway: a wholly-corrupt lot contributes one
    # bounded push, a real sustained shift still accumulates.
    from laser_trim_analyzer.ml.drift_types import SUSPECT_SIGMA_GATE
    clip_lo = baseline_mean - SUSPECT_SIGMA_GATE * baseline_std
    clip_hi = baseline_mean + SUSPECT_SIGMA_GATE * baseline_std
    det = _build_detector(metric, baseline_mean, baseline_std, len(baseline_lots),
                          thresholds_dict=thresholds)
    _feed(det, replay_lots, baseline_cutoff, clip_lo, clip_hi)

    _upsert_metric_state(
        db, model, metric,
        baseline_mean=baseline_mean, baseline_std=baseline_std,
        baseline_count=len(baseline_lots), is_trained=True, thresholds=thresholds,
        cusum_pos=det.cusum_pos, cusum_neg=det.cusum_neg, ewma_state=det.ewma_state,
        baseline_cutoff_date=baseline_lots[-1].end if baseline_lots else None,
        # The lot watermark: advance feeds only JUDGED lots ending after this. A newest small
        # lot still waiting for company ends after it, so it is pooled and judged later.
        last_sample_date=replay_lots[-1].end if replay_lots else baseline_cutoff,
        last_row_id=None,
        recent_window=list(det.recent_window),
    )
    return True


def _feed(det, lots, previous_end, clip_lo, clip_hi) -> None:
    """Feed judged lots to a detector, oldest first. Winsorized at the suspect gate, so one
    corrupt lot is one bounded push. After a pause of more than RECENT_LOT_DAYS the detector
    starts again from the baseline (2026-10-02): lots that far apart do not add up -- 6952's
    escape alarm was built from lots spread over 2021-2026."""
    for lot in lots:
        if previous_end is not None and (lot.start - previous_end).days > RECENT_LOT_DAYS:
            det.reset_runtime()
        det.update(min(max(float(lot.median), clip_lo), clip_hi))
        previous_end = lot.end


def _record_drift_rules(db) -> None:
    meta_set = getattr(db, "_meta_set", None)
    if meta_set is None:
        return
    with db.session() as s:
        meta_set(s, DRIFT_RULES_KEY, DRIFT_RULES_VERSION)


def ensure_drift_rules(db, sensitivity_preset: str = "standard",
                       progress_callback: Optional[Callable[[str, int, int], None]] = None) -> bool:
    """Retrain every model's drift state once if it was built under other rules than these
    (DRIFT_RULES_VERSION); True when it retrained. A database with no drift state has nothing
    to retrain -- what is trained later is trained under these rules -- so the version is just
    recorded. Called at startup, before the catch-up advance."""
    meta_get = getattr(db, "_meta_get", None)
    if meta_get is None:
        return False
    with db.session() as s:
        recorded = meta_get(s, DRIFT_RULES_KEY)
        has_state = s.query(ModelMetricState.id).first() is not None
    if recorded == DRIFT_RULES_VERSION:
        return False
    if not has_state:
        _record_drift_rules(db)
        return False
    logger.info("Drift rules changed (%s -> %s): retraining every model's drift state",
                recorded, DRIFT_RULES_VERSION)
    train_drift_detector(db, sensitivity_preset=sensitivity_preset,
                         progress_callback=progress_callback)
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
                             after_row_id=None, left_out: Optional[Dict[str, int]] = None):
    """Return [(file_date, value, row_id)] for a model+metric, oldest first.

    Never a sample from a file marked SUSPECT, nor a reading that cannot be real
    (`reading_is_impossible`); each one left out is counted into `left_out[reason]` when a dict
    is given (drift_exclusions). 7539-2's newest lot was four suspect files at 6.8 against a
    0.02 band; 8902's alarm was one suspect unit at 9.93 against 0.1 (2026-10-02).

    row_id is the source row's autoincrement id (smoothness_results.id for
    max_smoothness_value, track_results.id otherwise) — the advance watermark.

    Filters (advance_drift_state passes exactly one):
      * after_row_id — id strictly greater. Precise: ids always move forward
        on ingest, so same-day samples added after a run are still consumed.
      * after (datetime) — file_date strictly newer. Legacy fallback for state
        rows trained before last_row_id existed; day-granularity file_dates
        make it skip same-day arrivals, so it's used at most once per row.
    """
    if metric == "linearity_fail_fraction":
        # Per-UNIT linearity fail flag (1=FAIL, 0=accepted); the lot pipeline
        # aggregates these by MEAN into the lot's fail fraction. ERROR and
        # UNTRIMMED records are not gradeable and are excluded.
        from laser_trim_analyzer.database.models import StatusType
        with db.session() as s:
            q = (s.query(DBAR.file_date, DBAR.overall_status, DBAR.id, DBAR.data_quality)
                 .filter(DBAR.model == model,
                         DBAR.overall_status.in_([StatusType.PASS, StatusType.WARNING,
                                                  StatusType.FAIL])))
            if after_row_id is not None:
                q = q.filter(DBAR.id > after_row_id)
            elif after is not None:
                q = q.filter(DBAR.file_date > after)
            rows = q.order_by(DBAR.file_date).all()
        out = []
        for d, st_, rid, dq in rows:
            if d is None:
                continue
            if dq == "suspect":            # its verdict rests on the reading that is suspect
                _count(left_out, "suspect")
                continue
            out.append((d, 1.0 if getattr(st_, "name", str(st_)) == "FAIL" else 0.0, rid))
        return out

    if metric == "ft_fail_fraction":
        # Per-FT-RECORD fail flag on final_test_results, clustered on the
        # FINAL TEST date (the lot at the last station, not the trim lot).
        # Blind spot closed 2026-07-13: the watch previously had no eyes on
        # the most expensive failure point. COALESCE(test_date, file_date):
        # some FT files never parse a test_date cell — file_date covers them
        # (code-review finding #4). Dates before 2000 are file artifacts
        # (1899-12-30 epoch defaults) and are excluded.
        from sqlalchemy import func as _fn
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFT, StatusType)
        floor_2000 = datetime(2000, 1, 1)
        ft_date = _fn.coalesce(DBFT.test_date, DBFT.file_date)
        with db.session() as s:
            q = (s.query(ft_date, DBFT.overall_status, DBFT.id)
                 .filter(DBFT.model == model,
                         ft_date.isnot(None),
                         ft_date > floor_2000,
                         DBFT.overall_status.in_([StatusType.PASS, StatusType.WARNING,
                                                  StatusType.FAIL])))
            if after_row_id is not None:
                q = q.filter(DBFT.id > after_row_id)
            elif after is not None:
                q = q.filter(ft_date > after)
            rows = q.order_by(ft_date).all()
        return [(_coerce_dt(d), 1.0 if getattr(st_, "name", str(st_)) == "FAIL" else 0.0, rid)
                for (d, st_, rid) in rows if d is not None]

    if metric == "escape_fraction":
        # Of FT records confidently linked to a trim the app ACCEPTED
        # (PASS or WARNING), the flag is 1 when final test then FAILED —
        # an escape. Lot mean = the lot's escape rate. Requires link
        # confidence ≥ ESCAPE_MIN_CONFIDENCE so recycled-serial guesses
        # don't fabricate escapes.
        from sqlalchemy import func as _fn
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFT, StatusType)
        floor_2000 = datetime(2000, 1, 1)
        ft_date = _fn.coalesce(DBFT.test_date, DBFT.file_date)
        with db.session() as s:
            q = (s.query(ft_date, DBFT.overall_status, DBFT.id, DBAR.data_quality)
                 .join(DBAR, DBFT.linked_trim_id == DBAR.id)
                 .filter(DBFT.model == model,
                         ft_date.isnot(None),
                         ft_date > floor_2000,
                         DBFT.match_confidence >= ESCAPE_MIN_CONFIDENCE,
                         DBFT.overall_status.in_([StatusType.PASS, StatusType.FAIL]),
                         DBAR.overall_status.in_([StatusType.PASS, StatusType.WARNING])))
            if after_row_id is not None:
                q = q.filter(DBFT.id > after_row_id)
            elif after is not None:
                q = q.filter(ft_date > after)
            rows = q.order_by(ft_date).all()
        out = []
        for d, st_, rid, trim_dq in rows:
            if d is None:
                continue
            if trim_dq == "suspect":       # the trim verdict it tests rests on a suspect file
                _count(left_out, "suspect")
                continue
            out.append((_coerce_dt(d), 1.0 if getattr(st_, "name", str(st_)) == "FAIL" else 0.0,
                        rid))
        return out

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
        # A record that failed processing carries the analyser's 999.999 marker in these
        # columns, not a reading. A lot median usually shrugs one off -- but 8856's lot of
        # 2024-08-06 is 4 markers out of 7 tracks, so its median WAS 999.999.
        from laser_trim_analyzer.core.model_stats import failed_processing_statuses
        q = (s.query(DBAR.file_date, col, DBTR.id, DBAR.data_quality,
                     DBTR.untrimmed_resistance, DBTR.trimmed_resistance)
             .join(DBTR, DBTR.analysis_id == DBAR.id)
             .filter(DBAR.model == model, col.isnot(None),
                     DBAR.overall_status.notin_(failed_processing_statuses())))
        if after_row_id is not None:
            q = q.filter(DBTR.id > after_row_id)
        elif after is not None:
            q = q.filter(DBAR.file_date > after)
        for d, v, rid, dq, r_untrimmed, r_trimmed in q.order_by(DBAR.file_date).all():
            if d is None or v is None:
                continue
            reason = _left_out_reason(metric, v, dq, r_untrimmed, r_trimmed)
            if reason:
                _count(left_out, reason)
                continue
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
            # LOT MODE (last_row_id is None on lot-trained rows): recompute
            # lots and feed only CLOSED lots ending after the lot watermark
            # (row.last_updated). Legacy unit-mode rows (last_row_id set)
            # keep the old per-sample path until their next retrain.
            lot_mode = row.last_row_id is None
            if lot_mode:
                from laser_trim_analyzer.ml.lots import get_model_lots
                floor = None
                try:
                    req = db.get_baseline_requalification(mdl)
                    if req:
                        floor = datetime.fromisoformat(str(req[0])[:19])
                except Exception:
                    pass
                from laser_trim_analyzer.ml.lots import (
                    MEAN_AGGREGATED_METRICS as _MEAN, judged_lots)
                lots = get_model_lots(db, mdl, metric, after=floor)
                judged = judged_lots([l for l in lots if not l.is_open()],
                                     use_mean=metric in _MEAN)
                new_lots = [l for l in judged
                            if row.last_updated is None or l.end > row.last_updated]
                if not new_lots:
                    continue
                new_samples = [(l.end, l.median, None) for l in new_lots]
            else:
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
            if lot_mode:
                # Judged lots, with a fresh start after a long pause (_feed).
                _feed(det, new_lots, row.last_updated, c_lo, c_hi)
            else:
                for _d, v, _r in new_samples:
                    det.update(min(max(float(v), c_lo), c_hi))
            row.cusum_pos = det.cusum_pos
            row.cusum_neg = det.cusum_neg
            row.ewma_state = det.ewma_state
            row.recent_window = list(det.recent_window) or None
            # Samples are date-ordered; take explicit maxes (a backfill of
            # old-dated files can put the newest id mid-list and vice versa).
            row.last_updated = max(d for (d, _v, _r) in new_samples)
            if not lot_mode:
                row.last_row_id = max(
                    [rid for (_d, _v, rid) in new_samples] + [row.last_row_id or 0])
            s.commit()
            advanced += 1
    return advanced
