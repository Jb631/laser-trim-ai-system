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
from datetime import datetime
from typing import Deque, Dict, Optional

import numpy as np
from scipy import stats

from laser_trim_analyzer.ml.drift_types import (
    AlertType,
    DriftTier,
    MetricStatus,
    ModelAlertSummary,
    ModelDriftStatus,
    TRIGGER_METRICS,
    WATCHED_METRICS,
)


# EWMA smoothing constant.  Fixed in v1 per spec.
EWMA_LAMBDA: float = 0.2

# CUSUM allowance (k).  Half the baseline std.  Fixed in v1 per spec.
CUSUM_K_SIGMAS: float = 0.5

# Step-change window size N (number of recent samples averaged).
STEP_CHANGE_WINDOW: int = 5

# A baseline whose standard deviation is effectively zero (every baseline sample
# identical to float precision) cannot be monitored with a z-score / EWMA / CUSUM
# control chart: the control limits (h ∝ σ, L·σ_ewma ∝ σ) collapse toward zero, so
# any later value reads as astronomically many sigma — observed in production as
# +1.99e11 σ on a constant electrical-angle baseline and +5.2e8 σ on a constant
# trim-pass-count baseline. Such a metric is *non-monitorable*, not "wildly out of
# control"; treating it as STABLE keeps degenerate metrics from dominating Triage
# with garbage magnitudes (the #1 "it flags everything" complaint). Degeneracy is
# judged scale-invariantly: σ must clear a tiny fraction of the baseline mean's
# magnitude (coefficient of variation), with an absolute floor for near-zero means.
# The smallest *legitimate* baseline σ observed across all watched metrics is
# ~1.5e-4, far above what this guard rejects.
BASELINE_DEGENERATE_CV: float = 1e-6
BASELINE_DEGENERATE_MEAN_FLOOR: float = 1e-9


def is_degenerate_baseline(mean: Optional[float], std: Optional[float]) -> bool:
    """True when a baseline's spread is too small to support drift monitoring."""
    if std is None or not math.isfinite(std) or std <= 0.0:
        return True
    ref = max(abs(mean) if mean is not None else 0.0, BASELINE_DEGENERATE_MEAN_FLOOR)
    return std <= BASELINE_DEGENERATE_CV * ref


# Composite-as-primary drift signal. The composite trim-risk score is a logistic blend
# of the trim-effort family (untrimmed_error_max + these three watched features). Where
# it's DEPLOYED for a model, it is the primary trim-effort/failure-risk drift signal
# ("its per-group trend is the upstream-drift early warning", composite-risk design) and
# its constituent family features are demoted to EVIDENCE — they don't independently
# raise the model's tier. Reason: they're redundant with each other ("error_max + sigma
# give no lift") and were the dominant source of single-metric false positives. The
# orthogonal signals (untrimmed_resistance, measured_electrical_angle, linearity_error,
# max_smoothness_value) are different phenomena and still trigger on their own.
COMPOSITE_METRIC = "composite_trim_risk_score"
# The composite's input features. Where the composite is deployed it represents them,
# so they don't independently trigger (they would double-count and are redundant —
# "error_max + sigma give no lift"). They still trigger individually on models WITHOUT
# a deployed composite (e.g. error_max is the best standalone signal there).
COMPOSITE_FAMILY = frozenset({
    "untrimmed_error_max", "untrimmed_sigma_gradient",
    "resistance_change_percent", "trim_pass_count",
})


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

        # If we've never seen a sample, EWMA is uninitialized.  Treat as
        # baseline mean (no deviation) so all checks read as not-tripped.
        ewma_value = self.ewma_state if self.ewma_state is not None else mu

        recent_mean = (
            float(np.mean(list(self.recent_window))) if self.recent_window else None
        )
        recent_n = len(self.recent_window)

        # Non-monitorable baseline guard: a near-constant baseline produces
        # collapsing control limits and absurd sigma magnitudes. Report STABLE
        # rather than letting it flag (and top-rank) on garbage. See
        # is_degenerate_baseline for the rationale and threshold.
        if is_degenerate_baseline(mu, sigma):
            return MetricStatus(
                metric=self.metric,
                tier=DriftTier.STABLE,
                alert_type=None,
                magnitude=0.0,
                baseline_mean=mu,
                baseline_std=sigma,
                recent_mean=recent_mean,
                recent_count=recent_n,
                is_trained=True,
            )

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
                abs(ewma_value - mu) > L * sigma_ewma
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

                # Alert-type tiebreaker: compare normalized magnitudes at
                # the winning tier.  Step's "how far over the trigger" is
                # step_magnitude / z; CUSUM/EWMA's is max(cusum_pos/h,
                # |ewma-mu|/(L*sigma_ewma)).  Whichever is larger drives
                # the alert type.  This makes a sudden 4σ jump read as
                # STEP_CHANGE (step >> cusum at the OC tier) while a slow
                # 50-sample ramp reads as SLOW_DRIFT (CUSUM has integrated
                # past h faster than the recent window can shift).
                step_norm = (step_magnitude / z) if (step_trip and z > 0) else 0.0
                cusum_norm = (max(self.cusum_pos, -self.cusum_neg) / h) if (cusum_trip and h > 0) else 0.0
                ewma_norm = (
                    abs(ewma_value - mu) / (L * sigma_ewma)
                    if (ewma_trip and (L * sigma_ewma) > 0) else 0.0
                )
                slow_norm = max(cusum_norm, ewma_norm)

                if step_norm > slow_norm:
                    winning_alert_type = AlertType.STEP_CHANGE
                    winning_magnitude = step_magnitude
                else:
                    winning_alert_type = AlertType.SLOW_DRIFT
                    winning_magnitude = slow_norm
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
        for metric_name, detector in self.metrics.items():
            if metric_name in sample:
                value = sample[metric_name]
                if value is not None:
                    detector.update(value)
        self.last_processed = datetime.now()
        return self.get_status()

    def get_status(self) -> ModelDriftStatus:
        """Aggregate per-metric status into a model-level status.

        Overall tier = max of metric tiers.  Worst-metric and alert-type
        come from the metric driving the max tier.
        """
        per_metric = {
            name: det.get_status() for name, det in self.metrics.items()
        }

        # Find the worst metric.  Sort by (tier, recent σ-shift) descending.
        # We use |recent_mean - baseline_mean| / baseline_std as the tier
        # tiebreaker rather than MetricStatus.magnitude because magnitude
        # is scaled per alert-type (step uses sqrt(N)*shift/σ; slow_drift
        # uses cusum/h) and is not comparable across metrics.  The σ-shift
        # is the natural common unit for "how far has this metric moved".
        def _sigma_shift(ms) -> float:
            if ms.recent_mean is None or ms.baseline_std is None or ms.baseline_std <= 0:
                return 0.0
            return abs(ms.recent_mean - ms.baseline_mean) / ms.baseline_std

        # Only TRIGGER_METRICS may raise the tier (validated to predict failures);
        # everything else is evidence-only and stays in per_metric for the Model page.
        # Additionally, when the composite is deployed it represents its input features,
        # so those are demoted too (no double-count). See TRIGGER_METRICS / COMPOSITE_*.
        comp = per_metric.get(COMPOSITE_METRIC)
        composite_active = comp is not None and comp.is_trained
        rankable = {
            name: ms for name, ms in per_metric.items()
            if name in TRIGGER_METRICS
            and not (composite_active and name in COMPOSITE_FAMILY)
        }

        ranked = sorted(
            rankable.items(),
            key=lambda kv: (int(kv[1].tier), _sigma_shift(kv[1])),
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
