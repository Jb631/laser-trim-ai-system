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
#
# D-SIGMA: post-trim `sigma_gradient` is intentionally NOT watched for drift.
# It is computed on the trim-CORRECTED curve, so it's a lagging, confounded
# signal for element-production drift. The upstream `untrimmed_sigma_gradient`
# (raw sweep) is the correct process signal. Post-trim sigma remains only a
# finished-unit quality gate (the per-model threshold optimizer), not a drift
# metric. (7 metrics -> 7 glance pills on the Model page.)
WATCHED_METRICS: Tuple[str, ...] = (
    "untrimmed_sigma_gradient",
    "untrimmed_resistance",
    "linearity_error",
    "measured_electrical_angle",
    "trim_pass_count",
    "resistance_change_percent",
    "max_smoothness_value",
    "composite_trim_risk_score",
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
    # Signed shift of the recent data from baseline, in std-dev units:
    # (recent_mean - baseline_mean) / baseline_std on the worst metric. This is the
    # honest, human-readable "how far has it actually moved" number — unlike
    # `magnitude` (CUSUM/EWMA distance past the control limit), which can read in the
    # hundreds for a sub-1σ shift. None when recent data isn't available to confirm.
    sigma_shift: Optional[float] = None


@dataclass
class ModelSummary:
    """Compact per-model row for the Triage browse zone (Spec 3b)."""
    model: str
    tier: DriftTier
    last_processed: Optional[datetime] = None


@dataclass
class TrainingSummary:
    """Returned by train_drift_detector for progress reporting."""
    models_trained: int
    metrics_per_model: int = 8
    skipped_insufficient_data: List[Tuple[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0


# Human-readable labels for every metric the UI can surface (cards, pills,
# the drift table, exports).  Single source of truth so no page renders a raw
# key like ``untrimmed_resistance``.  Covers all WATCHED_METRICS plus the
# post-trim ``sigma_gradient`` quality-gate metric, which is no longer drift-
# watched but can still appear in per-model diagnostics.
METRIC_LABELS = {
    "sigma_gradient": "Sigma gradient (post-trim)",
    "untrimmed_sigma_gradient": "Sigma gradient (untrimmed)",
    "untrimmed_resistance": "Untrimmed resistance",
    "linearity_error": "Linearity error",
    "measured_electrical_angle": "Electrical angle",
    "trim_pass_count": "Trim pass count",
    "resistance_change_percent": "Resistance change %",
    "max_smoothness_value": "Smoothness (max)",
    "composite_trim_risk_score": "Composite trim-risk",
}


def metric_label(metric: str) -> str:
    """Human-readable label for a metric key (graceful passthrough)."""
    return METRIC_LABELS.get(metric, metric)
