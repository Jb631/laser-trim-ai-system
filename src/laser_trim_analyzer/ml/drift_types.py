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
    "untrimmed_error_max",          # strongest validated early-warning signal
    "untrimmed_sigma_gradient",
    "untrimmed_resistance",
    "linearity_error",
    "measured_electrical_angle",
    "trim_pass_count",
    "resistance_change_percent",
    "max_smoothness_value",
    "composite_trim_risk_score",
    # Per-lot LINEARITY FAIL FRACTION (2026-07-13, James): a lot can degrade
    # by failing MORE UNITS while its metric medians barely move (medians are
    # deliberately robust). This watches the outcome itself: the fraction of
    # the lot's gradeable units that failed linearity, vs the model's
    # historical lot fail rates.
    "linearity_fail_fraction",
    # FINAL-TEST side (2026-07-13, James): the trim-side watch is blind to the
    # most expensive station. These two cluster lots on final_test_results
    # test_date, not trim file_date.
    #   ft_fail_fraction — fraction of the FT lot's records that FAILED final
    #   test, vs the model's historical FT lot fail rates. Catches "the latest
    #   lot is failing final test out of family" after all labor is invested.
    "ft_fail_fraction",
    #   escape_fraction — of FT records CONFIDENTLY linked (≥0.5) to a trim
    #   the app graded PASS, the fraction that then FAILED final test. Catches
    #   the trim verdict losing its predictive power (assembly damage,
    #   station-to-station offset) — a signal neither station sees alone.
    "escape_fraction",
)

# Metrics whose drift may RAISE a model's tier (generate a flag). Chosen from the
# 2026-06 group-level validation of "does failure-ward drift actually predict a higher
# failure rate?": untrimmed_error_max (+14% at 1-2σ, +24% at 2-3σ — the best signal),
# untrimmed_resistance / resistance_change_percent (+18-19% at large drift), the
# composite (where deployed), and linearity_error (the direct outcome). Metrics NOT in
# this set are EVIDENCE-ONLY — still computed and shown on the Model page, but they no
# longer flag, because their drift did NOT predict failures: measured_electrical_angle
# (negative lift at every level — drift predicts FEWER linearity fails), and the noisy
# untrimmed_sigma_gradient / trim_pass_count / max_smoothness_value. (Electrical angle
# is its own customer spec; this only removes it as a LINEARITY-failure early-warning.)
TRIGGER_METRICS: frozenset = frozenset({
    "untrimmed_error_max",
    "untrimmed_resistance",
    "resistance_change_percent",
    "linearity_error",
    "composite_trim_risk_score",
    "linearity_fail_fraction",     # the outcome itself — always a trigger
    "ft_fail_fraction",            # the FINAL outcome — always a trigger
    "escape_fraction",             # trim verdict losing predictive power
})

# Which way is WORSE, per watched metric (2026-10-02, James: "im also concerned about dirty data
# and accuracy of the app telling me things are drifting"). +1: higher is worse -- more error, more
# fails, more escapes, more trim effort, more risk. 0: both ways are a change in the process --
# resistance and electrical angle have no "better" side. A move the BETTER way on a one-sided
# metric is good news, never an alarm: 8877 was flagged because its untrimmed error FELL from
# 0.11 to 0.05. (-1, lower is worse, is allowed; no metric needs it today.)
WORSE_DIRECTION: Dict[str, int] = {
    "untrimmed_error_max": 1,
    "untrimmed_sigma_gradient": 1,
    "untrimmed_resistance": 0,
    "linearity_error": 1,
    "measured_electrical_angle": 0,
    "trim_pass_count": 1,
    "resistance_change_percent": 0,
    "max_smoothness_value": 1,
    "composite_trim_risk_score": 1,
    "linearity_fail_fraction": 1,
    "ft_fail_fraction": 1,
    "escape_fraction": 1,
}
assert set(WORSE_DIRECTION) == set(WATCHED_METRICS), \
    "every watched metric needs a WORSE_DIRECTION"

# Only RECENT evidence raises an alarm (2026-10-02). Lots more than this many days apart do not
# add up -- after such a pause the detector starts again from the baseline -- and a metric whose
# newest judged lot is more than this many days older than the newest file in the database
# cannot raise the model's tier. 2511 was flagged on final-test lots from 2020-21; 6952 on escape
# lots spread over 2021-2026.
RECENT_LOT_DAYS = 90

# Fraction-valued metrics (0..1). Their lot observation is the MEAN (a rate,
# not a median — the median of 0/1 flags is uselessly 0 or 1) and every
# display renders them as PERCENT. Single source: lots.py aliases this as
# MEAN_AGGREGATED_METRICS.
FRACTION_METRICS: frozenset = frozenset({
    "linearity_fail_fraction",
    "ft_fail_fraction",
    "escape_fraction",
})

# Display grouping (2026-07-13, James: "the design needs clear sections for
# what im looking at"). Three groups: early-warning process signals at the
# trim station, the trim outcome, and the final-test outcome after assembly.
# Drift tab, pill row, and Settings all render in this order with these
# headings — the 12-metric wall reads as three questions instead.
METRIC_GROUPS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("Process signals — element & trim",
     "Early warning: what the elements and the trim process look like",
     ("untrimmed_error_max", "untrimmed_sigma_gradient", "untrimmed_resistance",
      "linearity_error", "measured_electrical_angle", "trim_pass_count",
      "resistance_change_percent", "max_smoothness_value",
      "composite_trim_risk_score")),
    ("Outcome — linearity at trim",
     "Share of each lot's units failing the customer linearity spec",
     ("linearity_fail_fraction",)),
    ("Outcome — final test (after assembly)",
     "The last, most expensive station: lot fail rate and escapes",
     ("ft_fail_fraction", "escape_fraction")),
)

# Grouped surfaces (drift tab, pills, evidence sheet) derive their metric set
# from METRIC_GROUPS — a watched metric missing from every group would train
# and flag yet silently vanish from all of them (code-review finding #10).
# Fail at import, not at a customer meeting.
assert {m for _t, _g, _ms in METRIC_GROUPS for m in _ms} == set(WATCHED_METRICS), \
    "METRIC_GROUPS must cover exactly WATCHED_METRICS"


def format_metric_value(metric: str, value, fmt_measure=None) -> str:
    """Render a metric value for display: percent for fraction metrics,
    otherwise the caller's numeric formatter (or a plain %g fallback)."""
    if value is None:
        return "—"
    if metric in FRACTION_METRICS:
        return f"{value * 100:.1f}%"
    return fmt_measure(value) if fmt_measure else f"{value:g}"


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
    # The end of the newest lot this metric's detector has JUDGED (its advance watermark), and
    # whether that is recent enough to alarm (RECENT_LOT_DAYS). A stale metric reads STABLE
    # whatever its state says; `newest_lot` lets a page say how old its evidence is.
    newest_lot: Optional[datetime] = None
    is_recent: bool = True


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
    # A `sigma_shift` field lived here until 2026-08-31. Only get_triage_alerts
    # ever populated it and only its ordering helper ever read it; both went
    # when the FOCUS list replaced that feed. The equivalent "how far has it
    # actually moved" number is computed where it is still shown — evidence.py
    # `_sigma_shift` for the export, and the detector's own worst-metric pick.


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
# Robustness gate shared by the drift engine and evidence surfaces: values
# beyond this many baseline-σ are treated as suspect DATA, not process signal
# (observed: linearity errors of 10.0 against 0.03±0.026 — ~380σ). The
# detector WINSORIZES (clips) such samples so one corrupt point can't own
# CUSUM; evidence displays EXCLUDE and disclose them.
SUSPECT_SIGMA_GATE = 8.0

METRIC_LABELS = {
    "linearity_fail_fraction": "Lot fail rate (linearity)",
    "ft_fail_fraction": "Final-test lot fail rate",
    "escape_fraction": "Escape rate (trim PASS → FT FAIL)",
    "untrimmed_error_max": "Untrimmed error (max)",
    "sigma_gradient": "Sigma gradient (post-trim)",
    "untrimmed_sigma_gradient": "Sigma gradient (untrimmed)",
    "untrimmed_resistance": "Untrimmed resistance",
    "linearity_error": "Linearity error",
    "measured_electrical_angle": "Electrical angle",
    "trim_pass_count": "Trim pass count",
    "resistance_change_percent": "Resistance change %",
    "max_smoothness_value": "Smoothness (max)",
    # Short enough for the Model page's 9-pill row without truncation
    # (the old "Composite trim-risk" clipped to "omposite trim-ris").
    "composite_trim_risk_score": "Composite risk",
}


def metric_label(metric: str) -> str:
    """Human-readable label for a metric key (graceful passthrough)."""
    return METRIC_LABELS.get(metric, metric)
