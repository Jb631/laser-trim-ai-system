"""Lot clustering — the observation unit for drift detection (2026-07-13).

Production runs in lots: a model runs for one or a few consecutive days, the
laser changes over, and the model returns days or weeks later as a NEW lot.
Empirical basis (home DB, 275 models): 41% of consecutive production-day gaps
are 1 day (same lot), with a valley at 4–7 days before the changeover mass —
so a gap of MORE THAN `LOT_GAP_DAYS` days starts a new lot. At that boundary
the data holds 3,683 lots, median 14 units/lot, 105 models with ≥8 lots.

The per-lot observation is the MEDIAN of unit values: robust to the known
corrupt readings (2.1e8 Ω class) without needing a trained baseline to gate
them, and consistent with the daily-median line the charts already draw.

An OPEN lot (last unit newer than LOT_GAP_DAYS ago) may still be receiving
units — it is previewed in the UI but never fed to the detector state.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median, pstdev
from typing import List, Optional, Tuple

LOT_GAP_DAYS = 3          # > this many days between production days = new lot
MIN_LOT_BASELINE_N = 3    # lots smaller than this are scored but not baselined
MIN_LOTS_TRAIN = 8        # fewer lots than this = NOT TRAINED (honest)
REPLAY_LOTS = 3           # newest closed lots replayed through the detector


@dataclass
class Lot:
    start: datetime
    end: datetime
    median: float
    n: int
    unit_std: float          # within-lot spread (context, not alarm basis)

    def is_open(self, now: Optional[datetime] = None) -> bool:
        """Still receiving units? A lot only closes once the changeover gap
        has actually elapsed after its last unit."""
        now = now or datetime.now()
        return (now - self.end) <= timedelta(days=LOT_GAP_DAYS)


def cluster_lots(dated_values: List[Tuple[datetime, float]],
                 gap_days: int = LOT_GAP_DAYS, use_mean: bool = False) -> List[Lot]:
    """Group (date, value) samples into production lots.

    Day-granularity clustering: consecutive production DAYS no more than
    `gap_days` apart belong to one lot. Values must be finite; the caller
    filters None/NaN.
    """
    if not dated_values:
        return []
    by_day: dict = {}
    for d, v in dated_values:
        day = datetime(d.year, d.month, d.day)
        by_day.setdefault(day, []).append(v)
    days = sorted(by_day)
    lots: List[Lot] = []
    cur_days: List[datetime] = []
    cur_vals: List[float] = []
    last = None
    for day in days:
        if last is not None and (day - last).days > gap_days:
            lots.append(_finish(cur_days, cur_vals, use_mean))
            cur_days, cur_vals = [], []
        cur_days.append(day)
        cur_vals.extend(by_day[day])
        last = day
    if cur_vals:
        lots.append(_finish(cur_days, cur_vals, use_mean))
    return lots


# Metrics whose per-lot value is the MEAN, not the median: a fail flag is
# 0/1, and the median of a lot of flags is uselessly 0 or 1 — the fraction
# is the observation. Single source of truth lives in drift_types
# (FRACTION_METRICS) so displays and aggregation can never disagree.
from laser_trim_analyzer.ml.drift_types import FRACTION_METRICS as MEAN_AGGREGATED_METRICS  # noqa: E402


def _finish(days: List[datetime], vals: List[float], use_mean: bool = False) -> Lot:
    center = (sum(vals) / len(vals)) if use_mean else float(median(vals))
    return Lot(start=days[0], end=days[-1], median=center,
               n=len(vals), unit_std=float(pstdev(vals)) if len(vals) > 1 else 0.0)


def get_model_lots(db, model: str, metric: str, after: Optional[datetime] = None,
                   gap_days: int = LOT_GAP_DAYS) -> List[Lot]:
    """Lots for one (model, metric), oldest first. `after` floors the window
    (baseline requalification). Reuses the trainer's sample loader so lot
    mode and any residual unit-mode consumers see identical data."""
    from laser_trim_analyzer.ml.drift_training import _load_samples_with_dates
    import math
    samples = [
        (d, float(v)) for (d, v, _rid) in _load_samples_with_dates(
            db, model, metric,
            after=(after - timedelta(seconds=1)) if after else None)
        if v is not None and d is not None
        and not (isinstance(v, float) and math.isnan(v))
    ]
    samples.sort(key=lambda t: t[0])
    return cluster_lots(samples, gap_days=gap_days,
                        use_mean=(metric in MEAN_AGGREGATED_METRICS))
