"""SPC series — ONE computation behind the focus list, the chart, and the export.

The FOCUS redesign (2026-08-29) exists because those three surfaces each did
their own arithmetic: the list ranked a model by one rule, the chart drew its
band from another, and the export printed a third. A supervisor who checked all
three got three different stories and trusted none of them. Everything
downstream now reads the SAME `SpcSeries` — so the list can never claim what
the chart doesn't show.

The observation is the LOT, not the unit (see `lots.py`): production runs in
lots, the laser changes over, and the model comes back days or weeks later as a
NEW lot. Two series shapes cover every watched metric:

  * FRACTION metrics (`drift_types.FRACTION_METRICS`) are rates — the share of
    a lot's units that FAILED. Their limits are binomial and must depend on the
    lot SIZE: 2 fails out of 5 units is noise, 2 out of 200 is a signal, and no
    single flat threshold can say both. Only the UPPER side alarms — a lot that
    ran BETTER than baseline is good news, not an excursion.
  * CONTINUOUS metrics are lot medians (robust to the known corrupt readings).
    Their limits are the ordinary ±3σ of the model's own baseline lot medians,
    two-sided: drifting either way is a process change worth seeing.

Two kinds of honesty are wired in deliberately:
  * Below `MIN_LOTS_TRAIN` lots the series is NOT JUDGED — limits are NaN, no
    point is flagged, and the UI draws no band. Silence beats an invented limit.
  * A model can be chronically bad AND in control (`CHRONIC_PBAR`). That is an
    engineering project, not today's fire, and `chronic` lets the list say so
    instead of re-alarming on the same known-bad model every single day.

Wording note: in this domain FAIL means the unit missed the customer LINEARITY
spec and is rejected. A WARNING unit is ACCEPTED (an internal sigma watch), so
nothing in here counts a warning as a failure.

Pure module by design — no DB, no Tk, no matplotlib. It takes (date, value)
samples and returns numbers, which is what makes it cheap to test and safe to
call from a worker thread.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite, sqrt
from statistics import fmean, pstdev
from typing import List, Optional, Tuple

from laser_trim_analyzer.ml.lots import (
    LOT_GAP_DAYS, MIN_LOT_BASELINE_N, MIN_LOTS_TRAIN, Lot, cluster_lots)

RECENT_K = 5            # membership window: last K lots (incl. open)
ACTIVE_DAYS = 60        # focus list requires production this recent
CHRONIC_PBAR = 0.15     # chronic-but-stable threshold (James-approved)
SERIES_WINDOW = 30      # lots carried per series
UNITS_WEEK_DAYS = 28    # volume window for units/week

# A judged series needs this many BASELINE lots on top of MIN_LOTS_TRAIN total:
# three points is the least that gives a center and a spread anyone should act on.
MIN_BASELINE_LOTS = 3
# Continuous baselines flatter than this are degenerate (one repeated value, or a
# quantized metric). ±3*0 would flag every point that isn't exactly the center.
_MIN_SD = 1e-12
_NAN = float("nan")


@dataclass(frozen=True)
class SpcPoint:
    """One lot on the chart — and the same numbers the list and export quote."""
    start: datetime
    end: datetime
    value: float            # fail fraction (fraction metrics) or lot median
    n: int
    ucl: float
    lcl: float
    center: float           # nan when the series is not judged
    ooc: bool               # out of control (upper-only for fractions)
    is_open: bool           # may still be receiving units — preview, not verdict
    note: str               # plain-English sentence when ooc, else ""


@dataclass
class SpcSeries:
    """A (model, metric) control chart: points, limits, and why they exist."""
    model: str
    metric: str
    points: List[SpcPoint]          # oldest->newest, last SERIES_WINDOW lots
    judged: bool                    # False = not enough history; no limits drawn
    p_base: float                   # baseline center (nan when not judged)
    baseline_n_lots: int
    baseline_units: int
    chronic: bool                   # judged, p_base >= CHRONIC_PBAR, no recent ooc


def _clean(samples: List[Tuple[datetime, float]], *, anchor: datetime,
           requal_floor: Optional[datetime]) -> List[Tuple[datetime, float]]:
    """Drop the samples SPC must never see.

    `requal_floor` is the baseline requalification date: after a documented
    process change the old lots describe a process that no longer exists.
    The forward cut is for file-date junk — a mis-set clock or a template date
    lands units months in the "future", and one such lot would otherwise become
    the newest lot and own the whole verdict. One day of slack covers timezone
    and end-of-shift skew.
    """
    horizon = anchor + timedelta(days=1)
    out: List[Tuple[datetime, float]] = []
    for d, v in samples:
        if d is None or v is None:
            continue
        v = float(v)
        if not isfinite(v):                       # lots.py's caller contract
            continue
        if requal_floor is not None and d < requal_floor:
            continue
        if d > horizon:
            continue
        out.append((d, v))
    return out                                    # cluster_lots sorts by day itself


def _baseline_lots(lots: List[Lot]) -> List[Lot]:
    """Everything OLDER than the recent window — what "normal" is measured from.

    The recent window is excluded so a bad run can't quietly raise the bar it is
    being judged against. Tiny lots are dropped because a 1-unit lot's fraction
    is 0% or 100% and would drag the center around; if that filter would leave
    nothing, we keep the small lots rather than refuse to judge at all.
    """
    older = lots[:-RECENT_K] if len(lots) > RECENT_K else []
    return [lot for lot in older if lot.n >= MIN_LOT_BASELINE_N] or older


def _is_open(lot: Lot, anchor: datetime) -> bool:
    """Might this lot still be receiving units? (Only the newest one can be.)

    Whole DAYS, not `Lot.is_open`'s timedelta compare: `lot.end` is midnight-
    normalized by `cluster_lots`, so a wall-clock anchor of "today 3pm" would
    read as 3.6 days and close a lot that ran this morning.
    """
    return (anchor - lot.end).days <= LOT_GAP_DAYS


def _unjudged(model: str, metric: str, window: List[Lot], anchor: datetime,
              baseline: List[Lot]) -> SpcSeries:
    """Points with no limits at all: the honest "not enough history yet" shape."""
    last = len(window) - 1
    points = [SpcPoint(start=lot.start, end=lot.end, value=lot.median, n=lot.n,
                       ucl=_NAN, lcl=_NAN, center=_NAN, ooc=False,
                       is_open=(i == last and _is_open(lot, anchor)), note="")
              for i, lot in enumerate(window)]
    return SpcSeries(model=model, metric=metric, points=points, judged=False,
                     p_base=_NAN, baseline_n_lots=len(baseline),
                     baseline_units=sum(lot.n for lot in baseline), chronic=False)


def build_fraction_series(model: str, metric: str,
                          samples: List[Tuple[datetime, float]],  # (date, 0.0/1.0)
                          *, anchor: datetime,
                          requal_floor: Optional[datetime] = None) -> SpcSeries:
    """p-chart for a 0/1 outcome flag: per-lot fail rate vs the model's own history.

    Limits are recomputed for EVERY lot from its own n, which is why a small lot
    gets a wide band and a big lot a tight one. Upper side only — see module docstring.
    """
    lots = cluster_lots(_clean(samples, anchor=anchor, requal_floor=requal_floor),
                        use_mean=True)     # a lot of 0/1 flags aggregates to its MEAN
    baseline = _baseline_lots(lots)
    # Baseline sees ALL lots; only the drawn window is trimmed.
    window = lots[-SERIES_WINDOW:]
    if len(lots) < MIN_LOTS_TRAIN or len(baseline) < MIN_BASELINE_LOTS:
        return _unjudged(model, metric, window, anchor, baseline)

    baseline_units = sum(lot.n for lot in baseline)
    # Unit-weighted, not lot-weighted: a 200-unit lot says more about the true
    # rate than a 4-unit one, and pbar is the pooled proportion by definition.
    p_base = sum(lot.median * lot.n for lot in baseline) / baseline_units

    points: List[SpcPoint] = []
    last = len(window) - 1
    for i, lot in enumerate(window):
        # Binomial standard error at THIS lot's size. The 1e-9 floor keeps a
        # perfect (p_base == 0 or 1) baseline from producing a zero-width band
        # that flags every non-identical lot.
        se = sqrt(max(p_base * (1.0 - p_base), 1e-9) / lot.n)
        ucl = p_base + 3.0 * se
        lcl = max(p_base - 3.0 * se, 0.0)          # a fail rate can't go below 0
        ooc = lot.median > ucl and lot.n >= MIN_LOT_BASELINE_N
        note = (f"{lot.median * 100:.0f}% of {lot.n} units failed "
                f"— expected at most {ucl * 100:.0f}%") if ooc else ""
        points.append(SpcPoint(start=lot.start, end=lot.end, value=lot.median,
                               n=lot.n, ucl=ucl, lcl=lcl, center=p_base, ooc=ooc,
                               is_open=(i == last and _is_open(lot, anchor)),
                               note=note))

    # Chronic = bad on average but behaving: schedule it, don't page anyone.
    chronic = p_base >= CHRONIC_PBAR and not any(pt.ooc for pt in points[-RECENT_K:])
    return SpcSeries(model=model, metric=metric, points=points, judged=True,
                     p_base=p_base, baseline_n_lots=len(baseline),
                     baseline_units=baseline_units, chronic=chronic)


def build_continuous_series(model: str, metric: str,
                            samples: List[Tuple[datetime, float]],
                            *, anchor: datetime,
                            requal_floor: Optional[datetime] = None) -> SpcSeries:
    """Individuals chart on LOT MEDIANS: ±3σ of this model's own normal, two-sided.

    Nothing here is a pass/fail disposition — it answers "is this lot out of
    family for this model", which is the question the focus list asks.
    """
    lots = cluster_lots(_clean(samples, anchor=anchor, requal_floor=requal_floor),
                        use_mean=False)    # median: robust to the 2.1e8 ohm class
    baseline = _baseline_lots(lots)
    window = lots[-SERIES_WINDOW:]
    if len(lots) < MIN_LOTS_TRAIN or len(baseline) < MIN_BASELINE_LOTS:
        return _unjudged(model, metric, window, anchor, baseline)

    centers = [lot.median for lot in baseline]
    center = fmean(centers)
    sd = pstdev(centers)          # spread BETWEEN lots — the lot-to-lot noise
    if sd < _MIN_SD:
        return _unjudged(model, metric, window, anchor, baseline)

    # One band for the whole series: unlike the p-chart, lot size doesn't move it.
    ucl = center + 3.0 * sd
    lcl = center - 3.0 * sd
    points: List[SpcPoint] = []
    last = len(window) - 1
    for i, lot in enumerate(window):
        ooc = abs(lot.median - center) > 3.0 * sd and lot.n >= MIN_LOT_BASELINE_N
        note = (f"lot median {lot.median:.4g} — outside this model's normal "
                f"({lcl:.4g} to {ucl:.4g})") if ooc else ""
        points.append(SpcPoint(start=lot.start, end=lot.end, value=lot.median,
                               n=lot.n, ucl=ucl, lcl=lcl, center=center, ooc=ooc,
                               is_open=(i == last and _is_open(lot, anchor)),
                               note=note))

    # `chronic` is a fail-RATE idea (a model that always fails 20%). A continuous
    # metric sitting off nominal isn't "chronic", it's just where this model runs.
    return SpcSeries(model=model, metric=metric, points=points, judged=True,
                     p_base=center, baseline_n_lots=len(baseline),
                     baseline_units=sum(lot.n for lot in baseline), chronic=False)
