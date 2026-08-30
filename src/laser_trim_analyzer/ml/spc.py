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

Two layers, in this order:
  * The BUILDERS (`build_fraction_series` / `build_continuous_series`) are pure
    — no DB, no Tk, no matplotlib. They take (date, value) samples and return
    numbers, which is what makes them cheap to test and safe on a worker thread.
  * The LOADERS at the bottom (`compute_spc_series`, `compute_focus_list`) are
    the only part that reads the database. They still never touch Tk, so the
    UI calls them from a worker and posts the result back through ui_dispatch.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import groupby
from math import isfinite, sqrt
from statistics import fmean, pstdev
from typing import Dict, List, Optional, Tuple

from laser_trim_analyzer.ml.lots import (
    LOT_GAP_DAYS, MIN_LOT_BASELINE_N, MIN_LOTS_TRAIN, Lot, cluster_lots)

logger = logging.getLogger(__name__)

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


def _horizon(anchor: Optional[datetime] = None) -> datetime:
    """One day past `anchor` (default: the wall clock) — the file-date junk cut.

    Defined ONCE because four places need the identical cut: `_clean`, the
    staleness test in `compute_focus_list`, `_newest_usable`, and the SQL in
    `_db_anchor`. If two of them disagreed by a day, a junk-dated row could set
    the clock in one and be dropped in another.
    """
    return (anchor or datetime.now()) + timedelta(days=1)


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
    horizon = _horizon(anchor)
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


# ===========================================================================
# DB layer — everything above is pure; everything below reads the database.
# ===========================================================================

CHRONIC_MAX = 5         # chronic strip is context, not a queue: show the worst few


@dataclass
class FocusEntry:
    """One row of the FOCUS list — and the series those numbers came from.

    The series is carried, not re-derived: the chart the user opens next is
    literally this object, so the headline can never disagree with the picture.
    """
    model: str
    series: SpcSeries
    excess_per_week: float          # units/week above this model's own baseline
    units_per_week: float           # recent production rate (volume context)
    p_base: float                   # the model's own normal fail rate
    p_recent: float                 # pooled fail rate over the flagged lots
    n_flagged_recent: int           # how many of the last RECENT_K lots alarmed
    last_lot_end: datetime
    verdict: str                    # headline: what it is costing right now
    sub_line: str                   # the evidence behind the headline
    # The WHY under the fail rate (James, 2026-08-30 smoke test: "it doesnt
    # tell me if its resistance, linearity or why the model is flagged").
    # Plain-language name of the worst-moving PROCESS metric on the drift
    # watch, e.g. "Untrimmed resistance ↑ (+2.1σ vs its baseline)" — or None
    # when no watched process metric is flagged, which the row renders as
    # "driver unclear — open the model". A hint from the drift detector, not
    # a p-chart verdict: the full diagnosis stays on the Model page.
    driver: Optional[str] = None
    # Recovery evidence (James, 2026-08-30, on 6126: a single hairline blip that
    # has run clean since ranked #3, above models burning right now). See
    # `_clean_since` for what counts as proof.
    clean_since: int = 0
    # What the list SORTS by: the measured cost, discounted by that evidence.
    # Deliberately separate from `excess_per_week` — the verdict keeps quoting
    # the MEASURED number, because discounting the measurement would misstate
    # what the excursion actually cost. Only the urgency is discounted.
    rank_score: float = 0.0
    # Do the trim station and final test hold this model to the SAME limits?
    # (James, 2026-08-30: "i also want to know when the trim and test specs
    # dont align.") Visibility only — it changes no verdict and no ranking;
    # it warns that the cross-station numbers on the model's page are
    # comparing two different requirements. See core/spec_alignment.py.
    spec_mismatch: bool = False


@dataclass
class FocusResult:
    """What the FOCUS page shows: today's fires, the known-bad, and the clock."""
    focus: List[FocusEntry]         # ranked by rank_score desc
    chronic: List[FocusEntry]       # ranked by p_base * units_per_week desc
    anchor: Optional[datetime]      # newest usable file date; None = empty DB


def _parse_floor(raw) -> Optional[datetime]:
    """Read a stored requalification date back into a datetime.

    `set_baseline_requalification` stores `str(effective_date)`, so the column
    holds whatever the caller passed — "2026-02-02T00:00:00" from an ISO
    string, "2026-02-02 00:00:00" from a datetime, or a bare "2026-02-02" from
    a date picker. Try the whole thing, then the date-and-time head, then just
    the date; a value we cannot read means NO floor, never a crash.
    """
    if raw is None:
        return None
    text_ = str(raw).strip()
    for candidate in (text_, text_[:19], text_[:10]):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def _requal_floor(db, model: str) -> Optional[datetime]:
    """Baseline requalification date for one model (None when never set)."""
    try:
        row = db.get_baseline_requalification(model)
    except Exception:
        return None                          # missing table on an old DB, etc.
    return _parse_floor(row[0]) if row else None


def _requal_floors(db) -> Dict[str, datetime]:
    """Every model's requalification floor in ONE query.

    The bulk path must not open a session per model — with a few hundred
    models that is a few hundred round trips for a table with a handful of
    rows. Reading ascending and overwriting leaves the LATEST (set_at, id) per
    model, which is the same winner `get_baseline_requalification` picks.
    """
    floors: Dict[str, datetime] = {}
    try:
        from sqlalchemy import text as _text
        with db.session() as s:
            rows = s.execute(_text(
                "SELECT model, effective_date FROM baseline_requalifications "
                "ORDER BY set_at, id")).fetchall()
    except Exception:
        return floors
    for model, raw in rows:
        floor = _parse_floor(raw)
        if model and floor is not None:
            floors[str(model)] = floor
    return floors


def _newest_usable(dates) -> Optional[datetime]:
    """The anchor: the newest file date the data is allowed to be judged from.

    "Now" is the wrong clock — the app is often opened days after the last
    production run, and the wall clock would age every real lot out of the
    recent window. The newest REAL date is the right clock. The +1 day cut
    throws out file-date junk (a mis-set machine clock, a template date) that
    would otherwise drag the anchor months into the future and make every
    active model look stale.
    """
    horizon = _horizon()
    usable = [d for d in dates if d is not None and d <= horizon]
    return max(usable) if usable else None


def _db_anchor(db) -> Optional[datetime]:
    """The WHOLE DATABASE's newest usable file date — the app's one clock.

    Deliberately not per-model. A model's own newest sample is a zero-day gap
    with itself, so anchoring a series on it makes that model's last lot OPEN
    forever: the chart draws a "still filling" marker and the evidence pack
    prints `Open lot: TRUE` on the final row of every export, for a model that
    stopped running three weeks ago. The focus list has always used this global
    clock, so the two surfaces contradicted each other on the same lot.

    Filters mirror `compute_focus_list`'s row query exactly (gradeable statuses,
    a real model, a real date) so the clock the list computes in memory and the
    clock a single-model call reads from SQL cannot come out different — the
    sweep's FOCUS parity check guards that. Cheap: one indexed MAX, not a scan
    of every row. A DB that cannot answer (old schema, locked file) returns
    None and the caller falls back, rather than taking the page down.
    """
    from sqlalchemy import func

    from laser_trim_analyzer.database.models import AnalysisResult as DBAR
    from laser_trim_analyzer.database.models import StatusType
    try:
        with db.session() as s:
            newest = (s.query(func.max(DBAR.file_date))
                      .filter(DBAR.overall_status.in_([StatusType.PASS,
                                                       StatusType.WARNING,
                                                       StatusType.FAIL]),
                              DBAR.model.isnot(None),
                              DBAR.file_date.isnot(None),
                              # Junk cut in SQL: a mis-set clock's row would win
                              # a bare MAX and drag the anchor months forward.
                              DBAR.file_date <= _horizon())
                      .scalar())
    except Exception:
        return None
    return _newest_usable([newest])       # one definition of "usable", re-applied


def _fail_flag(status) -> float:
    """1.0 when the unit missed the customer linearity spec, else 0.0.

    A WARNING unit shipped — it is an internal sigma watch, not a rejection —
    so it counts as a 0 here, exactly like `drift_training` does it.
    """
    return 1.0 if getattr(status, "name", str(status)) == "FAIL" else 0.0


def compute_spc_series(db, model: str, metric: str = "linearity_fail_fraction",
                       *, anchor: Optional[datetime] = None) -> SpcSeries:
    """The series for ONE (model, metric) — what the chart and the export draw.

    Reuses the trainer's sample loader so the chart cannot be looking at a
    different population than the detector that alarmed on it.

    ONE COMPUTATION, ONE CLOCK. The default anchor is the whole DATABASE's
    newest usable date (`_db_anchor`), the same clock `compute_focus_list`
    uses — never this model's own newest sample. Openness is the reason: a
    model is judged against a zero-day gap with itself, so a per-model anchor
    marks its last lot OPEN forever, and a lot the FOCUS row calls CLOSED would
    show up hollow ("· open") the moment the user clicked through to the chart,
    and print `Open lot: TRUE` in the AS9100 evidence pack. The list and every
    click-through must answer that question the same way.

    An explicitly passed `anchor` still wins — a replay or an as-of-date report
    is asking to be judged from a different moment, and that is legitimate.
    """
    from laser_trim_analyzer.ml.drift_training import _load_samples_with_dates
    from laser_trim_analyzer.ml.drift_types import FRACTION_METRICS

    samples = [(d, float(v))
               for (d, v, _rid) in _load_samples_with_dates(db, model, metric)
               if d is not None and v is not None]
    if anchor is None:
        # The DB-global clock, but never OLDER than this metric's own newest
        # sample. The anchor does two jobs — it dates the openness question AND
        # sets `_clean`'s forward cut — and most watched metrics do not live in
        # analysis_results: final test happens days AFTER the trim, so an FT lot
        # can legitimately postdate the newest trim file, and anchoring it on
        # the trim clock would silently drop the newest FT lots off the chart.
        # For linearity — the only metric the focus list builds, so the only one
        # parity is asserted on — `_db_anchor` is the max over every model's
        # usable rows and so is already >= this model's, making the max() a
        # no-op there. `_db_anchor` is None only on an empty/unreadable DB.
        anchor = (max(filter(None, (_db_anchor(db),
                                    _newest_usable(d for d, _v in samples))),
                      default=None) or datetime.now())
    build = (build_fraction_series if metric in FRACTION_METRICS
             else build_continuous_series)
    return build(model, metric, samples, anchor=anchor,
                 requal_floor=_requal_floor(db, model))


def _clean_since(series: SpcSeries) -> int:
    """How many CLOSED, in-band lots this model has run since its newest alarm.

    James, 2026-08-30, looking at 6126: one hairline blip (33.3% against a
    33.2% limit) that has run clean ever since sat at #3 on the list, above
    models that are failing units right now. "Blipped, recovering" and
    "burning" are different problems and the ranking has to say which is which.

    Two rules make this honest evidence rather than a way to bury a fire:

      * Counted from the NEWEST alarming lot, so a model that blipped, ran
        clean for a month, then blipped again is back at zero — the clean run
        in the middle is no longer proof of anything.
      * An OPEN lot never counts. It may still be receiving units, so its rate
        can only get worse; calling it proof would let a model discount itself
        with a lot that hasn't finished. Only finished lots are evidence.

    In-band means the lot's rate is inside its own limit. `ooc` alone is not
    enough: a lot too small to alarm (below `MIN_LOT_BASELINE_N`) can still sit
    well above the limit, and counting that as "ran clean" would be false.
    """
    points = series.points
    flagged = [i for i, pt in enumerate(points) if pt.ooc]
    if not flagged:
        return 0                      # nothing to recover from
    return sum(1 for pt in points[flagged[-1] + 1:]
               if not pt.is_open and isfinite(pt.ucl) and pt.value <= pt.ucl)


def compute_focus_list(db, *, anchor: Optional[datetime] = None) -> FocusResult:
    """Rank every active model by what its drift is costing THIS week.

    One query for the whole database, grouped in memory. The per-model
    alternative (one query each, several hundred models) is what made the old
    focus page slow enough that people stopped opening it.

    The ranking is deliberately in UNITS, not in percent: a model that went
    from 5% to 15% on a 10-unit-a-week trickle is a smaller fire than one that
    went from 8% to 12% on 400 units a week, and the person choosing what to
    work on this morning needs the list to say so.

    That cost is then discounted by the lots the model has run CLEAN since its
    alarm (`_clean_since`) — a blip that has settled down is not the same
    problem as one that is still burning, and the list ranks on the discounted
    `rank_score` while every printed number stays the measured one.
    """
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR
    from laser_trim_analyzer.database.models import StatusType

    with db.session() as s:
        # Mirrors _load_samples_with_dates' linearity branch, minus the model
        # filter: ERROR/UNTRIMMED rows are not gradeable and stay out.
        rows = (s.query(DBAR.model, DBAR.file_date, DBAR.overall_status)
                .filter(DBAR.overall_status.in_([StatusType.PASS, StatusType.WARNING,
                                                 StatusType.FAIL]),
                        DBAR.file_date.isnot(None), DBAR.model.isnot(None))
                .order_by(DBAR.model, DBAR.file_date).all())

    if anchor is None:
        anchor = _newest_usable(r.file_date for r in rows)
    if anchor is None:
        return FocusResult(focus=[], chronic=[], anchor=None)

    floors = _requal_floors(db)
    active_floor = anchor - timedelta(days=ACTIVE_DAYS)
    volume_floor = anchor - timedelta(days=UNITS_WEEK_DAYS)
    horizon = _horizon(anchor)                          # same cut `_clean` uses
    weeks = UNITS_WEEK_DAYS / 7.0

    focus: List[FocusEntry] = []
    chronic: List[FocusEntry] = []
    # ORDER BY model keeps each model's rows contiguous, so one pass groups them.
    for model, group in groupby(rows, key=lambda r: r.model):
        samples = [(r.file_date, _fail_flag(r.overall_status)) for r in group]
        # Staleness is decided on the dates the builder will actually KEEP.
        # A mis-set machine clock writes rows dated months or years ahead (the
        # work data has a file dated 12-18-2026); judged on raw dates, one such
        # row makes a dormant model look like it ran today, `_clean` then drops
        # it inside the builder, and the model lands on the focus list with a
        # verdict computed entirely from history that ended last spring.
        dates = [d for d, _v in samples if d <= horizon]
        if not dates or max(dates) < active_floor:
            continue        # not running lately — nothing to act on today
        series = build_fraction_series(model, "linearity_fail_fraction", samples,
                                       anchor=anchor,
                                       requal_floor=floors.get(model))
        if not series.judged or not series.points:
            continue        # too little history to have an opinion (see builders)

        # Volume drives the cost side of the ranking. Counting units (not lots)
        # over a fixed window keeps a model that runs one huge lot a month
        # comparable to one that runs a little every day.
        units_recent = sum(1 for d in dates if d >= volume_floor)
        units_per_week = units_recent / weeks

        flagged = [pt for pt in series.points[-RECENT_K:] if pt.ooc]
        if flagged:
            # Pooled over the flagged lots — a 200-unit excursion should weigh
            # more than a 6-unit one when we quote "the rate right now".
            flagged_units = sum(pt.n for pt in flagged)
            p_recent = sum(pt.value * pt.n for pt in flagged) / flagged_units
            excess_per_week = max(p_recent - series.p_base, 0.0) * units_per_week
            # Whole units read better at scale, but rounding to whole units
            # prints a real 0.4-units/week fire as "~0 more units/week" — the
            # headline would be telling the reader the cost is nothing. Below
            # 10, show the tenth; above it, the tenth is noise.
            shown_excess = (f"{excess_per_week:.1f}" if excess_per_week < 10
                            else f"{excess_per_week:.0f}")
            # Recovery discount. The evidence goes in the sub-line HERE, inside
            # the one computation, so the list, the chart caption and the export
            # cannot end up disagreeing about whether a model has settled down.
            clean_since = _clean_since(series)
            focus.append(FocusEntry(
                model=model, series=series, excess_per_week=excess_per_week,
                units_per_week=units_per_week, p_base=series.p_base,
                p_recent=p_recent, n_flagged_recent=len(flagged),
                last_lot_end=series.points[-1].end,
                verdict=(f"failing ~{shown_excess} more units/week "
                         "than its own baseline"),
                sub_line=(f"{len(flagged)} of last {RECENT_K} lots out of control"
                          f" · fail rate {series.p_base * 100:.0f}%"
                          f" → {p_recent * 100:.0f}%"
                          f" · ~{units_per_week:.0f} units/wk"
                          + (" · has run clean since" if clean_since >= 1 else "")),
                clean_since=clean_since,
                # Each closed clean lot halves, thirds, quarters the urgency —
                # never zeroes it. A model that blipped is still on the list
                # (the excursion happened), just below the ones burning today.
                rank_score=excess_per_week / (1.0 + clean_since)))
        elif series.chronic:
            # Bad but steady. Nothing changed today, so it is not a fire — and
            # re-alarming on it every morning is how a list loses its audience.
            chronic.append(FocusEntry(
                model=model, series=series, excess_per_week=0.0,
                units_per_week=units_per_week, p_base=series.p_base,
                p_recent=series.p_base, n_flagged_recent=0,
                last_lot_end=series.points[-1].end,
                verdict=(f"runs ~{series.p_base * 100:.0f}% fail, stable "
                         "— capability problem, not drift"),
                sub_line=f"~{units_per_week:.0f} units/wk"))

    # Sorted by the DISCOUNTED cost, not the raw one: "what should I work on
    # this morning" is not the same question as "who blew up hardest recently".
    focus.sort(key=lambda e: e.rank_score, reverse=True)
    # Enrich the rows that will actually render — with their WHY, and with
    # whether the two stations even grade this model the same way. This runs
    # AFTER membership and ranking so its cost is bounded by the list length
    # (~a dozen models), not the model count — and a failure here degrades to
    # "no hint", never to a broken list. Chronic rows are not enriched: their
    # verdict already names the problem (capability, not drift).
    for e in focus:
        e.driver = _likely_driver(db, e.model)
        e.spec_mismatch = _spec_mismatch(db, e.model)
    # Chronic ranks by steady bleed (rate x volume): the biggest standing loss
    # is the one worth an engineering project.
    chronic.sort(key=lambda e: e.p_base * e.units_per_week, reverse=True)
    return FocusResult(focus=focus, chronic=chronic[:CHRONIC_MAX], anchor=anchor)


def _spec_mismatch(db, model: str) -> bool:
    """Do the two stations grade this model against different limits?

    A flag, not a diagnosis: the row shows a marker, the Model page shows the
    sentence, and neither re-grades anything (spec governance is a separate
    decision). Same contract as `_likely_driver` — it runs after ranking, so
    its cost is bounded by the list length, and any failure degrades to False
    rather than taking the whole list down.
    """
    try:
        from laser_trim_analyzer.core.spec_alignment import compare_station_specs
        return compare_station_specs(db, model).status == "differs"
    except Exception:
        logger.exception("spec alignment check failed for %s", model)
        return False


def _likely_driver(db, model: str) -> Optional[str]:
    """The worst-moving PROCESS metric on the drift watch, in plain words.

    The FOCUS row already IS the outcome (the lot fail rate) — the reader's
    next question is WHY: which upstream metric moved underneath. That answer
    lives in the multi-metric drift detector's state, so read it — never
    retrain it. Outcome metrics (the FRACTION family) are excluded because
    "your fail rate is up because your fail rate is up" is not a why.

    Severity order: tier first, then how far the recent data actually sits
    from baseline (|shift| in σ), falling back to the detector's magnitude
    when no recent mean is recorded. Any hiccup returns None — a missing hint
    must never take the list down with it.
    """
    try:
        from laser_trim_analyzer.ml.drift_types import (
            DriftTier, FRACTION_METRICS, metric_label)
        from laser_trim_analyzer.ml.manager import get_model_drift_status

        status = get_model_drift_status(db, model)
        best = None                       # (sort_key, metric, shift)
        for m, ms in status.per_metric.items():
            if (m in FRACTION_METRICS or not ms.is_trained
                    or ms.tier <= DriftTier.STABLE):
                continue
            shift = None
            if ms.recent_mean is not None and ms.baseline_std:
                shift = (ms.recent_mean - ms.baseline_mean) / ms.baseline_std
            key = (int(ms.tier),
                   abs(shift) if shift is not None else abs(ms.magnitude))
            if best is None or key > best[0]:
                best = (key, m, shift)
        if best is None:
            return None
        _key, m, shift = best
        if shift is None:
            return f"{metric_label(m)} — flagged on the drift watch"
        arrow = "↑" if shift >= 0 else "↓"
        return f"{metric_label(m)} {arrow} ({shift:+.1f}σ vs its baseline)"
    except Exception:
        logger.exception("driver hint failed for %s", model)
        return None
