"""The Overview's data -- ONE loader, `load_overview(db)`, that never raises (Graphite redesign,
2026-10-02; spec docs/superpowers/specs/2026-10-02-graphite-redesign-design.md).

What the page draws, top to bottom, and where each part comes from:

  * the CARDS ("N models need a look"): the union of the "Drifting now" fail-rate list
    (`ml/spc.compute_focus_list`, through `focus_data.load_focus`, the one FOCUS loader) and the
    drift detector's flags (`ml/manager.get_drifting_models`). James, 2026-10-02: "keep all 16
    cards (fail rate up, or a signal moved), each with its reason". The fail-rate list's own order
    first, then the detector's other models by tier, then by how far over its line each one is.
  * "Everything else": every other ACTIVE model -- at least one graded trim in the 90 days --
    busiest first, with its pass % and the trend against the year before.
  * the INACTIVE models (`core/activity`, F5): counted on one line, never hidden.
  * the two data-health counts Home showed (final tests graded before the ignore-window fix,
    files skipped as unreadable).
  * (option B, 2026-10-04: James, "the dollars hidden on Company trends -> on the Overview") each
    card's and row's DOLLARS lost at final test -- the Company trends formula, final-test FAILs x
    the model's unit price x the cost ratio (core/cost_priorities) -- counted over THIS page's
    own 90 days, with the final-test filters `_ft_rates_by_day` reads. The header's TOTAL is the
    full cost: every model with a price that failed final test in those days, on the page or not,
    as Company trends counts it; and how many models failed final test with no price (counted,
    never read as $0). With no price loaded at all, the header asks for prices -- a missing input
    never reads as "$0 lost". And each model's LASERS: the systems with a graded trim in the 90
    days, in the shop's order.

PASS % is the app's headline yield (`core/yield_stats.compute_yield`'s linearity_yield):
(PASS + WARNING) / (PASS + WARNING + FAIL) -- a WARNING is a sigma watch, never a rejection
(CLAUDE.md). ERROR / PROCESSING_FAILED and UNTRIMMED are not graded; a file dated more than a day
ahead of now is not believed (`core/activity.trusted_until`); and a file marked suspect
(`data_quality = 'suspect'`) is left out -- the rule the drift watch and the FOCUS list follow
since 2026-10-02, so a card's number and the reason printed under it are read from the same files.

THE WINDOWS are anchored on the newest file that counts -- never the wall clock, like every other
"recent" in the app, so a copy of an older database reads as it did: the 90 days ending on that
file's day; "was" is the 365 days before them; the lines are the 12 calendar months ending in that
file's month. Every model's counts come from ONE grouped query (per model and day), never one
query per model.

Worker-safe: no Tk, no widget. Each part that fails is NAMED in `Overview.failed` and the others
still load: a crash must never read as "0 models need a look" (CLAUDE.md: a failure must never
look like a result).
"""
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from laser_trim_analyzer.core.activity import load_activity, trusted_until
from laser_trim_analyzer.core.ft_regrade import legacy_ft_count
from laser_trim_analyzer.core.ingest_run import unreadable_count
from laser_trim_analyzer.core.models import LASER_ORDER
from laser_trim_analyzer.gui.v6 import formats
from laser_trim_analyzer.gui.v6.focus_data import focus_failed, load_focus
from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import RECENT_LOT_DAYS, format_metric_value, metric_label
from laser_trim_analyzer.ml.manager import (
    drift_reference_date, get_drifting_models, get_model_drift_status)
from laser_trim_analyzer.ml.spc import RECENT_K

logger = logging.getLogger(__name__)

# Hand-trimmed AFTER the laser, so a laser PASS on them is hand-trim workload, not yield: their cards
# and rows carry a "hand trim" tag. ONE list, core/model_names (it also sets their 20x suspect line).
from laser_trim_analyzer.core.model_names import HAND_TRIM_MODELS  # noqa: E402

WINDOW_DAYS = 90        # "Last 90 days": what every card and row is about
PRIOR_DAYS = 365        # "was X%": the year before those 90 days
MONTHS = 12             # the lines: twelve calendar months, ending in the newest file's month
STEADY_POINTS = 5       # within +/-5 points of the year before reads "steady"
STILL_PASSING = 99.0    # a signal that moved on a model passing >= 99% says it is still passing

# The parts that load (and fail) on their own, worded as the page's banner names them.
PART_FOCUS = "the drifting-now list"
PART_DRIFT = "the drift watch's flags"
PART_RATES = "the pass rates"
PART_FT = "the final-test pass rates"     # read only for a card with no trim to show
PART_ACTIVITY = "which models are inactive"
PART_ON_FILE = "the other models on file"
PART_TREND = "the yield chart by laser"
PART_MONEY = "the dollars lost at final test"

# The share of a unit's price lost when it fails at final test, when the config gives none
# (config.ActiveModelsConfig.cost_ratio's own default; core/cost_priorities falls back the same).
DEFAULT_COST_RATIO = 0.5

# The chart of each laser and the company (James, 2026-10-04: "on the overveiw screen i no longer
# have each laser charted overall?"): the Company trends chart itself, its last 12 months by month.
TREND_DAYS = 365
TREND_PERIOD = "month"

# The signal a fail-rate card is about: the FOCUS list is a p-chart of each run's linearity fail
# fraction (ml/spc.compute_focus_list), and a click opens the Model page charting it.
FOCUS_METRIC = "linearity_fail_fraction"

# What puts a model on a card -- under "Needs a look" since option B (2026-10-04) -- said where
# the list is read: the retired FOCUS list said why a model was on it and when it left (final
# review of the redesign, 2026-10-02). RECENT_K, not a written 5: the sentence can only promise
# the window the computation uses.
CARD_RULE = (f"A model is under Needs a look while one of its last {RECENT_K} runs failed more "
             "often than its own history allows, or a watched signal has moved from its baseline "
             f"— in a run of the last {RECENT_LOT_DAYS} days. Its detail says which.")

_GRADED = ("PASS", "WARNING", "FAIL")
_ACCEPTED = ("PASS", "WARNING")

Months = List[Optional[float]]


@dataclass
class Card:
    """One model that needs a look, and the line saying why."""
    model: str
    reason: str
    # graded trims in the 90 days (final tests if final_test); None when the read it rests on
    # failed -- the banner names it, and the card prints no count it does not have
    units: Optional[int] = 0
    pass_pct: Optional[float] = None     # 0..100; None with nothing graded
    was_pct: Optional[float] = None      # the year before; None with nothing graded then
    months: Months = field(default_factory=list)    # MONTHS values, oldest first
    final_test: bool = False             # no trims to show: these are its final-test numbers
    hand_trim: bool = False
    metric: Optional[str] = None         # the signal its reason names first: what a click charts
    # The systems (the code's letters) with a graded trim in the 90 days, in the shop's order --
    # laser 1, 2, 3; [] for none (a final-test card); None when the pass rates could not be read.
    lasers: Optional[List[str]] = field(default_factory=list)
    # Final-test FAILs in the 90 days, and what they cost: x unit price x cost ratio. money is None
    # for a model with no price -- and both are None when the money read failed
    # (Overview.money_total None says which).
    ft_fails: Optional[int] = None
    money: Optional[float] = None
    newest: Optional[date] = None        # the day of its newest counted file (trim, or final test)


@dataclass
class Row:
    """One "Everything else" line."""
    model: str
    units: int
    pass_pct: Optional[float]
    was_pct: Optional[float]
    trend: str                           # "steady" / "up N pts" / "down N pts" / "new"
    tone: str                            # "steady" / "up" / "down" / "new"
    months: Months = field(default_factory=list)
    hand_trim: bool = False
    lasers: Optional[List[str]] = field(default_factory=list)      # as Card's
    ft_fails: Optional[int] = None
    money: Optional[float] = None
    newest: Optional[date] = None


@dataclass
class Overview:
    anchor: Optional[datetime] = None    # the newest file that counts; the windows end on its day
    cards: List[Card] = field(default_factory=list)
    others: List[Row] = field(default_factory=list)
    # The dollars lost at final test in the 90 days -- every model with a price that failed final
    # test then, on the page or not: the full cost. None when they could not be worked out (the
    # money read failed, or no window: the pass rates failed or no trim is on record). `unpriced`:
    # every model that failed final test then with no price. `no_prices`: no price was loaded at
    # all -- the header asks for them rather than say "$0 lost".
    money_total: Optional[float] = None
    unpriced: int = 0
    no_prices: bool = False
    # {model: its newest trim file, or None for "no trims on record"}; None = could not be worked out
    inactive: Optional[Dict[str, Optional[datetime]]] = None
    # The chart of each laser: database.get_company_yield_trend's shape; None = could not be loaded
    trend: Optional[Dict[str, Any]] = None
    # {model: its newest trim file} for every OTHER model on file -- on no card, in no list and not
    # inactive: most were trimmed before the 90 days; a value of None is a model with no trim file
    # at all (smoothness records only). The field None = could not be worked out.
    quiet: Optional[Dict[str, Optional[datetime]]] = None
    legacy_ft: int = 0
    unreadable: int = 0
    failed: Dict[str, str] = field(default_factory=dict)    # part -> "ExcType: message"


# ---- the words the page prints (pure, so they are tested without Tk) ---------------------------

def card_count(ov: Overview) -> Optional[int]:
    """How many models need a look -- or None when a card source failed: the count is unknown,
    and an unknown count printed as a number is a failure looking like a result."""
    if PART_FOCUS in ov.failed or PART_DRIFT in ov.failed:
        return None
    return len(ov.cards)


def need_a_look(count: Optional[int]) -> str:
    if count is None:
        return "Models that need a look"
    if count == 0:
        return "No models need a look"
    return f"{count} model needs a look" if count == 1 else f"{count:,} models need a look"


def dollars(x: float) -> str:
    """Whole dollars -- but never rounded to nothing: a loss under a dollar reads "<$1"."""
    n = round(x)
    if n == 0 and x > 0:
        return "<$1"
    return f"${n:,}"


def money_known(ov: Overview) -> bool:
    return ov.money_total is not None and PART_MONEY not in ov.failed


# With no price loaded at all, the header says where they come from (gui/v6/sections/backlog.py:
# one open-order upload sets each model's price) -- never "$0 lost" over a missing input.
NO_PRICES = "add prices in Settings → Backlog to see the dollars lost at final test"


def money_words(ov: Overview) -> str:
    """The header line's dollars: "$12,345 lost at final test in the last 90 days · 3 without a
    price"; "" with no trim file on record (no window to count over); NO_PRICES with no price
    loaded at all; and when the read -- or the pass rates that anchor its window -- failed, says
    so: never "$0" for a crash, nor for a price list that is empty."""
    if PART_MONEY in ov.failed or PART_RATES in ov.failed:
        return "dollars lost at final test could not be worked out"
    if ov.money_total is None:
        return ""
    if ov.no_prices:
        return NO_PRICES
    words = f"{dollars(ov.money_total)} lost at final test in the last {WINDOW_DAYS} days"
    if ov.unpriced:
        words += f" · {ov.unpriced:,} without a price"
    return words


def header_line(ov: Overview) -> str:
    """The Overview's one header line: "13 models need a look · $X lost at final test in the last
    90 days · newest file 29 Sep 2026" -- the clock every number on the page uses. Each part says
    only what is known: no count when a card source failed, no newest file without the rates."""
    parts = [need_a_look(card_count(ov)), money_words(ov)]
    if ov.anchor is not None:
        parts.append(f"newest file {formats.day(ov.anchor)}")
    elif PART_RATES not in ov.failed:
        parts.append("no trim files on record yet")
    return " · ".join(p for p in parts if p)


def money_text(ov: Overview, item) -> str:
    """One model's dollars, for its detail: "$1,234 · 12 failed final test", "no price · 3 failed
    final test", "$0" -- or "could not be worked out" (never "no price" over a failed read)."""
    if not money_known(ov) or item.ft_fails is None:
        return "could not be worked out"
    words = "no price" if item.money is None else dollars(item.money)
    if item.ft_fails:
        words += f" · {item.ft_fails:,} failed final test"
    return words


def pct_text(p: Optional[float]) -> str:
    """Whole percent, but never rounded to all or nothing: 99.6 reads 99%, 0.3 reads 1%."""
    if p is None:
        return "—"
    n = round(p)
    if n >= 100 and p < 100:
        n = 99
    if n <= 0 < p:
        n = 1
    return f"{n}%"


def trend_words(pass_pct: Optional[float], was_pct: Optional[float]) -> Tuple[str, str]:
    """(words, tone) for a row: against the year before, in whole points -- "steady" within
    +/-STEADY_POINTS, else "up N pts" / "down N pts"; "new" with nothing graded the year before."""
    if pass_pct is None or was_pct is None:
        return "new", "new"
    pts = round(pass_pct - was_pct)
    if abs(pts) <= STEADY_POINTS:
        return "steady", "steady"
    return (f"up {pts} pts", "up") if pts > 0 else (f"down {-pts} pts", "down")


def focus_reason(entry) -> str:
    return f"Fail rate {entry.p_base:.0%} → {entry.p_recent:.0%}"


def drift_reason(status, metric: str) -> str:
    """"Untrimmed resistance 4,693 → 5,897": the worst moving signal, its baseline and now."""
    ms = (getattr(status, "per_metric", None) or {}).get(metric)
    label = metric_label(metric)
    if ms is None:
        return label
    fmt = ThemeManager.fmt_measure
    return (f"{label} {format_metric_value(metric, ms.baseline_mean, fmt)} → "
            f"{format_metric_value(metric, ms.recent_mean, fmt)}")


# ---- the loader -----------------------------------------------------------------------------------

def load_overview(db, *, prices: Optional[Dict[Any, Any]] = None, cost_ratio: Optional[float] = None,
                  now: Optional[datetime] = None) -> Overview:
    """Everything the Overview draws. Never raises: each part that fails is named in `failed`.

    `prices` ({model: unit price}) and `cost_ratio` are the config's (active_models.model_prices
    and .cost_ratio, which the page hands over); without a price a model's dollars are None."""
    ov = Overview()

    # Pass rates first: the anchor (and so every window) comes from them.
    rates: Dict[str, Dict[date, Tuple[int, int]]] = {}
    systems: Dict[str, Dict[str, date]] = {}
    try:
        ov.anchor, rates, systems = _rates_by_day(db, now)
    except Exception as exc:
        logger.exception("Overview: the pass rates could not be worked out")
        ov.failed[PART_RATES] = _why(exc)
    anchor_day = ov.anchor.date() if ov.anchor is not None else None
    rates_known = PART_RATES not in ov.failed

    # The two card sources, each on its own guard.
    focus = []
    # Never raises; a crash comes back marked. No "last processed" stamp: this page never
    # prints one, and reading it runs the drift detector over every model a second time.
    result, _last = load_focus(db, stamp=False)
    error = focus_failed(result)
    if error:
        ov.failed[PART_FOCUS] = error
    else:
        focus = list(result.focus)
    flags, statuses = [], {}
    try:
        flags = list(get_drifting_models(db))
        if flags:
            reference = drift_reference_date(db)
            statuses = {a.model: get_model_drift_status(db, a.model, reference_date=reference)
                        for a in flags}
    except Exception as exc:
        logger.exception("Overview: the drift watch's flags could not be read")
        ov.failed[PART_DRIFT] = _why(exc)
        flags, statuses = [], {}

    # The cards: the fail-rate list's own order, then the detector's other flags by tier and
    # magnitude (get_drifting_models already sorts so; sorted again here, so the order is this
    # module's promise and not an accident of another's).
    by_flag = {a.model: a for a in flags}
    in_focus = [e.model for e in focus]
    rest = sorted((a for a in flags if a.model not in set(in_focus)),
                  key=lambda a: (int(a.tier), a.magnitude), reverse=True)
    order: List[Tuple[str, Optional[object]]] = [(e.model, e) for e in focus] + [(a.model, None) for a in rest]

    # Final-test numbers for a card with no graded trim to show (8506's shape on the work data).
    no_trims = [m for m, _e in order if not _graded_in_window(rates.get(m), anchor_day)]
    ft_rates: Dict[str, Dict[date, Tuple[int, int]]] = {}
    if no_trims and anchor_day is not None:
        try:
            ft_rates = _ft_rates_by_day(db, no_trims, anchor_day, now)
        except Exception as exc:
            logger.exception("Overview: the final-test pass rates could not be worked out")
            ov.failed[PART_FT] = _why(exc)

    for model, entry in order:
        days, final_test = rates.get(model), False
        if not _graded_in_window(days, anchor_day) and _graded_in_window(ft_rates.get(model), anchor_day):
            days, final_test = ft_rates.get(model), True
        # Unknown, not zero, when the read this card's numbers rest on failed: every card's when
        # the pass rates did, and a card with no trim to show when the final-test read did (its
        # numbers would have been final test's -- which ones is unknown).
        unknown = PART_RATES in ov.failed or (
            PART_FT in ov.failed and not _graded_in_window(rates.get(model), anchor_day))
        units, pass_pct, was_pct, months = ((None, None, None, []) if unknown
                                            else _summarise(days, anchor_day))
        newest = None if unknown else _newest(days, anchor_day)
        parts, metric = [], None
        if entry is not None:
            parts.append(focus_reason(entry))
            metric = getattr(getattr(entry, "series", None), "metric", None) or FOCUS_METRIC
        flag = by_flag.get(model)
        if flag is not None:
            parts.append(drift_reason(statuses.get(model), flag.worst_metric))
            metric = metric or flag.worst_metric
            if pass_pct is not None and pass_pct >= STILL_PASSING:
                parts.append("still passing")
        ov.cards.append(Card(model=model, reason=" · ".join(parts), units=units, pass_pct=pass_pct,
                             was_pct=was_pct, months=months, final_test=final_test,
                             hand_trim=model in HAND_TRIM_MODELS, metric=metric,
                             lasers=_lasers(systems.get(model), anchor_day) if rates_known else None,
                             newest=newest))

    # Everything else: the active models not on a card, busiest first.
    carded = {c.model for c in ov.cards}
    for model, days in rates.items():
        if model in carded or not _graded_in_window(days, anchor_day):
            continue
        units, pass_pct, was_pct, months = _summarise(days, anchor_day)
        words, tone = trend_words(pass_pct, was_pct)
        ov.others.append(Row(model=model, units=units, pass_pct=pass_pct, was_pct=was_pct,
                             trend=words, tone=tone, months=months,
                             hand_trim=model in HAND_TRIM_MODELS,
                             lasers=_lasers(systems.get(model), anchor_day),
                             newest=_newest(days, anchor_day)))
    ov.others.sort(key=lambda r: (-r.units, r.model))

    # The dollars lost at final test, over this page's own window -- which the pass rates anchor:
    # without them (or with no trim on record) there is no window, and the total stays unknown.
    if anchor_day is not None and rates_known:
        try:
            _charge(ov, _ft_fails_by_model(db, anchor_day, now),
                    _clean_prices(prices), _clean_ratio(cost_ratio))
        except Exception as exc:
            logger.exception("Overview: the dollars lost at final test could not be worked out")
            ov.failed[PART_MONEY] = _why(exc)

    activity = None
    try:
        activity = load_activity(db, now=now)
        ov.inactive = activity.inactive()
    except Exception as exc:
        logger.exception("Overview: could not work out which models are inactive")
        ov.failed[PART_ACTIVITY] = _why(exc)
    # Every other model on file, so no model is on file and nowhere on the page (F5, James: "i dont
    # want to hide them"): each one exactly once, on a card, in the list, here, or inactive. Unknown
    # without the pass rates (then no model is known to be active) or without the activity.
    if activity is not None and PART_RATES not in ov.failed:
        try:
            shown = {c.model for c in ov.cards} | {r.model for r in ov.others} | set(ov.inactive)
            quiet: Dict[str, Optional[datetime]] = {m: d for m, d in activity.newest.items()
                                                    if m not in shown}
            # ...and a model the Models picker lists with no trim file at all: smoothness records
            # (and final tests) only -- 8213-1 and 8508 on the work data of 30 Sep, each with over
            # a thousand final tests.
            quiet.update({m: None for m in _models_on_file(db) if m not in shown and m not in quiet})
            ov.quiet = quiet
        except Exception as exc:
            logger.exception("Overview: could not work out the other models on file")
            ov.failed[PART_ON_FILE] = _why(exc)

    try:
        ov.trend = db.get_company_yield_trend(days_back=TREND_DAYS, period=TREND_PERIOD)
    except Exception as exc:
        logger.exception("Overview: the yield chart by laser could not be loaded")
        ov.failed[PART_TREND] = _why(exc)

    ov.legacy_ft = legacy_ft_count(db)            # both never raise: a count must never break a page
    ov.unreadable = unreadable_count(db)
    return ov


def _models_on_file(db) -> set:
    """Every model the Models picker can open: one with a laser file or a smoothness file. A name
    found ONLY on final tests is not one of them (138 on the 30 Sep data -- 2475-08 beside the trim
    model 2475-8, among them): a naming question, TRACKER J5, not a model to list here."""
    from sqlalchemy import select, union
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, SmoothnessResult as DBSR
    with db.session() as s:
        rows = s.execute(union(select(DBAR.model), select(DBSR.model))).fetchall()   # distinct
    return {r[0] for r in rows if r[0]}


def _why(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _as_datetime(value) -> Optional[datetime]:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("T", " ")[:26])


def _since(anchor_day: date) -> datetime:
    """The first moment any window reaches back to: the first day of the year before the 90 days
    (the twelve months' first day is always later)."""
    first = anchor_day - timedelta(days=WINDOW_DAYS - 1 + PRIOR_DAYS)
    return datetime.combine(first, time.min)


def _trim_filters(DBAR, horizon: datetime) -> list:
    """What a trim file must be to count: graded, a real model and date, believed (not dated more
    than a day ahead), and not marked suspect (the drift watch's and the FOCUS list's rule)."""
    from sqlalchemy import or_
    from laser_trim_analyzer.database.models import StatusType
    return [DBAR.overall_status.in_([StatusType[s] for s in _GRADED]),
            DBAR.model.isnot(None), DBAR.file_date.isnot(None), DBAR.file_date <= horizon,
            or_(DBAR.data_quality.is_(None), DBAR.data_quality != "suspect")]


def _rates_by_day(db, now: Optional[datetime]):
    """(anchor, {model: {day: (graded, accepted)}}, {model: {system letter: its newest graded
    day}}) -- the anchor in one MAX, then every model's counts per day and laser since the earliest
    window opens, in ONE grouped query."""
    from sqlalchemy import case, func
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, StatusType
    horizon = trusted_until(now)
    filters = _trim_filters(DBAR, horizon)
    with db.session() as s:
        anchor = _as_datetime(s.query(func.max(DBAR.file_date)).filter(*filters).scalar())
        if anchor is None:
            return None, {}, {}
        day = func.date(DBAR.file_date)
        accepted = func.sum(case((DBAR.overall_status.in_([StatusType[x] for x in _ACCEPTED]), 1),
                                 else_=0))
        rows = (s.query(DBAR.model, day, DBAR.system, func.count(DBAR.id), accepted)
                .filter(*filters, DBAR.file_date >= _since(anchor.date()))
                .group_by(DBAR.model, day, DBAR.system).all())
    systems: Dict[str, Dict[str, date]] = {}
    for model, d, system, graded, _accepted in rows:
        letter = getattr(system, "value", system)
        if not model or d is None or not letter or not graded:
            continue
        d = date.fromisoformat(str(d)[:10])
        seen = systems.setdefault(model, {})
        seen[letter] = max(d, seen.get(letter, d))
    return anchor, _by_model_day((m, d, g, a) for m, d, _s, g, a in rows), systems


def _ft_counted(DBFT, now: Optional[datetime]) -> list:
    """What a final test must be to count on this page: dated, and believed (not more than a day
    ahead) -- both reads of final tests below share it."""
    return [DBFT.file_date.isnot(None), DBFT.file_date <= trusted_until(now)]


def _ft_rates_by_day(db, models: Iterable[str], anchor_day: date, now: Optional[datetime]):
    """The same counts from final_test_results, for the few card models with no trim to show."""
    from sqlalchemy import case, func
    from laser_trim_analyzer.database.models import FinalTestResult as DBFT, StatusType
    day = func.date(DBFT.file_date)
    accepted = func.sum(case((DBFT.overall_status.in_([StatusType[x] for x in _ACCEPTED]), 1),
                             else_=0))
    with db.session() as s:
        rows = (s.query(DBFT.model, day, func.count(DBFT.id), accepted)
                .filter(DBFT.model.in_(list(models)),
                        DBFT.overall_status.in_([StatusType[x] for x in _GRADED]),
                        *_ft_counted(DBFT, now), DBFT.file_date >= _since(anchor_day))
                .group_by(DBFT.model, day).all())
    return _by_model_day(rows)


def _ft_fails_by_model(db, anchor_day: date, now: Optional[datetime]) -> Dict[str, int]:
    """{model: final-test FAILs in the 90 calendar days ending on the anchor's day} for EVERY model
    that has one -- on the page or not: the header's total is the full cost. The final tests
    `_ft_rates_by_day` counts, the FAIL ones, in the window the cards and rows count."""
    from sqlalchemy import func
    from laser_trim_analyzer.database.models import FinalTestResult as DBFT, StatusType
    first, _prior = _window(anchor_day)
    with db.session() as s:
        # By calendar DAY, as the cards' and rows' own counts are (_summarise reads date()).
        rows = (s.query(DBFT.model, func.count(DBFT.id))
                .filter(DBFT.overall_status == StatusType.FAIL, *_ft_counted(DBFT, now),
                        func.date(DBFT.file_date).between(first.isoformat(), anchor_day.isoformat()))
                .group_by(DBFT.model).all())
    return {m: int(n) for m, n in rows if m and n}


def _clean_prices(prices) -> Dict[str, float]:
    """{model: price} as floats, keyed by the model's name as text (YAML may hand either a string
    or a number for either); an entry that is not a finite number is no price at all."""
    out: Dict[str, float] = {}
    for model, price in (prices or {}).items():
        try:
            value = float(price)
        except (TypeError, ValueError):
            logger.warning("Overview: the price for %s is not a number; treated as no price", model)
            continue
        if math.isfinite(value):
            out[str(model)] = value
    return out


def _clean_ratio(cost_ratio) -> float:
    try:
        value = float(cost_ratio)
    except (TypeError, ValueError):
        return DEFAULT_COST_RATIO
    return value if math.isfinite(value) else DEFAULT_COST_RATIO


def _charge(ov: Overview, fails: Dict[str, int], prices: Dict[str, float], ratio: float) -> None:
    """Each card's and row's final-test fails and dollars (its own); the header's total and the
    unpriced count over EVERY model that failed final test in the window; whether any price was
    loaded at all."""
    for item in list(ov.cards) + list(ov.others):
        item.ft_fails = fails.get(item.model, 0)
        price = prices.get(item.model)
        item.money = None if price is None else item.ft_fails * price * ratio
    ov.money_total = sum((n * prices[m] * ratio for m, n in fails.items() if m in prices), 0.0)
    ov.unpriced = sum(1 for m, n in fails.items() if n and m not in prices)
    ov.no_prices = not prices


def _lasers(newest_by_system: Optional[Dict[str, date]], anchor_day: Optional[date]) -> List[str]:
    """The systems with a graded trim in the 90 days, in the shop's order (laser 1, 2, 3); any
    other letter after them."""
    if not newest_by_system or anchor_day is None:
        return []
    first, _prior = _window(anchor_day)
    inside = [s for s, d in newest_by_system.items() if first <= d <= anchor_day]
    return sorted(inside, key=lambda s: (LASER_ORDER.index(s) if s in LASER_ORDER
                                         else len(LASER_ORDER), s))


def _newest(days: Optional[Dict[date, Tuple[int, int]]], anchor_day: Optional[date]) -> Optional[date]:
    """The newest day with a counted file, up to the anchor's day (a final test after it is past
    every window on the page)."""
    if not days or anchor_day is None:
        return None
    return max((d for d, (graded, _a) in days.items() if graded and d <= anchor_day), default=None)


def _by_model_day(rows) -> Dict[str, Dict[date, Tuple[int, int]]]:
    """{model: {day: (graded, accepted)}}, summed over any rows a day has (one per laser)."""
    out: Dict[str, Dict[date, Tuple[int, int]]] = {}
    for model, day, graded, accepted in rows:
        if not model or day is None:
            continue
        days = out.setdefault(model, {})
        d = date.fromisoformat(str(day)[:10])
        g, a = days.get(d, (0, 0))
        days[d] = (g + int(graded or 0), a + int(accepted or 0))
    return out


def _window(anchor_day: date) -> Tuple[date, date]:
    """(first day of the 90 days, first day of the year before them)."""
    first = anchor_day - timedelta(days=WINDOW_DAYS - 1)
    return first, first - timedelta(days=PRIOR_DAYS)


def _graded_in_window(days: Optional[Dict[date, Tuple[int, int]]], anchor_day: Optional[date]) -> bool:
    if not days or anchor_day is None:
        return False
    first, _prior = _window(anchor_day)
    return any(first <= d <= anchor_day and graded for d, (graded, _a) in days.items())


def month_starts(anchor: Optional[datetime]) -> List[date]:
    """The first day of each of the MONTHS months a card's or row's `months` covers, oldest first
    -- [] with no anchor. What a chart of `months` names its points by."""
    if anchor is None:
        return []
    return [date(y, m + 1, 1) for y, m in _month_keys(anchor.date())]


def _month_keys(anchor_day: date) -> List[Tuple[int, int]]:
    """The MONTHS (year, month) pairs ending in the anchor's month, oldest first."""
    last = anchor_day.year * 12 + anchor_day.month - 1
    return [divmod(k, 12) for k in range(last - MONTHS + 1, last + 1)]


def _summarise(days: Optional[Dict[date, Tuple[int, int]]], anchor_day: Optional[date]):
    """(units, pass %, was %, months) for one model's per-day counts."""
    if anchor_day is None:
        return 0, None, None, []
    keys = _month_keys(anchor_day)
    month_counts = {k: [0, 0] for k in keys}
    first, prior = _window(anchor_day)
    graded = accepted = p_graded = p_accepted = 0
    for d, (g, a) in (days or {}).items():
        if d > anchor_day:
            continue                       # a final test after the newest trim: past the window
        if d >= first:
            graded, accepted = graded + g, accepted + a
        elif d >= prior:
            p_graded, p_accepted = p_graded + g, p_accepted + a
        slot = month_counts.get((d.year, d.month - 1))
        if slot is not None:
            slot[0] += g
            slot[1] += a
    months = [(100.0 * a / g) if g else None for g, a in (month_counts[k] for k in keys)]
    return (graded, (100.0 * accepted / graded) if graded else None,
            (100.0 * p_accepted / p_graded) if p_graded else None, months)
