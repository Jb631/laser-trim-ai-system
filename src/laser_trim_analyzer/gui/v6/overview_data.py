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

PASS % is the app's headline yield (`core/yield_stats.compute_yield`'s linearity_yield):
(PASS + WARNING) / (PASS + WARNING + FAIL) -- a WARNING is a sigma watch, never a rejection
(CLAUDE.md). ERROR / PROCESSING_FAILED and UNTRIMMED are not graded; a file dated more than a day
ahead of now is not believed (`core/activity.trusted_until`); and a file marked suspect
(`data_quality = 'suspect'`) is left out -- the rule the drift watch and the FOCUS list follow
since 2026-10-02, so a card's number and the reason printed under it are read from the same files.

THE WINDOWS are anchored on the newest file that counts -- never the wall clock, like every other
"recent" in the app, so a copy of an older database reads as it did: the 90 days ending on that
file's day; "was" is the 365 days before them; the bars are the 12 calendar months ending in that
file's month. Every model's counts come from ONE grouped query (per model and day), never one
query per model.

Worker-safe: no Tk, no widget. Each part that fails is NAMED in `Overview.failed` and the others
still load: a crash must never read as "0 models need a look" (CLAUDE.md: a failure must never
look like a result).
"""
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

from laser_trim_analyzer.core.activity import load_activity, trusted_until
from laser_trim_analyzer.core.ft_regrade import legacy_ft_count
from laser_trim_analyzer.core.ingest_run import unreadable_count
from laser_trim_analyzer.gui.v6.focus_data import focus_failed, load_focus
from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import format_metric_value, metric_label
from laser_trim_analyzer.ml.manager import (
    drift_reference_date, get_drifting_models, get_model_drift_status)

logger = logging.getLogger(__name__)

# Hand-trimmed AFTER the laser, so a laser PASS on them is hand-trim workload, not yield: their cards
# and rows carry a "hand trim" tag. ONE list, core/model_names (it also sets their 20x suspect line).
from laser_trim_analyzer.core.model_names import HAND_TRIM_MODELS  # noqa: E402

WINDOW_DAYS = 90        # "Last 90 days": what every card and row is about
PRIOR_DAYS = 365        # "was X%": the year before those 90 days
MONTHS = 12             # the bars: twelve calendar months, ending in the newest file's month
STEADY_POINTS = 5       # within +/-5 points of the year before reads "steady"
STILL_PASSING = 99.0    # a signal that moved on a model passing >= 99% says it is still passing

# The parts that load (and fail) on their own, worded as the page's banner names them.
PART_FOCUS = "the drifting-now list"
PART_DRIFT = "the drift watch's flags"
PART_RATES = "the pass rates"
PART_FT = "the final-test pass rates"     # read only for a card with no trim to show
PART_ACTIVITY = "which models are inactive"

_GRADED = ("PASS", "WARNING", "FAIL")
_ACCEPTED = ("PASS", "WARNING")

Months = List[Optional[float]]


@dataclass
class Card:
    """One model that needs a look, and the line saying why."""
    model: str
    reason: str
    units: int = 0                       # graded trims in the 90 days (final tests if final_test)
    pass_pct: Optional[float] = None     # 0..100; None with nothing graded
    was_pct: Optional[float] = None      # the year before; None with nothing graded then
    months: Months = field(default_factory=list)    # MONTHS values, oldest first
    final_test: bool = False             # no trims to show: these are its final-test numbers
    hand_trim: bool = False


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


@dataclass
class Overview:
    anchor: Optional[datetime] = None    # the newest file that counts; the windows end on its day
    cards: List[Card] = field(default_factory=list)
    others: List[Row] = field(default_factory=list)
    # {model: its newest trim file, or None for "no trims on record"}; None = could not be worked out
    inactive: Optional[Dict[str, Optional[datetime]]] = None
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


def window_caption(ov: Overview) -> str:
    """"Last 90 days · newest file 30 Sep 2026" -- the clock every number on the page uses."""
    if ov.anchor is None:
        return "Last 90 days" if PART_RATES in ov.failed else "No trim files on record yet"
    d = ov.anchor
    return f"Last {WINDOW_DAYS} days · newest file {d.day} {d:%b %Y}"    # NOT %-d: raises on Windows


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

def load_overview(db, *, now: Optional[datetime] = None) -> Overview:
    """Everything the Overview draws. Never raises: each part that fails is named in `failed`."""
    ov = Overview()

    # Pass rates first: the anchor (and so every window) comes from them.
    rates: Dict[str, Dict[date, Tuple[int, int]]] = {}
    try:
        ov.anchor, rates = _rates_by_day(db, now)
    except Exception as exc:
        logger.exception("Overview: the pass rates could not be worked out")
        ov.failed[PART_RATES] = _why(exc)
    anchor_day = ov.anchor.date() if ov.anchor is not None else None

    # The two card sources, each on its own guard.
    focus = []
    result, _last = load_focus(db)               # never raises; a crash comes back marked
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
        units, pass_pct, was_pct, months = _summarise(days, anchor_day)
        parts = []
        if entry is not None:
            parts.append(focus_reason(entry))
        flag = by_flag.get(model)
        if flag is not None:
            parts.append(drift_reason(statuses.get(model), flag.worst_metric))
            if pass_pct is not None and pass_pct >= STILL_PASSING:
                parts.append("still passing")
        ov.cards.append(Card(model=model, reason=" · ".join(parts), units=units, pass_pct=pass_pct,
                             was_pct=was_pct, months=months, final_test=final_test,
                             hand_trim=model in HAND_TRIM_MODELS))

    # Everything else: the active models not on a card, busiest first.
    carded = {c.model for c in ov.cards}
    for model, days in rates.items():
        if model in carded or not _graded_in_window(days, anchor_day):
            continue
        units, pass_pct, was_pct, months = _summarise(days, anchor_day)
        words, tone = trend_words(pass_pct, was_pct)
        ov.others.append(Row(model=model, units=units, pass_pct=pass_pct, was_pct=was_pct,
                             trend=words, tone=tone, months=months,
                             hand_trim=model in HAND_TRIM_MODELS))
    ov.others.sort(key=lambda r: (-r.units, r.model))

    try:
        ov.inactive = load_activity(db, now=now).inactive()
    except Exception as exc:
        logger.exception("Overview: could not work out which models are inactive")
        ov.failed[PART_ACTIVITY] = _why(exc)

    ov.legacy_ft = legacy_ft_count(db)            # both never raise: a count must never break a page
    ov.unreadable = unreadable_count(db)
    return ov


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
    """(anchor, {model: {day: (graded, accepted)}}) -- the anchor in one MAX, then every model's
    counts per day since the earliest window opens, in ONE grouped query."""
    from sqlalchemy import case, func
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, StatusType
    horizon = trusted_until(now)
    filters = _trim_filters(DBAR, horizon)
    with db.session() as s:
        anchor = _as_datetime(s.query(func.max(DBAR.file_date)).filter(*filters).scalar())
        if anchor is None:
            return None, {}
        day = func.date(DBAR.file_date)
        accepted = func.sum(case((DBAR.overall_status.in_([StatusType[x] for x in _ACCEPTED]), 1),
                                 else_=0))
        rows = (s.query(DBAR.model, day, func.count(DBAR.id), accepted)
                .filter(*filters, DBAR.file_date >= _since(anchor.date()))
                .group_by(DBAR.model, day).all())
    return anchor, _by_model_day(rows)


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
                        DBFT.file_date.isnot(None), DBFT.file_date <= trusted_until(now),
                        DBFT.file_date >= _since(anchor_day))
                .group_by(DBFT.model, day).all())
    return _by_model_day(rows)


def _by_model_day(rows) -> Dict[str, Dict[date, Tuple[int, int]]]:
    out: Dict[str, Dict[date, Tuple[int, int]]] = {}
    for model, day, graded, accepted in rows:
        if not model or day is None:
            continue
        out.setdefault(model, {})[date.fromisoformat(str(day)[:10])] = (int(graded or 0),
                                                                         int(accepted or 0))
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
