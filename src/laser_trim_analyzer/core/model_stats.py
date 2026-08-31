"""The historical stats table — what James exports a model to Excel to compute.

James's week (app-shape spec, 2026-08-29): process folders, then export a model
to Excel *just* to work out historical average / min / max for resistance and
electrical angle, all units versus lin-passing units, and read it back. This
module is that arithmetic, so the screen can answer it and the round trip stops.

WHY IT CANNOT BE `AVG()`
------------------------
`untrimmed_resistance` and `trimmed_resistance` carry readings that are not
measurements: 1e12 Ω (the tester's open-circuit rail), 2.1e8 Ω, and 0. On 6607
— the model most likely to be opened first — SEVEN such readings out of 9,943
drag the average from a true 4,282 Ω to 32,079 Ω (7.5x wrong) and the maximum
from 29.6 kΩ to 2.1e8 Ω. A stats table that shipped those numbers would fail
the exact Excel round trip it exists to replace, so every metric runs through a
declared plausibility policy before anything is averaged.

DISCLOSE, NEVER HIDE. Every cell carries the count it dropped (`excluded`) and
the count that was never recorded (`missing`) next to the n it actually used.
Same rule as the future-dated-record warning in `core/yield_stats.py`: a number
quietly laundered is a source file nobody ever fixes.

THE TWO COLUMNS (D2)
--------------------
  * ALL          — every track row carrying a usable value, whatever status.
  * LIN-PASSING  — rows whose PARENT unit's `overall_status` is PASS or
                   WARNING. Linearity is the zero-tolerance customer
                   disposition; WARNING is the internal sigma drift-watch on
                   units that were ACCEPTED. Same rule as unit-level yield
                   (`core/yield_stats.compute_yield`) and the SPC fail flag.
The metrics live on `track_results` and the disposition lives on the parent
`analysis_results`, so the query joins — it never infers one from the other.

Two layers, in the order `ml/spc.py` established: `build_model_stats` is PURE
(no DB, no Tk — cheap to test, safe anywhere), and `compute_model_stats` is the
only part that reads the database. Neither raises into the GUI: an unreadable
database degrades to an empty table with a note, exactly like
`core/spec_alignment.py`, because the caller is a page that may not go down
over a table.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from statistics import median
from typing import List, NamedTuple, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---- plausibility policies -------------------------------------------------
# FINITE          drop None/NaN/inf, keep everything else. The default, and the
#                 only safe policy for a metric that can legitimately be
#                 negative or zero.
# POSITIVE_RATIO  drop non-finite, drop <= 0, then drop anything outside
#                 [median/RATIO_BAND, median*RATIO_BAND] where the median is
#                 taken over the POSITIVE values of that model+metric within
#                 the window.
FINITE = "finite"
POSITIVE_RATIO = "positive_ratio"

# 100x either side of the model's own median. Validated over the whole work
# database (2026-08-30): it drops 40 of 107,872 `untrimmed_resistance` readings
# and 30 of 100,721 positive `trimmed_resistance` readings, and with ONE
# exception every drop is an obvious corruption (1e12, 6.7e8, 5.3e7). It clips
# nothing real: 8630 keeps 29.3k-38.8k around a 35.7k median and drops only its
# four 1e12 rows. The exception is 8652 — four UNTRIMMED test sweeps split
# 2x~340 kΩ / 2x237 Ω, where the median falls between the two clusters and the
# smaller pair is dropped; the cell discloses it, which is the point.
RATIO_BAND = 100.0

# One entry per metric. Adding a metric to the table is one line here plus one
# line in _DISTRIBUTION_ROWS — that is the whole reason this is a table and not
# a chain of if-statements.
_POLICY = {
    "untrimmed_resistance": POSITIVE_RATIO,
    "trimmed_resistance": POSITIVE_RATIO,
    # NOT ratio-scale: the real database holds 2,416 legitimate NEGATIVE
    # electrical angles. Treating this column as a positive ratio would delete
    # real measurements, which is a worse lie than the one we are fixing.
    "measured_electrical_angle": FINITE,
    "final_linearity_error_shifted": FINITE,
    "margin_to_spec": FINITE,
    # FINITE deliberately, despite reaching 999.999 in the database: on 8340-1
    # every sigma_gradient above 1.0 belongs to a FAILED unit, so those are the
    # real signal the ALL column exists to show, not corruption. See the report
    # note — the 94 rows at exactly 999.999 look like a saturation sentinel and
    # are a data question, not a filtering one.
    "sigma_gradient": FINITE,
}


def metric_policy(metric: str) -> str:
    """The declared policy for a metric (FINITE for anything unlisted)."""
    return _POLICY.get(metric, FINITE)


# ---- the table's shape -----------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """One (row, column) box: the numbers, and what did not make it into them.

    Distribution rows fill avg/low/high; rate rows fill count/pct. Both always
    carry `n` (readings actually used), `excluded` (present but unusable — a
    non-finite or implausible reading) and `missing` (nothing was recorded).
    n + excluded + missing is the number of track rows the column considered,
    so a reader can always see what happened to every row.
    """
    n: int
    excluded: int
    missing: int
    avg: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    count: Optional[int] = None
    pct: Optional[float] = None

    @property
    def considered(self) -> int:
        return self.n + self.excluded + self.missing


@dataclass(frozen=True)
class StatRow:
    """One line of the table: the same metric under both dispositions."""
    key: str
    label: str
    unit: str                    # "ohms" | "deg" | "%" | "" — display hint only
    kind: str                    # "distribution" | "rate"
    all_: Cell
    lin_passing: Cell


@dataclass(frozen=True)
class ModelStats:
    """The whole table for one (model, window) — plus what it had to drop."""
    model: str
    rows: List[StatRow]
    tracks: int                  # track rows in the window (any status)
    units: int                   # distinct parent units in the window
    cutoff: Optional[datetime]
    lot: Optional[Tuple[datetime, datetime]]
    future_dated: int            # file-date junk skipped (see compute_yield)
    note: str                    # one plain sentence for the UI; "" when clean

    @property
    def distribution_rows(self) -> List[StatRow]:
        return [r for r in self.rows if r.kind == "distribution"]

    @property
    def rate_rows(self) -> List[StatRow]:
        return [r for r in self.rows if r.kind == "rate"]

    @property
    def excluded_total(self) -> int:
        """Implausible readings dropped anywhere in the ALL column.

        The ALL column only: LIN-PASSING is a subset of the same rows, and
        adding the two would count the same corrupt reading twice.
        """
        return sum(r.all_.excluded for r in self.rows)


class TrackRecord(NamedTuple):
    """One track row, as the bulk query selects it. Ordered, so the loader can
    hand `TrackRecord(*row)` straight through."""
    status: str                  # the PARENT unit's overall_status
    untrimmed_resistance: Optional[float]
    trimmed_resistance: Optional[float]
    measured_electrical_angle: Optional[float]
    final_linearity_error_shifted: Optional[float]
    margin_to_spec: Optional[float]
    sigma_gradient: Optional[float]
    linearity_pass: Optional[bool]
    untrimmed_error_max: Optional[float]
    linearity_spec: Optional[float]
    unit_id: Optional[int] = None


# key, label, unit
_DISTRIBUTION_ROWS = [
    ("untrimmed_resistance", "Untrimmed resistance", "ohms"),
    ("trimmed_resistance", "Trimmed resistance", "ohms"),
    ("measured_electrical_angle", "Electrical angle (measured)", "deg"),
    ("final_linearity_error_shifted", "Linearity error (max)", ""),
    ("margin_to_spec", "Margin to spec limit", "%"),
    ("sigma_gradient", "Sigma gradient", ""),
]

# The LIN-PASSING population: the parent unit's disposition, not the track's.
_LIN_PASSING = ("PASS", "WARNING")


def _status_name(status) -> str:
    return str(getattr(status, "name", None) or status).upper()


def is_lin_passing(status) -> bool:
    """Was this unit ACCEPTED on the customer's linearity requirement?

    PASS and WARNING both were: WARNING is the internal sigma watch on product
    that shipped. FAIL is the linearity rejection; ERROR and UNTRIMMED are not
    dispositions at all, so they are not in the accepted population either.
    """
    return _status_name(status) in _LIN_PASSING


# ---- the pure computation --------------------------------------------------

def _usable(value) -> Optional[float]:
    """A float we can do arithmetic with, or None (NULL, NaN, inf, non-numeric)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if isfinite(f) else None


def _band(values: Sequence[Optional[float]], policy: str
          ) -> Optional[Tuple[float, float]]:
    """The plausibility window for one (metric, population), or None for none.

    Computed ONCE per metric over the whole window's ALL population and then
    applied to both columns, so a reading the ALL column calls corrupt can
    never be quietly kept in the LIN-PASSING column.
    """
    if policy != POSITIVE_RATIO:
        return None
    positives = [v for v in values if v is not None and v > 0]
    if not positives:
        return None
    mid = median(positives)
    return (mid / RATIO_BAND, mid * RATIO_BAND)


def _distribution_cell(raw: Sequence, policy: str,
                       band: Optional[Tuple[float, float]]) -> Cell:
    """n / avg / min / max over what survived, with both drop counts kept."""
    kept: List[float] = []
    excluded = missing = 0
    for value in raw:
        if value is None:
            missing += 1
            continue
        f = _usable(value)
        if f is None:                       # NaN/inf: a reading, but not a number
            excluded += 1
            continue
        if policy == POSITIVE_RATIO:
            if f <= 0 or (band is not None and not (band[0] <= f <= band[1])):
                excluded += 1
                continue
        kept.append(f)
    if not kept:
        return Cell(n=0, excluded=excluded, missing=missing)
    return Cell(n=len(kept), excluded=excluded, missing=missing,
                avg=sum(kept) / len(kept), low=min(kept), high=max(kept))


def _rate_cell(flags: Sequence[Optional[bool]]) -> Cell:
    """n / count / pct over rows that could be judged at all.

    A row whose inputs were NULL is `missing`, never a 0 in the denominator —
    that is the difference between "10% of what we measured" and "10% of what
    we ran", and only the first one is true.
    """
    n = count = missing = 0
    for flag in flags:
        if flag is None:
            missing += 1
            continue
        n += 1
        count += 1 if flag else 0
    return Cell(n=n, excluded=0, missing=missing, count=count,
                pct=(count / n * 100.0) if n else None)


def _trim_passed(rec: TrackRecord) -> Optional[bool]:
    """Did the track pass linearity at the trim station? (`linearity_pass`.)"""
    return None if rec.linearity_pass is None else bool(rec.linearity_pass)


def _already_met_spec(rec: TrackRecord) -> Optional[bool]:
    """Was the part already inside the linearity spec BEFORE the laser ran?

    ASSUMPTION, flagged deliberately (2026-08-30): James asked for "% needing
    no trim (already met spec before trim)" and did not define the columns.
    This reads it as `untrimmed_error_max <= linearity_spec` — the worst point
    of the raw pre-trim sweep against the model's own spec, both non-NULL. It
    is conservative (no offset correction is applied first, so these are
    definite cases) and it is the same expression `yield_stats
    .compute_trim_necessity` uses for the same question.

    It is NOT the same POPULATION as that helper: this row is per TRACK over
    the selected window, so it lines up with the distribution rows above it,
    while `compute_trim_necessity` is per UNIT and counts only units that were
    actually trimmed. Two questions, deliberately not merged.
    """
    err = _usable(rec.untrimmed_error_max)
    spec = _usable(rec.linearity_spec)
    if err is None or spec is None:
        return None
    return err <= spec


_RATE_ROWS = [
    ("trim_passed_linearity", "Passed linearity at trim", "%", _trim_passed),
    ("already_met_spec", "Already met spec before trim", "%", _already_met_spec),
]


def build_model_stats(model: str, records: Sequence[TrackRecord], *,
                      cutoff: Optional[datetime] = None,
                      lot: Optional[Tuple[datetime, datetime]] = None,
                      future_dated: int = 0, note: str = "") -> ModelStats:
    """The table, from track records already loaded. Pure: no DB, no Tk.

    ONE PASS over the records per metric — the columns are split by the parent
    unit's status while the values are being read, never by re-querying.
    """
    lin_mask = [is_lin_passing(r.status) for r in records]
    rows: List[StatRow] = []
    for key, label, unit in _DISTRIBUTION_ROWS:
        policy = metric_policy(key)
        all_raw = [getattr(r, key) for r in records]
        # The band comes from the ALL population so both columns share it.
        band = _band([_usable(v) for v in all_raw], policy)
        lin_raw = [v for v, keep in zip(all_raw, lin_mask) if keep]
        rows.append(StatRow(key=key, label=label, unit=unit, kind="distribution",
                            all_=_distribution_cell(all_raw, policy, band),
                            lin_passing=_distribution_cell(lin_raw, policy, band)))
    for key, label, unit, flag_of in _RATE_ROWS:
        all_flags = [flag_of(r) for r in records]
        lin_flags = [f for f, keep in zip(all_flags, lin_mask) if keep]
        rows.append(StatRow(key=key, label=label, unit=unit, kind="rate",
                            all_=_rate_cell(all_flags),
                            lin_passing=_rate_cell(lin_flags)))
    units = len({r.unit_id for r in records if r.unit_id is not None})
    return ModelStats(model=model, rows=rows, tracks=len(records),
                      units=units or len(records), cutoff=cutoff, lot=lot,
                      future_dated=future_dated, note=note)


def _empty(model: str, note: str, cutoff=None, lot=None) -> ModelStats:
    """The honest nothing: the same eight rows, all reading n=0."""
    return build_model_stats(model, [], cutoff=cutoff, lot=lot, note=note)


# ---- DB layer — everything above is pure -----------------------------------

def _horizon() -> datetime:
    """One day past the wall clock: the file-date junk cut.

    Same cut `core/yield_stats.compute_yield` and `ml/spc._horizon` make, for
    the same reason — a mistyped filename date puts a record months in the
    future, and a table whose max came from a row the chart refuses to draw
    would be one more surface telling a different story.
    """
    return datetime.now() + timedelta(days=1)


def compute_model_stats(db, model: str, *, cutoff: Optional[datetime] = None,
                        lot: Optional[Tuple[datetime, datetime]] = None
                        ) -> ModelStats:
    """The stats table for one (model, window). ONE query, one pass.

    `cutoff` floors the window (file_date >= cutoff; None = the whole record).
    `lot` is an inclusive (start, end) day pair — a production run from the
    shared clustering (`ml/lots.py`); the end day is included in full, because
    `Lot.end` is midnight-normalized to the last production DAY.

    Deliberately ONE bulk column-select feeding every row: a session loop per
    metric is what made the old pages slow, and eight metrics over 10,000 track
    rows is one query's worth of work.

    Never raises. A read that fails returns the empty table with a note, so a
    worker thread can hand the result straight to the page.
    """
    if not model:
        return _empty(model, "no model selected", cutoff, lot)
    try:
        from laser_trim_analyzer.database.models import AnalysisResult as DBAR
        from laser_trim_analyzer.database.models import TrackResult as DBTR
        with db.session() as s:
            q = (s.query(DBAR.overall_status,
                         DBTR.untrimmed_resistance, DBTR.trimmed_resistance,
                         DBTR.measured_electrical_angle,
                         DBTR.final_linearity_error_shifted, DBTR.margin_to_spec,
                         DBTR.sigma_gradient, DBTR.linearity_pass,
                         DBTR.untrimmed_error_max, DBTR.linearity_spec,
                         DBAR.id)
                 .join(DBTR, DBTR.analysis_id == DBAR.id)
                 .filter(DBAR.model == model))
            horizon = _horizon()
            if cutoff is not None:
                q = q.filter(DBAR.file_date >= cutoff)
            if lot is not None:
                # `< end + 1 day` rather than `<= end`: the lot's end is a DAY,
                # and a file stamped later that day belongs to that run.
                q = q.filter(DBAR.file_date >= lot[0],
                             DBAR.file_date < lot[1] + timedelta(days=1))
            junk = q.filter(DBAR.file_date > horizon).count()
            rows = q.filter((DBAR.file_date <= horizon)
                            | (DBAR.file_date.is_(None))).all()
    except Exception:
        logger.exception("model stats: query failed for %s", model)
        return _empty(model, "could not read the database for this model",
                      cutoff, lot)
    if not rows:
        return _empty(model, f"no track measurements recorded for {model}"
                             " in this window", cutoff, lot)
    note = ""
    if junk:
        note = (f"{junk} record(s) dated in the future were left out — "
                "fix the source filename date")
    return build_model_stats(model, [TrackRecord(*r) for r in rows],
                             cutoff=cutoff, lot=lot, future_dated=junk,
                             note=note)
