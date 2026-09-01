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

A RECORD THAT FAILED PROCESSING IS NOT A MEASUREMENT
----------------------------------------------------
The same failure arrives a second way, through a column no value policy can
catch. All 94 ERROR records in the work database carry `sigma_gradient =
999.999` — the analyzer's saturation sentinel, written when the file could not
be analysed — and none of them carries a resistance at all. Averaged in, they
put 8856's sigma gradient at 433.5 against a true 0.0012: a 430,000x error, on
a model with 173 rows. So ERROR (and PROCESSING_FAILED) records are dropped
from BOTH columns before anything is computed, and counted in `errored`.

No value policy would have caught this: a 100x band around a 0.001 median is
[0.00001, 0.1], which would delete 8340-1's legitimate 1.x sigma gradients on
FAILED units. The record's own status is what says these are not measurements.
This is also the convention the rest of the app already follows —
`core/yield_stats.compute_yield` excludes ERROR from both yield denominators.

DISCLOSE, NEVER HIDE. Every cell carries the count it dropped (`excluded`), the
count from records that failed processing (`errored`) and the count that was
never recorded (`missing`), next to the n it actually used. Same rule as the
future-dated-record warning in `core/yield_stats.py`: a number quietly
laundered is a source file nobody ever fixes.

THE TWO COLUMNS (D2)
--------------------
  * ALL          — every track row carrying a usable value, whatever the
                   DISPOSITION: passes, sigma watches, linearity rejects and
                   untrimmed test sweeps all count. Only records whose
                   processing FAILED are out, for the reason above.
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
    # FINITE deliberately. 8340-1's sigma gradients above 1.0 all belong to
    # FAILED units — real signal the ALL column exists to show — and a ratio
    # band around this metric's ~0.001 median would delete them. The 999.999
    # sentinels that used to poison this row are ERROR records, removed by
    # status before any policy runs (see the module docstring).
    "sigma_gradient": FINITE,
}


def metric_policy(metric: str) -> str:
    """The declared policy for a metric (FINITE for anything unlisted)."""
    return _POLICY.get(metric, FINITE)


# ---- the table's shape -----------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """One (row, column) box: the numbers, and what did not make it into them.

    Distribution rows fill avg/low/high; rate rows fill count/pct. Every cell
    carries four counts, kept separate because they mean four different things
    to whoever has to act on them:
      * `n`        — readings actually used.
      * `excluded` — a reading that is not a measurement (non-finite, or
                     outside the metric's plausibility band). A source file
                     to go and fix.
      * `errored`  — records whose PROCESSING failed, dropped whole. Nothing
                     was measured; the analyser is what needs looking at.
      * `missing`  — nothing was recorded in that column at all.
    Their sum is every track row the column considered, so a reader can always
    account for every row.
    """
    n: int
    excluded: int
    missing: int
    avg: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    count: Optional[int] = None
    pct: Optional[float] = None
    errored: int = 0

    @property
    def considered(self) -> int:
        return self.n + self.excluded + self.missing + self.errored


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
    tracks: int                  # USABLE track rows in the window (any status
                                 # except a failed-processing record)
    # Distinct analysis RECORDS — files, not unit-days. A dual-track unit
    # writes one file per track, and a re-trimmed track writes one file per
    # attempt, so this is deliberately NOT called `units`: a unit-day spans
    # several records and its key is `analysis_results.unit_id`. Nothing on
    # this table is a unit-level statistic (see `_RATE_ROWS`).
    records: int
    cutoff: Optional[datetime]
    lot: Optional[Tuple[datetime, datetime]]
    future_dated: int            # file-date junk skipped (see compute_yield)
    note: str                    # one plain sentence for the UI; "" when clean
    errored: int = 0             # records dropped whole: processing failed

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
    record_id: Optional[int] = None


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
# Records whose PROCESSING failed. Not a disposition — a file the analyser
# could not read — so the values on them are not measurements. Both names
# bucket together in `core/yield_stats.compute_yield`; they do here too.
_FAILED_PROCESSING = ("ERROR", "PROCESSING_FAILED")


def _status_name(status) -> str:
    return str(getattr(status, "name", None) or status).upper()


def is_lin_passing(status) -> bool:
    """Was this unit ACCEPTED on the customer's linearity requirement?

    PASS and WARNING both were: WARNING is the internal sigma watch on product
    that shipped. FAIL is the linearity rejection; ERROR and UNTRIMMED are not
    dispositions at all, so they are not in the accepted population either.
    """
    return _status_name(status) in _LIN_PASSING


def failed_processing(status) -> bool:
    """Did this record fail to process at all?

    Then its columns hold whatever the analyser left behind, not readings —
    in the work database, all 94 such rows carry `sigma_gradient = 999.999`
    and no resistance whatsoever. UNTRIMMED is deliberately NOT here: an
    untrimmed test sweep really did measure the part before the laser ran.
    """
    return _status_name(status) in _FAILED_PROCESSING


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
                       band: Optional[Tuple[float, float]],
                       errored: int = 0) -> Cell:
    """n / avg / min / max over what survived, with every drop count kept."""
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
        if not _plausible(f, policy, band):
            excluded += 1
            continue
        kept.append(f)
    if not kept:
        return Cell(n=0, excluded=excluded, missing=missing, errored=errored)
    return Cell(n=len(kept), excluded=excluded, missing=missing, errored=errored,
                avg=sum(kept) / len(kept), low=min(kept), high=max(kept))


def _rate_cell(flags: Sequence[Optional[bool]], errored: int = 0) -> Cell:
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
    return Cell(n=n, excluded=0, missing=missing, errored=errored, count=count,
                pct=(count / n * 100.0) if n else None)


def _trim_passed(rec: TrackRecord) -> Optional[bool]:
    """Did THIS TRACK ROW pass linearity at the trim station?

    Per TRACK, per ATTEMPT — not a unit yield, which is why the row is called
    "Tracks that passed linearity". See `_RATE_ROWS` for why.
    """
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

    It is NOT the same POPULATION as that helper: this row is per TRACK ROW
    over the selected window, so it lines up with the distribution rows above
    it, while `compute_trim_necessity` is per UNIT and counts only units that
    were actually trimmed. Two questions, deliberately not merged — see
    `_RATE_ROWS`.
    """
    err = _usable(rec.untrimmed_error_max)
    spec = _usable(rec.linearity_spec)
    if err is None or spec is None:
        return None
    return err <= spec


# THE RATE ROWS COUNT TRACK ROWS, AND THEIR NAMES SAY SO.
#
# A UNIT-DAY is not one row. It spans several `analysis_results` rows in two
# ways that need opposite treatment: one file PER TRACK (a dual-track unit
# writes Track A and Track B minutes apart), and one file per RE-TRIM ATTEMPT
# (a failing track re-trimmed until it passes, only the last attempt counting).
# The unit-level rule — per track take the LAST attempt of the day, then the
# unit passes only if EVERY track's last attempt passed, grouped on
# `analysis_results.unit_id` — is real and it MATTERS: on 6607, 1,087 of 4,841
# unit-days (22%) have some track passing while not all tracks do, and linearity
# is zero-tolerance, so those units failed. (An earlier 1,368 here came from me
# and was wrong: it counted a track's SUPERSEDED failing attempts as masking,
# which is exactly what the last-attempt rule exists to discount. Collapse
# re-trims first — ROW_NUMBER over (unit_id, track_id) — then test the tracks.)
#
# Counting track rows separately is CORRECT for the distributions above — a
# two-track unit genuinely contributes two measurements — and it is a DIFFERENT
# QUESTION from unit yield here. Rather than mix two grains in one table (where
# the `n` column would silently mean two things down one column), these rows
# stay per-track and are NAMED per-track, so nobody can read a track rate as a
# unit yield. The unit basis already exists, computed properly, in
# `core/yield_stats.py` (`compute_unit_yield_monthly`) and on the Dashboard.
_RATE_ROWS = [
    ("trim_passed_linearity", "Tracks that passed linearity", "%", _trim_passed),
    ("already_met_spec", "Tracks already in spec before trim", "%",
     _already_met_spec),
]


def build_model_stats(model: str, records: Sequence[TrackRecord], *,
                      cutoff: Optional[datetime] = None,
                      lot: Optional[Tuple[datetime, datetime]] = None,
                      future_dated: int = 0, note: str = "") -> ModelStats:
    """The table, from track records already loaded. Pure: no DB, no Tk.

    ONE PASS over the records per metric — the columns are split by the parent
    unit's status while the values are being read, never by re-querying.

    Records whose processing FAILED come out first, before any metric is
    looked at: their columns hold sentinels, not readings. They are counted on
    the ALL cells (`errored`) so the drop is visible; the LIN-PASSING cells
    read 0, because an errored record was never in that population to begin
    with — same as a linearity reject.
    """
    errored = sum(1 for r in records if failed_processing(r.status))
    usable = [r for r in records if not failed_processing(r.status)]
    lin_mask = [is_lin_passing(r.status) for r in usable]
    rows: List[StatRow] = []
    for key, label, unit in _DISTRIBUTION_ROWS:
        policy = metric_policy(key)
        all_raw = [getattr(r, key) for r in usable]
        # The band comes from the ALL population so both columns share it.
        band = _band([_usable(v) for v in all_raw], policy)
        lin_raw = [v for v, keep in zip(all_raw, lin_mask) if keep]
        rows.append(StatRow(key=key, label=label, unit=unit, kind="distribution",
                            all_=_distribution_cell(all_raw, policy, band, errored),
                            lin_passing=_distribution_cell(lin_raw, policy, band)))
    for key, label, unit, flag_of in _RATE_ROWS:
        all_flags = [flag_of(r) for r in usable]
        lin_flags = [f for f, keep in zip(all_flags, lin_mask) if keep]
        rows.append(StatRow(key=key, label=label, unit=unit, kind="rate",
                            all_=_rate_cell(all_flags, errored),
                            lin_passing=_rate_cell(lin_flags)))
    records = len({r.record_id for r in usable if r.record_id is not None})
    notes = [note] if note else []
    if errored:
        notes.append(f"{errored} record(s) whose processing failed were left "
                     "out — their columns hold error sentinels, not readings")
    return ModelStats(model=model, rows=rows, tracks=len(usable),
                      records=records or len(usable), cutoff=cutoff, lot=lot,
                      future_dated=future_dated, note=" · ".join(notes),
                      errored=errored)


def _empty(model: str, note: str, cutoff=None, lot=None) -> ModelStats:
    """The honest nothing: the same eight rows, all reading n=0."""
    return build_model_stats(model, [], cutoff=cutoff, lot=lot, note=note)


# ---- DB layer — everything above is pure -----------------------------------

def lot_bounds(lot: Tuple[datetime, datetime]) -> Tuple[datetime, datetime]:
    """A lot's (start, end) as a half-open range of whole DAYS.

    A lot is a run of production DAYS (`ml/lots.py` clusters by day and hands
    back midnight-normalised bounds), so the window has to be a day range, not
    a timestamp range. That distinction became load-bearing on 2026-08-30:
    `analysis_results.file_date` used to be midnight for every trim row, and
    the parser now keeps the clock time from the filename because it is the
    only record of the ORDER of same-day re-trim attempts (90dc95e). On the
    live database most rows are still midnight and new ones are not, so this
    normalises explicitly instead of relying on either shape — a lot's last
    day includes the unit trimmed at 23:37, in both worlds.
    """
    start = datetime(lot[0].year, lot[0].month, lot[0].day)
    end = datetime(lot[1].year, lot[1].month, lot[1].day) + timedelta(days=1)
    return start, end


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
                # Whole days, half-open — see `lot_bounds`: a file stamped
                # 23:37 on the lot's last day belongs to that run.
                lot_from, lot_to = lot_bounds(lot)
                q = q.filter(DBAR.file_date >= lot_from,
                             DBAR.file_date < lot_to)
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


# ===========================================================================
# Display — the ONE place a number becomes words.
# ===========================================================================
# The table, the lot verdict sentence and the Excel sheet all format through
# here, so the screen and the export cannot round a number differently. Same
# rule `ml/spc.py` follows for its verdicts.

_OHM, _KOHM = "Ω", "kΩ"
# Above this many ohms the row reads better in kilohms: 4.28 kΩ, not 4282 Ω.
_KILO = 1000.0


def _num(value: float) -> str:
    """Three-ish significant digits, never scientific notation.

    These columns span 0.0028 (sigma gradient) to 344 (electrical angle) to
    35,000 (ohms), and `%g` would print 2.96e+04 in the middle of a table a
    person is reading against a print-out.
    """
    a = abs(value)
    if a >= 100:
        return f"{value:,.0f}"
    if a >= 10:
        return f"{value:.1f}"
    if a >= 1:
        return f"{value:.2f}"
    if a >= 0.01:
        return f"{value:.3f}"
    return f"{value:.4g}"


def display_unit(unit: str, reference: Optional[float]) -> str:
    """The unit this value should be SHOWN in (the data layer keeps raw ohms)."""
    if unit == "ohms":
        return _KOHM if (reference is not None and abs(reference) >= _KILO) else _OHM
    return {"deg": "°", "%": "%"}.get(unit, "")


def bare_number(value: Optional[float], unit_text: str) -> str:
    """The number alone, scaled to the unit — for a range that names it once."""
    if value is None:
        return "—"
    return _num(value / _KILO if unit_text == _KOHM else value)


def format_in(value: Optional[float], unit_text: str) -> str:
    """One value in an already-chosen unit. None reads as an em dash, never 0."""
    if value is None:
        return "—"
    text = bare_number(value, unit_text)
    return f"{text} {unit_text}" if unit_text in (_OHM, _KOHM) else f"{text}{unit_text}"


def row_unit(row: "StatRow") -> str:
    """ONE scale for the whole row.

    A row that printed "min 422 Ω" beside "max 29.6 kΩ" makes the reader do
    unit arithmetic to compare its own three numbers, so the row picks its
    scale once — from the ALL average, falling back to the lin-passing one.
    """
    reference = row.all_.avg if row.all_.avg is not None else row.lin_passing.avg
    return display_unit(row.unit, reference)


# ---- how the table READS ----------------------------------------------
# The strings themselves, not just the numbers. They live here so the
# screen (gui/v6/widgets/stats_table.py) and the Excel sheet
# (export/evidence.py) print the SAME characters — an export that did
# its own formatting is how a sheet ends up contradicting the page it
# was taken from.

# Column layout: the metric name, then four numbers under each disposition.
DIST_HEADERS = ("n", "avg", "min", "max")
RATE_HEADERS = ("n", "count", "%")


def cell_texts(row: StatRow, cell: Cell) -> List[str]:
    """The four (or three) strings one cell shows. Pure — this is the table."""
    if row.kind == "rate":
        return [f"{cell.n:,}",
                "—" if cell.count is None else f"{cell.count:,}",
                "—" if cell.pct is None else f"{cell.pct:.1f}%"]
    unit = row_unit(row)
    return [f"{cell.n:,}", format_in(cell.avg, unit),
            format_in(cell.low, unit), format_in(cell.high, unit)]


def disclosure_text(cell: Cell) -> str:
    """What this cell had to leave out — said out loud, never implied.

    Empty when there is nothing to disclose, so a clean row stays clean. The
    three reasons are worded separately on purpose, because they send whoever
    is reading to three different places: an IMPOSSIBLE reading is a source
    file to fix, a FAILED record is a file the analyser could not read, and a
    missing one is simply a measurement never taken. Every row carries its own
    line so a reader can account for every record from the row alone, without
    holding the summary above in their head.
    """
    parts = []
    if cell.excluded:
        parts.append(f"{cell.excluded:,} impossible reading"
                     f"{'s' if cell.excluded != 1 else ''} excluded")
    if cell.errored:
        parts.append(f"{cell.errored:,} from records that failed processing")
    if cell.missing:
        parts.append(f"{cell.missing:,} not recorded")
    return " · ".join(parts)


def lot_line(row: StatRow, cell: Cell, verdict: Optional[LotVerdict]) -> str:
    """The "this lot" line under a metric: its numbers, then what they mean.

    Rate rows get one too, and deliberately: "% of the lot that didn't trim" is
    one of the three questions James named this screen to answer (app-shape
    spec §2), and it is the count, not a distribution, that answers it. They
    carry no SPC verdict — the control limits here are built on lot medians of
    a continuous metric, and there is no such series behind a pass count.
    """
    if cell.n == 0:
        return "this lot: nothing recorded"
    if row.kind == "rate":
        return (f"this lot: {cell.count:,} of {cell.n:,} "
                f"({cell.pct:.1f}%)")
    unit = row_unit(row)
    numbers = (f"this lot: {cell.n:,} readings · avg {format_in(cell.avg, unit)} "
               f"· {format_in(cell.low, unit)} to {format_in(cell.high, unit)}")
    return f"{numbers} — {verdict.text}" if verdict else numbers


def summary_line(stats: ModelStats) -> str:
    """One line over the table: what population these numbers came from."""
    if not stats.tracks:
        return stats.note or "No measurements for this model."
    # The window phrase carries its OWN preposition. Gluing a fixed "over" onto
    # each branch put "measurements over since May 14, 2026" on the screen —
    # "over" reads correctly with a span ("all history", "the selected lot") and
    # not with a start date.
    window = "over all history"
    if stats.lot is not None:
        window = "over the selected lot"
    elif stats.cutoff is not None:
        window = f"since {stats.cutoff:%b %d, %Y}"
    text = f"{stats.tracks:,} track measurements {window}"
    dropped = stats.excluded_total
    if dropped:
        text += (f" · {dropped:,} impossible reading"
                 f"{'s' if dropped != 1 else ''} left out of the numbers below "
                 "(open-circuit and zero readings — the source files need fixing)")
    if stats.note:
        text += f" · {stats.note}"
    return text


# ===========================================================================
# Lot selection — the SHARED clustering, never a second one.
# ===========================================================================

@dataclass(frozen=True)
class LotChoice:
    """One production run, as the selector offers it."""
    start: datetime
    end: datetime
    n: int                       # units in the lot
    is_open: bool                # may still be receiving units — a preview
    label: str                   # what the selector shows

    @property
    def window(self) -> Tuple[datetime, datetime]:
        return (self.start, self.end)


def _lot_label(start: datetime, end: datetime, n: int, is_open: bool) -> str:
    span = (f"{start:%b %d}" if start.date() == end.date()
            else f"{start:%b %d}–{end:%b %d}")
    if start.year != end.year or start.year != datetime.now().year:
        span += f" {end:%Y}"
    label = f"{span} · {n} unit{'s' if n != 1 else ''}"
    return label + " · current lot" if is_open else label


def model_lots(db, model: str) -> List[LotChoice]:
    """This model's production runs, newest first.

    Read straight off the SPC core's default series (`compute_spc_series`),
    which is the clustering the FOCUS list ranks from and the p-chart draws.
    Re-clustering here with its own query is exactly how two surfaces end up
    disagreeing about where a lot started, so this does not do that — it maps
    the series' points, and inherits its window (the last 30 lots), its
    requalification floor and its clock for free.

    Never raises: the selector degrades to "all history" rather than taking
    the page down.
    """
    try:
        from laser_trim_analyzer.ml.spc import compute_spc_series
        series = compute_spc_series(db, model)
    except Exception:
        logger.exception("model lots: series failed for %s", model)
        return []
    return [LotChoice(start=p.start, end=p.end, n=p.n, is_open=p.is_open,
                      label=_lot_label(p.start, p.end, p.n, p.is_open))
            for p in reversed(series.points)]


def default_lot_index(lots: Sequence[LotChoice]) -> Optional[int]:
    """The lot to select on arrival: the CURRENT one when a lot is open.

    None means "all history" — the right default when nothing is running,
    because that is the question James opens this page to answer.
    """
    return 0 if lots and lots[0].is_open else None


# ===========================================================================
# Lot vs history — is this run out of family for this model?
# ===========================================================================

@dataclass(frozen=True)
class LotVerdict:
    """What the shared SPC core says about this lot, for one metric."""
    metric: str
    label: str
    status: str                  # above | below | within | unjudged | no data
    lot_typical: Optional[float]     # this lot's MEDIAN (the SPC observation)
    lot_n: int
    normal_low: Optional[float]      # the model's own ±3σ band on lot medians
    normal_high: Optional[float]
    text: str                        # the sentence the UI and Excel print


# Metrics that cannot physically be negative — verified over the whole work
# database (2026-08-30): only `measured_electrical_angle` ever is (2,416 rows,
# min -2.56); the other five have a minimum of exactly 0. A ±3σ band on lot
# medians dips below zero whenever the lot-to-lot spread is wide next to the
# centre, and "normal -1.02–9.80 kΩ" is not a resistance anyone recognises.
# The BAND ITSELF is never clamped — the above/below/within comparison stays
# exactly the shared SPC core's numbers, the same display-only discipline the
# evidence pack applies to a small lot's binomial UCL (export/evidence.py).
# Only the sentence changes, to "normal under 9.80 kΩ".
_NON_NEGATIVE = ({key for key, _label, _unit in _DISTRIBUTION_ROWS}
                 - {"measured_electrical_angle"})


def _verdict_text(key: str, label: str, unit: str, status: str,
                  typical, low, high) -> str:
    """The sentence. Written here so the screen and the export cannot differ."""
    shown = display_unit(unit, typical if typical is not None else low)
    value = format_in(typical, shown)
    if status == "no data":
        return f"{label}: no readings in this lot"
    if status == "unjudged":
        return (f"{label}: this lot is typically {value}, but the model has "
                "not run enough lots yet to say what normal is")
    if low is not None and low <= 0 and key in _NON_NEGATIVE:
        band = f"under {format_in(high, shown)}"
    else:
        # "8.1–9.6 kΩ", not "8.1 kΩ–9.6 kΩ": the unit names the range once.
        band = f"{bare_number(low, shown)}–{format_in(high, shown)}"
    if status == "above":
        return (f"{label} for this lot sits above everything this model has "
                f"done (typically {value} vs normal {band})")
    if status == "below":
        return (f"{label} for this lot sits below anything this model has "
                f"done (typically {value} vs normal {band})")
    return (f"{label} for this lot is within its normal "
            f"(typically {value} vs normal {band})")


def compute_lot_verdicts(db, model: str, lot: Tuple[datetime, datetime]
                         ) -> "dict":
    """Per-metric plain-English verdict for one lot: metric key -> LotVerdict.

    "Normal" is not invented here — it is the shared SPC core's band
    (`ml/spc.build_continuous_series`): the ±3σ of this model's own baseline
    LOT MEDIANS, the same band the Model page's lot chart draws. This function
    only supplies the samples and asks where the selected lot falls.

    The lot's own value is its MEDIAN, matching the observation `ml/lots.py`
    aggregates and the chart plots — not the table's average. The two live
    side by side deliberately: the average is what James reads off the table,
    the median is what the control limits were built from, and quoting the
    average against a median-derived band would be comparing two things.

    The samples get the SAME two cleanings the table applies: records whose
    processing failed are dropped whole, and the resistances run through their
    plausibility policy. A lot median is already robust to a stray reading, so
    this barely moves the band — verified against `compute_spc_series` on the
    real data: 8340-1's limits are IDENTICAL, 6607's differ by 0.02 Ω on a
    9.8 kΩ limit. It is here so a lot that is MOSTLY sentinel cannot produce a
    confident sentence, which is a live risk: 8856's 75 failed records carry
    999.999 in BOTH `final_linearity_error_shifted` and `sigma_gradient`.

    That last cleaning is the one place this deliberately does NOT match
    `compute_spc_series`, which still feeds failed records to the linearity
    chart. Flagged for James rather than changed here — the drift watch's
    trained baselines were built from that population, and silently moving
    them from a stats-table commit would be worse than the divergence.

    ONE query for every metric, over the FULL history: the band needs the
    model's whole record, not the page's 30/90-day window.
    """
    try:
        from laser_trim_analyzer.ml.spc import (
            build_continuous_series, series_anchor)
        from laser_trim_analyzer.database.models import AnalysisResult as DBAR
        from laser_trim_analyzer.database.models import TrackResult as DBTR
        with db.session() as s:
            rows = [r for r in
                    (s.query(DBAR.file_date,
                             DBTR.untrimmed_resistance, DBTR.trimmed_resistance,
                             DBTR.measured_electrical_angle,
                             DBTR.final_linearity_error_shifted,
                             DBTR.margin_to_spec, DBTR.sigma_gradient,
                             DBAR.overall_status)
                     .join(DBTR, DBTR.analysis_id == DBAR.id)
                     .filter(DBAR.model == model, DBAR.file_date.isnot(None))
                     .all())
                    if not failed_processing(r[-1])]
        anchor = series_anchor(db, (r[0] for r in rows))
        floor = _requal_floor(db, model)
    except Exception:
        logger.exception("lot verdicts: query failed for %s", model)
        return {}
    start, end = lot_bounds(lot)          # whole days, half-open (see lot_bounds)
    out = {}
    for index, (key, label, unit) in enumerate(_DISTRIBUTION_ROWS, start=1):
        policy = metric_policy(key)
        raw = [(r[0], _usable(r[index])) for r in rows]
        band = _band([v for _d, v in raw], policy)
        samples = [(d, v) for d, v in raw
                   if v is not None and _plausible(v, policy, band)]
        in_lot = [v for d, v in samples if start <= d < end]
        try:
            series = build_continuous_series(model, key, samples, anchor=anchor,
                                             requal_floor=floor)
        except Exception:
            logger.exception("lot verdicts: series failed for %s/%s", model, key)
            series = None
        typical = float(median(in_lot)) if in_lot else None
        if typical is None:
            status, low, high = "no data", None, None
        elif series is None or not series.judged or not series.points:
            status, low, high = "unjudged", None, None
        else:
            low, high = series.points[-1].lcl, series.points[-1].ucl
            status = ("above" if typical > high else
                      "below" if typical < low else "within")
        out[key] = LotVerdict(
            metric=key, label=label, status=status, lot_typical=typical,
            lot_n=len(in_lot), normal_low=low, normal_high=high,
            text=_verdict_text(key, label, unit, status, typical, low, high))
    return out


def _plausible(value: float, policy: str,
               band: Optional[Tuple[float, float]]) -> bool:
    """The same admission test `_distribution_cell` applies, as a predicate."""
    if policy != POSITIVE_RATIO:
        return True
    if value <= 0:
        return False
    return band is None or band[0] <= value <= band[1]


def _requal_floor(db, model: str):
    """The model's baseline requalification date, via the SPC core's reader."""
    try:
        from laser_trim_analyzer.ml.spc import _requal_floor as spc_floor
        return spc_floor(db, model)
    except Exception:
        return None
