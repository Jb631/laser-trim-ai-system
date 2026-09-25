"""Machine comparison: the same model, the same test, the same months -- on two lasers.

A pass rate is a verdict against a TEST, so two lasers are only comparable where they ran the
SAME model against the SAME limit table in the SAME months. Graded tracks are grouped into
(limit table, month) cells; a cell counts only where at least two lasers show up in it -- the
"same test, same months" rule lives there, before anything is pooled. Each laser's tracks are
then pooled per limit table over exactly those shared-month cells: never a laser's tracks from a
month the other laser did not also run there, and never two tables pooled as if they were one
test (`test_findings_limit_tables.py`'s domain rule applies here too).

**Recent months only** (final review, 2026-09-25, I1). A finding is a recommendation read as
current, so the shared months it pools must fall inside the WINDOW_MONTHS calendar months ending
with the FLEET's newest trim file (`now`, the engine's `fleet_latest`, the anchor cut_setting
uses) -- never the model's own latest: 8081-4's last file is 2016-03, and anchored to its own
"latest year" its 2015 comparison still read as a finding. Of the five findings before this, three
described 2013-2016 (7539-2 compared 2014-12..2015-02 though all three lasers still run it). A
comparable pair outside the window is a dated FACT (`in_window: False`), never a finding, and the
title names the months it covers. Without an anchor (`now=None`) the window ends at the model's
own latest graded track.

`facts` = {"window": {first, last, months, anchored_to}, "comparisons": [...]}: every COMPARABLE
measurement, not only the ones worth a finding. Per limit table, its shared months inside the
window and those outside it are two separate comparisons, each qualifying once its lasers clear
MIN_TRACKS_PER_LASER over those months, whether or not the gap reaches MIN_GAP_POINTS -- the
population floor gates a comparable measurement, the gap (and the window) only a FINDING. A laser
under the population floor is left out of facts as well as the comparison, never guessed in.

The window is WINDOW_MONTHS whole calendar months, the anchor's own month the last of them (a
month cell is pooled whole, so a partial month at the far end would reach past the window).

Measured on a copy of the work database, 2026-09-25, anchored to the fleet's newest file
(2026-09-22, so Oct 2024 - Sep 2026): one finding -- 6126, laser 2 (DLTS) 98% of 174 against laser
1 (LTS) 82% of 130, Oct 2024 - Feb 2025 -- and five dated facts (7539-2, 8081-4 and 8232-1 from
2013-2016; 6952 pooled 2014-12..2024-07; 6126's own months before the window), where there were
five findings with no window.

This says WHERE the two lasers differ, never WHY. The gap could be a laser setting, wear, an
operator habit, or something upstream that happens to correlate with which machine a lot landed
on -- nothing here tells those apart, so no gain is claimed and the summary says so in as many
words: settings that work on one laser may not transfer.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..model import Finding
from ..stats import pct

MIN_TRACKS_PER_LASER = 100   # a laser's pooled sample below this is a taste, not a rate
MIN_GAP_POINTS = 10.0        # below this the gap is not worth disturbing a working laser
WINDOW_MONTHS = 24           # a finding pools only the shared months inside this many, to `now`


def _month(t) -> str:
    return f"{t.file_date.year:04d}-{t.file_date.month:02d}"


def _months_back(d: datetime, n: int) -> str:
    """The calendar month `n` months before `d`'s own, as "YYYY-MM"."""
    k = d.year * 12 + (d.month - 1) - n
    return f"{k // 12:04d}-{k % 12 + 1:02d}"


def _mon(month: str) -> str:
    return datetime.strptime(month, "%Y-%m").strftime("%b %Y")


def _span(months: List[str]) -> str:
    """"Oct 2025", or "Sep 2024 – Feb 2025" -- what a title says the comparison covers."""
    first, last = _mon(months[0]), _mon(months[-1])
    return first if first == last else f"{first} – {last}"


def _rate(rows) -> Optional[float]:
    return pct([bool(t.linearity_pass) for t in rows])


def analyze(model: str, tracks, laser_label,
            now: Optional[datetime] = None) -> Tuple[Dict[str, Any], List[Finding]]:
    facts: Dict[str, Any] = {}
    findings: List[Finding] = []

    graded = [t for t in tracks if t.linearity_pass is not None and t.limit_table is not None
              and t.file_date is not None]
    if not graded:
        return facts, findings
    anchor = now if now is not None else max(t.file_date for t in graded)
    window = (_months_back(anchor, WINDOW_MONTHS - 1), _months_back(anchor, 0))
    facts = {"window": {"first": window[0], "last": window[1], "months": WINDOW_MONTHS,
                        "anchored_to": anchor.date().isoformat()},
             "comparisons": []}

    # (limit table key, "YYYY-MM") -> {laser code: [tracks]}
    cells: Dict[Tuple[str, str], Dict[str, List]] = {}
    for t in graded:
        cells.setdefault((t.limit_table.key, _month(t)), {}).setdefault(t.system, []).append(t)

    # A month counts for a table only once >= 2 lasers ran it that month -- computed once,
    # before any pooling, so a table that never shares a laser never produces a "shared" month.
    # This is per TABLE across ANY two lasers, not per PAIR: with three lasers running staggered
    # months on one table, a two-laser comparison below can pool a month that only ONE of its
    # two lasers actually ran, because a third laser covered it -- the plan's (table, month)
    # algorithm, not a bug.
    shared_months: Dict[str, List[str]] = {}
    for (table_key, month), by_laser in cells.items():
        if len(by_laser) >= 2:
            shared_months.setdefault(table_key, []).append(month)

    periods = []
    for table_key, months in sorted(shared_months.items()):
        months = sorted(months)
        inside = [m for m in months if window[0] <= m <= window[1]]
        outside = [m for m in months if not window[0] <= m <= window[1]]
        periods += [(table_key, outside, False), (table_key, inside, True)]
    for table_key, months, in_window in periods:
        if not months:
            continue
        pooled: Dict[str, List] = {}
        for month in months:
            for system, rows in cells[(table_key, month)].items():
                pooled.setdefault(system, []).extend(rows)

        # A laser that also ran this table is real data, but under the floor it is a taste, not
        # a rate: left out of the comparison AND the facts rather than guessed into best/worst
        # (6952's laser 3, at 62 of a 100 floor, must never win or lose by accident).
        qualifying = {s: rows for s, rows in pooled.items() if len(rows) >= MIN_TRACKS_PER_LASER}
        if len(qualifying) < 2:
            continue
        rates = {s: _rate(rows) for s, rows in qualifying.items()}
        rates = {s: r for s, r in rates.items() if r is not None}
        if len(rates) < 2:
            continue

        # One (table_key, month) cell always groups tracks under the SAME limit table, so every
        # qualifying laser's rows here carry an identical .graded count -- any one will do.
        graded_points = next(iter(qualifying.values()))[0].limit_table.graded
        by_laser_facts = {laser_label(s): {"n": len(qualifying[s]), "pass_pct": rates[s]}
                          for s in sorted(rates)}
        facts["comparisons"].append({"table": table_key, "graded_points": graded_points,
                                     "in_window": in_window, "months": months,
                                     "by_laser": by_laser_facts})
        if not in_window:
            continue                        # a dated fact: an old comparison is never a finding

        best = max(rates, key=lambda s: rates[s])
        worst = min(rates, key=lambda s: rates[s])
        gap = rates[best] - rates[worst]
        if gap < MIN_GAP_POINTS:
            continue

        title = (f"{laser_label(best)} passes {rates[best]:.0f}%, {laser_label(worst)} "
                 f"{rates[worst]:.0f}%, same test, {_span(months)}")
        summary = (
            f"On the same {graded_points}-point limit table, over the {len(months)} "
            f"month{'s' if len(months) != 1 else ''} both ran it ({months[0]} to {months[-1]}, "
            f"inside the {WINDOW_MONTHS} months to {window[1]}), "
            f"{laser_label(best)} passed {rates[best]:.0f}% of {len(qualifying[best]):,} tracks "
            f"against {laser_label(worst)}'s {rates[worst]:.0f}% of {len(qualifying[worst]):,}. "
            "Settings that work on one laser may not transfer; this compares what each laser "
            "achieved on the same test in the same months, not why.")
        # The finding itself only ever names the best/worst PAIR: n_units and systems cover
        # exactly those two. A third laser that also qualified on this table shows up in `facts`
        # and `by_laser_facts` above, never in a finding's own n_units or systems.
        findings.append(Finding(
            model=model, analyzer="machine_compare", category="Machine comparison",
            lever="laser_settings", systems=tuple(sorted((best, worst))),
            title=title,
            summary=summary,
            n_units=len(qualifying[best]) + len(qualifying[worst]),
            strength_name="percentage points, best laser vs worst",
            strength_value=round(gap, 1),
            expected_gain_points=None,             # where they differ, never why -- no gain claimed
            evidence={"table": table_key, "months": [months[0], months[-1]],
                     "window": [window[0], window[1]],
                     "by_laser": by_laser_facts, "best_laser": best, "worst_laser": worst,
                     "graded_points": graded_points}))
    return facts, findings
