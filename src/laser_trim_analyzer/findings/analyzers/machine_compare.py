"""Machine comparison: the same model, the same test, the same months -- on two lasers.

A pass rate is a verdict against a TEST, so two lasers are only comparable where they ran the
SAME model against the SAME limit table in the SAME months. Graded tracks are grouped into
(limit table, month) cells; a cell counts only where at least two lasers show up in it -- the
"same test, same months" rule lives there, before anything is pooled. Each laser's tracks are
then pooled per limit table over exactly those shared-month cells: never a laser's tracks from a
month the other laser did not also run there, and never two tables pooled as if they were one
test (`test_findings_limit_tables.py`'s domain rule applies here too).

Measured on the rebuild, read-only, 2026-09-24: 6126 -- same limit table, same months -- laser 2
(DLTS) 99% of 488 against laser 1 (LTS) 76% of 610; 6952 laser 2 96% of 156 against laser 1 84%
of 160, with laser 3 also on the same table and months at 85% of 62 -- too few to be crowned
best or worst, so it is left out of the comparison rather than guessed in. On the real data this
analyzer produces exactly two findings.

This says WHERE the two lasers differ, never WHY. The gap could be a laser setting, wear, an
operator habit, or something upstream that happens to correlate with which machine a lot landed
on -- nothing here tells those apart, so no gain is claimed and the summary says so in as many
words: settings that work on one laser may not transfer.
"""
from typing import Any, Dict, List, Optional, Tuple

from ..model import Finding
from ..stats import pct

MIN_TRACKS_PER_LASER = 100   # a laser's pooled sample below this is a taste, not a rate
MIN_GAP_POINTS = 10.0        # below this the gap is not worth disturbing a working laser


def _month(t) -> str:
    return f"{t.file_date.year:04d}-{t.file_date.month:02d}"


def _rate(rows) -> Optional[float]:
    return pct([bool(t.linearity_pass) for t in rows])


def analyze(model: str, tracks, laser_label) -> Tuple[Dict[str, Any], List[Finding]]:
    facts: Dict[str, Any] = {}
    findings: List[Finding] = []

    graded = [t for t in tracks if t.linearity_pass is not None and t.limit_table is not None
              and t.file_date is not None]
    if not graded:
        return facts, findings

    # (limit table key, "YYYY-MM") -> {laser code: [tracks]}
    cells: Dict[Tuple[str, str], Dict[str, List]] = {}
    for t in graded:
        cells.setdefault((t.limit_table.key, _month(t)), {}).setdefault(t.system, []).append(t)

    # A month counts for a table only once >= 2 lasers ran it that month -- computed once,
    # before any pooling, so a table that never shares a laser never produces a "shared" month.
    shared_months: Dict[str, List[str]] = {}
    for (table_key, month), by_laser in cells.items():
        if len(by_laser) >= 2:
            shared_months.setdefault(table_key, []).append(month)

    for table_key, months in sorted(shared_months.items()):
        months = sorted(months)
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
        facts[table_key] = {"months": months, "by_laser": by_laser_facts}

        best = max(rates, key=lambda s: rates[s])
        worst = min(rates, key=lambda s: rates[s])
        gap = rates[best] - rates[worst]
        if gap < MIN_GAP_POINTS:
            continue

        title = (f"{laser_label(best)} passes {rates[best]:.0f}%, {laser_label(worst)} "
                 f"{rates[worst]:.0f}%, same test and months")
        summary = (
            f"On the same {graded_points}-point limit table, over the {len(months)} "
            f"month{'s' if len(months) != 1 else ''} both ran it ({months[0]} to {months[-1]}), "
            f"{laser_label(best)} passed {rates[best]:.0f}% of {len(qualifying[best]):,} tracks "
            f"against {laser_label(worst)}'s {rates[worst]:.0f}% of {len(qualifying[worst]):,}. "
            "Settings that work on one laser may not transfer; this compares what each laser "
            "achieved on the same test in the same months, not why.")
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
                     "by_laser": by_laser_facts, "best_laser": best, "worst_laser": worst,
                     "graded_points": graded_points}))
    return facts, findings
