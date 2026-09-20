"""Limit tables: is this model being graded against ONE test, or several?

A laser grades a track against a LIMIT TABLE -- one (upper, lower) pair per position. The same
model, on the same laser and the same track, should be held to one table. When it is not, either
two tables were in service AT ONCE (a setup inconsistency: two station programs, two operators'
files) or the table CHANGED (somebody edited the limits). Either way every yield compared across
the two is a comparison between two different TESTS, and nobody can see that from a pass rate.

Found on the work database 2026-09-20: 17 (model, track, laser) groups graded against two or more
tables inside their latest year -- 8232-1 on laser 1 at 89 graded points and at 45 with the same
band; 8506A/B with every band loosened from +/-0.01 to +/-0.0375 V in July 2025 (pass 83% -> 100%).

Reports what was in service and how the tables differ. NEVER claims a gain: a laxer table passing
more units is a different test, not more yield. The lever is the laser limit table (same day).
Needs only the stored final limits, so it works on a database ingested before the pass capture.

Three things an independent review broke before they shipped, and how each is now built:
  * ONE straggler file on a retired table used to turn a clean change into "two tables at once",
    because overlap was measured between the very first and very last file. Spans are now the
    central 90% of each table's dates.
  * The same bowtie sampled at 1.0 and at 0.5 degrees read "same band" one way round and
    "different REQUIREMENTS" the other, because a coarse grid was interpolated across the band's
    corner. Bands are now compared by a BRACKET test (a value sampled between two neighbours of the
    other table must lie inside their range -- exact for straight AND staircase bands), and a
    difference counts only when BOTH directions report it.
  * "The denser table is the stricter test" is false when the two tables do not cover the same
    travel. The travel is compared, and said.
"""
from collections import Counter, defaultdict
from datetime import timedelta
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from ..data import LimitTable
from ..model import Finding
from ..stats import pct

MIN_TABLE_N = 30          # tracks before a table counts as "in service" rather than a one-off file
MIN_SHARE = 0.10          # ...and its share of the window
WINDOW_DAYS = 365         # the window = the last year of THIS track's own data (the model may be dormant)
SPAN_TRIM = 0.05          # a table's span = the central 90% of its dates: a straggler file is not a period
MIN_OVERLAP_DAYS = 30     # central spans overlapping this long = in service AT ONCE; less = a change
MIN_COMPARED = 5          # positions each table has inside the other's travel, before bands can be compared
BAND_TOL = 2e-5           # volts; limits are stored rounded to 1e-5
MORE_PASSING = 5.0        # points by which the STRICTER table must pass more before we say so
_EPS = 1e-9


def _span(tracks) -> Tuple[Any, Any]:
    """The central 90% of a table's dates -- robust to a rework file on a retired program."""
    dates = sorted(t.file_date for t in tracks)
    last = len(dates) - 1
    return dates[int(SPAN_TRIM * last)], dates[int(round((1.0 - SPAN_TRIM) * last))]


def _representative(tracks) -> LimitTable:
    """One table, many files: take the band (the positions) MOST of them carry, not the first file's."""
    band = Counter(t.limit_table.band for t in tracks).most_common(1)[0][0]
    return next(t.limit_table for t in tracks if t.limit_table.band == band)


def _neighbours(band, p: float) -> Optional[Tuple[float, float]]:
    """Half-widths of `band` at the grid points bracketing position p; None outside its travel."""
    if not band or p < band[0][0] - _EPS or p > band[-1][0] + _EPS:
        return None
    below = max((x for x in band if x[0] <= p + _EPS), key=lambda x: x[0])
    above = min((x for x in band if x[0] >= p - _EPS), key=lambda x: x[0])
    return below[1], above[1]


def _one_way(ref: LimitTable, probe: LimitTable) -> Dict[str, Any]:
    """Every probe point inside ref's travel: could ref's band take that half-width there?"""
    out = {"compared": 0, "wider": 0, "narrower": 0, "worst": 0.0}
    for p, h in probe.band:
        nb = _neighbours(ref.band, p)
        if nb is None:
            continue
        out["compared"] += 1
        if h > max(nb) + BAND_TOL:
            out["wider"] += 1
            out["worst"] = max(out["worst"], h - max(nb))
        elif h < min(nb) - BAND_TOL:
            out["narrower"] += 1
            out["worst"] = max(out["worst"], min(nb) - h)
    return out


def _travel(band) -> Optional[Tuple[float, float, float]]:
    if len(band) < 2:
        return None
    xs = [x for x, _ in band]
    return xs[0], xs[-1], median(b - a for a, b in zip(xs, xs[1:]))


def compare(old: LimitTable, new: LimitTable) -> Dict[str, Any]:
    """How `new` differs from `old`. Symmetric by construction: `new` is WIDER only when new's points
    sit above anything old could be there AND old's points sit below anything new could be there."""
    out: Dict[str, Any] = {"graded_old": old.graded, "graded_new": new.graded}
    fwd, bwd = _one_way(old, new), _one_way(new, old)
    out["compared_positions"] = fwd["compared"]
    to, tn = _travel(old.band), _travel(new.band)
    if to and tn:
        slack = 1.5 * max(to[2], tn[2])                  # half a grid step either way is the same travel
        out["travel_old"], out["travel_new"] = [to[0], to[1]], [tn[0], tn[1]]
        out["travel_differs"] = abs(to[0] - tn[0]) > slack or abs(to[1] - tn[1]) > slack
    else:
        out["travel_differs"] = False
    if min(fwd["compared"], bwd["compared"]) < MIN_COMPARED:
        out["kind"] = "not_comparable"
        return out
    out["wider"] = fwd["wider"] if bwd["narrower"] else 0
    out["narrower"] = fwd["narrower"] if bwd["wider"] else 0
    if out["wider"] or out["narrower"]:
        out["kind"] = "different_band"
        out["max_abs_diff"] = round(max(fwd["worst"], bwd["worst"]), 5)
    else:
        out["kind"] = "same_band_same_points" if old.graded == new.graded else "same_band_other_density"
    return out


def _difference(c: Dict[str, Any], p_old: Optional[float], p_new: Optional[float]) -> str:
    travel = ""
    if c.get("travel_differs"):
        travel = (f" They do not grade the same travel either: {c['travel_old'][0]:g} to {c['travel_old'][1]:g} "
                  f"against {c['travel_new'][0]:g} to {c['travel_new'][1]:g}.")
    if c["kind"] == "same_band_other_density":
        text = (f"Both carry the same band wherever their travel overlaps ({c['compared_positions']} positions "
                f"compared), but one grades {c['graded_old']} points and the other {c['graded_new']}.")
        if c.get("travel_differs"):
            return text + travel + (" So neither is simply the stricter test: the one that reaches further grades "
                                    "part of the track the other never looks at.")
        text += " Linearity is zero-tolerance per point, so the denser table is the stricter test of the two."
        if p_old is not None and p_new is not None:
            dense, sparse = (p_new, p_old) if c["graded_new"] > c["graded_old"] else (p_old, p_new)
            if dense > sparse + MORE_PASSING:
                text += (" Here the stricter table nevertheless passes MORE units, so something else differs "
                         "between the two groups as well (period, recipe or product variant).")
        return text
    if c["kind"] == "different_band":
        parts = ([f"WIDER at {c['wider']}"] if c["wider"] else []) + \
                ([f"NARROWER at {c['narrower']}"] if c["narrower"] else [])
        return (f"Of the {c['compared_positions']} positions the later table grades inside the earlier one's "
                f"travel, it is " + " and ".join(parts) + f" (by up to {c['max_abs_diff']:g} V of half-width). "
                f"These are different REQUIREMENTS, not different sampling." + travel)
    if c["kind"] == "same_band_same_points":
        return (f"The half-width of the band matches at the {c['compared_positions']} positions compared and both "
                f"grade {c['graded_new']} points, so the difference is elsewhere in the table: which rows are "
                f"left ungraded, or where the band is centred." + travel)
    return (f"Their travel overlaps at too few positions to compare the bands ({c['graded_old']} against "
            f"{c['graded_new']} graded points)." + travel)


def analyze(model: str, tracks, laser_label) -> Tuple[List[Dict[str, Any]], List[Finding]]:
    """(limit-table history for the facts strip, findings). Every table that was ever in real service
    is HISTORY; a FINDING needs two tables live inside the last year of that track's data."""
    history: List[Dict[str, Any]] = []
    findings: List[Finding] = []
    groups = defaultdict(list)
    for t in tracks:
        if t.limit_table is not None:
            groups[(t.system, t.track_name)].append(t)
    for (system, name), ts in sorted(groups.items()):
        by_table = defaultdict(list)
        for t in ts:
            by_table[t.limit_table.key].append(t)
        for v in sorted(by_table.values(), key=lambda v: _span(v)[0]):
            if len(v) >= MIN_TABLE_N:
                tab = v[0].limit_table
                history.append({"system": system, "track": name, "rows": tab.rows, "graded": tab.graded,
                                "n": len(v), "first": min(t.file_date for t in v).date().isoformat(),
                                "last": max(t.file_date for t in v).date().isoformat(),
                                "trim_pass_pct": pct([t.linearity_pass for t in v if t.linearity_pass is not None])})
        latest = max(t.file_date for t in ts)
        recent = [t for t in ts if t.file_date >= latest - timedelta(days=WINDOW_DAYS)]
        live = defaultdict(list)
        for t in recent:
            live[t.limit_table.key].append(t)
        live = sorted((v for v in live.values() if len(v) >= MIN_TABLE_N and len(v) / len(recent) >= MIN_SHARE),
                      key=len, reverse=True)
        if len(live) < 2:
            continue                                              # one table in service: say nothing
        older, newer = sorted(live[:2], key=lambda v: _span(v)[0])    # the two busiest, in TIME order
        (o_from, o_to), (n_from, n_to) = _span(older), _span(newer)
        overlap = (min(o_to, n_to) - max(o_from, n_from)).days
        rep_old, rep_new = _representative(older), _representative(newer)
        c = compare(rep_old, rep_new)
        p_old = pct([t.linearity_pass for t in older if t.linearity_pass is not None])
        p_new = pct([t.linearity_pass for t in newer if t.linearity_pass is not None])

        def one(tab, v, span, p):
            return (f"{tab.graded} graded points ({tab.rows} rows), {len(v):,} tracks, the central 90% of them "
                    f"between {span[0].date()} and {span[1].date()}"
                    + (f", {p:.0f}% left the laser inside limits" if p is not None else ""))

        period = f"In the 12 months to {latest.date()}"
        if overlap >= MIN_OVERLAP_DAYS:
            title = f"{laser_label(system)}: {len(live)} limit tables were in service at once"
            lead = (f"{period} {laser_label(system)} graded {name} against {len(live)} different limit tables "
                    f"at the same time (the two busiest overlap by {overlap} days). ")
            labels = ("The earlier one", "The later one")
        else:
            title = f"{laser_label(system)}: the limit table changed"
            lead = (f"{period} {laser_label(system)} changed the limit table it grades {name} against, around "
                    f"{n_from.date()}. ")
            labels = ("Before", "After")
        if len(live) > 2:
            lead += f"The two busiest of the {len(live)} are compared here. "
        findings.append(Finding(
            model=model, analyzer="limit_tables", category="Limit table", lever="laser_limit_table",
            systems=(system,), title=title,
            summary=(lead + f"{labels[0]}: {one(rep_old, older, (o_from, o_to), p_old)}. "
                     f"{labels[1]}: {one(rep_new, newer, (n_from, n_to), p_new)}. "
                     + _difference(c, p_old, p_new)
                     + " Pass rates measured against different tables are not comparable."),
            n_units=len(recent),
            strength_name="share of that year graded against the second-busiest table",
            strength_value=round(len(live[1]) / len(recent), 3),
            expected_gain_points=None,                  # a different test is not more yield
            evidence={"track": name, "concurrent": overlap >= MIN_OVERLAP_DAYS, "overlap_days": overlap,
                      "tables_live": len(live), "window_to": latest.date().isoformat(), "comparison": c,
                      "older": {"rows": rep_old.rows, "graded": rep_old.graded, "n": len(older),
                                "trim_pass_pct": p_old, "from": o_from.date().isoformat(),
                                "to": o_to.date().isoformat()},
                      "newer": {"rows": rep_new.rows, "graded": rep_new.graded, "n": len(newer),
                                "trim_pass_pct": p_new, "from": n_from.date().isoformat(),
                                "to": n_to.date().isoformat()}}))
    return history, findings
