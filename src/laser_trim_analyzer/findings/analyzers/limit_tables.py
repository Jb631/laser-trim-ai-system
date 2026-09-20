"""Limit tables: is this model being graded against ONE test, or several?

A laser grades a track against a LIMIT TABLE -- one (upper, lower) pair per position. The same
model, on the same laser and the same track, should be held to one table. When it is not, either
two tables are in service AT ONCE (a setup inconsistency: two station programs, two operators'
files) or the table CHANGED (somebody edited the limits). Either way every yield compared across
the two is a comparison between two different TESTS, and nobody can see that from a pass rate.

Found on the work database 2026-09-20: 17 (model, track, laser) groups graded against two or more
tables inside their latest year -- 8232-1 on laser 1 at 89 graded points and at 45 with the same
band; 8506A/B with every band loosened from +/-0.01 to +/-0.0375 V in July 2025 (pass 83% -> 100%).

Reports what is in service and how the tables differ. NEVER claims a gain: a laxer table passing
more units is a different test, not more yield. The lever is the laser limit table (same day).
Needs only the stored final limits, so it works on a database ingested before the pass capture.
"""
from collections import defaultdict
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..data import LimitTable
from ..model import Finding
from ..stats import pct

MIN_TABLE_N = 30          # tracks before a table counts as "in service" rather than a one-off file
MIN_SHARE = 0.10          # ...and its share of the window
WINDOW_DAYS = 365         # "now" = the last year of THIS track's data
MIN_OVERLAP_DAYS = 30     # date ranges overlapping this long = in service AT ONCE; less = a change
MIN_COMPARED = 5          # positions both tables cover, before their bands can be compared
BAND_TOL = 1e-5           # volts; limits are stored to about 1e-5
BAND_REL_TOL = 0.01       # ...or 1% of the band: interpolating across a bowtie corner is not a change
MORE_PASSING = 5.0        # points by which the STRICTER table must pass more before we say so


def _half_width_at(band: Tuple[Tuple[float, float], ...], p: float) -> Optional[float]:
    """Half-width of `band` at position p, linearly interpolated; None outside the travel it covers."""
    if not band or p < band[0][0] - 1e-9 or p > band[-1][0] + 1e-9:
        return None
    below = max((x for x in band if x[0] <= p + 1e-9), key=lambda x: x[0])
    above = min((x for x in band if x[0] >= p - 1e-9), key=lambda x: x[0])
    if above[0] == below[0]:
        return below[1]
    return below[1] + (above[1] - below[1]) * (p - below[0]) / (above[0] - below[0])


def compare(old: LimitTable, new: LimitTable) -> Dict[str, Any]:
    """How `new` differs from `old`, over the travel BOTH cover.

    Two tables rarely share a grid (a 0.5-degree table against a 1-degree one; two 171-row tables a
    fraction of a degree apart), so `old` is interpolated at `new`'s positions. Matching position
    for position left 8 of the 17 real cases "not comparable"; interpolating left none.
    """
    out: Dict[str, Any] = {"graded_old": old.graded, "graded_new": new.graded}
    pairs = [(h, _half_width_at(old.band, p)) for p, h in new.band]
    pairs = [(h, o) for h, o in pairs if o is not None]
    out["compared_positions"] = len(pairs)
    if len(pairs) < MIN_COMPARED:
        out["kind"] = "not_comparable"
        return out
    tol = [max(BAND_TOL, BAND_REL_TOL * o) for _, o in pairs]
    out["wider"] = sum(1 for (h, o), t in zip(pairs, tol) if h - o > t)
    out["narrower"] = sum(1 for (h, o), t in zip(pairs, tol) if o - h > t)
    out["max_abs_diff"] = round(max(abs(h - o) for h, o in pairs), 5)
    if out["wider"] == 0 and out["narrower"] == 0:
        out["kind"] = "same_band_same_points" if old.graded == new.graded else "same_band_other_density"
    else:
        out["kind"] = "different_band"
    return out


def _difference(c: Dict[str, Any], p_old: Optional[float], p_new: Optional[float]) -> str:
    if c["kind"] == "same_band_other_density":
        text = (f"Both carry the same band wherever their travel overlaps ({c['compared_positions']} positions "
                f"compared), but one grades {c['graded_old']} points and the other {c['graded_new']}. Linearity is "
                f"zero-tolerance per point, so the denser table is the stricter test of the two.")
        if p_old is not None and p_new is not None:
            dense, sparse = (p_new, p_old) if c["graded_new"] > c["graded_old"] else (p_old, p_new)
            if dense > sparse + MORE_PASSING:
                text += (" Here the stricter table nevertheless passes MORE units, so something else differs "
                         "between the two groups as well (period, recipe or product variant).")
        return text
    if c["kind"] == "different_band":
        parts = ([f"WIDER at {c['wider']}"] if c["wider"] else []) + \
                ([f"NARROWER at {c['narrower']}"] if c["narrower"] else [])
        return (f"Of the {c['compared_positions']} positions compared, the later table is " + " and ".join(parts)
                + f" (by up to {c['max_abs_diff']:g} V of half-width). These are different REQUIREMENTS, not "
                  f"different sampling.")
    if c["kind"] == "same_band_same_points":
        return ("Both carry the same band and grade the same number of points; they differ only in which rows "
                "are left ungraded.")
    return (f"Their travel overlaps at only {c['compared_positions']} positions, so the bands cannot be compared "
            f"({c['graded_old']} against {c['graded_new']} graded points).")


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
        first_of = lambda v: min(t.file_date for t in v)          # noqa: E731
        last_of = lambda v: max(t.file_date for t in v)           # noqa: E731
        for v in sorted(by_table.values(), key=first_of):
            if len(v) >= MIN_TABLE_N:
                tab = v[0].limit_table
                history.append({"system": system, "track": name, "rows": tab.rows, "graded": tab.graded,
                                "n": len(v), "first": first_of(v).date().isoformat(),
                                "last": last_of(v).date().isoformat(),
                                "trim_pass_pct": pct([t.linearity_pass for t in v if t.linearity_pass is not None])})
        cutoff = last_of(ts) - timedelta(days=WINDOW_DAYS)
        recent = [t for t in ts if t.file_date >= cutoff]
        live = defaultdict(list)
        for t in recent:
            live[t.limit_table.key].append(t)
        live = sorted((v for v in live.values() if len(v) >= MIN_TABLE_N and len(v) / len(recent) >= MIN_SHARE),
                      key=len, reverse=True)
        if len(live) < 2:
            continue                                              # one table in service: say nothing
        older, newer = sorted(live[:2], key=first_of)             # the two busiest, in TIME order
        overlap = (min(last_of(older), last_of(newer)) - max(first_of(older), first_of(newer))).days
        c = compare(older[0].limit_table, newer[0].limit_table)
        p_old = pct([t.linearity_pass for t in older if t.linearity_pass is not None])
        p_new = pct([t.linearity_pass for t in newer if t.linearity_pass is not None])

        def one(v, p):
            tab = v[0].limit_table
            return (f"{tab.graded} graded points ({tab.rows} rows), {len(v):,} tracks, {first_of(v).date()} to "
                    f"{last_of(v).date()}" + (f", {p:.0f}% left the laser inside limits" if p is not None else ""))

        if overlap >= MIN_OVERLAP_DAYS:
            title = f"{laser_label(system)}: {len(live)} limit tables in service at once"
            lead = (f"Over the last 12 months {laser_label(system)} graded {name} against {len(live)} different "
                    f"limit tables at the same time (the two busiest overlap by {overlap} days). ")
            labels = ("The earlier one", "The later one")
        else:
            title = f"{laser_label(system)}: the limit table changed"
            lead = (f"{laser_label(system)} changed the limit table it grades {name} against around "
                    f"{first_of(newer).date()}. ")
            labels = ("Before", "After")
        findings.append(Finding(
            model=model, analyzer="limit_tables", category="Limit table", lever="laser_limit_table",
            systems=(system,), title=title,
            summary=(lead + f"{labels[0]}: {one(older, p_old)}. {labels[1]}: {one(newer, p_new)}. "
                     + _difference(c, p_old, p_new)
                     + " Pass rates measured against different tables are not comparable."),
            n_units=len(recent),
            strength_name="share of the last 12 months graded against the second-busiest table",
            strength_value=round(len(live[1]) / len(recent), 3),
            expected_gain_points=None,                  # a different test is not more yield
            evidence={"track": name, "concurrent": overlap >= MIN_OVERLAP_DAYS, "overlap_days": overlap,
                      "tables_live": len(live), "comparison": c,
                      "older": {"rows": older[0].limit_table.rows, "graded": older[0].limit_table.graded,
                                "n": len(older), "trim_pass_pct": p_old,
                                "first": first_of(older).date().isoformat(), "last": last_of(older).date().isoformat()},
                      "newer": {"rows": newer[0].limit_table.rows, "graded": newer[0].limit_table.graded,
                                "n": len(newer), "trim_pass_pct": p_new,
                                "first": first_of(newer).date().isoformat(), "last": last_of(newer).date().isoformat()}}))
    return history, findings
