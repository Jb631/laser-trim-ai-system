"""Cut setting: the laser was told to cut this much. Was a different amount better?

The biggest single effect measured in this data so far. Holding model, machine,
limit table, incoming resistance and period constant (2026-09-20, home slice):

    8232-1 on laser 2 (DLTS), one 45-point table, 2018-20
        cut 0.55 -> 97% inside limits at the end of the laser
        cut 0.75 -> 78%
        cut 0.88 -> 38%          monotone: shorter was better
    8340-1 on laser 1 (LTS), one 53-point table, 2023-09 onward
        cut 3000 -> 19%   3500 -> 41%   4000 -> 23%   4341 -> 5%
                                         an OPTIMUM, not a slope

So the shop has already run the experiment, more than once, and the answer is
model-specific. This analyzer reads it back out.

Three things it must never do, each of which produced a wrong answer by hand
before the controls went in:

1. **Compare across lasers.** `laser_cut_length` is 0.55-0.88 on laser 2 and
   2500-4341 on laser 1. They are not the same quantity and pooling them is
   meaningless, so the group key carries the system.
2. **Compare across limit tables.** A pass rate is a verdict against a test.
   8340-1's best setting is spread over four tables while one of its rivals sits
   on a single table; comparing those two compares two tests. The group key
   carries the table.
3. **Credit the cut with the ink's doing.** The 0.88 period above also ran
   ~200 ohm lower on incoming resistance, which is a reason to cut longer --
   the causation could run either way. So a winner must win in BOTH halves of
   incoming resistance, or nothing is reported.

And one thing it must always SAY: settings are run in blocks. Only 8% of
8340-1's production days and 2% of 8232-1's used more than one setting, so this
is a between-period comparison and a material-era effect cannot be excluded from
the data alone. That disclosure travels with every finding, because the honest
recommendation is "test this at the machine", not "change this".
"""
from datetime import datetime, timedelta
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from ..model import Finding
from ..stats import pct

LOOKBACK_DAYS = 3 * 365     # long enough to hold several settings, recent enough to act on
MIN_PER_SETTING = 30        # a setting seen less than this is a trial, not a run
MIN_TOTAL = 150
MIN_GAIN_POINTS = 8.0       # below this the difference is not worth disturbing a working laser
MIN_PER_HALF = 15           # per incoming-resistance half, for the both-halves control
CURRENT_DAYS = 120          # what the laser is set to now
# Measured 2026-09-23 (rebuilt database): 80 live settings, median run 298 days, 13 under
# 60 -- including 8397-2's 23-day "best", which is why a trial cannot be crowned winner.
MIN_RUN_DAYS = 60
# 8397-2's newest file is Apr 2025 against the fleet's newest of Sep 2026 -- nowhere near
# "now". Past this many days behind the fleet, a group is reported as history, not a rate.
STALE_DAYS = 180


def _setting(t) -> Optional[float]:
    """The cut the laser was told to make on the FIRST pass.

    The first pass is the one a recommendation could change: later passes are a
    response to where the first one landed.
    """
    if not t.passes:
        return None
    v = t.passes[0].cut_setting
    return None if v is None else round(float(v), 4)


def _group_key(t) -> Optional[Tuple]:
    table = t.limit_table
    return None if table is None else (t.system, t.track_name, table.key)


def _rate(rows) -> Optional[float]:
    return pct([bool(t.linearity_pass) for t in rows if t.linearity_pass is not None])


def _overlap_share(rows) -> Optional[float]:
    """Share of production days that ran more than one cut setting.

    This is the whole strength-of-evidence question in one number. Near 0 means
    the settings were run in blocks and any difference between them is also a
    difference between periods.
    """
    days: Dict[Any, set] = {}
    for t in rows:
        s = _setting(t)
        if s is not None:
            days.setdefault(t.file_date.date(), set()).add(s)
    if not days:
        return None
    return 100.0 * sum(1 for v in days.values() if len(v) > 1) / len(days)


def _concurrency(rows, a: float, b: float) -> Tuple[int, int]:
    """(months both settings were in use, months either was) for two settings.

    Day-mixing alone understates the evidence. 8232-1's 4000 and 4100 shared
    only 8% of production DAYS but ran alongside each other for eight straight
    MONTHS -- which is a far better comparison than a before/after, and calling
    it "two periods" would have undersold it.
    """
    months: Dict[Any, set] = {}
    for t in rows:
        s = _setting(t)
        if s in (a, b):
            months.setdefault((t.file_date.year, t.file_date.month), set()).add(s)
    both = sum(1 for v in months.values() if len(v) > 1)
    return both, len(months)


def _changeover(rows, before: float, after: float) -> Optional[str]:
    """The month `after` took over from `before`, when it is a clean switch.

    A before/after is the weakest evidence this analyzer reports, but it is also
    the most ACTIONABLE shape: 8232-1's laser 1 ran 4100 from October 2025,
    moved to 4000 in July 2026, and has been 9 points worse since. Naming the
    month turns "try the other setting" into "look at what changed in July".
    """
    months: Dict[Any, set] = {}
    for t in rows:
        s = _setting(t)
        if s in (before, after):
            months.setdefault((t.file_date.year, t.file_date.month), set()).add(s)
    if not months:
        return None
    ordered = sorted(months)
    switch = None
    for ym in ordered:
        if months[ym] == {after}:
            switch = switch or ym
        elif before in months[ym]:
            switch = None                 # `before` came back: not a clean switch
    if switch is None:
        return None
    tail = [ym for ym in ordered if ym >= switch]
    if any(before in months[ym] for ym in tail):
        return None
    return f"{switch[0]}-{switch[1]:02d}"


def _halves(rows) -> Optional[Tuple[float, List, List]]:
    """Split on the median incoming resistance of the whole group."""
    rs = sorted(t.untrimmed_resistance for t in rows if t.untrimmed_resistance is not None)
    if len(rs) < MIN_PER_HALF * 2:
        return None
    mid = median(rs)
    lo = [t for t in rows if t.untrimmed_resistance is not None and t.untrimmed_resistance < mid]
    hi = [t for t in rows if t.untrimmed_resistance is not None and t.untrimmed_resistance >= mid]
    return mid, lo, hi


def _wins_both_halves(rows, best: float, current: float) -> Optional[Dict[str, Any]]:
    """The control that stops the ink being credited to the cut.

    Returns the per-half rates when `best` beats `current` in BOTH halves of
    incoming resistance, and None when it does not -- including when either half
    is too thin to say, because "could not check" is not "passed the check".
    """
    split = _halves(rows)
    if split is None:
        return None
    mid, lo, hi = split
    out: Dict[str, Any] = {"median_resistance": round(mid, 1)}
    for name, half in (("below", lo), ("above", hi)):
        b = [t for t in half if _setting(t) == best]
        c = [t for t in half if _setting(t) == current]
        if len(b) < MIN_PER_HALF or len(c) < MIN_PER_HALF:
            return None
        rb, rc = _rate(b), _rate(c)
        if rb is None or rc is None or rb <= rc:
            return None
        out[name] = {"best_n": len(b), "best_pct": round(rb, 1),
                     "current_n": len(c), "current_pct": round(rc, 1)}
    return out


def _span(rows) -> str:
    ds = sorted(t.file_date.date().isoformat() for t in rows)
    return f"{ds[0]} .. {ds[-1]}"


def _span_days(rows) -> int:
    return (max(t.file_date for t in rows) - min(t.file_date for t in rows)).days


def _describe_group(rows, by_setting: Dict[float, List]) -> Dict[str, Any]:
    return {"n": len(rows), "window": _span(rows),
            "days_with_more_than_one_setting_pct": _overlap_share(rows),
            "settings": [{"setting": s, "n": len(g), "pass_pct": _rate(g),
                          "window": _span(g),
                          "median_incoming_resistance": (
                              round(median([t.untrimmed_resistance for t in g
                                            if t.untrimmed_resistance is not None]), 1)
                              if any(t.untrimmed_resistance is not None for t in g) else None)}
                         for s, g in sorted(by_setting.items())]}


def analyze(model: str, tracks, laser_label,
           now: Optional[datetime] = None) -> Tuple[Dict[str, Any], List[Finding]]:
    """Facts for every group with more than one cut setting; a finding only where it is safe.

    `now` is the fleet's newest trim file, not this model's -- a model that stopped running
    months ago must not be reported as "now running" just because it once was.
    """
    facts: Dict[str, Any] = {}
    findings: List[Finding] = []
    dated = [t for t in tracks if t.file_date is not None and t.passes]
    if not dated:
        return facts, findings
    latest = max(t.file_date for t in dated)
    recent = [t for t in dated if t.file_date >= latest - timedelta(days=LOOKBACK_DAYS)]

    groups: Dict[Tuple, List] = {}
    for t in recent:
        key = _group_key(t)
        if key is not None and _setting(t) is not None and t.linearity_pass is not None:
            groups.setdefault(key, []).append(t)

    for key, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        system, track_name, _table = key
        by_setting: Dict[float, List] = {}
        for t in rows:
            by_setting.setdefault(_setting(t), []).append(t)
        live = {s: g for s, g in by_setting.items() if len(g) >= MIN_PER_SETTING}
        if len(live) < 2 or len(rows) < MIN_TOTAL:
            continue
        label = f"{laser_label(system)} · {track_name}"
        facts[label] = _describe_group(rows, live)

        # What the laser is set to NOW -- the only setting a recommendation can replace.
        newest = max(t.file_date for t in rows)
        current_rows = [t for t in rows if t.file_date >= newest - timedelta(days=CURRENT_DAYS)]
        tally: Dict[float, int] = {}
        for t in current_rows:
            s = _setting(t)
            if s in live:
                tally[s] = tally.get(s, 0) + 1
        if not tally:
            continue
        current = max(tally, key=lambda s: (tally[s], s))
        rates = {s: _rate(g) for s, g in live.items()}
        rates = {s: r for s, r in rates.items() if r is not None}
        if current not in rates or len(rates) < 2:
            continue
        # A trial is not a run: only a setting with MIN_RUN_DAYS of production behind it can be
        # the recommendation. The current setting is always eligible -- it is what we compare to.
        eligible = {s: r for s, r in rates.items()
                    if s == current or _span_days(live[s]) >= MIN_RUN_DAYS}
        if len(eligible) < 2:
            continue
        best = max(eligible, key=lambda s: (eligible[s], -abs(s - current)))
        facts[label]["current_setting"] = current
        if best == current:
            continue                                  # already on the best of the settings tried
        gain = eligible[best] - eligible[current]
        if gain < MIN_GAIN_POINTS:
            continue
        halves = _wins_both_halves(rows, best, current)
        if halves is None:
            continue                                  # the ink is not excluded, so nothing is claimed

        overlap = facts[label]["days_with_more_than_one_setting_pct"]
        both_months, all_months = _concurrency(rows, best, current)
        # Three grades of evidence, weakest last. Mixed on the same day is as close to
        # randomised as this shop gets; running side by side over months still rules out a
        # simple before/after; separate blocks do not, and must say so. Computed once, then
        # the strength text branches on it -- a single source of truth for both.
        if overlap is not None and overlap >= 25.0:
            grade = "same_days"
        elif all_months and both_months * 2 >= all_months:
            grade = "side_by_side"
        else:
            grade = "two_periods"
        if grade == "same_days":
            strength = (f"The two settings ran alongside each other on {overlap:.0f}% of production "
                        "days, so this is not a comparison of two periods.")
        elif grade == "side_by_side":
            strength = (f"They ran side by side in {both_months} of the {all_months} months either "
                        f"was in use -- rarely on the same day, but not one after the other -- so a "
                        f"simple before-and-after change in material does not explain it. Worth "
                        f"confirming by alternating {best:g} and {current:g} by lot for a week.")
        else:
            switched = _changeover(rows, best, current)
            where = (f"The laser moved from {best:g} to {current:g} in {switched} and has been "
                     f"{gain:.0f} points worse since. "
                     if switched else
                     f"The two settings were run in separate blocks -- only {overlap:.0f}% of "
                     f"production days used more than one, and they shared {both_months} of "
                     f"{all_months} months. ")
            strength = (where + "That makes this a comparison of two PERIODS, so a change in "
                        "material or handling over the same months cannot be ruled out from the "
                        f"data alone. It is cheap to settle: alternate {best:g} and {current:g} by "
                        "lot for a week and this page will read the answer back."
                        + (f" Worth asking why the setting changed in {switched}."
                           if switched else ""))
        annual = sum(1 for t in rows if t.file_date >= latest - timedelta(days=365)
                     and _setting(t) in (best, current))
        # "shorter"/"longer" is a claim about a physical length; "lower"/"higher" is a
        # claim about the number on the sheet. Only laser 2 (DLTS) licenses the first:
        # its label is `Laser Cut Length (mm)`. Laser 1 (LTS) carries the same label
        # with NO unit and values in the thousands (2,950 / 4,000 / 4,100) -- raw
        # machine counts whose scale nobody in this codebase has established. Asserting
        # "longer" there would be reading a unit off a label, which is exactly how
        # `pred_deltas` got described wrongly for a week.
        physical = system == "A"
        if physical:
            direction = "a shorter cut" if best < current else "a longer cut"
        else:
            direction = "a lower setting" if best < current else "a higher setting"
        # "now" is the FLEET's newest trim file, not this group's own -- a model that has
        # not been on this laser in months must not be reported as "now running".
        stale = now is not None and newest < now - timedelta(days=STALE_DAYS)
        if stale:
            title = (f"{laser_label(system)}: cut {best:g} passed {gain:.0f} points more often "
                     f"than {current:g}, the setting it last ran at ({newest:%b %Y})")
        else:
            title = (f"{laser_label(system)}: cut {best:g} passed {gain:.0f} points more often "
                     f"than the {current:g} now running")
        summary = (
            f"On the same limit table, {best:g} left {eligible[best]:.0f}% of "
            f"{len(live[best]):,} tracks inside their linearity limits against "
            f"{eligible[current]:.0f}% of {len(live[current]):,} at {current:g} "
            f"({direction}). It wins in both halves of incoming resistance "
            f"({halves['below']['best_pct']:.0f}% vs {halves['below']['current_pct']:.0f}% below "
            f"{halves['median_resistance']:g} ohm, {halves['above']['best_pct']:.0f}% vs "
            f"{halves['above']['current_pct']:.0f}% above), so the ink does not explain it. "
            + strength)
        if stale:
            summary = (f"This model has not run on this laser since {newest:%B %Y}, so nothing "
                       "here is running now -- it is the record of what worked. " + summary)
        findings.append(Finding(
            model=model, analyzer="cut_setting", category="Cut setting",
            lever="laser_settings", systems=(system,),
            title=title,
            summary=summary,
            n_units=len(live[best]) + len(live[current]),
            strength_name="points more often inside limits",
            strength_value=round(gain, 1),
            expected_gain_points=None if stale else round(gain, 1),
            gain_definition=("" if stale else
                             ("percentage points of tracks leaving the laser inside their per-point "
                              "linearity limits, best setting versus the setting now running, on one "
                              "machine and one limit table")),
            scope_annual_tracks=0 if stale else annual,
            evidence={"group": facts[label], "best": best, "current": current,
                      "halves": halves, "days_mixed_pct": overlap,
                      "months_side_by_side": both_months, "months_total": all_months,
                      "changeover": _changeover(rows, best, current),
                      "track": track_name, "grade": grade, "stale": stale,
                      "last_ran": newest.date().isoformat()}))
    return facts, findings
