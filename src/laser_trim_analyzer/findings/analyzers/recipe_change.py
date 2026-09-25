"""Setting-change detection: the laser's recipe changed -- did the result follow?

A track's recipe is what the laser was told to do: how many cuts, and the
cut-length setting of each. Quarter by quarter, per laser, find the dominant
recipe; when a STABLE quarter's recipe differs from the previous stable one,
that is a change, and the trim result either side of it is the evidence.
Reports what happened. Never claims the recipe CAUSED it: incoming resistance
either side is disclosed beside it, because the two often move together.

So is the LIMIT TABLE. A pass rate is a verdict against a test, and the test can
change in the same months as the recipe: 8232-1 on laser 1 went from one cut to
two during 2024 and from an 89-point table to a 45-point one during 2025. When
more than one table is materially present on either side the finding says so,
and -- where both sides have enough tracks on ONE common table -- gives the move
on that table alone, WITH the dates of each slice: the after-side tracks still on
the old table are whatever part of the after-period had not switched yet, so the
figure is like-for-like in the test but not in time, and it must say so.

(An independent review broke the first version of this: it disclosed only when
the BUSIEST table differed. A before-side graded 60% on a lax table and 40% on a
strict one, against an after-side 100% lax, reported "+15 points" with no
disclosure at all -- and the like-for-like move on the lax table was zero.)
"""
from collections import Counter
from statistics import median
from typing import Any, Dict, FrozenSet, List, Tuple

from ..model import Finding
from ..stats import pct

MIN_BUCKET = 30        # tracks in a quarter before its recipe counts as known
DOMINANT = 0.80        # share one recipe needs for the quarter to count as stable
MIN_SIDE = 100         # graded tracks needed on EACH side before a change is a finding
MIN_MOVE_POINTS = 10.0 # a change that moved the result less than this is history, not a finding
MAX_EVENTS = 3         # per laser, most recent first
MATERIAL = 0.10        # a limit table carrying this share of a side is part of how that side was graded

# This module's ENTIRE recipe is the cut-length setting of each pass (below, sourced from
# trim_passes.laser_cut_length -- see findings/data.py's TrackView.recipe). The very same physical
# setting is ALSO captured, once per file, as a file-level NOMINAL value in trim_setup.parameters
# -- under a different normalised key per laser (core/trim_setup.normalise_key on "Laser Cut
# Length" for laser 1/System B, "Laser Cut Length (mm)" for laser 2/System A; the CLAUDE.md "two
# different quantities" split is about laser 1 vs laser 2 scale, not about which capture is which).
# findings/analyzers/setup_change.py imports this constant (never re-lists the key names) so it
# never re-reports a laser-1/laser-2 cut-length change this module already owns -- fix round 1,
# 2026-09-25 (task-6-review.md, Important #1): a real cut-length change was readable from both
# analyzers, producing two redundantly-worded findings in the same "What changed" group for one
# real event. Owned here, not in setup_change, because THIS module is the one that knows what its
# own recipe consists of; setup_change should never need to re-derive that list by hand.
RECIPE_PARAMETER_KEYS: FrozenSet[str] = frozenset({"laser_cut_length", "laser_cut_length_mm"})


def _quarter(d) -> str:
    return f"{d.year} Q{(d.month - 1) // 3 + 1}"


def describe(recipe) -> str:
    n, cuts = recipe
    settings = ", ".join("?" if c is None else f"{c:g}" for c in cuts)
    word = "cut" if n == 1 else "cuts"
    return f"{n} {word}" + (f" (cut length {settings})" if any(c is not None for c in cuts) else "")


def _table_key(t):
    return t.limit_table.key if t.limit_table is not None else None


def _side(tracks) -> dict:
    graded = [t.linearity_pass for t in tracks if t.linearity_pass is not None]
    rs = [t.untrimmed_resistance for t in tracks if t.untrimmed_resistance]
    tables = Counter(_table_key(t) for t in tracks if t.limit_table is not None)
    with_table = sum(tables.values())
    busiest = None
    if tables:
        key, n = tables.most_common(1)[0]
        tab = next(t.limit_table for t in tracks if _table_key(t) == key)
        busiest = {"key": key, "rows": tab.rows, "graded": tab.graded, "share": round(n / with_table, 3)}
    material = sorted(k for k, n in tables.items() if n / with_table >= MATERIAL)
    return {"n": len(tracks), "trim_pass_pct": pct(graded), "graded_n": len(graded),
            "median_incoming_r": median(rs) if rs else None,
            "first": min(t.file_date for t in tracks).date().isoformat(),
            "last": max(t.file_date for t in tracks).date().isoformat(),
            "limit_table": busiest, "limit_tables_material": material}


def _like_for_like(before_tracks, after_tracks):
    """The move on ONE common limit table, when both sides have MIN_SIDE graded tracks on it."""
    best = None
    for key in {_table_key(t) for t in before_tracks} & {_table_key(t) for t in after_tracks}:
        if key is None:
            continue
        b = [t.linearity_pass for t in before_tracks if _table_key(t) == key and t.linearity_pass is not None]
        a = [t.linearity_pass for t in after_tracks if _table_key(t) == key and t.linearity_pass is not None]
        if min(len(b), len(a)) >= MIN_SIDE and (best is None or len(b) + len(a) > best["n"]):
            tab = next(t.limit_table for t in before_tracks if _table_key(t) == key)
            bd = [t.file_date for t in before_tracks if _table_key(t) == key and t.linearity_pass is not None]
            ad = [t.file_date for t in after_tracks if _table_key(t) == key and t.linearity_pass is not None]
            best = {"graded": tab.graded, "rows": tab.rows, "before_pct": pct(b), "after_pct": pct(a),
                    "before_n": len(b), "after_n": len(a), "n": len(b) + len(a),
                    "before_first": min(bd).date().isoformat(), "before_last": max(bd).date().isoformat(),
                    "after_first": min(ad).date().isoformat(), "after_last": max(ad).date().isoformat(),
                    "after_share": round(len(a) / max(1, sum(1 for t in after_tracks if t.linearity_pass is not None)), 3)}
    return best


def analyze(model: str, tracks, laser_label) -> Tuple[List[Dict[str, Any]], List[Finding]]:
    """(recipe history for the facts strip, findings). Every stable run is HISTORY;
    only a change that moved the result, with enough units either side, is a FINDING."""
    history: List[Dict[str, Any]] = []
    findings: List[Finding] = []
    for system in sorted({t.system for t in tracks}):
        cut = [t for t in tracks if t.system == system and t.passes]
        buckets = {}
        for t in cut:
            buckets.setdefault(_quarter(t.file_date), []).append(t)
        stable = []                                    # [(quarter, recipe, tracks)] in time order
        for q in sorted(buckets, key=lambda k: min(t.file_date for t in buckets[k])):
            ts = buckets[q]
            recipe, count = Counter(t.recipe for t in ts).most_common(1)[0]
            if len(ts) >= MIN_BUCKET and count / len(ts) >= DOMINANT:
                stable.append((q, recipe, [t for t in ts if t.recipe == recipe]))
        runs = []                                      # consecutive stable quarters, same recipe
        for q, recipe, ts in stable:
            if runs and runs[-1]["recipe"] == recipe:
                runs[-1]["tracks"] += ts
                runs[-1]["quarters"].append(q)
            else:
                runs.append({"recipe": recipe, "tracks": list(ts), "quarters": [q]})
        for run in runs:
            history.append({"system": system, "recipe": describe(run["recipe"]),
                            "quarters": run["quarters"], **_side(run["tracks"])})
        events = []
        for before, after in zip(runs, runs[1:]):
            b, a = _side(before["tracks"]), _side(after["tracks"])
            if b["trim_pass_pct"] is None or a["trim_pass_pct"] is None:
                continue
            if min(b["graded_n"], a["graded_n"]) < MIN_SIDE:
                continue                                   # too thin to call
            if abs(a["trim_pass_pct"] - b["trim_pass_pct"]) < MIN_MOVE_POINTS:
                continue                                   # it changed; nothing happened
            events.append((before, after, b, a))
        for before, after, b, a in events[-MAX_EVENTS:]:
            moved = a["trim_pass_pct"] - b["trim_pass_pct"]
            tb, ta = b["limit_table"], a["limit_table"]
            table_changed = bool(tb and ta and tb["key"] != ta["key"])
            # More than one test on EITHER side makes the pooled figures a blend of tests, whether or not
            # the busiest one moved: the mix is what changed in the review's 60/40 -> 100/0 case.
            mixed = (table_changed or len(b["limit_tables_material"]) > 1 or len(a["limit_tables_material"]) > 1
                     or b["limit_tables_material"] != a["limit_tables_material"])
            same_table = _like_for_like(before["tracks"], after["tracks"]) if mixed else None
            table_note = ""
            if mixed:
                if table_changed:
                    table_note = (f" The limit table changed too: most tracks were graded at {tb['graded']} points "
                                  f"before and {ta['graded']} after.")
                else:
                    table_note = (f" More than one limit table was in use ({len(b['limit_tables_material'])} before, "
                                  f"{len(a['limit_tables_material'])} after).")
                table_note += (" A pass rate is a verdict against a test, so part of this move may be a change in "
                               "which test was applied, not in the parts.")
                if same_table:
                    table_note += (f" On the {same_table['graded']}-point table alone the move was "
                                   f"{same_table['before_pct']:.0f}% ({same_table['before_n']:,} tracks, "
                                   f"{same_table['before_first']} to {same_table['before_last']}) to "
                                   f"{same_table['after_pct']:.0f}% ({same_table['after_n']:,} tracks, "
                                   f"{same_table['after_first']} to {same_table['after_last']})")
                    table_note += (" -- the tracks still graded that way, not the whole after-period."
                                   if same_table["after_share"] < 0.9 else ".")
                else:
                    table_note += (f" No single table has {MIN_SIDE} graded tracks on both sides, so there is no "
                                   f"like-for-like figure.")
            findings.append(Finding(
                model=model, analyzer="recipe_change", category="Setting change",
                lever="laser_settings", systems=(system,),
                title=f"{laser_label(system)}: recipe changed from {describe(before['recipe'])} "
                      f"to {describe(after['recipe'])}",
                summary=(f"Between {before['quarters'][-1]} and {after['quarters'][0]} the recipe on "
                         f"{laser_label(system)} changed. Units leaving the laser inside their trim limits went "
                         f"from {b['trim_pass_pct']:.0f}% ({b['graded_n']:,} tracks) to "
                         f"{a['trim_pass_pct']:.0f}% ({a['graded_n']:,} tracks), a move of {moved:+.0f} points. "
                         f"Median incoming resistance was {b['median_incoming_r']:,.0f} before and "
                         f"{a['median_incoming_r']:,.0f} after, so the recipe is not the only thing that changed."
                         if b["median_incoming_r"] and a["median_incoming_r"] else
                         f"Between {before['quarters'][-1]} and {after['quarters'][0]} the recipe on "
                         f"{laser_label(system)} changed; trim pass went from {b['trim_pass_pct']:.0f}% to "
                         f"{a['trim_pass_pct']:.0f}%.") + table_note,
                n_units=b["n"] + a["n"],
                strength_name="tracks on the smaller side of the change",
                strength_value=float(min(b["graded_n"], a["graded_n"])),
                expected_gain_points=None,             # a detection, not a recommendation
                evidence={"before": {**b, "recipe": describe(before["recipe"]), "quarters": before["quarters"]},
                          "after": {**a, "recipe": describe(after["recipe"]), "quarters": after["quarters"]},
                          "moved_points": moved, "limit_table_changed": table_changed,
                          "limit_tables_mixed": mixed, "same_table": same_table}))
    return history, findings
