"""Setting-change detection: the laser's recipe changed -- did the result follow?

A track's recipe is what the laser was told to do: how many cuts, and the
cut-length setting of each. Quarter by quarter, per laser, find the dominant
recipe; when a STABLE quarter's recipe differs from the previous stable one,
that is a change, and the trim result either side of it is the evidence.
Reports what happened. Never claims the recipe CAUSED it: incoming resistance
either side is disclosed beside it, because the two often move together.
"""
from collections import Counter
from statistics import median
from typing import Any, Dict, List, Tuple

from ..model import Finding
from ..stats import pct

MIN_BUCKET = 30        # tracks in a quarter before its recipe counts as known
DOMINANT = 0.80        # share one recipe needs for the quarter to count as stable
MIN_SIDE = 100         # graded tracks needed on EACH side before a change is a finding
MIN_MOVE_POINTS = 10.0 # a change that moved the result less than this is history, not a finding
MAX_EVENTS = 3         # per laser, most recent first


def _quarter(d) -> str:
    return f"{d.year} Q{(d.month - 1) // 3 + 1}"


def describe(recipe) -> str:
    n, cuts = recipe
    settings = ", ".join("?" if c is None else f"{c:g}" for c in cuts)
    word = "cut" if n == 1 else "cuts"
    return f"{n} {word}" + (f" (cut length {settings})" if any(c is not None for c in cuts) else "")


def _side(tracks) -> dict:
    graded = [t.linearity_pass for t in tracks if t.linearity_pass is not None]
    rs = [t.untrimmed_resistance for t in tracks if t.untrimmed_resistance]
    return {"n": len(tracks), "trim_pass_pct": pct(graded), "graded_n": len(graded),
            "median_incoming_r": median(rs) if rs else None,
            "first": min(t.file_date for t in tracks).date().isoformat(),
            "last": max(t.file_date for t in tracks).date().isoformat()}


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
                         f"{a['trim_pass_pct']:.0f}%."),
                n_units=b["n"] + a["n"],
                strength_name="tracks on the smaller side of the change",
                strength_value=float(min(b["graded_n"], a["graded_n"])),
                expected_gain_points=None,             # a detection, not a recommendation
                evidence={"before": {**b, "recipe": describe(before["recipe"]), "quarters": before["quarters"]},
                          "after": {**a, "recipe": describe(after["recipe"]), "quarters": after["quarters"]},
                          "moved_points": moved}))
    return history, findings
