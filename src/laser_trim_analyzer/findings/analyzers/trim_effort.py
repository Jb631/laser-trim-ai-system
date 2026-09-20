"""How much of the trimming is needed, and what each cut buys.

Company goal (James, 2026-09-20): "not have to trim at all and if we do trim
as little as possible to not overtrim and not tie up capacity at the laser."
Two questions follow. How many units arrive ALREADY inside their linearity
limits, and are cut anyway? And on tracks that get more than one cut, what
does the last cut add? Facts are always returned (the Model page shows them);
a FINDING is raised only where there is something a person could act on.
"""
from typing import Any, Dict, List, Tuple

from ..grading import in_limits
from ..model import Finding
from ..stats import pct

MIN_N = 100
ARRIVE_IN_SPEC_PCT = 10.0      # below this, "don't trim" is not yet a realistic lever
USELESS_PASS_POINTS = 5.0      # a last cut adding less than this is laser time for nothing


def _facts_for(tracks) -> Dict[str, Any]:
    cut = [t for t in tracks if t.passes]
    arrive = []                                        # graded BEFORE any cut, against cut 1's limits
    for t in cut:
        p1 = t.passes[0]
        if t.untrimmed_errors and len(t.untrimmed_errors) == len(p1.errors):
            g = in_limits(t.untrimmed_errors, p1.upper, p1.lower)
            if g is not None:
                arrive.append((t, g))
    after1 = [g for g in (in_limits(t.passes[0].errors, t.passes[0].upper, t.passes[0].lower) for t in cut) if g is not None]
    multi = [t for t in cut if len(t.passes) >= 2]
    m_first = [in_limits(t.passes[0].errors, t.passes[0].upper, t.passes[0].lower) for t in multi]
    m_last = [in_limits(t.passes[-1].errors, t.passes[-1].upper, t.passes[-1].lower) for t in multi]
    pairs = [(a, b) for a, b in zip(m_first, m_last) if a is not None and b is not None]
    counts: Dict[str, int] = {}
    for t in cut:
        k = str(len(t.passes)) if len(t.passes) < 3 else "3+"
        counts[k] = counts.get(k, 0) + 1
    in_spec = [t for t, g in arrive if g]
    below_floor = [t for t in in_spec if t.final_r_low and t.untrimmed_resistance
                   and t.untrimmed_resistance < t.final_r_low]
    return {"tracks_cut": len(cut), "cuts": counts,
            "graded_untrimmed_n": len(arrive),
            "arrive_in_spec_n": len(in_spec),
            "arrive_in_spec_pct": pct([g for _, g in arrive]),
            "arrive_in_spec_below_r_floor_n": len(below_floor),
            "in_limits_after_cut1_pct": pct(after1), "after_cut1_n": len(after1),
            "multi_cut_n": len(pairs),
            "multi_in_limits_after_first_pct": pct([a for a, _ in pairs]),
            "multi_in_limits_after_last_pct": pct([b for _, b in pairs])}


def analyze(model: str, tracks, laser_label) -> Tuple[Dict[str, Any], List[Finding]]:
    facts: Dict[str, Any] = {}
    findings: List[Finding] = []
    for system in sorted({t.system for t in tracks}):
        f = _facts_for([t for t in tracks if t.system == system])
        if not f["tracks_cut"]:
            continue
        facts[system] = f
        if (f["graded_untrimmed_n"] >= MIN_N and f["arrive_in_spec_pct"] is not None
                and f["arrive_in_spec_pct"] >= ARRIVE_IN_SPEC_PCT):
            mostly_r = f["arrive_in_spec_below_r_floor_n"] * 2 >= f["arrive_in_spec_n"]
            findings.append(Finding(
                model=model, analyzer="trim_effort", category="Trim avoidance",
                lever="ink" if mostly_r else "laser_settings", systems=(system,),
                title=f"{laser_label(system)}: {f['arrive_in_spec_pct']:.0f}% of units arrive already inside "
                      f"their linearity limits and are cut anyway",
                summary=(f"{f['arrive_in_spec_n']:,} of {f['graded_untrimmed_n']:,} tracks fit their per-point "
                         f"limits before the first cut. "
                         + (f"{f['arrive_in_spec_below_r_floor_n']:,} of them arrive below the final resistance "
                            f"floor, so they are being trimmed to raise resistance, not to fix linearity: the "
                            f"lever is the incoming-resistance target."
                            if mostly_r else
                            "Most of them already meet the final resistance floor too, so the cut itself is the "
                            "question: these units may not need the laser at all.")),
                n_units=f["graded_untrimmed_n"],
                strength_name="share arriving inside limits (%)",
                strength_value=f["arrive_in_spec_pct"],
                expected_gain_points=None,             # saves trimming, not yield points
                evidence={"facts": f}))
        if (f["multi_cut_n"] >= MIN_N and f["multi_in_limits_after_last_pct"] is not None):
            added = f["multi_in_limits_after_last_pct"] - f["multi_in_limits_after_first_pct"]
            if added < USELESS_PASS_POINTS:
                findings.append(Finding(
                    model=model, analyzer="trim_effort", category="Pass effectiveness",
                    lever="laser_settings", systems=(system,),
                    title=f"{laser_label(system)}: the extra cuts add only {added:+.0f} points",
                    summary=(f"On {f['multi_cut_n']:,} tracks that received more than one cut, "
                             f"{f['multi_in_limits_after_first_pct']:.0f}% were inside limits after the first cut and "
                             f"{f['multi_in_limits_after_last_pct']:.0f}% after the last. The later cuts are using "
                             f"laser time without moving the result."),
                    n_units=f["multi_cut_n"], strength_name="points added by the later cuts",
                    strength_value=added, expected_gain_points=None, evidence={"facts": f}))
    return facts, findings
