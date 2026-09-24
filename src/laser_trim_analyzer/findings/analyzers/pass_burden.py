"""Multi-pass burden: cuts the recipe did not ask for, and the laser time they cost.

Company goal (James, 2026-09-20): "not have to trim at all and if we do trim as
little as possible to not overtrim and not tie up capacity at the laser."

The trap this analyzer exists to avoid is the one that caught the analysis by
hand first. "86% of 8232-1 needs a second pass" looked like a quality problem
and was nothing of the kind: the shop moved from a one-cut recipe to a two-cut
recipe in 2024, and on a two-cut recipe a second cut is not an extra pass, it
is the recipe. Counting raw pass numbers measures the process sheet, not the
parts.

So the unit here is a group that shares one recipe -- one laser, one track
name, one first-cut setting -- and the question inside it is how often a track
needed MORE cuts than that recipe's own normal. That is laser time nobody
planned for, and it is the number worth ranking models by.

It claims no yield gain. Removing an unplanned pass frees capacity; whether it
also changes the verdict is `trim_effort`'s question, not this one.
"""
from collections import Counter
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..model import Finding

LOOKBACK_DAYS = 365
MIN_TRACKS = 150           # per recipe group
MIN_EXTRA_SHARE = 15.0     # below this the extra passes are noise against the recipe
MIN_GROUP_SHARE = 20.0     # a recipe group smaller than this share of the laser is a sideshow


def _first_cut(t) -> Optional[float]:
    if not t.passes:
        return None
    v = t.passes[0].cut_setting
    return None if v is None else round(float(v), 4)


def _facts_for(rows) -> Dict[str, Any]:
    """The recipe's normal cut count, and what was spent above it."""
    counts = Counter(len(t.passes) for t in rows)
    # On a tie, take the LARGER cut count as the recipe's normal. Counter.most_common
    # breaks ties by insertion order, so a group split evenly between one and two cuts
    # would report a 50% burden or a 0% one depending on which track happened to be
    # read first. Preferring the larger count is both deterministic and conservative:
    # a tie can never manufacture a burden, only fail to report a real one.
    best = max(counts.values())
    normal = max(k for k, v in counts.items() if v == best)
    extra_tracks = [t for t in rows if len(t.passes) > normal]
    extra_passes = sum(len(t.passes) - normal for t in extra_tracks)
    return {"n": len(rows), "normal_cuts": normal,
            "cut_counts": {str(k): v for k, v in sorted(counts.items())},
            "tracks_over_recipe": len(extra_tracks),
            "share_over_recipe": round(100.0 * len(extra_tracks) / len(rows), 1),
            "unplanned_passes": extra_passes,
            "unplanned_passes_per_100_tracks": round(100.0 * extra_passes / len(rows), 1)}


def analyze(model: str, tracks, laser_label) -> Tuple[Dict[str, Any], List[Finding]]:
    facts: Dict[str, Any] = {}
    findings: List[Finding] = []
    dated = [t for t in tracks if t.file_date is not None and t.passes]
    if not dated:
        return facts, findings
    latest = max(t.file_date for t in dated)
    recent = [t for t in dated if t.file_date >= latest - timedelta(days=LOOKBACK_DAYS)]

    by_laser: Dict[str, List] = {}
    for t in recent:
        by_laser.setdefault(t.system, []).append(t)

    for system, laser_rows in sorted(by_laser.items()):
        groups: Dict[Tuple, List] = {}
        for t in laser_rows:
            cut = _first_cut(t)
            if cut is not None:
                groups.setdefault((t.track_name, cut), []).append(t)
        for (track_name, cut), rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(rows) < MIN_TRACKS:
                continue
            f = _facts_for(rows)
            f["cut_setting"] = cut
            label = f"{laser_label(system)} · {track_name} · cut {cut:g}"
            facts[label] = f
            # A recipe only a fraction of the laser's work runs on cannot carry a
            # recommendation about the laser's capacity.
            share_of_laser = 100.0 * len(rows) / len(laser_rows)
            if share_of_laser < MIN_GROUP_SHARE:
                continue
            if f["share_over_recipe"] < MIN_EXTRA_SHARE:
                continue
            findings.append(Finding(
                model=model, analyzer="pass_burden", category="Multi-pass burden",
                lever="laser_settings", systems=(system,),
                title=(f"{laser_label(system)}: {f['share_over_recipe']:.0f}% of tracks need more "
                       f"than the {f['normal_cuts']} cut{'s' if f['normal_cuts'] != 1 else ''} "
                       f"the recipe asks for"),
                summary=(
                    f"On cut setting {cut:g}, {f['tracks_over_recipe']:,} of {f['n']:,} tracks in the "
                    f"last year took more than {f['normal_cuts']} cut"
                    f"{'s' if f['normal_cuts'] != 1 else ''} — "
                    f"{f['unplanned_passes']:,} laser passes nobody planned for, "
                    f"{f['unplanned_passes_per_100_tracks']:.0f} per 100 tracks. "
                    "The recipe's own normal is counted per recipe, so a model that simply runs a "
                    "two-cut process is not reported here; this is the work ON TOP of the process "
                    "sheet. Whether a different cut would avoid it is the cut-setting finding."),
                n_units=f["n"],
                strength_name="unplanned passes per 100 tracks",
                strength_value=f["unplanned_passes_per_100_tracks"],
                expected_gain_points=None,        # capacity, not yield points
                scope_annual_tracks=f["n"],
                # The track travels with the finding (it is in the facts LABEL, not in `f`): two
                # tracks' findings can read identically, and the Findings page tells their rows
                # apart -- and names the track in each -- by it.
                evidence={"facts": f, "share_of_laser": round(share_of_laser, 1),
                          "track": track_name}))
    return facts, findings
