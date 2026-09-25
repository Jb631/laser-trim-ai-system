"""Station setup: does the laser grade this model to the same limits as final test?

`core/spec_alignment.py` already answers this for the Model page's banner and the Triage list --
position-matched, knee-aware, "compare a unit against itself" (see that module's own docstring).
This turns the SAME comparison into a finding: TRACKER D1 names 8232-1 as the one customer-facing
case of the fleet's 36 (2026-09-20), and James asked for it directly (2026-08-30): "i also want to
know when the trim and test specs dont align."

The one difference from the banner: `compare_station_specs` turns a read failure into
"insufficient" on purpose, because its callers (a warning banner, a list row) may never take a
page down over a hint. An analyzer has no such excuse -- a failed read here is a failed
MEASUREMENT, and CLAUDE.md is explicit that a failure must never be able to look like a result. So
this calls the public `spec_alignment.sample_and_compare` directly, which raises instead of
degrading, and lets that exception reach the findings engine's own guard (`facts["errors"]
["station_setup"]`) uncaught -- never `compare_station_specs`.

Like `machine_compare` and `loss_origin` before it, facts hold every COMPARABLE measurement: once
the comparison clears its own floor (`MIN_MATCHED` matched positions -- spec_alignment's, not
reimplemented here), the result is a fact whether it reads "aligned" or "differs". Only "differs"
(more than `DIFFER_SHARE` of the matched positions) is also a finding. Below `MIN_MATCHED` the
comparison answers nothing, so nothing is recorded either way.

The title is spec ruling 1's (final review, 2026-09-25, M5): "{laser}: the laser grades to limits
about N times wider/narrower than final test over P% of the travel". The laser is the one whose
trim limits the comparison sampled (its newest linked pairs, of any laser -- so `systems` names
those lasers, while the readout stays the model's graded tracks in its latest year, as the
ruling asks). N is the median trim/final-test half-band ratio over the positions that DIFFER --
over all matched positions the medians can be equal when the difference is a part of the travel
-- and a direction is named only when ONE_WAY of those positions lean the same way: on the copy
of 2026-09-25, 8232-1's trim band is the wider at 100% of them (1.7 times), 6126's the narrower
at 99% (0.3), while 8275 (42% wider) and 8340-1 (38%) differ both ways, and say so.

This is VISIBILITY, exactly as spec_alignment's own docstring says: it never re-grades a unit,
picks a "correct" spec, or claims a yield gain -- only that a pass rate compared across trim and
final test on this model is not really a comparison of one test.
"""
from datetime import timedelta
from typing import Any, Dict, List, Tuple

from ...core import spec_alignment
from ...core.models import LASER_ORDER
from ..model import Finding

LOOKBACK_DAYS = 365          # "the model's latest year" -- matches loss_origin's own window
ONE_WAY = 0.9                # the share of differing positions that must lean one way to name it


def _shop_order(systems):
    """Laser 1, 2, 3 -- the shop's order, never the code's letters (A is laser TWO)."""
    return tuple(sorted({s for s in systems if s},
                        key=lambda s: (LASER_ORDER.index(s) if s in LASER_ORDER else len(LASER_ORDER), s)))


def _and(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _how(comparison, pct: float, lasers: int) -> str:
    """"the laser grades to limits about 3x wider than final test over 40% of the travel" --
    or, when the differing positions lean both ways, no single direction."""
    ratio, wider = comparison.differing_ratio, comparison.differing_wider_share
    grades = "the laser grades" if lasers == 1 else "the lasers grade"
    if ratio and wider is not None and wider >= ONE_WAY:
        return f"{grades} to limits about {ratio:.2g}× wider than final test over {pct:.0f}% of the travel"
    if ratio and wider is not None and 1.0 - wider >= ONE_WAY:
        return (f"{grades} to limits about {1.0 / ratio:.2g}× narrower than final test over "
                f"{pct:.0f}% of the travel")
    who = "the laser" if lasers == 1 else "the lasers"
    return (f"{who} and final test grade to different limits over {pct:.0f}% of the travel "
            "(wider in places, narrower in others)")


def analyze(model: str, db, tracks, laser_label) -> Tuple[Dict[str, Any], List[Finding]]:
    facts: Dict[str, Any] = {}
    findings: List[Finding] = []

    dated = [t for t in tracks if t.file_date is not None]
    if not dated:
        return facts, findings
    latest = max(t.file_date for t in dated)
    recent = [t for t in dated if t.file_date >= latest - timedelta(days=LOOKBACK_DAYS)]
    graded = [t for t in recent if t.linearity_pass is not None]
    if not graded:
        return facts, findings              # nothing to report a comparison's population against

    # May raise (a locked/old database, a bad table) -- deliberately not caught: the engine's own
    # guard names it in facts["errors"]["station_setup"], the same as any other analyzer's crash.
    comparison = spec_alignment.sample_and_compare(db, model)
    if comparison.status == "insufficient":
        return facts, findings              # below spec_alignment's own floor -- not a comparable
                                             # measurement, so no fact and no finding either

    sampled = _shop_order(comparison.trim_systems)
    facts.update({
        "status": comparison.status,
        "pct_positions_differing": comparison.pct_positions_differing,
        "matched_positions": comparison.matched_positions,
        "trim_typ_band": comparison.trim_typ_band,
        "ft_typ_band": comparison.ft_typ_band,
        "note": comparison.note,
        "sampled_lasers": [laser_label(s) for s in sampled],
        "differing_ratio": comparison.differing_ratio,
        "differing_wider_share": comparison.differing_wider_share,
    })
    if comparison.status != "differs":
        return facts, findings              # aligned: a real, comparable fact -- never a finding

    systems = sampled or _shop_order(t.system for t in graded)
    pct = comparison.pct_positions_differing
    findings.append(Finding(
        model=model, analyzer="station_setup", category="Station setup",
        lever="laser_limit_table", systems=systems,
        title=f"{_and([laser_label(s) for s in systems])}: {_how(comparison, pct * 100, len(systems))}",
        summary=(
            f"{comparison.note}. This is visibility, not a disposition: nothing here re-grades a "
            "unit or says which station's limits are correct, so no yield gain is claimed -- only "
            "that a pass rate compared across trim and final test here is not really a comparison "
            "of one test."),
        n_units=len(graded),
        strength_name="% of matched positions graded to different limits",
        strength_value=round(pct * 100.0, 1),
        expected_gain_points=None,          # visibility only -- never a rate this analyzer can claim
        evidence={k: facts[k] for k in ("pct_positions_differing", "matched_positions",
                                        "trim_typ_band", "ft_typ_band", "note", "sampled_lasers",
                                        "differing_ratio", "differing_wider_share")}))
    return facts, findings
