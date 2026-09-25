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

This is VISIBILITY, exactly as spec_alignment's own docstring says: it never re-grades a unit,
picks a "correct" spec, or claims a yield gain -- only that a pass rate compared across trim and
final test on this model is not really a comparison of one test.
"""
from datetime import timedelta
from typing import Any, Dict, List, Tuple

from ...core import spec_alignment
from ..model import Finding

LOOKBACK_DAYS = 365          # "the model's latest year" -- matches loss_origin's own window


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

    facts.update({
        "status": comparison.status,
        "pct_positions_differing": comparison.pct_positions_differing,
        "matched_positions": comparison.matched_positions,
        "trim_typ_band": comparison.trim_typ_band,
        "ft_typ_band": comparison.ft_typ_band,
        "note": comparison.note,
    })
    if comparison.status != "differs":
        return facts, findings              # aligned: a real, comparable fact -- never a finding

    systems = tuple(sorted({t.system for t in graded}))
    pct = comparison.pct_positions_differing
    findings.append(Finding(
        model=model, analyzer="station_setup", category="Station setup",
        lever="laser_limit_table", systems=systems,
        title=(f"The laser and final test grade to different limits over "
               f"{pct * 100:.0f}% of the travel"),
        summary=(
            f"{comparison.note}. This is visibility, not a disposition: nothing here re-grades a "
            "unit or says which station's limits are correct, so no yield gain is claimed -- only "
            "that a pass rate compared across trim and final test here is not really a comparison "
            "of one test."),
        n_units=len(graded),
        strength_name="% of matched positions graded to different limits",
        strength_value=round(pct * 100.0, 1),
        expected_gain_points=None,          # visibility only -- never a rate this analyzer can claim
        evidence={"pct_positions_differing": comparison.pct_positions_differing,
                 "matched_positions": comparison.matched_positions,
                 "trim_typ_band": comparison.trim_typ_band, "ft_typ_band": comparison.ft_typ_band,
                 "note": comparison.note}))
    return facts, findings
