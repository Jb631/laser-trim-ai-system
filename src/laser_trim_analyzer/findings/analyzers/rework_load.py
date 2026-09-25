"""Rework load: laser failures that pass final test after hand trim.

James's own words for the goal this measures against (TRACKER, 2026-09-20): "the goal of the
company is to not have to trim at all and if we do trim as little as possible ... not tie up
capacity at the laser." A unit that fails linearity at the laser and then passes final test looks,
naively, like the laser rejected it for nothing ("overkill"). It usually is not: the unit was HAND
TRIMMED between the two stations, and the confound was resolved on 2026-09-17 by the same-unit
error ratio against a pass/pass control group -- 6607's 512 and 8340-1's 544 "overkills" were
retracted on exactly that evidence (memory `overkill-retrim-confound`). So this analyzer never
reports the trim-FAIL -> FT-PASS count on its own faith; it reports it only once that control
group shows the count could not be explained by final test simply being the looser of the two
tests.

**The readout is never re-derived.** `DatabaseManager.get_model_trim_ft_agreement` is the ONE
definition of a unit's trim disposition -- the unit-DAY's, not the linked file's (per track, the
day's LAST attempt; every track must pass; commit 90dc95e, 2026-08-30, "Overkill counted the wrong
trim attempt"). Its `overkills` field is already exactly "failed trim, passed final test" by that
rule, so `rework_unit_days` here is that number, untouched -- counting it any other way risks a
second, silently different definition of "rework".

**The ratio evidence is not something that method returns**, so confirming the signature needs a
second read: final test's own `linearity_error` against the linked trim's tracks'
`final_linearity_error_shifted` (the WORST track, since linearity is zero-tolerance per track --
"max over tracks" in the brief). `get_model_trim_ft_agreement` was never built to expose either
column. Rather than approximate rework/pass-pass membership from the linked file's own verdict
alone (which IS the day's last attempt in the large majority of cases, per 90dc95e's own numbers,
but not all of them), `_linked_pairs` below reproduces the same unit-day join once more, purely to
reach the two extra columns. `scripts/app_qa_sweep.py` (search "the confound itself is gone")
already reproduces the identical join independently as a standing cross-check against the ORM
method -- this is the same reproduction, extended with the two error columns neither that sweep
nor the ORM method reads.

**Confirmation, never a screen.** The rework group's median ratio (final-test error / laser error)
must fall to CONFIRM_RATIO or below the pass/pass control group's own median -- a unit that merely
passed a looser final test would show a ratio near the control's, not below it (a unit that was
genuinely hand-trimmed shows its error falling BETWEEN the two stations, which a looser test alone
cannot produce). Below MIN_UNIT_DAYS rework unit-days, or MIN_CONTROL control pairs, a median is a
guess, not a rate, and the analyzer says nothing rather than call a thin sample confirmed.

No yield gain is claimed (`expected_gain_points=None`): this counts laser time hand trim is already
spending on the laser's own failures, in the "Laser time you could save" group alongside pass_burden
and trim_effort.
"""
from datetime import timedelta
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from ...core.model_stats import _FAILED_PROCESSING
from ..model import Finding

LOOKBACK_DAYS = 365
# get_model_trim_ft_agreement's own default -- kept identical on purpose so _linked_pairs draws
# from the SAME confidently-linked population the readout (get_model_trim_ft_agreement) counts.
MIN_CONFIDENCE = 0.70
MIN_UNIT_DAYS = 30    # the rework group itself -- below this a median ratio is a guess, not a rate
MIN_CONTROL = 30      # the pass/pass control group the ratio is confirmed against
CONFIRM_RATIO = 0.6   # the rework median must fall to this share of the control median, or below

# Never a measurement (CLAUDE.md: a record that failed processing is not one) and never a trim
# that happened (the blank-template path is UNTRIMMED) -- excluded here the same way
# findings/data.py excludes them from every other analyzer's `tracks`.
_STATUS_EXCLUDE_SQL = ", ".join(f"'{s}'" for s in (*_FAILED_PROCESSING, "UNTRIMMED"))

# Reproduces get_model_trim_ft_agreement's own unit-day join (per track, the day's last attempt;
# every track must pass -- see _linked_trim_ft_rows in database/manager.py) to reach the two error
# columns that method does not return. trim_agg is pre-aggregated per analysis_id before the outer
# join, so (unlike that ORM query, and its scripts/app_qa_sweep.py raw-SQL twin) no track-level fan
# -out reaches the outer SELECT and no GROUP BY is needed there.
_RATIO_SQL = f"""
    WITH att AS (
        SELECT a.unit_id AS uid, t.linearity_pass AS lp,
               ROW_NUMBER() OVER (PARTITION BY a.unit_id, t.track_id
                                   ORDER BY a.file_date DESC, a.id DESC) AS rn
        FROM analysis_results a JOIN track_results t ON t.analysis_id = a.id
        WHERE t.status NOT IN ({_STATUS_EXCLUDE_SQL})
          AND a.unit_id IS NOT NULL AND a.unit_id <> ''
    ),
    disp AS (
        SELECT uid, MIN(CASE WHEN lp = 1 THEN 1 ELSE 0 END) AS tp
        FROM att WHERE rn = 1 GROUP BY uid
    ),
    trim_agg AS (
        SELECT t.analysis_id AS aid,
               MIN(CASE WHEN t.linearity_pass = 1 THEN 1 ELSE 0 END) AS file_all_pass,
               MAX(t.final_linearity_error_shifted) AS max_final_error
        FROM track_results t
        WHERE t.status NOT IN ({_STATUS_EXCLUDE_SQL})
        GROUP BY t.analysis_id
    )
    SELECT f.linearity_error AS ft_error, trim_agg.max_final_error AS trim_error,
           COALESCE(disp.tp, trim_agg.file_all_pass) AS trim_pass, a.system AS system
    FROM final_test_results f
    JOIN analysis_results a ON a.id = f.linked_trim_id
    JOIN trim_agg ON trim_agg.aid = a.id
    LEFT JOIN disp ON disp.uid = a.unit_id
    WHERE f.model = :model AND f.linked_trim_id IS NOT NULL
      AND f.match_confidence >= :conf AND f.linearity_pass = 1
      AND f.file_date >= :cutoff
"""


def _linked_pairs(db, model: str, cutoff) -> List[Tuple[Optional[float], Optional[float], bool, str]]:
    """(final-test error, laser final error, unit-day trim PASS, laser) for every confidently
    linked, FT-PASS pair in the window -- one row per final-test record, the same grain
    `get_model_trim_ft_agreement` counts by. Either error can be None (not every linked pair has
    a usable reading on both sides, e.g. a track whose final_linearity_error_shifted was never
    computed) -- callers must filter that, never treat a missing reading as zero.
    """
    with db.session() as s:
        rows = s.execute(text(_RATIO_SQL), {
            "model": model, "conf": MIN_CONFIDENCE,
            # Bound as a formatted string, never a raw datetime -- text() does not get SQLAlchemy's
            # own DATETIME bind_processor (global-constraints.md); file_date is stored in this
            # exact format, so a plain string comparison sorts identically to a chronological one.
            "cutoff": f"{cutoff:%Y-%m-%d %H:%M:%S.%f}"}).fetchall()
    return [(row[0], row[1], bool(row[2]), str(row[3])) for row in rows]


def _ratio(ft_error: Optional[float], trim_error: Optional[float]) -> Optional[float]:
    """final-test error / laser error, or None when either reading is missing or unusable. A
    non-positive laser error is not a plausible "worst point" magnitude -- skipped, never divided
    by, the same plausibility discipline core/model_stats.py applies before averaging anything."""
    if ft_error is None or trim_error is None or ft_error < 0 or trim_error <= 0:
        return None
    return ft_error / trim_error


def analyze(model: str, db, tracks, laser_label) -> Tuple[Dict[str, Any], List[Finding]]:
    facts: Dict[str, Any] = {}
    findings: List[Finding] = []
    dated = [t for t in tracks if t.file_date is not None]
    if not dated:
        return facts, findings
    latest = max(t.file_date for t in dated)
    cutoff = latest - timedelta(days=LOOKBACK_DAYS)

    # May raise (a locked/unreadable database) -- deliberately not caught: the engine's own guard
    # names it in facts["errors"]["rework_load"], the same as any other analyzer's crash.
    agreement = db.get_model_trim_ft_agreement(model, cutoff_date=cutoff, min_confidence=MIN_CONFIDENCE)
    linked = agreement.get("linked") or 0
    n_rework = agreement.get("overkills") or 0
    facts["linked"] = linked
    facts["rework_unit_days"] = n_rework
    if not linked:
        facts["confirmed"] = False
        facts["note"] = "no final tests are linked to a trim analysis for this model in the window"
        return facts, findings
    if n_rework < MIN_UNIT_DAYS:
        facts["confirmed"] = False
        facts["note"] = f"{n_rework} rework unit-days in the window, below the {MIN_UNIT_DAYS} floor"
        return facts, findings

    pairs = _linked_pairs(db, model, cutoff)
    rework_ratios = [r for r in (_ratio(fe, te) for fe, te, tp, sysname in pairs if not tp)
                     if r is not None]
    control_ratios = [r for r in (_ratio(fe, te) for fe, te, tp, sysname in pairs if tp)
                      if r is not None]
    # The laser(s) whose FAILURES are being hand-trimmed -- not the model's every laser, and not
    # a re-derivation of the readout: purely which system each already-classified rework pair's
    # own linked trim ran on.
    rework_systems = tuple(sorted({sysname for _, _, tp, sysname in pairs if not tp}))
    facts["rework_ratio_n"] = len(rework_ratios)
    facts["control_n"] = len(control_ratios)
    if len(rework_ratios) < MIN_UNIT_DAYS or len(control_ratios) < MIN_CONTROL:
        facts["confirmed"] = False
        facts["note"] = ("not enough linked pairs with usable linearity-error readings to "
                         "confirm the signature")
        return facts, findings

    med_rework = median(rework_ratios)
    med_control = median(control_ratios)
    facts["median_ratio_rework"] = round(med_rework, 3)
    facts["median_ratio_control"] = round(med_control, 3)
    confirmed = med_rework <= CONFIRM_RATIO * med_control
    facts["confirmed"] = confirmed
    if not confirmed:
        facts["note"] = ("the rework group's final-test error did not fall enough against the "
                         "pass/pass control group to confirm hand trim rather than a looser test")
        return facts, findings

    systems = rework_systems or tuple(sorted({t.system for t in dated}))
    findings.append(Finding(
        model=model, analyzer="rework_load", category="Rework load",
        lever="laser_settings", systems=systems,
        title=(f"{laser_label(systems[0])}: {n_rework:,} units a year fail here and pass final "
               "test after rework"),
        summary=(
            f"{n_rework:,} unit-days in the last year failed linearity at the laser and then "
            "passed final test. That is hand trim, not an unnecessary rejection: on these units "
            f"final test's own linearity error runs to a median of {med_rework:.0%} of what the "
            f"laser measured, against {med_control:.0%} on units that passed both stations -- the "
            "error genuinely fell between the two stations, which a final test that was simply "
            "looser would not produce. No gain is claimed: this counts laser time hand trim is "
            "already spending on this model's own failures."),
        n_units=n_rework,
        strength_name="median final-test / laser error ratio, reworked units",
        strength_value=round(med_rework, 3),
        expected_gain_points=None,           # hand-trim labour avoided, never a yield rate
        evidence={"facts": {"rework_unit_days": n_rework, "control_n": len(control_ratios),
                            "median_ratio_rework": round(med_rework, 3),
                            "median_ratio_control": round(med_control, 3)}}))
    return facts, findings
