"""Rework load: laser failures that pass final test after hand trim.

James's own words for the goal this measures against (TRACKER, 2026-09-20): "the goal of the
company is to not have to trim at all and if we do trim as little as possible ... not tie up
capacity at the laser." A unit that fails linearity at the laser and then passes final test looks,
naively, like the laser rejected it for nothing ("overkill"). It usually is not: the unit was HAND
TRIMMED between the two stations (memory `overkill-retrim-confound`, 2026-09-17). So this analyzer
never reports the trim-FAIL -> FT-PASS count on its own faith; it reports it only once the units'
own errors show they were worked on between the stations.

**The count is unit-DAYS, from the one definition.** `DatabaseManager.get_model_trim_ft_agreement`
decides a unit's trim disposition (per track the day's LAST attempt; every track must pass; a file
that failed processing never decides it) and its `overkill_unit_days` counts the units behind the
overkills -- `overkill_unit_days_by_system` the same per laser (the linked file's). A finding names
ONE laser, and that laser's count is its readout and its title, untouched. Its `overkills` counts
final-test RECORDS -- 6607 tests each track in its own file, so 533 records were 261 unit-days
(review of 85222c4, 2026-09-25) -- and is not used here.

**Per laser** (final review, 2026-09-25, M3). Each laser's reworked units are compared with ITS
OWN untouched units: two stations downstream of two machines need not share an error floor, and
a control drawn across lasers would let one laser's untouched units vouch for the other's rework.
One verdict, and at most one finding, per laser; the numbers each rests on sit under
`facts["by_laser"]`.

**The metric is what hand trim changes** (review of 85222c4). Per linked (unit, track) pair: the
largest |corrected error| over the positions BOTH stations grade -- trim rows that carry limits,
inside the final test's graded window, on one position axis -- each station's sweep corrected with
its OWN stored offset (and slope where stored) through `core/analyzer.corrected_errors`. The
stored scalars (`final_linearity_error_shifted`, `linearity_error`) cannot do this: they are maxima
over each station's whole sweep, and they sit where the other station never grades (6607: the
laser's worst point was beyond final test's +/-14 on 1,090 of 1,091 pairs; 8340-1: on rows with no
limits; 8232-1: final test's outside its graded window) -- which is why 85222c4's "no signal" was
the wrong quantity, not the data. A pair with no common graded position is skipped and counted,
never scored; a reading of JUNK_VOLTS or more is not a linearity error, and is ignored and counted.

**Each final test on ITS track.** A final-test record belongs to one track: its own track letter
when it has one, else its serial -- digits only or an 'A' suffix is Track A, 'B'/'b' is Track B
(6607's two files per unit, confirmed by matching curves on untouched units). It is compared with
THAT track's last attempt of the day and judged by that track's own verdict: a final test of a
track that passed at the laser is an untouched control reading even when the unit's other track
failed. Then one reading per unit-day: per track the latest final test, and the unit's track with
the largest laser error -- same-track pairs only, never one track's final test over another's laser.

**Compare like with like, by a rank test.** Final test has an error floor, so its ratio to the
laser's error falls as the laser's error rises even on units nobody touched: a plain ratio of
medians confirms partly by construction. So the reworked units' ratios are compared with those of
the pass/pass units in the TOP THIRD of laser error -- the untouched units that started nearest
them. Confirmed when the reworked ratios are significantly LOWER (a one-sided Mann-Whitney U test,
`findings/stats.mann_whitney_lower`, at CONFIRM_P) AND the shift is not trivial (the rework median
at most MAX_EFFECT_RATIO of the top third's -- a large sample must not confirm a tiny shift), with
MIN_UNIT_DAYS reworked unit-days and MIN_CONTROL units in the top third. Fix round 2, 2026-09-25:
round 1's fixed 0.8 ratio cut rested on numbers that did not reproduce, and 6607's verdict under it
turned on how one unit-day's final tests were reduced. U, p, both medians, both sizes, the effect
ratio and the REDUCTION are facts always; the floors gate only the verdict. Confirmation, never a
screen.

No yield gain is claimed (`expected_gain_points=None`): this counts laser time hand trim is already
spending on the laser's own failures, in the "Laser time you could save" group alongside pass_burden
and trim_effort.
"""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ...core.analyzer import corrected_errors, max_abs_measured
from ...core.ft_overlay import normalize_track_id, positions_on_trim_axis, ungraded_indices
from ...core.model_stats import failed_processing_statuses
from ...core.models import LASER_ORDER
from ..model import Finding
from ..stats import mann_whitney_lower

LOOKBACK_DAYS = 365
# get_model_trim_ft_agreement's own default -- identical on purpose, so the sweeps compared here are
# drawn from the same confidently-linked population the readout counts.
MIN_CONFIDENCE = 0.70
MIN_UNIT_DAYS = 30      # reworked unit-days with a ratio -- below this the test is not a verdict
MIN_CONTROL = 20        # pass/pass unit-days in the TOP THIRD of laser error, the comparison group
CONFIRM_P = 0.01        # one-sided Mann-Whitney: the reworked ratios LOWER than the top third's
MAX_EFFECT_RATIO = 0.9  # ...and by enough: the rework median at most this share of theirs
JUNK_VOLTS = 1.0        # a worst |error| this large is not a linearity reading (a broken column)
# How one unit-day's final tests become ONE reading (round 1's same-track choice, unchanged): named
# in the facts, because a verdict that turned on it would not be a verdict.
REDUCTION = ("per track the latest final test against that track's last laser attempt; per "
             "unit-day the track with the largest laser error")

_CHUNK = 500          # ids per IN (...) -- far below SQLite's variable limit
_SERIAL_TRACK = re.compile(r"^\d+([AaBb])?$")


def ft_track_letter(serial) -> Optional[str]:
    """The track a final-test record is about, from its serial: digits only or an 'A' suffix is
    Track A, a 'B'/'b' suffix is Track B, anything else says nothing (None)."""
    m = _SERIAL_TRACK.match(str(serial or "").strip())
    if not m:
        return None
    return (m.group(1) or "A").upper()


def _ft_letter(ft_track_id, serial) -> Optional[str]:
    own = normalize_track_id(ft_track_id)
    return own if own in ("A", "B") else ft_track_letter(serial)


def pair_trim_track(letter: Optional[str],
                    unit_tracks: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The trim track (its last attempt of the day) a final-test record belongs to, or None.

    A unit-day with one track: that track -- unless the final test names another letter and the
    track has a letter of its own (only Track B was trimmed that day and this is Track A's final
    test: comparing them is the Track-A-final-test-against-Track-B-laser mistake). With two or
    more tracks the final test must say which, and exactly one track must answer to it.
    """
    if len(unit_tracks) == 1:
        (track_id, attempt), = unit_tracks.items()
        own = normalize_track_id(track_id)
        if letter is not None and own not in ("", "DEFAULT") and own != letter:
            return None
        return attempt
    if letter is None:
        return None
    matches = [a for tid, a in unit_tracks.items() if normalize_track_id(tid) == letter]
    return matches[0] if len(matches) == 1 else None


def _real(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and x == x and x not in (float("inf"), float("-inf")))


def _has_limits(side: Dict[str, Any], i: int) -> bool:
    up, lo = side.get("upper") or [], side.get("lower") or []
    return i < len(up) and i < len(lo) and _real(up[i]) and _real(lo[i])


def graded_maxima(ft: Dict[str, Any], trim: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """(final test's, the laser's) largest |corrected error| over the travel BOTH stations grade,
    or None when they grade no common stretch of it (or either side measured nothing there).

    `ft` / `trim`: positions, errors, upper, lower, offset, slope, theory -- and for final test its
    graded_start / graded_end. A trim row is graded when it carries limits; a final-test row when
    it also lies inside the station's graded window. Final test's positions are placed on the
    laser's axis first (core/ft_overlay.positions_on_trim_axis). Each station is corrected with
    its OWN offset and slope (core/analyzer.corrected_errors); a blank reading is ungraded,
    never 0.0 (core/analyzer.max_abs_measured).
    """
    t_pos = list(trim.get("positions") or [])
    f_pos, _how = positions_on_trim_axis(ft.get("positions") or [], t_pos)   # never rescaled
    if f_pos is None:
        return None
    f_err, t_err = list(ft.get("errors") or []), list(trim.get("errors") or [])
    f_corr = corrected_errors(f_err, ft.get("offset"), ft.get("slope"), ft.get("theory") or None)
    t_corr = corrected_errors(t_err, trim.get("offset"), trim.get("slope"),
                              trim.get("theory") or None)
    outside = ungraded_indices(len(f_err), ft.get("graded_start"), ft.get("graded_end"))
    f_pts = [(f_pos[i], f_corr[i]) for i in range(min(len(f_pos), len(f_corr)))
             if i not in outside and _has_limits(ft, i) and _real(f_pos[i]) and _real(f_corr[i])]
    t_pts = [(t_pos[i], t_corr[i]) for i in range(min(len(t_pos), len(t_corr)))
             if _has_limits(trim, i) and _real(t_pos[i]) and _real(t_corr[i])]
    if not f_pts or not t_pts:
        return None
    lo = max(min(p for p, _ in f_pts), min(p for p, _ in t_pts))
    hi = min(max(p for p, _ in f_pts), max(p for p, _ in t_pts))
    if not hi > lo:
        return None
    eps = 1e-9 * max(1.0, abs(lo), abs(hi))
    f_max = max_abs_measured([c for p, c in f_pts if lo - eps <= p <= hi + eps])
    t_max = max_abs_measured([c for p, c in t_pts if lo - eps <= p <= hi + eps])
    if f_max is None or t_max is None:
        return None
    return f_max, t_max


def _ratio(ft_error: Optional[float], laser_error: Optional[float]) -> Tuple[Optional[float], bool]:
    """(final-test error / laser error, was it junk). A reading of JUNK_VOLTS or more on either
    side is not a linearity error -- ignored, and the caller counts it. A laser reading of zero has
    no ratio at all."""
    if ft_error is None or laser_error is None:
        return None, False
    if ft_error >= JUNK_VOLTS or laser_error >= JUNK_VOLTS:
        return None, True
    if ft_error < 0 or laser_error <= 0:
        return None, False
    return ft_error / laser_error, False


@dataclass(frozen=True)
class _Reading:
    """One scored (final test, trim track) pair."""
    unit: Any                 # unit_id, or ("linked file", analysis id) -- the disposition's own key
    track_row: int            # the trim track's last attempt (track_results.id)
    rework: bool              # judged by THAT track's own verdict
    when: Tuple               # the final test's own moment, for "latest"
    laser: float
    ft: float
    ratio: float
    system: str = ""          # the trim track's laser, as its code letter


def _chunks(ids: Sequence[Any]) -> Iterable[List[Any]]:
    ids = list(ids)
    for i in range(0, len(ids), _CHUNK):
        yield ids[i:i + _CHUNK]


def _moment(dt: Optional[datetime]) -> Tuple[bool, datetime]:
    return (dt is not None, dt or datetime.min)


def _final_tests(db, model: str, cutoff: datetime) -> List[Dict[str, Any]]:
    """Every confidently linked final test that PASSED, in the window -- one dict per FT track."""
    from ...database.models import FinalTestResult as FT, FinalTestTrack as FTT
    with db.session() as s:
        rows = (s.query(FT.id, FT.serial, FT.test_date, FT.file_date, FT.linked_trim_id,
                        FTT.track_id, FTT.position_data, FTT.electrical_angle_data, FTT.error_data,
                        FTT.theory_data, FTT.upper_limits, FTT.lower_limits, FTT.optimal_offset,
                        FTT.optimal_slope, FTT.graded_start, FTT.graded_end)
                .outerjoin(FTT, FTT.final_test_id == FT.id)
                .filter(FT.model == model, FT.linked_trim_id.isnot(None),
                        FT.match_confidence >= MIN_CONFIDENCE,
                        FT.linearity_pass == True,  # noqa: E712 -- SQL boolean
                        FT.file_date >= cutoff)
                .all())
    return [{"id": r[0], "serial": r[1], "when": (_moment(r[2] or r[3]), r[0]), "linked": r[4],
             "track_id": r[5],
             "positions": list(r[6] or []) or list(r[7] or []), "errors": list(r[8] or []),
             "theory": list(r[9] or []), "upper": list(r[10] or []), "lower": list(r[11] or []),
             "offset": r[12], "slope": r[13], "graded_start": r[14], "graded_end": r[15]}
            for r in rows]


def _unit_tracks(db, linked_ids: Iterable[int]
                 ) -> Tuple[Dict[int, Any], Dict[Any, Dict[str, Dict[str, Any]]]]:
    """({linked analysis id: its unit key}, {unit key: {track_id: that track's last attempt}}).

    The same unit-day rule the disposition uses: per track the LAST attempt of the day (file time,
    then id), over every file of the unit-day, tracks with no disposition (untrimmed, or whose
    processing failed) left out; a file with no unit_id stands alone. A linked file whose every
    track is untrimmed or unreadable is left out of the first mapping -- it is not a comparison,
    and the readout does not link it either.
    """
    from ...database.models import AnalysisResult as AR, StatusType, TrackResult as TR
    no_disposition = [*failed_processing_statuses(), StatusType.UNTRIMMED]
    linked_ids = sorted(set(linked_ids))
    unit_of: Dict[int, Optional[str]] = {}
    attempts: List[Tuple] = []
    with db.session() as s:
        for chunk in _chunks(linked_ids):
            unit_of.update({aid: uid for aid, uid in
                            s.query(AR.id, AR.unit_id).filter(AR.id.in_(chunk)).all()})
        cols = (AR.id, AR.unit_id, AR.file_date, AR.system, TR.id, TR.track_id, TR.linearity_pass)
        units = sorted({uid for uid in unit_of.values() if uid})
        for chunk in _chunks(units):
            attempts += (s.query(*cols).join(TR, TR.analysis_id == AR.id)
                         .filter(AR.unit_id.in_(chunk), TR.status.notin_(no_disposition)).all())
        alone = [aid for aid, uid in unit_of.items() if not uid]
        for chunk in _chunks(alone):
            attempts += (s.query(*cols).join(TR, TR.analysis_id == AR.id)
                         .filter(AR.id.in_(chunk), TR.status.notin_(no_disposition)).all())
    by_unit: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    measured = set()
    for aid, uid, file_date, system, track_row, track_id, passed in attempts:
        measured.add(aid)
        key = uid if uid else ("linked file", aid)
        rank = (_moment(file_date), aid, track_row)
        tracks = by_unit.setdefault(key, {})
        if track_id not in tracks or rank > tracks[track_id]["rank"]:
            tracks[track_id] = {"rank": rank, "track_row": track_row, "passed": passed is True,
                                "system": str(getattr(system, "value", system) or "")}
    linked = {aid: (uid if uid else ("linked file", aid))
              for aid, uid in unit_of.items() if aid in measured}
    return linked, by_unit


def _trim_arrays(db, track_rows: Iterable[int]) -> Dict[int, Dict[str, Any]]:
    from ...database.models import TrackResult as TR
    out: Dict[int, Dict[str, Any]] = {}
    with db.session() as s:
        for chunk in _chunks(sorted(set(track_rows))):
            for r in (s.query(TR.id, TR.position_data, TR.error_data, TR.theory_data,
                              TR.upper_limits, TR.lower_limits, TR.optimal_offset, TR.optimal_slope)
                      .filter(TR.id.in_(chunk)).all()):
                out[r[0]] = {"positions": list(r[1] or []), "errors": list(r[2] or []),
                             "theory": list(r[3] or []), "upper": list(r[4] or []),
                             "lower": list(r[5] or []), "offset": r[6], "slope": r[7]}
    return out


def _scored_pairs(db, model: str, cutoff: datetime) -> Dict[str, Any]:
    """Every (final test, trim track) pair in the window that could be scored, one _Reading each,
    with what was left out: unpaired final tests, pairs with no common graded position, junk."""
    fts = _final_tests(db, model, cutoff)
    linked, by_unit = _unit_tracks(db, (f["linked"] for f in fts))
    paired: List[Tuple[Dict[str, Any], Any, Dict[str, Any]]] = []
    unpaired = 0
    rework_systems = set()
    for f in fts:
        unit = linked.get(f["linked"])
        if unit is None:                    # linked to a file with no measured track
            continue
        track = pair_trim_track(_ft_letter(f["track_id"], f["serial"]), by_unit.get(unit) or {})
        if track is None:
            unpaired += 1
            continue
        if not track["passed"]:
            rework_systems.add(track["system"])
        paired.append((f, unit, track))

    arrays = _trim_arrays(db, (t["track_row"] for _, _, t in paired))
    skipped = junk = 0
    readings: List[_Reading] = []
    for f, unit, track in paired:
        got = graded_maxima(f, arrays.get(track["track_row"]) or {})
        ratio, was_junk = (None, False) if got is None else _ratio(*got)
        if was_junk:
            junk += 1
            continue
        if ratio is None:
            skipped += 1                    # no common graded position (or nothing to divide by)
            continue
        readings.append(_Reading(unit=unit, track_row=track["track_row"], rework=not track["passed"],
                                 when=f["when"], laser=got[1], ft=got[0], ratio=ratio,
                                 system=track["system"]))
    return {"readings": readings, "skipped": skipped, "junk": junk, "unpaired": unpaired,
            "rework_systems": rework_systems}


def _one_per_unit_day(readings: Iterable[_Reading]) -> Tuple[List[_Reading], List[_Reading]]:
    """(reworked, pass/pass): ONE reading per unit-day, laser and group, by REDUCTION -- per track
    the latest final test, then the unit-day's track with the largest laser error (same-track
    pairs only, never one track's final test over another track's laser)."""
    latest: Dict[Tuple, _Reading] = {}
    for r in readings:
        key = (r.rework, r.unit, r.track_row)
        if key not in latest or r.when > latest[key].when:          # per track: the latest test
            latest[key] = r
    per_unit: Dict[Tuple, _Reading] = {}
    for r in latest.values():               # per unit-day: the track with the largest laser error
        key = (r.rework, r.unit, r.system)
        best = per_unit.get(key)
        if best is None or (r.laser, r.ft, r.track_row) > (best.laser, best.ft, best.track_row):
            per_unit[key] = r
    return ([r for (rw, _, _), r in per_unit.items() if rw],
            [r for (rw, _, _), r in per_unit.items() if not rw])


def _populations(db, model: str, cutoff: datetime) -> Dict[str, Any]:
    """The reworked and the pass/pass unit-days, one reading each, with what was left out."""
    scored = _scored_pairs(db, model, cutoff)
    rework, control = _one_per_unit_day(scored["readings"])
    return {"rework": rework, "control": control, "skipped": scored["skipped"],
            "junk": scored["junk"], "unpaired": scored["unpaired"],
            "rework_systems": scored["rework_systems"]}


def rank_comparison(rework: Sequence[_Reading], top: Sequence[_Reading]) -> Dict[str, Optional[float]]:
    """The test the verdict rests on, unrounded: U and one-sided p for "the reworked ratios sit
    LOWER than the top third's", both medians and the effect ratio -- None where uncomputable
    (either group empty; p also when every ratio is tied)."""
    out: Dict[str, Optional[float]] = {"u": None, "p": None, "median_rework": None,
                                       "median_top": None, "effect": None}
    if rework:
        out["median_rework"] = median(r.ratio for r in rework)
    if top:
        out["median_top"] = median(r.ratio for r in top)
    test = mann_whitney_lower([r.ratio for r in rework], [r.ratio for r in top])
    if test is not None:
        out["u"], out["p"] = test
    if out["median_rework"] is not None and out["median_top"]:
        out["effect"] = out["median_rework"] / out["median_top"]
    return out


def verdict(n_rework: int, n_top: int, test: Dict[str, Optional[float]]) -> Tuple[bool, Optional[str]]:
    """(confirmed, why not). The floors first, then significance, then the size of the shift."""
    p, effect = test["p"], test["effect"]
    if n_rework < MIN_UNIT_DAYS or n_top < MIN_CONTROL:
        return False, (f"{n_rework} reworked unit-days and {n_top} comparable pass/pass units "
                       "could be read over the travel both stations grade -- the test needs "
                       f"{MIN_UNIT_DAYS} and {MIN_CONTROL}")
    if p is None or not p < CONFIRM_P:
        shown = ("every ratio is tied, so no rank test can be run" if p is None
                 else f"one-sided rank test p = {p:.2g}")
        return False, ("the reworked units' error did not fall significantly more between the "
                       "stations than it did for the untouched units that started nearest them "
                       f"({shown}; confirming needs p below {CONFIRM_P:g})")
    if effect is None or effect > MAX_EFFECT_RATIO:
        size = "cannot be sized" if effect is None else f"is {effect:.0%} of those units'"
        return False, ("the reworked units' error fell more between the stations than it did for "
                       f"the untouched units that started nearest them (p = {p:.2g}), but their "
                       f"median ratio {size} -- too small a shift to call hand trim (needs "
                       f"{MAX_EFFECT_RATIO:.0%} or less)")
    return True, None


def _sig(x: Optional[float], digits: int = 4) -> Optional[float]:
    """`x` to `digits` significant figures (a p-value can be 1e-15; fixed decimals would zero it,
    and three figures would print 0.01014 as 0.0101 -- no longer visibly over 0.01)."""
    return None if x is None else float(f"{x:.{digits}g}")


def _round(x: Optional[float], places: int = 3) -> Optional[float]:
    return None if x is None else round(x, places)


def _top_third(control: List[_Reading]) -> List[_Reading]:
    """The pass/pass unit-days in the top third of laser error -- the untouched units most like
    the reworked ones (which the laser failed, so theirs ran high)."""
    ranked = sorted(control, key=lambda r: (r.laser, r.ratio, str(r.unit)))
    return ranked[2 * len(ranked) // 3:]


def _shop_order(systems: Iterable[str]) -> Tuple[str, ...]:
    """Laser 1, 2, 3 -- the shop's order, never the code's letters (A is laser TWO)."""
    def rank(s):
        return (LASER_ORDER.index(s) if s in LASER_ORDER else len(LASER_ORDER), s)
    return tuple(sorted({s for s in systems if s}, key=rank))


def _and(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _laser_facts(n_rework: int, rework: List[_Reading], control: List[_Reading]
                 ) -> Tuple[Dict[str, Any], Dict[str, Optional[float]], List[_Reading]]:
    """(one laser's facts, its test, its top third): every number its verdict rests on, whether
    or not the floors are met -- a fact always, a finding only when strong (loss_origin's rule),
    so whoever overturns a threshold has the numbers to do it with."""
    top = _top_third(control)
    test = rank_comparison(rework, top)
    facts = {"rework_unit_days": n_rework, "rework_ratio_n": len(rework), "control_n": len(control),
             "control_top_third_n": len(top),
             "control_top_third_min_laser_error": round(top[0].laser, 4) if top else None,
             "mann_whitney_u": _round(test["u"], 1), "p_value": _sig(test["p"]),
             "median_ratio_rework": _round(test["median_rework"]),
             "median_ratio_control_top_third": _round(test["median_top"]),
             "effect_ratio": _round(test["effect"])}
    confirmed, why_not = verdict(len(rework), len(top), test)
    facts["confirmed"] = confirmed
    if not confirmed:
        facts["note"] = why_not
    return facts, test, top


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
    by_system = agreement.get("overkill_unit_days_by_system") or {}
    facts["linked"] = linked
    facts["rework_unit_days"] = agreement.get("overkill_unit_days") or 0
    facts["by_laser"] = {}
    if not linked:
        facts["confirmed"] = False
        facts["note"] = "no final tests are linked to a trim analysis for this model in the window"
        return facts, findings

    pop = _populations(db, model, cutoff)
    facts.update({"skipped_pairs": pop["skipped"], "junk_readings": pop["junk"],
                  "unpaired_final_tests": pop["unpaired"], "reduction": REDUCTION})
    lasers = _shop_order(set(by_system) | {r.system for r in pop["rework"] + pop["control"]})
    for system in lasers:
        rework = [r for r in pop["rework"] if r.system == system]
        control = [r for r in pop["control"] if r.system == system]
        n_rework = by_system.get(system, 0)
        laser_facts, test, top = _laser_facts(n_rework, rework, control)
        facts["by_laser"][laser_label(system)] = laser_facts
        if laser_facts["confirmed"] and n_rework:
            findings.append(_finding(model, system, n_rework, rework, control, top, test,
                                     laser_facts, pop, laser_label))
    facts["confirmed"] = any(f["confirmed"] for f in facts["by_laser"].values())
    return facts, findings


def _finding(model: str, system: str, n_rework: int, rework, control, top, test,
             laser_facts: Dict[str, Any], pop: Dict[str, Any], laser_label) -> Finding:
    laser = laser_label(system)
    med_rework, med_top, p, effect = test["median_rework"], test["median_top"], test["p"], test["effect"]
    floor = top[0].laser
    comparison = (f"one-sided rank test (Mann-Whitney U, normal approximation with tie correction): "
                  f"the final-test / laser worst-error ratios of the {len(rework):,} reworked "
                  f"unit-days on {laser} against those of the {len(top):,} pass/pass unit-days in "
                  f"the top third of that laser's error ({floor:.4f} V and up), over the travel both "
                  f"stations grade; confirmed at p < {CONFIRM_P:g} with the rework median at "
                  f"{MAX_EFFECT_RATIO:g}x theirs or less")
    return Finding(
        model=model, analyzer="rework_load", category="Rework load",
        lever="laser_settings", systems=(system,),
        title=(f"{laser}: {n_rework:,} unit-days in the last year fail here and pass final test "
               "after rework"),
        summary=(
            f"{n_rework:,} unit-days in the last year failed linearity at {laser} and then passed "
            "final test. That is hand trim, not an unnecessary rejection: their error fell more "
            "between the stations than it did for the untouched units that started nearest them. "
            "Over the travel both stations grade, each corrected with its own offset, final test's "
            f"worst error on {len(rework):,} of these units is a median {med_rework:.0%} of the "
            f"laser's on the same track, against {med_top:.0%} on the {len(top):,} untouched units "
            f"of the same laser (their track passed at both stations) with the largest laser errors "
            f"(the top third, {floor:.4f} V and up); a one-sided rank test puts the difference at "
            f"p = {p:.2g}. No gain is claimed: this counts laser time hand trim is already spending "
            "on this model's own failures."),
        n_units=n_rework,
        strength_name=("reworked units' median final-test/laser error ratio, as a share of the "
                       "untouched units' that started nearest them"),
        strength_value=_round(effect),
        expected_gain_points=None,           # hand-trim labour avoided, never a yield rate
        evidence={"facts": {**laser_facts, "skipped_pairs": pop["skipped"],
                            "reduction": REDUCTION},
                  "comparison": comparison})
