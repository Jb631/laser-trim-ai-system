"""Trim vs Final-Test overlay: the linkage, the pairing, and the FT trace.

The overlay is the last feature V5's Compare page had that V6 did not
(gui/pages/compare.py, ~:798-946). The SEMANTICS live here so the V6 unit
chart and its print document can draw it without either copying grading math
into a widget or reaching back into a V5 page.

Three things this module exists to get right:

1. LINKAGE. `final_test_results.linked_trim_id -> analysis_results.id`, and
   only at `match_confidence >= 0.70` — the same threshold every corrected
   trim/FT number in the app uses (see DatabaseManager.get_escape_overkill_
   analysis and friends). Below it, the two records are not known to be the
   same physical unit, so overlaying them would be a guess drawn as a fact.

2. MULTIPLICITY. A unit can be final-tested more than once (valid data, like
   a re-trim). The overlay shows the NEWEST linked test and says which of how
   many it is, rather than silently picking one.

3. CORRECTION. An FT sweep carries its OWN offset, slope, theory column and
   spec limits, because the FT station makes its own adjustment. Grading it
   with the TRIM correction would draw a curve no instrument ever produced.
   The adjustment itself is the shared `corrected_errors` — the one
   definition of a graded trace in this codebase — never a local copy.

V5's Compare page keeps its own copy of the pairing helpers on purpose: it is
the frozen fallback UI, and this work does not touch it.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

from laser_trim_analyzer.export.unit_chart import corrected_errors

# The confidence floor for treating a trim record and a final-test record as
# the same unit. Matches the app-wide default (manager.py `min_confidence`).
MIN_MATCH_CONFIDENCE = 0.70

# How far apart two spans must be before they count as different UNITS rather
# than different sweep extents. A ratio, not a percentage, and deliberately
# loose: found on real data (analysis_id 104050) that an FT sweep covering
# ±14° of a ±20° trim travel is a genuinely SHORTER sweep, and stretching it
# to fill the trim axis moves every FT error to a position it was not measured
# at. Only an order-of-magnitude gap means "different column entirely" — e.g.
# analysis_id 106025, whose FT position column spans 0.07 against the trim's
# 340 degrees.
_SCALE_RATIO = 5.0


def normalize_track_id(track_id) -> str:
    """One comparable letter per track designator, across the conventions:
    'Track A' / 'TRK1' / 'A' / 'default'. Mirrors the pairing V5's Compare
    page does (compare._normalize_track_id)."""
    if track_id is None:
        return ""
    s = str(track_id).strip().upper()
    if not s:
        return ""
    if s.startswith("TRACK "):
        return s[6:].strip()
    return {"TRK1": "A", "TRK2": "B", "TRK3": "C"}.get(s, s)


def pair_ft_track(trim_track_id, ft_tracks: Sequence[Dict]) -> Optional[Dict]:
    """The FT track that belongs with this trim track, by designator.

    Pairing by index mismatches a Track-B trim against FT track A on every
    multi-track unit — the exact bug V5's _pair_tracks was written to fix.
    Legacy single-track data ('default' / '') pairs with the first track.
    """
    if not ft_tracks:
        return None
    want = normalize_track_id(trim_track_id)
    if not want or want == "DEFAULT":
        return ft_tracks[0]
    for ft in ft_tracks:
        if normalize_track_id(ft.get("track_id")) == want:
            return ft
    for ft in ft_tracks:
        if normalize_track_id(ft.get("track_id")) == "DEFAULT":
            return ft
    return ft_tracks[0]


def align_positions(ft_positions: Sequence[float],
                    trim_positions: Sequence[float]) -> Tuple[List[float], bool]:
    """(positions_to_draw, was_rescaled).

    Left ALONE whenever the two sweeps are on the same kind of scale — even a
    much shorter FT sweep, because covering less travel is a FACT about the
    sweep, and stretching it to fill the trim axis would move every FT error
    to a position it was never measured at. Only an order-of-magnitude gap
    (`_SCALE_RATIO`) means the FT column is in different units altogether;
    then it is mapped onto the trim span, and the second return value exists
    so the caller can SAY so rather than pass a transformed axis off as
    measured.
    """
    ft = [float(p) for p in (ft_positions or []) if p is not None]
    trim = [float(p) for p in (trim_positions or []) if p is not None]
    if len(ft) < 2 or len(trim) < 2:
        return list(ft), False
    f_lo, f_hi = min(ft), max(ft)
    t_lo, t_hi = min(trim), max(trim)
    f_span, t_span = f_hi - f_lo, t_hi - t_lo
    if f_span <= 0 or t_span <= 0:
        return list(ft), False
    ratio = max(f_span, t_span) / min(f_span, t_span)
    if ratio < _SCALE_RATIO:
        return list(ft), False
    scale = t_span / f_span
    return [t_lo + (p - f_lo) * scale for p in ft], True


# Two sweeps whose spans agree to within this share cover the same travel ...
_SAME_SPAN = 0.25
# ... and if their centres then sit further apart than this share of the trim's span, the
# final-test column counts that travel from a different zero.
_OTHER_ZERO = 0.10


def _position(p) -> Optional[float]:
    """A usable position, or None (missing, NaN, infinite, not a number)."""
    if p is None or isinstance(p, bool):
        return None
    try:
        v = float(p)
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def positions_on_trim_axis(ft_positions: Sequence[Any], trim_positions: Sequence[Any]
                           ) -> Tuple[Optional[List[Optional[float]]], bool]:
    """The FT sweep's positions placed on the TRIM sweep's axis, index for index, for comparing
    the two stations POSITION BY POSITION -- and whether they had to be shifted.

    `align_positions` is for a picture whose axis is labelled. A point-by-point comparison
    (findings/analyzers/rework_load: "the travel both stations grade") needs every FT point where
    it physically is, so this differs from it in three ways, each found on real data:

    * the same span counted from a DIFFERENT ZERO is shifted so the two centres coincide.
      8340-1's final test counts 0 to 0.61 against the trim's -0.305 to 0.305; some 8397-2
      files count 0.05 to 240.05 against +/-120. Compared unshifted, only half the travel
      overlaps -- and the wrong half is compared with the wrong half.
    * spans an order of magnitude apart (`_SCALE_RATIO`) give None, never a rescaled axis:
      a measurement is not moved onto a guessed axis. 25 real 8397-2 files hold one stray value
      and then zeros in their position column (a span of 0.046 against the trim's 240).
    * a missing position stays None IN PLACE. The FT arrays are index-aligned -- errors, limits,
      the graded window -- and dropping a None would slide every later point onto its
      neighbour's error.

    Everything else is left exactly where it was measured, including a genuinely SHORTER FT
    sweep centred on the same zero (6607: +/-14 of the trim's +/-22), for align_positions' own
    reason. None also when either side has fewer than two positions or no span.
    """
    ft_all = [_position(p) for p in (ft_positions or [])]
    ft = [p for p in ft_all if p is not None]
    trim = [p for p in (_position(p) for p in (trim_positions or [])) if p is not None]
    if len(ft) < 2 or len(trim) < 2:
        return None, False
    f_lo, f_hi = min(ft), max(ft)
    t_lo, t_hi = min(trim), max(trim)
    f_span, t_span = f_hi - f_lo, t_hi - t_lo
    if f_span <= 0 or t_span <= 0:
        return None, False
    if max(f_span, t_span) / min(f_span, t_span) >= _SCALE_RATIO:
        return None, False
    shift = (t_lo + t_hi) / 2.0 - (f_lo + f_hi) / 2.0
    if abs(f_span / t_span - 1.0) <= _SAME_SPAN and abs(shift) > _OTHER_ZERO * t_span:
        return [None if p is None else p + shift for p in ft_all], True
    return ft_all, False


def is_sweep_axis(positions: Sequence[float], min_monotone: float = 0.95) -> bool:
    """Is this column actually a swept position, i.e. monotonic?

    Some final-test files store something else entirely in position_data — the
    6828 family keeps an error-like column there, oscillating across a span of
    0.03 against a trim travel of 340 degrees. Mapped onto the trim axis it
    draws a smooth diagonal that LOOKS like a measurement and is not one, so
    the overlay refuses it instead. Measured on the work DB: 31 of 3,000
    linked FT sweeps (1%) fail this; the other 99% are clean.

    A small amount of readback jitter is tolerated — the test is whether the
    sweep overwhelmingly moves one way, not whether it is perfectly sorted.
    """
    pts = [float(p) for p in (positions or []) if p is not None]
    if len(pts) < 3:
        return False
    steps = [b - a for a, b in zip(pts, pts[1:]) if b != a]
    if not steps:
        return False
    forward = sum(1 for s in steps if s > 0)
    return max(forward, len(steps) - forward) / len(steps) >= min_monotone


def _unavailable(reason: str) -> Dict[str, Any]:
    return {"available": False, "reason": reason}


def resolve_linked_ft(db, analysis_id: int,
                      min_confidence: float = MIN_MATCH_CONFIDENCE) -> Dict[str, Any]:
    """Which final-test record (if any) belongs to this trim analysis.

    Returns {"available": True, "ft_id", "n_links", "confidence", "date",
    "filename", "result"} for the NEWEST qualifying link, or an unavailable
    payload naming the reason in plain words.
    """
    from laser_trim_analyzer.database.models import FinalTestResult as DBFT

    with db.session() as s:
        rows = (s.query(DBFT.id, DBFT.filename, DBFT.match_confidence,
                        DBFT.test_date, DBFT.file_date, DBFT.overall_status,
                        DBFT.station_linearity_pass)
                .filter(DBFT.linked_trim_id == analysis_id)
                .all())
    if not rows:
        return _unavailable("No final test is linked to this unit.")

    good = [r for r in rows
            if (r[2] if r[2] is not None else 0.0) >= min_confidence]
    if not good:
        best = max((r[2] or 0.0) for r in rows)
        return _unavailable(
            f"The linked final test is a low-confidence match "
            f"({best:.2f} < {min_confidence:.2f}) — not shown.")

    # Newest by the actual test moment, falling back to the filename date.
    def _when(r):
        return r[3] or r[4]

    good.sort(key=lambda r: (_when(r) is not None, _when(r)))
    newest = good[-1]
    when = _when(newest)
    status = newest[5]
    return {
        "available": True,
        "ft_id": newest[0],
        "filename": newest[1],
        "confidence": newest[2],
        "date": str(when).split(" ")[0] if when else "",
        "n_links": len(good),
        "result": getattr(status, "name", str(status) if status else ""),
        # The SHEET's own PASSED/FAILED, for reference beside the app's grade.
        # They can legitimately differ: the app corrects the offset, so a unit
        # the station rejected can grade in spec (7539-2 sn23 is the worked
        # example). That is information, not an error, and the chart says both
        # rather than picking one.
        "station_pass": newest[6],
    }


def load_ft_overlay(db, analysis_id: int, trim_track_id=None,
                    trim_positions: Optional[Sequence[float]] = None,
                    min_confidence: float = MIN_MATCH_CONFIDENCE) -> Dict[str, Any]:
    """Everything the chart needs to draw the FT trace, or a plain reason why
    it cannot — never an empty overlay the reader has to interpret.

    On success: positions (aligned to the trim sweep when the scales differ,
    with `rescaled` saying so), raw `errors`, `corrected` (the FT's OWN
    adjustment), the FT's OWN `upper_limits`/`lower_limits`, and a `label`
    naming the test date and, when the unit was tested more than once, which
    of how many this is.
    """
    link = resolve_linked_ft(db, analysis_id, min_confidence)
    if not link.get("available"):
        return link

    from laser_trim_analyzer.database.models import FinalTestTrack as DBFTT

    with db.session() as s:
        tracks = [{
            "track_id": t.track_id,
            "position_data": list(t.position_data or [])
                             or list(t.electrical_angle_data or []),
            "error_data": list(t.error_data or []),
            "upper_limits": list(t.upper_limits or []),
            "lower_limits": list(t.lower_limits or []),
            "theory_data": list(t.theory_data or []),
            "optimal_offset": t.optimal_offset,
            "optimal_slope": t.optimal_slope,
            "linearity_type": t.linearity_type,
            "linearity_pass": t.linearity_pass,
            "linearity_error": t.linearity_error,
            "linearity_spec": t.linearity_spec,
            "linearity_fail_points": t.linearity_fail_points,
            "station_flags": list(t.station_flags or []),
            "station_fail_points": t.station_fail_points,
            "graded_start": t.graded_start,
            "graded_end": t.graded_end,
        } for t in (s.query(DBFTT)
                    .filter(DBFTT.final_test_id == link["ft_id"])
                    .order_by(DBFTT.track_id).all())]

    track = pair_ft_track(trim_track_id, tracks)
    if track is None or not track["position_data"] or not track["error_data"]:
        return _unavailable(
            f"The linked final test ({link['date'] or 'no date'}) stored no "
            "point-by-point sweep — only its pass/fail result.")

    if not is_sweep_axis(track["position_data"]):
        return _unavailable(
            f"The linked final test ({link['date'] or 'no date'}) stored no "
            "usable position column — its values do not sweep, so there is no "
            "honest way to place the trace against the trim travel.")

    offset = float(track["optimal_offset"] or 0.0)
    k = float(track["optimal_slope"] or 0.0)
    positions, rescaled = align_positions(track["position_data"], trim_positions or [])

    label = f"Final test {link['date']}".strip()
    if link["n_links"] > 1:
        label += f" (newest of {link['n_links']})"
    if rescaled:
        label += " · x aligned to trim travel"

    out = dict(link)
    out.update({
        "track_id": track["track_id"],
        "positions": positions,
        "rescaled": rescaled,
        "errors": track["error_data"],
        # The FT station's own adjustment, through the ONE definition of a
        # graded trace. The trim's offset/k never touch this curve.
        "corrected": corrected_errors(track["error_data"], offset, k,
                                      track["theory_data"] or None),
        "upper_limits": track["upper_limits"],
        "lower_limits": track["lower_limits"],
        "offset": offset,
        "k": k,
        "linearity_type": track["linearity_type"],
        "linearity_pass": track["linearity_pass"],
        "linearity_fail_points": track["linearity_fail_points"],
        "linearity_error": track["linearity_error"],
        "label": label,
        # The window the STATION graded, as index bounds into `positions`, and
        # the points IT flagged. The renderer shades outside the window and
        # marks the flagged points; nothing here re-grades anything.
        "graded_start": track["graded_start"],
        "graded_end": track["graded_end"],
        "station_flags": track["station_flags"],
        "station_fail_points": track["station_fail_points"],
        "ungraded_indices": sorted(ungraded_indices(
            len(track["error_data"]), track["graded_start"], track["graded_end"])),
    })
    out["legend"] = ft_legend(out)
    return out


def ungraded_indices(n_points: int, graded_start, graded_end) -> set:
    """Indices outside the station's graded window. Empty when it stated none.

    Lives here rather than in the renderer because the print export and the
    on-screen modal must shade the same rows.
    """
    if graded_start is None and graded_end is None:
        return set()
    low = int(graded_start or 0)
    high = int(graded_end if graded_end is not None else n_points - 1)
    return set(range(0, max(0, low))) | set(range(min(n_points, high + 1), n_points))


def ft_legend(overlay: Dict[str, Any]) -> str:
    """One line naming BOTH verdicts, and saying so when they differ.

    The app's grade is the disposition and is stated first, with the number of
    points it failed. The station's is the reference. They can legitimately
    disagree — the app corrects the offset, so a sweep the station rejected
    can grade in spec (7539-2 sn23 is the worked example) — so a difference is
    NAMED and explained rather than hidden or dressed up as an error.
    """
    app_pass = overlay.get("linearity_pass")
    app_text = ("PASS" if app_pass else
                ("FAIL" if app_pass is False else "not graded"))
    fail_points = overlay.get("linearity_fail_points")
    if app_pass is False and fail_points:
        app_text += f" ({fail_points} point{'s' if fail_points != 1 else ''})"
    parts = [f"Final test: {app_text}"]

    station = overlay.get("station_pass")
    if station is not None:
        clause = f"station: {'PASSED' if station else 'FAILED'}"
        flagged = overlay.get("station_fail_points")
        if flagged:
            clause += f" ({flagged} flagged)"
        parts.append(clause)
        if app_pass is not None and bool(app_pass) != bool(station):
            parts.append("app grade differs — offset corrected")

    ungraded = overlay.get("ungraded_indices") or []
    if ungraded:
        parts.append(f"{len(ungraded)} point(s) the station did not grade")
    return " · ".join(parts)
