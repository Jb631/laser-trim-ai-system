"""One read of a model's tracks, shaped for the analyzers. No analyzer writes SQL."""
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from .grading import in_limits
from ..core.model_stats import _FAILED_PROCESSING
from ..core.trim_setup import resistance_limits as _track2_resistance_limits

# Status NAMES, as SQLAlchemy stores the enum. The one definition, never re-typed.
_FAILED_SQL = ", ".join(f"'{name}'" for name in _FAILED_PROCESSING)


@dataclass(frozen=True)
class PassView:
    index: int
    sheet: str
    errors: Tuple
    upper: Tuple
    lower: Tuple
    cut_setting: Optional[float]


@dataclass(frozen=True)
class LimitTable:
    """The per-point limits a track was graded against -- the TEST it was held to.

    Two tracks of one model on one laser are only comparable when `key` matches: the fleet survey
    of 2026-09-20 found 17 (model, track, laser) groups graded against two or more tables inside
    one year -- 8232-1 on laser 1 at 89 points AND at 45, 8506A/B with every band loosened 3.75x
    in July 2025. A yield compared across two tables is a comparison of two tests.
    """
    key: str                                   # fingerprint of the exact (upper, lower) content
    rows: int
    graded: int                                # rows carrying a usable limit pair
    band: Tuple[Tuple[float, float], ...]      # (position, half-width) for graded rows, by position


def limit_table_of(positions, upper, lower) -> Optional[LimitTable]:
    """The table behind one sweep's limits, or None when there is no usable one."""
    if not upper or not lower or len(upper) != len(lower):
        return None
    pts = [(None if not _real(u) else round(u, 5), None if not _real(l) else round(l, 5))
           for u, l in zip(upper, lower)]
    graded = [i for i, (u, l) in enumerate(pts) if u is not None and l is not None and u > l]
    if len(graded) < 3:
        return None
    band = []
    if positions and len(positions) == len(pts):
        band = sorted((round(positions[i], 3), round((pts[i][0] - pts[i][1]) / 2.0, 5))
                      for i in graded if _real(positions[i]))
    return LimitTable(key=hashlib.md5(json.dumps(pts).encode()).hexdigest()[:10],
                      rows=len(pts), graded=len(graded), band=tuple(band))


def _real(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x


def tables_of(tracks) -> "Counter[str]":
    """How many of `tracks` were graded against each limit table -- the TEST (key -> count);
    a track with no usable table is left out. A pass rate is a verdict against ONE test
    (CLAUDE.md), so every analyzer that compares verdicts reads its tables here: setup_change
    needs exactly one on each side of a change, loss_origin scores a laser on its busiest."""
    return Counter(t.limit_table.key for t in tracks if t.limit_table is not None)


def _is_blank_template(errors) -> bool:
    """A final sweep whose every reading is EXACTLY 0.0 is not a measurement.

    Laser 1 writes a template `Lin Error` sheet (measured == theory) when no cut is made. (Laser 3
    writes laser 2's sheets, not laser 1's, so it has no `Lin Error` sheet at all -- and none of the
    1,182 affected tracks is laser 3.)
    The parser stopped taking it for a final sweep on 2026-09-20, but a database ingested before
    that still holds 1,182 such tracks as flawless linearity PASSes -- so the loader refuses them
    too. No real sweep has zero noise at every point.
    """
    real = [e for e in (errors or ()) if _real(e)]
    return len(real) >= 10 and all(e == 0.0 for e in real)


@dataclass(frozen=True)
class TrackView:
    track_id: int
    file_date: datetime
    system: str                       # the code's letter: A / B / C
    untrimmed_errors: Optional[Tuple]
    untrimmed_resistance: Optional[float]
    trimmed_resistance: Optional[float]
    final_errors: Optional[Tuple]
    final_upper: Optional[Tuple]
    final_lower: Optional[Tuple]
    linearity_pass: Optional[bool]    # the app's stored verdict
    initial_r_low: Optional[float]
    initial_r_high: Optional[float]
    final_r_low: Optional[float]
    final_r_high: Optional[float]
    passes: Tuple[PassView, ...]      # REAL cuts only, in order
    track_name: str = "default"       # "Track A", "TRK1" ... a model's tracks may carry different limits
    final_positions: Optional[Tuple] = None
    # The laser settings beyond the cut recipe (trim_setup.parameters), for findings/analyzers/
    # setup_change.py. Track 1's own block, except a TRK2 track whose file captured a Track 2
    # block (Task 9): {**parameters, **track2_parameters} then -- see load_model_tracks below.
    setup: Optional[Dict[str, Any]] = None
    # Where `setup` came from (final review, 2026-09-25, I2): True for a TRK2 track whose file
    # captured NO Track 2 block, so `setup` is Track 1's block, not this track's. Laser 2's
    # 'Track Parameters' sheet holds every laser setting per track (Model Parameters holds only
    # identity, counts and axis limits), so an inherited block says nothing about Track 2's own
    # settings -- and the day the capture starts, Track 1's values give way to Track 2's, which
    # must never read as a change. Every other track's block is its own.
    setup_inherited: bool = False

    @cached_property
    def limit_table(self) -> Optional[LimitTable]:
        return limit_table_of(self.final_positions, self.final_upper, self.final_lower)

    @property
    def recipe(self) -> Tuple:
        """(number of cuts, cut-length setting of each cut) -- what the laser was told to do."""
        return (len(self.passes),
                tuple(round(p.cut_setting, 2) if p.cut_setting is not None else None
                      for p in self.passes))


def _arr(js) -> Optional[Tuple]:
    if js is None:
        return None
    try:
        v = json.loads(js) if isinstance(js, (str, bytes)) else js
    except ValueError:          # JSONDecodeError and UnicodeDecodeError both: a corrupt array is an
        return None             # ungradeable TRACK -- it must not abort the whole model's load
    return tuple(v) if isinstance(v, list) else None


def _dict(js) -> Optional[Dict[str, Any]]:
    """Same job as `_arr`, for a JSON OBJECT column (`trim_setup.track2_parameters`)."""
    if js is None:
        return None
    try:
        v = json.loads(js) if isinstance(js, (str, bytes)) else js
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def _date(v) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v).replace("T", " ").split(".")[0])
    except (TypeError, ValueError):
        return None


def _is_real_cut(sheet: Optional[str]) -> bool:
    # Laser 1 (LTS) writes a final `Lin Error` sheet that repeats the
    # last real cut (spec, Known limits). Counting it would add a pass that never ran.
    return not (sheet or "").strip().lower().startswith("lin error")


def load_model_tracks(db, model: str) -> List[TrackView]:
    with db.session() as s:
        pass_rows = s.execute(text(
            "SELECT p.track_result_id, p.pass_index, p.sheet, p.errors, p.upper_limits, "
            "       p.lower_limits, p.laser_cut_length "
            "FROM trim_passes p JOIN track_results t ON t.id = p.track_result_id "
            "JOIN analysis_results a ON a.id = t.analysis_id "
            "WHERE a.model = :m AND a.system IN ('A','B','C') "
            f"AND t.status NOT IN ({_FAILED_SQL}) "
            "ORDER BY p.track_result_id, p.pass_index"), {"m": model}).fetchall()
        track_rows = s.execute(text(
            "SELECT t.id, a.file_date, a.system, t.untrimmed_errors, t.untrimmed_resistance, "
            "       t.trimmed_resistance, t.error_data, t.upper_limits, t.lower_limits, t.linearity_pass, "
            "       s.initial_resistance_low, s.initial_resistance_high, "
            "       s.final_resistance_low, s.final_resistance_high, t.track_id, t.position_data, "
            "       s.track2_parameters, s.parameters "
            "FROM track_results t JOIN analysis_results a ON a.id = t.analysis_id "
            "LEFT JOIN trim_setup s ON s.analysis_id = a.id "
            "WHERE a.model = :m AND a.system IN ('A','B','C') "
            f"AND t.status NOT IN ({_FAILED_SQL}) "
            "ORDER BY a.file_date, t.id"), {"m": model}).fetchall()
    passes: Dict[int, List[PassView]] = {}
    for tid, idx, sheet, err, up, lo, cut in pass_rows:
        if not _is_real_cut(sheet):
            continue
        e, u, l = _arr(err), _arr(up), _arr(lo)
        if e is None or u is None or l is None:
            continue
        passes.setdefault(tid, []).append(PassView(idx, sheet or "", e, u, l, cut))
    out: List[TrackView] = []
    for r in track_rows:
        d = _date(r[1])
        if d is None:
            continue
        final_errors = _arr(r[6])
        verdict = None if r[9] is None else bool(r[9])
        if _is_blank_template(final_errors):
            final_errors, verdict = None, None        # the limits stay: they ARE the table in service
        track_name = str(r[14]) if r[14] else "default"
        initial_r_low, initial_r_high, final_r_low, final_r_high = r[10], r[11], r[12], r[13]
        # A TRK2 track is judged against ITS OWN resistance limits when the
        # setup row captured Track 2's block (two-track System A files only;
        # see TrimSetup.track2_parameters) -- as PAIRS: the initial window is
        # replaced only when the block gives BOTH its limits, and the same for
        # the final window. A block with one side of a window leaves that window
        # exactly as the file has it: overriding per field paired Track 2's low
        # limit with Track 1's high one (2026-09-24 review), a window neither
        # track was graded against, and one side alone could even invert it
        # (low above high). Absent or incomplete, r[10:14] (Track 1's/the
        # file's) are used, unchanged from before this feature existed.
        t2_params = _dict(r[16]) if track_name == "TRK2" else None
        if t2_params is not None:
            t2 = _track2_resistance_limits(t2_params)
            if (t2["initial_resistance_low"] is not None
                    and t2["initial_resistance_high"] is not None):
                initial_r_low = t2["initial_resistance_low"]
                initial_r_high = t2["initial_resistance_high"]
            if (t2["final_resistance_low"] is not None
                    and t2["final_resistance_high"] is not None):
                final_r_low = t2["final_resistance_low"]
                final_r_high = t2["final_resistance_high"]
        # setup (findings/analyzers/setup_change.py): Track 1's own block, except a TRK2 track
        # whose file captured a Track 2 block, which gets that block's values layered on top --
        # the same "captured overrides, else Track 1's" rule the resistance limits above follow.
        params = _dict(r[17])
        setup = {**(params or {}), **t2_params} if t2_params else params
        # ...and says which: a TRK2 track with no captured Track 2 block is reading Track 1's.
        setup_inherited = track_name == "TRK2" and not t2_params
        out.append(TrackView(
            track_id=r[0], file_date=d, system=str(r[2]),
            untrimmed_errors=_arr(r[3]), untrimmed_resistance=r[4], trimmed_resistance=r[5],
            final_errors=final_errors, final_upper=_arr(r[7]), final_lower=_arr(r[8]),
            linearity_pass=verdict,
            initial_r_low=initial_r_low, initial_r_high=initial_r_high,
            final_r_low=final_r_low, final_r_high=final_r_high,
            passes=tuple(passes.get(r[0], ())),
            track_name=track_name, final_positions=_arr(r[15]), setup=setup,
            setup_inherited=setup_inherited))
    return out


# The yardstick is only trusted on a model where it reproduces the app's own stored verdict.
# The bar travels WITH the result, so a screen can say why a model is not graded without
# keeping a second copy of these numbers.
YARDSTICK_MIN_N = 30
YARDSTICK_MIN_AGREEMENT = 0.99


def yardstick_fidelity(tracks: List[TrackView]) -> Dict[str, Any]:
    """Does margin_ratio reproduce the app's stored verdict on THIS model's final sweeps?"""
    same = n = 0
    for t in tracks:
        if t.linearity_pass is None or not t.final_errors:
            continue
        g = in_limits(t.final_errors, t.final_upper, t.final_lower)
        if g is None:
            continue
        n += 1
        same += (g == t.linearity_pass)
    return {"n": n, "agreement": (same / n) if n else None,
            "faithful": bool(n >= YARDSTICK_MIN_N and same / n >= YARDSTICK_MIN_AGREEMENT),
            "min_n": YARDSTICK_MIN_N, "min_agreement": YARDSTICK_MIN_AGREEMENT}
