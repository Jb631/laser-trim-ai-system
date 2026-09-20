"""One read of a model's tracks, shaped for the analyzers. No analyzer writes SQL."""
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from .grading import in_limits


@dataclass(frozen=True)
class PassView:
    index: int
    sheet: str
    errors: Tuple
    upper: Tuple
    lower: Tuple
    cut_setting: Optional[float]


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

    @property
    def recipe(self) -> Tuple:
        """(number of cuts, cut-length setting of each cut) -- what the laser was told to do."""
        return (len(self.passes),
                tuple(round(p.cut_setting, 2) if p.cut_setting is not None else None
                      for p in self.passes))


def _arr(js) -> Optional[Tuple]:
    if js is None:
        return None
    v = json.loads(js) if isinstance(js, (str, bytes)) else js
    return tuple(v) if isinstance(v, list) else None


def _date(v) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v).replace("T", " ").split(".")[0])
    except (TypeError, ValueError):
        return None


def _is_real_cut(sheet: Optional[str]) -> bool:
    # Lasers 1 and 3 (LTS, LTS3) write a final `Lin Error` sheet that repeats the
    # last real cut (spec, Known limits). Counting it would add a pass that never ran.
    return not (sheet or "").strip().lower().startswith("lin error")


def load_model_tracks(db, model: str) -> List[TrackView]:
    with db.session() as s:
        pass_rows = s.execute(text(
            "SELECT p.track_result_id, p.pass_index, p.sheet, p.errors, p.upper_limits, "
            "       p.lower_limits, p.laser_cut_length "
            "FROM trim_passes p JOIN track_results t ON t.id = p.track_result_id "
            "JOIN analysis_results a ON a.id = t.analysis_id "
            "WHERE a.model = :m ORDER BY p.track_result_id, p.pass_index"), {"m": model}).fetchall()
        track_rows = s.execute(text(
            "SELECT t.id, a.file_date, a.system, t.untrimmed_errors, t.untrimmed_resistance, "
            "       t.trimmed_resistance, t.error_data, t.upper_limits, t.lower_limits, t.linearity_pass, "
            "       s.initial_resistance_low, s.initial_resistance_high, "
            "       s.final_resistance_low, s.final_resistance_high "
            "FROM track_results t JOIN analysis_results a ON a.id = t.analysis_id "
            "LEFT JOIN trim_setup s ON s.analysis_id = a.id "
            "WHERE a.model = :m AND a.system IN ('A','B','C') "
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
        out.append(TrackView(
            track_id=r[0], file_date=d, system=str(r[2]),
            untrimmed_errors=_arr(r[3]), untrimmed_resistance=r[4], trimmed_resistance=r[5],
            final_errors=_arr(r[6]), final_upper=_arr(r[7]), final_lower=_arr(r[8]),
            linearity_pass=None if r[9] is None else bool(r[9]),
            initial_r_low=r[10], initial_r_high=r[11], final_r_low=r[12], final_r_high=r[13],
            passes=tuple(passes.get(r[0], ()))))
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
