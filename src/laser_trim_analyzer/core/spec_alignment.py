"""Do the trim station and the final test station grade the part the same way?

James, 2026-08-30: "i also want to know when the trim and test specs dont
align." Every cross-station number the app prints — escapes, overkills, the
Gap, trim-vs-FT agreement — silently assumes the two stations are holding the
unit to the SAME requirement. On a lot of models they are not: trim grades to
a hairline linearity band while final test allows an order of magnitude more.
When that is true, an "escape" is not a missed defect, it is two different
questions being subtracted from each other.

This module answers only that one question, in one sentence, so the person
reading those numbers knows what they are looking at. It is VISIBILITY, never
a disposition: nothing here re-grades a unit, changes a verdict, or picks a
"correct" spec. Spec governance is a separate, parked decision.

THE METHOD IS THE CENSUS'S — see `scripts/data_trust_census.py` (class 9) and
James's correction in commit 4c6ebd8. Limits are compared ONLY at
near-coincident positions:

  * The two stations sample different position tables, so comparing scalar
    spec summaries (or interpolating one station's limits onto the other's
    positions) manufactures disagreements that do not exist. The naive first
    cut called 81 models mismatched; the position-matched method said 18.
  * Bowtie specs step at a knee. Interpolating ACROSS that knee invents a
    limit half-way between the two sides that neither station ever applied.
    So a point only counts when the other station measured within half the
    finer station's own spacing — otherwise it is skipped, not guessed.

Pure of Tk and safe on a worker thread; the only I/O is the two small sampling
queries, and a module-level cache keeps repeat calls (the FOCUS list asks per
row, the Model page asks per load) off the database.
"""
import logging
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Below this many position matches the two arrays have not really been
# compared — a handful of coincidences is not evidence either way.
MIN_MATCHED = 20
# "Differs" means differing is the RULE, not an edge: more than half the
# matched positions disagree. Deliberately stricter than the census's 10%
# reporting threshold — the census is a measurement pass looking for anything
# worth investigating, this drives a warning banner a user sees every day, and
# a banner that cries wolf on a handful of knee points gets ignored.
DIFFER_SHARE = 0.5
# A limit "differs" when either bound is off by more than a fifth of the trim
# band's own width — the census's tolerance, so both surfaces agree on what a
# difference IS. Relative, because a ±0.03 V band and a ±3 V band cannot share
# an absolute epsilon.
BAND_TOL = 0.2

_Point = Tuple[float, float, float]          # (position, upper, lower)

# Keyed by (model, sample size). The specs in question are model-level design
# data that changes when a drawing changes, and the app is restarted daily, so
# a plain dict with no TTL is the right cache. The sample size is in the key
# only so a caller asking for a deeper sample cannot be handed a shallower
# answer computed earlier.
_CACHE: Dict[Tuple[str, int], "SpecComparison"] = {}


@dataclass(frozen=True)
class SpecComparison:
    """What the two stations require of the same part, at the same positions."""
    status: str                       # "differs" | "aligned" | "insufficient"
    pct_positions_differing: float    # 0..1; meaningless when "insufficient"
    matched_positions: int
    trim_typ_band: Optional[float]    # typical (median) HALF-band at matched pts
    ft_typ_band: Optional[float]
    note: str                         # one plain-language sentence for the UI


def clear_spec_alignment_cache() -> None:
    """Drop the memoized comparisons (tests, and a re-ingest of new files)."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# The comparison itself — pure, no database.
# ---------------------------------------------------------------------------

def _points(pos, upper, lower) -> List[_Point]:
    """Zip one track's arrays into sorted (position, upper, lower) triples.

    Rows are stored as SafeJSON, which hands back a list — and maps a stored
    NULL to `[]`, so an "empty" array is a normal, expected shape here, not an
    error. Any position with a missing bound is dropped: a half-known limit
    cannot be compared to anything.
    """
    if not pos or not upper or not lower:
        return []
    out = [(float(x), float(u), float(l))
           for x, u, l in zip(pos, upper, lower)
           if x is not None and u is not None and l is not None]
    out.sort()
    return out


def _spacing(points: Sequence[_Point]) -> float:
    """The station's typical step between positions (median positive gap)."""
    gaps = sorted(points[i + 1][0] - points[i][0]
                  for i in range(len(points) - 1)
                  if points[i + 1][0] > points[i][0])
    return gaps[len(gaps) // 2] if gaps else 1.0


def compare_arrays(pairs: Sequence[Tuple[Sequence[_Point], Sequence[_Point]]]
                   ) -> Tuple[int, int, List[float], List[float]]:
    """Position-matched limit comparison — the census's walk, extracted.

    Takes (trim points, ft points) PAIRS — normally the same unit's two
    stations — and returns (matched, differing, trim half-bands, ft half-bands)
    accumulated over them. The `j` cursor only ever moves forward because both
    sides are sorted, which is what keeps this linear instead of quadratic.
    """
    matched = differing = 0
    trim_bands: List[float] = []
    ft_bands: List[float] = []
    for T, F in pairs:
        if len(T) < 3 or len(F) < 3:
            continue
        # Half the FINER station's spacing: a match has to be a genuine
        # coincidence of positions, not the nearest thing within a coarse step.
        tol = min(_spacing(T), _spacing(F)) / 2.0
        j = 0
        for x, u, l in T:
            while j < len(F) - 1 and F[j][0] < x - tol:
                j += 1
            if abs(F[j][0] - x) > tol:
                continue                  # no FT station here — skip, never guess
            matched += 1
            width = max(u - l, 1e-9)
            trim_bands.append((u - l) / 2.0)
            ft_bands.append((F[j][1] - F[j][2]) / 2.0)
            if max(abs(u - F[j][1]), abs(l - F[j][2])) > BAND_TOL * width:
                differing += 1
    return matched, differing, trim_bands, ft_bands


def _fmt(v: float) -> str:
    """A limit as a person reads it. Volts, three decimals where that means
    something (these bands live between 0.005 V and 1 V)."""
    return f"{v:.3f}" if abs(v) >= 5e-4 else f"{v:.2g}"


def _band_text(bands: Sequence[float]) -> str:
    """"±0.030 V", or "±0.050–0.200 V" when the spec is a bowtie."""
    lo, hi = min(bands), max(bands)
    # Within 2% counts as one band: float noise across a sweep is not a bowtie.
    if hi - lo <= max(0.02 * hi, 1e-9):
        return f"±{_fmt(median(bands))} V"
    return f"±{_fmt(lo)}–{_fmt(hi)} V"


# ---------------------------------------------------------------------------
# DB layer — sampling the newest stored limit arrays on each side.
# ---------------------------------------------------------------------------

def _linked_pairs(db, model: str, limit: int) -> List[Tuple[List[_Point],
                                                            List[_Point]]]:
    """The census's population: the SAME unit's trim track and FT track.

    This is the pairing that makes the comparison mean something. Sampling the
    two stations independently looked equivalent — they are model-level specs,
    after all — but on the real database it is not: 6126's newest FT records
    sweep a different position table than its newest trim records, so an
    arbitrary pairing matched a fifth as many positions and read "aligned"
    where the census's linked pairs say 100% of matched positions differ.
    Compare a unit against ITSELF, exactly like `data_trust_census.py` does.
    """
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR
    from laser_trim_analyzer.database.models import FinalTestResult as DBFT
    from laser_trim_analyzer.database.models import FinalTestTrack as DBFTT
    from laser_trim_analyzer.database.models import TrackResult as DBTR
    with db.session() as s:
        rows = (s.query(DBTR.position_data, DBTR.upper_limits,
                        DBTR.lower_limits, DBFTT.position_data,
                        DBFTT.upper_limits, DBFTT.lower_limits)
                .select_from(DBFT)
                .join(DBAR, DBAR.id == DBFT.linked_trim_id)
                .join(DBTR, DBTR.analysis_id == DBAR.id)
                .join(DBFTT, DBFTT.final_test_id == DBFT.id)
                .filter(DBFT.model == model,
                        DBTR.position_data.isnot(None),
                        DBTR.upper_limits.isnot(None),
                        DBTR.lower_limits.isnot(None),
                        DBFTT.position_data.isnot(None),
                        DBFTT.upper_limits.isnot(None),
                        DBFTT.lower_limits.isnot(None))
                .order_by(DBFT.id.desc()).limit(limit).all())
    return [(_points(*r[:3]), _points(*r[3:])) for r in rows]


def _trim_arrays(db, model: str, limit: int) -> List[List[_Point]]:
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR
    from laser_trim_analyzer.database.models import TrackResult as DBTR
    with db.session() as s:
        rows = (s.query(DBTR.position_data, DBTR.upper_limits, DBTR.lower_limits)
                .join(DBAR, DBAR.id == DBTR.analysis_id)
                .filter(DBAR.model == model,
                        DBTR.position_data.isnot(None),
                        DBTR.upper_limits.isnot(None),
                        DBTR.lower_limits.isnot(None))
                .order_by(DBTR.id.desc()).limit(limit).all())
    return [_points(*r) for r in rows]


def _ft_arrays(db, model: str, limit: int) -> List[List[_Point]]:
    from laser_trim_analyzer.database.models import FinalTestResult as DBFT
    from laser_trim_analyzer.database.models import FinalTestTrack as DBFTT
    with db.session() as s:
        rows = (s.query(DBFTT.position_data, DBFTT.upper_limits,
                        DBFTT.lower_limits)
                .join(DBFT, DBFT.id == DBFTT.final_test_id)
                .filter(DBFT.model == model,
                        DBFTT.position_data.isnot(None),
                        DBFTT.upper_limits.isnot(None),
                        DBFTT.lower_limits.isnot(None))
                .order_by(DBFTT.id.desc()).limit(limit).all())
    return [_points(*r) for r in rows]


_INSUFFICIENT_NO_ARRAYS = SpecComparison(
    status="insufficient", pct_positions_differing=0.0, matched_positions=0,
    trim_typ_band=None, ft_typ_band=None,
    note="no stored spec limits on both stations — nothing to compare")


def compare_station_specs(db, model: str, *,
                          sample_per_side: int = 5) -> SpecComparison:
    """Are this model's trim and final-test limits the same requirement?

    Samples the newest `sample_per_side` LINKED trim/FT pairs — the same
    unit's two stations, the census's population — and compares their limits
    position by position. When the model has no linked pairs at all (older
    data, or FT files that never matched a trim run) it falls back to pairing
    the newest tracks of each station in order: less trustworthy, because the
    two sides may sweep different position tables, but better than refusing to
    answer for a model whose links were never built.

    Never raises. A read that fails degrades to "insufficient" — an unanswered
    question — because the callers are a warning banner and a list row, and
    neither may take a page down over a hint.
    """
    key = (model, sample_per_side)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    try:
        pairs = [(t, f) for t, f in _linked_pairs(db, model, sample_per_side)
                 if t and f]
        if not pairs:
            trim = [t for t in _trim_arrays(db, model, sample_per_side) if t]
            ft = [t for t in _ft_arrays(db, model, sample_per_side) if t]
            pairs = list(zip(trim, ft))
    except Exception:
        # Not cached: a locked/old database is a transient condition, and
        # caching "insufficient" would keep the banner silent for the session.
        logger.exception("spec alignment: sampling failed for %s", model)
        return _INSUFFICIENT_NO_ARRAYS
    if not pairs:
        return _cache(key, _INSUFFICIENT_NO_ARRAYS)

    matched, differing, trim_bands, ft_bands = compare_arrays(pairs)
    if matched < MIN_MATCHED:
        return _cache(key, SpecComparison(
            status="insufficient", pct_positions_differing=0.0,
            matched_positions=matched, trim_typ_band=None, ft_typ_band=None,
            note=(f"only {matched} positions are measured by both stations "
                  "— too few to compare their limits")))

    pct = differing / matched
    trim_typ, ft_typ = median(trim_bands), median(ft_bands)
    t_text, f_text = _band_text(trim_bands), _band_text(ft_bands)
    if pct > DIFFER_SHARE:
        note = (f"trim grades {t_text} where final test allows {f_text} "
                f"({pct * 100:.0f}% of matched points differ)")
        status = "differs"
    else:
        note = (f"trim and final test grade to the same limits at matched "
                f"points (trim {t_text}, final test {f_text})")
        status = "aligned"
    return _cache(key, SpecComparison(
        status=status, pct_positions_differing=pct, matched_positions=matched,
        trim_typ_band=trim_typ, ft_typ_band=ft_typ, note=note))


def _cache(key: Tuple[str, int], value: SpecComparison) -> SpecComparison:
    _CACHE[key] = value
    return value
