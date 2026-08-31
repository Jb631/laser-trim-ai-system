"""Trim-vs-FT spec alignment — the position-matched limit comparison.

James, 2026-08-30: "i also want to know when the trim and test specs dont
align." This is VISIBILITY only: nothing here re-grades a unit, it just says
whether the two stations are holding the part to the same requirement, so a
cross-station number (escapes, the Gap) is read knowing what it compares.

The comparison method is the census's (scripts/data_trust_census.py, James's
bowtie correction in 4c6ebd8): limits are compared ONLY at near-coincident
positions, never interpolated across a knee. These tests pin that — a
comparison that interpolated would call a stepped FT spec "differs" for a
reason that is pure arithmetic, not a real requirement difference.
"""
from datetime import datetime

import pytest

from laser_trim_analyzer.core.spec_alignment import (
    MIN_MATCHED, clear_spec_alignment_cache, compare_station_specs)
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import (
    AnalysisResult, FinalTestResult, FinalTestTrack, StatusType, SystemType,
    TrackResult)

D0 = datetime(2026, 1, 5)
N_PTS = 60                       # comfortably past MIN_MATCHED once matched


@pytest.fixture()
def db(tmp_path):
    clear_spec_alignment_cache()          # module-level cache must not leak
    yield DatabaseManager(tmp_path / "spec.db")
    clear_spec_alignment_cache()


def _positions(n=N_PTS, step=1.0):
    return [round(i * step, 6) for i in range(n)]


def _flat(n, half):
    return [half] * n, [-half] * n


def _add_trim(db, model, positions, upper, lower, *, n=1, tag="T"):
    """`n` trim units of `model` carrying this spec. Returns their row ids."""
    ids = []
    for k in range(n):
        with db.session() as s:
            a = AnalysisResult(model=model, serial=f"{model}-{tag}{k}",
                               system=SystemType.A, file_date=D0,
                               filename=f"{model}_{tag}{k}.xls",
                               overall_status=StatusType.PASS)
            s.add(a)
            s.flush()
            s.add(TrackResult(analysis_id=a.id, track_id="TRK1",
                              status=StatusType.PASS,
                              position_data=list(positions),
                              upper_limits=list(upper),
                              lower_limits=list(lower)))
            ids.append(a.id)
    return ids


def _add_ft(db, model, positions, upper, lower, *, n=1, link_to=None, tag="F"):
    """`n` final-test records; `link_to` is the trim row each one came from."""
    for k in range(n):
        with db.session() as s:
            f = FinalTestResult(
                model=model, serial=f"{model}-{tag}{k}",
                filename=f"{model}_{tag}{k}.xls", file_date=D0,
                overall_status=StatusType.PASS,
                linked_trim_id=(link_to[k] if link_to else None))
            s.add(f)
            s.flush()
            s.add(FinalTestTrack(final_test_id=f.id, track_id="TRK1",
                                 status=StatusType.PASS,
                                 position_data=list(positions),
                                 upper_limits=list(upper),
                                 lower_limits=list(lower)))


def _seed(db, model, trim_spec, ft_spec, *, n=2, linked=True):
    """One model, both stations — linked unit-for-unit unless told otherwise."""
    ids = _add_trim(db, model, *trim_spec, n=n)
    _add_ft(db, model, *ft_spec, n=n, link_to=ids if linked else None)


# ---- the three verdicts ---------------------------------------------------

def test_identical_specs_are_aligned(db):
    pos = _positions()
    up, lo = _flat(N_PTS, 0.05)
    _seed(db, "SAME", (pos, up, lo), (pos, up, lo))

    c = compare_station_specs(db, "SAME")
    assert c.status == "aligned"
    assert c.pct_positions_differing == 0.0
    assert c.matched_positions >= MIN_MATCHED
    assert c.trim_typ_band == pytest.approx(0.05)
    assert c.ft_typ_band == pytest.approx(0.05)
    assert "same" in c.note.lower()


def test_tight_trim_against_loose_final_test_differs(db):
    """The 6126 shape: trim grades to a hairline, FT allows an order more."""
    pos = _positions()
    _seed(db, "SPLIT", (pos, *_flat(N_PTS, 0.03)), (pos, *_flat(N_PTS, 0.10)))

    c = compare_station_specs(db, "SPLIT")
    assert c.status == "differs"
    assert c.pct_positions_differing == pytest.approx(1.0)
    assert c.trim_typ_band == pytest.approx(0.03)
    assert c.ft_typ_band == pytest.approx(0.10)
    assert "0.030" in c.note and "0.100" in c.note
    assert "100%" in c.note


def test_missing_final_test_arrays_are_insufficient_not_a_verdict(db):
    """No FT arrays is not "aligned" — it is nothing to say. Say nothing."""
    pos = _positions()
    _add_trim(db, "LONELY", pos, *_flat(N_PTS, 0.03), n=2)

    c = compare_station_specs(db, "LONELY")
    assert c.status == "insufficient"
    assert c.matched_positions == 0
    assert c.trim_typ_band is None and c.ft_typ_band is None


def test_too_few_matched_positions_are_insufficient(db):
    """Two stations that share almost no positions cannot be compared."""
    _seed(db, "SPARSE", (_positions(6), *_flat(6, 0.03)),
          (_positions(6), *_flat(6, 0.30)))

    c = compare_station_specs(db, "SPARSE")
    assert c.status == "insufficient"
    assert c.matched_positions < MIN_MATCHED


def test_unknown_model_is_insufficient(db):
    assert compare_station_specs(db, "NOPE").status == "insufficient"


# ---- the bowtie rule ------------------------------------------------------

def test_a_bowtie_knee_is_never_interpolated_across(db):
    """James's correction (census 4c6ebd8): compare only near-coincident points.

    FT here is a STEPPED (bowtie) spec on a coarse position table; trim samples
    twice as finely and holds the same band on each side of the knee. Every
    trim point that lands on an FT point agrees. The trim points that fall
    BETWEEN FT stations — including the one straddling the knee, where an
    interpolated FT limit would read half-way between 0.05 and 0.20 and look
    like a disagreement — must simply not be matched at all.
    """
    knee = 40.0
    ft_pos = [float(x) for x in range(0, 80, 2)]          # every 2 units
    ft_up = [0.05 if x < knee else 0.20 for x in ft_pos]
    ft_lo = [-u for u in ft_up]
    trim_pos = [float(x) for x in range(0, 80)]           # every 1 unit
    trim_up = [0.05 if x < knee else 0.20 for x in trim_pos]
    trim_lo = [-u for u in trim_up]
    _seed(db, "BOWTIE", (trim_pos, trim_up, trim_lo), (ft_pos, ft_up, ft_lo))

    c = compare_station_specs(db, "BOWTIE")
    assert c.status == "aligned"
    assert c.pct_positions_differing == 0.0
    # Tolerance is HALF the finer station's spacing (0.5 here), so only the
    # trim points that coincide with an FT station match: 40 of the 80, not 80.
    assert c.matched_positions == 40 * 2                  # 2 sampled pairs
    # Both sides carry both halves of the bowtie, so the note quotes a RANGE.
    assert c.trim_typ_band is not None and c.ft_typ_band is not None


def test_a_real_bowtie_difference_is_still_caught(db):
    """The rule must not become a way to never notice anything.

    Same stepped FT spec, but trim holds one flat hairline across the whole
    sweep — a genuine requirement difference at most matched points.
    """
    knee = 60.0
    ft_pos = [float(x) for x in range(0, 80, 2)]
    ft_up = [0.05 if x < knee else 0.60 for x in ft_pos]
    trim_pos = [float(x) for x in range(0, 80)]
    _seed(db, "REALDIFF", (trim_pos, *_flat(len(trim_pos), 0.60)),
          (ft_pos, ft_up, [-u for u in ft_up]))

    c = compare_station_specs(db, "REALDIFF")
    assert c.status == "differs"
    # 30 of the 40 matched positions sit below the knee, where trim allows
    # ±0.60 and final test allows ±0.05 — a real requirement difference.
    assert c.pct_positions_differing == pytest.approx(0.75)


def test_a_difference_over_part_of_the_travel_still_counts(db):
    """A quarter of the sweep graded differently is a real mismatch.

    James (2026-08-30) asked to know when the specs "dont align" — not only
    when they disagree everywhere. On the real database this is the common
    shape: models sit at EXACTLY 0% differing when the two stations agree, so
    a fifth to a third of positions disagreeing is signal, never noise. 8340-1
    (23.7%) read as "aligned" under the old half-the-points rule and printed
    escapes/Gap as if both stations asked the same question.
    """
    n = 80
    quarter = n // 4
    pos = [float(x) for x in range(n)]
    trim_up = [0.05] * n
    # Final test holds the same band except over the first quarter of travel.
    ft_up = [0.30 if x < quarter else 0.05 for x in range(n)]
    _seed(db, "PARTIAL", (pos, trim_up, [-u for u in trim_up]),
          (pos, ft_up, [-u for u in ft_up]))

    c = compare_station_specs(db, "PARTIAL")
    assert c.status == "differs"
    assert c.pct_positions_differing == pytest.approx(0.25)
    # The note has to carry the MAGNITUDE, because "differs" now spans a
    # quarter of the travel through all of it, and a reader deciding whether to
    # trust an escape count needs to know which they are looking at.
    assert "25%" in c.note


def test_a_handful_of_differing_points_stays_aligned(db):
    """The threshold still has to hold a floor, or the banner cries wolf."""
    n = 80
    pos = [float(x) for x in range(n)]
    trim_up = [0.05] * n
    ft_up = [0.30 if x < 4 else 0.05 for x in range(n)]      # 5% of positions
    _seed(db, "FEWPTS", (pos, trim_up, [-u for u in trim_up]),
          (pos, ft_up, [-u for u in ft_up]))

    c = compare_station_specs(db, "FEWPTS")
    assert c.status == "aligned"


# ---- which records get compared -------------------------------------------

def test_linked_pairs_beat_the_newest_unlinked_records(db):
    """Compare a unit against ITSELF — the census's population, and why.

    This is the 6126 regression. Sampling "the newest trim tracks and the
    newest FT tracks" independently looked equivalent (these are model-level
    specs, after all), but on the real database 6126's newest FT records sweep
    a DIFFERENT position table than its newest trim records: the arbitrary
    pairing matched a fifth as many positions and reported "aligned", while
    the census's linked pairs say 100% of matched positions differ.
    """
    pos = _positions()
    _seed(db, "LINKY", (pos, *_flat(N_PTS, 0.03)), (pos, *_flat(N_PTS, 0.30)))
    # Newer, UNLINKED final tests on a position table that never coincides
    # with the trim sweep — they would drown the real comparison.
    off = [x + 0.5 for x in pos]
    _add_ft(db, "LINKY", off, *_flat(N_PTS, 0.03), n=3, tag="U")

    c = compare_station_specs(db, "LINKY")
    assert c.status == "differs"
    assert c.pct_positions_differing == pytest.approx(1.0)
    assert c.ft_typ_band == pytest.approx(0.30)      # the LINKED FT spec


def test_a_model_with_no_links_still_gets_an_answer(db):
    """Older data never had links built. Silence there would hide real gaps."""
    pos = _positions()
    _seed(db, "NOLINK", (pos, *_flat(N_PTS, 0.03)), (pos, *_flat(N_PTS, 0.30)),
          linked=False)

    c = compare_station_specs(db, "NOLINK")
    assert c.status == "differs"
    assert c.pct_positions_differing == pytest.approx(1.0)


# ---- caching --------------------------------------------------------------

def test_the_result_is_cached_per_model(db):
    pos = _positions()
    _seed(db, "CACHED", (pos, *_flat(N_PTS, 0.03)), (pos, *_flat(N_PTS, 0.10)))
    first = compare_station_specs(db, "CACHED")

    class _Boom:
        def session(self, *a, **k):
            raise AssertionError("cached result must not re-query the database")

    assert compare_station_specs(_Boom(), "CACHED") == first
    clear_spec_alignment_cache()
    assert compare_station_specs(db, "CACHED") == first    # recomputes cleanly


def test_a_broken_database_degrades_to_insufficient(db):
    """The Model page and the FOCUS list must survive a bad read."""
    class _Boom:
        def session(self, *a, **k):
            raise RuntimeError("no such table")

    c = compare_station_specs(_Boom(), "BROKEN")
    assert c.status == "insufficient" and c.matched_positions == 0
