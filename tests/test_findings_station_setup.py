"""Station setup: does the laser grade this model to the same limits as final test?

Every silence test here was made to FAIL first by relaxing the control it names -- a green
test that cannot go red proves nothing. The comparison mechanics themselves (linked-pair
sampling, the bowtie-knee rule, the fallback to independent sampling) are already pinned,
exhaustively, by test_spec_alignment.py against `sample_and_compare`/`compare_station_specs`'s
shared implementation -- this file tests only what station_setup adds on top: the readout
population, the facts/finding split, and that a read failure is never swallowed here.
"""
from dataclasses import replace
from datetime import timedelta

import pytest

from laser_trim_analyzer.core import spec_alignment
from laser_trim_analyzer.findings.analyzers import station_setup
from findings_helpers import START, days, label, make_track
from test_spec_alignment import _add_trim, _add_ft, _flat, _positions, _seed, N_PTS


@pytest.fixture()
def db(tmp_path):
    """A real, empty DatabaseManager -- both globals injected per global-constraints.md, even
    though this analyzer is always called with `db` passed explicitly and never reaches
    get_database() itself; the rule is unconditional for anything that builds one."""
    import laser_trim_analyzer.database as _d
    import laser_trim_analyzer.database.manager as _m
    d = _m.DatabaseManager(tmp_path / "station.db")
    _m._db_manager = d
    _d._db_manager = d
    return d


def tracks_for(dates, system="B"):
    """Graded TrackViews, one per date -- the readout population (n_units/systems). Independent
    of the DB rows seeded for the comparison itself: station_setup's population and its spec
    comparison are two different samples of the same model, exactly as the analyzer treats them."""
    return [make_track(k, date=d, system=system) for k, d in enumerate(dates)]


def only(findings):
    assert len(findings) == 1, [f.title for f in findings]
    return findings[0]


class _Boom:
    """A database double that fails the test if station_setup ever queries it -- for the cases
    where there is no population to report on and no comparison should be attempted at all."""
    def session(self, *a, **k):
        raise AssertionError("must not query the database with nothing to report a comparison for")


# ---- a real mismatch is one finding; facts hold every comparable measurement -----------------

def test_a_mismatch_over_part_of_the_travel_is_one_finding(db):
    # 80 positions; final test grades the first 32 (40%) six times wider than trim -- comfortably
    # past both MIN_MATCHED (matched=80) and DIFFER_SHARE (40% > 10%). The 6607/8232-1 shape:
    # TRACKER D1's case is a difference over PART of the travel, never a uniformly different band.
    n, share = 80, 32
    pos = [float(x) for x in range(n)]
    trim_up, trim_lo = _flat(n, 0.05)
    ft_up = [0.30 if x < share else 0.05 for x in range(n)]
    ft_lo = [-u for u in ft_up]
    _seed(db, "M", (pos, trim_up, trim_lo), (pos, ft_up, ft_lo))

    tracks = tracks_for(days(START, 20))
    facts, findings = station_setup.analyze("M", db, tracks, label)
    f = only(findings)
    assert f.analyzer == "station_setup" and f.lever == "laser_limit_table"
    assert f.category == "Station setup"
    assert f.expected_gain_points is None and f.gain_definition == ""
    assert f.title == "The laser and final test grade to different limits over 40% of the travel"
    # The note-style sentence: the share, and BOTH stations' bands.
    assert "40%" in f.summary and "0.050" in f.summary and "0.300" in f.summary
    assert f.n_units == 20 and f.systems == ("B",)
    assert f.strength_value == pytest.approx(40.0)


def test_facts_and_evidence_carry_the_comparisons_own_fields(db):
    n, share = 80, 32
    pos = [float(x) for x in range(n)]
    trim_up, trim_lo = _flat(n, 0.05)
    ft_up = [0.30 if x < share else 0.05 for x in range(n)]
    ft_lo = [-u for u in ft_up]
    _seed(db, "M", (pos, trim_up, trim_lo), (pos, ft_up, ft_lo))

    facts, findings = station_setup.analyze("M", db, tracks_for(days(START, 20)), label)
    f = only(findings)
    assert set(facts) == {"status", "pct_positions_differing", "matched_positions",
                          "trim_typ_band", "ft_typ_band", "note"}
    assert facts["status"] == "differs"
    assert facts["pct_positions_differing"] == pytest.approx(0.40)
    assert facts["matched_positions"] == 160          # _seed's default n=2 linked units, 80 each
    assert facts["trim_typ_band"] == pytest.approx(0.05)
    assert set(f.evidence) == {"pct_positions_differing", "matched_positions", "trim_typ_band",
                               "ft_typ_band", "note"}
    assert f.evidence["note"] == facts["note"]


def test_systems_are_every_laser_among_the_graded_population(db):
    n, share = 80, 32
    pos = [float(x) for x in range(n)]
    trim_up, trim_lo = _flat(n, 0.05)
    ft_up = [0.30 if x < share else 0.05 for x in range(n)]
    ft_lo = [-u for u in ft_up]
    _seed(db, "M", (pos, trim_up, trim_lo), (pos, ft_up, ft_lo))

    tracks = tracks_for(days(START, 10), system="A") + tracks_for(days(START, 10), system="B")
    f = only(station_setup.analyze("M", db, tracks, label)[1])
    assert f.systems == ("A", "B") and f.n_units == 20


# ---- aligned stations: a real, comparable fact -- never a finding (the rule machine_compare
# and loss_origin both follow: the population floor gates a FACT, a further threshold only gates
# a FINDING) ------------------------------------------------------------------------------------

def test_aligned_stations_are_a_fact_never_a_finding(db):
    pos = _positions()
    up, lo = _flat(N_PTS, 0.05)
    _seed(db, "SAME", (pos, up, lo), (pos, up, lo))

    facts, findings = station_setup.analyze("SAME", db, tracks_for(days(START, 20)), label)
    assert findings == []
    # Still a COMPARABLE measurement (matched_positions clears MIN_MATCHED) -- so it is a fact,
    # exactly like machine_compare's under-the-gap laser and loss_origin's weak-AUC laser.
    assert facts["status"] == "aligned"
    assert facts["pct_positions_differing"] == 0.0
    assert facts["matched_positions"] >= spec_alignment.MIN_MATCHED


# ---- below spec_alignment's own floor: not a comparable measurement -- no fact, no finding ----

def test_too_few_matched_positions_is_nothing_not_even_a_fact(db):
    # 6 positions on each side, comfortably under MIN_MATCHED (20) -- test_spec_alignment.py's
    # own SPARSE shape. Below the floor the comparison answers nothing, so nothing is recorded.
    _seed(db, "SPARSE", (_positions(6), *_flat(6, 0.03)), (_positions(6), *_flat(6, 0.30)))
    facts, findings = station_setup.analyze("SPARSE", db, tracks_for(days(START, 20)), label)
    assert findings == [] and facts == {}


def test_no_stored_limits_on_either_station_is_also_nothing(db):
    # Trim data exists; final test never ran (or never matched) -- _INSUFFICIENT_NO_ARRAYS, the
    # same "below the floor" outcome as too few matched positions.
    _add_trim(db, "LONELY", _positions(), *_flat(N_PTS, 0.03), n=2)
    facts, findings = station_setup.analyze("LONELY", db, tracks_for(days(START, 20)), label)
    assert findings == [] and facts == {}


# ---- nothing to report a comparison against: the database is never even queried ---------------

def test_no_tracks_is_nothing_the_database_is_never_queried():
    assert station_setup.analyze("M", _Boom(), [], label) == ({}, [])


def test_ungraded_tracks_are_not_a_population_either():
    tracks = [replace(t, linearity_pass=None) for t in tracks_for(days(START, 20))]
    assert station_setup.analyze("M", _Boom(), tracks, label) == ({}, [])


def test_undated_tracks_are_not_a_population_either():
    tracks = [replace(t, file_date=None) for t in tracks_for(days(START, 20))]
    assert station_setup.analyze("M", _Boom(), tracks, label) == ({}, [])


# ---- only the model's latest year counts toward n_units (the brief's "graded tracks in the
# model's latest year") --------------------------------------------------------------------------

def test_only_the_latest_year_counts_toward_n_units(db):
    n, share = 80, 32
    pos = [float(x) for x in range(n)]
    trim_up, trim_lo = _flat(n, 0.05)
    ft_up = [0.30 if x < share else 0.05 for x in range(n)]
    ft_lo = [-u for u in ft_up]
    _seed(db, "M", (pos, trim_up, trim_lo), (pos, ft_up, ft_lo))

    recent = tracks_for(days(START, 20))
    stale = [make_track(900 + k, date=START - timedelta(days=800), system="B") for k in range(50)]
    f = only(station_setup.analyze("M", db, recent + stale, label)[1])
    assert f.n_units == 20                    # not 70 -- the 800-days-back tracks are last year's


def test_ungraded_tracks_among_recent_ones_do_not_count(db):
    n, share = 80, 32
    pos = [float(x) for x in range(n)]
    trim_up, trim_lo = _flat(n, 0.05)
    ft_up = [0.30 if x < share else 0.05 for x in range(n)]
    ft_lo = [-u for u in ft_up]
    _seed(db, "M", (pos, trim_up, trim_lo), (pos, ft_up, ft_lo))

    recent = tracks_for(days(START, 20))
    ungraded = [replace(t, linearity_pass=None) for t in tracks_for(days(START, 5))]
    f = only(station_setup.analyze("M", db, recent + ungraded, label)[1])
    assert f.n_units == 20


# ---- the whole point of calling sample_and_compare instead of compare_station_specs: a read
# failure is never caught here -- it must reach the caller (the findings engine's own guard,
# pinned separately in test_findings_engine_db.py through compute_for_model) ---------------------

def test_a_raising_sampler_is_not_caught_here(db, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(spec_alignment, "sample_and_compare", boom)
    with pytest.raises(RuntimeError, match="database is locked"):
        station_setup.analyze("M", db, tracks_for(days(START, 20)), label)
