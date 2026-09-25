"""setup_change: a laser setting beyond the cut recipe changed -- did the pass rate follow?

Every silence test here was made to FAIL first by relaxing the control it names (a floor, the
table rule, the exclusion/alias tables) -- a green test that cannot go red proves nothing. See
the mutation-check step in the task report for MIN_RUN_DAYS and the table rule specifically.
"""
from dataclasses import replace
from datetime import datetime

import pytest

from laser_trim_analyzer.findings import presentation as P
from laser_trim_analyzer.findings.analyzers import setup_change
from findings_helpers import START, days, era, label, make_track, on_table, table


def setup_era(first_id, start, n, setup, good_share, *, system="B", track_name="default",
              step_days=1.0, r_in=4500.0):
    """`n` tracks sharing one `setup` dict, `good_share` of them ending inside limits -- the
    setup_change analog of findings_helpers.era()/table_era() (which have no `setup` slot)."""
    out = []
    for k, d in enumerate(days(start, n, step_days)):
        good = (k % 100) < good_share * 100
        t = make_track(first_id + k, date=d, passes=((0.8 if good else 1.5, 1.0),),
                       system=system, r_in=r_in)
        out.append(replace(t, setup=dict(setup), track_name=track_name))
    return out


# ---- the happy path: one finding, the brief's exact evidence shape ----------------------------

def test_a_setting_change_that_clears_the_floors_is_one_finding():
    # 200 tracks/~199 days each side -- comfortably past MIN_TRACKS_SIDE (100) and MIN_RUN_DAYS
    # (60); good_share moves 80% -> 50%; incoming resistance also moves, so the disclosure fires.
    before = setup_era(0, START, 200, {"laser_power": 50}, 0.80, r_in=4500.0)
    after = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50, r_in=4300.0)
    history, findings = setup_change.analyze("M", before + after, label)
    assert len(findings) == 1 and len(history) == 1
    f = findings[0]
    assert f.analyzer == "setup_change" and f.category == "Setting change"
    assert f.lever == "laser_settings" and f.systems == ("B",)
    assert f.expected_gain_points is None and f.gain_definition == ""       # a detection, never a gain
    assert f.title == "Laser 1 (LTS): laser power changed from 50 to 62"
    assert f.n_units == 400 and f.strength_value == pytest.approx(200.0)
    assert set(f.evidence) == {"before", "after", "track", "setting"}
    assert f.evidence["track"] == "default" and f.evidence["setting"] == "laser_power"
    assert f.evidence["before"]["trim_pass_pct"] == pytest.approx(80.0)
    assert f.evidence["after"]["trim_pass_pct"] == pytest.approx(50.0)
    assert f.evidence["after"]["first"] == "2024-09-01"          # ISO date, per the brief
    assert f.evidence["before"]["value"] == 50.0 and f.evidence["after"]["value"] == 62.0
    assert "not the only thing that changed" in f.summary


def test_history_entries_mirror_the_findings():
    before = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    after = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50)
    history, _ = setup_change.analyze("M", before + after, label)
    (h,) = history
    assert h["system"] == "B" and h["track"] == "default" and h["setting"] == "laser_power"
    assert h["before"]["value"] == 50.0 and h["after"]["value"] == 62.0
    assert h["after"]["first"] == "2024-09-01"


# ---- never across a limit-table change (ruling 6: held constant, no like-for-like fallback) ----

def test_a_change_that_coincides_with_a_limit_table_change_says_nothing():
    strict, lax = table(23, 0.10), table(12, 0.10)
    before = [on_table(t, strict) for t in setup_era(0, START, 200, {"laser_power": 50}, 0.80)]
    after = [on_table(t, lax) for t in
             setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50)]
    history, findings = setup_change.analyze("M", before + after, label)
    assert findings == [] and history == []


def test_an_unchanged_limit_table_still_reports():
    """Sanity check for the test above: the SAME table on both sides (on_table with one `table()`
    call) does not itself block a finding -- only a DIFFERING table does."""
    one = table(12, 0.10)
    before = [on_table(t, one) for t in setup_era(0, START, 200, {"laser_power": 50}, 0.80)]
    after = [on_table(t, one) for t in
             setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50)]
    assert len(setup_change.analyze("M", before + after, label)[1]) == 1


# ---- MIN_RUN_DAYS and MIN_TRACKS_SIDE, isolated from each other --------------------------------

def test_a_run_under_the_day_floor_says_nothing():
    assert setup_change.MIN_RUN_DAYS == 60
    before = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    # 150 tracks * 0.2-day step = 29.8 days: past MIN_TRACKS_SIDE, under MIN_RUN_DAYS.
    after = setup_era(1000, datetime(2024, 9, 1), 150, {"laser_power": 62}, 0.50, step_days=0.2)
    history, findings = setup_change.analyze("M", before + after, label)
    assert findings == [] and history == []


def test_a_run_under_the_track_floor_says_nothing():
    assert setup_change.MIN_TRACKS_SIDE == 100
    before = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    # 60 tracks * 2-day step = 118 days: past MIN_RUN_DAYS, under MIN_TRACKS_SIDE.
    after = setup_era(1000, datetime(2024, 9, 1), 60, {"laser_power": 62}, 0.50, step_days=2.0)
    history, findings = setup_change.analyze("M", before + after, label)
    assert findings == [] and history == []


# ---- EXCLUDED: identity-like keys are never even considered ------------------------------------

def test_an_excluded_key_changing_says_nothing():
    before = setup_era(0, START, 200, {"alias": "Outer Track", "laser_power": 50}, 0.80)
    after = setup_era(1000, datetime(2024, 9, 1), 200, {"alias": "Inner Track", "laser_power": 50}, 0.80)
    history, findings = setup_change.analyze("M", before + after, label)
    assert findings == [] and history == []          # alias excluded; laser_power never moved


def test_every_excluded_key_is_the_documented_four():
    assert setup_change.EXCLUDED == frozenset({"alias", "model", "model_number", "track_parameters"})


# ---- fix round 1, Important #1: recipe_change owns cut length, setup_change never re-reports it -

def test_laser_cut_length_alone_is_never_reported_by_setup_change():
    before = setup_era(0, START, 200, {"laser_cut_length": 4100.0}, 0.80)
    after = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_cut_length": 4000.0}, 0.40)
    history, findings = setup_change.analyze("M", before + after, label)
    assert findings == [] and history == []


def test_laser_cut_length_mm_is_also_never_reported():
    before = setup_era(0, START, 200, {"laser_cut_length_mm": 0.75}, 0.80,
                       system="A", track_name="TRK1")
    after = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_cut_length_mm": 0.55}, 0.40,
                      system="A", track_name="TRK1")
    history, findings = setup_change.analyze("M", before + after, label)
    assert findings == [] and history == []


def test_the_recipe_exclusion_is_imported_from_recipe_change_not_copied():
    """Proven by IDENTITY (the same frozenset object), not by re-typing recipe_change's key names
    here -- so the two analyzers can never silently drift apart on what "the recipe" covers."""
    from laser_trim_analyzer.findings.analyzers import recipe_change
    assert setup_change.RECIPE_PARAMETER_KEYS is recipe_change.RECIPE_PARAMETER_KEYS


# ---- ALIASES: laser 1's `response` and laser 2's `response_linear_or_function` are one setting --

def test_normalise_key_confirms_the_alias_source_and_target():
    from laser_trim_analyzer.core.trim_setup import normalise_key
    assert normalise_key("Response") == "response"
    assert normalise_key("Response (Linear or Function)") == "response_linear_or_function"
    assert setup_change.ALIASES == {"response_linear_or_function": "response"}


def test_response_is_one_setting_after_aliasing_on_either_laser():
    b_before = setup_era(0, START, 200, {"response": "Linear"}, 0.80, system="B")
    b_after = setup_era(1000, datetime(2024, 9, 1), 200, {"response": "Function"}, 0.50, system="B")
    a_before = setup_era(2000, START, 200, {"response_linear_or_function": "LINEAR"}, 0.80,
                         system="A", track_name="TRK1")
    a_after = setup_era(3000, datetime(2024, 9, 1), 200, {"response_linear_or_function": "FUNCTION"},
                        0.50, system="A", track_name="TRK1")
    _, findings = setup_change.analyze("M", b_before + b_after + a_before + a_after, label)
    assert len(findings) == 2
    assert {f.evidence["setting"] for f in findings} == {"response"}     # one canonical key, either laser
    assert sorted(f.title for f in findings) == [
        "Laser 1 (LTS): response changed from Linear to Function",
        "Laser 2 (DLTS): response changed from LINEAR to FUNCTION"]


# ---- non-numeric values compare as strings; numeric values compare as numbers ------------------

def test_non_numeric_values_are_compared_as_strings():
    before = setup_era(0, START, 200, {"indexing_method": "ERROR-SPLIT"}, 0.80)
    after = setup_era(1000, datetime(2024, 9, 1), 200, {"indexing_method": "CENTER-SPLIT"}, 0.50)
    (f,) = setup_change.analyze("M", before + after, label)[1]
    assert f.title == "Laser 1 (LTS): indexing method changed from ERROR-SPLIT to CENTER-SPLIT"


def test_an_int_and_an_equal_float_are_not_a_change():
    """50 (int, one file) and 50.0 (float, the next) are the same reading -- proves values compare
    by NUMBER, not by python type, and proves a pass-rate move alone is not enough to report
    anything: good_share moves 80% -> 50% here with no real setting change behind it."""
    tracks = (setup_era(0, START, 200, {"laser_power": 50}, 0.80)
             + setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 50.0}, 0.50))
    history, findings = setup_change.analyze("M", tracks, label)
    assert findings == [] and history == []


# ---- silence on data with nothing to say --------------------------------------------------------

def test_a_constant_setup_says_nothing():
    tracks = setup_era(0, START, 400, {"laser_power": 50, "pulse_duration": 200}, 0.60)
    history, findings = setup_change.analyze("M", tracks, label)
    assert findings == [] and history == []


def test_no_tracks_says_nothing():
    assert setup_change.analyze("M", [], label) == ([], [])


def test_tracks_with_no_setup_say_nothing():
    tracks = era(0, START, 200, (1.0,), 0.8)          # findings_helpers.era() never sets `setup`
    assert setup_change.analyze("M", tracks, label) == ([], [])


def test_a_track_with_no_real_cut_is_not_counted_either():
    """`t.passes == ()` (no real cut) must not enter a run even when `setup` is populated -- the
    same "a file with no cut is not a trimmed unit" rule recipe_change follows."""
    before = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    after = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50)
    untrimmed = [replace(t, passes=()) for t in after]
    history, findings = setup_change.analyze("M", before + untrimmed, label)
    assert findings == [] and history == []


# ---- every qualifying boundary is reported, not just the most recent (no MAX_EVENTS cap) -------

def test_every_qualifying_boundary_is_reported():
    run1 = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    run2 = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50)
    run3 = setup_era(2000, datetime(2025, 4, 1), 200, {"laser_power": 70}, 0.30)
    history, findings = setup_change.analyze("M", run1 + run2 + run3, label)
    assert len(findings) == 2 and len(history) == 2
    moves = sorted((f.evidence["before"]["value"], f.evidence["after"]["value"]) for f in findings)
    assert moves == [(50.0, 62.0), (62.0, 70.0)]


# ---- grouping is per (model, laser, TRACK), not just per laser (ruling 6 / Task 9) --------------

def test_two_tracks_of_one_model_are_independent_groups():
    trk1 = (setup_era(0, START, 200, {"laser_power": 50}, 0.80, system="A", track_name="TRK1")
           + setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50,
                       system="A", track_name="TRK1"))
    # TRK2 never changes -- must not borrow TRK1's boundary, and must not suppress it either.
    trk2 = (setup_era(2000, START, 200, {"laser_power": 36}, 0.70, system="A", track_name="TRK2")
           + setup_era(3000, datetime(2024, 9, 1), 200, {"laser_power": 36}, 0.70,
                       system="A", track_name="TRK2"))
    _, findings = setup_change.analyze("M", trk1 + trk2, label)
    assert len(findings) == 1
    assert findings[0].evidence["track"] == "TRK1"


# ---- presentation wiring: history group, pass-rate-move readout, the "Mon YYYY ·" statement ----

def test_group_and_readout_through_presentation():
    before = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    after = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50)
    f = setup_change.analyze("M", before + after, label)[1][0]
    d = f.to_dict()
    assert P.group_key(d) == "history"
    assert P.readout(d) == pytest.approx(-30.0)
    assert P.statement(d) == f"Sep 2024 · {f.title}"


def test_setup_change_has_a_group_and_is_never_shown_under_other():
    assert P.ANALYZER_GROUP["setup_change"] == "history"
