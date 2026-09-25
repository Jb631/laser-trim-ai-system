"""setup_change: the laser's setup beyond the cut recipe changed -- did the pass rate follow?

The rule (final review of 2026-09-25, risk 10; C1 + I2): per (model, laser, track) the files are
segmented on the WHOLE setup; a STABLE setup clears MIN_TRACKS_SIDE graded tracks over
MIN_RUN_DAYS; a change is the boundary between two consecutive stable setups, short-lived setups
between them absorbed into one compound change; one finding per change naming every setting that
differs, with the file's own labels; facts keep every per-setting boundary.

Every silence test here was made to FAIL first by relaxing the control it names (a floor, the
table rule, the reading/identity/mirror lists, the provenance rule) -- a green test that cannot go
red proves nothing. Example values are invented.
"""
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from laser_trim_analyzer.findings import presentation as P
from laser_trim_analyzer.findings.analyzers import setup_change
from findings_helpers import START, days, era, label, make_track, on_table, table


def setup_era(first_id, start, n, setup, good_share, *, system="B", track_name="default",
              step_days=1.0, r_in=4500.0, cut=1.0, inherited=False, vary=None, dates=None):
    """`n` tracks sharing one `setup` dict, `good_share` of them ending inside limits. `vary(k)`
    adds per-track keys (a per-unit reading); `dates` overrides the evenly spaced dates."""
    out = []
    for k, d in enumerate(dates if dates is not None else days(start, n, step_days)):
        good = (k % 100) < good_share * 100
        t = make_track(first_id + k, date=d, passes=((0.8 if good else 1.5, cut),),
                       system=system, r_in=r_in)
        s = dict(setup)
        if vary is not None:
            s.update(vary(k))
        out.append(replace(t, setup=s, track_name=track_name, setup_inherited=inherited))
    return out


def spread(start, n, span_days):
    """`n` dates from `start` to EXACTLY `span_days` later -- so a floor can be hit on the nose."""
    return [start + timedelta(days=round(k * span_days / (n - 1))) for k in range(n)]


def after(tracks, gap_days=1):
    """The day after the last of `tracks` (+ `gap_days` - 1)."""
    return max(t.file_date for t in tracks) + timedelta(days=gap_days)


def only(findings):
    assert len(findings) == 1, [f.title for f in findings]
    return findings[0]


def analyze(tracks):
    return setup_change.analyze("M", tracks, label)


# ---- the happy path: one finding, the documented evidence and facts shape -----------------------

def test_a_setting_change_that_clears_the_floors_is_one_finding():
    before = setup_era(0, START, 200, {"laser_power": 50}, 0.80, r_in=4500.0)
    later = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50, r_in=4300.0)
    facts, findings = analyze(before + later)
    f = only(findings)
    assert f.analyzer == "setup_change" and f.category == "Setting change"
    assert f.lever == "laser_settings" and f.systems == ("B",)
    assert f.expected_gain_points is None and f.gain_definition == ""       # a detection, never a gain
    assert f.title == "Laser 1 (LTS): Laser Power 50 → 62"                   # the file's own label
    assert f.n_units == 400 and f.strength_value == pytest.approx(200.0)
    assert set(f.evidence) == {"before", "after", "track", "settings", "absorbed_setups",
                               "absorbed_tracks", "recipe_changed"}
    assert f.evidence["settings"] == [{"setting": "laser_power", "label": "Laser Power",
                                       "from": 50.0, "to": 62.0}]
    b, a = f.evidence["before"], f.evidence["after"]
    assert b["trim_pass_pct"] == pytest.approx(80.0) and a["trim_pass_pct"] == pytest.approx(50.0)
    assert a["first"] == "2024-09-01" and b["last"] == "2024-07-18"
    assert b["limit_table"] is not None and b["limit_table"] == a["limit_table"]
    assert b["track_ids"] == [0, 199] and a["track_ids"] == [1000, 1199]
    assert f.evidence["absorbed_setups"] == 0 and f.evidence["recipe_changed"] is False
    assert "not the only thing that changed" in f.summary
    assert facts == [{"system": "B", "track": "default", "setting": "laser_power",
                      "label": "Laser Power", "date": "2024-09-01", "from": 50.0, "to": 62.0,
                      "reported": True, "why_not": None}]


# ---- C1: the move is setup against setup, never a setting's own run -----------------------------

def test_a_settings_own_run_spanning_another_change_is_never_its_before():
    """6607's shape: pulse duration's OWN run of 50 spans a laser power change. Pooled, the pulse
    change reads 55% -> 50% (a fall, coral); setup against setup it is 30% -> 50% (a rise)."""
    s1 = setup_era(0, START, 200, {"laser_power": 50, "pulse_duration": 50}, 0.80)
    s2 = setup_era(1000, after(s1), 200, {"laser_power": 60, "pulse_duration": 50}, 0.30)
    s3 = setup_era(2000, after(s2), 200, {"laser_power": 60, "pulse_duration": 100}, 0.50)
    _, findings = analyze(s1 + s2 + s3)
    assert len(findings) == 2
    pulse = next(f for f in findings if f.evidence["settings"][0]["setting"] == "pulse_duration")
    assert pulse.evidence["before"]["trim_pass_pct"] == pytest.approx(30.0)
    assert pulse.evidence["after"]["trim_pass_pct"] == pytest.approx(50.0)
    assert pulse.evidence["before"]["first"] == s2[0].file_date.date().isoformat()   # S2 only
    assert pulse.evidence["before"]["n"] == 200
    assert P.readout(pulse.to_dict()) == pytest.approx(20.0)                         # the right sign
    power = next(f for f in findings if f is not pulse)
    assert power.title == "Laser 1 (LTS): Laser Power 50 → 60"
    assert power.evidence["after"]["n"] == 200                                       # S2, not S2 + S3


def test_seven_settings_changed_on_one_day_are_one_finding_naming_all_seven():
    """6644-04's shape: seven settings moved in one file -- one change, one row, every setting."""
    old = {"laser_current": 20, "laser_frequency": 10, "laser_speed_high": 0.05,
           "laser_speed_slow": 0.008, "initial_points_ignored": 2, "ending_points_ignored": 4,
           "establish_coordinates": "User-Select"}
    new = {"laser_current": 24, "laser_frequency": 250, "laser_speed_high": 0.2,
           "laser_speed_slow": 0.01, "initial_points_ignored": 1, "ending_points_ignored": 1,
           "establish_coordinates": "Error-Split"}
    s1 = setup_era(0, START, 200, old, 0.90)
    s2 = setup_era(1000, after(s1), 200, new, 0.75)
    facts, findings = analyze(s1 + s2)
    f = only(findings)
    assert f.title == ("Laser 1 (LTS): 7 settings changed (Ending points Ignored, Establish "
                       "Coordinates, Initial Points Ignored, Laser Current, Laser Frequency, Laser "
                       "Speed High and Laser Speed Slow)")
    assert [s["setting"] for s in f.evidence["settings"]] == [
        "ending_points_ignored", "establish_coordinates", "initial_points_ignored",
        "laser_current", "laser_frequency", "laser_speed_high", "laser_speed_slow"]
    for s in f.evidence["settings"]:
        assert f"{s['label']} went from" in f.summary
    assert len(facts) == 7 and all(b["reported"] for b in facts)


def test_a_short_lived_setup_between_two_stable_ones_is_absorbed_into_one_compound_change():
    """6607's 2026-01-06 power change and 2026-01-12 pulse change: six days apart, one change."""
    s1 = setup_era(0, START, 200, {"laser_power": 60, "pulse_duration": 100}, 0.66)
    blip = setup_era(500, after(s1), 6, {"laser_power": 55, "pulse_duration": 100}, 0.50)
    s2 = setup_era(1000, after(blip), 200, {"laser_power": 55, "pulse_duration": 200}, 0.75)
    facts, findings = analyze(s1 + blip + s2)
    f = only(findings)
    assert f.title == "Laser 1 (LTS): Laser Power 60 → 55, Pulse Duration 100 → 200"
    assert f.evidence["absorbed_setups"] == 1 and f.evidence["absorbed_tracks"] == 6
    assert f.evidence["before"]["n"] == 200 and f.evidence["after"]["n"] == 200    # neither side has it
    assert "1 short-lived setup ran in between (6 tracks)" in f.summary
    assert [(b["setting"], b["date"], b["reported"]) for b in facts] == [
        ("laser_power", blip[0].file_date.date().isoformat(), True),
        ("pulse_duration", s2[0].file_date.date().isoformat(), True)]


def test_a_setting_undone_inside_a_short_lived_setup_is_no_change():
    s1 = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    blip = setup_era(500, after(s1), 5, {"laser_power": 52}, 0.20)
    s2 = setup_era(1000, after(blip), 200, {"laser_power": 50}, 0.40)
    facts, findings = analyze(s1 + blip + s2)
    assert findings == []
    assert [(b["from"], b["to"], b["reported"]) for b in facts] == [(50.0, 52.0, False),
                                                                     (52.0, 50.0, False)]
    assert all(b["why_not"] == "undone before the next stable setup" for b in facts)


def test_a_file_that_does_not_carry_a_setting_keeps_its_last_value():
    """An older template, or a blank cell, leaves a key out of one file: that is 'not captured',
    never a new value -- so the change after it is still a change from the last value seen."""
    s1 = setup_era(0, START, 200, {"laser_power": 50, "pulse_duration": 50}, 0.80)
    gap = setup_era(500, after(s1), 1, {"pulse_duration": 50}, 0.80)
    s2 = setup_era(1000, after(gap), 200, {"laser_power": 62, "pulse_duration": 50}, 0.40)
    f = only(analyze(s1 + gap + s2)[1])
    assert f.title == "Laser 1 (LTS): Laser Power 50 → 62"
    assert f.evidence["before"]["n"] == 201                   # the gap file stays in its setup


def test_a_settings_first_capture_is_not_a_change():
    s1 = setup_era(0, START, 200, {"pulse_duration": 50}, 0.80)
    s2 = setup_era(1000, after(s1), 200, {"pulse_duration": 50, "laser_power": 62}, 0.40)
    assert analyze(s1 + s2) == ([], [])


# ---- a stable setup is exactly the floors; each floor isolated -----------------------------------

def _three(middle_n, middle_span_days):
    s1 = setup_era(0, START, 200, {"laser_power": 50, "pulse_duration": 50}, 0.80)
    mid_start = after(s1)
    mid = setup_era(1000, mid_start, middle_n, {"laser_power": 60, "pulse_duration": 50}, 0.30,
                    dates=spread(mid_start, middle_n, middle_span_days))
    s3 = setup_era(2000, after(mid), 200, {"laser_power": 60, "pulse_duration": 100}, 0.50)
    return s1 + mid + s3


def test_a_middle_setup_at_both_floors_is_stable_so_there_are_two_changes():
    assert (setup_change.MIN_TRACKS_SIDE, setup_change.MIN_RUN_DAYS) == (100, 60)
    assert len(analyze(_three(100, 60))[1]) == 2


def test_a_middle_setup_one_track_short_of_the_floor_is_no_stable_setup():
    """Not stable, so not two changes -- and the stable setups either side of it are then 62 days
    apart, a period rather than one change (the 60-day cap below). A compound change across a
    short-lived setup is pinned by the absorbed-setup test above."""
    facts, findings = analyze(_three(99, 60))
    assert findings == []
    assert facts and all("62 days apart" in b["why_not"] for b in facts)


def test_a_middle_setup_one_day_short_of_the_floor_is_no_stable_setup():
    facts, findings = analyze(_three(100, 59))
    assert findings == []
    assert facts and all("61 days apart" in b["why_not"] for b in facts)


def test_a_run_under_the_day_floor_says_nothing():
    before = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    # 150 tracks * 0.2-day step = 29.8 days: past MIN_TRACKS_SIDE, under MIN_RUN_DAYS.
    later = setup_era(1000, datetime(2024, 9, 1), 150, {"laser_power": 62}, 0.50, step_days=0.2)
    facts, findings = analyze(before + later)
    assert findings == []
    assert [b["why_not"] for b in facts] == ["no stable setup after it yet"]


def test_a_run_under_the_track_floor_says_nothing():
    before = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    # 60 tracks * 2-day step = 118 days: past MIN_RUN_DAYS, under MIN_TRACKS_SIDE.
    later = setup_era(1000, datetime(2024, 9, 1), 60, {"laser_power": 62}, 0.50, step_days=2.0)
    assert analyze(before + later)[1] == []


def test_a_change_with_no_stable_setup_before_it_says_nothing():
    early = setup_era(0, START, 40, {"laser_power": 50}, 0.80)
    later = setup_era(1000, after(early), 200, {"laser_power": 62}, 0.50)
    facts, findings = analyze(early + later)
    assert findings == [] and [b["why_not"] for b in facts] == ["no stable setup before it"]


# ---- the two stable setups meet within 60 days (controller ruling, 2026-09-25) ------------------
# Two stable setups more than MIN_RUN_DAYS apart are not ONE change: a transition longer than a
# stable setup's own minimum is a period. It stays a fact, never a finding.

def test_stable_setups_60_days_apart_are_one_change():
    s1 = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    s2 = setup_era(1000, after(s1, gap_days=60), 200, {"laser_power": 62}, 0.50)
    f = only(analyze(s1 + s2)[1])
    assert (s2[0].file_date - s1[-1].file_date).days == 60
    assert f.title == "Laser 1 (LTS): Laser Power 50 → 62"


def test_stable_setups_61_days_apart_are_a_period_not_a_change():
    s1 = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    s2 = setup_era(1000, after(s1, gap_days=61), 200, {"laser_power": 62}, 0.50)
    facts, findings = analyze(s1 + s2)
    assert findings == []
    assert [(b["setting"], b["reported"]) for b in facts] == [("laser_power", False)]
    assert "61 days apart" in facts[0]["why_not"]


def test_stable_setups_60_days_and_20_hours_apart_are_a_period_not_a_change():
    """The cap compares the whole gap, never whole days (re-review of the fix round, 2026-09-25):
    6126 laser 1 Track A's stable setups met 60 days 20 hours apart and were reported as one
    change, because the gap was floored to 60. Its reason names the hours, so it never reads as
    "60 days apart -- longer than ... 60-day minimum"."""
    s1 = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    s2 = setup_era(1000, s1[-1].file_date + timedelta(days=60, hours=20), 200,
                   {"laser_power": 62}, 0.50)
    facts, findings = analyze(s1 + s2)
    assert findings == []
    assert [(b["setting"], b["reported"]) for b in facts] == [("laser_power", False)]
    assert "60 days 20 hours apart" in facts[0]["why_not"]


@pytest.mark.parametrize("gap,said", [
    (timedelta(days=60, hours=20, minutes=1), "60 days 20 hours apart"),    # 6126's own shape
    (timedelta(days=60, minutes=30), "just over 60 days apart"),            # never "60 days apart"
    (timedelta(days=61), "61 days apart"),
])
def test_a_periods_reason_names_the_gap_to_the_whole_hour(gap, said):
    s1 = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    s2 = setup_era(1000, s1[-1].file_date + gap, 200, {"laser_power": 62}, 0.50)
    facts, findings = analyze(s1 + s2)
    assert findings == [] and said in facts[0]["why_not"], facts[0]["why_not"]


def test_stable_setups_59_days_and_23_hours_apart_are_one_change():
    s1 = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    s2 = setup_era(1000, s1[-1].file_date + timedelta(days=59, hours=23), 200,
                   {"laser_power": 62}, 0.50)
    assert only(analyze(s1 + s2)[1]).title == "Laser 1 (LTS): Laser Power 50 → 62"


def test_the_stability_floor_is_the_whole_span_too():
    """Kept consistent with the cap: a setup spans at least 60 days when its first and last files
    are 60 days apart, counted to the hour -- 59 days 23 hours is short of it."""
    s1 = setup_era(0, START, 200, {"laser_power": 50, "pulse_duration": 50}, 0.80)
    mid_start = after(s1)
    span = timedelta(days=59, hours=23)
    mid = setup_era(1000, mid_start, 100, {"laser_power": 60, "pulse_duration": 50}, 0.30,
                    dates=[mid_start + span * (k / 99) for k in range(100)])
    s3 = setup_era(2000, after(mid), 200, {"laser_power": 60, "pulse_duration": 100}, 0.50)
    facts, findings = analyze(s1 + mid + s3)
    assert findings == []                      # the middle setup is not stable, so no two changes
    assert facts and all("61 days 23 hours apart" in b["why_not"] for b in facts)


def test_the_gap_is_measured_between_the_stable_setups_not_across_the_short_one():
    """A short-lived setup halfway does not make a 61-day transition two short ones."""
    s1 = setup_era(0, START, 200, {"laser_power": 50, "pulse_duration": 50}, 0.80)
    blip = setup_era(500, after(s1, gap_days=30), 5, {"laser_power": 55, "pulse_duration": 50}, 0.50)
    s2 = setup_era(1000, s1[-1].file_date + timedelta(days=61), 200,
                   {"laser_power": 55, "pulse_duration": 100}, 0.50)
    facts, findings = analyze(s1 + blip + s2)
    assert findings == []
    assert all("61 days apart" in b["why_not"] for b in facts)


# ---- never across a limit-table change ----------------------------------------------------------

def test_a_change_that_coincides_with_a_limit_table_change_says_nothing():
    strict, lax = table(23, 0.10), table(12, 0.10)
    before = [on_table(t, strict) for t in setup_era(0, START, 200, {"laser_power": 50}, 0.80)]
    later = [on_table(t, lax) for t in
             setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50)]
    facts, findings = analyze(before + later)
    assert findings == []
    assert [b["why_not"] for b in facts] == [
        "the setups either side were not graded against one and the same limit table"]


def test_an_unchanged_limit_table_still_reports():
    one = table(12, 0.10)
    before = [on_table(t, one) for t in setup_era(0, START, 200, {"laser_power": 50}, 0.80)]
    later = [on_table(t, one) for t in
             setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50)]
    assert len(analyze(before + later)[1]) == 1


def test_a_side_graded_against_two_tables_says_nothing():
    one, two = table(12, 0.10), table(23, 0.10)
    mixed = [on_table(t, one if k % 2 else two) for k, t in
             enumerate(setup_era(0, START, 200, {"laser_power": 50}, 0.80))]
    later = [on_table(t, one) for t in
             setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50)]
    assert analyze(mixed + later)[1] == []


# ---- the recipe bounds a setup but is never named (recipe_change owns it) ------------------------

def test_a_cut_length_change_alone_is_never_reported_by_setup_change():
    before = setup_era(0, START, 200, {"laser_cut_length": 4100.0}, 0.80, cut=4100.0)
    later = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_cut_length": 4000.0}, 0.40,
                      cut=4000.0)
    facts, findings = analyze(before + later)
    assert findings == [] and facts == []


def test_laser_cut_length_mm_is_also_never_reported():
    before = setup_era(0, START, 200, {"laser_cut_length_mm": 0.75}, 0.80,
                       system="A", track_name="TRK1")
    later = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_cut_length_mm": 0.55}, 0.40,
                      system="A", track_name="TRK1")
    assert analyze(before + later) == ([], [])


def test_the_first_cut_of_the_pass_log_bounds_a_setup():
    """Laser 2's cut length lives only in the pass log: S1 -> S2 is a recipe change (not reported
    here), so the power change after it compares S2 with S3 -- never S1 + S2 pooled."""
    s1 = setup_era(0, START, 200, {"laser_power": 50}, 0.90, system="A", track_name="TRK1", cut=0.75)
    s2 = setup_era(1000, after(s1), 200, {"laser_power": 50}, 0.40, system="A", track_name="TRK1",
                   cut=0.55)
    s3 = setup_era(2000, after(s2), 200, {"laser_power": 62}, 0.70, system="A", track_name="TRK1",
                   cut=0.55)
    f = only(analyze(s1 + s2 + s3)[1])
    assert f.title == "Laser 2 (DLTS): Laser Power 50 → 62"
    assert f.evidence["before"]["trim_pass_pct"] == pytest.approx(40.0)
    assert f.evidence["before"]["first"] == s2[0].file_date.date().isoformat()


def test_a_setting_that_changed_with_the_recipe_is_named_and_the_recipe_disclosed():
    before = setup_era(0, START, 200, {"laser_cut_length": 4100.0, "laser_power": 50}, 0.80,
                       cut=4100.0)
    later = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_cut_length": 4000.0,
                                                        "laser_power": 62}, 0.40, cut=4000.0)
    f = only(analyze(before + later)[1])
    assert f.title == "Laser 1 (LTS): Laser Power 50 → 62"
    assert f.evidence["recipe_changed"] is True
    assert "The cut recipe changed at the same point too" in f.summary
    assert [s["setting"] for s in f.evidence["settings"]] == ["laser_power"]


def test_the_recipe_keys_are_imported_from_recipe_change_not_copied():
    from laser_trim_analyzer.findings.analyzers import recipe_change
    assert setup_change.RECIPE_PARAMETER_KEYS is recipe_change.RECIPE_PARAMETER_KEYS


# ---- readings, identity keys and mirrored label rows are never settings ------------------------

@pytest.mark.parametrize("reading", ["length_theoretical", "starting_position",
                                     "error_split_low_voltage", "laser_height", "index_position",
                                     "low_error_split_volts", "pot_angle", "low_end_volts"])
def test_a_per_unit_reading_never_bounds_and_is_never_named(reading):
    """A value that differs on EVERY file: as a setting it would split every file into its own
    setup (no stable setup, no finding); as the reading it is, it is invisible."""
    def vary(k):
        return {reading: 100.0 + 0.37 * k}
    before = setup_era(0, START, 200, {"laser_power": 50}, 0.80, vary=vary)
    later = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50, vary=vary)
    facts, findings = analyze(before + later)
    f = only(findings)
    assert [s["setting"] for s in f.evidence["settings"]] == ["laser_power"]
    assert [b["setting"] for b in facts] == ["laser_power"]


def test_the_readings_are_the_documented_list():
    assert setup_change.READINGS == frozenset({
        "length_theoretical", "angle_theoretical", "starting_position",
        "error_split_low_voltage", "error_split_high_voltage",
        "error_split_low_position", "error_split_high_position",
        "laser_height", "index_position", "index_voltage",
        "low_error_split_volts", "high_error_split_volts",
        "low_error_split_position", "high_error_split_position",
        "start_position", "pot_angle", "stop_angle",
        "low_end_volts", "high_end_volts", "low_end_position", "high_end_position"})


@pytest.mark.parametrize("count, system, name", [("num_of_lin_positions", "A", "TRK1"),
                                                  ("number_of_readings_lin", "B", "default")])
def test_a_readings_count_is_a_setting_so_its_change_bounds_a_setup(count, system, name):
    """How many points the sweep takes: it follows the measured length on a few models, but its
    change is a change of the sweep's rows -- and so of the limit table. Kept as a setting it
    bounds: S1 (57 points) and S2 (111) stay two setups, each on one table, so the power change
    after S2 compares S2 with S3. Were it a reading, S1 + S2 would be one setup on two tables and
    the power change would go unreported."""
    t57, t111 = table(57, 0.10), table(111, 0.10)
    kw = dict(system=system, track_name=name)
    s1 = [on_table(t, t57, name) for t in setup_era(0, START, 200, {count: 57, "laser_power": 50},
                                                    0.90, **kw)]
    s2 = [on_table(t, t111, name) for t in setup_era(1000, after(s1), 200,
                                                     {count: 111, "laser_power": 50}, 0.40, **kw)]
    s3 = [on_table(t, t111, name) for t in setup_era(2000, after(s2), 200,
                                                     {count: 111, "laser_power": 62}, 0.70, **kw)]
    facts, findings = analyze(s1 + s2 + s3)
    f = only(findings)
    assert [s["setting"] for s in f.evidence["settings"]] == ["laser_power"]
    assert f.evidence["before"]["first"] == s2[0].file_date.date().isoformat()
    assert [b["why_not"] for b in facts if b["setting"] == count] == [
        "the setups either side were not graded against one and the same limit table"]


def test_the_identity_keys_are_the_documented_list():
    assert setup_change.EXCLUDED == frozenset({
        "alias", "model", "model_number", "track_parameters", "customer", "drawing",
        "template_updated", "report_info"})


def test_an_identity_key_changing_says_nothing():
    before = setup_era(0, START, 200, {"alias": "Outer Track", "laser_power": 50}, 0.80)
    later = setup_era(1000, datetime(2024, 9, 1), 200, {"alias": "Inner Track", "laser_power": 50},
                      0.80)
    assert analyze(before + later) == ([], [])


def test_the_lists_never_overlap_and_every_label_is_the_files_own_text():
    lists = [setup_change.EXCLUDED, setup_change.READINGS, frozenset(setup_change.LABELS),
             setup_change.RECIPE_PARAMETER_KEYS]
    for i, a in enumerate(lists):
        for b in lists[i + 1:]:
            assert not (a & b), a & b
    for key, text in setup_change.LABELS.items():
        assert text and "_" not in text and text != key.replace("_", " "), (key, text)
    assert setup_change.LABELS["laser_prr"] == "Laser PRR"
    assert setup_change.LABELS["laser_duration_ns"] == "Laser Duration (ns)"
    assert setup_change.LABELS["num_of_trim_parameters"] == "# of Trim Parameters"


def test_a_mirrored_label_row_is_never_a_setting():
    """Laser 1's value-first sheet read label-first stores rows like {"no": "Use Table Theory?"}:
    the key is a VALUE, the value is the LABEL of a real key in the same block. When the real
    setting flips No -> Yes those rows rename and re-point -- they must neither bound nor be named."""
    old = {"use_table_theory": "NO", "no": "Use Table Theory?", "balance_ends": "Yes",
           "yes": "Balance Ends?"}
    new = {"use_table_theory": "YES", "yes": "Use Table Theory?", "balance_ends": "Yes"}
    before = setup_era(0, START, 200, old, 0.80)
    later = setup_era(1000, datetime(2024, 9, 1), 200, new, 0.50)
    facts, findings = analyze(before + later)
    f = only(findings)
    assert f.title == "Laser 1 (LTS): Use Table Theory NO → YES"
    assert [b["setting"] for b in facts] == ["use_table_theory"]


def test_a_setting_with_no_label_yet_still_bounds_and_is_named_as_such():
    before = setup_era(0, START, 200, {"zz_future_setting": 1}, 0.80)
    later = setup_era(1000, datetime(2024, 9, 1), 200, {"zz_future_setting": 2}, 0.50)
    facts, findings = analyze(before + later)
    f = only(findings)
    assert f.title == "Laser 1 (LTS): 'zz_future_setting' (no label yet) 1 → 2"
    assert f.evidence["settings"] == [{"setting": "zz_future_setting", "label": None,
                                       "from": 1.0, "to": 2.0}]
    assert facts[0]["label"] is None and facts[0]["reported"] is True


# ---- aliases: one setting under two names --------------------------------------------------------

def test_normalise_key_confirms_the_alias_sources_and_targets():
    from laser_trim_analyzer.core.trim_setup import normalise_key
    assert normalise_key("Response") == "response"
    assert normalise_key("Response (Linear or Function)") == "response_linear_or_function"
    assert normalise_key("Length Mamimum") == "length_mamimum"         # the template's own typo
    assert setup_change.ALIASES == {"response_linear_or_function": "response",
                                    "length_mamimum": "length_maximum"}


def test_response_is_one_setting_after_aliasing_on_either_laser():
    b_before = setup_era(0, START, 200, {"response": "Linear"}, 0.80, system="B")
    b_after = setup_era(1000, datetime(2024, 9, 1), 200, {"response": "Function"}, 0.50, system="B")
    a_before = setup_era(2000, START, 200, {"response_linear_or_function": "LINEAR"}, 0.80,
                         system="A", track_name="TRK1")
    a_after = setup_era(3000, datetime(2024, 9, 1), 200, {"response_linear_or_function": "FUNCTION"},
                        0.50, system="A", track_name="TRK1")
    _, findings = analyze(b_before + b_after + a_before + a_after)
    assert sorted(f.title for f in findings) == ["Laser 1 (LTS): Response Linear → Function",
                                                 "Laser 2 (DLTS): Response LINEAR → FUNCTION"]


def test_the_typo_and_the_fixed_label_are_one_setting():
    """A template that fixed "Length Mamimum" to "Length Maximum" and moved the value the same day
    is a change of the setting, not two settings each seen once."""
    before = setup_era(0, START, 200, {"length_mamimum": 355}, 0.80, system="A", track_name="TRK1")
    later = setup_era(1000, datetime(2024, 9, 1), 200, {"length_maximum": 357}, 0.50, system="A",
                      track_name="TRK1")
    assert only(analyze(before + later)[1]).title == "Laser 2 (DLTS): Length Maximum 355 → 357"


# ---- values: compared per their own kind --------------------------------------------------------

def test_non_numeric_values_are_compared_as_strings():
    before = setup_era(0, START, 200, {"indexing_method": "ERROR-SPLIT"}, 0.80)
    later = setup_era(1000, datetime(2024, 9, 1), 200, {"indexing_method": "CENTER-SPLIT"}, 0.50)
    assert only(analyze(before + later)[1]).title == \
        "Laser 1 (LTS): Indexing Method ERROR-SPLIT → CENTER-SPLIT"


def test_an_int_and_an_equal_float_are_not_a_change():
    tracks = (setup_era(0, START, 200, {"laser_power": 50}, 0.80)
              + setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 50.0}, 0.50))
    assert analyze(tracks) == ([], [])


# ---- I2: a Track 2 whose own settings were never captured has no known setup ---------------------

def test_an_inherited_track2_block_then_a_captured_one_identical_otherwise_is_no_finding():
    """Before the capture, TRK2's `setup` is Track 1's block (10,350 incoming low limit); after it,
    Track 2's own (4,000). That is a parser change, not a process change."""
    inherited = setup_era(0, START, 200, {"initial_resistance_lower_limit": 10350, "laser_power": 50},
                          0.80, system="A", track_name="TRK2", inherited=True)
    own = setup_era(1000, after(inherited), 200, {"initial_resistance_lower_limit": 4000,
                                                  "laser_power": 50}, 0.50, system="A",
                    track_name="TRK2")
    assert analyze(inherited + own) == ([], [])


def test_an_inherited_block_contributes_nothing_even_when_track_1s_values_change():
    first = setup_era(0, START, 200, {"laser_power": 50}, 0.80, system="A", track_name="TRK2",
                      inherited=True)
    second = setup_era(1000, after(first), 200, {"laser_power": 62}, 0.40, system="A",
                       track_name="TRK2", inherited=True)
    assert analyze(first + second) == ([], [])


# ---- silence on data with nothing to say --------------------------------------------------------

def test_a_constant_setup_says_nothing():
    tracks = setup_era(0, START, 400, {"laser_power": 50, "pulse_duration": 200}, 0.60)
    assert analyze(tracks) == ([], [])


def test_no_tracks_says_nothing():
    assert analyze([]) == ([], [])


def test_tracks_with_no_setup_say_nothing():
    assert analyze(era(0, START, 200, (1.0,), 0.8)) == ([], [])


def test_a_track_with_no_real_cut_is_not_counted_either():
    before = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    later = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50)
    untrimmed = [replace(t, passes=()) for t in later]
    assert analyze(before + untrimmed) == ([], [])


# ---- every change between consecutive stable setups is reported; tracks are independent --------

def test_every_qualifying_change_is_reported():
    run1 = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    run2 = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50)
    run3 = setup_era(2000, datetime(2025, 4, 1), 200, {"laser_power": 70}, 0.30)
    facts, findings = analyze(run1 + run2 + run3)
    assert sorted(f.title for f in findings) == ["Laser 1 (LTS): Laser Power 50 → 62",
                                                 "Laser 1 (LTS): Laser Power 62 → 70"]
    assert len(facts) == 2


def test_two_tracks_of_one_model_are_independent_groups():
    trk1 = (setup_era(0, START, 200, {"laser_power": 50}, 0.80, system="A", track_name="TRK1")
            + setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50,
                        system="A", track_name="TRK1"))
    trk2 = (setup_era(2000, START, 200, {"laser_power": 36}, 0.70, system="A", track_name="TRK2")
            + setup_era(3000, datetime(2024, 9, 1), 200, {"laser_power": 36}, 0.70,
                        system="A", track_name="TRK2"))
    f = only(analyze(trk1 + trk2)[1])
    assert f.evidence["track"] == "TRK1"


# ---- presentation wiring ------------------------------------------------------------------------

def test_group_and_readout_through_presentation():
    """The change happened somewhere between the last file of the stable setup before (18 Jul)
    and the first of the one after (1 Sep): a row names both months when they differ, so a
    comparison of two setups years apart can never read as a change of that one month."""
    before = setup_era(0, START, 200, {"laser_power": 50}, 0.80)
    later = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50)
    f = only(analyze(before + later)[1])
    d = f.to_dict()
    assert P.group_key(d) == "history"
    assert P.readout(d) == pytest.approx(-30.0)
    assert P.statement(d) == "Jul 2024 – Sep 2024 · Laser 1 (LTS): Laser Power 50 → 62"


def test_a_change_inside_one_month_names_that_month_once():
    before = setup_era(0, START, 200, {"laser_power": 50}, 0.80)          # ends 18 Jul 2024
    later = setup_era(1000, datetime(2024, 7, 20), 200, {"laser_power": 62}, 0.50)
    d = only(analyze(before + later)[1]).to_dict()
    assert P.statement(d) == "Jul 2024 · Laser 1 (LTS): Laser Power 50 → 62"


def test_setup_change_has_a_group_and_is_never_shown_under_other():
    assert P.ANALYZER_GROUP["setup_change"] == "history"


def test_median_incoming_resistance_ignores_the_open_circuit_junk_reading():
    before = setup_era(0, START, 200, {"laser_power": 50}, 0.80, r_in=4500.0)
    before = [replace(t, untrimmed_resistance=1e12) if k < 150 else t for k, t in enumerate(before)]
    later = setup_era(1000, datetime(2024, 9, 1), 200, {"laser_power": 62}, 0.50, r_in=4300.0)
    f = only(analyze(before + later)[1])
    assert f.evidence["before"]["median_incoming_r"] == pytest.approx(4500.0)
