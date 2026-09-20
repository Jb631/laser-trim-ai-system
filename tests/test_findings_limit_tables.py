from datetime import datetime

from findings_helpers import START, label, on_table, make_track, table, table_era

COARSE = table(12, 0.10)                      # 12 points, +/-0.10 everywhere
DENSE = table(23, 0.10)                       # the SAME band, sampled twice as densely
LOOSE = table(12, 0.30)                       # the same 12 points, three times wider


def _run(tracks):
    from laser_trim_analyzer.findings.analyzers import limit_tables
    return limit_tables.analyze("M", tracks, label)


def test_one_table_is_history_not_a_finding():
    history, findings = _run(table_era(0, START, 200, COARSE, 0.5))
    assert findings == []
    assert [(h["rows"], h["graded"], h["n"], h["trim_pass_pct"]) for h in history] == [(12, 12, 200, 50.0)]


def test_two_tables_at_once_same_band_other_density():
    tracks = table_era(0, START, 200, COARSE, 0.6) + table_era(1000, START, 200, DENSE, 0.3)
    history, findings = _run(tracks)
    assert len(history) == 2 and len(findings) == 1
    f = findings[0]
    assert f.lever == "laser_limit_table" and f.lead_time == "same day" and f.category == "Limit table"
    assert f.expected_gain_points is None and f.units_per_year is None      # a different test is not more yield
    assert "in service at once" in f.title and "Laser 1 (LTS)" in f.title and "System B" not in f.summary
    assert f.evidence["concurrent"] is True and f.evidence["comparison"]["kind"] == "same_band_other_density"
    assert "12 points and the other 23" in f.summary and "stricter test" in f.summary
    assert "60% left the laser inside limits" in f.summary and "30% left the laser inside limits" in f.summary
    assert "nevertheless passes MORE" not in f.summary                       # here the denser table passes fewer
    assert f.strength_value == 0.5 and f.n_units == 400


def test_a_change_is_told_in_time_order_and_says_which_way_the_band_moved():
    before = table_era(0, START, 100, COARSE, 0.7)                           # 100 days on the tight table...
    after = table_era(1000, datetime(2024, 5, 10), 100, LOOSE, 1.0)          # ...then the loose one; no overlap
    _, findings = _run(after + before)                                       # order of input must not matter
    f = findings[0]
    assert "the limit table changed" in f.title and "around 2024-05-14" in f.summary    # the 5th percentile of the new table's dates
    assert f.summary.index("Before:") < f.summary.index("After:")
    assert f.evidence["older"]["trim_pass_pct"] == 70.0 and f.evidence["newer"]["trim_pass_pct"] == 100.0
    c = f.evidence["comparison"]
    assert c["kind"] == "different_band" and c["wider"] == 12 and c["narrower"] == 0 and c["max_abs_diff"] == 0.2
    assert "WIDER at 12" in f.summary and "different REQUIREMENTS" in f.summary
    assert f.evidence["concurrent"] is False


def test_a_handful_of_odd_files_is_not_a_second_table():
    # 29 of 129 = 22% of the year, far above MIN_SHARE -- so it is MIN_TABLE_N alone that keeps this quiet.
    tracks = table_era(0, START, 100, COARSE, 0.5) + table_era(1000, START, 29, DENSE, 0.5)
    history, findings = _run(tracks)
    assert findings == [] and len(history) == 1
    assert len(_run(tracks + table_era(2000, START, 1, DENSE, 0.5))[1]) == 1       # the 30th file tips it


def test_a_table_retired_more_than_a_year_ago_is_history_only():
    old = table_era(0, datetime(2021, 1, 1), 100, DENSE, 0.4)
    new = table_era(1000, datetime(2024, 1, 1), 100, COARSE, 0.6)
    history, findings = _run(old + new)
    assert findings == [] and len(history) == 2
    assert [h["first"][:4] for h in history] == ["2021", "2024"]             # history runs in time order


def test_a_small_share_of_the_year_is_not_in_service():
    tracks = table_era(0, START, 360, COARSE, 0.5) + table_era(1000, START, 35, DENSE, 0.5)   # 35/395 = 8.9% < 10%
    assert _run(tracks)[1] == []


def test_when_the_stricter_table_passes_more_it_says_something_else_differs():
    tracks = table_era(0, START, 200, COARSE, 0.3) + table_era(1000, START, 200, DENSE, 0.8)
    f = _run(tracks)[1][0]
    assert "nevertheless passes MORE" in f.summary and "period, recipe or product variant" in f.summary


def test_two_tracks_of_one_model_may_carry_different_tables():
    tracks = (table_era(0, START, 200, COARSE, 0.5, name="Track A")
              + table_era(1000, START, 200, LOOSE, 0.5, name="Track B"))
    history, findings = _run(tracks)
    assert findings == [] and {h["track"] for h in history} == {"Track A", "Track B"}


def test_lasers_are_judged_separately():
    tracks = (table_era(0, START, 200, COARSE, 0.5, system="A") + table_era(1000, START, 200, DENSE, 0.5, system="B"))
    assert _run(tracks)[1] == []


def test_bands_are_compared_across_grids_that_share_no_position():
    from laser_trim_analyzer.findings.analyzers.limit_tables import compare
    from laser_trim_analyzer.findings.data import limit_table_of
    tight = limit_table_of(*table(11, 0.05))                                 # positions -10, -8, ... 10
    loose = limit_table_of(*table(11, 0.08, span=(-9.0, 9.0)))               # a grid that shares NO position with it
    c = compare(tight, loose)
    assert c["kind"] == "different_band" and c["wider"] == 11 and c["narrower"] == 0
    assert c["max_abs_diff"] == 0.03 and c["travel_differs"] is False        # one grid step is not a change of travel
    far = limit_table_of(*table(11, 0.05, span=(100.0, 120.0)))
    c = compare(tight, far)
    assert c["kind"] == "not_comparable" and c["compared_positions"] == 0 and c["travel_differs"] is True


def _bowtie(n, span=(-10.0, 10.0), waist=0.02, slope=0.01):
    """One band FUNCTION -- half-width = waist + slope * |position| -- sampled on `n` points."""
    lo, hi = span
    pos = tuple(lo + (hi - lo) * i / (n - 1) for i in range(n))
    half = tuple(waist + slope * abs(x) for x in pos)
    return pos, half, tuple(-h for h in half)


def test_one_band_sampled_at_two_densities_is_the_same_band_whichever_came_first():
    """Review finding: with the older table interpolated at the newer one's positions, a coarse grid
    that straddles the bowtie's corner read 'NARROWER at 1 ... different REQUIREMENTS' -- but only one
    way round. The same two tables must give the same answer in either order."""
    from laser_trim_analyzer.findings.analyzers.limit_tables import compare
    from laser_trim_analyzer.findings.data import limit_table_of
    coarse = limit_table_of(*_bowtie(20, span=(-9.5, 9.5)))                  # 1.0-degree steps, straddling the corner at 0
    dense = limit_table_of(*_bowtie(39, span=(-9.5, 9.5)))                   # 0.5-degree steps, a point ON the corner
    assert compare(coarse, dense)["kind"] == "same_band_other_density"
    assert compare(dense, coarse)["kind"] == "same_band_other_density"


def test_a_staircase_band_on_half_step_offset_grids_is_the_same_band():
    """The two REAL 8232-1 tables (56 rows on laser 2, 57 on laser 1) are one stepped band on grids half a
    step apart. Linear interpolation read that as 'wider at 22 and narrower at 22'."""
    from laser_trim_analyzer.findings.analyzers.limit_tables import compare
    from laser_trim_analyzer.findings.data import limit_table_of

    def stairs(positions):
        half = tuple(0.0055 + 0.0005 * int(abs(x) // 2.5) for x in positions)      # a step every 2.5 degrees
        return positions, half, tuple(-h for h in half)
    a = limit_table_of(*stairs(tuple(-28.0 + i for i in range(57))))
    b = limit_table_of(*stairs(tuple(-27.5 + i for i in range(56))))
    assert compare(a, b)["kind"] == "same_band_other_density" and compare(b, a)["kind"] == "same_band_other_density"


def test_a_one_percent_tightening_is_a_different_band():
    from laser_trim_analyzer.findings.analyzers.limit_tables import compare
    from laser_trim_analyzer.findings.data import limit_table_of
    c = compare(limit_table_of(*table(21, 0.0100)), limit_table_of(*table(21, 0.0099)))
    assert c["kind"] == "different_band" and c["narrower"] == 21 and c["wider"] == 0


def test_a_band_widened_only_at_its_ends_is_a_different_band():
    from laser_trim_analyzer.findings.analyzers.limit_tables import compare
    from laser_trim_analyzer.findings.data import limit_table_of
    c = compare(limit_table_of(*table(31, 0.02)), limit_table_of(*table(61, 0.02, end_half=0.18)))
    assert c["kind"] == "different_band" and c["wider"] == 2 and c["narrower"] == 0 and c["max_abs_diff"] == 0.16


def test_one_straggler_file_does_not_turn_a_change_into_two_tables_at_once():
    """Review finding: overlap was min(last) - max(first), so ONE rework file on the retired table,
    months later, fabricated 137 days of 'in service at once' and destroyed the change date."""
    old = table_era(0, datetime(2024, 1, 1), 180, COARSE, 0.5)
    new = table_era(1000, datetime(2024, 7, 1), 180, LOOSE, 0.9)
    straggler = table_era(5000, datetime(2024, 11, 15), 1, COARSE, 0.5)      # one file on the old program
    pilot = table_era(6000, datetime(2024, 1, 15), 1, LOOSE, 0.9)            # one early trial of the new one
    for extra in ([], straggler, pilot, straggler + pilot):
        (f,) = _run(old + new + extra)[1]
        assert "the limit table changed" in f.title, (len(extra), f.title)
        assert f.evidence["concurrent"] is False and f.evidence["overlap_days"] < 0
        assert "around 2024-07-" in f.summary                                # the change date survives


def test_a_dormant_model_is_described_in_its_own_period_not_the_last_12_months():
    tracks = (table_era(0, datetime(2018, 1, 1), 200, COARSE, 0.6)
              + table_era(1000, datetime(2018, 1, 1), 200, DENSE, 0.3))
    (f,) = _run(tracks)[1]
    assert "In the 12 months to 2018-07-19" in f.summary and "last 12 months" not in f.summary
    assert "were in service at once" in f.title and f.evidence["window_to"] == "2018-07-19"


def test_two_tables_over_different_travel_are_not_called_stricter_and_laxer():
    """89 points over +/-11 degrees against 45 over +/-22: the denser table never looks at half the track."""
    short = table(89, 0.10, span=(-11.0, 11.0))
    long_ = table(45, 0.10, span=(-22.0, 22.0))
    (f,) = _run(table_era(0, START, 200, long_, 0.3) + table_era(1000, START, 200, short, 0.8))[1]
    c = f.evidence["comparison"]
    assert c["kind"] == "same_band_other_density" and c["travel_differs"] is True
    assert "do not grade the same travel" in f.summary and "-22 to 22" in f.summary and "-11 to 11" in f.summary
    assert "stricter test" not in f.summary.replace("neither is simply the stricter test", "")
    assert "nevertheless passes MORE" not in f.summary                        # the cause is sitting in the travel


def test_three_live_tables_says_which_two_are_compared():
    tracks = (table_era(0, START, 200, COARSE, 0.5) + table_era(1000, START, 150, DENSE, 0.5)
              + table_era(2000, START, 100, LOOSE, 0.9))
    (f,) = _run(tracks)[1]
    assert "3 limit tables" in f.title and "The two busiest of the 3 are compared here." in f.summary
    assert f.evidence["tables_live"] == 3 and f.evidence["older"]["n"] + f.evidence["newer"]["n"] == 350


def test_tables_that_differ_only_in_which_rows_are_graded():
    pos, up, lo = COARSE
    ends_ignored = (pos, (None, None) + up[2:-2] + (None, None), (None, None) + lo[2:-2] + (None, None))
    (f,) = _run(table_era(0, START, 200, COARSE, 0.5) + table_era(1000, START, 200, ends_ignored, 0.7))[1]
    assert f.evidence["comparison"]["kind"] == "same_band_other_density"      # 12 graded against 8: say so
    assert "12 points and the other 8" in f.summary
    same_count = (pos, up[:6] + (None,) + up[7:], lo[:6] + (None,) + lo[7:])
    other_gap = (pos, up[:3] + (None,) + up[4:], lo[:3] + (None,) + lo[4:])
    (g,) = _run(table_era(0, START, 200, same_count, 0.5) + table_era(1000, START, 200, other_gap, 0.5))[1]
    assert g.evidence["comparison"]["kind"] == "same_band_same_points"
    assert "the difference is elsewhere in the table" in g.summary and "different REQUIREMENTS" not in g.summary


def test_tables_with_no_positions_are_found_but_not_compared():
    from dataclasses import replace
    a = [replace(t, final_positions=None) for t in table_era(0, START, 200, COARSE, 0.5)]
    b = [replace(t, final_positions=None) for t in table_era(1000, START, 200, LOOSE, 0.9)]
    (f,) = _run(a + b)[1]
    assert f.evidence["comparison"]["kind"] == "not_comparable" and "too few positions to compare" in f.summary


def test_a_side_with_no_verdicts_says_nothing_about_its_pass_rate():
    from dataclasses import replace
    from datetime import timedelta
    blind = [replace(t, linearity_pass=None) for t in table_era(1000, START + timedelta(days=10), 200, DENSE, 0.5)]
    (f,) = _run(table_era(0, START, 200, COARSE, 0.6) + blind)[1]
    assert f.evidence["older"]["trim_pass_pct"] == 60.0 and f.evidence["newer"]["trim_pass_pct"] is None
    assert f.summary.count("left the laser inside limits") == 1                # said once, for the side that has verdicts


def test_the_band_compared_is_the_one_most_files_carry_not_the_first_files():
    """One table, many files: a single file with a shifted position column must not decide the comparison."""
    from dataclasses import replace
    odd_positions = tuple(x + 300.0 for x in COARSE[0])
    a = table_era(0, START, 200, COARSE, 0.5)
    a[0] = replace(a[0], final_positions=odd_positions)                        # the FIRST file is the odd one
    (f,) = _run(a + table_era(1000, START, 200, LOOSE, 0.9))[1]
    assert f.evidence["comparison"]["kind"] == "different_band" and f.evidence["comparison"]["wider"] == 12


def test_a_table_is_its_content_not_its_float_noise():
    from laser_trim_analyzer.findings.data import limit_table_of
    pos, up, lo = table(12, 0.10)
    noisy = limit_table_of(pos, tuple(u + 2e-7 for u in up), lo)
    assert noisy.key == limit_table_of(pos, up, lo).key
    assert limit_table_of(pos, tuple(u + 1e-3 for u in up), lo).key != noisy.key
    blanked = limit_table_of(pos, (None, None) + up[2:], (None, None) + lo[2:])
    assert (blanked.rows, blanked.graded) == (12, 10) and blanked.key != noisy.key     # ignored rows ARE part of the table
    assert limit_table_of(pos, up[:5], lo) is None and limit_table_of(pos, None, lo) is None
    assert limit_table_of(pos, (None,) * 12, (None,) * 12) is None
    assert limit_table_of(None, up, lo).band == ()                                     # no positions: a key, but no band


def test_a_track_with_no_usable_limits_is_ignored():
    bare = on_table(make_track(0, date=START), (None, None, None))
    assert bare.limit_table is None
    assert _run([bare] * 40) == ([], [])
