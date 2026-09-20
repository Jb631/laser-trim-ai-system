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
    assert "the limit table changed" in f.title and "around 2024-05-10" in f.summary
    assert f.summary.index("Before:") < f.summary.index("After:")
    assert f.evidence["older"]["trim_pass_pct"] == 70.0 and f.evidence["newer"]["trim_pass_pct"] == 100.0
    c = f.evidence["comparison"]
    assert c["kind"] == "different_band" and c["wider"] == 12 and c["narrower"] == 0 and c["max_abs_diff"] == 0.2
    assert "WIDER at 12" in f.summary and "different REQUIREMENTS" in f.summary
    assert f.evidence["concurrent"] is False


def test_a_handful_of_odd_files_is_not_a_second_table():
    tracks = table_era(0, START, 300, COARSE, 0.5) + table_era(1000, START, 29, DENSE, 0.5)   # 29 < MIN_TABLE_N
    history, findings = _run(tracks)
    assert findings == [] and len(history) == 1


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


def test_bands_are_compared_by_interpolation_not_by_matching_positions():
    from laser_trim_analyzer.findings.analyzers.limit_tables import compare
    from laser_trim_analyzer.findings.data import limit_table_of
    bow_a = limit_table_of(*table(11, 0.05, end_half=0.15))                  # positions -10, -8, ... 10
    shifted = limit_table_of(*table(11, 0.05, span=(-9.0, 9.0)))             # a grid that shares NO position with it
    c = compare(bow_a, shifted)
    assert c["compared_positions"] == 11
    assert c["kind"] == "different_band" and c["narrower"] == 2              # only where bow_a's ends flare out
    far = limit_table_of(*table(11, 0.05, span=(100.0, 120.0)))
    assert compare(bow_a, far) == {"graded_old": 11, "graded_new": 11, "compared_positions": 0,
                                   "kind": "not_comparable"}


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
