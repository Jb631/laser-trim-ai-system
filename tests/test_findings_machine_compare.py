"""Machine comparison: the same model, the same test, the same months, two lasers.

Every silence test here was made to FAIL first by relaxing the control it names -- a green
test that cannot go red proves nothing.
"""
from dataclasses import replace
from datetime import datetime

import pytest

from laser_trim_analyzer.findings.analyzers import machine_compare
from findings_helpers import START, days, label, make_track, on_table, table

TAB = table(12)
OTHER = table(12, half=0.25)        # a genuinely different test: wider bands


def block(first_id, start, n, system, pass_share, *, tab=TAB, name="Track A", n_days=10):
    """`n` tracks for one laser, spread round-robin over `n_days` consecutive days from `start`
    -- so the track count and the calendar months it spans can be set independently."""
    ds = days(start, n_days)
    out = []
    for k in range(n):
        good = (k % 100) < pass_share * 100
        out.append(on_table(make_track(first_id + k, date=ds[k % n_days], system=system,
                                       linearity_pass=good), tab, name))
    return out


def only(findings):
    assert len(findings) == 1, [f.title for f in findings]
    return findings[0]


def comparisons(facts):
    return facts["comparisons"]


def one_comparison(facts):
    assert len(facts["comparisons"]) == 1, facts["comparisons"]
    return facts["comparisons"][0]


def test_two_lasers_same_table_same_months_a_real_gap_is_one_finding():
    tracks = block(0, START, 200, "A", 0.95) + block(1000, START, 200, "B", 0.75)
    facts, findings = machine_compare.analyze("M", tracks, label)
    f = only(findings)
    assert f.analyzer == "machine_compare" and f.lever == "laser_settings"
    assert f.category == "Machine comparison"
    assert f.expected_gain_points is None
    assert "Laser 2 (DLTS) passes 95%" in f.title
    assert "Laser 1 (LTS) 75%" in f.title
    assert set(f.systems) == {"A", "B"}
    assert f.n_units == 400


def test_facts_and_evidence_carry_the_documented_shape():
    tracks = block(0, START, 200, "A", 0.95) + block(1000, START, 200, "B", 0.75)
    facts, findings = machine_compare.analyze("M", tracks, label)
    f = only(findings)
    table_key = tracks[0].limit_table.key
    # With no fleet anchor the window ends at the model's own latest graded track (2024-01-10).
    assert facts["window"] == {"first": "2022-02", "last": "2024-01", "months": 24,
                               "anchored_to": "2024-01-10"}
    c = one_comparison(facts)
    assert c["table"] == table_key and c["in_window"] is True and c["graded_points"] == 12
    assert c["months"] == ["2024-01"]
    by_laser = c["by_laser"]
    assert by_laser["Laser 2 (DLTS)"] == {"n": 200, "pass_pct": pytest.approx(95.0)}
    assert by_laser["Laser 1 (LTS)"] == {"n": 200, "pass_pct": pytest.approx(75.0)}
    assert f.evidence["table"] == table_key
    assert f.evidence["months"] == ["2024-01", "2024-01"]
    assert f.evidence["window"] == ["2022-02", "2024-01"]
    assert f.evidence["by_laser"] == by_laser
    assert f.evidence["best_laser"] == "A" and f.evidence["worst_laser"] == "B"


def test_different_limit_tables_are_never_compared():
    # Each laser saw exactly one table -- comparing them would compare two TESTS, the mistake
    # the domain rule "a pass rate is a verdict against a test" exists to rule out.
    tracks = block(0, START, 200, "A", 0.95, tab=TAB) + block(1000, START, 200, "B", 0.75, tab=OTHER)
    facts, findings = machine_compare.analyze("M", tracks, label)
    assert findings == [] and comparisons(facts) == []


def test_different_months_are_never_pooled_as_shared():
    # Laser 2 (DLTS) all in 2024, laser 1 (LTS) all in 2025 -- no month has both, so nothing
    # is ever pooled together even though both ran the very same limit table.
    tracks = (block(0, START, 200, "A", 0.95)
              + block(1000, datetime(2025, 1, 1), 200, "B", 0.75))
    facts, findings = machine_compare.analyze("M", tracks, label)
    assert findings == [] and comparisons(facts) == []


def test_a_gap_under_ten_points_says_nothing():
    tracks = block(0, START, 200, "A", 0.86) + block(1000, START, 200, "B", 0.77)   # 9-point gap
    facts, findings = machine_compare.analyze("M", tracks, label)
    assert findings == []
    # Under MIN_GAP_POINTS is still a COMPARABLE measurement: both lasers cleared
    # MIN_TRACKS_PER_LASER over the same shared month, so it is a FACT, just never a finding --
    # the population floor gates facts, the gap only gates a finding (review fix, 2026-09-24).
    c = one_comparison(facts)
    assert c["months"] == ["2024-01"] and c["in_window"] is True
    assert c["by_laser"] == {
        "Laser 2 (DLTS)": {"n": 200, "pass_pct": pytest.approx(86.0)},
        "Laser 1 (LTS)": {"n": 200, "pass_pct": pytest.approx(77.0)},
    }


def test_one_laser_under_the_track_floor_says_nothing():
    # A big gap (95 vs 50), but laser A never reaches the 100-track floor. The two thresholds
    # are independent -- either alone must be enough to silence this.
    tracks = block(0, START, 99, "A", 0.95) + block(1000, START, 300, "B", 0.50)
    facts, findings = machine_compare.analyze("M", tracks, label)
    assert findings == [] and comparisons(facts) == []


def test_ungraded_tracks_are_not_counted():
    base = block(0, START, 200, "A", 0.95) + block(1000, START, 200, "B", 0.75)
    # 500 more tracks on laser A that would swing the rate hard if ever counted either way.
    extra = [replace(t, linearity_pass=None) for t in block(2000, START, 500, "A", 0.05)]
    facts, findings = machine_compare.analyze("M", base + extra, label)
    f = only(findings)
    assert f.evidence["by_laser"]["Laser 2 (DLTS)"]["n"] == 200
    assert f.evidence["by_laser"]["Laser 2 (DLTS)"]["pass_pct"] == pytest.approx(95.0)


def test_one_laser_only_is_not_a_comparison():
    tracks = block(0, START, 300, "B", 0.5)
    facts, findings = machine_compare.analyze("M", tracks, label)
    assert findings == [] and comparisons(facts) == []


def test_no_graded_tracks_is_no_facts_at_all():
    assert machine_compare.analyze("M", [], label) == ({}, [])


def test_a_third_laser_under_the_floor_is_left_out_of_the_comparison():
    # 6952's real shape (measured 2026-09-24, corrected in 0afa628 after the pool-by-track-name
    # bug -- see machine_compare.py's own docstring): laser 2 (DLTS) 95% of 288, laser 1 (LTS)
    # 84% of 221. A third laser also ran this table and months but stayed under the 100-track
    # floor. It must not be averaged in, and must not accidentally become best or worst.
    tracks = (block(0, START, 200, "A", 0.96)
              + block(1000, START, 200, "B", 0.84)
              + block(2000, START, 90, "C", 0.99))
    facts, findings = machine_compare.analyze("M", tracks, label)
    f = only(findings)
    assert f.evidence["best_laser"] == "A" and f.evidence["worst_laser"] == "B"
    assert "Laser 3" not in f.title
    assert set(one_comparison(facts)["by_laser"]) == {"Laser 2 (DLTS)", "Laser 1 (LTS)"}


# ---- I1 (final review, 2026-09-25): recent months only, anchored to the FLEET's latest file ----
# Three of the five real findings described 2013-2016 and read as current: 8081-4's last file is
# 2016-03, yet anchored to its OWN latest year it still qualified.

NOW = datetime(2026, 9, 22)          # the fleet's newest trim file, as the engine passes it


def test_a_pair_entirely_before_the_window_is_a_dated_fact_not_a_finding():
    tracks = (block(0, datetime(2015, 5, 1), 200, "A", 0.95)
              + block(1000, datetime(2015, 5, 1), 200, "B", 0.60))
    facts, findings = machine_compare.analyze("M", tracks, label, now=NOW)
    assert findings == []
    c = one_comparison(facts)
    assert c["in_window"] is False and c["months"] == ["2015-05"]
    assert c["by_laser"]["Laser 1 (LTS)"]["pass_pct"] == pytest.approx(60.0)   # the numbers stay
    assert facts["window"] == {"first": "2024-10", "last": "2026-09", "months": 24,
                               "anchored_to": "2026-09-22"}


def test_a_pair_inside_the_window_is_a_finding_whose_title_names_the_months():
    tracks = (block(0, datetime(2025, 10, 1), 200, "A", 0.95, n_days=40)
              + block(1000, datetime(2025, 10, 1), 200, "B", 0.60, n_days=40))
    f = only(machine_compare.analyze("M", tracks, label, now=NOW)[1])
    assert f.title == "Laser 2 (DLTS) passes 95%, Laser 1 (LTS) 60%, same test, Oct 2025 – Nov 2025"
    assert f.evidence["months"] == ["2025-10", "2025-11"]


def test_one_shared_month_is_named_once():
    tracks = (block(0, datetime(2025, 10, 1), 200, "A", 0.95)
              + block(1000, datetime(2025, 10, 1), 200, "B", 0.60))
    f = only(machine_compare.analyze("M", tracks, label, now=NOW)[1])
    assert f.title.endswith("same test, Oct 2025")


def test_the_windows_first_month_counts_and_the_month_before_it_does_not():
    assert machine_compare.WINDOW_MONTHS == 24
    first = (block(0, datetime(2024, 10, 1), 200, "A", 0.95)
             + block(1000, datetime(2024, 10, 1), 200, "B", 0.60))
    assert len(machine_compare.analyze("M", first, label, now=NOW)[1]) == 1
    before = (block(0, datetime(2024, 9, 1), 200, "A", 0.95)
              + block(1000, datetime(2024, 9, 1), 200, "B", 0.60))
    facts, findings = machine_compare.analyze("M", before, label, now=NOW)
    assert findings == [] and one_comparison(facts)["in_window"] is False


def test_a_table_shared_both_before_and_inside_the_window_is_two_comparisons():
    """The finding pools ONLY the recent months: 2015's 50% on laser 1 must not dilute it."""
    old = (block(0, datetime(2015, 5, 1), 200, "A", 0.95)
           + block(1000, datetime(2015, 5, 1), 200, "B", 0.50))
    new = (block(2000, datetime(2025, 10, 1), 200, "A", 0.95)
           + block(3000, datetime(2025, 10, 1), 200, "B", 0.80))
    facts, findings = machine_compare.analyze("M", old + new, label, now=NOW)
    f = only(findings)
    assert f.evidence["by_laser"]["Laser 1 (LTS)"] == {"n": 200, "pass_pct": pytest.approx(80.0)}
    assert [(c["in_window"], c["months"]) for c in comparisons(facts)] == [
        (False, ["2015-05"]), (True, ["2025-10"])]


def test_with_no_fleet_anchor_the_window_ends_at_the_models_own_latest():
    tracks = (block(0, datetime(2015, 5, 1), 200, "A", 0.95)
              + block(1000, datetime(2015, 5, 1), 200, "B", 0.60))
    assert len(machine_compare.analyze("M", tracks, label)[1]) == 1
