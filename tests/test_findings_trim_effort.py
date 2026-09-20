import pytest

from laser_trim_analyzer.findings.analyzers import trim_effort
from findings_helpers import START, days, make_track, label


def test_units_arriving_in_spec_below_the_resistance_floor_point_at_the_ink():
    ds = days(START, 200)
    tracks = [make_track(i, date=d, system="A", untrimmed_worst=0.6 if i % 4 == 0 else 2.0,
                         passes=((0.7, 0.75),), r_in=4300.0) for i, d in enumerate(ds)]
    facts, findings = trim_effort.analyze("M", tracks, label)
    assert facts["A"]["arrive_in_spec_pct"] == pytest.approx(25.0)
    f = [x for x in findings if x.category == "Trim avoidance"]
    assert len(f) == 1 and f[0].lever == "ink" and f[0].expected_gain_points is None

def test_a_model_where_nothing_arrives_in_spec_says_nothing():
    tracks = [make_track(i, date=d, untrimmed_worst=2.5, passes=((1.4, 1.0), (0.8, 2.0)))
              for i, d in enumerate(days(START, 200))]
    facts, findings = trim_effort.analyze("M", tracks, label)
    assert findings == []
    assert facts["B"]["arrive_in_spec_pct"] == 0.0
    assert facts["B"]["multi_in_limits_after_first_pct"] == 0.0
    assert facts["B"]["multi_in_limits_after_last_pct"] == 100.0

def test_later_cuts_that_add_nothing_are_a_finding():
    tracks = [make_track(i, date=d, untrimmed_worst=2.5,
                         passes=((1.4, 1.0), (1.4, 2.0)) if i % 2 else ((0.8, 1.0), (0.8, 2.0)))
              for i, d in enumerate(days(START, 200))]
    _, findings = trim_effort.analyze("M", tracks, label)
    f = [x for x in findings if x.category == "Pass effectiveness"]
    assert len(f) == 1 and f[0].lever == "laser_settings"

def test_a_thin_sample_says_nothing_but_still_reports_the_facts():
    # 25% arrive in spec and the later cuts add nothing -- both would be findings at n=200 --
    # but 60 tracks is not enough to say so.
    tracks = [make_track(i, date=d, system="A", untrimmed_worst=0.6 if i % 4 == 0 else 2.0,
                         passes=((1.4, 0.75), (1.4, 0.88)), r_in=4300.0) for i, d in enumerate(days(START, 60))]
    facts, findings = trim_effort.analyze("M", tracks, label)
    assert findings == []
    assert facts["A"]["arrive_in_spec_pct"] == 25.0 and facts["A"]["multi_cut_n"] == 60
