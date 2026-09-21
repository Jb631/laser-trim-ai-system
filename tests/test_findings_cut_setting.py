"""The cut-setting analyzer: it must find a real difference and refuse every fake one.

Every silence test here was made to FAIL first by relaxing the control it names --
a green test that cannot go red proves nothing.
"""
import pytest

from laser_trim_analyzer.findings.analyzers import cut_setting
from findings_helpers import START, days, label, make_track, on_table, table

TAB = table(12)
OTHER = table(12, half=0.25)        # a genuinely different test: wider bands


def track(i, d, cut, r_in, good, *, tab=TAB, system="B", name="Track A"):
    return on_table(make_track(i, date=d, passes=((0.8 if good else 1.5, cut),),
                               r_in=r_in, system=system), tab, name)


BUCKETS = 20        # 5% steps. At 10 a share of 0.35 rendered as 40%, identical to 0.40,
                    # and the "too small to act on" test silently had a gain of ZERO -- it
                    # passed the gain floor it was meant to exercise without ever reaching it.


def block(first_id, start, n, cut, share_low, share_high, *, tab=TAB, system="B", name="Track A"):
    """`n` tracks on one cut setting, half at 4,000 ohm and half at 5,000.

    `share_low` / `share_high` are the pass rates inside each resistance half, so a
    test can make a setting win overall while losing a half -- which is exactly the
    confound the analyzer has to refuse. Each half counts independently, so the rate
    a test asks for is the rate it gets (shares must be multiples of 1/BUCKETS).
    """
    out, seen = [], {True: 0, False: 0}
    for k, d in enumerate(days(start, n)):
        low = k % 2 == 0
        share = share_low if low else share_high
        c = seen[low]
        seen[low] = c + 1
        out.append(track(first_id + k, d, cut, 4000.0 if low else 5000.0,
                         c % BUCKETS < share * BUCKETS, tab=tab, system=system, name=name))
    return out


def two_blocks(share_a, share_b, *, cut_a=1.0, cut_b=2.0, n=120, highs=None):
    """`cut_a` early, `cut_b` recent -- so `cut_b` is what the laser is set to now."""
    la, ha = (share_a, share_a) if highs is None else highs[0]
    lb, hb = (share_b, share_b) if highs is None else highs[1]
    return (block(0, START, n, cut_a, la, ha)
            + block(1000, days(START, 201)[-1], n, cut_b, lb, hb))


def only(findings):
    assert len(findings) == 1, [f.title for f in findings]
    return findings[0]


def test_a_better_setting_that_wins_in_both_resistance_halves_is_reported():
    facts, findings = cut_setting.analyze("M", two_blocks(0.6, 0.3), label)
    f = only(findings)
    assert f.analyzer == "cut_setting" and f.lever == "laser_settings"
    assert f.evidence["best"] == 1.0 and f.evidence["current"] == 2.0
    assert f.expected_gain_points == pytest.approx(30.0)
    assert "cut 1 passed 30 points more often" in f.title
    # the group's own population, never the model's total
    assert f.scope_annual_tracks == 240
    assert f.tracks_per_year == pytest.approx(72.0)
    assert facts["Laser 1 (LTS) · Track A"]["current_setting"] == 2.0


def test_it_refuses_a_winner_that_loses_one_half_of_incoming_resistance():
    # 1.0 wins overall (45% vs 30%) but only because it is good on low-resistance
    # units; at high resistance 2.0 is better. That is the ink talking, not the cut.
    tracks = two_blocks(0, 0, highs=((0.8, 0.1), (0.2, 0.4)))
    facts, findings = cut_setting.analyze("M", tracks, label)
    assert findings == []
    rates = {s["setting"]: s["pass_pct"] for s in facts["Laser 1 (LTS) · Track A"]["settings"]}
    assert rates[1.0] > rates[2.0]              # the tempting comparison is still visible in facts


def test_the_setting_already_running_being_the_best_one_says_nothing(monkeypatch):
    # 8340-1's real shape: the shop moved to the best setting in January 2025.
    # There is nothing to recommend, so the analyzer must not manufacture one.
    # The gain floor is disabled here on purpose -- otherwise this passes on the
    # floor alone and proves nothing about "already on the best one".
    #
    # Three independent nets catch this case: the `best == current` guard, the gain
    # floor, and the strict `>` in the halves control. Removing any ONE leaves this
    # test green; removing all three turns it red (checked 2026-09-20). That is
    # defence in depth, not dead code -- do not delete one because it looks unused.
    monkeypatch.setattr(cut_setting, "MIN_GAIN_POINTS", 0.0)
    facts, findings = cut_setting.analyze("M", two_blocks(0.3, 0.6), label)
    assert findings == []
    assert facts["Laser 1 (LTS) · Track A"]["current_setting"] == 2.0
    rates = {s["setting"]: s["pass_pct"] for s in facts["Laser 1 (LTS) · Track A"]["settings"]}
    assert rates[2.0] > rates[1.0]              # the one running IS the best of the two


def test_a_difference_too_small_to_act_on_says_nothing():
    facts, findings = cut_setting.analyze("M", two_blocks(0.4, 0.35), label)
    assert findings == []
    assert len(facts["Laser 1 (LTS) · Track A"]["settings"]) == 2


def test_two_limit_tables_are_never_compared_with_each_other():
    # Each table saw exactly one setting. Comparing them would compare two TESTS --
    # the mistake that had to be corrected by hand on 8340-1.
    tracks = (block(0, START, 120, 1.0, 0.6, 0.6, tab=TAB)
              + block(1000, days(START, 201)[-1], 120, 2.0, 0.3, 0.3, tab=OTHER))
    facts, findings = cut_setting.analyze("M", tracks, label)
    assert findings == []
    assert facts == {}


def test_two_lasers_are_never_pooled():
    # laser_cut_length is 0.55-0.88 on laser 2 and 2500-4341 on laser 1: different
    # quantities. One setting per machine must therefore yield no comparison at all.
    tracks = (block(0, START, 120, 0.75, 0.6, 0.6, system="A")
              + block(1000, days(START, 201)[-1], 120, 4000.0, 0.3, 0.3, system="B"))
    facts, findings = cut_setting.analyze("M", tracks, label)
    assert findings == []
    assert facts == {}


def test_each_laser_keeps_its_own_answer():
    tracks = (block(0, START, 120, 0.75, 0.6, 0.6, system="A")
              + block(1000, days(START, 201)[-1], 120, 0.88, 0.3, 0.3, system="A")
              + block(2000, START, 120, 4000.0, 0.6, 0.6, system="B")
              + block(3000, days(START, 201)[-1], 120, 4100.0, 0.3, 0.3, system="B"))
    _, findings = cut_setting.analyze("M", tracks, label)
    assert {f.systems for f in findings} == {("A",), ("B",)}
    assert {f.evidence["best"] for f in findings} == {0.75, 4000.0}


def test_two_track_names_on_one_laser_are_kept_apart():
    # A model's tracks may carry different limits; Track A's answer is not Track B's.
    tracks = (block(0, START, 120, 1.0, 0.6, 0.6, name="Track A")
              + block(1000, days(START, 201)[-1], 120, 2.0, 0.3, 0.3, name="Track A")
              + block(2000, START, 120, 1.0, 0.3, 0.3, name="Track B")
              + block(3000, days(START, 201)[-1], 120, 2.0, 0.6, 0.6, name="Track B"))
    facts, findings = cut_setting.analyze("M", tracks, label)
    assert set(facts) == {"Laser 1 (LTS) · Track A", "Laser 1 (LTS) · Track B"}
    assert only(findings).evidence["best"] == 1.0      # only Track A has something to say


def test_a_clean_before_and_after_names_the_month_it_changed():
    f = only(cut_setting.analyze("M", two_blocks(0.6, 0.3), label)[1])
    assert f.evidence["changeover"] == "2024-07"
    assert "moved from 1 to 2 in 2024-07" in f.summary
    assert "comparison of two PERIODS" in f.summary
    assert "Worth asking why the setting changed in 2024-07" in f.summary


def test_settings_that_ran_side_by_side_are_not_called_two_periods():
    # Both settings run EVERY day, the worse one twice as often -- so it is what the
    # laser is mostly set to, and a change of material over the months cannot explain
    # the difference. The summary must not claim it might.
    seen = {}
    tracks = []
    i = 0
    for d in days(START, 120):
        for cut in (1.0, 2.0, 2.0):
            share = 0.6 if cut == 1.0 else 0.3
            low = seen.get(cut, 0) % 2 == 0
            seen[cut] = seen.get(cut, 0) + 1
            k = seen.get((cut, low), 0)
            seen[(cut, low)] = k + 1
            tracks.append(track(i, d, cut, 4000.0 if low else 5000.0, k % 10 < share * 10))
            i += 1
    _, findings = cut_setting.analyze("M", tracks, label)
    f = only(findings)
    assert f.evidence["best"] == 1.0 and f.evidence["current"] == 2.0
    assert f.evidence["changeover"] is None
    assert "not a comparison of two periods" in f.summary
    assert f.evidence["days_mixed_pct"] == pytest.approx(100.0)


def test_a_sample_too_thin_to_judge_says_nothing():
    facts, findings = cut_setting.analyze("M", two_blocks(0.6, 0.3, n=20), label)
    assert findings == []
    assert facts == {}                      # 20 a side is below MIN_PER_SETTING


def test_one_setting_only_is_not_a_comparison():
    facts, findings = cut_setting.analyze("M", block(0, START, 240, 1.0, 0.6, 0.6), label)
    assert findings == [] and facts == {}


def test_tracks_with_no_cut_recorded_are_ignored_rather_than_counted_as_a_setting():
    tracks = two_blocks(0.6, 0.3) + [
        on_table(make_track(9000 + k, date=d, passes=((1.5, None),)), TAB)
        for k, d in enumerate(days(START, 60))]
    f = only(cut_setting.analyze("M", tracks, label)[1])
    assert {s["setting"] for s in f.evidence["group"]["settings"]} == {1.0, 2.0}


def test_a_setting_older_than_the_lookback_is_not_offered_as_a_recommendation():
    old = block(0, days(START, 1)[0].replace(year=2019), 120, 1.0, 0.9, 0.9)
    recent = block(1000, START, 240, 2.0, 0.3, 0.3)
    facts, findings = cut_setting.analyze("M", old + recent, label)
    assert findings == []
    assert facts == {}                      # the 2019 block is outside LOOKBACK_DAYS entirely
