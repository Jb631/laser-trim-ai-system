"""Multi-pass burden must count cuts the RECIPE did not ask for, never raw pass counts.

The trap: "86% of 8232-1 needs a second pass" was a recipe change, not parts failing.
An analyzer that counts passes without knowing each recipe's own normal reports every
two-cut model as a problem. Every silence rule below was made to fail first.
"""
import pytest

from laser_trim_analyzer.findings.analyzers import pass_burden
from findings_helpers import START, days, label, make_track


def run(first_id, n, cuts_per_track, *, setting=1.0, system="B", name=None, start=START, step=1.0):
    """`n` tracks; cuts_per_track(k) says how many cuts track k received.

    `step` matters: the analyzer only looks at the last 365 days, so a block that
    spans longer than that is partly out of scope and the group can vanish entirely.
    """
    out = []
    for k, d in enumerate(days(start, n, step)):
        ps = tuple((1.5, setting) for _ in range(cuts_per_track(k)))
        t = make_track(first_id + k, date=d, passes=ps, system=system)
        if name is not None:
            from dataclasses import replace
            t = replace(t, track_name=name)
        out.append(t)
    return out


def test_a_two_cut_recipe_is_not_a_burden():
    # Every track gets exactly two cuts because that IS the process. Reporting this
    # would be reporting the process sheet.
    facts, findings = pass_burden.analyze("M", run(0, 300, lambda k: 2), label)
    assert findings == []
    f = facts["Laser 1 (LTS) · default · cut 1"]
    assert f["normal_cuts"] == 2 and f["share_over_recipe"] == 0.0


def test_cuts_on_top_of_the_recipe_are_reported():
    # Normal is one cut; a third of tracks take a second.
    facts, findings = pass_burden.analyze("M", run(0, 300, lambda k: 2 if k % 3 == 0 else 1), label)
    assert len(findings) == 1
    f = findings[0]
    assert f.analyzer == "pass_burden" and f.lever == "laser_settings"
    assert f.expected_gain_points is None          # capacity, never a yield claim
    assert f.tracks_per_year is None
    assert f.strength_value == pytest.approx(33.3, abs=0.2)
    assert "more than the 1 cut the recipe asks for" in f.title
    assert facts["Laser 1 (LTS) · default · cut 1"]["normal_cuts"] == 1


def test_the_same_extra_passes_under_a_two_cut_recipe_say_nothing():
    # Identical EXTRA work to the test above (a third of tracks take one more cut than
    # normal) -- but normal is 2 here, so raw pass counts would look far worse while the
    # burden is the same. This is the pair that pins "against the recipe, not absolute".
    _, over_one = pass_burden.analyze("M", run(0, 300, lambda k: 2 if k % 3 == 0 else 1), label)
    _, over_two = pass_burden.analyze("M", run(0, 300, lambda k: 3 if k % 3 == 0 else 2), label)
    assert over_one[0].strength_value == over_two[0].strength_value


def test_a_recipe_change_is_not_pooled_into_one_verdict():
    # Two cut settings, each internally consistent: one-cut recipe then two-cut recipe.
    # Pooled, the mode would be one and every two-cut track would read as unplanned.
    tracks = (run(0, 250, lambda k: 1, setting=1.0, step=0.5)
              + run(1000, 250, lambda k: 2, setting=2.0, step=0.5, start=days(START, 131)[-1]))
    facts, findings = pass_burden.analyze("M", tracks, label)
    assert findings == []
    assert facts["Laser 1 (LTS) · default · cut 1"]["normal_cuts"] == 1
    assert facts["Laser 1 (LTS) · default · cut 2"]["normal_cuts"] == 2


def test_each_laser_is_judged_on_its_own():
    tracks = (run(0, 300, lambda k: 1, system="A")
              + run(1000, 300, lambda k: 2 if k % 3 == 0 else 1, system="B"))
    _, findings = pass_burden.analyze("M", tracks, label)
    assert [f.systems for f in findings] == [("B",)]
    # n_units is what pooling would change: laser B alone is 300, both lasers 600.
    # Without it this passes even when the two lasers are merged.
    assert findings[0].n_units == 300


def test_track_names_are_kept_apart():
    tracks = (run(0, 300, lambda k: 1, name="Track A")
              + run(1000, 300, lambda k: 2 if k % 3 == 0 else 1, name="Track B"))
    facts, findings = pass_burden.analyze("M", tracks, label)
    assert set(facts) == {"Laser 1 (LTS) · Track A · cut 1", "Laser 1 (LTS) · Track B · cut 1"}
    assert len(findings) == 1 and "Track B" not in findings[0].title    # title names the laser


def test_a_sideshow_recipe_carries_no_recommendation_about_the_laser():
    # 240 tracks on a bad recipe beside 1,400 on a good one: under a fifth of the
    # laser's work, so it cannot carry a claim about the laser's capacity.
    tracks = (run(0, 1400, lambda k: 1, setting=1.0, step=0.25)
              + run(5000, 240, lambda k: 2 if k % 3 == 0 else 1, setting=2.0, step=0.25))
    facts, findings = pass_burden.analyze("M", tracks, label)
    assert findings == []
    # It IS a real burden -- a third of that recipe's tracks take a cut above its
    # normal of 1 -- and it is suppressed ONLY because the recipe is a small slice of
    # the laser. Without a real burden here this test could not reach that guard.
    f = facts["Laser 1 (LTS) · default · cut 2"]
    assert f["n"] == 240 and f["normal_cuts"] == 1 and f["share_over_recipe"] > 30.0


def test_an_evenly_split_group_never_manufactures_a_burden():
    # Exactly half take one cut and half take two: "normal" is a tie. Counter breaks
    # ties by insertion order, which would make the answer depend on which track was
    # read first. The larger count wins, deterministically, so a tie reports nothing.
    # The SMALLER count is inserted first on purpose: that is the order in which
    # most_common would pick 1 as normal and report a 50% burden out of a coin toss.
    facts, findings = pass_burden.analyze("M", run(0, 300, lambda k: 1 if k % 2 == 0 else 2), label)
    assert findings == []
    f = facts["Laser 1 (LTS) · default · cut 1"]
    assert f["normal_cuts"] == 2 and f["share_over_recipe"] == 0.0


def test_a_thin_group_says_nothing_at_all():
    facts, findings = pass_burden.analyze("M", run(0, 100, lambda k: 2 if k % 2 else 1), label)
    assert findings == [] and facts == {}


def test_extra_passes_just_below_the_threshold_say_nothing():
    facts, findings = pass_burden.analyze("M", run(0, 300, lambda k: 2 if k % 10 == 0 else 1), label)
    assert findings == []
    assert facts["Laser 1 (LTS) · default · cut 1"]["share_over_recipe"] == 10.0


def test_only_the_last_year_counts():
    old = run(0, 400, lambda k: 3, setting=1.0, start=START.replace(year=START.year - 3))
    new = run(5000, 300, lambda k: 1, setting=1.0)
    facts, findings = pass_burden.analyze("M", old + new, label)
    assert findings == []
    assert facts["Laser 1 (LTS) · default · cut 1"]["n"] == 300     # the 3-cut era is out of window


def test_tracks_that_were_never_cut_are_not_counted_as_a_zero_cut_recipe():
    tracks = run(0, 300, lambda k: 1) + [make_track(9000 + k, date=d, passes=())
                                         for k, d in enumerate(days(START, 100))]
    facts, findings = pass_burden.analyze("M", tracks, label)
    assert findings == []
    assert facts["Laser 1 (LTS) · default · cut 1"]["n"] == 300
    assert "0" not in facts["Laser 1 (LTS) · default · cut 1"]["cut_counts"]


def test_the_finding_names_its_track():
    # Two tracks' findings can read identically; the Findings page tells their rows apart by this.
    tracks = run(0, 300, lambda k: 2 if k % 3 == 0 else 1, name="Track B")
    _, findings = pass_burden.analyze("M", tracks, label)
    assert findings[0].evidence["track"] == "Track B"
