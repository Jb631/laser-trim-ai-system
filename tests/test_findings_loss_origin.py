"""Where the loss is made: does the incoming sweep already predict the laser's verdict?

Every silence test here was made to FAIL first by relaxing the control it names -- a green
test that cannot go red proves nothing.
"""
from dataclasses import replace
from datetime import timedelta

import pytest

from laser_trim_analyzer.findings.analyzers import loss_origin
from findings_helpers import START, days, label, make_track


def scored_tracks(first_id, start, n, system, worst, outcome, n_days=10):
    """`n` tracks, all graded `outcome` (True=PASS, False=FAIL), all with untrimmed max|error| =
    `worst` x band -- spread round-robin over `n_days` consecutive days (default 10) regardless
    of `n`, so any combination of these blocks lands inside one LOOKBACK_DAYS window."""
    ds = days(start, n_days)
    return [make_track(first_id + k, date=ds[k % n_days], system=system,
                       untrimmed_worst=worst, linearity_pass=outcome)
            for k in range(n)]


def mixed(first_id, start, n_each, system, *, hi, lo, hi_share):
    """`n_each` FAILs and `n_each` PASSes on one laser: FAILs are `hi_share` of them at `hi`
    (the rest at `lo`); PASSes are the MIRROR IMAGE -- `hi_share` at `lo`, the rest at `hi` -- so
    the two outcomes overlap by construction instead of separating perfectly. Worked by hand for
    hi_share=0.9: P(fail>pass) = .9*.9 = .81, ties (fail=pass) = .9*.1 + .1*.9 = .18 worth .09,
    total AUC = .81 + .09 = .90.
    """
    n_hi = round(n_each * hi_share)
    n_lo = n_each - n_hi
    fails = (scored_tracks(first_id, start, n_hi, system, hi, False, n_days=10)
            + scored_tracks(first_id + n_hi, start, n_lo, system, lo, False, n_days=10))
    # Mirror image of fails: the n_hi-sized share sits at LO here, and the n_lo-sized share at
    # HI -- e.g. hi_share=0.9 puts 90% of fails at hi and 90% of passes at lo.
    passes = (scored_tracks(first_id + 2 * n_each, start, n_hi, system, lo, True, n_days=10)
             + scored_tracks(first_id + 2 * n_each + n_hi, start, n_lo, system, hi, True, n_days=10))
    return fails + passes


def exact_split(first_id, start, n_each, system, n_hi, *, hi, lo):
    """Like `mixed()`, but takes `n_hi` directly instead of rounding `hi_share * n_each` -- AUC =
    n_hi / n_each EXACTLY (worked by hand from auc()'s rank-sum formula: two distinct values,
    n_each items at each, n_hi of the "hi" value's items are fails; verified numerically too --
    exact_split(1000, 699) gives auc() == 0.699 and exact_split(1000, 700) gives 0.7, not
    0.6999999999999998 or the like), so a boundary test lands on an exact float, never a rounding
    artefact of hi_share * n_each."""
    n_lo = n_each - n_hi
    fails = (scored_tracks(first_id, start, n_hi, system, hi, False, n_days=10)
            + scored_tracks(first_id + n_hi, start, n_lo, system, lo, False, n_days=10))
    passes = (scored_tracks(first_id + 2 * n_each, start, n_hi, system, lo, True, n_days=10)
             + scored_tracks(first_id + 2 * n_each + n_hi, start, n_lo, system, hi, True, n_days=10))
    return fails + passes


def only(findings):
    assert len(findings) == 1, [f.title for f in findings]
    return findings[0]


# ---- auc(): hand-made lists ----

def test_auc_perfect_separation_is_one():
    assert loss_origin.auc([3.0, 4.0, 5.0], [0.0, 1.0, 2.0]) == pytest.approx(1.0)


def test_auc_identical_distributions_is_half():
    assert loss_origin.auc([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.5)


def test_auc_reversed_is_zero():
    assert loss_origin.auc([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]) == pytest.approx(0.0)


def test_auc_ties_get_half_credit():
    # pairwise by hand: (2>1)=1, (2==2)=0.5, (3>1)=1, (3>2)=1 -> 3.5 / 4 = 0.875
    assert loss_origin.auc([2.0, 3.0], [1.0, 2.0]) == pytest.approx(0.875)


def test_auc_needs_both_outcomes():
    assert loss_origin.auc([], [1.0]) is None
    assert loss_origin.auc([1.0], []) is None
    assert loss_origin.auc([], []) is None


# ---- analyze(): facts hold every COMPARABLE measurement (population clears MIN_TRACKS and
# MIN_PER_OUTCOME); a finding only when the AUC also clears STRONG_AUC. Mirrors machine_compare
# (review fix, 2026-09-24): a floor below which a rate cannot be trusted excludes a laser from
# facts too, never only from findings -- nothing is guessed into either.

def test_a_strong_laser_is_one_finding():
    tracks = mixed(0, START, 500, "B", hi=3.0, lo=1.0, hi_share=0.9)
    facts, findings = loss_origin.analyze("M", tracks, label)
    f = only(findings)
    assert f.analyzer == "loss_origin" and f.lever == "deposition"
    assert f.category == "Where the loss is made"
    assert f.expected_gain_points is None
    assert f.title == "Laser 1 (LTS): incoming linearity predicts the laser verdict (AUC 0.90)"
    assert f.systems == ("B",)
    assert facts["Laser 1 (LTS)"]["auc_error"] == pytest.approx(0.90)
    assert facts["Laser 1 (LTS)"]["n"] == 1000 and facts["Laser 1 (LTS)"]["fails"] == 500


def test_facts_shape_is_exactly_documented():
    tracks = (scored_tracks(0, START, 200, "B", 2.0, False)
             + scored_tracks(1000, START, 200, "B", 2.0, True))
    facts, _ = loss_origin.analyze("M", tracks, label)
    assert set(facts["Laser 1 (LTS)"]) == {"n", "fails", "auc_error", "auc_resistance"}


def test_a_weak_laser_is_facts_only_no_finding():
    # Same distribution both sides -> AUC ~0.5: a real, adequately-populated fact, never a
    # finding (STRONG_AUC) -- e.g. 8232-1 and 8340-1 in the real data.
    tracks = (scored_tracks(0, START, 500, "B", 2.0, False)
             + scored_tracks(1000, START, 500, "B", 2.0, True))
    facts, findings = loss_origin.analyze("M", tracks, label)
    assert findings == []
    assert facts["Laser 1 (LTS)"]["auc_error"] == pytest.approx(0.5)
    assert facts["Laser 1 (LTS)"]["n"] == 1000


def test_under_the_outcome_floor_is_a_fact_never_a_finding():
    # Perfect separation (AUC 1.0, well over STRONG_AUC) and the total clears MIN_TRACKS, but
    # only 40 of the outcome MIN_PER_OUTCOME names -- e.g. 8202-1 in the real data (AUC 0.71,
    # 27 failures). Ruling 2, reversing the earlier (machine_compare-modelled) reading of this:
    # a fact ALWAYS once a laser has graded tracks here -- the outcome floor gates only whether
    # it is ALSO a finding, never whether the measurement is visible on the tab at all.
    tracks = (scored_tracks(0, START, 40, "B", 3.0, False)
             + scored_tracks(1000, START, 300, "B", 1.0, True))
    facts, findings = loss_origin.analyze("M", tracks, label)
    assert findings == []
    f = facts["Laser 1 (LTS)"]
    assert f["n"] == 340 and f["fails"] == 40
    assert f["auc_error"] == pytest.approx(1.0)           # a real, visible AUC -- just too thin to call
    assert f["auc_resistance"] is not None                # computed, not guessed away either


def test_under_the_track_floor_is_a_fact_never_a_finding():
    # 60 + 60 = 120 tracks: both outcomes individually clear the 50-per-outcome floor, but the
    # total is under MIN_TRACKS (300). Same ruling as the outcome floor above: still a fact,
    # never a finding -- the two floors are independent, but neither excludes a laser from facts.
    tracks = (scored_tracks(0, START, 60, "B", 3.0, False)
             + scored_tracks(1000, START, 60, "B", 1.0, True))
    facts, findings = loss_origin.analyze("M", tracks, label)
    assert findings == []
    f = facts["Laser 1 (LTS)"]
    assert f["n"] == 120 and f["fails"] == 60
    assert f["auc_error"] == pytest.approx(1.0)


# ---- boundary tests (review, 2026-09-24): each threshold isolated from the other two ----------

def test_track_count_299_is_no_finding_300_is_one_all_else_equal():
    # min(fails, passes) stays a safe 149/150 throughout -- only the TOTAL crosses MIN_TRACKS.
    below = (scored_tracks(0, START, 150, "B", 3.0, False)
            + scored_tracks(1000, START, 149, "B", 1.0, True))                  # 299
    facts, findings = loss_origin.analyze("M", below, label)
    assert findings == []
    assert facts["Laser 1 (LTS)"]["n"] == 299 and facts["Laser 1 (LTS)"]["auc_error"] == pytest.approx(1.0)

    at = (scored_tracks(0, START, 150, "B", 3.0, False)
         + scored_tracks(1000, START, 150, "B", 1.0, True))                     # 300
    f = only(loss_origin.analyze("M", at, label)[1])
    assert f.n_units == 300


def test_per_outcome_49_is_no_finding_50_is_one_all_else_equal():
    # The passing side stays comfortably over both floors throughout -- only the FAIL side's
    # count crosses MIN_PER_OUTCOME.
    passes = scored_tracks(2000, START, 260, "B", 1.0, True)
    below = scored_tracks(0, START, 49, "B", 3.0, False) + passes                # 49 + 260 = 309
    facts, findings = loss_origin.analyze("M", below, label)
    assert findings == []
    assert facts["Laser 1 (LTS)"]["fails"] == 49

    at = scored_tracks(0, START, 50, "B", 3.0, False) + passes                   # 50 + 260 = 310
    f = only(loss_origin.analyze("M", at, label)[1])
    assert f.n_units == 310


def test_auc_699_is_no_finding_700_is_one_all_else_equal():
    # n_each=1000 clears MIN_TRACKS (2000 total) and MIN_PER_OUTCOME (1000 each) by a wide
    # margin on both sides -- only the AUC itself crosses STRONG_AUC (0.70).
    below = exact_split(0, START, 1000, "B", 699, hi=3.0, lo=1.0)
    facts, findings = loss_origin.analyze("M", below, label)
    assert findings == []
    assert facts["Laser 1 (LTS)"]["auc_error"] == pytest.approx(0.699)

    at = exact_split(0, START, 1000, "B", 700, hi=3.0, lo=1.0)
    f = only(loss_origin.analyze("M", at, label)[1])
    assert f.strength_value == pytest.approx(0.7)


def test_lookback_is_measured_from_the_models_latest_across_lasers_not_each_lasers_own():
    # The docstring is explicit: the window is LOOKBACK_DAYS back from the MODEL's own latest
    # file_date across every laser (the engine's annual_volume convention), computed once before
    # the per-laser split -- never each laser's own latest date.
    anchor = START + timedelta(days=1000)
    # A single track on a DIFFERENT laser sets the model-wide latest date; it is ungraded, so it
    # never itself appears in facts or findings -- its only job is to anchor "latest" away from
    # laser B, so a bug that measured each laser's own latest instead would make BOTH batches
    # below read as "day zero" and both would wrongly count.
    anchor_track = replace(make_track(9000, date=anchor, system="A"), linearity_pass=None)

    boundary = anchor - timedelta(days=365)             # exactly LOOKBACK_DAYS back -- included (>=)
    just_outside = boundary - timedelta(days=1)          # one day further back -- excluded

    inside = (scored_tracks(0, boundary, 150, "B", 3.0, False, n_days=1)
             + scored_tracks(1000, boundary, 150, "B", 1.0, True, n_days=1))
    outside = (scored_tracks(2000, just_outside, 150, "B", 3.0, False, n_days=1)
              + scored_tracks(3000, just_outside, 150, "B", 1.0, True, n_days=1))

    facts, findings = loss_origin.analyze("M", [anchor_track] + inside + outside, label)
    # Only the 300 INSIDE tracks count -- if the 300 OUTSIDE ones were wrongly included too,
    # n would read 600.
    assert facts["Laser 1 (LTS)"]["n"] == 300
    f = only(findings)
    assert f.n_units == 300


def test_tracks_without_an_untrimmed_sweep_are_skipped_never_scored_zero():
    strong = mixed(0, START, 500, "B", hi=3.0, lo=1.0, hi_share=0.9)
    # 500 more FAILs with NO untrimmed sweep at all. If they were ever scored 0 (the lowest
    # possible score) they would only strengthen the separation; if merely counted without being
    # scored they would move n and fails without moving the auc_error they are claimed to
    # support. Neither must happen -- they must vanish from this analyzer entirely.
    blank = [replace(t, untrimmed_errors=None)
            for t in scored_tracks(2000, START, 500, "B", 9.0, False)]
    facts, findings = loss_origin.analyze("M", strong + blank, label)
    f = only(findings)
    assert facts["Laser 1 (LTS)"]["n"] == 1000            # not 1500
    assert facts["Laser 1 (LTS)"]["fails"] == 500          # not 1000
    assert facts["Laser 1 (LTS)"]["auc_error"] == pytest.approx(0.90)
    assert f.n_units == 1000


def test_an_all_none_untrimmed_tuple_is_also_skipped_not_zero():
    strong = mixed(0, START, 500, "B", hi=3.0, lo=1.0, hi_share=0.9)
    blank = [replace(t, untrimmed_errors=(None, None, None))
            for t in scored_tracks(2000, START, 200, "B", 9.0, False)]
    facts, findings = loss_origin.analyze("M", strong + blank, label)
    f = only(findings)
    assert facts["Laser 1 (LTS)"]["n"] == 1000
    assert f.n_units == 1000


def test_ungraded_tracks_are_never_counted():
    strong = mixed(0, START, 500, "B", hi=3.0, lo=1.0, hi_share=0.9)
    # 400 more tracks with no stored verdict -- if ever counted as a fail or a pass they would
    # move n, fails and the auc away from the values asserted below.
    ungraded = [replace(t, linearity_pass=None)
               for t in scored_tracks(2000, START, 400, "B", 9.0, False)]
    facts, findings = loss_origin.analyze("M", strong + ungraded, label)
    f = only(findings)
    assert facts["Laser 1 (LTS)"]["n"] == 1000
    assert f.n_units == 1000


def test_auc_resistance_skips_none_and_1e9_plus_junk():
    # Every FAIL's resistance is either None or the work database's 1e9+ junk marker; the error
    # score is a constant 2.0 x band on both sides (AUC 0.5, not the point of this test), just
    # large enough a population to clear both floors so the facts entry exists. If either junk
    # form were wrongly treated as usable, auc_resistance would come back a number (5e9 sits far
    # above the passes' valid resistance, so a bug that let it through would score a perfect
    # 1.0, not None) -- it must come back None: no fail has a USABLE resistance at all.
    fails_junk = [replace(t, untrimmed_resistance=5e9)
                 for t in scored_tracks(0, START, 80, "B", 2.0, False)]
    fails_none = [replace(t, untrimmed_resistance=None)
                 for t in scored_tracks(100, START, 80, "B", 2.0, False)]
    passes = scored_tracks(1000, START, 160, "B", 2.0, True)     # r_in defaults to 4500.0, valid
    facts, _ = loss_origin.analyze("M", fails_junk + fails_none + passes, label)
    assert facts["Laser 1 (LTS)"]["n"] == 320                     # the error side is unaffected
    assert facts["Laser 1 (LTS)"]["auc_resistance"] is None


def test_two_lasers_only_the_qualifying_one_finds():
    strong = mixed(0, START, 500, "B", hi=3.0, lo=1.0, hi_share=0.9)
    weak = (scored_tracks(2000, START, 500, "A", 2.0, False)
           + scored_tracks(3000, START, 500, "A", 2.0, True))
    facts, findings = loss_origin.analyze("M", strong + weak, label)
    f = only(findings)
    assert f.systems == ("B",)
    assert set(facts) == {"Laser 1 (LTS)", "Laser 2 (DLTS)"}
    assert facts["Laser 2 (DLTS)"]["auc_error"] == pytest.approx(0.5)


def test_no_tracks_is_no_facts_no_findings():
    assert loss_origin.analyze("M", [], label) == ({}, [])


def test_finding_never_claims_a_gain():
    tracks = mixed(0, START, 500, "B", hi=3.0, lo=1.0, hi_share=0.9)
    _, findings = loss_origin.analyze("M", tracks, label)
    f = only(findings)
    assert f.expected_gain_points is None and f.gain_definition == ""
