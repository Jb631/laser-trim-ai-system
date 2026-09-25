"""Rework load: laser-FAIL -> final-test-PASS unit-days, confirmed as hand trim, never overkill.

Every silence test here was made to FAIL first by relaxing the control it names -- a green test
that cannot go red proves nothing. The unit-day disposition itself (per track the day's last
attempt; every track must pass) is pinned against `DatabaseManager.get_model_trim_ft_agreement`
in test_retrim_disposition.py and by scripts/app_qa_sweep.py's independent re-derivation. These
tests pin what rework_load adds on top of it (review of 85222c4, fix brief 2026-09-25):

* THE METRIC -- per linked (unit, track) pair, the largest |corrected error| over the travel
  BOTH stations grade (trim rows with limits, inside the final test's graded window, on one
  position axis), each station corrected with its OWN offset -- never the stored whole-sweep
  maxima, which sit where one station does not grade at all;
* THE GRAIN -- unit-days, not final-test records, and each final-test record on ITS OWN track;
* THE COMPARISON -- rework against the TOP THIRD of pass/pass units by laser error (like with
  like), by a one-sided rank test (fix round 2, 2026-09-25: Mann-Whitney U, normal approximation
  with tie correction) at CONFIRM_P, with the effect no smaller than MAX_EFFECT_RATIO, and
  MIN_UNIT_DAYS and MIN_CONTROL at their exact boundaries -- each boundary red on round 1's fixed
  0.8 ratio cut wherever the two rules disagree;
* the facts/finding split (the test is a fact always; the floors gate only the verdict), and that
  a crash is never swallowed here.

Every fixture is a centred spike: a sweep reading 0.01 V everywhere but one point, so each
station's worst graded error is exactly the number the test names. Example data is invented.
"""
from datetime import timedelta

import pytest

from laser_trim_analyzer.database.manager import compute_unit_id
from laser_trim_analyzer.database.models import (
    AnalysisResult, FinalTestResult, FinalTestTrack, StatusType, SystemType, TrackResult)
from laser_trim_analyzer.findings.analyzers import rework_load
from laser_trim_analyzer.findings import presentation as P
from findings_helpers import START, days, label, make_track

MODEL = "M"
POS = [float(p) for p in range(-10, 11)]            # 21 positions, -10 .. 10, index 10 = centre
BAND = 0.5
UP, LO = [BAND] * len(POS), [-BAND] * len(POS)


@pytest.fixture()
def db(tmp_path):
    """A real, empty DatabaseManager -- both globals injected per global-constraints.md, even
    though rework_load is always called with `db` passed explicitly and never reaches
    get_database() itself; the rule is unconditional for anything that builds one."""
    import laser_trim_analyzer.database as _d
    import laser_trim_analyzer.database.manager as _m
    d = _m.DatabaseManager(tmp_path / "rework.db")
    _m._db_manager = d
    _d._db_manager = d
    return d


def spike(worst, at=10, n=len(POS), base=0.01):
    """A sweep reading `base` everywhere but `worst` at index `at`."""
    return [worst if i == at else base for i in range(n)]


def trim_file(s, shop, when, track_id, passed, errors, *, model=MODEL, system=SystemType.B,
              status=None, positions=POS, upper=UP, lower=LO, offset=0.0, slope=0.0, theory=None,
              unit_id=True):
    """One trim FILE carrying one track (the per-track TA/TB layout; a single-track model is one
    such file a day). `status` overrides the verdict's own status -- ERROR means the analyser
    could not read the file, so the track carries no verdict at all. Returns the analysis id."""
    if status is None:
        status = StatusType.PASS if passed else StatusType.FAIL
    a = AnalysisResult(model=model, serial=str(shop), system=system, file_date=when,
                       filename=f"{model}_{shop}_{track_id.replace(' ', '')}_{when:%Y%m%d_%H%M}.xls",
                       overall_status=status,
                       unit_id=compute_unit_id(model, str(shop), when) if unit_id else None)
    s.add(a)
    s.flush()
    s.add(TrackResult(analysis_id=a.id, track_id=track_id, status=status,
                      linearity_pass=None if status == StatusType.ERROR else passed,
                      position_data=list(positions), error_data=list(errors),
                      upper_limits=list(upper), lower_limits=list(lower),
                      optimal_offset=offset, optimal_slope=slope, theory_data=theory))
    return a.id


def final_test(s, serial, when, linked_id, errors, *, model=MODEL, positions=POS, upper=UP,
               lower=LO, window=(0, len(POS) - 1), offset=0.0, slope=0.0, theory=None,
               passed=True, confidence=1.0):
    """One final-test record (one 'default' track, as every real FT file here carries), linked
    straight to `linked_id` -- the matcher links the day's LAST trim file, so callers pass that."""
    f = FinalTestResult(model=model, serial=str(serial),
                        filename=f"{model}-sn{serial}_{when:%Y%m%d_%H%M%S}.xls", file_date=when,
                        test_date=when, overall_status=StatusType.PASS if passed else StatusType.FAIL,
                        linearity_pass=passed, linked_trim_id=linked_id, match_confidence=confidence)
    s.add(f)
    s.flush()
    s.add(FinalTestTrack(final_test_id=f.id, track_id="default",
                         status=StatusType.PASS if passed else StatusType.FAIL,
                         linearity_pass=passed, position_data=list(positions),
                         error_data=list(errors), upper_limits=list(upper), lower_limits=list(lower),
                         optimal_offset=offset, optimal_slope=slope, theory_data=theory,
                         graded_start=None if window is None else window[0],
                         graded_end=None if window is None else window[1]))
    return f.id


def seed(db, n_rework, n_control, *, rework=(0.30, 0.10), control=(0.30, 0.30), ft_records=1,
         first_shop=1, model=MODEL, system=SystemType.B, ft_positions=POS, at=10):
    """`n_rework` single-track unit-days that failed at the laser and passed final test, then
    `n_control` that passed both -- one unit-day a day. Each pair is (laser worst, final-test
    worst), a spike at index `at` on both stations; `ft_records` final tests per unit-day, all on
    the unit's one track. Returns the next free shop number."""
    shop = first_shop
    with db.session() as s:
        for k in range(n_rework + n_control):
            passed = k >= n_rework
            laser, ft = control if passed else rework
            day = START + timedelta(days=k)
            aid = trim_file(s, shop, day.replace(hour=9), "Track A", passed, spike(laser, at),
                            model=model, system=system)
            for j in range(ft_records):
                final_test(s, shop, day + timedelta(days=2, minutes=5 * j), aid, spike(ft, at),
                           model=model, positions=ft_positions)
            shop += 1
    return shop


def seed_ratios(db, rework, top, *, low=60, first_shop=1):
    """Unit-days whose final-test/laser ratio is EXACTLY what the test names: one reworked
    unit-day per value in `rework` and one pass/pass unit-day per value in `top`, all at laser
    0.30 V -- plus `low` pass/pass unit-days the laser left nearly perfect (0.05 V, ratio 1.0),
    which sort BELOW them by laser error. With the defaults (30 + 60) the top third of the
    pass/pass units is exactly `top`. Returns the next free shop number."""
    rows = ([(False, 0.30, r) for r in rework] + [(True, 0.30, c) for c in top]
            + [(True, 0.05, 1.0)] * low)
    shop = first_shop
    with db.session() as s:
        for k, (passed, laser, ratio) in enumerate(rows):
            day = START + timedelta(days=k)
            aid = trim_file(s, shop, day.replace(hour=9), "Track A", passed, spike(laser))
            final_test(s, shop, day + timedelta(days=2), aid, spike(laser * ratio))
            shop += 1
    return shop


def tracks_for(n, model=MODEL, system="B"):
    """The window population: only file_date matters to rework_load (it sets `latest` and so the
    cutoff) -- independent of the DB rows seeded for the comparison itself, exactly as
    station_setup treats its two samples."""
    return [make_track(k, date=d, system=system) for k, d in enumerate(days(START, n))]


def run(db, n=5):
    return rework_load.analyze(MODEL, db, tracks_for(n), label)


def only(findings):
    assert len(findings) == 1, [f.title for f in findings]
    return findings[0]


class _Boom:
    def session(self, *a, **k):
        raise AssertionError("must not query the database with nothing to report on")

    def get_model_trim_ft_agreement(self, *a, **k):
        raise AssertionError("must not query the database with nothing to report on")


# ---- a confirmed signature is one finding; the readout is unit-days ----------------------------

COMPARISON_IN_WORDS = ("their error fell more between the stations than it did for the untouched "
                       "units that started nearest them")


def test_a_confirmed_signature_is_one_finding_with_the_unit_day_readout(db):
    seed(db, 40, 90)             # rework: laser 0.30 -> final test 0.10 (1/3); control 0.30 -> 0.30
    facts, findings = run(db)
    f = only(findings)
    assert f.analyzer == "rework_load" and f.category == "Rework load"
    assert f.lever == "laser_settings"
    assert f.expected_gain_points is None and f.gain_definition == ""
    assert f.n_units == 40
    assert f.title == "Laser 1 (LTS): 40 units a year fail here and pass final test after rework"
    assert "hand trim" in f.summary and "33%" in f.summary and "100%" in f.summary
    assert COMPARISON_IN_WORDS in f.summary
    assert f.systems == ("B",)
    assert facts["rework_unit_days"] == 40 and facts["linked"] == 130
    assert facts["rework_ratio_n"] == 40 and facts["control_n"] == 90
    assert facts["control_top_third_n"] == 30                 # 90 - 2*90//3
    assert facts["skipped_pairs"] == 0 and facts["junk_readings"] == 0
    assert facts["confirmed"] is True
    assert facts["median_ratio_rework"] == pytest.approx(1 / 3, abs=1e-3)
    assert facts["median_ratio_control_top_third"] == pytest.approx(1.0, abs=1e-3)
    assert facts["effect_ratio"] == pytest.approx(1 / 3, abs=1e-3)
    assert facts["mann_whitney_u"] == 0.0       # every reworked unit below every untouched one
    assert facts["p_value"] < 1e-6
    assert facts["reduction"] == rework_load.REDUCTION


def test_evidence_carries_the_test_both_medians_both_sizes_and_the_comparison_made(db):
    seed(db, 40, 90)
    f = only(run(db)[1])
    assert set(f.evidence) == {"facts", "comparison"}
    ev = f.evidence["facts"]
    assert set(ev) == {"rework_unit_days", "rework_ratio_n", "control_n", "control_top_third_n",
                       "median_ratio_rework", "median_ratio_control_top_third", "effect_ratio",
                       "mann_whitney_u", "p_value", "skipped_pairs", "reduction"}
    assert (ev["rework_unit_days"], ev["rework_ratio_n"], ev["control_n"],
            ev["control_top_third_n"], ev["skipped_pairs"]) == (40, 40, 90, 30, 0)
    assert ev["effect_ratio"] == pytest.approx(1 / 3, abs=1e-3) and ev["p_value"] < 1e-6
    assert ev["reduction"] == rework_load.REDUCTION
    said = f.evidence["comparison"]
    assert "rank test" in said and "top third" in said and "0.01" in said and "0.9" in said


# ---- the thresholds are the controller's rulings (fix round 2, 2026-09-25) ---------------------
# Pinned here once. The boundary tests below never read these constants: each goes red on its
# own BEHAVIOUR when a threshold moves (every one was mutation-checked that way), so a green run
# proves the boundary is where the ruling put it, not merely that a number was typed.

def test_the_thresholds_are_the_ruled_values():
    assert rework_load.CONFIRM_P == 0.01 and rework_load.MAX_EFFECT_RATIO == 0.9
    assert rework_load.MIN_UNIT_DAYS == 30 and rework_load.MIN_CONTROL == 20
    assert rework_load.JUNK_VOLTS == 1.0
    assert rework_load.LOOKBACK_DAYS == 365 and rework_load.MIN_CONFIDENCE == 0.70
    assert not hasattr(rework_load, "CONFIRM_RATIO")          # round 1's fixed ratio cut is gone


# ---- the rank test itself: findings/stats.py, U built from the AUC helper -----------------------
# Reference values: scipy.stats.mannwhitneyu(x, y, alternative="less", method="asymptotic",
# use_continuity=False) -- the ruled normal approximation with tie correction, and no continuity
# correction. The app never imports scipy for this: stats.py stays dependency-free.

def test_u_is_the_auc_times_both_sizes_and_p_is_the_normal_approximation():
    from laser_trim_analyzer.findings.stats import auc, mann_whitney_lower
    x, y = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    u, p = mann_whitney_lower(x, y)
    assert u == auc(x, y) * 3 * 3 == 0.0
    assert p == pytest.approx(0.024767306717813357, rel=1e-9)
    x, y = [0.5, 0.7, 0.9, 1.1, 1.3], [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    u, p = mann_whitney_lower(x, y)
    assert u == pytest.approx(auc(x, y) * 5 * 6) and u == pytest.approx(6.0)
    assert p == pytest.approx(0.05017412323114538, rel=1e-9)


def test_the_rank_test_corrects_for_ties():
    from laser_trim_analyzer.findings.stats import mann_whitney_lower
    u, p = mann_whitney_lower([1.0, 1.0, 2.0, 3.0], [2.0, 3.0, 3.0, 4.0, 5.0])
    assert u == pytest.approx(2.5)                             # a tie counts half
    assert p == pytest.approx(0.029725546655776806, rel=1e-9)  # without the correction: 0.0331


def test_the_rank_test_is_one_sided_lower():
    from laser_trim_analyzer.findings.stats import mann_whitney_lower
    x, y = [0.5, 0.7, 0.9, 1.1, 1.3], [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    (_, p_lower), (_, p_swapped) = mann_whitney_lower(x, y), mann_whitney_lower(y, x)
    assert p_lower < 0.5 < p_swapped and p_lower + p_swapped == pytest.approx(1.0)


def test_no_rank_test_without_both_groups_or_without_any_variation():
    from laser_trim_analyzer.findings.stats import mann_whitney_lower
    assert mann_whitney_lower([], [1.0]) is None and mann_whitney_lower([1.0], []) is None
    assert mann_whitney_lower([2.0, 2.0], [2.0, 2.0, 2.0]) == (3.0, None)   # all tied: no p


# ---- compare like with like: the TOP THIRD of control, never the whole of it --------------------

def test_the_comparison_is_against_the_top_third_of_control_not_all_of_it(db):
    # Final test has an error floor, so FT/laser runs HIGH on units the laser left nearly perfect:
    # 60 pass/pass units at laser 0.05 read 2.0 and pull the whole control's median to 2.0. The
    # reworked units (laser 0.30) read 0.95 -- far below 2.0, which a comparison with the whole
    # control would confirm. Against the 30 untouched units that started nearest them (laser
    # 0.30, ratio 1.0) the shift is significant but far too small (0.95 > 0.9): not confirmed.
    nxt = seed(db, 40, 0, rework=(0.30, 0.285))
    nxt = seed(db, 0, 60, control=(0.05, 0.10), first_shop=nxt)
    seed(db, 0, 30, control=(0.30, 0.30), first_shop=nxt)
    facts, findings = run(db)
    assert findings == []
    assert facts["confirmed"] is False
    assert facts["control_n"] == 90 and facts["control_top_third_n"] == 30
    assert facts["median_ratio_control_top_third"] == pytest.approx(1.0, abs=1e-3)
    assert facts["effect_ratio"] == pytest.approx(0.95, abs=1e-3)
    assert facts["p_value"] < 1e-6


# ---- CONFIRM_P at its boundary: p just under confirms, just over does not -----------------------
# 30 reworked unit-days against 30 in the top third, every ratio distinct: 20 far below all of the
# top third, 9 above all of it, and one that beats exactly 22 of them (U = 9 x 30 + 22 = 292,
# p = 0.00975) or 23 (U = 293, p = 0.01014). The effect is 0.51 either way -- round 1's fixed 0.8
# ratio cut confirms BOTH, so "just over" is where the two rules part.

TOP_30 = [1.0 + 0.001 * j for j in range(30)]


def _straddle(beats):
    return ([0.5 + 0.001 * i for i in range(20)] + [1.0305 + 0.001 * i for i in range(9)]
            + [1.0 + 0.001 * beats - 0.0005])


def test_p_just_under_the_confirm_p_is_confirmed(db):
    seed_ratios(db, _straddle(22), TOP_30)
    facts, findings = run(db)
    assert only(findings).n_units == 30
    assert facts["mann_whitney_u"] == 292.0
    assert facts["p_value"] == pytest.approx(0.009747, abs=5e-6)
    assert facts["effect_ratio"] == pytest.approx(0.507, abs=1e-3)


def test_p_just_over_the_confirm_p_is_not(db):
    seed_ratios(db, _straddle(23), TOP_30)
    facts, findings = run(db)
    assert findings == []                    # a fixed 0.8 ratio cut WOULD confirm this (0.51)
    assert facts["mann_whitney_u"] == 293.0
    assert facts["p_value"] == pytest.approx(0.010139, abs=5e-6)
    assert facts["confirmed"] is False and "p =" in facts["note"]


# ---- MAX_EFFECT_RATIO at its boundary: 0.89 confirms, 0.91 does not -----------------------------

def test_an_effect_of_0_89_is_confirmed(db):
    seed_ratios(db, [0.89] * 30, [1.0] * 30)     # round 1's 0.8 cut would NOT confirm this
    facts, findings = run(db)
    assert only(findings).n_units == 30
    assert facts["effect_ratio"] == pytest.approx(0.89, abs=1e-3) and facts["p_value"] < 1e-6


def test_an_effect_of_0_91_is_not(db):
    seed_ratios(db, [0.91] * 30, [1.0] * 30)
    facts, findings = run(db)
    assert findings == []
    assert facts["effect_ratio"] == pytest.approx(0.91, abs=1e-3)
    assert facts["p_value"] < 1e-6          # significant -- but too small a shift to call hand trim
    assert facts["confirmed"] is False and facts["note"]


# ---- MIN_CONTROL at its boundary: 19 vs 20 control units in the TOP THIRD -----------------------

def test_19_control_units_in_the_top_third_is_nothing(db):
    seed(db, 40, 57)                                        # top third = 57 - 38 = 19
    facts, findings = run(db)
    assert findings == []
    assert facts["control_n"] == 57 and facts["control_top_third_n"] == 19
    assert facts["confirmed"] is False
    assert facts["p_value"] < 1e-6           # the test WOULD confirm: still a fact, never a verdict


def test_20_control_units_in_the_top_third_is_a_finding(db):
    seed(db, 40, 60)                         # top third = 60 - 40 = 20 (round 1 needed 30)
    facts, findings = run(db)
    assert only(findings).n_units == 40
    assert facts["control_top_third_n"] == 20


# ---- MIN_UNIT_DAYS at its boundary -- counted in unit-days, never final-test records -------------

def test_29_rework_unit_days_is_nothing_even_with_58_final_test_records(db):
    seed(db, 29, 60, ft_records=2)          # each unit final-tested twice: 58 records, 29 units
    facts, findings = run(db)
    assert findings == []
    assert facts["rework_unit_days"] == 29 and facts["rework_ratio_n"] == 29
    assert facts["confirmed"] is False
    # The floors gate only the verdict: the test is still run, and shown.
    assert facts["mann_whitney_u"] == 0.0 and facts["p_value"] < 1e-6


def test_30_rework_unit_days_is_a_finding_and_one_ratio_per_unit_day(db):
    seed(db, 30, 60, ft_records=2)          # 60 records, 30 unit-days; top third 20
    facts, findings = run(db)
    assert only(findings).n_units == 30
    assert facts["rework_unit_days"] == 30 and facts["rework_ratio_n"] == 30


def test_30_rework_unit_days_with_only_29_scorable_is_nothing(db):
    # The floor is on the population the test is run over: one unit-day's final test grades a
    # stretch of travel the laser never grades, so only 29 ratios exist.
    nxt = seed(db, 29, 90)
    with db.session() as s:
        aid = trim_file(s, nxt, START.replace(hour=9), "Track A", False,
                        spike(0.30, at=15), upper=[None] * 12 + [BAND] * 9,
                        lower=[None] * 12 + [-BAND] * 9)
        final_test(s, nxt, START + timedelta(days=2), aid, spike(0.10, at=2), window=(0, 4))
    facts, findings = run(db)
    assert findings == []
    assert facts["rework_unit_days"] == 30 and facts["rework_ratio_n"] == 29
    assert facts["skipped_pairs"] == 1
    assert facts["confirmed"] is False
    assert facts["median_ratio_rework"] == pytest.approx(1 / 3, abs=1e-3)
    assert facts["median_ratio_control_top_third"] == pytest.approx(1.0, abs=1e-3)


# ---- the test is a fact always; the floors gate only the verdict --------------------------------

def test_the_test_is_a_fact_even_far_below_the_floors(db):
    seed(db, 5, 9)                          # 5 reworked, 9 pass/pass: a top third of 3
    facts, findings = run(db)
    assert findings == []
    assert facts["rework_ratio_n"] == 5 and facts["control_top_third_n"] == 3
    assert facts["mann_whitney_u"] == 0.0 and facts["p_value"] is not None
    assert facts["effect_ratio"] == pytest.approx(1 / 3, abs=1e-3)
    assert facts["reduction"] == rework_load.REDUCTION and facts["confirmed"] is False


def test_with_no_reworked_unit_the_test_is_none_and_the_sizes_are_still_said(db):
    seed(db, 0, 30)
    facts, findings = run(db)
    assert findings == [] and facts["rework_unit_days"] == 0
    assert facts["rework_ratio_n"] == 0 and facts["control_n"] == 30
    assert facts["control_top_third_n"] == 10
    assert facts["mann_whitney_u"] is None and facts["p_value"] is None
    assert facts["effect_ratio"] is None and facts["median_ratio_rework"] is None
    assert facts["median_ratio_control_top_third"] == pytest.approx(1.0, abs=1e-3)
    assert facts["confirmed"] is False and facts["note"]


# ---- two-track units: each final test on ITS track, one ratio per unit-day ----------------------

def _two_track_unit(s, shop, day, a, b, fts):
    """One two-track unit-day, one trim file per track ten minutes apart (Track A first), and its
    final tests `fts` = [(serial, final-test worst)] linked, as the matcher links, to the day's
    LAST file -- Track B's. `a`/`b` = (passed, laser worst)."""
    trim_file(s, shop, day.replace(hour=9), "Track A", a[0], spike(a[1]))
    last = trim_file(s, shop, day.replace(hour=9, minute=10), "Track B", b[0], spike(b[1]))
    for j, (serial, ft) in enumerate(fts):
        final_test(s, serial, day + timedelta(days=2, minutes=5 * j), last, spike(ft))


def test_a_two_track_unit_day_pairs_each_final_test_with_its_own_track(db):
    # 6607's layout. Track A failed at the laser (0.30), Track B passed (0.05). Final test ran on
    # Track A twice -- a digits-only serial, then an 'A' suffix -- and both link to Track B's file.
    # Right: Track A's last attempt, its latest final test, 0.12 / 0.30 = 0.4, ONE ratio for the
    # unit-day. Wrong pairing (Track B's file) reads 0.12 / 0.05 = 2.4; counting records reads 60.
    with db.session() as s:
        for k in range(30):
            _two_track_unit(s, 100 + k, START + timedelta(days=k), (False, 0.30), (True, 0.05),
                            [(f"{100 + k}", 0.10), (f"{100 + k}A", 0.12)])
        for k in range(90):                     # control: both tracks passed, final test on each
            shop = 300 + k
            _two_track_unit(s, shop, START + timedelta(days=k), (True, 0.30), (True, 0.05),
                            [(f"{shop}", 0.30), (f"{shop}b", 0.05)])
    agreement = db.get_model_trim_ft_agreement(MODEL)
    assert agreement["overkills"] == 60 and agreement["overkill_unit_days"] == 30
    facts, findings = run(db)
    f = only(findings)
    assert f.n_units == 30
    assert facts["rework_unit_days"] == 30 and facts["rework_ratio_n"] == 30
    assert facts["median_ratio_rework"] == pytest.approx(0.4, abs=1e-3)
    assert facts["control_n"] == 90                         # one per unit-day, not one per track
    assert facts["median_ratio_control_top_third"] == pytest.approx(1.0, abs=1e-3)


def test_a_final_test_on_the_passing_track_of_a_failing_unit_is_judged_by_that_track(db):
    # Ten two-track units whose Track B failed at the laser but whose only final test is on Track A,
    # which passed there. The unit-day is a laser FAIL (every track must pass), so the readout
    # counts it -- but the final test is judged by ITS track's verdict: an untouched Track A, i.e.
    # control, never evidence of hand trim.
    nxt = seed(db, 40, 90)
    with db.session() as s:
        for k in range(10):
            _two_track_unit(s, nxt + k, START + timedelta(days=k), (True, 0.30), (False, 0.30),
                            [(f"{nxt + k}", 0.30)])
    facts, findings = run(db)
    assert only(findings).n_units == 50                     # the shared unit-day readout
    assert facts["rework_unit_days"] == 50
    assert facts["rework_ratio_n"] == 40                    # ...only 40 are rework by their own track
    assert facts["control_n"] == 100
    assert facts["median_ratio_rework"] == pytest.approx(1 / 3, abs=1e-3)


def test_a_lettered_final_test_of_a_track_not_trimmed_that_day_is_left_unpaired(db):
    # Only Track B was trimmed that day, but the final test is Track A's (digits-only serial):
    # comparing them is exactly the Track-A-FT-vs-Track-B-trim mistake -- unpaired, and counted.
    nxt = seed(db, 40, 90)
    with db.session() as s:
        for k in range(3):
            aid = trim_file(s, nxt + k, START.replace(hour=9), "Track B", False, spike(0.30))
            final_test(s, nxt + k, START + timedelta(days=2), aid, spike(0.01))
    facts, findings = run(db)
    assert facts["unpaired_final_tests"] == 3
    assert facts["rework_ratio_n"] == 40
    assert facts["median_ratio_rework"] == pytest.approx(1 / 3, abs=1e-3)


# ---- the metric: over the travel BOTH stations grade, on ONE position axis ----------------------

def test_a_final_test_counted_from_another_zero_is_compared_where_it_was_measured(db):
    # The 8340-1 shape: final test counts 0..20, the laser -10..10 -- the same travel. Both
    # stations' worst point is index 5 (laser position -5). Shifted onto the laser's axis the
    # ratios are 1/3 and 1.0; compared raw, only 0..10 would overlap and each ratio would divide by
    # the laser's 0.01 baseline (10 and 30).
    shifted = [p + 10.0 for p in POS]
    nxt = seed(db, 40, 0, at=5, ft_positions=shifted)
    seed(db, 0, 90, at=5, ft_positions=shifted, first_shop=nxt)
    facts, findings = run(db)
    assert only(findings).n_units == 40
    assert facts["skipped_pairs"] == 0
    assert facts["median_ratio_rework"] == pytest.approx(1 / 3, abs=1e-3)
    assert facts["median_ratio_control_top_third"] == pytest.approx(1.0, abs=1e-3)


def test_a_pair_with_no_common_graded_position_is_skipped_and_counted(db):
    # Five more reworked units whose final test grades only the first five positions (-10..-6)
    # while the laser grades only its last nine (2..10): no position both stations grade. They
    # are counted as skipped and never scored -- a zero-tolerance metric has no reading there.
    nxt = seed(db, 40, 90)
    with db.session() as s:
        for k in range(5):
            aid = trim_file(s, nxt + k, START.replace(hour=9), "Track A", False, spike(0.30, at=15),
                            upper=[None] * 12 + [BAND] * 9, lower=[None] * 12 + [-BAND] * 9)
            final_test(s, nxt + k, START + timedelta(days=2), aid, spike(0.10, at=2), window=(0, 4))
    facts, findings = run(db)
    assert only(findings).n_units == 45
    assert facts["skipped_pairs"] == 5
    assert facts["rework_unit_days"] == 45 and facts["rework_ratio_n"] == 40


def test_each_station_is_read_where_it_grades_with_its_own_offset():
    # Trim: its worst raw point (0.40, index 0) carries no limits -- the laser never graded it.
    # Final test: its worst point (0.35, index 20) is outside its graded window 1..19. Each side
    # carries its OWN offset (laser -0.05, final test +0.02); swapping them changes both answers.
    trim = {"positions": POS, "errors": [0.40] + spike(0.25)[1:], "upper": [None] + UP[1:],
            "lower": [None] + LO[1:], "offset": -0.05, "slope": 0.0, "theory": None}
    ft = {"positions": POS, "errors": spike(0.08)[:-1] + [0.35], "upper": UP, "lower": LO,
          "offset": 0.02, "slope": 0.0, "theory": None, "graded_start": 1, "graded_end": 19}
    got = rework_load.graded_maxima(ft, trim)
    assert got == pytest.approx((0.10, 0.20))           # 0.08 + 0.02 ; 0.25 - 0.05


def test_the_stored_slope_is_applied_where_theory_is_stored():
    # corrected = error + theory * k + offset, the one graded-trace definition (unit_chart).
    theory = [0.1 * i for i in range(len(POS))]
    trim = {"positions": POS, "errors": [0.0] * len(POS), "upper": UP, "lower": LO,
            "offset": 0.0, "slope": 0.1, "theory": theory}
    ft = {"positions": POS, "errors": spike(0.05), "upper": UP, "lower": LO, "offset": 0.0,
          "slope": 0.0, "theory": None, "graded_start": 0, "graded_end": 20}
    ft_max, laser_max = rework_load.graded_maxima(ft, trim)
    assert laser_max == pytest.approx(0.2)              # theory 2.0 at the last point x k 0.1
    assert ft_max == pytest.approx(0.05)


def test_a_blank_reading_is_ungraded_never_zero():
    trim = {"positions": POS, "errors": [None] * len(POS), "upper": UP, "lower": LO,
            "offset": 0.0, "slope": 0.0, "theory": None}
    ft = {"positions": POS, "errors": spike(0.05), "upper": UP, "lower": LO, "offset": 0.0,
          "slope": 0.0, "theory": None, "graded_start": 0, "graded_end": 20}
    assert rework_load.graded_maxima(ft, trim) is None


# ---- a reading of 1 V or more is junk: ignored, and counted -------------------------------------

def test_the_junk_rule_is_one_volt_either_side():
    assert rework_load._ratio(0.10, 0.999) == (pytest.approx(0.10 / 0.999), False)
    assert rework_load._ratio(0.10, 1.0) == (None, True)
    assert rework_load._ratio(1.0, 0.30) == (None, True)
    assert rework_load._ratio(0.999, 0.30) == (pytest.approx(0.999 / 0.30), False)
    assert rework_load._ratio(0.10, 0.0) == (None, False)          # no ratio of a zero reading


def test_junk_readings_are_ignored_and_counted(db):
    nxt = seed(db, 40, 90)
    nxt = seed(db, 3, 0, rework=(0.30, 1.2), first_shop=nxt)        # final test reads 1.2 V
    seed(db, 0, 2, control=(1.5, 0.30), first_shop=nxt)             # laser reads 1.5 V
    facts, findings = run(db)
    assert only(findings).n_units == 43
    assert facts["junk_readings"] == 5
    assert facts["rework_ratio_n"] == 40 and facts["control_n"] == 90
    assert facts["median_ratio_rework"] == pytest.approx(1 / 3, abs=1e-3)


# ---- a record that failed processing is not a measurement ---------------------------------------

def test_a_file_that_failed_processing_is_never_the_last_attempt_nor_the_verdict(db):
    # Every unit-day also carries a LATER Track A file the analyser could not read (ERROR, no
    # verdict, 0.9 V in its sweep). Were it the "last attempt" it would be paired (ratio 0.1/0.9)
    # and its missing verdict would make the pass/pass units laser FAILs. Neither happens.
    with db.session() as s:
        for k in range(130):
            passed = k >= 40
            day = START + timedelta(days=k)
            aid = trim_file(s, k + 1, day.replace(hour=9), "Track A", passed, spike(0.30))
            trim_file(s, k + 1, day.replace(hour=9, minute=30), "Track A", passed, spike(0.9),
                      status=StatusType.ERROR)
            final_test(s, k + 1, day + timedelta(days=2), aid, spike(0.30 if passed else 0.10))
        for k in range(5):                  # final tests linked ONLY to an unreadable file
            aid = trim_file(s, 500 + k, START.replace(hour=9), "Track A", False, spike(0.9),
                            status=StatusType.ERROR)
            final_test(s, 500 + k, START + timedelta(days=2), aid, spike(0.10))
    facts, findings = run(db)
    assert only(findings).n_units == 40
    assert facts["rework_unit_days"] == 40 and facts["linked"] == 130
    assert facts["rework_ratio_n"] == 40 and facts["control_n"] == 90
    assert facts["median_ratio_rework"] == pytest.approx(1 / 3, abs=1e-3)
    assert facts["median_ratio_control_top_third"] == pytest.approx(1.0, abs=1e-3)


# ---- no linked final tests: nothing, and facts say so -------------------------------------------

def test_no_linked_final_tests_is_nothing_with_facts_saying_so(db):
    with db.session() as s:                         # trim data only -- nothing is linked
        trim_file(s, 1, START.replace(hour=9), "Track A", False, spike(0.30))
    facts, findings = run(db)
    assert findings == []
    assert facts != {}
    assert facts["linked"] == 0 and facts["rework_unit_days"] == 0
    assert facts["confirmed"] is False and facts["note"]


# ---- nothing to report on: the database is never even queried -----------------------------------

def test_no_tracks_the_database_is_never_queried():
    assert rework_load.analyze(MODEL, _Boom(), [], label) == ({}, [])


def test_undated_tracks_the_database_is_never_queried():
    from dataclasses import replace
    tracks = [replace(t, file_date=None) for t in tracks_for(5)]
    assert rework_load.analyze(MODEL, _Boom(), tracks, label) == ({}, [])


# ---- a crash is never swallowed here -----------------------------------------------------------

def test_a_raising_agreement_lookup_is_not_caught_here(db, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(db, "get_model_trim_ft_agreement", boom)
    with pytest.raises(RuntimeError, match="database is locked"):
        run(db)


# ---- pairing a final-test record to a track ----------------------------------------------------

@pytest.mark.parametrize("serial,letter", [
    ("16", "A"), ("16A", "A"), ("16a", "A"), ("16B", "B"), ("16b", "B"),
    ("16l", None), ("SN", None), ("", None), (None, None), ("16AB", None)])
def test_a_final_test_serial_names_its_track(serial, letter):
    assert rework_load.ft_track_letter(serial) == letter


# ---- systems: every laser the rework ran on, named in shop order --------------------------------

def test_the_title_names_every_laser_the_rework_ran_on_in_shop_order(db):
    nxt = seed(db, 20, 90)                                            # laser 1 (B)
    seed(db, 20, 0, system=SystemType.A, first_shop=nxt)              # laser 2 (A)
    f = only(run(db)[1])
    assert f.systems == ("B", "A")                                    # laser 1 first, not "A" first
    assert f.title == ("Laser 1 (LTS) and Laser 2 (DLTS): 40 units a year fail here and pass "
                       "final test after rework")


# ---- presentation: laser_time group, readout in unit-days --------------------------------------

def test_group_and_readout_through_presentation(db):
    seed(db, 40, 90)
    f = only(run(db)[1])
    d = f.to_dict()
    assert P.group_key(d) == "laser_time"
    assert P.readout(d) == 40.0
    assert P.statement(d) == f.title
    # The group's column counts TRACKS; this readout counts unit-days, and says so on its row.
    assert P.value_text("laser_time", 40.0, [d]) == "40 unit-days"
