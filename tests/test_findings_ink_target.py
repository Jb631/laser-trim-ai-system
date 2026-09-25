from dataclasses import replace
from datetime import datetime

from laser_trim_analyzer.findings.analyzers import ink_target
from findings_helpers import START, ink_tracks, label


def test_resistance_that_separates_good_from_bad_is_a_finding_with_a_defined_gain():
    tracks = ink_tracks(600, START, (1.0, 2.0), lambda r: 0.75 if r < 4400 else 0.25)
    (f,) = ink_target.analyze("M", tracks, label)
    assert f.lever == "ink" and "lower" in f.title
    assert f.expected_gain_points > 10 and f.gain_definition
    assert f.strength_value < -0.10
    assert len(f.evidence["bins"]) == 5 and f.evidence["window"]["n"] >= 0.4 * 600

def test_resistance_that_separates_nothing_says_nothing():
    assert ink_target.analyze("M", ink_tracks(600, START, (1.0, 2.0), lambda r: 0.5), label) == []

def test_a_thin_sample_says_nothing():
    assert ink_target.analyze("M", ink_tracks(150, START, (1.0, 2.0), lambda r: 0.9 if r < 4400 else 0.1), label) == []

def test_a_recipe_change_cannot_masquerade_as_a_resistance_effect():
    # Era 1: one cut, HIGH resistance, 20% good.  Era 2: two cuts, LOW resistance, 60% good.
    # Pooled, low resistance "predicts" success. Inside either era it predicts nothing.
    old = ink_tracks(500, START, (1.0,), lambda r: 0.20, first_id=0, r_lo=4500.0, r_hi=5000.0)
    new = ink_tracks(500, datetime(2025, 1, 1), (1.0, 2.0), lambda r: 0.60, first_id=5000, r_lo=4000.0, r_hi=4500.0)
    from laser_trim_analyzer.findings.stats import spearman
    pooled = spearman([t.untrimmed_resistance for t in old + new], [1.0 if t.linearity_pass else 0.0 for t in old + new])
    assert pooled < -0.25                                  # the trap is really there...
    assert ink_target.analyze("M", old + new, label) == [] # ...and the analyzer does not fall in

def test_the_configured_window_reported_is_the_most_recent_one_not_the_highest_resistance_ones():
    tracks = ink_tracks(600, START, (1.0, 2.0), lambda r: 0.75 if r < 4400 else 0.25)
    newest = max(tracks, key=lambda t: t.file_date)
    highest = max(tracks, key=lambda t: t.untrimmed_resistance)
    assert newest.track_id != highest.track_id          # otherwise this test proves nothing
    tracks = [replace(t, initial_r_low=4000.0, initial_r_high=4400.0) if t.track_id == newest.track_id
              else replace(t, initial_r_low=4200.0, initial_r_high=4600.0) for t in tracks]
    (f,) = ink_target.analyze("M", tracks, label)
    assert f.evidence["configured_incoming"] == (4000.0, 4400.0)
    assert "4,000 to 4,400" in f.summary and "4,200 to 4,600" not in f.summary


def test_no_configured_window_is_said_plainly():
    (f,) = ink_target.analyze("M", ink_tracks(600, START, (1.0, 2.0), lambda r: 0.75 if r < 4400 else 0.25), label)
    assert f.evidence["configured_incoming"] is None
    assert "no configured incoming window" in f.summary


def test_a_change_of_limit_table_is_not_credited_to_resistance():
    """Two tables, two eras. The LAX table passes 80% and happens to run on low-resistance stock; the
    STRICT one passes 20% on high-resistance stock. Inside each table resistance does nothing at all.
    Pooled, resistance 'explains' a 60-point swing (rho about -0.6) -- which is the limit table talking."""
    from dataclasses import replace
    from datetime import datetime
    from findings_helpers import ink_tracks, on_table, table
    from laser_trim_analyzer.findings.analyzers import ink_target
    from laser_trim_analyzer.findings.stats import spearman
    strict, lax = table(23, 0.10), table(12, 0.10)
    a = [on_table(t, strict) for t in ink_tracks(400, START, (1.0, 2.0), lambda r: 0.2, r_lo=4500.0, r_hi=5000.0)]
    b = [on_table(t, lax) for t in ink_tracks(400, datetime(2025, 1, 1), (1.0, 2.0), lambda r: 0.8,
                                              first_id=1000, r_lo=4000.0, r_hi=4500.0)]
    pooled_rho = spearman([t.untrimmed_resistance for t in a + b], [1.0 if t.linearity_pass else 0.0 for t in a + b])
    assert pooled_rho < -0.4                                # the trap is real: pooled, it looks like a strong lever
    assert ink_target.analyze("M", a + b, label) == []      # held constant, there is nothing to say
    # ...and the table travels with a finding when there is one
    hot = [on_table(t, lax) for t in ink_tracks(600, START, (1.0, 2.0), lambda r: 0.75 if r < 4400 else 0.25)]
    f = ink_target.analyze("M", hot, label)[0]
    assert f.evidence["limit_table"] == {"rows": 12, "graded": 12}
    assert "Laser, recipe, limit table and the station's final-resistance window are held constant" in f.summary


def test_a_change_of_final_resistance_window_is_not_credited_to_resistance_either():
    """Second review finding. `_success` also asks whether the trimmed resistance landed inside the
    station's FINAL window -- so that window is part of the test. One laser, one recipe, one limit table;
    resistance does nothing inside either era; only the final window moved. Pooled, the analyzer used to
    answer "aim lower" with rho about -0.87 and a claimed gain of 50 yield points."""
    from dataclasses import replace
    from datetime import datetime
    from findings_helpers import ink_tracks
    from laser_trim_analyzer.findings.analyzers import ink_target
    from laser_trim_analyzer.findings.stats import spearman
    # Era A: a NARROW final window that the trimmed resistance misses, on low incoming stock.
    a = [replace(t, final_r_low=9000.0, final_r_high=9100.0)
         for t in ink_tracks(400, START, (1.0, 2.0), lambda r: 1.0, r_lo=4000.0, r_hi=4400.0)]
    # Era B: the window widened to include it, on high incoming stock. Same recipe, same table.
    b = [replace(t, final_r_low=4000.0, final_r_high=9000.0)
         for t in ink_tracks(400, datetime(2025, 1, 1), (1.0, 2.0), lambda r: 1.0,
                             first_id=1000, r_lo=4400.0, r_hi=5000.0)]
    ok = lambda ts: [1.0 if ink_target._success(t) else 0.0 for t in ts]        # noqa: E731
    assert sum(ok(a)) == 0 and sum(ok(b)) == len(b)          # the window alone decides the verdict
    pooled = spearman([t.untrimmed_resistance for t in a + b], ok(a + b))
    assert pooled > 0.8                                       # the trap: resistance "explains" everything
    assert ink_target.analyze("M", a + b, label) == []         # held constant, there is nothing to say


def test_the_window_that_is_held_constant_travels_with_the_finding():
    from findings_helpers import ink_tracks
    from laser_trim_analyzer.findings.analyzers import ink_target
    hot = ink_tracks(600, START, (1.0, 2.0), lambda r: 0.75 if r < 4400 else 0.25)
    (f,) = ink_target.analyze("M", hot, label)
    assert f.evidence["final_resistance_window"] == [5000.0, 5500.0]
    assert "final-resistance window are held constant" in f.summary
    assert "Period, operator and lot are not." in f.summary


def test_noise_almost_never_produces_a_finding_at_the_minimum_sample():
    """Final review, measured: the recommended window is a maximum over seven overlapping candidates,
    so with no out-of-sample test the analyzer spoke about PURE NOISE 14% of the time at n=200 (median
    claimed gain 7 points) and 2% at n=600. With the two-fold test it is 1% at 600 and 0% at 900.
    60 draws at the floor; the bound is deliberately loose (5%) so this test measures the guard rather
    than the seed -- remove the guard and lower the floor and it fails by a mile."""
    import random
    from laser_trim_analyzer.findings.analyzers import ink_target
    from findings_helpers import make_track, days
    spoke = 0
    for seed in range(60):
        rnd = random.Random(seed * 7919)
        tracks = [make_track(k, date=d, passes=((0.8 if rnd.random() < 0.5 else 1.5, 1.0),),
                             r_in=rnd.uniform(4000.0, 5000.0))
                  for k, d in enumerate(days(START, ink_target.MIN_N, 0.5))]
        spoke += bool(ink_target.analyze("M", tracks, label))
    assert spoke <= 3, f"{spoke} of 60 pure-noise draws produced a finding"


def test_a_window_that_only_works_in_the_period_it_was_chosen_from_is_not_reported():
    """The gate that makes the noise rate what it is, pinned on its own: a relationship that exists in
    the first half of the period and NOT in the second must not be reported, however strong it looks
    over the whole group."""
    from datetime import datetime
    from laser_trim_analyzer.findings.analyzers import ink_target
    from findings_helpers import ink_tracks
    real = ink_tracks(400, START, (1.0, 2.0), lambda r: 0.9 if r < 4400 else 0.1)
    gone = ink_tracks(400, datetime(2025, 1, 1), (1.0, 2.0), lambda r: 0.5, first_id=5000)
    assert ink_target.analyze("M", real + gone, label) == []
    # ...while the same strength present THROUGHOUT is reported, and the number is the out-of-sample one
    lasting = ink_tracks(800, START, (1.0, 2.0), lambda r: 0.9 if r < 4400 else 0.1)
    (f,) = ink_target.analyze("M", lasting, label)
    folds = f.evidence["out_of_sample_gain_each_fold"]
    assert len(folds) == 2 and min(folds) >= ink_target.MIN_GAIN_POINTS * ink_target.OUT_OF_SAMPLE
    assert f.expected_gain_points == round(sum(folds) / 2, 1)
    assert "measured OUT OF SAMPLE" in f.summary


def test_a_resistance_that_is_not_a_resistance_is_not_a_measurement():
    """The work database holds 1e12 Ω readings. One of them drags the quantile edges, so it does not
    merely add a row -- it can put a trillion ohms in the recommended window."""
    from laser_trim_analyzer.findings.analyzers import ink_target
    from findings_helpers import ink_tracks, make_track, days
    hot = ink_tracks(700, START, (1.0, 2.0), lambda r: 0.75 if r < 4400 else 0.25)
    junk = [make_track(9000 + k, date=d, passes=((0.8, 1.0),), r_in=1e12)
            for k, d in enumerate(days(START, 120, 0.5))]
    (clean,) = ink_target.analyze("M", hot, label)
    (with_junk,) = ink_target.analyze("M", hot + junk, label)
    assert with_junk.title == clean.title                   # the junk changed nothing at all
    assert with_junk.n_units == clean.n_units == 700
    assert "1,000,000,000,000" not in with_junk.title and "1e+12" not in with_junk.title


def test_the_rate_is_scaled_by_the_finding_s_own_group_not_the_whole_model():
    from datetime import datetime
    from laser_trim_analyzer.findings.analyzers import ink_target
    from findings_helpers import ink_tracks, table, on_table
    other = table(31, 0.10)
    hot = ink_tracks(700, START, (1.0, 2.0), lambda r: 0.75 if r < 4400 else 0.25)
    elsewhere = [on_table(t, other) for t in
                 ink_tracks(3000, datetime(2020, 1, 1), (1.0,), lambda r: 0.5, first_id=50000)]
    (f,) = ink_target.analyze("M", hot + elsewhere, label)
    assert f.n_units == 700 and f.scope_annual_tracks <= 700          # never the other 3,000
    assert f.tracks_per_year == f.expected_gain_points / 100.0 * f.scope_annual_tracks
    assert "which is what the rate is scaled by" in f.summary


def test_advice_drawn_from_a_setup_the_model_no_longer_runs_says_so():
    """With MIN_N at 600 the biggest eligible group can be an older recipe, while the model has since
    moved on to one too thin to judge. The finding must not read as current advice."""
    from datetime import datetime
    from laser_trim_analyzer.findings.analyzers import ink_target
    from findings_helpers import ink_tracks
    old_era = ink_tracks(800, START, (1.0,), lambda r: 0.9 if r < 4400 else 0.1)
    now = ink_tracks(200, datetime(2026, 1, 1), (1.0, 2.0), lambda r: 0.5, first_id=9000)   # < MIN_N
    (f,) = ink_target.analyze("M", old_era + now, label)
    assert f.evidence["superseded"] is True
    assert "NOT the setup the model runs today" in f.summary and "Treat it as history" in f.summary
    # ...and when the eligible group IS the current one, that sentence is absent
    (g,) = ink_target.analyze("M", old_era, label)
    assert g.evidence["superseded"] is False and "NOT the setup" not in g.summary


# ---- #8: the configured window can disagree with the one the data says did best -------------------

def _yield_tags(finding_dict):
    from laser_trim_analyzer.findings.presentation import arrange
    rows = [r for g in arrange([finding_dict]) if g.spec.key == "yield" for r in g.rows]
    assert len(rows) == 1                       # otherwise this helper is not asking what it thinks
    return rows[0].tags


def test_the_overlap_test_treats_touching_closed_intervals_as_overlapping():
    """Pinned on its own, ahead of any analyzer scenario, because it is what a mutation check bites
    on: closed intervals share their endpoints, so touching at one point is NOT a disagreement."""
    from laser_trim_analyzer.findings.analyzers.ink_target import _overlaps
    assert _overlaps((4000.0, 4400.0), (4400.0, 4800.0)) is True     # share the single point 4400
    assert _overlaps((4400.0, 4800.0), (4000.0, 4400.0)) is True     # order does not matter
    assert _overlaps((4000.0, 4400.0), (4400.001, 4800.0)) is False  # one thousandth apart: disjoint
    assert _overlaps((4800.0, 5000.0), (4000.0, 4400.0)) is False    # wholly outside, reversed order
    assert _overlaps((4000.0, 5000.0), (4200.0, 4300.0)) is True     # one wholly nested in the other


def test_a_recommended_window_wholly_outside_the_configured_one_disagrees():
    tracks = ink_tracks(600, START, (1.0, 2.0), lambda r: 0.75 if r < 4400 else 0.25)
    tracks = [replace(t, initial_r_low=4600.0, initial_r_high=4900.0) for t in tracks]
    (f,) = ink_target.analyze("M", tracks, label)
    assert f.evidence["window"]["r_high"] < 4600.0            # otherwise this test proves nothing
    assert f.evidence["configured_disagrees"] is True
    assert ("The station is set to accept 4,600 to 4,900 Ω, and the window that did best lies "
            "outside it.") in f.summary
    assert "outside the configured window" in _yield_tags(f.to_dict())


def test_an_overlapping_configured_window_does_not_disagree():
    tracks = ink_tracks(600, START, (1.0, 2.0), lambda r: 0.75 if r < 4400 else 0.25)
    tracks = [replace(t, initial_r_low=4200.0, initial_r_high=4600.0) for t in tracks]
    (f,) = ink_target.analyze("M", tracks, label)
    assert f.evidence["window"]["r_high"] > 4200.0             # otherwise this test proves nothing
    assert f.evidence["configured_disagrees"] is False
    assert "The station is set to accept 4,200 to 4,600 Ω incoming." in f.summary
    assert "lies outside it" not in f.summary
    assert "outside the configured window" not in _yield_tags(f.to_dict())


def test_no_configured_window_means_no_disagreement_and_no_tag():
    (f,) = ink_target.analyze(
        "M", ink_tracks(600, START, (1.0, 2.0), lambda r: 0.75 if r < 4400 else 0.25), label)
    assert f.evidence["configured_incoming"] is None
    assert f.evidence.get("configured_disagrees") is None
    assert "outside the configured window" not in _yield_tags(f.to_dict())
