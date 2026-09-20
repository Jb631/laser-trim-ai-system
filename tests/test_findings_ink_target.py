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
