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
