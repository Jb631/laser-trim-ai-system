from datetime import datetime

import pytest

from laser_trim_analyzer.findings.analyzers import recipe_change
from findings_helpers import START, era, label


def test_a_recipe_change_that_moved_the_result_is_a_finding():
    tracks = era(0, START, 240, (1.0,), 0.25) + era(1000, datetime(2024, 7, 1), 240, (1.0, 2.0), 0.55)
    history, findings = recipe_change.analyze("M", tracks, label)
    assert len(history) == 2
    assert len(findings) == 1
    f = findings[0]
    assert "1 cut" in f.title and "2 cuts" in f.title and f.lever == "laser_settings"
    assert f.expected_gain_points is None                 # a detection never claims a gain
    assert f.evidence["moved_points"] == pytest.approx(30.0, abs=3.0)
    assert "not the only thing that changed" in f.summary

def test_a_constant_recipe_says_nothing():
    history, findings = recipe_change.analyze("M", era(0, START, 400, (1.0,), 0.4), label)
    assert findings == [] and len(history) == 1

def test_a_change_that_moved_nothing_is_history_not_a_finding():
    tracks = era(0, START, 240, (1.0,), 0.40) + era(1000, datetime(2024, 7, 1), 240, (1.0, 2.0), 0.42)
    history, findings = recipe_change.analyze("M", tracks, label)
    assert findings == [] and len(history) == 2

def test_a_change_resting_on_a_thin_side_says_nothing():
    tracks = era(0, START, 40, (1.0,), 0.10) + era(1000, datetime(2024, 7, 1), 240, (1.0, 2.0), 0.60)
    _, findings = recipe_change.analyze("M", tracks, label)
    assert findings == []


def test_a_recipe_change_that_coincides_with_a_new_limit_table_says_so():
    """One cut on a STRICT table at 25%, then two cuts on a LAX table at 60%. The 35-point move is two
    changes at once, and only one of them is the recipe."""
    from datetime import datetime
    from findings_helpers import on_table, table
    strict, lax = table(23, 0.10), table(12, 0.10)
    before = [on_table(t, strict) for t in era(0, START, 400, (1.0,), 0.25)]
    after = [on_table(t, lax) for t in era(1000, datetime(2024, 10, 1), 400, (1.0, 2.0), 0.60)]
    _, findings = recipe_change.analyze("M", before + after, label)
    (f,) = findings
    assert f.evidence["limit_table_changed"] is True and f.evidence["same_table"] is None
    assert "The limit table changed too" in f.summary and "23 points before and 12 after" in f.summary
    assert "Neither table has enough tracks on both sides" in f.summary
    assert f.evidence["before"]["limit_table"]["graded"] == 23 and f.evidence["after"]["limit_table"]["graded"] == 12


def test_the_like_for_like_move_is_given_when_one_table_spans_the_change():
    from datetime import datetime
    from findings_helpers import on_table, table
    strict, lax = table(23, 0.10), table(12, 0.10)
    before = [on_table(t, strict) for t in era(0, START, 400, (1.0,), 0.25)]
    after_lax = [on_table(t, lax) for t in era(1000, datetime(2024, 10, 1), 300, (1.0, 2.0), 0.70)]
    after_strict = [on_table(t, strict) for t in era(2000, datetime(2024, 10, 1), 100, (1.0, 2.0), 0.40)]
    _, findings = recipe_change.analyze("M", before + after_lax + after_strict, label)
    (f,) = findings
    same = f.evidence["same_table"]
    assert same["graded"] == 23 and (same["before_pct"], same["after_pct"]) == (25.0, 40.0)
    assert "On the 23-point table alone the move was 25% (400 tracks) to 40% (100)" in f.summary


def test_an_unchanged_limit_table_adds_nothing_to_the_sentence():
    from datetime import datetime
    from findings_helpers import on_table, table
    one = table(12, 0.10)
    tracks = ([on_table(t, one) for t in era(0, START, 400, (1.0,), 0.25)]
              + [on_table(t, one) for t in era(1000, datetime(2024, 10, 1), 400, (1.0, 2.0), 0.60)])
    (f,) = recipe_change.analyze("M", tracks, label)[1]
    assert f.evidence["limit_table_changed"] is False and "limit table" not in f.summary
