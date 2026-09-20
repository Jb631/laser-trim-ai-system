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
    assert "No single table has 100 graded tracks on both sides" in f.summary
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
    assert ("On the 23-point table alone the move was 25% (400 tracks, 2024-01-01 to 2024-07-18) to "
            "40% (100 tracks, 2024-10-01 to 2024-11-19) -- the tracks still graded that way, not the whole "
            "after-period.") in f.summary
    assert same["after_share"] == 0.25 and same["after_first"] == "2024-10-01"


def test_an_unchanged_limit_table_adds_nothing_to_the_sentence():
    from datetime import datetime
    from findings_helpers import on_table, table
    one = table(12, 0.10)
    tracks = ([on_table(t, one) for t in era(0, START, 400, (1.0,), 0.25)]
              + [on_table(t, one) for t in era(1000, datetime(2024, 10, 1), 400, (1.0, 2.0), 0.60)])
    (f,) = recipe_change.analyze("M", tracks, label)[1]
    assert f.evidence["limit_table_changed"] is False and "limit table" not in f.summary


def test_a_change_in_the_MIX_of_limit_tables_is_disclosed_even_when_the_busiest_is_the_same():
    """Review finding. Before: half the tracks on a lax table (passing 60%), half on a strict one (passing 10%).
    After: everything on the lax table, still passing 60%. The pooled figures say +25 points; the like-for-like
    move on the lax table is ZERO. The first version disclosed nothing, because the busiest table did not change."""
    from datetime import datetime
    from findings_helpers import on_table, table
    strict, lax = table(23, 0.10), table(12, 0.10)
    before = ([on_table(t, lax) for t in era(0, START, 200, (1.0,), 0.60)]
              + [on_table(t, strict) for t in era(500, START, 200, (1.0,), 0.10)])
    after = [on_table(t, lax) for t in era(1000, datetime(2024, 10, 1), 400, (1.0, 2.0), 0.60)]
    (f,) = recipe_change.analyze("M", before + after, label)[1]
    assert f.evidence["limit_table_changed"] is False and f.evidence["limit_tables_mixed"] is True
    assert "More than one limit table was in use (2 before, 1 after)" in f.summary
    assert "may be a change in which test was applied, not in the parts" in f.summary
    same = f.evidence["same_table"]
    assert same["graded"] == 12 and same["before_pct"] == same["after_pct"] == 60.0
    assert "On the 12-point table alone the move was 60%" in f.summary and "to 60% (400 tracks" in f.summary
    assert f.evidence["moved_points"] == pytest.approx(25.0, abs=1.0)        # what the pooled figures claimed
