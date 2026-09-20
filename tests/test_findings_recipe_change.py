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
