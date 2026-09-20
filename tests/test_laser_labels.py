"""The shop's laser numbers do not follow the code's letters.

James, 2026-09-20: "DLTS is laser 2, LTS is laser 1 and LTS3 is laser 3."
System A is the DLTS format, B is LTS, C is LTS3 -- so A is laser TWO. An
assistant read A/B/C as 1/2/3 for a whole day; this pins the translation.
"""
from laser_trim_analyzer.core.models import (
    LASER_LABELS, LASER_ORDER, SystemType, laser_label)


def test_the_shop_numbering_is_not_alphabetical():
    assert laser_label(SystemType.A) == "Laser 2 (DLTS)"
    assert laser_label(SystemType.B) == "Laser 1 (LTS)"
    assert laser_label(SystemType.C) == "Laser 3 (LTS3)"


def test_letters_and_enums_give_the_same_label():
    for member in SystemType:
        assert laser_label(member) == laser_label(member.value)


def test_missing_or_unknown_is_never_a_wrong_laser():
    assert laser_label(None) == "N/A"
    assert laser_label("") == "N/A"
    assert laser_label("Z") == "Z"          # shown as-is, never guessed


def test_every_system_has_a_label_and_a_place_in_shop_order():
    letters = {m.value for m in SystemType}
    assert set(LASER_LABELS) == letters      # a 4th system cannot ship unlabeled
    assert set(LASER_ORDER) == letters - {"Unknown"}   # only real machines get a place
    assert laser_label(SystemType.UNKNOWN if hasattr(SystemType, "UNKNOWN") else "Unknown") == "Unknown laser"
    assert [laser_label(s) for s in LASER_ORDER] == [
        "Laser 1 (LTS)", "Laser 2 (DLTS)", "Laser 3 (LTS3)"]
