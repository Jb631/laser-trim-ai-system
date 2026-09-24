"""Readability is a number, and this pins it (spec 2026-09-23, sections 1 and 4).

James said the app was "hard to read". Measured, three of the old colour pairs sat below the
WCAG 4.5:1 minimum -- dimmed text 2.8, the accent on a card 3.5, the out-of-control tier 4.2.
These tests stop a later colour change from quietly putting one back.
"""
import colorsys

import pytest

from laser_trim_analyzer.gui.v6.theme import ThemeManager


def _lum(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def hue(h):
    h = h.lstrip("#")
    r, g, b = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return colorsys.rgb_to_hls(r, g, b)[0] * 360


T = ThemeManager()          # builds without a Tk root: family resolution just falls back
SURFACES = ("BG", "SURFACE", "CARD", "ELEVATED")


@pytest.mark.parametrize("surface", SURFACES)
def test_primary_text_reads_comfortably_on_every_surface(surface):
    assert contrast(T.TEXT_PRIMARY, getattr(T, surface)) >= 7.0


@pytest.mark.parametrize("text", ("TEXT_SECONDARY", "TEXT_DISABLED", "ACCENT"))
@pytest.mark.parametrize("surface", SURFACES)
def test_every_other_text_colour_clears_the_minimum_on_every_surface(text, surface):
    # TEXT_DISABLED included on purpose: this app uses it for real information (chart tick
    # labels, the volume axis), not only for disabled controls. #7b8aa0 failed here at 4.1 on
    # a card and 3.5 on ELEVATED, which is why it is #93a1b6.
    assert contrast(getattr(T, text), getattr(T, surface)) >= 4.5, f"{text} on {surface}"


@pytest.mark.parametrize("fg,bg", [
    ("PASS_FG", "PASS_BG"), ("FAIL_FG", "FAIL_BG"), ("NEUTRAL_FG", "NEUTRAL_BG"),
    ("WATCH_FG", "WATCH_BG"), ("CHECK", "CHECK_TINT"), ("ACCENT", "ACCENT_TINT"),
    ("TIER_WARNING", "TIER_WARNING_BG"), ("TIER_DRIFT", "TIER_DRIFT_BG"),
    ("TIER_OOC", "TIER_OOC_BG"),
])
def test_every_badge_pill_and_tier_reads_on_its_own_tint(fg, bg):
    assert contrast(getattr(T, fg), getattr(T, bg)) >= 4.5, f"{fg} on {bg}"


def test_text_on_a_teal_button_is_dark_and_readable():
    assert contrast(T.TEXT_INVERSE, T.ACCENT) >= 4.5
    assert contrast("#ffffff", T.ACCENT) < 4.5        # why it has to be dark


@pytest.mark.parametrize("line", ("ACCENT", "CHART_REFERENCE", "CHECK", "SERIES_A", "SERIES_B", "SERIES_C"))
def test_chart_lines_stand_out_from_a_card(line):
    assert contrast(getattr(T, line), T.CARD) >= 3.0  # WCAG graphics minimum


def test_pass_can_never_be_mistaken_for_the_accent():
    d = abs((hue(T.PASS_FG) - hue(T.ACCENT) + 180) % 360 - 180)
    assert d >= 45, f"PASS green and the accent are only {d:.0f} degrees apart"


def test_every_laser_series_keeps_clear_of_every_colour_that_means_something():
    meaning = [T.ACCENT, T.PASS_FG, T.FAIL_FG, T.WATCH_FG, T.TIER_OOC]
    for s in (T.SERIES_A, T.SERIES_B, T.SERIES_C):
        near = min(abs((hue(s) - hue(m) + 180) % 360 - 180) for m in meaning)
        assert near >= 30, f"{s} is {near:.0f} degrees from a colour that carries meaning"


def test_series_color_maps_each_laser_and_falls_back_for_unknown():
    assert T.series_color("A") == T.SERIES_A and T.series_color("B") == T.SERIES_B
    assert T.series_color("C") == T.SERIES_C and T.series_color("?") == T.CHART_REFERENCE
