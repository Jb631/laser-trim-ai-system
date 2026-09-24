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


@pytest.mark.parametrize("fill", ("SEGMENT_SELECTED", "SEGMENT_SELECTED_HOVER"))
def test_a_selected_tab_carries_light_text_and_still_shows(fill):
    # CTkSegmentedButton has ONE text colour for every segment, so the selected fill must suit
    # the same light text the unselected CARD segments carry -- dark text would fail on CARD
    # (TEXT_INVERSE on CARD is 1.18:1) -- and it must still stand apart from CARD.
    assert contrast(T.TEXT_PRIMARY, getattr(T, fill)) >= 4.5, f"TEXT_PRIMARY on {fill}"
    assert contrast(getattr(T, fill), T.CARD) >= 2.0, f"{fill} barely differs from CARD"


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


# ---- What the widgets actually DRAW (final review, 2026-09-24) -----------------------------------
# Every test above checks a pair of TOKENS. They all passed while every selected tab and segment in
# the app drew TEXT_PRIMARY on ACCENT -- 1.66:1 -- and every checkbox drew CustomTkinter's own
# default checkmark on the teal (about 1.4:1), because nothing asked which two colours a widget is
# really given. These build the real app and read the colours back off the widgets themselves.

def _hex(widget, color) -> str:
    """A CTk colour as #rrggbb. A (light, dark) pair resolves to dark -- V6 only ever runs dark
    (gui/v6/app.py) -- and a Tk colour NAME (CustomTkinter's own defaults, e.g. "gray90") is
    resolved by Tk itself, so a default that slipped through is measured, not skipped."""
    if isinstance(color, (tuple, list)):
        color = color[1]
    r, g, b = widget.winfo_rgb(color)
    return f"#{r >> 8:02x}{g >> 8:02x}{b >> 8:02x}"


def _walk(widget):
    """Every widget Tk has under `widget`. tkinter's OWN winfo_children, not CustomTkinter's:
    CTkTabview.winfo_children() hides the tab view's segmented button -- its tabs -- as "part of
    the tab view itself", so the public walk never reaches a single tab."""
    import tkinter
    yield widget
    for child in tkinter.Misc.winfo_children(widget):
        yield from _walk(child)


def test_every_tab_and_segmented_control_the_app_draws_is_readable(make_app):
    """Each segment is a CTkButton the control recolours as it is selected; read the text and the
    fill -- and the hover fill -- off every segment button of every segmented control in the app
    (a CTkTabview's tabs are one too)."""
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.widgets.sensitivity_slider import SensitivitySlider

    app = make_app()
    pages = app.page_container
    widgets = list(_walk(app))
    segs = [w for w in widgets if isinstance(w, ctk.CTkSegmentedButton)]
    # A walk that reached nothing would pass everything below: prove it reaches every control.
    named = {"the Triage scope toggle": pages.get_page("triage")._scope,
             "the Model chart toggle": pages.get_page("model")._chart_toggle,
             "the Model page's tabs": pages.get_page("model")._tabs._segmented_button}
    sliders = [w for w in widgets if isinstance(w, SensitivitySlider)]
    assert sliders, "the Settings sensitivity slider was never built"
    named.update({f"sensitivity slider {i}": s._seg for i, s in enumerate(sliders)})
    for name, seg in named.items():
        assert seg in segs, f"the walk never reached {name}"

    bad = []
    for seg in segs:
        selected, unselected = seg.cget("selected_color"), seg.cget("unselected_color")
        buttons = list(seg._buttons_dict.items())
        assert any(b.cget("fg_color") == selected for _, b in buttons), \
            f"{seg}: no segment is drawn selected, so the selected pair was never measured"
        apart = contrast(_hex(seg, selected), _hex(seg, unselected))
        if apart < 2.0:
            bad.append(f"{seg}: the selected segment is only {apart:.2f}:1 from an unselected one")
        for value, b in buttons:
            text = _hex(b, b.cget("text_color"))
            for state, fill in (("", b.cget("fg_color")), (", hovered", b.cget("hover_color"))):
                ratio = contrast(text, _hex(b, fill))
                if ratio < 4.5:
                    bad.append(f"{seg} segment {value!r}{state}: {text} on {_hex(b, fill)} = {ratio:.2f}:1")
    assert not bad, "\n".join(bad)


def test_every_checkbox_the_app_draws_shows_its_checkmark(make_app):
    """A checkmark is a graphic, so the WCAG graphics minimum (3:1) against the checked fill --
    and against the fill it takes while hovered."""
    import customtkinter as ctk

    app = make_app()
    pages = app.page_container
    model = pages.get_page("model")
    # The unit lists' per-row checkboxes exist only once there are rows to draw.
    model._units_tab.set_units([{"analysis_id": 1, "serial": "S-1"}])
    model._ft_units_tab.set_units([{"serial": "S-1", "result": "PASS"}])
    boxes = [w for w in _walk(app) if isinstance(w, ctk.CTkCheckBox)]

    def inside(container):
        return [b for b in boxes if str(b).startswith(str(container) + ".")]
    # Every place gui/v6 builds a checkbox is reached, or the loop below proves nothing about it.
    assert inside(pages.get_page("process")), "the Process page's checkbox was never reached"
    assert len(inside(pages.get_page("settings"))) >= 2, "the database clean-up checkboxes were never reached"
    assert inside(model._units_tab) and inside(model._ft_units_tab), "a unit list's row checkbox was never reached"

    bad = []
    for b in boxes:
        mark = _hex(b, b.cget("checkmark_color"))
        for state, fill in (("checked", b.cget("fg_color")), ("checked, hovered", b.cget("hover_color"))):
            ratio = contrast(mark, _hex(b, fill))
            if ratio < 3.0:
                bad.append(f"{b} {state}: checkmark {mark} on {_hex(b, fill)} = {ratio:.2f}:1")
    assert not bad, "\n".join(bad)


def test_no_v6_segmented_control_or_checkbox_is_built_without_its_readable_colours():
    """The two app-level tests above see only what the app builds at start-up (plus the unit
    rows they fill in). A control added later, somewhere they never open, would slip past them --
    so every CONSTRUCTION SITE in gui/v6 is checked in the source as well."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src/laser_trim_analyzer/gui/v6"
    wants = {"CTkSegmentedButton": ("selected_color", "SEGMENT_SELECTED"),
             "CTkTabview": ("segmented_button_selected_color", "SEGMENT_SELECTED"),
             "CTkCheckBox": ("checkmark_color", "TEXT_INVERSE")}
    bad, seen = [], {name: 0 for name in wants}
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", None))
            if name == "__init__" and path.name == "tab_view.py":
                name = "CTkTabview"                         # ThemedTabView's super().__init__(...)
            if name not in wants:
                continue
            seen[name] += 1
            keyword, token = wants[name]
            given = {k.arg: k.value for k in node.keywords}
            value = given.get(keyword)
            if not (isinstance(value, ast.Attribute) and value.attr == token):
                bad.append(f"{path.name}:{node.lineno} {name} without {keyword}=<theme>.{token}")
    # Floors, so a scan that silently matched nothing cannot pass: 3 segmented buttons, the tab
    # view, and 5 checkbox sites existed when this was written.
    assert seen["CTkSegmentedButton"] >= 3 and seen["CTkCheckBox"] >= 5 and seen["CTkTabview"] >= 1, seen
    assert not bad, bad
