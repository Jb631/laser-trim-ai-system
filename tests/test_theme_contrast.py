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


# CHECK too since the final review (2026-09-24): the coral is the text of a destructive link
# ("Clear selected", Settings -> Database), on the card like every other link.
@pytest.mark.parametrize("text", ("TEXT_SECONDARY", "TEXT_DISABLED", "ACCENT", "CHECK"))
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


# ---- CTkOptionMenu / CTkComboBox / CTkSwitch (Facelift step 2, Task 1) ---------------------------
# Step 1's review left these: 8 dropdowns whose arrow (ctk_optionmenu.py / ctk_combobox.py both
# fill their "dropdown_arrow" canvas item with text_color) sits on a button_color panel of
# ACCENT -- TEXT_PRIMARY on ACCENT is the same 1.66:1 the segmented buttons had -- and the Final
# Test overlay switch's knob (button_color, drawn as "slider_parts") sat on its ON track
# (progress_color=ACCENT) at 1.27:1. Same method as the two tests above: build the real widgets,
# read the colours they were actually configured with.

def _dropdown_menus(*roots):
    """De-duplicated: the two chart modals are built with master=app, and tkinter counts a
    Toplevel as one of its master's winfo_children() -- so walking `app` already reaches them,
    and walking them again by name would otherwise double every entry."""
    import customtkinter as ctk
    out, seen = [], set()
    for r in roots:
        for w in _walk(r):
            if isinstance(w, (ctk.CTkOptionMenu, ctk.CTkComboBox)) and id(w) not in seen:
                seen.add(id(w))
                out.append(w)
    return out


def test_every_dropdown_arrow_the_app_draws_is_readable(make_app):
    """The two unit-chart modals are Toplevels built on demand (opened from a unit row), not at
    app start-up, so -- same reasoning as the checkbox test filling unit rows above -- build them
    directly to reach their track-selector dropdowns too."""
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import FtUnitChartModal, UnitChartModal

    app = make_app()
    pages = app.page_container
    for name in ("model", "dashboard", "settings"):
        pages.get_page(name)          # per_model_specs' combo box lives on settings

    unit_modal = UnitChartModal(app, app.theme, app.db, {
        "serial": "S1", "overall_status": "PASS", "file_date": "2026-01-01",
        "analysis_id": None, "model": "M1", "system": "B"})
    ft_modal = FtUnitChartModal(app, app.theme, app.db, {
        "serial": "S1", "result": "PASS", "file_date": None,
        "id": None, "model": "M1", "system": "B"})
    try:
        menus = _dropdown_menus(app, unit_modal, ft_modal)
        named = {"the Model page's model selector": pages.get_page("model")._model_selector,
                 "the Model page's window menu": pages.get_page("model")._window_menu,
                 "the Model page's lot menu": pages.get_page("model")._lot_menu,
                 "the Dashboard's window menu": pages.get_page("dashboard")._window_menu,
                 "the Dashboard's trend-period menu": pages.get_page("dashboard")._trend_period_menu,
                 "the unit-chart modal's track menu": unit_modal._track_menu,
                 "the FT-chart modal's track menu": ft_modal._track_menu}
        for name, w in named.items():
            assert w in menus, f"the walk never reached {name}"
        # per_model_specs' own combo box, reached only through the walk (it has no public name):
        assert len(menus) >= len(named) + 1, "per_model_specs' model combo box was never reached"

        bad = []
        for w in menus:
            arrow = _hex(w, w.cget("text_color"))
            for state, fill in (("", w.cget("button_color")), (", hovered", w.cget("button_hover_color"))):
                ratio = contrast(arrow, _hex(w, fill))
                if ratio < 3.0:
                    bad.append(f"{w}{state}: arrow {arrow} on {_hex(w, fill)} = {ratio:.2f}:1")
        assert not bad, "\n".join(bad)
    finally:
        unit_modal.destroy()
        ft_modal.destroy()


def test_the_final_test_overlay_switch_knob_is_readable(make_app):
    """CTkSwitch's knob (button_color / button_hover_color, "slider_parts") can sit on either the
    ON track (progress_color, "progress_parts") or the OFF track (fg_color, "inner_parts") --
    ctk_switch.py's own _draw() colours both regardless of state. Must clear 3:1 against both,
    in both the resting and hovered knob colour."""
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import UnitChartModal

    app = make_app()
    modal = UnitChartModal(app, app.theme, app.db, {
        "serial": "S1", "overall_status": "PASS", "file_date": "2026-01-01",
        "analysis_id": None, "model": "M1", "system": "B"})
    try:
        sw = modal._ft_toggle
        tracks = {"on-track": _hex(sw, sw.cget("progress_color")),
                  "off-track": _hex(sw, sw.cget("fg_color"))}
        bad = []
        for state, knob in (("", sw.cget("button_color")), (", hovered", sw.cget("button_hover_color"))):
            k = _hex(sw, knob)
            for track_name, track in tracks.items():
                ratio = contrast(k, track)
                if ratio < 3.0:
                    bad.append(f"knob{state} {k} on the {track_name} {track} = {ratio:.2f}:1")
        assert not bad, "\n".join(bad)
    finally:
        modal.destroy()


def test_no_v6_dropdown_or_switch_is_built_with_the_defective_colour():
    """Static backstop, same idea as the segmented-button/checkbox one below.

    CTkOptionMenu/CTkComboBox: flagged only when EXPLICITLY given the wrong colour, not when
    button_color is left unset -- CTkOptionMenu's own un-themed default text_color/button_color
    measures 7.47:1, so an unstyled site is still SAFE, just inconsistent with the rest of the
    app's teal. `widgets/history_tab.py`'s menu used to be exactly that -- the one dropdown left
    unstyled after step 1's review fixed the other seven -- until facelift step 2 Task 3 themed
    it too (controller ruling); the `themed` count below pins that all 7 CTkOptionMenu sites
    carry the token outright, not merely "not wrong".
    CTkSwitch: the one construction site must carry the token outright -- there is no already-safe
    default to fall back on here (ctk_switch.py's own default button_color measured 1.27:1 against
    progress_color=ACCENT).
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src/laser_trim_analyzer/gui/v6"
    permissive = {"CTkOptionMenu": ("button_color", "SEGMENT_SELECTED"),
                  "CTkComboBox": ("button_color", "SEGMENT_SELECTED")}
    required = {"CTkSwitch": ("button_color", "TEXT_PRIMARY")}
    bad = []
    seen = {name: 0 for name in (*permissive, *required)}
    themed = {name: 0 for name in (*permissive, *required)}       # has_token, not just "not wrong"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", None))
            if name not in seen:
                continue
            seen[name] += 1
            keyword, token = (permissive.get(name) or required.get(name))
            given = {k.arg: k.value for k in node.keywords}
            value = given.get(keyword)
            has_token = isinstance(value, ast.Attribute) and value.attr == token
            if has_token:
                themed[name] += 1
            if name in required and not has_token:
                bad.append(f"{path.name}:{node.lineno} {name} without {keyword}=<theme>.{token}")
            elif name in permissive and value is not None and not has_token:
                wrong = value.attr if isinstance(value, ast.Attribute) else ast.dump(value)
                bad.append(f"{path.name}:{node.lineno} {name} styled with {keyword}={wrong}, want {token}")
    # Floors: model_page (combo box + 2 option menus) + dashboard_page (2) + unit_chart_modal (2)
    # + history_tab (1, now themed too -- facelift step 2 Task 3) = 7 CTkOptionMenu; model_page
    # + per_model_specs = 2 CTkComboBox; unit_chart_modal = 1 CTkSwitch.
    assert seen["CTkOptionMenu"] >= 7 and seen["CTkComboBox"] >= 2 and seen["CTkSwitch"] >= 1, seen
    # Every CTkOptionMenu site the walk found is now EXPLICITLY themed -- no more "unstyled but
    # safe" holdout (history_tab.py was the last one; this fails again the day a new dropdown
    # is added unstyled, the same way the old permissive-only check let history_tab.py through).
    assert themed["CTkOptionMenu"] == seen["CTkOptionMenu"], (themed, seen)
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
