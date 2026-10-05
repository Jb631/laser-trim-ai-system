"""Spec 3a — V6 shell + top bar (the sidebar until 2026-10-02) + theme + PageBase.
Foundations: docs/superpowers/plans/2026-06-01-spec3-rewrite-foundations.md (§2).
Spec: docs/superpowers/specs/2026-05-30-spec3-ui-shell-design.md (Sub-spec 3a).
Shared fixtures (tk_root, make_app) live in tests/conftest.py.
"""
import pytest

# ---- Task 1: ThemeManager -------------------------------------------------

def test_theme_exposes_color_tokens():
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    t = ThemeManager()
    # Graphite (James, 2026-10-02: "i like dark mode"; spec 2026-10-02-graphite-redesign).
    assert (t.BG, t.SURFACE, t.CARD, t.ELEVATED) == ("#0c0c0e", "#111114", "#151518", "#1c1c20")
    # The top bar reads the old sidebar tokens (theme.py keeps the names).
    assert (t.SIDEBAR_BG, t.SIDEBAR_ACTIVE, t.SIDEBAR_STRIPE) == ("#0c0c0e", "#151518", "#3b82f6")
    assert (t.ACCENT, t.ACCENT_HOVER, t.ACCENT_PRESSED) == ("#3b82f6", "#60a5fa", "#2563eb")
    assert (t.TEXT_PRIMARY, t.TEXT_SECONDARY, t.TEXT_DISABLED, t.TEXT_INVERSE) == \
        ("#ededed", "#8b8b93", "#85858e", "#0c0c0e")
    assert (t.DIVIDER, t.BORDER) == ("#1d1d21", "#26262b")
    assert (t.CHART_HIGHLIGHT, t.CHART_HISTORY) == ("#60a5fa", "#2a3a58")


def test_theme_exposes_tier_color_tokens():
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    t = ThemeManager()
    # TIER_STABLE is the card; amber, orange and red each on its own dark tint (Graphite).
    assert t.TIER_STABLE == "#151518"
    assert (t.TIER_WARNING_BG, t.TIER_WARNING) == ("#2a2210", "#fbbf24")
    assert (t.TIER_DRIFT_BG, t.TIER_DRIFT) == ("#2a1a0e", "#fb923c")
    assert (t.TIER_OOC_BG, t.TIER_OOC) == ("#2a1414", "#f87171")


def test_theme_spacing_and_radii():
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    t = ThemeManager()
    assert (t.SPACE_XS, t.SPACE_SM, t.SPACE_MD, t.SPACE_LG, t.SPACE_XL, t.SPACE_2XL) == \
        (4, 8, 12, 16, 24, 32)
    assert (t.RADIUS_SM, t.RADIUS_MD, t.RADIUS_LG) == (4, 6, 8)
    # Type scale moved one step up (spec 2026-09-23, section 1).
    assert (t.SIZE_CAPTION, t.SIZE_BODY, t.SIZE_HEADING, t.SIZE_TITLE, t.SIZE_DISPLAY) == \
        (12, 14, 17, 22, 30)
    assert t.SIZE_READOUT == 20
    # Chart text has its own scale, matplotlib points (spec 2026-09-23, section 1).
    assert (t.CHART_FONT_SMALL, t.CHART_FONT, t.CHART_FONT_LARGE) == (8.0, 9.0, 10.0)
    assert t.FONT_FAMILY[0] == "IBM Plex Sans" and "Segoe UI" in t.FONT_FAMILY


def test_theme_tier_color_pairs():
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.ml.drift_types import DriftTier
    t = ThemeManager()
    assert t.tier_color(DriftTier.STABLE) == ("#151518", "#ededed")
    assert t.tier_color(DriftTier.WARNING) == ("#2a2210", "#fbbf24")
    assert t.tier_color(DriftTier.DRIFT) == ("#2a1a0e", "#fb923c")
    assert t.tier_color(DriftTier.OUT_OF_CONTROL) == ("#2a1414", "#f87171")


def test_theme_tier_dot_color_stable_is_visible():
    """FIX I4: STABLE dot must NOT equal SURFACE (it'd be invisible on a SURFACE row)."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.ml.drift_types import DriftTier
    t = ThemeManager()
    assert t.tier_dot_color(DriftTier.STABLE) == t.TEXT_DISABLED
    assert t.tier_dot_color(DriftTier.STABLE) != t.SURFACE
    assert t.tier_dot_color(DriftTier.WARNING) == t.TIER_WARNING


def test_theme_font_returns_ctkfont(tk_root):
    """theme.font() resolves an available family (real fallback chain), returns CTkFont."""
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    t = ThemeManager()
    f = t.font(t.SIZE_BODY, "bold")
    assert isinstance(f, ctk.CTkFont)
    assert t.resolved_family in t.FONT_FAMILY  # picked one of the declared families
    # "bold" maps onto the resolved Medium family (weight "normal") when one is available,
    # else real bold weight on the regular family -- branch on the state actually resolved
    # on THIS machine, never assume one (no machine running the suite has Plex installed,
    # so today this takes the else branch; Task 4 bundles the fonts).
    if t.resolved_medium:
        assert f.cget("family") == t.resolved_medium and f.cget("weight") == "normal"
    else:
        assert f.cget("weight") == "bold"


def test_font_and_mono_map_bold_onto_the_medium_family_when_available(tk_root, monkeypatch):
    """The bold->Medium branch in font()/mono() is dead code on any machine without IBM
    Plex installed (every machine that runs this suite today). Fake the ONE thing that
    differs -- what fonts Tk reports as installed -- so ThemeManager's real
    _available_families()/_pick() resolution runs and actually picks the Medium families,
    instead of poking resolved_* directly (which would test nothing but the assignment).

    Patching tkinter.font.families (a plain module function) rather than the ThemeManager
    staticmethod it feeds: monkeypatch's own restore does getattr-then-setattr, which would
    silently strip the `@staticmethod` wrapper off a patched-and-restored class attribute
    and break every ThemeManager() built afterward for the rest of the test session --
    verified by reproducing it. A module-level function has no such descriptor trap.
    """
    import tkinter.font as tkfont
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    monkeypatch.setattr(tkfont, "families", lambda: [
        "IBM Plex Sans", "IBM Plex Sans Medium", "IBM Plex Mono", "IBM Plex Mono Medium"])
    t = ThemeManager()
    assert t.resolved_medium == "IBM Plex Sans Medium"
    assert t.resolved_mono_medium == "IBM Plex Mono Medium"

    f_bold = t.font(t.SIZE_BODY, "bold")
    assert (f_bold.cget("family"), f_bold.cget("weight")) == ("IBM Plex Sans Medium", "normal")

    f_normal = t.font(t.SIZE_BODY)
    assert (f_normal.cget("family"), f_normal.cget("weight")) == ("IBM Plex Sans", "normal")

    m_normal = t.mono(t.SIZE_BODY)
    assert m_normal.cget("family") == "IBM Plex Mono"

    m_bold = t.mono(t.SIZE_BODY, "bold")
    assert (m_bold.cget("family"), m_bold.cget("weight")) == ("IBM Plex Mono Medium", "normal")


def test_font_and_mono_bold_uses_bold_weight_when_no_medium_family_exists(tk_root, monkeypatch):
    """Same real resolution path, the other branch: no Medium family available (today's
    actual state on every dev/CI machine) -- "bold" must fall back to real bold weight on
    the regular family, for both font() and mono()."""
    import tkinter.font as tkfont
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    monkeypatch.setattr(tkfont, "families", lambda: ["IBM Plex Sans", "IBM Plex Mono"])
    t = ThemeManager()
    assert t.resolved_medium is None
    assert t.resolved_mono_medium is None

    f_bold = t.font(t.SIZE_BODY, "bold")
    assert (f_bold.cget("family"), f_bold.cget("weight")) == ("IBM Plex Sans", "bold")

    m_bold = t.mono(t.SIZE_BODY, "bold")
    assert (m_bold.cget("family"), m_bold.cget("weight")) == ("IBM Plex Mono", "bold")


# ---- Task 2: the top bar (it replaced the sidebar -- Graphite redesign, 2026-10-02) -------------
# James: "there is so much going on its hard to see what is what". Three destinations and one blue
# button: Overview, Models, Settings, and "Process new files". The KEYS never change -- FOCUS rows,
# set_model_route and every deep link navigate by key, and a label change must not break them.

def _walk(widget):
    import tkinter
    yield widget
    for child in tkinter.Misc.winfo_children(widget):
        yield from _walk(child)


def _bar(tk_root, selected=None, pressed=None):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.topbar import TopBar
    return TopBar(tk_root, on_select=(selected.append if selected is not None else lambda _k: None),
                  on_process=(lambda: pressed.append("pressed")) if pressed is not None else (lambda: None),
                  theme=ThemeManager())


def test_the_top_bar_has_three_destinations_by_key():
    from laser_trim_analyzer.gui.v6.topbar import TopBar
    assert TopBar.ITEMS == [("home", "Overview"), ("model", "Models"), ("settings", "Settings")]
    # The pages with no item stay one click away -- Process by the blue button, Findings and
    # Dashboard by the two links at the foot of the Overview.
    assert TopBar.OFF_BAR == ("process", "findings", "dashboard")


def test_the_sidebar_is_gone():
    import importlib.util
    assert importlib.util.find_spec("laser_trim_analyzer.gui.v6.sidebar") is None


def test_the_bar_says_the_apps_name_and_its_three_destinations(tk_root):
    import customtkinter as ctk
    bar = _bar(tk_root)
    texts = [w.cget("text") for w in _walk(bar) if isinstance(w, (ctk.CTkLabel, ctk.CTkButton))]
    assert texts[0] == "Laser Trim Analyzer"
    assert [t for t in texts if t in ("Overview", "Models", "Settings")] == ["Overview", "Models", "Settings"]


def test_a_destination_emits_its_key_not_its_label(tk_root):
    got = []
    bar = _bar(tk_root, selected=got)
    bar._items["model"]._on_click()
    bar._items["home"]._on_click()
    assert got == ["model", "home"]


def test_the_active_destination_is_bright_with_an_accent_underline(tk_root):
    bar = _bar(tk_root)
    t = bar.theme
    bar.set_active("settings")
    assert bar._active_name == "settings"
    for key, item in bar._items.items():
        on = key == "settings"
        assert item._label.cget("text_color") == (t.TEXT_PRIMARY if on else t.TEXT_SECONDARY), key
        assert item._underline.cget("fg_color") == (t.SIDEBAR_STRIPE if on else t.SIDEBAR_BG), key


def test_a_page_with_no_item_lights_none(tk_root):
    bar = _bar(tk_root)
    t = bar.theme
    bar.set_active("home")
    for key in ("process", "findings", "dashboard", "bogus"):
        assert not bar.lights(key), key             # ...so the page names itself (PageBase)
        bar.set_active(key)
        assert bar._active_name is None, key
        assert all(i._label.cget("text_color") == t.TEXT_SECONDARY for i in bar._items.values())
        assert all(i._underline.cget("fg_color") == t.SIDEBAR_BG for i in bar._items.values())
    bar.set_active("model")
    assert bar._active_name == "model"
    assert all(bar.lights(key) for key, _label in bar.ITEMS)


def test_the_bar_holds_one_blue_button_and_it_processes_new_files(tk_root):
    import customtkinter as ctk
    pressed = []
    bar = _bar(tk_root, pressed=pressed)
    t = bar.theme
    blue = [w for w in _walk(bar) if isinstance(w, ctk.CTkButton) and w.cget("fg_color") == t.ACCENT]
    assert blue == [bar._process_button]
    b = bar._process_button
    assert b.cget("text") == "Process new files"
    assert (b.cget("hover_color"), b.cget("text_color")) == (t.ACCENT_HOVER, t.TEXT_INVERSE)
    b.invoke()
    assert pressed == ["pressed"]


# ---- option B (2026-10-04): the bar finished --------------------------------------------------
# The finish list's first item: "top-bar items spread with gaps, no icons -> grouped, with icons".
# Each destination was a 200-px frame (CustomTkinter's default width, held by its underline), so the
# three sat 216 px apart across the bar. The app's name is in Marcellus (James: "6").

class _Mapped:
    """Lay a root out for real: off-screen, transparent, mapped. A withdrawn root leaves every child
    at Tk's 1x1 placeholder (scripts/render_pages.py, point 1), so x-positions mean nothing."""

    def __init__(self, root, size="1280x120"):
        self.root, self.size = root, size

    def __enter__(self):
        try:
            self.root.attributes("-alpha", 0.0)
        except Exception:
            pass
        self.root.geometry(f"{self.size}+20000+20000")
        self.root.deiconify()
        self.root.update_idletasks()
        self.root.update()
        return self.root

    def __exit__(self, *exc):
        self.root.withdraw()
        return False


def _tint(image) -> str:
    """The colour an icon was tinted to: the RGB of its solid pixels."""
    img = image.cget("dark_image")
    solid = {px[:3] for px in img.getdata() if px[3] == 255}
    assert len(solid) == 1, solid
    return "#%02x%02x%02x" % solid.pop()


def test_the_apps_name_is_in_the_title_face(tk_root, monkeypatch):
    """What Tk can see is faked, as the bold->Medium test above fakes it: on the Mac, Tk's family
    list is read once per process, so whether Marcellus shows up in it depends on which test asked
    first (font_loader registers it for the process; the app asks after loading, always)."""
    import tkinter.font as tkfont
    monkeypatch.setattr(tkfont, "families", lambda: ["IBM Plex Sans", "Marcellus"])
    bar = _bar(tk_root)
    t = bar.theme
    font = bar._title.cget("font")
    assert bar._title.cget("text") == "Laser Trim Analyzer"
    assert t.resolved_title == "Marcellus"
    assert font is t.title(t.SIZE_HEADING) and font.cget("family") == "Marcellus"
    # Where Tk cannot see it: the Sans's bold, never a face that is not there.
    monkeypatch.setattr(tkfont, "families", lambda: ["IBM Plex Sans"])
    bar = _bar(tk_root)
    assert bar._title.cget("font") is bar.theme.font(bar.theme.SIZE_HEADING, "bold")


def test_the_three_destinations_sit_together_beside_the_name(tk_root):
    from laser_trim_analyzer.gui.v6.topbar import TopBar
    bar = _bar(tk_root)
    bar.pack(side="top", fill="x")
    t = bar.theme
    items = [bar._items[key] for key, _label in TopBar.ITEMS]
    with _Mapped(tk_root):
        # Each one as wide as its icon and its word -- not CustomTkinter's 200-px default.
        assert all(i.winfo_reqwidth() < 120 for i in items), [i.winfo_reqwidth() for i in items]
        right_of = lambda w: w.winfo_x() + w.winfo_width()          # noqa: E731
        gaps = [items[0].winfo_x() - right_of(bar._title)]
        gaps += [b.winfo_x() - right_of(a) for a, b in zip(items, items[1:])]
        assert all(0 <= g <= t.SPACE_XL for g in gaps), gaps
        # Grouped at the left; the blue button alone on the right.
        assert right_of(items[-1]) < bar.winfo_width() / 2
        assert bar._process_button.winfo_x() > bar.winfo_width() / 2


def test_each_destination_carries_its_icon_tinted_like_its_words(tk_root):
    from laser_trim_analyzer.gui.v6 import icons
    from laser_trim_analyzer.gui.v6.topbar import ICON_SIZE, TopBar
    assert TopBar.ICONS == {"home": "overview", "model": "models", "settings": "settings"}
    bar = _bar(tk_root)
    t = bar.theme
    for active in ("model", "home", "process"):          # "process": a page with no item
        bar.set_active(active)
        for key, item in bar._items.items():
            colour = t.TEXT_PRIMARY if key == active else t.TEXT_SECONDARY
            image = item._icon.cget("image")
            assert image is icons.icon(TopBar.ICONS[key], colour, ICON_SIZE), (active, key)
            assert _tint(image) == colour and item._label.cget("text_color") == colour
    bar.set_active("home")
    item = bar._items["settings"]
    item._hover(True)                                    # the icon brightens with the word
    assert _tint(item._icon.cget("image")) == t.TEXT_PRIMARY == item._label.cget("text_color")
    item._hover(False)
    assert _tint(item._icon.cget("image")) == t.TEXT_SECONDARY


def test_a_click_on_the_icon_goes_where_the_word_goes(tk_root):
    got = []
    bar = _bar(tk_root, selected=got)
    bar.pack(side="top", fill="x")
    item = bar._items["settings"]
    with _Mapped(tk_root):
        # A CTkLabel's own bind() lands on the Tk label inside it -- where a click lands too.
        item._icon._label.event_generate("<Button-1>")
        item._label._label.event_generate("<Button-1>")
        tk_root.update()
    assert got == ["settings", "settings"]


def test_an_icon_is_drawn_in_each_new_window_not_a_dead_one():
    """icons.icon shared one CTkImage per (name, colour, size) for the whole process, and a
    CTkImage keeps the Tk images it has drawn -- which die with their window. The second window to
    draw the bar raised 'image "pyimage1" doesn't exist'. One per window now, as theme.font() does.
    (One root at a time here, each destroyed before the next: never two live roots.)"""
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6 import icons
    for _ in range(2):
        root = ctk.CTk()
        root.withdraw()
        try:
            image = icons.icon("overview", "#3b82f6", 18)
            ctk.CTkLabel(root, text="", image=image)          # drawn in THIS window
            assert icons.icon("overview", "#3b82f6", 18) is image   # and shared within it
        finally:
            root.destroy()


def test_the_blue_button_carries_the_process_icon_in_its_own_text_colour(tk_root):
    from laser_trim_analyzer.gui.v6 import icons
    from laser_trim_analyzer.gui.v6.topbar import ICON_SIZE
    bar = _bar(tk_root)
    t = bar.theme
    b = bar._process_button
    assert b.cget("image") is icons.icon("process", t.TEXT_INVERSE, ICON_SIZE)
    assert _tint(b.cget("image")) == t.TEXT_INVERSE and b.cget("compound") == "left"
    assert (b.cget("fg_color"), b.cget("text_color")) == (t.ACCENT, t.TEXT_INVERSE)


# ---- Task 3: PageBase + PageContainer -------------------------------------

def test_page_base_requires_build_content(tk_root):
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    class _Incomplete(PageBase):
        page_title = "X"
    import pytest
    with pytest.raises(NotImplementedError):
        _Incomplete(tk_root, theme=ThemeManager())


def test_page_base_runs_build_content_and_stores_app(tk_root):
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    calls = []

    class _P(PageBase):
        page_title = "T"
        def build_content(self, parent): calls.append(self.app)

    sentinel = object()
    p = _P(tk_root, theme=ThemeManager(), app=sentinel)
    assert calls == [sentinel]
    assert p.app is sentinel


def test_page_base_header_actions_receives_parent(tk_root):
    """FIX C5/C6: header_actions(parent) builds widgets WITH that parent (no reparenting)."""
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    seen = {}

    class _P(PageBase):
        page_title = "T"
        def build_content(self, parent): pass
        def header_actions(self, parent):
            btn = ctk.CTkButton(parent, text="Go")
            btn.pack(side="right")
            seen["parent_is_actions_frame"] = btn.master is parent

    _P(tk_root, theme=ThemeManager())
    assert seen["parent_is_actions_frame"] is True


# ---- option B (2026-10-04): a page the bar names never repeats it; the rest name themselves -------
# "Overview" twice, "Models" over "8232-1" (finish list 2): the top bar names Overview, Models and
# Settings, and their row under it holds only their actions -- and is not there at all when they
# have none. But the bar lights nothing for Process, Findings and Company trends, which then were
# named nowhere (review of option B, the same day): each says its name at the left of its row, in
# the title face at the heading size (PageBase.show_name, called by V6App from TopBar.lights).

def _header_labels(page):
    """Every label in the page's own row that is not one of its actions."""
    import customtkinter as ctk
    actions = set(_walk(page._header.actions_frame))
    return [w for w in _walk(page._header) if isinstance(w, ctk.CTkLabel) and w not in actions]


def test_a_page_with_no_actions_has_no_band_under_the_bar(tk_root):
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    class _P(PageBase):
        page_title = "Somewhere"
        def build_content(self, parent): pass

    p = _P(tk_root, theme=ThemeManager())         # never told to name itself: a page on the bar
    assert p.page_title == "Somewhere"            # kept: tests, render_pages and the window use it
    assert p._header.winfo_manager() == ""
    assert p.pack_slaves() == [p._content]        # no title, no rule, no empty band
    p.set_caption("A headline")
    assert p.pack_slaves() == [p._caption, p._content]


def test_a_page_with_actions_keeps_them_right_aligned_with_no_title(tk_root):
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    class _P(PageBase):
        page_title = "Somewhere"
        def build_content(self, parent): pass
        def header_actions(self, parent):
            ctk.CTkButton(parent, text="Go").pack(side="right")

    p = _P(tk_root, theme=ThemeManager())
    slaves = p.pack_slaves()
    assert slaves[0] is p._header and slaves[-1] is p._content
    assert p._header.actions_frame.pack_info()["side"] == "right"
    assert _header_labels(p) == []
    p.set_caption("A headline")                   # under the actions, above the content
    slaves = p.pack_slaves()
    assert slaves.index(p._header) < slaves.index(p._caption) < slaves.index(p._content)


def _with_marcellus(monkeypatch):
    """What Tk can see, faked as test_the_apps_name_is_in_the_title_face fakes it -- BEFORE a theme
    is built, since a theme reads the list once."""
    import tkinter.font as tkfont
    monkeypatch.setattr(tkfont, "families", lambda: ["IBM Plex Sans", "Marcellus"])


def test_a_page_told_to_name_itself_says_it_at_the_left_in_the_title_face(tk_root, monkeypatch):
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    _with_marcellus(monkeypatch)
    t = ThemeManager()

    class _P(PageBase):
        page_title = "Somewhere"
        def build_content(self, parent): pass

    p = _P(tk_root, theme=t)
    p.set_caption("A headline")                   # a caption first: the name still goes on top
    p.show_name()
    p.show_name()                                 # twice: one name, one row, one rule
    (label,) = _header_labels(p)
    assert label is p._header.title_label and label.cget("text") == "Somewhere"
    assert label.cget("font") is t.title(t.SIZE_HEADING)
    assert label.cget("font").cget("family") == "Marcellus"
    assert label.cget("text_color") == t.TEXT_PRIMARY and label.pack_info()["side"] == "left"
    assert p.pack_slaves() == [p._header, p._header_rule, p._caption, p._content]


def test_a_page_with_a_name_and_actions_has_both_in_one_row(tk_root):
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    class _P(PageBase):
        page_title = "Somewhere"
        def build_content(self, parent): pass
        def header_actions(self, parent):
            ctk.CTkButton(parent, text="Go").pack(side="right")

    p = _P(tk_root, theme=ThemeManager())
    p.show_name()
    assert [w.cget("text") for w in _header_labels(p)] == ["Somewhere"]
    assert p._header.title_label.pack_info()["side"] == "left"
    assert p._header.actions_frame.pack_info()["side"] == "right"
    assert p.pack_slaves() == [p._header, p._header_rule, p._content]


def test_only_the_pages_the_bar_does_not_light_name_themselves(make_app, monkeypatch):
    from laser_trim_analyzer.gui.v6.topbar import TopBar
    _with_marcellus(monkeypatch)
    app = make_app()
    t = app.theme
    pages = app.page_container._pages
    named = {key: [w.cget("text") for w in _header_labels(page)] for key, page in pages.items()}
    assert named == {"dashboard": ["Company trends"], "findings": ["Findings"],
                     "process": ["Process"], "home": [], "model": [], "settings": []}
    on_bar = {key for key, _label in TopBar.ITEMS}
    for key, page in pages.items():
        assert app.topbar.lights(key) == (key in on_bar), key
        has_actions = bool(page._header.actions_frame.winfo_children())
        assert (page._header.winfo_manager() == "pack") == (has_actions or key not in on_bar), key
        if key not in on_bar:
            label = page._header.title_label
            assert label.cget("font") is t.title(t.SIZE_HEADING), key
            assert label.cget("font").cget("family") == "Marcellus", key
            assert label.cget("text_color") == t.TEXT_PRIMARY, key
            assert page.pack_slaves()[:2] == [page._header, page._header_rule], key
    # Company trends: its name on the left and its window on the right, in one row.
    trends = pages["dashboard"]
    assert trends._header.title_label.pack_info()["side"] == "left"
    assert trends._window_menu.master is trends._header.actions_frame
    # The Model page keeps its picker, window and run menus and its buttons -- and no name; the
    # Overview has neither, so no row.
    model = pages["model"]
    assert model._header.winfo_manager() == "pack" and model._header.title_label is None
    assert model._model_selector.master is model._header.actions_frame
    assert pages["home"]._header.winfo_manager() == ""


def test_page_base_lifecycle_hooks(tk_root):
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    ev = []

    class _P(PageBase):
        page_title = "T"
        def build_content(self, parent): pass
        def on_show(self): ev.append("show")
        def on_hide(self): ev.append("hide")

    p = _P(tk_root, theme=ThemeManager())
    p.on_show(); p.on_hide()
    assert ev == ["show", "hide"]


def _pump_for(root, done, seconds=3.0):
    import time
    end = time.monotonic() + seconds
    while time.monotonic() < end and not done():
        root.update()
        time.sleep(0.01)


def _logged(caplog, text):
    return [r for r in caplog.records
            if r.exc_info and isinstance(r.exc_info[1], RuntimeError) and text in str(r.exc_info[1])]


def test_a_ui_update_that_raises_is_logged_never_swallowed(tk_root, caplog):
    """F4 review (OOS 3): safe_after's guard swallowed a callback's exception with no trace at all
    -- a render crash there left the screen stale and the log empty. It is logged now, with its
    traceback, and the next update still runs. (Direct path: no dispatcher.)"""
    import logging
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    class _P(PageBase):
        page_title = "T"
        def build_content(self, parent): pass

    page = _P(tk_root, theme=ThemeManager())
    ran = []

    def boom():
        ran.append("boom")
        raise RuntimeError("invented render crash")
    with caplog.at_level(logging.ERROR):
        page.safe_after(boom)
        page.safe_after(lambda: ran.append("next"))
        _pump_for(tk_root, lambda: len(ran) == 2)
    assert ran == ["boom", "next"]
    (record,) = _logged(caplog, "invented render crash")
    assert record.name == "laser_trim_analyzer.gui.v6.page_base"


def test_a_ui_update_that_raises_through_the_dispatcher_is_logged_too(make_app, caplog):
    import logging
    app = make_app()
    page = app.page_container.get_page("findings")
    ran = []

    def boom():
        ran.append("boom")
        raise RuntimeError("invented dispatched crash")
    with caplog.at_level(logging.ERROR):
        page.safe_after(boom)
        _pump_for(app, lambda: bool(ran))
        _pump_for(app, lambda: False, seconds=0.2)
    assert ran == ["boom"]
    assert len(_logged(caplog, "invented dispatched crash")) == 1


# ---- F5 review (Important 1): a callback that keeps failing is logged once, then counted -------
# The 4 Hz ProgressTicker posts its paint through safe_after for a whole ingest run (hours), so a
# paint that keeps failing wrote a traceback four times a second.

def _page(tk_root, monkeypatch, clock=None):
    from laser_trim_analyzer.gui.v6 import page_base
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    monkeypatch.setattr(page_base, "_failing", {}, raising=False)   # no memory from an earlier test
    if clock is not None:
        monkeypatch.setattr(page_base, "_clock", clock, raising=False)

    class _P(page_base.PageBase):
        page_title = "T"
        def build_content(self, parent): pass
    return _P(tk_root, theme=ThemeManager())


def _page_records(caplog):
    return [r for r in caplog.records if r.name == "laser_trim_analyzer.gui.v6.page_base"]


def test_a_callback_failing_on_every_tick_is_at_most_two_log_lines(tk_root, caplog, monkeypatch):
    import logging
    page = _page(tk_root, monkeypatch)
    ran = []

    def tick():                          # a FRESH lambda every tick, the way the ticker posts
        return lambda: (ran.append(1), 1 / 0)
    with caplog.at_level(logging.ERROR):
        for _ in range(100):
            page.safe_after(tick())
        _pump_for(tk_root, lambda: len(ran) == 100)
    records = _page_records(caplog)
    assert len(ran) == 100
    assert 1 <= len(records) <= 2, [r.getMessage() for r in records]
    assert records[0].exc_info and isinstance(records[0].exc_info[1], ZeroDivisionError)


def test_a_minute_later_one_line_says_how_many_more_failed(tk_root, caplog, monkeypatch):
    import logging
    now = [1000.0]
    page = _page(tk_root, monkeypatch, clock=lambda: now[0])
    ran = []

    def fail():
        ran.append(1)
        raise RuntimeError("invented paint crash")

    def run(n):
        start = len(ran)
        for _ in range(n):
            page.safe_after(fail)
        _pump_for(tk_root, lambda: len(ran) == start + n)
    with caplog.at_level(logging.ERROR):
        run(10)                          # the first, with its traceback; nine counted
        now[0] += 61
        run(1)                           # a minute on: ONE line, "10 more failures of ... since"
        run(5)                           # counted again, silently
    records = _page_records(caplog)
    assert len(records) == 2, [r.getMessage() for r in records]
    first, summary = records
    assert first.exc_info and "invented paint crash" in str(first.exc_info[1])
    assert "fail" in first.getMessage()                     # names the callback
    assert not summary.exc_info
    assert "10 more failures of" in summary.getMessage() and "since" in summary.getMessage()
    assert "RuntimeError: invented paint crash" in summary.getMessage()


def test_two_different_failing_callbacks_each_get_their_first_traceback(tk_root, caplog, monkeypatch):
    import logging
    page = _page(tk_root, monkeypatch)
    ran = []

    def first():
        ran.append("first")
        raise RuntimeError("invented first crash")

    def second():
        ran.append("second")
        raise ValueError("invented second crash")
    with caplog.at_level(logging.ERROR):
        for fn in (first, second, first, second):
            page.safe_after(fn)
        _pump_for(tk_root, lambda: len(ran) == 4)
    records = _page_records(caplog)
    assert [type(r.exc_info[1]).__name__ for r in records if r.exc_info] == ["RuntimeError", "ValueError"]
    assert len(records) == 2


def test_page_base_set_caption_shows_and_clears(tk_root):
    """set_caption packs a caption line under the title bar on text, and clears it on "".

    winfo_manager(), NOT winfo_ismapped(): under the withdrawn test root nothing is ever
    mapped, so an ismapped assertion would pass vacuously. And even
    in the real app "is it on screen" is the wrong question -- PageContainer switches pages
    with grid() + tkraise() (stacking order only), so a page it has switched away from stays
    winfo_ismapped() == 1, not 0. "Is the caption packed" is the actual question, and
    winfo_manager() answers it directly either way.
    """
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    class _P(PageBase):
        page_title = "T"
        def build_content(self, parent): pass

    p = _P(tk_root, theme=ThemeManager())
    assert p._caption.winfo_manager() == ""

    p.set_caption("Two models need a look this week")
    assert p._caption.cget("text") == "Two models need a look this week"
    assert p._caption.winfo_manager() == "pack"

    p.set_caption("")
    assert p._caption.cget("text") == ""
    assert p._caption.winfo_manager() == ""

    p.set_caption("Back again")
    assert p._caption.cget("text") == "Back again"
    assert p._caption.winfo_manager() == "pack"
    slaves = p.pack_slaves()
    assert slaves.count(p._caption) == 1                            # packed once, not duplicated
    assert slaves.index(p._caption) < slaves.index(p._content)      # before the content frame


@pytest.mark.parametrize("scale", (1.0, 1.5))
def test_page_base_caption_wraps_to_the_page_instead_of_overflowing(tk_root, scale):
    """Task 2 (facelift step 2, 2026-09-24): Investigate's caption can run to several
    clauses joined by " · " and genuinely overflows an unwrapped single line --
    render_pages.py --audit found it squeezed at both audited window sizes (a caption was
    never given a wraplength at all until this fix). blocks.wrap_to_width ties it to the
    PAGE's own width (self), the same helper and padding convention every other page-width
    label in gui/v6 now follows.

    <Configure> never fires under the withdrawn tk_root (tests/test_blocks.py's own
    finding), even though geometry propagation still runs -- map off-screen the same way
    render_pages.py does.
    """
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    class _P(PageBase):
        page_title = "T"
        def build_content(self, parent): pass

    import customtkinter as ctk
    # At 150% too (final review, 2026-09-24): this used to pin `wraplength == 500 - 32`, true
    # only at 100% -- the page width is real pixels, wraplength CustomTkinter's unscaled units,
    # and on the Windows laptop the widget scaling is the monitor's DPI factor. set_widget_scaling
    # on a live window pins it at its size for a second; _set_scaled_min_max is CTk's own release.
    ctk.set_widget_scaling(scale)
    tk_root._set_scaled_min_max()
    try:
        t = ThemeManager()
        p = _P(tk_root, theme=t)
        p.pack(fill="both", expand=True)
        tk_root.geometry("500x300")
        try:
            tk_root.attributes("-alpha", 0.0)
        except Exception:
            pass
        tk_root.geometry("+20000+20000")
        tk_root.deiconify()
        tk_root.update_idletasks()
        tk_root.update()
        try:
            p.set_caption("One clause · Two clause · Three clause · Four clause · Five clause · "
                          "Six clause · Seven clause · Eight clause · Nine clause · Ten clause")
            tk_root.update_idletasks()
            tk_root.update()
            assert p.winfo_width() == 500
            assert p._caption.cget("wraplength") == int(500 / scale) - t.SPACE_LG * 2
            # laid out inside the page, less the caption's own padx=SPACE_LG each side (scaled)
            assert p._caption._label.winfo_reqwidth() <= 500 - t.SPACE_LG * 2 * scale
        finally:
            tk_root.withdraw()
    finally:
        ctk.set_widget_scaling(1.0)
        tk_root._set_scaled_min_max()


def test_page_container_add_get_show(tk_root):
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.page_container import PageContainer
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    theme = ThemeManager(); ev = []

    class _P(PageBase):
        def build_content(self, parent): pass
        def on_show(self): ev.append(f"show:{self.page_title}")
        def on_hide(self): ev.append(f"hide:{self.page_title}")

    c = PageContainer(tk_root, theme=theme)
    a = _P(c, theme=theme, page_title="A"); b = _P(c, theme=theme, page_title="B")
    c.add_page("A", a); c.add_page("B", b)
    assert c.get_page("A") is a
    c.show("A"); c.show("B")
    assert ev == ["show:A", "hide:A", "show:B"]
    c.show("missing"); assert c.current_page == "B"  # unknown no-op


# ---- Task 4: V6App --------------------------------------------------------

def test_v6app_starts_on_home(make_app):
    app = make_app()
    assert app.page_container.current_page == "home"
    assert app.topbar._active_name == "home"


def test_v6app_show_page(make_app):
    app = make_app()
    app.show_page("settings")
    assert app.page_container.current_page == "settings"
    assert app.topbar._active_name == "settings"
    app.show_page("findings")                    # a page with no item: the bar lights none
    assert app.page_container.current_page == "findings"
    assert app.topbar._active_name is None


def test_v6app_show_unknown_no_op(make_app):
    app = make_app()
    before = app.page_container.current_page
    app.show_page("nope")
    assert app.page_container.current_page == before


def test_v6app_puts_the_bar_above_the_pages(make_app):
    app = make_app()
    assert not hasattr(app, "sidebar")
    assert int(app.topbar.grid_info()["row"]) == 0 and app.topbar.grid_info()["sticky"] == "ew"
    assert int(app.page_container.grid_info()["row"]) == 1
    assert int(app.grid_rowconfigure(1)["weight"]) == 1 and int(app.grid_rowconfigure(0)["weight"]) == 0
    assert int(app.grid_columnconfigure(0)["weight"]) == 1
    # ...and the status bar under them (option B, 2026-10-04): its own height, every page.
    foot = app.status_bar.grid_info()
    assert int(foot["row"]) == 2 and foot["sticky"] == "ew"
    assert int(app.grid_rowconfigure(2)["weight"]) == 0


def test_v6app_has_all_pages(make_app):
    """Findings added 2026-09-20 (process-findings engine). Triage retired 2026-10-02 (Graphite
    redesign): the Overview's cards and list are what it showed."""
    app = make_app()
    assert set(app.page_container._pages) == {"home", "dashboard", "process", "model",
                                              "settings", "findings"}


def test_the_page_titles_say_what_the_bar_says(make_app):
    app = make_app()
    pages = app.page_container
    assert pages.get_page("home").page_title == "Overview"
    assert pages.get_page("model").page_title == "Models"
    assert pages.get_page("settings").page_title == "Settings"


def test_every_key_on_the_bar_and_off_it_is_a_page(make_app):
    from laser_trim_analyzer.gui.v6.topbar import TopBar
    app = make_app()
    keys = [k for k, _ in TopBar.ITEMS] + list(TopBar.OFF_BAR)
    assert sorted(keys) == sorted(app.page_container._pages)


def test_v6app_auto_train_off_does_not_offer(make_app):
    """make_app passes auto_train_on_first_run=False → no first-run hook scheduled."""
    app = make_app()
    assert app._auto_train_on_first_run is False


# ---- Task 5: --v6 flag ----------------------------------------------------

def test_main_v6_flag_uses_v6app(monkeypatch, tmp_path):
    import sys
    from unittest.mock import MagicMock
    import laser_trim_analyzer.app as v5_mod
    import laser_trim_analyzer.gui.v6.app as v6_mod
    fake_v5, fake_v6 = MagicMock(), MagicMock()
    monkeypatch.setattr(v5_mod, "LaserTrimApp", fake_v5)
    monkeypatch.setattr(v6_mod, "V6App", fake_v6)
    monkeypatch.setattr(sys, "argv", ["laser_trim_analyzer", "--v6"])
    from laser_trim_analyzer.config import Config
    cfg = Config(); cfg.database.path = tmp_path / "t.db"
    import laser_trim_analyzer.config as cmod
    monkeypatch.setattr(cmod, "get_config", lambda: cfg)
    from laser_trim_analyzer.__main__ import main
    main()
    fake_v6.assert_called_once(); fake_v5.assert_not_called()


def test_main_default_uses_v5(monkeypatch, tmp_path):
    import sys
    from unittest.mock import MagicMock
    import laser_trim_analyzer.app as v5_mod
    import laser_trim_analyzer.gui.v6.app as v6_mod
    fake_v5, fake_v6 = MagicMock(), MagicMock()
    monkeypatch.setattr(v5_mod, "LaserTrimApp", fake_v5)
    monkeypatch.setattr(v6_mod, "V6App", fake_v6)
    monkeypatch.setattr(sys, "argv", ["laser_trim_analyzer"])
    from laser_trim_analyzer.config import Config
    cfg = Config(); cfg.database.path = tmp_path / "t.db"
    import laser_trim_analyzer.config as cmod
    monkeypatch.setattr(cmod, "get_config", lambda: cfg)
    from laser_trim_analyzer.__main__ import main
    main()
    fake_v5.assert_called_once(); fake_v6.assert_not_called()
