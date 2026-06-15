"""Spec 3a — V6 shell + sidebar + theme + PageBase.
Foundations: docs/superpowers/plans/2026-06-01-spec3-rewrite-foundations.md (§2).
Spec: docs/superpowers/specs/2026-05-30-spec3-ui-shell-design.md (Sub-spec 3a).
Shared fixtures (tk_root, make_app) live in tests/conftest.py.
"""

# ---- Task 1: ThemeManager -------------------------------------------------

def test_theme_exposes_color_tokens():
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    t = ThemeManager()
    assert (t.BG, t.SURFACE, t.CARD, t.ELEVATED) == ("#1a1f2e", "#1e2435", "#263244", "#2f3b50")
    assert (t.SIDEBAR_BG, t.SIDEBAR_ACTIVE, t.SIDEBAR_STRIPE) == ("#1a1f2e", "#263244", "#3b82f6")
    assert (t.ACCENT, t.ACCENT_HOVER, t.ACCENT_PRESSED) == ("#3b82f6", "#60a5fa", "#2563eb")
    assert (t.TEXT_PRIMARY, t.TEXT_SECONDARY, t.TEXT_DISABLED, t.TEXT_INVERSE) == \
        ("#e8eef5", "#9ca8bd", "#5a6478", "#1a1f2e")
    assert (t.DIVIDER, t.BORDER) == ("#2a3142", "#3a4456")


def test_theme_exposes_tier_color_tokens():
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    t = ThemeManager()
    assert t.TIER_STABLE == "#1e2435"
    assert (t.TIER_WARNING_BG, t.TIER_WARNING) == ("#3d2f1a", "#f59e0b")
    assert (t.TIER_DRIFT_BG, t.TIER_DRIFT) == ("#3d2418", "#f97316")
    assert (t.TIER_OOC_BG, t.TIER_OOC) == ("#3d1818", "#ef4444")


def test_theme_spacing_and_radii():
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    t = ThemeManager()
    assert (t.SPACE_XS, t.SPACE_SM, t.SPACE_MD, t.SPACE_LG, t.SPACE_XL, t.SPACE_2XL) == \
        (4, 8, 12, 16, 24, 32)
    assert (t.RADIUS_SM, t.RADIUS_MD, t.RADIUS_LG) == (4, 6, 8)
    assert (t.SIZE_CAPTION, t.SIZE_BODY, t.SIZE_HEADING, t.SIZE_TITLE, t.SIZE_DISPLAY) == \
        (11, 13, 16, 20, 28)
    assert t.FONT_FAMILY[0] == "Inter" and "Segoe UI" in t.FONT_FAMILY


def test_theme_tier_color_pairs():
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.ml.drift_types import DriftTier
    t = ThemeManager()
    assert t.tier_color(DriftTier.STABLE) == ("#1e2435", "#e8eef5")
    assert t.tier_color(DriftTier.WARNING) == ("#3d2f1a", "#f59e0b")
    assert t.tier_color(DriftTier.DRIFT) == ("#3d2418", "#f97316")
    assert t.tier_color(DriftTier.OUT_OF_CONTROL) == ("#3d1818", "#ef4444")


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


# ---- Task 2: Sidebar ------------------------------------------------------

def test_sidebar_items_in_order():
    from laser_trim_analyzer.gui.v6.sidebar import Sidebar
    assert Sidebar.ITEMS == [("dashboard", "Dashboard"), ("triage", "Triage"),
                             ("process", "Process"), ("model", "Model"), ("settings", "Settings")]


def test_sidebar_emits_selection(tk_root):
    from laser_trim_analyzer.gui.v6.sidebar import Sidebar
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    got = []
    sb = Sidebar(tk_root, on_select=got.append, theme=ThemeManager())
    sb._row_frames["model"]._on_click()
    assert got == ["model"]


def test_sidebar_set_active(tk_root):
    from laser_trim_analyzer.gui.v6.sidebar import Sidebar
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    sb = Sidebar(tk_root, on_select=lambda _: None, theme=ThemeManager())
    sb.set_active("triage"); assert sb._active_name == "triage"
    sb.set_active("settings"); assert sb._active_name == "settings"
    sb.set_active("bogus"); assert sb._active_name == "settings"  # unknown no-op


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

def test_v6app_starts_on_dashboard(make_app):
    app = make_app()
    assert app.page_container.current_page == "dashboard"
    assert app.sidebar._active_name == "dashboard"


def test_v6app_show_page(make_app):
    app = make_app()
    app.show_page("settings")
    assert app.page_container.current_page == "settings"
    assert app.sidebar._active_name == "settings"


def test_v6app_show_unknown_no_op(make_app):
    app = make_app()
    before = app.page_container.current_page
    app.show_page("nope")
    assert app.page_container.current_page == before


def test_v6app_has_all_pages(make_app):
    app = make_app()
    assert set(app.page_container._pages) == {"dashboard", "triage", "process", "model", "settings"}


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
