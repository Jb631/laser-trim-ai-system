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
    assert Sidebar.ITEMS == [("triage", "Triage"), ("process", "Process"),
                             ("model", "Model"), ("settings", "Settings")]


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
