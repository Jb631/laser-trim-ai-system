# Spec 3a — V6 App Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable V6 app shell at `python -m laser_trim_analyzer --v6` with industrial-dark theme, left sidebar nav (4 items + active stripe), `PageBase` chrome, and 4 placeholder pages. V5 shell (`python -m laser_trim_analyzer`, no flag) continues to work unchanged.

**Architecture:** New `src/laser_trim_analyzer/gui/v6/` subpackage holds the new shell. `V6App(ctk.CTk)` is the root window. `ThemeManager` exposes color/typography/spacing tokens. `Sidebar` + `PageContainer` + `PageBase` form the chrome. Each of the four target pages (Triage / Process / Model / Settings) is a placeholder subclass that just renders "coming in Spec 3X" — real content lands in 3b–3e.

**Tech Stack:** Python 3.x, customtkinter, pytest.

**Target branch:** `V6` only. Verify with `git branch --show-current` before Task 1.

**Spec reference:** `docs/superpowers/specs/2026-05-30-spec3-ui-shell-design.md` (Sub-spec 3a section).

---

## File Structure

**Files created:**
- `src/laser_trim_analyzer/gui/v6/__init__.py` — empty package marker
- `src/laser_trim_analyzer/gui/v6/theme.py` — `ThemeManager` class + all constants
- `src/laser_trim_analyzer/gui/v6/sidebar.py` — `Sidebar` + `_SidebarRow`
- `src/laser_trim_analyzer/gui/v6/page_container.py` — `PageContainer`
- `src/laser_trim_analyzer/gui/v6/page_base.py` — `PageBase` + `_PageHeader`
- `src/laser_trim_analyzer/gui/v6/app.py` — `V6App` + 4 placeholder page classes
- `tests/test_spec3a_shell.py` — unit tests (no Tk mainloop)

**Files modified:**
- `src/laser_trim_analyzer/__main__.py` — add `--v6` CLI flag handling

---

## Task 1: ThemeManager + theme constants

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/__init__.py` (empty file)
- Create: `src/laser_trim_analyzer/gui/v6/theme.py`
- Test: `tests/test_spec3a_shell.py` (CREATE)

- [ ] **Step 1: Create empty package marker**

Create `src/laser_trim_analyzer/gui/v6/__init__.py` with empty contents (just one blank line is fine).

- [ ] **Step 2: Create the test file with ThemeManager tests**

Create `tests/test_spec3a_shell.py` with this exact content:

```python
"""Spec 3a — V6 app shell + sidebar + theme + PageBase.

Each test maps to one element of the spec at
docs/superpowers/specs/2026-05-30-spec3-ui-shell-design.md (Sub-spec 3a).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Task 1: ThemeManager
# ---------------------------------------------------------------------------


def test_theme_exposes_color_tokens():
    """ThemeManager defines all spec'd color constants."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    t = ThemeManager()
    # Surfaces
    assert t.BG == "#1a1f2e"
    assert t.SURFACE == "#1e2435"
    assert t.CARD == "#263244"
    assert t.ELEVATED == "#2f3b50"
    # Sidebar
    assert t.SIDEBAR_BG == "#1a1f2e"
    assert t.SIDEBAR_ACTIVE == "#263244"
    assert t.SIDEBAR_STRIPE == "#3b82f6"
    # Accent
    assert t.ACCENT == "#3b82f6"
    assert t.ACCENT_HOVER == "#60a5fa"
    assert t.ACCENT_PRESSED == "#2563eb"
    # Text
    assert t.TEXT_PRIMARY == "#e8eef5"
    assert t.TEXT_SECONDARY == "#9ca8bd"
    assert t.TEXT_DISABLED == "#5a6478"
    assert t.TEXT_INVERSE == "#1a1f2e"
    # Borders
    assert t.DIVIDER == "#2a3142"
    assert t.BORDER == "#3a4456"


def test_theme_exposes_tier_color_tokens():
    """Tier colors preserved from V5 semantic."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    t = ThemeManager()
    assert t.TIER_STABLE == "#1e2435"
    assert t.TIER_WARNING_BG == "#3d2f1a"
    assert t.TIER_WARNING == "#f59e0b"
    assert t.TIER_DRIFT_BG == "#3d2418"
    assert t.TIER_DRIFT == "#f97316"
    assert t.TIER_OOC_BG == "#3d1818"
    assert t.TIER_OOC == "#ef4444"


def test_theme_exposes_spacing_scale():
    """Spacing scale is multiples of 4."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    t = ThemeManager()
    assert t.SPACE_XS == 4
    assert t.SPACE_SM == 8
    assert t.SPACE_MD == 12
    assert t.SPACE_LG == 16
    assert t.SPACE_XL == 24
    assert t.SPACE_2XL == 32


def test_theme_exposes_typography():
    """Font family fallback chain + size constants."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    t = ThemeManager()
    # Font family is a fallback tuple (CTk uses first available)
    assert t.FONT_FAMILY[0] == "Inter"
    assert "Segoe UI" in t.FONT_FAMILY
    # Size constants
    assert t.SIZE_CAPTION == 11
    assert t.SIZE_BODY == 13
    assert t.SIZE_HEADING == 16
    assert t.SIZE_TITLE == 20
    assert t.SIZE_DISPLAY == 28


def test_theme_exposes_corner_radii():
    """Corner radius scale."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    t = ThemeManager()
    assert t.RADIUS_SM == 4
    assert t.RADIUS_MD == 6
    assert t.RADIUS_LG == 8


def test_theme_tier_color_lookup():
    """ThemeManager exposes a tier_color(tier) helper returning (bg, fg)."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.ml.drift_types import DriftTier

    t = ThemeManager()
    bg, fg = t.tier_color(DriftTier.STABLE)
    assert bg == "#1e2435"   # blends into SURFACE
    bg, fg = t.tier_color(DriftTier.WARNING)
    assert bg == "#3d2f1a"
    assert fg == "#f59e0b"
    bg, fg = t.tier_color(DriftTier.DRIFT)
    assert bg == "#3d2418"
    assert fg == "#f97316"
    bg, fg = t.tier_color(DriftTier.OUT_OF_CONTROL)
    assert bg == "#3d1818"
    assert fg == "#ef4444"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_spec3a_shell.py -v`

Expected: 6 FAILs — ModuleNotFoundError on import.

- [ ] **Step 4: Create the ThemeManager module**

Create `src/laser_trim_analyzer/gui/v6/theme.py` with this content:

```python
"""Spec 3a — ThemeManager.

Single source of truth for V6 visual tokens: colors, typography, spacing,
corner radii.  Every V6 widget and page reads from a ThemeManager instance
passed in via constructor.  No magic numbers or hard-coded colors anywhere
else.

The token names are stable across pages; only their values may change if
James later wants a different palette.
"""
from dataclasses import dataclass
from typing import Tuple

from laser_trim_analyzer.ml.drift_types import DriftTier


@dataclass(frozen=True)
class ThemeManager:
    """Industrial-dark theme tokens.

    Frozen so accidental mutation can't desync widgets at runtime.
    """

    # ---- Color tokens ----------------------------------------------------

    # Surfaces
    BG: str = "#1a1f2e"
    SURFACE: str = "#1e2435"
    CARD: str = "#263244"
    ELEVATED: str = "#2f3b50"

    # Sidebar
    SIDEBAR_BG: str = "#1a1f2e"
    SIDEBAR_ACTIVE: str = "#263244"
    SIDEBAR_STRIPE: str = "#3b82f6"

    # Accent
    ACCENT: str = "#3b82f6"
    ACCENT_HOVER: str = "#60a5fa"
    ACCENT_PRESSED: str = "#2563eb"

    # Text
    TEXT_PRIMARY: str = "#e8eef5"
    TEXT_SECONDARY: str = "#9ca8bd"
    TEXT_DISABLED: str = "#5a6478"
    TEXT_INVERSE: str = "#1a1f2e"

    # Borders / dividers
    DIVIDER: str = "#2a3142"
    BORDER: str = "#3a4456"

    # ---- Tier colors -----------------------------------------------------

    TIER_STABLE: str = "#1e2435"      # blends with SURFACE
    TIER_WARNING_BG: str = "#3d2f1a"
    TIER_WARNING: str = "#f59e0b"
    TIER_DRIFT_BG: str = "#3d2418"
    TIER_DRIFT: str = "#f97316"
    TIER_OOC_BG: str = "#3d1818"
    TIER_OOC: str = "#ef4444"

    # ---- Typography ------------------------------------------------------

    # CTk picks the first available family at runtime
    FONT_FAMILY: Tuple[str, ...] = ("Inter", "Segoe UI", "system-ui")

    SIZE_CAPTION: int = 11
    SIZE_BODY: int = 13
    SIZE_HEADING: int = 16
    SIZE_TITLE: int = 20
    SIZE_DISPLAY: int = 28

    # ---- Spacing scale (multiples of 4) ----------------------------------

    SPACE_XS: int = 4
    SPACE_SM: int = 8
    SPACE_MD: int = 12
    SPACE_LG: int = 16
    SPACE_XL: int = 24
    SPACE_2XL: int = 32

    # ---- Corner radii ----------------------------------------------------

    RADIUS_SM: int = 4
    RADIUS_MD: int = 6
    RADIUS_LG: int = 8

    # ---- Helpers ---------------------------------------------------------

    def tier_color(self, tier: DriftTier) -> Tuple[str, str]:
        """Return (background, foreground) for the given drift tier.

        Foreground is the accent color for that tier (used for badge text,
        magnitude readouts).  For STABLE, foreground equals primary text
        because there's no semantic accent.
        """
        if tier == DriftTier.STABLE:
            return self.TIER_STABLE, self.TEXT_PRIMARY
        if tier == DriftTier.WARNING:
            return self.TIER_WARNING_BG, self.TIER_WARNING
        if tier == DriftTier.DRIFT:
            return self.TIER_DRIFT_BG, self.TIER_DRIFT
        if tier == DriftTier.OUT_OF_CONTROL:
            return self.TIER_OOC_BG, self.TIER_OOC
        # Unknown tier -- defensive
        return self.SURFACE, self.TEXT_PRIMARY
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_spec3a_shell.py -v`

Expected: 6 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/__init__.py src/laser_trim_analyzer/gui/v6/theme.py tests/test_spec3a_shell.py
git commit -m "feat(spec3a): ThemeManager with industrial-dark tokens

Frozen dataclass exposing all V6 visual tokens: surfaces, sidebar
colors, accent, text, borders, tier-color pairs, typography fallback
chain + sizes, spacing scale (multiples of 4), corner radii.

tier_color(DriftTier) helper returns (bg, fg) pair so cards/badges
can color-code consistently with V5's tier semantic.

Spec: docs/superpowers/specs/2026-05-30-spec3-ui-shell-design.md"
```

---

## Task 2: Sidebar widget

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/sidebar.py`
- Test: `tests/test_spec3a_shell.py` (APPEND)

CTk widget construction requires a root window. Tests use a module-scoped Tk root fixture to avoid repeated `CTk()` creation costs.

- [ ] **Step 1: Append the Tk root fixture + Sidebar tests**

Append to `tests/test_spec3a_shell.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: Sidebar widget
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tk_root():
    """Module-scoped headless CTk root for widget construction.

    CTk requires a root window before any widget can be instantiated.
    We create one per test module, never call mainloop(), and destroy
    it at module teardown.
    """
    import customtkinter as ctk

    root = ctk.CTk()
    # Hide the window so tests don't flash a window on screen
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


def test_sidebar_emits_selection_callback(tk_root):
    """Clicking a sidebar row calls on_select(name)."""
    from laser_trim_analyzer.gui.v6.sidebar import Sidebar
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    received: list[str] = []
    sidebar = Sidebar(
        tk_root,
        on_select=lambda name: received.append(name),
        theme=ThemeManager(),
    )
    # Simulate clicking the "model" row
    sidebar._row_frames["model"]._on_click()
    assert received == ["model"]


def test_sidebar_set_active_updates_state(tk_root):
    """set_active marks the named row as active and clears others."""
    from laser_trim_analyzer.gui.v6.sidebar import Sidebar
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    sidebar = Sidebar(tk_root, on_select=lambda _: None, theme=ThemeManager())
    sidebar.set_active("triage")
    assert sidebar._active_name == "triage"
    sidebar.set_active("settings")
    assert sidebar._active_name == "settings"


def test_sidebar_has_all_four_items(tk_root):
    """Sidebar declares exactly the 4 spec'd items in order."""
    from laser_trim_analyzer.gui.v6.sidebar import Sidebar

    expected = [("triage", "Triage"), ("process", "Process"),
                ("model", "Model"), ("settings", "Settings")]
    assert Sidebar.ITEMS == expected


def test_sidebar_set_active_unknown_name_no_op(tk_root):
    """set_active('bogus') is a defensive no-op, does not raise."""
    from laser_trim_analyzer.gui.v6.sidebar import Sidebar
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    sidebar = Sidebar(tk_root, on_select=lambda _: None, theme=ThemeManager())
    sidebar.set_active("triage")
    # No raise expected
    sidebar.set_active("bogus")
    # Previous active is preserved on no-op
    assert sidebar._active_name == "triage"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spec3a_shell.py -v -k sidebar`

Expected: 4 FAILs — Sidebar module doesn't exist.

- [ ] **Step 3: Create the Sidebar module**

Create `src/laser_trim_analyzer/gui/v6/sidebar.py`:

```python
"""Spec 3a — Sidebar widget.

Fixed-width left navigation with 4 items + active-stripe indicator.
Pure view: emits on_select(name) when a row is clicked; never reaches
into app state.  V6App.show_page() calls Sidebar.set_active(name) to
update the visual state.
"""
from typing import Callable, Dict, List, Optional, Tuple

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


SIDEBAR_WIDTH: int = 160
ROW_HEIGHT: int = 40
STRIPE_WIDTH: int = 3


class Sidebar(ctk.CTkFrame):
    """Left navigation. 4 fixed items, active-stripe indicator."""

    # Order matters: rendered top-to-bottom
    ITEMS: List[Tuple[str, str]] = [
        ("triage", "Triage"),
        ("process", "Process"),
        ("model", "Model"),
        ("settings", "Settings"),
    ]

    def __init__(
        self,
        master,
        on_select: Callable[[str], None],
        theme: ThemeManager,
        **kwargs,
    ):
        super().__init__(
            master,
            width=SIDEBAR_WIDTH,
            fg_color=theme.SIDEBAR_BG,
            corner_radius=0,
            **kwargs,
        )
        self.theme = theme
        self._on_select = on_select
        self._row_frames: Dict[str, _SidebarRow] = {}
        self._active_name: Optional[str] = None

        # Keep the frame from expanding past SIDEBAR_WIDTH
        self.pack_propagate(False)
        self.grid_propagate(False)

        # Title label
        title = ctk.CTkLabel(
            self,
            text="Laser Trim V6",
            font=(theme.FONT_FAMILY[0], theme.SIZE_CAPTION, "bold"),
            text_color=theme.TEXT_SECONDARY,
            anchor="w",
        )
        title.pack(
            side="top", fill="x",
            padx=theme.SPACE_LG, pady=(theme.SPACE_LG, theme.SPACE_MD),
        )

        # Item rows
        for name, label in self.ITEMS:
            row = _SidebarRow(
                self,
                name=name,
                label=label,
                theme=theme,
                on_click=self._row_clicked,
            )
            row.pack(side="top", fill="x")
            self._row_frames[name] = row

    def _row_clicked(self, name: str) -> None:
        """Internal click router → app's on_select."""
        self._on_select(name)

    def set_active(self, name: str) -> None:
        """Mark the named row as active; clear others.  Unknown names no-op."""
        if name not in self._row_frames:
            return
        if self._active_name == name:
            return
        if self._active_name and self._active_name in self._row_frames:
            self._row_frames[self._active_name].set_active(False)
        self._row_frames[name].set_active(True)
        self._active_name = name


class _SidebarRow(ctk.CTkFrame):
    """One sidebar item.  Holds a left-side stripe + label."""

    def __init__(
        self,
        master,
        name: str,
        label: str,
        theme: ThemeManager,
        on_click: Callable[[str], None],
    ):
        super().__init__(master, height=ROW_HEIGHT, fg_color=theme.SIDEBAR_BG)
        self.theme = theme
        self.name = name
        self._on_click_external = on_click
        self._active = False

        self.pack_propagate(False)

        # Stripe on the left -- width=0 when inactive (effectively hidden)
        self._stripe = ctk.CTkFrame(
            self, width=0, fg_color=theme.SIDEBAR_STRIPE, corner_radius=0,
        )
        self._stripe.pack(side="left", fill="y")

        # Label fills remaining row
        self._label = ctk.CTkLabel(
            self,
            text=label,
            font=(theme.FONT_FAMILY[0], theme.SIZE_BODY),
            text_color=theme.TEXT_SECONDARY,
            anchor="w",
        )
        self._label.pack(
            side="left", fill="both", expand=True,
            padx=(theme.SPACE_MD, theme.SPACE_SM),
        )

        # Bind clicks on both the frame and the label
        self.bind("<Button-1>", lambda e: self._on_click())
        self._label.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self) -> None:
        self._on_click_external(self.name)

    def set_active(self, active: bool) -> None:
        """Toggle the active visual state."""
        self._active = active
        if active:
            self._stripe.configure(width=STRIPE_WIDTH)
            self.configure(fg_color=self.theme.SIDEBAR_ACTIVE)
            self._label.configure(
                text_color=self.theme.TEXT_PRIMARY,
                font=(self.theme.FONT_FAMILY[0], self.theme.SIZE_BODY, "bold"),
            )
        else:
            self._stripe.configure(width=0)
            self.configure(fg_color=self.theme.SIDEBAR_BG)
            self._label.configure(
                text_color=self.theme.TEXT_SECONDARY,
                font=(self.theme.FONT_FAMILY[0], self.theme.SIZE_BODY),
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_spec3a_shell.py -v -k sidebar`

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/sidebar.py tests/test_spec3a_shell.py
git commit -m "feat(spec3a): Sidebar widget with active-stripe indicator

Fixed-width (160px) left navigation with 4 items (Triage / Process /
Model / Settings).  Pure view -- emits on_select(name) on row click,
never touches app state.  V6App calls set_active(name) to update
visual state.  Defensive: unknown names no-op.

Each row composed of a 0-or-3px stripe + label that swap font weight
and bg color on active state."
```

---

## Task 3: PageContainer + PageBase + _PageHeader

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/page_container.py`
- Create: `src/laser_trim_analyzer/gui/v6/page_base.py`
- Test: `tests/test_spec3a_shell.py` (APPEND)

- [ ] **Step 1: Append PageBase tests**

Append to `tests/test_spec3a_shell.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: PageBase + PageContainer + _PageHeader
# ---------------------------------------------------------------------------


def test_page_base_raises_when_build_content_not_overridden(tk_root):
    """A subclass must implement build_content() or PageBase raises."""
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    class _IncompletePage(PageBase):
        page_title = "Incomplete"
        # Intentionally does NOT override build_content

    with pytest.raises(NotImplementedError):
        _IncompletePage(tk_root, theme=ThemeManager())


def test_page_base_subclass_runs_build_content(tk_root):
    """PageBase calls subclass build_content(parent) during construction."""
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    calls: list[bool] = []

    class _TestPage(PageBase):
        page_title = "Test"

        def build_content(self, parent):
            calls.append(True)

    _TestPage(tk_root, theme=ThemeManager())
    assert calls == [True]


def test_page_base_lifecycle_hooks_default_to_noop(tk_root):
    """on_show / on_hide default no-op; subclasses don't have to override."""
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    class _TestPage(PageBase):
        page_title = "Test"

        def build_content(self, parent):
            pass

    page = _TestPage(tk_root, theme=ThemeManager())
    # No raise expected
    page.on_show()
    page.on_hide()


def test_page_base_lifecycle_hooks_call_subclass_override(tk_root):
    """When a subclass overrides on_show, it's invoked."""
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    shown: list[bool] = []
    hidden: list[bool] = []

    class _TestPage(PageBase):
        page_title = "Test"

        def build_content(self, parent):
            pass

        def on_show(self):
            shown.append(True)

        def on_hide(self):
            hidden.append(True)

    page = _TestPage(tk_root, theme=ThemeManager())
    page.on_show()
    page.on_hide()
    assert shown == [True]
    assert hidden == [True]


def test_page_container_holds_named_pages(tk_root):
    """PageContainer.add_page registers a page by name; get_page retrieves."""
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.page_container import PageContainer
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    theme = ThemeManager()

    class _DummyPage(PageBase):
        page_title = "Dummy"

        def build_content(self, parent):
            pass

    container = PageContainer(tk_root, theme=theme)
    p = _DummyPage(container, theme=theme)
    container.add_page("dummy", p)
    assert container.get_page("dummy") is p


def test_page_container_show_raises_page_with_lifecycle(tk_root):
    """show(name) calls on_hide on current page, raises new page, calls on_show."""
    from laser_trim_analyzer.gui.v6.page_base import PageBase
    from laser_trim_analyzer.gui.v6.page_container import PageContainer
    from laser_trim_analyzer.gui.v6.theme import ThemeManager

    theme = ThemeManager()
    events: list[str] = []

    class _DummyPage(PageBase):
        def __init__(self, master, theme, name):
            self._name = name
            super().__init__(master, theme=theme)
            self.page_title = name

        def build_content(self, parent):
            pass

        def on_show(self):
            events.append(f"show:{self._name}")

        def on_hide(self):
            events.append(f"hide:{self._name}")

    container = PageContainer(tk_root, theme=theme)
    p1 = _DummyPage(container, theme=theme, name="A")
    p2 = _DummyPage(container, theme=theme, name="B")
    container.add_page("A", p1)
    container.add_page("B", p2)

    container.show("A")
    container.show("B")
    assert events == ["show:A", "hide:A", "show:B"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spec3a_shell.py -v -k "page_base or page_container"`

Expected: 6 FAILs — modules don't exist.

- [ ] **Step 3: Create the PageContainer module**

Create `src/laser_trim_analyzer/gui/v6/page_container.py`:

```python
"""Spec 3a — PageContainer.

Stacked-frame container.  Each page (PageBase subclass) is added once at
app start; switching pages calls tkraise() + the page's lifecycle hooks.
Pages are never destroyed -- their internal state persists across switches.
"""
from typing import Dict, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.theme import ThemeManager


class PageContainer(ctk.CTkFrame):
    """Stacked-frame container for PageBase subclasses."""

    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(
            master, fg_color=theme.SURFACE, corner_radius=0, **kwargs,
        )
        self.theme = theme
        self._pages: Dict[str, PageBase] = {}
        self._current: Optional[str] = None

    def add_page(self, name: str, page: PageBase) -> None:
        """Register a page by name.  Page must be a child of this container."""
        self._pages[name] = page
        # Stack all pages at the same grid cell; tkraise() decides visibility
        page.grid(row=0, column=0, sticky="nsew")
        # Configure the grid cell once
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def get_page(self, name: str) -> Optional[PageBase]:
        """Look up a page by name.  Returns None if not registered."""
        return self._pages.get(name)

    def show(self, name: str) -> None:
        """Switch to the named page.  Calls on_hide on the current page,
        tkraise + on_show on the new page.  Unknown names no-op.
        """
        if name not in self._pages:
            return
        if self._current == name:
            return
        if self._current and self._current in self._pages:
            self._pages[self._current].on_hide()
        self._pages[name].tkraise()
        self._pages[name].on_show()
        self._current = name

    @property
    def current_page(self) -> Optional[str]:
        return self._current
```

- [ ] **Step 4: Create the PageBase module**

Create `src/laser_trim_analyzer/gui/v6/page_base.py`:

```python
"""Spec 3a — PageBase + _PageHeader.

All V6 pages inherit from PageBase.  PageBase builds the header strip
(page title + optional per-page action widgets) and a content frame the
subclass owns entirely.  Subclasses set ``page_title`` (class attribute),
override ``build_content(parent)``, and optionally ``header_actions()``,
``on_show()``, ``on_hide()``.
"""
from typing import List, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


HEADER_HEIGHT: int = 40


class PageBase(ctk.CTkFrame):
    """Base for all V6 pages.

    Subclass contract:
      * Set class attribute ``page_title``.
      * Override ``build_content(parent)`` — required; raises if missing.
      * Optionally override ``header_actions()`` to add per-page buttons.
      * Optionally override ``on_show()`` / ``on_hide()`` for refresh.
    """
    page_title: str = "Untitled"

    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(
            master, fg_color=theme.SURFACE, corner_radius=0, **kwargs,
        )
        self.theme = theme
        self._build_chrome()
        self.build_content(self._content)

    # ---- Public interface ------------------------------------------------

    def build_content(self, parent: ctk.CTkFrame) -> None:
        """Build the page's content widgets inside `parent`.  Required override."""
        raise NotImplementedError(
            f"{type(self).__name__} must override build_content(parent)"
        )

    def header_actions(self) -> List[ctk.CTkBaseClass]:
        """Optional widgets the header strip renders on the right.

        Default empty.  Subclass can override -- e.g. Settings adds the
        "Retrain" button, Model adds the model-selector + window dropdown.
        """
        return []

    def on_show(self) -> None:
        """Called when the page becomes visible.  Default no-op."""
        pass

    def on_hide(self) -> None:
        """Called when another page is about to be shown.  Default no-op."""
        pass

    # ---- Internal --------------------------------------------------------

    def _build_chrome(self) -> None:
        self._header = _PageHeader(
            self,
            theme=self.theme,
            title=self.page_title,
            actions=self.header_actions(),
        )
        self._header.pack(side="top", fill="x")

        # Divider line under the header
        divider = ctk.CTkFrame(
            self, height=1, fg_color=self.theme.DIVIDER, corner_radius=0,
        )
        divider.pack(side="top", fill="x")

        # Content region the subclass owns
        self._content = ctk.CTkFrame(self, fg_color="transparent")
        self._content.pack(
            fill="both", expand=True,
            padx=self.theme.SPACE_LG,
            pady=self.theme.SPACE_MD,
        )


class _PageHeader(ctk.CTkFrame):
    """Header strip: page title + right-anchored action widgets."""

    def __init__(
        self,
        master,
        theme: ThemeManager,
        title: str,
        actions: Optional[List[ctk.CTkBaseClass]] = None,
    ):
        super().__init__(
            master, height=HEADER_HEIGHT, fg_color=theme.SURFACE, corner_radius=0,
        )
        self.theme = theme
        self.pack_propagate(False)

        # Title (left)
        title_label = ctk.CTkLabel(
            self,
            text=title,
            font=(theme.FONT_FAMILY[0], theme.SIZE_TITLE, "bold"),
            text_color=theme.TEXT_PRIMARY,
            anchor="w",
        )
        title_label.pack(
            side="left", fill="y",
            padx=(theme.SPACE_LG, theme.SPACE_MD),
        )

        # Actions (right) -- pack in reverse so leftmost element wins visual order
        if actions:
            actions_frame = ctk.CTkFrame(self, fg_color="transparent")
            actions_frame.pack(
                side="right", fill="y",
                padx=(theme.SPACE_MD, theme.SPACE_LG),
            )
            for widget in actions:
                # Reparent the action widget into the actions_frame
                widget.master = actions_frame
                widget.pack(side="right", padx=theme.SPACE_XS)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_spec3a_shell.py -v -k "page_base or page_container"`

Expected: 6 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/page_base.py src/laser_trim_analyzer/gui/v6/page_container.py tests/test_spec3a_shell.py
git commit -m "feat(spec3a): PageBase + _PageHeader + PageContainer

PageBase contract: subclass sets page_title, overrides build_content,
optionally overrides header_actions / on_show / on_hide.  Base wires
the header strip + divider + content region.

PageContainer is a stacked-frame holder: add_page registers a page,
show(name) does tkraise + lifecycle hooks (on_hide on outgoing, on_show
on incoming).  Pages never destroyed -- state persists across switches."
```

---

## Task 4: V6App root + 4 placeholder pages

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/app.py`
- Test: `tests/test_spec3a_shell.py` (APPEND)

- [ ] **Step 1: Append V6App tests**

Append to `tests/test_spec3a_shell.py`:

```python
# ---------------------------------------------------------------------------
# Task 4: V6App + placeholder pages
# ---------------------------------------------------------------------------


def test_v6app_starts_on_triage_page(tmp_path, monkeypatch):
    """V6App's initial page is Triage."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.app import V6App

    # Use a tmp_path DB so the app's DB init doesn't touch real data
    cfg = Config()
    cfg.database.path = tmp_path / "v6.db"

    app = V6App(cfg)
    try:
        app.withdraw()  # hide
        assert app.page_container.current_page == "triage"
        assert app.sidebar._active_name == "triage"
    finally:
        app.destroy()


def test_v6app_show_page_routes_through_container(tmp_path):
    """V6App.show_page(name) delegates to PageContainer.show and updates sidebar."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.app import V6App

    cfg = Config()
    cfg.database.path = tmp_path / "v6.db"

    app = V6App(cfg)
    try:
        app.withdraw()
        app.show_page("settings")
        assert app.page_container.current_page == "settings"
        assert app.sidebar._active_name == "settings"
    finally:
        app.destroy()


def test_v6app_show_page_unknown_no_op(tmp_path):
    """Unknown page name doesn't change state."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.app import V6App

    cfg = Config()
    cfg.database.path = tmp_path / "v6.db"

    app = V6App(cfg)
    try:
        app.withdraw()
        before = app.page_container.current_page
        app.show_page("nonexistent")
        assert app.page_container.current_page == before
    finally:
        app.destroy()


def test_v6app_has_all_four_placeholder_pages(tmp_path):
    """V6App registers exactly the 4 spec'd pages."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.app import V6App

    cfg = Config()
    cfg.database.path = tmp_path / "v6.db"

    app = V6App(cfg)
    try:
        app.withdraw()
        names = set(app.page_container._pages.keys())
        assert names == {"triage", "process", "model", "settings"}
    finally:
        app.destroy()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spec3a_shell.py -v -k v6app`

Expected: 4 FAILs — V6App doesn't exist.

- [ ] **Step 3: Create the V6App + placeholders**

Create `src/laser_trim_analyzer/gui/v6/app.py`:

```python
"""Spec 3a — V6App root + 4 placeholder pages.

The placeholders just render a centered "coming in Spec 3X" message
so the shell + nav + theme can be exercised end-to-end before any
real page content lands.  Spec 3b/3c/3d/3e replace each placeholder
with the real page in turn.
"""
from typing import Optional

import customtkinter as ctk

from laser_trim_analyzer.config import Config
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.page_container import PageContainer
from laser_trim_analyzer.gui.v6.sidebar import Sidebar
from laser_trim_analyzer.gui.v6.theme import ThemeManager


# CustomTkinter global appearance: force dark so the theme constants control
# everything.  V6 has a single dark theme by design.
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class V6App(ctk.CTk):
    """Root window for V6.

    Layout: Sidebar on left (fixed-width), PageContainer fills remainder.
    Initial page is Triage.  Pages are constructed once at startup and
    persist across navigation.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.theme = ThemeManager()

        # DB connection -- created here so pages can pass it to API calls
        self.db = DatabaseManager(config.database.path)

        self._setup_window()
        self._build_layout()
        self._build_pages()

        self.show_page("triage")

        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    # ---- Public navigation API -------------------------------------------

    def show_page(self, name: str) -> None:
        """Switch to the named page and update the sidebar.

        Unknown names no-op (defensive).
        """
        if self.page_container.get_page(name) is None:
            return
        self.page_container.show(name)
        self.sidebar.set_active(name)

    # ---- Setup -----------------------------------------------------------

    def _setup_window(self) -> None:
        self.title("Laser Trim Analyzer V6")
        self.geometry(
            f"{self.config.gui.window_width}x{self.config.gui.window_height}"
        )
        self.minsize(900, 600)
        self.configure(fg_color=self.theme.BG)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def _build_layout(self) -> None:
        # Sidebar (left, fixed-width)
        self.sidebar = Sidebar(
            self,
            on_select=self.show_page,
            theme=self.theme,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsw")

        # Page container (right, flex)
        self.page_container = PageContainer(self, theme=self.theme)
        self.page_container.grid(row=0, column=1, sticky="nsew")

    def _build_pages(self) -> None:
        """Construct all 4 pages once.  Placeholders for 3a; replaced in
        3b/3c/3d/3e."""
        for name, label, next_spec in (
            ("triage", "Triage", "3b"),
            ("process", "Process", "3e"),
            ("model", "Model", "3c"),
            ("settings", "Settings", "3d"),
        ):
            page = _PlaceholderPage(
                self.page_container,
                theme=self.theme,
                page_title=label,
                next_spec=next_spec,
            )
            self.page_container.add_page(name, page)

    def _on_closing(self) -> None:
        self.destroy()

    def run(self) -> None:
        """Start the main loop."""
        self.mainloop()


class _PlaceholderPage(PageBase):
    """Stand-in page rendered before its real content lands in 3b/3c/3d/3e."""

    def __init__(self, master, theme, page_title, next_spec):
        # PageBase reads page_title as a class attribute, but we want it
        # instance-level here.  Set it before super().__init__ which calls
        # _build_chrome which reads page_title.
        self.page_title = page_title
        self._next_spec = next_spec
        super().__init__(master, theme=theme)

    def build_content(self, parent):
        message = ctk.CTkLabel(
            parent,
            text=f"{self.page_title} page — coming in Spec {self._next_spec}.",
            font=(self.theme.FONT_FAMILY[0], self.theme.SIZE_HEADING),
            text_color=self.theme.TEXT_SECONDARY,
        )
        message.pack(expand=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_spec3a_shell.py -v -k v6app`

Expected: 4 PASS.

- [ ] **Step 5: Run the entire test file**

Run: `pytest tests/test_spec3a_shell.py -v`

Expected: 20 PASS (T1: 6 + T2: 4 + T3: 6 + T4: 4).

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/app.py tests/test_spec3a_shell.py
git commit -m "feat(spec3a): V6App root + 4 placeholder pages

V6App: ctk.CTk root with Sidebar (left, fixed-width) + PageContainer
(right, flex).  Initial page is Triage.  show_page(name) delegates to
the container and updates sidebar.  Pages constructed once at startup;
no destroy on switch -- state persists.

Each of the 4 placeholder pages renders 'coming in Spec 3X' so the
shell can be visually verified end-to-end before real content lands."
```

---

## Task 5: `--v6` CLI flag in `__main__.py`

**Files:**
- Modify: `src/laser_trim_analyzer/__main__.py`
- Test: `tests/test_spec3a_shell.py` (APPEND)

- [ ] **Step 1: Append CLI-flag test**

Append to `tests/test_spec3a_shell.py`:

```python
# ---------------------------------------------------------------------------
# Task 5: --v6 CLI flag
# ---------------------------------------------------------------------------


def test_main_module_recognizes_v6_flag(monkeypatch, tmp_path):
    """When sys.argv includes --v6, main() instantiates V6App not the
    legacy LaserTrimApp.
    """
    import sys
    from unittest.mock import MagicMock

    # Mock both app classes so we don't actually run mainloop
    fake_v5 = MagicMock(name="LaserTrimApp")
    fake_v6 = MagicMock(name="V6App")

    # Patch the two app imports at their source modules
    import laser_trim_analyzer.app as v5_mod
    import laser_trim_analyzer.gui.v6.app as v6_mod
    monkeypatch.setattr(v5_mod, "LaserTrimApp", fake_v5)
    monkeypatch.setattr(v6_mod, "V6App", fake_v6)

    # Patch sys.argv to include --v6
    monkeypatch.setattr(sys, "argv", ["laser_trim_analyzer", "--v6"])

    # Also patch get_config so we don't depend on filesystem state
    from laser_trim_analyzer.config import Config
    cfg = Config()
    cfg.database.path = tmp_path / "test.db"
    import laser_trim_analyzer.config as config_mod
    monkeypatch.setattr(config_mod, "get_config", lambda: cfg)

    # Run main()
    from laser_trim_analyzer.__main__ import main
    main()

    # V6App should have been instantiated, not V5
    fake_v6.assert_called_once()
    fake_v5.assert_not_called()


def test_main_module_default_uses_v5_app(monkeypatch, tmp_path):
    """When --v6 is absent, main() instantiates the legacy LaserTrimApp."""
    import sys
    from unittest.mock import MagicMock

    fake_v5 = MagicMock(name="LaserTrimApp")
    fake_v6 = MagicMock(name="V6App")

    import laser_trim_analyzer.app as v5_mod
    import laser_trim_analyzer.gui.v6.app as v6_mod
    monkeypatch.setattr(v5_mod, "LaserTrimApp", fake_v5)
    monkeypatch.setattr(v6_mod, "V6App", fake_v6)

    monkeypatch.setattr(sys, "argv", ["laser_trim_analyzer"])

    from laser_trim_analyzer.config import Config
    cfg = Config()
    cfg.database.path = tmp_path / "test.db"
    import laser_trim_analyzer.config as config_mod
    monkeypatch.setattr(config_mod, "get_config", lambda: cfg)

    from laser_trim_analyzer.__main__ import main
    main()

    fake_v5.assert_called_once()
    fake_v6.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spec3a_shell.py -v -k "cli or main_module"`

Expected: 2 FAILs — `__main__.py` doesn't yet check for `--v6`.

- [ ] **Step 3: Update `__main__.py`**

In `src/laser_trim_analyzer/__main__.py`, find the `main()` function (around line 49). The current shape is:

```python
def main():
    """Main entry point for v3."""
    logger.info("Starting Laser Trim Analyzer v5...")

    try:
        from laser_trim_analyzer.app import LaserTrimApp
        from laser_trim_analyzer.config import get_config

        config = get_config()
        logger.info(f"Config loaded - Database: {config.database.path}")
        config.database.ensure_directory()
        app = LaserTrimApp(config)
        app.run()
    except ...:
        ...
```

Replace the body so it branches on `--v6`:

```python
def main():
    """Main entry point.

    By default, launches the legacy V5 UI (LaserTrimApp).
    Pass --v6 to launch the V6 UI shell (Spec 3a+).
    """
    use_v6 = "--v6" in sys.argv
    logger.info(
        f"Starting Laser Trim Analyzer v5... (UI: {'V6' if use_v6 else 'V5'})"
    )

    try:
        # Import here to avoid circular imports
        from laser_trim_analyzer.config import get_config

        # Load configuration
        config = get_config()
        logger.info(f"Config loaded - Database: {config.database.path}")

        # Ensure database directory exists
        config.database.ensure_directory()

        # Create and run the application
        if use_v6:
            from laser_trim_analyzer.gui.v6.app import V6App
            app = V6App(config)
        else:
            from laser_trim_analyzer.app import LaserTrimApp
            app = LaserTrimApp(config)
        app.run()

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -e .")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_spec3a_shell.py -v -k "cli or main_module"`

Expected: 2 PASS.

- [ ] **Step 5: Run full test file**

Run: `pytest tests/test_spec3a_shell.py -v`

Expected: 22 PASS total.

- [ ] **Step 6: Run full regression sweep**

Run:
```
pytest tests/test_spec1_untrimmed_sigma.py tests/test_log_derived_bugfixes_2026_05_30.py tests/test_5_8_2026_bugfixes.py tests/test_spec2_multi_metric_drift.py tests/test_spec3a_shell.py -v 2>&1 | tail -5
```

Expected: all PASS (~113 tests). Zero failures.

- [ ] **Step 7: Commit**

```bash
git add src/laser_trim_analyzer/__main__.py tests/test_spec3a_shell.py
git commit -m "feat(spec3a): --v6 CLI flag selects V6App

When --v6 is present in sys.argv, main() launches V6App from
gui/v6/app.py.  Without the flag, the legacy LaserTrimApp continues
to run unchanged.

This is the development-time switch for Specs 3a-3e.  Spec 3e removes
the flag and promotes V6 to the default."
```

---

## Post-implementation verification

Run the V6 app from the source tree to confirm it actually launches and is usable:

```
python -m laser_trim_analyzer --v6
```

Click each sidebar item. Each should:
1. Highlight the active row with a blue stripe and brighter background
2. Switch the page content to "{Page} — coming in Spec 3X"
3. Sidebar should be fixed-width on the left; page should fill the rest

Run again without the flag to confirm V5 still works:

```
python -m laser_trim_analyzer
```

V5 should look identical to before any V6 work landed.

## Out-of-scope reminders

- **Do not** add real page content — that's 3b/3c/3d/3e.
- **Do not** delete or modify the V5 `gui/app.py` or any `gui/pages/*` files.
- **Do not** wire any DB queries into the placeholder pages.
- **Do not** add animation, mouse-hover effects, or theme toggle.
- **Do not** add the first-startup auto-train hook (that's Spec 3d work).
