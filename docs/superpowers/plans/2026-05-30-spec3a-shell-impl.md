# Spec 3a — V6 App Shell Implementation Plan (rewritten 2026-06-01)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (or executing-plans). Steps use `- [ ]` checkboxes.
>
> **READ FIRST:** `docs/superpowers/plans/2026-06-01-spec3-rewrite-foundations.md` — verified API
> reference, corrected PageBase/V6App/theme/threading/test contracts, mission + QA rules, decision log.
> This plan implements §2 of that document. Where this plan and the original 2026-05-30 draft disagree,
> this plan wins (the original assumed widget reparenting and a module-level appearance side effect that
> don't work).

**Goal:** A runnable V6 shell at `python -m laser_trim_analyzer --v6` — industrial-dark theme, fixed
left sidebar (4 items + active stripe), `PageBase` chrome, 4 placeholder pages. V5 (`python -m
laser_trim_analyzer`, no flag) keeps working unchanged.

**Target branch:** `V6`. Verify `git branch --show-current` before Task 1.

**Fixes applied here (see foundations §6):** C5/C6 (no widget reparenting — `header_actions(parent)`),
C7 (shared test root + `make_app`, no double CTk roots), M5 (appearance mode in `__init__`, not module
import), M3 (real font-fallback via `theme.font()`), I4 (`tier_dot_color` for the invisible-STABLE-dot
bug), plus the corrected `V6App(config, db=None, auto_train_on_first_run=True)` contract.

---

## File Structure

**Created:**
- `src/laser_trim_analyzer/gui/v6/__init__.py` (empty)
- `src/laser_trim_analyzer/gui/v6/theme.py` — `ThemeManager` + tokens + `tier_color`/`tier_dot_color`/`font`
- `src/laser_trim_analyzer/gui/v6/sidebar.py` — `Sidebar` + `_SidebarRow`
- `src/laser_trim_analyzer/gui/v6/page_container.py` — `PageContainer`
- `src/laser_trim_analyzer/gui/v6/page_base.py` — `PageBase` + `_PageHeader`
- `src/laser_trim_analyzer/gui/v6/app.py` — `V6App` + `_PlaceholderPage`
- `tests/conftest.py` — shared `tk_root` + `make_app` (foundations §2.5)
- `tests/test_spec3a_shell.py`

**Modified:**
- `src/laser_trim_analyzer/__main__.py` — `--v6` flag

---

## Task 0: Shared test fixtures (conftest)

- [ ] **Step 1:** Create `tests/conftest.py` exactly as foundations §2.5, with the one correction that
  `make_app` imports the real dev path:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def tk_root():
    """One headless CTk root for all widget-construction tests (no mainloop)."""
    import customtkinter as ctk
    try:
        ctk.deactivate_automatic_dpi_awareness()
    except Exception:
        pass
    root = ctk.CTk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture
def make_app(tmp_path):
    """Factory for a V6App on an ISOLATED tmp DB, auto-train OFF, destroyed on teardown.

    Use for every V6App test. Do NOT also request `tk_root` in an app test
    (that would create two live CTk roots). Retirement (Spec 3e Graduation)
    changes only the import line below.
    """
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.gui.v6.app import V6App

    created = []

    def _factory(db_name="v6.db"):
        cfg = Config()
        cfg.database.path = tmp_path / db_name
        app = V6App(
            cfg,
            db=DatabaseManager(cfg.database.path),
            auto_train_on_first_run=False,
        )
        app.withdraw()
        created.append(app)
        return app

    yield _factory
    for app in created:
        try:
            app.destroy()
        except Exception:
            pass
```

If `tests/conftest.py` already exists, MERGE these two fixtures in (don't clobber existing fixtures);
keep the `sys.path` insert only once.

- [ ] **Step 2:** Commit.
```bash
git add tests/conftest.py
git commit -m "test(spec3a): shared headless tk_root + make_app fixtures"
```

---

## Task 1: ThemeManager + tokens + helpers

- [ ] **Step 1:** Create `src/laser_trim_analyzer/gui/v6/__init__.py` (empty).

- [ ] **Step 2:** Create `tests/test_spec3a_shell.py`:

```python
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
```

- [ ] **Step 3:** Run `pytest tests/test_spec3a_shell.py -v` → fails (no module).

- [ ] **Step 4:** Create `src/laser_trim_analyzer/gui/v6/theme.py`:

```python
"""Spec 3a — ThemeManager: the single source of V6 visual tokens.

Foundations §2.3. Frozen dataclass; every widget/page reads a shared instance.
Helpers: tier_color (bg, fg) pair, tier_dot_color (visible STABLE), font() (real
fallback to an available family).
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import customtkinter as ctk

from laser_trim_analyzer.ml.drift_types import DriftTier


@dataclass
class ThemeManager:
    # Surfaces
    BG: str = "#1a1f2e"; SURFACE: str = "#1e2435"; CARD: str = "#263244"; ELEVATED: str = "#2f3b50"
    # Sidebar
    SIDEBAR_BG: str = "#1a1f2e"; SIDEBAR_ACTIVE: str = "#263244"; SIDEBAR_STRIPE: str = "#3b82f6"
    # Accent
    ACCENT: str = "#3b82f6"; ACCENT_HOVER: str = "#60a5fa"; ACCENT_PRESSED: str = "#2563eb"
    # Text
    TEXT_PRIMARY: str = "#e8eef5"; TEXT_SECONDARY: str = "#9ca8bd"
    TEXT_DISABLED: str = "#5a6478"; TEXT_INVERSE: str = "#1a1f2e"
    # Borders
    DIVIDER: str = "#2a3142"; BORDER: str = "#3a4456"
    # Tiers (preserved V5 semantic)
    TIER_STABLE: str = "#1e2435"
    TIER_WARNING_BG: str = "#3d2f1a"; TIER_WARNING: str = "#f59e0b"
    TIER_DRIFT_BG: str = "#3d2418"; TIER_DRIFT: str = "#f97316"
    TIER_OOC_BG: str = "#3d1818"; TIER_OOC: str = "#ef4444"
    # Typography
    FONT_FAMILY: Tuple[str, ...] = ("Inter", "Segoe UI", "system-ui")
    SIZE_CAPTION: int = 11; SIZE_BODY: int = 13; SIZE_HEADING: int = 16
    SIZE_TITLE: int = 20; SIZE_DISPLAY: int = 28
    # Spacing / radii
    SPACE_XS: int = 4; SPACE_SM: int = 8; SPACE_MD: int = 12
    SPACE_LG: int = 16; SPACE_XL: int = 24; SPACE_2XL: int = 32
    RADIUS_SM: int = 4; RADIUS_MD: int = 6; RADIUS_LG: int = 8

    resolved_family: str = field(default="", init=False)

    def __post_init__(self):
        # Resolve the font family ONCE against what Tk actually has (real fallback).
        object.__setattr__(self, "resolved_family", self._resolve_family())

    def _resolve_family(self) -> str:
        try:
            import tkinter.font as tkfont
            available = set(tkfont.families())
            for fam in self.FONT_FAMILY:
                if fam in available:
                    return fam
        except Exception:
            pass
        return self.FONT_FAMILY[-1]  # last entry is the generic fallback

    # ---- Helpers ----
    def font(self, size: int, weight: str = "normal") -> ctk.CTkFont:
        return ctk.CTkFont(family=self.resolved_family, size=size, weight=weight)

    def tier_color(self, tier: DriftTier) -> Tuple[str, str]:
        """(background, foreground) for a tier. STABLE blends into SURFACE."""
        return {
            DriftTier.STABLE: (self.TIER_STABLE, self.TEXT_PRIMARY),
            DriftTier.WARNING: (self.TIER_WARNING_BG, self.TIER_WARNING),
            DriftTier.DRIFT: (self.TIER_DRIFT_BG, self.TIER_DRIFT),
            DriftTier.OUT_OF_CONTROL: (self.TIER_OOC_BG, self.TIER_OOC),
        }.get(tier, (self.SURFACE, self.TEXT_PRIMARY))

    def tier_dot_color(self, tier: DriftTier) -> str:
        """Visible dot color. STABLE → muted gray (NOT SURFACE, else invisible)."""
        if tier == DriftTier.STABLE:
            return self.TEXT_DISABLED
        return self.tier_color(tier)[1]
```

- [ ] **Step 5:** Run `pytest tests/test_spec3a_shell.py -v` → 6 PASS. Commit.
```bash
git add src/laser_trim_analyzer/gui/v6/__init__.py src/laser_trim_analyzer/gui/v6/theme.py tests/test_spec3a_shell.py
git commit -m "feat(spec3a): ThemeManager tokens + tier_color/tier_dot_color/font helpers"
```

---

## Task 2: Sidebar

Pure view; emits `on_select(name)`; `set_active(name)` updates the stripe. Retirement-proof title.

- [ ] **Step 1:** Append to `tests/test_spec3a_shell.py`:

```python
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
```

- [ ] **Step 2:** Run `-k sidebar` → fail.

- [ ] **Step 3:** Create `src/laser_trim_analyzer/gui/v6/sidebar.py` (same structure as the original
  draft; two changes: title text is `"Laser Trim Analyzer"` not `"Laser Trim V6"` since the name
  outlives V6, and all fonts go through `theme.font(...)`):

```python
"""Spec 3a — Sidebar. Pure view: on_select(name) on click; set_active(name) from V6App."""
from typing import Callable, Dict, List, Optional, Tuple

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

SIDEBAR_WIDTH = 160; ROW_HEIGHT = 40; STRIPE_WIDTH = 3


class Sidebar(ctk.CTkFrame):
    ITEMS: List[Tuple[str, str]] = [
        ("triage", "Triage"), ("process", "Process"),
        ("model", "Model"), ("settings", "Settings"),
    ]

    def __init__(self, master, on_select: Callable[[str], None], theme: ThemeManager, **kwargs):
        super().__init__(master, width=SIDEBAR_WIDTH, fg_color=theme.SIDEBAR_BG,
                         corner_radius=0, **kwargs)
        self.theme = theme
        self._on_select = on_select
        self._row_frames: Dict[str, _SidebarRow] = {}
        self._active_name: Optional[str] = None
        self.pack_propagate(False); self.grid_propagate(False)

        title = ctk.CTkLabel(self, text="Laser Trim Analyzer", font=theme.font(theme.SIZE_CAPTION, "bold"),
                             text_color=theme.TEXT_SECONDARY, anchor="w")
        title.pack(side="top", fill="x", padx=theme.SPACE_LG, pady=(theme.SPACE_LG, theme.SPACE_MD))

        for name, label in self.ITEMS:
            row = _SidebarRow(self, name=name, label=label, theme=theme, on_click=self._on_select)
            row.pack(side="top", fill="x")
            self._row_frames[name] = row

    def set_active(self, name: str) -> None:
        if name not in self._row_frames or self._active_name == name:
            return
        if self._active_name and self._active_name in self._row_frames:
            self._row_frames[self._active_name].set_active(False)
        self._row_frames[name].set_active(True)
        self._active_name = name


class _SidebarRow(ctk.CTkFrame):
    def __init__(self, master, name, label, theme: ThemeManager, on_click: Callable[[str], None]):
        super().__init__(master, height=ROW_HEIGHT, fg_color=theme.SIDEBAR_BG)
        self.theme = theme; self.name = name; self._cb = on_click; self._active = False
        self.pack_propagate(False)
        self._stripe = ctk.CTkFrame(self, width=0, fg_color=theme.SIDEBAR_STRIPE, corner_radius=0)
        self._stripe.pack(side="left", fill="y")
        self._label = ctk.CTkLabel(self, text=label, font=theme.font(theme.SIZE_BODY),
                                   text_color=theme.TEXT_SECONDARY, anchor="w")
        self._label.pack(side="left", fill="both", expand=True, padx=(theme.SPACE_MD, theme.SPACE_SM))
        for w in (self, self._label):
            w.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self): self._cb(self.name)

    def set_active(self, active: bool) -> None:
        self._active = active
        t = self.theme
        if active:
            self._stripe.configure(width=STRIPE_WIDTH); self.configure(fg_color=t.SIDEBAR_ACTIVE)
            self._label.configure(text_color=t.TEXT_PRIMARY, font=t.font(t.SIZE_BODY, "bold"))
        else:
            self._stripe.configure(width=0); self.configure(fg_color=t.SIDEBAR_BG)
            self._label.configure(text_color=t.TEXT_SECONDARY, font=t.font(t.SIZE_BODY))
```

- [ ] **Step 4:** `-k sidebar` → 3 PASS. Commit `feat(spec3a): Sidebar with active-stripe`.

---

## Task 3: PageContainer + PageBase + _PageHeader  ← contract fix (foundations §2.1)

- [ ] **Step 1:** Append tests. Note the corrected PageBase signature (`*, theme, app=None,
  page_title=None`) and that `header_actions` takes the actions `parent`:

```python
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
```

- [ ] **Step 2:** `-k "page_base or page_container"` → fail.

- [ ] **Step 3:** Create `src/laser_trim_analyzer/gui/v6/page_container.py` (unchanged from the original
  draft — it was correct):

```python
"""Spec 3a — PageContainer: stacked frames, tkraise + lifecycle hooks, no destroy on switch."""
from typing import Dict, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.theme import ThemeManager


class PageContainer(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=0, **kwargs)
        self.theme = theme
        self._pages: Dict[str, PageBase] = {}
        self._current: Optional[str] = None
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def add_page(self, name: str, page: PageBase) -> None:
        self._pages[name] = page
        page.grid(row=0, column=0, sticky="nsew")

    def get_page(self, name: str) -> Optional[PageBase]:
        return self._pages.get(name)

    def show(self, name: str) -> None:
        if name not in self._pages or self._current == name:
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

- [ ] **Step 4:** Create `src/laser_trim_analyzer/gui/v6/page_base.py` — **the corrected contract**
  (foundations §2.1). Note `_PageHeader` exposes `actions_frame` and PageBase calls
  `self.header_actions(self._header.actions_frame)`; no `widget.master =` anywhere:

```python
"""Spec 3a — PageBase + _PageHeader. Foundations §2.1.

Subclass contract:
  * class attr page_title (or pass page_title=...)
  * build_content(parent)  — REQUIRED
  * header_actions(parent)  — OPTIONAL; build widgets WITH `parent` and pack them right
  * on_show() / on_hide()   — OPTIONAL
PageBase stores `self.app` (V6App | None) and `self.theme`, and offers safe_after().
"""
from typing import Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

HEADER_HEIGHT = 44


class PageBase(ctk.CTkFrame):
    page_title: str = "Untitled"

    def __init__(self, master, *, theme: ThemeManager, app=None,
                 page_title: Optional[str] = None, **kwargs):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=0, **kwargs)
        self.theme = theme
        self.app = app
        if page_title is not None:
            self.page_title = page_title
        self._build_chrome()
        self.build_content(self._content)

    # ---- subclass interface ----
    def build_content(self, parent) -> None:
        raise NotImplementedError(f"{type(self).__name__} must override build_content(parent)")

    def header_actions(self, parent) -> None:
        """Optional. Construct action widgets with `parent` as master; pack side='right'."""
        return None

    def on_show(self) -> None: pass
    def on_hide(self) -> None: pass

    # ---- thread-safe UI update (foundations §2.4) ----
    def safe_after(self, fn, delay: int = 0) -> None:
        try:
            if self.winfo_exists():
                self.after(delay, lambda: fn() if self.winfo_exists() else None)
        except Exception:
            pass

    # ---- internal ----
    def _build_chrome(self) -> None:
        self._header = _PageHeader(self, theme=self.theme, title=self.page_title)
        self._header.pack(side="top", fill="x")
        ctk.CTkFrame(self, height=1, fg_color=self.theme.DIVIDER, corner_radius=0)\
            .pack(side="top", fill="x")
        self._content = ctk.CTkFrame(self, fg_color="transparent")
        self._content.pack(fill="both", expand=True,
                           padx=self.theme.SPACE_LG, pady=self.theme.SPACE_MD)
        # Build header actions into the header's actions frame (correct parent).
        self.header_actions(self._header.actions_frame)


class _PageHeader(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, title: str):
        super().__init__(master, height=HEADER_HEIGHT, fg_color=theme.SURFACE, corner_radius=0)
        self.theme = theme
        self.pack_propagate(False)
        ctk.CTkLabel(self, text=title, font=theme.font(theme.SIZE_TITLE, "bold"),
                     text_color=theme.TEXT_PRIMARY, anchor="w")\
            .pack(side="left", fill="y", padx=(theme.SPACE_LG, theme.SPACE_MD))
        # Right-aligned actions frame; subclasses pack widgets here.
        self.actions_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.actions_frame.pack(side="right", fill="y", padx=(theme.SPACE_MD, theme.SPACE_LG))
```

- [ ] **Step 5:** `-k "page_base or page_container"` → PASS. Commit `feat(spec3a): PageBase
  (header_actions(parent), app, safe_after) + PageContainer`.

---

## Task 4: V6App + placeholder pages  ← appearance-mode + db + auto-train fixes

- [ ] **Step 1:** Append tests (use `make_app`, never a second root):

```python
# ---- Task 4: V6App --------------------------------------------------------

def test_v6app_starts_on_triage(make_app):
    app = make_app()
    assert app.page_container.current_page == "triage"
    assert app.sidebar._active_name == "triage"


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


def test_v6app_has_four_pages(make_app):
    app = make_app()
    assert set(app.page_container._pages) == {"triage", "process", "model", "settings"}


def test_v6app_auto_train_off_does_not_offer(make_app):
    """make_app passes auto_train_on_first_run=False → no first-run hook scheduled."""
    app = make_app()
    assert app._auto_train_on_first_run is False
```

- [ ] **Step 2:** `-k v6app` → fail.

- [ ] **Step 3:** Create `src/laser_trim_analyzer/gui/v6/app.py` — appearance mode moved into
  `__init__` (M5), shared DB + injection + auto-train flag (foundations §2.2). Routing-hint methods are
  stubbed here and fleshed out in 3b/3c:

```python
"""Spec 3a — V6App root + placeholder pages. Foundations §2.2."""
from typing import Optional, Tuple

import customtkinter as ctk

from laser_trim_analyzer.config import Config
from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.page_container import PageContainer
from laser_trim_analyzer.gui.v6.sidebar import Sidebar
from laser_trim_analyzer.gui.v6.theme import ThemeManager


class V6App(ctk.CTk):
    def __init__(self, config: Config, db=None, auto_train_on_first_run: bool = True):
        super().__init__()
        # Appearance set HERE (not at import) so importing this module never mutates
        # global CTk state for V5 or test runs.
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.config = config
        self.theme = ThemeManager()
        # Share ONE DatabaseManager with the rest of the app. Production: db is None ->
        # get_database() (same singleton Processor uses). Tests inject an isolated one.
        self.db = db if db is not None else get_database()
        self._model_route: Optional[Tuple[str, Optional[str]]] = None
        self._auto_train_on_first_run = auto_train_on_first_run

        self._setup_window()
        self._build_layout()
        self._build_pages()
        self.show_page("triage")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        # First-run auto-train hook is registered in Spec 3d. Guarded by the flag.

    # ---- navigation ----
    def show_page(self, name: str) -> None:
        if self.page_container.get_page(name) is None:
            return
        self.page_container.show(name)
        self.sidebar.set_active(name)

    # ---- routing hint (3b adds consume_model_route, 3c adds consume_model_route_full) ----
    def set_model_route(self, model: str, focus_metric: Optional[str] = None) -> None:
        self._model_route = (model, focus_metric)

    # ---- setup ----
    def _setup_window(self) -> None:
        self.title("Laser Trim Analyzer")
        self.geometry(f"{self.config.gui.window_width}x{self.config.gui.window_height}")
        self.minsize(960, 640)
        self.configure(fg_color=self.theme.BG)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def _build_layout(self) -> None:
        self.sidebar = Sidebar(self, on_select=self.show_page, theme=self.theme)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.page_container = PageContainer(self, theme=self.theme)
        self.page_container.grid(row=0, column=1, sticky="nsew")

    def _build_pages(self) -> None:
        # All placeholders for 3a; each sub-spec replaces one with the real page.
        for name, label, nxt in (("triage", "Triage", "3b"), ("process", "Process", "3e"),
                                 ("model", "Model", "3c"), ("settings", "Settings", "3d")):
            self.page_container.add_page(
                name,
                _PlaceholderPage(self.page_container, theme=self.theme, app=self,
                                 page_title=label, next_spec=nxt),
            )

    def _on_closing(self) -> None:
        self.destroy()

    def run(self) -> None:
        self.mainloop()


class _PlaceholderPage(PageBase):
    def __init__(self, master, *, theme, app, page_title, next_spec):
        self._next_spec = next_spec
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    def build_content(self, parent):
        ctk.CTkLabel(parent, text=f"{self.page_title} — coming in Spec {self._next_spec}.",
                     font=self.theme.font(self.theme.SIZE_HEADING),
                     text_color=self.theme.TEXT_SECONDARY).pack(expand=True)
```

- [ ] **Step 4:** `-k v6app` → PASS. Then full file: `pytest tests/test_spec3a_shell.py -v` → all PASS.
  Commit `feat(spec3a): V6App (dark-mode in __init__, shared/injectable db, auto-train flag) + placeholders`.

---

## Task 5: `--v6` CLI flag

- [ ] **Step 1:** Append tests (these monkeypatch both app classes and `sys.argv`; unchanged in intent
  from the original, but assert against the corrected import sites):

```python
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
```

- [ ] **Step 2:** `-k "v6_flag or default_uses_v5"` → fail.

- [ ] **Step 3:** Edit `src/laser_trim_analyzer/__main__.py` `main()` to branch on `--v6` (import the
  chosen app lazily so V5 never imports V6 and vice-versa):

```python
def main():
    """Entry point. Default = V5 LaserTrimApp; --v6 = V6App (Spec 3a+)."""
    use_v6 = "--v6" in sys.argv
    logger.info(f"Starting Laser Trim Analyzer (UI: {'V6' if use_v6 else 'V5'})...")
    try:
        from laser_trim_analyzer.config import get_config
        config = get_config()
        logger.info(f"Config loaded - Database: {config.database.path}")
        config.database.ensure_directory()
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

- [ ] **Step 4:** `-k "v6_flag or default_uses_v5"` → PASS. Full file → all PASS.

- [ ] **Step 5:** Regression sweep (foundations §7) → 0 fail.

- [ ] **Step 6:** Commit `feat(spec3a): --v6 flag selects V6App (V5 default unchanged)`.

---

## Post-implementation verification

```
python -m laser_trim_analyzer --v6     # click each item: stripe + page swap; sidebar fixed-width left
python -m laser_trim_analyzer          # V5 identical to before
```

## Out of scope (3a)
- No real page content (3b–3e), no DB queries in placeholders, no first-startup auto-train (3d), no
  deletion/modification of V5 GUI (Graduation, gated, in 3e).
</content>
