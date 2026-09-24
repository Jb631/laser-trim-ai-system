# Facelift Step 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the whole V6 app the approved refined-dark look (new palette, IBM Plex, bigger type, sentence-case headers, shared building blocks) and rebuild the Findings page and the model page's Findings tab as the approved four-group design, with the two `cut_setting` fixes.

**Architecture:** Every colour, font and size already comes from `gui/v6/theme.py`, so the look is a change of token values plus a few new tokens. New pure-data module `findings/presentation.py` decides grouping, readouts, merging and order; a new shared widget `gui/v6/widgets/findings_view.py` draws it for both the page and the tab; `gui/v6/widgets/blocks.py` holds the reusable building blocks. Fonts are bundled under `gui/v6/fonts/` and loaded by `gui/v6/font_loader.py` for Tk (Windows) and matplotlib (every platform).

**Tech Stack:** Python 3, CustomTkinter 5.2.2, matplotlib, SQLAlchemy 2.0, pytest, fontTools (a matplotlib dependency).

**Spec:** `docs/superpowers/specs/2026-09-23-design-system-and-findings-page-design.md` (approved section by section by James, 2026-09-23). Visual reference: `docs/superpowers/specs/2026-09-23-findings-page-mockup.html`. Where this plan and the spec disagree, the spec wins.

## Global Constraints

- **Never open `data/analysis.db` read-write.** It is the fresh rebuild (6.23 GB). QA harnesses run on a COPY (`cp data/analysis.db /tmp/qa_copy.db`) and the copy is deleted afterwards.
- Any test or script that builds a `Processor` or `DatabaseManager` injects BOTH `laser_trim_analyzer.database.manager._db_manager` AND `laser_trim_analyzer.database._db_manager`.
- The `tk_root` fixture stays function-scoped. App tests use `make_app`; never request both in one test.
- Customer data (backlog names, PO numbers, prices) never appears in code, tests, fixtures or commit messages. Example data is INVENTED. Run `python scripts/check_no_customer_values.py` before any push.
- UI text names lasers the shop's way via `core.models.laser_label()` ("Laser 1 (LTS)"); never the code letters A/B/C.
- Workers never call Tk.
- No hex colour literal in `gui/v6` outside `theme.py` (Task 3 adds a test that enforces it).
- UI text is sentence case, except verdict badges (`PASS`, `FAIL`, `UNTRIMMED`, `NOT GRADED`, `SIGMA WATCH`), which are labels.
- `strftime("%-d")` is forbidden — it is a glibc extension and raises on Windows, the machine this runs on. Use `f"{dt.day} {dt:%b}"`.
- The gate is the whole suite: `python scripts/run_test_gate.py`. Every silence, refusal or guard a task adds is made to FAIL first (mutation-checked) before it is trusted.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Stage files by explicit path; never `git add -A`.

## File structure

| file | change | responsibility |
|---|---|---|
| `src/laser_trim_analyzer/gui/v6/theme.py` | modify | token values, new tokens, `mono()`, Medium-family mapping, `series_color()` |
| `src/laser_trim_analyzer/gui/v6/page_base.py` | modify | restyled title bar, sentence-case `_zone_header`, `set_caption()` |
| `src/laser_trim_analyzer/gui/v6/widgets/blocks.py` | create | shared building blocks (pill, tag, verdict badge, group header, row, buttons, banner) |
| `src/laser_trim_analyzer/gui/v6/font_loader.py` | create | load bundled Plex for Tk (Windows) and matplotlib (all platforms) |
| `src/laser_trim_analyzer/gui/v6/fonts/` | create | four Plex TTFs + `OFL.txt` |
| `src/laser_trim_analyzer/gui/v6/app.py` | modify | call the font loader before `ThemeManager()`; model tab route |
| chart widgets (`focus_chart.py`, `mini_trend_chart.py`, `history_tab.py`, `company_trend_chart.py`) | modify | chart font tokens, laser series colours |
| `src/laser_trim_analyzer/findings/analyzers/cut_setting.py` | modify | 60-day rule, staleness, evidence `track` and `grade` |
| `src/laser_trim_analyzer/findings/engine.py` | modify | pass the fleet's newest file date to `cut_setting` |
| `src/laser_trim_analyzer/findings/presentation.py` | create | group, readout, merge, order, caption — pure data |
| `src/laser_trim_analyzer/gui/v6/widgets/findings_view.py` | create | draws groups/rows/opened row for page and tab |
| `src/laser_trim_analyzer/gui/v6/pages/findings_page.py` | modify | rebuilt on `FindingsView` |
| `src/laser_trim_analyzer/gui/v6/widgets/findings_tab.py` | modify | rebuilt on `FindingsView`; facts restyled |
| `src/laser_trim_analyzer/gui/v6/pages/model_page.py` | modify | honour a requested tab on open |
| `scripts/render_pages.py` | create | render every page from a COPY database to PNGs for inspection |
| tests | create/modify | per task |

---

### Task 1: Tokens and the readability test

**Files:**
- Modify: `src/laser_trim_analyzer/gui/v6/theme.py`
- Modify: `tests/test_spec3a_shell.py` (pins the OLD values by design; update to the approved ones)
- Create: `tests/test_theme_contrast.py`

**Interfaces:**
- Produces: every token in the spec §1 tables, plus `ThemeManager.mono(size, weight="normal") -> CTkFont`, `ThemeManager.series_color(system: str) -> str`, attributes `resolved_family`, `resolved_medium`, `resolved_mono`, `resolved_mono_medium`, and chart font sizes `CHART_FONT_SMALL = 8.0`, `CHART_FONT = 9.0`, `CHART_FONT_LARGE = 10.0` (matplotlib points).

- [ ] **Step 1: Write the failing readability test** — `tests/test_theme_contrast.py`:

```python
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
```

- [ ] **Step 2: Run it and watch it fail** — `pytest tests/test_theme_contrast.py -q` → FAIL (missing tokens such as `PASS_FG`, `SERIES_A`; `TEXT_DISABLED` below 4.5).

- [ ] **Step 3: Replace the token block in `theme.py`** (lines 18–41 today) with exactly:

```python
    # Surfaces (refined dark, spec 2026-09-23)
    BG: str = "#111a28"; SURFACE: str = "#172233"; CARD: str = "#1c2a3e"; ELEVATED: str = "#243550"
    # Sidebar
    SIDEBAR_BG: str = "#111a28"; SIDEBAR_ACTIVE: str = "#1c2a3e"; SIDEBAR_STRIPE: str = "#4fd6b8"
    # Accent -- teal means "act here"
    ACCENT: str = "#4fd6b8"; ACCENT_HOVER: str = "#74e0c8"; ACCENT_PRESSED: str = "#36b99c"
    ACCENT_TINT: str = "#123a37"
    # Text
    TEXT_PRIMARY: str = "#f3f6fa"; TEXT_SECONDARY: str = "#b6c2d2"
    TEXT_DISABLED: str = "#93a1b6"; TEXT_INVERSE: str = "#0b1f1b"   # INVERSE = text on teal
    # Borders
    DIVIDER: str = "#26344b"; BORDER: str = "#34465f"
    # "Check this" -- coral
    CHECK: str = "#ff8f7a"; CHECK_TINT: str = "#3e2522"
    # Verdicts -- always drawn WITH their word, never colour alone
    PASS_FG: str = "#9bd66f"; PASS_BG: str = "#1f3322"
    FAIL_FG: str = "#ff8f7a"; FAIL_BG: str = "#3e2522"
    NEUTRAL_FG: str = "#c3cedb"; NEUTRAL_BG: str = "#243550"
    WATCH_FG: str = "#f5b544"; WATCH_BG: str = "#3a2f16"
    # Tiers (preserved V5 semantic; OOC brightened -- #ef4444 was 4.2:1 on its own background)
    TIER_STABLE: str = "#172233"
    TIER_WARNING_BG: str = "#3d2f1a"; TIER_WARNING: str = "#f59e0b"
    TIER_DRIFT_BG: str = "#3d2418"; TIER_DRIFT: str = "#f97316"
    TIER_OOC_BG: str = "#3d1818"; TIER_OOC: str = "#ff7a7a"
    # Charts. SERIES_* are keyed by the code's system letter; the UI still says "Laser 2 (DLTS)".
    CHART_REFERENCE: str = "#8a9bb3"
    SERIES_A: str = "#6aa8ff"; SERIES_B: str = "#b39cff"; SERIES_C: str = "#f28dc6"
    CHART_FONT_SMALL: float = 8.0; CHART_FONT: float = 9.0; CHART_FONT_LARGE: float = 10.0
    # Typography. On Windows (GDI) Plex Medium is its OWN family, not a weight of Plex Sans,
    # so "bold" is mapped onto it in font()/mono() when it is available.
    FONT_FAMILY: Tuple[str, ...] = ("IBM Plex Sans", "Segoe UI", "system-ui")
    FONT_FAMILY_MEDIUM: Tuple[str, ...] = ("IBM Plex Sans Medium",)
    MONO_FAMILY: Tuple[str, ...] = ("IBM Plex Mono", "Cascadia Mono", "Consolas", "Menlo", "Courier")
    MONO_FAMILY_MEDIUM: Tuple[str, ...] = ("IBM Plex Mono Medium",)
    SIZE_CAPTION: int = 12; SIZE_BODY: int = 14; SIZE_HEADING: int = 17
    SIZE_TITLE: int = 22; SIZE_DISPLAY: int = 30; SIZE_READOUT: int = 20
    # Spacing / radii (unchanged)
    SPACE_XS: int = 4; SPACE_SM: int = 8; SPACE_MD: int = 12
    SPACE_LG: int = 16; SPACE_XL: int = 24; SPACE_2XL: int = 32
    RADIUS_SM: int = 4; RADIUS_MD: int = 6; RADIUS_LG: int = 8
```

- [ ] **Step 4: Replace the resolution and font methods.** Keep the existing `font()` docstring's content (it documents the shared-cache reasoning). Replace `resolved_family` field, `__post_init__`, `_resolve_family` and `font` with:

```python
    resolved_family: str = field(default="", init=False)
    resolved_medium: Optional[str] = field(default=None, init=False)
    resolved_mono: str = field(default="", init=False)
    resolved_mono_medium: Optional[str] = field(default=None, init=False)
    _font_cache: Dict[Tuple[str, int, str], ctk.CTkFont] = field(
        default_factory=dict, init=False, repr=False, compare=False)
    _font_root: Optional[object] = field(
        default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        # Resolve each family ONCE against what Tk actually has (a real fallback).
        available = self._available_families()
        object.__setattr__(self, "resolved_family", self._pick(self.FONT_FAMILY, available))
        object.__setattr__(self, "resolved_medium", self._pick(self.FONT_FAMILY_MEDIUM, available, None))
        object.__setattr__(self, "resolved_mono", self._pick(self.MONO_FAMILY, available))
        object.__setattr__(self, "resolved_mono_medium", self._pick(self.MONO_FAMILY_MEDIUM, available, None))

    @staticmethod
    def _available_families() -> set:
        try:
            import tkinter.font as tkfont
            return set(tkfont.families())
        except Exception:          # no Tk root yet (tests, imports): resolve to the fallbacks
            return set()

    _NO_DEFAULT = object()

    @classmethod
    def _pick(cls, candidates, available, default=_NO_DEFAULT):
        for fam in candidates:
            if fam in available:
                return fam
        return candidates[-1] if default is cls._NO_DEFAULT else default

    def _resolve_family(self) -> str:
        """Kept for callers of the old API: the resolved Sans family."""
        return self._pick(self.FONT_FAMILY, self._available_families())

    def font(self, size: int, weight: str = "normal") -> ctk.CTkFont:
        """<keep the existing docstring text here, unchanged>"""
        if weight == "bold" and self.resolved_medium:
            return self._shared_font(self.resolved_medium, size, "normal")
        return self._shared_font(self.resolved_family, size, weight)

    def mono(self, size: int, weight: str = "normal") -> ctk.CTkFont:
        """The same shared font, in the mono family -- every NUMBER in the app uses this."""
        if weight == "bold" and self.resolved_mono_medium:
            return self._shared_font(self.resolved_mono_medium, size, "normal")
        return self._shared_font(self.resolved_mono, size, weight)

    def _shared_font(self, family: str, size: int, weight: str) -> ctk.CTkFont:
        root = getattr(tkinter, "_default_root", None)
        if root is not self._font_root:
            self._font_cache.clear()
            self._font_root = root
        key = (family, size, weight)
        cached = self._font_cache.get(key)
        if cached is None:
            cached = ctk.CTkFont(family=family, size=size, weight=weight)
            self._font_cache[key] = cached
        return cached

    def series_color(self, system: str) -> str:
        """Line colour for a laser's series, by the code's system letter."""
        return {"A": self.SERIES_A, "B": self.SERIES_B, "C": self.SERIES_C}.get(system, self.CHART_REFERENCE)
```

- [ ] **Step 5: Update `tests/test_spec3a_shell.py`** so its pinned values are the approved ones: lines 12–16 to the new surface/sidebar/accent/text values from Step 3; line 37 to `assert t.FONT_FAMILY[0] == "IBM Plex Sans" and "Segoe UI" in t.FONT_FAMILY`; line 44 to `("#172233", "#f3f6fa")`. Leave line 67's `resolved_family in FONT_FAMILY` assertion as it is.

- [ ] **Step 6: Run** `pytest tests/test_theme_contrast.py tests/test_spec3a_shell.py -q` → PASS.

- [ ] **Step 7: Mutation-check** — temporarily set `TEXT_DISABLED` back to `#7b8aa0`: the contrast test must go red. Temporarily set `SERIES_A` to `#9bd66f` (PASS green): the series test must go red. Restore both.

- [ ] **Step 8: Run the whole gate** `python scripts/run_test_gate.py` → GREEN (fix any other test that pinned an old value, in the test, with a comment naming the spec).

- [ ] **Step 9: Commit** `theme.py`, `tests/test_theme_contrast.py`, `tests/test_spec3a_shell.py` (and any test touched in Step 8) — message: "feat(theme): the refined-dark palette, IBM Plex families and a readability test".

---

### Task 2: Building blocks, sentence-case headers, page caption

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/blocks.py`
- Modify: `src/laser_trim_analyzer/gui/v6/page_base.py` (`_PageHeader` title style, `_zone_header`, new `set_caption`)
- Modify the seven `_zone_header` callers and two other all-caps labels:
  - `pages/triage_page.py:63` "WHAT THE APP IS TELLING YOU" → "What the app is telling you"
  - `pages/triage_page.py:68` "WHAT YOU'RE LOOKING AT" → "What you're looking at"
  - `pages/home_page.py:56` "BRING IN WHAT'S NEW" → "Bring in what's new"
  - `pages/home_page.py:132` "WHAT THE APP IS TELLING YOU" → "What the app is telling you"
  - `pages/model_page.py:158` "WHAT THE APP IS TELLING YOU" → "What the app is telling you"
  - `pages/model_page.py:201` "WHAT YOU'RE LOOKING AT" → "What you're looking at"
  - `pages/findings_page.py:26` — removed in Task 7 (the page is rebuilt)
  - `sidebar.py:54` "OTHER VIEWS" → "Other views"
  - `widgets/stats_table.py:204` "ALL UNITS" → "All units"
- Create: `tests/test_blocks.py`

**Interfaces:**
- Consumes: Task 1 tokens, `theme.mono()`.
- Produces (all return an UNPACKED widget; the caller packs it):
  - `count_pill(parent, theme, count: int, tone: str = "act") -> CTkLabel`
  - `tag(parent, theme, text: str) -> CTkLabel`
  - `verdict_badge(parent, theme, verdict: str) -> CTkLabel`
  - `group_header(parent, theme, title: str, count: int, *, column: str = "", tone: str = "act", meaning: str = "") -> CTkFrame`
  - `row(parent, theme, model: str, statement: str, value_text: str, *, tags=(), on_click=None, value_color=None) -> CTkFrame`
  - `primary_button(parent, theme, text: str, command) -> CTkButton`
  - `link_button(parent, theme, text: str, command) -> CTkButton`
  - `banner(parent, theme, text: str, tone: str = "check") -> CTkLabel`
  - `VERDICT_TOKENS: Dict[str, Tuple[str, str]]`
  - `PageBase.set_caption(text: str) -> None` — shows (or clears, on `""`) one caption line under the title bar.

- [ ] **Step 1: Write the failing test** — `tests/test_blocks.py`:

```python
"""The building blocks draw only from theme tokens and behave as the spec says (section 2)."""
import customtkinter as ctk
import pytest

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets import blocks


@pytest.fixture
def t(tk_root):
    return ThemeManager()


def _texts(w):
    out = []
    for c in w.winfo_children():
        if isinstance(c, (ctk.CTkLabel, ctk.CTkButton)):
            out.append(c.cget("text"))
        out.extend(_texts(c))
    return out


@pytest.mark.parametrize("verdict,fg,bg", [
    ("PASS", "PASS_FG", "PASS_BG"), ("fail", "FAIL_FG", "FAIL_BG"),
    ("UNTRIMMED", "NEUTRAL_FG", "NEUTRAL_BG"), ("NOT GRADED", "NEUTRAL_FG", "NEUTRAL_BG"),
    ("SIGMA WATCH", "WATCH_FG", "WATCH_BG"),
])
def test_every_verdict_carries_its_word_and_its_own_colour(tk_root, t, verdict, fg, bg):
    b = blocks.verdict_badge(tk_root, t, verdict)
    assert b.cget("text") == verdict.upper()
    assert b.cget("text_color") == getattr(t, fg) and b.cget("fg_color") == getattr(t, bg)


def test_an_unknown_verdict_is_neutral_never_a_pass_colour(tk_root, t):
    b = blocks.verdict_badge(tk_root, t, "SOMETHING NEW")
    assert b.cget("fg_color") == t.NEUTRAL_BG


def test_sigma_watch_is_never_drawn_as_a_failure(tk_root, t):
    # CLAUDE.md: sigma is a drift-watch signal, never a rejection.
    b = blocks.verdict_badge(tk_root, t, "SIGMA WATCH")
    assert b.cget("fg_color") != t.FAIL_BG and b.cget("text_color") != t.FAIL_FG


def test_pills_use_teal_to_act_and_coral_to_check(tk_root, t):
    assert blocks.count_pill(tk_root, t, 9).cget("text_color") == t.ACCENT
    c = blocks.count_pill(tk_root, t, 22, tone="check")
    assert c.cget("text_color") == t.CHECK and c.cget("text") == "22"
    assert blocks.count_pill(tk_root, t, 1234).cget("text") == "1,234"


def test_group_header_shows_title_count_unit_and_meaning(tk_root, t):
    g = blocks.group_header(tk_root, t, "Check the test", 22, column="tracks", tone="check",
                            meaning="Graded against more than one limit table")
    texts = _texts(g)
    for want in ("Check the test", "22", "tracks", "Graded against more than one limit table"):
        assert want in texts


def test_a_row_is_clickable_everywhere_and_lifts_on_hover(tk_root, t):
    hits = []
    r = blocks.row(tk_root, t, "6607", "Laser 1 (LTS): cut 6900 → try 6800", "~300",
                   tags=("both tracks",), on_click=lambda: hits.append(1))
    r.pack(); tk_root.update()
    assert {"6607", "Laser 1 (LTS): cut 6900 → try 6800", "~300", "both tracks"} <= set(_texts(r))
    r._on_click_all()                    # the handler bound to the row and every child
    assert hits == [1]
    r._set_hover(True);  assert r.cget("fg_color") == t.ELEVATED
    r._set_hover(False); assert r.cget("fg_color") == "transparent"


def test_the_primary_button_carries_dark_text_on_teal(tk_root, t):
    b = blocks.primary_button(tk_root, t, "Open 6607", lambda: None)
    assert b.cget("fg_color") == t.ACCENT and b.cget("text_color") == t.TEXT_INVERSE


def test_a_check_banner_is_coral(tk_root, t):
    b = blocks.banner(tk_root, t, "Findings could not be loaded")
    assert b.cget("text_color") == t.CHECK and b.cget("fg_color") == t.CHECK_TINT


def test_no_block_hard_codes_a_colour():
    import inspect, re
    src = inspect.getsource(blocks)
    assert not re.search(r'"#[0-9a-fA-F]{6}"', src), "colours come from theme.py only"
```

- [ ] **Step 2: Run and watch it fail** — `pytest tests/test_blocks.py -q` → FAIL (`blocks` does not exist).

- [ ] **Step 3: Create `blocks.py`:**

```python
"""The building blocks every V6 page is made of (spec 2026-09-23, section 2).

Plain functions that build CustomTkinter widgets from theme tokens and return them UNPACKED,
so the caller decides the layout. No colour or size in this file is a literal: all of it
comes from the ThemeManager, so a change to the look is a change to theme.py alone.
"""
from typing import Callable, Dict, Iterable, Optional, Tuple

import customtkinter as ctk

VERDICT_TOKENS: Dict[str, Tuple[str, str]] = {
    "PASS": ("PASS_FG", "PASS_BG"),
    "FAIL": ("FAIL_FG", "FAIL_BG"),
    "UNTRIMMED": ("NEUTRAL_FG", "NEUTRAL_BG"),
    "NOT GRADED": ("NEUTRAL_FG", "NEUTRAL_BG"),
    # Amber, never the FAIL coral: sigma is a drift-watch signal, not a rejection (CLAUDE.md).
    "SIGMA WATCH": ("WATCH_FG", "WATCH_BG"),
}


def count_pill(parent, theme, count: int, tone: str = "act") -> ctk.CTkLabel:
    t = theme
    fg, bg = (t.CHECK, t.CHECK_TINT) if tone == "check" else (t.ACCENT, t.ACCENT_TINT)
    return ctk.CTkLabel(parent, text=f"{count:,}", font=t.mono(t.SIZE_CAPTION), text_color=fg,
                        fg_color=bg, corner_radius=t.RADIUS_SM, padx=8, height=20)


def tag(parent, theme, text: str) -> ctk.CTkLabel:
    t = theme
    return ctk.CTkLabel(parent, text=text, font=t.font(t.SIZE_CAPTION), text_color=t.NEUTRAL_FG,
                        fg_color=t.NEUTRAL_BG, corner_radius=t.RADIUS_SM, padx=7, height=20)


def verdict_badge(parent, theme, verdict: str) -> ctk.CTkLabel:
    """The word AND the colour, never colour alone (colour-blind safe)."""
    t = theme
    word = str(verdict).upper()
    fg_name, bg_name = VERDICT_TOKENS.get(word, ("NEUTRAL_FG", "NEUTRAL_BG"))
    return ctk.CTkLabel(parent, text=word, font=t.mono(t.SIZE_CAPTION, "bold"),
                        text_color=getattr(t, fg_name), fg_color=getattr(t, bg_name),
                        corner_radius=t.RADIUS_SM, padx=8, height=22)


def group_header(parent, theme, title: str, count: int, *, column: str = "",
                 tone: str = "act", meaning: str = "") -> ctk.CTkFrame:
    t = theme
    wrap = ctk.CTkFrame(parent, fg_color="transparent")
    top = ctk.CTkFrame(wrap, fg_color="transparent")
    top.pack(fill="x")
    ctk.CTkLabel(top, text=title, font=t.font(t.SIZE_HEADING, "bold"), text_color=t.TEXT_PRIMARY,
                 anchor="w").pack(side="left")
    count_pill(top, t, count, tone).pack(side="left", padx=(t.SPACE_SM, 0))
    if column:
        ctk.CTkLabel(top, text=column, font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY,
                     anchor="e").pack(side="right")
    ctk.CTkFrame(wrap, height=1, fg_color=t.DIVIDER, corner_radius=0).pack(fill="x", pady=(t.SPACE_XS, 0))
    if meaning:
        ctk.CTkLabel(wrap, text=meaning, font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY,
                     anchor="w", justify="left", wraplength=1000).pack(fill="x", pady=(t.SPACE_XS, 0))
    return wrap


def row(parent, theme, model: str, statement: str, value_text: str, *, tags: Iterable[str] = (),
        on_click: Optional[Callable[[], None]] = None, value_color: Optional[str] = None) -> ctk.CTkFrame:
    """One line: model (mono) . statement and tags . readout (mono, right). Click anywhere on it."""
    t = theme
    frame = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=t.RADIUS_SM)
    frame.grid_columnconfigure(1, weight=1)
    ctk.CTkLabel(frame, text=model, font=t.mono(t.SIZE_CAPTION + 1), text_color=t.TEXT_SECONDARY,
                 anchor="w", width=96).grid(row=0, column=0, sticky="w", padx=(t.SPACE_SM, t.SPACE_MD),
                                            pady=t.SPACE_SM)
    mid = ctk.CTkFrame(frame, fg_color="transparent")
    mid.grid(row=0, column=1, sticky="ew")
    ctk.CTkLabel(mid, text=statement, font=t.font(t.SIZE_BODY), text_color=t.TEXT_PRIMARY,
                 anchor="w", justify="left", wraplength=720).pack(side="left")
    for text in tags:
        tag(mid, t, text).pack(side="left", padx=(t.SPACE_SM, 0))
    ctk.CTkLabel(frame, text=value_text, font=t.mono(t.SIZE_READOUT, "bold"),
                 text_color=value_color or t.TEXT_PRIMARY, anchor="e"
                 ).grid(row=0, column=2, sticky="e", padx=(t.SPACE_MD, t.SPACE_SM))
    ctk.CTkFrame(frame, height=1, fg_color=t.DIVIDER, corner_radius=0
                 ).grid(row=1, column=0, columnspan=3, sticky="ew")

    def set_hover(on: bool) -> None:
        frame.configure(fg_color=t.ELEVATED if on else "transparent")

    def click_all(_event=None) -> None:
        if on_click is not None:
            on_click()

    def leave(event) -> None:
        # <Leave> also fires when the pointer moves onto a CHILD label; only drop the hover
        # when the pointer has really left the row, or the row flickers as you cross it.
        under = frame.winfo_containing(event.x_root, event.y_root)
        if under is None or not str(under).startswith(str(frame)):
            set_hover(False)

    def bind_all(w) -> None:
        w.bind("<Button-1>", click_all, add="+")
        w.bind("<Enter>", lambda _e: set_hover(True), add="+")
        w.bind("<Leave>", leave, add="+")
        try:
            w.configure(cursor="hand2")
        except Exception:                 # some CTk internals refuse a cursor; clicks still work
            pass
        for child in w.winfo_children():
            bind_all(child)

    if on_click is not None:
        bind_all(frame)
    frame._on_click_all = click_all       # test hooks: the real bound handlers
    frame._set_hover = set_hover
    return frame


def primary_button(parent, theme, text: str, command) -> ctk.CTkButton:
    """At most ONE per screen. Dark text: white on this teal measures 1.8:1."""
    t = theme
    return ctk.CTkButton(parent, text=text, command=command, fg_color=t.ACCENT, hover_color=t.ACCENT_HOVER,
                         text_color=t.TEXT_INVERSE, font=t.font(t.SIZE_BODY, "bold"),
                         corner_radius=t.RADIUS_MD, height=32)


def link_button(parent, theme, text: str, command) -> ctk.CTkButton:
    """Teal text that acts -- 'Show all 53'."""
    t = theme
    return ctk.CTkButton(parent, text=text, command=command, fg_color="transparent",
                         hover_color=t.CARD, text_color=t.ACCENT, font=t.font(t.SIZE_BODY),
                         anchor="w", width=0, height=28)


def banner(parent, theme, text: str, tone: str = "check") -> ctk.CTkLabel:
    """A notice. 'check' is coral (something failed or needs a look); 'quiet' is plain."""
    t = theme
    fg, bg = (t.CHECK, t.CHECK_TINT) if tone == "check" else (t.TEXT_SECONDARY, t.CARD)
    return ctk.CTkLabel(parent, text=text, font=t.font(t.SIZE_BODY), text_color=fg, fg_color=bg,
                        corner_radius=t.RADIUS_MD, anchor="w", justify="left", wraplength=1000,
                        padx=12, pady=8)
```

- [ ] **Step 4: Restyle `page_base.py`.** Read `_PageHeader` and `_build_chrome` first. The title label uses `t.font(t.SIZE_TITLE, "bold")` and `t.TEXT_PRIMARY`. Replace `_zone_header` with:

```python
    def _zone_header(self, parent, title: str, caption: str) -> None:
        """A section inside a page: sentence-case title, its caption on the line below.

        This used to be an 11 px all-caps label in the accent colour -- the hardest text on
        the screen to read, and much of the 'dated' look (spec 2026-09-23). Callers now pass
        sentence case."""
        t = self.theme
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(side="top", fill="x", pady=(t.SPACE_SM, t.SPACE_XS))
        ctk.CTkLabel(wrap, text=title, font=t.font(t.SIZE_HEADING, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w").pack(fill="x")
        if caption:
            ctk.CTkLabel(wrap, text=caption, font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY,
                         anchor="w", justify="left", wraplength=1000).pack(fill="x")
```

and add, after `_build_chrome` creates the title bar, a caption label packed directly under it (initially empty and not packed), plus:

```python
    def set_caption(self, text: str) -> None:
        """One line under the page title -- a page's headline in words. '' hides it."""
        self._caption.configure(text=text or "")
        if text and not self._caption.winfo_ismapped():
            self._caption.pack(side="top", fill="x", padx=self.theme.SPACE_LG,
                               before=self._content, pady=(0, self.theme.SPACE_SM))
        elif not text and self._caption.winfo_ismapped():
            self._caption.pack_forget()
```

(`self._caption` is a `CTkLabel` in `SIZE_BODY`, `TEXT_SECONDARY`, anchored west. If `_content` is not packed with `pack`, adapt `set_caption` to that geometry manager and keep the behaviour.)

- [ ] **Step 5: Change the nine all-caps strings** listed under Files to sentence case exactly as written there.

- [ ] **Step 6: Add a structural test** to `tests/test_blocks.py`:

```python
def test_no_v6_heading_is_shouting():
    """Sentence case everywhere (spec section 2). Verdict words are labels, not headings."""
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parents[1] / "src/laser_trim_analyzer/gui/v6"
    allowed = {"PASS", "FAIL", "UNTRIMMED", "NOT GRADED", "SIGMA WATCH"}
    shouting = []
    for p in root.rglob("*.py"):
        for m in re.finditer(r'"([A-Z][A-Z \',&/—-]{8,})"', p.read_text()):
            if m.group(1).strip() not in allowed:
                shouting.append(f"{p.name}: {m.group(1)}")
    assert not shouting, shouting
```

(Run it. If it finds all-caps strings other than the nine listed, convert them too and list them in the commit message.)

- [ ] **Step 7: Run** `pytest tests/test_blocks.py -q` → PASS; mutation-check: change `VERDICT_TOKENS["SIGMA WATCH"]` to the FAIL pair → the sigma test goes red; restore.

- [ ] **Step 8: Gate** `python scripts/run_test_gate.py` → GREEN (tests that asserted an all-caps heading text are updated to the sentence-case text).

- [ ] **Step 9: Commit** — "feat(ui): shared building blocks and sentence-case headers".

---

### Task 3: Charts — the chart font scale and the laser colours

**Files:**
- Modify: `widgets/focus_chart.py`, `widgets/mini_trend_chart.py`, `widgets/history_tab.py`, `widgets/company_trend_chart.py` (all under `src/laser_trim_analyzer/gui/v6/`)
- Create: `tests/test_chart_tokens.py`

**Interfaces:**
- Consumes: `CHART_FONT_SMALL`, `CHART_FONT`, `CHART_FONT_LARGE`, `series_color()` (Task 1).

The 30 literal chart sizes move onto the chart scale, one step up (spec: "every size goes up one step"): **6, 6.5, 7, 7.5 → `t.CHART_FONT_SMALL` (8); 8 → `t.CHART_FONT` (9); 9 → `t.CHART_FONT_LARGE` (10)**. This applies to every `fontsize=<number>` and `labelsize=<number>` in those four files. `widgets/stats_table.py` `minsize=190`/`minsize=78` are grid column widths, NOT fonts — do not touch them.

In `company_trend_chart.py`, delete `_SYSTEM_COLORS = {"A": "#22c55e", "B": "#a78bfa", "C": "#f59e0b"}` and use `self.theme.series_color(system)` wherever it was read. (Laser 2's green collided with PASS green; laser 3's amber with SIGMA WATCH.)

- [ ] **Step 1: Write the failing test** — `tests/test_chart_tokens.py`:

```python
"""Chart text and colours come from the theme, like everything else (spec section 1)."""
import pathlib
import re

V6 = pathlib.Path(__file__).resolve().parents[1] / "src/laser_trim_analyzer/gui/v6"


def test_no_chart_font_size_is_a_literal():
    offenders = []
    for p in (V6 / "widgets").glob("*.py"):
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"\b(fontsize|labelsize)\s*=\s*[0-9]", line):
                offenders.append(f"{p.name}:{n}: {line.strip()}")
    assert not offenders, offenders


def test_no_hex_colour_outside_the_theme():
    offenders = []
    for p in V6.rglob("*.py"):
        if p.name == "theme.py":
            continue
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r'"#[0-9a-fA-F]{6}"', line):
                offenders.append(f"{p.relative_to(V6)}:{n}")
    assert not offenders, offenders


def test_the_table_column_widths_were_left_alone():
    src = (V6 / "widgets" / "stats_table.py").read_text()
    assert "minsize=190" in src and "minsize=78" in src
```

- [ ] **Step 2: Run and watch it fail** (30 font-size offenders, 3 hex colours).
- [ ] **Step 3: Apply the mapping** in the four files.
- [ ] **Step 4: Run** `pytest tests/test_chart_tokens.py -q` → PASS; `python scripts/run_test_gate.py` → GREEN.
- [ ] **Step 5: Render the charts on a copy** — `cp data/analysis.db /tmp/qa_copy.db && python scripts/chart_qa_render_all.py qa_output /tmp/qa_copy.db`, then OPEN the PNGs in `qa_output/` and check: text readable, laser lines distinguishable, no label clipped. Delete `/tmp/qa_copy.db`.
- [ ] **Step 6: Commit** — "feat(charts): chart text on the theme's scale, laser colours clear of meaning".

---

### Task 4: Bundle IBM Plex

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/fonts/IBMPlexSans-Regular.ttf`, `IBMPlexSans-Medium.ttf`, `IBMPlexMono-Regular.ttf`, `IBMPlexMono-Medium.ttf`, `OFL.txt`
- Create: `src/laser_trim_analyzer/gui/v6/font_loader.py`
- Modify: `src/laser_trim_analyzer/gui/v6/app.py` (call the loader before `ThemeManager()`)
- Modify: `pyproject.toml` (package data, if the build config needs it for the fonts to ship)
- Create: `tests/test_font_loader.py`

**The files — ASK JAMES FIRST.** Downloading needs his explicit permission (safety rule: name the files, source and size). Source: Google's official fonts repository, SIL Open Font Licence 1.1:
`https://github.com/google/fonts/raw/main/ofl/ibmplexsans/IBMPlexSans-Regular.ttf`, `…/IBMPlexSans-Medium.ttf`, `…/ofl/ibmplexmono/IBMPlexMono-Regular.ttf`, `…/IBMPlexMono-Medium.ttf`, and `…/ofl/ibmplexsans/OFL.txt` — about 0.8 MB in all. If permission is not given, skip to Task 5: every family tuple falls back (Segoe UI / Cascadia Mono on Windows), and this task resumes when it is.

**Interfaces:**
- Produces: `font_loader.FONT_DIR: Path`, `font_loader.FILES: Tuple[str, ...]`, `font_loader.load_bundled_fonts() -> Dict[str, Dict[str, bool]]` returning `{filename: {"tk": bool, "matplotlib": bool}}`; idempotent.

- [ ] **Step 1: Write the failing test** — `tests/test_font_loader.py`:

```python
"""The bundled fonts exist, are the families theme.py asks for, and fail loudly, not silently."""
import logging

from fontTools.ttLib import TTFont

from laser_trim_analyzer.gui.v6 import font_loader
from laser_trim_analyzer.gui.v6.theme import ThemeManager


def _family(path):
    """Name ID 1 -- the LEGACY family, which is what Windows GDI (and so Tk) uses."""
    return TTFont(str(path))["name"].getDebugName(1)


def test_every_bundled_file_and_the_licence_are_present():
    for name in font_loader.FILES:
        assert (font_loader.FONT_DIR / name).is_file(), name
    assert (font_loader.FONT_DIR / "OFL.txt").is_file()


def test_the_files_are_the_families_the_theme_asks_for():
    # This turns a Windows-only assumption into a fact checkable on any machine: GDI files
    # Plex Medium as its OWN family, which is why theme.font() maps "bold" onto it.
    t = ThemeManager
    got = {_family(font_loader.FONT_DIR / n) for n in font_loader.FILES}
    for want in (t.FONT_FAMILY[0], t.FONT_FAMILY_MEDIUM[0], t.MONO_FAMILY[0], t.MONO_FAMILY_MEDIUM[0]):
        assert want in got, f"{want!r} not provided by any bundled file ({sorted(got)})"


def test_a_missing_file_falls_back_and_says_so(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(font_loader, "FONT_DIR", tmp_path)          # nothing in it
    monkeypatch.setattr(font_loader, "_DONE", None)
    with caplog.at_level(logging.WARNING):
        result = font_loader.load_bundled_fonts()
    assert all(not r["tk"] and not r["matplotlib"] for r in result.values())
    assert any("IBMPlexSans-Regular.ttf" in rec.getMessage() for rec in caplog.records)


def test_matplotlib_can_use_plex_on_any_machine():
    import matplotlib.font_manager as fm
    font_loader.load_bundled_fonts()
    names = {f.name for f in fm.fontManager.ttflist}
    assert "IBM Plex Sans" in names and "IBM Plex Mono" in names


def test_loading_twice_is_harmless():
    a = font_loader.load_bundled_fonts()
    b = font_loader.load_bundled_fonts()
    assert a == b
```

- [ ] **Step 2: Run and watch it fail** (`font_loader` missing).

- [ ] **Step 3: Create `font_loader.py`:**

```python
"""Load the bundled IBM Plex fonts: privately for Tk on Windows, and for matplotlib everywhere.

Tk on Windows: CustomTkinter's FontManager.windows_load_font() calls AddFontResourceEx
PRIVATELY -- the font exists for this process only, no install, no admin, no IT request.
Its default passes FR_NOT_ENUM, which hides the family from tkinter.font.families(), and the
theme resolves families from that list -- so it would silently fall back to Segoe UI even with
Plex loaded. It is called with enumerable=True for exactly that reason.

macOS/Linux Tk: nothing here (FontManager.load_font returns False on macOS); the theme's family
tuples fall back. matplotlib reads TTF files directly, so charts get Plex on every platform.

A file that fails to load is logged at WARNING and the app continues on the fallback fonts --
never silently, never fatally.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FILES = ("IBMPlexSans-Regular.ttf", "IBMPlexSans-Medium.ttf",
         "IBMPlexMono-Regular.ttf", "IBMPlexMono-Medium.ttf")

_DONE: Optional[Dict[str, Dict[str, bool]]] = None


def _load_tk(path: Path) -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        from customtkinter import FontManager
        return bool(FontManager.windows_load_font(str(path), private=True, enumerable=True))
    except Exception:
        logger.exception("could not load %s for the window", path.name)
        return False


def _load_matplotlib(path: Path) -> bool:
    try:
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(str(path))
        return True
    except Exception:
        logger.exception("could not load %s for charts", path.name)
        return False


def load_bundled_fonts() -> Dict[str, Dict[str, bool]]:
    """Load every bundled file once. Returns {file: {"tk": bool, "matplotlib": bool}}."""
    global _DONE
    if _DONE is not None:
        return _DONE
    result: Dict[str, Dict[str, bool]] = {}
    for name in FILES:
        path = FONT_DIR / name
        if not path.is_file():
            logger.warning("bundled font %s is missing from %s; using the fallback fonts",
                           name, FONT_DIR)
            result[name] = {"tk": False, "matplotlib": False}
            continue
        result[name] = {"tk": _load_tk(path), "matplotlib": _load_matplotlib(path)}
        if sys.platform.startswith("win") and not result[name]["tk"]:
            logger.warning("bundled font %s did not load for the window; using the fallback", name)
    if any(r["matplotlib"] for r in result.values()):
        import matplotlib
        matplotlib.rcParams["font.family"] = ["IBM Plex Sans", "DejaVu Sans"]
    _DONE = result
    return result
```

- [ ] **Step 4: Call it in `app.py`** immediately BEFORE `self.theme = ThemeManager()`:

```python
        # Fonts first: the theme resolves its families from what Tk can see, so Plex must be
        # loaded before the ThemeManager is built (font_loader explains the Windows detail).
        from laser_trim_analyzer.gui.v6.font_loader import load_bundled_fonts
        load_bundled_fonts()
```

- [ ] **Step 5: Download the files (after James says yes)** into `src/laser_trim_analyzer/gui/v6/fonts/`. Check `pyproject.toml`: if the package build lists package data explicitly, add `gui/v6/fonts/*.ttf` and `OFL.txt`; if it is an editable install reading the source tree only, note that in the commit.
- [ ] **Step 6: Run** `pytest tests/test_font_loader.py -q` → PASS; mutation-check: set `enumerable=False` in `_load_tk` — on this Mac the test cannot see it, so ADD to the commit message that the Windows path is verified at work by James's check "the mono numbers are Plex". Then gate.
- [ ] **Step 7: Commit** the four TTFs, `OFL.txt`, `font_loader.py`, `app.py`, `tests/test_font_loader.py` (and `pyproject.toml` if changed) — "feat(fonts): bundle IBM Plex, loaded privately on Windows and for charts everywhere".

---

### Task 5: The two `cut_setting` fixes, and what "now" means

**Files:**
- Modify: `src/laser_trim_analyzer/findings/analyzers/cut_setting.py`
- Modify: `src/laser_trim_analyzer/findings/engine.py`
- Modify: `tests/test_findings_cut_setting.py`

**Interfaces:**
- `cut_setting.analyze(model, tracks, laser_label, now: Optional[datetime] = None)`; constants `MIN_RUN_DAYS = 60`, `STALE_DAYS = 180`.
- Evidence gains: `"track": str`, `"grade": "same_days" | "side_by_side" | "two_periods"`, `"stale": bool`, `"last_ran": "YYYY-MM-DD"`.
- `engine.compute_for_model(db, model, fleet_latest: Optional[datetime] = None)`; `engine._fleet_latest(db) -> Optional[datetime]`.

Measured before this was written (rebuilt database, 2026-09-23): 80 live settings; median run 298 days; 13 under 60 days, including 8397-2's 23-day "best". The database's newest trim file is 2026-09-22; 8397-2's newest is April 2025.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_findings_cut_setting.py`:

```python
from datetime import datetime, timedelta


def test_a_setting_that_ran_for_under_sixty_days_cannot_be_crowned_best():
    # 8397-2's shape: the "winner" ran for 23 days. It must not become the recommendation.
    short = [track(i, START + timedelta(hours=9 * i), 1.0, 4000.0 if i % 2 else 5000.0, True)
             for i in range(60)]                                    # 60 tracks over ~22 days, all passing
    long = block(1000, days(START, 201)[-1], 240, 2.0, 0.4, 0.4)
    facts, findings = cut_setting.analyze("M", short + long, label)
    assert findings == []
    listed = {s["setting"] for s in facts["Laser 1 (LTS) · Track A"]["settings"]}
    assert 1.0 in listed                    # still shown as context, just not as the answer


def test_a_setting_that_ran_long_enough_still_wins():
    f = only(cut_setting.analyze("M", two_blocks(0.6, 0.3), label)[1])
    assert f.evidence["best"] == 1.0


def test_a_model_that_has_not_run_for_six_months_is_not_now_running():
    tracks = two_blocks(0.6, 0.3)
    newest = max(t.file_date for t in tracks)
    f = only(cut_setting.analyze("M", tracks, label, now=newest + timedelta(days=400))[1])
    assert "now running" not in f.title and "last ran" in f.title
    assert f.expected_gain_points is None and f.tracks_per_year is None
    assert f.evidence["stale"] is True


def test_a_model_still_in_production_is_now_running():
    tracks = two_blocks(0.6, 0.3)
    newest = max(t.file_date for t in tracks)
    f = only(cut_setting.analyze("M", tracks, label, now=newest + timedelta(days=10))[1])
    assert "now running" in f.title and f.expected_gain_points is not None
    assert f.evidence["stale"] is False


def test_the_evidence_names_its_track_and_its_grade():
    f = only(cut_setting.analyze("M", two_blocks(0.6, 0.3), label)[1])
    assert f.evidence["track"] == "Track A"
    assert f.evidence["grade"] == "two_periods"
```

- [ ] **Step 2: Run and watch them fail.**

- [ ] **Step 3: Implement in `cut_setting.py`:**
  1. Add `MIN_RUN_DAYS = 60` and `STALE_DAYS = 180` beside the other constants, each with a one-line comment giving the measurement above.
  2. Add `def _span_days(rows) -> int: return (max(t.file_date for t in rows) - min(t.file_date for t in rows)).days`.
  3. `analyze(model, tracks, laser_label, now=None)`.
  4. After `rates` is computed, replace the choice of `best` with:
     ```python
     # A trial is not a run: only a setting with MIN_RUN_DAYS of production behind it can be
     # the recommendation. The current setting is always eligible -- it is what we compare to.
     eligible = {s: r for s, r in rates.items()
                 if s == current or _span_days(live[s]) >= MIN_RUN_DAYS}
     if len(eligible) < 2:
         continue
     best = max(eligible, key=lambda s: (eligible[s], -abs(s - current)))
     ```
     and use `eligible` in place of `rates` for `gain` and the summary.
  5. Compute the grade ONCE, before the strength text, and branch the strength text on it:
     ```python
     if overlap is not None and overlap >= 25.0:
         grade = "same_days"
     elif all_months and both_months * 2 >= all_months:
         grade = "side_by_side"
     else:
         grade = "two_periods"
     ```
  6. Staleness, before building the Finding:
     ```python
     stale = now is not None and newest < now - timedelta(days=STALE_DAYS)
     ```
     When `stale`: title `f"{laser_label(system)}: cut {best:g} passed {gain:.0f} points more often than {current:g}, the setting it last ran at ({newest:%b %Y})"`; summary starts `f"This model has not run on this laser since {newest:%B %Y}, so nothing here is running now -- it is the record of what worked. "`; `expected_gain_points=None`, `gain_definition=""`, `scope_annual_tracks=0`. Otherwise unchanged ("… than the {current:g} now running").
  7. Evidence adds `"track": track_name, "grade": grade, "stale": stale, "last_ran": newest.date().isoformat()`.

- [ ] **Step 4: Implement in `engine.py`:**
  ```python
  def _fleet_latest(db):
      """The newest trim file in the database -- what 'now' means to a finding."""
      from .data import _date
      with db.session() as s:
          v = s.execute(text("SELECT MAX(file_date) FROM analysis_results "
                             "WHERE system IN ('A','B','C')")).scalar()
      return _date(v)
  ```
  `compute_for_model(db, model, fleet_latest=None)`: if `fleet_latest is None`, `fleet_latest = _fleet_latest(db)`; call `cut_setting.analyze(model, tracks, _laser_label, now=fleet_latest)`. In `refresh_findings`, compute `fleet_latest = _fleet_latest(db)` once before the loop and pass it to every `compute_for_model` call.

- [ ] **Step 5: Run** `pytest tests/test_findings_cut_setting.py tests/test_findings_engine_db.py -q` → PASS.
- [ ] **Step 6: Mutation-check** each rule: set `MIN_RUN_DAYS = 0` → the 60-day test goes red; set `STALE_DAYS = 100000` → the six-months test goes red; restore.
- [ ] **Step 7: On a COPY, confirm 8397-2:** `cp data/analysis.db /tmp/f.db && python scripts/refresh_findings.py /tmp/f.db 8397-2` — its finding no longer claims "+48" or "now running". Record what it says now in the commit message. Delete `/tmp/f.db`.
- [ ] **Step 8: Gate, then commit** — "fix(findings): a short trial cannot be the best setting, and 'now running' means now".

---

### Task 6: `findings/presentation.py` — the rules, as data

**Files:**
- Create: `src/laser_trim_analyzer/findings/presentation.py`
- Create: `tests/test_findings_presentation.py`

**Interfaces:**
- Produces: `ROWS_PER_GROUP = 5`; `GroupSpec(key, title, meaning, column, tone, empty)`; `GROUPS` (4); `OTHER`; `ANALYZER_GROUP`; `Row(group, model, statement, value, findings, tags, when)` with `.key` and `.merged`; `Group(spec, rows)`; `group_key(finding) -> str`; `readout(finding) -> Optional[float]`; `statement(finding) -> str`; `arrange(findings, *, include_empty=True) -> List[Group]`; `caption(groups, findings) -> str`; `value_text(group_key, value) -> str`; `value_tone(group_key, value) -> Optional[str]` (`"up"`/`"down"` for history).

- [ ] **Step 1: Write the failing tests** — `tests/test_findings_presentation.py`:

```python
"""Every rule the Findings screens follow, tested without a window (spec section 3)."""
import logging

import pytest

from laser_trim_analyzer.findings import presentation as P


def f(analyzer, model="M", **kw):
    base = {"analyzer": analyzer, "model": model, "category": kw.pop("category", ""),
            "title": kw.pop("title", "t"), "summary": "s", "systems": kw.pop("systems", ["B"]),
            "n_units": kw.pop("n_units", 100), "tracks_per_year": kw.pop("tpy", None),
            "evidence": kw.pop("evidence", {}), "computed_at": kw.pop("computed_at", None)}
    base.update(kw)
    return base


def cut(model="M", best=6800.0, current=6900.0, tpy=100.0, grade="two_periods", track="Track A"):
    return f("cut_setting", model, category="Cut setting", tpy=tpy,
             evidence={"best": best, "current": current, "grade": grade, "track": track})


def test_every_analyzer_the_engine_runs_has_a_group():
    import pkgutil
    import laser_trim_analyzer.findings.analyzers as pkg
    names = {m.name for m in pkgutil.iter_modules(pkg.__path__)}
    missing = sorted(names - set(P.ANALYZER_GROUP))
    assert not missing, f"analyzers with no group on the Findings page: {missing}"


def test_an_unknown_analyzer_is_shown_under_other_never_dropped(caplog):
    with caplog.at_level(logging.ERROR):
        groups = P.arrange([f("brand_new")])
    other = [g for g in groups if g.spec.key == "other"]
    assert other and other[0].rows and other[0].rows[0].findings[0]["analyzer"] == "brand_new"
    assert any("brand_new" in r.getMessage() for r in caplog.records)


def test_the_four_groups_come_in_order_and_other_only_when_needed():
    keys = [g.spec.key for g in P.arrange([])]
    assert keys == ["yield", "laser_time", "check", "history"]


def test_a_model_tab_hides_empty_groups():
    keys = [g.spec.key for g in P.arrange([cut()], include_empty=False)]
    assert keys == ["yield"]


@pytest.mark.parametrize("category,field,value", [
    ("Trim avoidance", "arrive_in_spec_n", 1008),
    ("Pass effectiveness", "multi_cut_n", 420),
    ("Multi-pass burden", "tracks_over_recipe", 124),
])
def test_laser_time_readouts_count_tracks(category, field, value):
    x = f("trim_effort" if category != "Multi-pass burden" else "pass_burden",
          category=category, evidence={"facts": {field: value, "unplanned_passes": 999}})
    assert P.readout(x) == value


def test_a_history_readout_is_the_pass_rate_move():
    x = f("recipe_change", evidence={"before": {"trim_pass_pct": 80.0}, "after": {"trim_pass_pct": 58.0}})
    assert P.readout(x) == pytest.approx(-22.0)
    assert P.value_text("history", -22.0) == "-22" and P.value_tone("history", -22.0) == "down"
    assert P.value_tone("history", 21.0) == "up"


def test_two_tracks_saying_the_same_thing_become_one_row():
    rows = P.arrange([cut(tpy=182.0, track="Track A"), cut(tpy=118.0, track="Track B")])[0].rows
    assert len(rows) == 1 and rows[0].merged
    assert rows[0].value == pytest.approx(300.0)
    assert "both tracks" in rows[0].tags


def test_different_recommendations_never_merge():
    rows = P.arrange([cut(best=6800.0), cut(best=6700.0)])[0].rows
    assert len(rows) == 2


def test_a_merged_row_shows_its_weakest_evidence():
    rows = P.arrange([cut(grade="same_days"), cut(grade="two_periods", track="Track B")])[0].rows
    assert "two periods · test first" in rows[0].tags and "same days" not in rows[0].tags


def test_yield_sorts_by_rate_then_size_with_no_rate_last():
    rows = P.arrange([cut("A", tpy=None, best=1.0), cut("B", tpy=5.0, best=2.0),
                      cut("C", tpy=50.0, best=3.0)])[0].rows
    assert [r.model for r in rows] == ["C", "B", "A"]
    assert P.value_text("yield", None) == "—" and P.value_text("yield", 300.4) == "~300"


def test_history_is_newest_first():
    old = f("recipe_change", "OLD", evidence={"after": {"first": "2023-07-17"}})
    new = f("recipe_change", "NEW", evidence={"after": {"first": "2025-01-13"}})
    rows = [g for g in P.arrange([old, new]) if g.spec.key == "history"][0].rows
    assert [r.model for r in rows] == ["NEW", "OLD"]


def test_statements_read_like_the_approved_design():
    assert P.statement(cut()) == "Laser 1 (LTS): cut 6900 → try 6800"
    x = f("recipe_change", title="Laser 1 (LTS): recipe changed from 1 cut (cut length 6800) "
                                 "to 1 cut (cut length 6900)",
          evidence={"after": {"first": "2024-10-04"}})
    assert P.statement(x) == "Oct 2024 · Laser 1 (LTS): cut 6800 → 6900"
    y = f("recipe_change", title="Laser 1 (LTS): recipe changed from 1 cut (cut length 2950) "
                                 "to 2 cuts (cut length 2950, 4500)",
          evidence={"after": {"first": "2025-01-13"}})
    assert P.statement(y) == "Jan 2025 · Laser 1 (LTS): 1 cut → 2 cuts"


def test_the_caption_counts_rows_after_merging_and_dates_portably():
    fs = [cut(computed_at="2026-09-23 11:52:51"), cut(track="Track B"),
          f("limit_tables"), f("trim_effort", category="Trim avoidance")]
    cap = P.caption(P.arrange(fs), fs)
    assert cap.startswith("1 change worth testing · 1 way to save laser time · 1 test to check")
    assert cap.endswith("worked out 23 Sep")          # never strftime("%-d"): it raises on Windows


def test_value_text_for_counts():
    assert P.value_text("laser_time", 1008.0) == "1,008" and P.value_text("check", 22.0) == "22"
```

- [ ] **Step 2: Run and watch it fail.**

- [ ] **Step 3: Create `presentation.py`:**

```python
"""How the Findings page and tab arrange findings: group, number, merge, order.

Pure data, no Tk -- so every rule those screens follow can be tested without a window. The
screens draw what arrange() returns and decide nothing themselves.
Spec: docs/superpowers/specs/2026-09-23-design-system-and-findings-page-design.md, section 3.
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ROWS_PER_GROUP = 5


@dataclass(frozen=True)
class GroupSpec:
    key: str
    title: str
    meaning: str
    column: str          # the unit of the readout column, printed in the group header
    tone: str            # "act" -> teal count pill, "check" -> coral
    empty: str           # what the group says when it has nothing


GROUPS: Tuple[GroupSpec, ...] = (
    GroupSpec("yield", "Change a setting to raise yield",
              "A different setting did better on the same test", "tracks a year", "act",
              "Nothing here yet. A finding appears when a model ran two cut settings, or two "
              "incoming-resistance windows, on the same test and one did clearly better."),
    GroupSpec("laser_time", "Laser time you could save",
              "Units cut that didn't need it, or given more cuts than planned", "tracks", "act",
              "Nothing here yet. A finding appears when units arrive already inside their limits, "
              "or take more cuts than their recipe asks for."),
    GroupSpec("check", "Check the test",
              "Graded against more than one limit table, so pass rates across the change don't compare",
              "tracks", "check",
              "Nothing to check. A finding appears when a model is graded against more than one "
              "limit table."),
    GroupSpec("history", "What changed", "Recipe changes, newest first", "pass-rate move", "act",
              "No recipe changes found."),
)
OTHER = GroupSpec("other", "Other findings",
                  "From an analyzer this page does not know how to group yet", "", "check", "")

# Every analyzer -> its group. A finding from an analyzer NOT listed here goes under OTHER
# and is logged -- never dropped: on these screens silence is itself a result, so a finding
# must never be able to vanish. A test asserts this covers every module in findings/analyzers.
ANALYZER_GROUP: Dict[str, str] = {
    "cut_setting": "yield",
    "ink_target": "yield",
    "trim_effort": "laser_time",
    "pass_burden": "laser_time",
    "limit_tables": "check",
    "recipe_change": "history",
}

_GRADE_TAG = {"same_days": "same days", "side_by_side": "side by side",
              "two_periods": "two periods · test first"}
_GRADE_ORDER = ("two_periods", "side_by_side", "same_days")        # weakest first
_LASER_TIME_FIELD = {"Trim avoidance": "arrive_in_spec_n", "Pass effectiveness": "multi_cut_n",
                     "Multi-pass burden": "tracks_over_recipe"}
_RECIPE = re.compile(r"^(?P<laser>[^:]+): recipe changed from (?P<a>.+) to (?P<b>.+)$")
_CUTS = re.compile(r"^(?P<n>\d+) cuts?(?: \(cut length (?P<len>[^)]+)\))?$")


@dataclass
class Row:
    group: str
    model: str
    statement: str
    value: Optional[float]
    findings: List[Dict[str, Any]]
    tags: List[str] = field(default_factory=list)
    when: Optional[str] = None                      # ISO date, history rows

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.group, self.model, self.statement)

    @property
    def merged(self) -> bool:
        return len(self.findings) > 1


@dataclass
class Group:
    spec: GroupSpec
    rows: List[Row]


def group_key(finding: Dict[str, Any]) -> str:
    return ANALYZER_GROUP.get(finding.get("analyzer"), OTHER.key)


def _num(x) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _laser(finding) -> str:
    from laser_trim_analyzer.core.models import laser_label
    systems = finding.get("systems") or ()
    return laser_label(systems[0]) if systems else ""


def _month(iso) -> str:
    try:
        return datetime.fromisoformat(str(iso)[:10]).strftime("%b %Y")
    except (TypeError, ValueError):
        return ""


def readout(finding: Dict[str, Any]) -> Optional[float]:
    g = group_key(finding)
    ev = finding.get("evidence") or {}
    if g == "yield":
        return _num(finding.get("tracks_per_year"))
    if g == "laser_time":
        name = _LASER_TIME_FIELD.get(finding.get("category"))
        v = _num((ev.get("facts") or {}).get(name)) if name else None
        return v if v is not None else _num(finding.get("n_units"))
    if g == "history":
        pb = _num((ev.get("before") or {}).get("trim_pass_pct"))
        pa = _num((ev.get("after") or {}).get("trim_pass_pct"))
        return None if pb is None or pa is None else pa - pb
    return _num(finding.get("n_units"))


def _recipe_move(a: str, b: str) -> str:
    ma, mb = _CUTS.match(a.strip()), _CUTS.match(b.strip())
    if not (ma and mb):
        return f"{a} → {b}"
    na, nb = int(ma["n"]), int(mb["n"])
    if na != nb:
        return f"{na} cut{'s' if na != 1 else ''} → {nb} cut{'s' if nb != 1 else ''}"
    if ma["len"] and mb["len"]:
        return f"cut {ma['len']} → {mb['len']}"
    return f"{a} → {b}"


def statement(finding: Dict[str, Any]) -> str:
    title = str(finding.get("title") or "")
    ev = finding.get("evidence") or {}
    if finding.get("analyzer") == "cut_setting" and _num(ev.get("best")) is not None \
            and _num(ev.get("current")) is not None:
        best, current = float(ev["best"]), float(ev["current"])
        if ev.get("stale"):
            return f"{_laser(finding)}: {best:g} did better than {current:g}, last run {_month(ev.get('last_ran'))}"
        return f"{_laser(finding)}: cut {current:g} → try {best:g}"
    if group_key(finding) == "history":
        when = _month((ev.get("after") or {}).get("first"))
        m = _RECIPE.match(title)
        body = f"{m['laser']}: {_recipe_move(m['a'], m['b'])}" if m else title
        return f"{when} · {body}" if when else body
    return title


def _merge_key(finding) -> Optional[Tuple]:
    """Same model, analyzer and recommendation, differing only by track -> one row."""
    if finding.get("analyzer") != "cut_setting":
        return None
    ev = finding.get("evidence") or {}
    if _num(ev.get("best")) is None or _num(ev.get("current")) is None:
        return None
    return (finding.get("model"), tuple(finding.get("systems") or ()),
            float(ev["best"]), float(ev["current"]), bool(ev.get("stale")))


def _tags(members: Sequence[Dict[str, Any]]) -> List[str]:
    tags: List[str] = []
    if len(members) == 2:
        tags.append("both tracks")
    elif len(members) > 2:
        tags.append(f"{len(members)} tracks")
    grades = [(m.get("evidence") or {}).get("grade") for m in members]
    grades = [g for g in grades if g in _GRADE_TAG]
    if grades:
        tags.append(_GRADE_TAG[min(grades, key=_GRADE_ORDER.index)])   # the weakest, honestly
    return tags


def _sum(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) if vals else None


def _size(r: Row) -> int:
    return sum(int(f.get("n_units") or 0) for f in r.findings)


def _sort(key: str, rows: List[Row]) -> None:
    if key == "history":
        rows.sort(key=lambda r: r.model)
        rows.sort(key=lambda r: r.when or "", reverse=True)          # newest first; undated last
    elif key == "yield":
        rows.sort(key=lambda r: (r.value is None, -(r.value or 0.0), -_size(r), r.model))
    else:
        rows.sort(key=lambda r: (r.value is None, -(r.value or 0.0), r.model))


def arrange(findings: Sequence[Dict[str, Any]], *, include_empty: bool = True) -> List[Group]:
    buckets: Dict[str, List[Row]] = {s.key: [] for s in (*GROUPS, OTHER)}
    merged: Dict[Tuple, Row] = {}
    unmapped = set()
    for fnd in findings:
        g = group_key(fnd)
        if g == OTHER.key:
            unmapped.add(str(fnd.get("analyzer")))
        mk = _merge_key(fnd)
        if mk is not None and mk in merged:
            merged[mk].findings.append(fnd)
            continue
        ev = fnd.get("evidence") or {}
        r = Row(group=g, model=str(fnd.get("model") or ""), statement=statement(fnd), value=None,
                findings=[fnd], when=(ev.get("after") or {}).get("first") if g == "history" else None)
        if mk is not None:
            merged[mk] = r
        buckets[g].append(r)
    if unmapped:
        logger.error("Findings page: no group for analyzer(s) %s -- shown under 'Other findings'",
                     sorted(unmapped))
    out: List[Group] = []
    for spec in (*GROUPS, OTHER):
        rows = buckets[spec.key]
        for r in rows:
            r.value = _sum(readout(x) for x in r.findings)
            r.tags = _tags(r.findings)
        _sort(spec.key, rows)
        if rows or (include_empty and spec is not OTHER):
            out.append(Group(spec, rows))
    return out


def _plural(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def caption(groups: Sequence[Group], findings: Sequence[Dict[str, Any]]) -> str:
    n = {g.spec.key: len(g.rows) for g in groups}
    parts = [_plural(n.get("yield", 0), "change worth testing", "changes worth testing"),
             _plural(n.get("laser_time", 0), "way to save laser time", "ways to save laser time"),
             _plural(n.get("check", 0), "test to check", "tests to check")]
    stamps = [str(x.get("computed_at")) for x in findings if x.get("computed_at")]
    if stamps:
        try:
            dt = datetime.fromisoformat(max(stamps)[:19])
            parts.append(f"worked out {dt.day} {dt:%b}")          # NOT %-d: it raises on Windows
        except ValueError:
            pass
    return " · ".join(parts)


def value_text(group: str, value: Optional[float]) -> str:
    if value is None:
        return "—"
    if group == "yield":
        return f"~{value:,.0f}"
    if group == "history":
        return f"{value:+.0f}"
    return f"{value:,.0f}"


def value_tone(group: str, value: Optional[float]) -> Optional[str]:
    if group != "history" or value is None or value == 0:
        return None
    return "up" if value > 0 else "down"
```

- [ ] **Step 4: Run** → PASS. **Mutation-check:** delete `"limit_tables"` from `ANALYZER_GROUP` → the coverage test goes red; make `_merge_key` return `None` always → the merge test goes red; change `_GRADE_ORDER` to strongest-first → the weakest-evidence test goes red; restore each.
- [ ] **Step 5: Gate, commit** — "feat(findings): the Findings screens' rules as tested data".

---

### Task 7: The shared findings view, and the Findings page rebuilt

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/findings_view.py`
- Modify: `src/laser_trim_analyzer/gui/v6/pages/findings_page.py`
- Modify: `src/laser_trim_analyzer/gui/v6/app.py` — `set_model_route(model, focus_metric=None, tab=None)` stores `tab` in a separate `self._model_tab_route`; add `consume_model_tab() -> Optional[str]` (pops it). `consume_model_route` / `consume_model_route_full` are UNCHANGED.
- Modify: `src/laser_trim_analyzer/gui/v6/pages/model_page.py` — when the page shows a model, `tab = self.app.consume_model_tab()`; if it names a tab the page has (the Findings tab's name as registered in its tab view), select it.
- Create: `tests/test_findings_view.py`
- Modify: `tests/test_findings_page.py` (same behaviours, new structure)

**Interfaces:**
- `FindingsView(master, theme, *, on_open: Optional[Callable[[str], None]] = None, include_empty: bool = True)` — a `CTkFrame`.
  - `set_findings(findings: List[dict]) -> None`
  - `open_key: Optional[Tuple]` — the open row (None = none open)
  - `toggle(key) -> None`, `show_all(group_key: str) -> None`
  - `row_widgets: Dict[Tuple, CTkFrame]` (for tests)

- [ ] **Step 1: Write the failing tests** — `tests/test_findings_view.py`:

```python
"""The shared findings view: rows per group, one open at a time, Show all, open the model."""
import customtkinter as ctk

from laser_trim_analyzer.findings import presentation as P
from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.findings_view import FindingsView


def cut(model, tpy, best=6800.0, current=6900.0, track="Track A"):
    return {"analyzer": "cut_setting", "model": model, "category": "Cut setting", "title": "t",
            "summary": f"summary for {model} {track}", "systems": ["B"], "n_units": 100,
            "tracks_per_year": tpy,
            "evidence": {"best": best, "current": current, "grade": "two_periods", "track": track,
                         "group": {"settings": [
                             {"setting": best, "n": 813, "pass_pct": 87.1, "window": "a .. b"},
                             {"setting": current, "n": 1283, "pass_pct": 60.4, "window": "c .. d"}]}}}


def _texts(w):
    out = []
    for c in w.winfo_children():
        if isinstance(c, (ctk.CTkLabel, ctk.CTkButton)):
            out.append(c.cget("text"))
        out.extend(_texts(c))
    return out


def test_each_group_shows_five_rows_then_offers_the_rest(tk_root):
    v = FindingsView(tk_root, ThemeManager())
    v.set_findings([cut(f"M{i}", float(100 - i), best=float(i)) for i in range(8)])
    assert len(v.row_widgets) == P.ROWS_PER_GROUP
    assert "Show all 8" in _texts(v)
    v.show_all("yield")
    assert len(v.row_widgets) == 8 and "Show all 8" not in _texts(v)


def test_opening_a_row_shows_its_evidence_and_only_one_is_open(tk_root):
    v = FindingsView(tk_root, ThemeManager(), on_open=lambda m: None)
    v.set_findings([cut("6607", 182.0), cut("8232-1", 66.0, best=4100.0, current=4000.0)])
    first, second = list(v.row_widgets)
    v.toggle(first)
    assert v.open_key == first and any("summary for 6607" in x for x in _texts(v))
    assert "Open 6607" in _texts(v)
    v.toggle(second)
    assert v.open_key == second and not any("summary for 6607" in x for x in _texts(v))
    v.toggle(second)
    assert v.open_key is None


def test_the_open_button_goes_to_the_model(tk_root):
    opened = []
    v = FindingsView(tk_root, ThemeManager(), on_open=opened.append)
    v.set_findings([cut("6607", 182.0)])
    v.toggle(next(iter(v.row_widgets)))
    btn = [b for b in v._detail.winfo_children() if isinstance(b, ctk.CTkButton)][0]
    btn.invoke()
    assert opened == ["6607"]


def test_without_an_open_handler_there_is_no_open_button(tk_root):
    v = FindingsView(tk_root, ThemeManager(), on_open=None)     # the model page's own tab
    v.set_findings([cut("6607", 182.0)])
    v.toggle(next(iter(v.row_widgets)))
    assert not any(x.startswith("Open ") for x in _texts(v))


def test_a_merged_row_opens_with_each_track_named(tk_root):
    v = FindingsView(tk_root, ThemeManager(), on_open=lambda m: None)
    v.set_findings([cut("6607", 182.0, track="Track A"), cut("6607", 118.0, track="Track B")])
    assert len(v.row_widgets) == 1
    v.toggle(next(iter(v.row_widgets)))
    texts = _texts(v)
    assert "Track A" in texts and "Track B" in texts and "~300" in texts


def test_an_empty_group_says_what_would_fill_it(tk_root):
    v = FindingsView(tk_root, ThemeManager(), include_empty=True)
    v.set_findings([cut("6607", 182.0)])
    texts = _texts(v)
    assert any(x.startswith("No recipe changes found") for x in texts)


def test_the_tab_view_hides_empty_groups(tk_root):
    v = FindingsView(tk_root, ThemeManager(), include_empty=False)
    v.set_findings([cut("6607", 182.0)])
    assert "What changed" not in _texts(v)
```

- [ ] **Step 2: Run and watch them fail.**

- [ ] **Step 3: Create `findings_view.py`:**

```python
"""Draws arranged findings: group headers, rows, the one open row, Show all.

Shared by the Findings page (every model, all four groups, an Open button) and the model
page's Findings tab (one model, empty groups hidden, no Open button). The RULES live in
findings/presentation.py; this only draws them.
"""
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import customtkinter as ctk

from laser_trim_analyzer.findings import presentation as P
from laser_trim_analyzer.gui.v6.widgets import blocks


class FindingsView(ctk.CTkFrame):
    def __init__(self, master, theme, *, on_open: Optional[Callable[[str], None]] = None,
                 include_empty: bool = True, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._on_open = on_open
        self._include_empty = include_empty
        self._findings: List[Dict[str, Any]] = []
        self._expanded: Set[str] = set()
        self._rows: Dict[Tuple, P.Row] = {}
        self.row_widgets: Dict[Tuple, ctk.CTkFrame] = {}
        self.open_key: Optional[Tuple] = None
        self._detail: Optional[ctk.CTkFrame] = None

    # ---- public ----
    def set_findings(self, findings: List[Dict[str, Any]]) -> None:
        self._findings = list(findings or [])
        self._render()

    def toggle(self, key: Tuple) -> None:
        """Open a row in place; opening another, or the same one again, closes it."""
        if self._detail is not None:
            self._detail.destroy()
            self._detail = None
        if self.open_key == key:
            self.open_key = None
            return
        self.open_key = key
        row_widget = self.row_widgets.get(key)
        if row_widget is not None:
            self._detail = self._draw_detail(self._rows[key])
            self._detail.pack(after=row_widget, fill="x", padx=self.theme.SPACE_SM,
                              pady=(0, self.theme.SPACE_SM))

    def show_all(self, group_key: str) -> None:
        self._expanded.add(group_key)
        self._render()

    # ---- drawing ----
    def _render(self) -> None:
        t = self.theme
        for child in self.winfo_children():
            child.destroy()
        self._rows.clear()
        self.row_widgets.clear()
        self._detail = None
        was_open, self.open_key = self.open_key, None
        for group in P.arrange(self._findings, include_empty=self._include_empty):
            spec = group.spec
            blocks.group_header(self, t, spec.title, len(group.rows), column=spec.column,
                                tone=spec.tone, meaning=spec.meaning
                                ).pack(fill="x", pady=(t.SPACE_LG, t.SPACE_XS))
            if not group.rows:
                ctk.CTkLabel(self, text=spec.empty, font=t.font(t.SIZE_BODY),
                             text_color=t.TEXT_SECONDARY, anchor="w", justify="left",
                             wraplength=1000).pack(fill="x", padx=t.SPACE_SM)
                continue
            shown = group.rows if spec.key in self._expanded else group.rows[:P.ROWS_PER_GROUP]
            for r in shown:
                tone = P.value_tone(spec.key, r.value)
                color = t.PASS_FG if tone == "up" else t.CHECK if tone == "down" else None
                w = blocks.row(self, t, r.model, r.statement, P.value_text(spec.key, r.value),
                               tags=r.tags, on_click=lambda k=r.key: self.toggle(k), value_color=color)
                w.pack(fill="x")
                self._rows[r.key] = r
                self.row_widgets[r.key] = w
            if len(group.rows) > len(shown):
                blocks.link_button(self, t, f"Show all {len(group.rows)}",
                                   lambda k=spec.key: self.show_all(k)).pack(anchor="w", padx=t.SPACE_XS)
        if was_open in self.row_widgets:              # keep the open row open across Show all
            self.toggle(was_open)

    def _draw_detail(self, r: P.Row) -> ctk.CTkFrame:
        t = self.theme
        d = ctk.CTkFrame(self, fg_color=t.CARD, corner_radius=t.RADIUS_LG, border_width=1,
                         border_color=t.BORDER)
        for f in r.findings:
            ev = f.get("evidence") or {}
            if r.merged and ev.get("track"):
                ctk.CTkLabel(d, text=str(ev["track"]), font=t.font(t.SIZE_BODY, "bold"),
                             text_color=t.TEXT_PRIMARY, anchor="w").pack(fill="x", padx=t.SPACE_LG,
                                                                         pady=(t.SPACE_MD, 0))
            ctk.CTkLabel(d, text=str(f.get("summary") or ""), font=t.font(t.SIZE_BODY),
                         text_color=t.TEXT_PRIMARY, anchor="w", justify="left", wraplength=960
                         ).pack(fill="x", padx=t.SPACE_LG, pady=(t.SPACE_SM, 0))
            settings = ((ev.get("group") or {}).get("settings")) if f.get("analyzer") == "cut_setting" else None
            if settings:
                self._settings_table(d, settings)
        if self._on_open is not None:
            blocks.primary_button(d, t, f"Open {r.model}", lambda m=r.model: self._on_open(m)
                                  ).pack(anchor="w", padx=t.SPACE_LG, pady=t.SPACE_MD)
        else:
            ctk.CTkFrame(d, height=t.SPACE_SM, fg_color="transparent").pack()
        return d

    def _settings_table(self, parent, settings) -> None:
        t = self.theme
        grid = ctk.CTkFrame(parent, fg_color="transparent")
        grid.pack(anchor="w", padx=t.SPACE_LG, pady=(t.SPACE_SM, 0))
        for c, head in enumerate(("cut", "tracks", "in spec", "ran")):
            ctk.CTkLabel(grid, text=head, font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY,
                         anchor="w").grid(row=0, column=c, sticky="w", padx=(0, t.SPACE_XL))
        for i, s in enumerate(settings, start=1):
            pct = s.get("pass_pct")
            cells = (f"{s.get('setting'):g}" if isinstance(s.get("setting"), (int, float)) else str(s.get("setting")),
                     f"{int(s.get('n') or 0):,}", "—" if pct is None else f"{pct:.0f}%",
                     str(s.get("window") or ""))
            for c, text in enumerate(cells):
                ctk.CTkLabel(grid, text=text, font=t.mono(t.SIZE_BODY) if c < 3 else t.font(t.SIZE_CAPTION),
                             text_color=t.TEXT_PRIMARY if c < 3 else t.TEXT_SECONDARY, anchor="w"
                             ).grid(row=i, column=c, sticky="w", padx=(0, t.SPACE_XL))
```

- [ ] **Step 4: Rebuild `findings_page.py`.** Keep the module docstring's substance, `_query()` exactly as it is (its two guarded reads are the honesty rules), `reload_now`, `on_show`. Replace `build_content`, `_apply` and `_open`:

```python
    def build_content(self, parent):
        t = self.theme
        self._notices = ctk.CTkFrame(parent, fg_color="transparent")
        self._notices.pack(side="top", fill="x")
        self._list = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)
        self._view = FindingsView(self._list, t, on_open=self._open, include_empty=True)
        self._view.pack(fill="x")

    def _apply(self, data: Any) -> None:
        if isinstance(data, list):                 # back-compat: callers may still pass a plain row list
            data = {"rows": data, "errors": {}, "failed": None}
        t = self.theme
        rows = data.get("rows") or []
        errors = data.get("errors") or {}
        self._rows = rows
        for child in self._notices.winfo_children():
            child.destroy()
        if data.get("failed"):
            self.set_caption("")
            self._view.set_findings([])
            self._view.pack_forget()
            blocks.banner(self._notices, t,
                          f"Findings could not be loaded ({data['failed']}). This is an error, not an "
                          f"empty list — the log has the details.").pack(fill="x", pady=t.SPACE_SM)
            return
        if not self._view.winfo_ismapped():
            self._view.pack(fill="x")
        if data.get("errors_failed"):
            blocks.banner(self._notices, t,
                          f"Whether any model failed on the last refresh could not be checked "
                          f"({data['errors_failed']}), so this list may be missing models."
                          ).pack(fill="x", pady=(0, t.SPACE_SM))
        if errors:
            names = sorted(errors)
            shown = ", ".join(names[:10]) + (" …" if len(names) > 10 else "")
            blocks.banner(self._notices, t,
                          f"{len(names)} model(s) could not be fully worked out on the last refresh, "
                          f"so they may be missing from this list: {shown}. Open one to see what failed."
                          ).pack(fill="x", pady=(0, t.SPACE_SM))
        if not rows:
            self.set_caption("")
            blocks.banner(self._notices, t,
                          "No findings yet. They are worked out after each ingest that saves trim files; "
                          "a model with nothing worth acting on does not appear here.", tone="quiet"
                          ).pack(fill="x", pady=t.SPACE_SM)
            self._view.set_findings([])
            self._view.pack_forget()
            return
        self.set_caption(P.caption(P.arrange(rows), rows))
        self._view.set_findings(rows)

    def _open(self, model: str) -> None:
        self.app.set_model_route(model, tab="findings")
        self.app.show_page("model")
```

with imports `from laser_trim_analyzer.findings import presentation as P`, `from laser_trim_analyzer.gui.v6.widgets import blocks`, `from laser_trim_analyzer.gui.v6.widgets.findings_view import FindingsView`. Remove the old `_zone_header` call.

- [ ] **Step 5: Route to the tab** in `app.py` and `model_page.py` as described under Files. Read `model_page.py`'s tab-view code first and use its own method for selecting a tab; the Findings tab's registered name is whatever `model_page.py` uses today.

- [ ] **Step 6: Update `tests/test_findings_page.py`** to the new structure, keeping every behaviour it pins: the ranked list (now: the BIG finding's row precedes the SMALL one inside its group, and the SMALL no-rate row shows "—"); opening a model routes to `model` with `consume_model_route() == "BIG"` and `consume_model_tab() == "findings"`; the empty cache says "No findings yet"; a load failure says "This is an error, not an empty list"; `errors_failed` and model-error notices still appear. Read labels from `page._notices` and the view.

- [ ] **Step 7: Run** `pytest tests/test_findings_view.py tests/test_findings_page.py -q` → PASS. **Mutation-check:** make `toggle` never destroy the previous detail → the one-open test goes red; make `_render` ignore `ROWS_PER_GROUP` → the five-rows test goes red; restore.
- [ ] **Step 8: Gate, commit** — "feat(findings): the Findings page rebuilt as four groups, one open row at a time".

---

### Task 8: The model page's Findings tab on the same view

**Files:**
- Modify: `src/laser_trim_analyzer/gui/v6/widgets/findings_tab.py`
- Modify: `tests/test_findings_tab.py`

**Interfaces:**
- Consumes: `FindingsView(include_empty=False, on_open=None)` (Task 7), blocks (Task 2).

- [ ] **Step 1: Update the tests first** in `tests/test_findings_tab.py`: every behaviour they pin stays (crash named in the facts; "No laser trim tracks are stored"; the yardstick explanation; trim-effort facts per laser; limit tables shown only when there are 2+; recipe history; "Nothing to act on" when there are no findings) — and add:

```python
def test_the_tab_draws_findings_with_the_same_groups_as_the_page(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.findings_tab import FindingsTab
    tab = FindingsTab(tk_root, ThemeManager())
    cut = {"analyzer": "cut_setting", "model": "6607", "category": "Cut setting", "title": "t",
           "summary": "s", "systems": ["B"], "n_units": 1, "tracks_per_year": 182.0,
           "evidence": {"best": 6800.0, "current": 6900.0, "grade": "two_periods", "track": "Track A"}}
    tab.set_data({"facts": {"tracks": 10, "errors": {}}, "findings": [cut]})
    texts = _texts(tab)
    assert "Change a setting to raise yield" in texts
    assert "What changed" not in texts                  # empty groups hidden on one model's tab
    assert not any(x.startswith("Open ") for x in texts)  # already on the model
```

(define `_texts` as in `tests/test_findings_view.py` if the file lacks it.)

- [ ] **Step 2: Run and watch the new test fail.**
- [ ] **Step 3: Rebuild `findings_tab.py`:** keep `_num`, `_txt`, `_pct`, `_ANALYZER_NAMES` (add `"cut_setting": "the cut settings"` and `"pass_burden": "the multi-pass burden"` if missing) and `_facts()`'s logic. Replace the findings cards: after the facts, if there are findings draw a `FindingsView(self._body, t, on_open=None, include_empty=False)` with `set_findings(findings)`; if there are none keep today's "Nothing to act on…" line. Section headings become sentence case: "What was measured", "What to do about it", "Limit tables this model has been graded against", "Cut settings this model has been run at", "Cuts the recipe did not ask for (last year)", "Recipe history". Delete `_card` (its job is now `FindingsView`'s).
- [ ] **Step 4: Run** `pytest tests/test_findings_tab.py -q` → PASS; gate; commit — "feat(findings): the model's Findings tab uses the same view as the page".

---

### Task 9: See every page — the page renderer, and fixing what it shows

**Files:**
- Create: `scripts/render_pages.py`
- Modify: whatever pages the renders show overflowing (smallest fix per overflow; NO relayouts)

**Interfaces:**
- `python scripts/render_pages.py <copy.db> <outdir>` — refuses the production database (`_db_guard.is_production_db`), builds `V6App(config, db=DatabaseManager(copy), auto_train_on_first_run=False)` with BOTH globals injected, shows each page in `Sidebar.ITEMS` plus the model page for the model with the most findings and its Findings tab, pumps `app.update()` for up to 5 s per page so background loads land, and saves the window to `<outdir>/<page>.png` with `PIL.ImageGrab.grab(bbox=...)`. If the capture is blank or refused (macOS asks for Screen Recording permission), it says so and exits non-zero rather than saving blank images.

- [ ] **Step 1: Write `render_pages.py`** (read `V6App`, `Sidebar.ITEMS` and `page_container` first for the exact method names; mirror `scripts/refresh_findings.py` for the guard and the global injection).
- [ ] **Step 2: Run it** on `cp data/analysis.db /tmp/qa_copy.db` → `qa_output/pages/`.
- [ ] **Step 3: Look at every PNG.** For each page, check: text legible, nothing truncated or overlapping, numbers mono, one teal-filled button at most, headers sentence case. Fix overflows with the smallest change (a wraplength, a column minsize, a shorter label) — never a relayout. Re-render until clean.
- [ ] **Step 4: Gate, delete the copy, commit** — "feat(scripts): render every page for inspection; fix what bigger text overflowed".

---

### Task 10: Close out

- [ ] **Step 1:** `python scripts/run_test_gate.py` → GREEN; record the count.
- [ ] **Step 2:** `cp data/analysis.db /tmp/qa_copy.db`; `python scripts/app_qa_sweep.py /tmp/qa_copy.db` — the two known red lines (D3) may remain; any NEW failure is fixed. Add a sweep entry for the findings arrangement: every cached finding's analyzer maps to a group. `python scripts/chart_qa_render_all.py qa_output /tmp/qa_copy.db` and look at the charts again. Delete the copy.
- [ ] **Step 3:** `TRACKER.md` — tick F2 with what shipped; `BRING_TO_WORK.md` — a short section: pull, look at every page, check "the mono numbers are Plex" and "nothing is cut off", and (if Task 4's files are in) the one-line Windows check.
- [ ] **Step 4:** `python scripts/check_no_customer_values.py`; commit; push `V6` and `main`.
