"""Spec 3a — ThemeManager: the single source of V6 visual tokens.

Foundations §2.3. Frozen dataclass; every widget/page reads a shared instance.
Helpers: tier_color (bg, fg) pair, tier_dot_color (visible STABLE), font() (real
fallback to an available family).
"""
import tkinter
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import customtkinter as ctk

from laser_trim_analyzer.ml.drift_types import DriftTier


@dataclass
class ThemeManager:
    # GRAPHITE (James, 2026-10-02: "i like dark mode" -- picked from three dark options): a
    # neutral near-black, white text, ONE blue accent; no navy, no teal. The approved mockup's
    # colours are bg / card / border / text / secondary / accent / chart highlight / history
    # bars / worse / better; every other token below is derived from them and measured by
    # tests/test_theme_contrast.py.
    # Surfaces
    BG: str = "#0c0c0e"; SURFACE: str = "#111114"; CARD: str = "#151518"; ELEVATED: str = "#1c1c20"
    # The navigation bar (was the sidebar; the names are kept for the widgets that read them)
    SIDEBAR_BG: str = "#0c0c0e"; SIDEBAR_ACTIVE: str = "#151518"; SIDEBAR_STRIPE: str = "#3b82f6"
    # Accent -- blue means "act here", on ONE primary button per screen. Text on it is dark:
    # white on #3b82f6 is 3.7:1, below the 4.5 minimum.
    ACCENT: str = "#3b82f6"; ACCENT_HOVER: str = "#60a5fa"; ACCENT_PRESSED: str = "#2563eb"
    ACCENT_TINT: str = "#0f1a2e"
    # A selected tab or segment. CTkSegmentedButton (and so every CTkTabview) draws ALL its
    # segments' text in ONE colour, so the selected fill must carry the same light text as the
    # unselected CARD segments: a deeper blue, 5.7:1 with TEXT_PRIMARY and 2.7:1 from CARD; the
    # hover steps darker still (7.5:1 text, 2.1:1 from CARD). Measured with
    # tests/test_theme_contrast.py's own contrast().
    SEGMENT_SELECTED: str = "#1d4ed8"; SEGMENT_SELECTED_HOVER: str = "#1e40af"
    # Text. DISABLED is used for real information too (chart ticks, the volume axis), so it
    # clears 4.5:1 on every surface like the rest.
    TEXT_PRIMARY: str = "#ededed"; TEXT_SECONDARY: str = "#8b8b93"
    TEXT_DISABLED: str = "#85858e"; TEXT_INVERSE: str = "#0c0c0e"   # INVERSE = text on the accent
    # Borders
    DIVIDER: str = "#1d1d21"; BORDER: str = "#26262b"
    # "Check this" -- the worse/fail red, on its own tint
    CHECK: str = "#f87171"; CHECK_TINT: str = "#2a1414"
    # Verdicts -- always drawn WITH their word, never colour alone. Better = green, worse = red.
    PASS_FG: str = "#4ade80"; PASS_BG: str = "#0f2417"
    FAIL_FG: str = "#f87171"; FAIL_BG: str = "#2a1414"
    NEUTRAL_FG: str = "#b4b4bc"; NEUTRAL_BG: str = "#1c1c20"
    WATCH_FG: str = "#fbbf24"; WATCH_BG: str = "#2a2210"
    # Tiers: amber, orange, red, each on its own dark tint
    TIER_STABLE: str = "#151518"
    TIER_WARNING_BG: str = "#2a2210"; TIER_WARNING: str = "#fbbf24"
    TIER_DRIFT_BG: str = "#2a1a0e"; TIER_DRIFT: str = "#fb923c"
    TIER_OOC_BG: str = "#2a1414"; TIER_OOC: str = "#f87171"
    # Charts. HIGHLIGHT is the data line a chart is about, HISTORY its past bars (the approved
    # mockup's monthly pass bars). SERIES_* are keyed by the code's system letter -- the UI still
    # says "Laser 2 (DLTS)" -- and keep 30 degrees of hue from every colour that means something
    # (the blue accent, pass green, fail red, watch amber), AND at least 60 degrees from each other:
    # violet, fuchsia and pink sat 37 degrees apart, and James could not tell them apart on the
    # Overview's chart (2026-10-04: "the colors are too similare"). Laser 1 lime, laser 2 teal,
    # laser 3 purple -- warm to cool in the shop's order; the company line keeps CHART_HIGHLIGHT.
    CHART_REFERENCE: str = "#8b8b93"
    CHART_HIGHLIGHT: str = "#60a5fa"; CHART_HISTORY: str = "#2a3a58"
    SERIES_A: str = "#2dd4bf"; SERIES_B: str = "#a3e635"; SERIES_C: str = "#c084fc"
    CHART_FONT_SMALL: float = 8.0; CHART_FONT: float = 9.0; CHART_FONT_LARGE: float = 10.0
    # Typography. On Windows (GDI) Plex Medium is its OWN family, not a weight of Plex Sans,
    # so "bold" is mapped onto it in font()/mono() when it is available.
    FONT_FAMILY: Tuple[str, ...] = ("IBM Plex Sans", "Segoe UI", "system-ui")
    # Two spellings: the bundled file's real legacy family (name ID 1), read with fontTools,
    # is the abbreviated "IBM Plex Sans Medm" -- GDI truncates it, Plex Mono Medium is not
    # truncated the same way. "IBM Plex Sans Medium" is kept first for anything (tests, a
    # future re-export) that reports the unabbreviated name; _pick() takes whichever this
    # machine's Tk actually has. See tests/test_font_loader.py.
    FONT_FAMILY_MEDIUM: Tuple[str, ...] = ("IBM Plex Sans Medium", "IBM Plex Sans Medm")
    MONO_FAMILY: Tuple[str, ...] = ("IBM Plex Mono", "Cascadia Mono", "Consolas", "Menlo", "Courier")
    MONO_FAMILY_MEDIUM: Tuple[str, ...] = ("IBM Plex Mono Medium",)
    # Titles -- page titles, model names, the app's name -- in Marcellus (James, 2026-10-04, from
    # ten fonts drawn on his own header: "6"). Bundled and loaded by font_loader; where Tk cannot
    # see it, title() falls back to the Sans's bold.
    TITLE_FAMILY: Tuple[str, ...] = ("Marcellus",)
    SIZE_CAPTION: int = 12; SIZE_BODY: int = 14; SIZE_HEADING: int = 17
    SIZE_TITLE: int = 22; SIZE_DISPLAY: int = 30; SIZE_READOUT: int = 20
    # Spacing / radii (unchanged)
    SPACE_XS: int = 4; SPACE_SM: int = 8; SPACE_MD: int = 12
    SPACE_LG: int = 16; SPACE_XL: int = 24; SPACE_2XL: int = 32
    RADIUS_SM: int = 4; RADIUS_MD: int = 6; RADIUS_LG: int = 8

    resolved_family: str = field(default="", init=False)
    resolved_medium: Optional[str] = field(default=None, init=False)
    resolved_mono: str = field(default="", init=False)
    resolved_mono_medium: Optional[str] = field(default=None, init=False)
    resolved_title: Optional[str] = field(default=None, init=False)
    # One CTkFont per (family, size, weight) — see font(). Excluded from
    # repr/eq: it is a performance cache, not part of the theme's identity.
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
        object.__setattr__(self, "resolved_title", self._pick(self.TITLE_FAMILY, available, None))

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

    # ---- Helpers ----
    def font(self, size: int, weight: str = "normal") -> ctk.CTkFont:
        """One shared CTkFont per (family, size, weight).

        This used to mint a NEW CTkFont on every call, and it has ~144 call
        sites — so a 50-row UnitsTab alone built 250+ of them. A CTkFont is a
        `tkinter.font.Font`: constructing one is a `font create` round trip
        into Tcl plus a `font actual` to read the resolved family back, and
        dropping one is a `font delete`. Rebuilding a page therefore paid a
        few thousand Tcl round trips for maybe ten distinct fonts. Sharing
        font objects between widgets is CustomTkinter's documented usage.

        The cache is keyed on the Tk root as well, because a `Font` belongs to
        the interpreter that created it: hand a font from a destroyed root to
        a new widget and Tk raises. In the app that never happens (the theme
        dies with its V6App), but tests build and tear down roots, so the
        cache is dropped whenever the default root changes.

        Do not `configure()` a font you get from here — it is shared, and the
        change would land on every widget using that size. Nothing does today.
        """
        if weight == "bold" and self.resolved_medium:
            return self._shared_font(self.resolved_medium, size, "normal")
        return self._shared_font(self.resolved_family, size, weight)

    def mono(self, size: int, weight: str = "normal") -> ctk.CTkFont:
        """The same shared font, in the mono family -- every NUMBER in the app uses this."""
        if weight == "bold" and self.resolved_mono_medium:
            return self._shared_font(self.resolved_mono_medium, size, "normal")
        return self._shared_font(self.resolved_mono, size, weight)

    def title(self, size: int) -> ctk.CTkFont:
        """The title face (Marcellus) at `size` -- or, where Tk cannot see it, the Sans's bold."""
        if self.resolved_title:
            return self._shared_font(self.resolved_title, size, "normal")
        return self.font(size, "bold")

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

    def tier_color(self, tier: DriftTier) -> Tuple[str, str]:
        """(background, foreground) for a tier. STABLE blends into SURFACE."""
        return {
            DriftTier.STABLE: (self.TIER_STABLE, self.TEXT_PRIMARY),
            DriftTier.WARNING: (self.TIER_WARNING_BG, self.TIER_WARNING),
            DriftTier.DRIFT: (self.TIER_DRIFT_BG, self.TIER_DRIFT),
            DriftTier.OUT_OF_CONTROL: (self.TIER_OOC_BG, self.TIER_OOC),
        }.get(tier, (self.SURFACE, self.TEXT_PRIMARY))

    @staticmethod
    def fmt_measure(v, sig: int = 4) -> str:
        """Human-readable measurement: thousands separators instead of e+04.
        A QA reader sees 21,270 Ω on the chart axis but '2.127e+04' in the
        table read as noise (live-walk finding, 2026-07-08). Scientific
        notation only survives for genuinely extreme magnitudes."""
        if v is None:
            return "—"
        try:
            av = abs(float(v))
        except (TypeError, ValueError):
            return str(v)
        if av >= 1e7 or (av != 0 and av < 1e-4):
            return f"{v:.{sig}g}"
        if av >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{sig}g}"

    def tier_dot_color(self, tier: DriftTier) -> str:
        """Visible dot color. STABLE → muted gray (NOT SURFACE, else invisible)."""
        if tier == DriftTier.STABLE:
            return self.TEXT_DISABLED
        return self.tier_color(tier)[1]
