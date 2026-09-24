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
