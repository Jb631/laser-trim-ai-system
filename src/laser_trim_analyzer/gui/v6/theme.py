"""Spec 3a — ThemeManager: the single source of V6 visual tokens.

Foundations §2.3. Frozen dataclass; every widget/page reads a shared instance.
Helpers: tier_color (bg, fg) pair, tier_dot_color (visible STABLE), font() (real
fallback to an available family).
"""
from dataclasses import dataclass, field
from typing import Tuple

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
