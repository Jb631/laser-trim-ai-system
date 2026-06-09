"""Spec 3d — SensitivitySlider: 4-stop preset picker (loose/standard/tight/strict)."""
from typing import Callable

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

_PRESETS = ["loose", "standard", "tight", "strict"]


class SensitivitySlider(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, initial: str, on_change: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._on_change = on_change
        self._seg = ctk.CTkSegmentedButton(self, values=_PRESETS, command=on_change,
                                           fg_color=theme.CARD, selected_color=theme.ACCENT,
                                           selected_hover_color=theme.ACCENT_HOVER,
                                           unselected_color=theme.CARD, unselected_hover_color=theme.ELEVATED,
                                           text_color=theme.TEXT_PRIMARY, corner_radius=theme.RADIUS_SM)
        self._seg.pack(side="top", fill="x")
        self._seg.set(initial if initial in _PRESETS else "standard")

    def value(self) -> str:
        return self._seg.get()

    def set_value(self, preset: str) -> None:
        if preset in _PRESETS:
            self._seg.set(preset)
