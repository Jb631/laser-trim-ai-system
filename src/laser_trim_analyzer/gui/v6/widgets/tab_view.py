"""Spec 3c — ThemedTabView: CTkTabview with V6 theme tokens."""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class ThemedTabView(ctk.CTkTabview):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        # The selected tab is SEGMENT_SELECTED, not ACCENT: CTk draws every tab's text in the one
        # text_color, and TEXT_PRIMARY on ACCENT measured 1.66:1 (theme.py says why this teal).
        super().__init__(master, fg_color=theme.SURFACE, segmented_button_fg_color=theme.CARD,
                         segmented_button_selected_color=theme.SEGMENT_SELECTED,
                         segmented_button_selected_hover_color=theme.SEGMENT_SELECTED_HOVER,
                         segmented_button_unselected_color=theme.CARD,
                         segmented_button_unselected_hover_color=theme.ELEVATED,
                         text_color=theme.TEXT_PRIMARY, corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme
