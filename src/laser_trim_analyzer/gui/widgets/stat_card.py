"""
StatCard widget for displaying key metrics in the Laser Trim Analyzer GUI.
"""

import tkinter as tk
from tkinter import ttk, font
from typing import Optional, Union


class StatCard(ttk.Frame):
    """
    A card widget that displays a metric with title, value, and optional trend indicator.
    Perfect for showing key QA metrics like pass rate, sigma values, etc.
    """

    def __init__(
            self,
            parent,
            title: str = "Metric",
            value: Union[str, float, int] = "0",
            unit: str = "",
            trend: Optional[str] = None,  # "up", "down", or None
            color_scheme: str = "default",  # "default", "success", "warning", "danger"
            **kwargs
    ):
        super().__init__(parent, **kwargs)

        self.title = title
        self.value = value
        self.unit = unit
        self.trend = trend
        self.color_scheme = color_scheme

        # Define color schemes
        self.colors = {
            "default": {"bg": "#ffffff", "fg": "#2c3e50", "accent": "#3498db"},
            "success": {"bg": "#ffffff", "fg": "#27ae60", "accent": "#2ecc71"},
            "warning": {"bg": "#ffffff", "fg": "#f39c12", "accent": "#f1c40f"},
            "danger": {"bg": "#ffffff", "fg": "#e74c3c", "accent": "#c0392b"}
        }

        self._setup_ui()

    def _setup_ui(self):
        """Set up the card UI."""
        # Configure the frame (remove unsupported parameters)
        # CustomTkinter doesn't support relief, borderwidth, padding

        # Get colors for the scheme
        scheme = self.colors.get(self.color_scheme, self.colors["default"])

        # Title label
        title_font = font.Font(family="Segoe UI", size=10, weight="normal")
        self.title_label = ttk.Label(
            self,
            text=self.title,
            font=title_font,
            foreground="#7f8c8d"
        )
        self.title_label.pack(anchor="w")

        # Value container
        value_frame = ttk.Frame(self)
        value_frame.pack(fill="x", pady=(5, 0))

        # Value label
        value_font = font.Font(family="Segoe UI", size=24, weight="bold")
        self.value_label = ttk.Label(
            value_frame,
            text=str(self.value),
            font=value_font,
            foreground=scheme["fg"]
        )
        self.value_label.pack(side="left")

        # Unit label
        if self.unit:
            unit_font = font.Font(family="Segoe UI", size=12)
            self.unit_label = ttk.Label(
                value_frame,
                text=self.unit,
                font=unit_font,
                foreground="#95a5a6"
            )
            self.unit_label.pack(side="left", padx=(5, 0))

        # Trend indicator
        if self.trend:
            trend_text = "↑" if self.trend == "up" else "↓"
            trend_color = scheme["accent"] if self.trend == "up" else scheme["fg"]

            trend_font = font.Font(family="Segoe UI", size=16)
            self.trend_label = ttk.Label(
                value_frame,
                text=trend_text,
                font=trend_font,
                foreground=trend_color
            )
            self.trend_label.pack(side="right", padx=(10, 0))

    def update_value(self, value: Union[str, float, int], trend: Optional[str] = None):
        """Update the displayed value and optionally the trend."""
        self.value = value
        self.value_label.configure(text=str(value))

        if trend is not None:
            self.trend = trend
            if hasattr(self, 'trend_label'):
                self.trend_label.configure(text="↑" if trend == "up" else "↓")

    def set_color_scheme(self, scheme: str):
        """Change the color scheme of the card."""
        if scheme in self.colors:
            self.color_scheme = scheme
            colors = self.colors[scheme]
            self.value_label.configure(foreground=colors["fg"])
            if hasattr(self, 'trend_label') and self.trend:
                trend_color = colors["accent"] if self.trend == "up" else colors["fg"]
                self.trend_label.configure(foreground=trend_color)