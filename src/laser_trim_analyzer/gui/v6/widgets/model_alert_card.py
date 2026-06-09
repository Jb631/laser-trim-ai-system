"""Spec 3b — ModelAlertCard: one flagged-model summary card."""
from typing import Callable

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import AlertType, ModelAlertSummary, metric_label

CARD_WIDTH = 250
CARD_HEIGHT = 132


class ModelAlertCard(ctk.CTkFrame):
    def __init__(self, master, summary: ModelAlertSummary, theme: ThemeManager,
                 on_click: Callable[[str, str], None], **kwargs):
        bg, fg = theme.tier_color(summary.tier)
        super().__init__(master, width=CARD_WIDTH, height=CARD_HEIGHT, fg_color=bg,
                         corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme
        self.summary = summary
        self._cb = on_click
        self._fg = fg
        self.pack_propagate(False)
        self._build()
        self._bind_recursive(self)

    def _build(self):
        t = self.theme
        s = self.summary
        ctk.CTkLabel(self, text=s.model, font=t.font(t.SIZE_TITLE, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w")\
            .pack(side="top", fill="x", padx=t.SPACE_MD, pady=(t.SPACE_MD, 0))
        badge = "Step change" if s.alert_type == AlertType.STEP_CHANGE else "Slow drift"
        ctk.CTkLabel(self, text=f"{badge} · {metric_label(s.worst_metric)}",
                     font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY, anchor="w")\
            .pack(side="top", fill="x", padx=t.SPACE_MD)
        ctk.CTkLabel(self, text=f"{s.magnitude:+.1f}σ", font=t.font(t.SIZE_DISPLAY, "bold"),
                     text_color=self._fg, anchor="w")\
            .pack(side="top", fill="x", padx=t.SPACE_MD)
        # Q6: say what the σ is measured against.
        ctk.CTkLabel(self, text=f"beyond {s.tier.name.replace('_', ' ').title()} limit",
                     font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY, anchor="w")\
            .pack(side="top", fill="x", padx=t.SPACE_MD, pady=(0, t.SPACE_MD))

    def _bind_recursive(self, w):
        w.bind("<Button-1>", lambda e: self._on_click())
        for c in w.winfo_children():
            self._bind_recursive(c)

    def _on_click(self):
        # Deep-link with the triggering metric as the focus.
        self._cb(self.summary.model, self.summary.worst_metric)
