"""Spec 3b — FlaggedCardsZone: 'Needs attention' heading + wrapping card grid + empty state."""
from datetime import datetime
from typing import Callable, List, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
from laser_trim_analyzer.ml.drift_types import ModelAlertSummary

MAX_PER_ROW = 4


class FlaggedCardsZone(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_card_click: Callable[[str, str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_card_click
        self._heading = ctk.CTkLabel(self, text="Needs attention (0)",
                                     font=theme.font(theme.SIZE_HEADING, "bold"),
                                     text_color=theme.TEXT_PRIMARY, anchor="w")
        self._heading.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        self._body = ctk.CTkFrame(self, fg_color="transparent")
        self._body.pack(side="top", fill="x")

    def set_summaries(self, summaries: List[ModelAlertSummary],
                      last_processed: Optional[datetime] = None) -> None:
        for c in list(self._body.winfo_children()):
            c.destroy()
        self._heading.configure(text=f"Needs attention ({len(summaries)})")
        t = self.theme
        if not summaries:
            when = last_processed.strftime("%Y-%m-%d") if last_processed else "—"
            ctk.CTkLabel(self._body,
                         text=f"All models within tolerance — last processed {when}.",
                         font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY, anchor="w")\
                .pack(side="top", fill="x", pady=t.SPACE_LG)
            return
        row = None
        for i, s in enumerate(summaries):
            if i % MAX_PER_ROW == 0:
                row = ctk.CTkFrame(self._body, fg_color="transparent")
                row.pack(side="top", fill="x")
            ModelAlertCard(row, summary=s, theme=t, on_click=self._cb)\
                .pack(side="left", padx=(0, t.SPACE_MD), pady=t.SPACE_SM)
