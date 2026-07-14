"""Spec 3b — FlaggedCardsZone: 'Needs attention' heading + wrapping card grid + empty state."""
from datetime import datetime
from typing import Callable, List, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
from laser_trim_analyzer.ml.drift_types import ModelAlertSummary

MAX_PER_ROW = 4


# Bounded, scrollable region so many cards don't overflow the page.
# 290 (was 470; 2026-07-13 live report "cant scroll down"): the zone headers
# added above both regions squeezed the browse list to a ~2-row sliver on a
# laptop display. Two card rows stay visible; the rest scroll inside.
CARDS_HEIGHT = 290


class FlaggedCardsZone(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_card_click: Callable[[str, str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_card_click
        self._rendered: List[ctk.CTkBaseClass] = []
        self._heading = ctk.CTkLabel(self, text="Needs attention (0)",
                                     font=theme.font(theme.SIZE_HEADING, "bold"),
                                     text_color=theme.TEXT_PRIMARY, anchor="w")
        self._heading.pack(side="top", fill="x", pady=(0, theme.SPACE_XS))
        # Plain-language key for the card numbers. The Model page had this
        # gloss but the cards themselves didn't — σ appeared here unexplained
        # (user finding, 2026-07-08).
        ctk.CTkLabel(self,
                     text=("σ = how far the LAST LOT's median sits from the model's baseline "
                           "of historical lot medians. One verdict per lot — a big lot can't "
                           "ring the alarm harder than a small one. 'Out Of Control' = past "
                           "the alert limit for your sensitivity preset. Drift watch, not a spec."),
                     font=theme.font(theme.SIZE_CAPTION), text_color=theme.TEXT_SECONDARY,
                     anchor="w", justify="left", wraplength=1200)\
            .pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        # Scrollable, height-bounded card area. Before this the cards rendered into a
        # plain frame with no scroll, so 30-50 flagged cards overflowed the page and
        # pushed the browse list off-screen with no way to reach them.
        self._body = ctk.CTkScrollableFrame(self, fg_color="transparent", height=CARDS_HEIGHT)
        self._body.pack(side="top", fill="x")

    def set_summaries(self, summaries: List[ModelAlertSummary],
                      last_processed: Optional[datetime] = None) -> None:
        # Track rendered widgets explicitly — winfo_children() on a CTkScrollableFrame
        # returns its internal canvas/scrollbar, not the cards.
        for w in self._rendered:
            try:
                w.destroy()
            except Exception:
                pass
        self._rendered = []
        self._heading.configure(text=f"Needs attention ({len(summaries)})")
        t = self.theme
        if not summaries:
            when = last_processed.strftime("%Y-%m-%d") if last_processed else "—"
            lbl = ctk.CTkLabel(self._body,
                               text=f"All models within tolerance — last processed {when}.",
                               font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY, anchor="w")
            lbl.pack(side="top", fill="x", pady=t.SPACE_LG)
            self._rendered.append(lbl)
            return
        row = None
        for i, s in enumerate(summaries):
            if i % MAX_PER_ROW == 0:
                row = ctk.CTkFrame(self._body, fg_color="transparent")
                row.pack(side="top", fill="x")
                self._rendered.append(row)
            ModelAlertCard(row, summary=s, theme=t, on_click=self._cb)\
                .pack(side="left", padx=(0, t.SPACE_MD), pady=t.SPACE_SM)
