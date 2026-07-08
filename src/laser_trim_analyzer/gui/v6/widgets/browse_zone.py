"""Spec 3b — BrowseZone: search + scrollable model list (visible tier dot, last-processed date)."""
from typing import Callable, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import ModelSummary

ROW_CAP = 200  # render cap for responsiveness; cap is disclosed (Q10)


class BrowseZone(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_row_click: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_row_click
        self._models: List[ModelSummary] = []
        self._rows: List["_BrowseRow"] = []
        t = theme
        ctk.CTkLabel(self, text="All models", font=t.font(t.SIZE_HEADING, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w").pack(side="top", fill="x", pady=(0, t.SPACE_XS))
        # Row anatomy, spelled out (live-walk finding, 2026-07-08: unexplained
        # colored dots + an unlabeled date column read as decoration).
        ctk.CTkLabel(self, text=("Dot = drift status: red out-of-control · orange drift · "
                                 "yellow warning · gray stable/untrained. Date = last processed. "
                                 "'Active' scope = models with recent data or pinned in "
                                 "Settings → Active Models."),
                     font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY,
                     anchor="w", justify="left", wraplength=1200)\
            .pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        # NOTE: no textvariable — CTkEntry silently drops placeholder_text when
        # a textvariable is attached (the 'mystery empty box' finding). Filter
        # reacts on KeyRelease instead.
        self._search_entry = ctk.CTkEntry(self, placeholder_text="Type to filter models…",
                     font=t.font(t.SIZE_BODY), fg_color=t.CARD, border_color=t.BORDER,
                     text_color=t.TEXT_PRIMARY)
        self._search_entry.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        self._search_entry.bind("<KeyRelease>", lambda e: self._render())
        self._cap_label = ctk.CTkLabel(self, text="", font=t.font(t.SIZE_CAPTION),
                                       text_color=t.TEXT_SECONDARY, anchor="w")
        self._cap_label.pack(side="top", fill="x")
        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)

    def set_models(self, models: List[ModelSummary]) -> None:
        self._models = list(models)
        self._render()

    def set_filter(self, text: str) -> None:
        self._search_entry.delete(0, "end")
        if text:
            self._search_entry.insert(0, text)
        self._render()

    def _render(self) -> None:
        for r in self._rows:
            r.destroy()
        self._rows.clear()
        flt = self._search_entry.get().strip().lower()
        matches = [m for m in self._models if not flt or flt in m.model.lower()]
        for m in matches[:ROW_CAP]:
            row = _BrowseRow(self._list, summary=m, theme=self.theme, on_click=self._cb)
            row.pack(side="top", fill="x", pady=1)
            self._rows.append(row)
        if len(matches) > ROW_CAP:
            self._cap_label.configure(
                text=f"Showing {ROW_CAP} of {len(matches)} — narrow with search.")
        else:
            self._cap_label.configure(text="")


class _BrowseRow(ctk.CTkFrame):
    def __init__(self, master, summary: ModelSummary, theme: ThemeManager,
                 on_click: Callable[[str], None]):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=theme.RADIUS_SM)
        self.theme = theme
        self.summary = summary
        self._cb = on_click
        t = theme
        dot = ctk.CTkFrame(self, width=12, height=12, corner_radius=6,
                           fg_color=t.tier_dot_color(summary.tier))   # FIX I4: visible STABLE dot
        dot.pack(side="left", padx=(t.SPACE_SM, t.SPACE_XS))
        dot.pack_propagate(False)
        name = ctk.CTkLabel(self, text=summary.model, font=t.font(t.SIZE_BODY),
                            text_color=t.TEXT_PRIMARY, anchor="w")
        name.pack(side="left", fill="x", expand=True, padx=(t.SPACE_XS, t.SPACE_SM))
        date_txt = summary.last_processed.strftime("%Y-%m-%d") if summary.last_processed else "—"
        date = ctk.CTkLabel(self, text=date_txt, font=t.font(t.SIZE_CAPTION),
                            text_color=t.TEXT_SECONDARY)
        date.pack(side="right", padx=t.SPACE_SM)
        for w in (self, dot, name, date):
            w.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self):
        self._cb(self.summary.model)
