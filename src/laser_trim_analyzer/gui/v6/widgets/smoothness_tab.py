"""Spec 3c — SmoothnessTab: max_smoothness_value trend + recent test list (dict-driven)."""
from typing import Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart


class SmoothnessTab(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._chart = FocusChart(self, theme=theme)
        self._chart.pack(side="top", fill="x", pady=(0, theme.SPACE_MD))
        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)
        self._rows: List[ctk.CTkFrame] = []

    def set_records(self, records: List[Dict]) -> None:
        for r in self._rows:
            r.destroy()
        self._rows.clear()
        t = self.theme
        # Q5: pair date+value together.
        pairs = sorted(((r["file_date"], r["max_smoothness_value"]) for r in records
                        if r.get("file_date") is not None and r.get("max_smoothness_value") is not None),
                       key=lambda p: p[0])
        self._chart.set_series(metric="max_smoothness_value",
                               dates=[p[0] for p in pairs], values=[p[1] for p in pairs])
        if not records:
            lbl = ctk.CTkLabel(self._list, text="No smoothness records for this model.",
                               font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY)
            lbl.pack(pady=t.SPACE_LG)
            self._rows.append(lbl)
            return
        for r in sorted(records, key=lambda r: r.get("file_date") or 0, reverse=True):
            row = ctk.CTkFrame(self._list, fg_color=t.CARD)
            row.pack(side="top", fill="x", pady=1)
            when = r["file_date"].strftime("%Y-%m-%d") if r.get("file_date") else "—"
            def _fmt(v):
                return f"{v:.4g}" if isinstance(v, (int, float)) else "—"
            txt = (f"{r.get('serial') or '—'} · {when} · "
                   f"max={_fmt(r.get('max_smoothness_value'))} · avg={_fmt(r.get('avg_smoothness_value'))}")
            ctk.CTkLabel(row, text=txt, font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_PRIMARY)\
                .pack(side="left", padx=t.SPACE_SM, pady=t.SPACE_XS)
            self._rows.append(row)
