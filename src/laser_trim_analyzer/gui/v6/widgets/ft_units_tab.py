"""Model page — FtUnitsTab: final-test records for the model.

Work finding #3 (2026-07-10): "no way to view final test units." The Units
tab shows trim records; this one lists what happened at the LAST station —
result, date, and whether the record linked back to a trim unit.
"""
from typing import Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

_COLUMNS = [("serial", "Serial"), ("file_date", "Test date"), ("result", "Result"),
            ("linked", "Linked to trim"), ("match", "Match %")]


class FtUnitsTab(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._rows: List[ctk.CTkFrame] = []
        header = ctk.CTkFrame(self, fg_color=theme.CARD)
        header.pack(side="top", fill="x", pady=(0, theme.SPACE_XS))
        for _key, lbl in _COLUMNS:
            ctk.CTkLabel(header, text=lbl, font=theme.font(theme.SIZE_CAPTION, "bold"),
                         text_color=theme.TEXT_SECONDARY)\
                .pack(side="left", expand=True, fill="x",
                      padx=theme.SPACE_SM, pady=theme.SPACE_XS)
        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)
        self._cap = ctk.CTkLabel(self, text="", font=theme.font(theme.SIZE_CAPTION),
                                 text_color=theme.TEXT_SECONDARY, anchor="w")
        self._cap.pack(side="top", fill="x")

    def set_units(self, units: List[Dict]) -> None:
        for r in self._rows:
            try:
                r.destroy()
            except Exception:
                pass
        self._rows = []
        t = self.theme
        if not units:
            lbl = ctk.CTkLabel(self._list, text="No final-test records for this model "
                               "in the selected window.",
                               font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY)
            lbl.pack(side="top", fill="x", pady=t.SPACE_MD)
            self._rows.append(lbl)
            self._cap.configure(text="")
            return
        color = {"FAIL": t.TIER_OOC, "PASS": t.TEXT_PRIMARY, "WARNING": t.TIER_WARNING}
        for u in units:
            row = ctk.CTkFrame(self._list, fg_color=t.SURFACE)
            vals = [str(u.get("serial") or "—"),
                    (u["file_date"].strftime("%Y-%m-%d") if u.get("file_date") else "—"),
                    str(u.get("result") or "—"),
                    "yes" if u.get("linked") else "no",
                    (f"{u['match']}%" if u.get("match") is not None else "—")]
            for (key, _), v in zip(_COLUMNS, vals):
                c = color.get(v, t.TEXT_PRIMARY) if key == "result" else t.TEXT_PRIMARY
                ctk.CTkLabel(row, text=v, font=t.font(t.SIZE_BODY), text_color=c)\
                    .pack(side="left", expand=True, fill="x",
                          padx=t.SPACE_SM, pady=t.SPACE_XS)
            row.pack(side="top", fill="x", pady=1)
            self._rows.append(row)
        n_fail = sum(1 for u in units if u.get("result") == "FAIL")
        self._cap.configure(
            text=f"{len(units)} final-test record(s) · {n_fail} failed"
                 + ("   (showing newest 500)" if len(units) == 500 else ""))
