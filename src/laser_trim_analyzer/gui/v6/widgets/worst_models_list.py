"""Dashboard — WorstModelsList: ranked, clickable model rows (model/units/trim%/FT%)."""
from typing import Callable, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

_COLS = [("model", "Model"), ("units", "Units"), ("trim_rate", "Trim %"), ("ft_rate", "FT %")]


def _fmt(key, value) -> str:
    if value is None:
        return "—"
    if key in ("trim_rate", "ft_rate"):
        return f"{value:.0f}%"
    return str(value)


class WorstModelsList(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_row_click: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_row_click
        self._rows: List["_WorstRow"] = []
        t = theme
        ctk.CTkLabel(self, text="Lowest-yield models", font=t.font(t.SIZE_HEADING, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w").pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        header = ctk.CTkFrame(self, fg_color=t.CARD)
        header.pack(side="top", fill="x")
        header.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="wm")
        for i, (_key, label) in enumerate(_COLS):
            ctk.CTkLabel(header, text=label, font=t.font(t.SIZE_CAPTION, "bold"),
                         text_color=t.TEXT_SECONDARY, anchor="w")\
                .grid(row=0, column=i, sticky="ew", padx=t.SPACE_SM, pady=t.SPACE_XS)
        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)
        self._cap = ctk.CTkLabel(self, text="", font=t.font(t.SIZE_CAPTION),
                                 text_color=t.TEXT_SECONDARY, anchor="w")
        self._cap.pack(side="top", fill="x")
        self._empty = ctk.CTkLabel(self, text="", font=t.font(t.SIZE_BODY),
                                   text_color=t.TEXT_SECONDARY, anchor="w")

    def set_rows(self, rows: List[dict], total: int) -> None:
        for r in self._rows:
            r.destroy()
        self._rows.clear()
        t = self.theme
        if not rows:
            self._cap.configure(text="")
            self._empty.configure(text="No models with enough recent data to rank.")
            self._empty.pack(side="top", fill="x", pady=t.SPACE_MD)
            return
        self._empty.pack_forget()
        for row in rows:
            r = _WorstRow(self._list, row=row, theme=t, on_click=self._cb)
            r.pack(side="top", fill="x", pady=1)
            self._rows.append(r)
        self._cap.configure(text=f"Showing {len(rows)} of {total} (min 5 units, worst first)."
                            if total > len(rows) else f"{total} models (min 5 units, worst first).")


class _WorstRow(ctk.CTkFrame):
    def __init__(self, master, row: dict, theme: ThemeManager, on_click: Callable[[str], None]):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=theme.RADIUS_SM)
        self.row = row
        self._cb = on_click
        self.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="wm")
        for i, (key, _label) in enumerate(_COLS):
            lbl = ctk.CTkLabel(self, text=_fmt(key, row.get(key)), font=theme.font(theme.SIZE_BODY),
                               text_color=theme.TEXT_PRIMARY, anchor="w")
            lbl.grid(row=0, column=i, sticky="ew", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
            lbl.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self):
        self._cb(self.row["model"])
