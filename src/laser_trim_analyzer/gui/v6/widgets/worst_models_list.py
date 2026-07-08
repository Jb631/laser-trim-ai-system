"""Dashboard — WorstModelsList: ranked, clickable model rows (model/units/trim%/FT%)."""
from typing import Callable, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

# Gap = Trim% − FT%. Strongly NEGATIVE = units failing trim but passing final
# test — the overkill pattern (trim thresholds or specs rejecting good product).
# Strongly POSITIVE = passing trim but failing FT — escapes (worse). Ported
# from V5 Quality Health's ranked table (2026-07-07, feature restoration).
_COLS = [("model", "Model"), ("units", "Units"), ("trim_rate", "Trim %"),
         ("ft_rate", "FT %"), ("gap", "Gap (pts)")]


def _fmt(key, value) -> str:
    if value is None:
        return "—"
    if key in ("trim_rate", "ft_rate"):
        return f"{value:.0f}%"
    if key == "gap":
        return f"{value:+.0f}"
    return str(value)


class WorstModelsList(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_row_click: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_row_click
        self._rows: List["_WorstRow"] = []
        t = theme
        ctk.CTkLabel(self, text="Lowest-yield models", font=t.font(t.SIZE_HEADING, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w").pack(side="top", fill="x", pady=(0, t.SPACE_XS))
        # Column key (live-walk finding, 2026-07-08: 'Gap -62' meant nothing
        # cold). Same treatment as the sigma gloss: one plain-language line.
        ctk.CTkLabel(self, text=("Trim % / FT % = linearity yield in the window. "
                                 "Gap = Trim − FT in points; negative = grading worse at trim "
                                 "than at final test (overkill pattern). Models with ≥5 units."),
                     font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY,
                     anchor="w", justify="left", wraplength=1200)\
            .pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        header = ctk.CTkFrame(self, fg_color=t.CARD)
        header.pack(side="top", fill="x")
        header.grid_columnconfigure(tuple(range(len(_COLS))), weight=1, uniform="wm")
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
        # Gap derived here so the query stays untouched: Trim% − FT%.
        gap = None
        if row.get("trim_rate") is not None and row.get("ft_rate") is not None:
            gap = row["trim_rate"] - row["ft_rate"]
        row = {**row, "gap": gap}
        self.grid_columnconfigure(tuple(range(len(_COLS))), weight=1, uniform="wm")
        for i, (key, _label) in enumerate(_COLS):
            color = theme.TEXT_PRIMARY
            if key == "gap" and gap is not None and abs(gap) >= 15:
                # Big divergence between stations deserves the eye:
                # negative = overkill (rejecting good product), positive = escapes.
                color = theme.TIER_WARNING if gap < 0 else theme.TIER_OOC
            lbl = ctk.CTkLabel(self, text=_fmt(key, row.get(key)), font=theme.font(theme.SIZE_BODY),
                               text_color=color, anchor="w")
            lbl.grid(row=0, column=i, sticky="ew", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
            lbl.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self):
        self._cb(self.row["model"])
