"""Dashboard — PrioritiesPanel: 'this week's priorities', models ranked by the
money leaking at final test. Clickable rows route to the Model page for the
reason and evidence (mirrors WorstModelsList's drill-down)."""
from typing import Callable, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

_COLS = [("model", "Model"), ("dollar_impact", "$ lost / window"),
         ("ft_fails", "FT fails"), ("ft_fail_rate", "FT fail %"),
         ("price", "Unit price")]


def _fmt(key, v) -> str:
    if v is None:
        return "—"
    if key in ("dollar_impact", "price"):
        return f"${v:,.0f}"
    if key == "ft_fail_rate":
        return f"{v:.0f}%"
    return str(v)


class PrioritiesPanel(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager,
                 on_row_click: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_row_click
        self._rows: List["_PriorityRow"] = []
        t = theme
        ctk.CTkLabel(self, text="This week's priorities — money leaking at final test",
                     font=t.font(t.SIZE_HEADING, "bold"), text_color=t.TEXT_PRIMARY,
                     anchor="w").pack(side="top", fill="x", pady=(0, t.SPACE_XS))
        ctk.CTkLabel(self, text=(
            "$ lost = final-test failures × unit price × cost ratio, in the window — final "
            "test is the most expensive place to lose a unit. Models with no loaded price "
            "show counts only (add prices in Settings → Pricing). Click a model for the "
            "reason and evidence."),
            font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY, anchor="w",
            justify="left", wraplength=1200).pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        header = ctk.CTkFrame(self, fg_color=t.CARD)
        header.pack(side="top", fill="x")
        header.grid_columnconfigure(tuple(range(len(_COLS))), weight=1, uniform="pp")
        for i, (_key, label) in enumerate(_COLS):
            ctk.CTkLabel(header, text=label, font=t.font(t.SIZE_CAPTION, "bold"),
                         text_color=t.TEXT_SECONDARY, anchor="w")\
                .grid(row=0, column=i, sticky="ew", padx=t.SPACE_SM, pady=t.SPACE_XS)
        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent", height=260)
        self._list.pack(side="top", fill="both", expand=True)
        self._cap = ctk.CTkLabel(self, text="", font=t.font(t.SIZE_CAPTION),
                                 text_color=t.TEXT_SECONDARY, anchor="w")
        self._cap.pack(side="top", fill="x")

    def set_rows(self, rows: List[dict]) -> None:
        for r in self._rows:
            try:
                r.destroy()
            except Exception:
                pass
        self._rows = []
        for d in rows or []:
            row = _PriorityRow(self._list, d, self.theme, self._cb)
            row.pack(side="top", fill="x", pady=1)
            self._rows.append(row)
        if not rows:
            self._cap.configure(text="No final-test failures in this window.")
            return
        priced = sum(1 for r in rows if r.get("dollar_impact") is not None)
        total = sum(r["dollar_impact"] for r in rows if r.get("dollar_impact"))
        self._cap.configure(
            text=f"{len(rows)} model(s) · {priced} priced · ${total:,.0f} at risk in window"
                 + ("" if priced == len(rows)
                    else "  ·  price the rest in Settings → Pricing to rank them"))


class _PriorityRow(ctk.CTkFrame):
    def __init__(self, master, data: dict, theme: ThemeManager, on_click):
        super().__init__(master, fg_color=theme.SURFACE)
        self.model = data.get("model")
        self._cb = on_click
        self.grid_columnconfigure(tuple(range(len(_COLS))), weight=1, uniform="pp")
        for i, (key, _label) in enumerate(_COLS):
            txt = _fmt(key, data.get(key))
            # Dollar figure gets the accent colour so the leak reads at a glance.
            color = (theme.ACCENT if key == "dollar_impact" and data.get(key)
                     else theme.TEXT_PRIMARY)
            lbl = ctk.CTkLabel(self, text=txt, font=theme.font(theme.SIZE_BODY),
                               text_color=color, anchor="w")
            lbl.grid(row=0, column=i, sticky="ew", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
            lbl.bind("<Button-1>", lambda e: self._click())
        self.bind("<Button-1>", lambda e: self._click())

    def _click(self):
        if self._cb and self.model:
            self._cb(self.model)
