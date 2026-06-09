"""Spec 3c — UnitsTab: recent units for the model (dict rows). Click → per-unit chart modal."""
from typing import Callable, Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

_COLUMNS = [("serial", "Serial"), ("file_date", "Date"), ("overall_status", "Status"),
            ("sigma_gradient", "Sigma"), ("linearity_error", "Lin Err")]


class UnitsTab(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_unit_click: Callable[[dict], None],
                 on_export: Callable[[], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._on_unit_click = on_unit_click
        self._on_export = on_export
        self._units: List[dict] = []
        self._rows: List[_UnitRow] = []
        self._sort_key = "file_date"
        self._sort_rev = True
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        ctk.CTkButton(bar, text="Export to Excel", fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                      text_color=theme.TEXT_INVERSE, command=self._on_export,
                      corner_radius=theme.RADIUS_SM).pack(side="right")
        header = ctk.CTkFrame(self, fg_color=theme.CARD)
        header.pack(side="top", fill="x", pady=(0, theme.SPACE_XS))
        for key, lbl in _COLUMNS:
            h = ctk.CTkLabel(header, text=lbl, font=theme.font(theme.SIZE_CAPTION, "bold"),
                             text_color=theme.TEXT_SECONDARY)
            h.pack(side="left", expand=True, fill="x", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
            h.bind("<Button-1>", lambda e, k=key: self._sort_by(k))
        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)
        self._cap = ctk.CTkLabel(self, text="", font=theme.font(theme.SIZE_CAPTION),
                                 text_color=theme.TEXT_SECONDARY)
        self._cap.pack(side="top", fill="x")

    def set_units(self, units: List[dict]) -> None:
        self._units = list(units)
        self._render()

    def _sort_by(self, key):
        self._sort_rev = not self._sort_rev if self._sort_key == key else False
        self._sort_key = key
        self._render()

    def _render(self):
        for r in self._rows:
            r.destroy()
        self._rows.clear()
        ordered = sorted(self._units,
                         key=lambda u: (u.get(self._sort_key) is None, u.get(self._sort_key)),
                         reverse=self._sort_rev)
        for u in ordered:
            row = _UnitRow(self._list, unit=u, theme=self.theme, on_click=self._on_unit_click)
            row.pack(side="top", fill="x", pady=1)
            self._rows.append(row)


class _UnitRow(ctk.CTkFrame):
    def __init__(self, master, unit: dict, theme: ThemeManager, on_click):
        super().__init__(master, fg_color=theme.SURFACE)
        self.unit = unit
        self._cb = on_click
        for key, _ in _COLUMNS:
            v = unit.get(key)
            txt = (v.strftime("%Y-%m-%d") if hasattr(v, "strftime")
                   else f"{v:.4g}" if isinstance(v, float) else str(v) if v is not None else "—")
            lbl = ctk.CTkLabel(self, text=txt, font=theme.font(theme.SIZE_BODY),
                               text_color=theme.TEXT_PRIMARY)
            lbl.pack(side="left", expand=True, fill="x", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
            lbl.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self):
        self._cb(self.unit)
