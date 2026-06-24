"""Spec 3c — UnitsTab: recent units for the model + serial lookup. Click → per-unit chart modal.

Recent list is capped (most-recent N for the window). The search box bypasses that cap and
the window via on_search → a DB query, so you can pull up a specific unit by serial even if
it's old or outside the current window — the Job-2a 'find the unit I'm working on now' need.
"""
from typing import Callable, Dict, List, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

_COLUMNS = [("serial", "Serial"), ("file_date", "Date"), ("overall_status", "Status"),
            ("sigma_gradient", "Sigma"), ("linearity_error", "Lin Err")]


class UnitsTab(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_unit_click: Callable[[dict], None],
                 on_export: Callable[[], None],
                 on_search: Optional[Callable[[str], None]] = None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._on_unit_click = on_unit_click
        self._on_export = on_export
        self._on_search = on_search
        self._units: List[dict] = []
        self._rows: List[_UnitRow] = []
        self._sort_key = "file_date"
        self._sort_rev = True
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        ctk.CTkButton(bar, text="Export to Excel", fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                      text_color=theme.TEXT_INVERSE, command=self._on_export,
                      corner_radius=theme.RADIUS_SM).pack(side="right")
        # Serial lookup (bypasses the recent-cap and window — see module docstring).
        self._search = ctk.CTkEntry(bar, placeholder_text="Find serial…", width=180,
                                    font=theme.font(theme.SIZE_BODY))
        self._search.pack(side="left")
        self._search.bind("<Return>", lambda e: self._do_search())
        ctk.CTkButton(bar, text="Search", width=64, fg_color=theme.ACCENT,
                      hover_color=theme.ACCENT_HOVER, text_color=theme.TEXT_INVERSE,
                      command=self._do_search, corner_radius=theme.RADIUS_SM
                      ).pack(side="left", padx=(theme.SPACE_XS, 0))
        ctk.CTkButton(bar, text="Recent", width=64, fg_color="transparent",
                      border_width=1, border_color=theme.BORDER, text_color=theme.TEXT_SECONDARY,
                      command=self._clear_search, corner_radius=theme.RADIUS_SM
                      ).pack(side="left", padx=(theme.SPACE_XS, 0))
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

    def set_units(self, units: List[dict], caption: Optional[str] = None) -> None:
        self._units = list(units)
        self._caption = caption
        self._render()

    def _do_search(self):
        q = self._search.get().strip()
        if q and self._on_search:
            self._on_search(q)

    def _clear_search(self):
        self._search.delete(0, "end")
        if self._on_search:
            self._on_search("")   # empty query → loader restores the recent list

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
        cap = getattr(self, "_caption", None)
        self._cap.configure(text=cap if cap else f"{len(ordered)} unit(s)")


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
