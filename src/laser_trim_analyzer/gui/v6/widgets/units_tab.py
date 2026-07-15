"""Spec 3c — UnitsTab: recent units for the model + serial lookup. Click → per-unit chart modal.

Recent list is capped (most-recent N for the window). The search box bypasses that cap and
the window via on_search → a DB query, so you can pull up a specific unit by serial even if
it's old or outside the current window — the Job-2a 'find the unit I'm working on now' need.
"""
from typing import Callable, Dict, List, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

_COLUMNS = [("serial", "Serial"), ("file_date", "Date"), ("overall_status", "Status"),
            ("sigma_gradient", "Sigma gradient"), ("linearity_error", "Linearity error")]


class UnitsTab(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_unit_click: Callable[[dict], None],
                 on_export: Callable[[], None],
                 on_search: Optional[Callable[[str], None]] = None,
                 on_export_charts: Optional[Callable[[], None]] = None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._on_unit_click = on_unit_click
        self._on_export = on_export
        self._on_search = on_search
        self._units: List[dict] = []
        self._rows: List[_UnitRow] = []
        self._sort_key = "file_date"
        self._sort_rev = True
        self._selected: set = set()   # analysis_id of checked rows (subset export)
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        ctk.CTkButton(bar, text="Export to Excel", fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                      text_color=theme.TEXT_INVERSE, command=self._on_export,
                      corner_radius=theme.RADIUS_SM).pack(side="right")
        if on_export_charts is not None:
            # Chart export: the checked rows (or all shown if none are checked)
            # as a single multi-page print-ready PDF.
            ctk.CTkButton(bar, text="Export charts (PDF)", fg_color=theme.CARD,
                          hover_color=theme.ELEVATED, text_color=theme.TEXT_PRIMARY,
                          border_width=1, border_color=theme.BORDER,
                          command=on_export_charts, corner_radius=theme.RADIUS_SM)\
                .pack(side="right", padx=(0, theme.SPACE_SM))
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
        # Spacer keeps the column headers aligned with the per-row checkbox.
        ctk.CTkLabel(header, text="", width=26).pack(side="left", padx=(theme.SPACE_SM, 0))
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

    def set_caption(self, text: str) -> None:
        """Live status line (mass chart export progress etc.)."""
        self._cap.configure(text=text)

    def set_units(self, units: List[dict], caption: Optional[str] = None) -> None:
        self._units = list(units)
        self._caption = caption
        self._selected = set()       # new model/window → clear checkboxes
        self._render()

    def get_selected_units(self) -> List[dict]:
        """The checked rows; if none are checked, every unit currently shown."""
        if not self._selected:
            return list(self._units)
        return [u for u in self._units if u.get("analysis_id") in self._selected]

    def _toggle_select(self, unit: dict, checked: bool) -> None:
        uid = unit.get("analysis_id")
        if checked:
            self._selected.add(uid)
        else:
            self._selected.discard(uid)
        base = getattr(self, "_caption", None) or f"{len(self._units)} unit(s)"
        n = len(self._selected)
        self._cap.configure(text=base + (f" · {n} selected" if n else ""))

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
            uid = u.get("analysis_id")
            row = _UnitRow(self._list, unit=u, theme=self.theme, on_click=self._on_unit_click,
                           on_toggle=self._toggle_select, selected=(uid in self._selected))
            row.pack(side="top", fill="x", pady=1)
            self._rows.append(row)
        cap = getattr(self, "_caption", None)
        n = len(self._selected)
        base = cap if cap else f"{len(ordered)} unit(s)"
        self._cap.configure(text=base + (f" · {n} selected" if n else ""))


class _UnitRow(ctk.CTkFrame):
    def __init__(self, master, unit: dict, theme: ThemeManager, on_click,
                 on_toggle=None, selected: bool = False):
        super().__init__(master, fg_color=theme.SURFACE)
        self.unit = unit
        self._cb = on_click
        self._on_toggle = on_toggle
        # Per-row select checkbox for subset chart export. It's a real widget so
        # it consumes its own click and toggling never opens the chart modal.
        if on_toggle is not None:
            self._chk = ctk.CTkCheckBox(self, text="", width=26, checkbox_width=16,
                                        checkbox_height=16, fg_color=theme.ACCENT,
                                        hover_color=theme.ACCENT_HOVER, command=self._toggled)
            if selected:
                self._chk.select()
            self._chk.pack(side="left", padx=(theme.SPACE_SM, 0))
        # Status gets tier color so a fail-heavy list reads at a glance
        # (live-walk finding, 2026-07-08: a column of plain-white 'Fail').
        _status_color = {"Fail": theme.TIER_OOC, "FAIL": theme.TIER_OOC,
                         "Warning": theme.TIER_WARNING, "WARNING": theme.TIER_WARNING}
        for key, _ in _COLUMNS:
            v = unit.get(key)
            txt = (v.strftime("%Y-%m-%d") if hasattr(v, "strftime")
                   else f"{v:.4g}" if isinstance(v, float) else str(v) if v is not None else "—")
            color = _status_color.get(txt, theme.TEXT_PRIMARY) if key == "overall_status" \
                else theme.TEXT_PRIMARY
            lbl = ctk.CTkLabel(self, text=txt, font=theme.font(theme.SIZE_BODY),
                               text_color=color)
            lbl.pack(side="left", expand=True, fill="x", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
            lbl.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self):
        self._cb(self.unit)

    def _toggled(self):
        if self._on_toggle is not None:
            self._on_toggle(self.unit, bool(self._chk.get()))
