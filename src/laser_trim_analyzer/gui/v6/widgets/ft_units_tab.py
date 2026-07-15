"""Model page — FtUnitsTab: final-test records for the model.

Work finding #3 (2026-07-10): "no way to view final test units." The Units
tab shows trim records; this one lists what happened at the LAST station —
result, date, and whether the record linked back to a trim unit. Rows are
clickable (James 2026-07-14: "if you click on a final test unit it does not
show a chart?") — the callback opens the FT sweep modal.
"""
from typing import Callable, Dict, List, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

_COLUMNS = [("serial", "Serial"), ("file_date", "Test date"), ("result", "Result"),
            ("linked", "Linked to trim"), ("match", "Match %")]


class FtUnitsTab(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager,
                 on_unit_click: Optional[Callable[[Dict], None]] = None,
                 on_export_charts: Optional[Callable[[], None]] = None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_unit_click
        self._rows: List[ctk.CTkFrame] = []
        self._units: List[Dict] = []
        self._selected: set = set()   # FT record id of checked rows
        if on_export_charts is not None:
            bar = ctk.CTkFrame(self, fg_color="transparent")
            bar.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
            # Checked rows (or all shown if none checked) as one multi-page PDF.
            ctk.CTkButton(bar, text="Export charts (PDF)", fg_color=theme.CARD,
                          hover_color=theme.ELEVATED, text_color=theme.TEXT_PRIMARY,
                          border_width=1, border_color=theme.BORDER,
                          command=on_export_charts, corner_radius=theme.RADIUS_SM)\
                .pack(side="right")
        header = ctk.CTkFrame(self, fg_color=theme.CARD)
        header.pack(side="top", fill="x", pady=(0, theme.SPACE_XS))
        # Spacer keeps the column headers aligned with the per-row checkbox.
        ctk.CTkLabel(header, text="", width=26).pack(side="left", padx=(theme.SPACE_SM, 0))
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

    def set_caption(self, text: str) -> None:
        """Live status line (chart-export progress etc.)."""
        self._cap.configure(text=text)

    def get_selected_units(self) -> List[Dict]:
        """Checked FT rows; if none are checked, every record currently shown."""
        if not self._selected:
            return list(self._units)
        return [u for u in self._units if u.get("id") in self._selected]

    def _toggle_select(self, unit: Dict, checked: bool) -> None:
        rid = unit.get("id")
        if checked:
            self._selected.add(rid)
        else:
            self._selected.discard(rid)
        # Mirror UnitsTab: show a running selected-count so the subset export
        # scope is visible before the user clicks Export charts (PDF).
        base = getattr(self, "_base_cap", None) or f"{len(self._units)} final-test record(s)"
        n = len(self._selected)
        self._cap.configure(text=base + (f" · {n} selected" if n else ""))

    def set_units(self, units: List[Dict]) -> None:
        for r in self._rows:
            try:
                r.destroy()
            except Exception:
                pass
        self._rows = []
        self._units = list(units)
        self._selected = set()
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
            chk = ctk.CTkCheckBox(row, text="", width=26, checkbox_width=16,
                                  checkbox_height=16, fg_color=t.ACCENT,
                                  hover_color=t.ACCENT_HOVER)
            chk.configure(command=lambda u_=u, c_=chk: self._toggle_select(u_, bool(c_.get())))
            chk.pack(side="left", padx=(t.SPACE_SM, 0))
            vals = [str(u.get("serial") or "—"),
                    (u["file_date"].strftime("%Y-%m-%d") if u.get("file_date") else "—"),
                    str(u.get("result") or "—"),
                    "yes" if u.get("linked") else "no",
                    (f"{u['match']}%" if u.get("match") is not None else "—")]
            widgets = [row]
            for (key, _), v in zip(_COLUMNS, vals):
                c = color.get(v, t.TEXT_PRIMARY) if key == "result" else t.TEXT_PRIMARY
                lbl = ctk.CTkLabel(row, text=v, font=t.font(t.SIZE_BODY), text_color=c)
                lbl.pack(side="left", expand=True, fill="x",
                         padx=t.SPACE_SM, pady=t.SPACE_XS)
                widgets.append(lbl)
            if self._cb is not None:
                for w in widgets:
                    w.bind("<Button-1>", lambda e, u_=u: self._cb(u_))
            row.pack(side="top", fill="x", pady=1)
            self._rows.append(row)
        n_fail = sum(1 for u in units if u.get("result") == "FAIL")
        self._base_cap = (f"{len(units)} final-test record(s) · {n_fail} failed"
                          + ("   (showing newest 500)" if len(units) == 500 else ""))
        self._cap.configure(text=self._base_cap)
