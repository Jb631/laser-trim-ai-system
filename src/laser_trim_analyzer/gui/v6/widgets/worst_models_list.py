"""Dashboard — WorstModelsList: ranked, clickable model rows (model/units/trim%/FT%)."""
from typing import Callable, List, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets import blocks

# Gap = Trim% − FT%. Strongly NEGATIVE = units failing trim but passing final
# test — the overkill pattern (trim thresholds or specs rejecting good product).
# Strongly POSITIVE = passing trim but failing FT — escapes (worse). Ported
# from V5 Quality Health's ranked table (2026-07-07, feature restoration).
_GAP_TAG_THRESHOLD = 15


def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.0f}%"


class WorstModelsList(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_row_click: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_row_click
        self._rows: List[dict] = []             # the data behind the rows currently shown
        self._row_widgets: List[ctk.CTkFrame] = []
        t = theme
        ctk.CTkLabel(self, text="Lowest-yield models", font=t.font(t.SIZE_HEADING, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w").pack(side="top", fill="x", pady=(0, t.SPACE_XS))
        # Row key (live-walk finding, 2026-07-08: 'Gap -62' meant nothing cold).
        # Same treatment as the sigma gloss: one plain-language line. The
        # readout on the right of each row is Trim % (linearity yield, the
        # worst-first ranking); 'overkill'/'escapes' tags a gap of >=15 points.
        # No fixed pixel wraplength on page-width text (global-constraints.md)
        # -- follows this frame's own width via blocks.wrap_to_width, same as
        # the page caption; the narrower audited size (1280x720) is where the
        # old fixed guess (1200) clipped this line once it grew past a
        # 'overkill'/'escapes' explanation.
        self._gloss = ctk.CTkLabel(self, text=(
            "Readout = Trim % (linearity yield) in the window, worst first. Gap = Trim − FT "
            "in points; tagged 'overkill' when trim grades worse than final test by 15+ points "
            "(rejecting good product), 'escapes' the other way (worse). Models with ≥5 units."),
                                   font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY,
                                   anchor="w", justify="left")
        self._gloss.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        blocks.wrap_to_width(self._gloss, self, padding=0)
        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)
        self._cap = ctk.CTkLabel(self, text="", font=t.font(t.SIZE_CAPTION),
                                 text_color=t.TEXT_SECONDARY, anchor="w")
        self._cap.pack(side="top", fill="x")
        self._empty = ctk.CTkLabel(self, text="", font=t.font(t.SIZE_BODY),
                                   text_color=t.TEXT_SECONDARY, anchor="w")

    def set_rows(self, rows: Optional[List[dict]], total: int) -> None:
        """`rows=None` means the loader FAILED -- say so; `[]` means there was nothing
        to rank. The two must never read alike (a failure is not "no data")."""
        for w in self._row_widgets:
            w.destroy()
        self._row_widgets.clear()
        self._rows = list(rows or [])
        t = self.theme
        if not rows:
            self._cap.configure(text="")
            self._empty.configure(text=("Unavailable — the notice above says why." if rows is None
                                        else "No models with enough recent data to rank."))
            self._empty.pack(side="top", fill="x", pady=t.SPACE_MD)
            return
        self._empty.pack_forget()
        for row in rows:
            w = self._build_row(row)
            w.pack(side="top", fill="x")
            self._row_widgets.append(w)
        self._cap.configure(text=f"Showing {len(rows)} of {total} (min 5 units, worst first)."
                            if total > len(rows) else f"{total} models (min 5 units, worst first).")

    def _build_row(self, row: dict) -> ctk.CTkFrame:
        """blocks.row: model (mono) . statement and tags . readout (mono, right) --
        the readout is Trim % (what this list ranks by); the statement carries
        units and FT %; a big gap earns a plain-word tag instead of colour."""
        t = self.theme
        gap = None
        if row.get("trim_rate") is not None and row.get("ft_rate") is not None:
            gap = row["trim_rate"] - row["ft_rate"]
        statement = f"{row.get('units', 0)} units · final test {_pct(row.get('ft_rate'))}"
        if gap is not None:
            statement += f" · gap {gap:+.0f} pts"
        tags = []
        if gap is not None and abs(gap) >= _GAP_TAG_THRESHOLD:
            tags.append("overkill" if gap < 0 else "escapes")
        return blocks.row(self._list, t, row["model"], statement, _pct(row.get("trim_rate")),
                          tags=tags, on_click=lambda m=row["model"]: self._cb(m))
