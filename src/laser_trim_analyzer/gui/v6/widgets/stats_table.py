"""INVESTIGATE stats table — the screen that ends the Excel round trip.

James's week (app-shape spec, 2026-08-29): export a model to Excel just to work
out historical avg / min / max for resistance and electrical angle, all units
versus lin-passing units. This zone puts that on the page.

Split the way `widgets/focus_list_zone.py` is split, for the same reason: the
TEXT is a set of pure functions (testable without a Tk root, and the same
strings the Excel sheet prints), and the widget below is a thin render of them.
Every number and every sentence is written by `core/model_stats.py` — nothing
here reformats a value or decides what "normal" means, so the table, the
verdict and the export cannot come out different.

Main-thread only: the page's worker computes the ModelStats and posts it back
through the page's `safe_after`.
"""
from typing import Dict, List, Optional

import customtkinter as ctk

# The TEXT lives in core/model_stats.py, next to the numbers it formats, and is
# imported here rather than written here. That is what lets `export/evidence.py`
# print the identical strings without importing Tk — the sheet James hands an
# engineer is the screen, not a second rendering of the same data.
from laser_trim_analyzer.core.model_stats import (
    DIST_HEADERS, RATE_HEADERS, Cell, LotVerdict, ModelStats, StatRow,
    cell_texts, disclosure_text, format_in, lot_line, row_unit, summary_line)
from laser_trim_analyzer.gui.v6.theme import ThemeManager

class StatsTableZone(ctk.CTkFrame):
    """The stats table, and — when a lot is selected — how that lot compares."""

    CAPTION = ("LIN-PASSING = units the customer accepted (linearity passed; "
               "a sigma WARNING still ships). ALL = every unit, including the "
               "linearity rejects. Blank cells mean nothing was recorded — "
               "never a zero.")
    LOT_CAPTION = ("\"Typically\" is the lot's median — the value the control "
                   "chart plots and the limits were built from; the avg column "
                   "beside it is the plain average of the same lot.")

    def __init__(self, master, *, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color=theme.CARD,
                         corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme
        self._rendered: List[ctk.CTkBaseClass] = []
        self._empty = None

    # ---- render ----
    def set_stats(self, stats: Optional[ModelStats], *,
                  lot_stats: Optional[ModelStats] = None,
                  verdicts: Optional[Dict[str, LotVerdict]] = None,
                  lot_label: str = "") -> None:
        for w in self._rendered:
            try:
                w.destroy()
            except Exception:
                pass
        self._rendered = []
        t = self.theme
        if stats is None:
            self._line("Pick a model to see its history.", t.TEXT_SECONDARY)
            return
        self._line(summary_line(stats), t.TEXT_SECONDARY)
        if not stats.tracks:
            return
        if lot_label:
            self._line(f"Lot selected: {lot_label}", t.TEXT_PRIMARY, bold=True)

        lot_rows = {r.key: r for r in (lot_stats.rows if lot_stats else [])}
        self._grid(stats.distribution_rows, DIST_HEADERS, lot_rows,
                   verdicts or {})
        self._line("", t.TEXT_SECONDARY)
        self._grid(stats.rate_rows, RATE_HEADERS, lot_rows, verdicts or {})
        self._line(self.CAPTION, t.TEXT_SECONDARY, caption=True)
        if lot_stats is not None:
            self._line(self.LOT_CAPTION, t.TEXT_SECONDARY, caption=True)

    # ---- render helpers ----
    def _grid(self, rows, headers, lot_rows, verdicts) -> None:
        """One block: a header band, then a line per metric (+ its lot line)."""
        t = self.theme
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(side="top", fill="x", padx=t.SPACE_MD, pady=(0, t.SPACE_XS))
        self._rendered.append(frame)
        frame.grid_columnconfigure(0, weight=1, minsize=190)
        width = len(headers)
        for column in range(1, 1 + 2 * width):
            frame.grid_columnconfigure(column, minsize=78)
        # Disposition band — the split is the whole point of the table, so it
        # gets its own row of headings above the column names.
        for label, first in (("ALL UNITS", 1), ("LIN-PASSING (accepted)", 1 + width)):
            ctk.CTkLabel(frame, text=label, font=t.font(t.SIZE_CAPTION, "bold"),
                         text_color=t.TEXT_SECONDARY, anchor="w")\
                .grid(row=0, column=first, columnspan=width, sticky="w",
                      padx=(t.SPACE_SM, 0))
        for index, name in enumerate(headers * 2):
            ctk.CTkLabel(frame, text=name, font=t.font(t.SIZE_CAPTION),
                         text_color=t.TEXT_SECONDARY, anchor="e")\
                .grid(row=1, column=1 + index, sticky="e", padx=(t.SPACE_SM, 0))
        line = 2
        for row in rows:
            ctk.CTkLabel(frame, text=row.label, font=t.font(t.SIZE_BODY),
                         text_color=t.TEXT_PRIMARY, anchor="w")\
                .grid(row=line, column=0, sticky="w", pady=1)
            for offset, cell in ((0, row.all_), (width, row.lin_passing)):
                for index, text in enumerate(cell_texts(row, cell)):
                    ctk.CTkLabel(frame, text=text, font=t.font(t.SIZE_BODY),
                                 text_color=(t.TEXT_PRIMARY if cell.n
                                             else t.TEXT_DISABLED), anchor="e")\
                        .grid(row=line, column=1 + offset + index, sticky="e",
                              padx=(t.SPACE_SM, 0))
            line += 1
            note = disclosure_text(row.all_)
            if note:
                ctk.CTkLabel(frame, text=f"   {note}",
                             font=t.font(t.SIZE_CAPTION),
                             text_color=t.TEXT_SECONDARY, anchor="w")\
                    .grid(row=line, column=0, columnspan=1 + 2 * width,
                          sticky="w")
                line += 1
            lot_row = lot_rows.get(row.key)
            if lot_row is not None:
                verdict = verdicts.get(row.key)
                # Amber only when the lot is genuinely out of family: a page
                # that colours "within its normal" teaches people to ignore it.
                color = (t.TIER_WARNING if verdict and verdict.status in
                         ("above", "below") else t.TEXT_SECONDARY)
                ctk.CTkLabel(frame, text=f"   {lot_line(row, lot_row.all_, verdict)}",
                             font=t.font(t.SIZE_CAPTION), text_color=color,
                             anchor="w", justify="left", wraplength=1100)\
                    .grid(row=line, column=0, columnspan=1 + 2 * width,
                          sticky="w")
                line += 1

    def _line(self, text, color, *, bold=False, caption=False) -> None:
        t = self.theme
        label = ctk.CTkLabel(
            self, text=text,
            font=t.font(t.SIZE_CAPTION if caption else t.SIZE_BODY,
                        "bold" if bold else "normal"),
            text_color=color, anchor="w", justify="left", wraplength=1150)
        label.pack(side="top", fill="x", padx=t.SPACE_MD,
                   pady=(t.SPACE_XS, 0))
        self._rendered.append(label)
