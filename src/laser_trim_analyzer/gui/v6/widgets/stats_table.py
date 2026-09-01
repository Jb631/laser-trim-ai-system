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

WHY THE TABLE IS BANDED AND NOT RULED (James, 2026-08-30: "stats table didnt
look right i dunno if it needs lines or boarders?")
------------------------------------------------------------------------------
Two things were wrong, and neither is fixed by gridlines.

  * A metric is not one line. It is a number row plus up to two CAPTION lines —
    the "4 not recorded" disclosure and the "this lot: ..." comparison — and
    those captions span the full width, so nothing bound them to the metric
    they belong to. On a card background they float between two metrics and the
    reader has to guess upwards. So each metric gets ONE subtle background
    block (`theme.ELEVATED` on `theme.CARD`) covering its number row AND its
    captions, alternating with the bare card. The block is the binding; that is
    the load-bearing half of this fix.
  * The ALL and LIN-PASSING groups were told apart only by two labels on row 0,
    which is no help eight columns in. So there is ONE vertical rule between
    the groups and ONE horizontal rule under the header rows, and the group
    headings are centred over the columns they name.

Full gridlines were the obvious answer and are the wrong one: 8 numeric columns
boxed cell-by-cell reads as a spreadsheet screenshot, and the caption lines —
which span every column — cannot live in a boxed grid at all without looking
like data. Banding carries the grouping with no ink.

ONE grid holds the whole table, deliberately. A frame per metric would let each
metric size its own columns, and the numbers would stop lining up down the
page — the other half of "didnt look right".

Main-thread only: the page's worker computes the ModelStats and posts it back
through the page's `safe_after`.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import customtkinter as ctk

# The TEXT lives in core/model_stats.py, next to the numbers it formats, and is
# imported here rather than written here. That is what lets `export/evidence.py`
# print the identical strings without importing Tk — the sheet James hands an
# engineer is the screen, not a second rendering of the same data.
from laser_trim_analyzer.core.model_stats import (
    DIST_HEADERS, RATE_HEADERS, Cell, LotVerdict, ModelStats, StatRow,
    cell_texts, disclosure_text, format_in, lot_line, row_unit, summary_line)
from laser_trim_analyzer.gui.v6.theme import ThemeManager

# The fixed rows at the top of every block, and the first row a metric may use.
# Kept as names because the band plan's whole job is to stay off them.
ROW_GROUPS = 0        # ALL UNITS / LIN-PASSING (accepted)
ROW_HEADERS = 1       # n · avg · min · max
ROW_HEAD_RULE = 2     # the hairline under the headings
FIRST_DATA_ROW = 3


def band_plan(caption_counts: Sequence[int],
              first_row: int = FIRST_DATA_ROW) -> List[Tuple[int, int, bool]]:
    """Where each metric's background block sits: (start_row, span, banded).

    Pure, and tested as such, because the span is the whole point: a band one
    row short leaves the caption it was meant to bind floating on bare card,
    which is the exact complaint this layout answers. `caption_counts[i]` is how
    many sub-lines metric i prints (0-2: its disclosure, its lot line), so the
    span is that plus the number row itself.

    Bands ALTERNATE rather than banding everything: the gap between two blocks
    is what says "new metric", so two banded metrics in a row would merge into
    one apparent metric with four captions. The plan tiles the data rows with no
    gap and no overlap, and starts below the header rows — a band reaching up
    into the column names would tie them to the first metric.
    """
    plan: List[Tuple[int, int, bool]] = []
    row = first_row
    for index, captions in enumerate(caption_counts):
        span = 1 + max(int(captions), 0)
        plan.append((row, span, index % 2 == 0))
        row += span
    return plan


def _cell_column(index: int, width: int) -> int:
    """Grid column for the `index`-th numeric cell, counting across BOTH groups.

    The step at `width` is the rule column: the vertical line between ALL and
    LIN-PASSING is a real grid column, so every LIN-PASSING cell shifts one over.
    """
    return 1 + index + (1 if index >= width else 0)


def _caption_count(row: StatRow, lot_rows: Dict[str, StatRow]) -> int:
    """How many sub-lines this metric will print — what its band has to cover."""
    return ((1 if disclosure_text(row.all_) else 0)
            + (1 if row.key in lot_rows else 0))


class StatsTableZone(ctk.CTkFrame):
    """The stats table, and — when a lot is selected — how that lot compares."""

    CAPTION = ("LIN-PASSING = units the customer accepted (linearity passed; "
               "a sigma WARNING still ships). ALL = every unit, including the "
               "linearity rejects. Blank cells mean nothing was recorded — "
               "never a zero.")
    RATE_CAPTION = ("These two count TRACK measurements, not units: a two-track "
                    "unit is two rows here, and a re-trimmed track is one row "
                    "per attempt. Not a unit yield — for that, see the "
                    "Dashboard's yield, which takes each track's last attempt "
                    "of the day and requires every track to pass.")
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
        self._line(self.RATE_CAPTION, t.TEXT_SECONDARY, caption=True)
        self._line(self.CAPTION, t.TEXT_SECONDARY, caption=True)
        if lot_stats is not None:
            self._line(self.LOT_CAPTION, t.TEXT_SECONDARY, caption=True)

    # ---- render helpers ----
    def _grid(self, rows, headers, lot_rows, verdicts) -> None:
        """One block: two header rows, a rule under them, then a banded line
        per metric (with that metric's captions inside the same band)."""
        if not rows:
            return                  # header rows over nothing are just noise
        t = self.theme
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(side="top", fill="x", padx=t.SPACE_MD, pady=(0, t.SPACE_XS))
        self._rendered.append(frame)
        width = len(headers)
        rule_column = 1 + width
        full = 2 + 2 * width        # label + both groups + the rule between them
        frame.grid_columnconfigure(0, weight=1, minsize=190)
        for column in range(1, full):
            if column != rule_column:   # the rule column is as wide as its 1px
                frame.grid_columnconfigure(column, minsize=78)

        plan = band_plan([_caption_count(row, lot_rows) for row in rows])
        end_row = plan[-1][0] + plan[-1][1]

        # Bands FIRST. Same-parent widgets stack in CREATION order, so the
        # blocks have to exist before the labels that sit on them; .lower() as
        # well, so a later edit that moves this loop cannot silently hide the
        # table behind its own banding.
        for start, span, banded in plan:
            if not banded:
                continue
            # width/height 1 on purpose: a CTkFrame asks for 200x200 by default,
            # and a spanning slave that big would force the metric rows to be
            # 200px tall. sticky="nsew" gives it the real size.
            band = ctk.CTkFrame(frame, fg_color=t.ELEVATED, width=1, height=1,
                                corner_radius=t.RADIUS_SM)
            band.grid(row=start, rowspan=span, column=0, columnspan=full,
                      sticky="nsew")
            band.lower()
        # The two rules, and there are only two. Created after the bands so the
        # group rule stays visible where it crosses one.
        self._rule(frame, row=ROW_HEAD_RULE, column=0, columnspan=full,
                   sticky="ew", pady=(t.SPACE_XS, t.SPACE_XS))
        self._rule(frame, row=ROW_GROUPS, rowspan=end_row, column=rule_column,
                   sticky="ns", padx=(t.SPACE_MD, t.SPACE_SM))

        # Disposition band — the split is the whole point of the table, so it
        # gets its own row of headings, centred over the columns it names.
        for label, first in (("ALL UNITS", 1),
                             ("LIN-PASSING (accepted)", rule_column + 1)):
            ctk.CTkLabel(frame, text=label, font=t.font(t.SIZE_CAPTION, "bold"),
                         text_color=t.TEXT_SECONDARY, anchor="center")\
                .grid(row=ROW_GROUPS, column=first, columnspan=width,
                      sticky="ew", padx=(t.SPACE_SM, 0))
        for index, name in enumerate(headers * 2):
            ctk.CTkLabel(frame, text=name, font=t.font(t.SIZE_CAPTION),
                         text_color=t.TEXT_SECONDARY, anchor="e")\
                .grid(row=ROW_HEADERS, column=_cell_column(index, width),
                      sticky="e", padx=(t.SPACE_SM, 0))

        for (start, span, banded), row in zip(plan, rows):
            # A CTkLabel left "transparent" paints its MASTER's colour, not
            # nothing — over a band that means a card-coloured hole per cell.
            # So every label in a banded metric is painted the band's colour and
            # the block reads as one surface.
            fill = t.ELEVATED if banded else "transparent"
            last = start + span - 1
            pady = (t.SPACE_XS, t.SPACE_XS if start == last else 0)
            ctk.CTkLabel(frame, text=row.label, font=t.font(t.SIZE_BODY),
                         fg_color=fill, text_color=t.TEXT_PRIMARY, anchor="w")\
                .grid(row=start, column=0, sticky="w", pady=pady)
            for offset, cell in ((0, row.all_), (width, row.lin_passing)):
                for index, text in enumerate(cell_texts(row, cell)):
                    ctk.CTkLabel(frame, text=text, font=t.font(t.SIZE_BODY),
                                 fg_color=fill,
                                 text_color=(t.TEXT_PRIMARY if cell.n
                                             else t.TEXT_DISABLED), anchor="e")\
                        .grid(row=start, sticky="e", pady=pady,
                              column=_cell_column(offset + index, width),
                              padx=(t.SPACE_SM, 0))
            line = start + 1
            note = disclosure_text(row.all_)
            if note:
                self._caption(frame, f"   {note}", t.TEXT_SECONDARY, fill,
                              row=line, columnspan=full, last=line == last)
                line += 1
            lot_row = lot_rows.get(row.key)
            if lot_row is not None:
                verdict = verdicts.get(row.key)
                # Amber only when the lot is genuinely out of family: a page
                # that colours "within its normal" teaches people to ignore it.
                color = (t.TIER_WARNING if verdict and verdict.status in
                         ("above", "below") else t.TEXT_SECONDARY)
                self._caption(frame, f"   {lot_line(row, lot_row.all_, verdict)}",
                              color, fill, row=line, columnspan=full,
                              last=line == last)
                line += 1

    def _rule(self, frame, **grid_kw) -> None:
        """A hairline. Two per block and no more — see the module docstring."""
        ctk.CTkFrame(frame, fg_color=self.theme.BORDER, corner_radius=0,
                     width=1, height=1).grid(**grid_kw)

    def _caption(self, frame, text, color, fill, *, row, columnspan,
                 last) -> None:
        """A metric's sub-line: tight under its number row, inside its band."""
        t = self.theme
        ctk.CTkLabel(frame, text=text, font=t.font(t.SIZE_CAPTION),
                     fg_color=fill, text_color=color, anchor="w",
                     justify="left", wraplength=1100)\
            .grid(row=row, column=0, columnspan=columnspan, sticky="w",
                  pady=(0, t.SPACE_XS if last else 0))

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
