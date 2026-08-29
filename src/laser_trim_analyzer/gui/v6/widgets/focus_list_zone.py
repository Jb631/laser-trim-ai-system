"""FOCUS list — the ranked "what is drifting right now" zone (2026-08-29 redesign).

This replaces the wall of σ cards. That wall showed one card per alerting model
with no order anyone could act on, so the first question every morning ("what do
I work on?") took a human sort. This zone answers it directly: models ranked by
what their drift is COSTING this week, each with the picture behind the claim.

Three rules the code below is built around:

  * ONE computation. `verdict` and `sub_line` are written by `ml/spc.py` and
    rendered here verbatim — this widget never reformats a rate or re-derives a
    number. The redesign exists because the list, the chart and the export each
    did their own arithmetic and disagreed.
  * The sparkline draws from `spc_draw_params(entry.series)` — the SAME helper
    (and the same series object) the Model page's full chart uses. The little
    picture in the row therefore cannot flag a different lot than the big chart
    the row links to.
  * The figures die with the rows. Seven sparklines rebuilt on every refresh is
    seven leaked matplotlib Figures per refresh in a process that stays open all
    shift, so `_FocusRow.destroy` releases the canvas and clears the figure.

Main-thread only, by design: the page's worker computes the `FocusResult` and
posts it back through `ui_dispatch`; nothing in here touches a thread.
"""
from datetime import datetime
from math import isfinite
from typing import Callable, Dict, List, Optional

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.focus_chart import spc_draw_params
from laser_trim_analyzer.ml.spc import RECENT_K, FocusEntry, FocusResult

# Seven is a screenful: past that the list stops reading as "today's work" and
# starts reading as a database dump. The rest are one click away, never hidden.
FOCUS_CAP = 7
BODY_HEIGHT = 320                       # scroll-bounded, like the zone it replaces
SPARK_W, SPARK_H, SPARK_DPI = 260, 64, 96


def focus_row_texts(entry: FocusEntry,
                    anchor: Optional[datetime]) -> Dict[str, str]:
    """The four strings one FOCUS row shows. Pure, so the wording is testable.

    `line1`/`line2` are the entry's own verdict and sub-line, passed through
    untouched (see the one-computation rule above). The only sentence this
    function writes is `when`, and it says three things a supervisor asks in
    order: which lot, how old it is, and whether it is even finished yet.

    An anchor of None (empty database) prints the date with no age rather than
    inventing one against the wall clock — the wall clock is not this data's
    clock.
    """
    end = entry.last_lot_end
    when = f"last lot {end:%b %d}"
    if anchor is not None:
        # Clamped at 0: a per-model anchor can sit a few hours behind a lot's
        # midnight-normalized end, and "(-1d ago)" is nonsense.
        days = max((anchor - end).days, 0)
        when += " (today)" if days == 0 else f" ({days}d ago)"
    points = entry.series.points
    if points and points[-1].is_open:
        # Same disclosure the chart makes with its hollow marker: a lot that may
        # still be receiving units is a preview, not a verdict.
        when += " · lot still open"
    return {"title": entry.model, "when": when,
            "line1": entry.verdict, "line2": entry.sub_line}


def _draw_sparkline(ax, p: dict, t: ThemeManager) -> None:
    """Band + dots, nothing else.

    At 260x64 there is no room for a sentence, a tick or a legend — anything
    more turns into grey mush. The row's job is "does the newest lot sit outside
    the shaded normal", and the full chart (one click away) carries the words.

    The whole series is drawn, not a trimmed tail: `flag_idx`/`old_idx` are
    positions in `series.points`, so trimming here would silently mark the wrong
    lots and break the parity this zone is built on.
    """
    ax.clear()
    ax.set_facecolor(t.CARD)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    xs, values = p["xs"], p["values"]
    if not xs:
        return
    if p["judged"]:
        # Step, not a smooth band: each lot's limit comes from its OWN size.
        ax.fill_between(xs, p["band_lo"], p["ucls"], step="mid",
                        color=t.ELEVATED, alpha=0.9, lw=0, zorder=0)
    ax.plot(xs, values, "o", ms=3, ls="none", color=t.TEXT_SECONDARY, zorder=2)
    if p["old_idx"]:
        # Amber: really out of control, but older than the window the verdict is
        # about. Present as context, not re-alarmed.
        ax.plot([xs[i] for i in p["old_idx"]], [values[i] for i in p["old_idx"]],
                "o", ms=3.5, ls="none", color=t.TIER_WARNING, zorder=3)
    if p["flag_idx"]:
        ax.plot([xs[i] for i in p["flag_idx"]], [values[i] for i in p["flag_idx"]],
                "o", ms=5, ls="none", color=t.TIER_OOC, zorder=4)
    open_idx = p["open_idx"]
    if open_idx is not None:
        edge = (t.TIER_OOC if open_idx in p["flag_idx"] else
                t.TIER_WARNING if open_idx in p["old_idx"] else t.TEXT_SECONDARY)
        ax.plot([xs[open_idx]], [values[open_idx]], "o", ms=5.5, ls="none",
                mfc=t.CARD, mec=edge, mew=1.4, zorder=5)

    # y-window mirrors the full chart's: the band stays visible even when every
    # lot is clean (an all-zero series would otherwise autoscale to a sliver),
    # and a 100% lot is never clipped off the top.
    finite = [v for v in values if isfinite(v)]
    hi_c, lo_c = list(finite), list(finite)
    if p["judged"]:
        hi_c += [u for u in p["ucls"] if isfinite(u)]
        lo_c += [b for b in p["band_lo"] if isfinite(b)]
    hi = max(hi_c) if hi_c else 1.0
    lo = 0.0 if p["fraction"] else (min(lo_c) if lo_c else 0.0)
    if p["fraction"]:
        hi = min(max(hi, 0.05), 1.0)
    span = (hi - lo) or (abs(hi) * 0.1) or 1.0
    ax.set_ylim(lo - span * 0.10, hi + span * 0.10)
    ax.set_xlim(-0.6, len(xs) - 0.4)


class FocusListZone(ctk.CTkFrame):
    """Heading + caption + ranked rows (+ expander) + the chronic strip."""

    def __init__(self, master, theme: ThemeManager,
                 on_row_click: Callable[[str, str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_row_click
        self._result = FocusResult(focus=[], chronic=[], anchor=None)
        self._last_processed: Optional[datetime] = None
        self._expanded = False
        self._rows: List["_FocusRow"] = []
        self._chronic_rows: List["_ChronicRow"] = []
        # Explicit bookkeeping: winfo_children() on a CTkScrollableFrame returns
        # its internal canvas/scrollbar, not the rows.
        self._rendered: List[ctk.CTkBaseClass] = []
        t = theme
        self._heading = ctk.CTkLabel(self, text="", anchor="w",
                                     font=t.font(t.SIZE_HEADING, "bold"),
                                     text_color=t.TEXT_PRIMARY)
        self._heading.pack(side="top", fill="x", pady=(0, t.SPACE_XS))
        # The membership rule, stated where the list is read. "Why is this model
        # here / why did it leave?" was the top question about the old wall.
        self._caption = ctk.CTkLabel(self, text="", anchor="w", justify="left",
                                     wraplength=1200, font=t.font(t.SIZE_CAPTION),
                                     text_color=t.TEXT_SECONDARY)
        self._caption.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        self._body = ctk.CTkScrollableFrame(self, fg_color="transparent",
                                            height=BODY_HEIGHT)
        self._body.pack(side="top", fill="x")
        # Persistent, never destroyed: CTkButton schedules a click animation on
        # itself, so destroying it from inside its own command leaves a pending
        # `after` pointing at a dead widget. It is packed/unpacked instead.
        self._more_btn = ctk.CTkButton(self._body, text="", command=self._toggle,
                                       fg_color="transparent", hover_color=t.CARD,
                                       text_color=t.ACCENT, anchor="w", height=26,
                                       font=t.font(t.SIZE_BODY))
        self._chronic_heading = ctk.CTkLabel(self, text="", anchor="w",
                                             font=t.font(t.SIZE_BODY, "bold"),
                                             text_color=t.TEXT_SECONDARY)
        self._chronic_body = ctk.CTkFrame(self, fg_color="transparent")
        self._render()

    # ---- public API -------------------------------------------------------

    def set_result(self, result: FocusResult,
                   last_processed: Optional[datetime] = None) -> None:
        self._result = result
        self._last_processed = last_processed
        # Fresh data, fresh view: "show all" describes a list that no longer
        # exists once the ranking has been recomputed.
        self._expanded = False
        self._render()

    # ---- rendering --------------------------------------------------------

    def _caption_text(self, anchor: Optional[datetime]) -> str:
        # RECENT_K, not a hard-coded 5, so the sentence can only ever promise
        # the membership window the computation actually uses.
        rule = (f"a model is here only while its last {RECENT_K} lots include "
                "one outside its own control limits · ranked by extra failing "
                "units per week")
        if anchor is None:
            return rule
        return f"as of last processed data {anchor:%b %d} · {rule}"

    def _render(self) -> None:
        t, res = self.theme, self._result
        self._more_btn.pack_forget()            # re-packed last, after the rows
        for w in self._rendered:
            try:
                w.destroy()
            except Exception:
                pass
        self._rendered = []
        self._rows = []
        self._chronic_rows = []
        self._heading.configure(
            text=f"FOCUS — drifting now, biggest first ({len(res.focus)})")
        self._caption.configure(text=self._caption_text(res.anchor))

        if not res.focus:
            # Same copy as the zone this replaced: an empty list must say WHEN
            # it was empty, or it reads as "the app didn't run".
            stamp = self._last_processed or res.anchor
            when = stamp.strftime("%Y-%m-%d") if stamp else "—"
            lbl = ctk.CTkLabel(self._body, anchor="w",
                               text=f"All models within tolerance — last processed {when}.",
                               font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY)
            lbl.pack(side="top", fill="x", pady=t.SPACE_LG)
            self._rendered.append(lbl)
        else:
            shown = res.focus if self._expanded else res.focus[:FOCUS_CAP]
            for rank, entry in enumerate(shown, start=1):
                row = _FocusRow(self._body, entry=entry, rank=rank,
                                anchor=res.anchor, theme=t, on_click=self._cb)
                row.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
                self._rows.append(row)
                self._rendered.append(row)
            if len(res.focus) > FOCUS_CAP:
                hidden = len(res.focus) - FOCUS_CAP
                self._more_btn.configure(
                    text=(f"show only the top {FOCUS_CAP}" if self._expanded else
                          f"+ {hidden} more models with smaller signals — show all"))
                self._more_btn.pack(side="top", fill="x", pady=(t.SPACE_XS, 0))

        if res.chronic:
            # A different problem, said out loud: these models are bad on
            # average but IN control, so nothing changed today. Re-alarming on
            # them every morning is how a list loses its audience.
            self._chronic_heading.configure(
                text=f"CHRONICALLY HIGH — stable, different problem ({len(res.chronic)})")
            self._chronic_heading.pack(side="top", fill="x",
                                       pady=(t.SPACE_MD, t.SPACE_XS))
            self._chronic_body.pack(side="top", fill="x")
            for entry in res.chronic:
                row = _ChronicRow(self._chronic_body, entry=entry, theme=t,
                                  on_click=self._cb)
                row.pack(side="top", fill="x", pady=1)
                self._chronic_rows.append(row)
                self._rendered.append(row)
        else:
            self._chronic_heading.pack_forget()
            self._chronic_body.pack_forget()

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._render()


class _FocusRow(ctk.CTkFrame):
    """One ranked model: rank, name, when, verdict, evidence, and its picture."""

    def __init__(self, master, *, entry: FocusEntry, rank: int,
                 anchor: Optional[datetime], theme: ThemeManager,
                 on_click: Callable[[str, str], None]):
        super().__init__(master, fg_color=theme.CARD,
                         corner_radius=theme.RADIUS_MD)
        self.theme = theme
        self.entry = entry
        self.rank = rank
        self._cb = on_click
        self.texts = focus_row_texts(entry, anchor)
        # The chart's own draw helper — the row cannot mark a different lot.
        self.params = spc_draw_params(entry.series)
        self.fig: Optional[Figure] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self._clickables: List = []
        self._build()

    def _build(self) -> None:
        t, x = self.theme, self.texts
        num = ctk.CTkLabel(self, text=str(self.rank), width=26, anchor="nw",
                           font=t.font(t.SIZE_HEADING, "bold"),
                           text_color=t.TEXT_SECONDARY)
        num.pack(side="left", fill="y", padx=(t.SPACE_SM, 0), pady=t.SPACE_SM)

        # Sparkline packed before the text column so the picture keeps its fixed
        # 260px and the words take whatever is left.
        self.fig = Figure(figsize=(SPARK_W / SPARK_DPI, SPARK_H / SPARK_DPI),
                          dpi=SPARK_DPI, facecolor=t.CARD)
        ax = self.fig.add_subplot(111)
        _draw_sparkline(ax, self.params, t)
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.04)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        spark = self.canvas.get_tk_widget()
        spark.configure(width=SPARK_W, height=SPARK_H, highlightthickness=0,
                        bg=t.CARD)
        spark.pack(side="right", padx=t.SPACE_SM, pady=t.SPACE_SM)
        # draw(), not draw_idle(): these canvases are disposable — a refresh
        # destroys them — and draw_idle leaves an `after_idle` pointing at a
        # canvas that may already be gone (a Tcl background error nobody can
        # trace). Seven sparklines render in ~150 ms, inside one refresh.
        self.canvas.draw()

        col = ctk.CTkFrame(self, fg_color="transparent")
        col.pack(side="left", fill="both", expand=True, padx=(t.SPACE_SM, 0),
                 pady=t.SPACE_SM)
        top = ctk.CTkFrame(col, fg_color="transparent")
        top.pack(side="top", fill="x")
        name = ctk.CTkLabel(top, text=x["title"], anchor="w",
                            font=t.font(t.SIZE_HEADING, "bold"),
                            text_color=t.TEXT_PRIMARY)
        name.pack(side="left")
        when = ctk.CTkLabel(top, text=x["when"], anchor="e",
                            font=t.font(t.SIZE_CAPTION),
                            text_color=t.TEXT_SECONDARY)
        when.pack(side="right")
        line1 = ctk.CTkLabel(col, text=x["line1"], anchor="w", justify="left",
                             font=t.font(t.SIZE_BODY), text_color=t.TEXT_PRIMARY)
        line1.pack(side="top", fill="x")
        line2 = ctk.CTkLabel(col, text=x["line2"], anchor="w", justify="left",
                             font=t.font(t.SIZE_CAPTION),
                             text_color=t.TEXT_SECONDARY)
        line2.pack(side="top", fill="x")

        # Bind the widgets we created — explicitly, not by walking the tree:
        # CTk widgets forward .bind() to their own internal canvas/label, so a
        # recursive walk binds those a second time and fires the click twice.
        self._clickables = [self, num, spark, col, top, name, when, line1, line2]
        for w in self._clickables:
            w.bind("<Button-1>", lambda _e: self._on_click(), add="+")

    def _on_click(self) -> None:
        # Deep-link to the model page on the metric that put it on this list.
        self._cb(self.entry.model, self.entry.series.metric)

    def destroy(self):
        """Release the sparkline WITH the row (parents call this on children)."""
        try:
            if self.canvas is not None:
                self.canvas.get_tk_widget().destroy()
        except Exception:
            pass
        try:
            if self.fig is not None:
                self.fig.clf()
                import matplotlib.pyplot as plt
                plt.close(self.fig)
        except Exception:
            pass
        self.canvas = None
        super().destroy()


class _ChronicRow(ctk.CTkFrame):
    """One known-bad-but-stable model: one line, no sparkline, same click."""

    def __init__(self, master, *, entry: FocusEntry, theme: ThemeManager,
                 on_click: Callable[[str, str], None]):
        super().__init__(master, fg_color=theme.SURFACE,
                         corner_radius=theme.RADIUS_SM)
        self.theme = theme
        self.entry = entry
        self._cb = on_click
        t = theme
        name = ctk.CTkLabel(self, text=entry.model, anchor="w",
                            font=t.font(t.SIZE_BODY, "bold"),
                            text_color=t.TEXT_PRIMARY)
        name.pack(side="left", padx=(t.SPACE_SM, t.SPACE_XS), pady=2)
        detail = ctk.CTkLabel(self, text=f"· {entry.verdict} · {entry.sub_line}",
                              anchor="w", font=t.font(t.SIZE_CAPTION),
                              text_color=t.TEXT_SECONDARY)
        detail.pack(side="left", fill="x", expand=True, padx=(0, t.SPACE_SM))
        self._clickables = [self, name, detail]
        for w in self._clickables:
            w.bind("<Button-1>", lambda _e: self._on_click(), add="+")

    def _on_click(self) -> None:
        self._cb(self.entry.model, self.entry.series.metric)
