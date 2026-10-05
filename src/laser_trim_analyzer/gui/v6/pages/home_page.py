"""The Overview -- the landing page (key "home"; Graphite redesign, 2026-10-02; option B, 2026-10-04).

Specs: docs/superpowers/specs/2026-10-02-graphite-redesign-design.md, and
2026-10-04-option-b-design.md -- James, with Task Manager TMOG open: "now this is an example of a
finished peice of solftware"; shown three versions on his own data, "i like B": a LIST of models
with the selected model's DETAIL beside it. TMOG is the bar for finish, not a template.

Top to bottom:
  * the last finished run's one-line summary, quietly (`set_run_summary`; a run lands here), and
    under it "See the run ›" -- the Process page and that run's tally, without starting another;
  * the quiet data-health notices (final tests graded before the ignore-window fix, files skipped
    as unreadable) and, in the check colour, any part of this page that could not be loaded;
  * ONE header line: "13 models need a look · $X lost at final test in the last 90 days · newest
    file 29 Sep 2026" (overview_data.header_line: the dollars the Company trends page used to hide,
    finish item 13; never a count or a sum it does not have);
  * "Yield by laser — last 12 months": the Company trends chart itself as a compact strip, always
    visible, above the list (James, 2026-10-04: "on the overveiw screen i no longer have each laser
    charted overall?");
  * the rule that puts a model on the list (`overview_data.CARD_RULE`);
  * the LIST (left, scrolls on its own): "Needs a look (N)" -- the fail-rate list united with the
    drift watch's flags -- then "Everything else (N)", every other active model, busiest first.
    Each section names its columns (James, 2026-10-04, of the line and the bare % that were here:
    "these little charts and % dont mean anything?" -- he picked this from mockups on his data).
    Each row: the model over its units (and its hand-trim / final-test tags); WHAT CHANGED -- a
    card's reason ("A recent run failed 74% (usually 36%)"), a row's 90 days against the year
    before; twelve months of pass % as bars on one 0-100 scale (micro_charts.MonthBars), the months
    the 90 days touch bright; the 90-day pass %. The selected row is ELEVATED with a border. At its end, "Other models on file (N) ▸" and
    "Inactive models (N) ▸", folded, each expanding in place: every model on file is somewhere on
    this page -- labelled, never hidden (F5);
  * the DETAIL (right) of the selected row -- the first card on load; a click selects, a double
    click opens -- the model in the title face, its status word, a pass meter with "was", why it is
    here (in the fail colour), twelve months of pass %, its facts (units, lasers, the signal, the
    dollars lost at final test, the newest file) and "Open full page ›": its Summary, charting the
    signal a card names;
  * three quiet links: "All findings", "Company trends", "Process a specific folder".

The list and the detail fill the window below the strip (`_fit_split`), never shorter than
SPLIT_MIN: when the notices on top leave less than that, the page scrolls instead.

Every number comes from ONE loader, `gui/v6/overview_data.load_overview`, on a worker thread;
this page only draws what it is handed. Three states, never confused: loading (nothing has landed
yet -- no count, "Loading…"), loaded, and failed (each failed part named in a banner; a count or a
sum it does not have is never printed -- never "0 models need a look" over a crash).

Thread discipline (CLAUDE.md rule 5): the worker posts back through `safe_after`; Tk is touched
on the Tk thread only.
"""
import logging
import threading
import tkinter
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

import customtkinter as ctk

from laser_trim_analyzer.core.activity import activity_unknown_notice
from laser_trim_analyzer.core.ft_regrade import legacy_ft_notice
from laser_trim_analyzer.core.ingest_run import unreadable_notice
from laser_trim_analyzer.core.models import laser_label
from laser_trim_analyzer.gui.v6 import formats
from laser_trim_analyzer.gui.v6 import overview_data as od
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets import blocks
from laser_trim_analyzer.gui.v6.widgets.company_trend_chart import CompanyTrendChart
from laser_trim_analyzer.gui.v6.widgets.micro_charts import MonthBars, MonthChart, PassMeter
from laser_trim_analyzer.ml.drift_types import metric_label

logger = logging.getLogger(__name__)

# Sizes in CustomTkinter's units.
LIST_WIDTH = 480          # the list pane, its scrollbar included: room for a reason beside the bars
SPLIT_MIN = 260           # the list and the detail never shorter: the page scrolls instead
# A line of text is as tall as its text: CustomTkinter's default label is 28 units whatever its
# font, which made each list row 68 px and pushed the detail's last line below the window.
FIT = 0
# A row's columns: model | what changed (takes the rest, wraps) | the month bars | the 90-day %.
# Fixed widths, so the reasons, the bars and the percentages line up down the list.
MODEL_COLUMN = 88         # the model, its units, its tags ("hand trim" is 84 wide)
BARS_SIZE = (84, 26)      # twelve bars
PCT_COLUMN = 52           # the 90-day pass % ("100%", and its column's name "90 days")
# The column names (sentence case: test_blocks lints shouting).
HEAD_MODEL = "Model"
HEAD_WHY = "What changed"                  # Needs a look: why the model is there
HEAD_YEAR = "On the year before"           # Everything else: its 90 days against the year before
HEAD_PASS = "Pass %"                       # over the last two: the bars and the % are both pass %
HEAD_BARS = "12 months"
HEAD_PCT = "90 days"
# A reason's "before → after" kept on one line: wrapped, "→ 6,226" sat alone on the next.
_KEEP_TOGETHER = (" → ", "\u00a0→\u00a0")
FOLDED_COLUMNS = 1        # the folded lines' models, one to a line in the list's width
# The yield chart as a strip: the Company trends chart as it is, only shorter (its figure is 2.6 in)
STRIP_INCHES = 1.9
TREND_TITLE = "Yield by laser — last 12 months"

# The detail's facts, in order. "Signal" only for a model on the list's first part (a card).
FACT_UNITS = "Units, last 90 days"
FACT_LASERS = "Lasers"
FACT_SIGNAL = "Signal"
FACT_MONEY = "$ lost at final test, 90 days"
FACT_NEWEST = "Newest file"
FACTS = (FACT_UNITS, FACT_LASERS, FACT_SIGNAL, FACT_MONEY, FACT_NEWEST)

FAILED_NOTE = "Could not be worked out — the notice above says what failed."

_TONE = {"up": "PASS_FG", "down": "FAIL_FG", "steady": "TEXT_SECONDARY", "new": "TEXT_SECONDARY"}

Item = object             # an overview_data.Card or Row


class HomePage(PageBase):
    page_title = "Overview"

    def __init__(self, master, *, theme, app, page_title="Overview"):
        self._ov: Optional[od.Overview] = None      # what is on screen; None until a load lands
        self._card_rows: List[_ListRow] = []
        self._other_rows: List[_ListRow] = []
        self._selected: Optional[str] = None        # the model whose detail is shown
        self._quiet_open = False
        self._inactive_open = False
        self._split_height: Optional[int] = None
        # Reload generation (the Model page's _reload_gen pattern, facelift F4): a load's apply is
        # dropped unless it is still the newest -- workers finish in any order. Tk thread only.
        self._reload_gen = 0
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    # ---- construction ------------------------------------------------------
    def build_content(self, parent):
        t = self.theme
        # The page scrolls only when the notices on top leave the panes less than SPLIT_MIN.
        self._body = body = _PageScroll(parent, fg_color="transparent")
        body.pack(side="top", fill="both", expand=True)

        # The lines at the top, each packed only while it has something to say (_place_top).
        self._run_line = blocks.banner(body, t, "", tone="quiet", wrap_to=body)
        # The run's own tally is on the Process page. The blue button would start a NEW run and
        # wipe it (final review, 2026-10-02) -- this only opens the page.
        self._run_link = blocks.link_button(body, t, "See the run ›", self._open_process)
        self._legacy_ft_label = blocks.banner(body, t, "", tone="quiet", wrap_to=body)
        self._unreadable_label = blocks.banner(body, t, "", tone="quiet", wrap_to=body)
        self._load_banner = blocks.banner(body, t, "", wrap_to=body)        # check tone

        # The one header line: "Loading…" until a load lands -- never a count before it has looked.
        self._headline = ctk.CTkLabel(body, text="Loading…", anchor="w", justify="left", height=FIT,
                                      font=t.font(t.SIZE_HEADING, "bold"), text_color=t.TEXT_PRIMARY)
        self._headline.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        blocks.wrap_to_width(self._headline, body)          # built once with the page: binds once

        # Each laser and the company, month by month: the Company trends chart itself, as a strip.
        # Its place is held from the start; the chart is packed once a load has landed -- "Loading…"
        # until then, never "No trim data" before it has looked.
        self._trend_box = ctk.CTkFrame(body, fg_color="transparent")
        self._trend_box.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        self._trend_heading = ctk.CTkLabel(self._trend_box, text=TREND_TITLE, anchor="w", height=FIT,
                                           font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY)
        self._trend_heading.pack(side="top", fill="x")
        self._trend_note = ctk.CTkLabel(self._trend_box, text="Loading…", anchor="w",
                                        font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY)
        self._trend_note.pack(side="top", fill="x")
        self._trend_chart = CompanyTrendChart(self._trend_box, theme=t)
        _as_strip(self._trend_chart)

        # What puts a model on the list's first part, said where the list is read.
        self._need_rule = ctk.CTkLabel(body, text=od.CARD_RULE, anchor="w", justify="left",
                                       height=FIT, font=t.font(t.SIZE_CAPTION),
                                       text_color=t.TEXT_SECONDARY)
        self._need_rule.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        blocks.wrap_to_width(self._need_rule, body)

        # The two panes: the list (left, a fixed width) and the detail (right, the rest).
        self._split = ctk.CTkFrame(body, fg_color="transparent")
        self._split.pack(side="top", fill="x")
        self._split.grid_columnconfigure(0, minsize=self._split._apply_widget_scaling(LIST_WIDTH))
        self._split.grid_columnconfigure(1, weight=1)
        self._split.grid_rowconfigure(0, weight=1)
        inner = SPLIT_MIN - 2 * _pane_inset(t)
        self._list = ctk.CTkScrollableFrame(
            self._split, width=LIST_WIDTH - 2 * _pane_inset(t) - 16, height=inner, fg_color=t.CARD,
            border_color=t.BORDER, border_width=1, corner_radius=t.RADIUS_LG)
        self._list.grid(row=0, column=0, sticky="nsew")
        self._detail = _Detail(self._split, t, height=inner, on_open=self._open_selected)
        self._detail.grid(row=0, column=1, sticky="nsew", padx=(t.SPACE_MD, 0))
        body.panes = (self._list, self._detail)
        for frame in (body, self._list, self._detail):
            _scrollbar_only_when_needed(frame)
        self._build_list(self._list)

        self._links = ctk.CTkFrame(body, fg_color="transparent")
        self._links.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))
        self._findings_link = blocks.link_button(self._links, t, "All findings", self._open_findings)
        self._findings_link.pack(side="left")
        self._trends_link = blocks.link_button(self._links, t, "Company trends", self._open_trends)
        self._trends_link.pack(side="left", padx=(t.SPACE_LG, 0))
        # The blue button starts the remembered-folder run; a folder that is not on the list is
        # processed from the Process page, reached here without starting anything.
        self._process_link = blocks.link_button(self._links, t, "Process a specific folder",
                                                self._open_process)
        self._process_link.pack(side="left", padx=(t.SPACE_LG, 0))

        # The panes reach the foot of the window: refitted when the window's size changes and when
        # anything above them changes height (which moves the split: <Configure> fires on a move).
        body._parent_canvas.bind("<Configure>", self._fit_split, add="+")
        self._split.bind("<Configure>", self._fit_split, add="+")

    def _build_list(self, pane) -> None:
        t = self.theme
        pad = dict(padx=t.SPACE_SM)
        self._need_heading = ctk.CTkLabel(pane, text="Needs a look", anchor="w", height=FIT,
                                          font=t.font(t.SIZE_BODY, "bold"), text_color=t.TEXT_PRIMARY)
        self._need_heading.pack(side="top", fill="x", pady=(t.SPACE_SM, t.SPACE_XS), **pad)
        self._cards_head = _ListHeader(pane, t, HEAD_WHY)          # packed while rows are shown
        self._cards_frame = ctk.CTkFrame(pane, fg_color="transparent")
        self._cards_frame.pack(side="top", fill="x")
        # "Loading…" under each heading until the first load lands -- never a bare heading.
        self._cards_note = ctk.CTkLabel(pane, text="Loading…", anchor="w", justify="left",
                                        height=FIT, font=t.font(t.SIZE_CAPTION),
                                        text_color=t.TEXT_SECONDARY)
        self._cards_note.pack(side="top", fill="x", after=self._cards_frame, **pad)
        blocks.wrap_to_width(self._cards_note, pane, padding=2 * t.SPACE_SM)

        self._others_heading = ctk.CTkLabel(pane, text="Everything else", anchor="w", height=FIT,
                                            font=t.font(t.SIZE_BODY, "bold"),
                                            text_color=t.TEXT_PRIMARY)
        self._others_heading.pack(side="top", fill="x", pady=(t.SPACE_MD, t.SPACE_XS), **pad)
        self._others_head = _ListHeader(pane, t, HEAD_YEAR)
        self._others_frame = ctk.CTkFrame(pane, fg_color="transparent")
        self._others_frame.pack(side="top", fill="x")
        self._others_note = ctk.CTkLabel(pane, text="Loading…", anchor="w", justify="left",
                                         height=FIT, font=t.font(t.SIZE_CAPTION),
                                         text_color=t.TEXT_SECONDARY)
        self._others_note.pack(side="top", fill="x", after=self._others_frame, **pad)
        blocks.wrap_to_width(self._others_note, pane, padding=2 * t.SPACE_SM)

        # "Other models on file (N) ▸": trimmed, but on no card, in no list and not inactive --
        # on file, and otherwise nowhere on this page. Packed once the count is known.
        self._quiet_toggle = ctk.CTkButton(
            pane, text="", command=self._toggle_quiet, fg_color="transparent",
            hover_color=t.ELEVATED, text_color=t.TEXT_SECONDARY, font=t.font(t.SIZE_BODY),
            anchor="w", width=0, height=28)
        self._quiet_list = ctk.CTkFrame(pane, fg_color="transparent")
        # "Inactive models (N) ▸": packed once the count is known; the list below it only while
        # it is open.
        self._inactive_toggle = ctk.CTkButton(
            pane, text="", command=self._toggle_inactive, fg_color="transparent",
            hover_color=t.ELEVATED, text_color=t.TEXT_SECONDARY, font=t.font(t.SIZE_BODY),
            anchor="w", width=0, height=28)
        self._inactive_list = ctk.CTkFrame(pane, fg_color="transparent")

    # ---- data ----------------------------------------------------------------
    def _money_settings(self) -> Tuple[Dict, object]:
        """The config's prices and cost ratio, read on the Tk thread -- and the prices COPIED: the
        worker must not read a dict Settings may be rewriting."""
        am = getattr(getattr(self.app, "config", None), "active_models", None)
        prices = dict(getattr(am, "model_prices", None) or {})
        return prices, getattr(am, "cost_ratio", od.DEFAULT_COST_RATIO)

    def on_show(self):
        """Load on a worker; apply on the Tk thread -- unless a newer load has started since."""
        self._reload_gen += 1
        gen = self._reload_gen
        prices, ratio = self._money_settings()

        def work():
            data = od.load_overview(self.app.db, prices=prices, cost_ratio=ratio)

            def apply():
                if gen == self._reload_gen:          # else a newer load superseded this one
                    self._apply(data)
            self.safe_after(apply)
        threading.Thread(target=work, daemon=True).start()

    def reload_now(self) -> None:
        """Synchronous load + apply (the test path). The newest load: anything still in flight is
        dropped when it lands."""
        self._reload_gen += 1
        prices, ratio = self._money_settings()
        self._apply(od.load_overview(self.app.db, prices=prices, cost_ratio=ratio))

    def _apply(self, ov: od.Overview) -> None:
        """Draw one load. Each part under its own guard (the Model page's _try): a render error in
        one is logged and the others still draw. Tk thread."""
        if ov == self._ov:
            return                   # nothing changed since the last load: nothing to redraw
        self._ov = ov
        for what, draw in (("notices", lambda: self._draw_notices(ov)),
                           ("header line", lambda: self._headline.configure(text=od.header_line(ov))),
                           ("yield chart", lambda: self._draw_trend(ov)),
                           ("list", lambda: self._draw_list(ov)),
                           ("other models line", lambda: self._draw_quiet(ov)),
                           ("inactive line", lambda: self._draw_inactive(ov)),
                           ("detail", lambda: self._draw_detail(ov))):
            try:
                draw()
            except Exception:
                logger.exception("Overview: the %s could not be drawn", what)
        self.after_idle(self._fit_split)

    # ---- the top lines -------------------------------------------------------
    def set_run_summary(self, text: str, *, ok: bool = True) -> None:
        """A finished run's one-line summary, at the very top -- quiet, or in the check colours
        when a folder failed. Stays until the next run replaces it. Tk thread."""
        t = self.theme
        fg, bg = (t.TEXT_SECONDARY, t.CARD) if ok else (t.CHECK, t.CHECK_TINT)
        self._run_line.configure(text=text or "", text_color=fg, fg_color=bg)
        self._place_top()

    def _draw_notices(self, ov: od.Overview) -> None:
        self._legacy_ft_label.configure(text=legacy_ft_notice(ov.legacy_ft))
        self._unreadable_label.configure(text=unreadable_notice(ov.unreadable))
        self._load_banner.configure(text=_failure_text(ov))
        self._place_top()

    def _place_top(self) -> None:
        """The four top lines, in order, each only while it has text -- all above the header."""
        lines = (self._run_line, self._legacy_ft_label, self._unreadable_label, self._load_banner)
        for line in lines + (self._run_link,):
            line.pack_forget()
        for line in lines:
            if line.cget("text"):
                line.pack(side="top", fill="x", pady=(0, self.theme.SPACE_SM),
                          before=self._headline)
        if self._run_line.cget("text"):
            self._run_line.pack_configure(pady=0)
            self._run_link.pack(side="top", anchor="w", pady=(0, self.theme.SPACE_SM),
                                after=self._run_line)

    # ---- the chart of each laser -----------------------------------------------
    def _draw_trend(self, ov: od.Overview) -> None:
        """The Company trends chart, the last 12 months by month. A load that failed draws
        "Unavailable" (the banner names it), never an empty chart."""
        self._trend_note.pack_forget()
        if self._trend_chart.winfo_manager() == "":
            self._trend_chart.pack(side="top", fill="x", pady=(self.theme.SPACE_XS, 0))
        self._trend_chart.set_data(None if od.PART_TREND in ov.failed else (ov.trend or {}),
                                   period_label=od.TREND_PERIOD)

    # ---- the list -------------------------------------------------------------
    def _draw_list(self, ov: od.Overview) -> None:
        count = od.card_count(ov)
        rates_failed = od.PART_RATES in ov.failed
        self._need_heading.configure(
            text="Needs a look" if count is None else f"Needs a look ({count:,})")
        self._others_heading.configure(
            text="Everything else" if rates_failed else f"Everything else ({len(ov.others):,})")
        if count is None and not ov.cards:
            cards_note = FAILED_NOTE
        else:
            cards_note = "" if ov.cards else "No model needs a look."
        if rates_failed:
            others_note = FAILED_NOTE
        elif not ov.others:
            others_note = (f"No other model has a graded trim in the last {od.WINDOW_DAYS} days."
                           if ov.anchor is not None else "No trim files on record yet.")
        else:
            others_note = ""
        pad = self.theme.SPACE_SM
        _show_note(self._cards_note, cards_note, after=self._cards_frame, padx=pad)
        _show_note(self._others_note, others_note, after=self._others_frame, padx=pad)
        # Column names over rows only -- never over "No model needs a look." or a failure.
        for head, items, frame in ((self._cards_head, ov.cards, self._cards_frame),
                                   (self._others_head, ov.others, self._others_frame)):
            if items:
                head.pack(side="top", fill="x", padx=self.theme.SPACE_XS, pady=(0, 2),
                          before=frame)
            else:
                head.pack_forget()

        # The selection first: it rests on what was loaded, never on which rows drew.
        models = [c.model for c in ov.cards] + [r.model for r in ov.others]
        if self._selected not in models:
            self._selected = models[0] if models else None
        for row in self._card_rows + self._other_rows:
            row.destroy()
        self._card_rows, self._other_rows = [], []
        recent = od.window_months(ov.anchor)
        for items, frame, rows in ((ov.cards, self._cards_frame, self._card_rows),
                                   (ov.others, self._others_frame, self._other_rows)):
            for item in items:
                row = _ListRow(frame, self.theme, item, recent=recent,
                               selected=item.model == self._selected, on_select=self._select,
                               on_open=self._open_item)
                row.pack(side="top", fill="x", padx=self.theme.SPACE_XS, pady=(0, 2))
                rows.append(row)

    def _select(self, model: str) -> None:
        """Show `model`'s detail and mark its row. Tk thread."""
        if self._ov is None:
            return
        self._selected = model
        for row in self._card_rows + self._other_rows:
            row.set_selected(row.model == model)
        self._draw_detail(self._ov)

    def _item(self, model: Optional[str]) -> Optional[Item]:
        if self._ov is None or model is None:
            return None
        return next((x for x in list(self._ov.cards) + list(self._ov.others) if x.model == model),
                    None)

    # ---- the detail -------------------------------------------------------------
    def _draw_detail(self, ov: od.Overview) -> None:
        item = self._item(self._selected)
        if item is None:
            if (od.PART_RATES in ov.failed or od.card_count(ov) is None) and not (
                    ov.cards or ov.others):
                self._detail.say(FAILED_NOTE)
            elif ov.anchor is None:
                self._detail.say("No trim files on record yet.")
            else:
                self._detail.say(f"No model has a graded trim in the last {od.WINDOW_DAYS} days.")
            return
        self._detail.show(item, ov, status=status_word(ov, item),
                          colour=_series_colour(self.theme, item), months=od.month_starts(ov.anchor))

    # ---- the other models on file --------------------------------------------------------
    def _draw_quiet(self, ov: od.Overview) -> None:
        if not ov.quiet:
            # Unknown (the banner names why) or none: no line -- never "(0)" for a crash.
            self._quiet_toggle.pack_forget()
            self._quiet_list.pack_forget()
            return
        arrow = "▾" if self._quiet_open else "▸"
        self._quiet_toggle.configure(text=f"Other models on file ({len(ov.quiet):,}) {arrow}")
        if self._quiet_toggle.winfo_manager() == "":
            where = ({"before": self._inactive_toggle} if self._inactive_toggle.winfo_manager()
                     else {})
            self._quiet_toggle.pack(side="top", anchor="w", pady=(self.theme.SPACE_SM, 0),
                                    padx=self.theme.SPACE_XS, **where)
        self._fill_folded(self._quiet_list, ov.quiet, self._quiet_open, after=self._quiet_toggle,
                          none_text="no trim file on record")

    def _toggle_quiet(self) -> None:
        self._quiet_open = not self._quiet_open
        if self._ov is not None:
            self._draw_quiet(self._ov)

    # ---- the inactive line -------------------------------------------------------
    def _draw_inactive(self, ov: od.Overview) -> None:
        if ov.inactive is None or not ov.inactive:
            # Unknown (the banner names it) or nothing inactive: no line -- never "(0)" for a crash.
            self._inactive_toggle.pack_forget()
            self._inactive_list.pack_forget()
            return
        n = len(ov.inactive)
        arrow = "▾" if self._inactive_open else "▸"
        self._inactive_toggle.configure(text=f"Inactive models ({n:,}) {arrow}")
        if self._inactive_toggle.winfo_manager() == "":
            self._inactive_toggle.pack(side="top", anchor="w", pady=(self.theme.SPACE_SM, 0),
                                       padx=self.theme.SPACE_XS)
        self._fill_folded(self._inactive_list, ov.inactive or {}, self._inactive_open,
                          after=self._inactive_toggle)

    def _toggle_inactive(self) -> None:
        self._inactive_open = not self._inactive_open
        if self._ov is not None:
            self._draw_inactive(self._ov)

    def _fill_folded(self, frame, models, is_open: bool, *, after,
                     none_text: str = "no trims on record") -> None:
        """A folded line's list, expanded in place under its toggle: every model with its last
        trim, newest first, in FOLDED_COLUMNS columns."""
        t = self.theme
        for child in frame.winfo_children():
            child.destroy()
        if not is_open:
            frame.pack_forget()
            return
        lines = _inactive_lines(models, none_text)
        per = -(-len(lines) // FOLDED_COLUMNS)                # ceiling division
        for i in range(FOLDED_COLUMNS):
            chunk = lines[i * per:(i + 1) * per]
            if not chunk:
                break
            ctk.CTkLabel(frame, text="\n".join(chunk), anchor="nw", justify="left",
                         font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY
                         ).grid(row=0, column=i, sticky="nw", padx=(t.SPACE_SM, t.SPACE_XL))
        frame.pack(side="top", fill="x", pady=(t.SPACE_XS, t.SPACE_SM), after=after)

    # ---- the panes fill the window ------------------------------------------------
    def _fit_split(self, _event=None) -> None:
        """The list and the detail as tall as the page's view leaves them below the strip, never
        under SPLIT_MIN. Real pixels in, CustomTkinter units out (wrap_to_width's rule)."""
        try:
            view = self._body._parent_canvas.winfo_height()
            if view <= 1:
                return                          # not laid out yet: its <Configure> follows
            split = self._split
            below = (self._links.winfo_reqheight()
                     + split._apply_widget_scaling(self.theme.SPACE_SM) + 2)
            height = max(SPLIT_MIN, int(split._reverse_widget_scaling(view - split.winfo_y() - below)))
            if height == self._split_height:
                return
            self._split_height = height
            inner = height - 2 * _pane_inset(self.theme)
            for pane in (self._list, self._detail):
                pane.configure(height=inner)
        except Exception:                       # destroyed first (teardown order)
            logger.debug("Overview: the panes could not be fitted", exc_info=True)

    # ---- routing -------------------------------------------------------------------
    def _open_card(self, card: od.Card) -> None:
        """Its Summary, charting the signal its reason names first (final review, 2026-10-02: a
        fail-rate card opened on whatever the previous model had charted, or on History)."""
        self.app.set_model_route(card.model, focus_metric=card.metric, tab="summary")
        self.app.show_page("model")

    def _open_model(self, model: str) -> None:
        self.app.set_model_route(model, tab="summary")
        self.app.show_page("model")

    def _open_item(self, model: str) -> None:
        item = self._item(model)
        if isinstance(item, od.Card):
            self._open_card(item)
        elif item is not None:
            self._open_model(item.model)

    def _open_selected(self) -> None:
        self._open_item(self._selected)

    def _open_process(self) -> None:
        self.app.show_page("process")             # opens it -- never starts a run

    def _open_findings(self) -> None:
        self.app.show_page("findings")

    def _open_trends(self) -> None:
        self.app.show_page("dashboard")


# ---- the words (pure) -------------------------------------------------------------------------

def _failure_text(ov: od.Overview) -> str:
    """Every part that failed, named -- or "" when everything loaded."""
    lines = []
    parts = [f"{part} ({why})" for part, why in ov.failed.items() if part != od.PART_ACTIVITY]
    if parts:
        lines.append("⚠ Could not load: " + "; ".join(parts) + ". Those parts of this page are "
                     "missing — this is an error, not an all-clear. The log has the details.")
    if od.PART_ACTIVITY in ov.failed:
        lines.append("⚠ " + activity_unknown_notice(ov.failed[od.PART_ACTIVITY]))
    return "\n".join(lines)


def _inactive_lines(inactive, none_text: str = "no trims on record") -> List[str]:
    """"8150 · last trimmed Mar 2016", newest first; a model with no date last, with `none_text`
    ("no trims on record" for an inactive one: laser files, never cut; "no trim file on record"
    for one of the other models: smoothness records only)."""
    dated = sorted(((m, d) for m, d in inactive.items() if d is not None),
                   key=lambda md: (md[1], md[0]), reverse=True)
    never = sorted(m for m, d in inactive.items() if d is None)
    return ([f"{m} · last trimmed {formats.month(d)}" for m, d in dated]
            + [f"{m} · {none_text}" for m in never])


def status_word(ov: od.Overview, item) -> Optional[str]:
    """"Drifting" for a model on the list's first part; "Steady" for the rest -- but only while
    both card sources loaded: with one down, a model not on a card may be drifting unseen."""
    if isinstance(item, od.Card):
        return "Drifting"
    return "Steady" if od.card_count(ov) is not None else None


def _units_text(item) -> str:
    if item.units is None:
        return "— units"                       # its read failed: no count
    return f"{item.units:,} unit" + ("" if item.units == 1 else "s")


def _row_tags(item) -> List[str]:
    """The tags under a row's units: "hand trim" (its laser PASS is hand-trim workload, not
    yield), "final test" (no trim to show: its numbers are final test's)."""
    return ([w for w, on in (("hand trim", item.hand_trim),
                             ("final test", getattr(item, "final_test", False))) if on])


def _what_changed(theme, item) -> Tuple[str, str, str]:
    """(first line, its colour, the lines under it) for a row's middle column. A card: its signal
    in the fail colour, then what qualifies it -- "usually 36%", another signal, "still passing" --
    one per line, quiet (the mockup James picked, 2026-10-04). A row: its 90 days against the year
    before ("down 6 pts", "steady", "new") in that change's colour."""
    if isinstance(item, od.Card):
        lines = [line.replace(*_KEEP_TOGETHER) for line in (list(item.why) or [item.reason])]
        return lines[0], theme.FAIL_FG, "\n".join(lines[1:])
    return item.trend, getattr(theme, _TONE.get(item.tone, "TEXT_SECONDARY")), ""


def _trend_sentence(row: od.Row) -> str:
    if row.tone == "new":
        return "new — nothing graded in the year before"
    return f"{row.trend} on the year before"


def _facts(ov: od.Overview, item) -> Dict[str, Optional[str]]:
    """The detail's facts; a value of None is a fact this item does not have (no line drawn)."""
    if item.units is None:
        units = "—"
    else:
        units = f"{item.units:,}" + (" · final test" if getattr(item, "final_test", False) else "")
    if item.lasers is None:
        lasers = "—"                            # the pass rates failed: the banner says so
    elif not item.lasers:
        lasers = f"none in the last {od.WINDOW_DAYS} days"
    else:
        lasers = ", ".join(laser_label(s) for s in item.lasers)
    signal = None
    if isinstance(item, od.Card):
        signal = metric_label(item.metric) if item.metric else "—"
    return {FACT_UNITS: units, FACT_LASERS: lasers, FACT_SIGNAL: signal,
            FACT_MONEY: od.money_text(ov, item), FACT_NEWEST: formats.day(item.newest)}


def _series_colour(theme, item) -> str:
    """A model's line: its first laser's colour (laser 1, 2, 3 order); a model with no trim to
    show -- its numbers are final test's -- the neutral reference colour."""
    lasers = getattr(item, "lasers", None)
    if lasers and not getattr(item, "final_test", False):
        return theme.series_color(lasers[0])
    return theme.CHART_REFERENCE


# ---- building blocks ---------------------------------------------------------------------------

def _pane_inset(theme) -> int:
    """How far a pane's rounded border insets its content (CTkScrollableFrame grids its canvas
    that far in: corner radius + border width)."""
    return theme.RADIUS_LG + 1


def _as_strip(chart: CompanyTrendChart) -> None:
    """The Company trends chart, unedited, as a strip STRIP_INCHES tall: its figure and its Tk
    canvas both -- a figure resized alone is laid back out to the canvas's old 2.6 in on the next
    <Configure> (no figure manager to forward the size to)."""
    fig = chart._fig
    fig.set_size_inches(fig.get_size_inches()[0], STRIP_INCHES)
    chart.canvas.get_tk_widget().configure(height=round(STRIP_INCHES * fig.dpi))


def _show_note(label, text: str, *, after, padx: int) -> None:
    """A list note under its rows -- packed only while it has something to say."""
    label.configure(text=text)
    if text:
        label.pack(side="top", fill="x", after=after, padx=padx)
    else:
        label.pack_forget()


def _scrollbar_only_when_needed(frame) -> None:
    """A scrolling frame's bar shown only while there is something to scroll: three panes each
    showing a full-length bar over nothing is clutter, not information. Wraps the canvas's
    yscrollcommand (CustomTkinter 5.2.2 sets it to the bar's own set(), pinned). With the bar
    gone the canvas keeps the same inset on the right as on the left -- CustomTkinter grids it
    flush against the bar's column, and alone it would cover the pane's right border."""
    bar, canvas = frame._scrollbar, frame._parent_canvas

    def inset() -> float:
        outer = frame._parent_frame
        return frame._apply_widget_scaling(outer.cget("corner_radius") + outer.cget("border_width"))

    def on_scroll(first, last) -> None:
        bar.set(first, last)
        needed = float(first) > 0.0 or float(last) < 1.0
        shown = bar.winfo_manager() != ""
        if needed and not shown:
            bar.grid()
            canvas.grid_configure(padx=(inset(), 0))
        elif shown and not needed:
            bar.grid_remove()
            canvas.grid_configure(padx=(inset(), inset()))

    canvas.configure(yscrollcommand=on_scroll)


def _bind_click(widget, on_click: Callable[[], None],
                on_double: Optional[Callable[[], None]] = None) -> None:
    """One click (and a double click) anywhere on `widget` or inside it. CTk widgets' own .bind()
    already reaches their internal canvas/label, so those are skipped on the way down (blocks.row's
    rule: binding them twice fires one click twice)."""
    widget.bind("<Button-1>", lambda _e: on_click(), add="+")
    if on_double is not None:
        widget.bind("<Double-Button-1>", lambda _e: on_double(), add="+")
    try:
        widget.configure(cursor="hand2")
    except Exception:                 # some CTk internals refuse a cursor; clicks still work
        pass
    internals = {getattr(widget, n) for n in ("_canvas", "_label") if hasattr(widget, n)}
    for child in tkinter.Misc.winfo_children(widget):
        if child not in internals:
            _bind_click(child, on_click, on_double)


class _PageScroll(ctk.CTkScrollableFrame):
    """The page's own scrolling frame. CustomTkinter scrolls EVERY scrolling frame under the
    pointer, so a wheel over the list also scrolled the page beneath it whenever both could
    scroll: over a pane that has something to scroll, only that pane does."""

    panes: tuple = ()

    def check_if_master_is_canvas(self, widget):
        try:
            for pane in self.panes:
                if (pane.check_if_master_is_canvas(widget)
                        and pane._parent_canvas.yview() != (0.0, 1.0)):
                    return False
        except Exception:             # a widget CustomTkinter does not know: its own rule below
            pass
        return super().check_if_master_is_canvas(widget)


def _grid_columns(frame, theme) -> None:
    """A row's (and a header's) four columns: fixed but the second, which takes the rest. The bars'
    column holds the gap before them too -- a row's bars ask for it, a header's name does not, and
    the two would part by that much (measured: 8 px)."""
    frame.grid_columnconfigure(0, minsize=frame._apply_widget_scaling(MODEL_COLUMN))
    frame.grid_columnconfigure(1, weight=1)
    frame.grid_columnconfigure(2, minsize=frame._apply_widget_scaling(BARS_SIZE[0] + theme.SPACE_SM))
    frame.grid_columnconfigure(3, minsize=frame._apply_widget_scaling(PCT_COLUMN))


class _ListHeader(ctk.CTkFrame):
    """A section's column names, on the rows' own columns, over a thin line -- "Pass %" over the
    last two ("12 months" · "90 days"): side by side, "Pass by month" and "90 days" read as one
    phrase (seen on the work data's copy, 2026-10-04)."""

    def __init__(self, master, theme, why: str):
        t = theme
        super().__init__(master, fg_color="transparent")
        _grid_columns(self, t)
        pad = t.SPACE_SM
        font, colour = t.font(t.SIZE_CAPTION), t.TEXT_SECONDARY

        def name(text, row, column, **grid):
            label = ctk.CTkLabel(self, text=text, font=font, text_color=colour, height=FIT,
                                 anchor=grid.pop("anchor", "w"))
            label.grid(row=row, column=column, **grid)
            return label
        self.model = name(HEAD_MODEL, 1, 0, sticky="sw", padx=(pad, 0))
        self.why = name(why, 1, 1, sticky="sw", padx=(pad, 0))
        self.passed = name(HEAD_PASS, 0, 2, columnspan=2, sticky="w", padx=(pad, 0))
        self.bars = name(HEAD_BARS, 1, 2, sticky="w", padx=(pad, 0))
        self.pct = name(HEAD_PCT, 1, 3, sticky="e", padx=(0, pad), anchor="e")
        ctk.CTkFrame(self, height=1, fg_color=t.BORDER, corner_radius=0).grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(2, 0))


class _ListRow(ctk.CTkFrame):
    """One line of the list, on four columns: the model over its units and tags; what changed (it
    wraps); twelve months of pass % as bars; the 90-day pass %. A click selects it; a double click
    opens its page."""

    def __init__(self, master, theme, item, *, recent: int, selected: bool,
                 on_select: Callable[[str], None], on_open: Callable[[str], None]):
        t = theme
        super().__init__(master, fg_color=t.CARD, border_color=t.CARD, border_width=1,
                         corner_radius=t.RADIUS_MD)
        self.theme = t
        self.item = item
        self.model = item.model
        _grid_columns(self, t)
        pad = t.SPACE_XS + 2
        side = t.SPACE_SM
        ctk.CTkLabel(self, text=item.model, font=t.mono(t.SIZE_BODY), text_color=t.TEXT_PRIMARY,
                     anchor="w", height=FIT).grid(row=0, column=0, sticky="w", padx=(side, 0),
                                                  pady=(pad, 0))
        self._units = ctk.CTkLabel(self, text=_units_text(item), font=t.font(t.SIZE_CAPTION),
                                   text_color=t.TEXT_SECONDARY, anchor="w", height=FIT)
        self._units.grid(row=1, column=0, sticky="w", padx=(side, 0))
        self._tags = [blocks.tag(self, t, word) for word in _row_tags(item)]
        for i, tag in enumerate(self._tags):
            tag.grid(row=2 + i, column=0, sticky="w", padx=(side, 0), pady=(2, 0))
        # Under the model column's last line, a row that takes any height a long reason needs --
        # so the model, its units and its tags stay together at the top. Every cell spans into it.
        stretch = 2 + len(self._tags)
        self.grid_rowconfigure(stretch, weight=1, minsize=self._apply_widget_scaling(pad))
        span = stretch + 1

        # What changed: wraps to whatever width the fixed columns leave it (blocks.wrap_to_width,
        # bound to a frame built -- and destroyed -- with this row, so the binding never piles up).
        first, colour, more = _what_changed(t, item)
        self._why_box = ctk.CTkFrame(self, fg_color="transparent")
        self._why_box.grid(row=0, column=1, rowspan=span, sticky="new", padx=(side, 0),
                           pady=(pad, pad))
        self._why = ctk.CTkLabel(self._why_box, text=first, font=t.font(t.SIZE_CAPTION),
                                 text_color=colour, anchor="w", justify="left", height=FIT)
        self._why.pack(side="top", fill="x")
        blocks.wrap_to_width(self._why, self._why_box)
        self._why_more: Optional[ctk.CTkLabel] = None
        if more:
            self._why_more = ctk.CTkLabel(self._why_box, text=more, font=t.font(t.SIZE_CAPTION),
                                          text_color=t.TEXT_SECONDARY, anchor="w", justify="left",
                                          height=FIT)
            self._why_more.pack(side="top", fill="x")
            blocks.wrap_to_width(self._why_more, self._why_box)

        gap = self._apply_widget_scaling(side)              # a plain canvas: scaled here
        top = self._apply_widget_scaling(pad)
        if item.months:
            self._bars = MonthBars(self, t, item.months, recent=recent, bg=t.CARD,
                                   width=BARS_SIZE[0], height=BARS_SIZE[1])
        else:
            # Its read failed (the banner names it): a dash, as its units and its % say.
            self._bars = ctk.CTkLabel(self, text="—", font=t.font(t.SIZE_CAPTION),
                                      text_color=t.TEXT_SECONDARY, height=FIT)
        self._bars.grid(row=0, column=2, rowspan=span, sticky="nw", padx=(gap, 0), pady=(top, top))
        self._pct = ctk.CTkLabel(self, text=od.pct_text(item.pass_pct),
                                 font=t.mono(t.SIZE_BODY, "bold"), text_color=t.TEXT_PRIMARY,
                                 anchor="e", height=FIT)
        self._pct.grid(row=0, column=3, sticky="ne", padx=(0, side), pady=(pad, 0))
        self.set_selected(selected)
        _bind_click(self, lambda m=item.model: on_select(m), lambda m=item.model: on_open(m))

    def set_selected(self, on: bool) -> None:
        t = self.theme
        fill = t.ELEVATED if on else t.CARD
        self.configure(fg_color=fill, border_color=t.BORDER if on else t.CARD)
        if isinstance(self._bars, MonthBars):
            self._bars.set_background(fill)


class _Detail(ctk.CTkScrollableFrame):
    """The selected model: its name in the title face, its status word (and hand-trim tag) and
    "Open full page ›" on that first line -- never below the facts, where a short window would
    scroll it out of sight; its pass % large, "was", and a meter; why it is on the list; its
    twelve months; its facts. Built once; `show` fills it, `say` replaces it with one line."""

    CHART_HEIGHT = 100

    def __init__(self, master, theme, *, height: int, on_open: Callable[[], None]):
        t = theme
        super().__init__(master, height=height, fg_color=t.CARD, border_color=t.BORDER,
                         border_width=1, corner_radius=t.RADIUS_LG)
        self.theme = t
        side = t.SPACE_LG
        px = self._apply_widget_scaling          # a plain canvas's padding is scaled here
        self.note = ctk.CTkLabel(self, text="Loading…", anchor="w", justify="left", height=FIT,
                                 font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY)
        self.note.pack(side="top", fill="x", padx=side, pady=side)

        self.head = ctk.CTkFrame(self, fg_color="transparent")
        self.title = ctk.CTkLabel(self.head, text="", anchor="w", font=t.title(t.SIZE_TITLE),
                                  text_color=t.TEXT_PRIMARY, height=FIT)
        self.title.pack(side="left")
        self.status = ctk.CTkLabel(self.head, text="", font=t.font(t.SIZE_CAPTION),
                                   corner_radius=t.RADIUS_SM, padx=7, height=20)
        self.hand = blocks.tag(self.head, t, "hand trim")
        self.open_link = blocks.link_button(self.head, t, "Open full page ›", on_open)
        self.open_link.pack(side="right")

        self._nums = ctk.CTkFrame(self, fg_color="transparent")
        self.pct = ctk.CTkLabel(self._nums, text="—", font=t.mono(t.SIZE_DISPLAY, "bold"),
                                text_color=t.TEXT_PRIMARY, height=FIT)
        self.pct.pack(side="left")
        self.was = ctk.CTkLabel(self._nums, text="", font=t.font(t.SIZE_CAPTION),
                                text_color=t.TEXT_SECONDARY, height=FIT)
        self.was.pack(side="left", padx=(t.SPACE_SM, 0), pady=(t.SPACE_SM, 0))
        self.meter = PassMeter(self._nums, t, None, bg=t.CARD)
        self.meter.pack(side="left", padx=(px(t.SPACE_LG), 0), pady=(px(t.SPACE_XS), 0))

        self.reason = ctk.CTkLabel(self, text="", anchor="w", justify="left", height=FIT,
                                   font=t.font(t.SIZE_BODY), text_color=t.FAIL_FG)
        # Built once with the page and never rebuilt: the binding is made once (wrap_to_width).
        blocks.wrap_to_width(self.reason, self, padding=2 * side)
        self.chart_caption = ctk.CTkLabel(self, text="", anchor="w", height=FIT,
                                          font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY)
        self.chart = MonthChart(self, t, bg=t.CARD, height=self.CHART_HEIGHT)

        self._facts = ctk.CTkFrame(self, fg_color="transparent")
        self._facts.grid_columnconfigure(1, weight=1)
        self.facts: Dict[str, ctk.CTkLabel] = {}
        self._fact_names: Dict[str, ctk.CTkLabel] = {}
        for i, name in enumerate(FACTS):
            self._fact_names[name] = ctk.CTkLabel(self._facts, text=name, anchor="w", height=FIT,
                                                  font=t.font(t.SIZE_CAPTION),
                                                  text_color=t.TEXT_SECONDARY)
            self._fact_names[name].grid(row=i, column=0, sticky="w", padx=(0, t.SPACE_LG),
                                        pady=(0, 2))
            self.facts[name] = ctk.CTkLabel(self._facts, text="", anchor="w", height=FIT,
                                            font=t.font(t.SIZE_BODY), text_color=t.TEXT_PRIMARY)
            self.facts[name].grid(row=i, column=1, sticky="w", pady=(0, 2))

        # (widget, its pack options) in order. CustomTkinter scales a CTk widget's padding itself;
        # the chart -- a plain canvas -- gets its scaled here.
        self._parts = (
            (self.head, dict(fill="x", padx=(side, t.SPACE_SM), pady=(t.SPACE_MD, 0))),
            (self._nums, dict(fill="x", padx=side, pady=(t.SPACE_XS, 0))),
            (self.reason, dict(fill="x", padx=side, pady=(t.SPACE_SM, 0))),
            (self.chart_caption, dict(fill="x", padx=side, pady=(t.SPACE_MD, 0))),
            (self.chart, dict(fill="x", padx=px(side), pady=(px(t.SPACE_XS), 0))),
            (self._facts, dict(fill="x", padx=side, pady=(t.SPACE_MD, side))),
        )
        self._showing = False
        self._shown: Optional[str] = None        # the model drawn now

    def say(self, text: str) -> None:
        """One line instead of a model: loading, nothing to show, or what failed."""
        for widget, _how in self._parts:
            widget.pack_forget()
        self._showing = False
        self._shown = None
        self.note.configure(text=text)
        self.note.pack(side="top", fill="x", padx=self.theme.SPACE_LG, pady=self.theme.SPACE_LG)

    def show(self, item, ov: od.Overview, *, status: Optional[str], colour: str,
             months: List[date]) -> None:
        t = self.theme
        if not self._showing:
            self.note.pack_forget()
            for widget, how in self._parts:
                widget.pack(side="top", **how)
            self._showing = True
        if item.model != self._shown:
            self._parent_canvas.yview_moveto(0.0)     # a new model is read from its top
            self._shown = item.model
        self.title.configure(text=item.model)
        self.status.pack_forget()
        self.hand.pack_forget()
        if status:
            fg, bg = ((t.CHECK, t.CHECK_TINT) if status == "Drifting"
                      else (t.NEUTRAL_FG, t.NEUTRAL_BG))
            self.status.configure(text=status, text_color=fg, fg_color=bg)
            self.status.pack(side="left", padx=(t.SPACE_MD, 0))
        if item.hand_trim:
            self.hand.pack(side="left", padx=(t.SPACE_SM, 0))
        self.pct.configure(text=od.pct_text(item.pass_pct))
        self.was.configure(text=f"was {od.pct_text(item.was_pct)}" if item.was_pct is not None
                           else ("new" if item.units else ""))
        self.meter.set_value(item.pass_pct)
        if isinstance(item, od.Card):
            self.reason.configure(text=item.reason, text_color=t.FAIL_FG)
        else:
            self.reason.configure(text=_trend_sentence(item),
                                  text_color=getattr(t, _TONE.get(item.tone, "TEXT_SECONDARY")))
        what = "Final-test pass % by month" if getattr(item, "final_test", False) else "Pass % by month"
        self.chart_caption.configure(
            text=f"{what}, {formats.month(months[0])} – {formats.month(months[-1])}" if months
            else what)
        # A card whose read failed has no months: say so, never "No graded units" (the review of
        # option B, 2026-10-04). Its meter draws nothing for the same reason (PassMeter).
        self.chart.set_data(item.months, [formats.month(d).split(" ")[0] for d in months],
                            colour, value_text=od.pct_text,
                            empty_text=FAILED_NOTE if item.units is None else None)
        for name, value in _facts(ov, item).items():
            label, shown = self._fact_names[name], self.facts[name]
            if value is None:
                label.grid_remove()
                shown.grid_remove()
            else:
                shown.configure(text=value)
                label.grid()
                shown.grid()

    def shown_facts(self) -> Dict[str, str]:
        """{fact: its value} for every fact drawn now (a test hook, and the render audit's)."""
        return {name: v.cget("text") for name, v in self.facts.items()
                if v.winfo_manager() == "grid"}
