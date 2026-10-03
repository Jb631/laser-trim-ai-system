"""The Overview -- the landing page (key "home"; Graphite redesign, 2026-10-02).

Spec: docs/superpowers/specs/2026-10-02-graphite-redesign-design.md. James: "im just not happy
with the app, there is so much going on its hard to see what is what". This page REMOVES: Home's
ingest card went to the Process page (the top bar's "Process new files" runs it there), its
"Worth changing" list into each model's page, and Triage is retired -- what it showed is here.

Top to bottom:
  * the last finished run's one-line summary, quietly (`set_run_summary`; a run lands here), and
    under it "See the run ›" -- the Process page and that run's tally, without starting another;
  * the quiet data-health notices (final tests graded before the ignore-window fix, files skipped
    as unreadable) and, in the check colour, any part of this page that could not be loaded;
  * "N models need a look" over "Last 90 days · newest file DD Mon YYYY" and the rule that puts a
    model on a card (`overview_data.CARD_RULE`), and the CARDS: the fail-rate list united with the
    drift watch's flags, each saying in one line why it is there (James: "keep all 16 cards (fail
    rate up, or a signal moved), each with its reason"); a click opens its Summary, charting the
    signal its reason names;
  * "Everything else": every other active model, busiest first;
  * "Other models on file (N) ▸" and "Inactive models (N) ▸", collapsed, each expanding in place:
    every model on file is somewhere on this page -- labelled, never hidden (F5);
  * three quiet links: "All findings", "Company trends", "Process a specific folder".

Every number comes from ONE loader, `gui/v6/overview_data.load_overview`, on a worker thread;
this page only draws what it is handed. Three states, never confused: loading (nothing has landed
yet -- no count, "Loading…"), loaded, and failed (each failed part named in a banner; a count it
does not have is never printed -- never "0 models need a look" over a crash).

Thread discipline (CLAUDE.md rule 5): the worker posts back through `safe_after`; Tk is touched
on the Tk thread only.
"""
import logging
import threading
import tkinter
from typing import Callable, List, Optional

import customtkinter as ctk

from laser_trim_analyzer.core.activity import activity_unknown_notice
from laser_trim_analyzer.core.ft_regrade import legacy_ft_notice
from laser_trim_analyzer.core.ingest_run import unreadable_notice
from laser_trim_analyzer.gui.v6 import overview_data as od
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets import blocks

logger = logging.getLogger(__name__)

# Cards: four columns from about 1280 px of window, fewer below (each at least CARD_MIN wide).
MAX_COLUMNS = 4
CARD_MIN = 280
BARS_HEIGHT = 40
# "Everything else": fixed column widths, so the plain rows line up (CustomTkinter units).
LIST_COLUMNS = (180, 110, 70, 140)
INACTIVE_COLUMNS = 3

_TONE = {"up": "PASS_FG", "down": "FAIL_FG", "steady": "TEXT_SECONDARY", "new": "TEXT_SECONDARY"}


class HomePage(PageBase):
    page_title = "Overview"

    def __init__(self, master, *, theme, app, page_title="Overview"):
        self._ov: Optional[od.Overview] = None      # what is on screen; None until a load lands
        self._card_widgets: List[_ModelCard] = []
        self._columns = 0
        self._inactive_open = False
        # Reload generation (the Model page's _reload_gen pattern, facelift F4): a load's apply is
        # dropped unless it is still the newest -- workers finish in any order. Tk thread only.
        self._reload_gen = 0
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    # ---- construction ------------------------------------------------------
    def build_content(self, parent):
        t = self.theme
        # Scrollable: 16 cards, a list of every active model and an expanded inactive list are
        # taller than any window.
        self._body = body = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        body.pack(side="top", fill="both", expand=True)

        # The lines at the top, each packed only while it has something to say (_place_top).
        self._run_line = blocks.banner(body, t, "", tone="quiet", wrap_to=body)
        # The run's own tally is on the Process page. The blue button would start a NEW run and
        # wipe it (final review, 2026-10-02) -- this only opens the page.
        self._run_link = blocks.link_button(body, t, "See the run ›", self._open_process)
        self._legacy_ft_label = blocks.banner(body, t, "", tone="quiet", wrap_to=body)
        self._unreadable_label = blocks.banner(body, t, "", tone="quiet", wrap_to=body)
        self._load_banner = blocks.banner(body, t, "", wrap_to=body)        # check tone

        self._need_heading = ctk.CTkLabel(body, text=od.need_a_look(None), anchor="w",
                                          font=t.font(t.SIZE_HEADING, "bold"),
                                          text_color=t.TEXT_PRIMARY)
        self._need_heading.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))
        self._need_caption = ctk.CTkLabel(body, text="Loading…", anchor="w", justify="left",
                                          font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY)
        self._need_caption.pack(side="top", fill="x")
        self._need_rule = ctk.CTkLabel(body, text=od.CARD_RULE, anchor="w", justify="left",
                                       font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY)
        self._need_rule.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        blocks.wrap_to_width(self._need_rule, body)        # built once with the page: binds once
        self._cards_frame = ctk.CTkFrame(body, fg_color="transparent")
        self._cards_frame.pack(side="top", fill="x", pady=(0, t.SPACE_XL))
        # Bound ONCE, on a frame that lives as long as the page: re-grids the cards when the
        # number of columns the width allows changes.
        self._cards_frame.bind("<Configure>", self._regrid_cards, add="+")
        self._cards_note = ctk.CTkLabel(self._cards_frame, text="", anchor="w", justify="left",
                                        font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY)

        self._others_heading = ctk.CTkLabel(body, text="Everything else", anchor="w",
                                            font=t.font(t.SIZE_HEADING, "bold"),
                                            text_color=t.TEXT_PRIMARY)
        self._others_heading.pack(side="top", fill="x", pady=(0, t.SPACE_XS))
        self._others_frame = ctk.CTkFrame(body, fg_color="transparent")
        self._others_frame.pack(side="top", fill="x")
        # "Loading…" until the first load lands -- never a bare heading over nothing.
        self._others_note = ctk.CTkLabel(body, text="Loading…", anchor="w", justify="left",
                                         font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY)
        self._others_note.pack(side="top", fill="x", after=self._others_frame)

        # "Other models on file (N) ▸": trimmed, but on no card, in no list and not inactive --
        # on file, and otherwise nowhere on this page. Packed once the count is known.
        self._quiet_open = False
        self._quiet_toggle = ctk.CTkButton(
            body, text="", command=self._toggle_quiet, fg_color="transparent",
            hover_color=t.CARD, text_color=t.TEXT_SECONDARY, font=t.font(t.SIZE_BODY),
            anchor="w", width=0, height=28)
        self._quiet_list = ctk.CTkFrame(body, fg_color="transparent")

        # "Inactive models (N) ▸": packed once the count is known; the list below it only while
        # it is open.
        self._inactive_toggle = ctk.CTkButton(
            body, text="", command=self._toggle_inactive, fg_color="transparent",
            hover_color=t.CARD, text_color=t.TEXT_SECONDARY, font=t.font(t.SIZE_BODY),
            anchor="w", width=0, height=28)
        self._inactive_list = ctk.CTkFrame(body, fg_color="transparent")

        self._links = ctk.CTkFrame(body, fg_color="transparent")
        self._links.pack(side="top", fill="x", pady=(t.SPACE_LG, 0))
        self._findings_link = blocks.link_button(self._links, t, "All findings", self._open_findings)
        self._findings_link.pack(side="left")
        self._trends_link = blocks.link_button(self._links, t, "Company trends", self._open_trends)
        self._trends_link.pack(side="left", padx=(t.SPACE_LG, 0))
        # The blue button starts the remembered-folder run; a folder that is not on the list is
        # processed from the Process page, reached here without starting anything.
        self._process_link = blocks.link_button(self._links, t, "Process a specific folder",
                                                self._open_process)
        self._process_link.pack(side="left", padx=(t.SPACE_LG, 0))

    # ---- data ----------------------------------------------------------------
    def on_show(self):
        """Load on a worker; apply on the Tk thread -- unless a newer load has started since."""
        self._reload_gen += 1
        gen = self._reload_gen

        def work():
            data = od.load_overview(self.app.db)

            def apply():
                if gen == self._reload_gen:          # else a newer load superseded this one
                    self._apply(data)
            self.safe_after(apply)
        threading.Thread(target=work, daemon=True).start()

    def reload_now(self) -> None:
        """Synchronous load + apply (the test path). The newest load: anything still in flight is
        dropped when it lands."""
        self._reload_gen += 1
        self._apply(od.load_overview(self.app.db))

    def _apply(self, ov: od.Overview) -> None:
        """Draw one load. Each part under its own guard (the Model page's _try): a render error in
        one is logged and the others still draw. Tk thread."""
        if ov == self._ov:
            return                   # nothing changed since the last load: nothing to redraw
        self._ov = ov
        for what, draw in (("notices", lambda: self._draw_notices(ov)),
                           ("cards", lambda: self._draw_cards(ov)),
                           ("list", lambda: self._draw_others(ov)),
                           ("other models line", lambda: self._draw_quiet(ov)),
                           ("inactive line", lambda: self._draw_inactive(ov))):
            try:
                draw()
            except Exception:
                logger.exception("Overview: the %s could not be drawn", what)

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
        """The four top lines, in order, each only while it has text -- all above the heading."""
        lines = (self._run_line, self._legacy_ft_label, self._unreadable_label, self._load_banner)
        for line in lines + (self._run_link,):
            line.pack_forget()
        for line in lines:
            if line.cget("text"):
                line.pack(side="top", fill="x", pady=(0, self.theme.SPACE_SM),
                          before=self._need_heading)
        if self._run_line.cget("text"):
            self._run_line.pack_configure(pady=0)
            self._run_link.pack(side="top", anchor="w", pady=(0, self.theme.SPACE_SM),
                                after=self._run_line)

    # ---- the cards -------------------------------------------------------------
    def _draw_cards(self, ov: od.Overview) -> None:
        t = self.theme
        self._need_heading.configure(text=od.need_a_look(od.card_count(ov)))
        self._need_caption.configure(text=od.window_caption(ov))
        for w in self._card_widgets:
            w.destroy()
        self._card_widgets = [_ModelCard(self._cards_frame, t, card, self._open_card)
                              for card in ov.cards]
        failed = od.card_count(ov) is None
        self._cards_note.configure(
            text="Could not be worked out — the notice above says what failed."
            if failed and not ov.cards else "")
        self._columns = 0                       # force a fresh grid
        self._regrid_cards()

    def _columns_for_width(self) -> int:
        width = self._cards_frame.winfo_width()
        if width <= 1:
            return MAX_COLUMNS                  # not laid out yet: its <Configure> follows
        unscaled = self._cards_frame._reverse_widget_scaling(width)
        gap = self.theme.SPACE_MD
        return max(1, min(MAX_COLUMNS, int((unscaled + gap) // (CARD_MIN + gap))))

    def _regrid_cards(self, _event=None) -> None:
        columns = self._columns_for_width()
        if columns == self._columns:
            return
        self._columns = columns
        gap = self.theme.SPACE_MD
        for i in range(MAX_COLUMNS):
            self._cards_frame.grid_columnconfigure(i, weight=1 if i < columns else 0,
                                                   uniform="cards" if i < columns else "")
        self._cards_note.grid_forget()
        for i, w in enumerate(self._card_widgets):
            r, c = divmod(i, columns)
            w.grid(row=r, column=c, sticky="nsew",
                   padx=(0 if c == 0 else gap // 2, 0 if c == columns - 1 else gap // 2),
                   pady=(0, gap))
        if self._cards_note.cget("text"):
            self._cards_note.grid(row=0, column=0, columnspan=columns, sticky="w")

    # ---- "Everything else" ------------------------------------------------------
    def _draw_others(self, ov: od.Overview) -> None:
        t = self.theme
        for child in self._others_frame.winfo_children():
            child.destroy()
        if od.PART_RATES in ov.failed:
            note = "Could not be worked out — the notice above says what failed."
        elif not ov.others:
            note = (f"No other model has a graded trim in the last {od.WINDOW_DAYS} days."
                    if ov.anchor is not None else "")
        else:
            note = ""
        self._others_note.configure(text=note)
        if note:
            self._others_note.pack(side="top", fill="x", after=self._others_frame)
        else:
            self._others_note.pack_forget()
        for row in ov.others:
            _ListRow(self._others_frame, t, row, self._open_model).pack(side="top", fill="x")

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
            below = self._inactive_toggle if self._inactive_toggle.winfo_manager() else self._links
            self._quiet_toggle.pack(side="top", anchor="w", pady=(self.theme.SPACE_MD, 0),
                                    before=below)
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
            self._inactive_toggle.pack(side="top", anchor="w", pady=(self.theme.SPACE_MD, 0),
                                       before=self._links)
        self._fill_folded(self._inactive_list, ov.inactive or {}, self._inactive_open,
                          after=self._inactive_toggle)

    def _toggle_inactive(self) -> None:
        self._inactive_open = not self._inactive_open
        if self._ov is not None:
            self._draw_inactive(self._ov)

    def _fill_folded(self, frame, models, is_open: bool, *, after,
                     none_text: str = "no trims on record") -> None:
        """A folded line's list, expanded in place under its toggle: every model with its last
        trim, newest first, in INACTIVE_COLUMNS columns."""
        t = self.theme
        for child in frame.winfo_children():
            child.destroy()
        if not is_open:
            frame.pack_forget()
            return
        lines = _inactive_lines(models, none_text)
        per = -(-len(lines) // INACTIVE_COLUMNS)              # ceiling division
        for i in range(INACTIVE_COLUMNS):
            chunk = lines[i * per:(i + 1) * per]
            if not chunk:
                break
            ctk.CTkLabel(frame, text="\n".join(chunk), anchor="nw", justify="left",
                         font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY
                         ).grid(row=0, column=i, sticky="nw", padx=(0, t.SPACE_XL))
        frame.pack(side="top", fill="x", pady=(t.SPACE_XS, 0), after=after)

    # ---- routing -------------------------------------------------------------------
    def _open_card(self, card: od.Card) -> None:
        """Its Summary, charting the signal its reason names first (final review, 2026-10-02: a
        fail-rate card opened on whatever the previous model had charted, or on History)."""
        self.app.set_model_route(card.model, focus_metric=card.metric, tab="summary")
        self.app.show_page("model")

    def _open_model(self, model: str) -> None:
        self.app.set_model_route(model, tab="summary")
        self.app.show_page("model")

    def _open_process(self) -> None:
        self.app.show_page("process")             # opens it -- never starts a run

    def _open_findings(self) -> None:
        self.app.show_page("findings")

    def _open_trends(self) -> None:
        self.app.show_page("dashboard")


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
    return ([f"{m} · last trimmed {d:%b %Y}" for m, d in dated]
            + [f"{m} · {none_text}" for m in never])


def _bind_click(widget, on_click: Callable[[], None]) -> None:
    """One click anywhere on `widget` or inside it. CTk widgets' own .bind() already reaches
    their internal canvas/label, so those are skipped on the way down (blocks.row's rule: binding
    them twice fires one click twice)."""
    widget.bind("<Button-1>", lambda _e: on_click(), add="+")
    try:
        widget.configure(cursor="hand2")
    except Exception:                 # some CTk internals refuse a cursor; clicks still work
        pass
    internals = {getattr(widget, n) for n in ("_canvas", "_label") if hasattr(widget, n)}
    for child in tkinter.Misc.winfo_children(widget):
        if child not in internals:
            _bind_click(child, on_click)


class _ModelCard(ctk.CTkFrame):
    """One card: the model, its units, its pass % (large, mono) and "was", twelve monthly bars,
    and the reason it is here, in the fail colour. A click anywhere opens its Model page."""

    def __init__(self, master, theme, card: od.Card, on_open: Callable[[od.Card], None]):
        t = theme
        super().__init__(master, fg_color=t.CARD, border_color=t.BORDER, border_width=1,
                         corner_radius=t.RADIUS_LG)
        self.card = card
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=t.SPACE_MD, pady=t.SPACE_MD)
        top = ctk.CTkFrame(inner, fg_color="transparent")
        top.pack(side="top", fill="x")
        ctk.CTkLabel(top, text=card.model, anchor="w", font=t.font(t.SIZE_HEADING, "bold"),
                     text_color=t.TEXT_PRIMARY).pack(side="left")
        if card.hand_trim:
            blocks.tag(top, t, "hand trim").pack(side="left", padx=(t.SPACE_SM, 0))
        units = ("— units" if card.units is None             # its read failed: no count
                 else f"{card.units:,} unit" + ("" if card.units == 1 else "s"))
        if card.final_test:
            units += " · final test"
        ctk.CTkLabel(inner, text=units, anchor="w", font=t.font(t.SIZE_CAPTION),
                     text_color=t.TEXT_SECONDARY).pack(side="top", fill="x")
        nums = ctk.CTkFrame(inner, fg_color="transparent")
        nums.pack(side="top", fill="x", pady=(t.SPACE_XS, 0))
        self._pass = ctk.CTkLabel(nums, text=od.pct_text(card.pass_pct),
                                  font=t.mono(t.SIZE_DISPLAY, "bold"), text_color=t.TEXT_PRIMARY)
        self._pass.pack(side="left")
        was = (f"was {od.pct_text(card.was_pct)}" if card.was_pct is not None
               else ("new" if card.units else ""))
        self._was = ctk.CTkLabel(nums, text=was, font=t.font(t.SIZE_CAPTION),
                                 text_color=t.TEXT_SECONDARY)
        self._was.pack(side="left", padx=(t.SPACE_SM, 0), pady=(t.SPACE_SM, 0))
        self._bars = _MonthBars(inner, t, card.months)
        self._bars.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))
        self._reason = ctk.CTkLabel(inner, text=card.reason, anchor="w", justify="left",
                                    font=t.font(t.SIZE_BODY), text_color=t.FAIL_FG)
        self._reason.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))
        # `inner` is built and destroyed with this card, so the binding never outlives it.
        blocks.wrap_to_width(self._reason, inner)
        _bind_click(self, lambda c=card: on_open(c))


class _MonthBars(tkinter.Canvas):
    """Twelve months of pass %, oldest first, as bars on a plain Tk canvas (never matplotlib -- a
    dozen-plus of these on one page): the history colour, the newest month highlighted, nothing
    drawn for a month with no units. Redrawn to its width on every <Configure>."""

    def __init__(self, master, theme, months):
        try:
            scale = float(master._get_widget_scaling())
        except Exception:
            scale = 1.0
        self.theme = theme
        self.months = list(months or [])
        super().__init__(master, height=int(BARS_HEIGHT * scale), width=120, bg=theme.CARD,
                         highlightthickness=0, bd=0)
        self.bind("<Configure>", lambda _e: self.draw(), add="+")
        self.draw()

    def draw(self) -> None:
        t = self.theme
        self.delete("all")
        w = max(int(self.winfo_width()), int(self.cget("width")))
        h = max(int(self.winfo_height()), int(self.cget("height")))
        if not self.months:
            return
        n = len(self.months)
        gap = max(2, w // (n * 6))
        bar_w = max(1.0, (w - gap * (n - 1)) / n)
        self.create_line(0, h - 1, w, h - 1, fill=t.DIVIDER)            # the floor
        for i, v in enumerate(self.months):
            if v is None:
                continue
            x0 = i * (bar_w + gap)
            top = h - 1 - max(1.0, (h - 2) * v / 100.0)
            self.create_rectangle(x0, top, x0 + bar_w, h - 1, width=0, tags=("bar",),
                                  fill=t.CHART_HIGHLIGHT if i == n - 1 else t.CHART_HISTORY)


class _ListRow(ctk.CTkFrame):
    """One "Everything else" line: model (and its hand-trim tag) · units · pass % · trend, over a
    divider. A click opens the model."""

    def __init__(self, master, theme, row: od.Row, on_open: Callable[[str], None]):
        t = theme
        super().__init__(master, fg_color="transparent")
        self.row = row
        for i, width in enumerate(LIST_COLUMNS):
            self.grid_columnconfigure(i, minsize=self._apply_widget_scaling(width))
        self.grid_columnconfigure(len(LIST_COLUMNS), weight=1)
        name = ctk.CTkFrame(self, fg_color="transparent")
        name.grid(row=0, column=0, sticky="w", pady=t.SPACE_SM)
        ctk.CTkLabel(name, text=row.model, font=t.mono(t.SIZE_BODY), text_color=t.TEXT_PRIMARY,
                     anchor="w").pack(side="left")
        if row.hand_trim:
            blocks.tag(name, t, "hand trim").pack(side="left", padx=(t.SPACE_SM, 0))
        units = f"{row.units:,} unit" + ("" if row.units == 1 else "s")
        ctk.CTkLabel(self, text=units, font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY,
                     anchor="e").grid(row=0, column=1, sticky="e", padx=(0, t.SPACE_LG))
        ctk.CTkLabel(self, text=od.pct_text(row.pass_pct), font=t.mono(t.SIZE_BODY, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="e").grid(row=0, column=2, sticky="e",
                                                                 padx=(0, t.SPACE_LG))
        self._trend = ctk.CTkLabel(self, text=row.trend, font=t.font(t.SIZE_BODY),
                                   text_color=getattr(t, _TONE.get(row.tone, "TEXT_SECONDARY")),
                                   anchor="w")
        self._trend.grid(row=0, column=3, sticky="w")
        ctk.CTkFrame(self, height=1, fg_color=t.DIVIDER, corner_radius=0
                     ).grid(row=1, column=0, columnspan=len(LIST_COLUMNS) + 1, sticky="ew")
        _bind_click(self, lambda m=row.model: on_open(m))
