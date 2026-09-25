"""Dashboard — Production Health landing: trim + final-test yield panels with trend,
and a clickable lowest-yield-models list that routes to the Model page."""
import logging
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

import customtkinter as ctk

from laser_trim_analyzer.core.cost_priorities import compute_cost_priorities
from laser_trim_analyzer.core.yield_stats import (
    compute_unit_yield, compute_yield, worst_models_by_yield)
from laser_trim_analyzer.database.models import AnalysisResult as DBAR, FinalTestResult as DBFT
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets import blocks
from laser_trim_analyzer.gui.v6.widgets.company_trend_chart import CompanyTrendChart
from laser_trim_analyzer.gui.v6.widgets.priorities_panel import PrioritiesPanel
from laser_trim_analyzer.gui.v6.widgets.worst_models_list import WorstModelsList
from laser_trim_analyzer.gui.v6.widgets.yield_panel import YieldPanel

_WINDOW_DAYS = {"30d": 30, "90d": 90, "365d": 365, "All": 36500}
_TREND_PERIODS = {"Weekly": "week", "Monthly": "month"}
# Words for the caption (design doc §4: "over the last 90 days"), keyed by the
# same choices as _WINDOW_DAYS -- "All" reads as a duration, not "36500 days".
_WINDOW_LABELS = {"30d": "30 days", "90d": "90 days", "365d": "365 days", "All": "all time"}


def _rate_text(stats) -> str:
    """'61%' for the caption -- '—' when the loader failed (stats is None) or the
    window has no gradeable data yet (rate is None). Same linearity-yield-first
    basis as YieldPanel.set_yield, so the caption never disagrees with the panel
    it summarizes; rounded to a whole percent (design doc §4's own example:
    "Laser 61% · final test 83%"), where the panel itself keeps one decimal."""
    if not stats:
        return "—"
    rate = stats.get("linearity_yield")
    if rate is None:
        rate = stats.get("pass_rate")
    return f"{rate:.0f}%" if rate is not None else "—"


class DashboardPage(PageBase):
    page_title = "Dashboard"

    def __init__(self, master, *, theme, app, page_title="Dashboard"):
        self._window_choice = "90d"
        self._trend_period_choice = "Weekly"
        self._reload_gen = 0
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    def header_actions(self, parent):
        t = self.theme
        self._window_menu = ctk.CTkOptionMenu(parent, values=list(_WINDOW_DAYS), width=90,
                                              command=self._on_window_change, fg_color=t.CARD,
                                              button_color=t.SEGMENT_SELECTED,
                                              button_hover_color=t.SEGMENT_SELECTED_HOVER,
                                              text_color=t.TEXT_PRIMARY)
        self._window_menu.set(self._window_choice)
        self._window_menu.pack(side="left")

    def build_content(self, parent):
        t = self.theme
        # Full-page scroll (James, 2026-07-13: "i need full page scroll on
        # the dashboard") — same treatment as the Model page. Wheel over the
        # trend chart won't page-scroll (the chart canvas owns its events);
        # wheel anywhere else, or the scrollbar, works.
        body = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        body.pack(fill="both", expand=True)
        # A failed loader is named here, never drawn as zeros (CLAUDE.md) --
        # unpacked until _set_load_banner finds something to say. Declared
        # before the first real content so `before=self._priorities` (its
        # pack anchor, set in _set_load_banner) is always a live sibling.
        self._load_banner = blocks.banner(body, t, "")
        # This week's priorities — money leaking at final test — front and centre.
        self._priorities = PrioritiesPanel(body, theme=t, on_row_click=self._on_model_click)
        self._priorities.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        panels = ctk.CTkFrame(body, fg_color="transparent")
        panels.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        panels.grid_columnconfigure((0, 1), weight=1, uniform="yp")
        self._trim_panel = YieldPanel(panels, theme=t, title="Trim analysis yield")
        self._trim_panel.grid(row=0, column=0, sticky="ew", padx=(0, t.SPACE_SM))
        self._ft_panel = YieldPanel(panels, theme=t, title="Final-test yield")
        self._ft_panel.grid(row=0, column=1, sticky="ew", padx=(t.SPACE_SM, 0))

        # Company-as-a-whole trend (V5 Trends' surviving job): pass-rate over
        # time, per-system overlay, volume backdrop. Honors the page window.
        trend_hdr = ctk.CTkFrame(body, fg_color="transparent")
        trend_hdr.pack(side="top", fill="x")
        ctk.CTkLabel(trend_hdr, text="Company trend", font=t.font(t.SIZE_BODY, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w").pack(side="left")
        self._trend_period_menu = ctk.CTkOptionMenu(
            trend_hdr, values=list(_TREND_PERIODS), width=100,
            command=self._on_trend_period_change, fg_color=t.CARD,
            button_color=t.SEGMENT_SELECTED, button_hover_color=t.SEGMENT_SELECTED_HOVER,
            text_color=t.TEXT_PRIMARY)
        self._trend_period_menu.set(self._trend_period_choice)
        self._trend_period_menu.pack(side="right")
        self._company_trend = CompanyTrendChart(body, theme=t)
        self._company_trend.pack(side="top", fill="x", pady=(t.SPACE_XS, t.SPACE_MD))

        self._worst = WorstModelsList(body, theme=t, on_row_click=self._on_model_click)
        self._worst.pack(side="top", fill="both", expand=True)

    # ---- lifecycle ----
    def on_show(self):
        threading.Thread(target=self._reload_threaded, daemon=True).start()

    def _cutoff(self):
        return datetime.now() - timedelta(days=_WINDOW_DAYS.get(self._window_choice, 90))

    def reload_now(self):
        """Synchronous reload + apply (test path / main-thread apply)."""
        self._apply(*self._query())

    def _reload_threaded(self):
        self._reload_gen += 1
        gen = self._reload_gen
        data = self._query()
        self.safe_after(lambda: self._apply(*data) if gen == self._reload_gen else None)

    def _query(self):
        """Three loaders, each guarded on its own -- `failed` names every one that
        raised (design doc §4: "a banner per failed loader"; CLAUDE.md: "a failure
        must never look like a result"). The old single try/except around all of
        trim/ft/worst swallowed the exception with NO log at all and rendered a
        healthy-looking all-zero dict -- that bug is the reason this task exists.

        "yield" covers the trim panel, the FT panel and the worst-models list
        together: `worst_models_by_yield` and `compute_unit_yield` are the same
        yield data by a different cut, so one loader/one banner entry for all of
        it matches the page (two panels + one list under "yield is lost"),
        matches the caption (both panels' rates, one loader), and matches what
        the brief's own failing test names ("a check banner naming 'yield'").
        Workers never touch Tk here -- this runs on the worker thread; only the
        return values cross to _apply on the UI thread via safe_after.
        """
        cutoff = self._cutoff()
        failed: list = []
        trim = ft = None
        worst, total = [], 0
        try:
            trim = compute_yield(self.app.db, DBAR, cutoff)
            ft = compute_yield(self.app.db, DBFT, cutoff)
            worst, total = worst_models_by_yield(self.app.db, cutoff)
            try:
                trim["unit_yield"] = compute_unit_yield(self.app.db, cutoff)
            except Exception:
                logger.exception("Dashboard: unit yield failed")
                trim["unit_yield"] = None
        except Exception:
            logger.exception("Dashboard: yield query failed")
            failed.append("yield")
            trim, ft = None, None
            worst, total = [], 0
        # Weekly buckets over the 'All' window = 600+ points smeared into an
        # unreadable block (live-walk finding, 2026-07-08). Coarsen to monthly
        # when the window is too long for weeks, and SAY so on the chart.
        days_back = _WINDOW_DAYS.get(self._window_choice, 90)
        period = _TREND_PERIODS.get(self._trend_period_choice, "week")
        trend_note = None
        if period == "week" and days_back > 730:
            period = "month"
            trend_note = "shown monthly — weekly is too dense for this window"
        try:
            company_trend = self.app.db.get_company_yield_trend(
                days_back=days_back, period=period)
        except Exception:
            logger.exception("Dashboard: company trend failed")
            failed.append("company trend")
            company_trend = None
        try:
            am = self.app.config.active_models
            priorities = compute_cost_priorities(
                self.app.db, am.model_prices, am.cost_ratio, recent_days=days_back)
        except Exception:
            logger.exception("Dashboard: cost priorities failed")
            failed.append("priorities")
            priorities = []
        return trim, ft, worst, total, company_trend, period, trend_note, priorities, failed

    def _apply(self, trim, ft, worst, total, company_trend=None,
               period="week", trend_note=None, priorities=None, failed=None):
        failed = list(failed or [])
        # trim/ft are None exactly when the "yield" loader raised -- set_yield(None, ...)
        # keeps the panel at its blank "—" state instead of a fabricated 0%/0 count.
        self._trim_panel.set_yield(
            trim, total_label=f"{trim['total']} trim records" if trim is not None else "")
        if trim is not None:
            try:
                self._trim_panel.set_unit_yield(trim.get("unit_yield"))
            except Exception:
                logger.exception("Dashboard: unit yield render failed")
        else:
            self._trim_panel.set_unit_yield(None)
        self._ft_panel.set_yield(
            ft, total_label=(f"{ft['total']} final-test records (matched to trims)"
                             if ft is not None else ""))
        try:
            self._company_trend.set_data(
                company_trend, period_label=period, note=trend_note)
        except Exception:
            # Isolate the trend from the rest of the dashboard, but NEVER
            # silently (a swallowed error rendered as a blank chart).
            logger.exception("Dashboard: company trend render failed")
        self._worst.set_rows(worst, total)
        try:
            self._priorities.set_rows(priorities or [])
        except Exception:
            logger.exception("Dashboard: priorities render failed")
        window_words = _WINDOW_LABELS.get(self._window_choice, self._window_choice)
        self.set_caption(f"Laser {_rate_text(trim)} · final test {_rate_text(ft)} "
                         f"over the last {window_words}")
        self._set_load_banner(failed)

    def _set_load_banner(self, failed) -> None:
        """Name every loader that raised this pass (mirrors ModelPage's
        `_set_load_banner`, same wording, same one-banner-not-one-per-failure
        shape) -- `before=self._priorities`, a fixed anchor created first in
        build_content, so this never re-appends itself after the panels."""
        if not failed:
            self._load_banner.pack_forget()
            return
        self._load_banner.configure(
            text=("⚠ Could not load: " + ", ".join(failed) + ". Those parts of this "
                  "page may be empty or out of date — this is an error, not an "
                  "absence of data. The log has the details."))
        self._load_banner.pack(side="top", fill="x", pady=(0, self.theme.SPACE_MD),
                               before=self._priorities)

    # ---- events ----
    def _on_window_change(self, choice):
        self._window_choice = choice
        self.on_show()

    def _on_trend_period_change(self, choice):
        self._trend_period_choice = choice
        self.on_show()

    def _on_model_click(self, model):
        self.app.set_model_route(model)
        self.app.show_page("model")
