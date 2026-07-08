"""Dashboard — Production Health landing: trim + final-test yield panels with trend,
and a clickable lowest-yield-models list that routes to the Model page."""
import logging
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

import customtkinter as ctk

from laser_trim_analyzer.core.yield_stats import compute_yield, worst_models_by_yield
from laser_trim_analyzer.database.models import AnalysisResult as DBAR, FinalTestResult as DBFT
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.company_trend_chart import CompanyTrendChart
from laser_trim_analyzer.gui.v6.widgets.worst_models_list import WorstModelsList
from laser_trim_analyzer.gui.v6.widgets.yield_panel import YieldPanel

_WINDOW_DAYS = {"30d": 30, "90d": 90, "365d": 365, "All": 36500}
_TREND_PERIODS = {"Weekly": "week", "Monthly": "month"}


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
                                              button_color=t.ACCENT, button_hover_color=t.ACCENT_HOVER,
                                              text_color=t.TEXT_PRIMARY)
        self._window_menu.set(self._window_choice)
        self._window_menu.pack(side="left")

    def build_content(self, parent):
        t = self.theme
        panels = ctk.CTkFrame(parent, fg_color="transparent")
        panels.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        panels.grid_columnconfigure((0, 1), weight=1, uniform="yp")
        self._trim_panel = YieldPanel(panels, theme=t, title="Trim analysis yield")
        self._trim_panel.grid(row=0, column=0, sticky="ew", padx=(0, t.SPACE_SM))
        self._ft_panel = YieldPanel(panels, theme=t, title="Final-test yield")
        self._ft_panel.grid(row=0, column=1, sticky="ew", padx=(t.SPACE_SM, 0))

        # Company-as-a-whole trend (V5 Trends' surviving job): pass-rate over
        # time, per-system overlay, volume backdrop. Honors the page window.
        trend_hdr = ctk.CTkFrame(parent, fg_color="transparent")
        trend_hdr.pack(side="top", fill="x")
        ctk.CTkLabel(trend_hdr, text="Company trend", font=t.font(t.SIZE_BODY, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w").pack(side="left")
        self._trend_period_menu = ctk.CTkOptionMenu(
            trend_hdr, values=list(_TREND_PERIODS), width=100,
            command=self._on_trend_period_change, fg_color=t.CARD,
            button_color=t.ACCENT, button_hover_color=t.ACCENT_HOVER,
            text_color=t.TEXT_PRIMARY)
        self._trend_period_menu.set(self._trend_period_choice)
        self._trend_period_menu.pack(side="right")
        self._company_trend = CompanyTrendChart(parent, theme=t)
        self._company_trend.pack(side="top", fill="x", pady=(t.SPACE_XS, t.SPACE_MD))

        self._worst = WorstModelsList(parent, theme=t, on_row_click=self._on_model_click)
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
        cutoff = self._cutoff()
        try:
            trim = compute_yield(self.app.db, DBAR, cutoff)
            ft = compute_yield(self.app.db, DBFT, cutoff)
            worst, total = worst_models_by_yield(self.app.db, cutoff)
        except Exception:
            empty = {"passed": 0, "warnings": 0, "failed": 0, "errors": 0, "untrimmed": 0,
                     "gradeable": 0, "total": 0, "pass_rate": None, "trend": []}
            trim, ft, worst, total = dict(empty), dict(empty), [], 0
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
            company_trend = None
        return trim, ft, worst, total, company_trend, period, trend_note

    def _apply(self, trim, ft, worst, total, company_trend=None,
               period="week", trend_note=None):
        self._trim_panel.set_yield(trim, total_label=f"{trim['total']} units")
        self._ft_panel.set_yield(
            ft, total_label=f"{ft['total']} final-test records (matched to trims)")
        try:
            self._company_trend.set_data(
                company_trend, period_label=period, note=trend_note)
        except Exception:
            # Isolate the trend from the rest of the dashboard, but NEVER
            # silently (a swallowed error rendered as a blank chart).
            logger.exception("Company trend render failed")
        self._worst.set_rows(worst, total)

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
