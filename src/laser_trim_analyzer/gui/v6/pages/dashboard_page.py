"""Dashboard — Production Health landing: trim + final-test yield panels with trend,
and a clickable lowest-yield-models list that routes to the Model page."""
import threading
from datetime import datetime, timedelta

import customtkinter as ctk

from laser_trim_analyzer.core.yield_stats import compute_yield, worst_models_by_yield
from laser_trim_analyzer.database.models import AnalysisResult as DBAR, FinalTestResult as DBFT
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.worst_models_list import WorstModelsList
from laser_trim_analyzer.gui.v6.widgets.yield_panel import YieldPanel

_WINDOW_DAYS = {"30d": 30, "90d": 90, "365d": 365, "All": 36500}


class DashboardPage(PageBase):
    page_title = "Dashboard"

    def __init__(self, master, *, theme, app, page_title="Dashboard"):
        self._window_choice = "90d"
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
        return trim, ft, worst, total

    def _apply(self, trim, ft, worst, total):
        self._trim_panel.set_yield(trim, total_label=f"{trim['total']} units")
        self._ft_panel.set_yield(ft, total_label=f"{ft['total']} matched")
        self._worst.set_rows(worst, total)

    # ---- events ----
    def _on_window_change(self, choice):
        self._window_choice = choice
        self.on_show()

    def _on_model_click(self, model):
        self.app.set_model_route(model)
        self.app.show_page("model")
