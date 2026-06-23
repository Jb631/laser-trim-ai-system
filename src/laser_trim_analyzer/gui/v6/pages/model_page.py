"""Spec 3c — ModelPage: per-model investigation (selector + window + Copy/Export header,
8 pills, SPC focus chart, 3 tabs, demoted predictor). Foundations §3 + D4."""
import threading
from datetime import datetime, timedelta
from typing import List, Optional

import customtkinter as ctk

from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAR, ModelMetricState, SmoothnessResult as DBSR, TrackResult as DBTR)
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import DriftMetricsTab
from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
from laser_trim_analyzer.gui.v6.widgets.predictor_panel import PredictorPanel
from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import SmoothnessTab
from laser_trim_analyzer.gui.v6.widgets.tab_view import ThemedTabView
from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import UnitChartModal
from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab
from laser_trim_analyzer.ml.drift_training import TRACK_METRIC_COLUMNS
from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS
from laser_trim_analyzer.ml.manager import get_model_drift_status, list_known_models

_WINDOW_DAYS = {"30d": 30, "90d": 90, "365d": 365, "All": None}

# Default focus metric when no alert-triggered focus is supplied. The headline
# element-production drift signal (post-trim sigma_gradient is no longer
# watched — see drift_types.WATCHED_METRICS / the D-SIGMA rationale).
_DEFAULT_METRIC = "untrimmed_sigma_gradient"

# "recent" window for the baseline-vs-recent comparison shown in the Drift table.
_RECENT_DAYS = 30


class ModelPage(PageBase):
    page_title = "Model"

    def __init__(self, master, *, theme, app, page_title="Model"):
        self._current_model: Optional[str] = None
        self._current_metric: str = _DEFAULT_METRIC
        self._window_choice: str = "90d"
        self._reload_gen = 0
        self._user_picked_metric = False
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    @staticmethod
    def _resolve_focus_metric(status, user_picked, current):
        """Pick the metric to focus: the user's explicit pick wins; otherwise the
        model's worst flagged metric; otherwise the current fallback."""
        if user_picked:
            return current
        if status is not None and status.worst_metric and status.worst_metric in WATCHED_METRICS:
            return status.worst_metric
        return current

    # ---- header (built INTO the actions parent — no reparenting) ----
    def header_actions(self, parent):
        t = self.theme
        self._model_selector = ctk.CTkComboBox(parent, values=[], width=200,
                                                command=self._on_model_selected, fg_color=t.CARD,
                                                border_color=t.BORDER, button_color=t.ACCENT,
                                                button_hover_color=t.ACCENT_HOVER, text_color=t.TEXT_PRIMARY)
        self._model_selector.set("Select model…")
        self._model_selector.pack(side="left", padx=(0, t.SPACE_SM))
        self._window_menu = ctk.CTkOptionMenu(parent, values=list(_WINDOW_DAYS), width=80,
                                              command=self._on_window_change, fg_color=t.CARD,
                                              button_color=t.ACCENT, button_hover_color=t.ACCENT_HOVER,
                                              text_color=t.TEXT_PRIMARY)
        self._window_menu.set(self._window_choice)
        self._window_menu.pack(side="left", padx=(0, t.SPACE_SM))
        ctk.CTkButton(parent, text="Copy summary", command=self._on_copy_summary, fg_color=t.CARD,
                      hover_color=t.ELEVATED, text_color=t.TEXT_PRIMARY, corner_radius=t.RADIUS_SM)\
            .pack(side="left", padx=(0, t.SPACE_SM))
        ctk.CTkButton(parent, text="Export evidence pack", command=self._on_export, fg_color=t.ACCENT,
                      hover_color=t.ACCENT_HOVER, text_color=t.TEXT_INVERSE, corner_radius=t.RADIUS_SM)\
            .pack(side="left")

    def build_content(self, parent):
        t = self.theme
        self._empty_label = ctk.CTkLabel(
            parent, text="Pick a model above, or click one from Triage, to see its drift profile.",
            font=t.font(t.SIZE_HEADING), text_color=t.TEXT_SECONDARY)
        self._body = ctk.CTkFrame(parent, fg_color="transparent")
        self._pill_row = MetricPillRow(self._body, theme=t, on_pill_click=self._on_pill_click)
        self._pill_row.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        self._focus_chart = FocusChart(self._body, theme=t)
        self._focus_chart.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        self._tabs = ThemedTabView(self._body, theme=t)
        self._tabs.pack(side="top", fill="both", expand=True)
        self._drift_tab = DriftMetricsTab(self._tabs.add("Drift Metrics"), theme=t,
                                          on_metric_select=self._on_pill_click)
        self._drift_tab.pack(fill="both", expand=True)
        self._smoothness_tab = SmoothnessTab(self._tabs.add("Smoothness"), theme=t)
        self._smoothness_tab.pack(fill="both", expand=True)
        self._units_tab = UnitsTab(self._tabs.add("Units"), theme=t,
                                   on_unit_click=self._on_unit_click, on_export=self._on_export)
        self._units_tab.pack(fill="both", expand=True)
        self._predictor = PredictorPanel(self._body, theme=t, db=self.app.db)
        self._predictor.pack(side="top", fill="x", pady=(t.SPACE_MD, 0))
        self._show_empty()

    # ---- visibility ----
    def _show_empty(self):
        self._body.pack_forget()
        self._empty_label.pack(expand=True)

    def _show_body(self):
        self._empty_label.pack_forget()
        self._body.pack(fill="both", expand=True)

    # ---- lifecycle ----
    def on_show(self):
        model, focus = self.app.consume_model_route_full()
        if model:
            self._current_model = model
            self._user_picked_metric = False
        if focus and focus in WATCHED_METRICS:
            self._current_metric = focus
            self._user_picked_metric = True
        # Refresh the selector's model list each show.
        threading.Thread(target=self._refresh_selector_values, daemon=True).start()
        if self._current_model:
            self._show_body()
            self._pill_row.set_selected(self._current_metric)
            self._predictor.set_model(self._current_model)
            self._reload()
        else:
            self._show_empty()

    def _refresh_selector_values(self):
        try:
            models = [m.model for m in list_known_models(self.app.db)]
        except Exception:
            models = []
        def apply():
            self._model_selector.configure(values=models)
            if self._current_model:
                self._model_selector.set(self._current_model)
        self.safe_after(apply)

    # ---- reload with generation token (I3) ----
    def _reload(self):
        if not self._current_model:
            return
        self._reload_gen += 1
        gen = self._reload_gen
        model, metric = self._current_model, self._current_metric
        def work():
            try:
                status = get_model_drift_status(self.app.db, model)
                chosen = self._resolve_focus_metric(status, self._user_picked_metric, metric)
                dates, values, baseline = self._load_focus_series(model, chosen)
                units = self._load_units(model)
                smoothness = self._load_smoothness(model)
                recent = self._recent_means(model)
            except Exception:
                status, chosen = None, metric
                dates, values, baseline, units, smoothness, recent = [], [], (None, None), [], [], {}
            def apply():
                if gen != self._reload_gen:
                    return  # a newer reload superseded this one
                self._current_metric = chosen
                if status:
                    self._pill_row.set_status(status, recent_means=recent)
                    self._drift_tab.set_status(status, recent_means=recent)
                self._pill_row.set_selected(chosen)
                self._focus_chart.set_series(metric=chosen, dates=dates, values=values,
                                             baseline_mean=baseline[0], baseline_std=baseline[1])
                self._units_tab.set_units(units)
                self._smoothness_tab.set_records(smoothness)
            self.safe_after(apply)
        threading.Thread(target=work, daemon=True).start()

    # ---- loaders (all materialize to plain values inside the session — I8) ----
    def _window_cutoff(self) -> Optional[datetime]:
        days = _WINDOW_DAYS.get(self._window_choice)
        return None if days is None else datetime.now() - timedelta(days=days)

    def _load_focus_series(self, model, metric):
        cutoff = self._window_cutoff()
        with self.app.db.session() as s:
            if metric == "max_smoothness_value":
                q = s.query(DBSR.file_date, DBSR.max_smoothness_value).filter(
                    DBSR.model == model, DBSR.max_smoothness_value.isnot(None))
                if cutoff:
                    q = q.filter(DBSR.file_date >= cutoff)
                rows = q.order_by(DBSR.file_date).all()
            elif metric in TRACK_METRIC_COLUMNS:
                col = TRACK_METRIC_COLUMNS[metric]      # Q4: SAME column the detector trained on
                q = (s.query(DBAR.file_date, col).join(DBTR, DBTR.analysis_id == DBAR.id)
                     .filter(DBAR.model == model, col.isnot(None)))
                if cutoff:
                    q = q.filter(DBAR.file_date >= cutoff)
                rows = q.order_by(DBAR.file_date).all()
            else:
                rows = []
            pairs = [(r[0], r[1]) for r in rows if r[0] is not None and r[1] is not None]  # Q5
            ms = s.query(ModelMetricState).filter_by(model=model, metric=metric).first()
            baseline = (ms.baseline_mean, ms.baseline_std) if ms else (None, None)
        return [p[0] for p in pairs], [p[1] for p in pairs], baseline

    def _recent_means(self, model) -> dict:
        """Mean of each watched metric over the model's most recent window of DATA.

        Delegates to the shared evidence helper so the UI table, copy-summary, and the
        Excel evidence pack all derive 'recent' the same way (anchored to the model's
        latest file_date, since this is batch-loaded data that can be weeks old).
        Metric -> float|None.
        """
        from laser_trim_analyzer.export.evidence import compute_recent_means
        return compute_recent_means(self.app.db, model, recent_days=_RECENT_DAYS)

    def _load_units(self, model) -> List[dict]:
        cutoff = self._window_cutoff()
        with self.app.db.session() as s:
            q = (s.query(DBAR.id, DBAR.serial, DBAR.file_date, DBAR.overall_status,
                         DBTR.sigma_gradient, DBTR.final_linearity_error_shifted)
                 .join(DBTR, DBTR.analysis_id == DBAR.id).filter(DBAR.model == model))
            if cutoff:
                q = q.filter(DBAR.file_date >= cutoff)
            rows = q.order_by(DBAR.file_date.desc()).limit(200).all()
            return [{"analysis_id": r[0], "serial": r[1], "file_date": r[2],
                     "overall_status": getattr(r[3], "value", str(r[3])),
                     "sigma_gradient": r[4], "linearity_error": r[5]} for r in rows]

    def _load_smoothness(self, model) -> List[dict]:
        cutoff = self._window_cutoff()
        with self.app.db.session() as s:
            q = s.query(DBSR).filter(DBSR.model == model)
            if cutoff:
                q = q.filter(DBSR.file_date >= cutoff)
            rows = q.order_by(DBSR.file_date.desc()).limit(200).all()
            return [{"serial": r.serial, "file_date": r.file_date,
                     "max_smoothness_value": r.max_smoothness_value,
                     "avg_smoothness_value": r.avg_smoothness_value,
                     "overall_status": getattr(r.overall_status, "value", str(r.overall_status))}
                    for r in rows]

    # ---- events ----
    def _on_model_selected(self, model):
        if model and model != "Select model…":
            self._current_model = model
            self._user_picked_metric = False   # new model → auto-focus its worst metric
            self._show_body()
            self._predictor.set_model(model)
            self._reload()

    def _on_pill_click(self, metric):
        self._user_picked_metric = True        # explicit choice; don't auto-override it
        self._current_metric = metric
        self._pill_row.set_selected(metric)
        self._reload()

    def _on_window_change(self, choice):
        self._window_choice = choice
        self._reload()

    def _on_unit_click(self, unit):
        UnitChartModal(self, theme=self.theme, db=self.app.db, unit=unit)

    def _on_copy_summary(self):
        if not self._current_model:
            return
        from laser_trim_analyzer.export.evidence import build_summary_text
        try:
            status = get_model_drift_status(self.app.db, self._current_model)
            text = build_summary_text(self._current_model, status,
                                      recent_means=self._recent_means(self._current_model))
            self.clipboard_clear()
            self.clipboard_append(text)
        except Exception:
            pass

    def _on_export(self):
        if not self._current_model:
            return
        from tkinter import filedialog
        from laser_trim_analyzer.export.evidence import export_evidence_pack
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            initialfile=f"evidence_{self._current_model}.xlsx",
                                            filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        days = _WINDOW_DAYS.get(self._window_choice)
        threading.Thread(
            target=lambda: export_evidence_pack(self.app.db, self._current_model, path, days),
            daemon=True).start()
