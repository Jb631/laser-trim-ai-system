"""Spec 3c — ModelPage: per-model investigation (selector + window + Copy/Export header,
8 pills, SPC focus chart, 3 tabs, demoted predictor). Foundations §3 + D4."""
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Optional

import customtkinter as ctk

logger = logging.getLogger(__name__)

from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAR, ModelMetricState, SmoothnessResult as DBSR, TrackResult as DBTR)
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import DriftMetricsTab
from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
from laser_trim_analyzer.gui.v6.widgets.history_tab import HistoryTab
from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
from laser_trim_analyzer.gui.v6.widgets.predictor_panel import PredictorPanel
from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import SmoothnessTab
from laser_trim_analyzer.gui.v6.widgets.tab_view import ThemedTabView
from laser_trim_analyzer.gui.v6.widgets.trim_ft_tab import TrimFtTab
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
        self._pill_row.pack(side="top", fill="x", pady=(0, t.SPACE_XS))
        # Plain-language key for the pill numbers: σ was shown with no
        # explanation anywhere in the app (2026-07-07, user feedback).
        ctk.CTkLabel(self._body,
                     text=("σ = standard deviations from this model's trained baseline. "
                           "+1.0σ means the recent average runs one standard deviation "
                           "above normal for this model — the drift signal, not a spec."),
                     font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY,
                     anchor="w", justify="left", wraplength=980)\
            .pack(side="top", fill="x", pady=(0, t.SPACE_MD))
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
                                   on_unit_click=self._on_unit_click, on_export=self._on_export,
                                   on_search=self._on_unit_search)
        self._units_tab.pack(fill="both", expand=True)
        self._trimft_tab = TrimFtTab(self._tabs.add("Trim vs Final Test"), theme=t)
        self._trimft_tab.pack(fill="both", expand=True)
        self._history_tab = HistoryTab(self._tabs.add("History"), theme=t)
        self._history_tab.pack(fill="both", expand=True)
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
            # Each loader is independently guarded: one failure must not blank
            # the whole page (the old single try/except silently zeroed every
            # tab when any loader threw). Failures are logged, not swallowed.
            status, chosen = None, metric
            dates, values, baseline = [], [], (None, None)
            units, smoothness, recent = [], [], {}
            trim_ft, history = {}, {}
            # One anchored cutoff for every tab (None = All): anchored to the
            # model's latest data so stale-but-flagged models still show their
            # record instead of empty windows.
            cutoff = self._window_cutoff(model)
            try:
                status = get_model_drift_status(self.app.db, model)
                chosen = self._resolve_focus_metric(status, self._user_picked_metric, metric)
            except Exception:
                logger.exception("Model %s: drift status failed", model)
            try:
                dates, values, baseline = self._load_focus_series(model, chosen)
            except Exception:
                logger.exception("Model %s: focus series failed", model)
            try:
                units = self._load_units(model)
            except Exception:
                logger.exception("Model %s: units failed", model)
            try:
                smoothness = self._load_smoothness(model)
            except Exception:
                logger.exception("Model %s: smoothness failed", model)
            try:
                recent = self._recent_means(model)
            except Exception:
                logger.exception("Model %s: recent means failed", model)
            try:
                trim_ft = self.app.db.get_model_trim_ft_agreement(model, cutoff_date=cutoff)
            except Exception:
                logger.exception("Model %s: trim-vs-FT failed", model)
            try:
                history = self.app.db.get_model_measurement_history(model, cutoff_date=cutoff)
            except Exception:
                logger.exception("Model %s: history failed", model)

            def apply():
                if gen != self._reload_gen:
                    return  # a newer reload superseded this one
                self._current_metric = chosen

                def _try(what, fn):
                    # Per-widget guard: a render bug in one tab must not stop
                    # the tabs after it from updating.
                    try:
                        fn()
                    except Exception:
                        logger.exception("Model %s: %s render failed", model, what)

                if status:
                    _try("pills", lambda: self._pill_row.set_status(status, recent_means=recent))
                    _try("drift tab", lambda: self._drift_tab.set_status(status, recent_means=recent))
                _try("pill select", lambda: self._pill_row.set_selected(chosen))
                _try("focus chart", lambda: self._focus_chart.set_series(
                    metric=chosen, dates=dates, values=values,
                    baseline_mean=baseline[0], baseline_std=baseline[1]))
                _try("units tab", lambda: self._units_tab.set_units(units))
                _try("smoothness tab", lambda: self._smoothness_tab.set_records(smoothness))
                _try("trim-vs-FT tab", lambda: self._trimft_tab.set_data(trim_ft))
                _try("history tab", lambda: self._history_tab.set_data(history))
            self.safe_after(apply)
        threading.Thread(target=work, daemon=True).start()

    # ---- loaders (all materialize to plain values inside the session — I8) ----
    def _window_cutoff(self, model: Optional[str] = None) -> Optional[datetime]:
        """Window cutoff anchored to the MODEL'S latest data, not wall-clock now.

        This is batch-loaded historical data: a model's newest unit can be
        months old (loaded lots lag production). Anchoring to now() made every
        window empty for such models — Triage would flag a model, the click-
        through landed on 'No measurements in the selected window'. Same
        reasoning as compute_recent_means (evidence.py).
        """
        days = _WINDOW_DAYS.get(self._window_choice)
        if days is None:
            return None
        anchor = None
        model = model or self._current_model
        if model:
            try:
                from sqlalchemy import func
                with self.app.db.session() as s:
                    anchor = (s.query(func.max(DBAR.file_date))
                              .filter(DBAR.model == model).scalar())
            except Exception:
                logger.exception("window anchor query failed for %s", model)
        if anchor is None:
            anchor = datetime.now()
        return anchor - timedelta(days=days)

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

    def _search_units(self, model, query: str) -> List[dict]:
        """Serial lookup for the model — ignores the window and the recent cap so an old
        unit can still be found. Case-insensitive substring match on serial."""
        like = f"%{query}%"
        with self.app.db.session() as s:
            rows = (s.query(DBAR.id, DBAR.serial, DBAR.file_date, DBAR.overall_status,
                            DBTR.sigma_gradient, DBTR.final_linearity_error_shifted)
                    .join(DBTR, DBTR.analysis_id == DBAR.id)
                    .filter(DBAR.model == model, DBAR.serial.ilike(like))
                    .order_by(DBAR.file_date.desc()).limit(500).all())
            return [{"analysis_id": r[0], "serial": r[1], "file_date": r[2],
                     "overall_status": getattr(r[3], "value", str(r[3])),
                     "sigma_gradient": r[4], "linearity_error": r[5]} for r in rows]

    def _on_unit_search(self, query: str) -> None:
        model = self._current_model
        if not model:
            return
        query = (query or "").strip()
        if not query:
            # Cleared → restore the recent list for the current window.
            def restore():
                units = self._load_units(model)
                self.safe_after(lambda: self._units_tab.set_units(units))
            threading.Thread(target=restore, daemon=True).start()
            return

        def work():
            try:
                results = self._search_units(model, query)
            except Exception:
                results = []
            cap = (f"{len(results)} match(es) for '{query}'"
                   + (" (showing first 500)" if len(results) == 500 else "")
                   + " — all dates"
                   if results else f"No units matching '{query}' for {model}")
            self.safe_after(lambda: self._units_tab.set_units(results, caption=cap))
        threading.Thread(target=work, daemon=True).start()

    def _load_smoothness(self, model) -> List[dict]:
        cutoff = self._window_cutoff()
        with self.app.db.session() as s:
            q = s.query(DBSR).filter(DBSR.model == model)
            if cutoff:
                q = q.filter(DBSR.file_date >= cutoff)
            rows = q.order_by(DBSR.file_date.desc()).limit(200).all()
            return [{"serial": r.serial, "file_date": r.file_date,
                     "max_smoothness_value": r.max_smoothness_value,
                     "smoothness_spec": r.smoothness_spec,
                     "smoothness_pass": r.smoothness_pass,
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
        model = self._current_model
        from laser_trim_analyzer.export.evidence import build_summary_text

        # The drift status + recent-means queries are heavy (~10 aggregate
        # queries). Running them here blocked the UI behind the DB lock —
        # do them on a worker; only the clipboard write returns to Tk.
        def work():
            try:
                from laser_trim_analyzer.export.evidence import compute_recent_means
                status = get_model_drift_status(self.app.db, model)
                means, meta = compute_recent_means(self.app.db, model,
                                                   recent_days=_RECENT_DAYS, with_meta=True)
                text = build_summary_text(model, status, recent_means=means,
                                          recent_meta=meta)
            except Exception:
                logger.exception("Copy summary failed for %s", model)
                return
            def to_clipboard():
                self.clipboard_clear()
                self.clipboard_append(text)
            self.safe_after(to_clipboard)
        threading.Thread(target=work, daemon=True).start()

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
        # Always export the FULL record: the Excel pack is the model's history
        # of record for judging process direction (the on-screen window only
        # controls the view). James' workflow: analyze units on screen, export
        # the model to Excel for full history, unit charts for the team.
        threading.Thread(
            target=lambda: export_evidence_pack(self.app.db, self._current_model, path),
            daemon=True).start()
