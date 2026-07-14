"""Spec 3c — ModelPage: per-model investigation (selector + window + Copy/Export header,
8 pills, SPC focus chart, 3 tabs, demoted predictor). Foundations §3 + D4."""
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Optional

import customtkinter as ctk

logger = logging.getLogger(__name__)

from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAR, ModelMetricState, SmoothnessResult as DBSR, TrackResult as DBTR, StatusType)
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import DriftMetricsTab
from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
from laser_trim_analyzer.gui.v6.widgets.history_tab import HistoryTab
from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
from laser_trim_analyzer.gui.v6.widgets.predictor_panel import PredictorPanel
from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import SmoothnessTab
from laser_trim_analyzer.gui.v6.widgets.tab_view import ThemedTabView
from laser_trim_analyzer.gui.v6.widgets.trim_ft_tab import TrimFtTab
from laser_trim_analyzer.gui.v6.widgets.ft_units_tab import FtUnitsTab
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
        # Typing a model number + Enter must load it — the combobox command
        # only fires on dropdown picks, and scrolling 279 entries to reach a
        # typed-out name is unusable (found live, 2026-07-08).
        self._model_selector.bind(
            "<Return>", lambda e: self._on_model_selected(self._model_selector.get().strip()))
        # Thumbwheel (work finding #2): the wheel steps prev/next model while
        # hovering the selector — the dropdown itself can't wheel-scroll in CTk.
        self._model_selector.bind("<MouseWheel>", self._on_selector_wheel)
        self._model_selector.bind("<Button-4>", lambda e: self._on_selector_wheel(e, step=-1))
        self._model_selector.bind("<Button-5>", lambda e: self._on_selector_wheel(e, step=1))
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
        ctk.CTkButton(parent, text="Export model to Excel", command=self._on_export, fg_color=t.ACCENT,
                      hover_color=t.ACCENT_HOVER, text_color=t.TEXT_INVERSE, corner_radius=t.RADIUS_SM)\
            .pack(side="left")

    def build_content(self, parent):
        t = self.theme
        self._empty_label = ctk.CTkLabel(
            parent, text="Pick a model above, or click one from Triage, to see its drift profile.",
            font=t.font(t.SIZE_HEADING), text_color=t.TEXT_SECONDARY)
        # SCROLLABLE body (James live report 2026-07-13: "cant scroll down on
        # some of the pages" — the zone headers + grouped pill bands made the
        # page taller than the window, and a plain frame just clips). Wheel
        # over the matplotlib chart won't page-scroll (the chart canvas owns
        # its events); wheel anywhere else, or the scrollbar, works.
        self._body = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        # ---- ZONE 1: the app's read (2026-07-13, James: "clear sections for
        # what im looking at and what the app is telling me"). Verdict FIRST
        # (the answer), then the per-metric pills (the evidence), then the σ
        # key. Everything in this zone is an interpretation.
        self._zone_header(self._body, "WHAT THE APP IS TELLING YOU",
                          "drift-watch verdict — one verdict per lot, never a spec disposition")
        # THE daily question in one line (user #17, 2026-07-13: "not getting
        # the most out of the available data"): holding or drifting, which
        # way the window is moving vs the model's lifetime, and whether this
        # model has historically been difficult.
        self._verdict = ctk.CTkLabel(self._body, text="", font=t.font(t.SIZE_BODY, "bold"),
                                     text_color=t.TEXT_PRIMARY, anchor="w",
                                     justify="left", wraplength=1200)
        self._verdict.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        self._pill_row = MetricPillRow(self._body, theme=t, on_pill_click=self._on_pill_click)
        self._pill_row.pack(side="top", fill="x", pady=(0, t.SPACE_XS))
        # Plain-language key for the pill numbers: σ was shown with no
        # explanation anywhere in the app (2026-07-07, user feedback).
        ctk.CTkLabel(self._body,
                     text=("σ = how far the last LOT's median sits from this model's baseline "
                           "of historical lot medians (lot = production run; new lot after "
                           ">3 idle days). +1.0σ = the last lot ran one lot-σ above normal. "
                           "Drift signal, not a spec."),
                     font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY,
                     anchor="w", justify="left", wraplength=980)\
            .pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        # ---- ZONE 2: the data itself — where the read above is verified.
        self._zone_header(self._body, "WHAT YOU'RE LOOKING AT",
                          "the measurements — chart the pill you clicked; units & final tests in the tabs")
        self._focus_chart = FocusChart(self._body, theme=t)
        self._focus_chart.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        self._tabs = ThemedTabView(self._body, theme=t)
        self._tabs.pack(side="top", fill="both", expand=True)
        self._drift_tab = DriftMetricsTab(self._tabs.add("Drift Metrics"), theme=t,
                                          on_requalify=self._on_requalify,
                                          on_metric_select=self._on_pill_click)
        self._drift_tab.pack(fill="both", expand=True)
        self._smoothness_tab = SmoothnessTab(self._tabs.add("Smoothness"), theme=t)
        self._smoothness_tab.pack(fill="both", expand=True)
        self._units_tab = UnitsTab(self._tabs.add("Units"), theme=t,
                                   on_unit_click=self._on_unit_click, on_export=self._on_export,
                                   on_export_charts=self._on_export_charts,
                                   on_search=self._on_unit_search)
        self._units_tab.pack(fill="both", expand=True)
        self._ft_units_tab = FtUnitsTab(self._tabs.add("Final Test Units"), theme=t)
        # pack was missing — the tab constructed but never mapped, so it
        # rendered permanently EMPTY (code-review finding #5, 2026-07-13).
        self._ft_units_tab.pack(fill="both", expand=True)
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
            requal = None
            try:
                status = get_model_drift_status(self.app.db, model)
                chosen = self._resolve_focus_metric(status, self._user_picked_metric, metric)
                requal = self.app.db.get_baseline_requalification(model)
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
            ft_units = []
            try:
                ft_units = self._load_ft_units(model, cutoff)
            except Exception:
                logger.exception("Model %s: final-test units failed", model)
            verdict = None
            try:
                verdict = self._compute_verdict(model, cutoff, status, recent)
            except Exception:
                logger.exception("Model %s: verdict failed", model)
            smooth_models = []
            try:
                from sqlalchemy import func as _f
                with self.app.db.session() as s_:
                    smooth_models = (s_.query(DBSR.model, _f.count(DBSR.id))
                                     .group_by(DBSR.model)
                                     .order_by(_f.count(DBSR.id).desc()).limit(12).all())
            except Exception:
                logger.exception("smoothness model list failed")

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
                _try("baseline info", lambda: self._drift_tab.set_baseline_info(requal))
                _try("pill select", lambda: self._pill_row.set_selected(chosen))
                if verdict:
                    _try("verdict", lambda: self._verdict.configure(
                        text=verdict[0], text_color=verdict[1]))
                _try("focus chart", lambda: self._focus_chart.set_series(
                    metric=chosen, dates=dates, values=values,
                    baseline_mean=baseline[0], baseline_std=baseline[1]))
                _try("units tab", lambda: self._units_tab.set_units(units))
                _try("smoothness tab", lambda: self._smoothness_tab.set_records(smoothness))
                _try("smoothness hint", lambda: self._smoothness_tab.set_models_hint(smooth_models))
                _try("trim-vs-FT tab", lambda: self._trimft_tab.set_data(trim_ft))
                _try("FT units tab", lambda: self._ft_units_tab.set_units(ft_units))
                _try("history tab", lambda: self._history_tab.set_data(history))
            self.safe_after(apply)
        threading.Thread(target=work, daemon=True).start()

    # ---- loaders (all materialize to plain values inside the session — I8) ----
    def _window_cutoff(self, model: Optional[str] = None,
                       metric: Optional[str] = None) -> Optional[datetime]:
        """Window cutoff anchored to the MODEL'S latest data, not wall-clock now.

        This is batch-loaded historical data: a model's newest unit can be
        months old (loaded lots lag production). Anchoring to now() made every
        window empty for such models — Triage would flag a model, the click-
        through landed on 'No measurements in the selected window'. Same
        reasoning as compute_recent_means (evidence.py).

        FT-axis metrics anchor to the FINAL-TEST table's newest date instead
        (code-review finding #6, 2026-07-13): FT ingest lags trim ingest, so a
        trim-anchored 90-day window could exclude every FT row — clicking a
        flagged FT pill landed on an empty chart. FT-only models (no trim
        rows) get a working anchor for the same reason.
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
                    if metric in ("ft_fail_fraction", "escape_fraction"):
                        from laser_trim_analyzer.database.models import (
                            FinalTestResult as DBFT)
                        anchor = (s.query(func.max(func.coalesce(
                                      DBFT.test_date, DBFT.file_date)))
                                  .filter(DBFT.model == model).scalar())
                    if anchor is None:
                        anchor = (s.query(func.max(DBAR.file_date))
                                  .filter(DBAR.model == model).scalar())
            except Exception:
                logger.exception("window anchor query failed for %s", model)
        if anchor is None:
            anchor = datetime.now()
        if isinstance(anchor, str):        # raw SQLite string from coalesce
            anchor = datetime.fromisoformat(anchor[:19])
        return anchor - timedelta(days=days)

    def _load_focus_series(self, model, metric):
        cutoff = self._window_cutoff(metric=metric)
        with self.app.db.session() as s:
            if metric == "max_smoothness_value":
                q = s.query(DBSR.file_date, DBSR.max_smoothness_value).filter(
                    DBSR.model == model, DBSR.max_smoothness_value.isnot(None))
                if cutoff:
                    q = q.filter(DBSR.file_date >= cutoff)
                rows = q.order_by(DBSR.file_date).all()
            elif metric == "linearity_fail_fraction":
                from sqlalchemy import func as _fn, case as _case
                q = (s.query(DBAR.file_date,
                             _fn.avg(_case((DBAR.overall_status == StatusType.FAIL, 1.0),
                                           else_=0.0)))
                     .filter(DBAR.model == model,
                             DBAR.overall_status.in_([StatusType.PASS, StatusType.WARNING,
                                                      StatusType.FAIL])))
                if cutoff:
                    q = q.filter(DBAR.file_date >= cutoff)
                rows = q.group_by(_fn.date(DBAR.file_date)).order_by(DBAR.file_date).all()
            elif metric == "ft_fail_fraction":
                # Daily FINAL-TEST fail rate — same shape as the detector's
                # lot observations, on the FT date axis. COALESCE: some FT
                # files never parse a test_date cell; file_date covers them
                # (code-review finding #4).
                from sqlalchemy import func as _fn, case as _case
                from laser_trim_analyzer.database.models import FinalTestResult as DBFT
                ft_date = _fn.coalesce(DBFT.test_date, DBFT.file_date)
                q = (s.query(ft_date,
                             _fn.avg(_case((DBFT.overall_status == StatusType.FAIL, 1.0),
                                           else_=0.0)))
                     .filter(DBFT.model == model,
                             ft_date.isnot(None),
                             ft_date > datetime(2000, 1, 1),
                             DBFT.overall_status.in_([StatusType.PASS, StatusType.WARNING,
                                                      StatusType.FAIL])))
                if cutoff:
                    q = q.filter(ft_date >= cutoff)
                rows = q.group_by(_fn.date(ft_date)).order_by(ft_date).all()
            elif metric == "escape_fraction":
                # Daily escape rate: of confidently-linked FT records whose
                # trim was ACCEPTED, the share that failed final test.
                from sqlalchemy import func as _fn, case as _case
                from laser_trim_analyzer.database.models import FinalTestResult as DBFT
                from laser_trim_analyzer.ml.drift_training import ESCAPE_MIN_CONFIDENCE
                ft_date = _fn.coalesce(DBFT.test_date, DBFT.file_date)
                q = (s.query(ft_date,
                             _fn.avg(_case((DBFT.overall_status == StatusType.FAIL, 1.0),
                                           else_=0.0)))
                     .join(DBAR, DBFT.linked_trim_id == DBAR.id)
                     .filter(DBFT.model == model,
                             ft_date.isnot(None),
                             ft_date > datetime(2000, 1, 1),
                             DBFT.match_confidence >= ESCAPE_MIN_CONFIDENCE,
                             DBFT.overall_status.in_([StatusType.PASS, StatusType.FAIL]),
                             DBAR.overall_status.in_([StatusType.PASS, StatusType.WARNING])))
                if cutoff:
                    q = q.filter(ft_date >= cutoff)
                rows = q.group_by(_fn.date(ft_date)).order_by(ft_date).all()
            elif metric in TRACK_METRIC_COLUMNS:
                col = TRACK_METRIC_COLUMNS[metric]      # Q4: SAME column the detector trained on
                q = (s.query(DBAR.file_date, col).join(DBTR, DBTR.analysis_id == DBAR.id)
                     .filter(DBAR.model == model, col.isnot(None)))
                if cutoff:
                    q = q.filter(DBAR.file_date >= cutoff)
                rows = q.order_by(DBAR.file_date).all()
            else:
                rows = []
            # Q5; _coerce_dt: SQLite hands COALESCE dates back as strings.
            from laser_trim_analyzer.ml.drift_training import _coerce_dt
            pairs = [(_coerce_dt(r[0]), r[1]) for r in rows
                     if r[0] is not None and r[1] is not None]
            ms = s.query(ModelMetricState).filter_by(model=model, metric=metric).first()
            baseline = (ms.baseline_mean, ms.baseline_std) if ms else (None, None)
        return [p[0] for p in pairs], [p[1] for p in pairs], baseline

    def _compute_verdict(self, model, cutoff, status, recent_means) -> tuple:
        """One-line answer to the daily question: (text, color).

        Drift state from the trained detectors; direction = window linearity
        yield vs the model's lifetime; difficulty = lifetime yield in plain
        words. Everything shown is verifiable on the tabs below it.
        """
        from sqlalchemy import case as _case, func as _f
        t = self.theme

        def _yield(cut):
            with self.app.db.session() as s_:
                q = (s_.query(
                        _f.sum(_case((DBAR.overall_status.in_(
                            [StatusType.PASS, StatusType.WARNING]), 1), else_=0)),
                        _f.sum(_case((DBAR.overall_status.in_(
                            [StatusType.PASS, StatusType.WARNING, StatusType.FAIL]), 1), else_=0)))
                     .filter(DBAR.model == model))
                if cut is not None:
                    q = q.filter(DBAR.file_date >= cut)
                acc, grad = q.first()
            return (acc / grad * 100.0) if grad else None, (grad or 0)

        win_y, win_n = _yield(cutoff)
        life_y, life_n = _yield(None)

        # Drift state: worst non-stable watched metric by honest shift.
        worst = None
        trained = 0
        for m, ms in (getattr(status, "per_metric", {}) or {}).items():
            trained += 1
            tier = getattr(ms.tier, "name", str(ms.tier))
            if tier in ("STABLE",):
                continue
            rv = recent_means.get(m) if recent_means.get(m) is not None else ms.recent_mean
            shift = ((rv - ms.baseline_mean) / ms.baseline_std
                     if (rv is not None and ms.baseline_std) else None)
            key = abs(shift) if shift is not None else 0.0
            if worst is None or key > worst[2]:
                worst = (m, shift, key, tier)

        if trained == 0:
            state_txt, color = "NOT TRAINED — run drift training in Settings", t.TEXT_DISABLED
        elif worst is None:
            state_txt, color = "HOLDING — all watched metrics stable", t.TEXT_PRIMARY
        else:
            from laser_trim_analyzer.ml.drift_types import metric_label as _ml
            shift_txt = f"{worst[1]:+.1f}σ" if worst[1] is not None else "flagged"
            state_txt = f"DRIFTING — {_ml(worst[0])}: last lot {shift_txt} vs baseline lots"
            color = t.TIER_OOC if worst[3] == "OUT_OF_CONTROL" else t.TIER_DRIFT

        parts = [state_txt]
        try:
            from laser_trim_analyzer.core.yield_stats import compute_unit_yield
            u = compute_unit_yield(self.app.db, cutoff, model=model)
            if u.get("gradeable_units"):
                parts.append(
                    f"first-pass {u['first_pass_yield']:.0f}% → final "
                    f"{u['final_yield']:.0f}% ({u['attempts_per_section']:.2f} trims/section)")
        except Exception:
            logger.exception("verdict unit yield failed")
        if win_y is not None and life_y is not None and cutoff is not None:
            d = win_y - life_y
            trend = "better than" if d > 2 else ("worse than" if d < -2 else "in line with")
            parts.append(f"window yield {win_y:.0f}% ({win_n} units) {trend} "
                         f"lifetime {life_y:.0f}%")
        if life_y is not None:
            difficulty = ("historically difficult" if life_y < 75
                          else "historically mixed" if life_y < 90
                          else "historically strong")
            parts.append(f"{difficulty} ({life_y:.0f}% lifetime linearity yield, "
                         f"{life_n:,} unit{'s' if life_n != 1 else ''})")
        return "  ·  ".join(parts), color

    def _load_ft_units(self, model, cutoff) -> list:
        """Final-test records for the model (work finding #3, 2026-07-10:
        'no way to view final test units'). Newest first, capped at 500."""
        from laser_trim_analyzer.database.models import FinalTestResult as DBFT
        with self.app.db.session() as s:
            q = s.query(DBFT.serial, DBFT.file_date, DBFT.overall_status,
                        DBFT.linked_trim_id, DBFT.match_confidence)\
                 .filter(DBFT.model == model)
            if cutoff is not None:
                q = q.filter(DBFT.file_date >= cutoff)
            rows = q.order_by(DBFT.file_date.desc()).limit(500).all()
        return [{"serial": r[0], "file_date": r[1],
                 "result": getattr(r[2], "value", str(r[2])),
                 "linked": r[3] is not None,
                 "match": (round(r[4] * 100) if r[4] is not None else None)}
                for r in rows]

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

    def _on_selector_wheel(self, event, step=None):
        values = list(self._model_selector.cget("values") or [])
        if not values:
            return
        if step is None:  # Windows/mac <MouseWheel>: delta sign gives direction
            step = -1 if getattr(event, "delta", 0) > 0 else 1
        cur = self._model_selector.get()
        try:
            idx = values.index(cur)
        except ValueError:
            idx = -1 if step > 0 else 0
        new_val = values[max(0, min(len(values) - 1, idx + step))]
        if new_val != cur:
            self._model_selector.set(new_val)
            self._on_model_selected(new_val)

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

    def _on_requalify(self):
        """Per-model baseline requalification (design change). Confirm with
        effective date + reason, record the audit row, retrain THIS model
        from the effective date forward, reload."""
        model = self._current_model
        if not model:
            return
        from datetime import date as _date
        import tkinter as tk

        dlg = ctk.CTkToplevel(self)
        dlg.title(f"Requalify baseline — {model}")
        dlg.geometry("520x260")
        dlg.transient(self.winfo_toplevel())
        t = self.theme
        ctk.CTkLabel(dlg, text=(f"Reset {model}'s drift baselines because the design/"
                                "process changed. Data BEFORE the effective date is "
                                "excluded from the new baselines. If too little data "
                                "exists after the date, metrics read NOT TRAINED until "
                                "enough new lots accumulate. This action is recorded."),
                     font=t.font(t.SIZE_BODY), wraplength=480, justify="left",
                     text_color=t.TEXT_PRIMARY).pack(padx=16, pady=(16, 8), anchor="w")
        row1 = ctk.CTkFrame(dlg, fg_color="transparent"); row1.pack(fill="x", padx=16)
        ctk.CTkLabel(row1, text="Effective date (YYYY-MM-DD):", font=t.font(t.SIZE_BODY),
                     text_color=t.TEXT_SECONDARY).pack(side="left")
        date_e = ctk.CTkEntry(row1, width=130); date_e.pack(side="left", padx=8)
        date_e.insert(0, _date.today().isoformat())
        row2 = ctk.CTkFrame(dlg, fg_color="transparent"); row2.pack(fill="x", padx=16, pady=8)
        ctk.CTkLabel(row2, text="Reason (audit note):", font=t.font(t.SIZE_BODY),
                     text_color=t.TEXT_SECONDARY).pack(side="left")
        note_e = ctk.CTkEntry(row2, width=280); note_e.pack(side="left", padx=8)
        status_lbl = ctk.CTkLabel(dlg, text="", font=t.font(t.SIZE_CAPTION),
                                  text_color=t.TIER_WARNING)
        status_lbl.pack(padx=16, anchor="w")

        def go():
            from datetime import datetime as _dt
            raw = date_e.get().strip()
            try:
                eff = _dt.fromisoformat(raw)
            except ValueError:
                status_lbl.configure(text="Date must be YYYY-MM-DD.")
                return
            note = note_e.get().strip()
            status_lbl.configure(text="Requalifying + retraining this model…")
            def work():
                try:
                    self.app.db.set_baseline_requalification(model, eff.date().isoformat(), note)
                    from laser_trim_analyzer.ml.drift_training import train_drift_detector
                    preset = getattr(self.app.config.ml, "drift_sensitivity", "standard")
                    train_drift_detector(self.app.db, sensitivity_preset=preset, model=model)
                except Exception:
                    logger.exception("requalification failed for %s", model)
                    self.safe_after(lambda: status_lbl.winfo_exists() and status_lbl.configure(
                        text="Failed — see the log."))
                    return
                def done():
                    try:
                        dlg.destroy()
                    except Exception:
                        pass
                    self._reload()
                self.safe_after(done)
            threading.Thread(target=work, daemon=True).start()

        btns = ctk.CTkFrame(dlg, fg_color="transparent"); btns.pack(fill="x", padx=16, pady=12)
        ctk.CTkButton(btns, text="Requalify + retrain", fg_color=t.ACCENT,
                      hover_color=t.ACCENT_HOVER, text_color=t.TEXT_INVERSE,
                      command=go, corner_radius=t.RADIUS_SM).pack(side="right")
        ctk.CTkButton(btns, text="Cancel", fg_color=t.CARD, hover_color=t.ELEVATED,
                      text_color=t.TEXT_PRIMARY, border_width=1, border_color=t.BORDER,
                      command=dlg.destroy, corner_radius=t.RADIUS_SM).pack(side="right", padx=8)

    def _on_export_charts(self):
        """Export EVERY unit chart in the window as print-ready PNGs (work
        finding #4: 'no way to mass export charts'). One folder pick, then a
        worker renders build_unit_export_figure per unit."""
        if not self._current_model:
            return
        from tkinter import filedialog
        out_dir = filedialog.askdirectory(title="Folder for unit charts")
        if not out_dir:
            return
        model = self._current_model

        def work():
            import matplotlib
            matplotlib.use("Agg", force=False)
            from pathlib import Path as _P
            from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import (
                load_unit_track, compute_fail_points)
            from laser_trim_analyzer.export.unit_chart import build_unit_export_figure
            units = self._load_units(model)
            done = err = 0
            for i, u in enumerate(units):
                try:
                    data = load_unit_track(self.app.db, u["analysis_id"])
                    fp = compute_fail_points(
                        data["error_data"], data["upper_limits"], data["lower_limits"],
                        offset=data.get("optimal_offset") or 0.0)
                    date_s = (u["file_date"].strftime("%Y-%m-%d")
                              if u.get("file_date") else "nodate")
                    fig = build_unit_export_figure(
                        meta={"model": data.get("model") or model,
                              "serial": u.get("serial"),
                              "system": data.get("system"),
                              "trim_date": date_s,
                              "track_id": data.get("track_id"),
                              "n_tracks": data.get("n_tracks")},
                        data=data, fail_points=fp)
                    safe_serial = str(u.get("serial") or "unknown").replace("/", "-")
                    fig.savefig(_P(out_dir) / f"{model}_{safe_serial}_{date_s}.png",
                                dpi=150, facecolor=fig.get_facecolor())
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    done += 1
                except Exception:
                    logger.exception("chart export failed for %s", u.get("serial"))
                    err += 1
                if i % 10 == 0:
                    self.safe_after(lambda d=done, n=len(units): self._units_tab.set_caption(
                        f"Exporting charts… {d}/{n}"))
            self.safe_after(lambda: self._units_tab.set_caption(
                f"Exported {done} chart(s) to {out_dir}"
                + (f" · {err} failed (see log)" if err else "")))
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
