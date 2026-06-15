"""Spec 3c — Model page. Foundations §3/§4.3. Fixtures in tests/conftest.py."""
import pytest

# ---- Task 1: routing + column map ----------------------------------------

def test_consume_model_route_full(make_app):
    app = make_app()
    app.set_model_route("M1", "linearity_error")
    assert app.consume_model_route_full() == ("M1", "linearity_error")
    assert app.consume_model_route_full() == (None, None)


def test_consume_model_route_full_without_focus(make_app):
    app = make_app()
    app.set_model_route("M2")
    assert app.consume_model_route_full() == ("M2", None)


def test_track_metric_columns_public_and_linearity_maps_to_shifted():
    from laser_trim_analyzer.ml.drift_training import TRACK_METRIC_COLUMNS
    from laser_trim_analyzer.database.models import TrackResult as DBTR
    # Q4: the detector trains linearity_error on final_linearity_error_shifted; the UI must match.
    assert TRACK_METRIC_COLUMNS["linearity_error"] is DBTR.final_linearity_error_shifted
    assert "max_smoothness_value" not in TRACK_METRIC_COLUMNS  # lives on SmoothnessResult


# ---- Task 2: ThemedTabView ------------------------------------------------

def test_themed_tab_view(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.tab_view import ThemedTabView
    tv = ThemedTabView(tk_root, theme=ThemeManager())
    assert tv.add("Drift Metrics") is not None
    tv.add("Units"); tv.set("Units")
    assert tv.get() == "Units"


# ---- Task 3: MetricPillRow ------------------------------------------------

def _status(model="M1", **tiers):
    """Build a ModelDriftStatus; pass metric=Tier kwargs to override specific metrics."""
    from datetime import datetime
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, MetricStatus, ModelDriftStatus, WATCHED_METRICS)
    per = {}
    for m in WATCHED_METRICS:
        tier = tiers.get(m, DriftTier.STABLE)
        per[m] = MetricStatus(metric=m, tier=tier, alert_type=None,
                              magnitude=0.0 if tier == DriftTier.STABLE else 3.1,
                              baseline_mean=0.01, baseline_std=0.001,
                              recent_mean=0.012, recent_count=5, is_trained=True)
    return ModelDriftStatus(model=model, overall_tier=DriftTier.STABLE, worst_metric=None,
                            worst_alert_type=None, per_metric=per, last_processed=datetime.now())


def test_pill_row_has_eight_pills(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
    from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS
    row = MetricPillRow(tk_root, theme=ThemeManager(), on_pill_click=lambda _: None)
    row.set_status(_status())
    assert set(row._pills) == set(WATCHED_METRICS)


def test_pill_shows_readable_label(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
    row = MetricPillRow(tk_root, theme=ThemeManager(), on_pill_click=lambda _: None)
    row.set_status(_status())
    assert row._pills["untrimmed_resistance"]._name_label.cget("text") == "Untrimmed resistance"


def test_pill_click_and_select(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
    got = []
    row = MetricPillRow(tk_root, theme=ThemeManager(), on_pill_click=got.append)
    row.set_status(_status())
    # DEVIATION: sigma_gradient is no longer a watched metric (replaced by
    # untrimmed_sigma_gradient + composite_trim_risk_score). Use a real pill key.
    row._pills["untrimmed_sigma_gradient"]._on_click()
    assert got == ["untrimmed_sigma_gradient"]
    row.set_selected("linearity_error")
    assert row._selected_metric == "linearity_error"


# ---- Task 4: FocusChart ---------------------------------------------------

def test_focus_chart_set_series_no_crash(tk_root):
    from datetime import datetime, timedelta
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
    chart = FocusChart(tk_root, theme=ThemeManager())
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(20, 0, -1)]
    values = [0.01 + 0.0001 * i for i in range(20)]
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                     baseline_mean=0.011, baseline_std=0.0005)


def test_focus_chart_empty_no_crash(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
    chart = FocusChart(tk_root, theme=ThemeManager())
    chart.set_series(metric="linearity_error", dates=[], values=[])  # empty state, no raise


# ---- Task 5: DriftMetricsTab ----------------------------------------------

def test_drift_tab_row_per_metric_and_click(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import DriftMetricsTab
    from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS
    got = []
    tab = DriftMetricsTab(tk_root, theme=ThemeManager(), on_metric_select=got.append)
    tab.set_status(_status())
    assert set(tab._rows) == set(WATCHED_METRICS)
    # DEVIATION: untrimmed_sigma_gradient (sigma_gradient no longer watched).
    tab._rows["untrimmed_sigma_gradient"]._on_click()
    assert got == ["untrimmed_sigma_gradient"]


# ---- Task 6: SmoothnessTab ------------------------------------------------

def test_smoothness_tab_empty(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import SmoothnessTab
    SmoothnessTab(tk_root, theme=ThemeManager()).set_records([])  # no raise


def test_smoothness_tab_with_records(tk_root):
    from datetime import datetime
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import SmoothnessTab
    tab = SmoothnessTab(tk_root, theme=ThemeManager())
    tab.set_records([{"serial": "sn1", "file_date": datetime.now(),
                      "max_smoothness_value": 0.4, "avg_smoothness_value": 0.2}])
    assert len(tab._rows) == 1


# ---- Task 7: UnitsTab + UnitChartModal -----------------------------------

def test_units_tab_row_per_unit_keeps_duplicate_serials(tk_root):
    from datetime import datetime
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab
    tab = UnitsTab(tk_root, theme=ThemeManager(), on_unit_click=lambda u: None, on_export=lambda: None)
    # Same serial twice = two valid trims (Q2). Both rows must appear.
    units = [{"analysis_id": 1, "serial": "sn1", "file_date": datetime.now(),
              "overall_status": "Pass", "sigma_gradient": 0.01, "linearity_error": 0.004},
             {"analysis_id": 2, "serial": "sn1", "file_date": datetime.now(),
              "overall_status": "Fail", "sigma_gradient": 0.02, "linearity_error": 0.05}]
    tab.set_units(units)
    assert len(tab._rows) == 2


def test_unit_chart_modal_marks_fail_points(tk_root):
    """Q1: every point shown; out-of-limit points become fail_points."""
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import compute_fail_points
    err = [0.0, 0.5, -0.2, 0.9]
    upper = [0.4, 0.4, 0.4, 0.4]; lower = [-0.4, -0.4, -0.4, -0.4]
    assert compute_fail_points(err, upper, lower) == [1, 3]  # 0.5>0.4 and 0.9>0.4


# ---- Task 8: PredictorPanel -----------------------------------------------

def test_predictor_panel_collapsed_then_toggles(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.predictor_panel import PredictorPanel
    p = PredictorPanel(tk_root, theme=ThemeManager(), load_fn=lambda model: "Risk: LOW (demo)")
    assert p._expanded is False
    p.set_model("8340-1")
    p.toggle()                 # expand → triggers lazy load
    assert p._expanded is True
    assert "Risk" in p._body_label.cget("text")


def test_predictor_panel_load_failure_is_graceful(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.predictor_panel import PredictorPanel
    def boom(model): raise RuntimeError("no predictor")
    p = PredictorPanel(tk_root, theme=ThemeManager(), load_fn=boom)
    p.set_model("X"); p.toggle()
    assert "No predictor" in p._body_label.cget("text")


# ---- Task 9: evidence export ----------------------------------------------

def test_build_summary_text_has_evidence_metrics():
    from laser_trim_analyzer.export.evidence import build_summary_text
    txt = build_summary_text("8340-1", _status())
    assert "8340-1" in txt
    # Q8: the three evidence metrics James hands engineers must be present, readable.
    for label in ("Untrimmed resistance", "Linearity error", "Electrical angle"):
        assert label in txt


def test_export_evidence_pack_writes_xlsx(tmp_path):
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    from laser_trim_analyzer.export.evidence import export_evidence_pack
    db = DatabaseManager(tmp_path / "ev.db")
    with db.session() as s:
        ar = DBAR(filename="x.xls", file_path="/f/x.xls", file_hash="hx", model="8340-1", serial="sn1",
                  system=SystemType.A, file_date=datetime.now(), timestamp=datetime.now(),
                  overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1)
        s.add(ar); s.flush()
        # TrackResult.status is NOT NULL — set it on any committed row.
        s.add(DBTR(analysis_id=ar.id, track_id="TRK1", status=StatusType.PASS,
                   sigma_gradient=0.01, final_linearity_error_shifted=0.004))
        s.commit()
    out = tmp_path / "pack.xlsx"
    export_evidence_pack(db, "8340-1", out, window_days=365)
    assert out.exists() and out.stat().st_size > 0


# ---- Task 10: ModelPage ---------------------------------------------------

def test_model_page_consumes_route_on_show(make_app):
    from datetime import datetime
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, SystemType, StatusType
    app = make_app()
    with app.db.session() as s:
        s.add(DBAR(filename="x.xls", file_path="/f/x.xls", file_hash="hx", model="ROUTED-MODEL",
                   serial="sn1", system=SystemType.A, file_date=datetime.now(), timestamp=datetime.now(),
                   overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1))
        s.commit()
    app.set_model_route("ROUTED-MODEL", "linearity_error")
    app.show_page("model")
    page = app.page_container.get_page("model")
    assert page._current_model == "ROUTED-MODEL"
    assert page._current_metric == "linearity_error"


def test_model_page_empty_state_when_no_model(make_app):
    app = make_app()
    app.show_page("model")        # no route set
    page = app.page_container.get_page("model")
    assert page._current_model is None
    assert page._empty_label.winfo_ismapped() or page._empty_label.winfo_exists()


def test_model_page_focus_series_uses_shifted_linearity(make_app):
    """Q4: requesting linearity_error reads final_linearity_error_shifted."""
    from datetime import datetime
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    app = make_app()
    with app.db.session() as s:
        ar = DBAR(filename="x.xls", file_path="/f/x.xls", file_hash="hx", model="QM", serial="sn1",
                  system=SystemType.A, file_date=datetime.now(), timestamp=datetime.now(),
                  overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1)
        s.add(ar); s.flush()
        # TrackResult.status is NOT NULL — set it on any committed row.
        s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                   final_linearity_error_shifted=0.0042))
        s.commit()
    page = app.page_container.get_page("model")
    dates, values, baseline = page._load_focus_series("QM", "linearity_error")
    assert values == [0.0042]


# ---- Dashboard-round Model fixes ------------------------------------------

def test_resolve_focus_metric_prefers_worst_when_not_user_picked():
    from laser_trim_analyzer.gui.v6.pages.model_page import ModelPage
    from laser_trim_analyzer.ml.drift_types import DriftTier, ModelDriftStatus
    status = ModelDriftStatus(model="M", overall_tier=DriftTier.OUT_OF_CONTROL,
                              worst_metric="trim_pass_count", worst_alert_type=None, per_metric={})
    # not user-picked -> worst metric wins
    assert ModelPage._resolve_focus_metric(status, False, "untrimmed_sigma_gradient") == "trim_pass_count"
    # user picked -> keep their choice
    assert ModelPage._resolve_focus_metric(status, True, "linearity_error") == "linearity_error"
    # no worst (all stable) -> keep current fallback
    stable = ModelDriftStatus(model="M", overall_tier=DriftTier.STABLE, worst_metric=None,
                              worst_alert_type=None, per_metric={})
    assert ModelPage._resolve_focus_metric(stable, False, "untrimmed_sigma_gradient") == "untrimmed_sigma_gradient"


def test_model_recent_means_computed_from_data(make_app):
    """Recent column comes from the actual recent-window data, not the detector
    (which never persists a recent mean)."""
    from datetime import datetime
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    app = make_app()
    with app.db.session() as s:
        for i, val in enumerate((0.0040, 0.0044)):
            ar = DBAR(filename=f"r{i}.xls", file_path="/f/x.xls", file_hash=f"hr{i}",
                      model="RM", serial=f"sn{i}", system=SystemType.A, file_date=datetime.now(),
                      timestamp=datetime.now(), overall_status=StatusType.PASS,
                      has_multi_tracks=False, processing_time=0.1)
            s.add(ar); s.flush()
            s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                       final_linearity_error_shifted=val))
        s.commit()
    page = app.page_container.get_page("model")
    means = page._recent_means("RM")
    assert means["linearity_error"] == pytest.approx(0.0042)   # mean(0.0040, 0.0044)


def test_drift_tab_uses_grid_columns(tk_root):
    """Header and rows lay their cells out on a shared grid, so columns line up."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import DriftMetricsTab, _COLUMNS
    tab = DriftMetricsTab(tk_root, theme=ThemeManager(), on_metric_select=lambda _: None)
    tab.set_status(_status())
    row = tab._rows["linearity_error"]
    slaves = row.grid_slaves()
    assert len(slaves) == len(_COLUMNS)                                  # cells gridded, not packed
    assert sorted(int(w.grid_info()["column"]) for w in slaves) == list(range(len(_COLUMNS)))
