"""Spec 3c — Model page. Foundations §3/§4.3. Fixtures in tests/conftest.py."""

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
