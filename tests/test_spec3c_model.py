"""Spec 3c — Model page. Foundations §3/§4.3. Fixtures in tests/conftest.py."""
from datetime import datetime, timedelta

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


def test_model_page_banners_a_trim_vs_ft_spec_mismatch(make_app):
    """James (2026-08-30): "i also want to know when the trim and test specs
    dont align." The banner has to say WHAT differs and WHY that matters to
    the numbers already on the page — not just that something is wrong."""
    from laser_trim_analyzer.core.spec_alignment import SpecComparison
    app = make_app()
    page = app.page_container.get_page("model")
    page._set_spec_banner(SpecComparison(
        status="differs", pct_positions_differing=1.0, matched_positions=200,
        trim_typ_band=0.03, ft_typ_band=0.10,
        note=("trim grades ±0.030 V where final test allows ±0.100 V "
              "(100% of matched points differ)")))
    assert page._spec_banner.cget("text") == (
        "⚠ trim grades ±0.030 V where final test allows ±0.100 V "
        "(100% of matched points differ) — cross-station numbers "
        "(escapes, Gap) compare different requirements.")
    assert page._spec_banner.winfo_manager() == "pack"
    # It qualifies the VERDICT, so it must stay directly under it — pack()
    # would otherwise re-append it at the bottom of the body when re-shown.
    slaves = page._verdict.master.pack_slaves()
    assert slaves.index(page._spec_banner) == slaves.index(page._verdict) + 1


def test_model_page_says_nothing_when_the_stations_agree_or_are_unknown(make_app):
    """Amber on an unanswered question is how a warning stops being believed."""
    from laser_trim_analyzer.core.spec_alignment import SpecComparison
    app = make_app()
    page = app.page_container.get_page("model")
    page._set_spec_banner(SpecComparison(
        status="differs", pct_positions_differing=1.0, matched_positions=200,
        trim_typ_band=0.03, ft_typ_band=0.10, note="n"))
    for quiet in (SpecComparison("aligned", 0.0, 200, 0.03, 0.03, "same"),
                  SpecComparison("insufficient", 0.0, 0, None, None, "no data"),
                  None):                       # loader failed — say nothing
        page._set_spec_banner(quiet)
        assert page._spec_banner.winfo_manager() == ""


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
    """Recent column = the last CLOSED lot's median (lot mode, 2026-07-13).
    Units dated today form an OPEN lot and must NOT be the recent value —
    so the fixture backdates them past the changeover gap."""
    from datetime import datetime, timedelta
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    app = make_app()
    when = datetime.now() - timedelta(days=10)   # closed lot (gap > 3 days)
    with app.db.session() as s:
        for i, val in enumerate((0.0040, 0.0044)):
            ar = DBAR(filename=f"r{i}.xls", file_path="/f/x.xls", file_hash=f"hr{i}",
                      model="RM", serial=f"sn{i}", system=SystemType.A, file_date=when,
                      timestamp=when, overall_status=StatusType.PASS,
                      has_multi_tracks=False, processing_time=0.1)
            s.add(ar); s.flush()
            s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                       final_linearity_error_shifted=val))
        s.commit()
    page = app.page_container.get_page("model")
    means = page._recent_means("RM")
    assert means["linearity_error"] == pytest.approx(0.0042)   # lot median


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


# ---- FOCUS/SPC redesign (2026-08-29): the lot chart is the headline view ---
# Production runs in LOTS, so a lot — not a unit — is what goes in or out of
# control. The Model page opens on the SPC lot chart for every metric; the
# per-unit scatter that used to be the only view is one click away.
# Seeding mirrors tests/test_spc_db.py (real DatabaseManager, real compute).

D0 = datetime(2026, 1, 5)      # SPC anchors on the DATA's newest date, not "now"


def _add_lot(db, model, day, n, fails):
    """One production lot: n gradeable units on one file_date, `fails` of them FAIL."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, StatusType, SystemType)
    with db.session() as s:
        for i in range(n):
            # `system` is NOT NULL in the schema; the SPC query never reads it.
            s.add(DBAR(model=model, serial=f"{model}-{day:%m%d}-{i}", system=SystemType.A,
                       filename=f"{model}_{i}_{day:%m-%d-%Y}.xls", file_date=day,
                       overall_status=StatusType.FAIL if i < fails else StatusType.PASS))


def _seed(db, model, n_lots=12, fails_last=0, start=D0, n_per=20, base_fails=2):
    """Weekly lots at a steady baseline rate, then one lot at `fails_last`."""
    for k in range(n_lots - 1):
        _add_lot(db, model, start + timedelta(days=7 * k), n_per, base_fails)
    last_day = start + timedelta(days=7 * (n_lots - 1))
    _add_lot(db, model, last_day, n_per, fails_last if fails_last else base_fails)
    return last_day


def _seed_tracks(db, model, metric, n_lots=12, n_per=3, start=D0):
    """Weekly lots carrying a CONTINUOUS metric, drifting lot-to-lot so the
    baseline has a real spread (a flat baseline is degenerate -> unjudged)."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType, SystemType)
    from laser_trim_analyzer.ml.drift_training import TRACK_METRIC_COLUMNS
    col = TRACK_METRIC_COLUMNS[metric].key
    with db.session() as s:
        for k in range(n_lots):
            day = start + timedelta(days=7 * k)
            for i in range(n_per):
                ar = DBAR(model=model, serial=f"{model}-{k}-{i}", system=SystemType.A,
                          filename=f"{model}_{k}_{i}.xls", file_date=day,
                          overall_status=StatusType.PASS)
                s.add(ar); s.flush()
                # TrackResult.status is NOT NULL — set it on any committed row.
                s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                           **{col: 0.010 + 0.0004 * k + 0.0001 * i}))
        s.commit()


def _spc_app(make_app, monkeypatch, metric="linearity_fail_fraction", seed=None):
    """App routed to a drifting model, with the HEADLINE chart's draws recorded.

    The spies go on the page's own FocusChart INSTANCE, not the class: the
    Smoothness tab embeds a second FocusChart and draws into it on the same
    reload, so a class-level patch would mix the two charts' calls together.
    """
    calls = []
    app = make_app()
    (seed or (lambda db: _seed(db, "HOT", fails_last=12)))(app.db)
    app.set_model_route("HOT", metric)
    page = app.page_container.get_page("model")
    # on_show kicks off a BACKGROUND reload. Letting it race the synchronous
    # one below adds no coverage and makes the two threads fight over the DB's
    # single StaticPool connection (one RLock held for a whole session) — worth
    # ~2.5 min per run of this file. Suppress it; load once, synchronously.
    page._reload = lambda **kw: None
    app.show_page("model")                 # the real deep-link path
    del page._reload                       # back to the real bound method
    monkeypatch.setattr(page._focus_chart, "set_spc_series",
                        lambda series, **kw: calls.append(("spc", series)))
    monkeypatch.setattr(page._focus_chart, "set_series",
                        lambda **kw: calls.append(("units", kw)))
    page.reload_now()                      # synchronous path for tests
    return app, page, calls


def test_model_page_opens_on_the_spc_lot_chart(make_app, monkeypatch):
    """The headline chart is the LOT p-chart for the metric that routed here."""
    app, page, calls = _spc_app(make_app, monkeypatch)
    assert page._chart_view == "lots"
    assert [kind for kind, _ in calls] == ["spc"]      # units view NOT drawn
    series = calls[0][1]
    assert series.model == "HOT" and series.metric == "linearity_fail_fraction"
    assert series.judged and series.points             # real limits, real lots
    assert series.points[-1].ooc                       # the blown-out last lot


def test_model_page_units_toggle_draws_the_unit_view(make_app, monkeypatch):
    app, page, calls = _spc_app(make_app, monkeypatch)
    calls.clear()
    page._on_chart_view_change("Units")
    assert page._chart_view == "units"
    assert [kind for kind, _ in calls] == ["units"]
    assert calls[0][1]["metric"] == "linearity_fail_fraction"
    calls.clear()
    page._on_chart_view_change("Lots · SPC")           # and back
    assert page._chart_view == "lots"
    assert [kind for kind, _ in calls] == ["spc"]


def test_chart_toggle_re_renders_loaded_data_without_touching_the_db(make_app, monkeypatch):
    """Both series load in ONE _reload pass; the toggle is a view switch."""
    import laser_trim_analyzer.gui.v6.pages.model_page as mp
    app, page, calls = _spc_app(make_app, monkeypatch)

    def boom(*a, **k):
        raise AssertionError("toggle must re-render loaded data, not re-query")
    monkeypatch.setattr(mp, "compute_spc_series", boom)
    monkeypatch.setattr(page, "_load_focus_series", boom)
    page._on_chart_view_change("Units")
    page._on_chart_view_change("Lots · SPC")


def test_continuous_metric_also_opens_on_the_lot_chart(make_app, monkeypatch):
    """Default is Lots for EVERY metric — compute_spc_series routes a
    continuous metric through build_continuous_series on its own."""
    app, page, calls = _spc_app(
        make_app, monkeypatch, metric="untrimmed_sigma_gradient",
        seed=lambda db: _seed_tracks(db, "HOT", "untrimmed_sigma_gradient"))
    assert [kind for kind, _ in calls] == ["spc"]
    series = calls[0][1]
    assert series.metric == "untrimmed_sigma_gradient"
    assert series.judged and len(series.points) == 12   # lot medians, not units
