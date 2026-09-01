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
        note=("100% of the positions both stations measure are graded to "
              "different limits (trim ±0.030 V, final test ±0.100 V)")))
    assert page._spec_banner.cget("text") == (
        "⚠ 100% of the positions both stations measure are graded to "
        "different limits (trim ±0.030 V, final test ±0.100 V) — "
        "cross-station numbers (escapes, Gap) compare different "
        "requirements at those positions.")
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


# ---- INVESTIGATE: stats table + lot-vs-history (app-shape spec §2) --------

def _stats_row(key="untrimmed_resistance", *, n=3, avg=4281.8, low=422.0,
               high=29576.0, excluded=7, missing=1):
    from laser_trim_analyzer.core.model_stats import Cell, StatRow
    cell = Cell(n=n, excluded=excluded, missing=missing, avg=avg, low=low, high=high)
    return StatRow(key=key, label="Untrimmed resistance", unit="ohms",
                   kind="distribution", all_=cell, lin_passing=cell)


def test_stats_cells_render_one_unit_per_row():
    from laser_trim_analyzer.gui.v6.widgets.stats_table import cell_texts
    row = _stats_row()
    assert cell_texts(row, row.all_) == ["3", "4.28 kΩ", "0.422 kΩ", "29.6 kΩ"]


def test_rate_cells_render_count_and_percent():
    from laser_trim_analyzer.core.model_stats import Cell, StatRow
    from laser_trim_analyzer.gui.v6.widgets.stats_table import cell_texts
    cell = Cell(n=9934, excluded=0, missing=10, count=6325, pct=63.67)
    row = StatRow(key="trim_passed_linearity", label="Tracks that passed linearity",
                  unit="%", kind="rate", all_=cell, lin_passing=cell)
    assert cell_texts(row, cell) == ["9,934", "6,325", "63.7%"]


def test_the_table_says_what_it_left_out():
    """Disclose, never hide — every drop is named on the row that made it,
    and the three reasons send you to three different places."""
    from laser_trim_analyzer.core.model_stats import Cell
    from laser_trim_analyzer.gui.v6.widgets.stats_table import disclosure_text
    row = _stats_row()
    assert disclosure_text(row.all_) == "7 impossible readings excluded · 1 not recorded"
    assert disclosure_text(_stats_row(excluded=0, missing=0).all_) == ""
    assert disclosure_text(_stats_row(excluded=1, missing=0).all_) \
        == "1 impossible reading excluded"
    # The 8856 case: 75 records the analyser could not read. A row that showed
    # n=113 and said nothing would hide them.
    assert disclosure_text(Cell(n=113, excluded=0, missing=0, errored=75)) \
        == "75 from records that failed processing"


def test_summary_line_names_the_window_and_the_drops():
    from datetime import datetime
    from laser_trim_analyzer.core.model_stats import ModelStats
    from laser_trim_analyzer.gui.v6.widgets.stats_table import summary_line
    stats = ModelStats(model="6607", rows=[_stats_row()], tracks=9944, records=9944,
                       cutoff=None, lot=None, future_dated=0, note="")
    text = summary_line(stats)
    assert "9,944 track measurements over all history" in text
    assert "7 impossible readings left out" in text
    # Assert the WHOLE phrase, not just the date. Checking for "since May 13"
    # alone passed while the line actually read "measurements over since
    # May 13, 2026" — a weak assertion that let broken copy ship to the screen.
    windowed = ModelStats(model="6607", rows=[_stats_row()], tracks=302, records=302,
                          cutoff=datetime(2026, 5, 13), lot=None, future_dated=0,
                          note="")
    assert summary_line(windowed).startswith(
        "302 track measurements since May 13, 2026")
    lot_scoped = ModelStats(model="6607", rows=[_stats_row()], tracks=15, records=15,
                            cutoff=None, lot=object(), future_dated=0, note="")
    assert summary_line(lot_scoped).startswith(
        "15 track measurements over the selected lot")


def test_lot_line_carries_the_numbers_and_the_verdict():
    from laser_trim_analyzer.core.model_stats import Cell, LotVerdict
    from laser_trim_analyzer.gui.v6.widgets.stats_table import lot_line
    row = _stats_row()
    cell = Cell(n=69, excluded=0, missing=0, avg=4430.0, low=4100.0, high=4800.0)
    verdict = LotVerdict(metric="untrimmed_resistance", label="Untrimmed resistance",
                         status="within", lot_typical=4431.0, lot_n=69,
                         normal_low=1000.0, normal_high=9800.0,
                         text="Untrimmed resistance for this lot is within its normal")
    text = lot_line(row, cell, verdict)
    assert text.startswith("this lot: 69 readings · avg 4.43 kΩ · 4.10 kΩ to 4.80 kΩ")
    assert "within its normal" in text
    assert lot_line(row, Cell(n=0, excluded=0, missing=5), None) \
        == "this lot: nothing recorded"


def _stats_app(make_app, model="HOT"):
    """App on the Model page with a seeded model, loaded synchronously."""
    app = make_app()
    _seed_tracks(app.db, model, "untrimmed_resistance")
    app.set_model_route(model)
    page = app.page_container.get_page("model")
    page._reload = lambda **kw: None
    app.show_page("model")
    del page._reload
    page.reload_now()
    return app, page


def test_model_page_shows_the_stats_table(make_app):
    """The table replaces the Excel round trip, so it has to be ON the page."""
    app, page = _stats_app(make_app)
    texts = [w.cget("text") for w in _all_labels(page._stats_table)]
    assert any("track measurements" in t for t in texts)
    assert any("Untrimmed resistance" in t for t in texts)
    assert any("LIN-PASSING" in t for t in texts)
    assert any("Tracks that passed linearity" in t for t in texts)


def test_model_page_lot_selector_offers_all_history_plus_the_lots(make_app):
    app, page = _stats_app(make_app)
    values = page._lot_menu.cget("values")
    assert values[0] == "All history (no lot)"
    assert len(values) == 13                      # _seed_tracks builds 12 lots
    assert any("unit" in v for v in values[1:])


def test_model_page_defaults_to_the_current_lot_when_one_is_open(make_app):
    """James's default: when a lot is running, that is what he is asking about."""
    app, page = _stats_app(make_app)
    assert page._lot_label is not None
    assert "current lot" in page._lot_menu.get()


def test_choosing_all_history_sticks_across_a_reload(make_app):
    """The per-model default must not fight the user's own pick."""
    app, page = _stats_app(make_app)
    page._on_lot_change("All history (no lot)")
    assert page._lot_label is None
    page.reload_now()
    assert page._lot_label is None
    assert page._lot_menu.get() == "All history (no lot)"


def test_selecting_a_lot_adds_this_lot_lines_and_a_verdict(make_app):
    app, page = _stats_app(make_app)
    texts = [w.cget("text") for w in _all_labels(page._stats_table)]
    assert any(t.strip().startswith("this lot:") for t in texts)
    assert any("normal" in t for t in texts)      # the SPC core's own words


def test_existing_model_page_content_is_still_there(make_app):
    """Nothing currently reachable may be lost (app-shape spec §2)."""
    app, page = _stats_app(make_app)
    for attr in ("_focus_chart", "_pill_row", "_drift_tab", "_smoothness_tab",
                 "_units_tab", "_ft_units_tab", "_trimft_tab", "_history_tab",
                 "_predictor", "_verdict"):
        assert getattr(page, attr) is not None


def _all_labels(widget):
    """Every label in a widget tree (the table renders into nested frames)."""
    out = []
    for child in widget.winfo_children():
        if hasattr(child, "cget"):
            try:
                child.cget("text")
                out.append(child)
            except Exception:
                pass
        out.extend(_all_labels(child))
    return out


def test_summary_line_discloses_records_that_failed_processing():
    """The table's own note carries the 8856 disclosure to the top of the zone."""
    from laser_trim_analyzer.core.model_stats import ModelStats
    from laser_trim_analyzer.gui.v6.widgets.stats_table import summary_line
    stats = ModelStats(model="8856", rows=[_stats_row()], tracks=113, records=113,
                       cutoff=None, lot=None, future_dated=0, errored=75,
                       note="75 record(s) whose processing failed were left out "
                            "— their columns hold error sentinels, not readings")
    assert "processing failed" in summary_line(stats)


# ---- INVESTIGATE: the Excel sheet is the screen, not a second rendering ----

def _pack_with_stats(tmp_path, model="HOT"):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.export.evidence import export_evidence_pack
    db = DatabaseManager(tmp_path / "ev.db")
    _seed_tracks(db, model, "untrimmed_resistance")
    out = export_evidence_pack(db, model, tmp_path / "pack.xlsx")
    import pandas as pd
    return db, pd.read_excel(out, sheet_name=STATS_SHEET, header=2), out


STATS_SHEET = "Stats table"


def test_evidence_pack_has_a_stats_sheet(tmp_path):
    import pandas as pd
    db, sheet, out = _pack_with_stats(tmp_path)
    assert list(pd.read_excel(out, sheet_name=None)) [-1] == STATS_SHEET \
        or STATS_SHEET in pd.read_excel(out, sheet_name=None)
    assert len(sheet) == 8                       # six distributions + two rates
    assert list(sheet["Metric"])[0] == "Untrimmed resistance"


def test_excel_stats_sheet_prints_exactly_what_the_screen_shows(tmp_path):
    """The sheet is what James hands an engineer; it may not round a number
    differently from the page he read it off."""
    from laser_trim_analyzer.core.model_stats import (
        cell_texts, compute_model_stats, disclosure_text)
    db, sheet, _ = _pack_with_stats(tmp_path)
    stats = compute_model_stats(db, "HOT")
    by_metric = {r["Metric"]: r for _, r in sheet.iterrows()}
    for row in stats.rows:
        cells = by_metric[row.label]
        shown = cell_texts(row, row.all_)
        # n stays a NUMBER so Excel can sort and sum it; the screen prints the
        # same number with a thousands separator. The rest is characters.
        assert int(cells["ALL n"]) == row.all_.n
        assert [cells["ALL avg / count"], cells["ALL min / %"]] == shown[1:3]
        lin = cell_texts(row, row.lin_passing)
        assert int(cells["LIN-PASSING n"]) == row.lin_passing.n
        assert cells["LIN-PASSING avg / count"] == lin[1]
        # pandas reads an empty cell back as NaN; the sheet wrote "".
        left_out = cells["Left out"]
        left_out = "" if not isinstance(left_out, str) else left_out
        assert left_out == disclosure_text(row.all_)


def test_excel_stats_sheet_carries_the_lot_verdicts(tmp_path):
    from laser_trim_analyzer.core.model_stats import model_lots
    db, sheet, _ = _pack_with_stats(tmp_path)
    lots = model_lots(db, "HOT")
    assert lots                                   # the fixture builds 12
    verdicts = [v for v in sheet["Lot vs history"] if isinstance(v, str)]
    assert verdicts and any("normal" in v for v in verdicts)
    row = sheet.iloc[0]
    assert str(row["This lot n"]).strip() not in ("", "nan")


def test_excel_stats_sheet_says_which_window_and_lot_it_describes(tmp_path):
    """A table of numbers with no window on it is not evidence."""
    import pandas as pd
    db, _sheet, out = _pack_with_stats(tmp_path)
    head = pd.read_excel(out, sheet_name=STATS_SHEET, header=None, nrows=2)
    assert "track measurements over all history" in str(head.iloc[0, 0])
    assert "unit" in str(head.iloc[1, 0])          # the lot label


def test_a_rate_row_gets_a_this_lot_line_too():
    """"% of the lot that didn't trim" is one of the three questions this
    screen exists to answer, and a count is what answers it."""
    from laser_trim_analyzer.core.model_stats import Cell, StatRow, lot_line
    cell = Cell(n=69, excluded=0, missing=0, count=32, pct=46.376811594)
    row = StatRow(key="trim_passed_linearity", label="Tracks that passed linearity",
                  unit="%", kind="rate", all_=cell, lin_passing=cell)
    assert lot_line(row, cell, None) == "this lot: 32 of 69 (46.4%)"
    assert lot_line(row, Cell(n=0, excluded=0, missing=3), None) \
        == "this lot: nothing recorded"


def test_excel_lot_columns_match_the_screen_for_rate_rows_too(tmp_path):
    from laser_trim_analyzer.core.model_stats import (
        cell_texts, compute_model_stats, model_lots)
    db, sheet, _ = _pack_with_stats(tmp_path)
    lots = model_lots(db, "HOT")
    lot_stats = compute_model_stats(db, "HOT", lot=lots[0].window)
    by_metric = {r["Metric"]: r for _, r in sheet.iterrows()}
    for row in lot_stats.rate_rows:
        shown = cell_texts(row, row.all_)
        cells = by_metric[row.label]
        assert int(cells["This lot n"]) == row.all_.n
        assert cells["This lot avg / count"] == shown[1]
        assert cells["This lot min / %"] == shown[2]


def test_stats_sheet_exists_with_headers_for_a_model_with_no_rows(tmp_path):
    """Same invariant the other six sheets hold: the sheet ALWAYS exists, so
    "no records" cannot be mistaken for a broken export."""
    import pandas as pd
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.export.evidence import export_evidence_pack
    db = DatabaseManager(tmp_path / "empty.db")
    out = export_evidence_pack(db, "NO-SUCH-MODEL", tmp_path / "pack.xlsx")
    sheets = pd.read_excel(out, sheet_name=None)
    assert STATS_SHEET in sheets
    sheet = pd.read_excel(out, sheet_name=STATS_SHEET, header=2)
    for col in ("Metric", "ALL n", "LIN-PASSING n", "Left out", "Lot vs history"):
        assert col in sheet.columns
    # The eight rows still print, each honestly reading zero.
    assert len(sheet) == 8 and set(sheet["ALL n"]) == {0}


def test_window_and_lot_still_hold_when_file_date_carries_a_clock_time(tmp_path):
    """`analysis_results.file_date` stopped being midnight (90dc95e): the
    parser keeps the filename's clock time, because it is the only record of
    the order of same-day re-trim attempts. Every window here is a half-open
    range, never a same-day equality, so a time component must change nothing.
    """
    from datetime import datetime, timedelta
    from laser_trim_analyzer.core.model_stats import compute_model_stats
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType, SystemType)
    db = DatabaseManager(tmp_path / "clock.db")
    day = datetime(2026, 8, 11)
    with db.session() as s:
        for i, hour in enumerate((0, 9, 16, 23)):
            ar = DBAR(model="CLK", serial=f"sn{i}", system=SystemType.A,
                      filename=f"CLK_{i}.xls",
                      file_date=day + timedelta(hours=hour, minutes=37),
                      overall_status=StatusType.PASS)
            s.add(ar); s.flush()
            s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                       untrimmed_resistance=1000.0 + i))
        s.commit()
    # A lot is a run of DAYS, so the last day's 23:37 unit has to fall inside
    # a lot ending that day.
    assert compute_model_stats(db, "CLK", lot=(day, day)).tracks == 4
    # And a cutoff at the start of the day keeps all four, not just the 00:37 one.
    assert compute_model_stats(db, "CLK", cutoff=day).tracks == 4
    # Bounds handed in WITH a clock time still mean the whole day: nothing may
    # depend on the caller's value happening to be midnight.
    noon = day + timedelta(hours=12)
    assert compute_model_stats(db, "CLK", lot=(noon, noon)).tracks == 4
    # And the same code on midnight-only data (every other test in this file,
    # and most of the live database today) is unchanged.
    with db.session() as s:
        ar = DBAR(model="MID", serial="m1", system=SystemType.A,
                  filename="MID_1.xls", file_date=day, overall_status=StatusType.PASS)
        s.add(ar); s.flush()
        s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                   untrimmed_resistance=1000.0))
        s.commit()
    assert compute_model_stats(db, "MID", lot=(day, day)).tracks == 1
    assert compute_model_stats(db, "MID",
                               lot=(day - timedelta(days=1),
                                    day - timedelta(days=1))).tracks == 0
