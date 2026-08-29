"""Spec 3b — Triage. Foundations §4.1/§4.3. Fixtures in tests/conftest.py."""
from datetime import datetime, timedelta

# ---- Task 1: helpers ------------------------------------------------------

def test_metric_label_humanizes():
    from laser_trim_analyzer.ml.drift_types import metric_label
    assert metric_label("untrimmed_resistance") == "Untrimmed resistance"
    assert metric_label("linearity_error") == "Linearity error"
    assert metric_label("measured_electrical_angle") == "Electrical angle"
    assert metric_label("totally_unknown") == "totally_unknown"  # graceful passthrough


def test_list_known_models_empty(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.manager import list_known_models
    assert list_known_models(DatabaseManager(tmp_path / "e.db")) == []


def _add_ar(s, model, when):
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, SystemType, StatusType)
    s.add(DBAR(filename=f"{model}.xls", file_path=f"/f/{model}.xls", file_hash=f"h{model}{when.microsecond}",
               model=model, serial="sn1", system=SystemType.A, file_date=when, timestamp=when,
               overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1))


def test_list_known_models_distinct(tmp_path):
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.manager import list_known_models
    db = DatabaseManager(tmp_path / "d.db")
    with db.session() as s:
        for m in ("8340-1", "8232-1", "8877"):
            _add_ar(s, m, datetime.now())
        s.commit()
    assert {x.model for x in list_known_models(db)} == {"8340-1", "8232-1", "8877"}


def test_list_known_models_includes_smoothness_only(tmp_path):
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import SmoothnessResult as DBSR, StatusType
    from laser_trim_analyzer.ml.manager import list_known_models
    db = DatabaseManager(tmp_path / "s.db")
    with db.session() as s:
        s.add(DBSR(filename="s.xls", file_path="/f/s.xls", file_hash="hs", file_date=datetime.now(),
                   model="SMOOTH-ONLY", serial="sn1", test_date=datetime.now(),
                   overall_status=StatusType.PASS, timestamp=datetime.now()))
        s.commit()
    assert "SMOOTH-ONLY" in {x.model for x in list_known_models(db)}


def test_list_known_models_tier_merged_from_drift_api(tmp_path, monkeypatch):
    """Tier comes from a SINGLE get_drifting_models call; others default STABLE.
    Mock the drift API so the test is deterministic (not coupled to detector math)."""
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.drift_types import (
        AlertType, DriftTier, ModelAlertSummary)
    import laser_trim_analyzer.ml.manager as mgr
    db = DatabaseManager(tmp_path / "t.db")
    with db.session() as s:
        _add_ar(s, "FLAGGED", datetime.now())
        _add_ar(s, "CALM", datetime.now())
        s.commit()
    monkeypatch.setattr(mgr, "get_drifting_models", lambda _db, *a, **k: [
        ModelAlertSummary(model="FLAGGED", tier=DriftTier.DRIFT,
                          alert_type=AlertType.STEP_CHANGE,
                          worst_metric="untrimmed_resistance", magnitude=4.2)])
    by = {x.model: x.tier for x in mgr.list_known_models(db)}
    assert by["FLAGGED"] == DriftTier.DRIFT
    assert by["CALM"] == DriftTier.STABLE


def test_list_known_models_single_query_no_per_model_status(tmp_path, monkeypatch):
    """Regression guard for the N+1 bug: list_known_models must NOT call
    get_model_drift_status once per model."""
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    import laser_trim_analyzer.ml.manager as mgr
    db = DatabaseManager(tmp_path / "n.db")
    with db.session() as s:
        for i in range(5):
            _add_ar(s, f"M{i}", datetime.now())
        s.commit()
    calls = {"n": 0}
    real = mgr.get_model_drift_status
    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(mgr, "get_model_drift_status", counted)
    mgr.list_known_models(db)
    assert calls["n"] == 0  # tiers come from get_drifting_models, not per-model status


# ---- Task 2: triage alert ordering ---------------------------------------
# The v6 Triage page no longer renders these summaries — the FOCUS list took
# that surface on 2026-08-29 — but `get_triage_alerts` still feeds the drift
# table and the v5 pages, so its ordering rules stay under test here.


def _summary(model="8340-1", tier=None, metric="untrimmed_resistance", mag=4.2, alert=None, shift=4.2):
    from laser_trim_analyzer.ml.drift_types import AlertType, DriftTier, ModelAlertSummary
    return ModelAlertSummary(model=model, tier=tier or DriftTier.DRIFT,
                             alert_type=alert or AlertType.STEP_CHANGE,
                             worst_metric=metric, magnitude=mag, sigma_shift=shift)


def test_order_triage_alerts_tier_first_then_shift():
    """Tier stays the primary key (worst on top); |shift| only reorders WITHIN a tier."""
    from laser_trim_analyzer.ml.manager import _order_triage_alerts
    from laser_trim_analyzer.ml.drift_types import DriftTier
    a = _summary(model="DRIFT_BIG", tier=DriftTier.DRIFT, shift=9.0)
    b = _summary(model="OOC_SMALL", tier=DriftTier.OUT_OF_CONTROL, shift=0.2)
    c = _summary(model="DRIFT_SMALL", tier=DriftTier.DRIFT, shift=1.0)
    ordered = [x.model for x in _order_triage_alerts([a, b, c])]
    # OOC outranks both DRIFTs despite a tiny shift; within DRIFT, bigger shift leads.
    assert ordered == ["OOC_SMALL", "DRIFT_BIG", "DRIFT_SMALL"]


def test_order_triage_alerts_gate_hides_below_threshold():
    from laser_trim_analyzer.ml.manager import _order_triage_alerts
    from laser_trim_analyzer.ml.drift_types import DriftTier
    big = _summary(model="BIG", tier=DriftTier.DRIFT, shift=3.0)
    tiny = _summary(model="TINY", tier=DriftTier.DRIFT, shift=0.1)
    assert [x.model for x in _order_triage_alerts([big, tiny], min_sigma_shift=0.0)] == ["BIG", "TINY"]
    assert [x.model for x in _order_triage_alerts([big, tiny], min_sigma_shift=1.0)] == ["BIG"]


# ---- Task 4: BrowseZone ---------------------------------------------------

def _ms(model, tier=None):
    from laser_trim_analyzer.ml.drift_types import DriftTier, ModelSummary
    return ModelSummary(model=model, tier=tier or DriftTier.STABLE)


def test_browse_one_row_per_model(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
    z = BrowseZone(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    z.set_models([_ms(f"M{i}") for i in range(5)])
    assert len(z._rows) == 5


def test_browse_filter_substring(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
    z = BrowseZone(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    # Substring filter: "83" must appear as consecutive chars. 8340-1 and 8830-1
    # contain "83"; 8877 does not. (Plan's original datum "8232-1" was a typo —
    # 8-2-3-2 has no consecutive "83" — inconsistent with the substring impl.)
    z.set_models([_ms("8340-1"), _ms("8830-1"), _ms("8877")])
    z.set_filter("83")
    shown = {r.summary.model for r in z._rows}
    assert shown == {"8340-1", "8830-1"}


def test_browse_row_click_emits_model(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
    got = []
    z = BrowseZone(tk_root, theme=ThemeManager(), on_row_click=got.append)
    z.set_models([_ms("CLICKED")])
    z._rows[0]._on_click()
    assert got == ["CLICKED"]


def test_browse_discloses_cap(tk_root):
    """Q10: when more than the render cap exist, say so instead of silently truncating."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone, ROW_CAP
    z = BrowseZone(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    z.set_models([_ms(f"M{i:04d}") for i in range(ROW_CAP + 25)])
    assert len(z._rows) == ROW_CAP
    assert "Showing" in z._cap_label.cget("text") and str(ROW_CAP + 25) in z._cap_label.cget("text")


# ---- Task 5: TriagePage + routing ----------------------------------------
# Fixtures build the FOCUS list through the REAL DatabaseManager and the real
# compute path (no monkeypatched summaries): the point of the 2026-08-29
# rewire is that the page renders exactly what `compute_focus_list` decided,
# so a mocked feed would test nothing that ships.

D0 = datetime(2026, 1, 5)      # SPC anchors on the DATA's newest date, not "now"


def _labels(widget):
    """Every CTkLabel text under `widget` — what the page actually SAYS."""
    import customtkinter as ctk
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkLabel):
            out.append(c.cget("text"))
        out.extend(_labels(c))
    return out


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


def _drifting_app(make_app):
    """App whose DB holds one drifting CURRENT model and one long-dormant one.

    HOT owns the anchor and blew out on its last lot (10% -> 60%); OLD stopped
    running ~300 days earlier, so it is neither on the FOCUS list (compute's
    own ACTIVE_DAYS rule) nor in the Active browse scope.
    """
    app = make_app()
    _seed(app.db, "HOT", fails_last=12)
    _seed(app.db, "OLD", start=D0 - timedelta(days=300))
    triage = app.page_container.get_page("triage")
    triage.reload_now()       # synchronous path for tests
    return app, triage


def test_v6app_consume_model_route(make_app):
    app = make_app()
    assert app.consume_model_route() is None
    app.set_model_route("M", "linearity_error")
    assert app.consume_model_route() == "M"       # 3b consumes model only
    assert app.consume_model_route() is None       # one-shot


def test_triage_focus_zone_shows_the_drifting_model(make_app):
    app, triage = _drifting_app(make_app)
    # The heading's count is the zone's, not the page's — one computation.
    assert "(1)" in triage._focus._heading.cget("text")
    assert [r.entry.model for r in triage._focus._rows] == ["HOT"]


def test_triage_focus_click_routes_to_model_with_spc_metric(make_app):
    app, triage = _drifting_app(make_app)
    triage._focus._rows[0]._on_click()
    assert app.page_container.current_page == "model"
    # The real Model page consumes the route on show — the deep-link lands on
    # HOT with the metric that put it on the list preselected.
    model_page = app.page_container.get_page("model")
    assert model_page._current_model == "HOT"
    assert model_page._current_metric == "linearity_fail_fraction"
    assert app._model_route is None


def test_scope_toggle_filters_browse_only(make_app):
    """FOCUS membership belongs to compute_focus_list; the toggle is a browse filter."""
    app, triage = _drifting_app(make_app)
    n_focus = len(triage._focus._rows)
    assert {r.summary.model for r in triage._browse._rows} == {"HOT"}   # Active scope
    triage._on_scope_change("All models")
    assert {r.summary.model for r in triage._browse._rows} == {"HOT", "OLD"}
    assert len(triage._focus._rows) == n_focus == 1
    assert [r.entry.model for r in triage._focus._rows] == ["HOT"]
    triage._on_scope_change("Active")
    assert {r.summary.model for r in triage._browse._rows} == {"HOT"}
    assert len(triage._focus._rows) == n_focus


def test_triage_reload_now_populates(make_app):
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, SystemType, StatusType
    app = make_app()
    with app.db.session() as s:
        s.add(DBAR(filename="x.xls", file_path="/f/x.xls", file_hash="hx", model="LOAD-TEST",
                   serial="sn1", system=SystemType.A, file_date=datetime.now(), timestamp=datetime.now(),
                   overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1))
        s.commit()
    triage = app.page_container.get_page("triage")
    triage.reload_now()       # synchronous path for tests
    assert "LOAD-TEST" in _labels(triage)


def test_triage_empty_db_shows_within_tolerance(make_app):
    """No data at all must read as 'nothing to look at', never as a blank zone."""
    app = make_app()
    triage = app.page_container.get_page("triage")
    triage.reload_now()
    assert "(0)" in triage._focus._heading.cget("text")
    assert any("within tolerance" in t for t in _labels(triage._focus))
