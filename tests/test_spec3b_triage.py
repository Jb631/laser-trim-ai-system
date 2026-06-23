"""Spec 3b — Triage. Foundations §4.1/§4.3. Fixtures in tests/conftest.py."""

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


# ---- Task 2: ModelAlertCard ----------------------------------------------

def _labels(widget):
    import customtkinter as ctk
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkLabel):
            out.append(c.cget("text"))
        out.extend(_labels(c))
    return out


def _summary(model="8340-1", tier=None, metric="untrimmed_resistance", mag=4.2, alert=None, shift=4.2):
    from laser_trim_analyzer.ml.drift_types import AlertType, DriftTier, ModelAlertSummary
    return ModelAlertSummary(model=model, tier=tier or DriftTier.DRIFT,
                             alert_type=alert or AlertType.STEP_CHANGE,
                             worst_metric=metric, magnitude=mag, sigma_shift=shift)


def test_card_shows_model_readable_metric_and_shift(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
    # Headline is now the honest baseline shift (σ), not the CUSUM magnitude.
    card = ModelAlertCard(tk_root, summary=_summary(shift=4.2), theme=ThemeManager(),
                          on_click=lambda *_: None)
    texts = " | ".join(_labels(card))
    assert "8340-1" in texts
    assert "Untrimmed resistance" in texts      # readable, not the raw key
    assert "4.2" in texts and "σ" in texts      # the shift value
    assert "Step change" in texts
    assert "baseline" in texts                  # says what the σ is measured against


def test_card_shows_dash_when_shift_unknown(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
    card = ModelAlertCard(tk_root, summary=_summary(shift=None), theme=ThemeManager(),
                          on_click=lambda *_: None)
    texts = " | ".join(_labels(card))
    assert "—" in texts                         # no fabricated number when recent is unknown


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


def test_card_click_emits_model_and_focus_metric(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
    got = []
    card = ModelAlertCard(tk_root, summary=_summary(model="CLICK", metric="linearity_error"),
                          theme=ThemeManager(), on_click=lambda m, f: got.append((m, f)))
    card._on_click()
    assert got == [("CLICK", "linearity_error")]   # focus = the triggering metric


def test_card_uses_tier_background(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
    from laser_trim_analyzer.ml.drift_types import DriftTier
    t = ThemeManager()
    card = ModelAlertCard(tk_root, summary=_summary(tier=DriftTier.OUT_OF_CONTROL),
                          theme=t, on_click=lambda *_: None)
    assert card.cget("fg_color") == t.tier_color(DriftTier.OUT_OF_CONTROL)[0]


# ---- Task 3: FlaggedCardsZone --------------------------------------------

def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def test_zone_empty_state_names_last_processed(tk_root):
    from datetime import datetime
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.flagged_cards_zone import FlaggedCardsZone
    z = FlaggedCardsZone(tk_root, theme=ThemeManager(), on_card_click=lambda *_: None)
    z.set_summaries([], last_processed=datetime(2026, 5, 30))
    txt = " ".join(_labels(z))
    assert "within tolerance" in txt
    assert "2026-05-30" in txt


def test_zone_one_card_per_summary(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.flagged_cards_zone import FlaggedCardsZone
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
    z = FlaggedCardsZone(tk_root, theme=ThemeManager(), on_card_click=lambda *_: None)
    z.set_summaries([_summary(model=f"M{i}") for i in range(6)])
    assert sum(isinstance(w, ModelAlertCard) for w in _walk(z)) == 6


def test_zone_routes_click(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.flagged_cards_zone import FlaggedCardsZone
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
    got = []
    z = FlaggedCardsZone(tk_root, theme=ThemeManager(), on_card_click=lambda m, f: got.append((m, f)))
    z.set_summaries([_summary(model="ROUTED", metric="sigma_gradient")])
    next(w for w in _walk(z) if isinstance(w, ModelAlertCard))._on_click()
    assert got == [("ROUTED", "sigma_gradient")]


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

def test_v6app_consume_model_route(make_app):
    app = make_app()
    assert app.consume_model_route() is None
    app.set_model_route("M", "linearity_error")
    assert app.consume_model_route() == "M"       # 3b consumes model only
    assert app.consume_model_route() is None       # one-shot


def test_triage_card_click_routes_to_model(make_app):
    app = make_app()
    triage = app.page_container.get_page("triage")
    triage._on_card_click("FROM-CARD", "untrimmed_resistance")
    assert app.page_container.current_page == "model"
    # The real Model page (3c) consumes the route on show and applies it — the
    # deep-link lands on FROM-CARD with the triggering metric preselected.
    model_page = app.page_container.get_page("model")
    assert model_page._current_model == "FROM-CARD"
    assert model_page._current_metric == "untrimmed_resistance"
    assert app._model_route is None


def test_triage_reload_now_populates(make_app):
    from datetime import datetime
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
