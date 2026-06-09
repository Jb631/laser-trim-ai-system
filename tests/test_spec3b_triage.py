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


def _summary(model="8340-1", tier=None, metric="untrimmed_resistance", mag=4.2, alert=None):
    from laser_trim_analyzer.ml.drift_types import AlertType, DriftTier, ModelAlertSummary
    return ModelAlertSummary(model=model, tier=tier or DriftTier.DRIFT,
                             alert_type=alert or AlertType.STEP_CHANGE,
                             worst_metric=metric, magnitude=mag)


def test_card_shows_model_readable_metric_and_magnitude(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
    card = ModelAlertCard(tk_root, summary=_summary(), theme=ThemeManager(), on_click=lambda *_: None)
    texts = " | ".join(_labels(card))
    assert "8340-1" in texts
    assert "Untrimmed resistance" in texts      # readable, not the raw key
    assert "4.2" in texts and "σ" in texts
    assert "Step change" in texts


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
