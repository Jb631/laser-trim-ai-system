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
