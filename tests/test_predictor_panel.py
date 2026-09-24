"""PredictorPanel -- demoted per-unit predictor (Spec 3c; design doc
2026-09-24-facelift-step2-pages-design.md sect1 item 7).

Step 1(d) of facelift step 2 Task 3: the panel's first line must say what it IS -- how
well the predictor would have called these units -- not describe itself as a "Failure
predictor" (which reads as the app calling a unit failed) and must not use "pass"/"fail"
to describe a unit's own outcome.
"""
import pytest


class _FakeMetrics:
    def __init__(self, accuracy=None, f1=None, auc_roc=None):
        self.accuracy = accuracy
        self.f1 = f1
        self.auc_roc = auc_roc


class _FakePredictor:
    def __init__(self, *, is_trained=True, metrics=None, training_samples=None,
                 feature_importance=None):
        self.is_trained = is_trained
        self.metrics = metrics
        self.training_samples = training_samples
        self.feature_importance = feature_importance or {}


def _load_text(monkeypatch, model, predictor):
    """Drive _default_load's inner _load(model) directly against a fake ml manager --
    no real DB, no real training, same late-import seam _default_load itself uses
    (`from laser_trim_analyzer.ml import get_shared_ml_manager`)."""
    import laser_trim_analyzer.ml as ml_pkg
    from laser_trim_analyzer.gui.v6.widgets.predictor_panel import _default_load

    class _Mgr:
        predictors = {model: predictor} if predictor is not None else {}

    monkeypatch.setattr(ml_pkg, "get_shared_ml_manager", lambda db: _Mgr())
    return _default_load(db=object())(model)


def test_first_line_says_what_the_panel_is_not_a_verdict(monkeypatch):
    pred = _FakePredictor(metrics=_FakeMetrics(accuracy=0.82, f1=0.75, auc_roc=0.87),
                          training_samples=346)
    text = _load_text(monkeypatch, "HOT", pred)
    first_line = text.splitlines()[0]
    assert first_line.startswith(
        "How well the final-test predictor would have called these units")
    assert "pass" not in first_line.lower() and "fail" not in first_line.lower()
    assert "AUC 0.87" in first_line
    assert "It grades nothing" in first_line


def test_first_line_omits_the_parenthetical_auc_when_there_is_none(monkeypatch):
    pred = _FakePredictor(metrics=_FakeMetrics(accuracy=0.82, f1=0.75, auc_roc=None),
                          training_samples=40)
    text = _load_text(monkeypatch, "HOT", pred)
    first_line = text.splitlines()[0]
    assert first_line == ("How well the final-test predictor would have called these "
                          "units. It grades nothing.")


def test_performance_and_signals_still_follow_the_first_line(monkeypatch):
    pred = _FakePredictor(
        metrics=_FakeMetrics(accuracy=0.82, f1=0.75, auc_roc=0.87), training_samples=346,
        feature_importance={"sigma_gradient": 0.4, "resistance_change_percent": 0.3,
                            "trim_pass_count": 0.2, "unit_length": 0.1})
    text = _load_text(monkeypatch, "HOT", pred)
    assert "accuracy 82%" in text and "F1 0.75" in text and "346 training units" in text
    assert "Strongest signals:" in text
    assert "Diagnostic only" in text
    # AUC is said ONCE, in the first line, not repeated in the "Performance:" line.
    assert text.count("AUC") == 1


def test_no_trained_predictor_raises_lookuperror(monkeypatch):
    from laser_trim_analyzer.gui.v6.widgets.predictor_panel import _default_load
    import laser_trim_analyzer.ml as ml_pkg

    class _Mgr:
        predictors = {}
    monkeypatch.setattr(ml_pkg, "get_shared_ml_manager", lambda db: _Mgr())
    with pytest.raises(LookupError):
        _default_load(db=object())("NOPE")


def test_panel_shows_the_loaded_text_when_expanded(tk_root):
    """Plumbing check with the widget's own load_fn seam (no DB, no ml manager): expanding
    the panel loads and displays whatever load_fn returns -- here, confirming the new
    first-line text actually reaches the body label, not just the helper function."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.predictor_panel import PredictorPanel

    calls = []
    def load_fn(model):
        calls.append(model)
        return "How well the final-test predictor would have called these units (AUC 0.87). It grades nothing."

    panel = PredictorPanel(tk_root, theme=ThemeManager(), load_fn=load_fn)
    panel.set_model("HOT")
    assert calls == []                       # collapsed: no load yet
    panel.toggle()                           # expands -- no dispatcher on a bare tk_root, so this runs inline
    assert calls == ["HOT"]
    assert panel._body_label.cget("text").startswith(
        "How well the final-test predictor would have called these units")


def test_a_failed_load_still_names_where_to_train_it(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.predictor_panel import PredictorPanel

    def boom(model):
        raise LookupError("no trained predictor for this model")

    panel = PredictorPanel(tk_root, theme=ThemeManager(), load_fn=boom)
    panel.set_model("HOT")
    panel.toggle()
    assert "No predictor for HOT" in panel._body_label.cget("text")
    assert "Settings" in panel._body_label.cget("text")
