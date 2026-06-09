"""Spec 3d — Settings. Foundations §4.2/D2/D3. Fixtures in tests/conftest.py."""
import pytest


def _seed_metric_state(db, model="T", baseline_std=0.001, trained=True):
    from datetime import datetime
    from laser_trim_analyzer.database.models import ModelMetricState
    with db.session() as s:
        s.add(ModelMetricState(
            model=model, metric="sigma_gradient", baseline_mean=0.01, baseline_std=baseline_std,
            baseline_count=100, is_trained=trained,
            h_warning=1.0, h_drift=5.0, h_oc=10.0, L_warning=1.0, L_drift=2.0, L_oc=3.0,
            z_warning=1.0, z_drift=2.0, z_oc=3.0, cusum_pos=2.5, cusum_neg=0.0, ewma_state=0.01,
            last_updated=datetime.now()))
        s.commit()


def test_apply_sensitivity_preset_recomputes_to_known_values(tmp_path):
    """M8 fix: assert the EXACT recomputed thresholds, not a vague '> 1.0'."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.manager import apply_sensitivity_preset
    from laser_trim_analyzer.ml.multi_metric_drift_detector import compute_thresholds
    from laser_trim_analyzer.ml.drift_types import target_fp_for_tier, DriftTier

    db = DatabaseManager(tmp_path / "p.db")
    _seed_metric_state(db, baseline_std=0.001)
    n = apply_sensitivity_preset(db, "tight")
    assert n == 1
    exp_h, exp_L, exp_z = compute_thresholds(0.001, target_fp_for_tier("tight", DriftTier.WARNING))
    with db.session() as s:
        row = s.query(ModelMetricState).filter_by(model="T", metric="sigma_gradient").first()
        assert row.baseline_mean == pytest.approx(0.01)     # baseline untouched
        assert row.baseline_std == pytest.approx(0.001)
        assert row.cusum_pos == pytest.approx(2.5)          # runtime state preserved
        assert row.L_warning == pytest.approx(exp_L)
        assert row.z_warning == pytest.approx(exp_z)
        assert row.h_warning == pytest.approx(exp_h)


def test_apply_sensitivity_preset_skips_untrained(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.manager import apply_sensitivity_preset
    db = DatabaseManager(tmp_path / "u.db")
    _seed_metric_state(db, model="U", trained=False)
    assert apply_sensitivity_preset(db, "tight") == 0   # untrained skipped, no raise


# ---- Task 2: SettingsCard -------------------------------------------------

def test_settings_card_states(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.settings_card import SettingsCard
    t = ThemeManager()
    assert SettingsCard(tk_root, theme=t, title="A")._expanded is False
    c = SettingsCard(tk_root, theme=t, title="B", expanded=True)
    assert c._expanded is True
    c.toggle(); assert c._expanded is False


def test_settings_card_body_frame_is_fillable(tk_root):
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.settings_card import SettingsCard
    c = SettingsCard(tk_root, theme=ThemeManager(), title="C", expanded=True)
    ctk.CTkLabel(c.body_frame(), text="hi").pack()   # no raise


# ---- Task 3: SensitivitySlider --------------------------------------------

def test_sensitivity_slider(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.sensitivity_slider import SensitivitySlider
    got = []
    s = SensitivitySlider(tk_root, theme=ThemeManager(), initial="standard", on_change=got.append)
    assert s.value() == "standard"
    s.set_value("tight"); assert s.value() == "tight"


# ---- Task 4: TrainingModal ------------------------------------------------

def test_training_modal_uses_injected_train_fn(tk_root, tmp_path):
    """Inject a fake train_fn so the widget test never runs real training threads."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.training_modal import TrainingModal
    db = DatabaseManager(tmp_path / "tm.db")
    called = {}
    def fake_train(d, preset, progress_callback=None):
        called["preset"] = preset
        if progress_callback:
            progress_callback("M1", 0, 1)
        class _S: pass
        return _S()
    modal = TrainingModal(tk_root, theme=ThemeManager(), db=db, preset="standard", train_fn=fake_train)
    modal._run_training()           # synchronous path (no thread) for the test
    assert called["preset"] == "standard"
    modal.destroy()


# ---- Task 5: sections ------------------------------------------------------

class _FakeApp:
    def __init__(self, db, cfg):
        self.db = db
        self.config = cfg


def _fake_app(tmp_path, name="x.db"):
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.database.manager import DatabaseManager
    cfg = Config(); cfg.database.path = tmp_path / name
    return _FakeApp(DatabaseManager(cfg.database.path), cfg)


def test_alert_thresholds_section_builds(tk_root, tmp_path):
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.sections.alert_thresholds import build_alert_thresholds_section
    parent = ctk.CTkFrame(tk_root)
    build_alert_thresholds_section(parent, theme=ThemeManager(), app=_fake_app(tmp_path))


def test_ml_training_section_builds(tk_root, tmp_path):
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.sections.ml_training import build_ml_training_section
    build_ml_training_section(ctk.CTkFrame(tk_root), theme=ThemeManager(), app=_fake_app(tmp_path, "ml.db"))


def test_per_model_specs_section_builds(tk_root, tmp_path):
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.sections.per_model_specs import build_per_model_specs_section
    build_per_model_specs_section(ctk.CTkFrame(tk_root), theme=ThemeManager(), app=_fake_app(tmp_path, "sp.db"))


def test_per_model_specs_save_roundtrip(tmp_path):
    """exclude_points feeds ML correctness (flip FAIL<->PASS), so the human->JSON
    storage conversion must round-trip through the DB exactly."""
    from laser_trim_analyzer.core.analyzer import parse_exclude_points
    from laser_trim_analyzer.gui.v6.sections.per_model_specs import build_spec_save_data
    app = _fake_app(tmp_path, "spx.db")
    data = build_spec_save_data("8340-1", "±0.05%", "0.05", "0-2, 48-50", "")
    app.db.save_model_spec(data)
    got = app.db.get_model_spec("8340-1")
    assert got["linearity_spec_pct"] == 0.05
    assert got["linearity_spec_text"] == "±0.05%"
    assert parse_exclude_points(got["exclude_points"]) == {0, 1, 2, 48, 49, 50}
    assert got["exclude_points_ft"] is None
