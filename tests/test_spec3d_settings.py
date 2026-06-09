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
