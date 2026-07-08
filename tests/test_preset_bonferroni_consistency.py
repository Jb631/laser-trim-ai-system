"""Preset-path consistency (2026-07-06): preview_alert_count and
apply_sensitivity_preset must produce the SAME thresholds as training.

Training divides each tier's target FP by len(WATCHED_METRICS) (Bonferroni,
family-wise). The preview/apply paths skipped that division, so 'Save preset'
wrote thresholds ~9x looser (in FP target) than a retrain at the same preset,
and the preview counts didn't correspond to trained reality.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_drift_live_advance import _seed


def _thresholds(db, model, metric="untrimmed_resistance"):
    from laser_trim_analyzer.database.models import ModelMetricState
    with db.session() as s:
        r = s.query(ModelMetricState).filter_by(model=model, metric=metric).first()
        return (r.h_warning, r.L_warning, r.z_warning,
                r.h_drift, r.L_drift, r.z_drift,
                r.h_oc, r.L_oc, r.z_oc, r.last_updated)


def test_apply_preset_matches_training(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.drift_training import train_drift_detector
    from laser_trim_analyzer.ml.manager import apply_sensitivity_preset

    db = DatabaseManager(tmp_path / "p.db")
    _seed(db, "PRESET", [0.010 + (i % 5) * 0.0003 for i in range(60)])

    train_drift_detector(db, sensitivity_preset="standard")
    trained = _thresholds(db, "PRESET")

    # Applying the SAME preset must be a no-op on thresholds AND must not
    # touch last_updated (it's the advance_drift_state sample marker).
    n = apply_sensitivity_preset(db, "standard")
    assert n >= 1
    assert _thresholds(db, "PRESET") == trained

    # A different preset changes thresholds; re-applying standard restores them.
    apply_sensitivity_preset(db, "loose")
    assert _thresholds(db, "PRESET")[:9] != trained[:9]
    apply_sensitivity_preset(db, "standard")
    assert _thresholds(db, "PRESET") == trained


def test_corrected_helper_is_the_training_math(tmp_path):
    from laser_trim_analyzer.ml.drift_training import corrected_tier_thresholds
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, WATCHED_METRICS, target_fp_for_tier)
    from laser_trim_analyzer.ml.multi_metric_drift_detector import compute_thresholds

    sigma = 0.02
    out = corrected_tier_thresholds("standard", sigma)
    n = len(WATCHED_METRICS)
    for tier in (DriftTier.WARNING, DriftTier.DRIFT, DriftTier.OUT_OF_CONTROL):
        expected = compute_thresholds(sigma, target_fp_for_tier("standard", tier) / n)
        assert out[tier] == expected

    # And the corrected thresholds are strictly tighter than uncorrected ones
    # (higher h/L/z for a smaller FP target).
    for tier in (DriftTier.WARNING, DriftTier.DRIFT, DriftTier.OUT_OF_CONTROL):
        uncorrected = compute_thresholds(sigma, target_fp_for_tier("standard", tier))
        assert out[tier][0] > uncorrected[0]  # h
        assert out[tier][1] > uncorrected[1]  # L
        assert out[tier][2] > uncorrected[2]  # z
