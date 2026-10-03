"""V6 Batch I (H8): the drift detector must reflect drift in history after
training (replay) and respond to NEW data via advance_drift_state.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _seed(db, model, values, start=datetime(2025, 1, 6), units=1):
    """One weekly LOT per value, of `units` identical units. Lots of fewer than 3 units are
    judged in pools since 2026-10-02 (lots.judged_lots) -- a test that needs each value judged
    on its own passes units=3."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as AR, TrackResult as TR, SystemType, StatusType)
    with db.session() as s:
        for i, val in enumerate(values):
            for u in range(units):
                key = f"{model}-{i}" if units == 1 else f"{model}-{start:%Y%m%d}-{i}-{u}"
                ar = AR(filename=f"{key}.xls", file_path=f"/f/{model}/{key}",
                        file_hash=f"{key}-h", model=model, serial=f"sn{i}-{u}",
                        # Weekly spacing: each value is its own LOT under lot-mode
                        # clustering (gap > 3 days), preserving per-sample semantics.
                        system=SystemType.A,
                        file_date=start + timedelta(days=7 * i, minutes=u),
                        timestamp=start + timedelta(days=7 * i), overall_status=StatusType.PASS,
                        has_multi_tracks=False, processing_time=0.1)
                s.add(ar); s.flush()
                # Seed a TRIGGER metric (untrimmed_resistance) so the drifted model flags;
                # untrimmed_sigma_gradient is now evidence-only (drift didn't predict fails).
                s.add(TR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                         untrimmed_resistance=val))
        s.commit()


def test_training_replay_flags_drift_in_history(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.drift_training import train_drift_detector
    from laser_trim_analyzer.ml.manager import get_drifting_models

    db = DatabaseManager(tmp_path / "hist.db")
    # 42 stable samples, then a sharp upward ramp (the recent window).
    vals = [0.010 + (i % 3) * 0.0002 for i in range(42)]
    vals += [0.010 + (k + 1) * 0.003 for k in range(18)]
    _seed(db, "DRIFTY", vals, units=3)

    summary = train_drift_detector(db, sensitivity_preset="standard")
    assert summary.models_trained >= 1
    # The replayed recent window pushed CUSUM past threshold -> flagged at train.
    assert "DRIFTY" in [m.model for m in get_drifting_models(db)]


def test_advance_flags_new_drifted_data(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.drift_training import train_drift_detector, advance_drift_state
    from laser_trim_analyzer.ml.manager import get_drifting_models

    db = DatabaseManager(tmp_path / "adv.db")
    # All-stable history -> not flagged after training.
    _seed(db, "CALM", [0.010 + (i % 3) * 0.0002 for i in range(50)])
    train_drift_detector(db, sensitivity_preset="standard")
    assert "CALM" not in [m.model for m in get_drifting_models(db)]

    # New, clearly-drifted data arriving later -> advance should flag it.
    _seed(db, "CALM", [0.010 + (k + 1) * 0.004 for k in range(15)],
          start=datetime(2026, 6, 1))
    n = advance_drift_state(db, model="CALM")
    assert n >= 1
    assert "CALM" in [m.model for m in get_drifting_models(db)]
