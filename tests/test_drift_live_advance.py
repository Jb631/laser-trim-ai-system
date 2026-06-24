"""V6 Batch I (H8): the drift detector must reflect drift in history after
training (replay) and respond to NEW data via advance_drift_state.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _seed(db, model, values, start=datetime(2026, 1, 1)):
    from laser_trim_analyzer.database.models import (
        AnalysisResult as AR, TrackResult as TR, SystemType, StatusType)
    with db.session() as s:
        for i, val in enumerate(values):
            ar = AR(filename=f"{model}-{i}.xls", file_path=f"/f/{model}/{i}",
                    file_hash=f"{model}-h{i}", model=model, serial=f"sn{i}",
                    system=SystemType.A, file_date=start + timedelta(days=i),
                    timestamp=start + timedelta(days=i), overall_status=StatusType.PASS,
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
    _seed(db, "DRIFTY", vals)

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
