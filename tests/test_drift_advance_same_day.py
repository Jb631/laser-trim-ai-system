"""Watermark fix (2026-07-06): advance_drift_state must consume samples whose
file_date equals the last processed date.

file_date has day granularity. The old `file_date > last_updated` filter
permanently skipped any sample ingested after a training/advance run but
dated the same day — up to a day of data lost at every boundary. The fix
tracks the source row id (autoincrement) as the watermark; the date filter
survives only as a one-shot fallback for rows trained before the column
existed.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_drift_live_advance import _seed  # same fixtures, same trigger metric


def _state(db, model, metric="untrimmed_resistance"):
    from laser_trim_analyzer.database.models import ModelMetricState
    with db.session() as s:
        row = s.query(ModelMetricState).filter_by(model=model, metric=metric).first()
        return (row.last_row_id, row.last_updated, row.cusum_pos) if row else None


def test_training_sets_row_id_watermark(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.drift_training import train_drift_detector

    db = DatabaseManager(tmp_path / "wm.db")
    _seed(db, "WM", [0.010 + (i % 3) * 0.0002 for i in range(50)])
    train_drift_detector(db, sensitivity_preset="standard")
    last_row_id, _, _ = _state(db, "WM")
    assert last_row_id is not None and last_row_id > 0


def test_same_day_samples_are_consumed(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.drift_training import train_drift_detector, advance_drift_state

    db = DatabaseManager(tmp_path / "sd.db")
    start = datetime(2026, 1, 1)
    _seed(db, "SAME", [0.010 + (i % 3) * 0.0002 for i in range(50)], start=start)
    train_drift_detector(db, sensitivity_preset="standard")
    wm_before, last_dt, _ = _state(db, "SAME")
    last_day = start + timedelta(days=49)
    assert last_dt == last_day  # training consumed through the newest sample

    # A later ingest dated the SAME day as the last trained sample (the exact
    # case the date-only filter dropped forever).
    _seed(db, "SAME", [0.05, 0.05, 0.05], start=last_day)
    n = advance_drift_state(db, model="SAME")
    assert n >= 1, "same-day samples must be consumed, not skipped"
    wm_after, _, _ = _state(db, "SAME")
    assert wm_after > wm_before

    # Idempotent: nothing new -> nothing advanced.
    assert advance_drift_state(db, model="SAME") == 0


def test_legacy_row_without_watermark_falls_back_then_heals(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.drift_training import train_drift_detector, advance_drift_state

    db = DatabaseManager(tmp_path / "legacy.db")
    _seed(db, "LEG", [0.010 + (i % 3) * 0.0002 for i in range(50)])
    train_drift_detector(db, sensitivity_preset="standard")
    # Simulate a row trained before last_row_id existed.
    with db.session() as s:
        s.query(ModelMetricState).filter_by(model="LEG").update(
            {ModelMetricState.last_row_id: None})
        s.commit()

    # Next-day data: date fallback consumes it AND sets the watermark.
    _seed(db, "LEG", [0.011, 0.011], start=datetime(2026, 3, 1))
    assert advance_drift_state(db, model="LEG") >= 1
    wm, _, _ = _state(db, "LEG")
    assert wm is not None and wm > 0
