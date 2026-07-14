"""Lot-mode advance semantics (2026-07-13, replaces the unit-mode watermark
tests): the detector ingests CLOSED lots newer than the date watermark.

* Training writes last_updated = end of the newest closed lot and clears
  last_row_id (None marks a lot-mode row).
* An OPEN lot (last unit within LOT_GAP_DAYS of now) is never ingested.
* Once the lot closes, advance ingests it exactly ONCE.
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


def test_training_sets_lot_watermark(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.drift_training import train_drift_detector

    db = DatabaseManager(tmp_path / "wm.db")
    _seed(db, "WM", [0.010 + (i % 3) * 0.0002 for i in range(50)])
    train_drift_detector(db, sensitivity_preset="standard")
    last_row_id, last_updated, _ = _state(db, "WM")
    assert last_row_id is None            # lot-mode marker
    assert last_updated is not None       # = newest closed lot end


def test_open_lot_not_ingested_then_ingested_once_closed(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.drift_training import (
        train_drift_detector, advance_drift_state)

    db = DatabaseManager(tmp_path / "open.db")
    _seed(db, "OP", [0.010 + (i % 3) * 0.0002 for i in range(50)])
    train_drift_detector(db, sensitivity_preset="standard")
    _, wm0, cpos0 = _state(db, "OP")

    # A lot dated RIGHT NOW is still open -> advance must not touch it.
    _seed(db, "OP", [0.05] * 5, start=datetime.now())
    assert advance_drift_state(db, model="OP") == 0
    _, wm1, cpos1 = _state(db, "OP")
    assert wm1 == wm0 and cpos1 == cpos0

    # A lot that ended past the changeover gap is closed -> ingested once.
    _seed(db, "OP", [0.05] * 5, start=datetime.now() - timedelta(days=30))
    assert advance_drift_state(db, model="OP") >= 1
    _, wm2, cpos2 = _state(db, "OP")
    assert wm2 > wm0
    # Re-advance: nothing new -> no change.
    assert advance_drift_state(db, model="OP") == 0
    _, wm3, cpos3 = _state(db, "OP")
    assert wm3 == wm2 and cpos3 == cpos2
