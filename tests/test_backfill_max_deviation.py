"""backfill_max_deviation (database/maintenance.py): recomputes max_deviation /
max_deviation_position / deviation_uniformity, straight SQL, for track_results rows that
carry error_data but have never had max_deviation filled in.

Nothing in the app calls this method today -- there is no Settings button for it, unlike its
neighbours (recompute_overall_statuses, the cleanup trio). That absence of a caller is exactly
how a real bug hid after the C2 Task 5 move to database/maintenance.py: that module never
imported `text` from sqlalchemy, so every one of this method's three `session.execute(text(...))`
calls raised NameError -- caught by its own broad `except Exception`, logged as one ERROR line,
and it returned 0, indistinguishable from "nothing needed backfilling" (see task-5-report.md).
The AST of the moved method was byte-identical to the pre-move version; the missing import was
invisible to that proof because it lives in the module's globals, not in any function's own body.

This test calls the method directly, against real rows, and checks the COMPUTED values -- not
just a non-crashing return -- so a regression here fails loudly instead of quietly returning 0
again.
"""
import statistics
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _add_track(db, *, serial, error_data, position_data, optimal_offset, max_deviation=None):
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    with db.session() as s:
        ar = DBAR(filename=f"M1-{serial}.xls", file_path=f"/f/M1/{serial}",
                  file_hash=serial.ljust(64, "a"),
                  model="M1", serial=serial, system=SystemType.A,
                  file_date=datetime(2026, 5, 1), timestamp=datetime(2026, 5, 1),
                  overall_status=StatusType.PASS, has_multi_tracks=False,
                  processing_time=0.1)
        s.add(ar)
        s.flush()
        tr = DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                  error_data=error_data, position_data=position_data,
                  optimal_offset=optimal_offset, max_deviation=max_deviation)
        s.add(tr)
        s.flush()
        return tr.id


def _track_row(db, track_id):
    from laser_trim_analyzer.database.models import TrackResult as DBTR
    with db.session() as s:
        row = s.query(DBTR).filter(DBTR.id == track_id).first()
        return row.max_deviation, row.max_deviation_position, row.deviation_uniformity


def test_backfill_max_deviation_computes_and_writes_real_values(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(tmp_path / "bmd.db")

    # errors [0.01, -0.03, 0.02] shifted by optimal_offset 0.01 -> [0.02, -0.02, 0.03];
    # abs -> [0.02, 0.02, 0.03]; max 0.03 at index 2 -> position_data[2].
    needs_backfill = _add_track(
        db, serial="s1", error_data=[0.01, -0.03, 0.02], position_data=[10, 20, 30],
        optimal_offset=0.01, max_deviation=None)
    # Already has max_deviation -- must be left alone. Proves the WHERE clause
    # actually filters, not just that the method runs without raising.
    already_done = _add_track(
        db, serial="s2", error_data=[0.5], position_data=[1], optimal_offset=0.0,
        max_deviation=0.5)

    updated = db.backfill_max_deviation()
    assert updated == 1, (
        "exactly the one row missing max_deviation should be touched -- 0 here is "
        "the silent-NameError symptom this test exists to catch")

    max_dev, max_dev_pos, dev_unif = _track_row(db, needs_backfill)
    abs_errs = [0.02, 0.02, 0.03]
    assert max_dev == pytest.approx(0.03)
    assert max_dev_pos == pytest.approx(30)
    assert dev_unif == pytest.approx(statistics.stdev(abs_errs) / statistics.mean(abs_errs))

    # The row that already had a value is untouched.
    max_dev2, _, _ = _track_row(db, already_done)
    assert max_dev2 == pytest.approx(0.5)

    # Idempotent: nothing left to backfill on a second pass.
    assert db.backfill_max_deviation() == 0

    db.close()
