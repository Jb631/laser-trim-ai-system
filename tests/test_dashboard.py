"""Dashboard (Production Health) + helpers. Fixtures in tests/conftest.py."""
from datetime import datetime, timedelta

import pytest

from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAR, FinalTestResult as DBFT, SystemType, StatusType)


_SEQ = [0]


def _uid() -> int:
    """Monotonic id so seeded rows never collide on the unique constraints."""
    _SEQ[0] += 1
    return _SEQ[0]


def _add_ar(s, model, status, when):
    u = _uid()
    s.add(DBAR(filename=f"{model}-{status.name}-{u}.xls",
               file_path="/f/x.xls", file_hash=f"har{u}",
               model=model, serial=f"sn{u}", system=SystemType.A, file_date=when, timestamp=when,
               overall_status=status, has_multi_tracks=False, processing_time=0.1))


def _add_ft(s, model, status, when):
    u = _uid()
    s.add(DBFT(filename=f"ft-{model}-{status.name}-{u}.xls", file_path="/f/ft.xls",
               file_hash=f"hft{u}", model=model, serial=f"sn{u}",
               file_date=when, test_date=when, timestamp=when, overall_status=status))


def test_compute_yield_empty(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import compute_yield
    y = compute_yield(DatabaseManager(tmp_path / "e.db"), DBAR, None)
    assert y["total"] == 0 and y["pass_rate"] is None and y["trend"] == []


def test_compute_yield_buckets_and_rate(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import compute_yield
    db = DatabaseManager(tmp_path / "y.db")
    now = datetime.now()
    with db.session() as s:
        for _ in range(3):
            _add_ar(s, "M", StatusType.PASS, now)
        _add_ar(s, "M", StatusType.WARNING, now)
        _add_ar(s, "M", StatusType.FAIL, now)
        _add_ar(s, "M", StatusType.UNTRIMMED, now)   # excluded from rate
        s.commit()
    y = compute_yield(db, DBAR, None)
    assert (y["passed"], y["warnings"], y["failed"], y["untrimmed"]) == (3, 1, 1, 1)
    assert y["gradeable"] == 5
    assert y["pass_rate"] == pytest.approx(60.0)     # 3 / (3+1+1)


def test_compute_yield_windowed(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import compute_yield
    db = DatabaseManager(tmp_path / "w.db")
    now = datetime.now()
    with db.session() as s:
        _add_ar(s, "M", StatusType.PASS, now)
        _add_ar(s, "M", StatusType.PASS, now - timedelta(days=200))   # outside 90d
        s.commit()
    assert compute_yield(db, DBAR, now - timedelta(days=90))["total"] == 1


def test_compute_yield_on_final_test(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import compute_yield
    db = DatabaseManager(tmp_path / "ft.db")
    now = datetime.now()
    with db.session() as s:
        _add_ft(s, "M", StatusType.PASS, now)
        _add_ft(s, "M", StatusType.FAIL, now)
        s.commit()
    y = compute_yield(db, DBFT, None)
    assert y["passed"] == 1 and y["failed"] == 1 and y["pass_rate"] == pytest.approx(50.0)
