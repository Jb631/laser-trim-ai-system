"""SPC DB layer: compute_spc_series / compute_focus_list against a tmp DB."""
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import AnalysisResult, StatusType, SystemType
from laser_trim_analyzer.ml.spc import (
    RECENT_K, compute_focus_list, compute_spc_series)

D0 = datetime(2026, 1, 5)


@pytest.fixture()
def db(tmp_path):
    return DatabaseManager(tmp_path / "t.db")


def _add_lot(db, model, day, n, fails):
    with db.session() as s:
        for i in range(n):
            # `system` is NOT NULL in the schema; the SPC query never reads it.
            s.add(AnalysisResult(
                model=model, serial=f"{model}-{day:%m%d}-{i}", system=SystemType.A,
                filename=f"{model}_{i}_{day:%m-%d-%Y}.xls", file_date=day,
                overall_status=StatusType.FAIL if i < fails else StatusType.PASS))


def _seed(db, model, n_lots=12, fails_last=0, start=D0, n_per=20, base_fails=2):
    for k in range(n_lots - 1):
        _add_lot(db, model, start + timedelta(days=7 * k), n_per, base_fails)
    last_day = start + timedelta(days=7 * (n_lots - 1))
    _add_lot(db, model, last_day, n_per, fails_last if fails_last else base_fails)
    return last_day


def test_focus_membership_and_ranking(db):
    d_hot = _seed(db, "HOT", fails_last=12)     # drifting, 20 u/lot
    # CALM runs a day BEHIND HOT so the newest row in the DB is HOT's last lot
    # (the anchor is the whole database's newest usable date, not one model's).
    _seed(db, "CALM", fails_last=2, start=D0 - timedelta(days=1))
    res = compute_focus_list(db)
    assert res.anchor == d_hot
    assert [e.model for e in res.focus] == ["HOT"]
    e = res.focus[0]
    assert e.n_flagged_recent == 1 and e.last_lot_end == d_hot
    assert e.verdict.startswith("failing ~") and "units/week" in e.verdict
    assert f"1 of last {RECENT_K} lots out of control" in e.sub_line


def test_verdict_numbers_match_series(db):
    """The one-computation guarantee: row numbers derive from its own series."""
    _seed(db, "HOT", fails_last=12)
    e = compute_focus_list(db).focus[0]
    flagged = [p for p in e.series.points[-RECENT_K:] if p.ooc]
    p_recent = sum(p.value * p.n for p in flagged) / sum(p.n for p in flagged)
    assert e.p_recent == pytest.approx(p_recent)
    assert e.excess_per_week == pytest.approx(
        max(e.p_recent - e.p_base, 0.0) * e.units_per_week)


def test_inactive_model_leaves_the_list(db):
    _seed(db, "OLD", fails_last=12)                          # ends ~11 weeks in
    _seed(db, "NOW", start=D0 + timedelta(days=200))         # anchor mover
    res = compute_focus_list(db)
    assert all(e.model != "OLD" for e in res.focus)          # > ACTIVE_DAYS stale


def test_chronic_strip(db):
    _seed(db, "SICK", n_lots=12, base_fails=8, fails_last=8)  # stable 40%
    res = compute_focus_list(db)
    assert [e.model for e in res.chronic] == ["SICK"]
    assert not res.focus
    assert "capability problem, not drift" in res.chronic[0].verdict


def test_requalification_floor_respected(db):
    d = _seed(db, "REQ", n_lots=14, fails_last=2)
    db.set_baseline_requalification("REQ", (D0 + timedelta(days=7 * 4)).isoformat())
    s = compute_spc_series(db, "REQ")
    assert len(s.points) == 10


def test_empty_db(db):
    res = compute_focus_list(db)
    assert res.anchor is None and res.focus == [] and res.chronic == []
