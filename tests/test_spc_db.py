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


def test_future_dated_junk_does_not_revive_dormant_model(db):
    """A mis-set machine clock must not make a dormant model look alive."""
    _seed(db, "DORMANT", fails_last=12)                      # real run ended long ago
    _seed(db, "LIVE", start=D0 + timedelta(days=200))        # owns the anchor
    _add_lot(db, "DORMANT", D0 + timedelta(days=3650), 1, 1)  # one junk-dated FAIL
    res = compute_focus_list(db)
    assert all(e.model != "DORMANT" for e in res.focus)
    assert all(e.model != "DORMANT" for e in res.chronic)


def test_evidence_pack_carries_the_lot_series(db, tmp_path):
    """The export must quote the SAME lots the chart draws, not its own math.

    An evidence pack that recomputed the lot rates would put a third story in
    front of the engineer — the exact failure the SPC redesign exists to end.
    """
    import pandas as pd
    from laser_trim_analyzer.export.evidence import export_evidence_pack

    _seed(db, "HOT", fails_last=12)                 # drifting: last lot alarms
    series = compute_spc_series(db, "HOT")
    out = export_evidence_pack(db, "HOT", tmp_path / "e.xlsx")

    sheet = pd.read_excel(out, sheet_name="Lots (SPC)")
    assert any(p.ooc for p in series.points)        # the fixture really drifted
    assert sheet["Out of control"].tolist() == [p.ooc for p in series.points]
    assert sheet["Fail rate"].tolist() == pytest.approx([p.value for p in series.points])
    assert sheet["Units"].tolist() == [p.n for p in series.points]


def test_open_lot_uses_the_database_clock_not_the_models_own(db):
    """ONE clock: a model idle for weeks is CLOSED on its own chart too.

    The bug this pins: `compute_spc_series` used to anchor on the MODEL'S own
    newest sample, which is a 0-day gap by construction — so every model's last
    lot read as OPEN forever, the Model page drew a hollow "still filling"
    marker, and the evidence pack printed `Open lot: TRUE` on the final row of
    every export. The focus list, anchored on the whole database, said CLOSED
    for the same lot. Openness must be one answer, not two.
    """
    a_last = _seed(db, "A", fails_last=12)                    # stops producing here
    b_last = _seed(db, "B", start=D0 + timedelta(days=21))    # keeps running 21d longer

    res = compute_focus_list(db)
    assert res.anchor == b_last                               # the DB-global clock
    series = compute_spc_series(db, "A")                      # no anchor passed
    assert series.points[-1].end == a_last
    assert series.points[-1].is_open is False                 # 21 days idle == closed

    entry = next(e for e in res.focus if e.model == "A")       # A really is a fire
    assert entry.series.points[-1].is_open == series.points[-1].is_open


def test_explicit_anchor_still_wins(db):
    """A caller that knows better (a replay, a report as-of a date) is obeyed."""
    a_last = _seed(db, "A", fails_last=12)
    _seed(db, "B", start=D0 + timedelta(days=21))
    series = compute_spc_series(db, "A", anchor=a_last)
    assert series.points[-1].is_open is True


def test_a_metrics_own_newer_data_is_never_truncated(db):
    """The clock must not cut off a metric that lives in a different table.

    The anchor does two jobs: it dates the openness question AND sets `_clean`'s
    forward cut. Final test happens days AFTER the trim and smoothness is its
    own file, so those lots can legitimately postdate the newest trim file. A
    clock taken only from `analysis_results` would drop the newest lots of those
    metrics off the chart without saying a word.
    """
    from laser_trim_analyzer.database.models import SmoothnessResult

    _seed(db, "T", fails_last=2)                     # sets the analysis_results clock
    last_sm = D0 + timedelta(days=120)               # smoothness runs well past it
    with db.session() as s:
        for k in range(12):
            day = D0 + timedelta(days=10 * k) if k < 11 else last_sm
            for i in range(5):
                s.add(SmoothnessResult(
                    filename=f"S_{k}_{i}.xls", model="S", serial=f"S-{k}-{i}",
                    overall_status=StatusType.PASS,
                    file_date=day, max_smoothness_value=0.01 + 0.001 * i))

    series = compute_spc_series(db, "S", "max_smoothness_value")
    assert series.points, "smoothness series must not come back empty"
    assert series.points[-1].end == last_sm          # newest lot survived the cut
