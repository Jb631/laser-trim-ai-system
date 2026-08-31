"""SPC DB layer: compute_spc_series / compute_focus_list against a tmp DB."""
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import AnalysisResult, StatusType, SystemType
from laser_trim_analyzer.ml.spc import (
    CHRONIC_PBAR, RECENT_K, compute_focus_list, compute_spc_series)

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


# --------------------------------------------------------------------------
# Likely-driver hint (James, 2026-08-30: rows must say WHY a model is listed)
# --------------------------------------------------------------------------

def _status(model, per_metric):
    """Hand-built ModelDriftStatus so the selection logic is tested directly —
    seeding real detector state rows would test the hydrator, not the picker."""
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, MetricStatus, ModelDriftStatus)
    pm = {}
    for m, (tier, base, std, recent, mag) in per_metric.items():
        pm[m] = MetricStatus(metric=m, tier=tier, alert_type=None,
                             magnitude=mag, baseline_mean=base,
                             baseline_std=std, recent_mean=recent,
                             recent_count=5, is_trained=True)
    worst = max(pm, key=lambda k: int(pm[k].tier)) if pm else None
    return ModelDriftStatus(model=model, overall_tier=max(
        (s.tier for s in pm.values()), default=DriftTier.STABLE),
        worst_metric=worst, worst_alert_type=None, per_metric=pm)


def test_clean_since_counts_the_closed_lots_after_the_blip(db):
    """A model that blipped and then ran clean is recovering, not burning.

    James (2026-08-30, on 6126): a single hairline excursion followed by clean
    lots ranked above models that are failing right now. The count of closed,
    in-band lots since the newest alarm is the evidence that separates them.
    """
    # BLIP: 10 quiet lots, one 60% excursion, then TWO clean lots.
    for k in range(10):
        _add_lot(db, "BLIP", D0 + timedelta(days=7 * k), 20, 2)
    _add_lot(db, "BLIP", D0 + timedelta(days=7 * 10), 20, 12)      # the excursion
    for k in (11, 12):
        _add_lot(db, "BLIP", D0 + timedelta(days=7 * k), 20, 2)    # ran clean since
    # CLOCK runs a week past BLIP and so owns the anchor: BLIP's own last lot
    # is then CLOSED (an open lot is never recovery proof — see below).
    _seed(db, "CLOCK", start=D0 + timedelta(days=14))

    e = next(x for x in compute_focus_list(db).focus if x.model == "BLIP")
    assert not e.series.points[-1].is_open              # fixture sanity
    assert e.clean_since == 2
    assert e.rank_score == pytest.approx(e.excess_per_week / 3.0)
    assert e.sub_line.endswith(" · has run at baseline since")
    # The MEASURED cost is never discounted — only the urgency is.
    assert e.excess_per_week == pytest.approx(
        max(e.p_recent - e.p_base, 0.0) * e.units_per_week)


def test_no_clean_marker_when_the_newest_closed_lot_is_the_flagged_one(db):
    """Still burning: nothing has run since the alarm, so nothing is discounted."""
    for k in range(11):
        _add_lot(db, "HOT", D0 + timedelta(days=7 * k), 20, 2)
    _add_lot(db, "HOT", D0 + timedelta(days=7 * 11), 20, 12)       # newest = alarm
    _seed(db, "CLOCK", start=D0 + timedelta(days=7))               # closes HOT's lot

    e = next(x for x in compute_focus_list(db).focus if x.model == "HOT")
    assert e.series.points[-1].ooc and not e.series.points[-1].is_open
    assert e.clean_since == 0
    assert e.rank_score == pytest.approx(e.excess_per_week)
    assert "has run at baseline since" not in e.sub_line


def test_a_terrible_model_recovering_never_says_it_ran_clean(db):
    """The marker must not read as "fine" for a model that fails 3 units in 4.

    8340-1 on the real database: baseline 73%, blipped to 94%, then two lots
    back around 73% — inside its own control limits, so the recovery discount
    is right, but the first wording ("has run clean since") sat next to "fail
    rate 73% → 94%" and told the reader the model was fine. "At baseline" is
    what the arithmetic actually checked, so the row is true here and for a
    model whose baseline is 2%; the sub-line prints the baseline either way.
    """
    # Lots of 200, not 20: at a 75% baseline an n=20 lot has a 3-sigma limit
    # above 100%, so nothing can alarm at all — which is the same arithmetic
    # that makes 8340-1's real limits ~98% and its 89% lots "in band".
    for k in range(10):
        _add_lot(db, "AWFUL", D0 + timedelta(days=7 * k), 200, 150)  # 75% baseline
    _add_lot(db, "AWFUL", D0 + timedelta(days=7 * 10), 200, 190)     # 95% excursion
    for k in (11, 12):
        _add_lot(db, "AWFUL", D0 + timedelta(days=7 * k), 200, 150)  # back to 75%
    _seed(db, "CLOCK", start=D0 + timedelta(days=14))

    e = next(x for x in compute_focus_list(db).focus if x.model == "AWFUL")
    assert e.p_base >= CHRONIC_PBAR                     # fixture sanity: awful
    assert e.clean_since == 2                           # the discount is unchanged
    assert e.rank_score == pytest.approx(e.excess_per_week / 3.0)
    assert "clean" not in e.sub_line                    # never claims to be fine
    assert e.sub_line.endswith(" · has run at baseline since")
    # ...and the baseline it returned to is right there to be judged.
    assert f"fail rate {e.p_base * 100:.0f}%" in e.sub_line


def test_recovered_blip_ranks_below_a_smaller_active_fire(db):
    """The ranking answers "what do I work on", not "who blew up hardest"."""
    # RECOVERED: a big 60% blip, then two clean closed lots.
    for k in range(10):
        _add_lot(db, "RECOVERED", D0 + timedelta(days=7 * k), 20, 2)
    _add_lot(db, "RECOVERED", D0 + timedelta(days=7 * 10), 20, 12)
    for k in (11, 12):
        _add_lot(db, "RECOVERED", D0 + timedelta(days=7 * k), 20, 2)
    # ACTIVE: a smaller 35% excursion — but it is the newest lot, still burning.
    start_a = D0 + timedelta(days=7)
    for k in range(12):
        _add_lot(db, "ACTIVE", start_a + timedelta(days=7 * k), 20, 2)
    _add_lot(db, "ACTIVE", start_a + timedelta(days=7 * 12), 20, 7)

    focus = compute_focus_list(db).focus
    rec = next(e for e in focus if e.model == "RECOVERED")
    act = next(e for e in focus if e.model == "ACTIVE")
    assert rec.clean_since == 2 and act.clean_since == 0
    # The old rule put the recovered model first; the discount crosses them.
    assert rec.excess_per_week > act.excess_per_week
    assert rec.rank_score < act.rank_score
    order = [e.model for e in focus]
    assert order.index("ACTIVE") < order.index("RECOVERED")


def test_focus_is_ordered_by_rank_score(db):
    """The list's own contract, asserted on the list it actually returns."""
    for k in range(10):
        _add_lot(db, "RECOVERED", D0 + timedelta(days=7 * k), 20, 2)
    _add_lot(db, "RECOVERED", D0 + timedelta(days=7 * 10), 20, 12)
    for k in (11, 12):
        _add_lot(db, "RECOVERED", D0 + timedelta(days=7 * k), 20, 2)
    start_a = D0 + timedelta(days=7)
    for k in range(12):
        _add_lot(db, "ACTIVE", start_a + timedelta(days=7 * k), 20, 2)
    _add_lot(db, "ACTIVE", start_a + timedelta(days=7 * 12), 20, 7)

    scores = [e.rank_score for e in compute_focus_list(db).focus]
    assert len(scores) >= 2
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_an_open_lot_is_never_recovery_proof(db):
    """A lot that may still be receiving units cannot prove anything yet."""
    for k in range(11):
        _add_lot(db, "OPENISH", D0 + timedelta(days=7 * k), 20, 2)
    _add_lot(db, "OPENISH", D0 + timedelta(days=7 * 11), 20, 12)   # the excursion
    _add_lot(db, "OPENISH", D0 + timedelta(days=7 * 12), 20, 2)    # clean, but OPEN

    e = next(x for x in compute_focus_list(db).focus if x.model == "OPENISH")
    assert e.series.points[-1].is_open                             # fixture sanity
    assert e.clean_since == 0
    assert "has run at baseline since" not in e.sub_line
    assert e.rank_score == pytest.approx(e.excess_per_week)


def test_driver_picks_worst_process_metric_and_formats_it(db, monkeypatch):
    from laser_trim_analyzer.ml import manager as ml_manager
    from laser_trim_analyzer.ml.drift_types import DriftTier
    from laser_trim_analyzer.ml.spc import _likely_driver

    st = _status("M", {
        # outcome metric — MUST never be the driver even at the worst tier
        "linearity_fail_fraction": (DriftTier.OUT_OF_CONTROL, 0.1, 0.05, 0.5, 9.0),
        # two process metrics; resistance has moved further
        "untrimmed_resistance": (DriftTier.DRIFT, 1000.0, 50.0, 1105.0, 2.0),
        "untrimmed_error_max": (DriftTier.WARNING, 0.02, 0.01, 0.025, 1.0),
    })
    monkeypatch.setattr(ml_manager, "get_model_drift_status",
                        lambda _db, _m: st)
    out = _likely_driver(db, "M")
    assert out == "Untrimmed resistance ↑ (+2.1σ vs its baseline)"


def test_driver_none_when_only_outcome_metrics_flag(db, monkeypatch):
    from laser_trim_analyzer.ml import manager as ml_manager
    from laser_trim_analyzer.ml.drift_types import DriftTier
    from laser_trim_analyzer.ml.spc import _likely_driver

    st = _status("M", {
        "linearity_fail_fraction": (DriftTier.OUT_OF_CONTROL, 0.1, 0.05, 0.5, 9.0),
        "ft_fail_fraction": (DriftTier.DRIFT, 0.2, 0.05, 0.4, 3.0),
    })
    monkeypatch.setattr(ml_manager, "get_model_drift_status",
                        lambda _db, _m: st)
    assert _likely_driver(db, "M") is None


def test_driver_survives_a_broken_detector(db, monkeypatch):
    """The hint must degrade to None, never take the list down with it."""
    from laser_trim_analyzer.ml import manager as ml_manager
    from laser_trim_analyzer.ml.spc import _likely_driver

    def _boom(_db, _m):
        raise RuntimeError("hydration failed")
    monkeypatch.setattr(ml_manager, "get_model_drift_status", _boom)
    assert _likely_driver(db, "M") is None


def test_focus_entries_carry_a_driver_field(db, monkeypatch):
    from laser_trim_analyzer.ml import manager as ml_manager
    from laser_trim_analyzer.ml.drift_types import DriftTier

    _seed(db, "HOT", fails_last=12)
    st = _status("HOT", {
        "untrimmed_resistance": (DriftTier.DRIFT, 1000.0, 50.0, 1105.0, 2.0)})
    monkeypatch.setattr(ml_manager, "get_model_drift_status",
                        lambda _db, _m: st)
    res = compute_focus_list(db)
    assert res.focus and res.focus[0].driver == (
        "Untrimmed resistance ↑ (+2.1σ vs its baseline)")
    assert all(e.driver is None for e in res.chronic)
