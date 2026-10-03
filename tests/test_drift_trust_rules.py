"""The drift watch must not raise an alarm the data does not support (James, 2026-10-02:
"im also concerned about dirty data and accuracy of the app telling me things are drifting").

Four rules, each pinned here on a small database built for the purpose:

1. Dirty readings never feed drift: a file the processor marked SUSPECT, and a reading that
   cannot be real (an electrical angle above 400 degrees, a resistance >= 10 MOhm or <= 0, a
   resistance change computed from such a resistance, the analyser's 999.999 marker). How many
   were left out is reported, per metric. An angle at or below 0 is NOT impossible here: the
   home copy's 2,416 negative angles are -0.1 on the 8340 family and 7715, whose angles all sit
   near zero (0.4-3.3 on average) -- the same scale, not a fault.
2. A lot of fewer than 3 units is shown but cannot raise an alarm on its own: it is judged
   together with the lot after it (when that lot comes within 90 days), and the newest one waits.
3. Only recent evidence alarms: lots more than 90 days apart do not add up, and a metric whose
   newest judged lot is more than 90 days older than the newest file cannot raise the tier.
4. One-sided where there is a better direction: less error, fewer fails and fewer escapes are
   good news, never an alarm. Resistance and electrical angle stay two-sided (both ways are a
   change in the process).

Every dataset is INVENTED.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Every lot is weekly and well clear of "now", so it is a CLOSED lot (a lot closes once the
# 3-day changeover gap has elapsed after its last unit).
T0 = datetime(2025, 1, 6)


def _db(tmp_path, name="drift.db"):
    from laser_trim_analyzer.database.manager import DatabaseManager
    return DatabaseManager(tmp_path / name)


def _lot(db, model, day, values, *, column="final_linearity_error_shifted", status="PASS",
         data_quality="good", extra=None, tag=""):
    """One production lot: one file and one track per value, all on `day`."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as AR, TrackResult as TR, SystemType, StatusType)
    ids = []
    with db.session() as s:
        for i, v in enumerate(values):
            key = f"{model}-{day:%Y%m%d}-{tag}{i}"
            ar = AR(filename=f"{key}.xls", file_path=f"/f/{key}", file_hash=key, model=model,
                    serial=f"{tag}{day:%j}{i}", system=SystemType.A,
                    file_date=day + timedelta(minutes=i), timestamp=day,
                    overall_status=getattr(StatusType, status), has_multi_tracks=False,
                    processing_time=0.1, data_quality=data_quality)
            s.add(ar)
            s.flush()
            fields = {column: v}
            fields.update(extra or {})
            s.add(TR(analysis_id=ar.id, track_id="TRK1",
                     status=getattr(StatusType, status), **fields))
            ids.append(ar.id)
        s.commit()
    return ids


def _history(db, model, *, lots=12, start=T0, column="final_linearity_error_shifted",
             centre=0.011, step=0.001, n=5):
    """`lots` ordinary weekly lots whose medians wander 0.010 / 0.011 / 0.012 (for the default
    centre and step): a baseline with a real, known spread. Returns the date after the last."""
    day = start
    for i in range(lots):
        v = centre + (i % 3 - 1) * step
        _lot(db, model, day, [v] * n, column=column, tag="h")
        day += timedelta(days=7)
    return day


def _baseline(n_lots, centre=0.011, step=0.001):
    """The baseline the trainer computes from the first `n_lots` lots of `_history`: it keeps
    all but the newest 3 CLOSED lots, so `n_lots` = every closed lot of the model minus 3."""
    import numpy as np
    arr = np.asarray([centre + (i % 3 - 1) * step for i in range(n_lots)])
    return float(arr.mean()), float(arr.std(ddof=1))


def _status(db, model, metric):
    from laser_trim_analyzer.ml.manager import get_model_drift_status
    return get_model_drift_status(db, model).per_metric[metric]


def _flagged(db):
    from laser_trim_analyzer.ml.manager import get_drifting_models
    return {m.model for m in get_drifting_models(db)}


def _train(db):
    from laser_trim_analyzer.ml.drift_training import train_drift_detector
    return train_drift_detector(db, sensitivity_preset="standard")


# =============================================================================================
# rule 1 -- dirty readings never feed drift, and the count is reported

def test_a_suspect_files_readings_never_feed_drift(tmp_path):
    from laser_trim_analyzer.ml.drift_training import _load_samples_with_dates
    db = _db(tmp_path)
    _lot(db, "M1", T0, [0.010, 0.011, 0.012])
    _lot(db, "M1", T0 + timedelta(days=7), [6.8, 6.8, 6.8, 6.8], data_quality="suspect")
    values = [v for _d, v, _r in _load_samples_with_dates(db, "M1", "linearity_error")]
    assert sorted(values) == [0.010, 0.011, 0.012]


@pytest.mark.parametrize("column,metric,bad,good", [
    ("measured_electrical_angle", "measured_electrical_angle", [401.0, 712.3], [340.0, -0.1, 0.0]),
    ("untrimmed_resistance", "untrimmed_resistance", [1e12, 1.0e7, 0.0], [9500.0]),
    ("untrimmed_error_max", "untrimmed_error_max", [999.999], [0.08]),
])
def test_a_reading_that_cannot_be_real_never_feeds_drift(tmp_path, column, metric, bad, good):
    from laser_trim_analyzer.ml.drift_training import _load_samples_with_dates
    db = _db(tmp_path)
    _lot(db, "M1", T0, good + bad, column=column)
    values = [v for _d, v, _r in _load_samples_with_dates(db, "M1", metric)]
    assert values == good          # good first: one file a minute, in the order given


def test_resistance_change_from_an_impossible_resistance_never_feeds_drift(tmp_path):
    # 5.6e10 % is what a 1e12-ohm reading does to the change: the percentage is only as real as
    # the two resistances it is made from.
    from laser_trim_analyzer.ml.drift_training import _load_samples_with_dates
    db = _db(tmp_path)
    _lot(db, "M1", T0, [12.0], column="resistance_change_percent",
         extra={"untrimmed_resistance": 9000.0, "trimmed_resistance": 10080.0})
    _lot(db, "M1", T0 + timedelta(days=1), [5.6e10], column="resistance_change_percent",
         extra={"untrimmed_resistance": 9000.0, "trimmed_resistance": 1e12}, tag="x")
    _lot(db, "M1", T0 + timedelta(days=2), [11.0], column="resistance_change_percent",
         extra={"untrimmed_resistance": 1.2e7, "trimmed_resistance": 9000.0}, tag="y")
    values = [v for _d, v, _r in _load_samples_with_dates(db, "M1", "resistance_change_percent")]
    assert values == [12.0]


def test_a_suspect_file_is_left_out_of_the_lot_fail_rate(tmp_path):
    from laser_trim_analyzer.ml.drift_training import _load_samples_with_dates
    db = _db(tmp_path)
    _lot(db, "M1", T0, [0.010, 0.011], status="PASS")
    _lot(db, "M1", T0, [6.8], status="FAIL", data_quality="suspect", tag="s")
    flags = [v for _d, v, _r in _load_samples_with_dates(db, "M1", "linearity_fail_fraction")]
    assert flags == [0.0, 0.0]


def test_an_escape_linked_to_a_suspect_trim_is_not_counted(tmp_path):
    from laser_trim_analyzer.database.models import FinalTestResult as FT, StatusType
    from laser_trim_analyzer.ml.drift_training import _load_samples_with_dates
    db = _db(tmp_path)
    good = _lot(db, "M1", T0, [0.010], status="PASS")
    bad = _lot(db, "M1", T0, [0.010], status="PASS", data_quality="suspect", tag="s")
    with db.session() as s:
        for i, trim in enumerate(good + bad):
            s.add(FT(filename=f"ft{i}.xls", model="M1", serial=f"ft{i}", file_hash=f"ft{i}",
                     test_date=T0 + timedelta(days=20), file_date=T0 + timedelta(days=20),
                     overall_status=StatusType.FAIL, linked_trim_id=trim, match_confidence=0.9))
        s.commit()
    flags = [v for _d, v, _r in _load_samples_with_dates(db, "M1", "escape_fraction")]
    assert flags == [1.0]


def test_how_many_readings_were_left_out_is_reported_per_metric(tmp_path):
    from laser_trim_analyzer.ml.drift_training import drift_exclusions
    db = _db(tmp_path)
    _lot(db, "M1", T0, [0.010, 0.011, 0.012])
    _lot(db, "M1", T0 + timedelta(days=7), [6.8, 6.8], data_quality="suspect")
    _lot(db, "M1", T0 + timedelta(days=14), [999.999], tag="m")
    _lot(db, "M1", T0 + timedelta(days=21), [345.0, -0.1, 999.0, 401.5],
         column="measured_electrical_angle", tag="a")
    left_out = drift_exclusions(db, "M1")
    assert left_out["linearity_error"] == {"suspect": 2, "impossible": 1}
    assert left_out["measured_electrical_angle"] == {"suspect": 0, "impossible": 2}
    # a metric with nothing left out is reported as such, never missing
    assert left_out["untrimmed_resistance"] == {"suspect": 0, "impossible": 0}


def test_a_suspect_lot_no_longer_raises_an_alarm(tmp_path):
    # The 7539-2 shape (2026-10-02): an ordinary model whose newest lot is 4 suspect files at
    # 6.8 against a 0.02 band.
    db = _db(tmp_path)
    day = _history(db, "M1")
    _lot(db, "M1", day, [6.8] * 4, data_quality="suspect", tag="s")
    _train(db)
    assert "M1" not in _flagged(db)


# =============================================================================================
# rule 2 -- a lot of fewer than 3 units cannot raise an alarm on its own

def _lots_of(*sizes_and_days, use_mean=False):
    from laser_trim_analyzer.ml.lots import cluster_lots
    samples = []
    for day, values in sizes_and_days:
        samples += [(day, v) for v in values]
    return cluster_lots(samples, use_mean=use_mean)


def test_a_small_lot_is_judged_together_with_the_lot_after_it():
    from laser_trim_analyzer.ml.lots import judged_lots
    lots = _lots_of((T0, [1.0, 1.0, 1.0]), (T0 + timedelta(days=10), [9.0, 9.0]),
                    (T0 + timedelta(days=20), [2.0, 2.0, 2.0, 2.0]))
    judged = judged_lots(lots)
    assert [(l.start, l.end, l.n) for l in judged] == [
        (T0, T0, 3), (T0 + timedelta(days=10), T0 + timedelta(days=20), 6)]
    assert judged[1].median == 2.0          # the median of all six units, not of the small lot


def test_the_newest_small_lot_waits_for_the_next_one():
    from laser_trim_analyzer.ml.lots import judged_lots
    lots = _lots_of((T0, [1.0, 1.0, 1.0]), (T0 + timedelta(days=10), [9.0]))
    assert [l.n for l in judged_lots(lots)] == [3]


def test_two_small_lots_in_a_row_are_judged_as_one():
    from laser_trim_analyzer.ml.lots import judged_lots
    lots = _lots_of((T0, [9.0, 9.0]), (T0 + timedelta(days=10), [9.0, 9.0]))
    judged = judged_lots(lots)
    assert [(l.n, l.median) for l in judged] == [(4, 9.0)]


def test_a_small_lot_followed_only_months_later_is_not_judged():
    from laser_trim_analyzer.ml.lots import judged_lots
    lots = _lots_of((T0, [9.0, 9.0]), (T0 + timedelta(days=200), [1.0, 1.0, 1.0]))
    assert [(l.start, l.n, l.median) for l in judged_lots(lots)] == [
        (T0 + timedelta(days=200), 3, 1.0)]


def test_a_merged_fraction_lot_is_the_rate_over_all_its_units():
    from laser_trim_analyzer.ml.lots import judged_lots
    lots = _lots_of((T0, [1.0]), (T0 + timedelta(days=10), [0.0, 0.0, 0.0]), use_mean=True)
    judged = judged_lots(lots, use_mean=True)
    assert [(l.n, l.median) for l in judged] == [(4, 0.25)]


def test_one_unit_far_out_of_family_does_not_raise_an_alarm(tmp_path):
    # 8902 / 8415-1 (2026-10-02): a single unit was enough to flag the model.
    db = _db(tmp_path)
    day = _history(db, "M1")
    _lot(db, "M1", day, [0.5], tag="x")
    _train(db)
    assert "M1" not in _flagged(db)


def test_two_small_lots_in_a_row_that_agree_do_raise_it(tmp_path):
    db = _db(tmp_path)
    day = _history(db, "M1")
    _lot(db, "M1", day, [0.5, 0.5], tag="x")
    _lot(db, "M1", day + timedelta(days=7), [0.5, 0.5], tag="y")
    _train(db)
    assert "M1" in _flagged(db)


def test_a_waiting_small_lot_is_judged_once_the_next_lot_arrives(tmp_path):
    from laser_trim_analyzer.ml.drift_training import advance_drift_state
    db = _db(tmp_path)
    day = _history(db, "M1")
    _lot(db, "M1", day, [0.5], tag="x")
    _train(db)
    assert "M1" not in _flagged(db)
    _lot(db, "M1", day + timedelta(days=7), [0.5, 0.5, 0.5], tag="y")
    assert advance_drift_state(db, model="M1") >= 1
    assert "M1" in _flagged(db)


# =============================================================================================
# rule 3 -- only recent evidence raises an alarm

def test_lots_more_than_ninety_days_apart_do_not_add_up(tmp_path):
    # Each lot alone is 3.5 sigma out: not enough to alarm by itself (one lot moves the EWMA
    # by a fifth of its distance, and WARNING needs about 0.96 sigma), but two in a row are
    # (about 1.4 sigma). A 120-day pause between them breaks the chain.
    db = _db(tmp_path)
    mean, std = _baseline(11)
    day = _history(db, "M1")
    _lot(db, "M1", day, [mean + 3.5 * std] * 5, tag="x")
    _lot(db, "M1", day + timedelta(days=120), [mean + 3.5 * std] * 5, tag="y")
    _train(db)
    assert "M1" not in _flagged(db)


def test_the_same_two_lots_a_week_apart_do_add_up(tmp_path):
    # The control for the test above: without the pause, the pair alarms.
    db = _db(tmp_path)
    mean, std = _baseline(11)
    day = _history(db, "M1")
    _lot(db, "M1", day, [mean + 3.5 * std] * 5, tag="x")
    _lot(db, "M1", day + timedelta(days=7), [mean + 3.5 * std] * 5, tag="y")
    _train(db)
    assert "M1" in _flagged(db)


def test_an_alarm_with_no_lot_in_the_last_ninety_days_is_not_raised(tmp_path):
    # 2511 (2026-10-02): the lots behind its alarm were from 2020-21. Here M1's out-of-family
    # lots end 140 days before the newest file in the database (another model's).
    db = _db(tmp_path)
    day = _history(db, "M1")
    _lot(db, "M1", day, [0.5] * 5, tag="x")
    _lot(db, "M1", day + timedelta(days=7), [0.5] * 5, tag="y")
    _lot(db, "OTHER", day + timedelta(days=7 + 140), [0.011] * 3)
    _train(db)
    assert "M1" not in _flagged(db)
    status = _status(db, "M1", "linearity_error")
    assert status.is_recent is False
    assert status.newest_lot == day + timedelta(days=7)


def test_an_alarm_whose_newest_lot_is_recent_is_raised(tmp_path):
    # The control: the same lots, with the other model's newest file only 30 days later.
    db = _db(tmp_path)
    day = _history(db, "M1")
    _lot(db, "M1", day, [0.5] * 5, tag="x")
    _lot(db, "M1", day + timedelta(days=7), [0.5] * 5, tag="y")
    _lot(db, "OTHER", day + timedelta(days=7 + 30), [0.011] * 3)
    _train(db)
    assert "M1" in _flagged(db)
    assert _status(db, "M1", "linearity_error").is_recent is True


def test_advance_after_a_long_pause_starts_from_the_baseline(tmp_path):
    from laser_trim_analyzer.ml.drift_training import advance_drift_state
    db = _db(tmp_path)
    mean, std = _baseline(12)
    day = _history(db, "M1", lots=15)
    _train(db)
    # one lot out by 3.5 sigma -- alone, not an alarm ...
    _lot(db, "M1", day, [mean + 3.5 * std] * 5, tag="x")
    advance_drift_state(db, model="M1")
    assert "M1" not in _flagged(db)
    # ... and its twin four months later does not add to it
    _lot(db, "M1", day + timedelta(days=120), [mean + 3.5 * std] * 5, tag="y")
    advance_drift_state(db, model="M1")
    assert "M1" not in _flagged(db)


# =============================================================================================
# rule 4 -- an improvement is good news, not an alarm

def test_less_linearity_error_is_never_an_alarm(tmp_path):
    # 8877 (2026-10-02): its untrimmed error fell from 0.11 to 0.05 and it was flagged.
    db = _db(tmp_path)
    mean, std = _baseline(12)
    day = _history(db, "M1")
    for k in range(3):
        _lot(db, "M1", day + timedelta(days=7 * k), [mean - 9 * std] * 5, tag=f"x{k}")
    _train(db)
    assert "M1" not in _flagged(db)
    assert int(_status(db, "M1", "linearity_error").tier) == 0


def test_more_linearity_error_still_is(tmp_path):
    db = _db(tmp_path)
    mean, std = _baseline(12)
    day = _history(db, "M1")
    for k in range(3):
        _lot(db, "M1", day + timedelta(days=7 * k), [mean + 9 * std] * 5, tag=f"x{k}")
    _train(db)
    assert "M1" in _flagged(db)


def test_resistance_is_watched_both_ways(tmp_path):
    db = _db(tmp_path)
    day = _history(db, "LOW", column="untrimmed_resistance", centre=9000.0, step=60.0)
    for k in range(3):
        _lot(db, "LOW", day + timedelta(days=7 * k), [8000.0] * 5,
             column="untrimmed_resistance", tag=f"x{k}")
    day = _history(db, "HIGH", column="untrimmed_resistance", centre=9000.0, step=60.0)
    for k in range(3):
        _lot(db, "HIGH", day + timedelta(days=7 * k), [10000.0] * 5,
             column="untrimmed_resistance", tag=f"x{k}")
    _train(db)
    assert {"LOW", "HIGH"} <= _flagged(db)


def test_every_watched_metric_has_a_direction():
    from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS, WORSE_DIRECTION
    assert set(WORSE_DIRECTION) == set(WATCHED_METRICS)
    assert set(WORSE_DIRECTION.values()) <= {-1, 0, 1}
    assert WORSE_DIRECTION["untrimmed_resistance"] == 0
    assert WORSE_DIRECTION["measured_electrical_angle"] == 0
    assert WORSE_DIRECTION["linearity_error"] == 1
    assert WORSE_DIRECTION["escape_fraction"] == 1


@pytest.mark.parametrize("direction,shift,trips", [
    (1, +6.0, True), (1, -6.0, False), (-1, -6.0, True), (-1, +6.0, False),
    (0, +6.0, True), (0, -6.0, True)])
def test_the_detector_reads_only_the_worse_side(direction, shift, trips, monkeypatch):
    from laser_trim_analyzer.ml import multi_metric_drift_detector as mmd
    from laser_trim_analyzer.ml.drift_training import corrected_tier_thresholds, _build_detector
    monkeypatch.setitem(mmd.WORSE_DIRECTION, "linearity_error", direction)
    det = _build_detector("linearity_error", 1.0, 0.1, 20,
                          thresholds_dict=corrected_tier_thresholds("standard", 0.1))
    for _ in range(6):
        det.update(1.0 + shift * 0.1)
    assert (int(det.get_status().tier) > 0) is trips


# =============================================================================================
# the rules take effect without anyone remembering to retrain

def test_the_drift_state_is_retrained_once_when_the_rules_change(tmp_path):
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.drift_training import DRIFT_RULES_VERSION, ensure_drift_rules
    db = _db(tmp_path)
    day = _history(db, "M1")
    _lot(db, "M1", day, [0.5], tag="x")
    _train(db)
    # The state as older rules left it: recorded under another version, and carrying the
    # alarm those rules raised from the one-unit lot.
    with db.session() as s:
        db._meta_set(s, "drift_rules", "2026-07-13")
        row = s.query(ModelMetricState).filter_by(model="M1", metric="linearity_error").one()
        row.ewma_state = row.baseline_mean + 50 * row.baseline_std
        s.commit()
    assert "M1" in _flagged(db)
    assert ensure_drift_rules(db, "standard") is True       # retrained under these rules
    assert "M1" not in _flagged(db)
    with db.session() as s:
        assert db._meta_get(s, "drift_rules") == DRIFT_RULES_VERSION
    assert ensure_drift_rules(db, "standard") is False      # and only once


def test_a_database_with_no_drift_state_has_nothing_to_retrain(tmp_path):
    # Whatever is trained later is trained under the current rules, so the version is recorded
    # and no retrain is owed.
    from laser_trim_analyzer.ml.drift_training import DRIFT_RULES_VERSION, ensure_drift_rules
    db = _db(tmp_path)
    assert ensure_drift_rules(db, "standard") is False
    with db.session() as s:
        assert db._meta_get(s, "drift_rules") == DRIFT_RULES_VERSION


# =============================================================================================
# what the page says

def test_the_drift_tab_says_what_was_left_out_and_why():
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import describe_exclusions
    said = describe_exclusions({"linearity_error": {"suspect": 4, "impossible": 1},
                                "untrimmed_resistance": {"suspect": 4, "impossible": 2},
                                "escape_fraction": {"suspect": 1, "impossible": 0}})
    assert "4 readings from files marked suspect" in said
    assert "3 that cannot be real (Linearity error 1, Untrimmed resistance 2)" in said


def test_nothing_left_out_is_said_as_such():
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import describe_exclusions
    assert "nothing" in describe_exclusions({"linearity_error": {"suspect": 0, "impossible": 0}})


def test_a_count_that_failed_never_reads_as_nothing_left_out():
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import describe_exclusions
    said = describe_exclusions(None)
    assert "Could not count" in said and "nothing" not in said


def test_a_metric_with_no_recent_lot_says_how_old_its_evidence_is():
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import alert_text
    from laser_trim_analyzer.ml.drift_types import AlertType, DriftTier, MetricStatus
    base = dict(metric="ft_fail_fraction", tier=DriftTier.STABLE, alert_type=None,
                magnitude=0.0, baseline_mean=0.97, baseline_std=0.12, recent_mean=0.0,
                recent_count=3, is_trained=True)
    assert alert_text(MetricStatus(**base, newest_lot=datetime(2021, 11, 5),
                                   is_recent=False)) == "No lot since Nov 2021"
    assert alert_text(MetricStatus(**base)) == "—"
    assert alert_text(MetricStatus(**{**base, "tier": DriftTier.DRIFT,
                                      "alert_type": AlertType.SLOW_DRIFT})) == "Slow drift"


def test_startup_retrains_under_new_rules_then_reloads_the_page_on_screen(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.gui.v6.app import V6App
    from laser_trim_analyzer.ml.drift_training import DRIFT_RULES_VERSION

    class _Now:                                   # run the catch-up worker inline
        def __init__(self, target=None, daemon=None):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _Now)

    db = _db(tmp_path)
    day = _history(db, "M1")
    _lot(db, "M1", day, [0.5], tag="x")
    _train(db)
    with db.session() as s:                       # state left by older rules, with its alarm
        db._meta_set(s, "drift_rules", "2026-07-13")
        row = s.query(ModelMetricState).filter_by(model="M1", metric="linearity_error").one()
        row.ewma_state = row.baseline_mean + 50 * row.baseline_std
        s.commit()
    posted = []
    app = SimpleNamespace(db=db, config=SimpleNamespace(ml=SimpleNamespace(
                              drift_sensitivity="standard")),
                          ui=SimpleNamespace(post=posted.append),
                          _reload_visible_page=lambda: None)
    V6App._advance_drift_catchup(app)
    assert "M1" not in _flagged(db)
    with db.session() as s:
        assert db._meta_get(s, "drift_rules") == DRIFT_RULES_VERSION
    assert posted == [app._reload_visible_page]
