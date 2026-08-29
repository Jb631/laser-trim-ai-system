"""SPC core: pure lot-series builders (no DB, no Tk)."""
from datetime import datetime, timedelta

import math
import pytest

from laser_trim_analyzer.ml.spc import (
    ACTIVE_DAYS, CHRONIC_PBAR, RECENT_K, SERIES_WINDOW,
    SpcPoint, SpcSeries, build_continuous_series, build_fraction_series)

D0 = datetime(2026, 1, 5)


def _lot_samples(day: datetime, n: int, fails: int):
    """n units on one day, `fails` of them failing."""
    return [(day, 1.0 if i < fails else 0.0) for i in range(n)]


def _make_history(n_lots=12, n_per=20, fails_per=2, gap=7, start=D0):
    """n_lots clean lots, one per week (gap > LOT_GAP_DAYS => distinct lots)."""
    out = []
    for k in range(n_lots):
        out += _lot_samples(start + timedelta(days=gap * k), n_per, fails_per)
    return out


def test_binomial_limits_known_answer():
    # baseline p = 0.10 across 7 baseline lots of 20; UCL for n=20:
    # 0.10 + 3*sqrt(0.09/20) = 0.5012...  -> a 40% lot stays in control,
    # an 11/20 (55%) lot is out.
    samples = _make_history(n_lots=12, n_per=20, fails_per=2)
    anchor = samples[-1][0]
    s = build_fraction_series("M", "linearity_fail_fraction", samples, anchor=anchor)
    assert s.judged
    assert s.p_base == pytest.approx(0.10)
    last = s.points[-1]
    assert last.ucl == pytest.approx(0.10 + 3 * math.sqrt(0.09 / 20), abs=1e-9)
    assert not last.ooc


def test_ooc_flag_and_note_wording():
    samples = _make_history(n_lots=11, n_per=20, fails_per=2)
    bad_day = samples[-1][0] + timedelta(days=7)
    samples += _lot_samples(bad_day, 20, 11)          # 55% fail lot
    s = build_fraction_series("M", "linearity_fail_fraction", samples, anchor=bad_day)
    last = s.points[-1]
    assert last.ooc
    assert last.note == f"55% of 20 units failed — expected at most {last.ucl*100:.0f}%"


def test_membership_window_edge():
    # An excursion RECENT_K lots ago is still "recent"; one lot older is not.
    base = _make_history(n_lots=10, n_per=20, fails_per=2)
    bad_day = base[-1][0] + timedelta(days=7)
    hist = base + _lot_samples(bad_day, 20, 12)
    # exactly RECENT_K - 1 clean lots after the bad one -> bad lot is points[-RECENT_K]
    for k in range(1, RECENT_K):
        hist += _lot_samples(bad_day + timedelta(days=7 * k), 20, 2)
    anchor = hist[-1][0]
    s = build_fraction_series("M", "m", hist, anchor=anchor)
    assert s.points[-RECENT_K].ooc                      # in the window
    # one more clean lot pushes it out of the recent window
    hist2 = hist + _lot_samples(anchor + timedelta(days=7), 20, 2)
    s2 = build_fraction_series("M", "m", hist2, anchor=hist2[-1][0])
    assert not any(pt.ooc for pt in s2.points[-RECENT_K:])


def test_open_lot_marked_and_baseline_excludes_recent():
    samples = _make_history(n_lots=12, n_per=20, fails_per=2)
    anchor = samples[-1][0] + timedelta(days=1)         # newest lot still open
    s = build_fraction_series("M", "m", samples, anchor=anchor)
    assert s.points[-1].is_open and not any(p.is_open for p in s.points[:-1])
    assert s.baseline_n_lots == 12 - RECENT_K
    assert s.baseline_units == (12 - RECENT_K) * 20


def test_not_judged_below_min_lots():
    samples = _make_history(n_lots=7, n_per=20, fails_per=2)   # < MIN_LOTS_TRAIN
    s = build_fraction_series("M", "m", samples, anchor=samples[-1][0])
    assert not s.judged
    assert math.isnan(s.p_base) and math.isnan(s.points[-1].ucl)
    assert not any(pt.ooc for pt in s.points)


def test_requal_floor_drops_older_lots():
    samples = _make_history(n_lots=14, n_per=20, fails_per=2)
    floor = samples[0][0] + timedelta(days=7 * 4)       # drop first 4 lots
    s = build_fraction_series("M", "m", samples, anchor=samples[-1][0],
                              requal_floor=floor)
    assert len(s.points) == 10


def test_future_dated_samples_excluded():
    samples = _make_history(n_lots=12, n_per=20, fails_per=2)
    anchor = samples[-1][0]
    samples += _lot_samples(anchor + timedelta(days=200), 5, 5)   # file-date junk
    s = build_fraction_series("M", "m", samples, anchor=anchor)
    assert s.points[-1].end <= anchor


def test_chronic_flag():
    hot = _make_history(n_lots=12, n_per=20, fails_per=8)   # stable 40% fail
    s = build_fraction_series("M", "m", hot, anchor=hot[-1][0])
    assert s.judged and s.chronic and not any(p.ooc for p in s.points[-RECENT_K:])
    cool = _make_history(n_lots=12, n_per=20, fails_per=1)  # 5% < CHRONIC_PBAR
    s2 = build_fraction_series("M", "m", cool, anchor=cool[-1][0])
    assert not s2.chronic


def test_series_window_cap():
    samples = _make_history(n_lots=SERIES_WINDOW + 10, n_per=5, fails_per=0)
    s = build_fraction_series("M", "m", samples, anchor=samples[-1][0])
    assert len(s.points) == SERIES_WINDOW
    assert s.baseline_n_lots == SERIES_WINDOW + 10 - RECENT_K  # baseline saw ALL lots


def test_continuous_two_sided_and_degenerate():
    vals = []
    for k in range(12):
        day = D0 + timedelta(days=7 * k)
        vals += [(day, 100.0 + (k % 3)) for _ in range(10)]   # medians 100..102
    jump_day = vals[-1][0] + timedelta(days=7)
    vals += [(jump_day, 140.0) for _ in range(10)]
    s = build_continuous_series("M", "resistance", vals, anchor=jump_day)
    assert s.judged and s.points[-1].ooc
    assert "outside this model's normal" in s.points[-1].note
    flat = [(D0 + timedelta(days=7 * k), 5.0) for k in range(12) for _ in range(10)]
    s2 = build_continuous_series("M", "r", flat, anchor=flat[-1][0])
    assert not s2.judged                                    # zero spread guard
