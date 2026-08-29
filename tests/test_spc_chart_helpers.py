"""SPC chart helpers: the pure draw-parameter mapping (no Tk, no rendering).

`spc_draw_params` is the ONE place that decides what a chart surface shows for
an `SpcSeries` — which excursions are "right now" (red, annotated) and which are
older history (amber, counted). The full Model-page chart and the FOCUS-list
sparklines both call it, so a difference between them is impossible by
construction. These tests pin that mapping; the drawing itself is exercised by
`scripts/chart_qa_render_all.py`.
"""
from datetime import datetime, timedelta

import math

from laser_trim_analyzer.gui.v6.widgets.focus_chart import spc_draw_params
from laser_trim_analyzer.ml.spc import RECENT_K, build_fraction_series

D0 = datetime(2026, 1, 5)


def _lot_samples(day: datetime, n: int, fails: int):
    """n units on one day, `fails` of them failing linearity."""
    return [(day, 1.0 if i < fails else 0.0) for i in range(n)]


def _make_history(n_lots=12, n_per=20, fails_per=2, gap=7, start=D0):
    """n_lots clean lots, one per week (gap > LOT_GAP_DAYS => distinct lots)."""
    out = []
    for k in range(n_lots):
        out += _lot_samples(start + timedelta(days=gap * k), n_per, fails_per)
    return out


def _split_history():
    """17 weekly lots at 10% fail with a 60% excursion at lot 11 AND lot 17.

    Lot 11 (index 10) sits outside the recent window; lot 17 (index 16) is the
    newest lot. That is the case the whole split exists for: one excursion is
    today's fire, the other is history the reader should not be re-alarmed by.
    """
    hist = _make_history(n_lots=10)                       # lots 1-10, clean
    day = D0 + timedelta(days=7 * 10)
    hist += _lot_samples(day, 20, 12)                     # lot 11 — 60% fail
    for k in range(1, 6):                                 # lots 12-16, clean
        hist += _lot_samples(day + timedelta(days=7 * k), 20, 2)
    last = day + timedelta(days=7 * 6)
    hist += _lot_samples(last, 20, 12)                    # lot 17 — 60% fail
    return hist, last


def _series(hist, anchor):
    return build_fraction_series("8340-1", "linearity_fail_fraction", hist,
                                 anchor=anchor)


def test_vectors_match_the_series():
    samples = _make_history(n_lots=12)
    s = _series(samples, samples[-1][0])
    p = spc_draw_params(s)
    assert p["judged"] is True
    assert p["xs"] == list(range(len(s.points)))
    assert p["values"] == [pt.value for pt in s.points]
    assert p["ucls"] == [pt.ucl for pt in s.points]
    assert p["center"] == s.p_base
    assert p["n_labels"] == ["n=20"] * len(s.points)
    assert p["x_dates"] == [pt.end.strftime("%m/%d") for pt in s.points]
    # A fail fraction's band always starts at zero: a lot that ran BETTER than
    # baseline is good news, so there is no lower alarm edge to draw.
    assert p["band_lo"] == [0.0] * len(s.points)
    assert p["fraction"] is True


def test_recent_flag_vs_older_excursion_split():
    hist, last = _split_history()
    s = _series(hist, last)
    assert s.judged
    assert s.points[10].ooc and s.points[16].ooc          # both really are ooc
    p = spc_draw_params(s)
    assert p["flag_idx"] == [16]                          # inside the last K lots
    assert p["old_idx"] == [10]                           # older history
    assert p["old_ooc_count"] == 1


def test_labels_only_for_recent_flags():
    hist, last = _split_history()
    p = spc_draw_params(_series(hist, last))
    assert set(p["labels"]) == {16}                       # the older one is unlabeled
    assert p["labels"][16].startswith("60% of 20 units failed")


def test_focus_recent_widens_the_window():
    hist, last = _split_history()
    p = spc_draw_params(_series(hist, last), focus_recent=8)
    assert p["flag_idx"] == [10, 16]
    assert p["old_idx"] == [] and p["old_ooc_count"] == 0
    assert set(p["labels"]) == {10, 16}


def test_not_judged_has_no_flags_no_labels_no_limits():
    samples = _make_history(n_lots=6)                     # < MIN_LOTS_TRAIN
    bad_day = samples[-1][0] + timedelta(days=7)
    samples += _lot_samples(bad_day, 20, 18)              # 90% — still unjudged
    p = spc_draw_params(_series(samples, bad_day))
    assert p["judged"] is False
    assert p["flag_idx"] == [] and p["old_idx"] == []
    assert p["old_ooc_count"] == 0 and p["labels"] == {}
    assert math.isnan(p["center"])
    assert all(math.isnan(v) for v in p["ucls"])
    assert p["values"] and p["n_labels"]                  # the dots still draw


def test_open_lot_index():
    samples = _make_history(n_lots=12)
    last_day = samples[-1][0]
    p_open = spc_draw_params(_series(samples, last_day + timedelta(days=1)))
    assert p_open["open_idx"] == len(p_open["xs"]) - 1
    p_closed = spc_draw_params(_series(samples, last_day + timedelta(days=30)))
    assert p_closed["open_idx"] is None


def test_empty_series_is_drawable():
    s = _series([], D0)
    p = spc_draw_params(s)
    assert p["xs"] == [] and p["values"] == [] and p["labels"] == {}
    assert p["open_idx"] is None and p["old_ooc_count"] == 0
    assert len(p["n_labels"]) == 0 and len(p["x_dates"]) == 0


def test_recent_window_default_is_the_shared_constant():
    hist, last = _split_history()
    s = _series(hist, last)
    default = spc_draw_params(s)
    assert default == spc_draw_params(s, focus_recent=RECENT_K)
