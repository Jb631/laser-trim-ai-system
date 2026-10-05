"""The Overview's small charts on plain Tk canvases -- gui/v6/widgets/micro_charts.py (option B,
2026-10-04). Never matplotlib: the list draws one per model, dozens on one page.

  * MonthBars: a list row's twelve months of pass % as bars on one 0-100 scale (James, 2026-10-04,
    of the line before them: "these little charts and % dont mean anything?"): the months the
    90-day % covers bright, the rest muted, a month with no units a faint flat mark -- never a gap,
    never a bar that reads as 0% -- and a month with units never too short to see;
  * PassMeter: segments lit in the pass colour, the rest in the divider colour -- never all lit
    below 100%, never none lit above 0%, and nothing at all for a pass % that is not known;
  * MonthChart: the detail's twelve months, with its 0/50/100 guides and the months named -- and,
    with no month, why: "no graded units", or the failure the caller names.

Values here are INVENTED.
"""
import pytest

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets import micro_charts as mc

MONTHS = [None] * 8 + [50.0, None, 75.0, 100.0]


@pytest.fixture
def t(tk_root):
    return ThemeManager()


def _fills(canvas, tag):
    return [canvas.itemcget(i, "fill") for i in canvas.find_withtag(tag)]


def _coords(canvas, tag):
    return [tuple(canvas.coords(i)) for i in canvas.find_withtag(tag)]


# ---- MonthBars -----------------------------------------------------------------------------------

def test_month_bars_draw_a_bar_for_each_month_with_units_and_a_flat_mark_for_each_without(tk_root, t):
    bars = mc.MonthBars(tk_root, t, MONTHS, recent=3, bg=t.CARD)
    assert len(bars.find_withtag("bar")) == 3                   # the three months with units
    assert len(bars.find_withtag("empty")) == 9                 # every month without: a slot, no gap
    assert set(_fills(bars, "empty")) == {t.BORDER}
    lefts = [c[0] for c in _coords(bars, "bar") + _coords(bars, "empty")]
    assert len(set(round(x) for x in lefts)) == 12              # twelve slots, side by side
    assert bars.cget("bg") == t.CARD


def test_the_months_the_90_days_cover_are_bright_and_the_year_before_is_muted(tk_root, t):
    bars = mc.MonthBars(tk_root, t, [80.0] * 12, recent=4, bg=t.CARD)
    assert _fills(bars, "older") == [t.CHART_BAR_MUTED] * 8
    assert _fills(bars, "recent") == [t.CHART_BAR] * 4
    newest = max(c[0] for c in _coords(bars, "recent"))
    assert all(c[0] < min(x[0] for x in _coords(bars, "recent")) for c in _coords(bars, "older"))
    assert newest == max(c[0] for c in _coords(bars, "bar"))    # the bright ones are the newest


def test_every_bar_shares_one_0_to_100_scale_and_a_month_with_units_always_shows(tk_root, t):
    bars = mc.MonthBars(tk_root, t, [100.0, 50.0, 0.0, 2.0], recent=0, bg=t.CARD, height=26)
    (hundred, half, zero, two) = _coords(bars, "bar")
    bottom = hundred[3]
    assert {half[3], zero[3], two[3]} == {bottom}               # all stand on one baseline
    assert hundred[1] == 0.0                                    # 100% reaches the top...
    assert abs((bottom - half[1]) - bottom / 2) <= 1            # ...and 50% half of it
    assert bottom - zero[1] == mc.MonthBars.MIN_BAR * bars.scale     # 0% of units passing: shown
    assert bottom - two[1] == mc.MonthBars.MIN_BAR * bars.scale
    pair = mc.MonthBars(tk_root, t, [0.0, None], recent=0, bg=t.CARD)
    (bar,), (flat,) = _coords(pair, "bar"), _coords(pair, "empty")
    assert flat[3] - flat[1] < bar[3] - bar[1]                  # no units: flatter than 0%...
    assert flat[2] - flat[0] < bar[2] - bar[0]                  # ...and narrower than a bar


def test_month_bars_with_no_months_draw_nothing_and_their_background_follows_the_row(tk_root, t):
    bars = mc.MonthBars(tk_root, t, [], recent=3, bg=t.CARD)
    assert bars.find_all() == ()
    bars.set_background(t.ELEVATED)
    assert bars.cget("bg") == t.ELEVATED


def test_month_bars_are_sized_in_customtkinter_units(tk_root, t):
    import customtkinter as ctk
    holder = ctk.CTkFrame(tk_root)
    try:
        holder._set_scaling(1.5, 1.5)
        bars = mc.MonthBars(holder, t, MONTHS, recent=3, bg=t.CARD, width=96, height=26)
        assert (int(bars.cget("width")), int(bars.cget("height"))) == (144, 39)
    finally:
        holder._set_scaling(1.0, 1.0)


# ---- PassMeter -----------------------------------------------------------------------------------

@pytest.mark.parametrize("pct,lit", [(None, 0), (0.0, 0), (0.3, 1), (50.0, 10), (71.6, 14),
                                     (99.6, 19), (100.0, 20)])
def test_a_meter_lights_its_share_but_never_all_or_nothing_by_rounding(pct, lit):
    assert mc.PassMeter.lit(pct, 20) == lit


def test_a_meter_draws_lit_segments_in_the_pass_colour_and_the_rest_in_the_divider(tk_root, t):
    meter = mc.PassMeter(tk_root, t, 71.6, bg=t.CARD)
    assert _fills(meter, "on") == [t.PASS_FG] * 14
    assert _fills(meter, "off") == [t.DIVIDER] * 6
    lefts = [c[0] for c in _coords(meter, "segment")]
    assert lefts == sorted(lefts)                              # lit first, from the left
    meter.set_value(None)               # not known (its read failed): no meter -- an empty one is 0%
    assert meter.find_all() == ()
    meter.set_value(0.0)                                       # a real 0%: every segment, none lit
    assert _fills(meter, "on") == [] and len(_fills(meter, "off")) == 20


# ---- MonthChart ----------------------------------------------------------------------------------

LABELS = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]


def test_the_month_chart_names_its_months_and_its_guides(tk_root, t):
    chart = mc.MonthChart(tk_root, t, bg=t.CARD)
    chart.set_data(MONTHS, LABELS, t.SERIES_A)
    texts = lambda tag: [chart.itemcget(i, "text") for i in chart.find_withtag(tag)]   # noqa: E731
    assert texts("xlabel") == LABELS
    assert texts("ylabel") == ["0%", "50%", "100%"]
    assert set(_fills(chart, "guide")) == {t.DIVIDER} and len(chart.find_withtag("guide")) == 3
    assert len(chart.find_withtag("line")) == 1 and set(_fills(chart, "line")) == {t.SERIES_A}
    assert len(chart.find_withtag("dot")) == 3                 # every month with units
    assert texts("value") == ["100%"]                          # the newest month, said


def test_the_month_chart_with_no_month_says_so(tk_root, t):
    chart = mc.MonthChart(tk_root, t, bg=t.CARD)
    chart.set_data([None] * 12, LABELS, t.SERIES_A)
    assert chart.find_withtag("line") == () and chart.find_withtag("dot") == ()
    assert [chart.itemcget(i, "text") for i in chart.find_withtag("note")] == [
        "No graded units in these twelve months"]


def test_the_month_chart_names_a_failure_instead_of_no_graded_units(tk_root, t):
    chart = mc.MonthChart(tk_root, t, bg=t.CARD)
    chart.set_data([], LABELS, t.SERIES_A, empty_text="Could not be worked out.")
    assert [chart.itemcget(i, "text") for i in chart.find_withtag("note")] == [
        "Could not be worked out."]
    chart.set_data([None] * 12, LABELS, t.SERIES_A)            # the next model: no failure to name
    assert [chart.itemcget(i, "text") for i in chart.find_withtag("note")] == [mc.MonthChart.EMPTY]
