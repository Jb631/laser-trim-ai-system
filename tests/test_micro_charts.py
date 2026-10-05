"""The Overview's small charts on plain Tk canvases -- gui/v6/widgets/micro_charts.py (option B,
2026-10-04). Never matplotlib: the list draws one per model, dozens on one page.

  * MiniLine: twelve months of pass % as a line in a laser's colour, with a GAP where a month had
    no units (finish item 11: the cards' month bars read as broken); a month alone between gaps is
    a dot, never dropped;
  * PassMeter: segments lit in the pass colour, the rest in the divider colour -- never all lit
    below 100%, never none lit above 0%;
  * MonthChart: the detail's twelve months, with its 0/50/100 guides and the months named.

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


# ---- MiniLine ------------------------------------------------------------------------------------

def test_a_mini_line_joins_neighbouring_months_and_leaves_a_gap_for_a_month_with_no_units(tk_root, t):
    line = mc.MiniLine(tk_root, t, MONTHS, color=t.SERIES_B, bg=t.CARD)
    assert len(line.find_withtag("line")) == 1                 # Nov -> Dec only: one segment
    assert len(line.find_withtag("dot")) == 1                  # Aug stands alone: a dot, not dropped
    assert set(_fills(line, "line")) == {t.SERIES_B} and set(_fills(line, "dot")) == {t.SERIES_B}
    (x0, _y0, x1, _y1), = _coords(line, "line")
    (dx0, _dy0, dx1, _dy1), = _coords(line, "dot")
    assert dx1 < x0 < x1                                       # the dot is left of the segment
    assert line.cget("bg") == t.CARD


def test_a_mini_line_puts_a_hundred_at_the_top_and_nothing_at_the_bottom(tk_root, t):
    line = mc.MiniLine(tk_root, t, [0.0, 100.0], color=t.SERIES_A, bg=t.CARD)
    (_x0, y_low, _x1, y_high), = _coords(line, "line")
    assert y_high < y_low                                      # Tk's y grows downwards
    assert y_high <= 3 * line.scale and y_low >= int(line.cget("height")) - 3 * line.scale - 1


def test_a_mini_line_with_no_month_draws_nothing_and_its_background_follows_the_row(tk_root, t):
    line = mc.MiniLine(tk_root, t, [None] * 12, color=t.SERIES_C, bg=t.CARD)
    assert line.find_all() == ()
    line.set_background(t.ELEVATED)
    assert line.cget("bg") == t.ELEVATED


def test_a_mini_line_is_sized_in_customtkinter_units(tk_root, t):
    import customtkinter as ctk
    holder = ctk.CTkFrame(tk_root)
    try:
        holder._set_scaling(1.5, 1.5)
        line = mc.MiniLine(holder, t, MONTHS, color=t.SERIES_B, bg=t.CARD, width=80, height=20)
        assert (int(line.cget("width")), int(line.cget("height"))) == (120, 30)
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
    meter.set_value(None)                                      # unknown: nothing lit
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
