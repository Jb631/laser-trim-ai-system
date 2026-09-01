"""INVESTIGATE stats table — the VISUAL STRUCTURE of the grid.

James, 2026-08-30, looking at the shipped table: "stats table didnt look right i
dunno if it needs lines or boarders?" He could feel it without naming it. What
he was looking at: every metric can be followed by up to two caption lines (the
"4 not recorded" disclosure and the "this lot: ..." comparison) that span the
full width with NOTHING binding them to the metric above, and the ALL vs
LIN-PASSING column groups were told apart only by two labels on row 0.

The fix is banding plus two rules, not gridlines, so the thing under test is
`band_plan`: which grid rows one metric's background block has to cover. Kept
pure and tested here for the same reason `focus_row_texts` is — a band that is
one row short stops binding the caption it exists to bind, and that is arithmetic
a test can catch, unlike a screenshot nobody re-reads.

Deliberately a NEW file rather than more of `test_spec3c_model.py`: that file
tests what the table SAYS (`cell_texts`, `disclosure_text`, `summary_line`), and
this one tests how it is LAID OUT. A layout failure should not read as a copy
failure.
"""
from laser_trim_analyzer.core.model_stats import (
    DIST_HEADERS, Cell, ModelStats, StatRow)
from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.stats_table import (
    FIRST_DATA_ROW, ROW_GROUPS, ROW_HEADERS, ROW_HEAD_RULE, StatsTableZone,
    band_plan)


# ---- band_plan: the pure layout model -------------------------------------

def test_bands_alternate_so_two_metrics_never_merge_into_one_block():
    """The band IS the boundary. Two banded metrics in a row would read as one
    metric with four caption lines, which is the confusion being fixed."""
    plan = band_plan([0, 0, 0, 0, 0])
    assert [banded for _start, _span, banded in plan] == [
        True, False, True, False, True]


def test_a_band_covers_its_metrics_caption_lines():
    """The load-bearing assertion: the disclosure and the lot line sit INSIDE
    the metric's block, so the eye binds them to the number above them."""
    plan = band_plan([2, 0, 1])
    assert plan == [(FIRST_DATA_ROW, 3, True),
                    (FIRST_DATA_ROW + 3, 1, False),
                    (FIRST_DATA_ROW + 4, 2, True)]


def test_bands_tile_the_table_with_no_gap_and_no_overlap():
    """Each metric starts exactly where the previous one ended — a gap would
    put a caption line on bare card, which is where this started."""
    plan = band_plan([1, 0, 2, 1])
    for (start, span, _b), (next_start, _s, _nb) in zip(plan, plan[1:]):
        assert start + span == next_start


def test_the_header_rows_are_never_banded():
    """Rows 0-2 are the disposition headings, the column names and the rule
    under them. A band that reached up into those would tie the column names to
    the first metric, which is precisely backwards."""
    plan = band_plan([0, 3])
    assert plan[0][0] == FIRST_DATA_ROW
    assert FIRST_DATA_ROW > max(ROW_GROUPS, ROW_HEADERS, ROW_HEAD_RULE)
    covered = {r for start, span, _b in plan for r in range(start, start + span)}
    assert covered.isdisjoint({ROW_GROUPS, ROW_HEADERS, ROW_HEAD_RULE})


def test_an_empty_table_plans_no_bands():
    assert band_plan([]) == []


# ---- the render: the plan is what actually reaches the screen -------------

def _cell(**kw):
    base = dict(n=3, excluded=0, missing=0, avg=4281.8, low=422.0, high=29576.0)
    base.update(kw)
    return Cell(**base)


def _two_metric_stats():
    """One metric that discloses a drop (so it carries a caption line), one
    clean. The exact shape the band plan has to get right."""
    discloses = StatRow(key="untrimmed_resistance", label="Untrimmed resistance",
                        unit="ohms", kind="distribution",
                        all_=_cell(missing=4), lin_passing=_cell())
    clean = StatRow(key="measured_electrical_angle",
                    label="Electrical angle (measured)", unit="deg",
                    kind="distribution", all_=_cell(avg=344.0, low=340.0,
                                                    high=350.0),
                    lin_passing=_cell(avg=344.0, low=340.0, high=350.0))
    return ModelStats(model="6607", rows=[discloses, clean], tracks=3, records=3,
                      cutoff=None, lot=None, future_dated=0, note="")


def _frames_colored(widget, color):
    """Every CTkFrame painted `color`, anywhere under `widget`."""
    import customtkinter as ctk
    out = []
    for child in widget.winfo_children():
        if isinstance(child, ctk.CTkFrame) and child.cget("fg_color") == color:
            out.append(child)
        out.extend(_frames_colored(child, color))
    return out


def _span_of(frame):
    info = frame.grid_info()
    return int(info["row"]), int(info["rowspan"])


def test_the_rendered_bands_match_the_plan(tk_root):
    """The widget draws the plan, not a second layout of its own."""
    theme = ThemeManager()
    zone = StatsTableZone(tk_root, theme=theme)
    try:
        zone.set_stats(_two_metric_stats())
        bands = _frames_colored(zone, theme.ELEVATED)
        # Metric 1 is banded and owns its caption line (2 rows); metric 2 is the
        # unbanded gap, so it contributes no frame at all.
        assert [_span_of(b) for b in bands] == [(FIRST_DATA_ROW, 2)]
    finally:
        zone.destroy()


def test_the_two_rules_separate_the_headers_and_the_dispositions(tk_root):
    """One horizontal rule under the headers, one vertical rule between ALL and
    LIN-PASSING — the only two lines this table gets."""
    theme = ThemeManager()
    zone = StatsTableZone(tk_root, theme=theme)
    try:
        zone.set_stats(_two_metric_stats())
        rules = _frames_colored(zone, theme.BORDER)
        assert len(rules) == 2
        by_row = {int(r.grid_info()["row"]): r for r in rules}
        under_headers = by_row[ROW_HEAD_RULE]
        assert int(under_headers.grid_info()["columnspan"]) == 2 + 2 * len(DIST_HEADERS)
        # The vertical rule sits in its own column between the two groups and
        # runs from the group headings to the last caption line (3 data rows).
        vertical = by_row[ROW_GROUPS]
        assert int(vertical.grid_info()["column"]) == 1 + len(DIST_HEADERS)
        assert int(vertical.grid_info()["rowspan"]) == FIRST_DATA_ROW + 3
    finally:
        zone.destroy()
