"""The finish pass on the pages (option B, 2026-10-04): names, dates, numbers, buttons, the Model
page's top, the charts and the Process page.

James, with Task Manager TMOG open: "now this is an example of a finished peice of solftware". The
screenshots taken that day (docs/superpowers/specs/2026-10-04-option-b-design.md, "The finish list")
found three date formats, a page called two names, 0.005201 beside 0.011, flat text dressed as
buttons, bright blue dropdown arrows, four lines of red and grey before a chart, a fail-rate axis to
125% and a run-on line of network paths. Each test here pins one of those gone. Every example value
is invented.
"""
import ast
import pathlib
import re
from datetime import datetime

import pytest

V6 = pathlib.Path(__file__).resolve().parents[1] / "src/laser_trim_analyzer/gui/v6"


def _walk(widget):
    """Every widget Tk has under `widget` -- tkinter's own winfo_children (CustomTkinter's hides a
    tab view's tabs and a widget's internal parts)."""
    import tkinter
    yield widget
    for child in tkinter.Misc.winfo_children(widget):
        yield from _walk(child)


def _texts(widget):
    out = []
    for w in _walk(widget):
        try:
            text = w.cget("text")
        except Exception:
            continue
        if isinstance(text, str) and text:
            out.append(text)
    return out


def _shown_literals(path: pathlib.Path):
    """Every string literal in `path` a person could be SHOWN: not a docstring, and not the message
    of a logging call (a log line is never on screen -- "Dashboard: yield query failed" stays a
    log key)."""
    tree = ast.parse(path.read_text())
    hidden = {id(node.value) for node in ast.walk(tree)
              if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("debug", "info", "warning", "error", "exception")):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant):
                    hidden.add(id(sub))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in hidden:
            yield node.lineno, node.value


# ---- 1. "Company trends" everywhere (the page the Overview's link opens) ---------------------

def test_the_company_trends_page_is_called_company_trends(make_app):
    """The Overview's link says "Company trends" and the page it opened was titled "Dashboard". The
    key stays "dashboard" (every deep link navigates by key); what a person reads changes."""
    app = make_app()
    page = app.page_container.get_page("dashboard")
    assert page.page_title == "Company trends"
    app.show_page("dashboard")
    page.reload_now()
    assert not [t for t in _texts(page) if "Dashboard" in t]


def test_no_screen_in_the_app_says_dashboard():
    """Every string a V6 screen can show -- the stats table's note about where unit yield lives, the
    database clean-up's "open it again" line -- names the page "Company trends".

    app.py is left out: it registers the page under its key and passes it the key's old word as a
    title, which the page no longer reads (it names itself -- the test above)."""
    offenders = [f"{p.relative_to(V6)}:{line}: {text[:60]!r}"
                 for p in V6.rglob("*.py") if p.name != "app.py"
                 for line, text in _shown_literals(p)
                 if "Dashboard" in text]
    assert not offenders, offenders


def test_the_scan_for_shown_text_skips_log_lines_and_docstrings_only(tmp_path):
    src = tmp_path / "x.py"
    src.write_text('"""Dashboard docstring."""\n'
                   'logger.exception("Dashboard: yield query failed")\n'
                   'label = "see the Dashboard"\n')
    assert [text for _line, text in _shown_literals(src)] == ["see the Dashboard"]


# ---- 2. "Settings → Backlog", never the Pricing card that merged into it (2026-09-20) ---------

def test_the_money_table_sends_you_to_settings_backlog(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.priorities_panel import PrioritiesPanel
    panel = PrioritiesPanel(tk_root, theme=ThemeManager(), on_row_click=lambda m: None)
    panel.set_rows([
        {"model": "INV-1", "ft_total": 40, "ft_fails": 4, "ft_fail_rate": 10.0,
         "price": 12.5, "dollar_impact": 25.0},
        {"model": "INV-2", "ft_total": 30, "ft_fails": 3, "ft_fail_rate": 10.0,
         "price": None, "dollar_impact": None}])
    said = " ".join(_texts(panel))
    assert "Settings → Backlog" in said
    assert "Pricing" not in said


def test_the_cost_priorities_docstring_names_the_backlog_card():
    from laser_trim_analyzer.core import cost_priorities
    doc = cost_priorities.__doc__
    assert not re.search(r"Settings\s*→\s*Pricing", doc)
    assert "Settings → Backlog" in doc


# ---- 3. One way to write a date: gui/v6/formats (320e2b6) -------------------------------------
# The app said "29 Sep 2026", "Jun 26, 2026" and "2026-07-07" on three pages, and its charts said
# "09/19/25" and a rotated "2025-10". An ISO or slashed date anywhere on screen is the old way.

_OLD_DATE = re.compile(r"\d{4}-\d{2}(-\d{2})?|\d{1,2}/\d{1,2}(/\d{2,4})?")


def test_an_axis_says_the_year_where_it_changes_or_once_up_front():
    """The year on January (the brief: "year on January ticks"); an axis that never reaches a new
    year says it on its first label. Never on both: "Nov 2025" beside "Jan 2026" collided on a
    narrow chart (test_spec3c_model's collision guard)."""
    from laser_trim_analyzer.gui.v6 import formats
    days = [datetime(2025, 12, 22), datetime(2025, 12, 29), datetime(2026, 1, 5), None]
    assert formats.axis_labels(days) == ["22 Dec", "29 Dec", "5 Jan 2026", "—"]
    months = [datetime(2025, 10, 1), datetime(2025, 11, 1), datetime(2026, 1, 1), "2026-02-01"]
    assert formats.axis_labels(months, "month") == ["Oct", "Nov", "Jan 2026", "Feb"]
    assert formats.axis_labels([datetime(2026, 7, 1), datetime(2026, 8, 1)], "month") == [
        "Jul 2026", "Aug"]
    assert formats.axis_labels([datetime(2026, 9, 1), datetime(2026, 9, 8)]) == ["1 Sep 2026", "8 Sep"]
    assert formats.axis_labels([datetime(2016, 1, 1), datetime(2018, 1, 1)], "year") == ["2016", "2018"]


def test_the_stats_tables_line_dates_its_window_the_apps_way():
    """"238 track measurements since Jun 26, 2026" (the Units tab, 2026-10-04). core/model_stats
    writes it -- the Excel sheet prints the same words -- so it keeps its own copy of the day rule,
    pinned here to the one in gui/v6/formats."""
    from laser_trim_analyzer.core import model_stats
    from laser_trim_analyzer.core.model_stats import ModelStats
    from laser_trim_analyzer.gui.v6 import formats
    for d in (datetime(2026, 6, 26), datetime(2026, 1, 5, 17, 22), datetime(2019, 12, 31)):
        assert model_stats.day_text(d) == formats.day(d)
        assert model_stats.day_short_text(d) == formats.day_short(d)
    stats = ModelStats(model="INV-1", rows=[], tracks=238, records=238,
                       cutoff=datetime(2026, 6, 26), lot=None, future_dated=0, note="")
    assert model_stats.summary_line(stats) == "238 track measurements since 26 Jun 2026"


def test_a_lots_name_in_the_run_menu_is_written_the_apps_way():
    from laser_trim_analyzer.core.model_stats import _lot_label
    label = _lot_label(datetime(2025, 9, 19, 8), datetime(2025, 9, 28, 16), 34, False)
    assert label == "19 Sep–28 Sep 2025 · 34 units"
    assert _lot_label(datetime(2025, 9, 19), datetime(2025, 9, 19, 9), 1, True) == \
        "19 Sep 2025 · 1 unit · current lot"


def test_the_unit_lists_date_their_rows_the_apps_way(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.ft_units_tab import FtUnitsTab
    from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import SmoothnessTab
    from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab
    t = ThemeManager()
    when = datetime(2026, 6, 26, 14, 5)
    units = UnitsTab(tk_root, theme=t, on_unit_click=lambda u: None, on_export=lambda: None)
    units.set_units([{"analysis_id": 1, "serial": "S-1", "file_date": when,
                      "overall_status": "Pass", "sigma_gradient": 0.0052,
                      "linearity_error": 0.012, "track_status": "PASS"}])
    ft = FtUnitsTab(tk_root, theme=t)
    ft.set_units([{"serial": "S-1", "file_date": when, "result": "PASS", "linked": True,
                   "match": 97, "id": 1}])
    smooth = SmoothnessTab(tk_root, theme=t)
    smooth.set_records([{"serial": "S-1", "file_date": when, "max_smoothness_value": 0.12,
                         "smoothness_spec": 0.5, "smoothness_pass": True, "overall_status": "Pass"}])
    for widget in (units, ft, smooth):
        texts = " | ".join(_texts(widget))
        assert "26 Jun 2026" in texts, texts
        assert not _OLD_DATE.search(texts), texts


def test_the_unit_chart_windows_are_titled_with_the_apps_date(make_app):
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import FtUnitChartModal, UnitChartModal
    app = make_app()
    trim = UnitChartModal(app, app.theme, app.db, {
        "serial": "S1", "overall_status": "PASS", "file_date": datetime(2026, 1, 5, 9, 30),
        "analysis_id": None, "model": "M1", "system": "B"})
    final = FtUnitChartModal(app, app.theme, app.db, {
        "serial": "S1", "result": "FAIL", "file_date": datetime(2026, 1, 6, 11),
        "id": None, "model": "M1", "system": "B"})
    try:
        assert trim.title() == "Unit S1 — 5 Jan 2026 — PASS"
        assert final.title() == "Final test — S1 — 6 Jan 2026 — FAIL"
    finally:
        trim.destroy()
        final.destroy()


def test_the_drift_tab_dates_its_baseline_the_apps_way(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import DriftMetricsTab
    tab = DriftMetricsTab(tk_root, theme=ThemeManager(), on_metric_select=lambda m: None,
                          on_requalify=lambda: None)
    tab.set_baseline_info(("2026-03-01", "new ink", "2026-03-02 10:15:00"))
    assert tab._baseline_lbl.cget("text") == (
        "Baseline period: data since 1 Mar 2026 (requalified 2 Mar 2026 — new ink)")


def test_the_findings_tab_dates_its_facts_the_apps_way(tk_root):
    from test_findings_tab import FACTS, _tab
    tab = _tab(tk_root)
    tables = [{"system": "B", "track": "Track A", "rows": 111, "graded": 89, "n": 1649,
               "first": "2023-09-22", "last": "2026-01-12", "trim_pass_pct": 34.0},
              {"system": "B", "track": "Track A", "rows": 57, "graded": 45, "n": 823,
               "first": "2023-10-05", "last": "2026-09-11", "trim_pass_pct": 41.0}]
    cuts = {"B": {"n": 40, "window": "2025-01-06 .. 2025-06-30", "current_setting": 2950,
                  "days_with_more_than_one_setting_pct": None,
                  "settings": [{"setting": 2950, "n": 40, "pass_pct": 34.0,
                                "median_incoming_resistance": 4600.0,
                                "window": "2025-02-03 .. 2025-06-30"}]}}
    tab.set_data({"facts": dict(FACTS, limit_tables=tables, cut_setting=cuts), "findings": []})
    text = " | ".join(_texts(tab))
    assert "22 Sep 2023 → 12 Jan 2026" in text                  # a limit table's first and last
    assert "6 Jan 2025 – 30 Jun 2025" in text and "3 Feb 2025 – 30 Jun 2025" in text   # cut windows
    assert "22 Sep 2023 → 30 Jun 2024" in text                  # the recipe history (FACTS)
    assert not _OLD_DATE.search(text), text


def test_the_yield_panels_sparkline_dates_its_ends_the_apps_way(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.mini_trend_chart import MiniTrendChart
    chart = MiniTrendChart(tk_root, theme=ThemeManager())
    chart.set_points([("2026-07-07", 90.0), ("2026-07-08", 80.0), ("2026-07-09", 95.0)])
    said = [x.get_text() for x in chart._ax.texts]
    assert "7 Jul 2026" in said and "9 Jul 2026" in said
    assert not [s for s in said if _OLD_DATE.search(s)], said


def _drawn_xticks(fig, ax):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig).draw()
    return [lbl for lbl in ax.get_xticklabels() if lbl.get_text()]


def test_the_lot_chart_labels_its_lots_the_apps_way(tk_root):
    """It read "09/19/25" (2026-10-04). The first label carries the year, and so does the first
    lot of a new year -- never a month/day pair that reads backwards across January."""
    from datetime import timedelta
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
    from laser_trim_analyzer.ml.spc import build_fraction_series
    samples = []
    for k in range(14):
        day = datetime(2025, 10, 6) + timedelta(days=14 * k)
        samples += [(day, 1.0 if i < 2 else 0.0) for i in range(20)]
    series = build_fraction_series("INV-1", "linearity_fail_fraction", samples,
                                   anchor=samples[-1][0])
    chart = FocusChart(tk_root, theme=ThemeManager())
    chart.set_spc_series(series)
    labels = [lbl.get_text() for lbl in _drawn_xticks(chart._fig, chart._ax)]
    assert labels and not re.search(r"\d{4}$", labels[0]), labels      # 2025 told by the 2026 one
    said = [lb for lb in labels if re.search(r"\d{4}$", lb)]
    assert said and said[0].endswith(" 2026") and said[0].split()[1] == "Jan", labels
    assert not [lb for lb in labels if _OLD_DATE.search(lb)], labels
    assert all(re.fullmatch(r"\d{1,2} [A-Z][a-z]{2}( \d{4})?", lb) for lb in labels), labels
    # Lots inside one year say it on the first lot instead.
    one_year = build_fraction_series("INV-1", "linearity_fail_fraction",
                                     [(d, f) for d, f in samples if d.year == 2026],
                                     anchor=samples[-1][0])
    chart.set_spc_series(one_year)
    labels = [lbl.get_text() for lbl in _drawn_xticks(chart._fig, chart._ax)]
    assert labels[0].endswith(" 2026") and not [lb for lb in labels[1:] if " 20" in lb], labels


@pytest.mark.parametrize("days", (45, 400, 15 * 365))
def test_the_units_view_labels_its_dates_the_apps_way(tk_root, days):
    from datetime import timedelta
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
    end = datetime(2026, 9, 29)
    dates = [end - timedelta(days=days * i / 60) for i in range(60, -1, -1)]
    chart = FocusChart(tk_root, theme=ThemeManager())
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates,
                     values=[0.01 + 0.0001 * (i % 7) for i in range(len(dates))],
                     baseline_mean=0.0103, baseline_std=0.0002)
    labels = [lbl.get_text() for lbl in _drawn_xticks(chart._fig, chart._ax)]
    assert len(labels) >= 2, labels
    assert not [lb for lb in labels if _OLD_DATE.search(lb)], labels
    assert [lb for lb in labels if re.search(r"\d{4}$", lb)], labels    # the year is said
    word = r"(\d{1,2} [A-Z][a-z]{2}|[A-Z][a-z]{2})( \d{4})?|\d{4}"
    assert all(re.fullmatch(word, lb) for lb in labels), labels


def _trend(periods, by_month=True):
    rows = [{"period": p, "total": 100, "accepted": 80, "linearity_yield": 80.0} for p in periods]
    return {"periods": periods, "company": rows, "by_system": {"B": rows},
            "partial_last": False, "data_through": datetime(2026, 9, 29, 17, 22)}


def test_the_company_chart_names_its_months_and_never_tilts_them(tk_root):
    """A rotated "2025-10" under every month (2026-10-04) -> "Oct", "Nov", ... "Jan 2026": month
    names, flat, the year on January."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.company_trend_chart import CompanyTrendChart
    chart = CompanyTrendChart(tk_root, theme=ThemeManager())
    months = [f"{y}-{m:02d}" for y, m in [(2025, 10), (2025, 11), (2025, 12), (2026, 1),
                                          (2026, 2), (2026, 3)]]
    chart.set_data(_trend(months), period_label="month")
    ticks = _drawn_xticks(chart._fig, chart._ax)
    assert [lbl.get_text() for lbl in ticks] == ["Oct", "Nov", "Dec", "Jan 2026", "Feb", "Mar"]
    assert all(lbl.get_rotation() == 0 for lbl in ticks)
    said = [x.get_text() for x in chart._ax.texts]
    assert "Data through 29 Sep 2026" in said, said


def test_the_company_chart_names_its_weeks_by_their_monday(tk_root):
    """SQLite's %W weeks start on Monday; week 00 holds a year's days before its first Monday, so
    it is named by 1 January -- never by the Monday of the year before, which week 52 already is."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.company_trend_chart import CompanyTrendChart
    chart = CompanyTrendChart(tk_root, theme=ThemeManager())
    chart.set_data(_trend(["2025-W51", "2025-W52", "2026-W00", "2026-W01"]), period_label="week")
    labels = [lbl.get_text() for lbl in _drawn_xticks(chart._fig, chart._ax)]
    assert labels == ["22 Dec", "29 Dec", "1 Jan 2026", "5 Jan"]


def test_a_long_company_chart_puts_its_ticks_on_whole_months_with_january_named(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.company_trend_chart import CompanyTrendChart
    chart = CompanyTrendChart(tk_root, theme=ThemeManager())
    months = [f"{2020 + i // 12}-{i % 12 + 1:02d}" for i in range(4, 70)]       # May 2020 ..
    chart.set_data(_trend(months), period_label="month")
    labels = [lbl.get_text() for lbl in _drawn_xticks(chart._fig, chart._ax)]
    assert len(labels) <= 13, labels
    assert labels[0] == "Jul"                        # July 2020: the year is January's to say
    assert [lb for lb in labels if lb.startswith("Jan")] == [
        "Jan 2021", "Jan 2022", "Jan 2023", "Jan 2024", "Jan 2025"]


def test_the_history_tab_names_its_months(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.history_tab import HistoryTab
    tab = HistoryTab(tk_root, theme=ThemeManager())
    series = [(datetime(2025, 11, 3) + (datetime(2025, 11, 4) - datetime(2025, 11, 3)) * i, 344.0 + i % 3)
              for i in range(90)]
    tab.set_data({"series": {"measured_electrical_angle": series},
                  "stats": {"measured_electrical_angle": {"n": 90, "mean": 345.0, "std": 0.8,
                                                          "min": 344.0, "max": 346.0, "last": 345.0}},
                  "passrate_periods": [("2025-11", 8, 10, 80.0), ("2025-12", 9, 10, 90.0),
                                       ("2026-01", 7, 10, 70.0)]})
    bottom = [lbl.get_text() for lbl in _drawn_xticks(tab._fig, tab._ax_pr)]
    assert bottom == ["Nov", "Dec", "Jan 2026"], bottom
    top = [lbl.get_text() for lbl in _drawn_xticks(tab._fig, tab._ax_val)]
    assert top and not [lb for lb in top if _OLD_DATE.search(lb)], top


# ---- 4. Numbers: one rule per kind ------------------------------------------------------------
# The Units tab printed one sigma gradient as 0.005201 and the next as 0.011 (2026-10-04): each
# number chose its own precision. A row of the stats table, a column of the unit list and a row of
# the drift table now share theirs -- enough for three significant digits on the largest, and one
# on the smallest that is not zero, so nothing real is ever shown as a zero.

def _decimals(text):
    number = re.search(r"-?[\d,]+(\.(\d+))?", text)
    return len(number.group(2) or "") if number else None


def test_numbers_shown_together_share_their_decimals():
    from laser_trim_analyzer.core.model_stats import decimals_for, fixed
    assert decimals_for([0.005201, 0.002509, 0.011]) == 4
    assert [fixed(v, 4) for v in (0.005201, 0.002509, 0.011)] == ["0.0052", "0.0025", "0.0110"]
    # 8340-1's failed units carry sigma gradients above 1.0 beside ~0.0003: the small one keeps a
    # digit rather than read 0.00.
    assert decimals_for([0.05, 0.0003, 1.5]) == 4
    assert decimals_for([344.2, 330.1, 352.9]) == 0 and fixed(29576.0, 0) == "29,576"
    assert decimals_for([None, 0.0, None]) == 0 and fixed(None, 3) == "—"
    assert fixed(-0.00004, 2) == "0.00"                           # never "-0.00"


def test_a_stats_table_row_shares_one_precision_across_both_groups():
    from laser_trim_analyzer.core.model_stats import Cell, StatRow, cell_texts, lot_line
    every = Cell(n=238, excluded=0, missing=0, avg=0.005201, low=0.002509, high=0.011)
    passing = Cell(n=200, excluded=0, missing=0, avg=0.0049, low=0.002509, high=0.0081)
    row = StatRow(key="sigma_gradient", label="Sigma gradient", unit="", kind="distribution",
                  all_=every, lin_passing=passing)
    assert cell_texts(row, row.all_) == ["238", "0.0052", "0.0025", "0.0110"]
    assert cell_texts(row, row.lin_passing) == ["200", "0.0049", "0.0025", "0.0081"]
    lot = Cell(n=12, excluded=0, missing=0, avg=0.0061, low=0.0040, high=0.0092)
    assert lot_line(row, lot, None) == "this lot: 12 readings · avg 0.0061 · 0.0040 to 0.0092"


def test_every_percentage_in_the_stats_table_has_one_decimal():
    from laser_trim_analyzer.core.model_stats import Cell, StatRow, cell_texts
    margin = Cell(n=50, excluded=0, missing=0, avg=45.26, low=-12.0, high=150.0)
    row = StatRow(key="margin_to_spec", label="Margin to spec limit", unit="%",
                  kind="distribution", all_=margin, lin_passing=margin)
    assert cell_texts(row, row.all_)[1:] == ["45.3%", "-12.0%", "150.0%"]
    rate = Cell(n=9934, excluded=0, missing=10, count=6325, pct=63.67)
    rate_row = StatRow(key="trim_passed_linearity", label="Tracks that passed linearity",
                       unit="%", kind="rate", all_=rate, lin_passing=rate)
    assert cell_texts(rate_row, rate)[2] == "63.7%"


def test_a_column_of_the_unit_list_shares_one_precision(tk_root):
    """The screenshot's own three sigma gradients -- and a failed track's 999.999 marker, shown as
    "—", never counted in its column's precision."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab
    from test_error_reason import _cell_texts
    tab = UnitsTab(tk_root, theme=ThemeManager(), on_unit_click=lambda u: None, on_export=lambda: None)
    rows = [(0.005201, 0.0123), (0.002509, 0.00417), (0.011, 0.0081), (999.999, 999.999)]
    tab.set_units([{"analysis_id": i, "serial": f"S-{i}", "file_date": datetime(2026, 6, 26),
                    "overall_status": "Error" if s > 1 else "Pass", "sigma_gradient": s,
                    "linearity_error": e, "error_reason": None,
                    "track_status": "ERROR" if s > 1 else "PASS"} for i, (s, e) in enumerate(rows)])
    cells = {row.unit["analysis_id"]: _cell_texts(row) for row in tab._rows}
    assert [cells[i]["sigma_gradient"] for i in range(4)] == ["0.0052", "0.0025", "0.0110", "—"]
    assert [cells[i]["linearity_error"] for i in range(4)] == ["0.0123", "0.0042", "0.0081", "—"]


def test_a_drift_table_row_shares_one_precision(tk_root):
    """"0.003235 ± 0.0007776" beside a last lot of 0.0035 (8232-1's untrimmed sigma gradient row)."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import _MetricRow
    from laser_trim_analyzer.ml.drift_types import DriftTier, MetricStatus
    ms = MetricStatus(metric="untrimmed_sigma_gradient", tier=DriftTier.WARNING, alert_type=None,
                      magnitude=1.0, baseline_mean=0.003235, baseline_std=0.0007776,
                      recent_mean=0.0035, recent_count=5, is_trained=True)
    row = _MetricRow(tk_root, ms=ms, theme=ThemeManager(), on_click=lambda m: None)
    cells = [w.cget("text") for w in row.winfo_children() if hasattr(w, "cget")
             and isinstance(w.cget("text"), str)]
    assert "0.00324 ± 0.00078" in cells and "0.00350" in cells, cells


def test_the_history_tabs_figures_share_one_precision(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.history_tab import HistoryTab
    tab = HistoryTab(tk_root, theme=ThemeManager())
    tab.set_data({"series": {"untrimmed_resistance": [(datetime(2026, 1, 5), 4281.8)]},
                  "stats": {"untrimmed_resistance": {"n": 120, "mean": 4281.8, "std": 61.25,
                                                     "min": 4102.0, "max": 4499.9, "last": 4300.0}},
                  "passrate_periods": []})
    assert tab._stats.cget("text") == (
        "n=120   mean=4,282   σ=61   min=4,102   max=4,500   last=4,300")


# ---- 5. Real buttons, quiet dropdowns ----------------------------------------------------------
# "Copy summary" and "Export model to Excel" were flat card-coloured text with no border; the model
# picker, "90d" and the run menu each had a bright blue square for an arrow (2026-10-04). The ONE
# blue button is the top bar's.

def test_an_icon_survives_a_new_tk_root():
    """A Tk image belongs to the interpreter that made it -- theme.font()'s own lesson. The icon
    cache handed the second root a CTkImage whose picture lived in the first: "image "pyimage1"
    doesn't exist". The app has one root; the tests build one each."""
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6 import icons
    for _ in range(2):
        root = ctk.CTk()
        try:
            root.withdraw()
            button = ctk.CTkButton(root, text="Copy summary", compound="left",
                                   image=icons.icon("copy", "#8b8b93", 16))
            button.pack()
            root.update_idletasks()
        finally:
            root.destroy()


def test_a_secondary_button_is_a_real_button(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets import blocks
    t = ThemeManager()
    b = blocks.secondary_button(tk_root, t, "Copy summary", lambda: None, icon="copy")
    assert (b.cget("fg_color"), b.cget("border_width"), b.cget("border_color")) == (t.CARD, 1, t.BORDER)
    assert (b.cget("text_color"), b.cget("hover_color")) == (t.TEXT_PRIMARY, t.ELEVATED)
    assert b.cget("image") is not None and b.cget("height") == 32
    plain = blocks.secondary_button(tk_root, t, "Browse…", lambda: None)
    assert plain.cget("image") is None
    danger = blocks.secondary_button(tk_root, t, "Clear selected", lambda: None, tone="check")
    assert danger.cget("text_color") == t.CHECK and danger.cget("fg_color") == t.CARD


def test_the_model_pages_two_actions_are_buttons_with_their_icons(make_app):
    app = make_app()
    page = app.page_container.get_page("model")
    t = page.theme
    import customtkinter as ctk
    buttons = {w.cget("text"): w for w in _walk(page) if isinstance(w, ctk.CTkButton)}
    for name in ("Copy summary", "Export model to Excel"):
        b = buttons[name]
        assert (b.cget("fg_color"), b.cget("border_width")) == (t.CARD, 1), name
        assert b.cget("image") is not None, name


def test_a_dropdown_paints_its_arrow_quietly_after_every_draw(tk_root):
    """CustomTkinter repaints the arrow in the value's colour on every draw; the dropdown repaints
    it TEXT_SECONDARY after each one -- the value itself stays TEXT_PRIMARY."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets import blocks
    t = ThemeManager()
    menus = (blocks.dropdown(tk_root, t, ["30d", "90d"], width=80),
             blocks.combo_box(tk_root, t, ["INV-1", "INV-2"], width=200))
    for w in menus:
        w.pack()
        for _ in range(2):
            w._draw()                                     # CustomTkinter's own redraw
            assert w._canvas.itemcget("dropdown_arrow", "fill") == t.TEXT_SECONDARY
        assert w.cget("text_color") == t.TEXT_PRIMARY
        assert (w.cget("button_color"), w.cget("button_hover_color")) == (t.ELEVATED, t.BORDER)
        w.configure(state="disabled")
        assert w._canvas.itemcget("dropdown_arrow", "fill") != t.TEXT_SECONDARY   # greyed with it


def test_the_findings_page_opens_a_model_with_a_secondary_button(tk_root):
    """Opening a row was the Findings page's one teal-filled button -- drawn beside the top bar's,
    two blue buttons on one screen since the Graphite bar (2026-10-02)."""
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.findings_view import FindingsView
    from test_findings_view import cut
    t = ThemeManager()
    view = FindingsView(tk_root, t, on_open=lambda m: None)
    view.set_findings([cut("6607", 182.0)])
    view.toggle(next(iter(view.row_widgets)))
    (button,) = [b for b in view._detail.winfo_children() if isinstance(b, ctk.CTkButton)]
    assert button.cget("text") == "Open 6607"
    assert (button.cget("fg_color"), button.cget("border_width")) == (t.CARD, 1)
