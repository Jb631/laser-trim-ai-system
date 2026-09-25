"""Spec 3c — Model page. Foundations §3/§4.3. Fixtures in tests/conftest.py."""
from datetime import datetime, timedelta

import pytest

# ---- Task 1: routing + column map ----------------------------------------

def test_consume_model_route_full(make_app):
    app = make_app()
    app.set_model_route("M1", "linearity_error")
    assert app.consume_model_route_full() == ("M1", "linearity_error")
    assert app.consume_model_route_full() == (None, None)


def test_consume_model_route_full_without_focus(make_app):
    app = make_app()
    app.set_model_route("M2")
    assert app.consume_model_route_full() == ("M2", None)


def test_showing_a_model_selects_the_requested_tab(make_app):
    """Task 7: the Findings page's opened-row button routes here with tab="findings" --
    the page must land on its Findings tab, not whichever tab was last selected."""
    app = make_app()
    app.set_model_route("HOT", tab="findings")
    page = app.page_container.get_page("model")
    page._reload = lambda **kw: None            # suppress the background DB reload -- irrelevant here
    app.show_page("model")
    del page._reload
    assert page._tabs.get() == "Findings"


def test_showing_a_model_without_a_tab_request_leaves_the_tab_alone(make_app):
    app = make_app()
    app.set_model_route("HOT")
    page = app.page_container.get_page("model")
    page._reload = lambda **kw: None
    app.show_page("model")
    del page._reload
    assert page._tabs.get() == "Drift metrics"   # CTkTabview's own default: the first tab added


def test_an_unknown_tab_name_is_ignored_not_a_crash(make_app):
    app = make_app()
    app.set_model_route("HOT", tab="no-such-tab")
    page = app.page_container.get_page("model")
    page._reload = lambda **kw: None
    app.show_page("model")                       # must not raise
    del page._reload
    assert page._tabs.get() == "Drift metrics"


def test_the_seven_tabs_are_sentence_case_in_their_established_order(make_app):
    """Facelift step 2 Task 3, design doc §1 item 6: sentence case, same order as before --
    only the multi-word Title Case names change ("Units"/"History"/"Findings" are already
    one word, so sentence case leaves them alone). _name_list is CTkTabview's own record of
    tab names in the order add() was called, which is the order the segmented button shows
    them in."""
    app = make_app()
    app.set_model_route("HOT")
    page = app.page_container.get_page("model")
    page._reload = lambda **kw: None
    app.show_page("model")
    del page._reload
    assert page._tabs._name_list == [
        "Drift metrics", "Smoothness", "Units", "Final test units",
        "Trim vs final test", "History", "Findings"]
    # Step 1(b): the rename must not disturb the Findings-tab route (consume_model_tab ->
    # _select_tab) -- already pinned by test_showing_a_model_selects_the_requested_tab
    # above ("Findings" is one word, untouched by sentence-casing), re-asserted here so this
    # test alone documents both halves of the step 1(b) requirement in one place.
    app.set_model_route("HOT", tab="findings")
    page._reload = lambda **kw: None
    page.on_show()
    del page._reload
    assert page._tabs.get() == "Findings"


def test_track_metric_columns_public_and_linearity_maps_to_shifted():
    from laser_trim_analyzer.ml.drift_training import TRACK_METRIC_COLUMNS
    from laser_trim_analyzer.database.models import TrackResult as DBTR
    # Q4: the detector trains linearity_error on final_linearity_error_shifted; the UI must match.
    assert TRACK_METRIC_COLUMNS["linearity_error"] is DBTR.final_linearity_error_shifted
    assert "max_smoothness_value" not in TRACK_METRIC_COLUMNS  # lives on SmoothnessResult


# ---- Task 2: ThemedTabView ------------------------------------------------

def test_themed_tab_view(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.tab_view import ThemedTabView
    tv = ThemedTabView(tk_root, theme=ThemeManager())
    assert tv.add("Drift metrics") is not None
    tv.add("Units"); tv.set("Units")
    assert tv.get() == "Units"


# ---- Task 3: MetricPillRow ------------------------------------------------

def _status(model="M1", **tiers):
    """Build a ModelDriftStatus; pass metric=Tier kwargs to override specific metrics."""
    from datetime import datetime
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, MetricStatus, ModelDriftStatus, WATCHED_METRICS)
    per = {}
    for m in WATCHED_METRICS:
        tier = tiers.get(m, DriftTier.STABLE)
        per[m] = MetricStatus(metric=m, tier=tier, alert_type=None,
                              magnitude=0.0 if tier == DriftTier.STABLE else 3.1,
                              baseline_mean=0.01, baseline_std=0.001,
                              recent_mean=0.012, recent_count=5, is_trained=True)
    return ModelDriftStatus(model=model, overall_tier=DriftTier.STABLE, worst_metric=None,
                            worst_alert_type=None, per_metric=per, last_processed=datetime.now())


def test_pill_row_has_eight_pills(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
    from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS
    row = MetricPillRow(tk_root, theme=ThemeManager(), on_pill_click=lambda _: None)
    row.set_status(_status())
    assert set(row._pills) == set(WATCHED_METRICS)


def test_pill_shows_readable_label(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
    row = MetricPillRow(tk_root, theme=ThemeManager(), on_pill_click=lambda _: None)
    row.set_status(_status())
    assert row._pills["untrimmed_resistance"]._name_label.cget("text") == "Untrimmed resistance"


def test_pill_click_and_select(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.metric_pill_row import MetricPillRow
    got = []
    row = MetricPillRow(tk_root, theme=ThemeManager(), on_pill_click=got.append)
    row.set_status(_status())
    # DEVIATION: sigma_gradient is no longer a watched metric (replaced by
    # untrimmed_sigma_gradient + composite_trim_risk_score). Use a real pill key.
    row._pills["untrimmed_sigma_gradient"]._on_click()
    assert got == ["untrimmed_sigma_gradient"]
    row.set_selected("linearity_error")
    assert row._selected_metric == "linearity_error"


# ---- Task 4: FocusChart ---------------------------------------------------

def test_focus_chart_set_series_no_crash(tk_root):
    from datetime import datetime, timedelta
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
    chart = FocusChart(tk_root, theme=ThemeManager())
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(20, 0, -1)]
    values = [0.01 + 0.0001 * i for i in range(20)]
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                     baseline_mean=0.011, baseline_std=0.0005)


def test_focus_chart_empty_no_crash(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
    chart = FocusChart(tk_root, theme=ThemeManager())
    chart.set_series(metric="linearity_error", dates=[], values=[])  # empty state, no raise


# ---- Facelift step 2 Task 3b (2026-09-24) ---------------------------------
# Chart QA sweep + two reviews found defects on the Investigate page's charts
# and the unit chart it opens, none of them Task 3's. James then looked at
# qa_output/focus_6607_linearity_error.png ("that chart looks horrible" --
# 487 individual off-scale dots crowning the ceiling) and asked for the
# per-UNIT view to be redesigned in the same pass. Round 2 (same task,
# controller + James on the round-1 renders): the per-CALENDAR-DAY median
# still zigzagged on a sparse day, reading as a solid wall; the legend named
# limit lines that were not on screen when the band went off-scale; the
# "control limits off-scale" notice drew on top of the ceiling markers. No
# legend box at all now -- see test_no_legend_box_is_drawn below.

def _scatter_window_extent(coll, ax, renderer):
    """A scatter PathCollection's own get_window_extent() can come back an
    all-inf Bbox (nothing in its `_offsets` path ever sets a datalim for a
    collection built this way) -- build a real one from its transformed
    offsets plus the marker's own rendered half-size instead.

    Round 3: the off-scale markers no longer plot in data space at all (a
    BLENDED transform -- real dates for X, a points-offset axes-fraction for
    Y, so the whole marker draws regardless of the y-window's data scale) --
    `coll.get_offset_transform()` is what actually turns its raw offsets
    into display coordinates, which is NOT always `ax.transData`."""
    from matplotlib.transforms import Bbox
    offsets = coll.get_offsets()
    disp = coll.get_offset_transform().transform(offsets)
    sizes = coll.get_sizes()
    size = sizes[0] if len(sizes) else 0.0
    radius_px = (size ** 0.5) / 2.0 * (renderer.dpi / 72.0)
    xs, ys = disp[:, 0], disp[:, 1]
    return Bbox([[xs.min() - radius_px, ys.min() - radius_px],
                [xs.max() + radius_px, ys.max() + radius_px]])


def _offscale_marker_collections(ax):
    """The off-scale ceiling/floor markers, found by SIZE (there is no
    `label=` on anything any more -- round 2 dropped the legend that would
    have consumed it). `_OFFSCALE_MARKER_S` is the one true size, shared with
    the widget itself, so this can never quietly drift from what it draws."""
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import _OFFSCALE_MARKER_S
    out = []
    for c in ax.collections:
        sizes = c.get_sizes()
        if len(sizes) and sizes[0] == _OFFSCALE_MARKER_S:
            out.append(c)
    return out


def test_no_legend_box_is_drawn(tk_root):
    """Round 2: the old loc="best" legend sat on top of the data as often as
    not, and kept naming limit lines that were not even on screen once the
    band went off-scale (design doc / James, 2026-09-24 round 2). Replaced by
    the left-aligned-title + key-line header below -- there is no legend
    artist at all any more, on a chart shaped to have needed one under the
    old design (baseline, +-3sigma, an off-scale excursion)."""
    from datetime import datetime, timedelta
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    chart = FocusChart(tk_root, theme=ThemeManager())
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(20, 0, -1)]
    values = [0.0119 - 0.0001 * i for i in range(20)]
    values[-1] = 5.0
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                     baseline_mean=0.011, baseline_std=0.0005)
    assert chart._ax.get_legend() is None


def test_off_scale_note_never_touches_an_off_scale_marker(tk_root):
    """Task 3 gave the note its own band above the plot; on real data
    (focus_6607_linearity_error.png, 2023-2024) that band's bottom edge
    touched the row of off-scale markers drawn at the ceiling. The newest
    (rightmost) point here is the outlier, so its marker lands at the
    top-right -- exactly where the note also lives."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    chart = FocusChart(tk_root, theme=ThemeManager())
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(20, 0, -1)]     # oldest -> newest
    # A gentle trend keeps every OTHER point comfortably inside the
    # percentile-fit window, and only the newest (rightmost) point is the
    # outlier -- so its marker lands at the top-right, where the note also
    # lives, and nothing else registers as off-scale or "beyond limits" to
    # muddy the picture.
    values = [0.0119 - 0.0001 * i for i in range(20)]
    values[-1] = 5.0
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                     baseline_mean=0.011, baseline_std=0.0005)

    canvas = FigureCanvasAgg(chart._fig)
    canvas.draw()
    renderer = canvas.get_renderer()

    # "▲ " (not just "off-scale") to name only the marker-count note -- a
    # SEPARATE, unrelated note can also mention "off-scale" in its own
    # sentence (the baseline-spans-mixed-history disclosure) when a fixture
    # also happens to trigger it.
    notes = [t for t in chart._ax.texts if t.get_text().startswith("▲")]
    assert notes, "the outlier above must trigger the off-scale note"
    note_box = notes[0].get_window_extent(renderer)

    markers = _offscale_marker_collections(chart._ax)
    assert markers, "the outlier above must draw an aggregated off-scale marker"
    for coll in markers:
        marker_box = _scatter_window_extent(coll, chart._ax, renderer)
        assert not note_box.overlaps(marker_box), (
            f"note {note_box.bounds} overlaps the off-scale marker {marker_box.bounds}")


# ---- Round 3 (2026-09-24, controller + James on 189409c/9486393's renders) -

def test_off_scale_marker_renders_whole_not_clipped(tk_root):
    """Pinned exactly at the axis edge with clip_on=True (round 2), a marker
    was cut in half by the axes boundary. Inset by its own radius instead
    (round 3) -- confirmed here by checking the marker's own rendered bbox
    sits entirely INSIDE the axes' bbox, so clipping never has anything to
    do at all, rather than merely "doesn't touch the header text"."""
    from datetime import datetime, timedelta
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    chart = FocusChart(tk_root, theme=ThemeManager())
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(20, 0, -1)]
    values = [0.0119 - 0.0001 * i for i in range(20)]
    values[-1] = 5.0
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                     baseline_mean=0.011, baseline_std=0.0005)

    canvas = FigureCanvasAgg(chart._fig)
    canvas.draw()
    renderer = canvas.get_renderer()

    markers = _offscale_marker_collections(chart._ax)
    assert markers, "expected an aggregated off-scale marker"
    axes_box = chart._ax.get_window_extent(renderer)
    for coll in markers:
        marker_box = _scatter_window_extent(coll, chart._ax, renderer)
        assert marker_box.y0 >= axes_box.y0 - 0.5, (
            f"marker {marker_box.bounds} reaches below the axes {axes_box.bounds}")
        assert marker_box.y1 <= axes_box.y1 + 0.5, (
            f"marker {marker_box.bounds} reaches above the axes {axes_box.bounds}")


def test_rolling_median_draws_exactly_one_vertex_per_day(tk_root):
    """The line smeared vertically wherever a day carried many units --
    evaluating the rolling median at every UNIT let each one nudge the
    window's exact row membership, even on a shared day. A day with 20
    units must still contribute exactly one line vertex."""
    from datetime import datetime, timedelta

    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    today = datetime.now()
    busy_day = today - timedelta(days=10)
    dates, values = [], []
    for i in range(20):
        dates.append(busy_day)
        values.append(0.0100 + 0.0001 * (i % 5))
    for i in range(40, 0, -1):
        d = today - timedelta(days=i)
        if d == busy_day:
            continue
        dates.append(d)
        values.append(0.0102 + 0.0001 * (i % 5))

    chart = FocusChart(tk_root, theme=ThemeManager())
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                     baseline_mean=0.0105, baseline_std=0.0006)
    lines = [ln for ln in chart._ax.get_lines() if ln.get_linewidth() > 1.5]
    assert lines, "expected the rolling-median line to be drawn"
    xs = lines[0].get_xdata()
    busy_key = (busy_day.year, busy_day.month, busy_day.day)
    busy_count = sum(1 for x in xs if (x.year, x.month, x.day) == busy_key)
    assert busy_count == 1, f"expected exactly one vertex for the 20-unit day, got {busy_count}"


def test_rolling_median_needs_five_units_in_its_window(tk_root):
    """The line is a median of UNITS, and a median of four says little: fewer than
    five units in the trailing window draw no line at all. The once-a-day and the
    >30-day-gap rules each have a test that goes red without them; this rule had
    none (Task 3b review, 2026-09-24: set to 1, nothing failed)."""
    from datetime import datetime, timedelta

    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    today = datetime.now()

    def strong_line_vertices(n_units):
        dates = [today - timedelta(days=n_units - i) for i in range(n_units)]
        values = [0.0100 + 0.0001 * (i % 5) for i in range(n_units)]
        chart = FocusChart(tk_root, theme=ThemeManager())
        chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                         baseline_mean=0.0105, baseline_std=0.0006)
        lines = [ln for ln in chart._ax.get_lines() if ln.get_linewidth() > 1.5]
        return sum(len(ln.get_xdata()) for ln in lines)

    assert strong_line_vertices(4) == 0, "four units drew a median line"
    assert strong_line_vertices(6) >= 2, "six units in six days drew no median line"


def test_x_tick_labels_never_collide_at_12_months_or_15_years(tk_root):
    """"2026-022026-03" ran together at a 12-month window -- the implicit
    default formatter/locator packed ticks too densely for the available
    width. Checked at two window spans and two figure widths (the widget's
    own default, and a narrower one standing in for a cramped small-window
    layout -- the exact chart pixel width inside a 1280x720 app window
    depends on the rest of that page's layout, not reproduced here)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    for n_years, span_label in [(1, "12mo"), (15, "15yr")]:
        dates, values = _long_history_with_outliers(n_years=n_years)
        # (8, 3) is the widget's own default (FocusChart.__init__); (4.5,
        # 1.8) is deliberately cramped -- matplotlib's OWN default
        # locator/formatter (no explicit ConciseDateFormatter) collides at
        # this width on this fixture, confirmed empirically, which is what
        # makes this a real regression guard rather than a fixture that
        # happens to pass either way.
        for figsize in ((8, 3), (4.5, 1.8)):
            chart = FocusChart(tk_root, theme=ThemeManager())
            chart._fig.set_size_inches(*figsize)
            chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                             baseline_mean=0.011, baseline_std=0.0005)
            canvas = FigureCanvasAgg(chart._fig)
            canvas.draw()
            renderer = canvas.get_renderer()

            labels = [lbl for lbl in chart._ax.get_xticklabels() if lbl.get_text()]
            assert len(labels) >= 2, f"{span_label} @ {figsize}: expected multiple x ticks"
            boxes = [lbl.get_window_extent(renderer) for lbl in labels]
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    assert not boxes[i].overlaps(boxes[j]), (
                        f"{span_label} @ {figsize}: x tick labels "
                        f"{labels[i].get_text()!r} and {labels[j].get_text()!r} collide")


def test_y_tick_labels_resolve_to_plex_mono_once_loaded():
    """Numbers in Plex Mono (step 1's spec); the fonts review found no chart
    actually asked for it -- tick labels were Sans everywhere. Agg so
    get_fontname() reports the font matplotlib really resolved, not just the
    fallback list it was asked to try."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from laser_trim_analyzer.gui.v6 import font_loader
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
    from matplotlib.figure import Figure

    font_loader.load_bundled_fonts()

    theme = ThemeManager()
    chart = FocusChart.__new__(FocusChart)
    chart.theme = theme
    chart._fig = Figure(figsize=(8, 3), dpi=96, facecolor=theme.CARD)
    chart._ax = chart._fig.add_subplot(111)
    chart.canvas = FigureCanvasAgg(chart._fig)
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(10, 0, -1)]
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates,
                     values=[0.01 + 0.0001 * i for i in range(10)])

    chart.canvas.draw()
    labels = chart._ax.get_yticklabels()
    assert labels, "expected y tick labels to draw"
    assert all(lbl.get_fontname() == "IBM Plex Mono" for lbl in labels), (
        [lbl.get_fontname() for lbl in labels])


def _long_history_with_outliers(n_years=3, outlier_every=15):
    """~3 years of daily data on a trained baseline, with roughly 1 in
    `outlier_every` points forced off-scale -- the 6607 shape (hundreds of
    off-scale points spread over many months), built small enough to stay a
    fast unit test. `outlier_every` must keep the outlier fraction under the
    10th/90th-percentile y-window's own tail (10%) or the window widens to
    include them instead of clamping them off-scale -- 15 (~6.7%) has margin."""
    start = datetime.now() - timedelta(days=365 * n_years)
    dates, values = [], []
    d = start
    while d < datetime.now():
        dates.append(d)
        i = len(dates)
        values.append(50.0 if i % outlier_every == 0 else 0.01 + 0.0001 * (i % 7))
        d += timedelta(days=1)
    return dates, values


def test_units_view_opens_on_the_last_12_months_when_more_exists(tk_root):
    """James: "that chart looks horrible" on 6607's whole-history render.
    default_window_days is the Model page's opt-in for the Units toggle
    (model_page.py's _UNITS_VIEW_DEFAULT_DAYS) -- exercised directly here."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    dates, values = _long_history_with_outliers()
    chart = FocusChart(tk_root, theme=ThemeManager())
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                     baseline_mean=0.011, baseline_std=0.0005, default_window_days=366)

    import matplotlib.dates as mdates
    x0, x1 = chart._ax.get_xlim()
    span_days = (mdates.num2date(x1) - mdates.num2date(x0)).days
    # 366 days of data plus the function's own small x-axis padding (2% of
    # the span, or 1 day) -- comfortably under a year and a half, nowhere
    # near the ~3 years actually handed in.
    assert span_days <= 400, f"the view should open on ~12 months, not {span_days} days"


def test_units_view_shows_the_full_range_when_the_model_has_less(tk_root):
    """The other half of the same contract: a model with less than 12
    months of history is shown whole, not padded out or truncated."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(60, 0, -5)]     # ~60 days
    values = [0.01 + 0.0001 * i for i in range(len(dates))]
    chart = FocusChart(tk_root, theme=ThemeManager())
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                     default_window_days=366)

    import matplotlib.dates as mdates
    x0, x1 = chart._ax.get_xlim()
    span_days = (mdates.num2date(x1) - mdates.num2date(x0)).days
    assert span_days < 70, f"a 60-day history should not be widened: got {span_days} days"


def test_smoothness_tabs_chart_never_gets_the_units_view_default_window():
    """SmoothnessTab's embedded FocusChart states "the chart always sees
    every record" -- default_window_days is opt-in and None by default so
    that contract cannot be silently narrowed by this change."""
    import inspect
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
    sig = inspect.signature(FocusChart.set_series)
    assert sig.parameters["default_window_days"].default is None


def test_off_scale_markers_are_aggregated_to_at_most_one_per_month_shown(tk_root):
    """487 individual dots crowning the ceiling (6607) read as a solid bar,
    not as data. Aggregated to one marker per (calendar month, edge) that
    has any -- with every off-scale point forced ABOVE here, marker count
    must not exceed the number of distinct months actually drawn."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    dates, values = _long_history_with_outliers(n_years=2)
    chart = FocusChart(tk_root, theme=ThemeManager())
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                     baseline_mean=0.011, baseline_std=0.0005, default_window_days=366)

    x0, x1 = chart._ax.get_xlim()
    import matplotlib.dates as mdates
    shown = [d for d in dates if mdates.date2num(d) >= x0]
    months_shown = len({(d.year, d.month) for d in shown})

    markers = _offscale_marker_collections(chart._ax)
    n_offscale_markers = sum(len(c.get_offsets()) for c in markers)
    assert n_offscale_markers >= 1, "the fixture must actually produce off-scale markers"
    assert n_offscale_markers <= months_shown, (
        f"{n_offscale_markers} off-scale markers for only {months_shown} months shown")


def test_off_scale_note_states_the_true_total(tk_root):
    """The note keeps the total even though the markers no longer do --
    "▲ N off-scale ... the chart" must name every off-scale point in the
    current window, not just the (now aggregated) markers drawing it."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    dates, values = _long_history_with_outliers(n_years=1)
    n_expected_offscale = sum(1 for v in values if v == 50.0)
    chart = FocusChart(tk_root, theme=ThemeManager())
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                     baseline_mean=0.011, baseline_std=0.0005)

    notes = [t for t in chart._ax.texts if "off-scale" in t.get_text()]
    assert notes, "expected an off-scale note"
    assert str(n_expected_offscale) in notes[0].get_text(), notes[0].get_text()


def test_no_text_artist_intersects_the_axes_data_area(tk_root):
    """Round 2's whole point: nothing drawn above the axes (title, key line,
    off-scale note, the "control limits off-scale" disclosure) may ever be
    drawn INSIDE it -- the round-2 bug on 8340-1 was exactly that notice
    sitting on top of the ceiling markers. One fixture exercises all four at
    once (a trained baseline whose band is off-scale, AND an excursion beyond
    the visible window)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    dates, values = _long_history_with_outliers(n_years=1)
    chart = FocusChart(tk_root, theme=ThemeManager())
    # baseline_std huge relative to the visible data -> triggers the
    # "control limits off-scale" disclosure on its own line, ABOVE the key
    # line -- the exact stack that has to clear the axes twice over.
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                     baseline_mean=0.011, baseline_std=1.3)

    canvas = FigureCanvasAgg(chart._fig)
    canvas.draw()
    renderer = canvas.get_renderer()

    axes_box = chart._ax.get_window_extent(renderer)
    assert chart._ax.texts, "expected at least the key line to be drawn"
    for txt in chart._ax.texts:
        text_box = txt.get_window_extent(renderer)
        assert not text_box.overlaps(axes_box), (
            f"text {txt.get_text()!r} {text_box.bounds} overlaps "
            f"the axes data area {axes_box.bounds}")


def test_rolling_median_is_far_smoother_than_a_daily_median_would_be(tk_root):
    """Round 2, James: the old per-CALENDAR-DAY median zigzagged hard on a
    sparse day, reading as a solid wall over 12 months. A day-alternating
    high/low series is the worst case for a daily median -- it swings the
    full amplitude EVERY SINGLE DAY. "Point-to-point variance" (the
    coordinator's own words) means the variance of consecutive DIFFERENCES,
    not of the raw values -- this fixture's balanced 30-day windows converge
    to a fairly constant median, so the raw values are not very spread out
    once past the first month, but the daily median's point-to-point swing
    is exactly the zigzag being fixed, and that is what has to shrink."""
    from datetime import datetime, timedelta
    import statistics

    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    today = datetime.now()
    dates, values, daily_medians = [], [], []
    d = today - timedelta(days=119)
    while d <= today:
        day_vals = [0.05, 0.06] if (d.toordinal() % 2 == 0) else [0.25, 0.26]
        for v in day_vals:
            dates.append(d); values.append(v)
        daily_medians.append(statistics.median(day_vals))
        d += timedelta(days=1)
    daily_diffs = [b - a for a, b in zip(daily_medians, daily_medians[1:])]
    daily_diff_variance = statistics.pvariance(daily_diffs)

    chart = FocusChart(tk_root, theme=ThemeManager())
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                     baseline_mean=0.15, baseline_std=0.2)
    lines = [ln for ln in chart._ax.get_lines() if ln.get_linewidth() > 1.5]
    assert lines, "expected the rolling-median line to be drawn"
    drawn = [y for y in lines[0].get_ydata() if y == y]     # drop NaN breaks
    assert len(drawn) > 10
    drawn_diffs = [b - a for a, b in zip(drawn, drawn[1:])]
    drawn_diff_variance = statistics.pvariance(drawn_diffs)
    # A 30-day window sliding one day at a time still has a little residual
    # "boundary flutter" on a perfectly alternating adversarial signal like
    # this one (each step swaps ~2 of ~60 points in the window) -- measured
    # at ~5x smaller than the daily median's, a wide margin short of the
    # daily median's full every-step swing.
    assert drawn_diff_variance < daily_diff_variance * 0.25, (
        f"rolling median's point-to-point variance {drawn_diff_variance:.5f} is not far "
        f"below the daily median's {daily_diff_variance:.5f}")


def test_rolling_median_never_spans_a_gap_over_30_days(tk_root):
    """A resumed model after an idle stretch gets a break, never a straight
    line to its first point back (2016 -> 2023 on 6607 must read empty).

    The gap-break threshold is a FIXED 30 days, independent of which rolling
    window size is active (30D under 18 months of total span, 90D at or
    above) -- so the fixture that actually pins the explicit check needs a
    gap BIGGER than 30 days but SMALLER than the 90-day window a >18-month
    span switches to: only then does min_periods=5 alone fail to produce a
    break on its own (a 90-day window bridges a 45-day gap with real points
    on both sides), leaving the explicit >30-day check as the only thing
    that still splits the line. ~600 days of otherwise-dense data with one
    45-day gap in the middle."""
    from datetime import datetime, timedelta

    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    today = datetime.now()
    start = today - timedelta(days=600)
    dates, values = [], []
    d = start
    i = 0
    while d < today:
        if not (250 <= (d - start).days < 295):    # the 45-day gap
            dates.append(d); values.append(0.0100 + 0.0001 * (i % 5)); i += 1
        d += timedelta(days=1)

    chart = FocusChart(tk_root, theme=ThemeManager())
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                     baseline_mean=0.0105, baseline_std=0.0003)
    lines = [ln for ln in chart._ax.get_lines() if ln.get_linewidth() > 1.5]
    assert lines, "expected the rolling-median line to be drawn"
    # get_xdata() hands back the pandas Timestamps the line was plotted
    # with directly (matplotlib's datetime unit converter stores the
    # original objects, not its internal float date-nums), so day
    # differences come straight from Timestamp subtraction -- no
    # mdates.num2date round trip, which chokes on a Timestamp input.
    xs, ys = lines[0].get_xdata(), lines[0].get_ydata()
    segments = []
    seg = []
    for x, y in zip(xs, ys):
        if y != y:                                    # NaN -- a break
            if seg:
                segments.append(seg); seg = []
            continue
        seg.append(x)
    if seg:
        segments.append(seg)
    assert len(segments) >= 2, "expected the 45-day gap to split the line into >= 2 segments"
    # A segment itself may legitimately span many days (this fixture's two
    # dense runs are ~245 and ~305 days each) -- what must never happen is
    # a single STEP between two CONSECUTIVE plotted points wider than 30
    # days, which is exactly what a break across the 45-day gap prevents.
    for seg in segments:
        for a, b in zip(seg, seg[1:]):
            step_days = (b - a).days
            assert step_days <= 30, f"a single line segment steps {step_days} days"


def test_key_names_only_what_is_actually_drawn(tk_root):
    """A chart whose +-3sigma band is off-scale (so the dotted line is not
    meaningfully on screen) must not claim it is in the key -- the same rule
    that already keeps the median out of the key when there are too few
    units to compute one."""
    from datetime import datetime, timedelta

    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart

    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(60, 0, -1)]
    values = [0.05 + 0.001 * (i % 5) for i in range(len(dates))]

    chart = FocusChart(tk_root, theme=ThemeManager())
    # A wildly oversized baseline_std forces limits_off_scale.
    chart.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                     baseline_mean=0.05, baseline_std=1.3)
    keys = [t.get_text() for t in chart._ax.texts if t.get_text().startswith("━")
            or t.get_text().startswith("·")]
    assert keys, "expected a key line"
    assert "±3σ control limit" not in keys[0], keys[0]

    # The complementary case: a normal (in-scale) band DOES name it. This
    # fixture's own values span only ~0.05-0.054, so the std has to be small
    # too, or 7*std alone exceeds the "off-scale" test's own 6x-data-span
    # trigger and this "in-scale" case would (wrongly) go off-scale as well.
    chart2 = FocusChart(tk_root, theme=ThemeManager())
    chart2.set_series(metric="untrimmed_sigma_gradient", dates=dates, values=values,
                      baseline_mean=0.05, baseline_std=0.001)
    keys2 = [t.get_text() for t in chart2._ax.texts if t.get_text().startswith("━")
             or t.get_text().startswith("·")]
    assert keys2 and "±3σ control limit" in keys2[0], keys2


# ---- Task 5: DriftMetricsTab ----------------------------------------------

def test_drift_tab_row_per_metric_and_click(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import DriftMetricsTab
    from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS
    got = []
    tab = DriftMetricsTab(tk_root, theme=ThemeManager(), on_metric_select=got.append)
    tab.set_status(_status())
    assert set(tab._rows) == set(WATCHED_METRICS)
    # DEVIATION: untrimmed_sigma_gradient (sigma_gradient no longer watched).
    tab._rows["untrimmed_sigma_gradient"]._on_click()
    assert got == ["untrimmed_sigma_gradient"]


# ---- Task 6: SmoothnessTab ------------------------------------------------

def test_smoothness_tab_empty(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import SmoothnessTab
    SmoothnessTab(tk_root, theme=ThemeManager()).set_records([])  # no raise


def test_smoothness_tab_with_records(tk_root):
    from datetime import datetime
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import SmoothnessTab
    tab = SmoothnessTab(tk_root, theme=ThemeManager())
    tab.set_records([{"serial": "sn1", "file_date": datetime.now(),
                      "max_smoothness_value": 0.4, "avg_smoothness_value": 0.2}])
    assert len(tab._rows) == 1


# ---- Task 7: UnitsTab + UnitChartModal -----------------------------------

def test_units_tab_row_per_unit_keeps_duplicate_serials(tk_root):
    from datetime import datetime
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab
    tab = UnitsTab(tk_root, theme=ThemeManager(), on_unit_click=lambda u: None, on_export=lambda: None)
    # Same serial twice = two valid trims (Q2). Both rows must appear.
    units = [{"analysis_id": 1, "serial": "sn1", "file_date": datetime.now(),
              "overall_status": "Pass", "sigma_gradient": 0.01, "linearity_error": 0.004},
             {"analysis_id": 2, "serial": "sn1", "file_date": datetime.now(),
              "overall_status": "Fail", "sigma_gradient": 0.02, "linearity_error": 0.05}]
    tab.set_units(units)
    assert len(tab._rows) == 2


# ---- Row-render budget (2026-09-02 perf fix) -------------------------------
# Building one CTk row-frame per record blocked the Tk thread for 0.5-2.1 s on a
# model switch (measured: 200 units 545 ms, 500 FT rows 1419 ms, FT re-set
# 2105 ms). The tabs now render INITIAL_ROWS and offer "show all" — the same
# pattern FocusListZone uses, collapse-on-new-data included.
#
# These are STRUCTURAL assertions on purpose: a timing assertion would flake on
# CI while a row count cannot.

def _bench_units(n):
    from datetime import datetime, timedelta
    return [{"analysis_id": i, "serial": f"SN{i:05d}",
             "file_date": datetime(2026, 1, 1) + timedelta(days=i % 90),
             "overall_status": "Fail" if i % 3 == 0 else "Pass",
             "sigma_gradient": 0.01 + i * 1e-5,
             "linearity_error": 0.004 + i * 1e-5} for i in range(n)]


def _bench_ft(n):
    from datetime import datetime, timedelta
    return [{"id": i, "serial": f"SN{i:05d}",
             "file_date": datetime(2026, 1, 1) + timedelta(days=i % 90),
             "result": "FAIL" if i % 4 == 0 else "PASS",
             "linked": bool(i % 2), "match": 90 + (i % 10)} for i in range(n)]


def _bench_smooth(n):
    from datetime import datetime, timedelta
    return [{"serial": f"SN{i:05d}",
             "file_date": datetime(2026, 1, 1) + timedelta(days=i % 90),
             "max_smoothness_value": 0.4 + i * 1e-4, "smoothness_spec": 1.0,
             "smoothness_pass": bool(i % 3)} for i in range(n)]


def _units_tab(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab
    return UnitsTab(tk_root, theme=ThemeManager(), on_unit_click=lambda u: None,
                    on_export=lambda: None)


def test_units_tab_renders_only_the_budget_and_offers_the_real_total(tk_root):
    from laser_trim_analyzer.gui.v6.widgets.units_tab import INITIAL_ROWS
    tab = _units_tab(tk_root)
    tab.set_units(_bench_units(200))
    assert len(tab._rows) == INITIAL_ROWS == 50
    # The control is packed, and its label names the REAL total — not the
    # budget. "Show all" over an unstated count is how a cap hides data.
    assert tab._show_all_btn.winfo_manager() == "pack"
    assert "Show all 200" in tab._show_all_btn.cget("text")


def test_units_tab_show_all_builds_the_remainder(tk_root):
    tab = _units_tab(tk_root)
    tab.set_units(_bench_units(200))
    tab._show_all_btn.invoke()
    assert len(tab._rows) == 200


def test_units_tab_collapses_back_to_the_budget_on_new_data(tk_root):
    """FocusListZone's rule: "show all" describes a list that no longer exists."""
    from laser_trim_analyzer.gui.v6.widgets.units_tab import INITIAL_ROWS
    tab = _units_tab(tk_root)
    tab.set_units(_bench_units(200))
    tab._show_all_btn.invoke()
    assert len(tab._rows) == 200
    tab.set_units(_bench_units(200))          # new model selected
    assert tab._expanded is False
    assert len(tab._rows) == INITIAL_ROWS


def test_units_tab_show_all_rows_click_through_identically(tk_root):
    """A row built by "show all" opens the same modal as a first-50 row."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab
    seen = []
    tab = UnitsTab(tk_root, theme=ThemeManager(), on_unit_click=seen.append,
                   on_export=lambda: None)
    tab.set_units(_bench_units(200))
    tab._show_all_btn.invoke()
    tab._rows[150]._on_click()                # a row that only show-all built
    assert len(seen) == 1 and seen[0] is tab._rows[150].unit


def test_units_tab_under_the_budget_has_no_show_all_control(tk_root):
    tab = _units_tab(tk_root)
    tab.set_units(_bench_units(12))
    assert len(tab._rows) == 12
    assert tab._show_all_btn.winfo_manager() == ""


def test_row_budget_never_shrinks_an_export(tk_root):
    """The budget caps DRAWING. An export that quietly dropped the 150 unbuilt
    rows would be data loss dressed up as a performance fix."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.ft_units_tab import FtUnitsTab
    tab = _units_tab(tk_root)
    tab.set_units(_bench_units(200))
    assert len(tab._rows) == 50 and len(tab.get_selected_units()) == 200
    ft = FtUnitsTab(tk_root, theme=ThemeManager())
    ft.set_units(_bench_ft(500))
    assert len(ft._rows) == 50 and len(ft.get_selected_units()) == 500


def test_units_tab_sorting_keeps_the_expanded_view(tk_root):
    """Sorting reorders the same list, so it must not silently re-collapse."""
    tab = _units_tab(tk_root)
    tab.set_units(_bench_units(200))
    tab._show_all_btn.invoke()
    tab._sort_by("serial")
    assert len(tab._rows) == 200


def test_ft_units_tab_budget_show_all_and_collapse(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.ft_units_tab import FtUnitsTab
    from laser_trim_analyzer.gui.v6.widgets.units_tab import INITIAL_ROWS
    seen = []
    tab = FtUnitsTab(tk_root, theme=ThemeManager(), on_unit_click=seen.append)
    tab.set_units(_bench_ft(500))
    assert len(tab._rows) == INITIAL_ROWS
    assert "Show all 500" in tab._show_all_btn.cget("text")
    tab._show_all_btn.invoke()
    assert len(tab._rows) == 500
    tab.set_units(_bench_ft(500))
    assert len(tab._rows) == INITIAL_ROWS
    # Selection survives the refactor: checking a row still scopes the export.
    tab._toggle_select({"id": 3}, True)
    assert tab.get_selected_units() == [u for u in tab._units if u["id"] == 3]


def test_ft_units_tab_empty_state_has_no_show_all_control(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.ft_units_tab import FtUnitsTab
    tab = FtUnitsTab(tk_root, theme=ThemeManager())
    tab.set_units([])
    assert tab._show_all_btn.winfo_manager() == ""


def test_smoothness_tab_budget_show_all_and_collapse(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import SmoothnessTab
    from laser_trim_analyzer.gui.v6.widgets.units_tab import INITIAL_ROWS
    tab = SmoothnessTab(tk_root, theme=ThemeManager())
    tab.set_records(_bench_smooth(200))
    assert len(tab._rows) == INITIAL_ROWS
    assert "Show all 200" in tab._show_all_btn.cget("text")
    tab._show_all_btn.invoke()
    assert len(tab._rows) == 200
    tab.set_records(_bench_smooth(200))
    assert len(tab._rows) == INITIAL_ROWS


def test_unit_chart_modal_marks_fail_points(tk_root):
    """Q1: every point shown; out-of-limit points become fail_points."""
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import compute_fail_points
    err = [0.0, 0.5, -0.2, 0.9]
    upper = [0.4, 0.4, 0.4, 0.4]; lower = [-0.4, -0.4, -0.4, -0.4]
    assert compute_fail_points(err, upper, lower) == [1, 3]  # 0.5>0.4 and 0.9>0.4


def test_unit_chart_modal_ft_toggle_disabled_with_a_reason(tk_root, tmp_path):
    """V6 unit chart gains the trim-vs-FT overlay (V5 Compare's last unique
    feature). With no linked final test the toggle is DISABLED and says why —
    never an empty chart the user has to interpret."""
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType, SystemType)
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import UnitChartModal

    db = DatabaseManager(tmp_path / "ftmodal.db")
    when = datetime(2026, 3, 1)
    with db.session() as s:
        ar = DBAR(filename="u.xls", file_path="/t/u", file_hash="u".ljust(64, "0"),
                  model="8340-1", serial="77", system=SystemType.A, file_date=when,
                  timestamp=when, overall_status=StatusType.PASS,
                  has_multi_tracks=False, processing_time=0.1)
        s.add(ar); s.flush()
        s.add(DBTR(analysis_id=ar.id, track_id="TRK1", status=StatusType.PASS,
                   position_data=[0.0, 1.0, 2.0], error_data=[0.0, 0.0, 0.0],
                   upper_limits=[0.02] * 3, lower_limits=[-0.02] * 3,
                   optimal_offset=0.0, linearity_pass=True, linearity_fail_points=0))
        s.commit()
        aid = ar.id

    m = UnitChartModal(tk_root, ThemeManager(), db,
                       {"analysis_id": aid, "serial": "77", "file_date": when})
    try:
        m._render_sync()      # deterministic: no worker thread in the test
        assert str(m._ft_toggle.cget("state")) == "disabled"
        assert "final test" in m._note_lbl.cget("text").lower()
    finally:
        m.destroy()


# ---- Task 8: PredictorPanel -----------------------------------------------

def test_predictor_panel_collapsed_then_toggles(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.predictor_panel import PredictorPanel
    p = PredictorPanel(tk_root, theme=ThemeManager(), load_fn=lambda model: "Risk: LOW (demo)")
    assert p._expanded is False
    p.set_model("8340-1")
    p.toggle()                 # expand → triggers lazy load
    assert p._expanded is True
    assert "Risk" in p._body_label.cget("text")


def test_predictor_panel_load_failure_is_graceful(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.predictor_panel import PredictorPanel
    def boom(model): raise RuntimeError("no predictor")
    p = PredictorPanel(tk_root, theme=ThemeManager(), load_fn=boom)
    p.set_model("X"); p.toggle()
    assert "No predictor" in p._body_label.cget("text")


# ---- Task 9: evidence export ----------------------------------------------

def test_build_summary_text_has_evidence_metrics():
    from laser_trim_analyzer.export.evidence import build_summary_text
    txt = build_summary_text("8340-1", _status())
    assert "8340-1" in txt
    # Q8: the three evidence metrics James hands engineers must be present, readable.
    for label in ("Untrimmed resistance", "Linearity error", "Electrical angle"):
        assert label in txt


def test_export_evidence_pack_writes_xlsx(tmp_path):
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    from laser_trim_analyzer.export.evidence import export_evidence_pack
    db = DatabaseManager(tmp_path / "ev.db")
    with db.session() as s:
        ar = DBAR(filename="x.xls", file_path="/f/x.xls", file_hash="hx", model="8340-1", serial="sn1",
                  system=SystemType.A, file_date=datetime.now(), timestamp=datetime.now(),
                  overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1)
        s.add(ar); s.flush()
        # TrackResult.status is NOT NULL — set it on any committed row.
        s.add(DBTR(analysis_id=ar.id, track_id="TRK1", status=StatusType.PASS,
                   sigma_gradient=0.01, final_linearity_error_shifted=0.004))
        s.commit()
    out = tmp_path / "pack.xlsx"
    export_evidence_pack(db, "8340-1", out, window_days=365)
    assert out.exists() and out.stat().st_size > 0


# ---- Task 10: ModelPage ---------------------------------------------------

def test_model_page_consumes_route_on_show(make_app):
    from datetime import datetime
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, SystemType, StatusType
    app = make_app()
    with app.db.session() as s:
        s.add(DBAR(filename="x.xls", file_path="/f/x.xls", file_hash="hx", model="ROUTED-MODEL",
                   serial="sn1", system=SystemType.A, file_date=datetime.now(), timestamp=datetime.now(),
                   overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1))
        s.commit()
    app.set_model_route("ROUTED-MODEL", "linearity_error")
    app.show_page("model")
    page = app.page_container.get_page("model")
    assert page._current_model == "ROUTED-MODEL"
    assert page._current_metric == "linearity_error"


def test_model_page_empty_state_when_no_model(make_app):
    app = make_app()
    app.show_page("model")        # no route set
    page = app.page_container.get_page("model")
    assert page._current_model is None
    assert page._empty_label.winfo_ismapped() or page._empty_label.winfo_exists()


def test_model_page_focus_series_uses_shifted_linearity(make_app):
    """Q4: requesting linearity_error reads final_linearity_error_shifted."""
    from datetime import datetime
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    app = make_app()
    with app.db.session() as s:
        ar = DBAR(filename="x.xls", file_path="/f/x.xls", file_hash="hx", model="QM", serial="sn1",
                  system=SystemType.A, file_date=datetime.now(), timestamp=datetime.now(),
                  overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1)
        s.add(ar); s.flush()
        # TrackResult.status is NOT NULL — set it on any committed row.
        s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                   final_linearity_error_shifted=0.0042))
        s.commit()
    page = app.page_container.get_page("model")
    dates, values, baseline = page._load_focus_series("QM", "linearity_error")
    assert values == [0.0042]


def test_model_page_banners_a_trim_vs_ft_spec_mismatch(make_app):
    """James (2026-08-30): "i also want to know when the trim and test specs
    dont align." The banner has to say WHAT differs and WHY that matters to
    the numbers already on the page — not just that something is wrong."""
    from laser_trim_analyzer.core.spec_alignment import SpecComparison
    app = make_app()
    page = app.page_container.get_page("model")
    page._set_spec_banner(SpecComparison(
        status="differs", pct_positions_differing=1.0, matched_positions=200,
        trim_typ_band=0.03, ft_typ_band=0.10,
        note=("100% of the positions both stations measure are graded to "
              "different limits (trim ±0.030 V, final test ±0.100 V)")))
    assert page._spec_banner.cget("text") == (
        "⚠ 100% of the positions both stations measure are graded to "
        "different limits (trim ±0.030 V, final test ±0.100 V) — "
        "cross-station numbers (escapes, Gap) compare different "
        "requirements at those positions.")
    assert page._spec_banner.winfo_manager() == "pack"
    # It qualifies the page's verdict (now the caption, not a body label), so it must
    # stay pinned directly above "Worth changing" -- pack() would otherwise re-append
    # it at the bottom of the body when re-shown.
    slaves = page._spec_banner.master.pack_slaves()
    assert slaves.index(page._spec_banner) == slaves.index(page._worth_section) - 1


def test_model_page_says_nothing_when_the_stations_agree_or_are_unknown(make_app):
    """Amber on an unanswered question is how a warning stops being believed."""
    from laser_trim_analyzer.core.spec_alignment import SpecComparison
    app = make_app()
    page = app.page_container.get_page("model")
    page._set_spec_banner(SpecComparison(
        status="differs", pct_positions_differing=1.0, matched_positions=200,
        trim_typ_band=0.03, ft_typ_band=0.10, note="n"))
    for quiet in (SpecComparison("aligned", 0.0, 200, 0.03, 0.03, "same"),
                  SpecComparison("insufficient", 0.0, 0, None, None, "no data"),
                  None):                       # loader failed — say nothing
        page._set_spec_banner(quiet)
        assert page._spec_banner.winfo_manager() == ""


# ---- Dashboard-round Model fixes ------------------------------------------

def test_resolve_focus_metric_prefers_worst_when_not_user_picked():
    from laser_trim_analyzer.gui.v6.pages.model_page import ModelPage
    from laser_trim_analyzer.ml.drift_types import DriftTier, ModelDriftStatus
    status = ModelDriftStatus(model="M", overall_tier=DriftTier.OUT_OF_CONTROL,
                              worst_metric="trim_pass_count", worst_alert_type=None, per_metric={})
    # not user-picked -> worst metric wins
    assert ModelPage._resolve_focus_metric(status, False, "untrimmed_sigma_gradient") == "trim_pass_count"
    # user picked -> keep their choice
    assert ModelPage._resolve_focus_metric(status, True, "linearity_error") == "linearity_error"
    # no worst (all stable) -> keep current fallback
    stable = ModelDriftStatus(model="M", overall_tier=DriftTier.STABLE, worst_metric=None,
                              worst_alert_type=None, per_metric={})
    assert ModelPage._resolve_focus_metric(stable, False, "untrimmed_sigma_gradient") == "untrimmed_sigma_gradient"


def test_model_recent_means_computed_from_data(make_app):
    """Recent column = the last CLOSED lot's median (lot mode, 2026-07-13).
    Units dated today form an OPEN lot and must NOT be the recent value —
    so the fixture backdates them past the changeover gap."""
    from datetime import datetime, timedelta
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    app = make_app()
    when = datetime.now() - timedelta(days=10)   # closed lot (gap > 3 days)
    with app.db.session() as s:
        for i, val in enumerate((0.0040, 0.0044)):
            ar = DBAR(filename=f"r{i}.xls", file_path="/f/x.xls", file_hash=f"hr{i}",
                      model="RM", serial=f"sn{i}", system=SystemType.A, file_date=when,
                      timestamp=when, overall_status=StatusType.PASS,
                      has_multi_tracks=False, processing_time=0.1)
            s.add(ar); s.flush()
            s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                       final_linearity_error_shifted=val))
        s.commit()
    page = app.page_container.get_page("model")
    means = page._recent_means("RM")
    assert means["linearity_error"] == pytest.approx(0.0042)   # lot median


def test_drift_tab_uses_grid_columns(tk_root):
    """Header and rows lay their cells out on a shared grid, so columns line up."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import DriftMetricsTab, _COLUMNS
    tab = DriftMetricsTab(tk_root, theme=ThemeManager(), on_metric_select=lambda _: None)
    tab.set_status(_status())
    row = tab._rows["linearity_error"]
    slaves = row.grid_slaves()
    assert len(slaves) == len(_COLUMNS)                                  # cells gridded, not packed
    assert sorted(int(w.grid_info()["column"]) for w in slaves) == list(range(len(_COLUMNS)))


# ---- FOCUS/SPC redesign (2026-08-29): the lot chart is the headline view ---
# Production runs in LOTS, so a lot — not a unit — is what goes in or out of
# control. The Model page opens on the SPC lot chart for every metric; the
# per-unit scatter that used to be the only view is one click away.
# Seeding mirrors tests/test_spc_db.py (real DatabaseManager, real compute).

D0 = datetime(2026, 1, 5)      # SPC anchors on the DATA's newest date, not "now"


def _add_lot(db, model, day, n, fails):
    """One production lot: n gradeable units on one file_date, `fails` of them FAIL."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, StatusType, SystemType)
    with db.session() as s:
        for i in range(n):
            # `system` is NOT NULL in the schema; the SPC query never reads it.
            s.add(DBAR(model=model, serial=f"{model}-{day:%m%d}-{i}", system=SystemType.A,
                       filename=f"{model}_{i}_{day:%m-%d-%Y}.xls", file_date=day,
                       overall_status=StatusType.FAIL if i < fails else StatusType.PASS))


def _seed(db, model, n_lots=12, fails_last=0, start=D0, n_per=20, base_fails=2):
    """Weekly lots at a steady baseline rate, then one lot at `fails_last`."""
    for k in range(n_lots - 1):
        _add_lot(db, model, start + timedelta(days=7 * k), n_per, base_fails)
    last_day = start + timedelta(days=7 * (n_lots - 1))
    _add_lot(db, model, last_day, n_per, fails_last if fails_last else base_fails)
    return last_day


def _seed_tracks(db, model, metric, n_lots=12, n_per=3, start=D0):
    """Weekly lots carrying a CONTINUOUS metric, drifting lot-to-lot so the
    baseline has a real spread (a flat baseline is degenerate -> unjudged)."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType, SystemType)
    from laser_trim_analyzer.ml.drift_training import TRACK_METRIC_COLUMNS
    col = TRACK_METRIC_COLUMNS[metric].key
    with db.session() as s:
        for k in range(n_lots):
            day = start + timedelta(days=7 * k)
            for i in range(n_per):
                ar = DBAR(model=model, serial=f"{model}-{k}-{i}", system=SystemType.A,
                          filename=f"{model}_{k}_{i}.xls", file_date=day,
                          overall_status=StatusType.PASS)
                s.add(ar); s.flush()
                # TrackResult.status is NOT NULL — set it on any committed row.
                s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                           **{col: 0.010 + 0.0004 * k + 0.0001 * i}))
        s.commit()


def _spc_app(make_app, monkeypatch, metric="linearity_fail_fraction", seed=None):
    """App routed to a drifting model, with the HEADLINE chart's draws recorded.

    The spies go on the page's own FocusChart INSTANCE, not the class: the
    Smoothness tab embeds a second FocusChart and draws into it on the same
    reload, so a class-level patch would mix the two charts' calls together.
    """
    calls = []
    app = make_app()
    (seed or (lambda db: _seed(db, "HOT", fails_last=12)))(app.db)
    app.set_model_route("HOT", metric)
    page = app.page_container.get_page("model")
    # on_show kicks off a BACKGROUND reload. Letting it race the synchronous
    # one below adds no coverage and makes the two threads fight over the DB's
    # single StaticPool connection (one RLock held for a whole session) — worth
    # ~2.5 min per run of this file. Suppress it; load once, synchronously.
    page._reload = lambda **kw: None
    app.show_page("model")                 # the real deep-link path
    del page._reload                       # back to the real bound method
    monkeypatch.setattr(page._focus_chart, "set_spc_series",
                        lambda series, **kw: calls.append(("spc", series)))
    monkeypatch.setattr(page._focus_chart, "set_series",
                        lambda **kw: calls.append(("units", kw)))
    page.reload_now()                      # synchronous path for tests
    return app, page, calls


def test_model_page_opens_on_the_spc_lot_chart(make_app, monkeypatch):
    """The headline chart is the LOT p-chart for the metric that routed here."""
    app, page, calls = _spc_app(make_app, monkeypatch)
    assert page._chart_view == "lots"
    assert [kind for kind, _ in calls] == ["spc"]      # units view NOT drawn
    series = calls[0][1]
    assert series.model == "HOT" and series.metric == "linearity_fail_fraction"
    assert series.judged and series.points             # real limits, real lots
    assert series.points[-1].ooc                       # the blown-out last lot


def test_model_page_units_toggle_draws_the_unit_view(make_app, monkeypatch):
    app, page, calls = _spc_app(make_app, monkeypatch)
    calls.clear()
    page._on_chart_view_change("Units")
    assert page._chart_view == "units"
    assert [kind for kind, _ in calls] == ["units"]
    assert calls[0][1]["metric"] == "linearity_fail_fraction"
    # The Units view opens on the last 12 months (James: the all-history wall "looks
    # horrible"); dropping the kwarg at this call site would bring the wall back.
    assert calls[0][1].get("default_window_days") == 366
    calls.clear()
    page._on_chart_view_change("Lots · SPC")           # and back
    assert page._chart_view == "lots"
    assert [kind for kind, _ in calls] == ["spc"]


def test_chart_toggle_re_renders_loaded_data_without_touching_the_db(make_app, monkeypatch):
    """Both series load in ONE _reload pass; the toggle is a view switch."""
    import laser_trim_analyzer.gui.v6.pages.model_page as mp
    app, page, calls = _spc_app(make_app, monkeypatch)

    def boom(*a, **k):
        raise AssertionError("toggle must re-render loaded data, not re-query")
    monkeypatch.setattr(mp, "compute_spc_series", boom)
    monkeypatch.setattr(page, "_load_focus_series", boom)
    page._on_chart_view_change("Units")
    page._on_chart_view_change("Lots · SPC")


def test_continuous_metric_also_opens_on_the_lot_chart(make_app, monkeypatch):
    """Default is Lots for EVERY metric — compute_spc_series routes a
    continuous metric through build_continuous_series on its own."""
    app, page, calls = _spc_app(
        make_app, monkeypatch, metric="untrimmed_sigma_gradient",
        seed=lambda db: _seed_tracks(db, "HOT", "untrimmed_sigma_gradient"))
    assert [kind for kind, _ in calls] == ["spc"]
    series = calls[0][1]
    assert series.metric == "untrimmed_sigma_gradient"
    assert series.judged and len(series.points) == 12   # lot medians, not units


# ---- INVESTIGATE: stats table + lot-vs-history (app-shape spec §2) --------

def _stats_row(key="untrimmed_resistance", *, n=3, avg=4281.8, low=422.0,
               high=29576.0, excluded=7, missing=1):
    from laser_trim_analyzer.core.model_stats import Cell, StatRow
    cell = Cell(n=n, excluded=excluded, missing=missing, avg=avg, low=low, high=high)
    return StatRow(key=key, label="Untrimmed resistance", unit="ohms",
                   kind="distribution", all_=cell, lin_passing=cell)


def test_stats_cells_render_one_unit_per_row():
    from laser_trim_analyzer.gui.v6.widgets.stats_table import cell_texts
    row = _stats_row()
    assert cell_texts(row, row.all_) == ["3", "4.28 kΩ", "0.422 kΩ", "29.6 kΩ"]


def test_rate_cells_render_count_and_percent():
    from laser_trim_analyzer.core.model_stats import Cell, StatRow
    from laser_trim_analyzer.gui.v6.widgets.stats_table import cell_texts
    cell = Cell(n=9934, excluded=0, missing=10, count=6325, pct=63.67)
    row = StatRow(key="trim_passed_linearity", label="Tracks that passed linearity",
                  unit="%", kind="rate", all_=cell, lin_passing=cell)
    assert cell_texts(row, cell) == ["9,934", "6,325", "63.7%"]


def test_the_table_says_what_it_left_out():
    """Disclose, never hide — every drop is named on the row that made it,
    and the three reasons send you to three different places."""
    from laser_trim_analyzer.core.model_stats import Cell
    from laser_trim_analyzer.gui.v6.widgets.stats_table import disclosure_text
    row = _stats_row()
    assert disclosure_text(row.all_) == "7 impossible readings excluded · 1 not recorded"
    assert disclosure_text(_stats_row(excluded=0, missing=0).all_) == ""
    assert disclosure_text(_stats_row(excluded=1, missing=0).all_) \
        == "1 impossible reading excluded"
    # The 8856 case: 75 records the analyser could not read. A row that showed
    # n=113 and said nothing would hide them.
    assert disclosure_text(Cell(n=113, excluded=0, missing=0, errored=75)) \
        == "75 from records that failed processing"


def test_summary_line_names_the_window_and_the_drops():
    from datetime import datetime
    from laser_trim_analyzer.core.model_stats import ModelStats
    from laser_trim_analyzer.gui.v6.widgets.stats_table import summary_line
    stats = ModelStats(model="6607", rows=[_stats_row()], tracks=9944, records=9944,
                       cutoff=None, lot=None, future_dated=0, note="")
    text = summary_line(stats)
    assert "9,944 track measurements over all history" in text
    assert "7 impossible readings left out" in text
    # Assert the WHOLE phrase, not just the date. Checking for "since May 13"
    # alone passed while the line actually read "measurements over since
    # May 13, 2026" — a weak assertion that let broken copy ship to the screen.
    windowed = ModelStats(model="6607", rows=[_stats_row()], tracks=302, records=302,
                          cutoff=datetime(2026, 5, 13), lot=None, future_dated=0,
                          note="")
    assert summary_line(windowed).startswith(
        "302 track measurements since May 13, 2026")
    lot_scoped = ModelStats(model="6607", rows=[_stats_row()], tracks=15, records=15,
                            cutoff=None, lot=object(), future_dated=0, note="")
    assert summary_line(lot_scoped).startswith(
        "15 track measurements over the selected lot")


def test_lot_line_carries_the_numbers_and_the_verdict():
    from laser_trim_analyzer.core.model_stats import Cell, LotVerdict
    from laser_trim_analyzer.gui.v6.widgets.stats_table import lot_line
    row = _stats_row()
    cell = Cell(n=69, excluded=0, missing=0, avg=4430.0, low=4100.0, high=4800.0)
    verdict = LotVerdict(metric="untrimmed_resistance", label="Untrimmed resistance",
                         status="within", lot_typical=4431.0, lot_n=69,
                         normal_low=1000.0, normal_high=9800.0,
                         text="Untrimmed resistance for this lot is within its normal")
    text = lot_line(row, cell, verdict)
    assert text.startswith("this lot: 69 readings · avg 4.43 kΩ · 4.10 kΩ to 4.80 kΩ")
    assert "within its normal" in text
    assert lot_line(row, Cell(n=0, excluded=0, missing=5), None) \
        == "this lot: nothing recorded"


def _stats_app(make_app, model="HOT"):
    """App on the Model page with a seeded model, loaded synchronously."""
    app = make_app()
    _seed_tracks(app.db, model, "untrimmed_resistance")
    app.set_model_route(model)
    page = app.page_container.get_page("model")
    page._reload = lambda **kw: None
    app.show_page("model")
    del page._reload
    page.reload_now()
    return app, page


def test_model_page_shows_the_stats_table(make_app):
    """The table replaces the Excel round trip, so it has to be ON the page."""
    app, page = _stats_app(make_app)
    texts = [w.cget("text") for w in _all_labels(page._stats_table)]
    assert any("track measurements" in t for t in texts)
    assert any("Untrimmed resistance" in t for t in texts)
    assert any("Lin-passing" in t for t in texts)
    assert any("Tracks that passed linearity" in t for t in texts)


def test_model_page_lot_selector_offers_all_history_plus_the_lots(make_app):
    app, page = _stats_app(make_app)
    values = page._lot_menu.cget("values")
    assert values[0] == "All history (no lot)"
    assert len(values) == 13                      # _seed_tracks builds 12 lots
    assert any("unit" in v for v in values[1:])


def test_model_page_defaults_to_the_current_lot_when_one_is_open(make_app):
    """James's default: when a lot is running, that is what he is asking about."""
    app, page = _stats_app(make_app)
    assert page._lot_label is not None
    assert "current lot" in page._lot_menu.get()


def test_choosing_all_history_sticks_across_a_reload(make_app):
    """The per-model default must not fight the user's own pick."""
    app, page = _stats_app(make_app)
    page._on_lot_change("All history (no lot)")
    assert page._lot_label is None
    page.reload_now()
    assert page._lot_label is None
    assert page._lot_menu.get() == "All history (no lot)"


def test_selecting_a_lot_adds_this_lot_lines_and_a_verdict(make_app):
    app, page = _stats_app(make_app)
    texts = [w.cget("text") for w in _all_labels(page._stats_table)]
    assert any(t.strip().startswith("this lot:") for t in texts)
    assert any("normal" in t for t in texts)      # the SPC core's own words


def test_existing_model_page_content_is_still_there(make_app):
    """Nothing currently reachable may be lost (app-shape spec §2)."""
    app, page = _stats_app(make_app)
    for attr in ("_focus_chart", "_pill_row", "_drift_tab", "_smoothness_tab",
                 "_units_tab", "_ft_units_tab", "_trimft_tab", "_history_tab",
                 "_predictor", "_worth_section"):
        assert getattr(page, attr) is not None
    # The old _verdict label is gone -- its text is the page caption now (Task 2).
    assert not hasattr(page, "_verdict")


def _all_labels(widget):
    """Every label in a widget tree (the table renders into nested frames)."""
    out = []
    for child in widget.winfo_children():
        if hasattr(child, "cget"):
            try:
                child.cget("text")
                out.append(child)
            except Exception:
                pass
        out.extend(_all_labels(child))
    return out


def test_summary_line_discloses_records_that_failed_processing():
    """The table's own note carries the 8856 disclosure to the top of the zone."""
    from laser_trim_analyzer.core.model_stats import ModelStats
    from laser_trim_analyzer.gui.v6.widgets.stats_table import summary_line
    stats = ModelStats(model="8856", rows=[_stats_row()], tracks=113, records=113,
                       cutoff=None, lot=None, future_dated=0, errored=75,
                       note="75 record(s) whose processing failed were left out "
                            "— their columns hold error sentinels, not readings")
    assert "processing failed" in summary_line(stats)


# ---- INVESTIGATE: the Excel sheet is the screen, not a second rendering ----

def _pack_with_stats(tmp_path, model="HOT"):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.export.evidence import export_evidence_pack
    db = DatabaseManager(tmp_path / "ev.db")
    _seed_tracks(db, model, "untrimmed_resistance")
    out = export_evidence_pack(db, model, tmp_path / "pack.xlsx")
    import pandas as pd
    return db, pd.read_excel(out, sheet_name=STATS_SHEET, header=2), out


STATS_SHEET = "Stats table"


def test_evidence_pack_has_a_stats_sheet(tmp_path):
    import pandas as pd
    db, sheet, out = _pack_with_stats(tmp_path)
    assert list(pd.read_excel(out, sheet_name=None)) [-1] == STATS_SHEET \
        or STATS_SHEET in pd.read_excel(out, sheet_name=None)
    assert len(sheet) == 8                       # six distributions + two rates
    assert list(sheet["Metric"])[0] == "Untrimmed resistance"


def test_excel_stats_sheet_prints_exactly_what_the_screen_shows(tmp_path):
    """The sheet is what James hands an engineer; it may not round a number
    differently from the page he read it off."""
    from laser_trim_analyzer.core.model_stats import (
        cell_texts, compute_model_stats, disclosure_text)
    db, sheet, _ = _pack_with_stats(tmp_path)
    stats = compute_model_stats(db, "HOT")
    by_metric = {r["Metric"]: r for _, r in sheet.iterrows()}
    for row in stats.rows:
        cells = by_metric[row.label]
        shown = cell_texts(row, row.all_)
        # n stays a NUMBER so Excel can sort and sum it; the screen prints the
        # same number with a thousands separator. The rest is characters.
        assert int(cells["ALL n"]) == row.all_.n
        assert [cells["ALL avg / count"], cells["ALL min / %"]] == shown[1:3]
        lin = cell_texts(row, row.lin_passing)
        assert int(cells["LIN-PASSING n"]) == row.lin_passing.n
        assert cells["LIN-PASSING avg / count"] == lin[1]
        # pandas reads an empty cell back as NaN; the sheet wrote "".
        left_out = cells["Left out"]
        left_out = "" if not isinstance(left_out, str) else left_out
        assert left_out == disclosure_text(row.all_)


def test_excel_stats_sheet_carries_the_lot_verdicts(tmp_path):
    from laser_trim_analyzer.core.model_stats import model_lots
    db, sheet, _ = _pack_with_stats(tmp_path)
    lots = model_lots(db, "HOT")
    assert lots                                   # the fixture builds 12
    verdicts = [v for v in sheet["Lot vs history"] if isinstance(v, str)]
    assert verdicts and any("normal" in v for v in verdicts)
    row = sheet.iloc[0]
    assert str(row["This lot n"]).strip() not in ("", "nan")


def test_excel_stats_sheet_says_which_window_and_lot_it_describes(tmp_path):
    """A table of numbers with no window on it is not evidence."""
    import pandas as pd
    db, _sheet, out = _pack_with_stats(tmp_path)
    head = pd.read_excel(out, sheet_name=STATS_SHEET, header=None, nrows=2)
    assert "track measurements over all history" in str(head.iloc[0, 0])
    assert "unit" in str(head.iloc[1, 0])          # the lot label


def test_a_rate_row_gets_a_this_lot_line_too():
    """"% of the lot that didn't trim" is one of the three questions this
    screen exists to answer, and a count is what answers it."""
    from laser_trim_analyzer.core.model_stats import Cell, StatRow, lot_line
    cell = Cell(n=69, excluded=0, missing=0, count=32, pct=46.376811594)
    row = StatRow(key="trim_passed_linearity", label="Tracks that passed linearity",
                  unit="%", kind="rate", all_=cell, lin_passing=cell)
    assert lot_line(row, cell, None) == "this lot: 32 of 69 (46.4%)"
    assert lot_line(row, Cell(n=0, excluded=0, missing=3), None) \
        == "this lot: nothing recorded"


def test_excel_lot_columns_match_the_screen_for_rate_rows_too(tmp_path):
    from laser_trim_analyzer.core.model_stats import (
        cell_texts, compute_model_stats, model_lots)
    db, sheet, _ = _pack_with_stats(tmp_path)
    lots = model_lots(db, "HOT")
    lot_stats = compute_model_stats(db, "HOT", lot=lots[0].window)
    by_metric = {r["Metric"]: r for _, r in sheet.iterrows()}
    for row in lot_stats.rate_rows:
        shown = cell_texts(row, row.all_)
        cells = by_metric[row.label]
        assert int(cells["This lot n"]) == row.all_.n
        assert cells["This lot avg / count"] == shown[1]
        assert cells["This lot min / %"] == shown[2]


def test_stats_sheet_exists_with_headers_for_a_model_with_no_rows(tmp_path):
    """Same invariant the other six sheets hold: the sheet ALWAYS exists, so
    "no records" cannot be mistaken for a broken export."""
    import pandas as pd
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.export.evidence import export_evidence_pack
    db = DatabaseManager(tmp_path / "empty.db")
    out = export_evidence_pack(db, "NO-SUCH-MODEL", tmp_path / "pack.xlsx")
    sheets = pd.read_excel(out, sheet_name=None)
    assert STATS_SHEET in sheets
    sheet = pd.read_excel(out, sheet_name=STATS_SHEET, header=2)
    for col in ("Metric", "ALL n", "LIN-PASSING n", "Left out", "Lot vs history"):
        assert col in sheet.columns
    # The eight rows still print, each honestly reading zero.
    assert len(sheet) == 8 and set(sheet["ALL n"]) == {0}


def test_window_and_lot_still_hold_when_file_date_carries_a_clock_time(tmp_path):
    """`analysis_results.file_date` stopped being midnight (90dc95e): the
    parser keeps the filename's clock time, because it is the only record of
    the order of same-day re-trim attempts. Every window here is a half-open
    range, never a same-day equality, so a time component must change nothing.
    """
    from datetime import datetime, timedelta
    from laser_trim_analyzer.core.model_stats import compute_model_stats
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType, SystemType)
    db = DatabaseManager(tmp_path / "clock.db")
    day = datetime(2026, 8, 11)
    with db.session() as s:
        for i, hour in enumerate((0, 9, 16, 23)):
            ar = DBAR(model="CLK", serial=f"sn{i}", system=SystemType.A,
                      filename=f"CLK_{i}.xls",
                      file_date=day + timedelta(hours=hour, minutes=37),
                      overall_status=StatusType.PASS)
            s.add(ar); s.flush()
            s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                       untrimmed_resistance=1000.0 + i))
        s.commit()
    # A lot is a run of DAYS, so the last day's 23:37 unit has to fall inside
    # a lot ending that day.
    assert compute_model_stats(db, "CLK", lot=(day, day)).tracks == 4
    # And a cutoff at the start of the day keeps all four, not just the 00:37 one.
    assert compute_model_stats(db, "CLK", cutoff=day).tracks == 4
    # Bounds handed in WITH a clock time still mean the whole day: nothing may
    # depend on the caller's value happening to be midnight.
    noon = day + timedelta(hours=12)
    assert compute_model_stats(db, "CLK", lot=(noon, noon)).tracks == 4
    # And the same code on midnight-only data (every other test in this file,
    # and most of the live database today) is unchanged.
    with db.session() as s:
        ar = DBAR(model="MID", serial="m1", system=SystemType.A,
                  filename="MID_1.xls", file_date=day, overall_status=StatusType.PASS)
        s.add(ar); s.flush()
        s.add(DBTR(analysis_id=ar.id, track_id="T1", status=StatusType.PASS,
                   untrimmed_resistance=1000.0))
        s.commit()
    assert compute_model_stats(db, "MID", lot=(day, day)).tracks == 1
    assert compute_model_stats(db, "MID",
                               lot=(day - timedelta(days=1),
                                    day - timedelta(days=1))).tracks == 0


# ---- Task 2 (facelift step 2): "Worth changing" comes first ---------------
# docs/superpowers/specs/2026-09-24-facelift-step2-pages-design.md §1 "Investigate",
# ruling 1, items 1-4. The old _verdict body label is gone (its text is now the page
# caption); the two TIER_WARNING captions are blocks.banner check-tone banners; the
# model's cached findings move from the seventh tab to a "Worth changing on this
# model" section above the pills; the three-sentence σ key shrinks to one line, its
# full explanation moved to the Drift metrics tab.

# Verified live shape (tests/test_findings_tab.py's own FINDING): ink_target -> "yield",
# one of the three groups _WORTH_CHANGING_GROUPS includes. Invented values throughout.
_WORTH_CHANGING_FINDING = {
    "model": "HOT", "analyzer": "ink_target", "title": "Incoming resistance: aim lower",
    "category": "Ink target", "lever": "ink",
    "lever_label": "Ink formulation (incoming resistance)", "lead_time": "next lot",
    "expected_gain_points": 3.9, "tracks_per_year": 34.0, "annual_volume": 873,
    "summary": ("Invented for the test — within Laser 1 (LTS) running 2 cuts, incoming "
                "resistance below the median did better."),
    "strength_name": "Spearman", "strength_value": -0.15, "n_units": 346, "evidence": {},
}


def _worth_app(make_app, model="HOT", *, finding=True):
    """App on the Model page with a seeded model and (unless finding=False) one cached
    process finding, loaded synchronously -- same pattern as _stats_app above."""
    app = make_app()
    _seed_tracks(app.db, model, "untrimmed_resistance")
    if finding:
        app.db.replace_process_findings(
            model, {"tracks": 1}, [dict(_WORTH_CHANGING_FINDING, model=model)])
    app.set_model_route(model)
    page = app.page_container.get_page("model")
    page._reload = lambda **kw: None
    app.show_page("model")
    del page._reload
    page.reload_now()
    return app, page


def test_caption_carries_the_verdict_sentence(make_app, monkeypatch):
    """The old _verdict body label is gone; its text is now the page caption
    (PageBase.set_caption), so it inherits set_caption's rule: configured on every
    apply, "—" when the verdict could not be computed (M1, unchanged)."""
    app, page = _worth_app(make_app, finding=False)
    monkeypatch.setattr(page, "_compute_verdict", lambda *a, **k: (
        "Holding — invented verdict sentence for the test", page.theme.TEXT_PRIMARY))
    page.reload_now()
    assert page._caption.cget("text") == "Holding — invented verdict sentence for the test"
    assert not hasattr(page, "_verdict")


def test_worth_changing_header_precedes_pills_and_stats_table(make_app):
    app, page = _worth_app(make_app)
    texts = [w.cget("text") for w in _all_labels(page._worth_section)]
    assert any("Worth changing on this model" in t for t in texts)

    slaves = page._body.pack_slaves()
    assert slaves.index(page._worth_section) < slaves.index(page._pill_row)
    assert slaves.index(page._worth_section) < slaves.index(page._stats_table)

    view = page._worth_view
    assert view is not None
    assert view._include_empty is False
    assert view._on_open is None
    assert view._rows_per_group == 3
    assert view._groups == {"yield", "laser_time", "check"}
    assert [f["title"] for f in view._findings] == ["Incoming resistance: aim lower"]


def test_no_findings_shows_the_quiet_line_not_the_view(make_app):
    app, page = _worth_app(make_app, finding=False)
    texts = [w.cget("text") for w in _all_labels(page._worth_section)]
    assert any(t == "Nothing worth changing stands out for this model." for t in texts)
    assert page._worth_view is None


def test_a_failed_findings_load_banners_never_the_quiet_line(make_app, monkeypatch):
    """A crashed 'process findings' loader must say NOTHING in the Worth-changing
    section -- the general load banner (already naming "process findings", which
    contains "findings") does the talking; drawing "nothing worth changing" on top
    of a crash would be the CLAUDE.md hazard this page exists to remove."""
    app = make_app()
    _seed_tracks(app.db, "HOT", "untrimmed_resistance")
    app.set_model_route("HOT")
    page = app.page_container.get_page("model")
    page._reload = lambda **kw: None
    app.show_page("model")
    del page._reload

    def _boom(*a, **kw):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(app.db, "get_process_facts", _boom)
    page.reload_now()

    assert page._load_banner.winfo_manager() != ""
    assert "findings" in page._load_banner.cget("text")
    texts = [w.cget("text") for w in _all_labels(page._worth_section)]
    assert not any("Nothing worth changing" in t for t in texts)
    assert page._worth_view is None


def test_spec_and_load_banners_are_check_tone_blocks_hidden_when_quiet(make_app):
    app, page = _worth_app(make_app, finding=False)
    t = page.theme
    for banner in (page._spec_banner, page._load_banner):
        assert banner.cget("text_color") == t.CHECK
        assert banner.cget("fg_color") == t.CHECK_TINT
        assert banner.winfo_manager() == ""        # nothing to say on a healthy load


def test_sigma_key_is_one_line_full_explanation_moved_to_drift_tab(make_app):
    app, page = _worth_app(make_app, finding=False)
    assert page._sigma_key.cget("text") == (
        "σ = how far the last lot sits from this model's history of lots — a "
        "drift signal, not a spec.")
    slaves = page._body.pack_slaves()
    assert slaves.index(page._sigma_key) == slaves.index(page._pill_row) + 1

    full = page._drift_tab._sigma_key_lbl.cget("text")
    assert "baseline of historical lot medians" in full
    assert "Drift signal, not a spec." in full


def test_tabs_have_a_minimum_height_so_a_selected_tab_never_collapses(make_app):
    """render_pages.py --audit found 6607's Smoothness tab SQUEEZED OUT (unmapped) at
    1280x720 once "Worth changing" made everything above the tabs taller than the window.
    CTkTabview does not propagate the selected tab's own content size upward
    (customtkinter's ctk_tabview.py: `_configure_grid` grids the tab frame `sticky="nsew"`
    into a `weight=1` row, so it gets exactly however tall pack() allocates the tabview
    itself -- nothing about its content). `expand=True` only fills LEFTOVER room in the
    scrollable body once every other child has its natural size, which used to always be
    positive; once it is not, pack falls back to CTkTabview's own un-set default
    (measured ~250px), too short for even its button row plus a usable content row.
    A minimum height keeps expand=True's "grow when there's room" behaviour while giving
    every tab a floor it can never be squeezed under -- 520 comfortably held every tab's
    content in the audited data (measured 404-684px per tab)."""
    app = make_app()
    page = app.page_container.get_page("model")
    assert page._tabs.cget("height") >= 520
