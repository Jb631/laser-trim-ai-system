"""What the user's hands feel: the app must not block the Tk thread.

Measured with `scripts/ui_stall_probe.py` on 2026-09-02 after James said the
whole app felt sluggish. Every component was "fast" in isolation; the probe
found three things that were not, all of them waste ON the Tk thread:

  * a fresh CTkFont per widget (thousands of Tcl `font create`/`delete` calls),
  * CustomTkinter's scrollbar redrawing re-entrantly on every <Configure>,
  * matplotlib canvases drawing synchronously, once per configure event,
  * and — the biggest single stall of the four — a model switch destroying the
    PREVIOUS model's ~1,500 native widgets before it would draw the new ones.

These tests are STRUCTURAL, never timing-based: a millisecond assertion flakes
on a loaded machine, while "how many fonts were built", "how many draws
happened" and "is the old container still mapped" cannot. The timings live in
the commit messages and the probe.
"""
import time
import weakref

import customtkinter as ctk
import pytest

from laser_trim_analyzer.gui.v6.theme import ThemeManager


# ---- Commit 1: one CTkFont per (family, size, weight) ----------------------

def test_theme_font_returns_the_same_object_for_the_same_size(tk_root):
    t = ThemeManager()
    assert t.font(t.SIZE_BODY) is t.font(t.SIZE_BODY)
    assert t.font(t.SIZE_BODY, "bold") is t.font(t.SIZE_BODY, "bold")


def test_theme_font_still_separates_size_and_weight(tk_root):
    t = ThemeManager()
    assert t.font(t.SIZE_BODY) is not t.font(t.SIZE_CAPTION)
    assert t.font(t.SIZE_BODY) is not t.font(t.SIZE_BODY, "bold")
    assert t.font(t.SIZE_BODY, "bold").cget("weight") == "bold"
    assert t.font(t.SIZE_CAPTION).cget("size") == t.SIZE_CAPTION


def _count_live_fonts(monkeypatch):
    """Weak refs to every CTkFont built while the returned list is in scope."""
    built = []
    original_init = ctk.CTkFont.__init__

    def counting_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        built.append(weakref.ref(self))

    monkeypatch.setattr(ctk.CTkFont, "__init__", counting_init)
    return built


def test_two_hundred_rows_build_a_handful_of_fonts_not_one_per_row(tk_root, monkeypatch):
    """Font count must scale with distinct SIZES, not with row count."""
    from datetime import datetime, timedelta
    from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab

    units = [{"analysis_id": i, "serial": f"SN{i:05d}",
              "file_date": datetime(2026, 1, 1) + timedelta(days=i % 90),
              "overall_status": "Fail" if i % 3 == 0 else "Pass",
              "sigma_gradient": 0.01, "linearity_error": 0.004} for i in range(200)]

    built = _count_live_fonts(monkeypatch)
    tab = UnitsTab(tk_root, theme=ThemeManager(), on_unit_click=lambda u: None,
                   on_export=lambda: None)
    tab.set_units(units)
    tab._toggle_expand()          # render all 200, not the 50-row budget
    live = [ref for ref in built if ref() is not None]

    # 200 rows x (5 labels + 1 checkbox) used to be 1,250 fonts; it is 8 now —
    # three cached theme fonts plus the tab's own chrome. The ceiling has a
    # little headroom for a new toolbar widget but stays far below O(rows), so
    # re-introducing a per-row font fails this loudly.
    assert len(tab._rows) == 200
    assert len(live) <= 12, f"{len(live)} CTkFont objects for 200 rows"
    tab.destroy()


# ---- Commit 2: the CTkScrollbar redraw must not re-enter -------------------

@pytest.fixture
def patched_ctk():
    """The pinned-CustomTkinter patches, installed. Strict: a dependency bump
    that moves off 5.2.2 fails here instead of quietly un-patching the UI."""
    from laser_trim_analyzer.gui.v6 import ctk_patches
    assert ctk_patches.apply(strict=True)
    return ctk_patches


class _ScrollDepth:
    depth = 0


def test_nested_scrollbar_draw_does_not_pump_the_idle_queue_again(tk_root, patched_ctk):
    """The whole cascade: _draw -> update_idletasks -> (Tk fires a scroll
    callback) -> set -> _draw -> update_idletasks -> ... 46 levels deep.

    The nested redraw must still DRAW (the scrollbar has to track content) but
    must not drain the idle queue a second time.

    Re-entry is provoked from inside the idle pump, which is where Tk really
    does it — geometry management runs there, fires <Configure>, and a scroll
    region change calls set(). Provoking it anywhere else would test a path the
    guard is not even on, and pass without proving anything.
    """
    sb = ctk.CTkScrollbar(tk_root)
    sb.pack()
    tk_root.update_idletasks()

    draws = []
    patched_draw = type(sb)._draw

    def counting_draw(self, *args, **kwargs):
        _ScrollDepth.depth += 1
        draws.append(_ScrollDepth.depth)
        try:
            return patched_draw(self, *args, **kwargs)
        finally:
            _ScrollDepth.depth -= 1

    pumps = []
    real_pump = type(sb._canvas).update_idletasks

    def spying_pump():
        pumps.append(_ScrollDepth.depth)
        if len(pumps) == 1:                 # exactly as Tk does: from the pump
            sb.set(0.25, 0.75)
        return real_pump(sb._canvas)

    type(sb)._draw = counting_draw
    sb._canvas.update_idletasks = spying_pump
    try:
        sb.set(0.0, 0.5)
    finally:
        type(sb)._draw = patched_draw
        sb._canvas.__dict__.pop("update_idletasks", None)
        _ScrollDepth.depth = 0

    assert max(draws) == 2, "the re-entry the test provokes did not happen"
    assert pumps == [1], (
        f"the idle queue was pumped at _draw depths {pumps}; only the outermost "
        f"redraw may pump, or the cascade is back")
    # ...and the nested draw still took effect: set() is not a no-op.
    assert sb.get() == (0.25, 0.75)
    sb.destroy()


def test_scrollbar_redraw_guard_is_released_after_an_exception(tk_root, patched_ctk):
    """A raising redraw must not leave every later scrollbar un-pumped."""
    from laser_trim_analyzer.gui.v6.ctk_patches import _ScrollbarRedraw

    sb = ctk.CTkScrollbar(tk_root)
    sb.pack()

    def boom():
        raise RuntimeError("redraw blew up")

    # The last thing CustomTkinter's _draw does is pump this; make it raise.
    sb._canvas.update_idletasks = boom
    with pytest.raises(RuntimeError):
        sb.set(0.1, 0.4)
    del sb._canvas.update_idletasks

    assert _ScrollbarRedraw.in_progress is False
    sb.set(0.2, 0.5)          # and the next redraw works normally
    assert sb.get() == (0.2, 0.5)
    sb.destroy()


def test_scrollable_frame_with_100_children_still_scrolls_to_the_bottom(tk_root, patched_ctk):
    """Behavioural guard on the patch: the scrollbar must still track content.

    A guard that suppressed the redraw itself (rather than only the nested idle
    pump) would leave the thumb stuck at the top — the scrollbar would lie
    about where you are in the list. This asserts it does not.
    """
    theme = ThemeManager()
    frame = ctk.CTkScrollableFrame(tk_root, width=200, height=150)
    frame.pack(fill="both", expand=True)
    for i in range(100):
        ctk.CTkLabel(frame, text=f"row {i}", font=theme.font(theme.SIZE_BODY)).pack()
    tk_root.update_idletasks()

    top = frame._scrollbar.get()
    assert top[0] == pytest.approx(0.0, abs=1e-6)
    assert top[1] < 1.0, "100 rows in a 150px frame should overflow"

    frame._parent_canvas.yview_moveto(1.0)
    tk_root.update_idletasks()
    bottom = frame._scrollbar.get()
    assert bottom[1] == pytest.approx(1.0, abs=1e-6)
    assert bottom[0] > top[0], "the scrollbar did not follow the view to the bottom"
    frame.destroy()


# ---- Commit 3: charts render once per burst, and once per resize ----------

def _chart(tk_root):
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
    chart = FocusChart(tk_root, theme=ThemeManager())
    chart.pack(fill="both", expand=True)
    tk_root.update_idletasks()
    return chart


def test_five_rapid_set_series_calls_render_once(tk_root):
    """draw_idle() coalesces; five set_series in one apply() must not be five
    full Agg renders. Counted on the canvas's real draw, not on draw_idle."""
    from datetime import datetime, timedelta

    chart = _chart(tk_root)
    draws = []
    real_draw = chart.canvas.draw

    def counting_draw(*args, **kwargs):
        draws.append(1)
        return real_draw(*args, **kwargs)

    chart.canvas.draw = counting_draw
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(20)]
    for run in range(5):
        chart.set_series("sigma_gradient", dates, [0.01 + run + i * 1e-4 for i in range(20)])
    assert draws == [], "a set_series rendered synchronously; use draw_idle()"

    tk_root.update_idletasks()          # cash in the single pending idle draw
    assert len(draws) == 1, f"{len(draws)} renders for 5 set_series calls, expected 1"
    chart.destroy()


def _queued_after_ids(widget):
    return set(widget.tk.splitlist(widget.tk.call("after", "info")))


def test_the_chart_binds_its_debounce_to_configure(tk_root):
    """The wiring, asserted separately from the logic.

    The two tests below drive `on_configure` directly. That is deliberate — an
    earlier version generated real <Configure> events into a mapped
    CTkToplevel and spun the Tk event loop forever inside a full pytest
    session (it passed in isolation and hung the suite at test ~1340). So this
    test carries the other half: the handler really is on the widget's
    <Configure>, and a rename or a dropped bind fails here.
    """
    chart = _chart(tk_root)
    bindings = chart.canvas.get_tk_widget().bind("<Configure>")
    assert "on_configure" in bindings, f"debounce not bound; bindings={bindings!r}"
    # ...and matplotlib's own resize handler is still there, ahead of it: the
    # debounce cancels the idle draw that `resize` arms, so order matters.
    assert bindings.index("resize") < bindings.index("on_configure")
    chart.destroy()


def test_a_burst_of_configure_events_leaves_exactly_one_pending_redraw(tk_root):
    """A drag is ~90 <Configure> events. They must collapse to one after-call."""
    chart = _chart(tk_root)
    state = chart._redraw
    tk_widget = chart.canvas.get_tk_widget()
    state.pending = None

    ids = []
    for _ in range(9):                  # nine configure events, as a drag sends
        state.on_configure()
        ids.append(state.pending)

    assert all(i is not None for i in ids), "no redraw was scheduled at all"
    assert len(set(ids)) == len(ids), "the pending callback was never re-armed"
    # Only the LAST survives: every earlier one was cancelled, so nine configure
    # events leave one queued render rather than nine.
    queued = _queued_after_ids(tk_widget)
    assert [i for i in ids if i in queued] == [ids[-1]], (
        "more than the newest redraw is still queued — the burst was not debounced")
    assert state.pending == ids[-1]
    chart.destroy()


def test_the_debounced_redraw_defers_the_render_without_swallowing_it(tk_root):
    """Deferred, not dropped: the render still happens, just once and later."""
    from laser_trim_analyzer.gui.v6 import chart_redraw

    chart = _chart(tk_root)
    state = chart._redraw
    before = state.renders

    state.on_configure()
    assert state.pending is not None, "the configure event scheduled nothing"
    assert state.renders == before, "rendered during the burst instead of after it"

    state.render_now()                  # what the pending after-callback runs
    assert state.renders == before + 1, "the deferred render never happened"
    assert state.pending is None, "the render did not clear its own pending id"
    assert chart_redraw.QUIET_MS >= 60, "a quiet window shorter than a frame debounces nothing"
    chart.destroy()


# ---- Commit 4: the old model's widgets are buried on idle time ------------

def _pump(root, seconds: float = 2.0) -> None:
    """Spin the event loop the way the app's own probe does.

    `update()`, not `update_idletasks()`: the teardown slices are TIMER
    callbacks precisely so they cannot fire inside an idle pump, and a test
    that only pumped idles would never run a single one of them.
    """
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        root.update()
        time.sleep(0.002)


def _tree(widget) -> list:
    """The widget and every descendant, from Tkinter's own child dict."""
    out = [widget]
    for child in list(getattr(widget, "children", {}).values()):
        out.extend(_tree(child))
    return out


def _live(widgets) -> list:
    return [w for w in widgets if w.winfo_exists()]


def _rows_of(n, prefix):
    from datetime import datetime, timedelta
    return [{"analysis_id": f"{prefix}{i}", "serial": f"{prefix}{i:04d}",
             "file_date": datetime(2026, 1, 1) + timedelta(days=i),
             "overall_status": "Pass", "sigma_gradient": 0.01,
             "linearity_error": 0.004} for i in range(n)]


def _filled_frame(root, rows=30):
    theme = ThemeManager()
    host = ctk.CTkFrame(root)
    host.pack()
    for i in range(rows):
        ctk.CTkLabel(host, text=f"row {i}", font=theme.font(theme.SIZE_BODY)).pack()
    root.update_idletasks()
    return host


def test_retire_unmaps_the_container_now_and_destroys_it_later(tk_root):
    """The two halves of a teardown, split: the visible one now, the slow one
    later. This is the whole fix — 1.5 s of CTk destroy used to happen between
    the click and the new content."""
    from laser_trim_analyzer.gui.v6.retire import retire

    host = _filled_frame(tk_root)
    subtree = _tree(host)
    assert len(subtree) > 60, "30 CTkLabels should be ~90 native widgets"

    retirement = retire(host)

    assert host.winfo_manager() == "", "the old container is still in the layout"
    assert host not in tk_root.pack_slaves()
    assert host.winfo_exists(), "the destroy was not deferred at all"
    assert retirement.destroyed == 0, "widgets died during the switch, not after it"

    _pump(tk_root, 2.0)
    assert retirement.done
    assert _live(subtree) == [], "a deferred teardown that never finishes is a leak"


def test_the_teardown_is_sliced_instead_of_paid_in_one_stall(tk_root):
    """No slice may be big enough to be felt: the point is not that the work
    goes away (it cannot) but that it is never all in one frame."""
    from laser_trim_analyzer.gui.v6.retire import BATCH_NODES, Retirement

    host = _filled_frame(tk_root, rows=40)
    nodes = len(_tree(host))

    retirement = Retirement([host], batch=BATCH_NODES)
    retirement.unmap()
    slices = []
    while not retirement.done:
        slices.append(retirement.step())

    # A slice destroys at most one atomic subtree past its budget, so the
    # ceiling is 2x the batch — and with ~120 nodes that means several slices,
    # not one big one.
    assert max(slices) <= 2 * BATCH_NODES, f"a slice destroyed {max(slices)} widgets"
    assert len(slices) >= 3, f"{nodes} widgets went down in {len(slices)} slice(s)"
    assert retirement.destroyed == nodes
    assert host.winfo_exists() == 0


def test_a_retirement_with_no_live_toplevel_falls_back_to_destroying_now(tk_root):
    """Nothing may be left alive because there was nowhere to hang a timer."""
    from laser_trim_analyzer.gui.v6.retire import Retirement

    host = _filled_frame(tk_root)
    retirement = Retirement([host])
    retirement.host = None          # the window went away mid-switch
    retirement.unmap()
    retirement.schedule()           # nowhere to hang a timer -> drain now
    assert retirement.done
    assert host.winfo_exists() == 0


def test_a_pending_drain_never_holds_update_idletasks(tk_root):
    """The drain must live on the TIMER queue, never the idle queue.

    `update_idletasks()` runs idle handlers until the queue is EMPTY, so a
    chain that re-arms itself with `after_idle` never lets it return: the app
    freezes outright with the CPU busy, which is far worse than the stall this
    module exists to remove. CustomTkinter pumps the idle queue constantly
    (`ctk_scrollbar._draw`, `_update_dimensions_event`), so such a chain would
    hang the real app on the next scrollbar redraw, not in some corner case.

    Reproduces that shape: a big retirement in flight, then a tight loop of
    `update_idletasks()` with a wall-clock guard on every single call.
    """
    from laser_trim_analyzer.gui.v6.retire import retire

    host = _filled_frame(tk_root, rows=200)
    retirement = retire(host)
    assert not retirement.done, "nothing was left pending to test with"

    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
        started = time.perf_counter()
        tk_root.update_idletasks()
        held = (time.perf_counter() - started) * 1000
        assert held < 500, (
            f"update_idletasks() was held for {held:.0f} ms by a pending "
            f"teardown — the drain is on the idle queue, and the app is frozen")
        tk_root.update()            # timers: this is what drains the slices
        time.sleep(0.002)

    assert retirement.done, "the drain never finished while the loop pumped"
    assert host.winfo_exists() == 0


def _units_tab(root):
    from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab
    tab = UnitsTab(root, theme=ThemeManager(), on_unit_click=lambda u: None,
                   on_export=lambda: None)
    tab.pack(fill="both", expand=True)
    return tab


def test_a_second_switch_while_the_first_drains_shows_only_the_newest_rows(tk_root):
    """James clicks two models in a row. The pending burial of the first must
    not touch the second, and must not leave an orphan behind."""
    errors = []
    tk_root.report_callback_exception = lambda *a: errors.append(a)

    tab = _units_tab(tk_root)
    tab.set_units(_rows_of(40, "A"))
    first = tab._rows_host
    first_tree = _tree(first)
    tab.set_units(_rows_of(40, "B"))        # back to back, nothing pumped between
    second = tab._rows_host

    assert first is not second, "the new rows were built into the retiring container"
    assert first.winfo_exists(), "the second switch paid for the first one's teardown"
    # Exactly the newest list is on the page, and only it.
    assert tab._list.pack_slaves() == [second]
    assert len(tab._rows) == 40
    assert all(r.unit["serial"].startswith("B") for r in tab._rows)

    _pump(tk_root, 2.0)

    assert _live(first_tree) == [], "the first model's rows are still alive"
    assert second.winfo_exists()
    assert tab._list.pack_slaves() == [second]
    # Zero orphans: the scrollable list holds the live rows and the persistent
    # show-all button, and nothing else.
    assert set(tab._list.children.values()) == {second, tab._show_all_btn}
    assert errors == [], f"a deferred callback raised: {errors}"


def test_the_render_budget_and_show_all_survive_the_deferred_teardown(tk_root):
    """The cap is a drawing budget, not a data limit, and the button that
    lifts it is persistent — a teardown that swept it up would take the only
    way back to the other 70 rows with it."""
    tab = _units_tab(tk_root)
    button = tab._show_all_btn

    tab.set_units(_rows_of(120, "A"))
    tab.set_units(_rows_of(120, "B"))
    _pump(tk_root, 2.0)

    assert button.winfo_exists(), "the show-all button was swept up by the teardown"
    assert len(tab._rows) == 50, "the first-50 render budget changed"
    assert "Show all 120 units" in button.cget("text")
    assert tab._list.pack_slaves() == [tab._rows_host, button]

    tab._toggle_expand()
    _pump(tk_root, 2.0)
    assert len(tab._rows) == 120, "show-all no longer shows all"
    assert len(tab.get_selected_units()) == 120, "the export still sees every unit"


def _model_stats(model="6607"):
    from laser_trim_analyzer.core.model_stats import Cell, ModelStats, StatRow
    cell = Cell(n=3, excluded=0, missing=0, avg=4281.8, low=422.0, high=29576.0)
    rows = [StatRow(key="untrimmed_resistance", label="Untrimmed resistance",
                    unit="ohms", kind="distribution", all_=cell, lin_passing=cell),
            StatRow(key="measured_electrical_angle", label="Electrical angle",
                    unit="deg", kind="distribution", all_=cell, lin_passing=cell)]
    return ModelStats(model=model, rows=rows, tracks=3, records=3, cutoff=None,
                      lot=None, future_dated=0, note="")


def test_the_stats_table_retires_its_previous_render(tk_root):
    """One of these grid frames is ~450 native widgets — the single biggest
    teardown on a model switch."""
    from laser_trim_analyzer.gui.v6.widgets.stats_table import StatsTableZone

    zone = StatsTableZone(tk_root, theme=ThemeManager())
    zone.pack(fill="x")
    zone.set_stats(_model_stats("8232-1"))
    old = list(zone._rendered)
    old_tree = [w for root_w in old for w in _tree(root_w)]
    assert old, "nothing was rendered to retire"

    zone.set_stats(_model_stats("6607"))

    assert all(w.winfo_exists() for w in old), "the switch paid for the old render"
    assert all(w.winfo_manager() == "" for w in old), "the old render is still shown"
    # Only the new render is in the layout, in the order it was packed.
    assert zone.pack_slaves() == zone._rendered

    _pump(tk_root, 2.0)
    assert _live(old_tree) == [], "the old stats render never got destroyed"
    assert zone.pack_slaves() == zone._rendered


def _drift_status(model="M1"):
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, MetricStatus, ModelDriftStatus, WATCHED_METRICS)
    per = {m: MetricStatus(metric=m, tier=DriftTier.STABLE, alert_type=None,
                           magnitude=0.0, baseline_mean=0.01, baseline_std=0.001,
                           recent_mean=0.012, recent_count=5, is_trained=True)
           for m in WATCHED_METRICS}
    return ModelDriftStatus(model=model, overall_tier=DriftTier.STABLE,
                            worst_metric=None, worst_alert_type=None, per_metric=per)


def test_the_drift_tab_retires_its_metric_rows(tk_root):
    from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import DriftMetricsTab

    tab = DriftMetricsTab(tk_root, theme=ThemeManager(), on_metric_select=lambda m: None)
    tab.pack(fill="both", expand=True)
    tab.set_status(_drift_status("8232-1"))
    old = list(tab._rows.values()) + list(tab._group_headers)
    old_tree = [w for root_w in old for w in _tree(root_w)]
    assert len(tab._rows) > 0

    tab.set_status(_drift_status("6607"))

    assert all(w.winfo_exists() and w.winfo_manager() == "" for w in old)
    _pump(tk_root, 2.0)
    assert _live(old_tree) == [], "the previous model's drift rows are still alive"
    assert len(tab._rows) == len(_drift_status().per_metric)


def test_the_trim_vs_ft_tab_retires_its_previous_render(tk_root):
    from laser_trim_analyzer.gui.v6.widgets.trim_ft_tab import TrimFtTab

    data = {"trim_pass_rate": 91.0, "trim_pass": 91, "trim_total": 100,
            "ft_pass_rate": 88.0, "ft_pass": 88, "ft_total": 100, "linked": 80,
            "escapes": 3, "overkills": 2, "agreement_rate": 94.0, "agreements": 75,
            "escape_units": ["12", "3"], "overkill_units": ["7"],
            "trim_pass_count_avg": 1.4, "trim_pass_count_dist": {1: 60, 2: 20}}
    tab = TrimFtTab(tk_root, theme=ThemeManager())
    tab.pack(fill="both", expand=True)
    tab.set_data(data)
    old = list(tab._rendered)
    old_tree = [w for root_w in old for w in _tree(root_w)]
    assert old

    tab.set_data(data)
    assert all(w.winfo_exists() and w.winfo_manager() == "" for w in old)
    _pump(tk_root, 2.0)
    assert _live(old_tree) == [], "the previous model's agreement rows are still alive"
