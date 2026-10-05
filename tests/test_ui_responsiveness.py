"""What the user's hands feel: the app must not block the Tk thread.

Measured with `scripts/ui_stall_probe.py` on 2026-09-02 after James said the
whole app felt sluggish. Every component was "fast" in isolation; the probe
found three things that were not, all of them waste ON the Tk thread:

  * a fresh CTkFont per widget (thousands of Tcl `font create`/`delete` calls),
  * CustomTkinter's scrollbar redrawing re-entrantly on every <Configure>,
  * matplotlib canvases drawing synchronously, once per configure event.

These tests are STRUCTURAL, never timing-based: a millisecond assertion flakes
on a loaded machine, while "how many fonts were built" and "how many draws
happened" cannot. The timings live in the commit messages and the probe.
"""
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
    # "bold" maps onto the resolved Medium family (weight "normal") when one is available
    # (see test_spec3a_shell.py for that branch, forced with a monkeypatch); on THIS
    # machine there is none, so it is real bold weight on the regular family -- branch on
    # the state actually resolved rather than assume one.
    f_bold = t.font(t.SIZE_BODY, "bold")
    if t.resolved_medium:
        assert f_bold.cget("weight") == "normal"
    else:
        assert f_bold.cget("weight") == "bold"
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


# ---- The app's other private CustomTkinter uses (F4 review, Minor 2; option B review #4) --
# ctk_patches.py is not the only code that rests on the pin: ctk_patches.PRIVATE_USES lists every
# other file that overrides, wraps or reads CustomTkinter's private code, with the names it rests
# on. Whoever bumps the pin must be pointed at every one, and a changed meaning must fail here,
# naming the file to re-read -- not surface as an empty tab area at work. (2026-10-04: the note
# said "three more places" over four, and the message named three of them.)

def test_a_version_mismatch_names_every_private_override_it_puts_at_risk(monkeypatch):
    from laser_trim_analyzer.gui.v6 import ctk_patches
    monkeypatch.setattr(ctk_patches, "_applied", False)
    monkeypatch.setattr(ctk, "__version__", "9.9.9")
    with pytest.raises(RuntimeError) as raised:
        ctk_patches.apply(strict=True)
    message = str(raised.value)
    assert "ctk_patches.py" in message
    # Named outright, so an emptied list cannot pass: the two the F4 review found, and the ones
    # the option B review found missing from the note or the message.
    for path, name in (("widgets/tab_view.py", "_grid_forget_all_tabs"),
                       ("pages/model_page.py", "_open_dropdown_menu"),
                       ("widgets/blocks.py", "_QuietArrow"),
                       ("pages/home_page.py", "check_if_master_is_canvas"),
                       ("pages/home_page.py", "_scrollbar_only_when_needed"),
                       ("widgets/tab_view.py", "_button_height"),
                       ("widgets/tab_view.py", "_buttons_dict")):
        assert path in message and name in message, (path, name)
    doc = ctk_patches.__doc__
    assert "Three more places" not in doc                  # the count that was wrong; no count now
    for path, where, names in ctk_patches.PRIVATE_USES:
        assert path in doc, f"the docstring's list leaves out {path}"
        assert path in message and where in message, path
        assert all(name in message for name in names), (path, names)


def test_every_place_the_pin_list_names_still_uses_what_it_cites():
    """Read as text, no window: each file ctk_patches.PRIVATE_USES names exists under gui/v6 and
    still says each name it is listed with -- its own class or function, and each private name of
    CustomTkinter's it rests on. A list that names code which has moved sends whoever bumps the
    pin to re-read the wrong thing."""
    import pathlib
    import re
    from laser_trim_analyzer.gui.v6 import ctk_patches
    v6 = pathlib.Path(ctk_patches.__file__).resolve().parent
    assert len(ctk_patches.PRIVATE_USES) >= 6
    for path, where, names in ctk_patches.PRIVATE_USES:
        source = (v6 / path).read_text()
        for name in [w.strip() for w in where.split(",")] + list(names):
            assert re.search(rf"\b{re.escape(name)}\b", source), (
                f"{path} no longer says {name!r} -- update ctk_patches.PRIVATE_USES")


def test_ctktabview_still_defers_the_forget_that_tab_view_py_overrides(tk_root):
    """ThemedTabView._grid_forget_all_tabs is right only while CTkTabview's method takes
    `exclude_name` and set() calls it 100 ms LATE with the name it switched to (5.2.2)."""
    import inspect
    import time
    read = ("widgets/tab_view.py overrides CTkTabview._grid_forget_all_tabs (ThemedTabView): "
            "re-read that override against this customtkinter")
    params = inspect.signature(ctk.CTkTabview._grid_forget_all_tabs).parameters
    assert "exclude_name" in params and params["exclude_name"].default is None, read
    view = ctk.CTkTabview(tk_root)
    view.add("First")
    view.add("Second")
    calls = []
    view._grid_forget_all_tabs = lambda exclude_name=None: calls.append(exclude_name)
    view.set("Second")
    assert calls == [], f"{read} -- set() no longer defers it"
    end = time.monotonic() + 1.0
    while time.monotonic() < end and not calls:
        tk_root.update()
        time.sleep(0.01)
    assert calls == ["Second"], f"{read} -- set() called it with {calls!r}"
    view.destroy()


def test_ctkcombobox_still_opens_its_menu_through_the_hook_model_page_py_replaces(tk_root):
    read = ("pages/model_page.py replaces CTkComboBox._open_dropdown_menu with its model picker: "
            "re-read that override against this customtkinter")
    box = ctk.CTkComboBox(tk_root, values=["one", "two"])
    opened = []
    box._open_dropdown_menu = lambda: opened.append(True)
    box._clicked()
    assert opened == [True], read
    box.destroy()


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


# ---- The Mac's mouse wheel (James, 2026-10-04: "mouse is not scolling on the app?") -------------

def test_a_wheel_notch_moves_a_scrolling_frame_a_useful_step_on_the_mac(patched_ctk, tk_root,
                                                                        monkeypatch):
    """On macOS Tk reports a wheel notch as delta 1, and CustomTkinter scrolls that many canvas
    units -- 8 px: the Overview (about 4,000 px) took some 500 notches, which reads as "not
    scrolling". On the Mac each unit is MAC_WHEEL_STEP px; on Windows (a notch is delta 120,
    CustomTkinter scrolls delta / 6 units) nothing changes."""
    import customtkinter as ctk
    monkeypatch.setattr(patched_ctk.sys, "platform", "darwin")
    mac = ctk.CTkScrollableFrame(tk_root)
    assert int(mac._parent_canvas.cget("yscrollincrement")) == patched_ctk.MAC_WHEEL_STEP == 30
    monkeypatch.setattr(patched_ctk.sys, "platform", "win32")
    windows = ctk.CTkScrollableFrame(tk_root)
    assert int(windows._parent_canvas.cget("yscrollincrement")) != patched_ctk.MAC_WHEEL_STEP
