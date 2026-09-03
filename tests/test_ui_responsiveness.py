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
