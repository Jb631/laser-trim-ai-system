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
