"""Monkeypatches against the PINNED CustomTkinter (5.2.2, requirements-pinned.txt).

A patch against a pinned dependency is stable by construction: the version
cannot move under us without someone editing the pin, and `apply()` refuses to
patch a version it has not been read against. It is still a patch on someone
else's class, so each one below says what it changes and why a fork would be
worse. Applied once, from `V6App.__init__`.

---- The pin covers more than this file ------------------------------------

Other places override, wrap or read PRIVATE CustomTkinter code, in the app's own classes rather
than here, and are right only while 5.2.2's internals mean what they were read to mean (F4 review,
2026-09-25). A pin bump must re-read every one. `PRIVATE_USES` below names each, with the private
names it rests on; the version-mismatch message is built from it, so it cannot leave one out again
(option B review, 2026-10-04: this note counted "three" over four places, and the message named
three); tests/test_ui_responsiveness.py checks that this list, the message and the files agree.
What each one relies on is pinned by its own tests -- test_ui_responsiveness (tab_view's deferred
forget, model_page's hook), test_finish_pass_pages (the arrows' colour, the tabs' width),
test_spec3f_home (the Overview's wheel), test_stats_table and test_spec3c_model (a live change of
scaling):

  * widgets/tab_view.py -- `ThemedTabView._grid_forget_all_tabs` overrides CTkTabview's: it relies on
    that method taking `exclude_name`, and on `CTkTabview.set()` calling it 100 ms later with the
    name it switched to (ctk_tabview.py set()). The class also sets CTkTabview's class-level
    `_button_height` (34 px: CTkTabview's grid and its strip both read it), and reaches into the
    strip -- `_segmented_button` and its `_buttons_dict` -- for the tabs' font and each tab's
    width (finish pass, 2026-10-04).
  * pages/model_page.py -- replaces the model selector's `CTkComboBox._open_dropdown_menu` (the
    method `_clicked` calls) with the searchable model picker; and scrolls a tab to the part a
    link names, or Summary back to its chart, through a CTkScrollableFrame's own canvas,
    `_parent_canvas` (inside a try: a rename there would quietly stop the scroll, not raise).
  * widgets/blocks.py -- `_QuietArrow._draw` (every dropdown and combo box, finish pass 2026-10-04)
    repaints the "dropdown_arrow" item on the widget's `_canvas` after CTkOptionMenu's /
    CTkComboBox's own `_draw`, which paints it with text_color.
  * pages/home_page.py (the Overview, option B 2026-10-04) -- `_PageScroll.check_if_master_is_canvas`
    narrows CTkScrollableFrame's wheel routing so a pane that can scroll keeps the wheel; and
    `_scrollbar_only_when_needed` wraps the canvas's yscrollcommand (5.2.2 pins it to the bar's set()),
    reading `_scrollbar`, `_parent_canvas` and `_parent_frame`.
  * widgets/stats_table.py and widgets/drift_metrics_tab.py -- each overrides `_set_scaling`, the
    method CustomTkinter calls when the display scaling changes live, to re-apply its scaled column
    minimums (re-review, 2026-09-25); stats_table also raises its row bands above a frame's own
    `_canvas`.
  * Throughout gui/v6: CustomTkinter's unit helpers, `_apply_widget_scaling`,
    `_reverse_widget_scaling` and `_get_widget_scaling`, turn its unscaled units into real pixels
    and back (blocks.wrap_to_width says why). The text-wrapping callers sit inside a try, so a
    rename there would quietly stop a wrap rather than raise.

---- 1. CTkScrollbar's re-entrant redraw cascade ----------------------------

`CTkScrollbar._draw()` ends with `self._canvas.update_idletasks()`
(ctk_scrollbar.py:161 in 5.2.2). That call does not merely repaint the
scrollbar: it drains the WHOLE application's idle queue, which re-runs pending
geometry management, which fires `<Configure>` on every CTk widget waiting for
one, which runs `CTkBaseClass._update_dimensions_event -> _draw`, which moves a
scroll region, which calls the scrollbar's `set()` -> `_draw()` ->
`update_idletasks()` again. `scripts/ui_stall_probe.py` sampled the main thread
46 levels deep in exactly that loop, and cProfile counted 41,039 Tk event
callbacks dispatched into Python for ONE model switch — 542,449 `getint` calls
just parsing the event structs.

The fix is the smallest thing that breaks the loop: while a scrollbar redraw is
already in progress anywhere in the app, a nested redraw still draws, but does
not pump the idle queue a second time. The outermost `_draw` pumps exactly as
before, so anything that relied on "after set(), the scrollbar is painted"
still holds; only the recursion is gone.

Suppression is done by shadowing `update_idletasks` on the canvas instance for
the duration of the nested call, rather than by reimplementing `_draw`. That
keeps CustomTkinter's drawing code — the part that actually matters and the
part that changes between releases — untouched.

---- 2. The Mac's mouse wheel moved a scrolling frame 8 px a notch ----------

James, 2026-10-04: "mouse is not scolling on the app?" On macOS, Tk 8.6 reports a wheel
notch as `delta` 1 (more only when spun fast), and `CTkScrollableFrame._mouse_wheel_all`
scrolls `-delta` canvas UNITS -- 8 px each: the Overview, about 4,000 px tall, took some 500
notches, which reads as not scrolling at all. On Windows a notch is `delta` 120 and CustomTkinter
scrolls `delta / 6` = 20 units, which is why it was never seen at work. So on the Mac only, every
scrolling frame's canvas steps MAC_WHEEL_STEP px per unit; Windows keeps CustomTkinter's own.
Set at construction (the platform read then, not at import), so nothing else about the frame,
its scrollbar or a programmatic `yview_moveto` changes.
"""
import logging
import sys
from typing import Optional, Tuple

import customtkinter as ctk

logger = logging.getLogger(__name__)

# The version this file has been read against. See module docstring.
PINNED_CTK_VERSION = "5.2.2"

# Every place OUTSIDE this file that rests on CustomTkinter's private code (the module docstring
# says what each relies on): its file under gui/v6, the app's own class or function there, and the
# private names it uses. apply() names them all from here when the version moves.
PRIVATE_USES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("widgets/tab_view.py", "ThemedTabView",
     ("_grid_forget_all_tabs", "_button_height", "_segmented_button", "_buttons_dict")),
    ("pages/model_page.py", "ModelPage, _scroll_into_view",
     ("_open_dropdown_menu", "_parent_canvas")),
    ("widgets/blocks.py", "_QuietArrow",
     ("_draw", "_canvas", "dropdown_arrow")),
    ("pages/home_page.py", "_PageScroll, _scrollbar_only_when_needed",
     ("check_if_master_is_canvas", "_scrollbar", "_parent_canvas", "_parent_frame")),
    ("widgets/stats_table.py", "_TableGrid, StatsTableZone",
     ("_set_scaling", "_canvas")),
    ("widgets/drift_metrics_tab.py", "_Columns",
     ("_set_scaling",)),
)
# ...and called throughout gui/v6, so named in the message without a file.
UNIT_HELPERS: Tuple[str, ...] = ("_apply_widget_scaling", "_reverse_widget_scaling",
                                 "_get_widget_scaling")

# Pixels a scrolling frame moves per wheel unit on the Mac (patch 2): a notch, 30 px; a trackpad
# swipe, the same per unit it reports.
MAC_WHEEL_STEP = 30

_applied = False


class _ScrollbarRedraw:
    """Whether a CTkScrollbar redraw is on the stack, app-wide.

    App-wide and not per-widget on purpose: the cascade jumps between
    scrollbars (a scrollable frame's redraw resizes a sibling, whose scrollbar
    then calls `set`), so a per-instance flag would not close the loop.
    """

    in_progress = False


def _noop() -> None:
    """Stands in for `Canvas.update_idletasks` during a nested redraw."""


def _patch_scrollbar_reentrancy() -> None:
    original_draw = ctk.CTkScrollbar._draw

    def _draw(self, no_color_updates=False):
        if _ScrollbarRedraw.in_progress:
            # Nested: draw, but do not drain the idle queue again. An instance
            # attribute shadows Misc.update_idletasks for the duration. Restore
            # whatever was there rather than deleting: production has nothing,
            # but a test (or a future wrapper) may have installed its own, and
            # a blind delete would silently throw it away.
            missing = object()
            previous = self._canvas.__dict__.get("update_idletasks", missing)
            self._canvas.update_idletasks = _noop
            try:
                return original_draw(self, no_color_updates)
            finally:
                if previous is missing:
                    self._canvas.__dict__.pop("update_idletasks", None)
                else:
                    self._canvas.update_idletasks = previous

        _ScrollbarRedraw.in_progress = True
        try:
            return original_draw(self, no_color_updates)
        finally:
            # finally, not a plain reset: an exception in a redraw must not
            # leave every later scrollbar permanently un-pumped.
            _ScrollbarRedraw.in_progress = False

    _draw.__doc__ = original_draw.__doc__
    ctk.CTkScrollbar._draw = _draw


def _patch_mac_wheel_step() -> None:
    original_init = ctk.CTkScrollableFrame.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if sys.platform == "darwin":
            self._parent_canvas.configure(yscrollincrement=MAC_WHEEL_STEP)

    __init__.__doc__ = original_init.__doc__
    ctk.CTkScrollableFrame.__init__ = __init__


def _version_mismatch(version: Optional[str]) -> str:
    """What apply() says on a CustomTkinter other than the pinned one: every place to re-read,
    from PRIVATE_USES -- so a place added to the list is named here without anyone remembering to."""
    places = "; ".join(f"{path} ({where}: {', '.join(names)})"
                       for path, where, names in PRIVATE_USES)
    return (f"customtkinter {version} is not the pinned {PINNED_CTK_VERSION}; UI patches NOT "
            f"applied. Re-read ctk_patches.py against the new version (its docstring says what "
            f"each place relies on) -- and every other place that rests on CustomTkinter's "
            f"private code: {places}; and the unit helpers {', '.join(UNIT_HELPERS)}, called "
            f"throughout gui/v6.")


def apply(strict: bool = False) -> bool:
    """Install the patches. Idempotent; safe to call from every V6App.

    Returns True when the patches are in place. On a CustomTkinter other than
    the pinned one, logs and does nothing (the app still runs, just slower)
    unless `strict`, which raises — used by the tests, so a dependency bump
    fails loudly in CI instead of silently un-patching the UI.
    """
    global _applied
    if _applied:
        return True

    version: Optional[str] = getattr(ctk, "__version__", None)
    if version != PINNED_CTK_VERSION:
        message = _version_mismatch(version)
        if strict:
            raise RuntimeError(message)
        logger.warning(message)
        return False

    _patch_scrollbar_reentrancy()
    _patch_mac_wheel_step()
    _applied = True
    return True
