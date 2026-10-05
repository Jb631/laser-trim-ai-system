"""Monkeypatches against the PINNED CustomTkinter (5.2.2, requirements-pinned.txt).

A patch against a pinned dependency is stable by construction: the version
cannot move under us without someone editing the pin, and `apply()` refuses to
patch a version it has not been read against. It is still a patch on someone
else's class, so each one below says what it changes and why a fork would be
worse. Applied once, from `V6App.__init__`.

---- The pin covers more than this file ------------------------------------

Three more places override PRIVATE CustomTkinter code, in the app's own classes rather than here,
and are right only while 5.2.2's internals mean what they were read to mean (F4 review, 2026-09-25).
A pin bump must re-read them too -- the version-mismatch message below names all three, and
tests/test_ui_responsiveness.py (the first two) and tests/test_finish_pass_pages.py (the third)
fail, naming the file, if an internal changes:

  * widgets/tab_view.py -- `ThemedTabView._grid_forget_all_tabs` overrides CTkTabview's: it relies on
    that method taking `exclude_name`, and on `CTkTabview.set()` calling it 100 ms later with the
    name it switched to (ctk_tabview.py set()).
  * pages/model_page.py -- replaces the model selector's `CTkComboBox._open_dropdown_menu` (the
    method `_clicked` calls) with the searchable model picker.
  * widgets/blocks.py -- `_QuietArrow._draw` (every dropdown and combo box, finish pass 2026-10-04)
    repaints the "dropdown_arrow" canvas item after CTkOptionMenu's / CTkComboBox's own `_draw`,
    which paints it with text_color.
  * pages/home_page.py (the Overview, option B 2026-10-04) -- `_PageScroll.check_if_master_is_canvas`
    narrows CTkScrollableFrame's wheel routing so a pane that can scroll keeps the wheel; and
    `_scrollbar_only_when_needed` wraps the canvas's yscrollcommand (5.2.2 pins it to the bar's set()),
    reading `_scrollbar`, `_parent_canvas` and `_parent_frame`.

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
from typing import Optional

import customtkinter as ctk

logger = logging.getLogger(__name__)

# The version this file has been read against. See module docstring.
PINNED_CTK_VERSION = "5.2.2"

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
        message = (f"customtkinter {version} is not the pinned "
                   f"{PINNED_CTK_VERSION}; UI patches NOT applied. Re-read "
                   f"ctk_patches.py against the new version -- and the app's two other "
                   f"private overrides: widgets/tab_view.py (ThemedTabView."
                   f"_grid_forget_all_tabs, which relies on CTkTabview.set() deferring it "
                   f"with exclude_name), pages/model_page.py (the model selector's "
                   f"CTkComboBox._open_dropdown_menu, replaced by the model picker) and "
                   f"widgets/blocks.py (_QuietArrow._draw, which repaints the dropdowns' "
                   f"\"dropdown_arrow\" canvas item after CustomTkinter's own _draw).")
        if strict:
            raise RuntimeError(message)
        logger.warning(message)
        return False

    _patch_scrollbar_reentrancy()
    _patch_mac_wheel_step()
    _applied = True
    return True
