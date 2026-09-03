"""Monkeypatches against the PINNED CustomTkinter (5.2.2, requirements-pinned.txt).

A patch against a pinned dependency is stable by construction: the version
cannot move under us without someone editing the pin, and `apply()` refuses to
patch a version it has not been read against. It is still a patch on someone
else's class, so each one below says what it changes and why a fork would be
worse. Applied once, from `V6App.__init__`.

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
"""
import logging
from typing import Optional

import customtkinter as ctk

logger = logging.getLogger(__name__)

# The version this file has been read against. See module docstring.
PINNED_CTK_VERSION = "5.2.2"

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
                   f"ctk_patches.py against the new version.")
        if strict:
            raise RuntimeError(message)
        logger.warning(message)
        return False

    _patch_scrollbar_reentrancy()
    _applied = True
    return True
