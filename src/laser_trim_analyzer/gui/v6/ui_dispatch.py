"""Main-thread UI dispatch for V6 (2026-07-06 freeze fix).

Tkinter is not thread-safe: calling widget methods — including after() and
winfo_exists() — from a worker thread can stall or deadlock, especially when
the main thread is simultaneously blocked (e.g. waiting on the DB lock).
The old safe_after() did exactly that.

This module removes every cross-thread Tk call:
  * workers call UiDispatcher.post(fn) — a pure-Python queue put, always safe
  * the ROOT window polls the queue from the main loop and runs callbacks
    there, where Tk calls (winfo_exists, configure, after) are legal

Usage:
    dispatcher = UiDispatcher()
    dispatcher.attach(root)          # main thread, once
    dispatcher.post(lambda: ...)     # any thread

`post_ui(app, fn)` is a convenience for widgets that hold an `app` reference:
it routes through app.ui when present and falls back to calling fn directly
(correct in tests, which drive everything from the main thread).
"""
from __future__ import annotations

import logging
import queue
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_POLL_MS = 33          # ~30 Hz; imperceptible latency, negligible idle cost
_MAX_PER_TICK = 100    # bound work per tick so the UI never starves


class UiDispatcher:
    def __init__(self) -> None:
        self._q: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._root = None
        self._polling = False

    # ---- worker side (any thread) ----
    def post(self, fn: Callable[[], None]) -> None:
        """Queue fn to run on the UI thread. Never touches Tk. Never blocks."""
        self._q.put(fn)

    # ---- main-thread side ----
    def attach(self, root) -> None:
        """Start draining the queue from `root`'s event loop. Main thread only."""
        self._root = root
        if not self._polling:
            self._polling = True
            self._schedule()

    def _schedule(self) -> None:
        try:
            self._root.after(_POLL_MS, self._drain)
        except Exception:
            self._polling = False  # root destroyed — stop politely

    def _drain(self) -> None:
        for _ in range(_MAX_PER_TICK):
            try:
                fn = self._q.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception:
                # A UI callback must never kill the poll loop.
                logger.exception("UI dispatch callback failed")
        self._schedule()


def resolve_dispatcher(widget) -> Optional[UiDispatcher]:
    """Find the app's UiDispatcher from any widget by walking the master chain.

    Call on the MAIN thread (at construction time), never from a worker —
    attribute walking is cheap but the point is to capture the dispatcher
    before any thread needs it. Returns None when no dispatcher exists
    (tests / standalone widgets), in which case callers run inline.
    """
    w = widget
    while w is not None:
        ui = getattr(w, "ui", None)
        if ui is not None and hasattr(ui, "post"):
            return ui
        w = getattr(w, "master", None)
    return None


def post_ui(app, fn: Callable[[], None]) -> None:
    """Run fn on the UI thread via app.ui when available.

    Fallback runs fn inline — correct for tests and any context that is
    already on the main thread.
    """
    dispatcher: Optional[UiDispatcher] = getattr(app, "ui", None)
    if dispatcher is not None:
        dispatcher.post(fn)
    else:
        try:
            fn()
        except Exception:
            logger.exception("UI callback failed (inline fallback)")
