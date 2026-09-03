"""Deferred teardown: stop paying the OLD model's widget funeral in front of you.

`scripts/ui_stall_probe.py` (2026-09-02) measured a 1.5 s freeze when James
switched from one model to another on INVESTIGATE, and sampled the Tk thread
17 times out of 17 inside `ctk_label.destroy <- ctk_base_class.destroy`. Not
the query, not the chart, not the build: the app was destroying the PREVIOUS
model's widgets, and the click that started the switch had to wait for all of
them.

Why destroying is so expensive here — three O(n) list scans per widget, in the
pinned customtkinter 5.2.2:

  * `CTkLabel.destroy` -> `CTkFont.remove_size_configure_callback`, a
    `list.remove` over the shared font's callback list (every widget of that
    size is on it);
  * `CTkAppearanceModeBaseClass.destroy` -> `AppearanceModeTracker.remove`,
    a `list.remove` over every CTk widget alive in the process;
  * `CTkScalingBaseClass.destroy` -> `ScalingTracker.remove_widget`, which
    walks `winfo_parent` up to the toplevel first.

...and one CTkLabel is three native widgets (frame + canvas + label), so a
model switch is ~1,500 native destroys with quadratic bookkeeping on top. That
floor is not ours to move. What IS ours: WHEN it gets paid. Nobody is waiting
for the old model's labels to die — they are already off screen the moment the
new ones are drawn.

So `retire()` splits the two halves of a rebuild:

  1. unmap the old container NOW (`pack_forget`/`grid_forget`), which is what
     the eye actually reads as "it switched", and costs one geometry pass;
  2. destroy the subtree in small slices between event-loop passes — starting
     only once the NEW content's own geometry pass is done (`FIRST_SLICE_MS`)
     — so the thread is free for a click, a keystroke or a redraw between any
     two of them.

Total work is unchanged. Worst-case latency is what changes, and latency is
what a hand feels.

TIMER SLICES, NOT `after_idle`. CustomTkinter drains the idle queue constantly
(every scrollbar redraw calls `update_idletasks` — see `gui/v6/ctk_patches.py`
and the `chart_redraw` docstring), so an idle-queued slice can fire INSIDE the
build it was supposed to follow, destroying widgets while Tk is mid-geometry
pass. `after(1)` is a timer event: it only runs from a real event-loop pass.

RAPID SWITCHES. Every `retire()` call owns its own queue and its own timer
chain, and the caller always builds into a NEW container, so a second switch
while the first is still draining simply leaves two drains running. Nothing is
reused, nothing is shared, and no drain can be "interrupted" into leaving
widgets alive forever: the only ways a slice stops are an empty queue or a
dead toplevel (which destroyed the widgets itself).

Main thread only — this is Tk scheduling, so it obeys CLAUDE.md rule 5 by
construction: workers post to `safe_after`, and the apply they post is what
calls this.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Native widgets per slice. Sized against the measured cost inside the running
# app: a 25-node slice takes 18 ms at worst (timed over 228 of them across
# three model switches), well inside one 60 ms frame even when a slice
# overshoots by one atomic subtree (see `step`).
BATCH_NODES = 25

# How long the FIRST slice waits. Not zero, and this is measured, not taste: a
# rebuild is followed by Tk's own geometry pass over the new content —
# CTkScrollbar.set plus the nested CTkBaseClass._update_dimensions_event,
# together ~165 ms on the Model page — and the first two slices used to land
# inside it, adding themselves to the very stall the burial was moved out of.
# Let the new content finish arriving; the old widgets are in no hurry.
FIRST_SLICE_MS = 200

# ...and between slices. A timer, not an idle callback — see the module
# docstring. Tcl services idle handlers only when no other event is ready, and
# geometry and redraw are idle work, so a 1 ms chain would hold the display's
# turn down to a sliver. 8 ms leaves it a third of the time and still drains a
# ~1,500-widget model page inside two seconds.
SLICE_MS = 8


def _alive(widget: Any) -> bool:
    try:
        return bool(widget.winfo_exists())
    except Exception:
        return False


def _children(widget: Any) -> List[Any]:
    """Child widgets, from Tkinter's own dict — no Tcl round trip.

    `Misc.children` is what `Misc.destroy` itself walks, so it sees exactly
    what a destroy would recurse into, including a CTk widget's internal
    canvas.
    """
    return list(getattr(widget, "children", {}).values())


def _subtree_size(widget: Any) -> int:
    """Native widgets a `destroy()` of this widget would take down."""
    total = 1
    for child in _children(widget):
        total += _subtree_size(child)
    return total


def _unmap(widget: Any) -> None:
    """Take the widget out of the layout without destroying it.

    This is the half of a teardown the user can see, and it is O(1)-ish: one
    geometry pass on the parent. Everything after it is bookkeeping.
    """
    try:
        manager = widget.winfo_manager()
    except Exception:
        return
    try:
        if manager == "pack":
            widget.pack_forget()
        elif manager == "grid":
            widget.grid_forget()
        elif manager == "place":
            widget.place_forget()
    except Exception:
        logger.debug("retire: could not unmap %r", widget, exc_info=True)


class Retirement:
    """One deferred teardown: some widgets, unmapped now and destroyed later.

    Written as plain methods rather than a closure so a test can drive `step()`
    directly and assert the work really is sliced — the same reason
    `ChartRedrawDebounce` is a class.
    """

    def __init__(self, widgets, *, batch: int = BATCH_NODES,
                 host: Optional[Any] = None) -> None:
        # (widget, expanded): `expanded` marks a container whose children are
        # already queued, so the entry is just its shell.
        self._queue: List[Tuple[Any, bool]] = [(w, False) for w in widgets]
        self.batch = max(1, int(batch))
        self.destroyed = 0
        self.slices = 0
        # Scheduled on the TOPLEVEL, never on a widget being retired: a pending
        # `after` is registered against the widget that armed it, and that
        # widget's destroy deletes the Tcl command out from under it.
        self.host = host if host is not None else self._find_host(widgets)

    @staticmethod
    def _find_host(widgets) -> Optional[Any]:
        for widget in widgets:
            try:
                return widget.winfo_toplevel()
            except Exception:
                continue
        return None

    @property
    def done(self) -> bool:
        return not self._queue

    @property
    def pending(self) -> int:
        """Queued entries — not widgets. Diagnostics only."""
        return len(self._queue)

    def unmap(self) -> None:
        """Drop every root out of the layout, now. The visible half."""
        for widget, _ in self._queue:
            if _alive(widget):
                _unmap(widget)

    def _destroy(self, widget: Any, size: int) -> int:
        try:
            widget.destroy()
        except Exception:
            # A widget can die under us (its parent was destroyed, the window
            # closed); the queue is not a promise that it is still there.
            logger.debug("retire: destroy failed for %r", widget, exc_info=True)
        self.destroyed += size
        return size

    def step(self) -> int:
        """Destroy up to `batch` native widgets. Returns how many it did.

        A subtree that fits in a slice is destroyed whole — one `destroy()`
        call, exactly as the synchronous code would have made it. Only a
        subtree BIGGER than a whole slice is split, and then its children go
        first and its shell last, so no parent is ever left drawing into a
        canvas its own child list no longer has. (A CTk frame paints on
        `self._canvas`; that canvas is deliberately left with the shell.)
        """
        spent = 0
        self.slices += 1
        while self._queue and spent < self.batch:
            widget, expanded = self._queue.pop()
            if not _alive(widget):
                continue
            size = _subtree_size(widget)
            if expanded or size <= self.batch:
                spent += self._destroy(widget, size)
                continue
            self._queue.append((widget, True))
            internal = getattr(widget, "_canvas", None)
            self._queue.extend((child, False) for child in _children(widget)
                               if child is not internal)
        return spent

    def finish(self) -> int:
        """Drain the rest synchronously. The fallback and the test path."""
        total = 0
        while self._queue:
            total += self.step()
        return total

    def schedule(self, delay_ms: Optional[int] = None) -> None:
        """Arm the next slice, or finish now if there is nothing to arm it on."""
        if self.done:
            return
        if self.host is None or not _alive(self.host):
            # No live toplevel to hang a timer on (a bare widget in a test, or
            # a window on its way out). Better a synchronous teardown than a
            # queue nobody will ever drain.
            self.finish()
            return
        try:
            self.host.after(SLICE_MS if delay_ms is None else delay_ms, self._slice)
        except Exception:
            logger.debug("retire: could not schedule a slice", exc_info=True)
            self.finish()

    def _slice(self) -> None:
        if not _alive(self.host):
            return          # the toplevel went away, taking the widgets with it
        self.step()
        self.schedule()


def retire(*widgets: Any, batch: int = BATCH_NODES) -> Optional[Retirement]:
    """Unmap these widgets now; destroy them in idle-time slices.

    The drop-in replacement for a `for w in old: w.destroy()` loop in any
    rebuild-on-reload path. Returns the `Retirement` (for tests and for a
    caller that wants to force `finish()`), or None if there was nothing live
    to retire.

    The caller must have already dropped its own references to these widgets
    and must build the replacement into a NEW container: a retired widget
    lives on, unmapped, for a moment, and re-using one would put the old
    content back on the page.
    """
    live = [w for w in widgets if w is not None and _alive(w)]
    if not live:
        return None
    retirement = Retirement(live, batch=batch)
    retirement.unmap()
    retirement.schedule(FIRST_SLICE_MS)
    return retirement
