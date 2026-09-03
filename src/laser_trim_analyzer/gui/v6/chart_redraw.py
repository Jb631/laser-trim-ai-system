"""One chart render per resize, instead of one per <Configure> event.

matplotlib's Tk backend binds `<Configure>` to `resize()`, which resizes the
figure and calls `draw_idle()` (_backend_tk.py:284-316, pinned 3.11.1).
`draw_idle` coalesces via `after_idle`, which would be enough on its own — but
CustomTkinter drains the idle queue constantly (every scrollbar redraw calls
`update_idletasks`; see gui/v6/ctk_patches.py), so each pending idle draw is
cashed in almost immediately. The result is a full Agg render per configure
event: `ui_stall_probe.py` sampled the Tk thread inside `axis.draw` /
`text.draw` / `backend_tkagg.blit` on every resize.

So: while configure events keep arriving, cancel the pending idle draw and
re-arm a single one `QUIET_MS` after the last of them. A drag renders once, at
the end, at the size it ended on.

What this deliberately does NOT touch:
  * `canvas.draw()` — a synchronous, blocking render. The export paths and
    `scripts/chart_qa_render_all.py` call it and then read pixels, so it must
    keep rendering immediately, and nothing here gates it.
  * `focus_list_zone`'s sparklines. Those canvases are a fixed 260px, never
    resize with the window, and are destroyed on every refresh — an `after`
    aimed at one is the dead-widget Tcl error that widget's comment already
    warns about.

The logic lives on `ChartRedrawDebounce` as plain methods rather than inside
closures so that a test can drive `on_configure()` and `render_now()` directly.
That is not cosmetic: the first version of these tests generated real
<Configure> events into a mapped CTkToplevel and spun the Tk event loop
forever inside a full pytest session (fine in isolation, a hung suite at test
~1340). Driving the two methods is deterministic, needs no mapped window, and
still fails if the debounce is removed — the binding itself is asserted
separately.
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Long enough that a drag's configure stream never gets between two of them,
# short enough that letting go of the mouse feels like it redrew at once.
QUIET_MS = 120


class ChartRedrawDebounce:
    """Coalesces one matplotlib canvas's resize-driven redraws."""

    def __init__(self, canvas: Any, quiet_ms: int = QUIET_MS) -> None:
        self.canvas = canvas
        self.widget = canvas.get_tk_widget()
        self.quiet_ms = quiet_ms
        self.pending: Optional[str] = None
        self.renders = 0

    def _cancel(self, after_id: Optional[str]) -> None:
        if after_id:
            try:
                self.widget.after_cancel(after_id)
            except Exception:
                pass          # already fired, or the widget is gone

    def render_now(self) -> None:
        """The deferred render. Runs `quiet_ms` after the last configure."""
        self.pending = None
        try:
            if not self.widget.winfo_exists():
                return
            self.renders += 1
            self.canvas.draw_idle()
        except Exception:
            logger.debug("deferred chart redraw failed", exc_info=True)

    def on_configure(self, _event=None) -> None:
        """Withdraw the render this configure event just armed; re-arm one."""
        # matplotlib's own pending render, armed by resize() a moment ago.
        # `_idle_draw_id` is private, hence the getattr and the pinned version:
        # public API offers no way to withdraw an idle draw once armed.
        self._cancel(getattr(self.canvas, "_idle_draw_id", None))
        self.canvas._idle_draw_id = None
        self._cancel(self.pending)
        self.pending = self.widget.after(self.quiet_ms, self.render_now)


def debounce_resize_redraws(canvas: Any, quiet_ms: int = QUIET_MS) -> ChartRedrawDebounce:
    """Coalesce a canvas's resize-driven redraws. Returns the state object.

    Bind order matters and is not an accident: matplotlib binds `<Configure>`
    in `FigureCanvasTkAgg.__init__`, so binding here with `add="+"` runs
    `on_configure` AFTER `resize()` has already armed the idle draw it cancels.
    """
    debounce = ChartRedrawDebounce(canvas, quiet_ms)
    debounce.widget.bind("<Configure>", debounce.on_configure, add="+")
    return debounce
