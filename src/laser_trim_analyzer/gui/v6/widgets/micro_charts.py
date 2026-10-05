"""The Overview's small charts, on plain Tk canvases (option B, 2026-10-04).

Never matplotlib for these: the Overview's list draws one per model -- dozens on one page -- and
a figure each would cost a full Agg render apiece on every load and resize.

  MiniLine    twelve months of pass % as a line, in a laser's colour, with a GAP where a month had
              no units (finish item 11: the cards' month bars read as broken). A month alone
              between gaps is a dot -- never dropped.
  PassMeter   a pass % as a row of segments: lit in the pass colour, the rest in the divider colour,
              and never all lit below 100% nor none lit above 0% (overview_data.pct_text's rule).
  MonthChart  the detail's twelve months of pass %: 0 / 50 / 100 guides, the months named under it,
              and the newest month's value said.

Every size here is in CustomTkinter's UNSCALED units and is turned into real pixels with the
master's widget scaling, as the CTk widgets beside them are (at 150% Windows scaling a 96-unit
chart is 144 px), and every text is a theme font at that scaling. Colours are theme tokens,
handed in. The y scale is always 0-100: a pass rate drawn on a stretched scale makes two points
look like a cliff.
"""
import tkinter
from typing import Callable, List, Optional, Sequence

Values = Sequence[Optional[float]]


def _scaling(master) -> float:
    try:
        return float(master._get_widget_scaling())
    except Exception:                  # a plain Tk master: no CustomTkinter scaling to follow
        return 1.0


def _xs(n: int, left: float, right: float) -> List[float]:
    if n <= 1:
        return [(left + right) / 2.0] * n
    step = (right - left) / (n - 1)
    return [left + i * step for i in range(n)]


def _y(v: float, top: float, bottom: float) -> float:
    v = max(0.0, min(100.0, float(v)))
    return bottom - (bottom - top) * v / 100.0


def _runs(values: Values):
    """(segments, alone): index pairs of neighbouring months that both have a value, and the
    months with a value but neither neighbour."""
    known = [v is not None for v in values]
    segments = [(i, i + 1) for i in range(len(values) - 1) if known[i] and known[i + 1]]
    alone = [i for i, k in enumerate(known)
             if k and not (i > 0 and known[i - 1]) and not (i + 1 < len(known) and known[i + 1])]
    return segments, alone


class _Canvas(tkinter.Canvas):
    """A plain canvas sized in CustomTkinter units, on a theme background."""

    def __init__(self, master, theme, *, bg: str, width: int, height: int):
        self.theme = theme
        self.scale = _scaling(master)
        super().__init__(master, width=round(width * self.scale), height=round(height * self.scale),
                         bg=bg, highlightthickness=0, bd=0)
        self.bind("<Configure>", lambda _e: self.draw(), add="+")

    def size(self):
        """The real size in pixels: as laid out, or as asked for before it is."""
        w, h = int(self.winfo_width()), int(self.winfo_height())
        if w <= 1 or h <= 1:                 # not laid out yet: its <Configure> follows
            return int(self.cget("width")), int(self.cget("height"))
        return w, h

    def font(self, size: int, mono: bool = True):
        t = self.theme
        f = t.mono(size) if mono else t.font(size)
        return f.create_scaled_tuple(self.scale)

    def set_background(self, bg: str) -> None:
        self.configure(bg=bg)

    def draw(self) -> None:            # each chart draws itself
        raise NotImplementedError


class MiniLine(_Canvas):
    """Twelve months (or any number) of pass %, oldest first, as a line on a 0-100 scale."""

    def __init__(self, master, theme, values: Values, *, color: str, bg: str,
                 width: int = 88, height: int = 26):
        self.values = list(values or [])
        self.color = color
        super().__init__(master, theme, bg=bg, width=width, height=height)
        self.draw()

    def draw(self) -> None:
        self.delete("all")
        if not any(v is not None for v in self.values):
            return
        w, h = self.size()
        pad = 3 * self.scale
        xs = _xs(len(self.values), pad, w - pad)
        ys = [None if v is None else _y(v, pad, h - pad) for v in self.values]
        lw = max(1.0, 1.6 * self.scale)
        segments, alone = _runs(self.values)
        for i, j in segments:
            self.create_line(xs[i], ys[i], xs[j], ys[j], fill=self.color, width=lw,
                             capstyle="round", tags=("line",))
        r = max(1.5, 1.8 * self.scale)
        for i in alone:
            self.create_oval(xs[i] - r, ys[i] - r, xs[i] + r, ys[i] + r, fill=self.color,
                             outline="", tags=("dot",))


class PassMeter(_Canvas):
    """A pass % as `segments` blocks: lit in PASS_FG from the left, the rest DIVIDER."""

    def __init__(self, master, theme, pct: Optional[float], *, bg: str, segments: int = 20,
                 width: int = 220, height: int = 14):
        self.pct = pct
        self.segments = segments
        super().__init__(master, theme, bg=bg, width=width, height=height)
        self.draw()

    @staticmethod
    def lit(pct: Optional[float], segments: int) -> int:
        """How many segments a pass % lights: its share, rounded -- but never all of them below
        100% and never none above 0% (a 99.6% meter that reads full is a 100% claim)."""
        if pct is None:
            return 0
        n = round(max(0.0, min(100.0, pct)) / 100.0 * segments)
        if n >= segments and pct < 100:
            n = segments - 1
        if n <= 0 < pct:
            n = 1
        return n

    def set_value(self, pct: Optional[float]) -> None:
        self.pct = pct
        self.draw()

    def draw(self) -> None:
        t = self.theme
        self.delete("all")
        w, h = self.size()
        n = self.segments
        gap = max(1.0, 2 * self.scale)
        seg = max(1.0, (w - gap * (n - 1)) / n)
        lit = self.lit(self.pct, n)
        for i in range(n):
            x0 = i * (seg + gap)
            on = i < lit
            self.create_rectangle(x0, 0, x0 + seg, h, width=0, fill=t.PASS_FG if on else t.DIVIDER,
                                  tags=("segment", "on" if on else "off"))


class MonthChart(_Canvas):
    """Twelve months of pass %, oldest first: 0/50/100 guides, the months named under them, a line
    with a dot on every month that had units (a gap where one had none), and the newest value."""

    GUIDES = (0, 50, 100)
    EMPTY = "No graded units in these twelve months"

    def __init__(self, master, theme, *, bg: str, width: int = 480, height: int = 120):
        self.values: List[Optional[float]] = []
        self.labels: List[str] = []
        self.color = theme.CHART_REFERENCE
        self.value_text: Callable[[float], str] = lambda v: f"{round(v)}%"
        super().__init__(master, theme, bg=bg, width=width, height=height)

    def set_data(self, values: Values, labels: Sequence[str], color: str,
                 value_text: Optional[Callable[[float], str]] = None) -> None:
        self.values = list(values or [])
        self.labels = list(labels or [])
        self.color = color
        if value_text is not None:
            self.value_text = value_text
        self.draw()

    def draw(self) -> None:
        t = self.theme
        self.delete("all")
        w, h = self.size()
        small = self.font(t.SIZE_CAPTION)
        words = self.font(t.SIZE_CAPTION, mono=False)
        line_h = self._height_of(small)
        left = self._width_of(small, "100%") + 8 * self.scale
        right = w - self._width_of(small, "100%") / 2 - 4 * self.scale
        top = line_h + 4 * self.scale
        bottom = h - line_h - 6 * self.scale
        for g in self.GUIDES:
            y = _y(g, top, bottom)
            self.create_line(left, y, w, y, fill=t.DIVIDER, tags=("guide",))
            self.create_text(left - 6 * self.scale, y, text=f"{g}%", anchor="e", font=small,
                             fill=t.TEXT_DISABLED, tags=("ylabel",))
        xs = _xs(max(len(self.values), len(self.labels)), left + 6 * self.scale, right)
        if self.labels:
            widest = max(self._width_of(words, s) for s in self.labels)
            step = 1 if len(xs) < 2 or xs[1] - xs[0] >= widest + 6 * self.scale else 2
            last = len(self.labels) - 1
            for i, text in enumerate(self.labels):
                if (last - i) % step == 0:           # always the newest month
                    self.create_text(xs[i], h - 2 * self.scale, text=text, anchor="s", font=words,
                                     fill=t.TEXT_SECONDARY, tags=("xlabel",))
        if not any(v is not None for v in self.values):
            self.create_text((left + w) / 2, (top + bottom) / 2, text=self.EMPTY, font=words,
                             fill=t.TEXT_SECONDARY, tags=("note",))
            return
        ys = [None if v is None else _y(v, top, bottom) for v in self.values]
        segments, _alone = _runs(self.values)
        for i, j in segments:
            self.create_line(xs[i], ys[i], xs[j], ys[j], fill=self.color,
                             width=max(1.0, 2 * self.scale), capstyle="round", tags=("line",))
        r = max(2.0, 2.6 * self.scale)
        for i, y in enumerate(ys):
            if y is not None:
                self.create_oval(xs[i] - r, y - r, xs[i] + r, y + r, fill=self.color, outline="",
                                 tags=("dot",))
        newest = max(i for i, v in enumerate(self.values) if v is not None)
        self.create_text(xs[newest], ys[newest] - r - 2 * self.scale,
                         text=self.value_text(self.values[newest]), anchor="s", font=small,
                         fill=t.TEXT_PRIMARY, tags=("value",))

    def _width_of(self, font, text: str) -> float:
        return float(self.tk.call("font", "measure", font, text))

    def _height_of(self, font) -> float:
        return float(self.tk.call("font", "metrics", font, "-linespace"))
