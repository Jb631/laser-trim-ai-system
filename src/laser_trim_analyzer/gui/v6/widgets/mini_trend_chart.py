"""Dashboard — MiniTrendChart: a compact pass-rate-over-time line (no SPC overlays)."""
from typing import List, Tuple

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class MiniTrendChart(ctk.CTkFrame):
    _MAX_POINTS = 48

    @staticmethod
    def _downsample(ys: List[float], max_points: int) -> List[float]:
        """Average ys into at most max_points evenly spaced buckets (order-preserving)."""
        n = len(ys)
        if n <= max_points:
            return ys
        out = []
        for i in range(max_points):
            lo = i * n // max_points
            hi = (i + 1) * n // max_points
            chunk = ys[lo:hi] or ys[lo:lo + 1]
            out.append(sum(chunk) / len(chunk))
        return out

    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color=theme.CARD, corner_radius=theme.RADIUS_SM, **kwargs)
        self.theme = theme
        self._fig = Figure(figsize=(3.2, 1.0), dpi=96, facecolor=theme.CARD)
        self._ax = self._fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self._fig, master=self)   # NOTE: `canvas`, not `_canvas`
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.bind("<Destroy>", self._on_destroy)

    def set_points(self, points: List[Tuple[str, float]],
                   label: str = "daily linearity yield") -> None:
        """A sparkline with no dates, scale, or label means nothing (user
        finding, 2026-07-08). Minimum context now drawn: what the line IS,
        first/last date, and the latest value."""
        ax, t = self._ax, self.theme
        ax.clear()
        ax.set_facecolor(t.CARD)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if points:
            ys = [p[1] for p in points]
            # Bound the point count: per-day pass-rate over a long window (e.g. the
            # 'All' view spanning years) plots thousands of jagged points into a 3"
            # sparkline — an unreadable solid block. Average into <=_MAX_POINTS evenly
            # spaced buckets so the trend stays legible at any window length.
            ys_ds = self._downsample(ys, self._MAX_POINTS)
            ax.plot(range(len(ys_ds)), ys_ds, lw=1.5, color=t.ACCENT)
            ax.set_ylim(-14, 112)   # headroom for the labels below/above
            # Latest value, marked and named — the number the eye wants.
            ax.plot([len(ys_ds) - 1], [ys_ds[-1]], marker="o", ms=4, color=t.ACCENT_HOVER)
            ax.annotate(f"{ys[-1]:.0f}%", (len(ys_ds) - 1, ys_ds[-1]),
                        textcoords="offset points", xytext=(-2, 6), ha="right",
                        fontsize=7, color=t.TEXT_PRIMARY)
            # What it is + when it spans.
            first_d, last_d = str(points[0][0]), str(points[-1][0])
            ax.text(0.0, -0.02, f"{first_d}", transform=ax.transAxes, ha="left",
                    va="top", fontsize=6.5, color=t.TEXT_DISABLED)
            ax.text(1.0, -0.02, f"{last_d}", transform=ax.transAxes, ha="right",
                    va="top", fontsize=6.5, color=t.TEXT_DISABLED)
            ax.text(0.0, 1.02, f"{label} (0–100%)", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=6.5, color=t.TEXT_DISABLED)
        else:
            ax.text(0.5, 0.5, "no trend", transform=ax.transAxes, ha="center", va="center",
                    color=t.TEXT_DISABLED, fontsize=8)
        self._fig.tight_layout(pad=0.35)
        self.canvas.draw_idle()

    def _on_destroy(self, _evt=None):
        try:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
        except Exception:
            pass
