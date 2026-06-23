"""Spec 3c — FocusChart: one metric's time series with Rule-1 SPC overlays (Q7)."""
from datetime import datetime
from typing import List, Optional

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import metric_label


class FocusChart(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color=theme.CARD, corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme
        self._fig = Figure(figsize=(8, 3), dpi=96, facecolor=theme.CARD)
        self._ax = self._fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self._fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                          padx=theme.SPACE_SM, pady=theme.SPACE_SM)
        self.bind("<Destroy>", self._on_destroy)
        self._style()

    def _style(self):
        ax, t = self._ax, self.theme
        ax.set_facecolor(t.CARD)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(t.TEXT_SECONDARY)
        ax.tick_params(colors=t.TEXT_SECONDARY, labelsize=9)
        ax.title.set_color(t.TEXT_PRIMARY)

    def set_series(self, metric: str, dates: List[datetime], values: List[float],
                   baseline_mean: Optional[float] = None, baseline_std: Optional[float] = None,
                   recent_batch_start: Optional[datetime] = None) -> None:
        import numpy as np
        ax, t = self._ax, self.theme
        ax.clear()
        self._style()
        ax.set_title(metric_label(metric))
        if not dates or not values:
            ax.text(0.5, 0.5, "No measurements for this metric in the selected window.",
                    transform=ax.transAxes, ha="center", va="center", color=t.TEXT_SECONDARY)
            self.canvas.draw_idle()
            return

        arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
        finite = arr[np.isfinite(arr)]

        # SPC overlays (Rule 1 only).
        if baseline_mean is not None:
            ax.axhline(baseline_mean, color=t.TEXT_SECONDARY, ls="--", lw=1, label="Baseline mean")
            if baseline_std:
                ax.axhspan(baseline_mean - 2 * baseline_std, baseline_mean + 2 * baseline_std,
                           color=t.TIER_WARNING, alpha=0.08)
                for k in (3, -3):
                    ax.axhline(baseline_mean + k * baseline_std, color=t.TIER_OOC, ls=":", lw=1,
                               label="±3σ control limit" if k == 3 else None)
        if recent_batch_start is not None:
            ax.axvspan(recent_batch_start, dates[-1], color=t.ACCENT, alpha=0.10)

        ax.plot(dates, values, marker="o", ms=3, lw=1, color=t.ACCENT, label=metric_label(metric))

        # ---- Robust y-window. A control chart must keep the control band and the
        # bulk of the data legible; a lone Rule-1 outlier (e.g. one 0.07 point on a
        # 0.0015±0.0008 baseline) must NOT autoscale the axis and flatten everything.
        # Build the window from the control limits ∪ robust data percentiles, never
        # raw min/max. Out-of-window points are clamped to the edge below so the
        # violation markers stay visible.
        lo_c, hi_c = [], []
        if finite.size:
            if finite.size >= 10:
                p_lo, p_hi = np.percentile(finite, [2, 98])
            else:
                p_lo, p_hi = float(finite.min()), float(finite.max())
            lo_c.append(p_lo); hi_c.append(p_hi)
        if baseline_mean is not None and baseline_std:
            lo_c.append(baseline_mean - 3.5 * baseline_std)
            hi_c.append(baseline_mean + 3.5 * baseline_std)
        elif baseline_mean is not None:
            lo_c.append(baseline_mean); hi_c.append(baseline_mean)
        if lo_c and hi_c:
            lo, hi = min(lo_c), max(hi_c)
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                if finite.size:
                    lo, hi = float(finite.min()), float(finite.max())
                if hi <= lo:
                    lo, hi = lo - 1.0, hi + 1.0
            pad = (hi - lo) * 0.08 or abs(hi) * 0.1 or 1.0
            ax.set_ylim(lo - pad, hi + pad)

        # Mark out-of-3σ points (Rule 1 violations), clamping any that fall outside
        # the y-window to the visible edge so the violation isn't hidden off-screen.
        if baseline_mean is not None and baseline_std:
            ucl, lcl = baseline_mean + 3 * baseline_std, baseline_mean - 3 * baseline_std
            y0, y1 = ax.get_ylim()
            ox, oy, off = [], [], 0
            for d, v in zip(dates, values):
                if v is None or not np.isfinite(v):
                    continue
                if v > ucl or v < lcl:
                    cy = min(max(v, y0), y1)
                    off += (cy != v)
                    ox.append(d); oy.append(cy)
            if ox:
                ax.scatter(ox, oy, color=t.TIER_OOC, s=30, zorder=5, clip_on=False)
            if off:
                ax.text(0.99, 0.97, f"▲ {off} point(s) off-scale", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, color=t.TIER_OOC)
        ax.legend(loc="best", fontsize=8, facecolor=t.CARD, edgecolor=t.BORDER, labelcolor=t.TEXT_SECONDARY)
        self._fig.tight_layout()
        self.canvas.draw_idle()

    def _on_destroy(self, _evt=None):
        try:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
        except Exception:
            pass
