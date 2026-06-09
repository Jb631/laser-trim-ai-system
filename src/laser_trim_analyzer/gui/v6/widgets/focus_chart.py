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
        ax, t = self._ax, self.theme
        ax.clear()
        self._style()
        ax.set_title(metric_label(metric))
        if not dates or not values:
            ax.text(0.5, 0.5, "No measurements for this metric in the selected window.",
                    transform=ax.transAxes, ha="center", va="center", color=t.TEXT_SECONDARY)
            self.canvas.draw_idle()
            return

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
        # Mark out-of-3σ points in the OOC color (Rule 1 violations).
        if baseline_mean is not None and baseline_std:
            ucl, lcl = baseline_mean + 3 * baseline_std, baseline_mean - 3 * baseline_std
            ox = [d for d, v in zip(dates, values) if v > ucl or v < lcl]
            oy = [v for v in values if v > ucl or v < lcl]
            if ox:
                ax.scatter(ox, oy, color=t.TIER_OOC, s=30, zorder=5)
        ax.legend(loc="best", fontsize=8, facecolor=t.CARD, edgecolor=t.BORDER, labelcolor=t.TEXT_SECONDARY)
        self._fig.tight_layout()
        self.canvas.draw_idle()

    def _on_destroy(self, _evt=None):
        try:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
        except Exception:
            pass
