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
        self._median_line = None  # set by the baseline branch, drawn post-y-window

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

        if baseline_mean is None:
            # Trend view (e.g. Smoothness): SCATTER only. Connecting clustered, gappy
            # per-unit points with a line reads as jagged noise and implies a continuity
            # that isn't there (the "charts are all over the place" complaint).
            ax.scatter(dates, values, s=12, color=t.ACCENT, alpha=0.8, label=metric_label(metric))
        else:
            # Control chart for BATCH data (2026-07-07 redesign). file_date is
            # day-granularity: a batch lands as many units on ONE x position.
            # Drawing one line through raw units produced vertical zigzags
            # inside each batch-day and long horizontal jumps between batches —
            # "vertical lines then horizontal connected lines" (James). SPC
            # treatment for grouped data instead: individuals as dots, the
            # trend as a line through DAILY MEANS, broken across gaps >14 days
            # so idle periods don't render as fake continuity.
            ax.scatter(dates, values, s=13, color=t.ACCENT, alpha=0.5,
                       label=f"{metric_label(metric)} — units")
            by_day: dict = {}
            for d, v in zip(dates, values):
                if v is None or not np.isfinite(v):
                    continue
                by_day.setdefault(d, []).append(v)
            if by_day:
                mdates_ = sorted(by_day)
                mvals: list = []
                mx: list = []
                prev = None
                for d in mdates_:
                    if prev is not None and (d - prev).days > 14:
                        mx.append(prev)          # break the line across the gap
                        mvals.append(np.nan)
                    mx.append(d)
                    # MEDIAN, not mean: one scale-corrupt unit in a batch
                    # (e.g. 2.1e8 Ω) would drag the day's mean off-scale and
                    # draw a full-height spike. The median tracks the batch;
                    # the red Rule-1 markers still disclose the outliers.
                    mvals.append(float(np.median(by_day[d])))
                    prev = d
                self._median_line = (mx, mvals)  # drawn after the y-window is known

        # ---- Robust y-window. A control chart must keep the control band and the
        # bulk of the data legible; outliers (lone OR clustered — e.g. a 929 Ω
        # scale error among 4,700s) must NOT stretch the axis and flatten
        # everything. Both modes anchor on the central bulk (10-90 pct);
        # out-of-window points are clamped to the edge and DISCLOSED below —
        # in no-baseline mode they previously vanished silently.
        lo_c, hi_c = [], []
        if finite.size:
            if finite.size >= 10:
                p_lo, p_hi = np.percentile(finite, [10, 90])
            else:
                # Tiny n: window on the in-family points (within 3 sample-σ of
                # the median) so one scale error can't stretch the axis.
                med_v = float(np.median(finite))
                sd = float(np.std(finite)) or abs(med_v) * 0.05 or 1.0
                fam = [float(v) for v in finite if abs(v - med_v) <= 3 * sd]
                p_lo, p_hi = (min(fam), max(fam)) if fam else (med_v - 1.0, med_v + 1.0)
            lo_c.append(p_lo); hi_c.append(p_hi)
        limits_off_scale = None
        if baseline_mean is not None and baseline_std:
            # Include the ±3.5σ control band in the window ONLY when it is
            # commensurate with the visible data. A baseline trained across
            # mixed historical regimes can carry a σ that dwarfs the current
            # window (8340-1: σ=1.26 vs recent spread ~0.1) — always forcing
            # the band into view locked the y-axis at ±4.8 for EVERY time
            # window, so zooming 'didn't zoom'. When the band is >6x the data
            # bulk, fit to the data and annotate the off-scale limits instead.
            band_lo = baseline_mean - 3.5 * baseline_std
            band_hi = baseline_mean + 3.5 * baseline_std
            data_span = max((hi_c[0] - lo_c[0]) if lo_c else 0.0, 1e-12)
            if (band_hi - band_lo) <= 6.0 * data_span:
                lo_c.append(band_lo); hi_c.append(band_hi)
            else:
                limits_off_scale = (baseline_mean - 3 * baseline_std,
                                    baseline_mean + 3 * baseline_std)
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
        if limits_off_scale is not None:
            ax.text(0.01, 0.97,
                    f"±3σ control limits off-scale ({limits_off_scale[0]:.3g} … "
                    f"{limits_off_scale[1]:.3g}) — baseline spans mixed history",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                    color=t.TIER_WARNING)

        # Daily median trend line. A batch-day where MOST units are corrupt has
        # an off-scale median — drawing it (even clamped) reads as a spike. The
        # line covers only in-window days; off-window days become a GAP, and
        # their story is told by the clamped markers + disclosure below.
        med = self._median_line
        if med is not None:
            y0, y1 = ax.get_ylim()
            mx, mvals = med
            mgapped = [v if (np.isfinite(v) and y0 <= v <= y1) else np.nan
                       for v in mvals]
            ax.plot(mx, mgapped, lw=1.6, marker="o", ms=4, color=t.ACCENT_HOVER,
                    alpha=0.95, label="Daily median", zorder=4)
            self._median_line = None

        # Clamp + disclose EVERY out-of-window point (both modes). With a
        # trained baseline, out-of-limit points are Rule-1 violations (red);
        # without one they're still shown/counted, in warning amber (no SPC
        # meaning, just "far outside the bulk — check the data").
        y0, y1 = ax.get_ylim()
        has_limits = baseline_mean is not None and bool(baseline_std)
        ucl = (baseline_mean + 3 * baseline_std) if has_limits else None
        lcl = (baseline_mean - 3 * baseline_std) if has_limits else None
        ox, oy, off_vals = [], [], []
        for d, v in zip(dates, values):
            if v is None or not np.isfinite(v):
                continue
            beyond_limits = has_limits and (v > ucl or v < lcl)
            beyond_window = v > y1 or v < y0
            if beyond_limits or beyond_window:
                cy = min(max(v, y0), y1)
                if cy != v:
                    off_vals.append(v)
                ox.append(d); oy.append(cy)
        mark_color = t.TIER_OOC if has_limits else t.TIER_WARNING
        if ox:
            ax.scatter(ox, oy, color=mark_color, s=30, zorder=5, clip_on=False)
        if off_vals:
            # Name how far the worst excursion actually reaches — a clamped marker
            # alone hides magnitude, which is exactly what a QA reviewer needs.
            ext = max(off_vals, key=abs)
            ax.text(0.99, 0.97, f"▲ {len(off_vals)} off-scale (max {ext:.3g})",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    color=mark_color)
        ax.legend(loc="best", fontsize=8, facecolor=t.CARD, edgecolor=t.BORDER, labelcolor=t.TEXT_SECONDARY)
        self._fig.tight_layout()
        self.canvas.draw_idle()

    def _on_destroy(self, _evt=None):
        try:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
        except Exception:
            pass
