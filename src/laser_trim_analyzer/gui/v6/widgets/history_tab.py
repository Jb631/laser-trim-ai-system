"""Model page — 'History' tab (Job 2b).

Pull all of a model's record: a measured-value trend (electrical angle, untrimmed/trimmed
resistance, resistance change %, unit length — pick from the dropdown) over the full history,
plus monthly linearity pass-rate. Fed from db.get_model_measurement_history. Two stacked
axes share one canvas; the measured trend is a SCATTER (gappy per-unit points, no false
continuity) with a robust y-window so a few wild values can't flatten the bulk.
"""
from typing import Dict, List, Optional

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from laser_trim_analyzer.gui.v6.chart_redraw import debounce_resize_redraws
from laser_trim_analyzer.gui.v6.theme import ThemeManager

_LABELS = {
    "measured_electrical_angle": "Measured electrical angle",
    "untrimmed_resistance": "Untrimmed resistance",
    "trimmed_resistance": "Trimmed resistance",
    "resistance_change_percent": "Resistance change (%)",
    "unit_length": "Unit length",
}


class HistoryTab(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._data: Optional[Dict] = None
        self._metric: Optional[str] = None

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        ctk.CTkLabel(bar, text="Measure:", font=theme.font(theme.SIZE_BODY),
                     text_color=theme.TEXT_SECONDARY).pack(side="left", padx=(0, theme.SPACE_SM))
        self._menu = ctk.CTkOptionMenu(bar, values=["—"], width=220, command=self._on_pick,
                                       font=theme.font(theme.SIZE_BODY))
        self._menu.pack(side="left")
        self._stats = ctk.CTkLabel(bar, text="", font=theme.font(theme.SIZE_CAPTION),
                                   text_color=theme.TEXT_SECONDARY, anchor="w")
        self._stats.pack(side="left", padx=theme.SPACE_MD)

        self._fig = Figure(figsize=(8, 5), dpi=96, facecolor=theme.CARD)
        self._ax_val = self._fig.add_subplot(211)
        self._ax_pr = self._fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self._fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                         padx=theme.SPACE_SM, pady=theme.SPACE_SM)
        # One render at the end of a resize, not one per <Configure>.
        self._redraw = debounce_resize_redraws(self.canvas)
        self.bind("<Destroy>", self._on_destroy)

    def set_data(self, data: Dict) -> None:
        self._data = data or None
        stats = (data or {}).get("stats") or {}
        avail = [m for m in _LABELS if m in stats]   # only metrics that actually have data
        if avail:
            self._menu.configure(values=[_LABELS[m] for m in avail])
            if self._metric not in avail:
                self._metric = avail[0]
            self._menu.set(_LABELS[self._metric])
        else:
            self._menu.configure(values=["—"])
            self._menu.set("—")
            self._metric = None
        self._render()

    def _on_pick(self, label: str) -> None:
        for raw, lbl in _LABELS.items():
            if lbl == label:
                self._metric = raw
                break
        self._render()

    def _render(self) -> None:
        import numpy as np
        t = self.theme
        for ax in (self._ax_val, self._ax_pr):
            ax.clear()
            ax.set_facecolor(t.CARD)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("bottom", "left"):
                ax.spines[side].set_color(t.TEXT_SECONDARY)
            ax.tick_params(colors=t.TEXT_SECONDARY, labelsize=8)
            ax.title.set_color(t.TEXT_PRIMARY)

        d = self._data
        if not d or not self._metric:
            self._ax_val.text(0.5, 0.5, "No measured-value history for this model.",
                              transform=self._ax_val.transAxes, ha="center", va="center",
                              color=t.TEXT_SECONDARY)
            self._stats.configure(text="")
            self.canvas.draw_idle()
            return

        # --- Top: measured-value scatter over time (robust y-window) ---
        series = (d.get("series") or {}).get(self._metric) or []
        self._ax_val.set_title(_LABELS[self._metric])
        if series:
            dates = [p[0] for p in series]
            vals = np.array([p[1] for p in series], dtype=float)
            self._ax_val.scatter(dates, vals, s=10, color=t.ACCENT, alpha=0.75)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                lo, hi = (np.percentile(finite, [2, 98]) if finite.size >= 10
                          else (float(finite.min()), float(finite.max())))
                if hi <= lo:
                    lo, hi = float(finite.min()) - 1.0, float(finite.max()) + 1.0
                pad = (hi - lo) * 0.08 or abs(hi) * 0.1 or 1.0
                self._ax_val.set_ylim(lo - pad, hi + pad)
                # Flag any points clamped off the window so magnitude isn't hidden.
                off = finite[(finite < lo) | (finite > hi)]
                if off.size:
                    ext = off[np.argmax(np.abs(off - np.median(finite)))]
                    self._ax_val.text(0.99, 0.96, f"▲ {off.size} off-scale (to {ext:.4g})",
                                      transform=self._ax_val.transAxes, ha="right", va="top",
                                      fontsize=7, color=t.TIER_OOC)

        s = (d.get("stats") or {}).get(self._metric)
        if s:
            self._stats.configure(
                text=(f"n={s['n']}   mean={s['mean']:.4g}   σ={s['std']:.4g}   "
                      f"min={s['min']:.4g}   max={s['max']:.4g}   last={s['last']:.4g}"))
        else:
            self._stats.configure(text="")

        # --- Bottom: monthly linearity pass-rate line (bounded 0-100) ---
        pr = d.get("passrate_periods") or []
        pr = [p for p in pr if p[3] is not None]
        self._ax_pr.set_title("Linearity pass-rate by month")
        if pr:
            xs = list(range(len(pr)))
            rates = [p[3] for p in pr]
            self._ax_pr.plot(xs, rates, marker="o", ms=3, lw=1.2, color=t.ACCENT)
            self._ax_pr.set_ylim(0, 105)
            self._ax_pr.axhline(100, color=t.TEXT_SECONDARY, ls="--", lw=0.8, alpha=0.5)
            step = max(1, len(pr) // 8)
            self._ax_pr.set_xticks(xs[::step])
            self._ax_pr.set_xticklabels([pr[i][0] for i in xs[::step]], rotation=45,
                                        ha="right", fontsize=7)
            self._ax_pr.set_ylabel("% pass", color=t.TEXT_SECONDARY, fontsize=8)
        else:
            self._ax_pr.text(0.5, 0.5, "No linearity pass/fail history.",
                             transform=self._ax_pr.transAxes, ha="center", va="center",
                             color=t.TEXT_SECONDARY)
        # h_pad keeps the bottom panel's title clear of the top panel's
        # rotated tick labels (they collided at default padding).
        self._fig.tight_layout(h_pad=2.4)
        self.canvas.draw_idle()

    def _on_destroy(self, _evt=None):
        try:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
        except Exception:
            pass
