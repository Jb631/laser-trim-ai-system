"""Company yield trend — the company-as-a-whole view (2026-07-06).

Weekly/monthly company pass-rate line with per-system (A/B/C) overlays and a
light units-processed volume backdrop. Answers "is our process moving for
better or worse" at the company level; the per-system split shows whether the
LTS3 (System C) ramp tracks the established lasers.

Dumb widget: page fetches db.get_company_yield_trend and calls set_data.
"""
from typing import Any, Dict, Optional

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from laser_trim_analyzer.gui.v6.theme import ThemeManager

# Series colors: company = theme accent; systems get stable, distinct hues.
_SYSTEM_COLORS = {"A": "#22c55e", "B": "#a78bfa", "C": "#f59e0b"}


class CompanyTrendChart(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color=theme.CARD, corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme
        self._fig = Figure(figsize=(8, 2.6), dpi=96, facecolor=theme.CARD)
        self._ax = self._fig.add_subplot(111)
        self._vol_ax = self._ax.twinx()
        self.canvas = FigureCanvasTkAgg(self._fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                         padx=theme.SPACE_SM, pady=theme.SPACE_SM)
        self.bind("<Destroy>", self._on_destroy)
        self._style()

    def _style(self):
        t = self.theme
        for ax in (self._ax, self._vol_ax):
            ax.set_facecolor(t.CARD)
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(False)
        self._ax.spines["bottom"].set_visible(True)
        self._ax.spines["bottom"].set_color(t.TEXT_SECONDARY)
        self._ax.spines["left"].set_visible(True)
        self._ax.spines["left"].set_color(t.TEXT_SECONDARY)
        self._ax.tick_params(colors=t.TEXT_SECONDARY, labelsize=8)
        self._vol_ax.tick_params(colors=t.TEXT_DISABLED, labelsize=8)
        self._ax.title.set_color(t.TEXT_PRIMARY)

    def set_data(self, trend: Optional[Dict[str, Any]], period_label: str = "week") -> None:
        """Render the trend. NEVER fails blank: any internal error draws a
        visible message in the axes (a swallowed exception once left bare
        0-1 axes with no explanation — worse than an error)."""
        try:
            self._set_data(trend, period_label)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Company trend render failed")
            try:
                self._ax.clear()
                self._vol_ax.clear()
                self._style()
                self._ax.text(0.5, 0.5,
                              f"Chart error — see log.\n{type(exc).__name__}: {exc}",
                              transform=self._ax.transAxes, ha="center", va="center",
                              color=self.theme.TIER_OOC, fontsize=9)
                self.canvas.draw_idle()
            except Exception:
                pass

    def _set_data(self, trend: Optional[Dict[str, Any]], period_label: str = "week") -> None:
        t = self.theme
        ax, vol = self._ax, self._vol_ax
        ax.clear()
        vol.clear()
        self._style()
        # No axes title: the page's section header already says "Company trend"
        # and the period is on the toggle — the title row is better spent on
        # the legend (which used to sit on the data).

        periods = (trend or {}).get("periods") or []
        company = (trend or {}).get("company") or []
        by_system = (trend or {}).get("by_system") or {}
        partial_last = bool((trend or {}).get("partial_last"))
        data_through = (trend or {}).get("data_through")
        if not periods or not company:
            ax.text(0.5, 0.5, "No trim data in the selected window.",
                    transform=ax.transAxes, ha="center", va="center",
                    color=t.TEXT_SECONDARY)
            self.canvas.draw_idle()
            return

        x = list(range(len(periods)))

        # Volume backdrop (right axis): light bars, never competes with lines.
        vol.bar(x, [r["total"] for r in company], color=t.ELEVATED, alpha=0.9,
                width=0.8, zorder=1)
        vol.set_ylabel("units", color=t.TEXT_DISABLED, fontsize=8)
        vol.yaxis.set_label_position("right")  # twin label was ghosting at left

        # Per-system overlays first (thin), company line on top (bold).
        # Rates are LINEARITY yield (customer basis) — see get_company_yield_trend.
        for sys_name, series in by_system.items():
            ys = [r["linearity_yield"] for r in series]
            if all(v is None for v in ys):
                continue
            ax.plot(x, ys, lw=1.1, alpha=0.85, marker="o", ms=2.5,
                    color=_SYSTEM_COLORS.get(sys_name, t.TEXT_SECONDARY),
                    label=f"System {sys_name}", zorder=3)
        comp_rates = [r["linearity_yield"] for r in company]
        ax.plot(x, comp_rates, lw=2.2, marker="o", ms=3.5,
                color=t.ACCENT, label="Company", zorder=4)

        # Partial-period honesty: the newest bucket is still filling — draw its
        # company point hollow and say so, or a month-start "yield crash" is
        # the first thing every review meeting argues about.
        if partial_last and comp_rates and comp_rates[-1] is not None:
            ax.plot([x[-1]], [comp_rates[-1]], marker="o", ms=7, mew=1.6,
                    mfc="none", mec=t.ACCENT, zorder=5)
            ax.annotate("partial", (x[-1], comp_rates[-1]),
                        textcoords="offset points", xytext=(6, 8),
                        fontsize=7, color=t.TEXT_SECONDARY)

        # Readable x labels: at most ~10 ticks (always include the last).
        step = max(1, len(periods) // 10)
        ticks = list(range(0, len(periods), step))
        if ticks[-1] != len(periods) - 1:
            ticks.append(len(periods) - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([periods[i] for i in ticks], rotation=30, ha="right")
        ax.set_ylabel("linearity yield %", color=t.TEXT_SECONDARY, fontsize=8)

        # Data vintage: batch-loaded data lags production — say how fresh it is.
        if data_through is not None:
            ax.text(0.995, 1.02, f"Data through {data_through:%Y-%m-%d}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color=t.TEXT_SECONDARY)

        # Y window: show the informative band, not always 0-100.
        rates = [r["linearity_yield"] for r in company if r["linearity_yield"] is not None]
        for series in by_system.values():
            rates += [r["linearity_yield"] for r in series if r["linearity_yield"] is not None]
        if rates:
            lo, hi = min(rates), max(rates)
            pad = max((hi - lo) * 0.15, 2.0)
            ax.set_ylim(max(0.0, lo - pad), min(100.0, hi + pad) + 0.5)

        ax.set_zorder(vol.get_zorder() + 1)   # lines above bars
        ax.patch.set_visible(False)
        # Legend ABOVE the axes (where the redundant title used to be) so it
        # can never sit on the data.
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8,
                  ncol=4, frameon=False, labelcolor=t.TEXT_SECONDARY,
                  handlelength=1.6, columnspacing=1.2)
        self._fig.tight_layout()
        self.canvas.draw_idle()

    def _on_destroy(self, _evt=None):
        try:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
        except Exception:
            pass
