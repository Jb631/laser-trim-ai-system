"""Dashboard — YieldPanel: headline pass-rate %, Pass/Warn/Fail counts, total, trend."""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.mini_trend_chart import MiniTrendChart


class YieldPanel(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, title: str, **kwargs):
        super().__init__(master, fg_color=theme.CARD, corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme
        t = theme
        ctk.CTkLabel(self, text=title, font=t.font(t.SIZE_HEADING, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w")\
            .pack(side="top", fill="x", padx=t.SPACE_MD, pady=(t.SPACE_MD, 0))
        self._rate = ctk.CTkLabel(self, text="—", font=t.font(t.SIZE_DISPLAY, "bold"),
                                  text_color=t.TEXT_PRIMARY, anchor="w")
        self._rate.pack(side="top", fill="x", padx=t.SPACE_MD)
        self._counts = ctk.CTkLabel(self, text="", font=t.font(t.SIZE_BODY),
                                    text_color=t.TEXT_SECONDARY, anchor="w")
        self._counts.pack(side="top", fill="x", padx=t.SPACE_MD)
        self._total = ctk.CTkLabel(self, text="", font=t.font(t.SIZE_CAPTION),
                                   text_color=t.TEXT_SECONDARY, anchor="w")
        self._total.pack(side="top", fill="x", padx=t.SPACE_MD)
        self._trend = MiniTrendChart(self, theme=t)
        self._trend.pack(side="top", fill="x", padx=t.SPACE_MD, pady=(t.SPACE_SM, t.SPACE_MD))

    def set_yield(self, stats: dict, total_label: str) -> None:
        # HEADLINE = linearity yield (the CUSTOMER basis). Linearity is the
        # zero-tolerance requirement; WARNING units passed linearity — sigma is
        # an internal drift-watch flag, not a disposition. The old headline
        # counted only clean passes, presenting 59.8% "pass" when customer-
        # acceptable yield was 94% — a process-watch number dressed as yield.
        rate = stats.get("linearity_yield")
        if rate is None:
            rate = stats.get("pass_rate")
        self._rate.configure(text=f"{rate:.1f}% linearity yield" if rate is not None else "—")
        # Breakdown reconciles with the headline: Pass and Watch are both
        # accepted; Watch = sigma flagged (drift signal), Fail = linearity.
        parts = [f"Pass {stats.get('passed', 0)}",
                 f"Watch {stats.get('warnings', 0)}",
                 f"Fail {stats.get('failed', 0)}"]
        if stats.get("untrimmed", 0):
            parts.append(f"Untrimmed {stats['untrimmed']}")
        if stats.get("errors", 0):
            parts.append(f"Err {stats['errors']}")
        self._counts.configure(text=" · ".join(parts))
        if stats.get("future_dated"):
            # A record dated in the FUTURE is a mislabeled file, not production
            # — name it so it gets fixed instead of silently skewing trends.
            total_label += (f"   ⚠ {stats['future_dated']} future-dated record"
                            f"{'s' if stats['future_dated'] != 1 else ''} excluded "
                            f"(check filename dates)")
        self._total.configure(text=total_label)
        # Sparkline follows the headline basis ("rate" = linearity_yield; falls
        # back to pass_rate for callers that predate the field).
        self._trend.set_points([(p["date"], p.get("rate", p.get("pass_rate")))
                                for p in stats.get("trend", [])])
