"""Spec 3c — MetricPillRow: clickable, tier-colored metric pills.

Grouped layout (2026-07-13, James: "clear sections for what im looking at"):
pills render in METRIC_GROUPS rows — process signals on one band, the trim
and final-test OUTCOMES on a second — instead of one 12-pill overflow line.
"""
from typing import Callable, Dict, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import (
    DriftTier, METRIC_GROUPS, ModelDriftStatus, metric_label)

_MAX_PER_ROW = 6   # process group is 9 pills — wrap so nothing clips


class MetricPillRow(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_pill_click: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_pill_click
        self._pills: Dict[str, _Pill] = {}
        self._selected_metric: Optional[str] = None
        # Two visual bands: the process-signal group, then both outcome
        # groups side by side (they're 1 + 2 pills — a shared band reads
        # better than two near-empty rows).
        bands = [
            (METRIC_GROUPS[0][0], list(METRIC_GROUPS[0][2])),
            ("Outcomes — trim linearity · final test",
             list(METRIC_GROUPS[1][2]) + list(METRIC_GROUPS[2][2])),
        ]
        for band_title, metrics in bands:
            ctk.CTkLabel(self, text=band_title,
                         font=theme.font(theme.SIZE_CAPTION, "bold"),
                         text_color=theme.TEXT_SECONDARY, anchor="w")\
                .pack(side="top", fill="x", pady=(theme.SPACE_XS, 0))
            row = None
            for i, m in enumerate(metrics):
                if i % _MAX_PER_ROW == 0:
                    row = ctk.CTkFrame(self, fg_color="transparent")
                    row.pack(side="top", fill="x")
                p = _Pill(row, metric=m, theme=theme, on_click=self._cb)
                p.pack(side="left", padx=(0, theme.SPACE_SM), pady=theme.SPACE_XS)
                self._pills[m] = p

    def set_status(self, status: ModelDriftStatus, recent_means: dict = None) -> None:
        recent_means = recent_means or {}
        for m, pill in self._pills.items():
            ms = status.per_metric.get(m)
            if ms is not None:
                pill.set_metric_status(ms, recent_override=recent_means.get(m))

    def set_selected(self, metric: str) -> None:
        if self._selected_metric == metric:
            return
        if self._selected_metric and self._selected_metric in self._pills:
            self._pills[self._selected_metric].set_selected(False)
        if metric in self._pills:
            self._pills[metric].set_selected(True)
            self._selected_metric = metric


class _Pill(ctk.CTkFrame):
    def __init__(self, master, metric, theme: ThemeManager, on_click):
        super().__init__(master, fg_color=theme.CARD, corner_radius=theme.RADIUS_MD,
                         border_width=2, border_color=theme.CARD)
        self.metric = metric
        self.theme = theme
        self._cb = on_click
        self._selected = False
        self._name_label = ctk.CTkLabel(self, text=metric_label(metric),
                                        font=theme.font(theme.SIZE_CAPTION, "bold"),
                                        text_color=theme.TEXT_PRIMARY)
        self._name_label.pack(side="top", padx=theme.SPACE_SM, pady=(theme.SPACE_SM, 0))
        self._summary_label = ctk.CTkLabel(self, text="—", font=theme.font(theme.SIZE_CAPTION),
                                           text_color=theme.TEXT_SECONDARY)
        self._summary_label.pack(side="top", padx=theme.SPACE_SM, pady=(0, theme.SPACE_SM))
        for w in (self, self._name_label, self._summary_label):
            w.bind("<Button-1>", lambda e: self._on_click())

    def set_metric_status(self, ms, recent_override=None) -> None:
        bg, fg = self.theme.tier_color(ms.tier)
        self.configure(fg_color=bg)
        if not self._selected:
            self.configure(border_color=bg)
        # Headline the honest baseline shift (σ), consistent with the card and the
        # drift table — not the CUSUM magnitude. Falls back to magnitude only when no
        # recent data is available to compute the shift.
        recent_val = recent_override if recent_override is not None else ms.recent_mean
        shift = ((recent_val - ms.baseline_mean) / ms.baseline_std
                 if (recent_val is not None and ms.baseline_std) else None)
        if not ms.is_trained:
            text = "untrained"
        elif ms.tier == DriftTier.STABLE:
            text = "OK"
        elif shift is not None:
            text = f"{shift:+.1f}σ"
        else:
            text = f"{ms.magnitude:+.1f}σ"
        self._summary_label.configure(text=text, text_color=fg)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.configure(border_color=self.theme.ACCENT if selected else self.cget("fg_color"))

    def _on_click(self):
        self._cb(self.metric)
