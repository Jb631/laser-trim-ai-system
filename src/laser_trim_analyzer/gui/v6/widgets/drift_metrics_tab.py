"""Spec 3c — DriftMetricsTab: table of all 8 metrics for the model."""
from typing import Callable, Dict

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import ModelDriftStatus, WATCHED_METRICS, metric_label

_COLUMNS = ["Metric", "Tier", "Alert", "Baseline (mean±std)", "Recent", "Shift (σ)"]


class DriftMetricsTab(ctk.CTkScrollableFrame):
    def __init__(self, master, theme: ThemeManager, on_metric_select: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_metric_select
        self._rows: Dict[str, _MetricRow] = {}
        header = ctk.CTkFrame(self, fg_color=theme.CARD)
        header.pack(side="top", fill="x", pady=(0, theme.SPACE_XS))
        for i in range(len(_COLUMNS)):
            header.grid_columnconfigure(i, weight=1, uniform="dm")
        for i, col in enumerate(_COLUMNS):
            ctk.CTkLabel(header, text=col, font=theme.font(theme.SIZE_CAPTION, "bold"),
                         text_color=theme.TEXT_SECONDARY, anchor="w")\
                .grid(row=0, column=i, sticky="ew", padx=theme.SPACE_SM, pady=theme.SPACE_XS)

    def set_status(self, status: ModelDriftStatus, recent_means: dict = None) -> None:
        recent_means = recent_means or {}
        for r in self._rows.values():
            r.destroy()
        self._rows.clear()
        for m in WATCHED_METRICS:
            ms = status.per_metric.get(m)
            if ms is None:
                continue
            row = _MetricRow(self, ms=ms, theme=self.theme, on_click=self._cb,
                             recent_override=recent_means.get(m))
            row.pack(side="top", fill="x", pady=1)
            self._rows[m] = row


class _MetricRow(ctk.CTkFrame):
    def __init__(self, master, ms, theme: ThemeManager, on_click, recent_override=None):
        bg, _ = theme.tier_color(ms.tier)
        super().__init__(master, fg_color=bg)
        self.metric = ms.metric
        self._cb = on_click
        recent_val = recent_override if recent_override is not None else ms.recent_mean
        recent = f"{recent_val:.4g}" if recent_val is not None else "—"
        # Honest shift, verifiable against the Baseline & Recent cells beside it:
        # (recent - baseline) / baseline_std. Replaces the old `magnitude` (CUSUM
        # distance past the limit), which couldn't be reconciled with the numbers shown.
        shift = ((recent_val - ms.baseline_mean) / ms.baseline_std
                 if (recent_val is not None and ms.baseline_std) else None)
        shift_txt = f"{shift:+.2f}σ" if shift is not None else "—"
        cells = [metric_label(ms.metric), ms.tier.name.replace("_", " ").title(),
                 ms.alert_type.value if ms.alert_type else "—",
                 f"{ms.baseline_mean:.4g} ± {ms.baseline_std:.4g}", recent, shift_txt]
        for i in range(len(cells)):
            self.grid_columnconfigure(i, weight=1, uniform="dm")
        for i, txt in enumerate(cells):
            lbl = ctk.CTkLabel(self, text=txt, font=theme.font(theme.SIZE_BODY),
                               text_color=theme.TEXT_PRIMARY, anchor="w")
            lbl.grid(row=0, column=i, sticky="ew", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
            lbl.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self):
        self._cb(self.metric)
