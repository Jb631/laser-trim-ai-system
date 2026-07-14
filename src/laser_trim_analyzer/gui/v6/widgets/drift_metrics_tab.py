"""Spec 3c — DriftMetricsTab: grouped table of every watched metric.

Rows render in METRIC_GROUPS order under three plain-language group headers
(process signals / trim outcome / final-test outcome) so the 12-metric list
reads as three questions, not a wall (James, 2026-07-13)."""
from typing import Callable, Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import (
    AlertType, METRIC_GROUPS, ModelDriftStatus, format_metric_value,
    metric_label)

_COLUMNS = ["Metric", "Tier", "Alert", "Baseline (lot mean±σ)", "Last lot", "Shift (σ)"]


class DriftMetricsTab(ctk.CTkScrollableFrame):
    def __init__(self, master, theme: ThemeManager, on_metric_select: Callable[[str], None],
                 on_requalify: Callable[[], None] = None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_metric_select
        self._rows: Dict[str, _MetricRow] = {}
        self._group_headers: List = []
        self._on_requalify = on_requalify
        header = ctk.CTkFrame(self, fg_color=theme.CARD)
        header.pack(side="top", fill="x", pady=(0, theme.SPACE_XS))
        for i in range(len(_COLUMNS)):
            header.grid_columnconfigure(i, weight=1, uniform="dm")
        for i, col in enumerate(_COLUMNS):
            ctk.CTkLabel(header, text=col, font=theme.font(theme.SIZE_CAPTION, "bold"),
                         text_color=theme.TEXT_SECONDARY, anchor="w")\
                .grid(row=0, column=i, sticky="ew", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
        # Baseline provenance + the per-model requalification control
        # (James's policy 2026-07-13: manual reset on design change, with an
        # AS9100-auditable record of when and why).
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(side="bottom", fill="x", pady=(theme.SPACE_SM, 0))
        self._baseline_lbl = ctk.CTkLabel(
            footer, text="Baseline period: full history",
            font=theme.font(theme.SIZE_CAPTION), text_color=theme.TEXT_SECONDARY,
            anchor="w", justify="left", wraplength=900)
        self._baseline_lbl.pack(side="left", fill="x", expand=True)
        if on_requalify is not None:
            ctk.CTkButton(footer, text="Requalify baseline…", width=150,
                          fg_color=theme.CARD, hover_color=theme.ELEVATED,
                          text_color=theme.TEXT_PRIMARY, border_width=1,
                          border_color=theme.BORDER, corner_radius=theme.RADIUS_SM,
                          command=on_requalify).pack(side="right")

    def set_baseline_info(self, req) -> None:
        """Baseline-period disclosure. req = (effective_date, note, set_at)
        from the requalification audit table, or None (= full history)."""
        if not hasattr(self, "_baseline_lbl"):
            return
        if req:
            eff, note, at = req
            txt = (f"Baseline period: data since {str(eff)[:10]} "
                   f"(requalified {str(at)[:10]}"
                   + (f" — {note}" if note else "") + ")")
        else:
            txt = "Baseline period: full history (no requalification on record)"
        self._baseline_lbl.configure(text=txt)

    def set_status(self, status: ModelDriftStatus, recent_means: dict = None) -> None:
        recent_means = recent_means or {}
        for r in self._rows.values():
            r.destroy()
        # __dict__.get, not getattr: the QA sweep's headless widget stub
        # answers ANY attribute with a callable, which is not iterable.
        for h in (self.__dict__.get("_group_headers") or []):
            h.destroy()
        self._rows.clear()
        self._group_headers = []
        t = self.theme
        for group_title, group_gloss, metrics in METRIC_GROUPS:
            present = [m for m in metrics if status.per_metric.get(m) is not None]
            if not present:
                continue
            hdr = ctk.CTkLabel(
                self, text=f"{group_title}   ·   {group_gloss}",
                font=t.font(t.SIZE_CAPTION, "bold"), text_color=t.TEXT_SECONDARY,
                anchor="w")
            hdr.pack(side="top", fill="x", pady=(t.SPACE_SM, 2))
            self._group_headers.append(hdr)
            for m in present:
                ms = status.per_metric[m]
                row = _MetricRow(self, ms=ms, theme=t, on_click=self._cb,
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
        # NOTE: `theme` the parameter, NOT self.theme — self.theme is never
        # assigned on _MetricRow. Referencing it blanked the ENTIRE drift tab
        # for every model at work (2026-07-10) because the per-widget guard
        # swallowed the AttributeError. Now covered by the app sweep.
        # Fraction metrics (fail/escape rates) read as percent everywhere —
        # "5.2% ± 2.0%" not "0.052 ± 0.02" (2026-07-13).
        _fmt = lambda v: format_metric_value(ms.metric, v, theme.fmt_measure)  # noqa: E731
        recent = _fmt(recent_val)
        # Honest shift, verifiable against the Baseline & Recent cells beside it:
        # (recent - baseline) / baseline_std. Replaces the old `magnitude` (CUSUM
        # distance past the limit), which couldn't be reconciled with the numbers shown.
        shift = ((recent_val - ms.baseline_mean) / ms.baseline_std
                 if (recent_val is not None and ms.baseline_std) else None)
        shift_txt = f"{shift:+.2f}σ" if shift is not None else "—"
        # Humanized alert type — the raw enum value ("slow_drift") leaked into
        # this table while the Triage cards humanized it (2026-07-07 sweep).
        alert_txt = ("Step change" if ms.alert_type == AlertType.STEP_CHANGE
                     else "Slow drift") if ms.alert_type else "—"
        cells = [metric_label(ms.metric), ms.tier.name.replace("_", " ").title(),
                 alert_txt,
                 f"{_fmt(ms.baseline_mean)} ± {_fmt(ms.baseline_std)}", recent, shift_txt]
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
