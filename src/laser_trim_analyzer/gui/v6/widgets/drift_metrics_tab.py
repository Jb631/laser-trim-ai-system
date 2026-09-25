"""Spec 3c — DriftMetricsTab: grouped table of every watched metric.

Rows render in METRIC_GROUPS order under three plain-language group headers
(process signals / trim outcome / final-test outcome) so the 12-metric list
reads as three questions, not a wall (James, 2026-07-13)."""
from typing import Callable, Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets import blocks
from laser_trim_analyzer.ml.drift_types import (
    AlertType, METRIC_GROUPS, ModelDriftStatus, format_metric_value,
    metric_label)

_COLUMNS = ["Metric", "Tier", "Alert", "Baseline (lot mean±σ)", "Last lot", "Shift (σ)"]

# Two of the six uniform columns need more than an even 1/6 share once the row is
# squeezed into a narrower window: column 0 (the metric NAME -- "Escape rate (trim
# PASS -> FT FAIL)", 225px at SIZE_BODY, the longest of drift_types.METRIC_LABELS)
# and column 3 (baseline mean +/- std, formatted by format_metric_value -- measured
# up to 152px, e.g. "0.003235 +/- 0.0007776" for untrimmed_sigma_gradient on 8232-1).
# Applied to BOTH the header row and every _MetricRow: they are separate grid
# instances (each frame owns its own columns), so giving only one of them a minsize
# would pull its columns out of alignment with the other's -- shared here so both
# always agree. Measured, not guessed: every watched metric's rendered width across
# two real models (8232-1, 6607) stays under these with margin; the other four
# columns' widest real content (a tier name, an alert type, a smoothness value, a
# shift) tops out at 64px, so narrowing them a little more to make room is safe.
# CustomTkinter's UNSCALED units, like every other size in gui/v6 -- but grid_columnconfigure's
# minsize goes straight to Tk as REAL pixels (CTk scales a grid's padx, never a column's minsize),
# so both call sites below scale it (_apply_widget_scaling). Unscaled, the name column stayed
# 240 real px at 150% Windows scaling while its text grew to 318 px, and every long metric name
# was cut (render_pages.py --audit --scaling 1.5, final review 2026-09-24).
_COL_MINSIZE = {0: 240, 3: 175}


class DriftMetricsTab(ctk.CTkScrollableFrame):
    def __init__(self, master, theme: ThemeManager, on_metric_select: Callable[[str], None],
                 on_requalify: Callable[[], None] = None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_metric_select
        self._rows: Dict[str, _MetricRow] = {}
        self._group_headers: List = []
        self._on_requalify = on_requalify
        # The full σ explanation (facelift step 2, 2026-09-24): the model page's own key,
        # right under its pills, is now a single line pointing here -- this is where the
        # baseline/recent/shift numbers below actually live. Built once, here, and never
        # rebuilt: set_status()/clear() only ever touch _rows and _group_headers.
        self._sigma_key_lbl = ctk.CTkLabel(
            self, text=("σ = how far the last LOT's median sits from this model's baseline "
                        "of historical lot medians (lot = production run; new lot after "
                        ">3 idle days). +1.0σ = the last lot ran one lot-σ above normal. "
                        "Drift signal, not a spec."),
            font=theme.font(theme.SIZE_CAPTION), text_color=theme.TEXT_SECONDARY,
            anchor="w", justify="left")
        self._sigma_key_lbl.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        # `self` (this tab) is built once and never destroyed/rebuilt for the page's whole
        # lifetime, so this binds exactly once (blocks.wrap_to_width: call it once per
        # (label, container) lifetime, never from inside a re-render/apply path).
        blocks.wrap_to_width(self._sigma_key_lbl, self)
        header = ctk.CTkFrame(self, fg_color=theme.CARD)
        header.pack(side="top", fill="x", pady=(0, theme.SPACE_XS))
        for i in range(len(_COLUMNS)):
            header.grid_columnconfigure(i, weight=1, uniform="dm",
                                        minsize=header._apply_widget_scaling(_COL_MINSIZE.get(i, 0)))
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

    def clear(self) -> None:
        """Back to the just-constructed look: no rows, no group headers —
        only the column header and the baseline footer remain. Called when
        this model's drift status failed to load, so the PREVIOUS model's
        rows never sit on screen under this model's name."""
        for r in self._rows.values():
            r.destroy()
        # __dict__.get, not getattr: see set_status() above for why.
        for h in (self.__dict__.get("_group_headers") or []):
            h.destroy()
        self._rows.clear()
        self._group_headers = []


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
            # _COL_MINSIZE (module level): columns 0 and 3 need more than an equal
            # 1/6 share once six columns are squeezed into a narrower window; taken
            # from the other four, which hold short fixed-format values ("+1.50σ")
            # with plenty of spare width. Kept in sync with the header row's OWN
            # grid_columnconfigure call above (a separate grid instance) so the two
            # stay column-aligned.
            self.grid_columnconfigure(i, weight=1, uniform="dm",
                                      minsize=self._apply_widget_scaling(_COL_MINSIZE.get(i, 0)))
        for i, txt in enumerate(cells):
            lbl = ctk.CTkLabel(self, text=txt, font=theme.font(theme.SIZE_BODY),
                               text_color=theme.TEXT_PRIMARY, anchor="w")
            lbl.grid(row=0, column=i, sticky="ew", padx=theme.SPACE_SM, pady=theme.SPACE_XS)
            lbl.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self):
        self._cb(self.metric)
