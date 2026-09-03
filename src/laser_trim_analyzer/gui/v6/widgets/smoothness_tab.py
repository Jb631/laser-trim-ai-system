"""Spec 3c — SmoothnessTab: max_smoothness_value trend + recent test list (dict-driven)."""
from datetime import datetime
from typing import Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
# The shared render budget for the Model page's row lists — rationale lives in
# units_tab's module docstring. Smoothness data is small today, but this tab has
# the same 200-row exposure as the units tab and the same per-row build cost.
from laser_trim_analyzer.gui.v6.widgets.units_tab import RowBudgetMixin


class SmoothnessTab(RowBudgetMixin, ctk.CTkFrame):
    _ROW_NOUN = "smoothness tests"

    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._chart = FocusChart(self, theme=theme)
        self._chart.pack(side="top", fill="x", pady=(0, theme.SPACE_XS))
        # Which models HAVE smoothness data (work finding #14: with no
        # overview page, you had to guess which model to select).
        self._models_hint = ctk.CTkLabel(self, text="", font=theme.font(theme.SIZE_CAPTION),
                                         text_color=theme.TEXT_SECONDARY, anchor="w",
                                         justify="left", wraplength=1200)
        self._models_hint.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)
        self._rows: List[ctk.CTkFrame] = []
        self._records: List[Dict] = []
        self._expanded = False        # render budget — see units_tab docstring
        # One frame holding every row: a refresh is one teardown, not N.
        self._rows_host = ctk.CTkFrame(self._list, fg_color="transparent")
        self._rows_host.pack(side="top", fill="x")
        # Persistent — never destroyed from inside its own command.
        self._show_all_btn = ctk.CTkButton(self._list, text="", command=self._toggle_expand,
                                           fg_color="transparent", hover_color=theme.CARD,
                                           text_color=theme.ACCENT, anchor="w", height=26,
                                           font=theme.font(theme.SIZE_BODY))

    def set_models_hint(self, items) -> None:
        """items: [(model, count), ...] — models with smoothness records."""
        if items:
            listing = " · ".join(f"{m} ({n})" for m, n in items)
            self._models_hint.configure(
                text=f"Models with smoothness data: {listing}")
        else:
            self._models_hint.configure(text="")

    def set_records(self, records: List[Dict]) -> None:
        self._records = list(records)
        self._expanded = False   # "show all" described the previous model's list
        self._render()

    def _render(self) -> None:
        self._reset_rows_host()
        records, t = self._records, self.theme
        # Q5: pair date+value together. The CHART always sees every record —
        # the render budget caps rows, never the series behind the trend.
        pairs = sorted(((r["file_date"], r["max_smoothness_value"]) for r in records
                        if r.get("file_date") is not None and r.get("max_smoothness_value") is not None),
                       key=lambda p: p[0])
        self._chart.set_series(metric="max_smoothness_value",
                               dates=[p[0] for p in pairs], values=[p[1] for p in pairs])
        if not records:
            lbl = ctk.CTkLabel(self._rows_host, text="No smoothness records for this model.",
                               font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY)
            lbl.pack(pady=t.SPACE_LG)
            self._rows.append(lbl)
            return
        # None file_dates are legitimate (rendered as "—" below), but `or 0`
        # mixed datetime with int and sorted() raised TypeError — blanking this
        # tab and, before per-tab isolation, every tab applied after it.
        ordered = sorted(records, key=lambda r: r.get("file_date") or datetime.min, reverse=True)
        for r in self._budget_slice(ordered):
            row = ctk.CTkFrame(self._rows_host, fg_color=t.CARD)
            row.pack(side="top", fill="x", pady=1)
            when = r["file_date"].strftime("%Y-%m-%d") if r.get("file_date") else "—"
            def _fmt(v):
                return f"{v:.4g}" if isinstance(v, (int, float)) else "—"
            # Max deviation vs its spec is the smoothness quality measure. (The old "avg"
            # column was mean output voltage, not an average deviation — it could exceed
            # max — so it's dropped rather than shown as a misleading smoothness number.)
            sp = r.get("smoothness_pass")
            verdict = "PASS" if sp is True else "FAIL" if sp is False else "—"
            txt = (f"{r.get('serial') or '—'} · {when} · "
                   f"max dev={_fmt(r.get('max_smoothness_value'))} / spec "
                   f"{_fmt(r.get('smoothness_spec'))} · {verdict}")
            ctk.CTkLabel(row, text=txt, font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_PRIMARY)\
                .pack(side="left", padx=t.SPACE_SM, pady=t.SPACE_XS)
            self._rows.append(row)
        self._apply_show_all(len(ordered))
