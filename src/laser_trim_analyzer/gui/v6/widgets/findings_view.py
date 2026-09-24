"""Draws arranged findings: group headers, rows, the one open row, Show all.

Shared by the Findings page (every model, all four groups, an Open button) and the model
page's Findings tab (one model, empty groups hidden, no Open button). The RULES live in
findings/presentation.py; this only draws them.
"""
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import customtkinter as ctk

from laser_trim_analyzer.findings import presentation as P
from laser_trim_analyzer.gui.v6.widgets import blocks


def _join_cells(cells, fmt) -> str:
    """Join one value per track with ' · ', in the given (already-matched) order. A missing
    slot -- a track with no entry for this setting -- reads "—" in its OWN place, never
    silently dropped, so the joined string always has one part per track."""
    return " · ".join("—" if c is None else fmt(c) for c in cells)


class FindingsView(ctk.CTkFrame):
    def __init__(self, master, theme, *, on_open: Optional[Callable[[str], None]] = None,
                 include_empty: bool = True, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._on_open = on_open
        self._include_empty = include_empty
        self._findings: List[Dict[str, Any]] = []
        self._expanded: Set[str] = set()
        self._rows: Dict[Tuple, P.Row] = {}
        self.row_widgets: Dict[Tuple, ctk.CTkFrame] = {}
        self.open_key: Optional[Tuple] = None
        self._detail: Optional[ctk.CTkFrame] = None

    # ---- public ----
    def set_findings(self, findings: List[Dict[str, Any]]) -> None:
        self._findings = list(findings or [])
        self._render()

    def toggle(self, key: Tuple) -> None:
        """Open a row in place; opening another, or the same one again, closes it."""
        if self._detail is not None:
            self._detail.destroy()
            self._detail = None
        if self.open_key == key:
            self.open_key = None
            return
        self.open_key = key
        row_widget = self.row_widgets.get(key)
        if row_widget is not None:
            self._detail = self._draw_detail(self._rows[key])
            self._detail.pack(after=row_widget, fill="x", padx=self.theme.SPACE_SM,
                              pady=(0, self.theme.SPACE_SM))

    def show_all(self, group_key: str) -> None:
        self._expanded.add(group_key)
        self._render()

    # ---- drawing ----
    def _render(self) -> None:
        t = self.theme
        for child in self.winfo_children():
            child.destroy()
        self._rows.clear()
        self.row_widgets.clear()
        self._detail = None
        was_open, self.open_key = self.open_key, None
        for group in P.arrange(self._findings, include_empty=self._include_empty):
            spec = group.spec
            blocks.group_header(self, t, spec.title, len(group.rows), column=spec.column,
                                tone=spec.tone, meaning=spec.meaning
                                ).pack(fill="x", pady=(t.SPACE_LG, t.SPACE_XS))
            if not group.rows:
                ctk.CTkLabel(self, text=spec.empty, font=t.font(t.SIZE_BODY),
                             text_color=t.TEXT_SECONDARY, anchor="w", justify="left",
                             wraplength=1000).pack(fill="x", padx=t.SPACE_SM)
                continue
            shown = group.rows if spec.key in self._expanded else group.rows[:P.ROWS_PER_GROUP]
            for r in shown:
                tone = P.value_tone(spec.key, r.value, r.findings)
                color = t.PASS_FG if tone == "up" else t.CHECK if tone == "down" else None
                w = blocks.row(self, t, r.model, r.statement, P.value_text(spec.key, r.value),
                               tags=r.tags, on_click=lambda k=r.key: self.toggle(k), value_color=color)
                w.pack(fill="x")
                self._rows[r.key] = r
                self.row_widgets[r.key] = w
            if len(group.rows) > len(shown):
                blocks.link_button(self, t, f"Show all {len(group.rows)}",
                                   lambda k=spec.key: self.show_all(k)).pack(anchor="w", padx=t.SPACE_XS)
        if was_open in self.row_widgets:              # keep the open row open across Show all
            self.toggle(was_open)

    def _draw_detail(self, r: P.Row) -> ctk.CTkFrame:
        """Spec §3 'Opening a row': (1) the finding's own summary; (2) for cut_setting, ONE
        settings table -- setting, tracks, in spec -- one column per track when merged; (3)
        Open {model}. Review ruling 2026-09-24: the shipped first draft drew a heading +
        summary + a separate table PER finding; the approved mockup draws ONE narrative and
        ONE table whose cells join both tracks' numbers with " · "."""
        t = self.theme
        d = ctk.CTkFrame(self, fg_color=t.CARD, corner_radius=t.RADIUS_LG, border_width=1,
                         border_color=t.BORDER)
        # ONE summary for the row: the first finding's. The row's findings are already in
        # readout order (get_process_findings orders by tracks_per_year DESC, and arrange()
        # preserves that order as it merges), so "first" is the same finding a lone,
        # unmerged row would show -- not an arbitrary pick.
        first = r.findings[0]
        ctk.CTkLabel(d, text=str(first.get("summary") or ""), font=t.font(t.SIZE_BODY),
                     text_color=t.TEXT_PRIMARY, anchor="w", justify="left", wraplength=960
                     ).pack(fill="x", padx=t.SPACE_LG, pady=(t.SPACE_MD, 0))
        if first.get("analyzer") == "cut_setting":
            setting_rows = self._merged_settings(r.findings)
            if setting_rows:
                if r.merged:
                    # The table's cells below join each track's number in THIS order --
                    # name it, so the joined numbers are never ambiguous.
                    names = " · ".join(str((f.get("evidence") or {}).get("track") or "?")
                                       for f in r.findings)
                    ctk.CTkLabel(d, text=names, font=t.font(t.SIZE_CAPTION),
                                 text_color=t.TEXT_SECONDARY, anchor="w"
                                 ).pack(fill="x", padx=t.SPACE_LG, pady=(t.SPACE_SM, 0))
                self._settings_table(d, setting_rows)
        if self._on_open is not None:
            blocks.primary_button(d, t, f"Open {r.model}", lambda m=r.model: self._on_open(m)
                                  ).pack(anchor="w", padx=t.SPACE_LG, pady=t.SPACE_MD)
        else:
            ctk.CTkFrame(d, height=t.SPACE_SM, fg_color="transparent").pack()
        return d

    @staticmethod
    def _merged_settings(findings) -> List[Dict[str, Any]]:
        """One row per distinct cut setting across `findings`, in first-seen order (each
        finding's own evidence.group.settings, in that finding's own order). Each row's
        'cells' holds one entry per finding, MATCHED BY SETTING VALUE -- never by list
        position: two merged tracks can list their settings in a different order, or one can
        lack a setting the other has (that slot is None there, drawn as "—")."""
        per_finding: List[Dict[Any, Dict[str, Any]]] = []
        for f in findings:
            ev = f.get("evidence") or {}
            by_setting = {s.get("setting"): s for s in (ev.get("group") or {}).get("settings") or []}
            per_finding.append(by_setting)
        order: List[Any] = []
        seen: Set[Any] = set()
        for by_setting in per_finding:
            for setting in by_setting:
                if setting not in seen:
                    seen.add(setting)
                    order.append(setting)
        return [{"setting": setting, "cells": [by_setting.get(setting) for by_setting in per_finding]}
                for setting in order]

    def _settings_table(self, parent, setting_rows) -> None:
        t = self.theme
        grid = ctk.CTkFrame(parent, fg_color="transparent")
        grid.pack(anchor="w", padx=t.SPACE_LG, pady=(t.SPACE_SM, 0))
        for c, head in enumerate(("cut", "tracks", "in spec")):
            ctk.CTkLabel(grid, text=head, font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY,
                         anchor="w").grid(row=0, column=c, sticky="w", padx=(0, t.SPACE_XL))
        for i, row in enumerate(setting_rows, start=1):
            setting, cells = row["setting"], row["cells"]
            cut_text = f"{setting:g}" if isinstance(setting, (int, float)) else str(setting)
            tracks_text = _join_cells(cells, lambda s: f"{int(s.get('n') or 0):,}")
            spec_text = _join_cells(
                cells, lambda s: "—" if s.get("pass_pct") is None else f"{s.get('pass_pct'):.0f}%")
            for c, text in enumerate((cut_text, tracks_text, spec_text)):
                ctk.CTkLabel(grid, text=text, font=t.mono(t.SIZE_BODY), text_color=t.TEXT_PRIMARY,
                             anchor="w").grid(row=i, column=c, sticky="w", padx=(0, t.SPACE_XL))
