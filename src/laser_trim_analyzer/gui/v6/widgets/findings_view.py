"""Draws arranged findings: group headers, rows, the one open row, Show all.

Shared by the Findings page (every model, all four groups, an Open button) and the model
page's Findings tab (one model, empty groups hidden, no Open button). The RULES live in
findings/presentation.py; this only draws them.
"""
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import customtkinter as ctk

from laser_trim_analyzer.findings import presentation as P
from laser_trim_analyzer.gui.v6.widgets import blocks


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
                tone = P.value_tone(spec.key, r.value)
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
        t = self.theme
        d = ctk.CTkFrame(self, fg_color=t.CARD, corner_radius=t.RADIUS_LG, border_width=1,
                         border_color=t.BORDER)
        for f in r.findings:
            ev = f.get("evidence") or {}
            if r.merged and ev.get("track"):
                ctk.CTkLabel(d, text=str(ev["track"]), font=t.font(t.SIZE_BODY, "bold"),
                             text_color=t.TEXT_PRIMARY, anchor="w").pack(fill="x", padx=t.SPACE_LG,
                                                                         pady=(t.SPACE_MD, 0))
            ctk.CTkLabel(d, text=str(f.get("summary") or ""), font=t.font(t.SIZE_BODY),
                         text_color=t.TEXT_PRIMARY, anchor="w", justify="left", wraplength=960
                         ).pack(fill="x", padx=t.SPACE_LG, pady=(t.SPACE_SM, 0))
            settings = ((ev.get("group") or {}).get("settings")) if f.get("analyzer") == "cut_setting" else None
            if settings:
                self._settings_table(d, settings)
        if self._on_open is not None:
            blocks.primary_button(d, t, f"Open {r.model}", lambda m=r.model: self._on_open(m)
                                  ).pack(anchor="w", padx=t.SPACE_LG, pady=t.SPACE_MD)
        else:
            ctk.CTkFrame(d, height=t.SPACE_SM, fg_color="transparent").pack()
        return d

    def _settings_table(self, parent, settings) -> None:
        t = self.theme
        grid = ctk.CTkFrame(parent, fg_color="transparent")
        grid.pack(anchor="w", padx=t.SPACE_LG, pady=(t.SPACE_SM, 0))
        for c, head in enumerate(("cut", "tracks", "in spec", "ran")):
            ctk.CTkLabel(grid, text=head, font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY,
                         anchor="w").grid(row=0, column=c, sticky="w", padx=(0, t.SPACE_XL))
        for i, s in enumerate(settings, start=1):
            pct = s.get("pass_pct")
            cells = (f"{s.get('setting'):g}" if isinstance(s.get("setting"), (int, float)) else str(s.get("setting")),
                     f"{int(s.get('n') or 0):,}", "—" if pct is None else f"{pct:.0f}%",
                     str(s.get("window") or ""))
            for c, text in enumerate(cells):
                ctk.CTkLabel(grid, text=text, font=t.mono(t.SIZE_BODY) if c < 3 else t.font(t.SIZE_CAPTION),
                             text_color=t.TEXT_PRIMARY if c < 3 else t.TEXT_SECONDARY, anchor="w"
                             ).grid(row=i, column=c, sticky="w", padx=(0, t.SPACE_XL))
