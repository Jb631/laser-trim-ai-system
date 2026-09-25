"""Spec 3b — BrowseZone: search + scrollable model list (tier word, last-processed date)."""
from datetime import datetime
from typing import Callable, Dict, List, Optional

import customtkinter as ctk

from laser_trim_analyzer.core.activity import inactive_tag
from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets import blocks
from laser_trim_analyzer.ml.drift_types import ModelSummary

ROW_CAP = 200  # render cap for responsiveness; cap is disclosed (Q10)


def _tier_label(tier) -> str:
    """DriftTier -> sentence case ('OUT_OF_CONTROL' -> 'Out of control'), the same pattern
    drift_metrics_tab.py uses (ms.tier.name.replace('_', ' ').title()), lower-cased to sentence
    case since this is body text, not a verdict badge (global-constraints.md)."""
    return tier.name.replace("_", " ").capitalize()


class BrowseZone(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_row_click: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._cb = on_row_click
        self._models: List[ModelSummary] = []
        self._failed = False                           # set_models(None): the load failed
        self._inactive = {}                            # {model: newest trim file} -- see set_models
        self._rows: List[ctk.CTkFrame] = []
        self._header: Optional[ctk.CTkFrame] = None   # blocks.group_header wrap; rebuilt each _render()
        t = theme
        # Row anatomy, spelled out (live-walk finding, 2026-07-08: an unlabeled date column read
        # as decoration). The status word IS the row's own "statement" column now (blocks.row) --
        # this used to be a bare colour dot, word-less (a colour-blind reader had nothing to read).
        # No claim about ORDER (final review, 2026-09-24: it said "worst first" over an
        # alphabetical list): this is the lookup list; "Needs a look" above is the ranked one.
        self._legend = ctk.CTkLabel(self, text=(
                "Status = drift tier. Date = last processed. 'Active' scope = "
                "models with recent data or pinned in Settings → Active Models. A model not "
                "trimmed in the two years before the newest file reads Inactive instead — still "
                "listed, never hidden."),
                font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY, anchor="w", justify="left")
        self._legend.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        # Bound ONCE: `self` (this zone) is never destroyed/rebuilt for its own lifetime, so this
        # never stacks a second <Configure> handler (blocks.wrap_to_width's own rule). Replaces a
        # fixed wraplength=950 (global-constraints.md: no fixed pixel wraplength on page-width text).
        blocks.wrap_to_width(self._legend, self)
        # NOTE: no textvariable — CTkEntry silently drops placeholder_text when
        # a textvariable is attached (the 'mystery empty box' finding). Filter
        # reacts on KeyRelease instead.
        self._search_entry = ctk.CTkEntry(self, placeholder_text="Type to filter models…",
                     font=t.font(t.SIZE_BODY), fg_color=t.CARD, border_color=t.BORDER,
                     text_color=t.TEXT_PRIMARY)
        self._search_entry.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        self._search_entry.bind("<KeyRelease>", lambda e: self._render())
        self._cap_label = ctk.CTkLabel(self, text="", font=t.font(t.SIZE_CAPTION),
                                       text_color=t.TEXT_SECONDARY, anchor="w")
        self._cap_label.pack(side="top", fill="x")
        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)
        self._render()

    def set_models(self, models: Optional[List[ModelSummary]], *,
                   inactive: Optional[Dict[str, datetime]] = None) -> None:
        """`None` means the model list could not be LOADED -- say so, with no count: a failed
        query drawn as "All models · 0" reads as an empty database (final review, 2026-09-24).
        The page's banner names the failure.

        `inactive` = {model: newest trim file} of the models core/activity calls inactive (James,
        2026-09-25): their status reads "Inactive · last trimmed Mon YYYY", never a drift tier --
        a tier means nothing without recent data. Nothing is filtered out."""
        self._failed = models is None
        self._models = list(models or [])
        self._inactive = dict(inactive or {})
        self._render()

    def set_filter(self, text: str) -> None:
        self._search_entry.delete(0, "end")
        if text:
            self._search_entry.insert(0, text)
        self._render()

    def _render(self) -> None:
        t = self.theme
        for r in self._rows:
            r.destroy()
        self._rows.clear()
        if self._header is not None:
            try:
                self._header.destroy()
            except Exception:
                pass
        flt = self._search_entry.get().strip().lower()
        matches = [m for m in self._models if not flt or flt in m.model.lower()]
        # Rebuilt each render (same pattern as findings_view.py's own _render()): the count pill
        # can only be right once the current filter/scope is applied. Packed BEFORE the legend,
        # which is built once in __init__ and never moves.
        self._header = blocks.group_header(self, t, "All models",
                                           None if self._failed else len(matches))
        self._header.pack(side="top", fill="x", pady=(0, t.SPACE_XS), before=self._legend)
        if self._failed:
            lbl = ctk.CTkLabel(self._list, text="Unavailable — the notice above says why.",
                               font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY, anchor="w")
            lbl.pack(side="top", fill="x", pady=t.SPACE_MD)
            self._rows.append(lbl)                 # destroyed with the rows on the next render
            self._cap_label.configure(text="")
            return
        for m in matches[:ROW_CAP]:
            status = (inactive_tag(self._inactive[m.model]) if m.model in self._inactive
                      else _tier_label(m.tier))
            row = blocks.row(self._list, t, m.model, status,
                             m.last_processed.strftime("%Y-%m-%d") if m.last_processed else "—",
                             on_click=lambda mm=m.model: self._cb(mm))
            row._summary = m     # test hook, alongside blocks.row's own _on_click_all/_set_hover
            row.pack(side="top", fill="x")
            self._rows.append(row)
        if len(matches) > ROW_CAP:
            self._cap_label.configure(
                text=f"Showing {ROW_CAP} of {len(matches)} — narrow with search.")
        else:
            self._cap_label.configure(text="")
