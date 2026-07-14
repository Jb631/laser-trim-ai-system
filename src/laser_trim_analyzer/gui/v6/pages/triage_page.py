"""Spec 3b — TriagePage: flagged cards (top) + browse list (bottom).

The mission landing view: "anything to look at today?" Flagged-model cards
answer it at a glance; the browse list lets James reach any known model.
Clicking a card deep-links to the Model page with the triggering metric;
clicking a browse row deep-links with the model only.

Scope toggle (Active / All): defaults to ACTIVE so the operator focuses on
current-production models. Most flagged models in a long-lived DB are legacy
units last run years ago — real, but not "what to look at today". Flip to
"All models" to see everything.
"""
import threading

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
from laser_trim_analyzer.gui.v6.widgets.flagged_cards_zone import FlaggedCardsZone
from laser_trim_analyzer.ml.manager import (
    active_model_set, get_triage_alerts, list_known_models)


class TriagePage(PageBase):
    page_title = "Triage"

    def __init__(self, master, *, theme, app, page_title="Triage"):
        self._show_all = False   # default: focus on ACTIVE (current-production) models
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    def header_actions(self, parent):
        t = self.theme
        self._scope = ctk.CTkSegmentedButton(
            parent, values=["Active", "All models"], command=self._on_scope_change,
            fg_color=t.CARD, selected_color=t.ACCENT, selected_hover_color=t.ACCENT_HOVER,
            unselected_color=t.CARD, text_color=t.TEXT_PRIMARY)
        self._scope.set("All models" if self._show_all else "Active")
        self._scope.pack(side="left")

    def build_content(self, parent):
        # Same two-zone framing as the Model page (2026-07-13 design pass):
        # the app's read first, the raw model list below it.
        self._zone_header(parent, "WHAT THE APP IS TELLING YOU",
                          "models whose last lot moved out of family — worst first")
        self._cards = FlaggedCardsZone(parent, theme=self.theme, on_card_click=self._on_card_click)
        self._cards.pack(side="top", fill="x", pady=(0, self.theme.SPACE_LG))
        self._zone_header(parent, "WHAT YOU'RE LOOKING AT",
                          "every model on record — click one to see its data")
        self._browse = BrowseZone(parent, theme=self.theme, on_row_click=self._on_row_click)
        self._browse.pack(side="top", fill="both", expand=True)

    # ---- data ----
    def reload_now(self):
        """Synchronous reload + apply (the test path; also the main-thread apply)."""
        self._apply(*self._query())

    def on_show(self):
        """Reload on a background thread, apply on the Tk thread via safe_after."""
        def work():
            data = self._query()
            self.safe_after(lambda: self._apply(*data))
        threading.Thread(target=work, daemon=True).start()

    def _query(self):
        try:
            flagged = get_triage_alerts(self.app.db)
            models = list_known_models(self.app.db)
            cfg = getattr(self.app.config, "active_models", None)
            active = active_model_set(
                self.app.db,
                recent_days=getattr(cfg, "recent_days", 90) if cfg else 90,
                mps_models=getattr(cfg, "mps_models", None) if cfg else None)
        except Exception:
            flagged, models, active = [], [], set()
        last = max((m.last_processed for m in models if m.last_processed), default=None)
        return flagged, models, active, last

    def _apply(self, flagged, models, active, last):
        # Default scope hides inactive/legacy models; "All models" shows everything.
        if not self._show_all and active:
            flagged = [a for a in flagged if a.model in active]
            models = [m for m in models if m.model in active]
        # Focus order (work finding #5, 2026-07-10: "44 models out of control,
        # hard to know where to focus"): models still in production first
        # (recent data), then by how far they have actually moved. A legacy
        # model can wait; a drifting model you're building TODAY cannot.
        last_by_model = {m.model: m.last_processed for m in models if m.last_processed}
        if last_by_model:
            newest = max(last_by_model.values())
            def _key(a):
                seen = last_by_model.get(a.model)
                days_stale = (newest - seen).days if seen else 9999
                recent = 0 if days_stale <= 90 else 1
                return (recent, -(abs(a.sigma_shift) if a.sigma_shift is not None else 0.0))
            flagged = sorted(flagged, key=_key)
        self._cards.set_summaries(flagged, last_processed=last)
        self._browse.set_models(models)

    # ---- events ----
    def _on_scope_change(self, value):
        self._show_all = (value == "All models")
        self.on_show()

    # ---- routing ----
    def _on_card_click(self, model, focus_metric):
        self.app.set_model_route(model, focus_metric)
        self.app.show_page("model")

    def _on_row_click(self, model):
        self.app.set_model_route(model)        # no focus → Model page defaults the metric
        self.app.show_page("model")
