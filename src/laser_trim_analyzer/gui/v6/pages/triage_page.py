"""Spec 3b — TriagePage: flagged cards (top) + browse list (bottom).

The mission landing view: "anything to look at today?" Flagged-model cards
answer it at a glance; the browse list lets James reach any known model.
Clicking a card deep-links to the Model page with the triggering metric;
clicking a browse row deep-links with the model only.
"""
import threading

from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
from laser_trim_analyzer.gui.v6.widgets.flagged_cards_zone import FlaggedCardsZone
from laser_trim_analyzer.ml.manager import get_triage_alerts, list_known_models


class TriagePage(PageBase):
    page_title = "Triage"

    def build_content(self, parent):
        self._cards = FlaggedCardsZone(parent, theme=self.theme, on_card_click=self._on_card_click)
        self._cards.pack(side="top", fill="x", pady=(0, self.theme.SPACE_LG))
        self._browse = BrowseZone(parent, theme=self.theme, on_row_click=self._on_row_click)
        self._browse.pack(side="top", fill="both", expand=True)

    # ---- data ----
    def reload_now(self):
        """Synchronous reload + apply (the test path; also the main-thread apply)."""
        flagged, models, last = self._query()
        self._apply(flagged, models, last)

    def on_show(self):
        """Reload on a background thread, apply on the Tk thread via safe_after."""
        def work():
            flagged, models, last = self._query()
            self.safe_after(lambda: self._apply(flagged, models, last))
        threading.Thread(target=work, daemon=True).start()

    def _query(self):
        try:
            flagged = get_triage_alerts(self.app.db)
            models = list_known_models(self.app.db)
        except Exception:
            flagged, models = [], []
        last = max((m.last_processed for m in models if m.last_processed), default=None)
        return flagged, models, last

    def _apply(self, flagged, models, last):
        self._cards.set_summaries(flagged, last_processed=last)
        self._browse.set_models(models)

    # ---- routing ----
    def _on_card_click(self, model, focus_metric):
        self.app.set_model_route(model, focus_metric)
        self.app.show_page("model")

    def _on_row_click(self, model):
        self.app.set_model_route(model)        # no focus → Model page defaults the metric
        self.app.show_page("model")
