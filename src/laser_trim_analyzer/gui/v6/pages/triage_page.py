"""Spec 3b — TriagePage: the FOCUS list (top) + the browse list (bottom).

The mission landing view: "anything to look at today?" The FOCUS list answers
it in the order the work should be done; the browse list below lets James reach
any model on record. Clicking a FOCUS row deep-links to the Model page on the
metric that put the model there; clicking a browse row deep-links with the
model only.

2026-08-29 redesign — this page used to render a wall of per-model σ cards fed
by `get_triage_alerts`. The wall had no order anyone could act on and it never
cleared itself, so the first question every morning ("what do I work on?")
still needed a human sort. `compute_focus_list` now owns both halves of that
answer: MEMBERSHIP (a model is listed only while one of its last RECENT_K lots
sits outside its own control limits — so it self-clears) and ORDER (extra
failing units per week). This page computes that ONCE per refresh and hands the
same `FocusResult` to the zone; it never re-ranks, re-filters or re-derives a
number, because that is exactly how the old list and the old chart ended up
telling two different stories.

Scope toggle (Active / All): filters the BROWSE list only. Membership in FOCUS
is the computation's call — it already drops models with no recent production —
and dropping rows from it here would make the list contradict the rule printed
directly above it.
"""
import threading

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
from laser_trim_analyzer.gui.v6.widgets.focus_list_zone import FocusListZone
from laser_trim_analyzer.ml.manager import active_model_set, list_known_models
from laser_trim_analyzer.ml.spc import FocusResult, compute_focus_list

_EMPTY = FocusResult(focus=[], chronic=[], anchor=None)


class TriagePage(PageBase):
    page_title = "Triage"

    def __init__(self, master, *, theme, app, page_title="Triage"):
        self._show_all = False   # default: focus on ACTIVE (current-production) models
        # Last load, kept so the scope toggle can re-filter the browse list
        # without a second trip to the database.
        self._models = []
        self._active = set()
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
                          "drifting now, biggest first — one verdict per lot, self-clearing")
        self._focus = FocusListZone(parent, theme=self.theme,
                                    on_row_click=self._on_focus_click)
        self._focus.pack(side="top", fill="x", pady=(0, self.theme.SPACE_LG))
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
        """One DB pass -> (FocusResult, models, active, last). Worker-safe: no Tk."""
        try:
            result = compute_focus_list(self.app.db)
            models = list_known_models(self.app.db)
            cfg = getattr(self.app.config, "active_models", None)
            active = active_model_set(
                self.app.db,
                recent_days=getattr(cfg, "recent_days", 90) if cfg else 90,
                mps_models=getattr(cfg, "mps_models", None) if cfg else None)
        except Exception:
            result, models, active = _EMPTY, [], set()
        last = max((m.last_processed for m in models if m.last_processed), default=None)
        return result, models, active, last

    def _apply(self, result, models, active, last):
        # The FocusResult goes to the zone untouched — one computation owns the
        # membership, the ranking and the wording (see module docstring).
        self._focus.set_result(result, last_processed=last)
        self._models, self._active = models, active
        self._apply_browse()

    def _apply_browse(self):
        """Render the browse list at the current scope. No DB access."""
        models = self._models
        # Default scope hides inactive/legacy models: most models in a
        # long-lived DB were last run years ago — real, but not "today".
        if not self._show_all and self._active:
            models = [m for m in models if m.model in self._active]
        self._browse.set_models(models)

    # ---- events ----
    def _on_scope_change(self, value):
        self._show_all = (value == "All models")
        # Re-filter what is already loaded rather than re-querying: the toggle
        # is a VIEW filter over the browse list, and the FOCUS list must not
        # flicker (or re-rank) because someone widened the model list.
        self._apply_browse()

    # ---- routing ----
    def _on_focus_click(self, model, focus_metric):
        self.app.set_model_route(model, focus_metric)
        self.app.show_page("model")

    def _on_row_click(self, model):
        self.app.set_model_route(model)        # no focus → Model page defaults the metric
        self.app.show_page("model")
