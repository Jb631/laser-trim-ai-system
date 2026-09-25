"""Spec 3b — TriagePage: the FOCUS list (top) + the browse list (bottom).

The mission landing view: "anything to look at today?" The FOCUS list answers
it in the order the work should be done; the browse list below lets James reach
any model on record. Clicking a FOCUS row deep-links to the Model page on the
metric that put the model there; clicking a browse row deep-links with the
model only.

2026-08-29 redesign — this page used to render a wall of per-model σ cards fed
by a drift-alert accessor in ml/manager.py (deleted 2026-08-31 once this page
stopped being its caller). The wall had no order anyone could act on and it never
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
import logging
import threading

import customtkinter as ctk

logger = logging.getLogger(__name__)

from laser_trim_analyzer.gui.v6.focus_data import focus_failed, load_focus
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets import blocks
from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
from laser_trim_analyzer.gui.v6.widgets.focus_list_zone import FocusListZone
from laser_trim_analyzer.ml.manager import active_model_set, list_known_models

# The 1280x720 audit's one known clip (carried in from facelift step 1's review): the FOCUS
# list's sparkline rows are the richest thing on the page, so on a tall window it can fill it,
# leaving the browse list squeezed to nothing. _FOCUS_ZONE_MAX_H caps it at roughly today's look
# (a handful of rich rows); _FOCUS_ZONE_MIN_H is "a heading and one row, scrollable" -- never
# less; _BROWSE_MIN_H is what the browse list is guaranteed no matter how short the window is.
_FOCUS_ZONE_MAX_H = 320
_FOCUS_ZONE_MIN_H = 130
_BROWSE_MIN_H = 200


class TriagePage(PageBase):
    page_title = "Triage"

    def __init__(self, master, *, theme, app, page_title="Triage"):
        self._show_all = False   # default: focus on ACTIVE (current-production) models
        # Last load, kept so the scope toggle can re-filter the browse list
        # without a second trip to the database.
        self._models = []
        self._active = set()
        self._browse_failed = None     # "ExcType: message" when the model list failed to load
        self._focus_header = None      # blocks.group_header wrap; rebuilt in _apply() (needs the count)
        # Reload generation (the Model page's _reload_gen pattern, facelift F4): a load's apply is
        # dropped unless it is the newest -- workers finish in any order, and an older, slower
        # load used to put its state back over a newer one's. Bumped and read on the Tk thread.
        self._reload_gen = 0
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    def header_actions(self, parent):
        t = self.theme
        self._scope = ctk.CTkSegmentedButton(
            parent, values=["Active", "All models"], command=self._on_scope_change,
            fg_color=t.CARD, selected_color=t.SEGMENT_SELECTED,
            selected_hover_color=t.SEGMENT_SELECTED_HOVER, unselected_color=t.CARD,
            unselected_hover_color=t.ELEVATED, text_color=t.TEXT_PRIMARY)
        self._scope.set("All models" if self._show_all else "Active")
        self._scope.pack(side="left")

    def build_content(self, parent):
        t = self.theme
        self._content_parent = parent
        # A failed loader is named here, never drawn as zeros (final review, 2026-09-24: a
        # crashed FOCUS computation read "Needs a look · 0" and "All models within tolerance",
        # a failed model list "All models · 0"). Packed above the focus header only while a
        # load has failed (_set_load_banner).
        self._load_banner = blocks.banner(parent, t, "", wrap_to=parent)
        # The focus zone's own heading is suppressed (show_heading=False) -- Triage draws its
        # OWN "Needs a look" blocks.group_header in _apply(), once the count is known, rather
        # than the zone's generic default text (see focus_list_zone.py). The zone is wrapped in
        # a height-BOUNDED frame (see _fit_focus_zone) so a short window always leaves the
        # browse list below it a usable minimum, instead of squeezing it out entirely.
        self._focus_wrap = ctk.CTkFrame(parent, fg_color="transparent", height=_FOCUS_ZONE_MAX_H)
        self._focus_wrap.pack(side="top", fill="x", pady=(0, t.SPACE_LG))
        self._focus_wrap.pack_propagate(False)
        self._focus = FocusListZone(self._focus_wrap, theme=t,
                                    on_row_click=self._on_focus_click, show_heading=False)
        self._focus.pack(side="top", fill="both", expand=True)
        self._browse = BrowseZone(parent, theme=t, on_row_click=self._on_row_click)
        self._browse.pack(side="top", fill="both", expand=True)
        # Bound ONCE, on `parent` -- this page (self) is never destroyed/rebuilt for its own
        # lifetime, so this never stacks a second handler (same rule blocks.wrap_to_width's own
        # docstring states for a label+container bound once).
        parent.bind("<Configure>", self._fit_focus_zone, add="+")
        self._fit_focus_zone()

    def _fit_focus_zone(self, _event=None) -> None:
        """Give the focus zone at most a share of the page's actual height, leaving the browse
        list its guaranteed minimum -- see the module docstring's "1280x720 clip" note.

        ONE unit throughout (final review, 2026-09-24): CustomTkinter's unscaled units, the
        unit of the three constants above, the theme's spacing and configure(height=), which
        CTk scales itself. winfo_height() is REAL pixels, so every measured height is turned
        back first -- mixing the two let the zone take its full 320 at 150% Windows scaling
        (480 real px) and squeezed the browse list's guaranteed 200 to a 56-px sliver."""
        t = self.theme
        try:
            total_px = self._content_parent.winfo_height()
        except Exception:
            return
        if total_px <= 1:
            return          # not laid out yet; a real <Configure> follows once it is
        unscaled = self._reverse_widget_scaling
        total = unscaled(total_px)
        header_h = unscaled(self._focus_header.winfo_height()) if self._focus_header is not None else 0
        banner_h = 0
        if self._load_banner.winfo_manager():        # a failure notice takes its room too
            banner_h = unscaled(self._load_banner.winfo_height()) + t.SPACE_SM
        budget = total - header_h - banner_h - t.SPACE_XS - _BROWSE_MIN_H - t.SPACE_LG
        focus_h = int(max(_FOCUS_ZONE_MIN_H, min(_FOCUS_ZONE_MAX_H, budget)))
        if self._focus_wrap.cget("height") != focus_h:      # cget: unscaled, like configure
            self._focus_wrap.configure(height=focus_h)

    # ---- data ----
    def reload_now(self):
        """Synchronous reload + apply (the test path; also the main-thread apply). The newest
        load: anything still in flight is dropped when it lands."""
        self._reload_gen += 1
        self._apply(*self._query())

    def on_show(self):
        """Reload on a background thread, apply on the Tk thread via safe_after -- unless a newer
        load has started since."""
        self._reload_gen += 1
        gen = self._reload_gen

        def work():
            data = self._query()

            def apply():
                if gen == self._reload_gen:         # else a newer load superseded this one
                    self._apply(*data)
            self.safe_after(apply)
        threading.Thread(target=work, daemon=True).start()

    def _query(self):
        """One DB pass -> (FocusResult, models, active, last, browse_failed). Worker-safe: no Tk.

        The FOCUS half goes through `focus_data.load_focus`, which Home calls
        too — the two landing screens must not be able to disagree about what
        is drifting; a crash there comes back MARKED (focus_data.focus_failed).
        The browse half is this page's alone: a crash there is `browse_failed`
        ("ExcType: message"), and both are named in the page's banner.
        """
        browse_failed = None
        try:
            models = list_known_models(self.app.db)
            cfg = getattr(self.app.config, "active_models", None)
            active = active_model_set(
                self.app.db,
                recent_days=getattr(cfg, "recent_days", 90) if cfg else 90,
                mps_models=getattr(cfg, "mps_models", None) if cfg else None)
        except Exception as exc:
            # Named in the banner, never drawn as an empty list ("no models on
            # record") -- a query crash must not look like an empty database.
            logger.exception("Triage query failed")
            browse_failed = f"{type(exc).__name__}: {exc}"
            models, active = [], set()
        # A failed inventory is no stamp for the FOCUS list's "last processed": let
        # load_focus read it for itself rather than hand it an empty list.
        result, last = load_focus(self.app.db, models=None if browse_failed else models)
        return result, models, active, last, browse_failed

    def _apply(self, result, models, active, last, browse_failed=None):
        # Rebuilt whole each load, like every other dynamic blocks.group_header in the app
        # (findings_view.py's own _render()) -- the count pill can only be right once the data
        # is in hand, and a FAILED load has no count at all (count=None: no pill).
        if self._focus_header is not None:
            try:
                self._focus_header.destroy()
            except Exception:
                pass
        focus_error = focus_failed(result)
        self._set_load_banner(focus_error, browse_failed)
        self._focus_header = blocks.group_header(
            self._content_parent, self.theme, "Needs a look",
            None if focus_error else len(result.focus))
        self._focus_header.pack(side="top", fill="x", pady=(0, self.theme.SPACE_XS),
                                before=self._focus_wrap)
        # The fit below runs while this brand-new header is still 1 px tall; once it is laid out
        # nothing else changes size, so without this the zone kept a budget computed without its
        # header (the browse list ended 2 px short of its minimum at 100%, 4 at 150%). Bound on
        # the header itself, which is destroyed with its binding on the next apply.
        self._focus_header.bind("<Configure>", self._fit_focus_zone, add="+")
        # The FocusResult goes to the zone untouched — one computation owns the
        # membership, the ranking and the wording (see module docstring).
        self._focus.set_result(result, last_processed=last)
        self._models, self._active = models, active
        self._browse_failed = browse_failed
        self._apply_browse()
        self._fit_focus_zone()

    def _set_load_banner(self, focus_error, browse_error) -> None:
        """Name every load that failed this pass -- the same shape as the Model and Dashboard
        pages' `_set_load_banner` -- or unpack the banner. `before=self._focus_wrap`: the header
        is packed after this, also before the wrap, so the banner leads the page."""
        failed = []
        if focus_error:
            failed.append(f"the drifting-now list ({focus_error})")
        if browse_error:
            failed.append(f"the model list ({browse_error})")
        if not failed:
            self._load_banner.pack_forget()
            return
        self._load_banner.configure(
            text=("⚠ Could not load: " + "; ".join(failed) + ". Those parts of this page are "
                  "empty — this is an error, not an absence of data. The log has the details."))
        self._load_banner.pack(side="top", fill="x", pady=(0, self.theme.SPACE_SM),
                               before=self._focus_wrap)

    def _apply_browse(self):
        """Render the browse list at the current scope. No DB access."""
        if self._browse_failed:
            self._browse.set_models(None)          # unavailable -- never "All models · 0"
            return
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
