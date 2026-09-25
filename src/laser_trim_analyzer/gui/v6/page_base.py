"""Spec 3a — PageBase + _PageHeader. Foundations §2.1.

Subclass contract:
  * class attr page_title (or pass page_title=...)
  * build_content(parent)  — REQUIRED
  * header_actions(parent)  — OPTIONAL; build widgets WITH `parent` and pack them right
  * on_show() / on_hide()   — OPTIONAL
PageBase stores `self.app` (V6App | None) and `self.theme`, and offers safe_after().
"""
import logging
from typing import Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets import blocks

logger = logging.getLogger(__name__)

HEADER_HEIGHT = 44


class PageBase(ctk.CTkFrame):
    page_title: str = "Untitled"

    def __init__(self, master, *, theme: ThemeManager, app=None,
                 page_title: Optional[str] = None, **kwargs):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=0, **kwargs)
        self.theme = theme
        self.app = app
        if page_title is not None:
            self.page_title = page_title
        self._build_chrome()
        self.build_content(self._content)

    # ---- subclass interface ----
    def build_content(self, parent) -> None:
        raise NotImplementedError(f"{type(self).__name__} must override build_content(parent)")

    def header_actions(self, parent) -> None:
        """Optional. Construct action widgets with `parent` as master; pack side='right'."""
        return None

    def on_show(self) -> None: pass
    def on_hide(self) -> None: pass

    # ---- shared section chrome (2026-07-13 design pass: James asked for
    # "clear sections for what im looking at and what the app is telling
    # me" — pages mark INTERPRETATION zones vs DATA zones with this). ----
    def _zone_header(self, parent, title: str, caption: str) -> ctk.CTkFrame:
        """A section inside a page: sentence-case title, its caption on the line below.

        This used to be an 11 px all-caps label in the accent colour -- the hardest text on
        the screen to read, and much of the 'dated' look (spec 2026-09-23). Callers now pass
        sentence case. Returns the header's frame, so a caller can pack something above it."""
        t = self.theme
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(side="top", fill="x", pady=(t.SPACE_SM, t.SPACE_XS))
        ctk.CTkLabel(wrap, text=title, font=t.font(t.SIZE_HEADING, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w").pack(fill="x")
        if caption:
            ctk.CTkLabel(wrap, text=caption, font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY,
                         anchor="w", justify="left", wraplength=1000).pack(fill="x")
        return wrap

    def set_caption(self, text: str) -> None:
        """One line under the page title -- a page's headline in words. '' hides it.

        Keyed on the geometry manager's own state (winfo_manager() == "pack"), not
        winfo_ismapped(): PageContainer switches pages with grid() + tkraise(), which only
        changes stacking order, so a hidden (non-front) page stays winfo_ismapped() == 1 --
        ismapped was only ever false under this suite's withdrawn tk_root, never in the real
        app. But "is it on screen" was the wrong question regardless: what set_caption needs
        to know is "is the caption currently packed," and winfo_manager() answers that
        directly, independent of visibility."""
        self._caption.configure(text=text or "")
        packed = self._caption.winfo_manager() == "pack"
        if text and not packed:
            self._caption.pack(side="top", fill="x", padx=self.theme.SPACE_LG,
                               before=self._content, pady=(0, self.theme.SPACE_SM))
        elif not text and packed:
            self._caption.pack_forget()

    # ---- thread-safe UI update (foundations §2.4, reworked 2026-07-06) ----
    def safe_after(self, fn, delay: int = 0) -> None:
        """Run fn on the UI thread, guarded against widget destruction.

        Safe to call from ANY thread. The old implementation called
        winfo_exists()/after() directly from worker threads — Tkinter is not
        thread-safe, and with the main thread blocked (e.g. on the DB lock)
        that could stall or deadlock the app. Now workers only enqueue onto a
        plain queue (ui_dispatch.py); every Tk call happens on the main loop.

        A callback that raises is LOGGED, with its traceback, and the next one still runs (F4
        review: this used to be a silent `pass`, so a render crash left a stale screen and no
        trace). A page -- or an app -- already gone is not an error: nothing is left to update.
        """
        def guarded():
            try:
                alive = self.winfo_exists()
            except Exception:
                return                   # the app itself is being torn down
            if not alive:
                return
            try:
                fn()
            except Exception:
                logger.exception("%s: a screen update failed", type(self).__name__)

        dispatcher = getattr(self.app, "ui", None)
        if dispatcher is not None:
            if delay <= 0:
                dispatcher.post(guarded)
            else:
                # Register the delay on the main thread, then run guarded.
                dispatcher.post(lambda: self.winfo_exists() and self.after(delay, guarded))
            return

        # No dispatcher (tests / standalone page): legacy path — only correct
        # when called from the main thread, which is how tests drive pages.
        try:
            if self.winfo_exists():
                self.after(delay, guarded)
        except Exception:
            pass

    # ---- internal ----
    def _build_chrome(self) -> None:
        self._header = _PageHeader(self, theme=self.theme, title=self.page_title)
        self._header.pack(side="top", fill="x")
        ctk.CTkFrame(self, height=1, fg_color=self.theme.DIVIDER, corner_radius=0)\
            .pack(side="top", fill="x")
        # Created here but NOT packed -- set_caption() packs it (before self._content) the
        # first time a page gives it text, and unpacks it again on "".
        self._caption = ctk.CTkLabel(self, text="", font=self.theme.font(self.theme.SIZE_BODY),
                                     text_color=self.theme.TEXT_SECONDARY, anchor="w",
                                     justify="left")
        # No fixed pixel wraplength on page-width text (global-constraints.md): a caption is
        # one sentence, but Investigate's (facelift step 2, Task 2) can run to 4-5 clauses
        # joined by " · " and genuinely overflows an unwrapped single line -- found by
        # render_pages.py --audit, squeezed at both audited window sizes. `self` (this page)
        # is never destroyed/rebuilt for its own lifetime, so this binds exactly once
        # (blocks.wrap_to_width: call it once per (label, container) lifetime). Padding
        # matches set_caption's own padx=SPACE_LG on both sides, below.
        blocks.wrap_to_width(self._caption, self, padding=self.theme.SPACE_LG * 2)
        self._content = ctk.CTkFrame(self, fg_color="transparent")
        self._content.pack(fill="both", expand=True,
                           padx=self.theme.SPACE_LG, pady=self.theme.SPACE_MD)
        # Build header actions into the header's actions frame (correct parent).
        self.header_actions(self._header.actions_frame)


class _PageHeader(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, title: str):
        super().__init__(master, height=HEADER_HEIGHT, fg_color=theme.SURFACE, corner_radius=0)
        self.theme = theme
        self.pack_propagate(False)
        ctk.CTkLabel(self, text=title, font=theme.font(theme.SIZE_TITLE, "bold"),
                     text_color=theme.TEXT_PRIMARY, anchor="w")\
            .pack(side="left", fill="y", padx=(theme.SPACE_LG, theme.SPACE_MD))
        # Right-aligned actions frame; subclasses pack widgets here.
        self.actions_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.actions_frame.pack(side="right", fill="y", padx=(theme.SPACE_MD, theme.SPACE_LG))
