"""Spec 3a — PageBase + _PageHeader. Foundations §2.1.

Subclass contract:
  * class attr page_title (or pass page_title=...)
  * build_content(parent)  — REQUIRED
  * header_actions(parent)  — OPTIONAL; build widgets WITH `parent` and pack them right
  * on_show() / on_hide()   — OPTIONAL
PageBase stores `self.app` (V6App | None) and `self.theme`, and offers safe_after().

A page the top bar names never repeats its name (option B, 2026-10-04 -- the finish list's second
item: "Overview" twice, "Models" over "8232-1"). A page the bar has no item for -- Process, Findings,
Company trends -- lights nothing there, and so names itself: V6App calls `show_name()`, which puts
`page_title` at the left of the row under the bar, in the title face (review of option B, the same
day: those three were named nowhere). The row holds a page's header_actions, right-aligned, and is
there only when it has a name or an action (the Model page's pickers and buttons, Company trends'
window); a page with neither starts straight under the bar, with no empty band.
"""
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets import blocks

logger = logging.getLogger(__name__)

HEADER_HEIGHT = 44

# A screen update that KEEPS failing (F5 review): the 4 Hz ProgressTicker posts its paint through
# safe_after for a whole ingest run -- hours -- so one persistent bug would write a traceback four
# times a second. Each callback's FIRST failure is logged with its traceback; after that, at most
# one line per _REPEAT_SECONDS per callback says how many more failed. A callback is keyed by
# where its code lives (_callback_name): the ticker builds a fresh lambda every tick, and every
# one of them is the same callback. Tk thread only, like everything safe_after runs.
_REPEAT_SECONDS = 60.0
_clock = time.monotonic
_failing: Dict[str, Dict[str, Any]] = {}


def _callback_name(fn) -> str:
    """"HomePage._run.<locals>.<lambda> (home_page.py:355)": the qualname and where it starts."""
    target = getattr(fn, "__func__", fn)                  # a bound method's own function
    name = getattr(target, "__qualname__", None) or type(target).__qualname__
    code = getattr(target, "__code__", None)
    return f"{name} ({Path(code.co_filename).name}:{code.co_firstlineno})" if code else name


def _report_failure(page: str, fn) -> None:
    """Called from inside the except block that caught `fn`'s exception."""
    name = _callback_name(fn)
    now = _clock()
    seen = _failing.get(name)
    if seen is None:
        _failing[name] = {"said_at": now, "more": 0, "since": time.strftime("%H:%M:%S")}
        logger.exception("%s: a screen update failed -- %s", page, name)
        return
    seen["more"] += 1
    if now - seen["said_at"] >= _REPEAT_SECONDS:
        exc = sys.exc_info()[1]
        logger.error("%s: %d more failures of %s since %s (the last: %s: %s)", page, seen["more"],
                     name, seen["since"], type(exc).__name__, exc)
        seen.update(said_at=now, more=0, since=time.strftime("%H:%M:%S"))


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

    def show_name(self) -> None:
        """Say `page_title` at the left of the row under the bar, in the title face. For a page
        the top bar has no item for (V6App decides, from the bar's own items): with nothing lit
        there, nothing else on screen says where you are. Once -- calling it again changes
        nothing."""
        self._header.show_title(self.page_title)
        self._pack_header()

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
        """One line at the top of the page (under the row with its name or actions, when it has
        one) -- a page's headline in words. '' hides it.

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
        trace) -- once per callback, then counted (_report_failure). A page -- or an app --
        already gone is not an error: nothing is left to update.
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
                _report_failure(type(self).__name__, fn)

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
        # The row under the bar and the line under it: built here, packed at the end of this
        # method if header_actions put something in the row, or by show_name() (see the module
        # docstring).
        self._header = _PageHeader(self, theme=self.theme)
        self._header_rule = ctk.CTkFrame(self, height=1, fg_color=self.theme.DIVIDER,
                                         corner_radius=0)
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
        if self._header.actions_frame.winfo_children():
            self._pack_header()

    def _pack_header(self) -> None:
        """The row under the bar and its rule, above everything else on the page -- a caption
        included, whichever came first. Once."""
        if self._header.winfo_manager() == "pack":
            return
        self._header.pack(side="top", fill="x", before=self.pack_slaves()[0])
        self._header_rule.pack(side="top", fill="x", after=self._header)


class _PageHeader(ctk.CTkFrame):
    """A page's actions, right-aligned in one row -- and its name at the left, for a page the top
    bar does not name (show_title)."""

    def __init__(self, master, theme: ThemeManager):
        super().__init__(master, height=HEADER_HEIGHT, fg_color=theme.SURFACE, corner_radius=0)
        self.theme = theme
        self.title_label: Optional[ctk.CTkLabel] = None
        self.pack_propagate(False)
        # Right-aligned actions frame; subclasses pack widgets here.
        self.actions_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.actions_frame.pack(side="right", fill="y", padx=(theme.SPACE_MD, theme.SPACE_LG))

    def show_title(self, text: str) -> None:
        """The page's name at the left of the row: the title face at the heading size, where the
        page's own content starts (its SPACE_LG margin). Built once; again only sets the words."""
        t = self.theme
        if self.title_label is None:
            self.title_label = ctk.CTkLabel(self, text=text, font=t.title(t.SIZE_HEADING),
                                            text_color=t.TEXT_PRIMARY, anchor="w")
            self.title_label.pack(side="left", fill="y", padx=(t.SPACE_LG, t.SPACE_MD))
        else:
            self.title_label.configure(text=text)
