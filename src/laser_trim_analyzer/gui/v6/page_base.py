"""Spec 3a — PageBase + _PageHeader. Foundations §2.1.

Subclass contract:
  * class attr page_title (or pass page_title=...)
  * build_content(parent)  — REQUIRED
  * header_actions(parent)  — OPTIONAL; build widgets WITH `parent` and pack them right
  * on_show() / on_hide()   — OPTIONAL
PageBase stores `self.app` (V6App | None) and `self.theme`, and offers safe_after().
"""
from typing import Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

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

    # ---- thread-safe UI update (foundations §2.4) ----
    def safe_after(self, fn, delay: int = 0) -> None:
        try:
            if self.winfo_exists():
                self.after(delay, lambda: fn() if self.winfo_exists() else None)
        except Exception:
            pass

    # ---- internal ----
    def _build_chrome(self) -> None:
        self._header = _PageHeader(self, theme=self.theme, title=self.page_title)
        self._header.pack(side="top", fill="x")
        ctk.CTkFrame(self, height=1, fg_color=self.theme.DIVIDER, corner_radius=0)\
            .pack(side="top", fill="x")
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
