"""Spec 3a — PageContainer: stacked frames, tkraise + lifecycle hooks, no destroy on switch."""
from typing import Dict, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.theme import ThemeManager


class PageContainer(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=0, **kwargs)
        self.theme = theme
        self._pages: Dict[str, PageBase] = {}
        self._current: Optional[str] = None
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def add_page(self, name: str, page: PageBase) -> None:
        self._pages[name] = page
        page.grid(row=0, column=0, sticky="nsew")

    def get_page(self, name: str) -> Optional[PageBase]:
        return self._pages.get(name)

    def show(self, name: str) -> None:
        if name not in self._pages or self._current == name:
            return
        if self._current and self._current in self._pages:
            self._pages[self._current].on_hide()
        self._pages[name].tkraise()
        self._pages[name].on_show()
        self._current = name

    @property
    def current_page(self) -> Optional[str]:
        return self._current
