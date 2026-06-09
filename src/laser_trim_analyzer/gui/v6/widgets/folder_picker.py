"""Spec 3e — FolderPicker: path display + Browse button. Emits on_change(path)."""
from tkinter import filedialog
from typing import Callable, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class FolderPicker(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_change: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._on_change = on_change
        self._value: Optional[str] = None
        self._label = ctk.CTkLabel(self, text="No folder selected", font=theme.font(theme.SIZE_BODY),
                                   text_color=theme.TEXT_SECONDARY, anchor="w")
        self._label.pack(side="left", fill="x", expand=True, padx=(0, theme.SPACE_SM))
        ctk.CTkButton(self, text="Browse…", width=100, fg_color=theme.CARD, hover_color=theme.ELEVATED,
                      text_color=theme.TEXT_PRIMARY, command=self._browse,
                      corner_radius=theme.RADIUS_SM).pack(side="right")

    def value(self) -> Optional[str]:
        return self._value

    def set_value(self, path: str) -> None:
        self._value = path
        self._label.configure(text=path, text_color=self.theme.TEXT_PRIMARY)
        self._on_change(path)

    def _browse(self) -> None:
        path = filedialog.askdirectory(title="Select folder to process")
        if path:
            self.set_value(path)
