"""Spec 3a — Sidebar. Pure view: on_select(name) on click; set_active(name) from V6App."""
from typing import Callable, Dict, List, Optional, Tuple

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

SIDEBAR_WIDTH = 160; ROW_HEIGHT = 40; STRIPE_WIDTH = 3


class Sidebar(ctk.CTkFrame):
    ITEMS: List[Tuple[str, str]] = [
        ("dashboard", "Dashboard"), ("triage", "Triage"), ("process", "Process"),
        ("model", "Model"), ("settings", "Settings"),
    ]

    def __init__(self, master, on_select: Callable[[str], None], theme: ThemeManager, **kwargs):
        super().__init__(master, width=SIDEBAR_WIDTH, fg_color=theme.SIDEBAR_BG,
                         corner_radius=0, **kwargs)
        self.theme = theme
        self._on_select = on_select
        self._row_frames: Dict[str, _SidebarRow] = {}
        self._active_name: Optional[str] = None
        self.pack_propagate(False); self.grid_propagate(False)

        title = ctk.CTkLabel(self, text="Laser Trim Analyzer", font=theme.font(theme.SIZE_CAPTION, "bold"),
                             text_color=theme.TEXT_SECONDARY, anchor="w")
        title.pack(side="top", fill="x", padx=theme.SPACE_LG, pady=(theme.SPACE_LG, theme.SPACE_MD))

        for name, label in self.ITEMS:
            row = _SidebarRow(self, name=name, label=label, theme=theme, on_click=self._on_select)
            row.pack(side="top", fill="x")
            self._row_frames[name] = row

    def set_active(self, name: str) -> None:
        if name not in self._row_frames or self._active_name == name:
            return
        if self._active_name and self._active_name in self._row_frames:
            self._row_frames[self._active_name].set_active(False)
        self._row_frames[name].set_active(True)
        self._active_name = name


class _SidebarRow(ctk.CTkFrame):
    def __init__(self, master, name, label, theme: ThemeManager, on_click: Callable[[str], None]):
        super().__init__(master, height=ROW_HEIGHT, fg_color=theme.SIDEBAR_BG)
        self.theme = theme; self.name = name; self._cb = on_click; self._active = False
        self.pack_propagate(False)
        self._stripe = ctk.CTkFrame(self, width=0, fg_color=theme.SIDEBAR_STRIPE, corner_radius=0)
        self._stripe.pack(side="left", fill="y")
        self._label = ctk.CTkLabel(self, text=label, font=theme.font(theme.SIZE_BODY),
                                   text_color=theme.TEXT_SECONDARY, anchor="w")
        self._label.pack(side="left", fill="both", expand=True, padx=(theme.SPACE_MD, theme.SPACE_SM))
        for w in (self, self._label):
            w.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self): self._cb(self.name)

    def set_active(self, active: bool) -> None:
        self._active = active
        t = self.theme
        if active:
            self._stripe.configure(width=STRIPE_WIDTH); self.configure(fg_color=t.SIDEBAR_ACTIVE)
            self._label.configure(text_color=t.TEXT_PRIMARY, font=t.font(t.SIZE_BODY, "bold"))
        else:
            self._stripe.configure(width=0); self.configure(fg_color=t.SIDEBAR_BG)
            self._label.configure(text_color=t.TEXT_SECONDARY, font=t.font(t.SIZE_BODY))
