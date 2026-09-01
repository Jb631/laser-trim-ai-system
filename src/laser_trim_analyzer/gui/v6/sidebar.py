"""Spec 3a — Sidebar. Pure view: on_select(name) on click; set_active(name) from V6App."""
from typing import Callable, Dict, List, Optional, Tuple

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

SIDEBAR_WIDTH = 160; ROW_HEIGHT = 40; STRIPE_WIDTH = 3


class Sidebar(ctk.CTkFrame):
    """Three destinations, then everything else.

    2026-08-31 (app-shape spec §1): the app reorganizes around Home ·
    Investigate · Settings. Dashboard, Triage and Process are DE-EMPHASIZED,
    not removed — Dashboard's fate is James's call after he has lived with
    Home, and until then every one of them stays one click away.

    The KEYS are untouched. "model" is still "model"; only its label reads
    "Investigate". FOCUS rows, `set_model_route`, and every deep link in the
    app navigate by key, and renaming one for a label change would break
    click-through silently.
    """
    ITEMS: List[Tuple[str, str]] = [
        ("home", "Home"), ("model", "Investigate"), ("settings", "Settings"),
        ("dashboard", "Dashboard"), ("triage", "Triage"), ("process", "Process"),
    ]
    SEPARATOR_AFTER = "settings"
    MUTED = {"dashboard", "triage", "process"}

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
            row = _SidebarRow(self, name=name, label=label, theme=theme,
                              on_click=self._on_select, muted=name in self.MUTED)
            row.pack(side="top", fill="x")
            self._row_frames[name] = row
            if name == self.SEPARATOR_AFTER:
                self._separator = ctk.CTkFrame(self, height=1, corner_radius=0,
                                               fg_color=theme.DIVIDER)
                self._separator.pack(side="top", fill="x",
                                     padx=theme.SPACE_MD, pady=(theme.SPACE_MD, 0))
                ctk.CTkLabel(self, text="OTHER VIEWS", anchor="w",
                             font=theme.font(theme.SIZE_CAPTION, "bold"),
                             text_color=theme.TEXT_DISABLED)\
                    .pack(side="top", fill="x", padx=theme.SPACE_LG,
                          pady=(theme.SPACE_XS, theme.SPACE_XS))

    def set_active(self, name: str) -> None:
        if name not in self._row_frames or self._active_name == name:
            return
        if self._active_name and self._active_name in self._row_frames:
            self._row_frames[self._active_name].set_active(False)
        self._row_frames[name].set_active(True)
        self._active_name = name


class _SidebarRow(ctk.CTkFrame):
    def __init__(self, master, name, label, theme: ThemeManager,
                 on_click: Callable[[str], None], muted: bool = False):
        super().__init__(master, height=ROW_HEIGHT, fg_color=theme.SIDEBAR_BG)
        self.theme = theme; self.name = name; self._cb = on_click; self._active = False
        # Muted rows read as "still here, not where the work starts". They keep
        # the full row height and hit area — de-emphasis, not a hidden menu.
        self.muted = muted
        self._idle_color = theme.TEXT_DISABLED if muted else theme.TEXT_SECONDARY
        self.pack_propagate(False)
        self._stripe = ctk.CTkFrame(self, width=0, fg_color=theme.SIDEBAR_STRIPE, corner_radius=0)
        self._stripe.pack(side="left", fill="y")
        self._label = ctk.CTkLabel(self, text=label, font=theme.font(theme.SIZE_BODY),
                                   text_color=self._idle_color, anchor="w")
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
            self._label.configure(text_color=self._idle_color, font=t.font(t.SIZE_BODY))
