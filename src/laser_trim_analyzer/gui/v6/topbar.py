"""The top bar -- the app's navigation since the Graphite redesign (2026-10-02). Pure view.

James: "im just not happy with the app, there is so much going on its hard to see what is what".
The sidebar had seven destinations; this bar has three and one button: the app's name, then
Overview · Models · Settings, and on the right ONE blue "Process new files" (spec
docs/superpowers/specs/2026-10-02-graphite-redesign-design.md, "Navigation" -- James: "thats
fine"). Triage is retired; Findings and Dashboard left the bar for two quiet links at the foot of
the Overview; the Process page is reached by the button.

The KEYS never changed. "model" is still "model" (it reads "Models" now), "home" is still "home"
(it reads "Overview"). FOCUS rows, `set_model_route` and every deep link in the app navigate by
key, and renaming one for a label would break click-through silently.

`on_select(key)` on a click; `set_active(key)` from V6App.show_page -- a page with no item here
(Process, Findings, Dashboard) lights none. `on_process()` when the button is pressed: V6App shows
the Process page and starts the remembered-folder run there.
"""
from typing import Callable, Dict, List, Optional, Tuple

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

BAR_HEIGHT = 52
UNDERLINE = 2           # the active destination's thin accent line


class TopBar(ctk.CTkFrame):
    TITLE = "Laser Trim Analyzer"
    ITEMS: List[Tuple[str, str]] = [("home", "Overview"), ("model", "Models"), ("settings", "Settings")]
    # Registered pages with no item on the bar, and how each is still one click away.
    OFF_BAR: Tuple[str, ...] = ("process",      # the blue button
                                "findings",     # "All findings", at the foot of the Overview
                                "dashboard")    # "Company trends", beside it
    PROCESS_LABEL = "Process new files"

    def __init__(self, master, on_select: Callable[[str], None], on_process: Callable[[], None],
                 theme: ThemeManager, **kwargs):
        super().__init__(master, height=BAR_HEIGHT, fg_color=theme.SIDEBAR_BG, corner_radius=0,
                         **kwargs)
        t = self.theme = theme
        self._on_select = on_select
        self._items: Dict[str, _BarItem] = {}
        self._active_name: Optional[str] = None
        self.pack_propagate(False)

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(side="top", fill="both", expand=True, padx=t.SPACE_LG)
        ctk.CTkLabel(row, text=self.TITLE, font=t.font(t.SIZE_HEADING, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w").pack(side="left", padx=(0, t.SPACE_XL))
        for key, label in self.ITEMS:
            item = _BarItem(row, key=key, label=label, theme=t, on_click=self._on_select)
            item.pack(side="left", fill="y", padx=(0, t.SPACE_LG))
            self._items[key] = item
        # The ONE blue button in the app (spec: "accent #3b82f6 (the single primary button)").
        # Dark text: white on this blue measures 3.7:1, below the 4.5 minimum.
        self._process_button = ctk.CTkButton(
            row, text=self.PROCESS_LABEL, command=on_process, fg_color=t.ACCENT,
            hover_color=t.ACCENT_HOVER, text_color=t.TEXT_INVERSE,
            font=t.font(t.SIZE_BODY, "bold"), corner_radius=t.RADIUS_MD, height=32)
        self._process_button.pack(side="right")
        ctk.CTkFrame(self, height=1, fg_color=t.DIVIDER, corner_radius=0).pack(side="bottom", fill="x")

    def set_active(self, name: str) -> None:
        """Light `name`'s item -- or none, for a page that has no item on the bar."""
        for key, item in self._items.items():
            item.set_active(key == name)
        self._active_name = name if name in self._items else None


class _BarItem(ctk.CTkFrame):
    """One destination: its label, and under it the thin line that marks the page you are on."""

    def __init__(self, master, key: str, label: str, theme: ThemeManager,
                 on_click: Callable[[str], None]):
        super().__init__(master, fg_color="transparent")
        t = self.theme = theme
        self.key = key
        self._cb = on_click
        self._active = False
        self._label = ctk.CTkLabel(self, text=label, font=t.font(t.SIZE_BODY),
                                   text_color=t.TEXT_SECONDARY)
        self._label.pack(side="top", fill="both", expand=True)
        self._underline = ctk.CTkFrame(self, height=UNDERLINE, corner_radius=0,
                                       fg_color=t.SIDEBAR_BG)
        self._underline.pack(side="bottom", fill="x")
        for w in (self, self._label):
            w.bind("<Button-1>", lambda _e: self._on_click())
            w.bind("<Enter>", lambda _e: self._hover(True))
            w.bind("<Leave>", lambda _e: self._hover(False))
            try:
                w.configure(cursor="hand2")
            except Exception:            # a CTk internal that refuses a cursor; clicks still work
                pass

    def _on_click(self) -> None:
        self._cb(self.key)

    def _hover(self, on: bool) -> None:
        if not self._active:
            t = self.theme
            self._label.configure(text_color=t.TEXT_PRIMARY if on else t.TEXT_SECONDARY)

    def set_active(self, active: bool) -> None:
        """Bright text and the accent line -- the same font either way, so the items never shift
        sideways when the page changes."""
        self._active = active
        t = self.theme
        self._label.configure(text_color=t.TEXT_PRIMARY if active else t.TEXT_SECONDARY)
        self._underline.configure(fg_color=t.SIDEBAR_STRIPE if active else t.SIDEBAR_BG)
