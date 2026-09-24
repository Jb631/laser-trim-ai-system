"""The building blocks every V6 page is made of (spec 2026-09-23, section 2).

Plain functions that build CustomTkinter widgets from theme tokens and return them UNPACKED,
so the caller decides the layout. No colour or size in this file is a literal: all of it
comes from the ThemeManager, so a change to the look is a change to theme.py alone.
"""
from typing import Callable, Dict, Iterable, Optional, Tuple

import customtkinter as ctk

VERDICT_TOKENS: Dict[str, Tuple[str, str]] = {
    "PASS": ("PASS_FG", "PASS_BG"),
    "FAIL": ("FAIL_FG", "FAIL_BG"),
    "UNTRIMMED": ("NEUTRAL_FG", "NEUTRAL_BG"),
    "NOT GRADED": ("NEUTRAL_FG", "NEUTRAL_BG"),
    # Amber, never the FAIL coral: sigma is a drift-watch signal, not a rejection (CLAUDE.md).
    "SIGMA WATCH": ("WATCH_FG", "WATCH_BG"),
}


def count_pill(parent, theme, count: int, tone: str = "act") -> ctk.CTkLabel:
    t = theme
    fg, bg = (t.CHECK, t.CHECK_TINT) if tone == "check" else (t.ACCENT, t.ACCENT_TINT)
    return ctk.CTkLabel(parent, text=f"{count:,}", font=t.mono(t.SIZE_CAPTION), text_color=fg,
                        fg_color=bg, corner_radius=t.RADIUS_SM, padx=8, height=20)


def tag(parent, theme, text: str) -> ctk.CTkLabel:
    t = theme
    return ctk.CTkLabel(parent, text=text, font=t.font(t.SIZE_CAPTION), text_color=t.NEUTRAL_FG,
                        fg_color=t.NEUTRAL_BG, corner_radius=t.RADIUS_SM, padx=7, height=20)


def verdict_badge(parent, theme, verdict: str) -> ctk.CTkLabel:
    """The word AND the colour, never colour alone (colour-blind safe)."""
    t = theme
    word = str(verdict).upper()
    fg_name, bg_name = VERDICT_TOKENS.get(word, ("NEUTRAL_FG", "NEUTRAL_BG"))
    return ctk.CTkLabel(parent, text=word, font=t.mono(t.SIZE_CAPTION, "bold"),
                        text_color=getattr(t, fg_name), fg_color=getattr(t, bg_name),
                        corner_radius=t.RADIUS_SM, padx=8, height=22)


def group_header(parent, theme, title: str, count: int, *, column: str = "",
                 tone: str = "act", meaning: str = "") -> ctk.CTkFrame:
    t = theme
    wrap = ctk.CTkFrame(parent, fg_color="transparent")
    top = ctk.CTkFrame(wrap, fg_color="transparent")
    top.pack(fill="x")
    ctk.CTkLabel(top, text=title, font=t.font(t.SIZE_HEADING, "bold"), text_color=t.TEXT_PRIMARY,
                 anchor="w").pack(side="left")
    count_pill(top, t, count, tone).pack(side="left", padx=(t.SPACE_SM, 0))
    if column:
        ctk.CTkLabel(top, text=column, font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY,
                     anchor="e").pack(side="right")
    ctk.CTkFrame(wrap, height=1, fg_color=t.DIVIDER, corner_radius=0).pack(fill="x", pady=(t.SPACE_XS, 0))
    if meaning:
        ctk.CTkLabel(wrap, text=meaning, font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY,
                     anchor="w", justify="left", wraplength=1000).pack(fill="x", pady=(t.SPACE_XS, 0))
    return wrap


def row(parent, theme, model: str, statement: str, value_text: str, *, tags: Iterable[str] = (),
        on_click: Optional[Callable[[], None]] = None, value_color: Optional[str] = None) -> ctk.CTkFrame:
    """One line: model (mono) . statement and tags . readout (mono, right). Click anywhere on it."""
    t = theme
    frame = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=t.RADIUS_SM)
    frame.grid_columnconfigure(1, weight=1)
    ctk.CTkLabel(frame, text=model, font=t.mono(t.SIZE_CAPTION + 1), text_color=t.TEXT_SECONDARY,
                 anchor="w", width=96).grid(row=0, column=0, sticky="w", padx=(t.SPACE_SM, t.SPACE_MD),
                                            pady=t.SPACE_SM)
    mid = ctk.CTkFrame(frame, fg_color="transparent")
    mid.grid(row=0, column=1, sticky="ew")
    ctk.CTkLabel(mid, text=statement, font=t.font(t.SIZE_BODY), text_color=t.TEXT_PRIMARY,
                 anchor="w", justify="left", wraplength=720).pack(side="left")
    for text in tags:
        tag(mid, t, text).pack(side="left", padx=(t.SPACE_SM, 0))
    ctk.CTkLabel(frame, text=value_text, font=t.mono(t.SIZE_READOUT, "bold"),
                 text_color=value_color or t.TEXT_PRIMARY, anchor="e"
                 ).grid(row=0, column=2, sticky="e", padx=(t.SPACE_MD, t.SPACE_SM))
    ctk.CTkFrame(frame, height=1, fg_color=t.DIVIDER, corner_radius=0
                 ).grid(row=1, column=0, columnspan=3, sticky="ew")

    def set_hover(on: bool) -> None:
        frame.configure(fg_color=t.ELEVATED if on else "transparent")

    def click_all(_event=None) -> None:
        if on_click is not None:
            on_click()

    def leave(event) -> None:
        # <Leave> also fires when the pointer moves onto a CHILD label; only drop the hover
        # when the pointer has really left the row, or the row flickers as you cross it.
        under = frame.winfo_containing(event.x_root, event.y_root)
        if under is None or not str(under).startswith(str(frame)):
            set_hover(False)

    def bind_all(w) -> None:
        w.bind("<Button-1>", click_all, add="+")
        w.bind("<Enter>", lambda _e: set_hover(True), add="+")
        w.bind("<Leave>", leave, add="+")
        try:
            w.configure(cursor="hand2")
        except Exception:                 # some CTk internals refuse a cursor; clicks still work
            pass
        for child in w.winfo_children():
            bind_all(child)

    if on_click is not None:
        bind_all(frame)
    frame._on_click_all = click_all       # test hooks: the real bound handlers
    frame._set_hover = set_hover
    return frame


def primary_button(parent, theme, text: str, command) -> ctk.CTkButton:
    """At most ONE per screen. Dark text: white on this teal measures 1.8:1."""
    t = theme
    return ctk.CTkButton(parent, text=text, command=command, fg_color=t.ACCENT, hover_color=t.ACCENT_HOVER,
                         text_color=t.TEXT_INVERSE, font=t.font(t.SIZE_BODY, "bold"),
                         corner_radius=t.RADIUS_MD, height=32)


def link_button(parent, theme, text: str, command) -> ctk.CTkButton:
    """Teal text that acts -- 'Show all 53'."""
    t = theme
    return ctk.CTkButton(parent, text=text, command=command, fg_color="transparent",
                         hover_color=t.CARD, text_color=t.ACCENT, font=t.font(t.SIZE_BODY),
                         anchor="w", width=0, height=28)


def banner(parent, theme, text: str, tone: str = "check") -> ctk.CTkLabel:
    """A notice. 'check' is coral (something failed or needs a look); 'quiet' is plain."""
    t = theme
    fg, bg = (t.CHECK, t.CHECK_TINT) if tone == "check" else (t.TEXT_SECONDARY, t.CARD)
    return ctk.CTkLabel(parent, text=text, font=t.font(t.SIZE_BODY), text_color=fg, fg_color=bg,
                        corner_radius=t.RADIUS_MD, anchor="w", justify="left", wraplength=1000,
                        padx=12, pady=8)
