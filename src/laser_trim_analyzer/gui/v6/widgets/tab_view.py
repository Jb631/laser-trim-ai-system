"""Spec 3c — ThemedTabView: CTkTabview with V6 theme tokens.

A SCROLLING TAB CAME BACK BLANK (facelift F4, 2026-09-25). A CTkScrollableFrame draws its content
as a window item on a canvas. Hiding a tab unmaps everything in it, the canvas's content included,
and Tk maps that content again only when the canvas next redraws. Showing the tab again at the same
size gives the canvas no reason to redraw on this Mac, so the content stayed unmapped: Findings,
Units and the other scrolling tabs read blank after you clicked away and back, until a resize, a
scroll or a reload. (On Windows, re-showing a window paints it, which redraws the canvas; probably
fine there, unverified.) So each scrolling frame in a tab gets a `<Configure>` whenever its canvas
is mapped again -- CustomTkinter's own handler for that event resets the scroll region, and that
redraws the canvas. See `_redraw_scrolled_content_on_map`.
"""
import tkinter

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class ThemedTabView(ctk.CTkTabview):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        # The selected tab is SEGMENT_SELECTED, not ACCENT: CTk draws every tab's text in the one
        # text_color, and TEXT_PRIMARY on ACCENT measured 1.66:1 (theme.py says why this teal).
        super().__init__(master, fg_color=theme.SURFACE, segmented_button_fg_color=theme.CARD,
                         segmented_button_selected_color=theme.SEGMENT_SELECTED,
                         segmented_button_selected_hover_color=theme.SEGMENT_SELECTED_HOVER,
                         segmented_button_unselected_color=theme.CARD,
                         segmented_button_unselected_hover_color=theme.ELEVATED,
                         text_color=theme.TEXT_PRIMARY, corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme

    def insert(self, index: int, name: str):
        """CTkTabview.insert (add() calls it), plus: whenever this tab is shown -- by a click, by
        set(), by anything -- its scrolling frames are watched (see the module docstring).

        Bound on the tab frame's own <Map>, with tkinter's bind (CTkFrame.bind would bind its
        internal canvas instead), and resolved at that moment rather than here: a tab's content
        is built after add() returns, and may build a scrolling frame later still. The tab frame
        is mapped BEFORE the widgets inside it are, so the watch is in place in time for the
        canvases' own <Map> on the same showing."""
        tab = super().insert(index, name)
        tkinter.Misc.bind(tab, "<Map>", lambda _event, tab=tab: _redraw_scrolled_content_on_map(tab),
                          "+")
        return tab


def _scrolling_frames(widget):
    """Every CTkScrollableFrame under `widget`. tkinter's own winfo_children: CustomTkinter's
    overrides leave out a widget's internal parts, and a scrolling frame's content sits inside
    one (its canvas)."""
    for child in tkinter.Misc.winfo_children(widget):
        if isinstance(child, ctk.CTkScrollableFrame):
            yield child
        yield from _scrolling_frames(child)


def _redraw_scrolled_content_on_map(tab) -> None:
    """Make every scrolling frame in `tab` redraw its canvas each time the canvas is mapped again.

    The canvas is the frame's own Tk parent (`frame.master`). Its <Map> -- not the tab's -- is the
    moment a redraw can work: Tk skips drawing a canvas that is not mapped, and when the tab's
    <Map> fires the canvases inside it are still unmapped (measured). The <Configure> goes to the
    content frame, the event CustomTkinter already answers by resetting the canvas's scroll region,
    which redraws it and maps the content again. Bound once per canvas (a marker on it), so showing
    a tab a hundred times never stacks a hundred handlers."""
    for frame in _scrolling_frames(tab):
        canvas = frame.master
        if getattr(canvas, "_v6_redraws_on_map", False):
            continue
        tkinter.Misc.bind(canvas, "<Map>", lambda _event, frame=frame: _nudge(frame), "+")
        canvas._v6_redraws_on_map = True


def _nudge(frame) -> None:
    try:
        frame.event_generate("<Configure>", width=frame.winfo_width(),
                             height=frame.winfo_height())
    except tkinter.TclError:          # destroyed between the <Map> and this
        pass
