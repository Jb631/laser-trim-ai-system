"""
Scrollable ComboBox Widget for Laser Trim Analyzer.

CTkComboBox doesn't support scrolling in its dropdown when there are many items.
This widget creates a proper scrollable dropdown using CTkScrollableFrame.
"""

import logging
import time
from typing import List, Callable, Optional
import customtkinter as ctk

logger = logging.getLogger(__name__)


class ScrollableComboBox(ctk.CTkFrame):
    """
    A ComboBox with a scrollable dropdown list.

    Replaces CTkComboBox for cases with many options (e.g., model lists).
    Shows a button with current value, clicking opens a scrollable dropdown.
    Uses grab_set() for outside-click detection to avoid interfering with
    other widgets' focus/click handling (especially CTkEntry fields).
    """

    def __init__(
        self,
        master,
        values: List[str] = None,
        command: Callable[[str], None] = None,
        width: int = 150,
        height: int = 28,
        dropdown_height: int = 200,
        **kwargs
    ):
        super().__init__(master, fg_color="transparent", **kwargs)

        self._values = values or []
        self._command = command
        self._width = width
        self._height = height
        self._dropdown_height = dropdown_height
        self._current_value = self._values[0] if self._values else ""
        self._dropdown_window = None
        self._is_open = False
        self._option_buttons = []
        self._search_entry = None
        self._open_time = 0

        # Main button that shows current value and opens dropdown
        self._button = ctk.CTkButton(
            self,
            text=self._current_value,
            width=width,
            height=height,
            command=self._toggle_dropdown,
            anchor="w",
            fg_color=("gray85", "gray25"),
            hover_color=("gray75", "gray35"),
            text_color=("gray10", "gray90"),
        )
        self._button.pack(fill="x")

        # Dropdown arrow indicator
        self._arrow_label = ctk.CTkLabel(
            self._button,
            text="\u25BC",  # Down arrow
            width=20,
            fg_color="transparent",
        )
        self._arrow_label.place(relx=1.0, rely=0.5, anchor="e", x=-5)

        # Bind click on arrow too
        self._arrow_label.bind("<Button-1>", lambda e: self._toggle_dropdown())

    def _toggle_dropdown(self):
        """Toggle dropdown visibility."""
        if self._is_open:
            self._close_dropdown()
        else:
            self._open_dropdown()

    def _open_dropdown(self):
        """Open the dropdown list with search filtering."""
        if self._is_open or not self._values:
            return

        self._is_open = True
        self._open_time = time.time()
        self._arrow_label.configure(text="\u25B2")  # Up arrow

        # Create toplevel window for dropdown
        self._dropdown_window = ctk.CTkToplevel(self)
        self._dropdown_window.withdraw()  # Hide initially
        self._dropdown_window.overrideredirect(True)  # No window decorations
        self._dropdown_window.attributes("-topmost", True)

        # Link as transient child (helps OS focus management)
        try:
            self._dropdown_window.wm_transient(self.winfo_toplevel())
        except Exception:
            pass

        # Calculate position (below the button)
        x = self._button.winfo_rootx()
        y = self._button.winfo_rooty() + self._button.winfo_height()

        # Container frame for search + scrollable list
        container = ctk.CTkFrame(self._dropdown_window, fg_color=("gray90", "gray20"))
        container.pack(fill="both", expand=True, padx=2, pady=2)

        # Search entry at top of dropdown
        self._search_entry = ctk.CTkEntry(
            container,
            placeholder_text="Type to filter...",
            width=self._width - 14,
            height=28,
        )
        self._search_entry.pack(padx=5, pady=(5, 2), fill="x")
        self._search_entry.bind("<KeyRelease>", lambda e: self._filter_options(self._search_entry.get()))

        # Create scrollable frame for options
        scroll_frame = ctk.CTkScrollableFrame(
            container,
            width=self._width - 20,
            height=min(self._dropdown_height - 36, len(self._values) * 28),
            fg_color="transparent",
        )
        scroll_frame.pack(fill="both", expand=True, padx=2, pady=(0, 2))

        # Add option buttons
        self._option_buttons = []
        for value in self._values:
            is_selected = value == self._current_value
            btn = ctk.CTkButton(
                scroll_frame,
                text=value,
                width=self._width - 30,
                height=26,
                anchor="w",
                fg_color=("gray70", "gray40") if is_selected else "transparent",
                hover_color=("gray75", "gray35"),
                text_color=("gray10", "gray90"),
                command=lambda v=value: self._select_value(v),
            )
            btn.pack(fill="x", padx=2, pady=1)
            self._option_buttons.append((value, btn))

        # Bind click on dropdown to detect outside clicks (via grab redirect)
        self._dropdown_window.bind("<Button-1>", self._on_grab_click)

        # Bind escape to close
        self._dropdown_window.bind("<Escape>", lambda e: self._close_dropdown())

        # Position and show window (extra height for search entry)
        dropdown_h = min(self._dropdown_height + 40, len(self._values) * 28 + 40)
        self._dropdown_window.geometry(f"{self._width}x{dropdown_h}+{x}+{y}")
        self._dropdown_window.deiconify()

        # Focus search entry and set grab (captures outside clicks)
        def _setup_dropdown():
            try:
                if self._dropdown_window and self._is_open:
                    self._dropdown_window.lift()
                    self._search_entry.focus_set()
                    # grab_set redirects all app clicks to this window
                    # so outside clicks can be detected and used to close
                    self._dropdown_window.grab_set()
            except Exception:
                logger.debug("Dropdown grab/focus setup failed")
        self._dropdown_window.after(100, _setup_dropdown)

    def _on_grab_click(self, event):
        """Handle click while dropdown has grab. Closes if click is outside."""
        if not self._is_open or not self._dropdown_window:
            return

        # Don't close if dropdown just opened (prevents race)
        if time.time() - self._open_time < 0.3:
            return

        # Clicks inside the dropdown have normal coordinates;
        # clicks outside (redirected by grab) have out-of-bounds coordinates
        try:
            w = self._dropdown_window.winfo_width()
            h = self._dropdown_window.winfo_height()
            if 0 <= event.x <= w and 0 <= event.y <= h:
                return  # Click inside dropdown, let child widgets handle it
        except Exception:
            pass

        # Click was outside the dropdown — close it
        self._close_dropdown()

    def _filter_options(self, search_text: str):
        """Filter dropdown options based on search text."""
        search_lower = search_text.lower().strip()
        for value, btn in self._option_buttons:
            if not search_lower or search_lower in value.lower():
                btn.pack(fill="x", padx=2, pady=1)
            else:
                btn.pack_forget()

    def _close_dropdown(self):
        """Close the dropdown list and restore main window focus."""
        if not self._is_open:
            return

        self._is_open = False
        self._option_buttons = []
        self._search_entry = None
        self._arrow_label.configure(text="\u25BC")  # Down arrow

        if self._dropdown_window:
            try:
                self._dropdown_window.grab_release()
            except Exception:
                pass
            self._dropdown_window.destroy()
            self._dropdown_window = None

        # Restore main window focus so CTkEntry fields work properly
        try:
            top = self.winfo_toplevel()
            top.focus_force()
        except Exception:
            pass

    def _select_value(self, value: str):
        """Select a value from the dropdown."""
        self._current_value = value
        self._button.configure(text=value)
        self._close_dropdown()

        if self._command:
            self._command(value)

    def get(self) -> str:
        """Get current value."""
        return self._current_value

    def set(self, value: str) -> None:
        """Set current value."""
        if value not in self._values and self._values:
            logger.warning(f"Value '{value}' not in options, setting anyway")
        self._current_value = value
        self._button.configure(text=value)

    def configure(self, **kwargs):
        """Configure widget options."""
        if "values" in kwargs:
            self._values = kwargs.pop("values")
        if "command" in kwargs:
            self._command = kwargs.pop("command")
        if "state" in kwargs:
            state = kwargs.pop("state")
            self._button.configure(state=state)
        super().configure(**kwargs)

    def cget(self, key: str):
        """Get configuration option."""
        if key == "values":
            return self._values
        if key == "command":
            return self._command
        return super().cget(key)

    def destroy(self):
        """Clean up on destroy."""
        self._close_dropdown()
        super().destroy()
