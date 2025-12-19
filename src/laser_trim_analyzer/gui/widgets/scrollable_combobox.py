"""
Scrollable ComboBox Widget for Laser Trim Analyzer v3.

CTkComboBox doesn't support scrolling in its dropdown when there are many items.
This widget creates a proper scrollable dropdown using CTkScrollableFrame.
"""

import logging
from typing import List, Callable, Optional
import customtkinter as ctk

logger = logging.getLogger(__name__)


class ScrollableComboBox(ctk.CTkFrame):
    """
    A ComboBox with a scrollable dropdown list.

    Replaces CTkComboBox for cases with many options (e.g., model lists).
    Shows a button with current value, clicking opens a scrollable dropdown.
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
        """Open the dropdown list."""
        if self._is_open or not self._values:
            return

        self._is_open = True
        self._arrow_label.configure(text="\u25B2")  # Up arrow

        # Create toplevel window for dropdown
        self._dropdown_window = ctk.CTkToplevel(self)
        self._dropdown_window.withdraw()  # Hide initially
        self._dropdown_window.overrideredirect(True)  # No window decorations
        self._dropdown_window.attributes("-topmost", True)

        # Calculate position (below the button)
        x = self._button.winfo_rootx()
        y = self._button.winfo_rooty() + self._button.winfo_height()

        # Create scrollable frame for options
        scroll_frame = ctk.CTkScrollableFrame(
            self._dropdown_window,
            width=self._width - 20,
            height=min(self._dropdown_height, len(self._values) * 28),
            fg_color=("gray90", "gray20"),
        )
        scroll_frame.pack(fill="both", expand=True, padx=2, pady=2)

        # Add option buttons
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

        # Position and show window
        self._dropdown_window.geometry(f"{self._width}x{min(self._dropdown_height + 4, len(self._values) * 28 + 4)}+{x}+{y}")
        self._dropdown_window.deiconify()

        # Bind escape to close
        self._dropdown_window.bind("<Escape>", lambda e: self._close_dropdown())

        # Close when clicking outside - bind to toplevel losing focus
        self._dropdown_window.bind("<FocusOut>", self._on_focus_out)

        # Also close if the main window is clicked
        self.winfo_toplevel().bind("<Button-1>", self._on_toplevel_click, add="+")

    def _on_focus_out(self, event):
        """Handle focus out - close dropdown after a short delay."""
        # Use after to allow button clicks to register first
        if self._dropdown_window:
            self._dropdown_window.after(100, self._check_and_close)

    def _on_toplevel_click(self, event):
        """Handle click on main window - close dropdown if open."""
        if not self._is_open:
            return

        # Check if click is on our button
        try:
            bx = self._button.winfo_rootx()
            by = self._button.winfo_rooty()
            bw = self._button.winfo_width()
            bh = self._button.winfo_height()

            if bx <= event.x_root <= bx + bw and by <= event.y_root <= by + bh:
                return  # Click on button, let toggle handle it

            self._close_dropdown()
        except Exception:
            pass

    def _check_and_close(self):
        """Check if we should close the dropdown."""
        if not self._is_open or not self._dropdown_window:
            return

        try:
            # Check if focus is still in the dropdown
            focused = self._dropdown_window.focus_get()
            if focused is None:
                self._close_dropdown()
        except Exception:
            self._close_dropdown()

    def _close_dropdown(self):
        """Close the dropdown list."""
        if not self._is_open:
            return

        self._is_open = False
        self._arrow_label.configure(text="\u25BC")  # Down arrow

        # Unbind toplevel click handler
        try:
            self.winfo_toplevel().unbind("<Button-1>")
        except Exception:
            pass

        if self._dropdown_window:
            self._dropdown_window.destroy()
            self._dropdown_window = None

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

    def set(self, value: str):
        """Set current value."""
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
