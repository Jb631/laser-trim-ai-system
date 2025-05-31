"""
Base Page Class for GUI Pages

Provides common functionality for all application pages.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, Any
import logging


class BasePage(ttk.Frame):
    """
    Base class for all application pages.

    Provides common functionality like show/hide, refresh, and access to main window.
    """

    def __init__(self, parent: ttk.Frame, main_window: Any, **kwargs):
        """
        Initialize base page.

        Args:
            parent: Parent frame
            main_window: Reference to main window for accessing services
        """
        super().__init__(parent, **kwargs)
        self.main_window = main_window
        self.logger = logging.getLogger(self.__class__.__name__)

        # Page state
        self.is_visible = False
        self.needs_refresh = True

        # Common style configuration
        self.configure(style='TFrame')

        # Initialize the page
        self._create_page()

    def _create_page(self):
        """Create page content. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_page")

    def show(self):
        """Show the page."""
        self.pack(fill='both', expand=True)
        self.is_visible = True

        # Refresh if needed
        if self.needs_refresh:
            self.refresh()
            self.needs_refresh = False

        self.on_show()

    def hide(self):
        """Hide the page."""
        self.pack_forget()
        self.is_visible = False
        self.on_hide()

    def refresh(self):
        """Refresh page content. Override in subclasses."""
        pass

    def on_show(self):
        """Called when page is shown. Override in subclasses."""
        pass

    def on_hide(self):
        """Called when page is hidden. Override in subclasses."""
        pass

    def mark_needs_refresh(self):
        """Mark that the page needs refresh on next show."""
        self.needs_refresh = True

    @property
    def db_manager(self):
        """Get database manager from main window."""
        return self.main_window.db_manager

    @property
    def config(self):
        """Get configuration from main window."""
        return self.main_window.config

    @property
    def colors(self):
        """Get color scheme from main window."""
        return self.main_window.colors