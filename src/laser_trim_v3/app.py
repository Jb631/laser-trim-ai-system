"""
Laser Trim Analyzer v3 - Main Application

Clean, simple application window with 5 focused pages.
"""

import customtkinter as ctk
from typing import Dict, Optional
import logging
from pathlib import Path

from laser_trim_v3.config import Config

logger = logging.getLogger(__name__)

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class LaserTrimApp(ctk.CTk):
    """
    Main application window for Laser Trim Analyzer v3.

    Features:
    - Sidebar navigation with 5 focused pages
    - Clean, modern dark theme
    - Simplified from v2's complex navigation
    """

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self._current_page: Optional[str] = None
        self._pages: Dict[str, ctk.CTkFrame] = {}
        self._nav_buttons: Dict[str, ctk.CTkButton] = {}

        # Window setup
        self._setup_window()

        # Create UI
        self._create_sidebar()
        self._create_main_content()
        self._create_pages()

        # Show initial page
        self._show_page("dashboard")

        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_window(self):
        """Configure the main window."""
        self.title("Laser Trim Analyzer v3")
        self.geometry(f"{self.config.gui.window_width}x{self.config.gui.window_height}")
        self.minsize(800, 600)

        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def _create_sidebar(self):
        """Create the left sidebar with navigation."""
        # Sidebar frame
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1)  # Spacer row

        # App title
        self.title_label = ctk.CTkLabel(
            self.sidebar,
            text="Laser Trim\nAnalyzer",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Version label
        self.version_label = ctk.CTkLabel(
            self.sidebar,
            text="v3.0.0",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.version_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Navigation buttons
        nav_items = [
            ("dashboard", "Dashboard", 2),
            ("process", "Process Files", 3),
            ("analyze", "Analyze", 4),
            ("trends", "Trends", 5),
        ]

        for page_id, label, row in nav_items:
            btn = ctk.CTkButton(
                self.sidebar,
                text=label,
                command=lambda p=page_id: self._show_page(p),
                font=ctk.CTkFont(size=14),
                height=40,
                anchor="w",
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray70", "gray30")
            )
            btn.grid(row=row, column=0, padx=10, pady=5, sticky="ew")
            self._nav_buttons[page_id] = btn

        # Settings at bottom (row 7, after spacer row 6)
        settings_btn = ctk.CTkButton(
            self.sidebar,
            text="Settings",
            command=lambda: self._show_page("settings"),
            font=ctk.CTkFont(size=14),
            height=40,
            anchor="w",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30")
        )
        settings_btn.grid(row=7, column=0, padx=10, pady=(5, 20), sticky="ew")
        self._nav_buttons["settings"] = settings_btn

    def _create_main_content(self):
        """Create the main content area."""
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

    def _create_pages(self):
        """Create all pages."""
        # Import pages here to avoid circular imports
        from laser_trim_v3.gui.pages.dashboard import DashboardPage
        from laser_trim_v3.gui.pages.process import ProcessPage
        from laser_trim_v3.gui.pages.analyze import AnalyzePage
        from laser_trim_v3.gui.pages.trends import TrendsPage
        from laser_trim_v3.gui.pages.settings import SettingsPage

        # Create page instances
        self._pages["dashboard"] = DashboardPage(self.main_frame, self)
        self._pages["process"] = ProcessPage(self.main_frame, self)
        self._pages["analyze"] = AnalyzePage(self.main_frame, self)
        self._pages["trends"] = TrendsPage(self.main_frame, self)
        self._pages["settings"] = SettingsPage(self.main_frame, self)

        # Place all pages in the same grid cell (only one visible at a time)
        for page in self._pages.values():
            page.grid(row=0, column=0, sticky="nsew")

    def _show_page(self, page_id: str):
        """Show a specific page and update navigation state."""
        if page_id not in self._pages:
            logger.warning(f"Unknown page: {page_id}")
            return

        # Hide all pages
        for page in self._pages.values():
            page.grid_remove()

        # Show selected page
        self._pages[page_id].grid()
        self._current_page = page_id

        # Update navigation button styles
        for btn_id, btn in self._nav_buttons.items():
            if btn_id == page_id:
                btn.configure(fg_color=("gray75", "gray25"))
            else:
                btn.configure(fg_color="transparent")

        # Call page's on_show method if it exists
        page = self._pages[page_id]
        if hasattr(page, "on_show"):
            page.on_show()

        logger.debug(f"Showing page: {page_id}")

    def _on_closing(self):
        """Handle window close event."""
        logger.info("Application closing...")

        # Save config
        try:
            self.config.save()
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")

        # Cleanup pages
        for page in self._pages.values():
            if hasattr(page, "cleanup"):
                try:
                    page.cleanup()
                except Exception as e:
                    logger.warning(f"Page cleanup error: {e}")

        self.destroy()

    def run(self):
        """Start the application main loop."""
        logger.info("Starting application...")
        self.mainloop()
