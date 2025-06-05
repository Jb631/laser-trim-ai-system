"""
CustomTkinter Main Window for Laser Trim Analyzer
Modern dark-themed professional GUI
"""

import customtkinter as ctk
from typing import Dict, Optional, Any
import logging
from pathlib import Path
import sys

# Import pages
from laser_trim_analyzer.gui.pages import (
    HomePage, AnalysisPage, HistoricalPage, ModelSummaryPage,
    MLToolsPage, AIInsightsPage, SettingsPage
)

# Import additional pages
try:
    from laser_trim_analyzer.gui.pages.single_file_page import SingleFilePage
    from laser_trim_analyzer.gui.pages.batch_processing_page import BatchProcessingPage
    from laser_trim_analyzer.gui.pages.multi_track_page import MultiTrackPage
except ImportError as e:
    logging.warning(f"Could not import some pages: {e}")
    SingleFilePage = None
    BatchProcessingPage = None
    MultiTrackPage = None

# Import managers
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.gui.settings_manager import settings_manager
from laser_trim_analyzer.core.config import get_config
from laser_trim_analyzer.core.constants import APP_NAME

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class CTkMainWindow(ctk.CTk):
    """Main application window using CustomTkinter"""
    
    def __init__(self, config=None):
        super().__init__()
        
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Window setup
        self._setup_window()
        
        # Initialize services
        self._init_services()
        
        # Create UI
        self._create_ui()
        
        # Initialize pages
        self._create_pages()
        
        # Show initial page
        self._show_initial_page()
    
    def _setup_window(self):
        """Configure main window"""
        self.title(f"{APP_NAME} - Professional Edition")
        
        # Get screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        # Calculate window size (80% of screen)
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        
        # Set minimum size
        min_width = 1200
        min_height = 700
        window_width = max(window_width, min_width)
        window_height = max(window_height, min_height)
        
        # Center window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.minsize(min_width, min_height)
        
        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # State variables
        self.current_page = None
        self.pages: Dict[str, Any] = {}
        
        # Load saved window state
        try:
            if settings_manager.get("window.maximized", False):
                self.state('zoomed')
            else:
                saved_width = settings_manager.get("window.width", window_width)
                saved_height = settings_manager.get("window.height", window_height)
                saved_x = settings_manager.get("window.x", x)
                saved_y = settings_manager.get("window.y", y)
                self.geometry(f"{saved_width}x{saved_height}+{saved_x}+{saved_y}")
        except Exception as e:
            self.logger.warning(f"Could not restore window state: {e}")
    
    def _init_services(self):
        """Initialize backend services"""
        # Database manager
        self.db_manager = None
        if self.config.database.enabled:
            try:
                db_path = self.config.database.path
                if not str(db_path).startswith(('sqlite://', 'postgresql://', 'mysql://')):
                    db_path = f"sqlite:///{Path(db_path).absolute()}"
                self.db_manager = DatabaseManager(db_path)
                self.logger.info("Database connected")
            except Exception as e:
                self.logger.error(f"Could not connect to database: {e}")
                self.db_manager = None
    
    def _create_ui(self):
        """Create main UI layout"""
        # Create sidebar frame
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(10, weight=1)
        
        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Laser Trim\nAnalyzer",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Create navigation buttons
        self._create_sidebar()
        
        # Create main content frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
    
    def _create_sidebar(self):
        """Create sidebar navigation"""
        nav_buttons = [
            ("home", "🏠 Home", 1),
            ("analysis", "📊 Analysis", 2),
            ("single_file", "📄 Single File", 3),
            ("batch", "📦 Batch Processing", 4),
            ("multi_track", "🎯 Multi-Track", 5),
            ("model_summary", "📈 Model Summary", 6),
            ("historical", "📜 Historical", 7),
            ("ml_tools", "🤖 ML Tools", 8),
            ("ai_insights", "💡 AI Insights", 9),
            ("settings", "⚙️ Settings", 11),
        ]
        
        self.nav_buttons = {}
        
        for page_name, label, row in nav_buttons:
            button = ctk.CTkButton(
                self.sidebar_frame,
                text=label,
                command=lambda p=page_name: self._show_page(p),
                height=40,
                font=ctk.CTkFont(size=14),
                anchor="w"
            )
            button.grid(row=row, column=0, padx=20, pady=5, sticky="ew")
            self.nav_buttons[page_name] = button
        
        # Theme switcher at bottom
        self.appearance_mode_optionmenu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["Dark", "Light", "System"],
            command=self._change_appearance_mode
        )
        self.appearance_mode_optionmenu.grid(row=12, column=0, padx=20, pady=(10, 20), sticky="s")
        self.appearance_mode_optionmenu.set("Dark")
    
    def _create_pages(self):
        """Create all page instances"""
        page_classes = {
            "home": HomePage,
            "analysis": AnalysisPage,
            "single_file": SingleFilePage,
            "batch": BatchProcessingPage,
            "multi_track": MultiTrackPage,
            "model_summary": ModelSummaryPage,
            "historical": HistoricalPage,
            "ml_tools": MLToolsPage,
            "ai_insights": AIInsightsPage,
            "settings": SettingsPage,
        }
        
        for page_name, page_class in page_classes.items():
            if page_class is not None:
                try:
                    page = page_class(self.main_frame, self)
                    page.grid(row=0, column=0, sticky="nsew")
                    page.grid_remove()  # Hide initially
                    self.pages[page_name] = page
                except Exception as e:
                    self.logger.error(f"Could not create {page_name} page: {e}")
                    # Create placeholder frame
                    placeholder = ctk.CTkFrame(self.main_frame)
                    placeholder.grid(row=0, column=0, sticky="nsew")
                    placeholder.grid_remove()
                    
                    label = ctk.CTkLabel(
                        placeholder,
                        text=f"{page_name.title()} Page\n\nUnder Construction",
                        font=ctk.CTkFont(size=24)
                    )
                    label.pack(expand=True)
                    
                    self.pages[page_name] = placeholder
    
    def _show_page(self, page_name: str, **kwargs):
        """Show specified page and hide others
        
        Args:
            page_name: Name of the page to show
            **kwargs: Additional parameters to pass to the page
        """
        # Update button states
        for name, button in self.nav_buttons.items():
            if name == page_name:
                button.configure(fg_color=("gray75", "gray25"))
            else:
                button.configure(fg_color="transparent")
        
        # Hide current page
        if self.current_page and self.current_page in self.pages:
            self.pages[self.current_page].grid_remove()
            
            # Call on_hide if available
            if hasattr(self.pages[self.current_page], 'on_hide'):
                self.pages[self.current_page].on_hide()
        
        # Show new page
        if page_name in self.pages:
            self.pages[page_name].grid()
            self.current_page = page_name
            
            # Pass parameters to the page if it has a set_filter method
            if kwargs and hasattr(self.pages[page_name], 'set_filter'):
                self.pages[page_name].set_filter(**kwargs)
            
            # Call on_show if available
            if hasattr(self.pages[page_name], 'on_show'):
                self.pages[page_name].on_show()
            
            self.logger.debug(f"Showing page: {page_name} with params: {kwargs}")
    
    def _show_initial_page(self):
        """Determine and show initial page"""
        # Check if user has data
        has_data = False
        if self.db_manager:
            try:
                recent = self.db_manager.get_historical_data(limit=1)
                has_data = len(recent) > 0
            except Exception:
                pass
        
        # Show appropriate page
        if has_data:
            self._show_page("home")
        else:
            self._show_page("analysis")
            # Show welcome message after delay
            self.after(1000, self._show_welcome_message)
    
    def _show_welcome_message(self):
        """Show welcome message for new users"""
        if self.current_page == "analysis":
            # Custom CTk message box
            dialog = ctk.CTkToplevel(self)
            dialog.title("Welcome to Laser Trim Analyzer")
            dialog.geometry("500x300")
            dialog.transient(self)
            dialog.grab_set()
            
            # Center the dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() - 500) // 2
            y = (dialog.winfo_screenheight() - 300) // 2
            dialog.geometry(f"500x300+{x}+{y}")
            
            # Message
            message = ctk.CTkLabel(
                dialog,
                text="Welcome to the Laser Trim Analyzer!\n\n"
                     "To get started:\n"
                     "1. Click 'Browse Files' to select Excel files\n"
                     "2. Or drag and drop files onto the interface\n"
                     "3. Process files to build your analysis history\n"
                     "4. Explore ML tools and historical reports\n\n"
                     "Need help? Check the documentation.",
                font=ctk.CTkFont(size=14),
                justify="left"
            )
            message.pack(padx=20, pady=20, expand=True, fill="both")
            
            # OK button
            ok_button = ctk.CTkButton(
                dialog,
                text="OK",
                command=dialog.destroy,
                width=100
            )
            ok_button.pack(pady=(0, 20))
    
    def _change_appearance_mode(self, new_mode: str):
        """Change application appearance mode"""
        ctk.set_appearance_mode(new_mode.lower())
        settings_manager.set("appearance.mode", new_mode.lower())
    
    def run(self):
        """Start the application"""
        self.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        # Save window state
        try:
            settings_manager.set("window.width", self.winfo_width())
            settings_manager.set("window.height", self.winfo_height())
            settings_manager.set("window.x", self.winfo_x())
            settings_manager.set("window.y", self.winfo_y())
            settings_manager.set("window.maximized", self.state() == 'zoomed')
            settings_manager.save()
        except Exception as e:
            self.logger.warning(f"Could not save window state: {e}")
        
        # Destroy window
        self.destroy()