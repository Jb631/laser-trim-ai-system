"""
CustomTkinter Main Window for Laser Trim Analyzer
Modern dark-themed professional GUI
"""

import customtkinter as ctk
from typing import Dict, Optional, Any
import logging
from pathlib import Path
import sys
from tkinter import messagebox

# Import pages with better error handling
try:
    from laser_trim_analyzer.gui.pages import (
        HomePage, AnalysisPage, HistoricalPage, ModelSummaryPage,
        MLToolsPage, AIInsightsPage, SettingsPage,
        SingleFilePage, BatchProcessingPage, MultiTrackPage
    )
except ImportError as e:
    logging.error(f"Error importing pages: {e}", exc_info=True)
    # Try individual imports as fallback
    from laser_trim_analyzer.gui.pages import HomePage, AnalysisPage, HistoricalPage, ModelSummaryPage
    from laser_trim_analyzer.gui.pages import MLToolsPage, AIInsightsPage, SettingsPage
    try:
        from laser_trim_analyzer.gui.pages import SingleFilePage
    except ImportError:
        logging.warning("Could not import SingleFilePage")
        SingleFilePage = None
    try:
        from laser_trim_analyzer.gui.pages import BatchProcessingPage
    except ImportError:
        logging.warning("Could not import BatchProcessingPage")
        BatchProcessingPage = None
    try:
        from laser_trim_analyzer.gui.pages import MultiTrackPage
    except ImportError:
        logging.warning("Could not import MultiTrackPage")
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
        
        # Bind window close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
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
        self.processing_pages: set = set()  # Track pages that are currently processing
        
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
        
        # Add emergency reset button at the bottom
        self.reset_button = ctk.CTkButton(
            self.sidebar_frame,
            text="🔄 Reset if Frozen",
            command=self._emergency_reset,
            width=140,
            height=30,
            font=ctk.CTkFont(size=12),
            fg_color="#8B0000",  # Dark red
            hover_color="#B22222"  # Lighter red on hover
        )
        # Place it at the bottom with some padding
        self.reset_button.grid(row=20, column=0, padx=20, pady=(10, 20), sticky="s")
        
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
                    self.logger.info(f"Successfully created {page_name} page")
                except Exception as e:
                    self.logger.error(f"Could not create {page_name} page: {e}", exc_info=True)
                    # Create placeholder frame with error details
                    placeholder = ctk.CTkFrame(self.main_frame)
                    placeholder.grid(row=0, column=0, sticky="nsew")
                    placeholder.grid_remove()
                    
                    # Error frame
                    error_frame = ctk.CTkFrame(placeholder)
                    error_frame.pack(expand=True, padx=20, pady=20)
                    
                    label = ctk.CTkLabel(
                        error_frame,
                        text=f"{page_name.title()} Page",
                        font=ctk.CTkFont(size=24, weight="bold")
                    )
                    label.pack(pady=(0, 10))
                    
                    error_label = ctk.CTkLabel(
                        error_frame,
                        text="Error Loading Page",
                        font=ctk.CTkFont(size=16),
                        text_color="red"
                    )
                    error_label.pack(pady=5)
                    
                    # More detailed error message
                    error_msg = str(e)
                    if "No module named" in error_msg:
                        error_msg += "\n\nThis may be a missing dependency or import error."
                    elif "has no attribute" in error_msg:
                        error_msg += "\n\nThis may be a configuration or initialization error."
                    
                    detail_label = ctk.CTkLabel(
                        error_frame,
                        text=f"Error: {error_msg}",
                        font=ctk.CTkFont(size=12),
                        wraplength=400
                    )
                    detail_label.pack(pady=5)
                    
                    # Add retry button for some pages
                    if page_name in ['batch', 'ml_tools']:
                        retry_button = ctk.CTkButton(
                            error_frame,
                            text="Retry Loading",
                            command=lambda p=page_name: self._retry_page_load(p),
                            width=120
                        )
                        retry_button.pack(pady=10)
                    
                    self.pages[page_name] = placeholder
    
    def _show_page(self, page_name: str, **kwargs):
        """Show specified page and hide others
        
        Args:
            page_name: Name of the page to show
            **kwargs: Additional parameters to pass to the page
        """
        # Check if any page is currently processing
        if self.processing_pages and page_name != self.current_page:
            # Show warning dialog
            processing_page_names = list(self.processing_pages)
            message = f"Cannot switch pages while processing is in progress.\n\nCurrently processing on: {', '.join(processing_page_names)}"
            
            # Create warning dialog
            dialog = ctk.CTkToplevel(self)
            dialog.title("Processing in Progress")
            dialog.geometry("400x200")
            dialog.transient(self)
            dialog.grab_set()
            
            # Center the dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() - 400) // 2
            y = (dialog.winfo_screenheight() - 200) // 2
            dialog.geometry(f"400x200+{x}+{y}")
            
            # Warning icon and message
            warning_frame = ctk.CTkFrame(dialog)
            warning_frame.pack(expand=True, fill="both", padx=20, pady=20)
            
            warning_label = ctk.CTkLabel(
                warning_frame,
                text="⚠️",
                font=ctk.CTkFont(size=48)
            )
            warning_label.pack(pady=(0, 10))
            
            message_label = ctk.CTkLabel(
                warning_frame,
                text=message,
                font=ctk.CTkFont(size=14),
                wraplength=350
            )
            message_label.pack(pady=10)
            
            # OK button
            ok_button = ctk.CTkButton(
                warning_frame,
                text="OK",
                command=dialog.destroy,
                width=100
            )
            ok_button.pack(pady=10)
            
            self.logger.warning(f"Navigation blocked: processing in progress on {processing_page_names}")
            return  # Don't switch pages
        
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
    
    def _retry_page_load(self, page_name: str):
        """Retry loading a failed page"""
        try:
            # Remove old placeholder
            if page_name in self.pages:
                self.pages[page_name].destroy()
                del self.pages[page_name]
            
            # Try to recreate the page
            page_classes = {
                "batch": BatchProcessingPage,
                "ml_tools": MLToolsPage,
            }
            
            if page_name in page_classes and page_classes[page_name] is not None:
                page = page_classes[page_name](self.main_frame, self)
                page.grid(row=0, column=0, sticky="nsew")
                page.grid_remove()
                self.pages[page_name] = page
                self.logger.info(f"Successfully reloaded {page_name} page")
                
                # Show the newly loaded page
                self._show_page(page_name)
            else:
                self.logger.error(f"Cannot retry loading {page_name} - class not available")
                
        except Exception as e:
            self.logger.error(f"Failed to retry loading {page_name}: {e}", exc_info=True)
            messagebox.showerror("Retry Failed", f"Could not reload {page_name} page:\n{str(e)}")
    
    def register_processing(self, page_name: str):
        """Register that a page is currently processing
        
        Args:
            page_name: Name of the page that started processing
        """
        self.processing_pages.add(page_name)
        self.logger.info(f"Page '{page_name}' started processing. Current processing pages: {self.processing_pages}")
        
        # Update navigation buttons to show processing state
        try:
            for name, button in self.nav_buttons.items():
                if name != self.current_page:
                    button.configure(state="disabled")
        except Exception as e:
            self.logger.error(f"Error disabling navigation buttons: {e}")
    
    def unregister_processing(self, page_name: str):
        """Unregister that a page has finished processing
        
        Args:
            page_name: Name of the page that finished processing
        """
        self.processing_pages.discard(page_name)
        self.logger.info(f"Page '{page_name}' finished processing. Remaining processing pages: {self.processing_pages}")
        
        # Re-enable navigation buttons if no pages are processing
        if not self.processing_pages:
            try:
                for button in self.nav_buttons.values():
                    button.configure(state="normal")
                self.logger.info("All navigation buttons re-enabled")
            except Exception as e:
                self.logger.error(f"Error re-enabling navigation buttons: {e}")
                # Force clear processing state as a fallback
                self.processing_pages.clear()
        else:
            self.logger.debug(f"Navigation buttons remain disabled due to ongoing processing: {self.processing_pages}")
    
    def force_clear_processing_state(self):
        """Emergency method to clear all processing state"""
        self.logger.warning("Force clearing all processing state")
        self.processing_pages.clear()
        try:
            for button in self.nav_buttons.values():
                button.configure(state="normal")
            self.logger.info("Emergency: All navigation buttons re-enabled")
        except Exception as e:
            self.logger.error(f"Error in emergency navigation button enable: {e}")
            
    def get_processing_status(self) -> dict:
        """Get current processing status for debugging"""
        return {
            'processing_pages': list(self.processing_pages),
            'is_processing': self.is_processing(),
            'nav_button_states': {name: button.cget('state') for name, button in self.nav_buttons.items()}
        }
    
    def is_processing(self) -> bool:
        """Check if any page is currently processing
        
        Returns:
            True if any page is processing, False otherwise
        """
        return bool(self.processing_pages)
    
    def get_processing_pages(self) -> list:
        """Get list of pages currently processing
        
        Returns:
            List of page names that are processing
        """
        return list(self.processing_pages)
    
    def _emergency_reset(self):
        """Emergency reset to clear frozen state"""
        self.logger.warning("Emergency reset triggered by user")
        
        # Clear all processing states
        self.processing_pages.clear()
        
        # Re-enable all navigation buttons
        try:
            for button in self.nav_buttons.values():
                button.configure(state="normal")
        except Exception as e:
            self.logger.error(f"Error re-enabling buttons during emergency reset: {e}")
        
        # Try to re-enable controls on current page
        try:
            if self.current_page in self.pages:
                page = self.pages[self.current_page]
                
                # Call force cleanup if available
                if hasattr(page, 'force_cleanup'):
                    page.force_cleanup()
                
                # Try various state reset methods
                if hasattr(page, '_set_controls_state'):
                    try:
                        page._set_controls_state("normal")
                    except:
                        pass
                
                # Try to directly enable specific buttons
                for button_name in ['analyze_button', 'start_button', 'validate_button', 
                                  'browse_button', 'export_button', 'clear_button']:
                    if hasattr(page, button_name):
                        try:
                            button = getattr(page, button_name)
                            button.configure(state="normal")
                        except:
                            pass
                
                # Clear analyzing flag if it exists
                if hasattr(page, 'is_analyzing'):
                    page.is_analyzing = False
                    
        except Exception as e:
            self.logger.error(f"Error resetting page controls: {e}")
        
        # Show confirmation to user
        try:
            from tkinter import messagebox
            messagebox.showinfo("Reset Complete", 
                              "The app state has been reset.\n\n" +
                              "If buttons are still unresponsive, try navigating to " +
                              "another page and back.")
        except:
            pass
        
        self.logger.info("Emergency reset completed")
    
    def on_closing(self):
        """Handle window closing"""
        # Check if any processing is in progress
        if self.processing_pages:
            # Show confirmation dialog
            message = f"Processing is still in progress on: {', '.join(self.processing_pages)}\n\nAre you sure you want to exit?"
            
            dialog = ctk.CTkToplevel(self)
            dialog.title("Confirm Exit")
            dialog.geometry("400x200")
            dialog.transient(self)
            dialog.grab_set()
            
            # Center the dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() - 400) // 2
            y = (dialog.winfo_screenheight() - 200) // 2
            dialog.geometry(f"400x200+{x}+{y}")
            
            # Frame for content
            content_frame = ctk.CTkFrame(dialog)
            content_frame.pack(expand=True, fill="both", padx=20, pady=20)
            
            # Warning message
            warning_label = ctk.CTkLabel(
                content_frame,
                text="⚠️",
                font=ctk.CTkFont(size=48)
            )
            warning_label.pack(pady=(0, 10))
            
            message_label = ctk.CTkLabel(
                content_frame,
                text=message,
                font=ctk.CTkFont(size=14),
                wraplength=350
            )
            message_label.pack(pady=10)
            
            # Button frame
            button_frame = ctk.CTkFrame(content_frame)
            button_frame.pack(pady=10)
            
            # Result variable
            result = {"confirmed": False}
            
            def confirm_exit():
                result["confirmed"] = True
                dialog.destroy()
            
            def cancel_exit():
                dialog.destroy()
            
            # Yes/No buttons
            yes_button = ctk.CTkButton(
                button_frame,
                text="Yes, Exit",
                command=confirm_exit,
                width=100,
                fg_color="red",
                hover_color="darkred"
            )
            yes_button.pack(side="left", padx=5)
            
            no_button = ctk.CTkButton(
                button_frame,
                text="Cancel",
                command=cancel_exit,
                width=100
            )
            no_button.pack(side="left", padx=5)
            
            # Wait for dialog
            dialog.wait_window()
            
            # Check result
            if not result["confirmed"]:
                return  # Don't close
        
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