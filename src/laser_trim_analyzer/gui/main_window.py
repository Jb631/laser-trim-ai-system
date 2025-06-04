"""
Main window module for Laser Trim Analyzer GUI
Place this file at: src/laser_trim_analyzer/gui/main_window.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from datetime import datetime
import threading
from typing import Optional, Dict, List, Any, Callable
from pathlib import Path
from dataclasses import dataclass
import logging

# Import TkinterDnD2 for drag and drop support
try:
    from tkinterdnd2 import TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

# For modern UI elements
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import from your new structure
from laser_trim_analyzer.core.config import get_config, Config
from laser_trim_analyzer.core.constants import APP_NAME, DEFAULT_OUTPUT_FOLDER

from laser_trim_analyzer.core.models import AnalysisResult, FileMetadata
from laser_trim_analyzer.database.manager import DatabaseManager

# Import widgets and dialogs
from laser_trim_analyzer.gui.widgets.stat_card import StatCard
from laser_trim_analyzer.gui.widgets.progress_widget import ProgressWidget

# Try to import ML modules
try:
    from laser_trim_analyzer.ml.models import FailurePredictor, ThresholdOptimizer
    HAS_ML = True
except ImportError:
    HAS_ML = False

# Try to import API client
try:
    from laser_trim_analyzer.api.client import QAAIAnalyzer as AIServiceClient
    HAS_API = True
except ImportError:
    HAS_API = False

# Define FileInfo dataclass
@dataclass
class FileInfo:
    """Information about a file to be processed"""
    path: Path
    name: str
    size: int
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None

class MainWindow:
    """Main application window for Laser Trim Analyzer"""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the main window

        Args:
            config: Application configuration object
        """
        # Use TkinterDnD.Tk() if available, otherwise fallback to regular Tk
        if HAS_DND:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()
            logging.warning("tkinterdnd2 not available, drag and drop will be disabled")
            
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)

        # Setup window
        self._setup_window()

        # Initialize services
        self._init_services()

        # Create UI
        self._create_ui()

        # Show appropriate initial page based on user state
        self._determine_initial_page()

    def _setup_window(self):
        """Configure the main window with responsive design capabilities"""
        self.root.title(f"{APP_NAME} - Professional Edition")
        
        # Get screen dimensions for responsive sizing
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate responsive window size (80% of screen, with min/max limits)
        min_width, min_height = 1000, 700
        max_width, max_height = 1800, 1200
        
        window_width = max(min_width, min(max_width, int(screen_width * 0.8)))
        window_height = max(min_height, min(max_height, int(screen_height * 0.8)))
        
        # Center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(min_width, min_height)
        
        # Make window resizable with proper weight configuration
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Enable responsive window state
        self.root.state('normal')  # Start in normal state, not maximized
        
        # Try to set DPI awareness on Windows
        if sys.platform == 'win32':
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
            except:
                pass

        # Define color scheme from config or defaults
        self.colors = {
            'bg_primary': '#2b2b2b',      # Dark primary background
            'bg_secondary': '#3c3c3c',    # Dark secondary background
            'bg_dark': '#1e1e1e',         # Darker background
            'accent': '#0078d4',          # Modern blue accent
            'accent_light': '#4a9eff',    # Light blue accent
            'success': '#107c10',         # Dark green
            'warning': '#ff8c00',         # Orange
            'danger': '#d13438',          # Red
            'text_primary': '#ffffff',    # White text
            'text_secondary': '#cccccc',  # Light gray text
            'border': '#555555'           # Gray border
        }

        # Configure ttk styles
        self._setup_styles()

        # Initialize state variables
        self.current_page = tk.StringVar(value="")
        self.input_files: List[FileInfo] = []
        self.is_processing = False
        
        # Thread management
        self._active_threads = []
        self._shutdown_requested = False
        
        # Performance optimization flags
        self._ui_update_scheduled = False
        self._last_update_time = 0
        self._update_interval = 100  # milliseconds

        # Processing options - load from config or use defaults
        self.processing_mode = tk.StringVar(value=getattr(self.config.processing, 'default_mode', 'detail'))
        self.enable_database = tk.BooleanVar(value=getattr(self.config.database, 'enabled', False))
        self.enable_ml = tk.BooleanVar(value=getattr(self.config.ml, 'enabled', False) and HAS_ML)
        self.enable_api = tk.BooleanVar(value=getattr(self.config.api, 'enabled', False) and HAS_API)

        # Status variables
        self.db_status = tk.StringVar(value="Disconnected")
        self.ml_status = tk.StringVar(value="Not Available")
        self.api_status = tk.StringVar(value="Not Connected")
        self.status_text = tk.StringVar(value="Ready")
        
        # Responsive design state
        self._window_size_class = self._get_window_size_class()
        
        # Bind to window events
        self.root.bind('<Configure>', self._on_window_configure)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Bind global stop processing shortcut
        self.root.bind('<Control-q>', self._emergency_stop_all)
        self.root.bind('<Escape>', self._emergency_stop_current)

    def _get_window_size_class(self):
        """Get current window size class for responsive design."""
        width = self.root.winfo_width()
        if width < 1200:
            return "compact"
        elif width < 1600:
            return "normal"
        else:
            return "large"

    def _on_window_configure(self, event):
        """Handle window resize events."""
        if event.widget == self.root:
            new_size_class = self._get_window_size_class()
            if new_size_class != self._window_size_class:
                self._window_size_class = new_size_class
                self._schedule_ui_update()

    def _on_closing(self):
        """Handle window closing event."""
        try:
            # Save window state to settings
            from laser_trim_analyzer.gui.settings_manager import settings_manager
            settings_manager.set("window.width", self.root.winfo_width())
            settings_manager.set("window.height", self.root.winfo_height())
            settings_manager.set("window.x", self.root.winfo_x())
            settings_manager.set("window.y", self.root.winfo_y())
            settings_manager.set("window.maximized", self.root.state() == 'zoomed')
        except Exception as e:
            self.logger.warning(f"Could not save window state: {e}")
        
        # Stop all processing
        self._shutdown_requested = True
        
        # Wait for threads to finish (with timeout)
        for thread in self._active_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        self.root.destroy()

    def _emergency_stop_all(self, event=None):
        """Emergency stop all processing."""
        self._shutdown_requested = True
        self.status_text.set("Stopping all operations...")
        self.logger.warning("Emergency stop requested")

    def _emergency_stop_current(self, event=None):
        """Emergency stop current operation."""
        current_page_name = self.current_page.get()
        current_page = self.pages.get(current_page_name)
        
        if hasattr(current_page, 'stop_processing'):
            current_page.stop_processing()
            self.status_text.set("Operation stopped")

    def _setup_styles(self):
        """Configure ttk styles for modern dark appearance"""
        self.style = ttk.Style()
        
        # Use a dark-friendly theme as base
        self.style.theme_use('clam')
        
        # Configure frame styles for dark theme
        self.style.configure('TFrame', background=self.colors['bg_primary'])
        self.style.configure('Card.TFrame',
                             background=self.colors['bg_secondary'],
                             relief='flat', borderwidth=1)
        self.style.configure('Dark.TFrame', background=self.colors['bg_dark'])

        # Configure label styles for dark theme
        self.style.configure('TLabel',
                             background=self.colors['bg_primary'],
                             foreground=self.colors['text_primary'])
        self.style.configure('Title.TLabel',
                             font=('Segoe UI', 24, 'bold'),
                             background=self.colors['bg_primary'],
                             foreground=self.colors['text_primary'])
        self.style.configure('Heading.TLabel',
                             font=('Segoe UI', 14, 'bold'),
                             background=self.colors['bg_secondary'],
                             foreground=self.colors['text_primary'])

        # Configure button styles for dark theme
        self.style.configure('TButton',
                             background=self.colors['bg_secondary'],
                             foreground=self.colors['text_primary'],
                             bordercolor=self.colors['border'],
                             focuscolor=self.colors['accent'])
        self.style.map('TButton',
                       background=[('active', self.colors['accent']),
                                   ('pressed', self.colors['accent_light'])])
        
        self.style.configure('Primary.TButton',
                             font=('Segoe UI', 10, 'bold'))
        self.style.map('Primary.TButton',
                       background=[('active', self.colors['accent_light']),
                                   ('!active', self.colors['accent'])],
                       foreground=[('active', 'white'),
                                   ('!active', 'white')])

        self.style.configure('Nav.TButton',
                             font=('Segoe UI', 11),
                             borderwidth=0,
                             background=self.colors['bg_dark'],
                             foreground='white')

        # Configure entry styles for dark theme
        self.style.configure('TEntry',
                             fieldbackground=self.colors['bg_secondary'],
                             background=self.colors['bg_secondary'],
                             foreground=self.colors['text_primary'],
                             bordercolor=self.colors['border'])
        
        # Configure combobox styles for dark theme  
        self.style.configure('TCombobox',
                             fieldbackground=self.colors['bg_secondary'],
                             background=self.colors['bg_secondary'],
                             foreground=self.colors['text_primary'],
                             bordercolor=self.colors['border'])
        
        # Configure checkbutton styles for dark theme
        self.style.configure('TCheckbutton',
                             background=self.colors['bg_primary'],
                             foreground=self.colors['text_primary'],
                             focuscolor=self.colors['accent'])
        
        # Configure notebook (tab) styles for dark theme
        self.style.configure('TNotebook',
                             background=self.colors['bg_primary'],
                             bordercolor=self.colors['border'])
        self.style.configure('TNotebook.Tab',
                             background=self.colors['bg_secondary'],
                             foreground=self.colors['text_primary'],
                             padding=[20, 10])
        self.style.map('TNotebook.Tab',
                       background=[('selected', self.colors['accent']),
                                   ('active', self.colors['bg_secondary'])])
        
        # Configure treeview styles for dark theme
        self.style.configure('Treeview',
                             background=self.colors['bg_secondary'],
                             foreground=self.colors['text_primary'],
                             fieldbackground=self.colors['bg_secondary'],
                             bordercolor=self.colors['border'])
        self.style.configure('Treeview.Heading',
                             background=self.colors['bg_dark'],
                             foreground=self.colors['text_primary'])
        
        # Configure progressbar styles for dark theme
        self.style.configure('TProgressbar',
                             background=self.colors['accent'],
                             troughcolor=self.colors['bg_secondary'],
                             bordercolor=self.colors['border'])
        
        # Configure scrollbar styles for dark theme
        self.style.configure('TScrollbar',
                             background=self.colors['bg_secondary'],
                             troughcolor=self.colors['bg_primary'],
                             bordercolor=self.colors['border'],
                             arrowcolor=self.colors['text_secondary'])

    def _init_services(self):
        """Initialize backend services"""
        self.db_manager = None
        self.ml_predictor = None
        self.api_client = None
        self.analyzer = None

        # Initialize database
        if self.enable_database.get():
            try:
                if hasattr(self.config, 'database') and hasattr(self.config.database, 'path'):
                    db_path = self.config.database.path
                    # Convert to string and check if it's already a URL
                    db_path_str = str(db_path)
                    if not db_path_str.startswith(('sqlite://', 'postgresql://', 'mysql://')):
                        # Expand user path if needed
                        if db_path_str.startswith('~'):
                            db_path = Path(db_path_str).expanduser()
                        else:
                            db_path = Path(db_path_str)
                        db_path_str = f"sqlite:///{db_path.absolute()}"
                else:
                    # Default path
                    db_path = Path.home() / '.laser_trim_analyzer' / 'analysis.db'
                    db_path_str = f"sqlite:///{db_path.absolute()}"
                
                self.db_manager = DatabaseManager(db_path_str)
                self.db_status.set("Connected")
                self.