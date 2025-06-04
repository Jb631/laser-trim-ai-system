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
from laser_trim_analyzer.gui.widgets.file_drop_zone import FileDropZone
from laser_trim_analyzer.gui.widgets.progress_widget import ProgressWidget
from laser_trim_analyzer.gui.dialogs.settings_dialog import SettingsDialog

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
            print("Warning: tkinterdnd2 not available, drag and drop will be disabled")
            
        self.config = config or get_config()

        # Setup window
        self._setup_window()

        # Initialize services
        self._init_services()

        # Create UI
        self._create_ui()

        # Show home page
        self._show_page("home")

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
        self.current_page = tk.StringVar(value="home")
        self.input_files: List[FileInfo] = []
        self.is_processing = False
        
        # Thread management
        self._active_threads = []
        self._shutdown_requested = False
        
        # Performance optimization flags
        self._ui_update_scheduled = False
        self._last_update_time = 0
        self._update_interval = 100  # milliseconds

        # Processing options
        self.processing_mode = tk.StringVar(value="detail")
        self.enable_database = tk.BooleanVar(value=self.config.database.enabled)
        self.enable_ml = tk.BooleanVar(value=self.config.ml.enabled and HAS_ML)
        self.enable_api = tk.BooleanVar(value=self.config.api.enabled and HAS_API)

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

    def _setup_styles(self):
        """Configure ttk styles for modern dark appearance"""
        self.style = ttk.Style()
        
        # Use a dark-friendly theme as base
        self.style.theme_use('clam')
        
        # Update color scheme to be consistently dark
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
            except Exception as e:
                self.db_status.set("Error")
                print(f"Database initialization error: {e}")

        # Initialize ML
        if HAS_ML and self.enable_ml.get():
            try:
                from laser_trim_analyzer.ml.predictors import MLPredictor
                self.ml_predictor = MLPredictor(self.config, logger=logging.getLogger(__name__))
                if self.ml_predictor.initialize():
                    self.ml_status.set("Ready")
                else:
                    self.ml_status.set("Not Ready")
            except Exception as e:
                self.ml_status.set("Error")
                print(f"Failed to initialize ML Predictor: {e}")

        # Initialize API client
        if HAS_API and self.enable_api.get():
            try:
                api_key = self.config.api.api_key if hasattr(self.config.api, 'api_key') else None
                if api_key:
                    self.api_client = AIServiceClient(api_key=api_key)
                    self.api_status.set("Connected")
                else:
                    self.api_status.set("No API Key")
            except Exception as e:
                self.api_status.set("Error")
                print(f"API initialization error: {e}")

        # Initialize analyzer
        try:
            # Don't initialize BaseAnalyzer here - it's abstract
            # Specific analyzers will be created when needed
            self.analyzer = None
        except Exception as e:
            print(f"Analyzer initialization error: {e}")

    def _create_ui(self):
        """Create the main UI layout"""
        # Main container
        main_container = ttk.Frame(self.root, style='TFrame')
        main_container.pack(fill='both', expand=True)

        # Create header
        self._create_header(main_container)

        # Create body with sidebar and content
        body_frame = ttk.Frame(main_container, style='TFrame')
        body_frame.pack(fill='both', expand=True)

        # Create sidebar
        self._create_sidebar(body_frame)

        # Create content area
        self.content_frame = ttk.Frame(body_frame, style='TFrame')
        self.content_frame.pack(side='left', fill='both', expand=True)

        # Create pages
        self.pages = {}
        self._create_pages()

        # Create status bar
        self._create_status_bar(main_container)

    def _create_header(self, parent):
        """Create application header"""
        header = ttk.Frame(parent, style='Dark.TFrame', height=60)
        header.pack(fill='x')
        header.pack_propagate(False)

        # Logo and title
        title_frame = ttk.Frame(header, style='Dark.TFrame')
        title_frame.pack(side='left', padx=20, pady=10)

        app_title = ttk.Label(title_frame,
                              text="Laser Trim Analyzer",
                              font=('Segoe UI', 16, 'bold'),
                              foreground='white',
                              background=self.colors['bg_dark'])
        app_title.pack()

        subtitle = ttk.Label(title_frame,
                             text="Professional QA Analysis Suite",
                             font=('Segoe UI', 10),
                             foreground='#bbbbbb',
                             background=self.colors['bg_dark'])
        subtitle.pack()

        # Quick actions
        self._create_quick_actions(header)

    def _create_quick_actions(self, parent):
        """Create quick action buttons in header"""
        actions_frame = ttk.Frame(parent, style='Dark.TFrame')
        actions_frame.pack(side='right', padx=20)

        ttk.Button(actions_frame,
                   text="⚡ Quick Analysis",
                   style='Primary.TButton',
                   command=self._quick_analysis).pack(side='left', padx=5)

        ttk.Button(actions_frame,
                   text="📊 Export Report",
                   command=self._export_report).pack(side='left', padx=5)

        ttk.Button(actions_frame,
                   text="⚙️ Settings",
                   command=self._show_settings).pack(side='left', padx=5)

    def _create_sidebar(self, parent):
        """Create navigation sidebar"""
        sidebar = ttk.Frame(parent, style='Dark.TFrame', width=250)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)

        # Navigation items
        nav_items = [
            ("🏠 Home", "home"),
            ("📄 Single File", "single_file"),
            ("📊 Analysis", "analysis"),
            ("📦 Batch Processing", "batch_processing"),
            ("📈 Historical Data", "historical"),
            ("📋 Model Summary", "model_summary"),
            ("🔗 Multi-Track", "multi_track"),
            ("🤖 ML Tools", "ml_tools"),
            ("🧠 AI Insights", "ai_insights"),
            ("⚙️ Settings", "settings")
        ]

        for text, page in nav_items:
            btn = ttk.Button(sidebar,
                             text=text,
                             style='Nav.TButton',
                             command=lambda p=page: self._show_page(p))
            btn.pack(fill='x', padx=10, pady=5)

        # User info at bottom
        self._create_user_info(sidebar)

    def _create_user_info(self, parent):
        """Create user info section in sidebar"""
        user_frame = ttk.Frame(parent, style='Dark.TFrame')
        user_frame.pack(side='bottom', fill='x', padx=10, pady=20)

        ttk.Label(user_frame,
                  text="QA Specialist",
                  font=('Segoe UI', 10, 'bold'),
                  foreground='white',
                  background=self.colors['bg_dark']).pack()

        ttk.Label(user_frame,
                  text=datetime.now().strftime("%B %d, %Y"),
                  font=('Segoe UI', 9),
                  foreground='#bbbbbb',
                  background=self.colors['bg_dark']).pack()

    def _create_status_bar(self, parent):
        """Create status bar at bottom"""
        from laser_trim_analyzer.gui.widgets.status_bar import StatusBar

        # Map colors to what StatusBar expects
        status_bar_colors = {
            'bg': self.colors['bg_primary'],
            'border': self.colors['border'],
            'text': self.colors['text_primary'],
            'text_secondary': self.colors['text_secondary']
        }
        
        self.status_bar = StatusBar(parent, status_bar_colors)
        self.status_bar.pack(fill='x', side='bottom')

        # Create default indicators
        self.status_bar.create_default_indicators()

        # Update status indicators
        self.status_bar.update_status('database', self.db_status.get())
        self.status_bar.update_status('ml', self.ml_status.get())
        self.status_bar.update_status('api', self.api_status.get())

    def _create_pages(self):
        """Create all application pages"""
        from laser_trim_analyzer.gui.pages.home_page import HomePage
        from laser_trim_analyzer.gui.pages.analysis_page import AnalysisPage
        from laser_trim_analyzer.gui.pages.batch_processing_page import BatchProcessingPage
        from laser_trim_analyzer.gui.pages.historical_page import HistoricalPage
        from laser_trim_analyzer.gui.pages.model_summary_page import ModelSummaryPage
        from laser_trim_analyzer.gui.pages.multi_track_page import MultiTrackPage
        from laser_trim_analyzer.gui.pages.ml_tools_page import MLToolsPage
        from laser_trim_analyzer.gui.pages.ai_insights_page import AIInsightsPage
        from laser_trim_analyzer.gui.pages.settings_page import SettingsPage
        from laser_trim_analyzer.gui.pages.single_file_page import SingleFilePage

        # Create page instances
        self.pages = {}
        self.pages['home'] = HomePage(self.content_frame, self)
        self.pages['analysis'] = AnalysisPage(self.content_frame, self)
        self.pages['single_file'] = SingleFilePage(self.content_frame, main_window=self)
        self.pages['batch_processing'] = BatchProcessingPage(self.content_frame, main_window=self)
        self.pages['historical'] = HistoricalPage(self.content_frame, self)
        self.pages['model_summary'] = ModelSummaryPage(self.content_frame, self)
        self.pages['multi_track'] = MultiTrackPage(self.content_frame, self)
        self.pages['ml_tools'] = MLToolsPage(self.content_frame, self)
        self.pages['ai_insights'] = AIInsightsPage(self.content_frame, self)
        self.pages['settings'] = SettingsPage(self.content_frame, self)

    def _show_page(self, page_name: str):
        """Show the selected page"""
        # Hide all pages
        for page in self.pages.values():
            page.hide()

        # Show selected page
        if page_name in self.pages:
            self.pages[page_name].show()
            self.current_page.set(page_name)

    def _quick_analysis(self):
        """Quick analysis action"""
        self._show_page('analysis')
        if hasattr(self.pages['analysis'], 'browse_files'):
            self.pages['analysis'].browse_files()

    def _export_report(self):
        """Export report action"""
        current_page_name = self.current_page.get()
        current_page = self.pages.get(current_page_name)
        
        # Check if current page has export capability
        if hasattr(current_page, 'export_results'):
            current_page.export_results()
        elif current_page_name == 'analysis' and hasattr(current_page, 'export_file_results'):
            # For analysis page, export all loaded files
            messagebox.showinfo("Export", 
                               "Use the Export button on individual files or wait for analysis completion to export all results")
        elif current_page_name == 'home':
            # For home page, offer to export today's results
            if self.db_manager:
                self._export_today_results()
            else:
                messagebox.showwarning("No Database", "Database connection required for export")
        else:
            messagebox.showinfo("Export",
                                f"Export is not available from the {current_page_name.replace('_', ' ').title()} page.\n\n"
                                "Export is available from:\n"
                                "• Analysis page (after running analysis)\n"
                                "• Historical Data page\n"
                                "• Home page (today's results)")
    
    def _export_today_results(self):
        """Export today's analysis results"""
        try:
            from datetime import datetime, timedelta
            import pandas as pd
            
            # Get today's results from database
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            results = self.db_manager.get_historical_data(
                start_date=today_start,
                include_tracks=True
            )
            
            if not results:
                messagebox.showinfo("No Data", "No analysis results found for today")
                return
            
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension='.xlsx',
                filetypes=[('Excel files', '*.xlsx'), ('CSV files', '*.csv')],
                initialfile=f'analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            
            if not filename:
                return
            
            # Convert to DataFrame
            data = []
            for result in results:
                # Get primary track data
                primary_track = None
                if result.tracks:
                    primary_track = result.tracks[0]
                
                row = {
                    'Date': result.file_date if result.file_date else result.timestamp,
                    'Model': result.model,
                    'Serial': result.serial,
                    'System': result.system.value,
                    'Status': result.overall_status.value,
                    'Sigma Gradient': primary_track.sigma_gradient if primary_track else None,
                    'Sigma Pass': primary_track.sigma_pass if primary_track else None,
                    'Linearity Pass': primary_track.linearity_pass if primary_track else None,
                    'Risk Category': primary_track.risk_category.value if primary_track and primary_track.risk_category else None,
                    'Processing Time': result.processing_time
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Save to file
            if filename.endswith('.xlsx'):
                df.to_excel(filename, index=False)
            else:
                df.to_csv(filename, index=False)
            
            messagebox.showinfo("Export Complete", f"Data exported to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")

    def _show_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self.root, self.config)
        if dialog.show():
            # Settings were saved, reinitialize services if needed
            self._init_services()
    
    def _schedule_ui_update(self):
        """Schedule a UI update to prevent excessive refreshes"""
        if not self._ui_update_scheduled:
            self._ui_update_scheduled = True
            self.root.after(self._update_interval, self._perform_ui_update)
    
    def _perform_ui_update(self):
        """Perform scheduled UI updates"""
        import time
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Only update if enough time has passed
        if current_time - self._last_update_time >= self._update_interval:
            try:
                # Update status bar if it exists
                if hasattr(self, 'status_bar'):
                    self.status_bar.update_status('database', self.db_status.get())
                    self.status_bar.update_status('ml', self.ml_status.get())
                    self.status_bar.update_status('api', self.api_status.get())
                
                self._last_update_time = current_time
            except Exception as e:
                print(f"UI update error: {e}")
        
        self._ui_update_scheduled = False
    
    def _cleanup_threads(self):
        """Clean up any active threads"""
        self._shutdown_requested = True
        for thread in self._active_threads:
            if thread.is_alive():
                try:
                    thread.join(timeout=1.0)
                except:
                    pass
        self._active_threads.clear()
    
    def _on_closing(self):
        """Handle window close event"""
        try:
            # Clean up threads
            self._cleanup_threads()
            
            # Close database connection
            if self.db_manager:
                try:
                    self.db_manager.close()
                except:
                    pass
            
            # Destroy the window
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            print(f"Error during shutdown: {e}")
            # Force quit if cleanup fails
            import sys
            sys.exit(0)

    def _get_window_size_class(self) -> str:
        """Determine the current window size class."""
        try:
            width = self.root.winfo_width()
            if width <= 1000:
                return 'small'
            elif width <= 1400:
                return 'medium'
            else:
                return 'large'
        except tk.TclError:
            return 'medium'  # Default fallback

    def _on_window_configure(self, event):
        """Handle window resize events for responsive design."""
        # Only handle window resize events for the root window
        if event.widget != self.root:
            return
            
        new_size_class = self._get_window_size_class()
        if new_size_class != self._window_size_class:
            self._window_size_class = new_size_class
            self._handle_window_size_change(new_size_class)

    def _handle_window_size_change(self, size_class: str):
        """Handle window size class changes."""
        # Update UI layout based on new size class
        if hasattr(self, 'pages') and self.pages:
            current_page_name = self.current_page.get()
            if current_page_name in self.pages:
                current_page = self.pages[current_page_name]
                if hasattr(current_page, '_handle_responsive_layout'):
                    try:
                        current_page._handle_responsive_layout(size_class)
                    except Exception as e:
                        print(f"Error updating page layout: {e}")

    def _emergency_stop_all(self, event=None):
        """Emergency stop all processing (Ctrl+Q)."""
        if self.is_processing:
            result = messagebox.askyesno(
                "Emergency Stop",
                "Stop all processing operations?\n\n"
                "This will immediately halt all running processes."
            )
            if result:
                self._stop_all_processing()
                messagebox.showinfo("Stopped", "All processing operations have been stopped.")

    def _emergency_stop_current(self, event=None):
        """Stop current page processing (Escape)."""
        if self.is_processing:
            current_page_name = self.current_page.get()
            if hasattr(self, 'pages') and current_page_name in self.pages:
                current_page = self.pages[current_page_name]
                if hasattr(current_page, 'request_stop_processing'):
                    current_page.request_stop_processing()
                    self.status_text.set("Stop requested...")

    def _stop_all_processing(self):
        """Stop all processing in all pages."""
        self.is_processing = False
        self._shutdown_requested = True
        
        # Stop processing in all pages
        if hasattr(self, 'pages'):
            for page in self.pages.values():
                if hasattr(page, 'request_stop_processing'):
                    page.request_stop_processing()
        
        # Clean up threads
        self._cleanup_threads()
        
        self.status_text.set("All processing stopped")

    def run(self):
        """Start the application"""
        try:
            # Show the window after everything is initialized
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            
            # Start the main loop
            self.root.mainloop()
        except Exception as e:
            print(f"Application error: {e}")
            self._on_closing()


def main():
    """Main entry point for the GUI application"""
    # Load configuration
    try:
        from laser_trim_analyzer.core.config import load_config
        config = load_config()
    except:
        config = None

    # Create and run application
    app = MainWindow(config)
    app.run()


if __name__ == "__main__":
    main()