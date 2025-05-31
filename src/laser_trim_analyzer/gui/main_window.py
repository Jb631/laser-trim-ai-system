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
from laser_trim_analyzer.core.config import Config
from laser_trim_analyzer.core.constants import APP_NAME, DEFAULT_OUTPUT_FOLDER
from laser_trim_analyzer.core.models import AnalysisResult, FileInfo
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.analysis.base import BaseAnalyzer

# Import widgets and dialogs
from laser_trim_analyzer.gui.widgets.stat_card import StatCard
from laser_trim_analyzer.gui.widgets.file_drop_zone import FileDropZone
from laser_trim_analyzer.gui.widgets.progress_widget import ProgressWidget
from laser_trim_analyzer.gui.dialogs.settings_dialog import SettingsDialog

# Try to import ML modules
try:
    from laser_trim_analyzer.ml.predictors import FailurePredictor
    from laser_trim_analyzer.ml.models import ThresholdOptimizer

    HAS_ML = True
except ImportError:
    HAS_ML = False

# Try to import API client
try:
    from laser_trim_analyzer.api.client import AIServiceClient

    HAS_API = True
except ImportError:
    HAS_API = False


class MainWindow:
    """Main application window for Laser Trim Analyzer"""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the main window

        Args:
            config: Application configuration object
        """
        self.root = tk.Tk()
        self.config = config or Config()

        # Setup window
        self._setup_window()

        # Initialize services
        self._init_services()

        # Create UI
        self._create_ui()

        # Show home page
        self._show_page("home")

    def _setup_window(self):
        """Configure the main window"""
        self.root.title(f"{APP_NAME} - Professional Edition")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # Try to set DPI awareness on Windows
        if sys.platform == 'win32':
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
            except:
                pass

        # Define color scheme from config or defaults
        self.colors = self.config.ui.colors if hasattr(self.config, 'ui') else {
            'bg_primary': '#f0f2f5',
            'bg_secondary': '#ffffff',
            'bg_dark': '#1a1a2e',
            'accent': '#0f4c75',
            'accent_light': '#3282b8',
            'success': '#4caf50',
            'warning': '#ff9800',
            'danger': '#f44336',
            'text_primary': '#212121',
            'text_secondary': '#757575',
            'border': '#e0e0e0'
        }

        # Configure ttk styles
        self._setup_styles()

        # Initialize state variables
        self.current_page = tk.StringVar(value="home")
        self.input_files: List[FileInfo] = []
        self.is_processing = False

        # Processing options
        self.processing_mode = tk.StringVar(value="detail")
        self.enable_database = tk.BooleanVar(value=True)
        self.enable_ml = tk.BooleanVar(value=HAS_ML)
        self.enable_api = tk.BooleanVar(value=HAS_API)

        # Status variables
        self.db_status = tk.StringVar(value="Disconnected")
        self.ml_status = tk.StringVar(value="Not Available")
        self.api_status = tk.StringVar(value="Not Connected")
        self.status_text = tk.StringVar(value="Ready")

    def _setup_styles(self):
        """Configure ttk styles for modern appearance"""
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Configure frame styles
        self.style.configure('TFrame', background=self.colors['bg_primary'])
        self.style.configure('Card.TFrame',
                             background=self.colors['bg_secondary'],
                             relief='flat', borderwidth=1)
        self.style.configure('Dark.TFrame', background=self.colors['bg_dark'])

        # Configure label styles
        self.style.configure('Title.TLabel',
                             font=('Segoe UI', 24, 'bold'),
                             background=self.colors['bg_primary'],
                             foreground=self.colors['text_primary'])
        self.style.configure('Heading.TLabel',
                             font=('Segoe UI', 14, 'bold'),
                             background=self.colors['bg_secondary'],
                             foreground=self.colors['text_primary'])

        # Configure button styles
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

    def _init_services(self):
        """Initialize backend services"""
        self.db_manager = None
        self.ml_predictor = None
        self.api_client = None
        self.analyzer = None

        # Initialize database
        if self.enable_database.get():
            try:
                db_path = self.config.database.path if hasattr(self.config, 'database') else \
                    Path.home() / '.laser_trim_analyzer' / 'analysis.db'
                self.db_manager = DatabaseManager(str(db_path))
                self.db_status.set("Connected")
            except Exception as e:
                self.db_status.set("Error")
                print(f"Database initialization error: {e}")

        # Initialize ML
        if HAS_ML and self.enable_ml.get():
            try:
                self.ml_predictor = FailurePredictor()
                self.ml_status.set("Ready")
            except Exception as e:
                self.ml_status.set("Error")
                print(f"ML initialization error: {e}")

        # Initialize API client
        if HAS_API and self.enable_api.get():
            try:
                api_key = self.config.api.key if hasattr(self.config, 'api') else None
                if api_key:
                    self.api_client = AIServiceClient(api_key)
                    self.api_status.set("Connected")
                else:
                    self.api_status.set("No API Key")
            except Exception as e:
                self.api_status.set("Error")
                print(f"API initialization error: {e}")

        # Initialize analyzer
        try:
            self.analyzer = BaseAnalyzer(config=self.config)
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
            ("📊 Analysis", "analysis"),
            ("📈 Historical Data", "historical"),
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

        self.status_bar = StatusBar(parent, self.colors)
        self.status_bar.pack(fill='x', side='bottom')

        # Update status indicators
        self.status_bar.update_status('database', self.db_status.get())
        self.status_bar.update_status('ml', self.ml_status.get())
        self.status_bar.update_status('api', self.api_status.get())

    def _create_pages(self):
        """Create all application pages"""
        from laser_trim_analyzer.gui.pages.home_page import HomePage
        from laser_trim_analyzer.gui.pages.analysis_page import AnalysisPage
        from laser_trim_analyzer.gui.pages.historical_page import HistoricalPage
        from laser_trim_analyzer.gui.pages.ml_tools_page import MLToolsPage
        from laser_trim_analyzer.gui.pages.ai_insights_page import AIInsightsPage
        from laser_trim_analyzer.gui.pages.settings_page import SettingsPage

        # Create page instances
        self.pages['home'] = HomePage(self.content_frame, self)
        self.pages['analysis'] = AnalysisPage(self.content_frame, self)
        self.pages['historical'] = HistoricalPage(self.content_frame, self)
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
        if hasattr(self.pages.get(self.current_page.get()), 'export_report'):
            self.pages[self.current_page.get()].export_report()
        else:
            messagebox.showinfo("Export",
                                "Export is available from the Analysis and Historical pages")

    def _show_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self.root, self.config)
        if dialog.show():
            # Settings were saved, reinitialize services if needed
            self._init_services()

    def run(self):
        """Start the application"""
        self.root.mainloop()


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