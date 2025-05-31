"""
Modern Professional GUI for Laser Trim Analyzer
A contemporary, user-friendly interface for QA specialists
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from datetime import datetime
import threading
from typing import Optional, Dict, List, Any, Callable
import json

# For animations and modern effects
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

# Import your existing modules
from laser_trim_analyzer.processor.processor_module import DataDrivenLaserProcessor
from laser_trim_analyzer.utils.file_utils import ensure_directory
from laser_trim_analyzer.constants import APP_NAME, DEFAULT_OUTPUT_FOLDER_NAME

# Try to import database and ML modules
try:
    from laser_trim_analyzer.database.db_manager import DatabaseManager

    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False
    DatabaseManager = None

try:
    from ml_threshold_optimizer import MLThresholdOptimizer
    from enhanced_qa_dashboard import EnhancedQADashboard

    HAS_ML = True
except ImportError:
    HAS_ML = False
    MLThresholdOptimizer = None
    EnhancedQADashboard = None


class ModernQAApp:
    """Modern Professional GUI Application for Laser Trim Analyzer"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Laser Trim Analyzer - Professional Edition")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # Define color scheme
        self.colors = {
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

        # Initialize variables
        self.current_page = tk.StringVar(value="home")
        self.input_files = []
        self.processor = None
        self.db_manager = None
        self.ml_optimizer = None
        self.is_processing = False

        # Processing options
        self.processing_mode = tk.StringVar(value="detail")
        self.enable_database = tk.BooleanVar(value=True)
        self.enable_ml = tk.BooleanVar(value=HAS_ML)

        # Status variables
        self.db_status = tk.StringVar(value="Disconnected")
        self.ml_status = tk.StringVar(value="Not Available")
        self.progress_var = tk.DoubleVar(value=0)
        self.status_text = tk.StringVar(value="Ready")

        # Setup GUI
        self._setup_styles()
        self._create_main_layout()
        self._initialize_services()

        # Show home page
        self._show_page("home")

    def _setup_styles(self):
        """Configure ttk styles for modern appearance"""
        self.style = ttk.Style()

        # Configure main theme
        self.style.theme_use('clam')

        # Configure colors
        self.style.configure('TFrame', background=self.colors['bg_primary'])
        self.style.configure('Card.TFrame', background=self.colors['bg_secondary'],
                             relief='flat', borderwidth=1)
        self.style.configure('Dark.TFrame', background=self.colors['bg_dark'])

        # Configure labels
        self.style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'),
                             background=self.colors['bg_primary'],
                             foreground=self.colors['text_primary'])
        self.style.configure('Heading.TLabel', font=('Segoe UI', 14, 'bold'),
                             background=self.colors['bg_secondary'],
                             foreground=self.colors['text_primary'])
        self.style.configure('Card.TLabel', background=self.colors['bg_secondary'])

        # Configure buttons
        self.style.configure('Primary.TButton', font=('Segoe UI', 10, 'bold'))
        self.style.map('Primary.TButton',
                       background=[('active', self.colors['accent_light']),
                                   ('!active', self.colors['accent'])],
                       foreground=[('active', 'white'), ('!active', 'white')])

        self.style.configure('Nav.TButton', font=('Segoe UI', 11), borderwidth=0,
                             background=self.colors['bg_dark'], foreground='white')
        self.style.map('Nav.TButton',
                       background=[('active', self.colors['accent']),
                                   ('!active', self.colors['bg_dark'])])

        # Configure progressbar
        self.style.configure('Modern.Horizontal.TProgressbar',
                             troughcolor=self.colors['border'],
                             background=self.colors['accent'],
                             lightcolor=self.colors['accent_light'],
                             darkcolor=self.colors['accent'])

    def _create_main_layout(self):
        """Create the main application layout"""
        # Main container
        main_container = ttk.Frame(self.root, style='TFrame')
        main_container.pack(fill='both', expand=True)

        # Create header
        self._create_header(main_container)

        # Create body with sidebar and content
        body_frame = ttk.Frame(main_container, style='TFrame')
        body_frame.pack(fill='both', expand=True)

        # Sidebar
        self._create_sidebar(body_frame)

        # Content area
        self.content_frame = ttk.Frame(body_frame, style='TFrame')
        self.content_frame.pack(side='left', fill='both', expand=True)

        # Create pages (stacked frames)
        self.pages = {}
        self._create_home_page()
        self._create_analysis_page()
        self._create_historical_page()
        self._create_ml_tools_page()
        self._create_ai_insights_page()
        self._create_settings_page()

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

        app_title = ttk.Label(title_frame, text="Laser Trim Analyzer",
                              font=('Segoe UI', 16, 'bold'),
                              foreground='white', background=self.colors['bg_dark'])
        app_title.pack()

        subtitle = ttk.Label(title_frame, text="Professional QA Analysis Suite",
                             font=('Segoe UI', 10),
                             foreground='#bbbbbb', background=self.colors['bg_dark'])
        subtitle.pack()

        # Quick actions
        actions_frame = ttk.Frame(header, style='Dark.TFrame')
        actions_frame.pack(side='right', padx=20)

        ttk.Button(actions_frame, text="⚡ Quick Analysis",
                   style='Primary.TButton',
                   command=self._quick_analysis).pack(side='left', padx=5)

        ttk.Button(actions_frame, text="📊 Export Report",
                   command=self._export_report).pack(side='left', padx=5)

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
            btn = ttk.Button(sidebar, text=text, style='Nav.TButton',
                             command=lambda p=page: self._show_page(p))
            btn.pack(fill='x', padx=10, pady=5)

        # User info at bottom
        user_frame = ttk.Frame(sidebar, style='Dark.TFrame')
        user_frame.pack(side='bottom', fill='x', padx=10, pady=20)

        ttk.Label(user_frame, text="QA Specialist",
                  font=('Segoe UI', 10, 'bold'),
                  foreground='white', background=self.colors['bg_dark']).pack()

        ttk.Label(user_frame, text=datetime.now().strftime("%B %d, %Y"),
                  font=('Segoe UI', 9),
                  foreground='#bbbbbb', background=self.colors['bg_dark']).pack()

    def _create_status_bar(self, parent):
        """Create status bar at bottom"""
        status_bar = ttk.Frame(parent, style='TFrame', height=30)
        status_bar.pack(fill='x', side='bottom')
        status_bar.pack_propagate(False)

        # Status text
        ttk.Label(status_bar, textvariable=self.status_text,
                  font=('Segoe UI', 9)).pack(side='left', padx=10)

        # Connection indicators
        indicators_frame = ttk.Frame(status_bar)
        indicators_frame.pack(side='right', padx=10)

        # Database indicator
        self.db_indicator = ttk.Label(indicators_frame, text="● DB:",
                                      font=('Segoe UI', 9))
        self.db_indicator.pack(side='left', padx=5)

        ttk.Label(indicators_frame, textvariable=self.db_status,
                  font=('Segoe UI', 9)).pack(side='left')

        # ML indicator
        if HAS_ML:
            ttk.Label(indicators_frame, text="  ● ML:",
                      font=('Segoe UI', 9)).pack(side='left', padx=(10, 5))

            ttk.Label(indicators_frame, textvariable=self.ml_status,
                      font=('Segoe UI', 9)).pack(side='left')

    def _create_home_page(self):
        """Create home page"""
        page = ttk.Frame(self.content_frame, style='TFrame')
        self.pages['home'] = page

        # Welcome section
        welcome_frame = ttk.Frame(page, style='TFrame')
        welcome_frame.pack(fill='x', padx=20, pady=20)

        ttk.Label(welcome_frame, text="Welcome to Laser Trim Analyzer",
                  style='Title.TLabel').pack(anchor='w')

        ttk.Label(welcome_frame, text="Professional quality analysis for potentiometer manufacturing",
                  font=('Segoe UI', 12), foreground=self.colors['text_secondary']).pack(anchor='w', pady=(5, 20))

        # Quick stats cards
        stats_frame = ttk.Frame(page, style='TFrame')
        stats_frame.pack(fill='x', padx=20, pady=10)

        self._create_stat_card(stats_frame, "Today's Analyses", "0", self.colors['success'])
        self._create_stat_card(stats_frame, "Pass Rate", "0%", self.colors['accent'])
        self._create_stat_card(stats_frame, "Avg Sigma", "0.0000", self.colors['warning'])
        self._create_stat_card(stats_frame, "High Risk", "0", self.colors['danger'])

        # Recent activity
        activity_frame = self._create_card(page, "Recent Activity", height=300)

        # Activity list
        self.activity_list = ttk.Treeview(activity_frame, columns=('Time', 'Action', 'Status'),
                                          show='tree headings', height=10)
        self.activity_list.heading('Time', text='Time')
        self.activity_list.heading('Action', text='Action')
        self.activity_list.heading('Status', text='Status')
        self.activity_list.column('#0', width=0, stretch=False)
        self.activity_list.column('Time', width=150)
        self.activity_list.column('Action', width=400)
        self.activity_list.column('Status', width=100)
        self.activity_list.pack(fill='both', expand=True, padx=10, pady=10)

    def _create_analysis_page(self):
        """Create analysis page with drag-and-drop"""
        page = ttk.Frame(self.content_frame, style='TFrame')
        self.pages['analysis'] = page

        # Title
        title_frame = ttk.Frame(page, style='TFrame')
        title_frame.pack(fill='x', padx=20, pady=20)

        ttk.Label(title_frame, text="File Analysis",
                  style='Title.TLabel').pack(side='left')

        # File selection card
        file_card = self._create_card(page, "Select Files", height=200)

        # Drag and drop area
        self.drop_frame = ttk.Frame(file_card, style='Card.TFrame',
                                    relief='solid', borderwidth=2)
        self.drop_frame.pack(fill='both', expand=True, padx=20, pady=20)

        drop_label = ttk.Label(self.drop_frame,
                               text="Drag and drop files here\nor click to browse",
                               font=('Segoe UI', 12),
                               foreground=self.colors['text_secondary'],
                               background=self.colors['bg_secondary'])
        drop_label.pack(expand=True)

        # Bind drag and drop events (simplified for now)
        self.drop_frame.bind('<Button-1>', lambda e: self._browse_files())

        # Selected files list
        files_frame = self._create_card(page, "Selected Files", height=200)

        self.files_list = ttk.Treeview(files_frame, columns=('Size', 'Model', 'Status'),
                                       show='tree headings', height=6)
        self.files_list.heading('#0', text='File')
        self.files_list.heading('Size', text='Size')
        self.files_list.heading('Model', text='Model')
        self.files_list.heading('Status', text='Status')
        self.files_list.pack(fill='both', expand=True, padx=10, pady=10)

        # Analysis options
        options_frame = self._create_card(page, "Analysis Options")

        options_grid = ttk.Frame(options_frame, style='Card.TFrame')
        options_grid.pack(padx=20, pady=10)

        ttk.Radiobutton(options_grid, text="Detail Mode (with plots)",
                        variable=self.processing_mode, value="detail").grid(row=0, column=0, sticky='w', pady=5)

        ttk.Radiobutton(options_grid, text="Speed Mode (parallel)",
                        variable=self.processing_mode, value="speed").grid(row=1, column=0, sticky='w', pady=5)

        ttk.Checkbutton(options_grid, text="Save to database",
                        variable=self.enable_database).grid(row=0, column=1, sticky='w', padx=20, pady=5)

        ttk.Checkbutton(options_grid, text="Run ML analysis",
                        variable=self.enable_ml).grid(row=1, column=1, sticky='w', padx=20, pady=5)

        # Progress section
        progress_frame = ttk.Frame(page, style='TFrame')
        progress_frame.pack(fill='x', padx=20, pady=10)

        self.progress_bar = ttk.Progressbar(progress_frame, style='Modern.Horizontal.TProgressbar',
                                            variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=5)

        self.progress_label = ttk.Label(progress_frame, text="",
                                        font=('Segoe UI', 9),
                                        foreground=self.colors['text_secondary'])
        self.progress_label.pack()

        # Action buttons
        btn_frame = ttk.Frame(page, style='TFrame')
        btn_frame.pack(pady=20)

        self.analyze_btn = ttk.Button(btn_frame, text="Start Analysis",
                                      style='Primary.TButton',
                                      command=self._start_analysis)
        self.analyze_btn.pack(side='left', padx=5)

        ttk.Button(btn_frame, text="Clear Files",
                   command=self._clear_files).pack(side='left', padx=5)

    def _create_historical_page(self):
        """Create historical data page"""
        page = ttk.Frame(self.content_frame, style='TFrame')
        self.pages['historical'] = page

        # Title
        title_frame = ttk.Frame(page, style='TFrame')
        title_frame.pack(fill='x', padx=20, pady=20)

        ttk.Label(title_frame, text="Historical Data Analysis",
                  style='Title.TLabel').pack(side='left')

        # Query filters
        filters_card = self._create_card(page, "Query Filters")

        filters_grid = ttk.Frame(filters_card, style='Card.TFrame')
        filters_grid.pack(padx=20, pady=10)

        # Model filter
        ttk.Label(filters_grid, text="Model:", style='Card.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        self.model_filter = ttk.Entry(filters_grid, width=20)
        self.model_filter.grid(row=0, column=1, padx=10, pady=5)

        # Date range
        ttk.Label(filters_grid, text="Date Range:", style='Card.TLabel').grid(row=0, column=2, sticky='w', pady=5,
                                                                              padx=(20, 0))
        self.date_range = ttk.Combobox(filters_grid, values=['Last 7 days', 'Last 30 days', 'Last 90 days', 'All time'],
                                       width=15, state='readonly')
        self.date_range.set('Last 30 days')
        self.date_range.grid(row=0, column=3, padx=10, pady=5)

        # Query button
        ttk.Button(filters_grid, text="Run Query", style='Primary.TButton',
                   command=self._query_historical).grid(row=0, column=4, padx=20, pady=5)

        # Results chart
        chart_frame = self._create_card(page, "Trend Analysis", height=300)

        # Create matplotlib figure
        self.hist_figure = Figure(figsize=(10, 4), dpi=100)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_figure, chart_frame)
        self.hist_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        # Results table
        table_frame = self._create_card(page, "Query Results")

        self.hist_table = ttk.Treeview(table_frame, columns=('Date', 'Model', 'Serial', 'Sigma', 'Status'),
                                       show='tree headings', height=10)
        self.hist_table.heading('#0', text='File')
        self.hist_table.heading('Date', text='Date')
        self.hist_table.heading('Model', text='Model')
        self.hist_table.heading('Serial', text='Serial')
        self.hist_table.heading('Sigma', text='Sigma')
        self.hist_table.heading('Status', text='Status')
        self.hist_table.pack(fill='both', expand=True, padx=10, pady=10)

    def _create_ml_tools_page(self):
        """Create ML tools page"""
        page = ttk.Frame(self.content_frame, style='TFrame')
        self.pages['ml_tools'] = page

        ttk.Label(page, text="Machine Learning Tools",
                  style='Title.TLabel').pack(padx=20, pady=20)

        if not HAS_ML:
            ttk.Label(page, text="ML tools not available. Please install required dependencies.",
                      font=('Segoe UI', 12), foreground=self.colors['text_secondary']).pack(padx=20, pady=50)
            return

        # ML cards
        cards_frame = ttk.Frame(page, style='TFrame')
        cards_frame.pack(fill='both', expand=True, padx=20)

        # Threshold optimization
        thresh_card = self._create_card(cards_frame, "Threshold Optimization", width=400, height=250)
        ttk.Label(thresh_card, text="Optimize sigma thresholds using ML",
                  style='Card.TLabel', wraplength=350).pack(padx=20, pady=10)
        ttk.Button(thresh_card, text="Run Optimization",
                   style='Primary.TButton',
                   command=self._run_threshold_optimization).pack(pady=10)

        # Failure prediction
        pred_card = self._create_card(cards_frame, "Failure Prediction", width=400, height=250)
        ttk.Label(pred_card, text="Predict potential failures using historical data",
                  style='Card.TLabel', wraplength=350).pack(padx=20, pady=10)
        ttk.Button(pred_card, text="Train Model",
                   style='Primary.TButton',
                   command=self._train_failure_model).pack(pady=10)

    def _create_ai_insights_page(self):
        """Create AI insights page"""
        page = ttk.Frame(self.content_frame, style='TFrame')
        self.pages['ai_insights'] = page

        ttk.Label(page, text="AI-Powered Insights",
                  style='Title.TLabel').pack(padx=20, pady=20)

        # Insights dashboard
        insights_frame = ttk.Frame(page, style='TFrame')
        insights_frame.pack(fill='both', expand=True, padx=20)

        # Placeholder for future AI features
        ttk.Label(insights_frame,
                  text="AI insights will provide intelligent recommendations\nbased on your analysis patterns",
                  font=('Segoe UI', 14), foreground=self.colors['text_secondary'],
                  justify='center').pack(expand=True)

    def _create_settings_page(self):
        """Create settings page"""
        page = ttk.Frame(self.content_frame, style='TFrame')
        self.pages['settings'] = page

        ttk.Label(page, text="Settings",
                  style='Title.TLabel').pack(padx=20, pady=20)

        # Settings categories
        notebook = ttk.Notebook(page)
        notebook.pack(fill='both', expand=True, padx=20, pady=10)

        # General settings
        general_frame = ttk.Frame(notebook)
        notebook.add(general_frame, text="General")

        # Database settings
        db_frame = ttk.Frame(notebook)
        notebook.add(db_frame, text="Database")

        # Analysis settings
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Analysis")

        # Add settings content (simplified for now)
        ttk.Label(general_frame, text="General settings will be added here",
                  font=('Segoe UI', 10)).pack(padx=20, pady=20)

    def _create_card(self, parent, title, width=None, height=None):
        """Create a card widget"""
        card = ttk.Frame(parent, style='Card.TFrame')
        card.pack(fill='both', expand=True, padx=10, pady=10)

        if width:
            card.configure(width=width)
        if height:
            card.configure(height=height)
            card.pack_propagate(False)

        # Title
        title_label = ttk.Label(card, text=title, style='Heading.TLabel')
        title_label.pack(anchor='w', padx=15, pady=(15, 5))

        # Separator
        ttk.Separator(card, orient='horizontal').pack(fill='x', padx=15, pady=(0, 10))

        return card

    def _create_stat_card(self, parent, title, value, color):
        """Create a statistics card"""
        card = ttk.Frame(parent, style='Card.TFrame', width=200, height=100)
        card.pack(side='left', padx=10, pady=5)
        card.pack_propagate(False)

        # Color indicator
        indicator = tk.Frame(card, bg=color, width=5)
        indicator.pack(side='left', fill='y')

        # Content
        content = ttk.Frame(card, style='Card.TFrame')
        content.pack(fill='both', expand=True, padx=15, pady=15)

        ttk.Label(content, text=title, font=('Segoe UI', 10),
                  foreground=self.colors['text_secondary'],
                  background=self.colors['bg_secondary']).pack(anchor='w')

        ttk.Label(content, text=value, font=('Segoe UI', 20, 'bold'),
                  foreground=self.colors['text_primary'],
                  background=self.colors['bg_secondary']).pack(anchor='w')

    def _show_page(self, page_name):
        """Show the selected page"""
        # Hide all pages
        for page in self.pages.values():
            page.pack_forget()

        # Show selected page
        if page_name in self.pages:
            self.pages[page_name].pack(fill='both', expand=True)
            self.current_page.set(page_name)

    def _initialize_services(self):
        """Initialize database and ML services"""
        # Initialize database
        if HAS_DATABASE and self.enable_database.get():
            try:
                import platform
                if platform.system() == 'Windows':
                    app_data = os.environ.get('APPDATA', '')
                    db_dir = os.path.join(app_data, 'LaserTrimAnalyzer')
                else:
                    home = os.path.expanduser('~')
                    db_dir = os.path.join(home, '.laser_trim_analyzer')

                ensure_directory(db_dir)
                db_path = os.path.join(db_dir, "analysis_history.db")

                self.db_manager = DatabaseManager(db_path)
                self.db_status.set("Connected")
                self.db_indicator.configure(foreground=self.colors['success'])
            except Exception as e:
                self.db_status.set("Error")
                self.db_indicator.configure(foreground=self.colors['danger'])
                print(f"Database initialization error: {e}")

        # Initialize ML
        if HAS_ML and self.enable_ml.get():
            try:
                if self.db_manager:
                    self.ml_optimizer = MLThresholdOptimizer(self.db_manager.db_path)
                    self.ml_status.set("Ready")
                else:
                    self.ml_status.set("No DB")
            except Exception as e:
                self.ml_status.set("Error")
                print(f"ML initialization error: {e}")

    def _browse_files(self):
        """Browse for files to analyze"""
        files = filedialog.askopenfilenames(
            title="Select Excel files",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )

        if files:
            self.input_files = list(files)
            self._update_files_list()
            self._add_activity("Files selected", f"{len(files)} files", "Ready")

    def _update_files_list(self):
        """Update the files list display"""
        # Clear existing
        for item in self.files_list.get_children():
            self.files_list.delete(item)

        # Add files
        for file_path in self.input_files:
            filename = os.path.basename(file_path)
            size = os.path.getsize(file_path) / 1024  # KB

            # Extract model from filename
            model = filename.split('_')[0] if '_' in filename else "Unknown"

            self.files_list.insert('', 'end', text=filename,
                                   values=(f"{size:.1f} KB", model, "Ready"))

    def _clear_files(self):
        """Clear selected files"""
        self.input_files = []
        self._update_files_list()
        self.progress_var.set(0)
        self.progress_label.configure(text="")

    def _start_analysis(self):
        """Start the analysis process"""
        if not self.input_files:
            messagebox.showwarning("No Files", "Please select files to analyze")
            return

        if self.is_processing:
            return

        self.is_processing = True
        self.analyze_btn.configure(state='disabled')

        # Start analysis in separate thread
        thread = threading.Thread(target=self._run_analysis_thread, daemon=True)
        thread.start()

    def _run_analysis_thread(self):
        """Run analysis in separate thread"""
        try:
            # Create output directory
            output_dir = os.path.join(os.path.expanduser("~"), DEFAULT_OUTPUT_FOLDER_NAME)
            ensure_directory(output_dir)

            now = datetime.now()
            run_name = now.strftime("%Y%m%d_%H%M_") + now.strftime("%b-%d-%Y_at_%I-%M-%p")
            run_output_folder = os.path.join(output_dir, run_name)
            ensure_directory(run_output_folder)

            # Update status
            self.root.after(0, lambda: self.status_text.set("Initializing processor..."))
            self.root.after(0, lambda: self._add_activity("Analysis started", f"{len(self.input_files)} files",
                                                          "Processing"))

            # Initialize processor
            self.processor = DataDrivenLaserProcessor(
                run_output_folder,
                generate_unit_plots=(self.processing_mode.get() == "detail"),
                use_parallel=(self.processing_mode.get() == "speed"),
                max_workers=4,
                enable_database=self.enable_database.get()
            )

            # Process files
            total_files = len(self.input_files)
            for i, file_path in enumerate(self.input_files):
                filename = os.path.basename(file_path)

                # Update progress
                progress = (i / total_files) * 100
                self.root.after(0, lambda p=progress, f=filename: self._update_progress(p, f))

                # Process file
                result = self.processor.process_file(file_path)

                # Update file status in list
                status = "Pass" if result and result.get('Overall Status') == 'Pass' else "Fail"
                self.root.after(0, lambda f=filename, s=status: self._update_file_status(f, s))

            # Generate reports
            self.root.after(0, lambda: self.status_text.set("Generating reports..."))
            self.processor._generate_reports()

            # Complete
            self.root.after(0, lambda: self._analysis_complete(run_output_folder))

        except Exception as e:
            self.root.after(0, lambda: self._analysis_error(str(e)))

    def _update_progress(self, progress, filename):
        """Update progress display"""
        self.progress_var.set(progress)
        self.progress_label.configure(text=f"Processing: {filename}")
        self.status_text.set(f"Analyzing files... {progress:.0f}%")

    def _update_file_status(self, filename, status):
        """Update file status in list"""
        for item in self.files_list.get_children():
            if self.files_list.item(item)['text'] == filename:
                values = list(self.files_list.item(item)['values'])
                values[2] = status
                self.files_list.item(item, values=values)
                break

    def _analysis_complete(self, output_folder):
        """Handle analysis completion"""
        self.is_processing = False
        self.analyze_btn.configure(state='normal')
        self.progress_var.set(100)
        self.status_text.set("Analysis complete")
        self.progress_label.configure(text="All files processed successfully")

        self._add_activity("Analysis complete", f"Results saved to {os.path.basename(output_folder)}", "Success")

        # Ask to open results
        if messagebox.askyesno("Analysis Complete", "Analysis complete! Would you like to open the results?"):
            self._open_results(output_folder)

    def _analysis_error(self, error_msg):
        """Handle analysis error"""
        self.is_processing = False
        self.analyze_btn.configure(state='normal')
        self.status_text.set("Error occurred")

        self._add_activity("Analysis failed", error_msg, "Error")
        messagebox.showerror("Analysis Error", f"An error occurred:\n\n{error_msg}")

    def _add_activity(self, time_str, action, status):
        """Add item to activity list"""
        if 'activity_list' in dir(self):
            self.activity_list.insert('', 0, values=(
                datetime.now().strftime("%H:%M:%S") if time_str == "Files selected" or time_str.startswith(
                    "Analysis") else time_str,
                action,
                status
            ))

    def _quick_analysis(self):
        """Quick analysis action"""
        self._show_page('analysis')
        self._browse_files()

    def _export_report(self):
        """Export report action"""
        messagebox.showinfo("Export", "Export functionality will be implemented")

    def _query_historical(self):
        """Query historical data"""
        if not self.db_manager:
            messagebox.showerror("Error", "Database not connected")
            return

        # Get filter values
        model = self.model_filter.get() or None

        # Map date range to days
        date_mapping = {
            'Last 7 days': 7,
            'Last 30 days': 30,
            'Last 90 days': 90,
            'All time': None
        }
        days_back = date_mapping.get(self.date_range.get(), 30)

        try:
            # Query database
            df = self.db_manager.get_historical_data(model=model, days_back=days_back, limit=100)

            if df.empty:
                messagebox.showinfo("No Data", "No data found for the specified criteria")
                return

            # Update table
            for item in self.hist_table.get_children():
                self.hist_table.delete(item)

            for _, row in df.iterrows():
                self.hist_table.insert('', 'end',
                                       text=row.get('filename', ''),
                                       values=(
                                           row.get('timestamp', ''),
                                           row.get('model', ''),
                                           row.get('serial', ''),
                                           f"{row.get('sigma_gradient', 0):.4f}",
                                           'Pass' if row.get('sigma_pass', False) else 'Fail'
                                       ))

            # Update chart
            self._update_historical_chart(df)

        except Exception as e:
            messagebox.showerror("Query Error", f"Error querying database:\n{str(e)}")

    def _update_historical_chart(self, df):
        """Update historical data chart"""
        self.hist_figure.clear()
        ax = self.hist_figure.add_subplot(111)

        # Group by date and calculate pass rate
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_stats = df.groupby('date').agg({
            'sigma_pass': 'mean',
            'sigma_gradient': 'mean'
        })

        # Plot
        ax2 = ax.twinx()

        ax.plot(daily_stats.index, daily_stats['sigma_pass'] * 100, 'b-', label='Pass Rate (%)')
        ax2.plot(daily_stats.index, daily_stats['sigma_gradient'], 'r-', label='Avg Sigma')

        ax.set_xlabel('Date')
        ax.set_ylabel('Pass Rate (%)', color='b')
        ax2.set_ylabel('Average Sigma Gradient', color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')

        ax.set_title('Historical Trends')
        ax.grid(True, alpha=0.3)

        # Format x-axis
        self.hist_figure.autofmt_xdate()

        self.hist_canvas.draw()

    def _run_threshold_optimization(self):
        """Run ML threshold optimization"""
        if not self.ml_optimizer:
            messagebox.showerror("Error", "ML tools not available")
            return

        messagebox.showinfo("ML Tools", "Threshold optimization will be implemented")

    def _train_failure_model(self):
        """Train failure prediction model"""
        if not self.ml_optimizer:
            messagebox.showerror("Error", "ML tools not available")
            return

        messagebox.showinfo("ML Tools", "Failure prediction training will be implemented")

    def _open_results(self, folder_path):
        """Open results folder"""
        try:
            if sys.platform == 'win32':
                os.startfile(folder_path)
            elif sys.platform == 'darwin':
                os.system(f'open "{folder_path}"')
            else:
                os.system(f'xdg-open "{folder_path}"')
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder:\n{str(e)}")


def main():
    """Main entry point"""
    root = tk.Tk()

    # Try to set DPI awareness on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    app = ModernQAApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()