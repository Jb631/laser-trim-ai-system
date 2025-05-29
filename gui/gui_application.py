"""
Laser Trim AI System - GUI Application
Modern, intuitive interface for daily QA work
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
from tkinterdnd2 import TkinterDnD, DND_FILES
import threading
import queue
import os
import sys
import json
from datetime import datetime
from pathlib import Path
import webbrowser
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import core modules with error handling
try:
    from core.data_processor_adapter import LaserTrimDataProcessor
    from core.config import Config, ConfigManager

    logger.info("Core modules imported successfully")
except ImportError as e:
    logger.error(f"Core import error: {e}")
    # Fallback imports
    try:
        from core.data_processor import DataProcessor as LaserTrimDataProcessor
        from core.config import Config, ConfigManager
    except ImportError as e2:
        logger.error(f"Fallback core import failed: {e2}")
        raise

# Import ML modules with error handling
try:
    from ml_models.ml_analyzer_adapter import MLAnalyzer

    logger.info("ML modules imported successfully")
except ImportError as e:
    logger.warning(f"ML import error: {e} - ML features will be disabled")


    # Create dummy ML analyzer
    class MLAnalyzer:
        def __init__(self, config):
            self.config = config

        def analyze(self, result):
            return {
                'risk_score': 0.0,
                'failure_probability': 0.0,
                'predictions': {}
            }

# Import Excel reporter with error handling
try:
    from excel_reporter.excel_report_adapter import ExcelReportGenerator

    logger.info("Excel reporter imported successfully")
except ImportError as e:
    logger.warning(f"Excel reporter import error: {e}")
    try:
        from excel_reporter.excel_reporter import ExcelReporter as ExcelReportGenerator
    except ImportError:
        logger.error("Failed to import any Excel reporter")


        # Create dummy reporter
        class ExcelReportGenerator:
            def __init__(self, config):
                self.config = config

            def generate_report(self, results, filename):
                logger.error("Excel reporter not available")
                raise NotImplementedError("Excel reporter not available")

# Import database modules with error handling
try:
    from database.database_manager import DatabaseManager
    from database.historical_analyzer import HistoricalAnalyzer
    from database.trend_reporter import TrendReporter

    logger.info("Database modules imported successfully")
except ImportError as e:
    logger.warning(f"Database import error: {e} - Database features will be disabled")
    DatabaseManager = None
    HistoricalAnalyzer = None
    TrendReporter = None


class ModernButton(tk.Button):
    """Modern styled button with hover effects"""

    def __init__(self, parent, **kwargs):
        # Default styling
        default_style = {
            'font': ('Segoe UI', 10),
            'bg': '#2196F3',
            'fg': 'white',
            'bd': 0,
            'relief': tk.FLAT,
            'padx': 20,
            'pady': 10,
            'cursor': 'hand2'
        }

        # Merge with user kwargs
        for key, value in default_style.items():
            if key not in kwargs:
                kwargs[key] = value

        super().__init__(parent, **kwargs)

        # Hover effects
        self.default_bg = kwargs['bg']
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, e):
        self['bg'] = '#1976D2'  # Darker blue on hover

    def _on_leave(self, e):
        self['bg'] = self.default_bg


class ProgressDialog(tk.Toplevel):
    """Modern progress dialog for long operations"""

    def __init__(self, parent, title="Processing"):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x200")
        self.resizable(False, False)

        # Center the dialog
        self.transient(parent)
        self.grab_set()

        # Main frame
        main_frame = tk.Frame(self, bg='white', padx=30, pady=30)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        self.title_label = tk.Label(
            main_frame,
            text=title,
            font=('Segoe UI', 12, 'bold'),
            bg='white'
        )
        self.title_label.pack(pady=(0, 10))

        # Progress bar
        self.progress = ttk.Progressbar(
            main_frame,
            length=300,
            mode='determinate'
        )
        self.progress.pack(pady=10)

        # Status label
        self.status_label = tk.Label(
            main_frame,
            text="Starting...",
            font=('Segoe UI', 9),
            bg='white',
            fg='#666'
        )
        self.status_label.pack(pady=(0, 10))

        # Cancel button
        self.cancel_button = ModernButton(
            main_frame,
            text="Cancel",
            bg='#f44336',
            command=self.cancel
        )
        self.cancel_button.pack()

        self.cancelled = False

    def update_progress(self, value, status_text=""):
        """Update progress bar and status"""
        self.progress['value'] = value
        if status_text:
            self.status_label['text'] = status_text
        self.update()

    def cancel(self):
        """Cancel the operation"""
        self.cancelled = True
        self.destroy()


class LaserTrimAIApp:
    """Main GUI Application for Laser Trim AI System"""

    def __init__(self):
        # Initialize main window with drag and drop support
        self.root = TkinterDnD.Tk()
        self.root.title("Laser Trim AI System - QA Analysis")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Set icon if available
        try:
            icon_path = Path(__file__).parent / "assets" / "icon.ico"
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
        except:
            pass

        # Initialize components with error handling
        try:
            self.config_manager = ConfigManager()
            self.config = Config()  # Use the Config class that has the required attributes
            logger.info("Configuration initialized")
        except Exception as e:
            logger.error(f"Config initialization error: {e}")
            # Create minimal config
            self.config = type('Config', (), {
                'processing': type('processing', (), {
                    'filter_sampling_freq': 100,
                    'filter_cutoff_freq': 80,
                    'gradient_step_size': 3
                })(),
                'OUTPUT_DIR': 'output'
            })()

        try:
            self.processor = LaserTrimDataProcessor(self.config)
            logger.info("Data processor initialized")
        except Exception as e:
            logger.error(f"Processor initialization error: {e}")
            raise

        try:
            self.ml_analyzer = MLAnalyzer(self.config)
            logger.info("ML analyzer initialized")
        except Exception as e:
            logger.warning(f"ML analyzer initialization warning: {e}")
            self.ml_analyzer = None

        try:
            self.report_generator = ExcelReportGenerator(self.config)
            logger.info("Report generator initialized")
        except Exception as e:
            logger.warning(f"Report generator initialization warning: {e}")
            self.report_generator = None

        # State variables
        self.loaded_files = []
        self.current_results = None
        self.processing_queue = queue.Queue()

        # Setup UI
        self._setup_styles()
        self._create_menu()
        self._create_main_ui()
        self._create_status_bar()

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        logger.info("GUI application initialized successfully")

    def _setup_styles(self):
        """Configure ttk styles for modern look"""
        style = ttk.Style()

        # Configure notebook (tabs)
        style.configure('TNotebook', background='#f5f5f5')
        style.configure('TNotebook.Tab', padding=[20, 10])

        # Configure frames
        style.configure('Card.TFrame', background='white', relief=tk.FLAT)

        # Configure labels
        style.configure('Heading.TLabel', font=('Segoe UI', 14, 'bold'))
        style.configure('Subheading.TLabel', font=('Segoe UI', 11))

    def _create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Files...", command=self._load_files, accelerator="Ctrl+O")
        file_menu.add_command(label="Clear All", command=self._clear_all, accelerator="Ctrl+N")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing, accelerator="Ctrl+Q")

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Analysis", command=self._run_analysis, accelerator="F5")
        analysis_menu.add_command(label="Generate Report", command=self._generate_report, accelerator="Ctrl+R")

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self._load_files())
        self.root.bind('<Control-n>', lambda e: self._clear_all())
        self.root.bind('<Control-q>', lambda e: self._on_closing())
        self.root.bind('<F5>', lambda e: self._run_analysis())
        self.root.bind('<Control-r>', lambda e: self._generate_report())

    def _create_main_ui(self):
        """Create the main user interface"""
        # Main container
        main_container = tk.Frame(self.root, bg='#f5f5f5')
        main_container.pack(fill=tk.BOTH, expand=True)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self._create_analysis_tab()
        self._create_results_tab()

    def _create_analysis_tab(self):
        """Create the main analysis tab"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Analysis")

        # Left panel - File management
        left_panel = ttk.Frame(analysis_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # File drop zone
        self._create_drop_zone(left_panel)

        # File list
        self._create_file_list(left_panel)

        # Right panel - Controls and info
        right_panel = ttk.Frame(analysis_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Analysis controls
        self._create_analysis_controls(right_panel)

        # Quick stats
        self._create_quick_stats(right_panel)

    def _create_drop_zone(self, parent):
        """Create drag and drop zone for files"""
        drop_frame = ttk.Frame(parent, style='Card.TFrame')
        drop_frame.pack(fill=tk.X, pady=(0, 10))

        self.drop_zone = tk.Frame(
            drop_frame,
            bg='#e3f2fd',
            height=150,
            relief=tk.RIDGE,
            bd=2
        )
        self.drop_zone.pack(fill=tk.X, padx=20, pady=20)

        # Drop zone content
        drop_icon = tk.Label(
            self.drop_zone,
            text="📁",
            font=('Segoe UI', 48),
            bg='#e3f2fd'
        )
        drop_icon.pack(pady=(20, 10))

        drop_label = tk.Label(
            self.drop_zone,
            text="Drag and drop Excel files here\nor click to browse",
            font=('Segoe UI', 11),
            bg='#e3f2fd',
            fg='#1976D2'
        )
        drop_label.pack()

        # Enable drag and drop
        self.drop_zone.drop_target_register(DND_FILES)
        self.drop_zone.dnd_bind('<<Drop>>', self._handle_drop)

        # Click to browse
        self.drop_zone.bind('<Button-1>', lambda e: self._load_files())
        drop_icon.bind('<Button-1>', lambda e: self._load_files())
        drop_label.bind('<Button-1>', lambda e: self._load_files())

    def _create_file_list(self, parent):
        """Create file list with management controls"""
        list_frame = ttk.Frame(parent, style='Card.TFrame')
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = tk.Frame(list_frame, bg='white')
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))

        tk.Label(
            header_frame,
            text="Loaded Files",
            font=('Segoe UI', 12, 'bold'),
            bg='white'
        ).pack(side=tk.LEFT)

        self.file_count_label = tk.Label(
            header_frame,
            text="(0 files)",
            font=('Segoe UI', 10),
            bg='white',
            fg='#666'
        )
        self.file_count_label.pack(side=tk.LEFT, padx=10)

        # File listbox with scrollbar
        list_container = tk.Frame(list_frame, bg='white')
        list_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.file_listbox = tk.Listbox(
            list_container,
            yscrollcommand=scrollbar.set,
            font=('Segoe UI', 10),
            selectmode=tk.EXTENDED,
            bg='#fafafa',
            bd=1,
            relief=tk.SOLID,
            highlightthickness=0
        )
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)

        # File management buttons
        button_frame = tk.Frame(list_frame, bg='white')
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        ModernButton(
            button_frame,
            text="Remove Selected",
            bg='#f44336',
            command=self._remove_selected_files
        ).pack(side=tk.LEFT, padx=(0, 10))

        ModernButton(
            button_frame,
            text="Clear All",
            bg='#757575',
            command=self._clear_all
        ).pack(side=tk.LEFT)

    def _create_analysis_controls(self, parent):
        """Create analysis control panel"""
        control_frame = ttk.Frame(parent, style='Card.TFrame', width=300)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        control_frame.pack_propagate(False)

        # Header
        tk.Label(
            control_frame,
            text="Analysis Controls",
            font=('Segoe UI', 12, 'bold'),
            bg='white'
        ).pack(padx=20, pady=(20, 10))

        # Analysis settings
        settings_frame = tk.Frame(control_frame, bg='white')
        settings_frame.pack(fill=tk.X, padx=20, pady=10)

        # Checkboxes for analysis options
        self.ml_enabled = tk.BooleanVar(value=self.ml_analyzer is not None)
        tk.Checkbutton(
            settings_frame,
            text="Enable ML Analysis",
            variable=self.ml_enabled,
            font=('Segoe UI', 10),
            bg='white',
            activebackground='white',
            state=tk.NORMAL if self.ml_analyzer else tk.DISABLED
        ).pack(anchor=tk.W, pady=2)

        self.auto_report = tk.BooleanVar(value=True)
        tk.Checkbutton(
            settings_frame,
            text="Auto-generate Report",
            variable=self.auto_report,
            font=('Segoe UI', 10),
            bg='white',
            activebackground='white'
        ).pack(anchor=tk.W, pady=2)

        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, padx=20, pady=10)

        # Action buttons
        button_frame = tk.Frame(control_frame, bg='white')
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.analyze_button = ModernButton(
            button_frame,
            text="🚀 Run Analysis",
            bg='#4CAF50',
            font=('Segoe UI', 12, 'bold'),
            command=self._run_analysis
        )
        self.analyze_button.pack(fill=tk.X, pady=(0, 10))

        self.report_button = ModernButton(
            button_frame,
            text="📊 Generate Report",
            bg='#FF9800',
            command=self._generate_report,
            state=tk.DISABLED
        )
        self.report_button.pack(fill=tk.X)

    def _create_quick_stats(self, parent):
        """Create quick statistics panel"""
        stats_frame = ttk.Frame(parent, style='Card.TFrame', width=300)
        stats_frame.pack(fill=tk.X)
        stats_frame.pack_propagate(False)

        # Header
        tk.Label(
            stats_frame,
            text="Quick Statistics",
            font=('Segoe UI', 12, 'bold'),
            bg='white'
        ).pack(padx=20, pady=(20, 10))

        # Stats container
        self.stats_container = tk.Frame(stats_frame, bg='white')
        self.stats_container.pack(fill=tk.X, padx=20, pady=(0, 20))

        # Initial message
        self.stats_message = tk.Label(
            self.stats_container,
            text="No analysis run yet",
            font=('Segoe UI', 10),
            bg='white',
            fg='#999'
        )
        self.stats_message.pack(pady=20)

    def _create_results_tab(self):
        """Create results visualization tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")

        # Results display area
        self.results_text = tk.Text(
            results_frame,
            font=('Consolas', 10),
            wrap=tk.WORD,
            bg='#f5f5f5'
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Scrollbar for results
        results_scrollbar = ttk.Scrollbar(self.results_text)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=results_scrollbar.set)
        results_scrollbar.config(command=self.results_text.yview)

        # Initially disabled
        self.results_text.config(state=tk.DISABLED)

    def _create_status_bar(self):
        """Create status bar at bottom of window"""
        self.status_bar = tk.Frame(self.root, bg='#e0e0e0', height=30)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_bar.pack_propagate(False)

        self.status_label = tk.Label(
            self.status_bar,
            text="Ready",
            font=('Segoe UI', 9),
            bg='#e0e0e0',
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Version info
        version_label = tk.Label(
            self.status_bar,
            text="v1.0.0",
            font=('Segoe UI', 9),
            bg='#e0e0e0',
            anchor=tk.E
        )
        version_label.pack(side=tk.RIGHT, padx=10)

    def _handle_drop(self, event):
        """Handle file drop event"""
        files = self.root.tk.splitlist(event.data)
        self._add_files(files)

    def _load_files(self):
        """Open file dialog to load files"""
        files = filedialog.askopenfilenames(
            title="Select Excel Files",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )
        if files:
            self._add_files(files)

    def _add_files(self, files):
        """Add files to the loaded files list"""
        for file in files:
            if file not in self.loaded_files and file.lower().endswith(('.xlsx', '.xls')):
                self.loaded_files.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))

        self._update_file_count()
        self._update_ui_state()

    def _remove_selected_files(self):
        """Remove selected files from the list"""
        selected_indices = self.file_listbox.curselection()
        for index in reversed(selected_indices):
            self.loaded_files.pop(index)
            self.file_listbox.delete(index)

        self._update_file_count()
        self._update_ui_state()

    def _clear_all(self):
        """Clear all loaded files"""
        self.loaded_files.clear()
        self.file_listbox.delete(0, tk.END)
        self.current_results = None
        self._update_file_count()
        self._update_ui_state()
        self._clear_results()

    def _update_file_count(self):
        """Update file count label"""
        count = len(self.loaded_files)
        self.file_count_label.config(text=f"({count} file{'s' if count != 1 else ''})")

    def _update_ui_state(self):
        """Update UI elements based on current state"""
        has_files = len(self.loaded_files) > 0
        has_results = self.current_results is not None

        self.analyze_button.config(state=tk.NORMAL if has_files else tk.DISABLED)
        self.report_button.config(state=tk.NORMAL if has_results and self.report_generator else tk.DISABLED)

    def _run_analysis(self):
        """Run analysis on loaded files"""
        if not self.loaded_files:
            messagebox.showwarning("No Files", "Please load files before running analysis.")
            return

        # Create progress dialog
        progress = ProgressDialog(self.root, "Running Analysis")

        # Run analysis in separate thread
        thread = threading.Thread(
            target=self._analysis_worker,
            args=(progress,),
            daemon=True
        )
        thread.start()

    def _analysis_worker(self, progress):
        """Worker thread for running analysis"""
        try:
            total_files = len(self.loaded_files)
            results = []

            for i, file_path in enumerate(self.loaded_files):
                if progress.cancelled:
                    break

                # Update progress
                progress_value = (i / total_files) * 100
                status_text = f"Processing {os.path.basename(file_path)}..."
                self.root.after(0, lambda: progress.update_progress(progress_value, status_text))

                try:
                    # Process file
                    result = self.processor.analyze_file(file_path)

                    # Run ML analysis if enabled
                    if self.ml_enabled.get() and self.ml_analyzer and result:
                        try:
                            ml_result = self.ml_analyzer.analyze(result)
                            result['ml_predictions'] = ml_result
                        except Exception as e:
                            logger.warning(f"ML analysis failed: {e}")

                    results.append(result)

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results.append({
                        'filename': os.path.basename(file_path),
                        'error': str(e),
                        'overall_status': 'ERROR'
                    })

            if not progress.cancelled:
                # Store results
                self.current_results = results

                # Update UI
                self.root.after(0, self._display_results)

                # Auto-generate report if enabled
                if self.auto_report.get() and self.report_generator:
                    self.root.after(0, self._generate_report)

                # Update status
                self.root.after(0, lambda: self._update_status(f"Analysis complete: {len(results)} files processed"))

            # Close progress dialog
            self.root.after(0, progress.destroy)

        except Exception as e:
            logger.error(f"Analysis worker error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
            self.root.after(0, progress.destroy)

    def _display_results(self):
        """Display analysis results"""
        if not self.current_results:
            return

        # Switch to results tab
        self.notebook.select(1)

        # Enable text widget for editing
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        # Display results summary
        self.results_text.insert(tk.END, "ANALYSIS RESULTS\n", 'heading')
        self.results_text.insert(tk.END, "=" * 80 + "\n\n")

        # Summary statistics
        total_files = len(self.current_results)
        passed = sum(1 for r in self.current_results if r and r.get('overall_status') == 'PASS')
        failed = total_files - passed

        self.results_text.insert(tk.END, f"Total Files: {total_files}\n")
        self.results_text.insert(tk.END, f"Passed: {passed}\n", 'pass')
        self.results_text.insert(tk.END, f"Failed: {failed}\n", 'fail')
        self.results_text.insert(tk.END, "\n" + "-" * 80 + "\n\n")

        # Individual file results
        for result in self.current_results:
            if result:
                self._display_file_result(result)

        # Configure text tags
        self.results_text.tag_config('heading', font=('Segoe UI', 14, 'bold'))
        self.results_text.tag_config('pass', foreground='green')
        self.results_text.tag_config('fail', foreground='red')
        self.results_text.tag_config('filename', font=('Segoe UI', 11, 'bold'))

        # Disable editing
        self.results_text.config(state=tk.DISABLED)

        # Update quick stats
        self._update_quick_stats()
        self._update_ui_state()

    def _display_file_result(self, result):
        """Display individual file result"""
        filename = result.get('filename', 'Unknown')
        status = result.get('overall_status', 'ERROR')

        self.results_text.insert(tk.END, f"File: {filename}\n", 'filename')
        self.results_text.insert(tk.END, f"Status: {status}\n", 'pass' if status == 'PASS' else 'fail')

        # Display error if present
        if 'error' in result:
            self.results_text.insert(tk.END, f"Error: {result['error']}\n", 'fail')

        # Key metrics from analysis_results
        if 'analysis_results' in result:
            analysis = result['analysis_results']
            self.results_text.insert(tk.END, f"  Sigma Gradient: {analysis.get('sigma_gradient', 'N/A'):.6f}\n")
            self.results_text.insert(tk.END, f"  Threshold: {analysis.get('sigma_threshold', 'N/A'):.6f}\n")

        # Display track information if available
        if 'tracks' in result:
            for track_id, track_data in result['tracks'].items():
                if 'sigma_results' in track_data:
                    self.results_text.insert(tk.END, f"  Track {track_id}:\n")
                    self.results_text.insert(tk.END, f"    Sigma: {track_data['sigma_results'].sigma_gradient:.6f}\n")
                    self.results_text.insert(tk.END,
                                             f"    Pass: {'Yes' if track_data['sigma_results'].sigma_pass else 'No'}\n")

        # ML predictions if available
        if 'ml_predictions' in result:
            ml = result['ml_predictions']
            self.results_text.insert(tk.END, f"  ML Risk Score: {ml.get('risk_score', 'N/A'):.2f}\n")
            self.results_text.insert(tk.END, f"  Failure Probability: {ml.get('failure_probability', 'N/A'):.1%}\n")

        self.results_text.insert(tk.END, "\n")

    def _update_quick_stats(self):
        """Update quick statistics panel"""
        if not self.current_results:
            return

        # Clear existing stats
        for widget in self.stats_container.winfo_children():
            widget.destroy()

        # Calculate statistics
        total = len(self.current_results)
        passed = sum(1 for r in self.current_results if r and r.get('overall_status') == 'PASS')
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0

        # Display stats
        stats = [
            ("Total Files", str(total)),
            ("Passed", str(passed)),
            ("Failed", str(failed)),
            ("Pass Rate", f"{pass_rate:.1f}%"),
        ]

        for label, value in stats:
            stat_frame = tk.Frame(self.stats_container, bg='white')
            stat_frame.pack(fill=tk.X, pady=5)

            tk.Label(
                stat_frame,
                text=f"{label}:",
                font=('Segoe UI', 10),
                bg='white',
                anchor=tk.W
            ).pack(side=tk.LEFT)

            tk.Label(
                stat_frame,
                text=value,
                font=('Segoe UI', 10, 'bold'),
                bg='white',
                anchor=tk.E
            ).pack(side=tk.RIGHT)

    def _generate_report(self):
        """Generate Excel report"""
        if not self.current_results:
            messagebox.showwarning("No Results", "Please run analysis before generating report.")
            return

        if not self.report_generator:
            messagebox.showerror("Report Error", "Report generator not available.")
            return

        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialfile=f"LaserTrim_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

        if filename:
            try:
                # Generate report
                self.report_generator.generate_report(self.current_results, filename)

                # Update status
                self._update_status(f"Report saved: {os.path.basename(filename)}")

                # Ask if user wants to open the report
                if messagebox.askyesno("Report Generated", "Report generated successfully. Open now?"):
                    os.startfile(filename)

            except Exception as e:
                logger.error(f"Report generation error: {e}")
                messagebox.showerror("Report Error", f"Failed to generate report: {str(e)}")

    def _clear_results(self):
        """Clear results display"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)

        # Clear quick stats
        for widget in self.stats_container.winfo_children():
            if widget != self.stats_message:
                widget.destroy()
        self.stats_message.pack(pady=20)

    def _update_status(self, message):
        """Update status bar message"""
        self.status_label.config(text=message)

    def _show_about(self):
        """Show about dialog"""
        about_text = """Laser Trim AI System
Version 1.0.0

An AI-powered quality assurance system for
potentiometer laser trim analysis.

© 2024 Your Company"""

        messagebox.showinfo("About", about_text)

    def _on_closing(self):
        """Handle window closing"""
        if self.current_results and messagebox.askyesno("Confirm Exit", "You have unsaved results. Exit anyway?"):
            self.root.destroy()
        elif not self.current_results:
            self.root.destroy()

    def run(self):
        """Start the application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    try:
        logger.info("Starting Laser Trim AI System GUI...")
        app = LaserTrimAIApp()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application:\n{str(e)}")
        raise


if __name__ == "__main__":
    main()