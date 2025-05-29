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
from database import DatabaseManager, HistoricalAnalyzer, TrendReporter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_engine.data_processor import LaserTrimDataProcessor
from ml_models.ml_analyzer import MLAnalyzer
from excel_reporter.report_generator import ExcelReportGenerator
from config import Config, ConfigManager


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

        # Initialize components
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        self.processor = LaserTrimDataProcessor(self.config)
        self.ml_analyzer = MLAnalyzer(self.config)
        self.report_generator = ExcelReportGenerator(self.config)

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

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Settings...", command=self._open_settings)
        tools_menu.add_command(label="Model Training...", command=self._open_model_training)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._open_documentation)
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
        self._create_history_tab()

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
        self.ml_enabled = tk.BooleanVar(value=True)
        tk.Checkbutton(
            settings_frame,
            text="Enable ML Analysis",
            variable=self.ml_enabled,
            font=('Segoe UI', 10),
            bg='white',
            activebackground='white'
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

        self.parallel_processing = tk.BooleanVar(value=True)
        tk.Checkbutton(
            settings_frame,
            text="Parallel Processing",
            variable=self.parallel_processing,
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

    def _create_history_tab(self):
        """Create analysis history tab"""
        history_frame = ttk.Frame(self.notebook)
        self.notebook.add(history_frame, text="History")

        # History treeview
        columns = ('Date', 'Files', 'Status', 'Report')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='tree headings')

        # Configure columns
        self.history_tree.column('#0', width=0, stretch=False)
        self.history_tree.column('Date', width=150)
        self.history_tree.column('Files', width=100)
        self.history_tree.column('Status', width=100)
        self.history_tree.column('Report', width=200)

        # Configure headings
        for col in columns:
            self.history_tree.heading(col, text=col)

        self.history_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Load history
        self._load_history()

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
        self.report_button.config(state=tk.NORMAL if has_results else tk.DISABLED)

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

                # Process file
                result = self.processor.process_file(file_path)

                # Run ML analysis if enabled
                if self.ml_enabled.get() and result:
                    ml_result = self.ml_analyzer.analyze(result)
                    result['ml_predictions'] = ml_result

                results.append(result)

            if not progress.cancelled:
                # Store results
                self.current_results = results

                # Update UI
                self.root.after(0, self._display_results)

                # Auto-generate report if enabled
                if self.auto_report.get():
                    self.root.after(0, self._generate_report)

                # Update status
                self.root.after(0, lambda: self._update_status(f"Analysis complete: {len(results)} files processed"))

            # Close progress dialog
            self.root.after(0, progress.destroy)

        except Exception as e:
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
        filename = os.path.basename(result.get('filename', 'Unknown'))
        status = result.get('overall_status', 'ERROR')

        self.results_text.insert(tk.END, f"File: {filename}\n", 'filename')
        self.results_text.insert(tk.END, f"Status: {status}\n", 'pass' if status == 'PASS' else 'fail')

        # Key metrics
        if 'analysis_results' in result:
            analysis = result['analysis_results']
            self.results_text.insert(tk.END, f"  Sigma Gradient: {analysis.get('sigma_gradient', 'N/A'):.4f}\n")
            self.results_text.insert(tk.END, f"  Threshold: {analysis.get('sigma_threshold', 'N/A'):.4f}\n")

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

                # Add to history
                self._add_to_history(filename)

            except Exception as e:
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

    def _load_history(self):
        """Load analysis history"""
        # TODO: Implement history loading from database
        # For now, just add a placeholder
        self.history_tree.insert('', 'end', values=(
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            '0',
            'No history',
            '-'
        ))

    def _add_to_history(self, report_path):
        """Add entry to history"""
        self.history_tree.insert('', 0, values=(
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            str(len(self.current_results)),
            'Complete',
            os.path.basename(report_path)
        ))

    def _open_settings(self):
        """Open settings dialog"""
        SettingsDialog(self.root, self.config_manager)

    def _open_model_training(self):
        """Open model training dialog"""
        messagebox.showinfo("Model Training", "Model training interface coming soon!")

    def _open_documentation(self):
        """Open documentation in browser"""
        webbrowser.open("https://github.com/Jb631/laser-trim-ai-system/wiki")

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


class SettingsDialog(tk.Toplevel):
    """Settings dialog for configuring the application"""

    def __init__(self, parent, config_manager):
        super().__init__(parent)
        self.config_manager = config_manager
        self.config = config_manager.load_config()

        self.title("Settings")
        self.geometry("600x500")
        self.resizable(False, False)

        # Center the dialog
        self.transient(parent)
        self.grab_set()

        # Create UI
        self._create_ui()

    def _create_ui(self):
        """Create settings UI"""
        # Main container
        main_frame = tk.Frame(self, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(
            main_frame,
            text="Settings",
            font=('Segoe UI', 16, 'bold'),
            bg='white'
        ).pack(pady=(20, 10))

        # Notebook for categories
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # General settings
        self._create_general_tab(notebook)

        # Analysis settings
        self._create_analysis_tab(notebook)

        # ML settings
        self._create_ml_tab(notebook)

        # Buttons
        button_frame = tk.Frame(main_frame, bg='white')
        button_frame.pack(fill=tk.X, padx=20, pady=(10, 20))

        ModernButton(
            button_frame,
            text="Save",
            bg='#4CAF50',
            command=self._save_settings
        ).pack(side=tk.RIGHT, padx=(10, 0))

        ModernButton(
            button_frame,
            text="Cancel",
            bg='#757575',
            command=self.destroy
        ).pack(side=tk.RIGHT)

    def _create_general_tab(self, notebook):
        """Create general settings tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="General")

        # Settings
        settings_frame = tk.Frame(frame, bg='white')
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Auto-save results
        self.auto_save = tk.BooleanVar(value=self.config.output_settings.auto_save_results)
        tk.Checkbutton(
            settings_frame,
            text="Auto-save analysis results",
            variable=self.auto_save,
            font=('Segoe UI', 10),
            bg='white'
        ).pack(anchor=tk.W, pady=5)

        # Default output directory
        tk.Label(
            settings_frame,
            text="Default output directory:",
            font=('Segoe UI', 10),
            bg='white'
        ).pack(anchor=tk.W, pady=(20, 5))

        dir_frame = tk.Frame(settings_frame, bg='white')
        dir_frame.pack(fill=tk.X, pady=5)

        self.output_dir = tk.StringVar(value=self.config.output_settings.default_output_dir)
        tk.Entry(
            dir_frame,
            textvariable=self.output_dir,
            font=('Segoe UI', 10),
            state='readonly'
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(
            dir_frame,
            text="Browse",
            command=self._browse_output_dir
        ).pack(side=tk.RIGHT, padx=(10, 0))

    def _create_analysis_tab(self, notebook):
        """Create analysis settings tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Analysis")

        # Settings
        settings_frame = tk.Frame(frame, bg='white')
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Sigma threshold
        tk.Label(
            settings_frame,
            text="Default Sigma Threshold:",
            font=('Segoe UI', 10),
            bg='white'
        ).pack(anchor=tk.W, pady=(0, 5))

        self.sigma_threshold = tk.DoubleVar(value=self.config.analysis_settings.default_sigma_threshold)
        tk.Scale(
            settings_frame,
            from_=0.001,
            to=1.0,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            variable=self.sigma_threshold,
            length=300
        ).pack(anchor=tk.W, pady=5)

        # Filter settings
        tk.Label(
            settings_frame,
            text="Filter Cutoff Frequency (Hz):",
            font=('Segoe UI', 10),
            bg='white'
        ).pack(anchor=tk.W, pady=(20, 5))

        self.filter_cutoff = tk.IntVar(value=self.config.analysis_settings.filter_cutoff_freq)
        tk.Scale(
            settings_frame,
            from_=10,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.filter_cutoff,
            length=300
        ).pack(anchor=tk.W, pady=5)

    def _create_ml_tab(self, notebook):
        """Create ML settings tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Machine Learning")

        # Settings
        settings_frame = tk.Frame(frame, bg='white')
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # ML enabled
        self.ml_enabled = tk.BooleanVar(value=self.config.ml_settings.enable_ml_analysis)
        tk.Checkbutton(
            settings_frame,
            text="Enable ML analysis",
            variable=self.ml_enabled,
            font=('Segoe UI', 10),
            bg='white'
        ).pack(anchor=tk.W, pady=5)

        # Auto-retrain
        self.auto_retrain = tk.BooleanVar(value=self.config.ml_settings.auto_retrain)
        tk.Checkbutton(
            settings_frame,
            text="Auto-retrain models when new data available",
            variable=self.auto_retrain,
            font=('Segoe UI', 10),
            bg='white'
        ).pack(anchor=tk.W, pady=5)

        # Model path
        tk.Label(
            settings_frame,
            text="Model directory:",
            font=('Segoe UI', 10),
            bg='white'
        ).pack(anchor=tk.W, pady=(20, 5))

        self.model_dir = tk.StringVar(value=self.config.ml_settings.model_path)
        tk.Entry(
            settings_frame,
            textvariable=self.model_dir,
            font=('Segoe UI', 10),
            state='readonly'
        ).pack(fill=tk.X, pady=5)

    def _browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)

    def _save_settings(self):
        """Save settings and close"""
        # Update config
        self.config.output_settings.auto_save_results = self.auto_save.get()
        self.config.output_settings.default_output_dir = self.output_dir.get()
        self.config.analysis_settings.default_sigma_threshold = self.sigma_threshold.get()
        self.config.analysis_settings.filter_cutoff_freq = self.filter_cutoff.get()
        self.config.ml_settings.enable_ml_analysis = self.ml_enabled.get()
        self.config.ml_settings.auto_retrain = self.auto_retrain.get()

        # Save to file
        self.config_manager.save_config(self.config)

        # Close dialog
        self.destroy()
        messagebox.showinfo("Settings Saved", "Settings have been saved successfully.")


def main():
    """Main entry point"""
    app = LaserTrimAIApp()
    app.run()


class LaserTrimAnalyzerGUI:
    def __init__(self):
        # Existing initialization...
        self.init_database()

    def init_database(self):
        """Initialize database components."""
        try:
            self.db_manager = DatabaseManager(self.config)
            self.analyzer = HistoricalAnalyzer(self.db_manager, self.config)
            self.reporter = TrendReporter(self.db_manager, self.analyzer, self.config)
            self.add_database_menu()
        except Exception as e:
            print(f"Database initialization failed: {e}")

    def add_database_menu(self):
        """Add database menu to GUI."""
        db_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Database", menu=db_menu)

        db_menu.add_command(label="View Historical Data", command=self.view_historical_data)
        db_menu.add_command(label="Generate Trend Report", command=self.generate_trend_report)
        db_menu.add_command(label="Model Analysis", command=self.analyze_models)
        db_menu.add_separator()
        db_menu.add_command(label="Import Data", command=self.import_data)
        db_menu.add_command(label="Export Backup", command=self.export_backup)

if __name__ == "__main__":
    main()