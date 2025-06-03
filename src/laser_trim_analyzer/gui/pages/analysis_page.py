"""
Analysis Page for Laser Trim Analyzer

Provides interface for file selection, processing options,
and analysis execution with results display.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
import asyncio
import threading
import time as time_module
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from laser_trim_analyzer.core.models import AnalysisResult, ProcessingMode
from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.core.exceptions import ProcessingError
from laser_trim_analyzer.gui.pages.base_page import BasePage
from laser_trim_analyzer.gui.widgets.file_drop_zone import FileDropZone
from laser_trim_analyzer.gui.widgets.file_analysis_widget import FileAnalysisWidget
from laser_trim_analyzer.gui.widgets import add_mousewheel_support


class AnalysisPage(BasePage):
    """
    Analysis page for processing laser trim files.

    Features:
    - Drag-and-drop file selection
    - Real-time processing progress
    - ML insights display
    - Result visualization
    """

    def __init__(self, parent: ttk.Frame, main_window: Any):
        """Initialize analysis page."""
        # Initialize state with thread safety
        self.input_files: List[Path] = []
        self.file_widgets: Dict[str, FileAnalysisWidget] = {}
        self.processor: Optional[LaserTrimProcessor] = None
        self.is_processing = False
        self.current_task = None
        
        # File state management
        self._file_selection_lock = threading.Lock()
        self._file_metadata_cache = {}  # Cache file metadata to prevent loss
        self._processing_results = {}  # Store results by file path
        
        # Progress update throttling
        self.last_progress_update = 0
        self.progress_update_interval = 0.05  # Faster updates for responsiveness

        # Processing options with safe defaults
        self.processing_mode = tk.StringVar(value='detail')
        self.enable_plots = tk.BooleanVar(value=True)
        self.enable_ml = tk.BooleanVar(value=True)
        self.enable_database = tk.BooleanVar(value=True)

        # UI components
        self.drop_zone = None
        self.file_list_frame = None
        self.alert_stack = None
        self.progress_frame = None
        self.results_notebook = None
        
        # UI state management
        self._ui_update_lock = threading.Lock()
        self._last_ui_update = 0

        super().__init__(parent, main_window)

    def _create_page(self):
        """Create analysis page content with fully responsive layout."""
        # Create scrollable main frame without shifting
        main_container = ttk.Frame(self)
        main_container.pack(fill='both', expand=True)
        
        # Canvas and scrollbar
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add mouse wheel scrolling support
        add_mousewheel_support(scrollable_frame, canvas)
        
        # Pack scrollbar first to avoid shifting
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Main container in scrollable frame with responsive padding
        container = ttk.Frame(scrollable_frame, style='TFrame')
        container.pack(fill='both', expand=True, padx=25, pady=25)

        # Title and description with more space
        self._create_header(container)

        # Simplified alert stack - remove complex animations and features
        self.alert_stack = SimpleAlertStack(container)
        self.alert_stack.pack(fill='x', pady=(0, 15))

        # Main content area using grid for better responsive control
        content_frame = ttk.Frame(container)
        content_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        # Configure grid to be responsive
        content_frame.grid_columnconfigure(0, weight=3, minsize=400)  # Left column gets 60% (3/5)
        content_frame.grid_columnconfigure(1, weight=2, minsize=300)  # Right column gets 40% (2/5)
        content_frame.grid_rowconfigure(0, weight=1)

        # Left column - File selection and options
        left_frame = ttk.Frame(content_frame)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 15))

        # Right column - Results  
        right_frame = ttk.Frame(content_frame)
        right_frame.grid(row=0, column=1, sticky='nsew')

        # File selection section with responsive height
        self._create_file_section(left_frame)

        # Processing options section with responsive width
        self._create_options_section(left_frame)

        # Results section with responsive dimensions
        self._create_results_section(right_frame)

        # Progress section (spans full width below main content)
        self._create_progress_section(container)

    def _create_header(self, parent):
        """Create page header."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x', pady=(0, 20))

        ttk.Label(
            header_frame,
            text="File Analysis",
            style='Title.TLabel'
        ).pack(side='left')

        # Quick stats
        self.stats_label = ttk.Label(
            header_frame,
            text="",
            font=('Segoe UI', 10),
            foreground=self.colors['text_secondary']
        )
        self.stats_label.pack(side='right')

    def _create_file_section(self, parent):
        """Create file selection section with better height and usability."""
        # File selection frame (drop zone)
        file_frame = ttk.LabelFrame(
            parent,
            text="File Selection",
            padding=15  # Increased padding
        )
        file_frame.pack(fill='both', expand=True, pady=(0, 20))  # Increased bottom padding

        # Create drop zone
        self.drop_zone = FileDropZone(
            file_frame,
            accept_extensions=['.xlsx', '.xls'],
            on_files_dropped=self._handle_files_dropped,
            height=160  # Increased height for better visibility
        )
        self.drop_zone.pack(fill='x', pady=(0, 15))  # More spacing

        # File list with scrollbars and larger height
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill='both', expand=True)

        # Scrollable listbox with better height
        scrollbar_v = ttk.Scrollbar(list_frame, orient='vertical')
        scrollbar_h = ttk.Scrollbar(list_frame, orient='horizontal')

        self.file_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar_v.set,
            xscrollcommand=scrollbar_h.set,
            selectmode='extended',
            font=('Consolas', 9),
            height=18  # Increased height significantly
        )

        scrollbar_v.config(command=self.file_listbox.yview)
        scrollbar_h.config(command=self.file_listbox.xview)

        # Layout with grid for better control
        self.file_listbox.grid(row=0, column=0, sticky='nsew')
        scrollbar_v.grid(row=0, column=1, sticky='ns')
        scrollbar_h.grid(row=1, column=0, sticky='ew')

        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        # Bind double-click to remove files
        self.file_listbox.bind('<Double-Button-1>', self._remove_selected_files)

        # File list frame reference for dynamic updates
        self.file_list_frame = list_frame

        # Stats label with better formatting
        self.stats_label = ttk.Label(
            file_frame,
            text="No files loaded",
            font=('Segoe UI', 10),
            foreground=self.colors['text_secondary']
        )
        self.stats_label.pack(pady=(10, 0))  # Added top padding

    def _create_options_section(self, parent):
        """Create processing options section with responsive layout."""
        options_frame = ttk.LabelFrame(
            parent,
            text="Processing Options",
            padding=15
        )
        options_frame.pack(fill='x', pady=(0, 20))

        # Processing mode with responsive layout
        mode_frame = ttk.Frame(options_frame)
        mode_frame.pack(fill='x', pady=(0, 15))
        
        # Configure mode frame for responsiveness
        mode_frame.grid_columnconfigure(1, weight=1)  # Make combo expand

        ttk.Label(
            mode_frame,
            text="Processing Mode:",
            font=('Segoe UI', 11, 'bold')
        ).grid(row=0, column=0, sticky='w', padx=(0, 10))

        self.mode_var = tk.StringVar(value="Standard")
        mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=["Standard", "Fast", "Detailed", "Custom"],
            state='readonly'
        )
        mode_combo.grid(row=0, column=1, sticky='ew')  # Expand to fill available width

        # ML options with responsive layout
        ml_frame = ttk.LabelFrame(options_frame, text="ML Analysis", padding=10)
        ml_frame.pack(fill='x', pady=(0, 15))

        self.enable_ml_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            ml_frame,
            text="Enable ML failure prediction",
            variable=self.enable_ml_var
        ).pack(anchor='w', pady=(0, 5), fill='x')  # Fill width

        self.enable_optimization_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            ml_frame,
            text="Enable threshold optimization",
            variable=self.enable_optimization_var
        ).pack(anchor='w', fill='x')  # Fill width

        # Quality options with responsive layout
        quality_frame = ttk.LabelFrame(options_frame, text="Quality Analysis", padding=10)
        quality_frame.pack(fill='x', pady=(0, 15))

        self.detailed_analysis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            quality_frame,
            text="Detailed sigma analysis",
            variable=self.detailed_analysis_var
        ).pack(anchor='w', pady=(0, 5), fill='x')  # Fill width

        self.batch_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            quality_frame,
            text="Large batch processing mode",
            variable=self.batch_mode_var
        ).pack(anchor='w', fill='x')  # Fill width

        # Action buttons with responsive layout
        action_frame = ttk.Frame(options_frame)
        action_frame.pack(fill='x', pady=(15, 0))
        
        # Configure action frame for responsive button layout
        action_frame.grid_columnconfigure(0, weight=1)
        action_frame.grid_columnconfigure(1, weight=1)  
        action_frame.grid_columnconfigure(2, weight=1)

        self.analyze_button = ttk.Button(
            action_frame,
            text="🚀 Start Analysis",
            style='Primary.TButton',
            command=self._start_analysis,
            state='disabled'
        )
        self.analyze_button.grid(row=0, column=0, sticky='ew', padx=(0, 5))

        self.cancel_button = ttk.Button(
            action_frame,
            text="⏹ Cancel",
            command=self._cancel_analysis,
            state='disabled'
        )
        self.cancel_button.grid(row=0, column=1, sticky='ew', padx=(5, 5))

        clear_button = ttk.Button(
            action_frame,
            text="🗑 Clear All",
            command=self._clear_files
        )
        clear_button.grid(row=0, column=2, sticky='ew', padx=(5, 0))

        return options_frame

    def _create_results_section(self, parent):
        """Create results display section with responsive layout."""
        results_frame = ttk.LabelFrame(
            parent,
            text="Analysis Results",
            padding=15
        )
        results_frame.pack(fill='both', expand=True)

        # Results summary with responsive layout
        summary_frame = ttk.Frame(results_frame)
        summary_frame.pack(fill='x', pady=(0, 15))

        self.results_summary = ttk.Label(
            summary_frame,
            text="No analysis results yet",
            font=('Segoe UI', 11, 'bold'),
            foreground=self.colors['text_secondary']
        )
        self.results_summary.pack(expand=True)  # Center and expand

        # Create notebook for different result views - fully responsive
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill='both', expand=True)

        # Summary tab with responsive display area
        summary_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(summary_tab, text="📊 Summary")

        # Scrollable text widget with responsive layout
        summary_text_frame = ttk.Frame(summary_tab)
        summary_text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configure text frame for responsiveness
        summary_text_frame.grid_columnconfigure(0, weight=1)
        summary_text_frame.grid_rowconfigure(0, weight=1)

        # Text widget with scrollbar - responsive height
        self.summary_text = tk.Text(
            summary_text_frame,
            wrap='word',
            state='disabled',
            font=('Segoe UI', 10)
            # Remove fixed height to make it truly responsive
        )
        summary_scrollbar = ttk.Scrollbar(
            summary_text_frame,
            orient='vertical',
            command=self.summary_text.yview
        )
        self.summary_text.config(yscrollcommand=summary_scrollbar.set)

        # Use grid for better responsive control
        self.summary_text.grid(row=0, column=0, sticky='nsew')
        summary_scrollbar.grid(row=0, column=1, sticky='ns')

        # Details tab with responsive layout
        details_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(details_tab, text="📋 Details")

        # Treeview for detailed results with responsive dimensions
        details_frame = ttk.Frame(details_tab)
        details_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configure details frame for responsiveness
        details_frame.grid_columnconfigure(0, weight=1)
        details_frame.grid_rowconfigure(0, weight=1)

        # Treeview with scrollbars - responsive sizing
        tree_columns = ('File', 'Status', 'Sigma', 'Linearity', 'Risk')
        self.details_tree = ttk.Treeview(
            details_frame,
            columns=tree_columns,
            show='tree headings'
            # Remove fixed height for responsive behavior
        )

        # Configure columns with responsive widths
        self.details_tree.heading('#0', text='')
        self.details_tree.column('#0', width=0, stretch=False)
        
        # Make columns responsive to window size
        for i, col in enumerate(tree_columns):
            self.details_tree.heading(col, text=col)
            # Use relative widths instead of fixed
            min_width = [200, 80, 100, 100, 80][i]  # Minimum widths
            self.details_tree.column(col, width=min_width, stretch=True)

        # Scrollbars for treeview
        tree_v_scroll = ttk.Scrollbar(
            details_frame,
            orient='vertical',
            command=self.details_tree.yview
        )
        tree_h_scroll = ttk.Scrollbar(
            details_frame,
            orient='horizontal',
            command=self.details_tree.xview
        )

        self.details_tree.config(
            yscrollcommand=tree_v_scroll.set,
            xscrollcommand=tree_h_scroll.set
        )

        # Grid layout for responsive control
        self.details_tree.grid(row=0, column=0, sticky='nsew')
        tree_v_scroll.grid(row=0, column=1, sticky='ns')
        tree_h_scroll.grid(row=1, column=0, sticky='ew')

        # Charts tab for visualizations - responsive
        charts_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(charts_tab, text="📈 Charts")

        # Chart display area with responsive size
        chart_display_frame = ttk.Frame(charts_tab)
        chart_display_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Placeholder for charts - responsive
        self.chart_display_label = ttk.Label(
            chart_display_frame,
            text="Charts will be displayed here after analysis",
            font=('Segoe UI', 11),
            foreground=self.colors['text_secondary']
        )
        self.chart_display_label.pack(expand=True)

        # Export buttons with responsive layout
        export_frame = ttk.Frame(results_frame)
        export_frame.pack(fill='x', pady=(15, 0))
        
        # Configure export frame for responsive button layout
        export_frame.grid_columnconfigure(0, weight=1)
        export_frame.grid_columnconfigure(1, weight=1)

        export_results_btn = ttk.Button(
            export_frame,
            text="📁 Export Results",
            command=self._export_results
        )
        export_results_btn.grid(row=0, column=0, sticky='ew', padx=(0, 5))

        generate_report_btn = ttk.Button(
            export_frame,
            text="📊 Generate Report",
            command=self._generate_report
        )
        generate_report_btn.grid(row=0, column=1, sticky='ew', padx=(5, 0))

    def _create_progress_section(self, parent):
        """Create progress section that spans full width responsively."""
        # Progress bar (initially hidden)
        self.progress_frame = ttk.Frame(parent)
        # Don't pack initially - will be shown during analysis

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill='x', pady=(0, 10), expand=True)  # Make responsive

        self.progress_label = ttk.Label(
            self.progress_frame,
            text="",
            font=('Segoe UI', 10)
        )
        self.progress_label.pack(expand=True)  # Center the label

    def _remove_selected_files(self, event):
        """Remove selected files from the listbox."""
        try:
            selection = self.file_listbox.curselection()
            if selection:
                # Get selected file paths
                selected_files = []
                for index in selection:
                    file_path = self.file_listbox.get(index)
                    selected_files.append(file_path)
                
                # Remove from input_files list
                for file_path in selected_files:
                    # Find matching path in input_files
                    for i, input_file in enumerate(self.input_files):
                        if input_file.name in file_path:
                            self.input_files.pop(i)
                            break
                
                # Remove from listbox (reverse order to maintain indices)
                for index in reversed(selection):
                    self.file_listbox.delete(index)
                
                # Update UI
                self._update_stats()
                self._update_ui_state()
                
        except Exception as e:
            self.logger.error(f"Error removing selected files: {e}")

    def _export_results(self):
        """Export analysis results."""
        # Placeholder for export functionality
        self.alert_stack.add_alert(
            alert_type='info',
            title='Export',
            message='Export functionality will be implemented in a future update.',
            auto_dismiss=3
        )

    def _generate_report(self):
        """Generate analysis report."""
        # Placeholder for report generation
        self.alert_stack.add_alert(
            alert_type='info',
            title='Report Generation',
            message='Report generation functionality will be implemented in a future update.',
            auto_dismiss=3
        )

    def browse_files(self):
        """Open file browser to select files with improved handling."""
        try:
            # Show brief loading message
            self.alert_stack.add_alert(
                alert_type='info',
                title='File Dialog',
                message='Opening file browser...',
                auto_dismiss=2
            )
            
            # Ensure UI updates before showing dialog
            self.update_idletasks()
            
            files = filedialog.askopenfilenames(
                title="Select Excel Files",
                filetypes=[
                    ("Excel files", "*.xlsx *.xls"),
                    ("All files", "*.*")
                ]
            )

            if files:
                # Clear the loading message
                for alert in self.alert_stack.alerts[:]:
                    if hasattr(alert, 'title') and 'File Dialog' in alert.title:
                        alert.dismiss()
                        
                # Show selection feedback
                file_count = len(files)
                self.alert_stack.add_alert(
                    alert_type='info',
                    title='Files Selected',
                    message=f'Selected {file_count} file{"s" if file_count != 1 else ""} for analysis.',
                    auto_dismiss=2
                )
                
                self._add_files([Path(f) for f in files])
            else:
                # User cancelled - clear loading message
                for alert in self.alert_stack.alerts[:]:
                    if hasattr(alert, 'title') and 'File Dialog' in alert.title:
                        alert.dismiss()
                        
        except Exception as e:
            # Handle any file dialog errors
            self.logger.error(f"Error in file dialog: {e}")
            self.alert_stack.add_alert(
                alert_type='error',
                title='File Selection Error',
                message=f'Error opening file browser: {str(e)}',
                auto_dismiss=5
            )

    def _handle_files_dropped(self, files: List[str]):
        """Handle files dropped on drop zone."""
        self._add_files([Path(f) for f in files])

    def _add_files(self, files: List[Path]):
        """Add files to the processing list with enhanced state management and error handling."""
        if not files:
            return
            
        with self._file_selection_lock:
            total_files = len(files)
            
            # Show immediate feedback for large batches
            if total_files > 100:
                temp_alert = self.alert_stack.add_alert(
                    alert_type='info',
                    title='Processing Files',
                    message=f'Analyzing {total_files} files...',
                    auto_dismiss=None,
                    dismissible=False
                )
            
            # Filter for valid files efficiently with enhanced validation
            valid_files = []
            valid_extensions = {'.xlsx', '.xls'}  # Use set for faster lookup
            existing_files = {str(f) for f in self.input_files}  # Convert to string set for O(1) lookup
            
            for file in files:
                try:
                    # Enhanced file validation
                    if (file.suffix.lower() in valid_extensions and 
                        str(file) not in existing_files and
                        file.exists() and
                        file.is_file()):
                        
                        # Pre-cache basic file metadata to prevent loss
                        try:
                            file_stats = file.stat()
                            self._file_metadata_cache[str(file)] = {
                                'name': file.name,
                                'size': file_stats.st_size,
                                'modified': file_stats.st_mtime,
                                'path': str(file),
                                'status': 'Selected',
                                'added_time': time_module.time()
                            }
                        except Exception as e:
                            self.logger.warning(f"Failed to cache metadata for {file.name}: {e}")
                        
                        valid_files.append(file)
                        
                except Exception as e:
                    self.logger.warning(f"Error validating file {file}: {e}")
                    continue

            # Dismiss temporary alert for large batches
            if total_files > 100 and 'temp_alert' in locals():
                try:
                    temp_alert.dismiss()
                except:
                    pass

            if not valid_files:
                if total_files > 0:
                    self.alert_stack.add_alert(
                        alert_type='warning',
                        title='No Valid Files',
                        message='No new Excel files found to add.',
                        auto_dismiss=3
                    )
                return

            # Add to list with atomic operation
            try:
                self.input_files.extend(valid_files)
                self.logger.info(f"Added {len(valid_files)} files to analysis queue")
                
                # Update UI state immediately to show files
                self._update_stats()
                self._update_ui_state()
                
                # Persist file selection in cache
                for file in valid_files:
                    if str(file) in self._file_metadata_cache:
                        self._file_metadata_cache[str(file)]['status'] = 'Ready'
                
                # Create UI widgets based on batch size
                if len(valid_files) > 100:  # Reduced from 200 for smoother experience
                    self._switch_to_tree_view_mode()
                    self._populate_tree_view_immediate(valid_files)
                else:
                    # Use optimized widget creation for smaller batches
                    self._create_file_widgets_optimized(valid_files)
                    
            except Exception as e:
                self.logger.error(f"Error adding files to list: {e}")
                self.alert_stack.add_alert(
                    alert_type='error',
                    title='File Addition Error',
                    message=f'Error adding files: {str(e)}',
                    auto_dismiss=5
                )

    def _switch_to_tree_view_mode(self):
        """Switch the file list to tree view mode for large batches."""
        # Clear existing widgets
        for widget in self.file_widgets.values():
            widget.destroy()
        self.file_widgets.clear()
        
        # Remove the current file list frame content
        for child in self.file_list_frame.winfo_children():
            child.destroy()
            
        # Create tree view
        self.file_tree = ttk.Treeview(
            self.file_list_frame, 
            columns=('Model', 'Serial', 'Status'), 
            show='tree headings',
            height=15
        )
        
        # Configure columns
        self.file_tree.column('#0', width=300, stretch=True)
        self.file_tree.column('Model', width=80)
        self.file_tree.column('Serial', width=100)  
        self.file_tree.column('Status', width=80)
        
        # Set headings
        self.file_tree.heading('#0', text='Filename')
        self.file_tree.heading('Model', text='Model')
        self.file_tree.heading('Serial', text='Serial')
        self.file_tree.heading('Status', text='Status')
        
        # Add scrollbar
        tree_scrollbar = ttk.Scrollbar(self.file_list_frame, orient='vertical', command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        # Pack tree and scrollbar
        self.file_tree.pack(side='left', fill='both', expand=True)
        tree_scrollbar.pack(side='right', fill='y')
        
        # Configure status colors
        self.file_tree.tag_configure('pending', foreground='gray')
        self.file_tree.tag_configure('processing', foreground='blue')
        self.file_tree.tag_configure('pass', foreground=self.colors['success'])
        self.file_tree.tag_configure('fail', foreground=self.colors['danger'])
        self.file_tree.tag_configure('warning', foreground=self.colors['warning'])
        
        # Add context menu for tree items
        self._setup_tree_context_menu()
        
        # Show info message about tree view mode
        self.alert_stack.add_alert(
            alert_type='info',
            title='Large Batch Mode',
            message=f'Switched to optimized view for {len(self.input_files)} files. Right-click for options.',
            auto_dismiss=None,  # Don't auto-dismiss
            dismissible=True,   # But allow manual dismiss
            actions=[
                {'text': 'Got it', 'command': lambda: self._dismiss_large_batch_alert()}
            ]
        )

    def _populate_tree_view_immediate(self, files: List[Path]):
        """Immediately populate tree view - this is fast since it's just text data."""
        for file in files:
            # Extract model and serial from filename
            model = file.stem.split('_')[0] if '_' in file.stem else 'Unknown'
            serial = file.stem.split('_')[1] if '_' in file.stem else 'Unknown'
            
            # Insert into tree
            item_id = self.file_tree.insert(
                '',
                'end',
                text=file.name,
                values=(model, serial, 'Pending'),
                tags=('pending',)
            )
            
            # Store mapping for updates during processing
            self.file_widgets[str(file)] = {'tree_item': item_id, 'tree_mode': True}

    def _setup_tree_context_menu(self):
        """Setup context menu for tree view items."""
        self.tree_context_menu = tk.Menu(self, tearoff=0)
        self.tree_context_menu.add_command(label="View Details", command=self._tree_view_details)
        self.tree_context_menu.add_command(label="Remove File", command=self._tree_remove_file)
        self.tree_context_menu.add_separator()
        self.tree_context_menu.add_command(label="Export Results", command=self._tree_export_file)
        
        # Bind right-click
        self.file_tree.bind("<Button-3>", self._show_tree_context_menu)
        
    def _show_tree_context_menu(self, event):
        """Show context menu for tree item."""
        item = self.file_tree.identify_row(event.y)
        if item:
            self.file_tree.selection_set(item)
            self.tree_context_menu.post(event.x_root, event.y_root)
            
    def _tree_view_details(self):
        """View details for selected tree item."""
        selection = self.file_tree.selection()
        if selection:
            item = selection[0]
            filename = self.file_tree.item(item, 'text')
            # Find the file data
            file_path = None
            for f in self.input_files:
                if f.name == filename:
                    file_path = f
                    break
            if file_path:
                file_data = {'filename': filename, 'file_path': str(file_path)}
                self._show_file_details(file_data)
                
    def _tree_remove_file(self):
        """Remove selected file from tree."""
        selection = self.file_tree.selection()
        if selection:
            item = selection[0]
            filename = self.file_tree.item(item, 'text')
            # Remove from input_files
            self.input_files = [f for f in self.input_files if f.name != filename]
            # Remove from tree
            self.file_tree.delete(item)
            # Remove from widgets mapping
            for path, widget_data in list(self.file_widgets.items()):
                if isinstance(widget_data, dict) and widget_data.get('tree_item') == item:
                    del self.file_widgets[path]
                    break
            self._update_stats()
            self._update_ui_state()
            
    def _tree_export_file(self):
        """Export results for selected tree item."""
        selection = self.file_tree.selection()
        if selection:
            item = selection[0]
            filename = self.file_tree.item(item, 'text')
            self.alert_stack.add_alert(
                alert_type='info',
                title='Export',
                message=f'Export functionality for {filename} not yet implemented.',
                auto_dismiss=3
            )

    def _create_file_widgets_optimized(self, files: List[Path]):
        """Create file widgets with optimized performance for responsive UI."""
        total_files = len(files)
        
        # Determine batch size and update frequency based on total files - SMALLER BATCHES for smoother UI
        if total_files > 100:
            batch_size = 25  # Reduced from 50
            update_frequency = 15  # Reduced from 25
        else:
            batch_size = 10  # Reduced from 20
            update_frequency = 5   # Reduced from 10
            
        # Track progress with single alert
        progress_alert_ref = {'alert': None}
        
        # Show progress for larger batches
        if total_files > 50:
            progress_alert_ref['alert'] = self.alert_stack.add_alert(
                alert_type='info',
                title='Loading Files',
                message=f'Loading {total_files} files... 0%',
                auto_dismiss=None,
                dismissible=False
            )
        
        def create_widgets_batch():
            """Create widgets in optimized batches with proper error handling."""
            nonlocal progress_alert_ref
            
            try:
                for i in range(0, total_files, batch_size):
                    batch = files[i:i + batch_size]
                    
                    # Create widgets for this batch with error recovery
                    widgets_created = []
                    for file in batch:
                        try:
                            file_path_str = str(file)
                            if file_path_str not in self.file_widgets:  # Avoid duplicates
                                
                                # Get cached metadata or create basic info
                                file_info = self._file_metadata_cache.get(file_path_str, {
                                    'name': file.name,
                                    'path': file_path_str,
                                    'status': 'Ready'
                                })
                                
                                # Extract model and serial from filename
                                parts = file.stem.split('_')
                                model = parts[0] if len(parts) > 0 else 'Unknown'
                                serial = parts[1] if len(parts) > 1 else 'Unknown'
                                
                                widget = FileAnalysisWidget(
                                    self.file_list_frame,
                                    file_data={
                                        'filename': file.name,
                                        'file_path': file_path_str,
                                        'status': file_info.get('status', 'Ready'),
                                        'model': model,
                                        'serial': serial,
                                        'size': file_info.get('size', 0)
                                    },
                                    on_view_plot=self._view_plot,
                                    on_export=self._export_file_results,
                                    on_details=self._show_file_details
                                )
                                widgets_created.append(widget)
                                self.file_widgets[file_path_str] = widget
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to create widget for {file.name}: {e}")
                            continue
                    
                    # Pack all widgets in this batch at once
                    for widget in widgets_created:
                        try:
                            widget.pack(fill='x', pady=(0, 2))
                        except Exception as e:
                            self.logger.warning(f"Failed to pack widget: {e}")
                    
                    # Update progress periodically
                    current_count = min(i + batch_size, total_files)
                    progress = (current_count / total_files) * 100
                    
                    if current_count % update_frequency == 0 or current_count >= total_files:
                        if progress_alert_ref['alert']:
                            # Note: Simplified alerts don't support updating, so we'll just track progress
                            pass
                    
                    # Schedule next batch or completion - REDUCED DELAY for smoother experience
                    if i + batch_size < total_files:
                        delay = 2 if total_files > 200 else 5  # Reduced delays
                        self.after(delay, create_widgets_batch)
                        return
                    else:
                        # All done - cleanup and show completion
                        self.after(10, lambda: self._widget_creation_complete(progress_alert_ref))
                        return
                        
            except Exception as e:
                self.logger.error(f"Error in widget creation batch: {e}")
                # Try to complete gracefully
                self.after(10, lambda: self._widget_creation_complete(progress_alert_ref))
        
        # Start the async creation - IMMEDIATE START
        self.after(1, create_widgets_batch)  # Reduced from 5ms

    def _widget_creation_complete(self, alert_ref: dict):
        """Called when all widgets have been created."""
        # Note: Simplified alerts auto-dismiss, so no need to manually dismiss
        
        # Show completion message for large batches
        if len(self.input_files) > 50:
            self.alert_stack.add_alert(
                alert_type='success',
                title='Files Loaded',
                message=f'Successfully loaded {len(self.input_files)} files for analysis.',
                auto_dismiss=3
            )

    def _clear_files(self):
        """Clear all selected files with proper state cleanup."""
        with self._file_selection_lock:
            # Clear file lists
            self.input_files.clear()
            
            # Clear cached metadata
            self._file_metadata_cache.clear()
            
            # Clear processing results
            self._processing_results.clear()

            # Remove widgets or tree view
            if hasattr(self, 'file_tree'):
                # Tree view mode - destroy tree and recreate original layout
                self.file_tree.destroy()
                if hasattr(self, 'tree_scrollbar'):
                    self.tree_scrollbar.destroy()
                delattr(self, 'file_tree')
                
                # Recreate original file list frame layout
                self._recreate_file_list_layout()
            else:
                # Individual widget mode
                for widget in self.file_widgets.values():
                    if hasattr(widget, 'destroy'):
                        widget.destroy()
                        
            self.file_widgets.clear()

            # Clear results
            self._show_empty_results()

            # Update UI
            self._update_ui_state()
            self._update_stats()

    def _recreate_file_list_layout(self):
        """Recreate the original file list layout for widget mode."""
        # Clear any remaining children
        for child in self.file_list_frame.winfo_children():
            child.destroy()
            
        # The file_list_frame is ready for widgets again
        # No need to recreate it since _create_file_widgets_optimized will populate it

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        has_files = len(self.input_files) > 0

        # Update buttons
        self.analyze_button.config(
            state='normal' if has_files and not self.is_processing else 'disabled'
        )
        self.cancel_button.config(
            state='normal' if self.is_processing else 'disabled'
        )

        # Update drop zone - keep enabled during processing for transparency
        if self.is_processing:
            self.drop_zone.set_state('processing')  # New state to show processing
        else:
            self.drop_zone.set_state('normal')
            
        # Update file list visibility
        if has_files:
            # Ensure file list is visible
            if hasattr(self, 'file_tree'):
                self.file_tree.pack(side='left', fill='both', expand=True)
            elif self.file_list_frame:
                self.file_list_frame.pack(fill='both', expand=True)

    def _update_stats(self):
        """Update statistics label."""
        num_files = len(self.input_files)
        if num_files == 0:
            self.stats_label.config(text="")
        else:
            self.stats_label.config(text=f"{num_files} file{'s' if num_files != 1 else ''} selected")

    def _start_analysis(self):
        """Start processing selected files with enhanced responsiveness."""
        if not self.input_files or self.is_processing:
            return

        self.is_processing = True
        self._update_ui_state()

        # Show immediate responsive feedback
        self.alert_stack.add_alert(
            alert_type='info',
            title='Starting Analysis',
            message=f'Initializing analysis for {len(self.input_files)} files...',
            dismissible=False,
            allow_scroll=True  # Keep app scrollable during processing
        )

        # Show progress with responsive design
        self.progress_frame.pack(fill='x', pady=(10, 0))
        self.progress_var.set(0)
        self.progress_label.config(text="Preparing analysis environment...")

        # Clear previous alerts but keep critical ones
        self._clear_non_critical_alerts()
        
        # Ensure files remain visible during processing with immediate feedback
        self._ensure_files_visible_responsive()

        # Start processing in background thread with responsive UI updates
        thread = threading.Thread(target=self._process_files_thread_responsive, daemon=True)
        thread.start()

        # Schedule UI responsiveness checks
        self._schedule_responsiveness_checks()

    def _clear_non_critical_alerts(self):
        """Clear non-critical alerts while preserving important ones."""
        try:
            # With simplified alerts, just clear all alerts for a clean slate
            # The simplified system doesn't distinguish between alert types
            self.alert_stack.clear_all()
        except Exception as e:
            self.logger.warning(f"Error clearing alerts: {e}")

    def _ensure_files_visible_responsive(self):
        """Ensure selected files remain visible during processing with responsive updates."""
        try:
            total_files = len(self.file_widgets)
            processed_count = 0
            
            # Update files in batches to maintain responsiveness
            for file_path, widget_data in self.file_widgets.items():
                if isinstance(widget_data, dict) and widget_data.get('tree_mode'):
                    # Tree view mode - ensure item is visible
                    item_id = widget_data['tree_item']
                    if hasattr(self, 'file_tree') and self.file_tree.exists(item_id):
                        current_values = list(self.file_tree.item(item_id, 'values'))
                        current_values[2] = 'Ready'  # Status column
                        self.file_tree.item(item_id, values=current_values)
                        self.file_tree.item(item_id, tags=('ready',))
                else:
                    # Individual widget mode
                    if hasattr(widget_data, 'update_data'):
                        widget_data.update_data({'status': 'Ready'})
                
                processed_count += 1
                
                # Update UI every 10 files to maintain responsiveness
                if processed_count % 10 == 0:
                    self.update_idletasks()
                        
            # Configure tree view colors for processing states
            if hasattr(self, 'file_tree'):
                self.file_tree.tag_configure('ready', foreground='blue')
                self.file_tree.tag_configure('processing', foreground='orange')
                self.file_tree.tag_configure('completed', foreground='green')
                self.file_tree.tag_configure('error', foreground='red')
                
        except Exception as e:
            self.logger.warning(f"Error ensuring files visible: {e}")

    def _schedule_responsiveness_checks(self):
        """Schedule periodic UI responsiveness checks during processing."""
        def check_responsiveness():
            if self.is_processing:
                # Update UI to ensure responsiveness
                try:
                    self.update_idletasks()
                    # Schedule next check
                    self.after(500, check_responsiveness)  # Check every 500ms
                except:
                    pass  # If UI is being destroyed, stop checks
                    
        # Start responsiveness checks
        self.after(500, check_responsiveness)

    def _process_files_thread_responsive(self):
        """Background thread for file processing with responsive UI updates."""
        try:
            # Create event loop for async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run async processing with responsive updates
            results = loop.run_until_complete(self._process_files_async_responsive())

            # Update UI in main thread
            self.after(0, self._processing_complete_responsive, results)

        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            self.after(0, self._processing_error_responsive, str(e))

        finally:
            loop.close()

    async def _process_files_async_responsive(self) -> List[AnalysisResult]:
        """Async method to process files with responsive UI updates."""
        # Initialize processor with progress feedback
        self.after(0, self._update_progress_responsive, 5, "Initializing processor...")
        
        try:
            self.processor = LaserTrimProcessor(
                config=self.config,
                db_manager=self.db_manager if self.enable_database.get() else None,
                ml_predictor=self.main_window.ml_predictor if self.enable_ml.get() else None,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize processor: {e}")
            raise ProcessingError(f"Processor initialization failed: {e}")

        # Configure based on options
        self.config.processing.generate_plots = (
                self.enable_plots.get() and self.processing_mode.get() == 'detail'
        )

        # Create output directory with feedback
        self.after(0, self._update_progress_responsive, 10, "Creating output directory...")
        try:
            output_dir = self.config.data_directory / datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise ProcessingError(f"Output directory creation failed: {e}")

        results = []
        total_files = len(self.input_files)
        
        # Validate we have files to process
        if total_files == 0:
            raise ProcessingError("No files selected for processing")
        
        # Update initial progress
        self.after(0, self._update_progress_responsive, 15, f"Starting analysis of {total_files} files...")

        # Process files with enhanced error handling and state preservation
        for i, file_path in enumerate(self.input_files):
            # Calculate progress with proper scaling
            base_progress = 15 + (i / total_files) * 80  # Scale to 15-95%
            
            # Update progress with responsive feedback
            self.after(0, self._update_progress_responsive, 
                      base_progress, f"Processing {file_path.name} ({i+1}/{total_files})...")

            # Update file widget status immediately with state preservation
            self.after(0, self._update_file_status_responsive, str(file_path), 'Processing')

            try:
                # Process file with responsive progress callbacks
                result = await self.processor.process_file(
                    file_path,
                    output_dir / file_path.stem,
                    progress_callback=lambda msg, prog: self.after(
                        0, self._update_progress_responsive,
                        base_progress + (prog * 0.8),  # Scale sub-progress within file progress
                        f"{file_path.name}: {msg}"
                    )
                )

                # Store result to prevent loss
                self._processing_results[str(file_path)] = result

                # Save to database if enabled with non-blocking feedback
                if self.enable_database.get() and self.db_manager:
                    try:
                        self.after(0, self._update_progress_responsive, 
                                  base_progress + 70, f"Saving {file_path.name} to database...")
                        
                        # Check for duplicates first
                        existing_id = self.db_manager.check_duplicate_analysis(
                            result.metadata.model,
                            result.metadata.serial,
                            result.metadata.file_date
                        )
                        
                        if existing_id:
                            self.logger.info(f"Duplicate analysis found for {file_path.name} (ID: {existing_id})")
                            result.db_id = existing_id
                        else:
                            # Try normal save first
                            try:
                                result.db_id = self.db_manager.save_analysis(result)
                                self.logger.info(f"Saved analysis to database with ID: {result.db_id}")
                                
                                # Validate the save
                                if not self.db_manager.validate_saved_analysis(result.db_id):
                                    raise RuntimeError("Database validation failed")
                                    
                            except Exception as save_error:
                                self.logger.warning(f"Normal save failed, trying force save: {save_error}")
                                # Try force save as fallback
                                result.db_id = self.db_manager.force_save_analysis(result)
                                self.logger.info(f"Force saved analysis to database with ID: {result.db_id}")
                        
                    except Exception as e:
                        self.logger.error(f"Database save failed for {file_path.name}: {e}")
                        # Show non-blocking warning
                        self.after(0, self._show_non_blocking_warning, file_path.name, "Database save failed")

                results.append(result)

                # Update file widget with results (responsive)
                self.after(0, self._update_file_widget_responsive, str(file_path), result)

                # Allow UI breathing room between files
                await asyncio.sleep(0.1)  # 100ms pause between files

            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {e}")
                self.after(0, self._update_file_status_responsive, str(file_path), 'Error')
                self.after(0, self._show_file_error_responsive, file_path.name, str(e))
                
                # Store error state to preserve file information
                self._processing_results[str(file_path)] = {
                    'error': str(e),
                    'status': 'Error',
                    'file_path': str(file_path)
                }

        # Final progress update
        self.after(0, self._update_progress_responsive, 95, "Finalizing results...")
        
        return results

    def _update_progress_responsive(self, value: float, text: str):
        """Update progress display with enhanced responsiveness."""
        current_time = time_module.time()
        
        # More responsive throttling for better user feedback
        if current_time - self.last_progress_update < 0.05:  # 50ms instead of 100ms
            return
            
        self.last_progress_update = current_time
        
        try:
            self.progress_var.set(min(100, max(0, value)))  # Clamp between 0-100
            self.progress_label.config(text=text)
            
            # Force immediate UI update for responsiveness
            self.progress_bar.update_idletasks()
            self.progress_label.update_idletasks()
            
        except Exception as e:
            self.logger.warning(f"Progress update error: {e}")

    def _update_file_status_responsive(self, file_path: str, status: str):
        """Update file widget status with responsive feedback and state preservation."""
        try:
            # Update cached metadata to preserve state
            if file_path in self._file_metadata_cache:
                self._file_metadata_cache[file_path]['status'] = status
                self._file_metadata_cache[file_path]['last_updated'] = time_module.time()
            
            if file_path in self.file_widgets:
                widget_data = self.file_widgets[file_path]
                if isinstance(widget_data, dict) and widget_data.get('tree_mode'):
                    # Tree view mode
                    item_id = widget_data['tree_item']
                    if hasattr(self, 'file_tree') and self.file_tree.exists(item_id):
                        current_values = list(self.file_tree.item(item_id, 'values'))
                        current_values[2] = status  # Status is the 3rd column
                        self.file_tree.item(item_id, values=current_values)
                        
                        # Update tag for color coding with immediate visual feedback
                        status_tag = status.lower()
                        self.file_tree.item(item_id, tags=(status_tag,))
                        
                        # Ensure tree view updates immediately
                        self.file_tree.update_idletasks()
                else:
                    # Individual widget mode
                    if hasattr(widget_data, 'update_data'):
                        widget_data.update_data({'status': status})
                        # Force widget update
                        widget_data.update_idletasks()
                        
        except Exception as e:
            self.logger.warning(f"File status update error: {e}")

    def _update_file_widget_responsive(self, file_path: str, result: AnalysisResult):
        """Update file widget with analysis results using responsive updates and state preservation."""
        try:
            # Store result in cache for state preservation
            self._processing_results[file_path] = result
            
            # Update cached metadata
            if file_path in self._file_metadata_cache:
                self._file_metadata_cache[file_path].update({
                    'status': 'Completed',
                    'result': result,
                    'completed_time': time_module.time()
                })
            
            if file_path in self.file_widgets:
                widget_data = self.file_widgets[file_path]
                
                if isinstance(widget_data, dict) and widget_data.get('tree_mode'):
                    # Tree view mode - update tree item responsively
                    item_id = widget_data['tree_item']
                    primary_track = result.primary_track
                    
                    # Update tree item values
                    self.file_tree.item(item_id, values=(
                        result.metadata.model,
                        result.metadata.serial,
                        'Completed'  # Mark as completed after processing
                    ))
                    
                    # Update tag for color coding
                    status_tag = 'completed'
                    self.file_tree.item(item_id, tags=(status_tag,))
                    
                    # Store result data for context menu access
                    widget_data['result'] = result
                    
                    # Force immediate tree update
                    self.file_tree.update_idletasks()
                    
                else:
                    # Individual widget mode - use existing logic with responsiveness
                    primary_track = result.primary_track

                    widget_update_data = {
                        'filename': result.metadata.filename,
                        'model': result.metadata.model,
                        'serial': result.metadata.serial,
                        'status': 'Completed',  # Mark as completed
                        'timestamp': datetime.now(),
                        'has_multi_tracks': result.metadata.has_multi_tracks,
                        'sigma_gradient': primary_track.sigma_analysis.sigma_gradient,
                        'sigma_pass': primary_track.sigma_analysis.sigma_pass,
                        'linearity_pass': primary_track.linearity_analysis.linearity_pass,
                        'risk_category': primary_track.failure_prediction.risk_category.value if primary_track.failure_prediction else 'Unknown',
                        'plot_path': primary_track.plot_path
                    }

                    # Add tracks for multi-track files
                    if result.metadata.has_multi_tracks:
                        widget_update_data['tracks'] = {}
                        for track_id, track in result.tracks.items():
                            widget_update_data['tracks'][track_id] = {
                                'status': self._determine_track_status(track),
                                'sigma_gradient': track.sigma_analysis.sigma_gradient,
                                'sigma_pass': track.sigma_analysis.sigma_pass,
                                'linearity_pass': track.linearity_analysis.linearity_pass,
                                'risk_category': track.failure_prediction.risk_category.value if track.failure_prediction else 'Unknown'
                            }

                    widget_data.update_data(widget_update_data)
                    
                    # Force widget update
                    widget_data.update_idletasks()
                    
        except Exception as e:
            self.logger.warning(f"File widget update error: {e}")

    def _determine_track_status(self, track) -> str:
        """Determine track status for display."""
        if not track.sigma_analysis.sigma_pass or not track.linearity_analysis.linearity_pass:
            return 'Fail'
        elif track.failure_prediction and track.failure_prediction.risk_category.value == 'High':
            return 'Warning'
        else:
            return 'Pass'

    def _show_non_blocking_warning(self, filename: str, message: str):
        """Show a non-blocking warning that doesn't interrupt processing."""
        try:
            self.alert_stack.add_alert(
                alert_type='warning',
                title=f'Warning: {filename}',
                message=message,
                auto_dismiss=8,  # Auto-dismiss to avoid UI clutter
                allow_scroll=True,  # Keep app responsive
                dismissible=True
            )
        except Exception as e:
            self.logger.warning(f"Error showing warning: {e}")

    def _show_file_error_responsive(self, filename: str, error: str):
        """Show error alert for file with responsive design."""
        try:
            # Truncate very long error messages for better UX
            display_error = error[:150] + "..." if len(error) > 150 else error
            
            self.alert_stack.add_alert(
                alert_type='error',
                title=f'Error: {filename}',
                message=display_error,
                dismissible=True,
                allow_scroll=True,  # Keep app scrollable
                actions=[
                    {'text': 'View Details', 'command': lambda: self._show_detailed_error(filename, error)},
                    {'text': 'Continue', 'command': lambda: None}  # Just dismiss
                ]
            )
        except Exception as e:
            self.logger.warning(f"Error showing file error: {e}")

    def _show_detailed_error(self, filename: str, error: str):
        """Show detailed error information in a separate dialog."""
        try:
            dialog = tk.Toplevel(self.winfo_toplevel())
            dialog.title(f"Error Details - {filename}")
            dialog.geometry("600x400")
            dialog.configure(bg=self.colors['bg'])
            
            # Error details text
            text_frame = ttk.Frame(dialog)
            text_frame.pack(fill='both', expand=True, padx=20, pady=20)
            
            error_text = tk.Text(text_frame, wrap='word', font=('Consolas', 10))
            scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=error_text.yview)
            error_text.configure(yscrollcommand=scrollbar.set)
            
            error_text.insert('1.0', f"File: {filename}\n\nError Details:\n{error}")
            error_text.config(state='disabled')
            
            error_text.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')
            
            # Close button
            ttk.Button(
                dialog,
                text="Close",
                command=dialog.destroy
            ).pack(pady=(0, 20))
            
        except Exception as e:
            self.logger.error(f"Error showing detailed error dialog: {e}")

    def _processing_complete_responsive(self, results: List[AnalysisResult]):
        """Handle processing completion with responsive UI updates."""
        self.is_processing = False
        self._update_ui_state()

        # Hide progress with smooth transition
        self.after(100, lambda: self.progress_frame.pack_forget())

        # Show completion feedback
        passed = sum(1 for r in results if r.overall_status.value == 'Pass')
        total = len(results)

        if total > 0:
            # Calculate additional stats for better feedback
            processing_time = sum(getattr(r, 'processing_time', 0) for r in results)
            avg_time = processing_time / total if total > 0 else 0
            
            # Add comprehensive success alert with responsive design
            self.alert_stack.add_alert(
                alert_type='success',
                title='Analysis Complete ✅',
                message=f'Processed {total} files in {processing_time:.1f}s (avg: {avg_time:.1f}s/file)\nPass rate: {passed / total * 100:.1f}% ({passed}/{total})',
                dismissible=True,
                allow_scroll=True,  # Keep app responsive
                actions=[
                    {'text': 'View Report', 'command': lambda: self._show_results_responsive(results)},
                    {'text': 'Export All', 'command': lambda: self._export_all_results(results)},
                    {'text': 'New Analysis', 'command': lambda: self._prepare_new_analysis()}
                ]
            )

            # Update results display with responsive loading
            self._show_results_responsive(results)

            # Mark home page for refresh
            if 'home' in self.main_window.pages:
                self.main_window.pages['home'].mark_needs_refresh()
        else:
            self.alert_stack.add_alert(
                alert_type='warning',
                title='No Results',
                message='No files were successfully processed. Check file formats and try again.',
                auto_dismiss=8,
                allow_scroll=True
            )

    def _processing_error_responsive(self, error: str):
        """Handle processing error with responsive error reporting."""
        self.is_processing = False
        self._update_ui_state()
        self.progress_frame.pack_forget()

        # Show user-friendly error message
        user_error = "An error occurred during analysis. Please check your files and try again."
        
        self.alert_stack.add_alert(
            alert_type='error',
            title='Processing Failed',
            message=user_error,
            dismissible=True,
            allow_scroll=True,
            actions=[
                {'text': 'View Details', 'command': lambda: self._show_detailed_error("Processing", error)},
                {'text': 'Try Again', 'command': lambda: self._retry_analysis()},
                {'text': 'Get Help', 'command': lambda: self._show_help_dialog()}
            ]
        )

    def _show_results_responsive(self, results: List[AnalysisResult]):
        """Display analysis results with responsive loading."""
        # Clear existing results first
        for frame in [self.summary_frame, self.ml_frame, self.details_frame]:
            for widget in frame.winfo_children():
                widget.destroy()

        # Show loading indicator
        loading_label = ttk.Label(
            self.summary_frame,
            text="Loading results...",
            font=('Segoe UI', 11),
            foreground=self.colors['text_secondary']
        )
        loading_label.pack(expand=True)

        # Schedule results creation in chunks for responsiveness
        def create_results_async():
            try:
                # Remove loading indicator
                loading_label.destroy()
                
                # Create results views
                self._create_summary_view(self.summary_frame, results)
                
                # Allow UI to breathe
                self.after(50, lambda: self._create_ml_insights_responsive(results))
                self.after(100, lambda: self._create_details_view_responsive(results))
                
                # Switch to summary tab
                self.results_notebook.select(0)
                
            except Exception as e:
                self.logger.error(f"Error creating results view: {e}")
                loading_label.config(text=f"Error loading results: {str(e)}")

        # Schedule async results creation
        self.after(100, create_results_async)

    def _create_ml_insights_responsive(self, results: List[AnalysisResult]):
        """Create ML insights view with responsive loading."""
        try:
            # Clear existing content
            for widget in self.ml_frame.winfo_children():
                widget.destroy()
                
            # Show ML insights if available
            if any(hasattr(r, 'ml_predictions') for r in results):
                self._create_ml_insights_view(self.ml_frame, results)
            else:
                ttk.Label(
                    self.ml_frame,
                    text="No ML predictions available.",
                    font=('Segoe UI', 11),
                    foreground=self.colors['text_secondary']
                ).pack(expand=True)
                
        except Exception as e:
            self.logger.error(f"Error creating ML insights: {e}")

    def _create_details_view_responsive(self, results: List[AnalysisResult]):
        """Create details view with responsive loading."""
        try:
            # Clear existing content
            for widget in self.details_frame.winfo_children():
                widget.destroy()
                
            # Create details view
            self._create_details_view(self.details_frame, results)
            
        except Exception as e:
            self.logger.error(f"Error creating details view: {e}")

    def _prepare_new_analysis(self):
        """Prepare for a new analysis with responsive cleanup and state preservation."""
        try:
            # Clear current results
            self._show_empty_results()
            
            # Clear alerts except critical ones
            self._clear_non_critical_alerts()
            
            # Reset file statuses to ready while preserving file selection
            for file_path, widget_data in self.file_widgets.items():
                # Update cached metadata
                if file_path in self._file_metadata_cache:
                    self._file_metadata_cache[file_path]['status'] = 'Ready'
                    self._file_metadata_cache[file_path]['last_reset'] = time_module.time()
                
                if isinstance(widget_data, dict) and widget_data.get('tree_mode'):
                    item_id = widget_data['tree_item']
                    if hasattr(self, 'file_tree') and self.file_tree.exists(item_id):
                        current_values = list(self.file_tree.item(item_id, 'values'))
                        current_values[2] = 'Ready'  # Reset status
                        self.file_tree.item(item_id, values=current_values)
                        self.file_tree.item(item_id, tags=('ready',))
                else:
                    if hasattr(widget_data, 'update_data'):
                        widget_data.update_data({'status': 'Ready'})
            
            # Clear processing results but keep file metadata
            self._processing_results.clear()
            
            # Show ready message
            self.alert_stack.add_alert(
                alert_type='info',
                title='Ready for New Analysis',
                message='Files have been reset and are ready for analysis.',
                auto_dismiss=3,
                allow_scroll=True
            )
            
        except Exception as e:
            self.logger.error(f"Error preparing new analysis: {e}")

    def _retry_analysis(self):
        """Retry the analysis with current settings."""
        try:
            if self.input_files:
                self._start_analysis()
            else:
                self.alert_stack.add_alert(
                    alert_type='warning',
                    title='No Files Selected',
                    message='Please select files before starting analysis.',
                    auto_dismiss=5
                )
        except Exception as e:
            self.logger.error(f"Error retrying analysis: {e}")

    def _show_help_dialog(self):
        """Show help dialog for troubleshooting."""
        try:
            dialog = tk.Toplevel(self.winfo_toplevel())
            dialog.title("Analysis Help")
            dialog.geometry("500x400")
            dialog.configure(bg=self.colors['bg'])
            
            # Help content
            help_text = """
Analysis Troubleshooting Guide:

1. File Format Issues:
   • Ensure files are valid Excel (.xlsx, .xls) format
   • Check that files contain the expected data structure
   • Verify file permissions allow reading

2. Processing Errors:
   • Check available disk space for output files
   • Ensure database connection (if enabled)
   • Verify ML predictor is loaded correctly

3. Performance Issues:
   • Reduce batch size for large file sets
   • Disable plot generation for faster processing
   • Close other applications to free memory

4. Common Solutions:
   • Restart the application
   • Check log files for detailed error information
   • Update file paths if files have moved

Contact support if issues persist.
            """
            
            text_widget = tk.Text(dialog, wrap='word', font=('Segoe UI', 10), padx=20, pady=20)
            text_widget.insert('1.0', help_text.strip())
            text_widget.config(state='disabled')
            text_widget.pack(fill='both', expand=True)
            
            # Close button
            ttk.Button(
                dialog,
                text="Close",
                command=dialog.destroy
            ).pack(pady=(0, 20))
            
        except Exception as e:
            self.logger.error(f"Error showing help dialog: {e}")

    def _cancel_analysis(self):
        """Cancel ongoing analysis with responsive feedback."""
        if self.current_task:
            # TODO: Implement proper cancellation
            pass

        self.is_processing = False
        self._update_ui_state()
        
        # Hide progress smoothly
        self.after(100, lambda: self.progress_frame.pack_forget())

        # Show responsive cancellation feedback
        self.alert_stack.add_alert(
            alert_type='warning',
            title='Analysis Cancelled',
            message='Processing was cancelled by user. Files remain loaded for retry.',
            auto_dismiss=5,
            allow_scroll=True,
            actions=[
                {'text': 'Restart Analysis', 'command': lambda: self._start_analysis()},
                {'text': 'Clear Files', 'command': lambda: self._clear_files()}
            ]
        )

    def _dismiss_large_batch_alert(self):
        """Dismiss the large batch mode alert."""
        try:
            # Find and dismiss any alerts with "Large Batch Mode" in the title
            # Since we simplified the alerts, we'll just clear all info alerts
            self.alert_stack.clear_all()
        except Exception as e:
            self.logger.error(f"Error dismissing large batch alert: {e}")

class SimpleAlertStack(ttk.Frame):
    """Simplified alert stack without complex animations that cause glitching."""

    def __init__(self, parent, max_alerts: int = 3, **kwargs):
        """Initialize simplified alert stack."""
        super().__init__(parent, **kwargs)
        
        self.max_alerts = max_alerts
        self.alerts: List[tk.Frame] = []
        self.configure(relief='flat', borderwidth=0)

    def add_alert(self, alert_type: str = 'info',
                  title: str = "", message: str = "",
                  dismissible: bool = True,
                  auto_dismiss: Optional[int] = None,
                  actions: Optional[List[dict]] = None,
                  allow_scroll: bool = True) -> Optional[tk.Frame]:
        """Add a simple alert without complex animations."""
        try:
            # Remove oldest alert if at max capacity
            if len(self.alerts) >= self.max_alerts:
                oldest_alert = self.alerts[0]
                self._remove_alert(oldest_alert)

            # Create simple alert frame
            alert = self._create_simple_alert(
                alert_type, title, message, dismissible, auto_dismiss, actions
            )
            
            if alert:
                self.alerts.append(alert)
                alert.pack(fill='x', pady=(0, 5))
                
                # Auto-dismiss if requested
                if auto_dismiss:
                    self.after(auto_dismiss * 1000, lambda: self._remove_alert(alert))
            
            return alert
            
        except Exception as e:
            print(f"Error adding alert: {e}")
            return None

    def _create_simple_alert(self, alert_type: str, title: str, message: str, 
                           dismissible: bool, auto_dismiss: Optional[int], 
                           actions: Optional[List[dict]]) -> Optional[tk.Frame]:
        """Create a simple alert frame without animations."""
        try:
            # Color scheme
            colors = {
                'info': {'bg': '#d1ecf1', 'fg': '#0c5460', 'border': '#bee5eb'},
                'warning': {'bg': '#fff3cd', 'fg': '#856404', 'border': '#ffeaa7'},
                'error': {'bg': '#f8d7da', 'fg': '#721c24', 'border': '#f5c6cb'},
                'success': {'bg': '#d4edda', 'fg': '#155724', 'border': '#c3e6cb'}
            }
            
            color = colors.get(alert_type, colors['info'])
            
            # Main alert frame
            alert_frame = tk.Frame(
                self, 
                bg=color['bg'], 
                relief='solid', 
                bd=1,
                highlightbackground=color['border'],
                highlightthickness=1
            )
            
            # Content frame
            content_frame = tk.Frame(alert_frame, bg=color['bg'])
            content_frame.pack(fill='both', expand=True, padx=15, pady=10)
            
            # Left side - text
            text_frame = tk.Frame(content_frame, bg=color['bg'])
            text_frame.pack(side='left', fill='both', expand=True)
            
            if title:
                title_label = tk.Label(
                    text_frame, 
                    text=title,
                    font=('Segoe UI', 10, 'bold'),
                    bg=color['bg'],
                    fg=color['fg']
                )
                title_label.pack(anchor='w')
            
            if message:
                msg_label = tk.Label(
                    text_frame,
                    text=message,
                    font=('Segoe UI', 9),
                    bg=color['bg'],
                    fg=color['fg'],
                    wraplength=400,
                    justify='left'
                )
                msg_label.pack(anchor='w', pady=(2, 0))
            
            # Right side - actions and dismiss
            right_frame = tk.Frame(content_frame, bg=color['bg'])
            right_frame.pack(side='right')
            
            # Action buttons
            if actions:
                for action in actions:
                    btn = tk.Button(
                        right_frame,
                        text=action['text'],
                        command=action['command'],
                        font=('Segoe UI', 8),
                        bg='white',
                        fg=color['fg'],
                        relief='flat',
                        padx=10, pady=4,
                        cursor='hand2'
                    )
                    btn.pack(side='left', padx=(0, 5))
            
            # Dismiss button
            if dismissible:
                dismiss_btn = tk.Label(
                    right_frame,
                    text='✕',
                    font=('Segoe UI', 12, 'bold'),
                    bg=color['bg'],
                    fg=color['fg'],
                    cursor='hand2',
                    padx=8, pady=4
                )
                dismiss_btn.pack(side='right', padx=(10, 0))
                dismiss_btn.bind('<Button-1>', lambda e: self._remove_alert(alert_frame))
            
            return alert_frame
            
        except Exception as e:
            print(f"Error creating alert: {e}")
            return None

    def _remove_alert(self, alert: tk.Frame):
        """Remove alert without animations."""
        try:
            if alert in self.alerts:
                self.alerts.remove(alert)
            alert.destroy()
        except Exception as e:
            print(f"Error removing alert: {e}")

    def clear_all(self):
        """Clear all alerts."""
        try:
            for alert in self.alerts.copy():
                self._remove_alert(alert)
        except Exception as e:
            print(f"Error clearing alerts: {e}")

    def dismiss_alert(self):
        """Method for compatibility with old alert system."""
        pass

    def update_alert(self, message: str):
        """Method for compatibility with old alert system."""
        pass