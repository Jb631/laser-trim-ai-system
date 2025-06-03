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
from laser_trim_analyzer.gui.widgets.alert_banner import AlertBanner, AlertStack
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
        """Create analysis page content."""
        # Main container
        container = ttk.Frame(self, style='TFrame')
        container.pack(fill='both', expand=True, padx=20, pady=20)

        # Title and description
        self._create_header(container)

        # Alert stack for notifications
        self.alert_stack = AlertStack(container)
        self.alert_stack.pack(fill='x', pady=(0, 10))

        # Main content area with two columns
        content_frame = ttk.Frame(container)
        content_frame.pack(fill='both', expand=True)

        # Left column - File selection and options
        left_column = ttk.Frame(content_frame)
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Right column - Results
        right_column = ttk.Frame(content_frame)
        right_column.pack(side='right', fill='both', expand=True, padx=(10, 0))

        # Create sections
        self._create_file_selection_section(left_column)
        self._create_options_section(left_column)
        self._create_action_section(left_column)
        self._create_results_section(right_column)

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

    def _create_file_selection_section(self, parent):
        """Create file selection section."""
        # File selection frame
        file_frame = ttk.LabelFrame(
            parent,
            text="Select Files",
            padding=15
        )
        file_frame.pack(fill='both', expand=True, pady=(0, 10))

        # Drop zone
        self.drop_zone = FileDropZone(
            file_frame,
            on_files_dropped=self._handle_files_dropped,
            accept_extensions=['.xlsx', '.xls']
        )
        self.drop_zone.pack(fill='both', expand=True, pady=(0, 10))

        # Browse button
        ttk.Button(
            file_frame,
            text="Browse Files",
            command=self.browse_files
        ).pack()

        # Selected files list
        files_label_frame = ttk.LabelFrame(
            parent,
            text="Selected Files",
            padding=10
        )
        files_label_frame.pack(fill='both', expand=True, pady=(0, 10))

        # Scrollable frame for file widgets
        canvas = tk.Canvas(files_label_frame, height=200)
        scrollbar = ttk.Scrollbar(files_label_frame, orient='vertical', command=canvas.yview)
        self.file_list_frame = ttk.Frame(canvas)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas_frame = canvas.create_window((0, 0), window=self.file_list_frame, anchor='nw')

        # Add mouse wheel scrolling support
        add_mousewheel_support(self.file_list_frame, canvas)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Configure canvas scrolling
        def configure_scroll(e):
            canvas.configure(scrollregion=canvas.bbox('all'))

        self.file_list_frame.bind('<Configure>', configure_scroll)

    def _create_options_section(self, parent):
        """Create processing options section."""
        options_frame = ttk.LabelFrame(
            parent,
            text="Processing Options",
            padding=15
        )
        options_frame.pack(fill='x', pady=(0, 10))

        # Processing mode
        mode_frame = ttk.Frame(options_frame)
        mode_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(mode_frame, text="Processing Mode:").pack(side='left')

        ttk.Radiobutton(
            mode_frame,
            text="Detail (with plots)",
            variable=self.processing_mode,
            value='detail'
        ).pack(side='left', padx=(10, 20))

        ttk.Radiobutton(
            mode_frame,
            text="Speed (no plots)",
            variable=self.processing_mode,
            value='speed'
        ).pack(side='left')

        # Feature toggles
        features_frame = ttk.Frame(options_frame)
        features_frame.pack(fill='x')

        ttk.Checkbutton(
            features_frame,
            text="Generate plots",
            variable=self.enable_plots
        ).pack(side='left', padx=(0, 20))

        ttk.Checkbutton(
            features_frame,
            text="ML predictions",
            variable=self.enable_ml,
            state='normal' if self.main_window.ml_predictor else 'disabled'
        ).pack(side='left', padx=(0, 20))

        ttk.Checkbutton(
            features_frame,
            text="Save to database",
            variable=self.enable_database,
            state='normal' if self.main_window.db_manager else 'disabled'
        ).pack(side='left')

    def _create_action_section(self, parent):
        """Create action buttons section."""
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill='x', pady=(0, 10))

        self.analyze_button = ttk.Button(
            action_frame,
            text="Start Analysis",
            style='Primary.TButton',
            command=self._start_analysis,
            state='disabled'
        )
        self.analyze_button.pack(side='left', padx=(0, 10))

        self.cancel_button = ttk.Button(
            action_frame,
            text="Cancel",
            command=self._cancel_analysis,
            state='disabled'
        )
        self.cancel_button.pack(side='left', padx=(0, 10))

        ttk.Button(
            action_frame,
            text="Clear All",
            command=self._clear_files
        ).pack(side='left')

        # Progress bar
        self.progress_frame = ttk.Frame(parent)
        self.progress_frame.pack(fill='x', pady=(10, 0))
        # Don't pack initially

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill='x', pady=(0, 5))

        self.progress_label = ttk.Label(
            self.progress_frame,
            text="",
            font=('Segoe UI', 9)
        )
        self.progress_label.pack()

    def _create_results_section(self, parent):
        """Create results display section."""
        results_frame = ttk.LabelFrame(
            parent,
            text="Analysis Results",
            padding=15
        )
        results_frame.pack(fill='both', expand=True)

        # Notebook for different result views
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill='both', expand=True)

        # Summary tab
        self.summary_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.summary_frame, text="Summary")

        # ML Insights tab
        self.ml_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.ml_frame, text="ML Insights")

        # Details tab
        self.details_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.details_frame, text="Details")

        # Initialize with empty message
        self._show_empty_results()

    def _show_empty_results(self):
        """Show empty results message."""
        for frame in [self.summary_frame, self.ml_frame, self.details_frame]:
            for widget in frame.winfo_children():
                widget.destroy()

            ttk.Label(
                frame,
                text="No results yet. Select files and start analysis.",
                font=('Segoe UI', 11),
                foreground=self.colors['text_secondary']
            ).pack(expand=True)

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
                if len(valid_files) > 200:
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
        
        # Determine batch size and update frequency based on total files
        if total_files > 100:
            batch_size = 50
            update_frequency = 25
        else:
            batch_size = 20
            update_frequency = 10
            
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
                            try:
                                progress_alert_ref['alert'].update_alert(
                                    message=f'Loading {total_files} files... {progress:.0f}% ({current_count}/{total_files})'
                                )
                            except:
                                pass
                    
                    # Schedule next batch or completion
                    if i + batch_size < total_files:
                        delay = 5 if total_files > 200 else 10
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
        
        # Start the async creation
        self.after(5, create_widgets_batch)

    def _update_widget_creation_progress(self, progress: float, current_count: int, 
                                       total_count: int, alert_ref: dict):
        """Update progress of widget creation using a single alert."""
        if alert_ref['alert'] is None:
            # Create the progress alert if it doesn't exist
            alert_ref['alert'] = self.alert_stack.add_alert(
                alert_type='info',
                title='Loading Files',
                message=f'Creating interface... {progress:.0f}% ({current_count}/{total_count} files)',
                auto_dismiss=None,
                dismissible=False
            )
        else:
            # Update existing alert
            try:
                alert_ref['alert'].update_alert(
                    message=f'Creating interface... {progress:.0f}% ({current_count}/{total_count} files)'
                )
            except:
                # If update fails, recreate the alert
                alert_ref['alert'] = self.alert_stack.add_alert(
                    alert_type='info',
                    title='Loading Files',
                    message=f'Creating interface... {progress:.0f}% ({current_count}/{total_count} files)',
                    auto_dismiss=None,
                    dismissible=False
                )

    def _widget_creation_complete(self, alert_ref: dict):
        """Called when all widgets have been created."""
        # Dismiss the progress alert
        if alert_ref['alert']:
            try:
                alert_ref['alert'].dismiss()
            except:
                pass
        
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
            # Get current alerts and filter
            current_alerts = self.alert_stack.get_alerts()
            alerts_to_keep = []
            
            for alert in current_alerts:
                # Keep error and warning alerts, dismiss info alerts
                if hasattr(alert, 'alert_type') and alert.alert_type in ['error', 'warning']:
                    alerts_to_keep.append(alert)
                else:
                    try:
                        alert.dismiss()
                    except:
                        pass  # Silent dismissal failure
                        
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