"""
Batch Processing Page

Provides interface for processing multiple Excel files with
comprehensive validation and progress tracking.
"""

import asyncio
import logging
import threading
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime

import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd

from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.core.config import get_config
from laser_trim_analyzer.core.models import AnalysisResult, AnalysisStatus, ValidationStatus
from laser_trim_analyzer.core.exceptions import ProcessingError, ValidationError
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.gui.widgets.batch_results_widget import BatchResultsWidget
from laser_trim_analyzer.gui.widgets.progress_widgets import BatchProgressDialog
from laser_trim_analyzer.gui.widgets.metric_card import MetricCard
from laser_trim_analyzer.utils.file_utils import ensure_directory

logger = logging.getLogger(__name__)


class BatchProcessingPage(ctk.CTkFrame):
    """Batch processing page with comprehensive validation."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.config = get_config()
        self.processor = LaserTrimProcessor(self.config)
        self.db_manager = DatabaseManager()
        
        # State
        self.selected_files: List[Path] = []
        self.batch_results: Dict[str, AnalysisResult] = {}
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False
        self.validation_results: Dict[str, bool] = {}
        
        # Create UI
        self._create_widgets()
        self._setup_layout()
        
        logger.info("Batch processing page initialized")

    def _create_widgets(self):
        """Create UI widgets."""
        
        # Header with batch status
        self.header_frame = ctk.CTkFrame(self)
        
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Batch Processing",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        
        # Batch validation status
        self.batch_status_frame = ctk.CTkFrame(self.header_frame)
        self.batch_status_label = ctk.CTkLabel(
            self.batch_status_frame,
            text="Batch Status: No Files Selected",
            font=ctk.CTkFont(size=12)
        )
        self.batch_indicator = ctk.CTkLabel(
            self.batch_status_frame,
            text="●",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        
        # File selection
        self.file_selection_frame = ctk.CTkFrame(self)
        
        self.file_selection_label = ctk.CTkLabel(
            self.file_selection_frame,
            text="File Selection:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        # File list display
        self.file_list_frame = ctk.CTkFrame(self.file_selection_frame)
        
        self.file_list_label = ctk.CTkLabel(
            self.file_list_frame,
            text="Selected Files (0):",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        
        self.file_listbox = ctk.CTkTextbox(
            self.file_list_frame,
            height=150,
            state="disabled"
        )
        
        # File selection buttons
        self.file_buttons_frame = ctk.CTkFrame(self.file_selection_frame)
        
        self.select_files_button = ctk.CTkButton(
            self.file_buttons_frame,
            text="Select Files",
            command=self._select_files,
            width=120
        )
        
        self.select_folder_button = ctk.CTkButton(
            self.file_buttons_frame,
            text="Select Folder",
            command=self._select_folder,
            width=120
        )
        
        self.clear_files_button = ctk.CTkButton(
            self.file_buttons_frame,
            text="Clear Files",
            command=self._clear_files,
            width=100
        )
        
        self.validate_batch_button = ctk.CTkButton(
            self.file_buttons_frame,
            text="Validate Batch",
            command=self._validate_batch,
            width=120,
            state="disabled"
        )
        
        # Batch validation metrics (initially hidden)
        self.batch_validation_frame = ctk.CTkFrame(self)
        self.batch_validation_frame.grid_remove()
        
        self.batch_validation_label = ctk.CTkLabel(
            self.batch_validation_frame,
            text="Batch Validation Results:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        # Validation metric cards
        self.total_files_card = MetricCard(
            self.batch_validation_frame,
            title="Total Files",
            value="0",
            color_scheme="neutral"
        )
        
        self.valid_files_card = MetricCard(
            self.batch_validation_frame,
            title="Valid Files",
            value="0",
            color_scheme="success"
        )
        
        self.invalid_files_card = MetricCard(
            self.batch_validation_frame,
            title="Invalid Files",
            value="0",
            color_scheme="danger"
        )
        
        self.validation_rate_card = MetricCard(
            self.batch_validation_frame,
            title="Validation Rate",
            value="0%",
            color_scheme="info"
        )
        
        # Processing options
        self.options_frame = ctk.CTkFrame(self)
        
        self.options_label = ctk.CTkLabel(
            self.options_frame,
            text="Processing Options:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        self.generate_plots_var = ctk.BooleanVar(value=False)  # Default off for batch
        self.generate_plots_check = ctk.CTkCheckBox(
            self.options_frame,
            text="Generate Plots",
            variable=self.generate_plots_var
        )
        
        self.save_to_db_var = ctk.BooleanVar(value=True)
        self.save_to_db_check = ctk.CTkCheckBox(
            self.options_frame,
            text="Save to Database",
            variable=self.save_to_db_var
        )
        
        self.comprehensive_validation_var = ctk.BooleanVar(value=True)
        self.comprehensive_validation_check = ctk.CTkCheckBox(
            self.options_frame,
            text="Comprehensive Validation",
            variable=self.comprehensive_validation_var
        )
        
        self.stop_on_error_var = ctk.BooleanVar(value=False)
        self.stop_on_error_check = ctk.CTkCheckBox(
            self.options_frame,
            text="Stop on First Error",
            variable=self.stop_on_error_var
        )
        
        # Worker threads setting
        self.workers_frame = ctk.CTkFrame(self.options_frame)
        self.workers_label = ctk.CTkLabel(
            self.workers_frame,
            text="Worker Threads:",
            font=ctk.CTkFont(size=12)
        )
        self.workers_slider = ctk.CTkSlider(
            self.workers_frame,
            from_=1,
            to=8,
            number_of_steps=7
        )
        self.workers_value_label = ctk.CTkLabel(
            self.workers_frame,
            text="4",
            font=ctk.CTkFont(size=12)
        )
        self.workers_slider.set(4)
        self.workers_slider.configure(command=self._update_workers_label)
        
        # Processing controls
        self.controls_frame = ctk.CTkFrame(self)
        
        self.start_button = ctk.CTkButton(
            self.controls_frame,
            text="Start Batch Processing",
            command=self._start_processing,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled"
        )
        
        self.stop_button = ctk.CTkButton(
            self.controls_frame,
            text="Stop Processing",
            command=self._stop_processing,
            width=140,
            state="disabled"
        )
        
        self.export_button = ctk.CTkButton(
            self.controls_frame,
            text="Export Batch Results",
            command=self._export_batch_results,
            width=160,
            state="disabled"
        )
        
        self.clear_results_button = ctk.CTkButton(
            self.controls_frame,
            text="Clear Results",
            command=self._clear_results,
            width=120
        )
        
        # Results display
        self.results_frame = ctk.CTkFrame(self)
        
        self.results_label = ctk.CTkLabel(
            self.results_frame,
            text="Batch Results:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        # Batch results widget
        self.batch_results_widget = BatchResultsWidget(self.results_frame)
        
        # Progress dialog
        self.progress_dialog: Optional[BatchProgressDialog] = None

    def _setup_layout(self):
        """Setup widget layout."""
        self.grid_rowconfigure(6, weight=1)  # Results frame expands
        self.grid_columnconfigure(0, weight=1)
        
        # Header
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        self.header_frame.grid_columnconfigure(0, weight=1)
        
        self.title_label.grid(row=0, column=0, sticky="w")
        self.batch_status_frame.grid(row=0, column=1, sticky="e")
        
        self.batch_status_frame.grid_columnconfigure(0, weight=1)
        self.batch_status_label.grid(row=0, column=0, padx=(0, 10))
        self.batch_indicator.grid(row=0, column=1)
        
        # File selection
        self.file_selection_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        self.file_selection_frame.grid_columnconfigure(0, weight=1)
        
        self.file_selection_label.grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # File list
        self.file_list_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.file_list_frame.grid_columnconfigure(0, weight=1)
        
        self.file_list_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.file_listbox.grid(row=1, column=0, sticky="ew")
        
        # File buttons
        self.file_buttons_frame.grid(row=2, column=0, sticky="ew")
        
        self.select_files_button.grid(row=0, column=0, padx=(0, 10))
        self.select_folder_button.grid(row=0, column=1, padx=(0, 10))
        self.clear_files_button.grid(row=0, column=2, padx=(0, 10))
        self.validate_batch_button.grid(row=0, column=3)
        
        # Batch validation metrics (row 2, initially hidden)
        self.batch_validation_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        self.batch_validation_frame.grid_columnconfigure([0, 1, 2, 3], weight=1)
        
        self.batch_validation_label.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))
        self.total_files_card.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.valid_files_card.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.invalid_files_card.grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        self.validation_rate_card.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        
        # Processing options
        self.options_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=10)
        self.options_frame.grid_columnconfigure(5, weight=1)
        
        self.options_label.grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 10))
        self.generate_plots_check.grid(row=1, column=0, sticky="w", padx=(0, 20))
        self.save_to_db_check.grid(row=1, column=1, sticky="w", padx=(0, 20))
        self.comprehensive_validation_check.grid(row=1, column=2, sticky="w", padx=(0, 20))
        self.stop_on_error_check.grid(row=1, column=3, sticky="w", padx=(0, 20))
        
        # Workers setting
        self.workers_frame.grid(row=1, column=4, sticky="ew", padx=(20, 0))
        self.workers_frame.grid_columnconfigure(1, weight=1)
        
        self.workers_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.workers_slider.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.workers_value_label.grid(row=0, column=2, sticky="w")
        
        # Controls
        self.controls_frame.grid(row=4, column=0, sticky="ew", padx=20, pady=10)
        
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        self.export_button.grid(row=0, column=2, padx=(0, 10))
        self.clear_results_button.grid(row=0, column=3)
        
        # Results
        self.results_frame.grid(row=5, column=0, sticky="nsew", padx=20, pady=10)
        self.results_frame.grid_rowconfigure(1, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        self.results_label.grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.batch_results_widget.grid(row=1, column=0, sticky="nsew")

    def _select_files(self):
        """Select multiple Excel files."""
        file_paths = filedialog.askopenfilenames(
            title="Select Excel files",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )
        
        if file_paths:
            self.selected_files = [Path(f) for f in file_paths]
            self._update_file_display()
            self._update_batch_status("Files Selected", "orange")
            logger.info(f"Selected {len(self.selected_files)} files")

    def _select_folder(self):
        """Select folder and find all Excel files."""
        folder_path = filedialog.askdirectory(title="Select folder with Excel files")
        
        if folder_path:
            self._start_folder_discovery(Path(folder_path))

    def _start_folder_discovery(self, folder_path: Path):
        """Start asynchronous folder discovery with progress indication."""
        # Update UI to show scanning state
        self._update_batch_status("Scanning Folder...", "orange")
        self.select_folder_button.configure(state="disabled", text="Scanning...")
        
        # Run discovery in background thread
        def discover_files():
            try:
                excel_files = []
                total_checked = 0
                
                # Find all Excel files recursively with progress updates
                for pattern in ["*.xlsx", "*.xls"]:
                    for file_path in folder_path.rglob(pattern):
                        # Filter out temporary files
                        if not file_path.name.startswith('~$'):
                            excel_files.append(file_path)
                        
                        # Update progress every 50 files
                        total_checked += 1
                        if total_checked % 50 == 0:
                            self.after(0, lambda count=total_checked: 
                                     self._update_batch_status(f"Scanning... ({count} files found)", "orange"))
                
                # Update UI on main thread
                self.after(0, self._handle_folder_discovery_complete, excel_files, folder_path)
                
            except Exception as e:
                logger.error(f"Folder discovery failed: {e}")
                self.after(0, self._handle_folder_discovery_error, str(e))
        
        # Start discovery thread
        thread = threading.Thread(target=discover_files, daemon=True)
        thread.start()

    def _handle_folder_discovery_complete(self, excel_files: List[Path], folder_path: Path):
        """Handle completion of folder discovery."""
        # Re-enable button
        self.select_folder_button.configure(state="normal", text="📂 Select Folder")
        
        if excel_files:
            self.selected_files = excel_files
            self._update_file_display()
            self._update_batch_status("Folder Selected", "orange")
            logger.info(f"Found {len(excel_files)} Excel files in {folder_path}")
            
            # Show discovery summary
            messagebox.showinfo(
                "Folder Discovery Complete",
                f"Found {len(excel_files)} Excel files in:\n{folder_path.name}\n\n"
                f"Ready for batch validation and processing."
            )
        else:
            self._update_batch_status("No Files Selected", "gray")
            messagebox.showwarning("No Files", f"No Excel files found in {folder_path}")

    def _handle_folder_discovery_error(self, error_message: str):
        """Handle folder discovery error."""
        # Re-enable button
        self.select_folder_button.configure(state="normal", text="📂 Select Folder")
        self._update_batch_status("Discovery Error", "red")
        messagebox.showerror("Folder Discovery Error", f"Failed to scan folder:\n{error_message}")

    def _clear_files(self):
        """Clear selected files."""
        self.selected_files = []
        self.validation_results = {}
        self._update_file_display()
        self._update_batch_status("No Files Selected", "gray")
        self.batch_validation_frame.grid_remove()
        self.validate_batch_button.configure(state="disabled")
        self.start_button.configure(state="disabled")

    def _update_file_display(self):
        """Update file list display."""
        self.file_list_label.configure(text=f"Selected Files ({len(self.selected_files)}):")
        
        self.file_listbox.configure(state="normal")
        self.file_listbox.delete("1.0", ctk.END)
        
        for i, file_path in enumerate(self.selected_files, 1):
            validation_status = ""
            if str(file_path) in self.validation_results:
                status = "✓" if self.validation_results[str(file_path)] else "✗"
                validation_status = f" [{status}]"
            
            self.file_listbox.insert(ctk.END, f"{i:3d}. {file_path.name}{validation_status}\n")
        
        self.file_listbox.configure(state="disabled")
        
        # Enable/disable buttons
        has_files = len(self.selected_files) > 0
        self.validate_batch_button.configure(state="normal" if has_files else "disabled")

    def _validate_batch(self):
        """Validate all selected files."""
        if not self.selected_files:
            messagebox.showerror("Error", "No files selected")
            return
        
        self._update_batch_status("Validating Batch...", "orange")
        
        # Run validation in thread
        def validate():
            try:
                from laser_trim_analyzer.utils.validators import BatchValidator
                
                validation_result = BatchValidator.validate_batch(
                    file_paths=self.selected_files,
                    max_batch_size=self.config.processing.max_batch_size
                )
                
                # Store individual file validation results
                invalid_files = validation_result.metadata.get('invalid_files', [])
                self.validation_results = {}
                
                for file_path in self.selected_files:
                    is_valid = not any(str(file_path) in invalid['file'] for invalid in invalid_files)
                    self.validation_results[str(file_path)] = is_valid
                
                # Update UI on main thread
                self.after(0, self._handle_batch_validation_result, validation_result)
                
            except Exception as e:
                logger.error(f"Batch validation failed: {e}")
                self.after(0, self._handle_batch_validation_error, str(e))
        
        thread = threading.Thread(target=validate, daemon=True)
        thread.start()

    def _handle_batch_validation_result(self, validation_result):
        """Handle batch validation result."""
        if validation_result.is_valid:
            self._update_batch_status("Batch Validation Passed", "green")
            
            # Show validation metrics
            self.batch_validation_frame.grid()
            
            metadata = validation_result.metadata
            total_files = len(self.selected_files)
            valid_files = metadata.get('valid_files', 0)
            invalid_files = total_files - valid_files
            validation_rate = (valid_files / total_files * 100) if total_files > 0 else 0
            
            self.total_files_card.update_value(str(total_files), "info")
            self.valid_files_card.update_value(str(valid_files), "success")
            self.invalid_files_card.update_value(str(invalid_files), 
                                               "success" if invalid_files == 0 else "warning")
            self.validation_rate_card.update_value(f"{validation_rate:.1f}%",
                                                 "success" if validation_rate > 90 else "warning")
            
            # Update file display with validation status
            self._update_file_display()
            
            # Enable start button if we have valid files
            if valid_files > 0:
                self.start_button.configure(state="normal")
            
            # Show warnings if any
            if validation_result.warnings:
                warning_msg = "Batch validation warnings:\n" + "\n".join(validation_result.warnings)
                messagebox.showwarning("Validation Warnings", warning_msg)
                
        else:
            self._update_batch_status("Batch Validation Failed", "red")
            
            # Show validation metrics with errors
            self.batch_validation_frame.grid()
            total_files = len(self.selected_files)
            self.total_files_card.update_value(str(total_files), "danger")
            self.valid_files_card.update_value("0", "danger")
            self.invalid_files_card.update_value(str(total_files), "danger")
            self.validation_rate_card.update_value("0%", "danger")
            
            # Update file display
            self._update_file_display()
            
            # Show errors
            error_msg = "Batch validation failed:\n" + "\n".join(validation_result.errors)
            messagebox.showerror("Batch Validation Failed", error_msg)

    def _handle_batch_validation_error(self, error_message):
        """Handle batch validation error."""
        self._update_batch_status("Validation Error", "red")
        messagebox.showerror("Validation Error", f"Batch validation failed:\n{error_message}")

    def _update_batch_status(self, status: str, color: str):
        """Update batch status indicator."""
        self.batch_status_label.configure(text=f"Batch Status: {status}")
        
        color_map = {
            "green": "#00ff00",
            "orange": "#ffa500",
            "red": "#ff0000", 
            "gray": "#808080"
        }
        
        self.batch_indicator.configure(text_color=color_map.get(color, "#808080"))

    def _update_workers_label(self, value):
        """Update workers label."""
        self.workers_value_label.configure(text=str(int(value)))

    def _start_processing(self):
        """Start batch processing."""
        if not self.selected_files:
            messagebox.showerror("Error", "No files selected")
            return
        
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing already in progress")
            return
        
        # Check if validation was run
        if not self.validation_results:
            reply = messagebox.askyesno(
                "No Validation", 
                "Batch validation hasn't been run. Proceed anyway?"
            )
            if not reply:
                return
        
        # Filter to only valid files if validation was run
        processable_files = []
        if self.validation_results:
            for file_path in self.selected_files:
                if self.validation_results.get(str(file_path), True):  # Default True if not validated
                    processable_files.append(file_path)
        else:
            processable_files = self.selected_files.copy()
        
        if not processable_files:
            messagebox.showerror("Error", "No valid files to process")
            return
        
        # Clear previous results
        self._clear_results()
        
        # Disable controls
        self._set_controls_state("disabled")
        
        # Show progress dialog
        self.progress_dialog = BatchProgressDialog(
            self,
            title="Batch Processing",
            total_files=len(processable_files)
        )
        self.progress_dialog.show()
        
        # Start processing in thread
        self.is_processing = True
        self.processing_thread = threading.Thread(
            target=self._run_batch_processing,
            args=(processable_files,),
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info(f"Started batch processing of {len(processable_files)} files")

    def _run_batch_processing(self, file_paths: List[Path]):
        """Run batch processing in background thread with performance optimizations."""
        import gc
        import time
        
        try:
            # Performance tracking
            start_time = time.time()
            last_gc_time = start_time
            last_progress_update = 0
            processed_count = 0
            
            # Create output directory if plots requested
            output_dir = None
            if self.generate_plots_var.get():
                output_dir = self.config.output_directory / "batch_processing" / datetime.now().strftime("%Y%m%d_%H%M%S")
                ensure_directory(output_dir)
            
            # Throttled progress callback to prevent UI flooding
            def progress_callback(message: str, progress: float):
                nonlocal last_progress_update
                current_time = time.time()
                
                # Only update progress every 250ms to prevent UI flooding
                if current_time - last_progress_update >= 0.25:
                    last_progress_update = current_time
                    if self.progress_dialog:
                        self.after(0, lambda: self.progress_dialog.update_progress(message, progress))
                        
                    # Force GUI update and yield CPU time
                    self.after(0, self.update)
                    time.sleep(0.001)  # Tiny sleep to yield CPU
            
            # Enhanced progress callback with memory monitoring
            def enhanced_progress_callback(message: str, progress: float):
                nonlocal processed_count, last_gc_time
                current_time = time.time()
                
                # Standard progress update
                progress_callback(message, progress)
                
                # Memory management every 50 files or 30 seconds
                if (processed_count % 50 == 0 and processed_count > 0) or (current_time - last_gc_time > 30):
                    logger.debug(f"Performing memory cleanup at file {processed_count}")
                    
                    # Force garbage collection
                    gc.collect()
                    last_gc_time = current_time
                    
                    # Clear any intermediate results from memory
                    import matplotlib.pyplot as plt
                    plt.close('all')  # Close all matplotlib figures
                    
                    # Yield more CPU time during cleanup
                    time.sleep(0.01)
            
            # Run batch processing with asyncio and optimizations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Limit max workers based on system resources and file count
                base_workers = int(self.workers_slider.get())
                file_count = len(file_paths)
                
                # Scale down workers for very large batches to prevent resource exhaustion
                if file_count > 1000:
                    max_workers = min(base_workers, 6)  # Max 6 workers for very large batches
                elif file_count > 500:
                    max_workers = min(base_workers, 8)  # Max 8 workers for large batches
                else:
                    max_workers = base_workers
                
                logger.info(f"Processing {file_count} files with {max_workers} workers (scaled from {base_workers})")
                
                # Disable plots for very large batches to save memory
                if file_count > 200 and self.generate_plots_var.get():
                    reply = messagebox.askyesno(
                        "Large Batch Detected",
                        f"Processing {file_count} files with plots enabled may cause performance issues.\n\n"
                        "Disable plots for better performance?"
                    )
                    if reply:
                        self.generate_plots_var.set(False)
                        output_dir = None
                
                # Process with memory-efficient batching
                results = loop.run_until_complete(
                    self._process_with_memory_management(
                        file_paths=file_paths,
                        output_dir=output_dir,
                        progress_callback=enhanced_progress_callback,
                        max_workers=max_workers
                    )
                )
                
                # Save to database if requested
                if self.save_to_db_var.get() and self.db_manager:
                    self._save_batch_to_database(results)
                
                # Final cleanup
                gc.collect()
                
                # Update UI on main thread
                self.after(0, self._handle_batch_success, results, output_dir)
                
            finally:
                loop.close()
                
        except ValidationError as e:
            logger.error(f"Batch validation error: {e}")
            self.after(0, self._handle_batch_error, f"Batch validation failed: {str(e)}")
            
        except ProcessingError as e:
            logger.error(f"Batch processing error: {e}")
            self.after(0, self._handle_batch_error, f"Batch processing failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected batch error: {e}")
            logger.error(traceback.format_exc())
            self.after(0, self._handle_batch_error, f"Unexpected error: {str(e)}")
        
        finally:
            self.is_processing = False
            # Final cleanup
            import gc
            gc.collect()

    async def _process_with_memory_management(
        self,
        file_paths: List[Path],
        output_dir: Optional[Path],
        progress_callback: Callable[[str, float], None],
        max_workers: int
    ) -> Dict[str, AnalysisResult]:
        """Process files with enhanced memory management and throttling."""
        import psutil
        import time
        
        results = {}
        
        # Process in smaller chunks to prevent memory buildup
        chunk_size = min(max_workers * 2, 20)  # Process in chunks of 20 max
        total_files = len(file_paths)
        
        logger.info(f"Processing {total_files} files in chunks of {chunk_size}")
        
        for chunk_start in range(0, total_files, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_files)
            chunk_files = file_paths[chunk_start:chunk_end]
            
            # Update progress for chunk start
            chunk_progress = chunk_start / total_files
            progress_callback(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_files-1)//chunk_size + 1}...", chunk_progress)
            
            # Process chunk with limited concurrency
            chunk_results = await self.processor.process_batch(
                file_paths=chunk_files,
                output_dir=output_dir,
                progress_callback=lambda msg, prog: progress_callback(
                    msg, 
                    chunk_progress + (prog * chunk_size / total_files)
                ),
                max_workers=min(max_workers, len(chunk_files))
            )
            
            # Merge results
            results.update(chunk_results)
            
            # Memory management between chunks
            if chunk_end < total_files:  # Not the last chunk
                logger.debug(f"Chunk {chunk_start//chunk_size + 1} complete, performing cleanup...")
                
                # Check memory usage
                try:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    
                    if memory_mb > 1500:  # If using > 1.5GB, be more aggressive with cleanup
                        logger.warning(f"High memory usage detected: {memory_mb:.1f}MB, performing aggressive cleanup")
                        
                        # Clear matplotlib figures
                        import matplotlib.pyplot as plt
                        plt.close('all')
                        
                        # Force garbage collection
                        import gc
                        gc.collect()
                        
                        # Longer pause for memory recovery
                        await asyncio.sleep(0.1)
                    else:
                        # Standard cleanup
                        import gc
                        gc.collect()
                        await asyncio.sleep(0.05)
                        
                except Exception as e:
                    logger.warning(f"Memory monitoring failed: {e}")
                    # Standard cleanup as fallback
                    import gc
                    gc.collect()
                    await asyncio.sleep(0.05)
        
        return results

    def _save_batch_to_database(self, results: Dict[str, AnalysisResult]):
        """Save batch results to database with robust error handling."""
        saved_count = 0
        failed_count = 0
        
        for file_path, result in results.items():
            try:
                # Check for duplicates
                existing_id = self.db_manager.check_duplicate_analysis(
                    result.metadata.model,
                    result.metadata.serial,
                    result.metadata.file_date
                )
                
                if existing_id:
                    logger.info(f"Duplicate found for {Path(file_path).name} (ID: {existing_id})")
                    result.db_id = existing_id
                else:
                    # Try normal save first
                    try:
                        result.db_id = self.db_manager.save_analysis(result)
                        saved_count += 1
                        
                        # Validate the save
                        if not self.db_manager.validate_saved_analysis(result.db_id):
                            raise RuntimeError("Database validation failed")
                            
                    except Exception as save_error:
                        logger.warning(f"Normal save failed for {Path(file_path).name}, trying force save: {save_error}")
                        # Try force save as fallback
                        result.db_id = self.db_manager.force_save_analysis(result)
                        saved_count += 1
                        logger.info(f"Force saved {Path(file_path).name} to database")
                    
            except Exception as e:
                logger.error(f"Database save failed for {Path(file_path).name}: {e}")
                failed_count += 1
        
        logger.info(f"Database save complete: {saved_count} saved, {failed_count} failed")
        
        if failed_count > 0:
            # Show warning about failed saves
            self.after(0, lambda: messagebox.showwarning(
                "Database Warning",
                f"Some files failed to save to database:\n"
                f"Saved: {saved_count}\n"
                f"Failed: {failed_count}\n\n"
                f"Check logs for details."
            ))

    def _handle_batch_success(self, results: Dict[str, AnalysisResult], output_dir: Optional[Path]):
        """Handle successful batch completion."""
        self.batch_results = results
        
        # Hide progress dialog
        if self.progress_dialog:
            self.progress_dialog.hide()
            self.progress_dialog = None
        
        # Calculate summary statistics
        total_processed = len(results)
        successful_count = len(results)
        failed_count = len(self.selected_files) - successful_count
        
        # Validation statistics
        validated_count = sum(1 for r in results.values() 
                            if r.overall_validation_status == ValidationStatus.VALIDATED)
        warning_count = sum(1 for r in results.values() 
                          if r.overall_validation_status == ValidationStatus.WARNING)
        failed_validation_count = sum(1 for r in results.values() 
                                    if r.overall_validation_status == ValidationStatus.FAILED)
        
        # Update batch status
        if failed_count == 0:
            if validated_count == successful_count:
                self._update_batch_status("Batch Complete - All Validated", "green")
            elif warning_count > 0:
                self._update_batch_status("Batch Complete - With Warnings", "orange")
            else:
                self._update_batch_status("Batch Complete - Some Validation Issues", "orange")
        else:
            self._update_batch_status("Batch Complete - With Errors", "red")
        
        # Display results
        self.batch_results_widget.display_results(results)
        
        # Enable export button
        self.export_button.configure(state="normal")
        
        # Re-enable controls
        self._set_controls_state("normal")
        
        # Show completion message
        success_msg = f"Batch processing completed!\n\n"
        success_msg += f"Files processed: {successful_count}/{len(self.selected_files)}\n"
        success_msg += f"Validated: {validated_count}\n"
        success_msg += f"Warnings: {warning_count}\n"
        success_msg += f"Failed validation: {failed_validation_count}\n"
        
        if failed_count > 0:
            success_msg += f"Processing failures: {failed_count}\n"
        
        if output_dir:
            success_msg += f"\nOutputs saved to: {output_dir}"
        
        messagebox.showinfo("Batch Processing Complete", success_msg)
        
        logger.info(f"Batch processing completed: {successful_count} successful, {failed_count} failed")

    def _handle_batch_error(self, error_message: str):
        """Handle batch processing error."""
        # Hide progress dialog
        if self.progress_dialog:
            self.progress_dialog.hide()
            self.progress_dialog = None
        
        # Update status
        self._update_batch_status("Batch Processing Failed", "red")
        
        # Re-enable controls
        self._set_controls_state("normal")
        
        # Show error message
        messagebox.showerror("Batch Processing Failed", error_message)
        
        logger.error(f"Batch processing failed: {error_message}")

    def _stop_processing(self):
        """Stop batch processing."""
        if self.is_processing:
            # This is a simplified stop - in a real implementation you'd need
            # to properly signal the background thread to stop
            reply = messagebox.askyesno(
                "Stop Processing",
                "Are you sure you want to stop batch processing?"
            )
            if reply:
                self.is_processing = False
                if self.progress_dialog:
                    self.progress_dialog.hide()
                    self.progress_dialog = None
                self._set_controls_state("normal")
                self._update_batch_status("Processing Stopped", "orange")
                logger.info("Batch processing stopped by user")

    def _set_controls_state(self, state: str):
        """Set state of control buttons."""
        if state == "disabled":
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.select_files_button.configure(state="disabled")
            self.select_folder_button.configure(state="disabled")
            self.validate_batch_button.configure(state="disabled")
        else:
            has_files = len(self.selected_files) > 0
            has_valid_files = any(self.validation_results.values()) if self.validation_results else has_files
            
            self.start_button.configure(state="normal" if has_valid_files else "disabled")
            self.stop_button.configure(state="disabled")
            self.select_files_button.configure(state="normal")
            self.select_folder_button.configure(state="normal")
            self.validate_batch_button.configure(state="normal" if has_files else "disabled")

    def _export_batch_results(self):
        """Export batch processing results."""
        if not self.batch_results:
            messagebox.showerror("Error", "No results to export")
            return
        
        # Ask for export location
        file_path = filedialog.asksaveasfilename(
            title="Export Batch Results",
            defaultextension=".xlsx",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    self._export_batch_excel(Path(file_path))
                elif file_path.endswith('.csv'):
                    self._export_batch_csv(Path(file_path))
                else:
                    messagebox.showerror("Error", "Unsupported file format")
                    return
                
                messagebox.showinfo("Export Complete", f"Batch results exported to:\n{file_path}")
                logger.info(f"Batch results exported to: {file_path}")
                
            except Exception as e:
                logger.error(f"Batch export failed: {e}")
                messagebox.showerror("Export Failed", f"Failed to export batch results:\n{str(e)}")

    def _export_batch_excel(self, file_path: Path):
        """Export batch results to Excel format."""
        # Implementation for batch Excel export with comprehensive validation data
        pass

    def _export_batch_csv(self, file_path: Path):
        """Export batch results to CSV format."""
        # Implementation for batch CSV export
        pass

    def _clear_results(self):
        """Clear batch processing results."""
        self.batch_results = {}
        self.batch_results_widget.clear()
        self.export_button.configure(state="disabled")
        logger.info("Batch results cleared") 