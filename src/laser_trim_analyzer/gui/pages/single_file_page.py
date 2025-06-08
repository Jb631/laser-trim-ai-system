"""
Single File Analysis Page

Provides interface for analyzing individual Excel files with
comprehensive validation and real-time feedback.
"""

import asyncio
import logging
import threading
import traceback
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd

from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.core.config import get_config
from laser_trim_analyzer.core.models import AnalysisResult, AnalysisStatus, ValidationStatus
from laser_trim_analyzer.core.exceptions import ProcessingError, ValidationError
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.gui.widgets.analysis_display import AnalysisDisplayWidget
from laser_trim_analyzer.gui.widgets.progress_widgets import ProgressDialog
from laser_trim_analyzer.gui.widgets.metric_card import MetricCard
from laser_trim_analyzer.utils.plotting_utils import create_analysis_plot
from laser_trim_analyzer.utils.file_utils import ensure_directory
# from laser_trim_analyzer.gui.pages.base_page import BasePage  # Commented out to avoid widget mismatch
from laser_trim_analyzer.gui.widgets.progress_widget import ProgressWidget
from laser_trim_analyzer.gui.widgets.hover_fix import fix_hover_glitches, stabilize_layout

logger = logging.getLogger(__name__)


class SingleFilePage(ctk.CTkFrame):
    """Single file analysis page with comprehensive validation and responsive design."""

    def __init__(self, parent, main_window=None, **kwargs):
        # Initialize as CTkFrame to avoid widget hierarchy issues
        super().__init__(parent, **kwargs)
        self.main_window = main_window
        
        # Add missing BasePage functionality
        self.is_visible = False
        self.needs_refresh = True
        self._stop_requested = False
        
        self.analyzer_config = get_config()
        self.processor = LaserTrimProcessor(self.analyzer_config)
        
        # Note: Database manager will be accessed via main_window.db_manager when needed
        
        # State
        self.current_file: Optional[Path] = None
        self.current_result: Optional[AnalysisResult] = None
        self.analysis_thread: Optional[threading.Thread] = None
        self.is_analyzing = False
        self.progress_dialog = None
        self.watchdog_timer = None
        
        # Create the page content (now we need to call this explicitly)
        self._create_page()
        
        # Apply hover fixes after page creation
        self.after(100, self._apply_hover_fixes)
        
        logger.info("Single file analysis page initialized")
    
    def _apply_hover_fixes(self):
        """Apply hover fixes to prevent glitching and shifting."""
        try:
            # Fix hover glitches on all widgets
            fix_hover_glitches(self)
            
            # Stabilize layout to prevent shifting
            stabilize_layout(self.main_container)
            
            # Lock positions of key frames to prevent shifting
            if hasattr(self, 'file_frame'):
                self.file_frame.update_idletasks()
            if hasattr(self, 'options_frame'):
                self.options_frame.update_idletasks()
            if hasattr(self, 'controls_frame'):
                self.controls_frame.update_idletasks()
                
            logger.debug("Hover fixes applied successfully")
        except Exception as e:
            logger.warning(f"Failed to apply hover fixes: {e}")

    def _create_page(self):
        """Create page content with responsive design."""
        logger.warning(f"DEBUG: _create_page called for {self.__class__.__name__}")
        
        # Check if main_container already exists
        if hasattr(self, 'main_container'):
            logger.warning(f"DEBUG: main_container already exists! This indicates duplicate creation.")
            return
        
        # Main scrollable container (matching batch processing theme)
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)
        logger.warning(f"DEBUG: Created main_container successfully")
        
        # Create sections in order (matching batch processing pattern)
        self._create_header()
        self._create_file_selection()
        self._create_options_section()
        self._create_prevalidation_section()
        self._create_controls_section()
        self._create_results_section()

    def _create_header(self):
        """Create header section (matching batch processing theme)."""
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.pack(fill='x', pady=(0, 20))
        
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Single File Analysis",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=15)
        
        # Validation status frame (matching batch processing pattern)
        self.validation_status_frame = ctk.CTkFrame(self.header_frame)
        self.validation_status_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        self.validation_status_label = ctk.CTkLabel(
            self.validation_status_frame,
            text="Validation Status: Not Started",
            font=ctk.CTkFont(size=12)
        )
        self.validation_status_label.pack(side='left', padx=10, pady=10)
        
        self.validation_indicator = ctk.CTkLabel(
            self.validation_status_frame,
            text="●",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        self.validation_indicator.pack(side='right', padx=10, pady=10)

    def _create_file_selection(self):
        """Create file selection section (matching batch processing theme)."""
        self.file_frame = ctk.CTkFrame(self.main_container)
        self.file_frame.pack(fill='x', pady=(0, 20))
        
        self.file_label = ctk.CTkLabel(
            self.file_frame,
            text="File Selection:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.file_label.pack(anchor='w', padx=15, pady=(15, 5))
        
        # File input container
        self.file_input_frame = ctk.CTkFrame(self.file_frame)
        self.file_input_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        self.file_entry = ctk.CTkEntry(
            self.file_input_frame,
            placeholder_text="No file selected...",
            height=40,
            state="readonly"
        )
        self.file_entry.pack(side='left', fill='x', expand=True, padx=(10, 10), pady=10)
        
        self.browse_button = ctk.CTkButton(
            self.file_input_frame,
            text="Browse",
            command=self._browse_file,
            width=100,
            height=40
        )
        self.browse_button.pack(side='right', padx=(0, 10), pady=10)
        
        self.validate_button = ctk.CTkButton(
            self.file_input_frame,
            text="Pre-validate",
            command=self._pre_validate_file,
            width=120,
            height=40,
            state="disabled"
        )
        self.validate_button.pack(side='right', padx=(0, 10), pady=10)

    def _create_options_section(self):
        """Create analysis options section (matching batch processing theme)."""
        self.options_frame = ctk.CTkFrame(self.main_container)
        self.options_frame.pack(fill='x', pady=(0, 20))
        
        self.options_label = ctk.CTkLabel(
            self.options_frame,
            text="Analysis Options:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.options_label.pack(anchor='w', padx=15, pady=(15, 10))
        
        # Options container
        self.options_container = ctk.CTkFrame(self.options_frame)
        self.options_container.pack(fill='x', padx=15, pady=(0, 15))
        
        self.generate_plots_var = ctk.BooleanVar(value=True)
        self.generate_plots_check = ctk.CTkCheckBox(
            self.options_container,
            text="Generate Plots",
            variable=self.generate_plots_var
        )
        self.generate_plots_check.pack(side='left', padx=(10, 20), pady=10)
        
        self.save_to_db_var = ctk.BooleanVar(value=True)
        self.save_to_db_check = ctk.CTkCheckBox(
            self.options_container,
            text="Save to Database",
            variable=self.save_to_db_var
        )
        self.save_to_db_check.pack(side='left', padx=(0, 20), pady=10)
        
        self.comprehensive_validation_var = ctk.BooleanVar(value=True)
        self.comprehensive_validation_check = ctk.CTkCheckBox(
            self.options_container,
            text="Comprehensive Validation",
            variable=self.comprehensive_validation_var
        )
        self.comprehensive_validation_check.pack(side='left', padx=(0, 20), pady=10)

    def _create_prevalidation_section(self):
        """Create pre-validation results section (matching batch processing theme)."""
        self.prevalidation_frame = ctk.CTkFrame(self.main_container)
        # Initially hidden - will be shown when validation runs
        
        self.prevalidation_label = ctk.CTkLabel(
            self.prevalidation_frame,
            text="Pre-validation Results:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.prevalidation_label.pack(anchor='w', padx=15, pady=(15, 10))
        
        # Metrics container
        self.validation_metrics_frame = ctk.CTkFrame(self.prevalidation_frame)
        self.validation_metrics_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        # Create metric cards (matching batch processing layout)
        self.file_status_card = MetricCard(
            self.validation_metrics_frame,
            title="File Status",
            value="--",
            status="info"
        )
        self.file_status_card.pack(side='left', padx=10, pady=10, fill='both', expand=True)
        
        self.file_size_card = MetricCard(
            self.validation_metrics_frame,
            title="File Size", 
            value="--",
            status="info"
        )
        self.file_size_card.pack(side='left', padx=10, pady=10, fill='both', expand=True)
        
        self.sheet_count_card = MetricCard(
            self.validation_metrics_frame,
            title="Sheets Found",
            value="--",
            status="info"
        )
        self.sheet_count_card.pack(side='left', padx=10, pady=10, fill='both', expand=True)
        
        self.validation_grade_card = MetricCard(
            self.validation_metrics_frame,
            title="Validation Grade",
            value="--",
            status="info"
        )
        self.validation_grade_card.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    def _create_controls_section(self):
        """Create processing controls section (matching batch processing theme)."""
        self.controls_frame = ctk.CTkFrame(self.main_container)
        self.controls_frame.pack(fill='x', pady=(0, 20))
        
        self.controls_label = ctk.CTkLabel(
            self.controls_frame,
            text="Processing Controls:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.controls_label.pack(anchor='w', padx=15, pady=(15, 10))
        
        # Controls container
        self.controls_container = ctk.CTkFrame(self.controls_frame)
        self.controls_container.pack(fill='x', padx=15, pady=(0, 15))
        
        self.analyze_button = ctk.CTkButton(
            self.controls_container,
            text="Analyze File",
            command=self._start_analysis,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled",
            fg_color="green"
        )
        # Apply stable hover effect separately to prevent glitching
        self.analyze_button._hover_color = None  # Disable default hover
        self.analyze_button.pack(side='left', padx=(10, 10), pady=10)
        
        self.export_button = ctk.CTkButton(
            self.controls_container,
            text="Export Results",
            command=self._export_results,
            width=120,
            height=40,
            state="disabled"
        )
        # Apply stable hover effect separately to prevent glitching
        self.export_button._hover_color = None  # Disable default hover
        self.export_button.pack(side='left', padx=(0, 10), pady=10)
        
        # Bind additional click event as fallback
        self.export_button.bind("<Button-1>", lambda e: self._handle_export_click())
        # Also bind ButtonRelease for better compatibility
        self.export_button.bind("<ButtonRelease-1>", lambda e: self._handle_export_click())
        

        
        self.clear_button = ctk.CTkButton(
            self.controls_container,
            text="Clear",
            command=self._clear_results,
            width=100,
            height=40
        )
        self.clear_button.pack(side='left', padx=(0, 10), pady=10)

    def _create_results_section(self):
        """Create results display section with empty state handling."""
        self.results_frame = ctk.CTkFrame(self.main_container)
        self.results_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        self.results_label = ctk.CTkLabel(
            self.results_frame,
            text="Analysis Results:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.results_label.pack(anchor='w', padx=15, pady=(15, 10))
        
        # Empty state container
        self.empty_state_frame = ctk.CTkFrame(self.results_frame)
        self.empty_state_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        empty_icon = ctk.CTkLabel(
            self.empty_state_frame,
            text="📊",  # Unicode chart icon
            font=ctk.CTkFont(size=48)
        )
        empty_icon.pack(pady=(30, 10))
        
        empty_title = ctk.CTkLabel(
            self.empty_state_frame,
            text="No Analysis Results Yet",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        empty_title.pack(pady=(0, 10))
        
        empty_message = ctk.CTkLabel(
            self.empty_state_frame,
            text="Select an Excel file and click 'Analyze File' to begin.\n"
                 "Results will appear here once analysis is complete.",
            font=ctk.CTkFont(size=12),
            justify="center"
        )
        empty_message.pack(pady=(0, 30))
        
        # Results display widget (initially hidden)
        self.analysis_display = AnalysisDisplayWidget(self.results_frame)
        self.analysis_display.pack_forget()  # Hidden until results exist

    def _browse_file(self):
        """Browse for Excel file."""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Excel file",
                filetypes=[
                    ("Excel files", "*.xlsx *.xls"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                self.current_file = Path(file_path)
                
                # Update file entry
                self.file_entry.configure(state="normal")
                self.file_entry.delete(0, ctk.END)
                self.file_entry.insert(0, str(self.current_file))
                self.file_entry.configure(state="readonly")
                
                # Enable validation button
                self.validate_button.configure(state="normal")
                
                # Enable analyze button immediately when file is selected
                self.analyze_button.configure(state="normal")
                
                # Reset validation status
                self._update_validation_status("File Selected", "orange")
                
                logger.info(f"Selected file: {self.current_file}")
                
        except Exception as e:
            logger.error(f"Error browsing file: {e}")
            messagebox.showerror("Error", f"Failed to select file: {str(e)}")

    def _pre_validate_file(self):
        """Perform pre-validation of the selected file."""
        if not self.current_file:
            messagebox.showerror("Error", "No file selected")
            return
        
        # Disable validate button to prevent rapid clicks
        self.validate_button.configure(state="disabled")
        
        self._update_validation_status("Validating...", "orange")
        
        # Run validation in thread to avoid blocking UI
        def validate():
            try:
                # Simple validation - check if file exists and is readable
                if not self.current_file.exists():
                    self.after(0, self._handle_validation_error, "File does not exist")
                    return
                
                if not self.current_file.is_file():
                    self.after(0, self._handle_validation_error, "Path is not a file")
                    return
                
                # Check file size
                file_size_mb = self.current_file.stat().st_size / (1024 * 1024)
                max_size = getattr(self.analyzer_config.processing, 'max_file_size_mb', 100)
                
                if file_size_mb > max_size:
                    self.after(0, self._handle_validation_error, f"File too large: {file_size_mb:.1f}MB > {max_size}MB")
                    return
                
                # Try to read Excel file
                try:
                    import pandas as pd
                    excel_file = pd.ExcelFile(self.current_file)
                    sheet_names = excel_file.sheet_names
                    excel_file.close()
                except Exception as e:
                    self.after(0, self._handle_validation_error, f"Cannot read Excel file: {str(e)}")
                    return
                
                # Create validation result
                validation_result = {
                    'is_valid': True,
                    'warnings': [],
                    'errors': [],
                    'metadata': {
                        'file_size_mb': file_size_mb,
                        'sheet_names': sheet_names,
                        'validation_grade': 'A' if file_size_mb < 10 else 'B'
                    }
                }
                
                # Update UI on main thread
                self.after(0, self._handle_validation_result, validation_result)
                
            except Exception as e:
                logger.error(f"Pre-validation failed: {e}")
                self.after(0, self._handle_validation_error, str(e))
        
        thread = threading.Thread(target=validate, daemon=True)
        thread.start()

    def _handle_validation_result(self, validation_result):
        """Handle validation result on main thread."""
        try:
            if validation_result['is_valid']:
                self._update_validation_status("Validation Passed", "green")
                
                # Show pre-validation metrics
                self.prevalidation_frame.pack(fill='x', pady=(0, 20), before=self.controls_frame)
                
                # Update metric cards with validation data
                metadata = validation_result['metadata']
                
                self.file_status_card.update_value("Valid", "success")
                
                file_size = metadata.get('file_size_mb', 0)
                self.file_size_card.update_value(f"{file_size:.1f} MB", 
                                               "success" if file_size < 50 else "warning")
                
                sheet_count = len(metadata.get('sheet_names', []))
                self.sheet_count_card.update_value(str(sheet_count),
                                                 "success" if sheet_count > 0 else "danger")
                
                validation_grade = metadata.get('validation_grade', '--')
                self.validation_grade_card.update_value(validation_grade, "info")
                
                # Enable analysis button
                self.analyze_button.configure(state="normal")
                
                # Show warnings if any
                if validation_result['warnings']:
                    warning_msg = "Validation warnings:\n" + "\n".join(validation_result['warnings'])
                    messagebox.showwarning("Validation Warnings", warning_msg)
                
            else:
                self._update_validation_status("Validation Failed", "red")
                
                # Update file status card
                self.prevalidation_frame.pack(fill='x', pady=(0, 20), before=self.controls_frame)
                self.file_status_card.update_value("Invalid", "danger")
                
                # Show errors
                error_msg = "Validation failed:\n" + "\n".join(validation_result['errors'])
                messagebox.showerror("Validation Failed", error_msg)
            
            # Re-enable validate button after validation completes
            self.validate_button.configure(state="normal")
                
        except Exception as e:
            logger.error(f"Error handling validation result: {e}")
            self._handle_validation_error(f"Error processing validation result: {str(e)}")

    def _handle_validation_error(self, error_message):
        """Handle validation error on main thread."""
        self._update_validation_status("Validation Error", "red")
        
        # Re-enable validate button on error
        self.validate_button.configure(state="normal")
        
        messagebox.showerror("Validation Error", f"Pre-validation failed:\n{error_message}")
    
    def _setup_analysis_watchdog(self):
        """Set up a watchdog timer to prevent indefinite freezing."""
        # Cancel any existing watchdog
        if self.watchdog_timer:
            self.after_cancel(self.watchdog_timer)
        
        # Set up new watchdog (5 minutes timeout)
        def watchdog_timeout():
            if self.is_analyzing:
                logger.error("Analysis watchdog timeout - forcefully cleaning up")
                # Force cleanup
                self.is_analyzing = False
                
                # Try to unregister processing
                try:
                    if self.main_window:
                        self.main_window.unregister_processing("single_file")
                except Exception as e:
                    logger.error(f"Watchdog failed to unregister: {e}")
                
                # Try to re-enable controls
                try:
                    self._set_controls_state("normal")
                    if self.current_file:
                        self.analyze_button.configure(state="normal")
                except Exception as e:
                    logger.error(f"Watchdog failed to enable controls: {e}")
                
                # Hide progress dialog
                try:
                    if self.progress_dialog:
                        self.progress_dialog.hide()
                        self.progress_dialog = None
                except Exception as e:
                    logger.error(f"Watchdog failed to hide dialog: {e}")
                
                # Show timeout message
                try:
                    messagebox.showerror("Analysis Timeout", 
                                       "Analysis timed out after 5 minutes.\n"
                                       "The application has been reset to a usable state.")
                except:
                    pass
        
        # Schedule watchdog for 5 minutes
        self.watchdog_timer = self.after(300000, watchdog_timeout)  # 300000ms = 5 minutes
        logger.info("Analysis watchdog timer set for 5 minutes")
    
    def _cancel_watchdog(self):
        """Cancel the watchdog timer."""
        if self.watchdog_timer:
            try:
                self.after_cancel(self.watchdog_timer)
                self.watchdog_timer = None
                logger.debug("Cancelled analysis watchdog timer")
            except Exception as e:
                logger.error(f"Failed to cancel watchdog: {e}")

    def _update_validation_status(self, status: str, color: str):
        """Update validation status indicator."""
        try:
            self.validation_status_label.configure(text=f"Validation Status: {status}")
            
            color_map = {
                "green": "#00ff00",
                "orange": "#ffa500", 
                "red": "#ff0000",
                "gray": "#808080"
            }
            
            self.validation_indicator.configure(text_color=color_map.get(color, "#808080"))
        except Exception as e:
            logger.error(f"Error updating validation status: {e}")

    def _start_analysis(self):
        """Start file analysis."""
        if not self.current_file:
            messagebox.showerror("Error", "No file selected")
            return
            
        if self.is_analyzing:
            messagebox.showwarning("Warning", "Analysis already in progress")
            return
        
        # Disable analyze button immediately to prevent rapid clicks
        self.analyze_button.configure(state="disabled")
        
        try:
            # Reset previous results
            self._clear_results()
            
            # Disable all controls
            self._set_controls_state("disabled")
            
            # Flag that we're starting analysis BEFORE registering
            self.is_analyzing = True
            
            # Register with main window that we're processing
            processing_registered = False
            try:
                if self.main_window:
                    self.main_window.register_processing("single_file")
                    processing_registered = True
                    logger.info("Registered processing state")
            except Exception as e:
                logger.error(f"Failed to register processing: {e}")
                # Continue anyway but note the failure
            
            # Show progress dialog
            self.progress_dialog = ProgressDialog(
                self,
                title="Analyzing File",
                message="Starting analysis..."
            )
            self.progress_dialog.show()
            
            # Start analysis in thread
            self.analysis_thread = threading.Thread(
                target=self._run_analysis,
                daemon=True
            )
            self.analysis_thread.start()
            
            # Set up a watchdog timer (5 minutes timeout)
            self._setup_analysis_watchdog()
            
            logger.info(f"Started analysis of: {self.current_file}")
            
        except Exception as e:
            logger.error(f"Error starting analysis: {e}")
            messagebox.showerror("Error", f"Failed to start analysis: {str(e)}")
            
            # Reset state
            self.is_analyzing = False
            
            # Always try to re-enable controls
            try:
                self._set_controls_state("normal")
                # Ensure analyze button is re-enabled
                if self.current_file:
                    self.analyze_button.configure(state="normal")
            except Exception as cleanup_error:
                logger.error(f"Failed to re-enable controls: {cleanup_error}")
            
            # Unregister processing state if we registered it
            if self.main_window and processing_registered:
                try:
                    self.main_window.unregister_processing("single_file")
                except Exception as unreg_error:
                    logger.error(f"Failed to unregister processing: {unreg_error}")

    def _run_analysis(self):
        """Run analysis in background thread."""
        try:
            # Create output directory if plots requested
            output_dir = None
            if self.generate_plots_var.get():
                # Use data_directory from config and create output subdirectory
                base_dir = getattr(self.analyzer_config, 'data_directory', Path.home() / "LaserTrimResults")
                output_dir = base_dir / "single_analysis" / datetime.now().strftime("%Y%m%d_%H%M%S")
                ensure_directory(output_dir)
            
            # Progress callback with bounds checking and thread safety
            def progress_callback(message: str, progress: float):
                try:
                    if self.progress_dialog and not self.is_analyzing == False:
                        # Ensure progress is within bounds
                        safe_progress = max(0.0, min(1.0, progress))
                        # Use a more thread-safe approach
                        if hasattr(self, 'winfo_exists') and self.winfo_exists():
                            self.after_idle(lambda: update_progress_safe(message, safe_progress))
                except Exception as e:
                    logger.debug(f"Progress callback error: {e}")
            
            def update_progress_safe(message: str, progress: float):
                """Safely update progress from main thread."""
                try:
                    if self.progress_dialog and hasattr(self.progress_dialog, 'update_progress'):
                        self.progress_dialog.update_progress(message, progress)
                except Exception as e:
                    logger.debug(f"Progress update error: {e}")
            
            # Run analysis with asyncio in a more robust way
            try:
                # Check if there's already an event loop
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # No loop in this thread, create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run the async process_file method
                result = loop.run_until_complete(
                    self.processor.process_file(
                        file_path=self.current_file,
                        output_dir=output_dir,
                        progress_callback=progress_callback
                    )
                )
                
            except Exception as process_error:
                # If the processor fails, provide a detailed error message
                error_msg = f"Processing failed: {str(process_error)}"
                if "Config object has no attribute" in str(process_error):
                    error_msg = f"Configuration error: {str(process_error)}. Please check your configuration settings."
                elif "No module named" in str(process_error):
                    error_msg = f"Missing dependency: {str(process_error)}. Please ensure all required packages are installed."
                elif "Permission denied" in str(process_error):
                    error_msg = f"File access error: {str(process_error)}. Please check file permissions."
                else:
                    error_msg = f"Processing error: {str(process_error)}"
                
                raise ProcessingError(error_msg)
                
            # Save to database if requested
            if self.save_to_db_var.get() and hasattr(self.main_window, 'db_manager') and self.main_window.db_manager:
                try:
                    # Check for duplicates
                    existing_id = self.main_window.db_manager.check_duplicate_analysis(
                        result.metadata.model,
                        result.metadata.serial,
                        result.metadata.file_date
                    )
                    
                    if existing_id:
                        logger.info(f"Duplicate analysis found (ID: {existing_id})")
                        result.db_id = existing_id
                    else:
                        # Try normal save first
                        try:
                            result.db_id = self.main_window.db_manager.save_analysis(result)
                            logger.info(f"Saved to database with ID: {result.db_id}")
                            
                            # Validate the save
                            if not self.main_window.db_manager.validate_saved_analysis(result.db_id):
                                raise RuntimeError("Database validation failed")
                                
                        except Exception as save_error:
                            logger.warning(f"Normal save failed, trying force save: {save_error}")
                            # Try force save as fallback
                            result.db_id = self.main_window.db_manager.force_save_analysis(result)
                            logger.info(f"Force saved to database with ID: {result.db_id}")
                        
                except Exception as e:
                    logger.error(f"Database save failed: {e}")
                    # Continue without database save
            
            # Store result for UI update in finally block
            self.current_result = result
            self._analysis_output_dir = output_dir
            self._analysis_success = True
            logger.info("Analysis completed successfully")
                
        except ValidationError as e:
            logger.error(f"Validation error during analysis: {e}")
            self._analysis_error = f"Validation failed: {str(e)}"
            self._analysis_success = False
            
        except ProcessingError as e:
            logger.error(f"Processing error during analysis: {e}")
            self._analysis_error = f"Processing failed: {str(e)}"
            self._analysis_success = False
            
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}")
            logger.error(traceback.format_exc())
            self._analysis_error = f"Unexpected error: {str(e)}"
            self._analysis_success = False
        
        finally:
            self.is_analyzing = False
            
            # Cancel watchdog timer
            self._cancel_watchdog()
            
            # CRITICAL: Always unregister processing state 
            try:
                if self.main_window:
                    self.main_window.unregister_processing("single_file")
                    logger.info("Unregistered processing state in finally block")
            except Exception as e:
                logger.error(f"Failed to unregister processing in finally: {e}")
            
            # Schedule UI updates on main thread (non-blocking)
            def complete_ui_updates():
                try:
                    # Handle success or error
                    if hasattr(self, '_analysis_success') and self._analysis_success:
                        self._handle_success_ui()
                    elif hasattr(self, '_analysis_error'):
                        self._handle_error_ui(self._analysis_error)
                    
                    # Always re-enable controls
                    self._set_controls_state("normal")
                    if self.current_file:
                        self.analyze_button.configure(state="normal")
                    
                    logger.info("UI updates completed successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to complete UI updates: {e}")
                    # Last resort - just enable the analyze button
                    try:
                        if self.current_file:
                            self.analyze_button.configure(state="normal")
                    except:
                        pass
            
            # Schedule the UI update for the next event loop cycle
            try:
                self.after_idle(complete_ui_updates)
            except:
                # If scheduling fails, try direct execution
                try:
                    complete_ui_updates()
                except:
                    pass

    def _handle_success_ui(self):
        """Handle successful analysis UI updates - called from main thread."""
        try:
            result = self.current_result
            output_dir = getattr(self, '_analysis_output_dir', None)
            
            # Hide progress dialog
            if self.progress_dialog:
                try:
                    self.progress_dialog.hide()
                except Exception as e:
                    logger.error(f"Failed to hide progress dialog: {e}")
                finally:
                    self.progress_dialog = None
            
            # Update validation status based on result
            if hasattr(result, 'overall_validation_status'):
                if result.overall_validation_status == ValidationStatus.VALIDATED:
                    self._update_validation_status("Analysis Complete - Validated", "green")
                elif result.overall_validation_status == ValidationStatus.WARNING:
                    self._update_validation_status("Analysis Complete - With Warnings", "orange")
                elif result.overall_validation_status == ValidationStatus.FAILED:
                    self._update_validation_status("Analysis Complete - Validation Failed", "red")
                else:
                    self._update_validation_status("Analysis Complete - Not Validated", "gray")
            else:
                self._update_validation_status("Analysis Complete", "green")
            
            # Display results
            try:
                # Hide empty state and show analysis display
                self.empty_state_frame.pack_forget()
                self.analysis_display.pack(fill='both', expand=True, padx=15, pady=(0, 15))
                self.analysis_display.display_result(result)
            except Exception as e:
                logger.error(f"Failed to display analysis result: {e}")
                # Continue with other operations even if display fails
            
            # Enable export button
            try:
                self.export_button.configure(state="normal")
                logger.info("Export button enabled after successful analysis")
                
                # Debug: Check button state after enabling
                button_state = self.export_button.cget("state")
                logger.info(f"Export button state after enabling: {button_state}")
            except Exception as e:
                logger.error(f"Failed to enable export button: {e}")
            
            # CRITICAL: Always re-enable controls and unregister processing
            # Do this even if previous steps failed
            try:
                self._set_controls_state("normal")
            except Exception as e:
                logger.error(f"Failed to re-enable controls in success handler: {e}")
                # Try direct button enable as fallback
                try:
                    if self.current_file:
                        self.analyze_button.configure(state="normal")
                except:
                    pass
            
            # Show success notification
            if result:
                success_msg = f"Analysis completed successfully!\n\n"
                success_msg += f"Status: {result.overall_status.value}\n"
                
                if hasattr(result, 'overall_validation_status'):
                    success_msg += f"Validation: {result.overall_validation_status.value}\n"
                
                success_msg += f"Processing time: {result.processing_time:.2f}s"
                
                if output_dir:
                    success_msg += f"\n\nOutputs saved to: {output_dir}"
                
                # Show success message
                try:
                    messagebox.showinfo("Analysis Complete", success_msg)
                except Exception as e:
                    logger.error(f"Failed to show success message: {e}")
            
            logger.info("Analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in success UI handler: {e}")

    def _handle_error_ui(self, error_message: str):
        """Handle analysis error UI updates - called from main thread."""
        try:
            # Hide progress dialog
            if self.progress_dialog:
                try:
                    self.progress_dialog.hide()
                except Exception as e:
                    logger.error(f"Failed to hide progress dialog in error handler: {e}")
                finally:
                    self.progress_dialog = None
            
            # Update validation status
            try:
                self._update_validation_status("Analysis Failed", "red")
            except Exception as e:
                logger.error(f"Failed to update validation status in error handler: {e}")
            
            # CRITICAL: Always re-enable controls and unregister processing
            # These MUST happen even if other operations fail
            
            # Re-enable controls
            try:
                self._set_controls_state("normal")
            except Exception as e:
                logger.error(f"Failed to re-enable controls in error handler: {e}")
                # Try direct button enable as fallback
                try:
                    if self.current_file:
                        self.analyze_button.configure(state="normal")
                    self.browse_button.configure(state="normal")
                    self.clear_button.configure(state="normal")
                except Exception as fallback_error:
                    logger.error(f"Failed fallback control enable: {fallback_error}")
            
            # Unregister processing state - MUST happen
            try:
                if self.main_window:
                    self.main_window.unregister_processing("single_file")
                    logger.info("Unregistered processing in error handler")
            except Exception as e:
                logger.error(f"Failed to unregister processing in error handler: {e}")
            
            # Show error message (non-critical)
            try:
                messagebox.showerror("Analysis Failed", error_message)
            except Exception as e:
                logger.error(f"Failed to show error messagebox: {e}")
            
            logger.error(f"Analysis failed: {error_message}")
            
        except Exception as e:
            logger.error(f"Critical error in error handler: {e}")
            # Last resort - try to at least unregister and enable buttons
            try:
                if self.main_window:
                    self.main_window.unregister_processing("single_file")
            except:
                pass
            try:
                self.analyze_button.configure(state="normal")
            except:
                pass





    def _set_controls_state(self, state: str):
        """Set state of control buttons.
        
        Args:
            state: Either "normal" or "disabled"
        """
        try:
            # For normal state, check if we have a file selected
            if state == "normal":
                # Analyze button should be enabled only if a file is selected
                analyze_state = "normal" if self.current_file else "disabled"
                self.analyze_button.configure(state=analyze_state)
                
                # Validate button should be enabled only if a file is selected
                validate_state = "normal" if self.current_file else "disabled"
                self.validate_button.configure(state=validate_state)
                
                # Browse button should always be enabled in normal state
                self.browse_button.configure(state="normal")
                
                # Clear button should always be enabled in normal state
                self.clear_button.configure(state="normal")
                
                # Export button state depends on whether we have results
                export_state = "normal" if self.current_result else "disabled"
                self.export_button.configure(state=export_state)
                
                # Emergency and diagnostic buttons should always be enabled for troubleshooting
                if hasattr(self, 'emergency_button'):
                    self.emergency_button.configure(state="normal")
                if hasattr(self, 'diagnostic_button'):
                    self.diagnostic_button.configure(state="normal")
            else:
                # Disabled state - disable most controls but keep emergency buttons enabled
                self.analyze_button.configure(state=state)
                self.validate_button.configure(state=state)
                self.browse_button.configure(state=state)
                self.clear_button.configure(state=state)
                self.export_button.configure(state=state)
                
                # Keep emergency buttons enabled for troubleshooting
                if hasattr(self, 'emergency_button'):
                    self.emergency_button.configure(state="normal")
                if hasattr(self, 'diagnostic_button'):
                    self.diagnostic_button.configure(state="normal")
                
            logger.debug(f"Controls state set to: {state}")
            
        except Exception as e:
            logger.error(f"Error setting controls state: {e}")
            # Try to at least enable the analyze button if we have a file
            try:
                if state == "normal" and self.current_file:
                    self.analyze_button.configure(state="normal")
            except:
                pass

    def _handle_export_click(self):
        """Handle export button click as fallback."""
        logger.info("Export button clicked via fallback handler")
        # Force check if we should process the export
        try:
            button_state = str(self.export_button.cget("state"))
            logger.info(f"Fallback handler - Button state: {button_state}")
            logger.info(f"Fallback handler - Has result: {self.current_result is not None}")
            
            # Process export if we have results, regardless of button state
            # This helps work around any CTkButton state issues
            if self.current_result:
                logger.info("Fallback handler - Processing export despite button state")
                self._export_results()
            else:
                logger.warning("Export button clicked but no results available")
                messagebox.showwarning("No Results", "No analysis results to export. Please analyze a file first.")
        except Exception as e:
            logger.error(f"Error in fallback export handler: {e}")

    def _export_results(self):
        """Export analysis results."""
        logger.info("Export button clicked from main handler")
        
        # Pre-check pandas availability
        try:
            import pandas as pd
            logger.info("Pandas import test successful")
        except ImportError as e:
            logger.error(f"Pandas not available: {e}")
            messagebox.showerror("Missing Dependency", 
                               "The pandas library is required for export functionality.\n"
                               "Please install it using: pip install pandas openpyxl")
            return
        
        # Check if button is actually enabled
        try:
            button_state = self.export_button.cget("state")
            logger.info(f"Export button state: {button_state}")
        except Exception as e:
            logger.error(f"Could not get button state: {e}")
        
        if not self.current_result:
            logger.warning("No current_result available for export")
            messagebox.showerror("Error", "No results to export")
            return
        
        logger.info(f"Current result type: {type(self.current_result)}")
        logger.info(f"Current result has tracks: {hasattr(self.current_result, 'tracks')}")
        
        try:
            # Ask for export location
            logger.info("Opening file save dialog")
            file_path = filedialog.asksaveasfilename(
                title="Export Analysis Results",
                defaultextension=".xlsx",
                filetypes=[
                    ("Excel files", "*.xlsx"),
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                logger.info(f"User selected export path: {file_path}")
                try:
                    # Export based on file extension
                    if file_path.endswith('.xlsx'):
                        logger.info("Exporting to Excel format")
                        self._export_excel(Path(file_path))
                    elif file_path.endswith('.csv'):
                        logger.info("Exporting to CSV format")
                        self._export_csv(Path(file_path))
                    else:
                        logger.error(f"Unsupported file format: {file_path}")
                        messagebox.showerror("Error", "Unsupported file format")
                        return
                    
                    messagebox.showinfo("Export Complete", f"Results exported to:\n{file_path}")
                    logger.info(f"Results exported successfully to: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Export failed: {e}", exc_info=True)
                    messagebox.showerror("Export Failed", f"Failed to export results:\n{str(e)}")
            else:
                logger.info("User cancelled export dialog")
                    
        except Exception as e:
            logger.error(f"Error in export dialog: {e}", exc_info=True)
            messagebox.showerror("Error", f"Export dialog failed: {str(e)}")

    def _export_excel(self, file_path: Path):
        """Export results to Excel format."""
        logger.info(f"Starting Excel export to: {file_path}")
        
        try:
            import pandas as pd
            logger.info("Pandas imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import pandas: {e}")
            raise
        
        result = self.current_result
        logger.info(f"Result object: {result}")
        logger.info(f"Result metadata: {getattr(result, 'metadata', 'No metadata')}")
        
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                logger.info("ExcelWriter created successfully")
                # Get values safely with error handling
                file_name = getattr(result.metadata, 'file_name', 
                                   getattr(result.metadata, 'filename', 'Unknown'))
                model = getattr(result.metadata, 'model', 'Unknown')
                serial = getattr(result.metadata, 'serial', 'Unknown')
                
                # Handle system_type safely
                system_type = 'Unknown'
                if hasattr(result.metadata, 'system_type'):
                    system_type = getattr(result.metadata.system_type, 'value', str(result.metadata.system_type))
                elif hasattr(result.metadata, 'system'):
                    system_type = getattr(result.metadata.system, 'value', str(result.metadata.system))
                
                # Handle analysis_date safely
                analysis_date = 'Unknown'
                if hasattr(result.metadata, 'analysis_date'):
                    if hasattr(result.metadata.analysis_date, 'strftime'):
                        analysis_date = result.metadata.analysis_date.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        analysis_date = str(result.metadata.analysis_date)
                elif hasattr(result.metadata, 'file_date'):
                    if hasattr(result.metadata.file_date, 'strftime'):
                        analysis_date = result.metadata.file_date.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        analysis_date = str(result.metadata.file_date)
                
                # Handle overall_status safely
                overall_status = 'Unknown'
                if hasattr(result, 'overall_status'):
                    overall_status = getattr(result.overall_status, 'value', str(result.overall_status))
                
                # Handle validation_status safely
                validation_status = 'N/A'
                if hasattr(result, 'overall_validation_status'):
                    validation_status = getattr(result.overall_validation_status, 'value', 
                                              str(result.overall_validation_status))
                
                # Get track count safely
                track_count = 0
                if hasattr(result, 'tracks'):
                    track_count = len(result.tracks)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['File', 'Model', 'Serial', 'System Type', 'Analysis Date', 'Overall Status', 
                              'Validation Status', 'Processing Time (s)', 'Track Count'],
                    'Value': [
                        file_name,
                        model,
                        serial,
                        system_type,
                        analysis_date,
                        overall_status,
                        validation_status,
                        f"{getattr(result, 'processing_time', 0):.2f}",
                        track_count
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Track details sheet
                if hasattr(result, 'tracks') and result.tracks:
                    track_details = []
                    for track_id, track in result.tracks.items():
                        track_details.append({
                            'Track_ID': track_id,
                            'Sigma_Gradient': getattr(track.sigma_analysis, 'sigma_gradient', None) if track.sigma_analysis else None,
                            'Sigma_Threshold': getattr(track.sigma_analysis, 'sigma_threshold', None) if track.sigma_analysis else None,
                            'Sigma_Pass': getattr(track.sigma_analysis, 'sigma_pass', None) if track.sigma_analysis else None,
                            'Linearity_Spec': getattr(track.linearity_analysis, 'linearity_spec', None) if track.linearity_analysis else None,
                            'Linearity_Pass': getattr(track.linearity_analysis, 'linearity_pass', None) if track.linearity_analysis else None,
                            'Overall_Status': getattr(track.overall_status, 'value', str(track.overall_status)),
                            'Risk_Category': getattr(track.risk_category, 'value', str(track.risk_category)) if hasattr(track, 'risk_category') else 'Unknown'
                        })
                    
                    if track_details:
                        tracks_df = pd.DataFrame(track_details)
                        tracks_df.to_excel(writer, sheet_name='Track Details', index=False)
                        
                logger.info("Excel export completed successfully")
        except Exception as e:
            logger.error(f"Error during Excel export: {e}", exc_info=True)
            raise

    def _export_csv(self, file_path: Path):
        """Export results to CSV format."""
        logger.info(f"Starting CSV export to: {file_path}")
        
        try:
            import pandas as pd
            logger.info("Pandas imported successfully for CSV export")
        except ImportError as e:
            logger.error(f"Failed to import pandas for CSV: {e}")
            raise
        
        result = self.current_result
        logger.info(f"CSV export - Result object: {result}")
        logger.info(f"CSV export - Has tracks: {hasattr(result, 'tracks') and result.tracks}")
        
        # Create comprehensive export data
        export_data = []
        
        if hasattr(result, 'tracks') and result.tracks:
            for track_id, track in result.tracks.items():
                row = {
                    'File': getattr(result.metadata, 'file_name', getattr(result.metadata, 'filename', 'Unknown')),
                    'Model': getattr(result.metadata, 'model', 'Unknown'),
                    'Serial': getattr(result.metadata, 'serial', 'Unknown'),
                    'System_Type': getattr(getattr(result.metadata, 'system', None), 'value', 'Unknown'),
                    'Analysis_Date': getattr(result.metadata, 'file_date', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                    'Track_ID': track_id,
                    'Overall_Status': getattr(result.overall_status, 'value', str(result.overall_status)),
                    'Track_Status': getattr(track.overall_status, 'value', str(track.overall_status)),
                    'Processing_Time': f"{getattr(result, 'processing_time', 0):.2f}",
                    'Validation_Status': getattr(getattr(result, 'overall_validation_status', None), 'value', 'N/A')
                }
                
                # Add sigma analysis data
                if track.sigma_analysis:
                    row.update({
                        'Sigma_Gradient': getattr(track.sigma_analysis, 'sigma_gradient', None),
                        'Sigma_Threshold': getattr(track.sigma_analysis, 'sigma_threshold', None),
                        'Sigma_Pass': getattr(track.sigma_analysis, 'sigma_pass', None),
                        'Sigma_Improvement': getattr(track.sigma_analysis, 'improvement_percent', None)
                    })
                
                # Add linearity analysis data  
                if track.linearity_analysis:
                    row.update({
                        'Linearity_Spec': getattr(track.linearity_analysis, 'linearity_spec', None),
                        'Linearity_Pass': getattr(track.linearity_analysis, 'linearity_pass', None),
                        'Linearity_Error': getattr(track.linearity_analysis, 'linearity_error', None)
                    })
                
                # Add resistance analysis data
                if hasattr(track, 'resistance_analysis') and track.resistance_analysis:
                    row.update({
                        'Resistance_Before': getattr(track.resistance_analysis, 'resistance_before', None),
                        'Resistance_After': getattr(track.resistance_analysis, 'resistance_after', None),
                        'Resistance_Change_Percent': getattr(track.resistance_analysis, 'resistance_change_percent', None)
                    })
                
                # Add risk category
                if hasattr(track, 'risk_category'):
                    row['Risk_Category'] = getattr(track.risk_category, 'value', str(track.risk_category))
                
                export_data.append(row)
        else:
            # Single row for file-level data if no tracks
            export_data.append({
                'File': getattr(result.metadata, 'file_name', getattr(result.metadata, 'filename', 'Unknown')),
                'Model': getattr(result.metadata, 'model', 'Unknown'),
                'Serial': getattr(result.metadata, 'serial', 'Unknown'),
                'Overall_Status': getattr(result.overall_status, 'value', str(result.overall_status)),
                'Processing_Time': f"{getattr(result, 'processing_time', 0):.2f}",
                'Validation_Status': getattr(getattr(result, 'overall_validation_status', None), 'value', 'N/A')
            })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(export_data)
        df.to_csv(file_path, index=False)
        logger.info(f"CSV export completed successfully - {len(export_data)} rows exported")

    def _clear_results(self):
        """Clear analysis results."""
        try:
            self.current_result = None
            self.analysis_display.clear()
            self.export_button.configure(state="disabled")
            
            # Hide analysis display and show empty state
            self.analysis_display.pack_forget()
            self.empty_state_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))
            
            # Hide pre-validation frame if no file selected
            if not self.current_file:
                self.prevalidation_frame.pack_forget()
                self._update_validation_status("Not Started", "gray")
            
            logger.info("Results cleared")
            
        except Exception as e:
            logger.error(f"Error clearing results: {e}")
    
    # Add BasePage compatibility methods
    def on_show(self):
        """Called when page is shown."""
        self.is_visible = True
        
    def on_hide(self):
        """Called when page is hidden."""
        self.is_visible = False
        
        # If we're still registered as processing, unregister
        if self.main_window and self.is_analyzing:
            try:
                self.main_window.unregister_processing("single_file")
                logger.warning("Unregistered processing state on page hide")
            except Exception as e:
                logger.error(f"Failed to unregister on hide: {e}")
            
            # Also try to re-enable controls
            try:
                self._set_controls_state("normal")
            except:
                pass
        
    def refresh(self):
        """Refresh page content."""
        pass
        
    def mark_needs_refresh(self):
        """Mark that the page needs refresh on next show."""
        self.needs_refresh = True
        
    def request_stop_processing(self):
        """Request that any ongoing processing be stopped."""
        self._stop_requested = True
        
        # If we're analyzing, ensure we clean up properly
        if self.is_analyzing:
            logger.info("Stop requested during single file analysis")
            # The analysis thread will handle the cleanup in its finally block
        
    def reset_stop_request(self):
        """Reset the stop processing flag."""
        self._stop_requested = False
        
    def is_stop_requested(self) -> bool:
        """Check if processing should be stopped."""
        return self._stop_requested
    
    @property
    def db_manager(self):
        """Get database manager from main window."""
        return getattr(self.main_window, 'db_manager', None)

    @property
    def app_config(self):
        """Get configuration from main window."""
        return getattr(self.main_window, 'config', None)
