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

logger = logging.getLogger(__name__)


class SingleFileAnalysisPage(ctk.CTkFrame):
    """Single file analysis page with comprehensive validation."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.config = get_config()
        self.processor = LaserTrimProcessor(self.config)
        self.db_manager = DatabaseManager()
        
        # State
        self.current_file: Optional[Path] = None
        self.current_result: Optional[AnalysisResult] = None
        self.analysis_thread: Optional[threading.Thread] = None
        self.is_analyzing = False
        
        # Create UI
        self._create_widgets()
        self._setup_layout()
        
        logger.info("Single file analysis page initialized")

    def _create_widgets(self):
        """Create UI widgets."""
        
        # Header with validation status
        self.header_frame = ctk.CTkFrame(self)
        
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Single File Analysis",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        
        # Validation status indicator
        self.validation_status_frame = ctk.CTkFrame(self.header_frame)
        self.validation_status_label = ctk.CTkLabel(
            self.validation_status_frame,
            text="Validation Status: Not Started",
            font=ctk.CTkFont(size=12)
        )
        self.validation_indicator = ctk.CTkLabel(
            self.validation_status_frame,
            text="●",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        
        # File selection
        self.file_frame = ctk.CTkFrame(self)
        
        self.file_label = ctk.CTkLabel(
            self.file_frame,
            text="Select Excel File:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        self.file_entry = ctk.CTkEntry(
            self.file_frame,
            placeholder_text="No file selected...",
            width=400,
            state="readonly"
        )
        
        self.browse_button = ctk.CTkButton(
            self.file_frame,
            text="Browse",
            command=self._browse_file,
            width=100
        )
        
        self.validate_button = ctk.CTkButton(
            self.file_frame,
            text="Pre-validate",
            command=self._pre_validate_file,
            width=120,
            state="disabled"
        )
        
        # Analysis controls
        self.controls_frame = ctk.CTkFrame(self)
        
        self.analyze_button = ctk.CTkButton(
            self.controls_frame,
            text="Analyze File",
            command=self._start_analysis,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled"
        )
        
        self.export_button = ctk.CTkButton(
            self.controls_frame,
            text="Export Results",
            command=self._export_results,
            width=120,
            state="disabled"
        )
        
        self.clear_button = ctk.CTkButton(
            self.controls_frame,
            text="Clear",
            command=self._clear_results,
            width=100
        )
        
        # Analysis options
        self.options_frame = ctk.CTkFrame(self)
        
        self.options_label = ctk.CTkLabel(
            self.options_frame,
            text="Analysis Options:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        self.generate_plots_var = ctk.BooleanVar(value=True)
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
        
        # Pre-validation metrics (initially hidden)
        self.prevalidation_frame = ctk.CTkFrame(self)
        self.prevalidation_frame.grid_remove()  # Hidden initially
        
        self.prevalidation_label = ctk.CTkLabel(
            self.prevalidation_frame,
            text="Pre-validation Results:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        # Create metric cards for pre-validation
        self.file_status_card = MetricCard(
            self.prevalidation_frame,
            title="File Status",
            value="Unknown",
            color_scheme="neutral"
        )
        
        self.file_size_card = MetricCard(
            self.prevalidation_frame,
            title="File Size",
            value="0 MB",
            color_scheme="neutral"
        )
        
        self.sheet_count_card = MetricCard(
            self.prevalidation_frame,
            title="Sheet Count",
            value="0",
            color_scheme="neutral"
        )
        
        self.system_type_card = MetricCard(
            self.prevalidation_frame,
            title="System Type",
            value="Unknown",
            color_scheme="neutral"
        )
        
        # Results display
        self.results_frame = ctk.CTkFrame(self)
        
        self.results_label = ctk.CTkLabel(
            self.results_frame,
            text="Analysis Results:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        # Create scrollable frame for results
        self.results_scroll = ctk.CTkScrollableFrame(
            self.results_frame,
            height=400
        )
        
        # Analysis display widget
        self.analysis_display = AnalysisDisplayWidget(self.results_scroll)
        
        # Progress indicator
        self.progress_dialog: Optional[ProgressDialog] = None

    def _setup_layout(self):
        """Setup widget layout."""
        self.grid_rowconfigure(7, weight=1)  # Results frame expands
        self.grid_columnconfigure(0, weight=1)
        
        # Header
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        self.header_frame.grid_columnconfigure(0, weight=1)
        
        self.title_label.grid(row=0, column=0, sticky="w")
        self.validation_status_frame.grid(row=0, column=1, sticky="e")
        
        self.validation_status_frame.grid_columnconfigure(0, weight=1)
        self.validation_status_label.grid(row=0, column=0, padx=(0, 10))
        self.validation_indicator.grid(row=0, column=1)
        
        # File selection
        self.file_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        self.file_frame.grid_columnconfigure(1, weight=1)
        
        self.file_label.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))
        self.file_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 10))
        self.browse_button.grid(row=1, column=2, padx=(0, 10))
        self.validate_button.grid(row=1, column=3)
        
        # Analysis options
        self.options_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        self.options_frame.grid_columnconfigure(4, weight=1)
        
        self.options_label.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))
        self.generate_plots_check.grid(row=1, column=0, sticky="w", padx=(0, 20))
        self.save_to_db_check.grid(row=1, column=1, sticky="w", padx=(0, 20))
        self.comprehensive_validation_check.grid(row=1, column=2, sticky="w", padx=(0, 20))
        
        # Pre-validation metrics (row 3, initially hidden)
        self.prevalidation_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=10)
        self.prevalidation_frame.grid_columnconfigure([0, 1, 2, 3], weight=1)
        
        self.prevalidation_label.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))
        self.file_status_card.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.file_size_card.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.sheet_count_card.grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        self.system_type_card.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        
        # Controls
        self.controls_frame.grid(row=4, column=0, sticky="ew", padx=20, pady=10)
        
        self.analyze_button.grid(row=0, column=0, padx=(0, 10))
        self.export_button.grid(row=0, column=1, padx=(0, 10))
        self.clear_button.grid(row=0, column=2)
        
        # Results
        self.results_frame.grid(row=5, column=0, sticky="nsew", padx=20, pady=10)
        self.results_frame.grid_rowconfigure(1, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        self.results_label.grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.results_scroll.grid(row=1, column=0, sticky="nsew")
        
        self.analysis_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def _browse_file(self):
        """Browse for Excel file."""
        file_path = filedialog.askopenfilename(
            title="Select Excel file",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_file = Path(file_path)
            self.file_entry.configure(state="normal")
            self.file_entry.delete(0, ctk.END)
            self.file_entry.insert(0, str(self.current_file))
            self.file_entry.configure(state="readonly")
            
            # Enable validation button
            self.validate_button.configure(state="normal")
            
            # Reset validation status
            self._update_validation_status("File Selected", "orange")
            
            logger.info(f"Selected file: {self.current_file}")

    def _pre_validate_file(self):
        """Perform pre-validation of the selected file."""
        if not self.current_file:
            messagebox.showerror("Error", "No file selected")
            return
        
        self._update_validation_status("Validating...", "orange")
        
        # Run validation in thread to avoid blocking UI
        def validate():
            try:
                from laser_trim_analyzer.utils.validators import validate_excel_file
                
                validation_result = validate_excel_file(
                    file_path=self.current_file,
                    max_file_size_mb=self.config.processing.max_file_size_mb
                )
                
                # Update UI on main thread
                self.after(0, self._handle_validation_result, validation_result)
                
            except Exception as e:
                logger.error(f"Pre-validation failed: {e}")
                self.after(0, self._handle_validation_error, str(e))
        
        thread = threading.Thread(target=validate, daemon=True)
        thread.start()

    def _handle_validation_result(self, validation_result):
        """Handle validation result on main thread."""
        if validation_result.is_valid:
            self._update_validation_status("Validation Passed", "green")
            
            # Show pre-validation metrics
            self.prevalidation_frame.grid()
            
            # Update metric cards with validation data
            metadata = validation_result.metadata
            
            self.file_status_card.update_value("Valid", "success")
            
            file_size = metadata.get('file_size_mb', 0)
            self.file_size_card.update_value(f"{file_size:.1f} MB", 
                                           "success" if file_size < 50 else "warning")
            
            sheet_count = len(metadata.get('sheet_names', []))
            self.sheet_count_card.update_value(str(sheet_count),
                                             "success" if sheet_count > 0 else "danger")
            
            system_type = metadata.get('detected_system', 'Unknown')
            self.system_type_card.update_value(system_type, "info")
            
            # Enable analysis button
            self.analyze_button.configure(state="normal")
            
            # Show warnings if any
            if validation_result.warnings:
                warning_msg = "Validation warnings:\n" + "\n".join(validation_result.warnings)
                messagebox.showwarning("Validation Warnings", warning_msg)
            
        else:
            self._update_validation_status("Validation Failed", "red")
            
            # Update file status card
            self.prevalidation_frame.grid()
            self.file_status_card.update_value("Invalid", "danger")
            
            # Show errors
            error_msg = "Validation failed:\n" + "\n".join(validation_result.errors)
            messagebox.showerror("Validation Failed", error_msg)

    def _handle_validation_error(self, error_message):
        """Handle validation error on main thread."""
        self._update_validation_status("Validation Error", "red")
        messagebox.showerror("Validation Error", f"Pre-validation failed:\n{error_message}")

    def _update_validation_status(self, status: str, color: str):
        """Update validation status indicator."""
        self.validation_status_label.configure(text=f"Validation Status: {status}")
        
        color_map = {
            "green": "#00ff00",
            "orange": "#ffa500", 
            "red": "#ff0000",
            "gray": "#808080"
        }
        
        self.validation_indicator.configure(text_color=color_map.get(color, "#808080"))

    def _start_analysis(self):
        """Start file analysis."""
        if not self.current_file:
            messagebox.showerror("Error", "No file selected")
            return
            
        if self.is_analyzing:
            messagebox.showwarning("Warning", "Analysis already in progress")
            return
        
        # Reset previous results
        self._clear_results()
        
        # Disable controls
        self._set_controls_state("disabled")
        
        # Show progress dialog
        self.progress_dialog = ProgressDialog(
            self,
            title="Analyzing File",
            message="Starting analysis..."
        )
        self.progress_dialog.show()
        
        # Start analysis in thread
        self.is_analyzing = True
        self.analysis_thread = threading.Thread(
            target=self._run_analysis,
            daemon=True
        )
        self.analysis_thread.start()
        
        logger.info(f"Started analysis of: {self.current_file}")

    def _run_analysis(self):
        """Run analysis in background thread."""
        try:
            # Create output directory if plots requested
            output_dir = None
            if self.generate_plots_var.get():
                output_dir = self.config.output_directory / "single_analysis" / datetime.now().strftime("%Y%m%d_%H%M%S")
                ensure_directory(output_dir)
            
            # Progress callback
            def progress_callback(message: str, progress: float):
                if self.progress_dialog:
                    self.after(0, lambda: self.progress_dialog.update_progress(message, progress))
            
            # Run analysis with asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.processor.process_file(
                        file_path=self.current_file,
                        output_dir=output_dir,
                        progress_callback=progress_callback
                    )
                )
                
                # Save to database if requested
                if self.save_to_db_var.get() and self.db_manager:
                    try:
                        # Check for duplicates
                        existing_id = self.db_manager.check_duplicate_analysis(
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
                                result.db_id = self.db_manager.save_analysis(result)
                                logger.info(f"Saved to database with ID: {result.db_id}")
                                
                                # Validate the save
                                if not self.db_manager.validate_saved_analysis(result.db_id):
                                    raise RuntimeError("Database validation failed")
                                    
                            except Exception as save_error:
                                logger.warning(f"Normal save failed, trying force save: {save_error}")
                                # Try force save as fallback
                                result.db_id = self.db_manager.force_save_analysis(result)
                                logger.info(f"Force saved to database with ID: {result.db_id}")
                            
                    except Exception as e:
                        logger.error(f"Database save failed: {e}")
                        # Continue without database save
                
                # Update UI on main thread
                self.after(0, self._handle_analysis_success, result, output_dir)
                
            finally:
                loop.close()
                
        except ValidationError as e:
            logger.error(f"Validation error during analysis: {e}")
            self.after(0, self._handle_analysis_error, f"Validation failed: {str(e)}")
            
        except ProcessingError as e:
            logger.error(f"Processing error during analysis: {e}")
            self.after(0, self._handle_analysis_error, f"Processing failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}")
            logger.error(traceback.format_exc())
            self.after(0, self._handle_analysis_error, f"Unexpected error: {str(e)}")
        
        finally:
            self.is_analyzing = False

    def _handle_analysis_success(self, result: AnalysisResult, output_dir: Optional[Path]):
        """Handle successful analysis completion."""
        self.current_result = result
        
        # Hide progress dialog
        if self.progress_dialog:
            self.progress_dialog.hide()
            self.progress_dialog = None
        
        # Update validation status based on result
        if result.overall_validation_status == ValidationStatus.VALIDATED:
            self._update_validation_status("Analysis Complete - Validated", "green")
        elif result.overall_validation_status == ValidationStatus.WARNING:
            self._update_validation_status("Analysis Complete - With Warnings", "orange")
        elif result.overall_validation_status == ValidationStatus.FAILED:
            self._update_validation_status("Analysis Complete - Validation Failed", "red")
        else:
            self._update_validation_status("Analysis Complete - Not Validated", "gray")
        
        # Display results
        self.analysis_display.display_result(result)
        
        # Enable export button
        self.export_button.configure(state="normal")
        
        # Re-enable controls
        self._set_controls_state("normal")
        
        # Show success message with validation info
        success_msg = f"Analysis completed successfully!\n\n"
        success_msg += f"Status: {result.overall_status.value}\n"
        success_msg += f"Validation: {result.overall_validation_status.value}\n"
        success_msg += f"Processing time: {result.processing_time:.2f}s"
        
        if hasattr(result, 'validation_grade') and result.validation_grade:
            success_msg += f"\nValidation Grade: {result.validation_grade}"
        
        if output_dir:
            success_msg += f"\n\nOutputs saved to: {output_dir}"
        
        messagebox.showinfo("Analysis Complete", success_msg)
        
        logger.info(f"Analysis completed successfully: {result.overall_status.value}")

    def _handle_analysis_error(self, error_message: str):
        """Handle analysis error."""
        # Hide progress dialog
        if self.progress_dialog:
            self.progress_dialog.hide()
            self.progress_dialog = None
        
        # Update validation status
        self._update_validation_status("Analysis Failed", "red")
        
        # Re-enable controls
        self._set_controls_state("normal")
        
        # Show error message
        messagebox.showerror("Analysis Failed", error_message)
        
        logger.error(f"Analysis failed: {error_message}")

    def _set_controls_state(self, state: str):
        """Set state of control buttons."""
        self.analyze_button.configure(state=state)
        if state == "normal":
            self.validate_button.configure(state="normal" if self.current_file else "disabled")
        else:
            self.validate_button.configure(state=state)

    def _export_results(self):
        """Export analysis results."""
        if not self.current_result:
            messagebox.showerror("Error", "No results to export")
            return
        
        # Ask for export location
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
            try:
                # Export based on file extension
                if file_path.endswith('.xlsx'):
                    self._export_excel(Path(file_path))
                elif file_path.endswith('.csv'):
                    self._export_csv(Path(file_path))
                else:
                    messagebox.showerror("Error", "Unsupported file format")
                    return
                
                messagebox.showinfo("Export Complete", f"Results exported to:\n{file_path}")
                logger.info(f"Results exported to: {file_path}")
                
            except Exception as e:
                logger.error(f"Export failed: {e}")
                messagebox.showerror("Export Failed", f"Failed to export results:\n{str(e)}")

    def _export_excel(self, file_path: Path):
        """Export results to Excel format."""
        # Implementation for Excel export with validation data
        pass

    def _export_csv(self, file_path: Path):
        """Export results to CSV format."""
        # Implementation for CSV export
        pass

    def _clear_results(self):
        """Clear analysis results."""
        self.current_result = None
        self.analysis_display.clear()
        self.export_button.configure(state="disabled")
        
        # Hide pre-validation frame if no file selected
        if not self.current_file:
            self.prevalidation_frame.grid_remove()
            self._update_validation_status("Not Started", "gray")
        
        logger.info("Results cleared") 