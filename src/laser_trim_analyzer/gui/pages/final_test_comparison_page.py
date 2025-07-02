"""
Final Test Comparison Page for Laser Trim Analyzer

Allows comparison of laser trim data with final test data.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import logging

from laser_trim_analyzer.gui.widgets.animated_widgets import AnimatedButton
from laser_trim_analyzer.gui.widgets.hover_fix import fix_hover_glitches, stabilize_layout
from laser_trim_analyzer.core.models import ValidationResult, TrackData
from laser_trim_analyzer.database.models import AnalysisResult as DBAnalysisResult
from laser_trim_analyzer.utils.plotting_utils import save_plot
# from laser_trim_analyzer.gui.pages.base_page_ctk import BasePage  # Using CTkFrame instead


class FinalTestComparisonPage(ctk.CTkFrame):
    """
    Page for comparing laser trim data with final test data.
    
    Features:
    - Select unit from database by trim date/time, model, and serial
    - Load final test Excel file
    - Compare linearity and other metrics
    - Overlay charts with export/print capabilities
    """
    
    def __init__(self, parent, main_window, **kwargs):
        """Initialize final test comparison page."""
        # Initialize parent class (CTkFrame)
        super().__init__(parent, **kwargs)
        self.main_window = main_window
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Add BasePage-like functionality
        self.is_visible = False
        self.needs_refresh = True
        self._stop_requested = False
        
        # Page-specific attributes
        self.selected_unit = None
        self.final_test_data = None
        self.comparison_results = None
        self.current_plot = None
        self.canvas = None
        
        # Thread safety
        self._comparison_lock = threading.Lock()
        
        # Window resize handling
        self._resize_job = None
        
        # Create the page
        self._create_page()
        
        # Apply hover fixes after page creation
        self.after(100, self._apply_hover_fixes)
        
        # Bind resize event
        self.bind("<Configure>", self._on_window_resize)
        
    def _create_page(self):
        """Create page content."""
        # Main scrollable container
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create sections
        self._create_header()
        self._create_unit_selection_section()
        self._create_final_test_section()
        self._create_comparison_section()
        self._create_chart_section()
        
    def _apply_hover_fixes(self):
        """Apply hover fixes to prevent glitching and shifting."""
        try:
            # Check if widget is ready
            if not self.winfo_exists():
                return
                
            # Fix hover glitches on all widgets
            fix_hover_glitches(self)
            
            # Stabilize layout to prevent shifting if container exists
            if hasattr(self, 'main_container') and self.main_container.winfo_exists():
                stabilize_layout(self.main_container)
            
            self.logger.debug("Hover fixes applied successfully")
        except Exception as e:
            # Only log debug level since this is not critical
            self.logger.debug(f"Could not apply hover fixes: {e}")
    
    def _create_header(self):
        """Create header section."""
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.pack(fill='x', pady=(0, 20))
        
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Final Test Comparison",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=15)
        
        self.subtitle_label = ctk.CTkLabel(
            self.header_frame,
            text="Compare laser trim results with final test data",
            font=ctk.CTkFont(size=12)
        )
        self.subtitle_label.pack(pady=(0, 15))
        
    def _create_unit_selection_section(self):
        """Create unit selection section."""
        # Unit Selection Frame
        self.unit_frame = ctk.CTkFrame(self.main_container)
        self.unit_frame.pack(fill='x', pady=(0, 20))
        
        # Section title
        self.unit_title = ctk.CTkLabel(
            self.unit_frame,
            text="1. Select Unit from Database",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.unit_title.pack(anchor='w', padx=20, pady=(15, 10))
        
        # Selection controls frame
        self.selection_frame = ctk.CTkFrame(self.unit_frame)
        self.selection_frame.pack(fill='x', padx=20, pady=(0, 15))
        
        # Model selection (FIRST)
        model_frame = ctk.CTkFrame(self.selection_frame)
        model_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(model_frame, text="Model Number:").pack(side='left', padx=(0, 10))
        self.model_var = ctk.StringVar()
        self.model_dropdown = ctk.CTkComboBox(
            model_frame,
            variable=self.model_var,
            values=[],
            command=self._on_model_selected,
            width=200
        )
        self.model_dropdown.pack(side='left', padx=(0, 20))
        
        # Serial selection (SECOND)
        serial_frame = ctk.CTkFrame(self.selection_frame)
        serial_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(serial_frame, text="Serial Number:").pack(side='left', padx=(0, 10))
        self.serial_var = ctk.StringVar()
        self.serial_dropdown = ctk.CTkComboBox(
            serial_frame,
            variable=self.serial_var,
            values=[],
            command=self._on_serial_selected,
            width=200,
            state="disabled"
        )
        self.serial_dropdown.pack(side='left', padx=(0, 20))
        
        # Date selection (THIRD)
        date_frame = ctk.CTkFrame(self.selection_frame)
        date_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(date_frame, text="Trim Date:").pack(side='left', padx=(0, 10))
        self.date_var = ctk.StringVar()
        self.date_dropdown = ctk.CTkComboBox(
            date_frame,
            variable=self.date_var,
            values=[],
            command=self._on_date_selected,
            width=200,
            state="disabled"
        )
        self.date_dropdown.pack(side='left', padx=(0, 20))
        
        # Load unit button
        self.load_unit_button = AnimatedButton(
            self.unit_frame,
            text="Load Selected Unit",
            command=self._load_selected_unit,
            state="disabled",
            width=200
        )
        # Hover fixes will be applied later in _apply_hover_fixes method
        self.load_unit_button.pack(pady=10)
        
        # Unit info display
        self.unit_info_frame = ctk.CTkFrame(self.unit_frame)
        self.unit_info_frame.pack(fill='x', padx=20, pady=(10, 15))
        
        self.unit_info_label = ctk.CTkLabel(
            self.unit_info_frame,
            text="No unit selected",
            font=ctk.CTkFont(size=12)
        )
        self.unit_info_label.pack(pady=10)
        
    def _create_final_test_section(self):
        """Create final test file selection section."""
        # Final Test Frame
        self.final_test_frame = ctk.CTkFrame(self.main_container)
        self.final_test_frame.pack(fill='x', pady=(0, 20))
        
        # Section title
        self.final_test_title = ctk.CTkLabel(
            self.final_test_frame,
            text="2. Select Final Test Data",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.final_test_title.pack(anchor='w', padx=20, pady=(15, 10))
        
        # File selection frame
        self.file_selection_frame = ctk.CTkFrame(self.final_test_frame)
        self.file_selection_frame.pack(fill='x', padx=20, pady=(0, 15))
        
        # File path display
        self.file_path_var = ctk.StringVar(value="No file selected")
        self.file_path_label = ctk.CTkLabel(
            self.file_selection_frame,
            textvariable=self.file_path_var,
            font=ctk.CTkFont(size=12)
        )
        self.file_path_label.pack(pady=5)
        
        # Browse button
        self.browse_button = AnimatedButton(
            self.file_selection_frame,
            text="Browse for Final Test File",
            command=self._browse_final_test_file,
            state="disabled",
            width=200
        )
        # Hover fixes will be applied later in _apply_hover_fixes method
        self.browse_button.pack(pady=10)
        
        # File info display
        self.file_info_frame = ctk.CTkFrame(self.final_test_frame)
        self.file_info_frame.pack(fill='x', padx=20, pady=(0, 15))
        
        self.file_info_label = ctk.CTkLabel(
            self.file_info_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.file_info_label.pack(pady=5)
        
        # Position alignment option
        self.alignment_frame = ctk.CTkFrame(self.final_test_frame)
        self.alignment_frame.pack(fill='x', padx=20, pady=(0, 10))
        
        self.auto_align_var = ctk.BooleanVar(value=True)
        self.auto_align_checkbox = ctk.CTkCheckBox(
            self.alignment_frame,
            text="Auto-align positions (normalize both datasets to start at 0)",
            variable=self.auto_align_var,
            font=ctk.CTkFont(size=12)
        )
        self.auto_align_checkbox.pack(pady=5)
        
        self.alignment_info_label = ctk.CTkLabel(
            self.alignment_frame,
            text="Ensures accurate comparison when datasets use different position references",
            font=ctk.CTkFont(size=10),
            text_color="gray60"
        )
        self.alignment_info_label.pack(pady=(0, 5))
        
    def _create_comparison_section(self):
        """Create comparison controls section."""
        # Comparison Frame
        self.comparison_frame = ctk.CTkFrame(self.main_container)
        self.comparison_frame.pack(fill='x', pady=(0, 20))
        
        # Section title
        self.comparison_title = ctk.CTkLabel(
            self.comparison_frame,
            text="3. Run Comparison",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.comparison_title.pack(anchor='w', padx=20, pady=(15, 10))
        
        # Compare button
        self.compare_button = AnimatedButton(
            self.comparison_frame,
            text="Compare Data",
            command=self._run_comparison,
            state="disabled",
            width=200
        )
        # Hover fixes will be applied later in _apply_hover_fixes method
        self.compare_button.pack(pady=10)
        
        # Results display
        self.results_frame = ctk.CTkFrame(self.comparison_frame)
        self.results_frame.pack(fill='x', padx=20, pady=(10, 15))
        
        self.results_label = ctk.CTkLabel(
            self.results_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.results_label.pack(pady=5)
        
    def _create_chart_section(self):
        """Create chart display section."""
        # Chart Frame
        self.chart_frame = ctk.CTkFrame(self.main_container)
        self.chart_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Section title
        self.chart_title = ctk.CTkLabel(
            self.chart_frame,
            text="4. Comparison Results",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.chart_title.pack(anchor='w', padx=20, pady=(15, 10))
        
        # Chart controls
        self.chart_controls_frame = ctk.CTkFrame(self.chart_frame)
        self.chart_controls_frame.pack(fill='x', padx=20, pady=(0, 10))
        
        # Export button
        self.export_button = AnimatedButton(
            self.chart_controls_frame,
            text="Export Chart",
            command=self._export_chart,
            state="disabled",
            width=150
        )
        # Hover fixes will be applied later in _apply_hover_fixes method
        self.export_button.pack(side='left', padx=(0, 10))
        
        # Print button
        self.print_button = AnimatedButton(
            self.chart_controls_frame,
            text="Print Chart",
            command=self._print_chart,
            state="disabled",
            width=150
        )
        # Hover fixes will be applied later in _apply_hover_fixes method
        self.print_button.pack(side='left')
        
        # Chart display area
        self.chart_display_frame = ctk.CTkFrame(self.chart_frame)
        self.chart_display_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
    def on_show(self):
        """Called when page is shown."""
        # Refresh available models
        self._refresh_models()
        
    def _refresh_models(self):
        """Refresh available models from database."""
        if not hasattr(self.main_window, 'db_manager') or not self.main_window.db_manager:
            return
            
        try:
            # Get unique models from database
            with self.main_window.db_manager.get_session() as session:
                results = session.query(
                    DBAnalysisResult.model
                ).distinct().order_by(DBAnalysisResult.model).all()
                
                models = [r.model for r in results if r.model]
                
                # Update dropdown
                self.model_dropdown.configure(values=models)
                if models:
                    self.model_dropdown.set("Select a model...")
                    self.model_dropdown.configure(state="normal")  # Enable dropdown
                else:
                    self.model_dropdown.set("No models available")
                    
        except Exception as e:
            self.logger.error(f"Error refreshing models: {e}")
            
    def _on_model_selected(self, selected_model: str):
        """Handle model selection."""
        if selected_model == "Select a model..." or selected_model == "No models available":
            return
            
        # Enable serial dropdown and refresh serials for selected model
        self.serial_dropdown.configure(state="normal")
        self._refresh_serials_for_model(selected_model)
        
        # Reset downstream selections
        self.date_dropdown.configure(state="disabled", values=[])
        self.date_dropdown.set("")
        self.load_unit_button.configure(state="disabled")
        
    def _refresh_serials_for_model(self, model: str):
        """Refresh available serials for selected model."""
        if not hasattr(self.main_window, 'db_manager') or not self.main_window.db_manager:
            return
            
        try:
            # Get unique serials for this model
            with self.main_window.db_manager.get_session() as session:
                results = session.query(
                    DBAnalysisResult.serial
                ).filter(
                    DBAnalysisResult.model == model
                ).distinct().order_by(DBAnalysisResult.serial).all()
                
                serials = [r.serial for r in results if r.serial]
                
                # Update dropdown
                self.serial_dropdown.configure(values=serials)
                if serials:
                    self.serial_dropdown.set("Select a serial...")
                else:
                    self.serial_dropdown.set("No serials found")
                    
        except Exception as e:
            self.logger.error(f"Error refreshing serials: {e}")
            
    def _on_serial_selected(self, selected_serial: str):
        """Handle serial selection."""
        if selected_serial == "Select a serial..." or selected_serial == "No serials found":
            return
            
        # Enable date dropdown and refresh dates for selected model and serial
        self.date_dropdown.configure(state="normal")
        self._refresh_dates_for_serial(self.model_var.get(), selected_serial)
        
        # Reset downstream selections
        self.load_unit_button.configure(state="disabled")
        
    def _refresh_dates_for_serial(self, model: str, serial: str):
        """Refresh available dates for selected model and serial."""
        if not hasattr(self.main_window, 'db_manager') or not self.main_window.db_manager:
            return
            
        try:
            # Get unique trim dates for this model and serial
            with self.main_window.db_manager.get_session() as session:
                results = session.query(
                    DBAnalysisResult.file_date,
                    DBAnalysisResult.id
                ).filter(
                    DBAnalysisResult.model == model,
                    DBAnalysisResult.serial == serial
                ).order_by(DBAnalysisResult.file_date.desc()).all()
                
                # Format dates with time for display
                date_options = []
                self._date_to_id_map = {}  # Store mapping of date string to record ID
                for result in results:
                    if result.file_date:
                        date_str = result.file_date.strftime("%Y-%m-%d %H:%M:%S")
                        date_options.append(date_str)
                        self._date_to_id_map[date_str] = result.id
                
                # Update dropdown
                self.date_dropdown.configure(values=date_options)
                if date_options:
                    self.date_dropdown.set("Select a date...")
                else:
                    self.date_dropdown.set("No dates found")
                    
        except Exception as e:
            self.logger.error(f"Error refreshing dates: {e}")
            
    def _on_date_selected(self, selected_date: str):
        """Handle date selection."""
        if selected_date == "Select a date..." or selected_date == "No dates found":
            return
            
        # Enable load button
        self.load_unit_button.configure(state="normal")
        
    def _load_selected_unit(self):
        """Load the selected unit from database."""
        try:
            # Parse selections
            date_str = self.date_var.get()
            model = self.model_var.get()
            serial_with_time = self.serial_var.get()
            
            # Extract serial number from selection
            serial = serial_with_time.split(" (")[0] if " (" in serial_with_time else serial_with_time
            
            # Get the record ID from the mapping
            record_id = getattr(self, '_date_to_id_map', {}).get(date_str)
            if not record_id:
                self.show_error("Invalid date selection")
                return
            
            # Load unit from database by ID with tracks eagerly loaded
            with self.main_window.db_manager.get_session() as session:
                from sqlalchemy.orm import joinedload
                result = session.query(DBAnalysisResult).options(
                    joinedload(DBAnalysisResult.tracks)
                ).filter(
                    DBAnalysisResult.id == record_id
                ).first()
                
                if result:
                    self.selected_unit = result
                    # Update unit info display
                    info_text = f"Loaded: {model} - {serial}\n"
                    if result.file_date:
                        info_text += f"Trim Date: {result.file_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    else:
                        info_text += f"Trim Date: Unknown\n"
                    info_text += f"Analysis Date: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    info_text += f"Status: {result.overall_status}"
                    self.unit_info_label.configure(text=info_text)
                    
                    # Enable final test file selection
                    self.browse_button.configure(state="normal")
                    
                    # Show success message
                    self.show_success("Unit loaded successfully")
                else:
                    self.show_error("Unit not found in database")
                    
        except Exception as e:
            self.logger.error(f"Error loading unit: {e}")
            self.show_error(f"Error loading unit: {str(e)}")
            
    def _browse_final_test_file(self):
        """Browse for final test Excel file."""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Final Test Excel File",
                filetypes=[
                    ("Excel files", "*.xlsx *.xls"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                # Validate and load the file
                self._load_final_test_file(file_path)
                
        except Exception as e:
            self.logger.error(f"Error browsing for file: {e}")
            self.show_error(f"Error selecting file: {str(e)}")
            
    def _load_final_test_file(self, file_path: str):
        """Load and validate final test Excel file."""
        try:
            # Read Excel file - try default sheet first, then first sheet
            try:
                df = pd.read_excel(file_path, sheet_name="Sheet1")
            except ValueError:
                # Sheet1 doesn't exist, use first sheet
                df = pd.read_excel(file_path, sheet_name=0)
            
            # Validate required columns
            required_columns = {
                'A': 'Measured Volts',
                'B': 'Index',
                'C': 'Theoretical Volts',
                'D': 'Error in Volts',
                'E': 'Position',
                'G': 'Linearity Spec 1',
                'H': 'Linearity Spec 2'
            }
            
            # Check if columns exist (Excel columns are 0-indexed in pandas)
            column_mapping = {
                0: 'Measured Volts',
                1: 'Index',
                2: 'Theoretical Volts',
                3: 'Error in Volts',
                4: 'Position',
                6: 'Linearity Spec 1',
                7: 'Linearity Spec 2'
            }
            
            # Validate columns exist
            if len(df.columns) < 8:
                self.show_error("Final test data is not in the proper format. Missing required columns.")
                return
                
            # Create clean dataframe with proper column names
            self.final_test_data = pd.DataFrame({
                'measured_volts': df.iloc[:, 0],
                'index': df.iloc[:, 1],
                'theoretical_volts': df.iloc[:, 2],
                'error_volts': df.iloc[:, 3],
                'position': df.iloc[:, 4],
                'linearity_spec_upper': df.iloc[:, 6],
                'linearity_spec_lower': df.iloc[:, 7]
            })
            
            # Remove any rows with NaN values
            self.final_test_data = self.final_test_data.dropna()
            
            # Check if we have data after cleaning
            if self.final_test_data.empty:
                self.show_error("No valid data found after removing empty rows.")
                self.final_test_data = None
                return
            
            # Update file info
            self.file_path_var.set(os.path.basename(file_path))
            info_text = f"Loaded {len(self.final_test_data)} data points\n"
            if len(self.final_test_data) > 0:
                info_text += f"Position range: {self.final_test_data['position'].min():.2f} to {self.final_test_data['position'].max():.2f}"
            self.file_info_label.configure(text=info_text)
            
            # Enable compare button
            if self.selected_unit:
                self.compare_button.configure(state="normal")
                
            self.show_success("Final test file loaded successfully")
            
        except ValueError as e:
            if "Sheet1" in str(e):
                self.show_error("Final test data is not in the proper format. Sheet 'Sheet1' not found.")
            else:
                self.show_error(f"Final test data is not in the proper format. {str(e)}")
        except Exception as e:
            self.logger.error(f"Error loading final test file: {e}")
            self.show_error(f"Final test data is not in the proper format. {str(e)}")
            
    def _run_comparison(self):
        """Run comparison between laser trim and final test data."""
        try:
            # Disable button during processing
            self.compare_button.configure(state="disabled", text="Comparing...")
            
            # Run comparison in background thread
            threading.Thread(target=self._comparison_worker, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"Error starting comparison: {e}")
            self.show_error(f"Error running comparison: {str(e)}")
            self.compare_button.configure(state="normal", text="Compare Data")
            
    def _comparison_worker(self):
        """Worker thread for running comparison."""
        try:
            # Thread-safe access to data
            with self._comparison_lock:
                selected_unit = self.selected_unit
                final_test_data = self.final_test_data
            
            if not selected_unit or final_test_data is None:
                raise ValueError("Missing data for comparison")
            
            # Get laser trim data
            trim_data = self._extract_trim_data_safe(selected_unit)
            
            # Perform comparison
            comparison_results = self._compare_linearity(trim_data, final_test_data)
            
            # Update UI in main thread
            self.after(0, self._display_comparison_results, comparison_results)
            
        except Exception as e:
            self.after(0, self._comparison_error, str(e))
            
    def _extract_trim_data_safe(self, selected_unit) -> pd.DataFrame:
        """Extract linearity data from laser trim results (thread-safe)."""
        try:
            # Get track data from the related tracks (loaded via relationship)
            if hasattr(selected_unit, 'tracks') and selected_unit.tracks:
                # Get the first/primary track
                primary_track = selected_unit.tracks[0]
                
                # Extract position and error data from the track
                positions = primary_track.position_data
                errors = primary_track.error_data
                
                if positions and errors and len(positions) == len(errors):
                    # Convert JSON strings to lists if needed
                    if isinstance(positions, str):
                        import json
                        positions = json.loads(positions)
                    if isinstance(errors, str):
                        import json
                        errors = json.loads(errors)
                    
                    # Using actual database raw data (from trimmed sheets)
                    self.logger.info(f"Using actual raw data from database (trimmed): {len(positions)} points")
                    self._using_database_data = True
                    self._data_is_trimmed = True
                    
                    # Get optimal offset for information
                    optimal_offset = primary_track.optimal_offset or 0.0
                    self.logger.info(f"Optimal offset from database: {optimal_offset:.6f}")
                    
                    # The errors from database are raw voltage errors (measured - theoretical)
                    # These need to be converted to linearity errors by removing the best-fit line
                    if hasattr(primary_track, 'final_linearity_error_shifted') and hasattr(primary_track, 'final_linearity_error_raw'):
                        # Log which type of data we have
                        self.logger.info(f"Final linearity error (raw): {primary_track.final_linearity_error_raw:.6f} V")
                        self.logger.info(f"Final linearity error (shifted): {primary_track.final_linearity_error_shifted:.6f} V")
                    
                    # Calculate linearity errors from raw voltage errors
                    # First fit a line to the raw errors to find the trend
                    positions_array = np.array(positions)
                    errors_array = np.array(errors)
                    
                    # Fit a line: error = slope * position + intercept
                    # This represents the systematic error trend
                    coeffs = np.polyfit(positions_array, errors_array, 1)
                    slope, intercept = coeffs
                    
                    # Calculate the trend line values
                    trend_line = slope * positions_array + intercept
                    
                    # Linearity errors are deviations from this trend
                    linearity_errors = errors_array - trend_line
                    
                    # Now apply the optimal offset to minimize max deviation
                    errors_shifted = linearity_errors - optimal_offset
                    
                    # Log the transformation
                    self.logger.info(f"Raw voltage error range: [{min(errors):.6f}, {max(errors):.6f}] V")
                    self.logger.info(f"Best-fit line: slope={slope:.6e}, intercept={intercept:.6f}")
                    self.logger.info(f"Linearity error range (detrended): [{linearity_errors.min():.6f}, {linearity_errors.max():.6f}] V")
                    self.logger.info(f"Final shifted error range: [{errors_shifted.min():.6f}, {errors_shifted.max():.6f}] V")
                    
                    # For laser trim data, we have positions and linearity errors
                    # Get the linearity spec from the track (this is the symmetric spec)
                    linearity_spec = primary_track.linearity_spec or 0.0
                    self.logger.info(f"Laser trim linearity spec: ±{linearity_spec:.6f} V")
                    
                    # The shifted errors should be the linearity errors
                    # Verify the final linearity error matches what we calculated
                    calc_max_error = np.max(np.abs(errors_shifted))
                    if hasattr(primary_track, 'final_linearity_error_shifted'):
                        db_max_error = primary_track.final_linearity_error_shifted
                        self.logger.info(f"Calculated max linearity error: {calc_max_error:.6f} V")
                        self.logger.info(f"Database max linearity error: {db_max_error:.6f} V")
                        if abs(calc_max_error - db_max_error) > 0.01:
                            self.logger.warning(f"Mismatch between calculated and database linearity errors!")
                    
                    trim_df = pd.DataFrame({
                        'position': positions,
                        'error': errors_shifted.tolist(),  # Convert numpy array to list
                        'linearity_spec': linearity_spec  # Add spec to dataframe
                    })
                    
                    # Log some debug info
                    self.logger.info(f"Trim data range - Position: [{min(positions):.3f}, {max(positions):.3f}]")
                    self.logger.info(f"Trim data range - Error: [{min(errors_shifted):.6f}, {max(errors_shifted):.6f}]")
                    
                    return trim_df
                else:
                    # Fallback: create synthetic data if raw data not available
                    self.logger.warning("Raw position/error data not available, using synthetic representation from summary stats")
                    self._using_database_data = False
                    self._data_is_trimmed = True  # Summary stats are from trimmed data
                    
                    # Show warning to user
                    self.after(0, lambda: messagebox.showwarning(
                        "Synthetic Data Warning",
                        "This unit does not have raw position/error data in the database.\n\n"
                        "The chart will show a synthetic representation based on summary statistics, "
                        "which may not accurately reflect the actual error profile.\n\n"
                        "To see accurate data:\n"
                        "1. Locate the original Excel file for this unit\n"
                        "2. Re-process it through the Single File or Batch Processing page\n"
                        "3. The database will be updated with raw data\n"
                        "4. Return to this page for accurate comparison"
                    ))
                    
                    travel_length = primary_track.travel_length or 100
                    positions = np.linspace(0, travel_length, 100).tolist()
                    max_error = primary_track.final_linearity_error_raw or 0
                    
                    # Use the final linearity error (shifted) from database
                    final_error = primary_track.final_linearity_error_shifted or 0
                    
                    # Generate synthetic error pattern
                    errors = []
                    for i, pos in enumerate(positions):
                        phase = (pos / travel_length) * 2 * np.pi
                        error = final_error * np.sin(phase) * (1 + 0.3 * np.sin(phase * 3))
                        errors.append(error)
                    
                    # Get the linearity spec from the track (this is the symmetric spec)
                    linearity_spec = primary_track.linearity_spec or 0.0
                    
                    trim_df = pd.DataFrame({
                        'position': positions,
                        'error': errors,
                        'linearity_spec': linearity_spec  # Add spec to dataframe
                    })
                    return trim_df
                            
            raise ValueError("Could not extract trim data from database result - no track data found")
            
        except Exception as e:
            self.logger.error(f"Error extracting trim data: {e}")
            raise ValueError(f"Failed to extract trim data: {str(e)}")
            
    def _compare_linearity(self, trim_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare linearity between trim and test data."""
        results = {}
        
        # Get position ranges (keep in original units, e.g., inches)
        trim_positions = trim_data['position'].values.copy()  # Make a copy to avoid modifying original
        test_positions = test_data['position'].values.copy()
        
        # Log raw data for debugging
        self.logger.info(f"Raw trim positions - first 5: {trim_positions[:5]}, last 5: {trim_positions[-5:]}")
        self.logger.info(f"Raw test positions - first 5: {test_positions[:5]}, last 5: {test_positions[-5:]}")
        
        # Calculate travel lengths
        trim_length = trim_positions.max() - trim_positions.min()
        test_length = test_positions.max() - test_positions.min()
        
        self.logger.info(f"Original trim position range: [{trim_positions.min():.3f}, {trim_positions.max():.3f}], length: {trim_length:.3f}")
        self.logger.info(f"Original test position range: [{test_positions.min():.3f}, {test_positions.max():.3f}], length: {test_length:.3f}")
        
        # ROBUST POSITION ALIGNMENT STRATEGY
        alignment_note = "No alignment performed"
        
        # Check if auto-alignment is enabled
        if self.auto_align_var.get():
            # Handle various position reference scenarios:
            # 1. Both start at 0 - no adjustment needed
            # 2. Trim centered, test at 0 - shift trim to start at 0
            # 3. Test centered, trim at 0 - shift test to start at 0
            # 4. Both centered - shift both to start at 0
            # 5. Different arbitrary offsets - normalize both to start at 0
            
            alignment_note = "No alignment needed"
            
            # Check if lengths are significantly different (more than 5%)
            length_ratio = trim_length / test_length if test_length > 0 else 1
            if abs(length_ratio - 1.0) > 0.05:
                self.logger.warning(f"Travel lengths differ by {abs(length_ratio - 1.0) * 100:.1f}% - may be different units or setups")
                # Add warning to alignment note
                alignment_note = f"WARNING: Length mismatch {abs(length_ratio - 1.0) * 100:.1f}%"
            
            # Normalize both datasets to start at 0 for consistent comparison
            trim_min = trim_positions.min()
            test_min = test_positions.min()
            
            # Shift trim data if needed
            if abs(trim_min) > 0.001:  # Not already at 0
                trim_positions = trim_positions - trim_min
                self.logger.info(f"Shifted trim positions by {-trim_min:.3f} to start at 0")
                if alignment_note.startswith("WARNING"):
                    alignment_note += f", Trim shifted by {-trim_min:.3f}"
                else:
                    alignment_note = f"Trim shifted by {-trim_min:.3f}"
            
            # Shift test data if needed
            if abs(test_min) > 0.001:  # Not already at 0
                test_positions = test_positions - test_min
                self.logger.info(f"Shifted test positions by {-test_min:.3f} to start at 0")
                if alignment_note == "No alignment needed":
                    alignment_note = f"Test shifted by {-test_min:.3f}"
                elif not alignment_note.startswith("WARNING"):
                    alignment_note += f", Test shifted by {-test_min:.3f}"
                else:
                    alignment_note += f", Test shifted by {-test_min:.3f}"
        else:
            self.logger.info("Position alignment disabled by user")
            alignment_note = "Position alignment disabled"
            
            # Still check for length mismatch as a warning
            length_ratio = trim_length / test_length if test_length > 0 else 1
            if abs(length_ratio - 1.0) > 0.05:
                self.logger.warning(f"Travel lengths differ by {abs(length_ratio - 1.0) * 100:.1f}% - alignment recommended")
        
        # Store alignment info for display
        self._position_alignment_note = alignment_note
        
        # Log final aligned ranges
        self.logger.info(f"Aligned trim position range: [{trim_positions.min():.3f}, {trim_positions.max():.3f}]")
        self.logger.info(f"Aligned test position range: [{test_positions.min():.3f}, {test_positions.max():.3f}]")
        
        # Set common range
        min_pos = 0.0
        max_pos = max(trim_positions.max(), test_positions.max())
        
        self.logger.info(f"Common position range for comparison: [{min_pos:.3f}, {max_pos:.3f}]")
        
        common_positions = np.linspace(min_pos, max_pos, 200)  # 200 points for smooth curves
        
        # Log first few position values to check alignment
        self.logger.info(f"First 5 trim positions: {trim_positions[:5]}")
        self.logger.info(f"First 5 test positions: {test_positions[:5]}")
        self.logger.info(f"First 5 common positions: {common_positions[:5]}")
        
        # Check if positions are in same units (e.g., both in inches)
        trim_pos_diff = np.diff(trim_positions).mean() if len(trim_positions) > 1 else 0
        test_pos_diff = np.diff(test_positions).mean() if len(test_positions) > 1 else 0
        self.logger.info(f"Average position spacing - Trim: {trim_pos_diff:.6f}, Test: {test_pos_diff:.6f}")
        
        # Interpolate trim error data (already in volts)
        # Use fill_value='extrapolate' for scipy interp1d for better extrapolation
        from scipy.interpolate import interp1d
        
        # Create interpolation functions
        trim_interp_func = interp1d(trim_positions, trim_data['error'].values, 
                                   kind='linear', bounds_error=False, 
                                   fill_value='extrapolate')
        trim_error_interp = trim_interp_func(common_positions)
        
        # Calculate test error from measured and theoretical
        test_error = test_data['error_volts'].values
        test_interp_func = interp1d(test_positions, test_error,
                                   kind='linear', bounds_error=False,
                                   fill_value='extrapolate')
        test_error_interp = test_interp_func(common_positions)
        
        # Log error ranges
        self.logger.info(f"Trim error range: [{trim_error_interp.min():.6f}, {trim_error_interp.max():.6f}] V")
        self.logger.info(f"Test error range: [{test_error_interp.min():.6f}, {test_error_interp.max():.6f}] V")
        
        # Check for data integrity
        if np.any(np.isnan(trim_error_interp)) or np.any(np.isinf(trim_error_interp)):
            self.logger.warning(f"Found NaN or Inf in trim error interpolation!")
        if np.any(np.isnan(test_error_interp)) or np.any(np.isinf(test_error_interp)):
            self.logger.warning(f"Found NaN or Inf in test error interpolation!")
        
        # Handle spec limits for both datasets
        # For laser trim data, use symmetric spec limits
        if 'linearity_spec' in trim_data.columns:
            trim_spec = trim_data['linearity_spec'].values[0]  # Single value for all positions
            # Create symmetric spec limits arrays
            trim_spec_upper = np.full_like(common_positions, trim_spec)
            trim_spec_lower = np.full_like(common_positions, -trim_spec)
            self.logger.info(f"Using laser trim symmetric spec: ±{trim_spec:.6f} V")
        else:
            # Fallback if spec not available
            trim_spec_upper = None
            trim_spec_lower = None
            self.logger.warning("No laser trim spec found in data")
        
        # For final test data, interpolate the spec limits (they can vary with position)
        test_spec_upper_func = interp1d(test_positions, test_data['linearity_spec_upper'].values,
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
        test_spec_upper_interp = test_spec_upper_func(common_positions)
        
        test_spec_lower_func = interp1d(test_positions, test_data['linearity_spec_lower'].values,
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
        test_spec_lower_interp = test_spec_lower_func(common_positions)
        
        # Calculate differences
        error_diff = trim_error_interp - test_error_interp
        
        # Calculate statistics
        results['position'] = common_positions
        results['trim_error'] = trim_error_interp
        results['test_error'] = test_error_interp
        results['error_diff'] = error_diff
        # Store spec limits for both datasets
        if trim_spec_upper is not None:
            results['trim_spec_upper'] = trim_spec_upper
            results['trim_spec_lower'] = trim_spec_lower
        results['test_spec_upper'] = test_spec_upper_interp
        results['test_spec_lower'] = test_spec_lower_interp
        
        # Summary statistics
        results['stats'] = {
            'mean_error_diff': np.mean(error_diff),
            'std_error_diff': np.std(error_diff),
            'max_error_diff': np.max(np.abs(error_diff)),
            'trim_max_error': np.max(np.abs(trim_error_interp)),
            'test_max_error': np.max(np.abs(test_error_interp))
        }
        
        # Store original position ranges for chart display
        results['trim_position_range'] = (trim_positions.min(), trim_positions.max())
        results['test_position_range'] = (test_positions.min(), test_positions.max())
        
        # Store for later use
        self.comparison_results = results
        
        return results
        
    def _display_comparison_results(self, results: Dict[str, Any]):
        """Display comparison results."""
        try:
            # Update button
            self.compare_button.configure(state="normal", text="Compare Data")
            
            # Display statistics
            stats = results['stats']
            results_text = "Comparison Results:\n\n"
            results_text += f"Laser Trim Max Error: {stats['trim_max_error']:.6f} V\n"
            results_text += f"Final Test Max Error: {stats['test_max_error']:.6f} V\n\n"
            results_text += f"Mean Error Difference: {stats['mean_error_diff']:.6f} V\n"
            results_text += f"Std Dev Error Difference: {stats['std_error_diff']:.6f} V\n"
            results_text += f"Max Error Difference: {stats['max_error_diff']:.6f} V"
            
            self.results_label.configure(text=results_text)
            
            # Create comparison chart
            self._create_comparison_chart(results)
            
            # Enable export/print buttons
            self.export_button.configure(state="normal")
            self.print_button.configure(state="normal")
            
            self.show_success("Comparison completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error displaying results: {e}")
            self.show_error(f"Error displaying results: {str(e)}")
            
    def _comparison_error(self, error_msg: str):
        """Handle comparison error."""
        self.compare_button.configure(state="normal", text="Compare Data")
        self.show_error(f"Comparison failed: {error_msg}")
        
    def _create_comparison_chart(self, results: Dict[str, Any]):
        """Create comparison overlay chart focused on linearity error."""
        try:
            # Clear previous chart and figure
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
            if self.current_plot:
                plt.close(self.current_plot)
                self.current_plot = None
                
            # Get current window size to make chart responsive
            if hasattr(self.chart_display_frame, 'winfo_width'):
                frame_width = self.chart_display_frame.winfo_width()
                frame_height = self.chart_display_frame.winfo_height()
                # Default sizes if window not yet drawn
                if frame_width <= 1:
                    frame_width = 1200
                if frame_height <= 1:
                    frame_height = 800
            else:
                frame_width = 1200
                frame_height = 800
                
            # Calculate figure size based on frame size (leave some margin)
            fig_width = max(10, (frame_width - 40) / 100)  # Convert pixels to inches at 100 DPI
            fig_height = max(8, (frame_height - 40) / 100)
            
            # Create figure with responsive size
            fig = Figure(figsize=(fig_width, fig_height), dpi=100)
            
            # Create GridSpec for better control
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.15)
            
            # Main plot on top (66% of figure height)
            ax_main = fig.add_subplot(gs[0, 0])
            
            # Plot linearity errors with thicker lines
            # Ensure we're plotting valid data
            positions = results['position']
            trim_errors_mv = results['trim_error'] * 1000  # Convert to mV
            test_errors_mv = results['test_error'] * 1000  # Convert to mV
            
            # Log data being plotted
            self.logger.info(f"Plotting - Position range: [{positions.min():.3f}, {positions.max():.3f}]")
            self.logger.info(f"Plotting - Trim error range: [{trim_errors_mv.min():.3f}, {trim_errors_mv.max():.3f}] mV")
            self.logger.info(f"Plotting - Test error range: [{test_errors_mv.min():.3f}, {test_errors_mv.max():.3f}] mV")
            
            ax_main.plot(positions, trim_errors_mv, 'b-', 
                        label='Laser Trim', linewidth=2.5, alpha=0.9)
            ax_main.plot(positions, test_errors_mv, 'r--', 
                        label='Final Test', linewidth=2.5, alpha=0.9)
            
            # Add vertical lines to indicate actual data ranges
            trim_min, trim_max = results['trim_position_range']
            test_min, test_max = results['test_position_range']
            min_pos = results['position'].min()
            max_pos = results['position'].max()
            
            # Add subtle shading for extrapolated regions
            if trim_min > min_pos:
                ax_main.axvspan(min_pos, trim_min, alpha=0.05, color='blue', label='_nolegend_')
            if trim_max < max_pos:
                ax_main.axvspan(trim_max, max_pos, alpha=0.05, color='blue', label='_nolegend_')
            if test_min > min_pos:
                ax_main.axvspan(min_pos, test_min, alpha=0.05, color='red', label='_nolegend_')
            if test_max < max_pos:
                ax_main.axvspan(test_max, max_pos, alpha=0.05, color='red', label='_nolegend_')
            
            # Add spec lines for both datasets
            # Plot laser trim spec limits (if available)
            if 'trim_spec_upper' in results and 'trim_spec_lower' in results:
                trim_spec_upper = results['trim_spec_upper'] * 1000  # Convert to mV
                trim_spec_lower = results['trim_spec_lower'] * 1000
                
                # Plot laser trim spec limits (symmetric, blue)
                ax_main.plot(results['position'], trim_spec_upper, 'b:', linewidth=2, 
                            alpha=0.7, label='Trim Spec (±)')
                ax_main.plot(results['position'], trim_spec_lower, 'b:', linewidth=2, 
                            alpha=0.7, label='_nolegend_')
                
                # Shade trim spec region
                ax_main.fill_between(results['position'], trim_spec_lower, trim_spec_upper, 
                                    alpha=0.05, color='blue')
            
            # Plot final test spec limits
            if 'test_spec_upper' in results and 'test_spec_lower' in results:
                test_spec_upper = results['test_spec_upper'] * 1000  # Convert to mV
                test_spec_lower = results['test_spec_lower'] * 1000
                
                # Plot final test spec limits (can be asymmetric, green)
                ax_main.plot(results['position'], test_spec_upper, 'g:', linewidth=2, 
                            alpha=0.7, label='Test Spec')
                ax_main.plot(results['position'], test_spec_lower, 'g:', linewidth=2, 
                            alpha=0.7, label='_nolegend_')
                
                # Shade test spec region
                ax_main.fill_between(results['position'], test_spec_lower, test_spec_upper, 
                                    alpha=0.05, color='green')
            
            # Add zero line
            ax_main.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Formatting
            ax_main.set_xlabel('Position (inches)', fontsize=12)
            ax_main.set_ylabel('Linearity Error (mV)', fontsize=12)
            
            # Add data source indicator to title
            if getattr(self, '_using_database_data', False):
                data_source = "(Using Actual Raw Data)"
                title_color = 'black'
            else:
                data_source = "(Using Synthetic Data - Reprocess File for Accuracy)"
                title_color = 'darkred'
            ax_main.set_title(f'Linearity Error Comparison: Laser Trim vs Final Test {data_source}', 
                             fontsize=14, fontweight='bold', pad=15, color=title_color)
            ax_main.legend(loc='best', framealpha=0.9, fontsize=11)
            ax_main.grid(True, alpha=0.3, linestyle='--')
            
            # Set y-axis to show reasonable range
            trim_error_range = np.max(np.abs(results['trim_error'])) * 1000
            test_error_range = np.max(np.abs(results['test_error'])) * 1000
            max_error = max(trim_error_range, test_error_range) * 1.2
            
            # Consider spec limits for y-axis range
            if 'trim_spec_upper' in results and results['trim_spec_upper'] is not None:
                trim_spec_max = np.max(np.abs(results['trim_spec_upper'] * 1000))
                max_error = max(max_error, trim_spec_max * 1.1)
            
            if 'test_spec_upper' in results and results['test_spec_upper'] is not None:
                test_spec_max = max(np.max(np.abs(results['test_spec_upper'] * 1000)), 
                                   np.max(np.abs(results['test_spec_lower'] * 1000)))
                max_error = max(max_error, test_spec_max * 1.1)
            
            # Ensure a minimum range for visibility
            max_error = max(max_error, 50)  # At least ±50 mV
            
            ax_main.set_ylim(-max_error, max_error)
            
            # Set x-axis limits explicitly
            ax_main.set_xlim(positions.min(), positions.max())
            
            # Statistics text area below chart
            ax_stats = fig.add_subplot(gs[1, 0])
            ax_stats.axis('off')
            
            # Calculate statistics
            stats = results['stats']
            
            # Build statistics text with better formatting
            stats_text = "COMPARISON STATISTICS\n" + "="*25 + "\n\n"
            
            # Convert to mV for display
            stats_text += "Linearity Error Stats:\n"
            stats_text += "-"*25 + "\n"
            
            # Laser Trim stats
            trim_max_error = np.max(np.abs(results['trim_error'])) * 1000
            trim_mean_error = np.mean(results['trim_error']) * 1000
            trim_std_error = np.std(results['trim_error']) * 1000
            
            stats_text += "Laser Trim (Final):\n"
            stats_text += f"  Max Error: {trim_max_error:.2f} mV\n"
            stats_text += f"  Mean Error: {trim_mean_error:.2f} mV\n"
            stats_text += f"  Std Dev: {trim_std_error:.2f} mV\n\n"
            
            # Final Test stats
            test_max_error = np.max(np.abs(results['test_error'])) * 1000
            test_mean_error = np.mean(results['test_error']) * 1000
            test_std_error = np.std(results['test_error']) * 1000
            
            stats_text += "Final Test:\n"
            stats_text += f"  Max Error: {test_max_error:.2f} mV\n"
            stats_text += f"  Mean Error: {test_mean_error:.2f} mV\n"
            stats_text += f"  Std Dev: {test_std_error:.2f} mV\n\n"
            
            # Difference stats
            stats_text += "Differences:\n"
            stats_text += "-"*25 + "\n"
            stats_text += f"Mean Diff: {stats['mean_error_diff']*1000:.2f} mV\n"
            stats_text += f"Max Diff: {stats['max_error_diff']*1000:.2f} mV\n"
            stats_text += f"Std Dev: {stats['std_error_diff']*1000:.2f} mV\n\n"
            
            
            # Create a more compact horizontal layout
            # Format statistics in a single row format for better space usage
            compact_stats = []
            
            # Laser Trim stats
            compact_stats.append(f"Laser Trim: Max={trim_max_error:.2f}mV, Mean={trim_mean_error:.2f}mV, StdDev={trim_std_error:.2f}mV")
            
            # Final Test stats
            compact_stats.append(f"Final Test: Max={test_max_error:.2f}mV, Mean={test_mean_error:.2f}mV, StdDev={test_std_error:.2f}mV")
            
            # Differences
            compact_stats.append(f"Differences: Mean={stats['mean_error_diff']*1000:.2f}mV, Max={stats['max_error_diff']*1000:.2f}mV, StdDev={stats['std_error_diff']*1000:.2f}mV")
            
            # Spec compliance (check against respective specs)
            trim_status = "N/A"
            test_status = "N/A"
            
            # Check trim data against trim spec
            if 'trim_spec_upper' in results and 'trim_spec_lower' in results:
                trim_errors_mv = results['trim_error'] * 1000
                trim_spec_upper_mv = results['trim_spec_upper'] * 1000
                trim_spec_lower_mv = results['trim_spec_lower'] * 1000
                trim_in_spec = np.all((trim_errors_mv >= trim_spec_lower_mv) & (trim_errors_mv <= trim_spec_upper_mv))
                trim_status = "PASS" if trim_in_spec else "FAIL"
            
            # Check test data against test spec  
            if 'test_spec_upper' in results and 'test_spec_lower' in results:
                test_errors_mv = results['test_error'] * 1000
                test_spec_upper_mv = results['test_spec_upper'] * 1000
                test_spec_lower_mv = results['test_spec_lower'] * 1000
                test_in_spec = np.all((test_errors_mv >= test_spec_lower_mv) & (test_errors_mv <= test_spec_upper_mv))
                test_status = "PASS" if test_in_spec else "FAIL"
                
            compact_stats.append(f"Spec Compliance: Laser Trim={trim_status}, Final Test={test_status}")
            
            # Display statistics with proper spacing
            # Calculate vertical spacing based on number of lines
            num_lines = len(compact_stats)
            y_spacing = 0.9 / (num_lines + 1)  # Leave space at top and bottom
            
            for i, stat_line in enumerate(compact_stats):
                y_pos = 0.95 - (i + 1) * y_spacing
                ax_stats.text(0.5, y_pos, stat_line, 
                             transform=ax_stats.transAxes, fontsize=8, 
                             verticalalignment='center', horizontalalignment='center',
                             fontfamily='monospace', wrap=True)
            
            # Add unit info at bottom center
            if hasattr(self, 'selected_unit') and self.selected_unit:
                unit_info = f"Unit: {self.selected_unit.model} - {self.selected_unit.serial}"
                if self.selected_unit.file_date:
                    unit_info += f"  |  Trim Date: {self.selected_unit.file_date.strftime('%Y-%m-%d %H:%M')}"
                
                # Add data source confirmation
                if hasattr(self.selected_unit, 'tracks') and self.selected_unit.tracks:
                    track = self.selected_unit.tracks[0]
                    if hasattr(track, 'trimmed_resistance') and track.trimmed_resistance:
                        unit_info += f"  |  Trimmed R: {track.trimmed_resistance:.2f} Ω"
                
                unit_info += f"  |  Data: {'TRM/Lin Error sheets' if getattr(self, '_data_is_trimmed', True) else 'Unknown'}"
                
                # Add position alignment info
                if hasattr(self, '_position_alignment_note'):
                    unit_info += f"  |  Position Alignment: {self._position_alignment_note}"
                
                # Position unit info below the statistics
                ax_stats.text(0.5, 0.05, unit_info, transform=ax_stats.transAxes, 
                             fontsize=7, verticalalignment='bottom', horizontalalignment='center', 
                             style='italic', wrap=True)
            
            # Adjust layout - more space for chart, less for stats
            fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05, hspace=0.25)
            
            # Create canvas
            self.canvas = FigureCanvasTkAgg(fig, master=self.chart_display_frame)
            self.canvas.draw()
            
            # Add navigation toolbar for zoom/pan functionality
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar = NavigationToolbar2Tk(self.canvas, self.chart_display_frame)
            toolbar.update()
            
            # Pack canvas and toolbar
            self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
            toolbar.pack(side='bottom', fill='x')
            
            # Add custom zoom controls
            self._add_zoom_controls(ax_main)
            
            # Store current plot for export
            self.current_plot = fig
            
        except Exception as e:
            self.logger.error(f"Error creating chart: {e}")
            self.show_error(f"Error creating chart: {str(e)}")
            
    def _export_chart(self):
        """Export comparison chart."""
        if not self.current_plot:
            self.show_error("No chart to export")
            return
            
        try:
            # Ask for file location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                # Save the plot
                self.current_plot.savefig(file_path, dpi=300, bbox_inches='tight')
                self.show_success(f"Chart exported to {os.path.basename(file_path)}")
                
        except Exception as e:
            self.logger.error(f"Error exporting chart: {e}")
            self.show_error(f"Error exporting chart: {str(e)}")
            
    def _print_chart(self):
        """Print comparison chart."""
        if not self.current_plot:
            self.show_error("No chart to print")
            return
            
        try:
            # Create a temporary file for printing
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                self.current_plot.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
                
                # Use system print dialog with better error handling
                try:
                    if os.name == 'nt':  # Windows
                        os.startfile(tmp_file.name, 'print')
                    else:  # macOS and Linux
                        # Check if lpr is available
                        if os.system('which lpr > /dev/null 2>&1') == 0:
                            os.system(f'lpr {tmp_file.name}')
                        else:
                            self.show_error("Print command (lpr) not found. Please install CUPS printing system.")
                            return
                finally:
                    # Clean up temp file after a delay
                    self.after(5000, lambda: os.unlink(tmp_file.name) if os.path.exists(tmp_file.name) else None)
                    
                self.show_success("Chart sent to printer")
                
        except Exception as e:
            self.logger.error(f"Error printing chart: {e}")
            self.show_error(f"Error printing chart: {str(e)}")
            
    def _add_zoom_controls(self, ax):
        """Add keyboard shortcuts for zooming."""
        def on_key(event):
            """Handle keyboard events for zooming."""
            if event.key == '=':  # Zoom in on Y-axis
                ylim = ax.get_ylim()
                center = (ylim[0] + ylim[1]) / 2
                span = (ylim[1] - ylim[0]) * 0.8  # Reduce span by 20%
                ax.set_ylim(center - span/2, center + span/2)
                self.canvas.draw()
            elif event.key == '-':  # Zoom out on Y-axis
                ylim = ax.get_ylim()
                center = (ylim[0] + ylim[1]) / 2
                span = (ylim[1] - ylim[0]) * 1.25  # Increase span by 25%
                ax.set_ylim(center - span/2, center + span/2)
                self.canvas.draw()
            elif event.key == 'r':  # Reset zoom
                # Reset to original limits
                trim_error_range = np.max(np.abs(self.comparison_results['trim_error'])) * 1000
                test_error_range = np.max(np.abs(self.comparison_results['test_error'])) * 1000
                max_error = max(trim_error_range, test_error_range) * 1.2
                if 'spec_upper' in self.comparison_results and 'spec_lower' in self.comparison_results:
                    spec_max = max(np.max(np.abs(self.comparison_results['spec_upper'] * 1000)), 
                                  np.max(np.abs(self.comparison_results['spec_lower'] * 1000)))
                    max_error = max(max_error, spec_max * 1.1)
                ax.set_ylim(-max_error, max_error)
                self.canvas.draw()
            elif event.key == 'h':  # Help
                help_text = "Zoom Controls:\n" \
                           "+ or = : Zoom in (Y-axis)\n" \
                           "- : Zoom out (Y-axis)\n" \
                           "r : Reset zoom\n" \
                           "Use toolbar below for pan/zoom"
                messagebox.showinfo("Zoom Controls", help_text)
        
        # Connect key press event
        self.canvas.mpl_connect('key_press_event', on_key)
            
    def refresh(self):
        """Refresh page data."""
        # Refresh dates based on current selection
        if hasattr(self, 'model_var') and hasattr(self, 'serial_var'):
            model = self.model_var.get()
            serial = self.serial_var.get()
            if model and serial:
                self._refresh_dates_for_serial(model, serial)
        
    def show(self):
        """Show the page using grid layout."""
        self.grid(row=0, column=0, sticky="nsew")
        self.is_visible = True
        
        # Refresh if needed
        if self.needs_refresh:
            self.refresh()
            self.needs_refresh = False
            
        if hasattr(self, 'on_show'):
            self.on_show()
        
    def hide(self):
        """Hide the page."""
        self.grid_remove()
        self.is_visible = False
        
    def show_success(self, message: str):
        """Show success message."""
        try:
            # Create success dialog
            from tkinter import messagebox
            messagebox.showinfo("Success", message)
        except Exception as e:
            self.logger.error(f"Error showing success message: {e}")
            
    def show_error(self, message: str):
        """Show error message."""
        try:
            # Create error dialog
            from tkinter import messagebox
            messagebox.showerror("Error", message)
        except Exception as e:
            self.logger.error(f"Error showing error message: {e}")
            
    def _on_window_resize(self, event=None):
        """Handle window resize events."""
        # Cancel any pending resize job
        if self._resize_job:
            self.after_cancel(self._resize_job)
        
        # Schedule chart redraw after resize stops (debounce)
        if self.comparison_results and self.canvas:
            self._resize_job = self.after(300, self._redraw_chart)
    
    def _redraw_chart(self):
        """Redraw the chart with current window size."""
        if self.comparison_results:
            try:
                self._create_comparison_chart(self.comparison_results)
            except Exception as e:
                self.logger.error(f"Error redrawing chart: {e}")
                
    def _create_error_page(self, error_msg: str):
        """Create an error display page."""
        error_frame = ctk.CTkFrame(self)
        error_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        error_label = ctk.CTkLabel(
            error_frame,
            text=f"Error initializing {self.__class__.__name__}",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="red"
        )
        error_label.pack(pady=(20, 10))
        
        detail_label = ctk.CTkLabel(
            error_frame,
            text=error_msg,
            font=ctk.CTkFont(size=14),
            wraplength=600
        )
        detail_label.pack(pady=10)
    
    def cleanup(self):
        """Clean up resources when page is destroyed."""
        # Destroy dropdown menus to prevent "No more menus can be allocated" error
        combo_widgets = ['model_combo', 'part_number_combo', 'serial_combo']
        
        for widget_name in combo_widgets:
            if hasattr(self, widget_name):
                try:
                    widget = getattr(self, widget_name)
                    if hasattr(widget, '_dropdown_menu'):
                        widget._dropdown_menu.destroy()
                    widget.destroy()
                except Exception:
                    pass
        
        # Call parent cleanup if it exists
        if hasattr(super(), 'cleanup'):
            super().cleanup()