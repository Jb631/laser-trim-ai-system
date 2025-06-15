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
from laser_trim_analyzer.database.models import AnalysisResult
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
        
        # Create the page
        self._create_page()
        
        # Apply hover fixes after page creation
        self.after(100, self._apply_hover_fixes)
        
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
            # Fix hover glitches on all widgets
            fix_hover_glitches(self)
            
            # Stabilize layout to prevent shifting
            stabilize_layout(self.main_container)
            
            self.logger.debug("Hover fixes applied successfully")
        except Exception as e:
            self.logger.warning(f"Failed to apply hover fixes: {e}")
    
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
        
        # Date selection
        date_frame = ctk.CTkFrame(self.selection_frame)
        date_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(date_frame, text="Trim Date:").pack(side='left', padx=(0, 10))
        self.date_var = ctk.StringVar()
        self.date_dropdown = ctk.CTkComboBox(
            date_frame,
            variable=self.date_var,
            values=[],
            command=self._on_date_selected,
            width=200
        )
        self.date_dropdown.pack(side='left', padx=(0, 20))
        
        # Model selection
        model_frame = ctk.CTkFrame(self.selection_frame)
        model_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(model_frame, text="Model Number:").pack(side='left', padx=(0, 10))
        self.model_var = ctk.StringVar()
        self.model_dropdown = ctk.CTkComboBox(
            model_frame,
            variable=self.model_var,
            values=[],
            command=self._on_model_selected,
            width=200,
            state="disabled"
        )
        self.model_dropdown.pack(side='left', padx=(0, 20))
        
        # Serial selection
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
        # Refresh available dates
        self._refresh_dates()
        
    def _refresh_dates(self):
        """Refresh available dates from database."""
        if not hasattr(self.main_window, 'db_manager') or not self.main_window.db_manager:
            return
            
        try:
            # Get unique dates from database
            with self.main_window.db_manager.get_session() as session:
                results = session.query(
                    AnalysisResult.timestamp
                ).distinct().order_by(AnalysisResult.timestamp.desc()).all()
                
                # Format dates for display
                dates = []
                for result in results:
                    if result.timestamp:
                        date_str = result.timestamp.strftime("%Y-%m-%d")
                        if date_str not in dates:
                            dates.append(date_str)
                
                # Update dropdown
                self.date_dropdown.configure(values=dates)
                if dates:
                    self.date_dropdown.set("Select a date...")
                else:
                    self.date_dropdown.set("No data available")
                    
        except Exception as e:
            self.logger.error(f"Error refreshing dates: {e}")
            
    def _on_date_selected(self, selected_date: str):
        """Handle date selection."""
        if selected_date == "Select a date..." or selected_date == "No data available":
            return
            
        # Enable model dropdown and refresh models for selected date
        self.model_dropdown.configure(state="normal")
        self._refresh_models_for_date(selected_date)
        
        # Reset downstream selections
        self.serial_dropdown.configure(state="disabled", values=[])
        self.serial_dropdown.set("")
        self.load_unit_button.configure(state="disabled")
        
    def _refresh_models_for_date(self, date_str: str):
        """Refresh available models for selected date."""
        if not hasattr(self.main_window, 'db_manager') or not self.main_window.db_manager:
            return
            
        try:
            # Parse date
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            # Get unique models for this date
            with self.main_window.db_manager.get_session() as session:
                results = session.query(
                    AnalysisResult.model_number
                ).filter(
                    AnalysisResult.timestamp >= date,
                    AnalysisResult.timestamp < (date + pd.Timedelta(days=1))
                ).distinct().order_by(AnalysisResult.model_number).all()
                
                models = [r.model_number for r in results if r.model_number]
                
                # Update dropdown
                self.model_dropdown.configure(values=models)
                if models:
                    self.model_dropdown.set("Select a model...")
                else:
                    self.model_dropdown.set("No models found")
                    
        except Exception as e:
            self.logger.error(f"Error refreshing models: {e}")
            
    def _on_model_selected(self, selected_model: str):
        """Handle model selection."""
        if selected_model == "Select a model..." or selected_model == "No models found":
            return
            
        # Enable serial dropdown and refresh serials
        self.serial_dropdown.configure(state="normal")
        self._refresh_serials_for_model(self.date_var.get(), selected_model)
        
        # Reset downstream selections
        self.load_unit_button.configure(state="disabled")
        
    def _refresh_serials_for_model(self, date_str: str, model: str):
        """Refresh available serials for selected date and model."""
        if not hasattr(self.main_window, 'db_manager') or not self.main_window.db_manager:
            return
            
        try:
            # Parse date
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            # Get unique serials for this date and model
            with self.main_window.db_manager.get_session() as session:
                results = session.query(
                    AnalysisResult.serial_number,
                    AnalysisResult.timestamp
                ).filter(
                    AnalysisResult.timestamp >= date,
                    AnalysisResult.timestamp < (date + pd.Timedelta(days=1)),
                    AnalysisResult.model_number == model
                ).order_by(AnalysisResult.timestamp.desc()).all()
                
                # Format serials with time for display
                serial_options = []
                for serial, timestamp in results:
                    if serial and timestamp:
                        time_str = timestamp.strftime("%H:%M:%S")
                        serial_options.append(f"{serial} ({time_str})")
                
                # Update dropdown
                self.serial_dropdown.configure(values=serial_options)
                if serial_options:
                    self.serial_dropdown.set("Select a serial...")
                else:
                    self.serial_dropdown.set("No serials found")
                    
        except Exception as e:
            self.logger.error(f"Error refreshing serials: {e}")
            
    def _on_serial_selected(self, selected_serial: str):
        """Handle serial selection."""
        if selected_serial == "Select a serial..." or selected_serial == "No serials found":
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
            
            # Parse date
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            # Load unit from database
            with self.main_window.db_manager.get_session() as session:
                result = session.query(AnalysisResult).filter(
                    AnalysisResult.timestamp >= date,
                    AnalysisResult.timestamp < (date + pd.Timedelta(days=1)),
                    AnalysisResult.model_number == model,
                    AnalysisResult.serial_number == serial
                ).order_by(AnalysisResult.timestamp.desc()).first()
                
                if result:
                    self.selected_unit = result
                    # Update unit info display
                    info_text = f"Loaded: {model} - {serial}\n"
                    info_text += f"Trim Date: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
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
        # Parse the validation results with error handling
        import json
        try:
            validation_data = json.loads(selected_unit.validation_results)
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            self.logger.error(f"Error parsing validation results: {e}")
            raise ValueError("Invalid validation data format")
        
        # Extract track data
        tracks = []
        for track_name, track_data in validation_data.items():
            if track_name.startswith("Track") and isinstance(track_data, dict):
                if 'measured_values' in track_data and 'theoretical_values' in track_data:
                    track_df = pd.DataFrame({
                        'position': track_data.get('positions', list(range(len(track_data['measured_values'])))),
                        'measured': track_data['measured_values'],
                        'theoretical': track_data['theoretical_values']
                    })
                    tracks.append(track_df)
        
        # Combine tracks (for now, use first track if multiple)
        if tracks:
            trim_df = tracks[0]
            # Calculate error
            trim_df['error'] = trim_df['measured'] - trim_df['theoretical']
            return trim_df
        else:
            raise ValueError("No track data found in laser trim results")
            
    def _compare_linearity(self, trim_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare linearity between trim and test data."""
        results = {}
        
        # Normalize position ranges for comparison
        trim_positions = trim_data['position'].values
        test_positions = test_data['position'].values
        
        # Interpolate data to common positions
        common_positions = np.linspace(
            max(trim_positions.min(), test_positions.min()),
            min(trim_positions.max(), test_positions.max()),
            100
        )
        
        # Interpolate trim data
        trim_measured_interp = np.interp(common_positions, trim_positions, trim_data['measured'].values)
        trim_theoretical_interp = np.interp(common_positions, trim_positions, trim_data['theoretical'].values)
        trim_error_interp = trim_measured_interp - trim_theoretical_interp
        
        # Interpolate test data
        test_measured_interp = np.interp(common_positions, test_positions, test_data['measured_volts'].values)
        test_theoretical_interp = np.interp(common_positions, test_positions, test_data['theoretical_volts'].values)
        test_error_interp = test_measured_interp - test_theoretical_interp
        
        # Calculate differences
        measured_diff = trim_measured_interp - test_measured_interp
        error_diff = trim_error_interp - test_error_interp
        
        # Calculate statistics
        results['position'] = common_positions
        results['trim_measured'] = trim_measured_interp
        results['trim_error'] = trim_error_interp
        results['test_measured'] = test_measured_interp
        results['test_error'] = test_error_interp
        results['measured_diff'] = measured_diff
        results['error_diff'] = error_diff
        
        # Summary statistics
        results['stats'] = {
            'mean_measured_diff': np.mean(measured_diff),
            'std_measured_diff': np.std(measured_diff),
            'max_measured_diff': np.max(np.abs(measured_diff)),
            'mean_error_diff': np.mean(error_diff),
            'std_error_diff': np.std(error_diff),
            'max_error_diff': np.max(np.abs(error_diff))
        }
        
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
            results_text += f"Mean Measured Difference: {stats['mean_measured_diff']:.6f} V\n"
            results_text += f"Std Dev Measured Difference: {stats['std_measured_diff']:.6f} V\n"
            results_text += f"Max Measured Difference: {stats['max_measured_diff']:.6f} V\n\n"
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
        """Create comparison overlay chart."""
        try:
            # Clear previous chart and figure
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
            if self.current_plot:
                plt.close(self.current_plot)
                self.current_plot = None
                
            # Create figure with subplots
            fig = Figure(figsize=(12, 8), dpi=100)
            
            # Linearity comparison plot
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(results['position'], results['trim_measured'], 'b-', label='Laser Trim', linewidth=2)
            ax1.plot(results['position'], results['test_measured'], 'r--', label='Final Test', linewidth=2)
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Measured Voltage (V)')
            ax1.set_title('Linearity Comparison: Laser Trim vs Final Test')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Error comparison plot
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(results['position'], results['trim_error'], 'b-', label='Laser Trim Error', linewidth=2)
            ax2.plot(results['position'], results['test_error'], 'r--', label='Final Test Error', linewidth=2)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Error (V)')
            ax2.set_title('Linearity Error Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add statistics text
            stats = results['stats']
            stats_text = f"Mean Diff: {stats['mean_measured_diff']:.6f}V, "
            stats_text += f"Max Diff: {stats['max_measured_diff']:.6f}V"
            fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10)
            
            fig.tight_layout()
            
            # Create canvas
            self.canvas = FigureCanvasTkAgg(fig, master=self.chart_display_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
            
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
            
    def refresh(self):
        """Refresh page data."""
        self._refresh_dates()
        
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