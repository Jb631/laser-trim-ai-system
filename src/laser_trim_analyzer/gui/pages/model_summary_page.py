"""
Model Summary Page for Laser Trim Analyzer

Provides comprehensive analysis and reporting for specific models,
including sigma trending, key metrics, and export capabilities.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
import threading
import os
import logging
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
# import seaborn as sns  # Not used

# Optional libraries for advanced analysis
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# from laser_trim_analyzer.gui.pages.base_page_ctk import BasePage  # Using CTkFrame instead
from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
from laser_trim_analyzer.gui.widgets.stat_card import StatCard
from laser_trim_analyzer.gui.widgets.metric_card_ctk import MetricCard
from laser_trim_analyzer.gui.theme_helper import ThemeHelper
from laser_trim_analyzer.utils.date_utils import safe_datetime_convert


class ModelSummaryPage(ctk.CTkFrame):
    """Model summary and analysis page."""

    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Add BasePage-like functionality
        self.is_visible = False
        self.needs_refresh = True
        self._stop_requested = False
        
        # Page-specific attributes
        self.selected_model = None
        self.model_data = None
        self.current_stats = {}
        
        # Thread safety
        self._query_lock = threading.Lock()
        
        # Create the page
        self._create_page()

    def _create_page(self):
        """Create model summary page content with consistent theme (matching batch processing)."""
        # Main scrollable container (matching batch processing theme)
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create sections in order (matching batch processing pattern)
        self._create_header()
        self._create_model_selection()
        self._create_metrics_section()
        self._create_trend_section()
        self._create_analysis_section()
        self._create_actions_section()

    def _create_header(self):
        """Create header section (matching batch processing theme)."""
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.pack(fill='x', pady=(0, 20))

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Model Summary & Analysis",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=15)

    def _create_model_selection(self):
        """Create model selection section (matching batch processing theme)."""
        self.selection_frame = ctk.CTkFrame(self.main_container)
        self.selection_frame.pack(fill='x', pady=(0, 20))

        self.selection_label = ctk.CTkLabel(
            self.selection_frame,
            text="Model Selection:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.selection_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Selection container
        self.selection_container = ctk.CTkFrame(self.selection_frame)
        self.selection_container.pack(fill='x', padx=15, pady=(0, 15))

        # Model selection row
        selection_row = ctk.CTkFrame(self.selection_container)
        selection_row.pack(fill='x', padx=10, pady=(10, 10))

        model_label = ctk.CTkLabel(selection_row, text="Select Model:")
        model_label.pack(side='left', padx=10, pady=10)

        self.model_var = tk.StringVar()
        self.model_combo = ctk.CTkComboBox(
            selection_row,
            variable=self.model_var,
            width=200,
            height=30,
            command=self._on_model_selected
        )
        self.model_combo.pack(side='left', padx=(0, 20), pady=10)
        
        # Note: Mousewheel support for CTkComboBox dropdowns is not possible due to 
        # tkinter.Menu limitations. The dropdown uses a Menu widget which captures
        # all events in its own event loop.

        self.refresh_btn = ctk.CTkButton(
            selection_row,
            text="🔄 Refresh",
            command=self._load_models,
            width=100,
            height=30
        )
        self.refresh_btn.pack(side='left', padx=10, pady=10)

        # Model info display
        self.model_info_label = ctk.CTkLabel(
            self.selection_container,
            text="No model selected",
            font=ctk.CTkFont(size=12)
        )
        self.model_info_label.pack(padx=10, pady=(0, 10))

    def _create_metrics_section(self):
        """Create key metrics display section (matching batch processing theme)."""
        self.metrics_frame = ctk.CTkFrame(self.main_container)
        self.metrics_frame.pack(fill='x', pady=(0, 20))

        self.metrics_label = ctk.CTkLabel(
            self.metrics_frame,
            text="Key Performance Metrics:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.metrics_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Metrics container
        self.metrics_container = ctk.CTkFrame(self.metrics_frame)
        self.metrics_container.pack(fill='x', padx=15, pady=(0, 15))

        # Row 1 of metrics
        metrics_row1 = ctk.CTkFrame(self.metrics_container)
        metrics_row1.pack(fill='x', padx=10, pady=(10, 5))

        self.metric_cards = {}
        
        self.metric_cards['total_units'] = MetricCard(
            metrics_row1,
            title="Total Units",
            value="--",
            color_scheme="neutral"
        )
        self.metric_cards['total_units'].pack(side='left', fill='x', expand=True, padx=(10, 5), pady=10)

        self.metric_cards['pass_rate'] = MetricCard(
            metrics_row1,
            title="Overall Pass Rate",
            value="--%",
            color_scheme="success"
        )
        self.metric_cards['pass_rate'].pack(side='left', fill='x', expand=True, padx=(5, 5), pady=10)

        self.metric_cards['sigma_avg'] = MetricCard(
            metrics_row1,
            title="Avg Sigma Gradient",
            value="--",
            color_scheme="warning"
        )
        self.metric_cards['sigma_avg'].pack(side='left', fill='x', expand=True, padx=(5, 5), pady=10)

        self.metric_cards['recent_trend'] = MetricCard(
            metrics_row1,
            title="7-Day Trend",
            value="--",
            color_scheme="info"
        )
        self.metric_cards['recent_trend'].pack(side='left', fill='x', expand=True, padx=(5, 10), pady=10)

        # Row 2 of metrics
        metrics_row2 = ctk.CTkFrame(self.metrics_container)
        metrics_row2.pack(fill='x', padx=10, pady=(5, 10))

        self.metric_cards['sigma_std'] = MetricCard(
            metrics_row2,
            title="Sigma Std Dev",
            value="--",
            color_scheme="neutral"
        )
        self.metric_cards['sigma_std'].pack(side='left', fill='x', expand=True, padx=(10, 5), pady=10)

        self.metric_cards['high_risk'] = MetricCard(
            metrics_row2,
            title="High Risk Units",
            value="--",
            color_scheme="danger"
        )
        self.metric_cards['high_risk'].pack(side='left', fill='x', expand=True, padx=(5, 5), pady=10)

        self.metric_cards['validation_rate'] = MetricCard(
            metrics_row2,
            title="Validation Rate",
            value="--%",
            color_scheme="info"
        )
        self.metric_cards['validation_rate'].pack(side='left', fill='x', expand=True, padx=(5, 5), pady=10)

        self.metric_cards['date_range'] = MetricCard(
            metrics_row2,
            title="Date Range",
            value="--",
            color_scheme="neutral"
        )
        self.metric_cards['date_range'].pack(side='left', fill='x', expand=True, padx=(5, 10), pady=10)

    def _create_trend_section(self):
        """Create sigma trend chart section (matching batch processing theme)."""
        self.trend_frame = ctk.CTkFrame(self.main_container)
        self.trend_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.trend_label = ctk.CTkLabel(
            self.trend_frame,
            text="Sigma Gradient Trend:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.trend_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Chart container with minimum height
        self.trend_container = ctk.CTkFrame(self.trend_frame, height=400)
        self.trend_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        self.trend_container.pack_propagate(False)  # Prevent container from shrinking

        # Create matplotlib chart widget with proper sizing for control chart
        self.trend_chart = ChartWidget(
            self.trend_container,
            title="Sigma Gradient Control Chart",
            chart_type="line",
            figsize=(10, 6)  # Standard control chart size
        )
        self.trend_chart.pack(fill='both', expand=True, padx=10, pady=10)

    def _create_analysis_section(self):
        """Create additional analysis charts section (matching batch processing theme)."""
        self.analysis_frame = ctk.CTkFrame(self.main_container)
        self.analysis_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.analysis_label = ctk.CTkLabel(
            self.analysis_frame,
            text="Additional Analysis:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.analysis_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Analysis container with tabs and minimum height
        self.analysis_container = ctk.CTkFrame(self.analysis_frame, height=600)
        self.analysis_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        self.analysis_container.pack_propagate(False)  # Prevent container from shrinking

        # Analysis tabs with explicit height
        self.analysis_tabview = ctk.CTkTabview(self.analysis_container, height=550)
        self.analysis_tabview.pack(fill='both', expand=True, padx=10, pady=10)

        # Add tabs with more valuable charts
        self.analysis_tabview.add("Quality Overview")
        self.analysis_tabview.add("Trend Analysis") 
        self.analysis_tabview.add("Risk Assessment")
        self.analysis_tabview.add("Historical Data")
        self.analysis_tabview.add("Process Capability")

        # Quality Overview - Combined metrics dashboard
        overview_frame = ctk.CTkFrame(self.analysis_tabview.tab("Quality Overview"))
        overview_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        overview_info = ctk.CTkLabel(
            overview_frame,
            text="Key quality metrics at a glance - shows performance zones, defect rates, and critical issues.",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        overview_info.pack(anchor='w', padx=5, pady=(5, 0))
        
        self.overview_chart = ChartWidget(
            overview_frame,
            title="Quality Metrics Dashboard",
            chart_type="bar",
            figsize=(8, 5)  # Compact dashboard size
        )
        self.overview_chart.pack(fill='both', expand=True, padx=5, pady=5)

        # Trend Analysis - Performance over time with control limits
        trend_frame = ctk.CTkFrame(self.analysis_tabview.tab("Trend Analysis"))
        trend_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        trend_info = ctk.CTkLabel(
            trend_frame,
            text="Performance trends with statistical control limits. Shows if process is stable and predictable.",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        trend_info.pack(anchor='w', padx=5, pady=(5, 0))
        
        self.trend_analysis_chart = ChartWidget(
            trend_frame,
            title="X-bar & R Control Chart",
            chart_type="line",
            figsize=(10, 6)  # Standard SPC chart size
        )
        self.trend_analysis_chart.pack(fill='both', expand=True, padx=5, pady=5)

        # Risk Assessment - Visual risk matrix
        risk_frame = ctk.CTkFrame(self.analysis_tabview.tab("Risk Assessment"))
        risk_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        risk_info = ctk.CTkLabel(
            risk_frame,
            text="Risk matrix showing problem areas. Size = frequency, color = severity. Focus on large red bubbles.",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        risk_info.pack(anchor='w', padx=5, pady=(5, 0))
        
        self.risk_chart = ChartWidget(
            risk_frame,
            title="Risk Assessment Heat Map",
            chart_type="scatter",
            figsize=(8, 6)  # Square aspect for heat map
        )
        self.risk_chart.pack(fill='both', expand=True, padx=5, pady=5)

        # Historical Data - Detailed historical records table
        history_frame = ctk.CTkFrame(self.analysis_tabview.tab("Historical Data"))
        history_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        history_info = ctk.CTkLabel(
            history_frame,
            text="Complete historical data for this model. Sort by columns to find specific records.",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        history_info.pack(anchor='w', padx=5, pady=(5, 10))
        
        # Create regular frame for historical data table (avoid nested scrollable frames)
        self.history_scroll = ctk.CTkFrame(history_frame)
        self.history_scroll.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Historical data will be populated when model is selected
        self.history_header_frame = None
        self.history_data_frames = []

        # Process Capability - Cpk and statistical analysis
        cpk_frame = ctk.CTkFrame(self.analysis_tabview.tab("Process Capability"))
        cpk_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        cpk_info = ctk.CTkLabel(
            cpk_frame,
            text="Process capability index (Cpk) shows how well the process meets specifications. Target: Cpk > 1.33",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        cpk_info.pack(anchor='w', padx=5, pady=(5, 0))
        
        self.cpk_chart = ChartWidget(
            cpk_frame,
            title="Process Capability (Cpk) Analysis",
            chart_type="histogram",
            figsize=(8, 5)  # Standard histogram size
        )
        self.cpk_chart.pack(fill='both', expand=True, padx=5, pady=5)

    def _create_actions_section(self):
        """Create export and print controls section (matching batch processing theme)."""
        self.actions_frame = ctk.CTkFrame(self.main_container)
        self.actions_frame.pack(fill='x', pady=(0, 20))

        self.actions_label = ctk.CTkLabel(
            self.actions_frame,
            text="Export & Actions:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.actions_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Actions container
        self.actions_container = ctk.CTkFrame(self.actions_frame)
        self.actions_container.pack(fill='x', padx=15, pady=(0, 15))

        # Action buttons
        button_frame = ctk.CTkFrame(self.actions_container)
        button_frame.pack(fill='x', padx=10, pady=(10, 10))

        self.export_excel_btn = ctk.CTkButton(
            button_frame,
            text="📊 Export to Excel",
            command=self._export_to_excel,
            width=150,
            height=40
        )
        self.export_excel_btn.pack(side='left', padx=(10, 10), pady=10)

        self.export_chart_btn = ctk.CTkButton(
            button_frame,
            text="📈 Export Chart Data",
            command=self._export_chart_data,
            width=150,
            height=40
        )
        self.export_chart_btn.pack(side='left', padx=(0, 10), pady=10)

        self.generate_pdf_btn = ctk.CTkButton(
            button_frame,
            text="📄 Generate PDF Report",
            command=self._generate_pdf_report,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.generate_pdf_btn.pack(side='left', padx=(0, 10), pady=10)
        
        self.print_chart_btn = ctk.CTkButton(
            button_frame,
            text="🖨️ Print Chart",
            command=self._print_chart,
            width=120,
            height=40
        )
        self.print_chart_btn.pack(side='left', padx=(0, 10), pady=10)

    def _load_models(self):
        """Load available models from database."""
        if not self.db_manager:
            messagebox.showerror("Error", "Database not connected")
            return

        try:
            # Get unique models from database
            with self.db_manager.get_session() as session:
                from laser_trim_analyzer.database.manager import DBAnalysisResult
                
                results = session.query(DBAnalysisResult.model).distinct().filter(
                    DBAnalysisResult.model.isnot(None),
                    DBAnalysisResult.model != ''
                ).order_by(DBAnalysisResult.model).all()
                
                # Fix tuple index out of range error by safely extracting model values
                models = []
                for row in results:
                    if row and len(row) > 0 and row[0]:
                        models.append(row[0])

            # Fix: Use configure() instead of ['values'] for CTk widgets
            # Check if we're cleaning up before updating combo box
            if not getattr(self, '_cleaning_up', False):
                self.model_combo.configure(values=models)
                if models and not self.model_var.get():
                    self.model_var.set(models[0])
                    self._on_model_selected()

            self.logger.info(f"Loaded {len(models)} models for summary page")

        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            self.logger.error(f"Model loading traceback:\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")

    def _on_model_selected(self, event=None):
        """Handle model selection change."""
        model = self.model_var.get()
        if not model:
            return

        self.selected_model = model
        self.model_info_label.configure(text=f"Loading data for model: {model}...")
        self.logger.info(f"Loading model data for: {model}")

        # Clear existing charts
        try:
            self.trend_chart.clear()
            self.overview_chart.clear()
            self.trend_analysis_chart.clear()
            self.risk_chart.clear()
            self.cpk_chart.clear()
            # Clear historical data table
            self._clear_historical_table()
        except Exception as e:
            self.logger.warning(f"Error clearing charts: {e}")

        # Load model data in background thread
        thread = threading.Thread(target=self._load_model_data, args=(model,), daemon=True)
        thread.start()

    def _load_model_data(self, model: str):
        """Load comprehensive data for the selected model."""
        with self._query_lock:
            try:
                if not self.db_manager:
                    raise ValueError("Database not connected")

                # Get all historical data for this model
                historical_data = self.db_manager.get_historical_data(
                    model=model,
                    include_tracks=True,
                    limit=None  # Get all data
                )

                if not historical_data:
                    self.after(0, lambda: self.model_info_label.configure(
                        text=f"No data found for model: {model}"
                    ))
                    return

                # Convert to pandas DataFrame for analysis
                data_rows = []
                for analysis in historical_data:
                    for track in analysis.tracks:
                        # Use the dates from the database that were extracted from filename in the core processor
                        # The core processor now properly extracts dates from filenames during initial processing
                        trim_date = safe_datetime_convert(analysis.file_date)
                        timestamp = safe_datetime_convert(analysis.timestamp)
                        
                        # Ensure we have valid datetime objects (fallback to current time if needed)
                        if not trim_date:
                            trim_date = datetime.now()
                            self.logger.warning(f"Could not parse file_date for {analysis.filename}, using current time")
                        
                        if not timestamp:
                            timestamp = trim_date  # Use trim_date as fallback
                            self.logger.warning(f"Could not parse timestamp for {analysis.filename}, using trim_date")
                        
                        row = {
                        'analysis_id': analysis.id,
                        'filename': analysis.filename,
                        'trim_date': trim_date,
                        'timestamp': timestamp,
                        'model': analysis.model,
                        'serial': analysis.serial,
                        'system': analysis.system.value,
                        'overall_status': analysis.overall_status.value,
                        'track_id': track.track_id,
                        'track_status': track.status.value,
                        # Direct track attributes (no nested objects)
                        'sigma_gradient': getattr(track, 'sigma_gradient', None),
                        'sigma_threshold': getattr(track, 'sigma_threshold', None),
                        'sigma_pass': getattr(track, 'sigma_pass', False),
                        'linearity_spec': getattr(track, 'linearity_spec', None),
                        'linearity_error_raw': getattr(track, 'final_linearity_error_raw', None),
                        'linearity_error_shifted': getattr(track, 'final_linearity_error_shifted', None),
                        'linearity_pass': getattr(track, 'linearity_pass', False),
                        'linearity_fail_points': getattr(track, 'linearity_fail_points', 0),
                        'unit_length': getattr(track, 'unit_length', None),
                        'untrimmed_resistance': getattr(track, 'untrimmed_resistance', None),
                        'trimmed_resistance': getattr(track, 'trimmed_resistance', None),
                        'resistance_change': getattr(track, 'resistance_change', None),
                        'resistance_change_percent': getattr(track, 'resistance_change_percent', None),
                        'failure_probability': getattr(track, 'failure_probability', None),
                        'risk_category': track.risk_category.value if hasattr(track, 'risk_category') and track.risk_category else None,
                        'optimal_offset': getattr(track, 'optimal_offset', None),
                        'max_deviation': getattr(track, 'max_deviation', None),
                        'processing_time': analysis.processing_time
                        }
                        # Add computed linearity_error field (use shifted if available, else raw)
                        row['linearity_error'] = row['linearity_error_shifted'] or row['linearity_error_raw'] or 0
                        data_rows.append(row)

                self.model_data = pd.DataFrame(data_rows)
                
                # Log data summary with detailed date information for debugging
                self.logger.info(f"Loaded {len(self.model_data)} data rows for model {model}")
                if len(self.model_data) > 0:
                    self.logger.info(f"Date range: {self.model_data['trim_date'].min()} to {self.model_data['trim_date'].max()}")
                    self.logger.info(f"Columns: {list(self.model_data.columns)}")
                    
                    # Debug sigma gradient values
                    sigma_values = self.model_data['sigma_gradient'].dropna()
                    self.logger.info(f"=== SIGMA GRADIENT DEBUG INFO ===")
                    self.logger.info(f"Total rows: {len(self.model_data)}, Non-null sigma values: {len(sigma_values)}")
                    if len(sigma_values) > 0:
                        self.logger.info(f"Sigma gradient - Min: {sigma_values.min():.6f}, Max: {sigma_values.max():.6f}, "
                                        f"Mean: {sigma_values.mean():.6f}, Std: {sigma_values.std():.6f}")
                        # Show first few values
                        self.logger.info(f"First 10 sigma values: {sigma_values.head(10).tolist()}")
                        # Check for zeros
                        zero_count = (sigma_values == 0).sum()
                        if zero_count > 0:
                            self.logger.warning(f"Found {zero_count} zero sigma gradient values ({zero_count/len(sigma_values)*100:.1f}%)")
                    else:
                        self.logger.error("No valid sigma gradient values found!")
                    
                    # Add specific debugging for model 8340-1
                    if model == "8340-1":
                        self.logger.info(f"=== DEBUG INFO FOR MODEL 8340-1 ===")
                        sample_data = self.model_data[['filename', 'trim_date', 'sigma_gradient', 'sigma_threshold']].head(10)
                        for idx, row in sample_data.iterrows():
                            sigma_val = row['sigma_gradient'] if row['sigma_gradient'] is not None else 0.0
                            thresh_val = row['sigma_threshold'] if row['sigma_threshold'] is not None else 0.0
                            self.logger.info(f"File: {row['filename']}, Date: {row['trim_date']}, "
                                           f"Sigma: {sigma_val:.6f}, Threshold: {thresh_val:.6f}")
                        self.logger.info(f"=== END DEBUG INFO ===")
                
                    # Update UI on main thread safely
                    if self.winfo_exists():
                        self.after(0, self._update_model_display)

            except Exception as e:
                self.logger.error(f"Failed to load model data: {e}")
                self.logger.error(f"Traceback:\n{traceback.format_exc()}")
                # Show error safely
                if self.winfo_exists():
                    self.after(0, lambda: messagebox.showerror(
                        "Error", f"Failed to load model data:\n{str(e)}"
                    ))

    def _update_model_display(self):
        """Update all UI elements with model data."""
        if self.model_data is None or len(self.model_data) == 0:
            return

        model = self.selected_model
        df = self.model_data

        try:
            # Update info label with safe date formatting
            min_date = df['trim_date'].min()
            max_date = df['trim_date'].max()
            
            # Ensure dates are datetime objects and format safely
            if pd.isna(min_date) or pd.isna(max_date):
                date_range = "Date range unavailable"
            else:
                try:
                    date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                except (AttributeError, ValueError):
                    date_range = "Date format error"
            
            self.model_info_label.configure(
                text=f"Model: {model} | {len(df)} data points | Date range: {date_range}"
            )

            # Calculate and update metrics - handle errors gracefully
            try:
                self._update_metrics(df)
            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")

            # Update charts - handle errors gracefully for each chart
            try:
                self._update_trend_chart()
            except Exception as e:
                self.logger.error(f"Error updating trend chart: {e}")
                
            try:
                self._update_analysis_charts()
            except Exception as e:
                self.logger.error(f"Error updating analysis charts: {e}")

            # Update historical data table
            try:
                self._update_historical_table()
            except Exception as e:
                self.logger.error(f"Error updating historical table: {e}")
            
            # Update process capability chart
            try:
                self._update_cpk_chart()
            except Exception as e:
                self.logger.error(f"Error updating Cpk chart: {e}")

            # Always enable action buttons if we have data
            self.export_excel_btn.configure(state='normal')
            self.generate_pdf_btn.configure(state='normal')
            self.export_chart_btn.configure(state='normal')

            # Update quick stats
            if hasattr(self, 'quick_stats_label'):
                self.quick_stats_label.configure(
                    text=f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
                )
            
            self.logger.info(f"Successfully updated model display for {model} with {len(df)} data points")
            
        except Exception as e:
            self.logger.error(f"Error updating model display: {e}")
            self.model_info_label.configure(
                text=f"Error displaying model data: {str(e)}"
            )
            # Still enable buttons if we have data
            if self.model_data is not None and len(self.model_data) > 0:
                self.export_excel_btn.configure(state='normal')
                self.generate_pdf_btn.configure(state='normal')
                self.export_chart_btn.configure(state='normal')

    def _update_metrics(self, df: pd.DataFrame):
        """Update metric cards with calculated values."""
        # Basic metrics
        total_units = len(df['analysis_id'].unique())
        pass_rate = (df['track_status'] == 'Pass').mean() * 100
        
        # Calculate sigma metrics with proper handling
        sigma_values = df['sigma_gradient'].dropna()
        if len(sigma_values) > 0:
            sigma_avg = sigma_values.mean()
            sigma_std = sigma_values.std()
            
            # Log for debugging
            self.logger.debug(f"Sigma metrics - Count: {len(sigma_values)}, Avg: {sigma_avg:.6f}, Std: {sigma_std:.6f}")
            
            # Check if all values are zero or very small
            if sigma_avg < 1e-6 and sigma_std < 1e-6:
                self.logger.warning(f"Sigma values appear to be all zeros or very small: avg={sigma_avg}, std={sigma_std}")
        else:
            sigma_avg = 0.0
            sigma_std = 0.0
            self.logger.warning("No valid sigma gradient values for metrics calculation")

        # Advanced metrics
        validation_rate = df['linearity_pass'].mean() * 100 if df['linearity_pass'].notna().any() else 0
        resistance_avg = df['resistance_change_percent'].mean() if df['resistance_change_percent'].notna().any() else 0
        high_risk = (df['risk_category'] == 'High').sum() if df['risk_category'].notna().any() else 0

        # 7-day trend
        recent_df = df[df['trim_date'] >= (datetime.now() - timedelta(days=7))]
        if len(recent_df) > 0:
            recent_pass_rate = (recent_df['track_status'] == 'Pass').mean() * 100
            trend_change = recent_pass_rate - pass_rate
            trend_text = f"{trend_change:+.1f}%"
        else:
            trend_text = "No recent data"

        # Update cards with safe access
        try:
            self.metric_cards['total_units'].update_value(total_units)
            self.metric_cards['pass_rate'].update_value(f"{pass_rate:.1f}")
            self.metric_cards['sigma_avg'].update_value(f"{sigma_avg:.4f}")
            self.metric_cards['recent_trend'].update_value(trend_text)
            self.metric_cards['sigma_std'].update_value(f"{sigma_std:.4f}")
            self.metric_cards['validation_rate'].update_value(f"{validation_rate:.1f}")
            self.metric_cards['high_risk'].update_value(high_risk)
            
            # Date range formatting
            min_date = df['trim_date'].min()
            max_date = df['trim_date'].max()
            if pd.notna(min_date) and pd.notna(max_date):
                date_range = f"{min_date.strftime('%m/%d')} - {max_date.strftime('%m/%d')}"
            else:
                date_range = "N/A"
            self.metric_cards['date_range'].update_value(date_range)
        except Exception as e:
            self.logger.warning(f"Error updating individual metric cards: {e}")

        # Update colors based on values
        try:
            if pass_rate >= 95:
                self.metric_cards['pass_rate'].set_color_scheme('success')
            elif pass_rate >= 90:
                self.metric_cards['pass_rate'].set_color_scheme('warning')
            else:
                self.metric_cards['pass_rate'].set_color_scheme('danger')
        except Exception as e:
            self.logger.warning(f"Error updating metric card colors: {e}")

    def _update_trend_chart(self):
        """Update the sigma trend chart using enhanced SPC control chart."""
        if self.model_data is None or len(self.model_data) == 0:
            return

        df = self.model_data.copy()
        
        try:
            # Ensure dates are datetime objects
            if 'trim_date' in df.columns:
                df['trim_date'] = pd.to_datetime(df['trim_date'])
            
            # Clean data - remove NaN values
            chart_data = df[['trim_date', 'sigma_gradient']].copy()
            chart_data = chart_data.dropna(subset=['trim_date', 'sigma_gradient'])
            
            self.logger.info(f"Updating trend chart with {len(chart_data)} valid data points")
            
            if len(chart_data) > 0:
                # Sort by date for proper display
                chart_data = chart_data.sort_values('trim_date')
                
                # Define specification limits for sigma gradient (based on typical ranges: 0.0631-0.2121)
                # Using process-centered limits with some margin for process variation
                target_value = 0.1376  # Midpoint of typical range
                spec_limits = (0.050, 0.250)  # Slightly wider than observed range for realistic limits
                
                # Get ML threshold if available for additional reference
                ml_threshold = None
                if self.db_manager and self.selected_model:
                    ml_threshold = self.db_manager.get_latest_ml_threshold(self.selected_model)
                    if ml_threshold:
                        self.logger.info(f"Including ML threshold: {ml_threshold}")
                
                # Use the enhanced control chart method
                try:
                    self.trend_chart.plot_enhanced_control_chart(
                        data=chart_data,
                        value_column='sigma_gradient',
                        date_column='trim_date',
                        spec_limits=spec_limits,
                        target_value=target_value,
                        title="Sigma Gradient Trend"
                    )
                    
                    # Add ML threshold as additional reference line if available
                    if ml_threshold:
                        ax = self.trend_chart._get_or_create_axes()
                        ax.axhline(y=ml_threshold, 
                                  color=self.trend_chart.qa_colors['ml_threshold'], 
                                  linestyle='-.', 
                                  linewidth=2, 
                                  alpha=0.8,
                                  label=f'ML Threshold: {ml_threshold:.4f}')
                        
                        # Update legend to include ML threshold
                        handles, labels = ax.get_legend_handles_labels()
                        if handles and labels:
                            ax.legend(handles, labels, loc='upper right', fontsize=9)
                    
                    self.logger.info("Enhanced control chart created successfully")
                    
                except Exception as chart_error:
                    self.logger.error(f"Error creating enhanced control chart: {chart_error}")
                    self.trend_chart.show_error(
                        "Control Chart Error",
                        f"Failed to create statistical process control chart: {str(chart_error)}"
                    )
            else:
                self.trend_chart.show_placeholder("No Valid Data", "No sigma gradient data available for analysis")
                
        except Exception as e:
            self.logger.error(f"Error updating trend chart: {e}")
            self.trend_chart.show_error(
                "Chart Update Error",
                "Failed to update the trend chart. Please try refreshing the data."
            )

    def _update_analysis_charts(self):
        """Update the additional analysis charts with more valuable visualizations."""
        if self.model_data is None or len(self.model_data) == 0:
            self.logger.warning("No model data available for analysis charts")
            return

        df = self.model_data

        try:
            # 1. Quality Overview Dashboard - Key metrics at a glance
            try:
                # Calculate key quality metrics
                total_units = len(df)
                pass_rate = (df['track_status'] == 'Pass').mean() * 100
                
                # Sigma gradient statistics
                sigma_values = df['sigma_gradient'].dropna()
                
                # Use ML threshold if available, otherwise calculate based on pass/fail
                ml_threshold = None
                if self.db_manager and self.selected_model:
                    ml_threshold = self.db_manager.get_latest_ml_threshold(self.selected_model)
                
                if ml_threshold:
                    # Count values below ML threshold as "in spec"
                    in_spec_count = (sigma_values <= ml_threshold).sum()
                    in_spec_rate = (in_spec_count / len(sigma_values) * 100) if len(sigma_values) > 0 else 0
                else:
                    # Without ML threshold, use pass rate as proxy for in-spec rate
                    in_spec_rate = pass_rate
                
                # Calculate stability (based on standard deviation)
                sigma_std = sigma_values.std() if len(sigma_values) > 1 else 0
                stability_score = max(0, 100 - (sigma_std * 200))  # Lower std = higher stability
                
                # Risk level calculation
                if 'risk_category' in df.columns:
                    high_risk_count = (df['risk_category'] == 'High').sum()
                    risk_score = 100 - (high_risk_count / total_units * 100) if total_units > 0 else 100
                else:
                    risk_score = 100 if pass_rate > 95 else (pass_rate if pass_rate > 80 else pass_rate * 0.8)
                
                # Calculate trend for each metric (last 30 days if available)
                recent_df = df.tail(min(30, len(df)))
                old_pass_rate = (recent_df.head(15)['track_status'] == 'Pass').mean() * 100
                new_pass_rate = (recent_df.tail(15)['track_status'] == 'Pass').mean() * 100
                pass_trend = 'up' if new_pass_rate > old_pass_rate else ('down' if new_pass_rate < old_pass_rate else 'stable')
                
                # Get historical data for sparklines
                # Group by date to get daily averages
                daily_metrics = df.groupby(df['trim_date'].dt.date).agg({
                    'track_status': lambda x: (x == 'Pass').mean() * 100,
                    'sigma_gradient': lambda x: ((x >= 0.3) & (x <= 0.7)).mean() * 100 if len(x) > 0 else 0
                }).tail(30)  # Last 30 days
                
                # Prepare dashboard metrics
                dashboard_metrics = {
                    'Pass Rate': {
                        'value': pass_rate,
                        'status': 'green' if pass_rate >= 95 else ('yellow' if pass_rate >= 85 else 'red'),
                        'trend': pass_trend,
                        'history': daily_metrics['track_status'].tolist() if len(daily_metrics) > 0 else [],
                        'target': 95,
                        'label': 'Pass Rate'
                    },
                    'In-Spec Rate': {
                        'value': in_spec_rate,
                        'status': 'green' if in_spec_rate >= 90 else ('yellow' if in_spec_rate >= 75 else 'red'),
                        'trend': 'stable',  # Could calculate if needed
                        'history': daily_metrics['sigma_gradient'].tolist() if len(daily_metrics) > 0 else [],
                        'target': 90,
                        'label': 'Sigma In-Spec'
                    },
                    'Stability': {
                        'value': stability_score,
                        'status': 'green' if stability_score >= 80 else ('yellow' if stability_score >= 60 else 'red'),
                        'trend': 'stable',
                        'history': [],  # Could calculate rolling std
                        'target': 80,
                        'label': 'Process Stability'
                    },
                    'Risk Score': {
                        'value': risk_score,
                        'status': 'green' if risk_score >= 90 else ('yellow' if risk_score >= 70 else 'red'),
                        'trend': 'stable',
                        'history': [],
                        'target': 90,
                        'label': 'Risk Score'
                    }
                }
                
                # Create a simple quality metrics overview using bar chart
                metrics_names = list(dashboard_metrics.keys())
                metrics_values = [dashboard_metrics[name]['value'] for name in metrics_names]
                metrics_targets = [dashboard_metrics[name]['target'] for name in metrics_names]
                
                self.overview_chart.clear_chart()
                ax = self.overview_chart._get_or_create_axes()
                
                # Create bar chart with targets
                x_pos = range(len(metrics_names))
                bars = ax.bar(x_pos, metrics_values, 
                             color=[self.overview_chart.qa_colors['pass'] if v >= t 
                                   else self.overview_chart.qa_colors['warning'] if v >= t*0.8 
                                   else self.overview_chart.qa_colors['fail']
                                   for v, t in zip(metrics_values, metrics_targets)],
                             alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add target lines
                for i, target in enumerate(metrics_targets):
                    ax.axhline(y=target, xmin=(i-0.4)/len(metrics_names), xmax=(i+0.4)/len(metrics_names), 
                              color='red', linestyle='--', linewidth=2, alpha=0.7)
                
                # Add value labels on bars
                for bar, value in zip(bars, metrics_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                ax.set_xlabel('Quality Metrics', fontsize=12)
                ax.set_ylabel('Percentage (%)', fontsize=12)
                ax.set_title('Quality Metrics Overview', fontsize=14, fontweight='bold', pad=20)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(metrics_names, rotation=45, ha='right')
                ax.set_ylim(0, 105)
                ax.grid(True, alpha=0.3)
                
                # Apply theme styling
                self.overview_chart._apply_theme_to_axes(ax)
                self.overview_chart.canvas.draw_idle()
                
                self.logger.info("Updated quality health dashboard")
                
            except Exception as e:
                self.logger.error(f"Error creating quality overview: {e}")
                self.overview_chart.show_error(
                    "Overview Chart Error",
                    f"Failed to create quality overview: {str(e)}"
                )

            # 2. Early Warning System - Moving range, CUSUM, and violation detection
            try:
                # Sort by date
                df_sorted = df.sort_values('trim_date').copy()
                
                # Prepare data for early warning system
                if len(df_sorted) > 5:  # Need at least 5 points for meaningful analysis
                    # Get individual measurements (not daily averages) for better sensitivity
                    warning_data = df_sorted[['trim_date', 'sigma_gradient']].dropna()
                    
                    # Create a simple early warning time series chart
                    self.trend_analysis_chart.clear_chart()
                    ax = self.trend_analysis_chart._get_or_create_axes()
                    
                    # Plot time series with warning zones
                    dates = warning_data['trim_date']
                    values = warning_data['sigma_gradient']
                    
                    ax.plot(dates, values, 'o-', markersize=4, linewidth=1, 
                           color=self.trend_analysis_chart.qa_colors['primary'], 
                           alpha=0.8, label='Sigma Gradient')
                    
                    # Add warning limits
                    mean_val = values.mean()
                    std_val = values.std()
                    ucl = mean_val + 3 * std_val
                    lcl = mean_val - 3 * std_val
                    uwl = mean_val + 2 * std_val
                    lwl = mean_val - 2 * std_val
                    
                    ax.axhline(y=ucl, color=self.trend_analysis_chart.qa_colors['fail'], 
                              linestyle='--', linewidth=2, alpha=0.7, label=f'UCL: {ucl:.4f}')
                    ax.axhline(y=uwl, color=self.trend_analysis_chart.qa_colors['warning'], 
                              linestyle=':', linewidth=1.5, alpha=0.7, label=f'UWL: {uwl:.4f}')
                    ax.axhline(y=mean_val, color=self.trend_analysis_chart.qa_colors['control_center'], 
                              linestyle='-', linewidth=2, alpha=0.8, label=f'Mean: {mean_val:.4f}')
                    ax.axhline(y=lwl, color=self.trend_analysis_chart.qa_colors['warning'], 
                              linestyle=':', linewidth=1.5, alpha=0.7, label=f'LWL: {lwl:.4f}')
                    ax.axhline(y=lcl, color=self.trend_analysis_chart.qa_colors['fail'], 
                              linestyle='--', linewidth=2, alpha=0.7, label=f'LCL: {lcl:.4f}')
                    
                    # Highlight out-of-control points
                    violations = (values > ucl) | (values < lcl)
                    if violations.any():
                        ax.scatter(dates[violations], values[violations], 
                                  color='red', s=80, marker='x', linewidth=3, 
                                  label='Violations', zorder=5)
                    
                    warnings_mask = ((values > uwl) & (values <= ucl)) | ((values < lwl) & (values >= lcl))
                    if warnings_mask.any():
                        ax.scatter(dates[warnings_mask], values[warnings_mask], 
                                  color='orange', s=60, marker='^', linewidth=2, 
                                  label='Warnings', zorder=4)
                    
                    ax.set_xlabel('Date', fontsize=10)
                    ax.set_ylabel('Sigma Gradient', fontsize=10)
                    ax.set_title('Early Warning Trend Analysis', fontsize=12, fontweight='bold')
                    ax.legend(loc='upper right', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    # Apply theme styling and formatting
                    self.trend_analysis_chart._apply_theme_to_axes(ax)
                    
                    # Format dates on x-axis
                    import matplotlib.dates as mdates
                    from matplotlib.ticker import MaxNLocator
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    
                    self.trend_analysis_chart.canvas.draw_idle()
                    
                    # Calculate warning statistics
                    values = warning_data['sigma_gradient']
                    mean_val = values.mean()
                    std_val = values.std()
                    
                    # Count violations and warnings
                    ucl = mean_val + 3 * std_val
                    lcl = mean_val - 3 * std_val
                    uwl = mean_val + 2 * std_val
                    lwl = mean_val - 2 * std_val
                    
                    violations = ((values > ucl) | (values < lcl)).sum()
                    warnings = (((values > uwl) & (values <= ucl)) | ((values < lwl) & (values >= lcl))).sum()
                    
                    # Log summary
                    self.logger.info(f"Early Warning System: {violations} violations, {warnings} warnings out of {len(values)} points")
                    
                else:
                    self.trend_analysis_chart.show_placeholder(
                        "Insufficient data",
                        "Need at least 5 data points for early warning analysis"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error creating trend analysis: {e}")
                self.trend_analysis_chart.show_error(
                    "Trend Analysis Error",
                    f"Unable to calculate control limits: {str(e)}"
                )

            # 3. Failure Pattern Analysis - Simple pass/fail distribution
            try:
                # Create a simple failure analysis chart
                self.risk_chart.clear_chart()
                ax = self.risk_chart._get_or_create_axes()
                
                # Count pass/fail status
                status_counts = df['track_status'].value_counts()
                
                if len(status_counts) > 0:
                    # Create pie chart for pass/fail distribution
                    colors = [self.risk_chart.qa_colors['pass'] if status == 'Pass' 
                             else self.risk_chart.qa_colors['fail'] 
                             for status in status_counts.index]
                    
                    wedges, texts, autotexts = ax.pie(status_counts.values, 
                                                     labels=status_counts.index,
                                                     colors=colors,
                                                     autopct='%1.1f%%',
                                                     startangle=90,
                                                     textprops={'fontsize': 10})
                    
                    # Make percentage text bold
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    ax.set_title('Pass/Fail Distribution', fontsize=12, fontweight='bold', pad=20)
                    
                    # Add summary statistics as text
                    total_count = len(df)
                    fail_count = status_counts.get('Fail', 0)
                    pass_count = status_counts.get('Pass', 0)
                    fail_rate = (fail_count / total_count * 100) if total_count > 0 else 0
                    
                    summary_text = f'Total Units: {total_count}\nPass: {pass_count}\nFail: {fail_count}\nFail Rate: {fail_rate:.1f}%'
                    ax.text(1.2, 0.5, summary_text, transform=ax.transAxes, 
                           fontsize=10, verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                    
                    self.logger.info(f"Failure Pattern Analysis: {fail_count} failures out of {total_count} units ({fail_rate:.1f}%)")
                else:
                    ax.text(0.5, 0.5, 'No status data available', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                
                self.risk_chart.canvas.draw_idle()
                    
            except Exception as e:
                self.logger.error(f"Error creating risk assessment: {e}")
                self.risk_chart.show_error(
                    "Risk Analysis Error", 
                    f"Failed to create failure pattern analysis: {str(e)}"
                )

        except Exception as e:
            self.logger.error(f"Error updating analysis charts: {e}")
            # Update all charts with error message
            error_msg = f"Analysis error: {str(e)[:30]}..."
            self.overview_chart.show_placeholder("Error loading data", error_msg)
            self.trend_analysis_chart.show_placeholder("Error loading data", error_msg)
            self.risk_chart.show_placeholder("Error loading data", error_msg)

    def _export_to_excel(self):
        """Export model data to Excel file."""
        if self.model_data is None or len(self.model_data) == 0:
            messagebox.showwarning("No Data", "No data available to export")
            return

        try:
            # Get save location with proper file dialog
            initial_filename = f"model_summary_{self.selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            filename = filedialog.asksaveasfilename(
                title="Export Model Summary",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialfile=initial_filename
            )
            
            if not filename:
                return

            # Create comprehensive Excel export
            try:
                import openpyxl
            except ImportError:
                messagebox.showerror(
                    "Missing Dependency",
                    "The openpyxl library is required for Excel export.\n"
                    "Please install it using: pip install openpyxl"
                )
                return
                
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Raw data - ensure all columns are included
                # Log what columns we're exporting
                self.logger.info(f"Exporting {len(self.model_data)} rows with columns: {list(self.model_data.columns)}")
                
                # Check for missing columns and log
                expected_cols = ['linearity_fail_points', 'unit_length']
                for col in expected_cols:
                    if col not in self.model_data.columns:
                        self.logger.warning(f"Column '{col}' not found in model data")
                    else:
                        non_null = self.model_data[col].notna().sum()
                        self.logger.info(f"Column '{col}' has {non_null} non-null values out of {len(self.model_data)}")
                
                self.model_data.to_excel(writer, sheet_name='Raw Data', index=False)
                
                # Summary statistics
                summary_stats = self._calculate_summary_stats()
                if summary_stats:
                    summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                    summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
                
                # Daily aggregates
                try:
                    daily_agg = self.model_data.groupby(
                        self.model_data['trim_date'].dt.date
                    ).agg({
                        'sigma_gradient': ['count', 'mean', 'std', 'min', 'max'],
                        'track_status': lambda x: (x == 'Pass').mean() * 100 if len(x) > 0 else 0,
                        'linearity_pass': lambda x: x.mean() * 100 if x.notna().any() else 0,
                        'resistance_change_percent': lambda x: x.mean() if x.notna().any() else 0
                    }).round(4)
                    
                    # Fix column names for readability
                    daily_agg.columns = [
                        'Units_Count', 'Sigma_Mean', 'Sigma_StdDev', 'Sigma_Min', 'Sigma_Max',
                        'Pass_Rate_%', 'Linearity_Pass_%', 'Avg_Resistance_Change_%'
                    ]
                    daily_agg.to_excel(writer, sheet_name='Daily Summary')
                except Exception as e:
                    self.logger.warning(f"Could not create daily summary: {e}")
                
                # Monthly aggregates
                try:
                    monthly_agg = self.model_data.groupby(
                        self.model_data['trim_date'].dt.to_period('M')
                    ).agg({
                        'sigma_gradient': ['count', 'mean', 'std'],
                        'track_status': lambda x: (x == 'Pass').mean() * 100 if len(x) > 0 else 0,
                        'linearity_pass': lambda x: x.mean() * 100 if x.notna().any() else 0,
                    }).round(4)
                    
                    # Fix column names for readability
                    monthly_agg.columns = [
                        'Units_Count', 'Sigma_Mean', 'Sigma_StdDev',
                        'Pass_Rate_%', 'Linearity_Pass_%'
                    ]
                    monthly_agg.to_excel(writer, sheet_name='Monthly Summary')
                except Exception as e:
                    self.logger.warning(f"Could not create monthly summary: {e}")

            messagebox.showinfo("Export Complete", f"Model summary exported to:\n{filename}")
            self.logger.info(f"Exported model summary to {filename}")

        except Exception as e:
            error_msg = f"Failed to export data: {str(e)}"
            messagebox.showerror("Export Error", error_msg)
            self.logger.error(f"Export failed: {e}")

    def _export_chart_data(self):
        """Export chart data as CSV."""
        if self.model_data is None or len(self.model_data) == 0:
            messagebox.showwarning("No Data", "No data available to export")
            return

        try:
            # Get save location with proper file dialog
            initial_filename = f"chart_data_{self.selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            filename = filedialog.asksaveasfilename(
                title="Export Chart Data",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=initial_filename
            )
            
            if not filename:
                return

            # Export trend chart data
            chart_data = self.model_data[['trim_date', 'sigma_gradient', 'track_status', 'linearity_pass']].copy()
            chart_data.to_csv(filename, index=False)
            
            messagebox.showinfo("Export Complete", f"Chart data exported to:\n{filename}")
            self.logger.info(f"Exported chart data to {filename}")

        except Exception as e:
            error_msg = f"Failed to export chart data: {str(e)}"
            messagebox.showerror("Export Error", error_msg)
            self.logger.error(f"Chart data export failed: {e}")

    def _generate_pdf_report(self):
        """Generate PDF report with charts and metrics."""
        if self.model_data is None or len(self.model_data) == 0:
            messagebox.showwarning("No Data", "No data available for report")
            return

        try:
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt
            
            # Get save location with proper file dialog
            initial_filename = f"model_report_{self.selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            filename = filedialog.asksaveasfilename(
                title="Generate PDF Report",
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                initialfile=initial_filename
            )
            
            if not filename:
                return

            # Create PDF report
            with PdfPages(filename) as pdf:
                # Page 1: Summary metrics
                fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
                fig.suptitle(f'Model {self.selected_model} - Quality Analysis Report', fontsize=16, fontweight='bold')
                
                try:
                    # Summary statistics table
                    ax = axes[0, 0]
                    ax.axis('tight')
                    ax.axis('off')
                    ax.set_title('Key Metrics', fontweight='bold')
                    
                    stats = self._calculate_summary_stats()
                    if stats:
                        table_data = [[k, v] for k, v in list(stats.items())[:8]]
                        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                                       cellLoc='left', loc='center')
                        table.auto_set_font_size(False)
                        table.set_fontsize(9)
                except Exception as e:
                    self.logger.warning(f"Could not create statistics table: {e}")
                
                try:
                    # Sigma trend plot
                    ax = axes[0, 1]
                    df_sorted = self.model_data.sort_values('trim_date')
                    ax.scatter(df_sorted['trim_date'], df_sorted['sigma_gradient'], alpha=0.6, s=20)
                    ax.set_title('Sigma Gradient Trend', fontweight='bold')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Sigma Gradient')
                    ax.tick_params(axis='x', rotation=45)
                except Exception as e:
                    self.logger.warning(f"Could not create sigma trend plot: {e}")
                
                try:
                    # Distribution histogram
                    ax = axes[1, 0]
                    sigma_clean = self.model_data['sigma_gradient'].dropna()
                    if len(sigma_clean) > 0:
                        ax.hist(sigma_clean, bins=min(20, len(sigma_clean)//2), alpha=0.7, edgecolor='black')
                        ax.set_title('Sigma Distribution', fontweight='bold')
                        ax.set_xlabel('Sigma Gradient')
                        ax.set_ylabel('Frequency')
                except Exception as e:
                    self.logger.warning(f"Could not create distribution plot: {e}")
                
                try:
                    # Pass/Fail by month
                    ax = axes[1, 1]
                    monthly_stats = self.model_data.groupby(
                        self.model_data['trim_date'].dt.to_period('M')
                    ).agg({'track_status': lambda x: (x == 'Pass').mean() * 100 if len(x) > 0 else 0})
                    
                    if len(monthly_stats) > 0:
                        months = [str(m) for m in monthly_stats.index]
                        pass_rates = monthly_stats['track_status'].values
                        bars = ax.bar(range(len(months)), pass_rates)
                        
                        # Color bars based on pass rate
                        for i, (bar, rate) in enumerate(zip(bars, pass_rates)):
                            if rate >= 95:
                                bar.set_color('green')
                            elif rate >= 90:
                                bar.set_color('orange')
                            else:
                                bar.set_color('red')
                        
                        ax.set_title('Monthly Pass Rate', fontweight='bold')
                        ax.set_xlabel('Month')
                        ax.set_ylabel('Pass Rate (%)')
                        ax.set_xticks(range(len(months)))
                        ax.set_xticklabels(months, rotation=45)
                        ax.set_ylim(0, 100)
                except Exception as e:
                    self.logger.warning(f"Could not create pass/fail plot: {e}")
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

                # Page 2: Additional analysis (if needed)
                # Could add more detailed charts here

            messagebox.showinfo("Report Generated", f"PDF report generated:\n{filename}")
            self.logger.info(f"Generated PDF report: {filename}")

        except ImportError:
            messagebox.showerror("Missing Dependency", "PDF generation requires matplotlib. Please install it first.")
        except Exception as e:
            error_msg = f"Failed to generate report: {str(e)}"
            messagebox.showerror("Report Error", error_msg)
            self.logger.error(f"Report generation failed: {e}")

    def _print_chart(self):
        """Export current chart in print-friendly format."""
        if self.model_data is None or len(self.model_data) == 0:
            messagebox.showwarning("No Data", "No data available to print")
            return
        
        try:
            # Determine which tab is active
            current_tab = self.analysis_tabview.get()
            
            # Get the corresponding chart
            if current_tab == "Distribution":
                chart = self.distribution_chart
            elif current_tab == "Pass/Fail Trend":
                chart = self.passfail_chart
            elif current_tab == "Correlation":
                chart = self.correlation_chart
            else:
                # Default to trend chart
                chart = self.trend_chart
            
            # Create a print-friendly version
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            
            # Get save location
            initial_filename = f"chart_{self.selected_model}_{current_tab.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            filename = filedialog.asksaveasfilename(
                title="Save Chart for Printing",
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("PNG files", "*.png"), ("All files", "*.*")],
                initialfile=initial_filename
            )
            
            if not filename:
                return
            
            # Create a new figure with white background for printing
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            # Set white background
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            # Re-plot the data with print-friendly settings
            if current_tab == "Correlation" and hasattr(self, '_last_correlation_data'):
                x_data, y_data = self._last_correlation_data
                ax.scatter(x_data, y_data, alpha=0.6, s=50, color='blue')
                ax.set_xlabel('Sigma Gradient', fontsize=12)
                ax.set_ylabel('Linearity Error', fontsize=12)
                ax.set_title(f'Sigma vs Linearity Correlation - Model {self.selected_model}', fontsize=14, fontweight='bold')
                
                # Add correlation value
                if len(x_data) > 5:
                    correlation = np.corrcoef(x_data, y_data)[0, 1]
                    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                           transform=ax.transAxes, fontsize=11,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'))
            else:
                # For other charts, use the trend data
                df_sorted = self.model_data.sort_values('trim_date')
                ax.plot(df_sorted['trim_date'], df_sorted['sigma_gradient'], 
                       marker='o', linewidth=2, markersize=4, color='blue', alpha=0.8)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Sigma Gradient', fontsize=12)
                ax.set_title(f'Sigma Gradient Trend - Model {self.selected_model}', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
            
            # Add grid for readability
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add footer with metadata
            fig.text(0.5, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Model: {self.selected_model}', 
                    ha='center', fontsize=9, color='gray')
            
            plt.tight_layout()
            
            # Save the figure
            if filename.endswith('.pdf'):
                fig.savefig(filename, format='pdf', bbox_inches='tight', facecolor='white')
            else:
                fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            
            plt.close(fig)
            
            messagebox.showinfo("Export Successful", f"Chart exported to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export chart:\n{str(e)}")
            self.logger.error(f"Chart export error: {e}")
    
    def _calculate_summary_stats(self) -> Dict[str, str]:
        """Calculate comprehensive summary statistics."""
        if self.model_data is None or len(self.model_data) == 0:
            return {}

        df = self.model_data
        
        stats = {
            'Model': self.selected_model,
            'Total Data Points': f"{len(df):,}",
            'Unique Units': f"{df['analysis_id'].nunique():,}",
            'Date Range': f"{df['trim_date'].min().strftime('%Y-%m-%d')} to {df['trim_date'].max().strftime('%Y-%m-%d')}",
            'Overall Pass Rate': f"{(df['track_status'] == 'Pass').mean() * 100:.1f}%",
            'Sigma Pass Rate': f"{df['sigma_pass'].mean() * 100:.1f}%",
            'Linearity Pass Rate': f"{df['linearity_pass'].mean() * 100:.1f}%" if df['linearity_pass'].notna().any() else "N/A",
            'Avg Sigma Gradient': f"{df['sigma_gradient'].mean():.4f}" if df['sigma_gradient'].notna().any() else "N/A",
            'Sigma Std Dev': f"{df['sigma_gradient'].std():.4f}" if df['sigma_gradient'].notna().any() else "N/A",
            'Min Sigma': f"{df['sigma_gradient'].min():.4f}" if df['sigma_gradient'].notna().any() else "N/A",
            'Max Sigma': f"{df['sigma_gradient'].max():.4f}" if df['sigma_gradient'].notna().any() else "N/A",
            'Avg Resistance Change': f"{df['resistance_change_percent'].mean():.2f}%" if df['resistance_change_percent'].notna().any() else "N/A",
            'High Risk Units': f"{(df['risk_category'] == 'High').sum():,}" if df['risk_category'].notna().any() else "N/A",
            'Processing Time Avg': f"{df['processing_time'].mean():.2f}s" if df['processing_time'].notna().any() else "N/A"
        }
        
        return stats
    
    @property
    def db_manager(self):
        """Get database manager from main window."""
        return self.main_window.db_manager if hasattr(self.main_window, 'db_manager') else None

    def on_show(self):
        """Called when page is shown."""
        # Load models when page is first shown
        if not hasattr(self, '_models_loaded'):
            self._load_models()
            self._models_loaded = True
    
    def cleanup(self):
        """Clean up resources when page is destroyed."""
        # Mark as cleaning up to prevent further operations
        self._cleaning_up = True
        
        # Only destroy dropdown menu, not the entire combo box
        # The combo box will be destroyed when the frame is destroyed
        if hasattr(self, 'model_combo'):
            try:
                if hasattr(self.model_combo, '_dropdown_menu'):
                    self.model_combo._dropdown_menu.destroy()
            except Exception:
                pass
    
    def _clear_historical_table(self):
        """Clear the historical data table."""
        if self.history_header_frame:
            self.history_header_frame.destroy()
            self.history_header_frame = None
        
        for frame in self.history_data_frames:
            frame.destroy()
        self.history_data_frames.clear()
    
    def _update_historical_table(self):
        """Update the historical data table with model records."""
        if self.model_data is None or len(self.model_data) == 0:
            return
        
        # Clear existing table
        self._clear_historical_table()
        
        # Create header
        self.history_header_frame = ctk.CTkFrame(self.history_scroll)
        self.history_header_frame.pack(fill='x', padx=5, pady=(5, 10))
        
        # Define columns
        columns = ['Date', 'Serial', 'Status', 'Sigma', 'Linearity', 'Risk', 'File']
        col_widths = [100, 120, 80, 80, 80, 80, 200]
        
        for i, (col, width) in enumerate(zip(columns, col_widths)):
            label = ctk.CTkLabel(
                self.history_header_frame,
                text=col,
                font=ctk.CTkFont(size=12, weight="bold"),
                width=width,
                anchor='w'
            )
            label.grid(row=0, column=i, padx=5, pady=5, sticky='w')
        
        # Add data rows
        df = self.model_data.sort_values('trim_date', ascending=False)
        
        for idx, (_, row) in enumerate(df.head(100).iterrows()):  # Show last 100 records
            row_frame = ctk.CTkFrame(
                self.history_scroll,
                fg_color=("gray90", "gray20") if idx % 2 == 0 else ("gray95", "gray25")
            )
            row_frame.pack(fill='x', padx=5, pady=1)
            
            # Date
            date_label = ctk.CTkLabel(
                row_frame,
                text=row['trim_date'].strftime('%Y-%m-%d'),
                width=col_widths[0],
                anchor='w'
            )
            date_label.grid(row=0, column=0, padx=5, pady=3, sticky='w')
            
            # Serial
            serial_label = ctk.CTkLabel(
                row_frame,
                text=row['serial'],
                width=col_widths[1],
                anchor='w'
            )
            serial_label.grid(row=0, column=1, padx=5, pady=3, sticky='w')
            
            # Status
            status_color = "green" if row['track_status'] == 'Pass' else "red"
            status_label = ctk.CTkLabel(
                row_frame,
                text=row['track_status'],
                width=col_widths[2],
                anchor='w',
                text_color=status_color
            )
            status_label.grid(row=0, column=2, padx=5, pady=3, sticky='w')
            
            # Sigma
            sigma_text = f"{row['sigma_gradient']:.4f}" if pd.notna(row['sigma_gradient']) else "N/A"
            sigma_label = ctk.CTkLabel(
                row_frame,
                text=sigma_text,
                width=col_widths[3],
                anchor='w'
            )
            sigma_label.grid(row=0, column=3, padx=5, pady=3, sticky='w')
            
            # Linearity
            linearity = f"{row['linearity_error']:.2f}%" if pd.notna(row['linearity_error']) else "N/A"
            linearity_label = ctk.CTkLabel(
                row_frame,
                text=linearity,
                width=col_widths[4],
                anchor='w'
            )
            linearity_label.grid(row=0, column=4, padx=5, pady=3, sticky='w')
            
            # Risk
            risk_color = {"High": "red", "Medium": "orange", "Low": "green"}.get(row.get('risk_category', 'N/A'), "gray")
            risk_label = ctk.CTkLabel(
                row_frame,
                text=row.get('risk_category', 'N/A'),
                width=col_widths[5],
                anchor='w',
                text_color=risk_color
            )
            risk_label.grid(row=0, column=5, padx=5, pady=3, sticky='w')
            
            # Filename (truncated)
            filename = row['filename']
            if len(filename) > 30:
                filename = "..." + filename[-27:]
            file_label = ctk.CTkLabel(
                row_frame,
                text=filename,
                width=col_widths[6],
                anchor='w'
            )
            file_label.grid(row=0, column=6, padx=5, pady=3, sticky='w')
            
            self.history_data_frames.append(row_frame)
    
    def _update_cpk_chart(self):
        """Update process capability analysis chart."""
        if self.model_data is None or len(self.model_data) == 0:
            return
        
        df = self.model_data
        
        # Need at least some data for meaningful capability analysis
        if len(df) < 10:
            self.cpk_chart.show_placeholder("Insufficient Data", "Need at least 10 data points for capability analysis")
            return
        
        try:
            # Clean sigma gradient data
            sigma_data = df['sigma_gradient'].dropna()
            
            if len(sigma_data) < 10:
                self.cpk_chart.show_placeholder("Insufficient Valid Data", "Not enough valid sigma gradient values")
                return
            
            # Define specification limits for sigma gradient (based on observed ranges)
            spec_limits = (0.050, 0.250)  # LSL, USL
            target_value = 0.1376  # Target (midpoint of typical range)
            
            # Create DataFrame for the chart method
            capability_df = pd.DataFrame({'sigma_gradient': sigma_data})
            
            # Use the enhanced process capability histogram method
            self.cpk_chart.plot_process_capability_histogram(
                data=capability_df,
                value_column='sigma_gradient',
                spec_limits=spec_limits,
                target_value=target_value,
                title="Sigma Distribution"
            )
            
            # Calculate basic capability metrics for logging
            mean_val = sigma_data.mean()
            std_val = sigma_data.std()
            
            # Calculate Cp and Cpk
            if std_val > 0:
                cp = (spec_limits[1] - spec_limits[0]) / (6 * std_val)
                cpu = (spec_limits[1] - mean_val) / (3 * std_val)
                cpl = (mean_val - spec_limits[0]) / (3 * std_val)
                cpk = min(cpu, cpl)
                
                self.logger.info(f"Process Capability: Cp={cp:.3f}, Cpk={cpk:.3f}, Mean={mean_val:.4f}, Std={std_val:.4f}")
            else:
                self.logger.warning("Cannot calculate capability indices: standard deviation is zero")
            
        except Exception as e:
            self.logger.error(f"Error updating capability chart: {e}")
            self.cpk_chart.show_error(
                "Capability Analysis Error", 
                f"Failed to create process capability analysis: {str(e)}"
            )