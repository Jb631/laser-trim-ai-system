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
# import seaborn as sns  # Not used

# from laser_trim_analyzer.gui.pages.base_page_ctk import BasePage  # Using CTkFrame instead
from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
from laser_trim_analyzer.gui.widgets.stat_card import StatCard
from laser_trim_analyzer.gui.widgets.metric_card_ctk import MetricCard
from laser_trim_analyzer.gui.theme_helper import ThemeHelper
# from laser_trim_analyzer.gui.widgets import add_mousewheel_support  # Not used
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

        # Chart container
        self.trend_container = ctk.CTkFrame(self.trend_frame)
        self.trend_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Create matplotlib chart widget
        self.trend_chart = ChartWidget(
            self.trend_container,
            title="Sigma Gradient Trend",
            chart_type="line"
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

        # Analysis container with tabs
        self.analysis_container = ctk.CTkFrame(self.analysis_frame)
        self.analysis_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Analysis tabs
        self.analysis_tabview = ctk.CTkTabview(self.analysis_container)
        self.analysis_tabview.pack(fill='both', expand=True, padx=10, pady=10)

        # Add tabs with more valuable charts
        self.analysis_tabview.add("Quality Overview")
        self.analysis_tabview.add("Trend Analysis") 
        self.analysis_tabview.add("Risk Assessment")

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
            title="Quality Overview Dashboard",
            chart_type="bar"
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
            title="Process Control Chart",
            chart_type="line"
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
            title="Risk Assessment Matrix",
            chart_type="scatter"
        )
        self.risk_chart.pack(fill='both', expand=True, padx=5, pady=5)

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
                        'sigma_gradient': track.sigma_gradient,
                        'sigma_threshold': track.sigma_threshold,
                        'sigma_pass': track.sigma_pass,
                        'linearity_spec': track.linearity_spec,
                        'linearity_error_raw': track.final_linearity_error_raw,
                        'linearity_error_shifted': track.final_linearity_error_shifted,
                        'linearity_pass': track.linearity_pass,
                        'linearity_fail_points': track.linearity_fail_points,
                        'unit_length': track.unit_length,
                        'untrimmed_resistance': track.untrimmed_resistance,
                        'trimmed_resistance': track.trimmed_resistance,
                        'resistance_change': track.resistance_change,
                        'resistance_change_percent': track.resistance_change_percent,
                        'failure_probability': track.failure_probability,
                        'risk_category': track.risk_category.value if track.risk_category else None,
                        'optimal_offset': track.optimal_offset,
                        'max_deviation': track.max_deviation,
                        'processing_time': analysis.processing_time
                        }
                        data_rows.append(row)

                self.model_data = pd.DataFrame(data_rows)
                
                # Log data summary with detailed date information for debugging
                self.logger.info(f"Loaded {len(self.model_data)} data rows for model {model}")
                if len(self.model_data) > 0:
                    self.logger.info(f"Date range: {self.model_data['trim_date'].min()} to {self.model_data['trim_date'].max()}")
                    self.logger.info(f"Columns: {list(self.model_data.columns)}")
                    
                    # Add specific debugging for model 8340-1
                    if model == "8340-1":
                        self.logger.info(f"=== DEBUG INFO FOR MODEL 8340-1 ===")
                        sample_dates = self.model_data[['filename', 'trim_date', 'timestamp']].head(10)
                        for idx, row in sample_dates.iterrows():
                            self.logger.info(f"File: {row['filename']}, Trim Date: {row['trim_date']}, Timestamp: {row['timestamp']}")
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
        sigma_avg = df['sigma_gradient'].mean()
        sigma_std = df['sigma_gradient'].std()

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
        """Update the sigma trend chart."""
        if self.model_data is None or len(self.model_data) == 0:
            return

        df = self.model_data.copy()
        
        # For now, just show basic trend information in the label
        try:
            # Ensure dates are datetime objects
            if 'trim_date' in df.columns:
                df['trim_date'] = pd.to_datetime(df['trim_date'])
            
            # Sort by date
            df = df.sort_values('trim_date')
            
            # Calculate basic trend statistics
            sigma_values = df['sigma_gradient'].dropna()
            if len(sigma_values) > 1:
                # Calculate trend direction
                first_half = sigma_values[:len(sigma_values)//2].mean()
                second_half = sigma_values[len(sigma_values)//2:].mean()
                trend_direction = "↗️ Increasing" if second_half > first_half else "↘️ Decreasing"
                
                trend_info = f"Trend: {trend_direction}\n"
                trend_info += f"Data points: {len(sigma_values)}\n"
                trend_info += f"Range: {sigma_values.min():.4f} - {sigma_values.max():.4f}\n"
                trend_info += f"Average: {sigma_values.mean():.4f}"
                
                self.trend_chart.update_chart_data(df[['trim_date', 'sigma_gradient']])
            else:
                self.trend_chart.update_chart_data(pd.DataFrame({'trim_date': [], 'sigma_gradient': []}))
                
        except Exception as e:
            self.logger.error(f"Error updating trend chart: {e}")
            self.trend_chart.update_chart_data(pd.DataFrame({'trim_date': [], 'sigma_gradient': []}))

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
                fail_rate = 100 - pass_rate
                
                # Sigma gradient zones
                optimal_zone = ((df['sigma_gradient'] >= 0.3) & (df['sigma_gradient'] <= 0.7)).sum()
                warning_zone = ((df['sigma_gradient'] < 0.3) | (df['sigma_gradient'] > 0.7)).sum()
                critical_zone = (df['sigma_gradient'].isna()).sum()
                
                # Risk categories
                if 'risk_category' in df.columns:
                    high_risk = (df['risk_category'] == 'High').sum()
                    medium_risk = (df['risk_category'] == 'Medium').sum()
                    low_risk = (df['risk_category'] == 'Low').sum()
                else:
                    high_risk = medium_risk = low_risk = 0
                
                # Create overview data
                overview_data = pd.DataFrame({
                    'category': ['Pass Rate', 'Fail Rate', 'Optimal Zone', 'Warning Zone', 'Critical Zone', 
                                'High Risk', 'Medium Risk', 'Low Risk'],
                    'value': [pass_rate, fail_rate, 
                             (optimal_zone/total_units)*100 if total_units > 0 else 0,
                             (warning_zone/total_units)*100 if total_units > 0 else 0,
                             (critical_zone/total_units)*100 if total_units > 0 else 0,
                             (high_risk/total_units)*100 if total_units > 0 else 0,
                             (medium_risk/total_units)*100 if total_units > 0 else 0,
                             (low_risk/total_units)*100 if total_units > 0 else 0],
                    'color': ['green', 'red', 'green', 'orange', 'red', 'red', 'orange', 'green']
                })
                
                # Update overview chart
                self.overview_chart.clear_chart()
                fig = self.overview_chart.figure
                ax = fig.add_subplot(111)
                self.overview_chart._apply_theme_to_axes(ax)
                
                # Create grouped bar chart
                x = np.arange(len(overview_data))
                bars = ax.bar(x, overview_data['value'], color=overview_data['color'], alpha=0.7)
                
                # Add value labels on bars with theme-aware color
                theme_colors = ThemeHelper.get_theme_colors()
                text_color = theme_colors["fg"]["primary"]
                for bar, val in zip(bars, overview_data['value']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=10, color=text_color)
                
                # Add reference line at 95% (target)
                ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='Target (95%)')
                ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='Warning (90%)')
                
                ax.set_xticks(x)
                ax.set_xticklabels(overview_data['category'], rotation=45, ha='right')
                ax.set_ylabel('Percentage (%)')
                ax.set_ylim(0, 110)
                ax.set_title('Quality Metrics Overview', fontsize=14, fontweight='bold')
                legend = ax.legend(loc='upper right')
                if legend:
                    self.overview_chart._style_legend(legend)
                ax.grid(True, axis='y', alpha=0.3)
                
                fig.tight_layout()
                self.overview_chart.canvas.draw()
                
                self.logger.info("Updated quality overview dashboard")
                
            except Exception as e:
                self.logger.error(f"Error creating quality overview: {e}")
                self.overview_chart.show_placeholder("Error displaying overview", "Check data format")

            # 2. Trend Analysis - Process control chart with control limits
            try:
                # Sort by date and calculate rolling statistics
                df_sorted = df.sort_values('trim_date').copy()
                
                # Calculate daily averages if multiple units per day
                daily_stats = df_sorted.groupby(df_sorted['trim_date'].dt.date).agg({
                    'sigma_gradient': ['mean', 'std', 'count'],
                    'track_status': lambda x: (x == 'Pass').sum()
                }).reset_index()
                
                daily_stats.columns = ['date', 'mean_sigma', 'std_sigma', 'count', 'pass_count']
                daily_stats['pass_rate'] = (daily_stats['pass_count'] / daily_stats['count']) * 100
                
                if len(daily_stats) > 2:
                    # Calculate control limits
                    overall_mean = daily_stats['mean_sigma'].mean()
                    overall_std = daily_stats['mean_sigma'].std()
                    ucl = overall_mean + 3 * overall_std  # Upper Control Limit
                    lcl = overall_mean - 3 * overall_std  # Lower Control Limit
                    uwl = overall_mean + 2 * overall_std  # Upper Warning Limit
                    lwl = overall_mean - 2 * overall_std  # Lower Warning Limit
                    
                    # Update trend analysis chart
                    self.trend_analysis_chart.clear_chart()
                    fig = self.trend_analysis_chart.figure
                    ax = fig.add_subplot(111)
                    self.trend_analysis_chart._apply_theme_to_axes(ax)
                    
                    # Plot the main trend line
                    dates = pd.to_datetime(daily_stats['date'])
                    ax.plot(dates, daily_stats['mean_sigma'], 'b-o', markersize=6, 
                           linewidth=2, label='Daily Average', alpha=0.8)
                    
                    # Add control limits
                    ax.axhline(y=overall_mean, color='green', linestyle='-', linewidth=2, 
                              label=f'Center Line ({overall_mean:.3f})')
                    ax.axhline(y=ucl, color='red', linestyle='--', linewidth=1.5, 
                              label=f'UCL ({ucl:.3f})')
                    ax.axhline(y=lcl, color='red', linestyle='--', linewidth=1.5, 
                              label=f'LCL ({lcl:.3f})')
                    ax.axhline(y=uwl, color='orange', linestyle=':', linewidth=1, alpha=0.7)
                    ax.axhline(y=lwl, color='orange', linestyle=':', linewidth=1, alpha=0.7)
                    
                    # Shade zones
                    ax.fill_between(dates, lcl, ucl, alpha=0.1, color='green', label='Control Zone')
                    ax.fill_between(dates, lwl, uwl, alpha=0.1, color='yellow')
                    
                    # Highlight out-of-control points
                    ooc_mask = (daily_stats['mean_sigma'] > ucl) | (daily_stats['mean_sigma'] < lcl)
                    if ooc_mask.any():
                        ax.scatter(dates[ooc_mask], daily_stats['mean_sigma'][ooc_mask], 
                                  color='red', s=100, zorder=5, label='Out of Control')
                    
                    # Format x-axis
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Sigma Gradient')
                    ax.set_title('Process Control Chart', fontsize=14, fontweight='bold')
                    legend = ax.legend(loc='best', fontsize=9)
                    if legend:
                        self.trend_analysis_chart._style_legend(legend)
                    ax.grid(True, alpha=0.3)
                    
                    fig.tight_layout()
                    self.trend_analysis_chart.canvas.draw()
                    
                    self.logger.info(f"Updated trend analysis with {len(daily_stats)} data points")
                else:
                    self.trend_analysis_chart.show_placeholder(
                        "Insufficient data",
                        "Need at least 3 days of data for control limits"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error creating trend analysis: {e}")
                self.trend_analysis_chart.show_placeholder("Error displaying trends", "Check data format")

            # 3. Risk Assessment Matrix - Visual risk analysis
            try:
                # Calculate risk metrics for each failure mode
                risk_data = []
                
                # Group by failure patterns
                fail_df = df[df['track_status'] == 'Fail'].copy()
                
                if len(fail_df) > 0:
                    # Analyze failure modes
                    # 1. Out of spec failures
                    out_of_spec = fail_df[(fail_df['sigma_gradient'] < 0.3) | (fail_df['sigma_gradient'] > 0.7)]
                    if len(out_of_spec) > 0:
                        risk_data.append({
                            'failure_mode': 'Out of Spec',
                            'frequency': len(out_of_spec),
                            'severity': 8,  # High severity
                            'detectability': 3,  # Easy to detect
                            'rpn': len(out_of_spec) * 8 * 3
                        })
                    
                    # 2. Linearity failures
                    if 'linearity_pass' in fail_df.columns:
                        linearity_fail = fail_df[fail_df['linearity_pass'] == False]
                        if len(linearity_fail) > 0:
                            risk_data.append({
                                'failure_mode': 'Linearity Issues',
                                'frequency': len(linearity_fail),
                                'severity': 6,  # Medium severity
                                'detectability': 5,  # Moderate to detect
                                'rpn': len(linearity_fail) * 6 * 5
                            })
                    
                    # 3. High risk category failures
                    if 'risk_category' in fail_df.columns:
                        high_risk_fail = fail_df[fail_df['risk_category'] == 'High']
                        if len(high_risk_fail) > 0:
                            risk_data.append({
                                'failure_mode': 'High Risk Units',
                                'frequency': len(high_risk_fail),
                                'severity': 9,  # Very high severity
                                'detectability': 7,  # Hard to detect
                                'rpn': len(high_risk_fail) * 9 * 7
                            })
                    
                    # 4. Random failures (in spec but still failed)
                    in_spec_fail = fail_df[(fail_df['sigma_gradient'] >= 0.3) & (fail_df['sigma_gradient'] <= 0.7)]
                    if len(in_spec_fail) > 0:
                        risk_data.append({
                            'failure_mode': 'Random Failures',
                            'frequency': len(in_spec_fail),
                            'severity': 5,  # Medium severity
                            'detectability': 8,  # Very hard to detect
                            'rpn': len(in_spec_fail) * 5 * 8
                        })
                
                # Also analyze warning conditions from passing units
                pass_df = df[df['track_status'] == 'Pass'].copy()
                if len(pass_df) > 0:
                    # Near spec limit warnings
                    near_limit = pass_df[(pass_df['sigma_gradient'] < 0.35) | (pass_df['sigma_gradient'] > 0.65)]
                    if len(near_limit) > 0:
                        risk_data.append({
                            'failure_mode': 'Near Spec Limits',
                            'frequency': len(near_limit),
                            'severity': 3,  # Low severity (still passing)
                            'detectability': 2,  # Easy to detect
                            'rpn': len(near_limit) * 3 * 2
                        })
                
                if risk_data:
                    # Create risk matrix visualization
                    self.risk_chart.clear_chart()
                    fig = self.risk_chart.figure
                    ax = fig.add_subplot(111)
                    self.risk_chart._apply_theme_to_axes(ax)
                    
                    # Prepare data for bubble chart
                    x = [d['severity'] for d in risk_data]
                    y = [d['detectability'] for d in risk_data]
                    sizes = [d['frequency'] * 50 for d in risk_data]  # Scale for visibility
                    colors = [d['rpn'] for d in risk_data]
                    labels = [d['failure_mode'] for d in risk_data]
                    
                    # Create scatter plot with varying sizes and colors
                    scatter = ax.scatter(x, y, s=sizes, c=colors, cmap='RdYlGn_r', 
                                       alpha=0.6, edgecolors='black', linewidth=1)
                    
                    # Get theme colors for annotations
                    theme_colors = ThemeHelper.get_theme_colors()
                    is_dark = ctk.get_appearance_mode().lower() == "dark"
                    text_color = theme_colors["fg"]["primary"]
                    bg_color = theme_colors["bg"]["secondary"] if is_dark else 'white'
                    
                    # Add labels for each bubble
                    for i, label in enumerate(labels):
                        ax.annotate(f"{label}\n({risk_data[i]['frequency']})", 
                                   (x[i], y[i]), 
                                   xytext=(5, 5), 
                                   textcoords='offset points',
                                   fontsize=9,
                                   color=text_color,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, alpha=0.8, edgecolor=text_color))
                    
                    # Add colorbar for RPN
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Risk Priority Number (RPN)', rotation=270, labelpad=20, color=text_color)
                    cbar.ax.tick_params(colors=text_color)
                    
                    # Add risk zones
                    ax.axhspan(1, 3, xmin=0, xmax=0.33, alpha=0.1, color='green')  # Low risk
                    ax.axhspan(1, 3, xmin=0.33, xmax=0.67, alpha=0.1, color='yellow')  # Medium risk
                    ax.axhspan(1, 3, xmin=0.67, xmax=1, alpha=0.1, color='red')  # High risk
                    
                    ax.axhspan(3, 6, xmin=0, xmax=0.33, alpha=0.1, color='yellow')
                    ax.axhspan(3, 6, xmin=0.33, xmax=0.67, alpha=0.1, color='orange')
                    ax.axhspan(3, 6, xmin=0.67, xmax=1, alpha=0.1, color='red')
                    
                    ax.axhspan(6, 10, xmin=0, xmax=0.33, alpha=0.1, color='orange')
                    ax.axhspan(6, 10, xmin=0.33, xmax=1, alpha=0.1, color='red')
                    
                    # Set axis properties
                    ax.set_xlabel('Severity (1=Low, 10=High)')
                    ax.set_ylabel('Detectability (1=Easy, 10=Hard)')
                    ax.set_xlim(0, 11)
                    ax.set_ylim(0, 11)
                    ax.set_title('Risk Assessment Matrix (FMEA)', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # Add risk zone labels
                    ax.text(2, 9.5, 'LOW RISK', fontsize=8, color='green', fontweight='bold', ha='center')
                    ax.text(5, 9.5, 'MEDIUM RISK', fontsize=8, color='orange', fontweight='bold', ha='center')
                    ax.text(8.5, 9.5, 'HIGH RISK', fontsize=8, color='red', fontweight='bold', ha='center')
                    
                    fig.tight_layout()
                    self.risk_chart.canvas.draw()
                    
                    self.logger.info(f"Updated risk assessment with {len(risk_data)} failure modes")
                else:
                    self.risk_chart.show_placeholder(
                        "No failures detected",
                        "Risk assessment requires failure data"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error creating risk assessment: {e}")
                self.risk_chart.show_placeholder("Error displaying risk matrix", "Check data format")

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
                # Raw data
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
                    
                    daily_agg.columns = ['_'.join(str(col)).strip() for col in daily_agg.columns]
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
                    
                    monthly_agg.columns = ['_'.join(str(col)).strip() for col in monthly_agg.columns]
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
            'Avg Sigma Gradient': f"{df['sigma_gradient'].mean():.4f}",
            'Sigma Std Dev': f"{df['sigma_gradient'].std():.4f}",
            'Min Sigma': f"{df['sigma_gradient'].min():.4f}",
            'Max Sigma': f"{df['sigma_gradient'].max():.4f}",
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