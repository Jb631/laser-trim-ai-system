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
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
import seaborn as sns

from laser_trim_analyzer.gui.pages.base_page import BasePage
from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
from laser_trim_analyzer.gui.widgets.stat_card import StatCard
from laser_trim_analyzer.gui.widgets.metric_card import MetricCard
from laser_trim_analyzer.gui.widgets import add_mousewheel_support
from laser_trim_analyzer.utils.date_utils import safe_datetime_convert


class ModelSummaryPage(BasePage):
    """Model summary and analysis page."""

    def __init__(self, parent, main_window):
        self.selected_model = None
        self.model_data = None
        self.current_stats = {}
        super().__init__(parent, main_window)

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

        # Add tabs with charts
        self.analysis_tabview.add("Distribution")
        self.analysis_tabview.add("Pass/Fail Trend")
        self.analysis_tabview.add("Correlation")

        # Distribution chart
        self.distribution_chart = ChartWidget(
            self.analysis_tabview.tab("Distribution"),
            title="Sigma Gradient Distribution",
            chart_type="histogram"
        )
        self.distribution_chart.pack(fill='both', expand=True, padx=5, pady=5)

        # Pass/Fail chart
        self.passfail_chart = ChartWidget(
            self.analysis_tabview.tab("Pass/Fail Trend"),
            title="Monthly Pass Rate",
            chart_type="bar"
        )
        self.passfail_chart.pack(fill='both', expand=True, padx=5, pady=5)

        # Correlation chart
        self.correlation_chart = ChartWidget(
            self.analysis_tabview.tab("Correlation"),
            title="Sigma vs Linearity Correlation",
            chart_type="scatter"
        )
        self.correlation_chart.pack(fill='both', expand=True, padx=5, pady=5)

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
                
                models = [row[0] for row in results if row[0]]

            # Fix: Use configure() instead of ['values'] for CTk widgets
            self.model_combo.configure(values=models)
            if models and not self.model_var.get():
                self.model_var.set(models[0])
                self._on_model_selected()

            self.logger.info(f"Loaded {len(models)} models for summary page")

        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
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
            self.distribution_chart.clear()
            self.passfail_chart.clear()
            self.correlation_chart.clear()
        except Exception as e:
            self.logger.warning(f"Error clearing charts: {e}")

        # Load model data in background thread
        thread = threading.Thread(target=self._load_model_data, args=(model,), daemon=True)
        thread.start()

    def _load_model_data(self, model: str):
        """Load comprehensive data for the selected model."""
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
            
            # Update UI in main thread
            self.after(0, self._update_model_display)

        except Exception as e:
            self.logger.error(f"Failed to load model data: {e}")
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
        """Update the additional analysis charts."""
        if self.model_data is None or len(self.model_data) == 0:
            self.logger.warning("No model data available for analysis charts")
            return

        df = self.model_data

        try:
            # Distribution analysis
            sigma_data = df['sigma_gradient'].dropna()
            if len(sigma_data) > 0:
                distribution_info = f"Distribution Analysis:\n"
                distribution_info += f"Data points: {len(sigma_data)}\n"
                distribution_info += f"Mean: {sigma_data.mean():.4f}\n"
                distribution_info += f"Std Dev: {sigma_data.std():.4f}\n"
                distribution_info += f"Min: {sigma_data.min():.4f}\n"
                distribution_info += f"Max: {sigma_data.max():.4f}"
                
                self.distribution_chart.update_chart_data(df[['trim_date', 'sigma_gradient']])
                self.logger.info(f"Updated distribution analysis with {len(sigma_data)} data points")
            else:
                self.distribution_chart.update_chart_data(pd.DataFrame({'trim_date': [], 'sigma_gradient': []}))
                self.logger.warning("No valid sigma gradient data")

            # Pass/Fail analysis by month
            try:
                # Create month-year periods for better grouping
                df_copy = df.copy()
                df_copy['month_year'] = df_copy['trim_date'].dt.to_period('M')
                
                monthly_stats = df_copy.groupby('month_year').agg({
                    'track_status': lambda x: (x == 'Pass').mean() * 100 if len(x) > 0 else 0
                }).reset_index()
                
                if len(monthly_stats) > 0:
                    passfail_info = f"Monthly Pass Rate Analysis:\n"
                    for idx, row in monthly_stats.iterrows():
                        month = str(row['month_year'])
                        rate = row['track_status']
                        status_icon = "✅" if rate >= 95 else "⚠️" if rate >= 90 else "❌"
                        passfail_info += f"{month}: {rate:.1f}% {status_icon}\n"
                    
                    overall_rate = df['track_status'].eq('Pass').mean() * 100
                    passfail_info += f"\nOverall: {overall_rate:.1f}%"
                    
                    self.passfail_chart.update_chart_data(monthly_stats[['month_year', 'track_status']])
                    self.logger.info(f"Updated pass/fail analysis with {len(monthly_stats)} months")
                else:
                    self.passfail_chart.update_chart_data(pd.DataFrame({'month_year': [], 'track_status': []}))
                    self.logger.warning("No monthly statistics available")
                    
            except Exception as e:
                self.logger.error(f"Error creating pass/fail analysis: {e}")
                self.passfail_chart.update_chart_data(pd.DataFrame({'month_year': [], 'track_status': []}))

            # Correlation analysis
            try:
                # Check if linearity data exists
                linearity_col = None
                for col in ['linearity_error_shifted', 'linearity_error_raw']:
                    if col in df.columns and df[col].notna().any():
                        linearity_col = col
                        break
                
                if linearity_col:
                    # Filter out null values and get matching indices
                    sigma_clean = df['sigma_gradient'].dropna()
                    linearity_clean = df[linearity_col].dropna()
                    
                    # Find common indices
                    common_idx = sigma_clean.index.intersection(linearity_clean.index)
                    
                    if len(common_idx) > 5:  # Need at least 5 points for meaningful correlation
                        x_data = sigma_clean.loc[common_idx].values
                        y_data = linearity_clean.loc[common_idx].values
                        
                        # Calculate correlation
                        correlation = np.corrcoef(x_data, y_data)[0, 1]
                        
                        correlation_info = f"Sigma vs Linearity Correlation:\n"
                        correlation_info += f"Data points: {len(x_data)}\n"
                        correlation_info += f"Correlation: {correlation:.3f}\n"
                        
                        if abs(correlation) >= 0.7:
                            correlation_info += "Strong correlation 💪"
                        elif abs(correlation) >= 0.4:
                            correlation_info += "Moderate correlation 📊"
                        else:
                            correlation_info += "Weak correlation 📈"
                        
                        # Pass/fail breakdown
                        status_data = df.loc[common_idx, 'track_status']
                        pass_count = (status_data == 'Pass').sum()
                        fail_count = len(status_data) - pass_count
                        correlation_info += f"\n\nPass: {pass_count} | Fail: {fail_count}"
                        
                        self.correlation_chart.update_chart_data(pd.DataFrame({'x': x_data, 'y': y_data}))
                        self.logger.info(f"Updated correlation analysis with {len(x_data)} data points")
                    else:
                        self.correlation_chart.update_chart_data(pd.DataFrame({'x': [], 'y': []}))
                        self.logger.warning(f"Insufficient data for correlation analysis\n(Need at least 5 data points, have {len(common_idx)})")
                else:
                    self.correlation_chart.update_chart_data(pd.DataFrame({'x': [], 'y': []}))
                    self.logger.warning("No linearity data available\nfor correlation analysis")
                    
            except Exception as e:
                self.logger.error(f"Error creating correlation analysis: {e}")
                self.correlation_chart.update_chart_data(pd.DataFrame({'x': [], 'y': []}))

        except Exception as e:
            self.logger.error(f"Error updating analysis charts: {e}")
            # Update all charts with error message
            error_msg = f"Analysis error: {str(e)[:30]}..."
            self.distribution_chart.update_chart_data(pd.DataFrame({'trim_date': [], 'sigma_gradient': []}))
            self.passfail_chart.update_chart_data(pd.DataFrame({'month_year': [], 'track_status': []}))
            self.correlation_chart.update_chart_data(pd.DataFrame({'x': [], 'y': []}))

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

    def on_show(self):
        """Called when page is shown."""
        # Load models when page is first shown
        if not hasattr(self, '_models_loaded'):
            self._load_models()
            self._models_loaded = True 