"""
Model Summary Page for Laser Trim Analyzer

Provides comprehensive analysis and reporting for specific models,
including sigma trending, key metrics, and export capabilities.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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
        """Set up the model summary page with proper positioning."""
        # Create scrollable main frame without shifting
        main_container = ttk.Frame(self)
        main_container.pack(fill='both', expand=True)
        
        # Canvas and scrollbar
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
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
        
        # Create content in scrollable frame
        content_frame = scrollable_frame
        
        # Title and model selection
        self._create_header_section(content_frame)
        
        # Key metrics cards
        self._create_metrics_section(content_frame)
        
        # Sigma trend chart
        self._create_trend_section(content_frame)
        
        # Additional analysis charts
        self._create_analysis_section(content_frame)
        
        # Export and print controls
        self._create_actions_section(content_frame)

    def _create_header_section(self, parent):
        """Create header with title and model selection using responsive layout."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x', padx=20, pady=(20, 10))

        # Configure grid for responsive layout
        header_frame.columnconfigure(0, weight=1)
        header_frame.columnconfigure(1, weight=0, minsize=300)

        # Title on the left
        title_label = ttk.Label(
            header_frame,
            text="Model Summary & Analysis",
            font=('Segoe UI', 24, 'bold')
        )
        title_label.grid(row=0, column=0, sticky='w')

        # Model selection on the right with responsive layout
        selection_frame = ttk.Frame(header_frame)
        selection_frame.grid(row=0, column=1, sticky='e', padx=(10, 0))
        
        # Configure selection frame for responsiveness
        selection_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(
            selection_frame,
            text="Select Model:",
            font=('Segoe UI', 12)
        ).grid(row=0, column=0, sticky='w', padx=(0, 10))

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.model_var,
            state='readonly',
            font=('Segoe UI', 11)
        )
        self.model_combo.grid(row=0, column=1, sticky='ew', padx=(0, 10))
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_selected)

        # Refresh button
        ttk.Button(
            selection_frame,
            text="🔄 Refresh",
            command=self._load_models
        ).grid(row=0, column=2, sticky='w')

        # Selected model info (full width below header)
        self.model_info_label = ttk.Label(
            parent,
            text="No model selected",
            font=('Segoe UI', 11),
            foreground=self.colors['text_secondary']
        )
        self.model_info_label.pack(fill='x', padx=20, pady=(0, 10))

    def _create_metrics_section(self, parent):
        """Create key metrics display with responsive layout."""
        metrics_frame = ttk.LabelFrame(
            parent,
            text="Key Performance Metrics",
            padding=15
        )
        metrics_frame.pack(fill='x', padx=20, pady=10)

        # Create responsive grid of metric cards
        self.metrics_grid = ttk.Frame(metrics_frame)
        self.metrics_grid.pack(fill='x')

        # Configure grid for responsive layout - 4 columns
        for i in range(4):
            self.metrics_grid.columnconfigure(i, weight=1, minsize=160)

        # Initialize metric cards with responsive positioning
        self.metric_cards = {}
        
        # Row 1: Basic metrics
        self.metric_cards['total_units'] = StatCard(
            self.metrics_grid,
            title="Total Units",
            value="--",
            unit="",
            color_scheme="default"
        )
        self.metric_cards['total_units'].grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.metric_cards['pass_rate'] = StatCard(
            self.metrics_grid,
            title="Overall Pass Rate",
            value="--",
            unit="%",
            color_scheme="success"
        )
        self.metric_cards['pass_rate'].grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        self.metric_cards['sigma_avg'] = StatCard(
            self.metrics_grid,
            title="Avg Sigma Gradient",
            value="--",
            unit="",
            color_scheme="warning"
        )
        self.metric_cards['sigma_avg'].grid(row=0, column=2, padx=5, pady=5, sticky='ew')

        self.metric_cards['recent_trend'] = StatCard(
            self.metrics_grid,
            title="7-Day Trend",
            value="--",
            unit="",
            color_scheme="default"
        )
        self.metric_cards['recent_trend'].grid(row=0, column=3, padx=5, pady=5, sticky='ew')

        # Row 2: Advanced metrics
        self.metric_cards['sigma_std'] = StatCard(
            self.metrics_grid,
            title="Sigma Std Dev",
            value="--",
            unit="",
            color_scheme="info"
        )
        self.metric_cards['sigma_std'].grid(row=1, column=0, padx=5, pady=5, sticky='ew')

        self.metric_cards['linearity_rate'] = StatCard(
            self.metrics_grid,
            title="Linearity Pass Rate",
            value="--",
            unit="%",
            color_scheme="success"
        )
        self.metric_cards['linearity_rate'].grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        self.metric_cards['resistance_avg'] = StatCard(
            self.metrics_grid,
            title="Avg Resistance Change",
            value="--",
            unit="%",
            color_scheme="warning"
        )
        self.metric_cards['resistance_avg'].grid(row=1, column=2, padx=5, pady=5, sticky='ew')

        self.metric_cards['high_risk'] = StatCard(
            self.metrics_grid,
            title="High Risk Units",
            value="--",
            unit="",
            color_scheme="danger"
        )
        self.metric_cards['high_risk'].grid(row=1, column=3, padx=5, pady=5, sticky='ew')

    def _create_trend_section(self, parent):
        """Create sigma trend chart section with responsive layout."""
        trend_frame = ttk.LabelFrame(
            parent,
            text="Sigma Gradient Trending",
            padding=15
        )
        trend_frame.pack(fill='x', padx=20, pady=10)

        # Chart controls with responsive layout
        controls_frame = ttk.Frame(trend_frame)
        controls_frame.pack(fill='x', pady=(0, 10))
        
        # Configure controls for responsive layout
        controls_frame.grid_columnconfigure(1, weight=1)
        controls_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(controls_frame, text="Time Range:").grid(row=0, column=0, sticky='w', padx=(0, 10))

        self.time_range_var = tk.StringVar(value="Last 30 days")
        time_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.time_range_var,
            values=["Last 7 days", "Last 30 days", "Last 90 days", "Last 6 months", "Last year", "All time"],
            state='readonly'
        )
        time_combo.grid(row=0, column=1, sticky='ew', padx=(0, 20))
        time_combo.bind('<<ComboboxSelected>>', self._update_trend_chart)

        ttk.Label(controls_frame, text="Chart Type:").grid(row=0, column=2, sticky='w', padx=(0, 10))

        self.chart_type_var = tk.StringVar(value="Scatter with Trend")
        chart_type_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.chart_type_var,
            values=["Scatter with Trend", "Line Plot", "Box Plot by Week"],
            state='readonly'
        )
        chart_type_combo.grid(row=0, column=3, sticky='ew')
        chart_type_combo.bind('<<ComboboxSelected>>', self._update_trend_chart)

        # Sigma trend chart with responsive sizing
        self.sigma_chart = ChartWidget(
            trend_frame,
            chart_type='scatter',
            title="Sigma Gradient Over Time"
            # Remove fixed figsize for responsive behavior
        )
        self.sigma_chart.pack(fill='x', pady=(10, 0), expand=True)

    def _create_analysis_section(self, parent):
        """Create additional analysis charts with responsive layout."""
        analysis_frame = ttk.LabelFrame(
            parent,
            text="Detailed Analysis",
            padding=15
        )
        analysis_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Create notebook for different analysis views with responsive sizing
        self.analysis_notebook = ttk.Notebook(analysis_frame)
        self.analysis_notebook.pack(fill='both', expand=True)

        # Distribution Analysis tab with responsive charts
        dist_frame = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(dist_frame, text="Distribution Analysis")

        self.distribution_chart = ChartWidget(
            dist_frame,
            chart_type='histogram',
            title="Sigma Gradient Distribution"
            # Remove fixed figsize for responsive behavior
        )
        self.distribution_chart.pack(fill='both', expand=True, padx=10, pady=10)

        # Pass/Fail Analysis tab with responsive charts
        passfail_frame = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(passfail_frame, text="Pass/Fail Analysis")

        self.passfail_chart = ChartWidget(
            passfail_frame,
            chart_type='bar',
            title="Pass/Fail Rate by Time Period"
            # Remove fixed figsize for responsive behavior
        )
        self.passfail_chart.pack(fill='both', expand=True, padx=10, pady=10)

        # Quality Correlation tab with responsive charts
        correlation_frame = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(correlation_frame, text="Quality Correlations")

        self.correlation_chart = ChartWidget(
            correlation_frame,
            chart_type='scatter',
            title="Sigma vs. Linearity Error Correlation"
            # Remove fixed figsize for responsive behavior
        )
        self.correlation_chart.pack(fill='both', expand=True, padx=10, pady=10)

    def _create_actions_section(self, parent):
        """Create export and print action buttons with responsive layout."""
        actions_frame = ttk.LabelFrame(
            parent,
            text="Export & Reports",
            padding=15
        )
        actions_frame.pack(fill='x', padx=20, pady=(10, 20))

        # Button container with responsive layout
        btn_container = ttk.Frame(actions_frame)
        btn_container.pack(fill='x')
        
        # Configure button container for responsive layout
        btn_container.grid_columnconfigure(0, weight=1)
        btn_container.grid_columnconfigure(1, weight=1)
        btn_container.grid_columnconfigure(2, weight=1)
        btn_container.grid_columnconfigure(3, weight=2)  # Extra space for stats

        # Export to Excel button
        self.export_excel_btn = ttk.Button(
            btn_container,
            text="📊 Export to Excel",
            command=self._export_to_excel,
            style='Primary.TButton',
            state='disabled'
        )
        self.export_excel_btn.grid(row=0, column=0, sticky='ew', padx=(0, 10))

        # Generate PDF Report button
        self.generate_report_btn = ttk.Button(
            btn_container,
            text="📄 Generate PDF Report",
            command=self._generate_pdf_report,
            state='disabled'
        )
        self.generate_report_btn.grid(row=0, column=1, sticky='ew', padx=(0, 10))

        # Export chart data button
        self.export_chart_btn = ttk.Button(
            btn_container,
            text="📈 Export Chart Data",
            command=self._export_chart_data,
            state='disabled'
        )
        self.export_chart_btn.grid(row=0, column=2, sticky='ew', padx=(0, 10))

        # Quick stats label with responsive positioning
        self.quick_stats_label = ttk.Label(
            btn_container,
            text="",
            font=('Segoe UI', 10),
            foreground=self.colors['text_secondary']
        )
        self.quick_stats_label.grid(row=0, column=3, sticky='e')

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

            self.model_combo['values'] = models
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
        self.model_info_label.config(text=f"Loading data for model: {model}...")
        self.logger.info(f"Loading model data for: {model}")

        # Clear existing charts
        try:
            self.sigma_chart.clear_chart()
            self.distribution_chart.clear_chart() 
            self.passfail_chart.clear_chart()
            self.correlation_chart.clear_chart()
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
                self.after(0, lambda: self.model_info_label.config(
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
            
            self.model_info_label.config(
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
            self.export_excel_btn.config(state='normal')
            self.generate_report_btn.config(state='normal')
            self.export_chart_btn.config(state='normal')

            # Update quick stats
            self.quick_stats_label.config(
                text=f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            self.logger.info(f"Successfully updated model display for {model} with {len(df)} data points")
            
        except Exception as e:
            self.logger.error(f"Error updating model display: {e}")
            self.model_info_label.config(
                text=f"Error displaying model data: {str(e)}"
            )
            # Still enable buttons if we have data
            if self.model_data is not None and len(self.model_data) > 0:
                self.export_excel_btn.config(state='normal')
                self.generate_report_btn.config(state='normal')
                self.export_chart_btn.config(state='normal')

    def _update_metrics(self, df: pd.DataFrame):
        """Update metric cards with calculated values."""
        # Basic metrics
        total_units = len(df['analysis_id'].unique())
        pass_rate = (df['track_status'] == 'Pass').mean() * 100
        sigma_avg = df['sigma_gradient'].mean()
        sigma_std = df['sigma_gradient'].std()

        # Advanced metrics
        linearity_rate = df['linearity_pass'].mean() * 100 if df['linearity_pass'].notna().any() else 0
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

        # Update cards
        self.metric_cards['total_units'].update_value(total_units)
        self.metric_cards['pass_rate'].update_value(f"{pass_rate:.1f}")
        self.metric_cards['sigma_avg'].update_value(f"{sigma_avg:.4f}")
        self.metric_cards['recent_trend'].update_value(trend_text)
        self.metric_cards['sigma_std'].update_value(f"{sigma_std:.4f}")
        self.metric_cards['linearity_rate'].update_value(f"{linearity_rate:.1f}")
        self.metric_cards['resistance_avg'].update_value(f"{resistance_avg:.1f}")
        self.metric_cards['high_risk'].update_value(high_risk)

        # Update colors based on values
        if pass_rate >= 95:
            self.metric_cards['pass_rate'].set_color_scheme('success')
        elif pass_rate >= 90:
            self.metric_cards['pass_rate'].set_color_scheme('warning')
        else:
            self.metric_cards['pass_rate'].set_color_scheme('danger')

    def _update_trend_chart(self, event=None):
        """Update the sigma trend chart."""
        if self.model_data is None or len(self.model_data) == 0:
            return

        df = self.model_data.copy()
        
        # Apply time range filter
        time_range = self.time_range_var.get()
        if time_range != "All time":
            days_map = {
                "Last 7 days": 7,
                "Last 30 days": 30,
                "Last 90 days": 90,
                "Last 6 months": 180,
                "Last year": 365
            }
            days_back = days_map.get(time_range, 30)
            cutoff_date = datetime.now() - timedelta(days=days_back)
            df = df[df['trim_date'] >= cutoff_date]

        if len(df) == 0:
            self.sigma_chart.clear_chart()
            return

        # Sort by date
        df = df.sort_values('trim_date')

        # Create chart based on selected type
        chart_type = self.chart_type_var.get()
        self.sigma_chart.clear_chart()

        try:
            if chart_type == "Scatter with Trend":
                # Scatter plot with trend line
                x_data = df['trim_date'].tolist()
                y_data = df['sigma_gradient'].tolist()
                
                # Color by pass/fail
                colors = ['pass' if status == 'Pass' else 'fail' for status in df['track_status']]
                
                self.sigma_chart.plot_scatter(
                    x_data=x_data,
                    y_data=y_data,
                    colors=colors,
                    xlabel="Trim Date",
                    ylabel="Sigma Gradient",
                    alpha=0.6
                )
                
                # Fix date formatting on x-axis
                try:
                    ax = self.sigma_chart.figure.axes[0]
                    
                    # Import date formatting utilities
                    from matplotlib.dates import DateFormatter, MonthLocator, DayLocator
                    import matplotlib.dates as mdates
                    
                    # Convert datetime objects to matplotlib dates for proper formatting
                    x_dates_mpl = [mdates.date2num(d) for d in x_data]
                    
                    # Set date formatter based on data range
                    date_range = (max(x_data) - min(x_data)).days
                    
                    if date_range <= 30:  # Less than a month
                        ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
                        ax.xaxis.set_major_locator(DayLocator(interval=max(1, date_range // 10)))
                    elif date_range <= 90:  # Less than 3 months
                        ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
                        ax.xaxis.set_major_locator(DayLocator(interval=max(1, date_range // 8)))
                    else:  # More than 3 months
                        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                        ax.xaxis.set_major_locator(MonthLocator())
                    
                    # Rotate dates for better readability
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Ensure the x-axis uses proper date formatting
                    ax.xaxis_date()
                    
                    # Ensure tight layout
                    self.sigma_chart.figure.tight_layout()
                    
                except Exception as e:
                    self.logger.warning(f"Could not format dates on trend chart: {e}")
                
                # Add legend for pass/fail colors
                try:
                    import matplotlib.patches as mpatches
                    pass_patch = mpatches.Patch(color=self.sigma_chart.qa_colors['pass'], label='Pass')
                    fail_patch = mpatches.Patch(color=self.sigma_chart.qa_colors['fail'], label='Fail')
                    ax.legend(handles=[pass_patch, fail_patch], loc='upper right')
                except Exception as e:
                    self.logger.warning(f"Could not add legend to trend chart: {e}")
                
                # Add trend line with better error handling
                if len(df) > 2:  # Need at least 3 points for trend
                    try:
                        # Convert dates to numeric for trend calculation
                        x_numeric = [mdates.date2num(d) for d in x_data]
                        y_numeric = df['sigma_gradient'].values
                        
                        # Check for valid data
                        if len(x_numeric) == len(y_numeric) and len(set(x_numeric)) > 1:
                            # Filter out any NaN values
                            valid_mask = ~(np.isnan(x_numeric) | np.isnan(y_numeric))
                            x_clean = np.array(x_numeric)[valid_mask]
                            y_clean = np.array(y_numeric)[valid_mask]
                            
                            if len(x_clean) > 2 and len(set(x_clean)) > 1:
                                # Use robust polynomial fitting with error handling
                                try:
                                    z = np.polyfit(x_clean, y_clean, 1)
                                    trend_line = np.poly1d(z)
                                    
                                    # Create trend line data using original x data for proper plotting
                                    trend_y = [trend_line(x) for x in x_numeric]
                                    
                                    self.sigma_chart.plot_line(
                                        x_data=x_data,
                                        y_data=trend_y,
                                        label="Trend",
                                        color='trend',
                                        linewidth=2
                                    )
                                except np.linalg.LinAlgError:
                                    # SVD didn't converge or rank deficient
                                    self.logger.warning("Could not compute trend line: numerical issues")
                                except Exception as e:
                                    self.logger.warning(f"Trend line calculation failed: {e}")
                            else:
                                self.logger.warning("Insufficient valid data points for trend line")
                        else:
                            self.logger.warning("Invalid data for trend calculation")
                    except Exception as e:
                        self.logger.error(f"Error calculating trend line: {e}")

            elif chart_type == "Line Plot":
                # Daily averages line plot
                try:
                    daily_avg = df.groupby(df['trim_date'].dt.date)['sigma_gradient'].mean().reset_index()
                    
                    # Convert date column back to datetime for proper plotting
                    daily_avg['trim_date'] = pd.to_datetime(daily_avg['trim_date'])
                    
                    x_data_line = daily_avg['trim_date'].tolist()
                    y_data_line = daily_avg['sigma_gradient'].tolist()
                    
                    self.sigma_chart.plot_line(
                        x_data=x_data_line,
                        y_data=y_data_line,
                        label="Daily Average",
                        color='primary',
                        marker='o'
                    )
                    
                    # Fix date formatting on x-axis for line plot
                    try:
                        ax = self.sigma_chart.figure.axes[0]
                        from matplotlib.dates import DateFormatter, MonthLocator, DayLocator
                        import matplotlib.dates as mdates
                        
                        # Ensure x-axis uses proper date formatting
                        ax.xaxis_date()
                        
                        date_range = len(daily_avg)
                        
                        if date_range <= 30:
                            ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
                        else:
                            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                            ax.xaxis.set_major_locator(MonthLocator())
                        
                        ax.tick_params(axis='x', rotation=45)
                        self.sigma_chart.figure.tight_layout()
                        
                    except Exception as e:
                        self.logger.warning(f"Could not format dates on line chart: {e}")
                    
                    # Add trend line to line plot
                    if len(x_data_line) > 2:
                        try:
                            x_numeric = [mdates.date2num(d) for d in x_data_line]
                            y_numeric = np.array(y_data_line)
                            
                            # Filter out any NaN values
                            valid_mask = ~(np.isnan(x_numeric) | np.isnan(y_numeric))
                            x_clean = np.array(x_numeric)[valid_mask]
                            y_clean = y_numeric[valid_mask]
                            
                            if len(x_clean) > 2:
                                try:
                                    z = np.polyfit(x_clean, y_clean, 1)
                                    trend_line = np.poly1d(z)
                                    trend_y = [trend_line(x) for x in x_numeric]
                                    
                                    self.sigma_chart.plot_line(
                                        x_data=x_data_line,
                                        y_data=trend_y,
                                        label="Trend",
                                        color='trend',
                                        linewidth=2,
                                        linestyle='--'
                                    )
                                except np.linalg.LinAlgError:
                                    # SVD didn't converge or rank deficient
                                    self.logger.warning("Could not compute trend line for line plot")
                                except Exception as e:
                                    self.logger.warning(f"Trend line calculation failed for line plot: {e}")
                        except Exception as e:
                            self.logger.error(f"Error calculating trend line for line plot: {e}")
                        
                except Exception as e:
                    self.logger.error(f"Error creating line plot: {e}")

            elif chart_type == "Box Plot by Week":
                # Weekly box plots
                try:
                    df['week'] = df['trim_date'].dt.to_period('W').astype(str)
                    weeks = sorted(df['week'].unique())
                    
                    if len(weeks) > 0:
                        data_by_week = []
                        valid_weeks = []
                        
                        for week in weeks:
                            week_data = df[df['week'] == week]['sigma_gradient'].dropna().tolist()
                            if len(week_data) > 0:  # Only include weeks with data
                                data_by_week.append(week_data)
                                valid_weeks.append(week)
                        
                        if len(data_by_week) > 0:
                            self.sigma_chart.plot_box(
                                data=data_by_week,
                                labels=valid_weeks,
                                xlabel="Week",
                                ylabel="Sigma Gradient"
                            )
                        else:
                            self.logger.warning("No valid weekly data for box plot")
                    else:
                        self.logger.warning("No weekly data available")
                except Exception as e:
                    self.logger.error(f"Error creating box plot: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error updating trend chart: {e}")
            # Show error message on chart
            try:
                ax = self.sigma_chart.figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Chart Error: {str(e)[:50]}...', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral"))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                self.sigma_chart.canvas.draw()
            except:
                pass

    def _update_analysis_charts(self):
        """Update the additional analysis charts."""
        if self.model_data is None or len(self.model_data) == 0:
            self.logger.warning("No model data available for analysis charts")
            return

        df = self.model_data

        try:
            # Distribution chart
            self.distribution_chart.clear_chart()
            sigma_data = df['sigma_gradient'].dropna()
            if len(sigma_data) > 0:
                self.distribution_chart.plot_histogram(
                    data=sigma_data.tolist(),
                    bins=min(30, max(10, len(sigma_data) // 5)),  # Adaptive bin count
                    color='primary',
                    xlabel="Sigma Gradient",
                    ylabel="Frequency"
                )
                self.logger.info(f"Updated distribution chart with {len(sigma_data)} data points")
            else:
                self.logger.warning("No valid sigma gradient data for distribution chart")

            # Pass/Fail analysis by month
            self.passfail_chart.clear_chart()
            try:
                # Create month-year periods for better grouping
                df_copy = df.copy()
                df_copy['month_year'] = df_copy['trim_date'].dt.to_period('M')
                
                monthly_stats = df_copy.groupby('month_year').agg({
                    'track_status': lambda x: (x == 'Pass').mean() * 100 if len(x) > 0 else 0
                }).reset_index()
                
                if len(monthly_stats) > 0:
                    monthly_stats['month_str'] = monthly_stats['month_year'].astype(str)
                    
                    # Color coding based on pass rate
                    colors = []
                    for rate in monthly_stats['track_status']:
                        if rate >= 95:
                            colors.append('pass')
                        elif rate >= 90:
                            colors.append('warning')
                        else:
                            colors.append('fail')
                    
                    self.passfail_chart.plot_bar(
                        categories=monthly_stats['month_str'].tolist(),
                        values=monthly_stats['track_status'].tolist(),
                        colors=colors,
                        xlabel="Month",
                        ylabel="Pass Rate (%)"
                    )
                    self.logger.info(f"Updated pass/fail chart with {len(monthly_stats)} months")
                else:
                    self.logger.warning("No monthly statistics available for pass/fail chart")
                    
            except Exception as e:
                self.logger.error(f"Error creating pass/fail chart: {e}")

            # Correlation chart
            self.correlation_chart.clear_chart()
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
                        x_data = sigma_clean.loc[common_idx].tolist()
                        y_data = linearity_clean.loc[common_idx].tolist()
                        
                        # Color by pass/fail status
                        status_data = df.loc[common_idx, 'track_status']
                        colors = ['pass' if status == 'Pass' else 'fail' for status in status_data]
                        
                        self.correlation_chart.plot_scatter(
                            x_data=x_data,
                            y_data=y_data,
                            colors=colors,
                            xlabel="Sigma Gradient",
                            ylabel=f"Linearity Error ({linearity_col.replace('_', ' ').title()})",
                            alpha=0.6
                        )
                        
                        # Add legend for pass/fail colors
                        ax = self.correlation_chart.figure.axes[0]
                        
                        # Create legend handles
                        import matplotlib.patches as mpatches
                        pass_patch = mpatches.Patch(color=self.correlation_chart.qa_colors['pass'], label='Pass')
                        fail_patch = mpatches.Patch(color=self.correlation_chart.qa_colors['fail'], label='Fail')
                        ax.legend(handles=[pass_patch, fail_patch], loc='upper right')
                        
                        # Calculate and display correlation
                        correlation = np.corrcoef(x_data, y_data)[0, 1]
                        if not np.isnan(correlation):
                            # Add correlation text to chart
                            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                                   transform=ax.transAxes, fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
                        
                        self.logger.info(f"Updated correlation chart with {len(x_data)} data points")
                    else:
                        self.logger.warning(f"Insufficient data for correlation chart: {len(common_idx)} points")
                        # Show a message on the chart
                        ax = self.correlation_chart.figure.add_subplot(111)
                        ax.text(0.5, 0.5, 'Insufficient data for correlation analysis\n(Need at least 5 data points)', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=12,
                               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        self.correlation_chart.canvas.draw()
                else:
                    self.logger.warning("No linearity error data available for correlation chart")
                    # Show a message on the chart
                    ax = self.correlation_chart.figure.add_subplot(111)
                    ax.text(0.5, 0.5, 'No linearity data available\nfor correlation analysis', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    self.correlation_chart.canvas.draw()
                    
            except Exception as e:
                self.logger.error(f"Error creating correlation chart: {e}")

        except Exception as e:
            self.logger.error(f"Error updating analysis charts: {e}")
            messagebox.showerror("Chart Error", f"Failed to update analysis charts:\n{str(e)}")

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