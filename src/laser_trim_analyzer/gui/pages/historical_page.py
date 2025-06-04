"""
Historical Data Page for Laser Trim Analyzer

Provides interface for querying and analyzing historical QA data
with advanced analytics, charts, and export functionality.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import logging
import threading
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from laser_trim_analyzer.core.models import AnalysisResult, FileMetadata, AnalysisStatus
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.gui.pages.base_page import BasePage
from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
from laser_trim_analyzer.gui.widgets.metric_card import MetricCard
from laser_trim_analyzer.gui.widgets import add_mousewheel_support
from laser_trim_analyzer.utils.date_utils import safe_datetime_convert

# Get logger
logger = logging.getLogger(__name__)

class HistoricalPage(BasePage):
    """Advanced historical data analysis page with analytics features."""

    def __init__(self, parent, main_window):
        self.current_data = None
        self.analytics_results = {}
        self.trend_analysis_data = {}
        self.correlation_matrix = None
        super().__init__(parent, main_window)

    def _create_page(self):
        """Create enhanced historical page content with advanced analytics."""
        # Main scrollable container
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create enhanced sections
        self._create_header()
        self._create_query_section_ctk()
        self._create_analytics_dashboard()
        self._create_results_section_ctk()
        self._create_charts_section_ctk()
        self._create_advanced_analytics_section()

    def _create_header(self):
        """Create enhanced header section."""
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.pack(fill='x', pady=(0, 20))

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Advanced Historical Data Analytics",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=15)

        # Analytics status indicator
        self.analytics_status_frame = ctk.CTkFrame(self.header_frame)
        self.analytics_status_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        self.analytics_status_label = ctk.CTkLabel(
            self.analytics_status_frame,
            text="Analytics Status: Ready",
            font=ctk.CTkFont(size=12)
        )
        self.analytics_status_label.pack(side='left', padx=10, pady=10)
        
        self.analytics_indicator = ctk.CTkLabel(
            self.analytics_status_frame,
            text="●",
            font=ctk.CTkFont(size=16),
            text_color="green"
        )
        self.analytics_indicator.pack(side='right', padx=10, pady=10)

    def _create_analytics_dashboard(self):
        """Create quick analytics dashboard."""
        self.dashboard_frame = ctk.CTkFrame(self.main_container)
        self.dashboard_frame.pack(fill='x', pady=(0, 20))

        self.dashboard_label = ctk.CTkLabel(
            self.dashboard_frame,
            text="Analytics Dashboard:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.dashboard_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Dashboard metrics container
        self.metrics_container = ctk.CTkFrame(self.dashboard_frame)
        self.metrics_container.pack(fill='x', padx=15, pady=(0, 15))

        # Row 1 of metrics
        metrics_row1 = ctk.CTkFrame(self.metrics_container)
        metrics_row1.pack(fill='x', padx=10, pady=(10, 5))

        self.total_records_card = MetricCard(
            metrics_row1,
            title="Total Records",
            value="--",
            color_scheme="info"
        )
        self.total_records_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

        self.pass_rate_card = MetricCard(
            metrics_row1,
            title="Overall Pass Rate",
            value="--",
            color_scheme="success"
        )
        self.pass_rate_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

        self.trend_direction_card = MetricCard(
            metrics_row1,
            title="Trend Direction",
            value="--",
            color_scheme="neutral"
        )
        self.trend_direction_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

        self.prediction_accuracy_card = MetricCard(
            metrics_row1,
            title="Prediction Accuracy",
            value="--",
            color_scheme="warning"
        )
        self.prediction_accuracy_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

        # Row 2 of metrics
        metrics_row2 = ctk.CTkFrame(self.metrics_container)
        metrics_row2.pack(fill='x', padx=10, pady=(5, 10))

        self.sigma_correlation_card = MetricCard(
            metrics_row2,
            title="Sigma Correlation",
            value="--",
            color_scheme="info"
        )
        self.sigma_correlation_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

        self.linearity_stability_card = MetricCard(
            metrics_row2,
            title="Linearity Stability",
            value="--",
            color_scheme="success"
        )
        self.linearity_stability_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

        self.quality_score_card = MetricCard(
            metrics_row2,
            title="Quality Score",
            value="--",
            color_scheme="warning"
        )
        self.quality_score_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

        self.anomaly_count_card = MetricCard(
            metrics_row2,
            title="Anomalies Detected",
            value="--",
            color_scheme="danger"
        )
        self.anomaly_count_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

    def _create_advanced_analytics_section(self):
        """Create advanced analytics section."""
        self.advanced_frame = ctk.CTkFrame(self.main_container)
        self.advanced_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.advanced_label = ctk.CTkLabel(
            self.advanced_frame,
            text="Advanced Analytics:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.advanced_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Analytics controls
        controls_frame = ctk.CTkFrame(self.advanced_frame)
        controls_frame.pack(fill='x', padx=15, pady=(0, 10))

        self.trend_analysis_btn = ctk.CTkButton(
            controls_frame,
            text="📈 Trend Analysis",
            command=self._run_trend_analysis,
            width=140,
            height=40
        )
        self.trend_analysis_btn.pack(side='left', padx=(10, 5), pady=10)

        self.correlation_analysis_btn = ctk.CTkButton(
            controls_frame,
            text="🔗 Correlation Analysis",
            command=self._run_correlation_analysis,
            width=140,
            height=40
        )
        self.correlation_analysis_btn.pack(side='left', padx=(5, 5), pady=10)

        self.statistical_summary_btn = ctk.CTkButton(
            controls_frame,
            text="📊 Statistical Summary",
            command=self._generate_statistical_summary,
            width=140,
            height=40
        )
        self.statistical_summary_btn.pack(side='left', padx=(5, 5), pady=10)

        self.predictive_analysis_btn = ctk.CTkButton(
            controls_frame,
            text="🔮 Predictive Analysis",
            command=self._run_predictive_analysis,
            width=140,
            height=40
        )
        self.predictive_analysis_btn.pack(side='left', padx=(5, 5), pady=10)

        self.anomaly_detection_btn = ctk.CTkButton(
            controls_frame,
            text="🚨 Detect Anomalies",
            command=self._detect_anomalies,
            width=140,
            height=40
        )
        self.anomaly_detection_btn.pack(side='left', padx=(5, 10), pady=10)

        # Analytics results tabview
        self.analytics_tabview = ctk.CTkTabview(self.advanced_frame)
        self.analytics_tabview.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Add analytics tabs
        self.analytics_tabview.add("Trend Analysis")
        self.analytics_tabview.add("Correlation Matrix")
        self.analytics_tabview.add("Statistical Summary")
        self.analytics_tabview.add("Predictive Models")
        self.analytics_tabview.add("Anomaly Detection")

        # Trend analysis tab
        self.trend_analysis_chart = ChartWidget(
            self.analytics_tabview.tab("Trend Analysis"),
            chart_type='line',
            title="Performance Trends Over Time",
            figsize=(12, 6)
        )
        self.trend_analysis_chart.pack(fill='both', expand=True, padx=5, pady=5)

        # Correlation matrix tab
        self.correlation_chart = ChartWidget(
            self.analytics_tabview.tab("Correlation Matrix"),
            chart_type='heatmap',
            title="Parameter Correlation Matrix",
            figsize=(10, 8)
        )
        self.correlation_chart.pack(fill='both', expand=True, padx=5, pady=5)

        # Statistical summary tab
        self.stats_display = ctk.CTkTextbox(
            self.analytics_tabview.tab("Statistical Summary"),
            height=400,
            state="disabled"
        )
        self.stats_display.pack(fill='both', expand=True, padx=5, pady=5)

        # Predictive models tab
        predictive_frame = ctk.CTkFrame(self.analytics_tabview.tab("Predictive Models"))
        predictive_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.prediction_chart = ChartWidget(
            predictive_frame,
            chart_type='scatter',
            title="Actual vs Predicted Values",
            figsize=(10, 6)
        )
        self.prediction_chart.pack(fill='both', expand=True, padx=5, pady=5)

        # Anomaly detection tab
        self.anomaly_display = ctk.CTkTextbox(
            self.analytics_tabview.tab("Anomaly Detection"),
            height=400,
            state="disabled"
        )
        self.anomaly_display.pack(fill='both', expand=True, padx=5, pady=5)

    def _create_query_section_ctk(self):
        """Create query filters section (matching batch processing theme)."""
        self.query_frame = ctk.CTkFrame(self.main_container)
        self.query_frame.pack(fill='x', pady=(0, 20))

        self.query_label = ctk.CTkLabel(
            self.query_frame,
            text="Query Filters:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.query_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Filters container
        self.filters_container = ctk.CTkFrame(self.query_frame)
        self.filters_container.pack(fill='x', padx=15, pady=(0, 15))

        # First row of filters
        filter_row1 = ctk.CTkFrame(self.filters_container)
        filter_row1.pack(fill='x', padx=10, pady=(10, 5))

        # Model filter
        model_label = ctk.CTkLabel(filter_row1, text="Model:")
        model_label.pack(side='left', padx=10, pady=10)

        self.model_var = tk.StringVar()
        self.model_entry = ctk.CTkEntry(
            filter_row1,
            textvariable=self.model_var,
            width=120,
            height=30
        )
        self.model_entry.pack(side='left', padx=(0, 20), pady=10)

        # Serial filter
        serial_label = ctk.CTkLabel(filter_row1, text="Serial:")
        serial_label.pack(side='left', padx=10, pady=10)

        self.serial_var = tk.StringVar()
        self.serial_entry = ctk.CTkEntry(
            filter_row1,
            textvariable=self.serial_var,
            width=120,
            height=30
        )
        self.serial_entry.pack(side='left', padx=(0, 20), pady=10)

        # Date range
        date_label = ctk.CTkLabel(filter_row1, text="Date Range:")
        date_label.pack(side='left', padx=10, pady=10)

        self.date_range_var = tk.StringVar(value="Last 30 days")
        self.date_combo = ctk.CTkComboBox(
            filter_row1,
            variable=self.date_range_var,
            values=[
                "Today", "Last 7 days", "Last 30 days",
                "Last 90 days", "Last year", "All time"
            ],
            width=120,
            height=30
        )
        self.date_combo.pack(side='left', padx=(0, 10), pady=10)

        # Second row of filters
        filter_row2 = ctk.CTkFrame(self.filters_container)
        filter_row2.pack(fill='x', padx=10, pady=(5, 10))

        # Status filter
        status_label = ctk.CTkLabel(filter_row2, text="Status:")
        status_label.pack(side='left', padx=10, pady=10)

        self.status_var = tk.StringVar(value="All")
        self.status_combo = ctk.CTkComboBox(
            filter_row2,
            variable=self.status_var,
            values=["All", "Pass", "Fail", "Warning"],
            width=100,
            height=30
        )
        self.status_combo.pack(side='left', padx=(0, 20), pady=10)

        # Risk filter
        risk_label = ctk.CTkLabel(filter_row2, text="Risk:")
        risk_label.pack(side='left', padx=10, pady=10)

        self.risk_var = tk.StringVar(value="All")
        self.risk_combo = ctk.CTkComboBox(
            filter_row2,
            variable=self.risk_var,
            values=["All", "High", "Medium", "Low"],
            width=100,
            height=30
        )
        self.risk_combo.pack(side='left', padx=(0, 20), pady=10)

        # Limit filter
        limit_label = ctk.CTkLabel(filter_row2, text="Limit:")
        limit_label.pack(side='left', padx=10, pady=10)

        self.limit_var = tk.StringVar(value="100")
        self.limit_combo = ctk.CTkComboBox(
            filter_row2,
            variable=self.limit_var,
            values=["50", "100", "500", "1000", "All"],
            width=100,
            height=30
        )
        self.limit_combo.pack(side='left', padx=(0, 10), pady=10)

        # Action buttons
        button_frame = ctk.CTkFrame(self.filters_container)
        button_frame.pack(fill='x', padx=10, pady=(10, 10))

        self.query_btn = ctk.CTkButton(
            button_frame,
            text="Run Query",
            command=self._run_query,
            width=120,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="blue",
            hover_color="darkblue"
        )
        self.query_btn.pack(side='left', padx=(10, 10), pady=10)

        clear_btn = ctk.CTkButton(
            button_frame,
            text="Clear Filters",
            command=self._clear_filters,
            width=120,
            height=40
        )
        clear_btn.pack(side='left', padx=(0, 10), pady=10)

        export_btn = ctk.CTkButton(
            button_frame,
            text="Export Results",
            command=self._export_results,
            width=120,
            height=40
        )
        export_btn.pack(side='left', padx=(0, 10), pady=10)

    def _create_results_section_ctk(self):
        """Create results display section (matching batch processing theme)."""
        self.results_frame = ctk.CTkFrame(self.main_container)
        self.results_frame.pack(fill='x', pady=(0, 20))

        self.results_label = ctk.CTkLabel(
            self.results_frame,
            text="Query Results:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.results_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Results display
        self.results_display = ctk.CTkTextbox(
            self.results_frame,
            height=200,
            state="disabled"
        )
        self.results_display.pack(fill='both', expand=True, padx=15, pady=(0, 15))

    def _create_charts_section_ctk(self):
        """Create charts section (matching batch processing theme)."""
        self.charts_frame = ctk.CTkFrame(self.main_container)
        self.charts_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.charts_label = ctk.CTkLabel(
            self.charts_frame,
            text="Data Visualization:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.charts_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Charts container
        self.charts_container = ctk.CTkFrame(self.charts_frame)
        self.charts_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Chart tabs
        self.chart_tabview = ctk.CTkTabview(self.charts_container)
        self.chart_tabview.pack(fill='both', expand=True, padx=10, pady=10)

        # Add tabs
        self.chart_tabview.add("Pass Rate Trend")
        self.chart_tabview.add("Sigma Distribution")
        self.chart_tabview.add("Model Comparison")

        # Create actual chart widgets instead of placeholder labels
        try:
            from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
            
            self.trend_chart = ChartWidget(
                self.chart_tabview.tab("Pass Rate Trend"),
                chart_type='line',
                title="Pass Rate Over Time",
                figsize=(10, 4)
            )
            self.trend_chart.pack(fill='both', expand=True)
            
            self.dist_chart = ChartWidget(
                self.chart_tabview.tab("Sigma Distribution"),
                chart_type='histogram',
                title="Sigma Gradient Distribution",
                figsize=(10, 4)
            )
            self.dist_chart.pack(fill='both', expand=True)
            
            self.comp_chart = ChartWidget(
                self.chart_tabview.tab("Model Comparison"),
                chart_type='bar',
                title="Pass Rate by Model",
                figsize=(10, 4)
            )
            self.comp_chart.pack(fill='both', expand=True)
            
        except ImportError as e:
            logger.warning(f"Could not create chart widgets: {e}")
            # Fallback to placeholder labels
            self.trend_chart_label = ctk.CTkLabel(
                self.chart_tabview.tab("Pass Rate Trend"),
                text="Chart widgets not available",
                font=ctk.CTkFont(size=12)
            )
            self.trend_chart_label.pack(expand=True)

            self.distribution_chart_label = ctk.CTkLabel(
                self.chart_tabview.tab("Sigma Distribution"),
                text="Chart widgets not available",
                font=ctk.CTkFont(size=12)
            )
            self.distribution_chart_label.pack(expand=True)

            self.comparison_chart_label = ctk.CTkLabel(
                self.chart_tabview.tab("Model Comparison"),
                text="Chart widgets not available",
                font=ctk.CTkFont(size=12)
            )
            self.comparison_chart_label.pack(expand=True)
            
            # Set chart objects to None for safe access
            self.trend_chart = None
            self.dist_chart = None
            self.comp_chart = None

    def _run_query(self):
        """Run database query with current filters."""
        if not self.db_manager:
            messagebox.showerror("Error", "Database not connected")
            return

        # Update UI and run query in background
        self.query_btn.configure(state='disabled', text='Querying...')
        thread = threading.Thread(target=self._run_query_background, daemon=True)
        thread.start()

    def _run_query_background(self):
        """Run database query in background thread."""
        try:
            # Get filter values - use the actual variables that exist in the UI
            model = self.model_var.get().strip() if self.model_var.get().strip() else None
            serial = self.serial_var.get().strip() if self.serial_var.get().strip() else None
            status = self.status_var.get() if self.status_var.get() != "All" else None
            risk = self.risk_var.get() if self.risk_var.get() != "All" else None
            
            # Get date range
            start_date = None
            end_date = None
            date_range = self.date_range_var.get()
            
            if date_range != "All time":
                days_map = {
                    "Today": 1,
                    "Last 7 days": 7,
                    "Last 30 days": 30,
                    "Last 90 days": 90,
                    "Last year": 365
                }
                days_back = days_map.get(date_range, 30)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)

            # Get limit
            limit_str = self.limit_var.get()
            limit = None if limit_str == "All" else int(limit_str)

            # Query database using the correct parameters
            results = self.db_manager.get_historical_data(
                model=model,
                serial=serial,
                start_date=start_date,
                end_date=end_date,
                status=status,
                risk_category=risk,
                limit=limit,
                include_tracks=True
            )

            # Update UI in main thread
            self.after(0, self._display_results, results)
            self.after(0, lambda: self.query_btn.configure(state='normal', text='Run Query'))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Query Error", f"Failed to query database:\n{str(e)}"))
            self.after(0, lambda: self.query_btn.configure(state='normal', text='Run Query'))

    def _display_results(self, results):
        """Display query results in the CTk textbox."""
        try:
            # Clear existing results
            self.results_display.configure(state="normal")
            self.results_display.delete('1.0', 'end')
            
            if not results:
                self.results_display.insert('1.0', "No data found matching the criteria")
                self.results_display.configure(state="disabled")
                return
                
            # Display summary
            total_count = len(results)
            pass_count = sum(1 for r in results if r.overall_status.value == "Pass")
            fail_count = total_count - pass_count
            pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0
            
            summary = f"Query Results Summary:\n"
            summary += f"Total Records: {total_count}\n"
            summary += f"Pass: {pass_count} ({pass_rate:.1f}%)\n"
            summary += f"Fail: {fail_count} ({100-pass_rate:.1f}%)\n\n"
            
            # Display detailed results
            summary += "Detailed Results:\n"
            summary += "-" * 80 + "\n"
            summary += f"{'Date':<12} {'Model':<8} {'Serial':<12} {'Status':<8} {'Sigma':<10} {'Pass':<6}\n"
            summary += "-" * 80 + "\n"
            
            for result in results:
                date_str = result.timestamp.strftime('%Y-%m-%d') if hasattr(result, 'timestamp') else 'Unknown'
                model = getattr(result, 'model', 'Unknown')[:8]
                serial = getattr(result, 'serial', 'Unknown')[:12]
                status = result.overall_status.value[:8]
                
                # Get sigma and pass info from first track
                sigma = "N/A"
                sigma_pass = "N/A"
                if result.tracks and len(result.tracks) > 0:
                    track = result.tracks[0]
                    if hasattr(track, 'sigma_gradient') and track.sigma_gradient is not None:
                        sigma = f"{track.sigma_gradient:.4f}"
                    if hasattr(track, 'sigma_pass'):
                        sigma_pass = "✓" if track.sigma_pass else "✗"
                
                line = f"{date_str:<12} {model:<8} {serial:<12} {status:<8} {sigma:<10} {sigma_pass:<6}\n"
                summary += line
            
            self.results_display.insert('1.0', summary)
            self.results_display.configure(state="disabled")
            
            # Store current data for potential export
            self.current_data = results
            
            # CRITICAL FIX: Update charts when data is loaded
            try:
                self._update_charts(results)
            except Exception as e:
                logger.error(f"Error updating charts: {e}")
            
            logger.info(f"Displayed {len(results)} query results")
            
        except Exception as e:
            logger.error(f"Error displaying results: {e}")
            self.results_display.configure(state="normal")
            self.results_display.delete('1.0', 'end')
            self.results_display.insert('1.0', f"Error displaying results: {str(e)}")
            self.results_display.configure(state="disabled")

    def _update_charts(self, results):
        """Update all charts with query results."""
        if not results:
            return

        try:
            # Convert database results to DataFrame format for chart consumption
            chart_data = []
            for result in results:
                record = {
                    'date': result.timestamp if hasattr(result, 'timestamp') else None,
                    'file_date': getattr(result, 'file_date', None),
                    'model': getattr(result, 'model', 'Unknown'),
                    'serial': getattr(result, 'serial', 'Unknown'),
                    'status': result.overall_status.value,
                    'sigma_gradient': None
                }
                
                # Extract sigma gradient from first track if available
                if result.tracks and len(result.tracks) > 0:
                    track = result.tracks[0]
                    if hasattr(track, 'sigma_gradient') and track.sigma_gradient is not None:
                        record['sigma_gradient'] = track.sigma_gradient
                
                chart_data.append(record)
            
            # Store as DataFrame for chart methods
            import pandas as pd
            self.current_data = pd.DataFrame(chart_data)
            
            # Update individual charts
            self._update_trend_chart()
            self._update_distribution_chart()
            self._update_comparison_chart()
            
        except Exception as e:
            logger.error(f"Error preparing chart data: {e}")

    def _update_trend_chart(self):
        """Update pass rate trend chart."""
        if self.current_data is None or len(self.current_data) == 0 or not hasattr(self, 'trend_chart') or self.trend_chart is None:
            return

        # Group by date and calculate pass rate
        df = self.current_data.copy()
        # Use file_date if available, otherwise timestamp
        df['date'] = pd.to_datetime(df['file_date'].fillna(df['timestamp'])).dt.date

        daily_stats = df.groupby('date').agg({
            'status': lambda x: (x == 'Pass').mean() * 100
        }).reset_index()
        daily_stats.columns = ['date', 'pass_rate']

        # Sort by date
        daily_stats = daily_stats.sort_values('date')

        # Clear and plot
        self.trend_chart.clear_chart()

        if len(daily_stats) > 1:
            self.trend_chart.plot_line(
                x_data=daily_stats['date'].tolist(),
                y_data=daily_stats['pass_rate'].tolist(),
                label="Pass Rate",
                color='primary',
                marker='o',
                xlabel="Date",
                ylabel="Pass Rate (%)"
            )

            # Add average line
            avg_pass_rate = daily_stats['pass_rate'].mean()
            self.trend_chart.add_threshold_lines(
                {'Average': avg_pass_rate},
                orientation='horizontal'
            )

    def _update_distribution_chart(self):
        """Update sigma gradient distribution chart."""
        if (self.current_data is None or 'sigma_gradient' not in self.current_data.columns or 
            not hasattr(self, 'dist_chart') or self.dist_chart is None):
            return

        # Get sigma values
        sigma_values = self.current_data['sigma_gradient'].dropna()

        if len(sigma_values) == 0:
            return

        # Clear and plot
        self.dist_chart.clear_chart()
        self.dist_chart.plot_histogram(
            data=sigma_values.tolist(),
            bins=30,
            color='primary',
            xlabel="Sigma Gradient",
            ylabel="Frequency"
        )

    def _update_comparison_chart(self):
        """Update model comparison chart."""
        if (self.current_data is None or len(self.current_data) == 0 or 
            not hasattr(self, 'comp_chart') or self.comp_chart is None):
            return

        # Calculate pass rate by model
        model_stats = self.current_data.groupby('model').agg({
            'status': [
                lambda x: (x == 'Pass').mean() * 100,
                'count'
            ]
        }).reset_index()

        model_stats.columns = ['model', 'pass_rate', 'count']

        # Filter models with sufficient data
        model_stats = model_stats[model_stats['count'] >= 5]

        if len(model_stats) == 0:
            return

        # Sort by pass rate
        model_stats = model_stats.sort_values('pass_rate', ascending=False)

        # Determine colors based on pass rate
        colors = []
        for rate in model_stats['pass_rate']:
            if rate >= 95:
                colors.append('pass')
            elif rate >= 90:
                colors.append('warning')
            else:
                colors.append('fail')

        # Clear and plot
        self.comp_chart.clear_chart()
        self.comp_chart.plot_bar(
            categories=model_stats['model'].tolist(),
            values=model_stats['pass_rate'].tolist(),
            colors=colors,
            xlabel="Model",
            ylabel="Pass Rate (%)"
        )

    def _get_days_back(self) -> Optional[int]:
        """Convert date range selection to days."""
        date_range = self.date_range_var.get()

        mapping = {
            "Today": 1,
            "Last 7 days": 7,
            "Last 30 days": 30,
            "Last 90 days": 90,
            "Last year": 365,
            "All time": None
        }

        return mapping.get(date_range, 30)

    def _clear_filters(self):
        """Clear all filter inputs."""
        self.model_var.set("")
        self.serial_var.set("")
        self.date_range_var.set("Last 30 days")
        self.status_var.set("All")
        self.risk_var.set("All")
        self.limit_var.set("100")

    def _export_results(self):
        """Export current results to file."""
        if self.current_data is None or len(self.current_data) == 0:
            messagebox.showwarning("No Data", "No data to export")
            return

        # Ask for file location
        filename = filedialog.asksaveasfilename(
            defaultextension='.xlsx',
            filetypes=[
                ('Excel files', '*.xlsx'),
                ('CSV files', '*.csv'),
                ('All files', '*.*')
            ],
            initialfile=f'historical_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )

        if not filename:
            return

        try:
            if filename.endswith('.xlsx'):
                # Export to Excel with formatting
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    self.current_data.to_excel(writer, sheet_name='Historical Data', index=False)

                    # Add summary sheet
                    summary = pd.DataFrame({
                        'Metric': ['Total Records', 'Pass Rate', 'Average Sigma', 'Date Range'],
                        'Value': [
                            len(self.current_data),
                            f"{(self.current_data['status'] == 'Pass').mean() * 100:.2f}%",
                            f"{self.current_data['sigma_gradient'].mean():.6f}",
                            f"{self.current_data['timestamp'].min()} to {self.current_data['timestamp'].max()}"
                        ]
                    })
                    summary.to_excel(writer, sheet_name='Summary', index=False)

            else:
                # Export to CSV
                self.current_data.to_csv(filename, index=False)

            messagebox.showinfo("Export Complete", f"Data exported to:\n{filename}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")

    def _sort_column(self, col):
        """Sort treeview by column."""
        # Get data
        data = [(self.results_tree.set(child, col), child)
                for child in self.results_tree.get_children('')]

        # Sort data
        data.sort(reverse=False)

        # Rearrange items
        for index, (_, child) in enumerate(data):
            self.results_tree.move(child, '', index)

    def _view_details(self, event):
        """View detailed information for selected item."""
        selection = self.results_tree.selection()
        if not selection:
            return

        # Get selected item data
        item = self.results_tree.item(selection[0])
        values = item['values']

        if not values:
            return

        # Create details dialog
        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title("Analysis Details")
        dialog.geometry("600x400")

        # Create text widget with scrollbar
        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        text = tk.Text(text_frame, wrap='word', width=70, height=20)
        scroll = ttk.Scrollbar(text_frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)

        text.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

        # Add details
        details = f"""Analysis Details
{'=' * 50}
Date: {values[0]}
Model: {values[1]}
Serial: {values[2]}
System: {values[3]}
Status: {values[4]}

Metrics:
- Sigma Gradient: {values[5]}
- Sigma Pass: {values[6]}
- Linearity Pass: {values[7]}
- Risk Category: {values[8]}
- Processing Time: {values[9]}
"""

        text.insert('1.0', details)
        text.configure(state='disabled')

        # Close button
        ttk.Button(
            dialog,
            text="Close",
            command=dialog.destroy
        ).pack(pady=(0, 10))

    def _handle_selection(self, event):
        """Handle row selection in the tree."""
        try:
            selected_items = self.results_tree.selection()
            self.selected_ids = []
            
            for item in selected_items:
                values = self.results_tree.item(item, 'values')
                if values:
                    self.selected_ids.append(int(values[0]))  # ID is first column
                    
        except Exception as e:
            self.logger.error(f"Selection handling error: {e}")

    def export_results(self):
        """Export historical data to Excel."""
        if self.current_data is None or len(self.current_data) == 0:
            messagebox.showwarning("Export", "No data available to export")
            return
            
        try:
            # Get save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="Export Historical Data"
            )
            
            if not filename:
                return
                
            # Export to Excel
            self.current_data.to_excel(filename, index=False)
            messagebox.showinfo("Export", f"Data exported successfully to:\n{filename}")
            self.logger.info(f"Exported historical data to {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")
            self.logger.error(f"Export failed: {e}")

    def _update_analytics_status(self, status: str, color: str):
        """Update analytics status indicator."""
        self.analytics_status_label.configure(text=f"Analytics Status: {status}")
        
        color_map = {
            "green": "#00ff00",
            "orange": "#ffa500",
            "red": "#ff0000",
            "gray": "#808080"
        }
        
        self.analytics_indicator.configure(text_color=color_map.get(color, "#808080"))

    def _update_dashboard_metrics(self, data: List[Dict[str, Any]]):
        """Update analytics dashboard with current data metrics."""
        if not data:
            # Show empty state guidance
            self.total_records_card.update_value("0")
            self.pass_rate_card.update_value("--")
            self.trend_direction_card.update_value("--")
            self.prediction_accuracy_card.update_value("--")
            self.sigma_correlation_card.update_value("--")
            self.linearity_stability_card.update_value("--")
            self.quality_score_card.update_value("--")
            self.anomaly_count_card.update_value("--")
            
            # Update results tree with empty state message
            if hasattr(self, 'results_tree'):
                for item in self.results_tree.get_children():
                    self.results_tree.delete(item)
                    
                # Insert empty state guidance
                empty_message = self.results_tree.insert('', 'end', values=(
                    'No Data Available',
                    'To view historical data, please:',
                    '1. Go to Analysis tab',
                    '2. Process Excel files',
                    '3. Return here to see results',
                    '', '', '', ''
                ))
                
            return
            
        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(data)
            
            # Total records
            total_records = len(df)
            self.total_records_card.update_value(str(total_records))
            
            # Pass rate calculation
            if 'overall_status' in df.columns:
                pass_count = len(df[df['overall_status'] == 'PASS'])
                pass_rate = (pass_count / total_records) * 100 if total_records > 0 else 0
                self.pass_rate_card.update_value(f"{pass_rate:.1f}%")
                
                # Set color based on pass rate
                if pass_rate >= 95:
                    self.pass_rate_card.set_color_scheme('success')
                elif pass_rate >= 85:
                    self.pass_rate_card.set_color_scheme('warning')
                else:
                    self.pass_rate_card.set_color_scheme('danger')
            
            # Trend direction (simplified)
            if 'timestamp' in df.columns and len(df) > 1:
                # Sort by timestamp and check recent trend
                df_sorted = df.sort_values('timestamp')
                recent_data = df_sorted.tail(min(10, len(df_sorted)))
                
                if 'sigma_gradient' in df.columns:
                    sigma_trend = np.polyfit(range(len(recent_data)), recent_data['sigma_gradient'], 1)[0]
                    trend_direction = "Improving" if sigma_trend < 0 else "Declining" if sigma_trend > 0 else "Stable"
                    self.trend_direction_card.update_value(trend_direction)
                    
                    # Set color based on trend
                    color_scheme = 'success' if trend_direction == 'Improving' else 'danger' if trend_direction == 'Declining' else 'neutral'
                    self.trend_direction_card.set_color_scheme(color_scheme)
            
            # Sigma correlation (if multiple parameters available)
            if 'sigma_gradient' in df.columns and 'linearity_error' in df.columns:
                correlation = df['sigma_gradient'].corr(df['linearity_error'])
                self.sigma_correlation_card.update_value(f"{correlation:.3f}")
                
                # Set color based on correlation strength
                abs_corr = abs(correlation)
                if abs_corr > 0.7:
                    self.sigma_correlation_card.set_color_scheme('danger')
                elif abs_corr > 0.3:
                    self.sigma_correlation_card.set_color_scheme('warning')
                else:
                    self.sigma_correlation_card.set_color_scheme('success')
            
            # Linearity stability (coefficient of variation)
            if 'linearity_error' in df.columns:
                cv = df['linearity_error'].std() / df['linearity_error'].mean() if df['linearity_error'].mean() != 0 else 0
                stability = max(0, (1 - cv) * 100)  # Convert to stability percentage
                self.linearity_stability_card.update_value(f"{stability:.1f}%")
                
                # Set color based on stability
                if stability >= 80:
                    self.linearity_stability_card.set_color_scheme('success')
                elif stability >= 60:
                    self.linearity_stability_card.set_color_scheme('warning')
                else:
                    self.linearity_stability_card.set_color_scheme('danger')
            
            # Quality score (composite metric)
            quality_factors = []
            if pass_rate:
                quality_factors.append(pass_rate / 100)
            if 'linearity_stability_card' in locals() and stability:
                quality_factors.append(stability / 100)
                
            if quality_factors:
                quality_score = np.mean(quality_factors) * 100
                self.quality_score_card.update_value(f"{quality_score:.1f}%")
                
                if quality_score >= 90:
                    self.quality_score_card.set_color_scheme('success')
                elif quality_score >= 70:
                    self.quality_score_card.set_color_scheme('warning')
                else:
                    self.quality_score_card.set_color_scheme('danger')
            
        except Exception as e:
            logger.error(f"Error updating dashboard metrics: {e}")

    def _run_trend_analysis(self):
        """Run comprehensive trend analysis."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data for analysis")
            return
            
        self._update_analytics_status("Running Trend Analysis...", "orange")
        self.trend_analysis_btn.configure(state="disabled", text="Analyzing...")
        
        def analyze():
            try:
                trend_data = self._calculate_trend_analysis(self.current_data)
                self.trend_analysis_data = trend_data
                
                # Update UI
                self.after(0, lambda: self._display_trend_analysis(trend_data))
                
            except Exception as e:
                logger.error(f"Trend analysis failed: {e}")
                self.after(0, lambda: messagebox.showerror(
                    "Analysis Error", f"Trend analysis failed:\n{str(e)}"
                ))
            finally:
                self.after(0, lambda: self.trend_analysis_btn.configure(
                    state="normal", text="📈 Trend Analysis"
                ))
                self.after(0, lambda: self._update_analytics_status("Ready", "green"))

        threading.Thread(target=analyze, daemon=True).start()

    def _calculate_trend_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive trend analysis from historical data."""
        trend_data = {
            'time_series': {},
            'trends': {},
            'seasonality': {},
            'forecasts': {},
            'change_points': {}
        }
        
        try:
            df = pd.DataFrame(data)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                # Analyze sigma gradient trends
                if 'sigma_gradient' in df.columns:
                    sigma_trends = self._analyze_parameter_trend(df, 'sigma_gradient', 'timestamp')
                    trend_data['trends']['sigma_gradient'] = sigma_trends
                
                # Analyze linearity error trends
                if 'linearity_error' in df.columns:
                    linearity_trends = self._analyze_parameter_trend(df, 'linearity_error', 'timestamp')
                    trend_data['trends']['linearity_error'] = linearity_trends
                
                # Analyze pass rate trends over time
                if 'overall_status' in df.columns:
                    pass_rate_trends = self._analyze_pass_rate_trends(df)
                    trend_data['trends']['pass_rate'] = pass_rate_trends
                
                # Detect change points
                trend_data['change_points'] = self._detect_change_points(df)
                
                # Generate forecasts
                trend_data['forecasts'] = self._generate_forecasts(df)
                
        except Exception as e:
            logger.error(f"Error in trend analysis calculation: {e}")
            
        return trend_data

    def _analyze_parameter_trend(self, df: pd.DataFrame, parameter: str, time_col: str) -> Dict[str, Any]:
        """Analyze trend for a specific parameter."""
        analysis = {
            'slope': 0,
            'r_squared': 0,
            'trend_direction': 'stable',
            'significance': 'low',
            'volatility': 0
        }
        
        try:
            if parameter not in df.columns or df[parameter].isna().all():
                return analysis
                
            # Remove NaN values
            clean_df = df.dropna(subset=[parameter, time_col])
            if len(clean_df) < 3:
                return analysis
            
            # Convert timestamps to numeric for regression
            x_numeric = (clean_df[time_col] - clean_df[time_col].min()).dt.total_seconds()
            y = clean_df[parameter]
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y)
            
            analysis['slope'] = slope
            analysis['r_squared'] = r_value ** 2
            analysis['p_value'] = p_value
            
            # Determine trend direction
            if abs(slope) < std_err:
                analysis['trend_direction'] = 'stable'
            elif slope > 0:
                analysis['trend_direction'] = 'increasing'
            else:
                analysis['trend_direction'] = 'decreasing'
            
            # Determine significance
            if p_value < 0.01:
                analysis['significance'] = 'high'
            elif p_value < 0.05:
                analysis['significance'] = 'medium'
            else:
                analysis['significance'] = 'low'
            
            # Calculate volatility (coefficient of variation)
            analysis['volatility'] = y.std() / y.mean() if y.mean() != 0 else 0
            
        except Exception as e:
            logger.error(f"Error analyzing parameter trend for {parameter}: {e}")
            
        return analysis

    def _run_correlation_analysis(self):
        """Run correlation analysis between parameters."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data for analysis")
            return
            
        self._update_analytics_status("Running Correlation Analysis...", "orange")
        self.correlation_analysis_btn.configure(state="disabled", text="Analyzing...")
        
        def analyze():
            try:
                correlation_data = self._calculate_correlation_matrix(self.current_data)
                self.correlation_matrix = correlation_data
                
                # Update UI
                self.after(0, lambda: self._display_correlation_analysis(correlation_data))
                
            except Exception as e:
                logger.error(f"Correlation analysis failed: {e}")
                self.after(0, lambda: messagebox.showerror(
                    "Analysis Error", f"Correlation analysis failed:\n{str(e)}"
                ))
            finally:
                self.after(0, lambda: self.correlation_analysis_btn.configure(
                    state="normal", text="🔗 Correlation Analysis"
                ))
                self.after(0, lambda: self._update_analytics_status("Ready", "green"))

        threading.Thread(target=analyze, daemon=True).start()

    def _generate_statistical_summary(self):
        """Generate comprehensive statistical summary."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data for analysis")
            return
            
        self._update_analytics_status("Generating Statistical Summary...", "orange")
        self.statistical_summary_btn.configure(state="disabled", text="Generating...")
        
        def generate():
            try:
                summary_data = self._calculate_statistical_summary(self.current_data)
                self.after(0, lambda: self._display_statistical_summary(summary_data))
                
            except Exception as e:
                logger.error(f"Statistical summary generation failed: {e}")
                self.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to generate statistical summary:\n{str(e)}"
                ))
            finally:
                self.after(0, lambda: self.statistical_summary_btn.configure(
                    state="normal", text="📊 Statistical Summary"
                ))
                self.after(0, lambda: self._update_analytics_status("Ready", "green"))

        threading.Thread(target=generate, daemon=True).start()

    def _calculate_statistical_summary(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistical summary of historical data."""
        summary = {
            'total_records': 0,
            'pass_rate': 0,
            'parameter_stats': {},
            'model_breakdown': {},
            'temporal_analysis': {}
        }
        
        try:
            df = pd.DataFrame(data)
            summary['total_records'] = len(df)
            
            if 'status' in df.columns:
                pass_count = len(df[df['status'] == 'Pass'])
                summary['pass_rate'] = (pass_count / len(df)) * 100 if len(df) > 0 else 0
            
            # Analyze numerical parameters
            numerical_params = ['sigma_gradient', 'linearity_error', 'resistance_change_percent']
            for param in numerical_params:
                if param in df.columns:
                    values = df[param].dropna()
                    if len(values) > 0:
                        summary['parameter_stats'][param] = {
                            'mean': values.mean(),
                            'median': values.median(),
                            'std': values.std(),
                            'min': values.min(),
                            'max': values.max(),
                            'count': len(values)
                        }
            
            # Model breakdown
            if 'model' in df.columns:
                model_counts = df['model'].value_counts()
                summary['model_breakdown'] = model_counts.to_dict()
            
            # Temporal analysis
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                date_range = df['timestamp'].max() - df['timestamp'].min()
                summary['temporal_analysis'] = {
                    'date_range_days': date_range.days,
                    'earliest_record': df['timestamp'].min().isoformat(),
                    'latest_record': df['timestamp'].max().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error calculating statistical summary: {e}")
            summary['error'] = str(e)
            
        return summary

    def _display_statistical_summary(self, summary_data: Dict[str, Any]):
        """Display statistical summary in the UI."""
        try:
            self.stats_display.configure(state='normal')
            self.stats_display.delete('1.0', ctk.END)
            
            content = "COMPREHENSIVE STATISTICAL SUMMARY\n"
            content += "=" * 60 + "\n\n"
            
            # Basic statistics
            content += f"Total Records: {summary_data.get('total_records', 0)}\n"
            content += f"Pass Rate: {summary_data.get('pass_rate', 0):.2f}%\n\n"
            
            # Parameter statistics
            param_stats = summary_data.get('parameter_stats', {})
            if param_stats:
                content += "PARAMETER STATISTICS:\n"
                for param, stats in param_stats.items():
                    content += f"\n{param.replace('_', ' ').title()}:\n"
                    content += f"  Mean: {stats['mean']:.6f}\n"
                    content += f"  Median: {stats['median']:.6f}\n"
                    content += f"  Std Dev: {stats['std']:.6f}\n"
                    content += f"  Range: {stats['min']:.6f} - {stats['max']:.6f}\n"
                    content += f"  Sample Count: {stats['count']}\n"
                content += "\n"
            
            # Model breakdown
            model_breakdown = summary_data.get('model_breakdown', {})
            if model_breakdown:
                content += "MODEL BREAKDOWN:\n"
                for model, count in sorted(model_breakdown.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / summary_data.get('total_records', 1)) * 100
                    content += f"  {model}: {count} records ({percentage:.1f}%)\n"
                content += "\n"
            
            # Temporal analysis
            temporal = summary_data.get('temporal_analysis', {})
            if temporal:
                content += "TEMPORAL ANALYSIS:\n"
                content += f"  Date Range: {temporal.get('date_range_days', 0)} days\n"
                content += f"  Earliest: {temporal.get('earliest_record', 'Unknown')}\n"
                content += f"  Latest: {temporal.get('latest_record', 'Unknown')}\n"
            
            self.stats_display.insert('1.0', content)
            self.stats_display.configure(state='disabled')
            
        except Exception as e:
            logger.error(f"Error displaying statistical summary: {e}")

    def _calculate_correlation_matrix(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate correlation matrix for numerical parameters."""
        try:
            df = pd.DataFrame(data)
            
            # Select numerical columns for correlation
            numerical_cols = ['sigma_gradient', 'linearity_error', 'resistance_change_percent', 
                            'unit_length', 'travel_length']
            
            # Filter to available columns
            available_cols = [col for col in numerical_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return {'matrix': None, 'error': 'Insufficient numerical columns for correlation'}
            
            correlation_df = df[available_cols].corr()
            
            # Find strong correlations
            strong_correlations = []
            for i, col1 in enumerate(available_cols):
                for j, col2 in enumerate(available_cols):
                    if i < j:  # Avoid duplicates
                        corr_val = correlation_df.loc[col1, col2]
                        if abs(corr_val) > 0.5:  # Threshold for "strong" correlation
                            strong_correlations.append({
                                'param1': col1,
                                'param2': col2,
                                'correlation': corr_val,
                                'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate'
                            })
            
            return {
                'matrix': correlation_df,
                'strong_correlations': strong_correlations,
                'parameters': available_cols
            }
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return {'matrix': None, 'error': str(e)}

    def _run_predictive_analysis(self):
        """Run predictive analysis to forecast future performance."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data for analysis")
            return
            
        self._update_analytics_status("Running Predictive Analysis...", "orange")
        self.predictive_analysis_btn.configure(state="disabled", text="Predicting...")
        
        def analyze():
            try:
                prediction_data = self._build_predictive_models(self.current_data)
                
                # Update UI
                self.after(0, lambda: self._display_predictive_analysis(prediction_data))
                
            except Exception as e:
                logger.error(f"Predictive analysis failed: {e}")
                self.after(0, lambda: messagebox.showerror(
                    "Analysis Error", f"Predictive analysis failed:\n{str(e)}"
                ))
            finally:
                self.after(0, lambda: self.predictive_analysis_btn.configure(
                    state="normal", text="🔮 Predictive Analysis"
                ))
                self.after(0, lambda: self._update_analytics_status("Ready", "green"))

        threading.Thread(target=analyze, daemon=True).start()

    def _build_predictive_models(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build simple predictive models from historical data."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            df = pd.DataFrame(data)
            
            # Prepare features and targets
            feature_cols = ['sigma_gradient', 'linearity_error', 'unit_length', 'travel_length']
            available_features = [col for col in feature_cols if col in df.columns]
            
            if len(available_features) < 2:
                return {'error': 'Insufficient features for predictive modeling'}
            
            # Remove rows with missing values
            clean_df = df.dropna(subset=available_features)
            
            if len(clean_df) < 10:
                return {'error': 'Insufficient data points for reliable prediction'}
            
            models = {}
            
            # Predict sigma gradient
            if 'sigma_gradient' in clean_df.columns and len(available_features) > 1:
                target_features = [col for col in available_features if col != 'sigma_gradient']
                
                X = clean_df[target_features]
                y = clean_df['sigma_gradient']
                
                if len(X) > 5:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    
                    models['sigma_gradient'] = {
                        'model': model,
                        'features': target_features,
                        'mse': mean_squared_error(y_test, y_pred),
                        'r2_score': r2_score(y_test, y_pred),
                        'actual': y_test.tolist(),
                        'predicted': y_pred.tolist(),
                        'feature_importance': dict(zip(target_features, model.feature_importances_))
                    }
            
            return models
            
        except ImportError:
            return {'error': 'Scikit-learn not available for predictive modeling'}
        except Exception as e:
            logger.error(f"Error building predictive models: {e}")
            return {'error': str(e)}

    def _detect_anomalies(self):
        """Detect anomalies in the historical data."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data for analysis")
            return
            
        self._update_analytics_status("Detecting Anomalies...", "orange")
        self.anomaly_detection_btn.configure(state="disabled", text="Detecting...")
        
        def detect():
            try:
                anomaly_data = self._find_anomalies(self.current_data)
                
                # Update dashboard
                anomaly_count = len(anomaly_data.get('anomalies', []))
                self.anomaly_count_card.update_value(str(anomaly_count))
                
                if anomaly_count == 0:
                    self.anomaly_count_card.set_color_scheme('success')
                elif anomaly_count <= 5:
                    self.anomaly_count_card.set_color_scheme('warning')
                else:
                    self.anomaly_count_card.set_color_scheme('danger')
                
                # Update UI
                self.after(0, lambda: self._display_anomaly_results(anomaly_data))
                
            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}")
                self.after(0, lambda: messagebox.showerror(
                    "Analysis Error", f"Anomaly detection failed:\n{str(e)}"
                ))
            finally:
                self.after(0, lambda: self.anomaly_detection_btn.configure(
                    state="normal", text="🚨 Detect Anomalies"
                ))
                self.after(0, lambda: self._update_analytics_status("Ready", "green"))

        threading.Thread(target=detect, daemon=True).start()

    def _find_anomalies(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find anomalies using statistical methods."""
        try:
            df = pd.DataFrame(data)
            anomalies = []
            
            # Parameters to check for anomalies
            numerical_params = ['sigma_gradient', 'linearity_error', 'resistance_change_percent']
            
            for param in numerical_params:
                if param in df.columns:
                    values = df[param].dropna()
                    
                    if len(values) > 3:
                        # Use IQR method for anomaly detection
                        Q1 = values.quantile(0.25)
                        Q3 = values.quantile(0.75)
                        IQR = Q3 - Q1
                        
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Find outliers
                        outliers = df[(df[param] < lower_bound) | (df[param] > upper_bound)]
                        
                        for idx, row in outliers.iterrows():
                            anomalies.append({
                                'index': idx,
                                'parameter': param,
                                'value': row[param],
                                'expected_range': f"{lower_bound:.4f} - {upper_bound:.4f}",
                                'severity': 'high' if (row[param] < Q1 - 3*IQR or row[param] > Q3 + 3*IQR) else 'medium',
                                'timestamp': row.get('timestamp', 'Unknown'),
                                'model': row.get('model', 'Unknown'),
                                'serial': row.get('serial', 'Unknown')
                            })
            
            # Statistical summary
            summary = {
                'total_records': len(df),
                'anomalies_found': len(anomalies),
                'anomaly_rate': (len(anomalies) / len(df)) * 100 if len(df) > 0 else 0,
                'parameters_checked': numerical_params,
                'severity_breakdown': {
                    'high': len([a for a in anomalies if a['severity'] == 'high']),
                    'medium': len([a for a in anomalies if a['severity'] == 'medium'])
                }
            }
            
            return {
                'anomalies': anomalies,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error finding anomalies: {e}")
            return {'anomalies': [], 'summary': {}, 'error': str(e)}

    def _display_trend_analysis(self, trend_data: Dict[str, Any]):
        """Display trend analysis results."""
        try:
            # Update trend chart
            self._update_trend_analysis_chart(trend_data)
            
        except Exception as e:
            logger.error(f"Error displaying trend analysis: {e}")

    def _display_correlation_analysis(self, correlation_data: Dict[str, Any]):
        """Display correlation analysis results."""
        try:
            # Update correlation heatmap
            self._update_correlation_heatmap(correlation_data)
            
        except Exception as e:
            logger.error(f"Error displaying correlation analysis: {e}")

    def _display_predictive_analysis(self, prediction_data: Dict[str, Any]):
        """Display predictive analysis results."""
        try:
            if 'error' in prediction_data:
                # Show error message
                self.prediction_chart.clear_chart()
                return
                
            # Update prediction chart
            self._update_prediction_chart(prediction_data)
            
        except Exception as e:
            logger.error(f"Error displaying predictive analysis: {e}")

    def _display_anomaly_results(self, anomaly_data: Dict[str, Any]):
        """Display anomaly detection results."""
        try:
            self.anomaly_display.configure(state='normal')
            self.anomaly_display.delete('1.0', ctk.END)
            
            content = "ANOMALY DETECTION RESULTS\n"
            content += "=" * 50 + "\n\n"
            
            summary = anomaly_data.get('summary', {})
            anomalies = anomaly_data.get('anomalies', [])
            
            content += f"Total Records Analyzed: {summary.get('total_records', 0)}\n"
            content += f"Anomalies Found: {summary.get('anomalies_found', 0)}\n"
            content += f"Anomaly Rate: {summary.get('anomaly_rate', 0):.2f}%\n\n"
            
            # Severity breakdown
            severity = summary.get('severity_breakdown', {})
            content += "SEVERITY BREAKDOWN:\n"
            content += f"  High: {severity.get('high', 0)}\n"
            content += f"  Medium: {severity.get('medium', 0)}\n\n"
            
            # List anomalies
            if anomalies:
                content += "DETECTED ANOMALIES:\n"
                for i, anomaly in enumerate(anomalies[:20], 1):  # Show first 20
                    content += f"\n{i}. {anomaly['parameter'].upper()} ANOMALY\n"
                    content += f"   Value: {anomaly['value']:.6f}\n"
                    content += f"   Expected Range: {anomaly['expected_range']}\n"
                    content += f"   Severity: {anomaly['severity']}\n"
                    content += f"   Model/Serial: {anomaly['model']}/{anomaly['serial']}\n"
                    content += f"   Timestamp: {anomaly['timestamp']}\n"
                
                if len(anomalies) > 20:
                    content += f"\n... and {len(anomalies) - 20} more anomalies\n"
            else:
                content += "No anomalies detected. All data points are within expected ranges.\n"
            
            self.anomaly_display.insert('1.0', content)
            self.anomaly_display.configure(state='disabled')
            
        except Exception as e:
            logger.error(f"Error displaying anomaly results: {e}")