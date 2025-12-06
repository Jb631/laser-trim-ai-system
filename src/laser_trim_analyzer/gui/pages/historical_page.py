"""
Historical Data Page for Laser Trim Analyzer

Provides interface for querying and analyzing historical QA data
with advanced analytics, charts, and export functionality.

REFACTORED: This module now uses mixins from the historical/ package.
The implementation has been split into smaller, maintainable modules:
- historical/analytics_mixin.py: Trend analysis, correlation, prediction, anomalies
- historical/spc_mixin.py: Control charts, capability, Pareto, drift detection
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
import traceback

# Get logger first
logger = logging.getLogger(__name__)

# Optional analytics libraries
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not available - some analytics features will be disabled")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not available - ML features will be disabled")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from laser_trim_analyzer.core.models import AnalysisResult, FileMetadata, AnalysisStatus
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.core.config import get_config

# Import UnifiedProcessor for ML drift detection (Phase 3)
try:
    from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
    HAS_UNIFIED_PROCESSOR = True
except ImportError:
    HAS_UNIFIED_PROCESSOR = False
    logger.warning("UnifiedProcessor not available - using formula-based drift detection")
from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
from laser_trim_analyzer.gui.widgets.metric_card_ctk import MetricCard
from laser_trim_analyzer.utils.date_utils import safe_datetime_convert
from laser_trim_analyzer.gui.widgets import add_mousewheel_support

# Import mixins for modular functionality
from laser_trim_analyzer.gui.pages.historical.analytics_mixin import AnalyticsMixin
from laser_trim_analyzer.gui.pages.historical.spc_mixin import SPCMixin


class HistoricalPage(SPCMixin, AnalyticsMixin, ctk.CTkFrame):
    """QA-focused historical data analysis page with manufacturing insights.

    Combines functionality through mixins:
    - SPCMixin: SPC charts, capability, Pareto, drift detection
    - AnalyticsMixin: Trend analysis, correlation, prediction, anomalies
    """

    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Add BasePage-like functionality
        self.is_visible = False
        self.needs_refresh = True
        self._stop_requested = False
        
        # Initialize data storage
        self.current_data = None  # This will store raw database results
        self.current_data_df = None  # This will store DataFrame version
        self._analytics_data = []  # This will store prepared analytics data
        self.analytics_results = {}
        self.trend_analysis_data = {}
        self.correlation_matrix = None
        self._pending_filters = {}  # Store filters to apply after UI is created
        
        # QA-specific data storage
        self.risk_analysis_data = {}
        self.batch_comparison_data = {}
        self.failure_mode_data = {}
        self.trim_effectiveness_data = {}
        self.selected_analysis_id = None  # For detailed view
        
        # Thread safety
        self._analytics_lock = threading.Lock()
        
        # Create the page
        self._create_page()

    def _create_page(self):
        """Create QA-focused historical analysis page."""
        # Main scrollable container
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create sections in QA priority order
        self._create_header()
        self._create_query_section_ctk()
        self._create_qa_metrics_dashboard()  # Enhanced QA metrics
        self._create_risk_dashboard()  # New risk analysis section
        self._create_results_section_ctk()  # Enhanced results display
        self._create_manufacturing_insights()  # Replace basic charts
        self._create_process_control_section()  # Replace advanced analytics

    def _create_header(self):
        """Create QA-focused header section."""
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.pack(fill='x', pady=(0, 20))

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="QA Historical Analysis & Process Control",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=15)

        # Analytics status indicator
        self.analytics_status_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
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

    def _create_qa_metrics_dashboard(self):
        """Create company-wide metrics dashboard showing overall manufacturing performance."""
        self.dashboard_frame = ctk.CTkFrame(self.main_container)
        self.dashboard_frame.pack(fill='x', pady=(0, 20))

        self.dashboard_label = ctk.CTkLabel(
            self.dashboard_frame,
            text="Company-Wide Performance Metrics:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.dashboard_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Dashboard metrics container
        self.metrics_container = ctk.CTkFrame(self.dashboard_frame, fg_color="transparent")
        self.metrics_container.pack(fill='x', padx=15, pady=(0, 15))

        # Row 1 - Overall production and efficiency metrics
        metrics_row1 = ctk.CTkFrame(self.metrics_container, fg_color="transparent")
        metrics_row1.pack(fill='x', padx=10, pady=(10, 5))

        self.total_production_card = MetricCard(
            metrics_row1,
            title="Total Production",
            value="--",
            color_scheme="info"
        )
        self.total_production_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

        self.overall_yield_card = MetricCard(
            metrics_row1,
            title="Company Yield",
            value="--",
            color_scheme="success"
        )
        self.overall_yield_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

        self.productivity_card = MetricCard(
            metrics_row1,
            title="Daily Throughput",
            value="--",
            unit="units/day",
            color_scheme="warning"
        )
        self.productivity_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

        self.quality_index_card = MetricCard(
            metrics_row1,
            title="Quality Index",
            value="--",
            color_scheme="info"
        )
        self.quality_index_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

        # Row 2 - Process health and efficiency indicators
        metrics_row2 = ctk.CTkFrame(self.metrics_container, fg_color="transparent")
        metrics_row2.pack(fill='x', padx=10, pady=(5, 10))

        self.active_models_card = MetricCard(
            metrics_row2,
            title="Active Models",
            value="--",
            color_scheme="info"
        )
        self.active_models_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

        self.process_stability_card = MetricCard(
            metrics_row2,
            title="Process Stability",
            value="--",
            color_scheme="warning"
        )
        self.process_stability_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

        self.avg_efficiency_card = MetricCard(
            metrics_row2,
            title="Trim Efficiency",
            value="--",
            color_scheme="info"
        )
        self.avg_efficiency_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

        self.defect_trend_card = MetricCard(
            metrics_row2,
            title="Defect Trend",
            value="--",
            color_scheme="error"
        )
        self.defect_trend_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)
        
        # Row 3 - Add drift alert card for process control monitoring
        metrics_row3 = ctk.CTkFrame(self.metrics_container, fg_color="transparent")
        metrics_row3.pack(fill='x', padx=10, pady=(5, 10))
        
        self.drift_alert_card = MetricCard(
            metrics_row3,
            title="Drift Alerts",
            value="0",
            color_scheme="warning"
        )
        self.drift_alert_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

    def _create_risk_dashboard(self):
        """Create company-wide risk analysis dashboard."""
        self.risk_frame = ctk.CTkFrame(self.main_container)
        self.risk_frame.pack(fill='x', pady=(0, 20))

        self.risk_label = ctk.CTkLabel(
            self.risk_frame,
            text="Enterprise Risk Analysis:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.risk_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Risk visualization container
        self.risk_container = ctk.CTkFrame(self.risk_frame)
        self.risk_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Create tabs for different risk views
        self.risk_tabview = ctk.CTkTabview(self.risk_container)
        self.risk_tabview.pack(fill='both', expand=True, padx=10, pady=10)

        # Add company-wide risk analysis tabs
        self.risk_tabview.add("Risk Overview")
        self.risk_tabview.add("Model Comparison")
        self.risk_tabview.add("Production Risk Trends")

        # Risk overview chart - shows company-wide risk distribution
        self.risk_overview_chart = ChartWidget(
            self.risk_tabview.tab("Risk Overview"),
            chart_type='pie',
            title="Company-Wide Risk Distribution",
            figsize=(7, 7)  # Square aspect for pie chart
        )
        self.risk_overview_chart.pack(fill='both', expand=True, padx=5, pady=5)

        # Model comparison chart - compare risk across all models
        self.model_comparison_frame = ctk.CTkFrame(self.risk_tabview.tab("Model Comparison"))
        self.model_comparison_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Model comparison chart
        self.model_risk_chart = ChartWidget(
            self.model_comparison_frame,
            chart_type='bar',
            title="Risk Distribution by Model - Pareto Analysis",
            figsize=(12, 6)  # Wider for Pareto chart
        )
        self.model_risk_chart.pack(fill='both', expand=True)

        # Production risk trends - timeline view
        self.risk_trends_chart = ChartWidget(
            self.risk_tabview.tab("Production Risk Trends"),
            chart_type='line',
            title="Risk Trends with Moving Average",
            figsize=(12, 6)  # Standard trend chart size
        )
        self.risk_trends_chart.pack(fill='both', expand=True, padx=5, pady=5)

    def _create_process_control_section(self):
        """Create statistical process control section for manufacturing quality."""
        self.spc_frame = ctk.CTkFrame(self.main_container)
        self.spc_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.spc_label = ctk.CTkLabel(
            self.spc_frame,
            text="Statistical Process Control:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.spc_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Single SPC control button to generate all analyses
        spc_controls = ctk.CTkFrame(self.spc_frame, fg_color="transparent")
        spc_controls.pack(fill='x', padx=15, pady=(0, 10))

        self.generate_spc_btn = ctk.CTkButton(
            spc_controls,
            text="📊 Generate All SPC Analyses",
            command=self._generate_all_spc_analyses,
            width=200,
            height=40
        )
        self.generate_spc_btn.pack(side='left', padx=(10, 10), pady=10)
        
        # Info label
        spc_info_label = ctk.CTkLabel(
            spc_controls,
            text="Click to generate all statistical analyses, then use tabs below to view results",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        spc_info_label.pack(side='left', padx=(10, 10), pady=10)

        # SPC results tabview
        self.spc_tabview = ctk.CTkTabview(self.spc_frame)
        self.spc_tabview.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Add SPC tabs
        self.spc_tabview.add("Control Charts")
        self.spc_tabview.add("Process Capability")
        self.spc_tabview.add("Pareto Analysis")
        self.spc_tabview.add("Drift Detection")
        self.spc_tabview.add("Failure Modes")

        # Control charts tab
        self.control_chart = ChartWidget(
            self.spc_tabview.tab("Control Charts"),
            chart_type='line',
            title="X-bar & R Statistical Process Control Chart",
            figsize=(14, 7)  # Larger for SPC chart with dual axes
        )
        self.control_chart.pack(fill='both', expand=True, padx=5, pady=5)

        # Process capability tab
        self.capability_display = ctk.CTkTextbox(
            self.spc_tabview.tab("Process Capability"),
            height=400,
            state="disabled"
        )
        self.capability_display.pack(fill='both', expand=True, padx=5, pady=5)

        # Pareto analysis tab
        self.pareto_chart = ChartWidget(
            self.spc_tabview.tab("Pareto Analysis"),
            chart_type='bar',
            title="Failure Mode Pareto Analysis (80/20 Rule)",
            figsize=(12, 7)  # Larger for dual-axis Pareto
        )
        self.pareto_chart.pack(fill='both', expand=True, padx=5, pady=5)

        # Drift detection tab
        self.drift_chart = ChartWidget(
            self.spc_tabview.tab("Drift Detection"),
            chart_type='line',
            title="Process Drift Detection with EWMA",
            figsize=(12, 6)  # Standard for drift analysis
        )
        self.drift_chart.pack(fill='both', expand=True, padx=5, pady=5)

        # Failure modes tab
        self.failure_mode_display = ctk.CTkTextbox(
            self.spc_tabview.tab("Failure Modes"),
            height=400,
            state="disabled"
        )
        self.failure_mode_display.pack(fill='both', expand=True, padx=5, pady=5)

    def _create_query_section_ctk(self):
        """Create query filters section for company-wide analysis."""
        self.query_frame = ctk.CTkFrame(self.main_container)
        self.query_frame.pack(fill='x', pady=(0, 20))

        self.query_label = ctk.CTkLabel(
            self.query_frame,
            text="Company-Wide Analysis Filters:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.query_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Filters container
        self.filters_container = ctk.CTkFrame(self.query_frame, fg_color="transparent")
        self.filters_container.pack(fill='x', padx=15, pady=(0, 15))

        # First row of filters
        filter_row1 = ctk.CTkFrame(self.filters_container, fg_color="transparent")
        filter_row1.pack(fill='x', padx=10, pady=(10, 5))

        # Date range (primary filter for company-wide view)
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
            width=150,
            height=30
        )
        self.date_combo.pack(side='left', padx=(0, 20), pady=10)

        # Add mousewheel support to date dropdown
        add_mousewheel_support(self.date_combo)

        # Department/Production Line filter (optional - for larger companies)
        dept_label = ctk.CTkLabel(filter_row1, text="Department:")
        dept_label.pack(side='left', padx=10, pady=10)

        self.dept_var = tk.StringVar(value="All")
        self.dept_combo = ctk.CTkComboBox(
            filter_row1,
            variable=self.dept_var,
            values=["All", "Production A", "Production B", "QA Testing"],
            width=120,
            height=30
        )
        self.dept_combo.pack(side='left', padx=(0, 20), pady=10)

        # Add mousewheel support to department dropdown
        add_mousewheel_support(self.dept_combo)

        # Shift filter (for manufacturing insights)
        shift_label = ctk.CTkLabel(filter_row1, text="Shift:")
        shift_label.pack(side='left', padx=10, pady=10)

        self.shift_var = tk.StringVar(value="All")
        self.shift_combo = ctk.CTkComboBox(
            filter_row1,
            variable=self.shift_var,
            values=["All", "Day", "Evening", "Night"],
            width=100,
            height=30
        )
        self.shift_combo.pack(side='left', padx=(0, 10), pady=10)

        # Add mousewheel support to shift dropdown
        add_mousewheel_support(self.shift_combo)

        # Second row of filters
        filter_row2 = ctk.CTkFrame(self.filters_container, fg_color="transparent")
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

        # Add mousewheel support to status dropdown
        add_mousewheel_support(self.status_combo)

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

        # Add mousewheel support to risk dropdown
        add_mousewheel_support(self.risk_combo)

        # Limit filter
        limit_label = ctk.CTkLabel(filter_row2, text="Limit:")
        limit_label.pack(side='left', padx=10, pady=10)

        self.limit_var = tk.StringVar(value="100")
        self.limit_combo = ctk.CTkComboBox(
            filter_row2,
            variable=self.limit_var,
            values=["50", "100", "250", "500", "1000", "All"],
            width=80,
            height=30
        )
        self.limit_combo.pack(side='left', padx=(0, 20), pady=10)

        # Add mousewheel support to limit dropdown
        add_mousewheel_support(self.limit_combo)

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
        """Create enhanced results display with detailed view capability."""
        self.results_frame = ctk.CTkFrame(self.main_container)
        self.results_frame.pack(fill='x', pady=(0, 20))

        self.results_label = ctk.CTkLabel(
            self.results_frame,
            text="Query Results:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.results_label.pack(anchor='w', padx=15, pady=(15, 10))
        
        # Results summary with export button
        self.summary_frame = ctk.CTkFrame(self.results_frame)
        self.summary_frame.pack(fill='x', padx=15, pady=(0, 10))
        
        self.summary_label = ctk.CTkLabel(
            self.summary_frame,
            text="No results to display",
            font=ctk.CTkFont(size=12)
        )
        self.summary_label.pack(side='left', padx=10, pady=5)
        
        # Export selected button
        self.export_selected_btn = ctk.CTkButton(
            self.summary_frame,
            text="Export Selected",
            command=self._export_selected_results,
            width=120,
            height=30,
            state="disabled"
        )
        self.export_selected_btn.pack(side='right', padx=10, pady=5)

        # Results table using scrollable frame with grid layout
        self.table_container = ctk.CTkFrame(self.results_frame)
        self.table_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        # Table header
        self.table_header_frame = ctk.CTkFrame(self.table_container)
        self.table_header_frame.pack(fill='x', padx=0, pady=(0, 1))
        
        # Define enhanced column widths
        self.column_widths = {
            '': 30,  # Selection checkbox
            'Date': 100,
            'Model': 80,
            'Serial': 120,
            'Status': 60,
            'Yield': 60,
            'Sigma': 80,
            'Linearity': 80,
            'Risk': 60,
            'View': 60  # Details button
        }
        
        # Create header labels
        for i, (col, width) in enumerate(self.column_widths.items()):
            header_label = ctk.CTkLabel(
                self.table_header_frame,
                text=col,
                width=width,
                font=ctk.CTkFont(size=12, weight="bold"),
                anchor='w'
            )
            header_label.grid(row=0, column=i, padx=5, pady=5, sticky='w')
            self.table_header_frame.columnconfigure(i, weight=0, minsize=width)
        
        # Scrollable results area
        self.results_scroll_frame = ctk.CTkScrollableFrame(
            self.table_container,
            height=300
        )
        self.results_scroll_frame.pack(fill='both', expand=True, padx=0, pady=0)
        
        # Configure grid columns for results
        for i, width in enumerate(self.column_widths.values()):
            self.results_scroll_frame.columnconfigure(i, weight=0, minsize=width)

    def _create_manufacturing_insights(self):
        """Create company-wide manufacturing insights section."""
        self.insights_frame = ctk.CTkFrame(self.main_container)
        self.insights_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.insights_label = ctk.CTkLabel(
            self.insights_frame,
            text="Company-Wide Manufacturing Analytics:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.insights_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Charts container
        self.insights_container = ctk.CTkFrame(self.insights_frame)
        self.insights_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Enhanced chart tabs
        self.insights_tabview = ctk.CTkTabview(self.insights_container)
        self.insights_tabview.pack(fill='both', expand=True, padx=10, pady=10)

        # Add company-wide focused tabs
        self.insights_tabview.add("Production Overview")
        self.insights_tabview.add("Cross-Model Analysis")
        self.insights_tabview.add("Quality Trends")
        self.insights_tabview.add("Efficiency Metrics")

        # Production Overview - timeline of production volumes and yield
        overview_frame = ctk.CTkFrame(self.insights_tabview.tab("Production Overview"))
        overview_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        overview_info = ctk.CTkLabel(
            overview_frame,
            text="Company-wide production volume and yield rates over time. Track overall manufacturing performance.",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        overview_info.pack(anchor='w', padx=5, pady=(5, 0))
        
        self.production_chart = ChartWidget(
            overview_frame,
            chart_type='line',
            title="Production Volume & Yield with 30-Day MA",
            figsize=(12, 6)  # Standard dashboard size
        )
        self.production_chart.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Cross-Model Analysis - compare performance across all models
        model_frame = ctk.CTkFrame(self.insights_tabview.tab("Cross-Model Analysis"))
        model_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        model_info = ctk.CTkLabel(
            model_frame,
            text="Compare key metrics across all active models to identify best practices and problem areas.",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        model_info.pack(anchor='w', padx=5, pady=(5, 0))
        
        self.model_comparison_chart = ChartWidget(
            model_frame,
            chart_type='bar',  # Use 'bar' instead of 'grouped_bar'
            title="Model Performance Comparison",
            figsize=(10, 5)
        )
        self.model_comparison_chart.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Quality Trends - overall quality metrics over time
        quality_frame = ctk.CTkFrame(self.insights_tabview.tab("Quality Trends"))
        quality_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        quality_info = ctk.CTkLabel(
            quality_frame,
            text="Track company-wide quality indicators including defect rates, sigma levels, and compliance metrics.",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        quality_info.pack(anchor='w', padx=5, pady=(5, 0))
        
        self.quality_trends_chart = ChartWidget(
            quality_frame,
            chart_type='line',
            title="Quality KPIs with Control Limits",
            figsize=(12, 6)  # Standard for multi-metric chart
        )
        self.quality_trends_chart.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Efficiency Metrics - process efficiency and productivity
        efficiency_frame = ctk.CTkFrame(self.insights_tabview.tab("Efficiency Metrics"))
        efficiency_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        efficiency_info = ctk.CTkLabel(
            efficiency_frame,
            text="Manufacturing efficiency metrics including throughput, trim effectiveness, and resource utilization.",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        efficiency_info.pack(anchor='w', padx=5, pady=(5, 0))
        
        self.efficiency_chart = ChartWidget(
            efficiency_frame,
            chart_type='bar',
            title="Manufacturing Efficiency Metrics",
            figsize=(10, 6)  # Standard bar chart size
        )
        self.efficiency_chart.pack(fill='both', expand=True, padx=5, pady=5)

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
            # Log database information for debugging
            if hasattr(self.db_manager, 'db_path'):
                logger.info(f"Using database: {self.db_manager.db_path}")
            else:
                logger.info("Database path not available")
            
            # Get filter values for company-wide analysis
            dept = self.dept_var.get() if self.dept_var.get() != "All" else None
            shift = self.shift_var.get() if self.shift_var.get() != "All" else None
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

            # Query database for company-wide data
            # Note: For now we query all data and filter in memory since dept/shift aren't DB fields
            query_params = {
                'start_date': start_date,
                'end_date': end_date,
                'limit': limit
            }
            
            # Add optional parameters if they're supported
            if status:
                query_params['status'] = status
            
            # Try with risk_category if available
            try:
                results = self.db_manager.get_historical_data(
                    **query_params,
                    risk_category=risk,
                    include_tracks=True
                )
            except TypeError:
                # Fallback without risk_category if not supported
                logger.warning("risk_category parameter not supported, querying without it")
                results = self.db_manager.get_historical_data(**query_params)

            # Log results summary
            logger.info(f"Retrieved {len(results)} results from database")
            
            # Check for test data
            test_data_count = 0
            for result in results:
                if result.serial and result.serial.startswith('TEST'):
                    test_data_count += 1
            
            if test_data_count > 0:
                logger.warning(f"Found {test_data_count} test data entries (TEST serial numbers)")
                if test_data_count == len(results):
                    logger.warning("ALL data appears to be test data! Check database configuration.")
                    # Show warning to user
                    self.after(0, lambda: messagebox.showwarning(
                        "Fake Test Data Detected",
                        f"The query returned FAKE test data (serial numbers starting with 'TEST').\n\n"
                        f"This means the database was seeded with artificial test data.\n\n"
                        f"To use real data:\n"
                        f"1. Clean the database: python scripts/init_dev_database.py --clean\n"
                        f"2. Analyze actual laser trim files through the app\n"
                        f"3. Or switch to production: set LTA_ENV=production"
                    ))
            
            if results and results[0].tracks:
                first_track = results[0].tracks[0]
                logger.info(f"First track sigma_gradient: {getattr(first_track, 'sigma_gradient', 'NOT FOUND')}")
                # Log all numeric attributes to see what's available
                numeric_attrs = {}
                for attr in dir(first_track):
                    if not attr.startswith('_'):
                        val = getattr(first_track, attr, None)
                        if isinstance(val, (int, float)):
                            numeric_attrs[attr] = val
                logger.info(f"First track numeric attributes: {numeric_attrs}")
            
            # Update UI in main thread
            self.after(0, self._display_results, results)
            self.after(0, lambda: self.query_btn.configure(state='normal', text='Run Query'))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Query Error", f"Failed to query database:\n{str(e)}"))
            self.after(0, lambda: self.query_btn.configure(state='normal', text='Run Query'))

    def _display_results(self, results):
        """Display query results in the table format."""
        logger.info(f"Displaying {len(results)} results")
        try:
            # Clear existing results
            for widget in self.results_scroll_frame.winfo_children():
                widget.destroy()
            
            if not results:
                self.summary_label.configure(text="No data found matching the criteria")
                # Show empty state in table
                empty_label = ctk.CTkLabel(
                    self.results_scroll_frame,
                    text="No results to display",
                    font=ctk.CTkFont(size=14),
                    text_color="gray"
                )
                empty_label.grid(row=0, column=0, columnspan=len(self.column_widths), pady=50)
                return
                
            # Update summary
            total_count = len(results)
            pass_count = sum(1 for r in results if r.overall_status.value == "Pass")
            fail_count = total_count - pass_count
            pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0
            
            summary_text = f"Total: {total_count} | Pass: {pass_count} ({pass_rate:.1f}%) | Fail: {fail_count} ({100-pass_rate:.1f}%)"
            self.summary_label.configure(text=summary_text)
            
            # Track selected results
            self.selected_results = set()
            
            # Display results in enhanced table
            for row_idx, result in enumerate(results):
                # Extract data for each column
                date_str = result.file_date.strftime('%Y-%m-%d %H:%M') if hasattr(result, 'file_date') and result.file_date else (
                    result.timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(result, 'timestamp') else 'Unknown'
                )
                model = str(getattr(result, 'model', 'Unknown'))
                serial = str(getattr(result, 'serial', 'Unknown'))
                status = result.overall_status.value if hasattr(result, 'overall_status') else 'Unknown'
                
                # Calculate yield and metrics from tracks
                sigma = "--"
                linearity = "--"
                risk = "--"
                yield_value = "--"
                
                if result.tracks and len(result.tracks) > 0:
                    # Calculate average values across tracks
                    sigma_values = []
                    linearity_values = []
                    sigma_pass_count = 0
                    
                    for track in result.tracks:
                        # Get sigma gradient value directly from track
                        sigma_val = getattr(track, 'sigma_gradient', None)
                        if sigma_val is not None:
                            sigma_values.append(sigma_val)
                            if row_idx == 0:  # Debug first result
                                logger.debug(f"Track {getattr(track, 'track_id', 'unknown')}: sigma_gradient = {sigma_val}")
                        
                        # Get linearity error
                        linearity_shifted = getattr(track, 'final_linearity_error_shifted', None)
                        linearity_raw = getattr(track, 'final_linearity_error_raw', None)
                        if linearity_shifted is not None:
                            linearity_values.append(abs(linearity_shifted))
                        elif linearity_raw is not None:
                            linearity_values.append(abs(linearity_raw))
                        
                        # Check sigma pass
                        if getattr(track, 'sigma_pass', False):
                            sigma_pass_count += 1
                    
                    # Average sigma gradient
                    if sigma_values:
                        avg_sigma = np.mean(sigma_values)
                        sigma = f"{avg_sigma:.4f}"
                        if row_idx == 0:  # Log first result for debugging
                            logger.info(f"First result: found {len(sigma_values)} sigma values, average = {avg_sigma}")
                    else:
                        if row_idx == 0 and result.tracks:  # Log first result for debugging
                            logger.warning(f"First result: no sigma values found for {len(result.tracks)} tracks")
                    
                    # Average linearity error
                    if linearity_values:
                        linearity = f"{np.mean(linearity_values):.4f}"
                    
                    # Track yield (percentage of tracks passing)
                    if len(result.tracks) > 0:
                        yield_value = f"{(sigma_pass_count / len(result.tracks)) * 100:.0f}%"
                    
                    # Get risk from first track (or worst)
                    track = result.tracks[0]
                    if hasattr(track, 'risk_category') and track.risk_category:
                        risk = track.risk_category.value
                    else:
                        risk = "Unknown"
                
                # Create row frame for alternating colors
                row_frame = ctk.CTkFrame(
                    self.results_scroll_frame,
                    fg_color=("gray90", "gray20") if row_idx % 2 == 0 else ("gray95", "gray25"),
                    corner_radius=0
                )
                row_frame.grid(row=row_idx, column=0, columnspan=len(self.column_widths), sticky='ew', pady=1)
                
                # Configure grid for row frame
                for i, width in enumerate(self.column_widths.values()):
                    row_frame.columnconfigure(i, weight=0, minsize=width)
                
                # Column 0: Selection checkbox
                checkbox_var = tk.BooleanVar()
                checkbox = ctk.CTkCheckBox(
                    row_frame,
                    text="",
                    variable=checkbox_var,
                    width=20,
                    height=20,
                    command=lambda r=result, v=checkbox_var: self._toggle_result_selection(r, v)
                )
                checkbox.grid(row=0, column=0, padx=5, pady=4)
                
                # Create cells
                values = [date_str, model, serial, status, yield_value, sigma, linearity, risk]
                colors = ['Status', 'Risk', 'Yield']  # Columns that need color coding
                
                # Start from column 1 (skip checkbox column)
                for col_idx, (col_name, value) in enumerate(zip(list(self.column_widths.keys())[1:-1], values), start=1):
                    # Determine text color
                    text_color = None
                    if col_name == 'Status':
                        # Use theme-aware colors
                        if value == "Pass":
                            text_color = ("#27ae60", "#2ecc71")  # Green for pass
                        elif value == "Fail":
                            text_color = ("#e74c3c", "#c0392b")  # Red for fail
                        else:
                            text_color = ("#f39c12", "#d68910")  # Orange for warning
                    elif col_name == 'Risk':
                        if value == "Low":
                            text_color = ("#27ae60", "#2ecc71")  # Green for low risk
                        elif value == "Medium":
                            text_color = ("#f39c12", "#d68910")  # Orange for medium
                        elif value == "High":
                            text_color = ("#e74c3c", "#c0392b")  # Red for high
                        else:
                            text_color = ("#95a5a6", "#7f8c8d")  # Gray for unknown
                    elif col_name == 'Yield':
                        # Color code yield percentage
                        try:
                            yield_pct = float(value.rstrip('%'))
                            if yield_pct >= 95:
                                text_color = ("#27ae60", "#2ecc71")  # Green
                            elif yield_pct >= 90:
                                text_color = ("#f39c12", "#d68910")  # Orange
                            else:
                                text_color = ("#e74c3c", "#c0392b")  # Red
                        except:
                            pass
                    
                    label = ctk.CTkLabel(
                        row_frame,
                        text=value,
                        width=list(self.column_widths.values())[col_idx],
                        anchor='w',
                        font=ctk.CTkFont(size=11),
                        text_color=text_color,
                        fg_color="transparent"
                    )
                    label.grid(row=0, column=col_idx, padx=5, pady=4, sticky='w')
                
                # Last column: View details button
                view_btn = ctk.CTkButton(
                    row_frame,
                    text="View",
                    command=lambda r=result: self._show_detailed_analysis(r),
                    width=50,
                    height=24,
                    font=ctk.CTkFont(size=10)
                )
                view_btn.grid(row=0, column=len(self.column_widths)-1, padx=5, pady=4)
                
                # Add hover effect
                def on_enter(event, frame=row_frame, idx=row_idx):
                    frame.configure(fg_color=("gray85", "gray15"))
                    
                def on_leave(event, frame=row_frame, idx=row_idx):
                    frame.configure(
                        fg_color=("gray90", "gray20") if idx % 2 == 0 else ("gray95", "gray25")
                    )
                    
                row_frame.bind("<Enter>", on_enter)
                row_frame.bind("<Leave>", on_leave)
                
                # Make row clickable
                row_frame.bind("<Button-1>", lambda e, r=result: self._on_result_click(r))
            
            # Clear old data to free memory
            if self.current_data_df is not None:
                del self.current_data_df
                self.current_data_df = None
            
            # Store current data for potential export
            self.current_data = results
            
            # Update all visualizations when data is loaded
            try:
                self._update_manufacturing_insights(results)
                # Update risk analysis charts
                self._update_risk_analysis(results)
                # These methods need to be implemented or removed:
                # self._update_qa_metrics(results)
                # self._update_spc_charts(results)  # Add automatic SPC chart updates
                
                # Update dashboard metrics
                self._update_dashboard_metrics(self._analytics_data if hasattr(self, '_analytics_data') and self._analytics_data else [])
                
                # Enable export selected button
                self.export_selected_btn.configure(state="normal")
            except Exception as e:
                logger.error(f"Error updating visualizations: {e}")
            
            logger.info(f"Displayed {len(results)} query results")
            
        except Exception as e:
            logger.error(f"Error displaying results: {e}")
            self.summary_label.configure(text=f"Error displaying results: {str(e)}")

    def _update_manufacturing_insights(self, results):
        """Update manufacturing insight charts with QA-focused visualizations."""
        if not results:
            self.logger.warning("No results to update manufacturing insights")
            # Show placeholders for all charts when no data
            self.production_chart.show_placeholder("No data loaded", "Run a query to view production trends")
            self.model_comparison_chart.show_placeholder("No data loaded", "Run a query to compare models")
            self.quality_trends_chart.show_placeholder("No data loaded", "Run a query to view quality trends")
            self.efficiency_chart.show_placeholder("No data loaded", "Run a query to view efficiency metrics")
            return

        try:
            self.logger.info(f"Updating manufacturing insights with {len(results)} results")
            
            # Prepare data for analysis
            data_records = []
            for result in results:
                base_record = {
                    'date': result.file_date or result.timestamp,
                    'model': result.model,
                    'serial': result.serial,
                    'status': result.overall_status.value,
                    'system': result.system.value if hasattr(result, 'system') else 'Unknown'
                }
                
                # Extract detailed track data
                if result.tracks:
                    for track in result.tracks:
                        record = base_record.copy()
                        # Access attributes directly from track object (not nested)
                        record.update({
                            'track_id': getattr(track, 'track_id', None),
                            'sigma_gradient': getattr(track, 'sigma_gradient', None),
                            'sigma_pass': getattr(track, 'sigma_pass', False),
                            'linearity_error': abs(getattr(track, 'final_linearity_error_shifted', 
                                                 getattr(track, 'final_linearity_error_raw', 0))),
                            'linearity_pass': getattr(track, 'linearity_pass', False),
                            'trim_improvement': getattr(track, 'improvement_percent', 85.0),  # Default 85%
                            'untrimmed_rms': getattr(track, 'untrimmed_rms_error', None),
                            'trimmed_rms': getattr(track, 'trimmed_rms_error', None),
                            'risk_category': track.risk_category.value if hasattr(track, 'risk_category') and track.risk_category else 'Unknown',
                            'failure_probability': getattr(track, 'failure_probability', None),
                            'range_utilization': getattr(track, 'range_utilization_percent', None),
                            'processing_time': getattr(result, 'processing_time', 0)
                        })
                        data_records.append(record)
                else:
                    data_records.append(base_record)
            
            # Convert to DataFrame
            df = pd.DataFrame(data_records)
            self.current_data_df = df
            self._analytics_data = data_records  # Store for dashboard metrics update
            
            self.logger.info(f"Prepared DataFrame with {len(df)} records for manufacturing insights")
            self.logger.debug(f"DataFrame columns: {df.columns.tolist()}")
            self.logger.debug(f"DataFrame shape: {df.shape}")
            if not df.empty:
                self.logger.debug(f"Sample data:\n{df.head()}")
            
            # Update company-wide insight charts
            self._update_production_overview(df)
            self._update_model_comparison(df)
            self._update_quality_trends(df)
            self._update_efficiency_metrics(df)
            
            
        except Exception as e:
            logger.error(f"Error updating manufacturing insights: {e}")
    
    def _update_production_overview(self, df):
        """Update production overview chart with company-wide data."""
        if df.empty:
            self.production_chart.show_placeholder("No production data available", "Run a query to view production trends")
            return
            
        try:
            # Calculate daily production volume and yield
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Calculate pass rate based on both sigma and linearity pass
            df['overall_pass'] = df['sigma_pass'] & df['linearity_pass']
            
            # Group by date for company-wide view
            daily_data = df.groupby('date').agg({
                'overall_pass': ['sum', 'count', lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0]
            }).reset_index()
            daily_data.columns = ['date', 'passed_units', 'total_units', 'pass_rate']
            
            # Sort by date
            daily_data = daily_data.sort_values('date')
            
            if len(daily_data) > 0:
                # Add moving averages for trend smoothing
                if len(daily_data) >= 7:
                    daily_data['ma_7'] = daily_data['pass_rate'].rolling(window=7, min_periods=3).mean()
                if len(daily_data) >= 30:
                    daily_data['ma_30'] = daily_data['pass_rate'].rolling(window=30, min_periods=15).mean()
                
                # Clear and create custom chart with moving averages
                self.production_chart.clear_chart()
                ax = self.production_chart._get_or_create_axes()
                
                # Plot daily pass rate - adjust style based on data density
                num_points = len(daily_data)
                if num_points > 100:
                    # For many points, use line only without markers
                    ax.plot(daily_data['date'], daily_data['pass_rate'], 
                           '-', linewidth=1, 
                           color=self.production_chart.qa_colors['primary'], 
                           label='Daily Pass Rate', alpha=0.6)
                elif num_points > 50:
                    # For moderate points, use smaller markers
                    ax.plot(daily_data['date'], daily_data['pass_rate'], 
                           'o-', markersize=2, linewidth=1, 
                           color=self.production_chart.qa_colors['primary'], 
                           label='Daily Pass Rate', alpha=0.6)
                else:
                    # For few points, use normal markers
                    ax.plot(daily_data['date'], daily_data['pass_rate'], 
                           'o-', markersize=4, linewidth=1, 
                           color=self.production_chart.qa_colors['primary'], 
                           label='Daily Pass Rate', alpha=0.6)
                
                # Add moving averages
                if 'ma_7' in daily_data.columns:
                    ax.plot(daily_data['date'], daily_data['ma_7'], 
                           linewidth=2, color='blue', 
                           label='7-day MA', alpha=0.8)
                
                if 'ma_30' in daily_data.columns:
                    ax.plot(daily_data['date'], daily_data['ma_30'], 
                           linewidth=2.5, color='purple', 
                           label='30-day MA', alpha=0.9)
                
                # Add target line
                ax.axhline(y=95, color='green', linestyle='--', linewidth=2, 
                          label='Target: 95%', alpha=0.7)
                
                # Formatting
                ax.set_xlabel('Date')
                ax.set_ylabel('Pass Rate (%)')
                ax.set_title('Production Volume & Yield with Moving Averages', fontsize=14, fontweight='bold')
                ax.set_ylim(0, 105)
                
                # Format dates with better spacing
                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                
                # Adjust locator based on date range
                date_range = (daily_data['date'].max() - daily_data['date'].min()).days
                if date_range > 60:
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                elif date_range > 30:
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                else:
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, date_range // 10)))
                
                # Rotate and align labels to prevent overlap
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
                
                # Add extra padding at the bottom for date labels
                plt.tight_layout(pad=1.5)
                
                # Add legend
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                
                # Add annotation
                latest_rate = daily_data['pass_rate'].iloc[-1]
                trend = "↗" if len(daily_data) > 1 and daily_data['pass_rate'].iloc[-1] > daily_data['pass_rate'].iloc[-7:].mean() else "↘"
                annotation = f"Latest: {latest_rate:.1f}% {trend} | 7-day avg: {daily_data['ma_7'].iloc[-1]:.1f}%" if 'ma_7' in daily_data.columns else f"Latest: {latest_rate:.1f}%"
                self.production_chart.add_chart_annotation(ax, annotation, position='top')
                
                self.production_chart.canvas.draw_idle()
                self.logger.info("Production chart updated with moving averages")
            else:
                self.production_chart.show_placeholder("No production data available", "Run a query to view trends")
            
        except Exception as e:
            self.logger.error(f"Error updating production overview: {e}")
            self.production_chart.show_error(
                "Production Chart Error",
                f"Failed to calculate production metrics: {str(e)}"
            )
    
    def _update_model_comparison(self, df):
        """Update model comparison chart with cross-model performance metrics."""
        if df.empty:
            self.model_comparison_chart.show_placeholder("No data available", "Run a query to compare models")
            return
            
        try:
            # Calculate key metrics by model
            model_metrics = df.groupby('model').agg({
                'sigma_pass': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0,
                'linearity_pass': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0,
                'sigma_gradient': 'mean',
                'linearity_error': lambda x: x.abs().mean() if 'linearity_error' in df.columns else 0,
                'risk_category': lambda x: (x == 'High').sum() if 'risk_category' in df.columns else 0
            }).reset_index()
            
            if len(model_metrics) > 0:
                # Calculate overall pass rate for each model
                model_metrics['overall_pass'] = (model_metrics['sigma_pass'] + model_metrics['linearity_pass']) / 2
                
                # Limit to top 20 models if there are too many
                # Sort by pass rate to show best and worst performers
                if len(model_metrics) > 20:
                    # Get top 10 and bottom 10 performers
                    model_metrics_sorted = model_metrics.sort_values('overall_pass', ascending=False)
                    top_10 = model_metrics_sorted.head(10)
                    bottom_10 = model_metrics_sorted.tail(10)
                    model_metrics = pd.concat([top_10, bottom_10]).sort_values('overall_pass', ascending=False)
                    
                    self.logger.info(f"Limiting chart to top 10 and bottom 10 models out of {len(model_metrics_sorted)} total")
                
                # Prepare data for bar chart
                models = model_metrics['model'].tolist()
                overall_pass = model_metrics['overall_pass'].tolist()
                
                # Create data for bar chart
                chart_data = pd.DataFrame({
                    'month_year': models,  # Using month_year column for categories
                    'track_status': overall_pass  # Values to display
                })
                
                # Update the chart
                self.model_comparison_chart.update_chart_data(chart_data)
                
                # Add threshold line
                self.model_comparison_chart.add_threshold_lines({'Target': 95}, orientation='horizontal')
                
                # Add annotation if we limited the models
                if len(model_metrics_sorted if 'model_metrics_sorted' in locals() else model_metrics) > 20:
                    ax = self.model_comparison_chart._get_or_create_axes()
                    self.model_comparison_chart.add_chart_annotation(
                        ax, 
                        f"Showing top 10 and bottom 10 performers from {len(model_metrics_sorted)} models",
                        position='top'
                    )
            else:
                self.model_comparison_chart.show_placeholder("No model data available", "Need data from multiple models")
            
        except Exception as e:
            self.logger.error(f"Error updating model comparison: {e}")
            self.model_comparison_chart.show_placeholder("Error displaying chart", f"Error: {str(e)}")
    
    def _update_quality_trends(self, df):
        """Update quality trends chart with company-wide quality indicators."""
        if df.empty:
            self.quality_trends_chart.show_placeholder("No quality data available", "Run a query to view quality trends")
            return
            
        try:
            # Calculate daily quality metrics
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            daily_quality = df.groupby('date').agg({
                'sigma_gradient': 'mean',
                'linearity_error': lambda x: x.abs().mean() if 'linearity_error' in df.columns and x.notna().any() else 0,
                'sigma_pass': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0,
                'linearity_pass': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
            }).reset_index()
            
            # Calculate quality index (composite score)
            daily_quality['quality_index'] = (
                daily_quality['sigma_pass'] * 0.4 + 
                daily_quality['linearity_pass'] * 0.4 +
                (100 - daily_quality['linearity_error'] * 10) * 0.2  # Convert error to quality score
            )
            
            # Sort by date
            daily_quality = daily_quality.sort_values('date')
            
            if len(daily_quality) > 0:
                # Calculate control limits
                mean_quality = daily_quality['quality_index'].mean()
                std_quality = daily_quality['quality_index'].std()
                ucl = min(100, mean_quality + 3 * std_quality)
                lcl = max(0, mean_quality - 3 * std_quality)
                
                # Add moving average
                if len(daily_quality) >= 7:
                    daily_quality['ma_7'] = daily_quality['quality_index'].rolling(window=7, min_periods=3).mean()
                
                # Clear and create control chart
                self.quality_trends_chart.clear_chart()
                ax = self.quality_trends_chart._get_or_create_axes()
                
                # Plot quality index - adjust style based on data density
                dates = pd.to_datetime(daily_quality['date'])
                num_points = len(daily_quality)
                if num_points > 100:
                    # For many points, use line only without markers
                    ax.plot(dates, daily_quality['quality_index'], 
                           '-', linewidth=1, 
                           color=self.quality_trends_chart.qa_colors['primary'], 
                           label='Daily Quality Index', alpha=0.7)
                elif num_points > 50:
                    # For moderate points, use smaller markers
                    ax.plot(dates, daily_quality['quality_index'], 
                           'o-', markersize=2, linewidth=1, 
                           color=self.quality_trends_chart.qa_colors['primary'], 
                           label='Daily Quality Index', alpha=0.7)
                else:
                    # For few points, use normal markers
                    ax.plot(dates, daily_quality['quality_index'], 
                           'o-', markersize=4, linewidth=1, 
                           color=self.quality_trends_chart.qa_colors['primary'], 
                           label='Daily Quality Index', alpha=0.7)
                
                # Add moving average
                if 'ma_7' in daily_quality.columns:
                    ax.plot(pd.to_datetime(daily_quality['date']), daily_quality['ma_7'], 
                           linewidth=2.5, color='blue', 
                           label='7-day MA', alpha=0.8)
                
                # Add control limits
                ax.axhline(y=ucl, color='red', linestyle='--', linewidth=2, 
                          label=f'UCL: {ucl:.1f}', alpha=0.7)
                ax.axhline(y=mean_quality, color='green', linestyle='-', linewidth=2, 
                          label=f'Mean: {mean_quality:.1f}', alpha=0.8)
                ax.axhline(y=lcl, color='red', linestyle='--', linewidth=2, 
                          label=f'LCL: {lcl:.1f}', alpha=0.7)
                ax.axhline(y=95, color='darkgreen', linestyle=':', linewidth=2, 
                          label='Target: 95', alpha=0.6)
                
                # Highlight out-of-control points
                ooc_points = daily_quality[(daily_quality['quality_index'] > ucl) | 
                                         (daily_quality['quality_index'] < lcl)]
                if len(ooc_points) > 0:
                    ax.scatter(pd.to_datetime(ooc_points['date']), ooc_points['quality_index'], 
                              color='red', s=100, marker='x', linewidth=3, 
                              label='Out of Control', zorder=5)
                
                # Formatting
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel('Quality Index (%)', fontsize=10)
                ax.set_title('Quality KPIs with Statistical Control Limits', fontsize=14, fontweight='bold', pad=20)
                ax.set_ylim(0, 105)
                
                # Format dates with better spacing
                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                
                # Adjust date ticks based on range
                date_range = (daily_quality['date'].max() - daily_quality['date'].min()).days
                if date_range > 60:
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                elif date_range > 30:
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                else:
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, date_range // 10)))
                
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
                
                # Add legend with better positioning
                ax.legend(loc='upper left', fontsize=8, framealpha=0.9, bbox_to_anchor=(0.02, 0.98))
                ax.grid(True, alpha=0.3)
                
                # Ensure proper layout
                plt.tight_layout(pad=1.5)
                
                # Add annotation
                process_capability = 'Stable' if len(ooc_points) == 0 else 'Unstable'
                annotation = f"Process: {process_capability} | OOC Points: {len(ooc_points)}"
                self.quality_trends_chart.add_chart_annotation(ax, annotation, position='top')
                
                self.quality_trends_chart.canvas.draw_idle()
            else:
                self.quality_trends_chart.show_placeholder("No quality trend data", "Need more data points")
            
        except Exception as e:
            self.logger.error(f"Error updating quality trends: {e}")
            self.quality_trends_chart.show_placeholder("Error displaying trends", str(e))
    
    def _update_risk_analysis(self, results):
        """Update risk analysis charts with data."""
        if not results:
            self.risk_overview_chart.show_placeholder("No data loaded", "Run a query to view risk analysis")
            self.model_risk_chart.show_placeholder("No data loaded", "Run a query to compare model risks")
            self.risk_trends_chart.show_placeholder("No data loaded", "Run a query to view risk trends")
            return
            
        try:
            # Calculate risk categories
            risk_counts = {'Low Risk': 0, 'Medium Risk': 0, 'High Risk': 0, 'Critical': 0}
            model_risks = {}
            risk_timeline = {}
            
            for result in results:
                # Determine risk level based on status
                if result.overall_status.value == 'Pass':
                    risk_level = 'Low Risk'
                elif result.overall_status.value == 'Conditional Pass':
                    risk_level = 'Medium Risk'
                elif result.overall_status.value == 'Fail':
                    # Check if it's critical based on sigma gradient
                    if result.tracks and any(track.sigma_gradient > 0.05 for track in result.tracks if hasattr(track, 'sigma_gradient')):
                        risk_level = 'Critical'
                    else:
                        risk_level = 'High Risk'
                else:
                    risk_level = 'Medium Risk'
                    
                risk_counts[risk_level] += 1
                
                # Count by model
                if result.model not in model_risks:
                    model_risks[result.model] = {'Low Risk': 0, 'Medium Risk': 0, 'High Risk': 0, 'Critical': 0}
                model_risks[result.model][risk_level] += 1
                
                # Track over time
                date_key = result.file_date.date() if result.file_date else result.timestamp.date()
                if date_key not in risk_timeline:
                    risk_timeline[date_key] = {'Low Risk': 0, 'Medium Risk': 0, 'High Risk': 0, 'Critical': 0}
                risk_timeline[date_key][risk_level] += 1
            
            # Update risk overview pie chart
            if sum(risk_counts.values()) > 0:
                pie_data = pd.DataFrame({
                    'category': list(risk_counts.keys()),
                    'count': list(risk_counts.values())
                })
                self.risk_overview_chart.plot_pie(
                    pie_data['count'].tolist(),
                    pie_data['category'].tolist(),
                    colors=['green', 'yellow', 'orange', 'red']
                )
            
            # Update model comparison chart
            if model_risks:
                # Prepare data for stacked bar chart
                models = list(model_risks.keys())
                risk_categories = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical']
                
                # Create data for each risk category
                chart_data = pd.DataFrame()
                for risk in risk_categories:
                    chart_data[risk] = [model_risks[model][risk] for model in models]
                chart_data['model'] = models
                
                # Plot stacked bar chart
                self.model_risk_chart.clear_chart()
                fig = self.model_risk_chart.figure
                ax = fig.add_subplot(111)
                self.model_risk_chart._apply_theme_to_axes(ax)
                
                # Create stacked bars
                bottom = np.zeros(len(models))
                colors = ['green', 'yellow', 'orange', 'red']
                for i, risk in enumerate(risk_categories):
                    ax.bar(models, chart_data[risk], bottom=bottom, label=risk, color=colors[i])
                    bottom += chart_data[risk]
                
                ax.set_xlabel('Model')
                ax.set_ylabel('Count')
                ax.set_title('Risk Distribution by Model')
                ax.legend()
                
                # Rotate labels if many models
                if len(models) > 5:
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                self.model_risk_chart.canvas.draw()
            
            # Update risk trends chart
            if risk_timeline:
                # Convert to DataFrame for easier plotting
                timeline_df = pd.DataFrame(risk_timeline).T.fillna(0)
                timeline_df = timeline_df.sort_index()
                
                # Plot trends
                self.risk_trends_chart.clear_chart()
                fig = self.risk_trends_chart.figure
                ax = fig.add_subplot(111)
                self.risk_trends_chart._apply_theme_to_axes(ax)
                
                # Plot each risk category
                for i, risk in enumerate(['Critical', 'High Risk', 'Medium Risk', 'Low Risk']):
                    if risk in timeline_df.columns:
                        ax.plot(timeline_df.index, timeline_df[risk], 
                               marker='o', label=risk, color=['red', 'orange', 'yellow', 'green'][i])
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Count')
                ax.set_title('Risk Trends Over Time')
                ax.legend()
                
                # Format dates
                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                self.risk_trends_chart.canvas.draw()
                
        except Exception as e:
            self.logger.error(f"Error updating risk analysis: {e}")
            self.risk_overview_chart.show_placeholder("Error", f"Failed to analyze risk: {str(e)}")
            self.model_risk_chart.show_placeholder("Error", f"Failed to analyze risk: {str(e)}")
            self.risk_trends_chart.show_placeholder("Error", f"Failed to analyze risk: {str(e)}")
    
    def _update_efficiency_metrics(self, df):
        """Update efficiency metrics chart with process performance indicators."""
        if df.empty:
            self.efficiency_chart.show_placeholder("No efficiency data available", "Run a query to view efficiency metrics")
            return
            
        try:
            # Calculate efficiency metrics by production period (weekly)
            df['date'] = pd.to_datetime(df['date'])
            df['week'] = df['date'].dt.to_period('W')
            
            # Calculate weekly efficiency metrics
            weekly_efficiency = df.groupby('week').agg({
                'trim_improvement': lambda x: x.dropna().mean() if x.dropna().any() else 85.0,
                'sigma_gradient': 'count',  # Production volume
                'sigma_pass': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0,
                'linearity_pass': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
            }).reset_index()
            
            # Add processing time if available
            if 'processing_time' in df.columns:
                weekly_efficiency['avg_process_time'] = df.groupby('week')['processing_time'].mean().values
            else:
                weekly_efficiency['avg_process_time'] = 0
            
            weekly_efficiency.columns = ['week', 'trim_efficiency', 'production_volume', 'sigma_yield', 'linearity_yield', 'avg_process_time']
            
            # Calculate overall efficiency score
            weekly_efficiency['efficiency_score'] = (
                weekly_efficiency['trim_efficiency'] * 0.3 +
                weekly_efficiency['sigma_yield'] * 0.3 +
                weekly_efficiency['linearity_yield'] * 0.3 +
                (100 - weekly_efficiency['avg_process_time']) * 0.1  # Lower process time is better
            )
            
            if len(weekly_efficiency) > 0:
                # Prepare chart data
                chart_data = pd.DataFrame({
                    'month_year': weekly_efficiency['week'].astype(str),
                    'track_status': weekly_efficiency['efficiency_score']
                })
                
                # Update chart
                self.efficiency_chart.update_chart_data(chart_data)
                
                # Add reference lines
                self.efficiency_chart.add_threshold_lines({
                    'Excellent (≥90%)': 90,
                    'Good (≥80%)': 80,
                    'Needs Improvement (<70%)': 70
                }, orientation='horizontal')
            else:
                self.efficiency_chart.show_placeholder("No efficiency data", "Need more production data")
                
        except Exception as e:
            self.logger.error(f"Error updating efficiency metrics: {e}")
            self.efficiency_chart.show_placeholder("Error displaying efficiency", str(e))

    def _prepare_and_update_analytics(self, results):
        """Prepare data and update analytics dashboard."""
        if not results:
            self._update_dashboard_metrics([])
            return
            
        try:
            # Convert results to list of dictionaries for analytics
            analytics_data = []
            for result in results:
                # Base record
                record = {
                    'id': getattr(result, 'id', None),
                    'filename': getattr(result, 'filename', 'Unknown'),
                    'timestamp': getattr(result, 'timestamp', datetime.now()),
                    'file_date': getattr(result, 'file_date', None),
                    'model': getattr(result, 'model', 'Unknown'),
                    'serial': getattr(result, 'serial', 'Unknown'),
                    'overall_status': 'PASS' if result.overall_status.value.upper() == 'PASS' else 'FAIL',
                    'processing_time': getattr(result, 'processing_time', 0)
                }
                
                # Extract track data for detailed analytics
                if result.tracks and len(result.tracks) > 0:
                    # Average values across all tracks
                    sigma_values = []
                    linearity_errors = []
                    risk_categories = []
                    
                    for track in result.tracks:
                        # Debug logging for sigma values
                        sigma_val = getattr(track, 'sigma_gradient', None)
                        if i == 0:  # Only log for first result to avoid spam
                            logger.debug(f"Track {getattr(track, 'track_id', 'unknown')}: sigma_gradient = {sigma_val}")
                        if hasattr(track, 'sigma_gradient') and track.sigma_gradient is not None:
                            sigma_values.append(track.sigma_gradient)
                        
                        if hasattr(track, 'final_linearity_error_shifted') and track.final_linearity_error_shifted is not None:
                            linearity_errors.append(abs(track.final_linearity_error_shifted))
                        elif hasattr(track, 'final_linearity_error_raw') and track.final_linearity_error_raw is not None:
                            linearity_errors.append(abs(track.final_linearity_error_raw))
                        
                        if hasattr(track, 'risk_category') and track.risk_category:
                            risk_cat = track.risk_category.value if hasattr(track.risk_category, 'value') else str(track.risk_category)
                            risk_categories.append(risk_cat)
                    
                    # Add averaged values
                    if sigma_values:
                        record['sigma_gradient'] = np.mean(sigma_values)
                    if linearity_errors:
                        record['linearity_error'] = np.mean(linearity_errors)
                    if risk_categories:
                        # Take the worst risk category
                        if 'High' in risk_categories:
                            record['risk_category'] = 'High'
                        elif 'Medium' in risk_categories:
                            record['risk_category'] = 'Medium'
                        else:
                            record['risk_category'] = 'Low'
                
                analytics_data.append(record)
            
            # Store analytics data for other features
            self._analytics_data = analytics_data
            
            # Update dashboard metrics with prepared data
            self._update_dashboard_metrics(analytics_data)
            
        except Exception as e:
            logger.error(f"Error preparing analytics data: {e}")
            logger.error(traceback.format_exc())
            self._update_dashboard_metrics([])

    def _update_trend_chart(self):
        """Update pass rate trend chart."""
        if self.current_data_df is None or self.current_data_df.empty or not hasattr(self, 'trend_chart') or self.trend_chart is None:
            return

        try:
            # Group by date and calculate pass rate
            df = self.current_data_df.copy()
            # Use file_date if available, otherwise date
            df['date'] = pd.to_datetime(df['file_date'].fillna(df['date'])).dt.date

            daily_stats = df.groupby('date').agg({
                'status': lambda x: (x.str.upper() == 'PASS').mean() * 100
            }).reset_index()
            daily_stats.columns = ['date', 'pass_rate']

            # Sort by date
            daily_stats = daily_stats.sort_values('date')

            # Update chart with data
            if len(daily_stats) > 0:
                # Set title to clarify what we're showing
                self.trend_chart.title = "Pass Rate Over Time"
                
                # Convert dates to datetime for the chart
                daily_stats['date'] = pd.to_datetime(daily_stats['date'])
                # Prepare data in format expected by update_chart_data
                chart_data = pd.DataFrame({
                    'trim_date': daily_stats['date'],
                    'sigma_gradient': daily_stats['pass_rate']  # Using sigma_gradient column for pass rate data
                })
                self.logger.info(f"Updating trend chart with {len(chart_data)} data points")
                self.trend_chart.update_chart_data(chart_data)
            else:
                self.logger.warning("No data for trend chart")
                self.trend_chart.update_chart_data(pd.DataFrame())
        except Exception as e:
            self.logger.error(f"Error updating trend chart: {e}")
            self.trend_chart.update_chart_data(pd.DataFrame())

    def _update_distribution_chart(self):
        """Update sigma gradient distribution chart."""
        if (self.current_data_df is None or self.current_data_df.empty or 
            'sigma_gradient' not in self.current_data_df.columns or 
            not hasattr(self, 'dist_chart') or self.dist_chart is None):
            self.logger.warning("Cannot update distribution chart - missing data or chart widget")
            return

        # Get sigma values
        sigma_values = self.current_data_df['sigma_gradient'].dropna()

        if len(sigma_values) == 0:
            self.logger.warning("No sigma gradient values to plot")
            return

        self.logger.info(f"Updating distribution chart with {len(sigma_values)} sigma values")
        
        # Update chart with data
        # For histogram, we need sigma_gradient data
        chart_data = pd.DataFrame({
            'sigma_gradient': sigma_values
        })
        self.dist_chart.update_chart_data(chart_data)

    def _update_comparison_chart(self):
        """Update model comparison chart."""
        if (self.current_data_df is None or self.current_data_df.empty or 
            not hasattr(self, 'comp_chart') or self.comp_chart is None):
            self.logger.warning("Cannot update comparison chart - missing data or chart widget")
            return

        try:
            # Calculate pass rate by model
            model_stats = self.current_data_df.groupby('model').agg({
                'status': [
                    lambda x: (x.str.upper() == 'PASS').mean() * 100,
                    'count'
                ]
            }).reset_index()

            model_stats.columns = ['model', 'pass_rate', 'count']

            # Filter models with sufficient data
            model_stats = model_stats[model_stats['count'] >= 5]

            if len(model_stats) == 0:
                self.logger.warning("No models with sufficient data (>=5 samples)")
                return

            self.logger.info(f"Updating comparison chart with {len(model_stats)} models")
            
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

            # Update chart with data
            # For bar chart, we need month_year and track_status columns
            chart_data = pd.DataFrame({
                'month_year': model_stats['model'],  # Using month_year column for categories
                'track_status': model_stats['pass_rate']  # Using track_status column for values
            })
            self.comp_chart.update_chart_data(chart_data)
            
        except Exception as e:
            self.logger.error(f"Error updating comparison chart: {e}")
            self.logger.error(traceback.format_exc())

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
        if self.current_data is None or (isinstance(self.current_data, list) and len(self.current_data) == 0):
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
        
        # Show progress dialog for large exports
        export_df = self.current_data_df if self.current_data_df is not None else pd.DataFrame(self.current_data)
        if len(export_df) > 1000:  # Show progress for large datasets
            from laser_trim_analyzer.gui.widgets.progress_widgets_ctk import ProgressDialog
            progress_dialog = ProgressDialog(
                self,
                title="Exporting Data",
                message=f"Preparing to export {len(export_df)} records..."
            )
            progress_dialog.show()
            
            # Run export in thread
            def export_thread():
                try:
                    self._perform_export(filename, export_df, progress_dialog)
                except Exception as e:
                    self.after(0, lambda: progress_dialog.hide() if progress_dialog else None)
                    self.after(0, lambda: messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}"))
                    
            threading.Thread(target=export_thread, daemon=True).start()
        else:
            # Small dataset, export directly
            self._perform_export(filename, export_df, None)

    def _perform_export(self, filename: str, export_df: pd.DataFrame, progress_dialog=None):
        """Perform the actual export with optional progress updates."""
        try:
            if filename.endswith('.xlsx'):
                # Export to Excel with formatting
                if progress_dialog:
                    self.after(0, lambda: progress_dialog.update_progress(
                        "Writing Excel file...", 0.3
                    ))
                    
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # Write main data
                    export_df.to_excel(writer, sheet_name='Historical Data', index=False)
                    
                    if progress_dialog:
                        self.after(0, lambda: progress_dialog.update_progress(
                            "Creating summary sheet...", 0.7
                        ))

                    # Add summary sheet
                    summary = pd.DataFrame({
                        'Metric': ['Total Records', 'Pass Rate', 'Average Sigma', 'Date Range'],
                        'Value': [
                            len(export_df),
                            f"{(export_df['status'] == 'Pass').mean() * 100:.2f}%" if 'status' in export_df.columns else "N/A",
                            f"{export_df['sigma_gradient'].mean():.6f}" if 'sigma_gradient' in export_df.columns else "N/A",
                            f"{export_df['timestamp'].min()} to {export_df['timestamp'].max()}" if 'timestamp' in export_df.columns else "N/A"
                        ]
                    })
                    summary.to_excel(writer, sheet_name='Summary', index=False)

            else:
                # Export to CSV
                if progress_dialog:
                    self.after(0, lambda: progress_dialog.update_progress(
                        "Writing CSV file...", 0.5
                    ))
                export_df.to_csv(filename, index=False)
            
            if progress_dialog:
                self.after(0, lambda: progress_dialog.update_progress(
                    "Export complete!", 1.0
                ))
                self.after(500, lambda: progress_dialog.hide() if progress_dialog else None)

            self.after(0, lambda: messagebox.showinfo("Export Complete", f"Data exported to:\n{filename}"))

        except Exception as e:
            if progress_dialog:
                self.after(0, lambda: progress_dialog.hide())
            raise

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
        if self.current_data is None or (isinstance(self.current_data, list) and len(self.current_data) == 0):
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
                
            # Export to Excel - use DataFrame version
            if self.current_data_df is not None:
                self.current_data_df.to_excel(filename, index=False)
            else:
                # Fallback: convert list to DataFrame
                pd.DataFrame(self.current_data).to_excel(filename, index=False)
            messagebox.showinfo("Export", f"Data exported successfully to:\n{filename}")
            self.logger.info(f"Exported historical data to {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")
    
    def _on_result_click(self, result):
        """Handle click on a result row."""
        # Could be extended to show detailed view, navigate to file, etc.
        logger.debug(f"Result clicked: {getattr(result, 'model', 'Unknown')} - {getattr(result, 'serial', 'Unknown')}")

    def _update_analytics_status(self, status: str, color: str):
        """Update analytics status indicator."""
        self.analytics_status_label.configure(text=f"Analytics Status: {status}")
        
        # Use theme-aware colors
        theme_colors = {
            "green": ("#27ae60", "#2ecc71"),  # Success green
            "orange": ("#f39c12", "#d68910"),  # Warning orange
            "red": ("#e74c3c", "#c0392b"),    # Error red
            "gray": ("#95a5a6", "#7f8c8d")     # Neutral gray
        }
        
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        theme_color = theme_colors.get(color, theme_colors["gray"])
        self.analytics_indicator.configure(text_color=theme_color[1 if is_dark else 0])

    def _update_dashboard_metrics(self, data: List[Dict[str, Any]]):
        """Update company-wide metrics dashboard."""
        if not data:
            # Show empty state
            self.total_production_card.update_value("0")
            self.overall_yield_card.update_value("--%")
            self.active_models_card.update_value("0")
            self.productivity_card.update_value("--")
            self.quality_index_card.update_value("--%")
            self.process_stability_card.update_value("--%")
            self.avg_efficiency_card.update_value("--%")
            self.defect_trend_card.update_value("--%")
            
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
            
            # Total production
            total_production = len(df)
            self.total_production_card.update_value(str(total_production))
            
            # Overall yield (company-wide pass rate)
            if 'status' in df.columns:
                pass_count = len(df[df['status'] == 'Pass'])
                overall_yield = (pass_count / total_production) * 100 if total_production > 0 else 0
                self.overall_yield_card.update_value(f"{overall_yield:.1f}%")
                
                # Set color based on yield
                if overall_yield >= 95:
                    self.overall_yield_card.set_color_scheme('success')
                elif overall_yield >= 90:
                    self.overall_yield_card.set_color_scheme('warning')
                else:
                    self.overall_yield_card.set_color_scheme('error')
            
            # Active models (unique model count)
            if 'model' in df.columns:
                active_models = df['model'].nunique()
                self.active_models_card.update_value(str(active_models))
            
            # Productivity (units per day)
            if 'date' in df.columns and len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                date_range = (df['date'].max() - df['date'].min()).days + 1
                units_per_day = total_production / date_range if date_range > 0 else 0
                self.productivity_card.update_value(f"{units_per_day:.1f}")
                
            # Quality index (composite score)
            quality_factors = []
            if 'status' in df.columns:
                quality_factors.append(overall_yield / 100)
            
            if 'sigma_gradient' in df.columns:
                # Lower sigma is better, normalize to 0-1 scale
                avg_sigma = df['sigma_gradient'].mean()
                sigma_score = max(0, 1 - (avg_sigma - 0.3) / 0.4)  # Assuming 0.3-0.7 range
                quality_factors.append(sigma_score)
                # avg_sigma_card doesn't exist, skip this update
            
            if quality_factors:
                quality_index = np.mean(quality_factors) * 100
                self.quality_index_card.update_value(f"{quality_index:.0f}%")
                
                if quality_index >= 90:
                    self.quality_index_card.set_color_scheme('success')
                elif quality_index >= 75:
                    self.quality_index_card.set_color_scheme('warning')
                else:
                    self.quality_index_card.set_color_scheme('error')
            
            # Process stability (looking at consistency across production)
            if 'sigma_gradient' in df.columns:
                sigma_cv = df['sigma_gradient'].std() / df['sigma_gradient'].mean() if df['sigma_gradient'].mean() > 0 else 0
                stability = max(0, (1 - sigma_cv) * 100)
                self.process_stability_card.update_value(f"{stability:.0f}%")
                
                if stability >= 85:
                    self.process_stability_card.set_color_scheme('success')
                elif stability >= 70:
                    self.process_stability_card.set_color_scheme('warning')
                else:
                    self.process_stability_card.set_color_scheme('error')
            
            # Trim efficiency (improvement percentage)
            trim_efficiency = 85.0  # Placeholder - would calculate from actual before/after data
            self.avg_efficiency_card.update_value(f"{trim_efficiency:.0f}%")
            
            # Defect rate (inverse of yield)
            defect_rate = 100 - overall_yield if 'overall_yield' in locals() else 0
            self.defect_trend_card.update_value(f"{defect_rate:.1f}%")
            
            if defect_rate <= 5:
                self.defect_trend_card.set_color_scheme('success')
            elif defect_rate <= 10:
                self.defect_trend_card.set_color_scheme('warning')
            else:
                self.defect_trend_card.set_color_scheme('error')
            
        except Exception as e:
            logger.error(f"Error updating dashboard metrics: {e}")
            logger.error(traceback.format_exc())

    def _run_trend_analysis(self):
        """Run comprehensive trend analysis."""
        if self.current_data is None or (isinstance(self.current_data, list) and len(self.current_data) == 0):
            messagebox.showwarning("No Data", "Please run a query first to load data for analysis")
            return
            
        self._update_analytics_status("Running Trend Analysis...", "orange")
        self.trend_analysis_btn.configure(state="disabled", text="Analyzing...")
        
        def analyze():
            try:
                # Use the analytics data that has the proper structure
                if hasattr(self, '_analytics_data') and self._analytics_data:
                    trend_data = self._calculate_trend_analysis(self._analytics_data)
                else:
                    # Fallback to preparing data if not available
                    self._prepare_and_update_analytics(self.current_data)
                    trend_data = self._calculate_trend_analysis(self._analytics_data)
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
            
            # Linear regression if scipy is available
            if HAS_SCIPY:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y)
            else:
                # Simple linear regression without scipy
                n = len(x_numeric)
                if n > 1:
                    x_mean = x_numeric.mean()
                    y_mean = y.mean()
                    
                    numerator = ((x_numeric - x_mean) * (y - y_mean)).sum()
                    denominator = ((x_numeric - x_mean) ** 2).sum()
                    
                    slope = numerator / denominator if denominator != 0 else 0
                    intercept = y_mean - slope * x_mean
                    
                    # Calculate R-squared
                    y_pred = slope * x_numeric + intercept
                    ss_res = ((y - y_pred) ** 2).sum()
                    ss_tot = ((y - y_mean) ** 2).sum()
                    # FIXED: Protect against negative values in sqrt (can occur when model is poor)
                    # Clamp R² to [0, 1] range before taking sqrt
                    r_squared = max(0.0, min(1.0, 1 - (ss_res / ss_tot))) if ss_tot != 0 else 0
                    r_value = np.sqrt(r_squared)
                    
                    p_value = 0.05  # Placeholder
                    std_err = 0.1  # Placeholder
                else:
                    slope, intercept, r_value, p_value, std_err = 0, 0, 0, 1, 0
            
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
        if self.current_data is None or (isinstance(self.current_data, list) and len(self.current_data) == 0):
            messagebox.showwarning("No Data", "Please run a query first to load data for analysis")
            return
            
        self._update_analytics_status("Running Correlation Analysis...", "orange")
        self.correlation_analysis_btn.configure(state="disabled", text="Analyzing...")
        
        def analyze():
            try:
                # Use the analytics data that has the proper structure
                if hasattr(self, '_analytics_data') and self._analytics_data:
                    correlation_data = self._calculate_correlation_matrix(self._analytics_data)
                else:
                    # Fallback to preparing data if not available
                    self._prepare_and_update_analytics(self.current_data)
                    correlation_data = self._calculate_correlation_matrix(self._analytics_data)
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
        if self.current_data is None or (isinstance(self.current_data, list) and len(self.current_data) == 0):
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
        if self.current_data is None or (isinstance(self.current_data, list) and len(self.current_data) == 0):
            messagebox.showwarning("No Data", "Please run a query first to load data for analysis")
            return
            
        self._update_analytics_status("Running Predictive Analysis...", "orange")
        self.predictive_analysis_btn.configure(state="disabled", text="Predicting...")
        
        def analyze():
            try:
                # Use the analytics data that has the proper structure
                if hasattr(self, '_analytics_data') and self._analytics_data:
                    prediction_data = self._build_predictive_models(self._analytics_data)
                else:
                    # Fallback to preparing data if not available
                    self._prepare_and_update_analytics(self.current_data)
                    prediction_data = self._build_predictive_models(self._analytics_data)
                
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
        if not HAS_SKLEARN:
            return {'error': 'Machine learning libraries not available. Install scikit-learn for predictive features.'}
        
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
        if self.current_data is None or (isinstance(self.current_data, list) and len(self.current_data) == 0):
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
    
    def set_filter(self, **kwargs):
        """Set filter values for the historical page.
        
        Args:
            **kwargs: Filter parameters (e.g., risk_category='High')
        """
        # Store filters to apply when page is shown
        self._pending_filters = kwargs
        
        # If UI is already created, apply filters immediately
        if hasattr(self, 'risk_combo') and 'risk_category' in kwargs:
            self.risk_var.set(kwargs['risk_category'])
            # Run query automatically after setting filter
            self._run_query()
    
    def on_show(self):
        """Called when page is shown."""
        # Apply any pending filters
        if self._pending_filters:
            if hasattr(self, 'risk_combo') and 'risk_category' in self._pending_filters:
                self.risk_var.set(self._pending_filters['risk_category'])
            
            # Clear pending filters after applying
            self._pending_filters = {}
            
            # Run query automatically with the applied filter
            self._run_query()
        
        # Mark as visible
        self.is_visible = True
    
    @property
    def db_manager(self):
        """Get database manager from main window."""
        return self.main_window.db_manager if hasattr(self.main_window, 'db_manager') else None
    
    def _analyze_pass_rate_trends(self) -> Dict[str, Any]:
        """Analyze pass rate trends over time."""
        try:
            if self.current_data is None or len(self.current_data) == 0:
                return {}
            
            # Convert to dataframe for analysis
            df = self.current_data.copy()
            
            # Group by month and calculate pass rates
            df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')
            monthly_pass_rates = df.groupby('month').apply(
                lambda x: (x['overall_status'] == 'PASS').mean() * 100
            )
            
            # Calculate trend
            if len(monthly_pass_rates) > 1:
                x = np.arange(len(monthly_pass_rates))
                y = monthly_pass_rates.values
                slope, intercept = np.polyfit(x, y, 1)
                trend = 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable'
            else:
                trend = 'insufficient_data'
                slope = 0
            
            return {
                'monthly_pass_rates': monthly_pass_rates.to_dict(),
                'trend': trend,
                'slope': slope,
                'current_pass_rate': (df['overall_status'] == 'PASS').mean() * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing pass rate trends: {e}")
            return {}
    
    def _update_trend_analysis_chart(self, trend_data: Dict[str, Any]):
        """Update the trend analysis chart."""
        try:
            self.trend_chart.clear_chart()
            
            if not trend_data or 'monthly_pass_rates' not in trend_data:
                return
            
            # Get data
            monthly_rates = trend_data['monthly_pass_rates']
            if not monthly_rates:
                return
            
            # Create chart
            ax = self.trend_chart.figure.add_subplot(111)
            
            # Convert period to string for x-axis
            x_labels = [str(period) for period in monthly_rates.keys()]
            y_values = list(monthly_rates.values())
            
            # Plot bars
            bars = ax.bar(range(len(x_labels)), y_values)
            
            # Color bars based on pass rate
            for bar, rate in zip(bars, y_values):
                if rate >= 95:
                    bar.set_color('green')
                elif rate >= 90:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            # Add trend line
            if len(x_labels) > 1:
                x = np.arange(len(x_labels))
                slope = trend_data.get('slope', 0)
                intercept = np.mean(y_values) - slope * np.mean(x)
                trend_line = slope * x + intercept
                ax.plot(x, trend_line, 'b--', linewidth=2, label=f"Trend: {trend_data.get('trend', 'unknown')}")
                ax.legend()
            
            # Labels and formatting
            ax.set_xlabel('Month')
            ax.set_ylabel('Pass Rate (%)')
            ax.set_title('Pass Rate Trend Analysis')
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_ylim(0, 105)
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            
            self.trend_chart.figure.tight_layout()
            self.trend_chart.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating trend analysis chart: {e}")
    
    def _update_correlation_heatmap(self, correlation_data: pd.DataFrame):
        """Update the correlation heatmap."""
        try:
            self.correlation_chart.clear_chart()
            
            if correlation_data is None or correlation_data.empty:
                return
            
            # Create heatmap
            ax = self.correlation_chart.figure.add_subplot(111)
            
            # Create correlation matrix
            numeric_cols = ['sigma_gradient', 'linearity_error', 'resistance_change_percent', 
                          'failure_probability', 'processing_time']
            
            # Filter columns that exist
            available_cols = [col for col in numeric_cols if col in correlation_data.columns]
            if len(available_cols) < 2:
                ax.text(0.5, 0.5, 'Insufficient data for correlation analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                self.correlation_chart.canvas.draw()
                return
            
            corr_matrix = correlation_data[available_cols].corr()
            
            # Create heatmap manually
            im = ax.imshow(corr_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            
            # Set ticks
            ax.set_xticks(np.arange(len(available_cols)))
            ax.set_yticks(np.arange(len(available_cols)))
            ax.set_xticklabels(available_cols, rotation=45, ha='right')
            ax.set_yticklabels(available_cols)
            
            # Add colorbar
            cbar = self.correlation_chart.figure.colorbar(im, ax=ax)
            cbar.set_label('Correlation Coefficient')
            
            # Add values
            for i in range(len(available_cols)):
                for j in range(len(available_cols)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha='center', va='center', color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white')
            
            ax.set_title('Feature Correlation Heatmap')
            self.correlation_chart.figure.tight_layout()
            self.correlation_chart.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating correlation heatmap: {e}")
    
    def _update_prediction_chart(self, prediction_data: Dict[str, Any]):
        """Update the predictive analysis chart."""
        try:
            self.prediction_chart.clear_chart()
            
            if not prediction_data:
                return
            
            # Create chart
            ax = self.prediction_chart.figure.add_subplot(111)
            
            # Simple example: show risk distribution
            risk_categories = prediction_data.get('risk_distribution', {})
            if risk_categories:
                categories = list(risk_categories.keys())
                values = list(risk_categories.values())
                
                # Create pie chart
                colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                chart_colors = [colors.get(cat, 'gray') for cat in categories]
                
                ax.pie(values, labels=categories, colors=chart_colors, autopct='%1.1f%%')
                ax.set_title('Risk Distribution Prediction')
            else:
                ax.text(0.5, 0.5, 'No prediction data available', 
                       ha='center', va='center', transform=ax.transAxes)
            
            self.prediction_chart.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating prediction chart: {e}")
    
    def cleanup(self):
        """Clean up resources when page is destroyed."""
        # Destroy dropdown menus to prevent "No more menus can be allocated" error
        combo_widgets = ['model_combo', 'period_combo', 'chart_type_combo']
        
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

    def _update_risk_dashboard(self, results):
        """Update risk analysis dashboard with current data."""
        if not results:
            self.logger.warning("No results to update risk dashboard")
            return
        
        try:
            # Prepare risk data
            risk_counts = {'High': 0, 'Medium': 0, 'Low': 0, 'Unknown': 0}
            high_risk_units = []
            risk_trends = {}
            
            for result in results:
                # Count risk categories
                if result.tracks:
                    for track in result.tracks:
                        risk_cat = 'Unknown'
                        if hasattr(track, 'risk_category') and track.risk_category:
                            risk_cat = track.risk_category.value if hasattr(track.risk_category, 'value') else str(track.risk_category)
                        risk_counts[risk_cat] = risk_counts.get(risk_cat, 0) + 1
                        
                        # Collect high risk units
                        if risk_cat == 'High':
                            risk_score_val = getattr(track, 'failure_probability', 0)
                            if risk_score_val is None:
                                risk_score_val = 0
                            high_risk_units.append({
                                'date': result.file_date or result.timestamp,
                                'model': result.model,
                                'serial': result.serial,
                                'risk_score': risk_score_val,
                                'issue': self._identify_primary_issue(track)
                            })
                        
                        # Track risk trends by model
                        if result.model not in risk_trends:
                            risk_trends[result.model] = []
                        # Ensure we have valid date and risk score
                        date_val = result.file_date or result.timestamp
                        risk_score_val = getattr(track, 'failure_probability', 0)
                        if risk_score_val is None:
                            risk_score_val = 0
                        if date_val is not None:  # Only add if we have a valid date
                            risk_trends[result.model].append({
                                'date': date_val,
                                'risk_score': risk_score_val
                            })
            
            # Update risk distribution chart
            if risk_counts and any(risk_counts.values()):
                # Prepare data for pie chart
                categories = []
                values = []
                for cat, count in risk_counts.items():
                    if count > 0:  # Only include categories with data
                        categories.append(cat)
                        values.append(count)
                
                if categories:
                    # Create figure for pie chart
                    self.risk_dist_chart.clear_chart()
                    fig = self.risk_dist_chart.figure
                    fig.clear()  # Ensure the figure is completely cleared
                    ax = fig.add_subplot(111)
                    
                    # Create pie chart
                    colors = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60', 'Unknown': '#95a5a6'}
                    pie_colors = [colors.get(cat, '#95a5a6') for cat in categories]
                    
                    wedges, texts, autotexts = ax.pie(values, labels=categories, colors=pie_colors, 
                                                      autopct='%1.1f%%', startangle=90)
                    ax.set_title('Risk Distribution')
                    
                    # Equal aspect ratio ensures that pie is drawn as a circle
                    ax.axis('equal')
                    
                    fig.tight_layout()
                    self.risk_dist_chart.canvas.draw()
            
            # Update high risk units list
            # Sort by risk score (handle None values)
            high_risk_units.sort(key=lambda x: x.get('risk_score', 0) or 0, reverse=True)
            self._update_high_risk_list(high_risk_units[:20])  # Show top 20
            
            # Update risk trends chart
            self._update_risk_trends_chart(risk_trends)
            
        except Exception as e:
            self.logger.error(f"Error updating risk dashboard: {e}")
    
    def _update_spc_charts(self, results):
        """Update SPC charts automatically when data is loaded."""
        if not results:
            return
            
        try:
            # Build dataset for control chart (date + sigma_gradient + sigma_threshold)
            chart_rows = []
            for result in results:
                if result.tracks:
                    for track in result.tracks:
                        if getattr(track, 'sigma_gradient', None) is not None:
                            chart_rows.append({
                                'trim_date': result.file_date or result.timestamp,
                                'sigma_gradient': track.sigma_gradient,
                                'sigma_threshold': getattr(track, 'sigma_threshold', None),
                            })

            if not chart_rows:
                self.control_chart.show_placeholder("No control chart data", "Run a query to view control charts")
                return

            df = pd.DataFrame(chart_rows)
            # Coerce dates and drop invalid rows
            df['trim_date'] = pd.to_datetime(df['trim_date'], errors='coerce')
            df = df.dropna(subset=['trim_date', 'sigma_gradient'])
            if df.empty or len(df) < 3:
                self.control_chart.show_placeholder("Insufficient Data", "Need at least 3 data points for control chart")
                return

            # M3: Use model-specific sigma threshold from data (not hardcoded)
            # Sigma gradient is one-sided: must be < threshold (upper limit only)
            sigma_thresholds = df['sigma_threshold'].dropna()
            if len(sigma_thresholds) > 0:
                sigma_threshold_usl = float(sigma_thresholds.median())
                spec_limits = (0.0, sigma_threshold_usl)  # LSL=0, USL=threshold
                target_value = sigma_threshold_usl * 0.5  # Target at 50% of threshold
                self.logger.info(f"Historical SPC chart using model-specific threshold USL: {sigma_threshold_usl:.4f}")
            else:
                # Fallback only if threshold data is missing
                self.logger.warning("No sigma_threshold in results, using fallback limits")
                spec_limits = (0.0, 0.250)
                target_value = 0.125
            self.control_chart.plot_enhanced_control_chart(
                data=df.sort_values('trim_date'),
                value_column='sigma_gradient',
                date_column='trim_date',
                spec_limits=spec_limits,
                target_value=target_value,
                title="Sigma Gradient Control Chart"
            )
            
            # Note: Pareto and drift charts remain on-demand since they require specific analysis
            
        except Exception as e:
            self.logger.error(f"Error updating SPC charts: {e}")
    
    def _update_qa_metrics(self, results):
        """Update QA metrics cards with calculated values."""
        if not results:
            # Reset all metrics
            self.total_records_card.update_value("0")
            self.yield_card.update_value("0%")
            self.high_risk_card.update_value("0")
            self.sigma_pass_rate_card.update_value("0%")
            self.cpk_card.update_value("--")
            self.drift_alert_card.update_value("0")
            self.avg_linearity_card.update_value("--")
            self.unresolved_alerts_card.update_value("0")
            return
        
        try:
            # Calculate metrics
            total = len(results)
            pass_count = sum(1 for r in results if r.overall_status.value == "Pass")
            yield_rate = (pass_count / total * 100) if total > 0 else 0
            
            # Risk and sigma metrics
            high_risk_count = 0
            sigma_pass_count = 0
            linearity_errors = []
            sigma_values = []
            
            for result in results:
                if result.tracks:
                    for track in result.tracks:
                        # Count high risk
                        if hasattr(track, 'risk_category') and track.risk_category:
                            risk_cat = track.risk_category.value if hasattr(track.risk_category, 'value') else str(track.risk_category)
                            if risk_cat == 'High':
                                high_risk_count += 1
                        
                        # Count sigma pass
                        if hasattr(track, 'sigma_pass') and track.sigma_pass:
                            sigma_pass_count += 1
                        
                        # Collect linearity errors
                        if hasattr(track, 'final_linearity_error_shifted') and track.final_linearity_error_shifted is not None:
                            linearity_errors.append(abs(track.final_linearity_error_shifted))
                        
                        # Collect sigma values for Cpk
                        if hasattr(track, 'sigma_gradient') and track.sigma_gradient is not None:
                            sigma_values.append(track.sigma_gradient)
            
            # Calculate average linearity error
            avg_linearity = np.mean(linearity_errors) if linearity_errors else 0
            
            # Calculate Cpk for overall process
            cpk = "--"
            if len(sigma_values) > 3:
                mean_sigma = np.mean(sigma_values)
                std_sigma = np.std(sigma_values)
                if std_sigma > 0:
                    # Assuming spec limits
                    usl = 0.7
                    lsl = 0.3
                    cpu = (usl - mean_sigma) / (3 * std_sigma)
                    cpl = (mean_sigma - lsl) / (3 * std_sigma)
                    cpk_val = min(cpu, cpl)
                    cpk = f"{max(0, cpk_val):.2f}"
            
            # Update metrics
            self.total_records_card.update_value(str(total))
            self.yield_card.update_value(f"{yield_rate:.1f}%")
            self.high_risk_card.update_value(str(high_risk_count))
            
            total_tracks = sum(len(r.tracks) if r.tracks else 0 for r in results)
            sigma_pass_rate = (sigma_pass_count / total_tracks * 100) if total_tracks > 0 else 0
            self.sigma_pass_rate_card.update_value(f"{sigma_pass_rate:.1f}%")
            
            self.cpk_card.update_value(cpk)
            # Get drift alerts count (placeholder - would implement actual detection)
            drift_count = 0
            if hasattr(self, '_drift_alerts'):
                drift_count = self._drift_alerts
            self.drift_alert_card.update_value(str(drift_count))
            
            self.avg_linearity_card.update_value(f"{avg_linearity:.4f}" if avg_linearity > 0 else "--")
            
            # Get unresolved alerts from database
            unresolved_count = 0
            try:
                if hasattr(self.main_window, 'database_manager') or hasattr(self.main_window, 'db_manager'):
                    db_manager = getattr(self.main_window, 'database_manager', None) or getattr(self.main_window, 'db_manager', None)
                    if db_manager:
                        with db_manager.get_session() as session:
                            from laser_trim_analyzer.database.models import QAAlert
                            unresolved_count = session.query(QAAlert).filter(QAAlert.resolved == False).count()
            except Exception as e:
                self.logger.debug(f"Could not query QA alerts: {e}")
            
            self.unresolved_alerts_card.update_value(str(unresolved_count))
            
            # Update color schemes based on values
            if yield_rate >= 95:
                self.yield_card.set_color_scheme("success")
            elif yield_rate >= 90:
                self.yield_card.set_color_scheme("warning")
            else:
                self.yield_card.set_color_scheme("error")
            
            if high_risk_count == 0:
                self.high_risk_card.set_color_scheme("success")
            elif high_risk_count <= 5:
                self.high_risk_card.set_color_scheme("warning")
            else:
                self.high_risk_card.set_color_scheme("error")
            
        except Exception as e:
            self.logger.error(f"Error updating QA metrics: {e}")
    
    def _toggle_result_selection(self, result, checkbox_var):
        """Toggle selection of a result for batch operations."""
        try:
            if checkbox_var.get():
                self.selected_results.add(result.id)
            else:
                self.selected_results.discard(result.id)
            
            # Update export button state
            if len(self.selected_results) > 0:
                self.export_selected_btn.configure(state="normal")
            else:
                self.export_selected_btn.configure(state="disabled")
                
        except Exception as e:
            self.logger.error(f"Error toggling selection: {e}")
    
    def _export_selected_results(self):
        """Export only selected results to file."""
        if not self.selected_results:
            messagebox.showwarning("No Selection", "Please select results to export")
            return
        
        try:
            # Filter results
            selected_data = [r for r in self.current_data if r.id in self.selected_results]
            
            if not selected_data:
                messagebox.showwarning("No Data", "No selected data to export")
                return
            
            # Ask for file location
            filename = filedialog.asksaveasfilename(
                defaultextension='.xlsx',
                filetypes=[
                    ('Excel files', '*.xlsx'),
                    ('CSV files', '*.csv'),
                    ('All files', '*.*')
                ],
                initialfile=f'selected_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            
            if not filename:
                return
            
            # Convert to DataFrame
            export_df = pd.DataFrame([{
                'date': r.file_date or r.timestamp,
                'model': r.model,
                'serial': r.serial,
                'status': r.overall_status.value,
                'system': r.system.value if hasattr(r, 'system') else 'Unknown',
                'processing_time': r.processing_time
            } for r in selected_data])
            
            # Export
            if filename.endswith('.xlsx'):
                export_df.to_excel(filename, index=False)
            else:
                export_df.to_csv(filename, index=False)
            
            messagebox.showinfo("Export Complete", f"Selected data exported to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export selected data:\n{str(e)}")
    
    def _show_detailed_analysis(self, result):
        """Show detailed analysis view for a specific result."""
        try:
            # Create detailed view dialog
            dialog = tk.Toplevel(self.winfo_toplevel())
            dialog.title(f"Detailed Analysis - {result.model} - {result.serial}")
            dialog.geometry("900x700")
            
            # Create notebook for organized view
            notebook = ttk.Notebook(dialog)
            notebook.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Overview tab
            overview_frame = ttk.Frame(notebook)
            notebook.add(overview_frame, text="Overview")
            
            overview_text = tk.Text(overview_frame, wrap='word', width=100, height=30)
            overview_text.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Build overview content
            overview_content = f"""ANALYSIS OVERVIEW
{'=' * 80}
File: {result.filename}
Date: {result.file_date or result.timestamp}
Model: {result.model}
Serial: {result.serial}
System: {result.system.value if hasattr(result, 'system') else 'Unknown'}
Overall Status: {result.overall_status.value}
Processing Time: {result.processing_time:.2f}s

TRACK DETAILS:
"""
            
            # Add track details
            if result.tracks:
                for i, track in enumerate(result.tracks, 1):
                    overview_content += f"\n{'-' * 40}\nTrack {i}: {track.track_id}\n"
                    overview_content += f"Status: {track.status.value}\n"
                    overview_content += f"Sigma Gradient: {track.sigma_gradient:.6f}\n"
                    overview_content += f"Sigma Pass: {'Yes' if track.sigma_pass else 'No'}\n"
                    overview_content += f"Linearity Error: {abs(track.final_linearity_error_shifted or track.final_linearity_error_raw or 0):.6f}\n"
                    overview_content += f"Risk Category: {track.risk_category.value if track.risk_category else 'Unknown'}\n"
                    overview_content += f"Failure Probability: {track.failure_probability:.2%}\n"
                    overview_content += f"Trim Improvement: {track.trim_improvement_percent:.1f}%\n"
            
            overview_text.insert('1.0', overview_content)
            overview_text.configure(state='disabled')
            
            # Metrics tab
            metrics_frame = ttk.Frame(notebook)
            notebook.add(metrics_frame, text="Detailed Metrics")
            
            # Add charts if track data exists
            if result.tracks and len(result.tracks) > 0:
                track = result.tracks[0]  # Use first track for now
                
                # Create figure for metrics visualization
                from matplotlib.figure import Figure
                fig = Figure(figsize=(8, 6))
                
                # Position vs Error plot
                if hasattr(track, 'position_data') and hasattr(track, 'error_data'):
                    ax = fig.add_subplot(111)
                    ax.plot(track.position_data, track.error_data, 'b-', linewidth=2)
                    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                    ax.set_xlabel('Position')
                    ax.set_ylabel('Error')
                    ax.set_title('Position vs Error Profile')
                    ax.grid(True, alpha=0.3)
                    
                    canvas = FigureCanvasTkAgg(fig, metrics_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Close button
            close_btn = ttk.Button(dialog, text="Close", command=dialog.destroy)
            close_btn.pack(pady=10)
            
        except Exception as e:
            self.logger.error(f"Error showing detailed analysis: {e}")
            messagebox.showerror("Error", f"Failed to show detailed analysis:\n{str(e)}")
    
    def _identify_primary_issue(self, track):
        """Identify the primary issue for a track."""
        issues = []
        
        if hasattr(track, 'sigma_analysis') and not track.sigma_analysis.sigma_pass:
            issues.append("Sigma Fail")
        
        if hasattr(track, 'linearity_analysis') and not track.linearity_analysis.linearity_pass:
            issues.append("Linearity Fail")
        
        if hasattr(track, 'dynamic_range') and track.dynamic_range and track.dynamic_range.range_utilization_percent is not None and track.dynamic_range.range_utilization_percent < 80:
            issues.append("Low Range Utilization")
        
        if hasattr(track, 'failure_prediction') and track.failure_prediction and track.failure_prediction.failure_probability is not None and track.failure_prediction.failure_probability > 0.5:
            issues.append("High Failure Risk")
        
        return ", ".join(issues) if issues else "Multiple Issues"
    
    def _update_high_risk_list(self, high_risk_units):
        """Update the high risk units list display."""
        try:
            # Clear existing items
            for widget in self.high_risk_frame.winfo_children():
                if widget.winfo_class() != 'Frame' or widget.winfo_y() > 50:  # Keep header
                    widget.destroy()
            
            # Column widths to match header
            col_widths = [100, 100, 120, 100, 200]
            
            # Add high risk units
            for i, unit in enumerate(high_risk_units):
                row_frame = ctk.CTkFrame(self.high_risk_frame,
                                        fg_color=("gray90", "gray20") if i % 2 == 0 else ("gray95", "gray25"))
                row_frame.pack(fill='x', pady=1)
                
                # Configure column weights to match header
                for j, width in enumerate(col_widths):
                    row_frame.columnconfigure(j, minsize=width, weight=0)
                
                # Date
                date_label = ctk.CTkLabel(
                    row_frame,
                    text=unit['date'].strftime('%Y-%m-%d') if unit['date'] else 'Unknown',
                    width=col_widths[0],
                    anchor='w'
                )
                date_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
                
                # Model
                model_label = ctk.CTkLabel(row_frame, text=unit['model'], width=col_widths[1], anchor='w')
                model_label.grid(row=0, column=1, padx=5, pady=5, sticky='w')
                
                # Serial
                serial_label = ctk.CTkLabel(row_frame, text=unit['serial'], width=col_widths[2], anchor='w')
                serial_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')
                
                # Risk Score
                risk_score = unit.get('risk_score', 0)
                risk_text = f"{risk_score:.2%}" if risk_score is not None else "N/A"
                risk_label = ctk.CTkLabel(
                    row_frame,
                    text=risk_text,
                    width=col_widths[3],
                    text_color="red",
                    anchor='w'
                )
                risk_label.grid(row=0, column=3, padx=5, pady=5, sticky='w')
                
                # Primary Issue
                issue_label = ctk.CTkLabel(row_frame, text=unit['issue'][:40], width=col_widths[4], anchor='w')
                issue_label.grid(row=0, column=4, padx=5, pady=5, sticky='w')
                
        except Exception as e:
            self.logger.error(f"Error updating high risk list: {e}")
    
    def _update_risk_trends_chart(self, risk_trends):
        """Update risk trends chart with better visualization."""
        try:
            self.risk_trend_chart.clear_chart()
            fig = self.risk_trend_chart.figure
            fig.clear()
            ax = fig.add_subplot(111)
            
            # Apply theme
            self.risk_trend_chart._apply_theme_to_axes(ax)
            
            # Get theme colors
            from laser_trim_analyzer.gui.theme_helper import ThemeHelper
            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]
            
            # Prepare data for stacked area chart showing risk distribution over time
            if risk_trends:
                # Collect all dates
                all_dates = set()
                for trends in risk_trends.values():
                    for t in trends:
                        if t['date'] and t['risk_score'] is not None:
                            all_dates.add(t['date'].date() if hasattr(t['date'], 'date') else t['date'])
                
                if all_dates:
                    sorted_dates = sorted(all_dates)
                    
                    # Count risk levels by date
                    date_risk_counts = {date: {'High': 0, 'Medium': 0, 'Low': 0} for date in sorted_dates}
                    
                    for model, trends in risk_trends.items():
                        for t in trends:
                            if t['date'] and t['risk_score'] is not None:
                                date = t['date'].date() if hasattr(t['date'], 'date') else t['date']
                                if date in date_risk_counts:
                                    if t['risk_score'] >= 0.7:
                                        date_risk_counts[date]['High'] += 1
                                    elif t['risk_score'] >= 0.4:
                                        date_risk_counts[date]['Medium'] += 1
                                    else:
                                        date_risk_counts[date]['Low'] += 1
                    
                    # Prepare data for stacked area
                    dates = list(sorted_dates)
                    high_counts = [date_risk_counts[d]['High'] for d in dates]
                    medium_counts = [date_risk_counts[d]['Medium'] for d in dates]
                    low_counts = [date_risk_counts[d]['Low'] for d in dates]
                    
                    # Create stacked area chart
                    ax.fill_between(dates, 0, low_counts, color='#27ae60', alpha=0.7, label='Low Risk')
                    ax.fill_between(dates, low_counts, [l+m for l,m in zip(low_counts, medium_counts)], 
                                  color='#f39c12', alpha=0.7, label='Medium Risk')
                    ax.fill_between(dates, [l+m for l,m in zip(low_counts, medium_counts)], 
                                  [l+m+h for l,m,h in zip(low_counts, medium_counts, high_counts)], 
                                  color='#e74c3c', alpha=0.7, label='High Risk')
                    
                    # Add trend line for total units
                    total_counts = [l+m+h for l,m,h in zip(low_counts, medium_counts, high_counts)]
                    ax.plot(dates, total_counts, 'k-', linewidth=2, alpha=0.8, label='Total Units')
                    
                    # Add annotations for latest state
                    if total_counts and total_counts[-1] > 0:
                        latest_high = high_counts[-1]
                        latest_total = total_counts[-1]
                        high_pct = (latest_high / latest_total * 100) if latest_total > 0 else 0
                        
                        ax.text(0.98, 0.98, f'Latest: {high_pct:.1f}% High Risk ({latest_high}/{latest_total})',
                               transform=ax.transAxes, ha='right', va='top',
                               bbox=dict(boxstyle='round', facecolor='white' if theme_colors["bg"]["primary"] == '#ffffff' else '#2b2b2b', 
                                        alpha=0.8, edgecolor=text_color),
                               color=text_color, fontsize=10)
                    
                    ax.set_xlabel('Date', color=text_color)
                    ax.set_ylabel('Number of Units', color=text_color)
                    ax.set_title('Risk Distribution Over Time', fontsize=14, fontweight='bold', color=text_color)
                    
                    # Fix x-axis and y-axis tick label colors
                    ax.tick_params(axis='x', colors=text_color)
                    ax.tick_params(axis='y', colors=text_color)
                    
                    # Style legend
                    legend = ax.legend(loc='upper left')
                    if legend:
                        self.risk_trend_chart._style_legend(legend)
                    
                    ax.grid(True, alpha=0.3)
                    
                    # Format dates
                    import matplotlib.dates as mdates
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                    if len(dates) > 7:
                        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//7)))
                    
                    fig.autofmt_xdate(rotation=45, ha='right')
                else:
                    ax.text(0.5, 0.5, 'No risk data available\nRun a query to view trends', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, color=text_color)
            else:
                ax.text(0.5, 0.5, 'No risk data available\nRun a query to view trends', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color=text_color)
            
            self.risk_trend_chart.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating risk trends chart: {e}")
    
    def _generate_all_spc_analyses(self):
        """Generate all SPC analyses at once."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data")
            return
        
        try:
            # Show progress
            self.generate_spc_btn.configure(text="⏳ Generating Analyses...", state='disabled')
            
            # Generate all analyses
            self._generate_control_charts()
            self._run_capability_study()
            self._run_pareto_analysis()
            self._detect_process_drift()
            self._analyze_failure_modes()
            
            # Switch to first tab
            self.spc_tabview.set("Control Charts")
            
            # Show success message
            messagebox.showinfo("SPC Analysis Complete", 
                              "All statistical analyses have been generated.\n"
                              "Use the tabs to view different analyses.")
            
        except Exception as e:
            self.logger.error(f"Error generating SPC analyses: {e}")
            messagebox.showerror("Analysis Error", f"Failed to generate analyses:\n{str(e)}")
        finally:
            # Reset button
            self.generate_spc_btn.configure(text="📊 Generate All SPC Analyses", state='normal')
    
    def _generate_control_charts(self):
        """Generate statistical control charts for key parameters."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data")
            return
        
        try:
            # Switch to control charts tab
            self.spc_tabview.set("Control Charts")
            
            # Prepare data
            chart_data = []
            for result in self.current_data:
                if result.tracks:
                    for track in result.tracks:
                        if hasattr(track, 'sigma_gradient') and track.sigma_gradient is not None:
                            chart_data.append({
                                'date': result.file_date or result.timestamp,
                                'value': track.sigma_gradient,
                                'threshold': getattr(track, 'sigma_threshold', None),
                                'model': result.model
                            })

            if not chart_data:
                messagebox.showwarning("No Data", "No sigma gradient data available")
                return

            # Create enhanced control chart using the new method
            df = pd.DataFrame(chart_data).sort_values('date')

            # Rename columns to match expected format
            df = df.rename(columns={'date': 'trim_date', 'value': 'sigma_gradient', 'threshold': 'sigma_threshold'})

            # M3: Use model-specific sigma threshold (one-sided: LSL=0, USL=threshold)
            sigma_thresholds = df['sigma_threshold'].dropna()
            if len(sigma_thresholds) > 0:
                sigma_threshold_usl = float(sigma_thresholds.median())
                spec_limits = (0.0, sigma_threshold_usl)
                target_value = sigma_threshold_usl * 0.5
                self.logger.info(f"Historical trends using model-specific threshold USL: {sigma_threshold_usl:.4f}")
            else:
                # Fallback if threshold missing
                spec_limits = (0.0, 0.250)
                target_value = 0.125
                self.logger.warning("No sigma_threshold for historical trends, using fallback")
            
            # Use the enhanced control chart method
            self.control_chart.plot_enhanced_control_chart(
                data=df,
                value_column='sigma_gradient',
                date_column='trim_date',
                spec_limits=spec_limits,
                target_value=target_value,
                title="Historical Sigma Trends"
            )
            
            self.logger.info(f"Enhanced control chart created with {len(df)} data points")
            
        except Exception as e:
            self.logger.error(f"Error generating control charts: {e}")
            messagebox.showerror("Error", f"Failed to generate control charts:\n{str(e)}")
    
    def _run_capability_study(self):
        """Run process capability study."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data")
            return
        
        try:
            # Switch to capability tab
            self.spc_tabview.set("Process Capability")
            
            # Collect data
            sigma_values = []
            for i, result in enumerate(self.current_data):
                if result.tracks:
                    for track in result.tracks:
                        # Debug logging for sigma values
                        sigma_val = getattr(track, 'sigma_gradient', None)
                        if i == 0:  # Only log for first result to avoid spam
                            self.logger.debug(f"Track {getattr(track, 'track_id', 'unknown')}: sigma_gradient = {sigma_val}")
                        if hasattr(track, 'sigma_gradient') and track.sigma_gradient is not None:
                            sigma_values.append(track.sigma_gradient)
            
            if len(sigma_values) < 30:
                messagebox.showwarning("Insufficient Data", 
                                     f"Need at least 30 samples for capability study. Found: {len(sigma_values)}")
                return
            
            # Calculate statistics
            mean = np.mean(sigma_values)
            std = np.std(sigma_values, ddof=1)
            
            # Specification limits (example)
            usl = 0.7
            lsl = 0.3
            
            # Calculate capability indices
            cp = (usl - lsl) / (6 * std) if std > 0 else 0
            cpu = (usl - mean) / (3 * std) if std > 0 else 0
            cpl = (mean - lsl) / (3 * std) if std > 0 else 0
            cpk = min(cpu, cpl)
            
            # Percentage within spec
            within_spec = sum(1 for v in sigma_values if lsl <= v <= usl)
            pct_within_spec = (within_spec / len(sigma_values)) * 100
            
            # Generate report
            report = f"""PROCESS CAPABILITY STUDY REPORT
{'=' * 80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Parameter: Sigma Gradient

SAMPLE STATISTICS:
    Sample Size: {len(sigma_values)}
    Mean: {mean:.6f}
    Std Dev: {std:.6f}
    Min: {min(sigma_values):.6f}
    Max: {max(sigma_values):.6f}

SPECIFICATION LIMITS:
    LSL: {lsl:.6f}
    USL: {usl:.6f}
    Target: {(usl + lsl) / 2:.6f}

PROCESS CAPABILITY INDICES:
    Cp:  {cp:.3f}  {'(Poor)' if cp < 1.0 else '(Acceptable)' if cp < 1.33 else '(Good)'}
    Cpk: {cpk:.3f} {'(Poor)' if cpk < 1.0 else '(Acceptable)' if cpk < 1.33 else '(Good)'}
    Cpu: {cpu:.3f}
    Cpl: {cpl:.3f}

PERFORMANCE:
    % Within Spec: {pct_within_spec:.2f}%
    Expected PPM Defective: {((1 - pct_within_spec/100) * 1000000):.0f}

INTERPRETATION:
"""
            
            if cpk < 1.0:
                report += "    ⚠️  Process is not capable. Significant improvement needed.\n"
            elif cpk < 1.33:
                report += "    ⚡ Process is marginally capable. Consider improvement.\n"
            else:
                report += "    ✅ Process is capable and meeting specifications.\n"
            
            # Display report
            self.capability_display.configure(state='normal')
            self.capability_display.delete('1.0', tk.END)
            self.capability_display.insert('1.0', report)
            self.capability_display.configure(state='disabled')
            
        except Exception as e:
            self.logger.error(f"Error running capability study: {e}")
            messagebox.showerror("Error", f"Failed to run capability study:\n{str(e)}")
    
    def _run_pareto_analysis(self):
        """Run Pareto analysis on failure modes."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data")
            return
        
        try:
            # Switch to Pareto tab
            self.spc_tabview.set("Pareto Analysis")
            
            # Collect failure data
            failure_counts = {}
            
            for result in self.current_data:
                if result.tracks:
                    for track in result.tracks:
                        failures = []
                        
                        # Check various failure modes
                        if hasattr(track, 'sigma_pass') and not track.sigma_pass:
                            failures.append('Sigma Failure')
                        
                        if hasattr(track, 'linearity_pass') and not track.linearity_pass:
                            failures.append('Linearity Failure')
                        
                        if hasattr(track, 'risk_category') and track.risk_category:
                            risk = track.risk_category.value if hasattr(track.risk_category, 'value') else str(track.risk_category)
                            if risk == 'High':
                                failures.append('High Risk')
                        
                        if hasattr(track, 'range_utilization_percent') and track.range_utilization_percent is not None and track.range_utilization_percent < 80:
                            failures.append('Low Range Utilization')
                        
                        # Count failures
                        for failure in failures:
                            failure_counts[failure] = failure_counts.get(failure, 0) + 1
            
            if not failure_counts:
                messagebox.showinfo("No Failures", "No failures found in the current data")
                return
            
            # Sort by count (descending)
            sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Prepare data for Pareto chart
            categories = [f[0] for f in sorted_failures]
            counts = [f[1] for f in sorted_failures]
            
            # Calculate cumulative percentage
            total = sum(counts)
            cumulative_pct = []
            cumulative = 0
            for count in counts:
                cumulative += count
                cumulative_pct.append((cumulative / total) * 100)
            
            # Create Pareto chart
            self.pareto_chart.clear_chart()
            fig = self.pareto_chart.figure
            ax1 = fig.add_subplot(111)
            
            # Apply theme to axes
            self.pareto_chart._apply_theme_to_axes(ax1)
            
            # Get theme colors
            from laser_trim_analyzer.gui.theme_helper import ThemeHelper
            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]
            is_dark = ThemeHelper.is_dark_theme()
            
            # Use theme-appropriate colors
            bar_color = self.pareto_chart.qa_colors['primary']
            line_color = self.pareto_chart.qa_colors['warning']
            
            # Bar chart
            x = np.arange(len(categories))
            bars = ax1.bar(x, counts, color=bar_color, alpha=0.8, label='Failure Count')
            ax1.set_xlabel('Failure Mode')
            ax1.set_ylabel('Count', color=bar_color)
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories, rotation=45, ha='right')
            ax1.tick_params(axis='y', labelcolor=bar_color)
            
            # Add value labels on bars with proper contrast
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                # Use contrasting color for bar labels
                label_bg = 'white' if is_dark else 'black'
                label_fg = 'black' if is_dark else 'white'
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{count}', ha='center', va='bottom', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=label_bg, 
                                alpha=0.7, edgecolor='none'),
                        color=label_fg, fontsize=9, fontweight='bold')
            
            # Cumulative line
            ax2 = ax1.twinx()
            line = ax2.plot(x, cumulative_pct, color=line_color, marker='o', 
                          linewidth=2.5, markersize=8, label='Cumulative %')
            ax2.set_ylabel('Cumulative %', color=line_color)
            ax2.set_ylim(0, 105)
            ax2.tick_params(axis='y', labelcolor=line_color)
            
            # Add reference lines
            reference_lines = {
                '80% Rule': 80,
                '95% Coverage': 95
            }
            self.pareto_chart.add_reference_lines(ax2, reference_lines)
            
            # Add percentage labels on cumulative line
            for i, (xi, pct) in enumerate(zip(x, cumulative_pct)):
                if pct >= 80 and (i == 0 or cumulative_pct[i-1] < 80):  # Mark 80% point
                    ax2.annotate(f'80% at {categories[i]}', 
                               xy=(xi, pct), xytext=(xi+0.5, pct+5),
                               arrowprops=dict(arrowstyle='->', color=text_color, alpha=0.7),
                               fontsize=9, color=text_color)
                elif i == len(x) - 1:  # Label last point
                    ax2.text(xi, pct + 2, f'{pct:.0f}%', ha='center', fontsize=9, color=text_color)
            
            # Combined legend with better positioning
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            legend = ax1.legend(lines1 + lines2, labels1 + labels2, 
                              loc='center left', bbox_to_anchor=(1.1, 0.5))
            if legend:
                self.pareto_chart._style_legend(legend)
            
            # Title and annotations
            ax1.set_title('Pareto Analysis of Failure Modes', fontweight='bold')
            
            # Add informative annotation
            vital_few = sum(1 for pct in cumulative_pct if pct <= 80)
            annotation_text = (f"{vital_few} of {len(categories)} failure modes account for 80% of issues\n"
                             f"Total failures analyzed: {total}")
            self.pareto_chart.add_chart_annotation(ax1, annotation_text, position='bottom')
            
            # Grid styling already handled by _apply_theme_to_axes
            
            # Use constrained layout instead of tight_layout
            self.pareto_chart.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error running Pareto analysis: {e}")
            messagebox.showerror("Error", f"Failed to run Pareto analysis:\n{str(e)}")
    
    def _detect_process_drift(self):
        """
        Detect process drift using ML-first approach with formula fallback.

        Phase 3 ML Integration: Uses UnifiedProcessor.detect_drift() which
        follows the ThresholdOptimizer pattern (ADR-005):
        1. Check feature flag first
        2. Try ML prediction if model trained
        3. Fall back to statistical CUSUM method if ML not available
        4. Log which method used
        """
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data")
            return

        try:
            # Switch to drift detection tab
            self.spc_tabview.set("Drift Detection")

            # Use UnifiedProcessor for drift detection (ML-first with formula fallback)
            drift_report = self._run_drift_detection()

            if drift_report is None:
                messagebox.showwarning("Insufficient Data",
                                     "Need at least 20 samples for drift detection.")
                return

            # Get method used for logging
            method_used = drift_report.get('method_used', 'formula')
            self.logger.info(f"Drift detection using {method_used} method")

            # Prepare visualization data (extract sigma values for chart)
            drift_data = []
            for result in self.current_data:
                if result.tracks:
                    for track in result.tracks:
                        if hasattr(track, 'sigma_gradient') and track.sigma_gradient is not None:
                            drift_data.append({
                                'date': result.file_date or result.timestamp,
                                'value': track.sigma_gradient,
                                'model': result.model
                            })

            # Convert to DataFrame and sort by date
            df = pd.DataFrame(drift_data).sort_values('date').reset_index(drop=True)

            # Calculate moving averages for visualization
            window_size = min(10, len(df) // 4)
            if window_size < 3:
                window_size = min(3, len(df))
            df['ma'] = df['value'].rolling(window=window_size).mean()
            df['ma_std'] = df['value'].rolling(window=window_size).std()

            # Get target from drift report or calculate
            target = drift_report.get('target_value', df['value'].iloc[:window_size].mean())
            h = drift_report.get('cusum_threshold', 4.0)

            # Plot drift analysis with visualization
            self.drift_chart.clear_chart()
            fig = self.drift_chart.figure
            ax = fig.add_subplot(111)

            # Apply theme to axes
            self.drift_chart._apply_theme_to_axes(ax)

            # Get theme colors
            from laser_trim_analyzer.gui.theme_helper import ThemeHelper
            theme_colors = ThemeHelper.get_theme_colors()
            text_color = theme_colors["fg"]["primary"]

            # Map drift severity to color intensity
            severity = drift_report.get('drift_severity', 'negligible')
            severity_map = {
                'negligible': 0.0, 'low': 0.3, 'moderate': 0.6,
                'high': 0.8, 'critical': 1.0, 'insufficient_data': 0.0
            }
            drift_intensity = severity_map.get(severity, 0.0)

            # Plot individual values as gray dots
            ax.scatter(df.index, df['value'], color='gray', alpha=0.3, s=30, label='Individual Values')

            # Plot moving average with color based on drift severity from report
            for i in range(1, len(df)):
                if pd.notna(df['ma'].iloc[i]) and pd.notna(df['ma'].iloc[i-1]):
                    # Color based on overall drift severity
                    if drift_intensity < 0.3:
                        color = 'green'
                    elif drift_intensity < 0.6:
                        color = 'orange'
                    else:
                        color = 'red'

                    ax.plot([i-1, i], [df['ma'].iloc[i-1], df['ma'].iloc[i]],
                           color=color, linewidth=3, alpha=0.8)

            # Add reference lines
            ax.axhline(y=target, color='blue', linestyle='--', alpha=0.5, label=f'Target: {target:.3f}')
            ax.axhline(y=target + 2*df['value'].std(), color='orange', linestyle=':', alpha=0.5)
            ax.axhline(y=target - 2*df['value'].std(), color='orange', linestyle=':', alpha=0.5)

            # Add drift zones
            ax.fill_between([0, len(df)], [target - 3*df['value'].std()]*2, [target - 2*df['value'].std()]*2,
                           color='yellow', alpha=0.1, label='Warning Zone')
            ax.fill_between([0, len(df)], [target + 2*df['value'].std()]*2, [target + 3*df['value'].std()]*2,
                           color='yellow', alpha=0.1)
            ax.fill_between([0, len(df)], [ax.get_ylim()[0]]*2, [target - 3*df['value'].std()]*2,
                           color='red', alpha=0.1, label='Drift Zone')
            ax.fill_between([0, len(df)], [target + 3*df['value'].std()]*2, [ax.get_ylim()[1]]*2,
                           color='red', alpha=0.1)

            # Mark drift points from the report
            drift_point_indices = [p['index'] for p in drift_report.get('drift_points', [])]
            if drift_point_indices:
                drift_df = df.iloc[drift_point_indices]
                ax.scatter(drift_df.index, drift_df['value'],
                          color='red', s=100, marker='v',
                          label=f'Drift Detected ({len(drift_point_indices)} points)')

            ax.set_xlabel('Sample Number', fontsize=12)
            ax.set_ylabel('Sigma Gradient', fontsize=12)

            # Update title to show method used
            method_label = "ML" if method_used == 'ml' else "Statistical"
            ax.set_title(f'Process Drift Detection ({method_label} Method)',
                        fontsize=14, fontweight='bold')

            # Create custom legend
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='gray', marker='o', linestyle='', alpha=0.3, label='Individual Values'),
                Line2D([0], [0], color='green', linewidth=3, label='Stable (Moving Avg)'),
                Line2D([0], [0], color='orange', linewidth=3, label='Warning'),
                Line2D([0], [0], color='red', linewidth=3, label='Drift Detected'),
                Line2D([0], [0], color='blue', linestyle='--', label='Target'),
                Patch(facecolor='yellow', alpha=0.3, label='Warning Zone'),
                Patch(facecolor='red', alpha=0.3, label='Drift Zone')
            ]

            legend = ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
            if legend:
                self.drift_chart._style_legend(legend)

            ax.grid(True, alpha=0.3)

            # Add explanation text box with recommendations
            recommendations = drift_report.get('recommendations', [])
            rec_text = "\n".join(f"• {r}" for r in recommendations[:3]) if recommendations else "No recommendations"

            explanation = (
                f"Drift Analysis ({method_label}):\n"
                f"• Severity: {severity.upper()}\n"
                f"• Drift Rate: {drift_report.get('drift_rate', 0):.1%}\n"
                f"• Trend: {drift_report.get('drift_trend', 'unknown')}\n"
                f"• Samples: {drift_report.get('samples_analyzed', len(df))}\n\n"
                f"Recommendations:\n{rec_text}"
            )

            ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5',
                            facecolor='white' if ctk.get_appearance_mode().lower() == "light" else '#2b2b2b',
                            alpha=0.8, edgecolor=text_color),
                   color=text_color)

            fig.tight_layout()
            self.drift_chart.canvas.draw()

            # Update drift alert metric
            drift_detected = drift_report.get('drift_detected', False)
            drift_points_count = len(drift_report.get('drift_points', []))

            if drift_detected:
                self._drift_alerts = drift_points_count
                self.drift_alert_card.update_value(str(drift_points_count))
                self.drift_alert_card.set_color_scheme("error")

                # Show drift details with recommendations
                rec_text_msg = "\n".join(f"• {r}" for r in recommendations[:3])
                messagebox.showinfo("Drift Detected",
                                  f"Process drift detected!\n\n"
                                  f"Severity: {severity.upper()}\n"
                                  f"Drift Rate: {drift_report.get('drift_rate', 0):.1%}\n"
                                  f"Points: {drift_points_count}\n"
                                  f"Method: {method_label}\n\n"
                                  f"Recommendations:\n{rec_text_msg}")
            else:
                self._drift_alerts = 0
                self.drift_alert_card.update_value("0")
                self.drift_alert_card.set_color_scheme("success")

                messagebox.showinfo("No Drift",
                                  f"No significant process drift detected.\n"
                                  f"Severity: {severity}\n"
                                  f"Method: {method_label}\n\n"
                                  f"Process appears to be stable.")

        except Exception as e:
            self.logger.error(f"Error detecting process drift: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to detect process drift:\n{str(e)}")

    def _run_drift_detection(self) -> Optional[Dict[str, Any]]:
        """
        Run drift detection using UnifiedProcessor with ML-first approach.

        Returns:
            Drift report dictionary or None if insufficient data
        """
        try:
            # Get config for UnifiedProcessor
            config = get_config()

            # Create UnifiedProcessor for drift detection
            if HAS_UNIFIED_PROCESSOR:
                processor = UnifiedProcessor(config=config)
                drift_report = processor.detect_drift(self.current_data)

                # Check for insufficient data
                if drift_report.get('drift_severity') == 'insufficient_data':
                    return None

                return drift_report
            else:
                # Fallback to manual formula-based detection if UnifiedProcessor unavailable
                self.logger.warning("UnifiedProcessor not available, using inline formula")
                return self._detect_drift_inline_fallback()

        except Exception as e:
            self.logger.error(f"Error in drift detection: {e}")
            return self._detect_drift_inline_fallback()

    def _detect_drift_inline_fallback(self) -> Optional[Dict[str, Any]]:
        """
        Inline fallback drift detection when UnifiedProcessor is not available.

        Uses CUSUM statistical method similar to original implementation.
        """
        # Extract time series data
        drift_data = []
        for result in self.current_data:
            if result.tracks:
                for track in result.tracks:
                    sigma_val = getattr(track, 'sigma_gradient', None)
                    if sigma_val is not None:
                        drift_data.append({
                            'date': result.file_date or result.timestamp,
                            'sigma_gradient': sigma_val
                        })

        if len(drift_data) < 20:
            return None

        # Convert to DataFrame and sort by date
        df = pd.DataFrame(drift_data).sort_values('date').reset_index(drop=True)

        # Calculate CUSUM
        window = min(10, len(df) // 4)
        if window < 3:
            window = min(3, len(df))

        target = df['sigma_gradient'].iloc[:window].mean()
        std = df['sigma_gradient'].std()

        k = 0.5
        h = 4.0

        cusum_pos = []
        cusum_neg = []
        c_pos = 0.0
        c_neg = 0.0

        for value in df['sigma_gradient']:
            c_pos = max(0, c_pos + (value - target) / (std + 1e-6) - k)
            c_neg = max(0, c_neg + (target - value) / (std + 1e-6) - k)
            cusum_pos.append(c_pos)
            cusum_neg.append(c_neg)

        df['cusum_max'] = np.maximum(cusum_pos, cusum_neg)

        drift_mask = df['cusum_max'] > h
        drift_rate = drift_mask.sum() / len(df)

        # Classify severity
        if drift_rate < 0.05:
            severity = 'negligible'
        elif drift_rate < 0.10:
            severity = 'low'
        elif drift_rate < 0.15:
            severity = 'moderate'
        elif drift_rate < 0.25:
            severity = 'high'
        else:
            severity = 'critical'

        # Extract drift points
        drift_points = [
            {'index': int(idx), 'value': float(df.loc[idx, 'sigma_gradient'])}
            for idx in df[drift_mask].index
        ]

        return {
            'drift_detected': drift_rate > 0.1,
            'drift_severity': severity,
            'drift_rate': float(drift_rate),
            'drift_trend': 'stable',
            'drift_points': drift_points,
            'recommendations': ['Process appears stable. Continue standard monitoring.'],
            'feature_drift': {},
            'method_used': 'formula',
            'cusum_threshold': float(h),
            'target_value': float(target),
            'samples_analyzed': len(df)
        }
    
    def _analyze_failure_modes(self):
        """Analyze failure modes and patterns."""
        if not self.current_data:
            messagebox.showwarning("No Data", "Please run a query first to load data")
            return
        
        try:
            # Switch to failure modes tab
            self.spc_tabview.set("Failure Modes")
            
            # Analyze failure patterns
            failure_analysis = {
                'total_units': 0,
                'failed_units': 0,
                'failure_modes': {},
                'correlations': {},
                'recommendations': []
            }
            
            for result in self.current_data:
                failure_analysis['total_units'] += 1
                
                if result.overall_status.value != "Pass":
                    failure_analysis['failed_units'] += 1
                
                if result.tracks:
                    for track in result.tracks:
                        # Identify failure modes
                        failures = []
                        
                        if hasattr(track, 'sigma_pass') and not track.sigma_pass:
                            failures.append('Sigma Failure')
                            
                        if hasattr(track, 'linearity_pass') and not track.linearity_pass:
                            failures.append('Linearity Failure')
                            
                        if hasattr(track, 'risk_category') and track.risk_category:
                            risk = track.risk_category.value if hasattr(track.risk_category, 'value') else str(track.risk_category)
                            if risk == 'High':
                                failures.append('High Risk Classification')
                        
                        # Count failure combinations
                        if failures:
                            failure_key = ' + '.join(sorted(failures))
                            failure_analysis['failure_modes'][failure_key] = \
                                failure_analysis['failure_modes'].get(failure_key, 0) + 1
            
            # Generate recommendations based on analysis
            if failure_analysis['failure_modes']:
                top_failure = max(failure_analysis['failure_modes'].items(), key=lambda x: x[1])
                
                if 'Sigma Failure' in top_failure[0]:
                    failure_analysis['recommendations'].append(
                        "• Review and optimize trim parameters - sigma failures are prevalent"
                    )
                
                if 'Linearity Failure' in top_failure[0]:
                    failure_analysis['recommendations'].append(
                        "• Investigate mechanical alignment and calibration procedures"
                    )
                
                if 'High Risk' in top_failure[0]:
                    failure_analysis['recommendations'].append(
                        "• Implement additional quality checks for high-risk units"
                    )
            
            # Generate report
            report = f"""FAILURE MODE ANALYSIS REPORT
{'=' * 80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
    Total Units Analyzed: {failure_analysis['total_units']}
    Failed Units: {failure_analysis['failed_units']}
    Failure Rate: {(failure_analysis['failed_units'] / failure_analysis['total_units'] * 100):.2f}%

FAILURE MODE BREAKDOWN:
"""
            
            # Sort failure modes by frequency
            sorted_modes = sorted(failure_analysis['failure_modes'].items(), 
                                key=lambda x: x[1], reverse=True)
            
            for mode, count in sorted_modes:
                percentage = (count / failure_analysis['failed_units'] * 100) if failure_analysis['failed_units'] > 0 else 0
                report += f"    {mode}: {count} occurrences ({percentage:.1f}%)\n"
            
            report += f"\nRECOMMENDATIONS:\n"
            for rec in failure_analysis['recommendations']:
                report += f"{rec}\n"
            
            # Display report
            self.failure_mode_display.configure(state='normal')
            self.failure_mode_display.delete('1.0', tk.END)
            self.failure_mode_display.insert('1.0', report)
            self.failure_mode_display.configure(state='disabled')
            
        except Exception as e:
            self.logger.error(f"Error analyzing failure modes: {e}")
            messagebox.showerror("Error", f"Failed to analyze failure modes:\n{str(e)}")
