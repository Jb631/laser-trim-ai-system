"""
Trends Page - Historical analysis and ML insights.

Redesigned to show:
- "All Models" view: Summary of best/worst active models, models needing attention
- Specific model view: SPC scatter chart with threshold, rolling average, detailed stats

Features:
- Focus on models with recent activity (last 90 days)
- Adjustable rolling average window (7/14/30/60 days)
- Alert criteria: <80% pass rate, trending worse, high variance
- Best/worst performing models at a glance
"""

import customtkinter as ctk
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from laser_trim_analyzer.utils.threads import get_thread_manager

import numpy as np

from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.gui.widgets.scrollable_combobox import ScrollableComboBox

# Lazy import for ChartWidget - defer matplotlib loading until first use
if TYPE_CHECKING:
    from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle

logger = logging.getLogger(__name__)

# Module-level flag to track if ChartWidget has been imported
_chart_module_loaded = False
_ChartWidget = None
_ChartStyle = None


def _ensure_chart_module():
    """Lazily load ChartWidget module - defers matplotlib loading."""
    global _chart_module_loaded, _ChartWidget, _ChartStyle
    if not _chart_module_loaded:
        from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle
        _ChartWidget = ChartWidget
        _ChartStyle = ChartStyle
        _chart_module_loaded = True
        logger.debug("ChartWidget module loaded (matplotlib initialized)")
    return _ChartWidget, _ChartStyle


class TrendsPage(ctk.CTkFrame):
    """
    Trends page for historical analysis.

    Two modes:
    1. Summary Mode ("All Models") - Shows best/worst models, alerts
    2. Detail Mode (specific model) - Shows SPC scatter, rolling average, stats
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.selected_model: str = "All Models"
        self.selected_days: int = 3650  # Default to All Time
        self.rolling_window: int = 30
        self.chart_timeline_days: int = 0  # 0 = all data
        self.active_models_data: List[Dict[str, Any]] = []
        self.model_trend_data: Optional[Dict[str, Any]] = None

        # Track chart widgets for proper cleanup
        self._chart_widgets: List["ChartWidget"] = []

        # Lazy chart initialization flags
        self._summary_charts_initialized = False
        self._detail_charts_initialized = False

        self._create_ui()

    def _create_ui(self):
        """Create the trends page UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header
        header = ctk.CTkLabel(
            self,
            text="Trends & ML Insights",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        # Controls frame
        controls = ctk.CTkFrame(self)
        controls.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))

        # Model selector - use ScrollableComboBox for many models
        model_label = ctk.CTkLabel(controls, text="Model:")
        model_label.pack(side="left", padx=(15, 5), pady=15)

        self.model_dropdown = ScrollableComboBox(
            controls,
            values=["All Models"],
            command=self._on_model_change,
            width=150,
            dropdown_height=300,  # Scrollable dropdown
        )
        self.model_dropdown.set("All Models")
        self.model_dropdown.pack(side="left", padx=5, pady=15)

        # Date range for active models consideration (by trim date)
        date_label = ctk.CTkLabel(controls, text="Trim Date:")
        date_label.pack(side="left", padx=(20, 5), pady=15)

        self.date_dropdown = ctk.CTkOptionMenu(
            controls,
            values=["Last 30 Days", "Last 90 Days", "Last Year", "All Time"],
            command=self._on_date_change
        )
        self.date_dropdown.set("All Time")
        self.date_dropdown.pack(side="left", padx=5, pady=15)

        # Rolling average window (only shown in detail mode)
        self.rolling_label = ctk.CTkLabel(controls, text="Rolling Avg:")
        self.rolling_label.pack(side="left", padx=(20, 5), pady=15)

        self.rolling_dropdown = ctk.CTkOptionMenu(
            controls,
            values=["7 Days", "14 Days", "30 Days", "60 Days"],
            command=self._on_rolling_change
        )
        self.rolling_dropdown.set("30 Days")
        self.rolling_dropdown.pack(side="left", padx=5, pady=15)

        # Refresh button
        refresh_btn = ctk.CTkButton(
            controls,
            text="Refresh",
            command=self._refresh_data,
            width=100
        )
        refresh_btn.pack(side="right", padx=15, pady=15)

        # Status label
        self.status_label = ctk.CTkLabel(
            controls,
            text="",
            text_color="gray",
            font=ctk.CTkFont(size=10)
        )
        self.status_label.pack(side="right", padx=15, pady=15)

        # Main content area - will be dynamically updated
        self.content = ctk.CTkScrollableFrame(self)
        self.content.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.content.grid_columnconfigure(0, weight=1)

        # Create placeholder content
        self._create_summary_view()

    def _cleanup_charts(self):
        """Properly destroy chart widgets to free matplotlib resources."""
        import matplotlib.pyplot as plt

        for chart in self._chart_widgets:
            try:
                # Explicitly close the figure before destroying widget
                if hasattr(chart, 'figure') and chart.figure:
                    plt.close(chart.figure)
                chart.destroy()
            except Exception as e:
                logger.debug(f"Chart cleanup warning: {e}")
        self._chart_widgets.clear()

        # Reset initialization flags so charts get recreated on next show
        self._summary_charts_initialized = False
        self._detail_charts_initialized = False

        # Clear any stale data
        self.active_models_data = []
        self.model_trend_data = None

    def _create_summary_view(self):
        """Create the summary view (All Models mode)."""
        # Clean up existing charts first (frees matplotlib figures)
        self._cleanup_charts()
        self._summary_charts_initialized = False

        # Clear existing content
        for widget in self.content.winfo_children():
            widget.destroy()

        self.content.grid_rowconfigure(0, weight=0)  # Stats row - compact
        self.content.grid_rowconfigure(1, weight=1, minsize=200)  # Alerts chart
        self.content.grid_rowconfigure(2, weight=1, minsize=180)  # Top 5 / Recent Issues
        self.content.grid_rowconfigure(3, weight=1, minsize=180)  # Trending Worse / Low Data
        self.content.grid_rowconfigure(4, weight=1, minsize=250)  # Drift Detection section
        self.content.grid_rowconfigure(5, weight=0)  # ML section - compact

        # Summary stats at top
        stats_frame = ctk.CTkFrame(self.content)
        stats_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        stats_label = ctk.CTkLabel(
            stats_frame,
            text="Active Models Summary",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        stats_label.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="w", columnspan=6)

        # Stats in a horizontal row
        self.summary_stat_labels = {}
        stat_names = [
            ("active_models", "Active Models"),
            ("total_samples", "Total Samples"),
            ("avg_pass_rate", "Avg Pass Rate"),
            ("avg_sigma_rate", "Avg Sigma Pass"),
            ("avg_linearity_rate", "Avg Lin Pass"),
            ("models_at_risk", "Models at Risk"),
            ("best_model", "Best Model"),
            ("worst_model", "Worst Model"),
        ]

        for idx, (key, label) in enumerate(stat_names):
            stat_col = ctk.CTkFrame(stats_frame, fg_color="transparent")
            stat_col.grid(row=1, column=idx, padx=15, pady=(0, 15), sticky="w")

            ctk.CTkLabel(stat_col, text=label, text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w")
            value_label = ctk.CTkLabel(stat_col, text="--", font=ctk.CTkFont(size=14, weight="bold"))
            value_label.pack(anchor="w")
            self.summary_stat_labels[key] = value_label

        # Alerts chart (models requiring attention) - placeholder until data loads
        self._alerts_frame = ctk.CTkFrame(self.content)
        self._alerts_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        alerts_label = ctk.CTkLabel(
            self._alerts_frame,
            text="Models Requiring Attention",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        alerts_label.pack(padx=15, pady=(15, 5), anchor="w")

        # Placeholder label instead of ChartWidget
        self._alerts_placeholder = ctk.CTkLabel(
            self._alerts_frame,
            text="Loading models requiring attention...",
            text_color="gray"
        )
        self._alerts_placeholder.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.alerts_chart = None

        # Best/Worst models side by side
        self._models_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self._models_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        self._models_frame.grid_columnconfigure(0, weight=1)
        self._models_frame.grid_columnconfigure(1, weight=1)

        # Best performers - placeholder
        self._best_frame = ctk.CTkFrame(self._models_frame)
        self._best_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        best_label = ctk.CTkLabel(
            self._best_frame,
            text="Top Performing Models",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        best_label.pack(padx=15, pady=(15, 5), anchor="w")

        self._best_placeholder = ctk.CTkLabel(
            self._best_frame,
            text="Loading best models...",
            text_color="gray"
        )
        self._best_placeholder.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.best_chart = None

        # Recent Issues (replaces "worst performers") - placeholder
        self._recent_issues_frame = ctk.CTkFrame(self._models_frame)
        self._recent_issues_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        recent_label = ctk.CTkLabel(
            self._recent_issues_frame,
            text="Recent Issues (Last 30 Days)",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        recent_label.pack(padx=15, pady=(15, 5), anchor="w")

        self._recent_issues_placeholder = ctk.CTkLabel(
            self._recent_issues_frame,
            text="Loading recent issues...",
            text_color="gray"
        )
        self._recent_issues_placeholder.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.recent_issues_chart = None

        # Row 3: Trending Worse / Low Data Models
        self._row3_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self._row3_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
        self._row3_frame.grid_columnconfigure(0, weight=1)
        self._row3_frame.grid_columnconfigure(1, weight=1)

        # Trending Worse - placeholder
        self._trending_frame = ctk.CTkFrame(self._row3_frame)
        self._trending_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        trending_label = ctk.CTkLabel(
            self._trending_frame,
            text="Trending Worse",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        trending_label.pack(padx=15, pady=(15, 5), anchor="w")

        self._trending_placeholder = ctk.CTkLabel(
            self._trending_frame,
            text="Loading trending data...",
            text_color="gray"
        )
        self._trending_placeholder.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.trending_chart = None

        # Low Data Models - scrollable list (not a chart)
        self._low_data_frame = ctk.CTkFrame(self._row3_frame)
        self._low_data_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        low_data_label = ctk.CTkLabel(
            self._low_data_frame,
            text="Low Data Models (<10 samples)",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        low_data_label.pack(padx=15, pady=(15, 5), anchor="w")

        self._low_data_list = ctk.CTkScrollableFrame(self._low_data_frame, height=120)
        self._low_data_list.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        self._low_data_placeholder = ctk.CTkLabel(
            self._low_data_list,
            text="Loading low data models...",
            text_color="gray",
            font=ctk.CTkFont(size=10)
        )
        self._low_data_placeholder.pack(padx=10, pady=20)

        # Drift Detection section
        self._drift_frame = ctk.CTkFrame(self.content)
        self._drift_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=5)
        self._drift_frame.grid_columnconfigure(0, weight=0, minsize=200)  # Model list
        self._drift_frame.grid_columnconfigure(1, weight=1)  # Chart area

        drift_header = ctk.CTkFrame(self._drift_frame, fg_color="transparent")
        drift_header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=15, pady=(15, 5))

        drift_label = ctk.CTkLabel(
            drift_header,
            text="Drift Detection",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        drift_label.pack(side="left")

        # Refresh button for drift section
        drift_refresh_btn = ctk.CTkButton(
            drift_header,
            text="Refresh",
            command=self._refresh_drift_data,
            width=70,
            height=24,
            font=ctk.CTkFont(size=11)
        )
        drift_refresh_btn.pack(side="right", padx=5)

        # Model list frame (left side)
        model_list_frame = ctk.CTkFrame(self._drift_frame)
        model_list_frame.grid(row=1, column=0, sticky="nsew", padx=(15, 5), pady=(5, 15))

        list_label = ctk.CTkLabel(
            model_list_frame,
            text="Model Status",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        list_label.pack(padx=10, pady=(10, 5), anchor="w")

        # Scrollable frame for model list
        self._drift_model_list = ctk.CTkScrollableFrame(model_list_frame, width=180, height=180)
        self._drift_model_list.pack(fill="both", expand=True, padx=5, pady=(0, 10))

        self._drift_model_placeholder = ctk.CTkLabel(
            self._drift_model_list,
            text="Loading drift status...",
            text_color="gray",
            font=ctk.CTkFont(size=10)
        )
        self._drift_model_placeholder.pack(padx=10, pady=20)

        # Chart area (right side) - placeholder
        self._drift_chart_frame = ctk.CTkFrame(self._drift_frame)
        self._drift_chart_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 15), pady=(5, 15))

        self._drift_chart_placeholder = ctk.CTkLabel(
            self._drift_chart_frame,
            text="Select a model to view drift chart",
            text_color="gray"
        )
        self._drift_chart_placeholder.pack(fill="both", expand=True, padx=15, pady=30)
        self.drift_chart = None
        self._selected_drift_model = None

        # Details label below chart (shows CUSUM/EWMA when model selected)
        self._drift_details_label = ctk.CTkLabel(
            self._drift_frame,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self._drift_details_label.grid(row=2, column=1, sticky="w", padx=15, pady=(0, 10))

        # ML Recommendations at bottom
        ml_frame = ctk.CTkFrame(self.content)
        ml_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=(5, 10))

        ml_header = ctk.CTkFrame(ml_frame, fg_color="transparent")
        ml_header.pack(fill="x", padx=15, pady=(15, 5))

        ml_label = ctk.CTkLabel(
            ml_header,
            text="ML Insights",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        ml_label.pack(side="left")

        # View All button
        self._ml_view_all_btn = ctk.CTkButton(
            ml_header,
            text="View All Details",
            command=self._show_ml_details_dialog,
            width=100,
            height=24,
            font=ctk.CTkFont(size=11)
        )
        self._ml_view_all_btn.pack(side="right", padx=5)

        self.ml_text = ctk.CTkTextbox(ml_frame, height=100)
        self.ml_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.ml_text.configure(state="disabled")
        self._cached_alert_models = None  # Cache for dialog
        self._update_ml_summary(None)

    def _ensure_summary_charts_initialized(self):
        """Lazily initialize summary view charts - defers matplotlib loading."""
        if self._summary_charts_initialized:
            return

        ChartWidget, ChartStyle = _ensure_chart_module()

        # Create alerts chart
        if self._alerts_placeholder and self._alerts_placeholder.winfo_exists():
            self._alerts_placeholder.destroy()
        self.alerts_chart = ChartWidget(
            self._alerts_frame,
            style=ChartStyle(figure_size=(10, 3), dpi=100)
        )
        self._chart_widgets.append(self.alerts_chart)
        self.alerts_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.alerts_chart.show_placeholder("Loading models requiring attention...")

        # Create best chart
        if self._best_placeholder and self._best_placeholder.winfo_exists():
            self._best_placeholder.destroy()
        self.best_chart = ChartWidget(
            self._best_frame,
            style=ChartStyle(figure_size=(5, 3), dpi=100)
        )
        self._chart_widgets.append(self.best_chart)
        self.best_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.best_chart.show_placeholder("Loading best models...")

        # Create recent issues chart (replaces worst chart)
        if self._recent_issues_placeholder and self._recent_issues_placeholder.winfo_exists():
            self._recent_issues_placeholder.destroy()
        self.recent_issues_chart = ChartWidget(
            self._recent_issues_frame,
            style=ChartStyle(figure_size=(5, 3), dpi=100)
        )
        self._chart_widgets.append(self.recent_issues_chart)
        self.recent_issues_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.recent_issues_chart.show_placeholder("Loading recent issues...")

        # Create trending worse chart
        if self._trending_placeholder and self._trending_placeholder.winfo_exists():
            self._trending_placeholder.destroy()
        self.trending_chart = ChartWidget(
            self._trending_frame,
            style=ChartStyle(figure_size=(5, 3), dpi=100)
        )
        self._chart_widgets.append(self.trending_chart)
        self.trending_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.trending_chart.show_placeholder("Loading trending data...")

        self._summary_charts_initialized = True
        logger.debug("Summary charts initialized (matplotlib loaded)")

    def _create_detail_view(self):
        """Create the detail view (specific model mode)."""
        # Clean up existing charts first (frees matplotlib figures)
        self._cleanup_charts()
        self._detail_charts_initialized = False

        # Reset timeline filter to show all data when switching models
        self.chart_timeline_days = 0

        # Clear existing content
        for widget in self.content.winfo_children():
            widget.destroy()

        self.content.grid_rowconfigure(0, weight=0)  # Stats row - compact
        self.content.grid_rowconfigure(1, weight=2, minsize=300)  # Main scatter chart
        self.content.grid_rowconfigure(2, weight=1, minsize=200)  # Distribution
        self.content.grid_rowconfigure(3, weight=0)  # Alerts/ML - compact

        # Model stats at top
        stats_frame = ctk.CTkFrame(self.content)
        stats_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        stats_label = ctk.CTkLabel(
            stats_frame,
            text=f"Model: {self.selected_model}",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        stats_label.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="w", columnspan=6)

        # Stats in a horizontal row
        self.detail_stat_labels = {}
        stat_names = [
            ("total_samples", "Total Samples"),
            ("anomalies", "Anomalies"),
            ("sigma_pass_rate", "Sigma Pass"),
            ("linearity_pass_rate", "Linearity Pass"),
            ("overall_pass_rate", "Overall Pass"),
            ("avg_sigma", "Avg Sigma"),
            ("threshold", "Threshold"),
            ("trend", "Trend"),
        ]

        for idx, (key, label) in enumerate(stat_names):
            stat_col = ctk.CTkFrame(stats_frame, fg_color="transparent")
            stat_col.grid(row=1, column=idx, padx=15, pady=(0, 15), sticky="w")

            ctk.CTkLabel(stat_col, text=label, text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w")
            value_label = ctk.CTkLabel(stat_col, text="--", font=ctk.CTkFont(size=14, weight="bold"))
            value_label.pack(anchor="w")
            self.detail_stat_labels[key] = value_label

        # Main SPC scatter chart - placeholder
        self._scatter_frame = ctk.CTkFrame(self.content)
        self._scatter_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Header row with title and timeline filter
        scatter_header = ctk.CTkFrame(self._scatter_frame, fg_color="transparent")
        scatter_header.pack(fill="x", padx=15, pady=(15, 5))

        chart_label = ctk.CTkLabel(
            scatter_header,
            text=f"Sigma Gradient Trend ({self.rolling_window}-Day Rolling Average)",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        chart_label.pack(side="left", anchor="w")

        # Timeline filter for zooming into specific date ranges
        timeline_frame = ctk.CTkFrame(scatter_header, fg_color="transparent")
        timeline_frame.pack(side="right", anchor="e")

        timeline_label = ctk.CTkLabel(timeline_frame, text="Chart Range:", font=ctk.CTkFont(size=11))
        timeline_label.pack(side="left", padx=(0, 5))

        self.timeline_dropdown = ctk.CTkOptionMenu(
            timeline_frame,
            values=["All Data", "Recent 7 Days", "Recent 14 Days", "Recent 30 Days", "Recent 60 Days"],
            command=self._on_timeline_change,
            width=130,
        )
        self.timeline_dropdown.set("All Data")
        self.timeline_dropdown.pack(side="left")

        self._scatter_placeholder = ctk.CTkLabel(
            self._scatter_frame,
            text="Loading trend data...",
            text_color="gray"
        )
        self._scatter_placeholder.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.scatter_chart = None

        # Distribution chart - placeholder
        self._dist_frame = ctk.CTkFrame(self.content)
        self._dist_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        dist_label = ctk.CTkLabel(
            self._dist_frame,
            text="Sigma Distribution",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        dist_label.pack(padx=15, pady=(15, 5), anchor="w")

        self._dist_placeholder = ctk.CTkLabel(
            self._dist_frame,
            text="Loading distribution...",
            text_color="gray"
        )
        self._dist_placeholder.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.dist_chart = None

        # Bottom row: Alerts and ML side by side
        bottom_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        bottom_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)

        # Alerts section
        alerts_frame = ctk.CTkFrame(bottom_frame)
        alerts_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        alerts_label = ctk.CTkLabel(
            alerts_frame,
            text="Model Alerts",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        alerts_label.pack(padx=15, pady=(15, 10), anchor="w")

        self.detail_alerts_text = ctk.CTkTextbox(alerts_frame, height=100)
        self.detail_alerts_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.detail_alerts_text.configure(state="disabled")

        # ML recommendations section
        ml_frame = ctk.CTkFrame(bottom_frame)
        ml_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        ml_label = ctk.CTkLabel(
            ml_frame,
            text="ML Recommendations",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        ml_label.pack(padx=15, pady=(15, 10), anchor="w")

        self.detail_ml_text = ctk.CTkTextbox(ml_frame, height=100)
        self.detail_ml_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.detail_ml_text.configure(state="disabled")

    def _ensure_detail_charts_initialized(self):
        """Lazily initialize detail view charts - defers matplotlib loading."""
        if self._detail_charts_initialized:
            return

        ChartWidget, ChartStyle = _ensure_chart_module()

        # Create scatter chart
        if self._scatter_placeholder:
            self._scatter_placeholder.destroy()
        self.scatter_chart = ChartWidget(
            self._scatter_frame,
            style=ChartStyle(figure_size=(10, 4), dpi=100)
        )
        self._chart_widgets.append(self.scatter_chart)
        self.scatter_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.scatter_chart.show_placeholder("Loading trend data...")

        # Create distribution chart
        if self._dist_placeholder:
            self._dist_placeholder.destroy()
        self.dist_chart = ChartWidget(
            self._dist_frame,
            style=ChartStyle(figure_size=(10, 2.5), dpi=100)
        )
        self._chart_widgets.append(self.dist_chart)
        self.dist_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.dist_chart.show_placeholder("Loading distribution...")

        self._detail_charts_initialized = True
        logger.debug("Detail charts initialized (matplotlib loaded)")

    def _bind_mousewheel_scroll(self, combobox):
        """Bind mousewheel events to CTkComboBox dropdown."""
        def on_mousewheel_closed(event):
            values = combobox.cget("values")
            if not values:
                return "break"

            current = combobox.get()
            try:
                current_idx = list(values).index(current)
            except ValueError:
                current_idx = 0

            if hasattr(event, 'delta'):
                direction = -1 if event.delta > 0 else 1
            else:
                direction = -1 if event.num == 4 else 1

            new_idx = current_idx + direction
            if 0 <= new_idx < len(values):
                combobox.set(values[new_idx])
                command = combobox.cget("command")
                if command:
                    command(values[new_idx])
            return "break"

        combobox.bind("<MouseWheel>", on_mousewheel_closed)
        combobox.bind("<Button-4>", on_mousewheel_closed)
        combobox.bind("<Button-5>", on_mousewheel_closed)

        if hasattr(combobox, '_open_dropdown_menu'):
            original_open = combobox._open_dropdown_menu

            def patched_open(*args, **kwargs):
                result = original_open(*args, **kwargs)
                combobox.after(10, lambda: self._enable_dropdown_scroll(combobox))
                return result

            combobox._open_dropdown_menu = patched_open

    def _enable_dropdown_scroll(self, combobox):
        """Enable mousewheel scrolling on the open dropdown."""
        try:
            if hasattr(combobox, '_dropdown_menu') and combobox._dropdown_menu:
                dropdown = combobox._dropdown_menu
                canvas = None

                if hasattr(dropdown, '_scrollable_frame'):
                    sf = dropdown._scrollable_frame
                    if hasattr(sf, '_parent_canvas'):
                        canvas = sf._parent_canvas
                    elif hasattr(sf, '_canvas'):
                        canvas = sf._canvas
                elif hasattr(dropdown, '_canvas'):
                    canvas = dropdown._canvas

                if not canvas:
                    def find_canvas(widget):
                        for child in widget.winfo_children():
                            child_type = str(type(child)).lower()
                            if 'canvas' in child_type:
                                return child
                            found = find_canvas(child)
                            if found:
                                return found
                        return None
                    canvas = find_canvas(dropdown)

                if canvas:
                    def scroll_dropdown(event):
                        try:
                            delta = -1 * (event.delta // 120) if event.delta else (-1 if event.num == 4 else 1)
                            canvas.yview_scroll(delta, "units")
                        except Exception:
                            pass
                        return "break"

                    dropdown.bind_all("<MouseWheel>", scroll_dropdown)
                    dropdown.bind_all("<Button-4>", scroll_dropdown)
                    dropdown.bind_all("<Button-5>", scroll_dropdown)

                    def on_destroy(event):
                        try:
                            dropdown.unbind_all("<MouseWheel>")
                            dropdown.unbind_all("<Button-4>")
                            dropdown.unbind_all("<Button-5>")
                        except Exception:
                            pass

                    dropdown.bind("<Destroy>", on_destroy)
        except Exception as e:
            logger.debug(f"Could not enable dropdown scroll: {e}")

    def _on_model_change(self, model: str):
        """Handle model selection change."""
        self.selected_model = model
        logger.debug(f"Model changed to: {model}")

        # Switch view mode
        if model == "All Models":
            self._create_summary_view()
        else:
            self._create_detail_view()

        self._refresh_data()

    def _on_date_change(self, date_range: str):
        """Handle date range change."""
        days_map = {
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "Last Year": 365,
            "All Time": 3650,
        }
        self.selected_days = days_map.get(date_range, 3650)
        logger.debug(f"Date range changed to: {date_range} ({self.selected_days} days)")
        self._refresh_data()

    def _on_rolling_change(self, rolling: str):
        """Handle rolling average window change."""
        rolling_map = {
            "7 Days": 7,
            "14 Days": 14,
            "30 Days": 30,
            "60 Days": 60,
        }
        self.rolling_window = rolling_map.get(rolling, 30)
        logger.debug(f"Rolling window changed to: {rolling} ({self.rolling_window} days)")
        self._refresh_data()

    def _on_timeline_change(self, timeline: str):
        """Handle chart timeline filter change (zooms into the scatter chart)."""
        timeline_map = {
            "All Data": 0,
            "Recent 7 Days": 7,
            "Recent 14 Days": 14,
            "Recent 30 Days": 30,
            "Recent 60 Days": 60,
        }
        self.chart_timeline_days = timeline_map.get(timeline, 0)
        logger.debug(f"Chart timeline changed to: {timeline} ({self.chart_timeline_days} days)")
        # Re-render charts with filtered data (no need to reload from DB)
        if self.model_trend_data:
            self._update_scatter_chart_with_filter()

    def _update_scatter_chart_with_filter(self):
        """Update the scatter chart with current timeline filter applied."""
        if not self.model_trend_data or not self.model_trend_data.get("data_points"):
            return

        if not self.scatter_chart:
            return

        data_points = self.model_trend_data["data_points"]
        threshold = self.model_trend_data.get("threshold")

        # Apply timeline filter - use RELATIVE to data, not absolute calendar dates
        # This way "Last 7 Days" means "most recent 7 days of data that exists"
        if self.chart_timeline_days > 0 and data_points:
            # Find the most recent date in the data
            def get_date(d):
                return d["date"] if isinstance(d["date"], datetime) else datetime.strptime(str(d["date"])[:10], "%Y-%m-%d")

            most_recent = max(get_date(d) for d in data_points)
            cutoff_date = most_recent - timedelta(days=self.chart_timeline_days)
            filtered_points = [d for d in data_points if get_date(d) >= cutoff_date]
        else:
            filtered_points = data_points

        # Exclude anomalies from chart - they are shown as a count in the stats
        # Anomalies (trim failures with linear slope) would skew the visual trend
        normal_points = [d for d in filtered_points if not d.get("is_anomaly", False)]

        if not normal_points:
            anomaly_count = len([d for d in filtered_points if d.get("is_anomaly", False)])
            if anomaly_count > 0:
                self.scatter_chart.show_placeholder(f"All {anomaly_count} samples are anomalies\n(excluded from chart)")
            else:
                self.scatter_chart.show_placeholder(f"No data in selected range (last {self.chart_timeline_days} days)")
            return

        # Extract values for plotting (normal samples only) - include year in date format
        dates = [d["date"].strftime("%m/%d/%y") if hasattr(d["date"], 'strftime') else str(d["date"])[:8] for d in normal_points]
        sigma_values = [d["sigma_gradient"] for d in normal_points if d["sigma_gradient"] is not None]
        pass_flags = [d.get("sigma_pass", False) for d in normal_points]

        # Calculate rolling average for filtered data (normal samples only)
        rolling_vals = None
        window = min(self.rolling_window, len(sigma_values))
        if window > 1 and sigma_values:
            rolling_vals = []
            for i in range(len(sigma_values)):
                start = max(0, i - window + 1)
                window_vals = sigma_values[start:i+1]
                rolling_vals.append(np.mean(window_vals))

        # Determine title suffix based on filter
        filter_suffix = f" (Last {self.chart_timeline_days} Days)" if self.chart_timeline_days > 0 else ""

        # Plot without anomaly_flags since we've already filtered them out
        self.scatter_chart.plot_sigma_scatter(
            dates=dates,
            sigma_values=sigma_values,
            pass_flags=pass_flags,
            threshold=threshold,
            rolling_avg=rolling_vals,
            title=f"Sigma Gradient Trend - {self.selected_model}{filter_suffix}",
            ylabel="Sigma Gradient",
        )

    def _refresh_data(self):
        """Refresh data from database."""
        self.status_label.configure(text="Loading...")
        get_thread_manager().start_thread(target=self._load_data, name="trends-load-data")

    def _load_data(self):
        """Load data in background thread."""
        try:
            db = get_database()

            if self.selected_model == "All Models":
                # Summary mode
                self._load_summary_data(db)
            else:
                # Detail mode
                self._load_detail_data(db)

        except Exception as e:
            logger.error(f"Failed to load trend data: {e}")
            self.after(0, lambda: self._show_error(str(e)))

    def _load_summary_data(self, db):
        """Load data for summary mode."""
        # Get active models summary
        active_models = db.get_active_models_summary(
            days_back=self.selected_days,
            min_samples=5
        )

        # Get models requiring attention (filter to 20+ samples)
        alert_models = db.get_models_requiring_attention(
            days_back=self.selected_days,
            min_samples=20,  # Increased to filter out low-data models
            pass_rate_threshold=80.0,
            trend_threshold=10.0,
            rolling_days=self.rolling_window
        )

        # Get trending worse models
        trending_worse = db.get_trending_worse_models(
            days_back=self.selected_days,
            min_samples=20,
            trend_threshold=10.0,
            rolling_days=self.rolling_window
        )

        # Update model dropdown with active models
        model_names = ["All Models"] + [m["model"] for m in active_models]

        # Update UI on main thread
        self.after(0, lambda: self._update_summary_display(
            active_models, alert_models, model_names, trending_worse
        ))

    def _load_detail_data(self, db):
        """Load data for detail mode."""
        # Get detailed trend data for this model
        trend_data = db.get_model_trend_data(
            model=self.selected_model,
            days_back=self.selected_days,
            rolling_window=self.rolling_window
        )

        # Get alerts for this model
        alert_models = db.get_models_requiring_attention(
            days_back=self.selected_days,
            min_samples=5,
            pass_rate_threshold=80.0,
            trend_threshold=10.0,
            rolling_days=self.rolling_window
        )
        model_alerts = next((a for a in alert_models if a["model"] == self.selected_model), None)

        # Get ML recommendations
        ml_recommendations = self._get_ml_recommendations(trend_data)

        # Update model dropdown and get model stats for pass rate
        active_models = db.get_active_models_summary(self.selected_days, 5)
        model_names = ["All Models"] + [m["model"] for m in active_models]

        # Get the model's analysis-level stats (for consistent pass rate with alerts)
        model_stats = next((m for m in active_models if m["model"] == self.selected_model), None)

        # Update UI on main thread
        self.after(0, lambda: self._update_detail_display(
            trend_data, model_alerts, ml_recommendations, model_names, model_stats
        ))

    def _get_ml_recommendations(self, trend_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get ML recommendations for current model using per-model ML system."""
        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.ml import MLManager

            db = get_database()
            ml_manager = MLManager(db)

            # Try to load trained state
            ml_manager.load_all()

            # Get threshold from per-model optimizer
            threshold = ml_manager.get_threshold(self.selected_model)

            if threshold is not None:
                optimizer = ml_manager.threshold_optimizers.get(self.selected_model)
                profiler = ml_manager.profilers.get(self.selected_model)
                detector = ml_manager.drift_detectors.get(self.selected_model)

                result = {
                    "recommended_threshold": threshold,
                    "confidence": optimizer.confidence if optimizer else 0.5,
                    "method": optimizer.method if optimizer else "formula",
                    "basis": f"{optimizer.n_samples if optimizer else 0} samples",
                }

                # Add drift info if available
                if detector and detector.has_baseline:
                    result["drift_status"] = "Drifting" if detector.is_drifting else "Stable"
                    result["drift_direction"] = detector.drift_direction.value if detector.drift_direction else None

                # Add profile insights if available
                if profiler and profiler.profile:
                    result["pass_rate"] = profiler.profile.pass_rate
                    result["difficulty"] = profiler.profile.difficulty_score
                    result["insights"] = profiler.get_insights()[:3]  # Top 3 insights

                return result

            # No fallback - new per-model system is the only source
            # Train models in Settings to get ML recommendations

        except Exception as e:
            logger.debug(f"ML recommendations not available: {e}")

        return None

    def _update_summary_display(
        self,
        active_models: List[Dict[str, Any]],
        alert_models: List[Dict[str, Any]],
        model_names: List[str],
        trending_worse: Optional[List[Dict[str, Any]]] = None
    ):
        """Update summary display with loaded data."""
        # Ensure charts are initialized before use (lazy matplotlib loading)
        self._ensure_summary_charts_initialized()

        # Update model dropdown
        current_model = self.model_dropdown.get()
        self.model_dropdown.configure(values=model_names)
        if current_model in model_names:
            self.model_dropdown.set(current_model)
        else:
            self.model_dropdown.set("All Models")

        self.active_models_data = active_models

        # Filter models by sample count
        models_with_data = [m for m in active_models if m["total"] >= 20]  # 20+ samples
        low_data_models = [m for m in active_models if m["total"] < 10]  # <10 samples

        if not active_models:
            self._reset_summary_stats()
            self.alerts_chart.show_placeholder("No active models in selected period")
            self.best_chart.show_placeholder("No data")
            self.recent_issues_chart.show_placeholder("No data")
            self.trending_chart.show_placeholder("No data")
            self._update_low_data_list([])
            self.status_label.configure(text="No data")
            return

        # Calculate summary stats
        total_models = len(active_models)
        total_samples = sum(m["total"] for m in active_models)
        avg_pass_rate = sum(m["pass_rate"] for m in active_models) / total_models if total_models > 0 else 0
        avg_sigma_rate = sum(m.get("sigma_pass_rate", 0) for m in active_models) / total_models if total_models > 0 else 0
        avg_linearity_rate = sum(m.get("linearity_pass_rate", 0) for m in active_models) / total_models if total_models > 0 else 0
        models_at_risk = len(alert_models)

        # Best and worst models
        sorted_by_rate = sorted(active_models, key=lambda x: x["pass_rate"], reverse=True)
        best_model = sorted_by_rate[0]["model"] if sorted_by_rate else "--"
        worst_model = sorted_by_rate[-1]["model"] if sorted_by_rate else "--"
        best_rate = sorted_by_rate[0]["pass_rate"] if sorted_by_rate else 0
        worst_rate = sorted_by_rate[-1]["pass_rate"] if sorted_by_rate else 0

        # Update stat labels
        self.summary_stat_labels["active_models"].configure(text=str(total_models))
        self.summary_stat_labels["total_samples"].configure(text=f"{total_samples:,}")
        self.summary_stat_labels["avg_pass_rate"].configure(
            text=f"{avg_pass_rate:.1f}%",
            text_color="#27ae60" if avg_pass_rate >= 90 else "#f39c12" if avg_pass_rate >= 80 else "#e74c3c"
        )
        self.summary_stat_labels["avg_sigma_rate"].configure(
            text=f"{avg_sigma_rate:.1f}%",
            text_color="#27ae60" if avg_sigma_rate >= 90 else "#f39c12" if avg_sigma_rate >= 80 else "#e74c3c"
        )
        self.summary_stat_labels["avg_linearity_rate"].configure(
            text=f"{avg_linearity_rate:.1f}%",
            text_color="#27ae60" if avg_linearity_rate >= 90 else "#f39c12" if avg_linearity_rate >= 80 else "#e74c3c"
        )
        self.summary_stat_labels["models_at_risk"].configure(
            text=str(models_at_risk),
            text_color="#e74c3c" if models_at_risk > 0 else "#27ae60"
        )
        self.summary_stat_labels["best_model"].configure(
            text=f"{best_model} ({best_rate:.0f}%)",
            text_color="#27ae60"
        )
        self.summary_stat_labels["worst_model"].configure(
            text=f"{worst_model} ({worst_rate:.0f}%)",
            text_color="#e74c3c" if worst_rate < 80 else "#f39c12"
        )

        # Update alerts chart
        if alert_models:
            alert_model_names = [a["model"] for a in alert_models[:10]]
            self.alerts_chart.plot_alert_summary(
                models=alert_model_names,
                alerts=alert_models[:10],
                title=f"Models Requiring Attention ({len(alert_models)} total)"
            )
        else:
            self.alerts_chart.show_placeholder("All models performing well - no alerts!")

        # Update best models chart (filtered to 20+ samples)
        sorted_with_data = sorted(models_with_data, key=lambda x: x["pass_rate"], reverse=True)
        best_5 = sorted_with_data[:5]
        if best_5:
            self.best_chart.plot_pass_rate_bars(
                models=[m["model"] for m in best_5],
                pass_rates=[m["pass_rate"] for m in best_5],
                sample_counts=[m["total"] for m in best_5],
                title="Top 5 Performing Models",
                highlight_threshold=80.0
            )
        else:
            self.best_chart.show_placeholder("No models with 20+ samples")

        # Recent Issues: models with data in last 30 days AND pass_rate < 80% (filtered to 20+ samples)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_issues = [
            m for m in models_with_data
            if m.get("last_date") and m["last_date"] >= recent_cutoff and m["pass_rate"] < 80
        ]
        recent_issues = sorted(recent_issues, key=lambda x: x["pass_rate"])[:5]  # Worst first

        if recent_issues:
            self.recent_issues_chart.plot_pass_rate_bars(
                models=[m["model"] for m in recent_issues],
                pass_rates=[m["pass_rate"] for m in recent_issues],
                sample_counts=[m["total"] for m in recent_issues],
                title="Recent Issues (Last 30 Days)",
                highlight_threshold=80.0
            )
        else:
            self.recent_issues_chart.show_placeholder("No recent issues - great!")

        # Trending Worse section
        if trending_worse and len(trending_worse) > 0:
            top_trending = trending_worse[:5]
            self.trending_chart.plot_trending_worse(
                models=[m["model"] for m in top_trending],
                pass_rates=[m["pass_rate"] for m in top_trending],
                declines=[m["decline"] for m in top_trending],
                sample_counts=[m["total_samples"] for m in top_trending],
                title="Trending Worse (>10% decline)"
            )
        else:
            self.trending_chart.show_placeholder("No models trending worse - stable!")

        # Low Data Models section
        self._update_low_data_list(low_data_models)

        # Update ML summary
        self._update_ml_summary(alert_models)

        # Load drift detection data
        self._refresh_drift_data()

        # Update status
        self.status_label.configure(text=f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    def _update_low_data_list(self, low_data_models: List[Dict[str, Any]]):
        """Update the low data models list section."""
        # Clear existing items
        for widget in self._low_data_list.winfo_children():
            widget.destroy()

        if not low_data_models:
            placeholder = ctk.CTkLabel(
                self._low_data_list,
                text="No models with insufficient data",
                text_color="gray",
                font=ctk.CTkFont(size=10)
            )
            placeholder.pack(padx=10, pady=20)
            return

        # Sort by sample count (fewest first)
        sorted_models = sorted(low_data_models, key=lambda x: x["total"])

        for model_data in sorted_models[:10]:  # Show max 10
            model_name = model_data["model"]
            samples = model_data["total"]
            last_date = model_data.get("last_date")

            # Format: "Model: X samples (last: date)"
            date_str = last_date.strftime("%m/%d") if last_date else "N/A"
            text = f"{model_name}: {samples} samples (last: {date_str})"

            label = ctk.CTkLabel(
                self._low_data_list,
                text=text,
                font=ctk.CTkFont(size=10),
                text_color="#f39c12"  # Orange for warning
            )
            label.pack(padx=10, pady=2, anchor="w")

    def _update_detail_display(
        self,
        trend_data: Dict[str, Any],
        model_alerts: Optional[Dict[str, Any]],
        ml_recommendations: Optional[Dict[str, Any]],
        model_names: List[str],
        model_stats: Optional[Dict[str, Any]] = None
    ):
        """Update detail display with loaded data."""
        # Ensure charts are initialized before use (lazy matplotlib loading)
        self._ensure_detail_charts_initialized()

        # Update model dropdown
        current_model = self.model_dropdown.get()
        self.model_dropdown.configure(values=model_names)
        if current_model in model_names:
            self.model_dropdown.set(current_model)

        self.model_trend_data = trend_data

        if not trend_data or not trend_data.get("data_points"):
            self._reset_detail_stats()
            self.scatter_chart.show_placeholder("No data for this model in selected period")
            self.dist_chart.show_placeholder("No distribution data")
            self.status_label.configure(text="No data")
            return

        # Extract data
        data_points = trend_data["data_points"]
        rolling_averages = trend_data.get("rolling_averages", [])
        threshold = trend_data.get("threshold")
        total_samples = len(data_points)

        # Count and filter anomalies from statistics calculations
        # Anomalies (trim failures with linear slope) would skew averages
        anomaly_count = sum(1 for d in data_points if d.get("is_anomaly", False))
        normal_points = [d for d in data_points if not d.get("is_anomaly", False)]
        normal_sample_count = len(normal_points)

        # Calculate stats from NORMAL samples only (excludes anomalies)
        sigma_values = [d["sigma_gradient"] for d in normal_points if d["sigma_gradient"] is not None]

        # Sigma pass rate (track-level: did sigma gradient pass?) - from normal samples
        sigma_pass_count = sum(1 for d in normal_points if d.get("sigma_pass", False))
        sigma_pass_rate = (sigma_pass_count / normal_sample_count * 100) if normal_sample_count > 0 else 0

        # Overall pass rate - use model_stats from get_active_models_summary for consistency with alerts
        # This counts analysis-level pass (both sigma AND linearity must pass for all tracks)
        if model_stats:
            overall_pass_rate = model_stats.get("pass_rate", 0)
            linearity_pass_rate = model_stats.get("linearity_pass_rate", 0)
            total_analyses = model_stats.get("total", total_samples)
        else:
            # Fallback: count from track data (may differ from analysis-level count)
            overall_pass_count = sum(1 for d in data_points if d.get("status") == "PASS")
            overall_pass_rate = (overall_pass_count / total_samples * 100) if total_samples > 0 else 0
            linearity_pass_rate = 0  # Can't calculate without model_stats
            total_analyses = total_samples

        avg_sigma = np.mean(sigma_values) if sigma_values else 0
        std_sigma = np.std(sigma_values, ddof=1) if len(sigma_values) > 1 else 0

        # Trend direction
        if len(sigma_values) >= 6:
            first_half = np.mean(sigma_values[:len(sigma_values)//2])
            second_half = np.mean(sigma_values[len(sigma_values)//2:])
            if second_half > first_half * 1.1:
                trend = "Increasing"
                trend_color = "#e74c3c"
            elif second_half < first_half * 0.9:
                trend = "Decreasing"
                trend_color = "#27ae60"
            else:
                trend = "Stable"
                trend_color = "#3498db"
        else:
            trend = "Insufficient Data"
            trend_color = "gray"

        # Status based on overall pass rate
        if overall_pass_rate >= 95:
            status = "Excellent"
            status_color = "#27ae60"
        elif overall_pass_rate >= 80:
            status = "Good"
            status_color = "#3498db"
        elif overall_pass_rate >= 70:
            status = "Warning"
            status_color = "#f39c12"
        else:
            status = "Critical"
            status_color = "#e74c3c"

        # Update stat labels
        # Use analysis count from model_stats if available, otherwise track count
        display_count = total_analyses if model_stats else total_samples
        self.detail_stat_labels["total_samples"].configure(text=f"{display_count:,}")
        self.detail_stat_labels["anomalies"].configure(
            text=f"{anomaly_count}",
            text_color="#9b59b6" if anomaly_count > 0 else "gray"
        )
        self.detail_stat_labels["sigma_pass_rate"].configure(
            text=f"{sigma_pass_rate:.1f}%",
            text_color="#27ae60" if sigma_pass_rate >= 90 else "#f39c12" if sigma_pass_rate >= 80 else "#e74c3c"
        )
        self.detail_stat_labels["linearity_pass_rate"].configure(
            text=f"{linearity_pass_rate:.1f}%",
            text_color="#27ae60" if linearity_pass_rate >= 90 else "#f39c12" if linearity_pass_rate >= 80 else "#e74c3c"
        )
        self.detail_stat_labels["overall_pass_rate"].configure(
            text=f"{overall_pass_rate:.1f}%",
            text_color="#27ae60" if overall_pass_rate >= 90 else "#f39c12" if overall_pass_rate >= 80 else "#e74c3c"
        )
        self.detail_stat_labels["avg_sigma"].configure(text=f"{avg_sigma:.6f}")
        self.detail_stat_labels["threshold"].configure(
            text=f"{threshold:.6f}" if threshold else "--"
        )
        self.detail_stat_labels["trend"].configure(text=trend, text_color=trend_color)

        # Update scatter chart using the filter method (applies timeline filter)
        self._update_scatter_chart_with_filter()

        # Update distribution - only show histogram for 20+ samples
        # For small sample sizes, the trend chart above shows everything needed
        if len(sigma_values) >= 20:
            self.dist_chart.master.grid()  # Show the distribution frame
            self.dist_chart.plot_histogram(
                values=sigma_values,
                bins=min(30, len(sigma_values) // 3 + 1),
                title="Sigma Distribution",
                xlabel="Sigma Gradient",
                spec_limit=threshold
            )
        else:
            # Hide distribution chart for small sample sizes
            self.dist_chart.master.grid_remove()

        # Update alerts text
        self._update_detail_alerts(model_alerts)

        # Update ML text
        self._update_detail_ml(ml_recommendations)

        # Update status
        self.status_label.configure(text=f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    def _reset_summary_stats(self):
        """Reset summary statistics to default values."""
        for key in self.summary_stat_labels:
            self.summary_stat_labels[key].configure(text="--", text_color="white")

    def _reset_detail_stats(self):
        """Reset detail statistics to default values."""
        for key in self.detail_stat_labels:
            self.detail_stat_labels[key].configure(text="--", text_color="white")

    def _update_ml_summary(self, alert_models: Optional[List[Dict[str, Any]]]):
        """Update ML summary text for all models view with ML insights."""
        # Cache for the details dialog
        self._cached_alert_models = alert_models
        self._cached_ml_insights = self._get_ml_summary_insights()

        self.ml_text.configure(state="normal")
        self.ml_text.delete("1.0", "end")

        ml_insights = self._cached_ml_insights
        has_content = False

        if ml_insights:
            has_content = True
            trained = ml_insights.get("trained_models", 0)

            # Show trained models count first
            if trained > 0:
                self.ml_text.insert("end", f"ML Status: {trained} models trained  |  ")

            # Show drift status summary inline
            drifting = ml_insights.get("drifting_models", [])
            if drifting:
                self.ml_text.insert("end", f"Drift: {len(drifting)} model(s)\n\n")
            else:
                self.ml_text.insert("end", "Drift: None detected\n\n")

        # Show alert summary if any - show more items now
        if alert_models:
            has_content = True
            self.ml_text.insert("end", f"Models Requiring Attention ({len(alert_models)}):\n")
            for alert_model in alert_models[:8]:  # Show top 8 in summary
                model_name = alert_model.get("model", "Unknown")
                pass_rate = alert_model.get("pass_rate", 0)
                alerts = alert_model.get("alerts", [])

                # Get alert types for this model
                alert_types = [al["type"] for al in alerts]

                if "LOW_PASS_RATE" in alert_types:
                    self.ml_text.insert("end", f"  ⚠ {model_name}: {pass_rate:.1f}% pass rate\n")
                elif "TRENDING_WORSE" in alert_types:
                    self.ml_text.insert("end", f"  ↓ {model_name}: trending worse\n")
                elif "HIGH_VARIANCE" in alert_types:
                    self.ml_text.insert("end", f"  ~ {model_name}: high variance\n")
                else:
                    self.ml_text.insert("end", f"  • {model_name}\n")

            if len(alert_models) > 8:
                self.ml_text.insert("end", f"  ... click 'View All Details' for {len(alert_models) - 8} more\n")

        if not has_content:
            self.ml_text.insert("end", "All models performing well.\n")
            self.ml_text.insert("end", "Train models in Settings for ML insights.")

        self.ml_text.configure(state="disabled")

    def _get_ml_summary_insights(self) -> Optional[Dict[str, Any]]:
        """Get ML insights for summary view."""
        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.ml import MLManager

            db = get_database()
            ml_manager = MLManager(db)
            ml_manager.load_all()

            if not ml_manager.profilers:
                return None

            result = {
                "trained_models": len(ml_manager.profilers),
                "difficulty_ranking": [],
                "drifting_models": [],
            }

            # Get difficulty ranking (hardest first)
            for model, profiler in ml_manager.profilers.items():
                if profiler.profile:
                    result["difficulty_ranking"].append(
                        (model, profiler.profile.difficulty_score)
                    )
            result["difficulty_ranking"].sort(key=lambda x: -x[1])  # Descending

            # Get drifting models
            for model, detector in ml_manager.drift_detectors.items():
                if detector.has_baseline and detector.is_drifting:
                    direction = detector.drift_direction.value if detector.drift_direction else "unknown"
                    result["drifting_models"].append((model, direction))

            return result

        except Exception as e:
            logger.debug(f"Could not get ML summary insights: {e}")
            return None

    def _show_ml_details_dialog(self):
        """Show a dialog with full ML insights details."""
        from tkinter import Toplevel

        dialog = ctk.CTkToplevel(self)
        dialog.title("ML Insights - Full Details")
        dialog.geometry("700x600")
        dialog.transient(self)
        dialog.grab_set()

        # Main container with scrollable text
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        title_label = ctk.CTkLabel(
            main_frame,
            text="ML Insights - Full Details",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(pady=(10, 15))

        # Scrollable text area
        text_widget = ctk.CTkTextbox(main_frame, width=650, height=450)
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)

        # Fetch fresh data instead of using potentially stale cache
        ml_insights = self._get_ml_summary_insights()
        alert_models = getattr(self, '_cached_alert_models', None)

        # Section 1: ML Status Overview
        text_widget.insert("end", "=" * 60 + "\n")
        text_widget.insert("end", "ML SYSTEM STATUS\n")
        text_widget.insert("end", "=" * 60 + "\n\n")

        if ml_insights:
            trained = ml_insights.get("trained_models", 0)
            text_widget.insert("end", f"Trained Models: {trained}\n\n")

            # Drift status
            drifting = ml_insights.get("drifting_models", [])
            if drifting:
                text_widget.insert("end", f"DRIFT DETECTED in {len(drifting)} model(s):\n")
                for model, direction in drifting:
                    text_widget.insert("end", f"  ⚠ {model}: drifting {direction}\n")
                text_widget.insert("end", "\n")
            else:
                text_widget.insert("end", "Drift Status: All models stable\n\n")

            # Difficulty ranking
            difficulty = ml_insights.get("difficulty_ranking", [])
            if difficulty:
                text_widget.insert("end", "Model Difficulty Ranking (hardest first):\n")
                for rank, (model, score) in enumerate(difficulty, 1):
                    label = "Easy" if score < 0.3 else "Medium" if score < 0.6 else "Hard"
                    bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                    text_widget.insert("end", f"  {rank:2}. {model:<15} {bar} {label} ({score:.2f})\n")
                text_widget.insert("end", "\n")
        else:
            text_widget.insert("end", "No ML models trained.\n")
            text_widget.insert("end", "Go to Settings → Train Models to enable ML insights.\n\n")

        # Section 2: Models Requiring Attention
        text_widget.insert("end", "=" * 60 + "\n")
        text_widget.insert("end", "MODELS REQUIRING ATTENTION\n")
        text_widget.insert("end", "=" * 60 + "\n\n")

        if alert_models:
            text_widget.insert("end", f"Total: {len(alert_models)} models with alerts\n\n")

            for i, alert_model in enumerate(alert_models, 1):
                model_name = alert_model.get("model", "Unknown")
                pass_rate = alert_model.get("pass_rate", 0)
                total = alert_model.get("total", 0)
                alerts = alert_model.get("alerts", [])

                text_widget.insert("end", f"{i}. {model_name}\n")
                text_widget.insert("end", f"   Pass Rate: {pass_rate:.1f}%  |  Samples: {total}\n")

                if alerts:
                    text_widget.insert("end", "   Alerts:\n")
                    for alert in alerts:
                        alert_type = alert.get("type", "Unknown")
                        severity = alert.get("severity", "Medium")
                        message = alert.get("message", "")
                        icon = "!!" if severity == "High" else "!" if severity == "Medium" else "·"
                        text_widget.insert("end", f"     {icon} [{severity}] {alert_type}\n")
                        if message:
                            text_widget.insert("end", f"        {message}\n")
                text_widget.insert("end", "\n")
        else:
            text_widget.insert("end", "No models currently require attention.\n")
            text_widget.insert("end", "All models are performing within acceptable parameters.\n")

        # Section 3: Quick Tips
        text_widget.insert("end", "\n" + "=" * 60 + "\n")
        text_widget.insert("end", "QUICK TIPS\n")
        text_widget.insert("end", "=" * 60 + "\n\n")
        text_widget.insert("end", "• LOW_PASS_RATE: Model has <50% pass rate - investigate process\n")
        text_widget.insert("end", "• TRENDING_WORSE: Quality declining over time - check for drift\n")
        text_widget.insert("end", "• HIGH_VARIANCE: Inconsistent results - check equipment calibration\n")
        text_widget.insert("end", "• DRIFT (up): Sigma values increasing - process degrading\n")
        text_widget.insert("end", "• DRIFT (down): Sigma values decreasing - process improving\n")

        text_widget.configure(state="disabled")

        # Close button
        close_btn = ctk.CTkButton(
            main_frame,
            text="Close",
            command=dialog.destroy,
            width=100
        )
        close_btn.pack(pady=(10, 10))

        # Center dialog on parent
        dialog.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def _update_detail_alerts(self, model_alerts: Optional[Dict[str, Any]]):
        """Update detail alerts text."""
        self.detail_alerts_text.configure(state="normal")
        self.detail_alerts_text.delete("1.0", "end")

        if not model_alerts:
            self.detail_alerts_text.insert("end", "No alerts for this model.\n\n")
            self.detail_alerts_text.insert("end", "Model is performing within acceptable parameters.")
        else:
            alerts = model_alerts.get("alerts", [])
            self.detail_alerts_text.insert("end", f"{len(alerts)} Alert(s) Detected:\n\n")

            for alert in alerts:
                severity = alert.get("severity", "Medium")
                icon = "!!" if severity == "High" else "!"
                self.detail_alerts_text.insert("end", f"{icon} [{severity}] {alert.get('message', 'Unknown')}\n")

        self.detail_alerts_text.configure(state="disabled")

    def _update_detail_ml(self, ml_recommendations: Optional[Dict[str, Any]]):
        """Update detail ML recommendations text."""
        self.detail_ml_text.configure(state="normal")
        self.detail_ml_text.delete("1.0", "end")

        if ml_recommendations is None:
            self.detail_ml_text.insert("end", "ML recommendations not available.\n\n")
            self.detail_ml_text.insert("end", "Train models in Settings to see recommendations.")
        else:
            threshold = ml_recommendations.get("recommended_threshold", 0)
            confidence = ml_recommendations.get("confidence", 0)
            basis = ml_recommendations.get("basis", "historical data")
            method = ml_recommendations.get("method", "unknown")
            is_legacy = ml_recommendations.get("legacy", False)

            # Threshold section
            self.detail_ml_text.insert("end", f"Recommended Threshold: {threshold:.6f}\n")
            self.detail_ml_text.insert("end", f"  Method: {method}, Confidence: {confidence:.0%}\n")
            self.detail_ml_text.insert("end", f"  Based on: {basis}\n")

            if is_legacy:
                self.detail_ml_text.insert("end", "  (Using legacy optimizer - retrain for per-model ML)\n")

            # Drift status section (new ML system only)
            drift_status = ml_recommendations.get("drift_status")
            if drift_status:
                drift_direction = ml_recommendations.get("drift_direction", "")
                drift_text = f"{drift_status}"
                if drift_direction:
                    drift_text += f" ({drift_direction})"
                status_indicator = "⚠" if drift_status == "Drifting" else "✓"
                self.detail_ml_text.insert("end", f"\nDrift Status: {status_indicator} {drift_text}\n")

            # Profile insights section (new ML system only)
            pass_rate = ml_recommendations.get("pass_rate")
            difficulty = ml_recommendations.get("difficulty")
            if pass_rate is not None or difficulty is not None:
                self.detail_ml_text.insert("end", "\nModel Profile:\n")
                if pass_rate is not None:
                    self.detail_ml_text.insert("end", f"  Pass Rate: {pass_rate:.1f}%\n")
                if difficulty is not None:
                    diff_label = "Easy" if difficulty < 0.3 else "Medium" if difficulty < 0.6 else "Hard"
                    self.detail_ml_text.insert("end", f"  Difficulty: {diff_label} ({difficulty:.2f})\n")

            # Top insights (new ML system only)
            insights = ml_recommendations.get("insights", [])
            if insights:
                self.detail_ml_text.insert("end", "\nInsights:\n")
                for insight in insights[:3]:
                    self.detail_ml_text.insert("end", f"  • {insight}\n")

        self.detail_ml_text.configure(state="disabled")

    def _show_error(self, error: str):
        """Show error state."""
        self.status_label.configure(text="Error loading data")
        if hasattr(self, 'alerts_chart'):
            self.alerts_chart.show_placeholder(f"Error: {error}")
        if hasattr(self, 'scatter_chart'):
            self.scatter_chart.show_placeholder(f"Error: {error}")

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Trends page shown")
        # Recreate view if charts were cleaned up
        if not self._summary_charts_initialized and not self._detail_charts_initialized:
            if self.selected_model == "All Models":
                self._create_summary_view()
            else:
                self._create_detail_view()
        self._refresh_data()

    def on_hide(self):
        """Called when page becomes hidden - cleanup to free memory."""
        # Cleanup charts (frees matplotlib figures)
        self._cleanup_charts()

    def _refresh_drift_data(self):
        """Refresh drift detection data."""
        get_thread_manager().start_thread(target=self._load_drift_data, name="trends-load-drift")

    def _load_drift_data(self):
        """Load drift detection data in background."""
        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.ml import MLManager

            db = get_database()
            ml_manager = MLManager(db)
            ml_manager.load_all()

            # Get drift status for all models
            drift_status = ml_manager.get_drift_status()

            # Update UI on main thread
            self.after(0, lambda: self._update_drift_display(drift_status, ml_manager))

        except Exception as e:
            logger.error(f"Failed to load drift data: {e}")
            self.after(0, lambda: self._show_drift_error(str(e)))

    def _update_drift_display(self, drift_status: Dict[str, Dict[str, Any]], ml_manager):
        """Update drift detection display with data."""
        # Clear model list
        for widget in self._drift_model_list.winfo_children():
            widget.destroy()

        if not drift_status:
            no_data_label = ctk.CTkLabel(
                self._drift_model_list,
                text="No ML models trained.\nTrain models in Settings.",
                text_color="gray",
                font=ctk.CTkFont(size=10)
            )
            no_data_label.pack(padx=10, pady=20)
            return

        # Sort models: drifting first, then stable, then no baseline
        def sort_key(item):
            model, status = item
            if not status.get("has_baseline"):
                return (2, model)  # No baseline - last
            elif status.get("is_drifting"):
                return (0, model)  # Drifting - first
            else:
                return (1, model)  # Stable - middle

        sorted_models = sorted(drift_status.items(), key=sort_key)

        # Create model buttons with status indicators
        for model, status in sorted_models:
            has_baseline = status.get("has_baseline", False)
            is_drifting = status.get("is_drifting", False)

            # Determine status indicator
            if not has_baseline:
                indicator = "○"  # Empty circle - no baseline
                color = "gray"
                status_text = "No Data"
            elif is_drifting:
                indicator = "●"  # Filled circle - drifting
                direction = status.get("direction", "")
                color = "#e74c3c"  # Red
                status_text = f"DRIFTING ({direction})" if direction else "DRIFTING"
            else:
                indicator = "●"  # Filled circle - stable
                color = "#27ae60"  # Green
                status_text = "STABLE"

            # Create button for each model
            btn_frame = ctk.CTkFrame(self._drift_model_list, fg_color="transparent")
            btn_frame.pack(fill="x", padx=2, pady=1)

            indicator_label = ctk.CTkLabel(
                btn_frame,
                text=indicator,
                text_color=color,
                font=ctk.CTkFont(size=12),
                width=20
            )
            indicator_label.pack(side="left", padx=(5, 2))

            model_btn = ctk.CTkButton(
                btn_frame,
                text=f"{model}",
                command=lambda m=model, s=status, mgr=ml_manager: self._on_drift_model_select(m, s, mgr),
                fg_color="transparent",
                hover_color=("gray75", "gray25"),
                anchor="w",
                height=24,
                font=ctk.CTkFont(size=11)
            )
            model_btn.pack(side="left", fill="x", expand=True)

        # Auto-select first drifting model if any
        first_drifting = next(
            ((m, s) for m, s in sorted_models if s.get("is_drifting")),
            None
        )
        if first_drifting:
            self._on_drift_model_select(first_drifting[0], first_drifting[1], ml_manager)

    def _on_drift_model_select(self, model: str, status: Dict[str, Any], ml_manager):
        """Handle drift model selection - show drift chart."""
        self._selected_drift_model = model

        # Ensure chart is initialized
        ChartWidget, ChartStyle = _ensure_chart_module()

        if not self.drift_chart:
            if hasattr(self, '_drift_chart_placeholder') and self._drift_chart_placeholder:
                self._drift_chart_placeholder.destroy()
            self.drift_chart = ChartWidget(
                self._drift_chart_frame,
                style=ChartStyle(figure_size=(8, 3), dpi=100)
            )
            self._chart_widgets.append(self.drift_chart)
            self.drift_chart.pack(fill="both", expand=True, padx=10, pady=10)

        if not status.get("has_baseline"):
            self.drift_chart.show_placeholder(f"No baseline data for {model}")
            return

        # Get sigma data for this model from database
        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.database.models import TrackResult, AnalysisResult

            db = get_database()

            with db.session() as session:
                # Get all sigma values for this model, ordered by date
                results = (
                    session.query(
                        TrackResult.sigma_gradient,
                        AnalysisResult.file_date
                    )
                    .join(AnalysisResult)
                    .filter(AnalysisResult.model == model)
                    .filter(TrackResult.sigma_gradient.isnot(None))
                    .order_by(AnalysisResult.file_date)
                    .all()
                )

            if not results:
                self.drift_chart.show_placeholder(f"No data for {model}")
                return

            sigma_values = [r[0] for r in results]
            dates = [r[1] for r in results]

            # Get detector for control limits
            detector = ml_manager.drift_detectors.get(model)
            if detector:
                lower, center, upper = detector.get_control_limits()

                # Calculate baseline cutoff index based on baseline_cutoff_date
                baseline_cutoff_idx = int(len(sigma_values) * 0.7)  # Default
                if detector.baseline_cutoff_date:
                    for i, d in enumerate(dates):
                        if d and d > detector.baseline_cutoff_date:
                            baseline_cutoff_idx = i
                            break

                # Get current CUSUM/EWMA values
                cusum_value = max(detector.cusum_pos, detector.cusum_neg)
                ewma_value = detector.ewma_value

                self.drift_chart.plot_drift_chart(
                    dates=dates,
                    sigma_values=sigma_values,
                    baseline_cutoff_idx=baseline_cutoff_idx,
                    ucl=upper,
                    lcl=lower,
                    center=center,
                    is_drifting=status.get("is_drifting", False),
                    drift_direction=status.get("direction"),
                    cusum_value=cusum_value,
                    ewma_value=ewma_value,
                    model_name=model
                )

                # Update details label with technical values
                baseline_std = detector.baseline_std or 0
                ewma_display = f"{ewma_value:.4f}" if ewma_value is not None else "N/A"
                peak_cusum = detector._peak_cusum
                details_text = (
                    f"CUSUM: {cusum_value:.2f} / {detector.cusum_h:.1f} threshold  |  "
                    f"Peak CUSUM: {peak_cusum:.2f}  |  "
                    f"EWMA: {ewma_display}  |  "
                    f"Baseline Std: {baseline_std:.4f}"
                )
                self._drift_details_label.configure(text=details_text)
            else:
                self.drift_chart.show_placeholder(f"No detector loaded for {model}")
                self._drift_details_label.configure(text="")

        except Exception as e:
            logger.error(f"Error loading drift chart data: {e}")
            self.drift_chart.show_placeholder(f"Error loading data: {e}")

    def _show_drift_error(self, error: str):
        """Show error in drift section."""
        for widget in self._drift_model_list.winfo_children():
            widget.destroy()

        error_label = ctk.CTkLabel(
            self._drift_model_list,
            text=f"Error: {error}",
            text_color="#e74c3c",
            font=ctk.CTkFont(size=10)
        )
        error_label.pack(padx=10, pady=20)
