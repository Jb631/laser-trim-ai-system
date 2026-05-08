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
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from laser_trim_analyzer.utils.threads import get_thread_manager

import numpy as np

from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.config import get_config
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

        # Generation counter — bumped on every navigation/filter change.
        # Background loads capture this value at launch and the after()
        # callbacks discard themselves if a newer load has superseded
        # them. Without this, a stale _load_summary_data callback fires
        # AFTER _create_detail_view destroyed _alerts_frame and tries to
        # create a ChartWidget on the dead frame, raising
        # _tkinter.TclError "bad window path name". Same pattern as
        # ComparePage._load_generation.
        self._load_generation = 0

        self._create_ui()

    def _create_ui(self):
        """Create the trends page UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header with trend type selector
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)

        ctk.CTkLabel(
            header_frame,
            text="Trends & ML Insights",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(side="left")

        self._trend_type = ctk.CTkSegmentedButton(
            header_frame,
            values=["Priorities", "Standard", "Comparative", "Cpk Trend", "Yield", "Drift", "Process Drift", "Trim Difficulty"],
            command=self._on_trend_type_changed,
        )
        self._trend_type.set("Standard")
        self._trend_type.pack(side="right", padx=10)

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

        # Active Only filter - hide inactive models from dropdown
        self.active_only_var = ctk.BooleanVar(value=True)
        self.active_only_check = ctk.CTkCheckBox(
            controls,
            text="Active Only",
            variable=self.active_only_var,
            command=self._on_active_filter_change,
            width=100,
            checkbox_width=18,
            checkbox_height=18,
        )
        self.active_only_check.pack(side="left", padx=(5, 5), pady=15)

        # Minimum sample size filter
        min_samples_label = ctk.CTkLabel(controls, text="Min samples:")
        min_samples_label.pack(side="left", padx=(10, 2), pady=15)
        self.min_samples_entry = ctk.CTkEntry(controls, width=40, justify="center")
        self.min_samples_entry.pack(side="left", padx=2, pady=15)
        self.min_samples_entry.insert(0, "20")

        # Failure rate threshold filter
        fail_rate_label = ctk.CTkLabel(controls, text="Min fail%:")
        fail_rate_label.pack(side="left", padx=(10, 2), pady=15)
        self.fail_rate_entry = ctk.CTkEntry(controls, width=40, justify="center")
        self.fail_rate_entry.pack(side="left", padx=2, pady=15)
        self.fail_rate_entry.insert(0, "0")

        # Date range for active models consideration (by trim date)
        date_label = ctk.CTkLabel(controls, text="Trim Date:")
        date_label.pack(side="left", padx=(10, 5), pady=15)

        self.date_dropdown = ctk.CTkOptionMenu(
            controls,
            values=["Last 30 Days", "Last 90 Days", "Last Year", "All Time"],
            command=self._on_date_change
        )
        self.date_dropdown.set("All Time")
        self.date_dropdown.pack(side="left", padx=5, pady=15)

        # Element type filter
        et_label = ctk.CTkLabel(controls, text="Element:")
        et_label.pack(side="left", padx=(10, 2), pady=15)
        self._element_filter = ctk.CTkComboBox(
            controls,
            values=["All"],
            command=self._on_spec_filter_change,
            width=100,
            font=ctk.CTkFont(size=11)
        )
        self._element_filter.set("All")
        self._element_filter.pack(side="left", padx=2, pady=15)

        # Product class filter
        pc_label = ctk.CTkLabel(controls, text="Class:")
        pc_label.pack(side="left", padx=(5, 2), pady=15)
        self._class_filter = ctk.CTkComboBox(
            controls,
            values=["All"],
            command=self._on_spec_filter_change,
            width=100,
            font=ctk.CTkFont(size=11)
        )
        self._class_filter.set("All")
        self._class_filter.pack(side="left", padx=2, pady=15)

        # Rolling average window (only shown in detail mode — hidden by default)
        self.rolling_label = ctk.CTkLabel(controls, text="Rolling Avg:")
        self.rolling_dropdown = ctk.CTkOptionMenu(
            controls,
            values=["7 Days", "14 Days", "30 Days", "60 Days"],
            command=self._on_rolling_change
        )
        self.rolling_dropdown.set("30 Days")
        # Not packed until detail mode is activated

        # Refresh button
        refresh_btn = ctk.CTkButton(
            controls,
            text="Refresh",
            command=self._refresh_data,
            width=80
        )
        refresh_btn.pack(side="right", padx=5, pady=15)

        # Export PDF button
        export_pdf_btn = ctk.CTkButton(
            controls,
            text="Export PDF",
            command=self._export_summary_pdf,
            width=90
        )
        export_pdf_btn.pack(side="right", padx=5, pady=15)

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

        # Null all chart references so re-initialization guards work correctly
        self.linearity_chart = None
        self.scatter_chart = None
        self.dist_chart = None
        self.alerts_chart = None
        self.trending_chart = None
        self.best_chart = None
        self.recent_issues_chart = None
        self.heatmap_chart = None
        self.drift_chart = None

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
        self.content.grid_rowconfigure(2, weight=0, minsize=120)  # Impact Prioritization
        self.content.grid_rowconfigure(3, weight=1, minsize=180)  # Top 5 / Recent Issues
        self.content.grid_rowconfigure(4, weight=1, minsize=180)  # Trending Worse / Low Data
        self.content.grid_rowconfigure(5, weight=1, minsize=250)  # Drift Detection section
        # Rows 6 (Heat Map) and 7 (ML) configured below with their widgets

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
            ("avg_linearity_rate", "Linearity Pass"),
            ("avg_sigma_rate", "Sigma Pass"),
            ("avg_pass_rate", "Combined Pass"),
            ("models_at_risk", "Needs Attention"),
            ("best_model", "Best (Linearity)"),
            ("worst_model", "Worst (Linearity)"),
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

        # Impact Prioritization pointer — the full ranked list, near-miss
        # split, and cost impact now live on the Priorities tab. This
        # block used to duplicate that as a plain text dump; per UX audit
        # collapse it to a one-line pointer that opens the canonical view.
        self._impact_frame = ctk.CTkFrame(self.content)
        self._impact_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        ctk.CTkLabel(
            self._impact_frame,
            text="Where to Focus →",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=(15, 8), pady=12)

        ctk.CTkLabel(
            self._impact_frame,
            text="Top priority models, near-miss vs hard-fail, and cost impact",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(0, 8), pady=12)

        ctk.CTkButton(
            self._impact_frame,
            text="Open Priorities tab",
            width=160,
            command=lambda: (
                self._trend_type.set("Priorities"),
                self._show_priorities()
            )
        ).pack(side="right", padx=15, pady=10)

        # Best/Worst models side by side
        self._models_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self._models_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
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

        # Row 4: Trending Worse / Low Data Models
        self._row3_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self._row3_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=5)
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
        self._drift_frame.grid(row=5, column=0, sticky="nsew", padx=10, pady=5)
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

        # Heat Map section
        self.content.grid_rowconfigure(6, weight=1, minsize=250)  # Heat map
        self._heatmap_frame = ctk.CTkFrame(self.content)
        self._heatmap_frame.grid(row=6, column=0, sticky="nsew", padx=10, pady=5)

        heatmap_label = ctk.CTkLabel(
            self._heatmap_frame,
            text="Model Pass Rate Heat Map (by Week)",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        heatmap_label.pack(padx=15, pady=(15, 5), anchor="w")

        self._heatmap_placeholder = ctk.CTkLabel(
            self._heatmap_frame, text="Loading heat map...", text_color="gray"
        )
        self._heatmap_placeholder.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.heatmap_chart = None

        # ML Recommendations at bottom
        self.content.grid_rowconfigure(7, weight=0)  # ML section - compact
        ml_frame = ctk.CTkFrame(self.content)
        ml_frame.grid(row=7, column=0, sticky="ew", padx=10, pady=(5, 10))

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
        self._cached_ml_insights = None  # Cache for dialog
        self._update_ml_summary(None)

    def _ensure_summary_charts_initialized(self):
        """Lazily initialize summary view charts - defers matplotlib loading."""
        if self._summary_charts_initialized:
            return

        # Bail out if the summary frames have been destroyed (e.g. user
        # navigated to detail mode while a stale callback was queued).
        # Creating a ChartWidget on a destroyed parent raises
        # _tkinter.TclError "bad window path name".
        if (
            getattr(self, "_alerts_frame", None) is None
            or not self._alerts_frame.winfo_exists()
        ):
            logger.debug("Summary frames missing; skipping chart init")
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

        # Create heatmap chart
        if self._heatmap_placeholder and self._heatmap_placeholder.winfo_exists():
            self._heatmap_placeholder.destroy()
        self.heatmap_chart = ChartWidget(
            self._heatmap_frame,
            style=ChartStyle(figure_size=(10, 4), dpi=100)
        )
        self._chart_widgets.append(self.heatmap_chart)
        self.heatmap_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.heatmap_chart.show_placeholder("Loading heat map data...")

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
        self.content.grid_rowconfigure(1, weight=1, minsize=250)  # Sigma scatter
        self.content.grid_rowconfigure(2, weight=1, minsize=250)  # Linearity scatter - NEW
        self.content.grid_rowconfigure(3, weight=1, minsize=200)  # Distribution
        self.content.grid_rowconfigure(4, weight=0)  # Alerts/ML - compact

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
            ("linearity_pass_rate", "Linearity Pass"),
            ("sigma_pass_rate", "Sigma Pass"),
            ("overall_pass_rate", "Overall Pass"),
            ("sigma_cpk", "Sigma Cpk"),
            ("near_miss", "Near-Miss"),
            ("avg_trim_improvement", "Avg Trim Imp."),
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

        export_pdf_btn = ctk.CTkButton(
            timeline_frame,
            text="Export PDF",
            command=self._export_detail_pdf,
            width=100,
            height=28
        )
        export_pdf_btn.pack(side="right", padx=(10, 0))

        self._scatter_placeholder = ctk.CTkLabel(
            self._scatter_frame,
            text="Loading trend data...",
            text_color="gray"
        )
        self._scatter_placeholder.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.scatter_chart = None

        # Linearity scatter chart - placeholder
        self._linearity_frame = ctk.CTkFrame(self.content)
        self._linearity_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        # Header row
        linearity_header = ctk.CTkFrame(self._linearity_frame, fg_color="transparent")
        linearity_header.pack(fill="x", padx=15, pady=(15, 5))

        linearity_label = ctk.CTkLabel(
            linearity_header,
            text="Linearity Pass Rate Trend",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        linearity_label.pack(side="left", anchor="w")

        self._linearity_placeholder = ctk.CTkLabel(
            self._linearity_frame,
            text="Loading linearity trend data...",
            text_color="gray"
        )
        self._linearity_placeholder.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.linearity_chart = None

        # Distribution chart - placeholder
        self._dist_frame = ctk.CTkFrame(self.content)
        self._dist_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)

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
        bottom_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(5, 10))
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
        if self._scatter_placeholder and self._scatter_placeholder.winfo_exists():
            self._scatter_placeholder.destroy()
        self.scatter_chart = ChartWidget(
            self._scatter_frame,
            style=ChartStyle(figure_size=(10, 4), dpi=100)
        )
        self._chart_widgets.append(self.scatter_chart)
        self.scatter_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.scatter_chart.show_placeholder("Loading trend data...")

        # Create linearity chart
        if self._linearity_placeholder and self._linearity_placeholder.winfo_exists():
            self._linearity_placeholder.destroy()
        if not self.linearity_chart:
            self.linearity_chart = ChartWidget(
                self._linearity_frame,
                style=ChartStyle(figure_size=(10, 4), dpi=100)
            )
            self._chart_widgets.append(self.linearity_chart)
            self.linearity_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
            self.linearity_chart.show_placeholder("Loading linearity trend data...")

        # Create distribution chart
        if self._dist_placeholder and self._dist_placeholder.winfo_exists():
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

                    dropdown.bind("<MouseWheel>", scroll_dropdown)
                    dropdown.bind("<Button-4>", scroll_dropdown)
                    dropdown.bind("<Button-5>", scroll_dropdown)

                    def on_destroy(event):
                        try:
                            dropdown.unbind("<MouseWheel>")
                            dropdown.unbind("<Button-4>")
                            dropdown.unbind("<Button-5>")
                        except Exception:
                            pass

                    dropdown.bind("<Destroy>", on_destroy)
        except Exception as e:
            logger.debug(f"Could not enable dropdown scroll: {e}")

    def _on_model_change(self, model: str):
        """Handle model selection change."""
        self.selected_model = model
        logger.debug(f"Model changed to: {model}")

        # Show/hide rolling avg controls (only useful in detail mode)
        if model == "All Models":
            self.rolling_label.pack_forget()
            self.rolling_dropdown.pack_forget()
            self._create_summary_view()
        else:
            self.rolling_label.pack(side="left", padx=(10, 2), pady=15)
            self.rolling_dropdown.pack(side="left", padx=2, pady=15)
            self._create_detail_view()

        self._refresh_data()

    def _on_active_filter_change(self):
        """Handle Active Only checkbox change - refresh to update model list."""
        self._refresh_data()

    def _filter_model_list(
        self,
        prioritized_models: List[Dict],
        priority_data: Optional[List[Dict]] = None,
    ) -> List[str]:
        """
        Build filtered model list from prioritized models.

        Applies: active_only, min_samples, min_fail_rate, element_type,
        and product_class filters.
        MPS models are always included regardless of filters.
        Uses cached filter values captured on main thread (tkinter is not thread-safe).
        """
        active_only = getattr(self, '_cached_active_only', True)
        min_samples = getattr(self, '_cached_min_samples', 0)
        min_fail_rate = getattr(self, '_cached_min_fail_rate', 0)

        # Element type / product class filter from cached values
        et = getattr(self, '_cached_element_filter', "All")
        pc = getattr(self, '_cached_class_filter', "All")
        spec_filter_models = None
        if (et and et != "All") or (pc and pc != "All"):
            try:
                db = get_database()
                all_specs = db.get_all_model_specs()
                spec_filter_models = set()
                for s in all_specs:
                    if et and et != "All" and s.get("element_type") != et:
                        continue
                    if pc and pc != "All" and s.get("product_class") != pc:
                        continue
                    spec_filter_models.add(s["model"])
            except Exception:
                spec_filter_models = None

        mps_set = set(self.app.config.active_models.mps_models)

        # Build fail rate lookup from priority data if available
        fail_rates = {}
        if priority_data:
            for p in priority_data:
                rate = 100 - p.get("linearity_pass_rate", 100)
                fail_rates[p.get("model", "")] = rate

        model_names = ["All Models"]
        for m in prioritized_models:
            model = m['model']
            status = m['status']
            count = m.get('count', 0)
            is_mps = model in mps_set

            # Active-only filter
            if status == 'inactive' and active_only:
                continue

            # Element type / product class filter
            if spec_filter_models is not None and model not in spec_filter_models:
                continue

            # Min samples filter (MPS models bypass)
            if not is_mps and min_samples > 0 and count < min_samples:
                continue

            # Min fail rate filter (MPS models bypass)
            if not is_mps and min_fail_rate > 0 and model in fail_rates:
                if fail_rates[model] < min_fail_rate:
                    continue

            # Suffix for inactive models
            if status == 'inactive':
                model_names.append(f"{model} (inactive)")
            else:
                model_names.append(model)

        return model_names

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
            self._update_linearity_chart_with_filter()

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
                try:
                    return d["date"] if isinstance(d["date"], datetime) else datetime.strptime(str(d["date"])[:10], "%Y-%m-%d")
                except (ValueError, TypeError):
                    return None

            valid_dates = [dt for dt in (get_date(d) for d in data_points) if dt is not None]
            if not valid_dates:
                return
            most_recent = max(valid_dates)
            cutoff_date = most_recent - timedelta(days=self.chart_timeline_days)
            filtered_points = [d for d in data_points if (get_date(d) or datetime.min) >= cutoff_date]
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
        # Build aligned arrays - only include points with valid sigma
        valid_points = [d for d in normal_points if d["sigma_gradient"] is not None]
        dates = [d["date"].strftime("%m/%d/%y") if hasattr(d["date"], 'strftime') else str(d["date"])[:8] for d in valid_points]
        sigma_values = [d["sigma_gradient"] for d in valid_points]
        pass_flags = [d.get("sigma_pass", False) for d in valid_points]

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

    def _update_linearity_chart_with_filter(self):
        """Update linearity chart with aggregated pass rate by day."""
        if not self.model_trend_data or not self.linearity_chart:
            return

        # Get linearity pass rates by day from database
        linearity_pass_rates = self.model_trend_data.get("linearity_pass_rates_by_day", [])

        if not linearity_pass_rates:
            self.linearity_chart.show_placeholder("No linearity data available")
            return

        # Apply timeline filter - use RELATIVE to data, not absolute calendar dates
        if self.chart_timeline_days > 0 and linearity_pass_rates:
            # Find the most recent date in the data
            def _parse_date(date_str):
                """Parse date string, handling both date-only and datetime formats."""
                try:
                    return datetime.strptime(date_str[:10], "%Y-%m-%d")
                except (ValueError, TypeError):
                    return None

            most_recent_str = max(pr["date"] for pr in linearity_pass_rates)
            most_recent = _parse_date(most_recent_str)
            if most_recent:
                cutoff_date = most_recent - timedelta(days=self.chart_timeline_days)
                linearity_pass_rates = [
                    pr for pr in linearity_pass_rates
                    if (_parse_date(pr["date"]) or datetime.min) >= cutoff_date
                ]

        if len(linearity_pass_rates) < 2:
            self.linearity_chart.show_placeholder("Insufficient linearity data for selected range")
            return

        # Extract dates and pass rates
        dates = [pr["date"] for pr in linearity_pass_rates]
        pass_rates = [pr["pass_rate"] for pr in linearity_pass_rates]
        totals = [pr["total"] for pr in linearity_pass_rates]

        # Format dates for display
        display_dates = [datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d/%y") for d in dates]

        # Calculate rolling average of pass rates
        rolling_avg = None
        window_size = min(self.rolling_window, len(pass_rates))
        if window_size > 1 and pass_rates:
            rolling_avg = []
            for i in range(len(pass_rates)):
                start_idx = max(0, i - window_size + 1)
                # Weighted average by totals
                window_rates = pass_rates[start_idx:i + 1]
                window_totals = totals[start_idx:i + 1]
                total_weight = sum(window_totals)
                if total_weight > 0:
                    weighted_avg = sum(r * t for r, t in zip(window_rates, window_totals)) / total_weight
                else:
                    weighted_avg = 0.0
                rolling_avg.append(weighted_avg)

        # Determine title suffix based on filter
        filter_suffix = f" (Last {self.chart_timeline_days} Days)" if self.chart_timeline_days > 0 else ""

        # Plot as P-chart with binomial control limits
        self.linearity_chart.plot_pchart(
            dates=display_dates,
            pass_rates=pass_rates,
            sample_sizes=totals,
            title=f"Linearity P-Chart - {self.selected_model}{filter_suffix}",
            ylabel="Pass Rate (%)",
        )

    def _refresh_data(self):
        """Refresh data from database."""
        self.status_label.configure(text="Loading...")
        # Bump generation so any in-flight summary/detail load discards
        # its result when the after() callback fires (the user has
        # already moved on, possibly destroying the target frames).
        self._load_generation += 1
        gen = self._load_generation
        # Capture UI filter values on main thread (tkinter is not thread-safe)
        self._cached_active_only = self.active_only_var.get()
        try:
            self._cached_min_samples = int(self.min_samples_entry.get())
        except (ValueError, AttributeError):
            self._cached_min_samples = 0
        try:
            self._cached_min_fail_rate = float(self.fail_rate_entry.get())
        except (ValueError, AttributeError):
            self._cached_min_fail_rate = 0
        self._cached_element_filter = self._element_filter.get() if hasattr(self, '_element_filter') else "All"
        self._cached_class_filter = self._class_filter.get() if hasattr(self, '_class_filter') else "All"
        get_thread_manager().start_thread(
            target=lambda g=gen: self._load_data(g), name="trends-load-data"
        )

    def _load_data(self, gen: int):
        """Load data in background thread; gen lets stale callbacks bail."""
        try:
            db = get_database()

            if self.selected_model == "All Models":
                # Summary mode
                self._load_summary_data(db, gen)
            else:
                # Detail mode
                self._load_detail_data(db, gen)

        except Exception as e:
            logger.error(f"Failed to load trend data: {e}", exc_info=True)
            self.after(0, lambda: self._show_error(str(e)))

    def _load_summary_data(self, db, gen: int = 0):
        """Load data for summary mode."""
        # Get active models config for MPS prioritization
        config = get_config()
        mps_models = config.active_models.mps_models
        recent_days = config.active_models.recent_days

        # Get active models summary
        active_models = db.get_active_models_summary(
            days_back=self.selected_days,
            min_samples=5
        )

        # Get models requiring attention (filter to 20+ samples, linearity-focused)
        alert_models = db.get_models_requiring_attention(
            days_back=self.selected_days,
            min_samples=20,  # Increased to filter out low-data models
            pass_rate_threshold=80.0,
            trend_threshold=10.0,
            rolling_days=self.rolling_window,
            metric="linearity"
        )

        # Get trending worse models
        trending_worse = db.get_trending_worse_models(
            days_back=self.selected_days,
            min_samples=20,
            trend_threshold=10.0,
            rolling_days=self.rolling_window
        )

        # Get impact-ranked prioritization (linearity-focused)
        try:
            priority_models = db.get_linearity_prioritization(days_back=self.selected_days, min_samples=10)
        except Exception as e:
            logger.debug(f"Could not load prioritization: {e}")
            priority_models = []

        # Get prioritized model list for dropdown (MPS first, then active, then inactive)
        prioritized_models = db.get_models_list_prioritized(
            mps_models=mps_models,
            recent_days=recent_days
        )

        # Build model names list with filtering
        model_names = self._filter_model_list(prioritized_models, priority_data=priority_models)

        # Get heatmap data
        try:
            heatmap_data = db.get_heatmap_data(days_back=self.selected_days)
        except Exception as e:
            logger.debug(f"Could not load heatmap data: {e}")
            heatmap_data = None

        # Load ML insights on background thread (disk I/O for MLManager.load_all)
        ml_insights = self._get_ml_summary_insights()

        # Update UI on main thread; capture gen so we can discard stale loads
        self.after(0, lambda g=gen: self._update_summary_display_if_current(
            g, active_models, alert_models, model_names, trending_worse,
            mps_models=mps_models, recent_days=recent_days,
            priority_models=priority_models, heatmap_data=heatmap_data,
            ml_insights=ml_insights
        ))

    def _update_summary_display_if_current(self, gen: int, *args, **kwargs):
        """Apply summary load result only if no newer load has superseded it.

        Without this guard the callback runs even after _create_detail_view
        has destroyed the summary frames, which makes
        _ensure_summary_charts_initialized try to instantiate ChartWidget
        on a dead parent → TclError "bad window path name".
        """
        if gen != self._load_generation:
            return
        # Defensive: also confirm the page itself is still alive AND the
        # parent frame the charts will mount into still exists. If the
        # user has navigated to detail mode in between, the summary
        # frames are gone even though the gen check above should already
        # have bailed — this is belt-and-suspenders for unexpected paths.
        if not self.winfo_exists():
            return
        if (
            getattr(self, "_alerts_frame", None) is None
            or not self._alerts_frame.winfo_exists()
        ):
            return
        self._update_summary_display(*args, **kwargs)

    def _load_detail_data(self, db, gen: int = 0):
        """Load data for detail mode."""
        # Get active models config for MPS prioritization
        config = get_config()
        mps_models = config.active_models.mps_models
        recent_days = config.active_models.recent_days

        # Get detailed trend data for this model
        # Strip (inactive) suffix if present
        clean_model = self.selected_model
        if " (inactive)" in clean_model:
            clean_model = clean_model.replace(" (inactive)", "")

        trend_data = db.get_model_trend_data(
            model=clean_model,
            days_back=self.selected_days,
            rolling_window=self.rolling_window
        )

        # Get alerts for this model (linearity-focused)
        alert_models = db.get_models_requiring_attention(
            days_back=self.selected_days,
            min_samples=5,
            pass_rate_threshold=80.0,
            trend_threshold=10.0,
            rolling_days=self.rolling_window,
            metric="linearity"
        )
        model_alerts = next((a for a in alert_models if a["model"] == clean_model), None)

        # Get ML recommendations
        ml_recommendations = self._get_ml_recommendations(trend_data)

        # Get prioritized model list for dropdown (MPS first, then active, then inactive)
        prioritized_models = db.get_models_list_prioritized(
            mps_models=mps_models,
            recent_days=recent_days
        )

        # Build model names list with filtering
        model_names = self._filter_model_list(prioritized_models)

        # Get the model's analysis-level stats (for consistent pass rate with alerts)
        active_models = db.get_active_models_summary(self.selected_days, 5)
        model_stats = next((m for m in active_models if m["model"] == clean_model), None)

        # Get linearity margin analysis and prioritization data for this model
        try:
            margin_data = db.get_linearity_margin_analysis(clean_model, days_back=self.selected_days)
        except Exception as e:
            logger.debug(f"Could not load margin analysis: {e}")
            margin_data = {}
        model_priority = None
        try:
            prio_list = db.get_linearity_prioritization(days_back=self.selected_days, min_samples=5)
            model_priority = next((m for m in prio_list if m["model"] == clean_model), None)
        except Exception as e:
            logger.debug(f"Could not load prioritization: {e}")

        # Get Cpk
        try:
            cpk_data = db.get_model_cpk(clean_model, days_back=self.selected_days)
        except Exception as e:
            logger.debug(f"Could not load Cpk: {e}")
            cpk_data = None

        # Update UI on main thread; gen lets us discard stale results if
        # the user switched models or back to All Models in the meantime.
        self.after(0, lambda g=gen: self._update_detail_display_if_current(
            g, trend_data, model_alerts, ml_recommendations, model_names, model_stats,
            margin_data=margin_data, model_priority=model_priority, cpk_data=cpk_data
        ))

    def _update_detail_display_if_current(self, gen: int, *args, **kwargs):
        """Apply detail load result only if not superseded by a newer load."""
        if gen != self._load_generation:
            return
        if not self.winfo_exists():
            return
        # Confirm the detail frames the renderer will write into still exist.
        # _create_summary_view destroys these; running anyway crashes Tk.
        scatter_frame = getattr(self, "_scatter_frame", None)
        if scatter_frame is not None and not scatter_frame.winfo_exists():
            return
        self._update_detail_display(*args, **kwargs)

    def _get_ml_recommendations(self, trend_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get ML recommendations for current model using per-model ML system."""
        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.ml import get_shared_ml_manager

            db = get_database()
            # Shared cached MLManager — avoids reloading 134 predictor pickles
            # on every navigation. Settings → Train invalidates the cache.
            ml_manager = get_shared_ml_manager(db)

            # Get threshold from per-model optimizer (strip inactive suffix)
            clean_model_name = self.selected_model.replace(" (inactive)", "")
            threshold = ml_manager.get_threshold(clean_model_name)

            if threshold is not None:
                optimizer = ml_manager.threshold_optimizers.get(clean_model_name)
                profiler = ml_manager.profilers.get(clean_model_name)
                detector = ml_manager.drift_detectors.get(clean_model_name)

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
        trending_worse: Optional[List[Dict[str, Any]]] = None,
        mps_models: Optional[List[str]] = None,
        recent_days: int = 90,
        priority_models: Optional[List[Dict[str, Any]]] = None,
        heatmap_data: Optional[Dict[str, Any]] = None,
        ml_insights: Optional[Dict[str, Any]] = None,
    ):
        """Update summary display with loaded data."""
        if not self.winfo_exists():
            return
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

        # Classify alert models as active or inactive (need this early for models_at_risk)
        mps_set = set(mps_models or [])
        active_cutoff = datetime.now() - timedelta(days=recent_days)

        def is_active_model(model_data):
            """Check if model is MPS or recently active."""
            model_name = model_data.get("model", "")
            if model_name in mps_set:
                return True
            # Check last_date from model_data or find in active_models
            last_date = model_data.get("last_date")
            if not last_date:
                # Look up in active_models
                for am in active_models:
                    if am.get("model") == model_name:
                        last_date = am.get("last_date")
                        break
            return last_date and last_date >= active_cutoff

        # Separate active vs inactive alerts
        active_alerts = [a for a in alert_models if is_active_model(a)]
        inactive_alerts = [a for a in alert_models if not is_active_model(a)]

        # Calculate summary stats
        total_models = len(active_models)
        total_samples = sum(m["total"] for m in active_models)
        avg_pass_rate = sum(m["pass_rate"] for m in active_models) / total_models if total_models > 0 else 0
        avg_sigma_rate = sum(m.get("sigma_pass_rate", 0) for m in active_models) / total_models if total_models > 0 else 0
        avg_linearity_rate = sum(m.get("linearity_pass_rate", 0) for m in active_models) / total_models if total_models > 0 else 0
        models_at_risk = len(active_alerts)  # Only count active models at risk

        # Best and worst models
        sorted_by_rate = sorted(active_models, key=lambda x: x.get("linearity_pass_rate", 0), reverse=True)
        best_model = sorted_by_rate[0]["model"] if sorted_by_rate else "--"
        worst_model = sorted_by_rate[-1]["model"] if sorted_by_rate else "--"
        best_rate = sorted_by_rate[0].get("linearity_pass_rate", 0) if sorted_by_rate else 0
        worst_rate = sorted_by_rate[-1].get("linearity_pass_rate", 0) if sorted_by_rate else 0

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

        # Update alerts chart - show active alerts first, then inactive in separate section
        if active_alerts:
            self.alerts_chart.plot_alert_summary(
                models=[a["model"] for a in active_alerts[:10]],
                alerts=active_alerts[:10],
                title=f"Models Requiring Attention ({len(active_alerts)} active)"
            )
        elif inactive_alerts:
            # Only inactive alerts exist
            self.alerts_chart.plot_alert_summary(
                models=[f"{a['model']} (inactive)" for a in inactive_alerts[:10]],
                alerts=inactive_alerts[:10],
                title=f"Inactive Models with Issues ({len(inactive_alerts)} total)"
            )
        else:
            self.alerts_chart.show_placeholder("All models performing well - no alerts!")

        # Update impact prioritization section
        self._update_impact_display(priority_models or [])

        # Update best models chart (filtered to 20+ samples, sorted by linearity)
        sorted_with_data = sorted(models_with_data, key=lambda x: x.get("linearity_pass_rate", 0), reverse=True)
        best_5 = sorted_with_data[:5]
        if best_5:
            self.best_chart.plot_pass_rate_bars(
                models=[m["model"] for m in best_5],
                pass_rates=[m.get("linearity_pass_rate", 0) for m in best_5],
                sample_counts=[m["total"] for m in best_5],
                title="Top 5 by Linearity",
                highlight_threshold=80.0
            )
        else:
            self.best_chart.show_placeholder("No models with 20+ samples")

        # Recent Issues: models with data in last 30 days (relative to newest data) AND pass_rate < 80%
        # Only show active models (MPS or recently active)
        # Use most recent data date instead of today to handle stale datasets
        all_dates = [m["last_date"] for m in models_with_data if m.get("last_date")]
        latest_data_date = max(all_dates) if all_dates else datetime.now()
        recent_cutoff_30d = latest_data_date - timedelta(days=30)
        recent_issues = [
            m for m in models_with_data
            if m.get("last_date") and m["last_date"] >= recent_cutoff_30d and m["pass_rate"] < 80
            and (m["model"] in mps_set or (m.get("last_date") and m["last_date"] >= active_cutoff))
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

        # Trending Worse section - filter to active models only
        if trending_worse and len(trending_worse) > 0:
            # Filter to active models only
            active_trending = [
                m for m in trending_worse
                if m["model"] in mps_set or is_active_model(m)
            ]
            top_trending = active_trending[:5]
            if top_trending:
                self.trending_chart.plot_trending_worse(
                    models=[m["model"] for m in top_trending],
                    pass_rates=[m["pass_rate"] for m in top_trending],
                    declines=[m["decline"] for m in top_trending],
                    sample_counts=[m["total_samples"] for m in top_trending],
                    title="Trending Worse (>10% decline)"
                )
            else:
                self.trending_chart.show_placeholder("No active models trending worse - stable!")
        else:
            self.trending_chart.show_placeholder("No models trending worse - stable!")

        # Low Data Models section
        self._update_low_data_list(low_data_models)

        # Update heatmap
        self._update_heatmap(heatmap_data)

        # Update ML summary
        self._update_ml_summary(alert_models, ml_insights=ml_insights)

        # Load drift detection data
        self._refresh_drift_data()

        # Update status
        self.status_label.configure(text=f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    def _update_heatmap(self, heatmap_data: Optional[Dict[str, Any]]):
        """Update the heat map chart with model x week pass rate data."""
        try:
            if not self.heatmap_chart:
                return

            if not heatmap_data or not heatmap_data.get("models"):
                self.heatmap_chart.show_placeholder("No data for heat map")
                return

            self.heatmap_chart.plot_heatmap(
                models=heatmap_data["models"],
                periods=heatmap_data["periods"],
                values=heatmap_data["values"],
            )
        except Exception as e:
            logger.debug(f"Heatmap update error: {e}")
            if self.heatmap_chart:
                self.heatmap_chart.show_placeholder("Error loading heat map")

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

    def _update_impact_display(self, priority_models: List[Dict[str, Any]]):
        """No-op kept for backwards compatibility with existing call sites.

        The full impact ranking now lives on the Priorities tab; the
        Standard summary used to render a plain-text version of the same
        data which the UX audit flagged as redundant. The summary view
        now shows a pointer button to Priorities instead.
        """
        return

    def _update_detail_display(
        self,
        trend_data: Dict[str, Any],
        model_alerts: Optional[Dict[str, Any]],
        ml_recommendations: Optional[Dict[str, Any]],
        model_names: List[str],
        model_stats: Optional[Dict[str, Any]] = None,
        margin_data: Optional[Dict[str, Any]] = None,
        model_priority: Optional[Dict[str, Any]] = None,
        cpk_data: Optional[Dict[str, Any]] = None,
    ):
        """Update detail display with loaded data."""
        if not self.winfo_exists():
            return
        # Ensure charts are initialized before use (lazy matplotlib loading)
        self._ensure_detail_charts_initialized()

        # Update model dropdown
        current_model = self.model_dropdown.get()
        self.model_dropdown.configure(values=model_names)
        if current_model in model_names:
            self.model_dropdown.set(current_model)
        else:
            # Current model was filtered out (e.g. Active Only) — switch to summary
            self.model_dropdown.set("All Models")
            self.selected_model = "All Models"
            self.after(50, lambda: self._on_model_change("All Models"))
            return

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
            overall_pass_count = sum(1 for d in data_points if str(d.get("status", "")).upper() == "PASS")
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
        # Near-miss and trim effectiveness from prioritization data
        near_miss = model_priority.get("near_miss_count", 0) if model_priority else 0
        avg_trim_imp = model_priority.get("avg_trim_improvement", None) if model_priority else None
        self.detail_stat_labels["near_miss"].configure(
            text=f"{near_miss}",
            text_color="#f39c12" if near_miss > 0 else "gray"
        )
        self.detail_stat_labels["avg_trim_improvement"].configure(
            text=f"{avg_trim_imp:.1f}%" if avg_trim_imp is not None else "N/A",
            text_color="white"
        )
        self.detail_stat_labels["trend"].configure(text=trend, text_color=trend_color)

        # Cpk
        if cpk_data and cpk_data.get("sigma_cpk") is not None:
            cpk_val = cpk_data["sigma_cpk"]
            cpk_color = cpk_data.get("sigma_cpk_color", "gray")
            self.detail_stat_labels["sigma_cpk"].configure(
                text=f"{cpk_val:.2f}",
                text_color=cpk_color
            )
        else:
            self.detail_stat_labels["sigma_cpk"].configure(text="N/A", text_color="gray")

        # Update scatter chart using the filter method (applies timeline filter)
        self._update_scatter_chart_with_filter()

        # Update linearity chart
        self._update_linearity_chart_with_filter()

        # Update distribution - only show histogram for 20+ samples
        # For small sample sizes, the trend chart above shows everything needed
        if len(sigma_values) >= 20:
            self.dist_chart.master.grid()  # Show the distribution frame
            self.dist_chart.plot_histogram(
                values=sigma_values,
                bins=min(30, len(sigma_values) // 3 + 1),
                title="Linearity Error Distribution",
                xlabel="Linearity Error Rate",
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
        # impact_text was removed in favour of the Priorities-tab pointer
        # button — nothing to reset here. Left as comment for grep history.

    def _reset_detail_stats(self):
        """Reset detail statistics to default values."""
        for key in self.detail_stat_labels:
            self.detail_stat_labels[key].configure(text="--", text_color="white")

    def _update_ml_summary(self, alert_models: Optional[List[Dict[str, Any]]], ml_insights: Optional[Dict[str, Any]] = None):  # noqa: E501
        """ML insights summary on the Standard view.

        The drift portion that used to render here duplicated the Drift
        tab — now we just show a one-line trained-model count and let
        the user click 'View All Details' for the full breakdown.
        """
        """Update ML summary text for all models view with ML insights."""
        # Cache for the details dialog
        self._cached_alert_models = alert_models
        # Use pre-loaded insights (from background thread) or fall back to cached
        self._cached_ml_insights = ml_insights if ml_insights is not None else self._cached_ml_insights

        self.ml_text.configure(state="normal")
        self.ml_text.delete("1.0", "end")

        ml_insights = self._cached_ml_insights
        has_content = False

        if ml_insights:
            has_content = True
            trained = ml_insights.get("trained_models", 0)
            drifting = ml_insights.get("drifting_models", [])

            # Single status line — full drift list lives on the Drift tab,
            # listing model names again here was redundant per UX audit.
            if trained > 0:
                drift_note = (
                    f"{len(drifting)} drifting (see Drift tab)"
                    if drifting else "no drift"
                )
                self.ml_text.insert(
                    "end",
                    f"ML Status: {trained} models trained · {drift_note}\n\n",
                )

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
            from laser_trim_analyzer.ml import get_shared_ml_manager

            db = get_database()
            ml_manager = get_shared_ml_manager(db)

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

        # Use cached data (loaded on background thread during summary refresh)
        ml_insights = getattr(self, '_cached_ml_insights', None)
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

            # Threshold section — explain in plain language
            self.detail_ml_text.insert("end", f"Recommended Threshold: {threshold:.6f}\n")
            self.detail_ml_text.insert("end", f"  Method: {method}, Confidence: {confidence:.0%}\n")
            self.detail_ml_text.insert("end", f"  Based on: {basis}\n")
            # Add plain-English interpretation
            if confidence >= 0.7:
                self.detail_ml_text.insert("end", f"  → High confidence — consider using this threshold\n")
            elif confidence >= 0.4:
                self.detail_ml_text.insert("end", f"  → Moderate confidence — review before applying\n")
            else:
                self.detail_ml_text.insert("end", f"  → Low confidence — more data needed\n")

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
                    self.detail_ml_text.insert("end", f"  Pass Rate: {pass_rate * 100:.1f}%\n")
                if difficulty is not None:
                    diff_label = "Easy" if difficulty < 0.3 else "Medium" if difficulty < 0.6 else "Hard"
                    self.detail_ml_text.insert("end", f"  Difficulty: {diff_label} ({difficulty:.2f})\n")

            # Top insights (new ML system only)
            insights = ml_recommendations.get("insights", [])
            if insights:
                self.detail_ml_text.insert("end", "\nInsights:\n")
                for insight in insights[:3]:
                    msg = insight.message if hasattr(insight, 'message') else str(insight)
                    self.detail_ml_text.insert("end", f"  • {msg}\n")

        self.detail_ml_text.configure(state="disabled")

    def _show_error(self, error: str):
        """Show error state."""
        if not self.winfo_exists():
            return
        self.status_label.configure(text="Error loading data")
        if getattr(self, 'alerts_chart', None):
            self.alerts_chart.show_placeholder(f"Error: {error}")
        if getattr(self, 'scatter_chart', None):
            self.scatter_chart.show_placeholder(f"Error: {error}")

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Trends page shown")
        # Populate spec filter dropdowns
        self._populate_spec_filters()
        # Recreate view if charts were cleaned up
        if not self._summary_charts_initialized and not self._detail_charts_initialized:
            if self.selected_model == "All Models":
                self._create_summary_view()
            else:
                self._create_detail_view()
        self._refresh_data()

    def _on_trend_type_changed(self, value: str):
        """Handle trend type selector change."""
        if value == "Standard":
            if self.selected_model == "All Models":
                self._create_summary_view()
            else:
                self._create_detail_view()
            self._refresh_data()
        elif value == "Comparative":
            self._show_comparative_trends()
        elif value == "Cpk Trend":
            self._show_cpk_trend()
        elif value == "Yield":
            self._show_yield_trend()
        elif value == "Drift":
            self._show_drift_timeline()
        elif value == "Trim Difficulty":
            self._show_trim_difficulty()
        elif value == "Priorities":
            self._show_priorities()
        elif value == "Process Drift":
            self._show_process_drift()

    def show_drift_tab(self):
        """Public hook used by the Dashboard's Drift Alerts card to jump
        straight to the Drift timeline view on this page."""
        try:
            self._trend_type.set("Drift")
        except Exception:
            pass
        self._show_drift_timeline()

    def _create_dedicated_chart_view(self, title: str) -> "ChartWidget":
        """Create a dedicated full-width chart view for non-Standard tabs.

        Replaces the summary view with a single titled chart, avoiding the
        Standard tab's stats/headers bleeding into other tab views.
        """
        self._cleanup_charts()

        for widget in self.content.winfo_children():
            widget.destroy()

        ChartWidget, ChartStyle = _ensure_chart_module()

        frame = ctk.CTkFrame(self.content)
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text=title,
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        chart = ChartWidget(
            frame,
            style=ChartStyle(figure_size=(12, 6), dpi=100)
        )
        chart.grid(row=1, column=0, sticky="nsew", padx=15, pady=(5, 15))
        self._chart_widgets.append(chart)

        self.content.grid_rowconfigure(0, weight=1)

        return chart

    def _draw_empty_state(self, ax, message: str) -> None:
        """Render a visible empty-state message centered on an Axes.

        Used when a chart can't be produced (no data, missing spec, etc.) so
        the user sees an explanation instead of a silent blank page.
        """
        ax.clear()
        ax.text(
            0.5, 0.5, message,
            ha='center', va='center',
            transform=ax.transAxes,
            fontsize=11,
            color='#adb5bd',
            wrap=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _show_comparative_trends(self):
        """Show comparative pass rate trends for top models."""
        config = get_config()
        # Cap at 5 models so the line chart stays legible. Surface the cap
        # in the chart title so the user knows how many MPS models are
        # configured but not shown — previously the cap was silent.
        all_mps = config.active_models.mps_models or []
        mps = all_mps[:5]
        if not mps:
            chart = self._create_dedicated_chart_view("Comparative Pass Rate Trends")
            fig = chart.figure
            fig.clear()
            ax = fig.add_subplot(111)
            chart._style_axis(ax)
            self._draw_empty_state(
                ax,
                "No MPS models configured.\n\n"
                "Go to Settings → Active Models to set your\n"
                "MPS (Master Production Schedule) model list.",
            )
            chart.canvas.draw_idle()
            self.status_label.configure(text="No MPS models configured")
            return

        self.status_label.configure(text="Loading comparative trends...")

        selected_days = self.selected_days

        def _load():
            try:
                db = get_database()
                data = db.get_comparative_model_trends(mps, days_back=selected_days, period="week")
                self.after(0, lambda d=data, total=len(all_mps), shown=len(mps):
                            self._render_comparative_trends(d, total, shown))
            except Exception as e:
                logger.error(f"Comparative trends error: {e}")
                self.after(0, lambda: self.status_label.configure(
                    text=f"Comparative error: {e}"))

        get_thread_manager().start_thread(target=_load, name="comparative-trends")

    def _render_comparative_trends(self, data, total_mps: int = 0, shown_mps: int = 0):
        """Render comparative trends chart on the main thread."""
        if not self.winfo_exists():
            return
        try:
            if not data:
                self.status_label.configure(text="No data for comparative trends")
                return

            chart = self._create_dedicated_chart_view("Comparative Pass Rate Trends")
            fig = chart.figure
            fig.clear()
            ax = fig.add_subplot(111)
            chart._style_axis(ax)

            # Build a unified, chronologically-sorted x-axis covering every week any
            # model has data for. Then plot each model aligned to that shared axis,
            # using None for weeks the model has no data so matplotlib breaks the
            # line cleanly instead of zigzagging.
            #
            # The DB returns periods like "2026-W06"; sorted() works correctly on
            # these strings as long as the year-Wweek format is consistent (zero-
            # padded week, ISO-style year prefix).
            all_periods = sorted({
                t["period"]
                for trend in data.values()
                for t in (trend or [])
            })

            for model, trend in data.items():
                if not trend:
                    continue
                rate_by_period = {t["period"]: t["pass_rate"] for t in trend}
                rates = [rate_by_period.get(p) for p in all_periods]  # None where missing
                ax.plot(all_periods, rates, marker='o', markersize=4, label=model)

            ax.set_ylabel("Pass Rate (%)")
            # Surface the silent MPS cap in the title so users know more
            # models are configured than displayed.
            if total_mps and total_mps > shown_mps:
                ax.set_title(
                    f"Comparative Pass Rate Trends — showing {shown_mps} of "
                    f"{total_mps} MPS models"
                )
            else:
                ax.set_title("Comparative Pass Rate Trends")
            # 80% target reference line — same threshold the rest of the
            # page uses for alerts.
            ax.axhline(y=80, color="#198754", linestyle="--",
                       alpha=0.7, label="Target (80%)")
            # Thin x-axis ticks so dense weekly periods don't collide.
            if len(all_periods) > 12:
                step = max(1, len(all_periods) // 12)
                ax.set_xticks(range(0, len(all_periods), step))
                ax.set_xticklabels(
                    [all_periods[i] for i in range(0, len(all_periods), step)],
                    rotation=45, fontsize=8,
                )
            ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
            ax.set_ylim(0, 105)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            fig.tight_layout()
            chart.canvas.draw_idle()
            cap_note = (
                f" ({shown_mps} of {total_mps} MPS models)"
                if total_mps > shown_mps else ""
            )
            self.status_label.configure(
                text=f"Comparing {len(data)} models{cap_note}"
            )
        except Exception as e:
            logger.error(f"Comparative trends error: {e}")
            self.status_label.configure(text=f"Comparative error: {e}")

    def _show_cpk_trend(self):
        """Show Cpk trend over time for the selected model.

        If "All Models" is selected, fall back to a Cpk comparison chart across
        every model that has a linearity spec (via get_cpk_by_model). This is
        better than silently doing nothing, which was the previous behavior.
        """
        model = self.selected_model.replace(" (inactive)", "")
        selected_days = self.selected_days
        self.status_label.configure(text="Loading Cpk trend...")

        def _load():
            try:
                db = get_database()
                if not model or model == "All Models":
                    rows = db.get_cpk_by_model(days_back=selected_days)
                    self.after(0, lambda: self._render_cpk_trend(
                        model=model, branch="all", rows=rows))
                else:
                    spec = db.get_model_spec(model)
                    trend = None
                    if spec and spec.get("linearity_spec_pct"):
                        trend = db.get_cpk_trend_for_model(
                            model, spec["linearity_spec_pct"],
                            days_back=selected_days, period="month"
                        )
                    self.after(0, lambda: self._render_cpk_trend(
                        model=model, branch="single", spec=spec, trend=trend))
            except Exception as e:
                logger.error(f"Cpk trend error: {e}")
                self.after(0, lambda: self.status_label.configure(
                    text=f"Cpk trend error: {e}"))

        get_thread_manager().start_thread(target=_load, name="cpk-trend")

    def _render_cpk_trend(self, model, branch, rows=None, spec=None, trend=None):
        """Render Cpk trend chart on the main thread."""
        if not self.winfo_exists():
            return
        try:
            title = "Cpk by Model" if branch == "all" else f"Cpk Trend — {model}"
            chart = self._create_dedicated_chart_view(title)
            fig = chart.figure
            fig.clear()
            ax = fig.add_subplot(111)
            chart._style_axis(ax)

            # Branch 1: "All Models" -> Cpk comparison across every spec'd model.
            if branch == "all":
                if not rows:
                    self._draw_empty_state(
                        ax,
                        "No Cpk data available.\n\n"
                        "Cpk requires at least 10 linearity measurements per model\n"
                        "and a linearity spec defined in Model Specs.",
                    )
                    chart.canvas.draw_idle()
                    self.status_label.configure(text="No Cpk data")
                    return

                # Show worst-to-best — models that need attention first.
                rows_sorted = sorted(
                    rows, key=lambda r: r["cpk"] if r["cpk"] is not None else 99
                )
                # Cap to bottom 20 worst Cpk so 60+ rows can't pile into an
                # illegible wall of overlapping y-labels (the visual review
                # showed labels stacking on top of each other at scale).
                # An extreme outlier (e.g. Cpk=130 for one model) used to
                # crush every other bar into a sliver — clipping the x-axis
                # at 5.0 and capping rows fixes both problems together.
                CPK_ROW_CAP = 20
                CPK_X_LIMIT = 5.0
                total_models = len(rows_sorted)
                rows_shown = rows_sorted[:CPK_ROW_CAP]
                # Reverse so the worst sits at the top of the bar chart.
                rows_shown = list(reversed(rows_shown))
                models = [r["model"] for r in rows_shown]
                cpks_raw = [r["cpk"] if r["cpk"] is not None else 0 for r in rows_shown]
                # Clip bar widths for display, but keep the real value for the
                # label so users can still see "Cpk = 130" annotated even
                # though the bar runs off-axis.
                cpks_clipped = [min(c, CPK_X_LIMIT) for c in cpks_raw]
                colors = [
                    '#dc3545' if c < 1.0 else '#fd7e14' if c < 1.33 else '#198754'
                    for c in cpks_raw
                ]

                # Scale figure height to row count so 9pt y-labels stay readable.
                target_height = max(4.0, min(12.0, 1.5 + 0.35 * len(models)))
                fig.set_size_inches(
                    fig.get_size_inches()[0], target_height, forward=True
                )

                y_pos = list(range(len(models)))
                ax.barh(y_pos, cpks_clipped, color=colors,
                        edgecolor="#1a1a1a", linewidth=0.5)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(models, fontsize=9)

                # Annotate each bar with the actual Cpk value (not the clipped
                # one) so the operator can read the number directly.
                for i, (cpk_real, cpk_disp) in enumerate(zip(cpks_raw, cpks_clipped)):
                    over = " (clipped)" if cpk_real > CPK_X_LIMIT else ""
                    ax.text(
                        min(cpk_disp, CPK_X_LIMIT) + 0.05,
                        i,
                        f"{cpk_real:.2f}{over}",
                        va="center",
                        fontsize=8,
                        color="#cccccc",
                    )

                ax.axvline(x=1.33, color='#fd7e14', linestyle='--', alpha=0.7,
                           label='Capable (1.33)')
                ax.axvline(x=1.0, color='#dc3545', linestyle='--', alpha=0.7,
                           label='Minimum (1.0)')
                ax.set_xlim(0, CPK_X_LIMIT * 1.1)
                ax.set_xlabel("Cpk (clipped at 5.0)")

                if total_models > CPK_ROW_CAP:
                    ax.set_title(
                        f"Cpk by Model — bottom {CPK_ROW_CAP} of {total_models} "
                        f"(last {self.selected_days}d, n≥10)"
                    )
                else:
                    ax.set_title(
                        f"Cpk by Model — last {self.selected_days}d, n≥10"
                    )
                ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
                ax.grid(True, axis="x", alpha=0.2)
                try:
                    fig.tight_layout()
                except Exception:
                    pass
                chart.canvas.draw_idle()
                self.status_label.configure(
                    text=f"Cpk: showing worst {len(models)} of {total_models}"
                )
                return

            # Branch 2: single model -> time-series Cpk trend.
            if not spec or not spec.get("linearity_spec_pct"):
                self._draw_empty_state(
                    ax,
                    f"No linearity spec defined for {model}.\n\n"
                    "Cpk requires a linearity spec percent.\n"
                    "Go to Model Specs to add one.",
                )
                chart.canvas.draw_idle()
                self.status_label.configure(text=f"No linearity spec for {model}")
                return

            valid = [t for t in (trend or []) if t.get("cpk") is not None]
            if not valid:
                self._draw_empty_state(
                    ax,
                    f"No Cpk data for {model} in the last 180 days.\n\n"
                    "Cpk needs at least 10 samples per period.",
                )
                chart.canvas.draw_idle()
                self.status_label.configure(text=f"No Cpk data for {model}")
                return

            periods = [t["period"] for t in valid]
            cpk_values = [t["cpk"] for t in valid]

            ax.plot(periods, cpk_values, marker='o', color='#0d6efd', linewidth=2, label="Cpk")
            ax.axhline(y=1.67, color='#198754', linestyle='--', alpha=0.7, label='Excellent (1.67)')
            ax.axhline(y=1.33, color='#fd7e14', linestyle='--', alpha=0.7, label='Capable (1.33)')
            ax.axhline(y=1.0, color='#dc3545', linestyle='--', alpha=0.7, label='Minimum (1.0)')
            ax.set_ylabel("Cpk")
            ax.set_title(f"Cpk Trend - {model}")
            ax.legend(loc="best", fontsize=8)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            fig.tight_layout()
            chart.canvas.draw_idle()
            self.status_label.configure(text=f"Cpk trend for {model}")
        except Exception as e:
            logger.error(f"Cpk trend error: {e}")
            self.status_label.configure(text=f"Cpk trend error: {e}")

    def _show_yield_trend(self):
        """Show overall yield trend across all models."""
        self.status_label.configure(text="Loading yield trend...")
        selected_days = self.selected_days
        # Pick aggregation period to keep ~10–60 buckets on the chart.
        # Weekly buckets over 10 years = 520 ticks, unreadable; monthly
        # buckets over 90 days = 3 ticks, also useless. Heuristic:
        #   ≤ 90 d  → daily
        #   ≤ 730 d → weekly
        #   > 730 d → monthly
        if selected_days <= 90:
            period = "day"
        elif selected_days <= 730:
            period = "week"
        else:
            period = "month"

        def _load():
            try:
                db = get_database()
                data = db.get_yield_trend(days_back=selected_days, period=period)
                self.after(0, lambda d=data, p=period: self._render_yield_trend(d, p))
            except Exception as e:
                logger.error(f"Yield trend error: {e}", exc_info=True)
                self.after(0, lambda exc=e: self.status_label.configure(
                    text=f"Yield trend error: {exc}"))

        get_thread_manager().start_thread(target=_load, name="yield-trend")

    def _render_yield_trend(self, data, period: str = "week"):
        """Render yield trend chart on the main thread."""
        if not self.winfo_exists():
            return
        try:
            chart = self._create_dedicated_chart_view("Yield Trend")
            fig = chart.figure
            fig.clear()
            ax = fig.add_subplot(111)
            chart._style_axis(ax)

            if not data:
                self._draw_empty_state(
                    ax,
                    "No yield data in the last 180 days.\n\n"
                    "Process some files to populate yield history.",
                )
                chart.canvas.draw_idle()
                self.status_label.configure(text="No yield data")
                return

            periods = [d["period"] for d in data]
            rates = [d["pass_rate"] for d in data]

            if len(periods) < 2:
                # Single bucket = noise, not a trend. Tell the user the
                # date filter / aggregation produced no usable trend
                # rather than drawing a misleading single point.
                self._draw_empty_state(
                    ax,
                    "Not enough data to draw a yield trend.\n\n"
                    "Try a wider date range, or check that\n"
                    "files have been processed in this window.",
                )
                chart.canvas.draw_idle()
                self.status_label.configure(text="Yield trend: insufficient data")
                return

            x = list(range(len(periods)))
            target = 80.0
            # Fill between the curve and the target so colour shows the gap
            # to spec, not absolute area-under-the-curve which exaggerated
            # 90% yield as a giant blue block (per usability review).
            above = [max(r, target) for r in rates]
            below = [min(r, target) for r in rates]
            ax.fill_between(x, target, above, color="#198754", alpha=0.25,
                            label="Above target")
            ax.fill_between(x, below, target, color="#dc3545", alpha=0.25,
                            label="Below target")
            ax.plot(x, rates, marker="o", markersize=3,
                    color="#0d6efd", linewidth=1.5, label="Yield")

            # Rolling-average overlay smooths out weekly/daily noise so
            # the underlying trend is visible at long date ranges. Window
            # is ~10% of the visible periods, capped 4–12 buckets.
            if len(rates) >= 6:
                w = max(4, min(12, len(rates) // 10))
                rolling = []
                for i in range(len(rates)):
                    lo = max(0, i - w + 1)
                    window_vals = [v for v in rates[lo:i + 1] if v is not None]
                    rolling.append(
                        sum(window_vals) / len(window_vals) if window_vals else None
                    )
                ax.plot(x, rolling, color="#fd7e14", linewidth=2.0,
                        label=f"Rolling avg ({w}{period[0]})")

            ax.axhline(y=target, color="#198754", linestyle="--",
                       alpha=0.7, label="Target (80%)")

            step = max(1, len(periods) // 10)
            ax.set_xticks(range(0, len(periods), step))
            ax.set_xticklabels(
                [periods[i] for i in range(0, len(periods), step)],
                rotation=45, fontsize=8,
            )
            ax.set_ylabel("Yield (%)")
            granularity = {"day": "daily", "week": "weekly",
                           "month": "monthly"}.get(period, period)
            ax.set_title(
                f"Overall Yield Trend — last {self.selected_days}d, "
                f"{granularity} buckets ({len(periods)} points)"
            )
            # Auto-scale y-axis around the data with breathing room rather
            # than 0–105% which wasted half the chart on empty space.
            data_min = min(r for r in rates if r is not None)
            data_max = max(r for r in rates if r is not None)
            pad = max(2.0, (data_max - data_min) * 0.1)
            ax.set_ylim(max(0, data_min - pad), min(105, data_max + pad))
            ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
            ax.grid(True, axis="y", alpha=0.2)
            try:
                fig.tight_layout()
            except Exception:
                pass
            chart.canvas.draw_idle()
            self.status_label.configure(
                text=f"Yield trend: {len(periods)} {granularity} buckets"
            )
        except Exception as e:
            logger.error(f"Yield trend error: {e}")
            self.status_label.configure(text=f"Yield trend error: {e}")

    def _show_drift_timeline(self):
        """Show drift detection events as a timeline."""
        self.status_label.configure(text="Loading drift timeline...")
        selected_days = self.selected_days

        def _load():
            try:
                db = get_database()
                events = db.get_drift_events_timeline(days_back=selected_days)
                self.after(0, lambda: self._render_drift_timeline(events))
            except Exception as e:
                logger.error(f"Drift timeline error: {e}")
                self.after(0, lambda: self.status_label.configure(
                    text=f"Drift error: {e}"))

        get_thread_manager().start_thread(target=_load, name="drift-timeline")

    def _render_drift_timeline(self, events):
        """Render drift timeline chart on the main thread."""
        if not self.winfo_exists():
            return
        try:
            chart = self._create_dedicated_chart_view("Drift Detection Timeline")
            # Use ChartWidget.clear() (not fig.clear()) so the dark facecolor is
            # restored — fig.clear() resets it to matplotlib's default (black
            # under the dark_background style), making the chart unreadable.
            chart.clear()
            fig = chart.figure
            ax = fig.add_subplot(111)
            chart._style_axis(ax)

            if not events:
                self._draw_empty_state(ax, "No drift events detected")
            else:
                # Parse ISO date strings to datetime so matplotlib renders a
                # proper time axis. Previously every marker was passed as the
                # string slice e["date"][:10], which matplotlib treats as a
                # categorical value — every event with the same date string
                # collapsed onto a single x-position, making the chart look
                # like one stacked column of triangles regardless of when
                # the events happened.
                from datetime import datetime as _dt
                from matplotlib import dates as mdates

                def _parse_date(raw):
                    if not raw:
                        return None
                    try:
                        return _dt.fromisoformat(raw[:19] if len(raw) >= 19 else raw[:10])
                    except (ValueError, TypeError):
                        return None

                # Sort models so the most-recent drift sits at the top of the chart.
                # Color-code by direction (red = up = degrading, orange = down = improving).
                last_event = {}
                for e in events:
                    parsed = _parse_date(e.get("date"))
                    if parsed is None:
                        continue
                    if (
                        e["model"] not in last_event
                        or parsed > last_event[e["model"]]
                    ):
                        last_event[e["model"]] = parsed
                if not last_event:
                    self._draw_empty_state(
                        ax, "Drift events have no parseable detection dates"
                    )
                    chart.canvas.draw_idle()
                    self.status_label.configure(text="Drift dates malformed")
                    return
                models = sorted(last_event, key=lambda m: last_event[m], reverse=True)
                model_y = {m: i for i, m in enumerate(models)}

                # Dynamically size the figure so each model gets ~0.3" of vertical space.
                target_height = max(4.0, min(12.0, 1.5 + 0.3 * len(models)))
                fig.set_size_inches(fig.get_size_inches()[0], target_height, forward=True)

                up_x, up_y = [], []
                down_x, down_y = [], []
                for e in events:
                    if e["model"] not in model_y:
                        continue
                    parsed = _parse_date(e.get("date"))
                    if parsed is None:
                        continue
                    y = model_y[e["model"]]
                    if (e.get("direction") or "").lower() == "down":
                        down_x.append(parsed); down_y.append(y)
                    else:
                        up_x.append(parsed); up_y.append(y)

                if up_x:
                    ax.scatter(up_x, up_y, s=70, c='#dc3545', zorder=5,
                               marker='^', label='Drifting up (degrading)')
                if down_x:
                    ax.scatter(down_x, down_y, s=70, c='#f39c12', zorder=5,
                               marker='v', label='Drifting down (improving)')

                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax.set_yticks(range(len(models)))
                ax.set_yticklabels(models, fontsize=9)
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.set_xlabel("Detection date")
                ax.legend(loc='upper left', fontsize=9, framealpha=0.85)
                ax.grid(True, axis='x', alpha=0.2)

            ax.set_title("Drift Detection Timeline")
            try:
                fig.tight_layout()
            except Exception:
                pass  # tight_layout sometimes fails with rotated tick labels
            chart.canvas.draw_idle()
            self.status_label.configure(text=f"Drift timeline ({len(events)} events)")
        except Exception as e:
            logger.error(f"Drift timeline error: {e}")
            self.status_label.configure(text=f"Drift error: {e}")

    # Metrics shown on the Process Drift tab. The DB-side keys must
    # match DatabaseManager._PROCESS_DRIFT_METRICS.
    _PROCESS_DRIFT_PANELS = (
        ("untrimmed_resistance", "Untrimmed Resistance",
         "Carbon batch / tooling drift"),
        ("measured_electrical_angle", "Measured Electrical Angle",
         "Fixture / setup drift"),
        ("trim_pass_count", "Trim Passes",
         "Process difficulty creep"),
    )

    def _show_process_drift(self):
        """Three-panel view of physical-measurement drift per model.

        Same z-score-based drift detection ML uses, applied to the
        physical track signals (untrimmed resistance, electrical angle,
        trim pass count). A drifting baseline often shows up here weeks
        before sigma starts moving.
        """
        self.status_label.configure(text="Loading process drift...")
        baseline_days = max(self.selected_days, 60)
        # Recent window = ~15% of baseline, clamped 7–28 days.
        recent_days = max(7, min(28, baseline_days // 7))

        def _load():
            try:
                db = get_database()
                panels = []
                for metric, label, subtitle in self._PROCESS_DRIFT_PANELS:
                    try:
                        rows = db.get_process_drift_by_model(
                            metric=metric,
                            baseline_days=baseline_days,
                            recent_days=recent_days,
                        )
                    except Exception as e:
                        logger.debug(
                            f"Process drift load failed for {metric}: {e}",
                            exc_info=True,
                        )
                        rows = []
                    panels.append((metric, label, subtitle, rows))
                self.after(0, lambda p=panels, bd=baseline_days, rd=recent_days:
                            self._render_process_drift(p, bd, rd))
            except Exception as e:
                logger.error(f"Process drift load error: {e}", exc_info=True)
                self.after(0, lambda exc=e: self.status_label.configure(
                    text=f"Process drift error: {exc}"))

        get_thread_manager().start_thread(target=_load, name="process-drift")

    def _render_process_drift(self, panels, baseline_days: int, recent_days: int):
        """Render three stacked drift panels — one per physical metric."""
        if not self.winfo_exists():
            return
        try:
            chart = self._create_dedicated_chart_view("Process Drift")
            chart.clear()
            fig = chart.figure
            n = len(panels)
            # Per-panel rows shown
            per_panel_rows = [min(10, len(rows)) for _, _, _, rows in panels]
            total_rows = sum(max(1, r) for r in per_panel_rows)
            target_height = max(7.0, min(18.0, 1.5 + 0.45 * total_rows + 1.0 * n))
            fig.set_size_inches(fig.get_size_inches()[0], target_height,
                                 forward=True)
            gs = fig.add_gridspec(n, 1, hspace=0.65)
            db_metrics = self._db_metric_meta()

            any_drifting = 0
            for i, (metric, label, subtitle, rows) in enumerate(panels):
                ax = fig.add_subplot(gs[i])
                chart._style_axis(ax)
                meta = db_metrics.get(metric, {})
                unit = meta.get("unit", "")
                fmt = meta.get("fmt", "{:.2f}")

                if not rows:
                    ax.set_title(
                        f"{label} — no data",
                        loc="left", fontsize=12, fontweight="bold",
                        color="#ffffff",
                    )
                    self._draw_empty_state(
                        ax,
                        f"Need ≥20 baseline samples and ≥5 recent samples\n"
                        f"per model. None met the threshold.",
                    )
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    continue

                # Show top 10 by |z|.
                shown = rows[:10]
                drifting_in_panel = sum(1 for r in shown if r["is_drifting"])
                any_drifting += drifting_in_panel
                # Reverse so largest |z| ends at the top of the bar chart.
                shown_rev = list(reversed(shown))
                models = [r["model"] for r in shown_rev]
                z_scores = [r["z_score"] for r in shown_rev]

                def _color(r):
                    if not r["is_drifting"]:
                        return "#198754"  # within tolerance
                    z = abs(r["z_score"])
                    if z >= 4.0:
                        return "#6f42c1"  # extreme
                    if z >= 3.0:
                        return "#dc3545"  # severe
                    return "#fd7e14"      # moderate

                colors = [_color(r) for r in shown_rev]
                y_pos = list(range(len(models)))
                ax.barh(y_pos, z_scores, color=colors,
                        edgecolor="#1a1a1a", linewidth=0.5)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(models, fontsize=9)

                # Annotate each bar with baseline mean and recent mean
                # so the operator can see what the actual numbers are
                # rather than just the z-score.
                xmax = max((abs(z) for z in z_scores), default=1.0)
                for j, r in enumerate(shown_rev):
                    base_str = fmt.format(r["baseline_mean"])
                    recent_str = fmt.format(r["recent_mean"])
                    arrow = "↑" if r["direction"] == "up" else (
                        "↓" if r["direction"] == "down" else "·"
                    )
                    text = (
                        f"  z={r['z_score']:+.1f}  {arrow} "
                        f"{base_str} → {recent_str} {unit}"
                        f"  (n={r['baseline_n']}/{r['recent_n']})"
                    )
                    # Place the annotation at the bar end, biased outward
                    # so positive bars get text right of zero, negative bars
                    # left of zero. Use ha to keep it readable.
                    z = z_scores[j]
                    if z >= 0:
                        ax.text(z, j, text, va="center", ha="left",
                                fontsize=8, color="#cccccc")
                    else:
                        ax.text(z, j, text, va="center", ha="right",
                                fontsize=8, color="#cccccc")

                # ±2σ reference lines for the drift threshold.
                ax.axvline(x=0, color="#666666", linewidth=0.6)
                ax.axvline(x=2.0, color="#fd7e14", linestyle="--",
                           linewidth=0.8, alpha=0.7)
                ax.axvline(x=-2.0, color="#fd7e14", linestyle="--",
                           linewidth=0.8, alpha=0.7)
                ax.set_xlim(-max(3.0, xmax * 1.5), max(3.0, xmax * 1.5))
                ax.set_xlabel("z-score (recent vs baseline)")
                ax.set_title(
                    f"{label} — {subtitle}  ({drifting_in_panel}/"
                    f"{len(shown)} drifting)",
                    loc="left", fontsize=12, fontweight="bold",
                    color="#ffffff",
                )
                ax.grid(True, axis="x", alpha=0.2)

            try:
                fig.tight_layout()
            except Exception:
                pass
            chart.canvas.draw_idle()
            self.status_label.configure(
                text=(
                    f"Process drift — baseline {baseline_days}d vs "
                    f"recent {recent_days}d, {any_drifting} drifting"
                )
            )
        except Exception as e:
            logger.error(f"Process drift render error: {e}", exc_info=True)
            self.status_label.configure(text=f"Process drift error: {e}")

    @staticmethod
    def _db_metric_meta():
        """Cached lookup for axis-unit metadata (kept here so the GUI
        doesn't have to import the DB module's private constant)."""
        from laser_trim_analyzer.database.manager import DatabaseManager
        return DatabaseManager._PROCESS_DRIFT_METRICS

    def _show_priorities(self):
        """Three-section view that answers the operator's three highest-
        value questions: where to focus this week, near-miss vs hard-fail
        breakdown, and where money is being lost. All data already exists
        in the DB (linearity prioritization, near-miss summary) and config
        (model_prices, cost_ratio); this just composes them."""
        self.status_label.configure(text="Loading priorities...")
        selected_days = self.selected_days

        def _load():
            try:
                db = get_database()
                priorities = db.get_linearity_prioritization(
                    days_back=selected_days, min_samples=10
                )
                near_miss = db.get_near_miss_summary(days_back=selected_days)
                cfg = get_config()
                pricing = dict(cfg.active_models.model_prices or {})
                cost_ratio = float(getattr(cfg.active_models, "cost_ratio", 0.5))
                self.after(0, lambda p=priorities, nm=near_miss, pr=pricing,
                              cr=cost_ratio: self._render_priorities(p, nm, pr, cr))
            except Exception as e:
                logger.error(f"Priorities load error: {e}", exc_info=True)
                self.after(0, lambda exc=e: self.status_label.configure(
                    text=f"Priorities error: {exc}"))

        get_thread_manager().start_thread(target=_load, name="priorities")

    def _render_priorities(self, priorities, near_miss, pricing, cost_ratio):
        """Render the three-section priorities view on the main thread."""
        if not self.winfo_exists():
            return
        try:
            chart = self._create_dedicated_chart_view("Priorities")
            chart.clear()
            fig = chart.figure
            # Tall figure with 3 stacked subplots, sized to row content.
            n_focus = min(5, len(priorities or []))
            n_cost = min(15, sum(1 for p in (priorities or [])
                                 if pricing.get(p["model"])))
            target_height = 4.0 + 0.4 * n_focus + 0.35 * n_cost + 2.0
            fig.set_size_inches(fig.get_size_inches()[0],
                                 max(8.0, min(18.0, target_height)),
                                 forward=True)
            gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.4],
                                   hspace=0.6)

            # ─── Section 1: Focus This Week ─────────────────────────────
            ax1 = fig.add_subplot(gs[0])
            chart._style_axis(ax1)
            ax1.set_title("Focus This Week — Top Priority Models",
                          loc="left", fontsize=12, fontweight="bold",
                          color="#ffffff")
            if not priorities:
                self._draw_empty_state(ax1, "No priority models in this window")
            else:
                top = priorities[:n_focus]
                lines = []
                for i, p in enumerate(top, start=1):
                    rec = p.get("recommendation", "Monitor")
                    lines.append(
                        f"{i}. {p['model']:>8}  "
                        f"fail rate {100 - p['linearity_pass_rate']:>4.1f}%  "
                        f"({p['failed_units']} fails / {p['total_tracks']} tracks, "
                        f"{p.get('near_miss_count', 0)} near-miss)\n"
                        f"     → {rec}"
                    )
                ax1.text(0.01, 0.98, "\n".join(lines),
                         transform=ax1.transAxes, ha="left", va="top",
                         fontsize=10, color="#dddddd",
                         family="monospace")
            ax1.set_xticks([])
            ax1.set_yticks([])
            for spine in ax1.spines.values():
                spine.set_visible(False)

            # ─── Section 2: Near-Miss vs Hard-Fail ──────────────────────
            ax2 = fig.add_subplot(gs[1])
            chart._style_axis(ax2)
            total_failing = (near_miss or {}).get("total_failing", 0)
            if total_failing == 0:
                ax2.set_title("Failure Severity — last "
                              f"{self.selected_days}d",
                              loc="left", fontsize=12, fontweight="bold",
                              color="#ffffff")
                self._draw_empty_state(ax2, "No failing tracks in this window")
                ax2.set_xticks([])
                ax2.set_yticks([])
                for spine in ax2.spines.values():
                    spine.set_visible(False)
            else:
                buckets = near_miss["distribution"]
                labels = ["1-3 pts\n(near-miss)", "4-10 pts", "11-50 pts",
                          "50+ pts\n(hard-fail)"]
                values = [buckets.get("1-3 points", 0),
                          buckets.get("4-10 points", 0),
                          buckets.get("11-50 points", 0),
                          buckets.get("50+ points", 0)]
                colors = ["#198754", "#fd7e14", "#dc3545", "#6f42c1"]
                bars = ax2.bar(labels, values, color=colors,
                                edgecolor="#1a1a1a", linewidth=0.5)
                for bar, v in zip(bars, values):
                    pct = (v / total_failing) * 100 if total_failing else 0
                    ax2.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height(),
                             f"{v}\n({pct:.0f}%)", ha="center", va="bottom",
                             fontsize=9, color="#dddddd")
                near_pct = near_miss["near_miss_percent"]
                hard_pct = near_miss["hard_fail_percent"]
                ax2.set_title(
                    f"Failure Severity — {total_failing} failing tracks, "
                    f"{near_pct:.0f}% near-miss / {hard_pct:.0f}% hard-fail",
                    loc="left", fontsize=12, fontweight="bold",
                    color="#ffffff",
                )
                ax2.set_ylabel("Failing tracks")
                ax2.tick_params(axis="x", labelsize=9)
                ax2.grid(True, axis="y", alpha=0.2)
                # Headroom for the count+percent labels
                ax2.set_ylim(0, max(values) * 1.25 if max(values) else 1)

            # ─── Section 3: Cost Impact ─────────────────────────────────
            ax3 = fig.add_subplot(gs[2])
            chart._style_axis(ax3)
            with_price = [(p, pricing.get(p["model"])) for p in (priorities or [])
                          if pricing.get(p["model"])]
            if not with_price:
                ax3.set_title("Cost Impact", loc="left", fontsize=12,
                              fontweight="bold", color="#ffffff")
                self._draw_empty_state(
                    ax3,
                    "No model pricing configured.\n"
                    "Add prices in Settings → Active Models to see\n"
                    "estimated scrap cost per model.",
                )
                ax3.set_xticks([])
                ax3.set_yticks([])
                for spine in ax3.spines.values():
                    spine.set_visible(False)
            else:
                # Estimated cost = failed_units × price × cost_ratio
                cost_rows = [
                    {
                        "model": p["model"],
                        "failed": p["failed_units"],
                        "price": price,
                        "cost": p["failed_units"] * price * cost_ratio,
                        "near_miss": p.get("near_miss_count", 0),
                    }
                    for p, price in with_price
                    if p["failed_units"] > 0
                ]
                cost_rows.sort(key=lambda r: r["cost"], reverse=True)
                cost_rows = cost_rows[:n_cost]
                # Reverse so highest cost is at the top of the bar chart
                cost_rows = list(reversed(cost_rows))
                models = [r["model"] for r in cost_rows]
                costs = [r["cost"] for r in cost_rows]
                # Color bars by what fraction is near-miss (green = lots of
                # easy wins, red = mostly hard-fail / root cause work).
                colors_cost = []
                for r in cost_rows:
                    if r["failed"] == 0:
                        colors_cost.append("#888888")
                    else:
                        ratio = r["near_miss"] / r["failed"]
                        if ratio >= 0.5:
                            colors_cost.append("#198754")  # mostly easy wins
                        elif ratio >= 0.25:
                            colors_cost.append("#fd7e14")  # mixed
                        else:
                            colors_cost.append("#dc3545")  # mostly hard fail
                y_pos = list(range(len(models)))
                ax3.barh(y_pos, costs, color=colors_cost,
                          edgecolor="#1a1a1a", linewidth=0.5)
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(models, fontsize=9)
                for i, r in enumerate(cost_rows):
                    nm_pct = (
                        (r["near_miss"] / r["failed"]) * 100
                        if r["failed"] else 0
                    )
                    ax3.text(
                        r["cost"] * 1.01, i,
                        f"${r['cost']:,.0f}  ·  {r['failed']} fails  ·  "
                        f"{nm_pct:.0f}% near-miss",
                        va="center", fontsize=8, color="#cccccc",
                    )
                ax3.set_xlabel(
                    f"Est. scrap cost ($, last {self.selected_days}d, "
                    f"cost_ratio={cost_ratio:.2f})"
                )
                total_cost = sum(c for c in costs)
                ax3.set_title(
                    f"Cost Impact — top {len(cost_rows)} models  "
                    f"(${total_cost:,.0f} total)",
                    loc="left", fontsize=12, fontweight="bold",
                    color="#ffffff",
                )
                ax3.grid(True, axis="x", alpha=0.2)
                # Generous right margin for the annotation text
                ax3.set_xlim(0, max(costs) * 1.6 if costs else 1)

            try:
                fig.tight_layout()
            except Exception:
                pass
            chart.canvas.draw_idle()
            self.status_label.configure(
                text=(
                    f"Priorities — {len(priorities or [])} models analyzed, "
                    f"{(near_miss or {}).get('total_failing', 0)} fails"
                )
            )
        except Exception as e:
            logger.error(f"Priorities render error: {e}", exc_info=True)
            self.status_label.configure(text=f"Priorities error: {e}")

    def _show_trim_difficulty(self):
        """Show models ranked by how many laser-trim passes the equipment
        runs per unit on average. Higher = harder unit, more retrim work."""
        self.status_label.configure(text="Loading trim difficulty...")
        selected_days = self.selected_days

        def _load():
            try:
                db = get_database()
                rows = db.get_trim_difficulty_by_model(
                    days_back=selected_days, min_units=5, limit=25
                )
                self.after(0, lambda: self._render_trim_difficulty(rows))
            except Exception as e:
                logger.error(f"Trim difficulty error: {e}", exc_info=True)
                self.after(0, lambda exc=e: self.status_label.configure(
                    text=f"Trim difficulty error: {exc}"))

        get_thread_manager().start_thread(target=_load, name="trim-difficulty")

    def _render_trim_difficulty(self, rows):
        """Horizontal bar chart of avg trim passes per model, worst at top.
        Bars are colored by retrim rate (red = many units needed retrimming).
        Each bar is annotated with N units and max passes seen."""
        if not self.winfo_exists():
            return
        try:
            chart = self._create_dedicated_chart_view("Trim Difficulty by Model")
            chart.clear()
            fig = chart.figure
            ax = fig.add_subplot(111)
            chart._style_axis(ax)

            if not rows:
                self._draw_empty_state(
                    ax,
                    "No trim difficulty data yet.\n\n"
                    "trim_pass_count is captured at parse time —\n"
                    "process new files (or reanalyze existing ones)\n"
                    "to populate this view.",
                )
                chart.canvas.draw_idle()
                self.status_label.configure(text="No trim difficulty data")
                return

            # Reverse so highest avg sits at the top of the horizontal bar chart.
            rows_top_first = list(reversed(rows))
            models = [r["model"] for r in rows_top_first]
            avgs = [r["avg_passes"] for r in rows_top_first]
            retrim_rates = [r["retrim_rate"] for r in rows_top_first]

            # Color bars by retrim rate. 0% retrim = green, 50%+ = red.
            def _color(rate):
                if rate < 10:
                    return "#27ae60"   # green: easy
                if rate < 25:
                    return "#f1c40f"   # yellow: occasional retrim
                if rate < 50:
                    return "#e67e22"   # orange: frequent retrim
                return "#e74c3c"       # red: most units retrimmed

            colors = [_color(r) for r in retrim_rates]

            target_height = max(4.0, min(14.0, 1.5 + 0.35 * len(models)))
            fig.set_size_inches(fig.get_size_inches()[0], target_height, forward=True)

            y_pos = list(range(len(models)))
            ax.barh(y_pos, avgs, color=colors, edgecolor="#1a1a1a", linewidth=0.5)

            # Annotate each bar with "<N units> · max <M>" so the user gets
            # sample size and worst-case in one glance.
            for i, r in enumerate(rows_top_first):
                ax.text(
                    avgs[i] + 0.05,
                    i,
                    f"{r['count']} units · max {r['max_passes']} · "
                    f"retrim {r['retrim_rate']:.0f}%",
                    va="center",
                    fontsize=8,
                    color="#cccccc",
                )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(models, fontsize=9)
            ax.set_xlabel("Avg trim passes per unit")
            ax.set_title(
                f"Trim Difficulty by Model — last {self.selected_days} days "
                f"(top {len(models)} hardest)"
            )
            ax.axvline(x=1.0, linestyle="--", color="#888", linewidth=0.8, alpha=0.6)
            ax.grid(True, axis="x", alpha=0.2)
            try:
                fig.tight_layout()
            except Exception:
                pass
            chart.canvas.draw_idle()
            self.status_label.configure(
                text=f"Trim difficulty: {len(rows)} models ranked"
            )
        except Exception as e:
            logger.error(f"Trim difficulty render error: {e}", exc_info=True)
            self.status_label.configure(text=f"Trim difficulty error: {e}")

    def _populate_spec_filters(self):
        """Populate element type and product class filter dropdowns.

        DB queries run on a background thread to avoid freezing the UI during
        page navigation; widget updates are marshaled back to the main thread
        via self.after() because tkinter is not thread-safe.
        """
        def _load_spec_values():
            try:
                db = get_database()
                etypes = ["All"] + db.get_distinct_element_types()
                pclasses = ["All"] + db.get_distinct_product_classes()
                self.after(0, lambda: self._apply_spec_filter_values(etypes, pclasses))
            except Exception as e:
                logger.debug(f"Could not populate spec filters: {e}")

        get_thread_manager().start_thread(
            target=_load_spec_values,
            name="trends-populate-spec-filters",
        )

    def _apply_spec_filter_values(self, etypes, pclasses):
        """Main-thread callback to apply dropdown values."""
        try:
            self._element_filter.configure(values=etypes)
            self._class_filter.configure(values=pclasses)
        except Exception as e:
            logger.debug(f"Could not apply spec filter values: {e}")

    def _on_spec_filter_change(self, _=None):
        """Handle element type or product class filter change."""
        self._refresh_data()

    def on_hide(self):
        """Called when page becomes hidden."""
        # Don't destroy charts on hide — recreating 10 ChartWidgets on every
        # page visit is the #1 performance bottleneck. Instead, keep them alive
        # and just clear stale data. Charts are only destroyed on model switch
        # (via _create_summary_view / _create_detail_view) which already calls
        # _cleanup_charts internally.
        self.active_models_data = []
        self.model_trend_data = None

    def _refresh_drift_data(self):
        """Refresh drift detection data."""
        get_thread_manager().start_thread(target=self._load_drift_data, name="trends-load-drift")

    def _load_drift_data(self):
        """Load drift detection data in background."""
        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.ml import get_shared_ml_manager

            db = get_database()
            ml_manager = get_shared_ml_manager(db)

            # Get drift status for all models
            drift_status = ml_manager.get_drift_status()

            # Update UI on main thread
            self.after(0, lambda: self._update_drift_display(drift_status, ml_manager))

        except Exception as e:
            logger.error(f"Failed to load drift data: {e}")
            self.after(0, lambda: self._show_drift_error(str(e)))

    def _update_drift_display(self, drift_status: Dict[str, Dict[str, Any]], ml_manager):
        """Update drift detection display with data."""
        if not self.winfo_exists():
            return
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
                
                if direction == "up":
                    color = "#e74c3c"  # Red - quality degrading
                    status_text = "DRIFTING (up) ↑"
                else:  # down
                    color = "#f39c12"  # Yellow/Orange - quality improving
                    status_text = "DRIFTING (down) ↓"
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
                # Determine drift summary in plain language
                drift_exceeded = cusum_value > detector.cusum_h
                drift_summary = "Change detected" if drift_exceeded else "Within normal range"
                details_text = (
                    f"Drift Score: {cusum_value:.1f} / {detector.cusum_h:.1f} limit ({drift_summary})  |  "
                    f"Current Trend: {ewma_display}  |  "
                    f"Baseline Variation: {baseline_std:.4f}"
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
        if not self.winfo_exists():
            return
        for widget in self._drift_model_list.winfo_children():
            widget.destroy()

        error_label = ctk.CTkLabel(
            self._drift_model_list,
            text=f"Error: {error}",
            text_color="#e74c3c",
            font=ctk.CTkFont(size=10)
        )
        error_label.pack(padx=10, pady=20)

    def _export_summary_pdf(self):
        """Export summary view to PDF."""
        from tkinter import filedialog, messagebox
        from laser_trim_analyzer.export.trends_pdf import export_trends_summary_pdf

        if not self.active_models_data:
            logger.warning("No summary data to export")
            messagebox.showwarning("No Data", "No summary data available to export.")
            return

        # File dialog
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f"trends_summary_{timestamp}.pdf"

        file_path = filedialog.asksaveasfilename(
            title="Export Summary to PDF",
            defaultextension=".pdf",
            initialfile=default_name,
            initialdir=getattr(self.app.config, 'export_path', None),
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            export_trends_summary_pdf(self.active_models_data, Path(file_path))
            logger.info(f"Exported summary PDF to: {file_path}")
            messagebox.showinfo("Export Complete", f"PDF exported successfully to:\n{file_path}")
        except Exception as e:
            logger.error(f"Failed to export summary PDF: {e}", exc_info=True)
            messagebox.showerror("Export Failed", f"Failed to export PDF:\n{str(e)}")

    def _export_detail_pdf(self):
        """Export detail view to PDF."""
        from tkinter import filedialog, messagebox
        from laser_trim_analyzer.export.trends_pdf import export_trends_detail_pdf

        if not self.model_trend_data or not self.selected_model:
            logger.warning("No model data to export")
            messagebox.showwarning("No Data", "No model data available to export.")
            return

        # Gather stats from UI labels
        model_stats = {
            key: label.cget("text")
            for key, label in self.detail_stat_labels.items()
        }

        # File dialog
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_safe = self.selected_model.replace(" ", "_")
        default_name = f"trends_{model_safe}_{timestamp}.pdf"

        file_path = filedialog.asksaveasfilename(
            title="Export Model Trends to PDF",
            defaultextension=".pdf",
            initialfile=default_name,
            initialdir=getattr(self.app.config, 'export_path', None),
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            export_trends_detail_pdf(
                self.selected_model,
                self.model_trend_data,
                model_stats,
                Path(file_path)
            )
            logger.info(f"Exported detail PDF to: {file_path}")
            messagebox.showinfo("Export Complete", f"PDF exported successfully to:\n{file_path}")
        except Exception as e:
            logger.error(f"Failed to export detail PDF: {e}", exc_info=True)
            messagebox.showerror("Export Failed", f"Failed to export PDF:\n{str(e)}")
