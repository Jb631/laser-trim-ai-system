"""
Dashboard Page - Overview, alerts, quick stats.

The at-a-glance health overview page.
Wired to the database for real statistics.
"""

import customtkinter as ctk
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.utils.threads import get_thread_manager

# Lazy import for ChartWidget - defer matplotlib loading until first use
if TYPE_CHECKING:
    from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle

logger = logging.getLogger(__name__)


def _make_sparkline(values: List[float], length: int = 7) -> str:
    """Generate unicode sparkline from recent values.

    Uses block characters to show trend at a glance.
    """
    blocks = " ▁▂▃▄▅▆▇█"
    if not values:
        return ""
    recent = values[-length:]
    if len(recent) < 2:
        return ""
    v_min = min(recent)
    v_max = max(recent)
    v_range = v_max - v_min
    if v_range == 0:
        return blocks[4] * len(recent)  # Flat line at middle
    chars = []
    for v in recent:
        idx = int((v - v_min) / v_range * 7) + 1
        idx = max(1, min(8, idx))
        chars.append(blocks[idx])
    return "".join(chars)


class DashboardPage(ctk.CTkFrame):
    """
    Dashboard page showing:
    - Health score card (overall pass rate)
    - Database total (all files, date range, model count)
    - Last batch stats (recent processing run)
    - Recent alerts (top 5)
    - Quick action buttons
    - Pass rate trend chart
    - Top models by volume
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.stats: Optional[Dict[str, Any]] = None

        # Lazy chart initialization - defer matplotlib loading
        self._chart_initialized = False
        self.trend_chart = None

        self._create_ui()

    def _create_ui(self):
        """Create the dashboard UI."""
        # Configure grid: row 0=header, row 1=filters, row 2=attention, row 3=content
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # Header with refresh button
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        header_frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            header_frame,
            text="Dashboard",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.grid(row=0, column=0, sticky="w")

        export_btn = ctk.CTkButton(
            header_frame,
            text="Export Summary",
            width=120,
            fg_color="#2980b9",
            command=self._export_executive_summary
        )
        export_btn.grid(row=0, column=1, sticky="e", padx=(0, 10))

        refresh_btn = ctk.CTkButton(
            header_frame,
            text="⟳ Refresh",
            width=100,
            command=self._refresh_data
        )
        refresh_btn.grid(row=0, column=2, sticky="e")

        self.last_update_label = ctk.CTkLabel(
            header_frame,
            text="",
            text_color="gray",
            font=ctk.CTkFont(size=10)
        )
        self.last_update_label.grid(row=0, column=3, sticky="e", padx=(10, 0))

        # Filter row
        filter_frame = ctk.CTkFrame(self, fg_color="transparent")
        filter_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 5))

        ctk.CTkLabel(filter_frame, text="Filter:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 5))

        self._element_filter = ctk.CTkComboBox(
            filter_frame,
            values=["All Element Types"],
            command=self._on_filter_changed,
            width=160,
            state="readonly",
            font=ctk.CTkFont(size=11)
        )
        self._element_filter.pack(side="left", padx=5)
        self._element_filter.set("All Element Types")

        self._class_filter = ctk.CTkComboBox(
            filter_frame,
            values=["All Product Classes"],
            command=self._on_filter_changed,
            width=160,
            state="readonly",
            font=ctk.CTkFont(size=11)
        )
        self._class_filter.pack(side="left", padx=5)
        self._class_filter.set("All Product Classes")

        # Attention cards row (between filters and content)
        self._attention_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._attention_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 5))
        self._attention_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Content frame - use scrollable for smaller screens
        content = ctk.CTkScrollableFrame(self)
        content.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 20))
        content.grid_columnconfigure((0, 1, 2), weight=1, uniform="col")
        content.grid_rowconfigure(0, weight=0, minsize=120)  # Metric cards - fixed height
        content.grid_rowconfigure(1, weight=0, minsize=60)   # System/FT/Escape info - compact
        content.grid_rowconfigure(2, weight=1, minsize=200)  # Main row - expandable
        content.grid_rowconfigure(3, weight=0, minsize=100)  # Model breakdown - fixed

        # Linearity Quality Card (primary metric)
        self.health_card = self._create_metric_card(content)
        self.health_card.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self._update_card(
            self.health_card,
            title="Linearity Quality",
            value="--",
            subtitle="Loading...",
            color="gray"
        )

        # Files Processed Card
        self.files_card = self._create_metric_card(content)
        self.files_card.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self._update_card(
            self.files_card,
            title="Files Processed",
            value="0",
            subtitle="Loading...",
            color="gray"
        )

        # Sigma Process Health Card (leading indicator)
        self.batch_card = self._create_metric_card(content)
        self.batch_card.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self._update_card(
            self.batch_card,
            title="Sigma Process Health",
            value="--",
            subtitle="Loading...",
            color="gray"
        )

        # System Comparison / FT / Escape-Overkill info row (compact)
        self.system_info_frame = ctk.CTkFrame(content)
        self.system_info_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=(5, 5), sticky="ew")

        info_header = ctk.CTkLabel(
            self.system_info_frame,
            text="System Comparison",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        info_header.pack(padx=15, pady=(10, 2), anchor="w")

        self.system_a_label = ctk.CTkLabel(
            self.system_info_frame, text="System A: Loading...",
            text_color="gray", font=ctk.CTkFont(size=12), anchor="w"
        )
        self.system_a_label.pack(padx=15, pady=0, anchor="w", fill="x")

        self.system_b_label = ctk.CTkLabel(
            self.system_info_frame, text="System B: Loading...",
            text_color="gray", font=ctk.CTkFont(size=12), anchor="w"
        )
        self.system_b_label.pack(padx=15, pady=0, anchor="w", fill="x")

        self.ft_info_label = ctk.CTkLabel(
            self.system_info_frame, text="Final Test: Loading...",
            text_color="gray", font=ctk.CTkFont(size=12), anchor="w"
        )
        self.ft_info_label.pack(padx=15, pady=(4, 0), anchor="w", fill="x")

        self.escape_info_label = ctk.CTkLabel(
            self.system_info_frame, text="Prediction Accuracy: Loading...",
            text_color="gray", font=ctk.CTkFont(size=12), anchor="w"
        )
        self.escape_info_label.pack(padx=15, pady=0, anchor="w", fill="x")

        self.near_miss_label = ctk.CTkLabel(
            self.system_info_frame, text="Near-Miss: Loading...",
            text_color="gray", font=ctk.CTkFont(size=12), anchor="w"
        )
        self.near_miss_label.pack(padx=15, pady=(4, 0), anchor="w", fill="x")

        self.cost_impact_label = ctk.CTkLabel(
            self.system_info_frame, text="",
            text_color="gray", font=ctk.CTkFont(size=12), anchor="w"
        )
        self.cost_impact_label.pack(padx=15, pady=(4, 10), anchor="w", fill="x")

        # Row 2: P-chart trend — FULL WIDTH (3 columns) for readability
        chart_frame = ctk.CTkFrame(content)
        chart_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        chart_label = ctk.CTkLabel(
            chart_frame,
            text="Linearity Pass Rate Trend (90 Days)",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        chart_label.pack(padx=15, pady=(15, 5), anchor="w")

        # Placeholder frame for chart - actual ChartWidget created lazily on first show
        self._chart_frame = chart_frame
        self._chart_placeholder = ctk.CTkLabel(
            chart_frame,
            text="Loading trend data...",
            text_color="gray"
        )
        self._chart_placeholder.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Row 3: [Alerts+Drift | Pareto chart | Where to Focus]
        content.grid_rowconfigure(3, weight=1, minsize=250)

        # Alerts + Drift (column 0) — stacked vertically
        alerts_container = ctk.CTkFrame(content, fg_color="transparent")
        alerts_container.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        alerts_container.grid_columnconfigure(0, weight=1)
        alerts_container.grid_rowconfigure(0, weight=1)
        alerts_container.grid_rowconfigure(1, weight=1)

        # Recent Alerts Card
        self.alerts_frame = ctk.CTkFrame(alerts_container)
        self.alerts_frame.grid(row=0, column=0, padx=0, pady=(0, 5), sticky="nsew")
        self.alerts_frame.grid_columnconfigure(0, weight=1)
        self.alerts_frame.grid_rowconfigure(1, weight=1)

        alerts_label = ctk.CTkLabel(
            self.alerts_frame,
            text="Recent Alerts",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        alerts_label.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="w")

        self.alerts_list = ctk.CTkTextbox(self.alerts_frame, height=60)
        self.alerts_list.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 10))
        self.alerts_list.configure(state="disabled")
        self._update_alerts_display([])

        # Drift Alerts Card (from ML system)
        self.drift_frame = ctk.CTkFrame(alerts_container)
        self.drift_frame.grid(row=1, column=0, padx=0, pady=(5, 0), sticky="nsew")
        self.drift_frame.grid_columnconfigure(0, weight=1)
        self.drift_frame.grid_rowconfigure(1, weight=1)

        drift_label = ctk.CTkLabel(
            self.drift_frame,
            text="ML Drift Status",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        drift_label.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="w")

        self.drift_list = ctk.CTkTextbox(self.drift_frame, height=60)
        self.drift_list.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 10))
        self.drift_list.configure(state="disabled")
        self._update_drift_display([])

        # Pareto chart (column 1)
        self._pareto_frame = ctk.CTkFrame(content)
        self._pareto_frame.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")
        pareto_label = ctk.CTkLabel(
            self._pareto_frame, text="Failure Pareto",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        pareto_label.pack(padx=15, pady=(15, 5), anchor="w")
        self._pareto_placeholder = ctk.CTkLabel(
            self._pareto_frame, text="Loading...", text_color="gray"
        )
        self._pareto_placeholder.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.pareto_chart = None
        self.confusion_chart = None  # Not used as chart — text summary in system info row
        self.scatter_chart = None    # Not used as chart — data in system info row

        # Where to Focus panel (column 2)
        self.model_frame = ctk.CTkFrame(content)
        self.model_frame.grid(row=3, column=2, padx=10, pady=10, sticky="nsew")

        model_label = ctk.CTkLabel(
            self.model_frame,
            text="Where to Focus",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        model_label.pack(padx=15, pady=(15, 5), anchor="w")

        # Container for model focus cards (replaces raw textbox)
        self._focus_container = ctk.CTkScrollableFrame(
            self.model_frame, fg_color="transparent"
        )
        self._focus_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._focus_container.grid_columnconfigure(0, weight=1)

        # Keep model_text for backwards compat — hidden, used only if focus cards fail
        self.model_text = ctk.CTkTextbox(self.model_frame, height=0)
        self.model_text.configure(state="disabled")

        # Row 4: Performance by Element Type and Product Class
        content.grid_rowconfigure(4, weight=0, minsize=200)

        self._breakdown_frame = ctk.CTkFrame(content)
        self._breakdown_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        self._breakdown_frame.grid_columnconfigure((0, 1), weight=1, uniform="bk")
        self._breakdown_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self._breakdown_frame,
            text="Performance by Category (90 Days)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=15, pady=(15, 5), sticky="w")

        # Element type breakdown placeholder
        self._element_breakdown_frame = ctk.CTkFrame(self._breakdown_frame, fg_color="transparent")
        self._element_breakdown_frame.grid(row=1, column=0, padx=10, pady=(0, 15), sticky="nsew")
        self._element_breakdown_label = ctk.CTkLabel(
            self._element_breakdown_frame, text="Loading element type data...",
            text_color="gray"
        )
        self._element_breakdown_label.pack(fill="both", expand=True)
        self._element_chart = None

        # Product class breakdown placeholder
        self._class_breakdown_frame = ctk.CTkFrame(self._breakdown_frame, fg_color="transparent")
        self._class_breakdown_frame.grid(row=1, column=1, padx=10, pady=(0, 15), sticky="nsew")
        self._class_breakdown_label = ctk.CTkLabel(
            self._class_breakdown_frame, text="Loading product class data...",
            text_color="gray"
        )
        self._class_breakdown_label.pack(fill="both", expand=True)

        # Row 5: Process Health section (Cpk summary + failure modes)
        content.grid_rowconfigure(5, weight=0, minsize=120)
        process_frame = ctk.CTkFrame(content)
        process_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        process_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(
            process_frame, text="Process Health",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=15, pady=(10, 5), sticky="w")

        # Cpk summary (left)
        cpk_frame = ctk.CTkFrame(process_frame)
        cpk_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        ctk.CTkLabel(cpk_frame, text="Process Capability (Cpk)",
                    font=ctk.CTkFont(size=13, weight="bold")).pack(padx=10, pady=(10, 5), anchor="w")
        self._cpk_summary_label = ctk.CTkLabel(
            cpk_frame, text="Loading...", font=ctk.CTkFont(size=11), justify="left",
        )
        self._cpk_summary_label.pack(padx=10, pady=(0, 10), anchor="w")

        # Failure mode summary (right) - clickable to jump to Trends > Drift tab
        fail_frame = ctk.CTkFrame(process_frame, cursor="hand2")
        fail_frame.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="nsew")
        drift_title = ctk.CTkLabel(
            fail_frame, text="Drift Alerts  \u2192",
            font=ctk.CTkFont(size=13, weight="bold"),
            cursor="hand2",
        )
        drift_title.pack(padx=10, pady=(10, 5), anchor="w")
        self._drift_summary_label = ctk.CTkLabel(
            fail_frame, text="Loading...", font=ctk.CTkFont(size=11), justify="left",
            cursor="hand2",
        )
        self._drift_summary_label.pack(padx=10, pady=(0, 10), anchor="w")

        # Make the whole card clickable - navigate to Trends page so the
        # user can see the drift timeline (V5 reported the card "did nothing").
        def _open_drift_view(_event=None):
            try:
                self.app._show_page("trends")
                # If trends has a "show_drift" hook, jump straight to the Drift tab
                trends_page = self.app._pages.get("trends")
                if trends_page is not None and hasattr(trends_page, "show_drift_tab"):
                    trends_page.show_drift_tab()
            except Exception as e:
                logger.debug(f"Drift card navigation failed: {e}")

        for w in (fail_frame, drift_title, self._drift_summary_label):
            w.bind("<Button-1>", _open_drift_view)

    def _create_metric_card(self, parent) -> ctk.CTkFrame:
        """Create a metric card frame."""
        card = ctk.CTkFrame(parent)

        # Title label
        title_label = ctk.CTkLabel(
            card,
            text="",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        title_label.pack(padx=15, pady=(15, 5), anchor="w")
        card.title_label = title_label

        # Value label
        value_label = ctk.CTkLabel(
            card,
            text="",
            font=ctk.CTkFont(size=36, weight="bold")
        )
        value_label.pack(padx=15, anchor="w")
        card.value_label = value_label

        # Subtitle label
        subtitle_label = ctk.CTkLabel(
            card,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        subtitle_label.pack(padx=15, pady=(0, 15), anchor="w")
        card.subtitle_label = subtitle_label

        return card

    def _update_card(
        self, card: ctk.CTkFrame, title: str, value: str,
        subtitle: str, color: str = "white"
    ):
        """Update a metric card."""
        card.title_label.configure(text=title)
        card.value_label.configure(text=value, text_color=color)
        card.subtitle_label.configure(text=subtitle)

    def _ensure_chart_initialized(self):
        """Lazily initialize ChartWidget on first use - defers matplotlib loading."""
        if self._chart_initialized:
            return

        try:
            # Import matplotlib-dependent ChartWidget only when needed
            from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle

            # Remove placeholder
            self._chart_placeholder.destroy()

            # Create actual chart widget
            self.trend_chart = ChartWidget(
                self._chart_frame,
                style=ChartStyle(figure_size=(10, 3), dpi=80)
            )
            self.trend_chart.pack(fill="both", expand=True, padx=15, pady=(0, 15))
            self.trend_chart.show_placeholder("Loading trend data...")

            self._chart_initialized = True
            logger.debug("ChartWidget initialized (matplotlib loaded)")
        except Exception as e:
            logger.error(f"Failed to initialize chart (matplotlib issue?): {e}")
            self._chart_initialized = True  # Don't retry on every refresh

    def _ensure_pareto_chart_initialized(self):
        """Lazily initialize Pareto chart."""
        if self.pareto_chart is not None or getattr(self, '_pareto_init_failed', False):
            return

        try:
            from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle

            self._pareto_placeholder.destroy()
            self.pareto_chart = ChartWidget(
                self._pareto_frame,
                style=ChartStyle(figure_size=(5, 3), dpi=80)
            )
            self.pareto_chart.pack(fill="both", expand=True, padx=15, pady=(0, 15))
            self.pareto_chart.show_placeholder("Loading Pareto data...")
        except Exception as e:
            logger.error(f"Failed to initialize Pareto chart: {e}")
            self._pareto_init_failed = True  # Don't retry on every refresh

    def _refresh_data(self):
        """Refresh dashboard data in background thread."""
        self.last_update_label.configure(text="Refreshing...")
        # Capture filter values on main thread (tkinter is not thread-safe)
        element_type, product_class = self._get_active_filters()
        get_thread_manager().start_thread(
            target=self._load_data,
            args=(element_type, product_class),
            name="dashboard-load-data"
        )

    def _get_active_filters(self):
        """Get current filter selections (None if 'All'). Must be called from main thread."""
        et = self._element_filter.get() if hasattr(self, '_element_filter') else None
        pc = self._class_filter.get() if hasattr(self, '_class_filter') else None
        element_type = et if et and et != "All Element Types" else None
        product_class = pc if pc and pc != "All Product Classes" else None
        return element_type, product_class

    def _load_data(self, element_type=None, product_class=None):
        """Load data from database in background."""
        try:
            db = get_database()
            stats = db.get_dashboard_stats(
                days_back=90,
                element_type=element_type,
                product_class=product_class
            )
            self.stats = stats

            # Get last batch stats
            batch_stats = db.get_last_batch_stats()

            # Get overall stats
            overall_stats = db.get_overall_stats()

            # Also get recent alerts
            alerts = db.get_alerts(limit=5)

            # Get impact-ranked model prioritization (linearity-focused)
            try:
                priority_models = db.get_linearity_prioritization(days_back=90, min_samples=10)
            except Exception as e:
                logger.warning(f"Could not load prioritization: {e}")
                priority_models = []

            # Get ML drift status
            drift_alerts = self._get_drift_alerts(db)

            # Get system comparison, FT stats, and escape/overkill data
            try:
                system_comparison = db.get_system_comparison(days_back=90)
            except Exception as e:
                logger.warning(f"Could not load system comparison: {e}")
                system_comparison = None

            try:
                ft_stats = db.get_ft_dashboard_stats(days_back=90)
            except Exception as e:
                logger.warning(f"Could not load FT stats: {e}")
                ft_stats = None

            try:
                escape_stats = db.get_escape_overkill_analysis(days_back=90)
            except Exception as e:
                logger.warning(f"Could not load escape/overkill: {e}")
                escape_stats = None

            # Get trending worse for model display markers
            try:
                trending_worse = db.get_trending_worse_models(days_back=90, min_samples=20)
            except Exception as e:
                logger.warning(f"Could not load trending worse: {e}")
                trending_worse = []

            # Get near-miss summary
            try:
                near_miss_data = db.get_near_miss_summary(days_back=90)
            except Exception as e:
                logger.warning(f"Could not load near-miss data: {e}")
                near_miss_data = None

            # Get performance by element type and product class
            try:
                element_breakdown = db.get_pass_rate_by_category("element_type", days_back=90)
            except Exception as e:
                logger.warning(f"Could not load element breakdown: {e}")
                element_breakdown = []
            try:
                class_breakdown = db.get_pass_rate_by_category("product_class", days_back=90)
            except Exception as e:
                logger.warning(f"Could not load class breakdown: {e}")
                class_breakdown = []

            # Load Cpk, drift, and failure mode data in background (not on main thread)
            try:
                cpk_data = db.get_cpk_by_model(days_back=90)
            except Exception:
                cpk_data = []
            try:
                drift_events = db.get_drift_events_timeline(days_back=90)
            except Exception:
                drift_events = []
            try:
                failure_modes = db.get_failure_mode_summary(days_back=90)
            except Exception:
                failure_modes = []

            # Update UI on main thread
            self.after(0, lambda: self._update_display(
                stats, alerts, priority_models, drift_alerts, batch_stats, overall_stats,
                system_comparison, ft_stats, escape_stats, trending_worse, near_miss_data,
                element_breakdown, class_breakdown,
                cpk_data, drift_events, failure_modes
            ))

        except Exception as e:
            logger.error(f"Failed to load dashboard data: {e}")
            error_msg = str(e)
            self.after(0, lambda err=error_msg: self._show_error(err))

    def _get_drift_alerts(self, db) -> List[Dict[str, Any]]:
        """Get drift status from ML system."""
        try:
            from laser_trim_analyzer.ml import MLManager

            ml_manager = MLManager(db)
            ml_manager.load_all()

            drift_alerts = []
            for model, detector in ml_manager.drift_detectors.items():
                if detector.has_baseline and detector.is_drifting:
                    direction = detector.drift_direction.value if detector.drift_direction else "unknown"
                    drift_alerts.append({
                        "model": model,
                        "status": "Drifting",
                        "direction": direction,
                        "severity": "WARNING"
                    })

            # Sort by model name
            drift_alerts.sort(key=lambda x: x["model"])
            return drift_alerts

        except Exception as e:
            logger.debug(f"Could not get drift alerts: {e}")
            return []

    def _update_display(
        self,
        stats: Dict[str, Any],
        alerts: List[Dict[str, Any]],
        priority_models: List[Dict[str, Any]],
        drift_alerts: Optional[List[Dict[str, Any]]] = None,
        batch_stats: Optional[Dict[str, Any]] = None,
        overall_stats: Optional[Dict[str, Any]] = None,
        system_comparison: Optional[Dict[str, Any]] = None,
        ft_stats: Optional[Dict[str, Any]] = None,
        escape_stats: Optional[Dict[str, Any]] = None,
        trending_worse: Optional[List[Dict[str, Any]]] = None,
        near_miss_data: Optional[Dict[str, Any]] = None,
        element_breakdown: Optional[List[Dict[str, Any]]] = None,
        class_breakdown: Optional[List[Dict[str, Any]]] = None,
        cpk_data: Optional[List[Dict[str, Any]]] = None,
        drift_events: Optional[List[Dict[str, Any]]] = None,
        failure_modes: Optional[List[Dict[str, Any]]] = None,
    ):
        """Update display with loaded data."""
        if not self.winfo_exists():
            return
        # Use overall_stats for the Files Processed card (all-time totals)
        if overall_stats and overall_stats.get("total_files", 0) > 0:
            total_files = overall_stats.get("total_files", 0)
            passed = overall_stats.get("passed", 0)
            warnings = overall_stats.get("warnings", 0)
            failed = overall_stats.get("failed", 0)
            unique_models = overall_stats.get("unique_models", 0)
        else:
            total_files = stats.get("total_files", 0)
            passed = stats.get("passed", 0)
            warnings = 0
            failed = stats.get("failed", 0)
            unique_models = 0

        # Use 90-day stats for quality metric cards (consistent with sparklines)
        linearity_pass_rate = stats.get("linearity_pass_rate", 0) or 0.0
        sigma_pass_rate = stats.get("sigma_pass_rate", 0) or 0.0
        pass_rate = stats.get("pass_rate", 0) or 0.0

        # Build sparklines from daily trend data
        lin_trend = stats.get("linearity_daily_trend", []) or []
        lin_trend = [d for d in lin_trend if d.get("total", 0) > 0]
        lin_sparkline = _make_sparkline([d.get("pass_rate", 0) for d in lin_trend])

        # Linearity quality card with realistic thresholds
        if linearity_pass_rate >= 80:
            health_color = "#27ae60"  # Green
        elif linearity_pass_rate >= 65:
            health_color = "#2ecc71"  # Light green
        elif linearity_pass_rate >= 50:
            health_color = "#f39c12"  # Orange
        else:
            health_color = "#e74c3c"  # Red

        self._update_card(
            self.health_card,
            title="Linearity Quality",
            value=f"{linearity_pass_rate:.1f}%",
            subtitle=f"Sigma: {sigma_pass_rate:.1f}% | Overall: {pass_rate:.1f}%  {lin_sparkline}",
            color=health_color
        )

        # Update files card with overall stats
        self._update_card(
            self.files_card,
            title="Database Total",
            value=f"{total_files:,}",
            subtitle=f"✓{passed:,} ⚠{warnings:,} ✗{failed:,} | {unique_models} models",
            color="white"
        )

        # Sigma Process Health card (leading indicator)
        if sigma_pass_rate >= 80:
            sigma_color = "#27ae60"  # Green
        elif sigma_pass_rate >= 60:
            sigma_color = "#f39c12"  # Orange
        else:
            sigma_color = "#e74c3c"  # Red

        self._update_card(
            self.batch_card,
            title="Sigma Process Health",
            value=f"{sigma_pass_rate:.1f}%",
            subtitle="Leading indicator — watch for trends",
            color=sigma_color
        )

        # Update system comparison / FT / escape info
        self._update_system_ft_display(
            system_comparison, ft_stats, escape_stats, near_miss_data, priority_models
        )

        # Update alerts
        self._update_alerts_display(alerts)

        # Update drift alerts
        self._update_drift_display(drift_alerts or [])

        # Update trend chart with linearity trend
        self._update_trend_chart(stats)

        # Update model prioritization display
        self._update_model_display(priority_models, trending_worse)

        # Update Pareto chart
        self._update_pareto_chart(priority_models)

        # Update category breakdown charts
        self._update_breakdown_charts(element_breakdown or [], class_breakdown or [])

        # Update attention cards with priority models (using pre-loaded data)
        self._update_attention_cards(priority_models, cpk_data or [], drift_events or [])

        # Update process health (Cpk + drift) using pre-loaded data
        self._update_process_health(cpk_data or [], drift_events or [], failure_modes or [])

        # Update timestamp (V5-15: include date for clarity)
        self.last_update_label.configure(
            text=f"Updated: {datetime.now().strftime('%m/%d/%Y %H:%M')}"
        )

        logger.debug("Dashboard data refreshed")

    def _update_system_ft_display(
        self,
        system_comparison: Optional[Dict[str, Any]],
        ft_stats: Optional[Dict[str, Any]],
        escape_stats: Optional[Dict[str, Any]],
        near_miss_data: Optional[Dict[str, Any]] = None,
        priority_models: Optional[List[Dict[str, Any]]] = None,
    ):
        """Update the System A/B, Final Test, Escape/Overkill, Near-Miss, and Cost info labels."""
        # System A
        try:
            if system_comparison and "system_a" in system_comparison:
                a = system_comparison["system_a"]
                self.system_a_label.configure(
                    text=f"System A: {a.get('linearity_pass_rate', 0):.1f}% linearity | "
                         f"{a.get('sigma_pass_rate', 0):.1f}% sigma | "
                         f"{a.get('total_files', 0):,} files",
                    text_color=("gray10", "gray90")
                )
            else:
                self.system_a_label.configure(text="System A: No data", text_color="gray")
        except Exception as e:
            logger.debug(f"System A display error: {e}")
            self.system_a_label.configure(text="System A: No data", text_color="gray")

        # System B
        try:
            if system_comparison and "system_b" in system_comparison:
                b = system_comparison["system_b"]
                self.system_b_label.configure(
                    text=f"System B: {b.get('linearity_pass_rate', 0):.1f}% linearity | "
                         f"{b.get('sigma_pass_rate', 0):.1f}% sigma | "
                         f"{b.get('total_files', 0):,} files",
                    text_color=("gray10", "gray90")
                )
            else:
                self.system_b_label.configure(text="System B: No data", text_color="gray")
        except Exception as e:
            logger.debug(f"System B display error: {e}")
            self.system_b_label.configure(text="System B: No data", text_color="gray")

        # Final Test
        try:
            if ft_stats and ft_stats.get("total", 0) > 0:
                ft = ft_stats
                self.ft_info_label.configure(
                    text=f"Final Test: {ft.get('linearity_pass_rate', 0):.1f}% linearity pass | "
                         f"{ft.get('total', 0):,} tests | "
                         f"{ft.get('link_rate', 0):.0f}% linked to trim",
                    text_color=("gray10", "gray90")
                )
            else:
                self.ft_info_label.configure(text="Final Test: No FT data", text_color="gray")
        except Exception as e:
            logger.debug(f"FT display error: {e}")
            self.ft_info_label.configure(text="Final Test: No data", text_color="gray")

        # Escape / Overkill
        try:
            if escape_stats and escape_stats.get("total_linked", 0) > 0:
                esc = escape_stats
                self.escape_info_label.configure(
                    text=f"Prediction Accuracy: {esc.get('agreement_rate', 0):.1f}% agreement | "
                         f"{esc.get('escape_rate', 0):.1f}% escapes ({esc.get('escapes', 0)}) | "
                         f"{esc.get('overkill_rate', 0):.1f}% overkill ({esc.get('overkills', 0)})",
                    text_color=("gray10", "gray90")
                )
            else:
                self.escape_info_label.configure(
                    text="Prediction Accuracy: No linked trim/FT data", text_color="gray"
                )
        except Exception as e:
            logger.debug(f"Escape display error: {e}")
            self.escape_info_label.configure(text="Prediction Accuracy: No data", text_color="gray")

        # Near-Miss Analysis
        try:
            if near_miss_data and near_miss_data.get("total_failing", 0) > 0:
                nm = near_miss_data
                nm_color = "#e74c3c" if nm["near_miss_percent"] > 40 else "#f39c12" if nm["near_miss_percent"] > 20 else "gray"
                self.near_miss_label.configure(
                    text=f"Near-Miss (90d): {nm['near_miss_percent']:.0f}% of failures are near-miss "
                         f"({nm['near_miss_count']}/{nm['total_failing']} have 1-3 fail points) | "
                         f"{nm['hard_fail_percent']:.0f}% hard fail ({nm['hard_fail_count']})",
                    text_color=nm_color
                )
            else:
                self.near_miss_label.configure(
                    text="Near-Miss: No failure data", text_color="gray"
                )
        except Exception as e:
            logger.debug(f"Near-miss display error: {e}")
            self.near_miss_label.configure(text="Near-Miss: No data", text_color="gray")

        # Cost Impact Summary (requires pricing data)
        try:
            prices = self.app.config.active_models.model_prices
            cost_ratio = self.app.config.active_models.cost_ratio
            if prices and priority_models:
                total_cost = 0
                priced_failures = 0
                for m in priority_models:
                    model = m.get("model", "")
                    failed = m.get("failed_units", 0)
                    price = prices.get(model, 0)
                    if price > 0 and failed > 0:
                        total_cost += failed * price * cost_ratio
                        priced_failures += failed

                if total_cost > 0:
                    monthly_est = total_cost / 3  # 90-day data → monthly estimate
                    self.cost_impact_label.configure(
                        text=f"Cost Impact (90d): ${total_cost:,.0f} estimated scrap cost "
                             f"({priced_failures} priced failures) | "
                             f"~${monthly_est:,.0f}/month | "
                             f"Cost ratio: {cost_ratio:.0%}",
                        text_color="#e74c3c"
                    )
                else:
                    self.cost_impact_label.configure(
                        text="Cost Impact: No priced failures in period", text_color="gray"
                    )
            else:
                self.cost_impact_label.configure(
                    text="Cost Impact: No model pricing configured (set via config file)", text_color="gray"
                )
        except Exception as e:
            logger.debug(f"Cost impact display error: {e}")
            self.cost_impact_label.configure(text="", text_color="gray")

    def _update_alerts_display(self, alerts: List[Dict[str, Any]]):
        """Update alerts list (deduplicated by message)."""
        self.alerts_list.configure(state="normal")
        self.alerts_list.delete("1.0", "end")

        if not alerts:
            self.alerts_list.insert("end", "No active alerts - everything looks good!")
        else:
            seen_messages = set()
            for alert in alerts:
                severity = alert.get("severity", "INFO")
                message = alert.get("message", "")
                timestamp = alert.get("created_at", "")

                # Deduplicate by message content
                if message in seen_messages:
                    continue
                seen_messages.add(message)

                icon = "⚠️" if severity == "WARNING" else "🔴" if severity == "CRITICAL" else "ℹ️"
                self.alerts_list.insert(
                    "end",
                    f"{icon} [{severity}] {message}\n   {timestamp}\n\n"
                )

        self.alerts_list.configure(state="disabled")

    def _update_drift_display(self, drift_alerts: List[Dict[str, Any]]):
        """Update drift alerts display from ML system."""
        self.drift_list.configure(state="normal")
        self.drift_list.delete("1.0", "end")

        if not drift_alerts:
            self.drift_list.insert("end", "No drift detected - all models stable.\n")
            self.drift_list.insert("end", "(Train models in Settings to enable drift detection)")
        else:
            self.drift_list.insert("end", f"{len(drift_alerts)} model(s) drifting:\n")
            for alert in drift_alerts[:5]:  # Limit to 5
                model = alert.get("model", "Unknown")
                direction = alert.get("direction", "")
                direction_text = f" ({direction})" if direction else ""
                self.drift_list.insert("end", f"  ⚠ {model}{direction_text}\n")

        self.drift_list.configure(state="disabled")

    def _update_trend_chart(self, stats: Dict[str, Any]):
        """Update trend chart with linearity pass rate data."""
        # Ensure chart is initialized before use
        self._ensure_chart_initialized()

        if not self.trend_chart:
            return  # matplotlib failed to load

        # Use linearity-specific daily trend, fall back to overall
        trend_data = stats.get("linearity_daily_trend", []) or stats.get("daily_trend", [])

        if not trend_data:
            self.trend_chart.show_placeholder("No trend data available")
            return

        # Filter to only days with actual data (skip zero-total days)
        trend_data = [d for d in trend_data if d.get("total", 0) > 0]

        if not trend_data:
            self.trend_chart.show_placeholder("No data in selected period")
            return

        # Extract dates, pass rates, and sample sizes
        dates = [d.get("date", "") for d in trend_data]
        pass_rates = [d.get("pass_rate", 0) for d in trend_data]
        sample_sizes = [d.get("total", 1) for d in trend_data]

        if len(pass_rates) < 2:
            self.trend_chart.show_placeholder("Insufficient data for trend (need 2+ days)")
            return

        # Add data-as-of label using the latest date in the data
        latest_date = dates[-1] if dates else ""
        chart_title = f"Data as of: {latest_date}" if latest_date else ""

        # Plot P-chart with variable binomial control limits
        self.trend_chart.plot_pchart(
            dates=dates,
            pass_rates=pass_rates,
            sample_sizes=sample_sizes,
            title=chart_title,
            ylabel="Pass Rate %"
        )

    def _update_model_display(
        self,
        priority_models: List[Dict[str, Any]],
        trending_worse: Optional[List[Dict[str, Any]]] = None,
    ):
        """Update model focus panel with structured, color-coded entries."""
        # Clear existing focus cards
        for widget in self._focus_container.winfo_children():
            widget.destroy()

        # Build set of declining model names for quick lookup
        declining_models = set()
        if trending_worse:
            for tw in trending_worse:
                declining_models.add(tw.get("model", ""))

        if not priority_models:
            ctk.CTkLabel(
                self._focus_container,
                text="No model data available\n(need 10+ samples per model)",
                text_color="gray",
                font=ctk.CTkFont(size=11),
            ).grid(row=0, column=0, padx=10, pady=20)
            return

        # Top 5 models only — focus attention
        for i, m in enumerate(priority_models[:5]):
            model = m.get("model", "Unknown")
            lin_rate = m.get("linearity_pass_rate", 0)
            failed = m.get("failed_units", 0)
            near_miss = m.get("near_miss_count", 0)
            total = m.get("total_tracks", 0)
            rec = m.get("recommendation", "")
            is_declining = model in declining_models

            # Color by severity: red for high-impact, orange for medium, muted for lower
            if lin_rate < 50:
                border_color = "#c0392b"  # Dark red
            elif lin_rate < 70:
                border_color = "#e74c3c"  # Red
            elif lin_rate < 85:
                border_color = "#f39c12"  # Orange
            else:
                border_color = "#27ae60"  # Green (improving)

            # Mini card for each model
            card = ctk.CTkFrame(self._focus_container, border_width=2,
                                border_color=border_color)
            card.grid(row=i, column=0, padx=2, pady=3, sticky="ew")
            card.grid_columnconfigure(1, weight=1)

            # Rank badge
            rank_label = ctk.CTkLabel(
                card, text=f"#{i+1}",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=border_color, width=30,
            )
            rank_label.grid(row=0, column=0, rowspan=2, padx=(8, 4), pady=5)

            # Model name + declining tag
            name_text = model
            name_label = ctk.CTkLabel(
                card, text=name_text,
                font=ctk.CTkFont(size=13, weight="bold"),
                anchor="w",
            )
            name_label.grid(row=0, column=1, padx=4, pady=(5, 0), sticky="w")

            # Declining tag — visually prominent
            if is_declining:
                decline_label = ctk.CTkLabel(
                    card, text=" DECLINING ",
                    font=ctk.CTkFont(size=9, weight="bold"),
                    text_color="white", fg_color="#c0392b",
                    corner_radius=4,
                )
                decline_label.grid(row=0, column=2, padx=(4, 8), pady=(5, 0), sticky="e")

            # Stats line
            stats_text = f"Pass: {lin_rate:.0f}%  |  {failed} fail / {total} total  |  {near_miss} near-miss"
            stats_label = ctk.CTkLabel(
                card, text=stats_text,
                font=ctk.CTkFont(size=10),
                text_color="gray", anchor="w",
            )
            stats_label.grid(row=1, column=1, columnspan=2, padx=4, pady=(0, 5), sticky="w")

    def _update_pareto_chart(self, priority_models: List[Dict[str, Any]]):
        """Update Pareto chart with cost-weighted failure data when pricing available."""
        try:
            self._ensure_pareto_chart_initialized()
            if not self.pareto_chart:
                return

            if not priority_models:
                self.pareto_chart.show_placeholder("No failure data for Pareto chart")
                return

            # Get pricing data for cost-weighted Pareto
            prices = self.app.config.active_models.model_prices
            cost_ratio = self.app.config.active_models.cost_ratio

            failing_models = [m for m in priority_models if m.get("failed_units", 0) > 0]
            if not failing_models:
                self.pareto_chart.show_placeholder("No failures to display")
                return

            if prices:
                # Cost-weighted Pareto: failure_count * unit_price * cost_ratio
                labels = []
                values = []
                for m in failing_models:
                    model = m.get("model", "?")
                    failed = m.get("failed_units", 0)
                    price = prices.get(model, 0)
                    cost = failed * price * cost_ratio
                    if cost > 0:
                        labels.append(model)
                        values.append(cost)
                    elif price == 0:
                        # No price — still include by failure count (as $0)
                        labels.append(model)
                        values.append(0)

                if any(v > 0 for v in values):
                    # Format as $K for readability
                    title = "Failure Cost Impact ($)"
                    self.pareto_chart.plot_pareto(labels=labels, values=values, title=title)
                else:
                    # All prices are 0 — fall back to failure count
                    labels = [m.get("model", "?") for m in failing_models]
                    values = [m.get("failed_units", 0) for m in failing_models]
                    self.pareto_chart.plot_pareto(labels=labels, values=values)
            else:
                # No pricing — use failure count
                labels = [m.get("model", "?") for m in failing_models]
                values = [m.get("failed_units", 0) for m in failing_models]
                self.pareto_chart.plot_pareto(labels=labels, values=values)
        except Exception as e:
            logger.debug(f"Pareto chart update error: {e}")

    def _export_executive_summary(self):
        """Export executive summary report to Excel."""
        from tkinter import filedialog, messagebox

        file_path = filedialog.asksaveasfilename(
            title="Save Executive Summary",
            defaultextension=".xlsx",
            initialfile=f"quality_summary_{datetime.now().strftime('%Y%m%d')}.xlsx",
            initialdir=getattr(self.app.config, 'export_path', None),
            filetypes=[("Excel files", "*.xlsx")],
        )
        if not file_path:
            return

        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.export.excel import export_executive_summary

            db = get_database()
            stats = db.get_dashboard_stats(days_back=90)
            priority_models = db.get_linearity_prioritization(days_back=90, min_samples=10)
            near_miss_data = db.get_near_miss_summary(days_back=90)
            recommendations = db.get_screening_recommendations(days_back=90)
            pricing = self.app.config.active_models.model_prices
            cost_ratio = self.app.config.active_models.cost_ratio

            output = export_executive_summary(
                output_path=file_path,
                stats=stats,
                priority_models=priority_models,
                near_miss_data=near_miss_data,
                recommendations=recommendations,
                pricing=pricing,
                cost_ratio=cost_ratio,
            )
            messagebox.showinfo("Export Complete", f"Executive summary saved to:\n{output}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{e}")

    def _update_breakdown_charts(
        self,
        element_data: List[Dict[str, Any]],
        class_data: List[Dict[str, Any]],
    ):
        """Update category breakdown text displays."""
        # Element type breakdown
        try:
            if element_data:
                lines = []
                for item in sorted(element_data, key=lambda x: x["pass_rate"]):
                    cat = item["category"]
                    rate = item["pass_rate"]
                    total = item["total"]
                    color_indicator = "🟢" if rate >= 70 else ("🟡" if rate >= 50 else "🔴")
                    lines.append(f"{color_indicator} {cat}: {rate:.1f}% ({total:,} units)")
                text = "BY ELEMENT TYPE\n\n" + "\n".join(lines)
            else:
                text = "No element type data.\nImport model specs to enable."
            self._element_breakdown_label.configure(text=text)
        except Exception as e:
            logger.debug(f"Element breakdown error: {e}")

        # Product class breakdown
        try:
            if class_data:
                lines = []
                for item in sorted(class_data, key=lambda x: x["pass_rate"]):
                    cat = item["category"]
                    rate = item["pass_rate"]
                    total = item["total"]
                    color_indicator = "🟢" if rate >= 70 else ("🟡" if rate >= 50 else "🔴")
                    lines.append(f"{color_indicator} {cat}: {rate:.1f}% ({total:,} units)")
                text = "BY PRODUCT CLASS\n\n" + "\n".join(lines)
            else:
                text = "No product class data.\nImport model specs to enable."
            self._class_breakdown_label.configure(text=text)
        except Exception as e:
            logger.debug(f"Class breakdown error: {e}")

    def _update_attention_cards(self, priority_models, cpk_data=None, drift_events=None):
        """Update attention cards with models needing action.

        Args:
            priority_models: Pre-loaded priority model data.
            cpk_data: Pre-loaded Cpk data (from background thread).
            drift_events: Pre-loaded drift events (from background thread).
        """
        # Clear existing cards
        for widget in self._attention_frame.winfo_children():
            widget.destroy()

        SEVERITY_COLORS = {
            "critical": ("#dc3545", "#fff"),
            "warning": ("#fd7e14", "#fff"),
            "good": ("#198754", "#fff"),
        }

        items = []
        if priority_models:
            for m in priority_models[:3]:
                pr = m.get("linearity_pass_rate", 100)
                if pr < 70:
                    severity = "critical" if pr < 50 else "warning"
                    items.append({
                        "title": m.get("model", "?"),
                        "subtitle": f"{pr:.0f}% pass rate",
                        "value": f"{m.get('total_tracks', 0)} tracks",
                        "severity": severity,
                    })

        # Add low-Cpk models (using pre-loaded data)
        if cpk_data:
            for c in cpk_data[:2]:
                if c["cpk"] is not None and c["cpk"] < 1.0 and len(items) < 4:
                    items.append({
                        "title": c["model"],
                        "subtitle": f"Cpk = {c['cpk']:.2f} ({c['rating']})",
                        "value": f"Spec: {c['spec_pct']:.1f}%",
                        "severity": "critical" if c["cpk"] < 0.67 else "warning",
                    })

        # Add drift alerts (using pre-loaded data)
        if drift_events and len(items) < 4:
            drift_models = list(set(e["model"] for e in drift_events))
            items.append({
                "title": f"{len(drift_models)} Drift Alerts",
                "subtitle": ", ".join(drift_models[:3]),
                "value": "Process shift detected",
                "severity": "warning",
            })

        if not items:
            items.append({
                "title": "All Clear",
                "subtitle": "No models need immediate attention",
                "value": "",
                "severity": "good",
            })

        for i, item in enumerate(items[:4]):
            bg, fg = SEVERITY_COLORS.get(item["severity"], ("#6c757d", "#fff"))
            card = ctk.CTkFrame(self._attention_frame, fg_color=bg, corner_radius=8,
                               cursor="hand2")
            card.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
            ctk.CTkLabel(card, text=item["title"],
                        font=ctk.CTkFont(size=14, weight="bold"),
                        text_color=fg).pack(padx=10, pady=(8, 2), anchor="w")
            ctk.CTkLabel(card, text=item["subtitle"],
                        font=ctk.CTkFont(size=11),
                        text_color=fg).pack(padx=10, pady=0, anchor="w")
            if item.get("value"):
                ctk.CTkLabel(card, text=item["value"],
                            font=ctk.CTkFont(size=10),
                            text_color=fg).pack(padx=10, pady=(0, 8), anchor="w")
            else:
                ctk.CTkLabel(card, text="", height=4).pack()

            # Make card clickable — navigate to Analyze page for that model
            model_name = item.get("title", "")
            if model_name and model_name != "All Clear" and "Drift" not in model_name:
                card.bind("<Button-1>", lambda e, m=model_name: self._navigate_to_model(m))
                for child in card.winfo_children():
                    child.bind("<Button-1>", lambda e, m=model_name: self._navigate_to_model(m))

    def _navigate_to_model(self, model: str):
        """Navigate to Analyze page filtered to a specific model."""
        if hasattr(self.app, 'show_model_scorecard'):
            self.app.show_model_scorecard(model)

    def _update_process_health(self, cpk_data=None, drift_events=None, failure_modes=None):
        """Update process health section with Cpk and drift data.

        Args:
            cpk_data: Pre-loaded Cpk data (from background thread).
            drift_events: Pre-loaded drift events (from background thread).
            failure_modes: Pre-loaded failure mode summary (from background thread).
        """
        try:
            if cpk_data:
                ratings = {"Excellent": 0, "Capable": 0, "Marginal": 0, "Incapable": 0}
                for c in cpk_data:
                    r = c.get("rating", "Unknown")
                    if r in ratings:
                        ratings[r] += 1
                text = "\n".join(f"  {r}: {n} models" for r, n in ratings.items() if n > 0)
                self._cpk_summary_label.configure(text=text or "No Cpk data")
            else:
                self._cpk_summary_label.configure(text="Import model specs to enable Cpk")
        except Exception:
            self._cpk_summary_label.configure(text="Cpk unavailable")

        # Failure mode breakdown (using pre-loaded data)
        try:
            lines = []
            if failure_modes:
                lines.append("Failure Modes:")
                for m in failure_modes:
                    lines.append(f"  {m['mode']}: {m['count']}")
            if drift_events:
                drift_models = list(set(e["model"] for e in drift_events))
                lines.append(f"\nDrift: {len(drift_models)} models")
                for dm in drift_models[:3]:
                    lines.append(f"  {dm}")
            self._drift_summary_label.configure(
                text="\n".join(lines) if lines else "No failures or drift detected"
            )
        except Exception:
            self._drift_summary_label.configure(text="Data unavailable")

    def _show_error(self, error: str):
        """Show error state."""
        if not self.winfo_exists():
            return
        self._update_card(
            self.health_card,
            title="Linearity Quality",
            value="--",
            subtitle=f"Error: {error[:30]}...",
            color="red"
        )
        self.last_update_label.configure(text="Error loading data")

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Dashboard page shown")
        # Populate filter dropdowns
        self._populate_filters()
        # Refresh data when page is shown
        self._refresh_data()

    def _populate_filters(self):
        """Populate element type and product class filter dropdowns.

        Runs the DB queries on a background thread to keep the UI responsive
        during page navigation; dropdown .configure() is marshaled back to the
        main thread via self.after() because tkinter is not thread-safe.
        """
        def _load_filter_values():
            try:
                db = get_database()
                etypes = ["All Element Types"] + db.get_distinct_element_types()
                pclasses = ["All Product Classes"] + db.get_distinct_product_classes()
                # Hop back to the main thread before touching any widget.
                self.after(0, lambda: self._apply_filter_values(etypes, pclasses))
            except Exception as e:
                logger.debug(f"Could not populate filters: {e}")

        get_thread_manager().start_thread(
            target=_load_filter_values,
            name="dashboard-populate-filters",
        )

    def _apply_filter_values(self, etypes, pclasses):
        """Main-thread callback to apply dropdown values. Must not run on a worker."""
        try:
            self._element_filter.configure(values=etypes)
            self._class_filter.configure(values=pclasses)
        except Exception as e:
            logger.debug(f"Could not apply filter values: {e}")

    def _on_filter_changed(self, _=None):
        """Handle filter dropdown change."""
        self._refresh_data()

    def on_hide(self):
        """Called when page becomes hidden - cleanup to free memory."""
        # Clear charts to free matplotlib resources
        for chart in [self.trend_chart, self.pareto_chart]:
            if chart and hasattr(chart, 'figure'):
                try:
                    chart.clear()
                except Exception as e:
                    logger.debug(f"Chart cleanup warning: {e}")
