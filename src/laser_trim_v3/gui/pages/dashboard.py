"""
Dashboard Page - Overview, alerts, quick stats.

The at-a-glance health overview page.
Wired to the database for real statistics.
"""

import threading
import customtkinter as ctk
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from laser_trim_v3.database import get_database
from laser_trim_v3.gui.widgets.chart import ChartWidget, ChartStyle

logger = logging.getLogger(__name__)


class DashboardPage(ctk.CTkFrame):
    """
    Dashboard page showing:
    - Health score card (overall pass rate)
    - Recent alerts (top 5)
    - Processing stats (files today, this week)
    - Quick action buttons (Process New, View Trends)
    - Pass rate trend chart
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.stats: Optional[Dict[str, Any]] = None

        self._create_ui()

    def _create_ui(self):
        """Create the dashboard UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

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

        refresh_btn = ctk.CTkButton(
            header_frame,
            text="⟳ Refresh",
            width=100,
            command=self._refresh_data
        )
        refresh_btn.grid(row=0, column=1, sticky="e")

        self.last_update_label = ctk.CTkLabel(
            header_frame,
            text="",
            text_color="gray",
            font=ctk.CTkFont(size=10)
        )
        self.last_update_label.grid(row=0, column=2, sticky="e", padx=(10, 0))

        # Content frame
        content = ctk.CTkFrame(self)
        content.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        content.grid_columnconfigure((0, 1, 2), weight=1)
        content.grid_rowconfigure((0, 1), weight=1)

        # Health Score Card
        self.health_card = self._create_metric_card(content)
        self.health_card.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self._update_card(
            self.health_card,
            title="Overall Health",
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

        # Today's Stats Card
        self.today_card = self._create_metric_card(content)
        self.today_card.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self._update_card(
            self.today_card,
            title="Today",
            value="0",
            subtitle="Loading...",
            color="gray"
        )

        # Recent Alerts Card
        self.alerts_frame = ctk.CTkFrame(content)
        self.alerts_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.alerts_frame.grid_columnconfigure(0, weight=1)
        self.alerts_frame.grid_rowconfigure(1, weight=1)

        alerts_label = ctk.CTkLabel(
            self.alerts_frame,
            text="Recent Alerts",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        alerts_label.grid(row=0, column=0, padx=15, pady=15, sticky="w")

        self.alerts_list = ctk.CTkTextbox(self.alerts_frame, height=150)
        self.alerts_list.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))
        self.alerts_list.configure(state="disabled")
        self._update_alerts_display([])

        # Trend Chart
        chart_frame = ctk.CTkFrame(content)
        chart_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        chart_label = ctk.CTkLabel(
            chart_frame,
            text="Pass Rate Trend (7 Days)",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        chart_label.pack(padx=15, pady=15, anchor="w")

        self.trend_chart = ChartWidget(
            chart_frame,
            style=ChartStyle(figure_size=(5, 3), dpi=80)
        )
        self.trend_chart.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.trend_chart.show_placeholder("Loading trend data...")

        # Quick Actions Card
        actions_frame = ctk.CTkFrame(content)
        actions_frame.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")

        actions_label = ctk.CTkLabel(
            actions_frame,
            text="Quick Actions",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        actions_label.pack(padx=15, pady=15, anchor="w")

        process_btn = ctk.CTkButton(
            actions_frame,
            text="📁 Process New Files",
            command=lambda: self.app._show_page("process"),
            height=40
        )
        process_btn.pack(padx=15, pady=5, fill="x")

        analyze_btn = ctk.CTkButton(
            actions_frame,
            text="🔍 Analyze Results",
            command=lambda: self.app._show_page("analyze"),
            height=40
        )
        analyze_btn.pack(padx=15, pady=5, fill="x")

        trends_btn = ctk.CTkButton(
            actions_frame,
            text="📈 View Trends",
            command=lambda: self.app._show_page("trends"),
            height=40
        )
        trends_btn.pack(padx=15, pady=5, fill="x")

        settings_btn = ctk.CTkButton(
            actions_frame,
            text="⚙️ Settings",
            command=lambda: self.app._show_page("settings"),
            height=40,
            fg_color="gray"
        )
        settings_btn.pack(padx=15, pady=(5, 15), fill="x")

        # Model breakdown
        self.model_frame = ctk.CTkFrame(content)
        self.model_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        model_label = ctk.CTkLabel(
            self.model_frame,
            text="Top Models by Volume",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        model_label.pack(padx=15, pady=15, anchor="w")

        self.model_text = ctk.CTkTextbox(self.model_frame, height=100)
        self.model_text.pack(fill="x", padx=15, pady=(0, 15))
        self.model_text.configure(state="disabled")

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

    def _refresh_data(self):
        """Refresh dashboard data in background thread."""
        self.last_update_label.configure(text="Refreshing...")

        thread = threading.Thread(target=self._load_data, daemon=True)
        thread.start()

    def _load_data(self):
        """Load data from database in background."""
        try:
            db = get_database()
            stats = db.get_dashboard_stats()
            self.stats = stats

            # Also get recent alerts
            alerts = db.get_alerts(limit=5)

            # Get model breakdown
            model_stats = db.get_model_stats(limit=5)

            # Update UI on main thread
            self.after(0, lambda: self._update_display(stats, alerts, model_stats))

        except Exception as e:
            logger.error(f"Failed to load dashboard data: {e}")
            self.after(0, lambda: self._show_error(str(e)))

    def _update_display(
        self,
        stats: Dict[str, Any],
        alerts: List[Dict[str, Any]],
        model_stats: List[Dict[str, Any]]
    ):
        """Update display with loaded data."""
        # Update health card
        pass_rate = stats.get("pass_rate", 0)
        if pass_rate >= 95:
            health_color = "#27ae60"  # Green
            health_status = "Excellent"
        elif pass_rate >= 85:
            health_color = "#f39c12"  # Orange
            health_status = "Good"
        elif pass_rate >= 70:
            health_color = "#e67e22"  # Dark orange
            health_status = "Fair"
        else:
            health_color = "#e74c3c"  # Red
            health_status = "Needs Attention"

        self._update_card(
            self.health_card,
            title="Overall Health",
            value=f"{pass_rate:.1f}%",
            subtitle=f"Pass rate - {health_status}",
            color=health_color
        )

        # Update files card
        total_files = stats.get("total_files", 0)
        passed = stats.get("passed", 0)
        failed = stats.get("failed", 0)
        self._update_card(
            self.files_card,
            title="Files Processed",
            value=str(total_files),
            subtitle=f"✓ {passed} passed, ✗ {failed} failed",
            color="white"
        )

        # Update today card
        today_count = stats.get("today_count", 0)
        week_count = stats.get("week_count", 0)
        self._update_card(
            self.today_card,
            title="Today",
            value=str(today_count),
            subtitle=f"This week: {week_count}",
            color="white"
        )

        # Update alerts
        self._update_alerts_display(alerts)

        # Update trend chart
        self._update_trend_chart(stats)

        # Update model breakdown
        self._update_model_display(model_stats)

        # Update timestamp
        self.last_update_label.configure(
            text=f"Updated: {datetime.now().strftime('%H:%M:%S')}"
        )

        logger.debug("Dashboard data refreshed")

    def _update_alerts_display(self, alerts: List[Dict[str, Any]]):
        """Update alerts list."""
        self.alerts_list.configure(state="normal")
        self.alerts_list.delete("1.0", "end")

        if not alerts:
            self.alerts_list.insert("end", "No active alerts - everything looks good!")
        else:
            for alert in alerts:
                severity = alert.get("severity", "INFO")
                message = alert.get("message", "")
                timestamp = alert.get("created_at", "")

                icon = "⚠️" if severity == "WARNING" else "🔴" if severity == "CRITICAL" else "ℹ️"
                self.alerts_list.insert(
                    "end",
                    f"{icon} [{severity}] {message}\n   {timestamp}\n\n"
                )

        self.alerts_list.configure(state="disabled")

    def _update_trend_chart(self, stats: Dict[str, Any]):
        """Update trend chart with pass rate data."""
        trend_data = stats.get("daily_trend", [])

        if not trend_data:
            self.trend_chart.show_placeholder("No trend data available")
            return

        # Extract dates and pass rates
        dates = [d.get("date", "") for d in trend_data]
        pass_rates = [d.get("pass_rate", 0) for d in trend_data]

        if len(pass_rates) < 2:
            self.trend_chart.show_placeholder("Insufficient data for trend")
            return

        # Plot SPC-style chart
        self.trend_chart.plot_spc_control(
            values=pass_rates,
            dates=dates,
            ucl=100,  # Pass rate can't exceed 100%
            lcl=max(0, min(pass_rates) - 10),  # Show some context below min
            center=sum(pass_rates) / len(pass_rates),
            title="",
            ylabel="Pass Rate %"
        )

    def _update_model_display(self, model_stats: List[Dict[str, Any]]):
        """Update model breakdown display."""
        self.model_text.configure(state="normal")
        self.model_text.delete("1.0", "end")

        if not model_stats:
            self.model_text.insert("end", "No model data available")
        else:
            lines = []
            for stat in model_stats:
                model = stat.get("model", "Unknown")
                count = stat.get("count", 0)
                pass_rate = stat.get("pass_rate", 0)
                lines.append(f"  {model}: {count} files ({pass_rate:.1f}% pass rate)")
            self.model_text.insert("end", "\n".join(lines))

        self.model_text.configure(state="disabled")

    def _show_error(self, error: str):
        """Show error state."""
        self._update_card(
            self.health_card,
            title="Overall Health",
            value="--",
            subtitle=f"Error: {error[:30]}...",
            color="red"
        )
        self.last_update_label.configure(text="Error loading data")

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Dashboard page shown")
        # Refresh data when page is shown
        self._refresh_data()
