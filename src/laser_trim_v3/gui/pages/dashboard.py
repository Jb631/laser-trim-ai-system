"""
Dashboard Page - Overview, alerts, quick stats.

The at-a-glance health overview page.
"""

import customtkinter as ctk
import logging

logger = logging.getLogger(__name__)


class DashboardPage(ctk.CTkFrame):
    """
    Dashboard page showing:
    - Health score card (overall pass rate)
    - Recent alerts (top 5)
    - Processing stats (files today, this week)
    - Quick action buttons (Process New, View Trends)
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        self._create_ui()

    def _create_ui(self):
        """Create the dashboard UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkLabel(
            self,
            text="Dashboard",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        # Content frame
        content = ctk.CTkFrame(self)
        content.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        content.grid_columnconfigure((0, 1), weight=1)
        content.grid_rowconfigure((0, 1), weight=1)

        # Health Score Card
        health_card = self._create_card(
            content,
            title="Overall Health",
            value="--",
            subtitle="Processing required"
        )
        health_card.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Files Processed Card
        files_card = self._create_card(
            content,
            title="Files Processed",
            value="0",
            subtitle="Total in database"
        )
        files_card.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Recent Alerts Card
        alerts_frame = ctk.CTkFrame(content)
        alerts_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        alerts_label = ctk.CTkLabel(
            alerts_frame,
            text="Recent Alerts",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        alerts_label.pack(padx=15, pady=15, anchor="w")

        no_alerts = ctk.CTkLabel(
            alerts_frame,
            text="No alerts - everything looks good!",
            text_color="gray"
        )
        no_alerts.pack(padx=15, pady=10, anchor="w")

        # Quick Actions Card
        actions_frame = ctk.CTkFrame(content)
        actions_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        actions_label = ctk.CTkLabel(
            actions_frame,
            text="Quick Actions",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        actions_label.pack(padx=15, pady=15, anchor="w")

        process_btn = ctk.CTkButton(
            actions_frame,
            text="Process New Files",
            command=lambda: self.app._show_page("process")
        )
        process_btn.pack(padx=15, pady=5, fill="x")

        analyze_btn = ctk.CTkButton(
            actions_frame,
            text="Analyze Results",
            command=lambda: self.app._show_page("analyze")
        )
        analyze_btn.pack(padx=15, pady=5, fill="x")

        trends_btn = ctk.CTkButton(
            actions_frame,
            text="View Trends",
            command=lambda: self.app._show_page("trends")
        )
        trends_btn.pack(padx=15, pady=(5, 15), fill="x")

    def _create_card(self, parent, title: str, value: str, subtitle: str) -> ctk.CTkFrame:
        """Create a metric card."""
        card = ctk.CTkFrame(parent)

        title_label = ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        title_label.pack(padx=15, pady=(15, 5), anchor="w")

        value_label = ctk.CTkLabel(
            card,
            text=value,
            font=ctk.CTkFont(size=36, weight="bold")
        )
        value_label.pack(padx=15, anchor="w")

        subtitle_label = ctk.CTkLabel(
            card,
            text=subtitle,
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        subtitle_label.pack(padx=15, pady=(0, 15), anchor="w")

        return card

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Dashboard page shown")
        # TODO: Refresh stats from database
