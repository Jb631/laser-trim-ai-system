"""
Trends Page - Historical analysis and ML insights.

Shows model trends, drift detection, and ML recommendations.
"""

import customtkinter as ctk
import logging

logger = logging.getLogger(__name__)


class TrendsPage(ctk.CTkFrame):
    """
    Trends page for historical analysis.

    Features:
    - Model selector dropdown
    - Date range picker
    - SPC control chart (I/MR)
    - Trend statistics
    - Drift alerts panel
    - Threshold recommendations
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

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

        # Model selector
        model_label = ctk.CTkLabel(controls, text="Model:")
        model_label.pack(side="left", padx=(15, 5), pady=15)

        self.model_dropdown = ctk.CTkOptionMenu(
            controls,
            values=["All Models", "Loading..."],
            command=self._on_model_change
        )
        self.model_dropdown.pack(side="left", padx=5, pady=15)

        # Date range (simplified for now)
        date_label = ctk.CTkLabel(controls, text="Date Range:")
        date_label.pack(side="left", padx=(20, 5), pady=15)

        self.date_dropdown = ctk.CTkOptionMenu(
            controls,
            values=["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
            command=self._on_date_change
        )
        self.date_dropdown.set("Last 30 Days")
        self.date_dropdown.pack(side="left", padx=5, pady=15)

        # Refresh button
        refresh_btn = ctk.CTkButton(
            controls,
            text="Refresh",
            command=self._refresh_data,
            width=100
        )
        refresh_btn.pack(side="right", padx=15, pady=15)

        # Main content area
        content = ctk.CTkFrame(self)
        content.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # Chart area (left side)
        chart_frame = ctk.CTkFrame(content)
        chart_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        chart_frame.grid_columnconfigure(0, weight=1)
        chart_frame.grid_rowconfigure(0, weight=1)

        chart_placeholder = ctk.CTkLabel(
            chart_frame,
            text="SPC Control Chart\n\nWill display:\n- Sigma gradient trend over time\n- UCL/LCL control limits\n- Specification limit\n- ML-learned threshold\n\nProcess data to see trends.",
            text_color="gray",
            justify="center"
        )
        chart_placeholder.grid(row=0, column=0, padx=20, pady=20)

        # Stats panel (right side)
        stats_frame = ctk.CTkFrame(content)
        stats_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

        stats_label = ctk.CTkLabel(
            stats_frame,
            text="Statistics",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        stats_label.pack(padx=15, pady=(15, 10), anchor="w")

        # Placeholder stats
        stats = [
            ("Files Analyzed", "--"),
            ("Pass Rate", "--%"),
            ("Avg Sigma", "--"),
            ("Std Dev", "--"),
            ("Cpk", "--"),
        ]

        for label, value in stats:
            stat_row = ctk.CTkFrame(stats_frame, fg_color="transparent")
            stat_row.pack(fill="x", padx=15, pady=2)

            ctk.CTkLabel(stat_row, text=label, text_color="gray").pack(side="left")
            ctk.CTkLabel(stat_row, text=value).pack(side="right")

        # Drift alerts section
        drift_label = ctk.CTkLabel(
            stats_frame,
            text="Drift Alerts",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        drift_label.pack(padx=15, pady=(20, 10), anchor="w")

        drift_info = ctk.CTkLabel(
            stats_frame,
            text="No drift detected",
            text_color="gray"
        )
        drift_info.pack(padx=15, pady=5, anchor="w")

        # ML recommendations section
        ml_label = ctk.CTkLabel(
            stats_frame,
            text="ML Recommendations",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        ml_label.pack(padx=15, pady=(20, 10), anchor="w")

        ml_info = ctk.CTkLabel(
            stats_frame,
            text="Train ML models to see\nthreshold recommendations",
            text_color="gray",
            justify="left"
        )
        ml_info.pack(padx=15, pady=(5, 15), anchor="w")

    def _on_model_change(self, model: str):
        """Handle model selection change."""
        logger.debug(f"Model changed to: {model}")
        # TODO: Refresh data for selected model

    def _on_date_change(self, date_range: str):
        """Handle date range change."""
        logger.debug(f"Date range changed to: {date_range}")
        # TODO: Refresh data for selected date range

    def _refresh_data(self):
        """Refresh trend data from database."""
        logger.info("Refreshing trend data...")
        # TODO: Implement data refresh

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Trends page shown")
        # TODO: Load model list from database
