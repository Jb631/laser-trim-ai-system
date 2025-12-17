"""
Trends Page - Historical analysis and ML insights.

Shows model trends, drift detection, and ML recommendations.
Wired to the database and ML components.
"""

import threading
import customtkinter as ctk
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import numpy as np

from laser_trim_v3.database import get_database
from laser_trim_v3.gui.widgets.chart import ChartWidget, ChartStyle

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
        self.selected_model: str = "All Models"
        self.selected_days: int = 30
        self.trend_data: List[Dict[str, Any]] = []

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

        # Model selector - use ComboBox for dropdown with many models
        model_label = ctk.CTkLabel(controls, text="Model:")
        model_label.pack(side="left", padx=(15, 5), pady=15)

        self.model_dropdown = ctk.CTkComboBox(
            controls,
            values=["All Models"],
            command=self._on_model_change,
            width=150,
            state="normal"  # Allow typing to filter, dropdown still works
        )
        self.model_dropdown.set("All Models")
        self.model_dropdown.pack(side="left", padx=5, pady=15)
        # Bind mousewheel to scroll through options
        self._bind_mousewheel_scroll(self.model_dropdown)

        # Date range
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
            text="⟳ Refresh",
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

        # Main content area - scrollable for smaller screens
        content = ctk.CTkScrollableFrame(self)
        content.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        content.grid_columnconfigure(0, weight=1)
        # All sections stacked vertically
        content.grid_rowconfigure(0, weight=0)  # Statistics - compact
        content.grid_rowconfigure(1, weight=2, minsize=300)  # SPC chart - larger
        content.grid_rowconfigure(2, weight=1, minsize=200)  # Distribution chart
        content.grid_rowconfigure(3, weight=0)  # Drift detection - compact
        content.grid_rowconfigure(4, weight=0)  # ML recommendations - compact

        # Statistics panel (top - horizontal layout)
        stats_frame = ctk.CTkFrame(content)
        stats_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        stats_label = ctk.CTkLabel(
            stats_frame,
            text="Statistics",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        stats_label.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="w", columnspan=6)

        # Stats in a horizontal row for efficient space usage
        self.stat_labels = {}
        stat_names = [
            ("files_analyzed", "Files Analyzed"),
            ("pass_rate", "Pass Rate"),
            ("avg_sigma", "Avg Sigma"),
            ("std_dev", "Std Dev"),
            ("cpk", "Cpk"),
            ("trend", "Trend"),
        ]

        for idx, (key, label) in enumerate(stat_names):
            stat_col = ctk.CTkFrame(stats_frame, fg_color="transparent")
            stat_col.grid(row=1, column=idx, padx=15, pady=(0, 15), sticky="w")

            ctk.CTkLabel(stat_col, text=label, text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w")
            value_label = ctk.CTkLabel(stat_col, text="--", font=ctk.CTkFont(size=14, weight="bold"))
            value_label.pack(anchor="w")
            self.stat_labels[key] = value_label

        # SPC Chart (main chart - largest)
        chart_frame = ctk.CTkFrame(content)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        chart_label = ctk.CTkLabel(
            chart_frame,
            text="Sigma Gradient SPC Chart",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        chart_label.pack(padx=15, pady=(15, 5), anchor="w")

        self.chart = ChartWidget(
            chart_frame,
            style=ChartStyle(figure_size=(10, 4), dpi=100)
        )
        self.chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.chart.show_placeholder("Select a model and date range to view trends")

        # Distribution chart
        dist_frame = ctk.CTkFrame(content)
        dist_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        dist_label = ctk.CTkLabel(
            dist_frame,
            text="Sigma Distribution",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        dist_label.pack(padx=15, pady=(15, 5), anchor="w")

        self.dist_chart = ChartWidget(
            dist_frame,
            style=ChartStyle(figure_size=(10, 2.5), dpi=100)
        )
        self.dist_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.dist_chart.show_placeholder("Distribution will appear here")

        # Bottom row: Drift Detection and ML Recommendations side by side
        bottom_frame = ctk.CTkFrame(content, fg_color="transparent")
        bottom_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)

        # Drift alerts section (left)
        drift_frame = ctk.CTkFrame(bottom_frame)
        drift_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        drift_label = ctk.CTkLabel(
            drift_frame,
            text="Drift Detection",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        drift_label.pack(padx=15, pady=(15, 10), anchor="w")

        self.drift_text = ctk.CTkTextbox(drift_frame, height=100)
        self.drift_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.drift_text.configure(state="disabled")
        self._update_drift_display(None)

        # ML recommendations section (right)
        ml_frame = ctk.CTkFrame(bottom_frame)
        ml_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        ml_label = ctk.CTkLabel(
            ml_frame,
            text="ML Recommendations",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        ml_label.pack(padx=15, pady=(15, 10), anchor="w")

        self.ml_text = ctk.CTkTextbox(ml_frame, height=100)
        self.ml_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.ml_text.configure(state="disabled")
        self._update_ml_display(None)

    def _bind_mousewheel_scroll(self, combobox):
        """Bind mousewheel events to scroll through combobox values."""
        def on_mousewheel(event):
            values = combobox.cget("values")
            if not values:
                return

            current = combobox.get()
            try:
                current_idx = list(values).index(current)
            except ValueError:
                current_idx = 0

            # Scroll direction (Windows: event.delta, Linux: event.num)
            if hasattr(event, 'delta'):
                # Windows
                direction = -1 if event.delta > 0 else 1
            else:
                # Linux
                direction = -1 if event.num == 4 else 1

            new_idx = current_idx + direction
            if 0 <= new_idx < len(values):
                combobox.set(values[new_idx])
                # Trigger command callback if defined
                command = combobox.cget("command")
                if command:
                    command(values[new_idx])

        # Bind for Windows
        combobox.bind("<MouseWheel>", on_mousewheel)
        # Bind for Linux
        combobox.bind("<Button-4>", on_mousewheel)
        combobox.bind("<Button-5>", on_mousewheel)

    def _on_model_change(self, model: str):
        """Handle model selection change."""
        self.selected_model = model
        logger.debug(f"Model changed to: {model}")
        self._refresh_data()

    def _on_date_change(self, date_range: str):
        """Handle date range change."""
        days_map = {
            "Last 7 Days": 7,
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "All Time": 365 * 10,  # 10 years
        }
        self.selected_days = days_map.get(date_range, 30)
        logger.debug(f"Date range changed to: {date_range} ({self.selected_days} days)")
        self._refresh_data()

    def _refresh_data(self):
        """Refresh trend data from database."""
        self.status_label.configure(text="Loading...")
        thread = threading.Thread(target=self._load_data, daemon=True)
        thread.start()

    def _load_data(self):
        """Load data in background thread."""
        try:
            db = get_database()

            # Load model list
            models = db.get_models_list()

            # Load trend data
            trend_data = db.get_trend_data(
                model=None if self.selected_model == "All Models" else self.selected_model,
                days_back=self.selected_days
            )

            # Try to get ML recommendations
            ml_recommendations = None
            try:
                from laser_trim_v3.ml.threshold import ThresholdOptimizer
                from laser_trim_v3.config import get_config
                config = get_config()
                model_path = config.models.path / "threshold_optimizer.pkl"

                optimizer = ThresholdOptimizer()
                if model_path.exists():
                    optimizer.load(model_path)

                if optimizer.is_trained and self.selected_model != "All Models":
                    # Get a recommendation using predict_with_confidence
                    # Use average values from trend data as inputs
                    avg_unit_length = 100.0  # Default
                    avg_linearity_spec = 0.01  # Default

                    if trend_data:
                        unit_lengths = [d.get("unit_length", 100.0) for d in trend_data if d.get("unit_length")]
                        linearity_specs = [d.get("linearity_spec", 0.01) for d in trend_data if d.get("linearity_spec")]
                        if unit_lengths:
                            avg_unit_length = sum(unit_lengths) / len(unit_lengths)
                        if linearity_specs:
                            avg_linearity_spec = sum(linearity_specs) / len(linearity_specs)

                    threshold, lower, upper = optimizer.predict_with_confidence(
                        model=self.selected_model,
                        unit_length=avg_unit_length,
                        linearity_spec=avg_linearity_spec
                    )
                    ml_recommendations = {
                        "recommended_threshold": threshold,
                        "confidence": 1.0 - (upper - lower) / threshold if threshold > 0 else 0.5,
                        "basis": f"{optimizer.training_metadata.get('n_samples', 'unknown')} historical samples"
                    }
            except Exception as e:
                logger.debug(f"ML recommendations not available: {e}")

            # Try drift detection
            drift_result = None
            try:
                from laser_trim_v3.ml.drift import DriftDetector
                detector = DriftDetector()
                if trend_data and len(trend_data) > 10:
                    sigma_values = [d.get("sigma_gradient", 0) for d in trend_data if d.get("sigma_gradient") is not None]
                    if len(sigma_values) > 10:
                        drift_result = detector.detect_batch(np.array(sigma_values))
            except Exception as e:
                logger.debug(f"Drift detection not available: {e}")

            # Update UI on main thread
            self.after(0, lambda: self._update_display(
                models, trend_data, ml_recommendations, drift_result
            ))

        except Exception as e:
            logger.error(f"Failed to load trend data: {e}")
            self.after(0, lambda: self._show_error(str(e)))

    def _update_display(
        self,
        models: List[str],
        trend_data: List[Dict[str, Any]],
        ml_recommendations: Optional[Dict[str, Any]],
        drift_result: Optional[Any]
    ):
        """Update display with loaded data."""
        # Update model dropdown
        model_values = ["All Models"] + models
        current_model = self.model_dropdown.get()
        self.model_dropdown.configure(values=model_values)
        if current_model in model_values:
            self.model_dropdown.set(current_model)
        else:
            self.model_dropdown.set("All Models")

        self.trend_data = trend_data

        # Calculate statistics
        if trend_data:
            sigma_values = [d.get("sigma_gradient", 0) for d in trend_data if d.get("sigma_gradient")]
            threshold_values = [d.get("sigma_threshold", 0) for d in trend_data if d.get("sigma_threshold")]

            if sigma_values:
                total = len(trend_data)
                passed = sum(1 for d in trend_data if d.get("sigma_pass", False))
                pass_rate = (passed / total * 100) if total > 0 else 0

                avg_sigma = np.mean(sigma_values)
                std_sigma = np.std(sigma_values, ddof=1) if len(sigma_values) > 1 else 0

                # Calculate Cpk (process capability)
                if threshold_values and std_sigma > 0:
                    avg_threshold = np.mean(threshold_values)
                    cpk = (avg_threshold - avg_sigma) / (3 * std_sigma)
                else:
                    cpk = 0

                # Calculate trend direction
                if len(sigma_values) >= 3:
                    first_half = np.mean(sigma_values[:len(sigma_values)//2])
                    second_half = np.mean(sigma_values[len(sigma_values)//2:])
                    if second_half > first_half * 1.1:
                        trend = "↑ Increasing"
                        trend_color = "#e74c3c"
                    elif second_half < first_half * 0.9:
                        trend = "↓ Decreasing"
                        trend_color = "#27ae60"
                    else:
                        trend = "→ Stable"
                        trend_color = "#3498db"
                else:
                    trend = "-- Insufficient data"
                    trend_color = "gray"

                # Update stat labels
                self.stat_labels["files_analyzed"].configure(text=str(total))
                self.stat_labels["pass_rate"].configure(
                    text=f"{pass_rate:.1f}%",
                    text_color="#27ae60" if pass_rate >= 95 else "#f39c12" if pass_rate >= 85 else "#e74c3c"
                )
                self.stat_labels["avg_sigma"].configure(text=f"{avg_sigma:.6f}")
                self.stat_labels["std_dev"].configure(text=f"{std_sigma:.6f}")
                self.stat_labels["cpk"].configure(
                    text=f"{cpk:.2f}",
                    text_color="#27ae60" if cpk >= 1.33 else "#f39c12" if cpk >= 1.0 else "#e74c3c"
                )
                self.stat_labels["trend"].configure(text=trend, text_color=trend_color)

                # Update SPC chart
                self._update_spc_chart(trend_data)

                # Update distribution chart
                self._update_distribution_chart(sigma_values, threshold_values)
            else:
                self._reset_stats()
                self.chart.show_placeholder("No sigma data available")
                self.dist_chart.show_placeholder("No distribution data")
        else:
            self._reset_stats()
            self.chart.show_placeholder("No data available for selected filters")
            self.dist_chart.show_placeholder("No distribution data")

        # Update drift display
        self._update_drift_display(drift_result)

        # Update ML recommendations
        self._update_ml_display(ml_recommendations)

        # Update status
        self.status_label.configure(
            text=f"Updated: {datetime.now().strftime('%H:%M:%S')}"
        )

    def _reset_stats(self):
        """Reset statistics to default values."""
        for key in self.stat_labels:
            self.stat_labels[key].configure(text="--", text_color="white")

    def _update_spc_chart(self, trend_data: List[Dict[str, Any]]):
        """Update SPC control chart."""
        if not trend_data:
            self.chart.show_placeholder("No data to display")
            return

        sigma_values = [d.get("sigma_gradient", 0) for d in trend_data if d.get("sigma_gradient") is not None]
        threshold_values = [d.get("sigma_threshold", 0) for d in trend_data if d.get("sigma_threshold")]

        if len(sigma_values) < 2:
            self.chart.show_placeholder("Insufficient data for SPC chart")
            return

        # Calculate control limits
        mean_sigma = np.mean(sigma_values)
        std_sigma = np.std(sigma_values, ddof=1)
        ucl = mean_sigma + 3 * std_sigma
        lcl = max(0, mean_sigma - 3 * std_sigma)

        # Use average threshold as spec limit
        spec_limit = np.mean(threshold_values) if threshold_values else ucl

        # Create date labels
        dates = [d.get("date", i) for i, d in enumerate(trend_data)]
        if len(dates) > 20:
            # Show only every nth date to avoid crowding
            step = len(dates) // 10
            dates = [d if i % step == 0 else "" for i, d in enumerate(dates)]

        self.chart.plot_spc_control(
            values=sigma_values,
            dates=dates,
            ucl=min(ucl, spec_limit * 1.5),  # Cap UCL at 1.5x spec
            lcl=lcl,
            center=mean_sigma,
            title=f"Sigma Gradient Trend ({self.selected_model})",
            ylabel="Sigma Gradient"
        )

    def _update_distribution_chart(self, sigma_values: List[float], threshold_values: List[float]):
        """Update distribution histogram."""
        if not sigma_values or len(sigma_values) < 3:
            self.dist_chart.show_placeholder("Insufficient data for distribution")
            return

        spec_limit = np.mean(threshold_values) if threshold_values else None

        self.dist_chart.plot_histogram(
            values=sigma_values,
            bins=min(30, len(sigma_values) // 3 + 1),
            title="Sigma Gradient Distribution",
            xlabel="Sigma Gradient",
            spec_limit=spec_limit
        )

    def _update_drift_display(self, drift_result: Optional[Any]):
        """Update drift detection display."""
        self.drift_text.configure(state="normal")
        self.drift_text.delete("1.0", "end")

        if drift_result is None:
            self.drift_text.insert("end", "Drift detection not available.\n\nProcess more data to enable drift detection.")
        elif hasattr(drift_result, 'drift_detected') and drift_result.drift_detected:
            severity = drift_result.severity if hasattr(drift_result, 'severity') else "Unknown"
            direction = drift_result.direction if hasattr(drift_result, 'direction') else "Unknown"
            confidence = drift_result.confidence if hasattr(drift_result, 'confidence') else 0

            self.drift_text.insert("end", f"⚠️ DRIFT DETECTED\n\n")
            self.drift_text.insert("end", f"Severity: {severity}\n")
            self.drift_text.insert("end", f"Direction: {direction}\n")
            self.drift_text.insert("end", f"Confidence: {confidence:.1%}\n")
        else:
            self.drift_text.insert("end", "✓ No significant drift detected.\n\nProcess is stable.")

        self.drift_text.configure(state="disabled")

    def _update_ml_display(self, ml_recommendations: Optional[Dict[str, Any]]):
        """Update ML recommendations display."""
        self.ml_text.configure(state="normal")
        self.ml_text.delete("1.0", "end")

        if ml_recommendations is None:
            self.ml_text.insert("end", "ML recommendations not available.\n\nTrain the threshold optimizer to see model-specific recommendations.")
        else:
            threshold = ml_recommendations.get("recommended_threshold", 0)
            confidence = ml_recommendations.get("confidence", 0)
            basis = ml_recommendations.get("basis", "historical data")

            self.ml_text.insert("end", f"Recommended Threshold:\n")
            self.ml_text.insert("end", f"  {threshold:.6f}\n\n")
            self.ml_text.insert("end", f"Confidence: {confidence:.1%}\n")
            self.ml_text.insert("end", f"Based on: {basis}\n")

        self.ml_text.configure(state="disabled")

    def _show_error(self, error: str):
        """Show error state."""
        self.status_label.configure(text="Error loading data")
        self.chart.show_placeholder(f"Error: {error}")
        self.dist_chart.show_placeholder("Error loading data")

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Trends page shown")
        self._refresh_data()
