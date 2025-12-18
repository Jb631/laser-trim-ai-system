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

import threading
import customtkinter as ctk
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import numpy as np

from laser_trim_analyzer.database import get_database
from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle

logger = logging.getLogger(__name__)


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
        self.selected_days: int = 90
        self.rolling_window: int = 30
        self.active_models_data: List[Dict[str, Any]] = []
        self.model_trend_data: Optional[Dict[str, Any]] = None

        # Track chart widgets for proper cleanup
        self._chart_widgets: List[ChartWidget] = []

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

        self.model_dropdown = ctk.CTkComboBox(
            controls,
            values=["All Models"],
            command=self._on_model_change,
            width=150,
            state="normal"
        )
        self.model_dropdown.set("All Models")
        self.model_dropdown.pack(side="left", padx=5, pady=15)
        self._bind_mousewheel_scroll(self.model_dropdown)

        # Date range for active models consideration
        date_label = ctk.CTkLabel(controls, text="Activity Period:")
        date_label.pack(side="left", padx=(20, 5), pady=15)

        self.date_dropdown = ctk.CTkOptionMenu(
            controls,
            values=["Last 30 Days", "Last 60 Days", "Last 90 Days", "Last 180 Days"],
            command=self._on_date_change
        )
        self.date_dropdown.set("Last 90 Days")
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
        for chart in self._chart_widgets:
            try:
                chart.destroy()
            except Exception:
                pass
        self._chart_widgets.clear()

    def _create_summary_view(self):
        """Create the summary view (All Models mode)."""
        # Clean up existing charts first (frees matplotlib figures)
        self._cleanup_charts()

        # Clear existing content
        for widget in self.content.winfo_children():
            widget.destroy()

        self.content.grid_rowconfigure(0, weight=0)  # Stats row - compact
        self.content.grid_rowconfigure(1, weight=1, minsize=250)  # Alerts chart
        self.content.grid_rowconfigure(2, weight=1, minsize=250)  # Best/Worst charts
        self.content.grid_rowconfigure(3, weight=0)  # ML section - compact

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

        # Alerts chart (models requiring attention)
        alerts_frame = ctk.CTkFrame(self.content)
        alerts_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        alerts_label = ctk.CTkLabel(
            alerts_frame,
            text="Models Requiring Attention",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        alerts_label.pack(padx=15, pady=(15, 5), anchor="w")

        self.alerts_chart = ChartWidget(
            alerts_frame,
            style=ChartStyle(figure_size=(10, 3), dpi=100)
        )
        self._chart_widgets.append(self.alerts_chart)
        self.alerts_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.alerts_chart.show_placeholder("Loading models requiring attention...")

        # Best/Worst models side by side
        models_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        models_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        models_frame.grid_columnconfigure(0, weight=1)
        models_frame.grid_columnconfigure(1, weight=1)

        # Best performers
        best_frame = ctk.CTkFrame(models_frame)
        best_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        best_label = ctk.CTkLabel(
            best_frame,
            text="Top Performing Models",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        best_label.pack(padx=15, pady=(15, 5), anchor="w")

        self.best_chart = ChartWidget(
            best_frame,
            style=ChartStyle(figure_size=(5, 3), dpi=100)
        )
        self._chart_widgets.append(self.best_chart)
        self.best_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.best_chart.show_placeholder("Loading best models...")

        # Worst performers
        worst_frame = ctk.CTkFrame(models_frame)
        worst_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        worst_label = ctk.CTkLabel(
            worst_frame,
            text="Models Needing Improvement",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        worst_label.pack(padx=15, pady=(15, 5), anchor="w")

        self.worst_chart = ChartWidget(
            worst_frame,
            style=ChartStyle(figure_size=(5, 3), dpi=100)
        )
        self._chart_widgets.append(self.worst_chart)
        self.worst_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.worst_chart.show_placeholder("Loading models needing improvement...")

        # ML Recommendations at bottom
        ml_frame = ctk.CTkFrame(self.content)
        ml_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))

        ml_label = ctk.CTkLabel(
            ml_frame,
            text="ML Insights",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        ml_label.pack(padx=15, pady=(15, 10), anchor="w")

        self.ml_text = ctk.CTkTextbox(ml_frame, height=80)
        self.ml_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.ml_text.configure(state="disabled")
        self._update_ml_summary(None)

    def _create_detail_view(self):
        """Create the detail view (specific model mode)."""
        # Clean up existing charts first (frees matplotlib figures)
        self._cleanup_charts()

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
            ("sigma_pass_rate", "Sigma Pass"),
            ("overall_pass_rate", "Overall Pass"),
            ("avg_sigma", "Avg Sigma"),
            ("threshold", "Threshold"),
            ("trend", "Trend"),
            ("status", "Status"),
        ]

        for idx, (key, label) in enumerate(stat_names):
            stat_col = ctk.CTkFrame(stats_frame, fg_color="transparent")
            stat_col.grid(row=1, column=idx, padx=15, pady=(0, 15), sticky="w")

            ctk.CTkLabel(stat_col, text=label, text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w")
            value_label = ctk.CTkLabel(stat_col, text="--", font=ctk.CTkFont(size=14, weight="bold"))
            value_label.pack(anchor="w")
            self.detail_stat_labels[key] = value_label

        # Main SPC scatter chart
        chart_frame = ctk.CTkFrame(self.content)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        chart_label = ctk.CTkLabel(
            chart_frame,
            text=f"Sigma Gradient Trend ({self.rolling_window}-Day Rolling Average)",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        chart_label.pack(padx=15, pady=(15, 5), anchor="w")

        self.scatter_chart = ChartWidget(
            chart_frame,
            style=ChartStyle(figure_size=(10, 4), dpi=100)
        )
        self._chart_widgets.append(self.scatter_chart)
        self.scatter_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.scatter_chart.show_placeholder("Loading trend data...")

        # Distribution chart
        dist_frame = ctk.CTkFrame(self.content)
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
        self._chart_widgets.append(self.dist_chart)
        self.dist_chart.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.dist_chart.show_placeholder("Loading distribution...")

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
            "Last 60 Days": 60,
            "Last 90 Days": 90,
            "Last 180 Days": 180,
        }
        self.selected_days = days_map.get(date_range, 90)
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

    def _refresh_data(self):
        """Refresh data from database."""
        self.status_label.configure(text="Loading...")
        thread = threading.Thread(target=self._load_data, daemon=True)
        thread.start()

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

        # Get models requiring attention
        alert_models = db.get_models_requiring_attention(
            days_back=self.selected_days,
            min_samples=5,
            pass_rate_threshold=80.0,
            trend_threshold=10.0,
            rolling_days=self.rolling_window
        )

        # Update model dropdown with active models
        model_names = ["All Models"] + [m["model"] for m in active_models]

        # Update UI on main thread
        self.after(0, lambda: self._update_summary_display(
            active_models, alert_models, model_names
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
        """Get ML recommendations for current model."""
        try:
            from laser_trim_analyzer.ml.threshold import ThresholdOptimizer
            from laser_trim_analyzer.config import get_config
            config = get_config()
            model_path = config.models.path / "threshold_optimizer.pkl"

            optimizer = ThresholdOptimizer()
            if model_path.exists():
                optimizer.load(model_path)

            if optimizer.is_trained:
                # Get average values from trend data
                avg_unit_length = 100.0
                avg_linearity_spec = 0.01

                if trend_data.get("data_points"):
                    # Use threshold from data if available
                    pass

                threshold, lower, upper = optimizer.predict_with_confidence(
                    model=self.selected_model,
                    unit_length=avg_unit_length,
                    linearity_spec=avg_linearity_spec
                )
                return {
                    "recommended_threshold": threshold,
                    "confidence": 1.0 - (upper - lower) / threshold if threshold > 0 else 0.5,
                    "basis": f"{optimizer.training_metadata.get('n_samples', 'unknown')} historical samples"
                }
        except Exception as e:
            logger.debug(f"ML recommendations not available: {e}")

        return None

    def _update_summary_display(
        self,
        active_models: List[Dict[str, Any]],
        alert_models: List[Dict[str, Any]],
        model_names: List[str]
    ):
        """Update summary display with loaded data."""
        # Update model dropdown
        current_model = self.model_dropdown.get()
        self.model_dropdown.configure(values=model_names)
        if current_model in model_names:
            self.model_dropdown.set(current_model)
        else:
            self.model_dropdown.set("All Models")

        self.active_models_data = active_models

        if not active_models:
            self._reset_summary_stats()
            self.alerts_chart.show_placeholder("No active models in selected period")
            self.best_chart.show_placeholder("No data")
            self.worst_chart.show_placeholder("No data")
            self.status_label.configure(text="No data")
            return

        # Calculate summary stats
        total_models = len(active_models)
        total_samples = sum(m["total"] for m in active_models)
        avg_pass_rate = sum(m["pass_rate"] for m in active_models) / total_models if total_models > 0 else 0
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

        # Update best/worst charts
        # Best 5 models
        best_5 = sorted_by_rate[:5]
        self.best_chart.plot_pass_rate_bars(
            models=[m["model"] for m in best_5],
            pass_rates=[m["pass_rate"] for m in best_5],
            sample_counts=[m["total"] for m in best_5],
            title="Top 5 Performing Models",
            highlight_threshold=80.0
        )

        # Worst 5 models
        worst_5 = sorted_by_rate[-5:] if len(sorted_by_rate) >= 5 else sorted_by_rate
        worst_5 = sorted(worst_5, key=lambda x: x["pass_rate"])  # Lowest first
        self.worst_chart.plot_pass_rate_bars(
            models=[m["model"] for m in worst_5],
            pass_rates=[m["pass_rate"] for m in worst_5],
            sample_counts=[m["total"] for m in worst_5],
            title="Bottom 5 Models",
            highlight_threshold=80.0
        )

        # Update ML summary
        self._update_ml_summary(alert_models)

        # Update status
        self.status_label.configure(text=f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    def _update_detail_display(
        self,
        trend_data: Dict[str, Any],
        model_alerts: Optional[Dict[str, Any]],
        ml_recommendations: Optional[Dict[str, Any]],
        model_names: List[str],
        model_stats: Optional[Dict[str, Any]] = None
    ):
        """Update detail display with loaded data."""
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

        # Calculate stats
        sigma_values = [d["sigma_gradient"] for d in data_points if d["sigma_gradient"] is not None]

        # Sigma pass rate (track-level: did sigma gradient pass?)
        sigma_pass_count = sum(1 for d in data_points if d.get("sigma_pass", False))
        sigma_pass_rate = (sigma_pass_count / total_samples * 100) if total_samples > 0 else 0

        # Overall pass rate - use model_stats from get_active_models_summary for consistency with alerts
        # This counts analysis-level pass (both sigma AND linearity must pass for all tracks)
        if model_stats:
            overall_pass_rate = model_stats.get("pass_rate", 0)
            total_analyses = model_stats.get("total", total_samples)
        else:
            # Fallback: count from track data (may differ from analysis-level count)
            overall_pass_count = sum(1 for d in data_points if d.get("status") == "PASS")
            overall_pass_rate = (overall_pass_count / total_samples * 100) if total_samples > 0 else 0
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
        self.detail_stat_labels["sigma_pass_rate"].configure(
            text=f"{sigma_pass_rate:.1f}%",
            text_color="#27ae60" if sigma_pass_rate >= 90 else "#f39c12" if sigma_pass_rate >= 80 else "#e74c3c"
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
        self.detail_stat_labels["status"].configure(text=status, text_color=status_color)

        # Update scatter chart
        dates = [d["date"].strftime("%m/%d") if hasattr(d["date"], 'strftime') else str(d["date"])[:5] for d in data_points]
        pass_flags = [d.get("sigma_pass", False) for d in data_points]

        # Rolling average values
        rolling_vals = None
        if rolling_averages:
            # Calculate sigma rolling average (not pass rate)
            # Use a simple moving average of sigma values
            window = min(self.rolling_window, len(sigma_values))
            if window > 1:
                rolling_vals = []
                for i in range(len(sigma_values)):
                    start = max(0, i - window + 1)
                    window_vals = sigma_values[start:i+1]
                    rolling_vals.append(np.mean(window_vals))

        self.scatter_chart.plot_sigma_scatter(
            dates=dates,
            sigma_values=sigma_values,
            pass_flags=pass_flags,
            threshold=threshold,
            rolling_avg=rolling_vals,
            title=f"Sigma Gradient Trend - {self.selected_model}",
            ylabel="Sigma Gradient"
        )

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
        """Update ML summary text for all models view."""
        self.ml_text.configure(state="normal")
        self.ml_text.delete("1.0", "end")

        if not alert_models:
            self.ml_text.insert("end", "All models performing within acceptable parameters.\n\n")
            self.ml_text.insert("end", "No immediate action required. Continue monitoring for changes.")
        else:
            self.ml_text.insert("end", f"Summary of {len(alert_models)} models requiring attention:\n\n")

            # Count by alert type
            low_pass_rate = sum(1 for a in alert_models for al in a.get("alerts", []) if al["type"] == "LOW_PASS_RATE")
            trending_worse = sum(1 for a in alert_models for al in a.get("alerts", []) if al["type"] == "TRENDING_WORSE")
            high_variance = sum(1 for a in alert_models for al in a.get("alerts", []) if al["type"] == "HIGH_VARIANCE")

            if low_pass_rate:
                self.ml_text.insert("end", f"Low Pass Rate: {low_pass_rate} models below 80%\n")
            if trending_worse:
                self.ml_text.insert("end", f"Trending Worse: {trending_worse} models declining\n")
            if high_variance:
                self.ml_text.insert("end", f"High Variance: {high_variance} models unstable\n")

            self.ml_text.insert("end", "\nRecommendation: Review flagged models for process adjustments.")

        self.ml_text.configure(state="disabled")

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
            self.detail_ml_text.insert("end", "Train the threshold optimizer to see model-specific recommendations.")
        else:
            threshold = ml_recommendations.get("recommended_threshold", 0)
            confidence = ml_recommendations.get("confidence", 0)
            basis = ml_recommendations.get("basis", "historical data")

            self.detail_ml_text.insert("end", f"Recommended Threshold:\n")
            self.detail_ml_text.insert("end", f"  {threshold:.6f}\n\n")
            self.detail_ml_text.insert("end", f"Confidence: {confidence:.1%}\n")
            self.detail_ml_text.insert("end", f"Based on: {basis}\n")

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
        self._refresh_data()
