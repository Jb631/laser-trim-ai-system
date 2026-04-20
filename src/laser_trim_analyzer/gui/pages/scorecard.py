"""
Model Scorecard — single-page summary of a model's health and performance.

Sections:
1. Header: Model name, element type, product class
2. Key Metrics: Pass rate, Cpk, volume, avg deviation
3. Specifications: Linearity spec, type
4. Cpk Detail: Cpk/Ppk values, stats
5. Drift Status: Current status
"""

import logging
from typing import Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.widgets.scrollable_combobox import ScrollableComboBox

logger = logging.getLogger(__name__)


class ScorecardPage(ctk.CTkFrame):
    """Model scorecard — comprehensive single-model view."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.current_model: Optional[str] = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Model selector at top
        selector_frame = ctk.CTkFrame(self, fg_color="transparent")
        selector_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            selector_frame, text="Model Scorecard",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(side="left", padx=(0, 20))

        self.model_selector = ScrollableComboBox(
            selector_frame,
            values=["Select a model..."],
            command=self._on_model_selected,
            width=200,
        )
        self.model_selector.pack(side="left", padx=5)

        ctk.CTkButton(
            selector_frame, text="Refresh", width=80,
            command=self._refresh,
        ).pack(side="left", padx=5)

        # Scrollable content area
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        self.placeholder = ctk.CTkLabel(
            self.scroll_frame,
            text="Select a model to view its scorecard",
            font=ctk.CTkFont(size=14), text_color="gray",
        )
        self.placeholder.grid(row=0, column=0, pady=50)

    def on_show(self):
        """Called when page becomes visible."""
        self._populate_model_list()
        if self.current_model:
            self._load_scorecard(self.current_model)

    def show_model(self, model: str):
        """Show scorecard for a specific model (called from other pages)."""
        self.current_model = model
        self.model_selector.set(model)
        self._load_scorecard(model)

    def _populate_model_list(self):
        """Load available models into the dropdown."""
        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            models = db.get_models_list()
            if models:
                self.model_selector.configure(values=models)
        except Exception as e:
            logger.warning(f"Failed to load model list: {e}")

    def _on_model_selected(self, model: str):
        if model and model != "Select a model...":
            self.current_model = model
            self._load_scorecard(model)

    def _refresh(self):
        if self.current_model:
            self._load_scorecard(self.current_model)

    def _load_scorecard(self, model: str):
        """Load and display scorecard data for a model."""
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        ctk.CTkLabel(
            self.scroll_frame,
            text="Loading scorecard...",
            text_color="gray",
        ).grid(row=0, column=0, pady=20)

        def fetch():
            try:
                from laser_trim_analyzer.database import get_database
                db = get_database()
                data = db.get_model_scorecard_data(model, days_back=90)
                self.after(0, lambda: self._display_scorecard(data))
            except Exception as e:
                logger.error(f"Failed to load scorecard: {e}")
                self.after(0, lambda: self._display_scorecard_error(e))

        from laser_trim_analyzer.utils.threads import get_thread_manager
        get_thread_manager().start_thread(target=fetch, name="scorecard-load")

    def _display_scorecard(self, data: dict):
        """Display scorecard data on the UI thread."""
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self._render_scorecard(data)

    def _display_scorecard_error(self, error: Exception):
        """Display scorecard loading error on the UI thread."""
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        ctk.CTkLabel(
            self.scroll_frame,
            text=f"Error loading scorecard: {error}",
            text_color="#dc3545",
        ).grid(row=0, column=0, pady=20)

    def _render_scorecard(self, data: dict):
        """Render the scorecard sections from loaded data."""
        row = 0
        row = self._render_header(data, row)
        row = self._render_key_metrics(data, row)
        if data.get("spec"):
            row = self._render_specs(data, row)
        if data.get("cpk"):
            row = self._render_cpk(data, row)
        if data.get("drift_status"):
            row = self._render_drift(data, row)

    def _render_header(self, data: dict, row: int) -> int:
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text=data["model"],
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=0, column=0, padx=15, pady=(10, 2), sticky="w")

        spec = data.get("spec", {})
        if spec:
            parts = []
            if spec.get("element_type"):
                parts.append(spec["element_type"])
            if spec.get("product_class"):
                parts.append(f"Class {spec['product_class']}")
            if spec.get("linearity_type"):
                parts.append(spec["linearity_type"])
            if parts:
                ctk.CTkLabel(
                    frame, text=" | ".join(parts),
                    font=ctk.CTkFont(size=12), text_color="gray",
                ).grid(row=1, column=0, padx=15, pady=(0, 10), sticky="w")

        return row + 1

    def _render_key_metrics(self, data: dict, row: int) -> int:
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.grid(row=row, column=0, sticky="ew", pady=5)
        frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        pr = data.get("pass_rate", 0)
        pr_color = "#198754" if pr >= 80 else "#fd7e14" if pr >= 60 else "#dc3545"
        self._metric_card(frame, 0, f"{pr:.1f}%", "Pass Rate", pr_color)
        self._metric_card(frame, 1, f"{data.get('total', 0):,}", "Units (90d)", "#0d6efd")

        cpk = data.get("cpk", {})
        if cpk and cpk.get("cpk") is not None:
            cpk_val = cpk["cpk"]
            cpk_color = "#198754" if cpk_val >= 1.33 else "#fd7e14" if cpk_val >= 1.0 else "#dc3545"
            self._metric_card(frame, 2, f"{cpk_val:.2f}", f"Cpk ({cpk.get('rating', '')})", cpk_color)
        else:
            self._metric_card(frame, 2, "N/A", "Cpk", "#6c757d")

        avg_dev = data.get("avg_deviation")
        if avg_dev is not None:
            self._metric_card(frame, 3, f"{avg_dev:.4f}", "Avg Error", "#6c757d")
        else:
            self._metric_card(frame, 3, "N/A", "Avg Error", "#6c757d")

        return row + 1

    def _metric_card(self, parent, col: int, value: str, label: str, color: str):
        card = ctk.CTkFrame(parent)
        card.grid(row=0, column=col, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=22, weight="bold"),
                    text_color=color).pack(padx=10, pady=(8, 0))
        ctk.CTkLabel(card, text=label, font=ctk.CTkFont(size=10),
                    text_color="gray").pack(padx=10, pady=(0, 8))

    def _render_specs(self, data: dict, row: int) -> int:
        spec = data["spec"]
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.grid(row=row, column=0, sticky="ew", pady=5)

        ctk.CTkLabel(frame, text="Specifications",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(padx=15, pady=(10, 5), anchor="w")

        details = []
        if spec.get("linearity_spec_pct"):
            details.append(f"Linearity: +/-{spec['linearity_spec_pct']}%")
        if spec.get("linearity_type"):
            details.append(f"Type: {spec['linearity_type']}")
        for d in details:
            ctk.CTkLabel(frame, text=f"  {d}", font=ctk.CTkFont(size=11)).pack(padx=15, pady=1, anchor="w")
        ctk.CTkLabel(frame, text="", height=5).pack()
        return row + 1

    def _render_cpk(self, data: dict, row: int) -> int:
        cpk = data["cpk"]
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.grid(row=row, column=0, sticky="ew", pady=5)

        ctk.CTkLabel(frame, text="Process Capability",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(padx=15, pady=(10, 5), anchor="w")

        lines = [
            f"  Cpk = {cpk['cpk']:.3f}   (within-subgroup)" if cpk.get('cpk') else None,
            f"  Ppk = {cpk['ppk']:.3f}   (overall)" if cpk.get('ppk') else None,
            f"  Cp  = {cpk['cp']:.3f}   (potential)" if cpk.get('cp') else None,
            f"  Mean = {cpk['mean']:.4f},  Sigma = {cpk['std_overall']:.4f}" if cpk.get('mean') is not None else None,
            f"  Samples: {cpk.get('n_samples', 0):,}",
            f"  Spec limits: {cpk.get('lsl', 0):.2f} to {cpk.get('usl', 0):.2f}",
        ]
        for line in lines:
            if line:
                ctk.CTkLabel(frame, text=line, font=ctk.CTkFont(size=11)).pack(padx=15, pady=1, anchor="w")
        ctk.CTkLabel(frame, text="", height=5).pack()
        return row + 1

    def _render_drift(self, data: dict, row: int) -> int:
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.grid(row=row, column=0, sticky="ew", pady=5)

        ctk.CTkLabel(frame, text="Drift Status",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(padx=15, pady=(10, 5), anchor="w")

        status = data.get("drift_status", "Unknown")
        color = "#198754" if status == "stable" else "#dc3545"
        ctk.CTkLabel(frame, text=f"  Status: {status}",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=color).pack(padx=15, pady=5, anchor="w")
        ctk.CTkLabel(frame, text="", height=5).pack()
        return row + 1
