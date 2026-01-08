"""
Settings Page - Configuration.

Manage database path, export location, ML training, theme.
"""

import customtkinter as ctk
import logging
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Optional, Any

from laser_trim_analyzer.utils.threads import get_thread_manager

logger = logging.getLogger(__name__)


class SettingsPage(ctk.CTkFrame):
    """
    Settings page for configuration.

    Features:
    - Database path selector
    - Default export location
    - ML training trigger with status
    - Theme toggle (dark/light)
    - Version info
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.export_path: Optional[Path] = None
        self._ml_manager: Optional[Any] = None  # Set during training

        self._create_ui()

    def _create_ui(self):
        """Create the settings page UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkLabel(
            self,
            text="Settings",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        # Settings container
        container = ctk.CTkScrollableFrame(self)
        container.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        container.grid_columnconfigure(0, weight=1)

        # Database section
        self._create_database_section(container)

        # Export section
        self._create_export_section(container)

        # Processing section
        self._create_processing_section(container)

        # ML section
        self._create_ml_section(container)

        # Appearance section
        self._create_appearance_section(container)

        # About section
        self._create_about_section(container)

    def _create_database_section(self, container):
        """Create database settings section."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            frame,
            text="Database",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w")

        # Current path
        ctk.CTkLabel(frame, text="Database Path:").grid(row=1, column=0, padx=15, pady=5, sticky="w")

        self.db_path_label = ctk.CTkLabel(
            frame,
            text=str(self.app.config.database.path),
            text_color="gray",
            wraplength=400
        )
        self.db_path_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkButton(
            frame,
            text="Change",
            command=self._change_database_path,
            width=80
        ).grid(row=1, column=2, padx=15, pady=5)

        # Database info
        self.db_info_label = ctk.CTkLabel(
            frame,
            text="",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        )
        self.db_info_label.grid(row=2, column=0, columnspan=3, padx=15, pady=(0, 15), sticky="w")

    def _create_export_section(self, container):
        """Create export settings section."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            frame,
            text="Export",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w")

        # Default export location
        ctk.CTkLabel(frame, text="Default Export Location:").grid(row=1, column=0, padx=15, pady=5, sticky="w")

        self.export_path_label = ctk.CTkLabel(
            frame,
            text="Not set (will ask each time)",
            text_color="gray",
            wraplength=400
        )
        self.export_path_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkButton(
            frame,
            text="Set",
            command=self._set_export_path,
            width=80
        ).grid(row=1, column=2, padx=15, pady=5)

        # Include raw data checkbox
        self.include_raw_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            frame,
            text="Include raw position/error data in exports",
            variable=self.include_raw_var
        ).grid(row=2, column=0, columnspan=3, padx=15, pady=(5, 15), sticky="w")

    def _create_processing_section(self, container):
        """Create processing settings section."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            frame,
            text="Processing",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=2, padx=15, pady=(15, 10), sticky="w")

        settings = [
            ("Batch Size:", str(self.app.config.processing.batch_size)),
            ("Turbo Mode Threshold:", f"{self.app.config.processing.turbo_mode_threshold} files"),
        ]

        for i, (label, value) in enumerate(settings, start=1):
            ctk.CTkLabel(frame, text=label).grid(row=i, column=0, padx=15, pady=5, sticky="w")
            ctk.CTkLabel(frame, text=value, text_color="gray").grid(row=i, column=1, padx=10, pady=(5, 15 if i == len(settings) else 5), sticky="w")

    def _create_ml_section(self, container):
        """Create ML settings section with per-model training."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            frame,
            text="Machine Learning (Per-Model)",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w")

        # ML enabled checkbox
        self.ml_enabled_var = ctk.BooleanVar(value=self.app.config.ml.enabled)
        ctk.CTkCheckBox(
            frame,
            text="Enable ML Features (per-model thresholds, failure prediction, drift detection)",
            variable=self.ml_enabled_var,
            command=self._toggle_ml
        ).grid(row=1, column=0, columnspan=3, padx=15, pady=5, sticky="w")

        # Train models button
        self.train_btn = ctk.CTkButton(
            frame,
            text="Train Models",
            command=self._train_models,
            width=130,
            fg_color="#2980b9"
        )
        self.train_btn.grid(row=2, column=0, padx=15, pady=(10, 5), sticky="w")

        # Apply to DB button
        self.apply_btn = ctk.CTkButton(
            frame,
            text="Apply to DB",
            command=self._apply_ml_to_db,
            width=130,
            fg_color="#27ae60"
        )
        self.apply_btn.grid(row=2, column=1, padx=5, pady=(10, 5), sticky="w")

        self.ml_status_label = ctk.CTkLabel(
            frame,
            text="Status: Not trained",
            text_color="gray"
        )
        self.ml_status_label.grid(row=2, column=2, padx=15, pady=(10, 5), sticky="w")

        # Training requirements note
        ctk.CTkLabel(
            frame,
            text="Train Models: Learns per-model thresholds, failure predictors, and drift baselines.\n"
                 "Apply to DB: Updates all records with learned thresholds and failure probabilities.",
            text_color="gray",
            font=ctk.CTkFont(size=11),
            justify="left"
        ).grid(row=3, column=0, columnspan=3, padx=15, pady=(0, 10), sticky="w")

        # Progress bar (hidden initially)
        self.ml_progress = ctk.CTkProgressBar(frame)
        self.ml_progress.grid(row=4, column=0, columnspan=3, padx=15, pady=5, sticky="ew")
        self.ml_progress.set(0)
        self.ml_progress.grid_remove()  # Hide initially

        self.ml_progress_label = ctk.CTkLabel(
            frame,
            text="",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        )
        self.ml_progress_label.grid(row=5, column=0, columnspan=3, padx=15, pady=(0, 5), sticky="w")
        self.ml_progress_label.grid_remove()  # Hide initially

        # Model status summary (collapsed by default)
        self.model_status_label = ctk.CTkLabel(
            frame,
            text="",
            text_color="gray",
            font=ctk.CTkFont(size=11),
            justify="left"
        )
        self.model_status_label.grid(row=6, column=0, columnspan=3, padx=15, pady=(5, 15), sticky="w")

    def _create_appearance_section(self, container):
        """Create appearance settings section."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            frame,
            text="Appearance",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=2, padx=15, pady=(15, 10), sticky="w")

        ctk.CTkLabel(frame, text="Theme:").grid(row=1, column=0, padx=15, pady=(5, 15), sticky="w")

        self.theme_dropdown = ctk.CTkOptionMenu(
            frame,
            values=["Dark", "Light", "System"],
            command=self._change_theme,
            width=150
        )
        self.theme_dropdown.set(self.app.config.gui.theme.capitalize())
        self.theme_dropdown.grid(row=1, column=1, padx=10, pady=(5, 15), sticky="w")

    def _create_about_section(self, container):
        """Create about section."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)

        title = ctk.CTkLabel(
            frame,
            text="About",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.pack(padx=15, pady=(15, 10), anchor="w")

        version_info = ctk.CTkLabel(
            frame,
            text=f"Laser Trim Analyzer v{self.app.config.version}\n\n"
                 "A streamlined, ML-integrated application for laser trim data analysis.\n\n"
                 "Key Features:\n"
                 "  • Per-model ML threshold optimization\n"
                 "  • Drift detection with CUSUM/EWMA\n"
                 "  • Memory-safe processing for 8GB systems\n"
                 "  • Excel export with batch processing",
            text_color="gray",
            justify="left"
        )
        version_info.pack(padx=15, pady=(0, 15), anchor="w")

    def _change_database_path(self):
        """Change the database path."""
        path = filedialog.asksaveasfilename(
            title="Select Database Location",
            defaultextension=".db",
            initialfile="laser_trim.db",
            filetypes=[("SQLite Database", "*.db"), ("All files", "*.*")]
        )
        if path:
            self.app.config.database.path = Path(path)
            self.db_path_label.configure(text=str(path))
            logger.info(f"Database path changed to: {path}")
            messagebox.showinfo(
                "Database Path Changed",
                f"Database path changed to:\n{path}\n\nRestart required for changes to take effect."
            )

    def _set_export_path(self):
        """Set the default export location."""
        path = filedialog.askdirectory(title="Select Default Export Location")
        if path:
            self.export_path = Path(path)
            self.export_path_label.configure(text=str(path))
            logger.info(f"Default export path set to: {path}")

    def _toggle_ml(self):
        """Toggle ML features."""
        self.app.config.ml.enabled = self.ml_enabled_var.get()
        logger.info(f"ML features {'enabled' if self.app.config.ml.enabled else 'disabled'}")

    def _train_models(self):
        """Trigger per-model ML training."""
        self.train_btn.configure(state="disabled")
        self.apply_btn.configure(state="disabled")
        self.ml_status_label.configure(text="Status: Training...", text_color="gray")
        self.ml_progress.grid()  # Show progress bar
        self.ml_progress_label.grid()
        self.ml_progress.set(0)

        get_thread_manager().start_thread(target=self._run_training, name="ml-training")

    def _run_training(self):
        """Run per-model ML training in background thread."""
        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.ml import MLManager, TrainingProgress

            db = get_database()

            # Create ML manager
            ml_manager = MLManager(db)

            # Progress callback
            def on_progress(progress: TrainingProgress):
                if progress.models_total > 0:
                    pct = progress.models_complete / progress.models_total
                    self.after(0, lambda p=pct, m=progress.message: self._update_training_progress(p, m))

            # Train all models
            results = ml_manager.train_all_models(
                min_samples=20,
                progress_callback=on_progress
            )

            # Save trained state
            ml_manager.save_all()

            # Store manager for apply step
            self._ml_manager = ml_manager

            # Count results
            trained = sum(1 for r in results.values() if r.success and r.threshold_calculated)
            predictors = sum(1 for r in results.values() if r.predictor_trained)
            drift = sum(1 for r in results.values() if r.drift_baseline_set)

            msg = f"Trained {trained} models ({predictors} predictors, {drift} drift baselines)"
            status_details = self._format_training_status(results)

            self.after(0, lambda: self._on_training_complete(True, msg, status_details))

        except Exception as e:
            logger.exception(f"ML training failed: {e}")
            self.after(0, lambda: self._on_training_complete(False, str(e)))

    def _update_training_progress(self, progress: float, message: str):
        """Update training progress bar."""
        self.ml_progress.set(progress)
        self.ml_progress_label.configure(text=message)

    def _format_training_status(self, results: dict) -> str:
        """Format training results for display."""
        lines = []
        for model_name, result in sorted(results.items()):
            if result.success:
                thresh = f"T={result.threshold_value:.5f}" if result.threshold_value else "T=N/A"
                pred = "P" if result.predictor_trained else "-"
                drift = "D" if result.drift_baseline_set else "-"
                lines.append(f"  {model_name}: {thresh} [{pred}{drift}] ({result.n_samples} samples)")
            else:
                lines.append(f"  {model_name}: {result.error or 'insufficient data'}")
        return "\n".join(lines[:10])  # Limit to first 10 for display

    def _apply_ml_to_db(self):
        """Apply learned ML to database."""
        if not hasattr(self, '_ml_manager') or not self._ml_manager.trained_models:
            from tkinter import messagebox
            messagebox.showwarning(
                "No Trained Models",
                "Please train models first before applying to database."
            )
            return

        self.train_btn.configure(state="disabled")
        self.apply_btn.configure(state="disabled")
        self.ml_status_label.configure(text="Status: Applying...", text_color="gray")
        self.ml_progress.grid()
        self.ml_progress_label.grid()
        self.ml_progress.set(0)

        get_thread_manager().start_thread(target=self._run_apply_ml, name="apply-ml")

    def _run_apply_ml(self):
        """Run ML application in background thread."""
        try:
            from laser_trim_analyzer.ml import ApplyProgress

            # Progress callback
            def on_progress(progress: ApplyProgress):
                if progress.records_total > 0:
                    pct = progress.records_complete / progress.records_total
                    self.after(0, lambda p=pct, m=progress.message: self._update_training_progress(p, m))

            # Apply to database
            counts = self._ml_manager.apply_to_database(
                progress_callback=on_progress,
                run_drift_detection=True
            )

            # Format result message
            msg = f"Updated {counts['updated']} tracks"
            if counts['drift_alerts']:
                msg += f", {len(counts['drift_alerts'])} drift alerts"
            if counts['errors'] > 0:
                msg += f", {counts['errors']} errors"

            self.after(0, lambda: self._on_apply_complete(True, msg))

        except Exception as e:
            logger.exception(f"ML apply failed: {e}")
            self.after(0, lambda: self._on_apply_complete(False, str(e)))

    def _on_apply_complete(self, success: bool, message: str):
        """Handle apply completion."""
        self.train_btn.configure(state="normal")
        self.apply_btn.configure(state="normal")
        self.ml_progress.grid_remove()
        self.ml_progress_label.grid_remove()

        if success:
            self.ml_status_label.configure(text=f"Applied: {message}", text_color="#27ae60")
            logger.info(f"ML apply successful: {message}")
        else:
            self.ml_status_label.configure(text=f"Apply failed: {message[:40]}...", text_color="#e74c3c")
            logger.error(f"ML apply failed: {message}")

    def _on_training_complete(self, success: bool, message: str, status_details: str = ""):
        """Handle training completion."""
        self.train_btn.configure(state="normal")
        self.apply_btn.configure(state="normal")
        self.ml_progress.grid_remove()
        self.ml_progress_label.grid_remove()

        if success:
            self.ml_status_label.configure(text=f"Status: {message}", text_color="#27ae60")
            if status_details:
                self.model_status_label.configure(text=status_details)
            logger.info(f"ML training successful: {message}")
        else:
            self.ml_status_label.configure(text=f"Status: Failed - {message[:40]}...", text_color="#e74c3c")
            logger.error(f"ML training failed: {message}")

    def _change_theme(self, theme: str):
        """Change the application theme."""
        theme_lower = theme.lower()
        ctk.set_appearance_mode(theme_lower)
        self.app.config.gui.theme = theme_lower
        logger.info(f"Theme changed to: {theme_lower}")

    def _update_database_info(self):
        """Update database info display."""
        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            stats = db.get_dashboard_stats(days_back=365)
            total = stats.get("total_files", 0)
            self.db_info_label.configure(
                text=f"Connected - {total} files in database"
            )
        except Exception as e:
            self.db_info_label.configure(text=f"Not connected: {str(e)[:30]}")

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Settings page shown")
        # Update database info
        self._update_database_info()

        # Check ML model status from database
        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.ml import MLManager

            db = get_database()
            ml_manager = MLManager(db)
            ml_manager.load_all()

            if ml_manager.trained_models:
                self.ml_status_label.configure(
                    text=f"Status: {len(ml_manager.trained_models)} models trained",
                    text_color="#27ae60"
                )
            else:
                self.ml_status_label.configure(text="Status: Not trained", text_color="gray")
        except Exception:
            self.ml_status_label.configure(text="Status: Not trained", text_color="gray")
