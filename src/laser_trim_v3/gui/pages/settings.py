"""
Settings Page - Configuration.

Manage database path, export location, ML training, theme.
"""

import threading
import customtkinter as ctk
import logging
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Optional

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
        """Create ML settings section."""
        frame = ctk.CTkFrame(container)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            frame,
            text="Machine Learning",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w")

        # ML enabled checkbox
        self.ml_enabled_var = ctk.BooleanVar(value=self.app.config.ml.enabled)
        ctk.CTkCheckBox(
            frame,
            text="Enable ML Features (threshold optimization, drift detection)",
            variable=self.ml_enabled_var,
            command=self._toggle_ml
        ).grid(row=1, column=0, columnspan=3, padx=15, pady=5, sticky="w")

        # Train models button
        self.train_btn = ctk.CTkButton(
            frame,
            text="Train ML Models",
            command=self._train_models,
            width=150
        )
        self.train_btn.grid(row=2, column=0, padx=15, pady=(10, 5), sticky="w")

        self.ml_status_label = ctk.CTkLabel(
            frame,
            text="Status: Not trained",
            text_color="gray"
        )
        self.ml_status_label.grid(row=2, column=1, padx=15, pady=(10, 5), sticky="w")

        # Training requirements note
        ctk.CTkLabel(
            frame,
            text="Note: Trains ThresholdOptimizer + DriftDetector. Requires 50+ files.",
            text_color="gray",
            font=ctk.CTkFont(size=11),
            justify="left"
        ).grid(row=3, column=0, columnspan=3, padx=15, pady=(0, 10), sticky="w")

        # Re-analyze all database button
        self.reanalyze_btn = ctk.CTkButton(
            frame,
            text="Re-analyze All DB",
            command=self._reanalyze_database,
            width=150
        )
        self.reanalyze_btn.grid(row=4, column=0, padx=15, pady=(10, 5), sticky="w")

        self.reanalyze_status_label = ctk.CTkLabel(
            frame,
            text="",
            text_color="gray"
        )
        self.reanalyze_status_label.grid(row=4, column=1, padx=15, pady=(10, 5), sticky="w")

        # Re-analyze note
        ctk.CTkLabel(
            frame,
            text="Re-analyze All: Updates ALL database records with current ML thresholds.\n"
                 "This ensures consistent pass/fail status across your entire history.",
            text_color="gray",
            font=ctk.CTkFont(size=11),
            justify="left"
        ).grid(row=5, column=0, columnspan=3, padx=15, pady=(0, 15), sticky="w")

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
                 "A simplified, ML-integrated application for laser trim data analysis.\n\n"
                 "v3 Redesign Goals:\n"
                 "  • ~30 files instead of 110 (73% reduction)\n"
                 "  • ~7,000 lines instead of 144,000 (95% reduction)\n"
                 "  • Memory-safe processing for 8GB systems\n"
                 "  • ML-integrated with automatic formula fallback",
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
        """Trigger ML model training."""
        self.train_btn.configure(state="disabled")
        self.ml_status_label.configure(text="Status: Training...")

        thread = threading.Thread(target=self._run_training, daemon=True)
        thread.start()

    def _run_training(self):
        """Run ML training in background thread - trains BOTH models."""
        try:
            import numpy as np
            import pandas as pd
            from laser_trim_v3.database import get_database
            from laser_trim_v3.ml import ThresholdOptimizer
            from laser_trim_v3.ml.drift import DriftDetector
            from laser_trim_v3.config import get_config

            db = get_database()
            config = get_config()

            # Get training data from database
            training_data = db.get_trend_data(days_back=365 * 10, limit=50000)  # Get all available data

            if len(training_data) < 50:
                self.after(0, lambda: self._on_training_complete(
                    False,
                    f"Need at least 50 files, found {len(training_data)}"
                ))
                return

            # Prepare training DataFrame for ThresholdOptimizer
            # Required columns: model, unit_length, linearity_spec, sigma_gradient
            records = []
            sigma_values = []  # For drift detector baseline
            for record in training_data:
                sigma_gradient = record.get("sigma_gradient")
                sigma_threshold = record.get("sigma_threshold")
                model = record.get("model")

                # Skip records without required data
                if sigma_gradient is None or sigma_threshold is None or not model:
                    continue

                records.append({
                    "model": model,
                    "unit_length": record.get("unit_length") or record.get("travel_length") or 100.0,
                    "linearity_spec": record.get("linearity_spec") or 0.01,
                    "sigma_gradient": sigma_gradient,
                    "sigma_threshold": sigma_threshold,
                    "sigma_pass": record.get("sigma_pass", sigma_gradient <= sigma_threshold),
                })
                sigma_values.append(sigma_gradient)

            if len(records) < 50:
                self.after(0, lambda: self._on_training_complete(
                    False,
                    f"Only {len(records)} valid training samples (need 50+)"
                ))
                return

            # Create DataFrame
            df = pd.DataFrame(records)

            # Log data summary
            unique_models = df["model"].nunique()
            logger.info(f"Training with {len(df)} samples across {unique_models} models")

            # ===== 1. Train Threshold Optimizer =====
            optimizer = ThresholdOptimizer()
            result = optimizer.train(df)

            if not result.success:
                self.after(0, lambda: self._on_training_complete(
                    False,
                    result.error or "Threshold training failed"
                ))
                return

            # Save the trained threshold model
            model_path = config.models.path / "threshold_optimizer.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            optimizer.save(model_path)
            logger.info(f"Threshold optimizer saved to {model_path}")

            # ===== 2. Train Drift Detector =====
            drift_msg = ""
            try:
                drift_detector = DriftDetector()

                # Set baseline from sigma values
                sigma_array = np.array(sigma_values)
                if drift_detector.set_baseline(sigma_array):
                    # Train anomaly detector on numeric features
                    drift_detector.train_anomaly_detector(df[["sigma_gradient", "unit_length", "linearity_spec"]])

                    # Save drift detector
                    drift_path = config.models.path / "drift_detector.pkl"
                    drift_detector.save(drift_path)
                    drift_msg = ", Drift detector trained"
                    logger.info(f"Drift detector saved to {drift_path}")
            except Exception as e:
                logger.warning(f"Drift detector training failed (non-critical): {e}")
                drift_msg = ", Drift detector skipped"

            # Success message with metrics
            msg = f"Trained on {result.n_samples} samples, R²={result.r2_score:.3f}, {unique_models} models{drift_msg}"

            self.after(0, lambda: self._on_training_complete(True, msg))

        except Exception as e:
            logger.exception(f"ML training failed: {e}")
            self.after(0, lambda: self._on_training_complete(False, str(e)))

    def _on_training_complete(self, success: bool, message: str):
        """Handle training completion."""
        self.train_btn.configure(state="normal")

        if success:
            self.ml_status_label.configure(text=f"Status: {message}", text_color="#27ae60")
            logger.info(f"ML training successful: {message}")
        else:
            self.ml_status_label.configure(text=f"Status: Failed - {message[:40]}...", text_color="#e74c3c")
            logger.error(f"ML training failed: {message}")

    def _reanalyze_database(self):
        """Trigger re-analysis of all database records with current ML thresholds."""
        from tkinter import messagebox

        # Confirm with user
        result = messagebox.askyesno(
            "Re-analyze All Database Records",
            "This will update ALL records in the database with the current ML thresholds.\n\n"
            "• Sigma thresholds will be recalculated for each record\n"
            "• Pass/Fail status will be updated based on new thresholds\n"
            "• Overall status will be recalculated\n\n"
            "This may take several minutes for large databases.\n\n"
            "Continue?"
        )

        if not result:
            return

        self.reanalyze_btn.configure(state="disabled")
        self.reanalyze_status_label.configure(text="Starting...", text_color="gray")

        thread = threading.Thread(target=self._run_reanalysis, daemon=True)
        thread.start()

    def _run_reanalysis(self):
        """Run database re-analysis in background thread."""
        try:
            from laser_trim_v3.database import get_database
            from laser_trim_v3.database.models import (
                AnalysisResult as DBAnalysisResult,
                TrackResult as DBTrackResult,
                StatusType as DBStatusType,
            )
            from laser_trim_v3.ml import ThresholdOptimizer
            from laser_trim_v3.config import get_config

            db = get_database()
            config = get_config()

            # Load the trained ML model
            optimizer = ThresholdOptimizer()
            model_path = config.models.path / "threshold_optimizer.pkl"

            if model_path.exists():
                optimizer.load(model_path)
                logger.info("Loaded trained ML model for re-analysis")
            else:
                logger.info("No trained ML model found, using formula-based thresholds")

            # Get all track records from database
            with db.session() as session:
                # Get count first
                total_tracks = session.query(DBTrackResult).count()
                total_analyses = session.query(DBAnalysisResult).count()

                if total_tracks == 0:
                    self.after(0, lambda: self._on_reanalysis_complete(
                        False, "No records in database"
                    ))
                    return

                self.after(0, lambda: self.reanalyze_status_label.configure(
                    text=f"Processing {total_tracks} tracks...", text_color="gray"
                ))

                # Process tracks in batches to avoid memory issues
                batch_size = 500
                updated_count = 0
                status_changed = 0

                # Get all analyses with their tracks
                analyses = session.query(DBAnalysisResult).all()

                for idx, analysis in enumerate(analyses):
                    analysis_changed = False

                    for track in analysis.tracks:
                        old_pass = track.sigma_pass
                        old_threshold = track.sigma_threshold

                        # Get parameters for threshold calculation
                        model_name = analysis.model or "UNKNOWN"
                        unit_length = track.unit_length or track.travel_length or 100.0
                        linearity_spec = track.linearity_spec or 0.01

                        # Calculate new threshold using ML model
                        new_threshold = optimizer.predict(
                            model=model_name,
                            unit_length=unit_length,
                            linearity_spec=linearity_spec
                        )

                        # Update threshold
                        track.sigma_threshold = new_threshold

                        # Recalculate pass/fail
                        if track.sigma_gradient is not None:
                            track.sigma_pass = track.sigma_gradient <= new_threshold
                        else:
                            track.sigma_pass = True  # No data = pass

                        # Recalculate track status based on both sigma and linearity
                        if track.sigma_pass and track.linearity_pass:
                            track.status = DBStatusType.PASS
                        elif not track.sigma_pass or not track.linearity_pass:
                            track.status = DBStatusType.FAIL
                        else:
                            track.status = DBStatusType.WARNING

                        updated_count += 1

                        if old_pass != track.sigma_pass:
                            status_changed += 1
                            analysis_changed = True

                    # Recalculate overall analysis status
                    if analysis_changed or True:  # Always recalculate to be safe
                        # Skip analyses with no tracks (keep existing ERROR status)
                        if not analysis.tracks:
                            if analysis.overall_status != DBStatusType.ERROR:
                                analysis.overall_status = DBStatusType.ERROR
                            continue

                        all_pass = all(t.sigma_pass and t.linearity_pass for t in analysis.tracks)
                        any_fail = any(not t.sigma_pass or not t.linearity_pass for t in analysis.tracks)

                        if all_pass:
                            analysis.overall_status = DBStatusType.PASS
                        elif any_fail:
                            analysis.overall_status = DBStatusType.FAIL
                        else:
                            analysis.overall_status = DBStatusType.WARNING

                    # Update progress periodically
                    if (idx + 1) % 100 == 0:
                        progress = int((idx + 1) / len(analyses) * 100)
                        self.after(0, lambda p=progress, u=updated_count: self.reanalyze_status_label.configure(
                            text=f"Progress: {p}% ({u} tracks updated)", text_color="gray"
                        ))

                # Commit all changes
                session.commit()

            # Success
            msg = f"Updated {updated_count} tracks across {total_analyses} files. {status_changed} pass/fail changes."
            self.after(0, lambda: self._on_reanalysis_complete(True, msg))

        except Exception as e:
            logger.exception(f"Database re-analysis failed: {e}")
            self.after(0, lambda: self._on_reanalysis_complete(False, str(e)))

    def _on_reanalysis_complete(self, success: bool, message: str):
        """Handle re-analysis completion."""
        self.reanalyze_btn.configure(state="normal")

        if success:
            self.reanalyze_status_label.configure(text=message, text_color="#27ae60")
            logger.info(f"Database re-analysis successful: {message}")
        else:
            self.reanalyze_status_label.configure(text=f"Failed: {message[:50]}...", text_color="#e74c3c")
            logger.error(f"Database re-analysis failed: {message}")

    def _change_theme(self, theme: str):
        """Change the application theme."""
        theme_lower = theme.lower()
        ctk.set_appearance_mode(theme_lower)
        self.app.config.gui.theme = theme_lower
        logger.info(f"Theme changed to: {theme_lower}")

    def _update_database_info(self):
        """Update database info display."""
        try:
            from laser_trim_v3.database import get_database
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

        # Check ML model status
        try:
            from laser_trim_v3.ml import ThresholdOptimizer
            from laser_trim_v3.config import get_config
            config = get_config()
            model_path = config.models.path / "threshold_optimizer.pkl"
            if model_path.exists():
                self.ml_status_label.configure(text="Status: Trained model loaded", text_color="#27ae60")
            else:
                self.ml_status_label.configure(text="Status: Not trained", text_color="gray")
        except Exception:
            self.ml_status_label.configure(text="Status: ML unavailable", text_color="gray")
