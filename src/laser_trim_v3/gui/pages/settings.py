"""
Settings Page - Configuration.

Manage database path, export location, ML training, theme.
"""

import customtkinter as ctk
import logging
from tkinter import filedialog, messagebox
from pathlib import Path

logger = logging.getLogger(__name__)


class SettingsPage(ctk.CTkFrame):
    """
    Settings page for configuration.

    Features:
    - Database path selector
    - Export location
    - ML training trigger
    - Theme toggle (dark/light)
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

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
        self._create_section(container, "Database", [
            ("Database Path", self.app.config.database.path, self._change_database_path),
        ])

        # Processing section
        self._create_section(container, "Processing", [
            ("Batch Size", str(self.app.config.processing.batch_size), None),
            ("Turbo Mode Threshold", str(self.app.config.processing.turbo_mode_threshold), None),
        ])

        # ML section
        ml_frame = ctk.CTkFrame(container)
        ml_frame.grid(sticky="ew", padx=10, pady=10)
        ml_frame.grid_columnconfigure(1, weight=1)

        ml_label = ctk.CTkLabel(
            ml_frame,
            text="Machine Learning",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        ml_label.grid(row=0, column=0, columnspan=2, padx=15, pady=(15, 10), sticky="w")

        # ML enabled checkbox
        self.ml_enabled_var = ctk.BooleanVar(value=self.app.config.ml.enabled)
        ml_check = ctk.CTkCheckBox(
            ml_frame,
            text="Enable ML Features",
            variable=self.ml_enabled_var,
            command=self._toggle_ml
        )
        ml_check.grid(row=1, column=0, columnspan=2, padx=15, pady=5, sticky="w")

        # Train models button
        train_btn = ctk.CTkButton(
            ml_frame,
            text="Train ML Models",
            command=self._train_models,
            width=150
        )
        train_btn.grid(row=2, column=0, padx=15, pady=(10, 15), sticky="w")

        ml_status = ctk.CTkLabel(
            ml_frame,
            text="Models not trained",
            text_color="gray"
        )
        ml_status.grid(row=2, column=1, padx=15, pady=(10, 15), sticky="w")

        # Appearance section
        appearance_frame = ctk.CTkFrame(container)
        appearance_frame.grid(sticky="ew", padx=10, pady=10)
        appearance_frame.grid_columnconfigure(1, weight=1)

        appearance_label = ctk.CTkLabel(
            appearance_frame,
            text="Appearance",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        appearance_label.grid(row=0, column=0, columnspan=2, padx=15, pady=(15, 10), sticky="w")

        theme_label = ctk.CTkLabel(appearance_frame, text="Theme:")
        theme_label.grid(row=1, column=0, padx=15, pady=10, sticky="w")

        self.theme_dropdown = ctk.CTkOptionMenu(
            appearance_frame,
            values=["Dark", "Light", "System"],
            command=self._change_theme
        )
        self.theme_dropdown.set(self.app.config.gui.theme.capitalize())
        self.theme_dropdown.grid(row=1, column=1, padx=15, pady=10, sticky="w")

        # About section
        about_frame = ctk.CTkFrame(container)
        about_frame.grid(sticky="ew", padx=10, pady=10)

        about_label = ctk.CTkLabel(
            about_frame,
            text="About",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        about_label.pack(padx=15, pady=(15, 10), anchor="w")

        version_info = ctk.CTkLabel(
            about_frame,
            text=f"Laser Trim Analyzer v{self.app.config.version}\n\n"
                 "A simplified, ML-integrated application for\n"
                 "laser trim data analysis.",
            text_color="gray",
            justify="left"
        )
        version_info.pack(padx=15, pady=(0, 15), anchor="w")

    def _create_section(self, parent, title: str, items: list):
        """Create a settings section."""
        frame = ctk.CTkFrame(parent)
        frame.grid(sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        title_label = ctk.CTkLabel(
            frame,
            text=title,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w")

        for i, (label, value, callback) in enumerate(items, start=1):
            item_label = ctk.CTkLabel(frame, text=f"{label}:")
            item_label.grid(row=i, column=0, padx=15, pady=5, sticky="w")

            value_label = ctk.CTkLabel(frame, text=str(value), text_color="gray")
            value_label.grid(row=i, column=1, padx=10, pady=5, sticky="w")

            if callback:
                change_btn = ctk.CTkButton(
                    frame,
                    text="Change",
                    command=callback,
                    width=80
                )
                change_btn.grid(row=i, column=2, padx=15, pady=5)

    def _change_database_path(self):
        """Change the database path."""
        path = filedialog.asksaveasfilename(
            title="Select Database Location",
            defaultextension=".db",
            filetypes=[("SQLite Database", "*.db"), ("All files", "*.*")]
        )
        if path:
            self.app.config.database.path = Path(path)
            logger.info(f"Database path changed to: {path}")
            messagebox.showinfo("Settings", f"Database path changed to:\n{path}\n\nRestart required for changes to take effect.")

    def _toggle_ml(self):
        """Toggle ML features."""
        self.app.config.ml.enabled = self.ml_enabled_var.get()
        logger.info(f"ML features {'enabled' if self.app.config.ml.enabled else 'disabled'}")

    def _train_models(self):
        """Trigger ML model training."""
        # TODO: Implement ML training
        messagebox.showinfo("ML Training", "ML model training not yet implemented in v3.\n\nThis will train:\n- Threshold Optimizer\n- Drift Detector")
        logger.info("ML training requested")

    def _change_theme(self, theme: str):
        """Change the application theme."""
        theme_lower = theme.lower()
        ctk.set_appearance_mode(theme_lower)
        self.app.config.gui.theme = theme_lower
        logger.info(f"Theme changed to: {theme_lower}")

    def on_show(self):
        """Called when the page is shown."""
        logger.debug("Settings page shown")
