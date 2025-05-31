"""
SettingsPage - In-app settings page

Provides a settings interface within the main application window.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Dict, Any, Optional

from laser_trim_analyzer.core.config import Config
from laser_trim_analyzer.gui.dialogs.settings_dialog import SettingsDialog


class SettingsPage(ttk.Frame):
    """
    Settings page for the main application.

    Provides quick access to common settings and
    links to the full settings dialog.
    """

    def __init__(self, parent, main_window):
        """
        Initialize SettingsPage.

        Args:
            parent: Parent widget
            main_window: Reference to main application window
        """
        super().__init__(parent)

        self.main_window = main_window
        self.config = main_window.config

        self._setup_ui()
        self._load_current_settings()

    def _setup_ui(self):
        """Set up the settings page UI."""
        # Title
        title_frame = ttk.Frame(self)
        title_frame.pack(fill='x', padx=20, pady=20)

        ttk.Label(
            title_frame,
            text="Settings",
            font=('Segoe UI', 24, 'bold')
        ).pack(side='left')

        ttk.Button(
            title_frame,
            text="Advanced Settings...",
            command=self._open_full_settings
        ).pack(side='right')

        # Quick settings container
        container = ttk.Frame(self)
        container.pack(fill='both', expand=True, padx=20)

        # Processing settings
        self._create_processing_section(container)

        # Database settings
        self._create_database_section(container)

        # ML settings
        self._create_ml_section(container)

        # Appearance settings
        self._create_appearance_section(container)

    def _create_processing_section(self, parent):
        """Create processing settings section."""
        frame = ttk.LabelFrame(parent, text="Processing", padding=15)
        frame.pack(fill='x', pady=10)

        # Max workers
        workers_frame = ttk.Frame(frame)
        workers_frame.pack(fill='x', pady=5)

        ttk.Label(workers_frame, text="Parallel Workers:").pack(side='left')

        self.workers_var = tk.IntVar(value=self.config.processing.max_workers)
        ttk.Spinbox(
            workers_frame,
            textvariable=self.workers_var,
            from_=1,
            to=16,
            width=10,
            command=self._on_workers_changed
        ).pack(side='left', padx=10)

        ttk.Label(
            workers_frame,
            text="(Higher = faster processing, more CPU usage)",
            font=('Segoe UI', 9),
            foreground='gray'
        ).pack(side='left')

        # Generate plots
        self.plots_var = tk.BooleanVar(value=self.config.processing.generate_plots)
        ttk.Checkbutton(
            frame,
            text="Generate analysis plots",
            variable=self.plots_var,
            command=self._on_plots_changed
        ).pack(anchor='w', pady=5)

        # Cache
        self.cache_var = tk.BooleanVar(value=self.config.processing.cache_enabled)
        ttk.Checkbutton(
            frame,
            text="Enable result caching",
            variable=self.cache_var,
            command=self._on_cache_changed
        ).pack(anchor='w', pady=5)

    def _create_database_section(self, parent):
        """Create database settings section."""
        frame = ttk.LabelFrame(parent, text="Database", padding=15)
        frame.pack(fill='x', pady=10)

        # Enable database
        self.db_var = tk.BooleanVar(value=self.config.database.enabled)
        db_check = ttk.Checkbutton(
            frame,
            text="Save results to database",
            variable=self.db_var,
            command=self._on_database_changed
        )
        db_check.pack(anchor='w', pady=5)

        # Database path
        path_frame = ttk.Frame(frame)
        path_frame.pack(fill='x', pady=5)

        ttk.Label(path_frame, text="Database:").pack(side='left')

        self.db_path_label = ttk.Label(
            path_frame,
            text=str(self.config.database.path),
            font=('Segoe UI', 9),
            foreground='#2196f3'
        )
        self.db_path_label.pack(side='left', padx=10)

        # Database status
        status_text = "Not connected"
        status_color = 'gray'

        if self.main_window.db_manager:
            status_text = "Connected"
            status_color = 'green'

        self.db_status_label = ttk.Label(
            frame,
            text=f"Status: {status_text}",
            font=('Segoe UI', 9),
            foreground=status_color
        )
        self.db_status_label.pack(anchor='w')

    def _create_ml_section(self, parent):
        """Create ML settings section."""
        frame = ttk.LabelFrame(parent, text="Machine Learning", padding=15)
        frame.pack(fill='x', pady=10)

        # Enable ML
        self.ml_var = tk.BooleanVar(value=self.config.ml.enabled)
        ml_check = ttk.Checkbutton(
            frame,
            text="Enable ML predictions",
            variable=self.ml_var,
            command=self._on_ml_changed
        )
        ml_check.pack(anchor='w', pady=5)

        # ML features
        features_frame = ttk.Frame(frame)
        features_frame.pack(fill='x', padx=20)

        self.failure_pred_var = tk.BooleanVar(
            value=self.config.ml.failure_prediction_enabled
        )
        ttk.Checkbutton(
            features_frame,
            text="Failure prediction",
            variable=self.failure_pred_var,
            command=self._on_ml_feature_changed
        ).pack(anchor='w', pady=2)

        self.threshold_opt_var = tk.BooleanVar(
            value=self.config.ml.threshold_optimization_enabled
        )
        ttk.Checkbutton(
            features_frame,
            text="Threshold optimization",
            variable=self.threshold_opt_var,
            command=self._on_ml_feature_changed
        ).pack(anchor='w', pady=2)

        # ML status
        ml_status = "Not available"
        ml_color = 'gray'

        if self.main_window.ml_predictor:
            ml_status = "Ready"
            ml_color = 'green'

        self.ml_status_label = ttk.Label(
            frame,
            text=f"Status: {ml_status}",
            font=('Segoe UI', 9),
            foreground=ml_color
        )
        self.ml_status_label.pack(anchor='w', pady=5)

    def _create_appearance_section(self, parent):
        """Create appearance settings section."""
        frame = ttk.LabelFrame(parent, text="Appearance", padding=15)
        frame.pack(fill='x', pady=10)

        # Theme
        theme_frame = ttk.Frame(frame)
        theme_frame.pack(fill='x', pady=5)

        ttk.Label(theme_frame, text="Theme:").pack(side='left')

        self.theme_var = tk.StringVar(value=self.config.gui.theme)
        theme_combo = ttk.Combobox(
            theme_frame,
            textvariable=self.theme_var,
            values=['clam', 'alt', 'default', 'classic'],
            state='readonly',
            width=15
        )
        theme_combo.pack(side='left', padx=10)
        theme_combo.bind('<<ComboboxSelected>>', self._on_theme_changed)

        # Features visibility
        visibility_frame = ttk.Frame(frame)
        visibility_frame.pack(fill='x', pady=10)

        ttk.Label(
            visibility_frame,
            text="Show tabs:",
            font=('Segoe UI', 10, 'bold')
        ).pack(anchor='w')

        self.show_historical_var = tk.BooleanVar(
            value=self.config.gui.show_historical_tab
        )
        ttk.Checkbutton(
            visibility_frame,
            text="Historical Data",
            variable=self.show_historical_var,
            command=self._on_visibility_changed
        ).pack(anchor='w', padx=20, pady=2)

        self.show_ml_var = tk.BooleanVar(
            value=self.config.gui.show_ml_insights
        )
        ttk.Checkbutton(
            visibility_frame,
            text="ML Insights",
            variable=self.show_ml_var,
            command=self._on_visibility_changed
        ).pack(anchor='w', padx=20, pady=2)

    def _load_current_settings(self):
        """Load current settings values."""
        # This is called in __init__, values are set during widget creation
        pass

    def _on_workers_changed(self):
        """Handle workers spinbox change."""
        self.config.processing.max_workers = self.workers_var.get()
        self._save_config()

    def _on_plots_changed(self):
        """Handle plots checkbox change."""
        self.config.processing.generate_plots = self.plots_var.get()
        self._save_config()

    def _on_cache_changed(self):
        """Handle cache checkbox change."""
        self.config.processing.cache_enabled = self.cache_var.get()
        self._save_config()

    def _on_database_changed(self):
        """Handle database checkbox change."""
        self.config.database.enabled = self.db_var.get()
        self._save_config()

        # Notify user to restart for changes to take effect
        if self.db_var.get() and not self.main_window.db_manager:
            messagebox.showinfo(
                "Restart Required",
                "Database connection will be established on next application start."
            )

    def _on_ml_changed(self):
        """Handle ML checkbox change."""
        self.config.ml.enabled = self.ml_var.get()

        # Enable/disable ML feature checkboxes
        state = 'normal' if self.ml_var.get() else 'disabled'
        for widget in self.winfo_children():
            if isinstance(widget, ttk.Checkbutton) and widget != self.ml_check:
                widget.configure(state=state)

        self._save_config()

    def _on_ml_feature_changed(self):
        """Handle ML feature checkbox change."""
        self.config.ml.failure_prediction_enabled = self.failure_pred_var.get()
        self.config.ml.threshold_optimization_enabled = self.threshold_opt_var.get()
        self._save_config()

    def _on_theme_changed(self, event=None):
        """Handle theme selection change."""
        new_theme = self.theme_var.get()
        self.config.gui.theme = new_theme

        # Apply theme
        try:
            self.main_window.style.theme_use(new_theme)
            self._save_config()
        except Exception as e:
            messagebox.showerror(
                "Theme Error",
                f"Failed to apply theme: {str(e)}"
            )
            # Revert
            self.theme_var.set(self.config.gui.theme)

    def _on_visibility_changed(self):
        """Handle tab visibility change."""
        self.config.gui.show_historical_tab = self.show_historical_var.get()
        self.config.gui.show_ml_insights = self.show_ml_var.get()
        self._save_config()

        messagebox.showinfo(
            "Restart Required",
            "Tab visibility changes will take effect on next application start."
        )

    def _save_config(self):
        """Save configuration to file."""
        try:
            config_path = Path.home() / ".laser_trim_analyzer" / "config.yaml"
            config_path.parent.mkdir(exist_ok=True)
            self.config.to_yaml(config_path)
        except Exception as e:
            print(f"Error saving config: {e}")

    def _open_full_settings(self):
        """Open the full settings dialog."""
        dialog = SettingsDialog(self.main_window.root, self.config)
        if dialog.show():
            # Settings were saved, update our display
            self._load_current_settings()

            # Update main window services if needed
            self.main_window._init_services()

    def show(self):
        """Show the settings page."""
        self.pack(fill='both', expand=True)

    def hide(self):
        """Hide the settings page."""
        self.pack_forget()