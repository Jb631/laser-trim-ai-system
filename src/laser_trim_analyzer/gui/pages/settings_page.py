"""
SettingsPage - In-app settings page

Provides a settings interface within the main application window.
"""

import customtkinter as ctk
from tkinter import messagebox
from pathlib import Path
from typing import Dict, Any, Optional

from laser_trim_analyzer.core.config import Config
from laser_trim_analyzer.gui.pages.base_page_ctk import BasePage
from laser_trim_analyzer.gui.settings_manager import settings_manager, SettingsDialog


class SettingsPage(BasePage):
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
        # Store config before calling super().__init__
        self.config = main_window.config if hasattr(main_window, 'config') else main_window
        
        super().__init__(parent, main_window)

    def _create_page(self):
        """Create settings page content (matching batch processing theme)."""
        # Main scrollable container (matching batch processing theme)
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Create sections in order (matching batch processing pattern)
        self._create_header()
        self._create_processing_section()
        self._create_database_section()
        self._create_ml_section()
        self._create_appearance_section()

        self._load_current_settings()

    def _create_header(self):
        """Create header section (matching batch processing theme)."""
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.pack(fill='x', pady=(0, 20))

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Settings",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(side='left', padx=15, pady=15)

        self.advanced_button = ctk.CTkButton(
            self.header_frame,
            text="Advanced Settings...",
            command=self._open_full_settings,
            width=150,
            height=40
        )
        self.advanced_button.pack(side='right', padx=15, pady=15)

    def _create_processing_section(self):
        """Create processing settings section (matching batch processing theme)."""
        self.processing_frame = ctk.CTkFrame(self.main_container)
        self.processing_frame.pack(fill='x', pady=(0, 20))

        self.processing_label = ctk.CTkLabel(
            self.processing_frame,
            text="Processing Settings:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.processing_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Processing options container
        self.processing_container = ctk.CTkFrame(self.processing_frame)
        self.processing_container.pack(fill='x', padx=15, pady=(0, 15))

        # Max workers
        workers_frame = ctk.CTkFrame(self.processing_container)
        workers_frame.pack(fill='x', padx=10, pady=(10, 5))

        workers_label = ctk.CTkLabel(workers_frame, text="Parallel Workers:")
        workers_label.pack(side='left', padx=10, pady=10)

        self.workers_var = ctk.StringVar(value=str(getattr(self.config.processing, 'max_workers', 4) if hasattr(self.config, 'processing') else 4))
        self.workers_entry = ctk.CTkEntry(
            workers_frame,
            textvariable=self.workers_var,
            width=60,
            height=30
        )
        self.workers_entry.pack(side='left', padx=10, pady=10)
        self.workers_entry.bind('<KeyRelease>', self._on_workers_changed)

        help_label = ctk.CTkLabel(
            workers_frame,
            text="(Higher = faster processing, more CPU usage)",
            font=ctk.CTkFont(size=10)
        )
        help_label.pack(side='left', padx=10, pady=10)

        # Checkboxes
        self.plots_var = ctk.BooleanVar(value=getattr(self.config.processing, 'generate_plots', True) if hasattr(self.config, 'processing') else True)
        self.plots_check = ctk.CTkCheckBox(
            self.processing_container,
            text="Generate analysis plots",
            variable=self.plots_var,
            command=self._on_plots_changed
        )
        self.plots_check.pack(anchor='w', padx=10, pady=5)

        self.cache_var = ctk.BooleanVar(value=getattr(self.config.processing, 'cache_enabled', True) if hasattr(self.config, 'processing') else True)
        self.cache_check = ctk.CTkCheckBox(
            self.processing_container,
            text="Enable result caching",
            variable=self.cache_var,
            command=self._on_cache_changed
        )
        self.cache_check.pack(anchor='w', padx=10, pady=5)

    def _create_database_section(self):
        """Create database settings section (matching batch processing theme)."""
        self.database_frame = ctk.CTkFrame(self.main_container)
        self.database_frame.pack(fill='x', pady=(0, 20))

        self.database_label = ctk.CTkLabel(
            self.database_frame,
            text="Database Settings:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.database_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Database options container
        self.database_container = ctk.CTkFrame(self.database_frame)
        self.database_container.pack(fill='x', padx=15, pady=(0, 15))

        # Enable database
        self.db_var = ctk.BooleanVar(value=getattr(self.config.database, 'enabled', True) if hasattr(self.config, 'database') else True)
        self.db_check = ctk.CTkCheckBox(
            self.database_container,
            text="Save results to database",
            variable=self.db_var,
            command=self._on_database_changed
        )
        self.db_check.pack(anchor='w', padx=10, pady=(10, 5))

        # Database path
        path_frame = ctk.CTkFrame(self.database_container)
        path_frame.pack(fill='x', padx=10, pady=5)

        path_label = ctk.CTkLabel(path_frame, text="Database:")
        path_label.pack(side='left', padx=10, pady=10)

        db_path = getattr(self.config.database, 'path', 'default.db') if hasattr(self.config, 'database') else 'default.db'
        self.db_path_label = ctk.CTkLabel(
            path_frame,
            text=str(db_path),
            font=ctk.CTkFont(size=10)
        )
        self.db_path_label.pack(side='left', padx=10, pady=10)

        # Database status
        status_text = "Not connected"
        if hasattr(self.main_window, 'db_manager') and self.main_window.db_manager:
            status_text = "Connected"

        self.db_status_label = ctk.CTkLabel(
            self.database_container,
            text=f"Status: {status_text}",
            font=ctk.CTkFont(size=10)
        )
        self.db_status_label.pack(anchor='w', padx=10, pady=(0, 10))

    def _create_ml_section(self):
        """Create ML settings section (matching batch processing theme)."""
        self.ml_frame = ctk.CTkFrame(self.main_container)
        self.ml_frame.pack(fill='x', pady=(0, 20))

        self.ml_label = ctk.CTkLabel(
            self.ml_frame,
            text="Machine Learning Settings:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.ml_label.pack(anchor='w', padx=15, pady=(15, 10))

        # ML options container
        self.ml_container = ctk.CTkFrame(self.ml_frame)
        self.ml_container.pack(fill='x', padx=15, pady=(0, 15))

        # Enable ML
        self.ml_var = ctk.BooleanVar(value=getattr(self.config.ml, 'enabled', True) if hasattr(self.config, 'ml') else True)
        self.ml_check = ctk.CTkCheckBox(
            self.ml_container,
            text="Enable ML predictions",
            variable=self.ml_var,
            command=self._on_ml_changed
        )
        self.ml_check.pack(anchor='w', padx=10, pady=(10, 5))

        # ML features frame
        features_frame = ctk.CTkFrame(self.ml_container)
        features_frame.pack(fill='x', padx=10, pady=5)

        self.failure_pred_var = ctk.BooleanVar(
            value=getattr(self.config.ml, 'failure_prediction_enabled', True) if hasattr(self.config, 'ml') else True
        )
        self.failure_pred_check = ctk.CTkCheckBox(
            features_frame,
            text="Failure prediction",
            variable=self.failure_pred_var,
            command=self._on_ml_feature_changed
        )
        self.failure_pred_check.pack(anchor='w', padx=10, pady=2)

        self.threshold_opt_var = ctk.BooleanVar(
            value=getattr(self.config.ml, 'threshold_optimization_enabled', True) if hasattr(self.config, 'ml') else True
        )
        self.threshold_opt_check = ctk.CTkCheckBox(
            features_frame,
            text="Threshold optimization",
            variable=self.threshold_opt_var,
            command=self._on_ml_feature_changed
        )
        self.threshold_opt_check.pack(anchor='w', padx=10, pady=2)

    def _create_appearance_section(self):
        """Create appearance settings section (matching batch processing theme)."""
        self.appearance_frame = ctk.CTkFrame(self.main_container)
        self.appearance_frame.pack(fill='x', pady=(0, 20))

        self.appearance_label = ctk.CTkLabel(
            self.appearance_frame,
            text="Appearance Settings:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.appearance_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Appearance options container
        self.appearance_container = ctk.CTkFrame(self.appearance_frame)
        self.appearance_container.pack(fill='x', padx=15, pady=(0, 15))

        # Theme selection
        theme_frame = ctk.CTkFrame(self.appearance_container)
        theme_frame.pack(fill='x', padx=10, pady=(10, 5))

        theme_label = ctk.CTkLabel(theme_frame, text="Theme:")
        theme_label.pack(side='left', padx=10, pady=10)

        # Get current theme from settings manager
        current_theme = settings_manager.get('theme.mode', 'dark')
        self.theme_var = ctk.StringVar(value=current_theme)
        self.theme_combo = ctk.CTkComboBox(
            theme_frame,
            variable=self.theme_var,
            values=["dark", "light", "system"],
            width=120,
            height=30,
            command=self._on_theme_changed
        )
        self.theme_combo.pack(side='left', padx=10, pady=10)

        # UI visibility options
        self.advanced_mode_var = ctk.BooleanVar(value=settings_manager.get('advanced.experimental_features', False))
        self.advanced_mode_check = ctk.CTkCheckBox(
            self.appearance_container,
            text="Show advanced options",
            variable=self.advanced_mode_var,
            command=self._on_visibility_changed
        )
        self.advanced_mode_check.pack(anchor='w', padx=10, pady=5)

    def _load_current_settings(self):
        """Load current settings values."""
        # This is called in __init__, values are set during widget creation
        pass

    def _on_workers_changed(self, event=None):
        """Handle workers entry change."""
        try:
            workers = int(self.workers_var.get())
            if hasattr(self.config, 'processing'):
                self.config.processing.max_workers = workers
            # Also save in settings manager for persistence
            settings_manager.set('performance.thread_pool_size', workers)
            self._save_config()
        except ValueError:
            # If invalid number, reset to default
            default_workers = getattr(self.config.processing, 'max_workers', 4) if hasattr(self.config, 'processing') else 4
            self.workers_var.set(str(default_workers))

    def _on_plots_changed(self):
        """Handle plots checkbox change."""
        value = self.plots_var.get()
        if hasattr(self.config, 'processing'):
            self.config.processing.generate_plots = value
        # Save in settings manager
        settings_manager.set('display.include_charts', value)
        self._save_config()

    def _on_cache_changed(self):
        """Handle cache checkbox change."""
        value = self.cache_var.get()
        if hasattr(self.config, 'processing'):
            self.config.processing.cache_enabled = value
        # Save in settings manager
        settings_manager.set('performance.enable_caching', value)
        self._save_config()

    def _on_database_changed(self):
        """Handle database checkbox change."""
        value = self.db_var.get()
        if hasattr(self.config, 'database'):
            self.config.database.enabled = value
        # Save in settings manager
        settings_manager.set('data.auto_backup', value)
        self._save_config()
        
        # Update status
        status_text = "Connected" if value and hasattr(self.main_window, 'db_manager') and self.main_window.db_manager else "Not connected"
        self.db_status_label.configure(text=f"Status: {status_text}")

    def _on_ml_changed(self):
        """Handle ML checkbox change."""
        value = self.ml_var.get()
        if hasattr(self.config, 'ml'):
            self.config.ml.enabled = value
        # Save in settings manager
        settings_manager.set('analysis.enable_ml_predictions', value)
        self._save_config()
        
        # Enable/disable ML feature checkboxes
        state = "normal" if value else "disabled"
        if hasattr(self, 'failure_pred_check'):
            self.failure_pred_check.configure(state=state)
        if hasattr(self, 'threshold_opt_check'):
            self.threshold_opt_check.configure(state=state)

    def _on_ml_feature_changed(self):
        """Handle ML feature checkbox change."""
        if hasattr(self.config, 'ml'):
            self.config.ml.failure_prediction_enabled = self.failure_pred_var.get()
            self.config.ml.threshold_optimization_enabled = self.threshold_opt_var.get()
        # Save in settings manager
        settings_manager.set('notifications.notification_types.ml_insights', self.failure_pred_var.get())
        self._save_config()

    def _on_theme_changed(self, value=None):
        """Handle theme combobox change."""
        theme = self.theme_var.get()
        ctk.set_appearance_mode(theme)
        # Save in settings manager
        settings_manager.set('theme.mode', theme)
        settings_manager.apply_theme()
        # Update config if it has theme setting
        if hasattr(self.config, 'gui') and hasattr(self.config.gui, 'theme'):
            self.config.gui.theme = theme
        self._save_config()

    def _on_visibility_changed(self):
        """Handle visibility change."""
        value = self.advanced_mode_var.get()
        if hasattr(self.config, 'gui'):
            if hasattr(self.config.gui, 'show_historical_tab'):
                self.config.gui.show_historical_tab = value
            if hasattr(self.config.gui, 'show_ml_insights'):
                self.config.gui.show_ml_insights = value
        # Save in settings manager
        settings_manager.set('advanced.experimental_features', value)
        self._save_config()

    def _save_config(self):
        """Save configuration changes."""
        try:
            # Save settings manager configuration
            settings_manager.save_settings()
            
            # If the main config has a save method, use it
            if hasattr(self.config, 'save'):
                self.config.save()
            elif hasattr(self.config, 'write') or hasattr(self.config, 'dump'):
                # Try alternative save methods
                config_file = Path.home() / ".laser_trim_analyzer" / "config.yaml"
                config_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Create a dictionary from the config for saving
                config_dict = {
                    'database': {
                        'enabled': getattr(self.config.database, 'enabled', True) if hasattr(self.config, 'database') else True,
                        'path': str(getattr(self.config.database, 'path', '')) if hasattr(self.config, 'database') else ''
                    },
                    'processing': {
                        'max_workers': getattr(self.config.processing, 'max_workers', 4) if hasattr(self.config, 'processing') else 4,
                        'generate_plots': getattr(self.config.processing, 'generate_plots', True) if hasattr(self.config, 'processing') else True,
                        'cache_enabled': getattr(self.config.processing, 'cache_enabled', True) if hasattr(self.config, 'processing') else True
                    },
                    'ml': {
                        'enabled': getattr(self.config.ml, 'enabled', True) if hasattr(self.config, 'ml') else True,
                        'failure_prediction_enabled': getattr(self.config.ml, 'failure_prediction_enabled', True) if hasattr(self.config, 'ml') else True,
                        'threshold_optimization_enabled': getattr(self.config.ml, 'threshold_optimization_enabled', True) if hasattr(self.config, 'ml') else True
                    }
                }
                
                # Save to YAML file
                import yaml
                with open(config_file, 'w') as f:
                    yaml.dump(config_dict, f)
                    
            self.logger.info("Settings saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save settings: {e}")
            messagebox.showwarning("Warning", "Some settings may not persist after restart")

    def _open_full_settings(self):
        """Open the full settings dialog."""
        try:
            dialog = SettingsDialog(self, settings_manager)
            # Center the dialog
            dialog.update_idletasks()
            width = dialog.winfo_width()
            height = dialog.winfo_height()
            x = (dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (dialog.winfo_screenheight() // 2) - (height // 2)
            dialog.geometry(f"{width}x{height}+{x}+{y}")
            dialog.focus()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open settings dialog: {e}")

    def show(self):
        """Show the settings page."""
        self.pack(fill='both', expand=True)

    def hide(self):
        """Hide the settings page."""
        self.pack_forget()