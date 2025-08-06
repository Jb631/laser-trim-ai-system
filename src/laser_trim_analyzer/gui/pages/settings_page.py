"""
SettingsPage - In-app settings page

Provides a settings interface within the main application window.
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import threading
import os
import yaml

from laser_trim_analyzer.core.config import Config
# from laser_trim_analyzer.gui.pages.base_page_ctk import BasePage  # Using CTkFrame instead
from laser_trim_analyzer.gui.settings_manager import settings_manager, SettingsDialog


class SettingsPage(ctk.CTkFrame):
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
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Add BasePage-like functionality
        self.is_visible = False
        self.needs_refresh = True
        self._stop_requested = False
        
        # Store config reference
        self.config = main_window.config if hasattr(main_window, 'config') else main_window
        
        # Thread safety
        self._settings_lock = threading.Lock()
        
        # Create the page
        self._create_page()

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
        self.processing_container = ctk.CTkFrame(self.processing_frame, fg_color="transparent")
        self.processing_container.pack(fill='x', padx=15, pady=(0, 15))

        # Max workers
        workers_frame = ctk.CTkFrame(self.processing_container, fg_color="transparent")
        workers_frame.pack(fill='x', padx=10, pady=(10, 5))

        workers_label = ctk.CTkLabel(workers_frame, text="Parallel Workers:")
        workers_label.pack(side='left', padx=10, pady=10)

        # Load from settings_manager first, then fall back to config
        workers_value = settings_manager.get('performance.thread_pool_size', 
                                            getattr(self.config.processing, 'max_workers', 4) if hasattr(self.config, 'processing') else 4)
        self.workers_var = ctk.StringVar(value=str(workers_value))
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
        # Load from settings_manager first, then fall back to config
        plots_value = settings_manager.get('display.include_charts',
                                          getattr(self.config.processing, 'generate_plots', True) if hasattr(self.config, 'processing') else True)
        self.plots_var = ctk.BooleanVar(value=plots_value)
        self.plots_check = ctk.CTkCheckBox(
            self.processing_container,
            text="Generate analysis plots",
            variable=self.plots_var,
            command=self._on_plots_changed
        )
        self.plots_check.pack(anchor='w', padx=10, pady=5)

        # Load from settings_manager first, then fall back to config
        cache_value = settings_manager.get('performance.enable_caching',
                                          getattr(self.config.processing, 'cache_enabled', True) if hasattr(self.config, 'processing') else True)
        self.cache_var = ctk.BooleanVar(value=cache_value)
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
        self.database_container = ctk.CTkFrame(self.database_frame, fg_color="transparent")
        self.database_container.pack(fill='x', padx=15, pady=(0, 15))

        # Enable database
        # Load from settings_manager first, then fall back to config
        db_value = settings_manager.get('data.auto_backup',
                                       getattr(self.config.database, 'enabled', True) if hasattr(self.config, 'database') else True)
        self.db_var = ctk.BooleanVar(value=db_value)
        self.db_check = ctk.CTkCheckBox(
            self.database_container,
            text="Save results to database",
            variable=self.db_var,
            command=self._on_database_changed
        )
        self.db_check.pack(anchor='w', padx=10, pady=(10, 5))

        # Deployment mode section
        mode_frame = ctk.CTkFrame(self.database_container, fg_color="transparent")
        mode_frame.pack(fill='x', padx=10, pady=(10, 5))
        
        mode_label = ctk.CTkLabel(mode_frame, text="Database Mode:")
        mode_label.pack(side='left', padx=10, pady=10)
        
        # Get current deployment mode from config
        current_mode = self._get_deployment_mode()
        self.db_mode_var = ctk.StringVar(value=current_mode)
        self.db_mode_combo = ctk.CTkComboBox(
            mode_frame,
            variable=self.db_mode_var,
            values=["Single User (Local)", "Multi-User (Network)"],
            width=200,
            height=30,
            command=self._on_db_mode_changed
        )
        self.db_mode_combo.pack(side='left', padx=10, pady=10)

        # Database path
        path_frame = ctk.CTkFrame(self.database_container, fg_color="transparent")
        path_frame.pack(fill='x', padx=10, pady=5)

        path_label = ctk.CTkLabel(path_frame, text="Database:")
        path_label.pack(side='left', padx=10, pady=10)

        # Database path with change button
        self.db_path_frame = ctk.CTkFrame(path_frame, fg_color="transparent")
        self.db_path_frame.pack(side='left', fill='x', expand=True)
        
        db_path = self._get_current_db_path()
        self.db_path_label = ctk.CTkLabel(
            self.db_path_frame,
            text=str(db_path),
            font=ctk.CTkFont(size=10)
        )
        self.db_path_label.pack(side='left', padx=10, pady=10)
        
        # Change/Browse path button
        button_text = "Browse" if current_mode == "Single User (Local)" else "Change"
        self.change_path_button = ctk.CTkButton(
            self.db_path_frame,
            text=button_text,
            width=80,
            height=25,
            command=self._change_db_path if current_mode == "Multi-User (Network)" else self._browse_db_path
        )
        # Always show the button for both modes
        self.change_path_button.pack(side='left', padx=5, pady=10)

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
        
        # Mode change warning
        self.db_warning_label = ctk.CTkLabel(
            self.database_container,
            text="⚠️ Changing database mode requires application restart",
            font=ctk.CTkFont(size=10),
            text_color="orange"
        )
        # Initially hidden
        self.db_warning_label.pack_forget()

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
        self.ml_container = ctk.CTkFrame(self.ml_frame, fg_color="transparent")
        self.ml_container.pack(fill='x', padx=15, pady=(0, 15))

        # Enable ML
        # Load from settings_manager first, then fall back to config
        ml_value = settings_manager.get('analysis.enable_ml_predictions',
                                       getattr(self.config.ml, 'enabled', True) if hasattr(self.config, 'ml') else True)
        self.ml_var = ctk.BooleanVar(value=ml_value)
        self.ml_check = ctk.CTkCheckBox(
            self.ml_container,
            text="Enable ML predictions",
            variable=self.ml_var,
            command=self._on_ml_changed
        )
        self.ml_check.pack(anchor='w', padx=10, pady=(10, 5))

        # ML features frame
        features_frame = ctk.CTkFrame(self.ml_container, fg_color="transparent")
        features_frame.pack(fill='x', padx=10, pady=5)

        # Load from settings_manager first, then fall back to config
        failure_pred_value = settings_manager.get('notifications.notification_types.ml_insights',
                                                 getattr(self.config.ml, 'failure_prediction_enabled', True) if hasattr(self.config, 'ml') else True)
        self.failure_pred_var = ctk.BooleanVar(value=failure_pred_value)
        self.failure_pred_check = ctk.CTkCheckBox(
            features_frame,
            text="Failure prediction",
            variable=self.failure_pred_var,
            command=self._on_ml_feature_changed
        )
        self.failure_pred_check.pack(anchor='w', padx=10, pady=2)

        # Load from settings_manager first, then fall back to config
        threshold_opt_value = settings_manager.get('analysis.threshold_optimization',
                                                  getattr(self.config.ml, 'threshold_optimization_enabled', True) if hasattr(self.config, 'ml') else True)
        self.threshold_opt_var = ctk.BooleanVar(value=threshold_opt_value)
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
        self.appearance_container = ctk.CTkFrame(self.appearance_frame, fg_color="transparent")
        self.appearance_container.pack(fill='x', padx=15, pady=(0, 15))

        # Theme selection
        theme_frame = ctk.CTkFrame(self.appearance_container, fg_color="transparent")
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
        # Already using settings_manager correctly
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
        settings_manager.set('analysis.threshold_optimization', self.threshold_opt_var.get())
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
        self.on_show()

    def hide(self):
        """Hide the settings page."""
        self.on_hide()
        self.pack_forget()
    
    def on_show(self):
        """Called when page is shown."""
        self.is_visible = True
        if self.needs_refresh:
            self._load_current_settings()
            self.needs_refresh = False
    
    def on_hide(self):
        """Called when page is hidden."""
        self.is_visible = False
    
    def _get_deployment_mode(self):
        """Get current deployment mode from config."""
        try:
            # First check deployment.yaml
            deployment_config_path = Path("config/deployment.yaml")
            if deployment_config_path.exists():
                with open(deployment_config_path, 'r') as f:
                    deployment_config = yaml.safe_load(f)
                    mode = deployment_config.get('deployment_mode', 'single_user')
                    return "Single User (Local)" if mode == 'single_user' else "Multi-User (Network)"
            
            # Fallback to checking main config
            if hasattr(self.config, 'deployment_mode'):
                mode = self.config.deployment_mode
                return "Single User (Local)" if mode == 'single_user' else "Multi-User (Network)"
            
            # Default to single user
            return "Single User (Local)"
        except Exception as e:
            self.logger.error(f"Error getting deployment mode: {e}")
            return "Single User (Local)"
    
    def _get_current_db_path(self):
        """Get current database path based on deployment mode."""
        try:
            mode = self._get_deployment_mode()
            
            # Check deployment.yaml
            deployment_config_path = Path("config/deployment.yaml")
            if deployment_config_path.exists():
                with open(deployment_config_path, 'r') as f:
                    deployment_config = yaml.safe_load(f)
                    
                    if mode == "Single User (Local)":
                        db_config = deployment_config.get('database', {}).get('single_user', {})
                        path = db_config.get('path', './data/laser_trim.db')
                    else:
                        db_config = deployment_config.get('database', {}).get('multi_user', {})
                        path = db_config.get('path', '//server/share/laser_trim/database.db')
                    
                    # Handle relative paths for portable deployment
                    if path.startswith('./'):
                        # Show as absolute path for clarity
                        app_dir = Path.cwd()
                        path = str(app_dir / path[2:])
                    else:
                        # Expand environment variables
                        path = os.path.expandvars(path)
                    return path
            
            # Fallback to main config
            if hasattr(self.config, 'database') and hasattr(self.config.database, 'path'):
                return str(self.config.database.path)
            
            return "default.db"
        except Exception as e:
            self.logger.error(f"Error getting database path: {e}")
            return "default.db"
    
    def _on_db_mode_changed(self, value=None):
        """Handle database mode change."""
        # Show warning
        self.db_warning_label.pack(anchor='w', padx=10, pady=(5, 10))
        
        # Update path display
        new_mode = self.db_mode_var.get()
        
        # Update button text and command based on mode
        if new_mode == "Multi-User (Network)":
            self.change_path_button.configure(text="Change", command=self._change_db_path)
        else:
            self.change_path_button.configure(text="Browse", command=self._browse_db_path)
        
        # Update the deployment configuration
        self._update_deployment_mode(new_mode)
        
        # Update path label
        self.db_path_label.configure(text=self._get_current_db_path())
    
    def _update_deployment_mode(self, mode):
        """Update deployment mode in config file."""
        try:
            deployment_config_path = Path("config/deployment.yaml")
            
            # Load existing config
            if deployment_config_path.exists():
                with open(deployment_config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                # Create default config structure
                config = {
                    'deployment_mode': 'single_user',
                    'database': {
                        'single_user': {
                            'path': './data/laser_trim.db'  # Portable by default
                        },
                        'multi_user': {
                            'path': '//server/share/laser_trim/database.db'
                        }
                    }
                }
            
            # Update mode
            config['deployment_mode'] = 'single_user' if mode == "Single User (Local)" else 'multi_user'
            
            # Save config
            deployment_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(deployment_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Updated deployment mode to: {mode}")
            
            # Show restart required message
            messagebox.showinfo(
                "Restart Required",
                "The database mode has been changed. Please restart the application for the changes to take effect."
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update deployment mode: {e}")
            messagebox.showerror("Error", f"Failed to update deployment mode: {e}")
    
    def _change_db_path(self):
        """Open dialog to change network database path."""
        try:
            # Create a custom dialog for network path input
            dialog = ctk.CTkToplevel(self)
            dialog.title("Change Network Database Path")
            dialog.geometry("600x250")
            
            # Center the dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (300)
            y = (dialog.winfo_screenheight() // 2) - (125)
            dialog.geometry(f"600x250+{x}+{y}")
            
            # Make dialog modal
            dialog.transient(self)
            dialog.grab_set()
            
            # Create content
            label = ctk.CTkLabel(
                dialog,
                text="Enter the network path to the shared database:",
                font=ctk.CTkFont(size=14)
            )
            label.pack(pady=20)
            
            # Example label
            example_label = ctk.CTkLabel(
                dialog,
                text="Example: \\\\server\\share\\LaserTrimAnalyzer\\database.db",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            example_label.pack(pady=(0, 10))
            
            # Path entry
            current_path = self._get_current_db_path()
            path_var = ctk.StringVar(value=current_path)
            path_entry = ctk.CTkEntry(
                dialog,
                textvariable=path_var,
                width=500,
                height=35
            )
            path_entry.pack(pady=10)
            
            # Buttons frame
            button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            button_frame.pack(pady=20)
            
            def save_path():
                new_path = path_var.get().strip()
                if new_path:
                    self._update_network_db_path(new_path)
                    self.db_path_label.configure(text=new_path)
                    dialog.destroy()
                else:
                    messagebox.showwarning("Invalid Path", "Please enter a valid network path.")
            
            # Save button
            save_button = ctk.CTkButton(
                button_frame,
                text="Save",
                command=save_path,
                width=100
            )
            save_button.pack(side='left', padx=10)
            
            # Cancel button
            cancel_button = ctk.CTkButton(
                button_frame,
                text="Cancel",
                command=dialog.destroy,
                width=100
            )
            cancel_button.pack(side='left', padx=10)
            
            # Focus on entry
            path_entry.focus()
            
        except Exception as e:
            self.logger.error(f"Error in change database path dialog: {e}")
            messagebox.showerror("Error", f"Failed to open path dialog: {e}")
    
    def _update_network_db_path(self, new_path):
        """Update the network database path in deployment config."""
        try:
            deployment_config_path = Path("config/deployment.yaml")
            
            # Load existing config
            if deployment_config_path.exists():
                with open(deployment_config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                config = {}
            
            # Update network path
            if 'database' not in config:
                config['database'] = {}
            if 'multi_user' not in config['database']:
                config['database']['multi_user'] = {}
            
            # Convert backslashes to forward slashes for YAML
            config['database']['multi_user']['path'] = new_path.replace('\\', '/')
            
            # Save config
            with open(deployment_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Updated network database path to: {new_path}")
            
            # Show restart message
            messagebox.showinfo(
                "Restart Required",
                "The database path has been changed. Please restart the application for the changes to take effect."
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update network database path: {e}")
            messagebox.showerror("Error", f"Failed to update database path: {e}")
    
    def _browse_db_path(self):
        """Open file dialog to browse for local database path."""
        try:
            # Get current path
            current_path = self._get_current_db_path()
            current_dir = str(Path(current_path).parent) if current_path else str(Path.cwd())
            
            # Create custom dialog
            dialog = ctk.CTkToplevel(self)
            dialog.title("Select Database Location")
            dialog.geometry("700x400")
            
            # Center the dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - 350
            y = (dialog.winfo_screenheight() // 2) - 200
            dialog.geometry(f"700x400+{x}+{y}")
            
            # Make dialog modal
            dialog.transient(self)
            dialog.grab_set()
            
            # Title
            title_label = ctk.CTkLabel(
                dialog,
                text="Choose Database Location",
                font=ctk.CTkFont(size=16, weight="bold")
            )
            title_label.pack(pady=20)
            
            # Options frame
            options_frame = ctk.CTkFrame(dialog)
            options_frame.pack(fill='both', expand=True, padx=20, pady=10)
            
            # Read the actual path from deployment.yaml to determine type
            actual_path = current_path
            deployment_config_path = Path("config/deployment.yaml")
            if deployment_config_path.exists():
                with open(deployment_config_path, 'r') as f:
                    deployment_config = yaml.safe_load(f)
                    actual_path = deployment_config.get('database', {}).get('single_user', {}).get('path', './data/laser_trim.db')
            
            # Determine initial selection based on actual path
            initial_selection = "custom"  # Default to custom
            if actual_path.startswith("./"):
                initial_selection = "portable"
            elif "%USERPROFILE%/Documents" in actual_path:
                initial_selection = "documents"
            
            # Radio button variable
            location_var = ctk.StringVar(value=initial_selection)
            
            # Option 1: Portable (with app)
            portable_radio = ctk.CTkRadioButton(
                options_frame,
                text="Portable (travels with application)",
                variable=location_var,
                value="portable",
                font=ctk.CTkFont(size=14)
            )
            portable_radio.pack(anchor='w', padx=20, pady=10)
            
            portable_desc = ctk.CTkLabel(
                options_frame,
                text="📁 Database stored in: [App Folder]/data/laser_trim.db",
                font=ctk.CTkFont(size=11),
                text_color="gray"
            )
            portable_desc.pack(anchor='w', padx=40, pady=(0, 10))
            
            # Option 2: Documents
            documents_radio = ctk.CTkRadioButton(
                options_frame,
                text="Documents Folder (persistent)",
                variable=location_var,
                value="documents",
                font=ctk.CTkFont(size=14)
            )
            documents_radio.pack(anchor='w', padx=20, pady=10)
            
            docs_path = str(Path.home() / "Documents" / "LaserTrimAnalyzer" / "laser_trim.db")
            documents_desc = ctk.CTkLabel(
                options_frame,
                text=f"📁 Database stored in: {docs_path}",
                font=ctk.CTkFont(size=11),
                text_color="gray"
            )
            documents_desc.pack(anchor='w', padx=40, pady=(0, 10))
            
            # Option 3: Custom
            custom_radio = ctk.CTkRadioButton(
                options_frame,
                text="Custom Location",
                variable=location_var,
                value="custom",
                font=ctk.CTkFont(size=14)
            )
            custom_radio.pack(anchor='w', padx=20, pady=10)
            
            # Custom path entry
            custom_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
            custom_frame.pack(fill='x', padx=40, pady=(0, 10))
            
            custom_path_var = ctk.StringVar(value=current_path)
            # Enable custom entry if custom is initially selected
            entry_state = "normal" if initial_selection == "custom" else "disabled"
            custom_entry = ctk.CTkEntry(
                custom_frame,
                textvariable=custom_path_var,
                width=400,
                height=30,
                state=entry_state
            )
            custom_entry.pack(side='left', padx=(0, 10))
            
            def browse_custom():
                file_path = filedialog.asksaveasfilename(
                    title="Select Database File",
                    defaultextension=".db",
                    filetypes=[("Database Files", "*.db"), ("All Files", "*.*")],
                    initialdir=current_dir,
                    initialfile="laser_trim.db"
                )
                if file_path:
                    custom_path_var.set(file_path)
                    location_var.set("custom")
            
            browse_btn = ctk.CTkButton(
                custom_frame,
                text="Browse...",
                width=80,
                height=30,
                command=browse_custom
            )
            browse_btn.pack(side='left')
            
            # Enable/disable custom entry based on selection
            def on_radio_change():
                if location_var.get() == "custom":
                    custom_entry.configure(state="normal")
                else:
                    custom_entry.configure(state="disabled")
            
            portable_radio.configure(command=on_radio_change)
            documents_radio.configure(command=on_radio_change)
            custom_radio.configure(command=on_radio_change)
            
            # Buttons frame
            button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            button_frame.pack(pady=20)
            
            def save_selection():
                choice = location_var.get()
                
                if choice == "portable":
                    new_path = "./data/laser_trim.db"
                elif choice == "documents":
                    new_path = "%USERPROFILE%/Documents/LaserTrimAnalyzer/laser_trim.db"
                else:  # custom
                    new_path = custom_path_var.get().strip()
                    if not new_path:
                        messagebox.showwarning("Invalid Path", "Please enter a valid database path.")
                        return
                
                # Update the configuration
                self._update_local_db_path(new_path)
                
                # Update display (show full path)
                display_path = new_path
                if new_path.startswith('./'):
                    display_path = str(Path.cwd() / new_path[2:])
                elif '%' in new_path:
                    display_path = os.path.expandvars(new_path)
                
                self.db_path_label.configure(text=display_path)
                dialog.destroy()
            
            save_button = ctk.CTkButton(
                button_frame,
                text="Save",
                command=save_selection,
                width=100
            )
            save_button.pack(side='left', padx=10)
            
            cancel_button = ctk.CTkButton(
                button_frame,
                text="Cancel",
                command=dialog.destroy,
                width=100
            )
            cancel_button.pack(side='left', padx=10)
            
        except Exception as e:
            self.logger.error(f"Error in browse database path dialog: {e}")
            messagebox.showerror("Error", f"Failed to open browse dialog: {e}")
    
    def _update_local_db_path(self, new_path):
        """Update the local database path in deployment config."""
        try:
            deployment_config_path = Path("config/deployment.yaml")
            
            # Load existing config
            if deployment_config_path.exists():
                with open(deployment_config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                config = {}
            
            # Update local path
            if 'database' not in config:
                config['database'] = {}
            if 'single_user' not in config['database']:
                config['database']['single_user'] = {}
            
            config['database']['single_user']['path'] = new_path
            
            # Save config
            with open(deployment_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Updated local database path to: {new_path}")
            
            # Show restart message
            messagebox.showinfo(
                "Restart Required",
                "The database location has been changed. Please restart the application for the changes to take effect."
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update local database path: {e}")
            messagebox.showerror("Error", f"Failed to update database path: {e}")
    
    def cleanup(self):
        """Clean up resources when page is destroyed."""
        try:
            # Stop any running operations
            self._stop_requested = True
            
            # Save any pending changes
            with self._settings_lock:
                self._save_config()
            
            self.logger.debug("Settings page cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")