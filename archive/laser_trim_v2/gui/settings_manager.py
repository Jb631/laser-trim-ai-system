"""
Settings Manager for Laser Trim Analyzer

Handles user preferences, themes, and application configuration
with persistent storage and real-time updates.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import customtkinter as ctk
from tkinter import messagebox
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SettingsManager:
    """Manages application settings and user preferences."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or str(Path.home() / ".laser_trim_analyzer" / "settings.json")
        self.settings = {}
        self.callbacks = {}  # Setting change callbacks
        
        # Production default settings
        self.default_settings = {
            # UI/UX Preferences - minimal defaults
            "theme": {
                "mode": "system",  # Use system default
                "color_scheme": "blue",  
                "font_family": None,  # Use system default
                "font_size": None,  # Use system default
                "enable_animations": False,  # Disabled by default for performance
                "compact_mode": False,
                "high_contrast": False
            },
            
            # Analysis Preferences - production defaults
            "analysis": {
                "auto_save_results": False,  # Require explicit save in production
                "default_export_format": "xlsx",  # Standard Excel format
                "sigma_threshold": None,  # Must be configured per model
                "linearity_threshold": None,  # Must be configured per model
                "enable_ml_predictions": False,  # Disabled until trained
                "confidence_threshold": 0.95,  # High confidence for production
                "batch_size_limit": 100  # Conservative initial limit
            },
            
            # Display Preferences
            "display": {
                "show_tooltips": True,
                "show_progress_details": True,
                "chart_style": "modern",  # modern, classic, minimal
                "grid_lines": True,
                "color_blind_friendly": False,
                "decimal_places": 4,
                "scientific_notation": False
            },
            
            # Performance Settings
            "performance": {
                "max_memory_usage": 2048,  # MB
                "thread_pool_size": 4,
                "enable_caching": True,
                "cache_size_limit": 500,  # MB
                "auto_cleanup": True,
                "background_processing": True
            },
            
            # Export Settings
            "export": {
                "default_directory": str(Path.home() / "Documents"),
                "include_metadata": True,
                "include_charts": True,
                "chart_resolution": "high",  # low, medium, high
                "watermark": False,
                "auto_timestamp": True,
                "compression": "normal"  # none, low, normal, high
            },
            
            # Data Management
            "data": {
                "auto_backup": True,
                "backup_frequency": "daily",  # hourly, daily, weekly
                "retention_period": 90,  # days
                "compress_backups": True,
                "sync_cloud": False,
                "cloud_provider": "none"  # none, google_drive, onedrive, dropbox
            },
            
            # Notifications
            "notifications": {
                "enable_notifications": True,
                "sound_alerts": False,
                "desktop_notifications": True,
                "email_reports": False,
                "email_address": "",
                "notification_types": {
                    "analysis_complete": True,
                    "errors": True,
                    "warnings": True,
                    "ml_insights": True
                }
            },
            
            # Advanced Features
            "advanced": {
                "debug_mode": False,
                "log_level": "INFO",
                "experimental_features": False,
                "telemetry": True,
                "auto_updates": True,
                "beta_features": False
            },
            
            # Recent Files and Paths
            "recent": {
                "files": [],
                "directories": [],
                "export_paths": [],
                "max_recent_items": 10
            },
            
            # Window State
            "window": {
                "width": 1200,
                "height": 800,
                "x": 100,
                "y": 100,
                "maximized": False,
                "last_page": "single_file"
            }
        }
        
        self.load_settings()
    
    def load_settings(self):
        """Load settings from file."""
        try:
            config_path = Path(self.config_file)
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                
                # Handle both old format (direct settings) and new format (with metadata)
                if isinstance(loaded_data, dict) and 'settings' in loaded_data:
                    # New format with metadata
                    saved_settings = loaded_data['settings']
                else:
                    # Old format or direct settings
                    saved_settings = loaded_data
                
                # Merge with defaults (in case new settings were added)
                self.settings = self._merge_settings(self.default_settings.copy(), saved_settings)
                
                logger.info(f"Settings loaded from {config_path}")
            else:
                # First time - create with defaults
                self.settings = self.default_settings.copy()
                self.save_settings()
                logger.info("Created default settings")
                
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            self.settings = self.default_settings.copy()
    
    def save_settings(self):
        """Save settings to file."""
        try:
            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata
            settings_with_meta = {
                "metadata": {
                    "version": "1.0",
                    "last_saved": datetime.now().isoformat(),
                    "app_version": "2.0.0"
                },
                "settings": self.settings
            }
            
            # Write to a temporary file first
            temp_path = config_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(settings_with_meta, f, indent=2, ensure_ascii=False)
                f.flush()  # Ensure all data is written
                os.fsync(f.fileno())  # Force write to disk
            
            # Replace the original file atomically
            temp_path.replace(config_path)
            
            logger.info(f"Settings saved to {config_path}")
            
        except (IOError, OSError) as e:
            logger.error(f"File system error saving settings: {e}")
            raise
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            raise
    
    def get(self, key_path: str, default=None):
        """Get setting value using dot notation (e.g., 'theme.mode')."""
        try:
            keys = key_path.split('.')
            value = self.settings
            
            for key in keys:
                value = value[key]
            
            return value
            
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any, save: bool = True):
        """Set setting value using dot notation."""
        try:
            keys = key_path.split('.')
            current = self.settings
            
            # Navigate to parent
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set value
            old_value = current.get(keys[-1])
            current[keys[-1]] = value
            
            # Trigger callbacks
            if key_path in self.callbacks:
                for callback in self.callbacks[key_path]:
                    try:
                        callback(key_path, old_value, value)
                    except Exception as e:
                        logger.error(f"Error in settings callback for {key_path}: {e}")
            
            # Save if requested
            if save:
                self.save_settings()
                
            logger.debug(f"Setting {key_path} = {value}")
            
        except Exception as e:
            logger.error(f"Error setting {key_path}: {e}")
    
    def save(self):
        """Alias for save_settings() for compatibility."""
        self.save_settings()
    
    def register_callback(self, key_path: str, callback: Callable[[str, Any, Any], None]):
        """Register callback for setting changes."""
        if key_path not in self.callbacks:
            self.callbacks[key_path] = []
        self.callbacks[key_path].append(callback)
    
    def unregister_callback(self, key_path: str, callback: Callable):
        """Unregister callback for setting changes."""
        if key_path in self.callbacks:
            try:
                self.callbacks[key_path].remove(callback)
            except ValueError:
                pass
    
    def reset_to_defaults(self, section: Optional[str] = None):
        """Reset settings to defaults."""
        if section:
            if section in self.default_settings:
                self.settings[section] = self.default_settings[section].copy()
        else:
            self.settings = self.default_settings.copy()
        
        self.save_settings()
        logger.info(f"Settings reset to defaults{f' for section {section}' if section else ''}")
    
    def export_settings(self, file_path: str):
        """Export settings to file."""
        try:
            export_data = {
                "metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "app_version": "2.0.0",
                    "export_type": "user_settings"
                },
                "settings": self.settings
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Settings exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting settings: {e}")
            return False
    
    def import_settings(self, file_path: str, merge: bool = True):
        """Import settings from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_data = json.load(f)
            
            if "settings" in imported_data:
                imported_settings = imported_data["settings"]
            else:
                imported_settings = imported_data
            
            if merge:
                self.settings = self._merge_settings(self.settings, imported_settings)
            else:
                self.settings = imported_settings
            
            self.save_settings()
            logger.info(f"Settings imported from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing settings: {e}")
            return False
    
    def add_recent_file(self, file_path: str):
        """Add file to recent files list."""
        recent_files = self.get("recent.files", [])
        
        # Remove if already exists
        if file_path in recent_files:
            recent_files.remove(file_path)
        
        # Add to beginning
        recent_files.insert(0, file_path)
        
        # Limit size
        max_items = self.get("recent.max_recent_items", 10)
        recent_files = recent_files[:max_items]
        
        self.set("recent.files", recent_files)
    
    def add_recent_directory(self, dir_path: str):
        """Add directory to recent directories list."""
        recent_dirs = self.get("recent.directories", [])
        
        if dir_path in recent_dirs:
            recent_dirs.remove(dir_path)
        
        recent_dirs.insert(0, dir_path)
        
        max_items = self.get("recent.max_recent_items", 10)
        recent_dirs = recent_dirs[:max_items]
        
        self.set("recent.directories", recent_dirs)
    
    def get_theme_settings(self) -> Dict[str, Any]:
        """Get current theme settings."""
        return self.get("theme", {})
    
    def apply_theme(self):
        """Apply current theme settings to the application."""
        theme_settings = self.get_theme_settings()
        
        # Set appearance mode
        mode = theme_settings.get("mode", "dark")
        if mode == "system":
            ctk.set_appearance_mode("system")
        else:
            ctk.set_appearance_mode(mode)
        
        # Set color theme
        color_scheme = theme_settings.get("color_scheme", "blue")
        ctk.set_default_color_theme(color_scheme)
        
        logger.info(f"Applied theme: {mode} mode with {color_scheme} color scheme")
    
    def validate_settings(self) -> List[str]:
        """Validate current settings and return list of issues."""
        issues = []
        
        try:
            # Validate theme settings
            theme = self.get("theme", {})
            if theme.get("mode") not in ["light", "dark", "system"]:
                issues.append("Invalid theme mode")
            
            if theme.get("color_scheme") not in ["blue", "green", "red", "orange"]:
                issues.append("Invalid color scheme")
            
            # Validate analysis settings
            analysis = self.get("analysis", {})
            sigma_threshold = analysis.get("sigma_threshold", 0.1)
            if not isinstance(sigma_threshold, (int, float)) or sigma_threshold <= 0:
                issues.append("Invalid sigma threshold")
            
            # Validate performance settings
            performance = self.get("performance", {})
            max_memory = performance.get("max_memory_usage", 2048)
            if not isinstance(max_memory, int) or max_memory < 512:
                issues.append("Invalid memory usage limit")
            
            # Validate export directory
            export_dir = self.get("export.default_directory")
            if export_dir and not os.path.exists(export_dir):
                issues.append("Export directory does not exist")
                
        except Exception as e:
            issues.append(f"Settings validation error: {str(e)}")
        
        return issues
    
    def _merge_settings(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge settings dictionaries."""
        result = base.copy()
        
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_settings(result[key], value)
            else:
                result[key] = value
        
        return result


class SettingsDialog(ctk.CTkToplevel):
    """Settings dialog for user preferences."""
    
    def __init__(self, parent, settings_manager: SettingsManager):
        super().__init__(parent)
        
        self.settings_manager = settings_manager
        self.temp_settings = {}  # Temporary settings before applying
        
        self.title("Application Settings")
        self.geometry("800x600")
        self.transient(parent)
        self.grab_set()
        
        # Center window
        self.center_window()
        
        self.create_widgets()
        self.load_current_settings()
    
    def center_window(self):
        """Center the dialog on screen."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_widgets(self):
        """Create dialog widgets."""
        # Main container
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="Application Settings",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=(10, 20))
        
        # Settings tabs
        self.tabview = ctk.CTkTabview(main_frame)
        self.tabview.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Create tabs
        self.create_theme_tab()
        self.create_analysis_tab()
        self.create_display_tab()
        self.create_performance_tab()
        self.create_export_tab()
        self.create_advanced_tab()
        
        # Buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Reset to Defaults",
            command=self.reset_to_defaults,
            width=120
        ).pack(side='left', padx=(10, 5), pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Import Settings",
            command=self.import_settings,
            width=120
        ).pack(side='left', padx=5, pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Export Settings",
            command=self.export_settings,
            width=120
        ).pack(side='left', padx=5, pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.cancel,
            width=100
        ).pack(side='right', padx=(5, 10), pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Apply",
            command=self.apply_settings,
            width=100,
            fg_color="green",
            hover_color="darkgreen"
        ).pack(side='right', padx=5, pady=10)
    
    def create_theme_tab(self):
        """Create theme settings tab."""
        self.tabview.add("Theme")
        tab = self.tabview.tab("Theme")
        
        # Theme mode
        ctk.CTkLabel(tab, text="Appearance Mode:", font=ctk.CTkFont(weight="bold")).pack(anchor='w', padx=10, pady=(10, 5))
        self.theme_mode = ctk.CTkSegmentedButton(
            tab,
            values=["Light", "Dark", "System"],
            command=self.on_theme_change
        )
        self.theme_mode.pack(fill='x', padx=10, pady=(0, 10))
        
        # Color scheme
        ctk.CTkLabel(tab, text="Color Scheme:", font=ctk.CTkFont(weight="bold")).pack(anchor='w', padx=10, pady=(10, 5))
        self.color_scheme = ctk.CTkSegmentedButton(
            tab,
            values=["Blue", "Green", "Red", "Orange"]
        )
        self.color_scheme.pack(fill='x', padx=10, pady=(0, 10))
        
        # Font settings
        font_frame = ctk.CTkFrame(tab)
        font_frame.pack(fill='x', padx=10, pady=10)
        
        ctk.CTkLabel(font_frame, text="Font Family:").pack(side='left', padx=10, pady=10)
        self.font_family = ctk.CTkComboBox(
            font_frame,
            values=["Segoe UI", "Arial", "Helvetica", "Times New Roman", "Courier New"],
            width=150
        )
        self.font_family.pack(side='left', padx=(0, 20), pady=10)
        
        ctk.CTkLabel(font_frame, text="Font Size:").pack(side='left', padx=10, pady=10)
        self.font_size = ctk.CTkComboBox(
            font_frame,
            values=["10", "11", "12", "13", "14", "16", "18"],
            width=80
        )
        self.font_size.pack(side='left', padx=(0, 10), pady=10)
        
        # Animation settings
        self.enable_animations = ctk.CTkCheckBox(tab, text="Enable Animations")
        self.enable_animations.pack(anchor='w', padx=10, pady=5)
        
        self.compact_mode = ctk.CTkCheckBox(tab, text="Compact Mode")
        self.compact_mode.pack(anchor='w', padx=10, pady=5)
        
        self.high_contrast = ctk.CTkCheckBox(tab, text="High Contrast Mode")
        self.high_contrast.pack(anchor='w', padx=10, pady=5)
    
    def create_analysis_tab(self):
        """Create analysis settings tab."""
        self.tabview.add("Analysis")
        tab = self.tabview.tab("Analysis")
        
        # Threshold settings
        threshold_frame = ctk.CTkFrame(tab)
        threshold_frame.pack(fill='x', padx=10, pady=10)
        
        ctk.CTkLabel(threshold_frame, text="Sigma Threshold:").pack(side='left', padx=10, pady=10)
        self.sigma_threshold = ctk.CTkEntry(threshold_frame, width=100)
        self.sigma_threshold.pack(side='left', padx=(0, 20), pady=10)
        
        ctk.CTkLabel(threshold_frame, text="Linearity Threshold:").pack(side='left', padx=10, pady=10)
        self.linearity_threshold = ctk.CTkEntry(threshold_frame, width=100)
        self.linearity_threshold.pack(side='left', padx=(0, 10), pady=10)
        
        # ML settings
        self.enable_ml_predictions = ctk.CTkCheckBox(tab, text="Enable ML Predictions")
        self.enable_ml_predictions.pack(anchor='w', padx=10, pady=5)
        
        confidence_frame = ctk.CTkFrame(tab)
        confidence_frame.pack(fill='x', padx=10, pady=10)
        
        ctk.CTkLabel(confidence_frame, text="ML Confidence Threshold:").pack(side='left', padx=10, pady=10)
        self.confidence_threshold = ctk.CTkSlider(
            confidence_frame,
            from_=0.1,
            to=1.0,
            number_of_steps=9
        )
        self.confidence_threshold.pack(side='left', fill='x', expand=True, padx=(0, 10), pady=10)
        
        self.confidence_label = ctk.CTkLabel(confidence_frame, text="0.8")
        self.confidence_label.pack(side='right', padx=10, pady=10)
        
        # Batch processing
        batch_frame = ctk.CTkFrame(tab)
        batch_frame.pack(fill='x', padx=10, pady=10)
        
        ctk.CTkLabel(batch_frame, text="Batch Size Limit:").pack(side='left', padx=10, pady=10)
        self.batch_size_limit = ctk.CTkEntry(batch_frame, width=100)
        self.batch_size_limit.pack(side='left', padx=(0, 10), pady=10)
        
        self.auto_save_results = ctk.CTkCheckBox(tab, text="Auto-save Analysis Results")
        self.auto_save_results.pack(anchor='w', padx=10, pady=5)
    
    def create_display_tab(self):
        """Create display settings tab."""
        self.tabview.add("Display")
        tab = self.tabview.tab("Display")
        
        # Chart settings
        ctk.CTkLabel(tab, text="Chart Style:", font=ctk.CTkFont(weight="bold")).pack(anchor='w', padx=10, pady=(10, 5))
        self.chart_style = ctk.CTkSegmentedButton(
            tab,
            values=["Modern", "Classic", "Minimal"]
        )
        self.chart_style.pack(fill='x', padx=10, pady=(0, 10))
        
        # Display options
        self.show_tooltips = ctk.CTkCheckBox(tab, text="Show Tooltips")
        self.show_tooltips.pack(anchor='w', padx=10, pady=5)
        
        self.show_progress_details = ctk.CTkCheckBox(tab, text="Show Progress Details")
        self.show_progress_details.pack(anchor='w', padx=10, pady=5)
        
        self.grid_lines = ctk.CTkCheckBox(tab, text="Show Grid Lines in Charts")
        self.grid_lines.pack(anchor='w', padx=10, pady=5)
        
        self.color_blind_friendly = ctk.CTkCheckBox(tab, text="Color Blind Friendly Mode")
        self.color_blind_friendly.pack(anchor='w', padx=10, pady=5)
        
        # Number formatting
        number_frame = ctk.CTkFrame(tab)
        number_frame.pack(fill='x', padx=10, pady=10)
        
        ctk.CTkLabel(number_frame, text="Decimal Places:").pack(side='left', padx=10, pady=10)
        self.decimal_places = ctk.CTkComboBox(
            number_frame,
            values=["2", "3", "4", "5", "6"],
            width=80
        )
        self.decimal_places.pack(side='left', padx=(0, 20), pady=10)
        
        self.scientific_notation = ctk.CTkCheckBox(number_frame, text="Scientific Notation")
        self.scientific_notation.pack(side='left', padx=10, pady=10)
    
    def create_performance_tab(self):
        """Create performance settings tab."""
        self.tabview.add("Performance")
        tab = self.tabview.tab("Performance")
        
        # Memory settings
        memory_frame = ctk.CTkFrame(tab)
        memory_frame.pack(fill='x', padx=10, pady=10)
        
        ctk.CTkLabel(memory_frame, text="Max Memory Usage (MB):").pack(side='left', padx=10, pady=10)
        self.max_memory_usage = ctk.CTkEntry(memory_frame, width=100)
        self.max_memory_usage.pack(side='left', padx=(0, 20), pady=10)
        
        ctk.CTkLabel(memory_frame, text="Thread Pool Size:").pack(side='left', padx=10, pady=10)
        self.thread_pool_size = ctk.CTkComboBox(
            memory_frame,
            values=["1", "2", "4", "8", "16"],
            width=80
        )
        self.thread_pool_size.pack(side='left', padx=(0, 10), pady=10)
        
        # Cache settings
        self.enable_caching = ctk.CTkCheckBox(tab, text="Enable Caching")
        self.enable_caching.pack(anchor='w', padx=10, pady=5)
        
        cache_frame = ctk.CTkFrame(tab)
        cache_frame.pack(fill='x', padx=10, pady=10)
        
        ctk.CTkLabel(cache_frame, text="Cache Size Limit (MB):").pack(side='left', padx=10, pady=10)
        self.cache_size_limit = ctk.CTkEntry(cache_frame, width=100)
        self.cache_size_limit.pack(side='left', padx=(0, 10), pady=10)
        
        # Processing options
        self.auto_cleanup = ctk.CTkCheckBox(tab, text="Auto Cleanup Temporary Files")
        self.auto_cleanup.pack(anchor='w', padx=10, pady=5)
        
        self.background_processing = ctk.CTkCheckBox(tab, text="Enable Background Processing")
        self.background_processing.pack(anchor='w', padx=10, pady=5)
    
    def create_export_tab(self):
        """Create export settings tab."""
        self.tabview.add("Export")
        tab = self.tabview.tab("Export")
        
        # Default format
        ctk.CTkLabel(tab, text="Default Export Format:", font=ctk.CTkFont(weight="bold")).pack(anchor='w', padx=10, pady=(10, 5))
        self.export_format = ctk.CTkSegmentedButton(
            tab,
            values=["Excel", "CSV", "PDF"]
        )
        self.export_format.pack(fill='x', padx=10, pady=(0, 10))
        
        # Directory selection
        dir_frame = ctk.CTkFrame(tab)
        dir_frame.pack(fill='x', padx=10, pady=10)
        
        ctk.CTkLabel(dir_frame, text="Default Directory:").pack(side='left', padx=10, pady=10)
        self.export_directory = ctk.CTkEntry(dir_frame, width=300)
        self.export_directory.pack(side='left', fill='x', expand=True, padx=(0, 5), pady=10)
        
        ctk.CTkButton(
            dir_frame,
            text="Browse",
            command=self.browse_export_directory,
            width=80
        ).pack(side='right', padx=10, pady=10)
        
        # Export options
        self.include_metadata = ctk.CTkCheckBox(tab, text="Include Metadata in Exports")
        self.include_metadata.pack(anchor='w', padx=10, pady=5)
        
        self.include_charts = ctk.CTkCheckBox(tab, text="Include Charts in Exports")
        self.include_charts.pack(anchor='w', padx=10, pady=5)
        
        self.auto_timestamp = ctk.CTkCheckBox(tab, text="Auto-add Timestamp to Filenames")
        self.auto_timestamp.pack(anchor='w', padx=10, pady=5)
        
        # Chart resolution
        chart_frame = ctk.CTkFrame(tab)
        chart_frame.pack(fill='x', padx=10, pady=10)
        
        ctk.CTkLabel(chart_frame, text="Chart Resolution:").pack(side='left', padx=10, pady=10)
        self.chart_resolution = ctk.CTkSegmentedButton(
            chart_frame,
            values=["Low", "Medium", "High"]
        )
        self.chart_resolution.pack(side='left', padx=(20, 10), pady=10)
    
    def create_advanced_tab(self):
        """Create advanced settings tab."""
        self.tabview.add("Advanced")
        tab = self.tabview.tab("Advanced")
        
        # Debug options
        self.debug_mode = ctk.CTkCheckBox(tab, text="Debug Mode")
        self.debug_mode.pack(anchor='w', padx=10, pady=5)
        
        log_frame = ctk.CTkFrame(tab)
        log_frame.pack(fill='x', padx=10, pady=10)
        
        ctk.CTkLabel(log_frame, text="Log Level:").pack(side='left', padx=10, pady=10)
        self.log_level = ctk.CTkComboBox(
            log_frame,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            width=100
        )
        self.log_level.pack(side='left', padx=(0, 10), pady=10)
        
        # Feature flags
        self.experimental_features = ctk.CTkCheckBox(tab, text="Enable Experimental Features")
        self.experimental_features.pack(anchor='w', padx=10, pady=5)
        
        self.beta_features = ctk.CTkCheckBox(tab, text="Enable Beta Features")
        self.beta_features.pack(anchor='w', padx=10, pady=5)
        
        # Telemetry
        self.telemetry = ctk.CTkCheckBox(tab, text="Send Anonymous Usage Data")
        self.telemetry.pack(anchor='w', padx=10, pady=5)
        
        self.auto_updates = ctk.CTkCheckBox(tab, text="Check for Updates Automatically")
        self.auto_updates.pack(anchor='w', padx=10, pady=5)
    
    def on_theme_change(self, value):
        """Handle theme mode change with live preview."""
        mode_map = {"Light": "light", "Dark": "dark", "System": "system"}
        mode = mode_map.get(value, "dark")
        
        if mode == "system":
            ctk.set_appearance_mode("system")
        else:
            ctk.set_appearance_mode(mode)
    
    def browse_export_directory(self):
        """Browse for export directory."""
        from tkinter import filedialog
        
        directory = filedialog.askdirectory(
            title="Select Default Export Directory",
            initialdir=self.export_directory.get() or str(Path.home())
        )
        
        if directory:
            self.export_directory.delete(0, 'end')
            self.export_directory.insert(0, directory)
    
    def load_current_settings(self):
        """Load current settings into dialog."""
        # Theme settings
        theme_mode = self.settings_manager.get("theme.mode", "dark")
        mode_map = {"light": "Light", "dark": "Dark", "system": "System"}
        self.theme_mode.set(mode_map.get(theme_mode, "Dark"))
        
        color_scheme = self.settings_manager.get("theme.color_scheme", "blue")
        scheme_map = {"blue": "Blue", "green": "Green", "red": "Red", "orange": "Orange"}
        self.color_scheme.set(scheme_map.get(color_scheme, "Blue"))
        
        self.font_family.set(self.settings_manager.get("theme.font_family", "Segoe UI"))
        self.font_size.set(str(self.settings_manager.get("theme.font_size", 12)))
        
        # Checkboxes
        self.enable_animations.select() if self.settings_manager.get("theme.enable_animations", True) else self.enable_animations.deselect()
        self.compact_mode.select() if self.settings_manager.get("theme.compact_mode", False) else self.compact_mode.deselect()
        self.high_contrast.select() if self.settings_manager.get("theme.high_contrast", False) else self.high_contrast.deselect()
        
        # Analysis settings
        self.sigma_threshold.insert(0, str(self.settings_manager.get("analysis.sigma_threshold", 0.1)))
        self.linearity_threshold.insert(0, str(self.settings_manager.get("analysis.linearity_threshold", 0.05)))
        self.confidence_threshold.set(self.settings_manager.get("analysis.confidence_threshold", 0.8))
        self.batch_size_limit.insert(0, str(self.settings_manager.get("analysis.batch_size_limit", 1000)))
        
        # Continue loading other settings...
        # (Implementation continues for all other settings)
    
    def apply_settings(self):
        """Apply settings and close dialog."""
        try:
            # Collect all settings
            # Theme
            mode_map = {"Light": "light", "Dark": "dark", "System": "system"}
            self.settings_manager.set("theme.mode", mode_map[self.theme_mode.get()])
            
            scheme_map = {"Blue": "blue", "Green": "green", "Red": "red", "Orange": "orange"}
            self.settings_manager.set("theme.color_scheme", scheme_map[self.color_scheme.get()])
            
            self.settings_manager.set("theme.font_family", self.font_family.get())
            self.settings_manager.set("theme.font_size", int(self.font_size.get()))
            self.settings_manager.set("theme.enable_animations", bool(self.enable_animations.get()))
            self.settings_manager.set("theme.compact_mode", bool(self.compact_mode.get()))
            self.settings_manager.set("theme.high_contrast", bool(self.high_contrast.get()))
            
            # Analysis
            sigma_val = self.sigma_threshold.get().strip()
            if sigma_val and sigma_val.lower() != 'none':
                try:
                    self.settings_manager.set("analysis.sigma_threshold", float(sigma_val))
                except ValueError:
                    self.settings_manager.set("analysis.sigma_threshold", None)
            else:
                self.settings_manager.set("analysis.sigma_threshold", None)
                
            linearity_val = self.linearity_threshold.get().strip()
            if linearity_val and linearity_val.lower() != 'none':
                try:
                    self.settings_manager.set("analysis.linearity_threshold", float(linearity_val))
                except ValueError:
                    self.settings_manager.set("analysis.linearity_threshold", None)
            else:
                self.settings_manager.set("analysis.linearity_threshold", None)
            
            # Confidence threshold is from a slider, should always be a float
            confidence_val = self.confidence_threshold.get()
            if isinstance(confidence_val, (int, float)):
                self.settings_manager.set("analysis.confidence_threshold", float(confidence_val))
            else:
                self.settings_manager.set("analysis.confidence_threshold", 0.95)
            self.settings_manager.set("analysis.batch_size_limit", int(self.batch_size_limit.get()))
            
            # Apply theme immediately
            self.settings_manager.apply_theme()
            
            self.destroy()
            
        except Exception as e:
            logger.error(f"Error applying settings: {e}")
            messagebox.showerror("Error", f"Failed to apply settings:\n{str(e)}")
    
    def cancel(self):
        """Cancel changes and close dialog."""
        self.destroy()
    
    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        if messagebox.askyesno("Reset Settings", 
                               "Are you sure you want to reset all settings to defaults?"):
            self.settings_manager.reset_to_defaults()
            self.load_current_settings()
    
    def import_settings(self):
        """Import settings from file."""
        from tkinter import filedialog
        
        file_path = filedialog.askopenfilename(
            title="Import Settings",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            if self.settings_manager.import_settings(file_path):
                self.load_current_settings()
                messagebox.showinfo("Success", "Settings imported successfully!")
            else:
                messagebox.showerror("Error", "Failed to import settings.")
    
    def export_settings(self):
        """Export settings to file."""
        from tkinter import filedialog
        
        file_path = filedialog.asksaveasfilename(
            title="Export Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            if self.settings_manager.export_settings(file_path):
                messagebox.showinfo("Success", "Settings exported successfully!")
            else:
                messagebox.showerror("Error", "Failed to export settings.")


# Global settings manager instance
settings_manager = SettingsManager() 
