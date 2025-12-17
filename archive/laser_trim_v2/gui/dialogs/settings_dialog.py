"""
SettingsDialog - Comprehensive settings dialog with validation

Provides a tabbed interface for editing all application settings
with input validation and persistence.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import yaml

from laser_trim_analyzer.core.config import Config


class SettingsDialog(tk.Toplevel):
    """
    Comprehensive settings dialog for the application.

    Features:
    - Tabbed interface for different settings categories
    - Input validation
    - Save/load configuration
    - Apply without closing
    - Reset to defaults
    """

    def __init__(self, parent, config: Config):
        """
        Initialize SettingsDialog.

        Args:
            parent: Parent window
            config: Current configuration object
        """
        super().__init__(parent)

        self.config = config
        self.original_config = self._deep_copy_config(config)
        self.result = False

        # Window setup
        self.title("Settings")
        self.geometry("800x600")
        self.minsize(700, 500)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Variables for settings
        self._create_variables()

        # Create UI
        self._create_ui()

        # Load current settings
        self._load_settings()

        # Center window
        self._center_window()

    def _create_variables(self):
        """Create tkinter variables for settings."""
        self.vars = {}

        # Database settings
        self.vars['db_enabled'] = tk.BooleanVar()
        self.vars['db_path'] = tk.StringVar()
        self.vars['db_echo'] = tk.BooleanVar()
        self.vars['db_pool_size'] = tk.IntVar()

        # Processing settings
        self.vars['max_workers'] = tk.IntVar()
        self.vars['generate_plots'] = tk.BooleanVar()
        self.vars['plot_dpi'] = tk.IntVar()
        self.vars['cache_enabled'] = tk.BooleanVar()
        self.vars['cache_ttl'] = tk.IntVar()

        # Analysis settings
        self.vars['sigma_scaling'] = tk.DoubleVar()
        self.vars['gradient_step'] = tk.IntVar()
        self.vars['filter_sampling'] = tk.IntVar()
        self.vars['filter_cutoff'] = tk.IntVar()
        self.vars['lm_compliance_mode'] = tk.BooleanVar()
        self.vars['num_zones'] = tk.IntVar()
        self.vars['high_risk_threshold'] = tk.DoubleVar()
        self.vars['low_risk_threshold'] = tk.DoubleVar()

        # ML settings
        self.vars['ml_enabled'] = tk.BooleanVar()
        self.vars['ml_model_path'] = tk.StringVar()
        self.vars['failure_prediction'] = tk.BooleanVar()
        self.vars['threshold_optimization'] = tk.BooleanVar()
        self.vars['retrain_interval'] = tk.IntVar()

        # API settings
        self.vars['api_enabled'] = tk.BooleanVar()
        self.vars['api_base_url'] = tk.StringVar()
        self.vars['api_key'] = tk.StringVar()
        self.vars['api_timeout'] = tk.IntVar()
        self.vars['api_retries'] = tk.IntVar()

        # GUI settings
        self.vars['theme'] = tk.StringVar()
        self.vars['window_width'] = tk.IntVar()
        self.vars['window_height'] = tk.IntVar()
        self.vars['show_historical'] = tk.BooleanVar()
        self.vars['show_ml_insights'] = tk.BooleanVar()
        self.vars['autosave_enabled'] = tk.BooleanVar()
        self.vars['autosave_interval'] = tk.IntVar()

    def _create_ui(self):
        """Create the dialog UI."""
        # Main container
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill='both', expand=True)

        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)

        # Create tabs
        self._create_general_tab()
        self._create_database_tab()
        self._create_processing_tab()
        self._create_analysis_tab()
        self._create_ml_tab()
        self._create_api_tab()
        self._create_gui_tab()

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))

        # Buttons
        ttk.Button(
            button_frame,
            text="OK",
            command=self._on_ok
        ).pack(side='right', padx=(5, 0))

        ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel
        ).pack(side='right')

        ttk.Button(
            button_frame,
            text="Apply",
            command=self._on_apply
        ).pack(side='right', padx=(0, 20))

        ttk.Button(
            button_frame,
            text="Reset to Defaults",
            command=self._on_reset
        ).pack(side='left')

        ttk.Button(
            button_frame,
            text="Export...",
            command=self._export_config
        ).pack(side='left', padx=(10, 0))

        ttk.Button(
            button_frame,
            text="Import...",
            command=self._import_config
        ).pack(side='left', padx=(5, 0))

    def _create_general_tab(self):
        """Create general settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="General")

        # Scrollable frame
        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Settings content
        frame = ttk.LabelFrame(scrollable_frame, text="Application Settings", padding="10")
        frame.pack(fill='x', padx=10, pady=10)

        # Data directory
        ttk.Label(frame, text="Data Directory:").grid(row=0, column=0, sticky='w', pady=5)
        data_frame = ttk.Frame(frame)
        data_frame.grid(row=0, column=1, sticky='ew', pady=5)

        self.data_dir_var = tk.StringVar(value=str(self.config.data_directory))
        ttk.Entry(data_frame, textvariable=self.data_dir_var).pack(side='left', fill='x', expand=True)
        ttk.Button(
            data_frame,
            text="Browse...",
            command=lambda: self._browse_directory(self.data_dir_var)
        ).pack(side='left', padx=(5, 0))

        # Log directory
        ttk.Label(frame, text="Log Directory:").grid(row=1, column=0, sticky='w', pady=5)
        log_frame = ttk.Frame(frame)
        log_frame.grid(row=1, column=1, sticky='ew', pady=5)

        self.log_dir_var = tk.StringVar(value=str(self.config.log_directory))
        ttk.Entry(log_frame, textvariable=self.log_dir_var).pack(side='left', fill='x', expand=True)
        ttk.Button(
            log_frame,
            text="Browse...",
            command=lambda: self._browse_directory(self.log_dir_var)
        ).pack(side='left', padx=(5, 0))

        # Debug mode
        ttk.Label(frame, text="Debug Mode:").grid(row=2, column=0, sticky='w', pady=5)
        self.debug_var = tk.BooleanVar(value=self.config.debug)
        ttk.Checkbutton(frame, variable=self.debug_var).grid(row=2, column=1, sticky='w', pady=5)

        frame.columnconfigure(1, weight=1)

    def _create_database_tab(self):
        """Create database settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Database")

        # Database settings
        frame = ttk.LabelFrame(tab, text="Database Configuration", padding="10")
        frame.pack(fill='x', padx=10, pady=10)

        # Enable database
        ttk.Label(frame, text="Enable Database:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Checkbutton(
            frame,
            variable=self.vars['db_enabled'],
            command=self._toggle_database_settings
        ).grid(row=0, column=1, sticky='w', pady=5)

        # Database path
        ttk.Label(frame, text="Database Path:").grid(row=1, column=0, sticky='w', pady=5)
        db_frame = ttk.Frame(frame)
        db_frame.grid(row=1, column=1, sticky='ew', pady=5)

        self.db_path_entry = ttk.Entry(db_frame, textvariable=self.vars['db_path'])
        self.db_path_entry.pack(side='left', fill='x', expand=True)

        self.db_browse_btn = ttk.Button(
            db_frame,
            text="Browse...",
            command=self._browse_db_file
        )
        self.db_browse_btn.pack(side='left', padx=(5, 0))

        # Echo SQL
        ttk.Label(frame, text="Echo SQL Statements:").grid(row=2, column=0, sticky='w', pady=5)
        self.db_echo_check = ttk.Checkbutton(frame, variable=self.vars['db_echo'])
        self.db_echo_check.grid(row=2, column=1, sticky='w', pady=5)

        # Pool size
        ttk.Label(frame, text="Connection Pool Size:").grid(row=3, column=0, sticky='w', pady=5)
        self.db_pool_spin = ttk.Spinbox(
            frame,
            textvariable=self.vars['db_pool_size'],
            from_=1,
            to=20,
            width=10
        )
        self.db_pool_spin.grid(row=3, column=1, sticky='w', pady=5)

        frame.columnconfigure(1, weight=1)

        # Connection test
        test_frame = ttk.Frame(tab)
        test_frame.pack(fill='x', padx=10, pady=10)

        ttk.Button(
            test_frame,
            text="Test Connection",
            command=self._test_db_connection
        ).pack(side='left')

        self.db_test_label = ttk.Label(test_frame, text="")
        self.db_test_label.pack(side='left', padx=10)

    def _create_processing_tab(self):
        """Create processing settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Processing")

        # Processing settings
        frame = ttk.LabelFrame(tab, text="Processing Options", padding="10")
        frame.pack(fill='x', padx=10, pady=10)

        # Max workers
        ttk.Label(frame, text="Max Parallel Workers:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Spinbox(
            frame,
            textvariable=self.vars['max_workers'],
            from_=1,
            to=16,
            width=10
        ).grid(row=0, column=1, sticky='w', pady=5)

        # Generate plots
        ttk.Label(frame, text="Generate Plots:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Checkbutton(
            frame,
            variable=self.vars['generate_plots'],
            command=self._toggle_plot_settings
        ).grid(row=1, column=1, sticky='w', pady=5)

        # Plot DPI
        ttk.Label(frame, text="Plot Resolution (DPI):").grid(row=2, column=0, sticky='w', pady=5)
        self.plot_dpi_spin = ttk.Spinbox(
            frame,
            textvariable=self.vars['plot_dpi'],
            from_=72,
            to=300,
            increment=50,
            width=10
        )
        self.plot_dpi_spin.grid(row=2, column=1, sticky='w', pady=5)

        # Cache settings
        ttk.Label(frame, text="Enable Result Cache:").grid(row=3, column=0, sticky='w', pady=5)
        ttk.Checkbutton(
            frame,
            variable=self.vars['cache_enabled'],
            command=self._toggle_cache_settings
        ).grid(row=3, column=1, sticky='w', pady=5)

        # Cache TTL
        ttk.Label(frame, text="Cache TTL (seconds):").grid(row=4, column=0, sticky='w', pady=5)
        self.cache_ttl_spin = ttk.Spinbox(
            frame,
            textvariable=self.vars['cache_ttl'],
            from_=60,
            to=86400,
            increment=300,
            width=10
        )
        self.cache_ttl_spin.grid(row=4, column=1, sticky='w', pady=5)

        frame.columnconfigure(1, weight=1)

        # File handling
        file_frame = ttk.LabelFrame(tab, text="File Handling", padding="10")
        file_frame.pack(fill='x', padx=10, pady=10)

        # File extensions
        ttk.Label(file_frame, text="Accepted Extensions:").grid(row=0, column=0, sticky='nw', pady=5)
        self.extensions_text = tk.Text(file_frame, height=3, width=30)
        self.extensions_text.grid(row=0, column=1, sticky='ew', pady=5)
        self.extensions_text.insert('1.0', '\n'.join(self.config.processing.file_extensions))

        # Skip patterns
        ttk.Label(file_frame, text="Skip Patterns:").grid(row=1, column=0, sticky='nw', pady=5)
        self.skip_text = tk.Text(file_frame, height=3, width=30)
        self.skip_text.grid(row=1, column=1, sticky='ew', pady=5)
        self.skip_text.insert('1.0', '\n'.join(self.config.processing.skip_patterns))

        file_frame.columnconfigure(1, weight=1)

    def _create_analysis_tab(self):
        """Create analysis settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Analysis")

        # Sigma analysis
        sigma_frame = ttk.LabelFrame(tab, text="Sigma Analysis", padding="10")
        sigma_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(sigma_frame, text="Scaling Factor:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Spinbox(
            sigma_frame,
            textvariable=self.vars['sigma_scaling'],
            from_=1.0,
            to=100.0,
            increment=0.5,
            width=10,
            format="%.1f"
        ).grid(row=0, column=1, sticky='w', pady=5)

        ttk.Label(sigma_frame, text="Gradient Step:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Spinbox(
            sigma_frame,
            textvariable=self.vars['gradient_step'],
            from_=1,
            to=10,
            width=10
        ).grid(row=1, column=1, sticky='w', pady=5)

        sigma_frame.columnconfigure(1, weight=1)

        # Filtering
        filter_frame = ttk.LabelFrame(tab, text="Filtering", padding="10")
        filter_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(filter_frame, text="Sampling Frequency:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Spinbox(
            filter_frame,
            textvariable=self.vars['filter_sampling'],
            from_=10,
            to=1000,
            increment=10,
            width=10
        ).grid(row=0, column=1, sticky='w', pady=5)

        ttk.Label(filter_frame, text="Cutoff Frequency:").grid(row=1, column=0, sticky='w', pady=5)
        self.cutoff_spin = ttk.Spinbox(
            filter_frame,
            textvariable=self.vars['filter_cutoff'],
            from_=10,
            to=500,
            increment=10,
            width=10
        )
        self.cutoff_spin.grid(row=1, column=1, sticky='w', pady=5)
        
        # Lockheed Martin Compliance Mode
        ttk.Label(filter_frame, text="LM Compliance Mode:").grid(row=2, column=0, sticky='w', pady=5)
        self.lm_compliance_check = ttk.Checkbutton(
            filter_frame,
            variable=self.vars['lm_compliance_mode'],
            command=self._toggle_lm_compliance
        )
        self.lm_compliance_check.grid(row=2, column=1, sticky='w', pady=5)
        
        # Warning label for LM compliance mode
        self.lm_warning_label = ttk.Label(
            filter_frame,
            text="⚠️ Uses original LM recursive filter (80Hz) - may override cutoff frequency",
            foreground="orange",
            font=('Segoe UI', 8)
        )
        self.lm_warning_label.grid(row=3, column=0, columnspan=2, sticky='w', pady=2)

        filter_frame.columnconfigure(1, weight=1)

        # Risk thresholds
        risk_frame = ttk.LabelFrame(tab, text="Risk Thresholds", padding="10")
        risk_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(risk_frame, text="High Risk Threshold:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Spinbox(
            risk_frame,
            textvariable=self.vars['high_risk_threshold'],
            from_=0.0,
            to=1.0,
            increment=0.05,
            width=10,
            format="%.2f"
        ).grid(row=0, column=1, sticky='w', pady=5)

        ttk.Label(risk_frame, text="Low Risk Threshold:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Spinbox(
            risk_frame,
            textvariable=self.vars['low_risk_threshold'],
            from_=0.0,
            to=1.0,
            increment=0.05,
            width=10,
            format="%.2f"
        ).grid(row=1, column=1, sticky='w', pady=5)

        risk_frame.columnconfigure(1, weight=1)

    def _create_ml_tab(self):
        """Create ML settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Machine Learning")

        # ML settings
        frame = ttk.LabelFrame(tab, text="ML Configuration", padding="10")
        frame.pack(fill='x', padx=10, pady=10)

        # Enable ML
        ttk.Label(frame, text="Enable ML Features:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Checkbutton(
            frame,
            variable=self.vars['ml_enabled'],
            command=self._toggle_ml_settings
        ).grid(row=0, column=1, sticky='w', pady=5)

        # Model path
        ttk.Label(frame, text="Model Directory:").grid(row=1, column=0, sticky='w', pady=5)
        model_frame = ttk.Frame(frame)
        model_frame.grid(row=1, column=1, sticky='ew', pady=5)

        self.ml_path_entry = ttk.Entry(model_frame, textvariable=self.vars['ml_model_path'])
        self.ml_path_entry.pack(side='left', fill='x', expand=True)

        self.ml_browse_btn = ttk.Button(
            model_frame,
            text="Browse...",
            command=lambda: self._browse_directory(self.vars['ml_model_path'])
        )
        self.ml_browse_btn.pack(side='left', padx=(5, 0))

        # Features
        ttk.Label(frame, text="Failure Prediction:").grid(row=2, column=0, sticky='w', pady=5)
        self.failure_pred_check = ttk.Checkbutton(frame, variable=self.vars['failure_prediction'])
        self.failure_pred_check.grid(row=2, column=1, sticky='w', pady=5)

        ttk.Label(frame, text="Threshold Optimization:").grid(row=3, column=0, sticky='w', pady=5)
        self.threshold_opt_check = ttk.Checkbutton(frame, variable=self.vars['threshold_optimization'])
        self.threshold_opt_check.grid(row=3, column=1, sticky='w', pady=5)

        # Retrain interval
        ttk.Label(frame, text="Retrain Interval (days):").grid(row=4, column=0, sticky='w', pady=5)
        self.retrain_spin = ttk.Spinbox(
            frame,
            textvariable=self.vars['retrain_interval'],
            from_=1,
            to=365,
            width=10
        )
        self.retrain_spin.grid(row=4, column=1, sticky='w', pady=5)

        frame.columnconfigure(1, weight=1)

    def _create_api_tab(self):
        """Create API settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="API")

        # API settings
        frame = ttk.LabelFrame(tab, text="API Configuration", padding="10")
        frame.pack(fill='x', padx=10, pady=10)

        # Enable API
        ttk.Label(frame, text="Enable API Integration:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Checkbutton(
            frame,
            variable=self.vars['api_enabled'],
            command=self._toggle_api_settings
        ).grid(row=0, column=1, sticky='w', pady=5)

        # Base URL
        ttk.Label(frame, text="API Base URL:").grid(row=1, column=0, sticky='w', pady=5)
        self.api_url_entry = ttk.Entry(frame, textvariable=self.vars['api_base_url'], width=40)
        self.api_url_entry.grid(row=1, column=1, sticky='ew', pady=5)

        # API Key
        ttk.Label(frame, text="API Key:").grid(row=2, column=0, sticky='w', pady=5)
        self.api_key_entry = ttk.Entry(frame, textvariable=self.vars['api_key'], show='*', width=40)
        self.api_key_entry.grid(row=2, column=1, sticky='ew', pady=5)

        # Show/hide API key
        self.show_api_key = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Show API Key",
            variable=self.show_api_key,
            command=self._toggle_api_key_visibility
        ).grid(row=2, column=2, padx=5)

        # Timeout
        ttk.Label(frame, text="Timeout (seconds):").grid(row=3, column=0, sticky='w', pady=5)
        self.api_timeout_spin = ttk.Spinbox(
            frame,
            textvariable=self.vars['api_timeout'],
            from_=1,
            to=300,
            width=10
        )
        self.api_timeout_spin.grid(row=3, column=1, sticky='w', pady=5)

        # Retries
        ttk.Label(frame, text="Max Retries:").grid(row=4, column=0, sticky='w', pady=5)
        self.api_retries_spin = ttk.Spinbox(
            frame,
            textvariable=self.vars['api_retries'],
            from_=0,
            to=10,
            width=10
        )
        self.api_retries_spin.grid(row=4, column=1, sticky='w', pady=5)

        frame.columnconfigure(1, weight=1)

        # Test connection
        test_frame = ttk.Frame(tab)
        test_frame.pack(fill='x', padx=10, pady=10)

        ttk.Button(
            test_frame,
            text="Test API Connection",
            command=self._test_api_connection
        ).pack(side='left')

        self.api_test_label = ttk.Label(test_frame, text="")
        self.api_test_label.pack(side='left', padx=10)

    def _create_gui_tab(self):
        """Create GUI settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Interface")

        # Appearance
        appearance_frame = ttk.LabelFrame(tab, text="Appearance", padding="10")
        appearance_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(appearance_frame, text="Theme:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Combobox(
            appearance_frame,
            textvariable=self.vars['theme'],
            values=['clam', 'alt', 'default', 'classic'],
            state='readonly',
            width=20
        ).grid(row=0, column=1, sticky='w', pady=5)

        ttk.Label(appearance_frame, text="Window Width:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Spinbox(
            appearance_frame,
            textvariable=self.vars['window_width'],
            from_=800,
            to=2560,
            increment=50,
            width=10
        ).grid(row=1, column=1, sticky='w', pady=5)

        ttk.Label(appearance_frame, text="Window Height:").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Spinbox(
            appearance_frame,
            textvariable=self.vars['window_height'],
            from_=600,
            to=1440,
            increment=50,
            width=10
        ).grid(row=2, column=1, sticky='w', pady=5)

        appearance_frame.columnconfigure(1, weight=1)

        # Features
        features_frame = ttk.LabelFrame(tab, text="Features", padding="10")
        features_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(features_frame, text="Show Historical Tab:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Checkbutton(features_frame, variable=self.vars['show_historical']).grid(row=0, column=1, sticky='w', pady=5)

        ttk.Label(features_frame, text="Show ML Insights:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Checkbutton(features_frame, variable=self.vars['show_ml_insights']).grid(row=1, column=1, sticky='w',
                                                                                     pady=5)

        features_frame.columnconfigure(1, weight=1)

        # Auto-save
        autosave_frame = ttk.LabelFrame(tab, text="Auto-save", padding="10")
        autosave_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(autosave_frame, text="Enable Auto-save:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Checkbutton(
            autosave_frame,
            variable=self.vars['autosave_enabled'],
            command=self._toggle_autosave_settings
        ).grid(row=0, column=1, sticky='w', pady=5)

        ttk.Label(autosave_frame, text="Interval (seconds):").grid(row=1, column=0, sticky='w', pady=5)
        self.autosave_spin = ttk.Spinbox(
            autosave_frame,
            textvariable=self.vars['autosave_interval'],
            from_=60,
            to=3600,
            increment=60,
            width=10
        )
        self.autosave_spin.grid(row=1, column=1, sticky='w', pady=5)

        autosave_frame.columnconfigure(1, weight=1)

    def _load_settings(self):
        """Load current settings into UI."""
        # Database
        self.vars['db_enabled'].set(self.config.database.enabled)
        self.vars['db_path'].set(str(self.config.database.path))
        self.vars['db_echo'].set(self.config.database.echo)
        self.vars['db_pool_size'].set(self.config.database.pool_size)

        # Processing
        self.vars['max_workers'].set(self.config.processing.max_workers)
        self.vars['generate_plots'].set(self.config.processing.generate_plots)
        self.vars['plot_dpi'].set(self.config.processing.plot_dpi)
        self.vars['cache_enabled'].set(self.config.processing.cache_enabled)
        self.vars['cache_ttl'].set(self.config.processing.cache_ttl)

        # Analysis
        self.vars['sigma_scaling'].set(self.config.analysis.sigma_scaling_factor)
        self.vars['gradient_step'].set(self.config.analysis.matlab_gradient_step)
        self.vars['filter_sampling'].set(self.config.analysis.filter_sampling_frequency)
        self.vars['filter_cutoff'].set(self.config.analysis.filter_cutoff_frequency)
        self.vars['lm_compliance_mode'].set(self.config.analysis.lockheed_martin_compliance_mode)
        self.vars['num_zones'].set(self.config.analysis.default_num_zones)
        self.vars['high_risk_threshold'].set(self.config.analysis.high_risk_threshold)
        self.vars['low_risk_threshold'].set(self.config.analysis.low_risk_threshold)

        # ML
        self.vars['ml_enabled'].set(self.config.ml.enabled)
        self.vars['ml_model_path'].set(str(self.config.ml.model_path))
        self.vars['failure_prediction'].set(self.config.ml.failure_prediction_enabled)
        self.vars['threshold_optimization'].set(self.config.ml.threshold_optimization_enabled)
        self.vars['retrain_interval'].set(self.config.ml.retrain_interval_days)

        # API
        self.vars['api_enabled'].set(self.config.api.enabled)
        self.vars['api_base_url'].set(self.config.api.base_url)
        self.vars['api_key'].set(self.config.api.api_key or '')
        self.vars['api_timeout'].set(self.config.api.timeout)
        self.vars['api_retries'].set(self.config.api.max_retries)

        # GUI
        self.vars['theme'].set(self.config.gui.theme)
        self.vars['window_width'].set(self.config.gui.window_width)
        self.vars['window_height'].set(self.config.gui.window_height)
        self.vars['show_historical'].set(self.config.gui.show_historical_tab)
        self.vars['show_ml_insights'].set(self.config.gui.show_ml_insights)
        self.vars['autosave_enabled'].set(self.config.gui.autosave_enabled)
        self.vars['autosave_interval'].set(self.config.gui.autosave_interval)

        # Update UI states
        self._toggle_database_settings()
        self._toggle_plot_settings()
        self._toggle_cache_settings()
        self._toggle_ml_settings()
        self._toggle_api_settings()
        self._toggle_autosave_settings()
        self._toggle_lm_compliance()

    def _save_settings(self):
        """Save UI values to config object."""
        # General
        self.config.data_directory = Path(self.data_dir_var.get())
        self.config.log_directory = Path(self.log_dir_var.get())
        self.config.debug = self.debug_var.get()

        # Database
        self.config.database.enabled = self.vars['db_enabled'].get()
        self.config.database.path = Path(self.vars['db_path'].get())
        self.config.database.echo = self.vars['db_echo'].get()
        self.config.database.pool_size = self.vars['db_pool_size'].get()

        # Processing
        self.config.processing.max_workers = self.vars['max_workers'].get()
        self.config.processing.generate_plots = self.vars['generate_plots'].get()
        self.config.processing.plot_dpi = self.vars['plot_dpi'].get()
        self.config.processing.cache_enabled = self.vars['cache_enabled'].get()
        self.config.processing.cache_ttl = self.vars['cache_ttl'].get()

        # File extensions and patterns
        extensions = self.extensions_text.get('1.0', 'end-1c').strip().split('\n')
        self.config.processing.file_extensions = [ext.strip() for ext in extensions if ext.strip()]

        patterns = self.skip_text.get('1.0', 'end-1c').strip().split('\n')
        self.config.processing.skip_patterns = [pat.strip() for pat in patterns if pat.strip()]

        # Analysis
        self.config.analysis.sigma_scaling_factor = self.vars['sigma_scaling'].get()
        self.config.analysis.matlab_gradient_step = self.vars['gradient_step'].get()
        self.config.analysis.filter_sampling_frequency = self.vars['filter_sampling'].get()
        self.config.analysis.filter_cutoff_frequency = self.vars['filter_cutoff'].get()
        self.config.analysis.lockheed_martin_compliance_mode = self.vars['lm_compliance_mode'].get()
        self.config.analysis.default_num_zones = self.vars['num_zones'].get()
        self.config.analysis.high_risk_threshold = self.vars['high_risk_threshold'].get()
        self.config.analysis.low_risk_threshold = self.vars['low_risk_threshold'].get()

        # ML
        self.config.ml.enabled = self.vars['ml_enabled'].get()
        self.config.ml.model_path = Path(self.vars['ml_model_path'].get())
        self.config.ml.failure_prediction_enabled = self.vars['failure_prediction'].get()
        self.config.ml.threshold_optimization_enabled = self.vars['threshold_optimization'].get()
        self.config.ml.retrain_interval_days = self.vars['retrain_interval'].get()

        # API
        self.config.api.enabled = self.vars['api_enabled'].get()
        self.config.api.base_url = self.vars['api_base_url'].get()
        self.config.api.api_key = self.vars['api_key'].get() or None
        self.config.api.timeout = self.vars['api_timeout'].get()
        self.config.api.max_retries = self.vars['api_retries'].get()

        # GUI
        self.config.gui.theme = self.vars['theme'].get()
        self.config.gui.window_width = self.vars['window_width'].get()
        self.config.gui.window_height = self.vars['window_height'].get()
        self.config.gui.show_historical_tab = self.vars['show_historical'].get()
        self.config.gui.show_ml_insights = self.vars['show_ml_insights'].get()
        self.config.gui.autosave_enabled = self.vars['autosave_enabled'].get()
        self.config.gui.autosave_interval = self.vars['autosave_interval'].get()

    def _validate_settings(self) -> List[str]:
        """Validate settings and return list of errors."""
        errors = []

        # Validate paths
        if not self.data_dir_var.get():
            errors.append("Data directory cannot be empty")

        if not self.log_dir_var.get():
            errors.append("Log directory cannot be empty")

        # Validate risk thresholds
        if self.vars['high_risk_threshold'].get() <= self.vars['low_risk_threshold'].get():
            errors.append("High risk threshold must be greater than low risk threshold")

        # Validate API settings
        if self.vars['api_enabled'].get():
            if not self.vars['api_base_url'].get():
                errors.append("API base URL is required when API is enabled")

        # Validate ML settings
        if self.vars['ml_enabled'].get():
            if not self.vars['ml_model_path'].get():
                errors.append("Model directory is required when ML is enabled")

        return errors

    def _toggle_database_settings(self):
        """Enable/disable database settings based on checkbox."""
        enabled = self.vars['db_enabled'].get()
        state = 'normal' if enabled else 'disabled'

        self.db_path_entry.configure(state=state)
        self.db_browse_btn.configure(state=state)
        self.db_echo_check.configure(state=state)
        self.db_pool_spin.configure(state=state)

    def _toggle_plot_settings(self):
        """Enable/disable plot settings based on checkbox."""
        enabled = self.vars['generate_plots'].get()
        state = 'normal' if enabled else 'disabled'

        self.plot_dpi_spin.configure(state=state)

    def _toggle_cache_settings(self):
        """Enable/disable cache settings based on checkbox."""
        enabled = self.vars['cache_enabled'].get()
        state = 'normal' if enabled else 'disabled'

        self.cache_ttl_spin.configure(state=state)

    def _toggle_ml_settings(self):
        """Enable/disable ML settings based on checkbox."""
        enabled = self.vars['ml_enabled'].get()
        state = 'normal' if enabled else 'disabled'

        self.ml_path_entry.configure(state=state)
        self.ml_browse_btn.configure(state=state)
        self.failure_pred_check.configure(state=state)
        self.threshold_opt_check.configure(state=state)
        self.retrain_spin.configure(state=state)

    def _toggle_api_settings(self):
        """Enable/disable API settings based on checkbox."""
        enabled = self.vars['api_enabled'].get()
        state = 'normal' if enabled else 'disabled'

        self.api_url_entry.configure(state=state)
        self.api_key_entry.configure(state=state)
        self.api_timeout_spin.configure(state=state)
        self.api_retries_spin.configure(state=state)

    def _toggle_autosave_settings(self):
        """Enable/disable autosave settings based on checkbox."""
        enabled = self.vars['autosave_enabled'].get()
        state = 'normal' if enabled else 'disabled'

        self.autosave_spin.configure(state=state)

    def _toggle_api_key_visibility(self):
        """Toggle API key visibility."""
        if self.show_api_key.get():
            self.api_key_entry.configure(show='')
        else:
            self.api_key_entry.configure(show='*')

    def _toggle_lm_compliance(self):
        """Toggle LM compliance mode."""
        enabled = self.vars['lm_compliance_mode'].get()
        
        if enabled:
            # Show warning and disable cutoff frequency editing
            self.cutoff_spin.configure(state='disabled')
            self.lm_warning_label.configure(foreground='red')
            # Informatively show what the LM mode will use
            original_cutoff = self.vars['filter_cutoff'].get()
            self.lm_warning_label.configure(
                text=f"⚠️ LM Mode: Uses 80Hz cutoff (overrides {original_cutoff}Hz) with recursive filter"
            )
        else:
            # Enable cutoff frequency editing and reset warning
            self.cutoff_spin.configure(state='normal')
            self.lm_warning_label.configure(foreground='gray')
            self.lm_warning_label.configure(
                text="Uses Butterworth filter with configurable cutoff frequency"
            )

    def _browse_directory(self, var: tk.StringVar):
        """Browse for directory."""
        directory = filedialog.askdirectory(
            initialdir=var.get() or Path.home()
        )
        if directory:
            var.set(directory)

    def _browse_db_file(self):
        """Browse for database file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension='.db',
            filetypes=[('SQLite Database', '*.db'), ('All Files', '*.*')],
            initialfile='analyzer.db'
        )
        if file_path:
            self.vars['db_path'].set(file_path)

    def _test_db_connection(self):
        """Test database connection using proper Config-based resolution."""
        import logging

        self.db_test_label.configure(text="Testing...", foreground='orange')
        self.update()

        try:
            from laser_trim_analyzer.database.manager import DatabaseManager
            from laser_trim_analyzer.config.base import load_config

            # Load current config and temporarily update the database path
            # This ensures we use the same Config-based resolution as the rest of the app
            test_config = load_config()
            test_config.database.path = Path(self.vars['db_path'].get())
            test_config.database.enabled = self.vars['db_enabled'].get()

            # Create DatabaseManager with Config object (not string path)
            logger = logging.getLogger(__name__)
            logger.info(f"Testing database connection to: {self.vars['db_path'].get()}")

            db = DatabaseManager(test_config)
            db.init_db()

            # Log the resolved path for verification
            db_info = db.database_path_info
            if isinstance(db_info, dict):
                actual_path = db_info.get('current_path', 'Unknown')
                logger.info(f"Database manager resolved path to: {actual_path}")

            db.close()

            self.db_test_label.configure(text="✓ Connection successful", foreground='green')
        except Exception as e:
            self.db_test_label.configure(text=f"✗ Failed: {str(e)[:50]}...", foreground='red')
            logging.getLogger(__name__).error(f"Database connection test failed: {e}")

    def _test_api_connection(self):
        """Test API connection."""
        self.api_test_label.configure(text="Testing...", foreground='orange')
        self.update()

        try:
            import requests

            # Try to connect
            response = requests.get(
                self.vars['api_base_url'].get() + '/health',
                timeout=5,
                headers={'Authorization': f'Bearer {self.vars["api_key"].get()}'}
            )

            if response.ok:
                self.api_test_label.configure(text="✓ Connection successful", foreground='green')
            else:
                self.api_test_label.configure(text=f"✗ Failed: {response.status_code}", foreground='red')

        except Exception as e:
            self.api_test_label.configure(text=f"✗ Failed: {str(e)[:50]}...", foreground='red')

    def _export_config(self):
        """Export configuration to file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension='.yaml',
            filetypes=[
                ('YAML Files', '*.yaml'),
                ('JSON Files', '*.json'),
                ('All Files', '*.*')
            ]
        )

        if file_path:
            try:
                self._save_settings()

                if file_path.endswith('.json'):
                    # Export as JSON
                    config_dict = self.config.model_dump()
                    with open(file_path, 'w') as f:
                        json.dump(config_dict, f, indent=2, default=str)
                else:
                    # Export as YAML
                    self.config.to_yaml(Path(file_path))

                messagebox.showinfo("Export Successful", f"Configuration exported to:\n{file_path}")

            except Exception as e:
                messagebox.showerror("Export Failed", f"Failed to export configuration:\n{str(e)}")

    def _import_config(self):
        """Import configuration from file."""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ('YAML Files', '*.yaml'),
                ('JSON Files', '*.json'),
                ('All Files', '*.*')
            ]
        )

        if file_path:
            try:
                # Load configuration
                self.config = Config.from_yaml(Path(file_path))

                # Reload UI
                self._load_settings()

                messagebox.showinfo("Import Successful", "Configuration imported successfully")

            except Exception as e:
                messagebox.showerror("Import Failed", f"Failed to import configuration:\n{str(e)}")

    def _on_ok(self):
        """Handle OK button."""
        if self._on_apply():
            self.result = True
            self.destroy()

    def _on_cancel(self):
        """Handle Cancel button."""
        # Restore original config
        self.config.__dict__.update(self.original_config.__dict__)
        self.result = False
        self.destroy()

    def _on_apply(self) -> bool:
        """Handle Apply button."""
        # Validate settings
        errors = self._validate_settings()

        if errors:
            messagebox.showerror(
                "Validation Error",
                "Please correct the following errors:\n\n" + "\n".join(f"• {e}" for e in errors)
            )
            return False

        # Save settings
        try:
            self._save_settings()

            # Save to file
            config_path = Path.home() / ".laser_trim_analyzer" / "config.yaml"
            config_path.parent.mkdir(exist_ok=True)
            self.config.to_yaml(config_path)

            messagebox.showinfo("Settings Saved", "Settings have been saved successfully")
            return True

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save settings:\n{str(e)}")
            return False

    def _on_reset(self):
        """Handle Reset to Defaults button."""
        if messagebox.askyesno(
                "Reset Settings",
                "Are you sure you want to reset all settings to their default values?"
        ):
            # Create new default config
            self.config = Config()

            # Reload UI
            self._load_settings()

            messagebox.showinfo("Reset Complete", "Settings have been reset to defaults")

    def _deep_copy_config(self, config: Config) -> Config:
        """Create a deep copy of the configuration."""
        # Use model_dump and recreate
        config_dict = config.model_dump()
        return Config(**config_dict)

    def _center_window(self):
        """Center the window on the parent."""
        self.update_idletasks()

        # Get window size
        w = self.winfo_width()
        h = self.winfo_height()

        # Get parent position
        parent_x = self.master.winfo_x()
        parent_y = self.master.winfo_y()
        parent_w = self.master.winfo_width()
        parent_h = self.master.winfo_height()

        # Calculate position
        x = parent_x + (parent_w - w) // 2
        y = parent_y + (parent_h - h) // 2

        self.geometry(f"{w}x{h}+{x}+{y}")

    def show(self) -> bool:
        """Show the dialog and return True if OK was clicked."""
        self.wait_window()
        return self.result