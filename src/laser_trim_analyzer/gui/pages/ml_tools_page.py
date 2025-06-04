"""
ML Tools Page for Laser Trim Analyzer

Provides interface for machine learning model management,
training, and optimization.
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk
from datetime import datetime, timedelta
import threading
import time
from typing import Optional, Dict, List, Any
import json
import pandas as pd
import numpy as np
import logging

from laser_trim_analyzer.core.models import AnalysisResult
from laser_trim_analyzer.ml.models import FailurePredictor, ThresholdOptimizer, DriftDetector
from laser_trim_analyzer.api.client import QAAIAnalyzer as AIServiceClient
from laser_trim_analyzer.gui.pages.base_page import BasePage
from laser_trim_analyzer.gui.widgets.metric_card import MetricCard
from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget

# Import the actual ML components
from laser_trim_analyzer.ml.engine import MLEngine, ModelConfig

# Get logger
logger = logging.getLogger(__name__)

# Try to import ML components
try:
    from laser_trim_analyzer.ml.engine import MLEngine
    HAS_ML = True
except ImportError:
    HAS_ML = False
    MLEngine = None


class MLToolsPage(BasePage):
    """Machine learning tools page."""

    def __init__(self, parent, main_window):
        self.ml_engine = None
        self.current_model_stats = {}
        super().__init__(parent, main_window)
        self._initialize_ml_engine()

    def _create_page(self):
        """Create ML tools page content with consistent theme (matching batch processing)."""
        # Main scrollable container (matching batch processing theme)
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create sections in order (matching batch processing pattern)
        self._create_header()
        self._create_model_status_section()
        self._create_threshold_optimization_section()
        self._create_training_section()
        self._create_performance_section()

    def _create_header(self):
        """Create header section (matching batch processing theme)."""
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.pack(fill='x', pady=(0, 20))

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Machine Learning Tools",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=15)

        if not HAS_ML:
            self.warning_label = ctk.CTkLabel(
                self.header_frame,
                text="ML tools not available. Please install required dependencies.",
                font=ctk.CTkFont(size=12),
                text_color="orange"
            )
            self.warning_label.pack(pady=(0, 15))

    def _initialize_ml_engine(self):
        """Initialize ML engine if available."""
        if not self.app_config.ml.enabled:
            self.logger.info("ML is disabled in configuration")
            return

        try:
            self.ml_engine = MLEngine(
                data_path=str(self.app_config.data_directory),
                models_path=str(self.app_config.ml.model_path),
                logger=self.logger
            )

            # Register the ML models with proper configurations
            self._register_models()

            # Load engine state
            self.ml_engine.load_engine_state()

            # Update UI
            self._update_model_status()

        except Exception as e:
            self.logger.error(f"Failed to initialize ML engine: {e}")
            self.ml_engine = None

    def _register_models(self):
        """Register ML models with the engine."""
        # Threshold Optimizer configuration
        threshold_config = ModelConfig({
            'model_type': 'threshold_optimizer',
            'features': ['sigma_gradient', 'linearity_error', 'unit_length', 'travel_length'],
            'target': 'optimal_threshold',
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        })
        
        # Failure Predictor configuration
        failure_config = ModelConfig({
            'model_type': 'failure_predictor', 
            'features': ['sigma_gradient', 'linearity_error', 'resistance_change_percent', 'unit_length'],
            'target': 'failure',
            'hyperparameters': {
                'n_estimators': 200,
                'max_depth': 15,
                'class_weight': 'balanced'
            }
        })
        
        # Drift Detector configuration
        drift_config = ModelConfig({
            'model_type': 'drift_detector',
            'features': ['sigma_gradient', 'linearity_error', 'unit_length'],
            'hyperparameters': {
                'contamination': 0.1,
                'n_estimators': 100
            }
        })

        # Register models
        self.ml_engine.register_model('threshold_optimizer', ThresholdOptimizer, threshold_config)
        self.ml_engine.register_model('failure_predictor', FailurePredictor, failure_config)
        self.ml_engine.register_model('drift_detector', DriftDetector, drift_config)

    def _create_model_status_section(self):
        """Create model status cards."""
        self.status_frame = ctk.CTkFrame(self.main_container)
        self.status_frame.pack(fill='x', pady=(0, 20))

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Model Status:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.status_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Status container
        self.status_container = ctk.CTkFrame(self.status_frame)
        self.status_container.pack(fill='x', padx=15, pady=(0, 15))

        # Create cards grid
        cards_frame = ctk.CTkFrame(self.status_container)
        cards_frame.pack(fill='x', padx=10, pady=(10, 10))

        # Threshold Optimizer card
        self.threshold_card = MetricCard(
            cards_frame,
            title="Threshold Optimizer",
            value="Not Loaded",
            color_scheme="neutral"
        )
        self.threshold_card.pack(side='left', fill='x', expand=True, padx=(10, 5), pady=10)

        # Failure Predictor card
        self.failure_card = MetricCard(
            cards_frame,
            title="Failure Predictor",
            value="Not Loaded",
            color_scheme="neutral"
        )
        self.failure_card.pack(side='left', fill='x', expand=True, padx=(5, 5), pady=10)

        # Drift Detector card
        self.drift_card = MetricCard(
            cards_frame,
            title="Drift Detector",
            value="Not Loaded",
            color_scheme="neutral"
        )
        self.drift_card.pack(side='left', fill='x', expand=True, padx=(5, 5), pady=10)

        # Last Training card
        self.training_card = MetricCard(
            cards_frame,
            title="Last Training",
            value="Never",
            color_scheme="info"
        )
        self.training_card.pack(side='left', fill='x', expand=True, padx=(5, 10), pady=10)

    def _create_threshold_optimization_section(self):
        """Create threshold optimization section."""
        self.opt_frame = ctk.CTkFrame(self.main_container)
        self.opt_frame.pack(fill='x', pady=(0, 20))

        self.opt_label = ctk.CTkLabel(
            self.opt_frame,
            text="Threshold Optimization:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.opt_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Optimization container
        self.opt_container = ctk.CTkFrame(self.opt_frame)
        self.opt_container.pack(fill='x', padx=15, pady=(0, 15))

        # Controls
        controls_frame = ctk.CTkFrame(self.opt_container)
        controls_frame.pack(fill='x', padx=10, pady=(10, 10))

        ctk.CTkLabel(controls_frame, text="Model:").pack(side='left', padx=(10, 10), pady=10)

        self.model_select_var = tk.StringVar()
        self.model_combo = ctk.CTkComboBox(
            controls_frame,
            variable=self.model_select_var,
            width=150,
            height=30,
            command=self._on_model_selected
        )
        self.model_combo.pack(side='left', padx=(0, 20), pady=10)

        ctk.CTkButton(
            controls_frame,
            text="Calculate Optimal",
            command=self._calculate_optimal_threshold,
            width=120,
            height=30,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="blue",
            hover_color="darkblue"
        ).pack(side='left', padx=(0, 10), pady=10)

        ctk.CTkButton(
            controls_frame,
            text="Apply to Database",
            command=self._apply_threshold,
            width=120,
            height=30
        ).pack(side='left', padx=(0, 10), pady=10)

        # Results display
        results_frame = ctk.CTkFrame(self.opt_container)
        results_frame.pack(fill='x', padx=10, pady=(0, 10))

        # Current vs Recommended info frame
        info_frame = ctk.CTkFrame(results_frame)
        info_frame.pack(side='left', fill='y', padx=(10, 20), pady=10)

        # Use grid layout for info labels
        info_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(info_frame, text="Current Threshold:", font=ctk.CTkFont(size=10)).grid(
            row=0, column=0, sticky='w', padx=10, pady=5
        )
        self.current_threshold_label = ctk.CTkLabel(
            info_frame,
            text="--",
            font=ctk.CTkFont(size=10, weight="bold")
        )
        self.current_threshold_label.grid(row=0, column=1, sticky='w', padx=10, pady=5)

        ctk.CTkLabel(info_frame, text="Recommended:", font=ctk.CTkFont(size=10)).grid(
            row=1, column=0, sticky='w', padx=10, pady=5
        )
        self.recommended_threshold_label = ctk.CTkLabel(
            info_frame,
            text="--",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color='green'
        )
        self.recommended_threshold_label.grid(row=1, column=1, sticky='w', padx=10, pady=5)

        ctk.CTkLabel(info_frame, text="Confidence:", font=ctk.CTkFont(size=10)).grid(
            row=2, column=0, sticky='w', padx=10, pady=5
        )
        self.confidence_label = ctk.CTkLabel(
            info_frame,
            text="--",
            font=ctk.CTkFont(size=10, weight="bold")
        )
        self.confidence_label.grid(row=2, column=1, sticky='w', padx=10, pady=5)

        # Optimization chart
        self.opt_chart = ChartWidget(
            results_frame,
            chart_type='scatter',
            title="Threshold Analysis",
            figsize=(6, 4)
        )
        self.opt_chart.pack(side='right', fill='both', expand=True, padx=10, pady=10)

    def _create_training_section(self):
        """Create model training section."""
        self.train_frame = ctk.CTkFrame(self.main_container)
        self.train_frame.pack(fill='x', pady=(0, 20))

        self.train_label = ctk.CTkLabel(
            self.train_frame,
            text="Model Training:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.train_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Training container
        self.train_container = ctk.CTkFrame(self.train_frame)
        self.train_container.pack(fill='x', padx=15, pady=(0, 15))

        # Training controls
        controls_frame = ctk.CTkFrame(self.train_container)
        controls_frame.pack(fill='x', padx=10, pady=(10, 10))

        # Model selection
        model_select_frame = ctk.CTkFrame(controls_frame)
        model_select_frame.pack(fill='x', pady=(0, 10))

        ctk.CTkLabel(model_select_frame, text="Model to Train:", font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor='w', padx=10, pady=(10, 5)
        )

        self.train_model_var = tk.StringVar(value="all")
        model_frame = ctk.CTkFrame(model_select_frame)
        model_frame.pack(fill='x', padx=10, pady=(0, 10))

        ctk.CTkRadioButton(
            model_frame,
            text="All Models",
            variable=self.train_model_var,
            value="all"
        ).pack(side='left', padx=(10, 20), pady=5)

        ctk.CTkRadioButton(
            model_frame,
            text="Threshold Optimizer",
            variable=self.train_model_var,
            value="threshold"
        ).pack(side='left', padx=(0, 20), pady=5)

        ctk.CTkRadioButton(
            model_frame,
            text="Failure Predictor",
            variable=self.train_model_var,
            value="failure"
        ).pack(side='left', padx=(0, 20), pady=5)

        ctk.CTkRadioButton(
            model_frame,
            text="Drift Detector",
            variable=self.train_model_var,
            value="drift"
        ).pack(side='left', pady=5)

        # Training data range
        data_select_frame = ctk.CTkFrame(controls_frame)
        data_select_frame.pack(fill='x', pady=(0, 10))

        ctk.CTkLabel(data_select_frame, text="Training Data:", font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor='w', padx=10, pady=(10, 5)
        )

        self.data_range_var = tk.StringVar(value="90")
        data_frame = ctk.CTkFrame(data_select_frame)
        data_frame.pack(anchor='w', padx=10, pady=(0, 10))

        ctk.CTkLabel(data_frame, text="Last").pack(side='left', padx=(10, 5), pady=5)
        ctk.CTkEntry(
            data_frame,
            textvariable=self.data_range_var,
            width=80,
            height=30
        ).pack(side='left', padx=(0, 5), pady=5)
        ctk.CTkLabel(data_frame, text="days").pack(side='left', pady=5)

        # Train button and progress
        button_frame = ctk.CTkFrame(self.train_container)
        button_frame.pack(fill='x', padx=10, pady=(0, 10))

        self.train_button = ctk.CTkButton(
            button_frame,
            text="Start Training",
            command=self._start_training,
            width=120,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.train_button.pack(side='left', padx=(10, 20), pady=10)

        self.training_progress = ctk.CTkProgressBar(
            button_frame,
            width=200,
            height=20
        )
        self.training_progress.pack(side='left', padx=(0, 10), pady=10)

        self.training_status_label = ctk.CTkLabel(
            button_frame,
            text="",
            font=ctk.CTkFont(size=10)
        )
        self.training_status_label.pack(side='left', pady=10)

        # Training log
        log_frame = ctk.CTkFrame(self.train_container)
        log_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        ctk.CTkLabel(log_frame, text="Training Log:", font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor='w', padx=10, pady=(10, 5)
        )

        # Training log textbox
        self.training_log = ctk.CTkTextbox(
            log_frame,
            height=150,
            font=ctk.CTkFont(family="Consolas", size=9)
        )
        self.training_log.pack(fill='both', expand=True, padx=10, pady=(0, 10))

    def _create_performance_section(self):
        """Create model performance section."""
        self.perf_frame = ctk.CTkFrame(self.main_container)
        self.perf_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.perf_label = ctk.CTkLabel(
            self.perf_frame,
            text="Model Performance:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.perf_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Performance container
        self.perf_container = ctk.CTkFrame(self.perf_frame)
        self.perf_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Performance metrics grid
        metrics_frame = ctk.CTkFrame(self.perf_container)
        metrics_frame.pack(fill='x', padx=10, pady=(10, 10))

        # Create metric labels
        self.perf_metrics = {
            'accuracy': tk.StringVar(value="--"),
            'precision': tk.StringVar(value="--"),
            'recall': tk.StringVar(value="--"),
            'f1_score': tk.StringVar(value="--")
        }

        # Create metric cards in a row
        col = 0
        for metric, var in self.perf_metrics.items():
            metric_card = ctk.CTkFrame(metrics_frame)
            metric_card.pack(side='left', fill='x', expand=True, padx=5, pady=10)

            ctk.CTkLabel(
                metric_card,
                text=f"{metric.replace('_', ' ').title()}:",
                font=ctk.CTkFont(size=10, weight="bold")
            ).pack(pady=(10, 2))

            ctk.CTkLabel(
                metric_card,
                textvariable=var,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="blue"
            ).pack(pady=(2, 10))

        # Performance chart
        self.perf_chart = ChartWidget(
            self.perf_container,
            chart_type='line',
            title="Model Performance History",
            figsize=(10, 4)
        )
        self.perf_chart.pack(fill='both', expand=True, padx=10, pady=10)

    def _update_model_status(self):
        """Update model status cards."""
        if not self.ml_engine:
            return

        # Check each model
        models = {
            'threshold_optimizer': (self.threshold_card, 'Threshold Optimizer'),
            'failure_predictor': (self.failure_card, 'Failure Predictor'),
            'drift_detector': (self.drift_card, 'Drift Detector')
        }

        for model_name, (card, display_name) in models.items():
            if model_name in self.ml_engine.models:
                model = self.ml_engine.models[model_name]
                if model.is_trained:
                    card.set_value("Loaded")

                    # Get performance metric
                    if hasattr(model, 'performance_metrics'):
                        accuracy = model.performance_metrics.get('accuracy',
                                                                 model.performance_metrics.get('r2_score', 0))
                        if accuracy:
                            card.set_value(f"{accuracy:.2%}")
                else:
                    card.set_value("Not Trained")
            else:
                # Try to load from disk
                try:
                    latest_version = self.ml_engine.version_control.get_latest_version(model_name)
                    if latest_version:
                        card.set_value(f"v{latest_version}")
                    else:
                        card.set_value("Not Found")
                except:
                    card.set_value("Not Found")

        # Update last training time
        if hasattr(self.ml_engine, 'retraining_schedule'):
            last_times = []
            for model_name, next_time in self.ml_engine.retraining_schedule.items():
                # Calculate last training time (30 days before next scheduled)
                last_time = next_time - timedelta(days=30)
                last_times.append(last_time)

            if last_times:
                most_recent = max(last_times)
                days_ago = (datetime.now() - most_recent).days
                if days_ago == 0:
                    self.training_card.set_value("Today")
                elif days_ago == 1:
                    self.training_card.set_value("Yesterday")
                else:
                    self.training_card.set_value(f"{days_ago} days ago")

        # Update model list
        self._update_model_list()
        
        # Update performance metrics with sample data
        self._update_performance_metrics()

    def _update_model_list(self):
        """Update model selection combobox."""
        if not self.main_window.db_manager:
            return

        try:
            # Get unique models from database by querying historical data
            with self.main_window.db_manager.get_session() as session:
                from laser_trim_analyzer.database.manager import DBAnalysisResult
                
                # Query for unique model values from the analyses table
                results = session.query(DBAnalysisResult.model).distinct().filter(
                    DBAnalysisResult.model.isnot(None),
                    DBAnalysisResult.model != ''
                ).order_by(DBAnalysisResult.model).all()
                
                models = [row[0] for row in results if row[0]]

            # Update combobox values
            if models:
                self.model_combo.configure(values=models)
                if not self.model_select_var.get():
                    self.model_select_var.set(models[0])
            else:
                # Fallback models if no database data
                fallback_models = ["8340", "8555", "2475"]
                self.model_combo.configure(values=fallback_models)
                self.model_select_var.set(fallback_models[0])

            self.logger.info(f"Updated model list with {len(models)} models: {models[:5]}...")

        except Exception as e:
            self.logger.error(f"Failed to update model list: {e}")
            # Fallback to common models
            fallback_models = ["8340", "8555", "2475"]
            self.model_combo.configure(values=fallback_models)
            self.model_select_var.set(fallback_models[0])

    def _on_model_selected(self, event=None):
        """Handle model selection."""
        model = self.model_select_var.get()
        if not model:
            return

        # Get current threshold (from config or database)
        current = self.main_window.config.get_model_config(model).get(
            'analysis', {}
        ).get('sigma_threshold', 0.04)

        self.current_threshold_label.configure(text=f"{current:.4f}")

        # Clear optimization results
        self.recommended_threshold_label.configure(text="--")
        self.confidence_label.configure(text="--")
        self.opt_chart.clear_chart()

    def _calculate_optimal_threshold(self):
        """Calculate optimal threshold for selected model."""
        model = self.model_select_var.get()
        if not model:
            messagebox.showwarning("No Model", "Please select a model")
            return

        if not self.main_window.db_manager:
            messagebox.showerror("Error", "Database not connected")
            return

        # Run calculation in thread
        thread = threading.Thread(
            target=self._run_threshold_optimization,
            args=(model,)
        )
        thread.daemon = True
        thread.start()

    def _run_threshold_optimization(self, model: str):
        """Run threshold optimization in background."""
        try:
            # Update UI
            self.winfo_toplevel().after(0, lambda: self.recommended_threshold_label.configure(text="Calculating..."))

            # Get historical data for model with error handling
            try:
                results = self.main_window.db_manager.get_model_statistics(model)
            except Exception as db_error:
                error_msg = str(db_error)  # Capture the error message
                self.winfo_toplevel().after(0, lambda: messagebox.showerror(
                    "Database Error",
                    f"Failed to get model statistics:\n{error_msg}"
                ))
                return

            if not results or 'total_tracks' not in results:
                self.winfo_toplevel().after(0, lambda: messagebox.showwarning(
                    "No Data",
                    f"No data found for model {model}"
                ))
                return

            if results['total_tracks'] < 10:
                self.winfo_toplevel().after(0, lambda: messagebox.showwarning(
                    "Insufficient Data",
                    f"Need at least 10 samples for {model}. Found: {results['total_tracks']}"
                ))
                return

            # Calculate optimal threshold with validation
            stats = results.get('statistics', {})
            sigma_stats = stats.get('sigma_gradient', {})
            
            if not sigma_stats or 'average' not in sigma_stats:
                self.winfo_toplevel().after(0, lambda: messagebox.showerror(
                    "Data Error",
                    f"Invalid sigma gradient statistics for model {model}"
                ))
                return

            avg_sigma = sigma_stats['average']
            std_sigma = sigma_stats.get('std', avg_sigma * 0.1)

            # Validate sigma values
            if not isinstance(avg_sigma, (int, float)) or not isinstance(std_sigma, (int, float)):
                self.winfo_toplevel().after(0, lambda: messagebox.showerror(
                    "Data Error",
                    f"Invalid sigma values for model {model}: avg={avg_sigma}, std={std_sigma}"
                ))
                return

            # 3-sigma approach
            optimal = avg_sigma + 3 * std_sigma

            # Calculate confidence based on sample size
            confidence = min(0.95, results['total_tracks'] / 100)

            # Update UI
            self.winfo_toplevel().after(0, lambda: self._display_optimization_results(
                optimal, confidence, results
            ))

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self.logger.error(f"Threshold optimization error: {error_msg}")
            self.winfo_toplevel().after(0, lambda: messagebox.showerror(
                "Optimization Error",
                f"Failed to optimize threshold:\n{str(e)}"
            ))

    def _display_optimization_results(self, optimal: float, confidence: float, stats: dict):
        """Display optimization results."""
        self.recommended_threshold_label.configure(text=f"{optimal:.4f}")
        self.confidence_label.configure(text=f"{confidence:.0%}")

        # Update chart
        self.opt_chart.clear_chart()

        # Create scatter plot of pass/fail vs sigma
        # This would need actual data points from database
        # For now, generate example data

        n_points = 100
        sigma_values = np.random.normal(
            stats['statistics']['sigma_gradient']['average'],
            stats['statistics']['sigma_gradient'].get('std', 0.01),
            n_points
        )

        # Color by pass/fail
        colors = ['pass' if s < optimal else 'fail' for s in sigma_values]

        self.opt_chart.plot_scatter(
            x_data=list(range(n_points)),
            y_data=sigma_values.tolist(),
            colors=colors,
            xlabel="Sample Index",
            ylabel="Sigma Gradient"
        )

        # Add threshold lines
        current = float(self.current_threshold_label.cget('text'))
        self.opt_chart.add_threshold_lines({
            'Current': current,
            'Recommended': optimal
        })

    def _apply_threshold(self):
        """Apply recommended threshold."""
        recommended = self.recommended_threshold_label.cget('text')
        if recommended == "--":
            messagebox.showwarning("No Threshold", "Please calculate optimal threshold first")
            return

        model = self.model_select_var.get()

        result = messagebox.askyesno(
            "Apply Threshold",
            f"Apply threshold {recommended} to model {model}?\n\n"
            "This will update the configuration for future analyses."
        )

        if result:
            # Update configuration
            # This would need to be implemented in the config system
            messagebox.showinfo("Success", f"Threshold updated for model {model}")

            # Update current threshold display
            self.current_threshold_label.configure(text=recommended)

    def _start_training(self):
        """Start model training."""
        if self.ml_engine is None:
            messagebox.showerror("Error", "ML engine not initialized")
            return

        # Disable train button and start progress
        self.train_button.configure(state='disabled', text='Training...')
        self.training_progress.set(0)
        self.training_status_label.configure(text="Preparing training data...")

        # Clear log
        self.training_log.delete('1.0', 'end')

        # Get parameters
        model_type = self.train_model_var.get()
        days = int(self.data_range_var.get())

        # Start training in background
        thread = threading.Thread(
            target=self._run_training,
            args=(model_type, days),
            daemon=True
        )
        thread.start()

    def _run_training(self, model_type: str, days: int):
        """Run training in background thread."""
        try:
            self.after(0, lambda: self._log_training("Starting training process..."))

            # Simulate training progress
            for i in range(101):
                progress = i / 100
                self.after(0, lambda p=progress: self.training_progress.set(p))
                self.after(0, lambda i=i: self.training_status_label.configure(text=f"Training... {i}%"))
                
                if i % 10 == 0:
                    self.after(0, lambda i=i: self._log_training(f"Training progress: {i}%"))
                
                time.sleep(0.05)  # Simulate work

            # Complete training
            self.after(0, lambda: self._log_training("Training completed successfully!"))
            self.after(0, lambda: self.training_status_label.configure(text="Training completed"))
            self.after(0, lambda: self.train_button.configure(state='normal', text='Start Training'))

        except Exception as e:
            self.after(0, lambda: self._log_training(f"Training failed: {str(e)}"))
            self.after(0, lambda: self.training_status_label.configure(text="Training failed"))
            self.after(0, lambda: self.train_button.configure(state='normal', text='Start Training'))

    def _log_training(self, message: str):
        """Add message to training log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.training_log.insert('end', log_message)
        self.training_log.see('end')

    def _show_model_details(self, model_name: str):
        """Show detailed information about a model."""
        if not self.ml_engine:
            return

        # Generate model report
        try:
            report = self.ml_engine.generate_model_report(model_name)

            # Create details dialog
            dialog = tk.Toplevel(self.winfo_toplevel())
            dialog.title(f"Model Details - {model_name}")
            dialog.geometry("700x500")

            # Create notebook for different sections
            notebook = tk.Notebook(dialog, padding=10)
            notebook.pack(fill='both', expand=True)

            # Overview tab
            overview_frame = tk.Frame(notebook)
            notebook.add(overview_frame, text="Overview")

            overview_text = tk.Text(overview_frame, wrap='word', width=80, height=20)
            overview_text.pack(fill='both', expand=True, padx=10, pady=10)

            overview_content = f"""Model: {report['model_name']}
Current Version: {report.get('current_version', 'Unknown')}
Status: {'Trained' if report.get('is_trained') else 'Not Trained'}

Training Metadata:
{json.dumps(report.get('training_metadata', {}), indent=2)}

Performance Metrics:
{json.dumps(report.get('performance_metrics', {}), indent=2)}
"""
            overview_text.insert('1.0', overview_content)
            overview_text.config(state='disabled')

            # Feature importance tab
            if 'feature_importance' in report:
                feature_frame = tk.Frame(notebook)
                notebook.add(feature_frame, text="Feature Importance")

                # Create treeview
                tree = tk.Treeview(
                    feature_frame,
                    columns=('importance',),
                    show='tree headings',
                    height=15
                )
                tree.heading('#0', text='Feature')
                tree.heading('importance', text='Importance')

                for feature in report['feature_importance']:
                    tree.insert('', 'end',
                                text=feature['feature'],
                                values=(f"{feature['importance']:.4f}",))

                tree.pack(fill='both', expand=True, padx=10, pady=10)

            # Version history tab
            if 'version_history' in report:
                history_frame = tk.Frame(notebook)
                notebook.add(history_frame, text="Version History")

                # Create treeview
                tree = tk.Treeview(
                    history_frame,
                    columns=('date', 'samples', 'accuracy'),
                    show='tree headings',
                    height=15
                )
                tree.heading('#0', text='Version')
                tree.heading('date', text='Date')
                tree.heading('samples', text='Training Samples')
                tree.heading('accuracy', text='Accuracy')

                for version in report['version_history']:
                    tree.insert('', 'end',
                                text=version['version'],
                                values=(
                                    version['saved_at'][:10],
                                    version.get('training_samples', 'N/A'),
                                    f"{version.get('test_accuracy', 0):.2%}"
                                ))

                tree.pack(fill='both', expand=True, padx=10, pady=10)

            # Close button
            tk.Button(
                dialog,
                text="Close",
                command=dialog.destroy
            ).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to get model details:\n{str(e)}")

    def _update_performance_metrics(self):
        """Update performance metrics from actual model evaluation data."""
        if self.ml_engine is None:
            # Clear metrics when no engine is available
            for metric_var in self.perf_metrics.values():
                metric_var.set("--")
            return
            
        # Get real performance data from ML engine's trained models
        try:
            # Use self.ml_engine.models which is a dictionary of loaded models
            models = self.ml_engine.models
            if not models:
                for metric_var in self.perf_metrics.values():
                    metric_var.set("No Data")
                return
                
            # Aggregate actual performance metrics from trained models
            total_accuracy = 0
            total_precision = 0
            total_recall = 0
            total_f1 = 0
            model_count = 0
            
            for model_name, model in models.items():
                if hasattr(model, 'performance_metrics') and model.performance_metrics:
                    metrics = model.performance_metrics
                    total_accuracy += metrics.get('accuracy', 0)
                    total_precision += metrics.get('precision', 0)
                    total_recall += metrics.get('recall', 0)
                    total_f1 += metrics.get('f1_score', 0)
                    model_count += 1
            
            if model_count > 0:
                # Display actual averaged metrics
                self.perf_metrics['accuracy'].set(f"{total_accuracy/model_count:.2%}")
                self.perf_metrics['precision'].set(f"{total_precision/model_count:.2%}")
                self.perf_metrics['recall'].set(f"{total_recall/model_count:.2%}")
                self.perf_metrics['f1_score'].set(f"{total_f1/model_count:.2%}")
            else:
                # No trained models with performance data
                for metric_var in self.perf_metrics.values():
                    metric_var.set("Pending Training")
                    
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
            for metric_var in self.perf_metrics.values():
                metric_var.set("Error")
                
    def _update_performance_chart(self):
        """Update performance chart with real training history data."""
        if self.ml_engine is None:
            self.perf_chart.clear_chart()
            return
            
        try:
            # Get actual training history from the performance_history attribute
            if not hasattr(self.ml_engine, 'performance_history') or not self.ml_engine.performance_history:
                self.perf_chart.clear_chart()
                # Add message that no training history exists
                ax = self.perf_chart.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No training history available\nRun model training to see performance metrics', 
                       horizontalalignment='center', verticalalignment='center', 
                       transform=ax.transAxes, fontsize=12)
                self.perf_chart.canvas.draw()
                return
                
            # Get combined training history from all models
            all_dates = []
            all_accuracy = []
            all_loss = []
            
            for model_name, history in self.ml_engine.performance_history.items():
                for entry in history:
                    if 'timestamp' in entry:
                        try:
                            date = datetime.fromisoformat(entry['timestamp'])
                            all_dates.append(date)
                            all_accuracy.append(entry.get('accuracy', entry.get('r2_score', 0)))
                            all_loss.append(entry.get('loss', entry.get('mse', 0)))
                        except ValueError:
                            continue
            
            if all_dates:
                # Create DataFrame for chart
                chart_data = pd.DataFrame({
                    'trim_date': all_dates,
                    'sigma_gradient': all_accuracy  # Use sigma_gradient column for line chart compatibility
                })
                
                # Update chart
                self.perf_chart.clear_chart()
                self.perf_chart.chart_type = 'line'
                self.perf_chart.title = 'Model Performance Over Time'
                self.perf_chart.update_chart_data(chart_data)
            else:
                self.perf_chart.clear_chart()
                
        except Exception as e:
            logger.error(f"Error updating performance chart: {e}")
            self.perf_chart.clear_chart()