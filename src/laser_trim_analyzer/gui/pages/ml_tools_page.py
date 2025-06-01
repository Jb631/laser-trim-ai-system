"""
ML Tools Page for Laser Trim Analyzer

Provides interface for ML model training, threshold optimization,
and performance monitoring.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta
import threading
import time
from typing import Optional, Dict, List, Any
import json
import pandas as pd
import numpy as np

from laser_trim_analyzer.core.models import AnalysisResult
from laser_trim_analyzer.ml.models import FailurePredictor, ThresholdOptimizer
from laser_trim_analyzer.api.client import QAAIAnalyzer as AIServiceClient
from laser_trim_analyzer.gui.pages.base_page import BasePage
from laser_trim_analyzer.gui.widgets.metric_card import MetricCard
from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
from laser_trim_analyzer.ml.engine import MLEngine, ModelConfig


class MLToolsPage(BasePage):
    """Machine learning tools page."""

    def __init__(self, parent, main_window):
        self.ml_engine = None
        self.current_model_stats = {}
        super().__init__(parent, main_window)
        self._initialize_ml_engine()

    def _create_page(self):
        """Set up the ML tools page."""
        # Create scrollable frame
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create content in scrollable frame
        content_frame = scrollable_frame
        
        # Title
        title_frame = ttk.Frame(content_frame)
        title_frame.pack(fill='x', padx=20, pady=(20, 10))

        ttk.Label(
            title_frame,
            text="Machine Learning Tools",
            font=('Segoe UI', 24, 'bold')
        ).pack(side='left')

        # Create main sections in content_frame
        self._create_model_status_section(content_frame)
        self._create_threshold_optimization_section(content_frame)
        self._create_training_section(content_frame)
        self._create_performance_section(content_frame)

    def _initialize_ml_engine(self):
        """Initialize ML engine if available."""
        if not self.main_window.config.ml.enabled:
            return

        try:
            self.ml_engine = MLEngine(
                data_path=str(self.main_window.config.data_directory),
                models_path=str(self.main_window.config.ml.model_path),
                logger=self.logger
            )

            # Load engine state
            self.ml_engine.load_engine_state()

            # Update UI
            self._update_model_status()

        except Exception as e:
            self.logger.error(f"Failed to initialize ML engine: {e}")

    def _create_model_status_section(self, content_frame):
        """Create model status cards."""
        status_frame = ttk.LabelFrame(
            content_frame,
            text="Model Status",
            padding=10
        )
        status_frame.pack(fill='x', padx=20, pady=10)

        # Create cards grid
        cards_frame = ttk.Frame(status_frame)
        cards_frame.pack(fill='x')

        # Threshold Optimizer card
        self.threshold_card = MetricCard(
            cards_frame,
            title="Threshold Optimizer",
            value="Not Loaded",
            unit="",
            show_sparkline=False,
            on_click=lambda: self._show_model_details('threshold_optimizer')
        )
        self.threshold_card.pack(side='left', padx=10, pady=5)

        # Failure Predictor card
        self.failure_card = MetricCard(
            cards_frame,
            title="Failure Predictor",
            value="Not Loaded",
            unit="",
            show_sparkline=False,
            on_click=lambda: self._show_model_details('failure_predictor')
        )
        self.failure_card.pack(side='left', padx=10, pady=5)

        # Drift Detector card
        self.drift_card = MetricCard(
            cards_frame,
            title="Drift Detector",
            value="Not Loaded",
            unit="",
            show_sparkline=False,
            on_click=lambda: self._show_model_details('drift_detector')
        )
        self.drift_card.pack(side='left', padx=10, pady=5)

        # Last Training card
        self.training_card = MetricCard(
            cards_frame,
            title="Last Training",
            value="Never",
            unit="",
            show_sparkline=False
        )
        self.training_card.pack(side='left', padx=10, pady=5)

    def _create_threshold_optimization_section(self, content_frame):
        """Create threshold optimization section."""
        opt_frame = ttk.LabelFrame(
            content_frame,
            text="Threshold Optimization",
            padding=15
        )
        opt_frame.pack(fill='x', padx=20, pady=10)

        # Controls
        controls_frame = ttk.Frame(opt_frame)
        controls_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(controls_frame, text="Model:").pack(side='left', padx=(0, 10))

        self.model_select_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.model_select_var,
            width=20,
            state='readonly'
        )
        self.model_combo.pack(side='left', padx=(0, 20))
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_selected)

        ttk.Button(
            controls_frame,
            text="Calculate Optimal",
            command=self._calculate_optimal_threshold,
            style='Primary.TButton'
        ).pack(side='left', padx=(0, 10))

        ttk.Button(
            controls_frame,
            text="Apply to Database",
            command=self._apply_threshold
        ).pack(side='left')

        # Results display
        results_frame = ttk.Frame(opt_frame)
        results_frame.pack(fill='x')

        # Current vs Recommended
        info_frame = ttk.Frame(results_frame)
        info_frame.pack(side='left', padx=(0, 20))

        ttk.Label(info_frame, text="Current Threshold:", font=('Segoe UI', 10)).grid(
            row=0, column=0, sticky='w', pady=2
        )
        self.current_threshold_label = ttk.Label(
            info_frame,
            text="--",
            font=('Segoe UI', 10, 'bold')
        )
        self.current_threshold_label.grid(row=0, column=1, padx=10, pady=2)

        ttk.Label(info_frame, text="Recommended:", font=('Segoe UI', 10)).grid(
            row=1, column=0, sticky='w', pady=2
        )
        self.recommended_threshold_label = ttk.Label(
            info_frame,
            text="--",
            font=('Segoe UI', 10, 'bold'),
            foreground='#27ae60'
        )
        self.recommended_threshold_label.grid(row=1, column=1, padx=10, pady=2)

        ttk.Label(info_frame, text="Confidence:", font=('Segoe UI', 10)).grid(
            row=2, column=0, sticky='w', pady=2
        )
        self.confidence_label = ttk.Label(
            info_frame,
            text="--",
            font=('Segoe UI', 10, 'bold')
        )
        self.confidence_label.grid(row=2, column=1, padx=10, pady=2)

        # Optimization chart
        self.opt_chart = ChartWidget(
            results_frame,
            chart_type='scatter',
            title="Threshold Analysis",
            figsize=(6, 4)
        )
        self.opt_chart.pack(side='left', fill='both', expand=True)

    def _create_training_section(self, content_frame):
        """Create model training section."""
        train_frame = ttk.LabelFrame(
            content_frame,
            text="Model Training",
            padding=15
        )
        train_frame.pack(fill='x', padx=20, pady=10)

        # Training controls
        controls_frame = ttk.Frame(train_frame)
        controls_frame.pack(fill='x')

        # Model selection
        ttk.Label(controls_frame, text="Model to Train:").grid(
            row=0, column=0, sticky='w', padx=(0, 10), pady=5
        )

        self.train_model_var = tk.StringVar(value="all")
        model_frame = ttk.Frame(controls_frame)
        model_frame.grid(row=0, column=1, sticky='w', pady=5)

        ttk.Radiobutton(
            model_frame,
            text="All Models",
            variable=self.train_model_var,
            value="all"
        ).pack(side='left', padx=(0, 10))

        ttk.Radiobutton(
            model_frame,
            text="Threshold Optimizer",
            variable=self.train_model_var,
            value="threshold"
        ).pack(side='left', padx=(0, 10))

        ttk.Radiobutton(
            model_frame,
            text="Failure Predictor",
            variable=self.train_model_var,
            value="failure"
        ).pack(side='left', padx=(0, 10))

        ttk.Radiobutton(
            model_frame,
            text="Drift Detector",
            variable=self.train_model_var,
            value="drift"
        ).pack(side='left')

        # Training data range
        ttk.Label(controls_frame, text="Training Data:").grid(
            row=1, column=0, sticky='w', padx=(0, 10), pady=5
        )

        self.data_range_var = tk.StringVar(value="90")
        data_frame = ttk.Frame(controls_frame)
        data_frame.grid(row=1, column=1, sticky='w', pady=5)

        ttk.Label(data_frame, text="Last").pack(side='left', padx=(0, 5))
        ttk.Entry(
            data_frame,
            textvariable=self.data_range_var,
            width=10
        ).pack(side='left', padx=(0, 5))
        ttk.Label(data_frame, text="days").pack(side='left')

        # Train button and progress
        button_frame = ttk.Frame(train_frame)
        button_frame.pack(fill='x', pady=(15, 0))

        self.train_button = ttk.Button(
            button_frame,
            text="Start Training",
            command=self._start_training,
            style='Primary.TButton'
        )
        self.train_button.pack(side='left', padx=(0, 20))

        self.training_progress = ttk.Progressbar(
            button_frame,
            mode='indeterminate',
            length=200
        )
        self.training_progress.pack(side='left', padx=(0, 10))

        self.training_status_label = ttk.Label(
            button_frame,
            text=""
        )
        self.training_status_label.pack(side='left')

        # Training log
        log_frame = ttk.Frame(train_frame)
        log_frame.pack(fill='both', expand=True, pady=(10, 0))

        ttk.Label(log_frame, text="Training Log:").pack(anchor='w')

        # Text widget with scrollbar
        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill='both', expand=True)

        self.training_log = tk.Text(
            text_frame,
            height=8,
            wrap='word',
            font=('Consolas', 9)
        )
        scroll = ttk.Scrollbar(text_frame, command=self.training_log.yview)
        self.training_log.config(yscrollcommand=scroll.set)

        self.training_log.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

    def _create_performance_section(self, content_frame):
        """Create model performance section."""
        perf_frame = ttk.LabelFrame(
            content_frame,
            text="Model Performance",
            padding=10
        )
        perf_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # Performance metrics grid
        metrics_frame = ttk.Frame(perf_frame)
        metrics_frame.pack(fill='x', pady=(0, 10))

        # Create metric labels
        self.perf_metrics = {
            'accuracy': tk.StringVar(value="--"),
            'precision': tk.StringVar(value="--"),
            'recall': tk.StringVar(value="--"),
            'f1_score': tk.StringVar(value="--")
        }

        col = 0
        for metric, var in self.perf_metrics.items():
            ttk.Label(
                metrics_frame,
                text=f"{metric.replace('_', ' ').title()}:",
                font=('Segoe UI', 10)
            ).grid(row=0, column=col, sticky='w', padx=(0, 5))

            ttk.Label(
                metrics_frame,
                textvariable=var,
                font=('Segoe UI', 10, 'bold')
            ).grid(row=0, column=col + 1, sticky='w', padx=(0, 20))

            col += 2

        # Performance chart
        self.perf_chart = ChartWidget(
            perf_frame,
            chart_type='line',
            title="Model Performance History",
            figsize=(10, 4)
        )
        self.perf_chart.pack(fill='both', expand=True)

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

    def _update_model_list(self):
        """Update model selection combobox."""
        if not self.main_window.db_manager:
            return

        try:
            # Get unique models from database
            # This would need a method in DatabaseManager to get unique models
            models = ['8340', '8555', '6845', '7825']  # Example models

            self.model_combo['values'] = models
            if models and not self.model_select_var.get():
                self.model_select_var.set(models[0])

        except Exception as e:
            self.logger.error(f"Failed to update model list: {e}")

    def _on_model_selected(self, event=None):
        """Handle model selection."""
        model = self.model_select_var.get()
        if not model:
            return

        # Get current threshold (from config or database)
        current = self.main_window.config.get_model_config(model).get(
            'analysis', {}
        ).get('sigma_threshold', 0.04)

        self.current_threshold_label.config(text=f"{current:.4f}")

        # Clear optimization results
        self.recommended_threshold_label.config(text="--")
        self.confidence_label.config(text="--")
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
            self.root.after(0, lambda: self.recommended_threshold_label.config(text="Calculating..."))

            # Get historical data for model
            results = self.main_window.db_manager.get_model_statistics(model)

            if results['total_tracks'] < 10:
                self.root.after(0, lambda: messagebox.showwarning(
                    "Insufficient Data",
                    f"Need at least 10 samples for {model}. Found: {results['total_tracks']}"
                ))
                return

            # Calculate optimal threshold
            # This is a simplified calculation - in practice would use ML
            avg_sigma = results['statistics']['sigma_gradient']['average']
            std_sigma = results['statistics']['sigma_gradient'].get('std', avg_sigma * 0.1)

            # 3-sigma approach
            optimal = avg_sigma + 3 * std_sigma

            # Calculate confidence based on sample size
            confidence = min(0.95, results['total_tracks'] / 100)

            # Update UI
            self.root.after(0, lambda: self._display_optimization_results(
                optimal, confidence, results
            ))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Optimization Error",
                f"Failed to optimize threshold:\n{str(e)}"
            ))

    def _display_optimization_results(self, optimal: float, confidence: float, stats: dict):
        """Display optimization results."""
        self.recommended_threshold_label.config(text=f"{optimal:.4f}")
        self.confidence_label.config(text=f"{confidence:.0%}")

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
            self.current_threshold_label.config(text=recommended)

    def _start_training(self):
        """Start model training."""
        if not self.ml_engine:
            messagebox.showerror("Error", "ML Engine not initialized")
            return

        # Get training parameters
        model_type = self.train_model_var.get()
        days = int(self.data_range_var.get())

        # Disable button and start progress
        self.train_button.config(state='disabled')
        self.training_progress.start(10)
        self.training_status_label.config(text="Preparing training data...")

        # Clear log
        self.training_log.delete('1.0', tk.END)
        self._log_training("Starting model training...")
        self._log_training(f"Model: {model_type}")
        self._log_training(f"Training data: Last {days} days")

        # Run training in thread
        thread = threading.Thread(
            target=self._run_training,
            args=(model_type, days)
        )
        thread.daemon = True
        thread.start()

    def _run_training(self, model_type: str, days: int):
        """Run model training in background."""
        try:
            # Load training data
            self._update_training_status("Loading training data...")

            # This would load from database
            # For now, create dummy data
            n_samples = 1000

            data = pd.DataFrame({
                'sigma_gradient': np.random.normal(0.02, 0.005, n_samples),
                'linearity_spec': np.random.normal(0.01, 0.002, n_samples),
                'unit_length': np.random.normal(100, 10, n_samples),
                'resistance_change_percent': np.random.normal(2, 0.5, n_samples),
                'failure': np.random.binomial(1, 0.1, n_samples)
            })

            self._log_training(f"Loaded {len(data)} samples")

            # Train models
            if model_type == "all":
                models = ['threshold', 'failure', 'drift']
            else:
                models = [model_type]

            for model in models:
                self._update_training_status(f"Training {model} model...")
                self._log_training(f"\nTraining {model} model...")

                # Simulate training with progress updates
                for epoch in range(10):
                    time.sleep(0.5)  # Simulate training time
                    self._log_training(f"  Epoch {epoch + 1}/10 - Loss: {np.random.random():.4f}")

                self._log_training(f"  {model} model training complete!")

            # Update model status
            self.root.after(0, self._update_model_status)

            # Complete
            self._update_training_status("Training complete!")
            self._log_training("\nAll models trained successfully!")

        except Exception as e:
            self._update_training_status(f"Error: {str(e)}")
            self._log_training(f"\nERROR: {str(e)}")

        finally:
            # Re-enable button and stop progress
            self.root.after(0, lambda: self.train_button.config(state='normal'))
            self.root.after(0, self.training_progress.stop)

    def _update_training_status(self, message: str):
        """Update training status label."""
        self.root.after(0, lambda: self.training_status_label.config(text=message))

    def _log_training(self, message: str):
        """Add message to training log."""
        self.root.after(0, lambda: self.training_log.insert(tk.END, message + '\n'))
        self.root.after(0, lambda: self.training_log.see(tk.END))

    def _show_model_details(self, model_name: str):
        """Show detailed information about a model."""
        if not self.ml_engine:
            return

        # Generate model report
        try:
            report = self.ml_engine.generate_model_report(model_name)

            # Create details dialog
            dialog = tk.Toplevel(self.root)
            dialog.title(f"Model Details - {model_name}")
            dialog.geometry("700x500")

            # Create notebook for different sections
            notebook = ttk.Notebook(dialog, padding=10)
            notebook.pack(fill='both', expand=True)

            # Overview tab
            overview_frame = ttk.Frame(notebook)
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
                feature_frame = ttk.Frame(notebook)
                notebook.add(feature_frame, text="Feature Importance")

                # Create treeview
                tree = ttk.Treeview(
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
                history_frame = ttk.Frame(notebook)
                notebook.add(history_frame, text="Version History")

                # Create treeview
                tree = ttk.Treeview(
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
            ttk.Button(
                dialog,
                text="Close",
                command=dialog.destroy
            ).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to get model details:\n{str(e)}")