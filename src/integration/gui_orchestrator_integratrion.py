"""
GUI-Orchestrator Integration Module
===================================

This module integrates the orchestrator with the GUI application,
providing threading support and progress tracking.

Author: QA Team
Date: 2024
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import json

from laser_trim_orchestrator import LaserTrimOrchestrator, ProcessingStatus


class GUIOrchestrator:
    """
    Bridges the GUI application with the orchestrator,
    handling threading and progress updates.
    """

    def __init__(self, gui_app):
        """
        Initialize GUI orchestrator integration.

        Args:
            gui_app: Reference to the main GUI application
        """
        self.gui_app = gui_app
        self.orchestrator = None
        self.progress_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.processing_thread = None
        self.is_processing = False

        # Initialize orchestrator with GUI's config
        self._initialize_orchestrator()

    def _initialize_orchestrator(self):
        """Initialize the orchestrator with current settings."""
        try:
            # Get settings from GUI
            enable_ml = self.gui_app.ml_enabled.get() if hasattr(self.gui_app, 'ml_enabled') else True
            enable_parallel = self.gui_app.parallel_processing.get() if hasattr(self.gui_app,
                                                                                'parallel_processing') else True

            # Create orchestrator
            self.orchestrator = LaserTrimOrchestrator(
                config_path=self.gui_app.config_manager.config_path,
                enable_parallel=enable_parallel,
                enable_ml=enable_ml,
                enable_db=True  # Always enable database for GUI
            )

            return True

        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize orchestrator: {str(e)}")
            return False

    def process_files_async(self, file_paths: List[str],
                            progress_callback: Callable,
                            completion_callback: Callable):
        """
        Process files asynchronously with progress updates.

        Args:
            file_paths: List of file paths to process
            progress_callback: Function to call with progress updates
            completion_callback: Function to call when complete
        """
        if self.is_processing:
            messagebox.showwarning("Processing", "Analysis already in progress!")
            return

        self.is_processing = True

        # Create progress tracking wrapper
        def progress_wrapper():
            try:
                # Create temporary folder for individual files
                temp_dir = Path("temp_processing") / datetime.now().strftime('%Y%m%d_%H%M%S')
                temp_dir.mkdir(parents=True, exist_ok=True)

                # Copy files to temp directory
                for file_path in file_paths:
                    Path(file_path).rename(temp_dir / Path(file_path).name)

                # Process with orchestrator
                total_files = len(file_paths)

                # Monitor progress
                for i, file_path in enumerate(file_paths):
                    if not self.is_processing:
                        break

                    # Update progress
                    progress = (i / total_files) * 100
                    self.progress_queue.put({
                        'type': 'progress',
                        'value': progress,
                        'message': f'Processing {Path(file_path).name}...'
                    })

                # Run orchestrator
                results = self.orchestrator.process_folder(
                    temp_dir,
                    generate_report=self.gui_app.auto_report.get()
                )

                # Send completion
                self.result_queue.put({
                    'type': 'complete',
                    'results': results
                })

            except Exception as e:
                self.result_queue.put({
                    'type': 'error',
                    'error': str(e)
                })

            finally:
                self.is_processing = False

        # Start processing thread
        self.processing_thread = threading.Thread(target=progress_wrapper, daemon=True)
        self.processing_thread.start()

        # Start monitoring queues
        self._monitor_queues(progress_callback, completion_callback)

    def _monitor_queues(self, progress_callback: Callable, completion_callback: Callable):
        """Monitor progress and result queues."""

        def check_queues():
            # Check progress queue
            try:
                while True:
                    progress_data = self.progress_queue.get_nowait()
                    if progress_data['type'] == 'progress':
                        progress_callback(progress_data['value'], progress_data['message'])
            except queue.Empty:
                pass

            # Check result queue
            try:
                result = self.result_queue.get_nowait()
                if result['type'] == 'complete':
                    completion_callback(result['results'])
                    return
                elif result['type'] == 'error':
                    messagebox.showerror("Processing Error", result['error'])
                    completion_callback(None)
                    return
            except queue.Empty:
                pass

            # Continue monitoring if still processing
            if self.is_processing:
                self.gui_app.root.after(100, check_queues)

        # Start monitoring
        self.gui_app.root.after(100, check_queues)

    def stop_processing(self):
        """Stop current processing."""
        self.is_processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            # Note: Can't directly stop thread, but flag will stop at next file
            messagebox.showinfo("Stopping", "Processing will stop after current file completes.")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if self.orchestrator:
            return self.orchestrator.get_system_status()
        return {'error': 'Orchestrator not initialized'}

    def train_ml_models(self, progress_dialog):
        """Train ML models with progress dialog."""

        def training_thread():
            try:
                progress_dialog.update_progress(10, "Loading historical data...")

                # Train models
                results = self.orchestrator.train_ml_models(days_back=90)

                if 'error' in results:
                    raise Exception(results['error'])

                progress_dialog.update_progress(100, "Training complete!")

                # Show results
                self.gui_app.root.after(100, lambda: self._show_training_results(results))

            except Exception as e:
                self.gui_app.root.after(100, lambda: messagebox.showerror("Training Error", str(e)))

            finally:
                self.gui_app.root.after(100, progress_dialog.destroy)

        # Start training thread
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()

    def _show_training_results(self, results: Dict[str, Any]):
        """Show ML training results in a dialog."""
        dialog = tk.Toplevel(self.gui_app.root)
        dialog.title("ML Training Results")
        dialog.geometry("400x300")

        # Results text
        text = tk.Text(dialog, wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True)

        # Format results
        text.insert(tk.END, "Machine Learning Model Training Complete\n\n", 'heading')

        if 'threshold_optimizer' in results:
            text.insert(tk.END, "Threshold Optimizer:\n", 'subheading')
            text.insert(tk.END, f"  MAE: {results['threshold_optimizer'].get('mae', 'N/A'):.4f}\n")
            text.insert(tk.END, f"  R² Score: {results['threshold_optimizer'].get('r2_score', 'N/A'):.4f}\n\n")

        if 'failure_predictor' in results:
            text.insert(tk.END, "Failure Predictor:\n", 'subheading')
            text.insert(tk.END, f"  Accuracy: {results['failure_predictor'].get('accuracy', 'N/A'):.2%}\n")
            text.insert(tk.END, f"  Precision: {results['failure_predictor'].get('precision', 'N/A'):.2%}\n\n")

        if 'drift_detector' in results:
            text.insert(tk.END, "Drift Detector:\n", 'subheading')
            text.insert(tk.END, f"  Anomalies: {results['drift_detector'].get('n_anomalies', 'N/A')}\n")
            text.insert(tk.END, f"  Anomaly Rate: {results['drift_detector'].get('anomaly_rate', 'N/A'):.1%}\n\n")

        text.insert(tk.END, f"Models saved to: {results.get('saved_version', 'Unknown')}")

        # Configure tags
        text.tag_config('heading', font=('Arial', 12, 'bold'))
        text.tag_config('subheading', font=('Arial', 10, 'bold'))

        text.config(state=tk.DISABLED)

        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)


class EnhancedProgressDialog(tk.Toplevel):
    """Enhanced progress dialog with detailed status."""

    def __init__(self, parent, title="Processing", show_details=True):
        super().__init__(parent)
        self.title(title)
        self.geometry("500x300" if show_details else "400x200")
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Variables
        self.cancelled = False
        self.show_details = show_details

        # Create UI
        self._create_ui()

    def _create_ui(self):
        """Create progress dialog UI."""
        # Main frame
        main_frame = tk.Frame(self, bg='white', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        self.title_label = tk.Label(
            main_frame,
            text="Processing Files",
            font=('Segoe UI', 12, 'bold'),
            bg='white'
        )
        self.title_label.pack(pady=(0, 10))

        # Overall progress
        tk.Label(main_frame, text="Overall Progress:", bg='white').pack(anchor=tk.W)

        self.overall_progress = ttk.Progressbar(
            main_frame,
            length=450 if self.show_details else 350,
            mode='determinate'
        )
        self.overall_progress.pack(pady=(5, 10))

        # Current file
        self.current_file_label = tk.Label(
            main_frame,
            text="Preparing...",
            font=('Segoe UI', 9),
            bg='white',
            fg='#666'
        )
        self.current_file_label.pack(pady=(0, 10))

        # Details frame (if enabled)
        if self.show_details:
            details_frame = tk.LabelFrame(
                main_frame,
                text="Details",
                bg='white',
                font=('Segoe UI', 9)
            )
            details_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

            # Details text
            self.details_text = tk.Text(
                details_frame,
                height=6,
                width=50,
                wrap=tk.WORD,
                bg='#f5f5f5',
                font=('Consolas', 8)
            )
            self.details_text.pack(padx=5, pady=5)

            # Scrollbar
            scrollbar = ttk.Scrollbar(self.details_text)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.details_text.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=self.details_text.yview)

        # Buttons
        button_frame = tk.Frame(main_frame, bg='white')
        button_frame.pack(fill=tk.X)

        self.cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            bg='#f44336',
            fg='white',
            bd=0,
            padx=20,
            pady=5,
            cursor='hand2',
            command=self.cancel
        )
        self.cancel_button.pack(side=tk.RIGHT)

    def update_progress(self, value: float, status: str = "", details: str = ""):
        """Update progress dialog."""
        self.overall_progress['value'] = value

        if status:
            self.current_file_label['text'] = status

        if details and self.show_details:
            self.details_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {details}\n")
            self.details_text.see(tk.END)

        self.update()

    def add_detail(self, detail: str, level: str = "info"):
        """Add detail message."""
        if not self.show_details:
            return

        # Color code by level
        tag = level
        timestamp = datetime.now().strftime('%H:%M:%S')

        self.details_text.insert(tk.END, f"{timestamp} ", 'timestamp')
        self.details_text.insert(tk.END, f"[{level.upper()}] ", tag)
        self.details_text.insert(tk.END, f"{detail}\n")

        # Configure tags
        self.details_text.tag_config('timestamp', foreground='#999')
        self.details_text.tag_config('info', foreground='#2196F3')
        self.details_text.tag_config('success', foreground='#4CAF50')
        self.details_text.tag_config('warning', foreground='#FF9800')
        self.details_text.tag_config('error', foreground='#f44336')

        self.details_text.see(tk.END)

    def cancel(self):
        """Cancel operation."""
        self.cancelled = True
        self.cancel_button.config(text="Cancelling...", state=tk.DISABLED)


def integrate_orchestrator_with_gui(gui_app):
    """
    Integrate the orchestrator with an existing GUI application.

    Args:
        gui_app: The main GUI application instance

    Returns:
        GUIOrchestrator instance
    """
    # Create orchestrator wrapper
    gui_orchestrator = GUIOrchestrator(gui_app)

    # Replace GUI's analysis method
    original_run_analysis = gui_app._run_analysis

    def enhanced_run_analysis():
        """Enhanced analysis using orchestrator."""
        if not gui_app.loaded_files:
            messagebox.showwarning("No Files", "Please load files before running analysis.")
            return

        # Create progress dialog
        progress = EnhancedProgressDialog(gui_app.root, "Running Analysis", show_details=True)

        # Progress callback
        def on_progress(value, message):
            progress.update_progress(value, message)

            # Add details based on message
            if "Processing" in message:
                progress.add_detail(message, "info")
            elif "Complete" in message:
                progress.add_detail(message, "success")
            elif "Error" in message or "Failed" in message:
                progress.add_detail(message, "error")

        # Completion callback
        def on_complete(results):
            progress.destroy()

            if results:
                # Update GUI with results
                gui_app.current_results = results
                gui_app._display_results()

                # Show summary
                messagebox.showinfo(
                    "Analysis Complete",
                    f"Processed {results['total_files']} files\n"
                    f"Successful: {results['successful']}\n"
                    f"Failed: {results['failed']}\n"
                    f"Time: {results['processing_time']:.1f}s"
                )

                # Open report if generated
                if results.get('report_path'):
                    if messagebox.askyesno("Report Generated", "Open report now?"):
                        import os
                        os.startfile(results['report_path'])
            else:
                messagebox.showerror("Analysis Failed", "An error occurred during analysis.")

        # Start processing
        gui_orchestrator.process_files_async(
            gui_app.loaded_files,
            on_progress,
            on_complete
        )

    # Replace method
    gui_app._run_analysis = enhanced_run_analysis

    # Add new menu items
    if hasattr(gui_app, 'menubar'):
        # Add System menu
        system_menu = tk.Menu(gui_app.menubar, tearoff=0)
        gui_app.menubar.add_cascade(label="System", menu=system_menu)

        system_menu.add_command(
            label="System Status",
            command=lambda: show_system_status(gui_app, gui_orchestrator)
        )
        system_menu.add_command(
            label="Train ML Models",
            command=lambda: train_ml_models_dialog(gui_app, gui_orchestrator)
        )
        system_menu.add_separator()
        system_menu.add_command(
            label="Process History",
            command=lambda: show_process_history(gui_app, gui_orchestrator)
        )

    return gui_orchestrator


def show_system_status(gui_app, gui_orchestrator):
    """Show system status dialog."""
    status = gui_orchestrator.get_system_status()

    dialog = tk.Toplevel(gui_app.root)
    dialog.title("System Status")
    dialog.geometry("500x400")

    # Create notebook for different sections
    notebook = ttk.Notebook(dialog)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Components tab
    comp_frame = ttk.Frame(notebook)
    notebook.add(comp_frame, text="Components")

    for comp, state in status.get('components', {}).items():
        frame = tk.Frame(comp_frame)
        frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(frame, text=comp.replace('_', ' ').title() + ":", width=20, anchor=tk.W).pack(side=tk.LEFT)

        color = 'green' if state == 'active' else 'orange'
        tk.Label(frame, text=state.upper(), fg=color, font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

    # Metrics tab
    metrics_frame = ttk.Frame(notebook)
    notebook.add(metrics_frame, text="Metrics")

    for metric, value in status.get('metrics', {}).items():
        frame = tk.Frame(metrics_frame)
        frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(frame, text=metric.replace('_', ' ').title() + ":", width=20, anchor=tk.W).pack(side=tk.LEFT)
        tk.Label(frame, text=str(value)).pack(side=tk.LEFT)

    # ML Models tab
    if 'ml_models' in status:
        ml_frame = ttk.Frame(notebook)
        notebook.add(ml_frame, text="ML Models")

        for model, trained in status['ml_models'].items():
            frame = tk.Frame(ml_frame)
            frame.pack(fill=tk.X, padx=10, pady=5)

            tk.Label(frame, text=model.replace('_', ' ').title() + ":", width=20, anchor=tk.W).pack(side=tk.LEFT)

            text = "Trained" if trained else "Not Trained"
            color = 'green' if trained else 'red'
            tk.Label(frame, text=text, fg=color, font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

    # Database tab
    if 'database' in status:
        db_frame = ttk.Frame(notebook)
        notebook.add(db_frame, text="Database")

        for key, value in status['database'].items():
            frame = tk.Frame(db_frame)
            frame.pack(fill=tk.X, padx=10, pady=5)

            tk.Label(frame, text=key.replace('_', ' ').title() + ":", width=20, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(frame, text=str(value)).pack(side=tk.LEFT)

    # Close button
    ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)


def train_ml_models_dialog(gui_app, gui_orchestrator):
    """Show ML training dialog."""
    dialog = tk.Toplevel(gui_app.root)
    dialog.title("Train ML Models")
    dialog.geometry("400x200")

    # Options frame
    frame = tk.Frame(dialog, padx=20, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(frame, text="Train machine learning models with historical data",
             font=('Arial', 10)).pack(pady=(0, 20))

    # Data source selection
    data_source = tk.StringVar(value="database")

    tk.Radiobutton(frame, text="From Database (last 90 days)",
                   variable=data_source, value="database").pack(anchor=tk.W)
    tk.Radiobutton(frame, text="From CSV file",
                   variable=data_source, value="file").pack(anchor=tk.W)

    # Buttons
    button_frame = tk.Frame(frame)
    button_frame.pack(fill=tk.X, pady=(20, 0))

    def start_training():
        dialog.destroy()

        # Create progress dialog
        progress = EnhancedProgressDialog(gui_app.root, "Training ML Models", show_details=True)
        progress.add_detail("Initializing training process...", "info")

        # Start training
        gui_orchestrator.train_ml_models(progress)

    ttk.Button(button_frame, text="Start Training", command=start_training).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT)


def show_process_history(gui_app, gui_orchestrator):
    """Show processing history."""
    dialog = tk.Toplevel(gui_app.root)
    dialog.title("Processing History")
    dialog.geometry("800x600")

    # Create treeview
    columns = ('Date', 'Files', 'Successful', 'Failed', 'Time', 'Report')
    tree = ttk.Treeview(dialog, columns=columns, show='tree headings')

    # Configure columns
    tree.column('#0', width=0, stretch=False)
    tree.column('Date', width=150)
    tree.column('Files', width=80)
    tree.column('Successful', width=100)
    tree.column('Failed', width=80)
    tree.column('Time', width=80)
    tree.column('Report', width=200)

    # Configure headings
    for col in columns:
        tree.heading(col, text=col)

    # Add scrollbar
    scrollbar = ttk.Scrollbar(dialog, orient='vertical', command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)

    # Pack
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Load history (from database if available)
    if gui_orchestrator.orchestrator and gui_orchestrator.orchestrator.db_manager:
        try:
            # Get recent runs
            with gui_orchestrator.orchestrator.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp, total_files, processed_files, failed_files, 
                           processing_time, notes
                    FROM analysis_runs
                    ORDER BY timestamp DESC
                    LIMIT 100
                ''')

                for row in cursor.fetchall():
                    tree.insert('', 'end', values=(
                        row['timestamp'],
                        row['total_files'],
                        row['processed_files'],
                        row['failed_files'],
                        f"{row['processing_time']:.1f}s" if row['processing_time'] else 'N/A',
                        row['notes'] or ''
                    ))
        except Exception as e:
            tree.insert('', 'end', values=(
                'Error loading history',
                str(e),
                '', '', '', ''
            ))

    # Buttons
    button_frame = tk.Frame(dialog)
    button_frame.pack(fill=tk.X, pady=10)

    ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT, padx=10)
    ttk.Button(button_frame, text="Export",
               command=lambda: export_history(tree)).pack(side=tk.RIGHT)


def export_history(tree):
    """Export history to CSV."""
    from tkinter import filedialog
    import csv

    filename = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if filename:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write headers
            headers = [tree.heading(col)['text'] for col in tree['columns']]
            writer.writerow(headers)

            # Write data
            for item in tree.get_children():
                values = tree.item(item)['values']
                writer.writerow(values)

        messagebox.showinfo("Export Complete", f"History exported to {filename}")


# Example usage
if __name__ == "__main__":
    # This would be called from your main GUI application
    # from gui_application import LaserTrimAIApp
    #
    # app = LaserTrimAIApp()
    # orchestrator = integrate_orchestrator_with_gui(app)
    # app.run()

    print("GUI Orchestrator Integration Module")
    print("Import this module and use integrate_orchestrator_with_gui()")