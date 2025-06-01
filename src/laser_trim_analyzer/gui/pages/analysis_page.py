"""
Analysis Page for Laser Trim Analyzer

Handles file processing and displays results with ML insights.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import threading
import asyncio
import time

from laser_trim_analyzer.gui.pages.base_page import BasePage
from laser_trim_analyzer.gui.widgets.file_drop_zone import FileDropZone
from laser_trim_analyzer.gui.widgets.file_analysis_widget import FileAnalysisWidget
from laser_trim_analyzer.gui.widgets.alert_banner import AlertBanner, AlertStack
from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.core.models import AnalysisResult, ProcessingMode


class AnalysisPage(BasePage):
    """
    Analysis page for processing laser trim files.

    Features:
    - Drag-and-drop file selection
    - Real-time processing progress
    - ML insights display
    - Result visualization
    """

    def __init__(self, parent: ttk.Frame, main_window: Any):
        """Initialize analysis page."""
        # Initialize state
        self.input_files: List[Path] = []
        self.file_widgets: Dict[str, FileAnalysisWidget] = {}
        self.processor: Optional[LaserTrimProcessor] = None
        self.is_processing = False
        self.current_task = None
        
        # Progress update throttling
        self.last_progress_update = 0
        self.progress_update_interval = 0.1  # Update every 100ms max

        # Processing options
        self.processing_mode = tk.StringVar(value='detail')
        self.enable_plots = tk.BooleanVar(value=True)
        self.enable_ml = tk.BooleanVar(value=True)
        self.enable_database = tk.BooleanVar(value=True)

        # UI components
        self.drop_zone = None
        self.file_list_frame = None
        self.alert_stack = None
        self.progress_frame = None
        self.results_notebook = None

        super().__init__(parent, main_window)

    def _create_page(self):
        """Create analysis page content."""
        # Main container
        container = ttk.Frame(self, style='TFrame')
        container.pack(fill='both', expand=True, padx=20, pady=20)

        # Title and description
        self._create_header(container)

        # Alert stack for notifications
        self.alert_stack = AlertStack(container)
        self.alert_stack.pack(fill='x', pady=(0, 10))

        # Main content area with two columns
        content_frame = ttk.Frame(container)
        content_frame.pack(fill='both', expand=True)

        # Left column - File selection and options
        left_column = ttk.Frame(content_frame)
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Right column - Results
        right_column = ttk.Frame(content_frame)
        right_column.pack(side='right', fill='both', expand=True, padx=(10, 0))

        # Create sections
        self._create_file_selection_section(left_column)
        self._create_options_section(left_column)
        self._create_action_section(left_column)
        self._create_results_section(right_column)

    def _create_header(self, parent):
        """Create page header."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x', pady=(0, 20))

        ttk.Label(
            header_frame,
            text="File Analysis",
            style='Title.TLabel'
        ).pack(side='left')

        # Quick stats
        self.stats_label = ttk.Label(
            header_frame,
            text="",
            font=('Segoe UI', 10),
            foreground=self.colors['text_secondary']
        )
        self.stats_label.pack(side='right')

    def _create_file_selection_section(self, parent):
        """Create file selection section."""
        # File selection frame
        file_frame = ttk.LabelFrame(
            parent,
            text="Select Files",
            padding=15
        )
        file_frame.pack(fill='both', expand=True, pady=(0, 10))

        # Drop zone
        self.drop_zone = FileDropZone(
            file_frame,
            on_files_dropped=self._handle_files_dropped,
            accept_extensions=['.xlsx', '.xls']
        )
        self.drop_zone.pack(fill='both', expand=True, pady=(0, 10))

        # Browse button
        ttk.Button(
            file_frame,
            text="Browse Files",
            command=self.browse_files
        ).pack()

        # Selected files list
        files_label_frame = ttk.LabelFrame(
            parent,
            text="Selected Files",
            padding=10
        )
        files_label_frame.pack(fill='both', expand=True, pady=(0, 10))

        # Scrollable frame for file widgets
        canvas = tk.Canvas(files_label_frame, height=200)
        scrollbar = ttk.Scrollbar(files_label_frame, orient='vertical', command=canvas.yview)
        self.file_list_frame = ttk.Frame(canvas)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas_frame = canvas.create_window((0, 0), window=self.file_list_frame, anchor='nw')

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Configure canvas scrolling
        def configure_scroll(e):
            canvas.configure(scrollregion=canvas.bbox('all'))

        self.file_list_frame.bind('<Configure>', configure_scroll)

    def _create_options_section(self, parent):
        """Create processing options section."""
        options_frame = ttk.LabelFrame(
            parent,
            text="Processing Options",
            padding=15
        )
        options_frame.pack(fill='x', pady=(0, 10))

        # Processing mode
        mode_frame = ttk.Frame(options_frame)
        mode_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(mode_frame, text="Processing Mode:").pack(side='left')

        ttk.Radiobutton(
            mode_frame,
            text="Detail (with plots)",
            variable=self.processing_mode,
            value='detail'
        ).pack(side='left', padx=(10, 20))

        ttk.Radiobutton(
            mode_frame,
            text="Speed (no plots)",
            variable=self.processing_mode,
            value='speed'
        ).pack(side='left')

        # Feature toggles
        features_frame = ttk.Frame(options_frame)
        features_frame.pack(fill='x')

        ttk.Checkbutton(
            features_frame,
            text="Generate plots",
            variable=self.enable_plots
        ).pack(side='left', padx=(0, 20))

        ttk.Checkbutton(
            features_frame,
            text="ML predictions",
            variable=self.enable_ml,
            state='normal' if self.main_window.ml_predictor else 'disabled'
        ).pack(side='left', padx=(0, 20))

        ttk.Checkbutton(
            features_frame,
            text="Save to database",
            variable=self.enable_database,
            state='normal' if self.main_window.db_manager else 'disabled'
        ).pack(side='left')

    def _create_action_section(self, parent):
        """Create action buttons section."""
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill='x', pady=(0, 10))

        self.analyze_button = ttk.Button(
            action_frame,
            text="Start Analysis",
            style='Primary.TButton',
            command=self._start_analysis,
            state='disabled'
        )
        self.analyze_button.pack(side='left', padx=(0, 10))

        self.cancel_button = ttk.Button(
            action_frame,
            text="Cancel",
            command=self._cancel_analysis,
            state='disabled'
        )
        self.cancel_button.pack(side='left', padx=(0, 10))

        ttk.Button(
            action_frame,
            text="Clear All",
            command=self._clear_files
        ).pack(side='left')

        # Progress bar
        self.progress_frame = ttk.Frame(parent)
        self.progress_frame.pack(fill='x', pady=(10, 0))
        # Don't pack initially

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill='x', pady=(0, 5))

        self.progress_label = ttk.Label(
            self.progress_frame,
            text="",
            font=('Segoe UI', 9)
        )
        self.progress_label.pack()

    def _create_results_section(self, parent):
        """Create results display section."""
        results_frame = ttk.LabelFrame(
            parent,
            text="Analysis Results",
            padding=15
        )
        results_frame.pack(fill='both', expand=True)

        # Notebook for different result views
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill='both', expand=True)

        # Summary tab
        self.summary_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.summary_frame, text="Summary")

        # ML Insights tab
        self.ml_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.ml_frame, text="ML Insights")

        # Details tab
        self.details_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.details_frame, text="Details")

        # Initialize with empty message
        self._show_empty_results()

    def _show_empty_results(self):
        """Show empty results message."""
        for frame in [self.summary_frame, self.ml_frame, self.details_frame]:
            for widget in frame.winfo_children():
                widget.destroy()

            ttk.Label(
                frame,
                text="No results yet. Select files and start analysis.",
                font=('Segoe UI', 11),
                foreground=self.colors['text_secondary']
            ).pack(expand=True)

    def browse_files(self):
        """Open file browser to select files."""
        files = filedialog.askopenfilenames(
            title="Select Excel Files",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )

        if files:
            self._add_files([Path(f) for f in files])

    def _handle_files_dropped(self, files: List[str]):
        """Handle files dropped on drop zone."""
        self._add_files([Path(f) for f in files])

    def _add_files(self, files: List[Path]):
        """Add files to the processing list."""
        # Filter for valid files
        valid_files = []
        for file in files:
            if file.suffix.lower() in ['.xlsx', '.xls']:
                if file not in self.input_files:
                    valid_files.append(file)

        if not valid_files:
            return

        # Add to list
        self.input_files.extend(valid_files)

        # Create file widgets
        for file in valid_files:
            widget = FileAnalysisWidget(
                self.file_list_frame,
                file_data={
                    'filename': file.name,
                    'file_path': str(file),
                    'status': 'Pending',
                    'model': file.stem.split('_')[0] if '_' in file.stem else 'Unknown',
                    'serial': file.stem.split('_')[1] if '_' in file.stem else 'Unknown'
                },
                on_view_plot=self._view_plot,
                on_export=self._export_file_results,
                on_details=self._show_file_details
            )
            widget.pack(fill='x', pady=(0, 5))
            self.file_widgets[str(file)] = widget

        # Update UI
        self._update_ui_state()
        self._update_stats()

    def _clear_files(self):
        """Clear all selected files."""
        self.input_files.clear()

        # Remove widgets
        for widget in self.file_widgets.values():
            widget.destroy()
        self.file_widgets.clear()

        # Clear results
        self._show_empty_results()

        # Update UI
        self._update_ui_state()
        self._update_stats()

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        has_files = len(self.input_files) > 0

        # Update buttons
        self.analyze_button.config(
            state='normal' if has_files and not self.is_processing else 'disabled'
        )
        self.cancel_button.config(
            state='normal' if self.is_processing else 'disabled'
        )

        # Update drop zone
        if self.is_processing:
            self.drop_zone.set_state('disabled')
        else:
            self.drop_zone.set_state('normal')

    def _update_stats(self):
        """Update statistics label."""
        num_files = len(self.input_files)
        if num_files == 0:
            self.stats_label.config(text="")
        else:
            self.stats_label.config(text=f"{num_files} file{'s' if num_files != 1 else ''} selected")

    def _start_analysis(self):
        """Start processing selected files."""
        if not self.input_files or self.is_processing:
            return

        self.is_processing = True
        self._update_ui_state()

        # Show progress
        self.progress_frame.pack(fill='x', pady=(10, 0))
        self.progress_var.set(0)
        self.progress_label.config(text="Initializing...")

        # Clear previous alerts
        self.alert_stack.clear_all()

        # Start processing in background thread
        thread = threading.Thread(target=self._process_files_thread, daemon=True)
        thread.start()

    def _cancel_analysis(self):
        """Cancel ongoing analysis."""
        if self.current_task:
            # TODO: Implement proper cancellation
            pass

        self.is_processing = False
        self._update_ui_state()
        self.progress_frame.pack_forget()

        self.alert_stack.add_alert(
            alert_type='warning',
            title='Analysis Cancelled',
            message='Processing was cancelled by user.',
            auto_dismiss=5
        )

    def _process_files_thread(self):
        """Background thread for file processing."""
        try:
            # Create event loop for async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run async processing
            results = loop.run_until_complete(self._process_files_async())

            # Update UI in main thread
            self.after(0, self._processing_complete, results)

        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            self.after(0, self._processing_error, str(e))

        finally:
            loop.close()

    async def _process_files_async(self) -> List[AnalysisResult]:
        """Async method to process files."""
        # Initialize processor
        self.processor = LaserTrimProcessor(
            config=self.config,
            db_manager=self.db_manager if self.enable_database.get() else None,
            ml_predictor=self.main_window.ml_predictor if self.enable_ml.get() else None,
            logger=self.logger
        )

        # Configure based on options
        self.config.processing.generate_plots = (
                self.enable_plots.get() and self.processing_mode.get() == 'detail'
        )

        # Create output directory
        output_dir = self.config.data_directory / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total_files = len(self.input_files)

        for i, file_path in enumerate(self.input_files):
            # Update progress
            progress = (i / total_files) * 100
            self.after(0, self._update_progress, progress, f"Processing {file_path.name}...")

            # Update file widget status
            self.after(0, self._update_file_status, str(file_path), 'Processing')

            try:
                # Process file
                result = await self.processor.process_file(
                    file_path,
                    output_dir / file_path.stem,
                    progress_callback=lambda msg, prog: self.after(
                        0, self._update_progress,
                        progress + (prog * (100 / total_files)),
                        f"{file_path.name}: {msg}"
                    )
                )

                results.append(result)

                # Update file widget with results
                self.after(0, self._update_file_widget, str(file_path), result)

            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {e}")
                self.after(0, self._update_file_status, str(file_path), 'Error')
                self.after(0, self._show_file_error, file_path.name, str(e))

        return results

    def _update_progress(self, value: float, text: str):
        """Update progress display."""
        current_time = time.time()
        
        # Throttle updates to prevent GUI choppiness
        if current_time - self.last_progress_update < self.progress_update_interval:
            return
            
        self.last_progress_update = current_time
        self.progress_var.set(value)
        self.progress_label.config(text=text)

    def _update_file_status(self, file_path: str, status: str):
        """Update file widget status."""
        if file_path in self.file_widgets:
            self.file_widgets[file_path].update_data({'status': status})

    def _update_file_widget(self, file_path: str, result: AnalysisResult):
        """Update file widget with analysis results."""
        if file_path in self.file_widgets:
            # Prepare data for widget
            primary_track = result.primary_track

            widget_data = {
                'filename': result.metadata.filename,
                'model': result.metadata.model,
                'serial': result.metadata.serial,
                'status': result.overall_status.value,
                'timestamp': datetime.now(),
                'has_multi_tracks': result.metadata.has_multi_tracks,
                'sigma_gradient': primary_track.sigma_analysis.sigma_gradient,
                'sigma_pass': primary_track.sigma_analysis.sigma_pass,
                'linearity_pass': primary_track.linearity_analysis.linearity_pass,
                'risk_category': primary_track.failure_prediction.risk_category.value if primary_track.failure_prediction else 'Unknown',
                'plot_path': primary_track.plot_path
            }

            # Add tracks for multi-track files
            if result.metadata.has_multi_tracks:
                widget_data['tracks'] = {}
                for track_id, track in result.tracks.items():
                    widget_data['tracks'][track_id] = {
                        'status': self._determine_track_status(track),
                        'sigma_gradient': track.sigma_analysis.sigma_gradient,
                        'sigma_pass': track.sigma_analysis.sigma_pass,
                        'linearity_pass': track.linearity_analysis.linearity_pass,
                        'risk_category': track.failure_prediction.risk_category.value if track.failure_prediction else 'Unknown'
                    }

            self.file_widgets[file_path].update_data(widget_data)

    def _determine_track_status(self, track) -> str:
        """Determine track status for display."""
        if not track.sigma_analysis.sigma_pass or not track.linearity_analysis.linearity_pass:
            return 'Fail'
        elif track.failure_prediction and track.failure_prediction.risk_category.value == 'High':
            return 'Warning'
        else:
            return 'Pass'

    def _show_file_error(self, filename: str, error: str):
        """Show error alert for file."""
        self.alert_stack.add_alert(
            alert_type='error',
            title=f'Error Processing {filename}',
            message=error,
            dismissible=True
        )

    def _processing_complete(self, results: List[AnalysisResult]):
        """Handle processing completion."""
        self.is_processing = False
        self._update_ui_state()

        # Hide progress
        self.progress_frame.pack_forget()

        # Show summary
        passed = sum(1 for r in results if r.overall_status.value == 'Pass')
        total = len(results)

        if total > 0:
            self.alert_stack.add_alert(
                alert_type='success',
                title='Analysis Complete',
                message=f'Processed {total} files. Pass rate: {passed / total * 100:.1f}%',
                auto_dismiss=10,
                actions=[
                    {'text': 'View Report', 'command': lambda: self._show_results(results)}
                ]
            )

            # Update results display
            self._show_results(results)

            # Mark home page for refresh
            if 'home' in self.main_window.pages:
                self.main_window.pages['home'].mark_needs_refresh()
        else:
            self.alert_stack.add_alert(
                alert_type='warning',
                title='No Results',
                message='No files were successfully processed.',
                auto_dismiss=5
            )

    def _processing_error(self, error: str):
        """Handle processing error."""
        self.is_processing = False
        self._update_ui_state()
        self.progress_frame.pack_forget()

        self.alert_stack.add_alert(
            alert_type='error',
            title='Processing Failed',
            message=f'An error occurred during processing: {error}',
            dismissible=True
        )

    def _show_results(self, results: List[AnalysisResult]):
        """Display analysis results."""
        # Clear existing results
        for frame in [self.summary_frame, self.ml_frame, self.details_frame]:
            for widget in frame.winfo_children():
                widget.destroy()

        # Show summary
        self._create_summary_view(self.summary_frame, results)

        # Show ML insights if available
        if any(hasattr(r, 'ml_predictions') for r in results):
            self._create_ml_insights_view(self.ml_frame, results)
        else:
            ttk.Label(
                self.ml_frame,
                text="No ML predictions available.",
                font=('Segoe UI', 11),
                foreground=self.colors['text_secondary']
            ).pack(expand=True)

        # Show details
        self._create_details_view(self.details_frame, results)

        # Switch to summary tab
        self.results_notebook.select(0)

    def _create_summary_view(self, parent: ttk.Frame, results: List[AnalysisResult]):
        """Create summary view of results."""
        # Summary stats
        stats_frame = ttk.Frame(parent)
        stats_frame.pack(fill='x', padx=10, pady=10)

        total = len(results)
        passed = sum(1 for r in results if r.overall_status.value == 'Pass')
        failed = total - passed

        # Get all tracks for detailed stats
        all_tracks = []
        for result in results:
            all_tracks.extend(result.tracks.values())

        # Calculate metrics
        avg_sigma = sum(t.sigma_analysis.sigma_gradient for t in all_tracks) / len(all_tracks) if all_tracks else 0
        high_risk = sum(
            1 for t in all_tracks if t.failure_prediction and t.failure_prediction.risk_category.value == 'High')

        # Create stat cards
        from laser_trim_analyzer.gui.widgets.stat_card import StatCard

        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack()

        # Configure grid
        for i in range(2):
            stats_grid.columnconfigure(i, weight=1, minsize=150)

        # Total files
        StatCard(
            stats_grid,
            title="Total Files",
            value=total,
            unit=""
        ).grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        # Pass rate
        pass_rate = (passed / total * 100) if total > 0 else 0
        pass_color = "success" if pass_rate >= 95 else "warning" if pass_rate >= 90 else "danger"
        StatCard(
            stats_grid,
            title="Pass Rate",
            value=f"{pass_rate:.1f}",
            unit="%",
            color_scheme=pass_color
        ).grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        # Average sigma
        sigma_color = "success" if avg_sigma <= 0.02 else "warning" if avg_sigma <= 0.03 else "danger"
        StatCard(
            stats_grid,
            title="Avg Sigma",
            value=f"{avg_sigma:.4f}",
            unit="",
            color_scheme=sigma_color
        ).grid(row=1, column=0, padx=5, pady=5, sticky='ew')

        # High risk
        risk_color = "success" if high_risk <= 2 else "warning" if high_risk <= 5 else "danger"
        StatCard(
            stats_grid,
            title="High Risk",
            value=high_risk,
            unit="",
            color_scheme=risk_color
        ).grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        # Results table
        table_frame = ttk.LabelFrame(parent, text="Results Summary", padding=10)
        table_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        # Create treeview
        columns = ('Model', 'Serial', 'Status', 'Sigma', 'Risk')
        tree = ttk.Treeview(table_frame, columns=columns, show='tree headings', height=10)

        # Configure columns
        tree.column('#0', width=200, stretch=True)
        tree.column('Model', width=80)
        tree.column('Serial', width=100)
        tree.column('Status', width=80)
        tree.column('Sigma', width=100)
        tree.column('Risk', width=80)

        # Set headings
        tree.heading('#0', text='File')
        for col in columns:
            tree.heading(col, text=col)

        # Add data
        for result in results:
            primary = result.primary_track
            tree.insert(
                '',
                'end',
                text=result.metadata.filename,
                values=(
                    result.metadata.model,
                    result.metadata.serial,
                    result.overall_status.value,
                    f"{primary.sigma_analysis.sigma_gradient:.4f}",
                    primary.failure_prediction.risk_category.value if primary.failure_prediction else 'Unknown'
                ),
                tags=(result.overall_status.value.lower(),)
            )

        # Configure tags
        tree.tag_configure('pass', foreground=self.colors['success'])
        tree.tag_configure('fail', foreground=self.colors['danger'])
        tree.tag_configure('warning', foreground=self.colors['warning'])

        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def _create_ml_insights_view(self, parent: ttk.Frame, results: List[AnalysisResult]):
        """Create ML insights view."""
        # Insights container
        container = ttk.Frame(parent)
        container.pack(fill='both', expand=True, padx=10, pady=10)

        # Collect all ML predictions
        predictions = []
        warnings = []
        recommendations = []

        for result in results:
            if hasattr(result, 'ml_predictions') and result.ml_predictions:
                pred = result.ml_predictions
                predictions.append({
                    'file': result.metadata.filename,
                    'prediction': pred
                })

                if 'warnings' in pred:
                    warnings.extend(pred['warnings'])
                if 'recommendations' in pred:
                    recommendations.extend(pred['recommendations'])

        if not predictions:
            ttk.Label(
                container,
                text="No ML predictions were generated.",
                font=('Segoe UI', 11),
                foreground=self.colors['text_secondary']
            ).pack(expand=True)
            return

        # Warnings section
        if warnings:
            warnings_frame = ttk.LabelFrame(container, text="ML Warnings", padding=10)
            warnings_frame.pack(fill='x', pady=(0, 10))

            # Limit to 5 warnings
            for warning in warnings[:5]:
                warning_label = ttk.Label(
                    warnings_frame,
                    text=f"⚠️ {warning}",
                    font=('Segoe UI', 10),
                    foreground=self.colors['warning'],
                    wraplength=400
                )
                warning_label.pack(anchor='w', pady=2)

            if len(warnings) > 5:
                ttk.Label(
                    warnings_frame,
                    text=f"... and {len(warnings) - 5} more warnings",
                    font=('Segoe UI', 9, 'italic'),
                    foreground=self.colors['text_secondary']
                ).pack(anchor='w')

        # Recommendations section
        if recommendations:
            rec_frame = ttk.LabelFrame(container, text="ML Recommendations", padding=10)
            rec_frame.pack(fill='x', pady=(0, 10))

            # Group similar recommendations
            unique_recs = list(set(recommendations))

            for i, rec in enumerate(unique_recs[:5], 1):
                rec_label = ttk.Label(
                    rec_frame,
                    text=f"{i}. {rec}",
                    font=('Segoe UI', 10),
                    wraplength=400
                )
                rec_label.pack(anchor='w', pady=2)

        # Predictions chart
        chart_frame = ttk.LabelFrame(container, text="Risk Distribution", padding=10)
        chart_frame.pack(fill='both', expand=True)

        # Count risk categories
        risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
        for pred_data in predictions:
            if 'risk_category' in pred_data['prediction']:
                risk = pred_data['prediction']['risk_category']
                if risk in risk_counts:
                    risk_counts[risk] += 1

        # Create simple bar chart
        from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget

        chart = ChartWidget(chart_frame, chart_type='bar', figsize=(6, 4))
        chart.pack(fill='both', expand=True)

        categories = list(risk_counts.keys())
        values = list(risk_counts.values())
        colors = ['pass', 'warning', 'fail']

        chart.plot_bar(
            categories, values, colors=colors,
            xlabel="Risk Category", ylabel="Count"
        )

    def _create_details_view(self, parent: ttk.Frame, results: List[AnalysisResult]):
        """Create detailed results view."""
        # Create text widget with scrollbar
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        text_widget = tk.Text(text_frame, wrap='word', height=20)
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Configure tags
        text_widget.tag_configure('heading', font=('Segoe UI', 12, 'bold'))
        text_widget.tag_configure('subheading', font=('Segoe UI', 10, 'bold'))
        text_widget.tag_configure('pass', foreground=self.colors['success'])
        text_widget.tag_configure('fail', foreground=self.colors['danger'])
        text_widget.tag_configure('warning', foreground=self.colors['warning'])

        # Add content
        for result in results:
            # File header
            text_widget.insert('end', f"\n{result.metadata.filename}\n", 'heading')
            text_widget.insert('end', f"Model: {result.metadata.model} | Serial: {result.metadata.serial}\n")
            text_widget.insert('end', f"Status: ")

            status_tag = result.overall_status.value.lower()
            text_widget.insert('end', f"{result.overall_status.value}\n", status_tag)

            # Track details
            for track_id, track in result.tracks.items():
                text_widget.insert('end', f"\n  {track_id}:\n", 'subheading')
                text_widget.insert('end', f"    Sigma Gradient: {track.sigma_analysis.sigma_gradient:.6f}\n")
                text_widget.insert('end', f"    Sigma Threshold: {track.sigma_analysis.sigma_threshold:.6f}\n")
                text_widget.insert('end', f"    Sigma Pass: {'Yes' if track.sigma_analysis.sigma_pass else 'No'}\n")
                text_widget.insert('end',
                                   f"    Linearity Pass: {'Yes' if track.linearity_analysis.linearity_pass else 'No'}\n")

                if track.failure_prediction:
                    text_widget.insert('end',
                                       f"    Failure Probability: {track.failure_prediction.failure_probability:.2%}\n")
                    text_widget.insert('end', f"    Risk Category: {track.failure_prediction.risk_category.value}\n")

            text_widget.insert('end', "\n" + "-" * 50 + "\n")

        # Make read-only
        text_widget.configure(state='disabled')

    def _view_plot(self, file_data: Dict[str, Any]):
        """View plot for a file."""
        plot_path = file_data.get('plot_path')
        if plot_path and Path(plot_path).exists():
            # Open in default image viewer
            import webbrowser
            webbrowser.open(str(plot_path))
        else:
            messagebox.showwarning("No Plot", "Plot file not found.")

    def _export_file_results(self, file_data: Dict[str, Any]):
        """Export individual file results."""
        # TODO: Implement export functionality
        messagebox.showinfo("Export", f"Export functionality for {file_data['filename']} coming soon!")

    def _show_file_details(self, file_data: Dict[str, Any]):
        """Show detailed results for a file."""
        # Create details dialog
        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title(f"Analysis Details - {file_data.get('filename', 'Unknown')}")
        dialog.geometry("700x600")
        dialog.configure(bg=self.colors['bg'])
        
        # Create scrollable text widget
        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap='word', font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Configure tags
        text_widget.tag_configure('heading', font=('Segoe UI', 14, 'bold'))
        text_widget.tag_configure('subheading', font=('Segoe UI', 12, 'bold'))
        text_widget.tag_configure('label', font=('Segoe UI', 10, 'bold'))
        text_widget.tag_configure('pass', foreground='#27ae60')
        text_widget.tag_configure('fail', foreground='#e74c3c')
        text_widget.tag_configure('warning', foreground='#f39c12')
        
        # Add content
        text_widget.insert('end', f"{file_data.get('filename', 'Unknown File')}\n", 'heading')
        text_widget.insert('end', "=" * 60 + "\n\n")
        
        # File information
        text_widget.insert('end', "File Information\n", 'subheading')
        text_widget.insert('end', f"Model: ", 'label')
        text_widget.insert('end', f"{file_data.get('model', 'N/A')}\n")
        text_widget.insert('end', f"Serial: ", 'label')
        text_widget.insert('end', f"{file_data.get('serial', 'N/A')}\n")
        text_widget.insert('end', f"Status: ", 'label')
        
        status = file_data.get('status', 'Unknown')
        status_tag = status.lower() if status.lower() in ['pass', 'fail', 'warning'] else None
        text_widget.insert('end', f"{status}\n", status_tag)
        
        text_widget.insert('end', f"Timestamp: ", 'label')
        timestamp = file_data.get('timestamp', datetime.now())
        if isinstance(timestamp, datetime):
            text_widget.insert('end', f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            text_widget.insert('end', f"{timestamp}\n")
        
        text_widget.insert('end', "\n")
        
        # Analysis Results
        text_widget.insert('end', "Analysis Results\n", 'subheading')
        text_widget.insert('end', f"Sigma Gradient: ", 'label')
        text_widget.insert('end', f"{file_data.get('sigma_gradient', 'N/A'):.6f}\n")
        
        text_widget.insert('end', f"Sigma Pass: ", 'label')
        sigma_pass = file_data.get('sigma_pass')
        if sigma_pass is not None:
            text_widget.insert('end', "YES\n" if sigma_pass else "NO\n", 'pass' if sigma_pass else 'fail')
        else:
            text_widget.insert('end', "N/A\n")
            
        text_widget.insert('end', f"Linearity Pass: ", 'label')
        lin_pass = file_data.get('linearity_pass')
        if lin_pass is not None:
            text_widget.insert('end', "YES\n" if lin_pass else "NO\n", 'pass' if lin_pass else 'fail')
        else:
            text_widget.insert('end', "N/A\n")
            
        text_widget.insert('end', f"Risk Category: ", 'label')
        risk = file_data.get('risk_category', 'Unknown')
        risk_tag = 'fail' if risk == 'High' else 'warning' if risk == 'Medium' else 'pass'
        text_widget.insert('end', f"{risk}\n", risk_tag)
        
        # Multi-track details if available
        if file_data.get('has_multi_tracks') and 'tracks' in file_data:
            text_widget.insert('end', "\nTrack Details\n", 'subheading')
            for track_id, track_info in file_data['tracks'].items():
                text_widget.insert('end', f"\n{track_id}:\n", 'label')
                text_widget.insert('end', f"  Status: {track_info.get('status', 'N/A')}\n")
                text_widget.insert('end', f"  Sigma Gradient: {track_info.get('sigma_gradient', 'N/A'):.6f}\n")
                text_widget.insert('end', f"  Sigma Pass: {'YES' if track_info.get('sigma_pass') else 'NO'}\n")
                text_widget.insert('end', f"  Linearity Pass: {'YES' if track_info.get('linearity_pass') else 'NO'}\n")
                text_widget.insert('end', f"  Risk: {track_info.get('risk_category', 'N/A')}\n")
        
        # Make read-only
        text_widget.configure(state='disabled')
        
        # Add close button
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(
            button_frame,
            text="Close",
            command=dialog.destroy
        ).pack(side='right', padx=10)
        
        # If plot exists, add view plot button
        if file_data.get('plot_path') and Path(str(file_data['plot_path'])).exists():
            ttk.Button(
                button_frame,
                text="View Plot",
                command=lambda: self._view_plot(file_data)
            ).pack(side='right', padx=5)