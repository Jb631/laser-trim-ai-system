"""
Example Integration of Progress System with GUI Pages

This module demonstrates how to integrate the comprehensive progress system
with the existing GUI pages and operations.
"""

from pathlib import Path
from typing import List, Optional
import logging

from PySide6.QtCore import QThread, Signal, Slot, QObject
from PySide6.QtWidgets import QWidget, QMessageBox

from ..core.processor import LaserTrimProcessor
from ..core.models import ProcessingResult
from ..analysis.analytics_engine import AnalyticsEngine
from ..ml.engine import MLEngine

from .progress_system import (
    ProgressManager, ProgressTracker, ProgressDialog,
    ProgressType, ProgressState, create_progress_manager
)
from .progress_integration import (
    FileProcessorWithProgress, BatchProcessorWithProgress,
    AnalysisEngineWithProgress, MLEngineWithProgress
)

logger = logging.getLogger(__name__)


class SingleFilePageWithProgress:
    """Example integration for single file processing page."""
    
    def __init__(self, parent_widget: QWidget):
        self.parent_widget = parent_widget
        self.progress_manager = create_progress_manager()
        self.file_processor = FileProcessorWithProgress(self.progress_manager)
        
        # Connect signals
        self.file_processor.processing_complete.connect(self._on_processing_complete)
        self.file_processor.processing_error.connect(self._on_processing_error)
        
    def process_file(self, file_path: Path):
        """Process a single file with progress tracking."""
        # Process with dialog
        result = self.file_processor.process_file(
            file_path,
            show_dialog=True,
            parent_widget=self.parent_widget
        )
        
        if result:
            # Additional analysis with progress
            self._run_analysis(result)
            
    def _run_analysis(self, result: ProcessingResult):
        """Run additional analysis on the result."""
        analytics_engine = AnalyticsEngine()
        analyzer = AnalysisEngineWithProgress(analytics_engine, self.progress_manager)
        
        # Define analysis types to run
        analysis_types = ["linearity", "resistance", "consistency", "sigma"]
        
        # Run analysis with progress
        analysis_results = analyzer.run_analysis(
            result.to_dict(),
            analysis_types,
            show_progress=True,
            parent_widget=self.parent_widget
        )
        
        # Display results
        self._display_results(result, analysis_results)
        
    @Slot(ProcessingResult)
    def _on_processing_complete(self, result: ProcessingResult):
        """Handle processing completion."""
        logger.info(f"Processing complete: {result.metadata.part_number}")
        
    @Slot(str)
    def _on_processing_error(self, error_message: str):
        """Handle processing error."""
        QMessageBox.critical(
            self.parent_widget,
            "Processing Error",
            f"An error occurred during processing:\n{error_message}"
        )
        
    def _display_results(self, result: ProcessingResult, analysis_results: dict):
        """Display processing and analysis results."""
        # This would update the GUI with the results
        pass


class BatchProcessingPageWithProgress:
    """Example integration for batch processing page."""
    
    def __init__(self, parent_widget: QWidget):
        self.parent_widget = parent_widget
        self.progress_manager = create_progress_manager()
        self.batch_thread: Optional[BatchProcessorWithProgress] = None
        
    def process_batch(self, files: List[Path]):
        """Process multiple files with multi-level progress tracking."""
        # Create progress dialog with multi-level display
        dialog = ProgressDialog(
            "Batch Processing",
            f"Processing {len(files)} files...",
            ProgressType.MULTI_LEVEL,
            cancellable=True,
            parent=self.parent_widget
        )
        
        # Create batch processor thread
        self.batch_thread = BatchProcessorWithProgress(
            files,
            self.progress_manager,
            self.parent_widget
        )
        
        # Connect signals
        self.batch_thread.file_completed.connect(self._on_file_completed)
        self.batch_thread.batch_completed.connect(self._on_batch_completed)
        self.batch_thread.error_occurred.connect(self._on_error_occurred)
        
        # Connect dialog cancellation
        dialog.cancelled.connect(self.batch_thread.cancel)
        
        # Connect progress updates to dialog
        self.progress_manager.progress_updated.connect(
            lambda pid, data: self._update_dialog(dialog, pid, data)
        )
        
        # Start processing
        self.batch_thread.start()
        
        # Show dialog (blocks until complete or cancelled)
        result = dialog.exec()
        
        # Clean up
        if self.batch_thread.isRunning():
            self.batch_thread.cancel()
            self.batch_thread.wait()
            
    def _update_dialog(self, dialog: ProgressDialog, progress_id: str, data):
        """Update dialog with progress data."""
        if hasattr(dialog.progress_widget, 'update_progress'):
            dialog.progress_widget.update_progress(
                progress_id,
                data.current,
                data.total
            )
            
        # Update time estimates
        if data.elapsed_time:
            elapsed = data.elapsed_time.total_seconds()
            dialog.elapsed_label.setText(
                f"Elapsed: {self._format_time(elapsed)}"
            )
            
        if data.estimated_remaining:
            remaining = data.estimated_remaining.total_seconds()
            dialog.remaining_label.setText(
                f"Remaining: {self._format_time(remaining)}"
            )
            
        # Update speed
        if data.elapsed_time and data.current > 0:
            speed = data.current / data.elapsed_time.total_seconds()
            dialog.speed_label.setText(f"Speed: {speed:.1f} items/sec")
            
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
            
    @Slot(Path, ProcessingResult)
    def _on_file_completed(self, file_path: Path, result: ProcessingResult):
        """Handle file completion."""
        logger.info(f"Completed: {file_path.name}")
        
    @Slot(list)
    def _on_batch_completed(self, results: List[ProcessingResult]):
        """Handle batch completion."""
        QMessageBox.information(
            self.parent_widget,
            "Batch Complete",
            f"Successfully processed {len(results)} files."
        )
        
    @Slot(Path, str)
    def _on_error_occurred(self, file_path: Path, error: str):
        """Handle file error."""
        logger.error(f"Error processing {file_path}: {error}")


class MLToolsPageWithProgress:
    """Example integration for ML tools page."""
    
    def __init__(self, parent_widget: QWidget):
        self.parent_widget = parent_widget
        self.progress_manager = create_progress_manager()
        self.ml_engine = MLEngine()
        self.ml_with_progress = MLEngineWithProgress(self.ml_engine, self.progress_manager)
        
    def train_model(self, training_data):
        """Train ML model with progress tracking."""
        # Show training progress
        self.ml_with_progress.train_model(
            training_data,
            epochs=100,
            show_progress=True,
            parent_widget=self.parent_widget
        )
        
        QMessageBox.information(
            self.parent_widget,
            "Training Complete",
            "Model training completed successfully."
        )
        
    def predict_batch(self, data_items: List):
        """Run batch predictions with progress."""
        predictions = self.ml_with_progress.predict_batch(
            data_items,
            show_progress=True,
            parent_widget=self.parent_widget
        )
        
        return predictions


class LongRunningOperationExample:
    """Example of a long-running operation with resume capability."""
    
    def __init__(self, parent_widget: QWidget):
        self.parent_widget = parent_widget
        self.progress_manager = create_progress_manager()
        
    def start_operation(self, items: List):
        """Start a long-running operation that can be resumed."""
        operation_id = "long_operation"
        
        # Check if we can resume
        existing_progress = self.progress_manager.get_progress(operation_id)
        
        if existing_progress and self.progress_manager.is_resumable(operation_id):
            result = QMessageBox.question(
                self.parent_widget,
                "Resume Operation",
                f"Found incomplete operation with {existing_progress.current}/{existing_progress.total} items processed.\n"
                "Would you like to resume?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if result != QMessageBox.Yes:
                # Start fresh
                self.progress_manager.clear_completed()
                existing_progress = None
                
        # Create or resume progress
        with ProgressTracker(
            self.progress_manager,
            operation_id,
            "Long Running Operation",
            len(items),
            show_dialog=True,
            dialog_parent=self.parent_widget,
            cancellable=True,
            resumable=True
        ) as tracker:
            
            # Start from where we left off
            start_index = existing_progress.current if existing_progress else 0
            
            for i in range(start_index, len(items)):
                if tracker.is_cancelled():
                    # Save state for resume
                    logger.info(f"Operation cancelled at item {i}")
                    break
                    
                # Process item
                tracker.update(i, f"Processing item {i+1}/{len(items)}")
                self._process_item(items[i])
                
                # Add details
                tracker.add_details(f"Processed: {items[i]}")
                
            if not tracker.is_cancelled():
                QMessageBox.information(
                    self.parent_widget,
                    "Operation Complete",
                    f"Successfully processed all {len(items)} items."
                )
                
    def _process_item(self, item):
        """Process a single item."""
        import time
        time.sleep(0.5)  # Simulate work


class ProgressSystemIntegrationHelper:
    """Helper class for easy integration with existing GUI components."""
    
    @staticmethod
    def add_progress_to_widget(widget: QWidget) -> ProgressManager:
        """Add progress management to a widget."""
        if not hasattr(widget, '_progress_manager'):
            widget._progress_manager = create_progress_manager()
        return widget._progress_manager
        
    @staticmethod
    def wrap_method_with_progress(
        widget: QWidget,
        method_name: str,
        operation_name: str,
        total_steps: int = 100
    ):
        """Wrap a widget method with progress tracking."""
        progress_manager = ProgressSystemIntegrationHelper.add_progress_to_widget(widget)
        original_method = getattr(widget, method_name)
        
        def wrapped_method(*args, **kwargs):
            with ProgressTracker(
                progress_manager,
                f"{method_name}_progress",
                operation_name,
                total_steps,
                show_dialog=True,
                dialog_parent=widget
            ) as tracker:
                # Pass tracker if method accepts it
                import inspect
                sig = inspect.signature(original_method)
                if 'progress_tracker' in sig.parameters:
                    kwargs['progress_tracker'] = tracker
                    
                return original_method(*args, **kwargs)
                
        setattr(widget, method_name, wrapped_method)
        
    @staticmethod
    def create_progress_callback(
        widget: QWidget,
        operation_name: str
    ) -> callable:
        """Create a progress callback for use in existing methods."""
        progress_manager = ProgressSystemIntegrationHelper.add_progress_to_widget(widget)
        progress_id = f"{operation_name}_progress"
        
        # Create progress
        progress_manager.create_progress(
            progress_id,
            operation_name,
            100,
            metadata={'auto_created': True}
        )
        progress_manager.start_progress(progress_id)
        
        def callback(current: int, total: int, message: str = ""):
            progress_manager.update_progress(
                progress_id,
                current=current,
                message=message
            )
            
            # Auto-complete when done
            if current >= total:
                progress_manager.complete_progress(progress_id)
                
        return callback


# Example usage in existing code
def integrate_with_existing_gui():
    """Example of integrating progress system with existing GUI."""
    
    # Example 1: Add to existing single file page
    class ExistingSingleFilePage(QWidget):
        def __init__(self):
            super().__init__()
            # Add progress integration
            self.progress_integration = SingleFilePageWithProgress(self)
            
        def on_file_selected(self, file_path: str):
            # Use progress-enabled processing
            self.progress_integration.process_file(Path(file_path))
            
    # Example 2: Wrap existing method
    class ExistingBatchPage(QWidget):
        def __init__(self):
            super().__init__()
            # Wrap batch processing with progress
            ProgressSystemIntegrationHelper.wrap_method_with_progress(
                self,
                'process_batch',
                'Batch Processing',
                100
            )
            
        def process_batch(self, files: List[str], progress_tracker=None):
            for i, file in enumerate(files):
                if progress_tracker:
                    progress_tracker.update(
                        int((i / len(files)) * 100),
                        f"Processing {Path(file).name}"
                    )
                # Process file...
                
    # Example 3: Add progress callback to existing processor
    class ExistingProcessor:
        def process(self, data, progress_callback=None):
            steps = 10
            for i in range(steps):
                if progress_callback:
                    progress_callback(i, steps, f"Step {i+1}")
                # Do processing...
                
    # Use with callback
    widget = QWidget()
    processor = ExistingProcessor()
    callback = ProgressSystemIntegrationHelper.create_progress_callback(
        widget,
        "Data Processing"
    )
    processor.process(data=[], progress_callback=callback)