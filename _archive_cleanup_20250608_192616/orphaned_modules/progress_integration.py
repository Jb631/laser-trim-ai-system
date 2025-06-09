"""
Progress System Integration Examples

This module provides concrete examples of integrating the progress system
with various components of the laser trim analyzer application.
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
import logging

from PySide6.QtCore import QObject, Signal, QThread, Slot
from PySide6.QtWidgets import QWidget

from ..core.processor import LaserTrimProcessor
from ..core.models import ProcessingResult, FileMetadata
from ..analysis.analytics_engine import AnalyticsEngine
from ..ml.engine import MLEngine
from .progress_system import (
    ProgressManager, ProgressTracker, ProgressDialog,
    ProgressType, ThreadedProgressUpdater
)

logger = logging.getLogger(__name__)


class FileProcessorWithProgress(QObject):
    """File processor with integrated progress tracking"""
    
    # Signals
    processing_complete = Signal(ProcessingResult)
    processing_error = Signal(str)
    
    def __init__(self, progress_manager: ProgressManager):
        super().__init__()
        self.progress_manager = progress_manager
        self.processor = LaserTrimProcessor()
        
    def process_file(self, 
                    file_path: Path,
                    show_dialog: bool = True,
                    parent_widget: Optional[QWidget] = None) -> Optional[ProcessingResult]:
        """Process a single file with progress tracking"""
        
        progress_id = f"file_{file_path.stem}"
        
        try:
            with ProgressTracker(
                self.progress_manager,
                progress_id,
                f"Processing {file_path.name}",
                100,  # We'll use percentage
                show_dialog=show_dialog,
                dialog_parent=parent_widget,
                cancellable=True
            ) as tracker:
                
                # Step 1: Load file (20%)
                tracker.update(10, "Loading file...")
                metadata = FileMetadata.from_file(file_path)
                
                if tracker.is_cancelled():
                    return None
                
                # Step 2: Parse data (40%)
                tracker.update(30, "Parsing Excel data...")
                # Simulate parsing with actual processor
                
                if tracker.is_cancelled():
                    return None
                
                # Step 3: Process tracks (60%)
                tracker.update(50, "Processing tracks...")
                result = self.processor.process_file(str(file_path))
                
                if tracker.is_cancelled():
                    return None
                
                # Step 4: Run analysis (80%)
                tracker.update(70, "Running analysis...")
                # Additional analysis steps
                
                # Step 5: Finalize (100%)
                tracker.update(90, "Finalizing results...")
                tracker.update(100, "Complete!")
                
                self.processing_complete.emit(result)
                return result
                
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            self.processing_error.emit(str(e))
            return None


class BatchProcessorWithProgress(QThread):
    """Batch processor with multi-level progress tracking"""
    
    # Signals
    file_completed = Signal(Path, ProcessingResult)
    batch_completed = Signal(list)  # List of results
    error_occurred = Signal(Path, str)
    
    def __init__(self, 
                 files: List[Path],
                 progress_manager: ProgressManager,
                 parent_widget: Optional[QWidget] = None):
        super().__init__()
        self.files = files
        self.progress_manager = progress_manager
        self.parent_widget = parent_widget
        self.processor = LaserTrimProcessor()
        self._cancelled = False
        
    def run(self):
        """Run batch processing"""
        results = []
        
        # Create main progress
        main_id = "batch_processing"
        self.progress_manager.create_progress(
            main_id,
            "Batch Processing",
            len(self.files),
            metadata={
                'cancellable': True,
                'resumable': True,
                'total_files': len(self.files)
            }
        )
        self.progress_manager.start_progress(main_id)
        
        for i, file in enumerate(self.files):
            if self._cancelled:
                break
                
            # Update main progress
            self.progress_manager.update_progress(
                main_id,
                current=i,
                message=f"Processing file {i+1} of {len(self.files)}: {file.name}"
            )
            
            # Create sub-progress for this file
            file_id = f"file_{i}_{file.stem}"
            self.progress_manager.create_progress(
                file_id,
                file.name,
                4,  # 4 steps per file
                parent_id=main_id,
                metadata={'file_path': str(file)}
            )
            self.progress_manager.start_progress(file_id)
            
            try:
                # Step 1: Load
                self.progress_manager.update_progress(file_id, 1, "Loading...")
                
                # Step 2: Parse
                self.progress_manager.update_progress(file_id, 2, "Parsing...")
                
                # Step 3: Process
                self.progress_manager.update_progress(file_id, 3, "Processing...")
                result = self.processor.process_file(str(file))
                
                # Step 4: Complete
                self.progress_manager.update_progress(file_id, 4, "Complete")
                self.progress_manager.complete_progress(file_id)
                
                results.append(result)
                self.file_completed.emit(file, result)
                
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                self.progress_manager.error_progress(file_id, str(e))
                self.error_occurred.emit(file, str(e))
        
        # Complete main progress
        self.progress_manager.complete_progress(main_id)
        self.batch_completed.emit(results)
        
    def cancel(self):
        """Cancel batch processing"""
        self._cancelled = True
        self.progress_manager.cancel_progress("batch_processing")


class AnalysisEngineWithProgress:
    """Analytics engine with progress tracking"""
    
    def __init__(self, 
                 analytics_engine: AnalyticsEngine,
                 progress_manager: ProgressManager):
        self.engine = analytics_engine
        self.progress_manager = progress_manager
        
    def run_analysis(self,
                    data: Dict[str, Any],
                    analysis_types: List[str],
                    show_progress: bool = True,
                    parent_widget: Optional[QWidget] = None) -> Dict[str, Any]:
        """Run analysis with progress tracking"""
        
        results = {}
        
        with ProgressTracker(
            self.progress_manager,
            "analysis",
            "Running Analysis",
            len(analysis_types),
            show_dialog=show_progress,
            dialog_parent=parent_widget
        ) as tracker:
            
            for i, analysis_type in enumerate(analysis_types):
                if tracker.is_cancelled():
                    break
                    
                tracker.update(i, f"Running {analysis_type} analysis...")
                
                # Create sub-progress for detailed tracking
                sub_id = f"analysis_{analysis_type}"
                with ProgressTracker(
                    self.progress_manager,
                    sub_id,
                    analysis_type,
                    100,
                    parent_id="analysis"
                ) as sub_tracker:
                    
                    # Simulate analysis steps
                    sub_tracker.update(25, "Preparing data...")
                    sub_tracker.update(50, "Running calculations...")
                    sub_tracker.update(75, "Generating results...")
                    
                    # Run actual analysis
                    result = self.engine.analyze(data, analysis_type)
                    results[analysis_type] = result
                    
                    sub_tracker.update(100, "Complete")
                    
                tracker.add_details(f"Completed: {analysis_type}")
                
        return results


class MLEngineWithProgress:
    """ML engine with progress tracking for training and prediction"""
    
    def __init__(self,
                 ml_engine: MLEngine,
                 progress_manager: ProgressManager):
        self.engine = ml_engine
        self.progress_manager = progress_manager
        
    def train_model(self,
                   training_data: Any,
                   epochs: int = 100,
                   show_progress: bool = True,
                   parent_widget: Optional[QWidget] = None):
        """Train model with progress tracking"""
        
        with ProgressTracker(
            self.progress_manager,
            "ml_training",
            "Training ML Model",
            epochs,
            show_dialog=show_progress,
            dialog_parent=parent_widget,
            cancellable=True,
            resumable=True
        ) as tracker:
            
            for epoch in range(epochs):
                if tracker.is_cancelled():
                    break
                    
                # Update progress
                tracker.update(
                    epoch,
                    f"Epoch {epoch+1}/{epochs} - Loss: {0.1 * (epochs - epoch):.4f}"
                )
                
                # Simulate training step
                # In real implementation, this would be actual training
                import time
                time.sleep(0.1)
                
                # Log details
                tracker.add_details(
                    f"Epoch {epoch+1}: loss={0.1 * (epochs - epoch):.4f}, "
                    f"accuracy={0.9 + 0.001 * epoch:.4f}"
                )
                
    def predict_batch(self,
                     data_items: List[Any],
                     show_progress: bool = True,
                     parent_widget: Optional[QWidget] = None) -> List[Any]:
        """Batch prediction with progress tracking"""
        
        predictions = []
        
        # Use indeterminate progress for quick operations
        if len(data_items) < 10:
            dialog = ProgressDialog(
                "Running Predictions",
                "Processing...",
                ProgressType.INDETERMINATE,
                cancellable=False,
                parent=parent_widget
            )
            if show_progress:
                dialog.show()
                
            # Run predictions
            for item in data_items:
                prediction = self.engine.predict(item)
                predictions.append(prediction)
                
            dialog.accept()
            
        else:
            # Use regular progress for larger batches
            with ProgressTracker(
                self.progress_manager,
                "ml_prediction",
                "Running Predictions",
                len(data_items),
                show_dialog=show_progress,
                dialog_parent=parent_widget
            ) as tracker:
                
                for i, item in enumerate(data_items):
                    if tracker.is_cancelled():
                        break
                        
                    tracker.update(i, f"Predicting item {i+1}/{len(data_items)}")
                    
                    prediction = self.engine.predict(item)
                    predictions.append(prediction)
                    
        return predictions


class ResumableOperation:
    """Example of a resumable operation with progress tracking"""
    
    def __init__(self, progress_manager: ProgressManager):
        self.progress_manager = progress_manager
        self.state_file = Path.home() / ".laser_trim_analyzer" / "operation_state.json"
        
    def long_running_operation(self,
                             items: List[Any],
                             operation_id: str = "resumable_op"):
        """Long running operation that can be resumed"""
        
        # Check if we're resuming
        progress = self.progress_manager.get_progress(operation_id)
        start_index = 0
        
        if progress and self.progress_manager.is_resumable(operation_id):
            # Resume from where we left off
            start_index = progress.current
            self.progress_manager.resume_progress(operation_id)
            logger.info(f"Resuming operation from item {start_index}")
        else:
            # Start new operation
            self.progress_manager.create_progress(
                operation_id,
                "Long Running Operation",
                len(items),
                metadata={
                    'resumable': True,
                    'cancellable': True,
                    'items': [str(item) for item in items]
                }
            )
            self.progress_manager.start_progress(operation_id)
            
        # Process items
        for i in range(start_index, len(items)):
            progress = self.progress_manager.get_progress(operation_id)
            if not progress or progress.state != "running":
                break
                
            # Update progress
            self.progress_manager.update_progress(
                operation_id,
                current=i,
                message=f"Processing item {i+1}/{len(items)}"
            )
            
            # Process item
            self._process_item(items[i])
            
            # Save state periodically
            if i % 10 == 0:
                self._save_state(operation_id, i)
                
        # Complete or handle interruption
        progress = self.progress_manager.get_progress(operation_id)
        if progress and progress.current >= progress.total:
            self.progress_manager.complete_progress(operation_id)
            
    def _process_item(self, item: Any):
        """Process a single item"""
        import time
        time.sleep(0.5)  # Simulate work
        
    def _save_state(self, operation_id: str, current_index: int):
        """Save operation state"""
        import json
        state = {
            'operation_id': operation_id,
            'current_index': current_index,
            'timestamp': str(Path.ctime(Path.cwd()))
        }
        
        self.state_file.parent.mkdir(exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f)


class ProgressIntegrationHelper:
    """Helper class for easy progress integration"""
    
    @staticmethod
    def wrap_function_with_progress(
        func: Callable,
        progress_manager: ProgressManager,
        operation_name: str,
        total_steps: int = 100,
        show_dialog: bool = True
    ) -> Callable:
        """Wrap a function with progress tracking"""
        
        def wrapped(*args, **kwargs):
            with ProgressTracker(
                progress_manager,
                f"wrapped_{func.__name__}",
                operation_name,
                total_steps,
                show_dialog=show_dialog
            ) as tracker:
                
                # Pass tracker to function if it accepts it
                import inspect
                sig = inspect.signature(func)
                if 'progress_tracker' in sig.parameters:
                    kwargs['progress_tracker'] = tracker
                    
                return func(*args, **kwargs)
                
        return wrapped
    
    @staticmethod
    def create_progress_callback(
        progress_manager: ProgressManager,
        progress_id: str
    ) -> Callable[[int, int, str], None]:
        """Create a progress callback function"""
        
        def callback(current: int, total: int, message: str = ""):
            progress_manager.update_progress(
                progress_id,
                current=current,
                message=message
            )
            
        return callback


# Example usage in existing components
def integrate_with_existing_processor():
    """Example of integrating progress with existing processor"""
    
    progress_manager = ProgressManager()
    
    # Modify existing processor to accept progress callback
    class EnhancedProcessor(LaserTrimProcessor):
        def process_file(self, 
                        file_path: str,
                        progress_callback: Optional[Callable] = None) -> ProcessingResult:
            """Process file with optional progress callback"""
            
            # Call progress at various stages
            if progress_callback:
                progress_callback(10, 100, "Loading file...")
                
            # ... existing processing logic ...
            
            if progress_callback:
                progress_callback(50, 100, "Processing tracks...")
                
            # ... more processing ...
            
            if progress_callback:
                progress_callback(100, 100, "Complete!")
                
            return super().process_file(file_path)
    
    # Use with progress
    processor = EnhancedProcessor()
    callback = ProgressIntegrationHelper.create_progress_callback(
        progress_manager,
        "file_processing"
    )
    
    result = processor.process_file("test.xls", progress_callback=callback)