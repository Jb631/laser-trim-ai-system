"""
Comprehensive Progress Indicator System

This module provides a unified progress tracking system with multiple indicator types,
multi-level tracking, time estimation, cancellation support, and persistence capabilities.
"""

import time
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import deque
from enum import Enum
import logging

from PySide6.QtCore import (
    Qt, Signal, QObject, QTimer, QPropertyAnimation, 
    QEasingCurve, QThread, QMutex, QMutexLocker
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QDialog, QDialogButtonBox, QTextEdit,
    QStackedWidget, QFrame, QGridLayout, QGroupBox
)
from PySide6.QtGui import QPainter, QPen, QBrush, QConicalGradient, QPaintEvent

logger = logging.getLogger(__name__)


class ProgressType(Enum):
    """Types of progress indicators"""
    LINEAR = "linear"
    CIRCULAR = "circular"
    INDETERMINATE = "indeterminate"
    MULTI_LEVEL = "multi_level"


class ProgressState(Enum):
    """States of a progress operation"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgressData:
    """Data structure for progress information"""
    id: str
    name: str
    current: int = 0
    total: int = 100
    message: str = ""
    state: ProgressState = ProgressState.IDLE
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage"""
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100)
    
    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """Calculate elapsed time"""
        if not self.start_time:
            return None
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    @property
    def estimated_remaining(self) -> Optional[timedelta]:
        """Estimate remaining time"""
        if not self.start_time or self.current <= 0:
            return None
        
        elapsed = self.elapsed_time
        if not elapsed:
            return None
        
        rate = self.current / elapsed.total_seconds()
        if rate <= 0:
            return None
        
        remaining_items = self.total - self.current
        remaining_seconds = remaining_items / rate
        return timedelta(seconds=remaining_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        data['state'] = self.state.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressData':
        """Create from dictionary"""
        # Convert ISO format strings back to datetime objects
        if data.get('start_time'):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        if 'state' in data:
            data['state'] = ProgressState(data['state'])
        return cls(**data)


class CircularProgressWidget(QWidget):
    """Custom circular progress indicator widget"""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(100, 100)
        self._value = 0
        self._max_value = 100
        self._thickness = 10
        self._color = Qt.blue
        self._background_color = Qt.lightGray
        self._text_visible = True
        self._indeterminate = False
        self._rotation_angle = 0
        
        # Animation for indeterminate mode
        self._rotation_timer = QTimer(self)
        self._rotation_timer.timeout.connect(self._rotate)
        
    def setValue(self, value: int):
        """Set progress value"""
        self._value = max(0, min(value, self._max_value))
        self.update()
    
    def setMaximum(self, maximum: int):
        """Set maximum value"""
        self._max_value = max(1, maximum)
        self.update()
    
    def setIndeterminate(self, indeterminate: bool):
        """Set indeterminate mode"""
        self._indeterminate = indeterminate
        if indeterminate:
            self._rotation_timer.start(50)  # Update every 50ms
        else:
            self._rotation_timer.stop()
            self._rotation_angle = 0
        self.update()
    
    def _rotate(self):
        """Rotate for indeterminate animation"""
        self._rotation_angle = (self._rotation_angle + 10) % 360
        self.update()
    
    def paintEvent(self, event: QPaintEvent):
        """Custom paint event"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate dimensions
        width = self.width()
        height = self.height()
        side = min(width, height)
        radius = (side - self._thickness) / 2
        
        # Center the drawing area
        painter.translate(width / 2, height / 2)
        
        # Draw background circle
        painter.setPen(QPen(self._background_color, self._thickness))
        painter.drawEllipse(-radius, -radius, radius * 2, radius * 2)
        
        if self._indeterminate:
            # Draw spinning arc for indeterminate mode
            painter.setPen(QPen(self._color, self._thickness))
            painter.rotate(self._rotation_angle)
            painter.drawArc(-radius, -radius, radius * 2, radius * 2, 
                          0, 90 * 16)  # 90 degree arc
        else:
            # Draw progress arc
            if self._value > 0:
                painter.setPen(QPen(self._color, self._thickness))
                span_angle = int((self._value / self._max_value) * 360 * 16)
                painter.drawArc(-radius, -radius, radius * 2, radius * 2,
                              90 * 16, -span_angle)
        
        # Draw text
        if self._text_visible and not self._indeterminate:
            painter.setPen(Qt.black)
            painter.rotate(-self._rotation_angle if self._indeterminate else 0)
            percent = int((self._value / self._max_value) * 100)
            painter.drawText(-radius, -radius, radius * 2, radius * 2,
                           Qt.AlignCenter, f"{percent}%")


class MultiLevelProgressWidget(QWidget):
    """Widget for displaying multi-level progress"""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._progress_bars: Dict[str, QProgressBar] = {}
        
    def _setup_ui(self):
        """Set up the UI"""
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(5)
        
    def add_progress(self, progress_id: str, name: str, level: int = 0):
        """Add a progress bar"""
        # Create progress bar with indentation based on level
        container = QWidget()
        hlayout = QHBoxLayout(container)
        hlayout.setContentsMargins(level * 20, 0, 0, 0)
        
        label = QLabel(name)
        label.setMinimumWidth(150)
        hlayout.addWidget(label)
        
        progress_bar = QProgressBar()
        progress_bar.setTextVisible(True)
        hlayout.addWidget(progress_bar)
        
        self.layout.addWidget(container)
        self._progress_bars[progress_id] = progress_bar
        
    def update_progress(self, progress_id: str, current: int, total: int):
        """Update a progress bar"""
        if progress_id in self._progress_bars:
            progress_bar = self._progress_bars[progress_id]
            progress_bar.setMaximum(total)
            progress_bar.setValue(current)
            
    def remove_progress(self, progress_id: str):
        """Remove a progress bar"""
        if progress_id in self._progress_bars:
            progress_bar = self._progress_bars.pop(progress_id)
            progress_bar.parent().deleteLater()


class ProgressManager(QObject):
    """Central manager for all progress operations"""
    
    # Signals
    progress_updated = Signal(str, ProgressData)  # id, data
    progress_started = Signal(str, ProgressData)
    progress_completed = Signal(str, ProgressData)
    progress_cancelled = Signal(str, ProgressData)
    progress_error = Signal(str, ProgressData, str)  # id, data, error_message
    
    def __init__(self):
        super().__init__()
        self._progress_data: Dict[str, ProgressData] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._persistence_file = Path.home() / ".laser_trim_analyzer" / "progress_state.json"
        self._persistence_file.parent.mkdir(exist_ok=True)
        self._mutex = QMutex()
        self._save_timer = QTimer()
        self._save_timer.timeout.connect(self._save_state)
        self._save_timer.start(1000)  # Save every second
        
        # Load persisted state
        self._load_state()
        
    def create_progress(self, 
                       progress_id: str,
                       name: str,
                       total: int,
                       parent_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> ProgressData:
        """Create a new progress tracker"""
        with QMutexLocker(self._mutex):
            progress = ProgressData(
                id=progress_id,
                name=name,
                total=total,
                parent_id=parent_id,
                metadata=metadata or {}
            )
            
            self._progress_data[progress_id] = progress
            
            # Update parent's children list
            if parent_id and parent_id in self._progress_data:
                self._progress_data[parent_id].children.append(progress_id)
            
            return progress
    
    def start_progress(self, progress_id: str):
        """Start a progress operation"""
        with QMutexLocker(self._mutex):
            if progress_id in self._progress_data:
                progress = self._progress_data[progress_id]
                progress.state = ProgressState.RUNNING
                progress.start_time = datetime.now()
                self.progress_started.emit(progress_id, progress)
    
    def update_progress(self, 
                       progress_id: str,
                       current: Optional[int] = None,
                       message: Optional[str] = None,
                       increment: int = 0):
        """Update progress"""
        with QMutexLocker(self._mutex):
            if progress_id not in self._progress_data:
                return
            
            progress = self._progress_data[progress_id]
            
            if current is not None:
                progress.current = current
            elif increment > 0:
                progress.current = min(progress.current + increment, progress.total)
            
            if message is not None:
                progress.message = message
            
            # Update parent progress based on children
            if progress.parent_id:
                self._update_parent_progress(progress.parent_id)
            
            self.progress_updated.emit(progress_id, progress)
            
            # Check if completed
            if progress.current >= progress.total:
                self.complete_progress(progress_id)
    
    def _update_parent_progress(self, parent_id: str):
        """Update parent progress based on children"""
        if parent_id not in self._progress_data:
            return
        
        parent = self._progress_data[parent_id]
        if not parent.children:
            return
        
        # Calculate total progress from children
        total_progress = 0
        total_weight = 0
        
        for child_id in parent.children:
            if child_id in self._progress_data:
                child = self._progress_data[child_id]
                total_progress += child.progress_percent
                total_weight += 1
        
        if total_weight > 0:
            parent.current = int((total_progress / total_weight) * parent.total / 100)
    
    def complete_progress(self, progress_id: str):
        """Mark progress as completed"""
        with QMutexLocker(self._mutex):
            if progress_id in self._progress_data:
                progress = self._progress_data[progress_id]
                progress.state = ProgressState.COMPLETED
                progress.end_time = datetime.now()
                progress.current = progress.total
                self.progress_completed.emit(progress_id, progress)
    
    def cancel_progress(self, progress_id: str):
        """Cancel a progress operation"""
        with QMutexLocker(self._mutex):
            if progress_id in self._progress_data:
                progress = self._progress_data[progress_id]
                progress.state = ProgressState.CANCELLED
                progress.end_time = datetime.now()
                
                # Cancel all children
                for child_id in progress.children:
                    self.cancel_progress(child_id)
                
                self.progress_cancelled.emit(progress_id, progress)
    
    def error_progress(self, progress_id: str, error_message: str):
        """Mark progress as error"""
        with QMutexLocker(self._mutex):
            if progress_id in self._progress_data:
                progress = self._progress_data[progress_id]
                progress.state = ProgressState.ERROR
                progress.end_time = datetime.now()
                progress.metadata['error'] = error_message
                self.progress_error.emit(progress_id, progress, error_message)
    
    def get_progress(self, progress_id: str) -> Optional[ProgressData]:
        """Get progress data"""
        with QMutexLocker(self._mutex):
            return self._progress_data.get(progress_id)
    
    def get_all_progress(self) -> Dict[str, ProgressData]:
        """Get all progress data"""
        with QMutexLocker(self._mutex):
            return self._progress_data.copy()
    
    def is_cancellable(self, progress_id: str) -> bool:
        """Check if progress can be cancelled"""
        with QMutexLocker(self._mutex):
            progress = self._progress_data.get(progress_id)
            return (progress is not None and 
                   progress.state == ProgressState.RUNNING and
                   progress.metadata.get('cancellable', True))
    
    def is_resumable(self, progress_id: str) -> bool:
        """Check if progress can be resumed"""
        with QMutexLocker(self._mutex):
            progress = self._progress_data.get(progress_id)
            return (progress is not None and 
                   progress.state in [ProgressState.PAUSED, ProgressState.CANCELLED] and
                   progress.metadata.get('resumable', False))
    
    def pause_progress(self, progress_id: str):
        """Pause a progress operation"""
        with QMutexLocker(self._mutex):
            if progress_id in self._progress_data:
                progress = self._progress_data[progress_id]
                if progress.state == ProgressState.RUNNING:
                    progress.state = ProgressState.PAUSED
                    progress.metadata['pause_time'] = datetime.now().isoformat()
    
    def resume_progress(self, progress_id: str):
        """Resume a paused progress operation"""
        with QMutexLocker(self._mutex):
            if progress_id in self._progress_data:
                progress = self._progress_data[progress_id]
                if progress.state == ProgressState.PAUSED:
                    progress.state = ProgressState.RUNNING
                    
                    # Adjust start time for pause duration
                    if 'pause_time' in progress.metadata:
                        pause_time = datetime.fromisoformat(progress.metadata['pause_time'])
                        pause_duration = datetime.now() - pause_time
                        if progress.start_time:
                            progress.start_time += pause_duration
                        del progress.metadata['pause_time']
    
    def clear_completed(self):
        """Clear completed progress entries"""
        with QMutexLocker(self._mutex):
            completed_ids = [
                pid for pid, progress in self._progress_data.items()
                if progress.state in [ProgressState.COMPLETED, ProgressState.ERROR]
            ]
            for pid in completed_ids:
                del self._progress_data[pid]
    
    def _save_state(self):
        """Save progress state to disk"""
        try:
            with QMutexLocker(self._mutex):
                # Only save resumable progress
                resumable_progress = {
                    pid: progress.to_dict()
                    for pid, progress in self._progress_data.items()
                    if progress.metadata.get('resumable', False)
                }
                
                with open(self._persistence_file, 'w') as f:
                    json.dump(resumable_progress, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress state: {e}")
    
    def _load_state(self):
        """Load progress state from disk"""
        try:
            if self._persistence_file.exists():
                with open(self._persistence_file, 'r') as f:
                    data = json.load(f)
                
                for pid, progress_data in data.items():
                    progress = ProgressData.from_dict(progress_data)
                    # Mark as paused if it was running
                    if progress.state == ProgressState.RUNNING:
                        progress.state = ProgressState.PAUSED
                    self._progress_data[pid] = progress
        except Exception as e:
            logger.error(f"Failed to load progress state: {e}")


class ProgressDialog(QDialog):
    """Cancellable progress dialog"""
    
    cancelled = Signal()
    
    def __init__(self, 
                 title: str,
                 message: str = "",
                 progress_type: ProgressType = ProgressType.LINEAR,
                 cancellable: bool = True,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self._progress_type = progress_type
        self._cancellable = cancellable
        self._cancelled = False
        
        self._setup_ui(message)
        
    def _setup_ui(self, message: str):
        """Set up the UI"""
        layout = QVBoxLayout(self)
        
        # Message label
        self.message_label = QLabel(message)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        
        # Progress widget
        if self._progress_type == ProgressType.LINEAR:
            self.progress_widget = QProgressBar()
            self.progress_widget.setTextVisible(True)
        elif self._progress_type == ProgressType.CIRCULAR:
            self.progress_widget = CircularProgressWidget()
        elif self._progress_type == ProgressType.INDETERMINATE:
            self.progress_widget = QProgressBar()
            self.progress_widget.setRange(0, 0)  # Indeterminate mode
        elif self._progress_type == ProgressType.MULTI_LEVEL:
            self.progress_widget = MultiLevelProgressWidget()
        
        layout.addWidget(self.progress_widget)
        
        # Time estimates
        self.time_frame = QFrame()
        time_layout = QGridLayout(self.time_frame)
        
        time_layout.addWidget(QLabel("Elapsed:"), 0, 0)
        self.elapsed_label = QLabel("00:00:00")
        time_layout.addWidget(self.elapsed_label, 0, 1)
        
        time_layout.addWidget(QLabel("Remaining:"), 1, 0)
        self.remaining_label = QLabel("Calculating...")
        time_layout.addWidget(self.remaining_label, 1, 1)
        
        time_layout.addWidget(QLabel("Speed:"), 2, 0)
        self.speed_label = QLabel("0 items/sec")
        time_layout.addWidget(self.speed_label, 2, 1)
        
        layout.addWidget(self.time_frame)
        
        # Details text (expandable)
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(100)
        self.details_text.setVisible(False)
        layout.addWidget(self.details_text)
        
        # Buttons
        button_box = QDialogButtonBox()
        
        self.details_button = QPushButton("Show Details")
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self._toggle_details)
        button_box.addButton(self.details_button, QDialogButtonBox.ActionRole)
        
        if self._cancellable:
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self._on_cancel)
            button_box.addButton(self.cancel_button, QDialogButtonBox.RejectRole)
        
        layout.addWidget(button_box)
        
        # Update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_time_display)
        self.update_timer.start(100)  # Update every 100ms
        
    def _toggle_details(self, checked: bool):
        """Toggle details visibility"""
        self.details_text.setVisible(checked)
        self.details_button.setText("Hide Details" if checked else "Show Details")
        
    def _on_cancel(self):
        """Handle cancel button"""
        self._cancelled = True
        self.cancelled.emit()
        self.reject()
        
    def _update_time_display(self):
        """Update time display"""
        # This would be connected to actual progress data
        pass
    
    def set_progress(self, current: int, total: int):
        """Set progress value"""
        if hasattr(self.progress_widget, 'setMaximum'):
            self.progress_widget.setMaximum(total)
            self.progress_widget.setValue(current)
        elif hasattr(self.progress_widget, 'update_progress'):
            # For multi-level progress
            self.progress_widget.update_progress("main", current, total)
    
    def set_message(self, message: str):
        """Update message"""
        self.message_label.setText(message)
        
    def append_details(self, text: str):
        """Append to details text"""
        self.details_text.append(text)
        
    def is_cancelled(self) -> bool:
        """Check if cancelled"""
        return self._cancelled


class ProgressTracker:
    """Context manager for tracking progress"""
    
    def __init__(self,
                 manager: ProgressManager,
                 progress_id: str,
                 name: str,
                 total: int,
                 show_dialog: bool = False,
                 dialog_parent: Optional[QWidget] = None,
                 **kwargs):
        self.manager = manager
        self.progress_id = progress_id
        self.name = name
        self.total = total
        self.show_dialog = show_dialog
        self.dialog_parent = dialog_parent
        self.kwargs = kwargs
        self.dialog: Optional[ProgressDialog] = None
        self._cancelled = False
        
    def __enter__(self):
        """Enter context"""
        # Create progress
        self.manager.create_progress(
            self.progress_id,
            self.name,
            self.total,
            metadata=self.kwargs
        )
        self.manager.start_progress(self.progress_id)
        
        # Show dialog if requested
        if self.show_dialog:
            self.dialog = ProgressDialog(
                self.name,
                cancellable=self.kwargs.get('cancellable', True),
                parent=self.dialog_parent
            )
            self.dialog.cancelled.connect(self._on_cancelled)
            self.dialog.show()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        if exc_type is not None:
            # Error occurred
            self.manager.error_progress(self.progress_id, str(exc_val))
        elif self._cancelled:
            # Was cancelled
            self.manager.cancel_progress(self.progress_id)
        else:
            # Normal completion
            self.manager.complete_progress(self.progress_id)
        
        # Close dialog
        if self.dialog:
            self.dialog.accept()
            
    def _on_cancelled(self):
        """Handle cancellation"""
        self._cancelled = True
        self.manager.cancel_progress(self.progress_id)
        
    def update(self, current: Optional[int] = None, message: Optional[str] = None, increment: int = 0):
        """Update progress"""
        self.manager.update_progress(self.progress_id, current, message, increment)
        
        # Update dialog
        if self.dialog:
            progress = self.manager.get_progress(self.progress_id)
            if progress:
                self.dialog.set_progress(progress.current, progress.total)
                if message:
                    self.dialog.set_message(message)
                    
    def is_cancelled(self) -> bool:
        """Check if cancelled"""
        return self._cancelled
    
    def add_details(self, text: str):
        """Add details text"""
        if self.dialog:
            self.dialog.append_details(text)


# Integration examples
class FileProcessingExample:
    """Example of progress tracking for file processing"""
    
    def __init__(self, progress_manager: ProgressManager):
        self.progress_manager = progress_manager
        
    def process_files(self, files: List[Path], parent_widget: Optional[QWidget] = None):
        """Process multiple files with progress tracking"""
        
        # Main progress tracker
        with ProgressTracker(
            self.progress_manager,
            "file_processing",
            "Processing Files",
            len(files),
            show_dialog=True,
            dialog_parent=parent_widget,
            cancellable=True,
            resumable=True
        ) as tracker:
            
            for i, file in enumerate(files):
                if tracker.is_cancelled():
                    break
                
                # Update main progress
                tracker.update(i, f"Processing {file.name}")
                
                # Sub-progress for file analysis
                with ProgressTracker(
                    self.progress_manager,
                    f"file_{i}",
                    f"Analyzing {file.name}",
                    100,
                    parent_id="file_processing"
                ) as file_tracker:
                    
                    # Simulate file processing steps
                    for step in range(100):
                        if file_tracker.is_cancelled():
                            break
                        
                        file_tracker.update(step, f"Step {step}/100")
                        time.sleep(0.01)  # Simulate work
                        
                tracker.add_details(f"Completed: {file.name}")


class BatchAnalysisExample:
    """Example of progress tracking for batch analysis"""
    
    def __init__(self, progress_manager: ProgressManager):
        self.progress_manager = progress_manager
        
    def analyze_batch(self, items: List[Any], parent_widget: Optional[QWidget] = None):
        """Analyze batch with multi-level progress"""
        
        dialog = ProgressDialog(
            "Batch Analysis",
            "Analyzing items...",
            ProgressType.MULTI_LEVEL,
            parent=parent_widget
        )
        dialog.show()
        
        # Create progress hierarchy
        main_progress = self.progress_manager.create_progress(
            "batch_main",
            "Overall Progress",
            len(items)
        )
        
        # Add to dialog
        if hasattr(dialog.progress_widget, 'add_progress'):
            dialog.progress_widget.add_progress("batch_main", "Overall Progress", 0)
        
        self.progress_manager.start_progress("batch_main")
        
        for i, item in enumerate(items):
            # Create sub-progress
            sub_id = f"item_{i}"
            sub_progress = self.progress_manager.create_progress(
                sub_id,
                f"Item {i+1}",
                3,  # 3 steps per item
                parent_id="batch_main"
            )
            
            # Add to dialog
            if hasattr(dialog.progress_widget, 'add_progress'):
                dialog.progress_widget.add_progress(sub_id, f"Item {i+1}", 1)
            
            self.progress_manager.start_progress(sub_id)
            
            # Simulate analysis steps
            for step in range(3):
                self.progress_manager.update_progress(sub_id, step + 1, f"Step {step + 1}")
                time.sleep(0.5)  # Simulate work
                
                # Update dialog
                if hasattr(dialog.progress_widget, 'update_progress'):
                    dialog.progress_widget.update_progress(sub_id, step + 1, 3)
            
            self.progress_manager.complete_progress(sub_id)
            self.progress_manager.update_progress("batch_main", i + 1)
            
            # Update dialog
            if hasattr(dialog.progress_widget, 'update_progress'):
                dialog.progress_widget.update_progress("batch_main", i + 1, len(items))
        
        self.progress_manager.complete_progress("batch_main")
        dialog.accept()


# Convenience functions
def create_progress_manager() -> ProgressManager:
    """Create and return a progress manager instance"""
    return ProgressManager()


def show_progress_dialog(title: str, 
                        message: str = "",
                        progress_type: ProgressType = ProgressType.LINEAR,
                        parent: Optional[QWidget] = None) -> ProgressDialog:
    """Show a progress dialog"""
    dialog = ProgressDialog(title, message, progress_type, parent=parent)
    dialog.show()
    return dialog


# Thread-safe progress updater
class ThreadedProgressUpdater(QThread):
    """Thread for updating progress from background operations"""
    
    progress_update = Signal(str, int, int, str)  # id, current, total, message
    
    def __init__(self, progress_manager: ProgressManager):
        super().__init__()
        self.progress_manager = progress_manager
        self.active = True
        
    def run(self):
        """Run thread"""
        while self.active:
            # Emit updates for all active progress
            for pid, progress in self.progress_manager.get_all_progress().items():
                if progress.state == ProgressState.RUNNING:
                    self.progress_update.emit(
                        pid,
                        progress.current,
                        progress.total,
                        progress.message
                    )
            self.msleep(100)  # Update every 100ms
            
    def stop(self):
        """Stop thread"""
        self.active = False
        self.wait()