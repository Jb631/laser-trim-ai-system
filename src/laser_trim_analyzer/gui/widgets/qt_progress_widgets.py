"""Qt-based Progress widgets for the laser trim analyzer GUI."""

from typing import Optional, Dict, Any
from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QFrame, QApplication,
    QDialog, QTextEdit, QGroupBox
)
from PySide6.QtGui import QPalette, QColor

# Import from the new progress system
from ..progress_system import (
    ProgressManager, ProgressData, ProgressType, ProgressState,
    CircularProgressWidget as BaseCircularProgress,
    MultiLevelProgressWidget as BaseMultiLevelProgress
)


class ProgressWidget(QWidget):
    """A progress widget with cancel functionality integrated with ProgressManager."""
    
    cancelled = Signal()
    
    def __init__(self, 
                 progress_manager: Optional[ProgressManager] = None,
                 progress_id: Optional[str] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.progress_manager = progress_manager
        self.progress_id = progress_id
        self._setup_ui()
        self._is_cancelled = False
        
        # Connect to progress manager if provided
        if self.progress_manager and self.progress_id:
            self.progress_manager.progress_updated.connect(self._on_progress_updated)
            
    def _setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        self.title_label = QLabel("Processing...")
        self.title_label.setObjectName("progressTitle")
        layout.addWidget(self.title_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(25)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel()
        self.status_label.setObjectName("progressStatus")
        layout.addWidget(self.status_label)
        
        # Time estimates
        self.time_frame = QFrame()
        time_layout = QHBoxLayout(self.time_frame)
        
        self.elapsed_label = QLabel("Elapsed: --:--")
        time_layout.addWidget(self.elapsed_label)
        
        time_layout.addStretch()
        
        self.eta_label = QLabel("ETA: --:--")
        time_layout.addWidget(self.eta_label)
        
        layout.addWidget(self.time_frame)
        
        # Cancel button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        # Update timer for time display
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_time_display)
        self.update_timer.start(1000)  # Update every second
        
    def _on_cancel(self):
        """Handle cancel button click."""
        self._is_cancelled = True
        self.cancelled.emit()
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelling...")
        
        # Cancel in progress manager
        if self.progress_manager and self.progress_id:
            self.progress_manager.cancel_progress(self.progress_id)
            
    def _on_progress_updated(self, progress_id: str, progress_data: ProgressData):
        """Handle progress updates from manager."""
        if progress_id != self.progress_id:
            return
            
        self.set_progress(progress_data.current, progress_data.total)
        if progress_data.message:
            self.set_status(progress_data.message)
            
    def _update_time_display(self):
        """Update time display."""
        if not self.progress_manager or not self.progress_id:
            return
            
        progress = self.progress_manager.get_progress(self.progress_id)
        if not progress:
            return
            
        # Update elapsed time
        if progress.elapsed_time:
            elapsed = progress.elapsed_time
            self.elapsed_label.setText(f"Elapsed: {self._format_time(elapsed.total_seconds())}")
            
        # Update ETA
        if progress.estimated_remaining:
            remaining = progress.estimated_remaining
            self.eta_label.setText(f"ETA: {self._format_time(remaining.total_seconds())}")
        else:
            self.eta_label.setText("ETA: Calculating...")
            
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS or HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
        
    def set_title(self, title: str):
        """Set the progress title."""
        self.title_label.setText(title)
        
    def set_progress(self, value: int, maximum: int = 100):
        """Set the progress value."""
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        
    def set_status(self, status: str):
        """Set the status message."""
        self.status_label.setText(status)
        
    def is_cancelled(self) -> bool:
        """Check if the operation was cancelled."""
        return self._is_cancelled
        
    def reset(self):
        """Reset the progress widget."""
        self._is_cancelled = False
        self.progress_bar.setValue(0)
        self.status_label.clear()
        self.cancel_button.setEnabled(True)
        self.elapsed_label.setText("Elapsed: --:--")
        self.eta_label.setText("ETA: --:--")


class IndeterminateProgressWidget(QWidget):
    """An indeterminate progress widget for operations without known duration."""
    
    def __init__(self, 
                 use_circular: bool = False,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.use_circular = use_circular
        self._setup_ui()
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._animate)
        
    def _setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Message
        self.message_label = QLabel("Processing...")
        self.message_label.setObjectName("progressMessage")
        layout.addWidget(self.message_label, alignment=Qt.AlignCenter)
        
        # Progress indicator
        if self.use_circular:
            self.progress_indicator = BaseCircularProgress()
            self.progress_indicator.setIndeterminate(True)
            layout.addWidget(self.progress_indicator, alignment=Qt.AlignCenter)
        else:
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 0)  # Indeterminate mode
            self.progress_bar.setTextVisible(False)
            layout.addWidget(self.progress_bar)
        
    def start(self):
        """Start the indeterminate progress animation."""
        self._animation_timer.start(50)
        
    def stop(self):
        """Stop the indeterminate progress animation."""
        self._animation_timer.stop()
        
    def _animate(self):
        """Animate the progress indicator."""
        # The QProgressBar handles animation in indeterminate mode
        pass
        
    def set_message(self, message: str):
        """Set the progress message."""
        self.message_label.setText(message)


class BatchProgressWidget(BaseMultiLevelProgress):
    """A widget for showing batch operation progress with multi-level support."""
    
    def __init__(self, 
                 progress_manager: Optional[ProgressManager] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.progress_manager = progress_manager
        self._setup_extra_ui()
        
        # Connect to progress manager
        if self.progress_manager:
            self.progress_manager.progress_updated.connect(self._on_progress_updated)
            self.progress_manager.progress_started.connect(self._on_progress_started)
            self.progress_manager.progress_completed.connect(self._on_progress_completed)
            
    def _setup_extra_ui(self):
        """Set up additional UI elements."""
        # Add stats section
        stats_frame = QFrame()
        stats_layout = QHBoxLayout(stats_frame)
        
        self.items_label = QLabel("Items: 0/0")
        stats_layout.addWidget(self.items_label)
        
        stats_layout.addStretch()
        
        self.time_label = QLabel("Time: 00:00")
        stats_layout.addWidget(self.time_label)
        
        stats_layout.addStretch()
        
        self.speed_label = QLabel("Speed: 0 items/sec")
        stats_layout.addWidget(self.speed_label)
        
        self.layout.addWidget(stats_frame)
        
        # Timer for updating stats
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._update_stats)
        self.stats_timer.start(1000)
        
    def _on_progress_started(self, progress_id: str, progress_data: ProgressData):
        """Handle progress started event."""
        # Determine level from parent hierarchy
        level = 0
        parent_id = progress_data.parent_id
        while parent_id and self.progress_manager:
            level += 1
            parent_progress = self.progress_manager.get_progress(parent_id)
            if parent_progress:
                parent_id = parent_progress.parent_id
            else:
                break
                
        self.add_progress(progress_id, progress_data.name, level)
        
    def _on_progress_updated(self, progress_id: str, progress_data: ProgressData):
        """Handle progress update event."""
        self.update_progress(progress_id, progress_data.current, progress_data.total)
        
    def _on_progress_completed(self, progress_id: str, progress_data: ProgressData):
        """Handle progress completed event."""
        # Keep completed items visible for a short time
        QTimer.singleShot(2000, lambda: self.remove_progress(progress_id))
        
    def _update_stats(self):
        """Update statistics display."""
        if not self.progress_manager:
            return
            
        # Calculate overall stats
        all_progress = self.progress_manager.get_all_progress()
        active_progress = [p for p in all_progress.values() 
                          if p.state == ProgressState.RUNNING]
        
        if not active_progress:
            return
            
        # Find main progress (no parent)
        main_progress = next((p for p in active_progress if not p.parent_id), None)
        if main_progress:
            self.items_label.setText(f"Items: {main_progress.current}/{main_progress.total}")
            
            # Update time
            if main_progress.elapsed_time:
                elapsed = main_progress.elapsed_time.total_seconds()
                self.time_label.setText(f"Time: {self._format_time(elapsed)}")
                
                # Calculate speed
                if main_progress.current > 0:
                    speed = main_progress.current / elapsed
                    self.speed_label.setText(f"Speed: {speed:.1f} items/sec")
                    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
        
    def set_overall_progress(self, current: int, total: int):
        """Set the overall progress (compatibility method)."""
        if "overall" in self._progress_bars:
            self.update_progress("overall", current, total)
        else:
            self.add_progress("overall", "Overall Progress", 0)
            self.update_progress("overall", current, total)
        
    def set_current_progress(self, value: int, maximum: int = 100):
        """Set the current item progress (compatibility method)."""
        if "current" in self._progress_bars:
            self.update_progress("current", value, maximum)
            
    def set_current_item(self, name: str):
        """Set the current item name (compatibility method)."""
        if "current" not in self._progress_bars:
            self.add_progress("current", name, 1)
        else:
            # Update the label
            for i in range(self.layout.count()):
                widget = self.layout.itemAt(i).widget()
                if widget and hasattr(widget, 'findChild'):
                    label = widget.findChild(QLabel)
                    if label and label.text().startswith("Current"):
                        label.setText(name)
                        break
        
    def set_current_status(self, status: str):
        """Set the current item status (compatibility method)."""
        # This is now handled through progress messages
        pass
        
    def set_elapsed_time(self, seconds: int):
        """Set the elapsed time (compatibility method)."""
        self.time_label.setText(f"Time: {self._format_time(seconds)}")


class CircularProgressWidget(BaseCircularProgress):
    """A circular progress indicator widget with enhanced features."""
    
    def __init__(self, 
                 size: int = 100,
                 progress_manager: Optional[ProgressManager] = None,
                 progress_id: Optional[str] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.progress_manager = progress_manager
        self.progress_id = progress_id
        
        # Connect to progress manager if provided
        if self.progress_manager and self.progress_id:
            self.progress_manager.progress_updated.connect(self._on_progress_updated)
            
    def _on_progress_updated(self, progress_id: str, progress_data: ProgressData):
        """Handle progress updates from manager."""
        if progress_id != self.progress_id:
            return
            
        self.set_progress(progress_data.current, progress_data.total)
        
    def set_progress(self, value: int, maximum: int = 100):
        """Set the progress value."""
        self.setMaximum(maximum)
        self.setValue(value)
        
    @Property(int)
    def progress(self):
        """Get the progress value."""
        return self._value
        
    @progress.setter
    def progress(self, value: int):
        """Set the progress value."""
        self.setValue(value)