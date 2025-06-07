"""
Test Progress System

Comprehensive tests for the progress indicator system including all features:
- Multiple progress types
- Multi-level tracking
- Time estimation
- Cancellation
- Persistence
- Integration
"""

import pytest
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtTest import QTest

from src.laser_trim_analyzer.gui.progress_system import (
    ProgressManager, ProgressData, ProgressType, ProgressState,
    CircularProgressWidget, MultiLevelProgressWidget,
    ProgressDialog, ProgressTracker, ThreadedProgressUpdater
)
from src.laser_trim_analyzer.gui.progress_integration import (
    FileProcessorWithProgress, BatchProcessorWithProgress,
    AnalysisEngineWithProgress, ResumableOperation
)


@pytest.fixture
def app(qtbot):
    """Create QApplication for tests."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def progress_manager():
    """Create a progress manager instance."""
    return ProgressManager()


class TestProgressData:
    """Test ProgressData class."""
    
    def test_progress_data_creation(self):
        """Test creating progress data."""
        data = ProgressData(
            id="test_1",
            name="Test Progress",
            current=50,
            total=100,
            message="Processing..."
        )
        
        assert data.id == "test_1"
        assert data.name == "Test Progress"
        assert data.current == 50
        assert data.total == 100
        assert data.message == "Processing..."
        assert data.state == ProgressState.IDLE
        assert data.progress_percent == 50.0
        
    def test_elapsed_time_calculation(self):
        """Test elapsed time calculation."""
        data = ProgressData(id="test", name="Test")
        data.start_time = datetime.now() - timedelta(seconds=10)
        
        elapsed = data.elapsed_time
        assert elapsed is not None
        assert 9 <= elapsed.total_seconds() <= 11  # Allow for small timing variations
        
    def test_estimated_remaining_calculation(self):
        """Test remaining time estimation."""
        data = ProgressData(id="test", name="Test", current=25, total=100)
        data.start_time = datetime.now() - timedelta(seconds=10)
        
        remaining = data.estimated_remaining
        assert remaining is not None
        # Should estimate ~30 seconds remaining (75 items at 2.5 items/second)
        assert 25 <= remaining.total_seconds() <= 35
        
    def test_serialization(self):
        """Test serialization to/from dict."""
        data = ProgressData(
            id="test",
            name="Test",
            current=50,
            total=100,
            state=ProgressState.RUNNING,
            start_time=datetime.now(),
            metadata={'key': 'value'}
        )
        
        # Convert to dict
        data_dict = data.to_dict()
        assert isinstance(data_dict['start_time'], str)
        assert data_dict['state'] == 'running'
        
        # Convert back
        restored = ProgressData.from_dict(data_dict)
        assert restored.id == data.id
        assert restored.state == ProgressState.RUNNING
        assert isinstance(restored.start_time, datetime)


class TestProgressManager:
    """Test ProgressManager class."""
    
    def test_create_progress(self, progress_manager):
        """Test creating progress entries."""
        progress = progress_manager.create_progress(
            "test_1",
            "Test Progress",
            100,
            metadata={'test': True}
        )
        
        assert progress.id == "test_1"
        assert progress.name == "Test Progress"
        assert progress.total == 100
        assert progress.metadata['test'] is True
        
        # Verify it's stored
        retrieved = progress_manager.get_progress("test_1")
        assert retrieved is not None
        assert retrieved.id == progress.id
        
    def test_progress_hierarchy(self, progress_manager):
        """Test parent-child progress relationships."""
        # Create parent
        parent = progress_manager.create_progress("parent", "Parent", 100)
        
        # Create children
        child1 = progress_manager.create_progress("child1", "Child 1", 50, parent_id="parent")
        child2 = progress_manager.create_progress("child2", "Child 2", 50, parent_id="parent")
        
        # Verify relationships
        assert child1.parent_id == "parent"
        assert child2.parent_id == "parent"
        assert "child1" in parent.children
        assert "child2" in parent.children
        
    def test_progress_updates(self, progress_manager, qtbot):
        """Test progress updates and signals."""
        # Create progress
        progress_manager.create_progress("test", "Test", 100)
        progress_manager.start_progress("test")
        
        # Connect signal spy
        with qtbot.waitSignal(progress_manager.progress_updated) as blocker:
            progress_manager.update_progress("test", current=50, message="Half way")
            
        # Verify signal data
        assert blocker.args[0] == "test"
        assert blocker.args[1].current == 50
        assert blocker.args[1].message == "Half way"
        
    def test_parent_progress_aggregation(self, progress_manager):
        """Test parent progress calculation from children."""
        # Create hierarchy
        progress_manager.create_progress("parent", "Parent", 100)
        progress_manager.create_progress("child1", "Child 1", 100, parent_id="parent")
        progress_manager.create_progress("child2", "Child 2", 100, parent_id="parent")
        
        # Update children
        progress_manager.update_progress("child1", current=50)
        progress_manager.update_progress("child2", current=75)
        
        # Check parent progress
        parent = progress_manager.get_progress("parent")
        # Parent should be at (50 + 75) / 2 = 62.5% = 62/100
        assert parent.current == 62
        
    def test_progress_cancellation(self, progress_manager, qtbot):
        """Test progress cancellation."""
        # Create and start progress
        progress_manager.create_progress("test", "Test", 100)
        progress_manager.start_progress("test")
        
        # Cancel with signal spy
        with qtbot.waitSignal(progress_manager.progress_cancelled) as blocker:
            progress_manager.cancel_progress("test")
            
        # Verify state
        progress = progress_manager.get_progress("test")
        assert progress.state == ProgressState.CANCELLED
        assert blocker.args[0] == "test"
        
    def test_progress_persistence(self, progress_manager, tmp_path):
        """Test saving and loading progress state."""
        # Override persistence file
        progress_manager._persistence_file = tmp_path / "test_progress.json"
        
        # Create resumable progress
        progress_manager.create_progress(
            "resumable",
            "Resumable Task",
            100,
            metadata={'resumable': True}
        )
        progress_manager.update_progress("resumable", current=50)
        
        # Save state
        progress_manager._save_state()
        
        # Create new manager and load
        new_manager = ProgressManager()
        new_manager._persistence_file = tmp_path / "test_progress.json"
        new_manager._load_state()
        
        # Verify loaded progress
        loaded = new_manager.get_progress("resumable")
        assert loaded is not None
        assert loaded.current == 50
        assert loaded.total == 100
        assert loaded.metadata['resumable'] is True


class TestProgressWidgets:
    """Test progress widget classes."""
    
    def test_circular_progress_widget(self, qtbot):
        """Test circular progress widget."""
        widget = CircularProgressWidget()
        qtbot.addWidget(widget)
        
        # Test normal mode
        widget.setValue(50)
        widget.setMaximum(100)
        assert widget._value == 50
        assert widget._max_value == 100
        
        # Test indeterminate mode
        widget.setIndeterminate(True)
        assert widget._indeterminate is True
        qtbot.wait(100)  # Let animation run
        
        widget.setIndeterminate(False)
        assert widget._rotation_timer.isActive() is False
        
    def test_multi_level_progress_widget(self, qtbot):
        """Test multi-level progress widget."""
        widget = MultiLevelProgressWidget()
        qtbot.addWidget(widget)
        
        # Add progress bars
        widget.add_progress("main", "Main Task", 0)
        widget.add_progress("sub1", "Subtask 1", 1)
        widget.add_progress("sub2", "Subtask 2", 1)
        
        # Update progress
        widget.update_progress("main", 50, 100)
        widget.update_progress("sub1", 25, 50)
        widget.update_progress("sub2", 10, 20)
        
        # Verify progress bars exist
        assert "main" in widget._progress_bars
        assert "sub1" in widget._progress_bars
        assert "sub2" in widget._progress_bars
        
        # Remove progress
        widget.remove_progress("sub1")
        assert "sub1" not in widget._progress_bars


class TestProgressDialog:
    """Test progress dialog."""
    
    def test_progress_dialog_creation(self, qtbot):
        """Test creating progress dialog."""
        dialog = ProgressDialog(
            "Test Dialog",
            "Testing progress...",
            ProgressType.LINEAR,
            cancellable=True
        )
        qtbot.addWidget(dialog)
        
        assert dialog.windowTitle() == "Test Dialog"
        assert dialog.message_label.text() == "Testing progress..."
        assert dialog._cancellable is True
        
    def test_dialog_cancellation(self, qtbot):
        """Test dialog cancellation."""
        dialog = ProgressDialog("Test", cancellable=True)
        qtbot.addWidget(dialog)
        
        # Click cancel with signal spy
        with qtbot.waitSignal(dialog.cancelled) as blocker:
            QTest.mouseClick(dialog.cancel_button, Qt.LeftButton)
            
        assert dialog.is_cancelled() is True
        
    def test_dialog_progress_updates(self, qtbot):
        """Test updating dialog progress."""
        dialog = ProgressDialog("Test", progress_type=ProgressType.LINEAR)
        qtbot.addWidget(dialog)
        
        # Update progress
        dialog.set_progress(50, 100)
        assert dialog.progress_widget.value() == 50
        
        # Update message
        dialog.set_message("New message")
        assert dialog.message_label.text() == "New message"
        
        # Append details
        dialog.append_details("Detail 1")
        dialog.append_details("Detail 2")
        assert "Detail 1" in dialog.details_text.toPlainText()
        assert "Detail 2" in dialog.details_text.toPlainText()


class TestProgressTracker:
    """Test progress tracker context manager."""
    
    def test_progress_tracker_basic(self, progress_manager):
        """Test basic progress tracker usage."""
        with ProgressTracker(
            progress_manager,
            "test_op",
            "Test Operation",
            100
        ) as tracker:
            # Progress should be created and started
            progress = progress_manager.get_progress("test_op")
            assert progress is not None
            assert progress.state == ProgressState.RUNNING
            
            # Update progress
            tracker.update(50, "Half way")
            progress = progress_manager.get_progress("test_op")
            assert progress.current == 50
            assert progress.message == "Half way"
            
        # Progress should be completed
        progress = progress_manager.get_progress("test_op")
        assert progress.state == ProgressState.COMPLETED
        
    def test_progress_tracker_with_error(self, progress_manager):
        """Test progress tracker with error."""
        try:
            with ProgressTracker(
                progress_manager,
                "test_error",
                "Test Error",
                100
            ) as tracker:
                tracker.update(50)
                raise ValueError("Test error")
        except ValueError:
            pass
            
        # Progress should be in error state
        progress = progress_manager.get_progress("test_error")
        assert progress.state == ProgressState.ERROR
        assert "Test error" in progress.metadata.get('error', '')
        
    def test_progress_tracker_cancellation(self, progress_manager):
        """Test progress tracker cancellation."""
        with ProgressTracker(
            progress_manager,
            "test_cancel",
            "Test Cancel",
            100,
            cancellable=True
        ) as tracker:
            tracker.update(25)
            tracker._on_cancelled()  # Simulate cancellation
            
        # Progress should be cancelled
        progress = progress_manager.get_progress("test_cancel")
        assert progress.state == ProgressState.CANCELLED


class TestProgressIntegration:
    """Test integration classes."""
    
    @patch('src.laser_trim_analyzer.core.processor.LaserTrimProcessor')
    def test_file_processor_integration(self, mock_processor, progress_manager, qtbot):
        """Test file processor with progress."""
        # Setup mock
        mock_result = MagicMock()
        mock_processor.return_value.process_file.return_value = mock_result
        
        # Create processor
        processor = FileProcessorWithProgress(progress_manager)
        
        # Process file
        result = processor.process_file(
            Path("test.xls"),
            show_dialog=False
        )
        
        assert result == mock_result
        
    def test_batch_processor_integration(self, progress_manager, qtbot):
        """Test batch processor with progress."""
        files = [Path(f"test{i}.xls") for i in range(3)]
        
        # Create batch processor
        batch_processor = BatchProcessorWithProgress(
            files,
            progress_manager
        )
        
        # Mock processor
        batch_processor.processor = MagicMock()
        batch_processor.processor.process_file.return_value = MagicMock()
        
        # Connect signal spy
        with qtbot.waitSignals(
            [batch_processor.batch_completed],
            timeout=5000
        ):
            batch_processor.start()
            batch_processor.wait()
            
        # Verify all files processed
        assert batch_processor.processor.process_file.call_count == 3
        
    def test_resumable_operation(self, progress_manager, tmp_path):
        """Test resumable operation."""
        operation = ResumableOperation(progress_manager)
        operation.state_file = tmp_path / "test_state.json"
        
        items = list(range(10))
        
        # Start operation
        operation.long_running_operation(items, "test_resumable")
        
        # Simulate interruption after 5 items
        progress = progress_manager.get_progress("test_resumable")
        progress.current = 5
        progress.state = ProgressState.PAUSED
        
        # Resume operation
        operation.long_running_operation(items, "test_resumable")
        
        # Should complete from where it left off
        progress = progress_manager.get_progress("test_resumable")
        assert progress.state == ProgressState.COMPLETED
        assert progress.current == len(items)


class TestThreadedProgressUpdater:
    """Test threaded progress updater."""
    
    def test_threaded_updater(self, progress_manager, qtbot):
        """Test threaded progress updates."""
        # Create active progress
        progress_manager.create_progress("test", "Test", 100)
        progress_manager.start_progress("test")
        
        # Create updater
        updater = ThreadedProgressUpdater(progress_manager)
        
        # Connect signal spy
        signal_received = False
        def on_update(pid, current, total, message):
            nonlocal signal_received
            if pid == "test":
                signal_received = True
                
        updater.progress_update.connect(on_update)
        
        # Start updater
        updater.start()
        qtbot.wait(200)  # Wait for at least one update
        
        # Stop updater
        updater.stop()
        
        assert signal_received is True


def test_progress_system_example(progress_manager, qtbot):
    """Test complete progress system example."""
    # Simulate a complex operation with nested progress
    
    # Main operation
    with ProgressTracker(
        progress_manager,
        "main_op",
        "Main Operation",
        3,
        show_dialog=False
    ) as main_tracker:
        
        # Sub-operation 1
        with ProgressTracker(
            progress_manager,
            "sub_op_1",
            "Loading Data",
            100,
            parent_id="main_op"
        ) as sub1:
            for i in range(100):
                sub1.update(i)
                if i % 20 == 0:
                    qtbot.wait(10)  # Simulate work
                    
        main_tracker.update(1, "Data loaded")
        
        # Sub-operation 2
        with ProgressTracker(
            progress_manager,
            "sub_op_2",
            "Processing Data",
            50,
            parent_id="main_op"
        ) as sub2:
            for i in range(50):
                sub2.update(i, f"Processing item {i}")
                if i % 10 == 0:
                    qtbot.wait(10)
                    
        main_tracker.update(2, "Data processed")
        
        # Sub-operation 3
        with ProgressTracker(
            progress_manager,
            "sub_op_3",
            "Saving Results",
            10,
            parent_id="main_op"
        ) as sub3:
            for i in range(10):
                sub3.update(i)
                qtbot.wait(10)
                
        main_tracker.update(3, "Complete!")
        
    # Verify all completed
    for pid in ["main_op", "sub_op_1", "sub_op_2", "sub_op_3"]:
        progress = progress_manager.get_progress(pid)
        assert progress.state == ProgressState.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])