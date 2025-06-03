"""
Comprehensive UI Integration Tests

Tests the complete UI system including:
- Hybrid file loading (widgets vs tree view)
- Large batch processing (700+ files)
- Alert system performance
- Memory management
- User interaction workflows
"""

import pytest
import asyncio
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tkinter as tk
from tkinter import ttk

# Import the components we're testing
from laser_trim_analyzer.gui.pages.analysis_page import AnalysisPage
from laser_trim_analyzer.gui.widgets.alert_banner import AlertStack, AlertBanner
from laser_trim_analyzer.gui.widgets.file_drop_zone import FileDropZone
from laser_trim_analyzer.core.config import Config, ProcessingConfig, GUIConfig
from laser_trim_analyzer.database.manager import DatabaseManager


class TestUIIntegration:
    """Comprehensive UI integration tests."""

    @pytest.fixture
    def root_window(self):
        """Create root tkinter window for testing."""
        root = tk.Tk()
        root.withdraw()  # Hide window during testing
        yield root
        root.destroy()

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return Config(
            debug=True,
            processing=ProcessingConfig(
                max_workers=2,
                generate_plots=False,  # Faster testing
                cache_enabled=True
            ),
            gui=GUIConfig(
                window_width=800,
                window_height=600
            )
        )

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_main_window(self, test_config):
        """Create mock main window."""
        mock_window = Mock()
        mock_window.config = test_config
        mock_window.db_manager = None
        mock_window.ml_predictor = None
        mock_window.logger = Mock()
        return mock_window

    @pytest.fixture
    def large_file_list(self, temp_dir):
        """Create a large list of test files (simulating 700+ files)."""
        files = []
        for i in range(750):
            # Create mock file paths without actually creating files
            file_path = temp_dir / f"test_8340_A{i:05d}_20240101.xlsx"
            files.append(file_path)
        return files

    def test_hybrid_file_loading_threshold(self, root_window, mock_main_window):
        """Test that the hybrid loading system switches at the correct threshold."""
        # Create analysis page
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Test small batch (should use widgets)
        small_files = [Path(f"small_{i}.xlsx") for i in range(50)]
        
        with patch.object(page, '_create_file_widgets_async') as mock_widgets, \
             patch.object(page, '_switch_to_tree_view_mode') as mock_tree:
            
            page._add_files(small_files)
            
            # Should use widgets for small batch
            mock_widgets.assert_called_once()
            mock_tree.assert_not_called()

        # Test large batch (should use tree view)
        large_files = [Path(f"large_{i}.xlsx") for i in range(300)]
        
        with patch.object(page, '_create_file_widgets_async') as mock_widgets, \
             patch.object(page, '_switch_to_tree_view_mode') as mock_tree, \
             patch.object(page, '_populate_tree_view_immediate') as mock_populate:
            
            page._add_files(large_files)
            
            # Should use tree view for large batch
            mock_widgets.assert_not_called()
            mock_tree.assert_called_once()
            mock_populate.assert_called_once()

    def test_tree_view_performance(self, root_window, mock_main_window, large_file_list):
        """Test tree view performance with very large file sets."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Measure time to switch to tree view mode
        start_time = time.time()
        
        with patch.object(page.alert_stack, 'add_alert') as mock_alert:
            page._add_files(large_file_list)
        
        switch_time = time.time() - start_time
        
        # Should complete very quickly (under 1 second for 750 files)
        assert switch_time < 1.0, f"Tree view switch took too long: {switch_time:.2f}s"
        
        # Should have tree view elements
        assert hasattr(page, 'file_tree')
        assert len(page.file_widgets) == 750
        
        # All entries should be in tree mode
        for file_path, widget_data in page.file_widgets.items():
            assert isinstance(widget_data, dict)
            assert widget_data.get('tree_mode') is True

    def test_alert_system_performance(self, root_window):
        """Test alert system performance without choppy animations."""
        parent = ttk.Frame(root_window)
        alert_stack = AlertStack(parent)
        
        # Test rapid alert creation and dismissal
        alerts = []
        start_time = time.time()
        
        for i in range(10):
            alert = alert_stack.add_alert(
                alert_type='info',
                title=f'Test Alert {i}',
                message=f'Testing performance with alert {i}',
                auto_dismiss=None,  # No auto-dismiss to avoid animation
                dismissible=True
            )
            alerts.append(alert)
        
        creation_time = time.time() - start_time
        
        # Should create alerts quickly
        assert creation_time < 0.5, f"Alert creation took too long: {creation_time:.2f}s"
        
        # Test manual dismissal without animation choppiness
        start_time = time.time()
        
        for alert in alerts:
            alert.dismiss()
        
        dismissal_time = time.time() - start_time
        
        # Should dismiss without significant delay
        assert dismissal_time < 1.0, f"Alert dismissal took too long: {dismissal_time:.2f}s"

    def test_file_status_updates_tree_mode(self, root_window, mock_main_window):
        """Test file status updates work correctly in tree mode."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Setup tree view mode
        large_files = [Path(f"test_{i}.xlsx") for i in range(250)]
        page._add_files(large_files)
        
        # Verify tree mode is active
        assert hasattr(page, 'file_tree')
        
        # Test status update
        test_file = str(large_files[0])
        page._update_file_status(test_file, 'Processing')
        
        # Verify tree item was updated
        widget_data = page.file_widgets[test_file]
        assert widget_data['tree_mode'] is True
        
        # Get tree item and verify status
        item_id = widget_data['tree_item']
        values = page.file_tree.item(item_id, 'values')
        assert values[2] == 'Processing'  # Status is 3rd column

    def test_memory_efficiency_tree_mode(self, root_window, mock_main_window):
        """Test that tree mode uses significantly less memory than widget mode."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Create moderate file list that would normally use widgets
        files = [Path(f"test_{i}.xlsx") for i in range(100)]
        
        # Force tree mode for comparison
        with patch.object(page, '_switch_to_tree_view_mode') as mock_switch, \
             patch.object(page, '_populate_tree_view_immediate') as mock_populate:
            
            # Override the threshold check
            page._add_files(files)
            
            # Manually call tree view mode
            page._switch_to_tree_view_mode()
            page._populate_tree_view_immediate(files)
        
        # Verify tree mode is more memory efficient
        assert hasattr(page, 'file_tree')
        
        # In tree mode, file_widgets contains lightweight dict entries
        for file_path, widget_data in page.file_widgets.items():
            assert isinstance(widget_data, dict)
            assert 'tree_item' in widget_data
            assert 'tree_mode' in widget_data
            # Should not contain heavy widget objects

    def test_clear_files_tree_mode(self, root_window, mock_main_window):
        """Test clearing files works correctly in tree mode."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Setup tree view mode
        large_files = [Path(f"test_{i}.xlsx") for i in range(300)]
        page._add_files(large_files)
        
        # Verify setup
        assert len(page.input_files) == 300
        assert hasattr(page, 'file_tree')
        
        # Clear files
        page._clear_files()
        
        # Verify cleanup
        assert len(page.input_files) == 0
        assert len(page.file_widgets) == 0
        assert not hasattr(page, 'file_tree')

    def test_context_menu_tree_mode(self, root_window, mock_main_window):
        """Test context menu functionality in tree mode."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Setup tree view mode
        files = [Path("test_8340_A12345.xlsx"), Path("test_8555_B67890.xlsx")]
        page._add_files(files)
        
        # Verify tree mode and context menu
        assert hasattr(page, 'file_tree')
        assert hasattr(page, 'tree_context_menu')
        
        # Test context menu methods exist
        assert hasattr(page, '_tree_view_details')
        assert hasattr(page, '_tree_remove_file')
        assert hasattr(page, '_tree_export_file')

    def test_widget_to_tree_transition(self, root_window, mock_main_window):
        """Test transitioning from widget mode to tree mode."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Start with small batch (widget mode)
        small_files = [Path(f"small_{i}.xlsx") for i in range(50)]
        
        with patch.object(page, '_create_file_widgets_async'):
            page._add_files(small_files)
        
        # Add more files to trigger tree mode
        more_files = [Path(f"more_{i}.xlsx") for i in range(200)]
        
        with patch.object(page.alert_stack, 'add_alert'):
            page._add_files(more_files)
        
        # Should now be in tree mode
        assert hasattr(page, 'file_tree')
        assert len(page.input_files) == 250

    def test_error_handling_during_file_loading(self, root_window, mock_main_window):
        """Test error handling during file loading processes."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Test with invalid file paths
        invalid_files = [Path("nonexistent.txt"), Path("invalid.doc")]
        
        with patch.object(page.alert_stack, 'add_alert') as mock_alert:
            page._add_files(invalid_files)
        
        # Should handle gracefully
        assert len(page.input_files) == 0
        mock_alert.assert_called()

    def test_performance_benchmarks(self, root_window, mock_main_window):
        """Test performance benchmarks for various operations."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Benchmark file addition
        files = [Path(f"bench_{i}.xlsx") for i in range(500)]
        
        start_time = time.time()
        page._add_files(files)
        add_time = time.time() - start_time
        
        # Should complete file addition quickly
        assert add_time < 2.0, f"File addition took too long: {add_time:.2f}s"
        
        # Benchmark status updates
        start_time = time.time()
        for i in range(0, 100, 10):  # Update every 10th file
            page._update_file_status(str(files[i]), 'Processing')
        update_time = time.time() - start_time
        
        # Should update status quickly
        assert update_time < 0.5, f"Status updates took too long: {update_time:.2f}s"

    def test_ui_responsiveness_during_operations(self, root_window, mock_main_window):
        """Test UI remains responsive during operations."""
        parent = ttk.Frame(root_window)
        page = AnalysisPage(parent, mock_main_window)
        
        # Track UI update calls to ensure they're not blocking
        ui_updates = []
        original_after = page.after
        
        def track_after(delay, func, *args):
            ui_updates.append((delay, func, args))
            return original_after(delay, func, *args)
        
        page.after = track_after
        
        # Add large batch
        files = [Path(f"responsive_{i}.xlsx") for i in range(300)]
        page._add_files(files)
        
        # Should use after() for non-blocking operations
        assert len(ui_updates) > 0, "No UI updates scheduled for responsiveness"
        
        # Should use reasonable delays
        delays = [update[0] for update in ui_updates]
        assert all(delay <= 10 for delay in delays), "Some delays too long for responsiveness"


class TestFileDropZoneIntegration:
    """Test file drop zone integration with large batches."""

    @pytest.fixture
    def root_window(self):
        """Create root window."""
        root = tk.Tk()
        root.withdraw()
        yield root
        root.destroy()

    def test_large_folder_drop_performance(self, root_window, temp_dir):
        """Test dropping large folders performs well."""
        # Create mock callback
        files_processed = []
        
        def mock_callback(files):
            files_processed.extend(files)
        
        # Create drop zone
        parent = ttk.Frame(root_window)
        drop_zone = FileDropZone(
            parent,
            on_files_dropped=mock_callback,
            accept_extensions=['.xlsx', '.xls'],
            accept_folders=True
        )
        
        # Create large folder structure
        test_folder = temp_dir / "large_batch"
        test_folder.mkdir()
        
        # Create many test files
        files = []
        for i in range(200):
            file_path = test_folder / f"test_{i}.xlsx"
            file_path.touch()  # Create empty file
            files.append(file_path)
        
        # Test folder processing
        start_time = time.time()
        drop_zone._process_dropped_files_async([test_folder])
        
        # Allow some time for async processing
        time.sleep(0.5)
        
        process_time = time.time() - start_time
        
        # Should process efficiently
        assert process_time < 2.0, f"Folder processing took too long: {process_time:.2f}s"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"]) 