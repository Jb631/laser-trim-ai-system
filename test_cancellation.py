#!/usr/bin/env python3
"""
Test script to verify the cancellation functionality in analysis_page.py

This script tests:
1. Cancellation flag is properly set and reset
2. Processing thread stops when cancelled
3. UI updates properly during cancellation
4. Files show correct status after cancellation
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import tkinter as tk
from pathlib import Path
import threading
import time
import asyncio


class TestAnalysisCancellation(unittest.TestCase):
    """Test cancellation functionality in AnalysisPage"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a root window for testing
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window
        
        # Mock the main window
        self.mock_main_window = Mock()
        self.mock_main_window.db_manager = None
        
    def tearDown(self):
        """Clean up after tests"""
        self.root.destroy()
        
    @patch('src.laser_trim_analyzer.gui.pages.analysis_page.LaserTrimProcessor')
    def test_cancellation_flag_set(self, mock_processor_class):
        """Test that cancellation flag is properly set"""
        from src.laser_trim_analyzer.gui.pages.analysis_page import AnalysisPage
        
        # Create page instance
        page = AnalysisPage(self.root, self.mock_main_window)
        
        # Verify initial state
        self.assertFalse(page._cancel_requested)
        self.assertIsNone(page._processing_thread)
        
        # Call cancel method
        page._cancel_analysis()
        
        # Verify cancellation flag is set
        self.assertTrue(page._cancel_requested)
        
    @patch('src.laser_trim_analyzer.gui.pages.analysis_page.LaserTrimProcessor')
    def test_cancellation_during_processing(self, mock_processor_class):
        """Test cancellation during active processing"""
        from src.laser_trim_analyzer.gui.pages.analysis_page import AnalysisPage
        
        # Create page instance
        page = AnalysisPage(self.root, self.mock_main_window)
        
        # Mock a running thread
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        page._processing_thread = mock_thread
        page.is_processing = True
        
        # Mock progress label
        page.progress_label = Mock()
        
        # Call cancel method
        with patch.object(page, '_handle_cancellation_cleanup'):
            page._cancel_analysis()
        
        # Verify cancellation was initiated
        self.assertTrue(page._cancel_requested)
        page.progress_label.config.assert_called_with(text="Cancelling analysis...")
        
    @patch('src.laser_trim_analyzer.gui.pages.analysis_page.LaserTrimProcessor')
    async def test_async_processing_checks_cancellation(self, mock_processor_class):
        """Test that async processing checks cancellation flag"""
        from src.laser_trim_analyzer.gui.pages.analysis_page import AnalysisPage
        
        # Create page instance
        page = AnalysisPage(self.root, self.mock_main_window)
        
        # Set up test files
        page.input_files = [Path("test1.xls"), Path("test2.xls")]
        
        # Mock processor
        mock_processor = AsyncMock()
        page.processor = mock_processor
        
        # Set cancellation flag before processing
        page._cancel_requested = True
        
        # Try to process files
        with self.assertRaises(asyncio.CancelledError):
            await page._process_files_async_responsive()
            
    @patch('src.laser_trim_analyzer.gui.pages.analysis_page.LaserTrimProcessor')
    def test_finalize_cancellation(self, mock_processor_class):
        """Test finalization of cancellation"""
        from src.laser_trim_analyzer.gui.pages.analysis_page import AnalysisPage
        
        # Create page instance
        page = AnalysisPage(self.root, self.mock_main_window)
        
        # Set up initial state
        page.is_processing = True
        page._cancel_requested = True
        page._processing_thread = Mock()
        
        # Mock UI components
        page.progress_frame = Mock()
        page.alert_stack = Mock()
        page._update_ui_state = Mock()
        
        # Set up test files
        test_file = Path("test.xls")
        page.input_files = [test_file]
        page._processing_results = {str(test_file): {'status': 'Processing'}}
        page.file_widgets = {str(test_file): Mock()}
        
        # Call finalize cancellation
        page._finalize_cancellation()
        
        # Verify state is reset
        self.assertFalse(page.is_processing)
        self.assertFalse(page._cancel_requested)
        self.assertIsNone(page._processing_thread)
        
        # Verify UI was updated
        page._update_ui_state.assert_called_once()
        page.alert_stack.add_alert.assert_called_once()
        
    @patch('src.laser_trim_analyzer.gui.pages.analysis_page.LaserTrimProcessor')
    def test_start_analysis_resets_cancellation(self, mock_processor_class):
        """Test that starting analysis resets cancellation flag"""
        from src.laser_trim_analyzer.gui.pages.analysis_page import AnalysisPage
        
        # Create page instance
        page = AnalysisPage(self.root, self.mock_main_window)
        
        # Set up test state
        page._cancel_requested = True  # Previously cancelled
        page.input_files = [Path("test.xls")]
        
        # Mock UI components
        page.alert_stack = Mock()
        page.progress_frame = Mock()
        page.progress_var = Mock()
        page.progress_label = Mock()
        page._update_ui_state = Mock()
        page._clear_non_critical_alerts = Mock()
        page._ensure_files_visible_responsive = Mock()
        page._schedule_responsiveness_checks = Mock()
        
        # Start analysis
        with patch('threading.Thread'):
            page._start_analysis()
        
        # Verify cancellation flag was reset
        self.assertFalse(page._cancel_requested)
        

def run_tests():
    """Run all cancellation tests"""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAnalysisCancellation)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)