"""
Batch Processing Page Module

This module contains the BatchProcessingPage and its supporting mixins.
The implementation is split into manageable modules:
- batch_processing_page.py: Main page class with UI creation
- export_mixin.py: Export functionality (Excel, CSV, HTML)
- processing_mixin.py: Batch processing logic (future)
"""

from laser_trim_analyzer.gui.pages.batch_processing_page import BatchProcessingPage

__all__ = ['BatchProcessingPage']
