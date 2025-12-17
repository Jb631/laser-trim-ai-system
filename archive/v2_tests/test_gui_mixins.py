"""
Tests for GUI mixin modules (Phase 4/5 file splitting).

These tests verify:
1. Mixin module imports work correctly
2. Mixin classes have expected methods
3. Mixin method signatures are correct
4. Pages correctly use mixins

Note: These tests avoid instantiating GUI components to prevent GUI windows opening.
"""
import pytest
import sys
import inspect
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestBatchProcessingMixinImports:
    """Test batch processing mixin imports."""

    def test_import_processing_mixin(self):
        """Test ProcessingMixin can be imported."""
        from laser_trim_analyzer.gui.pages.batch.processing_mixin import ProcessingMixin
        assert ProcessingMixin is not None

    def test_import_export_mixin(self):
        """Test ExportMixin can be imported from batch."""
        from laser_trim_analyzer.gui.pages.batch.export_mixin import ExportMixin
        assert ExportMixin is not None


class TestHistoricalMixinImports:
    """Test historical page mixin imports."""

    def test_import_spc_mixin(self):
        """Test SPCMixin can be imported."""
        from laser_trim_analyzer.gui.pages.historical.spc_mixin import SPCMixin
        assert SPCMixin is not None

    def test_import_analytics_mixin(self):
        """Test AnalyticsMixin can be imported."""
        from laser_trim_analyzer.gui.pages.historical.analytics_mixin import AnalyticsMixin
        assert AnalyticsMixin is not None


class TestMultiTrackMixinImports:
    """Test multi-track page mixin imports."""

    def test_import_analysis_mixin(self):
        """Test AnalysisMixin can be imported."""
        from laser_trim_analyzer.gui.pages.multi_track.analysis_mixin import AnalysisMixin
        assert AnalysisMixin is not None

    def test_import_multi_track_export_mixin(self):
        """Test ExportMixin can be imported from multi_track."""
        from laser_trim_analyzer.gui.pages.multi_track.export_mixin import ExportMixin
        assert ExportMixin is not None


class TestProcessingMixinMethods:
    """Test ProcessingMixin has expected methods."""

    def test_has_start_processing(self):
        """Test ProcessingMixin has _start_processing method."""
        from laser_trim_analyzer.gui.pages.batch.processing_mixin import ProcessingMixin
        assert hasattr(ProcessingMixin, '_start_processing')

    def test_has_run_batch_processing(self):
        """Test ProcessingMixin has _run_batch_processing method."""
        from laser_trim_analyzer.gui.pages.batch.processing_mixin import ProcessingMixin
        assert hasattr(ProcessingMixin, '_run_batch_processing')

    def test_has_process_with_memory_management(self):
        """Test ProcessingMixin has _process_with_memory_management method."""
        from laser_trim_analyzer.gui.pages.batch.processing_mixin import ProcessingMixin
        assert hasattr(ProcessingMixin, '_process_with_memory_management')

    def test_has_process_with_turbo_mode(self):
        """Test ProcessingMixin has _process_with_turbo_mode method."""
        from laser_trim_analyzer.gui.pages.batch.processing_mixin import ProcessingMixin
        assert hasattr(ProcessingMixin, '_process_with_turbo_mode')

    def test_has_process_single_file_safe(self):
        """Test ProcessingMixin has _process_single_file_safe method."""
        from laser_trim_analyzer.gui.pages.batch.processing_mixin import ProcessingMixin
        assert hasattr(ProcessingMixin, '_process_single_file_safe')

    def test_has_handle_batch_cancelled(self):
        """Test ProcessingMixin has _handle_batch_cancelled method."""
        from laser_trim_analyzer.gui.pages.batch.processing_mixin import ProcessingMixin
        assert hasattr(ProcessingMixin, '_handle_batch_cancelled')


class TestBatchExportMixinMethods:
    """Test batch ExportMixin has expected methods."""

    def test_has_export_batch_results(self):
        """Test ExportMixin has _export_batch_results method."""
        from laser_trim_analyzer.gui.pages.batch.export_mixin import ExportMixin
        assert hasattr(ExportMixin, '_export_batch_results')

    def test_has_export_batch_excel(self):
        """Test ExportMixin has _export_batch_excel method."""
        from laser_trim_analyzer.gui.pages.batch.export_mixin import ExportMixin
        assert hasattr(ExportMixin, '_export_batch_excel')

    def test_has_export_batch_html(self):
        """Test ExportMixin has _export_batch_html method."""
        from laser_trim_analyzer.gui.pages.batch.export_mixin import ExportMixin
        assert hasattr(ExportMixin, '_export_batch_html')

    def test_has_export_batch_csv(self):
        """Test ExportMixin has _export_batch_csv method."""
        from laser_trim_analyzer.gui.pages.batch.export_mixin import ExportMixin
        assert hasattr(ExportMixin, '_export_batch_csv')

    def test_has_export_batch_excel_legacy(self):
        """Test ExportMixin has _export_batch_excel_legacy fallback method."""
        from laser_trim_analyzer.gui.pages.batch.export_mixin import ExportMixin
        assert hasattr(ExportMixin, '_export_batch_excel_legacy')


class TestSPCMixinMethods:
    """Test SPCMixin has expected methods."""

    def test_spc_mixin_has_methods(self):
        """Test SPCMixin has SPC-related methods."""
        from laser_trim_analyzer.gui.pages.historical.spc_mixin import SPCMixin
        # Get all methods
        methods = [m for m in dir(SPCMixin) if not m.startswith('__')]
        # Should have at least some methods
        assert len(methods) > 0


class TestAnalyticsMixinMethods:
    """Test historical AnalyticsMixin has expected methods."""

    def test_analytics_mixin_has_methods(self):
        """Test AnalyticsMixin has analytics-related methods."""
        from laser_trim_analyzer.gui.pages.historical.analytics_mixin import AnalyticsMixin
        # Get all methods
        methods = [m for m in dir(AnalyticsMixin) if not m.startswith('__')]
        # Should have at least some methods
        assert len(methods) > 0


class TestMultiTrackAnalysisMixinMethods:
    """Test multi-track AnalysisMixin has expected methods."""

    def test_analysis_mixin_has_methods(self):
        """Test AnalysisMixin has analysis-related methods."""
        from laser_trim_analyzer.gui.pages.multi_track.analysis_mixin import AnalysisMixin
        # Get all methods
        methods = [m for m in dir(AnalysisMixin) if not m.startswith('__')]
        # Should have at least some methods
        assert len(methods) > 0


class TestMultiTrackExportMixinMethods:
    """Test multi-track ExportMixin has expected methods."""

    def test_export_mixin_has_methods(self):
        """Test ExportMixin has export-related methods."""
        from laser_trim_analyzer.gui.pages.multi_track.export_mixin import ExportMixin
        # Get all methods
        methods = [m for m in dir(ExportMixin) if not m.startswith('__')]
        # Should have at least some methods
        assert len(methods) > 0


class TestBatchProcessingPageMixins:
    """Test BatchProcessingPage uses mixins correctly."""

    def test_batch_page_uses_processing_mixin(self):
        """Test BatchProcessingPage inherits from ProcessingMixin."""
        from laser_trim_analyzer.gui.pages.batch_processing_page import BatchProcessingPage
        from laser_trim_analyzer.gui.pages.batch.processing_mixin import ProcessingMixin
        assert issubclass(BatchProcessingPage, ProcessingMixin)

    def test_batch_page_uses_export_mixin(self):
        """Test BatchProcessingPage inherits from ExportMixin."""
        from laser_trim_analyzer.gui.pages.batch_processing_page import BatchProcessingPage
        from laser_trim_analyzer.gui.pages.batch.export_mixin import ExportMixin
        assert issubclass(BatchProcessingPage, ExportMixin)


class TestHistoricalPageMixins:
    """Test HistoricalPage uses mixins correctly."""

    def test_historical_page_uses_spc_mixin(self):
        """Test HistoricalPage inherits from SPCMixin."""
        from laser_trim_analyzer.gui.pages.historical_page import HistoricalPage
        from laser_trim_analyzer.gui.pages.historical.spc_mixin import SPCMixin
        assert issubclass(HistoricalPage, SPCMixin)

    def test_historical_page_uses_analytics_mixin(self):
        """Test HistoricalPage inherits from AnalyticsMixin."""
        from laser_trim_analyzer.gui.pages.historical_page import HistoricalPage
        from laser_trim_analyzer.gui.pages.historical.analytics_mixin import AnalyticsMixin
        assert issubclass(HistoricalPage, AnalyticsMixin)


class TestMultiTrackPageMixins:
    """Test MultiTrackPage uses mixins correctly."""

    def test_multi_track_page_uses_analysis_mixin(self):
        """Test MultiTrackPage inherits from AnalysisMixin."""
        from laser_trim_analyzer.gui.pages.multi_track_page import MultiTrackPage
        from laser_trim_analyzer.gui.pages.multi_track.analysis_mixin import AnalysisMixin
        assert issubclass(MultiTrackPage, AnalysisMixin)

    def test_multi_track_page_uses_export_mixin(self):
        """Test MultiTrackPage inherits from ExportMixin."""
        from laser_trim_analyzer.gui.pages.multi_track_page import MultiTrackPage
        from laser_trim_analyzer.gui.pages.multi_track.export_mixin import ExportMixin
        assert issubclass(MultiTrackPage, ExportMixin)


class TestMixinDocstrings:
    """Test mixins have proper documentation."""

    def test_processing_mixin_has_docstring(self):
        """Test ProcessingMixin has class docstring."""
        from laser_trim_analyzer.gui.pages.batch.processing_mixin import ProcessingMixin
        assert ProcessingMixin.__doc__ is not None
        assert len(ProcessingMixin.__doc__) > 50

    def test_export_mixin_has_docstring(self):
        """Test batch ExportMixin has class docstring."""
        from laser_trim_analyzer.gui.pages.batch.export_mixin import ExportMixin
        assert ExportMixin.__doc__ is not None
        assert len(ExportMixin.__doc__) > 50


class TestMixinMethodSignatures:
    """Test mixin methods have correct signatures."""

    def test_start_processing_signature(self):
        """Test _start_processing has correct signature (self only)."""
        from laser_trim_analyzer.gui.pages.batch.processing_mixin import ProcessingMixin
        sig = inspect.signature(ProcessingMixin._start_processing)
        params = list(sig.parameters.keys())
        assert params == ['self']

    def test_run_batch_processing_signature(self):
        """Test _run_batch_processing accepts file_paths parameter."""
        from laser_trim_analyzer.gui.pages.batch.processing_mixin import ProcessingMixin
        sig = inspect.signature(ProcessingMixin._run_batch_processing)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'file_paths' in params

    def test_export_batch_results_signature(self):
        """Test _export_batch_results accepts format_type parameter."""
        from laser_trim_analyzer.gui.pages.batch.export_mixin import ExportMixin
        sig = inspect.signature(ExportMixin._export_batch_results)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'format_type' in params

    def test_process_with_turbo_mode_signature(self):
        """Test _process_with_turbo_mode has expected parameters."""
        from laser_trim_analyzer.gui.pages.batch.processing_mixin import ProcessingMixin
        sig = inspect.signature(ProcessingMixin._process_with_turbo_mode)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'file_paths' in params
        assert 'output_dir' in params
        assert 'progress_callback' in params


class TestMixinDependencies:
    """Test mixin module dependencies are importable."""

    def test_processing_mixin_dependencies(self):
        """Test ProcessingMixin dependencies are available."""
        # These are used by ProcessingMixin
        from laser_trim_analyzer.core.exceptions import ProcessingError, ValidationError
        from laser_trim_analyzer.core.models import AnalysisResult
        from laser_trim_analyzer.utils.file_utils import ensure_directory
        assert ProcessingError is not None
        assert ValidationError is not None
        assert AnalysisResult is not None
        assert ensure_directory is not None

    def test_export_mixin_dependencies(self):
        """Test ExportMixin dependencies are available."""
        from pathlib import Path
        from datetime import datetime
        import logging
        assert Path is not None
        assert datetime is not None
        assert logging is not None
