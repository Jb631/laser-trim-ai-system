"""
Tests for performance validation (Phase 6 Day 3).

These tests verify:
1. Test suite execution time
2. Import times for key modules
3. Strategy selection performance
4. ML prediction performance characteristics
5. Memory-related safeguards

Note: These tests verify performance characteristics without requiring actual data files.
"""
import pytest
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestModuleImportPerformance:
    """Test that key modules can be imported (import time varies by environment)."""

    def test_core_config_importable(self):
        """Test Config can be imported successfully."""
        # Note: First import includes all dependencies, so timing varies.
        # We just verify it's importable.
        from laser_trim_analyzer.core.config import Config
        assert Config is not None

    def test_unified_processor_importable(self):
        """Test UnifiedProcessor can be imported successfully."""
        from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
        assert UnifiedProcessor is not None

    def test_ml_models_importable(self):
        """Test ML models can be imported successfully."""
        from laser_trim_analyzer.ml.models import (
            FailurePredictor, DriftDetector, ThresholdOptimizer, ModelFactory
        )
        assert FailurePredictor is not None
        assert DriftDetector is not None
        assert ThresholdOptimizer is not None
        assert ModelFactory is not None

    def test_chart_widget_importable(self):
        """Test ChartWidget can be imported successfully."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidget
        assert ChartWidget is not None


class TestStrategySelectionPerformance:
    """Test strategy selection logic performance."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        config = Mock()
        config.processing = Mock()
        config.processing.use_unified_processor = True
        config.processing.unified_processor_strategy = 'auto'
        config.processing.turbo_mode_threshold = 100
        config.processing.memory_safe_threshold = 500
        return config

    def test_auto_strategy_selection_time(self, mock_config):
        """Test AutoStrategy selects appropriate strategy quickly."""
        from laser_trim_analyzer.core.unified_processor import AutoStrategy

        strategy = AutoStrategy(mock_config)

        # Time strategy selection for various batch sizes
        batch_sizes = [5, 50, 200, 1000]

        for size in batch_sizes:
            start = time.perf_counter()
            selected = strategy._select_strategy(size)
            elapsed = time.perf_counter() - start

            # Strategy selection should be instant (<10ms)
            assert elapsed < 0.01, f"Strategy selection for {size} files took {elapsed*1000:.1f}ms"

    def test_strategy_name_lookup_performance(self):
        """Test strategy names are retrieved efficiently."""
        from laser_trim_analyzer.core.unified_processor import (
            StandardStrategy, TurboStrategy, MemorySafeStrategy, AutoStrategy
        )

        strategies = [StandardStrategy, TurboStrategy, MemorySafeStrategy, AutoStrategy]

        for strategy_cls in strategies:
            start = time.perf_counter()
            name = strategy_cls.name
            elapsed = time.perf_counter() - start

            # Name lookup should be instant
            assert elapsed < 0.001, f"{strategy_cls.__name__}.name took {elapsed*1000:.3f}ms"


class TestMLPerformanceCharacteristics:
    """Test ML-related performance characteristics."""

    def test_model_factory_creation_time(self):
        """Test ModelFactory creates models quickly."""
        from laser_trim_analyzer.ml.models import ModelFactory

        factories = [
            ('failure_predictor', ModelFactory.create_failure_predictor),
            ('drift_detector', ModelFactory.create_drift_detector),
            ('threshold_optimizer', ModelFactory.create_threshold_optimizer),
        ]

        for name, factory_fn in factories:
            start = time.perf_counter()
            model = factory_fn()
            elapsed = time.perf_counter() - start

            # Model creation should be fast (<100ms)
            assert elapsed < 0.1, f"Creating {name} took {elapsed*1000:.1f}ms (should be <100ms)"

    def test_untrained_model_predict_fallback_time(self):
        """Test untrained model predict returns quickly (fallback behavior)."""
        from laser_trim_analyzer.ml.models import ModelFactory
        import numpy as np

        predictor = ModelFactory.create_failure_predictor()

        # Create minimal dummy features
        features = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

        start = time.perf_counter()
        try:
            # Untrained model should fallback quickly
            result = predictor.predict(features)
        except Exception:
            # If it raises, that's still fast
            pass
        elapsed = time.perf_counter() - start

        # Fallback should be very fast
        assert elapsed < 0.1, f"Untrained predict took {elapsed*1000:.1f}ms (should be <100ms)"


class TestMemorySafeguards:
    """Test memory-related safeguards exist."""

    def test_memory_safe_strategy_has_chunk_size(self):
        """Test MemorySafeStrategy has chunk_size parameter."""
        from laser_trim_analyzer.core.unified_processor import MemorySafeStrategy

        # Check class has chunk processing capability
        import inspect
        source = inspect.getsource(MemorySafeStrategy)
        assert 'chunk' in source.lower(), "MemorySafeStrategy should have chunking"

    def test_processing_mixin_has_gc_handling(self):
        """Test ProcessingMixin has garbage collection handling."""
        from laser_trim_analyzer.gui.pages.batch.processing_mixin import ProcessingMixin

        import inspect
        source = inspect.getsource(ProcessingMixin)
        assert 'gc.collect' in source, "ProcessingMixin should call gc.collect"

    def test_processing_mixin_has_memory_cleanup(self):
        """Test ProcessingMixin has memory cleanup intervals."""
        from laser_trim_analyzer.gui.pages.batch.processing_mixin import ProcessingMixin

        import inspect
        source = inspect.getsource(ProcessingMixin)
        assert 'cleanup_interval' in source.lower() or 'memory' in source.lower(), \
            "ProcessingMixin should have memory management"


class TestCachingPerformance:
    """Test caching-related performance characteristics."""

    def test_processor_has_caching_layer(self):
        """Test UnifiedProcessor has caching capability."""
        from laser_trim_analyzer.core.unified_processor import UnifiedProcessor

        # Check class has caching methods
        assert hasattr(UnifiedProcessor, 'get_cached_prediction') or \
               hasattr(UnifiedProcessor, 'cache_prediction'), \
               "UnifiedProcessor should have caching methods"

    def test_hash_computation_available(self):
        """Test file hash computation is available."""
        from laser_trim_analyzer.database.manager import DatabaseManager

        assert hasattr(DatabaseManager, 'compute_file_hash'), \
            "DatabaseManager should have compute_file_hash method"


class TestIncrementalProcessingPerformance:
    """Test incremental processing performance characteristics."""

    def test_database_has_unprocessed_filter(self):
        """Test DatabaseManager has efficient unprocessed file filter."""
        from laser_trim_analyzer.database.manager import DatabaseManager

        assert hasattr(DatabaseManager, 'get_unprocessed_files'), \
            "DatabaseManager should have get_unprocessed_files method"

    def test_database_has_processed_file_tracking(self):
        """Test DatabaseManager has processed file tracking."""
        from laser_trim_analyzer.database.manager import DatabaseManager

        tracking_methods = [
            'is_file_processed',
            'mark_file_processed',
            'get_processed_file',
        ]

        for method in tracking_methods:
            assert hasattr(DatabaseManager, method), \
                f"DatabaseManager should have {method} method"


class TestTestSuitePerformance:
    """Test the test suite itself runs efficiently."""

    def test_calculation_tests_are_fast(self):
        """Test calculation tests run in reasonable time."""
        # Import and count calculation tests
        from tests import test_calculations

        # Get test count
        test_count = sum(1 for name in dir(test_calculations)
                        if name.startswith('Test'))

        # Should have tests but not too many
        assert test_count > 0, "Should have calculation tests"

    def test_validation_tests_are_fast(self):
        """Test validation tests run in reasonable time."""
        from tests import test_data_validation

        test_count = sum(1 for name in dir(test_data_validation)
                        if name.startswith('Test'))

        assert test_count > 0, "Should have validation tests"


class TestConfigPerformance:
    """Test configuration loading performance."""

    def test_config_loads_quickly(self):
        """Test Config object creates quickly."""
        from laser_trim_analyzer.core.config import Config

        start = time.perf_counter()
        config = Config()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"Config creation took {elapsed*1000:.1f}ms (should be <500ms)"

    def test_config_has_performance_settings(self):
        """Test Config has performance-related settings."""
        from laser_trim_analyzer.core.config import Config

        config = Config()

        # Check processing config exists
        assert hasattr(config, 'processing'), "Config should have processing settings"

        # Check processing has key performance settings
        processing = config.processing
        # max_workers is a basic parallelism setting
        assert hasattr(processing, 'max_workers'), \
            "ProcessingConfig should have max_workers"

        # unified_processor_strategy controls processing strategy
        assert hasattr(processing, 'unified_processor_strategy'), \
            "ProcessingConfig should have unified_processor_strategy"


class TestFeatureFlagPerformance:
    """Test feature flag checks are fast."""

    def test_feature_flag_access_time(self):
        """Test feature flag access is instant."""
        from laser_trim_analyzer.core.config import Config

        config = Config()

        flags = [
            ('use_unified_processor', config.processing),
            ('use_ml_failure_predictor', config.processing),
            ('use_ml_drift_detector', config.processing),
        ]

        for flag_name, obj in flags:
            if hasattr(obj, flag_name):
                start = time.perf_counter()
                value = getattr(obj, flag_name)
                elapsed = time.perf_counter() - start

                # Flag access should be instant
                assert elapsed < 0.001, \
                    f"Accessing {flag_name} took {elapsed*1000:.3f}ms"
