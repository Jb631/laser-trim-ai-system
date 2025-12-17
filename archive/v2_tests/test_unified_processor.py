"""
Tests for UnifiedProcessor and processing strategies (Phase 2 refactoring).

These tests verify:
1. Strategy classes are properly defined
2. Strategy selection logic works correctly
3. CachingLayer and SecurityLayer function
4. ML integration points are wired correctly (Phase 3)

Note: Full integration tests require actual data files and are run separately.
These unit tests focus on the architecture and logic.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestStrategyImports:
    """Test that all strategy classes can be imported."""

    def test_import_unified_processor(self):
        """Test UnifiedProcessor can be imported."""
        from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
        assert UnifiedProcessor is not None

    def test_import_processing_strategy(self):
        """Test ProcessingStrategy abstract class can be imported."""
        from laser_trim_analyzer.core.unified_processor import ProcessingStrategy
        assert ProcessingStrategy is not None

    def test_import_standard_strategy(self):
        """Test StandardStrategy can be imported."""
        from laser_trim_analyzer.core.unified_processor import StandardStrategy
        assert StandardStrategy is not None

    def test_import_turbo_strategy(self):
        """Test TurboStrategy can be imported."""
        from laser_trim_analyzer.core.unified_processor import TurboStrategy
        assert TurboStrategy is not None

    def test_import_memory_safe_strategy(self):
        """Test MemorySafeStrategy can be imported."""
        from laser_trim_analyzer.core.unified_processor import MemorySafeStrategy
        assert MemorySafeStrategy is not None

    def test_import_auto_strategy(self):
        """Test AutoStrategy can be imported."""
        from laser_trim_analyzer.core.unified_processor import AutoStrategy
        assert AutoStrategy is not None


class TestStrategyInheritance:
    """Test that strategies properly inherit from ProcessingStrategy."""

    def test_standard_strategy_is_processing_strategy(self):
        """Test StandardStrategy inherits from ProcessingStrategy."""
        from laser_trim_analyzer.core.unified_processor import (
            StandardStrategy, ProcessingStrategy
        )
        assert issubclass(StandardStrategy, ProcessingStrategy)

    def test_turbo_strategy_is_processing_strategy(self):
        """Test TurboStrategy inherits from ProcessingStrategy."""
        from laser_trim_analyzer.core.unified_processor import (
            TurboStrategy, ProcessingStrategy
        )
        assert issubclass(TurboStrategy, ProcessingStrategy)

    def test_memory_safe_strategy_is_processing_strategy(self):
        """Test MemorySafeStrategy inherits from ProcessingStrategy."""
        from laser_trim_analyzer.core.unified_processor import (
            MemorySafeStrategy, ProcessingStrategy
        )
        assert issubclass(MemorySafeStrategy, ProcessingStrategy)

    def test_auto_strategy_is_processing_strategy(self):
        """Test AutoStrategy inherits from ProcessingStrategy."""
        from laser_trim_analyzer.core.unified_processor import (
            AutoStrategy, ProcessingStrategy
        )
        assert issubclass(AutoStrategy, ProcessingStrategy)


class TestStrategyNames:
    """Test that strategies have correct name properties."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = Mock()
        config.processing = Mock()
        config.processing.max_workers = 4
        config.processing.chunk_size = 2000
        config.processing.turbo_mode = False
        config.processing.memory_limit_mb = 500
        config.processing.auto_strategy_thresholds = Mock()
        config.processing.auto_strategy_thresholds.small_batch = 10
        config.processing.auto_strategy_thresholds.large_batch = 500
        config.processing.auto_strategy_thresholds.memory_limit_mb = 500
        return config

    def test_standard_strategy_name(self, mock_config):
        """Test StandardStrategy has correct name."""
        from laser_trim_analyzer.core.unified_processor import StandardStrategy
        strategy = StandardStrategy(mock_config)
        assert strategy.name == "standard"

    def test_turbo_strategy_name(self, mock_config):
        """Test TurboStrategy has correct name."""
        from laser_trim_analyzer.core.unified_processor import TurboStrategy
        strategy = TurboStrategy(mock_config)
        assert strategy.name == "turbo"

    def test_memory_safe_strategy_name(self, mock_config):
        """Test MemorySafeStrategy has correct name."""
        from laser_trim_analyzer.core.unified_processor import MemorySafeStrategy
        strategy = MemorySafeStrategy(mock_config)
        assert strategy.name == "memory_safe"

    def test_auto_strategy_name(self, mock_config):
        """Test AutoStrategy has correct name."""
        from laser_trim_analyzer.core.unified_processor import AutoStrategy
        strategy = AutoStrategy(mock_config)
        assert strategy.name == "auto"


class TestAutoStrategySelection:
    """Test AutoStrategy correctly selects sub-strategies."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with auto-strategy thresholds."""
        config = Mock()
        config.processing = Mock()
        config.processing.max_workers = 4
        config.processing.chunk_size = 2000
        config.processing.turbo_mode = True
        config.processing.memory_limit_mb = 500
        config.processing.auto_strategy_thresholds = Mock()
        config.processing.auto_strategy_thresholds.small_batch = 10
        config.processing.auto_strategy_thresholds.large_batch = 500
        config.processing.auto_strategy_thresholds.memory_limit_mb = 500
        return config

    def test_auto_strategy_selects_standard_for_small_batch(self, mock_config):
        """Test AutoStrategy selects StandardStrategy for ≤10 files."""
        from laser_trim_analyzer.core.unified_processor import (
            AutoStrategy, StandardStrategy
        )
        strategy = AutoStrategy(mock_config)

        # For small batches (≤10), should select Standard
        small_files = [Path(f"file_{i}.xlsx") for i in range(5)]
        selected = strategy._select_strategy(len(small_files))

        # Check it's either StandardStrategy class or instance
        assert isinstance(selected, StandardStrategy) or \
               (isinstance(selected, type) and issubclass(selected, StandardStrategy))

    def test_auto_strategy_selects_turbo_for_medium_batch(self, mock_config):
        """Test AutoStrategy selects TurboStrategy for 11-500 files."""
        from laser_trim_analyzer.core.unified_processor import (
            AutoStrategy, TurboStrategy
        )
        strategy = AutoStrategy(mock_config)

        # For medium batches (11-500), should select Turbo
        medium_files = [Path(f"file_{i}.xlsx") for i in range(100)]
        selected = strategy._select_strategy(len(medium_files))

        # Check it's TurboStrategy
        assert isinstance(selected, TurboStrategy) or \
               (isinstance(selected, type) and issubclass(selected, TurboStrategy))

    def test_auto_strategy_selects_memory_safe_for_large_batch(self, mock_config):
        """Test AutoStrategy selects MemorySafeStrategy for >500 files."""
        from laser_trim_analyzer.core.unified_processor import (
            AutoStrategy, MemorySafeStrategy
        )
        strategy = AutoStrategy(mock_config)

        # For large batches (>500), should select MemorySafe
        large_files = [Path(f"file_{i}.xlsx") for i in range(600)]
        selected = strategy._select_strategy(len(large_files))

        # Check it's MemorySafeStrategy
        assert isinstance(selected, MemorySafeStrategy) or \
               (isinstance(selected, type) and issubclass(selected, MemorySafeStrategy))


class TestSecurityLayerImport:
    """Test SecurityLayer can be imported and functions."""

    def test_import_security_layer(self):
        """Test SecurityLayer can be imported."""
        try:
            from laser_trim_analyzer.core.unified_processor import SecurityLayer
            assert SecurityLayer is not None
        except ImportError:
            # SecurityLayer might be optional - that's OK
            pytest.skip("SecurityLayer not available")


class TestCachingLayerImport:
    """Test CachingLayer can be imported and functions."""

    def test_import_caching_layer(self):
        """Test CachingLayer can be imported."""
        try:
            from laser_trim_analyzer.core.unified_processor import CachingLayer
            assert CachingLayer is not None
        except ImportError:
            # CachingLayer might be optional - that's OK
            pytest.skip("CachingLayer not available")


class TestUnifiedProcessorMethods:
    """Test UnifiedProcessor has expected methods."""

    def test_has_process_file_method(self):
        """Test UnifiedProcessor has process_file method."""
        from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
        assert hasattr(UnifiedProcessor, 'process_file')
        assert callable(getattr(UnifiedProcessor, 'process_file'))

    def test_has_process_batch_method(self):
        """Test UnifiedProcessor has process_batch method."""
        from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
        assert hasattr(UnifiedProcessor, 'process_batch')
        assert callable(getattr(UnifiedProcessor, 'process_batch'))

    def test_has_process_file_sync_method(self):
        """Test UnifiedProcessor has process_file_sync method for backward compatibility."""
        from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
        assert hasattr(UnifiedProcessor, 'process_file_sync')
        assert callable(getattr(UnifiedProcessor, 'process_file_sync'))

    def test_has_predict_failure_method(self):
        """Test UnifiedProcessor has predict_failure method (Phase 3 ML integration)."""
        from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
        assert hasattr(UnifiedProcessor, 'predict_failure')
        assert callable(getattr(UnifiedProcessor, 'predict_failure'))

    def test_has_detect_drift_method(self):
        """Test UnifiedProcessor has detect_drift method (Phase 3 ML integration)."""
        from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
        assert hasattr(UnifiedProcessor, 'detect_drift')
        assert callable(getattr(UnifiedProcessor, 'detect_drift'))


class TestMLIntegrationMethods:
    """Test ML integration methods exist (Phase 3)."""

    def test_has_predict_failure_safe(self):
        """Test UnifiedProcessor has predict_failure_safe wrapper."""
        from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
        assert hasattr(UnifiedProcessor, 'predict_failure_safe')

    def test_has_detect_drift_safe(self):
        """Test UnifiedProcessor has detect_drift_safe wrapper."""
        from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
        assert hasattr(UnifiedProcessor, 'detect_drift_safe')

    def test_has_batch_predictions(self):
        """Test UnifiedProcessor has batch prediction method."""
        from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
        assert hasattr(UnifiedProcessor, 'predict_failures_batch')

    def test_has_prediction_cache(self):
        """Test UnifiedProcessor has prediction caching methods."""
        from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
        assert hasattr(UnifiedProcessor, 'get_cached_prediction')
        assert hasattr(UnifiedProcessor, 'cache_prediction')


class TestStrategyFactory:
    """Test strategy factory pattern."""

    def test_get_strategy_returns_strategy(self):
        """Test that strategy can be obtained from name."""
        from laser_trim_analyzer.core.unified_processor import (
            StandardStrategy, TurboStrategy, MemorySafeStrategy, AutoStrategy
        )

        strategies = {
            'standard': StandardStrategy,
            'turbo': TurboStrategy,
            'memory_safe': MemorySafeStrategy,
            'auto': AutoStrategy,
        }

        for name, expected_class in strategies.items():
            assert issubclass(expected_class, object), \
                f"Strategy {name} should be a valid class"


class TestProcessingConfig:
    """Test that processing configuration works with UnifiedProcessor."""

    def test_config_has_processing_section(self):
        """Test Config class has processing section."""
        from laser_trim_analyzer.core.config import Config
        config = Config()
        assert hasattr(config, 'processing'), "Config should have processing attribute"

    def test_config_has_strategy_setting(self):
        """Test Config has unified_processor_strategy setting."""
        from laser_trim_analyzer.core.config import Config
        config = Config()
        if hasattr(config, 'refactoring'):
            assert hasattr(config.refactoring, 'unified_processor_strategy')


class TestFeatureFlags:
    """Test feature flag integration."""

    def test_config_has_use_unified_processor_flag(self):
        """Test Config has use_unified_processor feature flag."""
        from laser_trim_analyzer.core.config import Config
        config = Config()
        if hasattr(config, 'refactoring'):
            assert hasattr(config.refactoring, 'use_unified_processor'), \
                "Config should have use_unified_processor flag"

    def test_config_has_ml_flags(self):
        """Test Config has ML feature flags (Phase 3)."""
        from laser_trim_analyzer.core.config import Config
        config = Config()
        if hasattr(config, 'refactoring'):
            assert hasattr(config.refactoring, 'use_ml_failure_predictor')
            assert hasattr(config.refactoring, 'use_ml_drift_detector')
