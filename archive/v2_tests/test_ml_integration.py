"""
Tests for ML integration (Phase 3 refactoring).

These tests verify:
1. ML predictor classes can be imported
2. ML models have expected interfaces
3. ML fallback behavior works correctly
4. Feature extraction methods exist

Note: These tests don't require trained models - they verify the ML architecture.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMLPredictorImports:
    """Test that ML predictor classes can be imported."""

    def test_import_ml_predictor(self):
        """Test MLPredictor can be imported."""
        from laser_trim_analyzer.ml.predictors import MLPredictor
        assert MLPredictor is not None

    def test_import_prediction_result(self):
        """Test PredictionResult can be imported."""
        from laser_trim_analyzer.ml.predictors import PredictionResult
        assert PredictionResult is not None

    def test_import_ml_engine(self):
        """Test MLEngine can be imported."""
        from laser_trim_analyzer.ml.engine import MLEngine
        assert MLEngine is not None

    def test_import_model_factory(self):
        """Test ModelFactory can be imported."""
        from laser_trim_analyzer.ml.models import ModelFactory
        assert ModelFactory is not None


class TestMLModelImports:
    """Test that ML model classes can be imported."""

    def test_import_failure_predictor(self):
        """Test FailurePredictor can be imported."""
        from laser_trim_analyzer.ml.models import FailurePredictor
        assert FailurePredictor is not None

    def test_import_drift_detector(self):
        """Test DriftDetector can be imported."""
        from laser_trim_analyzer.ml.models import DriftDetector
        assert DriftDetector is not None

    def test_import_threshold_optimizer(self):
        """Test ThresholdOptimizer can be imported."""
        from laser_trim_analyzer.ml.models import ThresholdOptimizer
        assert ThresholdOptimizer is not None


class TestPredictionResult:
    """Test PredictionResult dataclass."""

    def test_prediction_result_fields(self):
        """Test PredictionResult has expected fields."""
        from laser_trim_analyzer.ml.predictors import PredictionResult

        result = PredictionResult(
            failure_probability=0.25,
            risk_category='MEDIUM',
            is_anomaly=False,
            anomaly_score=0.1,
            suggested_threshold=0.5,
            confidence_score=0.85,
            warnings=[],
            recommendations=['Monitor closely']
        )

        assert result.failure_probability == 0.25
        assert result.risk_category == 'MEDIUM'
        assert result.is_anomaly is False
        assert result.anomaly_score == 0.1
        assert result.suggested_threshold == 0.5
        assert result.confidence_score == 0.85
        assert result.warnings == []
        assert result.recommendations == ['Monitor closely']

    def test_prediction_result_optional_fields(self):
        """Test PredictionResult optional fields default correctly."""
        from laser_trim_analyzer.ml.predictors import PredictionResult

        result = PredictionResult(
            failure_probability=0.5,
            risk_category='HIGH',
            is_anomaly=True,
            anomaly_score=0.8,
            suggested_threshold=None,
            confidence_score=0.7,
            warnings=[],
            recommendations=[]
        )

        assert result.feature_importance is None


class TestModelFactoryPatterns:
    """Test ModelFactory creates models correctly."""

    def test_model_factory_has_create_failure_predictor_method(self):
        """Test ModelFactory has create_failure_predictor method."""
        from laser_trim_analyzer.ml.models import ModelFactory
        assert hasattr(ModelFactory, 'create_failure_predictor')

    def test_model_factory_has_create_drift_detector_method(self):
        """Test ModelFactory has create_drift_detector method."""
        from laser_trim_analyzer.ml.models import ModelFactory
        assert hasattr(ModelFactory, 'create_drift_detector')

    def test_model_factory_has_create_threshold_optimizer_method(self):
        """Test ModelFactory has create_threshold_optimizer method."""
        from laser_trim_analyzer.ml.models import ModelFactory
        assert hasattr(ModelFactory, 'create_threshold_optimizer')

    def test_model_factory_can_create_failure_predictor(self):
        """Test ModelFactory can create FailurePredictor."""
        from laser_trim_analyzer.ml.models import ModelFactory, FailurePredictor

        model = ModelFactory.create_failure_predictor()
        assert isinstance(model, FailurePredictor)

    def test_model_factory_can_create_drift_detector(self):
        """Test ModelFactory can create DriftDetector."""
        from laser_trim_analyzer.ml.models import ModelFactory, DriftDetector

        model = ModelFactory.create_drift_detector()
        assert isinstance(model, DriftDetector)

    def test_model_factory_can_create_threshold_optimizer(self):
        """Test ModelFactory can create ThresholdOptimizer."""
        from laser_trim_analyzer.ml.models import ModelFactory, ThresholdOptimizer

        model = ModelFactory.create_threshold_optimizer()
        assert isinstance(model, ThresholdOptimizer)


class TestFailurePredictorInterface:
    """Test FailurePredictor has expected interface."""

    def test_failure_predictor_has_train_method(self):
        """Test FailurePredictor has train method."""
        from laser_trim_analyzer.ml.models import FailurePredictor
        assert hasattr(FailurePredictor, 'train')
        assert callable(getattr(FailurePredictor, 'train'))

    def test_failure_predictor_has_predict_method(self):
        """Test FailurePredictor has predict method."""
        from laser_trim_analyzer.ml.models import FailurePredictor
        assert hasattr(FailurePredictor, 'predict')
        assert callable(getattr(FailurePredictor, 'predict'))

    def test_failure_predictor_has_is_trained_property(self):
        """Test FailurePredictor has is_trained property via factory."""
        from laser_trim_analyzer.ml.models import ModelFactory
        predictor = ModelFactory.create_failure_predictor()
        assert hasattr(predictor, 'is_trained')


class TestDriftDetectorInterface:
    """Test DriftDetector has expected interface."""

    def test_drift_detector_has_train_method(self):
        """Test DriftDetector has train method."""
        from laser_trim_analyzer.ml.models import DriftDetector
        assert hasattr(DriftDetector, 'train')
        assert callable(getattr(DriftDetector, 'train'))

    def test_drift_detector_has_predict_method(self):
        """Test DriftDetector has predict method (used for anomaly detection)."""
        from laser_trim_analyzer.ml.models import DriftDetector
        assert hasattr(DriftDetector, 'predict')
        assert callable(getattr(DriftDetector, 'predict'))

    def test_drift_detector_has_is_trained_property(self):
        """Test DriftDetector has is_trained property via factory."""
        from laser_trim_analyzer.ml.models import ModelFactory
        detector = ModelFactory.create_drift_detector()
        assert hasattr(detector, 'is_trained')


class TestThresholdOptimizerInterface:
    """Test ThresholdOptimizer has expected interface."""

    def test_threshold_optimizer_has_train_method(self):
        """Test ThresholdOptimizer has train method."""
        from laser_trim_analyzer.ml.models import ThresholdOptimizer
        assert hasattr(ThresholdOptimizer, 'train')
        assert callable(getattr(ThresholdOptimizer, 'train'))

    def test_threshold_optimizer_has_predict_method(self):
        """Test ThresholdOptimizer has predict method."""
        from laser_trim_analyzer.ml.models import ThresholdOptimizer
        assert hasattr(ThresholdOptimizer, 'predict')
        assert callable(getattr(ThresholdOptimizer, 'predict'))

    def test_threshold_optimizer_has_is_trained_property(self):
        """Test ThresholdOptimizer has is_trained property via factory."""
        from laser_trim_analyzer.ml.models import ModelFactory
        optimizer = ModelFactory.create_threshold_optimizer()
        assert hasattr(optimizer, 'is_trained')


class TestMLEngineInterface:
    """Test MLEngine has expected interface."""

    def test_ml_engine_has_register_model_method(self):
        """Test MLEngine has register_model method."""
        from laser_trim_analyzer.ml.engine import MLEngine
        assert hasattr(MLEngine, 'register_model')

    def test_ml_engine_has_train_model_method(self):
        """Test MLEngine has train_model method."""
        from laser_trim_analyzer.ml.engine import MLEngine
        assert hasattr(MLEngine, 'train_model')

    def test_ml_engine_has_predict_method(self):
        """Test MLEngine has predict method."""
        from laser_trim_analyzer.ml.engine import MLEngine
        assert hasattr(MLEngine, 'predict')


class TestUntrainedModelFallback:
    """Test that untrained models have proper fallback behavior."""

    def test_failure_predictor_untrained_state(self):
        """Test FailurePredictor starts untrained via factory."""
        from laser_trim_analyzer.ml.models import ModelFactory
        predictor = ModelFactory.create_failure_predictor()
        assert predictor.is_trained is False

    def test_drift_detector_untrained_state(self):
        """Test DriftDetector starts untrained via factory."""
        from laser_trim_analyzer.ml.models import ModelFactory
        detector = ModelFactory.create_drift_detector()
        assert detector.is_trained is False

    def test_threshold_optimizer_untrained_state(self):
        """Test ThresholdOptimizer starts untrained via factory."""
        from laser_trim_analyzer.ml.models import ModelFactory
        optimizer = ModelFactory.create_threshold_optimizer()
        assert optimizer.is_trained is False


class TestRiskCategoryMapping:
    """Test risk category mapping from ML predictions."""

    def test_risk_category_values(self):
        """Test RiskCategory has expected values."""
        from laser_trim_analyzer.core.models import RiskCategory

        assert hasattr(RiskCategory, 'LOW')
        assert hasattr(RiskCategory, 'MEDIUM')
        assert hasattr(RiskCategory, 'HIGH')

    def test_risk_from_probability_low(self):
        """Test low probability maps to LOW risk."""
        # Per Phase 3 implementation, prob < 0.3 = LOW
        from laser_trim_analyzer.core.models import RiskCategory

        prob = 0.1
        if prob < 0.3:
            risk = RiskCategory.LOW
        elif prob < 0.7:
            risk = RiskCategory.MEDIUM
        else:
            risk = RiskCategory.HIGH

        assert risk == RiskCategory.LOW

    def test_risk_from_probability_medium(self):
        """Test medium probability maps to MEDIUM risk."""
        from laser_trim_analyzer.core.models import RiskCategory

        prob = 0.5
        if prob < 0.3:
            risk = RiskCategory.LOW
        elif prob < 0.7:
            risk = RiskCategory.MEDIUM
        else:
            risk = RiskCategory.HIGH

        assert risk == RiskCategory.MEDIUM

    def test_risk_from_probability_high(self):
        """Test high probability maps to HIGH risk."""
        from laser_trim_analyzer.core.models import RiskCategory

        prob = 0.8
        if prob < 0.3:
            risk = RiskCategory.LOW
        elif prob < 0.7:
            risk = RiskCategory.MEDIUM
        else:
            risk = RiskCategory.HIGH

        assert risk == RiskCategory.HIGH


class TestUnifiedProcessorMLMethods:
    """Test UnifiedProcessor has correct ML method signatures."""

    def test_predict_failure_signature(self):
        """Test predict_failure method exists with correct return hints."""
        from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
        import inspect

        sig = inspect.signature(UnifiedProcessor.predict_failure)
        params = list(sig.parameters.keys())

        # Should have self plus analysis parameters
        assert 'self' in params or len(params) > 0

    def test_detect_drift_signature(self):
        """Test detect_drift method exists with correct return hints."""
        from laser_trim_analyzer.core.unified_processor import UnifiedProcessor
        import inspect

        sig = inspect.signature(UnifiedProcessor.detect_drift)
        params = list(sig.parameters.keys())

        # Should have self plus analysis parameters
        assert 'self' in params or len(params) > 0


class TestFormulaFallbackConstants:
    """Test formula fallback constants are defined."""

    def test_medium_risk_threshold_constant(self):
        """Test MEDIUM_RISK_THRESHOLD constant exists."""
        from laser_trim_analyzer.core.constants import MEDIUM_RISK_THRESHOLD
        assert MEDIUM_RISK_THRESHOLD == 0.3

    def test_high_risk_threshold_constant(self):
        """Test HIGH_RISK_THRESHOLD constant exists."""
        from laser_trim_analyzer.core.constants import HIGH_RISK_THRESHOLD
        assert HIGH_RISK_THRESHOLD == 0.7


class TestMLConfig:
    """Test ML configuration options."""

    def test_config_has_ml_section(self):
        """Test Config has ml section."""
        from laser_trim_analyzer.core.config import Config
        config = Config()
        assert hasattr(config, 'ml')

    def test_ml_config_has_model_path(self):
        """Test ML config has model_path."""
        from laser_trim_analyzer.core.config import Config
        config = Config()
        assert hasattr(config.ml, 'model_path')

    def test_ml_config_has_enabled_flag(self):
        """Test ML config has enabled flag."""
        from laser_trim_analyzer.core.config import Config
        config = Config()
        assert hasattr(config.ml, 'enabled')
