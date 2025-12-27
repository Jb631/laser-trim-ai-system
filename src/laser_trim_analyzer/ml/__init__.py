"""
Machine Learning modules for Laser Trim Analyzer v3.

Per-Model ML System:
- predictor: Per-model failure probability prediction
- threshold_optimizer: Per-model threshold optimization
- drift_detector: Per-model drift detection
- profiler: Per-model statistical profiling
- manager: Orchestrates all per-model ML
"""

from laser_trim_analyzer.ml.predictor import (
    ModelPredictor,
    PredictorConfig,
    PredictorMetrics,
    PredictorTrainingResult,
    extract_features,
    FEATURE_COLUMNS,
)

from laser_trim_analyzer.ml.threshold_optimizer import (
    ModelThresholdOptimizer,
    ThresholdResult,
    ThresholdOptimizerState,
)

from laser_trim_analyzer.ml.drift_detector import (
    ModelDriftDetector,
    DriftResult,
    DriftDirection,
    DriftDetectorState,
)

from laser_trim_analyzer.ml.profiler import (
    ModelProfiler,
    ModelProfile,
    ProfileStatistics,
    ModelInsight,
    calculate_cross_model_metrics,
)

from laser_trim_analyzer.ml.manager import (
    MLManager,
    ModelTrainingResult,
    TrainingProgress,
    ApplyProgress,
)

__all__ = [
    # Predictor
    "ModelPredictor",
    "PredictorConfig",
    "PredictorMetrics",
    "PredictorTrainingResult",
    "extract_features",
    "FEATURE_COLUMNS",
    # Threshold Optimizer
    "ModelThresholdOptimizer",
    "ThresholdResult",
    "ThresholdOptimizerState",
    # Drift Detector
    "ModelDriftDetector",
    "DriftResult",
    "DriftDirection",
    "DriftDetectorState",
    # Profiler
    "ModelProfiler",
    "ModelProfile",
    "ProfileStatistics",
    "ModelInsight",
    "calculate_cross_model_metrics",
    # Manager
    "MLManager",
    "ModelTrainingResult",
    "TrainingProgress",
    "ApplyProgress",
]
