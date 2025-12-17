"""
Machine Learning modules for v3.

Modules:
- threshold: ML-based threshold optimization (RandomForestRegressor)
- drift: Hybrid drift detection (CUSUM + EWMA + IsolationForest)

Simplified from v2's complex ML architecture:
- v2: ~1,800 lines across 6 files
- v3: ~650 lines across 2 files
"""

from laser_trim_analyzer.ml.threshold import (
    ThresholdOptimizer,
    ThresholdConfig,
    TrainingResult,
)

from laser_trim_analyzer.ml.drift import (
    DriftDetector,
    DriftConfig,
    DriftResult,
    DriftDirection,
)

__all__ = [
    # Threshold optimization
    "ThresholdOptimizer",
    "ThresholdConfig",
    "TrainingResult",
    # Drift detection
    "DriftDetector",
    "DriftConfig",
    "DriftResult",
    "DriftDirection",
]
