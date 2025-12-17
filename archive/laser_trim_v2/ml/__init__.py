# src/laser_trim_analyzer/ml/__init__.py
"""
Machine learning components for predictive analysis.

This module provides ML capabilities for failure prediction,
threshold optimization, and drift detection.
"""

import logging

# Check for required dependencies
ML_AVAILABLE = True
MISSING_DEPS = []

try:
    import numpy
except ImportError:
    ML_AVAILABLE = False
    MISSING_DEPS.append('numpy')

try:
    import pandas
except ImportError:
    ML_AVAILABLE = False
    MISSING_DEPS.append('pandas')

try:
    import sklearn
except ImportError:
    ML_AVAILABLE = False
    MISSING_DEPS.append('scikit-learn')

try:
    import joblib
except ImportError:
    ML_AVAILABLE = False
    MISSING_DEPS.append('joblib')

# Log status
logger = logging.getLogger(__name__)

if not ML_AVAILABLE:
    logger.warning(f"ML features disabled. Missing dependencies: {', '.join(MISSING_DEPS)}")
    logger.warning(f"Install with: pip install {' '.join(MISSING_DEPS)}")
else:
    logger.info("ML module initialized with all dependencies")

# Import ML components only if dependencies are available
if ML_AVAILABLE:
    try:
        from laser_trim_analyzer.ml.models import FailurePredictor, ThresholdOptimizer, DriftDetector
        from laser_trim_analyzer.ml.engine import MLEngine, ModelConfig
        from laser_trim_analyzer.ml.ml_manager import get_ml_manager
        
        __all__ = [
            "FailurePredictor",
            "ThresholdOptimizer",
            "DriftDetector",
            "MLEngine",
            "ModelConfig",
            "get_ml_manager",
            "ML_AVAILABLE",
            "MISSING_DEPS"
        ]
    except ImportError as e:
        logger.error(f"Failed to import ML components: {e}")
        ML_AVAILABLE = False
        __all__ = ["ML_AVAILABLE", "MISSING_DEPS"]
else:
    __all__ = ["ML_AVAILABLE", "MISSING_DEPS"]