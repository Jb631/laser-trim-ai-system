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

# ---------------------------------------------------------------------------
# Shared MLManager cache
# ---------------------------------------------------------------------------
# Constructing an MLManager and calling load_all() reads up to ~135 pickled
# predictor files from disk and rehydrates ML state for every model. The GUI
# was instantiating it on every page show / dashboard refresh / trend load —
# 134 predictors reloaded six times in 30 minutes during the visual review.
#
# get_shared_ml_manager() returns a process-wide cached MLManager and only
# refreshes after `max_age_seconds`. Training paths must call
# invalidate_shared_ml_manager() so subsequent reads pick up the new models.
import threading as _threading
import time as _time
from typing import Optional as _Optional

_shared_ml_manager: _Optional["MLManager"] = None
_shared_ml_manager_loaded_at: _Optional[float] = None
_shared_ml_manager_lock = _threading.Lock()


def get_shared_ml_manager(db, max_age_seconds: float = 300.0) -> "MLManager":
    """Return a cached MLManager with load_all() already called.

    Reuses the same instance across the app for up to ``max_age_seconds``
    so navigating the GUI does not re-read every predictor pickle.
    """
    global _shared_ml_manager, _shared_ml_manager_loaded_at
    with _shared_ml_manager_lock:
        now = _time.time()
        stale = (
            _shared_ml_manager is None
            or _shared_ml_manager_loaded_at is None
            or (now - _shared_ml_manager_loaded_at) > max_age_seconds
        )
        if stale:
            mgr = MLManager(db)
            load_failed = False
            try:
                mgr.load_all()
            except Exception:
                # Surface but do not crash callers — they treat empty/missing
                # ML state as "ML unavailable" and fall back to formula
                # thresholds. Don't promote a partially-loaded manager into
                # the cache: leave loaded_at=None so the next caller retries
                # rather than handing out inconsistent state for 5 minutes.
                import logging as _logging
                _logging.getLogger(__name__).debug(
                    "Shared MLManager load_all failed", exc_info=True
                )
                load_failed = True
            _shared_ml_manager = mgr
            _shared_ml_manager_loaded_at = None if load_failed else now
        return _shared_ml_manager


def invalidate_shared_ml_manager() -> None:
    """Force the next get_shared_ml_manager() to load fresh state.

    Call after training, applying thresholds, or any operation that
    changes on-disk ML state.
    """
    global _shared_ml_manager, _shared_ml_manager_loaded_at
    with _shared_ml_manager_lock:
        _shared_ml_manager = None
        _shared_ml_manager_loaded_at = None

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
    # Shared cache
    "get_shared_ml_manager",
    "invalidate_shared_ml_manager",
]
