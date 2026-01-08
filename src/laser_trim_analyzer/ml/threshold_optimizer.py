"""
Per-Model Threshold Optimizer for Laser Trim Analyzer v3.

Calculates optimal sigma threshold per model based on pass/fail outcomes.
Uses severity weighting (fail_points) to influence threshold calculation.

Part of the per-model ML redesign - handles threshold optimization
while ModelPredictor handles failure probability prediction.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """Result of threshold calculation."""
    threshold: float
    confidence: float  # 0-1, based on sample size and separation
    method: str  # 'separation', 'percentile', 'weighted', 'fallback'

    # Statistics used in calculation
    pass_sigma_mean: float = 0.0
    pass_sigma_std: float = 0.0
    pass_sigma_max: float = 0.0
    fail_sigma_min: float = 0.0
    fail_sigma_mean: float = 0.0

    # Sample counts
    n_pass: int = 0
    n_fail: int = 0

    # Severity info
    avg_fail_severity: float = 0.0  # Average fail_points when failed


@dataclass
class ThresholdOptimizerState:
    """Serializable state for database storage."""
    model_name: str
    threshold: Optional[float]
    confidence: Optional[float]
    method: Optional[str]
    n_samples: int
    n_pass: int
    n_fail: int
    pass_sigma_mean: float
    pass_sigma_std: float
    fail_sigma_mean: float
    avg_fail_severity: float
    calculated_date: Optional[datetime]


class ModelThresholdOptimizer:
    """
    Per-model threshold optimization with severity weighting.

    Learns optimal sigma threshold from:
    - Pass/fail outcomes (from Trim and Final Test data)
    - Severity weighting (fail_points count)

    Threshold calculation strategies:
    1. Clean separation: midpoint between max(passing) and min(failing)
    2. Overlap: severity-weighted percentile approach
    3. Insufficient data: conservative formula-based fallback
    """

    # Minimum samples for reliable threshold
    MIN_SAMPLES = 20
    MIN_FAILURES = 3  # Need some failures to learn from

    def __init__(self, model_name: str):
        """
        Initialize threshold optimizer for a specific model.

        Args:
            model_name: Product model number (e.g., "6828", "8340-1")
        """
        self.model_name = model_name

        # Learned threshold
        self.threshold: Optional[float] = None
        self.confidence: Optional[float] = None
        self.method: Optional[str] = None

        # Statistics from training
        self.n_samples: int = 0
        self.n_pass: int = 0
        self.n_fail: int = 0
        self.pass_sigma_mean: float = 0.0
        self.pass_sigma_std: float = 0.0
        self.pass_sigma_max: float = 0.0
        self.fail_sigma_min: float = 0.0
        self.fail_sigma_mean: float = 0.0
        self.avg_fail_severity: float = 0.0

        # State
        self.is_calculated: bool = False
        self.calculated_date: Optional[datetime] = None

    def calculate_threshold(
        self,
        sigma_values: pd.Series,
        passed: pd.Series,
        fail_points: Optional[pd.Series] = None,
        linearity_spec: Optional[float] = None
    ) -> ThresholdResult:
        """
        Calculate optimal sigma threshold from training data.

        Args:
            sigma_values: Series of sigma_gradient values
            passed: Series of boolean (True=passed, False=failed)
            fail_points: Optional series of fail point counts (severity)
            linearity_spec: Optional spec for fallback calculation

        Returns:
            ThresholdResult with calculated threshold and metadata
        """
        # Convert to numpy for calculations
        sigma = np.array(sigma_values)
        is_pass = np.array(passed).astype(bool)

        # Get pass/fail groups
        pass_sigma = sigma[is_pass]
        fail_sigma = sigma[~is_pass]

        self.n_samples = len(sigma)
        self.n_pass = len(pass_sigma)
        self.n_fail = len(fail_sigma)

        # Calculate statistics
        if len(pass_sigma) > 0:
            self.pass_sigma_mean = float(np.mean(pass_sigma))
            self.pass_sigma_std = float(np.std(pass_sigma))
            self.pass_sigma_max = float(np.max(pass_sigma))

        if len(fail_sigma) > 0:
            self.fail_sigma_min = float(np.min(fail_sigma))
            self.fail_sigma_mean = float(np.mean(fail_sigma))

        # Calculate severity if provided
        if fail_points is not None:
            fp = np.array(fail_points)
            fail_fp = fp[~is_pass]
            if len(fail_fp) > 0:
                self.avg_fail_severity = float(np.mean(fail_fp))

        # Determine threshold using appropriate strategy
        if self.n_samples < self.MIN_SAMPLES:
            # Insufficient data - use fallback
            result = self._fallback_threshold(linearity_spec)
        elif self.n_fail < self.MIN_FAILURES:
            # No/few failures - use percentile of passing
            result = self._percentile_threshold(pass_sigma)
        elif self.fail_sigma_min > self.pass_sigma_max:
            # Clean separation - use midpoint
            result = self._separation_threshold(pass_sigma, fail_sigma)
        else:
            # Overlap - use severity-weighted approach
            result = self._weighted_threshold(
                sigma, is_pass,
                np.array(fail_points) if fail_points is not None else None
            )

        # Store results
        self.threshold = result.threshold
        self.confidence = result.confidence
        self.method = result.method
        self.is_calculated = True
        self.calculated_date = datetime.now()

        logger.info(
            f"ThresholdOptimizer[{self.model_name}] calculated - "
            f"Threshold: {result.threshold:.6f}, Confidence: {result.confidence:.2f}, "
            f"Method: {result.method}, Samples: {self.n_samples} (pass={self.n_pass}, fail={self.n_fail})"
        )

        return result

    def _separation_threshold(
        self,
        pass_sigma: np.ndarray,
        fail_sigma: np.ndarray
    ) -> ThresholdResult:
        """Calculate threshold when pass/fail are cleanly separated."""
        max_pass = float(np.max(pass_sigma))
        min_fail = float(np.min(fail_sigma))

        # Midpoint between groups
        threshold = (max_pass + min_fail) / 2

        # High confidence due to clean separation
        separation_ratio = (min_fail - max_pass) / (max_pass + 1e-10)
        confidence = min(0.95, 0.7 + 0.25 * separation_ratio)

        return ThresholdResult(
            threshold=threshold,
            confidence=confidence,
            method='separation',
            pass_sigma_mean=self.pass_sigma_mean,
            pass_sigma_std=self.pass_sigma_std,
            pass_sigma_max=max_pass,
            fail_sigma_min=min_fail,
            fail_sigma_mean=self.fail_sigma_mean,
            n_pass=self.n_pass,
            n_fail=self.n_fail,
            avg_fail_severity=self.avg_fail_severity,
        )

    def _percentile_threshold(self, pass_sigma: np.ndarray) -> ThresholdResult:
        """Calculate threshold from passing data only (few/no failures)."""
        # Use 95th percentile of passing + 10% margin
        p95 = float(np.percentile(pass_sigma, 95))
        threshold = p95 * 1.1

        # Lower confidence since we don't have failure data
        # More samples = higher confidence
        sample_factor = min(1.0, len(pass_sigma) / 200)
        confidence = 0.5 + 0.2 * sample_factor

        return ThresholdResult(
            threshold=threshold,
            confidence=confidence,
            method='percentile',
            pass_sigma_mean=self.pass_sigma_mean,
            pass_sigma_std=self.pass_sigma_std,
            pass_sigma_max=float(np.max(pass_sigma)),
            n_pass=self.n_pass,
            n_fail=self.n_fail,
        )

    def _weighted_threshold(
        self,
        sigma: np.ndarray,
        is_pass: np.ndarray,
        fail_points: Optional[np.ndarray]
    ) -> ThresholdResult:
        """
        Calculate threshold with severity weighting when overlap exists.

        Uses fail_points to weight samples - more severe failures
        have more influence on threshold placement.
        """
        pass_sigma = sigma[is_pass]
        fail_sigma = sigma[~is_pass]

        # Calculate weights from severity
        if fail_points is not None:
            fail_weights = fail_points[~is_pass]
            # Normalize weights to [1, 3] range
            max_fp = fail_weights.max() if len(fail_weights) > 0 and fail_weights.max() > 0 else 1
            fail_weights = 1 + 2 * (fail_weights / max_fp)
        else:
            fail_weights = np.ones(len(fail_sigma))

        # Weighted percentile of failures (lower = more aggressive)
        # Higher severity failures pull threshold down
        if len(fail_sigma) > 0:
            # Weighted percentile algorithm:
            # 1. Sort failures by sigma (ascending) - low sigma failures are borderline
            sorted_idx = np.argsort(fail_sigma)
            sorted_sigma = fail_sigma[sorted_idx]
            sorted_weights = fail_weights[sorted_idx]

            # 2. Build cumulative weight distribution
            # Each sample contributes its weight to cumulative sum
            cumsum = np.cumsum(sorted_weights)
            # Normalize to [0,1] range for percentile lookup
            cumsum_norm = cumsum / cumsum[-1]

            # 3. Find weighted 10th percentile of failures
            # This captures the low-sigma failures that severe failures "pull down"
            # More severe failures = more weight = shifts distribution toward low sigma
            p10_idx = np.searchsorted(cumsum_norm, 0.10)
            weighted_fail_low = sorted_sigma[min(p10_idx, len(sorted_sigma) - 1)]
        else:
            logger.debug(f"Using fallback fail_sigma_min for {self.model_name}")
            weighted_fail_low = self.fail_sigma_min

        # 95th percentile of passing
        p95_pass = float(np.percentile(pass_sigma, 95))

        # Threshold between high-passing and low-failing (weighted)
        # Bias towards catching failures (threshold closer to passing)
        threshold = 0.6 * p95_pass + 0.4 * weighted_fail_low

        # Moderate confidence due to overlap
        overlap_ratio = len(pass_sigma[pass_sigma > weighted_fail_low]) / len(pass_sigma)
        confidence = max(0.4, 0.7 - 0.3 * overlap_ratio)

        return ThresholdResult(
            threshold=threshold,
            confidence=confidence,
            method='weighted',
            pass_sigma_mean=self.pass_sigma_mean,
            pass_sigma_std=self.pass_sigma_std,
            pass_sigma_max=float(np.max(pass_sigma)),
            fail_sigma_min=self.fail_sigma_min,
            fail_sigma_mean=self.fail_sigma_mean,
            n_pass=self.n_pass,
            n_fail=self.n_fail,
            avg_fail_severity=self.avg_fail_severity,
        )

    def _fallback_threshold(self, linearity_spec: Optional[float]) -> ThresholdResult:
        """Formula-based fallback when insufficient data."""
        # Default formula: spec / 200
        spec = linearity_spec or 0.01
        threshold = spec / 200

        # Ensure reasonable bounds
        threshold = max(0.00005, min(threshold, 0.01))

        return ThresholdResult(
            threshold=threshold,
            confidence=0.3,  # Low confidence for fallback
            method='fallback',
            n_pass=self.n_pass,
            n_fail=self.n_fail,
        )

    def get_threshold(self) -> Optional[float]:
        """Get calculated threshold, or None if not calculated."""
        return self.threshold

    def get_confidence(self) -> Optional[float]:
        """Get threshold confidence, or None if not calculated."""
        return self.confidence

    def get_state(self) -> ThresholdOptimizerState:
        """Get serializable state for database storage."""
        return ThresholdOptimizerState(
            model_name=self.model_name,
            threshold=self.threshold,
            confidence=self.confidence,
            method=self.method,
            n_samples=self.n_samples,
            n_pass=self.n_pass,
            n_fail=self.n_fail,
            pass_sigma_mean=self.pass_sigma_mean,
            pass_sigma_std=self.pass_sigma_std,
            fail_sigma_mean=self.fail_sigma_mean,
            avg_fail_severity=self.avg_fail_severity,
            calculated_date=self.calculated_date,
        )

    def load_state(self, state: ThresholdOptimizerState) -> None:
        """Load state from database."""
        self.model_name = state.model_name
        self.threshold = state.threshold
        self.confidence = state.confidence
        self.method = state.method
        self.n_samples = state.n_samples
        self.n_pass = state.n_pass
        self.n_fail = state.n_fail
        self.pass_sigma_mean = state.pass_sigma_mean
        self.pass_sigma_std = state.pass_sigma_std
        self.fail_sigma_mean = state.fail_sigma_mean
        self.avg_fail_severity = state.avg_fail_severity
        self.calculated_date = state.calculated_date
        self.is_calculated = state.threshold is not None

    def get_statistics(self) -> Dict[str, float]:
        """Get threshold statistics as dictionary."""
        return {
            'threshold': self.threshold or 0,
            'confidence': self.confidence or 0,
            'n_samples': self.n_samples,
            'n_pass': self.n_pass,
            'n_fail': self.n_fail,
            'pass_sigma_mean': self.pass_sigma_mean,
            'pass_sigma_std': self.pass_sigma_std,
            'pass_sigma_max': self.pass_sigma_max,
            'fail_sigma_min': self.fail_sigma_min,
            'fail_sigma_mean': self.fail_sigma_mean,
            'avg_fail_severity': self.avg_fail_severity,
        }
