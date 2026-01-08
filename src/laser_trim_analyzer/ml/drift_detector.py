"""
Per-Model Drift Detector for Laser Trim Analyzer v3.

Detects quality shifts (drift) using CUSUM and EWMA algorithms.
Each product model gets its own baseline and detection state.

Part of the per-model ML redesign - replaces global DriftDetector.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class DriftDirection(Enum):
    """Direction of detected drift."""
    UP = "up"  # Quality degrading (sigma increasing)
    DOWN = "down"  # Quality improving (sigma decreasing)
    STABLE = "stable"  # No significant drift


@dataclass
class DriftResult:
    """Result of drift detection check."""
    is_drifting: bool
    direction: DriftDirection
    severity: float  # 0-1, how severe the drift is
    cusum_signal: bool
    ewma_signal: bool
    cusum_value: float
    ewma_value: float
    deviation_from_baseline: float  # In standard deviations
    message: Optional[str] = None


@dataclass
class DriftDetectorState:
    """Serializable state for database storage."""
    model_name: str
    has_baseline: bool

    # Baseline statistics
    baseline_mean: Optional[float]
    baseline_std: Optional[float]
    baseline_p5: Optional[float]
    baseline_p50: Optional[float]
    baseline_p95: Optional[float]
    baseline_samples: int
    baseline_cutoff_date: Optional[datetime]  # Files older than this were used for baseline

    # CUSUM state
    cusum_pos: float
    cusum_neg: float
    cusum_threshold: float

    # EWMA state
    ewma_value: Optional[float]
    ewma_lambda: float

    # Current drift status
    is_drifting: bool
    drift_direction: Optional[str]
    drift_start_date: Optional[datetime]

    # Tracking
    samples_since_baseline: int
    updated_date: Optional[datetime]


class ModelDriftDetector:
    """
    Per-model drift detection with CUSUM and EWMA.

    Detects quality shifts in sigma_gradient values that may indicate:
    - Process drift (gradual degradation)
    - Step changes (sudden quality shifts)
    - Cyclical patterns (recurring issues)

    Uses two complementary methods:
    - CUSUM: Detects small persistent shifts
    - EWMA: Detects larger recent changes

    Each model has its own baseline statistics and detection state.
    """

    # Default CUSUM parameters (based on Montgomery's SPC recommendations)
    # K = 0.5: Standard slack value; detects shifts of ~1 std dev within ~10 samples
    DEFAULT_CUSUM_K = 0.5  # Slack value (in std devs) - smaller = more sensitive
    # H = 5.0: Decision threshold; balances false alarms vs detection speed
    DEFAULT_CUSUM_H = 5.0  # Decision threshold (in std devs) - larger = fewer false alarms

    # Default EWMA parameters (per Roberts 1959 EWMA control chart theory)
    # Lambda = 0.2: Weights recent observations; 0.2 means last ~5 samples dominate
    DEFAULT_EWMA_LAMBDA = 0.2  # Smoothing factor (0-1) - larger = more reactive
    # L = 3.0: Standard 3-sigma control limit for ~0.27% false alarm rate
    DEFAULT_EWMA_L = 3.0  # Control limit (in std devs)

    # Minimum samples for reliable baseline statistics (per CLT guidelines)
    MIN_BASELINE_SAMPLES = 30

    def __init__(
        self,
        model_name: str,
        cusum_k: float = DEFAULT_CUSUM_K,
        cusum_h: float = DEFAULT_CUSUM_H,
        ewma_lambda: float = DEFAULT_EWMA_LAMBDA,
        ewma_l: float = DEFAULT_EWMA_L
    ):
        """
        Initialize drift detector for a specific model.

        Args:
            model_name: Product model number
            cusum_k: CUSUM slack value (default 0.5 std devs)
            cusum_h: CUSUM decision threshold (default 5.0 std devs)
            ewma_lambda: EWMA smoothing factor (default 0.2)
            ewma_l: EWMA control limit (default 3.0 std devs)
        """
        self.model_name = model_name

        # CUSUM parameters
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h

        # EWMA parameters
        self.ewma_lambda = ewma_lambda
        self.ewma_l = ewma_l

        # Baseline statistics (set from historical data)
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None
        self.baseline_p5: Optional[float] = None
        self.baseline_p50: Optional[float] = None
        self.baseline_p95: Optional[float] = None
        self.baseline_samples: int = 0
        self.baseline_cutoff_date: Optional[datetime] = None  # Files older than this were used for baseline

        # CUSUM state (accumulates deviations)
        self.cusum_pos: float = 0.0  # Positive accumulator (detecting upward drift)
        self.cusum_neg: float = 0.0  # Negative accumulator (detecting downward drift)

        # EWMA state
        self.ewma_value: Optional[float] = None

        # Drift status
        self.is_drifting: bool = False
        self.drift_direction: Optional[DriftDirection] = None
        self.drift_start_date: Optional[datetime] = None

        # Tracking
        self.samples_since_baseline: int = 0
        self.has_baseline: bool = False
        self.updated_date: Optional[datetime] = None

        # Recovery tracking - require consecutive samples near baseline before resetting CUSUM
        self._consecutive_recovered: int = 0
        self.RECOVERY_SAMPLES_REQUIRED: int = 5  # Need 5 consecutive "recovered" samples to reset

        # Peak CUSUM tracking - shows maximum CUSUM reached during detection period
        self._peak_cusum: float = 0.0

    def set_baseline(self, sigma_values: np.ndarray, cutoff_date: Optional[datetime] = None) -> bool:
        """
        Set baseline statistics from historical data.

        Should be called with a representative sample of sigma_gradient
        values from when the process was in control.

        Args:
            sigma_values: Array of sigma_gradient values
            cutoff_date: Date marking end of baseline period. Files newer than this
                         will be checked for drift. If None, uses current datetime.

        Returns:
            True if baseline was set successfully
        """
        if len(sigma_values) < self.MIN_BASELINE_SAMPLES:
            logger.warning(
                f"DriftDetector[{self.model_name}]: Need at least "
                f"{self.MIN_BASELINE_SAMPLES} samples for baseline, got {len(sigma_values)}"
            )
            return False

        # Calculate baseline statistics
        self.baseline_mean = float(np.mean(sigma_values))
        self.baseline_std = float(np.std(sigma_values))
        self.baseline_p5 = float(np.percentile(sigma_values, 5))
        self.baseline_p50 = float(np.percentile(sigma_values, 50))
        self.baseline_p95 = float(np.percentile(sigma_values, 95))
        self.baseline_samples = len(sigma_values)
        self.baseline_cutoff_date = cutoff_date if cutoff_date else datetime.now()

        # Reset detection state
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.ewma_value = self.baseline_mean
        self.samples_since_baseline = 0
        self.is_drifting = False
        self.drift_direction = None
        self.drift_start_date = None
        self._consecutive_recovered = 0
        self._peak_cusum = 0.0

        self.has_baseline = True
        self.updated_date = datetime.now()

        logger.info(
            f"DriftDetector[{self.model_name}] baseline set - "
            f"Mean: {self.baseline_mean:.6f}, Std: {self.baseline_std:.6f}, "
            f"Samples: {self.baseline_samples}, Cutoff: {self.baseline_cutoff_date}"
        )

        return True

    def detect(self, sigma_value: float) -> DriftResult:
        """
        Check if a new sigma value indicates drift.

        Args:
            sigma_value: New sigma_gradient value to check

        Returns:
            DriftResult with detection status and details
        """
        if not self.has_baseline or self.baseline_mean is None or self.baseline_std is None:
            return DriftResult(
                is_drifting=False,
                direction=DriftDirection.STABLE,
                severity=0.0,
                cusum_signal=False,
                ewma_signal=False,
                cusum_value=0.0,
                ewma_value=0.0,
                deviation_from_baseline=0.0,
                message="No baseline set"
            )

        # Standardize value
        if self.baseline_std > 0:
            z = (sigma_value - self.baseline_mean) / self.baseline_std
        else:
            z = 0.0

        # Update CUSUM
        cusum_signal, cusum_direction = self._update_cusum(z)

        # Track peak CUSUM value (maximum reached during detection period)
        current_cusum = max(self.cusum_pos, self.cusum_neg)
        self._peak_cusum = max(self._peak_cusum, current_cusum)

        # Update EWMA
        ewma_signal, ewma_direction = self._update_ewma(sigma_value)

        # Check if process has recovered (EWMA within 0.5 std of baseline mean)
        # Require consecutive samples near baseline before resetting CUSUM
        if self.ewma_value is not None and self.baseline_mean is not None and self.baseline_std is not None:
            deviation_from_mean = abs(self.ewma_value - self.baseline_mean)
            if deviation_from_mean <= 0.5 * self.baseline_std:
                # Sample is near baseline - increment recovery counter
                self._consecutive_recovered += 1

                # Only reset CUSUM after enough consecutive recovered samples
                if self._consecutive_recovered >= self.RECOVERY_SAMPLES_REQUIRED:
                    if self.cusum_pos > self.cusum_h or self.cusum_neg > self.cusum_h:
                        logger.debug(
                            f"DriftDetector[{self.model_name}] EWMA stable for "
                            f"{self._consecutive_recovered} samples, resetting CUSUM"
                        )
                    self.cusum_pos = 0.0
                    self.cusum_neg = 0.0
                    cusum_signal = False
            else:
                # Sample is outside recovery zone - reset counter
                self._consecutive_recovered = 0

        # Combine signals - only signal drift if CUSUM triggered AND still outside normal range
        is_drifting = cusum_signal or ewma_signal

        # Determine direction based on CURRENT state (EWMA vs mean)
        if is_drifting and self.ewma_value is not None and self.baseline_mean is not None:
            if self.ewma_value > self.baseline_mean:
                direction = DriftDirection.UP
            elif self.ewma_value < self.baseline_mean:
                direction = DriftDirection.DOWN
            else:
                direction = DriftDirection.STABLE
        else:
            direction = DriftDirection.STABLE

        # Calculate severity based on how far EWMA is from mean (in std devs)
        if all(v is not None for v in [self.ewma_value, self.baseline_mean, self.baseline_std]) and self.baseline_std > 0:
            severity = min(1.0, abs(self.ewma_value - self.baseline_mean) / (3 * self.baseline_std))
        else:
            severity = 0.0

        # Track drift start
        if is_drifting and not self.is_drifting:
            self.drift_start_date = datetime.now()

        self.is_drifting = is_drifting
        self.drift_direction = direction if is_drifting else None
        self.samples_since_baseline += 1
        self.updated_date = datetime.now()

        # Generate message
        message = None
        if is_drifting:
            if direction == DriftDirection.UP:
                message = f"Quality degrading - sigma currently above baseline ({severity:.0%} severity)"
            elif direction == DriftDirection.DOWN:
                message = f"Quality improved - sigma currently below baseline ({severity:.0%} severity)"
            else:
                message = f"Process deviation detected ({severity:.0%} severity)"

        return DriftResult(
            is_drifting=is_drifting,
            direction=direction,
            severity=severity,
            cusum_signal=cusum_signal,
            ewma_signal=ewma_signal,
            cusum_value=max(self.cusum_pos, self.cusum_neg),
            ewma_value=self.ewma_value or 0.0,
            deviation_from_baseline=z,
            message=message
        )

    def _update_cusum(self, z: float) -> Tuple[bool, DriftDirection]:
        """
        Update CUSUM accumulators and check for signal.

        CUSUM (Cumulative Sum) accumulates deviations from target.
        Good at detecting small persistent shifts.

        Args:
            z: Standardized value (in std devs from mean)

        Returns:
            Tuple of (signal_detected, direction)
        """
        # Update accumulators
        # Positive CUSUM: accumulates positive deviations
        self.cusum_pos = max(0, self.cusum_pos + z - self.cusum_k)
        # Negative CUSUM: accumulates negative deviations
        self.cusum_neg = max(0, self.cusum_neg - z - self.cusum_k)

        # Check for signals
        if self.cusum_pos > self.cusum_h:
            return True, DriftDirection.UP
        elif self.cusum_neg > self.cusum_h:
            return True, DriftDirection.DOWN
        else:
            return False, DriftDirection.STABLE

    def _update_ewma(self, value: float) -> Tuple[bool, DriftDirection]:
        """
        Update EWMA and check for signal.

        EWMA (Exponentially Weighted Moving Average) gives more weight
        to recent values. Good at detecting larger recent changes.

        Args:
            value: Raw sigma_gradient value

        Returns:
            Tuple of (signal_detected, direction)
        """
        if self.ewma_value is None:
            self.ewma_value = value
            return False, DriftDirection.STABLE

        # Update EWMA
        self.ewma_value = self.ewma_lambda * value + (1 - self.ewma_lambda) * self.ewma_value

        # Check control limits
        if self.baseline_std is not None and self.baseline_std > 0:
            # Calculate EWMA standard deviation
            ewma_std = self.baseline_std * np.sqrt(
                self.ewma_lambda / (2 - self.ewma_lambda) *
                (1 - (1 - self.ewma_lambda) ** (2 * self.samples_since_baseline))
            )

            upper_limit = self.baseline_mean + self.ewma_l * ewma_std
            lower_limit = self.baseline_mean - self.ewma_l * ewma_std

            if self.ewma_value > upper_limit:
                return True, DriftDirection.UP
            elif self.ewma_value < lower_limit:
                return True, DriftDirection.DOWN

        return False, DriftDirection.STABLE

    def reset(self) -> None:
        """Reset detection state while keeping baseline."""
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.ewma_value = self.baseline_mean
        self.samples_since_baseline = 0
        self.is_drifting = False
        self.drift_direction = None
        self.drift_start_date = None
        self._consecutive_recovered = 0
        self._peak_cusum = 0.0
        self.updated_date = datetime.now()

        logger.debug(f"DriftDetector[{self.model_name}] reset")

    def get_control_limits(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Get control limits for charting.

        Returns:
            Tuple of (lower_limit, center, upper_limit), or (None, None, None)
        """
        if not self.has_baseline or self.baseline_mean is None or self.baseline_std is None:
            return None, None, None

        center = self.baseline_mean
        # Use 3-sigma limits
        lower = self.baseline_mean - 3 * self.baseline_std
        upper = self.baseline_mean + 3 * self.baseline_std

        return lower, center, upper

    def get_state(self) -> DriftDetectorState:
        """Get serializable state for database storage."""
        return DriftDetectorState(
            model_name=self.model_name,
            has_baseline=self.has_baseline,
            baseline_mean=self.baseline_mean,
            baseline_std=self.baseline_std,
            baseline_p5=self.baseline_p5,
            baseline_p50=self.baseline_p50,
            baseline_p95=self.baseline_p95,
            baseline_samples=self.baseline_samples,
            baseline_cutoff_date=self.baseline_cutoff_date,
            cusum_pos=self.cusum_pos,
            cusum_neg=self.cusum_neg,
            cusum_threshold=self.cusum_h,
            ewma_value=self.ewma_value,
            ewma_lambda=self.ewma_lambda,
            is_drifting=self.is_drifting,
            drift_direction=self.drift_direction.value if self.drift_direction else None,
            drift_start_date=self.drift_start_date,
            samples_since_baseline=self.samples_since_baseline,
            updated_date=self.updated_date,
        )

    def load_state(self, state: DriftDetectorState) -> None:
        """Load state from database."""
        self.model_name = state.model_name
        self.has_baseline = state.has_baseline
        self.baseline_mean = state.baseline_mean
        self.baseline_std = state.baseline_std
        self.baseline_p5 = state.baseline_p5
        self.baseline_p50 = state.baseline_p50
        self.baseline_p95 = state.baseline_p95
        self.baseline_samples = state.baseline_samples
        self.baseline_cutoff_date = state.baseline_cutoff_date
        self.cusum_pos = state.cusum_pos
        self.cusum_neg = state.cusum_neg
        self.cusum_h = state.cusum_threshold
        self.ewma_value = state.ewma_value
        self.ewma_lambda = state.ewma_lambda
        self.is_drifting = state.is_drifting
        self.drift_direction = DriftDirection(state.drift_direction) if state.drift_direction else None
        self.drift_start_date = state.drift_start_date
        self.samples_since_baseline = state.samples_since_baseline
        self.updated_date = state.updated_date

    def get_statistics(self) -> Dict[str, float]:
        """Get detector statistics as dictionary."""
        return {
            'has_baseline': 1.0 if self.has_baseline else 0.0,
            'baseline_mean': self.baseline_mean or 0.0,
            'baseline_std': self.baseline_std or 0.0,
            'baseline_p5': self.baseline_p5 or 0.0,
            'baseline_p50': self.baseline_p50 or 0.0,
            'baseline_p95': self.baseline_p95 or 0.0,
            'baseline_samples': float(self.baseline_samples),
            'cusum_pos': self.cusum_pos,
            'cusum_neg': self.cusum_neg,
            'ewma_value': self.ewma_value or 0.0,
            'is_drifting': 1.0 if self.is_drifting else 0.0,
            'samples_since_baseline': float(self.samples_since_baseline),
        }
