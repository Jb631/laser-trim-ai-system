"""
Drift Detector for Laser Trim Analyzer v3.

Detects manufacturing process drift using a hybrid approach:
1. CUSUM (Cumulative Sum) - catches gradual shifts
2. EWMA (Exponentially Weighted Moving Average) - catches trends
3. IsolationForest - catches multi-dimensional anomalies

Simplified from v2's implementation with better statistical methods.
"""

import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from laser_trim_analyzer.config import get_config

logger = logging.getLogger(__name__)


class DriftDirection(Enum):
    """Direction of detected drift."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass
class DriftConfig:
    """Configuration for drift detector."""
    # CUSUM parameters
    cusum_threshold: float = 5.0  # Standard deviations
    cusum_slack: float = 0.5  # Slack parameter (k)

    # EWMA parameters
    ewma_lambda: float = 0.2  # Smoothing factor (0-1)
    ewma_control_limit: float = 3.0  # Control limit in std devs

    # Isolation Forest parameters
    n_estimators: int = 100
    contamination: float = 0.05  # Expected fraction of outliers

    # Window sizes
    baseline_window: int = 100  # Samples for establishing baseline
    analysis_window: int = 50  # Samples for drift analysis


@dataclass
class DriftResult:
    """Result of drift detection."""
    drift_detected: bool
    direction: DriftDirection
    severity: str  # "none", "minor", "moderate", "severe"
    confidence: float  # 0-1
    cusum_signal: bool
    ewma_signal: bool
    anomaly_signal: bool
    details: Dict[str, Any]


class DriftDetector:
    """
    Hybrid drift detector combining statistical and ML methods.

    Uses three complementary approaches:
    1. CUSUM: Detects small, persistent shifts (mean changes)
    2. EWMA: Detects trends and gradual changes
    3. IsolationForest: Detects multi-dimensional anomalies

    All three are lightweight and work well on 8GB RAM systems.
    """

    def __init__(self, config: Optional[DriftConfig] = None):
        """
        Initialize drift detector.

        Args:
            config: Optional configuration for detection parameters
        """
        self.config = config or DriftConfig()
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None
        self.baseline_set = False

        # CUSUM accumulators
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

        # EWMA state
        self.ewma_value: Optional[float] = None

        # Isolation Forest (if available)
        self.isolation_forest: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.is_trained = False

        # History for analysis
        self.drift_history: List[Dict[str, Any]] = []

    def set_baseline(self, values: np.ndarray) -> bool:
        """
        Establish baseline from historical "normal" data.

        Args:
            values: Array of values representing normal operation

        Returns:
            True if baseline was set successfully
        """
        if len(values) < 20:
            logger.warning("Need at least 20 samples to set baseline")
            return False

        self.baseline_mean = float(np.mean(values))
        self.baseline_std = float(np.std(values, ddof=1))

        # Prevent division by zero
        if self.baseline_std < 1e-10:
            self.baseline_std = 1e-10

        # Reset CUSUM accumulators
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

        # Initialize EWMA
        self.ewma_value = self.baseline_mean

        self.baseline_set = True
        logger.info(f"Baseline set: mean={self.baseline_mean:.6f}, std={self.baseline_std:.6f}")

        return True

    def train_anomaly_detector(self, data: pd.DataFrame) -> bool:
        """
        Train IsolationForest for multi-dimensional anomaly detection.

        Args:
            data: DataFrame with numeric features (sigma_gradient, etc.)

        Returns:
            True if training successful
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not available, skipping anomaly detector training")
            return False

        try:
            # Prepare features
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns for anomaly detector")
                return False

            X = data[numeric_cols].dropna()
            if len(X) < 50:
                logger.warning("Need at least 50 samples for anomaly detector")
                return False

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Train Isolation Forest
            self.isolation_forest = IsolationForest(
                n_estimators=self.config.n_estimators,
                contamination=self.config.contamination,
                random_state=42,
                n_jobs=-1
            )
            self.isolation_forest.fit(X_scaled)

            self.is_trained = True
            logger.info(f"Anomaly detector trained on {len(X)} samples, {len(numeric_cols)} features")

            return True

        except Exception as e:
            logger.error(f"Failed to train anomaly detector: {e}")
            return False

    def detect(self, value: float) -> DriftResult:
        """
        Detect drift for a single value (online detection).

        Args:
            value: New observation value

        Returns:
            DriftResult with detection status
        """
        if not self.baseline_set:
            logger.warning("Baseline not set, cannot detect drift")
            return DriftResult(
                drift_detected=False,
                direction=DriftDirection.UNKNOWN,
                severity="none",
                confidence=0.0,
                cusum_signal=False,
                ewma_signal=False,
                anomaly_signal=False,
                details={"error": "Baseline not set"}
            )

        # Standardize value
        z = (value - self.baseline_mean) / self.baseline_std

        # 1. CUSUM detection
        cusum_signal, cusum_direction = self._cusum_detect(z)

        # 2. EWMA detection
        ewma_signal, ewma_direction = self._ewma_detect(value)

        # 3. Combine signals
        drift_detected = cusum_signal or ewma_signal
        signals = [cusum_signal, ewma_signal]

        # Determine direction (majority vote)
        if cusum_direction == ewma_direction:
            direction = cusum_direction
        elif cusum_signal:
            direction = cusum_direction
        else:
            direction = ewma_direction

        # Calculate severity and confidence
        n_signals = sum(signals)
        if n_signals == 0:
            severity = "none"
            confidence = 0.0
        elif n_signals == 1:
            severity = "minor"
            confidence = 0.5
        else:
            severity = "moderate"
            confidence = 0.75

        # Increase severity if deviation is large
        if abs(z) > 4:
            severity = "severe"
            confidence = min(confidence + 0.2, 1.0)

        return DriftResult(
            drift_detected=drift_detected,
            direction=direction,
            severity=severity,
            confidence=confidence,
            cusum_signal=cusum_signal,
            ewma_signal=ewma_signal,
            anomaly_signal=False,  # Not used in single-value detection
            details={
                "z_score": z,
                "cusum_pos": self.cusum_pos,
                "cusum_neg": self.cusum_neg,
                "ewma_value": self.ewma_value,
                "baseline_mean": self.baseline_mean,
                "baseline_std": self.baseline_std,
            }
        )

    def detect_batch(
        self,
        values: np.ndarray,
        features: Optional[pd.DataFrame] = None
    ) -> DriftResult:
        """
        Detect drift in a batch of values.

        Args:
            values: Array of recent observations
            features: Optional DataFrame for multi-dimensional anomaly detection

        Returns:
            DriftResult with aggregate detection
        """
        if len(values) == 0:
            return DriftResult(
                drift_detected=False,
                direction=DriftDirection.UNKNOWN,
                severity="none",
                confidence=0.0,
                cusum_signal=False,
                ewma_signal=False,
                anomaly_signal=False,
                details={"error": "No values provided"}
            )

        # If no baseline, use first part of batch as baseline
        if not self.baseline_set:
            baseline_size = min(len(values) // 3, self.config.baseline_window)
            if baseline_size >= 20:
                self.set_baseline(values[:baseline_size])
                values = values[baseline_size:]
            else:
                # Can't establish baseline, do simple statistical test
                return self._simple_statistical_test(values)

        # Run detection on each value
        results = []
        for v in values:
            result = self.detect(v)
            results.append(result)

        # Aggregate results
        n_cusum = sum(r.cusum_signal for r in results)
        n_ewma = sum(r.ewma_signal for r in results)
        n_drift = sum(r.drift_detected for r in results)

        # Check for multi-dimensional anomalies if features provided
        anomaly_signal = False
        anomaly_rate = 0.0
        if features is not None and self.is_trained and HAS_SKLEARN:
            anomaly_signal, anomaly_rate = self._detect_anomalies(features)

        # Determine overall drift
        drift_rate = n_drift / len(results) if results else 0
        drift_detected = drift_rate > 0.3 or anomaly_rate > 0.15

        # Determine direction from trend
        if len(values) >= 5:
            slope = np.polyfit(range(len(values)), values, 1)[0]
            if slope > self.baseline_std * 0.1:
                direction = DriftDirection.INCREASING
            elif slope < -self.baseline_std * 0.1:
                direction = DriftDirection.DECREASING
            else:
                direction = DriftDirection.STABLE
        else:
            direction = DriftDirection.UNKNOWN

        # Calculate severity
        if not drift_detected:
            severity = "none"
            confidence = 1.0 - drift_rate
        elif drift_rate < 0.3:
            severity = "minor"
            confidence = 0.5
        elif drift_rate < 0.5:
            severity = "moderate"
            confidence = 0.7
        else:
            severity = "severe"
            confidence = 0.9

        # Record in history
        self.drift_history.append({
            "timestamp": datetime.now(),
            "drift_detected": drift_detected,
            "direction": direction.value,
            "severity": severity,
            "drift_rate": drift_rate,
            "anomaly_rate": anomaly_rate,
            "n_samples": len(values)
        })

        return DriftResult(
            drift_detected=drift_detected,
            direction=direction,
            severity=severity,
            confidence=confidence,
            cusum_signal=n_cusum > len(results) * 0.3,
            ewma_signal=n_ewma > len(results) * 0.3,
            anomaly_signal=anomaly_signal,
            details={
                "n_samples": len(results),
                "drift_rate": drift_rate,
                "cusum_rate": n_cusum / len(results),
                "ewma_rate": n_ewma / len(results),
                "anomaly_rate": anomaly_rate,
                "mean_shift": (np.mean(values) - self.baseline_mean) / self.baseline_std,
            }
        )

    def _cusum_detect(self, z: float) -> Tuple[bool, DriftDirection]:
        """
        CUSUM (Cumulative Sum) detection.

        Detects small, persistent shifts in the mean.
        """
        k = self.config.cusum_slack
        h = self.config.cusum_threshold

        # Update accumulators
        self.cusum_pos = max(0, self.cusum_pos + z - k)
        self.cusum_neg = max(0, self.cusum_neg - z - k)

        # Check for signal
        signal_pos = self.cusum_pos > h
        signal_neg = self.cusum_neg > h

        if signal_pos:
            return True, DriftDirection.INCREASING
        elif signal_neg:
            return True, DriftDirection.DECREASING
        else:
            return False, DriftDirection.STABLE

    def _ewma_detect(self, value: float) -> Tuple[bool, DriftDirection]:
        """
        EWMA (Exponentially Weighted Moving Average) detection.

        Detects trends and gradual changes.
        """
        lam = self.config.ewma_lambda
        L = self.config.ewma_control_limit

        # Update EWMA
        if self.ewma_value is None:
            self.ewma_value = value
        else:
            self.ewma_value = lam * value + (1 - lam) * self.ewma_value

        # Calculate control limits
        # EWMA std = std * sqrt(lam/(2-lam))
        ewma_std = self.baseline_std * np.sqrt(lam / (2 - lam))
        ucl = self.baseline_mean + L * ewma_std
        lcl = self.baseline_mean - L * ewma_std

        # Check for signal
        if self.ewma_value > ucl:
            return True, DriftDirection.INCREASING
        elif self.ewma_value < lcl:
            return True, DriftDirection.DECREASING
        else:
            return False, DriftDirection.STABLE

    def _detect_anomalies(self, features: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect anomalies using IsolationForest.

        Returns:
            (signal, anomaly_rate)
        """
        if not self.is_trained or self.scaler is None or self.isolation_forest is None:
            return False, 0.0

        try:
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            X = features[numeric_cols].dropna()

            if len(X) == 0:
                return False, 0.0

            X_scaled = self.scaler.transform(X)
            predictions = self.isolation_forest.predict(X_scaled)

            # -1 = anomaly, 1 = normal
            anomaly_rate = (predictions == -1).mean()
            signal = anomaly_rate > self.config.contamination * 2  # Double expected rate

            return signal, float(anomaly_rate)

        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return False, 0.0

    def _simple_statistical_test(self, values: np.ndarray) -> DriftResult:
        """
        Simple statistical test when no baseline available.

        Uses t-test comparing first half to second half.
        """
        if len(values) < 10:
            return DriftResult(
                drift_detected=False,
                direction=DriftDirection.UNKNOWN,
                severity="none",
                confidence=0.0,
                cusum_signal=False,
                ewma_signal=False,
                anomaly_signal=False,
                details={"error": "Insufficient data"}
            )

        mid = len(values) // 2
        first_half = values[:mid]
        second_half = values[mid:]

        # t-test
        t_stat, p_value = stats.ttest_ind(first_half, second_half)

        drift_detected = p_value < 0.05
        direction = DriftDirection.INCREASING if t_stat < 0 else DriftDirection.DECREASING

        if p_value < 0.01:
            severity = "severe"
            confidence = 0.95
        elif p_value < 0.05:
            severity = "moderate"
            confidence = 0.75
        else:
            severity = "none"
            direction = DriftDirection.STABLE
            confidence = 1 - p_value

        return DriftResult(
            drift_detected=drift_detected,
            direction=direction,
            severity=severity,
            confidence=confidence,
            cusum_signal=False,
            ewma_signal=False,
            anomaly_signal=False,
            details={
                "t_statistic": t_stat,
                "p_value": p_value,
                "first_half_mean": np.mean(first_half),
                "second_half_mean": np.mean(second_half),
            }
        )

    def reset(self) -> None:
        """Reset detector state (keep baseline)."""
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        if self.baseline_mean is not None:
            self.ewma_value = self.baseline_mean

    def save(self, path: Path) -> bool:
        """Save detector state to disk."""
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            state = {
                'config': self.config,
                'baseline_mean': self.baseline_mean,
                'baseline_std': self.baseline_std,
                'baseline_set': self.baseline_set,
                'scaler': self.scaler,
                'isolation_forest': self.isolation_forest,
                'is_trained': self.is_trained,
            }

            with open(path, 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"Drift detector saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save drift detector: {e}")
            return False

    def load(self, path: Path) -> bool:
        """Load detector state from disk."""
        try:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Drift detector file not found: {path}")
                return False

            with open(path, 'rb') as f:
                state = pickle.load(f)

            self.config = state.get('config', self.config)
            self.baseline_mean = state.get('baseline_mean')
            self.baseline_std = state.get('baseline_std')
            self.baseline_set = state.get('baseline_set', False)
            self.scaler = state.get('scaler')
            self.isolation_forest = state.get('isolation_forest')
            self.is_trained = state.get('is_trained', False)

            # Reset runtime state
            self.reset()

            logger.info(f"Drift detector loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load drift detector: {e}")
            return False

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection state."""
        return {
            "baseline_set": self.baseline_set,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "anomaly_detector_trained": self.is_trained,
            "cusum_pos": self.cusum_pos,
            "cusum_neg": self.cusum_neg,
            "ewma_value": self.ewma_value,
            "drift_history_length": len(self.drift_history),
        }
