"""
Base analyzer class for all laser trim analysis modules.

Provides common interface, validation, and utility methods.
"""

import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type
)
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal

from laser_trim_analyzer.core.config import Config, get_config
from laser_trim_analyzer.core.models import (
    AnalysisStatus, RiskCategory, SystemType
)
from laser_trim_analyzer.core.constants import (
    FILTER_SAMPLING_FREQUENCY, FILTER_CUTOFF_FREQUENCY,
    END_POINT_FILTER_COUNT
)


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analyzers.

    Provides common functionality for data validation, filtering,
    and utility methods used across different analysis types.
    """

    def __init__(self, config: Optional[Config] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize base analyzer.

        Args:
            config: Application configuration
            logger: Logger instance
        """
        self.config = config or get_config()
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Analysis parameters from config
        self.analysis_config = self.config.analysis

    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Any:
        """
        Perform analysis on the provided data.

        Args:
            data: Input data dictionary

        Returns:
            Analysis result (specific to each analyzer)
        """
        pass

    def validate_data(self, positions: List[float],
                      errors: List[float]) -> Tuple[bool, List[str]]:
        """
        Validate input data for analysis.

        Args:
            positions: Position measurements
            errors: Error measurements

        Returns:
            Tuple of (is_valid, validation_messages)
        """
        messages = []

        # Check if data exists
        if not positions or not errors:
            messages.append("Missing position or error data")
            return False, messages

        # Check data length consistency
        if len(positions) != len(errors):
            messages.append(f"Position/error length mismatch: {len(positions)} vs {len(errors)}")
            return False, messages

        # Check minimum data points
        if len(positions) < 10:
            messages.append(f"Insufficient data points: {len(positions)} (minimum: 10)")
            return False, messages

        # Check for NaN values
        if any(np.isnan(positions)) or any(np.isnan(errors)):
            messages.append("Data contains NaN values")
            return False, messages

        # Check for duplicate positions
        unique_positions = len(set(positions))
        if unique_positions < len(positions) * 0.9:  # Allow some duplicates
            messages.append(f"Too many duplicate positions: {len(positions) - unique_positions}")

        # Check position range
        pos_range = max(positions) - min(positions)
        if pos_range <= 0:
            messages.append("Invalid position range (no variation)")
            return False, messages

        # Check for reasonable values
        if max(abs(e) for e in errors) > 100:  # Assuming % units
            messages.append("Error values seem unreasonably large (>100)")

        return len(messages) == 0, messages

    def apply_butterworth_filter(self, data: List[float],
                                 order: int = 5) -> np.ndarray:
        """
        Apply Butterworth low-pass filter to data.

        Args:
            data: Input data to filter
            order: Filter order (default: 5)

        Returns:
            Filtered data array
        """
        try:
            # Convert to numpy array
            data_array = np.array(data)
            
            # Check if data is too short for filtering
            min_length = 3 * order  # Rule of thumb for filtfilt
            if len(data_array) < min_length:
                self.logger.warning(
                    f"Data length ({len(data_array)}) too short for order {order} Butterworth filter. "
                    f"Returning original data."
                )
                return data_array
            
            # Calculate normalized cutoff frequency
            nyquist_freq = self.analysis_config.filter_sampling_frequency / 2
            normalized_cutoff = self.analysis_config.filter_cutoff_frequency / nyquist_freq

            # Ensure cutoff is valid
            if normalized_cutoff >= 1.0:
                self.logger.warning(
                    f"Cutoff frequency {self.analysis_config.filter_cutoff_frequency} Hz "
                    f"is too high for sampling frequency {self.analysis_config.filter_sampling_frequency} Hz. "
                    "Using 0.4 * Nyquist frequency."
                )
                normalized_cutoff = 0.4  # More conservative for stability
            elif normalized_cutoff > 0.95:
                # Silently adjust if close to Nyquist
                normalized_cutoff = 0.4

            # Design filter
            b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)

            # Apply filter (using filtfilt for zero phase shift)
            # Use padding for short sequences
            if len(data_array) < 50:
                # For very short sequences, use simpler filtering
                filtered_data = signal.lfilter(b, a, data_array)
            else:
                # For longer sequences, use filtfilt with padding
                filtered_data = signal.filtfilt(b, a, data_array, padtype='odd', padlen=min(len(data_array)//4, 50))

            return filtered_data

        except Exception as e:
            self.logger.error(f"Error applying Butterworth filter: {e}")
            # Return original data if filtering fails
            return np.array(data)

    def remove_endpoints(self, data: List[float],
                         count: Optional[int] = None) -> List[float]:
        """
        Remove specified number of points from each end of data.

        Args:
            data: Input data list
            count: Number of points to remove (default from config)

        Returns:
            Trimmed data list
        """
        if count is None:
            count = END_POINT_FILTER_COUNT

        if len(data) <= 2 * count:
            self.logger.warning(
                f"Data length ({len(data)}) too short for endpoint removal "
                f"of {count} points from each end"
            )
            return data

        return data[count:-count] if count > 0 else data

    def calculate_travel_length(self, positions: List[float]) -> float:
        """
        Calculate total travel length from positions.

        Args:
            positions: Position measurements

        Returns:
            Total travel length
        """
        if not positions:
            return 0.0

        return max(positions) - min(positions)

    def interpolate_limits(self, positions: List[float],
                           upper_limits: List[Optional[float]],
                           lower_limits: List[Optional[float]]) -> Tuple[List[float], List[float]]:
        """
        Interpolate missing limit values.

        Args:
            positions: Position values
            upper_limits: Upper limit values (may contain None)
            lower_limits: Lower limit values (may contain None)

        Returns:
            Tuple of (interpolated_upper, interpolated_lower)
        """
        # Convert to numpy arrays for easier manipulation
        positions_arr = np.array(positions)

        # Helper function to interpolate a limit series
        def interpolate_series(limits: List[Optional[float]]) -> List[float]:
            # Find valid (non-None) indices
            valid_indices = [i for i, val in enumerate(limits) if val is not None]

            if not valid_indices:
                # No valid data, return zeros
                return [0.0] * len(limits)

            if len(valid_indices) == len(limits):
                # All valid, return as-is
                return [float(val) for val in limits]

            # Get valid positions and values
            valid_positions = positions_arr[valid_indices]
            valid_values = [limits[i] for i in valid_indices]

            # Interpolate missing values
            interpolated = np.interp(positions_arr, valid_positions, valid_values)

            return interpolated.tolist()

        interpolated_upper = interpolate_series(upper_limits)
        interpolated_lower = interpolate_series(lower_limits)

        return interpolated_upper, interpolated_lower

    def determine_risk_category(self, failure_probability: float) -> RiskCategory:
        """
        Determine risk category based on failure probability.

        Args:
            failure_probability: Calculated failure probability (0-1)

        Returns:
            Risk category enum
        """
        if failure_probability >= self.analysis_config.high_risk_threshold:
            return RiskCategory.HIGH
        elif failure_probability >= self.analysis_config.low_risk_threshold:
            return RiskCategory.MEDIUM
        else:
            return RiskCategory.LOW

    def get_model_specific_config(self, model: str) -> Dict[str, Any]:
        """
        Get model-specific configuration overrides.

        Args:
            model: Model identifier

        Returns:
            Configuration dictionary for the model
        """
        return self.config.get_model_config(model)

    def calculate_linearity_spec(self, upper_limits: List[float],
                                 lower_limits: List[float]) -> float:
        """
        Calculate linearity specification from limits.

        Args:
            upper_limits: Upper limit values
            lower_limits: Lower limit values

        Returns:
            Linearity specification value
        """
        # Remove None values and calculate
        valid_upper = [u for u in upper_limits if u is not None]
        valid_lower = [l for l in lower_limits if l is not None]

        if valid_upper and valid_lower:
            # Average half-width of the tolerance band
            avg_upper = np.mean(valid_upper)
            avg_lower = np.mean(valid_lower)
            return (avg_upper - avg_lower) / 2

        # Fallback: if no valid limits, return a default
        return 0.01  # 1% default

    def log_analysis_summary(self, analysis_type: str,
                             result: Any,
                             processing_time: float):
        """
        Log analysis summary for debugging.

        Args:
            analysis_type: Type of analysis performed
            result: Analysis result object
            processing_time: Time taken for analysis
        """
        self.logger.info(
            f"{analysis_type} analysis completed in {processing_time:.3f}s"
        )

        # Log key metrics if available
        if hasattr(result, 'model_dump'):
            key_metrics = result.model_dump()
            self.logger.debug(f"Results: {key_metrics}")