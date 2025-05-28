"""
Sigma gradient calculator with validated algorithm.

This module implements the exact sigma gradient calculation from your
validated MATLAB code, preserving the precise algorithm that you've
already confirmed matches your legacy system.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from src.core.filter_utils import apply_filter
from src.core.constants import (
    MATLAB_GRADIENT_STEP,
    DEFAULT_SIGMA_SCALING_FACTOR,
    MODEL_8340_1_THRESHOLD,
    MIN_DATA_POINTS
)

logger = logging.getLogger(__name__)


@dataclass
class SigmaResult:
    """Container for sigma calculation results."""
    sigma_gradient: float
    sigma_threshold: float
    passed: bool
    filtered_error: List[float]
    gradients: List[float]
    gradient_positions: List[float]
    linearity_spec: float
    unit_length: Optional[float]
    travel_length: float


class SigmaCalculator:
    """
    Calculates sigma gradients using the validated algorithm.

    This implementation exactly matches the MATLAB calc_gradient.m function
    that has been validated against your legacy system.
    """

    def __init__(self, scaling_factor: float = DEFAULT_SIGMA_SCALING_FACTOR):
        """
        Initialize calculator with scaling factor.

        Args:
            scaling_factor: Scaling factor for threshold calculation (default: 24.0)
        """
        self.scaling_factor = scaling_factor
        self.gradient_step = MATLAB_GRADIENT_STEP

    def calculate(
            self,
            position: List[float],
            error: List[float],
            upper_limit: Optional[List[float]] = None,
            lower_limit: Optional[List[float]] = None,
            unit_length: Optional[float] = None,
            model: Optional[str] = None
    ) -> SigmaResult:
        """
        Calculate sigma gradient and determine pass/fail.

        This is the main entry point that orchestrates the complete
        sigma analysis matching your validated process.

        Args:
            position: Position values
            error: Error values
            upper_limit: Upper tolerance limits (optional)
            lower_limit: Lower tolerance limits (optional)
            unit_length: Unit length/angle from cell B26 (optional)
            model: Model number for special threshold cases

        Returns:
            SigmaResult with all calculation details
        """
        # Validate inputs
        if not self._validate_inputs(position, error):
            raise ValueError("Invalid input data for sigma calculation")

        # Calculate travel length
        travel_length = max(position) - min(position)

        # Determine linearity specification
        linearity_spec = self._calculate_linearity_spec(
            error, upper_limit, lower_limit
        )

        # Apply filter to error data (matching MATLAB exactly)
        filtered_error = apply_filter(error)

        # Calculate gradients
        gradients, gradient_positions = self._calculate_gradients(
            position, filtered_error
        )

        # Calculate sigma (standard deviation of gradients)
        sigma_gradient = self._calculate_sigma(gradients)

        # Calculate threshold
        sigma_threshold = self._calculate_threshold(
            linearity_spec, travel_length, unit_length, model
        )

        # Determine pass/fail
        passed = sigma_gradient <= sigma_threshold

        # Log results
        logger.info(
            f"Sigma calculation complete: "
            f"gradient={sigma_gradient:.6f}, "
            f"threshold={sigma_threshold:.6f}, "
            f"passed={passed}"
        )

        return SigmaResult(
            sigma_gradient=sigma_gradient,
            sigma_threshold=sigma_threshold,
            passed=passed,
            filtered_error=filtered_error,
            gradients=gradients,
            gradient_positions=gradient_positions,
            linearity_spec=linearity_spec,
            unit_length=unit_length,
            travel_length=travel_length
        )

    def _validate_inputs(self, position: List[float], error: List[float]) -> bool:
        """Validate input data."""
        if not position or not error:
            logger.error("Empty position or error data")
            return False

        if len(position) != len(error):
            logger.error(f"Position ({len(position)}) and error ({len(error)}) length mismatch")
            return False

        if len(position) < MIN_DATA_POINTS:
            logger.error(f"Insufficient data points: {len(position)} < {MIN_DATA_POINTS}")
            return False

        return True

    def _calculate_linearity_spec(
            self,
            error: List[float],
            upper_limit: Optional[List[float]],
            lower_limit: Optional[List[float]]
    ) -> float:
        """
        Calculate linearity specification from limits or error range.

        Matches the legacy logic for determining linearity spec.
        """
        # Try to calculate from limits first
        if upper_limit and lower_limit:
            valid_indices = [
                i for i in range(len(upper_limit))
                if i < len(lower_limit) and
                   upper_limit[i] is not None and
                   lower_limit[i] is not None
            ]

            if valid_indices:
                avg_upper = np.mean([upper_limit[i] for i in valid_indices])
                avg_lower = np.mean([lower_limit[i] for i in valid_indices])
                linearity_spec = (avg_upper - avg_lower) / 2
                logger.debug(f"Linearity spec from limits: {linearity_spec:.6f}")
                return linearity_spec

        # Fallback to max absolute error
        linearity_spec = max(abs(e) for e in error)
        logger.debug(f"Linearity spec from max error: {linearity_spec:.6f}")
        return linearity_spec

    def _calculate_gradients(
            self,
            position: List[float],
            filtered_error: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate gradients using the validated MATLAB algorithm.

        This exactly matches the MATLAB implementation:
        for z=1:1:length(position)-z_step
            grad(k) = (error_volts_filt(ind2)-error_volts_filt(ind1)) /
                      (position(ind2)-position(ind1))
        end
        """
        gradients = []
        gradient_positions = []

        # Calculate gradients with step size (default: 3)
        for i in range(len(position) - self.gradient_step):
            # Indices matching MATLAB
            ind1 = i
            ind2 = i + self.gradient_step

            # Position difference
            dx = position[ind2] - position[ind1]

            # Error difference (filtered)
            dy = filtered_error[ind2] - filtered_error[ind1]

            # Calculate gradient
            if dx != 0:  # Avoid division by zero
                gradient = dy / dx
                gradients.append(gradient)
                gradient_positions.append(position[ind2])

        logger.debug(f"Calculated {len(gradients)} gradients with step size {self.gradient_step}")

        return gradients, gradient_positions

    def _calculate_sigma(self, gradients: List[float]) -> float:
        """
        Calculate sigma (standard deviation) of gradients.

        Uses ddof=1 to match MATLAB's std() function behavior.
        """
        if not gradients:
            logger.warning("No gradients to calculate sigma")
            return 0.0

        # Calculate standard deviation with Bessel's correction (ddof=1)
        # This matches MATLAB's std() function
        sigma = np.std(gradients, ddof=1)

        logger.debug(f"Sigma gradient: {sigma:.6f} from {len(gradients)} gradients")

        return sigma

    def _calculate_threshold(
            self,
            linearity_spec: float,
            travel_length: float,
            unit_length: Optional[float],
            model: Optional[str]
    ) -> float:
        """
        Calculate sigma threshold using validated formula.

        Special cases:
        - Model 8340-1: Fixed threshold of 0.4
        - Others: (linearity_spec / length) * scaling_factor

        Args:
            linearity_spec: Linearity specification
            travel_length: Travel length from data
            unit_length: Unit length from cell B26 (preferred over travel_length)
            model: Model number for special cases

        Returns:
            Calculated threshold
        """
        # Special case for 8340-1
        if model and model.upper().startswith("8340-1"):
            logger.info(f"Using fixed threshold {MODEL_8340_1_THRESHOLD} for model {model}")
            return MODEL_8340_1_THRESHOLD

        # Use unit_length if available, otherwise travel_length
        length_to_use = unit_length if unit_length is not None else travel_length

        # Validate length
        if length_to_use <= 0:
            logger.warning(f"Invalid length {length_to_use}, using travel_length {travel_length}")
            length_to_use = travel_length

        # Calculate threshold using validated formula
        threshold = (linearity_spec / length_to_use) * self.scaling_factor

        logger.debug(
            f"Threshold calculation: ({linearity_spec:.6f} / {length_to_use:.4f}) * "
            f"{self.scaling_factor} = {threshold:.6f}"
        )

        return threshold

    def calculate_batch(
            self,
            data_list: List[Dict[str, Any]]
    ) -> List[SigmaResult]:
        """
        Calculate sigma for multiple datasets.

        Useful for processing multiple tracks or files.

        Args:
            data_list: List of data dictionaries, each containing
                      position, error, limits, unit_length, model

        Returns:
            List of SigmaResult objects
        """
        results = []

        for i, data in enumerate(data_list):
            try:
                result = self.calculate(
                    position=data['position'],
                    error=data['error'],
                    upper_limit=data.get('upper_limit'),
                    lower_limit=data.get('lower_limit'),
                    unit_length=data.get('unit_length'),
                    model=data.get('model')
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing batch item {i}: {e}")
                # Create failed result
                results.append(SigmaResult(
                    sigma_gradient=0.0,
                    sigma_threshold=0.0,
                    passed=False,
                    filtered_error=[],
                    gradients=[],
                    gradient_positions=[],
                    linearity_spec=0.0,
                    unit_length=None,
                    travel_length=0.0
                ))

        return results


# Convenience function for simple one-off calculations
def calculate_sigma(
        position: List[float],
        error: List[float],
        **kwargs
) -> SigmaResult:
    """
    Convenience function for one-off sigma calculations.

    Args:
        position: Position values
        error: Error values
        **kwargs: Additional arguments passed to SigmaCalculator.calculate()

    Returns:
        SigmaResult object
    """
    calculator = SigmaCalculator()
    return calculator.calculate(position, error, **kwargs)