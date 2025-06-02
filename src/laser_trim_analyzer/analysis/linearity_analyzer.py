"""
Linearity analyzer for laser trim data.

Analyzes how well the potentiometer output follows a linear relationship
with position, including optimal offset calculation.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import optimize

from laser_trim_analyzer.core.models import LinearityAnalysis
from laser_trim_analyzer.analysis.base import BaseAnalyzer


class LinearityAnalyzer(BaseAnalyzer):
    """
    Analyzer for linearity metrics and optimal offset calculation.

    Determines how closely the potentiometer follows ideal linear behavior
    and calculates the optimal vertical offset to minimize errors.
    """

    def analyze(self, data: Dict[str, Any]) -> LinearityAnalysis:
        """
        Analyze linearity metrics from trim data.

        Args:
            data: Dictionary containing:
                - positions: List[float] - Position measurements
                - errors: List[float] - Error measurements
                - upper_limits: List[Optional[float]] - Upper spec limits
                - lower_limits: List[Optional[float]] - Lower spec limits
                - linearity_spec: Optional[float] - Linearity specification

        Returns:
            LinearityAnalysis model with calculated metrics
        """
        start_time = time.time()

        # Extract data
        positions = data['positions']
        errors = data['errors']
        upper_limits = data.get('upper_limits', [])
        lower_limits = data.get('lower_limits', [])
        linearity_spec = data.get('linearity_spec')

        self.logger.debug(f"Linearity analysis input: {len(positions)} positions, {len(errors)} errors")
        self.logger.debug(f"Limits received: {len(upper_limits)} upper, {len(lower_limits)} lower")
        self.logger.debug(f"Initial linearity_spec: {linearity_spec}")

        # Validate data
        is_valid, messages = self.validate_data(positions, errors)
        if not is_valid:
            self.logger.error(f"Data validation failed: {messages}")
            # Return failed analysis
            return LinearityAnalysis(
                linearity_spec=linearity_spec or 0.01,
                optimal_offset=0.0,
                final_linearity_error_raw=999.999,
                final_linearity_error_shifted=999.999,
                linearity_pass=False,
                linearity_fail_points=len(positions)
            )

        # Interpolate missing limits
        if upper_limits and lower_limits:
            self.logger.debug("Interpolating limits...")
            upper_limits, lower_limits = self.interpolate_limits(
                positions, upper_limits, lower_limits
            )
            self.logger.debug(f"After interpolation: {len(upper_limits)} upper, {len(lower_limits)} lower")

        # Calculate linearity spec if not provided
        if linearity_spec is None:
            self.logger.debug("Calculating linearity spec from limits...")
            linearity_spec = self.calculate_linearity_spec(upper_limits, lower_limits)
            self.logger.debug(f"Calculated linearity_spec: {linearity_spec}")
        else:
            self.logger.debug(f"Using provided linearity_spec: {linearity_spec}")

        # Calculate optimal offset
        optimal_offset = self._calculate_optimal_offset(
            errors, upper_limits, lower_limits
        )
        self.logger.debug(f"Calculated optimal offset: {optimal_offset}")

        # Apply offset
        shifted_errors = [e + optimal_offset for e in errors]

        # Calculate raw and shifted errors
        final_error_raw = max(abs(e) for e in errors)
        final_error_shifted = max(abs(e) for e in shifted_errors)

        # Count fail points and check linearity
        fail_points = self._count_fail_points(
            shifted_errors, upper_limits, lower_limits
        )
        linearity_pass = fail_points == 0

        self.logger.debug(f"Linearity analysis results: raw_error={final_error_raw:.4f}, shifted_error={final_error_shifted:.4f}, fail_points={fail_points}, pass={linearity_pass}")

        # Calculate deviation analysis
        max_deviation, max_deviation_position = self._analyze_deviation(
            positions, errors
        )

        # Create result
        result = LinearityAnalysis(
            linearity_spec=linearity_spec,
            optimal_offset=optimal_offset,
            final_linearity_error_raw=final_error_raw,
            final_linearity_error_shifted=final_error_shifted,
            linearity_pass=linearity_pass,
            linearity_fail_points=fail_points,
            max_deviation=max_deviation,
            max_deviation_position=max_deviation_position
        )

        # Log summary
        processing_time = time.time() - start_time
        self.log_analysis_summary("Linearity", result, processing_time)

        return result

    def _calculate_optimal_offset(self, errors: List[float],
                                  upper_limits: List[float],
                                  lower_limits: List[float]) -> float:
        """
        Calculate optimal vertical offset to center errors within limits.

        Uses multiple methods and selects the best one:
        1. Median of differences from band center
        2. Optimization to minimize violations
        3. Simple centering

        Args:
            errors: Error values
            upper_limits: Upper specification limits
            lower_limits: Lower specification limits

        Returns:
            Optimal offset value
        """
        # Method 1: Median difference from band center
        if upper_limits and lower_limits:
            valid_indices = []
            differences = []

            for i in range(len(errors)):
                if (i < len(upper_limits) and i < len(lower_limits) and
                        upper_limits[i] is not None and lower_limits[i] is not None):
                    valid_indices.append(i)
                    midpoint = (upper_limits[i] + lower_limits[i]) / 2
                    differences.append(midpoint - errors[i])

            if differences:
                median_offset = np.median(differences)

                # Method 2: Optimization to minimize violations
                def violation_count(offset):
                    """Count points outside limits with given offset."""
                    count = 0
                    for i in valid_indices:
                        shifted_error = errors[i] + offset
                        if (shifted_error > upper_limits[i] or
                                shifted_error < lower_limits[i]):
                            count += 1
                    return count

                # Try to find offset that minimizes violations
                try:
                    # Search around median offset
                    search_range = max(abs(d) for d in differences) if differences else 1.0
                    result = optimize.minimize_scalar(
                        violation_count,
                        bounds=(median_offset - search_range,
                                median_offset + search_range),
                        method='bounded'
                    )

                    optimal_offset = result.x

                    # Verify this is better than median
                    if violation_count(optimal_offset) >= violation_count(median_offset):
                        optimal_offset = median_offset

                except Exception as e:
                    self.logger.warning(f"Optimization failed, using median: {e}")
                    optimal_offset = median_offset

                return float(optimal_offset)

        # Fallback: Center errors around zero
        return float(-np.mean(errors))

    def _count_fail_points(self, errors: List[float],
                           upper_limits: List[float],
                           lower_limits: List[float]) -> int:
        """
        Count number of points that fail linearity check.

        Args:
            errors: Error values (already shifted)
            upper_limits: Upper specification limits
            lower_limits: Lower specification limits

        Returns:
            Number of failing points
        """
        fail_count = 0

        for i in range(len(errors)):
            # Check if limits exist for this point
            if (i < len(upper_limits) and i < len(lower_limits) and
                    upper_limits[i] is not None and lower_limits[i] is not None):

                # Check if error is within limits
                if errors[i] > upper_limits[i] or errors[i] < lower_limits[i]:
                    fail_count += 1

        return fail_count

    def _analyze_deviation(self, positions: List[float],
                           errors: List[float]) -> Tuple[float, float]:
        """
        Analyze deviation from ideal linear behavior.

        Fits a linear model and finds maximum deviation.

        Args:
            positions: Position values
            errors: Error values

        Returns:
            Tuple of (max_deviation, position_of_max_deviation)
        """
        if len(positions) < 2:
            return 0.0, 0.0

        try:
            # Fit linear model to errors vs position
            positions_arr = np.array(positions)
            errors_arr = np.array(errors)

            # Linear regression
            coeffs = np.polyfit(positions_arr, errors_arr, 1)
            slope, intercept = coeffs

            # Calculate ideal linear values
            ideal_errors = slope * positions_arr + intercept

            # Calculate deviations
            deviations = np.abs(errors_arr - ideal_errors)

            # Find maximum deviation
            max_idx = np.argmax(deviations)
            max_deviation = float(deviations[max_idx])
            max_position = float(positions_arr[max_idx])

            return max_deviation, max_position

        except Exception as e:
            self.logger.error(f"Error in deviation analysis: {e}")
            return 0.0, 0.0