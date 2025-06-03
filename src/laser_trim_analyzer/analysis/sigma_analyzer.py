"""
Sigma gradient analyzer for laser trim data.

Calculates sigma gradient metrics to assess trim quality and stability.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from laser_trim_analyzer.core.models import SigmaAnalysis
from laser_trim_analyzer.core.constants import (
    MATLAB_GRADIENT_STEP, DEFAULT_SIGMA_SCALING_FACTOR,
    SPECIAL_MODELS, END_POINT_FILTER_COUNT
)
from laser_trim_analyzer.analysis.base import BaseAnalyzer


class SigmaAnalyzer(BaseAnalyzer):
    """
    Analyzer for calculating sigma gradient metrics.

    The sigma gradient measures the variability of error gradients
    across the potentiometer's travel, indicating trim quality.
    """

    def analyze(self, data: Dict[str, Any]) -> SigmaAnalysis:
        """
        Analyze sigma gradient from trim data.

        Args:
            data: Dictionary containing:
                - positions: List[float] - Position measurements
                - errors: List[float] - Error measurements
                - upper_limits: List[Optional[float]] - Upper spec limits
                - lower_limits: List[Optional[float]] - Lower spec limits
                - model: str - Model identifier
                - unit_length: Optional[float] - Unit angle/length
                - travel_length: Optional[float] - Total travel length

        Returns:
            SigmaAnalysis model with calculated metrics
        """
        start_time = time.time()

        # Extract data
        positions = data['positions']
        errors = data['errors']
        upper_limits = data.get('upper_limits', [])
        lower_limits = data.get('lower_limits', [])
        model = data.get('model', 'Unknown')
        unit_length = data.get('unit_length')
        travel_length = data.get('travel_length')

        # Validate data
        is_valid, messages = self.validate_data(positions, errors)
        if not is_valid:
            self.logger.error(f"Data validation failed: {messages}")
            # Return failed analysis
            return SigmaAnalysis(
                sigma_gradient=999.999,
                sigma_threshold=0.001,
                sigma_pass=False,
                gradient_margin=-999.0,
                scaling_factor=self.analysis_config.sigma_scaling_factor
            )

        # Apply filtering to errors
        filtered_errors = self.apply_butterworth_filter(errors)

        # Remove endpoints if configured
        if len(positions) > 2 * END_POINT_FILTER_COUNT:
            positions = self.remove_endpoints(positions)
            filtered_errors = self.remove_endpoints(filtered_errors)
            if upper_limits:
                upper_limits = self.remove_endpoints(upper_limits)
            if lower_limits:
                lower_limits = self.remove_endpoints(lower_limits)

        # Calculate sigma gradient
        sigma_gradient = self._calculate_sigma_gradient(
            positions, filtered_errors
        )

        # Calculate travel length if not provided
        if travel_length is None:
            travel_length = self.calculate_travel_length(positions)

        # Calculate linearity spec from limits
        linearity_spec = self._calculate_linearity_spec_from_limits(
            upper_limits, lower_limits, errors
        )

        # Calculate threshold
        sigma_threshold = self._calculate_threshold(
            model, linearity_spec, travel_length, unit_length
        )

        # Determine pass/fail
        sigma_pass = sigma_gradient <= sigma_threshold
        gradient_margin = sigma_threshold - sigma_gradient

        # Create result
        result = SigmaAnalysis(
            sigma_gradient=sigma_gradient,
            sigma_threshold=sigma_threshold,
            sigma_pass=sigma_pass,
            gradient_margin=gradient_margin,
            scaling_factor=self._get_scaling_factor(model)
        )

        # Log summary
        processing_time = time.time() - start_time
        self.log_analysis_summary("Sigma", result, processing_time)

        return result

    def _calculate_sigma_gradient(self, positions: List[float],
                                  errors: List[float]) -> float:
        """
        Calculate sigma gradient using MATLAB-compatible algorithm.

        Args:
            positions: Position values
            errors: Filtered error values

        Returns:
            Calculated sigma gradient
        """
        gradients = []
        step_size = self.analysis_config.matlab_gradient_step

        # Calculate gradients with specified step size
        for i in range(len(positions) - step_size):
            dx = positions[i + step_size] - positions[i]
            dy = errors[i + step_size] - errors[i]

            # FIXED: Add proper bounds checking for division by zero
            if abs(dx) > 1e-10:  # Ensure dx is not effectively zero
                gradient = dy / dx
                # Additional bounds checking for numerical stability
                if not (np.isnan(gradient) or np.isinf(gradient)):
                    gradients.append(gradient)
                else:
                    self.logger.warning(f"Invalid gradient calculated at index {i}: {gradient}")
            else:
                self.logger.debug(f"Skipping near-zero position difference at index {i}: dx={dx}")

        if not gradients:
            self.logger.warning("No valid gradients calculated - insufficient position variation")
            return 0.0

        # Calculate standard deviation with N-1 degrees of freedom
        # Additional safety check for edge cases
        if len(gradients) == 1:
            self.logger.warning("Only one valid gradient point - using absolute value")
            return abs(gradients[0])
        
        try:
            sigma_gradient = np.std(gradients, ddof=1)
            
            # Bounds checking for the final result
            if np.isnan(sigma_gradient) or np.isinf(sigma_gradient):
                self.logger.error(f"Invalid sigma gradient calculated: {sigma_gradient}")
                return 0.0
                
            # Sanity check - sigma gradient should be positive and reasonable
            if sigma_gradient < 0:
                self.logger.warning(f"Negative sigma gradient: {sigma_gradient}, taking absolute value")
                sigma_gradient = abs(sigma_gradient)
                
            if sigma_gradient > 1000:  # Unreasonably large value
                self.logger.warning(f"Unusually large sigma gradient: {sigma_gradient}, may indicate data issues")
            
            return float(sigma_gradient)
            
        except Exception as e:
            self.logger.error(f"Error calculating sigma gradient: {e}")
            return 0.0

    def _calculate_linearity_spec_from_limits(self,
                                              upper_limits: List[Optional[float]],
                                              lower_limits: List[Optional[float]],
                                              errors: List[float]) -> float:
        """
        Calculate linearity specification from limit data.

        Args:
            upper_limits: Upper specification limits
            lower_limits: Lower specification limits
            errors: Error values (fallback)

        Returns:
            Linearity specification value
        """
        # Filter out None values
        valid_upper = [u for u in upper_limits if u is not None]
        valid_lower = [l for l in lower_limits if l is not None]

        if valid_upper and valid_lower:
            # Calculate average half-width of tolerance band
            avg_upper = np.mean(valid_upper)
            avg_lower = np.mean(valid_lower)
            linearity_spec = (avg_upper - avg_lower) / 2
        else:
            # Fallback: use maximum absolute error
            linearity_spec = max(abs(e) for e in errors) if errors else 0.01
            self.logger.warning(
                f"No valid limits found, using max error for linearity spec: {linearity_spec:.4f}"
            )

        return float(linearity_spec)

    def _calculate_threshold(self, model: str, linearity_spec: float,
                             travel_length: float, unit_length: Optional[float]) -> float:
        """
        Calculate sigma threshold based on model and specifications.

        Args:
            model: Model identifier
            linearity_spec: Linearity specification
            travel_length: Total travel length
            unit_length: Unit angle/length (optional)

        Returns:
            Calculated threshold value
        """
        # Check for model-specific fixed thresholds
        if model in SPECIAL_MODELS:
            if 'fixed_sigma_threshold' in SPECIAL_MODELS[model]:
                return SPECIAL_MODELS[model]['fixed_sigma_threshold']

        # Model-specific calculations with more realistic thresholds
        if model.startswith('8555'):
            # Empirical threshold for 8555 models - more stringent
            base_threshold = 0.0015  # Reduced from 0.0025
            spec_factor = linearity_spec / 0.01 if linearity_spec > 0 else 1.0
            threshold = base_threshold * spec_factor

        elif model.startswith('8340-1'):
            # Hard-coded for 8340-1 (redundant with SPECIAL_MODELS but kept for clarity)
            threshold = 0.4

        else:
            # Traditional calculation with adjusted scaling
            scaling_factor = self._get_scaling_factor(model)

            # Use unit length if available, otherwise travel length
            effective_length = unit_length if unit_length and unit_length > 0 else travel_length

            if effective_length and effective_length > 0:
                # More stringent calculation
                threshold = (linearity_spec / effective_length) * (scaling_factor * 0.5)  # Reduced by 50%
            else:
                # Fallback calculation - also more stringent
                threshold = scaling_factor * 0.01  # Reduced from 0.02
                self.logger.warning(
                    f"No valid length for threshold calculation, using default: {threshold:.4f}"
                )

        # Apply minimum threshold to avoid unreasonably small values
        min_threshold = 0.0001
        if threshold < min_threshold:
            self.logger.warning(
                f"Calculated threshold {threshold:.6f} below minimum, using {min_threshold:.6f}"
            )
            threshold = min_threshold
        
        # Apply maximum threshold to catch unreasonably high values
        max_threshold = 0.05  # Maximum reasonable threshold
        if threshold > max_threshold:
            self.logger.warning(
                f"Calculated threshold {threshold:.6f} above maximum, using {max_threshold:.6f}"
            )
            threshold = max_threshold

        return float(threshold)

    def _get_scaling_factor(self, model: str) -> float:
        """
        Get scaling factor for sigma threshold calculation.

        Args:
            model: Model identifier

        Returns:
            Scaling factor value
        """
        # Check model-specific configuration
        model_config = self.get_model_specific_config(model)
        if 'analysis' in model_config and 'sigma_scaling_factor' in model_config['analysis']:
            return model_config['analysis']['sigma_scaling_factor']

        # Use default from config
        return self.analysis_config.sigma_scaling_factor