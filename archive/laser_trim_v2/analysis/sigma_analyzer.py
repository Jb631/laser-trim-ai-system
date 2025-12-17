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

        # Validate linearity_spec before calculating threshold
        if linearity_spec is None or np.isnan(linearity_spec) or linearity_spec <= 0:
            self.logger.warning(f"Invalid linearity_spec {linearity_spec}, using default 0.01")
            linearity_spec = 0.01
        
        # Log the inputs for threshold calculation
        self.logger.debug(f"Threshold calculation inputs - model: {model}, linearity_spec: {linearity_spec}, "
                         f"travel_length: {travel_length}, unit_length: {unit_length}")
        
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

        # Log initial data characteristics
        self.logger.info(f"Calculating sigma gradient: {len(positions)} positions, "
                        f"range [{min(positions):.3f}, {max(positions):.3f}], step_size={step_size}")
        
        # Calculate gradients with specified step size
        for i in range(len(positions) - step_size):
            dx = positions[i + step_size] - positions[i]
            dy = errors[i + step_size] - errors[i]

            # FIXED: Add proper bounds checking for division by zero
            # Use a more reasonable threshold based on typical position resolution
            if abs(dx) > 1e-6:  # Changed from 1e-10 to 1e-6 for better numerical stability
                gradient = dy / dx
                # Additional bounds checking for numerical stability
                if not (np.isnan(gradient) or np.isinf(gradient)):
                    gradients.append(gradient)
                else:
                    self.logger.warning(f"Invalid gradient calculated at index {i}: {gradient}")
            else:
                self.logger.warning(f"Skipping near-zero position difference at index {i}: dx={dx:.12f}, "
                                  f"positions[{i}]={positions[i]:.6f}, positions[{i+step_size}]={positions[i+step_size]:.6f}")

        if not gradients:
            self.logger.error(f"No valid gradients calculated - insufficient position variation. "
                             f"Positions range: {min(positions):.6f} to {max(positions):.6f}, "
                             f"Errors range: {min(errors):.6f} to {max(errors):.6f}, "
                             f"Step size: {step_size}, Total points: {len(positions)}")
            # Return a small non-zero value instead of 0 to indicate processing occurred
            return 0.000001

        # Calculate standard deviation with N-1 degrees of freedom
        # Additional safety check for edge cases
        if len(gradients) == 1:
            self.logger.warning("Only one valid gradient point - using absolute value")
            return abs(gradients[0])
        
        try:
            # Add detailed validation logging
            gradient_stats = {
                'count': len(gradients),
                'mean': np.mean(gradients),
                'median': np.median(gradients),
                'min': np.min(gradients),
                'max': np.max(gradients),
                'range': np.max(gradients) - np.min(gradients)
            }
            
            sigma_gradient = np.std(gradients, ddof=1)
            
            # Log comprehensive calculation details for validation
            self.logger.info(f"Sigma gradient calculation validation:")
            self.logger.info(f"  - Input: {len(positions)} positions, {len(errors)} errors")
            self.logger.info(f"  - Valid gradients: {gradient_stats['count']} (out of {len(positions) - step_size} possible)")
            self.logger.info(f"  - Gradient statistics:")
            self.logger.info(f"    * Mean: {gradient_stats['mean']:.6f}")
            self.logger.info(f"    * Median: {gradient_stats['median']:.6f}")
            self.logger.info(f"    * Min: {gradient_stats['min']:.6f}")
            self.logger.info(f"    * Max: {gradient_stats['max']:.6f}")
            self.logger.info(f"    * Range: {gradient_stats['range']:.6f}")
            self.logger.info(f"  - Calculated sigma (std dev): {sigma_gradient:.6f}")
            
            # Validation checks
            if gradient_stats['count'] < len(positions) * 0.5:
                self.logger.warning(f"Less than 50% of gradients were valid ({gradient_stats['count']}/{len(positions) - step_size})")
            
            # Check if sigma is reasonable compared to gradient range
            if gradient_stats['range'] > 0:
                sigma_to_range_ratio = sigma_gradient / gradient_stats['range']
                self.logger.info(f"  - Sigma to range ratio: {sigma_to_range_ratio:.3f}")
                if sigma_to_range_ratio > 0.5:
                    self.logger.warning("High sigma relative to range - may indicate outliers")
            
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
        # Debug logging to understand the data
        valid_upper = [u for u in upper_limits if u is not None and not np.isnan(u)]
        valid_lower = [l for l in lower_limits if l is not None and not np.isnan(l)]
        
        self.logger.debug(f"Sigma analyzer - Valid upper limits: {len(valid_upper)}, Valid lower limits: {len(valid_lower)}")
        
        if valid_upper and valid_lower:
            # Log the actual values to debug
            self.logger.debug(f"First few upper limits: {valid_upper[:5]}")
            self.logger.debug(f"First few lower limits: {valid_lower[:5]}")
            self.logger.debug(f"First few errors: {errors[:5] if errors else []}")
            
            # Check the scale of the data
            max_upper = max(valid_upper)
            min_lower = min(valid_lower)
            error_range = max(abs(e) for e in errors) if errors else 0
            
            self.logger.debug(f"Upper limit max: {max_upper:.6f}, Lower limit min: {min_lower:.6f}")
            self.logger.debug(f"Error range: {error_range:.6f}")
        
        # Use the base class method which has NaN handling
        spec = self.calculate_linearity_spec(upper_limits, lower_limits)
        self.logger.debug(f"Calculated linearity spec: {spec}")
        return spec

    def _calculate_threshold(self, model: str, linearity_spec: float,
                             travel_length: float, unit_length: Optional[float]) -> float:
        """
        Calculate sigma threshold - ML-based first, formula-based fallback.

        Priority order:
        1. ML-learned threshold from ThresholdOptimizer (if available)
        2. Formula-based calculation as fallback

        NO HARDCODED THRESHOLDS - even for 8340-1

        Args:
            model: Model identifier
            linearity_spec: Linearity specification
            travel_length: Total travel length
            unit_length: Unit angle/length (optional)

        Returns:
            Calculated threshold value
        """
        # PRIORITY 1: Try to get ML-learned threshold
        try:
            from laser_trim_analyzer.database.manager import DatabaseManager
            db_manager = DatabaseManager(self.config)
            ml_threshold = db_manager.get_latest_ml_threshold(model)

            if ml_threshold is not None:
                self.logger.info(f"Using ML-learned threshold for {model}: {ml_threshold:.6f}")
                return ml_threshold
            else:
                self.logger.info(f"No ML-learned threshold found for {model}, using formula-based calculation")
        except Exception as e:
            self.logger.warning(f"Could not retrieve ML threshold: {e}, using formula-based calculation")

        # PRIORITY 2: Formula-based calculation (fallback only)
        # NOTE: These formulas should be temporary until ML models are trained

        # Model-specific calculations
        if model.startswith('8555'):
            # Empirical formula for 8555 models
            base_threshold = 0.0015
            spec_factor = linearity_spec / 0.01 if linearity_spec > 0 else 1.0
            threshold = base_threshold * spec_factor
            self.logger.debug(f"Using 8555 formula-based threshold: {threshold:.6f}")

        elif model.startswith('8340-1'):
            # Formula-based for 8340-1 (NO MORE HARDCODED 0.4!)
            # Using conservative calculation until ML threshold available
            threshold = linearity_spec * DEFAULT_SIGMA_SCALING_FACTOR * 0.5
            self.logger.debug(f"Using 8340-1 formula-based threshold: {threshold:.6f}")

        else:
            # Traditional calculation with adjusted scaling
            scaling_factor = self._get_scaling_factor(model)

            # Use unit length if available, otherwise travel length
            effective_length = unit_length if unit_length and unit_length > 0 else travel_length

            self.logger.debug(f"Effective length for threshold calculation: {effective_length}, "
                            f"scaling_factor: {scaling_factor}")

            if effective_length and effective_length > 0:
                # Standard calculation
                if linearity_spec < 0.001 and effective_length > 1:
                    self.logger.warning(f"Very small linearity spec ({linearity_spec:.6f}) with large length ({effective_length:.2f})")
                    threshold = linearity_spec * scaling_factor * 0.1
                    self.logger.debug(f"Using edge case calculation: {threshold:.6f}")
                else:
                    threshold = (linearity_spec / effective_length) * (scaling_factor * 0.5)
                    self.logger.debug(f"Using standard formula-based threshold: {threshold:.6f}")
                    self.logger.debug(f"Calculated threshold using formula: ({linearity_spec} / {effective_length}) * ({scaling_factor} * 0.5) = {threshold}")
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

        # Final validation to ensure threshold is valid
        if not np.isfinite(threshold) or threshold <= 0:
            self.logger.warning(f"Invalid threshold calculated: {threshold}, using default 0.1")
            threshold = 0.1
            
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