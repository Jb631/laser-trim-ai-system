"""
Analysis module for Laser Trim Analyzer v3.

Combines sigma, linearity, and resistance analysis into one clean module.
Simplified from v2's multiple analyzer classes (~800 lines -> ~400 lines).

ML Integration:
- Per-model thresholds loaded from database (via Processor)
- Falls back to formula when ML threshold not available
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np
from scipy.signal import butter, filtfilt
from scipy import optimize

from laser_trim_analyzer.utils.constants import (
    DEFAULT_SIGMA_SCALING_FACTOR,
    MATLAB_GRADIENT_STEP,
    END_POINT_FILTER_COUNT,
    BUTTERWORTH_CUTOFF,
    BUTTERWORTH_ORDER,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
)
from laser_trim_analyzer.core.models import TrackData, AnalysisStatus, RiskCategory

logger = logging.getLogger(__name__)


class Analyzer:
    """
    Combined analyzer for laser trim data.

    Performs:
    - Sigma gradient analysis (measures trim quality)
    - Linearity analysis (optimal offset, fail points)
    - Resistance analysis (change percentage)
    - Risk assessment (failure probability)

    ML integration:
    - Per-model thresholds from database (passed as dictionary)
    - Automatic fallback to formula when ML not available
    """

    def __init__(
        self,
        scaling_factor: float = DEFAULT_SIGMA_SCALING_FACTOR,
        model_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.scaling_factor = scaling_factor
        # Per-model thresholds from ML training (model_name -> threshold)
        self._model_thresholds = model_thresholds or {}

    def analyze_track(
        self,
        track_data: Dict[str, Any],
        model: Optional[str] = None,
        linearity_type: Optional[str] = None,
        station_compensation: Optional[float] = None,
    ) -> TrackData:
        """
        Perform complete analysis on a track.

        Args:
            track_data: Dictionary from parser containing:
                - track_id, positions, errors, upper_limits, lower_limits
                - travel_length, linearity_spec, unit_length
                - untrimmed_resistance, trimmed_resistance
            model: Model number (for ML threshold lookup)

        Returns:
            TrackData object with all analysis results
        """
        track_id = track_data["track_id"]
        positions = track_data["positions"]
        errors = track_data["errors"]
        upper_limits = track_data.get("upper_limits", [])
        lower_limits = track_data.get("lower_limits", [])
        travel_length = track_data.get("travel_length", 0.0)
        linearity_spec = track_data.get("linearity_spec", 0.01)
        unit_length = track_data.get("unit_length")
        untrimmed_resistance = track_data.get("untrimmed_resistance")
        trimmed_resistance = track_data.get("trimmed_resistance")
        measured_electrical_angle = track_data.get("measured_electrical_angle")

        logger.info(f"Analyzing track {track_id}: {len(positions)} points")

        # Validate data
        if len(positions) < 10 or len(errors) < 10:
            logger.warning(f"Insufficient data points for track {track_id}")
            return self._create_failed_track(track_id, "Insufficient data points")

        # Sigma analysis (with optional ML threshold)
        sigma_gradient, sigma_threshold = self._calculate_sigma(
            positions, errors, linearity_spec, travel_length, unit_length,
            model=model
        )
        sigma_pass = sigma_gradient <= sigma_threshold

        # Linearity analysis (spec-aware)
        (optimal_offset, optimal_slope, linearity_error, linearity_pass,
         fail_points, raw_linearity_error, raw_fail_points) = self._calculate_linearity(
            positions, errors, upper_limits, lower_limits, linearity_spec,
            linearity_type=linearity_type
        )

        # Risk assessment
        failure_probability, risk_category = self._assess_risk(
            sigma_gradient, sigma_threshold, linearity_pass
        )

        # Anomaly detection (linear slope pattern = trim failure)
        is_anomaly, anomaly_reason = self._detect_anomaly(
            positions, errors, sigma_gradient, sigma_threshold, fail_points
        )

        # Determine overall status
        if sigma_pass and linearity_pass:
            status = AnalysisStatus.PASS
        elif sigma_pass or linearity_pass:
            status = AnalysisStatus.WARNING
        else:
            status = AnalysisStatus.FAIL

        # Calculate trim effectiveness metrics
        trim_metrics = self._calculate_trim_effectiveness(
            errors, track_data.get("untrimmed_errors", []),
            untrimmed_resistance, trimmed_resistance
        )

        # Calculate failure margin metrics
        shifted_errors = [e * optimal_slope + optimal_offset for e in errors]
        margin_metrics = self._calculate_failure_margins(
            shifted_errors, upper_limits, lower_limits
        )

        # Calculate max deviation position and uniformity
        max_deviation = linearity_error  # Already max(abs(e)) from shifted errors
        max_deviation_position = None
        deviation_uniformity = None
        if shifted_errors and positions:
            abs_errors = [abs(e) for e in shifted_errors]
            max_idx = abs_errors.index(max(abs_errors))
            if max_idx < len(positions):
                max_deviation_position = positions[max_idx]
            # Deviation uniformity: std/mean of absolute errors (0=uniform, higher=concentrated)
            if len(abs_errors) > 1:
                import statistics
                mean_abs = statistics.mean(abs_errors)
                if mean_abs > 0:
                    deviation_uniformity = statistics.stdev(abs_errors) / mean_abs

        return TrackData(
            track_id=track_id,
            status=status,
            travel_length=travel_length,
            linearity_spec=linearity_spec,
            # Sigma results
            sigma_gradient=sigma_gradient,
            sigma_threshold=sigma_threshold,
            sigma_pass=sigma_pass,
            # Linearity results
            optimal_offset=optimal_offset,
            optimal_slope=optimal_slope,
            linearity_error=linearity_error,
            linearity_pass=linearity_pass,
            linearity_fail_points=fail_points,
            # Spec-aware optimization
            station_compensation=station_compensation,
            linearity_type=linearity_type,
            raw_linearity_error=raw_linearity_error,
            optimized_linearity_error=linearity_error,
            raw_fail_points=raw_fail_points,
            # Unit properties (sanitize invalid values)
            unit_length=unit_length if unit_length and unit_length >= 0 else None,
            untrimmed_resistance=untrimmed_resistance,
            trimmed_resistance=trimmed_resistance,
            measured_electrical_angle=measured_electrical_angle,
            # Risk
            failure_probability=failure_probability,
            risk_category=risk_category,
            # Anomaly detection
            is_anomaly=is_anomaly,
            anomaly_reason=anomaly_reason,
            # Max deviation metrics
            max_deviation=max_deviation,
            max_deviation_position=max_deviation_position,
            deviation_uniformity=deviation_uniformity,
            # Failure margin metrics
            **margin_metrics,
            # Trim effectiveness
            **trim_metrics,
            # Raw data (for plotting)
            position_data=positions,
            error_data=errors,
            upper_limits=upper_limits,
            lower_limits=lower_limits,
            untrimmed_positions=track_data.get("untrimmed_positions"),
            untrimmed_errors=track_data.get("untrimmed_errors"),
        )

    def _calculate_sigma(
        self,
        positions: List[float],
        errors: List[float],
        linearity_spec: float,
        travel_length: float,
        unit_length: Optional[float],
        model: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Calculate sigma gradient and threshold.

        The sigma gradient measures the standard deviation of error gradients,
        indicating trim quality and stability.

        Threshold is determined by:
        1. Per-model ML threshold from database (if trained)
        2. Formula-based calculation (fallback)
        """
        # Apply Butterworth filter to smooth errors
        filtered_errors = self._apply_butterworth_filter(errors)

        # Remove endpoints (only if enough points remain for meaningful gradient calc)
        # Need at least 2*END_POINT_FILTER_COUNT + 3 points so that after removal
        # we still have >= 3 points for gradient calculation
        if len(positions) >= 2 * END_POINT_FILTER_COUNT + 3:
            positions = positions[END_POINT_FILTER_COUNT:-END_POINT_FILTER_COUNT]
            filtered_errors = filtered_errors[END_POINT_FILTER_COUNT:-END_POINT_FILTER_COUNT]
        elif len(positions) > 2 * END_POINT_FILTER_COUNT:
            logger.debug(f"Track too short after endpoint removal ({len(positions)} points, "
                        f"need {2 * END_POINT_FILTER_COUNT + 3}), skipping endpoint filter")

        # Calculate gradients
        gradients = []
        step_size = MATLAB_GRADIENT_STEP

        # Use the minimum length to avoid index errors
        min_len = min(len(positions), len(filtered_errors))
        if min_len <= step_size:
            logger.warning(f"Array too short for gradient: {min_len} points")
            return 0.0, self._get_threshold(model, unit_length, linearity_spec, travel_length)

        for i in range(min_len - step_size):
            dx = positions[i + step_size] - positions[i]
            dy = filtered_errors[i + step_size] - filtered_errors[i]

            if abs(dx) > 1e-6:
                gradient = dy / dx
                if not (np.isnan(gradient) or np.isinf(gradient)):
                    gradients.append(gradient)

        # Calculate sigma (standard deviation of gradients)
        if len(gradients) < 2:
            logger.warning("Insufficient gradients for sigma calculation")
            sigma_gradient = 0.0
        else:
            sigma_gradient = float(np.std(gradients, ddof=1))

        # Determine threshold
        sigma_threshold = self._get_threshold(
            model, unit_length, linearity_spec, travel_length
        )

        logger.debug(f"Sigma: gradient={sigma_gradient:.6f}, threshold={sigma_threshold:.6f}")

        return sigma_gradient, sigma_threshold

    def _get_threshold(
        self,
        model: Optional[str],
        unit_length: Optional[float],
        linearity_spec: float,
        travel_length: float,
    ) -> float:
        """
        Get sigma threshold, using per-model ML threshold if available.

        Priority:
        1. Per-model ML threshold (from database training)
        2. Formula-based calculation (fallback)
        """
        # Try per-model threshold from ML training
        if model and model in self._model_thresholds:
            ml_threshold = self._model_thresholds[model]
            logger.debug(f"Using ML threshold for {model}: {ml_threshold:.6f}")
            return ml_threshold

        # Fall back to formula-based calculation
        return self._formula_threshold(linearity_spec, travel_length)

    def _formula_threshold(self, linearity_spec: float, travel_length: float) -> float:
        """
        Calculate threshold using formula (fallback).

        Formula: threshold = linearity_spec / (scaling_factor * travel_factor)
        """
        if travel_length > 0 and linearity_spec > 0:
            travel_factor = max(1.0, travel_length / 100.0)  # Normalize to 100 degrees
            sigma_threshold = linearity_spec / (self.scaling_factor * travel_factor)
        else:
            sigma_threshold = 0.001  # Fallback

        # Ensure threshold is reasonable
        return max(sigma_threshold, 0.0001)

    def _apply_butterworth_filter(self, errors: List[float]) -> List[float]:
        """Apply Butterworth low-pass filter to smooth errors."""
        try:
            if len(errors) < 10:
                return errors

            b, a = butter(BUTTERWORTH_ORDER, BUTTERWORTH_CUTOFF, btype='low')
            filtered = filtfilt(b, a, errors)
            return list(filtered)
        except Exception as e:
            logger.debug(f"Butterworth filter failed: {e}")
            return errors

    def _calculate_linearity(
        self,
        positions: List[float],
        errors: List[float],
        upper_limits: List[float],
        lower_limits: List[float],
        linearity_spec: float,
        linearity_type: Optional[str] = None,
    ) -> Tuple[float, float, float, bool, int, float, int]:
        """
        Calculate linearity metrics with spec-aware optimal adjustment.

        Returns:
            (optimal_offset, optimal_slope, linearity_error, linearity_pass,
             fail_points, raw_linearity_error, raw_fail_points)
        """
        # Calculate raw results (no adjustment)
        raw_fail_points = self._count_fail_points(errors, upper_limits, lower_limits)
        raw_linearity_error = max(abs(e) for e in errors) if errors else 0.0

        # Calculate optimal adjustment (constrained by linearity type)
        optimal_offset, optimal_slope = self._calculate_optimal_adjustment(
            positions, errors, upper_limits, lower_limits, linearity_type
        )

        # Apply adjustment: adjusted = error * slope + offset
        shifted_errors = [e * optimal_slope + optimal_offset for e in errors]

        # Calculate optimized max error
        linearity_error = max(abs(e) for e in shifted_errors) if shifted_errors else 0.0

        # Count fail points after adjustment
        fail_points = self._count_fail_points(shifted_errors, upper_limits, lower_limits)

        # Linearity passes only if ALL points are within limits (zero tolerance)
        linearity_pass = fail_points == 0

        logger.debug(
            f"Linearity: type={linearity_type}, offset={optimal_offset:.6f}, "
            f"slope={optimal_slope:.6f}, error={linearity_error:.6f}, "
            f"fail_points={fail_points} (raw={raw_fail_points}), pass={linearity_pass}"
        )

        return (optimal_offset, optimal_slope, linearity_error, linearity_pass,
                fail_points, raw_linearity_error, raw_fail_points)

    def _calculate_optimal_adjustment(
        self,
        positions: List[float],
        errors: List[float],
        upper_limits: List[float],
        lower_limits: List[float],
        linearity_type: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Calculate optimal offset and slope adjustment, constrained by linearity type.

        Linearity types control which degrees of freedom are available:
        - Absolute / Term Base: No adjustment (offset=0, slope=1.0)
        - Independent: Free offset + slope optimization (most common)
        - Zero-Based: Slope optimization only (offset=0)
        - None/unknown: Falls back to offset-only (legacy behavior)

        Returns:
            (optimal_offset, optimal_slope) tuple
        """
        lin_type = (linearity_type or "").strip().lower()

        # Absolute and Term Base: no adjustment allowed
        if lin_type in ("absolute", "term base"):
            return 0.0, 1.0

        # Zero-Based: slope only (offset locked at 0)
        if lin_type == "zero-based":
            slope = self._optimize_slope_only(errors, upper_limits, lower_limits)
            return 0.0, slope

        # Independent: full offset + slope optimization
        if lin_type == "independent":
            return self._optimize_offset_and_slope(errors, upper_limits, lower_limits)

        # Unknown/None: legacy offset-only behavior for backwards compatibility
        offset = self._calculate_optimal_offset(errors, upper_limits, lower_limits)
        return offset, 1.0

    def _optimize_slope_only(
        self,
        errors: List[float],
        upper_limits: List[float],
        lower_limits: List[float],
    ) -> float:
        """Optimize slope with offset fixed at zero."""
        n = min(len(errors), len(upper_limits), len(lower_limits))
        if n == 0:
            return 1.0

        def objective(slope_val: float) -> float:
            violations = 0
            max_err = 0.0
            for i in range(n):
                adjusted = errors[i] * slope_val
                ul = upper_limits[i]
                ll = lower_limits[i]
                if ul is not None and ll is not None:
                    if not (np.isnan(ul) or np.isnan(ll)):
                        if adjusted > ul or adjusted < ll:
                            violations += 1
                        max_err = max(max_err, abs(adjusted))
            return violations * 1e6 + max_err

        try:
            result = optimize.minimize_scalar(
                objective, bounds=(0.8, 1.2), method='bounded',
                options={'xatol': 1e-6}
            )
            return float(result.x)
        except Exception:
            return 1.0

    def _optimize_offset_and_slope(
        self,
        errors: List[float],
        upper_limits: List[float],
        lower_limits: List[float],
    ) -> Tuple[float, float]:
        """Optimize both offset and slope for Independent linearity."""
        n = min(len(errors), len(upper_limits), len(lower_limits))
        if n == 0:
            return 0.0, 1.0

        # Calculate band center differences for initial offset guess
        differences = []
        for i in range(n):
            ul = upper_limits[i]
            ll = lower_limits[i]
            if ul is not None and ll is not None:
                if not (np.isnan(ul) or np.isnan(ll)):
                    midpoint = (ul + ll) / 2
                    differences.append(midpoint - errors[i])
        initial_offset = float(np.median(differences)) if differences else 0.0

        def objective(params):
            offset, slope = params
            violations = 0
            max_err = 0.0
            for i in range(n):
                adjusted = errors[i] * slope + offset
                ul = upper_limits[i]
                ll = lower_limits[i]
                if ul is not None and ll is not None:
                    if not (np.isnan(ul) or np.isnan(ll)):
                        if adjusted > ul or adjusted < ll:
                            violations += 1
                        max_err = max(max_err, abs(adjusted))
            return violations * 1e6 + max_err

        try:
            # Stage 1: coarse grid search
            best_params = (initial_offset, 1.0)
            best_cost = objective(best_params)
            offset_range = abs(initial_offset) + 0.02
            for slope_candidate in [0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10]:
                for offset_factor in np.linspace(-offset_range, offset_range, 11):
                    cost = objective((offset_factor, slope_candidate))
                    if cost < best_cost:
                        best_cost = cost
                        best_params = (offset_factor, slope_candidate)

            # Stage 2: Nelder-Mead refinement
            result = optimize.minimize(
                objective, x0=best_params, method='Nelder-Mead',
                options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 500}
            )
            return float(result.x[0]), float(result.x[1])
        except Exception:
            return initial_offset, 1.0

    def _calculate_optimal_offset(
        self,
        errors: List[float],
        upper_limits: List[float],
        lower_limits: List[float]
    ) -> float:
        """
        Calculate optimal offset to center errors within limits.

        Uses median of differences from band center as initial guess,
        then optimizes to minimize violations.
        """
        if not upper_limits or not lower_limits:
            return 0.0

        # Ensure same length
        n = min(len(errors), len(upper_limits), len(lower_limits))
        if n == 0:
            return 0.0

        # Calculate differences from band center
        differences = []
        for i in range(n):
            if upper_limits[i] is not None and lower_limits[i] is not None:
                if not (np.isnan(upper_limits[i]) or np.isnan(lower_limits[i])):
                    midpoint = (upper_limits[i] + lower_limits[i]) / 2
                    differences.append(midpoint - errors[i])

        if not differences:
            return 0.0

        # Use median as initial estimate
        median_offset = float(np.median(differences))

        # Optimize to minimize violations
        def violation_count(offset: float) -> float:
            count = 0
            for i in range(n):
                shifted = errors[i] + offset
                if upper_limits[i] is not None and lower_limits[i] is not None:
                    if not (np.isnan(upper_limits[i]) or np.isnan(lower_limits[i])):
                        if shifted > upper_limits[i] or shifted < lower_limits[i]:
                            count += 1
            return count

        try:
            # Search around median
            search_range = abs(median_offset) + 0.01
            result = optimize.minimize_scalar(
                violation_count,
                bounds=(median_offset - search_range, median_offset + search_range),
                method='bounded'
            )
            return float(result.x)
        except Exception:
            return median_offset

    def _count_fail_points(
        self,
        errors: List[float],
        upper_limits: List[float],
        lower_limits: List[float]
    ) -> int:
        """Count points outside specification limits."""
        if not upper_limits or not lower_limits:
            return 0

        n = min(len(errors), len(upper_limits), len(lower_limits))
        count = 0

        for i in range(n):
            if upper_limits[i] is not None and lower_limits[i] is not None:
                if not (np.isnan(upper_limits[i]) or np.isnan(lower_limits[i])):
                    if errors[i] > upper_limits[i] or errors[i] < lower_limits[i]:
                        count += 1

        return count

    def _calculate_failure_margins(
        self,
        shifted_errors: List[float],
        upper_limits: List[float],
        lower_limits: List[float],
    ) -> Dict[str, Optional[float]]:
        """
        Calculate how far points are from spec limits.

        For failed tracks: max_violation and avg_violation (how far out of spec)
        For passing tracks: margin_to_spec (how close to the nearest limit, as % of spec width)

        Returns:
            Dict with max_violation, avg_violation, margin_to_spec
        """
        result = {"max_violation": None, "avg_violation": None, "margin_to_spec": None}

        if not upper_limits or not lower_limits:
            return result

        n = min(len(shifted_errors), len(upper_limits), len(lower_limits))
        if n == 0:
            return result

        violations = []
        margins = []

        for i in range(n):
            ul = upper_limits[i]
            ll = lower_limits[i]
            if ul is None or ll is None:
                continue
            if np.isnan(ul) or np.isnan(ll):
                continue

            e = shifted_errors[i]
            spec_width = ul - ll
            if spec_width <= 0:
                continue

            # Check if point is outside limits
            if e > ul:
                violations.append(e - ul)
            elif e < ll:
                violations.append(ll - e)
            else:
                # Point is within limits — calculate margin to nearest limit
                margin_upper = ul - e
                margin_lower = e - ll
                nearest_margin = min(margin_upper, margin_lower)
                margins.append(nearest_margin / spec_width * 100)  # As % of spec width

        if violations:
            result["max_violation"] = max(violations)
            result["avg_violation"] = sum(violations) / len(violations)
        elif margins:
            # Only set margin_to_spec for passing tracks (no violations)
            result["margin_to_spec"] = min(margins)

        return result

    def _assess_risk(
        self,
        sigma_gradient: float,
        sigma_threshold: float,
        linearity_pass: bool
    ) -> Tuple[float, RiskCategory]:
        """
        Assess failure risk based on sigma margin and linearity.

        Returns:
            (failure_probability, risk_category)
        """
        # Calculate sigma ratio (gradient / threshold)
        if sigma_threshold > 0:
            sigma_ratio = sigma_gradient / sigma_threshold
        else:
            sigma_ratio = 1.0

        # Base probability on sigma ratio
        # ratio >= 1.0 means fail, ratio near 0 means very safe
        if sigma_ratio >= 1.0:
            base_probability = 0.9 + min(0.1, (sigma_ratio - 1.0) * 0.1)
        else:
            base_probability = sigma_ratio * 0.8

        # Adjust for linearity
        if not linearity_pass:
            base_probability = min(1.0, base_probability + 0.2)

        failure_probability = min(1.0, max(0.0, base_probability))

        # Categorize risk
        if failure_probability >= HIGH_RISK_THRESHOLD:
            risk_category = RiskCategory.HIGH
        elif failure_probability >= MEDIUM_RISK_THRESHOLD:
            risk_category = RiskCategory.MEDIUM
        else:
            risk_category = RiskCategory.LOW

        return failure_probability, risk_category

    def _detect_anomaly(
        self,
        positions: List[float],
        errors: List[float],
        sigma_gradient: float,
        sigma_threshold: float,
        fail_points: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect anomalous units (likely trim failures).

        Characteristics of trim failures:
        - Error data forms a highly linear pattern (high R² when fitting line)
        - Often shows monotonic decrease/increase (steep slope)
        - Very high sigma gradient (>>threshold)
        - Many fail points

        Returns:
            (is_anomaly, reason) tuple
        """
        if len(positions) < 10 or len(errors) < 10:
            return False, None

        try:
            # Fit linear regression to error data
            # For normal trimmed data, R² should be low (random errors around 0)
            # For trim failures, R² will be high (linear slope pattern)
            x = np.array(positions[:len(errors)])
            y = np.array(errors)

            # Simple linear regression
            n = len(x)
            x_mean = np.mean(x)
            y_mean = np.mean(y)

            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)

            if abs(denominator) < 1e-10:
                return False, None

            slope = numerator / denominator
            intercept = y_mean - slope * x_mean

            # Calculate R² (coefficient of determination)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)

            if ss_tot < 1e-10:
                return False, None

            r_squared = 1 - (ss_res / ss_tot)

            # Calculate error range (max - min)
            error_range = np.max(y) - np.min(y)

            # Anomaly detection: Focus on linear slope pattern (trim failure signature)
            #
            # Normal trimmed data should have random errors around 0 (low R²)
            # Trim failures show a perfect linear slope (very high R²) because
            # the trimmer didn't actually cut/adjust the resistor
            #
            # Criteria: High R² AND significant error range (not just noise)
            # R² > 0.95 is very strict - only catches truly linear patterns
            # Error range threshold is spec-relative: 10x the sigma threshold
            # catches truly anomalous data while working for normalized errors
            # (typically 0.001-0.01 range). Floor of 0.02 (2% of full scale)
            # is a reasonable anomaly threshold for normalized data.

            if r_squared > 0.95 and error_range > max(sigma_threshold * 10, 0.02):
                return True, f"Linear slope pattern (R²={r_squared:.3f}, range={error_range:.2f})"

            return False, None

        except Exception as e:
            logger.debug(f"Anomaly detection error: {e}")
            return False, None

    def _calculate_trim_effectiveness(
        self,
        trimmed_errors: List[float],
        untrimmed_errors: List[float],
        untrimmed_resistance: Optional[float],
        trimmed_resistance: Optional[float],
    ) -> Dict[str, Any]:
        """
        Calculate trim effectiveness metrics by comparing pre/post trim data.

        Returns dict of metrics to unpack into TrackData.
        """
        result: Dict[str, Any] = {}

        # Resistance change
        if untrimmed_resistance and trimmed_resistance and untrimmed_resistance > 0:
            result["resistance_change"] = trimmed_resistance - untrimmed_resistance

        # Linearity improvement from trim
        if untrimmed_errors and len(untrimmed_errors) > 0:
            valid_untrimmed = [e for e in untrimmed_errors
                               if e is not None and not np.isnan(e)]
            valid_trimmed = [e for e in trimmed_errors
                             if e is not None and not np.isnan(e)]

            if valid_untrimmed and valid_trimmed:
                untrimmed_rms = float(np.sqrt(np.mean(np.array(valid_untrimmed) ** 2)))
                trimmed_rms = float(np.sqrt(np.mean(np.array(valid_trimmed) ** 2)))

                result["untrimmed_rms_error"] = untrimmed_rms
                result["trimmed_rms_error"] = trimmed_rms

                if untrimmed_rms > 0:
                    result["trim_improvement_percent"] = (
                        (untrimmed_rms - trimmed_rms) / untrimmed_rms
                    ) * 100

                    max_untrimmed = max(abs(e) for e in valid_untrimmed)
                    max_trimmed = max(abs(e) for e in valid_trimmed)
                    if max_untrimmed > 0:
                        result["max_error_reduction_percent"] = (
                            (max_untrimmed - max_trimmed) / max_untrimmed
                        ) * 100

        return result

    def _create_failed_track(self, track_id: str, reason: str) -> TrackData:
        """Create a failed track result."""
        return TrackData(
            track_id=track_id,
            status=AnalysisStatus.ERROR,
            travel_length=0.0,
            linearity_spec=0.01,
            sigma_gradient=999.999,
            sigma_threshold=0.001,
            sigma_pass=False,
            optimal_offset=0.0,
            linearity_error=999.999,
            linearity_pass=False,
            linearity_fail_points=0,
            failure_probability=1.0,
            risk_category=RiskCategory.UNKNOWN,
            is_anomaly=True,
            anomaly_reason=reason,
        )
