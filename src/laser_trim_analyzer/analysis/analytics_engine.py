"""
Unified Analytics Engine for Laser Trim Analyzer

This module provides a clean, extensible interface for all potentiometer analysis types.
Designed for easy testing, configuration, and extension with custom analytics.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Protocol, Type, Callable
import numpy as np
import pandas as pd
from contextlib import contextmanager


# ============================================================================
# Data Classes for Structured Results
# ============================================================================

@dataclass
class LinearityResult:
    """Results from linearity analysis."""
    max_deviation: float
    max_deviation_position: float
    avg_deviation: float
    deviation_uniformity: float
    linearity_slope: float
    linearity_intercept: float
    optimal_offset: float
    linearity_pass: bool
    max_error_with_offset: float
    num_fail_points: int
    valid_points_checked: int

    @property
    def is_valid(self) -> bool:
        """Check if the analysis produced valid results."""
        return self.max_deviation is not None


@dataclass
class ResistanceResult:
    """Results from resistance change analysis."""
    untrimmed_resistance: Optional[float]
    trimmed_resistance: Optional[float]
    resistance_change: Optional[float]
    resistance_change_percent: Optional[float]

    @property
    def has_change(self) -> bool:
        """Check if resistance changed during trimming."""
        return (self.resistance_change is not None and
                abs(self.resistance_change) > 0.001)


@dataclass
class ZoneResult:
    """Results from a single zone analysis."""
    zone_number: int
    position_range: tuple[float, float]
    avg_error: float
    max_error: float
    error_variance: float
    data_points: int


@dataclass
class ZoneAnalysisResult:
    """Complete zone analysis results."""
    num_zones: int
    zone_results: List[ZoneResult]
    worst_zone: Optional[ZoneResult] = None

    def __post_init__(self):
        """Identify worst zone after initialization."""
        if self.zone_results:
            self.worst_zone = max(self.zone_results, key=lambda z: z.max_error)


@dataclass
class TrimEffectivenessResult:
    """Results from trim effectiveness analysis."""
    improvement_percent: float
    untrimmed_rms: float
    trimmed_rms: float
    max_error_reduction: float
    untrimmed_max_error: float
    trimmed_max_error: float

    @property
    def is_effective(self) -> bool:
        """Check if trimming was effective."""
        return self.improvement_percent > 10.0  # 10% improvement threshold


@dataclass
class FailureProbabilityResult:
    """Results from failure probability analysis."""
    failure_score: float
    failure_probability: float
    risk_category: str
    gradient_margin: float

    @property
    def is_high_risk(self) -> bool:
        """Check if unit is high risk."""
        return self.risk_category == "High"


@dataclass
class DynamicRangeResult:
    """Results from dynamic range analysis."""
    range_utilization_percent: float
    minimum_margin: float
    minimum_margin_position: float
    average_margin: float
    margin_bias: str
    margin_bias_percent: float

    @property
    def is_near_limits(self) -> bool:
        """Check if operating near tolerance limits."""
        return self.minimum_margin < 0.001  # Less than 1mV margin


@dataclass
class AnalysisInput:
    """Input data structure for analysis."""
    position: List[float]
    error: List[float]
    upper_limit: Optional[List[float]] = None
    lower_limit: Optional[List[float]] = None
    untrimmed_data: Optional[Dict[str, List[float]]] = None
    trimmed_data: Optional[Dict[str, List[float]]] = None

    # Additional parameters
    sigma_gradient: Optional[float] = None
    sigma_threshold: Optional[float] = None
    linearity_spec: Optional[float] = None
    travel_length: Optional[float] = None
    untrimmed_resistance: Optional[float] = None
    trimmed_resistance: Optional[float] = None

    def validate(self) -> List[str]:
        """Validate input data and return list of issues."""
        issues = []

        if not self.position:
            issues.append("Position data is empty")
        if not self.error:
            issues.append("Error data is empty")
        if len(self.position) != len(self.error):
            issues.append("Position and error arrays have different lengths")
            
        # Validate upper and lower limits if provided
        if self.upper_limit and len(self.upper_limit) != len(self.position):
            issues.append("Upper limit array length doesn't match position array")
        if self.lower_limit and len(self.lower_limit) != len(self.position):
            issues.append("Lower limit array length doesn't match position array")
            
        # Check for NaN or invalid values
        try:
            import numpy as np
            if np.isnan(np.array(self.position, dtype=float)).any():
                issues.append("Position data contains NaN values")
            if np.isnan(np.array(self.error, dtype=float)).any():
                issues.append("Error data contains NaN values")
        except Exception:
            # If we can't check for NaN, just continue
            pass

        return issues


@dataclass
class CompleteAnalysisResult:
    """Container for all analysis results."""
    linearity: Optional[LinearityResult] = None
    resistance: Optional[ResistanceResult] = None
    zones: Optional[ZoneAnalysisResult] = None
    trim_effectiveness: Optional[TrimEffectivenessResult] = None
    failure_probability: Optional[FailureProbabilityResult] = None
    dynamic_range: Optional[DynamicRangeResult] = None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all results."""
        summary = {
            "overall_pass": True,
            "risk_level": "Low",
            "key_metrics": {}
        }

        if self.linearity:
            summary["key_metrics"]["linearity_pass"] = self.linearity.linearity_pass
            summary["overall_pass"] &= self.linearity.linearity_pass

        if self.failure_probability:
            summary["risk_level"] = self.failure_probability.risk_category
            summary["key_metrics"]["failure_probability"] = self.failure_probability.failure_probability

        if self.trim_effectiveness:
            summary["key_metrics"]["trim_improvement"] = self.trim_effectiveness.improvement_percent

        return summary


# ============================================================================
# Analysis Profiles
# ============================================================================

class AnalysisProfile(Enum):
    """Predefined analysis profiles."""
    BASIC = auto()  # Linearity and resistance only
    STANDARD = auto()  # All standard analyses
    DETAILED = auto()  # All analyses with extra metrics
    CUSTOM = auto()  # User-defined selection


@dataclass
class AnalysisConfig:
    """Configuration for analysis engine."""
    profile: AnalysisProfile = AnalysisProfile.STANDARD
    enabled_analyses: List[str] = field(default_factory=list)

    # Thresholds and parameters
    high_risk_threshold: float = 0.7
    low_risk_threshold: float = 0.3
    num_zones: int = 5

    # Feature flags
    calculate_optimal_offset: bool = True
    use_median_offset: bool = True  # Use median vs mean for offset

    def __post_init__(self):
        """Set default enabled analyses based on profile."""
        if not self.enabled_analyses:
            if self.profile == AnalysisProfile.BASIC:
                self.enabled_analyses = ["linearity", "resistance"]
            elif self.profile == AnalysisProfile.STANDARD:
                self.enabled_analyses = [
                    "linearity", "resistance", "zones",
                    "trim_effectiveness", "failure_probability"
                ]
            elif self.profile == AnalysisProfile.DETAILED:
                self.enabled_analyses = [
                    "linearity", "resistance", "zones",
                    "trim_effectiveness", "failure_probability",
                    "dynamic_range"
                ]


# ============================================================================
# Base Analyzer Protocol
# ============================================================================

class Analyzer(Protocol):
    """Protocol for analyzer plugins."""

    @property
    def name(self) -> str:
        """Unique name for this analyzer."""
        ...

    @property
    def required_inputs(self) -> List[str]:
        """List of required input fields."""
        ...

    def analyze(self, input_data: AnalysisInput,
                config: AnalysisConfig) -> Any:
        """Run analysis and return results."""
        ...


# ============================================================================
# Core Analyzers
# ============================================================================

class LinearityAnalyzer:
    """Analyzer for linearity deviation."""

    name = "linearity"
    required_inputs = ["position", "error"]

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def analyze(self, input_data: AnalysisInput,
                config: AnalysisConfig) -> LinearityResult:
        """Analyze linearity deviation."""
        try:
            position = np.array(input_data.position)
            error = np.array(input_data.error)

            # Fit linear trend
            slope, intercept = np.polyfit(position, error, 1)
            ideal_error = slope * position + intercept

            # Calculate deviations
            deviation = error - ideal_error
            abs_deviation = np.abs(deviation)

            max_idx = np.argmax(abs_deviation)

            # Calculate optimal offset if limits are provided
            optimal_offset = 0.0
            linearity_pass = False
            max_error_with_offset = np.max(np.abs(error))
            num_fail_points = 0
            valid_points = len(error)

            if (config.calculate_optimal_offset and
                    input_data.upper_limit and input_data.lower_limit):
                optimal_offset = self._calculate_optimal_offset(
                    error, input_data.upper_limit,
                    input_data.lower_limit, config.use_median_offset
                )

                # Check linearity with offset
                check_result = self._check_linearity_with_offset(
                    error, input_data.upper_limit,
                    input_data.lower_limit, optimal_offset
                )

                linearity_pass = check_result["pass"]
                max_error_with_offset = check_result["max_error"]
                num_fail_points = check_result["fail_points"]
                valid_points = check_result["valid_points"]

            return LinearityResult(
                max_deviation=abs_deviation[max_idx],
                max_deviation_position=position[max_idx],
                avg_deviation=np.mean(abs_deviation),
                deviation_uniformity=np.std(deviation),
                linearity_slope=slope,
                linearity_intercept=intercept,
                optimal_offset=optimal_offset,
                linearity_pass=linearity_pass,
                max_error_with_offset=max_error_with_offset,
                num_fail_points=num_fail_points,
                valid_points_checked=valid_points
            )

        except Exception as e:
            self.logger.error(f"Linearity analysis failed: {e}")
            raise

    def _calculate_optimal_offset(self, error: np.ndarray,
                                  upper_limit: List[float],
                                  lower_limit: List[float],
                                  use_median: bool) -> float:
        """Calculate optimal vertical offset."""
        valid_indices = [
            i for i in range(len(error))
            if upper_limit[i] is not None and lower_limit[i] is not None
        ]

        if not valid_indices:
            return 0.0

        valid_error = error[valid_indices]
        midpoints = [(upper_limit[i] + lower_limit[i]) / 2
                     for i in valid_indices]

        differences = valid_error - np.array(midpoints)

        if use_median:
            return -np.median(differences)
        else:
            return -np.mean(differences)

    def _check_linearity_with_offset(self, error: np.ndarray,
                                     upper_limit: List[float],
                                     lower_limit: List[float],
                                     offset: float) -> Dict[str, Any]:
        """Check linearity pass/fail with offset."""
        shifted_error = error + offset
        fail_points = 0
        max_error = 0.0
        valid_points = 0

        for i in range(len(shifted_error)):
            if upper_limit[i] is not None and lower_limit[i] is not None:
                valid_points += 1
                current_error = shifted_error[i]
                max_error = max(max_error, abs(current_error))

                if not (lower_limit[i] <= current_error <= upper_limit[i]):
                    fail_points += 1

        return {
            "pass": fail_points == 0,
            "max_error": max_error,
            "fail_points": fail_points,
            "valid_points": valid_points
        }


class ResistanceAnalyzer:
    """Analyzer for resistance changes."""

    name = "resistance"
    required_inputs = ["untrimmed_resistance", "trimmed_resistance"]

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def analyze(self, input_data: AnalysisInput,
                config: AnalysisConfig) -> ResistanceResult:
        """Analyze resistance change."""
        untrimmed = input_data.untrimmed_resistance
        trimmed = input_data.trimmed_resistance

        # Handle None/NaN values
        if pd.isna(untrimmed):
            untrimmed = None
        if pd.isna(trimmed):
            trimmed = None

        # Calculate change if both values exist
        if untrimmed is not None and trimmed is not None:
            change = trimmed - untrimmed
            change_percent = (change / untrimmed * 100) if untrimmed != 0 else 0
        else:
            change = None
            change_percent = None

        return ResistanceResult(
            untrimmed_resistance=untrimmed,
            trimmed_resistance=trimmed,
            resistance_change=change,
            resistance_change_percent=change_percent
        )


class ZoneAnalyzer:
    """Analyzer for travel zone performance."""

    name = "zones"
    required_inputs = ["position", "error"]

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def analyze(self, input_data: AnalysisInput,
                config: AnalysisConfig) -> ZoneAnalysisResult:
        """Analyze travel zones."""
        position = input_data.position
        error = input_data.error
        num_zones = config.num_zones

        min_pos = min(position)
        max_pos = max(position)
        zone_size = (max_pos - min_pos) / num_zones

        zone_results = []

        for i in range(num_zones):
            zone_start = min_pos + i * zone_size
            zone_end = zone_start + zone_size

            # Find points in this zone
            zone_indices = [
                j for j, pos in enumerate(position)
                if zone_start <= pos < zone_end
            ]

            if zone_indices:
                zone_error = [error[j] for j in zone_indices]

                zone_results.append(ZoneResult(
                    zone_number=i + 1,
                    position_range=(zone_start, zone_end),
                    avg_error=np.mean(zone_error),
                    max_error=max(abs(e) for e in zone_error),
                    error_variance=np.var(zone_error),
                    data_points=len(zone_indices)
                ))

        return ZoneAnalysisResult(
            num_zones=num_zones,
            zone_results=zone_results
        )


class TrimEffectivenessAnalyzer:
    """Analyzer for trim effectiveness."""

    name = "trim_effectiveness"
    required_inputs = ["untrimmed_data", "trimmed_data"]

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def analyze(self, input_data: AnalysisInput,
                config: AnalysisConfig) -> TrimEffectivenessResult:
        """Analyze trim effectiveness."""
        untrimmed = input_data.untrimmed_data
        trimmed = input_data.trimmed_data

        if not untrimmed or not trimmed:
            raise ValueError("Both untrimmed and trimmed data required")

        # Calculate RMS errors
        untrimmed_rms = np.sqrt(np.mean(np.square(untrimmed["error"])))
        trimmed_rms = np.sqrt(np.mean(np.square(trimmed["error"])))

        # Calculate improvement
        improvement = 0.0
        if untrimmed_rms > 0:
            improvement = ((untrimmed_rms - trimmed_rms) / untrimmed_rms) * 100

        # Calculate max errors
        untrimmed_max = max(abs(e) for e in untrimmed["error"])
        trimmed_max = max(abs(e) for e in trimmed["error"])

        max_reduction = 0.0
        if untrimmed_max > 0:
            max_reduction = ((untrimmed_max - trimmed_max) / untrimmed_max) * 100

        return TrimEffectivenessResult(
            improvement_percent=improvement,
            untrimmed_rms=untrimmed_rms,
            trimmed_rms=trimmed_rms,
            max_error_reduction=max_reduction,
            untrimmed_max_error=untrimmed_max,
            trimmed_max_error=trimmed_max
        )


class FailureProbabilityAnalyzer:
    """Analyzer for failure probability."""

    name = "failure_probability"
    required_inputs = ["sigma_gradient", "sigma_threshold", "linearity_spec"]

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def analyze(self, input_data: AnalysisInput,
                config: AnalysisConfig) -> FailureProbabilityResult:
        """Analyze failure probability."""
        sigma_gradient = input_data.sigma_gradient
        sigma_threshold = input_data.sigma_threshold
        linearity_spec = input_data.linearity_spec

        if not all([sigma_gradient, sigma_threshold, linearity_spec]):
            raise ValueError("Required parameters missing")

        # Calculate normalized gradient
        normalized_gradient = sigma_gradient / sigma_threshold if sigma_threshold > 0 else 999
        gradient_margin = 1 - normalized_gradient

        # Weighted score calculation
        weights = {"gradient_margin": 0.7, "linearity_spec": 0.3}

        score = (weights["gradient_margin"] * gradient_margin +
                 weights["linearity_spec"] * (0.02 / max(0.001, linearity_spec)))

        # Convert to probability using sigmoid
        failure_probability = 1 / (1 + np.exp(2 * score))

        # Determine risk category
        if failure_probability > config.high_risk_threshold:
            risk_category = "High"
        elif failure_probability > config.low_risk_threshold:
            risk_category = "Medium"
        else:
            risk_category = "Low"

        return FailureProbabilityResult(
            failure_score=score,
            failure_probability=failure_probability,
            risk_category=risk_category,
            gradient_margin=gradient_margin
        )


class DynamicRangeAnalyzer:
    """Analyzer for dynamic range utilization."""

    name = "dynamic_range"
    required_inputs = ["position", "error", "upper_limit", "lower_limit"]

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def analyze(self, input_data: AnalysisInput,
                config: AnalysisConfig) -> DynamicRangeResult:
        """Analyze dynamic range utilization."""
        # Get valid indices where limits exist
        valid_indices = [
            i for i in range(len(input_data.error))
            if (i < len(input_data.upper_limit) and
                input_data.upper_limit[i] is not None and
                i < len(input_data.lower_limit) and
                input_data.lower_limit[i] is not None)
        ]

        if not valid_indices:
            raise ValueError("No valid limit data")

        # Filter data
        error = [input_data.error[i] for i in valid_indices]
        position = [input_data.position[i] for i in valid_indices]
        upper = [input_data.upper_limit[i] for i in valid_indices]
        lower = [input_data.lower_limit[i] for i in valid_indices]

        # Calculate metrics
        available_range = np.mean([u - l for u, l in zip(upper, lower)])
        used_range = max(error) - min(error)
        utilization = (used_range / available_range * 100) if available_range > 0 else 0

        # Calculate margins
        upper_margins = [u - e for u, e in zip(upper, error)]
        lower_margins = [e - l for e, l in zip(error, lower)]
        margins = [min(u, l) for u, l in zip(upper_margins, lower_margins)]

        min_idx = np.argmin(margins)

        # Determine bias
        closer_to_upper = sum(1 for u, l in zip(upper_margins, lower_margins) if u < l)
        total = len(margins)

        if closer_to_upper > total / 2:
            bias = "Upper"
            bias_percent = (closer_to_upper / total) * 100
        elif closer_to_upper < total / 2:
            bias = "Lower"
            bias_percent = ((total - closer_to_upper) / total) * 100
        else:
            bias = "Balanced"
            bias_percent = 50.0

        return DynamicRangeResult(
            range_utilization_percent=utilization,
            minimum_margin=margins[min_idx],
            minimum_margin_position=position[min_idx],
            average_margin=np.mean(margins),
            margin_bias=bias,
            margin_bias_percent=bias_percent
        )


# ============================================================================
# Plugin System
# ============================================================================

class AnalyzerPlugin(ABC):
    """Base class for custom analyzer plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this analyzer."""
        pass

    @property
    @abstractmethod
    def required_inputs(self) -> List[str]:
        """List of required input fields."""
        pass

    @abstractmethod
    def analyze(self, input_data: AnalysisInput,
                config: AnalysisConfig) -> Any:
        """Run analysis and return results."""
        pass


# ============================================================================
# Main Analytics Engine
# ============================================================================

class AnalyticsEngine:
    """Unified analytics engine for all potentiometer analyses."""

    def __init__(self, config: Optional[AnalysisConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the analytics engine.

        Args:
            config: Analysis configuration
            logger: Logger instance
        """
        self.config = config or AnalysisConfig()
        self.logger = logger or logging.getLogger(__name__)

        # Initialize core analyzers
        self._analyzers: Dict[str, Analyzer] = {
            "linearity": LinearityAnalyzer(self.logger),
            "resistance": ResistanceAnalyzer(self.logger),
            "zones": ZoneAnalyzer(self.logger),
            "trim_effectiveness": TrimEffectivenessAnalyzer(self.logger),
            "failure_probability": FailureProbabilityAnalyzer(self.logger),
            "dynamic_range": DynamicRangeAnalyzer(self.logger)
        }

        # Plugin storage
        self._plugins: Dict[str, AnalyzerPlugin] = {}
        
        # Results cache for performance
        self._results_cache = {}
        self._cache_enabled = True
        self._max_cache_size = 100

        self.logger.info(f"Analytics engine initialized with profile: {self.config.profile}")

    def register_plugin(self, plugin: AnalyzerPlugin) -> None:
        """
        Register a custom analyzer plugin.

        Args:
            plugin: Plugin instance to register
        """
        if plugin.name in self._analyzers or plugin.name in self._plugins:
            raise ValueError(f"Analyzer '{plugin.name}' already registered")

        self._plugins[plugin.name] = plugin
        self.logger.info(f"Registered plugin analyzer: {plugin.name}")

    def analyze(self, input_data: AnalysisInput) -> CompleteAnalysisResult:
        """
        Run all enabled analyses on the input data.

        Args:
            input_data: Input data for analysis

        Returns:
            Complete analysis results
        """
        # Validate input
        issues = input_data.validate()
        if issues:
            raise ValueError(f"Input validation failed: {', '.join(issues)}")
            
        # Check cache if enabled
        if self._cache_enabled:
            # Create a simple cache key based on input data
            import hashlib
            import json
            
            # Create a simplified representation for hashing
            cache_dict = {
                "position_len": len(input_data.position),
                "error_len": len(input_data.error),
                "position_sum": sum(input_data.position),
                "error_sum": sum(input_data.error),
                "sigma_gradient": input_data.sigma_gradient,
                "sigma_threshold": input_data.sigma_threshold,
                "linearity_spec": input_data.linearity_spec,
                "profile": self.config.profile.name,
                "enabled_analyses": ",".join(sorted(self.config.enabled_analyses))
            }
            
            cache_key = hashlib.md5(json.dumps(cache_dict, sort_keys=True).encode()).hexdigest()
            
            if cache_key in self._results_cache:
                self.logger.debug(f"Using cached result for analysis")
                return self._results_cache[cache_key]

        result = CompleteAnalysisResult()

        # Run each enabled analysis
        for analysis_name in self.config.enabled_analyses:
            try:
                if analysis_name in self._analyzers:
                    analyzer = self._analyzers[analysis_name]
                elif analysis_name in self._plugins:
                    analyzer = self._plugins[analysis_name]
                else:
                    self.logger.warning(f"Unknown analyzer: {analysis_name}")
                    continue

                # Check required inputs
                missing = [
                    req for req in analyzer.required_inputs
                    if not hasattr(input_data, req) or getattr(input_data, req) is None
                ]

                if missing:
                    self.logger.warning(
                        f"Skipping {analysis_name}: missing inputs {missing}"
                    )
                    continue

                # Run analysis with better error handling
                self.logger.debug(f"Running {analysis_name} analysis")
                try:
                    analysis_result = analyzer.analyze(input_data, self.config)
                except Exception as analysis_error:
                    self.logger.error(f"Error in {analysis_name} analyzer: {analysis_error}")
                    # Re-raise in detailed mode, otherwise continue with next analyzer
                    if self.config.profile == AnalysisProfile.DETAILED:
                        raise
                    continue

                # Store result
                if analysis_name == "linearity":
                    result.linearity = analysis_result
                elif analysis_name == "resistance":
                    result.resistance = analysis_result
                elif analysis_name == "zones":
                    result.zones = analysis_result
                elif analysis_name == "trim_effectiveness":
                    result.trim_effectiveness = analysis_result
                elif analysis_name == "failure_probability":
                    result.failure_probability = analysis_result
                elif analysis_name == "dynamic_range":
                    result.dynamic_range = analysis_result
                else:
                    # Store plugin results in a dict (extend as needed)
                    if not hasattr(result, 'plugin_results'):
                        result.plugin_results = {}
                    result.plugin_results[analysis_name] = analysis_result

            except Exception as e:
                self.logger.error(f"Error in {analysis_name} analysis: {e}")
                if self.config.profile == AnalysisProfile.DETAILED:
                    raise  # Re-raise in detailed mode for debugging
                    
        # Cache the result if enabled
        if self._cache_enabled:
            # Manage cache size
            if len(self._results_cache) >= self._max_cache_size:
                # Remove oldest entry (first key)
                if self._results_cache:
                    del self._results_cache[next(iter(self._results_cache))]
                    
            # Store in cache
            self._results_cache[cache_key] = result

        return result

    def analyze_single(self, analysis_name: str,
                       input_data: AnalysisInput) -> Any:
        """
        Run a single analysis.

        Args:
            analysis_name: Name of the analysis to run
            input_data: Input data

        Returns:
            Analysis result
        """
        if analysis_name in self._analyzers:
            analyzer = self._analyzers[analysis_name]
        elif analysis_name in self._plugins:
            analyzer = self._plugins[analysis_name]
        else:
            raise ValueError(f"Unknown analyzer: {analysis_name}")

        return analyzer.analyze(input_data, self.config)

    @contextmanager
    def custom_config(self, **kwargs):
        """
        Context manager for temporary config changes.

        Example:
            with engine.custom_config(num_zones=10):
                result = engine.analyze(data)
        """
        original_values = {}

        # Store original values and apply changes
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                original_values[key] = getattr(self.config, key)
                setattr(self.config, key, value)

        try:
            yield self
        finally:
            # Restore original values
            for key, value in original_values.items():
                setattr(self.config, key, value)

    def get_available_analyses(self) -> List[str]:
        """Get list of all available analyses."""
        return list(self._analyzers.keys()) + list(self._plugins.keys())

    def get_required_inputs(self, analysis_name: str) -> List[str]:
        """Get required inputs for a specific analysis."""
        if analysis_name in self._analyzers:
            return self._analyzers[analysis_name].required_inputs
        elif analysis_name in self._plugins:
            return self._plugins[analysis_name].required_inputs
        else:
            raise ValueError(f"Unknown analyzer: {analysis_name}")
            
    def clear_cache(self):
        """Clear the results cache."""
        if hasattr(self, '_results_cache'):
            cache_size = len(self._results_cache)
            self._results_cache.clear()
            self.logger.debug(f"Cleared analytics cache ({cache_size} entries)")
        else:
            self._results_cache = {}
            self.logger.debug("Initialized empty analytics cache")
        
    def set_cache_enabled(self, enabled: bool):
        """Enable or disable results caching."""
        self._cache_enabled = enabled
        self.logger.debug(f"Analytics cache {'enabled' if enabled else 'disabled'}")
        
    def set_max_cache_size(self, size: int):
        """Set maximum cache size."""
        self._max_cache_size = size
        
        # Trim cache if needed
        if len(self._results_cache) > size:
            # Remove oldest entries
            keys_to_remove = list(self._results_cache.keys())[:(len(self._results_cache) - size)]
            for key in keys_to_remove:
                del self._results_cache[key]
            
            self.logger.debug(f"Trimmed analytics cache to {size} entries")


# ============================================================================
# Example Custom Plugin
# ============================================================================

class TrendAnalyzerPlugin(AnalyzerPlugin):
    """Example plugin for analyzing error trends."""

    @property
    def name(self) -> str:
        return "trend"

    @property
    def required_inputs(self) -> List[str]:
        return ["position", "error"]

    def analyze(self, input_data: AnalysisInput,
                config: AnalysisConfig) -> Dict[str, Any]:
        """Analyze error trends."""
        position = np.array(input_data.position)
        error = np.array(input_data.error)

        # Simple trend analysis
        first_half = error[:len(error) // 2]
        second_half = error[len(error) // 2:]

        trend = "increasing" if np.mean(second_half) > np.mean(first_half) else "decreasing"

        return {
            "trend_direction": trend,
            "first_half_mean": np.mean(first_half),
            "second_half_mean": np.mean(second_half),
            "trend_magnitude": abs(np.mean(second_half) - np.mean(first_half))
        }


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import random

    # Create sample data
    position = [i * 0.01 for i in range(100)]
    error = [random.gauss(0, 0.001) for _ in range(100)]
    upper_limit = [0.01] * 100
    lower_limit = [-0.01] * 100

    # Create input data
    input_data = AnalysisInput(
        position=position,
        error=error,
        upper_limit=upper_limit,
        lower_limit=lower_limit,
        sigma_gradient=0.0012,
        sigma_threshold=0.002,
        linearity_spec=0.01,
        untrimmed_resistance=10000,
        trimmed_resistance=10100
    )

    # Create engine with standard profile
    engine = AnalyticsEngine(
        config=AnalysisConfig(profile=AnalysisProfile.STANDARD),
        logger=logging.getLogger("analytics_demo")
    )

    # Register custom plugin
    engine.register_plugin(TrendAnalyzerPlugin())

    # Run analysis
    results = engine.analyze(input_data)

    # Access results
    if results.linearity:
        print(f"Linearity pass: {results.linearity.linearity_pass}")
        print(f"Max deviation: {results.linearity.max_deviation:.6f}")

    if results.failure_probability:
        print(f"Risk category: {results.failure_probability.risk_category}")
        print(f"Failure probability: {results.failure_probability.failure_probability:.2%}")

    # Get summary
    summary = results.get_summary()
    print(f"Overall summary: {summary}")
