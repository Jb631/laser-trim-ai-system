"""
Test Suite for Unified Analytics Engine

Comprehensive tests for all analyzers and engine functionality.
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import logging
from dataclasses import asdict

# Import the analytics engine components
from analytics_engine import (
    AnalyticsEngine, AnalysisConfig, AnalysisProfile, AnalysisInput,
    LinearityResult, ResistanceResult, ZoneResult, ZoneAnalysisResult,
    TrimEffectivenessResult, FailureProbabilityResult, DynamicRangeResult,
    CompleteAnalysisResult, AnalyzerPlugin, TrendAnalyzerPlugin
)


class TestAnalysisDataClasses(unittest.TestCase):
    """Test data class functionality."""

    def test_linearity_result(self):
        """Test LinearityResult dataclass."""
        result = LinearityResult(
            max_deviation=0.001,
            max_deviation_position=50.0,
            avg_deviation=0.0005,
            deviation_uniformity=0.0002,
            linearity_slope=0.01,
            linearity_intercept=0.0,
            optimal_offset=0.0001,
            linearity_pass=True,
            max_error_with_offset=0.0009,
            num_fail_points=0,
            valid_points_checked=100
        )

        self.assertTrue(result.is_valid)
        self.assertTrue(result.linearity_pass)
        self.assertEqual(result.num_fail_points, 0)

    def test_resistance_result(self):
        """Test ResistanceResult dataclass."""
        result = ResistanceResult(
            untrimmed_resistance=10000,
            trimmed_resistance=10100,
            resistance_change=100,
            resistance_change_percent=1.0
        )

        self.assertTrue(result.has_change)
        self.assertEqual(result.resistance_change, 100)

    def test_zone_analysis_result(self):
        """Test ZoneAnalysisResult with automatic worst zone detection."""
        zones = [
            ZoneResult(1, (0, 20), 0.001, 0.002, 0.0001, 100),
            ZoneResult(2, (20, 40), 0.002, 0.005, 0.0002, 100),  # Worst
            ZoneResult(3, (40, 60), 0.001, 0.003, 0.0001, 100),
        ]

        result = ZoneAnalysisResult(num_zones=3, zone_results=zones)

        self.assertIsNotNone(result.worst_zone)
        self.assertEqual(result.worst_zone.zone_number, 2)
        self.assertEqual(result.worst_zone.max_error, 0.005)

    def test_complete_analysis_summary(self):
        """Test CompleteAnalysisResult summary generation."""
        result = CompleteAnalysisResult(
            linearity=LinearityResult(
                max_deviation=0.001, max_deviation_position=50.0,
                avg_deviation=0.0005, deviation_uniformity=0.0002,
                linearity_slope=0.01, linearity_intercept=0.0,
                optimal_offset=0.0001, linearity_pass=False,
                max_error_with_offset=0.0009, num_fail_points=5,
                valid_points_checked=100
            ),
            failure_probability=FailureProbabilityResult(
                failure_score=-0.5, failure_probability=0.8,
                risk_category="High", gradient_margin=-0.2
            )
        )

        summary = result.get_summary()

        self.assertFalse(summary["overall_pass"])
        self.assertEqual(summary["risk_level"], "High")
        self.assertFalse(summary["key_metrics"]["linearity_pass"])
        self.assertEqual(summary["key_metrics"]["failure_probability"], 0.8)


class TestAnalysisInput(unittest.TestCase):
    """Test AnalysisInput validation."""

    def test_valid_input(self):
        """Test validation with valid input."""
        input_data = AnalysisInput(
            position=[0, 1, 2, 3],
            error=[0.1, 0.2, 0.1, 0.2]
        )

        issues = input_data.validate()
        self.assertEqual(len(issues), 0)

    def test_invalid_input(self):
        """Test validation with invalid input."""
        # Empty data
        input1 = AnalysisInput(position=[], error=[1, 2])
        issues1 = input1.validate()
        self.assertIn("Position data is empty", issues1)

        # Mismatched lengths
        input2 = AnalysisInput(position=[1, 2, 3], error=[0.1, 0.2])
        issues2 = input2.validate()
        self.assertIn("Position and error arrays have different lengths", issues2)


class TestAnalysisConfig(unittest.TestCase):
    """Test AnalysisConfig functionality."""

    def test_profile_defaults(self):
        """Test default analysis selection for profiles."""
        # Basic profile
        config_basic = AnalysisConfig(profile=AnalysisProfile.BASIC)
        self.assertEqual(config_basic.enabled_analyses, ["linearity", "resistance"])

        # Standard profile
        config_std = AnalysisConfig(profile=AnalysisProfile.STANDARD)
        self.assertIn("linearity", config_std.enabled_analyses)
        self.assertIn("failure_probability", config_std.enabled_analyses)
        self.assertNotIn("dynamic_range", config_std.enabled_analyses)

        # Detailed profile
        config_detail = AnalysisConfig(profile=AnalysisProfile.DETAILED)
        self.assertIn("dynamic_range", config_detail.enabled_analyses)

    def test_custom_config(self):
        """Test custom configuration."""
        config = AnalysisConfig(
            profile=AnalysisProfile.CUSTOM,
            enabled_analyses=["linearity", "zones"],
            num_zones=10,
            high_risk_threshold=0.8
        )

        self.assertEqual(config.enabled_analyses, ["linearity", "zones"])
        self.assertEqual(config.num_zones, 10)
        self.assertEqual(config.high_risk_threshold, 0.8)


class TestAnalyzers(unittest.TestCase):
    """Test individual analyzers."""

    def setUp(self):
        """Set up test data."""
        self.position = list(range(100))
        self.error = [0.001 * np.sin(i * 0.1) for i in range(100)]
        self.upper_limit = [0.01] * 100
        self.lower_limit = [-0.01] * 100
        self.config = AnalysisConfig()
        self.logger = logging.getLogger("test")

    def test_linearity_analyzer(self):
        """Test LinearityAnalyzer."""
        from analytics_engine import LinearityAnalyzer

        analyzer = LinearityAnalyzer(self.logger)
        input_data = AnalysisInput(
            position=self.position,
            error=self.error,
            upper_limit=self.upper_limit,
            lower_limit=self.lower_limit
        )

        result = analyzer.analyze(input_data, self.config)

        self.assertIsInstance(result, LinearityResult)
        self.assertIsNotNone(result.max_deviation)
        self.assertIsNotNone(result.optimal_offset)
        self.assertIsInstance(result.linearity_pass, bool)

    def test_resistance_analyzer(self):
        """Test ResistanceAnalyzer."""
        from analytics_engine import ResistanceAnalyzer

        analyzer = ResistanceAnalyzer(self.logger)
        input_data = AnalysisInput(
            position=self.position,
            error=self.error,
            untrimmed_resistance=10000,
            trimmed_resistance=10100
        )

        result = analyzer.analyze(input_data, self.config)

        self.assertIsInstance(result, ResistanceResult)
        self.assertEqual(result.resistance_change, 100)
        self.assertEqual(result.resistance_change_percent, 1.0)

    def test_zone_analyzer(self):
        """Test ZoneAnalyzer."""
        from analytics_engine import ZoneAnalyzer

        analyzer = ZoneAnalyzer(self.logger)
        input_data = AnalysisInput(
            position=self.position,
            error=self.error
        )

        result = analyzer.analyze(input_data, self.config)

        self.assertIsInstance(result, ZoneAnalysisResult)
        self.assertEqual(result.num_zones, 5)
        self.assertEqual(len(result.zone_results), 5)
        self.assertIsNotNone(result.worst_zone)

    def test_trim_effectiveness_analyzer(self):
        """Test TrimEffectivenessAnalyzer."""
        from analytics_engine import TrimEffectivenessAnalyzer

        analyzer = TrimEffectivenessAnalyzer(self.logger)

        untrimmed_data = {"error": [0.01 * i for i in range(100)]}
        trimmed_data = {"error": [0.005 * i for i in range(100)]}

        input_data = AnalysisInput(
            position=self.position,
            error=self.error,
            untrimmed_data=untrimmed_data,
            trimmed_data=trimmed_data
        )

        result = analyzer.analyze(input_data, self.config)

        self.assertIsInstance(result, TrimEffectivenessResult)
        self.assertGreater(result.improvement_percent, 0)
        self.assertTrue(result.is_effective)

    def test_failure_probability_analyzer(self):
        """Test FailureProbabilityAnalyzer."""
        from analytics_engine import FailureProbabilityAnalyzer

        analyzer = FailureProbabilityAnalyzer(self.logger)
        input_data = AnalysisInput(
            position=self.position,
            error=self.error,
            sigma_gradient=0.0015,
            sigma_threshold=0.002,
            linearity_spec=0.01
        )

        result = analyzer.analyze(input_data, self.config)

        self.assertIsInstance(result, FailureProbabilityResult)
        self.assertIn(result.risk_category, ["High", "Medium", "Low"])
        self.assertGreaterEqual(result.failure_probability, 0)
        self.assertLessEqual(result.failure_probability, 1)

    def test_dynamic_range_analyzer(self):
        """Test DynamicRangeAnalyzer."""
        from analytics_engine import DynamicRangeAnalyzer

        analyzer = DynamicRangeAnalyzer(self.logger)
        input_data = AnalysisInput(
            position=self.position,
            error=self.error,
            upper_limit=self.upper_limit,
            lower_limit=self.lower_limit
        )

        result = analyzer.analyze(input_data, self.config)

        self.assertIsInstance(result, DynamicRangeResult)
        self.assertGreaterEqual(result.range_utilization_percent, 0)
        self.assertLessEqual(result.range_utilization_percent, 100)
        self.assertIn(result.margin_bias, ["Upper", "Lower", "Balanced"])


class TestAnalyticsEngine(unittest.TestCase):
    """Test the main AnalyticsEngine."""

    def setUp(self):
        """Set up test environment."""
        self.config = AnalysisConfig(profile=AnalysisProfile.STANDARD)
        self.engine = AnalyticsEngine(config=self.config)

        # Create test data
        self.position = list(range(100))
        self.error = [0.001 * np.sin(i * 0.1) for i in range(100)]

        self.input_data = AnalysisInput(
            position=self.position,
            error=self.error,
            upper_limit=[0.01] * 100,
            lower_limit=[-0.01] * 100,
            sigma_gradient=0.0015,
            sigma_threshold=0.002,
            linearity_spec=0.01,
            untrimmed_resistance=10000,
            trimmed_resistance=10100,
            untrimmed_data={"error": self.error},
            trimmed_data={"error": [e * 0.8 for e in self.error]}
        )

    def test_engine_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.config.profile, AnalysisProfile.STANDARD)
        self.assertIn("linearity", self.engine.get_available_analyses())
        self.assertIn("failure_probability", self.engine.get_available_analyses())

    def test_full_analysis(self):
        """Test running full analysis."""
        result = self.engine.analyze(self.input_data)

        self.assertIsInstance(result, CompleteAnalysisResult)
        self.assertIsNotNone(result.linearity)
        self.assertIsNotNone(result.resistance)
        self.assertIsNotNone(result.zones)
        self.assertIsNotNone(result.failure_probability)
        self.assertIsNone(result.dynamic_range)  # Not in standard profile

    def test_single_analysis(self):
        """Test running single analysis."""
        result = self.engine.analyze_single("linearity", self.input_data)

        self.assertIsInstance(result, LinearityResult)
        self.assertIsNotNone(result.max_deviation)

    def test_missing_inputs(self):
        """Test handling of missing inputs."""
        # Create input without resistance data
        limited_input = AnalysisInput(
            position=self.position,
            error=self.error
        )

        # Should skip resistance analysis
        result = self.engine.analyze(limited_input)

        self.assertIsNotNone(result.linearity)  # Should work
        self.assertIsNone(result.resistance)  # Should be skipped

    def test_plugin_registration(self):
        """Test plugin registration."""
        plugin = TrendAnalyzerPlugin()
        self.engine.register_plugin(plugin)

        self.assertIn("trend", self.engine.get_available_analyses())

        # Test duplicate registration
        with self.assertRaises(ValueError):
            self.engine.register_plugin(plugin)

    def test_custom_config_context(self):
        """Test custom config context manager."""
        original_zones = self.engine.config.num_zones

        with self.engine.custom_config(num_zones=10):
            self.assertEqual(self.engine.config.num_zones, 10)
            result = self.engine.analyze_single("zones", self.input_data)
            self.assertEqual(result.num_zones, 10)

        # Config should be restored
        self.assertEqual(self.engine.config.num_zones, original_zones)

    def test_get_required_inputs(self):
        """Test getting required inputs for analyses."""
        linearity_inputs = self.engine.get_required_inputs("linearity")
        self.assertIn("position", linearity_inputs)
        self.assertIn("error", linearity_inputs)

        resistance_inputs = self.engine.get_required_inputs("resistance")
        self.assertIn("untrimmed_resistance", resistance_inputs)
        self.assertIn("trimmed_resistance", resistance_inputs)

    def test_error_handling(self):
        """Test error handling in analysis."""
        # Create invalid input
        invalid_input = AnalysisInput(position=[], error=[])

        with self.assertRaises(ValueError):
            self.engine.analyze(invalid_input)

    def test_plugin_analysis(self):
        """Test running analysis with plugin."""
        # Register plugin
        plugin = TrendAnalyzerPlugin()
        self.engine.register_plugin(plugin)

        # Enable it
        self.engine.config.enabled_analyses.append("trend")

        # Run analysis
        result = self.engine.analyze(self.input_data)

        # Check plugin results
        self.assertTrue(hasattr(result, 'plugin_results'))
        self.assertIn("trend", result.plugin_results)
        self.assertIn("trend_direction", result.plugin_results["trend"])


class TestCustomPlugin(unittest.TestCase):
    """Test custom plugin functionality."""

    def test_trend_analyzer_plugin(self):
        """Test the example TrendAnalyzerPlugin."""
        plugin = TrendAnalyzerPlugin()

        # Create test data with clear trend
        position = list(range(100))
        error = [i * 0.001 for i in range(100)]  # Increasing trend

        input_data = AnalysisInput(position=position, error=error)
        config = AnalysisConfig()

        result = plugin.analyze(input_data, config)

        self.assertEqual(result["trend_direction"], "increasing")
        self.assertGreater(result["second_half_mean"], result["first_half_mean"])


class TestIntegration(unittest.TestCase):
    """Integration tests for realistic scenarios."""

    def test_qa_workflow(self):
        """Test typical QA workflow."""
        # Create realistic test data
        np.random.seed(42)
        position = np.linspace(0, 100, 1000)

        # Untrimmed: larger errors
        untrimmed_error = 0.005 * np.sin(position * 0.1) + np.random.normal(0, 0.001, 1000)

        # Trimmed: reduced errors
        trimmed_error = 0.003 * np.sin(position * 0.1) + np.random.normal(0, 0.0005, 1000)

        # Limits
        upper_limit = [0.01] * 1000
        lower_limit = [-0.01] * 1000

        # Create input
        input_data = AnalysisInput(
            position=position.tolist(),
            error=trimmed_error.tolist(),
            upper_limit=upper_limit,
            lower_limit=lower_limit,
            untrimmed_data={"error": untrimmed_error.tolist()},
            trimmed_data={"error": trimmed_error.tolist()},
            sigma_gradient=0.0012,
            sigma_threshold=0.002,
            linearity_spec=0.01,
            travel_length=100.0,
            untrimmed_resistance=10000,
            trimmed_resistance=10050
        )

        # Run analysis with detailed profile
        engine = AnalyticsEngine(
            config=AnalysisConfig(profile=AnalysisProfile.DETAILED)
        )

        result = engine.analyze(input_data)

        # Verify all analyses ran
        self.assertIsNotNone(result.linearity)
        self.assertIsNotNone(result.resistance)
        self.assertIsNotNone(result.zones)
        self.assertIsNotNone(result.trim_effectiveness)
        self.assertIsNotNone(result.failure_probability)
        self.assertIsNotNone(result.dynamic_range)

        # Check trim effectiveness
        self.assertGreater(result.trim_effectiveness.improvement_percent, 0)

        # Get summary
        summary = result.get_summary()
        self.assertIn("overall_pass", summary)
        self.assertIn("risk_level", summary)


if __name__ == "__main__":
    unittest.main()