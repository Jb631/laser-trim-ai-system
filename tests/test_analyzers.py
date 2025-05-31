"""
Tests for individual analysis modules.

Tests each analyzer independently and verifies:
- Correct calculations
- Edge case handling
- Integration with data models
"""

import pytest
import numpy as np
from datetime import datetime

from laser_trim_analyzer.core.config import Config
from laser_trim_analyzer.core.models import (
    UnitProperties, SigmaAnalysis, LinearityAnalysis,
    ResistanceAnalysis, RiskCategory
)
from laser_trim_analyzer.analysis.sigma_analyzer import SigmaAnalyzer
from laser_trim_analyzer.analysis.linearity_analyzer import LinearityAnalyzer
from laser_trim_analyzer.analysis.resistance_analyzer import ResistanceAnalyzer


class TestSigmaAnalyzer:
    """Test suite for sigma gradient analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create sigma analyzer instance."""
        config = Config()
        return SigmaAnalyzer(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample position and error data."""
        positions = np.linspace(0, 100, 100)
        # Add some noise to make it realistic
        errors = 0.001 * positions + np.random.normal(0, 0.01, 100)
        return positions.tolist(), errors.tolist()

    @pytest.fixture
    def unit_props(self):
        """Create sample unit properties."""
        return UnitProperties(
            unit_length=300,
            untrimmed_resistance=10000,
            trimmed_resistance=10200
        )

    @pytest.mark.asyncio
    async def test_sigma_calculation(self, analyzer, sample_data, unit_props):
        """Test basic sigma gradient calculation."""
        positions, errors = sample_data

        result = await analyzer.analyze(
            positions=positions,
            errors=errors,
            unit_props=unit_props,
            model="8340"
        )

        assert isinstance(result, SigmaAnalysis)
        assert result.sigma_gradient > 0
        assert result.sigma_threshold > 0
        assert isinstance(result.sigma_pass, bool)
        assert result.gradient_margin == result.sigma_threshold - result.sigma_gradient
        assert result.scaling_factor == 24.0  # Default

    @pytest.mark.asyncio
    async def test_model_specific_thresholds(self, analyzer, sample_data, unit_props):
        """Test model-specific threshold calculations."""
        positions, errors = sample_data

        # Test 8340-1 special case
        result_8340_1 = await analyzer.analyze(
            positions=positions,
            errors=errors,
            unit_props=unit_props,
            model="8340-1"
        )
        assert result_8340_1.sigma_threshold == 0.4  # Fixed threshold

        # Test 8555 model
        result_8555 = await analyzer.analyze(
            positions=positions,
            errors=errors,
            unit_props=unit_props,
            model="8555"
        )
        # Should use different calculation
        assert result_8555.sigma_threshold != result_8340_1.sigma_threshold

    @pytest.mark.asyncio
    async def test_edge_cases(self, analyzer, unit_props):
        """Test edge cases in sigma analysis."""
        # Empty data
        result = await analyzer.analyze(
            positions=[],
            errors=[],
            unit_props=unit_props,
            model="8340"
        )
        assert result.sigma_gradient == 0
        assert not result.sigma_pass

        # Single point
        result = await analyzer.analyze(
            positions=[0],
            errors=[0],
            unit_props=unit_props,
            model="8340"
        )
        assert result.sigma_gradient == 0

        # Constant errors (no gradient)
        positions = list(range(100))
        errors = [0.01] * 100
        result = await analyzer.analyze(
            positions=positions,
            errors=errors,
            unit_props=unit_props,
            model="8340"
        )
        assert result.sigma_gradient < 0.001  # Should be near zero

    @pytest.mark.asyncio
    async def test_filtering_effect(self, analyzer, unit_props):
        """Test that filtering is applied correctly."""
        positions = list(range(100))

        # Create errors with a spike
        errors = [0.01] * 100
        errors[50] = 0.5  # Spike

        result = await analyzer.analyze(
            positions=positions,
            errors=errors,
            unit_props=unit_props,
            model="8340"
        )

        # Filtering should reduce the impact of the spike
        assert result.sigma_gradient < 0.1


class TestLinearityAnalyzer:
    """Test suite for linearity analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create linearity analyzer instance."""
        config = Config()
        return LinearityAnalyzer(config)

    @pytest.fixture
    def linear_data(self):
        """Create perfectly linear data."""
        positions = np.linspace(0, 100, 100)
        errors = np.zeros(100)  # Perfect linearity
        upper_limits = [0.05] * 100
        lower_limits = [-0.05] * 100
        return positions.tolist(), errors.tolist(), upper_limits, lower_limits

    @pytest.fixture
    def nonlinear_data(self):
        """Create non-linear data."""
        positions = np.linspace(0, 100, 100)
        errors = 0.001 * positions ** 2  # Quadratic error
        upper_limits = [0.05] * 100
        lower_limits = [-0.05] * 100
        return positions.tolist(), errors.tolist(), upper_limits, lower_limits

    @pytest.mark.asyncio
    async def test_perfect_linearity(self, analyzer, linear_data):
        """Test analysis of perfectly linear data."""
        positions, errors, upper, lower = linear_data

        result = await analyzer.analyze(
            positions=positions,
            errors=errors,
            upper_limits=upper,
            lower_limits=lower,
            spec=0.05
        )

        assert isinstance(result, LinearityAnalysis)
        assert result.linearity_pass
        assert result.linearity_fail_points == 0
        assert result.optimal_offset == 0  # No offset needed
        assert result.final_linearity_error_shifted < 0.001

    @pytest.mark.asyncio
    async def test_nonlinear_data(self, analyzer, nonlinear_data):
        """Test analysis of non-linear data."""
        positions, errors, upper, lower = nonlinear_data

        result = await analyzer.analyze(
            positions=positions,
            errors=errors,
            upper_limits=upper,
            lower_limits=lower,
            spec=0.05
        )

        assert not result.linearity_pass
        assert result.linearity_fail_points > 0
        assert result.max_deviation > 0
        assert result.max_deviation_position is not None

    @pytest.mark.asyncio
    async def test_optimal_offset_calculation(self, analyzer):
        """Test optimal offset calculation."""
        positions = list(range(100))
        # Errors with constant offset
        errors = [0.02] * 100  # All errors at +0.02
        upper_limits = [0.05] * 100
        lower_limits = [-0.05] * 100

        result = await analyzer.analyze(
            positions=positions,
            errors=errors,
            upper_limits=upper,
            lower_limits=lower,
            spec=0.05
        )

        # Optimal offset should center the errors
        assert abs(result.optimal_offset + 0.02) < 0.001
        assert result.linearity_pass  # Should pass after offset

    @pytest.mark.asyncio
    async def test_missing_limits(self, analyzer):
        """Test handling of missing limit data."""
        positions = list(range(10))
        errors = [0.01] * 10
        upper_limits = [None, 0.05, 0.05, None, 0.05, None, 0.05, 0.05, None, 0.05]
        lower_limits = [None, -0.05, -0.05, None, -0.05, None, -0.05, -0.05, None, -0.05]

        result = await analyzer.analyze(
            positions=positions,
            errors=errors,
            upper_limits=upper_limits,
            lower_limits=lower_limits,
            spec=0.05
        )

        # Should still calculate with available limits
        assert result is not None
        assert result.linearity_spec == 0.05


class TestResistanceAnalyzer:
    """Test suite for resistance analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create resistance analyzer instance."""
        config = Config()
        return ResistanceAnalyzer(config)

    @pytest.mark.asyncio
    async def test_resistance_calculation(self, analyzer):
        """Test basic resistance analysis."""
        unit_props = UnitProperties(
            unit_length=300,
            untrimmed_resistance=10000,
            trimmed_resistance=10250
        )

        result = await analyzer.analyze(unit_props)

        assert isinstance(result, ResistanceAnalysis)
        assert result.untrimmed_resistance == 10000
        assert result.trimmed_resistance == 10250
        assert result.resistance_change == 250
        assert result.resistance_change_percent == 2.5

    @pytest.mark.asyncio
    async def test_missing_resistance_data(self, analyzer):
        """Test handling of missing resistance data."""
        unit_props = UnitProperties(unit_length=300)

        result = await analyzer.analyze(unit_props)

        assert result.untrimmed_resistance is None
        assert result.trimmed_resistance is None
        assert result.resistance_change is None
        assert result.resistance_change_percent is None

    @pytest.mark.asyncio
    async def test_negative_resistance_change(self, analyzer):
        """Test negative resistance change."""
        unit_props = UnitProperties(
            unit_length=300,
            untrimmed_resistance=10000,
            trimmed_resistance=9800  # Decreased
        )

        result = await analyzer.analyze(unit_props)

        assert result.resistance_change == -200
        assert result.resistance_change_percent == -2.0

    @pytest.mark.asyncio
    async def test_zero_untrimmed_resistance(self, analyzer):
        """Test edge case of zero untrimmed resistance."""
        unit_props = UnitProperties(
            unit_length=300,
            untrimmed_resistance=0,
            trimmed_resistance=100
        )

        result = await analyzer.analyze(unit_props)

        # Should handle division by zero gracefully
        assert result.resistance_change == 100
        assert result.resistance_change_percent is None or np.isinf(result.resistance_change_percent)


class TestAnalyzerIntegration:
    """Test integration between different analyzers."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()

    @pytest.fixture
    def analyzers(self, config):
        """Create all analyzers."""
        return {
            'sigma': SigmaAnalyzer(config),
            'linearity': LinearityAnalyzer(config),
            'resistance': ResistanceAnalyzer(config)
        }

    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self, analyzers):
        """Test complete analysis pipeline."""
        # Generate test data
        positions = np.linspace(0, 100, 100)
        errors = np.random.normal(0, 0.01, 100)
        upper_limits = [0.05] * 100
        lower_limits = [-0.05] * 100

        unit_props = UnitProperties(
            unit_length=300,
            untrimmed_resistance=10000,
            trimmed_resistance=10150
        )

        # Run sigma analysis
        sigma_result = await analyzers['sigma'].analyze(
            positions=positions.tolist(),
            errors=errors.tolist(),
            unit_props=unit_props,
            model="8340"
        )

        # Run linearity analysis using sigma threshold as spec
        linearity_result = await analyzers['linearity'].analyze(
            positions=positions.tolist(),
            errors=errors.tolist(),
            upper_limits=upper_limits,
            lower_limits=lower_limits,
            spec=sigma_result.sigma_threshold
        )

        # Run resistance analysis
        resistance_result = await analyzers['resistance'].analyze(unit_props)

        # Verify all results are valid
        assert sigma_result is not None
        assert linearity_result is not None
        assert resistance_result is not None

        # Verify interdependencies
        assert linearity_result.linearity_spec == sigma_result.sigma_threshold
        assert resistance_result.resistance_change_percent == 1.5

    @pytest.mark.asyncio
    async def test_risk_assessment_integration(self, analyzers):
        """Test risk assessment based on multiple analyses."""
        # Create high-risk data
        positions = np.linspace(0, 100, 100)
        errors = np.random.normal(0, 0.05, 100)  # High errors
        upper_limits = [0.05] * 100
        lower_limits = [-0.05] * 100

        unit_props = UnitProperties(
            unit_length=300,
            untrimmed_resistance=10000,
            trimmed_resistance=11000  # 10% change
        )

        # Run analyses
        sigma_result = await analyzers['sigma'].analyze(
            positions=positions.tolist(),
            errors=errors.tolist(),
            unit_props=unit_props,
            model="8340"
        )

        linearity_result = await analyzers['linearity'].analyze(
            positions=positions.tolist(),
            errors=errors.tolist(),
            upper_limits=upper_limits,
            lower_limits=lower_limits,
            spec=0.05
        )

        resistance_result = await analyzers['resistance'].analyze(unit_props)

        # Calculate risk factors (simplified version)
        sigma_risk = not sigma_result.sigma_pass
        linearity_risk = not linearity_result.linearity_pass
        resistance_risk = abs(resistance_result.resistance_change_percent) > 5

        # Determine overall risk
        risk_count = sum([sigma_risk, linearity_risk, resistance_risk])

        if risk_count >= 2:
            risk_category = RiskCategory.HIGH
        elif risk_count == 1:
            risk_category = RiskCategory.MEDIUM
        else:
            risk_category = RiskCategory.LOW

        # In this case, should be high risk
        assert risk_category == RiskCategory.HIGH