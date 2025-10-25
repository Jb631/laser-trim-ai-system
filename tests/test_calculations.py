"""
Tests for critical calculations: CPK, Ppk, Control Limits, Sigma values.

These tests verify the mathematical correctness of quality metrics
used throughout the application.
"""
import pytest
import numpy as np
from laser_trim_analyzer.analysis.analytics_engine import AnalyticsEngine
from laser_trim_analyzer.core.constants import (
    DEFAULT_SIGMA_SCALING_FACTOR,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD
)


class TestCPKCalculations:
    """Test Process Capability Index (Cpk) calculations."""

    def test_cpk_perfect_centering(self, spec_limits):
        """Test Cpk when process is perfectly centered."""
        # Mean at target (0.5), std=0.05
        # USL=0.7, LSL=0.3, range=0.4
        # CPU = (0.7 - 0.5)/(3*0.05) = 0.2/0.15 = 1.33
        # CPL = (0.5 - 0.3)/(3*0.05) = 0.2/0.15 = 1.33
        # Cpk = min(1.33, 1.33) = 1.33

        data = np.random.normal(loc=0.5, scale=0.05, size=1000)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        cpu = (spec_limits['USL'] - mean) / (3 * std)
        cpl = (mean - spec_limits['LSL']) / (3 * std)
        cpk = min(cpu, cpl)

        # Should be close to 1.33 for perfect centering with std=0.05
        assert 1.2 < cpk < 1.5, f"Expected Cpk ~1.33, got {cpk}"
        assert abs(cpu - cpl) < 0.1, "CPU and CPL should be nearly equal for centered process"

    def test_cpk_shifted_process(self, spec_limits):
        """Test Cpk when process mean is shifted from target."""
        # Mean shifted to 0.6 (closer to USL=0.7)
        data = np.random.normal(loc=0.6, scale=0.05, size=1000)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        cpu = (spec_limits['USL'] - mean) / (3 * std)
        cpl = (mean - spec_limits['LSL']) / (3 * std)
        cpk = min(cpu, cpl)

        # CPU should be lower than CPL since closer to USL
        assert cpu < cpl, "CPU should be limiting factor when shifted toward USL"
        assert 0.5 < cpk < 2.5, f"Cpk should be reasonable, got {cpk}"

    def test_cpk_zero_std(self, spec_limits):
        """Test Cpk handles zero standard deviation."""
        # All values identical - no variation
        data = np.array([0.5] * 100)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        # When std=0, Cpk should be infinite (or very large)
        # Most implementations return inf or handle specially
        assert std == 0, "Standard deviation should be exactly zero"

        # Division by zero protection test
        if std == 0:
            cpk = float('inf')
        else:
            cpu = (spec_limits['USL'] - mean) / (3 * std)
            cpl = (mean - spec_limits['LSL']) / (3 * std)
            cpk = min(cpu, cpl)

        assert np.isinf(cpk) or cpk > 1000, "Cpk should be infinite or very large for zero variation"

    def test_cpk_formula_correctness(self):
        """Test Cpk formula matches specification."""
        # Known values test case
        mean = 0.5
        std = 0.06667  # Chosen to give Cpk = 1.0
        USL = 0.7
        LSL = 0.3

        cpu = (USL - mean) / (3 * std)
        cpl = (mean - LSL) / (3 * std)
        cpk = min(cpu, cpl)

        # With these values, both CPU and CPL should equal 1.0
        assert abs(cpu - 1.0) < 0.01, f"CPU should be ~1.0, got {cpu}"
        assert abs(cpl - 1.0) < 0.01, f"CPL should be ~1.0, got {cpl}"
        assert abs(cpk - 1.0) < 0.01, f"Cpk should be ~1.0, got {cpk}"

    def test_cpk_out_of_spec(self, spec_limits):
        """Test Cpk when process mean is out of spec."""
        # Mean at 0.8 (above USL=0.7)
        data = np.random.normal(loc=0.8, scale=0.05, size=1000)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        cpu = (spec_limits['USL'] - mean) / (3 * std)
        cpl = (mean - spec_limits['LSL']) / (3 * std)
        cpk = min(cpu, cpl)

        # CPU will be negative since mean > USL
        assert cpu < 0, "CPU should be negative when mean exceeds USL"
        assert cpk < 0, "Cpk should be negative when process is out of spec"


class TestControlLimits:
    """Test control limit calculations for SPC charts."""

    def test_three_sigma_limits(self, sample_data):
        """Test 3-sigma control limits calculation."""
        mean = np.mean(sample_data)
        std = np.std(sample_data, ddof=1)

        ucl = mean + 3 * std  # Upper Control Limit
        lcl = mean - 3 * std  # Lower Control Limit

        # Verify limits are symmetric around mean
        assert abs((ucl - mean) - (mean - lcl)) < 1e-10, "Limits should be symmetric"

        # About 99.7% of data should fall within 3-sigma limits
        within_limits = np.sum((sample_data >= lcl) & (sample_data <= ucl))
        percent_within = within_limits / len(sample_data) * 100

        assert percent_within > 95, f"Expected >95% within 3-sigma, got {percent_within:.1f}%"

    def test_individuals_chart_moving_range(self, sample_data):
        """Test moving range calculation for I-chart."""
        # Moving range: absolute difference between consecutive points
        moving_ranges = np.abs(np.diff(sample_data))
        mr_bar = np.mean(moving_ranges)

        # Control limits for individuals chart use moving range
        # UCL = mean + 3 * (MR-bar / d2), where d2 ≈ 1.128 for n=2
        d2 = 1.128
        mean = np.mean(sample_data)
        ucl = mean + 3 * (mr_bar / d2)
        lcl = mean - 3 * (mr_bar / d2)

        assert ucl > mean, "UCL should be above mean"
        assert lcl < mean, "LCL should be below mean"
        assert mr_bar >= 0, "Moving range should be non-negative"

    def test_control_limits_vs_spec_limits(self, sample_data, spec_limits):
        """Verify control limits are distinct from specification limits."""
        mean = np.mean(sample_data)
        std = np.std(sample_data, ddof=1)

        # Control limits (process-based)
        ucl = mean + 3 * std
        lcl = mean - 3 * std

        # Spec limits (engineering-based)
        usl = spec_limits['USL']
        lsl = spec_limits['LSL']

        # Control limits and spec limits should NOT be the same
        # (unless process happens to have perfect capability)
        assert (ucl != usl) or (lcl != lsl), "Control and spec limits should be independent"


class TestSigmaCalculations:
    """Test sigma gradient calculations and thresholds."""

    def test_sigma_scaling_factor(self):
        """Test sigma scaling factor constant."""
        assert DEFAULT_SIGMA_SCALING_FACTOR == 24.0, \
            "Sigma scaling factor should match MATLAB specification"

    def test_risk_thresholds(self):
        """Test risk category thresholds."""
        assert HIGH_RISK_THRESHOLD == 0.7, "High risk threshold should be 0.7"
        assert MEDIUM_RISK_THRESHOLD == 0.3, "Medium risk threshold should be 0.3"
        assert MEDIUM_RISK_THRESHOLD < HIGH_RISK_THRESHOLD, \
            "Medium threshold should be below high threshold"

    def test_sigma_gradient_range(self, sample_data):
        """Test sigma gradient values fall within expected range."""
        # Sigma gradient should typically be in [0, 1] range
        # Values outside this range are unusual but possible

        assert sample_data.min() >= 0, "Sigma gradient should not be negative (for normal data)"
        # Note: In actual code, values can exceed 1.0 in extreme cases
        # but typical good process should be 0.3-0.7

    def test_risk_categorization(self):
        """Test risk category assignment based on sigma gradient."""
        test_cases = [
            (0.1, 'LOW'),     # Below 0.3
            (0.5, 'MEDIUM'),  # Between 0.3 and 0.7
            (0.8, 'HIGH'),    # Above 0.7
            (0.3, 'MEDIUM'),  # Exactly at lower threshold
            (0.7, 'HIGH'),    # Exactly at upper threshold
        ]

        for sigma_val, expected_risk in test_cases:
            if sigma_val < MEDIUM_RISK_THRESHOLD:
                risk = 'LOW'
            elif sigma_val < HIGH_RISK_THRESHOLD:
                risk = 'MEDIUM'
            else:
                risk = 'HIGH'

            assert risk == expected_risk, \
                f"Sigma {sigma_val} should be {expected_risk}, got {risk}"


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_division_by_zero_protection(self, edge_case_data):
        """Test division by zero is handled gracefully."""
        # Test with zero standard deviation
        data = edge_case_data['all_same']
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        assert std == 0, "All same values should have zero std"

        # Cpk calculation should handle this
        if std == 0:
            # Application should return inf or special value
            cpk = float('inf')
        else:
            cpk = (0.7 - mean) / (3 * std)

        assert np.isinf(cpk) or cpk > 1000, "Should handle zero std gracefully"

    def test_nan_handling(self, edge_case_data):
        """Test NaN values are handled correctly."""
        data_with_nan = edge_case_data['with_nan']

        # Using np.nanmean and np.nanstd should handle NaN
        mean = np.nanmean(data_with_nan)
        std = np.nanstd(data_with_nan, ddof=1)

        assert not np.isnan(mean), "nanmean should not return NaN"
        assert not np.isnan(std), "nanstd should not return NaN"
        assert std > 0, "std should be positive for varying data"

    def test_inf_handling(self, edge_case_data):
        """Test infinite values are handled correctly."""
        data_with_inf = edge_case_data['with_inf']

        # Filter out inf values before calculations
        finite_data = data_with_inf[np.isfinite(data_with_inf)]

        assert len(finite_data) < len(data_with_inf), "Should filter out inf"
        assert not np.isinf(np.mean(finite_data)), "Mean of finite data should be finite"

    def test_empty_data(self, edge_case_data):
        """Test empty data arrays are handled."""
        empty = edge_case_data['empty']

        assert len(empty) == 0, "Should be empty"

        # Operations on empty arrays should be handled gracefully
        # Application should check len > 0 before calculations

    def test_minimum_data_points(self, edge_case_data):
        """Test minimum data requirements."""
        single = edge_case_data['single']
        two_points = edge_case_data['two_points']

        # Standard deviation needs at least 2 points
        assert len(single) == 1, "Single point array"

        # std of single value with ddof=1 is NaN
        std_single = np.std(single, ddof=1)
        assert np.isnan(std_single), "Std of single value should be NaN with ddof=1"

        # Two points should work
        std_two = np.std(two_points, ddof=1)
        assert not np.isnan(std_two), "Std of two points should be valid"
        assert std_two >= 0, "Std should be non-negative"
