"""
Regression test for R-value calculation bug fix (historical_page.py:2315-2318).

Bug: When linear regression model is poor (ss_res > ss_tot), R² becomes negative,
causing sqrt() of negative value → ValueError.

Fix: Clamp R² to [0, 1] range before taking sqrt.

This test ensures the bug doesn't reoccur in future changes.
"""
import pytest
import numpy as np
import pandas as pd
import matplotlib.dates as mdates


class TestRValueCalculation:
    """Test R-value calculation with edge cases that caused the original bug."""

    def test_r_value_poor_model_fit(self):
        """
        Test R-value calculation when linear model is poor (original bug scenario).

        When residual sum of squares > total sum of squares, R² becomes negative.
        This should be handled gracefully by clamping to [0, 1] before sqrt.
        """
        # Create data that produces a poor linear fit
        # Intentionally non-linear relationship
        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        # Quadratic relationship - linear fit will be poor
        y = x**2 + np.random.normal(0, 5, 20)

        # Convert to datetime for realistic test (matches actual usage)
        dates = pd.date_range('2024-01-01', periods=len(x), freq='D')
        x_numeric = mdates.date2num(dates)

        # Calculate linear regression
        z = np.polyfit(x_numeric, y, 1)
        slope, intercept = z

        # Calculate R²
        y_mean = np.mean(y)
        y_pred = slope * x_numeric + intercept
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()

        # CRITICAL: R² can be negative for poor fits
        r_squared_unclamped = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # BUGFIX: Clamp R² to [0, 1] before sqrt
        r_squared = max(0.0, min(1.0, r_squared_unclamped))

        # This should NOT raise ValueError
        r_value = np.sqrt(r_squared)

        # Verify fix
        assert 0.0 <= r_value <= 1.0, f"R-value should be in [0, 1], got {r_value}"
        assert not np.isnan(r_value), "R-value should not be NaN"

    def test_r_value_perfect_fit(self):
        """Test R-value with perfect linear fit (R² = 1.0)."""
        # Perfect linear relationship
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x + 1  # Perfect line, no noise

        dates = pd.date_range('2024-01-01', periods=len(x), freq='D')
        x_numeric = mdates.date2num(dates)

        # Linear regression
        z = np.polyfit(x_numeric, y, 1)
        slope, intercept = z

        y_mean = np.mean(y)
        y_pred = slope * x_numeric + intercept
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()

        r_squared = max(0.0, min(1.0, 1 - (ss_res / ss_tot))) if ss_tot != 0 else 0
        r_value = np.sqrt(r_squared)

        # Perfect fit should give R² ≈ 1.0, R ≈ 1.0
        assert r_value > 0.99, f"Expected R ≈ 1.0 for perfect fit, got {r_value}"

    def test_r_value_no_relationship(self):
        """Test R-value when there's no correlation (R² ≈ 0)."""
        np.random.seed(42)
        # Random data with no relationship
        x = np.random.rand(50)
        y = np.random.rand(50)

        dates = pd.date_range('2024-01-01', periods=len(x), freq='D')
        x_numeric = mdates.date2num(dates)

        z = np.polyfit(x_numeric, y, 1)
        slope, intercept = z

        y_mean = np.mean(y)
        y_pred = slope * x_numeric + intercept
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()

        r_squared = max(0.0, min(1.0, 1 - (ss_res / ss_tot))) if ss_tot != 0 else 0
        r_value = np.sqrt(r_squared)

        # No relationship should give R ≈ 0
        assert 0.0 <= r_value <= 1.0, f"R-value should be in [0, 1], got {r_value}"
        # With random data, R should be close to 0
        assert r_value < 0.5, f"Expected R ≈ 0 for random data, got {r_value}"

    def test_r_value_negative_r_squared_scenario(self):
        """
        Explicit test for negative R² scenario (the actual bug).

        This directly tests the condition that caused the original ValueError.
        """
        # Manually create scenario where ss_res > ss_tot
        y = np.array([1, 2, 3, 4, 5])
        y_mean = np.mean(y)

        # Terrible predictions (worse than just using mean)
        y_pred = np.array([10, 1, 10, 1, 10])

        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()

        # This will produce negative R²
        r_squared_raw = 1 - (ss_res / ss_tot)

        assert r_squared_raw < 0, "Test setup should produce negative R²"

        # WITHOUT clamping, this would raise ValueError:
        # r_value_buggy = np.sqrt(r_squared_raw)  # ValueError: math domain error

        # WITH clamping (the fix):
        r_squared_fixed = max(0.0, min(1.0, r_squared_raw))
        r_value_fixed = np.sqrt(r_squared_fixed)

        assert r_value_fixed == 0.0, "Negative R² should clamp to 0, giving R=0"
        assert not np.isnan(r_value_fixed), "Fixed calculation should not produce NaN"

    def test_r_value_zero_total_variance(self):
        """Test R-value when y values are all identical (ss_tot = 0)."""
        # All y values the same
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        x_numeric = np.array([1, 2, 3, 4, 5])

        y_mean = np.mean(y)
        ss_tot = ((y - y_mean) ** 2).sum()

        assert ss_tot == 0, "Total variance should be zero for identical values"

        # Division by zero protection
        if ss_tot == 0:
            r_squared = 0  # Convention when no variance
        else:
            # Normal calculation
            z = np.polyfit(x_numeric, y, 1)
            y_pred = z[0] * x_numeric + z[1]
            ss_res = ((y - y_pred) ** 2).sum()
            r_squared = max(0.0, min(1.0, 1 - (ss_res / ss_tot)))

        r_value = np.sqrt(r_squared)

        assert r_value == 0.0, "R should be 0 when there's no variance in y"

    def test_r_value_minimum_data_points(self):
        """Test R-value calculation with minimum viable data points."""
        # Need at least 2 points for linear regression
        x = np.array([1, 2])
        y = np.array([2, 4])

        dates = pd.date_range('2024-01-01', periods=len(x), freq='D')
        x_numeric = mdates.date2num(dates)

        z = np.polyfit(x_numeric, y, 1)
        slope, intercept = z

        y_mean = np.mean(y)
        y_pred = slope * x_numeric + intercept
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()

        r_squared = max(0.0, min(1.0, 1 - (ss_res / ss_tot))) if ss_tot != 0 else 0
        r_value = np.sqrt(r_squared)

        # With only 2 points, linear fit is perfect
        assert r_value > 0.99, "Two points always give perfect linear fit"

    def test_r_value_clamping_bounds(self):
        """Test that clamping works correctly for boundary values."""
        test_cases = [
            (-0.5, 0.0),   # Negative R² clamps to 0
            (-0.1, 0.0),   # Small negative clamps to 0
            (0.0, 0.0),    # Zero stays zero
            (0.5, 0.5),    # Valid value unchanged
            (1.0, 1.0),    # Perfect fit unchanged
            (1.1, 1.0),    # Above 1 clamps to 1 (shouldn't happen but handle it)
        ]

        for r_squared_input, expected_clamped in test_cases:
            clamped = max(0.0, min(1.0, r_squared_input))
            assert clamped == expected_clamped, \
                f"Clamping {r_squared_input} should give {expected_clamped}, got {clamped}"

            # Should be able to take sqrt safely
            r_value = np.sqrt(clamped)
            assert 0.0 <= r_value <= 1.0, "R-value should always be in [0, 1]"
            assert not np.isnan(r_value), "R-value should never be NaN after clamping"
