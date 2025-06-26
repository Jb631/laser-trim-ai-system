#!/usr/bin/env python3
"""
Verify sigma gradient calculation correctness.

This script tests the sigma gradient calculation with known data
and compares against expected results.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.laser_trim_analyzer.analysis.sigma_analyzer import SigmaAnalyzer
from src.laser_trim_analyzer.core.config import get_config

def create_test_data(pattern="linear"):
    """Create test data with known characteristics."""
    if pattern == "linear":
        # Linear data - should have very low sigma gradient
        positions = np.linspace(0, 100, 101)
        errors = 0.5 * positions + 1.0  # Perfect linear relationship
        expected_sigma = 0.0  # No variation in gradient
        
    elif pattern == "noisy_linear":
        # Linear with noise - moderate sigma gradient
        positions = np.linspace(0, 100, 101)
        base_errors = 0.5 * positions + 1.0
        noise = np.random.normal(0, 0.1, len(positions))
        errors = base_errors + noise
        # Expected sigma depends on noise level and step size
        expected_sigma = 0.001  # Approximate
        
    elif pattern == "sinusoidal":
        # Sinusoidal - varying gradient
        positions = np.linspace(0, 100, 101)
        errors = 5 * np.sin(positions * 0.1) + 50
        # Gradient will vary between -0.5 and 0.5
        expected_sigma = 0.35  # Approximate std dev of sine derivative
        
    elif pattern == "step":
        # Step function - high sigma at transition
        positions = np.linspace(0, 100, 101)
        errors = np.where(positions < 50, 10, 20)
        expected_sigma = 10.0  # Large due to step
        
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return positions.tolist(), errors.tolist(), expected_sigma

def verify_calculation():
    """Verify sigma gradient calculation with test patterns."""
    config = get_config()
    analyzer = SigmaAnalyzer(config)
    
    print("Sigma Gradient Calculation Verification")
    print("=" * 50)
    
    test_patterns = ["linear", "noisy_linear", "sinusoidal", "step"]
    
    for pattern in test_patterns:
        print(f"\nTest Pattern: {pattern}")
        print("-" * 30)
        
        positions, errors, expected_sigma = create_test_data(pattern)
        
        # Prepare data for analyzer
        data = {
            'positions': positions,
            'errors': errors,
            'upper_limits': [e + 5 for e in errors],  # Simple limits
            'lower_limits': [e - 5 for e in errors],
            'model': 'TEST_MODEL',
            'unit_length': 100.0,
            'travel_length': 100.0
        }
        
        # Run analysis
        result = analyzer.analyze(data)
        calculated_sigma = result.sigma_gradient
        
        print(f"Expected sigma (approx): {expected_sigma:.6f}")
        print(f"Calculated sigma: {calculated_sigma:.6f}")
        
        # For linear pattern, we expect very low sigma
        if pattern == "linear":
            if calculated_sigma < 0.0001:
                print("✓ PASS: Linear pattern has near-zero sigma gradient")
            else:
                print("✗ FAIL: Linear pattern should have near-zero sigma gradient")
        
        # For other patterns, check if in reasonable range
        elif pattern == "noisy_linear":
            if 0.0001 < calculated_sigma < 0.01:
                print("✓ PASS: Noisy linear pattern has small sigma gradient")
            else:
                print("✗ FAIL: Noisy linear pattern sigma out of expected range")
                
        elif pattern == "sinusoidal":
            if 0.1 < calculated_sigma < 1.0:
                print("✓ PASS: Sinusoidal pattern has moderate sigma gradient")
            else:
                print("✗ FAIL: Sinusoidal pattern sigma out of expected range")
                
        elif pattern == "step":
            if calculated_sigma > 1.0:
                print("✓ PASS: Step pattern has high sigma gradient")
            else:
                print("✗ FAIL: Step pattern should have high sigma gradient")

def manual_calculation_example():
    """Show manual calculation for verification."""
    print("\n\nManual Calculation Example")
    print("=" * 50)
    
    # Simple example data
    positions = [0, 1, 2, 3, 4, 5]
    errors = [1.0, 1.5, 2.1, 2.9, 3.6, 4.5]
    step_size = 1  # Default MATLAB_GRADIENT_STEP
    
    print(f"Positions: {positions}")
    print(f"Errors: {errors}")
    print(f"Step size: {step_size}")
    
    # Calculate gradients manually
    gradients = []
    for i in range(len(positions) - step_size):
        dx = positions[i + step_size] - positions[i]
        dy = errors[i + step_size] - errors[i]
        if abs(dx) > 1e-6:
            gradient = dy / dx
            gradients.append(gradient)
            print(f"  i={i}: dx={dx:.3f}, dy={dy:.3f}, gradient={gradient:.3f}")
    
    # Calculate standard deviation
    if len(gradients) > 1:
        mean = sum(gradients) / len(gradients)
        variance = sum((g - mean)**2 for g in gradients) / (len(gradients) - 1)
        std_dev = np.sqrt(variance)
        
        print(f"\nGradients: {[f'{g:.3f}' for g in gradients]}")
        print(f"Mean: {mean:.6f}")
        print(f"Variance: {variance:.6f}")
        print(f"Sigma (Std Dev): {std_dev:.6f}")
        
        # Verify with numpy
        numpy_std = np.std(gradients, ddof=1)
        print(f"NumPy verification: {numpy_std:.6f}")
        print(f"Match: {'✓' if abs(std_dev - numpy_std) < 1e-10 else '✗'}")

if __name__ == "__main__":
    verify_calculation()
    manual_calculation_example()