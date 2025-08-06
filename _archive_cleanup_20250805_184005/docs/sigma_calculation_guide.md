# Sigma Gradient Calculation Guide

## Overview

The sigma gradient is a key quality metric that measures the variability of error gradients across a potentiometer's travel. It indicates trim quality and stability.

## Calculation Method

The sigma gradient calculation follows these steps:

1. **Calculate Position Gradients**: For each position pair separated by a step size (default: 3), calculate:
   - `dx = positions[i + step] - positions[i]`
   - `dy = errors[i + step] - errors[i]`
   - `gradient = dy / dx` (if dx > 1e-6)

2. **Filter Invalid Gradients**: Skip gradients where:
   - Position difference (dx) is too small (< 1e-6)
   - Result is NaN or infinite

3. **Calculate Standard Deviation**: 
   - Sigma gradient = standard deviation of all valid gradients
   - Uses N-1 degrees of freedom (sample standard deviation)

## Expected Sigma Ranges by Model

### General Guidelines

- **Excellent Quality**: < 0.001
- **Good Quality**: 0.001 - 0.01
- **Acceptable**: 0.01 - 0.1
- **Poor**: > 0.1

### Model-Specific Thresholds

#### 8340-1 Series
- **Fixed Threshold**: 0.4
- **Expected Range**: 0.05 - 0.35
- **Note**: This model has higher tolerances due to its design

#### 8555 Series
- **Base Threshold**: 0.0015
- **Expected Range**: 0.0005 - 0.002
- **Note**: More stringent requirements for precision applications

#### Standard Models
- **Threshold Calculation**: `(linearity_spec / travel_length) * (scaling_factor * 0.5)`
- **Default Scaling Factor**: 24.0
- **Expected Range**: 0.001 - 0.05

## Validation Logging

When the application runs, it logs detailed validation information:

```
Sigma gradient calculation validation:
  - Input: 501 positions, 501 errors
  - Valid gradients: 498 (out of 498 possible)
  - Gradient statistics:
    * Mean: 0.001234
    * Median: 0.001156
    * Min: -0.002345
    * Max: 0.003456
    * Range: 0.005801
  - Calculated sigma (std dev): 0.001823
  - Sigma to range ratio: 0.314
```

## How to Verify Calculations

1. **Check Log Files**: Look for "Sigma gradient calculation validation" in the logs
2. **Run Verification Script**: 
   ```bash
   python scripts/verify_sigma_calculation.py
   ```
3. **Review Statistics**:
   - Valid gradient count should be > 50% of possible gradients
   - Sigma to range ratio should typically be < 0.5
   - Mean and median should be close for normal distributions

## Common Issues and Solutions

### Issue: All Sigmas Returning 0
- **Cause**: Position difference threshold too strict
- **Solution**: Threshold changed from 1e-10 to 1e-6

### Issue: Unusually High Sigma Values
- **Causes**: 
  - Data contains outliers or noise spikes
  - Step changes in error values
- **Solution**: Review raw data for anomalies

### Issue: Too Few Valid Gradients
- **Causes**:
  - Insufficient position variation
  - Data points too close together
- **Solution**: Check data collection parameters

## Manual Verification Example

For positions [0, 1, 2, 3, 4, 5] and errors [1.0, 1.5, 2.1, 2.9, 3.6, 4.5]:

1. Calculate gradients (step=1):
   - i=0: gradient = (1.5-1.0)/(1-0) = 0.5
   - i=1: gradient = (2.1-1.5)/(2-1) = 0.6
   - i=2: gradient = (2.9-2.1)/(3-2) = 0.8
   - i=3: gradient = (3.6-2.9)/(4-3) = 0.7
   - i=4: gradient = (4.5-3.6)/(5-4) = 0.9

2. Calculate mean: (0.5+0.6+0.8+0.7+0.9)/5 = 0.7

3. Calculate variance: Σ(gradient - mean)² / (n-1) = 0.025

4. Sigma = √variance = 0.158