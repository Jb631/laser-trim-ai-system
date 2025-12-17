# Laser Trim Analyzer - Test Suite

Automated test suite for critical calculations and functionality.

## Test Coverage

### 1. `test_calculations.py` - Critical Calculations
- **CPK/Ppk Calculations** - Process capability indices
  - Perfect centering scenarios
  - Shifted process scenarios
  - Zero standard deviation handling
  - Formula correctness verification
  - Out-of-spec scenarios

- **Control Limits** - Statistical Process Control
  - 3-sigma control limits
  - Individuals chart moving range
  - Control limits vs specification limits distinction

- **Sigma Calculations** - Risk thresholds
  - Sigma scaling factor (24.0 - MATLAB spec)
  - Risk categorization (LOW <0.3, MEDIUM 0.3-0.7, HIGH >0.7)
  - Sigma gradient range validation

- **Numerical Stability**
  - Division by zero protection
  - NaN handling
  - Infinity handling
  - Empty data handling
  - Minimum data point requirements

### 2. `test_r_value.py` - Regression Test for Bug Fix
**Bug Fixed**: R-value calculation crash when R² is negative (historical_page.py:2315-2318)

Tests:
- Poor model fit scenarios (original bug trigger)
- Perfect linear fit
- No correlation scenarios
- Explicit negative R² handling
- Zero total variance
- Minimum data points
- Clamping bounds verification

### 3. `test_data_validation.py` - Edge Cases
Tests the multi-layer validation system:

- **NaN Handling**
  - NaN in numeric fields
  - NaN in DataFrames
  - All-NaN columns

- **Infinity Handling**
  - Positive/negative infinity detection
  - Finite value filtering

- **Empty Data**
  - Empty arrays
  - Empty DataFrames
  - Empty lists
  - Empty columns

- **NULL/None Values**
  - None vs NaN distinction
  - DataFrame None handling

- **Boundary Conditions**
  - Zero values
  - Negative values
  - Extreme values (very large/small)
  - Mixed positive/negative

- **Minimum Data Points**
  - Single point
  - Two points
  - Insufficient data for histograms

- **String Validation**
  - Empty strings
  - Whitespace-only
  - SQL keyword detection
  - Path traversal detection
  - Null byte detection

- **Type Safety**
  - Type conversions
  - Type checking

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test File
```bash
pytest tests/test_calculations.py
pytest tests/test_r_value.py
pytest tests/test_data_validation.py
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run with Coverage Report
```bash
pytest tests/ --cov=src/laser_trim_analyzer --cov-report=html
```

### Run Specific Test Class
```bash
pytest tests/test_calculations.py::TestCPKCalculations
pytest tests/test_r_value.py::TestRValueCalculation
```

### Run Specific Test Method
```bash
pytest tests/test_calculations.py::TestCPKCalculations::test_cpk_perfect_centering
pytest tests/test_r_value.py::TestRValueCalculation::test_r_value_poor_model_fit
```

## Test Fixtures (conftest.py)

- `sample_data` - 100 random sigma gradient values (normal distribution)
- `sample_dataframe` - DataFrame with sigma_gradient, linearity_error, model_number, trim_date
- `edge_case_data` - Dictionary of edge cases (empty, nan, inf, zeros, etc.)
- `spec_limits` - Standard specification limits (LSL=0.3, USL=0.7, target=0.5)

## Requirements

Tests require:
- pytest
- numpy
- pandas
- matplotlib (for date conversion in R-value tests)

Install with:
```bash
pip install pytest pytest-cov
```

## Integration with CI/CD

Add to GitHub Actions or similar:
```yaml
- name: Run Tests
  run: pytest tests/ -v --cov=src/laser_trim_analyzer
```

## Test Development Guidelines

1. **Test Names** - Use descriptive names starting with `test_`
2. **Assertions** - Include helpful failure messages
3. **Edge Cases** - Always test boundary conditions
4. **Fixtures** - Use fixtures for common test data
5. **Documentation** - Document what bug/behavior is being tested

## Coverage Goals

- **Critical Calculations**: 100% (CPK, sigma, control limits)
- **Bug Fixes**: 100% regression coverage
- **Data Validation**: Edge cases covered
- **Overall Target**: 80%+ coverage of core analysis modules

## Future Test Additions

Potential areas for expansion:
1. Database operations (CRUD, transactions)
2. ML predictions (when models are trained)
3. Chart rendering (snapshot testing)
4. Excel parsing (various formats)
5. Integration tests (end-to-end workflows)

## Notes

- Tests are **fast** - no database or GUI dependencies
- Tests are **isolated** - use fixtures, no shared state
- Tests are **reproducible** - use np.random.seed for deterministic results
- Tests **prevent regressions** - especially for fixed bugs
