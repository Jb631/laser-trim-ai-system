# Migration Guide: Unified Analytics Engine

## Overview

This guide helps you migrate from the scattered analyzer modules to the new unified `AnalyticsEngine`. The new system provides:

- ✅ Structured results (dataclasses instead of dicts)
- ✅ Clean, consistent interface
- ✅ Configuration profiles
- ✅ Plugin support
- ✅ Better error handling
- ✅ Comprehensive testing

## Quick Comparison

### Old Way (Scattered Analyzers)
```python
from laser_trim_analyzer.analytics.linearity_analyzer import analyze_linearity_deviation
from laser_trim_analyzer.analytics.failure_analyzer import calculate_failure_probability

# Each function returns a dict
linearity_result = analyze_linearity_deviation(position, error, logger)
failure_result = calculate_failure_probability(sigma_grad, lin_spec, travel, threshold, logger)

# Access results
max_dev = linearity_result.get("max_deviation")  # Could be None
risk = failure_result.get("risk_category", "Unknown")
```

### New Way (Unified Engine)
```python
from analytics_engine import AnalyticsEngine, AnalysisInput, AnalysisConfig, AnalysisProfile

# Create engine once
engine = AnalyticsEngine(config=AnalysisConfig(profile=AnalysisProfile.STANDARD))

# Create input data
input_data = AnalysisInput(
    position=position,
    error=error,
    sigma_gradient=sigma_grad,
    linearity_spec=lin_spec
)

# Run all analyses
results = engine.analyze(input_data)

# Access typed results
max_dev = results.linearity.max_deviation  # Type-safe access
risk = results.failure_probability.risk_category  # Always has a value
```

## Migration Steps

### 1. Install the New Module

Place the `analytics_engine.py` file in your project:

```bash
laser_trim_analyzer/
├── analytics/
│   └── analytics_engine.py  # New unified engine
├── analytics_legacy/        # Move old analyzers here
│   ├── linearity_analyzer.py
│   ├── failure_analyzer.py
│   └── ...
```

### 2. Update Your Imports

Replace scattered imports with the unified engine:

```python
# Old imports (remove these)
# from laser_trim_analyzer.analytics.linearity_analyzer import analyze_linearity_deviation
# from laser_trim_analyzer.analytics.failure_analyzer import calculate_failure_probability
# from laser_trim_analyzer.analytics.zone_analyzer import analyze_travel_zones

# New import
from laser_trim_analyzer.analytics.analytics_engine import (
    AnalyticsEngine, 
    AnalysisInput, 
    AnalysisConfig,
    AnalysisProfile
)
```

### 3. Update Data Preparation

Convert your data preparation to use `AnalysisInput`:

```python
# Old way - passing individual parameters
result = analyze_linearity_deviation(position_list, error_list, logger)

# New way - structured input
input_data = AnalysisInput(
    position=position_list,
    error=error_list,
    upper_limit=upper_limits,
    lower_limit=lower_limits,
    sigma_gradient=calculated_sigma,
    sigma_threshold=calculated_threshold,
    linearity_spec=spec_value,
    untrimmed_resistance=untrimmed_res,
    trimmed_resistance=trimmed_res
)
```

### 4. Update Analysis Calls

Replace individual function calls with engine methods:

```python
# Old way - multiple function calls
linearity_result = analyze_linearity_deviation(pos, err, logger)
zone_result = analyze_travel_zones(pos, err, 5, logger)
failure_result = calculate_failure_probability(sigma, spec, travel, threshold, logger)

# New way - single engine call
engine = AnalyticsEngine()
results = engine.analyze(input_data)

# Or run specific analysis
linearity_result = engine.analyze_single("linearity", input_data)
```

### 5. Update Result Access

Change from dict access to typed attributes:

```python
# Old way - dict access
if linearity_result:
    max_dev = linearity_result.get("max_deviation", 0)
    if max_dev is not None:
        print(f"Max deviation: {max_dev}")

# New way - typed access
if results.linearity:
    print(f"Max deviation: {results.linearity.max_deviation}")
    print(f"Linearity pass: {results.linearity.linearity_pass}")
```

### 6. Update Your Processor Module

Here's how to update the `processor_module.py`:

```python
# In your processor_module.py

class DataDrivenLaserProcessor:
    def __init__(self, output_dir: str, **kwargs):
        # ... existing init code ...
        
        # Initialize analytics engine
        self.analytics_engine = AnalyticsEngine(
            config=AnalysisConfig(profile=AnalysisProfile.STANDARD),
            logger=self.logger
        )
    
    def _run_advanced_analytics(self, track_result_dict, untrimmed_data, 
                               final_data, untrimmed_params, final_optimal_offset):
        """Run advanced analytics using the unified engine."""
        
        # Prepare input data
        input_data = AnalysisInput(
            position=untrimmed_data["position"],
            error=untrimmed_data["error"],
            upper_limit=untrimmed_data.get("upper_limit"),
            lower_limit=untrimmed_data.get("lower_limit"),
            untrimmed_data=untrimmed_data,
            trimmed_data=final_data,
            sigma_gradient=untrimmed_params["sigma_gradient"],
            sigma_threshold=untrimmed_params["sigma_threshold"],
            linearity_spec=untrimmed_params["linearity_spec"],
            travel_length=untrimmed_params["travel_length"],
            untrimmed_resistance=track_result_dict.get("Untrimmed Resistance"),
            trimmed_resistance=track_result_dict.get("Trimmed Resistance")
        )
        
        # Run analysis
        try:
            results = self.analytics_engine.analyze(input_data)
            
            # Update track results
            if results.linearity:
                track_result_dict.update({
                    "Max Deviation": results.linearity.max_deviation,
                    "Max Deviation Position": results.linearity.max_deviation_position,
                    "Deviation Uniformity": results.linearity.deviation_uniformity,
                })
            
            if results.trim_effectiveness:
                track_result_dict.update({
                    "Trim Improvement (%)": results.trim_effectiveness.improvement_percent,
                    "Untrimmed RMS Error": results.trim_effectiveness.untrimmed_rms,
                    "Trimmed RMS Error": results.trim_effectiveness.trimmed_rms,
                    "Max Error Reduction (%)": results.trim_effectiveness.max_error_reduction,
                })
            
            if results.zones and results.zones.worst_zone:
                track_result_dict.update({
                    "Worst Zone": results.zones.worst_zone.zone_number,
                    "Worst Zone Position": results.zones.worst_zone.position_range[0],
                })
            
            if results.failure_probability:
                track_result_dict.update({
                    "Failure Probability": results.failure_probability.failure_probability,
                    "Risk Category": results.failure_probability.risk_category,
                    "Gradient Margin": results.failure_probability.gradient_margin,
                })
            
            if results.dynamic_range:
                track_result_dict.update({
                    "Range Utilization (%)": results.dynamic_range.range_utilization_percent,
                    "Minimum Margin": results.dynamic_range.minimum_margin,
                    "Minimum Margin Position": results.dynamic_range.minimum_margin_position,
                    "Margin Bias": results.dynamic_range.margin_bias,
                })
                
        except Exception as e:
            self.logger.error(f"Error in advanced analytics: {e}")
```

## Configuration Profiles

The new engine supports different analysis profiles:

### Basic Profile
For quick checks - only linearity and resistance:
```python
engine = AnalyticsEngine(
    config=AnalysisConfig(profile=AnalysisProfile.BASIC)
)
```

### Standard Profile (Default)
Most common analyses:
```python
engine = AnalyticsEngine(
    config=AnalysisConfig(profile=AnalysisProfile.STANDARD)
)
```

### Detailed Profile
All analyses including dynamic range:
```python
engine = AnalyticsEngine(
    config=AnalysisConfig(profile=AnalysisProfile.DETAILED)
)
```

### Custom Profile
Choose specific analyses:
```python
config = AnalysisConfig(
    profile=AnalysisProfile.CUSTOM,
    enabled_analyses=["linearity", "zones", "failure_probability"],
    num_zones=10,  # Custom parameters
    high_risk_threshold=0.8
)
engine = AnalyticsEngine(config=config)
```

## Adding Custom Analyzers

Create plugins for specialized analyses:

```python
from analytics_engine import AnalyzerPlugin

class FrequencyAnalyzerPlugin(AnalyzerPlugin):
    """Analyze frequency content of error signal."""
    
    @property
    def name(