# Machine Learning Models Documentation

## Overview

The ML models module provides intelligent analysis capabilities for laser trim data:

- **Threshold Optimization**: Learns optimal sigma thresholds from historical data
- **Failure Prediction**: Predicts unit failures with >90% accuracy target
- **Drift Detection**: Identifies manufacturing process drift in real-time
- **Feature Analysis**: Provides explainable AI insights

## Architecture

```
ml_models.py
├── LaserTrimMLModels (Main Class)
│   ├── Threshold Optimizer (Random Forest Regressor)
│   ├── Failure Predictor (Random Forest Classifier)
│   ├── Drift Detector (Isolation Forest)
│   └── Feature Engineering Pipeline
├── Model Persistence System
├── Feature Importance Tracking
└── Integration Functions
```

## Key Features

### 1. Adaptive Threshold Optimization

The threshold optimizer learns from historical pass/fail data to recommend optimal thresholds:

```python
# Train the optimizer
results = ml_models.train_threshold_optimizer(historical_data)

# Get optimal threshold for new unit
prediction = ml_models.predict_optimal_threshold(unit_features)
print(f"Optimal threshold: {prediction['optimal_threshold']:.3f}")
print(f"Confidence: {prediction['confidence']:.1%}")
```

**How it works:**
- Uses Random Forest regression with hyperparameter tuning
- Considers multiple features beyond just sigma gradient
- Provides confidence scores based on tree variance
- Tracks feature contributions for explainability

### 2. High-Accuracy Failure Prediction

Predicts failures with target >90% accuracy:

```python
# Train predictor
results = ml_models.train_failure_predictor(
    historical_data,
    failure_window_days=30,
    target_accuracy=0.90
)

# Predict failure risk
risk = ml_models.predict_failure_probability(unit_features)
print(f"Failure probability: {risk['failure_probability']:.1%}")
print(f"Risk level: {risk['risk_level']}")
```

**Features:**
- Handles class imbalance automatically
- Feature selection for interpretability
- Risk categorization (MINIMAL/LOW/MEDIUM/HIGH/CRITICAL)
- Identifies top risk factors

### 3. Manufacturing Drift Detection

Detects anomalies indicating process drift:

```python
# Train drift detector
results = ml_models.train_drift_detector(
    historical_data,
    contamination=0.1  # Expected 10% anomaly rate
)

# Check for drift
drift = ml_models.detect_manufacturing_drift(unit_features)
if drift['is_drift']:
    print(f"Drift detected! Severity: {drift['severity']}")
    print(f"Recommendation: {drift['recommendation']}")
```

**Capabilities:**
- Uses Isolation Forest for anomaly detection
- Provides drift severity levels
- Identifies which features are drifting
- Gives actionable recommendations

### 4. Feature Engineering

Automatic feature engineering from raw data:

```python
features = ml_models.prepare_features(raw_data)
```

**Engineered features include:**
- Basic measurements (sigma gradient, linearity spec, etc.)
- Statistical features (mean, std, skewness, kurtosis of errors)
- Domain-specific ratios (sigma/spec ratio, length ratios)
- Time-based features (hour, day of week, trends)
- Model-specific encoding

## Model Training

### Complete Training Pipeline

```python
# Create ML models instance
ml_models = create_ml_models(config)

# Train all models at once
results = train_all_models(ml_models, historical_data)

# Results include:
# - Performance metrics for each model
# - Feature importance rankings
# - Model version for tracking
```

### Individual Model Training

```python
# Train specific models
threshold_results = ml_models.train_threshold_optimizer(data)
failure_results = ml_models.train_failure_predictor(data)
drift_results = ml_models.train_drift_detector(data)
```

## Model Persistence

### Saving Models

```python
# Save with auto-versioning
version = ml_models.save_models()

# Save with specific version
version = ml_models.save_models('v2.0_production')
```

Saves:
- All trained models (.joblib files)
- Scalers for data normalization
- Feature importance scores
- Model metadata and performance metrics

### Loading Models

```python
# Load specific version
ml_models.load_models('v2.0_production')

# Load latest version
ml_models.load_models('latest')
```

## Feature Importance Analysis

### Get Feature Importance Report

```python
report = ml_models.get_feature_importance_report()

# Report includes:
# - Top features for each model
# - Overall most important features
# - Feature categorization
```

### Understanding Feature Importance

Features are categorized into:
- **Measurements**: Direct sensor readings
- **Statistics**: Calculated error statistics
- **Ratios**: Domain-specific calculated ratios
- **Model-specific**: Features related to product models
- **Time-based**: Temporal patterns

## Integration with Data Processor

```python
# Complete workflow example
config = Config()
processor = LaserTrimDataProcessor(config)
ml_models = create_ml_models(config)

# Process unit data
unit_data = processor.process_file('unit_data.xlsx')

# Get ML insights
threshold = ml_models.predict_optimal_threshold(unit_data)
failure_risk = ml_models.predict_failure_probability(unit_data)
drift_check = ml_models.detect_manufacturing_drift(unit_data)

# Make decisions based on ML
if failure_risk['risk_level'] == 'CRITICAL':
    # Trigger immediate inspection
    pass
```

## Performance Metrics

### Threshold Optimizer
- **MAE (Mean Absolute Error)**: Target < 0.1
- **R² Score**: Target > 0.8
- **Feature importance**: Tracks which features matter most

### Failure Predictor
- **Accuracy**: Target > 90%
- **Precision**: Minimize false positives
- **Recall**: Catch actual failures
- **F1 Score**: Balance precision and recall

### Drift Detector
- **Contamination rate**: Configurable (default 10%)
- **Anomaly threshold**: Auto-calculated
- **Feature drift scores**: Identifies changing features

## Best Practices

### 1. Data Requirements

- **Minimum samples**: 100 for basic training, 1000+ recommended
- **Feature completeness**: Handle missing values appropriately
- **Class balance**: System handles imbalanced data automatically

### 2. Model Updates

```python
# Retrain periodically with new data
if days_since_last_training > 30:
    ml_models.train_threshold_optimizer(new_data, force_retrain=True)
```

### 3. Monitoring Model Performance

```python
# Track prediction confidence
if prediction['confidence'] < 0.7:
    # Consider retraining or manual review
    pass
```

### 4. Handling Edge Cases

```python
# Always check for errors
prediction = ml_models.predict_failure_probability(features)
if 'error' in prediction:
    # Fall back to rule-based system
    pass
```

## Troubleshooting

### Common Issues

1. **"Insufficient data for training"**
   - Need at least 10 samples for threshold optimizer
   - Need examples of both pass and fail for failure predictor

2. **"Model not trained"**
   - Ensure models are trained before prediction
   - Check if models loaded successfully

3. **Low prediction confidence**
   - May indicate data outside training distribution
   - Consider retraining with more diverse data

### Performance Optimization

1. **Reduce training time**:
   ```python
   # Use fewer hyperparameter combinations
   param_grid = {
       'n_estimators': [100],  # Instead of [50, 100, 200]
       'max_depth': [10]       # Instead of [5, 10, None]
   }
   ```

2. **Reduce memory usage**:
   ```python
   # Use fewer trees
   rf = RandomForestClassifier(n_estimators=50)  # Instead of 100
   ```

## Advanced Usage

### Custom Feature Engineering

```python
class CustomMLModels(LaserTrimMLModels):
    def prepare_features(self, data, target_type='classification'):
        # Call parent method
        features = super().prepare_features(data, target_type)
        
        # Add custom features
        features['custom_metric'] = data['value1'] / data['value2']
        
        return features
```

### Ensemble Predictions

```python
# Combine multiple model predictions
threshold_pred = ml_models.predict_optimal_threshold(features)
failure_pred = ml_models.predict_failure_probability(features)

# Weighted decision
if (threshold_pred['confidence'] > 0.8 and 
    failure_pred['failure_probability'] > 0.7):
    # High confidence in both models
    action = 'immediate_inspection'
```

### Real-time Monitoring Dashboard

```python
# Integration with monitoring system
class MLMonitor:
    def __init__(self, ml_models):
        self.ml_models = ml_models
        self.alert_threshold = 0.8
        
    def check_unit(self, unit_data):
        # Get all predictions
        results = {
            'threshold': self.ml_models.predict_optimal_threshold(unit_data),
            'failure': self.ml_models.predict_failure_probability(unit_data),
            'drift': self.ml_models.detect_manufacturing_drift(unit_data)
        }
        
        # Generate alerts
        alerts = []
        if results['failure']['failure_probability'] > self.alert_threshold:
            alerts.append('HIGH_FAILURE_RISK')
        if results['drift']['is_drift']:
            alerts.append('MANUFACTURING_DRIFT')
            
        return results, alerts
```

## Next Steps

1. **Collect historical data** for model training
2. **Define failure criteria** specific to your process
3. **Set up periodic retraining** schedule
4. **Integrate with production systems** for real-time analysis
5. **Monitor model performance** and adjust as needed

Remember: The ML models are designed to augment, not replace, engineering judgment. Always validate predictions against domain knowledge and safety requirements.