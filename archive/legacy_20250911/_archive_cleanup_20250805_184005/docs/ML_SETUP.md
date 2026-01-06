# ML Setup Guide for Laser Trim Analyzer

## Overview

The Laser Trim Analyzer includes advanced Machine Learning (ML) features for:
- **Failure Prediction**: Predicts component failure probability
- **Threshold Optimization**: Optimizes analysis thresholds based on historical data
- **Drift Detection**: Detects manufacturing drift patterns

## Prerequisites

### Required Dependencies

The ML features require the following Python packages:

```bash
# Core ML dependencies
scikit-learn>=1.3.0
joblib>=1.3.0
numpy>=1.24.0
pandas>=2.0.0

# Optional but recommended
psutil>=5.9.0  # For system resource monitoring
```

### Installation

1. **Install all dependencies at once:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Or install only ML dependencies:**
   ```bash
   pip install scikit-learn joblib numpy pandas psutil
   ```

## Troubleshooting

### ML Engine Shows "Missing Dependencies"

If the ML Tools page shows "Missing Dependencies" status:

1. Check which dependencies are missing:
   ```bash
   python3 test_ml_init.py
   ```

2. Install the missing packages:
   ```bash
   pip install scikit-learn joblib pandas numpy
   ```

3. Restart the application

### ML Engine Shows "Error" Status

1. Check the application logs for detailed error messages
2. Ensure you have sufficient disk space for model storage
3. Verify write permissions to `~/.laser_trim_analyzer/models/`

### Models Show "Not Trained" Status

This is normal for first-time use. Models need to be trained with your data:

1. Go to ML Tools page
2. Load historical data
3. Click "Train Models" to train all models
4. Models will be saved automatically for future use

## Model Storage

ML models are stored in:
- **Linux/Mac**: `~/.laser_trim_analyzer/models/`
- **Windows**: `%USERPROFILE%\.laser_trim_analyzer\models\`

Each model is saved as:
- `threshold_optimizer.joblib`
- `failure_predictor.joblib`
- `drift_detector.joblib`

## System Requirements

### Minimum Requirements
- **RAM**: 4GB (8GB recommended for large datasets)
- **Disk Space**: 500MB for model storage
- **Python**: 3.8 or higher

### Performance Tips
1. Close other applications when training models
2. Use smaller batch sizes if memory is limited
3. Enable model caching for faster predictions

## Verifying Installation

Run the test script to verify ML components:

```bash
python3 test_ml_init.py
```

Expected output:
```
✓ NumPy 1.24.0 installed
✓ Pandas 2.0.0 installed
✓ scikit-learn 1.3.0 installed
✓ joblib 1.3.0 installed
✓ psutil 5.9.0 installed
✓ MLEngine initialized successfully
✓ ML Manager obtained successfully
```

## Using ML Features

### Without ML Dependencies

The application will still function normally without ML dependencies:
- All core analysis features work
- Historical data can be viewed
- Reports can be generated

Only the ML-specific features will be disabled:
- Failure predictions
- Threshold optimization
- Drift detection
- ML insights

### With ML Dependencies

Once dependencies are installed:
1. ML Tools page becomes fully functional
2. Predictive analytics are available
3. Automated threshold optimization is enabled
4. Real-time drift detection works

## Contact Support

If you continue to experience issues:
1. Check the application logs
2. Run the diagnostic script
3. Contact support with the diagnostic output