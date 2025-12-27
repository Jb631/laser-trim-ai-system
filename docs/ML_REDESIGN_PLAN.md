# Per-Model ML System Redesign

## Overview

This document outlines the redesign of the ML system to be truly per-model, using all available data features and leveraging Final Test results as ground truth.

## Current State (Problems)

### ThresholdOptimizer
- Single global RandomForest model for all product models
- Model number hashed to 0-999 as a feature (loses specificity)
- Only uses: model_hash, unit_length, linearity_spec
- Target: sigma_gradient * 1.2 (arbitrary margin)
- Not learning from actual pass/fail outcomes

### DriftDetector
- Single global baseline for all product models
- One mean/std for entire dataset
- Cannot detect model-specific drift
- CUSUM/EWMA parameters same for all models

### What's Missing
- Per-model learning and baselines
- Final Test pass/fail as ground truth (we have this data!)
- Rich feature set from trim data
- Model-specific drift detection
- Correlation between trim metrics and final test outcomes

---

## Proposed Architecture

### Core Principle
**Each product model gets its own trained ML model and baselines.**

### Data Flow
```
Trim File → Parse → Analyze → Store in DB
                                    ↓
Final Test File → Parse → Match to Trim → Store Link
                                    ↓
                    ML Training (per model)
                                    ↓
              Threshold + Drift + Failure Prediction
```

---

## Feature Engineering

### Features Available from Trim Data

| Feature | Description | Source |
|---------|-------------|--------|
| sigma_gradient | Std dev of error gradients | TrackResult |
| linearity_error | Max deviation after offset | TrackResult |
| linearity_fail_points | Count of out-of-spec points | TrackResult |
| optimal_offset | Adjustment needed to center | TrackResult |
| travel_length | Total travel distance | TrackResult |
| unit_length | Physical unit size | TrackResult |
| linearity_spec | Specification tolerance | TrackResult |
| resistance_change | (trimmed - untrimmed) / untrimmed | TrackResult |
| sigma_to_spec_ratio | sigma_gradient / linearity_spec | Derived |
| error_to_spec_ratio | linearity_error / linearity_spec | Derived |
| offset_magnitude | abs(optimal_offset) | Derived |

### Target Variable (Ground Truth)

**Primary**: Final Test pass/fail (when linked)
- Best ground truth - actual post-assembly outcome
- Requires matched Final Test data

**Secondary**: Linearity pass/fail (always available)
- Used when no Final Test link exists
- Good proxy for trim quality

---

## Per-Model ML Components

### 1. ModelPredictor (New - replaces ThresholdOptimizer)

```python
class ModelPredictor:
    """
    Per-model ML predictor for failure probability and optimal thresholds.

    Each product model gets its own:
    - Trained RandomForest classifier (predicts failure probability)
    - Optimal sigma threshold (learned from data)
    - Feature statistics (for normalization)
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.classifier: RandomForestClassifier = None
        self.threshold: float = None
        self.feature_stats: Dict = {}
        self.is_trained: bool = False
        self.training_samples: int = 0
        self.training_date: datetime = None

    def train(self, features: pd.DataFrame, labels: pd.Series) -> TrainingResult:
        """Train on this model's data only."""
        pass

    def predict_failure_probability(self, features: Dict) -> float:
        """Predict probability that this unit will fail Final Test."""
        pass

    def get_optimal_threshold(self) -> float:
        """Get learned optimal sigma threshold for this model."""
        pass
```

### 2. ModelDriftDetector (New - replaces DriftDetector)

```python
class ModelDriftDetector:
    """
    Per-model drift detection with model-specific baselines.

    Each product model gets its own:
    - Baseline statistics (mean, std, percentiles)
    - CUSUM accumulators
    - EWMA state
    - Drift alerts specific to this model
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.baseline_mean: float = None
        self.baseline_std: float = None
        self.baseline_p95: float = None
        self.cusum_pos: float = 0.0
        self.cusum_neg: float = 0.0
        self.ewma_value: float = None
        self.sample_count: int = 0

    def set_baseline(self, sigma_values: np.ndarray) -> bool:
        """Set baseline from this model's historical data."""
        pass

    def detect(self, sigma_value: float) -> DriftResult:
        """Check if this value indicates drift for this model."""
        pass

    def get_control_limits(self) -> Tuple[float, float, float]:
        """Get (lower, center, upper) control limits for this model."""
        pass
```

### 3. MLManager (New - orchestrates per-model learning)

```python
class MLManager:
    """
    Manages per-model ML predictors and drift detectors.

    Responsibilities:
    - Load/save per-model ML state
    - Train models on demand
    - Provide predictions during analysis
    - Aggregate drift alerts across models
    """

    def __init__(self):
        self.predictors: Dict[str, ModelPredictor] = {}
        self.drift_detectors: Dict[str, ModelDriftDetector] = {}
        self.storage_path: Path = None

    def get_predictor(self, model_name: str) -> ModelPredictor:
        """Get or create predictor for a model."""
        pass

    def get_drift_detector(self, model_name: str) -> ModelDriftDetector:
        """Get or create drift detector for a model."""
        pass

    def train_model(self, model_name: str, min_samples: int = 50) -> TrainingResult:
        """Train ML for a specific product model."""
        pass

    def train_all_models(self, min_samples: int = 50) -> Dict[str, TrainingResult]:
        """Train ML for all models with sufficient data."""
        pass

    def save_all(self) -> None:
        """Persist all trained models to disk."""
        pass

    def load_all(self) -> None:
        """Load all trained models from disk."""
        pass
```

---

## Database Schema Updates

### New Table: model_ml_state

Stores per-model ML state and statistics.

```sql
CREATE TABLE model_ml_state (
    id INTEGER PRIMARY KEY,
    model VARCHAR(50) UNIQUE NOT NULL,

    -- Training metadata
    is_trained BOOLEAN DEFAULT FALSE,
    training_date DATETIME,
    training_samples INTEGER DEFAULT 0,
    training_features TEXT,  -- JSON list of feature names used

    -- Learned threshold
    sigma_threshold FLOAT,
    threshold_confidence FLOAT,  -- 0-1, based on sample size + separation

    -- Feature statistics (for normalization)
    feature_means TEXT,  -- JSON dict
    feature_stds TEXT,   -- JSON dict

    -- Performance metrics
    accuracy FLOAT,
    precision_score FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    auc_roc FLOAT,

    -- Drift detection baselines
    drift_baseline_mean FLOAT,
    drift_baseline_std FLOAT,
    drift_baseline_p95 FLOAT,
    drift_sample_count INTEGER DEFAULT 0,

    -- CUSUM/EWMA state (for online detection)
    cusum_pos FLOAT DEFAULT 0,
    cusum_neg FLOAT DEFAULT 0,
    ewma_value FLOAT,

    -- Timestamps
    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_date DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Model Storage

Per-model RandomForest models stored as pickle files:
```
data/ml_models/
├── predictors/
│   ├── 6828.pkl
│   ├── 8340.pkl
│   └── ...
└── drift/
    ├── 6828.pkl
    ├── 8340.pkl
    └── ...
```

Lightweight state in database, heavy models on disk.

---

## Training Pipeline

### Step 1: Gather Training Data (per model)

```python
def get_training_data(model_name: str) -> pd.DataFrame:
    """
    Get training data for a specific model.

    Priority:
    1. Trim records WITH linked Final Test (ground truth)
    2. Trim records without Final Test (use linearity as proxy)
    """
    with db.session() as session:
        # Get linked pairs first
        linked = (
            session.query(TrackResult, FinalTestResult)
            .join(AnalysisResult)
            .join(FinalTestResult,
                  FinalTestResult.linked_trim_id == AnalysisResult.id)
            .filter(AnalysisResult.model == model_name)
            .all()
        )

        # Get unlinked trim data
        unlinked = (
            session.query(TrackResult)
            .join(AnalysisResult)
            .outerjoin(FinalTestResult,
                       FinalTestResult.linked_trim_id == AnalysisResult.id)
            .filter(
                AnalysisResult.model == model_name,
                FinalTestResult.id == None
            )
            .all()
        )

        # Build feature DataFrame
        # ...
```

### Step 2: Feature Extraction

```python
def extract_features(track: TrackResult) -> Dict[str, float]:
    """Extract ML features from a track result."""
    return {
        'sigma_gradient': track.sigma_gradient,
        'linearity_error': track.final_linearity_error_shifted,
        'fail_points': track.linearity_fail_points,
        'optimal_offset': abs(track.optimal_offset),
        'travel_length': track.travel_length,
        'linearity_spec': track.linearity_spec,
        'sigma_to_spec': track.sigma_gradient / track.linearity_spec,
        'error_to_spec': track.final_linearity_error_shifted / track.linearity_spec,
        'resistance_change': calculate_resistance_change(track),
    }
```

### Step 3: Train Model

```python
def train_model_predictor(model_name: str, data: pd.DataFrame) -> ModelPredictor:
    """Train predictor for a specific product model."""

    predictor = ModelPredictor(model_name)

    # Split features and labels
    X = data[FEATURE_COLUMNS]
    y = data['failed']  # 1 = failed Final Test (or linearity)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Normalize features
    predictor.scaler = StandardScaler()
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)

    # Train RandomForest
    predictor.classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',  # Handle imbalanced data
        n_jobs=-1
    )
    predictor.classifier.fit(X_train_scaled, y_train)

    # Calculate optimal threshold
    predictor.threshold = calculate_optimal_threshold(data)

    # Evaluate
    y_pred = predictor.classifier.predict(X_test_scaled)
    predictor.metrics = calculate_metrics(y_test, y_pred)

    return predictor
```

### Step 4: Calculate Optimal Threshold

```python
def calculate_optimal_threshold(data: pd.DataFrame) -> float:
    """
    Calculate optimal sigma threshold that best separates pass/fail.

    Strategy:
    1. If we have good pass/fail separation: use midpoint
    2. If overlap exists: use 95th percentile of passing + margin
    3. Minimum threshold based on spec
    """
    passing = data[data['failed'] == False]['sigma_gradient']
    failing = data[data['failed'] == True]['sigma_gradient']

    if len(failing) == 0:
        # No failures - use 95th percentile + 10% margin
        return float(np.percentile(passing, 95) * 1.1)

    max_pass = passing.max()
    min_fail = failing.min()

    if min_fail > max_pass:
        # Clean separation - use midpoint
        return (max_pass + min_fail) / 2
    else:
        # Overlap - use 95th percentile approach
        return float(np.percentile(passing, 95) * 1.1)
```

---

## Integration Points

### 1. Analyzer Integration

```python
class Analyzer:
    def __init__(self, ml_manager: Optional[MLManager] = None):
        self.ml_manager = ml_manager

    def _get_threshold(self, model: str, ...) -> float:
        """Get threshold - now per-model."""
        if self.ml_manager:
            predictor = self.ml_manager.get_predictor(model)
            if predictor.is_trained:
                return predictor.threshold

        # Fallback to formula
        return self._formula_threshold(...)

    def analyze_track(self, track_data: Dict, model: str) -> TrackData:
        # ... existing analysis ...

        # Add ML predictions if available
        if self.ml_manager:
            predictor = self.ml_manager.get_predictor(model)
            if predictor.is_trained:
                features = extract_features(track_data)
                result.failure_probability = predictor.predict_failure_probability(features)

            # Check for drift
            detector = self.ml_manager.get_drift_detector(model)
            if detector.has_baseline:
                drift_result = detector.detect(result.sigma_gradient)
                if drift_result.is_drifting:
                    # Create alert
                    pass
```

### 2. Settings Page

```
ML Settings:
├── [Train All Models] - Train ML for all models with 50+ samples
├── [Re-analyze Database] - Update all records with trained models
├── Status: "Trained 15 models, 3 need more data"
└── Model Details:
    ├── 6828: 405 samples, Acc=94%, Threshold=0.00051
    ├── 8340: 120 samples, Acc=89%, Threshold=0.00063
    └── 6607: 9 samples, Needs 41 more samples
```

### 3. Trends Page

Add per-model ML insights:
- Drift alerts specific to each model
- Threshold recommendations
- Feature importance per model
- Performance metrics

---

## Implementation Phases

### Phase 1: Core ML Infrastructure (Day 1)
- [ ] Create `ModelPredictor` class
- [ ] Create `ModelDriftDetector` class
- [ ] Create `MLManager` class
- [ ] Add `model_ml_state` database table
- [ ] Implement save/load for per-model state

### Phase 2: Training Pipeline (Day 1-2)
- [ ] Feature extraction from TrackResult
- [ ] Training data gathering (linked + unlinked)
- [ ] Per-model RandomForest training
- [ ] Optimal threshold calculation
- [ ] Performance metrics calculation
- [ ] Drift baseline calculation

### Phase 3: Integration (Day 2)
- [ ] Update Analyzer to use MLManager
- [ ] Update processor to pass model to analyzer
- [ ] Add drift detection during analysis
- [ ] Create drift alerts

### Phase 4: UI Updates (Day 2-3)
- [ ] Update Settings page ML section
- [ ] Add per-model training status
- [ ] Update Trends page with ML insights
- [ ] Add drift alerts to Dashboard

### Phase 5: Cleanup (Day 3)
- [ ] Remove old ThresholdOptimizer (or keep as fallback)
- [ ] Remove old DriftDetector (or keep as fallback)
- [ ] Update documentation
- [ ] Test with real data

---

## Performance Considerations

### Memory (8GB constraint)
- Load predictors lazily (only when model is analyzed)
- Keep only active models in memory
- Use SQLite for state, pickle for heavy models
- Process in batches during training

### Speed
- Cache predictions for same model during batch processing
- Pre-load commonly used models on startup
- Training is background/on-demand only

### Storage
- ~100KB per model pickle file
- 100 models = ~10MB total
- State table is lightweight

---

## Success Metrics

1. **Threshold Accuracy**: % of units where sigma_pass matches linearity_pass
2. **Prediction Accuracy**: % of correct Final Test pass/fail predictions
3. **Drift Detection**: Catching quality shifts before they cause failures
4. **False Positive Rate**: Flagging good units as bad (want < 5%)
5. **Coverage**: % of models with trained ML (want > 80%)

---

## Questions to Resolve

1. **Minimum samples per model?** Suggest 50 for training, 20 for threshold-only
2. **Retraining frequency?** On-demand via Settings, or periodic?
3. **Handle new models?** Use global fallback until 50 samples collected
4. **Feature selection?** Start with all, use importance to prune later

---

## Summary

This redesign transforms the ML system from:
- **Global models** → **Per-model models**
- **Hashed model numbers** → **Separate trained models**
- **Linearity-only proxy** → **Final Test ground truth**
- **Limited features** → **Rich feature set**
- **Global drift detection** → **Per-model baselines**

The result will be more accurate predictions, model-specific thresholds, and actionable drift alerts per product model.
