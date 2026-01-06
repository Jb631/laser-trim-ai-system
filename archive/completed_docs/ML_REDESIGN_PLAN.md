# Per-Model ML System Redesign

## Overview

This document outlines the redesign of the ML system to be truly per-model, using all available data features and leveraging both Trim and Final Test results for learning.

## Key Design Principles

1. **Processing and ML are completely separate** - No ML during file processing
2. **ML trains from database** - After processing 70k+ files, train from stored data
3. **Two-button workflow** - "Train Models" then "Apply ML to DB"
4. **Use ALL available data** - Trim + Final Test, pass/fail + severity (fail points)

---

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
- Learning from BOTH Trim and Final Test data
- Severity weighting (fail point count)
- Rich feature set from all available data
- Model-specific drift detection
- Statistical profiling per model

---

## Workflow

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: PROCESS (Fast - no ML)                         │
│  - Parse all 70k+ files                                 │
│  - Analyze with formula thresholds                      │
│  - Store everything in DB                               │
│  - NO ML computation during this step                   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 2: TRAIN ML (Settings button - "Train Models")    │
│  - Query all trim + final test data from DB             │
│  - For each model with enough data:                     │
│    - Build statistical profile                          │
│    - Train failure predictor (RandomForest)             │
│    - Calculate optimal threshold (severity-weighted)    │
│    - Establish drift baselines                          │
│  - Store results in model_ml_state table                │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 3: APPLY ML (Settings button - "Apply to DB")     │
│  - Re-calculate sigma_pass using learned thresholds     │
│  - Add failure_probability to each track                │
│  - Run drift detection per model                        │
│  - Update records with ML insights                      │
│  - Background task with progress bar                    │
└─────────────────────────────────────────────────────────┘
```

---

## Data Sources for Learning

### Training Data (Both contribute)

| Source | Ground Truth | Severity | Priority |
|--------|--------------|----------|----------|
| Final Test | linearity pass/fail | fail_points count | Primary (when linked) |
| Trim File | linearity pass/fail | fail_points count | Secondary (always available) |

**Severity Weighting**: More fail points = worse outcome, influences threshold calculation

### Features Available from DB

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

---

## Per-Model ML Components

### 1. ModelPredictor
Predicts failure probability from trim features.

```python
class ModelPredictor:
    """Per-model failure probability predictor."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.classifier: RandomForestClassifier = None
        self.scaler: StandardScaler = None
        self.is_trained: bool = False
        self.training_samples: int = 0
        self.metrics: Dict = {}  # accuracy, precision, recall, f1, auc

    def train(self, features: pd.DataFrame, labels: pd.Series,
              severity: pd.Series) -> bool:
        """Train on this model's data with severity weighting."""
        pass

    def predict_failure_probability(self, features: Dict) -> float:
        """Predict probability that this unit will fail."""
        pass

    def get_feature_importance(self) -> Dict[str, float]:
        """Get which features matter most for this model."""
        pass
```

### 2. ModelThresholdOptimizer
Learns optimal sigma threshold from outcomes.

```python
class ModelThresholdOptimizer:
    """Per-model threshold optimization with severity weighting."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.threshold: float = None
        self.confidence: float = None  # Based on sample size + separation

    def calculate_threshold(self, data: pd.DataFrame) -> float:
        """
        Calculate optimal sigma threshold.

        Uses:
        - Pass/fail outcomes from Trim AND Final Test
        - Severity weighting (more fail points = worse)
        - Distribution analysis (separation, overlap)
        """
        pass
```

### 3. ModelDriftDetector
Detects quality shifts per model.

```python
class ModelDriftDetector:
    """Per-model drift detection with model-specific baselines."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.baseline_mean: float = None
        self.baseline_std: float = None
        self.baseline_p95: float = None
        self.cusum_pos: float = 0.0
        self.cusum_neg: float = 0.0
        self.ewma_value: float = None

    def set_baseline(self, sigma_values: np.ndarray) -> bool:
        """Set baseline from this model's historical data."""
        pass

    def detect(self, sigma_value: float) -> DriftResult:
        """Check if this value indicates drift."""
        pass

    def get_control_limits(self) -> Tuple[float, float, float]:
        """Get (lower, center, upper) control limits."""
        pass
```

### 4. ModelProfiler (NEW)
Builds statistical profile for insights.

```python
class ModelProfiler:
    """Per-model statistical profiling for insights."""

    def __init__(self, model_name: str):
        self.model_name = model_name

        # Distribution stats
        self.sigma_mean: float = None
        self.sigma_std: float = None
        self.sigma_p5: float = None
        self.sigma_p50: float = None
        self.sigma_p95: float = None

        self.error_mean: float = None
        self.error_std: float = None

        # Quality metrics
        self.pass_rate: float = None
        self.fail_rate: float = None
        self.avg_fail_points: float = None  # When failures occur

        # Correlations
        self.track_correlation: float = None  # Track 1 vs Track 2
        self.sigma_error_correlation: float = None

        # Comparisons
        self.spec_margin: float = None  # How much margin to spec?
        self.difficulty_score: float = None  # Relative to other models

    def build_profile(self, data: pd.DataFrame) -> None:
        """Build full statistical profile from DB data."""
        pass

    def get_insights(self) -> List[str]:
        """Generate human-readable insights."""
        # e.g., "This model has tight spec margin (5%)"
        # e.g., "Track 1 and Track 2 failures are highly correlated (0.92)"
        pass
```

### 5. MLManager
Orchestrates all per-model ML.

```python
class MLManager:
    """Manages all per-model ML components."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.predictors: Dict[str, ModelPredictor] = {}
        self.threshold_optimizers: Dict[str, ModelThresholdOptimizer] = {}
        self.drift_detectors: Dict[str, ModelDriftDetector] = {}
        self.profilers: Dict[str, ModelProfiler] = {}

    # Training (from DB data)
    def train_model(self, model_name: str) -> TrainingResult:
        """Train all ML components for a specific model."""
        pass

    def train_all_models(self, min_samples: int = 50,
                         progress_callback=None) -> Dict[str, TrainingResult]:
        """Train ML for all models with sufficient data."""
        pass

    # Application (update DB with ML results)
    def apply_to_database(self, progress_callback=None) -> ApplyResult:
        """Update all DB records with ML predictions."""
        pass

    # Queries
    def get_threshold(self, model_name: str) -> Optional[float]:
        """Get learned threshold, or None for formula fallback."""
        pass

    def get_failure_probability(self, model_name: str,
                                features: Dict) -> Optional[float]:
        """Get failure probability prediction."""
        pass

    def get_model_insights(self, model_name: str) -> Dict:
        """Get full insights for a model (profile + predictions + drift)."""
        pass

    def get_cross_model_comparison(self) -> pd.DataFrame:
        """Compare all models (difficulty, pass rates, etc.)."""
        pass

    # Persistence
    def save_all(self) -> None:
        """Save all trained state to DB."""
        pass

    def load_all(self) -> None:
        """Load all trained state from DB."""
        pass
```

---

## Insights Available from Full Data

### Per-Model Insights

| Insight | Description | Source |
|---------|-------------|--------|
| Optimal threshold | Best sigma cutoff for this model | ThresholdOptimizer |
| Failure probability | Likelihood of Final Test failure | Predictor |
| Drift status | Is quality shifting? | DriftDetector |
| Pass/fail rate | Historical quality | Profiler |
| Spec margin | How close to spec limits? | Profiler |
| Track correlation | Do tracks fail together? | Profiler |
| Feature importance | What drives failures? | Predictor |
| Difficulty score | Harder/easier than average? | Profiler |

### Cross-Model Insights

| Insight | Description |
|---------|-------------|
| Model ranking | Which models are hardest to trim? |
| Spec analysis | Which models have too tight/loose specs? |
| Failure patterns | Common failure modes across models |
| Drift alerts | Which models are currently drifting? |
| Training status | Which models need more data? |

### Time-Based Insights

| Insight | Description |
|---------|-------------|
| Quality trends | Improving or degrading over time? |
| Drift detection | When did quality shift? |
| Batch effects | Quality variations by date/batch |

---

## Database Schema

### New Table: model_ml_state

```sql
CREATE TABLE model_ml_state (
    id INTEGER PRIMARY KEY,
    model VARCHAR(50) UNIQUE NOT NULL,

    -- Training metadata
    is_trained BOOLEAN DEFAULT FALSE,
    training_date DATETIME,
    training_samples INTEGER DEFAULT 0,
    trim_samples INTEGER DEFAULT 0,
    final_test_samples INTEGER DEFAULT 0,

    -- Learned threshold
    sigma_threshold FLOAT,
    threshold_confidence FLOAT,

    -- Predictor metrics
    predictor_accuracy FLOAT,
    predictor_precision FLOAT,
    predictor_recall FLOAT,
    predictor_f1 FLOAT,
    predictor_auc FLOAT,
    feature_importance TEXT,  -- JSON dict

    -- Profile statistics
    sigma_mean FLOAT,
    sigma_std FLOAT,
    sigma_p5 FLOAT,
    sigma_p50 FLOAT,
    sigma_p95 FLOAT,
    error_mean FLOAT,
    error_std FLOAT,
    pass_rate FLOAT,
    fail_rate FLOAT,
    avg_fail_points FLOAT,
    track_correlation FLOAT,
    spec_margin FLOAT,
    difficulty_score FLOAT,

    -- Drift baselines
    drift_baseline_mean FLOAT,
    drift_baseline_std FLOAT,
    drift_baseline_p95 FLOAT,
    cusum_pos FLOAT DEFAULT 0,
    cusum_neg FLOAT DEFAULT 0,
    ewma_value FLOAT,
    is_drifting BOOLEAN DEFAULT FALSE,
    drift_direction VARCHAR(10),  -- 'up', 'down', or NULL

    -- Timestamps
    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_date DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Model Storage

RandomForest models stored as pickle files:
```
data/ml_models/
├── predictors/
│   ├── 6828.pkl
│   ├── 8340.pkl
│   └── ...
└── scalers/
    ├── 6828.pkl
    ├── 8340.pkl
    └── ...
```

---

## Settings Page UI

```
┌─────────────────────────────────────────────────────────┐
│ ML Settings                                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Training                                                │
│ ┌─────────────────┐  ┌─────────────────┐               │
│ │  Train Models   │  │  Apply to DB    │               │
│ └─────────────────┘  └─────────────────┘               │
│                                                         │
│ Status: 15 models trained, 3 need more data            │
│ Last trained: 2025-12-26 14:30                         │
│                                                         │
│ ┌─────────────────────────────────────────────────────┐│
│ │ Model    Samples  Threshold  Accuracy  Status       ││
│ │ 6828     405      0.00051    94%       ✓ Trained    ││
│ │ 8340     120      0.00063    89%       ✓ Trained    ││
│ │ 6607     9        -          -         Need 41 more ││
│ │ 8125     85       0.00048    91%       ⚠ Drifting   ││
│ └─────────────────────────────────────────────────────┘│
│                                                         │
│ Minimum samples for training: [50    ]                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Create `ModelPredictor` class in `ml/predictor.py`
- [ ] Create `ModelThresholdOptimizer` class in `ml/threshold_optimizer.py`
- [ ] Create `ModelDriftDetector` class in `ml/drift_detector.py`
- [ ] Create `ModelProfiler` class in `ml/profiler.py`
- [ ] Create `MLManager` class in `ml/manager.py`
- [ ] Add `model_ml_state` table to database

### Phase 2: Training Pipeline
- [ ] Feature extraction from DB records
- [ ] Training data gathering (Trim + Final Test, with severity)
- [ ] Per-model RandomForest training
- [ ] Severity-weighted threshold calculation
- [ ] Profile statistics calculation
- [ ] Drift baseline calculation

### Phase 3: Apply to Database
- [ ] Batch update of sigma_pass using learned thresholds
- [ ] Add failure_probability to track records
- [ ] Run drift detection and store alerts
- [ ] Progress reporting for UI

### Phase 4: UI Integration
- [ ] Settings page ML section with two buttons
- [ ] Per-model training status table
- [ ] Trends page with ML insights
- [ ] Dashboard drift alerts
- [ ] Model comparison views

### Phase 5: Cleanup
- [ ] Remove or disable old ThresholdOptimizer
- [ ] Remove or disable old DriftDetector
- [ ] Update documentation
- [ ] Test with 70k+ files

---

## Performance Considerations

### Processing Speed (70k+ files)
- NO ML during processing - just parse, analyze with formula, store
- ML is completely separate step after processing

### Training Speed
- Query DB in batches
- Train models in parallel where possible
- Progress callback for UI updates
- Expected: 30-60 seconds for all models

### Apply Speed
- Batch updates to DB
- Progress callback for UI
- Expected: 1-2 minutes for 70k records

### Memory (8GB constraint)
- Load predictors lazily
- Process in batches during training
- Don't keep all data in memory at once

---

## Success Metrics

1. **Processing Speed**: No slowdown vs current (no ML during processing)
2. **Threshold Accuracy**: % where learned threshold matches linearity outcome better than formula
3. **Prediction Accuracy**: % correct Final Test predictions (>85% target)
4. **Drift Detection**: Catch quality shifts before they cause batch failures
5. **Coverage**: >80% of models with trained ML

---

## Summary

This redesign:
- **Separates processing from ML** - Process 70k+ files fast, train ML after
- **Uses ALL data** - Trim + Final Test, pass/fail + severity
- **Per-model everything** - Thresholds, predictions, drift, profiles
- **Two-button workflow** - Train, then Apply
- **Rich insights** - Statistical profiles, cross-model comparisons, drift alerts
