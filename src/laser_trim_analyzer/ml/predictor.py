"""
Per-Model Failure Predictor for Laser Trim Analyzer v3.

Predicts failure probability from trim features using RandomForest.
Each product model gets its own trained classifier.

Part of the per-model ML redesign - replaces global ThresholdOptimizer
for failure prediction (threshold calculation moved to ThresholdOptimizer).
"""

import logging
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# Feature columns used for prediction
FEATURE_COLUMNS = [
    'sigma_gradient',
    'linearity_error',
    'fail_points',
    'optimal_offset',
    'linearity_spec',
    'sigma_to_spec',
    'error_to_spec',
]


@dataclass
class PredictorConfig:
    """Configuration for model predictor."""
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 3
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    # Minimum samples required for training
    min_samples: int = 50


@dataclass
class PredictorMetrics:
    """Metrics from predictor training/evaluation."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: float = 0.0
    cv_mean: float = 0.0
    cv_std: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (excluding confusion matrix)."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'auc_roc': self.auc_roc,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std,
        }


@dataclass
class PredictorTrainingResult:
    """Result of predictor training."""
    success: bool
    model_name: str
    metrics: Optional[PredictorMetrics] = None
    n_samples: int = 0
    n_positive: int = 0  # Number of failures in training data
    n_negative: int = 0  # Number of passes in training data
    feature_importance: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class ModelPredictor:
    """
    Per-model failure probability predictor.

    Uses RandomForest to predict probability of failure based on trim features.
    Each product model (e.g., 6828, 8340) gets its own trained predictor.

    Features used:
    - sigma_gradient: Primary quality metric
    - linearity_error: Max deviation after offset
    - fail_points: Count of out-of-spec points (severity)
    - optimal_offset: Centering adjustment needed
    - linearity_spec: Specification tolerance
    - sigma_to_spec: Ratio of sigma to spec (normalized)
    - error_to_spec: Ratio of error to spec (normalized)

    Training data comes from:
    - Final Test results (primary ground truth when linked)
    - Trim file linearity pass/fail (secondary, always available)
    """

    def __init__(self, model_name: str, config: Optional[PredictorConfig] = None):
        """
        Initialize predictor for a specific product model.

        Args:
            model_name: Product model number (e.g., "6828", "8340-1")
            config: Optional configuration for hyperparameters
        """
        self.model_name = model_name
        self.config = config or PredictorConfig()

        # ML components
        self.classifier: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None

        # Training state
        self.is_trained: bool = False
        self.training_date: Optional[datetime] = None
        self.training_samples: int = 0
        self.metrics: Optional[PredictorMetrics] = None
        self.feature_importance: Dict[str, float] = {}

        # Feature statistics (for reference)
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        severity: Optional[pd.Series] = None
    ) -> PredictorTrainingResult:
        """
        Train the predictor on this model's data.

        Args:
            features: DataFrame with feature columns
            labels: Series with 1=failed, 0=passed
            severity: Optional Series with fail_points for sample weighting

        Returns:
            PredictorTrainingResult with metrics and status
        """
        try:
            # Validate input
            if len(features) < self.config.min_samples:
                return PredictorTrainingResult(
                    success=False,
                    model_name=self.model_name,
                    n_samples=len(features),
                    error=f"Need at least {self.config.min_samples} samples, got {len(features)}"
                )

            # Check for required columns
            available_features = [c for c in FEATURE_COLUMNS if c in features.columns]
            if len(available_features) < 3:
                return PredictorTrainingResult(
                    success=False,
                    model_name=self.model_name,
                    error=f"Need at least 3 feature columns, got {available_features}"
                )

            # Use only available features
            X = features[available_features].copy()
            y = labels.copy()

            # Handle missing values
            X = X.fillna(0)

            # Store feature statistics
            for col in X.columns:
                self.feature_means[col] = float(X[col].mean())
                self.feature_stds[col] = float(X[col].std())

            # Calculate sample weights from severity if provided
            sample_weight = None
            if severity is not None:
                # Higher weight for more severe failures
                # Normalize to [1, 3] range
                max_severity = severity.max() if severity.max() > 0 else 1
                sample_weight = 1 + 2 * (severity / max_severity)
                sample_weight = sample_weight.fillna(1)

            # Split data
            if sample_weight is not None:
                X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
                    X, y, sample_weight,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    stratify=y if y.nunique() > 1 else None
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    stratify=y if y.nunique() > 1 else None
                )
                sw_train = sw_test = None

            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train classifier
            self.classifier = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                class_weight='balanced',  # Handle imbalanced data
                random_state=self.config.random_state,
                n_jobs=-1
            )

            self.classifier.fit(X_train_scaled, y_train, sample_weight=sw_train)

            # Evaluate
            y_pred = self.classifier.predict(X_test_scaled)
            y_proba = self.classifier.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            metrics = PredictorMetrics(
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, zero_division=0),
                recall=recall_score(y_test, y_pred, zero_division=0),
                f1=f1_score(y_test, y_pred, zero_division=0),
                confusion_matrix=confusion_matrix(y_test, y_pred)
            )

            # AUC-ROC only if we have both classes
            if y_test.nunique() > 1:
                metrics.auc_roc = roc_auc_score(y_test, y_proba)

            # Cross-validation
            cv_scores = cross_val_score(
                self.classifier, X_train_scaled, y_train,
                cv=min(self.config.cv_folds, len(y_train) // 2),
                scoring='accuracy'
            )
            metrics.cv_mean = float(cv_scores.mean())
            metrics.cv_std = float(cv_scores.std())

            # Feature importance
            self.feature_importance = dict(zip(
                available_features,
                self.classifier.feature_importances_
            ))

            # Update state
            self.is_trained = True
            self.training_date = datetime.now()
            self.training_samples = len(X_train)
            self.metrics = metrics

            n_positive = int(y.sum())
            n_negative = len(y) - n_positive

            logger.info(
                f"ModelPredictor[{self.model_name}] trained - "
                f"Acc: {metrics.accuracy:.3f}, F1: {metrics.f1:.3f}, "
                f"AUC: {metrics.auc_roc:.3f}, Samples: {len(X_train)}"
            )

            return PredictorTrainingResult(
                success=True,
                model_name=self.model_name,
                metrics=metrics,
                n_samples=len(X_train),
                n_positive=n_positive,
                n_negative=n_negative,
                feature_importance=self.feature_importance
            )

        except Exception as e:
            logger.exception(f"Error training ModelPredictor[{self.model_name}]: {e}")
            return PredictorTrainingResult(
                success=False,
                model_name=self.model_name,
                error=str(e)
            )

    def predict_failure_probability(self, features: Dict[str, float]) -> Optional[float]:
        """
        Predict probability that this unit will fail.

        Args:
            features: Dict with feature values (sigma_gradient, linearity_error, etc.)

        Returns:
            Failure probability (0-1), or None if not trained
        """
        if not self.is_trained or self.classifier is None or self.scaler is None:
            return None

        try:
            # Only use features the model was trained on
            trained_features = list(self.feature_importance.keys())
            X = pd.DataFrame([{
                col: features.get(col, self.feature_means.get(col, 0))
                for col in trained_features
            }])

            X_scaled = self.scaler.transform(X)
            proba = self.classifier.predict_proba(X_scaled)[0, 1]

            return float(proba)

        except Exception as e:
            logger.warning(f"Prediction failed for {self.model_name}: {e}")
            return None

    def predict_with_confidence(
        self,
        features: Dict[str, float]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Predict failure probability with confidence interval.

        Uses tree ensemble to estimate uncertainty.

        Args:
            features: Dict with feature values

        Returns:
            Tuple of (probability, lower_bound, upper_bound), or (None, None, None)
        """
        if not self.is_trained or self.classifier is None or self.scaler is None:
            return None, None, None

        try:
            trained_features = list(self.feature_importance.keys())
            X = pd.DataFrame([{
                col: features.get(col, self.feature_means.get(col, 0))
                for col in trained_features
            }])
            X_scaled = self.scaler.transform(X)

            # Get predictions from all trees
            tree_predictions = np.array([
                tree.predict_proba(X_scaled)[0, 1]
                for tree in self.classifier.estimators_
            ])

            mean_proba = float(np.mean(tree_predictions))
            lower = float(np.percentile(tree_predictions, 5))
            upper = float(np.percentile(tree_predictions, 95))

            return mean_proba, lower, upper

        except Exception as e:
            logger.warning(f"Confidence prediction failed for {self.model_name}: {e}")
            return None, None, None

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        return self.feature_importance.copy()

    def save(self, path: Path) -> bool:
        """
        Save trained predictor to disk.

        Args:
            path: Path to save file

        Returns:
            True if successful
        """
        if not self.is_trained:
            logger.warning(f"Cannot save untrained predictor: {self.model_name}")
            return False

        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'model_name': self.model_name,
                'classifier': self.classifier,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'training_date': self.training_date,
                'training_samples': self.training_samples,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance,
                'feature_means': self.feature_means,
                'feature_stds': self.feature_stds,
                'config': self.config,
            }

            with open(path, 'wb') as f:
                pickle.dump(data, f)

            # Write hash file for integrity verification on load
            file_hash = self._compute_file_hash(path)
            hash_path = path.with_suffix('.hash')
            hash_path.write_text(file_hash)

            logger.debug(f"Predictor saved: {self.model_name} -> {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save predictor {self.model_name}: {e}")
            return False

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA256 hash of a file for integrity verification."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def load(self, path: Path) -> bool:
        """
        Load trained predictor from disk.

        Args:
            path: Path to load from

        Returns:
            True if successful
        """
        try:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Predictor file not found: {path}")
                return False

            # Verify file integrity before loading pickle
            hash_path = path.with_suffix('.hash')
            if hash_path.exists():
                expected_hash = hash_path.read_text().strip()
                actual_hash = self._compute_file_hash(path)
                if expected_hash != actual_hash:
                    logger.error(f"Predictor file integrity check failed: {path}")
                    logger.error("File may have been corrupted or tampered with")
                    return False

            with open(path, 'rb') as f:
                data = pickle.load(f)

            self.model_name = data['model_name']
            self.classifier = data['classifier']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']
            self.training_date = data['training_date']
            self.training_samples = data['training_samples']
            self.metrics = data['metrics']
            self.feature_importance = data['feature_importance']
            self.feature_means = data['feature_means']
            self.feature_stds = data['feature_stds']
            self.config = data.get('config', self.config)

            logger.debug(f"Predictor loaded: {self.model_name} <- {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load predictor from {path}: {e}")
            return False

    def get_state_dict(self) -> Dict[str, Any]:
        """Get state as dictionary for database storage."""
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_date': self.training_date,
            'training_samples': self.training_samples,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'feature_importance': self.feature_importance,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
        }


def extract_features(track_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract ML features from track data.

    This is the standard feature extraction used by both training and prediction.

    Args:
        track_data: Dict with track measurements (from TrackResult or analysis)

    Returns:
        Dict of feature name -> value
    """
    # Get raw values with defaults
    sigma_gradient = track_data.get('sigma_gradient', 0) or 0
    linearity_error = track_data.get('final_linearity_error_shifted') or track_data.get('linearity_error', 0) or 0
    fail_points = track_data.get('linearity_fail_points', 0) or 0
    optimal_offset = track_data.get('optimal_offset', 0) or 0
    linearity_spec = track_data.get('linearity_spec', 0.01) or 0.01

    # Calculate derived features
    sigma_to_spec = sigma_gradient / linearity_spec if linearity_spec else 0
    error_to_spec = abs(linearity_error) / linearity_spec if linearity_spec else 0

    return {
        'sigma_gradient': sigma_gradient,
        'linearity_error': abs(linearity_error),
        'fail_points': fail_points,
        'optimal_offset': abs(optimal_offset),
        'linearity_spec': linearity_spec,
        'sigma_to_spec': sigma_to_spec,
        'error_to_spec': error_to_spec,
    }
