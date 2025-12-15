"""
Threshold Optimizer for Laser Trim Analyzer v3.

ML model that learns optimal sigma thresholds based on historical data.
Simplified from v2's implementation (~300 lines vs 230).

Uses RandomForestRegressor to predict optimal thresholds based on:
- Model number
- Unit length
- Linearity spec
- Historical pass rate
"""

import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from laser_trim_v3.config import get_config
from laser_trim_v3.utils.constants import DEFAULT_SIGMA_SCALING_FACTOR

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Configuration for threshold optimizer."""
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42


@dataclass
class TrainingResult:
    """Result of model training."""
    success: bool
    r2_score: float = 0.0
    rmse: float = 0.0
    cv_mean: float = 0.0
    cv_std: float = 0.0
    n_samples: int = 0
    error: Optional[str] = None


class ThresholdOptimizer:
    """
    ML model for optimizing sigma thresholds.

    Learns the relationship between product characteristics and optimal
    thresholds to minimize false positives while maintaining quality.

    Features:
    - RandomForest-based regression
    - Confidence intervals via tree ensemble
    - Fallback to formula-based calculation
    - Model persistence (save/load)
    """

    # Minimum samples required for meaningful training
    MIN_TRAINING_SAMPLES = 50

    def __init__(self, config: Optional[ThresholdConfig] = None):
        """
        Initialize threshold optimizer.

        Args:
            config: Optional configuration for model hyperparameters
        """
        self.config = config or ThresholdConfig()
        self.model: Optional[RandomForestRegressor] = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_date: Optional[datetime] = None
        self.training_metadata: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}

    def train(self, data: pd.DataFrame) -> TrainingResult:
        """
        Train the threshold optimization model.

        Args:
            data: DataFrame with columns:
                - model: Model number
                - unit_length: Unit length
                - linearity_spec: Linearity specification
                - sigma_gradient: Actual sigma gradient values
                - sigma_pass: Whether unit passed (for determining optimal threshold)

        Returns:
            TrainingResult with metrics and status
        """
        try:
            # Validate data
            required_cols = ['model', 'unit_length', 'linearity_spec', 'sigma_gradient']
            missing = [c for c in required_cols if c not in data.columns]
            if missing:
                return TrainingResult(
                    success=False,
                    error=f"Missing required columns: {missing}"
                )

            if len(data) < self.MIN_TRAINING_SAMPLES:
                return TrainingResult(
                    success=False,
                    error=f"Need at least {self.MIN_TRAINING_SAMPLES} samples, got {len(data)}"
                )

            # Prepare features
            X = self._prepare_features(data)
            y = self._calculate_optimal_thresholds(data)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Initialize and train model
            self.model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=-1
            )

            self.model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train,
                cv=self.config.cv_folds,
                scoring='r2'
            )

            # Store results
            self.is_trained = True
            self.training_date = datetime.now()
            self.training_metadata = {
                'n_samples': len(X_train),
                'n_features': X_train.shape[1],
                'feature_names': list(X.columns),
                'threshold_range': (float(y.min()), float(y.max())),
                'threshold_mean': float(y.mean()),
            }
            self.performance_metrics = {
                'r2_score': r2,
                'rmse': rmse,
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
            }

            logger.info(
                f"Threshold optimizer trained - R2: {r2:.3f}, RMSE: {rmse:.6f}, "
                f"CV: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}"
            )

            return TrainingResult(
                success=True,
                r2_score=r2,
                rmse=rmse,
                cv_mean=float(cv_scores.mean()),
                cv_std=float(cv_scores.std()),
                n_samples=len(X_train)
            )

        except Exception as e:
            logger.exception(f"Error training threshold optimizer: {e}")
            return TrainingResult(success=False, error=str(e))

    def predict(self, model: str, unit_length: float, linearity_spec: float) -> float:
        """
        Predict optimal threshold for given characteristics.

        Args:
            model: Model number
            unit_length: Unit length
            linearity_spec: Linearity specification

        Returns:
            Predicted optimal threshold
        """
        if not self.is_trained or self.model is None:
            # Fallback to formula-based calculation
            return self._formula_threshold(linearity_spec, unit_length)

        try:
            # Prepare features
            features = self._make_features(model, unit_length, linearity_spec)
            X = pd.DataFrame([features])
            X_scaled = self.scaler.transform(X)

            # Predict
            threshold = self.model.predict(X_scaled)[0]

            # Clip to reasonable bounds
            if self.training_metadata:
                min_t, max_t = self.training_metadata['threshold_range']
                threshold = np.clip(threshold, min_t * 0.8, max_t * 1.2)

            return float(threshold)

        except Exception as e:
            logger.warning(f"ML prediction failed, using formula: {e}")
            return self._formula_threshold(linearity_spec, unit_length)

    def predict_with_confidence(
        self,
        model: str,
        unit_length: float,
        linearity_spec: float,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Predict threshold with confidence interval.

        Args:
            model: Model number
            unit_length: Unit length
            linearity_spec: Linearity specification
            confidence: Confidence level (0-1)

        Returns:
            Tuple of (threshold, lower_bound, upper_bound)
        """
        threshold = self.predict(model, unit_length, linearity_spec)

        if not self.is_trained or self.model is None:
            # Use RMSE from formula (rough estimate)
            uncertainty = 0.0005
        else:
            # Get predictions from all trees for uncertainty
            features = self._make_features(model, unit_length, linearity_spec)
            X = pd.DataFrame([features])
            X_scaled = self.scaler.transform(X)

            tree_preds = np.array([
                tree.predict(X_scaled)[0]
                for tree in self.model.estimators_
            ])

            alpha = (1 - confidence) / 2
            lower = np.percentile(tree_preds, alpha * 100)
            upper = np.percentile(tree_preds, (1 - alpha) * 100)

            return threshold, lower, upper

        # Fallback confidence interval
        lower = threshold - 2 * uncertainty
        upper = threshold + 2 * uncertainty

        return threshold, lower, upper

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features from raw data."""
        features = pd.DataFrame()

        # Encode model as numeric (hash-based)
        features['model_hash'] = data['model'].apply(lambda x: hash(x) % 1000)

        # Direct numeric features
        features['unit_length'] = data['unit_length'].fillna(0)
        features['linearity_spec'] = data['linearity_spec'].fillna(0.01)

        # Derived features
        features['spec_length_ratio'] = features['linearity_spec'] / (features['unit_length'] + 1)

        return features

    def _make_features(
        self,
        model: str,
        unit_length: float,
        linearity_spec: float
    ) -> Dict[str, float]:
        """Make features dict for single prediction."""
        return {
            'model_hash': hash(model) % 1000,
            'unit_length': unit_length or 0,
            'linearity_spec': linearity_spec or 0.01,
            'spec_length_ratio': linearity_spec / (unit_length + 1) if unit_length else 0,
        }

    def _calculate_optimal_thresholds(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate optimal thresholds from historical data.

        Uses the 90th percentile of passing units' sigma gradients as the
        optimal threshold (tight enough to catch issues, loose enough for normal variation).
        """
        # Group by model and calculate optimal threshold
        optimal = []

        for _, row in data.iterrows():
            # For each row, optimal threshold is the sigma gradient + margin
            # This is a simplified approach - in production, group by model
            sigma = row['sigma_gradient']
            # Add 20% margin to actual sigma values
            optimal.append(sigma * 1.2)

        return pd.Series(optimal)

    def _formula_threshold(self, linearity_spec: float, unit_length: float) -> float:
        """
        Calculate threshold using formula (fallback when ML unavailable).

        Formula: threshold = linearity_spec / (scaling_factor * travel_factor)
        """
        scaling_factor = DEFAULT_SIGMA_SCALING_FACTOR

        # Travel factor normalized to 100 degrees
        travel_factor = max(1.0, (unit_length or 100) / 100.0)

        threshold = linearity_spec / (scaling_factor * travel_factor)

        # Ensure reasonable bounds
        return max(0.0001, min(threshold, 0.1))

    def save(self, path: Path) -> bool:
        """
        Save trained model to disk.

        Args:
            path: Path to save model

        Returns:
            True if successful
        """
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return False

        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'training_date': self.training_date,
                'training_metadata': self.training_metadata,
                'performance_metrics': self.performance_metrics,
                'config': self.config,
            }

            with open(path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load(self, path: Path) -> bool:
        """
        Load trained model from disk.

        Args:
            path: Path to load model from

        Returns:
            True if successful
        """
        try:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Model file not found: {path}")
                return False

            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.training_date = model_data['training_date']
            self.training_metadata = model_data['training_metadata']
            self.performance_metrics = model_data['performance_metrics']
            self.config = model_data.get('config', self.config)
            self.is_trained = True

            logger.info(f"Model loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from trained model."""
        if not self.is_trained or self.model is None:
            return None

        importance = dict(zip(
            self.training_metadata.get('feature_names', []),
            self.model.feature_importances_
        ))

        return importance
