"""
ML Engine Module - Core ML Infrastructure for Potentiometer QA

This module provides a clean, professional ML system with:
- Centralized ML operations management
- Model versioning and storage
- Automated retraining pipeline
- Feature engineering pipeline
- Performance tracking
"""

import os
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union
)
from abc import ABC, abstractmethod
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic_core.core_schema import none_schema
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from laser_trim_analyzer.utils.logging_utils import setup_logger, log_exception


class ModelConfig:
    """Configuration class for ML models."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize model configuration."""
        if config_dict is None:
            config_dict = {}

        # Default configurations
        self.model_type = config_dict.get('model_type', 'threshold_optimizer')
        self.version = config_dict.get('version', '1.0.0')
        self.features = config_dict.get('features', [])
        self.target = config_dict.get('target', None)
        self.hyperparameters = config_dict.get('hyperparameters', {})
        self.training_params = config_dict.get('training_params', {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5
        })
        self.retraining_criteria = config_dict.get('retraining_criteria', {
            'min_samples': 1000,
            'max_days_since_training': 30,
            'performance_threshold': 0.85
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_type': self.model_type,
            'version': self.version,
            'features': self.features,
            'target': self.target,
            'hyperparameters': self.hyperparameters,
            'training_params': self.training_params,
            'retraining_criteria': self.retraining_criteria
        }

    @classmethod
    def from_file(cls, filepath: str) -> 'ModelConfig':
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)

    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class BaseMLModel(ABC):
    """Abstract base class for all ML models in the system."""

    def __init__(self, config: ModelConfig, logger: Optional[logging.Logger] = None):
        """Initialize base ML model."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.is_trained = False
        self.training_metadata = {}
        self.performance_metrics = {}

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance. Must be implemented by subclasses."""
        pass

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': self.config.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        return None

    def save(self, filepath: str) -> None:
        """Save model and metadata."""
        model_data = {
            'model': self.model,
            'config': self.config.to_dict(),
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata,
            'performance_metrics': self.performance_metrics
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model and metadata."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.config = ModelConfig(model_data['config'])
        self.is_trained = model_data['is_trained']
        self.training_metadata = model_data['training_metadata']
        self.performance_metrics = model_data['performance_metrics']
        self.logger.info(f"Model loaded from {filepath}")


class FeatureEngineering:
    """Centralized feature engineering for potentiometer QA data."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize feature engineering."""
        self.logger = logger or logging.getLogger(__name__)
        self.feature_stats = {}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw data.

        Args:
            df: Raw data DataFrame

        Returns:
            DataFrame with engineered features
        """
        df_features = df.copy()

        try:
            # Time-based features
            if 'timestamp' in df.columns:
                df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
                df_features['hour'] = df_features['timestamp'].dt.hour
                df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
                df_features['month'] = df_features['timestamp'].dt.month
                df_features['quarter'] = df_features['timestamp'].dt.quarter

            # Ratio features
            if 'sigma_gradient' in df.columns and 'sigma_threshold' in df.columns:
                df_features['sigma_ratio'] = df_features['sigma_gradient'] / df_features['sigma_threshold']
                df_features['sigma_margin'] = 1 - df_features['sigma_ratio']
                df_features['sigma_margin_percent'] = df_features['sigma_margin'] * 100

            # Resistance features
            if 'untrimmed_resistance' in df.columns and 'trimmed_resistance' in df.columns:
                df_features['resistance_change_abs'] = abs(
                    df_features['trimmed_resistance'] - df_features['untrimmed_resistance']
                )
                df_features['resistance_stability'] = 1 - (
                        df_features['resistance_change_abs'] / df_features['untrimmed_resistance']
                )

            # Specification compliance features
            if 'linearity_spec' in df.columns and 'final_linearity_error_shifted' in df.columns:
                df_features['linearity_margin'] = (
                        df_features['linearity_spec'] - df_features['final_linearity_error_shifted']
                )
                df_features['linearity_compliance_ratio'] = (
                        df_features['final_linearity_error_shifted'] / df_features['linearity_spec']
                )

            # Travel efficiency features
            if 'travel_length' in df.columns and 'unit_length' in df.columns:
                df_features['travel_efficiency'] = df_features['travel_length'] / df_features['unit_length']

            # Composite quality score
            quality_components = []
            if 'sigma_margin' in df_features.columns:
                quality_components.append(df_features['sigma_margin'])
            if 'resistance_stability' in df_features.columns:
                quality_components.append(df_features['resistance_stability'])
            if 'linearity_margin' in df_features.columns:
                quality_components.append(df_features['linearity_margin'] / df_features['linearity_spec'])

            if quality_components:
                df_features['composite_quality_score'] = np.mean(quality_components, axis=0)

            # Log feature statistics
            self._calculate_feature_stats(df_features)

            self.logger.info(f"Created {len(df_features.columns)} features from {len(df.columns)} original columns")

        except Exception as e:
            log_exception(self.logger, e, "Error in feature engineering")

        return df_features

    def _calculate_feature_stats(self, df: pd.DataFrame) -> None:
        """Calculate and store feature statistics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        self.feature_stats = {
            col: {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'null_count': df[col].isnull().sum(),
                'null_percent': df[col].isnull().sum() / len(df) * 100
            }
            for col in numeric_cols
        }

    def normalize_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize features using stored statistics.

        Args:
            df: DataFrame with features
            method: Normalization method ('standard', 'minmax')

        Returns:
            Normalized DataFrame
        """
        df_norm = df.copy()

        for col in df.select_dtypes(include=[np.number]).columns:
            if col in self.feature_stats:
                stats = self.feature_stats[col]

                if method == 'standard':
                    if stats['std'] > 0:
                        df_norm[col] = (df[col] - stats['mean']) / stats['std']
                elif method == 'minmax':
                    range_val = stats['max'] - stats['min']
                    if range_val > 0:
                        df_norm[col] = (df[col] - stats['min']) / range_val

        return df_norm

    def get_feature_importance_analysis(self, model: BaseMLModel) -> pd.DataFrame:
        """Analyze feature importance across the pipeline."""
        importance_df = model.get_feature_importance()

        if importance_df is not None and self.feature_stats:
            # Add feature statistics
            importance_df['null_percent'] = importance_df['feature'].map(
                lambda x: self.feature_stats.get(x, {}).get('null_percent', 0)
            )
            importance_df['std'] = importance_df['feature'].map(
                lambda x: self.feature_stats.get(x, {}).get('std', 0)
            )

            # Calculate feature quality score
            importance_df['quality_score'] = (
                    importance_df['importance'] * (1 - importance_df['null_percent'] / 100)
            )

        return importance_df


class ModelVersionControl:
    """Manages model versions and storage."""

    def __init__(self, base_path: str, logger: Optional[logging.Logger] = None):
        """Initialize model version control."""
        self.base_path = Path(base_path)
        self.logger = logger or logging.getLogger(__name__)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.models_path = self.base_path / 'models'
        self.configs_path = self.base_path / 'configs'
        self.metadata_path = self.base_path / 'metadata'

        for path in [self.models_path, self.configs_path, self.metadata_path]:
            path.mkdir(exist_ok=True)

    def save_model(self, model: BaseMLModel, model_name: str,
                   version: Optional[str] = None) -> str:
        """
        Save model with versioning.

        Args:
            model: Model to save
            model_name: Name of the model
            version: Version string (auto-generated if None)

        Returns:
            Version identifier
        """
        if version is None:
            version = self._generate_version(model_name)

        # Create model directory
        model_dir = self.models_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = model_dir / 'model.pkl'
        model.save(str(model_file))

        # Save config
        config_file = self.configs_path / model_name / f'{version}.json'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        model.config.save(str(config_file))

        # Save metadata
        metadata = {
            'model_name': model_name,
            'version': version,
            'saved_at': datetime.now().isoformat(),
            'training_metadata': model.training_metadata,
            'performance_metrics': model.performance_metrics,
            'config_hash': self._hash_config(model.config.to_dict())
        }

        metadata_file = self.metadata_path / model_name / f'{version}.json'
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved model {model_name} version {version}")
        return version

    def load_model(self, model_class: type, model_name: str,
                   version: Optional[str] = None) -> BaseMLModel:
        """
        Load model by name and version.

        Args:
            model_class: Class of the model to load
            model_name: Name of the model
            version: Version to load (latest if None)

        Returns:
            Loaded model instance
        """
        if version is None:
            version = self.get_latest_version(model_name)

        if version is None:
            raise ValueError(f"No versions found for model {model_name}")

        # Load config
        config_file = self.configs_path / model_name / f'{version}.json'
        config = ModelConfig.from_file(str(config_file))

        # Create model instance
        model = model_class(config, self.logger)

        # Load model state
        model_file = self.models_path / model_name / version / 'model.pkl'
        model.load(str(model_file))

        self.logger.info(f"Loaded model {model_name} version {version}")
        return model

    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest version of a model."""
        model_path = self.models_path / model_name
        if not model_path.exists():
            return None

        versions = [d.name for d in model_path.iterdir() if d.is_dir()]
        if not versions:
            return None

        # Sort by semantic versioning
        versions.sort(key=lambda v: tuple(map(int, v.split('.'))))
        return versions[-1]

    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model with metadata."""
        metadata_path = self.metadata_path / model_name
        if not metadata_path.exists():
            return []

        versions = []
        for metadata_file in metadata_path.glob('*.json'):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            versions.append(metadata)

        # Sort by saved date
        versions.sort(key=lambda x: x['saved_at'], reverse=True)
        return versions

    def _generate_version(self, model_name: str) -> str:
        """Generate next version number."""
        latest = self.get_latest_version(model_name)

        if latest is None:
            return "1.0.0"

        major, minor, patch = map(int, latest.split('.'))
        return f"{major}.{minor}.{patch + 1}"

    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate hash of configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


class MLEngine:
    """
    Central ML Engine that manages all ML operations for potentiometer QA.

    This class provides:
    - Centralized model management
    - Automated retraining pipeline
    - Performance tracking
    - Feature engineering pipeline
    - Model deployment and serving
    """

    def __init__(self, data_path: str, models_path: str,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize ML Engine.

        Args:
            data_path: Path to data storage
            models_path: Path to model storage
            logger: Optional logger instance
        """
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.logger = logger or setup_logger(str(self.models_path / 'logs'))

        # Initialize components
        self.feature_engineering = FeatureEngineering(self.logger)
        self.version_control = ModelVersionControl(str(self.models_path), self.logger)

        # Model registry
        self.models: Dict[str, BaseMLModel] = {}
        self.model_configs: Dict[str, ModelConfig] = {}

        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}

        # Retraining scheduler
        self.retraining_schedule: Dict[str, datetime] = {}

        self.logger.info("ML Engine initialized")

    def register_model(self, model_name: str, model_class: type,
                       config: ModelConfig) -> None:
        """
        Register a model with the engine.

        Args:
            model_name: Unique name for the model
            model_class: Model class (must inherit from BaseMLModel)
            config: Model configuration
        """
        if not issubclass(model_class, BaseMLModel):
            raise ValueError(f"{model_class} must inherit from BaseMLModel")

        self.model_configs[model_name] = config
        self.logger.info(f"Registered model: {model_name}")

    def train_model(self, model_name: str, model_class: type,
                    data: pd.DataFrame, save: bool = True) -> BaseMLModel:
        """
        Train a model with feature engineering.

        Args:
            model_name: Name of the model to train
            model_class: Model class to instantiate
            data: Training data
            save: Whether to save the model after training

        Returns:
            Trained model instance
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not registered")

        config = self.model_configs[model_name]

        # Feature engineering
        self.logger.info(f"Applying feature engineering for {model_name}")
        data_features = self.feature_engineering.create_features(data)

        # Prepare training data
        feature_cols = [col for col in config.features if col in data_features.columns]
        if not feature_cols:
            raise ValueError(f"No valid features found for {model_name}")

        X = data_features[feature_cols]
        y = data_features[config.target] if config.target and config.target in data_features.columns else None

        # For supervised models, target is required
        if config.target and y is None:
            raise ValueError(f"Target column {config.target} not found")

        # Create and train model
        model = model_class(config, self.logger)

        self.logger.info(f"Training {model_name} with {len(X)} samples")
        training_results = model.train(X, y)

        # Track performance
        self._track_performance(model_name, training_results)

        # Save model if requested
        if save:
            version = self.version_control.save_model(model, model_name)
            self.logger.info(f"Saved {model_name} as version {version}")

        # Update model registry
        self.models[model_name] = model

        # Update retraining schedule
        self.retraining_schedule[model_name] = datetime.now() + timedelta(
            days=config.retraining_criteria.get('max_days_since_training', 30)
        )

        return model

    def predict(self, model_name: str, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a registered model.

        Args:
            model_name: Name of the model to use
            data: Input data

        Returns:
            Predictions array
        """
        if model_name not in self.models:
            # Try to load latest version
            self.logger.info(f"Loading latest version of {model_name}")
            model_class = self._get_model_class(model_name)
            self.models[model_name] = self.version_control.load_model(
                model_class, model_name
            )

        model = self.models[model_name]

        # Apply feature engineering
        data_features = self.feature_engineering.create_features(data)

        # Select features
        feature_cols = [col for col in model.config.features if col in data_features.columns]
        X = data_features[feature_cols]

        # Make predictions
        predictions = model.predict(X)

        return predictions

    def evaluate_model(self, model_name: str, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            model_name: Name of the model to evaluate
            test_data: Test dataset

        Returns:
            Performance metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        model = self.models[model_name]

        # Apply feature engineering
        data_features = self.feature_engineering.create_features(test_data)

        # Prepare test data
        feature_cols = [col for col in model.config.features if col in data_features.columns]
        X_test = data_features[feature_cols]
        y_test = data_features[model.config.target]

        # Evaluate
        metrics = model.evaluate(X_test, y_test)

        # Track performance
        self._track_performance(model_name, metrics, is_evaluation=True)

        return metrics

    def check_retraining_needed(self, model_name: str,
                                current_data: pd.DataFrame) -> bool:
        """
        Check if model needs retraining based on criteria.

        Args:
            model_name: Name of the model
            current_data: Current data for comparison

        Returns:
            True if retraining is needed
        """
        if model_name not in self.models and model_name not in self.model_configs:
            return False

        config = self.model_configs.get(model_name)
        if not config:
            return False

        criteria = config.retraining_criteria

        # Check time since last training
        if model_name in self.retraining_schedule:
            if datetime.now() > self.retraining_schedule[model_name]:
                self.logger.info(f"{model_name} scheduled for retraining (time-based)")
                return True

        # Check data volume
        if len(current_data) >= criteria.get('min_samples', 1000):
            # Check performance degradation
            if model_name in self.models:
                metrics = self.evaluate_model(model_name, current_data)

                # Use appropriate metric based on model type
                performance_metric = metrics.get('accuracy', metrics.get('r2_score', 0))

                if performance_metric < criteria.get('performance_threshold', 0.85):
                    self.logger.info(
                        f"{model_name} needs retraining (performance: {performance_metric:.3f})"
                    )
                    return True

        return False

    def automated_retraining_pipeline(self, data_source: callable) -> Dict[str, Any]:
        """
        Run automated retraining pipeline for all registered models.

        Args:
            data_source: Callable that returns current DataFrame

        Returns:
            Summary of retraining results
        """
        results = {}

        # Get current data
        current_data = data_source()

        for model_name, config in self.model_configs.items():
            try:
                if self.check_retraining_needed(model_name, current_data):
                    self.logger.info(f"Retraining {model_name}")

                    # Get model class
                    model_class = self._get_model_class(model_name)

                    # Retrain model
                    model = self.train_model(model_name, model_class, current_data)

                    results[model_name] = {
                        'status': 'retrained',
                        'performance': model.performance_metrics,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    results[model_name] = {
                        'status': 'no_retraining_needed',
                        'timestamp': datetime.now().isoformat()
                    }

            except Exception as e:
                log_exception(self.logger, e, f"Error in retraining pipeline for {model_name}")
                results[model_name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }

        return results

    def get_model_performance_history(self, model_name: str) -> pd.DataFrame:
        """Get performance history for a model."""
        if model_name not in self.performance_history:
            return pd.DataFrame()

        return pd.DataFrame(self.performance_history[model_name])

    def generate_model_report(self, model_name: str) -> Dict[str, Any]:
        """Generate comprehensive report for a model."""
        report = {
            'model_name': model_name,
            'generated_at': datetime.now().isoformat()
        }

        # Current version info
        if model_name in self.models:
            model = self.models[model_name]
            report['current_version'] = model.config.version
            report['is_trained'] = model.is_trained
            report['training_metadata'] = model.training_metadata
            report['performance_metrics'] = model.performance_metrics

            # Feature importance
            feature_importance = self.feature_engineering.get_feature_importance_analysis(model)
            if feature_importance is not None:
                report['feature_importance'] = feature_importance.to_dict('records')

        # Version history
        report['version_history'] = self.version_control.list_versions(model_name)

        # Performance history
        perf_history = self.get_model_performance_history(model_name)
        if not perf_history.empty:
            report['performance_trend'] = {
                'mean_performance': perf_history.select_dtypes(include=[np.number]).mean().to_dict(),
                'std_performance': perf_history.select_dtypes(include=[np.number]).std().to_dict(),
                'recent_trend': 'improving' if len(perf_history) > 1 and
                                               perf_history.iloc[-1].get('accuracy', 0) >
                                               perf_history.iloc[-2].get('accuracy', 0) else 'stable'
            }

        return report

    def _track_performance(self, model_name: str, metrics: Dict[str, Any],
                           is_evaluation: bool = False) -> None:
        """Track model performance over time."""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []

        record = {
            'timestamp': datetime.now().isoformat(),
            'type': 'evaluation' if is_evaluation else 'training',
            **metrics
        }

        self.performance_history[model_name].append(record)

        # Keep only recent history (last 100 records)
        if len(self.performance_history[model_name]) > 100:
            self.performance_history[model_name] = self.performance_history[model_name][-100:]

    def _get_model_class(self, model_name: str) -> type:
        """Get model class by name."""
        # Import the actual model classes
        try:
            from laser_trim_analyzer.ml.models import (
                ThresholdOptimizer, 
                FailurePredictor, 
                DriftDetector,
                ModelEnsemble,
                AdaptiveThresholdOptimizer
            )
            
            # Map model names to their classes
            model_classes = {
                'threshold_optimizer': ThresholdOptimizer,
                'adaptive_threshold_optimizer': AdaptiveThresholdOptimizer,
                'failure_predictor': FailurePredictor,
                'drift_detector': DriftDetector,
                'ensemble': ModelEnsemble
            }
            
            return model_classes.get(model_name)
            
        except ImportError as e:
            self.logger.error(f"Failed to import model class for {model_name}: {e}")
            return None

    def save_engine_state(self) -> None:
        """Save the current state of the ML engine."""
        state = {
            'model_configs': {name: config.to_dict()
                              for name, config in self.model_configs.items()},
            'performance_history': self.performance_history,
            'retraining_schedule': {name: dt.isoformat()
                                    for name, dt in self.retraining_schedule.items()}
        }

        state_file = self.models_path / 'engine_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        self.logger.info("ML Engine state saved")

    def load_engine_state(self) -> None:
        """Load saved engine state."""
        state_file = self.models_path / 'engine_state.json'

        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Restore configs
            for name, config_dict in state.get('model_configs', {}).items():
                self.model_configs[name] = ModelConfig(config_dict)

            # Restore performance history
            self.performance_history = state.get('performance_history', {})

            # Restore retraining schedule
            for name, dt_str in state.get('retraining_schedule', {}).items():
                try:
                    if isinstance(dt_str, str):
                        self.retraining_schedule[name] = datetime.fromisoformat(dt_str)
                    elif isinstance(dt_str, datetime):
                        self.retraining_schedule[name] = dt_str
                    else:
                        # Skip invalid values
                        self.logger.warning(f"Invalid datetime value for {name}: {dt_str}")
                        continue
                except ValueError as e:
                    self.logger.warning(f"Failed to parse datetime for {name}: {dt_str} - {e}")
                    continue

            self.logger.info("ML Engine state loaded")
