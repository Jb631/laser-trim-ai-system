"""
Machine Learning Models for Laser Trim Analysis
=============================================

This module implements AI-powered analysis including:
- Threshold optimization based on historical data
- Failure prediction with high accuracy
- Manufacturing drift detection
- Feature importance analysis

Author: QA Specialist
Date: 2024
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support,
                             mean_absolute_error, r2_score)
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# Import from your existing modules
from core.config import Config
from core.data_processor import LaserTrimDataProcessor


class LaserTrimMLModels:
    """
    Machine Learning models for intelligent laser trim analysis.

    This class provides:
    - Adaptive threshold optimization
    - Failure prediction with explainable features
    - Manufacturing drift detection
    - Model persistence and versioning
    """

    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        """
        Initialize ML models with configuration.

        Args:
            config: Configuration object
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or self._setup_logger()

        # Model storage
        self.models = {
            'threshold_optimizer': None,
            'failure_predictor': None,
            'drift_detector': None,
            'feature_selector': None
        }

        # Scalers for data normalization
        self.scalers = {
            'threshold': StandardScaler(),
            'failure': RobustScaler(),  # Robust to outliers
            'drift': StandardScaler()
        }

        # Feature importance tracking
        self.feature_importance = {}

        # Model metadata
        self.model_metadata = {
            'version': '1.0',
            'last_trained': None,
            'training_samples': 0,
            'performance_metrics': {}
        }

        # Create models directory
        self.models_dir = Path(config.output_dir) / 'ml_models'
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("ML Models initialized")

    def _setup_logger(self) -> logging.Logger:
        """Set up logger if none provided."""
        logger = logging.getLogger('LaserTrimML')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def prepare_features(self, data: pd.DataFrame,
                         target_type: str = 'classification') -> pd.DataFrame:
        """
        Prepare and engineer features for ML models.

        Args:
            data: Raw data from data processor
            target_type: 'classification' or 'regression'

        Returns:
            Feature-engineered DataFrame
        """
        features = pd.DataFrame()

        # Basic measurements
        features['sigma_gradient'] = data.get('sigma_gradient', 0)
        features['linearity_spec'] = data.get('linearity_spec', 0)
        features['travel_length'] = data.get('travel_length', 0)
        features['unit_length'] = data.get('unit_length', 0)

        # Resistance features
        features['resistance_change'] = data.get('resistance_change', 0)
        features['resistance_change_percent'] = data.get('resistance_change_percent', 0)

        # Error statistics
        if 'error_data' in data:
            errors = np.array(data['error_data'])
            features['error_mean'] = np.mean(errors)
            features['error_std'] = np.std(errors)
            features['error_max'] = np.max(np.abs(errors))
            features['error_skew'] = self._calculate_skewness(errors)
            features['error_kurtosis'] = self._calculate_kurtosis(errors)

        # Calculated ratios (domain knowledge)
        features['sigma_to_spec_ratio'] = (
            features['sigma_gradient'] / features['linearity_spec']
            if features['linearity_spec'].any() else 0
        )

        features['length_ratio'] = (
            features['unit_length'] / features['travel_length']
            if features['travel_length'].any() else 1
        )

        # Time-based features if available
        if 'timestamp' in data:
            features['hour_of_day'] = pd.to_datetime(data['timestamp']).dt.hour
            features['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)

        # Model-specific features
        if 'model' in data:
            # One-hot encode model types
            model_dummies = pd.get_dummies(data['model'], prefix='model')
            features = pd.concat([features, model_dummies], axis=1)

        # Zone analysis features
        if 'zone_analysis' in data:
            features['worst_zone'] = data['zone_analysis'].get('worst_zone', 0)
            features['zone_variance'] = data['zone_analysis'].get('variance', 0)

        # Handle missing values
        features = features.fillna(0)

        # Remove infinite values
        features = features.replace([np.inf, -np.inf], 0)

        return features

    def train_threshold_optimizer(self, historical_data: pd.DataFrame,
                                  force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train model to optimize sigma thresholds based on historical pass/fail data.

        Args:
            historical_data: Historical analysis results
            force_retrain: Force retraining even if model exists

        Returns:
            Training results and metrics
        """
        self.logger.info("Training threshold optimizer...")

        # Check if we need to train
        if self.models['threshold_optimizer'] and not force_retrain:
            self.logger.info("Model already trained. Use force_retrain=True to retrain.")
            return self.model_metadata['performance_metrics'].get('threshold_optimizer', {})

        # Prepare features
        X = self.prepare_features(historical_data, 'regression')

        # Target: optimal threshold (we'll calculate this from historical data)
        y = self._calculate_optimal_thresholds(historical_data)

        # Remove any samples with invalid targets
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 10:
            self.logger.warning("Insufficient data for training threshold optimizer")
            return {'error': 'Insufficient data'}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scalers['threshold'].fit_transform(X_train)
        X_test_scaled = self.scalers['threshold'].transform(X_test)

        # Train Random Forest Regressor with hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rf = RandomForestRegressor(random_state=42, n_jobs=-1)

        # Use GridSearchCV for hyperparameter optimization
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=0
        )

        grid_search.fit(X_train_scaled, y_train)

        # Best model
        self.models['threshold_optimizer'] = grid_search.best_estimator_

        # Evaluate
        y_pred = self.models['threshold_optimizer'].predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Feature importance
        self.feature_importance['threshold_optimizer'] = dict(
            zip(X.columns, self.models['threshold_optimizer'].feature_importances_)
        )

        # Sort by importance
        self.feature_importance['threshold_optimizer'] = dict(
            sorted(self.feature_importance['threshold_optimizer'].items(),
                   key=lambda x: x[1], reverse=True)
        )

        results = {
            'mae': mae,
            'r2_score': r2,
            'best_params': grid_search.best_params_,
            'feature_importance': self.feature_importance['threshold_optimizer'],
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }

        # Update metadata
        self.model_metadata['performance_metrics']['threshold_optimizer'] = results
        self.model_metadata['last_trained'] = datetime.now().isoformat()
        self.model_metadata['training_samples'] = len(X)

        self.logger.info(f"Threshold optimizer trained - MAE: {mae:.4f}, R²: {r2:.4f}")

        return results

    def train_failure_predictor(self, historical_data: pd.DataFrame,
                                failure_window_days: int = 30,
                                target_accuracy: float = 0.90) -> Dict[str, Any]:
        """
        Train model to predict failures with high accuracy.

        Args:
            historical_data: Historical analysis results
            failure_window_days: Days to look ahead for failures
            target_accuracy: Target accuracy (default 90%)

        Returns:
            Training results and metrics
        """
        self.logger.info(f"Training failure predictor (target accuracy: {target_accuracy * 100}%)...")

        # Prepare features
        X = self.prepare_features(historical_data, 'classification')

        # Create failure labels (1 if failed within window, 0 otherwise)
        y = self._create_failure_labels(historical_data, failure_window_days)

        if len(np.unique(y)) < 2:
            self.logger.warning("Insufficient failure examples for training")
            return {'error': 'Insufficient failure examples'}

        # Handle class imbalance
        from sklearn.utils import class_weight
        classes = np.unique(y)
        weights = class_weight.compute_class_weight(
            'balanced', classes=classes, y=y
        )
        class_weights = dict(zip(classes, weights))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scalers['failure'].fit_transform(X_train)
        X_test_scaled = self.scalers['failure'].transform(X_test)

        # Feature selection for better interpretability
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=50, random_state=42),
            threshold='median'
        )
        selector.fit(X_train_scaled, y_train)
        X_train_selected = selector.transform(X_train_scaled)
        X_test_selected = selector.transform(X_test_scaled)

        # Store feature selector
        self.models['feature_selector'] = selector

        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()

        # Train model with hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }

        rf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight=class_weights
        )

        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=0
        )

        grid_search.fit(X_train_selected, y_train)

        # Best model
        self.models['failure_predictor'] = grid_search.best_estimator_

        # Evaluate
        y_pred = self.models['failure_predictor'].predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Feature importance (for selected features)
        feature_imp = dict(
            zip(selected_features,
                self.models['failure_predictor'].feature_importances_)
        )

        self.feature_importance['failure_predictor'] = dict(
            sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
        )

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'best_params': grid_search.best_params_,
            'selected_features': selected_features,
            'feature_importance': self.feature_importance['failure_predictor'],
            'target_accuracy_met': accuracy >= target_accuracy,
            'class_weights': class_weights
        }

        # Update metadata
        self.model_metadata['performance_metrics']['failure_predictor'] = results

        self.logger.info(f"Failure predictor trained - Accuracy: {accuracy:.2%}, "
                         f"Precision: {precision:.2%}, Recall: {recall:.2%}")

        if accuracy < target_accuracy:
            self.logger.warning(f"Target accuracy not met. Achieved: {accuracy:.2%}")

        return results

    def train_drift_detector(self, historical_data: pd.DataFrame,
                             contamination: float = 0.1) -> Dict[str, Any]:
        """
        Train anomaly detection model to identify manufacturing drift.

        Args:
            historical_data: Historical analysis results
            contamination: Expected proportion of outliers

        Returns:
            Training results
        """
        self.logger.info("Training manufacturing drift detector...")

        # Prepare features
        X = self.prepare_features(historical_data, 'anomaly')

        # Scale features
        X_scaled = self.scalers['drift'].fit_transform(X)

        # Train Isolation Forest for anomaly detection
        self.models['drift_detector'] = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )

        self.models['drift_detector'].fit(X_scaled)

        # Detect anomalies in training data
        anomaly_scores = self.models['drift_detector'].decision_function(X_scaled)
        anomaly_labels = self.models['drift_detector'].predict(X_scaled)

        # Calculate drift metrics
        n_anomalies = (anomaly_labels == -1).sum()
        anomaly_rate = n_anomalies / len(X)

        # Find feature contributions to anomalies
        # Compare normal vs anomaly feature distributions
        normal_mask = anomaly_labels == 1
        anomaly_mask = anomaly_labels == -1

        feature_drift = {}
        for col in X.columns:
            if X[normal_mask][col].std() > 0:  # Avoid division by zero
                drift_score = abs(
                    X[normal_mask][col].mean() - X[anomaly_mask][col].mean()
                ) / X[normal_mask][col].std()
                feature_drift[col] = drift_score

        # Sort by drift score
        feature_drift = dict(sorted(feature_drift.items(),
                                    key=lambda x: x[1], reverse=True))

        results = {
            'contamination': contamination,
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': anomaly_rate,
            'feature_drift_scores': feature_drift,
            'anomaly_threshold': np.percentile(anomaly_scores, contamination * 100)
        }

        # Update metadata
        self.model_metadata['performance_metrics']['drift_detector'] = results

        self.logger.info(f"Drift detector trained - Found {n_anomalies} anomalies "
                         f"({anomaly_rate:.1%} of data)")

        return results

    def predict_optimal_threshold(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict optimal threshold for given features.

        Args:
            features: Feature dictionary for prediction

        Returns:
            Prediction results with confidence
        """
        if not self.models['threshold_optimizer']:
            return {'error': 'Model not trained'}

        # Prepare features
        X = self.prepare_features(pd.DataFrame([features]), 'regression')
        X_scaled = self.scalers['threshold'].transform(X)

        # Predict
        threshold = self.models['threshold_optimizer'].predict(X_scaled)[0]

        # Get prediction confidence using tree variance
        tree_predictions = np.array([
            tree.predict(X_scaled)
            for tree in self.models['threshold_optimizer'].estimators_
        ])
        confidence = 1 - (tree_predictions.std() / tree_predictions.mean())

        # Get feature contributions
        feature_contributions = {}
        for i, (feature, importance) in enumerate(
                self.feature_importance['threshold_optimizer'].items()
        ):
            if i < 5:  # Top 5 features
                feature_contributions[feature] = {
                    'importance': importance,
                    'value': X[feature].values[0] if feature in X.columns else 0
                }

        return {
            'optimal_threshold': float(threshold),
            'confidence': float(confidence),
            'feature_contributions': feature_contributions
        }

    def predict_failure_probability(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict probability of failure for given features.

        Args:
            features: Feature dictionary for prediction

        Returns:
            Failure prediction with probability and risk factors
        """
        if not self.models['failure_predictor']:
            return {'error': 'Model not trained'}

        # Prepare features
        X = self.prepare_features(pd.DataFrame([features]), 'classification')
        X_scaled = self.scalers['failure'].transform(X)

        # Apply feature selection
        if self.models['feature_selector']:
            X_selected = self.models['feature_selector'].transform(X_scaled)
        else:
            X_selected = X_scaled

        # Predict
        failure_prob = self.models['failure_predictor'].predict_proba(X_selected)[0, 1]
        failure_prediction = self.models['failure_predictor'].predict(X_selected)[0]

        # Get risk factors (top contributing features)
        risk_factors = []
        for feature, importance in self.feature_importance['failure_predictor'].items():
            if feature in X.columns:
                value = X[feature].values[0]
                risk_factors.append({
                    'feature': feature,
                    'importance': importance,
                    'value': float(value)
                })

        # Sort by importance and take top 5
        risk_factors = sorted(risk_factors, key=lambda x: x['importance'], reverse=True)[:5]

        # Determine risk level
        if failure_prob >= 0.8:
            risk_level = 'CRITICAL'
        elif failure_prob >= 0.6:
            risk_level = 'HIGH'
        elif failure_prob >= 0.4:
            risk_level = 'MEDIUM'
        elif failure_prob >= 0.2:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'

        return {
            'failure_probability': float(failure_prob),
            'failure_prediction': bool(failure_prediction),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'confidence': float(np.max(
                self.models['failure_predictor'].predict_proba(X_selected)[0]
            ))
        }

    def detect_manufacturing_drift(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect if current production shows drift from normal.

        Args:
            features: Feature dictionary for detection

        Returns:
            Drift detection results
        """
        if not self.models['drift_detector']:
            return {'error': 'Model not trained'}

        # Prepare features
        X = self.prepare_features(pd.DataFrame([features]), 'anomaly')
        X_scaled = self.scalers['drift'].transform(X)

        # Detect
        anomaly_score = self.models['drift_detector'].decision_function(X_scaled)[0]
        is_anomaly = self.models['drift_detector'].predict(X_scaled)[0] == -1

        # Get drift indicators (features most different from normal)
        drift_indicators = []
        if 'feature_drift_scores' in self.model_metadata['performance_metrics'].get('drift_detector', {}):
            drift_scores = self.model_metadata['performance_metrics']['drift_detector']['feature_drift_scores']
            for feature in list(drift_scores.keys())[:5]:
                if feature in X.columns:
                    drift_indicators.append({
                        'feature': feature,
                        'drift_score': drift_scores[feature],
                        'current_value': float(X[feature].values[0])
                    })

        # Calculate severity
        threshold = self.model_metadata['performance_metrics'].get(
            'drift_detector', {}
        ).get('anomaly_threshold', 0)

        if anomaly_score < threshold:
            severity = 'HIGH'
        elif anomaly_score < threshold * 0.5:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'

        return {
            'is_drift': bool(is_anomaly),
            'anomaly_score': float(anomaly_score),
            'severity': severity if is_anomaly else 'NONE',
            'drift_indicators': drift_indicators,
            'recommendation': self._get_drift_recommendation(is_anomaly, severity, drift_indicators)
        }

    def save_models(self, version: Optional[str] = None) -> str:
        """
        Save all trained models and metadata.

        Args:
            version: Optional version string

        Returns:
            Path to saved models
        """
        if not version:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')

        save_dir = self.models_dir / f'models_v{version}'
        save_dir.mkdir(exist_ok=True)

        # Save models
        for name, model in self.models.items():
            if model is not None:
                model_path = save_dir / f'{name}.joblib'
                joblib.dump(model, model_path)
                self.logger.info(f"Saved {name} to {model_path}")

        # Save scalers
        scalers_path = save_dir / 'scalers.joblib'
        joblib.dump(self.scalers, scalers_path)

        # Save feature importance
        feature_imp_path = save_dir / 'feature_importance.json'
        with open(feature_imp_path, 'w') as f:
            json.dump(self.feature_importance, f, indent=2)

        # Save metadata
        self.model_metadata['version'] = version
        metadata_path = save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2, default=str)

        self.logger.info(f"All models saved to {save_dir}")

        return str(save_dir)

    def load_models(self, version: str) -> bool:
        """
        Load previously saved models.

        Args:
            version: Version to load

        Returns:
            Success status
        """
        load_dir = self.models_dir / f'models_v{version}'

        if not load_dir.exists():
            self.logger.error(f"Model version {version} not found")
            return False

        try:
            # Load models
            for name in self.models.keys():
                model_path = load_dir / f'{name}.joblib'
                if model_path.exists():
                    self.models[name] = joblib.load(model_path)
                    self.logger.info(f"Loaded {name}")

            # Load scalers
            scalers_path = load_dir / 'scalers.joblib'
            if scalers_path.exists():
                self.scalers = joblib.load(scalers_path)

            # Load feature importance
            feature_imp_path = load_dir / 'feature_importance.json'
            if feature_imp_path.exists():
                with open(feature_imp_path, 'r') as f:
                    self.feature_importance = json.load(f)

            # Load metadata
            metadata_path = load_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)

            self.logger.info(f"All models loaded from version {version}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            return False

    def get_feature_importance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive feature importance report.

        Returns:
            Feature importance analysis
        """
        report = {
            'threshold_optimizer': {},
            'failure_predictor': {},
            'summary': {}
        }

        # Threshold optimizer features
        if 'threshold_optimizer' in self.feature_importance:
            report['threshold_optimizer'] = {
                'top_features': list(self.feature_importance['threshold_optimizer'].items())[:10],
                'total_features': len(self.feature_importance['threshold_optimizer'])
            }

        # Failure predictor features
        if 'failure_predictor' in self.feature_importance:
            report['failure_predictor'] = {
                'top_features': list(self.feature_importance['failure_predictor'].items())[:10],
                'total_features': len(self.feature_importance['failure_predictor'])
            }

        # Combined importance
        all_features = {}
        for model_features in self.feature_importance.values():
            for feature, importance in model_features.items():
                if feature not in all_features:
                    all_features[feature] = 0
                all_features[feature] += importance

        # Average importance
        n_models = len([m for m in self.feature_importance.values() if m])
        if n_models > 0:
            for feature in all_features:
                all_features[feature] /= n_models

        report['summary'] = {
            'most_important_overall': sorted(all_features.items(),
                                             key=lambda x: x[1], reverse=True)[:10],
            'feature_categories': self._categorize_features(all_features)
        }

        return report

    # Helper methods

    def _calculate_optimal_thresholds(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate optimal thresholds from historical data."""
        # This is a simplified version - in practice, you'd use domain knowledge
        # For now, we'll use the threshold that would have caught 95% of failures

        thresholds = []
        for _, row in data.iterrows():
            if 'sigma_gradient' in row and 'passed' in row:
                # If failed, optimal threshold should be lower than gradient
                # If passed, current threshold is acceptable
                if row['passed']:
                    threshold = row.get('sigma_threshold', row['sigma_gradient'] * 1.2)
                else:
                    threshold = row['sigma_gradient'] * 0.9
                thresholds.append(threshold)
            else:
                thresholds.append(np.nan)

        return np.array(thresholds)

    def _create_failure_labels(self, data: pd.DataFrame,
                               window_days: int) -> np.ndarray:
        """Create failure labels based on future failures."""
        # Simplified version - in practice, you'd track actual failures
        # For now, we'll use high sigma gradient as proxy for failure

        labels = []
        for _, row in data.iterrows():
            # High risk if sigma gradient > threshold
            if 'sigma_gradient' in row and 'sigma_threshold' in row:
                risk_ratio = row['sigma_gradient'] / row['sigma_threshold']
                # Label as failure if risk ratio > 0.9
                labels.append(1 if risk_ratio > 0.9 else 0)
            else:
                labels.append(0)

        return np.array(labels)

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _categorize_features(self, features: Dict[str, float]) -> Dict[str, List[str]]:
        """Categorize features by type."""
        categories = {
            'measurements': [],
            'statistics': [],
            'ratios': [],
            'model_specific': [],
            'time_based': []
        }

        for feature in features:
            if any(x in feature for x in ['gradient', 'spec', 'length', 'resistance']):
                categories['measurements'].append(feature)
            elif any(x in feature for x in ['mean', 'std', 'max', 'skew', 'kurtosis']):
                categories['statistics'].append(feature)
            elif 'ratio' in feature or '/' in feature:
                categories['ratios'].append(feature)
            elif 'model_' in feature:
                categories['model_specific'].append(feature)
            elif any(x in feature for x in ['hour', 'day', 'week']):
                categories['time_based'].append(feature)

        return categories

    def _get_drift_recommendation(self, is_drift: bool, severity: str,
                                  indicators: List[Dict]) -> str:
        """Generate recommendation based on drift detection."""
        if not is_drift:
            return "Production parameters within normal range."

        recommendations = []

        if severity == 'HIGH':
            recommendations.append("URGENT: Significant manufacturing drift detected.")
            recommendations.append("Recommend immediate process inspection.")
        elif severity == 'MEDIUM':
            recommendations.append("WARNING: Moderate drift detected.")
            recommendations.append("Schedule process review within 24 hours.")
        else:
            recommendations.append("NOTICE: Minor drift detected.")
            recommendations.append("Monitor closely for trend continuation.")

        # Add specific recommendations based on indicators
        for indicator in indicators[:3]:
            feature = indicator['feature']
            if 'resistance' in feature:
                recommendations.append(f"Check resistance measurement equipment calibration.")
            elif 'sigma' in feature:
                recommendations.append(f"Review laser trim parameters and alignment.")
            elif 'error' in feature:
                recommendations.append(f"Inspect measurement system accuracy.")

        return " ".join(recommendations)


# Convenience functions for integration

def create_ml_models(config: Config, logger: Optional[logging.Logger] = None) -> LaserTrimMLModels:
    """
    Factory function to create ML models instance.

    Args:
        config: Configuration object
        logger: Optional logger

    Returns:
        LaserTrimMLModels instance
    """
    return LaserTrimMLModels(config, logger)


def train_all_models(ml_models: LaserTrimMLModels,
                     historical_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Train all models with historical data.

    Args:
        ml_models: ML models instance
        historical_data: Historical analysis data

    Returns:
        Combined training results
    """
    results = {}

    # Train threshold optimizer
    results['threshold_optimizer'] = ml_models.train_threshold_optimizer(historical_data)

    # Train failure predictor
    results['failure_predictor'] = ml_models.train_failure_predictor(historical_data)

    # Train drift detector
    results['drift_detector'] = ml_models.train_drift_detector(historical_data)

    # Save models
    version = ml_models.save_models()
    results['saved_version'] = version

    return results


# In ml_models.py, add continuous learning:

class ContinuousLearningMixin:
    """Mixin for continuous learning from historical data."""

    def update_from_history(self, days_back=90):
        """Update model with recent historical data."""
        if not hasattr(self, 'db_manager'):
            return

        # Get recent data
        df = self.db_manager.get_historical_data(days_back=days_back)

        if len(df) > 100:  # Minimum data required
            # Prepare features and labels
            X, y = self.prepare_training_data(df)

            # Incremental learning
            if hasattr(self.model, 'partial_fit'):
                self.model.partial_fit(X, y)
            else:
                # Retrain completely
                self.train(X, y)

            # Save updated model
            self.save_model()

            print(f"Model updated with {len(df)} historical records")