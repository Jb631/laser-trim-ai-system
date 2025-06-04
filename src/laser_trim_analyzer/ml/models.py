"""
ML Models for Potentiometer QA

This module contains specific ML model implementations:
1. ThresholdOptimizer - Determines optimal sigma thresholds
2. FailurePredictor - Predictive failure analysis
3. DriftDetector - Manufacturing drift detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from scipy import stats
import logging
from typing import Dict, List, Any, Optional, Tuple

from laser_trim_analyzer.ml.engine import BaseMLModel, ModelConfig
from laser_trim_analyzer.utils.logging_utils import log_exception


class ThresholdOptimizer(BaseMLModel):
    """
    ML model for optimizing sigma thresholds based on historical data.

    This model learns the relationship between product characteristics and
    optimal thresholds to minimize false positives while maintaining quality.
    """

    def __init__(self, config: ModelConfig, logger: Optional[logging.Logger] = None):
        """Initialize threshold optimizer."""
        super().__init__(config, logger)
        self.scaler = StandardScaler()
        self.optimal_threshold_history = []

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train the threshold optimization model.

        Args:
            X: Features DataFrame
            y: Target values (optimal thresholds)

        Returns:
            Training results and metrics
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.training_params.get('test_size', 0.2),
                random_state=self.config.training_params.get('random_state', 42)
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Initialize model with hyperparameters
            self.model = RandomForestRegressor(
                n_estimators=self.config.hyperparameters.get('n_estimators', 100),
                max_depth=self.config.hyperparameters.get('max_depth', 10),
                min_samples_split=self.config.hyperparameters.get('min_samples_split', 5),
                min_samples_leaf=self.config.hyperparameters.get('min_samples_leaf', 2),
                random_state=self.config.training_params.get('random_state', 42),
                n_jobs=-1
            )

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Evaluate on test set
            y_pred = self.model.predict(X_test_scaled)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))

            # Cross-validation score
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train,
                cv=self.config.training_params.get('cv_folds', 5),
                scoring='r2'
            )

            # Store results
            self.is_trained = True
            self.training_metadata = {
                'n_samples': len(X_train),
                'n_features': X_train.shape[1],
                'feature_names': list(X.columns),
                'threshold_range': (float(y.min()), float(y.max())),
                'threshold_mean': float(y.mean()),
                'threshold_std': float(y.std())
            }

            self.performance_metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'mae': mae,
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std())
            }

            # Calculate optimal threshold distribution
            self._analyze_threshold_distribution(y_train, y_pred)

            self.logger.info(
                f"Threshold optimizer trained - R²: {r2:.3f}, RMSE: {rmse:.6f}, "
                f"CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
            )

            return self.performance_metrics

        except Exception as e:
            log_exception(self.logger, e, "Error training threshold optimizer")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict optimal thresholds for new data.

        Args:
            X: Features DataFrame

        Returns:
            Predicted optimal thresholds
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)

            # Apply bounds based on historical data
            if self.training_metadata:
                min_thresh, max_thresh = self.training_metadata['threshold_range']
                predictions = np.clip(predictions, min_thresh * 0.8, max_thresh * 1.2)

            return predictions

        except Exception as e:
            log_exception(self.logger, e, "Error in threshold prediction")
            raise

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X)

        metrics = {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2_score': r2_score(y, predictions),
            'mae': np.mean(np.abs(y - predictions)),
            'mape': np.mean(np.abs((y - predictions) / y)) * 100  # Mean Absolute Percentage Error
        }

        return metrics

    def recommend_threshold(self, model_features: Dict[str, float],
                            confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Recommend threshold with confidence interval.

        Args:
            model_features: Dictionary of model features
            confidence_level: Confidence level for interval

        Returns:
            Recommended threshold with confidence bounds
        """
        # Convert to DataFrame
        X = pd.DataFrame([model_features])

        # Get prediction
        threshold = self.predict(X)[0]

        # Estimate prediction interval using RF uncertainty
        if hasattr(self.model, 'estimators_'):
            # Get predictions from all trees
            all_predictions = np.array([
                tree.predict(self.scaler.transform(X))[0]
                for tree in self.model.estimators_
            ])

            # Calculate confidence interval
            lower = np.percentile(all_predictions, (1 - confidence_level) / 2 * 100)
            upper = np.percentile(all_predictions, (1 + confidence_level) / 2 * 100)
            uncertainty = np.std(all_predictions)
        else:
            # Fallback to simple interval
            uncertainty = self.performance_metrics.get('rmse', 0.001)
            lower = threshold - 2 * uncertainty
            upper = threshold + 2 * uncertainty

        return {
            'recommended_threshold': threshold,
            'confidence_interval': (lower, upper),
            'uncertainty': uncertainty,
            'confidence_level': confidence_level
        }

    def _analyze_threshold_distribution(self, y_true: pd.Series,
                                        y_pred: np.ndarray) -> None:
        """Analyze threshold distribution for insights."""
        # Ensure both arrays have the same length for broadcasting
        min_length = min(len(y_true), len(y_pred))
        y_true_aligned = y_true.iloc[:min_length] if isinstance(y_true, pd.Series) else y_true[:min_length]
        y_pred_aligned = y_pred[:min_length]
        
        residuals = y_true_aligned - y_pred_aligned

        # Store analysis results
        self.optimal_threshold_history.append({
            'mean_residual': float(residuals.mean()),
            'std_residual': float(residuals.std()),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals))
        })


class FailurePredictor(BaseMLModel):
    """
    ML model for predicting potential failures based on quality metrics.

    This model identifies units likely to fail early based on patterns
    in historical data.
    """

    def __init__(self, config: ModelConfig, logger: Optional[logging.Logger] = None):
        """Initialize failure predictor."""
        super().__init__(config, logger)
        self.scaler = StandardScaler()
        self.failure_patterns = {}

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train the failure prediction model.

        Args:
            X: Features DataFrame
            y: Target values (failure/no failure)

        Returns:
            Training results and metrics
        """
        try:
            # Handle class imbalance
            class_weights = self._calculate_class_weights(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.training_params.get('test_size', 0.2),
                random_state=self.config.training_params.get('random_state', 42),
                stratify=y
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Initialize model
            self.model = RandomForestClassifier(
                n_estimators=self.config.hyperparameters.get('n_estimators', 200),
                max_depth=self.config.hyperparameters.get('max_depth', 15),
                min_samples_split=self.config.hyperparameters.get('min_samples_split', 10),
                min_samples_leaf=self.config.hyperparameters.get('min_samples_leaf', 5),
                class_weight=class_weights,
                random_state=self.config.training_params.get('random_state', 42),
                n_jobs=-1
            )

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train,
                cv=self.config.training_params.get('cv_folds', 5),
                scoring='f1'
            )

            # Analyze failure patterns
            self._analyze_failure_patterns(X_train, y_train)

            # Store results
            self.is_trained = True
            self.training_metadata = {
                'n_samples': len(X_train),
                'n_features': X_train.shape[1],
                'feature_names': list(X.columns),
                'class_distribution': y_train.value_counts().to_dict(),
                'class_weights': class_weights
            }

            self.performance_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_f1_mean': float(cv_scores.mean()),
                'cv_f1_std': float(cv_scores.std())
            }

            self.logger.info(
                f"Failure predictor trained - Accuracy: {accuracy:.3f}, "
                f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}"
            )

            return self.performance_metrics

        except Exception as e:
            log_exception(self.logger, e, "Error training failure predictor")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict failure probability."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            return predictions

        except Exception as e:
            log_exception(self.logger, e, "Error in failure prediction")
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get failure probability scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)
            return probabilities[:, 1]  # Return probability of failure

        except Exception as e:
            log_exception(self.logger, e, "Error in failure probability prediction")
            raise

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1_score': f1_score(y, predictions, zero_division=0),
            'avg_failure_prob': float(probabilities.mean())
        }

        return metrics

    def get_risk_assessment(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Get comprehensive risk assessment for units.

        Args:
            features: Features DataFrame

        Returns:
            DataFrame with risk scores and categories
        """
        # Get failure probabilities
        failure_probs = self.predict_proba(features)

        # Determine risk categories
        risk_categories = pd.cut(
            failure_probs,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )

        # Create assessment DataFrame
        assessment = pd.DataFrame({
            'failure_probability': failure_probs,
            'risk_category': risk_categories,
            'risk_score': failure_probs * 100  # Convert to percentage
        })

        # Add feature contributions if available
        if hasattr(self.model, 'feature_importances_'):
            # Calculate feature contributions for high-risk units
            high_risk_mask = failure_probs > 0.7
            if high_risk_mask.any():
                contributions = self._calculate_feature_contributions(
                    features[high_risk_mask]
                )
                assessment.loc[high_risk_mask, 'top_risk_factors'] = contributions

        return assessment

    def _calculate_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        class_counts = y.value_counts()
        total = len(y)

        # Calculate balanced weights
        weights = {}
        for class_label in class_counts.index:
            weights[class_label] = total / (len(class_counts) * class_counts[class_label])

        return weights

    def _analyze_failure_patterns(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Analyze patterns in failure data."""
        # Get failed units
        failed_mask = y == 1
        failed_features = X[failed_mask]

        if len(failed_features) > 0:
            # Calculate statistics for failed units
            self.failure_patterns = {
                'mean_values': failed_features.mean().to_dict(),
                'std_values': failed_features.std().to_dict(),
                'correlation_with_failure': X.corrwith(y).to_dict()
            }

    def _calculate_feature_contributions(self, X: pd.DataFrame) -> List[str]:
        """Calculate top contributing features for predictions."""
        contributions = []

        for _, row in X.iterrows():
            # Get feature values and importances
            feature_values = row.values
            feature_importances = self.model.feature_importances_

            # Calculate weighted contributions
            weighted_contributions = feature_values * feature_importances

            # Get top 3 contributing features
            top_indices = np.argsort(weighted_contributions)[-3:][::-1]
            top_features = [self.config.features[i] for i in top_indices]

            contributions.append(', '.join(top_features))

        return contributions


class DriftDetector(BaseMLModel):
    """
    ML model for detecting manufacturing drift in potentiometer quality.

    This model identifies when the manufacturing process is drifting
    from normal operating conditions.
    """

    def __init__(self, config: ModelConfig, logger: Optional[logging.Logger] = None):
        """Initialize drift detector."""
        super().__init__(config, logger)
        self.scaler = StandardScaler()
        self.baseline_statistics = {}
        self.drift_history = []

    def train(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, Any]:
        """
        Train the drift detection model.

        Note: This is an unsupervised model, so y is not used but kept
        for interface compatibility.

        Args:
            X: Features DataFrame (normal operating data)
            y: Not used (included for interface compatibility)

        Returns:
            Training results and metrics
        """
        try:
            # Calculate baseline statistics
            self._calculate_baseline_statistics(X)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Initialize Isolation Forest for anomaly detection
            self.model = IsolationForest(
                n_estimators=self.config.hyperparameters.get('n_estimators', 100),
                contamination=self.config.hyperparameters.get('contamination', 0.05),
                max_samples=self.config.hyperparameters.get('max_samples', 'auto'),
                random_state=self.config.training_params.get('random_state', 42),
                n_jobs=-1
            )

            # Fit model
            self.model.fit(X_scaled)

            # Calculate anomaly scores on training data
            anomaly_scores = self.model.score_samples(X_scaled)
            predictions = self.model.predict(X_scaled)

            # Calculate drift threshold
            self.drift_threshold = np.percentile(anomaly_scores, 5)

            # Store results
            self.is_trained = True
            self.training_metadata = {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'feature_names': list(X.columns),
                'baseline_period': {
                    'start': X.index.min() if hasattr(X.index, 'min') else None,
                    'end': X.index.max() if hasattr(X.index, 'max') else None
                },
                'drift_threshold': float(self.drift_threshold),
                'contamination_rate': float((predictions == -1).mean())
            }

            self.performance_metrics = {
                'anomaly_score_mean': float(anomaly_scores.mean()),
                'anomaly_score_std': float(anomaly_scores.std()),
                'detected_anomalies': int((predictions == -1).sum()),
                'detection_rate': float((predictions == -1).mean())
            }

            self.logger.info(
                f"Drift detector trained - Baseline samples: {len(X)}, "
                f"Anomaly threshold: {self.drift_threshold:.3f}"
            )

            return self.performance_metrics

        except Exception as e:
            log_exception(self.logger, e, "Error training drift detector")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Detect drift in new data.

        Returns:
            Array of drift indicators (1 = drift detected, 0 = normal)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            X_scaled = self.scaler.transform(X)

            # Get anomaly predictions (-1 for anomaly, 1 for normal)
            predictions = self.model.predict(X_scaled)

            # Convert to binary (1 for drift, 0 for normal)
            drift_indicators = (predictions == -1).astype(int)

            return drift_indicators

        except Exception as e:
            log_exception(self.logger, e, "Error in drift detection")
            raise

    def evaluate(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, float]:
        """
        Evaluate drift detection performance.

        Since this is unsupervised, we calculate drift statistics.
        """
        drift_indicators = self.predict(X)
        anomaly_scores = self.get_anomaly_scores(X)

        metrics = {
            'drift_rate': float(drift_indicators.mean()),
            'anomaly_score_mean': float(anomaly_scores.mean()),
            'anomaly_score_std': float(anomaly_scores.std()),
            'samples_analyzed': len(X)
        }

        # Calculate statistical drift metrics
        if self.baseline_statistics:
            stat_drift = self._calculate_statistical_drift(X)
            metrics.update(stat_drift)

        return metrics

    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores for samples."""
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")

        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)

    def analyze_drift(self, X: pd.DataFrame, window_size: int = 100) -> Dict[str, Any]:
        """
        Analyze drift patterns in time-series data.

        Args:
            X: Features DataFrame (should be time-ordered)
            window_size: Window size for rolling analysis

        Returns:
            Comprehensive drift analysis
        """
        # Get anomaly scores
        anomaly_scores = self.get_anomaly_scores(X)
        drift_indicators = self.predict(X)

        # Calculate rolling statistics
        rolling_drift_rate = pd.Series(drift_indicators).rolling(
            window=window_size, min_periods=1
        ).mean()

        rolling_anomaly_mean = pd.Series(anomaly_scores).rolling(
            window=window_size, min_periods=1
        ).mean()

        # Detect drift periods
        drift_periods = self._detect_drift_periods(rolling_drift_rate)

        # Statistical tests
        stat_tests = self._perform_statistical_tests(X)

        # Feature-level drift
        feature_drift = self._analyze_feature_drift(X)

        analysis = {
            'overall_drift_rate': float(drift_indicators.mean()),
            'drift_trend': self._determine_drift_trend(rolling_drift_rate),
            'drift_periods': drift_periods,
            'statistical_tests': stat_tests,
            'feature_drift': feature_drift,
            'max_anomaly_score': float(anomaly_scores.max()),
            'current_drift_rate': float(rolling_drift_rate.iloc[-1]) if len(rolling_drift_rate) > 0 else 0
        }

        # Store in history
        self.drift_history.append({
            'timestamp': pd.Timestamp.now(),
            'drift_rate': analysis['overall_drift_rate'],
            'n_samples': len(X)
        })

        return analysis

    def get_drift_report(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive drift report."""
        analysis = self.analyze_drift(X)

        report = {
            'summary': {
                'drift_detected': analysis['overall_drift_rate'] > 0.1,
                'drift_severity': self._classify_drift_severity(analysis['overall_drift_rate']),
                'trend': analysis['drift_trend'],
                'action_required': analysis['overall_drift_rate'] > 0.15
            },
            'details': analysis,
            'recommendations': self._generate_drift_recommendations(analysis),
            'affected_features': [
                feat for feat, drift in analysis['feature_drift'].items()
                if drift['is_drifting']
            ]
        }

        return report

    def _calculate_baseline_statistics(self, X: pd.DataFrame) -> None:
        """Calculate baseline statistics for drift detection."""
        self.baseline_statistics = {
            col: {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'median': X[col].median(),
                'q1': X[col].quantile(0.25),
                'q3': X[col].quantile(0.75)
            }
            for col in X.columns
        }

    def _calculate_statistical_drift(self, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical drift metrics."""
        drift_metrics = {}

        for col in X.columns:
            if col in self.baseline_statistics:
                baseline = self.baseline_statistics[col]
                current_mean = X[col].mean()
                current_std = X[col].std()

                # Calculate drift magnitude
                mean_drift = abs(current_mean - baseline['mean']) / (baseline['std'] + 1e-6)
                std_drift = abs(current_std - baseline['std']) / (baseline['std'] + 1e-6)

                drift_metrics[f'{col}_mean_drift'] = mean_drift
                drift_metrics[f'{col}_std_drift'] = std_drift

        return drift_metrics

    def _detect_drift_periods(self, rolling_drift_rate: pd.Series,
                              threshold: float = 0.15) -> List[Dict[str, Any]]:
        """Detect continuous drift periods."""
        drift_periods = []
        in_drift = False
        start_idx = None

        for i, rate in enumerate(rolling_drift_rate):
            if rate > threshold and not in_drift:
                in_drift = True
                start_idx = i
            elif rate <= threshold and in_drift:
                in_drift = False
                drift_periods.append({
                    'start': start_idx,
                    'end': i,
                    'duration': i - start_idx,
                    'max_drift_rate': float(rolling_drift_rate[start_idx:i].max())
                })

        # Handle ongoing drift
        if in_drift:
            drift_periods.append({
                'start': start_idx,
                'end': len(rolling_drift_rate) - 1,
                'duration': len(rolling_drift_rate) - start_idx,
                'max_drift_rate': float(rolling_drift_rate[start_idx:].max()),
                'ongoing': True
            })

        return drift_periods

    def _perform_statistical_tests(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical tests for drift."""
        tests = {}

        for col in X.columns:
            if col in self.baseline_statistics:
                # Kolmogorov-Smirnov test
                baseline_mean = self.baseline_statistics[col]['mean']
                baseline_std = self.baseline_statistics[col]['std']

                # Generate baseline sample for comparison
                baseline_sample = np.random.normal(
                    baseline_mean, baseline_std, size=min(1000, len(X))
                )

                ks_stat, ks_pvalue = stats.ks_2samp(X[col].values, baseline_sample)

                tests[col] = {
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pvalue),
                    'significant_drift': ks_pvalue < 0.05
                }

        return tests

    def _analyze_feature_drift(self, X: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze drift at feature level."""
        feature_drift = {}

        for col in X.columns:
            if col in self.baseline_statistics:
                baseline = self.baseline_statistics[col]

                # Calculate drift metrics
                mean_change = (X[col].mean() - baseline['mean']) / baseline['mean'] * 100
                std_change = (X[col].std() - baseline['std']) / baseline['std'] * 100

                # Check if distribution shifted
                is_drifting = (abs(mean_change) > 10 or abs(std_change) > 20)

                feature_drift[col] = {
                    'mean_change_percent': mean_change,
                    'std_change_percent': std_change,
                    'is_drifting': is_drifting,
                    'direction': 'increasing' if mean_change > 0 else 'decreasing'
                }

        return feature_drift

    def _determine_drift_trend(self, rolling_drift_rate: pd.Series) -> str:
        """Determine overall drift trend."""
        if len(rolling_drift_rate) < 10:
            return 'insufficient_data'

        # Fit linear trend
        x = np.arange(len(rolling_drift_rate))
        slope, _ = np.polyfit(x, rolling_drift_rate, 1)

        if abs(slope) < 0.0001:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'

    """
    ML Models for Potentiometer QA (Part 2 - Continued)

    Additional model implementations and utilities
    """

    # Continue the DriftDetector class
    def _classify_drift_severity(self, drift_rate: float) -> str:
        """Classify drift severity."""
        if drift_rate < 0.05:
            return 'negligible'
        elif drift_rate < 0.10:
            return 'low'
        elif drift_rate < 0.15:
            return 'moderate'
        elif drift_rate < 0.25:
            return 'high'
        else:
            return 'critical'

    def _generate_drift_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on drift analysis."""
        recommendations = []

        drift_rate = analysis['overall_drift_rate']
        drift_trend = analysis['drift_trend']
        feature_drift = analysis['feature_drift']

        # Overall drift recommendations
        if drift_rate > 0.15:
            recommendations.append(
                "URGENT: Significant manufacturing drift detected. Immediate investigation required.")
            recommendations.append("Consider halting production until root cause is identified.")
        elif drift_rate > 0.10:
            recommendations.append("WARNING: Moderate drift detected. Schedule maintenance check.")
            recommendations.append("Increase monitoring frequency.")

        # Trend-based recommendations
        if drift_trend == 'increasing':
            recommendations.append("Drift is increasing over time. Implement corrective actions soon.")
        elif drift_trend == 'decreasing':
            recommendations.append("Drift is decreasing. Continue monitoring to ensure trend continues.")

        # Feature-specific recommendations
        drifting_features = [f for f, d in feature_drift.items() if d['is_drifting']]
        if drifting_features:
            recommendations.append(f"Focus on these drifting parameters: {', '.join(drifting_features[:3])}")

            # Specific recommendations based on feature patterns
            for feature in drifting_features[:3]:
                drift_info = feature_drift[feature]
                direction = drift_info['direction']

                if 'resistance' in feature.lower():
                    if direction == 'increasing':
                        recommendations.append(f"Check material composition - {feature} is trending higher")
                    else:
                        recommendations.append(f"Verify material thickness - {feature} is trending lower")

                elif 'sigma' in feature.lower():
                    recommendations.append(f"Review laser calibration - {feature} showing {direction} trend")

                elif 'linearity' in feature.lower():
                    recommendations.append(f"Inspect trim patterns - {feature} degradation detected")

        # Statistical test recommendations
        stat_tests = analysis.get('statistical_tests', {})
        significant_features = [f for f, t in stat_tests.items() if t.get('significant_drift', False)]
        if len(significant_features) > 3:
            recommendations.append("Multiple parameters show statistical drift. Comprehensive process review needed.")

        return recommendations


class ModelEnsemble(BaseMLModel):
    """
    Ensemble model that combines predictions from multiple models.

    This provides more robust predictions by leveraging the strengths
    of different model types.
    """

    def __init__(self, models: List[BaseMLModel], config: ModelConfig,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize ensemble model.

        Args:
            models: List of trained models to ensemble
            config: Model configuration
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.models = models
        self.weights = None
        self.combination_method = config.hyperparameters.get('combination_method', 'weighted_average')

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train the ensemble (optimize weights).

        Args:
            X: Features DataFrame
            y: Target values

        Returns:
            Training results
        """
        try:
            # Ensure all models are trained
            for model in self.models:
                if not model.is_trained:
                    raise ValueError(f"Model {model.__class__.__name__} must be trained before ensemble")

            # Split data for weight optimization
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Get predictions from each model
            predictions = []
            for model in self.models:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_val)
                else:
                    pred = model.predict(X_val)
                predictions.append(pred)

            predictions = np.array(predictions).T

            # Optimize weights
            if self.combination_method == 'weighted_average':
                self.weights = self._optimize_weights(predictions, y_val)
            elif self.combination_method == 'voting':
                self.weights = np.ones(len(self.models)) / len(self.models)
            else:
                raise ValueError(f"Unknown combination method: {self.combination_method}")

            # Calculate ensemble performance
            ensemble_pred = self._combine_predictions(predictions)

            self.is_trained = True
            self.training_metadata = {
                'n_models': len(self.models),
                'model_types': [m.__class__.__name__ for m in self.models],
                'weights': self.weights.tolist(),
                'combination_method': self.combination_method
            }

            # Calculate metrics based on problem type
            if len(np.unique(y)) == 2:  # Classification
                self.performance_metrics = {
                    'accuracy': accuracy_score(y_val, ensemble_pred > 0.5),
                    'auc_roc': roc_auc_score(y_val, ensemble_pred)
                }
            else:  # Regression
                self.performance_metrics = {
                    'rmse': np.sqrt(mean_squared_error(y_val, ensemble_pred)),
                    'r2': r2_score(y_val, ensemble_pred)
                }

            return self.performance_metrics

        except Exception as e:
            log_exception(self.logger, e, "Error training ensemble")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")

        predictions = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions).T
        return self._combine_predictions(predictions)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        predictions = self.predict(X)

        # Determine if classification or regression
        if len(np.unique(y)) == 2:
            return {
                'accuracy': accuracy_score(y, predictions > 0.5),
                'precision': precision_score(y, predictions > 0.5),
                'recall': recall_score(y, predictions > 0.5),
                'f1': f1_score(y, predictions > 0.5)
            }
        else:
            return {
                'rmse': np.sqrt(mean_squared_error(y, predictions)),
                'mae': mean_absolute_error(y, predictions),
                'r2': r2_score(y, predictions)
            }

    def _optimize_weights(self, predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Optimize ensemble weights using scipy optimization."""
        from scipy.optimize import minimize

        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / weights.sum()
            # Calculate weighted prediction
            ensemble_pred = np.dot(predictions, weights)
            # Return negative correlation (to maximize)
            return -np.corrcoef(ensemble_pred, y_true)[0, 1]

        # Initial weights (equal)
        initial_weights = np.ones(predictions.shape[1]) / predictions.shape[1]

        # Constraints: weights sum to 1, all weights >= 0
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1) for _ in range(predictions.shape[1])]

        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        return result.x

    def _combine_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Combine predictions using specified method."""
        if self.combination_method == 'weighted_average':
            return np.dot(predictions, self.weights)
        elif self.combination_method == 'voting':
            return np.mean(predictions, axis=1)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")


class AdaptiveThresholdOptimizer(ThresholdOptimizer):
    """
    Enhanced threshold optimizer that adapts to changing conditions.

    This model continuously learns and adjusts thresholds based on
    feedback from production outcomes.
    """

    def __init__(self, config: ModelConfig, logger: Optional[logging.Logger] = None):
        """Initialize adaptive threshold optimizer."""
        super().__init__(config, logger)
        self.adaptation_history = []
        self.feedback_buffer = []
        self.adaptation_rate = config.hyperparameters.get('adaptation_rate', 0.1)

    def update_with_feedback(self, features: pd.DataFrame,
                             actual_outcomes: pd.Series,
                             predicted_thresholds: np.ndarray) -> None:
        """
        Update model based on production feedback.

        Args:
            features: Features used for prediction
            actual_outcomes: Actual pass/fail outcomes
            predicted_thresholds: Thresholds that were used
        """
        # Store feedback
        self.feedback_buffer.append({
            'features': features,
            'outcomes': actual_outcomes,
            'thresholds': predicted_thresholds,
            'timestamp': pd.Timestamp.now()
        })

        # Trigger adaptation if buffer is large enough
        if len(self.feedback_buffer) >= self.config.training_params.get('adaptation_batch_size', 50):
            self._adapt_model()

    def _adapt_model(self) -> None:
        """Adapt model based on accumulated feedback."""
        # Combine feedback data
        all_features = pd.concat([fb['features'] for fb in self.feedback_buffer])
        all_outcomes = pd.concat([fb['outcomes'] for fb in self.feedback_buffer])
        all_thresholds = np.concatenate([fb['thresholds'] for fb in self.feedback_buffer])

        # Calculate optimal thresholds based on outcomes
        optimal_thresholds = self._calculate_optimal_thresholds(
            all_features, all_outcomes, all_thresholds
        )

        # Create incremental training data
        X_adapt = all_features
        y_adapt = optimal_thresholds

        # Incremental update (partial fit)
        if hasattr(self.model, 'partial_fit'):
            X_scaled = self.scaler.transform(X_adapt)
            self.model.partial_fit(X_scaled, y_adapt)
        else:
            # For models without partial_fit, retrain with combined data
            self._retrain_with_adaptation(X_adapt, y_adapt)

        # Record adaptation
        self.adaptation_history.append({
            'timestamp': pd.Timestamp.now(),
            'n_samples': len(all_features),
            'avg_threshold_change': float(np.mean(np.abs(optimal_thresholds - all_thresholds))),
            'performance_before': self._calculate_performance(all_outcomes, all_thresholds),
            'performance_after': self._calculate_performance(all_outcomes, optimal_thresholds)
        })

        # Clear feedback buffer
        self.feedback_buffer = []

        self.logger.info(f"Model adapted with {len(all_features)} feedback samples")

    def _calculate_optimal_thresholds(self, features: pd.DataFrame,
                                      outcomes: pd.Series,
                                      used_thresholds: np.ndarray) -> np.ndarray:
        """Calculate optimal thresholds based on actual outcomes."""
        optimal_thresholds = used_thresholds.copy()

        # Group by outcome
        passed_mask = outcomes == 1
        failed_mask = ~passed_mask

        # For passed units, we can potentially increase threshold (reduce false positives)
        if passed_mask.any():
            # Find units that passed with margin
            margins = features.loc[passed_mask, 'sigma_gradient'] / used_thresholds[passed_mask]
            high_margin_mask = margins < 0.8  # 20% margin

            # Increase threshold for high-margin passes
            adjustment = self.adaptation_rate * (1 - margins[high_margin_mask])
            optimal_thresholds[passed_mask][high_margin_mask] *= (1 + adjustment)

        # For failed units, we may need to decrease threshold
        if failed_mask.any():
            # Find units that failed but were close to threshold
            ratios = features.loc[failed_mask, 'sigma_gradient'] / used_thresholds[failed_mask]
            close_fail_mask = ratios < 1.2  # Failed by less than 20%

            # Decrease threshold for close failures
            adjustment = self.adaptation_rate * (ratios[close_fail_mask] - 1)
            optimal_thresholds[failed_mask][close_fail_mask] *= (1 - adjustment)

        return optimal_thresholds

    def _calculate_performance(self, outcomes: pd.Series,
                               thresholds: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics for given thresholds."""
        try:
            # Calculate basic metrics
            pass_rate = float(outcomes.mean()) if len(outcomes) > 0 else 0.0
            threshold_mean = float(thresholds.mean()) if len(thresholds) > 0 else 0.0
            threshold_std = float(thresholds.std()) if len(thresholds) > 0 else 0.0
            
            # Calculate threshold effectiveness
            threshold_variance = threshold_std / threshold_mean if threshold_mean > 0 else 0.0
            
            # Calculate consistency metrics
            threshold_range = float(np.max(thresholds) - np.min(thresholds)) if len(thresholds) > 0 else 0.0
            threshold_cv = threshold_std / threshold_mean if threshold_mean > 0 else 0.0
            
            return {
                'pass_rate': pass_rate,
                'threshold_mean': threshold_mean,
                'threshold_std': threshold_std,
                'threshold_variance': threshold_variance,
                'threshold_range': threshold_range,
                'threshold_cv': threshold_cv,
                'sample_count': len(outcomes)
            }
            
        except Exception as e:
            # Fallback to basic metrics if calculation fails
            return {
                'pass_rate': float(outcomes.mean()) if len(outcomes) > 0 else 0.0,
                'threshold_mean': float(thresholds.mean()) if len(thresholds) > 0 else 0.0,
                'threshold_std': float(thresholds.std()) if len(thresholds) > 0 else 0.0,
                'sample_count': len(outcomes)
            }

    def _retrain_with_adaptation(self, X_new: pd.DataFrame, y_new: pd.Series) -> None:
        """Retrain model with new and historical data."""
        # This would ideally combine new data with historical training data
        # For now, we'll do a weighted combination

        # Get predictions on new data as a baseline
        baseline_predictions = self.predict(X_new)

        # Weighted average of baseline and new targets
        weighted_targets = (1 - self.adaptation_rate) * baseline_predictions + \
                           self.adaptation_rate * y_new

        # Retrain on new data with weighted targets
        self.train(X_new, weighted_targets)


class AnomalyScorer:
    """
    Utility class for calculating various anomaly scores.

    Used by multiple models to identify unusual patterns.
    """

    @staticmethod
    def mahalanobis_distance(X: np.ndarray, mean: np.ndarray,
                             cov_inv: np.ndarray) -> np.ndarray:
        """Calculate Mahalanobis distance for multivariate anomaly detection."""
        diff = X - mean
        return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    @staticmethod
    def local_outlier_factor(X: np.ndarray, k: int = 20) -> np.ndarray:
        """Calculate Local Outlier Factor scores."""
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=k, contamination='auto')
        lof.fit(X)
        return -lof.negative_outlier_factor_

    @staticmethod
    def reconstruction_error(X: np.ndarray, model) -> np.ndarray:
        """Calculate reconstruction error for autoencoder-based detection."""
        X_reconstructed = model.predict(X)
        return np.mean((X - X_reconstructed) ** 2, axis=1)


# Model factory for easy instantiation
class ModelFactory:
    """Factory class for creating ML models with appropriate configurations."""

    @staticmethod
    def create_threshold_optimizer(adaptive: bool = False, **kwargs) -> ThresholdOptimizer:
        """Create threshold optimizer model."""
        config = ModelConfig({
            'model_type': 'threshold_optimizer',
            'version': '1.0',
            'features': ['sigma_gradient', 'linearity_spec', 'travel_length',
                      'unit_length', 'resistance_change_percent'],
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                **kwargs
            }
        })

        if adaptive:
            return AdaptiveThresholdOptimizer(config)
        return ThresholdOptimizer(config)

    @staticmethod
    def create_failure_predictor(**kwargs) -> FailurePredictor:
        """Create failure predictor model."""
        config = ModelConfig({
            'model_type': 'failure_predictor',
            'version': '1.0',
            'features': ['sigma_gradient', 'sigma_threshold', 'linearity_pass',
                      'resistance_change_percent', 'trim_improvement_percent',
                      'failure_probability', 'worst_zone'],
            'hyperparameters': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                **kwargs
            }
        })

        return FailurePredictor(config)

    @staticmethod
    def create_drift_detector(**kwargs) -> DriftDetector:
        """Create drift detector model."""
        config = ModelConfig({
            'model_type': 'drift_detector',
            'version': '1.0',
            'features': ['sigma_gradient', 'linearity_spec', 'resistance_change_percent',
                      'travel_length', 'unit_length'],
            'hyperparameters': {
                'n_estimators': 100,
                'contamination': 0.05,
                'max_samples': 'auto',
                **kwargs
            }
        })

        return DriftDetector(config)

    @staticmethod
    def create_ensemble(models: List[BaseMLModel],
                        combination_method: str = 'weighted_average',
                        **kwargs) -> ModelEnsemble:
        """Create ensemble model."""
        config = ModelConfig({
            'model_type': 'ensemble',
            'version': '1.0',
            'features': [],  # Inherited from component models
            'hyperparameters': {
                'combination_method': combination_method,
                **kwargs
            }
        })

        return ModelEnsemble(models, config)