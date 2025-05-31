"""
ML Predictor for Real-Time Analysis

This module provides real-time ML predictions during laser trim analysis.
It loads pre-trained models and provides fast inference for immediate feedback.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import threading
import time
from collections import deque
import warnings

warnings.filterwarnings('ignore')


class MLPredictor:
    """
    Real-time ML predictor for laser trim analysis.

    This class provides immediate predictions and recommendations during
    file processing, helping QA specialists catch issues early.
    """

    def __init__(self,
                 model_dir: str,
                 db_manager=None,
                 cache_size: int = 1000,
                 update_interval_hours: int = 24,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the ML predictor.

        Args:
            model_dir: Directory containing pre-trained models
            db_manager: Database manager for historical data
            cache_size: Number of predictions to cache
            update_interval_hours: How often to check for model updates
            logger: Optional logger instance
        """
        self.model_dir = model_dir
        self.db_manager = db_manager
        self.cache_size = cache_size
        self.update_interval_hours = update_interval_hours
        self.logger = logger or logging.getLogger(__name__)

        # Model storage
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.feature_stats = {}

        # Prediction cache (for speed)
        self.prediction_cache = {}
        self.cache_queue = deque(maxlen=cache_size)

        # Model update tracking
        self.last_update_check = datetime.now()
        self.model_versions = {}

        # Performance tracking
        self.prediction_times = deque(maxlen=100)
        self.anomaly_counts = {'total': 0, 'detected': 0}

        # Load models on startup
        self._load_models()

        # Start background model update thread
        self.update_thread = threading.Thread(target=self._model_update_loop, daemon=True)
        self.update_thread.start()

    def _load_models(self) -> None:
        """Load pre-trained models from disk."""
        try:
            # Failure prediction model
            failure_model_path = os.path.join(self.model_dir, 'failure_predictor.pkl')
            if os.path.exists(failure_model_path):
                with open(failure_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.models['failure_predictor'] = model_data['model']
                    self.scalers['failure_predictor'] = model_data['scaler']
                    self.model_versions['failure_predictor'] = model_data.get('version', '1.0')
                self.logger.info(f"Loaded failure predictor v{self.model_versions['failure_predictor']}")

            # Anomaly detection model
            anomaly_model_path = os.path.join(self.model_dir, 'anomaly_detector.pkl')
            if os.path.exists(anomaly_model_path):
                with open(anomaly_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.models['anomaly_detector'] = model_data['model']
                    self.scalers['anomaly_detector'] = model_data['scaler']
                    self.model_versions['anomaly_detector'] = model_data.get('version', '1.0')
                self.logger.info(f"Loaded anomaly detector v{self.model_versions['anomaly_detector']}")

            # Threshold optimizer model
            threshold_model_path = os.path.join(self.model_dir, 'threshold_optimizer.pkl')
            if os.path.exists(threshold_model_path):
                with open(threshold_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.models['threshold_optimizer'] = model_data['model']
                    self.scalers['threshold_optimizer'] = model_data['scaler']
                    self.model_versions['threshold_optimizer'] = model_data.get('version', '1.0')
                self.logger.info(f"Loaded threshold optimizer v{self.model_versions['threshold_optimizer']}")

            # Load feature statistics for handling missing values
            stats_path = os.path.join(self.model_dir, 'feature_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.feature_stats = json.load(f)
                self.logger.info("Loaded feature statistics")

            # Load recommended thresholds
            threshold_path = os.path.join(self.model_dir, 'recommended_thresholds.json')
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    self.thresholds = json.load(f)
                self.logger.info("Loaded recommended thresholds")

        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")

    def predict_real_time(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make real-time predictions during file processing.

        Args:
            analysis_data: Current analysis results

        Returns:
            Dictionary with predictions and recommendations
        """
        start_time = time.time()

        # Create cache key
        cache_key = self._create_cache_key(analysis_data)

        # Check cache first
        if cache_key in self.prediction_cache:
            self.logger.debug(f"Cache hit for {analysis_data.get('File', 'Unknown')}")
            return self.prediction_cache[cache_key]

        # Prepare features
        features = self._prepare_features(analysis_data)

        predictions = {
            'timestamp': datetime.now().isoformat(),
            'file': analysis_data.get('File', 'Unknown'),
            'model': analysis_data.get('Model', 'Unknown'),
            'predictions': {},
            'recommendations': [],
            'warnings': [],
            'historical_comparison': {}
        }

        # 1. Predict failure probability
        if 'failure_predictor' in self.models and features is not None:
            try:
                failure_prob = self._predict_failure(features)
                predictions['predictions']['failure_probability'] = failure_prob

                # Add warning if high risk
                if failure_prob > 0.7:
                    predictions['warnings'].append({
                        'type': 'HIGH_FAILURE_RISK',
                        'message': f'High failure probability: {failure_prob:.2%}',
                        'severity': 'critical'
                    })
                    predictions['recommendations'].append(
                        'Immediate inspection recommended - high failure risk detected'
                    )
            except Exception as e:
                self.logger.error(f"Error predicting failure: {str(e)}")

        # 2. Detect anomalies
        if 'anomaly_detector' in self.models and features is not None:
            try:
                is_anomaly, anomaly_score = self._detect_anomaly(features)
                predictions['predictions']['is_anomaly'] = is_anomaly
                predictions['predictions']['anomaly_score'] = anomaly_score

                if is_anomaly:
                    self.anomaly_counts['detected'] += 1
                    predictions['warnings'].append({
                        'type': 'ANOMALY_DETECTED',
                        'message': f'Unusual pattern detected (score: {anomaly_score:.3f})',
                        'severity': 'warning'
                    })
                    predictions['recommendations'].append(
                        'This unit shows unusual characteristics - verify measurement accuracy'
                    )

                self.anomaly_counts['total'] += 1

            except Exception as e:
                self.logger.error(f"Error detecting anomaly: {str(e)}")

        # 3. Suggest threshold adjustments
        threshold_suggestion = self._suggest_threshold(analysis_data)
        if threshold_suggestion:
            predictions['predictions']['suggested_threshold'] = threshold_suggestion['threshold']
            predictions['predictions']['threshold_confidence'] = threshold_suggestion['confidence']

            # Add recommendation if threshold adjustment needed
            current_threshold = analysis_data.get('Sigma Threshold', 0)
            if abs(threshold_suggestion['threshold'] - current_threshold) / current_threshold > 0.1:
                predictions['recommendations'].append(
                    f"Consider adjusting sigma threshold from {current_threshold:.4f} "
                    f"to {threshold_suggestion['threshold']:.4f} ({threshold_suggestion['change_percent']:.1f}% change)"
                )

        # 4. Compare to historical patterns
        if self.db_manager:
            historical_comparison = self._compare_to_history(analysis_data)
            predictions['historical_comparison'] = historical_comparison

            # Add warnings based on historical comparison
            if historical_comparison.get('deviation_from_mean', 0) > 2:
                predictions['warnings'].append({
                    'type': 'HISTORICAL_DEVIATION',
                    'message': f"Sigma gradient {historical_comparison['deviation_from_mean']:.1f} std devs from historical mean",
                    'severity': 'warning'
                })

        # 5. Generate quality score
        quality_score = self._calculate_quality_score(analysis_data, predictions)
        predictions['predictions']['quality_score'] = quality_score

        # Add to cache
        self.prediction_cache[cache_key] = predictions
        self.cache_queue.append(cache_key)

        # Clean old cache entries
        if len(self.cache_queue) >= self.cache_size:
            old_key = self.cache_queue.popleft()
            if old_key in self.prediction_cache:
                del self.prediction_cache[old_key]

        # Track prediction time
        prediction_time = time.time() - start_time
        self.prediction_times.append(prediction_time)
        predictions['prediction_time_ms'] = prediction_time * 1000

        return predictions

    def _prepare_features(self, analysis_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare features from analysis data, handling missing values."""
        try:
            # Define features to extract
            feature_names = [
                'Sigma Gradient', 'Sigma Threshold', 'Unit Length',
                'Travel Length', 'Linearity Spec', 'Resistance Change (%)',
                'Trim Improvement (%)', 'Final Linearity Error (Shifted)',
                'Failure Probability', 'Range Utilization (%)'
            ]

            features = []
            for feature in feature_names:
                value = analysis_data.get(feature)

                # Handle missing values
                if value is None or pd.isna(value):
                    # Use mean from feature statistics
                    if feature in self.feature_stats:
                        value = self.feature_stats[feature]['mean']
                    else:
                        value = 0.0

                features.append(float(value))

            return np.array(features).reshape(1, -1)

        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return None

    def _predict_failure(self, features: np.ndarray) -> float:
        """Predict failure probability."""
        model = self.models['failure_predictor']
        scaler = self.scalers['failure_predictor']

        # Scale features
        features_scaled = scaler.transform(features)

        # Get probability of failure (class 1)
        probabilities = model.predict_proba(features_scaled)
        return probabilities[0][1]  # Probability of failure class

    def _detect_anomaly(self, features: np.ndarray) -> Tuple[bool, float]:
        """Detect if the unit is an anomaly."""
        model = self.models['anomaly_detector']
        scaler = self.scalers['anomaly_detector']

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict anomaly (-1 for anomaly, 1 for normal)
        prediction = model.predict(features_scaled)[0]
        anomaly_score = model.score_samples(features_scaled)[0]

        is_anomaly = (prediction == -1)

        return is_anomaly, anomaly_score

    def _suggest_threshold(self, analysis_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Suggest optimal threshold based on current data."""
        model_name = analysis_data.get('Model', 'Unknown')

        # Check if we have pre-computed thresholds
        if model_name in self.thresholds:
            current_threshold = analysis_data.get('Sigma Threshold', 0)
            suggested_threshold = self.thresholds[model_name]['recommended']

            if current_threshold > 0:
                change_percent = ((suggested_threshold - current_threshold) / current_threshold) * 100
            else:
                change_percent = 0

            return {
                'threshold': suggested_threshold,
                'confidence': self.thresholds[model_name].get('confidence', 0.8),
                'change_percent': change_percent,
                'based_on_samples': self.thresholds[model_name].get('sample_count', 0)
            }

        # If no pre-computed threshold, use ML model if available
        if 'threshold_optimizer' in self.models:
            features = self._prepare_features(analysis_data)
            if features is not None:
                model = self.models['threshold_optimizer']
                scaler = self.scalers['threshold_optimizer']

                features_scaled = scaler.transform(features)
                suggested_threshold = model.predict(features_scaled)[0]

                current_threshold = analysis_data.get('Sigma Threshold', 0)
                if current_threshold > 0:
                    change_percent = ((suggested_threshold - current_threshold) / current_threshold) * 100
                else:
                    change_percent = 0

                return {
                    'threshold': suggested_threshold,
                    'confidence': 0.7,  # Lower confidence for ML prediction
                    'change_percent': change_percent,
                    'based_on_samples': 0
                }

        return None

    def _compare_to_history(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current unit to historical patterns."""
        try:
            model = analysis_data.get('Model', 'Unknown')

            # Get historical statistics from database
            stats = self.db_manager.get_model_statistics(model)

            if not stats or stats['total_tracks'] == 0:
                return {'no_history': True}

            current_sigma = analysis_data.get('Sigma Gradient', 0)
            hist_mean = stats['avg_sigma_gradient']
            hist_std = stats.get('std_sigma_gradient', 0)

            # Calculate deviation from historical mean
            if hist_std > 0:
                deviation = (current_sigma - hist_mean) / hist_std
            else:
                deviation = 0

            # Determine percentile
            # Simplified - in practice, you'd calculate actual percentile
            if deviation <= -2:
                percentile = 2.5
            elif deviation <= -1:
                percentile = 16
            elif deviation <= 0:
                percentile = 50
            elif deviation <= 1:
                percentile = 84
            elif deviation <= 2:
                percentile = 97.5
            else:
                percentile = 99

            return {
                'historical_mean': hist_mean,
                'historical_std': hist_std,
                'deviation_from_mean': deviation,
                'percentile': percentile,
                'total_historical_samples': stats['total_tracks'],
                'historical_pass_rate': stats.get('sigma_pass_rate', 0) * 100
            }

        except Exception as e:
            self.logger.error(f"Error comparing to history: {str(e)}")
            return {'error': str(e)}

    def _calculate_quality_score(self,
                                 analysis_data: Dict[str, Any],
                                 predictions: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0

        # Deduct points for failures
        if not analysis_data.get('Sigma Pass', True):
            score -= 30
        if not analysis_data.get('Linearity Pass', True):
            score -= 20

        # Deduct for high failure probability
        failure_prob = predictions['predictions'].get('failure_probability', 0)
        score -= failure_prob * 20

        # Deduct for anomaly
        if predictions['predictions'].get('is_anomaly', False):
            score -= 10

        # Deduct for historical deviation
        deviation = predictions['historical_comparison'].get('deviation_from_mean', 0)
        score -= min(abs(deviation) * 5, 20)

        # Ensure score is between 0 and 100
        return max(0, min(100, score))

    def _create_cache_key(self, analysis_data: Dict[str, Any]) -> str:
        """Create a cache key from analysis data."""
        # Use key metrics that uniquely identify the analysis
        key_parts = [
            analysis_data.get('File', ''),
            str(analysis_data.get('Model', '')),
            str(analysis_data.get('Serial', '')),
            str(round(analysis_data.get('Sigma Gradient', 0), 6)),
            str(round(analysis_data.get('Sigma Threshold', 0), 6))
        ]
        return '|'.join(key_parts)

    def _model_update_loop(self) -> None:
        """Background thread to check for model updates."""
        while True:
            try:
                time.sleep(3600)  # Check every hour

                # Check if it's time to update
                if (datetime.now() - self.last_update_check).total_seconds() > self.update_interval_hours * 3600:
                    self.check_for_model_updates()
                    self.last_update_check = datetime.now()

            except Exception as e:
                self.logger.error(f"Error in model update loop: {str(e)}")

    def check_for_model_updates(self) -> None:
        """Check for and load updated models."""
        self.logger.info("Checking for model updates...")

        # Check each model file's modification time
        models_updated = False

        for model_name in ['failure_predictor', 'anomaly_detector', 'threshold_optimizer']:
            model_path = os.path.join(self.model_dir, f'{model_name}.pkl')

            if os.path.exists(model_path):
                # Check if file has been modified
                mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))

                if model_name not in self.model_versions or \
                        mod_time > self.last_update_check:
                    self.logger.info(f"Updating {model_name}...")
                    models_updated = True

        if models_updated:
            # Clear cache when models are updated
            self.prediction_cache.clear()
            self.cache_queue.clear()

            # Reload models
            self._load_models()
            self.logger.info("Models updated successfully")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get predictor performance statistics."""
        avg_prediction_time = np.mean(self.prediction_times) if self.prediction_times else 0

        return {
            'average_prediction_time_ms': avg_prediction_time * 1000,
            'cache_hit_rate': len(self.prediction_cache) / max(1, len(self.cache_queue)),
            'anomaly_detection_rate': self.anomaly_counts['detected'] / max(1, self.anomaly_counts['total']),
            'total_predictions': self.anomaly_counts['total'],
            'models_loaded': list(self.models.keys()),
            'model_versions': self.model_versions
        }

    def train_online(self, analysis_data: Dict[str, Any], actual_outcome: str) -> None:
        """
        Update models with new data (online learning).

        Args:
            analysis_data: The analysis results
            actual_outcome: Actual outcome (e.g., 'failed', 'passed')
        """
        # This would implement online learning if supported by the models
        # For now, just log the feedback
        self.logger.info(f"Received feedback: {analysis_data.get('File')} -> {actual_outcome}")

        # You could collect this data and retrain models periodically

    def export_predictions_log(self, output_path: str) -> None:
        """Export prediction history for analysis."""
        # This would export detailed logs of all predictions made
        self.logger.info(f"Exporting predictions to {output_path}")
        # Implementation depends on how you want to track predictions


# Integration function to add ML predictions to processor
def integrate_ml_predictor(processor, model_dir: str):
    """
    Integrate ML predictor into the laser trim processor.

    Args:
        processor: DataDrivenLaserProcessor instance
        model_dir: Directory containing ML models
    """
    # Initialize predictor
    predictor = MLPredictor(
        model_dir=model_dir,
        db_manager=processor.db_manager,
        logger=processor.logger
    )

    # Store original process_file method
    original_process_file = processor.process_file

    # Create enhanced process_file method
    def enhanced_process_file(file_path: str) -> Optional[Dict[str, Any]]:
        # Run original processing
        result = original_process_file(file_path)

        if result and result.get('Overall Status') != 'Critical Failure':
            # Get ML predictions
            predictions = predictor.predict_real_time(result)

            # Add predictions to result
            result['ml_predictions'] = predictions['predictions']
            result['ml_warnings'] = predictions['warnings']
            result['ml_recommendations'] = predictions['recommendations']
            result['ml_quality_score'] = predictions['predictions'].get('quality_score', 0)

            # Log warnings
            for warning in predictions['warnings']:
                processor.logger.warning(
                    f"{result['File']} - ML Warning: {warning['message']}"
                )

            # Log recommendations
            for rec in predictions['recommendations']:
                processor.logger.info(
                    f"{result['File']} - ML Recommendation: {rec}"
                )

        return result

    # Replace method
    processor.process_file = enhanced_process_file
    processor.ml_predictor = predictor

    processor.logger.info("ML Predictor integrated successfully")


# Example usage
if __name__ == "__main__":
    # Example of using the ML predictor standalone

    # Sample analysis data
    sample_data = {
        'File': 'test_unit_001.xlsx',
        'Model': '8340',
        'Serial': 'A12345',
        'Sigma Gradient': 0.0234,
        'Sigma Threshold': 0.04,
        'Unit Length': 150.0,
        'Travel Length': 145.2,
        'Linearity Spec': 0.02,
        'Resistance Change (%)': 2.3,
        'Trim Improvement (%)': 45.2,
        'Final Linearity Error (Shifted)': 0.015,
        'Failure Probability': 0.23,
        'Range Utilization (%)': 78.5,
        'Sigma Pass': True,
        'Linearity Pass': True
    }

    # Initialize predictor
    predictor = MLPredictor(
        model_dir='/path/to/models',
        cache_size=1000
    )

    # Get predictions
    predictions = predictor.predict_real_time(sample_data)

    # Display results
    print("\n=== ML PREDICTIONS ===")
    print(f"File: {predictions['file']}")
    print(f"\nPredictions:")
    for key, value in predictions['predictions'].items():
        print(f"  {key}: {value}")

    print(f"\nWarnings:")
    for warning in predictions['warnings']:
        print(f"  [{warning['severity'].upper()}] {warning['message']}")

    print(f"\nRecommendations:")
    for rec in predictions['recommendations']:
        print(f"  - {rec}")

    print(f"\nQuality Score: {predictions['predictions'].get('quality_score', 0):.1f}/100")
    print(f"Prediction Time: {predictions['prediction_time_ms']:.1f} ms")