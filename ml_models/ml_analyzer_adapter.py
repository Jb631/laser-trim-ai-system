"""
Adapter to make LaserTrimMLModels compatible with GUI expectations
"""
from ml_models.ml_models import LaserTrimMLModels
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import logging


class MLAnalyzer:
    """Adapter class for GUI compatibility"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        try:
            self.ml_models = LaserTrimMLModels(config, logger=self.logger)
            # Try to load existing models
            self._load_models()
        except Exception as e:
            self.logger.warning(f"ML models initialization warning: {e}")
            self.ml_models = None

    def _load_models(self):
        """Try to load pre-trained models"""
        try:
            # Get models directory from config
            models_dir = getattr(self.config, 'models_dir', 'output/ml_models')
            from pathlib import Path
            models_path = Path(models_dir)

            if models_path.exists():
                # Find latest model version
                model_versions = [d for d in models_path.iterdir() if d.is_dir() and d.name.startswith('models_v')]
                if model_versions:
                    latest = max(model_versions, key=lambda d: d.stat().st_mtime)
                    version = latest.name.replace('models_v', '')
                    self.ml_models.load_models(version)
                    self.logger.info(f"Loaded ML models version: {version}")
        except Exception as e:
            self.logger.warning(f"Could not load pre-trained models: {e}")

    def analyze(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results using ML models"""
        ml_result = {
            'risk_score': 0.0,
            'failure_probability': 0.0,
            'predictions': {},
            'risk_level': 'Low',
            'confidence': 0.0
        }

        if not self.ml_models or not self.ml_models.models.get('failure_predictor'):
            # Return default values if ML models not available
            self.logger.debug("ML models not available, returning default values")
            return ml_result

        try:
            # Extract features from result
            if 'tracks' in result:
                all_predictions = []

                for track_id, track_data in result['tracks'].items():
                    if 'sigma_results' in track_data:
                        # Prepare features
                        features = {
                            'sigma_gradient': track_data['sigma_results'].sigma_gradient,
                            'sigma_threshold': track_data['sigma_results'].sigma_threshold,
                            'linearity_spec': track_data.get('unit_properties', {}).linearity_spec if hasattr(
                                track_data.get('unit_properties', {}), 'linearity_spec') else 0.01,
                            'travel_length': track_data.get('unit_properties', {}).travel_length if hasattr(
                                track_data.get('unit_properties', {}), 'travel_length') else 120,
                            'unit_length': track_data.get('unit_properties', {}).unit_length if hasattr(
                                track_data.get('unit_properties', {}), 'unit_length') else 150,
                            'model': result.get('file_info', {}).get('filename', '').split('_')[0],
                            'timestamp': datetime.now(),
                            'resistance_change': 5.0,  # Default value
                            'resistance_change_percent': 2.0,  # Default value
                            'error_data': []  # Empty for now
                        }

                        # Get predictions
                        try:
                            failure_pred = self.ml_models.predict_failure_probability(features)
                            ml_result['predictions'][track_id] = failure_pred
                            all_predictions.append(failure_pred)
                        except Exception as e:
                            self.logger.debug(f"Prediction failed for track {track_id}: {e}")

                # Aggregate predictions
                if all_predictions:
                    # Use the highest risk from all tracks
                    max_risk_pred = max(all_predictions, key=lambda x: x.get('failure_probability', 0))
                    ml_result['risk_score'] = max_risk_pred.get('failure_probability', 0)
                    ml_result['failure_probability'] = max_risk_pred.get('failure_probability', 0)
                    ml_result['risk_level'] = max_risk_pred.get('risk_level', 'Low')
                    ml_result['confidence'] = max_risk_pred.get('confidence', 0)

        except Exception as e:
            self.logger.error(f"ML analysis error: {e}")

        return ml_result