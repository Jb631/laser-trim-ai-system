"""
ML Manager - Production-Ready ML Engine Management

Handles ML engine initialization, status tracking, and model lifecycle management
for the Laser Trim Analyzer application.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import threading
import time
import numpy as np
import pandas as pd
from pathlib import Path
import json

from laser_trim_analyzer.ml.models import ThresholdOptimizer, FailurePredictor, DriftDetector
from laser_trim_analyzer.ml.engine import MLEngine, ModelConfig


class MLEngineManager:
    """Production-ready ML engine manager with proper initialization and status tracking."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the ML engine manager."""
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # Engine state
        self.ml_engine = None
        self._status = "Not Initialized"
        self._status_color = "gray"
        self._initialization_error = None
        self._models_status = {}
        
        # Model registry
        self.available_models = {
            'threshold_optimizer': {
                'class': ThresholdOptimizer,
                'description': 'Optimizes analysis thresholds',
                'required': True
            },
            'failure_predictor': {
                'class': FailurePredictor,
                'description': 'Predicts component failure probability',
                'required': True
            },
            'drift_detector': {
                'class': DriftDetector,
                'description': 'Detects manufacturing drift patterns',
                'required': True
            }
        }
        
        # Initialize in background thread
        self._init_thread = threading.Thread(target=self._initialize_engine, daemon=True)
        self._init_thread.start()
    
    def _initialize_engine(self):
        """Initialize the ML engine and all models."""
        try:
            with self._lock:
                self._status = "Initializing..."
                self._status_color = "orange"
                
            self.logger.info("Starting ML engine initialization")
            
            # Create ML engine instance
            self.ml_engine = MLEngine()
            
            # Initialize each model
            success_count = 0
            for model_name, model_info in self.available_models.items():
                try:
                    self._initialize_model(model_name, model_info)
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to initialize {model_name}: {e}")
                    self._models_status[model_name] = {
                        'status': 'Error',
                        'trained': False,
                        'error': str(e)
                    }
            
            # Update overall status
            with self._lock:
                if success_count == len(self.available_models):
                    self._status = "Ready"
                    self._status_color = "green"
                    self.logger.info("ML engine initialization complete - all models ready")
                elif success_count > 0:
                    self._status = "Partially Ready"
                    self._status_color = "orange"
                    self.logger.warning(f"ML engine partially ready - {success_count}/{len(self.available_models)} models initialized")
                else:
                    self._status = "Failed"
                    self._status_color = "red"
                    self.logger.error("ML engine initialization failed - no models available")
                    
        except Exception as e:
            self.logger.error(f"Critical error during ML engine initialization: {e}")
            with self._lock:
                self._status = "Error"
                self._status_color = "red"
                self._initialization_error = str(e)
    
    def _initialize_model(self, model_name: str, model_info: Dict[str, Any]):
        """Initialize a specific model."""
        self.logger.info(f"Initializing model: {model_name}")
        
        # Create model config
        config = ModelConfig({
            'model_type': model_name,
            'version': '1.0.0',
            'features': self._get_default_features(model_name),
            'target': self._get_default_target(model_name)
        })
        
        # Create model instance
        model_class = model_info['class']
        model = model_class(config, logger=self.logger)
        
        # Register with ML engine
        if not hasattr(self.ml_engine, 'models'):
            self.ml_engine.models = {}
        self.ml_engine.models[model_name] = model
        
        # Check if model is trained (try to load from disk)
        model_path = self._get_model_path(model_name)
        if model_path.exists():
            try:
                model.load(str(model_path))
                is_trained = True
                self.logger.info(f"Loaded trained model from {model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load saved model {model_name}: {e}")
                is_trained = False
        else:
            is_trained = False
        
        # Update model status
        self._models_status[model_name] = {
            'status': 'Ready' if is_trained else 'Not Trained',
            'trained': is_trained,
            'description': model_info['description'],
            'last_training': datetime.now() if is_trained else None,
            'performance': model.performance_metrics if is_trained else {}
        }
    
    def _get_default_features(self, model_name: str) -> List[str]:
        """Get default features for a model type."""
        feature_sets = {
            'threshold_optimizer': [
                'sigma_gradient', 'linearity_spec', 'resistance_tolerance',
                'model_type_encoded', 'historical_pass_rate', 'production_volume'
            ],
            'failure_predictor': [
                'sigma_gradient', 'linearity_error', 'resistance_change',
                'trim_stability', 'spec_margin', 'environmental_factor'
            ],
            'drift_detector': [
                'timestamp_encoded', 'batch_mean_sigma', 'batch_std_sigma',
                'rolling_mean_diff', 'cumulative_deviation', 'trend_slope'
            ]
        }
        return feature_sets.get(model_name, [])
    
    def _get_default_target(self, model_name: str) -> str:
        """Get default target variable for a model type."""
        targets = {
            'threshold_optimizer': 'optimal_threshold',
            'failure_predictor': 'failure_within_90_days',
            'drift_detector': 'drift_detected'
        }
        return targets.get(model_name, 'target')
    
    def _get_model_path(self, model_name: str) -> Path:
        """Get the path where a model should be saved/loaded."""
        models_dir = Path.home() / '.laser_trim_analyzer' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir / f"{model_name}.joblib"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current ML engine status."""
        with self._lock:
            return {
                'status': self._status,
                'color': self._status_color,
                'error': self._initialization_error,
                'models': self._models_status.copy(),
                'engine_ready': self.ml_engine is not None,
                'models_count': len(self._models_status),
                'trained_count': sum(1 for m in self._models_status.values() if m.get('trained', False))
            }
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a specific model instance."""
        if self.ml_engine and hasattr(self.ml_engine, 'models'):
            return self.ml_engine.models.get(model_name)
        return None
    
    def train_model(self, model_name: str, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train a specific model with provided data."""
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        # Prepare features and target
        features = self._get_default_features(model_name)
        target = self._get_default_target(model_name)
        
        # For demo/testing, create synthetic target if not in data
        if target not in training_data.columns:
            self.logger.warning(f"Target {target} not found, creating synthetic data for demo")
            training_data = self._create_synthetic_target(training_data, model_name, target)
        
        # Extract features and target
        X = training_data[features]
        y = training_data[target]
        
        # Train the model
        metrics = model.train(X, y)
        
        # Update status
        self._models_status[model_name]['trained'] = True
        self._models_status[model_name]['last_training'] = datetime.now()
        self._models_status[model_name]['performance'] = metrics
        self._models_status[model_name]['status'] = 'Ready'
        
        # Save model
        model_path = self._get_model_path(model_name)
        model.save(str(model_path))
        
        return metrics
    
    def _create_synthetic_target(self, data: pd.DataFrame, model_name: str, target: str) -> pd.DataFrame:
        """Create synthetic target data for demo purposes."""
        df = data.copy()
        
        if model_name == 'threshold_optimizer':
            # Optimal threshold based on sigma gradient distribution
            df[target] = np.random.normal(0.001, 0.0002, len(df))
            df[target] = np.clip(df[target], 0.0005, 0.002)
            
        elif model_name == 'failure_predictor':
            # Binary failure prediction based on sigma gradient
            high_sigma = df['sigma_gradient'] > df['sigma_gradient'].quantile(0.9)
            df[target] = (high_sigma | (np.random.random(len(df)) < 0.1)).astype(int)
            
        elif model_name == 'drift_detector':
            # Drift detection based on patterns
            df[target] = (np.random.random(len(df)) < 0.05).astype(int)
        
        return df
    
    def predict(self, model_name: str, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using a specific model."""
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        if not model.is_trained:
            raise ValueError(f"Model {model_name} is not trained")
        
        features = self._get_default_features(model_name)
        X = data[features]
        
        return model.predict(X)
    
    def get_all_models_info(self) -> Dict[str, Any]:
        """Get detailed information about all models."""
        info = {}
        for model_name, status in self._models_status.items():
            model = self.get_model(model_name)
            info[model_name] = {
                **status,
                'has_instance': model is not None,
                'feature_count': len(self._get_default_features(model_name)),
                'features': self._get_default_features(model_name),
                'target': self._get_default_target(model_name)
            }
        return info


# Global ML manager instance
_ml_manager = None
_ml_manager_lock = threading.Lock()


def get_ml_manager() -> MLEngineManager:
    """Get or create the global ML manager instance."""
    global _ml_manager
    
    with _ml_manager_lock:
        if _ml_manager is None:
            _ml_manager = MLEngineManager()
    
    return _ml_manager