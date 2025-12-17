"""
ML Predictor Integration for Laser Trim Analyzer v2.

This module provides a clean interface between the ML components and the main processor,
handling model loading, caching, and real-time predictions.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
import traceback

from laser_trim_analyzer.core.config import Config
from laser_trim_analyzer.core.models import (
    AnalysisResult, TrackData, RiskCategory, AnalysisStatus
)
from laser_trim_analyzer.ml.engine import MLEngine, ModelConfig
from laser_trim_analyzer.ml.models import ModelFactory
from laser_trim_analyzer.database.models import MLPrediction as DBMLPrediction

# Import secure logging with fallback
try:
    from laser_trim_analyzer.core.secure_logging import (
        SecureLogger, LogLevel, logged_function, get_logger
    )
    HAS_SECURE_LOGGING = True
except ImportError:
    HAS_SECURE_LOGGING = False
    # Fallback to standard logging
    SecureLogger = None
    logged_function = lambda **kwargs: lambda func: func
    get_logger = lambda name: logging.getLogger(name)

# Import security module if available
try:
    from laser_trim_analyzer.core.security import get_security_validator
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False

# Remove the import of the deleted ML predictor implementation
# from laser_trim_analyzer.ml.ml_predictor_class import MLPredictor as MLPredictorImpl


@dataclass
class PredictionResult:
    """Structured prediction result."""
    failure_probability: float
    risk_category: str
    is_anomaly: bool
    anomaly_score: float
    suggested_threshold: Optional[float]
    confidence_score: float
    warnings: List[Dict[str, Any]]
    recommendations: List[str]
    feature_importance: Optional[Dict[str, float]] = None


class MLPredictor:
    """
    ML Predictor for real-time analysis integration.

    This class wraps the existing ML predictor implementation and provides
    a clean interface for the processor while managing models through MLEngine.
    """

    def __init__(
            self,
            config: Config,
            ml_engine: Optional[MLEngine] = None,
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ML Predictor.

        Args:
            config: Application configuration
            ml_engine: ML Engine instance for model management
            logger: Logger instance
        """
        self.config = config
        
        # Initialize secure logger
        if HAS_SECURE_LOGGING and not logger:
            self.logger = get_logger(
                'laser_trim_analyzer.ml.predictors',
                log_level=LogLevel.DEBUG if config.debug else LogLevel.INFO
            )
        else:
            self.logger = logger or logging.getLogger(__name__)
        
        # Log initialization
        self._log_with_context('info', "Initializing ML Predictor",
            context={
                'model_path': str(config.ml.model_path),
                'enabled': config.ml.enabled
            })

        # Initialize ML Engine if not provided
        if ml_engine is None:
            # Get data directory from config or use default
            data_dir = getattr(config, 'data_directory', './data')
            if hasattr(config, 'ml') and hasattr(config.ml, 'model_path'):
                models_dir = config.ml.model_path
            else:
                models_dir = './models'
            
            ml_engine = MLEngine(
                data_path=str(data_dir),
                models_path=str(models_dir),
                logger=self.logger
            )
        self.ml_engine = ml_engine

        # Initialize the implementation predictor
        self._impl_predictor = None
        self._models_loaded = False

        # Performance tracking
        self.prediction_count = 0
        self.total_prediction_time = 0.0

        # Feature names mapping
        self.feature_mapping = {
            'sigma_gradient': 'Sigma Gradient',
            'sigma_threshold': 'Sigma Threshold',
            'unit_length': 'Unit Length',
            'travel_length': 'Travel Length',
            'linearity_spec': 'Linearity Spec',
            'resistance_change_percent': 'Resistance Change (%)',
            'trim_improvement_percent': 'Trim Improvement (%)',
            'final_linearity_error_shifted': 'Final Linearity Error (Shifted)',
            'failure_probability': 'Failure Probability',
            'range_utilization_percent': 'Range Utilization (%)'
        }

    def _log_with_context(self, level: str, message: str, context: Optional[Dict] = None, **kwargs):
        """Helper method to log with context support."""
        if hasattr(self.logger, level) and callable(getattr(self.logger, level)):
            log_method = getattr(self.logger, level)
            
            # Check if this is a secure logger by looking for the module
            if HAS_SECURE_LOGGING and hasattr(self.logger, '__module__') and 'secure_logging' in str(self.logger.__module__):
                # This is a secure logger, use context parameter
                if context:
                    log_method(message, context=context, **kwargs)
                else:
                    log_method(message, **kwargs)
            else:
                # This is a standard logger, append context to message
                if context:
                    context_str = ', '.join(f'{k}={v}' for k, v in context.items())
                    message = f"{message} [{context_str}]"
                # Remove any 'context' from kwargs to avoid the error
                kwargs.pop('context', None)
                log_method(message, **kwargs)

    @logged_function(log_inputs=True, log_outputs=True, log_performance=True)
    def initialize(self) -> bool:
        """
        Initialize ML models and predictor.

        Returns:
            True if initialization successful
        """
        try:
            if HAS_SECURE_LOGGING and hasattr(self.logger, 'start_performance_tracking'):
                self.logger.start_performance_tracking('ml_initialization')
            
            # Create model directory if it doesn't exist
            model_dir = self.config.ml.model_path
            model_dir.mkdir(parents=True, exist_ok=True)
            
            self._log_with_context('debug', "Created model directory", context={'model_dir': str(model_dir)})

            # Initialize implementation predictor
            self._impl_predictor = None

            # Load or register models with MLEngine
            self._register_models()
            
            # Try to load existing trained models from disk
            self._load_existing_models()

            # Check if models need training
            if self._check_models_need_training():
                self._log_with_context('info', "ML models need training. Will use default predictions.",
                    context={
                        'models_checked': ['failure_predictor', 'anomaly_detector', 'threshold_optimizer'],
                        'models_found': False
                    })
                # Don't fail initialization, just mark models as not loaded
                self._models_loaded = False
            else:
                self._models_loaded = True
                self._log_with_context('info', "ML models loaded successfully", context={'models_loaded': True})

            if HAS_SECURE_LOGGING and hasattr(self.logger, 'end_performance_tracking'):
                init_time = self.logger.end_performance_tracking('ml_initialization')
                self._log_with_context('info', "ML Predictor initialized successfully",
                    performance={'initialization_time_ms': init_time * 1000}
                )
            else:
                self._log_with_context('info', "ML Predictor initialized successfully")
            
            return True

        except Exception as e:
            self._log_with_context('error', "Failed to initialize ML Predictor",
                context={'error': str(e)})
            return False

    @logged_function(log_performance=True)
    def _register_models(self):
        """Register ML models with the engine."""
        from laser_trim_analyzer.ml.models import ThresholdOptimizer, FailurePredictor, DriftDetector
        
        try:
            self._log_with_context('debug', "Starting model registration", context={'ml_engine_id': id(self.ml_engine)})
            
            # Ensure models dictionary exists
            if not hasattr(self.ml_engine, 'models'):
                self.ml_engine.models = {}
            
            # Create model configs first
            failure_config = ModelConfig({
                'model_type': 'failure_predictor',
                'version': '1.0.0',
                'features': ['sigma_gradient', 'sigma_threshold', 'linearity_pass',
                          'resistance_change_percent', 'trim_improvement_percent',
                          'failure_probability', 'worst_zone'],
                'hyperparameters': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5
                },
                'training_params': {
                    'test_size': 0.2,
                    'random_state': 42,
                    'cv_folds': 5
                }
            })
            
            threshold_config = ModelConfig({
                'model_type': 'threshold_optimizer',
                'version': '1.0.0',
                'features': ['sigma_gradient', 'linearity_spec', 'travel_length',
                          'unit_length', 'resistance_change_percent'],
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                },
                'training_params': {
                    'test_size': 0.2,
                    'random_state': 42,
                    'cv_folds': 5
                }
            })
            
            drift_config = ModelConfig({
                'model_type': 'drift_detector',
                'version': '1.0.0',
                'features': ['sigma_gradient', 'linearity_spec', 'resistance_change_percent',
                          'travel_length', 'unit_length'],
                'hyperparameters': {
                    'n_estimators': 100,
                    'contamination': 0.05,
                    'max_samples': 'auto'
                },
                'training_params': {
                    'test_size': 0.2,
                    'random_state': 42
                }
            })
            
            # Create and register failure predictor
            failure_predictor = FailurePredictor(failure_config, self.logger)
            failure_predictor.model_type = 'failure_predictor'
            failure_predictor.is_trained = False
            failure_predictor.version = '1.0.0'
            failure_predictor.last_trained = None
            failure_predictor.training_samples = 0
            failure_predictor.performance_metrics = {}
            failure_predictor.prediction_count = 0
            
            self.ml_engine.models['failure_predictor'] = failure_predictor
            self._log_with_context('info', "Registered model: failure_predictor",
                context={
                    'model_type': 'failure_predictor',
                    'version': '1.0.0',
                    'features': failure_config.features,
                    'hyperparameters': failure_config.hyperparameters
                })
            
            # Create and register threshold optimizer
            threshold_optimizer = ThresholdOptimizer(threshold_config, self.logger)
            threshold_optimizer.model_type = 'threshold_optimizer'
            threshold_optimizer.is_trained = False
            threshold_optimizer.version = '1.0.0'
            threshold_optimizer.last_trained = None
            threshold_optimizer.training_samples = 0
            threshold_optimizer.performance_metrics = {}
            threshold_optimizer.prediction_count = 0
            
            self.ml_engine.models['threshold_optimizer'] = threshold_optimizer
            self._log_with_context('info', "Registered model: threshold_optimizer",
                context={
                    'model_type': 'threshold_optimizer',
                    'version': '1.0.0',
                    'features': threshold_config.features,
                    'hyperparameters': threshold_config.hyperparameters
                })
            
            # Create and register drift detector
            drift_detector = DriftDetector(drift_config, self.logger)
            drift_detector.model_type = 'drift_detector'
            drift_detector.is_trained = False
            drift_detector.version = '1.0.0'
            drift_detector.last_trained = None
            drift_detector.training_samples = 0
            drift_detector.performance_metrics = {}
            drift_detector.prediction_count = 0
            
            self.ml_engine.models['drift_detector'] = drift_detector
            self._log_with_context('info', "Registered model: drift_detector",
                context={
                    'model_type': 'drift_detector',
                    'version': '1.0.0',
                    'features': drift_config.features,
                    'hyperparameters': drift_config.hyperparameters
                })

            # Store configs if supported
            if not hasattr(self.ml_engine, 'model_configs'):
                self.ml_engine.model_configs = {}
                
            self.ml_engine.model_configs['failure_predictor'] = failure_config
            self.ml_engine.model_configs['threshold_optimizer'] = threshold_config
            self.ml_engine.model_configs['drift_detector'] = drift_config
            
            self._log_with_context('info', "Successfully registered models",
                context={'model_count': len(self.ml_engine.models)})
            
        except Exception as e:
            self._log_with_context('error', "Error registering models",
                context={'error': str(e)})
            raise
    
    def _load_existing_models(self):
        """Load existing trained models from disk if available."""
        try:
            # Get model path from config with defensive checks
            model_path = None
            if hasattr(self.config, 'ml') and hasattr(self.config.ml, 'model_path'):
                model_path = self.config.ml.model_path

            # Handle relative paths - convert Path to string first
            if model_path:
                model_path_str = str(model_path)
                if model_path_str.startswith('./'):
                    model_dir = Path.cwd() / model_path_str[2:]
                else:
                    model_dir = Path(model_path)
            else:
                model_dir = None
            
            if not model_dir or not model_dir.exists():
                self._log_with_context('debug', "Model path does not exist, skipping model loading",
                                      context={'model_path': str(model_dir)})
                return
            
            loaded_models = []
            
            # Try to load each registered model
            for model_name, model in self.ml_engine.models.items():
                try:
                    # Check for model files (look for exact name or versioned)
                    model_files = list(model_dir.glob(f"{model_name}.pkl"))
                    if not model_files:
                        model_files = list(model_dir.glob(f"{model_name}.joblib"))
                    if not model_files:
                        model_files = list(model_dir.glob(f"{model_name}_*.pkl"))
                    if not model_files:
                        model_files = list(model_dir.glob(f"{model_name}_*.joblib"))
                    
                    if model_files:
                        # Sort by modification time to get the latest
                        latest_model_file = max(model_files, key=lambda p: p.stat().st_mtime)
                        
                        # Load the model state
                        import joblib
                        loaded_state = joblib.load(latest_model_file)
                        
                        # Apply the loaded state to the model
                        if hasattr(model, 'model') and 'model' in loaded_state:
                            model.model = loaded_state['model']
                            model.is_trained = True
                            
                            # Load other attributes if available
                            if 'version' in loaded_state:
                                model.version = loaded_state['version']
                            if 'last_trained' in loaded_state:
                                model.last_trained = loaded_state['last_trained']
                            if 'training_samples' in loaded_state:
                                model.training_samples = loaded_state['training_samples']
                            if 'performance_metrics' in loaded_state:
                                model.performance_metrics = loaded_state['performance_metrics']
                            
                            loaded_models.append(model_name)
                            self._log_with_context('info', f"Loaded trained model: {model_name}",
                                                 context={'model_file': str(latest_model_file),
                                                        'is_trained': True})
                        
                except Exception as e:
                    self._log_with_context('warning', f"Could not load model {model_name}: {e}",
                                         context={'model_name': model_name, 'error': str(e)})
            
            if loaded_models:
                self._log_with_context('info', f"Successfully loaded {len(loaded_models)} trained models",
                                     context={'loaded_models': loaded_models})
                
        except Exception as e:
            self._log_with_context('error', f"Error loading existing models: {e}",
                                 context={'error': str(e), 'traceback': traceback.format_exc()[:500]})

    def _check_models_need_training(self) -> bool:
        """Check if models need initial training."""
        model_files = ['failure_predictor.pkl', 'anomaly_detector.pkl', 'threshold_optimizer.pkl']
        model_dir = self.config.ml.model_path

        for model_file in model_files:
            if not (model_dir / model_file).exists():
                return True
        return False

    @logged_function(log_performance=True)
    async def predict(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """
        Make predictions for an analysis result with input validation.

        Args:
            analysis_result: Complete analysis result

        Returns:
            Dictionary with predictions for all tracks
        """
        # Handle case where analysis_result might be a dict instead of object
        if isinstance(analysis_result, dict):
            # Convert dict to object-like access
            metadata = analysis_result.get('metadata', {})
            tracks = analysis_result.get('tracks', {})
        else:
            # Normal object access
            metadata = getattr(analysis_result, 'metadata', None)
            tracks = getattr(analysis_result, 'tracks', {})
        
        # Ensure we have valid metadata
        if not metadata:
            self.logger.debug("No metadata available for ML prediction - skipping")
            return {}
        
        # Get attributes from metadata safely
        if isinstance(metadata, dict):
            model = metadata.get('model', 'Unknown')
            serial = metadata.get('serial', 'Unknown')
            file_date = metadata.get('file_date', 'Unknown')
        else:
            model = getattr(metadata, 'model', 'Unknown')
            serial = getattr(metadata, 'serial', 'Unknown') 
            file_date = getattr(metadata, 'file_date', 'Unknown')
        
        if HAS_SECURE_LOGGING and hasattr(self.logger, 'start_performance_tracking'):
            self.logger.start_performance_tracking('ml_prediction')
            self.logger.debug(
                "Starting ML prediction",
                context={
                    'model': model,
                    'serial': serial,
                    'track_count': len(tracks),
                    'file_date': str(file_date)
                }
            )
        
        # Validate inputs if security is available
        if HAS_SECURITY:
            try:
                validator = get_security_validator()
                
                # Validate model and serial
                model_result = validator.validate_input(
                    model,
                    'model_number'
                )
                if not model_result.is_safe:
                    if HAS_SECURE_LOGGING:
                        self.logger.warning(
                            f"Invalid model number for ML: {model_result.validation_errors}",
                            security={'validation_errors': model_result.validation_errors}
                        )
                    else:
                        self.logger.warning(f"Invalid model number for ML: {model_result.validation_errors}")
                    return {}
                    
                serial_result = validator.validate_input(
                    serial,
                    'serial_number'
                )
                if not serial_result.is_safe:
                    if HAS_SECURE_LOGGING:
                        self.logger.warning(
                            f"Invalid serial number for ML: {serial_result.validation_errors}",
                            security={'validation_errors': serial_result.validation_errors}
                        )
                    else:
                        self.logger.warning(f"Invalid serial number for ML: {serial_result.validation_errors}")
                    return {}
            except Exception as e:
                if HAS_SECURE_LOGGING:
                    self.logger.warning(
                        f"Security validation error in ML: {e}"
                    )
                else:
                    self.logger.warning(f"Security validation error in ML: {e}")
                # Continue with prediction if security check fails
        
        if not self._models_loaded:
            # Return empty predictions instead of error
            return {}

        predictions = {}

        # Process each track
        for track_id, track_data in tracks.items():
            try:
                # Handle track_data as either dict or object
                if isinstance(track_data, dict):
                    sigma_gradient = track_data.get('sigma_analysis', {}).get('sigma_gradient', 0)
                    linearity_pass = track_data.get('linearity_analysis', {}).get('linearity_pass', False)
                else:
                    sigma_gradient = getattr(track_data.sigma_analysis, 'sigma_gradient', 0) if hasattr(track_data, 'sigma_analysis') else 0
                    linearity_pass = getattr(track_data.linearity_analysis, 'linearity_pass', False) if hasattr(track_data, 'linearity_analysis') else False
                
                if HAS_SECURE_LOGGING:
                    self.logger.trace(
                        f"Predicting for track {track_id}",
                        context={
                            'track_id': track_id,
                            'sigma_gradient': sigma_gradient,
                            'linearity_pass': linearity_pass
                        }
                    )
                
                track_predictions = await self._predict_for_track(
                    track_data,
                    model,
                    serial,
                    track_id
                )
                predictions[track_id] = track_predictions

            except Exception as e:
                if HAS_SECURE_LOGGING:
                    self.logger.error(
                        f"Prediction failed for track {track_id}: {e}",
                        context={'track_id': track_id, 'model': model}
                    )
                else:
                    self.logger.error(f"Prediction failed for track {track_id}: {e}")
                predictions[track_id] = self._get_default_predictions()

        # Add file-level predictions
        predictions['overall'] = self._aggregate_predictions(predictions)
        
        if HAS_SECURE_LOGGING and hasattr(self.logger, 'end_performance_tracking'):
            pred_time = self.logger.end_performance_tracking('ml_prediction')
            self.logger.info(
                "ML prediction completed",
                context={
                    'tracks_processed': len(predictions) - 1,  # Excluding 'overall'
                    'overall_risk': predictions.get('overall', {}).get('overall_risk', 'Unknown')
                },
                performance={'total_prediction_time_ms': pred_time * 1000}
            )

        return predictions

    async def _predict_for_track(
            self,
            track_data: TrackData,
            model: str,
            serial: str,
            track_id: str
    ) -> PredictionResult:
        """Make predictions for a single track."""
        import time
        start_time = time.time()
        
        if HAS_SECURE_LOGGING:
            self.logger.debug(
                f"Preparing data for track {track_id}",
                context={'model': model, 'serial': serial}
            )

        # Prepare data for predictor
        analysis_data = self._prepare_track_data(track_data, model, serial, track_id)
        
        if HAS_SECURE_LOGGING:
            self.logger.trace(
                "Track data prepared",
                context={
                    'track_id': track_id,
                    'features_extracted': list(analysis_data.keys()),
                    'sigma_gradient': analysis_data.get('Sigma Gradient'),
                    'risk_category': analysis_data.get('Risk Category')
                }
            )

        # Get predictions from implementation
        ml_predictions = self._impl_predictor.predict_real_time(analysis_data)
        
        if HAS_SECURE_LOGGING:
            self.logger.debug(
                "Raw ML predictions received",
                context={
                    'track_id': track_id,
                    'prediction_keys': list(ml_predictions.keys()) if ml_predictions else [],
                    'has_predictions': 'predictions' in ml_predictions
                }
            )

        # Parse predictions into structured result
        pred_data = ml_predictions.get('predictions', {})

        # Validate and sanitize prediction outputs
        failure_prob = pred_data.get('failure_probability', 0.0)
        anomaly_score = pred_data.get('anomaly_score', 0.0)
        quality_score = pred_data.get('quality_score', 0.0)
        suggested_threshold = pred_data.get('suggested_threshold')
        
        # Ensure probabilities are in valid range [0, 1]
        failure_prob = max(0.0, min(1.0, float(failure_prob) if failure_prob is not None else 0.0))
        anomaly_score = max(0.0, min(1.0, float(anomaly_score) if anomaly_score is not None else 0.0))
        confidence_score = max(0.0, min(1.0, float(quality_score) / 100.0 if quality_score is not None else 0.0))
        
        # Validate suggested threshold
        if suggested_threshold is not None:
            try:
                suggested_threshold = float(suggested_threshold)
                # Ensure threshold is within reasonable range
                if suggested_threshold < 0 or suggested_threshold > 1000:
                    self.logger.warning(f"Invalid suggested threshold: {suggested_threshold}")
                    suggested_threshold = None
            except (ValueError, TypeError):
                suggested_threshold = None
        
        # Sanitize warnings and recommendations
        warnings = ml_predictions.get('warnings', [])
        recommendations = ml_predictions.get('recommendations', [])
        
        # Ensure warnings and recommendations are lists of strings
        if not isinstance(warnings, list):
            warnings = []
        else:
            warnings = [str(w) for w in warnings[:10]]  # Limit to 10 warnings
            
        if not isinstance(recommendations, list):
            recommendations = []
        else:
            recommendations = [str(r) for r in recommendations[:10]]  # Limit to 10 recommendations
        
        result = PredictionResult(
            failure_probability=failure_prob,
            risk_category=self._determine_risk_category(failure_prob),
            is_anomaly=bool(pred_data.get('is_anomaly', False)),
            anomaly_score=anomaly_score,
            suggested_threshold=suggested_threshold,
            confidence_score=confidence_score,
            warnings=warnings,
            recommendations=recommendations
        )

        # Track performance
        prediction_time = time.time() - start_time
        self.prediction_count += 1
        self.total_prediction_time += prediction_time
        
        if HAS_SECURE_LOGGING:
            self.logger.debug(
                f"Track prediction completed for {track_id}",
                context={
                    'track_id': track_id,
                    'failure_probability': failure_prob,
                    'risk_category': result.risk_category,
                    'is_anomaly': result.is_anomaly,
                    'anomaly_score': anomaly_score,
                    'confidence_score': confidence_score,
                    'suggested_threshold': suggested_threshold,
                    'warning_count': len(warnings),
                    'recommendation_count': len(recommendations)
                },
                performance={'track_prediction_time_ms': prediction_time * 1000}
            )

        return result

    @logged_function(log_performance=True)
    def _prepare_track_data(
            self,
            track_data: TrackData,
            model: str,
            serial: str,
            track_id: str
    ) -> Dict[str, Any]:
        """Prepare and validate track data for ML predictor."""
        if HAS_SECURE_LOGGING:
            self.logger.trace(
                "Preparing track data for ML",
                context={'track_id': track_id, 'model': model}
            )
        
        # Validate numeric inputs if security is available
        if HAS_SECURITY:
            try:
                validator = get_security_validator()
                
                # Validate critical numeric values
                numeric_validations = [
                    ('sigma_gradient', track_data.sigma_analysis.sigma_gradient),
                    ('sigma_threshold', track_data.sigma_analysis.sigma_threshold),
                    ('unit_length', track_data.unit_properties.unit_length),
                    ('travel_length', track_data.travel_length),
                    ('linearity_spec', track_data.linearity_analysis.linearity_spec)
                ]
                
                for field_name, value in numeric_validations:
                    if value is not None:
                        result = validator.validate_input(
                            value,
                            'number',
                            {'min': 0, 'max': 1e6}  # Reasonable range for measurements
                        )
                        if not result.is_safe:
                            if HAS_SECURE_LOGGING:
                                self.logger.warning(
                                    f"Invalid {field_name} value: {result.validation_errors}",
                                    security={
                                        'field': field_name,
                                        'validation_errors': result.validation_errors,
                                        'original_value': str(value)[:50]  # Truncate for safety
                                    }
                                )
                            else:
                                self.logger.warning(f"Invalid {field_name} value: {result.validation_errors}")
                            # Use sanitized value
                            value = result.sanitized_value or 0
            except Exception as e:
                if HAS_SECURE_LOGGING:
                    self.logger.warning(
                        f"ML input validation error: {e}"
                    )
                else:
                    self.logger.warning(f"ML input validation error: {e}")
        
        data = {
            'File': f"{model}_{serial}.xlsx",
            'Model': model,
            'Serial': serial,
            'Track': track_id,
            'Sigma Gradient': track_data.sigma_analysis.sigma_gradient,
            'Sigma Threshold': track_data.sigma_analysis.sigma_threshold,
            'Sigma Pass': track_data.sigma_analysis.sigma_pass,
            'Unit Length': track_data.unit_properties.unit_length,
            'Travel Length': track_data.travel_length,
            'Linearity Spec': track_data.linearity_analysis.linearity_spec,
            'Linearity Pass': track_data.linearity_analysis.linearity_pass,
            'Resistance Change (%)': track_data.resistance_analysis.resistance_change_percent,
            'Risk Category': track_data.failure_prediction.risk_category.value if track_data.failure_prediction else 'Unknown'
        }

        # Add optional fields
        if track_data.trim_effectiveness:
            data['Trim Improvement (%)'] = track_data.trim_effectiveness.improvement_percent

        if track_data.linearity_analysis:
            data['Final Linearity Error (Shifted)'] = track_data.linearity_analysis.final_linearity_error_shifted

        if track_data.failure_prediction:
            data['Failure Probability'] = track_data.failure_prediction.failure_probability

        if track_data.dynamic_range:
            data['Range Utilization (%)'] = track_data.dynamic_range.range_utilization_percent
        
        if HAS_SECURE_LOGGING:
            self.logger.trace(
                "Track data prepared successfully",
                context={
                    'track_id': track_id,
                    'feature_count': len(data),
                    'has_optional_features': {
                        'trim_effectiveness': 'Trim Improvement (%)' in data,
                        'linearity_error': 'Final Linearity Error (Shifted)' in data,
                        'failure_probability': 'Failure Probability' in data,
                        'range_utilization': 'Range Utilization (%)' in data
                    }
                }
            )

        return data

    def _determine_risk_category(self, failure_probability: float) -> str:
        """Determine risk category from failure probability."""
        if failure_probability > self.config.analysis.high_risk_threshold:
            return RiskCategory.HIGH.value
        elif failure_probability > self.config.analysis.low_risk_threshold:
            return RiskCategory.MEDIUM.value
        else:
            return RiskCategory.LOW.value

    def _aggregate_predictions(
            self,
            track_predictions: Dict[str, PredictionResult]
    ) -> Dict[str, Any]:
        """Aggregate track predictions for overall assessment."""
        if not track_predictions:
            return {}

        # Filter out 'overall' key if it exists
        track_preds = {k: v for k, v in track_predictions.items() if k != 'overall'}

        if not track_preds:
            return {}

        # Calculate overall metrics
        failure_probs = [p.failure_probability for p in track_preds.values()]
        anomaly_flags = [p.is_anomaly for p in track_preds.values()]

        # Collect all warnings and recommendations
        all_warnings = []
        all_recommendations = []

        for pred in track_preds.values():
            all_warnings.extend(pred.warnings)
            all_recommendations.extend(pred.recommendations)

        # Remove duplicates
        unique_recommendations = list(set(all_recommendations))

        return {
            'max_failure_probability': max(failure_probs),
            'avg_failure_probability': np.mean(failure_probs),
            'any_anomaly': any(anomaly_flags),
            'anomaly_count': sum(anomaly_flags),
            'overall_risk': self._determine_risk_category(max(failure_probs)),
            'total_warnings': len(all_warnings),
            'unique_recommendations': len(unique_recommendations),
            'top_recommendations': unique_recommendations[:3]  # Top 3 recommendations
        }

    def _get_default_predictions(self) -> PredictionResult:
        """Get default predictions when ML fails."""
        return PredictionResult(
            failure_probability=0.0,
            risk_category=RiskCategory.UNKNOWN.value,
            is_anomaly=False,
            anomaly_score=0.0,
            suggested_threshold=None,
            confidence_score=0.0,
            warnings=[],
            recommendations=[]
        )

    @logged_function(log_inputs=True, log_outputs=True, log_performance=True)
    def check_for_drift(self, recent_results: List[AnalysisResult]) -> Dict[str, Any]:
        """
        Check for manufacturing drift in recent results.

        Args:
            recent_results: List of recent analysis results

        Returns:
            Drift analysis results
        """
        if not self._models_loaded or not recent_results:
            if HAS_SECURE_LOGGING:
                self.logger.debug(
                    "Drift check skipped",
                    context={
                        'models_loaded': self._models_loaded,
                        'results_count': len(recent_results) if recent_results else 0
                    }
                )
            return {}

        try:
            if HAS_SECURE_LOGGING and hasattr(self.logger, 'start_performance_tracking'):
                self.logger.start_performance_tracking('drift_detection')
                self.logger.info(
                    "Starting drift detection",
                    context={'result_count': len(recent_results)}
                )
            
            # Convert results to DataFrame
            data_for_drift = []
            for result in recent_results:
                for track_id, track in result.tracks.items():
                    data_for_drift.append({
                        'timestamp': result.metadata.file_date,
                        'model': result.metadata.model,
                        'sigma_gradient': track.sigma_analysis.sigma_gradient,
                        'linearity_spec': track.linearity_analysis.linearity_spec,
                        'resistance_change_percent': track.resistance_analysis.resistance_change_percent or 0,
                        'travel_length': track.travel_length,
                        'unit_length': track.unit_properties.unit_length or 0
                    })

            df = pd.DataFrame(data_for_drift)
            
            if HAS_SECURE_LOGGING:
                self.logger.debug(
                    "Data prepared for drift detection",
                    context={
                        'data_points': len(data_for_drift),
                        'models': df['model'].unique().tolist() if 'model' in df else [],
                        'date_range': {
                            'start': str(df['timestamp'].min()) if 'timestamp' in df and len(df) > 0 else None,
                            'end': str(df['timestamp'].max()) if 'timestamp' in df and len(df) > 0 else None
                        }
                    }
                )

            # Use MLEngine's drift detector if available
            if 'drift_detector' in self.ml_engine.models:
                drift_detector = self.ml_engine.models['drift_detector']
                drift_analysis = drift_detector.analyze_drift(df)
                
                if HAS_SECURE_LOGGING and hasattr(self.logger, 'end_performance_tracking'):
                    drift_time = self.logger.end_performance_tracking('drift_detection')
                    self.logger.info(
                        "Drift detection completed",
                        context={
                            'drift_detected': drift_analysis.get('drift_detected', False),
                            'drift_metrics': drift_analysis.get('metrics', {})
                        },
                        performance={'drift_detection_time_ms': drift_time * 1000}
                    )
                
                return drift_analysis

            return {'drift_detected': False, 'message': 'Drift detector not available'}

        except Exception as e:
            if HAS_SECURE_LOGGING:
                self.logger.error(
                    f"Drift detection failed: {e}",
                    context={'result_count': len(recent_results)}
                )
            else:
                self.logger.error(f"Drift detection failed: {e}")
            return {'error': str(e)}

    @logged_function(log_inputs=True, log_outputs=True, log_performance=True)
    def suggest_threshold_adjustments(
            self,
            model: str,
            recent_results: List[AnalysisResult]
    ) -> Optional[Dict[str, float]]:
        """
        Suggest threshold adjustments for a model.

        Args:
            model: Model number
            recent_results: Recent analysis results for the model

        Returns:
            Suggested thresholds or None
        """
        if not self._models_loaded:
            if HAS_SECURE_LOGGING:
                self.logger.debug(
                    "Threshold suggestion skipped - models not loaded",
                    context={'model': model}
                )
            return None

        try:
            if HAS_SECURE_LOGGING:
                self.logger.info(
                    "Suggesting threshold adjustments",
                    context={
                        'model': model,
                        'result_count': len(recent_results),
                        'date_range': {
                            'start': str(min(r.metadata.file_date for r in recent_results)) if recent_results else None,
                            'end': str(max(r.metadata.file_date for r in recent_results)) if recent_results else None
                        }
                    }
                )
            
            # Get recommended thresholds from predictor
            if hasattr(self._impl_predictor, 'thresholds'):
                model_thresholds = self._impl_predictor.thresholds.get(model)
                if model_thresholds:
                    suggestions = {
                        'recommended': model_thresholds.get('recommended'),
                        'confidence': model_thresholds.get('confidence', 0.8),
                        'based_on_samples': model_thresholds.get('sample_count', 0)
                    }
                    
                    if HAS_SECURE_LOGGING:
                        self.logger.info(
                            "Threshold suggestions generated",
                            context={
                                'model': model,
                                'recommended_threshold': suggestions['recommended'],
                                'confidence': suggestions['confidence'],
                                'sample_count': suggestions['based_on_samples']
                            }
                        )
                    
                    return suggestions

            if HAS_SECURE_LOGGING:
                self.logger.debug(
                    "No threshold data available for model",
                    context={'model': model}
                )
            
            return None

        except Exception as e:
            if HAS_SECURE_LOGGING:
                self.logger.error(
                    f"Threshold suggestion failed: {e}",
                    context={'model': model}
                )
            else:
                self.logger.error(f"Threshold suggestion failed: {e}")
            return None

    @logged_function(log_outputs=True)
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get ML predictor performance statistics."""
        stats = {
            'predictions_made': self.prediction_count,
            'average_prediction_time': (
                    self.total_prediction_time / max(1, self.prediction_count)
            ),
            'models_loaded': self._models_loaded
        }

        # Add implementation stats if available
        if self._impl_predictor:
            impl_stats = self._impl_predictor.get_performance_stats()
            stats.update({
                'cache_hit_rate': impl_stats.get('cache_hit_rate', 0),
                'anomaly_detection_rate': impl_stats.get('anomaly_detection_rate', 0),
                'model_versions': impl_stats.get('model_versions', {})
            })
        
        if HAS_SECURE_LOGGING:
            self.logger.info(
                "ML performance stats retrieved",
                context=stats
            )

        return stats

    @logged_function(log_performance=True)
    def update_models(self) -> bool:
        """
        Check for and load updated models.

        Returns:
            True if models were updated
        """
        if HAS_SECURE_LOGGING:
            self.logger.info("Checking for model updates")
        
        if self._impl_predictor:
            self._impl_predictor.check_for_model_updates()
            
            if HAS_SECURE_LOGGING:
                self.logger.info(
                    "Model update check completed",
                    context={'models_updated': True}
                )
            
            return True
        
        if HAS_SECURE_LOGGING:
            self.logger.debug(
                "No implementation predictor available for updates",
                context={'models_updated': False}
            )
        
        return False

    @logged_function(log_performance=True)
    def format_ml_predictions_for_db(
            self,
            predictions: Dict[str, Any],
            analysis_id: int
    ) -> List[DBMLPrediction]:
        """
        Format ML predictions for database storage.

        Args:
            predictions: ML predictions dictionary
            analysis_id: Database ID of the analysis

        Returns:
            List of database prediction objects
        """
        if HAS_SECURE_LOGGING:
            self.logger.debug(
                "Formatting ML predictions for database storage",
                context={
                    'analysis_id': analysis_id,
                    'track_count': len([k for k in predictions.keys() if k != 'overall'])
                }
            )
        
        db_predictions = []

        for track_id, pred in predictions.items():
            if track_id == 'overall':
                continue

            if isinstance(pred, PredictionResult):
                db_pred = DBMLPrediction(
                    analysis_id=analysis_id,
                    prediction_type='comprehensive',
                    model_version=self._impl_predictor.model_versions.get(
                        'failure_predictor', '1.0'
                    ) if self._impl_predictor else '1.0',
                    predicted_failure_probability=pred.failure_probability,
                    predicted_risk_category=pred.risk_category,
                    confidence_score=pred.confidence_score,
                    recommendations=pred.recommendations,
                    drift_detected=pred.is_anomaly,
                    drift_percentage=abs(pred.anomaly_score) * 100 if pred.is_anomaly else None
                )

                # Add threshold recommendation if available
                if pred.suggested_threshold:
                    db_pred.recommended_threshold = pred.suggested_threshold

                db_predictions.append(db_pred)
                
                if HAS_SECURE_LOGGING:
                    self.logger.trace(
                        f"Formatted prediction for track {track_id}",
                        context={
                            'track_id': track_id,
                            'risk_category': pred.risk_category,
                            'failure_probability': pred.failure_probability,
                            'has_threshold': pred.suggested_threshold is not None
                        }
                    )
        
        if HAS_SECURE_LOGGING:
            self.logger.info(
                "ML predictions formatted for database",
                context={
                    'analysis_id': analysis_id,
                    'predictions_formatted': len(db_predictions),
                    'model_version': db_predictions[0].model_version if db_predictions else 'N/A'
                }
            )

        return db_predictions