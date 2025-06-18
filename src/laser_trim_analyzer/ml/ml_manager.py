"""
ML Manager - Production-Ready ML Engine Management

Handles ML engine initialization, status tracking, and model lifecycle management
for the Laser Trim Analyzer application.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import threading
import time
from pathlib import Path
import json
import pickle
import traceback
import gc

# Check for required dependencies
try:
    import numpy as np
    import pandas as pd
    HAS_NUMPY_PANDAS = True
except ImportError as e:
    HAS_NUMPY_PANDAS = False
    logging.error(f"Missing required dependencies: {e}")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logging.warning("psutil not available, system resource monitoring disabled")

try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.error("scikit-learn not available, ML features will be disabled")

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    logging.warning("joblib not available, model persistence may be limited")

from laser_trim_analyzer.ml.models import ThresholdOptimizer, FailurePredictor, DriftDetector
from laser_trim_analyzer.ml.engine import MLEngine, ModelConfig
from laser_trim_analyzer.core.error_handlers import (
    ErrorCode, ErrorCategory, ErrorSeverity,
    error_handler, handle_errors, check_system_resources
)
from laser_trim_analyzer.core.exceptions import MLPredictionError


class MLEngineManager:
    """Production-ready ML engine manager with proper initialization and status tracking."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the ML engine manager."""
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # Check dependencies and raise error if missing (ML is NOT optional)
        if not HAS_NUMPY_PANDAS:
            error_msg = "NumPy and Pandas are required for ML features. Install with: pip install numpy pandas"
            self._status = "Missing Dependencies"
            self._status_color = "red"
            self._initialization_error = error_msg
            self.ml_engine = None
            self._models_status = {}
            self.logger.error(f"Cannot initialize ML engine: {error_msg}")
            raise ImportError(error_msg)
            
        if not HAS_SKLEARN:
            error_msg = "scikit-learn is required for ML features. Install with: pip install scikit-learn joblib"
            self._status = "Missing scikit-learn"
            self._status_color = "red"
            self._initialization_error = error_msg
            self.ml_engine = None
            self._models_status = {}
            self.logger.error(f"Cannot initialize ML engine: {error_msg}")
            raise ImportError(error_msg)
        
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
        
        # Initialize in background thread only if dependencies are available
        if HAS_NUMPY_PANDAS and HAS_SKLEARN:
            self._init_thread = threading.Thread(target=self._initialize_engine, daemon=True)
            self._init_thread.start()
        else:
            self.logger.warning("ML initialization skipped due to missing dependencies")
    
    def _initialize_engine(self):
        """Initialize the ML engine and all models with comprehensive error handling."""
        try:
            with self._lock:
                self._status = "Initializing..."
                self._status_color = "orange"
                
            self.logger.info("Starting ML engine initialization")
            
            # Check system resources before initialization
            if HAS_PSUTIL:
                try:
                    resources = check_system_resources()
                    if resources.get('memory', {}).get('status') == 'critical':
                        self.logger.warning("Critical memory status detected")
                        error_handler.handle_error(
                            error=MemoryError("Insufficient memory for ML models"),
                            category=ErrorCategory.RESOURCE,
                            severity=ErrorSeverity.WARNING,
                            code=ErrorCode.INSUFFICIENT_MEMORY,
                            user_message="Limited memory available. ML features may be limited.",
                            recovery_suggestions=[
                                "Close other applications to free memory",
                                "ML models will run with reduced features"
                            ]
                        )
                except Exception as e:
                    self.logger.warning(f"Could not check system resources: {e}")
            else:
                self.logger.info("System resource monitoring not available (psutil not installed)")
            
            # Create ML engine instance
            try:
                self.logger.info("Creating MLEngine instance")
                # Set up paths for ML engine
                from pathlib import Path
                home_dir = Path.home()
                data_path = home_dir / '.laser_trim_analyzer' / 'ml_data'
                models_path = home_dir / '.laser_trim_analyzer' / 'models'
                
                # Ensure directories exist
                data_path.mkdir(parents=True, exist_ok=True)
                models_path.mkdir(parents=True, exist_ok=True)
                
                self.logger.info(f"ML data path: {data_path}")
                self.logger.info(f"ML models path: {models_path}")
                
                self.ml_engine = MLEngine(
                    data_path=str(data_path),
                    models_path=str(models_path),
                    logger=self.logger
                )
                self.logger.info("MLEngine instance created successfully")
            except ImportError as e:
                self.logger.error(f"Import error creating MLEngine: {e}")
                error_handler.handle_error(
                    error=e,
                    category=ErrorCategory.ML_MODEL,
                    severity=ErrorSeverity.ERROR,
                    code=ErrorCode.MODEL_INITIALIZATION_FAILED,
                    user_message="ML dependencies are missing. Please install scikit-learn and joblib.",
                    additional_data={'technical_details': str(e)},
                    recovery_suggestions=[
                        "Run: pip install scikit-learn joblib pandas numpy",
                        "Verify Python environment is correct",
                        "Check application logs for missing dependencies"
                    ]
                )
                raise
            except Exception as e:
                self.logger.error(f"Error creating MLEngine: {e}")
                error_handler.handle_error(
                    error=e,
                    category=ErrorCategory.ML_MODEL,
                    severity=ErrorSeverity.ERROR,
                    code=ErrorCode.MODEL_INITIALIZATION_FAILED,
                    user_message="Failed to initialize ML engine. ML features will be disabled.",
                    additional_data={'technical_details': str(e)},
                    recovery_suggestions=[
                        "Check ML dependencies are installed",
                        "Verify Python environment is correct",
                        "Restart the application"
                    ]
                )
                raise
            
            # Initialize each model
            success_count = 0
            failed_models = []
            
            self.logger.info(f"Initializing {len(self.available_models)} models")
            
            for model_name, model_info in self.available_models.items():
                try:
                    self.logger.info(f"Initializing model: {model_name}")
                    self._initialize_model(model_name, model_info)
                    success_count += 1
                    self.logger.info(f"Successfully initialized {model_name}")
                except MemoryError as e:
                    self.logger.error(f"Memory error initializing {model_name}: {e}")
                    failed_models.append((model_name, e))
                    self._models_status[model_name] = {
                        'status': 'Memory Error',
                        'trained': False,
                        'error': 'Insufficient memory to load model'
                    }
                    
                    # Try to free memory and continue
                    gc.collect()
                    
                except FileNotFoundError as e:
                    self.logger.warning(f"Model file not found for {model_name}: {e}")
                    failed_models.append((model_name, e))
                    self._models_status[model_name] = {
                        'status': 'Not Found',
                        'trained': False,
                        'error': 'Model file not found - training required'
                    }
                    
                except ImportError as e:
                    self.logger.error(f"Import error for {model_name}: {e}")
                    failed_models.append((model_name, e))
                    self._models_status[model_name] = {
                        'status': 'Import Error',
                        'trained': False,
                        'error': f'Missing dependency: {str(e)}',
                        'traceback': traceback.format_exc()
                    }
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize {model_name}: {e}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    failed_models.append((model_name, e))
                    self._models_status[model_name] = {
                        'status': 'Error',
                        'trained': False,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
            
            # Report failures if any
            if failed_models and model_info.get('required', False):
                error_handler.handle_error(
                    error=MLPredictionError(f"Failed to initialize {len(failed_models)} required models"),
                    category=ErrorCategory.ML_MODEL,
                    severity=ErrorSeverity.WARNING,
                    code=ErrorCode.MODEL_INITIALIZATION_FAILED,
                    user_message=f"Some ML models failed to initialize. Limited ML features available.",
                    recovery_suggestions=[
                        "Train models using the ML Tools page",
                        "Check model files in the models directory",
                        "Verify all dependencies are installed"
                    ],
                    additional_data={
                        'failed_models': [name for name, _ in failed_models],
                        'success_count': success_count,
                        'total_count': len(self.available_models)
                    }
                )
            
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
                
            error_handler.handle_error(
                error=e,
                category=ErrorCategory.ML_MODEL,
                severity=ErrorSeverity.ERROR,
                code=ErrorCode.MODEL_INITIALIZATION_FAILED,
                user_message="ML engine initialization failed. ML features will be unavailable.",
                additional_data={'technical_details': traceback.format_exc()},
                recovery_suggestions=[
                    "Check the application logs for details",
                    "Verify ML dependencies are installed",
                    "Try restarting the application"
                ]
            )
    
    def _initialize_model(self, model_name: str, model_info: Dict[str, Any]):
        """Initialize a specific model with error handling."""
        self.logger.info(f"Initializing model: {model_name}")
        
        try:
            # Create model config
            config = ModelConfig({
                'model_type': model_name,
                'version': '1.0.0',
                'features': self._get_default_features(model_name),
                'target': self._get_default_target(model_name)
            })
            
            # Create model instance
            model_class = model_info['class']
            self.logger.info(f"Creating instance of {model_class.__name__}")
            
            try:
                model = model_class(config, logger=self.logger)
            except TypeError as e:
                # Handle case where model class doesn't accept config/logger
                self.logger.warning(f"Model {model_name} doesn't accept config/logger, trying default init")
                model = model_class()
                
            self.logger.info(f"Successfully created {model_name} instance")
            
            # Register with ML engine
            if not hasattr(self.ml_engine, 'models'):
                self.ml_engine.models = {}
            self.ml_engine.models[model_name] = model
            
            # Check if model is trained (try to load from disk)
            model_path = self._get_model_path(model_name)
            is_trained = False
            load_error = None
            
            if model_path.exists():
                try:
                    # Validate model file before loading
                    if not self._validate_model_file(model_path):
                        raise ValueError("Model file validation failed")
                    
                    model.load(model_path)
                    is_trained = True
                    self.logger.info(f"Loaded trained model from {model_path}")
                    
                except (pickle.UnpicklingError, EOFError) as e:
                    load_error = "Model file is corrupted"
                    self.logger.error(f"Corrupted model file for {model_name}: {e}")
                    
                except MemoryError as e:
                    load_error = "Insufficient memory to load model"
                    self.logger.error(f"Memory error loading {model_name}: {e}")
                    raise  # Re-raise memory errors
                    
                except AttributeError as e:
                    load_error = "Model version mismatch"
                    self.logger.error(f"Version mismatch for {model_name}: {e}")
                    
                    error_handler.handle_error(
                        error=e,
                        category=ErrorCategory.ML_MODEL,
                        severity=ErrorSeverity.WARNING,
                        code=ErrorCode.MODEL_VERSION_MISMATCH,
                        user_message=f"Model '{model_name}' has version mismatch. Retraining required.",
                        recovery_suggestions=[
                            "Retrain the model using current data",
                            "Delete old model file and retrain"
                        ]
                    )
                    
                except Exception as e:
                    load_error = str(e)
                    self.logger.warning(f"Could not load saved model {model_name}: {e}")
            else:
                self.logger.info(f"No saved model found for {model_name} at {model_path}")
            
            # Update model status
            self._models_status[model_name] = {
                'status': 'Ready' if is_trained else 'Not Trained',
                'trained': is_trained,
                'description': model_info['description'],
                'last_training': datetime.now() if is_trained else None,
                'performance': getattr(model, 'performance_metrics', {}) if is_trained else {},
                'load_error': load_error,
                'model_path': str(model_path)
            }
            
            # If model is required but not trained, log warning
            if not is_trained and model_info.get('required', False):
                self.logger.warning(f"Required model '{model_name}' is not trained")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize model {model_name}: {e}")
            raise
    
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
    
    def save_model(self, model_name: str) -> bool:
        """Save a trained model to disk."""
        try:
            if model_name not in self.ml_engine.models:
                self.logger.error(f"Model {model_name} not found in ML engine")
                return False
                
            model = self.ml_engine.models[model_name]
            model_path = self._get_model_path(model_name)
            
            # Always save as .pkl for consistency
            model_path = model_path.parent / f"{model_name}.pkl"
            
            # Save the model
            if hasattr(model, 'save'):
                model.save(str(model_path))
                self.logger.info(f"Saved model {model_name} using model.save()")
            else:
                # Fallback to pickle
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                self.logger.info(f"Saved model {model_name} using pickle")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model {model_name}: {e}")
            return False
    
    def _get_model_path(self, model_name: str) -> Path:
        """Get the path where a model should be saved/loaded."""
        models_dir = Path.home() / '.laser_trim_analyzer' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        # Check for both .pkl and .joblib extensions
        pkl_path = models_dir / f"{model_name}.pkl"
        joblib_path = models_dir / f"{model_name}.joblib"
        
        # Prefer .pkl if it exists, otherwise use .joblib for backward compatibility
        if pkl_path.exists():
            return pkl_path
        else:
            return joblib_path
    
    def get_status(self) -> Dict[str, Any]:
        """Get current ML engine status."""
        with self._lock:
            status_dict = {
                'status': self._status,
                'color': self._status_color,
                'error': self._initialization_error,
                'models': self._models_status.copy(),
                'engine_ready': self.ml_engine is not None,
                'models_count': len(self._models_status),
                'trained_count': sum(1 for m in self._models_status.values() if m.get('trained', False))
            }
            
            # Add dependency information if missing
            if not HAS_NUMPY_PANDAS or not HAS_SKLEARN:
                missing_deps = []
                if not HAS_NUMPY_PANDAS:
                    missing_deps.extend(['numpy', 'pandas'])
                if not HAS_SKLEARN:
                    missing_deps.append('scikit-learn')
                if not HAS_JOBLIB:
                    missing_deps.append('joblib')
                
                status_dict['missing_dependencies'] = missing_deps
                status_dict['install_command'] = f"pip install {' '.join(missing_deps)}"
                
            return status_dict
    
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
    
    def _validate_model_file(self, model_path: Path) -> bool:
        """Validate a model file before loading."""
        try:
            # Check file exists and has reasonable size
            if not model_path.exists():
                return False
                
            file_size = model_path.stat().st_size
            if file_size == 0:
                self.logger.error(f"Model file {model_path} is empty")
                return False
                
            # Check file size is reasonable (not corrupted)
            if file_size > 500 * 1024 * 1024:  # 500MB limit
                self.logger.error(f"Model file {model_path} is unusually large: {file_size / 1024 / 1024:.1f}MB")
                return False
                
            # Try to read file header to check if it's a valid pickle/joblib file
            with open(model_path, 'rb') as f:
                header = f.read(16)
                # Check for joblib/pickle magic bytes
                if not (header.startswith(b'\x80') or b'joblib' in header or b'sklearn' in header):
                    self.logger.warning(f"Model file {model_path} may not be a valid model file")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating model file {model_path}: {e}")
            return False
    
    @handle_errors(
        category=ErrorCategory.ML_MODEL,
        severity=ErrorSeverity.ERROR,
        max_retries=2
    )
    def predict(self, model_name: str, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Make predictions with comprehensive error handling."""
        # Validate inputs
        if not model_name:
            raise ValueError("Model name is required")
            
        if data is None or data.empty:
            raise ValueError("Input data is required and cannot be empty")
            
        # Check if ML engine is initialized
        if not self.ml_engine:
            error_handler.handle_error(
                error=MLPredictionError("ML engine not initialized"),
                category=ErrorCategory.ML_MODEL,
                severity=ErrorSeverity.WARNING,
                code=ErrorCode.MODEL_NOT_FOUND,
                user_message="ML predictions unavailable - engine not initialized",
                recovery_suggestions=[
                    "Wait for ML engine to initialize",
                    "Check application logs for initialization errors"
                ]
            )
            return None, None
            
        # Get model
        model = self.get_model(model_name)
        if not model:
            error_handler.handle_error(
                error=MLPredictionError(f"Model '{model_name}' not found"),
                category=ErrorCategory.ML_MODEL,
                severity=ErrorSeverity.WARNING,
                code=ErrorCode.MODEL_NOT_FOUND,
                user_message=f"ML model '{model_name}' is not available",
                recovery_suggestions=[
                    "Check if the model is properly installed",
                    "Try restarting the application"
                ]
            )
            return None, None
            
        # Check if model is trained
        model_status = self._models_status.get(model_name, {})
        if not model_status.get('trained', False):
            error_handler.handle_error(
                error=MLPredictionError(f"Model '{model_name}' is not trained"),
                category=ErrorCategory.ML_MODEL,
                severity=ErrorSeverity.WARNING,
                code=ErrorCode.INSUFFICIENT_TRAINING_DATA,
                user_message=f"ML model '{model_name}' needs training before use",
                recovery_suggestions=[
                    "Train the model using the ML Tools page",
                    "Load a pre-trained model"
                ]
            )
            return None, None
            
        try:
            # Get required features
            required_features = self._get_default_features(model_name)
            
            # Validate features exist in data
            missing_features = set(required_features) - set(data.columns)
            if missing_features:
                error_handler.handle_error(
                    error=MLPredictionError(f"Missing features: {missing_features}"),
                    category=ErrorCategory.ML_MODEL,
                    severity=ErrorSeverity.ERROR,
                    code=ErrorCode.FEATURE_MISMATCH,
                    user_message="Input data missing required features for ML prediction",
                    recovery_suggestions=[
                        "Ensure all required features are calculated",
                        "Check data processing pipeline"
                    ],
                    additional_data={
                        'required_features': required_features,
                        'provided_features': list(data.columns),
                        'missing_features': list(missing_features)
                    }
                )
                return None, None
                
            # Extract features
            X = data[required_features]
            
            # Check for invalid values
            if X.isnull().any().any():
                null_features = X.columns[X.isnull().any()].tolist()
                self.logger.warning(f"Null values found in features: {null_features}")
                # Fill nulls with appropriate defaults
                X = X.fillna(X.mean())
                
            # Make prediction
            predictions = model.predict(X)
            
            # Get additional metadata if available
            metadata = {}
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X)
                    metadata['probabilities'] = probabilities
                except Exception as e:
                    self.logger.debug(f"Could not get prediction probabilities: {e}")
                    
            if hasattr(model, 'feature_importance_'):
                metadata['feature_importance'] = dict(zip(required_features, model.feature_importance_))
                
            return predictions, metadata
            
        except MemoryError as e:
            error_handler.handle_error(
                error=e,
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
                code=ErrorCode.INSUFFICIENT_MEMORY,
                user_message="Not enough memory to run ML prediction",
                recovery_suggestions=[
                    "Close other applications",
                    "Process smaller batches of data",
                    "Reduce model complexity"
                ]
            )
            raise
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {model_name}: {e}")
            error_handler.handle_error(
                error=e,
                category=ErrorCategory.ML_MODEL,
                severity=ErrorSeverity.ERROR,
                code=ErrorCode.MODEL_PREDICTION_FAILED,
                user_message=f"ML prediction failed: {str(e)}",
                additional_data={
                    'technical_details': traceback.format_exc(),
                    'model_name': model_name,
                    'data_shape': data.shape,
                    'error_type': type(e).__name__
                }
            )
            raise MLPredictionError(f"Prediction failed: {str(e)}") from e
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        if model_name not in self._models_status:
            return {
                'error': f"Model '{model_name}' not found",
                'available_models': list(self.available_models.keys())
            }
            
        status = self._models_status[model_name].copy()
        
        # Add additional info
        status['features'] = self._get_default_features(model_name)
        status['target'] = self._get_default_target(model_name)
        
        # Add model size if available
        model_path = self._get_model_path(model_name)
        if model_path.exists():
            status['model_size_mb'] = model_path.stat().st_size / (1024 * 1024)
            
        return status
    
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