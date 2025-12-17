"""
Base ML Model Abstract Class

Separated from engine.py to avoid circular imports.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import logging
from datetime import datetime
import pickle
from pathlib import Path


class BaseMLModel(ABC):
    """Abstract base class for all ML models in the system."""
    
    def __init__(self, config: Any, logger: Optional[logging.Logger] = None):
        """Initialize base ML model."""
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.model = None
        self.is_trained = False
        self.training_date = None
        self.metadata = {}
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the model. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Make predictions. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance. Must be implemented by subclasses."""
        pass
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            'class': self.__class__.__name__,
            'is_trained': self.is_trained,
            'training_date': self.training_date,
            **self.metadata
        }
    
    def save(self, filepath) -> bool:
        """Save the trained model to disk."""
        try:
            if not self.is_trained:
                self.logger.warning("Attempting to save untrained model")
                return False
                
            # Convert to Path object if string
            from pathlib import Path
            filepath = Path(filepath)
                
            # Create directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state with additional metadata
            model_data = {
                'model': self.model,
                'metadata': self.get_metadata(),
                'training_date': self.training_date,
                'is_trained': self.is_trained,
                'version': getattr(self, 'version', '1.0.0'),
                'last_trained': getattr(self, 'last_trained', self.training_date),
                'training_samples': getattr(self, 'training_samples', 0),
                'performance_metrics': getattr(self, 'performance_metrics', {}),
                'model_type': getattr(self, 'model_type', self.__class__.__name__)
            }
            
            # Save scaler if exists
            if hasattr(self, 'scaler'):
                model_data['scaler'] = self.scaler
            
            # Try joblib first, fallback to pickle
            try:
                import joblib
                joblib.dump(model_data, filepath)
            except ImportError:
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
                
            self.logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load(self, filepath) -> bool:
        """Load a trained model from disk."""
        try:
            # Convert to Path object if string
            from pathlib import Path
            filepath = Path(filepath)
            
            if not filepath.exists():
                self.logger.error(f"Model file not found: {filepath}")
                return False
            
            # Try joblib first, fallback to pickle
            try:
                import joblib
                model_data = joblib.load(filepath)
            except (ImportError, Exception):
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
            self.model = model_data['model']
            self.metadata = model_data.get('metadata', {})
            self.training_date = model_data.get('training_date')
            self.is_trained = model_data.get('is_trained', True)
            
            # Load additional metadata if present
            if 'version' in model_data:
                self.version = model_data['version']
            if 'last_trained' in model_data:
                self.last_trained = model_data['last_trained']
            if 'training_samples' in model_data:
                self.training_samples = model_data['training_samples']
            if 'performance_metrics' in model_data:
                self.performance_metrics = model_data['performance_metrics']
            if 'model_type' in model_data:
                self.model_type = model_data['model_type']
            
            # Load scaler if exists
            if 'scaler' in model_data:
                self.scaler = model_data['scaler']
                
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False