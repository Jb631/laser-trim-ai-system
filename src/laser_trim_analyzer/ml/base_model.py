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
            
            # Save model state
            model_data = {
                'model': self.model,
                'metadata': self.get_metadata(),
                'training_date': self.training_date,
                'is_trained': self.is_trained
            }
            
            # Save scaler if exists
            if hasattr(self, 'scaler'):
                model_data['scaler'] = self.scaler
                
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
                
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data['model']
            self.metadata = model_data.get('metadata', {})
            self.training_date = model_data.get('training_date')
            self.is_trained = model_data.get('is_trained', True)
            
            # Load scaler if exists
            if 'scaler' in model_data:
                self.scaler = model_data['scaler']
                
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False