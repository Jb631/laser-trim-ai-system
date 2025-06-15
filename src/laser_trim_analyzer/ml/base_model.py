"""
Base ML Model Abstract Class

Separated from engine.py to avoid circular imports.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import logging
from datetime import datetime


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