"""
Cache integration examples and utilities for the laser trim analyzer.

This module demonstrates how to integrate the cache manager with various
components of the application.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from .cache_manager import get_cache_manager, initialize_cache_manager
from .models import ProcessingResult, AnalysisResult
from ..utils.file_utils import calculate_file_hash


class CachedFileProcessor:
    """File processor with integrated caching."""
    
    def __init__(self):
        self.cache = get_cache_manager()
    
    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process file with caching."""
        file_path = Path(file_path)
        
        # Check if file content is cached
        cached_content = self.cache.get_file_content(file_path)
        if cached_content:
            return self._process_cached_content(cached_content)
        
        # Process file normally
        result = self._process_file_impl(file_path)
        
        # Cache the processed content
        self.cache.cache_file_content(file_path, result.data, ttl=3600)
        
        return result
    
    def _process_file_impl(self, file_path: Path) -> ProcessingResult:
        """Actual file processing implementation."""
        # This would contain the actual file processing logic
        pass
    
    def _process_cached_content(self, content: Any) -> ProcessingResult:
        """Process already cached content."""
        # This would recreate the ProcessingResult from cached data
        pass


class CachedMLPredictor:
    """ML predictor with integrated caching."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.cache = get_cache_manager()
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make prediction with caching."""
        # Create hash of input data
        input_hash = self._hash_input(data)
        
        # Check cache
        cached_prediction = self.cache.get_model_prediction(
            self.model_name, input_hash
        )
        if cached_prediction is not None:
            return cached_prediction
        
        # Make actual prediction
        prediction = self._predict_impl(data)
        
        # Cache result
        self.cache.cache_model_prediction(
            self.model_name, input_hash, prediction, ttl=7200  # 2 hours
        )
        
        return prediction
    
    def _hash_input(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """Create hash of input data for caching."""
        if isinstance(data, pd.DataFrame):
            # Hash DataFrame
            return hashlib.sha256(
                pd.util.hash_pandas_object(data).values.tobytes()
            ).hexdigest()
        else:
            # Hash numpy array
            return hashlib.sha256(data.tobytes()).hexdigest()
    
    def _predict_impl(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Actual prediction implementation."""
        # This would contain the actual ML prediction logic
        pass


class CachedAnalyzer:
    """Analyzer with integrated result caching."""
    
    def __init__(self):
        self.cache = get_cache_manager()
    
    def analyze(self, data: Dict[str, Any], analysis_type: str) -> AnalysisResult:
        """Perform analysis with caching."""
        # Create analysis ID
        analysis_id = self._create_analysis_id(data, analysis_type)
        
        # Check cache
        cached_result = self.cache.get_analysis_result(analysis_id)
        if cached_result:
            return cached_result
        
        # Perform analysis
        result = self._analyze_impl(data, analysis_type)
        
        # Cache result
        self.cache.cache_analysis_result(
            analysis_id, result, ttl=86400  # 24 hours
        )
        
        return result
    
    def _create_analysis_id(self, data: Dict[str, Any], analysis_type: str) -> str:
        """Create unique ID for analysis."""
        # Create deterministic hash of input data
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        return f"{analysis_type}:{data_hash}"
    
    def _analyze_impl(self, data: Dict[str, Any], analysis_type: str) -> AnalysisResult:
        """Actual analysis implementation."""
        # This would contain the actual analysis logic
        pass


class CachedConfigManager:
    """Configuration manager with caching."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.cache = get_cache_manager()
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration with caching."""
        cache_key = f"config:{self.config_path}:{section or 'all'}"
        
        # Check cache
        cached = self.cache.get(cache_key, cache_type='config')
        if cached:
            return cached
        
        # Load config
        config = self._load_config(section)
        
        # Cache with short TTL
        self.cache.set(cache_key, config, cache_type='config', ttl=300)
        
        return config
    
    def _load_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file."""
        # This would contain the actual config loading logic
        pass


def setup_application_cache(
    memory_size_mb: int = 200,
    file_cache_size_mb: int = 1000,
    cache_dir: Optional[Path] = None
) -> None:
    """
    Setup application-wide cache with recommended settings.
    
    Args:
        memory_size_mb: Memory cache size in MB
        file_cache_size_mb: File cache size in MB
        cache_dir: Directory for file cache
    """
    initialize_cache_manager(
        memory_cache_size=memory_size_mb * 1024 * 1024,
        file_cache_size=file_cache_size_mb * 1024 * 1024,
        cache_dir=cache_dir,
        monitor_memory=True,
        memory_threshold=0.85
    )


# Decorator examples
cache_manager = get_cache_manager()


@cache_manager.decorator(cache_type='general', ttl=600)
def expensive_calculation(x: float, y: float) -> float:
    """Example of cached function."""
    import time
    time.sleep(1)  # Simulate expensive operation
    return x ** y + y ** x


@cache_manager.decorator(
    cache_type='model_predictions',
    ttl=3600,
    key_func=lambda model, data: f"{model}:{hashlib.sha256(str(data).encode()).hexdigest()[:8]}"
)
def cached_model_predict(model: str, data: List[float]) -> float:
    """Example of cached ML prediction."""
    # Simulate model prediction
    return sum(data) / len(data)


class SmartCacheInvalidator:
    """Smart cache invalidation based on file changes and dependencies."""
    
    def __init__(self):
        self.cache = get_cache_manager()
        self.dependencies: Dict[str, List[str]] = {}
    
    def register_dependency(self, cache_key: str, depends_on: Union[str, List[str]]):
        """Register cache dependencies."""
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        
        for dep in depends_on:
            if dep not in self.dependencies:
                self.dependencies[dep] = []
            self.dependencies[dep].append(cache_key)
    
    def invalidate_dependents(self, changed_item: str):
        """Invalidate all cache entries that depend on changed item."""
        if changed_item in self.dependencies:
            for cache_key in self.dependencies[changed_item]:
                # Try to delete from all cache types
                for cache_type in ['general', 'file_content', 'model_predictions', 
                                  'config', 'analysis_results']:
                    self.cache.delete(cache_key, cache_type=cache_type)
    
    def invalidate_file_caches(self, file_path: Union[str, Path]):
        """Invalidate all caches related to a file."""
        file_path = str(Path(file_path).resolve())
        
        # Invalidate file content cache
        patterns = [
            f"file:{file_path}:*",
            f"analysis:*{Path(file_path).stem}*",
            f"model:*{Path(file_path).stem}*"
        ]
        
        for pattern in patterns:
            for cache_type in ['file_content', 'analysis_results', 'model_predictions']:
                self.cache.invalidate_pattern(pattern, cache_type)


# Usage example
if __name__ == "__main__":
    # Initialize cache
    setup_application_cache()
    
    # Get cache stats
    cache = get_cache_manager()
    stats = cache.get_stats()
    print("Cache Statistics:")
    for name, info in stats.items():
        print(f"\n{name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Example usage
    result1 = expensive_calculation(2, 3)  # Takes 1 second
    result2 = expensive_calculation(2, 3)  # Returns instantly from cache
    
    # Clear specific cache
    cache.clear('general')
    
    # Invalidate pattern
    cache.invalidate_pattern("file:*/test_*", 'file_content')