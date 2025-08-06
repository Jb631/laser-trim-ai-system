"""
Memory optimization utilities for ML operations.

This module provides tools and utilities for reducing memory usage during
machine learning operations, including feature extraction, model management,
and batch prediction with memory constraints.
"""

import gc
import os
import sys
import json
import pickle
import psutil
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import torch
import joblib
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.random_projection import SparseRandomProjection
import h5py
import zarr

logger = logging.getLogger(__name__)


@dataclass
class MemoryProfile:
    """Memory usage profile for ML operations."""
    operation_name: str
    peak_memory_mb: float
    average_memory_mb: float
    duration_seconds: float
    num_samples: int
    features_shape: Optional[Tuple[int, ...]] = None
    model_size_mb: Optional[float] = None
    timestamp: float = field(default_factory=lambda: pd.Timestamp.now().timestamp())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation_name': self.operation_name,
            'peak_memory_mb': self.peak_memory_mb,
            'average_memory_mb': self.average_memory_mb,
            'duration_seconds': self.duration_seconds,
            'num_samples': self.num_samples,
            'features_shape': self.features_shape,
            'model_size_mb': self.model_size_mb,
            'timestamp': self.timestamp
        }


class MemoryMonitor:
    """Monitor memory usage during operations."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.peak_memory = 0
        self.measurements = []
        
    def start(self):
        """Start memory monitoring."""
        gc.collect()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.measurements = [self.initial_memory]
        
    def measure(self):
        """Take a memory measurement."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.measurements.append(current_memory)
        self.peak_memory = max(self.peak_memory, current_memory)
        return current_memory
        
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        if not self.measurements:
            return {'peak_mb': 0, 'average_mb': 0, 'delta_mb': 0}
            
        return {
            'peak_mb': self.peak_memory,
            'average_mb': np.mean(self.measurements),
            'delta_mb': self.peak_memory - self.initial_memory
        }


@contextmanager
def memory_profiler(operation_name: str) -> Iterator[MemoryProfile]:
    """Context manager for profiling memory usage."""
    import time
    
    monitor = MemoryMonitor()
    monitor.start()
    start_time = time.time()
    
    profile = MemoryProfile(
        operation_name=operation_name,
        peak_memory_mb=0,
        average_memory_mb=0,
        duration_seconds=0,
        num_samples=0
    )
    
    try:
        yield profile
    finally:
        duration = time.time() - start_time
        stats = monitor.get_stats()
        
        profile.peak_memory_mb = stats['peak_mb']
        profile.average_memory_mb = stats['average_mb']
        profile.duration_seconds = duration
        
        logger.info(f"Memory profile for '{operation_name}': "
                   f"Peak: {stats['peak_mb']:.2f}MB, "
                   f"Avg: {stats['average_mb']:.2f}MB, "
                   f"Duration: {duration:.2f}s")


class DataGenerator:
    """Memory-efficient data generator for feature extraction."""
    
    def __init__(self, data_source: Union[str, pd.DataFrame, np.ndarray],
                 batch_size: int = 32,
                 feature_columns: Optional[List[str]] = None,
                 preprocessor: Optional[Callable] = None):
        """
        Initialize data generator.
        
        Args:
            data_source: Path to data file or data array/dataframe
            batch_size: Number of samples per batch
            feature_columns: Columns to use as features (for DataFrame)
            preprocessor: Optional preprocessing function
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.feature_columns = feature_columns
        self.preprocessor = preprocessor
        self._setup_data_source()
        
    def _setup_data_source(self):
        """Setup data source for iteration."""
        if isinstance(self.data_source, str):
            # File-based data source
            self.data_path = Path(self.data_source)
            if self.data_path.suffix == '.h5':
                self._setup_h5_source()
            elif self.data_path.suffix == '.zarr':
                self._setup_zarr_source()
            elif self.data_path.suffix in ['.csv', '.parquet']:
                self._setup_pandas_source()
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        else:
            # In-memory data source
            self.data = self.data_source
            self.num_samples = len(self.data)
            
    def _setup_h5_source(self):
        """Setup HDF5 data source."""
        with h5py.File(self.data_path, 'r') as f:
            self.num_samples = f['features'].shape[0]
            self.feature_shape = f['features'].shape[1:]
            
    def _setup_zarr_source(self):
        """Setup Zarr data source."""
        store = zarr.open(self.data_path, mode='r')
        self.num_samples = store['features'].shape[0]
        self.feature_shape = store['features'].shape[1:]
        
    def _setup_pandas_source(self):
        """Setup pandas data source."""
        # Just get metadata, don't load full dataset
        if self.data_path.suffix == '.csv':
            df_sample = pd.read_csv(self.data_path, nrows=5)
        else:
            df_sample = pd.read_parquet(self.data_path, columns=self.feature_columns)
            
        self.num_samples = sum(1 for _ in open(self.data_path)) - 1  # Rough estimate
        if self.feature_columns:
            self.feature_shape = (len(self.feature_columns),)
        else:
            self.feature_shape = (len(df_sample.columns),)
            
    def __len__(self) -> int:
        """Get number of batches."""
        return (self.num_samples + self.batch_size - 1) // self.batch_size
        
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over batches."""
        if isinstance(self.data_source, str):
            yield from self._iterate_file()
        else:
            yield from self._iterate_memory()
            
    def _iterate_file(self) -> Iterator[np.ndarray]:
        """Iterate over file-based data."""
        if self.data_path.suffix == '.h5':
            with h5py.File(self.data_path, 'r') as f:
                dataset = f['features']
                for i in range(0, self.num_samples, self.batch_size):
                    batch = dataset[i:i + self.batch_size]
                    if self.preprocessor:
                        batch = self.preprocessor(batch)
                    yield batch
                    
        elif self.data_path.suffix == '.zarr':
            store = zarr.open(self.data_path, mode='r')
            dataset = store['features']
            for i in range(0, self.num_samples, self.batch_size):
                batch = dataset[i:i + self.batch_size]
                if self.preprocessor:
                    batch = self.preprocessor(batch)
                yield batch
                
        elif self.data_path.suffix == '.csv':
            for chunk in pd.read_csv(self.data_path, chunksize=self.batch_size):
                if self.feature_columns:
                    batch = chunk[self.feature_columns].values
                else:
                    batch = chunk.values
                if self.preprocessor:
                    batch = self.preprocessor(batch)
                yield batch
                
    def _iterate_memory(self) -> Iterator[np.ndarray]:
        """Iterate over in-memory data."""
        for i in range(0, self.num_samples, self.batch_size):
            if isinstance(self.data, pd.DataFrame):
                if self.feature_columns:
                    batch = self.data.iloc[i:i + self.batch_size][self.feature_columns].values
                else:
                    batch = self.data.iloc[i:i + self.batch_size].values
            else:
                batch = self.data[i:i + self.batch_size]
                
            if self.preprocessor:
                batch = self.preprocessor(batch)
            yield batch


class ModelOptimizer:
    """Optimize models for memory efficiency."""
    
    @staticmethod
    def quantize_model(model: torch.nn.Module, 
                      quantization_type: str = 'dynamic') -> torch.nn.Module:
        """
        Quantize PyTorch model to reduce memory usage.
        
        Args:
            model: PyTorch model to quantize
            quantization_type: Type of quantization ('dynamic', 'static', 'qat')
            
        Returns:
            Quantized model
        """
        if quantization_type == 'dynamic':
            # Dynamic quantization (easiest, works for most models)
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        elif quantization_type == 'static':
            # Static quantization (requires calibration)
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            # Note: Calibration would happen here with representative data
            torch.quantization.convert(model, inplace=True)
            quantized_model = model
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
            
        # Log size reduction
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024 / 1024
        logger.info(f"Model quantization: {original_size:.2f}MB -> {quantized_size:.2f}MB "
                   f"({(1 - quantized_size/original_size) * 100:.1f}% reduction)")
        
        return quantized_model
    
    @staticmethod
    def prune_model(model: torch.nn.Module, 
                   pruning_ratio: float = 0.1) -> torch.nn.Module:
        """
        Prune model weights to reduce memory usage.
        
        Args:
            model: PyTorch model to prune
            pruning_ratio: Fraction of weights to prune
            
        Returns:
            Pruned model
        """
        import torch.nn.utils.prune as prune
        
        parameters_to_prune = []
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
                
        # Apply structured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        
        # Remove pruning reparameterization to make permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
            
        # Calculate sparsity
        total_params = 0
        pruned_params = 0
        for module, param_name in parameters_to_prune:
            param = getattr(module, param_name)
            total_params += param.numel()
            pruned_params += (param == 0).sum().item()
            
        sparsity = pruned_params / total_params
        logger.info(f"Model pruning completed: {sparsity * 100:.1f}% sparsity achieved")
        
        return model
    
    @staticmethod
    def compress_sklearn_model(model: BaseEstimator, 
                             compression_level: int = 3) -> bytes:
        """
        Compress sklearn model for storage.
        
        Args:
            model: Sklearn model to compress
            compression_level: Joblib compression level (0-9)
            
        Returns:
            Compressed model bytes
        """
        import io
        buffer = io.BytesIO()
        joblib.dump(model, buffer, compress=compression_level)
        compressed_bytes = buffer.getvalue()
        
        # Log compression stats
        uncompressed_size = len(pickle.dumps(model))
        compressed_size = len(compressed_bytes)
        logger.info(f"Model compression: {uncompressed_size/1024/1024:.2f}MB -> "
                   f"{compressed_size/1024/1024:.2f}MB "
                   f"({(1 - compressed_size/uncompressed_size) * 100:.1f}% reduction)")
        
        return compressed_bytes


class BatchPredictor:
    """Memory-efficient batch prediction with memory limits."""
    
    def __init__(self, model: Any, 
                 memory_limit_mb: float = 1024,
                 optimal_batch_size: Optional[int] = None):
        """
        Initialize batch predictor.
        
        Args:
            model: Model for prediction
            memory_limit_mb: Maximum memory usage in MB
            optimal_batch_size: Pre-computed optimal batch size
        """
        self.model = model
        self.memory_limit_mb = memory_limit_mb
        self.optimal_batch_size = optimal_batch_size
        self.memory_monitor = MemoryMonitor()
        
    def _estimate_batch_size(self, sample_features: np.ndarray) -> int:
        """Estimate optimal batch size based on memory constraints."""
        if self.optimal_batch_size:
            return self.optimal_batch_size
            
        # Estimate memory per sample
        sample_memory = sample_features.nbytes / 1024 / 1024  # MB
        
        # Account for model overhead (rough estimate)
        if hasattr(self.model, 'get_params'):
            # Sklearn model
            model_overhead = 2.0  # Conservative multiplier
        else:
            # Deep learning model
            model_overhead = 3.0
            
        # Calculate batch size with safety margin
        safety_margin = 0.7
        estimated_batch_size = int(
            (self.memory_limit_mb * safety_margin) / (sample_memory * model_overhead)
        )
        
        # Ensure reasonable batch size
        return max(1, min(estimated_batch_size, 10000))
        
    def predict(self, data: Union[np.ndarray, DataGenerator], 
                return_generator: bool = False) -> Union[np.ndarray, Iterator[np.ndarray]]:
        """
        Perform memory-efficient batch prediction.
        
        Args:
            data: Input data or data generator
            return_generator: Whether to return a generator instead of full results
            
        Returns:
            Predictions array or generator
        """
        if isinstance(data, DataGenerator):
            return self._predict_generator(data, return_generator)
        else:
            return self._predict_array(data, return_generator)
            
    def _predict_array(self, data: np.ndarray, 
                      return_generator: bool) -> Union[np.ndarray, Iterator[np.ndarray]]:
        """Predict on numpy array."""
        batch_size = self._estimate_batch_size(data[0:1])
        logger.info(f"Using batch size: {batch_size} for prediction")
        
        def prediction_generator():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                # Monitor memory
                self.memory_monitor.measure()
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                if current_memory > self.memory_limit_mb:
                    logger.warning(f"Memory usage ({current_memory:.0f}MB) exceeds limit "
                                 f"({self.memory_limit_mb:.0f}MB), triggering garbage collection")
                    gc.collect()
                    
                # Make prediction
                if hasattr(self.model, 'predict'):
                    batch_pred = self.model.predict(batch)
                else:
                    # PyTorch model
                    self.model.eval()
                    with torch.no_grad():
                        batch_tensor = torch.from_numpy(batch).float()
                        batch_pred = self.model(batch_tensor).numpy()
                        
                yield batch_pred
                
        if return_generator:
            return prediction_generator()
        else:
            # Collect all predictions
            predictions = []
            for batch_pred in prediction_generator():
                predictions.append(batch_pred)
            return np.vstack(predictions)
            
    def _predict_generator(self, data_gen: DataGenerator, 
                         return_generator: bool) -> Union[np.ndarray, Iterator[np.ndarray]]:
        """Predict on data generator."""
        def prediction_generator():
            for batch in data_gen:
                # Monitor memory
                self.memory_monitor.measure()
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                if current_memory > self.memory_limit_mb:
                    logger.warning(f"Memory usage ({current_memory:.0f}MB) exceeds limit "
                                 f"({self.memory_limit_mb:.0f}MB), triggering garbage collection")
                    gc.collect()
                    
                # Make prediction
                if hasattr(self.model, 'predict'):
                    batch_pred = self.model.predict(batch)
                else:
                    # PyTorch model
                    self.model.eval()
                    with torch.no_grad():
                        batch_tensor = torch.from_numpy(batch).float()
                        batch_pred = self.model(batch_tensor).numpy()
                        
                yield batch_pred
                
        if return_generator:
            return prediction_generator()
        else:
            # Collect all predictions
            predictions = []
            for batch_pred in prediction_generator():
                predictions.append(batch_pred)
            return np.vstack(predictions)


class FeatureOptimizer:
    """Optimize feature storage and processing."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize feature optimizer.
        
        Args:
            storage_path: Path for storing optimized features
        """
        self.storage_path = storage_path or Path.cwd() / 'optimized_features'
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def compress_features(self, features: np.ndarray, 
                        method: str = 'pca',
                        compression_ratio: float = 0.5) -> Tuple[np.ndarray, Any]:
        """
        Compress features to reduce memory usage.
        
        Args:
            features: Input features
            method: Compression method ('pca', 'incremental_pca', 'random_projection')
            compression_ratio: Target compression ratio
            
        Returns:
            Compressed features and transformer
        """
        n_samples, n_features = features.shape
        n_components = int(n_features * compression_ratio)
        
        with memory_profiler(f"feature_compression_{method}") as profile:
            profile.num_samples = n_samples
            profile.features_shape = features.shape
            
            if method == 'pca':
                transformer = PCA(n_components=n_components)
                compressed = transformer.fit_transform(features)
                
            elif method == 'incremental_pca':
                transformer = IncrementalPCA(n_components=n_components, batch_size=1000)
                compressed = transformer.fit_transform(features)
                
            elif method == 'random_projection':
                transformer = SparseRandomProjection(n_components=n_components)
                compressed = transformer.fit_transform(features)
                
            else:
                raise ValueError(f"Unknown compression method: {method}")
                
        # Log compression stats
        original_size = features.nbytes / 1024 / 1024
        compressed_size = compressed.nbytes / 1024 / 1024
        logger.info(f"Feature compression: {original_size:.2f}MB -> {compressed_size:.2f}MB "
                   f"({(1 - compressed_size/original_size) * 100:.1f}% reduction)")
        
        return compressed, transformer
        
    def save_features_efficient(self, features: np.ndarray, 
                              name: str,
                              format: str = 'h5',
                              chunk_size: int = 1000,
                              compression: str = 'gzip') -> Path:
        """
        Save features efficiently to disk.
        
        Args:
            features: Features to save
            name: Name for the feature file
            format: Storage format ('h5', 'zarr', 'memmap')
            chunk_size: Chunk size for storage
            compression: Compression algorithm
            
        Returns:
            Path to saved features
        """
        file_path = self.storage_path / f"{name}.{format}"
        
        with memory_profiler(f"save_features_{format}") as profile:
            profile.num_samples = features.shape[0]
            profile.features_shape = features.shape
            
            if format == 'h5':
                with h5py.File(file_path, 'w') as f:
                    f.create_dataset('features', data=features, 
                                   chunks=(chunk_size, features.shape[1]),
                                   compression=compression)
                    f.attrs['shape'] = features.shape
                    f.attrs['dtype'] = str(features.dtype)
                    
            elif format == 'zarr':
                zarr.save_array(file_path, features, 
                              chunks=(chunk_size, features.shape[1]),
                              compressor=zarr.Blosc(cname='zstd', clevel=3))
                              
            elif format == 'memmap':
                # Memory-mapped file for large datasets
                memmap = np.memmap(file_path, dtype=features.dtype, 
                                 mode='w+', shape=features.shape)
                memmap[:] = features
                del memmap  # Flush to disk
                
                # Save metadata
                meta_path = file_path.with_suffix('.meta')
                with open(meta_path, 'w') as f:
                    json.dump({
                        'shape': features.shape,
                        'dtype': str(features.dtype)
                    }, f)
                    
            else:
                raise ValueError(f"Unknown format: {format}")
                
        logger.info(f"Features saved to {file_path} "
                   f"(size: {file_path.stat().st_size / 1024 / 1024:.2f}MB)")
        
        return file_path
        
    def load_features_lazy(self, file_path: Path) -> Union[h5py.Dataset, zarr.Array, np.memmap]:
        """
        Load features lazily without loading into memory.
        
        Args:
            file_path: Path to feature file
            
        Returns:
            Lazy-loaded feature array
        """
        suffix = file_path.suffix
        
        if suffix == '.h5':
            f = h5py.File(file_path, 'r')
            return f['features']
            
        elif suffix == '.zarr':
            return zarr.open(file_path, mode='r')
            
        elif suffix == '.memmap':
            # Load metadata
            meta_path = file_path.with_suffix('.meta')
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                
            return np.memmap(file_path, dtype=meta['dtype'], 
                            mode='r', shape=tuple(meta['shape']))
                            
        else:
            raise ValueError(f"Unknown file format: {suffix}")


class ModelManager:
    """Manage model loading/unloading for memory efficiency."""
    
    def __init__(self, cache_dir: Optional[Path] = None,
                 max_loaded_models: int = 3):
        """
        Initialize model manager.
        
        Args:
            cache_dir: Directory for model cache
            max_loaded_models: Maximum number of models to keep in memory
        """
        self.cache_dir = cache_dir or Path.cwd() / 'model_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_loaded_models = max_loaded_models
        
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = Lock()
        
    def register_model(self, name: str, model_path: Path, 
                      metadata: Optional[Dict] = None):
        """Register a model for management."""
        self.model_metadata[name] = {
            'path': model_path,
            'metadata': metadata or {},
            'size_mb': model_path.stat().st_size / 1024 / 1024
        }
        
    def load_model(self, name: str) -> Any:
        """
        Load model with automatic memory management.
        
        Args:
            name: Model name
            
        Returns:
            Loaded model
        """
        with self.lock:
            # Check if already loaded
            if name in self.loaded_models:
                self.access_times[name] = pd.Timestamp.now().timestamp()
                return self.loaded_models[name]
                
            # Check if we need to unload models
            if len(self.loaded_models) >= self.max_loaded_models:
                self._evict_oldest_model()
                
            # Load model
            model = self._load_model_from_disk(name)
            self.loaded_models[name] = model
            self.access_times[name] = pd.Timestamp.now().timestamp()
            
            return model
            
    def _load_model_from_disk(self, name: str) -> Any:
        """Load model from disk."""
        if name not in self.model_metadata:
            raise ValueError(f"Model '{name}' not registered")
            
        model_info = self.model_metadata[name]
        model_path = model_info['path']
        
        with memory_profiler(f"load_model_{name}") as profile:
            profile.model_size_mb = model_info['size_mb']
            
            if model_path.suffix == '.pkl':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif model_path.suffix == '.joblib':
                model = joblib.load(model_path)
            elif model_path.suffix in ['.pt', '.pth']:
                model = torch.load(model_path, map_location='cpu')
            else:
                raise ValueError(f"Unknown model format: {model_path.suffix}")
                
        logger.info(f"Loaded model '{name}' ({model_info['size_mb']:.2f}MB)")
        return model
        
    def _evict_oldest_model(self):
        """Evict least recently used model."""
        if not self.loaded_models:
            return
            
        # Find oldest model
        oldest_name = min(self.access_times.items(), key=lambda x: x[1])[0]
        
        # Unload model
        self.unload_model(oldest_name)
        
    def unload_model(self, name: str):
        """Unload model from memory."""
        if name in self.loaded_models:
            del self.loaded_models[name]
            del self.access_times[name]
            gc.collect()
            logger.info(f"Unloaded model '{name}'")
            
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        return list(self.loaded_models.keys())
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage of loaded models."""
        usage = {}
        for name in self.loaded_models:
            if name in self.model_metadata:
                usage[name] = self.model_metadata[name]['size_mb']
        return usage
        
    @contextmanager
    def temporary_model(self, name: str):
        """Context manager for temporary model loading."""
        model = self.load_model(name)
        try:
            yield model
        finally:
            # Always unload after use
            self.unload_model(name)


class MemoryOptimizedPipeline:
    """Complete memory-optimized ML pipeline."""
    
    def __init__(self, memory_limit_mb: float = 2048,
                 storage_path: Optional[Path] = None):
        """
        Initialize memory-optimized pipeline.
        
        Args:
            memory_limit_mb: Total memory limit for pipeline
            storage_path: Path for temporary storage
        """
        self.memory_limit_mb = memory_limit_mb
        self.storage_path = storage_path or Path.cwd() / 'ml_temp'
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.feature_optimizer = FeatureOptimizer(self.storage_path / 'features')
        self.model_manager = ModelManager(self.storage_path / 'models')
        self.memory_profiles: List[MemoryProfile] = []
        
    def process_data(self, data_path: Path, 
                    model_name: str,
                    batch_size: int = 32,
                    feature_compression: Optional[str] = None) -> Path:
        """
        Process data through ML pipeline with memory optimization.
        
        Args:
            data_path: Path to input data
            model_name: Name of registered model
            batch_size: Batch size for processing
            feature_compression: Optional feature compression method
            
        Returns:
            Path to results
        """
        results_path = self.storage_path / f"results_{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.h5"
        
        # Create data generator
        data_gen = DataGenerator(str(data_path), batch_size=batch_size)
        
        # Setup batch predictor with model
        with self.model_manager.temporary_model(model_name) as model:
            predictor = BatchPredictor(
                model, 
                memory_limit_mb=self.memory_limit_mb * 0.5  # Use half for prediction
            )
            
            # Process in batches and save results
            with h5py.File(results_path, 'w') as f:
                # Create resizable dataset
                first_batch = next(iter(data_gen))
                first_pred = predictor.predict(first_batch)
                
                pred_dataset = f.create_dataset(
                    'predictions',
                    shape=(0,) + first_pred.shape[1:],
                    maxshape=(None,) + first_pred.shape[1:],
                    chunks=(batch_size,) + first_pred.shape[1:],
                    compression='gzip'
                )
                
                # Process all batches
                offset = 0
                for batch, batch_pred in zip(data_gen, predictor.predict(data_gen, return_generator=True)):
                    # Resize dataset
                    pred_dataset.resize(offset + len(batch_pred), axis=0)
                    pred_dataset[offset:offset + len(batch_pred)] = batch_pred
                    offset += len(batch_pred)
                    
                    # Monitor memory
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    if current_memory > self.memory_limit_mb * 0.9:
                        logger.warning("Approaching memory limit, triggering cleanup")
                        gc.collect()
                        
                f.attrs['model_name'] = model_name
                f.attrs['total_samples'] = offset
                f.attrs['processing_time'] = pd.Timestamp.now().isoformat()
                
        logger.info(f"Processing complete. Results saved to {results_path}")
        return results_path
        
    def get_memory_report(self) -> pd.DataFrame:
        """Get memory usage report."""
        if not self.memory_profiles:
            return pd.DataFrame()
            
        df = pd.DataFrame([p.to_dict() for p in self.memory_profiles])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df.sort_values('timestamp')


# Utility functions
def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize pandas DataFrame memory usage.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Memory-optimized DataFrame
    """
    initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
        
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
        
    # Optimize object columns
    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype('category')
            
    final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    logger.info(f"DataFrame memory optimization: {initial_memory:.2f}MB -> {final_memory:.2f}MB "
               f"({(1 - final_memory/initial_memory) * 100:.1f}% reduction)")
    
    return df


def estimate_model_memory(model: Any) -> float:
    """
    Estimate memory usage of a model.
    
    Args:
        model: Model to estimate
        
    Returns:
        Estimated memory in MB
    """
    if hasattr(model, 'parameters'):
        # PyTorch model
        total_params = sum(p.numel() for p in model.parameters())
        # Assume float32 (4 bytes per parameter)
        memory_mb = (total_params * 4) / 1024 / 1024
    else:
        # Sklearn or other model - use pickle size as estimate
        import io
        buffer = io.BytesIO()
        pickle.dump(model, buffer)
        memory_mb = len(buffer.getvalue()) / 1024 / 1024
        
    return memory_mb


def clear_memory():
    """Force memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()