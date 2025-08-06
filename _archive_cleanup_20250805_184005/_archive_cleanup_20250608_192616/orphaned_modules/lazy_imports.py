"""Lazy import system for deferred loading of heavy dependencies."""

import sys
import importlib
import logging
from typing import Any, Optional, Dict, Callable, TYPE_CHECKING, List
from functools import wraps
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

# Type checking imports - these are only imported when type checking
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy
    import tensorflow as tf
    import torch
    import sklearn
    import cv2
    from PIL import Image
    import plotly
    import seaborn as sns


class LazyModule:
    """A lazy-loaded module that imports on first access."""
    
    def __init__(self, module_name: str, import_func: Optional[Callable] = None):
        self._module_name = module_name
        self._module = None
        self._import_func = import_func or (lambda: importlib.import_module(module_name))
        self._lock = threading.Lock()
        
    def _load(self):
        """Load the module if not already loaded."""
        if self._module is None:
            with self._lock:
                if self._module is None:  # Double-check pattern
                    logger.debug(f"Lazy loading module: {self._module_name}")
                    try:
                        self._module = self._import_func()
                    except ImportError as e:
                        logger.error(f"Failed to import {self._module_name}: {e}")
                        raise
                        
    def __getattr__(self, name: str) -> Any:
        """Get attribute from the loaded module."""
        self._load()
        return getattr(self._module, name)
        
    def __dir__(self):
        """Return available attributes."""
        self._load()
        return dir(self._module)
        
    def __repr__(self):
        """String representation."""
        if self._module is None:
            return f"<LazyModule '{self._module_name}' (not loaded)>"
        return f"<LazyModule '{self._module_name}' (loaded)>"


class LazyImporter:
    """Manager for lazy imports across the application."""
    
    def __init__(self):
        self._modules: Dict[str, LazyModule] = {}
        self._aliases: Dict[str, str] = {}
        
    def register(self, module_name: str, alias: Optional[str] = None, 
                 import_func: Optional[Callable] = None) -> LazyModule:
        """Register a module for lazy loading."""
        if module_name not in self._modules:
            self._modules[module_name] = LazyModule(module_name, import_func)
            
        if alias:
            self._aliases[alias] = module_name
            
        return self._modules[module_name]
        
    def get(self, name: str) -> LazyModule:
        """Get a lazy module by name or alias."""
        if name in self._aliases:
            name = self._aliases[name]
            
        if name not in self._modules:
            raise KeyError(f"Module '{name}' not registered for lazy loading")
            
        return self._modules[name]
        
    def __getattr__(self, name: str) -> LazyModule:
        """Allow attribute-style access to modules."""
        return self.get(name)
        
    def preload(self, *module_names: str) -> None:
        """Preload specified modules."""
        for name in module_names:
            module = self.get(name)
            module._load()
            
    def is_loaded(self, name: str) -> bool:
        """Check if a module is already loaded."""
        if name in self._aliases:
            name = self._aliases[name]
            
        if name in self._modules:
            return self._modules[name]._module is not None
            
        return False
        
    def get_loaded_modules(self) -> List[str]:
        """Get list of currently loaded modules."""
        return [
            name for name, module in self._modules.items()
            if module._module is not None
        ]


# Global lazy importer instance
lazy_imports = LazyImporter()

# Register common heavy dependencies
lazy_imports.register('numpy', 'np')
lazy_imports.register('pandas', 'pd')
lazy_imports.register('matplotlib.pyplot', 'plt')
lazy_imports.register('scipy')
lazy_imports.register('tensorflow', 'tf')
lazy_imports.register('torch')
lazy_imports.register('sklearn')
lazy_imports.register('cv2')
lazy_imports.register('PIL.Image', 'Image', lambda: importlib.import_module('PIL').Image)
lazy_imports.register('plotly')
lazy_imports.register('seaborn', 'sns')


def lazy_import(module_name: str, alias: Optional[str] = None):
    """Decorator for lazy importing at function level."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Import module if needed
            if not hasattr(wrapper, '_module'):
                wrapper._module = importlib.import_module(module_name)
                
            # Inject module into function's globals
            if alias:
                func.__globals__[alias] = wrapper._module
            else:
                func.__globals__[module_name] = wrapper._module
                
            return func(*args, **kwargs)
            
        return wrapper
    return decorator


def lazy_property(import_func: Callable[[], Any]):
    """Property decorator for lazy loading of expensive computations."""
    class LazyProperty:
        def __init__(self, func):
            self.func = func
            self.value = None
            self._lock = threading.Lock()
            
        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
                
            if self.value is None:
                with self._lock:
                    if self.value is None:
                        # Import dependencies
                        import_func()
                        # Compute value
                        self.value = self.func(obj)
                        
            return self.value
            
    return LazyProperty


class LazyModuleProxy:
    """A proxy that behaves like a module but loads lazily."""
    
    def __init__(self, module_name: str):
        self._module_name = module_name
        self._real_module = None
        
    def __getattr__(self, name: str) -> Any:
        if self._real_module is None:
            logger.debug(f"Lazy loading {self._module_name} via proxy")
            self._real_module = importlib.import_module(self._module_name)
            # Update sys.modules to point to real module
            sys.modules[self._module_name] = self._real_module
            
        return getattr(self._real_module, name)


def install_lazy_module(module_name: str, package_name: Optional[str] = None):
    """Install a module as lazy-loaded in sys.modules."""
    if package_name is None:
        package_name = module_name
        
    if package_name not in sys.modules:
        sys.modules[package_name] = LazyModuleProxy(module_name)
        

# Utility functions for common patterns

def lazy_numpy():
    """Get numpy lazily."""
    return lazy_imports.get('numpy')


def lazy_pandas():
    """Get pandas lazily."""
    return lazy_imports.get('pandas')


def lazy_matplotlib():
    """Get matplotlib.pyplot lazily."""
    return lazy_imports.get('matplotlib.pyplot')


# Context manager for temporary imports
class temporary_import:
    """Context manager for temporary imports that are unloaded after use."""
    
    def __init__(self, module_name: str, alias: Optional[str] = None):
        self.module_name = module_name
        self.alias = alias or module_name
        self.module = None
        self._original_modules = {}
        
    def __enter__(self):
        # Save original state
        if self.module_name in sys.modules:
            self._original_modules[self.module_name] = sys.modules[self.module_name]
            
        # Import module
        self.module = importlib.import_module(self.module_name)
        return self.module
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        if self.module_name in self._original_modules:
            sys.modules[self.module_name] = self._original_modules[self.module_name]
        else:
            # Remove from sys.modules
            sys.modules.pop(self.module_name, None)
            
        # Clear references
        self.module = None
        
        # Force garbage collection
        import gc
        gc.collect()


# Performance monitoring for lazy imports
class ImportMonitor:
    """Monitor lazy import performance."""
    
    def __init__(self):
        self.import_times: Dict[str, float] = {}
        self.import_counts: Dict[str, int] = defaultdict(int)
        
    def record_import(self, module_name: str, import_time: float):
        """Record an import event."""
        self.import_times[module_name] = import_time
        self.import_counts[module_name] += 1
        
    def get_stats(self) -> Dict[str, Any]:
        """Get import statistics."""
        return {
            'total_imports': sum(self.import_counts.values()),
            'unique_modules': len(self.import_times),
            'total_time': sum(self.import_times.values()),
            'slowest_imports': sorted(
                self.import_times.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


# Global import monitor
import_monitor = ImportMonitor()


# Example usage patterns
def example_lazy_usage():
    """Example of how to use lazy imports."""
    
    # Method 1: Using global lazy_imports
    np = lazy_imports.np  # Not loaded yet
    array = np.array([1, 2, 3])  # Loaded on first use
    
    # Method 2: Using decorator
    @lazy_import('pandas', 'pd')
    def process_dataframe():
        return pd.DataFrame({'a': [1, 2, 3]})
        
    # Method 3: Using context manager
    with temporary_import('scipy.stats', 'stats') as stats:
        result = stats.norm.pdf(0)
        
    # Method 4: Manual lazy loading
    def get_tensorflow():
        if not lazy_imports.is_loaded('tensorflow'):
            logger.info("Loading TensorFlow...")
            
        return lazy_imports.tf