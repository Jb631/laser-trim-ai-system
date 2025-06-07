"""
Comprehensive caching system for the laser trim analyzer application.

This module provides a unified caching interface with multiple cache strategies,
memory-aware operations, and thread-safe access.
"""

import os
import sys
import time
import json
import pickle
import hashlib
import threading
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Tuple, Union, List, TypeVar, Generic
from enum import Enum
import psutil
import logging

from ..core.exceptions import CacheError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheStrategy(Enum):
    """Available cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based eviction
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheEntry(Generic[T]):
    """Represents a single cache entry with metadata."""
    key: str
    value: T
    size: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if this entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get total size of cached data in bytes."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend with configurable eviction strategies."""
    
    def __init__(
        self,
        max_size: int = 1024 * 1024 * 100,  # 100MB default
        max_entries: int = 10000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[float] = None
    ):
        self.max_size = max_size
        self.max_entries = max_entries
        self.strategy = strategy
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._total_size = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache with thread safety."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
                
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                return None
            
            # Update access metadata
            entry.update_access()
            
            # Move to end for LRU
            if self.strategy == CacheStrategy.LRU:
                self._cache.move_to_end(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store value in cache with eviction if needed."""
        with self._lock:
            # Calculate size
            size = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Evict entries if needed
            self._evict_if_needed(size)
            
            # Create and store new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                ttl=ttl or self.default_ttl
            )
            
            self._cache[key] = entry
            self._total_size += size
    
    def delete(self, key: str) -> bool:
        """Remove value from cache."""
        with self._lock:
            return self._remove_entry(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._total_size = 0
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                self._remove_entry(key)
                return False
            return True
    
    def size(self) -> int:
        """Get total size of cached data."""
        return self._total_size
    
    def _remove_entry(self, key: str) -> bool:
        """Remove an entry and update size tracking."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._total_size -= entry.size
            return True
        return False
    
    def _evict_if_needed(self, required_size: int) -> None:
        """Evict entries based on strategy to make room."""
        while (self._total_size + required_size > self.max_size or 
               len(self._cache) >= self.max_entries):
            
            if not self._cache:
                break
                
            # Select entry to evict based on strategy
            if self.strategy == CacheStrategy.LRU:
                # Remove least recently used (first item)
                key = next(iter(self._cache))
            elif self.strategy == CacheStrategy.LFU:
                # Remove least frequently used
                key = min(self._cache.keys(), 
                         key=lambda k: self._cache[k].access_count)
            elif self.strategy == CacheStrategy.FIFO:
                # Remove oldest entry
                key = min(self._cache.keys(), 
                         key=lambda k: self._cache[k].created_at)
            elif self.strategy == CacheStrategy.SIZE:
                # Remove largest entry
                key = max(self._cache.keys(), 
                         key=lambda k: self._cache[k].size)
            else:  # TTL
                # Remove expired or oldest
                expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
                if expired_keys:
                    key = expired_keys[0]
                else:
                    key = next(iter(self._cache))
            
            self._remove_entry(key)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of an object."""
        try:
            # For simple types
            if isinstance(obj, (str, bytes, bytearray)):
                return len(obj)
            elif isinstance(obj, (int, float, bool)):
                return sys.getsizeof(obj)
            elif isinstance(obj, (list, tuple, set, frozenset)):
                return sys.getsizeof(obj) + sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sys.getsizeof(obj) + sum(
                    self._estimate_size(k) + self._estimate_size(v) 
                    for k, v in obj.items()
                )
            else:
                # For complex objects, use pickle size as estimate
                return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            # Fallback to basic size
            return sys.getsizeof(obj)


class FileCache(CacheBackend):
    """File-based cache backend for persistent storage."""
    
    def __init__(self, cache_dir: Path, max_size: int = 1024 * 1024 * 500):  # 500MB
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._metadata_file = self.cache_dir / ".cache_metadata.json"
        self._metadata = self._load_metadata()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from file cache."""
        with self._lock:
            if key not in self._metadata:
                return None
            
            entry_meta = self._metadata[key]
            
            # Check TTL
            if entry_meta.get('ttl') and time.time() - entry_meta['created_at'] > entry_meta['ttl']:
                self.delete(key)
                return None
            
            # Read from file
            cache_file = self.cache_dir / entry_meta['filename']
            if not cache_file.exists():
                self.delete(key)
                return None
            
            try:
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                
                # Update access time
                entry_meta['last_accessed'] = time.time()
                entry_meta['access_count'] = entry_meta.get('access_count', 0) + 1
                self._save_metadata()
                
                return value
            except Exception as e:
                logger.error(f"Error reading cache file {cache_file}: {e}")
                self.delete(key)
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store value in file cache."""
        with self._lock:
            # Generate filename
            filename = hashlib.sha256(key.encode()).hexdigest()[:16] + ".cache"
            cache_file = self.cache_dir / filename
            
            # Serialize and save
            try:
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                size = len(data)
                
                # Check if we need to evict
                self._evict_if_needed(size)
                
                # Write file
                with open(cache_file, 'wb') as f:
                    f.write(data)
                
                # Update metadata
                self._metadata[key] = {
                    'filename': filename,
                    'size': size,
                    'created_at': time.time(),
                    'last_accessed': time.time(),
                    'access_count': 0,
                    'ttl': ttl
                }
                self._save_metadata()
                
            except Exception as e:
                logger.error(f"Error writing cache file {cache_file}: {e}")
                raise CacheError(f"Failed to cache value: {e}")
    
    def delete(self, key: str) -> bool:
        """Remove value from file cache."""
        with self._lock:
            if key not in self._metadata:
                return False
            
            entry_meta = self._metadata[key]
            cache_file = self.cache_dir / entry_meta['filename']
            
            # Remove file
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.error(f"Error deleting cache file {cache_file}: {e}")
            
            # Remove metadata
            del self._metadata[key]
            self._save_metadata()
            return True
    
    def clear(self) -> None:
        """Clear all cache files."""
        with self._lock:
            # Remove all cache files
            for entry in self._metadata.values():
                cache_file = self.cache_dir / entry['filename']
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass
            
            # Clear metadata
            self._metadata.clear()
            self._save_metadata()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            if key not in self._metadata:
                return False
            
            entry_meta = self._metadata[key]
            cache_file = self.cache_dir / entry_meta['filename']
            
            if not cache_file.exists():
                self.delete(key)
                return False
            
            # Check TTL
            if entry_meta.get('ttl') and time.time() - entry_meta['created_at'] > entry_meta['ttl']:
                self.delete(key)
                return False
            
            return True
    
    def size(self) -> int:
        """Get total size of cache directory."""
        total = 0
        for entry in self._metadata.values():
            total += entry.get('size', 0)
        return total
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata from file."""
        if not self._metadata_file.exists():
            return {}
        
        try:
            with open(self._metadata_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to file."""
        try:
            with open(self._metadata_file, 'w') as f:
                json.dump(self._metadata, f)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def _evict_if_needed(self, required_size: int) -> None:
        """Evict old entries if cache size exceeds limit."""
        current_size = self.size()
        
        if current_size + required_size <= self.max_size:
            return
        
        # Sort by last accessed time and remove oldest
        sorted_entries = sorted(
            self._metadata.items(),
            key=lambda x: x[1].get('last_accessed', 0)
        )
        
        for key, _ in sorted_entries:
            if current_size + required_size <= self.max_size:
                break
            
            entry_size = self._metadata[key].get('size', 0)
            self.delete(key)
            current_size -= entry_size


class CacheManager:
    """
    Unified cache manager for the entire application.
    
    Provides different cache types for various use cases:
    - General data caching
    - File content caching
    - Model prediction caching
    - Configuration caching
    - Analysis result caching
    """
    
    def __init__(
        self,
        memory_cache_size: int = 1024 * 1024 * 100,  # 100MB
        file_cache_size: int = 1024 * 1024 * 500,    # 500MB
        cache_dir: Optional[Path] = None,
        monitor_memory: bool = True,
        memory_threshold: float = 0.8  # 80% memory usage threshold
    ):
        self.cache_dir = cache_dir or Path.home() / ".laser_trim_analyzer" / "cache"
        self.monitor_memory = monitor_memory
        self.memory_threshold = memory_threshold
        
        # Initialize different cache backends
        self._caches = {
            'general': MemoryCache(
                max_size=memory_cache_size // 4,
                strategy=CacheStrategy.LRU
            ),
            'file_content': MemoryCache(
                max_size=memory_cache_size // 4,
                strategy=CacheStrategy.LRU,
                default_ttl=3600  # 1 hour
            ),
            'model_predictions': MemoryCache(
                max_size=memory_cache_size // 4,
                strategy=CacheStrategy.LFU
            ),
            'config': MemoryCache(
                max_size=memory_cache_size // 8,
                strategy=CacheStrategy.TTL,
                default_ttl=300  # 5 minutes
            ),
            'analysis_results': FileCache(
                cache_dir=self.cache_dir / "analysis",
                max_size=file_cache_size
            )
        }
        
        # Cache invalidation callbacks
        self._invalidation_callbacks: Dict[str, List[Callable]] = {}
        
        # Start memory monitor if enabled
        if self.monitor_memory:
            self._start_memory_monitor()
    
    def get(
        self, 
        key: str, 
        cache_type: str = 'general',
        default: Optional[T] = None
    ) -> Optional[T]:
        """
        Retrieve value from specified cache.
        
        Args:
            key: Cache key
            cache_type: Type of cache to use
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        if cache_type not in self._caches:
            raise ValueError(f"Unknown cache type: {cache_type}")
        
        value = self._caches[cache_type].get(key)
        return value if value is not None else default
    
    def set(
        self,
        key: str,
        value: Any,
        cache_type: str = 'general',
        ttl: Optional[float] = None
    ) -> None:
        """
        Store value in specified cache.
        
        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cache to use
            ttl: Time to live in seconds
        """
        if cache_type not in self._caches:
            raise ValueError(f"Unknown cache type: {cache_type}")
        
        self._caches[cache_type].set(key, value, ttl)
    
    def delete(self, key: str, cache_type: str = 'general') -> bool:
        """Delete value from specified cache."""
        if cache_type not in self._caches:
            raise ValueError(f"Unknown cache type: {cache_type}")
        
        return self._caches[cache_type].delete(key)
    
    def clear(self, cache_type: Optional[str] = None) -> None:
        """
        Clear cache(s).
        
        Args:
            cache_type: Specific cache to clear, or None for all
        """
        if cache_type:
            if cache_type not in self._caches:
                raise ValueError(f"Unknown cache type: {cache_type}")
            self._caches[cache_type].clear()
        else:
            for cache in self._caches.values():
                cache.clear()
    
    def exists(self, key: str, cache_type: str = 'general') -> bool:
        """Check if key exists in specified cache."""
        if cache_type not in self._caches:
            raise ValueError(f"Unknown cache type: {cache_type}")
        
        return self._caches[cache_type].exists(key)
    
    def cache_file_content(
        self,
        file_path: Union[str, Path],
        content: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Cache file content with file path as key."""
        key = self._file_cache_key(file_path)
        self.set(key, content, cache_type='file_content', ttl=ttl)
    
    def get_file_content(
        self,
        file_path: Union[str, Path]
    ) -> Optional[Any]:
        """Retrieve cached file content."""
        key = self._file_cache_key(file_path)
        return self.get(key, cache_type='file_content')
    
    def cache_model_prediction(
        self,
        model_name: str,
        input_hash: str,
        prediction: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Cache ML model prediction."""
        key = f"model:{model_name}:{input_hash}"
        self.set(key, prediction, cache_type='model_predictions', ttl=ttl)
    
    def get_model_prediction(
        self,
        model_name: str,
        input_hash: str
    ) -> Optional[Any]:
        """Retrieve cached model prediction."""
        key = f"model:{model_name}:{input_hash}"
        return self.get(key, cache_type='model_predictions')
    
    def cache_analysis_result(
        self,
        analysis_id: str,
        result: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Cache analysis result persistently."""
        key = f"analysis:{analysis_id}"
        self.set(key, result, cache_type='analysis_results', ttl=ttl)
    
    def get_analysis_result(self, analysis_id: str) -> Optional[Any]:
        """Retrieve cached analysis result."""
        key = f"analysis:{analysis_id}"
        return self.get(key, cache_type='analysis_results')
    
    def invalidate_pattern(self, pattern: str, cache_type: str = 'general') -> int:
        """
        Invalidate all cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match (supports * wildcard)
            cache_type: Cache type to search
            
        Returns:
            Number of entries invalidated
        """
        if cache_type not in self._caches:
            raise ValueError(f"Unknown cache type: {cache_type}")
        
        cache = self._caches[cache_type]
        count = 0
        
        # Get all keys (implementation depends on backend)
        if isinstance(cache, MemoryCache):
            with cache._lock:
                keys = list(cache._cache.keys())
        elif isinstance(cache, FileCache):
            with cache._lock:
                keys = list(cache._metadata.keys())
        else:
            return 0
        
        # Match and delete
        import fnmatch
        for key in keys:
            if fnmatch.fnmatch(key, pattern):
                if cache.delete(key):
                    count += 1
        
        return count
    
    def register_invalidation_callback(
        self,
        cache_type: str,
        callback: Callable[[str], None]
    ) -> None:
        """Register callback for cache invalidation events."""
        if cache_type not in self._invalidation_callbacks:
            self._invalidation_callbacks[cache_type] = []
        self._invalidation_callbacks[cache_type].append(callback)
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        stats = {}
        
        for name, cache in self._caches.items():
            cache_stats = {
                'type': type(cache).__name__,
                'size_bytes': cache.size(),
                'size_mb': cache.size() / (1024 * 1024)
            }
            
            if isinstance(cache, MemoryCache):
                cache_stats.update({
                    'entries': len(cache._cache),
                    'strategy': cache.strategy.value,
                    'max_size_mb': cache.max_size / (1024 * 1024),
                    'max_entries': cache.max_entries
                })
            elif isinstance(cache, FileCache):
                cache_stats.update({
                    'entries': len(cache._metadata),
                    'cache_dir': str(cache.cache_dir),
                    'max_size_mb': cache.max_size / (1024 * 1024)
                })
            
            stats[name] = cache_stats
        
        # Add memory info
        if self.monitor_memory:
            memory = psutil.virtual_memory()
            stats['system_memory'] = {
                'total_mb': memory.total / (1024 * 1024),
                'available_mb': memory.available / (1024 * 1024),
                'percent_used': memory.percent,
                'threshold': self.memory_threshold * 100
            }
        
        return stats
    
    def _file_cache_key(self, file_path: Union[str, Path]) -> str:
        """Generate cache key for file path."""
        path = Path(file_path).resolve()
        stat = path.stat()
        # Include file size and modification time in key
        return f"file:{path}:{stat.st_size}:{stat.st_mtime}"
    
    def _start_memory_monitor(self) -> None:
        """Start background thread to monitor memory usage."""
        def monitor():
            while True:
                try:
                    memory = psutil.virtual_memory()
                    if memory.percent > self.memory_threshold * 100:
                        logger.warning(
                            f"Memory usage high: {memory.percent:.1f}%. "
                            "Clearing some caches..."
                        )
                        # Clear memory caches in order of priority
                        self.clear('general')
                        self.clear('file_content')
                        
                        # Force garbage collection
                        import gc
                        gc.collect()
                except Exception as e:
                    logger.error(f"Error in memory monitor: {e}")
                
                time.sleep(30)  # Check every 30 seconds
        
        # Start daemon thread
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def decorator(
        self,
        cache_type: str = 'general',
        ttl: Optional[float] = None,
        key_func: Optional[Callable] = None
    ):
        """
        Decorator for caching function results.
        
        Args:
            cache_type: Type of cache to use
            ttl: Time to live in seconds
            key_func: Function to generate cache key from args
            
        Example:
            @cache_manager.decorator(cache_type='model_predictions', ttl=3600)
            def predict(model, data):
                return model.predict(data)
        """
        def wrapper(func):
            def wrapped(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = ":".join(key_parts)
                
                # Check cache
                cached = self.get(cache_key, cache_type=cache_type)
                if cached is not None:
                    return cached
                
                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, cache_type=cache_type, ttl=ttl)
                
                return result
            
            return wrapped
        return wrapper


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def initialize_cache_manager(**kwargs) -> CacheManager:
    """Initialize global cache manager with custom settings."""
    global _cache_manager
    _cache_manager = CacheManager(**kwargs)
    return _cache_manager