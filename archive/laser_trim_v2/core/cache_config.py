"""
Cache configuration and management utilities.

This module provides configuration options and utilities for managing
the application's caching system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .cache_manager import CacheStrategy, initialize_cache_manager, get_cache_manager, MemoryCache, FileCache
from .config import Config


@dataclass
class CacheConfig:
    """Configuration for cache system."""
    
    # Cache sizes in MB
    memory_cache_size_mb: int = 200
    file_cache_size_mb: int = 1000
    
    # Cache directory
    cache_dir: Optional[Path] = None
    
    # Cache strategies
    general_strategy: CacheStrategy = CacheStrategy.LRU
    file_content_strategy: CacheStrategy = CacheStrategy.LRU
    model_prediction_strategy: CacheStrategy = CacheStrategy.LFU
    config_strategy: CacheStrategy = CacheStrategy.TTL
    
    # TTL settings (in seconds)
    file_content_ttl: float = 3600  # 1 hour
    model_prediction_ttl: float = 7200  # 2 hours
    config_ttl: float = 300  # 5 minutes
    analysis_result_ttl: float = 86400  # 24 hours
    
    # Memory monitoring
    enable_memory_monitor: bool = True
    memory_threshold: float = 0.85  # 85% memory usage threshold
    
    # Cache limits
    max_entries_per_cache: int = 10000
    max_file_cache_entries: int = 1000
    
    # Performance settings
    enable_compression: bool = True
    compression_threshold_kb: int = 10  # Compress items larger than 10KB
    
    @classmethod
    def from_config(cls, config: Config) -> 'CacheConfig':
        """Create cache config from application config."""
        cache_section = config.get('cache', {})
        
        return cls(
            memory_cache_size_mb=cache_section.get('memory_size_mb', 200),
            file_cache_size_mb=cache_section.get('file_size_mb', 1000),
            cache_dir=Path(cache_section.get('directory')) if cache_section.get('directory') else None,
            enable_memory_monitor=cache_section.get('enable_memory_monitor', True),
            memory_threshold=cache_section.get('memory_threshold', 0.85),
            file_content_ttl=cache_section.get('file_content_ttl', 3600),
            model_prediction_ttl=cache_section.get('model_prediction_ttl', 7200),
            config_ttl=cache_section.get('config_ttl', 300),
            analysis_result_ttl=cache_section.get('analysis_result_ttl', 86400)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'memory_cache_size_mb': self.memory_cache_size_mb,
            'file_cache_size_mb': self.file_cache_size_mb,
            'cache_dir': str(self.cache_dir) if self.cache_dir else None,
            'general_strategy': self.general_strategy.value,
            'file_content_strategy': self.file_content_strategy.value,
            'model_prediction_strategy': self.model_prediction_strategy.value,
            'config_strategy': self.config_strategy.value,
            'file_content_ttl': self.file_content_ttl,
            'model_prediction_ttl': self.model_prediction_ttl,
            'config_ttl': self.config_ttl,
            'analysis_result_ttl': self.analysis_result_ttl,
            'enable_memory_monitor': self.enable_memory_monitor,
            'memory_threshold': self.memory_threshold,
            'max_entries_per_cache': self.max_entries_per_cache,
            'max_file_cache_entries': self.max_file_cache_entries,
            'enable_compression': self.enable_compression,
            'compression_threshold_kb': self.compression_threshold_kb
        }


class CachePreset(Enum):
    """Predefined cache configurations for different use cases."""
    
    MINIMAL = "minimal"  # Minimal caching for low memory systems
    BALANCED = "balanced"  # Balanced performance and memory usage
    PERFORMANCE = "performance"  # Maximum performance, high memory usage
    PERSISTENT = "persistent"  # Focus on file-based persistent caching


CACHE_PRESETS = {
    CachePreset.MINIMAL: CacheConfig(
        memory_cache_size_mb=50,
        file_cache_size_mb=200,
        max_entries_per_cache=1000,
        enable_memory_monitor=True,
        memory_threshold=0.7
    ),
    CachePreset.BALANCED: CacheConfig(
        memory_cache_size_mb=200,
        file_cache_size_mb=1000,
        max_entries_per_cache=10000,
        enable_memory_monitor=True,
        memory_threshold=0.85
    ),
    CachePreset.PERFORMANCE: CacheConfig(
        memory_cache_size_mb=500,
        file_cache_size_mb=2000,
        max_entries_per_cache=50000,
        enable_memory_monitor=True,
        memory_threshold=0.9,
        model_prediction_strategy=CacheStrategy.LFU,
        file_content_strategy=CacheStrategy.LFU
    ),
    CachePreset.PERSISTENT: CacheConfig(
        memory_cache_size_mb=100,
        file_cache_size_mb=5000,
        max_entries_per_cache=5000,
        max_file_cache_entries=10000,
        enable_memory_monitor=True,
        memory_threshold=0.8,
        analysis_result_ttl=604800  # 7 days
    )
}


def setup_cache_from_config(config: Config) -> None:
    """Initialize cache system from application configuration."""
    cache_config = CacheConfig.from_config(config)
    setup_cache(cache_config)


def setup_cache(cache_config: CacheConfig) -> None:
    """Initialize cache system with given configuration."""
    # Determine cache directory
    if cache_config.cache_dir is None:
        cache_config.cache_dir = Path.home() / ".laser_trim_analyzer" / "cache"
    
    # Initialize cache manager with custom configuration
    from .cache_manager import MemoryCache, FileCache
    
    manager = initialize_cache_manager(
        memory_cache_size=cache_config.memory_cache_size_mb * 1024 * 1024,
        file_cache_size=cache_config.file_cache_size_mb * 1024 * 1024,
        cache_dir=cache_config.cache_dir,
        monitor_memory=cache_config.enable_memory_monitor,
        memory_threshold=cache_config.memory_threshold
    )
    
    # Configure individual caches with specific strategies
    manager._caches['general'] = MemoryCache(
        max_size=(cache_config.memory_cache_size_mb * 1024 * 1024) // 4,
        max_entries=cache_config.max_entries_per_cache,
        strategy=cache_config.general_strategy
    )
    
    manager._caches['file_content'] = MemoryCache(
        max_size=(cache_config.memory_cache_size_mb * 1024 * 1024) // 4,
        max_entries=cache_config.max_entries_per_cache,
        strategy=cache_config.file_content_strategy,
        default_ttl=cache_config.file_content_ttl
    )
    
    manager._caches['model_predictions'] = MemoryCache(
        max_size=(cache_config.memory_cache_size_mb * 1024 * 1024) // 4,
        max_entries=cache_config.max_entries_per_cache,
        strategy=cache_config.model_prediction_strategy,
        default_ttl=cache_config.model_prediction_ttl
    )
    
    manager._caches['config'] = MemoryCache(
        max_size=(cache_config.memory_cache_size_mb * 1024 * 1024) // 8,
        max_entries=cache_config.max_entries_per_cache // 10,
        strategy=cache_config.config_strategy,
        default_ttl=cache_config.config_ttl
    )
    
    manager._caches['analysis_results'] = FileCache(
        cache_dir=cache_config.cache_dir / "analysis",
        max_size=cache_config.file_cache_size_mb * 1024 * 1024
    )


def setup_cache_from_preset(preset: CachePreset) -> None:
    """Initialize cache system with a predefined preset."""
    cache_config = CACHE_PRESETS[preset]
    setup_cache(cache_config)


def get_cache_info() -> Dict[str, Any]:
    """Get information about current cache configuration and status."""
    manager = get_cache_manager()
    stats = manager.get_stats()
    
    # Add configuration info
    info = {
        'stats': stats,
        'cache_types': list(manager._caches.keys()),
        'total_memory_usage_mb': sum(
            cache_stats.get('size_mb', 0)
            for cache_stats in stats.values()
            if isinstance(cache_stats, dict) and 'size_mb' in cache_stats
        )
    }
    
    return info


def clear_old_cache_files(max_age_days: int = 7) -> int:
    """
    Clear cache files older than specified days.
    
    Args:
        max_age_days: Maximum age of cache files in days
        
    Returns:
        Number of files cleared
    """
    manager = get_cache_manager()
    cache_dir = manager.cache_dir
    
    if not cache_dir.exists():
        return 0
    
    import time
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    cleared = 0
    
    # Walk through cache directory
    for cache_file in cache_dir.rglob("*.cache"):
        try:
            # Check file age
            file_age = current_time - cache_file.stat().st_mtime
            if file_age > max_age_seconds:
                cache_file.unlink()
                cleared += 1
        except Exception:
            pass
    
    return cleared


def optimize_cache() -> Dict[str, Any]:
    """
    Optimize cache by clearing expired entries and reorganizing.
    
    Returns:
        Optimization statistics
    """
    manager = get_cache_manager()
    stats = {
        'expired_cleared': 0,
        'old_files_cleared': 0,
        'memory_freed_mb': 0
    }
    
    # Clear expired entries from memory caches
    for cache_name, cache in manager._caches.items():
        if hasattr(cache, '_cache'):  # MemoryCache
            initial_size = cache.size()
            
            # Force expiration check
            expired_keys = []
            with cache._lock:
                for key, entry in cache._cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
            
            for key in expired_keys:
                cache.delete(key)
                stats['expired_cleared'] += 1
            
            final_size = cache.size()
            stats['memory_freed_mb'] += (initial_size - final_size) / (1024 * 1024)
    
    # Clear old cache files
    stats['old_files_cleared'] = clear_old_cache_files()
    
    return stats


# Environment-based cache setup
def setup_cache_from_environment() -> None:
    """Setup cache based on environment variables."""
    cache_config = CacheConfig(
        memory_cache_size_mb=int(os.getenv('CACHE_MEMORY_SIZE_MB', '200')),
        file_cache_size_mb=int(os.getenv('CACHE_FILE_SIZE_MB', '1000')),
        cache_dir=Path(os.getenv('CACHE_DIR')) if os.getenv('CACHE_DIR') else None,
        enable_memory_monitor=os.getenv('CACHE_MONITOR_MEMORY', 'true').lower() == 'true',
        memory_threshold=float(os.getenv('CACHE_MEMORY_THRESHOLD', '0.85'))
    )
    setup_cache(cache_config)