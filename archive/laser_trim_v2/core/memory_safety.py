"""
Memory Safety Module for Laser Trim Analyzer

Provides comprehensive memory safety checks, buffer overflow prevention,
and resource leak detection for the entire application.
"""

import gc
import sys
import weakref
import threading
import psutil
import resource
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time


class MemoryIssueType(Enum):
    """Types of memory safety issues."""
    BUFFER_OVERFLOW = "buffer_overflow"
    MEMORY_LEAK = "memory_leak"
    EXCESSIVE_ALLOCATION = "excessive_allocation"
    UNCLOSED_RESOURCE = "unclosed_resource"
    ARRAY_BOUNDS = "array_bounds"
    STRING_OVERFLOW = "string_overflow"
    CACHE_OVERFLOW = "cache_overflow"
    THREAD_LEAK = "thread_leak"


@dataclass
class MemorySafetyConfig:
    """Configuration for memory safety checks."""
    max_array_size: int = 10_000_000  # 10M elements
    max_string_length: int = 1_000_000  # 1MB
    max_memory_percent: float = 80.0  # Max 80% of system memory
    max_cache_entries: int = 1000
    max_figure_count: int = 50
    check_interval: float = 60.0  # Check every 60 seconds
    enable_tracking: bool = True
    enable_auto_cleanup: bool = True
    emergency_threshold_mb: int = 500  # Emergency cleanup if < 500MB free


@dataclass
class MemoryIssue:
    """Represents a detected memory safety issue."""
    issue_type: MemoryIssueType
    description: str
    location: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    size_bytes: Optional[int] = None
    count: Optional[int] = None
    suggestion: Optional[str] = None


class ResourceTracker:
    """Tracks resource allocation and detects leaks."""
    
    def __init__(self):
        self._resources: Dict[str, weakref.WeakSet] = {
            'figures': weakref.WeakSet(),
            'files': weakref.WeakSet(),
            'threads': weakref.WeakSet(),
            'sessions': weakref.WeakSet(),
            'engines': weakref.WeakSet()
        }
        self._allocation_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        
    def register(self, resource_type: str, resource: Any) -> None:
        """Register a resource for tracking."""
        with self._lock:
            if resource_type not in self._resources:
                self._resources[resource_type] = weakref.WeakSet()
            self._resources[resource_type].add(resource)
            self._allocation_counts[resource_type] = self._allocation_counts.get(resource_type, 0) + 1
    
    def unregister(self, resource_type: str, resource: Any) -> None:
        """Unregister a resource."""
        with self._lock:
            if resource_type in self._resources:
                self._resources[resource_type].discard(resource)
    
    def get_unclosed_resources(self) -> Dict[str, List[Any]]:
        """Get list of unclosed resources."""
        with self._lock:
            unclosed = {}
            for resource_type, resources in self._resources.items():
                active = list(resources)
                if active:
                    unclosed[resource_type] = active
            return unclosed
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get resource allocation statistics."""
        with self._lock:
            stats = {}
            for resource_type, resources in self._resources.items():
                stats[resource_type] = {
                    'active': len(resources),
                    'total_allocated': self._allocation_counts.get(resource_type, 0)
                }
            return stats


class MemorySafetyValidator:
    """Comprehensive memory safety validator."""
    
    def __init__(self, config: Optional[MemorySafetyConfig] = None):
        self.config = config or MemorySafetyConfig()
        self.logger = logging.getLogger(__name__)
        self._resource_tracker = ResourceTracker()
        self._issues: List[MemoryIssue] = []
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Start memory tracking if enabled
        if self.config.enable_tracking:
            tracemalloc.start()
            
        # Start monitoring thread if auto cleanup enabled
        if self.config.enable_auto_cleanup:
            self._start_monitoring()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_monitoring()
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._monitoring_thread = threading.Thread(
                target=self._monitor_resources,
                daemon=True
            )
            self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
    
    def _monitor_resources(self):
        """Background resource monitoring."""
        while not self._stop_monitoring.is_set():
            try:
                # Check memory usage
                memory_info = self.check_memory_usage()
                if memory_info['percent'] > self.config.max_memory_percent:
                    self._handle_high_memory(memory_info)
                
                # Check for unclosed resources
                unclosed = self._resource_tracker.get_unclosed_resources()
                if unclosed:
                    self._handle_unclosed_resources(unclosed)
                
                # Check matplotlib figures
                self._check_matplotlib_figures()
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
            
            # Wait for next check
            self._stop_monitoring.wait(self.config.check_interval)
    
    def validate_array_operation(
        self,
        array: Union[np.ndarray, pd.DataFrame, pd.Series, list],
        operation: str,
        indices: Optional[Union[int, slice, Tuple]] = None
    ) -> Tuple[bool, Optional[MemoryIssue]]:
        """Validate array operation for bounds and memory safety."""
        try:
            # Check array size
            if isinstance(array, (np.ndarray, pd.DataFrame, pd.Series)):
                size = array.size
            else:
                size = len(array)
            
            if size > self.config.max_array_size:
                return False, MemoryIssue(
                    issue_type=MemoryIssueType.EXCESSIVE_ALLOCATION,
                    description=f"Array too large: {size} elements",
                    location=operation,
                    severity='high',
                    size_bytes=size * 8,  # Assume 8 bytes per element
                    suggestion=f"Consider processing in chunks or increasing max_array_size (current: {self.config.max_array_size})"
                )
            
            # Check bounds if indices provided
            if indices is not None:
                if isinstance(indices, int):
                    if indices < 0 or indices >= size:
                        return False, MemoryIssue(
                            issue_type=MemoryIssueType.ARRAY_BOUNDS,
                            description=f"Index {indices} out of bounds for array of size {size}",
                            location=operation,
                            severity='critical',
                            suggestion="Check array bounds before accessing"
                        )
                elif isinstance(indices, slice):
                    start = indices.start or 0
                    stop = indices.stop or size
                    if start < 0 or stop > size:
                        return False, MemoryIssue(
                            issue_type=MemoryIssueType.ARRAY_BOUNDS,
                            description=f"Slice {indices} out of bounds for array of size {size}",
                            location=operation,
                            severity='critical',
                            suggestion="Validate slice bounds before operation"
                        )
            
            return True, None
            
        except Exception as e:
            return False, MemoryIssue(
                issue_type=MemoryIssueType.ARRAY_BOUNDS,
                description=f"Array validation error: {str(e)}",
                location=operation,
                severity='high'
            )
    
    def validate_string_operation(
        self,
        string: str,
        operation: str,
        max_length: Optional[int] = None
    ) -> Tuple[bool, Optional[MemoryIssue]]:
        """Validate string operation for overflow."""
        max_len = max_length or self.config.max_string_length
        
        if len(string) > max_len:
            return False, MemoryIssue(
                issue_type=MemoryIssueType.STRING_OVERFLOW,
                description=f"String too long: {len(string)} chars",
                location=operation,
                severity='medium',
                size_bytes=len(string),
                suggestion=f"Truncate string to {max_len} characters"
            )
        
        return True, None
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # System memory
        virtual_memory = psutil.virtual_memory()
        
        return {
            'process_rss_mb': memory_info.rss / 1024 / 1024,
            'process_vms_mb': memory_info.vms / 1024 / 1024,
            'system_total_mb': virtual_memory.total / 1024 / 1024,
            'system_available_mb': virtual_memory.available / 1024 / 1024,
            'percent': virtual_memory.percent
        }
    
    def _handle_high_memory(self, memory_info: Dict[str, Any]):
        """Handle high memory usage."""
        issue = MemoryIssue(
            issue_type=MemoryIssueType.EXCESSIVE_ALLOCATION,
            description=f"High memory usage: {memory_info['percent']:.1f}%",
            location='system',
            severity='high' if memory_info['percent'] < 90 else 'critical',
            size_bytes=int(memory_info['process_rss_mb'] * 1024 * 1024),
            suggestion="Consider freeing memory or increasing system resources"
        )
        self._issues.append(issue)
        
        # Perform emergency cleanup if needed
        if memory_info['system_available_mb'] < self.config.emergency_threshold_mb:
            self.emergency_cleanup()
    
    def _handle_unclosed_resources(self, unclosed: Dict[str, List[Any]]):
        """Handle unclosed resources."""
        for resource_type, resources in unclosed.items():
            if len(resources) > 10:  # Threshold for concern
                issue = MemoryIssue(
                    issue_type=MemoryIssueType.UNCLOSED_RESOURCE,
                    description=f"{len(resources)} unclosed {resource_type}",
                    location=resource_type,
                    severity='medium',
                    count=len(resources),
                    suggestion=f"Ensure all {resource_type} are properly closed"
                )
                self._issues.append(issue)
    
    def _check_matplotlib_figures(self):
        """Check for unclosed matplotlib figures."""
        fig_count = len(plt.get_fignums())
        if fig_count > self.config.max_figure_count:
            issue = MemoryIssue(
                issue_type=MemoryIssueType.MEMORY_LEAK,
                description=f"{fig_count} matplotlib figures open",
                location='matplotlib',
                severity='medium',
                count=fig_count,
                suggestion="Close figures with plt.close() after use"
            )
            self._issues.append(issue)
            
            # Auto-close oldest figures if enabled
            if self.config.enable_auto_cleanup:
                fignums = plt.get_fignums()
                for fignum in fignums[:-10]:  # Keep last 10
                    plt.close(fignum)
    
    def emergency_cleanup(self):
        """Perform emergency memory cleanup."""
        self.logger.warning("Performing emergency memory cleanup")
        
        # Force garbage collection
        for _ in range(3):
            gc.collect()
        
        # Close all matplotlib figures
        plt.close('all')
        
        # Clear caches in tracked resources
        # This is where you'd clear application-specific caches
        
        # Log memory after cleanup
        memory_after = self.check_memory_usage()
        self.logger.info(f"Memory after cleanup: {memory_after['process_rss_mb']:.1f} MB")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory safety report."""
        report = {
            'memory_usage': self.check_memory_usage(),
            'resource_stats': self._resource_tracker.get_stats(),
            'issues': [{
                'type': issue.issue_type.value,
                'description': issue.description,
                'location': issue.location,
                'severity': issue.severity,
                'suggestion': issue.suggestion
            } for issue in self._issues[-100:]],  # Last 100 issues
            'matplotlib_figures': len(plt.get_fignums()),
            'gc_stats': gc.get_stats()
        }
        
        # Add tracemalloc snapshot if available
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            report['top_memory_consumers'] = [
                {
                    'file': stat.traceback.format()[0],
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count
                }
                for stat in top_stats
            ]
        
        return report


# Decorators for memory-safe operations

def memory_safe_array(max_size: Optional[int] = None):
    """Decorator to ensure array operations are memory safe."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get validator instance
            validator = get_memory_validator()
            
            # Check array arguments
            for i, arg in enumerate(args):
                if isinstance(arg, (np.ndarray, pd.DataFrame, pd.Series, list)):
                    valid, issue = validator.validate_array_operation(
                        arg, f"{func.__name__}:arg{i}"
                    )
                    if not valid:
                        raise MemoryError(f"Memory safety violation: {issue.description}")
            
            # Execute function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def memory_safe_string(max_length: Optional[int] = None):
    """Decorator to ensure string operations are memory safe."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get validator instance
            validator = get_memory_validator()
            
            # Check string arguments
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    valid, issue = validator.validate_string_operation(
                        arg, f"{func.__name__}:arg{i}", max_length
                    )
                    if not valid:
                        raise ValueError(f"String safety violation: {issue.description}")
            
            # Execute function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


@contextmanager
def memory_safe_context(operation: str, max_memory_mb: Optional[int] = None):
    """Context manager for memory-safe operations."""
    validator = get_memory_validator()
    initial_memory = validator.check_memory_usage()
    
    try:
        yield validator
    finally:
        # Check memory growth
        final_memory = validator.check_memory_usage()
        growth_mb = final_memory['process_rss_mb'] - initial_memory['process_rss_mb']
        
        if max_memory_mb and growth_mb > max_memory_mb:
            issue = MemoryIssue(
                issue_type=MemoryIssueType.EXCESSIVE_ALLOCATION,
                description=f"Operation '{operation}' used {growth_mb:.1f} MB",
                location=operation,
                severity='high',
                size_bytes=int(growth_mb * 1024 * 1024),
                suggestion=f"Optimize to use less than {max_memory_mb} MB"
            )
            validator._issues.append(issue)


# Global instance management
_memory_validator: Optional[MemorySafetyValidator] = None
_validator_lock = threading.Lock()


def get_memory_validator(config: Optional[MemorySafetyConfig] = None) -> MemorySafetyValidator:
    """Get or create the global memory validator instance."""
    global _memory_validator
    
    with _validator_lock:
        if _memory_validator is None:
            _memory_validator = MemorySafetyValidator(config)
        
        return _memory_validator


def cleanup_memory_validator():
    """Cleanup the global memory validator."""
    global _memory_validator
    
    with _validator_lock:
        if _memory_validator:
            _memory_validator.stop_monitoring()
            _memory_validator = None


# Safe resource wrappers

class SafeFileHandle:
    """Memory-safe file handle wrapper."""
    
    def __init__(self, file_path: Path, mode: str = 'r', max_size_mb: int = 100):
        self.file_path = file_path
        self.mode = mode
        self.max_size_mb = max_size_mb
        self._handle = None
        self._validator = get_memory_validator()
        
    def __enter__(self):
        # Check file size before opening
        if 'r' in self.mode and self.file_path.exists():
            size_mb = self.file_path.stat().st_size / 1024 / 1024
            if size_mb > self.max_size_mb:
                raise MemoryError(f"File too large: {size_mb:.1f} MB > {self.max_size_mb} MB")
        
        self._handle = open(self.file_path, self.mode)
        self._validator._resource_tracker.register('files', self._handle)
        return self._handle
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle:
            self._validator._resource_tracker.unregister('files', self._handle)
            self._handle.close()


class SafeCache:
    """Memory-safe cache implementation."""
    
    def __init__(self, max_entries: int = 1000, max_memory_mb: int = 100):
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self._lock = threading.Lock()
        self._validator = get_memory_validator()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with size checking."""
        with self._lock:
            # Check cache size
            if len(self._cache) >= self.max_entries:
                # Evict least recently used
                lru_key = min(self._access_times, key=self._access_times.get)
                del self._cache[lru_key]
                del self._access_times[lru_key]
            
            # Add new item
            self._cache[key] = value
            self._access_times[key] = time.time()
            
            # Check memory usage
            if len(self._cache) % 100 == 0:  # Check every 100 items
                memory = self._validator.check_memory_usage()
                if memory['process_rss_mb'] > self.max_memory_mb:
                    # Clear half the cache
                    items_to_remove = len(self._cache) // 2
                    sorted_keys = sorted(self._access_times, key=self._access_times.get)
                    for key in sorted_keys[:items_to_remove]:
                        del self._cache[key]
                        del self._access_times[key]
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()


# Export main components
__all__ = [
    'MemorySafetyValidator',
    'MemorySafetyConfig',
    'MemoryIssue',
    'MemoryIssueType',
    'ResourceTracker',
    'memory_safe_array',
    'memory_safe_string',
    'memory_safe_context',
    'get_memory_validator',
    'cleanup_memory_validator',
    'SafeFileHandle',
    'SafeCache'
]