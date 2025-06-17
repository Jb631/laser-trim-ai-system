"""
Resource Manager for Laser Trim Analyzer

Provides comprehensive memory and resource management for batch processing,
including dynamic adjustment of processing parameters based on available resources.
"""

import gc
import logging
import psutil
import threading
import time
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import warnings

# Import error handling utilities
from laser_trim_analyzer.core.error_handlers import (
    ErrorCode, ErrorCategory, ErrorSeverity,
    error_handler, handle_errors
)

logger = logging.getLogger(__name__)


@dataclass
class ResourceStatus:
    """Current system resource status."""
    # Memory metrics
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    memory_percent: float
    process_memory_mb: float
    
    # CPU metrics
    cpu_percent: float
    cpu_count: int
    
    # Disk metrics
    disk_free_mb: float
    disk_percent: float
    
    # Status flags
    memory_critical: bool
    memory_warning: bool
    cpu_high: bool
    disk_low: bool
    
    # Recommendations
    max_concurrent_files: int
    recommended_batch_size: int
    should_disable_plots: bool
    should_reduce_workers: bool
    
    # Timestamp
    timestamp: datetime


class ResourceManager:
    """
    Manages system resources for batch processing operations.
    
    Features:
    - Real-time resource monitoring
    - Dynamic adjustment of processing parameters
    - Memory pressure detection and mitigation
    - Graceful degradation under resource constraints
    - Automatic recovery from resource exhaustion
    """
    
    # Resource thresholds
    MEMORY_WARNING_THRESHOLD = 80  # % of total memory
    MEMORY_CRITICAL_THRESHOLD = 90  # % of total memory
    CPU_HIGH_THRESHOLD = 80  # % CPU usage
    DISK_LOW_THRESHOLD_MB = 500  # MB of free disk space
    
    # Processing limits based on memory
    MEMORY_PER_FILE_MB = 50  # Estimated memory per file
    MIN_FREE_MEMORY_MB = 200  # Minimum free memory to maintain
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize resource manager."""
        self.logger = logger or logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_thread = None
        self._resource_callbacks: List[Callable[[ResourceStatus], None]] = []
        self._last_gc_time = time.time()
        self._gc_interval = 30  # seconds
        
        # Cache for resource status
        self._current_status: Optional[ResourceStatus] = None
        self._status_lock = threading.Lock()
        
        # Memory pressure mitigation
        self._memory_pressure_count = 0
        self._last_memory_warning = 0
        
        self.logger.info("Resource manager initialized")
    
    def start_monitoring(self, interval: float = 5.0):
        """Start resource monitoring."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Resource monitoring stopped")
    
    def add_callback(self, callback: Callable[[ResourceStatus], None]):
        """Add a callback for resource status updates."""
        self._resource_callbacks.append(callback)
    
    def get_current_status(self) -> ResourceStatus:
        """Get current resource status."""
        with self._status_lock:
            if self._current_status:
                return self._current_status
        
        # Generate fresh status if none cached
        return self._check_resources()
    
    def _monitor_resources(self, interval: float):
        """Monitor resources in background thread."""
        while self._monitoring:
            try:
                status = self._check_resources()
                
                # Update cached status
                with self._status_lock:
                    self._current_status = status
                
                # Call callbacks
                for callback in self._resource_callbacks:
                    try:
                        callback(status)
                    except Exception as e:
                        self.logger.error(f"Resource callback error: {e}")
                
                # Handle resource pressure
                self._handle_resource_pressure(status)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
            
            time.sleep(interval)
    
    def _check_resources(self) -> ResourceStatus:
        """Check current system resources."""
        try:
            # Memory info
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # CPU info
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Disk info (for current working directory)
            disk = psutil.disk_usage(os.getcwd())
            disk_free_mb = disk.free / (1024 * 1024)
            
            # Determine status flags
            memory_critical = memory.percent >= self.MEMORY_CRITICAL_THRESHOLD
            memory_warning = memory.percent >= self.MEMORY_WARNING_THRESHOLD
            cpu_high = cpu_percent >= self.CPU_HIGH_THRESHOLD
            disk_low = disk_free_mb < self.DISK_LOW_THRESHOLD_MB
            
            # Calculate recommendations
            available_for_processing = memory.available / (1024 * 1024) - self.MIN_FREE_MEMORY_MB
            max_concurrent = max(1, int(available_for_processing / self.MEMORY_PER_FILE_MB))
            
            # Adjust for CPU
            if cpu_high:
                max_concurrent = min(max_concurrent, cpu_count // 2)
            
            # Batch size recommendation
            if memory_critical:
                batch_size = min(10, max_concurrent)
            elif memory_warning:
                batch_size = min(25, max_concurrent * 2)
            else:
                batch_size = min(50, max_concurrent * 5)
            
            return ResourceStatus(
                total_memory_mb=memory.total / (1024 * 1024),
                available_memory_mb=memory.available / (1024 * 1024),
                used_memory_mb=memory.used / (1024 * 1024),
                memory_percent=memory.percent,
                process_memory_mb=process_memory,
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                disk_free_mb=disk_free_mb,
                disk_percent=disk.percent,
                memory_critical=memory_critical,
                memory_warning=memory_warning,
                cpu_high=cpu_high,
                disk_low=disk_low,
                max_concurrent_files=max_concurrent,
                recommended_batch_size=batch_size,
                should_disable_plots=memory_warning or disk_low,
                should_reduce_workers=memory_critical or cpu_high,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to check resources: {e}")
            # Return conservative defaults on error
            return ResourceStatus(
                total_memory_mb=4096,  # Assume 4GB
                available_memory_mb=1024,
                used_memory_mb=3072,
                memory_percent=75,
                process_memory_mb=500,
                cpu_percent=50,
                cpu_count=4,
                disk_free_mb=1000,
                disk_percent=50,
                memory_critical=False,
                memory_warning=False,
                cpu_high=False,
                disk_low=False,
                max_concurrent_files=2,
                recommended_batch_size=10,
                should_disable_plots=True,
                should_reduce_workers=False,
                timestamp=datetime.now()
            )
    
    def _handle_resource_pressure(self, status: ResourceStatus):
        """Handle resource pressure situations."""
        current_time = time.time()
        
        # Memory pressure handling
        if status.memory_critical:
            self._memory_pressure_count += 1
            
            # Log warning if not recently warned
            if current_time - self._last_memory_warning > 60:  # Once per minute
                error_handler.handle_error(
                    error=MemoryError("Critical memory usage detected"),
                    category=ErrorCategory.RESOURCE,
                    severity=ErrorSeverity.WARNING,
                    code=ErrorCode.INSUFFICIENT_MEMORY,
                    user_message=f"Memory usage critical: {status.memory_percent:.1f}%",
                    recovery_suggestions=[
                        "Close other applications to free memory",
                        "Reduce batch size or concurrent files",
                        "Consider processing files in smaller groups"
                    ],
                    additional_data={
                        'memory_percent': status.memory_percent,
                        'available_mb': status.available_memory_mb,
                        'process_mb': status.process_memory_mb
                    }
                )
                self._last_memory_warning = current_time
            
            # Force garbage collection
            if current_time - self._last_gc_time > 10:  # More frequent during pressure
                self.force_cleanup()
                
        elif status.memory_warning:
            # Periodic GC during warning state
            if current_time - self._last_gc_time > self._gc_interval:
                self.force_cleanup()
        else:
            # Reset pressure count when memory is normal
            self._memory_pressure_count = 0
        
        # Disk space warning
        if status.disk_low:
            error_handler.handle_error(
                error=OSError("Low disk space"),
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.WARNING,
                code=ErrorCode.INSUFFICIENT_DISK_SPACE,
                user_message=f"Low disk space: {status.disk_free_mb:.1f}MB free",
                recovery_suggestions=[
                    "Free up disk space",
                    "Choose a different output directory",
                    "Disable plot generation to save space"
                ]
            )
    
    def force_cleanup(self):
        """Force memory cleanup."""
        self.logger.debug("Forcing memory cleanup")
        
        # Clear matplotlib figures
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass
        
        # Force garbage collection
        gc.collect()
        
        self._last_gc_time = time.time()
    
    @handle_errors(
        category=ErrorCategory.RESOURCE,
        severity=ErrorSeverity.WARNING
    )
    def check_batch_feasibility(
        self, 
        file_count: int,
        enable_plots: bool = False,
        concurrent_files: Optional[int] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if batch processing is feasible with current resources.
        
        Args:
            file_count: Number of files to process
            enable_plots: Whether plots will be generated
            concurrent_files: Requested concurrent files (None for auto)
            
        Returns:
            Tuple of (is_feasible, recommendations)
        """
        status = self.get_current_status()
        
        # Estimate memory requirements
        memory_per_file = self.MEMORY_PER_FILE_MB
        if enable_plots:
            memory_per_file *= 1.5  # 50% more with plots
        
        # Determine actual concurrent files
        if concurrent_files is None:
            actual_concurrent = status.max_concurrent_files
        else:
            actual_concurrent = min(concurrent_files, status.max_concurrent_files)
        
        # Calculate total memory needed
        peak_memory_needed = actual_concurrent * memory_per_file
        
        # Check feasibility
        is_feasible = True
        warnings = []
        errors = []
        
        if peak_memory_needed > status.available_memory_mb:
            is_feasible = False
            errors.append(f"Insufficient memory: need {peak_memory_needed:.0f}MB, have {status.available_memory_mb:.0f}MB")
        
        if status.memory_critical:
            warnings.append("System memory is already critically low")
        
        if status.disk_low:
            warnings.append("Disk space is low")
        
        if file_count > 1000 and enable_plots:
            warnings.append("Large batch with plots enabled may cause performance issues")
        
        # Build recommendations
        recommendations = {
            'max_concurrent_files': actual_concurrent,
            'recommended_batch_size': min(status.recommended_batch_size, file_count),
            'disable_plots': status.should_disable_plots or (file_count > 200),
            'reduce_workers': status.should_reduce_workers,
            'estimated_memory_usage_mb': peak_memory_needed,
            'available_memory_mb': status.available_memory_mb,
            'warnings': warnings,
            'errors': errors
        }
        
        # Log recommendations
        if not is_feasible:
            self.logger.warning(f"Batch processing not feasible: {errors}")
        elif warnings:
            self.logger.warning(f"Batch processing warnings: {warnings}")
        
        return is_feasible, recommendations
    
    def get_adaptive_batch_size(self, total_files: int, current_index: int = 0) -> int:
        """
        Get adaptive batch size based on current resources and progress.
        
        Args:
            total_files: Total number of files
            current_index: Current processing index
            
        Returns:
            Recommended batch size
        """
        status = self.get_current_status()
        base_size = status.recommended_batch_size
        
        # Adjust based on progress
        progress_percent = (current_index / total_files * 100) if total_files > 0 else 0
        
        # Start conservatively, increase if resources are stable
        if progress_percent < 10:
            # First 10% - be conservative
            batch_size = min(base_size, 10)
        elif progress_percent < 25 and self._memory_pressure_count == 0:
            # If no issues, can increase
            batch_size = base_size
        elif self._memory_pressure_count > 2:
            # Reduce if experiencing pressure
            batch_size = max(1, base_size // 2)
        else:
            batch_size = base_size
        
        # Never exceed remaining files
        remaining = total_files - current_index
        # Ensure we never return 0 to prevent infinite loops
        return max(1, min(batch_size, remaining))
    
    def should_pause_processing(self) -> bool:
        """
        Check if processing should be paused due to resource constraints.
        
        Returns:
            True if processing should pause
        """
        status = self.get_current_status()
        
        # Pause if memory is critical and pressure is sustained
        if status.memory_critical and self._memory_pressure_count > 5:
            self.logger.warning("Recommending processing pause due to sustained memory pressure")
            return True
        
        # Pause if available memory is extremely low
        if status.available_memory_mb < 100:
            self.logger.warning("Recommending processing pause due to extremely low memory")
            return True
        
        return False
    
    def wait_for_resources(self, timeout: float = 30.0, check_cancelled=None) -> bool:
        """
        Wait for resources to become available.
        
        Args:
            timeout: Maximum time to wait in seconds
            check_cancelled: Optional callable to check if operation was cancelled
            
        Returns:
            True if resources became available, False if timeout or cancelled
        """
        start_time = time.time()
        
        self.logger.info("Waiting for resources to become available...")
        
        while time.time() - start_time < timeout:
            # Check if cancelled
            if check_cancelled and check_cancelled():
                self.logger.info("Resource wait cancelled")
                return False
                
            status = self.get_current_status()
            
            if not status.memory_critical and status.available_memory_mb > self.MIN_FREE_MEMORY_MB:
                self.logger.info("Resources available, resuming processing")
                return True
            
            # Force cleanup while waiting
            self.force_cleanup()
            
            # Use shorter sleep intervals for better responsiveness
            sleep_time = 0.1
            sleep_count = 0
            while sleep_count < 20 and time.time() - start_time < timeout:  # 20 * 0.1 = 2 seconds total
                if check_cancelled and check_cancelled():
                    return False
                time.sleep(sleep_time)
                sleep_count += 1
        
        self.logger.warning("Timeout waiting for resources")
        return False


class BatchResourceOptimizer:
    """
    Optimizes batch processing based on resource availability.
    """
    
    def __init__(self, resource_manager: ResourceManager):
        """Initialize optimizer with resource manager."""
        self.resource_manager = resource_manager
        self.logger = logging.getLogger(__name__)
    
    def optimize_processing_params(
        self,
        file_count: int,
        requested_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize processing parameters based on resources.
        
        Args:
            file_count: Number of files to process
            requested_params: Requested processing parameters
            
        Returns:
            Optimized parameters
        """
        # Get current resources
        status = self.resource_manager.get_current_status()
        
        # Start with requested params
        optimized = requested_params.copy()
        
        # Check feasibility
        is_feasible, recommendations = self.resource_manager.check_batch_feasibility(
            file_count,
            enable_plots=requested_params.get('generate_plots', False),
            concurrent_files=requested_params.get('max_concurrent_files')
        )
        
        # Apply recommendations
        if not is_feasible or recommendations['warnings']:
            self.logger.info("Applying resource-based optimizations")
            
            # Adjust concurrent files
            optimized['max_concurrent_files'] = recommendations['max_concurrent_files']
            
            # Adjust batch size
            optimized['batch_size'] = recommendations['recommended_batch_size']
            
            # Disable plots if recommended
            if recommendations['disable_plots']:
                optimized['generate_plots'] = False
                self.logger.info("Disabling plot generation to save resources")
            
            # Reduce workers if needed
            if recommendations['reduce_workers']:
                current_workers = optimized.get('max_workers', 4)
                optimized['max_workers'] = max(1, current_workers // 2)
                self.logger.info(f"Reducing workers from {current_workers} to {optimized['max_workers']}")
            
            # Enable memory-saving options
            optimized['clear_cache_interval'] = min(25, file_count // 10)
            optimized['gc_interval'] = 20
            
            # Add resource warnings to UI
            if recommendations['warnings']:
                optimized['resource_warnings'] = recommendations['warnings']
        
        return optimized
    
    def get_processing_strategy(self, file_count: int) -> Dict[str, Any]:
        """
        Get recommended processing strategy based on file count and resources.
        
        Args:
            file_count: Number of files to process
            
        Returns:
            Processing strategy recommendations
        """
        status = self.resource_manager.get_current_status()
        
        # Determine strategy based on file count and resources
        if file_count < 50:
            # Small batch - standard processing
            strategy = 'standard'
            chunk_size = file_count
            enable_plots = True
            max_workers = min(4, status.cpu_count)
            
        elif file_count < 200:
            # Medium batch - balanced processing
            strategy = 'balanced'
            chunk_size = 50
            enable_plots = not status.memory_warning
            max_workers = min(4, status.cpu_count // 2)
            
        elif file_count < 1000:
            # Large batch - optimized processing
            strategy = 'optimized'
            chunk_size = status.recommended_batch_size
            enable_plots = False
            max_workers = min(2, status.cpu_count // 2)
            
        else:
            # Very large batch - streaming processing
            strategy = 'streaming'
            chunk_size = min(25, status.recommended_batch_size)
            enable_plots = False
            max_workers = 1
        
        return {
            'strategy': strategy,
            'chunk_size': chunk_size,
            'enable_plots': enable_plots,
            'max_workers': max_workers,
            'enable_streaming': strategy == 'streaming',
            'memory_monitoring': file_count > 100,
            'aggressive_cleanup': status.memory_warning or file_count > 500,
            'description': self._get_strategy_description(strategy)
        }
    
    def _get_strategy_description(self, strategy: str) -> str:
        """Get human-readable description of processing strategy."""
        descriptions = {
            'standard': "Standard processing with full features",
            'balanced': "Balanced processing with selective optimizations",
            'optimized': "Optimized processing for large batches",
            'streaming': "Memory-efficient streaming for very large batches"
        }
        return descriptions.get(strategy, "Custom processing strategy")


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get or create global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
        _resource_manager.start_monitoring()
    return _resource_manager


def cleanup_resource_manager():
    """Cleanup global resource manager."""
    global _resource_manager
    if _resource_manager:
        _resource_manager.stop_monitoring()
        _resource_manager = None