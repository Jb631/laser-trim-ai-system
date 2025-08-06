"""
Performance monitoring utilities for preventing system-wide slowdowns.

Monitors system resources and provides throttling to maintain system responsiveness.
"""

import time
import threading
import logging
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    memory_usage_mb: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    files_processed: int = 0
    processing_rate: float = 0.0  # files per second
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0
    system_responsive: bool = True
    throttling_active: bool = False


class PerformanceMonitor:
    """
    Monitors system performance and provides throttling to prevent system-wide slowdowns.
    """
    
    def __init__(
        self,
        memory_limit_mb: float = 1500.0,
        cpu_throttle_threshold: float = 95.0,
        update_interval: float = 1.0,  # seconds
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize performance monitor.
        
        Args:
            memory_limit_mb: Memory usage threshold for throttling
            cpu_throttle_threshold: CPU usage threshold for throttling
            update_interval: How often to update metrics
            logger: Logger instance
        """
        self.memory_limit_mb = memory_limit_mb
        self.cpu_throttle_threshold = cpu_throttle_threshold
        self.update_interval = update_interval
        self.logger = logger or logging.getLogger(__name__)
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._metrics = PerformanceMetrics()
        self._start_time: Optional[float] = None
        self._total_files = 0
        
        # Callbacks
        self._metric_callbacks: list[Callable[[PerformanceMetrics], None]] = []
        self._throttle_callbacks: list[Callable[[], None]] = []
        
        # Performance tracking
        self._last_gc_time = 0.0
        self._last_matplotlib_cleanup = 0.0
        
        if not HAS_PSUTIL:
            self.logger.warning("psutil not available - performance monitoring limited")

    def start_monitoring(self, total_files: int = 0):
        """Start performance monitoring."""
        if self._monitoring:
            self.logger.warning("Performance monitoring already active")
            return
            
        self._monitoring = True
        self._start_time = time.time()
        self._total_files = total_files
        self._metrics = PerformanceMetrics()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="PerformanceMonitor"
        )
        self._monitor_thread.start()
        
        self.logger.info(f"Performance monitoring started for {total_files} files")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self._monitoring:
            return
            
        self._monitoring = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
            
        self.logger.info("Performance monitoring stopped")

    def update_progress(self, files_processed: int):
        """Update processing progress."""
        self._metrics.files_processed = files_processed
        
        # Calculate processing rate
        if self._start_time:
            elapsed = time.time() - self._start_time
            self._metrics.elapsed_time = elapsed
            
            if elapsed > 0:
                self._metrics.processing_rate = files_processed / elapsed
                
                # Estimate remaining time
                if self._total_files > 0 and self._metrics.processing_rate > 0:
                    remaining_files = self._total_files - files_processed
                    self._metrics.estimated_remaining = remaining_files / self._metrics.processing_rate

    def add_metric_callback(self, callback: Callable[[PerformanceMetrics], None]):
        """Add callback for metric updates."""
        self._metric_callbacks.append(callback)

    def add_throttle_callback(self, callback: Callable[[], None]):
        """Add callback for throttling events."""
        self._throttle_callbacks.append(callback)

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self._metrics

    def should_throttle(self) -> bool:
        """Check if processing should be throttled."""
        return not self._metrics.system_responsive

    def perform_cleanup(self, force: bool = False) -> bool:
        """
        Perform memory cleanup if needed.
        
        Args:
            force: Force cleanup regardless of thresholds
            
        Returns:
            True if cleanup was performed
        """
        current_time = time.time()
        cleanup_performed = False
        
        # Garbage collection
        if force or (current_time - self._last_gc_time > 30):  # Every 30 seconds minimum
            import gc
            collected = gc.collect()
            self._last_gc_time = current_time
            cleanup_performed = True
            
            if collected > 0:
                self.logger.debug(f"Garbage collected {collected} objects")
        
        # Matplotlib cleanup
        if force or (current_time - self._last_matplotlib_cleanup > 15):  # Every 15 seconds
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
                self._last_matplotlib_cleanup = current_time
                cleanup_performed = True
                self.logger.debug("Closed all matplotlib figures")
            except Exception as e:
                self.logger.debug(f"Matplotlib cleanup failed: {e}")
        
        return cleanup_performed

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self._update_metrics()
                self._check_thresholds()
                self._notify_callbacks()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                time.sleep(self.update_interval * 2)  # Back off on errors

    def _update_metrics(self):
        """Update system metrics."""
        if not HAS_PSUTIL:
            return
            
        try:
            # Get current process
            process = psutil.Process()
            
            # Memory metrics
            memory_info = process.memory_info()
            self._metrics.memory_usage_mb = memory_info.rss / (1024 * 1024)
            self._metrics.memory_percent = process.memory_percent()
            
            # CPU metrics (averaged over update interval)
            self._metrics.cpu_percent = process.cpu_percent()
            
        except Exception as e:
            self.logger.debug(f"Failed to update system metrics: {e}")

    def _check_thresholds(self):
        """Check performance thresholds and update system responsiveness."""
        # Check memory threshold
        memory_throttle = self._metrics.memory_usage_mb > self.memory_limit_mb
        
        # Check CPU threshold
        cpu_throttle = self._metrics.cpu_percent > self.cpu_throttle_threshold
        
        # Update system responsiveness
        was_responsive = self._metrics.system_responsive
        self._metrics.system_responsive = not (memory_throttle or cpu_throttle)
        self._metrics.throttling_active = memory_throttle or cpu_throttle
        
        # Log threshold violations
        if memory_throttle:
            self.logger.warning(f"Memory threshold exceeded: {self._metrics.memory_usage_mb:.1f}MB > {self.memory_limit_mb}MB")
            
        if cpu_throttle:
            self.logger.warning(f"CPU threshold exceeded: {self._metrics.cpu_percent:.1f}% > {self.cpu_throttle_threshold}%")
        
        # Trigger throttle callbacks if system became unresponsive
        if was_responsive and not self._metrics.system_responsive:
            for callback in self._throttle_callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.error(f"Throttle callback error: {e}")

    def _notify_callbacks(self):
        """Notify metric callbacks."""
        for callback in self._metric_callbacks:
            try:
                callback(self._metrics)
            except Exception as e:
                self.logger.debug(f"Metric callback error: {e}")


class AdaptiveThrottler:
    """
    Provides adaptive throttling based on system performance.
    """
    
    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        base_delay: float = 0.001,  # 1ms base delay
        max_delay: float = 0.1,     # 100ms max delay
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize adaptive throttler.
        
        Args:
            performance_monitor: Performance monitor instance
            base_delay: Base delay when system is responsive
            max_delay: Maximum delay when system is under stress
            logger: Logger instance
        """
        self.performance_monitor = performance_monitor
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.logger = logger or logging.getLogger(__name__)
        
        self._current_delay = base_delay

    async def throttle(self):
        """Apply adaptive throttling delay."""
        import asyncio
        
        metrics = self.performance_monitor.get_metrics()
        
        if metrics.throttling_active:
            # Calculate adaptive delay based on system stress
            memory_factor = min(1.0, metrics.memory_usage_mb / self.performance_monitor.memory_limit_mb)
            cpu_factor = min(1.0, metrics.cpu_percent / self.performance_monitor.cpu_throttle_threshold)
            
            stress_factor = max(memory_factor, cpu_factor)
            self._current_delay = self.base_delay + (self.max_delay - self.base_delay) * stress_factor
            
            self.logger.debug(f"Adaptive throttling: delay={self._current_delay*1000:.1f}ms, stress={stress_factor:.2f}")
        else:
            self._current_delay = self.base_delay
        
        await asyncio.sleep(self._current_delay)

    def get_current_delay(self) -> float:
        """Get current throttling delay."""
        return self._current_delay


def create_performance_optimized_config(file_count: int) -> Dict[str, Any]:
    """
    Create performance-optimized configuration based on file count.
    
    Args:
        file_count: Number of files to process
        
    Returns:
        Dictionary with optimized configuration values
    """
    config_overrides = {}
    
    if file_count > 1000:
        # Very large batches
        config_overrides.update({
            'max_workers': 6,
            'concurrent_batch_size': 15,
            'memory_limit_mb': 3072,  # 3GB
            'generate_plots': False,
            'ui_update_throttle_ms': 500,  # Less frequent UI updates
            'garbage_collection_interval': 25,
            'matplotlib_cleanup_interval': 15,
        })
    elif file_count > 500:
        # Large batches
        config_overrides.update({
            'max_workers': 8,
            'concurrent_batch_size': 20,
            'memory_limit_mb': 2048,  # 2GB
            'ui_update_throttle_ms': 350,
            'garbage_collection_interval': 40,
            'matplotlib_cleanup_interval': 20,
        })
    elif file_count > 100:
        # Medium batches
        config_overrides.update({
            'max_workers': 10,
            'concurrent_batch_size': 25,
            'memory_limit_mb': 1536,  # 1.5GB
            'ui_update_throttle_ms': 250,
            'garbage_collection_interval': 50,
        })
    
    return config_overrides 