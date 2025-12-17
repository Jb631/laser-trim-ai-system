"""
Enhanced Batch Processing Logging Module

Provides comprehensive logging capabilities for large-scale batch processing,
including performance metrics, memory tracking, and detailed error logging.
"""

import logging
import json
import time
import psutil
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import threading
from collections import deque


class BatchProcessingLogger:
    """Enhanced logger for batch processing operations."""
    
    def __init__(self, batch_id: str, log_dir: Path, enable_performance_tracking: bool = True):
        """
        Initialize batch processing logger.
        
        Args:
            batch_id: Unique identifier for this batch
            log_dir: Directory for log files
            enable_performance_tracking: Enable detailed performance metrics
        """
        self.batch_id = batch_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_performance_tracking = enable_performance_tracking
        
        # Setup loggers
        self._setup_loggers()
        
        # Performance tracking
        self.start_time = time.time()
        self.file_processing_times = deque(maxlen=100)  # Keep last 100 for rolling average
        self.memory_samples = deque(maxlen=1000)  # Keep last 1000 memory samples
        
        # Counters
        self.counters = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'total_tracks': 0,
            'failed_tracks': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'warnings': 0
        }
        
        # Error tracking
        self.error_summary = {}
        self.warning_summary = {}
        
        # Memory tracking
        self._last_gc_time = time.time()
        self._gc_count = 0
        
        # Performance metrics
        self.performance_metrics = {
            'peak_memory_mb': 0,
            'average_file_time': 0,
            'files_per_second': 0,
            'estimated_completion': None
        }
        
        # Start background monitoring if enabled
        self._monitoring_thread = None
        if enable_performance_tracking:
            self._start_monitoring()
    
    def _setup_loggers(self):
        """Setup specialized loggers for different aspects."""
        # Main batch logger
        self.main_logger = self._create_logger(
            'batch_main',
            self.log_dir / f'batch_{self.batch_id}_main.log',
            logging.INFO
        )
        
        # Performance logger
        self.perf_logger = self._create_logger(
            'batch_performance',
            self.log_dir / f'batch_{self.batch_id}_performance.log',
            logging.INFO,
            format_string='%(asctime)s.%(msecs)03d - %(message)s'
        )
        
        # Error logger
        self.error_logger = self._create_logger(
            'batch_errors',
            self.log_dir / f'batch_{self.batch_id}_errors.log',
            logging.ERROR
        )
        
        # Detail logger (for debugging)
        self.detail_logger = self._create_logger(
            'batch_detail',
            self.log_dir / f'batch_{self.batch_id}_detail.log',
            logging.DEBUG
        )
        
        # Summary logger (for final report)
        self.summary_logger = self._create_logger(
            'batch_summary',
            self.log_dir / f'batch_{self.batch_id}_summary.log',
            logging.INFO
        )
    
    def _create_logger(self, name: str, log_file: Path, level: int, 
                      format_string: Optional[str] = None) -> logging.Logger:
        """Create a logger with file handler."""
        logger = logging.getLogger(f"{name}_{self.batch_id}")
        logger.setLevel(level)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # File handler
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setLevel(level)
        
        # Formatter
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.propagate = False
        
        return logger
    
    def log_batch_start(self, total_files: int, batch_config: Dict[str, Any]):
        """Log batch processing start."""
        self.counters['total_files'] = total_files
        
        self.main_logger.info("="*80)
        self.main_logger.info(f"BATCH PROCESSING STARTED - ID: {self.batch_id}")
        self.main_logger.info("="*80)
        self.main_logger.info(f"Total files to process: {total_files}")
        self.main_logger.info(f"Configuration: {json.dumps(batch_config, indent=2)}")
        
        # Log system info
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        self.main_logger.info(f"System: {cpu_count} CPUs, {memory.total / (1024**3):.1f}GB RAM")
        self.main_logger.info(f"Available memory: {memory.available / (1024**3):.1f}GB ({memory.percent}% used)")
    
    def log_file_start(self, file_path: Path, file_index: int):
        """Log individual file processing start."""
        self.detail_logger.debug(f"Starting file {file_index}/{self.counters['total_files']}: {file_path.name}")
        return time.time()
    
    def log_file_complete(self, file_path: Path, start_time: float, 
                         result: Optional[Any] = None, error: Optional[Exception] = None):
        """Log file processing completion."""
        processing_time = time.time() - start_time
        self.file_processing_times.append(processing_time)
        
        if error:
            self.counters['failed_files'] += 1
            self.counters['errors'] += 1
            
            error_type = type(error).__name__
            self.error_summary[error_type] = self.error_summary.get(error_type, 0) + 1
            
            self.error_logger.error(f"Failed to process {file_path.name}: {error}", exc_info=True)
            self.main_logger.error(f"File failed: {file_path.name} - {error_type}: {str(error)}")
        else:
            self.counters['processed_files'] += 1
            self.detail_logger.info(f"Completed {file_path.name} in {processing_time:.2f}s")
            
            # Log performance metrics
            if self.enable_performance_tracking:
                memory = psutil.virtual_memory()
                self.perf_logger.info(
                    f"File: {file_path.name} | "
                    f"Time: {processing_time:.2f}s | "
                    f"Memory: {memory.used / (1024**3):.1f}GB ({memory.percent}%) | "
                    f"Progress: {self.counters['processed_files']}/{self.counters['total_files']}"
                )
    
    def log_memory_cleanup(self, force_gc: bool = False):
        """Log memory cleanup operations."""
        if force_gc or (time.time() - self._last_gc_time) > 60:  # GC at most once per minute
            before_memory = psutil.Process().memory_info().rss / (1024**2)
            gc.collect()
            after_memory = psutil.Process().memory_info().rss / (1024**2)
            
            freed = before_memory - after_memory
            self._gc_count += 1
            self._last_gc_time = time.time()
            
            self.perf_logger.info(
                f"Garbage collection #{self._gc_count}: "
                f"Freed {freed:.1f}MB (before: {before_memory:.1f}MB, after: {after_memory:.1f}MB)"
            )
    
    def log_batch_progress(self, custom_message: Optional[str] = None):
        """Log current batch progress."""
        processed = self.counters['processed_files']
        failed = self.counters['failed_files']
        total = self.counters['total_files']
        
        if total == 0:
            return
        
        progress_pct = (processed + failed) / total * 100
        success_rate = (processed / (processed + failed) * 100) if (processed + failed) > 0 else 0
        
        # Calculate performance metrics
        elapsed_time = time.time() - self.start_time
        if self.file_processing_times:
            avg_time = sum(self.file_processing_times) / len(self.file_processing_times)
            files_per_second = 1 / avg_time if avg_time > 0 else 0
            
            remaining = total - processed - failed
            eta_seconds = remaining * avg_time
            eta = datetime.fromtimestamp(time.time() + eta_seconds).strftime('%Y-%m-%d %H:%M:%S')
        else:
            files_per_second = 0
            eta = "Unknown"
        
        message = (
            f"Progress: {progress_pct:.1f}% ({processed + failed}/{total}) | "
            f"Success: {success_rate:.1f}% | "
            f"Speed: {files_per_second:.1f} files/s | "
            f"ETA: {eta}"
        )
        
        if custom_message:
            message = f"{custom_message} | {message}"
        
        self.main_logger.info(message)
    
    def log_warning(self, message: str, category: Optional[str] = None):
        """Log a warning message."""
        self.counters['warnings'] += 1
        if category:
            self.warning_summary[category] = self.warning_summary.get(category, 0) + 1
        
        self.main_logger.warning(message)
    
    def log_error(self, message: str, error: Optional[Exception] = None, 
                  category: Optional[str] = None):
        """Log an error message."""
        self.counters['errors'] += 1
        if category or (error and type(error).__name__):
            cat = category or type(error).__name__
            self.error_summary[cat] = self.error_summary.get(cat, 0) + 1
        
        if error:
            self.error_logger.error(message, exc_info=True)
        else:
            self.error_logger.error(message)
    
    @contextmanager
    def track_operation(self, operation_name: str):
        """Context manager to track operation timing."""
        start_time = time.time()
        self.detail_logger.debug(f"Starting operation: {operation_name}")
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.detail_logger.debug(f"Completed operation: {operation_name} in {duration:.3f}s")
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        def monitor():
            while self._monitoring_thread and not getattr(self._monitoring_thread, 'stop', False):
                try:
                    # Sample memory usage
                    memory = psutil.virtual_memory()
                    process_memory = psutil.Process().memory_info().rss / (1024**2)
                    
                    self.memory_samples.append({
                        'timestamp': time.time(),
                        'system_percent': memory.percent,
                        'process_mb': process_memory
                    })
                    
                    # Update peak memory
                    if process_memory > self.performance_metrics['peak_memory_mb']:
                        self.performance_metrics['peak_memory_mb'] = process_memory
                    
                    time.sleep(5)  # Sample every 5 seconds
                except Exception as e:
                    self.detail_logger.error(f"Monitoring error: {e}")
        
        self._monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self._monitoring_thread.start()
    
    def finalize_batch(self):
        """Finalize batch processing and generate summary."""
        # Stop monitoring
        if self._monitoring_thread:
            self._monitoring_thread.stop = True
            self._monitoring_thread = None
        
        # Calculate final metrics
        total_time = time.time() - self.start_time
        processed = self.counters['processed_files']
        failed = self.counters['failed_files']
        
        # Generate summary
        summary = {
            'batch_id': self.batch_id,
            'duration_seconds': total_time,
            'duration_formatted': f"{total_time/3600:.1f} hours" if total_time > 3600 else f"{total_time/60:.1f} minutes",
            'counters': self.counters,
            'success_rate': (processed / (processed + failed) * 100) if (processed + failed) > 0 else 0,
            'average_file_time': sum(self.file_processing_times) / len(self.file_processing_times) if self.file_processing_times else 0,
            'files_per_second': (processed + failed) / total_time if total_time > 0 else 0,
            'peak_memory_mb': self.performance_metrics['peak_memory_mb'],
            'error_summary': self.error_summary,
            'warning_summary': self.warning_summary,
            'gc_collections': self._gc_count
        }
        
        # Log summary
        self.summary_logger.info("="*80)
        self.summary_logger.info("BATCH PROCESSING SUMMARY")
        self.summary_logger.info("="*80)
        self.summary_logger.info(json.dumps(summary, indent=2))
        
        # Save summary to JSON file
        summary_file = self.log_dir / f'batch_{self.batch_id}_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.main_logger.info(f"Batch processing completed. Summary saved to: {summary_file}")
        
        return summary


def setup_batch_logging(batch_id: Optional[str] = None, 
                       log_dir: Optional[Path] = None,
                       enable_performance: bool = True) -> BatchProcessingLogger:
    """
    Setup batch processing logger.
    
    Args:
        batch_id: Unique batch identifier (auto-generated if not provided)
        log_dir: Directory for logs (uses default if not provided)
        enable_performance: Enable performance tracking
    
    Returns:
        Configured BatchProcessingLogger instance
    """
    if batch_id is None:
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if log_dir is None:
        log_dir = Path.home() / ".laser_trim_analyzer" / "batch_logs"
    
    return BatchProcessingLogger(batch_id, log_dir, enable_performance)