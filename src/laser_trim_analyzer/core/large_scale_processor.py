"""
Large-Scale Processing Engine for Laser Trim Analyzer

Optimized for processing thousands of files with:
- Memory management and garbage collection
- Chunked processing and streaming
- Database batch operations
- Progress tracking and recovery
- Performance monitoring
"""

import asyncio
import gc
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, AsyncGenerator, Tuple
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from laser_trim_analyzer.core.config import Config
from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.core.models import AnalysisResult
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.core.error_handlers import (
    ErrorCode, ErrorCategory, ErrorSeverity,
    error_handler, handle_errors
)

# Import secure logging with fallback
try:
    from laser_trim_analyzer.core.secure_logging import (
        SecureLogger, LogLevel, get_logger, logged_function
    )
    HAS_SECURE_LOGGING = True
except ImportError:
    HAS_SECURE_LOGGING = False
    SecureLogger = None
    
    # Fallback to standard logging
    def get_logger(name: str, **kwargs) -> logging.Logger:
        return logging.getLogger(name)

# Import resource management
try:
    from laser_trim_analyzer.core.resource_manager import (
        get_resource_manager, BatchResourceOptimizer
    )
    HAS_RESOURCE_MANAGER = True
except ImportError:
    HAS_RESOURCE_MANAGER = False

# Import memory safety if available
try:
    from laser_trim_analyzer.core.memory_safety import (
        get_memory_validator, memory_safe_context, MemorySafetyConfig
    )
    HAS_MEMORY_SAFETY = True
except ImportError:
    HAS_MEMORY_SAFETY = False


class LargeScaleProcessor:
    """
    Specialized processor for handling thousands of files efficiently.
    
    Features:
    - Chunked batch processing
    - Memory management and cleanup
    - Progress tracking and recovery
    - Performance monitoring
    - Database batch operations
    """
    
    def __init__(
        self, 
        config: Config,
        db_manager: Optional[DatabaseManager] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize large-scale processor."""
        self.config = config
        self.db_manager = db_manager
        
        # Use secure logger if available
        if HAS_SECURE_LOGGING and not logger:
            self.logger = get_logger(
                __name__,
                log_level=LogLevel.DEBUG if config.debug else LogLevel.INFO,
                enable_performance_tracking=True,
                enable_input_sanitization=True
            )
            self.logger.info(
                "LargeScaleProcessor initialized with secure logging",
                context={
                    'config': {
                        'chunk_size': config.processing.chunk_size,
                        'max_concurrent_files': config.processing.max_concurrent_files,
                        'memory_limit_mb': getattr(config.processing, 'memory_limit_mb', 2000),
                        'gc_interval': config.processing.gc_interval,
                        'debug_mode': config.debug
                    },
                    'database_enabled': db_manager is not None,
                    'resource_manager': HAS_RESOURCE_MANAGER,
                    'memory_safety': HAS_MEMORY_SAFETY
                }
            )
        else:
            self.logger = logger or logging.getLogger(__name__)
        
        # Create base processor
        self.processor = LaserTrimProcessor(config, db_manager, logger=logger)
        
        # Performance tracking
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'start_time': None,
            'current_memory_mb': 0,
            'peak_memory_mb': 0,
            'files_per_second': 0.0,
            'estimated_completion': None
        }
        
        # State management
        self._is_processing = False
        self._should_stop = False
        self._processed_files = set()
        self._failed_files = set()
        
        # Performance monitoring
        self._memory_monitor = None
        self._last_gc_time = time.time()
        self._last_cache_clear = 0
        
        # Resource management
        if HAS_RESOURCE_MANAGER:
            self.resource_manager = get_resource_manager()
            self.resource_optimizer = BatchResourceOptimizer(self.resource_manager)
        else:
            self.resource_manager = None
            self.resource_optimizer = None

    async def process_large_directory(
        self,
        directory: Path,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float, Dict], None]] = None,
        file_filter: Optional[Callable[[Path], bool]] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, AnalysisResult]:
        """
        Process all files in a directory efficiently.
        
        Args:
            directory: Directory containing files to process
            output_dir: Output directory for results
            progress_callback: Progress callback with stats
            file_filter: Optional function to filter files
            resume_from: Resume from specific file (for crash recovery)
            
        Returns:
            Dictionary of successful results
        """
        # Start performance tracking
        if HAS_SECURE_LOGGING and isinstance(self.logger, SecureLogger):
            self.logger.start_performance_tracking('process_large_directory')
        
        start_time = time.time()
        
        self.logger.info(
            f"Starting large-scale processing of directory: {directory}",
            context={
                'directory': str(directory),
                'output_dir': str(output_dir) if output_dir else None,
                'resume_from': resume_from,
                'has_file_filter': file_filter is not None
            } if HAS_SECURE_LOGGING else None
        )
        
        # Discover files
        files = await self._discover_files(directory, file_filter)
        discovery_time = time.time() - start_time
        
        self.logger.info(
            f"Discovered {len(files)} files for processing",
            context={
                'total_files': len(files),
                'discovery_time_seconds': discovery_time,
                'files_per_second': len(files) / discovery_time if discovery_time > 0 else 0
            } if HAS_SECURE_LOGGING else None
        )
        
        if not files:
            self.logger.warning("No files found to process")
            return {}
        
        # Handle resume functionality
        if resume_from:
            files = self._filter_resume_files(files, resume_from)
            self.logger.info(f"Resuming from {resume_from}, {len(files)} files remaining")
        
        # Check resources and optimize parameters
        if self.resource_manager:
            self.logger.debug(
                "Checking resource feasibility for batch",
                context={
                    'batch_size': len(files),
                    'plots_enabled': self.config.processing.generate_plots
                } if HAS_SECURE_LOGGING else None
            )
            
            # Check if batch is feasible
            is_feasible, recommendations = self.resource_manager.check_batch_feasibility(
                len(files),
                enable_plots=self.config.processing.generate_plots
            )
            
            self.logger.info(
                f"Resource feasibility check complete: {'feasible' if is_feasible else 'not feasible'}",
                context={
                    'is_feasible': is_feasible,
                    'recommendations': recommendations,
                    'current_memory_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                    'available_memory_mb': psutil.virtual_memory().available / (1024 * 1024)
                } if HAS_SECURE_LOGGING else None
            )
            
            if not is_feasible:
                error_handler.handle_error(
                    error=MemoryError("Insufficient resources for batch"),
                    category=ErrorCategory.RESOURCE,
                    severity=ErrorSeverity.ERROR,
                    code=ErrorCode.INSUFFICIENT_MEMORY,
                    user_message="Insufficient system resources for this batch size.",
                    recovery_suggestions=[
                        "Process files in smaller batches",
                        "Close other applications",
                        "Disable plot generation"
                    ],
                    additional_data=recommendations
                )
                # Continue with best effort
            
            # Apply recommendations
            if recommendations.get('disable_plots'):
                self.config.processing.generate_plots = False
                self.logger.info(
                    "Disabling plots due to resource constraints",
                    context={'recommendations': recommendations} if HAS_SECURE_LOGGING else None
                )
            
            if recommendations.get('warnings'):
                for warning in recommendations['warnings']:
                    self.logger.warning(
                        f"Resource warning: {warning}",
                        context={'warning': warning} if HAS_SECURE_LOGGING else None
                    )
        
        # Check if we should enable high-performance mode
        if len(files) >= self.config.processing.disable_plots_large_batch:
            self.logger.info(f"Large batch detected ({len(files)} files), enabling optimizations")
            self._enable_high_performance_mode()
        
        return await self._process_file_batch(files, output_dir, progress_callback)

    async def _discover_files(
        self, 
        directory: Path, 
        file_filter: Optional[Callable[[Path], bool]] = None
    ) -> List[Path]:
        """Efficiently discover files to process."""
        files = []
        valid_extensions = set(self.config.processing.file_extensions)
        skip_patterns = self.config.processing.skip_patterns
        
        self.logger.info(
            "Scanning directory for files...",
            context={
                'directory': str(directory),
                'valid_extensions': list(valid_extensions),
                'skip_patterns': skip_patterns,
                'has_file_filter': file_filter is not None
            } if HAS_SECURE_LOGGING else None
        )
        
        scan_start = time.time()
        skipped_count = 0
        
        def should_skip_file(file_path: Path) -> bool:
            """Check if file should be skipped."""
            filename = file_path.name
            
            # Check skip patterns
            for pattern in skip_patterns:
                if pattern.startswith('*'):
                    if filename.endswith(pattern[1:]):
                        return True
                elif pattern.endswith('*'):
                    if filename.startswith(pattern[:-1]):
                        return True
                elif pattern in filename:
                    return True
            
            return False
        
        # Use fast directory scanning
        for file_path in directory.rglob('*'):
            if not file_path.is_file():
                continue
                
            # Check extension
            if file_path.suffix.lower() not in valid_extensions:
                skipped_count += 1
                continue
                
            # Check skip patterns
            if should_skip_file(file_path):
                skipped_count += 1
                self.logger.debug(f"Skipping file due to pattern: {file_path.name}")
                continue
                
            # Check file size
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > self.config.processing.max_file_size_mb:
                    self.logger.warning(
                        f"Skipping large file: {file_path.name} ({file_size_mb:.1f} MB)",
                        context={
                            'file_path': str(file_path),
                            'file_size_mb': file_size_mb,
                            'max_size_mb': self.config.processing.max_file_size_mb
                        } if HAS_SECURE_LOGGING else None
                    )
                    skipped_count += 1
                    continue
            except Exception:
                skipped_count += 1
                continue
            
            # Apply custom filter
            if file_filter and not file_filter(file_path):
                skipped_count += 1
                continue
                
            # Check if already processed (if enabled)
            if (self.config.processing.skip_duplicate_files and 
                self.db_manager and 
                await self._is_file_already_processed(file_path)):
                self.logger.debug(f"Skipping already processed file: {file_path.name}")
                skipped_count += 1
                continue
            
            files.append(file_path)
        
        # Sort files for consistent processing order
        files.sort(key=lambda f: f.name)
        
        # Log discovery completion
        scan_time = time.time() - scan_start
        self.logger.info(
            f"File discovery complete: found {len(files)} valid files",
            context={
                'files_found': len(files),
                'files_skipped': skipped_count,
                'scan_time_seconds': scan_time,
                'files_per_second': len(files) / scan_time if scan_time > 0 else 0
            } if HAS_SECURE_LOGGING else None
        )
        
        return files

    async def _process_file_batch(
        self,
        files: List[Path],
        output_dir: Optional[Path],
        progress_callback: Optional[Callable[[str, float, Dict], None]] = None
    ) -> Dict[str, AnalysisResult]:
        """Process files in optimized batches."""
        self._is_processing = True
        self.stats['total_files'] = len(files)
        self.stats['start_time'] = time.time()
        
        # Start performance tracking for batch
        if HAS_SECURE_LOGGING and isinstance(self.logger, SecureLogger):
            self.logger.start_performance_tracking('batch_processing')
        
        self.logger.info(
            "Starting batch processing",
            context={
                'total_files': len(files),
                'output_dir': str(output_dir) if output_dir else None,
                'batch_strategy': {
                    'chunk_size': self.config.processing.chunk_size,
                    'max_concurrent': self.config.processing.max_concurrent_files,
                    'gc_interval': self.config.processing.gc_interval
                }
            } if HAS_SECURE_LOGGING else None
        )
        
        results = {}
        failed_files = []
        
        # Start memory monitoring
        self._start_memory_monitoring()
        
        try:
            # Process in chunks to manage memory
            if self.resource_manager:
                # Get adaptive chunk size based on resources
                chunk_size = self.resource_manager.get_adaptive_batch_size(len(files), 0)
                # Respect config limits
                chunk_size = min(
                    chunk_size,
                    self.config.processing.max_concurrent_files,
                    self.config.processing.chunk_size
                )
            else:
                chunk_size = min(
                    self.config.processing.max_concurrent_files,
                    self.config.processing.chunk_size
                )
            
            for i in range(0, len(files), chunk_size):
                if self._should_stop:
                    self.logger.info("Processing stopped by user request")
                    break
                
                chunk_number = (i // chunk_size) + 1
                total_chunks = (len(files) + chunk_size - 1) // chunk_size
                
                # Log chunk start
                self.logger.debug(
                    f"Processing chunk {chunk_number}/{total_chunks}",
                    context={
                        'chunk_number': chunk_number,
                        'total_chunks': total_chunks,
                        'chunk_start_index': i,
                        'chunk_size': min(chunk_size, len(files) - i),
                        'memory_usage_mb': psutil.Process().memory_info().rss / (1024 * 1024)
                    } if HAS_SECURE_LOGGING else None
                )
                
                chunk_start_time = time.time()
                chunk_files = files[i:i + chunk_size]
                chunk_results, chunk_failed = await self._process_chunk(
                    chunk_files, output_dir, progress_callback, i, len(files)
                )
                chunk_processing_time = time.time() - chunk_start_time
                
                # Merge results
                results.update(chunk_results)
                failed_files.extend(chunk_failed)
                
                # Update stats
                self.stats['processed_files'] = len(results) + len(failed_files)
                self.stats['failed_files'] = len(failed_files)
                
                # Log chunk completion
                self.logger.info(
                    f"Completed chunk {chunk_number}/{total_chunks}",
                    context={
                        'chunk_number': chunk_number,
                        'successful': len(chunk_results),
                        'failed': len(chunk_failed),
                        'processing_time_seconds': chunk_processing_time,
                        'files_per_second': len(chunk_files) / chunk_processing_time if chunk_processing_time > 0 else 0,
                        'cumulative_success': len(results),
                        'cumulative_failed': len(failed_files)
                    } if HAS_SECURE_LOGGING else None
                )
                
                # Batch commit to database
                if (self.db_manager and 
                    len(chunk_results) >= self.config.processing.batch_commit_interval):
                    await self._batch_commit_results(chunk_results)
                
                # Memory management
                await self._perform_memory_cleanup(i + len(chunk_files))
                
                # Update progress
                if progress_callback:
                    progress = (i + len(chunk_files)) / len(files)
                    self._update_performance_stats()
                    progress_callback(
                        f"Processed {i + len(chunk_files)}/{len(files)} files",
                        progress,
                        self.stats.copy()
                    )
        
        finally:
            self._is_processing = False
            self._stop_memory_monitoring()
            
            # Final database commit
            if self.db_manager and results:
                await self._batch_commit_results(results)
        
        # Log final statistics
        self._log_final_stats(results, failed_files)
        
        return results

    async def _process_chunk(
        self,
        chunk_files: List[Path],
        output_dir: Optional[Path],
        progress_callback: Optional[Callable[[str, float, Dict], None]],
        chunk_start_idx: int,
        total_files: int
    ) -> Tuple[Dict[str, AnalysisResult], List[Tuple[str, str]]]:
        """Process a chunk of files concurrently."""
        results = {}
        failed_files = []
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.config.processing.max_concurrent_files)
        
        async def process_single_file(file_path: Path, file_idx: int) -> Optional[AnalysisResult]:
            """Process a single file with semaphore control."""
            async with semaphore:
                try:
                    def file_progress_callback(message: str, progress: float):
                        if progress_callback:
                            overall_progress = (chunk_start_idx + file_idx + progress) / total_files
                            progress_callback(
                                f"Processing {file_path.name}: {message}",
                                overall_progress,
                                self.stats.copy()
                            )
                    
                    result = await self.processor.process_file(
                        file_path=file_path,
                        output_dir=output_dir,
                        progress_callback=file_progress_callback
                    )
                    return result
                    
                except Exception as e:
                    self.logger.error(
                        f"Failed to process {file_path.name}: {e}",
                        context={
                            'file_path': str(file_path),
                            'file_size_mb': file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0,
                            'error_type': type(e).__name__,
                            'file_index': file_idx,
                            'chunk_index': chunk_start_idx + file_idx
                        } if HAS_SECURE_LOGGING else None,
                        error=e if HAS_SECURE_LOGGING else None
                    )
                    failed_files.append((str(file_path), str(e)))
                    return None
        
        # Process files concurrently
        tasks = [
            process_single_file(file_path, idx) 
            for idx, file_path in enumerate(chunk_files)
        ]
        
        # Wait for completion
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for file_path, result in zip(chunk_files, completed_results):
            if isinstance(result, Exception):
                failed_files.append((str(file_path), str(result)))
            elif result is not None:
                results[str(file_path)] = result
        
        return results, failed_files

    async def _perform_memory_cleanup(self, processed_count: int):
        """Perform memory cleanup operations."""
        current_time = time.time()
        
        # Track memory before cleanup
        before_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Use resource manager if available
        if self.resource_manager:
            status = self.resource_manager.get_current_status()
            if status.memory_warning or status.memory_critical:
                self.logger.info(
                    f"Memory pressure detected: {status.memory_percent:.1f}% used",
                    context={
                        'memory_percent': status.memory_percent,
                        'memory_warning': status.memory_warning,
                        'memory_critical': status.memory_critical,
                        'current_memory_mb': before_memory
                    } if HAS_SECURE_LOGGING else None
                )
                self.resource_manager.force_cleanup()
                
                # Check if we should pause
                if self.resource_manager.should_pause_processing():
                    self.logger.warning(
                        "Pausing processing due to memory pressure",
                        context={'memory_status': status.__dict__} if HAS_SECURE_LOGGING else None
                    )
                    if not self.resource_manager.wait_for_resources(timeout=30):
                        self.logger.error("Failed to recover resources, continuing anyway")
        
        # Regular garbage collection
        if (processed_count % self.config.processing.gc_interval == 0 or
            current_time - self._last_gc_time > 30):  # Force GC every 30 seconds
            
            self.logger.debug(
                f"Performing garbage collection at file {processed_count}",
                context={
                    'processed_count': processed_count,
                    'time_since_last_gc': current_time - self._last_gc_time,
                    'memory_before_mb': before_memory
                } if HAS_SECURE_LOGGING else None
            )
            
            gc_start = time.time()
            collected = gc.collect()
            gc_time = time.time() - gc_start
            
            after_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_freed = before_memory - after_memory
            
            self.logger.debug(
                f"Garbage collection complete",
                context={
                    'objects_collected': collected,
                    'gc_time_seconds': gc_time,
                    'memory_freed_mb': memory_freed,
                    'memory_after_mb': after_memory
                } if HAS_SECURE_LOGGING else None
            )
            
            self._last_gc_time = current_time
        
        # Clear processor cache
        if (processed_count % self.config.processing.clear_cache_interval == 0 and
            hasattr(self.processor, '_file_cache')):
            
            cache_size = len(self.processor._file_cache)
            if cache_size > 0:
                self.logger.info(
                    f"Clearing processor cache ({cache_size} entries)",
                    context={
                        'cache_size': cache_size,
                        'processed_count': processed_count,
                        'memory_before_clear': psutil.Process().memory_info().rss / (1024 * 1024)
                    } if HAS_SECURE_LOGGING else None
                )
                self.processor._file_cache.clear()
                self._last_cache_clear = processed_count

    def _enable_high_performance_mode(self):
        """Enable optimizations for large batches."""
        # Disable plot generation for performance
        original_plots = self.config.processing.generate_plots
        self.config.processing.generate_plots = False
        
        # Increase worker count if not already at max
        if self.config.processing.max_workers < 8:
            self.config.processing.max_workers = min(8, psutil.cpu_count())
        
        # Enable bulk database operations
        self.config.processing.enable_bulk_insert = True
        
        self.logger.info(f"High-performance mode enabled: plots={self.config.processing.generate_plots}, "
                        f"workers={self.config.processing.max_workers}")

    def _start_memory_monitoring(self):
        """Start monitoring memory usage with safety limits."""
        def monitor_memory():
            MAX_ITERATIONS = 10000  # Safety limit to prevent infinite loop
            iteration_count = 0
            
            while self._is_processing and iteration_count < MAX_ITERATIONS:
                iteration_count += 1
                
                try:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    self.stats['current_memory_mb'] = memory_mb
                    
                    if memory_mb > self.stats['peak_memory_mb']:
                        self.stats['peak_memory_mb'] = memory_mb
                    
                    # Check if memory_limit_mb exists in config
                    memory_limit = getattr(self.config.processing, 'memory_limit_mb', 2000)
                    
                    # Use memory safety context if available
                    if HAS_MEMORY_SAFETY:
                        validator = get_memory_validator()
                        memory_info = validator.check_memory_usage()
                        
                        if memory_info['percent'] > 90:
                            self.logger.critical(f"System memory critical: {memory_info['percent']:.1f}%")
                            validator.emergency_cleanup()
                            self._is_processing = False
                            break
                    
                    # Log high memory usage
                    if memory_mb > memory_limit * 0.8:
                        self.logger.warning(
                            f"High memory usage: {memory_mb:.1f} MB",
                            context={
                                'current_memory_mb': memory_mb,
                                'memory_limit_mb': memory_limit,
                                'memory_percent': (memory_mb / memory_limit) * 100,
                                'system_memory_percent': psutil.virtual_memory().percent,
                                'files_processed': self.stats.get('processed_files', 0)
                            } if HAS_SECURE_LOGGING else None
                        )
                        
                        # If we have resource manager, use it for better handling
                        if self.resource_manager:
                            if self.resource_manager.should_pause_processing():
                                self.logger.warning("Recommending pause due to memory pressure")
                                # Set a flag that processing functions can check
                                self._memory_pressure = True
                        
                        # Force cleanup
                        gc.collect()
                    else:
                        self._memory_pressure = False
                    
                    time.sleep(5)  # Check every 5 seconds
                    
                except AttributeError:
                    # Handle missing attributes gracefully
                    self.logger.warning("Memory monitoring: config attribute missing")
                    time.sleep(5)
                except Exception as e:
                    self.logger.error(f"Memory monitoring error: {e}")
                    time.sleep(10)  # Longer delay after error
        
        self._memory_monitor = threading.Thread(target=monitor_memory, daemon=True)
        self._memory_monitor.start()

    def _stop_memory_monitoring(self):
        """Stop memory monitoring."""
        if self._memory_monitor and self._memory_monitor.is_alive():
            # Thread will stop when _is_processing becomes False
            self._memory_monitor.join(timeout=1.0)

    def _update_performance_stats(self):
        """Update performance statistics."""
        if self.stats['start_time']:
            elapsed_time = time.time() - self.stats['start_time']
            if elapsed_time > 0:
                self.stats['files_per_second'] = self.stats['processed_files'] / elapsed_time
                
                # Estimate completion time
                remaining_files = self.stats['total_files'] - self.stats['processed_files']
                if self.stats['files_per_second'] > 0:
                    estimated_seconds = remaining_files / self.stats['files_per_second']
                    self.stats['estimated_completion'] = datetime.now().timestamp() + estimated_seconds

    async def _is_file_already_processed(self, file_path: Path) -> bool:
        """Check if file has already been processed."""
        if not self.db_manager:
            return False
        
        try:
            # Check database for existing results
            # This would need to be implemented in DatabaseManager
            return False  # Placeholder
        except Exception:
            return False

    async def _batch_commit_results(self, results: Dict[str, AnalysisResult]):
        """Commit results to database in batches."""
        if not self.db_manager or not results:
            return
        
        commit_start_time = time.time()
        
        # Log batch commit start
        self.logger.debug(
            "Starting database batch commit",
            context={
                'batch_size': len(results),
                'has_batch_save': hasattr(self.db_manager, 'save_analysis_batch')
            } if HAS_SECURE_LOGGING else None
        )
        
        try:
            # Convert to batch format for database
            batch_data = []
            for file_path, result in results.items():
                batch_data.append(result)
            
            # Use batch save if available
            if hasattr(self.db_manager, 'save_analysis_batch'):
                self.logger.info(
                    f"Committing {len(batch_data)} results to database using batch save",
                    context={
                        'batch_size': len(batch_data),
                        'memory_usage_mb': psutil.Process().memory_info().rss / (1024 * 1024)
                    } if HAS_SECURE_LOGGING else None
                )
                
                save_start = time.time()
                analysis_ids = self.db_manager.save_analysis_batch(batch_data)
                save_time = time.time() - save_start
                
                # Update results with database IDs
                for result, db_id in zip(batch_data, analysis_ids):
                    result.db_id = db_id
                    
                self.logger.info(
                    f"Successfully committed {len(analysis_ids)} results to database",
                    context={
                        'saved_count': len(analysis_ids),
                        'save_time_seconds': save_time,
                        'saves_per_second': len(analysis_ids) / save_time if save_time > 0 else 0
                    } if HAS_SECURE_LOGGING else None
                )
            else:
                # Fall back to individual saves
                self.logger.info(
                    f"Batch save not available, using individual saves for {len(batch_data)} results"
                )
                saved_count = 0
                duplicate_count = 0
                
                for idx, result in enumerate(batch_data):
                    try:
                        # Check for duplicates
                        existing_id = self.db_manager.check_duplicate_analysis(
                            result.metadata.model,
                            result.metadata.serial,
                            result.metadata.file_date
                        )
                        
                        if existing_id:
                            result.db_id = existing_id
                            duplicate_count += 1
                        else:
                            result.db_id = self.db_manager.save_analysis(result)
                            saved_count += 1
                            
                        # Log progress for large batches
                        if idx > 0 and idx % 100 == 0:
                            self.logger.debug(
                                f"Individual save progress: {idx}/{len(batch_data)}",
                                context={
                                    'progress': idx,
                                    'total': len(batch_data),
                                    'saved': saved_count,
                                    'duplicates': duplicate_count
                                } if HAS_SECURE_LOGGING else None
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Failed to save result: {e}",
                            context={
                                'result_index': idx,
                                'model': result.metadata.model,
                                'serial': result.metadata.serial
                            } if HAS_SECURE_LOGGING else None
                        )
                
                self.logger.info(
                    f"Individually saved {saved_count} results to database",
                    context={
                        'saved_count': saved_count,
                        'duplicate_count': duplicate_count,
                        'total_time_seconds': time.time() - commit_start_time
                    } if HAS_SECURE_LOGGING else None
                )
            
        except Exception as e:
            self.logger.error(
                f"Database batch commit failed: {e}",
                context={
                    'error_type': type(e).__name__,
                    'batch_size': len(results)
                } if HAS_SECURE_LOGGING else None,
                error=e if HAS_SECURE_LOGGING else None
            )

    def _filter_resume_files(self, files: List[Path], resume_from: str) -> List[Path]:
        """Filter files for resume functionality."""
        resume_index = None
        for i, file_path in enumerate(files):
            if file_path.name == resume_from or str(file_path) == resume_from:
                resume_index = i
                break
        
        if resume_index is not None:
            return files[resume_index:]
        else:
            self.logger.warning(f"Resume file {resume_from} not found, processing all files")
            return files

    def _log_final_stats(self, results: Dict, failed_files: List):
        """Log final processing statistics."""
        total_time = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        # End performance tracking if using secure logging
        if HAS_SECURE_LOGGING and isinstance(self.logger, SecureLogger):
            batch_duration = self.logger.end_performance_tracking('batch_processing')
            dir_duration = self.logger.end_performance_tracking('process_large_directory')
        else:
            batch_duration = dir_duration = total_time
        
        # Calculate detailed statistics
        total_processed = len(results) + len(failed_files)
        success_rate = (len(results) / total_processed * 100) if total_processed > 0 else 0
        
        # Log summary
        self.logger.info("="*60)
        self.logger.info("LARGE-SCALE PROCESSING COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Total files processed: {total_processed}")
        self.logger.info(f"Successful: {len(results)}")
        self.logger.info(f"Failed: {len(failed_files)}")
        self.logger.info(f"Success rate: {success_rate:.1f}%")
        self.logger.info(f"Total processing time: {total_time/60:.1f} minutes")
        self.logger.info(f"Average files per second: {self.stats['files_per_second']:.2f}")
        self.logger.info(f"Peak memory usage: {self.stats['peak_memory_mb']:.1f} MB")
        self.logger.info("="*60)
        
        # Log detailed statistics with secure logging
        if HAS_SECURE_LOGGING:
            self.logger.info(
                "Batch processing summary",
                context={
                    'total_files': total_processed,
                    'successful': len(results),
                    'failed': len(failed_files),
                    'success_rate_percent': success_rate,
                    'processing_time': {
                        'total_seconds': total_time,
                        'total_minutes': total_time / 60,
                        'batch_processing_seconds': batch_duration,
                        'directory_processing_seconds': dir_duration
                    },
                    'performance': {
                        'files_per_second': self.stats['files_per_second'],
                        'files_per_minute': self.stats['files_per_second'] * 60 if self.stats['files_per_second'] else 0
                    },
                    'memory': {
                        'peak_mb': self.stats['peak_memory_mb'],
                        'current_mb': self.stats['current_memory_mb'],
                        'system_available_mb': psutil.virtual_memory().available / (1024 * 1024)
                    },
                    'configuration': {
                        'chunk_size': self.config.processing.chunk_size,
                        'max_concurrent': self.config.processing.max_concurrent_files,
                        'plots_enabled': self.config.processing.generate_plots,
                        'database_enabled': self.db_manager is not None
                    }
                },
                performance={
                    'total_processing_time_ms': total_time * 1000,
                    'avg_file_time_ms': (total_time / total_processed * 1000) if total_processed > 0 else 0
                }
            )
            
            # Log failed files summary if any
            if failed_files:
                self.logger.warning(
                    f"Failed to process {len(failed_files)} files",
                    context={
                        'failed_count': len(failed_files),
                        'failed_files': [str(f[0]) for f in failed_files[:10]],  # First 10
                        'error_types': {}  # Could analyze error types here
                    }
                )

    def stop_processing(self):
        """Stop processing gracefully."""
        self.logger.info("Stopping large-scale processing...")
        self._should_stop = True

    def get_stats(self) -> Dict:
        """Get current processing statistics."""
        return self.stats.copy()


# Utility function for easy access
async def process_large_directory(
    directory: Path,
    config: Optional[Config] = None,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, AnalysisResult]:
    """
    Convenience function for processing large directories.
    
    Args:
        directory: Directory to process
        config: Configuration (uses default if None)
        output_dir: Output directory
        progress_callback: Progress callback function
        
    Returns:
        Processing results
    """
    if config is None:
        from laser_trim_analyzer.core.config import get_config
        config = get_config()
    
    # Initialize database if enabled
    db_manager = None
    if config.database.enabled:
        try:
            from laser_trim_analyzer.database.manager import DatabaseManager
            db_path = f"sqlite:///{config.database.path.absolute()}"
            db_manager = DatabaseManager(db_path)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Database initialization failed: {e}")
    
    # Create processor and run
    processor = LargeScaleProcessor(config, db_manager)
    return await processor.process_large_directory(
        directory, output_dir, progress_callback
    ) 
