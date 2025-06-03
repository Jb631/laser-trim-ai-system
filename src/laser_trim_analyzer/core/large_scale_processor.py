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
        self.logger.info(f"Starting large-scale processing of directory: {directory}")
        
        # Discover files
        files = await self._discover_files(directory, file_filter)
        self.logger.info(f"Discovered {len(files)} files for processing")
        
        if not files:
            self.logger.warning("No files found to process")
            return {}
        
        # Handle resume functionality
        if resume_from:
            files = self._filter_resume_files(files, resume_from)
            self.logger.info(f"Resuming from {resume_from}, {len(files)} files remaining")
        
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
        
        self.logger.info("Scanning directory for files...")
        
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
                continue
                
            # Check skip patterns
            if should_skip_file(file_path):
                continue
                
            # Check file size
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > self.config.processing.max_file_size_mb:
                    self.logger.warning(f"Skipping large file: {file_path.name} ({file_size_mb:.1f} MB)")
                    continue
            except Exception:
                continue
            
            # Apply custom filter
            if file_filter and not file_filter(file_path):
                continue
                
            # Check if already processed (if enabled)
            if (self.config.processing.skip_duplicate_files and 
                self.db_manager and 
                await self._is_file_already_processed(file_path)):
                self.logger.debug(f"Skipping already processed file: {file_path.name}")
                continue
            
            files.append(file_path)
        
        # Sort files for consistent processing order
        files.sort(key=lambda f: f.name)
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
        
        results = {}
        failed_files = []
        
        # Start memory monitoring
        self._start_memory_monitoring()
        
        try:
            # Process in chunks to manage memory
            chunk_size = min(
                self.config.processing.max_concurrent_files,
                self.config.processing.chunk_size
            )
            
            for i in range(0, len(files), chunk_size):
                if self._should_stop:
                    self.logger.info("Processing stopped by user request")
                    break
                
                chunk_files = files[i:i + chunk_size]
                chunk_results, chunk_failed = await self._process_chunk(
                    chunk_files, output_dir, progress_callback, i, len(files)
                )
                
                # Merge results
                results.update(chunk_results)
                failed_files.extend(chunk_failed)
                
                # Update stats
                self.stats['processed_files'] = len(results) + len(failed_files)
                self.stats['failed_files'] = len(failed_files)
                
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
                    self.logger.error(f"Failed to process {file_path.name}: {e}")
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
        
        # Garbage collection
        if (processed_count % self.config.processing.gc_interval == 0 or
            current_time - self._last_gc_time > 30):  # Force GC every 30 seconds
            
            self.logger.debug(f"Performing garbage collection at file {processed_count}")
            gc.collect()
            self._last_gc_time = current_time
        
        # Clear processor cache
        if (processed_count % self.config.processing.clear_cache_interval == 0 and
            hasattr(self.processor, '_file_cache')):
            
            cache_size = len(self.processor._file_cache)
            if cache_size > 0:
                self.logger.info(f"Clearing processor cache ({cache_size} entries)")
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
        """Start monitoring memory usage."""
        def monitor_memory():
            while self._is_processing:
                try:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    self.stats['current_memory_mb'] = memory_mb
                    
                    if memory_mb > self.stats['peak_memory_mb']:
                        self.stats['peak_memory_mb'] = memory_mb
                    
                    # Log high memory usage
                    if memory_mb > self.config.processing.memory_limit_mb * 0.8:
                        self.logger.warning(f"High memory usage: {memory_mb:.1f} MB")
                    
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"Memory monitoring error: {e}")
                    break
        
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
        
        try:
            # Convert to batch format for database
            batch_data = []
            for file_path, result in results.items():
                batch_data.append(result)
            
            # Batch insert to database
            # This would need to be implemented in DatabaseManager
            self.logger.info(f"Committing {len(batch_data)} results to database")
            
        except Exception as e:
            self.logger.error(f"Database batch commit failed: {e}")

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
        
        self.logger.info("="*60)
        self.logger.info("LARGE-SCALE PROCESSING COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Total files processed: {len(results) + len(failed_files)}")
        self.logger.info(f"Successful: {len(results)}")
        self.logger.info(f"Failed: {len(failed_files)}")
        self.logger.info(f"Success rate: {len(results)/(len(results)+len(failed_files))*100:.1f}%")
        self.logger.info(f"Total processing time: {total_time/60:.1f} minutes")
        self.logger.info(f"Average files per second: {self.stats['files_per_second']:.2f}")
        self.logger.info(f"Peak memory usage: {self.stats['peak_memory_mb']:.1f} MB")
        self.logger.info("="*60)

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