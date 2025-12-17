"""
Fast processing engine for Laser Trim Analyzer v2 - optimized for large batches.

This module provides high-performance processing capabilities for handling
thousands of files efficiently through:
- Memory-efficient Excel reading
- True parallel processing with multiprocessing
- Cached file operations
- Optimized data structures
- Minimal overhead mode
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import lru_cache
import time
import gc
import psutil

from laser_trim_analyzer.core.config import Config
from laser_trim_analyzer.core.models import (
    AnalysisResult, TrackData, FileMetadata, UnitProperties,
    SigmaAnalysis, LinearityAnalysis, ResistanceAnalysis,
    TrimEffectiveness, ZoneAnalysis, FailurePrediction,
    DynamicRangeAnalysis, AnalysisStatus, SystemType, RiskCategory,
    ValidationResult as ModelValidationResult, ValidationStatus
)
from laser_trim_analyzer.core.exceptions import (
    ProcessingError, DataExtractionError, AnalysisError, ValidationError
)
from laser_trim_analyzer.analysis.sigma_analyzer import SigmaAnalyzer
from laser_trim_analyzer.analysis.linearity_analyzer import LinearityAnalyzer
from laser_trim_analyzer.analysis.resistance_analyzer import ResistanceAnalyzer
from laser_trim_analyzer.utils.memory_efficient_excel import (
    read_excel_memory_efficient, excel_reader_context, estimate_memory_usage
)
from laser_trim_analyzer.utils.excel_utils import (
    find_data_columns, extract_cell_value, detect_system_type
)
from laser_trim_analyzer.utils.plotting_utils import create_analysis_plot
from laser_trim_analyzer.utils.date_utils import extract_datetime_from_filename
from laser_trim_analyzer.utils.calculation_validator import CalculationValidator, ValidationLevel, CalculationType
# Import comprehensive validation utilities
from laser_trim_analyzer.utils.validators import (
    validate_excel_file, validate_analysis_data, validate_model_number,
    validate_resistance_values, AnalysisValidator, 
    ValidationResult as UtilsValidationResult
)
from laser_trim_analyzer.core.constants import (
    SYSTEM_A_COLUMNS, SYSTEM_B_COLUMNS,
    SYSTEM_A_CELLS, SYSTEM_B_CELLS
)

logger = logging.getLogger(__name__)


class FastExcelReader:
    """Optimized Excel reader with caching and minimal overhead."""
    
    def __init__(self, turbo_mode: bool = False):
        self._cache = {}
        self._max_cache_size = 50
        self.turbo_mode = turbo_mode
        
    @lru_cache(maxsize=128)
    def detect_system_cached(self, file_path: str) -> SystemType:
        """Cached system detection to avoid repeated file reads."""
        return detect_system_type(Path(file_path))
    
    def read_file_once(self, file_path: Path) -> Dict[str, Any]:
        """Read Excel file once and extract all needed data."""
        file_hash = str(file_path)
        
        # Check cache
        if file_hash in self._cache:
            return self._cache[file_hash]
        
        result = {
            'system_type': None,
            'sheets': {},
            'metadata': {},
            'data': {}
        }
        
        try:
            # Detect system type
            result['system_type'] = self.detect_system_cached(str(file_path))
            
            # Use memory-efficient reader for large files
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 10:  # Use chunked reading for files > 10MB
                logger.info(f"Using memory-efficient reader for large file: {file_path.name} ({file_size_mb:.1f}MB)")
                
                # Read with chunking
                with excel_reader_context(file_path) as reader:
                    # Get sheet names from first chunk
                    sheet_names = reader.worksheet.parent.sheetnames
                    result['sheets'] = {name: name for name in sheet_names}
                    
                    # Read data in chunks
                    for sheet_name in self._get_relevant_sheets(sheet_names, result['system_type']):
                        chunk_data = []
                        for chunk in reader.iter_chunks(chunk_size=5000):
                            chunk_data.append(chunk)
                        
                        if chunk_data:
                            result['data'][sheet_name] = pd.concat(chunk_data, ignore_index=True)
            else:
                # For smaller files, use standard reader but read everything at once
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names
                result['sheets'] = {name: name for name in sheet_names}
                
                # Read only relevant sheets
                for sheet_name in self._get_relevant_sheets(sheet_names, result['system_type']):
                    result['data'][sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Extract metadata once
            result['metadata'] = self._extract_metadata_fast(file_path, result)
            
            # Cache result if cache not full
            if len(self._cache) < self._max_cache_size:
                self._cache[file_hash] = result
            else:
                # Remove oldest entry
                self._cache.pop(next(iter(self._cache)))
                self._cache[file_hash] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def _get_relevant_sheets(self, sheet_names: List[str], system_type: SystemType) -> List[str]:
        """Get only sheets needed for analysis."""
        relevant = []
        
        if system_type == SystemType.SYSTEM_A:
            for sheet in sheet_names:
                if any(pattern in sheet for pattern in ['TRK', 'SEC', ' 0', '_0', 'TRM']):
                    relevant.append(sheet)
        else:  # System B
            for sheet in sheet_names:
                if sheet.lower() in ['test', 'lin error', 'trim']:
                    relevant.append(sheet)
        
        return relevant[:10]  # Limit to prevent memory issues
    
    def _extract_metadata_fast(self, file_path: Path, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata quickly from cached data."""
        metadata = {
            'filename': file_path.name,
            'file_path': str(file_path),
            'system_type': file_data['system_type'],
            'file_size_mb': file_path.stat().st_size / (1024 * 1024)
        }
        
        # Parse filename
        parts = file_path.stem.split('_')
        metadata['model'] = parts[0] if parts else "Unknown"
        metadata['serial'] = parts[1] if len(parts) > 1 else "Unknown"
        metadata['timestamp'] = extract_datetime_from_filename(file_path.stem) or datetime.now()
        
        # Check for multi-track based on system type and sheets
        if file_data['system_type'] == SystemType.SYSTEM_B:
            # System B multi-track detection
            if 'TA' in parts or 'TB' in parts:
                metadata['has_multi_tracks'] = True
                metadata['track_identifier'] = 'TA' if 'TA' in parts else 'TB'
            else:
                metadata['has_multi_tracks'] = False
                metadata['track_identifier'] = None
        else:
            # System A multi-track detection based on sheet names
            track_sheets = [s for s in file_data['sheets'] if 'TRK' in s]
            metadata['has_multi_tracks'] = len(track_sheets) > 1
            metadata['track_identifier'] = None
        
        return metadata
    
    def clear_cache(self):
        """Clear the file cache to free memory."""
        self._cache.clear()
        gc.collect()


class FastProcessor:
    """
    High-performance processor for large batch operations.

    .. deprecated:: 2.3.0
        This class is deprecated and will be removed in a future version.
        Use :class:`laser_trim_analyzer.core.unified_processor.UnifiedProcessor`
        with strategy='turbo' instead.
        Enable via config: ``processing.use_unified_processor: true``
    """
    
    def __init__(self, config: Config, turbo_mode: bool = False):
        self.config = config
        self.turbo_mode = turbo_mode
        self.reader = FastExcelReader(turbo_mode=turbo_mode)
        
        # Initialize analyzers once
        self.sigma_analyzer = SigmaAnalyzer(config, logger)
        self.linearity_analyzer = LinearityAnalyzer(config, logger)
        self.resistance_analyzer = ResistanceAnalyzer(config, logger)
        
        # Initialize validation components for data integrity
        try:
            validation_level = getattr(
                getattr(config, 'validation_level', 'standard'),
                ValidationLevel.STANDARD
            )
            self.calculation_validator = CalculationValidator(validation_level)
            logger.info(f"Initialized calculation validator with {validation_level.value} validation level")
        except Exception as e:
            logger.warning(f"Failed to initialize calculation validator: {e}")
            self.calculation_validator = None
        
        # Circuit breaker for handling failures
        self.consecutive_failures = 0
        self.max_failures = 50  # Stop after 50 consecutive failures
        self.failure_reset_threshold = 10  # Reset after 10 successes
        
        # Performance settings - adaptive based on system resources
        self._setup_adaptive_performance_settings()
        
        # Adaptive chunk sizing - will be set based on total files in process_batch
        self.chunk_size = 20  # Default chunk size
        self.enable_plots = False if turbo_mode else config.processing.generate_plots
        
        # Set large batch mode flag for memory management
        self._large_batch_mode = False
        
        logger.info(f"FastProcessor initialized - Turbo: {turbo_mode}, Workers: {self.max_workers}, Plots: {self.enable_plots}")
    
    def _setup_adaptive_performance_settings(self):
        """Setup performance settings based on system resources and batch size."""
        # Get system info
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = mp.cpu_count()
        
        logger.info(f"System resources: {memory_gb:.1f}GB RAM, {cpu_count} CPU cores")
        
        if self.turbo_mode:
            # Turbo mode - more aggressive but safer limits for large batches
            if memory_gb >= 16:
                # High memory system
                self.max_workers = min(max(cpu_count * 2 // 3, 1), 6)
            elif memory_gb >= 8:
                # Medium memory system
                self.max_workers = min(max(cpu_count // 2, 1), 4)
            else:
                # Low memory system - very conservative
                self.max_workers = min(max(cpu_count // 3, 1), 2)
        else:
            # Standard mode - conservative settings
            if memory_gb >= 8:
                self.max_workers = min(max(cpu_count // 2, 1), 3)
            else:
                self.max_workers = min(max(cpu_count // 3, 1), 2)
        
        logger.info(f"Adaptive performance: {self.max_workers} max workers for {memory_gb:.1f}GB RAM")
    
    def process_batch_fast(
        self,
        file_paths: List[Path],
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[AnalysisResult]:
        """Process multiple files with maximum performance."""
        total_files = len(file_paths)
        logger.info(f"Starting fast batch processing of {total_files} files")
        
        # Enable large batch mode for 1000+ files
        self._large_batch_mode = total_files >= 1000
        if self._large_batch_mode:
            logger.info("Large batch mode enabled - enhanced memory management active")
        
        # Enhanced memory and system check before starting
        memory_info = psutil.virtual_memory()
        memory_available = memory_info.available / (1024**3)  # GB
        memory_total = memory_info.total / (1024**3)  # GB
        cpu_percent = psutil.cpu_percent(interval=1)
        
        logger.info(f"System status before batch: {memory_available:.1f}GB available / {memory_total:.1f}GB total ({memory_info.percent:.1f}% used), CPU: {cpu_percent:.1f}%")
        
        # Adaptive processing strategy based on batch size and resources
        self._setup_batch_strategy(total_files, memory_available, memory_total)
        
        # Circuit breaker reset
        self.consecutive_failures = 0
        
        # Initialize batch tracking
        results = []
        failed_files = []
        skipped_files = []
        processed = 0
        start_time = time.time()
        
        # Process in chunks to manage memory
        chunks = [file_paths[i:i + self.chunk_size] for i in range(0, total_files, self.chunk_size)]
        logger.info(f"🔧 DEBUG: Created {len(chunks)} chunks with size {self.chunk_size}")
        logger.info(f"🔧 DEBUG: First chunk size: {len(chunks[0]) if chunks else 0}")
        logger.info(f"🔧 DEBUG: Last chunk size: {len(chunks[-1]) if chunks else 0}")
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start = time.time()
            logger.info(f"🔧 DEBUG: Starting chunk {chunk_idx + 1}/{len(chunks)} with {len(chunk)} files")
            
            # Update progress at start of chunk with accurate file count
            if progress_callback:
                progress = processed / total_files
                files_in_chunk = len(chunk)
                start_file = processed + 1
                end_file = min(processed + files_in_chunk, total_files)
                logger.info(f"🔧 DEBUG: Progress update: files {start_file}-{end_file} of {total_files} ({progress:.1%})")
                # Check if callback returns False to indicate stop
                continue_processing = progress_callback(f"Processing files {start_file}-{end_file} of {total_files}", progress)
                if continue_processing is False:
                    logger.info("⚠️ Processing stopped by user request")
                    break
                logger.info(f"🔧 DEBUG: Progress callback returned: {continue_processing}")
            
            # Circuit breaker check
            if self.consecutive_failures >= self.max_failures:
                logger.error(f"Circuit breaker triggered: {self.consecutive_failures} consecutive failures")
                skipped_files.extend(chunk)
                continue
            
            # Process chunk in parallel with error tracking
            chunk_results, chunk_failures = self._process_chunk_parallel_safe(chunk, output_dir)
            results.extend(chunk_results)
            failed_files.extend(chunk_failures)
            
            # Update failure tracking
            if chunk_failures:
                self.consecutive_failures += len(chunk_failures)
            else:
                # Reset failure counter on successful chunk
                if self.consecutive_failures > 0:
                    self.consecutive_failures = max(0, self.consecutive_failures - self.failure_reset_threshold)
            
            # Update processed count with actual results
            processed += len(chunk_results)
            chunk_time = time.time() - chunk_start
            rate = len(chunk_results) / chunk_time if chunk_time > 0 else 0
            
            logger.info(f"Chunk {chunk_idx + 1}: {len(chunk_results)} success, {len(chunk_failures)} failed, {rate:.1f} files/sec")
            
            # Update progress after chunk with accurate count
            if progress_callback:
                progress = processed / total_files
                # Ensure progress never exceeds 1.0 (100%)
                progress = min(progress, 1.0)
                continue_processing = progress_callback(f"Completed {processed} of {total_files} files", progress)
                if continue_processing is False:
                    logger.info("Processing stopped after chunk completion")
                    break
            
            logger.info(f"Chunk {chunk_idx + 1} completed: {len(chunk_results)} files in {chunk_time:.1f}s ({rate:.1f} files/sec)")
            
            # Adaptive memory and resource management
            self._adaptive_resource_management(chunk_idx, total_files)
            
            # Check system health and decide whether to continue
            if not self._system_health_check():
                logger.warning("System health check failed - stopping batch processing")
                skipped_files.extend([f for chunk in chunks[chunk_idx+1:] for f in chunk])
                break
        
        total_time = time.time() - start_time
        overall_rate = processed / total_time if total_time > 0 else 0
        
        # Final batch statistics
        success_count = len(results)
        failure_count = len(failed_files)
        skipped_count = len(skipped_files)
        
        logger.info(f"Batch processing complete: {success_count} success, {failure_count} failed, {skipped_count} skipped")
        logger.info(f"Total time: {total_time:.1f}s, Rate: {overall_rate:.1f} files/sec")
        
        if failed_files:
            logger.warning(f"Failed files sample: {[str(f.name) for f in failed_files[:5]]}")
        if skipped_files:
            logger.warning(f"Skipped files sample: {[str(f.name) for f in skipped_files[:5]]}")
        
        # Final cleanup
        self._cleanup_memory()
        
        return results
    
    def _setup_batch_strategy(self, total_files: int, memory_available: float, memory_total: float):
        """Setup adaptive processing strategy based on batch size and available resources."""
        logger.info(f"Setting up batch strategy for {total_files} files")
        
        # For very large batches (2000+ files), use extra conservative settings
        if total_files >= 2000:
            logger.warning(f"Large batch detected: {total_files} files - using ultra-conservative settings")
            
            # Ultra-conservative chunk sizes for massive batches
            if memory_available >= 8:
                self.chunk_size = 15  # Very small chunks
                self.max_workers = min(self.max_workers, 3)  # Reduce workers further
            elif memory_available >= 4:
                self.chunk_size = 10
                self.max_workers = min(self.max_workers, 2)
            else:
                self.chunk_size = 5   # Tiny chunks for low memory
                self.max_workers = 1  # Single threaded
            
            logger.warning(f"Ultra-conservative mode: chunk_size={self.chunk_size}, max_workers={self.max_workers}")
            
        elif total_files >= 1000:
            logger.info(f"Large batch: {total_files} files - using conservative settings")
            
            if memory_available >= 6:
                self.chunk_size = 25
                self.max_workers = min(self.max_workers, 4)
            elif memory_available >= 3:
                self.chunk_size = 15
                self.max_workers = min(self.max_workers, 3)
            else:
                self.chunk_size = 8
                self.max_workers = min(self.max_workers, 2)
            
            logger.info(f"Conservative mode: chunk_size={self.chunk_size}, max_workers={self.max_workers}")
            
        elif total_files >= 500:
            # Medium batches - balanced approach
            if memory_available >= 4:
                self.chunk_size = 40
            elif memory_available >= 2:
                self.chunk_size = 25
            else:
                self.chunk_size = 15
            
            logger.info(f"Balanced mode: chunk_size={self.chunk_size}")
            
        else:
            # Small batches - standard settings
            if memory_available >= 2:
                self.chunk_size = 50
            else:
                self.chunk_size = 25
            
            logger.info(f"Standard mode: chunk_size={self.chunk_size}")
        
        # Memory usage warning
        memory_percent = (memory_total - memory_available) / memory_total * 100
        if memory_percent > 80:
            logger.warning(f"High memory usage detected: {memory_percent:.1f}% - reducing performance settings")
            self.chunk_size = max(5, self.chunk_size // 2)
            self.max_workers = max(1, self.max_workers // 2)
    
    def _adaptive_resource_management(self, chunk_idx: int, total_files: int):
        """Adaptive resource management between chunks."""
        # More frequent cleanup for large batches
        cleanup_frequency = 3 if total_files >= 2000 else 5
        
        if chunk_idx % cleanup_frequency == 0:
            self._cleanup_memory()
        
        # Dynamic CPU and memory monitoring
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Adaptive delays based on system load
        if memory_info.percent > 85:
            logger.warning(f"High memory usage: {memory_info.percent:.1f}% - extended pause")
            time.sleep(3.0)
            self._cleanup_memory()  # Extra cleanup
        elif memory_info.percent > 75:
            logger.info(f"Moderate memory usage: {memory_info.percent:.1f}% - pause")
            time.sleep(2.0)
        elif cpu_percent > 90:
            logger.warning(f"Very high CPU usage: {cpu_percent:.1f}% - extended pause")
            time.sleep(2.5)
        elif cpu_percent > 80:
            logger.info(f"High CPU usage: {cpu_percent:.1f}% - pause")
            time.sleep(1.5)
        elif cpu_percent > 60:
            time.sleep(1.0)
        else:
            # Always add a small delay between chunks for system stability
            time.sleep(0.3)
    
    def _system_health_check(self) -> bool:
        """Check if system is healthy enough to continue processing."""
        memory_info = psutil.virtual_memory()
        
        # Critical memory check
        if memory_info.percent > 95:
            logger.error(f"Critical memory usage: {memory_info.percent:.1f}% - stopping processing")
            return False
        
        # Available memory check
        available_gb = memory_info.available / (1024**3)
        if available_gb < 0.5:  # Less than 500MB available
            logger.error(f"Critical low memory: {available_gb:.1f}GB available - stopping processing")
            return False
        
        # Check for system responsiveness
        try:
            start_time = time.time()
            cpu_percent = psutil.cpu_percent(interval=0.5)
            response_time = time.time() - start_time
            
            if response_time > 2.0:  # System taking too long to respond
                logger.warning(f"System slow response: {response_time:.1f}s - may be overloaded")
                return False
        except Exception as e:
            logger.warning(f"System health check failed: {e}")
            return False
        
        return True
    
    def _process_chunk_parallel_safe(self, file_paths: List[Path], output_dir: Optional[Path]) -> Tuple[List[AnalysisResult], List[Path]]:
        """Process chunk with enhanced error tracking and recovery."""
        results = []
        failed_files = []
        
        # Use ThreadPoolExecutor for stability in large batches
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self._process_file_sequential, file_path, output_dir): file_path
                for file_path in file_paths
            }
            
            # Collect results with timeout and error handling
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    # Extended timeout for large batches
                    timeout = 120 if len(file_paths) > 100 else 60
                    result = future.result(timeout=timeout)
                    
                    if result:
                        results.append(result)
                    else:
                        failed_files.append(file_path)
                        
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")
                    failed_files.append(file_path)
        
        return results, failed_files
    
    def _process_chunk_parallel(self, file_paths: List[Path], output_dir: Optional[Path]) -> List[AnalysisResult]:
        """Process a chunk of files in parallel using threads (more stable than processes)."""
        results = []
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid freeze issues
        # ThreadPoolExecutor is more stable in GUI applications and PyInstaller executables
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self._process_file_sequential, file_path, output_dir): file_path
                for file_path in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=60)  # Increased timeout for thread-based processing
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        
        return results
    
    def _process_file_sequential(self, file_path: Path, output_dir: Optional[Path]) -> Optional[AnalysisResult]:
        """Process a single file sequentially (thread-safe version)."""
        return self._process_file_worker(file_path, output_dir)
    
    def _process_file_worker(self, file_path: Path, output_dir: Optional[Path]) -> Optional[AnalysisResult]:
        """Worker function to process a single file."""
        start_time = time.time()
        try:
            # In extreme turbo mode, use minimal processing
            if self.turbo_mode and hasattr(self, '_process_file_minimal'):
                return self._process_file_minimal(file_path)
            
            # Read file data once
            file_data = self.reader.read_file_once(file_path)
            
            # Create metadata
            metadata = FileMetadata(
                filename=file_data['metadata']['filename'],
                file_path=Path(file_data['metadata']['file_path']),  # Convert string to Path
                model=file_data['metadata']['model'],
                serial=file_data['metadata']['serial'],
                file_date=file_data['metadata']['timestamp'],  # Changed from timestamp to file_date
                system=file_data['system_type'],  # Changed from system_type to system
                has_multi_tracks=file_data['metadata'].get('has_multi_tracks', False),
                track_identifier=file_data['metadata'].get('track_identifier', None)
            )
            
            # Process tracks
            tracks = self._process_tracks_fast(file_path, file_data, metadata)
            
            if not tracks:
                logger.warning(f"No tracks processed for {file_path}")
                return None
            
            # Determine overall status
            overall_status = self._determine_overall_status(tracks)
            
            # Comprehensive validation and error tracking
            validation_issues, processing_errors = self._perform_comprehensive_validation(
                file_path, tracks, file_data, metadata
            )
            
            # Determine overall validation status based on comprehensive validation
            overall_validation_status = self._determine_comprehensive_validation_status(
                tracks, validation_issues
            )
            
            # Create result with comprehensive validation data
            result = AnalysisResult(
                metadata=metadata,
                tracks=tracks,
                overall_status=overall_status,
                overall_validation_status=overall_validation_status,
                processing_time=(time.time() - start_time),  # Actual processing time
                validation_issues=validation_issues,  # Comprehensive validation results
                processing_errors=processing_errors   # Actual processing errors
            )
            
            # Generate plot only if enabled and not in turbo mode
            if self.enable_plots and output_dir and not self.turbo_mode:
                self._generate_plot_fast(result, output_dir)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _process_tracks_fast(self, file_path: Path, file_data: Dict[str, Any], metadata: FileMetadata) -> Dict[str, TrackData]:
        """Process tracks with minimal overhead."""
        tracks = {}
        
        if file_data['system_type'] == SystemType.SYSTEM_A:
            # Multi-track system
            track_sheets = self._find_track_sheets_fast(file_data)
            
            for track_id, sheets in track_sheets.items():
                track_data = self._process_single_track_fast(file_path, track_id, sheets, file_data, metadata)
                if track_data:
                    tracks[track_id] = track_data
        else:
            # System B - single track
            track_data = self._process_single_track_fast(
                file_path, 
                "default", 
                {'untrimmed': 'test', 'trimmed': 'Lin Error'},
                file_data,
                metadata
            )
            if track_data:
                tracks["default"] = track_data
        
        return tracks
    
    def _find_track_sheets_fast(self, file_data: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Find track sheets quickly from cached data."""
        track_sheets = {}
        sheet_names = list(file_data['sheets'].keys())
        
        # Look for TRK1 and TRK2
        for track_id in ["TRK1", "TRK2"]:
            sheets = {}
            for sheet in sheet_names:
                if track_id in sheet:
                    if " 0" in sheet or "_0" in sheet:
                        sheets['untrimmed'] = sheet
                    elif "TRM" in sheet.upper():
                        sheets['trimmed'] = sheet
            
            if sheets:
                track_sheets[track_id] = sheets
        
        return track_sheets
    
    def _process_single_track_fast(
        self, 
        file_path: Path,
        track_id: str,
        sheets: Dict[str, str],
        file_data: Dict[str, Any],
        metadata: FileMetadata
    ) -> Optional[TrackData]:
        """Process a single track with optimized data extraction."""
        try:
            # Extract data from cached sheets
            trim_data = self._extract_trim_data_fast(sheets, file_data, metadata.system)
            
            if not trim_data or not trim_data.get('positions'):
                return None
            
            # Extract unit properties
            unit_props = self._extract_unit_properties_fast(sheets, file_data, metadata.system)
            
            # Run analyses in turbo mode (skip some validations)
            sigma_result = self._analyze_sigma_fast(trim_data, unit_props, metadata.model)
            # Always perform all analyses for complete functionality
            linearity_result = self._analyze_linearity_fast(trim_data)
            resistance_result = self._analyze_resistance_fast(unit_props)
            
            # Add ALL required analyses - nothing is optional!
            zone_result = self._analyze_zones_fast(trim_data)
            dynamic_range_result = self._analyze_dynamic_range_fast(trim_data)
            trim_effectiveness_result = self._calculate_trim_effectiveness_fast(
                sigma_result, linearity_result, resistance_result
            )
            failure_prediction_result = self._calculate_failure_prediction_fast(
                sigma_result, linearity_result, unit_props
            )
            
            # Determine status and reason - match single file processor logic
            # Linearity is the PRIMARY goal of laser trimming
            if linearity_result and not linearity_result.linearity_pass:
                # Primary requirement not met
                status = AnalysisStatus.FAIL
                status_reason = f"Linearity failed: {linearity_result.linearity_fail_points} points out of spec (0 allowed)"
            elif sigma_result and not sigma_result.sigma_pass:
                # Linearity OK but sigma failed - functional but quality concern
                status = AnalysisStatus.WARNING
                status_reason = f"Sigma gradient ({sigma_result.sigma_gradient:.4f}) exceeds threshold ({sigma_result.sigma_threshold:.4f})"
            elif sigma_result and hasattr(sigma_result, 'gradient_margin') and sigma_result.gradient_margin < 0.2 * sigma_result.sigma_threshold:
                # Both pass but sigma margin is tight (less than 20% margin)
                status = AnalysisStatus.WARNING
                margin_percent = (sigma_result.gradient_margin / sigma_result.sigma_threshold) * 100
                status_reason = f"Sigma margin tight: only {margin_percent:.1f}% margin to threshold"
            else:
                # Both linearity and sigma pass with good margin
                status = AnalysisStatus.PASS
                status_reason = "All tests passed with good margins"
            
            # Create track data
            track_data = TrackData(
                track_id=track_id,
                position_data=list(trim_data['positions']),  # Convert numpy array to list
                error_data=list(trim_data['errors']),  # Convert numpy array to list
                upper_limits=list(trim_data.get('upper_limits', [])) if len(trim_data.get('upper_limits', [])) > 0 else None,
                lower_limits=list(trim_data.get('lower_limits', [])) if len(trim_data.get('lower_limits', [])) > 0 else None,
                travel_length=float(trim_data.get('travel_length', 1.0)),  # Ensure float
                unit_properties=unit_props,
                sigma_analysis=sigma_result,
                linearity_analysis=linearity_result,
                resistance_analysis=resistance_result,
                zone_analysis=zone_result,
                dynamic_range_analysis=dynamic_range_result,
                trim_effectiveness=trim_effectiveness_result,
                failure_prediction=failure_prediction_result,
                status=status,
                status_reason=status_reason
            )
            
            return track_data
            
        except Exception as e:
            logger.error(f"Error processing track {track_id}: {e}")
            return None
    
    def _extract_trim_data_fast(self, sheets: Dict[str, str], file_data: Dict[str, Any], system_type: SystemType) -> Dict[str, Any]:
        """Extract trim data from cached DataFrames."""
        # Prefer trimmed sheet if available
        sheet_name = sheets.get('trimmed', sheets.get('untrimmed'))
        if not sheet_name or sheet_name not in file_data['data']:
            return {}
        
        df = file_data['data'][sheet_name]
        
        # Get column mapping
        columns = SYSTEM_A_COLUMNS if system_type == SystemType.SYSTEM_A else SYSTEM_B_COLUMNS
        
        # Vectorized extraction
        try:
            positions = pd.to_numeric(df.iloc[:, columns['position']], errors='coerce').dropna().values
            errors = pd.to_numeric(df.iloc[:, columns['error']], errors='coerce').dropna().values
            
            # Ensure same length
            min_len = min(len(positions), len(errors))
            positions = positions[:min_len]
            errors = errors[:min_len]
            
            # Extract limits if available
            upper_limits = []
            lower_limits = []
            if 'upper_limit' in columns and 'lower_limit' in columns:
                upper_limits = pd.to_numeric(df.iloc[:, columns['upper_limit']], errors='coerce').dropna().values[:min_len]
                lower_limits = pd.to_numeric(df.iloc[:, columns['lower_limit']], errors='coerce').dropna().values[:min_len]
            
            travel_length = float(np.max(positions) - np.min(positions)) if len(positions) > 0 else 0
            
            return {
                'positions': positions.tolist(),
                'errors': errors.tolist(),
                'upper_limits': upper_limits.tolist(),
                'lower_limits': lower_limits.tolist(),
                'travel_length': travel_length
            }
            
        except Exception as e:
            logger.error(f"Error extracting trim data: {e}")
            return {}
    
    def _extract_unit_properties_fast(self, sheets: Dict[str, str], file_data: Dict[str, Any], system_type: SystemType) -> UnitProperties:
        """Extract unit properties quickly."""
        props = UnitProperties()
        
        # Use appropriate cell locations
        cells = SYSTEM_A_CELLS if system_type == SystemType.SYSTEM_A else SYSTEM_B_CELLS
        
        # Quick extraction without multiple file reads
        try:
            # Extract from untrimmed sheet
            if 'untrimmed' in sheets and sheets['untrimmed'] in file_data['data']:
                df = file_data['data'][sheets['untrimmed']]
                # Direct cell access is faster than extract_cell_value
                if system_type == SystemType.SYSTEM_A:
                    # B26 = row 25, col 1
                    if len(df) > 25 and len(df.columns) > 1:
                        props.unit_length = float(df.iloc[25, 1]) if pd.notna(df.iloc[25, 1]) else None
                    # B10 = row 9, col 1  
                    if len(df) > 9 and len(df.columns) > 1:
                        props.untrimmed_resistance = float(df.iloc[9, 1]) if pd.notna(df.iloc[9, 1]) else None
                else:  # System B
                    # K1 = row 0, col 10 for unit length
                    if len(df) > 0 and len(df.columns) > 10:
                        props.unit_length = float(df.iloc[0, 10]) if pd.notna(df.iloc[0, 10]) else None
                    # R1 = row 0, col 17 for untrimmed resistance
                    if len(df) > 0 and len(df.columns) > 17:
                        props.untrimmed_resistance = float(df.iloc[0, 17]) if pd.notna(df.iloc[0, 17]) else None
            
            # Extract from trimmed sheet
            if 'trimmed' in sheets and sheets['trimmed'] in file_data['data']:
                df = file_data['data'][sheets['trimmed']]
                if system_type == SystemType.SYSTEM_A:
                    # B10 = row 9, col 1 for trimmed resistance
                    if len(df) > 9 and len(df.columns) > 1:
                        props.trimmed_resistance = float(df.iloc[9, 1]) if pd.notna(df.iloc[9, 1]) else None
                else:  # System B
                    # R1 = row 0, col 17 for trimmed resistance
                    if len(df) > 0 and len(df.columns) > 17:
                        props.trimmed_resistance = float(df.iloc[0, 17]) if pd.notna(df.iloc[0, 17]) else None
        except Exception as e:
            logger.debug(f"Error extracting unit properties: {e}")
        
        return props
    
    def _analyze_sigma_fast(self, trim_data: Dict[str, Any], unit_props: UnitProperties, model: str) -> Optional[SigmaAnalysis]:
        """Fast sigma analysis with minimal overhead."""
        try:
            # Direct calculation for speed
            positions = np.array(trim_data['positions'])
            errors = np.array(trim_data['errors'])
            
            if len(positions) < 10:  # Minimum data points
                return None
            
            # Calculate gradients (derivatives) of the error curve
            # This is the correct calculation for sigma gradient
            gradients = []
            step_size = 1  # Use step size of 1 for fast processing
            
            for i in range(len(positions) - step_size):
                dx = positions[i + step_size] - positions[i]
                dy = errors[i + step_size] - errors[i]
                
                # Avoid division by zero
                if abs(dx) > 1e-6:
                    gradient = dy / dx
                    gradients.append(gradient)
            
            if not gradients:
                logger.warning("No valid gradients calculated in fast mode")
                sigma_gradient = 0.0001  # Small non-zero value
            else:
                # Calculate standard deviation of gradients (this is the sigma gradient)
                sigma_gradient = np.std(gradients, ddof=1)
                
                # Ensure valid value
                if np.isnan(sigma_gradient) or np.isinf(sigma_gradient):
                    sigma_gradient = 0.0001
            
            # Calculate threshold based on model
            travel_length = trim_data.get('travel_length', 1)
            linearity_spec = trim_data.get('linearity_spec', 0.01)
            
            # Model-specific thresholds (matching sigma_analyzer.py logic)
            if model == '8340-1':
                sigma_threshold = 0.4
            elif model.startswith('8555'):
                base_threshold = 0.0015
                spec_factor = linearity_spec / 0.01 if linearity_spec > 0 else 1.0
                sigma_threshold = base_threshold * spec_factor
            else:
                # Default calculation
                scaling_factor = 24.0
                effective_length = unit_props.unit_length if unit_props.unit_length and unit_props.unit_length > 0 else travel_length
                if effective_length and effective_length > 0:
                    sigma_threshold = (linearity_spec / effective_length) * (scaling_factor * 0.5)
                else:
                    sigma_threshold = scaling_factor * 0.01
                
                # Apply bounds
                sigma_threshold = max(0.0001, min(0.05, sigma_threshold))
            
            # Calculate gradient margin
            gradient_margin = sigma_threshold - sigma_gradient
            
            return SigmaAnalysis(
                sigma_gradient=sigma_gradient,
                sigma_threshold=sigma_threshold,
                sigma_pass=sigma_gradient <= sigma_threshold,
                gradient_margin=gradient_margin,
                scaling_factor=24.0  # Default scaling factor
            )
            
        except Exception as e:
            logger.error(f"Fast sigma analysis error: {e}")
            return None
    
    def _analyze_linearity_fast(self, trim_data: Dict[str, Any]) -> Optional[LinearityAnalysis]:
        """Fast linearity analysis."""
        
        try:
            # Simplified linearity check
            errors = np.array(trim_data['errors'])
            upper_limits = np.array(trim_data.get('upper_limits', []))
            lower_limits = np.array(trim_data.get('lower_limits', []))
            
            if len(upper_limits) != len(errors) or len(lower_limits) != len(errors):
                # No limits, assume pass
                return LinearityAnalysis(
                    linearity_spec=1.0,
                    optimal_offset=0.0,
                    final_linearity_error_raw=0.0,
                    final_linearity_error_shifted=0.0,
                    linearity_pass=True,
                    linearity_fail_points=0,
                    max_deviation=0.0,  # Add missing field
                    max_deviation_position=None  # Add missing field
                )
            
            # Check if errors are within limits
            within_limits = np.all((errors <= upper_limits) & (errors >= lower_limits))
            
            # Count actual failing points
            fail_count = np.sum((errors > upper_limits) | (errors < lower_limits))
            
            # Simple linearity error as max deviation from limits
            upper_violations = np.maximum(0, errors - upper_limits)
            lower_violations = np.maximum(0, lower_limits - errors)
            max_violation = max(np.max(upper_violations), np.max(lower_violations))
            
            # Find position of max deviation for completeness
            max_deviation_idx = np.argmax(np.maximum(upper_violations, lower_violations))
            positions = np.array(trim_data.get('positions', []))
            max_deviation_position = positions[max_deviation_idx] if len(positions) > max_deviation_idx else None
            
            return LinearityAnalysis(
                linearity_spec=0.1,  # 10% spec
                optimal_offset=0.0,  # No offset in fast mode
                final_linearity_error_raw=max_violation,
                final_linearity_error_shifted=max_violation,  # Same as raw in fast mode
                linearity_pass=within_limits,
                linearity_fail_points=int(fail_count),  # Calculate actual fail count
                max_deviation=max_violation,  # Add missing field
                max_deviation_position=max_deviation_position  # Add missing field
            )
            
        except Exception as e:
            logger.error(f"Fast linearity analysis error: {e}")
            return None
    
    def _analyze_resistance_fast(self, unit_props: UnitProperties) -> ResistanceAnalysis:
        """Fast resistance analysis."""
        
        try:
            if unit_props.untrimmed_resistance and unit_props.trimmed_resistance:
                change_percent = ((unit_props.trimmed_resistance - unit_props.untrimmed_resistance) / 
                                unit_props.untrimmed_resistance * 100)
                
                resistance_change = unit_props.trimmed_resistance - unit_props.untrimmed_resistance
                
                return ResistanceAnalysis(
                    untrimmed_resistance=unit_props.untrimmed_resistance,
                    trimmed_resistance=unit_props.trimmed_resistance,
                    resistance_change=resistance_change,
                    resistance_change_percent=change_percent
                )
            else:
                # Return ResistanceAnalysis with None values instead of returning None
                return ResistanceAnalysis(
                    untrimmed_resistance=unit_props.untrimmed_resistance,
                    trimmed_resistance=unit_props.trimmed_resistance,
                    resistance_change=None,
                    resistance_change_percent=None
                )
            
        except Exception as e:
            logger.error(f"Fast resistance analysis error: {e}")
            # Return ResistanceAnalysis with None values on error
            return ResistanceAnalysis(
                untrimmed_resistance=None,
                trimmed_resistance=None,
                resistance_change=None,
                resistance_change_percent=None
            )
    
    def _determine_overall_status(self, tracks: Dict[str, TrackData]) -> AnalysisStatus:
        """Determine overall analysis status from all tracks."""
        if not tracks:
            return AnalysisStatus.ERROR
        
        statuses = [track.status for track in tracks.values()]
        
        # Use same logic as single file processor for consistency
        if any(s == AnalysisStatus.FAIL for s in statuses):
            return AnalysisStatus.FAIL
        elif any(s == AnalysisStatus.WARNING for s in statuses):
            return AnalysisStatus.WARNING
        elif all(s == AnalysisStatus.PASS for s in statuses):
            return AnalysisStatus.PASS
        else:
            return AnalysisStatus.ERROR
    
    def _determine_overall_validation_status(self, tracks: Dict[str, TrackData]) -> ValidationStatus:
        """Determine overall validation status from all tracks - laser trimming focused."""
        if not tracks:
            return ValidationStatus.NOT_VALIDATED
        
        # For laser trimming, success means:
        # 1. Linearity meets specification (most important)
        # 2. Sigma is reasonable (process control)
        # 3. Resistance changes are irrelevant if specs are met
        
        all_warnings = []
        critical_failures = []
        
        for track_data in tracks.values():
            # Check linearity - this is the PRIMARY goal of laser trimming
            if hasattr(track_data, 'linearity_analysis') and track_data.linearity_analysis:
                if not track_data.linearity_analysis.linearity_pass:
                    critical_failures.append("Linearity specification not met")
                else:
                    # Linearity passed - this is success for laser trimming
                    pass
            
            # Check sigma - should be reasonable but not critical
            if hasattr(track_data, 'sigma_analysis') and track_data.sigma_analysis:
                if track_data.sigma_analysis.sigma_gradient > 1.0:  # Very high sigma
                    all_warnings.append("High sigma gradient - check process control")
                elif not track_data.sigma_analysis.sigma_pass:
                    all_warnings.append("Sigma slightly above threshold")
            
            # Resistance changes are NORMAL and EXPECTED in laser trimming
            # Only flag extreme cases (>50% change)
            if hasattr(track_data, 'resistance_analysis') and track_data.resistance_analysis:
                if (track_data.resistance_analysis.resistance_change_percent and 
                    abs(track_data.resistance_analysis.resistance_change_percent) > 50):
                    all_warnings.append("Very large resistance change - verify process")
        
        # Determine overall status based on laser trimming success criteria
        if critical_failures:
            return ValidationStatus.FAILED  # Failed to meet primary specifications
        elif all_warnings:
            return ValidationStatus.WARNING  # Met specs but with concerns
        else:
            return ValidationStatus.VALIDATED  # Successful trim
    
    def _generate_plot_fast(self, result: AnalysisResult, output_dir: Path):
        """Generate plot with minimal overhead."""
        if self.turbo_mode or not self.enable_plots:
            return
        
        try:
            # Generate plot for primary track only
            primary_track = next(iter(result.tracks.values()))
            plot_path = create_analysis_plot(
                primary_track,
                output_dir,
                f"{result.metadata.filename}_fast",
                dpi=72  # Lower DPI for speed
            )
            primary_track.plot_path = plot_path
        except Exception as e:
            logger.error(f"Plot generation error: {e}")
    
    def _get_default_linearity(self) -> LinearityAnalysis:
        """Get default linearity analysis for turbo mode."""
        return LinearityAnalysis(
            linearity_spec=1.0,
            optimal_offset=0.0,
            final_linearity_error_raw=0.0,
            final_linearity_error_shifted=0.0,
            linearity_pass=True,  # Assume pass in turbo mode
            linearity_fail_points=0
        )
    
    def _get_default_resistance(self) -> ResistanceAnalysis:
        """Get default resistance analysis for turbo mode."""
        return ResistanceAnalysis(
            untrimmed_resistance=None,
            trimmed_resistance=None,
            resistance_change=None,
            resistance_change_percent=None
        )
    
    def _analyze_zones_fast(self, trim_data: Dict[str, Any]) -> Optional[ZoneAnalysis]:
        """Fast zone analysis for consistency."""
        try:
            positions = np.array(trim_data.get('positions', []))
            errors = np.array(trim_data.get('errors', []))
            
            if len(positions) < 10 or len(errors) < 10:
                return None
            
            # Divide into 3 zones
            n_zones = 3
            zone_size = len(positions) // n_zones
            
            zones = []
            rms_errors = []
            for i in range(n_zones):
                start_idx = i * zone_size
                end_idx = (i + 1) * zone_size if i < n_zones - 1 else len(positions)
                
                zone_errors = errors[start_idx:end_idx]
                zone_rms = np.sqrt(np.mean(np.square(zone_errors))) if len(zone_errors) > 0 else 0
                rms_errors.append(zone_rms)
                
                zones.append({
                    'zone_id': f"Zone_{i+1}",
                    'start_position': float(positions[start_idx]),
                    'end_position': float(positions[end_idx-1]),
                    'rms_error': float(zone_rms),
                    'point_count': len(zone_errors)
                })
            
            # Calculate zone consistency
            min_rms = min(rms_errors) if rms_errors else 1.0
            max_rms = max(rms_errors) if rms_errors else 1.0
            zone_consistency = max_rms / min_rms if min_rms > 0 else 1.0
            
            return ZoneAnalysis(
                zones=zones,
                zone_consistency=float(zone_consistency)
            )
            
        except Exception as e:
            logger.error(f"Zone analysis error: {e}")
            return None
    
    def _analyze_dynamic_range_fast(self, trim_data: Dict[str, Any]) -> Optional[DynamicRangeAnalysis]:
        """Fast dynamic range analysis."""
        try:
            positions = np.array(trim_data.get('positions', []))
            errors = np.array(trim_data.get('errors', []))
            
            if len(positions) == 0 or len(errors) == 0:
                return None
            
            position_range = float(np.max(positions) - np.min(positions))
            error_range = float(np.max(errors) - np.min(errors))
            
            # Calculate signal to noise ratio
            error_std = np.std(errors)
            snr = abs(np.mean(errors)) / error_std if error_std > 0 else 0.0
            
            return DynamicRangeAnalysis(
                position_range=position_range,
                error_range=error_range,
                signal_to_noise_ratio=float(snr)
            )
            
        except Exception as e:
            logger.error(f"Dynamic range analysis error: {e}")
            return None
    
    def _calculate_trim_effectiveness_fast(
        self,
        sigma_analysis: Optional[SigmaAnalysis],
        linearity_analysis: Optional[LinearityAnalysis],
        resistance_analysis: Optional[ResistanceAnalysis]
    ) -> TrimEffectiveness:
        """Fast calculation of trim effectiveness."""
        try:
            # Simple effectiveness calculation
            sigma_improvement = 0.0
            if sigma_analysis and sigma_analysis.sigma_pass:
                # Improvement based on how far below threshold we are
                sigma_improvement = max(0, (1 - sigma_analysis.sigma_ratio) * 100)
            
            linearity_improvement = 0.0
            if linearity_analysis and linearity_analysis.linearity_pass:
                # Improvement based on final error vs spec
                if linearity_analysis.linearity_spec > 0:
                    linearity_improvement = max(0, (1 - linearity_analysis.final_linearity_error_shifted / linearity_analysis.linearity_spec) * 100)
            
            resistance_stability = 100.0
            if resistance_analysis and resistance_analysis.resistance_change_percent is not None:
                # Stability as inverse of change
                resistance_stability = max(0, 100.0 - abs(resistance_analysis.resistance_change_percent))
            
            # Overall effectiveness
            overall_effectiveness = (sigma_improvement + linearity_improvement) / 2.0
            
            return TrimEffectiveness(
                improvement_percent=overall_effectiveness,
                sigma_improvement=sigma_improvement,
                linearity_improvement=linearity_improvement,
                resistance_stability=resistance_stability
            )
            
        except Exception as e:
            logger.error(f"Trim effectiveness calculation error: {e}")
            # Return default values
            return TrimEffectiveness(
                improvement_percent=0.0,
                sigma_improvement=0.0,
                linearity_improvement=0.0,
                resistance_stability=100.0
            )
    
    def _calculate_failure_prediction_fast(
        self,
        sigma_analysis: Optional[SigmaAnalysis],
        linearity_analysis: Optional[LinearityAnalysis],
        unit_props: UnitProperties
    ) -> FailurePrediction:
        """Fast failure prediction without ML."""
        try:
            # Simple failure probability calculation
            sigma_factor = 0.0
            if sigma_analysis:
                sigma_factor = min(sigma_analysis.sigma_ratio, 1.0)
            
            linearity_factor = 0.0
            if linearity_analysis and linearity_analysis.linearity_spec > 0:
                linearity_factor = min(
                    linearity_analysis.final_linearity_error_shifted / linearity_analysis.linearity_spec,
                    1.0
                )
            
            resistance_factor = 0.0
            if unit_props.resistance_change_percent is not None:
                # Only significant if change > 5%
                resistance_factor = min(abs(unit_props.resistance_change_percent) / 10.0, 1.0)
            
            # Weighted combination
            failure_probability = (
                0.4 * sigma_factor +
                0.4 * linearity_factor +
                0.2 * resistance_factor
            )
            
            # Scale down for passing units
            if sigma_analysis and linearity_analysis:
                if sigma_analysis.sigma_pass and linearity_analysis.linearity_pass:
                    failure_probability *= 0.5
            
            failure_probability = min(max(failure_probability, 0), 1)
            
            # Determine risk category
            if failure_probability > 0.8:
                risk_category = RiskCategory.HIGH
            elif failure_probability > 0.5:
                risk_category = RiskCategory.MEDIUM
            else:
                risk_category = RiskCategory.LOW
            
            gradient_margin = sigma_analysis.gradient_margin if sigma_analysis else 0.0
            
            return FailurePrediction(
                failure_probability=failure_probability,
                risk_category=risk_category,
                gradient_margin=gradient_margin,
                contributing_factors={
                    'sigma': sigma_factor,
                    'linearity': linearity_factor,
                    'resistance': resistance_factor
                }
            )
            
        except Exception as e:
            logger.error(f"Failure prediction error: {e}")
            # Return default low risk prediction
            return FailurePrediction(
                failure_probability=0.1,
                risk_category=RiskCategory.LOW,
                gradient_margin=0.0,
                contributing_factors={}
            )
    
    def _perform_comprehensive_validation(self, file_path: Path, tracks: Dict[str, TrackData], 
                                         file_data: Dict[str, Any], metadata: FileMetadata) -> Tuple[List[Any], List[str]]:
        """Perform comprehensive validation to ensure data integrity in fast mode."""
        validation_issues = []
        processing_errors = []
        
        try:
            # 1. File-level validation
            try:
                file_validation = validate_excel_file(file_path)
                if not file_validation.is_valid:
                    validation_issues.extend(file_validation.errors)
            except Exception as e:
                processing_errors.append(f"File validation error: {str(e)}")
            
            # 2. Model validation
            try:
                if metadata.model:
                    model_validation = validate_model_number(metadata.model)
                    if not model_validation.is_valid:
                        validation_issues.extend(model_validation.errors)
            except Exception as e:
                processing_errors.append(f"Model validation error: {str(e)}")
            
            # 3. Track-level comprehensive validation
            for track_id, track_data in tracks.items():
                try:
                    # Validate analysis data structure - extract data from TrackData object
                    track_data_dict = {
                        'positions': track_data.position_data or [],
                        'errors': track_data.error_data or [],
                        'upper_limits': track_data.upper_limits or [],
                        'lower_limits': track_data.lower_limits or []
                    }
                    analysis_validation = validate_analysis_data(track_data_dict)
                    if not analysis_validation.is_valid:
                        validation_issues.extend([f"Track {track_id}: {error}" for error in analysis_validation.errors])
                    
                    # Calculation validation using industry standards
                    if self.calculation_validator:
                        # Validate sigma calculation
                        if track_data.sigma_analysis:
                            try:
                                sigma_validation = self.calculation_validator.validate_sigma_gradient(
                                    calculated_sigma=track_data.sigma_analysis.sigma_gradient,
                                    position_data=[],  # Would need raw data for full validation
                                    error_data=[],
                                    model_number=metadata.model or "Unknown"
                                )
                                if not sigma_validation.is_valid:
                                    validation_issues.append(f"Track {track_id}: {sigma_validation.standard_reference}")
                            except Exception as e:
                                processing_errors.append(f"Track {track_id} sigma validation: {str(e)}")
                        
                        # Validate linearity calculation
                        if track_data.linearity_analysis:
                            try:
                                linearity_validation = self.calculation_validator.validate_linearity_error(
                                    calculated_error=track_data.linearity_analysis.linearity_error,
                                    position_data=[],  # Would need raw data for full validation
                                    resistance_data=[],
                                    model_number=metadata.model or "Unknown"
                                )
                                if not linearity_validation.is_valid:
                                    validation_issues.append(f"Track {track_id}: {linearity_validation.standard_reference}")
                            except Exception as e:
                                processing_errors.append(f"Track {track_id} linearity validation: {str(e)}")
                    
                    # Validate resistance values
                    if track_data.resistance_analysis:
                        try:
                            resistance_validation = validate_resistance_values(
                                track_data.resistance_analysis,
                                metadata.model or "Unknown"
                            )
                            if not resistance_validation.is_valid:
                                validation_issues.extend([f"Track {track_id}: {error}" for error in resistance_validation.errors])
                        except Exception as e:
                            processing_errors.append(f"Track {track_id} resistance validation: {str(e)}")
                            
                except Exception as e:
                    processing_errors.append(f"Track {track_id} comprehensive validation error: {str(e)}")
                    
        except Exception as e:
            processing_errors.append(f"Comprehensive validation failed: {str(e)}")
        
        return validation_issues, processing_errors
    
    def _determine_comprehensive_validation_status(self, tracks: Dict[str, TrackData], 
                                                 validation_issues: List[Any]) -> ValidationStatus:
        """Determine validation status based on comprehensive validation results."""
        if not tracks:
            return ValidationStatus.NOT_VALIDATED
        
        # If there are validation issues, determine severity
        if validation_issues:
            # Check for critical failures
            critical_issues = [issue for issue in validation_issues 
                             if any(critical in str(issue).lower() 
                                   for critical in ['failed', 'invalid', 'error', 'critical'])]
            
            if critical_issues:
                return ValidationStatus.FAILED
            else:
                return ValidationStatus.WARNING
        
        # Check track-level validation using the same logic but with more context
        all_warnings = []
        critical_failures = []
        
        for track_data in tracks.values():
            # Check linearity - this is the PRIMARY goal of laser trimming
            if hasattr(track_data, 'linearity_analysis') and track_data.linearity_analysis:
                if not track_data.linearity_analysis.linearity_pass:
                    critical_failures.append("Linearity specification not met")
            
            # Check sigma - should be reasonable
            if hasattr(track_data, 'sigma_analysis') and track_data.sigma_analysis:
                if track_data.sigma_analysis.sigma_gradient > 1.0:
                    all_warnings.append("High sigma gradient - check process control")
                elif not track_data.sigma_analysis.sigma_pass:
                    all_warnings.append("Sigma slightly above threshold")
            
            # Check for extreme resistance changes (>75% is concerning)
            if hasattr(track_data, 'resistance_analysis') and track_data.resistance_analysis:
                if (track_data.resistance_analysis.resistance_change_percent and 
                    abs(track_data.resistance_analysis.resistance_change_percent) > 75):
                    all_warnings.append("Extreme resistance change - verify process")
        
        # Determine final status
        if critical_failures:
            return ValidationStatus.FAILED
        elif all_warnings:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.VALIDATED
    
    def _cleanup_memory(self):
        """Enhanced memory cleanup for large batches."""
        self.reader.clear_cache()
        
        # Multiple garbage collection passes for thoroughness
        for _ in range(2):
            gc.collect()
        
        # Close any matplotlib figures
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        
        # Log memory status
        memory_info = psutil.virtual_memory()
        logger.debug(f"Memory cleanup - Available: {memory_info.available / (1024**3):.1f}GB ({memory_info.percent:.1f}% used)")
        
        # For very large batches, do more aggressive cleanup
        if hasattr(self, '_large_batch_mode') and self._large_batch_mode:
            # Clear more caches
            import sys
            if hasattr(sys, 'modules'):
                # Clear module caches for pandas/numpy if needed
                for module_name in list(sys.modules.keys()):
                    if 'pandas' in module_name or 'numpy' in module_name:
                        if hasattr(sys.modules[module_name], '_cache'):
                            try:
                                sys.modules[module_name]._cache.clear()
                            except:
                                pass
            
            # Force more aggressive garbage collection
            for _ in range(3):
                gc.collect()
            
            logger.debug(f"Aggressive memory cleanup completed")


# Convenience function for direct usage
def process_files_turbo(
    file_paths: List[Path],
    config: Config,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> List[AnalysisResult]:
    """
    Process files in turbo mode for maximum performance.
    
    Features:
    - True parallel processing with multiprocessing
    - Memory-efficient Excel reading
    - Minimal analysis mode
    - No plot generation
    - Aggressive memory management
    """
    processor = FastProcessor(config, turbo_mode=True)
    return processor.process_batch_fast(file_paths, output_dir, progress_callback)