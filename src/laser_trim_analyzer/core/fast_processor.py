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
    """High-performance processor for large batch operations."""
    
    def __init__(self, config: Config, turbo_mode: bool = False):
        self.config = config
        self.turbo_mode = turbo_mode
        self.reader = FastExcelReader(turbo_mode=turbo_mode)
        
        # Initialize analyzers once
        self.sigma_analyzer = SigmaAnalyzer(config, logger)
        self.linearity_analyzer = LinearityAnalyzer(config, logger)
        self.resistance_analyzer = ResistanceAnalyzer(config, logger)
        
        # Performance settings - limit workers to prevent 100% CPU usage
        # Even in turbo mode, leave some CPU capacity for the system
        if turbo_mode:
            # Use at most 75% of cores, minimum 1, maximum 4 to prevent overload
            self.max_workers = min(max(mp.cpu_count() * 3 // 4, 1), 4)
        else:
            # Standard mode uses half the cores, max 2
            self.max_workers = min(max(mp.cpu_count() // 2, 1), 2)
        # Adaptive chunk sizing - will be set based on total files in process_batch
        self.chunk_size = 20  # Default chunk size
        self.enable_plots = False if turbo_mode else config.processing.generate_plots
        
        logger.info(f"FastProcessor initialized - Turbo: {turbo_mode}, Workers: {self.max_workers}, Plots: {self.enable_plots}")
    
    def process_batch_fast(
        self,
        file_paths: List[Path],
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[AnalysisResult]:
        """Process multiple files with maximum performance."""
        total_files = len(file_paths)
        logger.info(f"Starting fast batch processing of {total_files} files")
        
        # Check memory before starting
        memory_available = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        logger.info(f"Available memory: {memory_available:.1f}GB")
        
        # Adaptive chunk sizing based on total files and available memory
        # Reduced chunk sizes to prevent CPU overload
        if self.turbo_mode and total_files > 200:
            # For large batches in turbo mode, use smaller chunks to prevent CPU spikes
            if memory_available >= 4:
                self.chunk_size = 50   # Reduced from 200
            elif memory_available >= 2:
                self.chunk_size = 30   # Reduced from 100
            else:
                self.chunk_size = 20   # Reduced from 50
            logger.info(f"Turbo mode with {total_files} files: using chunk size {self.chunk_size}")
        elif memory_available < 2 and total_files > 100:
            logger.warning("Low memory detected - using conservative chunk size")
            self.chunk_size = 10  # Reduced from 20
        else:
            # Default chunk size for smaller batches
            self.chunk_size = 10  # Reduced from 20
        
        results = []
        processed = 0
        start_time = time.time()
        
        # Process in chunks to manage memory
        chunks = [file_paths[i:i + self.chunk_size] for i in range(0, total_files, self.chunk_size)]
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start = time.time()
            
            # Update progress at start of chunk with accurate file count
            if progress_callback:
                progress = processed / total_files
                files_in_chunk = len(chunk)
                start_file = processed + 1
                end_file = min(processed + files_in_chunk, total_files)
                # Check if callback returns False to indicate stop
                continue_processing = progress_callback(f"Processing files {start_file}-{end_file} of {total_files}", progress)
                if continue_processing is False:
                    logger.info("Processing stopped by user request")
                    break
            
            # Process chunk in parallel
            chunk_results = self._process_chunk_parallel(chunk, output_dir)
            results.extend(chunk_results)
            
            # Update processed count with actual results
            processed += len(chunk_results)
            chunk_time = time.time() - chunk_start
            rate = len(chunk_results) / chunk_time if chunk_time > 0 else 0
            
            # Update progress after chunk with accurate count
            if progress_callback:
                progress = processed / total_files
                continue_processing = progress_callback(f"Completed {processed} of {total_files} files", progress)
                if continue_processing is False:
                    logger.info("Processing stopped after chunk completion")
                    break
            
            logger.info(f"Chunk {chunk_idx + 1} completed: {len(chunk_results)} files in {chunk_time:.1f}s ({rate:.1f} files/sec)")
            
            # Memory management between chunks
            if chunk_idx % 5 == 0:
                self._cleanup_memory()
            
            # CPU throttling - add delay between chunks to prevent 100% CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 80:
                logger.info(f"High CPU usage ({cpu_percent:.1f}%), pausing for 2 seconds")
                time.sleep(2.0)
            elif cpu_percent > 60:
                logger.debug(f"Moderate CPU usage ({cpu_percent:.1f}%), pausing for 1 second")
                time.sleep(1.0)
            else:
                # Always add a small delay between chunks
                time.sleep(0.5)
            
            # Check if we should continue
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                logger.warning(f"Critical memory usage: {memory_percent}% - stopping batch")
                break
        
        total_time = time.time() - start_time
        overall_rate = processed / total_time if total_time > 0 else 0
        
        logger.info(f"Batch processing complete: {processed} files in {total_time:.1f}s ({overall_rate:.1f} files/sec)")
        
        # Final cleanup
        self._cleanup_memory()
        
        return results
    
    def _process_chunk_parallel(self, file_paths: List[Path], output_dir: Optional[Path]) -> List[AnalysisResult]:
        """Process a chunk of files in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self._process_file_worker, file_path, output_dir): file_path
                for file_path in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per file
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        
        return results
    
    def _process_file_worker(self, file_path: Path, output_dir: Optional[Path]) -> Optional[AnalysisResult]:
        """Worker function to process a single file."""
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
            
            # Determine overall validation status using the same logic as the main processor
            overall_validation_status = self._determine_overall_validation_status(tracks)
            
            # Create result
            result = AnalysisResult(
                metadata=metadata,
                tracks=tracks,
                overall_status=overall_status,
                overall_validation_status=overall_validation_status,
                processing_time=(time.time() - time.time()),  # Simplified timing
                validation_issues=[],  # Empty list for fast mode
                processing_errors=[]   # Empty list for fast mode
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
            
            # Determine status and reason
            status = AnalysisStatus.PASS
            status_reason = "All tests passed"
            
            if linearity_result and not linearity_result.linearity_pass:
                status = AnalysisStatus.FAIL
                status_reason = f"Linearity failed: {linearity_result.linearity_fail_points} points out of spec"
            elif sigma_result and not sigma_result.sigma_pass:
                status = AnalysisStatus.FAIL
                status_reason = f"Sigma gradient ({sigma_result.sigma_gradient:.4f}) exceeds threshold ({sigma_result.sigma_threshold:.4f})"
            elif sigma_result and hasattr(sigma_result, 'gradient_margin') and sigma_result.gradient_margin < 0.2 * sigma_result.sigma_threshold:
                status = AnalysisStatus.WARNING
                margin_percent = (sigma_result.gradient_margin / sigma_result.sigma_threshold) * 100
                status_reason = f"Sigma margin tight: only {margin_percent:.1f}% margin to threshold"
            
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
        
        all_pass = all(track.status == AnalysisStatus.PASS for track in tracks.values())
        any_fail = any(track.status == AnalysisStatus.FAIL for track in tracks.values())
        
        if all_pass:
            return AnalysisStatus.PASS
        elif any_fail:
            return AnalysisStatus.FAIL
        else:
            return AnalysisStatus.WARNING
    
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
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup."""
        self.reader.clear_cache()
        gc.collect()
        
        # Close any matplotlib figures
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        
        # Log memory status
        memory_info = psutil.virtual_memory()
        logger.info(f"Memory cleanup - Available: {memory_info.available / (1024**3):.1f}GB ({memory_info.percent}% used)")


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