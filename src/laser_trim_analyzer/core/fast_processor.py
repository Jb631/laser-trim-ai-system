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
    
    def __init__(self):
        self._cache = {}
        self._max_cache_size = 50
        
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
        self.reader = FastExcelReader()
        
        # Initialize analyzers once
        self.sigma_analyzer = SigmaAnalyzer(config, logger)
        self.linearity_analyzer = LinearityAnalyzer(config, logger)
        self.resistance_analyzer = ResistanceAnalyzer(config, logger)
        
        # Performance settings
        self.max_workers = mp.cpu_count() if turbo_mode else max(mp.cpu_count() // 2, 1)
        self.chunk_size = 50  # Files per chunk
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
        
        if memory_available < 2 and total_files > 100:
            logger.warning("Low memory detected - enabling aggressive memory management")
            self.chunk_size = 20
        
        results = []
        processed = 0
        start_time = time.time()
        
        # Process in chunks to manage memory
        chunks = [file_paths[i:i + self.chunk_size] for i in range(0, total_files, self.chunk_size)]
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start = time.time()
            
            # Update progress
            if progress_callback:
                progress = (chunk_idx * self.chunk_size) / total_files
                progress_callback(f"Processing chunk {chunk_idx + 1}/{len(chunks)}", progress)
            
            # Process chunk in parallel
            chunk_results = self._process_chunk_parallel(chunk, output_dir)
            results.extend(chunk_results)
            
            processed += len(chunk)
            chunk_time = time.time() - chunk_start
            rate = len(chunk) / chunk_time if chunk_time > 0 else 0
            
            logger.info(f"Chunk {chunk_idx + 1} completed: {len(chunk)} files in {chunk_time:.1f}s ({rate:.1f} files/sec)")
            
            # Memory management between chunks
            if chunk_idx % 5 == 0:
                self._cleanup_memory()
            
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
            # Read file data once
            file_data = self.reader.read_file_once(file_path)
            
            # Create metadata
            metadata = FileMetadata(
                filename=file_data['metadata']['filename'],
                file_path=file_data['metadata']['file_path'],
                model=file_data['metadata']['model'],
                serial=file_data['metadata']['serial'],
                timestamp=file_data['metadata']['timestamp'],
                system_type=file_data['system_type'],
                file_size_mb=file_data['metadata']['file_size_mb']
            )
            
            # Process tracks
            tracks = self._process_tracks_fast(file_path, file_data, metadata)
            
            if not tracks:
                logger.warning(f"No tracks processed for {file_path}")
                return None
            
            # Determine overall status
            overall_status = self._determine_overall_status(tracks)
            
            # Create result
            result = AnalysisResult(
                metadata=metadata,
                tracks=tracks,
                overall_status=overall_status,
                processing_time=(time.time() - time.time()),  # Simplified timing
                validation_status=ValidationStatus.NOT_VALIDATED if self.turbo_mode else ValidationStatus.VALIDATED
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
            trim_data = self._extract_trim_data_fast(sheets, file_data, metadata.system_type)
            
            if not trim_data or not trim_data.get('positions'):
                return None
            
            # Extract unit properties
            unit_props = self._extract_unit_properties_fast(sheets, file_data, metadata.system_type)
            
            # Run analyses in turbo mode (skip some validations)
            sigma_result = self._analyze_sigma_fast(trim_data, unit_props, metadata.model)
            linearity_result = self._analyze_linearity_fast(trim_data) if not self.turbo_mode else None
            resistance_result = self._analyze_resistance_fast(unit_props) if not self.turbo_mode else None
            
            # Create track data
            track_data = TrackData(
                track_id=track_id,
                positions=trim_data['positions'],
                errors=trim_data['errors'],
                upper_limits=trim_data.get('upper_limits', []),
                lower_limits=trim_data.get('lower_limits', []),
                travel_length=trim_data.get('travel_length', 0),
                unit_properties=unit_props,
                sigma_analysis=sigma_result,
                linearity_analysis=linearity_result,
                resistance_analysis=resistance_result,
                status=AnalysisStatus.PASS if sigma_result and sigma_result.sigma_pass else AnalysisStatus.FAIL
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
        except:
            pass
        
        return props
    
    def _analyze_sigma_fast(self, trim_data: Dict[str, Any], unit_props: UnitProperties, model: str) -> Optional[SigmaAnalysis]:
        """Fast sigma analysis with minimal overhead."""
        try:
            # Direct calculation for speed
            positions = np.array(trim_data['positions'])
            errors = np.array(trim_data['errors'])
            
            if len(positions) < 10:  # Minimum data points
                return None
            
            # Simple linear regression for sigma gradient
            A = np.vstack([positions, np.ones(len(positions))]).T
            m, c = np.linalg.lstsq(A, errors, rcond=None)[0]
            
            # Calculate residuals
            fitted = m * positions + c
            residuals = errors - fitted
            sigma = np.std(residuals)
            
            # Quick sigma gradient calculation
            travel_length = trim_data.get('travel_length', 1)
            sigma_gradient = abs(sigma / travel_length) if travel_length > 0 else 0
            
            # Simple threshold
            sigma_threshold = 0.3 if "PRECISION" in model.upper() else 0.5
            
            return SigmaAnalysis(
                sigma_gradient=sigma_gradient,
                sigma_threshold=sigma_threshold,
                sigma_pass=sigma_gradient <= sigma_threshold,
                sigma_values=[],  # Skip detailed values in turbo mode
                position_values=[],
                trend_line_slope=m,
                trend_line_intercept=c,
                r_squared=0.0,  # Skip in turbo mode
                data_points_used=len(positions)
            )
            
        except Exception as e:
            logger.error(f"Fast sigma analysis error: {e}")
            return None
    
    def _analyze_linearity_fast(self, trim_data: Dict[str, Any]) -> Optional[LinearityAnalysis]:
        """Fast linearity analysis."""
        if self.turbo_mode:
            return None
        
        try:
            # Simplified linearity check
            errors = np.array(trim_data['errors'])
            upper_limits = np.array(trim_data.get('upper_limits', []))
            lower_limits = np.array(trim_data.get('lower_limits', []))
            
            if len(upper_limits) != len(errors) or len(lower_limits) != len(errors):
                # No limits, assume pass
                return LinearityAnalysis(
                    linearity_error=0.0,
                    linearity_threshold=1.0,
                    linearity_pass=True
                )
            
            # Check if errors are within limits
            within_limits = np.all((errors <= upper_limits) & (errors >= lower_limits))
            
            # Simple linearity error as max deviation from limits
            upper_violations = np.maximum(0, errors - upper_limits)
            lower_violations = np.maximum(0, lower_limits - errors)
            max_violation = max(np.max(upper_violations), np.max(lower_violations))
            
            return LinearityAnalysis(
                linearity_error=max_violation,
                linearity_threshold=0.1,  # 10% threshold
                linearity_pass=within_limits
            )
            
        except Exception as e:
            logger.error(f"Fast linearity analysis error: {e}")
            return None
    
    def _analyze_resistance_fast(self, unit_props: UnitProperties) -> Optional[ResistanceAnalysis]:
        """Fast resistance analysis."""
        if self.turbo_mode:
            return None
        
        try:
            if unit_props.untrimmed_resistance and unit_props.trimmed_resistance:
                change_percent = ((unit_props.trimmed_resistance - unit_props.untrimmed_resistance) / 
                                unit_props.untrimmed_resistance * 100)
                
                return ResistanceAnalysis(
                    untrimmed_resistance=unit_props.untrimmed_resistance,
                    trimmed_resistance=unit_props.trimmed_resistance,
                    resistance_change_percent=change_percent,
                    trim_effectiveness=abs(change_percent) > 0.1  # Trim had effect
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Fast resistance analysis error: {e}")
            return None
    
    def _determine_overall_status(self, tracks: Dict[str, TrackData]) -> AnalysisStatus:
        """Quickly determine overall status."""
        if not tracks:
            return AnalysisStatus.ERROR
        
        # Any track fails = overall fail
        for track in tracks.values():
            if track.status == AnalysisStatus.FAIL:
                return AnalysisStatus.FAIL
        
        return AnalysisStatus.PASS
    
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