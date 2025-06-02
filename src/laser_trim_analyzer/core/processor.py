"""
Core processing engine for Laser Trim Analyzer v2.

Handles file processing, data extraction, analysis coordination,
ML integration, and result generation.
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
import numpy as np
import re

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
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.analysis.sigma_analyzer import SigmaAnalyzer
from laser_trim_analyzer.analysis.linearity_analyzer import LinearityAnalyzer
from laser_trim_analyzer.analysis.resistance_analyzer import ResistanceAnalyzer
from laser_trim_analyzer.utils.file_utils import ensure_directory, calculate_file_hash
from laser_trim_analyzer.utils.excel_utils import (
    read_excel_sheet, extract_cell_value, find_data_columns, detect_system_type
)
from laser_trim_analyzer.utils.plotting_utils import create_analysis_plot
from laser_trim_analyzer.utils.date_utils import extract_datetime_from_filename
from laser_trim_analyzer.utils.calculation_validator import CalculationValidator, ValidationLevel, CalculationType
# Import comprehensive validation utilities
from laser_trim_analyzer.utils.validators import (
    validate_excel_file, validate_analysis_data, validate_model_number,
    validate_resistance_values, AnalysisValidator, BatchValidator,
    ValidationResult as UtilsValidationResult
)
# Try to import ML components
try:
    from laser_trim_analyzer.ml.predictors import MLPredictor, PredictionResult
    HAS_ML = True
except ImportError:
    HAS_ML = False
    MLPredictor = None
    PredictionResult = None

# Type checking support
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from laser_trim_analyzer.ml.predictors import MLPredictor as MLPredictorType, PredictionResult as PredictionResultType
else:
    if not HAS_ML:
        MLPredictorType = Any
        PredictionResultType = Any
    else:
        MLPredictorType = MLPredictor
        PredictionResultType = PredictionResult


class LaserTrimProcessor:
    """
    Main processing engine for laser trim analysis.

    Coordinates all aspects of file processing, analysis, and result generation.
    """

    def __init__(
            self,
            config: Config,
            db_manager: Optional[DatabaseManager] = None,
            ml_predictor: Optional['MLPredictorType'] = None,
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the processor.

        Args:
            config: Application configuration
            db_manager: Database manager for result storage
            ml_predictor: ML predictor for real-time predictions
            logger: Logger instance
        """
        self.config = config
        self.db_manager = db_manager
        self.ml_predictor = ml_predictor
        self.logger = logger or logging.getLogger(__name__)

        # Initialize ML predictor if enabled and not provided
        if self.config.ml.enabled and not self.ml_predictor and HAS_ML:
            self.ml_predictor = MLPredictor(config, logger=self.logger)
            if not self.ml_predictor.initialize():
                self.logger.warning("ML predictor initialization failed")
                self.ml_predictor = None

        # Initialize analyzers
        self.sigma_analyzer = SigmaAnalyzer(config, logger)
        self.linearity_analyzer = LinearityAnalyzer(config, logger)
        self.resistance_analyzer = ResistanceAnalyzer(config, logger)

        # Initialize calculation validator
        validation_level_map = {
            'relaxed': ValidationLevel.RELAXED,
            'standard': ValidationLevel.STANDARD,
            'strict': ValidationLevel.STRICT
        }
        validation_level = validation_level_map.get(
            getattr(config, 'validation_level', 'standard'),
            ValidationLevel.STANDARD
        )
        self.calculation_validator = CalculationValidator(validation_level)
        self.logger.info(f"Initialized calculation validator with {validation_level.value} validation level")

        # Processing state
        self._executor = None
        self._processing_tasks = []
        self._is_processing = False

        # Initialize processing statistics
        self._processing_stats = {
            "files_processed": 0,
            "validation_failures": 0,
            "processing_errors": 0,
            "cache_hits": 0
        }

        # Cache for performance
        self._file_cache = {}
        self._cache_enabled = config.processing.cache_enabled

    async def process_file(
            self,
            file_path: Path,
            output_dir: Optional[Path] = None,
            progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AnalysisResult:
        """
        Process a single Excel file with comprehensive validation.

        Args:
            file_path: Path to Excel file
            output_dir: Output directory for plots
            progress_callback: Progress callback function

        Returns:
            AnalysisResult with complete analysis data and validation status
        """
        file_path = Path(file_path)
        
        if progress_callback:
            progress_callback("Starting file validation...", 0.0)

        # Step 1: Pre-flight file validation
        self.logger.info(f"Starting comprehensive validation for: {file_path.name}")
        
        try:
            file_validation = validate_excel_file(
                file_path=file_path,
                max_file_size_mb=self.config.processing.max_file_size_mb
            )
            
            if not file_validation.is_valid:
                self._processing_stats["validation_failures"] += 1
                error_msg = f"File validation failed: {'; '.join(file_validation.errors)}"
                self.logger.error(error_msg)
                raise ValidationError(error_msg)
            
            # Log validation warnings
            if file_validation.warnings:
                for warning in file_validation.warnings:
                    self.logger.warning(f"File validation warning: {warning}")
            
            # Extract metadata from validation
            detected_system = file_validation.metadata.get('detected_system', 'Unknown')
            file_size_mb = file_validation.metadata.get('file_size_mb', 0)
            sheet_names = file_validation.metadata.get('sheet_names', [])
            
            self.logger.info(f"File validation passed - System: {detected_system}, Size: {file_size_mb:.2f}MB, Sheets: {len(sheet_names)}")
            
        except Exception as e:
            self._processing_stats["validation_failures"] += 1
            self.logger.error(f"File validation error: {e}")
            raise ValidationError(f"File validation failed: {e}")

        if progress_callback:
            progress_callback("File validation complete, checking cache...", 0.1)

        # Step 2: Cache check
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.suffix.lower() in self.config.processing.file_extensions:
            raise ValidationError(f"Unsupported file type: {file_path.suffix}")

        file_hash = calculate_file_hash(file_path)
        if self._cache_enabled and file_hash in self._file_cache:
            self._processing_stats["cache_hits"] += 1
            self.logger.info(f"Using cached result for {file_path.name}")
            return self._file_cache[file_hash]

        self.logger.info(f"Processing file: {file_path.name}")
        start_time = datetime.now()

        try:
            if progress_callback:
                progress_callback("Detecting system and extracting metadata...", 0.2)

            # Step 3: System detection and metadata extraction
            system_type = detect_system_type(file_path)
            self.logger.info(f"Detected system type: {system_type.value}")

            # Extract file metadata
            metadata = await self._extract_file_metadata(file_path, system_type)
            
            # Step 4: Model validation
            if metadata.model:
                model_validation = validate_model_number(metadata.model)
                if not model_validation.is_valid:
                    self.logger.warning(f"Model validation warnings: {model_validation.warnings}")
                if model_validation.errors:
                    self.logger.error(f"Model validation errors: {model_validation.errors}")

            if progress_callback:
                progress_callback("Finding track sheets...", 0.3)

            # Step 5: Track detection and processing
            track_sheets = await self._find_track_sheets(file_path, system_type)
            self.logger.info(f"Found {len(track_sheets)} track sheets")

            if not track_sheets:
                raise DataExtractionError("No valid track sheets found")

            # Process tracks with validation
            tracks = {}
            total_tracks = len(track_sheets)
            
            for idx, (track_id, sheets) in enumerate(track_sheets.items()):
                if progress_callback:
                    progress = 0.4 + (0.5 * idx / total_tracks)
                    progress_callback(f"Processing track {track_id}...", progress)

                try:
                    track_data = await self._process_track_with_validation(
                        file_path, track_id, sheets, system_type, metadata.model
                    )
                    tracks[track_id] = track_data
                    self.logger.debug(f"Successfully processed track {track_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process track {track_id}: {e}")
                    # Continue with other tracks rather than failing completely
                    continue

            if not tracks:
                raise ProcessingError("No tracks could be processed successfully")

            if progress_callback:
                progress_callback("Performing final analysis...", 0.9)

            # Step 6: Create final result with comprehensive validation
            primary_track_id = self._determine_primary_track(tracks)
            primary_track = tracks.get(primary_track_id)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Determine overall status based on validation and analysis
            overall_status = self._determine_overall_status(tracks)
            overall_validation_status = self._determine_overall_validation_status(tracks)
            validation_grade = self._calculate_overall_validation_grade(tracks)

            # Collect validation issues
            validation_issues = []
            validation_warnings = []
            validation_recommendations = []
            
            for track_data in tracks.values():
                if hasattr(track_data, 'validation_warnings'):
                    validation_warnings.extend(track_data.validation_warnings)
                if hasattr(track_data, 'validation_recommendations'):
                    validation_recommendations.extend(track_data.validation_recommendations)

            # Create comprehensive result
            result = AnalysisResult(
                metadata=metadata,
                tracks=tracks,
                primary_track=primary_track,
                overall_status=overall_status,
                processing_time=processing_time,
                overall_validation_status=overall_validation_status,
                validation_grade=validation_grade,
                validation_issues=validation_issues,
                validation_warnings=validation_warnings,
                validation_recommendations=validation_recommendations
            )

            # Step 7: Generate outputs if requested
            if output_dir:
                await self._generate_outputs(result, output_dir, file_path)

            # Cache result
            if self._cache_enabled:
                self._file_cache[file_hash] = result

            # Update statistics
            self._processing_stats["files_processed"] += 1
            
            if progress_callback:
                progress_callback("Processing complete!", 1.0)

            self.logger.info(f"Successfully processed {file_path.name} in {processing_time:.2f}s")
            return result

        except Exception as e:
            self._processing_stats["processing_errors"] += 1
            self.logger.error(f"Processing failed for {file_path.name}: {e}")
            
            if progress_callback:
                progress_callback(f"Processing failed: {str(e)}", 1.0)
            
            raise ProcessingError(f"Failed to process {file_path.name}: {e}")

    async def _process_track_with_validation(
            self,
            file_path: Path,
            track_id: str,
            sheets: Dict[str, str],
            system_type: SystemType,
            model: str
    ) -> TrackData:
        """Process a single track with comprehensive validation."""
        self.logger.debug(f"Processing track {track_id} with validation")
        
        # Extract analysis data
        analysis_data = await self._extract_analysis_data(file_path, sheets, system_type)
        
        # Step 1: Validate analysis data structure
        data_validation = validate_analysis_data(analysis_data, system_type.value)
        
        validation_warnings = []
        validation_recommendations = []
        
        if not data_validation.is_valid:
            # Log but don't fail - some data issues can be handled
            self.logger.warning(f"Data validation issues for track {track_id}: {data_validation.errors}")
            validation_warnings.extend([f"Data: {error}" for error in data_validation.errors])
        
        if data_validation.warnings:
            validation_warnings.extend([f"Data: {warning}" for warning in data_validation.warnings])

        # Step 2: Extract unit properties with validation
        unit_props = await self._extract_unit_properties(file_path, system_type, sheets)
        
        # Validate resistance values if available
        if unit_props.untrimmed_resistance or unit_props.trimmed_resistance:
            resistance_validation = validate_resistance_values(
                unit_props.untrimmed_resistance,
                unit_props.trimmed_resistance
            )
            
            if not resistance_validation.is_valid:
                validation_warnings.extend([f"Resistance: {error}" for error in resistance_validation.errors])
            
            if resistance_validation.warnings:
                validation_warnings.extend([f"Resistance: {warning}" for warning in resistance_validation.warnings])

        # Step 3: Perform core analyses with industry validation
        sigma_analysis = await self._analyze_sigma(analysis_data, unit_props, model)
        linearity_analysis = await self._analyze_linearity(analysis_data, unit_props, model)
        resistance_analysis = await self._analyze_resistance(unit_props)

        # Step 4: Validate sigma values
        if sigma_analysis.sigma_gradient is not None and sigma_analysis.sigma_threshold is not None:
            sigma_validation = self.validate_sigma_values(
                sigma_analysis.sigma_gradient,
                sigma_analysis.sigma_threshold,
                model
            )
            
            if not sigma_validation.is_valid:
                validation_warnings.extend([f"Sigma: {error}" for error in sigma_validation.errors])
            
            if sigma_validation.warnings:
                validation_warnings.extend([f"Sigma: {warning}" for warning in sigma_validation.warnings])

        # Step 5: Additional analyses
        zone_analysis = await self._analyze_zones(analysis_data)
        dynamic_range_analysis = await self._analyze_dynamic_range(analysis_data)
        trim_effectiveness = await self._calculate_trim_effectiveness(
            sigma_analysis, linearity_analysis, resistance_analysis
        )

        # Step 6: ML prediction if available
        failure_prediction = None
        if self.ml_predictor:
            failure_prediction = await self._predict_failure(
                sigma_analysis, linearity_analysis, unit_props
            )

        # Step 7: Determine track-level validation status
        track_validation_status = ValidationStatus.VALIDATED
        if validation_warnings:
            if any("error" in warning.lower() or "fail" in warning.lower() for warning in validation_warnings):
                track_validation_status = ValidationStatus.FAILED
            else:
                track_validation_status = ValidationStatus.WARNING

        # Collect industry validation recommendations
        industry_recommendations = []
        for analysis in [sigma_analysis, linearity_analysis, resistance_analysis]:
            if hasattr(analysis, 'validation_result') and analysis.validation_result:
                if analysis.validation_result.recommendations:
                    industry_recommendations.extend(analysis.validation_result.recommendations)

        validation_recommendations.extend(industry_recommendations)

        # Create track data with comprehensive validation info
        track_data = TrackData(
            track_id=track_id,
            unit_properties=unit_props,
            sigma_analysis=sigma_analysis,
            linearity_analysis=linearity_analysis,
            resistance_analysis=resistance_analysis,
            zone_analysis=zone_analysis,
            dynamic_range_analysis=dynamic_range_analysis,
            trim_effectiveness=trim_effectiveness,
            failure_prediction=failure_prediction
        )

        # Add validation fields to track data
        track_data.validation_warnings = validation_warnings
        track_data.validation_recommendations = validation_recommendations
        track_data.validation_status = track_validation_status

        return track_data

    async def process_batch(
            self,
            file_paths: List[Path],
            output_dir: Optional[Path] = None,
            progress_callback: Optional[Callable[[str, float], None]] = None,
            max_workers: Optional[int] = None
    ) -> Dict[str, AnalysisResult]:
        """
        Process multiple files with comprehensive batch validation.

        Args:
            file_paths: List of file paths to process
            output_dir: Output directory for results
            progress_callback: Progress callback function
            max_workers: Maximum concurrent workers

        Returns:
            Dictionary mapping file paths to analysis results
        """
        if progress_callback:
            progress_callback("Starting batch validation...", 0.0)

        # Step 1: Batch-level validation
        self.logger.info(f"Starting batch processing of {len(file_paths)} files")
        
        batch_validation = BatchValidator.validate_batch(
            file_paths=file_paths,
            max_batch_size=self.config.processing.max_batch_size
        )
        
        if not batch_validation.is_valid:
            error_msg = f"Batch validation failed: {'; '.join(batch_validation.errors)}"
            self.logger.error(error_msg)
            raise ValidationError(error_msg)
        
        # Log batch validation info
        if batch_validation.warnings:
            for warning in batch_validation.warnings:
                self.logger.warning(f"Batch validation: {warning}")
        
        valid_files = batch_validation.metadata.get('valid_files', 0)
        invalid_files = batch_validation.metadata.get('invalid_files', [])
        
        self.logger.info(f"Batch validation: {valid_files} valid files, {len(invalid_files)} invalid files")
        
        # Filter to only valid files for processing
        processable_files = []
        for file_path in file_paths:
            if not any(str(file_path) in invalid['file'] for invalid in invalid_files):
                processable_files.append(file_path)
        
        if not processable_files:
            raise ValidationError("No valid files found in batch after validation")

        if progress_callback:
            progress_callback(f"Processing {len(processable_files)} validated files...", 0.1)

        # Step 2: Process files with limited concurrency
        max_workers = max_workers or self.config.processing.max_workers
        results = {}
        failed_files = []

        # Process in batches to avoid overwhelming the system
        batch_size = min(max_workers, len(processable_files))
        
        for i in range(0, len(processable_files), batch_size):
            batch_files = processable_files[i:i + batch_size]
            base_progress = 0.1 + (0.8 * i / len(processable_files))
            
            # Process batch concurrently
            tasks = []
            for j, file_path in enumerate(batch_files):
                def make_progress_callback(file_idx):
                    def callback(message, progress):
                        overall_progress = base_progress + (0.8 / len(processable_files) * (progress))
                        if progress_callback:
                            progress_callback(f"Processing {file_path.name}: {message}", overall_progress)
                    return callback

                task = self.process_file(
                    file_path=file_path,
                    output_dir=output_dir,
                    progress_callback=make_progress_callback(j)
                )
                tasks.append((file_path, task))

            # Wait for batch completion
            for file_path, task in tasks:
                try:
                    result = await task
                    results[str(file_path)] = result
                    self.logger.info(f"Successfully processed {file_path.name}")
                except Exception as e:
                    failed_files.append((str(file_path), str(e)))
                    self.logger.error(f"Failed to process {file_path.name}: {e}")

        # Step 3: Generate batch summary
        if progress_callback:
            progress_callback("Generating batch summary...", 0.95)

        successful_count = len(results)
        failed_count = len(failed_files)
        
        self.logger.info(f"Batch processing complete: {successful_count} successful, {failed_count} failed")
        
        if failed_files:
            self.logger.warning("Failed files:")
            for file_path, error in failed_files:
                self.logger.warning(f"  {file_path}: {error}")

        if progress_callback:
            progress_callback("Batch processing complete!", 1.0)

        return results

    async def _extract_metadata(self, file_path: Path) -> FileMetadata:
        """Extract file metadata."""
        # Parse filename for model and serial
        filename = file_path.stem
        parts = filename.split('_')

        model = parts[0] if parts else "Unknown"
        serial = parts[1] if len(parts) > 1 else "Unknown"
        
        # Check for System B multi-track identifier (TA, TB, TC, etc.)
        track_identifier = None
        for part in parts:
            if re.match(r'^T[A-Z]$', part):  # Matches TA, TB, TC, etc.
                track_identifier = part
                break
        
        if track_identifier:
            self.logger.info(f"Detected System B multi-track file with identifier: {track_identifier}")

        # Extract date and time from filename (primary source)
        file_date = extract_datetime_from_filename(filename)
        
        if file_date:
            self.logger.info(f"Successfully extracted trim date {file_date} from filename '{filename}'")
        else:
            # Fallback to file modification time only if filename parsing completely fails
            self.logger.warning(f"Could not extract date from filename '{filename}', using file modification time as fallback")
            stat = file_path.stat()
            file_date = datetime.fromtimestamp(stat.st_mtime)
            self.logger.info(f"Using file modification date: {file_date}")

        metadata = FileMetadata(
            filename=file_path.name,
            file_path=file_path,
            file_date=file_date,
            model=model,
            serial=serial,
            system=SystemType.UNKNOWN,  # Will be detected later
            track_identifier=track_identifier  # Store track ID for System B
        )

        return metadata

    async def _extract_file_metadata(self, file_path: Path, system_type: SystemType) -> FileMetadata:
        """Extract file metadata with system type."""
        metadata = await self._extract_metadata(file_path)
        metadata.system = system_type
        return metadata

    async def _find_track_sheets(self, file_path: Path, system_type: SystemType) -> Dict[str, Dict[str, str]]:
        """Find track sheets based on system type."""
        if system_type == SystemType.SYSTEM_A:
            # Multi-track system - multiple sheets in one file
            track_ids = await self._find_system_a_tracks(file_path)
            track_sheets = {}
            
            try:
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names
            except Exception as e:
                self.logger.error(f"Could not read Excel file: {e}")
                return {}
            
            for track_id in track_ids:
                sheets = {}
                for sheet in sheet_names:
                    if track_id in sheet or (track_id == "TRK1" and ("1 0" in sheet or "1_0" in sheet or "SEC1" in sheet)):
                        if " 0" in sheet or "_0" in sheet or sheet.endswith(" 0"):
                            sheets['untrimmed'] = sheet
                        elif "TRM" in sheet or "TRIM" in sheet.upper():
                            sheets['trimmed'] = sheet
                
                if sheets:
                    track_sheets[track_id] = sheets
            
            return track_sheets
        else:
            # System B - single track with predefined sheets
            return {
                "default": {
                    'untrimmed': 'test',
                    'trimmed': 'Lin Error'
                }
            }

    async def _extract_analysis_data(self, file_path: Path, sheets: Dict[str, str], system_type: SystemType) -> Dict[str, Any]:
        """Extract analysis data from sheets."""
        loop = asyncio.get_event_loop()

        # Extract data from untrimmed sheet
        untrimmed_data = await loop.run_in_executor(
            None, self._extract_trim_data, file_path, sheets.get('untrimmed'), system_type.value[0]
        )

        # Extract data from trimmed sheet if available
        trimmed_data = None
        if 'trimmed' in sheets:
            trimmed_data = await loop.run_in_executor(
                None, self._extract_trim_data, file_path, sheets['trimmed'], system_type.value[0]
            )

        # Use trimmed data as primary if available, otherwise untrimmed
        data_source = trimmed_data if trimmed_data and trimmed_data.get('positions') else untrimmed_data

        return {
            'sheets': sheets,
            'untrimmed_data': untrimmed_data,
            'trimmed_data': trimmed_data,
            'positions': data_source.get('positions', []),
            'errors': data_source.get('errors', []),
            'upper_limits': data_source.get('upper_limits', []),
            'lower_limits': data_source.get('lower_limits', []),
            'travel_length': data_source.get('travel_length', 0)
        }

    async def _generate_outputs(self, result: AnalysisResult, output_dir: Path, file_path: Path):
        """Generate output files and plots."""
        # Generate plots if enabled
        if self.config.processing.generate_plots:
            await self._generate_plots(result, output_dir)
        
        # Generate summary report
        await self._generate_summary_report(result, output_dir, file_path)

    async def _generate_summary_report(self, result: AnalysisResult, output_dir: Path, file_path: Path):
        """Generate a summary report for the analysis."""
        try:
            # Create summary data
            summary_data = []
            flat_data = result.to_flat_dict()
            summary_data.append(flat_data)

            df = pd.DataFrame(summary_data)

            # Save to Excel
            excel_path = output_dir / f"{file_path.stem}_analysis_report.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Analysis Summary', index=False)

                # Add validation details if available
                if result.validation_warnings or result.validation_recommendations:
                    validation_data = []
                    
                    for warning in result.validation_warnings:
                        validation_data.append({'Type': 'Warning', 'Message': warning})
                    
                    for recommendation in result.validation_recommendations:
                        validation_data.append({'Type': 'Recommendation', 'Message': recommendation})
                    
                    if validation_data:
                        validation_df = pd.DataFrame(validation_data)
                        validation_df.to_excel(writer, sheet_name='Validation Details', index=False)

            self.logger.info(f"Summary report saved to {excel_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")

    def _determine_primary_track(self, tracks: Dict[str, TrackData]) -> str:
        """Determine the primary track ID."""
        if not tracks:
            return "default"
        
        # Prefer TRK1 for System A, otherwise return first track
        if "TRK1" in tracks:
            return "TRK1"
        elif "default" in tracks:
            return "default"
        else:
            return list(tracks.keys())[0]

    async def _generate_plots(
            self,
            result: AnalysisResult,
            output_dir: Path
    ) -> None:
        """Generate analysis plots."""
        loop = asyncio.get_event_loop()
        
        for track_id, track_data in result.tracks.items():
            try:
                # Run blocking plot creation in thread pool
                plot_path = await loop.run_in_executor(
                    None,
                    create_analysis_plot,
                    track_data,
                    output_dir,
                    f"{result.metadata.filename}_{track_id}",
                    self.config.processing.plot_dpi
                )
                track_data.plot_path = plot_path
            except Exception as e:
                self.logger.error(f"Plot generation failed for {track_id}: {e}")

    async def _add_ml_predictions(self, result: AnalysisResult) -> None:
        """Add ML predictions to result."""
        if not self.ml_predictor:
            return

        try:
            # Get predictions for primary track
            flat_data = result.to_flat_dict()
            predictions = await self.ml_predictor.predict(flat_data)

            # Add predictions to result metadata
            if predictions:
                result.ml_predictions = predictions

                # Update risk categories based on ML
                if 'risk_assessment' in predictions:
                    for track_id, assessment in predictions['risk_assessment'].items():
                        if track_id in result.tracks:
                            track = result.tracks[track_id]
                            if track.failure_prediction:
                                track.failure_prediction.risk_category = RiskCategory(
                                    assessment.get('risk_category', 'UNKNOWN')
                                )

        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")

    async def _generate_batch_report(
            self,
            results: List[AnalysisResult],
            output_dir: Path
    ) -> None:
        """Generate summary report for batch processing."""
        if not results:
            return

        # Create summary DataFrame
        summary_data = []
        for result in results:
            flat_data = result.to_flat_dict()
            summary_data.append(flat_data)

        df = pd.DataFrame(summary_data)

        # Save to Excel
        excel_path = output_dir / "batch_summary.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Summary', index=False)

            # Add statistics sheet
            stats = pd.DataFrame({
                'Metric': [
                    'Total Files',
                    'Pass Rate (%)',
                    'Average Sigma Gradient',
                    'High Risk Units'
                ],
                'Value': [
                    len(results),
                    (df['overall_status'] == 'Pass').mean() * 100,
                    df['sigma_gradient'].mean(),
                    (df['risk_category'] == 'HIGH').sum()
                ]
            })
            stats.to_excel(writer, sheet_name='Statistics', index=False)

        self.logger.info(f"Batch report saved to {excel_path}")

    async def find_related_tracks(
            self,
            file_path: Path,
            metadata: FileMetadata
    ) -> List[Path]:
        """
        Find related track files for System B multi-track analysis.
        
        For System B, multi-track units have separate files like:
        - Model_Serial_TA_Date.xls (Track A)
        - Model_Serial_TB_Date.xls (Track B)
        """
        if metadata.system != SystemType.SYSTEM_B or not metadata.track_identifier:
            return []

        related_files = []
        base_dir = file_path.parent
        
        # Create pattern to find related files
        # Remove the track identifier and create a pattern
        filename = file_path.stem
        
        # Replace the track identifier with a wildcard pattern
        pattern_base = filename.replace(metadata.track_identifier, 'T*')
        
        self.logger.debug(f"Looking for related tracks with pattern: {pattern_base}")
        
        # Search for files matching the pattern
        for potential_file in base_dir.glob(f"{pattern_base}.xls*"):
            if potential_file != file_path:  # Don't include the original file
                # Verify it's actually a track file
                potential_parts = potential_file.stem.split('_')
                for part in potential_parts:
                    if re.match(r'^T[A-Z]$', part) and part != metadata.track_identifier:
                        related_files.append(potential_file)
                        break
        
        self.logger.info(f"Found {len(related_files)} related track files for {file_path.name}")
        return related_files

    async def analyze_multi_track_unit(
            self,
            primary_file: Path,
            output_dir: Optional[Path] = None,
            progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a complete multi-track unit by processing all related files.
        
        This method processes all tracks for a unit and provides comparison analysis.
        """
        try:
            # Process the primary file first
            primary_result = await self.process_file(primary_file, output_dir, progress_callback)
            
            # Find related track files
            related_files = await self.find_related_tracks(primary_file, primary_result.metadata)
            
            if not related_files:
                self.logger.info(f"No related tracks found for {primary_file.name}, treating as single track")
                return {
                    'unit_id': f"{primary_result.metadata.model}_{primary_result.metadata.serial}",
                    'tracks': {primary_result.metadata.track_identifier or 'default': primary_result},
                    'comparison': None
                }
            
            # Process all related tracks
            all_track_results = {
                primary_result.metadata.track_identifier or 'default': primary_result
            }
            
            for related_file in related_files:
                try:
                    related_result = await self.process_file(related_file, output_dir, progress_callback)
                    track_id = related_result.metadata.track_identifier or 'related'
                    all_track_results[track_id] = related_result
                except Exception as e:
                    self.logger.error(f"Failed to process related file {related_file.name}: {e}")
            
            # Perform multi-track comparison analysis
            comparison = await self._compare_tracks(all_track_results)
            
            unit_analysis = {
                'unit_id': f"{primary_result.metadata.model}_{primary_result.metadata.serial}",
                'tracks': all_track_results,
                'comparison': comparison,
                'overall_status': self._determine_unit_status(all_track_results),
                'has_multi_track_issues': comparison.get('has_issues', False) if comparison else False
            }
            
            self.logger.info(f"Completed multi-track analysis for unit {unit_analysis['unit_id']} with {len(all_track_results)} tracks")
            return unit_analysis
            
        except Exception as e:
            self.logger.error(f"Multi-track analysis failed for {primary_file.name}: {e}")
            return None

    async def _compare_tracks(self, track_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare multiple tracks within a unit to identify inconsistencies.
        
        This is critical for multi-track quality analysis.
        """
        if len(track_results) < 2:
            return {'comparison_performed': False, 'reason': 'Insufficient tracks for comparison'}
        
        comparison = {
            'comparison_performed': True,
            'track_count': len(track_results),
            'track_ids': list(track_results.keys()),
            'sigma_comparison': {},
            'linearity_comparison': {},
            'resistance_comparison': {},
            'consistency_issues': [],
            'has_issues': False
        }
        
        # Extract track data for comparison
        sigma_gradients = {}
        linearity_errors = {}
        resistance_changes = {}
        
        for track_id, result in track_results.items():
            primary_track = result.primary_track
            if primary_track:
                sigma_gradients[track_id] = primary_track.sigma_analysis.sigma_gradient
                linearity_errors[track_id] = primary_track.linearity_analysis.final_linearity_error_shifted
                if primary_track.unit_properties.resistance_change_percent:
                    resistance_changes[track_id] = primary_track.unit_properties.resistance_change_percent
        
        # Sigma gradient comparison
        if len(sigma_gradients) >= 2:
            sigma_values = list(sigma_gradients.values())
            sigma_mean = np.mean(sigma_values)
            sigma_std = np.std(sigma_values)
            sigma_range = max(sigma_values) - min(sigma_values)
            
            comparison['sigma_comparison'] = {
                'values': sigma_gradients,
                'mean': sigma_mean,
                'std_dev': sigma_std,
                'range': sigma_range,
                'cv_percent': (sigma_std / sigma_mean * 100) if sigma_mean > 0 else 0
            }
            
            # Flag if coefficient of variation > 10% (tracks should be similar)
            if comparison['sigma_comparison']['cv_percent'] > 10:
                comparison['consistency_issues'].append(
                    f"High sigma gradient variation between tracks (CV: {comparison['sigma_comparison']['cv_percent']:.1f}%)"
                )
        
        # Linearity comparison
        if len(linearity_errors) >= 2:
            linearity_values = list(linearity_errors.values())
            linearity_mean = np.mean(linearity_values)
            linearity_std = np.std(linearity_values)
            linearity_range = max(linearity_values) - min(linearity_values)
            
            comparison['linearity_comparison'] = {
                'values': linearity_errors,
                'mean': linearity_mean,
                'std_dev': linearity_std,
                'range': linearity_range,
                'cv_percent': (linearity_std / linearity_mean * 100) if linearity_mean > 0 else 0
            }
            
            # Flag if coefficient of variation > 15%
            if comparison['linearity_comparison']['cv_percent'] > 15:
                comparison['consistency_issues'].append(
                    f"High linearity error variation between tracks (CV: {comparison['linearity_comparison']['cv_percent']:.1f}%)"
                )
        
        # Resistance change comparison
        if len(resistance_changes) >= 2:
            resistance_values = list(resistance_changes.values())
            resistance_mean = np.mean(resistance_values)
            resistance_std = np.std(resistance_values)
            resistance_range = max(resistance_values) - min(resistance_values)
            
            comparison['resistance_comparison'] = {
                'values': resistance_changes,
                'mean': resistance_mean,
                'std_dev': resistance_std,
                'range': resistance_range
            }
            
            # Flag if range > 5% (resistance should change similarly)
            if resistance_range > 5.0:
                comparison['consistency_issues'].append(
                    f"Large resistance change variation between tracks (Range: {resistance_range:.1f}%)"
                )
        
        # Check for pass/fail inconsistencies
        track_statuses = [result.overall_status.value for result in track_results.values()]
        unique_statuses = set(track_statuses)
        
        if len(unique_statuses) > 1:
            comparison['consistency_issues'].append(
                f"Inconsistent pass/fail status between tracks: {unique_statuses}"
            )
        
        # Set flag if any issues found
        comparison['has_issues'] = len(comparison['consistency_issues']) > 0
        
        self.logger.info(f"Track comparison completed: {len(comparison['consistency_issues'])} issues found")
        return comparison

    def _determine_unit_status(self, track_results: Dict[str, Any]) -> AnalysisStatus:
        """Determine overall unit status from all track results."""
        statuses = [result.overall_status for result in track_results.values()]
        
        # If any track fails, unit fails
        if any(status == AnalysisStatus.FAIL for status in statuses):
            return AnalysisStatus.FAIL
        
        # If any track has warning, unit has warning
        if any(status == AnalysisStatus.WARNING for status in statuses):
            return AnalysisStatus.WARNING
        
        # If all tracks pass, unit passes
        if all(status == AnalysisStatus.PASS for status in statuses):
            return AnalysisStatus.PASS
        
        return AnalysisStatus.ERROR

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics including validation metrics."""
        stats = self._processing_stats.copy()
        stats.update({
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._file_cache),
            "validation_success_rate": (
                (stats["files_processed"] - stats["validation_failures"]) / 
                max(stats["files_processed"], 1) * 100
            ),
            "processing_success_rate": (
                (stats["files_processed"] - stats["processing_errors"]) / 
                max(stats["files_processed"], 1) * 100
            )
        })
        return stats

    def _determine_overall_validation_status(self, tracks: Dict[str, TrackData]) -> ValidationStatus:
        """Determine overall validation status from all tracks."""
        if not tracks:
            return ValidationStatus.NOT_VALIDATED
        
        statuses = []
        for track_data in tracks.values():
            if hasattr(track_data, 'validation_status'):
                statuses.append(track_data.validation_status)
            # Also check individual analysis validation statuses
            for analysis in [track_data.sigma_analysis, track_data.linearity_analysis, track_data.resistance_analysis]:
                if hasattr(analysis, 'validation_status'):
                    statuses.append(analysis.validation_status)
        
        if not statuses:
            return ValidationStatus.NOT_VALIDATED
        
        # Priority: FAILED > WARNING > VALIDATED > NOT_VALIDATED
        if ValidationStatus.FAILED in statuses:
            return ValidationStatus.FAILED
        elif ValidationStatus.WARNING in statuses:
            return ValidationStatus.WARNING
        elif ValidationStatus.VALIDATED in statuses:
            return ValidationStatus.VALIDATED
        else:
            return ValidationStatus.NOT_VALIDATED

    def _calculate_overall_validation_grade(self, tracks: Dict[str, TrackData]) -> str:
        """Calculate overall validation grade from all tracks."""
        if not tracks:
            return "Not Available"
        
        grades = []
        grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0, "F": 0}
        
        for track_data in tracks.values():
            # Collect grades from different analyses
            for analysis in [track_data.sigma_analysis, track_data.linearity_analysis, track_data.resistance_analysis]:
                if hasattr(analysis, 'validation_result') and analysis.validation_result:
                    grade = analysis.validation_result.validation_grade
                    if grade and grade[0] in grade_values:
                        grades.append(grade_values[grade[0]])
        
        if not grades:
            return "Not Available"
        
        # Calculate average grade
        avg_grade = sum(grades) / len(grades)
        
        # Convert back to letter grade
        if avg_grade >= 3.5:
            return "A - Excellent"
        elif avg_grade >= 2.5:
            return "B - Good"
        elif avg_grade >= 1.5:
            return "C - Acceptable"
        elif avg_grade >= 0.5:
            return "D - Poor"
        else:
            return "F - Failed"

    # Add validation utility methods referenced in track processing
    def validate_sigma_values(self, sigma_gradient: float, sigma_threshold: float, model: str) -> UtilsValidationResult:
        """Validate sigma values against model-specific criteria."""
        errors = []
        warnings = []
        metadata = {}
        
        # Basic validation
        if sigma_gradient is None or sigma_threshold is None:
            errors.append("Missing sigma values")
        
        if sigma_gradient and sigma_gradient < 0:
            errors.append("Sigma gradient cannot be negative")
        
        if sigma_threshold and sigma_threshold < 0:
            errors.append("Sigma threshold cannot be negative")
        
        # Model-specific validation
        if model and sigma_gradient:
            if "PRECISION" in model.upper() and sigma_gradient > 0.1:
                warnings.append(f"High sigma gradient ({sigma_gradient:.3f}) for precision model")
            elif "COMMERCIAL" in model.upper() and sigma_gradient > 1.0:
                warnings.append(f"High sigma gradient ({sigma_gradient:.3f}) for commercial model")
        
        # Relationship validation
        if sigma_gradient and sigma_threshold:
            if sigma_gradient > sigma_threshold:
                warnings.append("Sigma gradient exceeds threshold - indicates poor process control")
            
            ratio = sigma_gradient / sigma_threshold if sigma_threshold > 0 else float('inf')
            metadata['sigma_ratio'] = ratio
            
            if ratio > 0.8:
                warnings.append(f"Sigma ratio ({ratio:.2f}) approaching threshold - monitor closely")
        
        return UtilsValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )

    async def _find_system_a_tracks(self, file_path: Path) -> List[str]:
        """Find available tracks in System A file."""
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
        except Exception as e:
            self.logger.warning(f"Could not read Excel file to find tracks: {e}")
            return ["default"]

        # Look for track patterns
        tracks = []
        for sheet in sheet_names:
            if "TRK1" in sheet and "TRK1" not in tracks:
                tracks.append("TRK1")
            elif "TRK2" in sheet and "TRK2" not in tracks:
                tracks.append("TRK2")

        return tracks if tracks else ["default"]

    def _extract_trim_data(
            self,
            file_path: Path,
            sheet_name: str,
            system: str
    ) -> Dict[str, Any]:
        """Extract trim data from a sheet."""
        try:
            df = read_excel_sheet(file_path, sheet_name)

            # Find data columns based on system
            columns = find_data_columns(df, system)
            self.logger.debug(f"Found columns for {system}: {columns}")

            if not columns:
                return {}

            # Extract data and convert to numeric
            position_series = pd.to_numeric(df.iloc[:, columns['position']], errors='coerce')
            error_series = pd.to_numeric(df.iloc[:, columns['error']], errors='coerce')
            
            # Extract upper and lower limits if available
            upper_limits = []
            lower_limits = []
            if 'upper_limit' in columns and 'lower_limit' in columns:
                self.logger.debug(f"Attempting to extract limits from columns {columns['upper_limit']} and {columns['lower_limit']}")
                upper_series = pd.to_numeric(df.iloc[:, columns['upper_limit']], errors='coerce')
                lower_series = pd.to_numeric(df.iloc[:, columns['lower_limit']], errors='coerce')
                upper_limits = upper_series.dropna().tolist()
                lower_limits = lower_series.dropna().tolist()
                self.logger.debug(f"Extracted limits - Upper: {len(upper_limits)} values, Lower: {len(lower_limits)} values")
                if upper_limits and lower_limits:
                    self.logger.debug(f"Sample upper limits: {upper_limits[:5]}")
                    self.logger.debug(f"Sample lower limits: {lower_limits[:5]}")
            else:
                self.logger.warning(f"Limit columns not found in system {system} columns: {columns}")
            
            # Drop NaN values
            positions = position_series.dropna().tolist()
            errors = error_series.dropna().tolist()

            # Ensure same length
            min_len = min(len(positions), len(errors))
            positions = positions[:min_len]
            errors = errors[:min_len]
            
            # Ensure limits have same length
            if upper_limits:
                upper_limits = upper_limits[:min_len]
            if lower_limits:
                lower_limits = lower_limits[:min_len]

            # Calculate travel length
            travel_length = max(positions) - min(positions) if positions else 0

            result = {
                'positions': positions,
                'errors': errors,
                'upper_limits': upper_limits,
                'lower_limits': lower_limits,
                'travel_length': travel_length
            }
            
            self.logger.debug(f"Final extracted data: {len(positions)} positions, {len(errors)} errors, {len(upper_limits)} upper limits, {len(lower_limits)} lower limits")
            
            return result

        except Exception as e:
            self.logger.error(f"Error extracting trim data: {e}")
            return {}

    async def _extract_unit_properties(
            self,
            file_path: Path,
            system_type: SystemType,
            sheets: Dict[str, str]
    ) -> UnitProperties:
        """Extract unit properties."""
        loop = asyncio.get_event_loop()

        def extract():
            props = UnitProperties()

            # Cell locations based on system
            if system_type == SystemType.SYSTEM_A:
                unit_length_cell = "B26"
                resistance_cell = "B10"
            else:
                unit_length_cell = "K1"
                resistance_cell = "R1"

            # Log what we're looking for
            self.logger.debug(f"Extracting properties for {system_type.value} from sheets: {sheets}")

            # Extract unit length
            if 'untrimmed' in sheets:
                unit_length = extract_cell_value(
                    file_path, sheets['untrimmed'], unit_length_cell
                )
                if unit_length and isinstance(unit_length, (int, float)):
                    props.unit_length = float(unit_length)
                    self.logger.debug(f"Found unit_length: {props.unit_length}")

            # Extract resistances
            if 'untrimmed' in sheets:
                untrimmed_r = extract_cell_value(
                    file_path, sheets['untrimmed'], resistance_cell
                )
                if untrimmed_r and isinstance(untrimmed_r, (int, float)):
                    props.untrimmed_resistance = float(untrimmed_r)
                    self.logger.debug(f"Found untrimmed_resistance: {props.untrimmed_resistance}")

            if 'trimmed' in sheets:
                trimmed_r = extract_cell_value(
                    file_path, sheets['trimmed'], resistance_cell
                )
                if trimmed_r and isinstance(trimmed_r, (int, float)):
                    props.trimmed_resistance = float(trimmed_r)
                    self.logger.debug(f"Found trimmed_resistance: {props.trimmed_resistance}")

            return props

        return await loop.run_in_executor(None, extract)

    async def _analyze_sigma(
            self,
            data: Dict[str, Any],
            unit_props: UnitProperties,
            model: str
    ) -> SigmaAnalysis:
        """Perform sigma gradient analysis."""
        # Prepare data dictionary for analyzer
        analysis_data = {
            'positions': data.get('positions', []),
            'errors': data.get('errors', []),
            'upper_limits': data.get('upper_limits', []),
            'lower_limits': data.get('lower_limits', []),
            'model': model,
            'unit_length': unit_props.unit_length,
            'travel_length': data.get('travel_length', 0)
        }
        
        sigma_analysis = self.sigma_analyzer.analyze(analysis_data)
        
        # Add industry standard validation
        try:
            validation_result = self.calculation_validator.validate_sigma_gradient(
                calculated_sigma=sigma_analysis.sigma_gradient,
                position_data=analysis_data['positions'],
                error_data=analysis_data['errors'],
                target_function="linear"
            )
            
            # Convert validator ValidationResult to our models.ValidationResult
            from laser_trim_analyzer.core.models import ValidationResult as ModelValidationResult, ValidationStatus
            
            # Determine validation status
            if validation_result.is_valid:
                if validation_result.warnings:
                    validation_status = ValidationStatus.WARNING
                else:
                    validation_status = ValidationStatus.VALIDATED
            else:
                validation_status = ValidationStatus.FAILED
            
            model_validation_result = ModelValidationResult(
                calculation_type="sigma_gradient",
                is_valid=validation_result.is_valid,
                expected_value=validation_result.expected_value,
                actual_value=validation_result.actual_value,
                deviation_percent=validation_result.deviation_percent,
                tolerance_used=validation_result.tolerance_used,
                standard_reference=validation_result.standard_reference,
                warnings=validation_result.warnings,
                recommendations=validation_result.recommendations,
                validation_status=validation_status
            )
            
            sigma_analysis.validation_result = model_validation_result
            sigma_analysis.validation_status = validation_status
            
            self.logger.info(f"Sigma validation: {validation_status.value} (Grade: {model_validation_result.validation_grade})")
            
        except Exception as e:
            self.logger.error(f"Sigma validation failed: {e}")
            sigma_analysis.validation_status = ValidationStatus.FAILED
        
        return sigma_analysis

    async def _analyze_linearity(
            self,
            data: Dict[str, Any],
            unit_props: UnitProperties,
            model: str
    ) -> LinearityAnalysis:
        """Perform linearity analysis."""
        # Prepare data dictionary for analyzer
        analysis_data = {
            'positions': data.get('positions', []),
            'errors': data.get('errors', []),
            'upper_limits': data.get('upper_limits', []),
            'lower_limits': data.get('lower_limits', []),
            # Don't pass linearity_spec - let the analyzer calculate it from limits
            # 'linearity_spec': sigma_analysis.sigma_threshold
        }
        
        linearity_analysis = self.linearity_analyzer.analyze(analysis_data)
        
        # Add industry standard validation
        try:
            # For linearity validation, we need resistance data
            # Use error data as proxy for resistance deviations
            positions = analysis_data.get('positions', [])
            errors = analysis_data.get('errors', [])
            
            if positions and errors:
                # Create synthetic resistance data from position (assuming linear pot)
                resistance_data = [(pos * 100) + err for pos, err in zip(positions, errors)]
                
                validation_result = self.calculation_validator.validate_linearity_error(
                    calculated_linearity=linearity_analysis.final_linearity_error_shifted,
                    position_data=positions,
                    resistance_data=resistance_data
                )
                
                # Convert to model ValidationResult
                from laser_trim_analyzer.core.models import ValidationResult as ModelValidationResult, ValidationStatus
                
                # Determine validation status
                if validation_result.is_valid:
                    if validation_result.warnings:
                        validation_status = ValidationStatus.WARNING
                    else:
                        validation_status = ValidationStatus.VALIDATED
                else:
                    validation_status = ValidationStatus.FAILED
                
                model_validation_result = ModelValidationResult(
                    calculation_type="linearity_error",
                    is_valid=validation_result.is_valid,
                    expected_value=validation_result.expected_value,
                    actual_value=validation_result.actual_value,
                    deviation_percent=validation_result.deviation_percent,
                    tolerance_used=validation_result.tolerance_used,
                    standard_reference=validation_result.standard_reference,
                    warnings=validation_result.warnings,
                    recommendations=validation_result.recommendations,
                    validation_status=validation_status
                )
                
                linearity_analysis.validation_result = model_validation_result
                linearity_analysis.validation_status = validation_status
                
                self.logger.info(f"Linearity validation: {validation_status.value} (Grade: {model_validation_result.validation_grade})")
            
        except Exception as e:
            self.logger.error(f"Linearity validation failed: {e}")
            linearity_analysis.validation_status = ValidationStatus.FAILED
        
        return linearity_analysis

    async def _analyze_resistance(
            self,
            unit_props: UnitProperties
    ) -> ResistanceAnalysis:
        """Perform resistance analysis."""
        # Prepare data dictionary for analyzer
        analysis_data = {
            'untrimmed_resistance': unit_props.untrimmed_resistance,
            'trimmed_resistance': unit_props.trimmed_resistance
        }
        
        resistance_analysis = self.resistance_analyzer.analyze(analysis_data)
        
        # Add industry standard validation
        try:
            if (unit_props.untrimmed_resistance and unit_props.trimmed_resistance and 
                unit_props.untrimmed_resistance > 0):
                
                # Calculate validation using resistance change
                calculated_resistance = unit_props.trimmed_resistance
                
                # For validation, we need geometric parameters
                # Use default values if not available
                length = 10.0  # mm - default length
                width = 1.0    # mm - default width
                sheet_resistance = unit_props.untrimmed_resistance / (length / width)  # Calculate sheet resistance
                
                validation_result = self.calculation_validator.validate_resistance_calculation(
                    calculated_resistance=calculated_resistance,
                    length=length,
                    width=width,
                    sheet_resistance=sheet_resistance
                )
                
                # Convert to model ValidationResult
                from laser_trim_analyzer.core.models import ValidationResult as ModelValidationResult, ValidationStatus
                
                # Determine validation status
                if validation_result.is_valid:
                    if validation_result.warnings:
                        validation_status = ValidationStatus.WARNING
                    else:
                        validation_status = ValidationStatus.VALIDATED
                else:
                    validation_status = ValidationStatus.FAILED
                
                model_validation_result = ModelValidationResult(
                    calculation_type="resistance_calculation",
                    is_valid=validation_result.is_valid,
                    expected_value=validation_result.expected_value,
                    actual_value=validation_result.actual_value,
                    deviation_percent=validation_result.deviation_percent,
                    tolerance_used=validation_result.tolerance_used,
                    standard_reference=validation_result.standard_reference,
                    warnings=validation_result.warnings,
                    recommendations=validation_result.recommendations,
                    validation_status=validation_status
                )
                
                resistance_analysis.validation_result = model_validation_result
                resistance_analysis.validation_status = validation_status
                
                self.logger.info(f"Resistance validation: {validation_status.value} (Grade: {model_validation_result.validation_grade})")
            
        except Exception as e:
            self.logger.error(f"Resistance validation failed: {e}")
            resistance_analysis.validation_status = ValidationStatus.FAILED
        
        return resistance_analysis

    async def _analyze_zones(self, data: Dict[str, Any]) -> Optional[ZoneAnalysis]:
        """Analyze position zones for consistency."""
        positions = data.get('positions', [])
        errors = data.get('errors', [])
        
        if not positions or not errors or len(positions) < 10:
            return None
        
        # Simple zone analysis - divide into thirds
        n_zones = 3
        zone_size = len(positions) // n_zones
        
        zones = []
        for i in range(n_zones):
            start_idx = i * zone_size
            end_idx = (i + 1) * zone_size if i < n_zones - 1 else len(positions)
            
            zone_errors = errors[start_idx:end_idx]
            zone_rms = np.sqrt(np.mean(np.square(zone_errors))) if zone_errors else 0
            
            zones.append({
                'zone_id': f"Zone_{i+1}",
                'start_position': positions[start_idx],
                'end_position': positions[end_idx-1],
                'rms_error': zone_rms,
                'point_count': len(zone_errors)
            })
        
        return ZoneAnalysis(
            zones=zones,
            zone_consistency=max(z['rms_error'] for z in zones) / min(z['rms_error'] for z in zones) if zones else 1.0
        )

    async def _analyze_dynamic_range(self, data: Dict[str, Any]) -> Optional[DynamicRangeAnalysis]:
        """Analyze dynamic range characteristics."""
        positions = data.get('positions', [])
        errors = data.get('errors', [])
        
        if not positions or not errors:
            return None
        
        return DynamicRangeAnalysis(
            position_range=max(positions) - min(positions),
            error_range=max(errors) - min(errors),
            signal_to_noise_ratio=abs(np.mean(errors)) / np.std(errors) if np.std(errors) > 0 else 0
        )

    async def _calculate_trim_effectiveness(
            self,
            sigma_analysis: SigmaAnalysis,
            linearity_analysis: LinearityAnalysis,
            resistance_analysis: ResistanceAnalysis
    ) -> TrimEffectiveness:
        """Calculate overall trim effectiveness."""
        # Calculate improvement metrics
        sigma_improvement = 0.0
        if hasattr(sigma_analysis, 'improvement_percent'):
            sigma_improvement = sigma_analysis.improvement_percent
        
        linearity_improvement = 0.0
        if hasattr(linearity_analysis, 'improvement_percent'):
            linearity_improvement = linearity_analysis.improvement_percent
        
        resistance_change = 0.0
        if resistance_analysis.resistance_change_percent:
            resistance_change = abs(resistance_analysis.resistance_change_percent)
        
        # Overall effectiveness score
        overall_effectiveness = (sigma_improvement + linearity_improvement) / 2.0
        
        return TrimEffectiveness(
            improvement_percent=overall_effectiveness,
            sigma_improvement=sigma_improvement,
            linearity_improvement=linearity_improvement,
            resistance_stability=100.0 - resistance_change  # Stability as inverse of change
        )

    async def _predict_failure(
            self,
            sigma_analysis: SigmaAnalysis,
            linearity_analysis: LinearityAnalysis,
            unit_props: UnitProperties
    ) -> Optional[FailurePrediction]:
        """Predict failure probability using ML if available."""
        if not self.ml_predictor:
            return self._calculate_failure_prediction(sigma_analysis, linearity_analysis, unit_props)
        
        try:
            # Prepare features for ML prediction
            features = {
                'sigma_gradient': sigma_analysis.sigma_gradient,
                'sigma_ratio': sigma_analysis.sigma_ratio,
                'linearity_error': linearity_analysis.final_linearity_error_shifted,
                'resistance_change': unit_props.resistance_change_percent or 0.0
            }
            
            # Get ML prediction
            ml_result = await self.ml_predictor.predict(features)
            
            if ml_result and 'failure_probability' in ml_result:
                return FailurePrediction(
                    failure_probability=ml_result['failure_probability'],
                    risk_category=RiskCategory(ml_result.get('risk_category', 'MEDIUM')),
                    gradient_margin=sigma_analysis.gradient_margin,
                    contributing_factors=ml_result.get('contributing_factors', {}),
                    ml_confidence=ml_result.get('confidence', 0.0)
                )
        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")
        
        # Fallback to traditional calculation
        return self._calculate_failure_prediction(sigma_analysis, linearity_analysis, unit_props)

    def _calculate_failure_prediction(
            self,
            sigma_analysis: SigmaAnalysis,
            linearity_analysis: LinearityAnalysis,
            unit_props: UnitProperties
    ) -> FailurePrediction:
        """Calculate failure probability and risk category."""
        # Simple failure probability calculation
        # In practice, this would use ML models

        # Factors - ensure they are between 0 and 1
        # Sigma factor: only high if approaching or exceeding threshold
        sigma_factor = min(sigma_analysis.sigma_ratio, 1.0)
        
        # Linearity factor: ratio of error to spec
        linearity_factor = 0.0
        if linearity_analysis.linearity_spec > 0:
            linearity_factor = min(
                linearity_analysis.final_linearity_error_shifted / linearity_analysis.linearity_spec,
                1.0
            )

        # Resistance factor: only significant if change is large
        resistance_factor = 0.0  # Default
        if unit_props.resistance_change_percent is not None:
            # Only consider it a problem if resistance change is > 5%
            resistance_factor = min(abs(unit_props.resistance_change_percent) / 10.0, 1.0)

        # Weighted combination - adjusted for more reasonable probabilities
        failure_probability = (
                0.4 * sigma_factor +  # 40% weight on sigma
                0.4 * linearity_factor +  # 40% weight on linearity
                0.2 * resistance_factor  # 20% weight on resistance
        )
        
        # Apply a scaling factor to make probabilities more reasonable
        # Most passing units should have low failure probability
        if sigma_analysis.sigma_pass and linearity_analysis.linearity_pass:
            # Scale down for passing units
            failure_probability *= 0.5
        
        failure_probability = min(max(failure_probability, 0), 1)

        # Determine risk category with adjusted thresholds
        # Use stricter criteria for HIGH risk
        if failure_probability > 0.8:  # Only very bad units are HIGH risk
            risk_category = RiskCategory.HIGH
        elif failure_probability > 0.5:  # Medium risk for moderate issues  
            risk_category = RiskCategory.MEDIUM
        else:
            risk_category = RiskCategory.LOW

        return FailurePrediction(
            failure_probability=failure_probability,
            risk_category=risk_category,
            gradient_margin=sigma_analysis.gradient_margin,
            contributing_factors={
                'sigma': sigma_factor,
                'linearity': linearity_factor,
                'resistance': resistance_factor
            }
        )

    def _determine_track_status(
            self,
            sigma_analysis: SigmaAnalysis,
            linearity_analysis: LinearityAnalysis
    ) -> AnalysisStatus:
        """Determine track status based on analyses."""
        if not sigma_analysis.sigma_pass or not linearity_analysis.linearity_pass:
            return AnalysisStatus.FAIL
        elif sigma_analysis.gradient_margin < 0.1 * sigma_analysis.sigma_threshold:
            return AnalysisStatus.WARNING
        else:
            return AnalysisStatus.PASS

    def _determine_overall_status(self, tracks: Dict[str, TrackData]) -> AnalysisStatus:
        """Determine overall file status from track statuses."""
        if not tracks:
            return AnalysisStatus.ERROR

        statuses = [track.status for track in tracks.values()]

        if any(s == AnalysisStatus.FAIL for s in statuses):
            return AnalysisStatus.FAIL
        elif any(s == AnalysisStatus.WARNING for s in statuses):
            return AnalysisStatus.WARNING
        elif all(s == AnalysisStatus.PASS for s in statuses):
            return AnalysisStatus.PASS
        else:
            return AnalysisStatus.ERROR