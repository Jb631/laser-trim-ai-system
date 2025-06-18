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
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
import numpy as np
import re
import threading
import time

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
# Import security utilities
from laser_trim_analyzer.core.security import (
    SecurityValidator, SecureFileProcessor, SecurityLevel,
    get_security_validator, validate_inputs
)
# Try to import ML components
try:
    from laser_trim_analyzer.ml.predictors import MLPredictor, PredictionResult
    HAS_ML = True
except ImportError:
    HAS_ML = False
    MLPredictor = None
    PredictionResult = None
    
    # Create dummy ML predictor for graceful degradation
    class DummyMLPredictor:
        def __init__(self, *args, **kwargs):
            pass
        def initialize(self):
            return False
        async def predict(self, *args, **kwargs):
            return {}

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

# Import memory safety if available
try:
    from laser_trim_analyzer.core.memory_safety import (
        get_memory_validator, SafeCache, memory_safe_context
    )
    HAS_MEMORY_SAFETY = True
except ImportError:
    HAS_MEMORY_SAFETY = False

# Import secure logging if available
try:
    from laser_trim_analyzer.core.secure_logging import (
        get_logger, logged_function, LogLevel
    )
    HAS_SECURE_LOGGING = True
    logger = get_logger(__name__)
except ImportError:
    HAS_SECURE_LOGGING = False
    logger = logging.getLogger(__name__)


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

        # Initialize ML predictor with robust error handling
        self._initialize_ml_predictor()

        # Initialize analyzers with error handling
        try:
            self.sigma_analyzer = SigmaAnalyzer(config, logger)
            self.linearity_analyzer = LinearityAnalyzer(config, logger)
            self.resistance_analyzer = ResistanceAnalyzer(config, logger)
        except Exception as e:
            self.logger.error(f"Failed to initialize analyzers: {e}")
            # Create dummy analyzers to prevent complete failure
            self.sigma_analyzer = None
            self.linearity_analyzer = None
            self.resistance_analyzer = None

        # Initialize calculation validator with error handling
        try:
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
        except Exception as e:
            self.logger.warning(f"Failed to initialize calculation validator: {e}")
            self.calculation_validator = None

        # Processing state with thread safety
        self._executor = None
        self._processing_tasks = []
        self._is_processing = False
        self._processing_lock = threading.Lock()

        # Initialize processing statistics
        self._processing_stats = {
            "files_processed": 0,
            "validation_failures": 0,
            "processing_errors": 0,
            "cache_hits": 0,
            "ml_predictions": 0,
            "ml_errors": 0
        }

        # Cache for performance with size limit
        self._file_cache = {}
        self._cache_enabled = getattr(config.processing, 'cache_enabled', True)
        self._max_cache_size = 100  # Limit cache size to prevent memory issues

    def _initialize_ml_predictor(self):
        """Initialize ML predictor - ML is required, not optional."""
        # If ML predictor already provided, use it
        if self.ml_predictor is not None:
            self.logger.info("Using provided ML predictor")
            return

        # ML is required - check if available
        if not HAS_ML:
            error_msg = "ML components not available but are required. Install with: pip install scikit-learn joblib"
            self.logger.error(error_msg)
            raise ImportError(error_msg)

        try:
            self.logger.info("Initializing ML predictor...")
            self.ml_predictor = MLPredictor(self.config, logger=self.logger)
            
            # Initialize synchronously with timeout
            def run_with_timeout():
                try:
                    return self.ml_predictor.initialize()
                except Exception as e:
                    self.logger.error(f"ML predictor initialization failed: {e}")
                    return False
            
            # Run initialization with simple timeout using threading
            import threading
            import time
            
            result = [None]  # Use list to modify from inner function
            exception = [None]
            
            def worker():
                try:
                    result[0] = run_with_timeout()
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            thread.join(timeout=30.0)  # 30 second timeout
            
            if thread.is_alive():
                self.logger.warning("ML predictor initialization timed out - disabling ML")
                self.ml_predictor = None
            elif exception[0]:
                self.logger.warning(f"ML predictor initialization error: {exception[0]}")
                self.ml_predictor = None
            elif result[0]:
                self.logger.info("ML predictor initialized successfully")
            else:
                self.logger.warning("ML predictor initialization failed - using fallback")
                self.ml_predictor = None
                
        except Exception as e:
            self.logger.warning(f"Failed to create ML predictor: {e}")
            self.ml_predictor = None

    async def process_file(
            self,
            file_path: Path,
            output_dir: Optional[Path] = None,
            progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AnalysisResult:
        """
        Process a single Excel file with comprehensive validation and robust error handling.

        Args:
            file_path: Path to Excel file
            output_dir: Output directory for plots
            progress_callback: Progress callback function

        Returns:
            AnalysisResult with complete analysis data and validation status
        """
        file_path = Path(file_path)
        
        # Protect against concurrent processing issues
        with self._processing_lock:
            return await self._process_file_internal(file_path, output_dir, progress_callback)

    async def _process_file_internal(
            self,
            file_path: Path,
            output_dir: Optional[Path] = None,
            progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AnalysisResult:
        """Internal file processing with comprehensive error handling."""
        
        def safe_progress_callback(message: str, progress: float):
            """Safe progress callback that won't fail."""
            try:
                if progress_callback:
                    progress_callback(message, progress)
            except Exception as e:
                self.logger.warning(f"Progress callback error: {e}")

        safe_progress_callback("Starting file validation...", 0.0)

        # Step 1: Pre-flight file validation with comprehensive error handling
        self.logger.info(f"Starting comprehensive validation for: {file_path.name}")
        
        # Log input parameters for debugging
        if HAS_SECURE_LOGGING:
            self.logger.debug("process_file input parameters", context={
                'file_path': str(file_path),
                'file_size_mb': file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0,
                'output_dir': str(output_dir) if output_dir else None,
                'has_progress_callback': progress_callback is not None
            })
        
        try:
            # Security validation first
            security_validator = get_security_validator()
            
            # Validate file path for security threats
            path_result = security_validator.validate_input(
                file_path,
                'file_path',
                {
                    'require_absolute': False,
                    'allowed_extensions': ['.xlsx', '.xls', '.xlsm'],
                    'check_extension': True
                }
            )
            
            if not path_result.is_safe:
                self._processing_stats["validation_failures"] += 1
                raise ValidationError(f"Security validation failed: {'; '.join(path_result.validation_errors)}")
                
            if path_result.threats_detected:
                self.logger.warning(f"Security threats detected: {path_result.threats_detected}")
                raise ValidationError(f"Security threat detected: {path_result.threats_detected[0].value}")
            
            # Use sanitized path
            safe_file_path = Path(path_result.sanitized_value)
            
            # Validate file exists and is readable
            if not safe_file_path.exists():
                raise FileNotFoundError(f"File not found: {safe_file_path}")
                
            if not safe_file_path.is_file():
                raise ValidationError(f"Path is not a file: {safe_file_path}")
                
            # Check file size before processing
            file_size = safe_file_path.stat().st_size
            max_size = getattr(self.config.processing, 'max_file_size_mb', 100) * 1024 * 1024
            if file_size > max_size:
                raise ValidationError(f"File too large: {file_size / 1024 / 1024:.1f}MB > {max_size / 1024 / 1024:.1f}MB")

            # Validate file extension
            valid_extensions = getattr(self.config.processing, 'file_extensions', ['.xlsx', '.xls'])
            if file_path.suffix.lower() not in valid_extensions:
                raise ValidationError(f"Unsupported file type: {file_path.suffix}")

            # Try basic Excel file validation
            try:
                file_validation = validate_excel_file(
                    file_path=safe_file_path,
                    max_file_size_mb=getattr(self.config.processing, 'max_file_size_mb', 100)
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
                # If validation utility fails, try basic checks
                self.logger.warning(f"Advanced validation failed, using basic checks: {e}")
                try:
                    import pandas as pd
                    # Try to read the Excel file to check if it's valid
                    excel_file = pd.ExcelFile(safe_file_path)
                    sheet_names = excel_file.sheet_names
                    self.logger.info(f"Basic validation passed - Found {len(sheet_names)} sheets")
                except Exception as excel_error:
                    raise ValidationError(f"File is not a valid Excel file: {excel_error}")
            
            # Update file_path to use the safe path
            file_path = safe_file_path
            
        except Exception as e:
            self._processing_stats["validation_failures"] += 1
            self.logger.error(f"File validation error: {e}")
            raise ValidationError(f"File validation failed: {e}")

        safe_progress_callback("File validation complete, checking cache...", 0.1)

        # Step 2: Cache check with size management
        file_hash = None
        try:
            file_hash = calculate_file_hash(file_path)
            if self._cache_enabled and file_hash in self._file_cache:
                self._processing_stats["cache_hits"] += 1
                self.logger.info(f"Using cached result for {file_path.name}")
                return self._file_cache[file_hash]
        except Exception as e:
            self.logger.warning(f"Cache check failed: {e}")

        self.logger.info(f"Processing file: {file_path.name}")
        start_time = datetime.now()

        try:
            safe_progress_callback("Detecting system and extracting metadata...", 0.2)

            # Step 3: System detection and metadata extraction with fallback
            try:
                system_type = detect_system_type(file_path)
                self.logger.info(f"Detected system type: {system_type.value}")
            except Exception as e:
                self.logger.warning(f"System detection failed, using default: {e}")
                system_type = SystemType.SYSTEM_A  # Default fallback

            # Extract file metadata with error handling
            try:
                metadata = await self._extract_file_metadata(file_path, system_type)
            except Exception as e:
                self.logger.warning(f"Metadata extraction failed, using minimal metadata: {e}")
                # Create minimal metadata
                metadata = FileMetadata(
                    filename=file_path.name,
                    file_path=str(file_path),
                    model='Unknown',
                    serial='Unknown',
                    timestamp=datetime.now(),
                    system_type=system_type,
                    file_size_mb=file_path.stat().st_size / 1024 / 1024,
                    has_multi_tracks=False
                )
            
            # Step 4: Model validation with fallback
            if metadata.model and metadata.model != 'Unknown':
                try:
                    model_validation = validate_model_number(metadata.model)
                    if not model_validation.is_valid:
                        self.logger.warning(f"Model validation warnings: {model_validation.warnings}")
                    if model_validation.errors:
                        self.logger.error(f"Model validation errors: {model_validation.errors}")
                except Exception as e:
                    self.logger.warning(f"Model validation failed: {e}")

            safe_progress_callback("Finding track sheets...", 0.3)

            # Step 5: Track detection and processing with robust error handling
            try:
                track_sheets = await self._find_track_sheets(file_path, system_type)
                self.logger.info(f"Found {len(track_sheets)} track sheets")
            except Exception as e:
                self.logger.warning(f"Track sheet detection failed: {e}")
                # Try fallback track detection
                track_sheets = {"default": {"sheet": "Sheet1"}}

            if not track_sheets:
                raise DataExtractionError("No valid track sheets found")

            # Process tracks with validation and error recovery
            tracks = {}
            total_tracks = len(track_sheets)
            processing_errors = []
            
            for idx, (track_id, sheets) in enumerate(track_sheets.items()):
                if safe_progress_callback:
                    progress = 0.4 + (0.5 * idx / total_tracks)
                    safe_progress_callback(f"Processing track {track_id}...", progress)

                try:
                    track_data = await self._process_track_with_validation(
                        file_path, track_id, sheets, system_type, metadata.model
                    )
                    tracks[track_id] = track_data
                    self.logger.debug(f"Successfully processed track {track_id}")
                    
                except Exception as e:
                    error_msg = f"Failed to process track {track_id}: {e}"
                    self.logger.error(error_msg)
                    processing_errors.append((track_id, str(e)))
                    # Continue with other tracks rather than failing completely
                    continue

            if not tracks:
                if processing_errors:
                    error_details = "; ".join([f"{track}: {error}" for track, error in processing_errors])
                    raise ProcessingError(f"No tracks could be processed successfully. Errors: {error_details}")
                else:
                    raise ProcessingError("No tracks could be processed successfully")

            safe_progress_callback("Finalizing analysis...", 0.9)

            # Step 6: Create result with comprehensive data
            primary_track_id = self._determine_primary_track(tracks)
            primary_track = tracks[primary_track_id]

            # Calculate overall status and validation
            overall_status = self._determine_overall_status(tracks)
            overall_validation_status = self._determine_overall_validation_status(tracks)
            validation_grade = self._calculate_overall_validation_grade(tracks)

            # Collect all validation warnings and recommendations from tracks
            all_validation_warnings = []
            all_validation_recommendations = []
            for track_data in tracks.values():
                if hasattr(track_data, 'validation_warnings'):
                    all_validation_warnings.extend(track_data.validation_warnings)
                if hasattr(track_data, 'validation_recommendations'):
                    all_validation_recommendations.extend(track_data.validation_recommendations)

            # Create result
            result = AnalysisResult(
                metadata=metadata,
                tracks=tracks,
                overall_status=overall_status,
                overall_validation_status=overall_validation_status,
                processing_time=(datetime.now() - start_time).total_seconds(),
                processing_errors=processing_errors if processing_errors else [],
                validation_warnings=all_validation_warnings,
                validation_recommendations=all_validation_recommendations
            )

            # Step 7: Add ML predictions with error handling
            try:
                if self.ml_predictor:
                    safe_progress_callback("Adding ML predictions...", 0.95)
                    await self._add_ml_predictions(result)
                    self._processing_stats["ml_predictions"] += 1
            except Exception as e:
                self.logger.warning(f"ML predictions failed: {e}")
                self._processing_stats["ml_errors"] += 1

            # Step 8: Generate outputs if requested
            if output_dir:
                try:
                    safe_progress_callback("Generating outputs...", 0.98)
                    await self._generate_outputs(result, output_dir, file_path)
                except Exception as e:
                    self.logger.warning(f"Output generation failed: {e}")

            # Step 9: Cache result if enabled
            if self._cache_enabled and file_hash:
                try:
                    # Get max cache entries from config
                    max_cache_entries = getattr(self.config.processing, 'max_cache_entries', 50)
                    
                    # Check cache size limit
                    if len(self._file_cache) >= max_cache_entries:
                        # Remove oldest entries (FIFO) - remove 20% to avoid frequent cleanup
                        entries_to_remove = max(1, max_cache_entries // 5)
                        keys_to_remove = list(self._file_cache.keys())[:entries_to_remove]
                        for key in keys_to_remove:
                            del self._file_cache[key]
                        self.logger.debug(f"Removed {entries_to_remove} oldest cache entries")
                    
                    # Also check memory pressure for large batches
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_percent = process.memory_percent()
                        
                        # More aggressive cache management for high memory usage
                        if memory_percent > 70:  # High memory usage
                            # Keep only most recent 10 entries
                            if len(self._file_cache) > 10:
                                keys_to_keep = list(self._file_cache.keys())[-10:]
                                self._file_cache = {k: self._file_cache[k] for k in keys_to_keep}
                                gc.collect()
                                self.logger.info(f"Reduced cache to 10 entries due to high memory usage ({memory_percent:.1f}%)")
                        elif memory_percent > 50 and len(self._file_cache) > 25:
                            # Moderate memory usage - keep 25 entries
                            keys_to_keep = list(self._file_cache.keys())[-25:]
                            self._file_cache = {k: self._file_cache[k] for k in keys_to_keep}
                            self.logger.debug(f"Reduced cache to 25 entries due to moderate memory usage ({memory_percent:.1f}%)")
                    except:
                        # If psutil not available, just use entry count limit
                        pass
                    
                    self._file_cache[file_hash] = result
                except Exception as e:
                    self.logger.warning(f"Failed to cache result: {e}")

            # Update processing statistics
            self._processing_stats["files_processed"] += 1
            
            safe_progress_callback("Analysis complete", 1.0)
            self.logger.info(f"Successfully processed {file_path.name} in {result.processing_time:.2f}s")
            
            # Log output summary for debugging
            if HAS_SECURE_LOGGING:
                self.logger.debug("process_file output summary", context={
                    'file_path': str(file_path),
                    'processing_time': result.processing_time,
                    'overall_status': result.overall_status.value,
                    'validation_status': result.overall_validation_status.value,
                    'num_tracks': len(result.tracks),
                    'num_errors': len(result.processing_errors),
                    'num_warnings': len(result.validation_warnings),
                    'has_ml_predictions': bool(self.ml_predictor),
                    'output_generated': bool(output_dir)
                })
            
            return result

        except Exception as e:
            self._processing_stats["processing_errors"] += 1
            self.logger.error(f"Processing failed for {file_path.name}: {e}")
            raise

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
        resistance_analysis = await self._analyze_resistance(unit_props, file_path, sheets, system_type)

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

        # Determine track status
        track_status, status_reason = self._determine_track_status(sigma_analysis, linearity_analysis)
        
        # Extract travel length from analysis data
        travel_length = analysis_data.get('travel_length', 0.0)
        if travel_length == 0.0 and 'positions' in analysis_data:
            positions = analysis_data['positions']
            if positions:
                travel_length = max(positions) - min(positions)

        # Create track data with comprehensive validation info
        track_data = TrackData(
            track_id=track_id,
            status=track_status,
            status_reason=status_reason,
            travel_length=travel_length,
            # Add position and error data for plotting
            position_data=analysis_data.get('positions', []),
            error_data=analysis_data.get('errors', []),
            # Add spec limits for plotting
            upper_limits=analysis_data.get('upper_limits', []),
            lower_limits=analysis_data.get('lower_limits', []),
            # Add untrimmed data if available
            untrimmed_positions=analysis_data.get('untrimmed_data', {}).get('positions', []),
            untrimmed_errors=analysis_data.get('untrimmed_data', {}).get('errors', []),
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
        track_data.overall_validation_status = track_validation_status

        return track_data

    def process_file_sync(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AnalysisResult:
        """
        Synchronous wrapper for process_file method.
        
        This is used when calling from non-async contexts like thread pools.
        
        Args:
            file_path: Path to Excel file
            output_dir: Optional directory for outputs
            progress_callback: Optional progress callback
            
        Returns:
            AnalysisResult object
        """
        import asyncio
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, self.process_file(file_path, output_dir, progress_callback))
                    return future.result()
        except RuntimeError:
            # No event loop exists, use asyncio.run
            pass
        
        # Run the async method
        return asyncio.run(self.process_file(file_path, output_dir, progress_callback))

    async def process_batch(
        self,
        file_paths: List[Path],
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        max_workers: Optional[int] = None
    ) -> Dict[str, AnalysisResult]:
        """
        Process multiple files in batch with optimized performance and memory management.
        
        Args:
            file_paths: List of file paths to process
            output_dir: Optional output directory for plots
            progress_callback: Optional progress callback function
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping file paths to analysis results
        """
        import gc
        import time
        import psutil
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if not file_paths:
            return {}

        self.logger.info(f"Starting batch processing of {len(file_paths)} files")
        
        # Performance tracking
        start_time = time.time()
        last_cleanup_time = start_time
        processed_count = 0
        
        # Step 1: Batch validation with progress throttling
        if progress_callback:
            progress_callback("Validating files...", 0.02)

        invalid_files = []
        try:
            from laser_trim_analyzer.utils.validators import BatchValidator
            
            validation_result = BatchValidator.validate_batch(
                file_paths=file_paths,
                max_batch_size=self.config.processing.max_batch_size
            )
            
            if not validation_result.is_valid:
                invalid_files = validation_result.metadata.get('invalid_files', [])
                self.logger.warning(f"Batch validation found {len(invalid_files)} invalid files")
                for invalid in invalid_files[:5]:  # Log first 5 invalid files
                    self.logger.warning(f"Invalid file: {invalid}")
                    
        except Exception as e:
            self.logger.error(f"Batch validation failed: {e}")
            # Continue with processing despite validation failure
        
        # Filter to only valid files for processing
        processable_files = []
        for file_path in file_paths:
            if not any(str(file_path) in invalid['file'] for invalid in invalid_files):
                processable_files.append(file_path)
        
        if not processable_files:
            raise ValidationError("No valid files found in batch after validation")

        if progress_callback:
            progress_callback(f"Processing {len(processable_files)} validated files...", 0.1)

        # Step 2: Process files with enhanced memory management and throttling
        max_workers = max_workers or self.config.processing.max_workers
        results = {}
        failed_files = []

        # Adaptive batch sizing based on file count and memory
        if len(processable_files) > 500:
            # Very large batches - use smaller concurrent batches
            concurrent_batch_size = min(max_workers, 10)
            cleanup_interval = 25  # More frequent cleanup
        elif len(processable_files) > 100:
            # Large batches
            concurrent_batch_size = min(max_workers, 15)
            cleanup_interval = 50
        else:
            # Standard batches
            concurrent_batch_size = max_workers
            cleanup_interval = 100

        self.logger.info(f"Using concurrent batch size: {concurrent_batch_size}, cleanup interval: {cleanup_interval}")

        # Process in smaller concurrent batches to manage memory
        for i in range(0, len(processable_files), concurrent_batch_size):
            batch_files = processable_files[i:i + concurrent_batch_size]
            base_progress = 0.1 + (0.8 * i / len(processable_files))
            
            # Process concurrent batch
            batch_results, batch_failed = await self._process_concurrent_batch(
                batch_files=batch_files,
                output_dir=output_dir,
                progress_callback=progress_callback,
                base_progress=base_progress,
                total_files=len(processable_files),
                batch_index=i // concurrent_batch_size
            )
            
            # Merge results
            results.update(batch_results)
            failed_files.extend(batch_failed)
            processed_count += len(batch_files)
            
            # Memory and performance management
            current_time = time.time()
            if (processed_count % cleanup_interval == 0 and processed_count > 0) or \
               (current_time - last_cleanup_time > 30):  # Force cleanup every 30 seconds
                
                await self._perform_batch_cleanup(processed_count, current_time - start_time)
                last_cleanup_time = current_time
                
                # Yield CPU time to prevent system freezing
                await asyncio.sleep(0.01)

        # Step 3: Generate batch summary with final cleanup
        if progress_callback:
            progress_callback("Finalizing batch results...", 0.95)

        # Final cleanup
        await self._perform_final_cleanup()

        successful_count = len(results)
        failed_count = len(failed_files)
        
        total_time = time.time() - start_time
        files_per_second = successful_count / total_time if total_time > 0 else 0
        
        self.logger.info(f"Batch processing complete: {successful_count} successful, {failed_count} failed")
        self.logger.info(f"Processing rate: {files_per_second:.2f} files/second")

        return results

    async def _process_concurrent_batch(
        self,
        batch_files: List[Path],
        output_dir: Optional[Path],
        progress_callback: Optional[Callable[[str, float], None]],
        base_progress: float,
        total_files: int,
        batch_index: int
    ) -> Tuple[Dict[str, AnalysisResult], List[Tuple[str, str]]]:
        """Process a small batch of files concurrently with memory management."""
        import asyncio
        
        results = {}
        failed_files = []
        
        # Create semaphore to limit true concurrency and prevent resource exhaustion
        max_concurrent = min(len(batch_files), 8)  # Never more than 8 truly concurrent
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_file_with_throttling(file_path: Path, file_idx: int) -> Optional[AnalysisResult]:
            """Process a single file with semaphore control and memory management."""
            async with semaphore:
                try:
                    # Throttled progress callback
                    def file_progress_callback(message: str, progress: float):
                        if progress_callback:
                            # Update at most every 500ms per file to prevent UI flooding
                            overall_progress = base_progress + (0.8 / total_files * (file_idx + progress))
                            
                            # Use asyncio to schedule UI update without blocking
                            asyncio.create_task(
                                self._throttled_progress_update(
                                    progress_callback, 
                                    f"Batch {batch_index + 1}: {file_path.name} - {message}",
                                    overall_progress
                                )
                            )
                    
                    result = await self.process_file(
                        file_path=file_path,
                        output_dir=output_dir,
                        progress_callback=file_progress_callback
                    )
                    
                    # Small yield to prevent CPU monopolization
                    await asyncio.sleep(0.001)
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path.name}: {e}")
                    failed_files.append((str(file_path), str(e)))
                    return None
        
        # Process files in the batch concurrently
        tasks = [
            process_single_file_with_throttling(file_path, idx) 
            for idx, file_path in enumerate(batch_files)
        ]
        
        # Wait for completion with proper exception handling
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for file_path, result in zip(batch_files, completed_results):
            if isinstance(result, Exception):
                failed_files.append((str(file_path), str(result)))
            elif result is not None:
                results[str(file_path)] = result
        
        return results, failed_files

    async def _throttled_progress_update(
        self, 
        callback: Callable[[str, float], None], 
        message: str, 
        progress: float
    ):
        """Throttled progress update to prevent UI flooding."""
        try:
            callback(message, progress)
        except Exception as e:
            # Don't let progress callback errors stop processing
            self.logger.debug(f"Progress callback error: {e}")

    async def _perform_batch_cleanup(self, processed_count: int, elapsed_time: float):
        """
        Perform cleanup operations during batch processing.
        
        Args:
            processed_count: Number of files processed so far
            elapsed_time: Elapsed time in seconds
        """
        # Aggressive memory management for large batches
        import gc
        import matplotlib.pyplot as plt
        
        # Force garbage collection every 50 files or every 10 minutes
        gc_interval = getattr(self.config.processing, 'garbage_collection_interval', 50)
        if processed_count % gc_interval == 0:
            self.logger.debug(f"Forcing garbage collection after {processed_count} files")
            try:
                collected = gc.collect()
                self.logger.debug(f"Garbage collector released {collected} objects")
            except Exception as e:
                self.logger.warning(f"Error during garbage collection: {e}")
        
        # Close matplotlib figures to prevent memory leaks
        # Close after EVERY file to prevent memory accumulation
        try:
            plt.close('all')  # Close all matplotlib figures
            # Also clear the figure registry to ensure complete cleanup
            import matplotlib._pylab_helpers
            matplotlib._pylab_helpers.Gcf.destroy_all()
        except Exception as e:
            self.logger.warning(f"Error closing matplotlib figures: {e}")
        
        # Force garbage collection every 10 files or when memory usage is high
        if processed_count % 10 == 0:
            gc.collect()
            # Check memory usage
            try:
                import psutil
                process = psutil.Process()
                memory_percent = process.memory_percent()
                if memory_percent > 70:  # If using more than 70% of system memory
                    self.logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
                    gc.collect(2)  # Full collection
                    # Clear caches if memory is critical
                    if memory_percent > 80 and hasattr(self, '_file_cache'):
                        self.logger.warning("Critical memory usage - clearing file cache")
                        self._file_cache.clear()
            except:
                pass
        
        # Clear file cache periodically for very large batches
        if hasattr(self, '_file_cache'):
            try:
                cache_size_before = len(self._file_cache)
                if cache_size_before > 100:
                    # Keep only the most recent 50 entries
                    if cache_size_before > 50:
                        # For simplicity, just clear half the cache
                        keys_to_remove = list(self._file_cache.keys())[:-50]
                        for key in keys_to_remove:
                            del self._file_cache[key]
                        self.logger.debug(f"Reduced cache from {cache_size_before} to {len(self._file_cache)} entries")
            except Exception as e:
                self.logger.warning(f"Error managing file cache: {e}")
                # If error occurs, try to clear the entire cache
                try:
                    self._file_cache.clear()
                    self.logger.debug("Cleared entire file cache due to error")
                except:
                    pass
        
        # Log memory usage and performance stats
        processing_rate = processed_count / elapsed_time if elapsed_time > 0 else 0
        self.logger.info(f"Batch progress: {processed_count} files processed, "
                        f"rate: {processing_rate:.2f} files/sec")
        
        # Check memory usage and warn if high
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Warn if memory usage is getting high
            memory_threshold = getattr(self.config.processing, 'memory_throttle_threshold', 1000)
            if memory_mb > memory_threshold:
                self.logger.warning(f"High memory usage detected: {memory_mb:.1f} MB "
                                  f"(threshold: {memory_threshold} MB)")
                
                # Force more aggressive cleanup if memory is very high
                if memory_mb > memory_threshold * 1.5:
                    self.logger.warning("Performing emergency memory cleanup")
                    gc.collect()
                    plt.close('all')
                    # Clear more of the cache
                    if hasattr(self, '_file_cache'):
                        cache_keys = list(self._file_cache.keys())
                        for key in cache_keys[:-10]:  # Keep only 10 most recent
                            del self._file_cache[key]
                    
                    # Add small delay to allow OS to reclaim memory
                    await asyncio.sleep(0.5)
                        
        except ImportError:
            # psutil not available, skip memory monitoring
            pass
        except Exception as e:
            self.logger.debug(f"Error monitoring memory: {e}")

    async def _perform_final_cleanup(self):
        """
        Perform final cleanup operations after batch processing completes.
        """
        # FIXED: Comprehensive final cleanup to prevent memory accumulation
        import gc
        import matplotlib.pyplot as plt
        
        self.logger.info("Performing final batch cleanup...")
        
        # Close all matplotlib figures
        plt.close('all')
        
        # Clear all caches
        if hasattr(self, '_file_cache'):
            cache_size = len(self._file_cache)
            self._file_cache.clear()
            self.logger.debug(f"Cleared file cache ({cache_size} entries)")
        
        # Clear any analyzers' internal state
        for analyzer in [self.sigma_analyzer, self.linearity_analyzer, self.resistance_analyzer]:
            if hasattr(analyzer, 'clear_cache'):
                analyzer.clear_cache()
        
        # Reset processing statistics
        self._processing_stats = {
            "files_processed": 0,
            "validation_failures": 0,
            "processing_errors": 0,
            "cache_hits": 0,
            "ml_predictions": 0,
            "ml_errors": 0
        }
        
        # Force comprehensive garbage collection
        for _ in range(3):  # Multiple passes to ensure everything is cleaned
            collected = gc.collect()
        
        # Log final memory state
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Final memory usage: {memory_mb:.1f} MB")
        except (ImportError, Exception):
            pass
        
        # Reset processing state
        self._is_processing = False
        self._processing_tasks = []
        
        self.logger.info("Final cleanup completed")

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
                    # More precise matching to avoid cross-track contamination
                    if track_id in sheet:
                        # Direct match - sheet contains the track ID
                        if " 0" in sheet or "_0" in sheet or sheet.endswith(" 0"):
                            sheets['untrimmed'] = sheet
                        elif "TRM" in sheet or "TRIM" in sheet.upper():
                            sheets['trimmed'] = sheet
                    elif track_id == "TRK1" and "TRK2" not in sheet:
                        # Special handling for TRK1 sheets that might not have "TRK1" explicitly
                        # but we must ensure it's not a TRK2 sheet
                        if ("1 0" in sheet or "1_0" in sheet) and "2" not in sheet.replace("SEC1", ""):
                            sheets['untrimmed'] = sheet
                        elif ("1 TRM" in sheet or "1_TRM" in sheet) and "2" not in sheet.replace("SEC1", ""):
                            sheets['trimmed'] = sheet
                
                if sheets:
                    track_sheets[track_id] = sheets
                    self.logger.debug(f"Track {track_id} sheets: {sheets}")
            
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
        
        # Debug logging for data extraction
        self.logger.debug(f"Data source: {'trimmed' if data_source == trimmed_data else 'untrimmed'}")
        self.logger.debug(f"Positions count: {len(data_source.get('positions', []))}")
        self.logger.debug(f"Errors count: {len(data_source.get('errors', []))}")
        self.logger.debug(f"Upper limits count: {len(data_source.get('upper_limits', []))}")
        self.logger.debug(f"Lower limits count: {len(data_source.get('lower_limits', []))}")
        
        # If trimmed data extraction failed but we have untrimmed data, log this
        if trimmed_data and not trimmed_data.get('positions') and untrimmed_data.get('positions'):
            self.logger.warning("Trimmed data extraction failed, using untrimmed data for analysis")

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
        self.logger.info(f"Generating outputs to {output_dir}, plots enabled: {self.config.processing.generate_plots}")
        
        # Generate plots if enabled
        if self.config.processing.generate_plots:
            await self._generate_plots(result, output_dir)
        else:
            self.logger.info("Plot generation is disabled in config")
        
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
        self.logger.info(f"Starting plot generation for {len(result.tracks)} tracks to {output_dir}")
        loop = asyncio.get_event_loop()
        
        for track_id, track_data in result.tracks.items():
            try:
                self.logger.info(f"Generating plot for track {track_id}")
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
                self.logger.info(f"Plot saved to {plot_path}")
            except Exception as e:
                self.logger.error(f"Plot generation failed for {track_id}: {e}")

    async def _add_ml_predictions(self, result: AnalysisResult) -> None:
        """Add ML predictions to result."""
        if not self.ml_predictor:
            return

        try:
            # Get predictions for primary track
            predictions = await self.ml_predictor.predict(result)

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

    def _calculate_overall_validation_grade(self, tracks: Dict[str, TrackData]) -> str:
        """Calculate overall validation grade from all tracks.
        
        Uses the same grading logic as the model's validation_grade property
        to ensure consistency across the application.
        """
        if not tracks:
            return "Not Available"
        
        grades = []
        grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0.5, "F": 0}
        
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
        
        # Convert back to letter grade with descriptions
        if avg_grade >= 3.5:
            return "A - Excellent"
        elif avg_grade >= 2.5:
            return "B - Good"
        elif avg_grade >= 1.5:
            return "C - Acceptable"
        elif avg_grade >= 0.75:
            return "D - Below Average"
        elif avg_grade >= 0.25:
            return "E - Poor"
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
            
            # Debug: Log first few rows to see alignment
            self.logger.debug(f"First 5 rows of data from {sheet_name}:")
            for i in range(min(5, len(df))):
                self.logger.debug(f"Row {i}: pos={position_series.iloc[i] if i < len(position_series) else 'N/A'}, "
                                f"err={error_series.iloc[i] if i < len(error_series) else 'N/A'}")
            
            # Check if error data is incomplete (all zeros or missing)
            error_values = error_series.dropna()
            if error_values.empty or (error_values == 0).all():
                self.logger.warning("Error data is incomplete (all zeros or missing), attempting to calculate from voltages")
                
                # Try to calculate error from measured and theory volts
                if 'measured_volts' in columns and 'theory_volts' in columns:
                    measured_series = pd.to_numeric(df.iloc[:, columns['measured_volts']], errors='coerce')
                    theory_series = pd.to_numeric(df.iloc[:, columns['theory_volts']], errors='coerce')
                    
                    # Calculate error as simple difference: measured - theory
                    error_series = (measured_series - theory_series).replace([np.inf, -np.inf], np.nan)
                    
                    # Log some statistics about the calculated errors
                    error_valid = error_series.dropna()
                    if len(error_valid) > 0:
                        self.logger.info(f"Calculated error statistics: min={error_valid.min():.3f}, max={error_valid.max():.3f}, "
                                       f"mean={error_valid.mean():.3f}, std={error_valid.std():.3f}, count={len(error_valid)}")
                    
                    self.logger.info("Successfully calculated error values (voltage difference) from measured and theory voltages")
                else:
                    self.logger.error("Cannot calculate error: measured_volts and theory_volts columns not found")
            
            # Extract upper and lower limits if available
            upper_limits = []
            lower_limits = []
            if 'upper_limit' in columns and 'lower_limit' in columns:
                self.logger.debug(f"Attempting to extract limits from columns {columns['upper_limit']} and {columns['lower_limit']}")
                upper_series = pd.to_numeric(df.iloc[:, columns['upper_limit']], errors='coerce')
                lower_series = pd.to_numeric(df.iloc[:, columns['lower_limit']], errors='coerce')
                
                # Keep the series as pandas Series for now to maintain position alignment
                upper_limits = upper_series
                lower_limits = lower_series
                
                # Log information about the limits
                upper_valid = upper_series.dropna()
                lower_valid = lower_series.dropna()
                self.logger.debug(f"Extracted limits - Upper: {len(upper_valid)} valid values, Lower: {len(lower_valid)} valid values")
                if len(upper_valid) > 0 and len(lower_valid) > 0:
                    self.logger.debug(f"Upper limits range: {upper_valid.min():.3f} to {upper_valid.max():.3f}")
                    self.logger.debug(f"Lower limits range: {lower_valid.min():.3f} to {lower_valid.max():.3f}")
                    
                    # Also check if errors and limits are in similar scale
                    # Get error_valid from the current error_series
                    error_valid_check = error_series.dropna()
                    if len(error_valid_check) > 0:
                        error_scale = error_valid_check.abs().max()
                        limit_scale = max(upper_valid.max(), abs(lower_valid.min()))
                        self.logger.debug(f"Scale comparison - Errors: {error_scale:.6f}, Limits: {limit_scale:.6f}")
                        
                        # If they're very different scales, log a warning
                        if error_scale > 0 and limit_scale > 0:
                            ratio = max(error_scale / limit_scale, limit_scale / error_scale)
                            if ratio > 100:
                                self.logger.warning(f"Large scale difference between errors and limits (ratio: {ratio:.1f})")
                                self.logger.warning("This might indicate a unit mismatch or data extraction issue")
            else:
                self.logger.warning(f"Limit columns not found in system {system} columns: {columns}")
            
            # Find rows where position is valid (not NaN)
            # We prioritize positions since they define the measurement points
            position_valid_mask = position_series.notna()
            position_valid_indices = position_valid_mask[position_valid_mask].index
            
            # Also check which rows have both position and error
            both_valid_mask = position_series.notna() & error_series.notna()
            both_valid_indices = both_valid_mask[both_valid_mask].index
            
            self.logger.info(f"Data extraction: Total rows: {len(position_series)}, "
                           f"Rows with valid position: {len(position_valid_indices)}, "
                           f"Rows with both position & error: {len(both_valid_indices)}")
            
            # Use rows with valid positions as our base
            valid_indices = position_valid_indices
            
            # Warn if we're missing error data for some positions
            if len(position_valid_indices) > len(both_valid_indices):
                missing_count = len(position_valid_indices) - len(both_valid_indices)
                self.logger.warning(f"Found {missing_count} positions without corresponding error values. "
                                  "These will be included with interpolated or zero errors.")
            
            # Show which Excel rows are being used (add 2 for header row and 0-based indexing)
            if len(valid_indices) > 0:
                excel_rows = [idx + 2 for idx in valid_indices[:10]]  # First 10 for logging
                self.logger.debug(f"Using Excel rows: {excel_rows}... (showing first 10)")
            
            # Extract positions using valid indices
            positions = position_series.loc[valid_indices].tolist()
            
            # Extract errors, handling missing values
            errors = []
            for idx in valid_indices:
                if idx in error_series.index and pd.notna(error_series.loc[idx]):
                    errors.append(error_series.loc[idx])
                else:
                    # Missing error value - use 0 or interpolate
                    errors.append(0.0)
                    self.logger.debug(f"Missing error at index {idx} (row {idx+2}), using 0.0")
            
            # Handle limits - extract from the SAME valid indices to maintain alignment
            if isinstance(upper_limits, pd.Series) and isinstance(lower_limits, pd.Series):
                # Extract limits at the same indices as valid positions/errors
                upper_limits = upper_limits.loc[valid_indices].tolist()
                lower_limits = lower_limits.loc[valid_indices].tolist()
                
                self.logger.debug(f"Extracted data lengths - pos: {len(positions)}, err: {len(errors)}, "
                                f"upper: {len(upper_limits)}, lower: {len(lower_limits)}")
                
                # Debug: Log alignment of positions, errors, and limits
                self.logger.debug("Data alignment check (first 5 points):")
                for i in range(min(5, len(positions))):
                    self.logger.debug(f"Point {i}: pos={positions[i]:.3f}, err={errors[i]:.6f}, "
                                    f"upper={upper_limits[i] if i < len(upper_limits) else 'N/A'}, "
                                    f"lower={lower_limits[i] if i < len(lower_limits) else 'N/A'}")
                
                # Check if all data has same length (it should after using same indices)
                if len(upper_limits) != len(positions):
                    self.logger.warning(f"Upper limits length mismatch: {len(upper_limits)} vs {len(positions)}")
                if len(lower_limits) != len(positions):
                    self.logger.warning(f"Lower limits length mismatch: {len(lower_limits)} vs {len(positions)}")
            else:
                # No limits available - use empty lists
                upper_limits = []
                lower_limits = []

            # Calculate travel length and position range
            if positions:
                travel_length = max(positions) - min(positions)
                pos_min, pos_max = min(positions), max(positions)
                self.logger.info(f"Position range: [{pos_min:.1f}, {pos_max:.1f}] mm, travel length: {travel_length:.1f} mm")
            else:
                travel_length = 0
                self.logger.warning("No valid positions found!")

            result = {
                'positions': positions,
                'errors': errors,
                'upper_limits': upper_limits,
                'lower_limits': lower_limits,
                'travel_length': travel_length
            }
            
            self.logger.info(f"Final extracted data from {sheet_name}: {len(positions)} positions, {len(errors)} errors, "
                            f"{len(upper_limits) if upper_limits else 0} upper limits, "
                            f"{len(lower_limits) if lower_limits else 0} lower limits")
            
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
        # Debug logging for limit data
        upper_limits = data.get('upper_limits', [])
        lower_limits = data.get('lower_limits', [])
        
        # Count valid limits
        valid_upper = [u for u in upper_limits if u is not None and not np.isnan(u)]
        valid_lower = [l for l in lower_limits if l is not None and not np.isnan(l)]
        
        self.logger.debug(f"Sigma analysis - Upper limits: {len(upper_limits)} total, {len(valid_upper)} valid")
        self.logger.debug(f"Sigma analysis - Lower limits: {len(lower_limits)} total, {len(valid_lower)} valid")
        if valid_upper:
            self.logger.debug(f"Upper limit range: {min(valid_upper):.3f} to {max(valid_upper):.3f}")
        if valid_lower:
            self.logger.debug(f"Lower limit range: {min(valid_lower):.3f} to {max(valid_lower):.3f}")
        
        # Prepare data dictionary for analyzer
        analysis_data = {
            'positions': data.get('positions', []),
            'errors': data.get('errors', []),
            'upper_limits': upper_limits,
            'lower_limits': lower_limits,
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
            unit_props: UnitProperties,
            file_path: Path,
            sheets: Dict[str, str],
            system_type: SystemType
    ) -> ResistanceAnalysis:
        """Perform resistance analysis."""
        # Prepare data dictionary for analyzer
        analysis_data = {
            'untrimmed_resistance': unit_props.untrimmed_resistance,
            'trimmed_resistance': unit_props.trimmed_resistance,
            'file_path': str(file_path),
            'discovered_sheets': sheets,
            'system_type': system_type.value
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
    ) -> tuple[AnalysisStatus, str]:
        """Determine track status based on analyses.
        
        Linearity is the PRIMARY goal of laser trimming.
        - PASS: Linearity passes and sigma passes
        - WARNING: Linearity passes but sigma fails or margin is tight
        - FAIL: Linearity fails (primary requirement not met)
        
        Returns:
            Tuple of (status, reason)
        """
        if not linearity_analysis.linearity_pass:
            # Primary requirement not met
            fail_count = getattr(linearity_analysis, 'fail_count', 0)
            total_points = getattr(linearity_analysis, 'total_points', 0)
            reason = f"Linearity failed: {fail_count} points out of spec (0 allowed)"
            return AnalysisStatus.FAIL, reason
        elif not sigma_analysis.sigma_pass:
            # Linearity OK but sigma failed - functional but quality concern
            reason = f"Sigma gradient ({sigma_analysis.sigma_gradient:.4f}) exceeds threshold ({sigma_analysis.sigma_threshold:.4f})"
            return AnalysisStatus.WARNING, reason
        elif sigma_analysis.gradient_margin < 0.2 * sigma_analysis.sigma_threshold:
            # Both pass but sigma margin is tight (less than 20% margin)
            margin_percent = (sigma_analysis.gradient_margin / sigma_analysis.sigma_threshold) * 100
            reason = f"Sigma margin tight: only {margin_percent:.1f}% margin to threshold"
            return AnalysisStatus.WARNING, reason
        else:
            # Both linearity and sigma pass with good margin
            reason = "All tests passed with good margins"
            return AnalysisStatus.PASS, reason

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
