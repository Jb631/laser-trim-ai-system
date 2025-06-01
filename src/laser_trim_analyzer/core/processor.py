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
    DynamicRangeAnalysis, AnalysisStatus, SystemType, RiskCategory
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

        # Processing state
        self._executor = None
        self._processing_tasks = []
        self._is_processing = False

        # Cache for performance
        self._file_cache = {}
        self._cache_enabled = config.processing.cache_enabled

    async def process_file(
            self,
            file_path: Path,
            output_dir: Optional[Path] = None,
            progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AnalysisResult:
        """Process a single Excel file with ML predictions."""
        """
        Process a single Excel file.

        Args:
            file_path: Path to Excel file
            output_dir: Output directory for plots
            progress_callback: Progress callback function

        Returns:
            AnalysisResult with complete analysis data
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.suffix.lower() in self.config.processing.file_extensions:
            raise ValidationError(f"Unsupported file type: {file_path.suffix}")

        # Check cache
        file_hash = calculate_file_hash(file_path)
        if self._cache_enabled and file_hash in self._file_cache:
            self.logger.info(f"Using cached result for {file_path.name}")
            return self._file_cache[file_hash]

        self.logger.info(f"Processing file: {file_path.name}")
        start_time = datetime.now()

        try:
            # Progress callback
            if progress_callback:
                progress_callback("Reading file...", 0.1)

            # Extract metadata
            metadata = await self._extract_metadata(file_path)

            # Detect system type
            system_type = await self._detect_system(file_path)
            metadata.system = system_type

            if progress_callback:
                progress_callback("Extracting data...", 0.2)

            # Extract and analyze tracks
            tracks = await self._process_tracks(
                file_path, system_type, metadata.model, progress_callback
            )

            # Determine overall status
            overall_status = self._determine_overall_status(tracks)

            result = AnalysisResult(
                metadata=metadata,
                overall_status=overall_status,
                processing_time=(datetime.now() - start_time).total_seconds(),
                tracks=tracks
            )

            if progress_callback:
                progress_callback("Running ML analysis...", 0.75)

            # ML predictions - moved before plot generation
            if self.ml_predictor and self.config.ml.enabled:
                try:
                    ml_predictions = await self.ml_predictor.predict(result)

                    # Apply ML predictions to result
                    self._apply_ml_predictions(result, ml_predictions)

                    # Check for critical warnings
                    self._process_ml_warnings(result, ml_predictions)

                except Exception as e:
                    self.logger.error(f"ML prediction failed: {e}")
                    result.processing_errors.append(f"ML prediction failed: {str(e)}")

            if progress_callback:
                progress_callback("Generating plots...", 0.8)

            # Generate plots if enabled
            if self.config.processing.generate_plots and output_dir:
                await self._generate_plots(result, output_dir)

            if progress_callback:
                progress_callback("Saving results...", 0.9)

            # Save to database with ML predictions
            if self.db_manager:
                try:
                    # Check for duplicate analysis first
                    existing_id = self.db_manager.check_duplicate_analysis(
                        result.metadata.model,
                        result.metadata.serial,
                        result.metadata.file_date
                    )
                    
                    if existing_id:
                        self.logger.info(
                            f"Skipping database save - duplicate analysis found (ID: {existing_id})"
                        )
                        result.db_id = existing_id
                    else:
                        result.db_id = self.db_manager.save_analysis(result)
                        self.logger.info(f"Saved to database with ID: {result.db_id}")
                except Exception as e:
                    self.logger.error(f"Database save failed: {e}")

            # Cache result
            if self._cache_enabled:
                self._file_cache[file_hash] = result

            if progress_callback:
                progress_callback("Complete", 1.0)

            self.logger.info(
                f"Completed {file_path.name}: {overall_status.value} "
                f"({result.processing_time:.2f}s)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Processing failed for {file_path.name}: {e}")
            raise ProcessingError(f"Failed to process {file_path.name}: {str(e)}")

    # Add new methods for ML integration
    def _apply_ml_predictions(self, result: AnalysisResult, ml_predictions: Dict[str, Any]) -> None:
        """Apply ML predictions to analysis result."""
        # Store raw predictions
        result.ml_predictions = ml_predictions

        # Update track-level predictions
        for track_id, track_data in result.tracks.items():
            if track_id in ml_predictions and isinstance(ml_predictions[track_id], PredictionResult):
                pred = ml_predictions[track_id]

                # Update risk category if ML confidence is high
                if pred.confidence_score > self.config.ml.failure_prediction_confidence_threshold:
                    if track_data.failure_prediction:
                        # Update existing prediction
                        track_data.failure_prediction.risk_category = RiskCategory(pred.risk_category)
                        track_data.failure_prediction.failure_probability = pred.failure_probability

                        # Add ML confidence to contributing factors
                        track_data.failure_prediction.contributing_factors['ml_confidence'] = pred.confidence_score

                # Add ML-specific fields
                track_data.ml_anomaly_detected = pred.is_anomaly
                track_data.ml_suggested_threshold = pred.suggested_threshold
                track_data.ml_warnings = pred.warnings
                track_data.ml_recommendations = pred.recommendations

        # Update overall status if critical issues detected
        overall_pred = ml_predictions.get('overall', {})
        if overall_pred.get('overall_risk') == 'HIGH' and result.overall_status == AnalysisStatus.PASS:
            result.overall_status = AnalysisStatus.WARNING
            result.validation_issues.append("ML detected high risk despite passing traditional metrics")

    def _process_ml_warnings(self, result: AnalysisResult, ml_predictions: Dict[str, Any]) -> None:
        """Process ML warnings and add to validation issues."""
        critical_warnings = []

        for track_id, pred in ml_predictions.items():
            if track_id == 'overall':
                continue

            if isinstance(pred, PredictionResult):
                for warning in pred.warnings:
                    if warning.get('severity') == 'critical':
                        critical_warnings.append(f"{track_id}: {warning.get('message', 'Unknown warning')}")

        # Add critical warnings to validation issues
        if critical_warnings:
            result.validation_issues.extend(critical_warnings)

        # Log all recommendations
        all_recommendations = []
        for track_id, pred in ml_predictions.items():
            if isinstance(pred, PredictionResult) and pred.recommendations:
                all_recommendations.extend(pred.recommendations)

        if all_recommendations:
            # Remove duplicates and log top recommendations
            unique_recommendations = list(set(all_recommendations))
            self.logger.info(
                f"ML Recommendations for {result.metadata.filename}: {'; '.join(unique_recommendations[:3])}")

    async def _save_with_ml_predictions(
            self,
            result: AnalysisResult,
            ml_predictions: Optional[Dict[str, Any]]
    ) -> int:
        """Save analysis result with ML predictions to database."""
        # Save main analysis result
        analysis_id = self.db_manager.save_analysis(result)

        # Save ML predictions if available
        if ml_predictions and self.ml_predictor:
            try:
                db_predictions = self.ml_predictor.format_ml_predictions_for_db(
                    ml_predictions, analysis_id
                )

                for db_pred in db_predictions:
                    self.db_manager.save_ml_prediction(db_pred)

            except Exception as e:
                self.logger.error(f"Failed to save ML predictions: {e}")

        return analysis_id

    # Add method for batch ML analysis
    async def analyze_drift(self, recent_results: List[AnalysisResult]) -> Optional[Dict[str, Any]]:
        """
        Analyze manufacturing drift across recent results.

        Args:
            recent_results: List of recent analysis results

        Returns:
            Drift analysis results or None
        """
        if not self.ml_predictor or not self.config.ml.enabled:
            return None

        try:
            drift_analysis = self.ml_predictor.check_for_drift(recent_results)

            if drift_analysis.get('drift_detected'):
                self.logger.warning(f"Manufacturing drift detected: {drift_analysis.get('message', 'Unknown')}")

                # Could trigger alerts or notifications here
                if self.db_manager and drift_analysis.get('severity') == 'high':
                    # Create QA alert in database
                    pass

            return drift_analysis

        except Exception as e:
            self.logger.error(f"Drift analysis failed: {e}")
            return None

    async def process_batch(
            self,
            input_dir: Path,
            output_dir: Path,
            file_pattern: str = "*.xlsx",
            max_workers: Optional[int] = None,
            progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[AnalysisResult]:
        """Process multiple files in batch with ML analysis."""
        """
        Process multiple files in batch.

        Args:
            input_dir: Input directory containing Excel files
            output_dir: Output directory for results
            file_pattern: File pattern to match
            max_workers: Maximum parallel workers
            progress_callback: Progress callback (current, total, filename)

        Returns:
            List of analysis results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        ensure_directory(output_dir)

        # Find files
        files = list(input_dir.glob(file_pattern))
        files = [f for f in files if f.suffix.lower() in self.config.processing.file_extensions]

        # Filter out temporary files
        files = [f for f in files if not any(
            f.name.startswith(pattern.replace('*', ''))
            for pattern in self.config.processing.skip_patterns
        )]

        if not files:
            self.logger.warning(f"No files found matching {file_pattern} in {input_dir}")
            return []

        self.logger.info(f"Found {len(files)} files to process")

        # Set up processing
        max_workers = max_workers or self.config.processing.max_workers
        self._is_processing = True
        results = []

        # Create output subdirectory
        batch_output = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        ensure_directory(batch_output)

        try:
            # Process files with controlled concurrency
            semaphore = asyncio.Semaphore(max_workers)

            async def process_with_semaphore(file_path: Path, index: int) -> Optional[AnalysisResult]:
                async with semaphore:
                    if progress_callback:
                        progress_callback(index + 1, len(files), file_path.name)

                    try:
                        # Create file-specific output directory
                        file_output = batch_output / file_path.stem
                        ensure_directory(file_output)

                        # Process file
                        result = await self.process_file(file_path, file_output)
                        return result

                    except Exception as e:
                        self.logger.error(f"Failed to process {file_path.name}: {e}")
                        return None

            # Create tasks
            tasks = [
                process_with_semaphore(file_path, i)
                for i, file_path in enumerate(files)
            ]

            # Process all files
            results = await asyncio.gather(*tasks)

            # Filter out None results
            results = [r for r in results if r is not None]
            
            if results and len(results) >= 10:  # Need minimum samples for drift detection
                drift_analysis = await self.analyze_drift(results[-50:])  # Last 50 results

                if drift_analysis and drift_analysis.get('drift_detected'):
                    # Add drift warning to batch report
                    self.logger.warning("Drift detected in batch processing")

            # Generate batch report
            await self._generate_batch_report(results, batch_output)

            self.logger.info(f"Batch processing complete: {len(results)}/{len(files)} successful")

            return results

        finally:
            self._is_processing = False

    async def _extract_metadata(self, file_path: Path) -> FileMetadata:
        """Extract file metadata."""
        # Parse filename for model and serial
        filename = file_path.stem
        parts = filename.split('_')

        model = parts[0] if parts else "Unknown"
        serial = parts[1] if len(parts) > 1 else "Unknown"

        # Try to extract date and time from filename
        # Expected formats:
        # MODEL_SERIAL_TEST DATA_MM-DD-YYYY_HH-MM AM/PM
        # MODEL_SERIAL_TA_Test Data_MM-DD-YYYY_HH-MM AMTrimmed Correct
        file_date = None
        
        # Try different date extraction strategies
        # Strategy 1: Look for MM-DD-YYYY pattern
        date_pattern = r'(\d{1,2})-(\d{1,2})-(\d{4})'
        date_match = re.search(date_pattern, filename)
        
        if date_match:
            month, day, year = date_match.groups()
            
            # Look for time pattern HH-MM
            time_pattern = r'(\d{1,2})-(\d{2})'
            # Find time pattern after the date
            remaining = filename[date_match.end():]
            time_match = re.search(time_pattern, remaining)
            
            if time_match:
                hour, minute = time_match.groups()
                
                # Determine AM/PM
                am_pm = ""
                if "AM" in remaining.upper():
                    am_pm = "AM"
                elif "PM" in remaining.upper():
                    am_pm = "PM"
                
                try:
                    # Build datetime string
                    if am_pm:
                        datetime_str = f"{month}-{day}-{year} {hour}:{minute} {am_pm}"
                        file_date = datetime.strptime(datetime_str, "%m-%d-%Y %I:%M %p")
                    else:
                        # If no AM/PM, assume 24-hour format
                        datetime_str = f"{month}-{day}-{year} {hour}:{minute}"
                        file_date = datetime.strptime(datetime_str, "%m-%d-%Y %H:%M")
                    
                    self.logger.debug(f"Extracted trim date from filename: {file_date}")
                except Exception as e:
                    self.logger.warning(f"Could not parse date/time: {e}")
                    # Just use the date without time
                    try:
                        file_date = datetime.strptime(f"{month}-{day}-{year}", "%m-%d-%Y")
                        self.logger.debug(f"Extracted date only from filename: {file_date}")
                    except:
                        pass
        
        # If no date from filename, use file modification time as fallback
        if not file_date:
            stat = file_path.stat()
            file_date = datetime.fromtimestamp(stat.st_mtime)
            self.logger.debug(f"Using file modification date: {file_date}")

        metadata = FileMetadata(
            filename=file_path.name,
            file_path=file_path,
            file_date=file_date,
            model=model,
            serial=serial,
            system=SystemType.UNKNOWN  # Will be detected later
        )

        return metadata

    async def _detect_system(self, file_path: Path) -> SystemType:
        """Detect system type from file structure."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, detect_system_type, file_path)

    async def _process_tracks(
            self,
            file_path: Path,
            system_type: SystemType,
            model: str,
            progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, TrackData]:
        """Process all tracks in the file."""
        tracks = {}

        if system_type == SystemType.SYSTEM_A:
            # Multi-track system
            track_ids = await self._find_system_a_tracks(file_path)

            for i, track_id in enumerate(track_ids):
                if progress_callback:
                    progress = 0.2 + (0.6 * i / len(track_ids))
                    progress_callback(f"Processing {track_id}...", progress)

                track_data = await self._process_single_track(
                    file_path, system_type, model, track_id
                )
                if track_data:
                    tracks[track_id] = track_data

        else:
            # Single track system
            if progress_callback:
                progress_callback("Processing data...", 0.5)

            track_data = await self._process_single_track(
                file_path, system_type, model, "default"
            )
            if track_data:
                tracks["default"] = track_data

        if not tracks:
            raise DataExtractionError("No valid track data found")

        return tracks

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

    async def _process_single_track(
            self,
            file_path: Path,
            system_type: SystemType,
            model: str,
            track_id: str
    ) -> Optional[TrackData]:
        """Process a single track."""
        try:
            # Extract data based on system type
            if system_type == SystemType.SYSTEM_A:
                data = await self._extract_system_a_data(file_path, track_id)
            else:
                data = await self._extract_system_b_data(file_path)

            if not data:
                return None

            # Extract unit properties
            unit_props = await self._extract_unit_properties(
                file_path, system_type, data['sheets']
            )

            # Perform analyses
            sigma_analysis = await self._analyze_sigma(data, unit_props, model)
            linearity_analysis = await self._analyze_linearity(data, sigma_analysis)
            resistance_analysis = await self._analyze_resistance(unit_props)

            # Additional analyses
            trim_effectiveness = None
            if 'untrimmed_data' in data and 'trimmed_data' in data:
                trim_effectiveness = await self._analyze_trim_effectiveness(
                    data['untrimmed_data'], data['trimmed_data']
                )

            # Risk assessment
            failure_prediction = self._calculate_failure_prediction(
                sigma_analysis, linearity_analysis, unit_props
            )

            # Determine track status
            status = self._determine_track_status(
                sigma_analysis, linearity_analysis
            )

            # Create track data
            track_data = TrackData(
                track_id=track_id,
                status=status,
                travel_length=data.get('travel_length', 0),
                position_data=data.get('positions'),
                error_data=data.get('errors'),
                untrimmed_positions=data.get('untrimmed_data', {}).get('positions') if 'untrimmed_data' in data else None,
                untrimmed_errors=data.get('untrimmed_data', {}).get('errors') if 'untrimmed_data' in data else None,
                unit_properties=unit_props,
                sigma_analysis=sigma_analysis,
                linearity_analysis=linearity_analysis,
                resistance_analysis=resistance_analysis,
                trim_effectiveness=trim_effectiveness,
                failure_prediction=failure_prediction
            )

            return track_data

        except Exception as e:
            self.logger.error(f"Error processing track {track_id}: {e}")
            return None

    async def _extract_system_a_data(
            self,
            file_path: Path,
            track_id: str
    ) -> Optional[Dict[str, Any]]:
        """Extract data for System A track."""
        loop = asyncio.get_event_loop()

        # Find relevant sheets
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
        except Exception as e:
            self.logger.error(f"Could not read Excel file: {e}")
            return None
            
        self.logger.debug(f"Sheets found in {file_path.name}: {sheet_names}")
        sheets = {}

        for sheet in sheet_names:
            # More flexible sheet matching
            if track_id in sheet or (track_id == "TRK1" and ("1 0" in sheet or "1_0" in sheet or "SEC1" in sheet)):
                if " 0" in sheet or "_0" in sheet or sheet.endswith(" 0"):
                    sheets['untrimmed'] = sheet
                elif "TRM" in sheet or "TRIM" in sheet.upper():
                    sheets['trimmed'] = sheet

        # If no sheets found with track ID, look for general patterns
        if not sheets:
            for sheet in sheet_names:
                if " 0" in sheet or "_0" in sheet or sheet.endswith(" 0"):
                    sheets['untrimmed'] = sheet
                    break
                    
            for sheet in sheet_names:
                if "TRM" in sheet or "TRIM" in sheet.upper():
                    sheets['trimmed'] = sheet
                    break

        if not sheets:
            self.logger.warning(f"No sheets found for track {track_id} in {file_path.name}")
            return None

        self.logger.debug(f"Selected sheets for {track_id}: {sheets}")

        # Extract data from untrimmed sheet
        untrimmed_data = await loop.run_in_executor(
            None, self._extract_trim_data, file_path, sheets.get('untrimmed'), 'A'
        )

        # Extract data from trimmed sheet if available
        trimmed_data = None
        if 'trimmed' in sheets:
            trimmed_data = await loop.run_in_executor(
                None, self._extract_trim_data, file_path, sheets['trimmed'], 'A'
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

    async def _extract_system_b_data(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract data for System B."""
        loop = asyncio.get_event_loop()

        sheets = {
            'untrimmed': 'test',
            'trimmed': 'Lin Error'
        }

        # Extract data
        untrimmed_data = await loop.run_in_executor(
            None, self._extract_trim_data, file_path, sheets['untrimmed'], 'B'
        )

        trimmed_data = await loop.run_in_executor(
            None, self._extract_trim_data, file_path, sheets['trimmed'], 'B'
        )

        # Use trimmed data as primary
        data_source = trimmed_data if trimmed_data else untrimmed_data

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

            if not columns:
                return {}

            # Extract data and convert to numeric
            position_series = pd.to_numeric(df.iloc[:, columns['position']], errors='coerce')
            error_series = pd.to_numeric(df.iloc[:, columns['error']], errors='coerce')
            
            # Extract upper and lower limits if available
            upper_limits = []
            lower_limits = []
            if 'upper_limit' in columns and 'lower_limit' in columns:
                upper_series = pd.to_numeric(df.iloc[:, columns['upper_limit']], errors='coerce')
                lower_series = pd.to_numeric(df.iloc[:, columns['lower_limit']], errors='coerce')
                upper_limits = upper_series.dropna().tolist()
                lower_limits = lower_series.dropna().tolist()
            
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

            return {
                'positions': positions,
                'errors': errors,
                'upper_limits': upper_limits,
                'lower_limits': lower_limits,
                'travel_length': travel_length
            }

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
        
        return self.sigma_analyzer.analyze(analysis_data)

    async def _analyze_linearity(
            self,
            data: Dict[str, Any],
            sigma_analysis: SigmaAnalysis
    ) -> LinearityAnalysis:
        """Perform linearity analysis."""
        # Prepare data dictionary for analyzer
        analysis_data = {
            'positions': data.get('positions', []),
            'errors': data.get('errors', []),
            'upper_limits': data.get('upper_limits', []),
            'lower_limits': data.get('lower_limits', []),
            'linearity_spec': sigma_analysis.sigma_threshold
        }
        
        return self.linearity_analyzer.analyze(analysis_data)

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
        
        return self.resistance_analyzer.analyze(analysis_data)

    async def _analyze_trim_effectiveness(
            self,
            untrimmed_data: Dict[str, Any],
            trimmed_data: Dict[str, Any]
    ) -> Optional[TrimEffectiveness]:
        """Analyze trim effectiveness."""
        if not untrimmed_data or not trimmed_data:
            return None

        untrimmed_errors = untrimmed_data.get('errors', [])
        trimmed_errors = trimmed_data.get('errors', [])

        if not untrimmed_errors or not trimmed_errors:
            return None

        # Calculate RMS errors
        untrimmed_rms = np.sqrt(np.mean(np.square(untrimmed_errors)))
        trimmed_rms = np.sqrt(np.mean(np.square(trimmed_errors)))

        # Calculate improvement
        improvement = ((untrimmed_rms - trimmed_rms) / untrimmed_rms * 100) if untrimmed_rms > 0 else 0

        # Max error reduction
        untrimmed_max = max(abs(e) for e in untrimmed_errors)
        trimmed_max = max(abs(e) for e in trimmed_errors)
        max_reduction = ((untrimmed_max - trimmed_max) / untrimmed_max * 100) if untrimmed_max > 0 else 0

        return TrimEffectiveness(
            improvement_percent=improvement,
            untrimmed_rms_error=untrimmed_rms,
            trimmed_rms_error=trimmed_rms,
            max_error_reduction_percent=max_reduction
        )

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