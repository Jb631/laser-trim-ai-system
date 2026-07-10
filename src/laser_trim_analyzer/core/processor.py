"""
Unified processor for Laser Trim Analyzer v3.

Combines parsing and analysis into a single processing pipeline.
Simplified from v2's 4 processor classes (~5,700 lines -> ~500 lines).

Memory-safe design for 8GB RAM systems:
- Limits concurrent processing based on available memory
- Uses generators to avoid accumulating results in memory
- Explicit garbage collection between batches
- Monitors memory and throttles if needed

ML Integration:
- Per-model thresholds from MLManager (loaded from database)
- Automatic fallback to formula when ML unavailable
"""

import gc
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Generator
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from laser_trim_analyzer.core.parser import (
    ExcelParser, NonTrimWorkbookError, detect_file_type)
from laser_trim_analyzer.core.analyzer import Analyzer
from laser_trim_analyzer.core.models import (
    FileMetadata,
    TrackData,
    AnalysisResult,
    AnalysisStatus,
    ProcessingStatus,
    BatchSummary,
    SystemType,
)
from laser_trim_analyzer.config import Config, get_config
from laser_trim_analyzer.core.final_test_parser import FinalTestParser
from laser_trim_analyzer.core.smoothness_parser import SmoothnessParser, is_smoothness_file
from laser_trim_analyzer.utils.hashing import calculate_file_hash

logger = logging.getLogger(__name__)

# Memory thresholds. NOTE: these are the ACTUAL trigger points (the previous
# comments claimed 75/85, which was misleading). If an 8GB target needs earlier
# throttling, lower these constants -- don't just edit the comments.
MEMORY_WARNING_PERCENT = 90   # Throttle (reduce workers) above this RAM usage %
MEMORY_CRITICAL_PERCENT = 95  # Force sequential processing above this RAM usage %
MAX_WORKERS_LOW_MEMORY = 2    # Workers when memory is tight


class Processor:
    """
    Unified processor for laser trim files.

    Features:
    - Single file and batch processing
    - Incremental mode (skip already processed files)
    - Progress callbacks for UI integration
    - Auto-strategy based on file count
    - Per-model ML thresholds (loaded from database)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        use_ml: bool = True,
    ):
        """
        Initialize processor.

        Args:
            config: Configuration object
            use_ml: Whether to attempt loading ML thresholds from database
        """
        self.config = config or get_config()
        self.parser = ExcelParser()
        self.final_test_parser = FinalTestParser()  # For Final Test files
        self.smoothness_parser = SmoothnessParser()  # For Output Smoothness files
        self._processed_hashes: set = set()
        self._processed_filenames: Optional[set] = None  # None = not loaded yet
        # path -> (file_size, mtime_epoch) for ProcessedFile rows. Lets
        # _is_processed skip re-hashing (full file read) when the on-disk
        # stat still matches what we recorded at processing time.
        self._processed_stat: Dict[str, tuple] = {}
        # basename -> [(stored_path, size|None, mtime_ts|None)]: rescue index
        # for path-form changes (see _is_processed). Filled by
        # _load_processed_hashes; counters log what the rescue did per batch.
        self._processed_basename: Dict[str, list] = {}
        self._scan_adopted = 0
        self._scan_rebound = 0
        # (file_hash, size, mtime datetime) tuples queued when a hash-confirm
        # succeeded but the recorded stat was missing/stale — flushed once per
        # batch so the NEXT scan takes the fast stat path.
        self._stat_heal: List[tuple] = []

        # Load per-model thresholds and predictors from database
        self._model_thresholds: Dict[str, float] = {}
        self._model_predictors: Dict = {}  # model_name -> ModelPredictor
        # Composite trim-risk models (lazy-loaded per model, keyed by model name).
        # False sentinel means "checked and absent/non-deployed", to avoid repeated
        # filesystem lookups for models that don't have a deployed model yet.
        self._composite_models: Dict = {}
        # Storage path for composite risk pickle files (mirrors MLManager convention).
        self.ml_storage_path = Path("data/ml_models")
        if use_ml:
            self._load_ml_thresholds()

        # Create analyzer with per-model thresholds
        self.analyzer = Analyzer(
            model_thresholds=self._model_thresholds
        )

    def _load_ml_thresholds(self) -> None:
        """Load trained per-model thresholds from database."""
        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.ml import get_shared_ml_manager

            db = get_database()
            ml_manager = get_shared_ml_manager(db)

            # Extract thresholds from trained models
            for model_name in ml_manager.trained_models:
                optimizer = ml_manager.threshold_optimizers.get(model_name)
                if optimizer and optimizer.is_calculated:
                    self._model_thresholds[model_name] = optimizer.threshold

            # Extract trained predictors for failure probability
            for model_name, predictor in ml_manager.predictors.items():
                if predictor.is_trained:
                    self._model_predictors[model_name] = predictor

            if self._model_thresholds:
                logger.info(f"Loaded ML thresholds for {len(self._model_thresholds)} models")
            else:
                logger.debug("No trained ML thresholds found, using formula")

            if self._model_predictors:
                logger.info(f"Loaded ML predictors for {len(self._model_predictors)} models")

        except Exception as e:
            logger.debug(f"Could not load ML thresholds: {e}")
            self._model_thresholds = {}
            self._model_predictors = {}

    def process_file(self, file_path: Path, generate_plots: bool = True) -> Optional[AnalysisResult]:
        """
        Process a single file (trim or final test).

        Automatically detects file type and routes to appropriate handler.

        Args:
            file_path: Path to Excel file
            generate_plots: Whether to generate plot images

        Returns:
            AnalysisResult with all track data (for trim files)
            or special final_test result marker
        """
        start_time = time.time()
        file_path = Path(file_path)

        logger.info(f"Processing: {file_path.name}")

        # Detect file type
        file_type = detect_file_type(file_path)

        if file_type == "non_trim":
            logger.info(f"Skipping non-trim file: {file_path.name}")
            self._mark_file_skipped(file_path)
            return None

        if file_type == "final_test":
            return self._process_final_test_file(file_path, start_time)

        if file_type == "smoothness":
            return self._process_smoothness_file(file_path, start_time)

        # Process as trim file (existing logic)
        try:
            # Parse file
            try:
                parsed = self.parser.parse_file(file_path)
            except NonTrimWorkbookError as e:
                # Parameter/report workbook named like test data — a known
                # non-data layout. Skip like non_trim; never an ERROR row.
                logger.info(f"Skipping parameter/report workbook {file_path.name}: {e}")
                self._mark_file_skipped(file_path)
                return None
            metadata = parsed["metadata"]
            tracks_data = parsed["tracks"]
            file_hash = parsed["file_hash"]

            if not tracks_data:
                return self._create_error_result(
                    metadata, "No valid track data found", start_time
                )

            # Look up full spec (linearity type + angle spec + tol + tol_type)
            # for this model. These drive the slope-from-tolerance rule in
            # the analyzer.
            spec = self._get_spec_for_analysis(metadata.model, is_final_test=False)
            linearity_type = spec["linearity_type"]

            # Analyze each track (pass model for ML threshold lookup)
            analyzed_tracks: List[TrackData] = []
            predictor = self._model_predictors.get(metadata.model)
            for track_data in tracks_data:
                # Test-sweep-only files (no laser-trim runs) skip analysis —
                # sigma/linearity aren't defined without a trim result. We
                # still record the track so the untrimmed sweep is visible
                # in the app and the file isn't silently dropped.
                if track_data.get("is_untrimmed_only"):
                    untrimmed_positions = track_data.get("untrimmed_positions") or []
                    untrimmed_errors = track_data.get("untrimmed_errors") or []
                    track_result = TrackData(
                        track_id=track_data.get("track_id", "default"),
                        status=AnalysisStatus.UNTRIMMED,
                        travel_length=track_data.get("travel_length") or 0.0,
                        linearity_spec=track_data.get("linearity_spec") or 0.0,
                        sigma_gradient=None,
                        sigma_threshold=None,
                        sigma_pass=None,
                        optimal_offset=None,
                        linearity_error=None,
                        linearity_pass=None,
                        linearity_fail_points=0,
                        unit_length=track_data.get("unit_length"),
                        untrimmed_resistance=track_data.get("untrimmed_resistance"),
                        trimmed_resistance=None,
                        measured_electrical_angle=track_data.get("measured_electrical_angle"),
                        station_compensation=track_data.get("station_compensation"),
                        linearity_type=str(linearity_type.value) if hasattr(linearity_type, "value") else (str(linearity_type) if linearity_type else None),
                        trim_pass_count=track_data.get("trim_pass_count", 0),
                        theory_volts=track_data.get("theory_volts"),
                        test_volts=track_data.get("test_volts"),
                        untrimmed_positions=untrimmed_positions or None,
                        untrimmed_errors=untrimmed_errors or None,
                    )
                    analyzed_tracks.append(track_result)
                    continue

                track_data["exclude_points"] = spec["exclude_points"]
                track_result = self.analyzer.analyze_track(
                    track_data,
                    model=metadata.model,
                    linearity_type=linearity_type,
                    angle_spec=spec["angle_spec"],
                    angle_tol=spec["angle_tol"],
                    angle_tol_type=spec["angle_tol_type"],
                    station_compensation=track_data.get("station_compensation"),
                )

                # 8340 dual-spec reclassification: operators trim every unit on
                # the 8340 sheet, then classify post-trim — units passing the
                # tight ±0.02 spec stay as 8340; units that fail tight but pass
                # the wider 8340-3 ±0.05 spec become 8340-3; units failing both
                # stay as 8340 (a standard fail). Only triggers when the parsed
                # model is "8340" (exact) AND a wide spec was extracted from
                # cols 9/10 of the Lin Error sheet.
                if (
                    metadata.model == "8340"
                    and not track_result.linearity_pass
                    and track_data.get("upper_limits_wide")
                ):
                    wide_data = dict(track_data)
                    wide_data["upper_limits"] = track_data["upper_limits_wide"]
                    wide_data["lower_limits"] = track_data["lower_limits_wide"]
                    wide_data["linearity_spec"] = (
                        track_data["linearity_spec_wide"] or track_result.linearity_spec
                    )
                    wide_spec = self._get_spec_for_analysis("8340-3", is_final_test=False)
                    wide_result = self.analyzer.analyze_track(
                        wide_data,
                        model="8340-3",
                        linearity_type=wide_spec["linearity_type"],
                        angle_spec=wide_spec["angle_spec"],
                        angle_tol=wide_spec["angle_tol"],
                        angle_tol_type=wide_spec["angle_tol_type"],
                        station_compensation=track_data.get("station_compensation"),
                    )
                    if wide_result.linearity_pass:
                        logger.info(
                            f"{file_path.name}: reclassified 8340 -> 8340-3 "
                            f"(passed wider ±{track_data['linearity_spec_wide']:.3f} spec)"
                        )
                        track_result = wide_result
                        metadata.model = "8340-3"
                        # Refresh predictor for the new model so the ML
                        # override below uses the 8340-3 model's predictor.
                        predictor = self._model_predictors.get("8340-3")

                # Override failure_probability with ML predictor if available
                if predictor:
                    try:
                        lin_error = abs(track_result.linearity_error or 0.0)
                        lin_spec = track_result.linearity_spec or 0.01
                        sigma = track_result.sigma_gradient or 0.0
                        features = {
                            'sigma_gradient': sigma,
                            'linearity_error': lin_error,
                            'fail_points': track_result.linearity_fail_points or 0,
                            'optimal_offset': track_result.optimal_offset or 0.0,
                            'linearity_spec': lin_spec,
                            'sigma_to_spec': sigma / lin_spec if lin_spec > 0 else 0.0,
                            'error_to_spec': lin_error / lin_spec if lin_spec > 0 else 0.0,
                        }
                        prob = predictor.predict_failure_probability(features)
                        if prob is not None:
                            track_result.failure_probability = prob
                            # Update risk category to match new probability
                            from laser_trim_analyzer.core.models import RiskCategory
                            from laser_trim_analyzer.utils.constants import (
                                HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD
                            )
                            if prob >= HIGH_RISK_THRESHOLD:
                                track_result.risk_category = RiskCategory.HIGH
                            elif prob >= MEDIUM_RISK_THRESHOLD:
                                track_result.risk_category = RiskCategory.MEDIUM
                            else:
                                track_result.risk_category = RiskCategory.LOW
                    except Exception as e:
                        logger.debug(f"ML predictor failed for track {track_result.track_id}: {e}")

                # Composite trim-risk score for the live unit (drift early-warning).
                try:
                    _model = metadata.model
                    crm = self._composite_models.get(_model)
                    if crm is None:
                        from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
                        _p = self.ml_storage_path / "composite_risk" / f"{_model}.pkl"
                        crm = CompositeRiskModel.load(_p) if _p.exists() else False
                        self._composite_models[_model] = crm
                    if crm and crm.is_trained and crm.result and crm.result.deployed:
                        track_result.composite_trim_risk_score = crm.predict_proba({
                            "untrimmed_error_max": track_result.untrimmed_error_max,
                            "untrimmed_sigma_gradient": track_result.untrimmed_sigma_gradient,
                            "resistance_change_percent": getattr(track_result, "resistance_change_percent", None),
                            "trim_pass_count": track_result.trim_pass_count,
                        })
                except Exception:
                    pass  # scoring is non-essential; never fail a unit over it

                analyzed_tracks.append(track_result)

            # Determine overall status
            overall_status = self._determine_overall_status(analyzed_tracks)

            # Validate track data quality
            quality_issues = self._validate_track_data(analyzed_tracks)
            # Future-dated file (mistyped date in the filename, or a wrong
            # station clock): one such FT record dated 5 months ahead skewed
            # the dashboard trend (2026-07-08). Flag it at the source.
            if (metadata.file_date is not None
                    and metadata.file_date > datetime.now() + timedelta(days=1)):
                quality_issues.append(
                    f"future-dated file ({metadata.file_date:%Y-%m-%d}) — "
                    f"check the date in the filename")
            data_quality = "suspect" if quality_issues else "good"
            if quality_issues:
                logger.warning(
                    f"Data quality issues in {file_path.name}: {', '.join(quality_issues)}"
                )

            processing_time = time.time() - start_time

            result = AnalysisResult(
                metadata=metadata,
                overall_status=overall_status,
                processing_time=processing_time,
                tracks=analyzed_tracks,
                data_quality=data_quality,
                data_quality_issues=quality_issues,
            )

            logger.info(f"Completed: {file_path.name} - {overall_status.value} "
                       f"({processing_time:.2f}s)")

            return result

        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            return self._create_error_result(
                self._create_minimal_metadata(file_path),
                f"File not found: {e}",
                start_time
            )
        except Exception as e:
            logger.exception(f"Error processing {file_path.name}: {e}")
            return self._create_error_result(
                self._create_minimal_metadata(file_path),
                str(e),
                start_time
            )

    def _process_final_test_file(self, file_path: Path, start_time: float) -> AnalysisResult:
        """
        Process a Final Test file.

        Parses the file, saves to database, and returns a special marker result.
        Final Test files don't go through the same analysis pipeline as trim files.

        Args:
            file_path: Path to Final Test Excel file
            start_time: Processing start time

        Returns:
            AnalysisResult with file_type='final_test' marker
        """
        from laser_trim_analyzer.database import get_database

        try:
            # Parse the Final Test file
            parsed = self.final_test_parser.parse_file(file_path)

            metadata = parsed["metadata"]
            tracks = parsed["tracks"]
            test_results = parsed["test_results"]
            file_hash = parsed["file_hash"]

            # Add file path to metadata
            metadata["file_path"] = str(file_path)

            processing_time = time.time() - start_time

            # Create minimal metadata for result
            minimal_metadata = FileMetadata(
                filename=metadata.get("filename", file_path.name),
                file_path=str(file_path),
                model=metadata.get("model") or "unknown",
                serial=metadata.get("serial") or "unknown",
                system=SystemType.UNKNOWN,  # Final test doesn't have system type
                file_date=metadata.get("file_date"),
            )

            # Handle Final Test files with no track data
            # This can happen with some file formats or parsing failures
            if not tracks:
                logger.warning(f"Final Test file has no track data: {file_path.name}")
                # Still save the header row so the file is tracked as processed
                db = get_database()
                db.save_final_test(
                    metadata=metadata,
                    tracks=tracks,
                    test_results=test_results,
                    file_hash=file_hash,
                )
                error_result = self._create_error_result(
                    minimal_metadata,
                    "Final Test file has no track data",
                    start_time
                )
                error_result.file_type = "final_test"  # Prevent saving as trim record
                return error_result

            # Look up full spec for FT analysis. Use the FT-specific resolver
            # so multi-section parts (e.g. 8508) pick up the per-section spec
            # based on the trailing letter on the serial (e.g. '31B' -> 8508-B).
            ft_model = metadata.get("model", "unknown")
            ft_serial = metadata.get("serial")
            ft_spec = self._get_spec_for_analysis(ft_model, ft_serial, is_final_test=True)
            ft_linearity_type = ft_spec["linearity_type"]
            ft_compensation = metadata.get("station_compensation")

            # Run analyzer BEFORE saving so slope/offset/linearity_type flow into
            # the final_test_tracks rows. The analyzer output is used both for
            # the display result (analyzed_tracks) and for enriching the raw
            # track dicts passed to save_final_test.
            analyzed_tracks = []
            for track in tracks:
                # Handle None values explicitly (dict.get returns None if key exists with None value).
                # Format 2 FT files have no spec limits and the parser returns None — treat as
                # FAIL rather than silently calling unknown-status units PASS.
                linearity_pass = track.get("linearity_pass")
                if linearity_pass is None:
                    logger.warning(
                        f"FT track {track.get('track_id', '?')} of {file_path.name}: "
                        f"linearity_pass unknown (no spec limits) — defaulting to FAIL"
                    )
                    linearity_pass = False

                # Use analyzer for spec-aware optimization when we have error data
                positions = track.get("positions") or track.get("electrical_angles") or []
                errors = track.get("errors") or []
                upper_lims = track.get("upper_limits") or []
                lower_lims = track.get("lower_limits") or []

                if positions and errors and upper_lims and lower_lims:
                    # Full analysis through analyzer
                    track_dict = {
                        "track_id": track.get("track_id", "default"),
                        "positions": positions,
                        "errors": errors,
                        "upper_limits": upper_lims,
                        "lower_limits": lower_lims,
                        "travel_length": max(positions) - min(positions) if positions else 1.0,
                        "linearity_spec": track.get("linearity_spec") or 0.01,
                    }
                    # Pass theory_values so the analyzer can run slope optimization
                    # (adjusted = error + theory * k + offset)
                    theory_vals = track.get("theory_values")
                    if theory_vals:
                        track_dict["theory_volts"] = theory_vals
                    track_dict["exclude_points"] = ft_spec["exclude_points"]
                    track_result = self.analyzer.analyze_track(
                        track_dict,
                        model=ft_model,
                        linearity_type=ft_linearity_type,
                        angle_spec=ft_spec["angle_spec"],
                        angle_tol=ft_spec["angle_tol"],
                        angle_tol_type=ft_spec["angle_tol_type"],
                        station_compensation=track.get("station_compensation") or ft_compensation,
                    )
                    analyzed_tracks.append(track_result)

                    # Enrich the raw parser track dict with spec-aware values so
                    # save_final_test can persist them on final_test_tracks.
                    track["optimal_offset"] = getattr(track_result, "optimal_offset", 0.0)
                    track["optimal_slope"] = getattr(track_result, "optimal_slope", 0.0)
                    track["linearity_type"] = (
                        str(ft_linearity_type.value) if hasattr(ft_linearity_type, "value")
                        else (str(ft_linearity_type) if ft_linearity_type else None)
                    )
                    # Overwrite the parser's raw-error fail count with the
                    # analyzer's corrected-error count. Pass/fail is judged
                    # on corrected errors (error + theory*k + offset), not raw,
                    # so this is what should land in the DB.
                    corrected_fail_points = getattr(track_result, "linearity_fail_points", None)
                    if corrected_fail_points is not None:
                        track["linearity_fail_points"] = corrected_fail_points
                    corrected_lin_pass = getattr(track_result, "linearity_pass", None)
                    if corrected_lin_pass is not None:
                        track["linearity_pass"] = corrected_lin_pass
                    corrected_lin_error = getattr(track_result, "linearity_error", None)
                    if corrected_lin_error is not None:
                        track["linearity_error"] = corrected_lin_error
                else:
                    # Minimal TrackData when no error data available
                    track_data = TrackData(
                        track_id=track.get("track_id", "default"),
                        status=AnalysisStatus.PASS if linearity_pass else AnalysisStatus.FAIL,
                        travel_length=1.0,
                        linearity_spec=track.get("linearity_spec") or 0.01,
                        sigma_gradient=0.0,
                        sigma_threshold=0.01,
                        sigma_pass=True,
                        optimal_offset=0.0,
                        linearity_error=track.get("linearity_error") or 0.0,
                        linearity_pass=linearity_pass,
                        linearity_fail_points=track.get("linearity_fail_points") or 0,
                        linearity_type=ft_linearity_type,
                        station_compensation=track.get("station_compensation") or ft_compensation,
                        position_data=positions,
                        error_data=errors,
                    )
                    analyzed_tracks.append(track_data)

                    # Enrich with identity correction so the columns are always
                    # populated for every track.
                    track["optimal_offset"] = 0.0
                    track["optimal_slope"] = 0.0
                    track["linearity_type"] = (
                        str(ft_linearity_type.value) if hasattr(ft_linearity_type, "value")
                        else (str(ft_linearity_type) if ft_linearity_type else None)
                    )

            # Determine overall status from CORRECTED analyzer results so the
            # top-level pass/fail reflects pass/fail on corrected errors
            # (raw * slope + offset vs spec limits), not the parser's raw count.
            overall_status = AnalysisStatus.PASS
            for at in analyzed_tracks:
                if getattr(at, "linearity_pass", True) is False:
                    overall_status = AnalysisStatus.FAIL
                    break
            test_results = dict(test_results)
            test_results["linearity_pass"] = (overall_status == AnalysisStatus.PASS)

            # Save to database (now with enriched tracks)
            db = get_database()
            final_test_id = db.save_final_test(
                metadata=metadata,
                tracks=tracks,
                test_results=test_results,
                file_hash=file_hash
            )

            result = AnalysisResult(
                metadata=minimal_metadata,
                overall_status=overall_status,
                processing_time=processing_time,
                tracks=analyzed_tracks,
            )

            # Mark this as a final test file for special handling
            result.file_type = "final_test"
            result.final_test_id = final_test_id

            logger.info(f"Processed Final Test: {file_path.name} - {overall_status.value} "
                       f"(ID: {final_test_id}, {processing_time:.2f}s)")

            return result

        except Exception as e:
            logger.exception(f"Error processing Final Test {file_path.name}: {e}")
            error_result = self._create_error_result(
                self._create_minimal_metadata(file_path),
                f"Final Test error: {e}",
                start_time
            )
            error_result.file_type = "final_test"  # Prevent saving as trim record
            return error_result

    def process_batch(
        self,
        file_paths: List[Path],
        progress_callback: Optional[Callable[[ProcessingStatus], None]] = None,
        incremental: bool = True,
    ) -> Generator[AnalysisResult, None, BatchSummary]:
        """
        Process multiple files with progress reporting.

        Args:
            file_paths: List of file paths
            progress_callback: Called with status updates
            incremental: Skip already processed files

        Yields:
            AnalysisResult for each file

        Returns:
            BatchSummary after processing completes
        """
        total_files = len(file_paths)
        summary = BatchSummary(total_files=total_files, start_time=datetime.now())

        logger.info(f"Starting batch processing: {total_files} files, "
                   f"incremental={incremental}")

        # Load processed hashes if incremental
        if incremental:
            self._load_processed_hashes()

        # Choose strategy based on file count
        turbo_threshold = self.config.processing.turbo_mode_threshold
        use_parallel = total_files >= turbo_threshold

        if use_parallel:
            logger.info(f"Using parallel processing ({total_files} >= {turbo_threshold})")
            yield from self._process_parallel(
                file_paths, progress_callback, incremental, summary
            )
        else:
            logger.info(f"Using sequential processing ({total_files} < {turbo_threshold})")
            yield from self._process_sequential(
                file_paths, progress_callback, incremental, summary
            )

        # Persist any stat repairs collected during the incremental scan (rows
        # whose content matched by hash but whose recorded size/mtime was
        # missing or stale). One write, non-fatal — next scan gets the fast path.
        if self._scan_adopted or self._scan_rebound:
            logger.info(
                f"Incremental scan: {self._scan_rebound} known files re-matched under a new "
                f"path form, {self._scan_adopted} legacy records adopted by unique filename "
                "(database predates the stat fast-path)")
            self._scan_adopted = self._scan_rebound = 0
        if self._stat_heal:
            try:
                from laser_trim_analyzer.database import get_database
                get_database().update_processed_file_stats(self._stat_heal)
                logger.info(f"Repaired stat records for {len(self._stat_heal)} processed files")
            except Exception as e:
                logger.debug(f"Could not persist stat repairs: {e}")
            finally:
                self._stat_heal = []

        # Finalize summary. Yield is over the GRADEABLE population (files that
        # actually have a trim result); UNTRIMMED test-sweeps are excluded from
        # the denominator so the rate isn't diluted.
        summary.end_time = datetime.now()
        gradeable = summary.gradeable_count
        if gradeable > 0:
            summary.pass_rate = (summary.passed / gradeable) * 100

        logger.info(f"Batch complete: {summary.processed}/{total_files} processed, "
                   f"{summary.passed} passed, {summary.failed} failed")

        return summary

    def _process_sequential(
        self,
        file_paths: List[Path],
        progress_callback: Optional[Callable],
        incremental: bool,
        summary: BatchSummary,
    ) -> Generator[AnalysisResult, None, None]:
        """Process files sequentially with memory management."""
        gc_interval = 50  # Run GC every 50 files

        for i, file_path in enumerate(file_paths):
            file_path = Path(file_path)

            # Check if already processed
            if incremental and self._is_processed(file_path):
                summary.skipped += 1
                if progress_callback:
                    progress_callback(ProcessingStatus(
                        filename=file_path.name,
                        status="skipped",
                        message="Already processed",
                        progress_percent=(i + 1) / len(file_paths) * 100,
                    ))
                continue

            # Check memory and pause if critical
            if self._check_memory_critical():
                logger.warning("Memory critical - forcing garbage collection")
                gc.collect()
                time.sleep(0.5)  # Brief pause to let OS reclaim

            # Report progress
            if progress_callback:
                progress_callback(ProcessingStatus(
                    filename=file_path.name,
                    status="processing",
                    progress_percent=i / len(file_paths) * 100,
                ))

            # Process
            result = self.process_file(file_path)

            # Skip non-trim files (process_file returns None for these)
            if result is None:
                summary.skipped += 1
                if progress_callback:
                    progress_callback(ProcessingStatus(
                        filename=file_path.name,
                        status="skipped",
                        message="Non-trim file skipped",
                        progress_percent=(i + 1) / len(file_paths) * 100,
                    ))
                continue

            # Update summary
            self._update_summary(summary, result)

            # Report completion
            if progress_callback:
                progress_callback(ProcessingStatus(
                    filename=file_path.name,
                    status="completed",
                    progress_percent=(i + 1) / len(file_paths) * 100,
                    result=result,
                ))

            yield result

            # Periodic garbage collection
            if (i + 1) % gc_interval == 0:
                gc.collect()
                logger.debug(f"GC after {i + 1} files")

    def _process_parallel(
        self,
        file_paths: List[Path],
        progress_callback: Optional[Callable],
        incremental: bool,
        summary: BatchSummary,
    ) -> Generator[AnalysisResult, None, None]:
        """
        Process files in parallel with memory-aware throttling.

        On 8GB systems, limits workers and monitors memory to prevent crashes.
        Falls back to sequential processing if memory is critical.
        """
        # Filter out already processed files (fast filename-based check)
        if incremental:
            # Report scanning progress for large batches
            if progress_callback and len(file_paths) > 100:
                progress_callback(ProcessingStatus(
                    filename="",
                    status="scanning",
                    message=f"Scanning {len(file_paths)} files against database...",
                    progress_percent=0,
                ))

            # _processed_filenames / _processed_hashes were loaded once before this
            # block. The parallel workers below only READ them via _is_processed — no
            # mutation during the parallel section, so no lock is required. If you
            # ever add mutation here, switch to a lock-protected set or freeze first.
            files_to_process = []
            for _i, _f in enumerate(file_paths):
                if not self._is_processed(Path(_f)):
                    files_to_process.append(_f)
                # Heartbeat: statting 170k files on a share takes minutes —
                # silence reads as a lockup (work finding, 2026-07-10).
                if progress_callback and _i % 2000 == 1999:
                    progress_callback(ProcessingStatus(
                        filename="", status="scanning",
                        message=f"Checking against database… {_i + 1:,}/{len(file_paths):,}",
                        progress_percent=0))
            summary.skipped = len(file_paths) - len(files_to_process)

            # Sanity guard, v2 (2026-07-10). v1 aborted on "matched 0 of N"
            # — which also described a folder of 349 GENUINELY NEW files and
            # blocked James's normal daily batch at work. The reliable wrong-
            # path/wrong-database signal is different: many files whose
            # FILENAMES the database already knows failing recognition anyway.
            # A folder of truly new files has zero known names and sails
            # through; an empty database has no known names either (first
            # run). Unchecking incremental stays the explicit full-reprocess
            # override.
            if summary.skipped == 0 and len(self._processed_filenames) >= 1000:
                known_name_misses = sum(
                    1 for f in files_to_process
                    if Path(f).name in self._processed_basename)
                if known_name_misses >= 50:
                    raise RuntimeError(
                        f"{known_name_misses} of {len(files_to_process)} files have filenames "
                        "the database already knows, yet NONE were recognized as processed. "
                        "The folder is probably browsed under a different path than before, or "
                        "the app is pointed at the wrong database. Refusing to reprocess "
                        "everything. (To force a full reprocess, uncheck incremental mode.)")

            if progress_callback:
                # Always announced (work finding #11: "processing page doesn't
                # tell me how many new files") — this is the number the user
                # is waiting for before anything starts parsing.
                progress_callback(ProcessingStatus(
                    filename="",
                    status="scanning",
                    message=f"Found {len(files_to_process)} new files to process ({summary.skipped} already in database)",
                    progress_percent=0,
                ))
        else:
            files_to_process = list(file_paths)

        if not files_to_process:
            logger.info("No new files to process")
            return

        # Determine worker count based on available memory
        max_workers = self._get_safe_worker_count(len(files_to_process))
        logger.info(f"Using {max_workers} workers for parallel processing")

        # If memory is already critical, fall back to sequential
        if self._check_memory_critical():
            logger.warning("Memory critical - falling back to sequential processing")
            yield from self._process_sequential(
                [Path(f) for f in files_to_process],
                progress_callback, False, summary
            )
            return

        completed = 0
        batch_size = 20  # Process in batches to control memory

        for batch_start in range(0, len(files_to_process), batch_size):
            batch = files_to_process[batch_start:batch_start + batch_size]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit batch tasks
                future_to_file = {
                    executor.submit(self.process_file, Path(f)): f
                    for f in batch
                }

                # Process as completed
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    completed += 1

                    try:
                        result = future.result()

                        # Skip non-trim files (process_file returns None)
                        if result is None:
                            summary.skipped += 1
                            if progress_callback:
                                progress_callback(ProcessingStatus(
                                    filename=Path(file_path).name,
                                    status="skipped",
                                    message="Non-trim file skipped",
                                    progress_percent=completed / len(files_to_process) * 100,
                                ))
                            continue

                        self._update_summary(summary, result)

                        if progress_callback:
                            progress_callback(ProcessingStatus(
                                filename=Path(file_path).name,
                                status="completed",
                                progress_percent=completed / len(files_to_process) * 100,
                                result=result,
                            ))

                        yield result

                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        # Count as processed-with-error so the buckets sum to
                        # `processed`, matching the sequential path (where
                        # process_file returns an ERROR result via _update_summary).
                        summary.processed += 1
                        summary.errors += 1

                        if progress_callback:
                            progress_callback(ProcessingStatus(
                                filename=Path(file_path).name,
                                status="failed",
                                message=str(e),
                                progress_percent=completed / len(files_to_process) * 100,
                            ))

            # GC between batches
            gc.collect()

            # Check memory and reduce workers if needed
            if self._check_memory_warning():
                logger.warning("Memory warning - reducing workers")
                max_workers = max(1, max_workers - 1)

    def _get_safe_worker_count(self, file_count: int) -> int:
        """Determine safe number of workers based on available memory."""
        if not HAS_PSUTIL:
            return min(2, file_count)  # Conservative default

        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)

            # For 8GB systems with ~4GB available
            if available_gb < 2:
                return 1  # Very low memory - sequential
            elif available_gb < 4:
                return 2  # Low memory - minimal parallelism
            elif available_gb < 6:
                return 3  # Moderate memory
            else:
                return min(4, file_count)  # Good memory

        except Exception:
            return 2  # Safe default

    def _check_memory_critical(self) -> bool:
        """Check if memory usage is critical (>85%)."""
        if not HAS_PSUTIL:
            return False
        try:
            return psutil.virtual_memory().percent > MEMORY_CRITICAL_PERCENT
        except Exception:
            return False

    def _check_memory_warning(self) -> bool:
        """Check if memory usage is high (>75%)."""
        if not HAS_PSUTIL:
            return False
        try:
            return psutil.virtual_memory().percent > MEMORY_WARNING_PERCENT
        except Exception:
            return False

    @staticmethod
    def _validate_track_data(tracks: List[TrackData]) -> List[str]:
        """
        Validate analyzed track data for quality issues.

        Checks each track for signs of bad/corrupt data. Returns a list of
        issue descriptions. Empty list = all checks passed.

        These checks flag suspect data — they don't reject records. The
        data_quality field lets downstream analyses filter them out.
        """
        issues = []

        for track in tracks:
            tid = track.track_id

            # Check 1: Negative sigma gradient (should always be >= 0).
            # None is valid for UNTRIMMED tracks; only flag actual negatives.
            if track.sigma_gradient is not None and track.sigma_gradient < 0:
                issues.append(f"{tid}: negative sigma_gradient ({track.sigma_gradient:.4f})")

            # Check 2: All-zero error data (element wasn't actually measured)
            if track.error_data:
                if all(v == 0 or v is None for v in track.error_data):
                    issues.append(f"{tid}: all-zero error data")

            # Check 3: Position array too short (incomplete measurement)
            if track.position_data:
                if len(track.position_data) < 10:
                    issues.append(f"{tid}: position array too short ({len(track.position_data)} points)")

            # Check 4: Position and error array length mismatch
            if track.position_data and track.error_data:
                if len(track.position_data) != len(track.error_data):
                    issues.append(
                        f"{tid}: array length mismatch "
                        f"(position={len(track.position_data)}, error={len(track.error_data)})"
                    )

            # Check 5: Scale-anomalous linearity error. The file carries its
            # own spec band; a linearity error 10x beyond that band is a unit/
            # scale corruption, not a real measurement (observed in production:
            # error=10.007 against a ±0.05 band — ~380σ — which alone set a
            # +16σ drift headline on a stable model). Flag, don't reject.
            band_vals = [abs(v) for v in ((track.upper_limits or []) +
                                          (track.lower_limits or [])) if v is not None]
            band = max(band_vals) if band_vals else None
            if band and band > 0 and track.linearity_error is not None:
                if track.linearity_error > 10.0 * band:
                    issues.append(
                        f"{tid}: scale-anomalous linearity error "
                        f"({track.linearity_error:.4g} vs spec band ±{band:.4g})"
                    )

        return issues

    def _process_smoothness_file(self, file_path: Path, start_time: float) -> AnalysisResult:
        """Process an Output Smoothness file."""
        from laser_trim_analyzer.database import get_database

        try:
            parsed = self.smoothness_parser.parse_file(file_path)
            metadata = parsed["metadata"]
            tracks = parsed["tracks"]
            file_hash = parsed["file_hash"]

            # Refuse to save empty parses. The previous code silently saved a
            # fake "Pass" track with zeroed smoothness values when the parser
            # returned [], which is why every record showed Max Smoothness:
            # 0.0000. Better to error loudly so the user knows the file
            # format isn't recognised.
            if not tracks:
                raise ValueError(
                    f"Smoothness parser returned no tracks for {file_path.name}. "
                    f"The file format may not be recognised. Check the column "
                    f"layout matches the expected Betatronix or generic format."
                )

            db = get_database()
            result_id = db.save_smoothness_result(
                metadata=metadata, tracks=tracks, file_hash=file_hash
            )

            processing_time = time.time() - start_time

            minimal_metadata = FileMetadata(
                filename=metadata.get("filename", file_path.name),
                file_path=str(file_path),
                model=metadata.get("model", "unknown"),
                serial=metadata.get("serial", "unknown"),
                system=SystemType.UNKNOWN,
                file_date=metadata.get("file_date"),
            )

            overall_status = AnalysisStatus.PASS
            if any(not t.get("smoothness_pass", True) for t in tracks):
                overall_status = AnalysisStatus.FAIL

            # Create minimal TrackData objects mirroring the smoothness tracks.
            # tracks is guaranteed non-empty by the check above.
            analyzed_tracks = [
                TrackData(
                    track_id=t.get("track_id", "default"),
                    status=AnalysisStatus.PASS if t.get("smoothness_pass", True) else AnalysisStatus.FAIL,
                    travel_length=1.0, linearity_spec=0.01,
                    sigma_gradient=0.0, sigma_threshold=0.01, sigma_pass=True,
                    optimal_offset=0.0, linearity_error=0.0,
                    linearity_pass=True, linearity_fail_points=0,
                )
                for t in tracks
            ]

            result = AnalysisResult(
                metadata=minimal_metadata, overall_status=overall_status,
                processing_time=processing_time, tracks=analyzed_tracks,
            )
            result.file_type = "smoothness"
            result.smoothness_id = result_id

            logger.info(
                f"Processed Smoothness: {file_path.name} - {overall_status.value} "
                f"(ID: {result_id}, {processing_time:.2f}s)"
            )
            return result

        except Exception as e:
            logger.exception(f"Error processing Smoothness {file_path.name}: {e}")
            error_result = self._create_error_result(
                self._create_minimal_metadata(file_path),
                f"Smoothness error: {e}", start_time
            )
            error_result.file_type = "smoothness"  # Prevent saving as trim record
            return error_result

    def _get_linearity_type(self, model: str) -> Optional[str]:
        """Look up linearity type from model_specs table."""
        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            spec = db.get_model_spec(model)
            if spec:
                return spec.get("linearity_type")
        except Exception as e:
            logger.debug(f"Could not look up model spec for {model}: {e}")
        return None

    def _get_spec_for_analysis(
        self,
        model: str,
        serial: Optional[str] = None,
        is_final_test: bool = False,
    ) -> Dict[str, Optional[object]]:
        """
        Look up the spec fields the analyzer needs for slope+offset
        optimization: (linearity_type, angle_spec, angle_tol, angle_tol_type).

        For Final Test records we use resolve_spec_for_ft so multi-section
        parts like 8508 (stored as 8508-A, -B, -C, -D in model_specs) map to
        the right section based on the trailing letter on the serial.

        Returns a dict with keys: linearity_type, angle_spec, angle_tol,
        angle_tol_type. All values may be None if the model isn't in
        model_specs yet.
        """
        empty = {
            "linearity_type": None,
            "angle_spec": None,
            "angle_tol": None,
            "angle_tol_type": None,
            "exclude_points": None,
        }
        if not model:
            return empty
        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
            if is_final_test:
                spec = db.resolve_spec_for_ft(model, serial)
            else:
                spec = db.get_model_spec(model)
            if not spec:
                return empty

            # Use FT-specific exclude points when analyzing FT files,
            # fall back to trim exclude points if FT field is empty.
            if is_final_test:
                exclude = spec.get("exclude_points_ft") or spec.get("exclude_points")
            else:
                exclude = spec.get("exclude_points")

            return {
                "linearity_type": spec.get("linearity_type"),
                "angle_spec": spec.get("electrical_angle"),
                "angle_tol": spec.get("electrical_angle_tol"),
                "angle_tol_type": spec.get("electrical_angle_tol_type"),
                "exclude_points": exclude,
            }
        except Exception as e:
            logger.debug(f"Could not look up model spec for {model}: {e}")
            return empty

    def _determine_overall_status(self, tracks: List[TrackData]) -> AnalysisStatus:
        """Determine overall file status from track results."""
        if not tracks:
            return AnalysisStatus.ERROR

        statuses = [t.status for t in tracks]

        # UNTRIMMED-only files (every track is just a test sweep) get their
        # own top-level status so the GUI / queries can recognise them.
        if all(s == AnalysisStatus.UNTRIMMED for s in statuses):
            return AnalysisStatus.UNTRIMMED

        # Mixed: ignore UNTRIMMED tracks when judging pass/fail — they don't
        # have a trim result to grade. Decision is based on the real tracks.
        judged = [s for s in statuses if s != AnalysisStatus.UNTRIMMED]
        if not judged:
            # Shouldn't happen (caught above), but fall back safely.
            return AnalysisStatus.UNTRIMMED

        if all(s == AnalysisStatus.PASS for s in judged):
            return AnalysisStatus.PASS
        elif any(s == AnalysisStatus.ERROR for s in judged):
            return AnalysisStatus.ERROR
        elif any(s == AnalysisStatus.FAIL for s in judged):
            return AnalysisStatus.FAIL
        else:
            return AnalysisStatus.WARNING

    def _update_summary(self, summary: BatchSummary, result: AnalysisResult) -> None:
        """Update batch summary with result."""
        summary.processed += 1
        summary.total_processing_time += result.processing_time

        # Single source of truth for status bucketing (incl. the UNTRIMMED
        # bucket, which was previously dropped -- diluting pass_rate and
        # making counts not sum to `processed`).
        summary.record_status(result.overall_status)

        # Update average sigma. UNTRIMMED tracks carry sigma_gradient=None
        # because no trim ran; filter them out before averaging. Skip the
        # update entirely if every track in this result is untrimmed.
        if result.tracks:
            sigmas = [t.sigma_gradient for t in result.tracks if t.sigma_gradient is not None]
            if sigmas:
                file_avg = sum(sigmas) / len(sigmas)
                if summary.avg_sigma_gradient is None:
                    summary.avg_sigma_gradient = file_avg
                else:
                    # Running average across processed files
                    n = summary.processed
                    summary.avg_sigma_gradient = (
                        (summary.avg_sigma_gradient * (n - 1) + file_avg) / n
                    )

        # Count high risk
        if any(t.risk_category.value == "High" for t in result.tracks):
            summary.high_risk_count += 1

        # Count anomalies (trim failures with linear slope pattern)
        if any(getattr(t, 'is_anomaly', False) for t in result.tracks):
            summary.anomalies += 1

    def get_unprocessed_count(self, file_paths: List[Path], clear_cache: bool = True) -> tuple:
        """
        Get count of files that need processing (not yet in database).

        Memory-optimized for 8GB systems - clears cache after counting.

        Args:
            file_paths: List of file paths to check
            clear_cache: If True, clears the path cache after counting to free memory

        Returns:
            Tuple of (unprocessed_count, already_processed_count)
        """
        # Only load from DB if not already cached
        if self._processed_filenames is None:
            self._load_processed_hashes()

        already_processed = 0

        # Use set intersection for efficiency (avoids per-item lookup)
        file_path_strs = {str(p) for p in file_paths}
        already_processed = len(file_path_strs & self._processed_filenames)
        unprocessed = len(file_paths) - already_processed

        # Free memory on constrained systems
        if clear_cache:
            self._processed_filenames = None

        return unprocessed, already_processed

    def _is_processed(self, file_path: Path) -> bool:
        """Check if file has already been processed.

        Identity is the CONTENT hash, not the path. A path miss is a cheap
        early-out (a brand-new filename can't have been processed); a path hit
        is confirmed by hash so a re-export of NEW content to a reused filename
        is processed rather than silently skipped.

        Falls back to the (hash-based) DB query only if the cache wasn't loaded.
        """
        try:
            # If cache was loaded (even if empty), use it.
            if self._processed_filenames is not None:
                path_str = str(file_path)
                # Full-path miss is NOT "definitely new" anymore. The work
                # incident (2026-07-09): the same share browsed under a
                # different path form (mapped drive vs UNC vs new root) missed
                # on every file and the app set out to reprocess the entire
                # history. Rescue by BASENAME (filenames carry model_serial_
                # datetime, so they identify the export): confirm cheaply by
                # recorded (size, mtime); legacy rows with no recorded stat
                # (pre-fast-path databases) are ADOPTED on a unique basename
                # match — hashing tens of thousands of files over the share
                # is the lockup we're preventing. Ambiguity falls back to the
                # hash identity check.
                if path_str not in self._processed_filenames:
                    entries = self._processed_basename.get(file_path.name)
                    if not entries:
                        return False        # truly new filename
                    try:
                        st = file_path.stat()
                    except OSError:
                        return False        # can't stat -> let it process
                    for _stored, size, mtime in entries:
                        if (size is not None and mtime is not None
                                and st.st_size == size
                                and abs(st.st_mtime - mtime) <= 2.0):
                            # Same name, same size, same mtime: the file we
                            # already processed, reached via a new path form.
                            self._processed_filenames.add(path_str)
                            self._processed_stat[path_str] = (st.st_size, st.st_mtime)
                            self._scan_rebound += 1
                            return True
                    if len(entries) == 1 and entries[0][1] is None:
                        # Single legacy record without stats (database predates
                        # the stat fast-path). Trust the unique name; record
                        # the observed stat so future scans are exact.
                        self._processed_filenames.add(path_str)
                        self._processed_stat[path_str] = (st.st_size, st.st_mtime)
                        self._scan_adopted += 1
                        return True
                    # Ambiguous (same name in several folders / stat mismatch):
                    # fall through to the hash identity check below.
                    try:
                        file_hash = calculate_file_hash(file_path)
                    except Exception:
                        return False
                    if file_hash in self._processed_hashes:
                        self._processed_filenames.add(path_str)
                        self._stat_heal.append((
                            file_hash, st.st_size,
                            datetime.fromtimestamp(st.st_mtime),
                        ))
                        self._processed_stat[path_str] = (st.st_size, st.st_mtime)
                        self._scan_rebound += 1
                        return True
                    return False

                # Stat fast-path: if the recorded size AND mtime still match the
                # file on disk, the content can't have changed — skip without
                # reading the file at all. This is what makes re-scanning a
                # mostly-processed network folder fast; hashing every path-hit
                # meant reading every byte of every known file over the share.
                # mtime tolerance 2s covers FAT/SMB timestamp granularity.
                recorded = self._processed_stat.get(path_str)
                current_stat = None
                if recorded is not None:
                    try:
                        current_stat = file_path.stat()
                    except OSError:
                        return False  # can't stat -> don't skip; let it process
                    rec_size, rec_mtime = recorded
                    if (current_stat.st_size == rec_size
                            and abs(current_stat.st_mtime - rec_mtime) <= 2.0):
                        return True

                # Stat missing or stale -> CONFIRM content by hash before
                # skipping. The file_hash is the identity, not the path, so a
                # re-export of new content to a fixed filename is NOT dropped.
                try:
                    file_hash = calculate_file_hash(file_path)
                except Exception:
                    return False  # can't hash -> don't skip; let it process
                if file_hash in self._processed_hashes:
                    # Same content, stale/missing stat record — queue a repair
                    # so the NEXT scan takes the fast stat path for this file.
                    try:
                        st = current_stat or file_path.stat()
                        self._stat_heal.append((
                            file_hash, st.st_size,
                            datetime.fromtimestamp(st.st_mtime),
                        ))
                        self._processed_stat[path_str] = (st.st_size, st.st_mtime)
                    except OSError:
                        pass
                    return True
                return False

            # Cache not loaded - query database directly (already hash-based).
            from laser_trim_analyzer.database import get_database
            db = get_database()
            return db.is_file_processed(file_path)

        except Exception as e:
            logger.warning(f"Could not check if file is processed: {e}")
            return False

    def _mark_file_skipped(self, file_path: Path) -> None:
        """Record a non-trim file as processed so it's skipped on future runs.

        Prevents re-opening and re-checking junk files every time the same
        folder is processed — important for network drives with many files.
        """
        try:
            from laser_trim_analyzer.database import get_database

            db = get_database()
            file_hash = calculate_file_hash(file_path)
            stat = file_path.stat()

            db.mark_file_skipped(
                filename=file_path.name,
                file_path=str(file_path),
                file_hash=file_hash,
                file_size=stat.st_size,
                file_modified_date=datetime.fromtimestamp(stat.st_mtime),
            )

            # Update in-memory cache if loaded (path + content hash).
            if self._processed_filenames is not None:
                self._processed_filenames.add(str(file_path))
                if self._processed_hashes is not None:
                    self._processed_hashes.add(file_hash)

        except Exception as e:
            logger.debug(f"Could not record skipped file {file_path.name}: {e}")

    def _load_processed_hashes(self) -> None:
        """Load processed file info from database into memory cache.

        Loads full file paths for O(1) lookup during batch processing.
        Only loads successfully processed files - errors will be retried.

        IMPORTANT: Uses full file_path (not just filename) to handle duplicate
        filenames in different folders correctly. For example:
        - FolderA/test.xls and FolderB/test.xls are different files
        """
        self._processed_filenames = set()  # Full paths, not just filenames
        self._processed_hashes = set()
        # basename -> [(stored_path, size|None, mtime_ts|None)] — the rescue
        # index for path-form changes (drive letter vs UNC, new mount root).
        # Work incident 2026-07-09: the SAME share browsed under a different
        # path form made every full-path lookup miss, so the app tried to
        # reprocess tens of thousands of known files.
        self._processed_basename = {}
        self._scan_adopted = 0
        self._scan_rebound = 0
        try:
            from laser_trim_analyzer.database import get_database
            from laser_trim_analyzer.database.models import ProcessedFile as DBProcessedFile
            from laser_trim_analyzer.database.models import FinalTestResult

            db = get_database()
            with db.session() as session:
                # Load paths AND content hashes for successfully processed files.
                # The hash is the identity (skip only when content matches); the
                # path is a cheap early-out. Errors (success=False) are excluded
                # so they're retried.
                rows = session.query(
                    DBProcessedFile.file_path, DBProcessedFile.file_hash,
                    DBProcessedFile.file_size, DBProcessedFile.file_modified_date
                ).filter(DBProcessedFile.success == True).all()
                self._processed_filenames = set(r.file_path for r in rows if r.file_path)
                self._processed_hashes = set(r.file_hash for r in rows if r.file_hash)
                from pathlib import PurePath
                for r in rows:
                    if r.file_path:
                        mt = (r.file_modified_date.timestamp()
                              if r.file_modified_date is not None else None)
                        self._processed_basename.setdefault(
                            PurePath(r.file_path).name, []
                        ).append((r.file_path, r.file_size, mt))
                # Stat fast-path (ProcessedFile rows only — FT/smoothness tables
                # don't carry size/mtime, so those fall back to hash-confirm).
                self._processed_stat = {
                    r.file_path: (r.file_size, r.file_modified_date.timestamp())
                    for r in rows
                    if r.file_path and r.file_size is not None
                    and r.file_modified_date is not None
                }

                # Also load Final Test file paths + hashes (always "successful" if in DB)
                ft_rows = session.query(
                    FinalTestResult.file_path, FinalTestResult.file_hash
                ).all()
                ft_count = 0
                for row in ft_rows:
                    if row.file_path:
                        self._processed_filenames.add(row.file_path)
                        self._processed_basename.setdefault(
                            PurePath(row.file_path).name, []
                        ).append((row.file_path, None, None))
                        ft_count += 1
                    if row.file_hash:
                        self._processed_hashes.add(row.file_hash)

                # Also load Smoothness file paths
                smoothness_count = 0
                try:
                    from laser_trim_analyzer.database.models import SmoothnessResult as DBSmoothnessResult
                    smoothness_rows = session.query(
                        DBSmoothnessResult.file_path, DBSmoothnessResult.file_hash
                    ).all()
                    for row in smoothness_rows:
                        if row.file_path:
                            self._processed_filenames.add(row.file_path)
                            self._processed_basename.setdefault(
                                PurePath(row.file_path).name, []
                            ).append((row.file_path, None, None))
                            smoothness_count += 1
                        if row.file_hash:
                            self._processed_hashes.add(row.file_hash)
                except Exception as e:
                    logger.debug(f"Could not load smoothness paths: {e}")

            logger.info(f"Loaded {len(self._processed_filenames)} processed file paths "
                       f"({len(self._processed_filenames) - ft_count - smoothness_count} trim, "
                       f"{ft_count} final test, {smoothness_count} smoothness)")
        except Exception as e:
            logger.warning(f"Could not load processed files from database: {e}")
            self._processed_filenames = set()
            self._processed_stat = {}
            self._processed_basename = {}

    def _create_error_result(
        self, metadata: FileMetadata, error_msg: str, start_time: float
    ) -> AnalysisResult:
        """Create an error result."""
        return AnalysisResult(
            metadata=metadata,
            overall_status=AnalysisStatus.ERROR,
            processing_time=time.time() - start_time,
            tracks=[],
            errors=[error_msg],
        )

    def _create_minimal_metadata(self, file_path: Path) -> FileMetadata:
        """Create minimal metadata for error cases."""
        from laser_trim_analyzer.core.models import SystemType

        return FileMetadata(
            filename=file_path.name,
            file_path=file_path,
            file_date=datetime.now(),
            model="Unknown",
            serial="Unknown",
            system=SystemType.UNKNOWN,
        )
