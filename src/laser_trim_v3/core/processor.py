"""
Unified processor for Laser Trim Analyzer v3.

Combines parsing and analysis into a single processing pipeline.
Simplified from v2's 4 processor classes (~5,700 lines -> ~500 lines).

Memory-safe design for 8GB RAM systems:
- Limits concurrent processing based on available memory
- Uses generators to avoid accumulating results in memory
- Explicit garbage collection between batches
- Monitors memory and throttles if needed
"""

import gc
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Callable, Generator
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from laser_trim_v3.core.parser import ExcelParser
from laser_trim_v3.core.analyzer import Analyzer
from laser_trim_v3.core.models import (
    FileMetadata,
    TrackData,
    AnalysisResult,
    AnalysisStatus,
    ProcessingStatus,
    BatchSummary,
)
from laser_trim_v3.config import Config, get_config

logger = logging.getLogger(__name__)

# Memory thresholds for 8GB systems
MEMORY_WARNING_PERCENT = 75  # Start throttling at 75% usage
MEMORY_CRITICAL_PERCENT = 85  # Force sequential at 85% usage
MAX_WORKERS_LOW_MEMORY = 2  # Workers when memory is tight


class Processor:
    """
    Unified processor for laser trim files.

    Features:
    - Single file and batch processing
    - Incremental mode (skip already processed files)
    - Progress callbacks for UI integration
    - Auto-strategy based on file count
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.parser = ExcelParser()
        self.analyzer = Analyzer()
        self._processed_hashes: set = set()

    def process_file(self, file_path: Path, generate_plots: bool = True) -> AnalysisResult:
        """
        Process a single file.

        Args:
            file_path: Path to Excel file
            generate_plots: Whether to generate plot images

        Returns:
            AnalysisResult with all track data
        """
        start_time = time.time()
        file_path = Path(file_path)

        logger.info(f"Processing: {file_path.name}")

        try:
            # Parse file
            parsed = self.parser.parse_file(file_path)
            metadata = parsed["metadata"]
            tracks_data = parsed["tracks"]
            file_hash = parsed["file_hash"]

            if not tracks_data:
                return self._create_error_result(
                    metadata, "No valid track data found", start_time
                )

            # Analyze each track
            analyzed_tracks: List[TrackData] = []
            for track_data in tracks_data:
                track_result = self.analyzer.analyze_track(track_data)
                analyzed_tracks.append(track_result)

            # Determine overall status
            overall_status = self._determine_overall_status(analyzed_tracks)

            processing_time = time.time() - start_time

            result = AnalysisResult(
                metadata=metadata,
                overall_status=overall_status,
                processing_time=processing_time,
                tracks=analyzed_tracks,
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

        # Finalize summary
        summary.end_time = datetime.now()
        if summary.processed > 0:
            summary.pass_rate = (summary.passed / summary.processed) * 100

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
        # Filter out already processed files
        if incremental:
            files_to_process = [
                f for f in file_paths if not self._is_processed(Path(f))
            ]
            summary.skipped = len(file_paths) - len(files_to_process)
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

    def _determine_overall_status(self, tracks: List[TrackData]) -> AnalysisStatus:
        """Determine overall file status from track results."""
        if not tracks:
            return AnalysisStatus.ERROR

        statuses = [t.status for t in tracks]

        if all(s == AnalysisStatus.PASS for s in statuses):
            return AnalysisStatus.PASS
        elif any(s == AnalysisStatus.ERROR for s in statuses):
            return AnalysisStatus.ERROR
        elif any(s == AnalysisStatus.FAIL for s in statuses):
            return AnalysisStatus.FAIL
        else:
            return AnalysisStatus.WARNING

    def _update_summary(self, summary: BatchSummary, result: AnalysisResult) -> None:
        """Update batch summary with result."""
        summary.processed += 1
        summary.total_processing_time += result.processing_time

        if result.overall_status == AnalysisStatus.PASS:
            summary.passed += 1
        elif result.overall_status in (AnalysisStatus.FAIL, AnalysisStatus.ERROR):
            summary.failed += 1

        # Update average sigma
        if result.tracks:
            sigmas = [t.sigma_gradient for t in result.tracks]
            if summary.avg_sigma_gradient is None:
                summary.avg_sigma_gradient = sum(sigmas) / len(sigmas)
            else:
                # Running average
                n = summary.processed
                summary.avg_sigma_gradient = (
                    (summary.avg_sigma_gradient * (n - 1) + sum(sigmas) / len(sigmas)) / n
                )

        # Count high risk
        if any(t.risk_category.value == "High" for t in result.tracks):
            summary.high_risk_count += 1

    def _is_processed(self, file_path: Path) -> bool:
        """Check if file has already been processed."""
        # Quick check using cached hashes
        try:
            file_hash = self._quick_hash(file_path)
            return file_hash in self._processed_hashes
        except Exception:
            return False

    def _quick_hash(self, file_path: Path) -> str:
        """Calculate quick hash using file metadata + partial content."""
        stat = file_path.stat()
        # Combine size, mtime, and first 1KB
        hasher = hashlib.md5()
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(stat.st_mtime).encode())
        with open(file_path, "rb") as f:
            hasher.update(f.read(1024))
        return hasher.hexdigest()

    def _load_processed_hashes(self) -> None:
        """Load processed file hashes from database."""
        # TODO: Implement database lookup
        # For now, start fresh each session
        self._processed_hashes = set()

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
        from laser_trim_v3.core.models import SystemType

        return FileMetadata(
            filename=file_path.name,
            file_path=file_path,
            file_date=datetime.now(),
            model="Unknown",
            serial="Unknown",
            system=SystemType.UNKNOWN,
        )
