"""
Unified Processing Engine for Laser Trim Analyzer v2.

This module consolidates 6 separate processor classes into a single
UnifiedProcessor with pluggable strategies for different processing scenarios.

Architecture (ADR-004):
- UnifiedProcessor: Main class with shared components and analysis methods
- ProcessingStrategy: Abstract interface for processing approaches
- StandardStrategy: Sequential processing (replaces LaserTrimProcessor)
- TurboStrategy: Parallel processing (replaces FastProcessor)
- MemorySafeStrategy: Chunked processing (replaces LargeScaleProcessor)
- AutoStrategy: Auto-selects best strategy based on conditions
- CachingLayer: Optional caching wrapper
- SecurityLayer: Optional security validation wrapper

Phase 2 Implementation - Processor Unification
"""

import asyncio
import gc
import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import (
    Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Type, Union
)

import numpy as np
import pandas as pd

# Core imports
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
from laser_trim_analyzer.utils.calculation_validator import (
    CalculationValidator, ValidationLevel, CalculationType
)

# Optional imports with graceful fallback
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

try:
    from laser_trim_analyzer.ml.predictors import MLPredictor
    HAS_ML = True
except ImportError:
    HAS_ML = False
    MLPredictor = None

try:
    from laser_trim_analyzer.core.security import (
        SecurityValidator, get_security_validator, validate_inputs
    )
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False

logger = logging.getLogger(__name__)


# =============================================================================
# Strategy Interface
# =============================================================================

class ProcessingStrategy(ABC):
    """
    Abstract base class for all processing strategies.

    Strategies define HOW files are processed (sequential, parallel, chunked),
    while the UnifiedProcessor handles WHAT is done (analysis, validation, storage).
    """

    def __init__(self, config: Config):
        """Initialize strategy with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging and debugging."""
        pass

    @abstractmethod
    async def process_file(
        self,
        file_path: Path,
        processor: 'UnifiedProcessor',
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AnalysisResult:
        """
        Process a single file.

        Args:
            file_path: Path to the file to process
            processor: UnifiedProcessor instance for shared methods
            progress_callback: Optional callback for progress updates

        Returns:
            AnalysisResult from processing the file
        """
        pass

    @abstractmethod
    async def process_batch(
        self,
        files: List[Path],
        processor: 'UnifiedProcessor',
        progress_callback: Optional[Callable[[str, float, int, int], None]] = None
    ) -> AsyncGenerator[AnalysisResult, None]:
        """
        Process multiple files.

        Args:
            files: List of file paths to process
            processor: UnifiedProcessor instance for shared methods
            progress_callback: Optional callback(message, progress, current, total)

        Yields:
            AnalysisResult for each processed file
        """
        pass


# =============================================================================
# Concrete Strategies
# =============================================================================

class StandardStrategy(ProcessingStrategy):
    """
    Sequential processing strategy.

    Replaces LaserTrimProcessor for standard sequential file processing.
    Best for: Small batches (<10 files), debugging, single file processing.
    """

    @property
    def name(self) -> str:
        return "standard"

    async def process_file(
        self,
        file_path: Path,
        processor: 'UnifiedProcessor',
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AnalysisResult:
        """Process file sequentially using processor's shared methods."""
        # Delegate to processor's internal processing logic
        return await processor._process_file_internal(file_path, progress_callback)

    async def process_batch(
        self,
        files: List[Path],
        processor: 'UnifiedProcessor',
        progress_callback: Optional[Callable[[str, float, int, int], None]] = None
    ) -> AsyncGenerator[AnalysisResult, None]:
        """Process files one by one sequentially."""
        total = len(files)

        for idx, file_path in enumerate(files):
            try:
                if progress_callback:
                    progress = (idx / total) * 100
                    progress_callback(f"Processing {file_path.name}...", progress, idx, total)

                result = await self.process_file(file_path, processor)
                yield result

            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                # Create error result instead of failing completely
                yield processor._create_error_result(file_path, str(e))


class TurboStrategy(ProcessingStrategy):
    """
    Parallel processing strategy using ProcessPoolExecutor.

    Replaces FastProcessor for high-performance batch processing.
    Best for: Medium batches (10-500 files), multi-core systems.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.max_workers = getattr(config.processing, 'turbo_workers', 4)
        self._executor: Optional[ProcessPoolExecutor] = None

    @property
    def name(self) -> str:
        return "turbo"

    def _get_executor(self) -> ProcessPoolExecutor:
        """Get or create the process pool executor."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self._executor

    async def process_file(
        self,
        file_path: Path,
        processor: 'UnifiedProcessor',
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AnalysisResult:
        """For single files, delegate to standard processing."""
        return await processor._process_file_internal(file_path, progress_callback)

    async def process_batch(
        self,
        files: List[Path],
        processor: 'UnifiedProcessor',
        progress_callback: Optional[Callable[[str, float, int, int], None]] = None
    ) -> AsyncGenerator[AnalysisResult, None]:
        """Process files in parallel using thread pool (async-safe)."""
        total = len(files)
        completed = 0

        # Use ThreadPoolExecutor for async compatibility
        # (ProcessPoolExecutor doesn't work well with async)
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create futures with file path tracking
            future_to_file: Dict[int, Path] = {}
            pending_futures: set = set()

            for file_path in files:
                future = loop.run_in_executor(
                    executor,
                    self._process_file_sync,
                    file_path,
                    processor
                )
                future_to_file[id(future)] = file_path
                pending_futures.add(future)

            # Process as completed using asyncio.wait
            while pending_futures:
                done, pending_futures = await asyncio.wait(
                    pending_futures,
                    return_when=asyncio.FIRST_COMPLETED
                )

                for future in done:
                    file_path = future_to_file.get(id(future), Path("unknown"))
                    try:
                        result = future.result()
                        completed += 1

                        if progress_callback:
                            progress = (completed / total) * 100
                            progress_callback(f"Completed {file_path.name}", progress, completed, total)

                        yield result

                    except Exception as e:
                        self.logger.error(f"Parallel processing failed for {file_path}: {e}")
                        completed += 1
                        yield processor._create_error_result(file_path, str(e))

    def _process_file_sync(self, file_path: Path, processor: 'UnifiedProcessor') -> AnalysisResult:
        """Synchronous wrapper for file processing."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                processor._process_file_internal(file_path, None)
            )
        finally:
            loop.close()

    def shutdown(self):
        """Shutdown the executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None


class MemorySafeStrategy(ProcessingStrategy):
    """
    Memory-managed chunked processing strategy.

    Replaces LargeScaleProcessor for handling very large batches.
    Best for: Large batches (>500 files), memory-constrained systems.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.chunk_size = getattr(config.processing, 'chunk_size', 100)
        self.memory_threshold_mb = getattr(config.processing, 'memory_threshold_mb', 1000)

    @property
    def name(self) -> str:
        return "memory_safe"

    def _get_available_memory_mb(self) -> float:
        """Get available system memory in MB."""
        if HAS_PSUTIL:
            return psutil.virtual_memory().available / (1024 * 1024)
        return float('inf')  # Assume unlimited if psutil not available

    def _enforce_memory_limit(self):
        """Check memory and force GC if needed."""
        available = self._get_available_memory_mb()
        if available < self.memory_threshold_mb:
            self.logger.warning(f"Low memory ({available:.0f}MB), forcing garbage collection")
            gc.collect()

    def _chunk_files(self, files: List[Path]) -> List[List[Path]]:
        """Split files into chunks."""
        return [
            files[i:i + self.chunk_size]
            for i in range(0, len(files), self.chunk_size)
        ]

    async def process_file(
        self,
        file_path: Path,
        processor: 'UnifiedProcessor',
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AnalysisResult:
        """For single files, just check memory and process."""
        self._enforce_memory_limit()
        return await processor._process_file_internal(file_path, progress_callback)

    async def process_batch(
        self,
        files: List[Path],
        processor: 'UnifiedProcessor',
        progress_callback: Optional[Callable[[str, float, int, int], None]] = None
    ) -> AsyncGenerator[AnalysisResult, None]:
        """Process files in memory-safe chunks."""
        chunks = self._chunk_files(files)
        total_files = len(files)
        processed = 0

        for chunk_idx, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} files)")

            # Check memory before each chunk
            self._enforce_memory_limit()

            for file_path in chunk:
                try:
                    if progress_callback:
                        progress = (processed / total_files) * 100
                        progress_callback(
                            f"Chunk {chunk_idx + 1}: {file_path.name}",
                            progress, processed, total_files
                        )

                    result = await processor._process_file_internal(file_path, None)
                    processed += 1
                    yield result

                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {e}")
                    processed += 1
                    yield processor._create_error_result(file_path, str(e))

            # GC after each chunk
            gc.collect()
            self.logger.debug(f"Chunk {chunk_idx + 1} complete, memory: {self._get_available_memory_mb():.0f}MB available")


class AutoStrategy(ProcessingStrategy):
    """
    Automatically selects the best strategy based on conditions.

    Selection criteria:
    - File count <= 10: StandardStrategy
    - File count 11-500 with adequate memory: TurboStrategy
    - File count > 500 or low memory: MemorySafeStrategy
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self._standard = StandardStrategy(config)
        self._turbo = TurboStrategy(config)
        self._memory_safe = MemorySafeStrategy(config)

    @property
    def name(self) -> str:
        return "auto"

    def _select_strategy(self, file_count: int) -> ProcessingStrategy:
        """Select best strategy based on conditions."""
        # Check available memory
        available_memory = float('inf')
        if HAS_PSUTIL:
            available_memory = psutil.virtual_memory().available / (1024 * 1024)

        # Selection logic
        if file_count <= 10:
            self.logger.info(f"Auto-selected StandardStrategy for {file_count} files")
            return self._standard
        elif file_count > 500 or available_memory < 500:
            self.logger.info(
                f"Auto-selected MemorySafeStrategy for {file_count} files "
                f"(memory: {available_memory:.0f}MB)"
            )
            return self._memory_safe
        else:
            self.logger.info(f"Auto-selected TurboStrategy for {file_count} files")
            return self._turbo

    async def process_file(
        self,
        file_path: Path,
        processor: 'UnifiedProcessor',
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AnalysisResult:
        """For single files, use standard strategy."""
        return await self._standard.process_file(file_path, processor, progress_callback)

    async def process_batch(
        self,
        files: List[Path],
        processor: 'UnifiedProcessor',
        progress_callback: Optional[Callable[[str, float, int, int], None]] = None
    ) -> AsyncGenerator[AnalysisResult, None]:
        """Auto-select strategy and process batch."""
        strategy = self._select_strategy(len(files))

        async for result in strategy.process_batch(files, processor, progress_callback):
            yield result


# =============================================================================
# Optional Layers (Decorators)
# =============================================================================

class CachingLayer:
    """
    Optional caching layer that wraps a strategy.

    Provides hash-based result caching to avoid reprocessing identical files.
    """

    def __init__(self, max_cache_size: int = 100):
        self._cache: Dict[str, AnalysisResult] = {}
        self._max_size = max_cache_size
        self._hits = 0
        self._misses = 0
        self.logger = logging.getLogger(f"{__name__}.CachingLayer")

    def get(self, file_path: Path) -> Optional[AnalysisResult]:
        """Get cached result if available."""
        try:
            file_hash = calculate_file_hash(file_path)
            if file_hash in self._cache:
                self._hits += 1
                self.logger.debug(f"Cache hit for {file_path.name}")
                return self._cache[file_hash]
        except Exception as e:
            self.logger.warning(f"Cache lookup failed: {e}")

        self._misses += 1
        return None

    def set(self, file_path: Path, result: AnalysisResult):
        """Cache a result."""
        try:
            file_hash = calculate_file_hash(file_path)

            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[file_hash] = result

        except Exception as e:
            self.logger.warning(f"Cache set failed: {e}")

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


class SecurityLayer:
    """
    Optional security validation layer.

    Validates file paths and content before processing.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SecurityLayer")
        self._validator = None
        if HAS_SECURITY:
            self._validator = get_security_validator()

    def validate(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate file for security threats.

        Returns:
            Tuple of (is_safe, error_message)
        """
        if not self._validator:
            return True, None

        try:
            result = self._validator.validate_input(
                file_path,
                'file_path',
                {
                    'require_absolute': False,
                    'allowed_extensions': ['.xlsx', '.xls', '.xlsm'],
                    'check_extension': True
                }
            )

            if not result.is_safe:
                return False, f"Security validation failed: {'; '.join(result.validation_errors)}"

            if result.threats_detected:
                return False, f"Security threat detected: {result.threats_detected[0].value}"

            return True, None

        except Exception as e:
            self.logger.warning(f"Security validation error: {e}")
            return True, None  # Fail open for availability


# =============================================================================
# Unified Processor
# =============================================================================

class UnifiedProcessor:
    """
    Unified processing engine that consolidates all processor functionality.

    Replaces:
    - LaserTrimProcessor (via StandardStrategy)
    - FastProcessor (via TurboStrategy)
    - LargeScaleProcessor (via MemorySafeStrategy)
    - CachedFileProcessor (via CachingLayer)
    - SecureFileProcessor (via SecurityLayer)

    Usage:
        processor = UnifiedProcessor(
            config,
            strategy='auto',  # or 'standard', 'turbo', 'memory_safe'
            enable_caching=True,
            enable_security=True,
            incremental=True
        )

        # Single file
        result = await processor.process_file(file_path)

        # Batch
        async for result in processor.process_batch(files):
            print(result.status)
    """

    # Strategy registry
    STRATEGIES: Dict[str, Type[ProcessingStrategy]] = {
        'standard': StandardStrategy,
        'turbo': TurboStrategy,
        'memory_safe': MemorySafeStrategy,
        'auto': AutoStrategy,
    }

    def __init__(
        self,
        config: Config,
        db_manager: Optional[DatabaseManager] = None,
        strategy: str = 'auto',
        enable_caching: bool = False,
        enable_security: bool = True,
        incremental: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the unified processor.

        Args:
            config: Application configuration
            db_manager: Database manager for result storage
            strategy: Processing strategy ('auto', 'standard', 'turbo', 'memory_safe')
            enable_caching: Enable file result caching
            enable_security: Enable security validation layer
            incremental: Skip already-processed files (uses ProcessedFile table)
            logger: Logger instance
        """
        self.config = config
        self.db_manager = db_manager
        self.incremental = incremental
        self.logger = logger or logging.getLogger(__name__)

        # Initialize strategy
        self._strategy_name = strategy
        self.strategy = self._create_strategy(strategy)

        # Initialize optional layers
        self.caching = CachingLayer() if enable_caching else None
        self.security = SecurityLayer() if enable_security else None

        # Initialize shared analyzers
        self._init_analyzers()

        # Initialize ML predictor
        self._init_ml_predictor()

        # Processing state
        self._processing_lock = threading.Lock()
        self._is_processing = False
        self._stats = {
            'files_processed': 0,
            'cache_hits': 0,
            'skipped_incremental': 0,
            'errors': 0,
            'ml_predictions': 0
        }

    def _create_strategy(self, strategy_name: str) -> ProcessingStrategy:
        """Create processing strategy instance."""
        strategy_class = self.STRATEGIES.get(strategy_name)
        if not strategy_class:
            self.logger.warning(f"Unknown strategy '{strategy_name}', using 'auto'")
            strategy_class = AutoStrategy
        return strategy_class(self.config)

    def _init_analyzers(self):
        """Initialize shared analysis components."""
        try:
            self.sigma_analyzer = SigmaAnalyzer(self.config, self.logger)
            self.linearity_analyzer = LinearityAnalyzer(self.config, self.logger)
            self.resistance_analyzer = ResistanceAnalyzer(self.config, self.logger)
            self.logger.info("Initialized analysis components")
        except Exception as e:
            self.logger.error(f"Failed to initialize analyzers: {e}")
            self.sigma_analyzer = None
            self.linearity_analyzer = None
            self.resistance_analyzer = None

        # Calculation validator
        try:
            self.calculation_validator = CalculationValidator(ValidationLevel.STANDARD)
        except Exception as e:
            self.logger.warning(f"Failed to initialize calculation validator: {e}")
            self.calculation_validator = None

    def _init_ml_predictor(self):
        """Initialize ML predictor."""
        self.ml_predictor = None

        if not HAS_ML:
            self.logger.warning("ML components not available")
            return

        try:
            self.ml_predictor = MLPredictor(self.config, logger=self.logger)

            # Initialize with timeout
            def init_ml():
                try:
                    return self.ml_predictor.initialize()
                except Exception as e:
                    self.logger.error(f"ML init failed: {e}")
                    return False

            result = [None]
            thread = threading.Thread(target=lambda: result.__setitem__(0, init_ml()), daemon=True)
            thread.start()
            thread.join(timeout=30.0)

            if not thread.is_alive() and result[0]:
                self.logger.info("ML predictor initialized")
            else:
                self.logger.warning("ML predictor initialization failed or timed out")
                self.ml_predictor = None

        except Exception as e:
            self.logger.warning(f"Failed to create ML predictor: {e}")
            self.ml_predictor = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def process_file(
        self,
        file_path: Path,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AnalysisResult:
        """
        Process a single file.

        Args:
            file_path: Path to Excel file
            output_dir: Output directory for plots (optional)
            progress_callback: Progress callback function

        Returns:
            AnalysisResult with complete analysis data
        """
        file_path = Path(file_path)

        with self._processing_lock:
            # Security check
            if self.security:
                is_safe, error = self.security.validate(file_path)
                if not is_safe:
                    raise ValidationError(error)

            # Check incremental (skip if already processed)
            if self.incremental and self.db_manager:
                if self.db_manager.is_file_processed(file_path):
                    self._stats['skipped_incremental'] += 1
                    self.logger.info(f"Skipping already-processed file: {file_path.name}")
                    # Return cached result if available
                    cached = self.db_manager.get_processed_file(file_path)
                    if cached and cached.analysis:
                        return cached.analysis

            # Check cache
            if self.caching:
                cached_result = self.caching.get(file_path)
                if cached_result:
                    self._stats['cache_hits'] += 1
                    return cached_result

            # Process via strategy
            result = await self.strategy.process_file(file_path, self, progress_callback)

            # Cache result
            if self.caching:
                self.caching.set(file_path, result)

            # Mark as processed
            if self.incremental and self.db_manager:
                self.db_manager.mark_file_processed(file_path)

            self._stats['files_processed'] += 1
            return result

    async def process_batch(
        self,
        files: List[Path],
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float, int, int], None]] = None
    ) -> AsyncGenerator[AnalysisResult, None]:
        """
        Process multiple files.

        Args:
            files: List of file paths to process
            output_dir: Output directory for plots (optional)
            progress_callback: Progress callback(message, progress%, current, total)

        Yields:
            AnalysisResult for each processed file
        """
        files = [Path(f) for f in files]

        # Filter already-processed if incremental
        if self.incremental and self.db_manager:
            original_count = len(files)
            files = self.db_manager.get_unprocessed_files(files)
            skipped = original_count - len(files)
            if skipped > 0:
                self._stats['skipped_incremental'] += skipped
                self.logger.info(f"Skipping {skipped} already-processed files")

        if not files:
            self.logger.info("No files to process")
            return

        # Process via strategy
        async for result in self.strategy.process_batch(files, self, progress_callback):
            # Cache result
            if self.caching and result.metadata:
                self.caching.set(Path(result.metadata.file_path), result)

            # Mark as processed
            if self.incremental and self.db_manager and result.metadata:
                self.db_manager.mark_file_processed(Path(result.metadata.file_path))

            self._stats['files_processed'] += 1
            yield result

    def process_file_sync(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AnalysisResult:
        """
        Synchronous wrapper for process_file method.

        This provides API compatibility with LaserTrimProcessor for use
        in thread pools and non-async contexts.

        Args:
            file_path: Path to Excel file
            output_dir: Optional directory for outputs
            progress_callback: Optional progress callback

        Returns:
            AnalysisResult object
        """
        file_path = Path(file_path)

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.process_file(file_path, output_dir, progress_callback)
                    )
                    return future.result()
        except RuntimeError:
            # No event loop exists, use asyncio.run
            pass

        # Run the async method
        return asyncio.run(self.process_file(file_path, output_dir, progress_callback))

    # -------------------------------------------------------------------------
    # Internal Processing (used by strategies)
    # -------------------------------------------------------------------------

    async def _process_file_internal(
        self,
        file_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AnalysisResult:
        """
        Internal file processing implementation.

        This method contains the actual processing logic shared by all strategies.
        It extracts data, runs analyses, and creates the result.
        """
        # This will be fully implemented in Day 2 when we extract from LaserTrimProcessor
        # For now, delegate to the existing LaserTrimProcessor for compatibility

        from laser_trim_analyzer.core.processor import LaserTrimProcessor

        legacy_processor = LaserTrimProcessor(
            config=self.config,
            db_manager=self.db_manager,
            ml_predictor=self.ml_predictor,
            logger=self.logger
        )

        return await legacy_processor.process_file(file_path, None, progress_callback)

    def _create_error_result(self, file_path: Path, error_message: str) -> AnalysisResult:
        """Create an error result for failed processing."""
        # Create minimal error result with correct field names
        return AnalysisResult(
            metadata=FileMetadata(
                filename=file_path.name,
                file_path=file_path,
                file_date=datetime.now(),
                model='Unknown',
                serial='Unknown',
                system=SystemType.SYSTEM_A,
                has_multi_tracks=False
            ),
            tracks={},
            processing_time=0,
            overall_status=AnalysisStatus.ERROR,
            validation_status=ValidationStatus.FAILED,
            error_message=error_message
        )

    # -------------------------------------------------------------------------
    # Shared Analysis Methods (used by all strategies)
    # -------------------------------------------------------------------------

    def analyze_sigma(self, track_data: TrackData) -> SigmaAnalysis:
        """Perform sigma analysis on track data."""
        if not self.sigma_analyzer:
            raise AnalysisError("Sigma analyzer not available")
        return self.sigma_analyzer.analyze(track_data)

    def analyze_linearity(self, track_data: TrackData) -> LinearityAnalysis:
        """Perform linearity analysis on track data."""
        if not self.linearity_analyzer:
            raise AnalysisError("Linearity analyzer not available")
        return self.linearity_analyzer.analyze(track_data)

    def analyze_resistance(self, track_data: TrackData) -> ResistanceAnalysis:
        """Perform resistance analysis on track data."""
        if not self.resistance_analyzer:
            raise AnalysisError("Resistance analyzer not available")
        return self.resistance_analyzer.analyze(track_data)

    def determine_overall_status(self, tracks: Dict[str, TrackData]) -> AnalysisStatus:
        """
        Determine overall status from all track analyses.

        Single implementation replacing duplicated logic in LaserTrimProcessor/FastProcessor.
        """
        if not tracks:
            return AnalysisStatus.FAILED

        statuses = []
        for track_id, track_data in tracks.items():
            if hasattr(track_data, 'status'):
                statuses.append(track_data.status)

        if not statuses:
            return AnalysisStatus.UNKNOWN

        # If any failed, overall is failed
        if any(s == AnalysisStatus.FAILED for s in statuses):
            return AnalysisStatus.FAILED

        # If any warning, overall is warning
        if any(s == AnalysisStatus.WARNING for s in statuses):
            return AnalysisStatus.WARNING

        # If all passed, overall is passed
        if all(s == AnalysisStatus.PASSED for s in statuses):
            return AnalysisStatus.PASSED

        return AnalysisStatus.UNKNOWN

    def determine_overall_validation_status(
        self, tracks: Dict[str, TrackData]
    ) -> ValidationStatus:
        """
        Determine overall validation status from all tracks.

        Single implementation replacing duplicated logic.
        """
        if not tracks:
            return ValidationStatus.FAILED

        statuses = []
        for track_id, track_data in tracks.items():
            if hasattr(track_data, 'validation_status'):
                statuses.append(track_data.validation_status)

        if not statuses:
            return ValidationStatus.UNKNOWN

        # If any failed, overall failed
        if any(s == ValidationStatus.FAILED for s in statuses):
            return ValidationStatus.FAILED

        # If any warning, overall warning
        if any(s == ValidationStatus.WARNING for s in statuses):
            return ValidationStatus.WARNING

        # All passed
        if all(s == ValidationStatus.PASSED for s in statuses):
            return ValidationStatus.PASSED

        return ValidationStatus.UNKNOWN

    # -------------------------------------------------------------------------
    # ML Failure Prediction (Phase 3 - ADR-005)
    # -------------------------------------------------------------------------

    def predict_failure(
        self,
        sigma_analysis: SigmaAnalysis,
        linearity_analysis: LinearityAnalysis,
        resistance_analysis: Optional[ResistanceAnalysis] = None
    ) -> FailurePrediction:
        """
        ML-first failure prediction with formula fallback.

        Following the ThresholdOptimizer pattern (ADR-005):
        1. Check feature flag
        2. Try ML prediction if model is trained
        3. Fall back to formula if ML unavailable
        4. Log which method was used

        Args:
            sigma_analysis: Results from sigma analysis
            linearity_analysis: Results from linearity analysis
            resistance_analysis: Optional resistance analysis results

        Returns:
            FailurePrediction with probability, risk category, and contributing factors
        """
        # Check feature flag first
        use_ml = getattr(self.config.processing, 'use_ml_failure_predictor', False)

        if not use_ml:
            self.logger.debug("ML failure predictor disabled by feature flag")
            return self._calculate_formula_failure(
                sigma_analysis, linearity_analysis, resistance_analysis
            )

        # Try ML prediction
        if self._can_use_ml_failure_predictor():
            try:
                prediction = self._predict_failure_ml(
                    sigma_analysis, linearity_analysis, resistance_analysis
                )
                if prediction:
                    self._stats['ml_predictions'] += 1
                    return prediction
            except Exception as e:
                self.logger.warning(f"ML failure prediction failed, using fallback: {e}")

        # Formula fallback
        self.logger.info("Using formula-based failure prediction (ML unavailable)")
        return self._calculate_formula_failure(
            sigma_analysis, linearity_analysis, resistance_analysis
        )

    def _can_use_ml_failure_predictor(self) -> bool:
        """Check if ML failure predictor is available and trained."""
        if not HAS_ML:
            return False

        if not self.ml_predictor:
            return False

        # Check if failure_predictor model is registered and trained
        try:
            if hasattr(self.ml_predictor, 'ml_engine'):
                engine = self.ml_predictor.ml_engine
                if hasattr(engine, 'models') and 'failure_predictor' in engine.models:
                    model = engine.models['failure_predictor']
                    return getattr(model, 'is_trained', False)
        except Exception as e:
            self.logger.debug(f"Error checking ML predictor availability: {e}")

        return False

    def _predict_failure_ml(
        self,
        sigma_analysis: SigmaAnalysis,
        linearity_analysis: LinearityAnalysis,
        resistance_analysis: Optional[ResistanceAnalysis] = None
    ) -> Optional[FailurePrediction]:
        """
        Perform ML-based failure prediction.

        Args:
            sigma_analysis: Sigma analysis results
            linearity_analysis: Linearity analysis results
            resistance_analysis: Optional resistance analysis

        Returns:
            FailurePrediction if successful, None otherwise
        """
        try:
            # Extract features for ML model
            features = self._extract_failure_features(
                sigma_analysis, linearity_analysis, resistance_analysis
            )

            # Get the failure predictor model
            engine = self.ml_predictor.ml_engine
            model = engine.models['failure_predictor']

            # Create DataFrame for prediction
            feature_df = pd.DataFrame([features])

            # Get prediction probability
            failure_prob = float(model.predict_proba(feature_df)[0])

            # Determine risk category from probability
            risk_category = self._risk_from_probability(failure_prob)

            # Get feature importance as contributing factors
            contributing_factors = self._get_contributing_factors(
                features, model
            )

            self.logger.info(
                f"ML failure prediction: prob={failure_prob:.3f}, "
                f"risk={risk_category.value}"
            )

            return FailurePrediction(
                failure_probability=failure_prob,
                risk_category=risk_category,
                gradient_margin=sigma_analysis.gradient_margin,
                contributing_factors=contributing_factors
            )

        except Exception as e:
            self.logger.error(f"ML failure prediction error: {e}")
            return None

    def _calculate_formula_failure(
        self,
        sigma_analysis: SigmaAnalysis,
        linearity_analysis: LinearityAnalysis,
        resistance_analysis: Optional[ResistanceAnalysis] = None
    ) -> FailurePrediction:
        """
        Formula-based failure prediction (fallback method).

        This replicates the logic from LaserTrimProcessor._analyze_track_data()
        for when ML is not available.
        """
        # Base probability
        failure_prob = 0.1

        # Increase probability based on failures
        if not sigma_analysis.sigma_pass:
            failure_prob += 0.3

        if not linearity_analysis.linearity_pass:
            failure_prob += 0.3

        if resistance_analysis and hasattr(resistance_analysis, 'resistance_change_percent'):
            if abs(resistance_analysis.resistance_change_percent) > 50:
                failure_prob += 0.2

        # Cap at 1.0
        failure_prob = min(failure_prob, 1.0)

        # Determine risk category
        risk_category = self._risk_from_probability(failure_prob)

        # Build contributing factors
        contributing_factors = {
            'sigma_pass': 1.0 if sigma_analysis.sigma_pass else 0.0,
            'linearity_pass': 1.0 if linearity_analysis.linearity_pass else 0.0
        }

        if resistance_analysis and hasattr(resistance_analysis, 'resistance_change_percent'):
            contributing_factors['resistance_stable'] = (
                1.0 if abs(resistance_analysis.resistance_change_percent) <= 50 else 0.0
            )

        self.logger.debug(
            f"Formula failure prediction: prob={failure_prob:.3f}, "
            f"risk={risk_category.value}"
        )

        return FailurePrediction(
            failure_probability=failure_prob,
            risk_category=risk_category,
            gradient_margin=sigma_analysis.gradient_margin,
            contributing_factors=contributing_factors
        )

    def _extract_failure_features(
        self,
        sigma_analysis: SigmaAnalysis,
        linearity_analysis: LinearityAnalysis,
        resistance_analysis: Optional[ResistanceAnalysis] = None
    ) -> Dict[str, float]:
        """
        Extract features for ML failure prediction.

        Returns a dictionary of feature names to values matching the
        FailurePredictor's expected input features.
        """
        features = {
            'sigma_gradient': sigma_analysis.sigma_gradient,
            'sigma_threshold': sigma_analysis.sigma_threshold,
            'sigma_pass': 1.0 if sigma_analysis.sigma_pass else 0.0,
            'gradient_margin': sigma_analysis.gradient_margin,
            'linearity_pass': 1.0 if linearity_analysis.linearity_pass else 0.0,
        }

        # Add linearity-specific features if available
        if hasattr(linearity_analysis, 'final_linearity_error_shifted'):
            features['final_linearity_error_shifted'] = (
                linearity_analysis.final_linearity_error_shifted or 0.0
            )
        if hasattr(linearity_analysis, 'linearity_spec'):
            features['linearity_spec'] = linearity_analysis.linearity_spec or 0.0

        # Add resistance features if available
        if resistance_analysis:
            if hasattr(resistance_analysis, 'resistance_change_percent'):
                features['resistance_change_percent'] = (
                    resistance_analysis.resistance_change_percent or 0.0
                )
            if hasattr(resistance_analysis, 'untrimmed_resistance'):
                features['untrimmed_resistance'] = (
                    resistance_analysis.untrimmed_resistance or 0.0
                )
            if hasattr(resistance_analysis, 'trimmed_resistance'):
                features['trimmed_resistance'] = (
                    resistance_analysis.trimmed_resistance or 0.0
                )

        return features

    def _risk_from_probability(self, probability: float) -> RiskCategory:
        """Convert failure probability to risk category."""
        if probability >= 0.7:
            return RiskCategory.HIGH
        elif probability >= 0.4:
            return RiskCategory.MEDIUM
        else:
            return RiskCategory.LOW

    def _get_contributing_factors(
        self,
        features: Dict[str, float],
        model
    ) -> Dict[str, float]:
        """
        Get contributing factors from ML model feature importances.

        Args:
            features: Input features dictionary
            model: Trained ML model with feature_importances_

        Returns:
            Dictionary of feature name to importance contribution
        """
        contributing_factors = {}

        try:
            if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
                importances = model.model.feature_importances_
                feature_names = list(features.keys())

                # Match importances to feature names
                for i, (name, value) in enumerate(features.items()):
                    if i < len(importances):
                        # Weighted contribution = feature value * importance
                        contributing_factors[name] = float(value * importances[i])
        except Exception as e:
            self.logger.debug(f"Could not calculate contributing factors: {e}")
            # Fall back to simple pass/fail indicators
            contributing_factors = {
                k: v for k, v in features.items()
                if k in ('sigma_pass', 'linearity_pass')
            }

        return contributing_factors

    # -------------------------------------------------------------------------
    # Drift Detection (Phase 3 - ML Integration)
    # -------------------------------------------------------------------------

    def detect_drift(
        self,
        results: List[AnalysisResult],
        window_size: int = 100
    ) -> Dict[str, Any]:
        """
        Detect manufacturing drift using ML-first with formula fallback.

        This follows the ThresholdOptimizer pattern (ADR-005):
        1. Check feature flag first
        2. Try ML prediction if model trained
        3. Fall back to statistical CUSUM method if ML not available
        4. Log which method used

        Args:
            results: List of AnalysisResult objects from historical query
            window_size: Window size for rolling analysis

        Returns:
            Dictionary containing drift analysis results:
            - drift_detected: bool
            - drift_severity: str (negligible, low, moderate, high, critical)
            - drift_rate: float (0.0 to 1.0)
            - drift_trend: str (stable, increasing, decreasing)
            - drift_points: List[Dict] (indices and values where drift detected)
            - recommendations: List[str]
            - feature_drift: Dict[str, Dict] (per-feature drift info)
            - method_used: str ('ml' or 'formula')
        """
        # Check feature flag first
        use_ml = getattr(self.config.processing, 'use_ml_drift_detector', False)

        if not use_ml:
            self.logger.debug("ML drift detector disabled by feature flag")
            return self._detect_drift_formula(results, window_size)

        # Try ML prediction
        if self._can_use_ml_drift_detector():
            try:
                report = self._detect_drift_ml(results, window_size)
                if report:
                    self._stats['ml_predictions'] = self._stats.get('ml_predictions', 0) + 1
                    return report
            except Exception as e:
                self.logger.warning(f"ML drift detection failed, using fallback: {e}")

        # Formula fallback (statistical CUSUM method)
        self.logger.info("Using formula-based drift detection (ML unavailable)")
        return self._detect_drift_formula(results, window_size)

    def _can_use_ml_drift_detector(self) -> bool:
        """Check if ML drift detector is available and trained."""
        try:
            if not self.ml_predictor:
                return False

            # Check if drift_detector model exists and is trained
            if hasattr(self.ml_predictor, 'ml_engine'):
                models = getattr(self.ml_predictor.ml_engine, 'models', {})
                drift_model = models.get('drift_detector')
                if drift_model and getattr(drift_model, 'is_trained', False):
                    return True

            return False
        except Exception as e:
            self.logger.debug(f"Error checking ML drift detector: {e}")
            return False

    def _detect_drift_ml(
        self,
        results: List[AnalysisResult],
        window_size: int
    ) -> Optional[Dict[str, Any]]:
        """
        ML-based drift detection using the trained DriftDetector model.

        Args:
            results: List of AnalysisResult objects
            window_size: Window size for rolling analysis

        Returns:
            Drift report dictionary or None if ML fails
        """
        try:
            # Extract features from results for ML model
            features_df = self._extract_drift_features(results)

            if features_df.empty or len(features_df) < 20:
                self.logger.warning(
                    f"Insufficient data for ML drift detection: {len(features_df)} samples"
                )
                return None

            # Get the drift detector model
            drift_model = self.ml_predictor.ml_engine.models['drift_detector']

            # Get comprehensive drift report from model
            report = drift_model.get_drift_report(features_df)

            # Enhance report with method info
            result = {
                'drift_detected': report['summary']['drift_detected'],
                'drift_severity': report['summary']['drift_severity'],
                'drift_rate': report['details']['overall_drift_rate'],
                'drift_trend': report['details']['drift_trend'],
                'drift_points': self._extract_drift_points(
                    features_df,
                    drift_model.predict(features_df)
                ),
                'recommendations': report['recommendations'],
                'feature_drift': report['details'].get('feature_drift', {}),
                'affected_features': report.get('affected_features', []),
                'method_used': 'ml',
                'action_required': report['summary'].get('action_required', False)
            }

            self.logger.info(
                f"ML drift detection: {result['drift_severity']} severity, "
                f"{result['drift_rate']:.1%} drift rate"
            )

            return result

        except Exception as e:
            self.logger.error(f"ML drift detection error: {e}")
            return None

    def _detect_drift_formula(
        self,
        results: List[AnalysisResult],
        window_size: int
    ) -> Dict[str, Any]:
        """
        Formula-based drift detection using CUSUM statistical method.

        This is the fallback when ML is not available.

        Args:
            results: List of AnalysisResult objects
            window_size: Window size for rolling analysis

        Returns:
            Drift report dictionary
        """
        # Extract time series data
        drift_data = []
        for result in results:
            if result.tracks:
                for track in result.tracks:
                    sigma_val = getattr(track, 'sigma_gradient', None)
                    if sigma_val is not None:
                        drift_data.append({
                            'date': result.file_date or result.timestamp,
                            'sigma_gradient': sigma_val,
                            'sigma_pass': 1 if getattr(track, 'sigma_pass', True) else 0,
                            'linearity_pass': 1 if getattr(track, 'linearity_pass', True) else 0,
                            'model': result.model
                        })

        if len(drift_data) < 20:
            return {
                'drift_detected': False,
                'drift_severity': 'insufficient_data',
                'drift_rate': 0.0,
                'drift_trend': 'insufficient_data',
                'drift_points': [],
                'recommendations': ['Need at least 20 samples for drift detection'],
                'feature_drift': {},
                'method_used': 'formula'
            }

        # Convert to DataFrame and sort by date
        df = pd.DataFrame(drift_data).sort_values('date').reset_index(drop=True)

        # Calculate CUSUM for drift detection
        window = min(window_size, len(df) // 4, 10)
        if window < 3:
            window = min(3, len(df))

        target = df['sigma_gradient'].iloc[:window].mean()
        std = df['sigma_gradient'].std()

        k = 0.5  # Slack parameter
        h = 4.0  # Decision interval

        cusum_pos = []
        cusum_neg = []
        c_pos = 0.0
        c_neg = 0.0

        for value in df['sigma_gradient']:
            c_pos = max(0, c_pos + (value - target) / (std + 1e-6) - k)
            c_neg = max(0, c_neg + (target - value) / (std + 1e-6) - k)
            cusum_pos.append(c_pos)
            cusum_neg.append(c_neg)

        df['cusum_pos'] = cusum_pos
        df['cusum_neg'] = cusum_neg
        df['cusum_max'] = np.maximum(df['cusum_pos'], df['cusum_neg'])

        # Detect drift points (where CUSUM exceeds threshold)
        drift_mask = df['cusum_max'] > h
        drift_count = drift_mask.sum()
        drift_rate = drift_count / len(df)

        # Extract drift points info
        drift_points = []
        for idx in df[drift_mask].index:
            drift_points.append({
                'index': int(idx),
                'date': str(df.loc[idx, 'date']),
                'value': float(df.loc[idx, 'sigma_gradient']),
                'cusum': float(df.loc[idx, 'cusum_max'])
            })

        # Classify severity
        severity = self._classify_drift_severity_formula(drift_rate)

        # Determine trend
        if len(df) >= 10:
            recent = df['cusum_max'].tail(len(df) // 4).mean()
            early = df['cusum_max'].head(len(df) // 4).mean()
            if recent > early * 1.2:
                trend = 'increasing'
            elif recent < early * 0.8:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'

        # Generate recommendations
        recommendations = self._generate_drift_recommendations_formula(
            drift_rate, severity, trend, df
        )

        # Calculate feature-level drift
        feature_drift = {}
        for col in ['sigma_gradient', 'sigma_pass', 'linearity_pass']:
            if col in df.columns:
                if len(df) >= window * 2:
                    early_mean = df[col].head(window).mean()
                    late_mean = df[col].tail(window).mean()
                    change_pct = ((late_mean - early_mean) / (early_mean + 1e-6)) * 100

                    feature_drift[col] = {
                        'mean_change_percent': float(change_pct),
                        'is_drifting': abs(change_pct) > 10,
                        'direction': 'increasing' if change_pct > 0 else 'decreasing'
                    }

        result = {
            'drift_detected': drift_rate > 0.1,
            'drift_severity': severity,
            'drift_rate': float(drift_rate),
            'drift_trend': trend,
            'drift_points': drift_points,
            'recommendations': recommendations,
            'feature_drift': feature_drift,
            'method_used': 'formula',
            'cusum_threshold': float(h),
            'target_value': float(target),
            'samples_analyzed': len(df)
        }

        self.logger.info(
            f"Formula drift detection: {severity} severity, "
            f"{drift_rate:.1%} drift rate ({len(drift_points)} points)"
        )

        return result

    def _extract_drift_features(
        self,
        results: List[AnalysisResult]
    ) -> pd.DataFrame:
        """
        Extract features from AnalysisResult objects for ML drift detection.

        Args:
            results: List of AnalysisResult objects

        Returns:
            DataFrame with features for each sample
        """
        records = []
        for result in results:
            if result.tracks:
                for track in result.tracks:
                    record = {
                        'sigma_gradient': getattr(track, 'sigma_gradient', None),
                        'linearity_spec': getattr(track, 'linearity_spec', None),
                        'resistance_change_percent': getattr(
                            track, 'resistance_change_percent', None
                        ),
                        'travel_length': getattr(track, 'travel_length', None),
                        'unit_length': getattr(track, 'unit_length', None),
                    }
                    # Filter out None values
                    record = {k: v for k, v in record.items() if v is not None}
                    if record:  # Only add if we have some features
                        records.append(record)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # Fill NaN with column means for ML
        df = df.fillna(df.mean())
        return df

    def _extract_drift_points(
        self,
        features_df: pd.DataFrame,
        drift_indicators: np.ndarray
    ) -> List[Dict]:
        """
        Extract drift point details from ML predictions.

        Args:
            features_df: DataFrame with features
            drift_indicators: Array of 0/1 drift indicators from ML

        Returns:
            List of drift point dictionaries
        """
        drift_points = []
        for idx in np.where(drift_indicators == 1)[0]:
            point = {
                'index': int(idx),
                'value': float(features_df.iloc[idx].get('sigma_gradient', 0))
            }
            drift_points.append(point)
        return drift_points

    def _classify_drift_severity_formula(self, drift_rate: float) -> str:
        """Classify drift severity based on drift rate."""
        if drift_rate < 0.05:
            return 'negligible'
        elif drift_rate < 0.10:
            return 'low'
        elif drift_rate < 0.15:
            return 'moderate'
        elif drift_rate < 0.25:
            return 'high'
        else:
            return 'critical'

    def _generate_drift_recommendations_formula(
        self,
        drift_rate: float,
        severity: str,
        trend: str,
        df: pd.DataFrame
    ) -> List[str]:
        """Generate recommendations based on formula drift analysis."""
        recommendations = []

        if severity in ('critical', 'high'):
            recommendations.append(
                "URGENT: Significant manufacturing drift detected. "
                "Immediate investigation required."
            )
            recommendations.append(
                "Consider halting production until root cause is identified."
            )
        elif severity == 'moderate':
            recommendations.append(
                "WARNING: Moderate drift detected. Schedule maintenance check."
            )
            recommendations.append("Increase monitoring frequency.")

        if trend == 'increasing':
            recommendations.append(
                "Drift is increasing over time. Implement corrective actions soon."
            )
        elif trend == 'decreasing':
            recommendations.append(
                "Drift is decreasing. Continue monitoring to ensure trend continues."
            )

        if not recommendations:
            recommendations.append(
                "Process appears stable. Continue standard monitoring."
            )

        return recommendations

    # -------------------------------------------------------------------------
    # Properties and Statistics
    # -------------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        stats = dict(self._stats)
        if self.caching:
            stats['cache_hit_rate'] = self.caching.hit_rate
        return stats

    @property
    def strategy_name(self) -> str:
        """Get current strategy name."""
        return self.strategy.name

    def set_strategy(self, strategy_name: str):
        """Change the processing strategy."""
        self._strategy_name = strategy_name
        self.strategy = self._create_strategy(strategy_name)
        self.logger.info(f"Changed strategy to: {strategy_name}")

    def clear_cache(self):
        """Clear the result cache."""
        if self.caching:
            self.caching.clear()
            self.logger.info("Cleared result cache")

    def reset_stats(self):
        """Reset processing statistics."""
        self._stats = {
            'files_processed': 0,
            'cache_hits': 0,
            'skipped_incremental': 0,
            'errors': 0,
            'ml_predictions': 0
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_processor(
    config: Config,
    db_manager: Optional[DatabaseManager] = None,
    strategy: str = 'auto',
    **kwargs
) -> UnifiedProcessor:
    """
    Factory function to create a UnifiedProcessor.

    Args:
        config: Application configuration
        db_manager: Database manager
        strategy: Processing strategy
        **kwargs: Additional UnifiedProcessor arguments

    Returns:
        Configured UnifiedProcessor instance
    """
    return UnifiedProcessor(
        config=config,
        db_manager=db_manager,
        strategy=strategy,
        **kwargs
    )
