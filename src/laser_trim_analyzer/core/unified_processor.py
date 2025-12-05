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
