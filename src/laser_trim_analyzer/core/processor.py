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
import threading
import time
from collections import deque
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Generator, Tuple
import logging
from concurrent.futures import BrokenExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout   # builtin only from 3.11

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from laser_trim_analyzer.core.parser import (
    ExcelParser, NonTrimWorkbookError, detect_file_type)
from laser_trim_analyzer.core.analyzer import Analyzer
from laser_trim_analyzer.core.ft_regrade import grade_ft_track
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
from laser_trim_analyzer.utils.hashing import calculate_file_hash, shares_one_stat, stat_once
from laser_trim_analyzer.database.manager import FinalTestWrite, SkipMarkerWrite, SmoothnessWrite
from laser_trim_analyzer.database.specs import SpecSnapshot

logger = logging.getLogger(__name__)

# Memory thresholds. NOTE: these are the ACTUAL trigger points (the previous
# comments claimed 75/85, which was misleading). If an 8GB target needs earlier
# throttling, lower these constants -- don't just edit the comments.
MEMORY_WARNING_PERCENT = 90   # Throttle (reduce workers) above this RAM usage %
MEMORY_CRITICAL_PERCENT = 95  # Force sequential processing above this RAM usage %
MAX_WORKERS_LOW_MEMORY = 2    # Workers when memory is tight

# Statuses that assert a customer-facing disposition.
_GRADEABLE_STATUSES = (AnalysisStatus.PASS, AnalysisStatus.WARNING, AnalysisStatus.FAIL)


def enforce_measurement_backed_verdict(track, source: str = "") -> Optional[str]:
    """A graded track must carry the measurement its grade came from.

    PASS/WARNING/FAIL is a customer disposition on a zero-tolerance
    characteristic. Without position/error arrays there is nothing behind that
    claim: it cannot be plotted, re-graded, or defended in an audit, and it is
    exactly the shape Fix Missing Tracks exists to repair. So an array-less
    graded track is an ungradeable READ, not a quiet verdict — demote it to
    ERROR and withdraw the pass flags, the same call the parser already makes
    when a file's limit columns are unusable (linearity_spec_warning).

    The verdict is withdrawn, never inverted. Marking it FAIL would invent a
    linearity rejection the unit never earned; NULL flags are honest and are
    what the repair tool looks for.

    UNTRIMMED tracks are untouched: a test sweep with no laser-trim run has no
    trimmed arrays BY DESIGN (the parser moves its sweep into untrimmed_* and
    clears the trimmed arrays), and it claims no verdict to back up. Keying on
    the TRACK's own status rather than the file's is what keeps a normal
    two-track unit — one trimmed track, one untrimmed sweep — out of this.

    Mutates `track` in place. Returns the reason it fired, or None.
    """
    if track.status not in _GRADEABLE_STATUSES:
        return None
    if track.position_data and track.error_data:
        return None

    reason = (f"graded {track.status.value} with no measurement "
              f"(positions={len(track.position_data or [])}, "
              f"errors={len(track.error_data or [])})")
    logger.error(f"Ingest guard: track {track.track_id}"
                 f"{' of ' + source if source else ''} {reason} — recording as "
                 f"ERROR (ungraded) instead of storing an unbacked verdict")
    track.status = AnalysisStatus.ERROR
    track.linearity_pass = None
    track.sigma_pass = None
    # Record the reason the same way the parser does (linearity_spec_warning),
    # only when the track didn't already carry one -- so error_reason_of()
    # (below) can find it without a third field to check. Never overwrites an
    # existing warning: that text is what actually happened first.
    track.linearity_spec_warning = track.linearity_spec_warning or reason
    return reason


def error_reason_of(tracks, overall_status) -> Optional[str]:
    """Why an ERROR result is an ERROR, from the tracks that made it one. None otherwise.

    The words already exist on the track (the analyzer's linearity_spec_warning or
    anomaly_reason); this only brings them to the one place the app asks."""
    if overall_status != AnalysisStatus.ERROR:
        return None
    parts = []
    for t in tracks:
        if getattr(t, "status", None) != AnalysisStatus.ERROR:
            continue
        why = getattr(t, "linearity_spec_warning", None) or getattr(t, "anomaly_reason", None)
        if why:
            parts.append(f"{t.track_id}: {why}")
    return ("; ".join(parts)[:500]) or "ERROR with no recorded reason"


ML_FALLBACK_WARNING = (
    "ML state could not be loaded ({why}): sigma thresholds fall back to the formula defaults, "
    "and no failure-probability predictor runs, for every model it did not load -- the ingest "
    "goes on (sigma is a drift signal, never a rejection)")


def load_ml_state(db) -> Tuple[Dict[str, float], Dict[str, object]]:
    """(sigma thresholds, trained predictors) per model -- the ML state the analysis uses.

    The one extraction rule, for a Processor built without a snapshot (`_load_ml_thresholds`) and
    for `take_spec_snapshot` alike. Never fatal, as it never was: any failure means no ML state,
    and the analyzer falls back to its formula thresholds. But never SILENT (review m-3, ruling
    of 2026-09-25): a failed load WARNS, naming what failed -- once per call, which the ingest
    makes once per folder (its snapshot). It does not refuse the folder: predictors never load on
    the Mac by design, and sigma never rejects a unit.
    """
    thresholds: Dict[str, float] = {}
    predictors: Dict[str, object] = {}
    try:
        from laser_trim_analyzer.ml import get_shared_ml_manager
        ml_manager = get_shared_ml_manager(db)
        # The manager swallows its own load failures (and the shared cache serves the half-loaded
        # manager for five minutes): it records why, and this is where that is said.
        failed = getattr(ml_manager, "load_error", None)
        if failed:
            logger.warning(ML_FALLBACK_WARNING.format(why=failed))

        # Extract thresholds from trained models
        for model_name in ml_manager.trained_models:
            optimizer = ml_manager.threshold_optimizers.get(model_name)
            if optimizer and optimizer.is_calculated:
                thresholds[model_name] = optimizer.threshold

        # Extract trained predictors for failure probability
        for model_name, predictor in ml_manager.predictors.items():
            if predictor.is_trained:
                predictors[model_name] = predictor

        if thresholds:
            logger.info(f"Loaded ML thresholds for {len(thresholds)} models")
        else:
            logger.debug("No trained ML thresholds found, using formula")

        if predictors:
            logger.info(f"Loaded ML predictors for {len(predictors)} models")

    except Exception as e:
        logger.warning(ML_FALLBACK_WARNING.format(why=f"{type(e).__name__}: {e}"))
        return {}, {}
    return thresholds, predictors


def take_spec_snapshot(*, use_ml: bool = True, db=None) -> SpecSnapshot:
    """What the analysis reads from the database, read ONCE: at folder start (ruling 16).

    Every model spec, and with `use_ml` the ML thresholds and predictors -- from the database the
    Processor's own lookups use (`get_database()`) unless one is named. A Processor built with it
    answers every spec question from it and never asks the database; a spec edited while a folder
    runs reaches the next folder's snapshot. Plain, picklable data (see SpecSnapshot). A failure to
    read the specs RAISES: analysing a folder spec-less would store different numbers without a
    word (spec §4.3).
    """
    if db is None:
        from laser_trim_analyzer.database import get_database
        db = get_database()
    thresholds, predictors = load_ml_state(db) if use_ml else ({}, {})
    return SpecSnapshot(specs=tuple(db.get_all_model_specs()),
                        ml_thresholds=thresholds, ml_predictors=predictors)


def is_content_refusal(exc: Optional[BaseException]) -> bool:
    """Did the FILE's own content cause this save error? (Ruling of 2026-09-25.)

    Only such an error -- a validation refusal (a ValueError, such as "Serial cannot be empty"),
    or a malformed or duplicate unit (an IntegrityError) -- may record a file as unreadable.
    Anything else a save raises -- OperationalError, any other database or driver error, disk
    I/O, schema drift, a bug -- the save cannot attribute to the file: the database refused it,
    not the file. Such a file is NEVER marked; it stays new, is counted as an error, and feeds the
    ingest's stop rule. None (a batch-level failure, no exception of the file's) is not the
    file's either.
    """
    if exc is None:
        return False
    from sqlalchemy.exc import IntegrityError as _SAIntegrityError
    import sqlite3 as _sqlite3
    return isinstance(exc, (ValueError, _SAIntegrityError, _sqlite3.IntegrityError))


def record_row_id(result: Optional[AnalysisResult], write: Any, row_id: Optional[int]) -> None:
    """The saved row's id onto the result, as the analysis did when it saved: a final test's (not
    a header-only row's: that result is its ERROR), a smoothness file's. For every writer."""
    if result is None:
        return
    if isinstance(write, FinalTestWrite):
        if result.overall_status != AnalysisStatus.ERROR:
            result.final_test_id = row_id
            logger.debug(f"Processed Final Test: {result.metadata.filename} - "
                         f"{result.overall_status.value} (ID: {row_id})")
    elif isinstance(write, SmoothnessWrite):
        result.smoothness_id = row_id
        logger.debug(f"Processed Smoothness: {result.metadata.filename} - "
                     f"{result.overall_status.value} (ID: {row_id})")


@dataclass
class Outcome:
    """What analysing ONE file found, and every database write it asks for -- as values.

    `Processor.analyse_path` makes one and touches no database (ingest-speed spec 4.1-4.2, ruling
    9). The writes the pool threads used to make themselves, in the middle of the analysis --
    the per-path skip markers, the final-test save, the smoothness save -- come back in `writes`,
    in the order the old code made them, and the CONSUMER applies them: through a writer when
    one is given (the ingest), or at once through the public methods (`Processor.apply_outcome`,
    which `process_file` and V5's loop use), exactly as before.

    `result` is what `process_file` returns for the file: an AnalysisResult, or None for a file
    that is not test data. A trim result is still its caller's to save, as it always was; `stat`
    and `file_hash` are the (size, mtime) and SHA-256 of the bytes that were PARSED, for that save
    (ruling 10): (None, None) when the file could not be statted. `started` is when the analysis
    began (`time.time()`), for the ERROR result a failed save becomes. `identity_error`: why a
    trim file that stats could not be hashed -- its save must fail, as `save_analysis`'s did.
    `internal`: the analysis itself raised (in the pool) -- nothing to save; the file is an error,
    and new again next run.
    """
    path: str
    result: Optional[AnalysisResult] = None
    writes: Tuple[Any, ...] = ()
    stat: Optional[Tuple[int, float]] = None
    file_hash: Optional[str] = None
    started: float = 0.0
    identity_error: Optional[str] = None
    internal: Optional[str] = None


class WriterStop(Exception):
    """Raised by a writer to END the batch: the folder cannot go on -- e.g. ruling 22's two
    consecutive failed batch commits. It propagates out of `process_batch` (the pool's per-file
    error handling never swallows it), and the folder fails with this message."""


# Files per chunk (spec 4.8): the unit the pool is fed in, the one place Stop and memory are asked
# about -- so a Stop lands on a whole chunk (test_ingest_cancel) -- and the batch writer's K.
CHUNK = 20

# The in-flight cap's hysteresis (ingest-speed A5, spec section 5, ruling 21): above 90% memory one
# fewer file in flight, above 95% one at a time, and below 80% on CAP_CALM_CHECKS checks running one
# more, up to the pool's size. The 10-point gap and the two-check wait keep one reading from making
# it oscillate.
CAP_UP_BELOW_PERCENT = 80
CAP_CALM_CHECKS = 2


def memory_percent() -> Optional[float]:
    """How full memory is, in percent -- or None when it cannot be read (no psutil). The ONE
    probe the ingest's throttle asks (tests script it)."""
    if not HAS_PSUTIL:
        return None
    try:
        return float(psutil.virtual_memory().percent)
    except Exception:
        return None


def next_cap(k: int, percent: float, calm_checks: int, size: int) -> Tuple[int, int]:
    """(the next in-flight cap, the calm checks so far), after one memory reading at a chunk
    boundary (ruling 21). `k` is the cap now, `size` the pool's -- the cap never goes above it,
    nor below one. Pure."""
    if percent > MEMORY_CRITICAL_PERCENT:
        return 1, 0
    if percent > MEMORY_WARNING_PERCENT:
        return max(1, k - 1), 0
    if percent < CAP_UP_BELOW_PERCENT:
        calm = calm_checks + 1
        if calm >= CAP_CALM_CHECKS and k < size:
            return k + 1, 0
        return k, min(calm, CAP_CALM_CHECKS)
    return k, 0


class _PoolBroke(Exception):
    """A worker process died (BrokenExecutor): the dispatch hands the rest to threads."""

    def __init__(self, cause: BaseException):
        super().__init__(str(cause))
        self.cause = cause


def _broken(future) -> bool:
    """Did this future fail because its POOL broke (a worker died), not because of its file?"""
    return (future.done() and not future.cancelled()
            and isinstance(future.exception(), BrokenExecutor))


class _ThreadPool:
    """The in-process pool: threads running the processor's own `analyse_path` -- the fallback
    when worker processes cannot run, and every small run (spec 4.5-4.6).

    `lookahead = 1`: the next chunk goes out only once the last one is done -- the batch
    boundary this path has always had; `in_process`: the parent parses, so the dispatch collects
    garbage between chunks, as it did between batches. One pool for the whole folder: memory
    throttles how many files are in flight (the cap), not the pool's size."""

    lookahead = 1
    in_process = True

    def __init__(self, processor, size: int, why: Optional[str] = None):
        self.size = max(1, int(size))
        self.closed = False
        self._analyse = processor.analyse_path
        self._executor = ThreadPoolExecutor(max_workers=self.size)
        self.mode = (f"{self.size} thread{'s' if self.size != 1 else ''}"
                     + (f" ({why})" if why else ""))

    def submit(self, path, disk_stat=None):
        return self._executor.submit(self._analyse, Path(path), disk_stat)

    def close(self, grace: Optional[float] = None) -> int:
        """Threads cannot be terminated: a running file finishes, the queued ones are dropped."""
        if not self.closed:
            self.closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
        return 0


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
        snapshot: Optional[SpecSnapshot] = None,
        ml_storage_path: Optional[Path] = None,
    ):
        """
        Initialize processor.

        Args:
            config: Configuration object
            use_ml: Whether to attempt loading ML thresholds from database
            snapshot: What the analysis would read from the database, read once at folder start
                (`take_spec_snapshot`). Given one, every spec question and the ML thresholds and
                predictors come from it and the analysis never asks the database; without one,
                they come from `get_database()` as they always have (the V5 loop, scripts).
                Only a Processor with a snapshot can be run in worker processes (ingest_worker).
            ml_storage_path: Where the composite trim-risk models load from. Default: the app
                directory's data/ml_models (config.ml_models_directory). A worker process is
                handed its PARENT's resolved folder here, never resolving its own (ingest-speed
                Task 11: a spawned child's app directory is the real install, whatever the
                parent -- a test, say -- was pointed at).
        """
        self.config = config or get_config()
        self._snapshot = snapshot
        self._use_ml = use_ml
        # Which pool the last batch was analysed in, and why ("8 processes (ready in 1.3 s)",
        # "4 threads (37 files; worker processes start at 200)"): the batch line prints it.
        self.last_workers = ""
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
        self._error_basenames: set = set()
        self._scan_adopted = 0
        self._scan_rebound = 0
        # (size, mtime) per path captured DURING folder discovery (scandir
        # returns them with the directory listing — same network round trip).
        # With this filled, the incremental check needs ZERO per-file I/O:
        # V4-era speed (seconds for 170k files) with content-change safety.
        # Friday 2026-07-10: the per-file stat() fallback cost 73 minutes.
        self._disk_stats: Dict[str, tuple] = {}
        # (file_hash, size, mtime datetime) tuples queued when a hash-confirm
        # succeeded but the recorded stat was missing/stale — flushed once per
        # batch so the NEXT scan takes the fast stat path.
        self._stat_heal: List[tuple] = []
        # Per-batch phase timings + counts (see process_batch). Initialized
        # here so _is_processed can count outside a batch too.
        self.last_scan_stats: Dict[str, float] = {
            "total_files": 0, "load_seconds": 0.0, "check_seconds": 0.0,
            "verify_seconds": 0.0, "needs_hash": 0, "memory_hits": 0,
            "new_files": 0, "verified_processed": 0, "heal_queued": 0,
            "heal_updated": 0, "process_seconds": 0.0, "total_seconds": 0.0,
        }

        # Load per-model thresholds and predictors from database
        self._model_thresholds: Dict[str, float] = {}
        self._model_predictors: Dict = {}  # model_name -> ModelPredictor
        # Composite trim-risk models (lazy-loaded per model, keyed by model name).
        # False sentinel means "checked and absent/non-deployed", to avoid repeated
        # filesystem lookups for models that don't have a deployed model yet.
        self._composite_models: Dict = {}
        # Storage path for composite risk pickle files (mirrors MLManager convention): the app
        # directory's data/ml_models, never the working directory's (config.ml_models_directory).
        if ml_storage_path is None:
            from laser_trim_analyzer.config import ml_models_directory
            ml_storage_path = ml_models_directory()
        self.ml_storage_path = Path(ml_storage_path)
        if use_ml:
            if snapshot is not None:
                self._model_thresholds = dict(snapshot.ml_thresholds)
                self._model_predictors = dict(snapshot.ml_predictors)
            else:
                self._load_ml_thresholds()

        # Create analyzer with per-model thresholds
        self.analyzer = Analyzer(
            model_thresholds=self._model_thresholds
        )

    def _load_ml_thresholds(self) -> None:
        """Load trained per-model thresholds from database (no snapshot given)."""
        try:
            from laser_trim_analyzer.database import get_database
            db = get_database()
        except Exception as e:
            logger.debug(f"Could not load ML thresholds: {e}")
            self._model_thresholds = {}
            self._model_predictors = {}
            return
        self._model_thresholds, self._model_predictors = load_ml_state(db)

    def process_file(self, file_path: Path, generate_plots: bool = True) -> Optional[AnalysisResult]:
        """
        Process a single file (trim or final test).

        Automatically detects file type and routes to appropriate handler: the analysis
        (`analyse_path`), then the writes it asks for made at once (`apply_outcome`) -- what
        this method always did, for its callers outside the ingest (V5, track_repair, scripts).

        Args:
            file_path: Path to Excel file
            generate_plots: Whether to generate plot images

        Returns:
            AnalysisResult with all track data (for trim files)
            or special final_test result marker
        """
        file_path = Path(file_path)
        return self.apply_outcome(self.analyse_path(file_path,
                                                    self._disk_stats.get(str(file_path))))

    @shares_one_stat
    def analyse_path(self, file_path: Path, disk_stat: Optional[tuple] = None) -> Outcome:
        """Analyse ONE file and return what it found, touching no database (spec 4.1, ruling 9).

        Today's `process_file` body minus every write: a skip marker, a final-test or smoothness
        save come back as values in the Outcome, for the consumer to apply (see Outcome). The spec
        questions are answered by the Processor's snapshot when it has one (ruling 16); without
        one they still ask `get_database()`, as `process_file` always did.

        `disk_stat` is the walk's (size, mtime) for this path, when the walk took one: a final
        test or smoothness file records it, as it always has. One stat per path for the whole
        analysis (`@shares_one_stat`): what the parse read is what the Outcome says it read.
        """
        start_time = time.time()
        file_path = Path(file_path)

        logger.debug(f"Processing: {file_path.name}")

        # Detect file type
        file_type = detect_file_type(file_path)

        if file_type == "non_trim":
            logger.debug(f"Skipping non-trim file: {file_path.name}")
            return self._skipped(file_path, start_time)

        if file_type == "final_test":
            return self._final_test_outcome(file_path, start_time, disk_stat)

        if file_type == "smoothness":
            return self._smoothness_outcome(file_path, start_time, disk_stat)

        return self._trim_outcome(file_path, start_time)

    def _trim_outcome(self, file_path: Path, start_time: float) -> Outcome:
        """A trim file's analysis, as an Outcome (the trim half of the old process_file)."""
        # Process as trim file (existing logic)
        try:
            # Parse file
            try:
                parsed = self.parser.parse_file(file_path)
            except NonTrimWorkbookError as e:
                # Parameter/report workbook named like test data — a known
                # non-data layout. Skip like non_trim; never an ERROR row.
                logger.debug(f"Skipping parameter/report workbook {file_path.name}: {e}")
                return self._skipped(file_path, start_time)
            metadata = parsed["metadata"]
            tracks_data = parsed["tracks"]
            file_hash = parsed["file_hash"]

            if not tracks_data:
                return self._trim_done(file_path, start_time, self._create_error_result(
                    metadata, "No valid track data found", start_time
                ), file_hash)

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
                        trim_passes=track_data.get("trim_passes") or [],
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
                        logger.debug(
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

                # Last gate before this verdict becomes a stored row.
                enforce_measurement_backed_verdict(track_result, file_path.name)

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
                trim_setup=parsed.get("trim_setup"),
                error_reason=error_reason_of(analyzed_tracks, overall_status),
            )

            logger.debug(f"Completed: {file_path.name} - {overall_status.value} "
                       f"({processing_time:.2f}s)")

            return self._trim_done(file_path, start_time, result, file_hash)

        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            return self._trim_done(file_path, start_time, self._create_error_result(
                self._create_minimal_metadata(file_path),
                f"File not found: {e}",
                start_time
            ))
        except Exception as e:
            writes: Tuple[Any, ...] = ()
            if self._is_permanent_failure(e):
                logger.warning(f"{file_path.name} permanently unprocessable — "
                               f"recorded as skipped: {e}")
                marker = self._skip_marker(file_path)
                writes = (marker,) if marker is not None else ()
            else:
                logger.exception(f"Error processing {file_path.name}: {e}")
            return self._trim_done(file_path, start_time, self._create_error_result(
                self._create_minimal_metadata(file_path),
                str(e),
                start_time
            ), writes=writes)

    def _trim_done(self, file_path: Path, start_time: float, result: AnalysisResult,
                   file_hash: Optional[str] = None, writes: Tuple[Any, ...] = ()) -> Outcome:
        """A trim file's Outcome, carrying the (size, mtime) and SHA-256 of the bytes that were
        parsed -- what `save_analysis`'s `_file_identity` takes at save time, taken HERE: inside
        the analysis's one-stat scope the stat is the parse's own, and the hash is the parse's
        (or, for a file the parse never hashed, the file's). (None, None) for a file that cannot
        be statted, as `_file_identity` gives; a file that stats but cannot be read records why,
        since its save must fail as `save_analysis`'s did."""
        stat, identity_error = None, None
        try:
            st = stat_once(file_path)
        except OSError:
            st = None
        if st is not None:
            stat = (st.st_size, st.st_mtime)
            if file_hash is None:
                try:
                    file_hash = calculate_file_hash(file_path, known_stat=st)
                except Exception as e:
                    identity_error = f"{type(e).__name__}: {e}"
        return Outcome(path=str(file_path), result=result, writes=tuple(writes), stat=stat,
                       file_hash=file_hash, started=start_time, identity_error=identity_error)

    def _skipped(self, file_path: Path, start_time: float) -> Outcome:
        """Not test data (non_trim, a parameter workbook): no result, and a skip marker."""
        marker = self._skip_marker(file_path)
        return Outcome(path=str(file_path), result=None,
                       writes=(marker,) if marker is not None else (), started=start_time)

    def _disk_stat_for_save(self, file_path: Path, disk_stat: Optional[tuple] = None) -> tuple:
        """(size, mtime datetime) for a file, or (None, None) if unavailable.

        Prefers the stat captured during folder discovery (scandir hands it
        over with the listing — no extra network round trip): handed in as
        `disk_stat` (a worker process has no `_disk_stats`), else looked up.
        Recorded on the FT/smoothness row so the NEXT scan recognises the file
        from memory instead of reading every byte of it to hash -- which is why
        the mtime is converted exactly so, `datetime.fromtimestamp`, local: the
        scan compares it with the walk's (review of Tasks 5-6, the Task 9
        hand-off).
        """
        st = disk_stat if disk_stat is not None else self._disk_stats.get(str(file_path))
        if st is None:
            try:
                _st = file_path.stat()
                st = (_st.st_size, _st.st_mtime)
            except OSError:
                return (None, None)
        try:
            return (st[0], datetime.fromtimestamp(st[1]))
        except (OSError, OverflowError, ValueError):
            return (None, None)

    def _final_test_outcome(self, file_path: Path, start_time: float,
                            disk_stat: Optional[tuple] = None) -> Outcome:
        """
        Analyse a Final Test file: its Outcome carries the `save_final_test` it asks for.

        Parses and grades the file, and returns a special marker result.
        Final Test files don't go through the same analysis pipeline as trim files.
        The save is the consumer's (`apply_outcome`, or the ingest's writer); a save that
        raises is handled there by the same rule as a failure here (`_final_test_failure`).

        Args:
            file_path: Path to Final Test Excel file
            start_time: Processing start time
            disk_stat: The walk's (size, mtime) for this path, if it took one

        Returns:
            Outcome whose result has file_type='final_test'
        """
        try:
            # Parse the Final Test file
            parsed = self.final_test_parser.parse_file(file_path)

            metadata = parsed["metadata"]
            tracks = parsed["tracks"]
            test_results = parsed["test_results"]
            file_hash = parsed["file_hash"]

            # Add file path to metadata
            metadata["file_path"] = str(file_path)
            ft_size, ft_mtime = self._disk_stat_for_save(file_path, disk_stat)

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
                header = FinalTestWrite(
                    metadata=metadata,
                    tracks=tracks,
                    test_results=test_results,
                    file_hash=file_hash,
                    file_size=ft_size,
                    file_modified_date=ft_mtime,
                )
                error_result = self._create_error_result(
                    minimal_metadata,
                    "Final Test file has no track data",
                    start_time
                )
                error_result.file_type = "final_test"  # Prevent saving as trim record
                return self._side_outcome(file_path, start_time, error_result, header, disk_stat)

            # Look up full spec for FT analysis. Use the FT-specific resolver
            # so multi-section parts (e.g. 8508) pick up the per-section spec
            # based on the trailing letter on the serial (e.g. '31B' -> 8508-B).
            ft_model = metadata.get("model", "unknown")
            ft_serial = metadata.get("serial")
            ft_spec = self._get_spec_for_analysis(ft_model, ft_serial, is_final_test=True)
            ft_compensation = metadata.get("station_compensation")

            # Run analyzer BEFORE saving so slope/offset/linearity_type flow into
            # the final_test_tracks rows. The analyzer output is used both for
            # the display result (analyzed_tracks) and for enriching the raw
            # track dicts passed to save_final_test.
            #
            # The grade itself lives in core.ft_regrade.grade_ft_track, shared
            # with the re-grade repair pass so the two can never disagree about
            # what a final test's verdict is. It grades only the rows the
            # station graded (the parser's graded_window), merged with the
            # model spec's exclude_points_ft.
            analyzed_tracks = [
                grade_ft_track(self.analyzer, track, ft_spec, model=ft_model,
                               ft_compensation=ft_compensation,
                               filename=file_path.name)
                for track in tracks
            ]

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

            # The save (now with enriched tracks) is the consumer's: apply_outcome, or the
            # ingest's writer. Its row id lands on the result there (final_test_id).
            write = FinalTestWrite(
                metadata=metadata,
                tracks=tracks,
                test_results=test_results,
                file_hash=file_hash,
                file_size=ft_size,
                file_modified_date=ft_mtime,
            )

            result = AnalysisResult(
                metadata=minimal_metadata,
                overall_status=overall_status,
                processing_time=processing_time,
                tracks=analyzed_tracks,
            )

            # Mark this as a final test file for special handling
            result.file_type = "final_test"

            return self._side_outcome(file_path, start_time, result, write, disk_stat)

        except Exception as e:
            error_result, marker = self._final_test_failure(file_path, e, start_time)
            return Outcome(path=str(file_path), result=error_result,
                           writes=(marker,) if marker is not None else (), started=start_time)

    def _final_test_failure(self, file_path: Path, exc: Exception, start_time: float,
                            saving: bool = False
                            ) -> Tuple[AnalysisResult, Optional[SkipMarkerWrite]]:
        """The final-test failure rule, ONE place: for an exception while the file was analysed
        (`_final_test_outcome`) and for one while its row was saved (`apply_outcome`, the
        ingest's writer) -- what the old single try/except did for both. It logs the traceback of
        the exception it is given. Returns the ERROR result and the skip marker to write, if any.

        `saving`: the exception came from the SAVE. Then only a content refusal may mark the file
        (`is_content_refusal`): a database or system error leaves it new, never unreadable.
        """
        marker = None
        if saving and not is_content_refusal(exc):
            logger.error(f"Final Test {file_path.name}: the save failed with a database or system "
                         f"error, not the file's ({type(exc).__name__}: {exc}) -- it is NOT "
                         f"recorded as unreadable, and is new again next run", exc_info=exc)
        elif self._is_permanent_failure(exc):
            # Permanently unprocessable (or already saved): record as
            # skipped so the next scan doesn't re-attempt it forever.
            logger.warning(f"Final Test {file_path.name} permanently "
                           f"unprocessable — recorded as skipped: {exc}")
            marker = self._skip_marker(file_path)
        else:
            # exc_info=exc: the traceback of THIS exception, whether or not it is the one being
            # handled (the ingest's writer applies this rule to a save that failed in a batch).
            logger.error(f"Error processing Final Test {file_path.name}: {exc}", exc_info=exc)
            # Same as the smoothness branch below: an FT error result is
            # never saved, so without this the file comes back tomorrow.
            if not self._is_transient_failure(exc):
                marker = self._skip_marker(
                    file_path, reason=f"{type(exc).__name__}: {exc}"[:200])
        error_result = self._create_error_result(
            self._create_minimal_metadata(file_path),
            f"Final Test error: {exc}",
            start_time
        )
        error_result.file_type = "final_test"  # Prevent saving as trim record
        return error_result, marker

    def _side_outcome(self, file_path: Path, start_time: float, result: AnalysisResult,
                      write: Any, disk_stat: Optional[tuple]) -> Outcome:
        """A final test's or smoothness file's Outcome: its one save, with the stat it records."""
        st = disk_stat if disk_stat is not None else self._disk_stats.get(str(file_path))
        if st is None and write.file_size is not None and write.file_modified_date is not None:
            st = (write.file_size, write.file_modified_date.timestamp())
        return Outcome(path=str(file_path), result=result, writes=(write,),
                       stat=tuple(st) if st is not None else None,
                       file_hash=write.file_hash, started=start_time)

    def process_batch(
        self,
        file_paths: List[Path],
        progress_callback: Optional[Callable[[ProcessingStatus], None]] = None,
        incremental: bool = True,
        disk_stats: Optional[Dict[str, tuple]] = None,
        cancel: Optional["threading.Event"] = None,
        writer: Optional[Any] = None,
    ) -> Generator[AnalysisResult, None, BatchSummary]:
        """
        Process multiple files with progress reporting.

        Args:
            file_paths: List of file paths
            progress_callback: Called with status updates
            incremental: Skip already processed files
            cancel: Cooperative stop. Checked at each batch boundary — the
                files already handed to the pool always finish and are yielded
                for persistence, and only then does the loop end. Nothing is
                killed, so no batch is ever half-written. The summary comes
                back exactly as it would from a finished run.
            writer: Where each file's writes go (spec 4.1, ruling 9). The pool
                runs `analyse_path`, which writes nothing; every Outcome comes
                back to THIS thread and is handed to `writer.add(outcome)`,
                which returns the result to yield. With no writer, each
                Outcome's writes are made at once (`apply_outcome`) -- what the
                pool threads used to do themselves, so V5's loop keeps working.

        Yields:
            AnalysisResult for each file

        Returns:
            BatchSummary after processing completes
        """
        total_files = len(file_paths)
        self._disk_stats = disk_stats or {}
        summary = BatchSummary(total_files=total_files, start_time=datetime.now())
        # Phase timings + counts for this batch. Read by the Process page for
        # its one-line phase breakdown and by the QA sweep (needs_hash must be
        # 0 on a second pass over the same folder) — never parsed from logs.
        t_batch = time.monotonic()
        self.last_scan_stats: Dict[str, float] = {
            "total_files": total_files, "load_seconds": 0.0,
            "check_seconds": 0.0, "verify_seconds": 0.0, "needs_hash": 0,
            "memory_hits": 0, "new_files": 0, "verified_processed": 0,
            "heal_queued": 0, "heal_updated": 0, "process_seconds": 0.0,
            "total_seconds": 0.0,
        }

        logger.info(f"Starting batch processing: {total_files} files, "
                   f"incremental={incremental}")

        # Load processed hashes if incremental
        if incremental:
            t_load = time.monotonic()
            self._load_processed_hashes()
            self.last_scan_stats["load_seconds"] = time.monotonic() - t_load

        # Choose strategy based on file count
        turbo_threshold = self.config.processing.turbo_mode_threshold
        use_parallel = total_files >= turbo_threshold

        if use_parallel:
            logger.info(f"Using parallel processing ({total_files} >= {turbo_threshold})")
            yield from self._process_parallel(
                file_paths, progress_callback, incremental, summary, cancel, writer
            )
        else:
            logger.info(f"Using sequential processing ({total_files} < {turbo_threshold})")
            self.last_workers = f"sequential ({total_files:,} files, under {turbo_threshold:,})"
            yield from self._process_sequential(
                file_paths, progress_callback, incremental, summary, cancel, writer
            )

        # The generator's end (a Stop lands here too): the writer commits what it
        # still holds (spec 3.1, ruling 5) before anything reads the database.
        if writer is not None and hasattr(writer, "flush"):
            writer.flush()

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
            queued = len(self._stat_heal)
            self.last_scan_stats["heal_queued"] = queued
            try:
                from laser_trim_analyzer.database import get_database
                n = get_database().update_processed_file_stats(self._stat_heal)
                self.last_scan_stats["heal_updated"] = n.get("total", 0)
                # Queued vs ACTUALLY UPDATED, per table. The old line reported
                # the queue length only, so "Repaired stat records for 150,938
                # processed files" was printed on every scan while the UPDATE
                # matched nothing (FT files have no processed_files row) —
                # six weeks of a silent no-op re-hashing the whole share.
                logger.info(
                    "Repaired stat records: %d queued -> %d final_test, "
                    "%d processed_files, %d smoothness%s",
                    queued, n.get("final_test_results", 0),
                    n.get("processed_files", 0), n.get("smoothness_results", 0),
                    "" if n.get("total", 0) >= queued else
                    f"  ⚠ {queued - n.get('total', 0)} matched NO row")
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

        s = self.last_scan_stats
        s["total_seconds"] = time.monotonic() - t_batch
        s["process_seconds"] = max(0.0, s["total_seconds"] - s["load_seconds"]
                                   - s["check_seconds"] - s["verify_seconds"])
        # One line, every phase. When a batch is slow again this says which
        # phase ate the time instead of leaving a log forensics pass guessing.
        logger.info(
            "Batch phases: load %.1fs | check %.1fs (%s in memory, %s new) | "
            "verify %s files %.1fs | process %s files %.1fs",
            s["load_seconds"], s["check_seconds"], f"{s['memory_hits']:,}",
            f"{s['new_files']:,}", f"{s['needs_hash']:,}", s["verify_seconds"],
            f"{summary.processed:,}", s["process_seconds"])
        logger.info(f"Batch complete: {summary.processed}/{total_files} processed, "
                   f"{summary.passed} passed, {summary.failed} failed")

        return summary

    def _process_sequential(
        self,
        file_paths: List[Path],
        progress_callback: Optional[Callable],
        incremental: bool,
        summary: BatchSummary,
        cancel: Optional["threading.Event"] = None,
        writer: Optional[Any] = None,
    ) -> Generator[AnalysisResult, None, None]:
        """Process files sequentially with memory management."""
        gc_interval = 50  # Run GC every 50 files

        for i, file_path in enumerate(file_paths):
            # Cooperative stop. One file IS the batch here, so the boundary is
            # the top of the loop: whatever was already parsed has been
            # yielded and saved.
            if cancel is not None and cancel.is_set():
                logger.info("Batch cancelled after %d of %d files (sequential)",
                            i, len(file_paths))
                return
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

            # Process: the analysis, then its writes -- the writer's when there is one, made
            # at once when there is not (spec 4.1).
            result = self._hand_over(
                self.analyse_path(file_path, self._disk_stats.get(str(file_path))), writer)

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
        cancel: Optional["threading.Event"] = None,
        writer: Optional[Any] = None,
    ) -> Generator[AnalysisResult, None, None]:
        """
        Process files in parallel with memory-aware throttling.

        The pool threads run `analyse_path` and write nothing; each Outcome is
        handed over HERE, on the consumer's thread (`_hand_over`).

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

            # Two-phase filter (2026-08-29). Phase 1 answers the whole folder
            # from memory; phase 2 does the file I/O for the leftovers IN
            # PARALLEL. Before this, one serial pass mixed both: on the FT
            # share every known file needed a hash (its table stored no stat),
            # so a scan read 170k files end to end over SMB — 70 minutes.
            # Phase 1 is single-threaded because _classify_scan mutates the
            # caches on a rescue hit; phase 2's workers touch nothing shared
            # (repairs come back as values and are applied here).
            if self._processed_filenames is None:   # called outside process_batch
                self._load_processed_hashes()
            stats = self.last_scan_stats
            t_check = time.monotonic()
            new_idx: set = set()
            needs_io: List[tuple] = []      # (index, path)
            for _i, _f in enumerate(file_paths):
                decision = self._classify_scan(Path(_f))
                if decision == "new":
                    new_idx.add(_i)
                elif decision == "needs_hash":
                    needs_io.append((_i, _f))
                # Heartbeat: statting 170k files on a share takes minutes —
                # silence reads as a lockup (work finding, 2026-07-10).
                if progress_callback and _i % 2000 == 1999:
                    progress_callback(ProcessingStatus(
                        filename="", status="scanning",
                        message=f"Checking against database… {_i + 1:,}/{len(file_paths):,}",
                        progress_percent=0))
            stats["check_seconds"] = time.monotonic() - t_check
            stats["new_files"] = len(new_idx)
            stats["needs_hash"] = len(needs_io)
            stats["memory_hits"] = len(file_paths) - len(new_idx) - len(needs_io)

            # Phase 2: stat/hash the files memory couldn't settle. Network
            # round trips overlap, so 8 workers ≈ 8x. This is the one-time
            # pass that stamps legacy rows; afterwards it's near-empty.
            if needs_io:
                t_verify = time.monotonic()
                if progress_callback:
                    progress_callback(ProcessingStatus(
                        filename="", status="scanning",
                        message=f"Verifying changed/legacy files… 0/{len(needs_io):,}",
                        progress_percent=0))
                resolved = 0
                with ThreadPoolExecutor(max_workers=8) as verifier:
                    futures = {
                        verifier.submit(self._resolve_scan_io, Path(_f)): _i
                        for _i, _f in needs_io
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            processed, repair = future.result()
                        except Exception as e:
                            logger.debug(f"Scan verify failed for {file_paths[idx]}: {e}")
                            processed, repair = False, None
                        if repair is not None:
                            self._apply_scan_repair(repair)   # this thread only
                        if processed:
                            stats["verified_processed"] += 1
                        else:
                            new_idx.add(idx)
                        resolved += 1
                        if progress_callback and resolved % 500 == 0:
                            progress_callback(ProcessingStatus(
                                filename="", status="scanning",
                                message=(f"Verifying changed/legacy files… "
                                         f"{resolved:,}/{len(needs_io):,}"),
                                progress_percent=0))
                stats["verify_seconds"] = time.monotonic() - t_verify

            # Keep discovery order (the batches below process in sequence).
            files_to_process = [f for i, f in enumerate(file_paths) if i in new_idx]
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

            n_retry = sum(1 for f in files_to_process
                          if Path(f).name in self._error_basenames)
            n_new = len(files_to_process) - n_retry
            scan_msg = (f"Found {len(files_to_process)} files to process: "
                        f"{n_new} new, {n_retry} retrying earlier errors "
                        f"({summary.skipped} already in database)")
            # LOGGED as well as shown — the old UI-only message left every
            # log forensics pass guessing what the scan concluded.
            logger.info(scan_msg)
            if progress_callback:
                progress_callback(ProcessingStatus(
                    filename="", status="scanning",
                    message=scan_msg, progress_percent=0,
                ))
                # Credit the already-known files to the progress bar in ONE
                # event. The run's denominator is every file found on disk
                # (the pre-scan cannot afford to work out which are new — see
                # ingest_run._prescan), so without this an incremental pass
                # over 150k known files sits at 0/150,000 from start to
                # finish. One event, not 150k: the flood is exactly what the
                # coalescer exists to prevent (2026-07-13).
                if summary.skipped:
                    progress_callback(ProcessingStatus(
                        filename="", status="known", count=summary.skipped,
                        message=f"{summary.skipped:,} already in database",
                        progress_percent=0,
                    ))
        else:
            files_to_process = list(file_paths)

        if not files_to_process:
            logger.info("No new files to process")
            self.last_workers = "none (nothing new to analyse)"
            return

        # If memory is already critical, fall back to sequential
        if self._check_memory_critical():
            logger.warning("Memory critical - falling back to sequential processing")
            self.last_workers = "sequential (memory was critical)"
            yield from self._process_sequential(
                [Path(f) for f in files_to_process],
                progress_callback, False, summary, cancel=cancel, writer=writer
            )
            return

        # Worker PROCESSES when they can run this analysis, else threads (ingest-speed A3, spec
        # 4.5-4.6); one pool for the whole folder either way.
        pool = self._open_pool(len(files_to_process), progress_callback)
        logger.info(f"Analysing {len(files_to_process):,} files on {pool.mode}")
        yield from self._dispatch(pool, files_to_process, progress_callback, summary, cancel,
                                  writer)

    def _open_pool(self, n_files: int, progress_callback: Optional[Callable] = None):
        """The pool this folder's files are analysed in: worker PROCESSES (spec 4.5, rulings
        13-14) when there are enough files left and this Processor carries its snapshot -- its
        analysis then asks the database nothing, and a worker may never open one -- else threads.
        A pool that cannot start hands the folder to threads, saying why (ruling 19). While the
        workers start the progress line says so: at work an endpoint scanner can make that take
        a while (spec 4.5), and silence reads as a lockup."""
        threads = self._get_safe_worker_count(n_files)
        why = self._why_not_processes(n_files)
        if why is None:
            from laser_trim_analyzer.core import ingest_worker
            n, why = ingest_worker.worker_count()
            if n >= 1:
                if progress_callback:
                    progress_callback(ProcessingStatus(
                        filename="", status="scanning", progress_percent=0,
                        message=f"Starting {n} worker processes for {n_files:,} files…"))
                try:
                    return ingest_worker.WorkerPool.start(ingest_worker.context_for(self), n)
                except ingest_worker.PoolFailed as e:
                    why = f"processes could not start: {e}"
                    logger.warning("Worker processes could not start (%s): this folder is "
                                   "analysed on %d threads instead", e, threads)
        return _ThreadPool(self, threads, why)

    def _why_not_processes(self, n_files: int) -> Optional[str]:
        """Why this folder cannot use worker processes -- or None when it can."""
        from laser_trim_analyzer.core import ingest_worker
        if n_files < ingest_worker.PROCESS_MIN_FILES:
            return (f"{n_files:,} files; worker processes start at "
                    f"{ingest_worker.PROCESS_MIN_FILES:,}")
        if getattr(self, "_snapshot", None) is None:
            return "no spec snapshot: this analysis reads the database, which a worker never opens"
        return None

    def _dispatch(self, pool, files, progress_callback, summary: BatchSummary, cancel, writer):
        """Every file to `pool` in CHUNK-file chunks, and every Outcome handed over on THIS
        thread, in the order the files finish (spec 4.8).

        A chunk is taken only when the pool can take it -- up to `pool.lookahead` chunks in flight
        (processes: two, so the workers never wait while the consumer saves; threads: one, the
        batch boundary this path always had) -- and that is the one place Stop is asked: every
        file handed out finishes and is handed over, so a stopped run lands on a whole chunk.
        Memory is asked there too (`_between_chunks`): the in-flight cap. A pool that breaks
        mid-run hands its in-flight files, and the rest, to threads (ruling 19); one closed from
        outside (the window closing) ends the dispatch there -- what was not handed over is new
        next run. The consumer waits at most 1 s at a time, ticking the writer (spec 3.1)."""
        total = len(files)
        queue: deque = deque()
        inflight: Dict[Any, Path] = {}          # submission order (as_completed's input)
        taken = completed = 0
        stopped = False
        cap, calm = pool.size, 0
        tick = getattr(writer, "tick", None)
        self.last_workers = pool.mode
        try:
            while True:
                try:
                    while True:
                        # Out: the taken chunks' files as the window allows; a new chunk when the
                        # pool can take one.
                        while True:
                            window = pool.lookahead * CHUNK if cap >= pool.size else cap
                            if queue and len(inflight) < window:
                                path = queue.popleft()
                                try:
                                    future = pool.submit(path, self._disk_stats.get(str(path)))
                                except BrokenExecutor as e:
                                    queue.appendleft(path)
                                    raise _PoolBroke(e) from e
                                except RuntimeError:
                                    if pool.closed:     # closed from outside (the window)
                                        return
                                    raise
                                inflight[future] = path
                            elif (not queue and not stopped and taken < total
                                  and len(inflight) <= (pool.lookahead - 1) * CHUNK):
                                # Cooperative stop, asked HERE and nowhere deeper: never a file
                                # half-saved, and a whole chunk.
                                if cancel is not None and cancel.is_set():
                                    stopped = True
                                    logger.info("Batch cancelled after %d of %d files -- the "
                                                "files already handed out finish and are saved",
                                                taken, total)
                                else:
                                    if taken:
                                        cap, calm = self._between_chunks(pool, cap, calm)
                                    chunk = [Path(f) for f in files[taken:taken + CHUNK]]
                                    queue.extend(chunk)
                                    taken += len(chunk)
                            else:
                                break
                        if not inflight:
                            return
                        # In: one finished file, handed over -- then out again, to keep the
                        # window full.
                        try:
                            for future in as_completed(list(inflight), timeout=1.0):
                                if pool.closed:
                                    return
                                path = inflight.pop(future)
                                if _broken(future):
                                    queue.appendleft(path)
                                    raise _PoolBroke(future.exception())
                                completed += 1
                                yield from self._one_completed(
                                    future, path, writer, summary, progress_callback,
                                    completed, total)
                                break
                        except FuturesTimeout:
                            if pool.closed:
                                return
                            if tick is not None:
                                tick()
                except _PoolBroke as broke:
                    if pool.closed:
                        return
                    queue.extendleft(reversed(list(inflight.values())))
                    inflight.clear()
                    pool = self._pool_broke(pool, broke.cause, completed, total)
                    cap, calm = pool.size, 0
        finally:
            pool.close()

    def _between_chunks(self, pool, cap: int, calm: int) -> Tuple[int, int]:
        """Before every chunk after the first: garbage collection when the parent parses (as
        between batches before A3), and the in-flight cap for the memory there is now (A5,
        `next_cap`) -- down under pressure, and BACK once memory has been calm for two checks
        running, which the old throttle never did. Each change is logged once."""
        if pool.in_process:
            gc.collect()
        percent = memory_percent()
        if percent is None:
            return cap, calm
        new, calm = next_cap(cap, percent, calm, pool.size)
        if new > cap:
            logger.info(f"workers {cap} → {new}: memory back to {percent:.0f}%")
        elif new < cap:
            logger.warning(f"workers {cap} → {new}: memory at {percent:.0f}%"
                           + (" (critical)" if percent > MEMORY_CRITICAL_PERCENT else ""))
        return new, calm

    def _pool_broke(self, pool, cause: BaseException, completed: int, total: int):
        """A worker process died mid-run: its files in flight, and the rest of the folder, go to
        threads (spec 4.6) -- a worker holds no database handle, so nothing is half-written."""
        pool.close(grace=0.0)
        threads = self._get_safe_worker_count(max(1, total - completed))
        why = (f"worker processes broke after {completed:,} files: "
               f"{type(cause).__name__}: {cause}")
        logger.warning("The ingest's %s broke after %d of %d files (%s: %s): the files in "
                       "flight and the rest of this folder are analysed on %d threads",
                       pool.mode, completed, total, type(cause).__name__, cause, threads)
        fallback = _ThreadPool(self, threads, why)
        self.last_workers = f"{pool.mode}, then {fallback.mode}"
        return fallback

    def _one_completed(self, future, file_path, writer, summary: BatchSummary,
                       progress_callback, completed: int, total: int):
        """One finished future of the pool, on the consumer's thread: its Outcome handed over,
        the summary and progress updated, the result yielded (a generator: nothing is yielded for
        a file that is not test data, or one that failed). A writer's WriterStop propagates; any
        other failure is this file's error, as it always was.

        An `internal` Outcome -- the analysis raised, or (in a worker process) reached for a
        database, spec 4.3 -- has nothing to save or yield: it is logged at ERROR, HERE, by the
        parent, handed to the writer (which counts it: an error, new again next run) and counted
        as a processed error."""
        try:
            try:
                outcome = future.result()
            except Exception as e:
                # The analysis itself raised: nothing to save.
                outcome = Outcome(path=str(file_path), internal=f"{type(e).__name__}: {e}")
            if outcome.internal is not None:
                logger.error(f"Error processing {file_path}: {outcome.internal}")
                if writer is not None:
                    writer.add(outcome)
                summary.processed += 1
                summary.errors += 1
                if progress_callback:
                    progress_callback(ProcessingStatus(
                        filename=Path(file_path).name,
                        status="failed",
                        message=outcome.internal,
                        progress_percent=completed / total * 100,
                    ))
                return
            result = self._hand_over(outcome, writer)

            # Skip non-trim files (process_file returns None)
            if result is None:
                summary.skipped += 1
                if progress_callback:
                    progress_callback(ProcessingStatus(
                        filename=Path(file_path).name,
                        status="skipped",
                        message="Non-trim file skipped",
                        progress_percent=completed / total * 100,
                    ))
                return

            self._update_summary(summary, result)

            if progress_callback:
                progress_callback(ProcessingStatus(
                    filename=Path(file_path).name,
                    status="completed",
                    progress_percent=completed / total * 100,
                    result=result,
                ))

            yield result

        except WriterStop:
            raise
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
                    progress_percent=completed / total * 100,
                ))

    def _hand_over(self, outcome: Outcome, writer) -> Optional[AnalysisResult]:
        """One file's Outcome to whoever makes its writes -- on THIS thread, the consumer's: the
        writer when there is one (the ingest), else at once, here (V5's loop). Returns the result
        the loop yields."""
        if writer is None:
            return self.apply_outcome(outcome)
        return writer.add(outcome)

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
        """Is memory critical (above MEMORY_CRITICAL_PERCENT)? Through the one probe."""
        percent = memory_percent()
        return percent is not None and percent > MEMORY_CRITICAL_PERCENT

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

    def _smoothness_outcome(self, file_path: Path, start_time: float,
                            disk_stat: Optional[tuple] = None) -> Outcome:
        """Analyse an Output Smoothness file: its Outcome carries the `save_smoothness_result` it
        asks for (the consumer's; a save that raises is handled there by `_smoothness_failure`,
        the same rule as a failure here)."""
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

            os_size, os_mtime = self._disk_stat_for_save(file_path, disk_stat)
            write = SmoothnessWrite(
                metadata=metadata, tracks=tracks, file_hash=file_hash,
                file_size=os_size, file_modified_date=os_mtime,
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
            result.file_type = "smoothness"   # its row id (smoothness_id) lands where it is saved

            return self._side_outcome(file_path, start_time, result, write, disk_stat)

        except Exception as e:
            error_result, marker = self._smoothness_failure(file_path, e, start_time)
            return Outcome(path=str(file_path), result=error_result,
                           writes=(marker,) if marker is not None else (), started=start_time)

    def _smoothness_failure(self, file_path: Path, exc: Exception, start_time: float,
                            saving: bool = False
                            ) -> Tuple[AnalysisResult, Optional[SkipMarkerWrite]]:
        """The smoothness failure rule, ONE place (see `_final_test_failure`, `saving` included)."""
        if saving and not is_content_refusal(exc):
            logger.error(f"Smoothness {file_path.name}: the save failed with a database or system "
                         f"error, not the file's ({type(exc).__name__}: {exc}) -- it is NOT "
                         f"recorded as unreadable, and is new again next run", exc_info=exc)
        else:
            logger.error(f"Error processing Smoothness {file_path.name}: {exc}", exc_info=exc)
        error_result = self._create_error_result(
            self._create_minimal_metadata(file_path),
            f"Smoothness error: {exc}", start_time
        )
        error_result.file_type = "smoothness"  # Prevent saving as trim record
        # …and record WHY, or nothing is written anywhere at all: the error
        # result is not saved (file_type keeps it out of save_analysis), so
        # the scan offered these files again every single run. 67 of them
        # on 2026-09-17 — opened, refused, forgotten, repeat.
        marker = None
        if not self._is_transient_failure(exc) and not (saving and not is_content_refusal(exc)):
            marker = self._skip_marker(
                file_path, reason=f"{type(exc).__name__}: {exc}"[:200])
        return error_result, marker

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
            # The folder's snapshot when there is one -- the same resolver, never the database
            # (ruling 16); else the database, as before.
            source = getattr(self, "_snapshot", None)
            if source is None:
                from laser_trim_analyzer.database import get_database
                source = get_database()
            if is_final_test:
                spec = source.resolve_spec_for_ft(model, serial)
            else:
                spec = source.get_model_spec(model)
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

    def _classify_scan(self, file_path: Path) -> str:
        """Decide from MEMORY ALONE whether a file is already processed.

        Returns "processed", "new", or "needs_hash" (the caller must do file
        I/O — a stat() and possibly a full-content hash — to decide). Performs
        no file I/O itself; the only stat data it consults is `_disk_stats`,
        captured during folder discovery. Splitting this out is what lets the
        parallel path answer the whole known-folder in memory and confine the
        I/O to the handful of files that genuinely need it.

        May mutate the in-memory caches on a rescue hit — call it from ONE
        thread only (the sequential path and the parallel filter's phase 1).

        Full-path miss is NOT "definitely new". The work incident
        (2026-07-09): the same share browsed under a different path form
        (mapped drive vs UNC vs new root) missed on every file and the app set
        out to reprocess the entire history. Rescue by BASENAME (filenames
        carry model_serial_datetime, so they identify the export): confirm
        cheaply by recorded (size, mtime); legacy rows with no recorded stat
        (pre-fast-path databases) are ADOPTED on a unique basename match —
        hashing tens of thousands of files over the share is the lockup we're
        preventing. Ambiguity falls back to the hash identity check.
        """
        path_str = str(file_path)
        if path_str not in self._processed_filenames:
            entries = self._processed_basename.get(file_path.name)
            if not entries:
                return "new"            # truly new filename
            st = self._disk_stats.get(path_str)
            if st is None:
                return "needs_hash"     # no discovery stat -> needs a stat()
            for _stored, size, mtime in entries:
                if (size is not None and mtime is not None
                        and st[0] == size
                        and abs(st[1] - mtime) <= 2.0):
                    # Same name, same size, same mtime: the file we already
                    # processed, reached via a new path form.
                    self._processed_filenames.add(path_str)
                    self._processed_stat[path_str] = st
                    self._scan_rebound += 1
                    return "processed"
            if len(entries) == 1 and entries[0][1] is None:
                # Single legacy record without stats (database predates the
                # stat fast-path). Trust the unique name; record the observed
                # stat so future scans are exact.
                self._processed_filenames.add(path_str)
                self._processed_stat[path_str] = st
                self._scan_adopted += 1
                return "processed"
            # Ambiguous (same name in several folders / stat mismatch): the
            # hash identity check decides.
            return "needs_hash"

        # Stat fast-path: if the recorded size AND mtime still match the file
        # on disk, the content can't have changed — skip without reading the
        # file at all. This is what makes re-scanning a mostly-processed
        # network folder fast; hashing every path-hit meant reading every byte
        # of every known file over the share. mtime tolerance 2s covers
        # FAT/SMB timestamp granularity.
        recorded = self._processed_stat.get(path_str)
        if recorded is not None:
            cur = self._disk_stats.get(path_str)
            if cur is not None and cur[0] == recorded[0] and abs(cur[1] - recorded[1]) <= 2.0:
                return "processed"
        # Stat missing or stale -> CONFIRM content by hash before skipping.
        return "needs_hash"

    def _resolve_scan_io(self, file_path: Path) -> tuple:
        """Resolve a "needs_hash" classification, doing the file I/O it needs.

        Returns (is_processed, repair) where `repair` is None or a
        (kind, path_str, stat_tuple, file_hash) tuple for
        `_apply_scan_repair` to apply on the calling thread. Touches no
        shared state itself, so it is safe to run in a worker thread.
        """
        path_str = str(file_path)
        known_path = path_str in self._processed_filenames
        st = self._disk_stats.get(path_str)
        if st is None:
            try:
                st_ = file_path.stat()
                st = (st_.st_size, st_.st_mtime)
            except OSError:
                st = None

        if not known_path:
            if st is None:
                return False, None      # can't stat -> let it process
            entries = self._processed_basename.get(file_path.name) or ()
            for _stored, size, mtime in entries:
                if (size is not None and mtime is not None
                        and st[0] == size and abs(st[1] - mtime) <= 2.0):
                    return True, ("rebound", path_str, st, None)
            if len(entries) == 1 and entries[0][1] is None:
                return True, ("adopted", path_str, st, None)
            try:
                file_hash = calculate_file_hash(file_path)
            except Exception:
                return False, None
            if file_hash in self._processed_hashes:
                return True, ("rebound_heal", path_str, st, file_hash)
            return False, None

        recorded = self._processed_stat.get(path_str)
        if (recorded is not None and st is not None
                and st[0] == recorded[0] and abs(st[1] - recorded[1]) <= 2.0):
            return True, None
        # The file_hash is the identity, not the path, so a re-export of new
        # content to a fixed filename is NOT dropped.
        try:
            file_hash = calculate_file_hash(file_path)
        except Exception:
            return False, None          # can't hash -> don't skip
        if file_hash in self._processed_hashes:
            # Same content, stale/missing stat record — queue a repair so the
            # NEXT scan takes the fast stat path for this file.
            return True, ("heal", path_str, st, file_hash)
        return False, None

    def _apply_scan_repair(self, repair: tuple) -> None:
        """Apply one `_resolve_scan_io` repair to the in-memory caches and the
        stat-heal queue. Single-threaded: the caller owns the ordering."""
        kind, path_str, st, file_hash = repair
        if kind in ("rebound", "adopted", "rebound_heal"):
            self._processed_filenames.add(path_str)
        if st is not None:
            self._processed_stat[path_str] = st
            if kind in ("rebound_heal", "heal") and file_hash:
                self._stat_heal.append((
                    file_hash, st[0], datetime.fromtimestamp(st[1]),
                ))
        if kind in ("rebound", "rebound_heal"):
            self._scan_rebound += 1
        elif kind == "adopted":
            self._scan_adopted += 1

    def _is_processed(self, file_path: Path) -> bool:
        """Check if file has already been processed.

        Identity is the CONTENT hash, not the path. A path miss is a cheap
        early-out (a brand-new filename can't have been processed); a path hit
        is confirmed by hash so a re-export of NEW content to a reused filename
        is processed rather than silently skipped.

        Memory decision first (`_classify_scan`), then — only for the files it
        can't settle — the stat/hash I/O inline. The parallel path splits those
        two phases so the I/O runs in a thread pool.

        Falls back to the (hash-based) DB query only if the cache wasn't loaded.
        """
        try:
            # If cache was loaded (even if empty), use it.
            if self._processed_filenames is not None:
                stats = self.last_scan_stats
                t0 = time.monotonic()
                decision = self._classify_scan(file_path)
                stats["check_seconds"] += time.monotonic() - t0
                if decision in ("processed", "new"):
                    stats["memory_hits" if decision == "processed"
                          else "new_files"] += 1
                    return decision == "processed"
                stats["needs_hash"] += 1
                t1 = time.monotonic()
                processed, repair = self._resolve_scan_io(file_path)
                if repair is not None:
                    self._apply_scan_repair(repair)
                stats["verify_seconds"] += time.monotonic() - t1
                if processed:
                    stats["verified_processed"] += 1
                return processed

            # Cache not loaded - query database directly (already hash-based).
            from laser_trim_analyzer.database import get_database
            db = get_database()
            return db.is_file_processed(file_path)

        except Exception as e:
            logger.warning(f"Could not check if file is processed: {e}")
            return False

    # Known-PERMANENT failure signatures (full-log taxonomy, 2026-07-10).
    # ~4,000 files on the work share fail for reasons that can never succeed
    # on retry — decade-old naming with no serial, pre-2003 Excel formats,
    # oscilloscope captures, duplicates. Re-attempting them on EVERY scan
    # spams thousands of errors and wastes share bandwidth. Files matching
    # these are recorded as skipped (with their stat), so they only ever
    # re-parse if their CONTENT changes — or on an explicit full reprocess
    # (incremental unchecked) after a parser upgrade.
    _PERMANENT_FAILURES = (
        "Serial cannot be empty",              # no serial in file/filename
        "Model cannot be empty",               # no model resolvable
        "Excel file format cannot be determined",  # pre-2003/corrupt workbook
        "directory corruption",                # corrupt OLE container (xlrd)
        "Not an Excel file",                   # e.g. a file literally named '.xls'
        "UNIQUE constraint failed",            # duplicate of a record already saved
    )

    @classmethod
    def _is_permanent_failure(cls, exc: Exception) -> bool:
        msg = str(exc)
        return any(sig in msg for sig in cls._PERMANENT_FAILURES)

    # Known-TRANSIENT failures — the mirror of the taxonomy above, and the one
    # class that must NEVER be recorded as unreadable (2026-09-17). A workbook
    # someone has open in Excel, a share that blinked, or an export a station
    # is still writing has told us nothing about its contents; remembering it
    # as unreadable would mean never reading that unit's data again.
    #
    # A half-written export needs no help from this list: when the station
    # finishes writing it, its size and mtime change, and `_classify_scan`'s
    # stat check offers it again by itself, with nobody pressing anything.
    #
    # The classifier takes an exception OR a reason STRING, because the trim
    # path's decision is made in the database manager, which only ever has the
    # text (see manager._record_processed_file).
    _TRANSIENT_FAILURES = (
        "Permission denied",            # locked by Excel, or a share ACL blip
        "No such file or directory",    # moved or deleted mid-run
        "used by another process",      # Windows lock (WinError 32)
        "timed out",                    # SMB stall
        "Network is unreachable",
        "network path",                 # share dropped (WinError 53/64)
        "Errno 13",
        "Errno 2",
    )

    @classmethod
    def _is_transient_failure(cls, exc) -> bool:
        if isinstance(exc, (FileNotFoundError, PermissionError, TimeoutError)):
            return True
        if isinstance(exc, OSError) and not cls._is_permanent_failure(exc):
            # Any other OS-level error is the share or the lock, not the
            # workbook — unless its message is a known-permanent parse failure.
            return True
        return any(sig in str(exc) for sig in cls._TRANSIENT_FAILURES)

    def _mark_file_skipped(self, file_path: Path,
                           reason: Optional[str] = None) -> None:
        """Record a file as processed so it's skipped on future runs.

        With NO reason this keeps its original meaning: the file is not test
        data (a parameter workbook, a report, a duplicate), and it is skipped
        for good. Prevents re-opening and re-checking junk files every time the
        same folder is processed — important for network drives with many
        files.

        With a reason it is a READ FAILURE: the name says test data but this
        build of the parser cannot turn it into a record. Those markers are
        tagged, so Settings → "Retry unreadable files" clears exactly them
        after a parser upgrade and leaves the thousands of non-trim markers
        alone. Callers must have ruled out `_is_transient_failure` first.

        Either way the marker is per-PATH and carries the file's size and
        mtime, so a file whose bytes CHANGE on disk is offered again by itself.

        The analysis no longer calls this: it asks for the marker as a value
        (`_skip_marker`) and the consumer writes it (`_write_marker`). This is
        the two together, for a caller that wants the marker written now.
        """
        marker = self._skip_marker(file_path, reason)
        if marker is not None:
            self._write_marker(marker)

    def _skip_marker(self, file_path: Path,
                     reason: Optional[str] = None) -> Optional[SkipMarkerWrite]:
        """The marker `_mark_file_skipped` writes, as a VALUE: the file's hash and (size, mtime)
        taken here, exactly as they were, and nothing written. None if they cannot be taken --
        never fatal, as the write never was (the file is merely offered again)."""
        try:
            file_hash = calculate_file_hash(file_path)
            stat = file_path.stat()
            return SkipMarkerWrite(
                filename=file_path.name,
                file_path=str(file_path),
                file_hash=file_hash,
                file_size=stat.st_size,
                file_modified_date=datetime.fromtimestamp(stat.st_mtime),
                error_message=reason,
                failed_read=reason is not None,
            )
        except Exception as e:
            logger.debug(f"Could not record skipped file {file_path.name}: {e}")
            return None

    def _write_marker(self, marker: SkipMarkerWrite, db=None) -> None:
        """Write one skip marker now, through `mark_file_skipped` (on `db`, else `get_database()`),
        and remember the path and hash in this run's caches. Never fatal."""
        try:
            if db is None:
                from laser_trim_analyzer.database import get_database
                db = get_database()
            db.mark_file_skipped(
                filename=marker.filename,
                file_path=marker.file_path,
                file_hash=marker.file_hash,
                file_size=marker.file_size,
                file_modified_date=marker.file_modified_date,
                error_message=marker.error_message,
                failed_read=marker.failed_read,
            )
            self.remember_marker(marker)
        except Exception as e:
            logger.debug(f"Could not record skipped file {marker.filename}: {e}")

    def remember_marker(self, marker: SkipMarkerWrite) -> None:
        """A marker has been WRITTEN: this run's in-memory caches know the path and its content
        now, as they did when the analysis wrote markers itself (a writer calls this once it has
        committed one)."""
        if self._processed_filenames is not None:
            self._processed_filenames.add(marker.file_path)
            if self._processed_hashes is not None:
                self._processed_hashes.add(marker.file_hash)

    def apply_outcome(self, outcome: Outcome, db=None) -> Optional[AnalysisResult]:
        """Make an Outcome's writes AT ONCE, through the public methods, and return what
        `process_file` returns -- the writes the analysis used to make itself (V5's loop,
        `process_file`, and any caller with no writer).

        Markers never fail the file (as `_mark_file_skipped` never did). A final-test or
        smoothness save that raises -- or whose database cannot be reached -- is handled by the
        rule the old try/except applied to it (`_final_test_failure`, `_smoothness_failure`): the
        marker that rule asks for is written, and its ERROR result is returned. A trim result is
        NOT saved here: its caller saves it, as it always has. `db`: where to write, else
        `get_database()`, as before.
        """
        result = outcome.result
        for write in outcome.writes:
            if isinstance(write, SkipMarkerWrite):
                self._write_marker(write, db)
                continue
            try:
                target = db
                if target is None:
                    from laser_trim_analyzer.database import get_database
                    target = get_database()
                if isinstance(write, FinalTestWrite):
                    row_id = target.save_final_test(
                        metadata=write.metadata, tracks=write.tracks,
                        test_results=write.test_results, file_hash=write.file_hash,
                        file_size=write.file_size, file_modified_date=write.file_modified_date)
                else:
                    row_id = target.save_smoothness_result(
                        metadata=write.metadata, tracks=write.tracks, file_hash=write.file_hash,
                        file_size=write.file_size, file_modified_date=write.file_modified_date)
            except Exception as e:
                result, marker = self.failed_save(write, outcome.path, e, outcome.started)
                if marker is not None:
                    self._write_marker(marker, db)
                return result
            record_row_id(result, write, row_id)
        return result

    def failed_save(self, write: Any, file_path, exc: Optional[BaseException],
                    started: float) -> Tuple[AnalysisResult, Optional[SkipMarkerWrite]]:
        """What a final-test or smoothness file whose SAVE failed becomes: its ERROR result, and
        the skip marker to write, if any -- the rule its failure always got
        (`_final_test_failure` / `_smoothness_failure`, `saving=True`: only a content refusal may
        mark the file). ONE public rule, for `apply_outcome` and for the ingest's batch writer
        alike (ingest-speed Task 11: the writer no longer reaches into the private helpers)."""
        rule = (self._final_test_failure if isinstance(write, FinalTestWrite)
                else self._smoothness_failure)
        return rule(Path(file_path), exc, started, saving=True)


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
        # Files whose LAST attempt errored (success=0) are retried by design.
        # Knowing which ones lets the scan message say "retrying earlier
        # errors" instead of looking like a runaway reprocess (2026-07-13:
        # James reset the whole work DB because the designed retry of
        # Friday's 3,908 casualties was indistinguishable from one).
        self._error_basenames = set()
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
                from pathlib import PurePath as _PP
                self._error_basenames = {
                    _PP(r[0]).name for r in session.query(DBProcessedFile.file_path)
                    .filter(DBProcessedFile.success == False).all() if r[0]}
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
                # Stat fast-path. FT/smoothness rows join this too (2026-08-29
                # — before that they had no size/mtime columns at all, so every
                # known FT file fell through to a full-content hash on EVERY
                # scan; rows still carrying NULL stats keep that behavior for
                # one pass and then heal themselves).
                self._processed_stat = {
                    r.file_path: (r.file_size, r.file_modified_date.timestamp())
                    for r in rows
                    if r.file_path and r.file_size is not None
                    and r.file_modified_date is not None
                }

                # Also load Final Test file paths + hashes (always "successful" if in DB)
                ft_rows = session.query(
                    FinalTestResult.file_path, FinalTestResult.file_hash,
                    FinalTestResult.file_size, FinalTestResult.file_modified_date
                ).all()
                ft_count = 0
                for row in ft_rows:
                    if row.file_path:
                        mt = (row.file_modified_date.timestamp()
                              if row.file_modified_date is not None else None)
                        size = row.file_size
                        self._processed_filenames.add(row.file_path)
                        self._processed_basename.setdefault(
                            PurePath(row.file_path).name, []
                        ).append((row.file_path, size, mt))
                        if size is not None and mt is not None:
                            self._processed_stat[row.file_path] = (size, mt)
                        ft_count += 1
                    if row.file_hash:
                        self._processed_hashes.add(row.file_hash)

                # Also load Smoothness file paths
                smoothness_count = 0
                try:
                    from laser_trim_analyzer.database.models import SmoothnessResult as DBSmoothnessResult
                    smoothness_rows = session.query(
                        DBSmoothnessResult.file_path, DBSmoothnessResult.file_hash,
                        DBSmoothnessResult.file_size,
                        DBSmoothnessResult.file_modified_date
                    ).all()
                    for row in smoothness_rows:
                        if row.file_path:
                            mt = (row.file_modified_date.timestamp()
                                  if row.file_modified_date is not None else None)
                            size = row.file_size
                            self._processed_filenames.add(row.file_path)
                            self._processed_basename.setdefault(
                                PurePath(row.file_path).name, []
                            ).append((row.file_path, size, mt))
                            if size is not None and mt is not None:
                                self._processed_stat[row.file_path] = (size, mt)
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
            self._error_basenames = set()

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
            error_reason=error_msg[:500],
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
