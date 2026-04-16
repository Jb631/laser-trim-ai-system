"""
Database Manager for Laser Trim Analyzer v3.

Simplified from v2's 2,900+ line manager to ~600 lines.
Focuses on essential operations with clean session management.

Operations:
- Save/retrieve analysis results
- Incremental processing tracking
- Historical data queries
- Model statistics
- QA alerts management
"""

import hashlib
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Iterator, Tuple
from contextlib import contextmanager

from sqlalchemy import create_engine, exists, func, and_, or_, desc, text, case
from sqlalchemy.orm import sessionmaker, Session, joinedload, subqueryload
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import IntegrityError, OperationalError

from laser_trim_analyzer.database.models import (
    Base,
    AnalysisResult as DBAnalysisResult,
    TrackResult as DBTrackResult,
    MLPrediction as DBMLPrediction,
    QAAlert as DBQAAlert,
    BatchInfo as DBBatchInfo,
    ProcessedFile as DBProcessedFile,
    ModelSpec,
    SystemType as DBSystemType,
    StatusType as DBStatusType,
    RiskCategory as DBRiskCategory,
    AlertType as DBAlertType,
)
from laser_trim_analyzer.core.models import (
    AnalysisResult,
    TrackData,
    AnalysisStatus,
    SystemType,
    RiskCategory,
)
from laser_trim_analyzer.config import get_config

logger = logging.getLogger(__name__)


def _model_sort_key(model: str) -> tuple:
    """
    Sort key for numerical model sorting.
    Extracts leading digits from model number for numerical comparison.
    E.g., '8340-1' -> (8340, '8340-1'), '8397-2' -> (8397, '8397-2')
    """
    base = model.split('-')[0] if model else ''
    num = int(''.join(c for c in base if c.isdigit()) or '0')
    return (num, model)


def _status_matches(status: Any, *targets: DBStatusType) -> bool:
    """Check if status matches any of the target StatusType values.

    Handles both enum values and string values (from legacy data).
    SQLAlchemy Enum columns store the enum NAME (e.g., 'PASS') but may
    have legacy data with the VALUE (e.g., 'Pass').

    Args:
        status: The status value from database (enum or string)
        *targets: One or more StatusType enum values to match against

    Returns:
        True if status matches any target
    """
    if status is None:
        return False

    # Get both names (PASS) and values (Pass) we're looking for
    target_names = {t.name for t in targets}
    target_values = {t.value for t in targets}

    # Handle both enum and string
    if isinstance(status, DBStatusType):
        return status in targets
    elif isinstance(status, str):
        # Check both name and value for backwards compatibility
        return status in target_names or status in target_values
    else:
        return False


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class DatabaseManager:
    """
    Simplified database manager for v3.

    Features:
    - Single-file SQLite database (self-contained)
    - Context manager for safe transactions
    - Incremental processing support
    - Memory-efficient queries
    """

    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the database manager.

        Args:
            database_path: Path to SQLite database. If None, uses config default.
        """
        config = get_config()

        if database_path is None:
            database_path = config.database.path

        # Ensure parent directory exists
        database_path = Path(database_path)
        database_path.parent.mkdir(parents=True, exist_ok=True)

        self.database_path = database_path
        self.database_url = f"sqlite:///{database_path}"

        logger.info(f"Using database: {database_path}")

        # Create engine with SQLite-appropriate settings
        # timeout=30 gives SQLite time to wait for locks instead of failing immediately
        self._engine = create_engine(
            self.database_url,
            echo=False,
            poolclass=StaticPool,  # Good for SQLite
            connect_args={
                "check_same_thread": False,
                "timeout": 30,  # Wait up to 30s for locks
            },
        )

        # Enable WAL mode for better concurrency (allows readers during writes)
        with self._engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA busy_timeout=30000"))  # 30 second timeout
            conn.commit()

        # Create session factory
        self._SessionFactory = sessionmaker(bind=self._engine)

        # Thread-local session storage
        self._thread_local = threading.local()

        # Single write lock for thread safety (SQLite is single-writer)
        # Using one lock instead of separate locks reduces contention
        self._write_lock = threading.Lock()

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Create all tables if they don't exist and run migrations."""
        try:
            Base.metadata.create_all(self._engine, checkfirst=True)
            self._run_migrations()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")

    @staticmethod
    def _reparse_filename(filename: str) -> tuple:
        """Re-parse a filename to extract model and serial using improved logic.

        Standalone version of ExcelParser._parse_filename() for use in migrations
        without importing the parser (avoids circular dependencies).
        """
        import re
        from pathlib import Path

        def _is_valid_suffix(suffix):
            if suffix.lower().startswith('shop'):
                return False
            if re.match(r'^[A-Za-z0-9]{1,3}$', suffix):
                return True
            if suffix.lower() == 'outer':
                return True
            return False

        name = Path(filename).stem
        parts = re.split(r'[_\s]+', name)

        # Concatenated model+sn+serial (e.g., "7928sn1040", "8340-1-sn201")
        if parts:
            concat_match = re.match(
                r'^(\d{4,}[A-Za-z]?(?:-\d+[A-Za-z]?)?)-?[sS][nN](\d+[a-zA-Z]?)$',
                parts[0]
            )
            if concat_match:
                return concat_match.group(1), concat_match.group(2)

        if len(parts) >= 2:
            model = "Unknown"
            serial = "Unknown"

            for part in parts:
                model_match = re.match(r'^(\d{4,}[A-Za-z]?)((?:-[A-Za-z0-9]+)*)$', part)
                if model_match:
                    base = model_match.group(1)
                    suffixes_str = model_match.group(2)

                    if not suffixes_str:
                        model = part
                        break

                    suffixes = suffixes_str.split('-')[1:]

                    if suffixes[-1].lower() == 'sn':
                        model = base + '-'.join([''] + suffixes[:-1]) if len(suffixes) > 1 else base
                        break

                    valid_parts = [base]
                    for s in suffixes:
                        if _is_valid_suffix(s):
                            valid_parts.append(s)
                        else:
                            model = '-'.join(valid_parts)
                            serial = s
                            break
                    else:
                        model = part
                    break

            if serial == "Unknown" and model != "Unknown":
                skip_keywords = {
                    'test', 'data', 'deg', 'ta', 'tb', 'trimmed', 'correct',
                    'scrap', 'cut', 'wiper', 'path', 'am', 'pm',
                    'fail', 'pass', 'final', 'template', 'primary',
                    'customer', 'report', 'master', 'noise', 'copy', 'of',
                    'pre', 'outer', 'redunat', 'redundant',
                }
                for i, part in enumerate(parts):
                    if part == model:
                        continue
                    if '-' in part and part.startswith(model + '-'):
                        continue
                    if part.lower() in skip_keywords:
                        continue
                    if i + 1 < len(parts) and parts[i + 1].lower() == 'deg':
                        continue
                    if re.match(r'^\d{1,4}-\d{1,2}-\d{1,4}$', part):
                        continue
                    if re.match(r'^\d{1,2}-\d{2}(-\d{2})?$', part):
                        continue
                    if re.match(r'^\d{8,}', part):
                        continue
                    if re.match(r'^[A-Z]{1,3}\d+', part, re.IGNORECASE):
                        serial = part
                        break
                    elif re.match(r'^\d+$', part):
                        serial = part
                        break

            return model, serial

        # Single-part filename: only return as model if it looks like a valid model number
        if re.match(r'^\d{4,}[A-Za-z]?$', name):
            return name, "Unknown"
        return "Unknown", "Unknown"

    def _run_migrations(self) -> None:
        """Run database migrations for schema updates."""
        needs_rematch = False

        with self.session() as session:
            # Migration: Add is_anomaly and anomaly_reason columns to track_results
            try:
                # Check if columns exist by attempting a query
                session.execute(text("SELECT is_anomaly FROM track_results LIMIT 1"))
            except OperationalError:
                # Columns don't exist, add them
                logger.info("Running migration: Adding is_anomaly and anomaly_reason columns")
                try:
                    session.execute(text("ALTER TABLE track_results ADD COLUMN is_anomaly BOOLEAN DEFAULT 0"))
                    session.execute(text("ALTER TABLE track_results ADD COLUMN anomaly_reason TEXT"))
                    session.commit()
                    logger.info("Migration completed: Added anomaly detection columns")
                except Exception as e:
                    logger.warning(f"Migration warning (may already exist): {e}")

            # Migration: Add drift_baseline_cutoff_date column to model_ml_state
            try:
                session.execute(text("SELECT drift_baseline_cutoff_date FROM model_ml_state LIMIT 1"))
            except OperationalError:
                logger.info("Running migration: Adding drift_baseline_cutoff_date column")
                try:
                    session.execute(text("ALTER TABLE model_ml_state ADD COLUMN drift_baseline_cutoff_date DATETIME"))
                    session.commit()
                    logger.info("Migration completed: Added drift_baseline_cutoff_date column")
                except Exception as e:
                    logger.warning(f"Migration warning (may already exist): {e}")

            # Migration: Add peak_cusum column to model_ml_state
            try:
                session.execute(text("SELECT peak_cusum FROM model_ml_state LIMIT 1"))
            except OperationalError:
                logger.info("Running migration: Adding peak_cusum column")
                try:
                    session.execute(text("ALTER TABLE model_ml_state ADD COLUMN peak_cusum FLOAT DEFAULT 0"))
                    session.commit()
                    logger.info("Migration completed: Added peak_cusum column")
                except Exception as e:
                    logger.warning(f"Migration warning (may already exist): {e}")

            # Migration: Normalize status values from 'Pass' to 'PASS' format
            # SQLAlchemy stores enum NAME (PASS), not value (Pass)
            # This fixes data corrupted by bulk SQL that used .value instead of .name
            try:
                # Check if there are any title-case values that need fixing
                result = session.execute(text(
                    "SELECT COUNT(*) FROM analysis_results WHERE overall_status IN ('Pass', 'Fail', 'Warning', 'Error')"
                )).scalar()
                if result and result > 0:
                    logger.info(f"Running migration: Normalizing {result} status values to uppercase")
                    # Fix analysis_results
                    session.execute(text("UPDATE analysis_results SET overall_status = 'PASS' WHERE overall_status = 'Pass'"))
                    session.execute(text("UPDATE analysis_results SET overall_status = 'FAIL' WHERE overall_status = 'Fail'"))
                    session.execute(text("UPDATE analysis_results SET overall_status = 'WARNING' WHERE overall_status = 'Warning'"))
                    session.execute(text("UPDATE analysis_results SET overall_status = 'ERROR' WHERE overall_status = 'Error'"))
                    # Fix track_results
                    session.execute(text("UPDATE track_results SET status = 'PASS' WHERE status = 'Pass'"))
                    session.execute(text("UPDATE track_results SET status = 'FAIL' WHERE status = 'Fail'"))
                    session.execute(text("UPDATE track_results SET status = 'WARNING' WHERE status = 'Warning'"))
                    session.execute(text("UPDATE track_results SET status = 'ERROR' WHERE status = 'Error'"))
                    session.commit()
                    logger.info("Migration completed: Status values normalized")
            except Exception as e:
                logger.warning(f"Status normalization warning: {e}")

            # Migration: Clean up "-shop" model name parsing artifacts
            # Files like "8444-shop0_date.xlsx" were incorrectly parsed with
            # model="8444-shop0" instead of model="8444", serial="shop0"
            try:
                shop_count = session.execute(text(
                    "SELECT COUNT(*) FROM analysis_results WHERE LOWER(model) LIKE '%-shop%'"
                )).scalar()
                if shop_count and shop_count > 0:
                    logger.info(f"Running migration: Cleaning up {shop_count} shop model name records")

                    # Fix analysis_results: split "8444-shop0" into model="8444", serial="shop0"
                    shop_records = session.execute(text(
                        "SELECT id, model FROM analysis_results WHERE LOWER(model) LIKE '%-shop%'"
                    )).fetchall()

                    for row in shop_records:
                        old_model = row[1]
                        # Find the "-shop" split point (case-insensitive)
                        lower = old_model.lower()
                        shop_idx = lower.find('-shop')
                        if shop_idx > 0:
                            base_model = old_model[:shop_idx]
                            new_serial = old_model[shop_idx + 1:]  # "shop0", "shop101", etc.
                            session.execute(text(
                                "UPDATE analysis_results SET model = :model, serial = :serial WHERE id = :id"
                            ), {"model": base_model, "serial": new_serial, "id": row[0]})

                    # Delete model_ml_state entries for fake shop model names
                    ml_deleted = session.execute(text(
                        "DELETE FROM model_ml_state WHERE LOWER(model) LIKE '%-shop%'"
                    )).rowcount
                    logger.info(f"Deleted {ml_deleted} fake ML state entries for shop models")

                    session.commit()
                    needs_rematch = True
                    logger.info(f"Migration completed: Cleaned up {shop_count} shop model name records")
            except Exception as e:
                logger.warning(f"Shop model cleanup warning: {e}")

            # Migration: Re-parse "Unknown" model records with improved parser logic
            # Handles: multi-hyphen models (7280-1-CT), -sn serial indicators,
            # concatenated sn patterns, "final NNN" serials, etc.
            try:
                unknown_records = session.execute(text(
                    "SELECT id, filename FROM analysis_results WHERE model = 'Unknown'"
                )).fetchall()
                if unknown_records:
                    logger.info(f"Running migration: Re-parsing {len(unknown_records)} Unknown model records")
                    fixed = 0
                    for row in unknown_records:
                        rec_id, filename = row[0], row[1]
                        model, serial = self._reparse_filename(filename)
                        if model != "Unknown":
                            session.execute(text(
                                "UPDATE analysis_results SET model = :model, serial = :serial WHERE id = :id"
                            ), {"model": model, "serial": serial, "id": rec_id})
                            fixed += 1
                    if fixed > 0:
                        session.commit()
                        needs_rematch = True
                        logger.info(f"Migration completed: Fixed {fixed} of {len(unknown_records)} Unknown model records")
                    else:
                        logger.info("Migration: No Unknown records could be re-parsed")
            except Exception as e:
                logger.warning(f"Unknown model re-parse warning: {e}")

            # Migration: Ensure all performance indexes exist
            # create_all() only creates indexes for NEW tables. Existing databases
            # may be missing indexes that were added to models.py later.
            # CREATE INDEX IF NOT EXISTS is idempotent — safe to run every startup.
            try:
                index_statements = [
                    # analysis_results indexes
                    "CREATE INDEX IF NOT EXISTS idx_filename_date ON analysis_results(filename, file_date)",
                    "CREATE INDEX IF NOT EXISTS idx_file_date ON analysis_results(file_date)",
                    "CREATE INDEX IF NOT EXISTS idx_model_serial ON analysis_results(model, serial)",
                    "CREATE INDEX IF NOT EXISTS idx_model_serial_date ON analysis_results(model, serial, file_date)",
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON analysis_results(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_status ON analysis_results(overall_status)",
                    "CREATE INDEX IF NOT EXISTS idx_system ON analysis_results(system)",
                    "CREATE INDEX IF NOT EXISTS idx_status_timestamp ON analysis_results(overall_status, timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_model_status ON analysis_results(model, overall_status)",
                    # track_results indexes
                    "CREATE INDEX IF NOT EXISTS idx_track_analysis ON track_results(analysis_id, track_id)",
                    "CREATE INDEX IF NOT EXISTS idx_track_sigma_gradient ON track_results(sigma_gradient)",
                    "CREATE INDEX IF NOT EXISTS idx_track_sigma_pass ON track_results(sigma_pass)",
                    "CREATE INDEX IF NOT EXISTS idx_track_linearity_pass ON track_results(linearity_pass)",
                    "CREATE INDEX IF NOT EXISTS idx_track_risk_category ON track_results(risk_category)",
                    "CREATE INDEX IF NOT EXISTS idx_track_failure_probability ON track_results(failure_probability)",
                    "CREATE INDEX IF NOT EXISTS idx_track_status ON track_results(status)",
                    "CREATE INDEX IF NOT EXISTS idx_track_analysis_prob ON track_results(analysis_id, failure_probability)",
                    # final_test_results indexes
                    "CREATE INDEX IF NOT EXISTS idx_ft_filename_date ON final_test_results(filename, file_date)",
                    "CREATE INDEX IF NOT EXISTS idx_ft_model_serial ON final_test_results(model, serial)",
                    "CREATE INDEX IF NOT EXISTS idx_ft_model_serial_date ON final_test_results(model, serial, file_date)",
                    "CREATE INDEX IF NOT EXISTS idx_ft_timestamp ON final_test_results(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_ft_status ON final_test_results(overall_status)",
                    "CREATE INDEX IF NOT EXISTS idx_ft_linked_trim ON final_test_results(linked_trim_id)",
                    "CREATE INDEX IF NOT EXISTS idx_ft_test_date ON final_test_results(test_date)",
                ]
                created = 0
                for stmt in index_statements:
                    session.execute(text(stmt))
                    created += 1
                session.commit()
                logger.info(f"Index migration: ensured {created} indexes exist")
            except Exception as e:
                logger.warning(f"Index migration warning: {e}")

            # Migration: Add failure margin columns to track_results
            try:
                session.execute(text("SELECT max_violation FROM track_results LIMIT 1"))
            except OperationalError:
                logger.info("Running migration: Adding failure margin columns")
                try:
                    session.execute(text("ALTER TABLE track_results ADD COLUMN max_violation FLOAT"))
                    session.execute(text("ALTER TABLE track_results ADD COLUMN avg_violation FLOAT"))
                    session.execute(text("ALTER TABLE track_results ADD COLUMN margin_to_spec FLOAT"))
                    session.commit()
                    logger.info("Migration completed: Added failure margin columns")
                except Exception as e:
                    logger.warning(f"Failure margin migration warning (may already exist): {e}")

            # Migration: Add measured_electrical_angle column to track_results
            try:
                session.execute(text("SELECT measured_electrical_angle FROM track_results LIMIT 1"))
            except OperationalError:
                logger.info("Running migration: Adding measured_electrical_angle column")
                try:
                    session.execute(text("ALTER TABLE track_results ADD COLUMN measured_electrical_angle FLOAT"))
                    session.commit()
                    logger.info("Migration completed: Added measured_electrical_angle column")
                except Exception as e:
                    logger.warning(f"measured_electrical_angle migration warning (may already exist): {e}")

            # Migration: Add max deviation columns to track_results
            try:
                session.execute(text("SELECT max_deviation FROM track_results LIMIT 1"))
            except OperationalError:
                logger.info("Running migration: Adding max deviation columns")
                try:
                    session.execute(text("ALTER TABLE track_results ADD COLUMN max_deviation FLOAT"))
                    session.execute(text("ALTER TABLE track_results ADD COLUMN max_deviation_position FLOAT"))
                    session.execute(text("ALTER TABLE track_results ADD COLUMN deviation_uniformity FLOAT"))
                    session.commit()
                    logger.info("Migration completed: Added max deviation columns")
                except Exception as e:
                    logger.warning(f"Max deviation migration warning (may already exist): {e}")

            # Migration: Add data_quality columns to analysis_results
            try:
                session.execute(text("SELECT data_quality FROM analysis_results LIMIT 1"))
            except OperationalError:
                logger.info("Running migration: Adding data_quality columns")
                try:
                    session.execute(text(
                        "ALTER TABLE analysis_results ADD COLUMN data_quality VARCHAR(20) DEFAULT 'good'"
                    ))
                    session.execute(text(
                        "ALTER TABLE analysis_results ADD COLUMN data_quality_issues TEXT"
                    ))
                    session.commit()
                    logger.info("Migration completed: Added data_quality columns")
                except Exception as e:
                    logger.warning(f"Data quality migration warning (may already exist): {e}")

            # Migration: Add Phase 2 spec-aware optimization columns to track_results
            phase2_columns = {
                "optimal_slope": "FLOAT DEFAULT 1.0",
                "station_compensation": "FLOAT",
                "linearity_type": "VARCHAR(30)",
                "raw_linearity_error": "FLOAT",
                "optimized_linearity_error": "FLOAT",
                "raw_fail_points": "INTEGER",
            }
            for col_name, col_type in phase2_columns.items():
                try:
                    session.execute(text(
                        f"ALTER TABLE track_results ADD COLUMN {col_name} {col_type}"
                    ))
                except Exception:
                    pass  # Column already exists
            try:
                session.commit()
                logger.info("Phase 2 migration: ensured spec-aware columns exist")
            except Exception:
                pass

            # Migration: Add match_method column to final_test_results
            try:
                session.execute(text("SELECT match_method FROM final_test_results LIMIT 1"))
            except OperationalError:
                try:
                    session.execute(text("ALTER TABLE final_test_results ADD COLUMN match_method VARCHAR(30)"))
                    session.commit()
                    logger.info("Migration: Added match_method column to final_test_results")
                except Exception:
                    pass

        # After session closes, re-run FT matching if model names were corrected
        if needs_rematch:
            try:
                logger.info("Re-matching Final Test records after model name cleanup...")
                stats = self.rematch_final_tests()
                logger.info(f"Post-cleanup FT rematch: {stats}")
            except Exception as e:
                logger.warning(f"FT rematch after cleanup failed: {e}")

    @contextmanager
    def session(self) -> Iterator[Session]:
        """
        Provide a transactional session context.

        Usage:
            with db_manager.session() as session:
                session.add(record)
                # Auto-commits on success, rolls back on exception
        """
        session = self._SessionFactory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()

    # =========================================================================
    # Analysis Results
    # =========================================================================

    def save_analysis(self, analysis: AnalysisResult) -> int:
        """
        Save a single analysis result to the database.

        Args:
            analysis: AnalysisResult from processing

        Returns:
            Database ID of the saved analysis, or -1 for final test files
        """
        # Skip Final Test files - they're already saved in processor via save_final_test
        if getattr(analysis, 'file_type', 'trim') == 'final_test':
            logger.debug(f"Skipping save_analysis for Final Test: {analysis.metadata.filename}")
            return getattr(analysis, 'final_test_id', -1) or -1

        # Skip Smoothness files - they're already saved in processor via save_smoothness_result
        if getattr(analysis, 'file_type', 'trim') == 'smoothness':
            logger.debug(f"Skipping save_analysis for Smoothness: {analysis.metadata.filename}")
            return getattr(analysis, 'smoothness_id', -1) or -1

        # Use write lock for thread safety with SQLite
        with self._write_lock:
            with self.session() as session:
                # Check for existing record by filename (stable identifier)
                # This ensures re-analysis updates the existing record even if
                # model/serial/date parsing changed
                existing = session.query(DBAnalysisResult).filter(
                    DBAnalysisResult.filename == analysis.metadata.filename
                ).first()

                if existing:
                    logger.info(f"Updating existing analysis: {analysis.metadata.filename}")
                    return self._update_existing_analysis(session, analysis)

                # No existing record, create new one
                db_analysis = self._map_analysis_to_db(analysis)
                session.add(db_analysis)
                session.flush()  # Get ID before commit

                # Record as processed file
                self._record_processed_file(
                    session,
                    analysis.metadata.file_path,
                    db_analysis.id
                )

                logger.info(f"Saved new analysis: {analysis.metadata.filename} (ID: {db_analysis.id})")
                return db_analysis.id

    def save_batch(self, analyses: List[AnalysisResult]) -> List[int]:
        """
        Save multiple analysis results efficiently.

        Args:
            analyses: List of AnalysisResult objects

        Returns:
            List of database IDs
        """
        saved_ids = []

        with self.session() as session:
            for analysis in analyses:
                # Skip Final Test files - they're already saved in processor
                if getattr(analysis, 'file_type', 'trim') == 'final_test':
                    saved_ids.append(getattr(analysis, 'final_test_id', -1) or -1)
                    continue

                # Check for existing record by filename
                existing = session.query(DBAnalysisResult).filter(
                    DBAnalysisResult.filename == analysis.metadata.filename
                ).first()

                if existing:
                    # Update existing record
                    updated_id = self._update_existing_analysis(session, analysis)
                    saved_ids.append(updated_id)
                else:
                    # Create new record
                    db_analysis = self._map_analysis_to_db(analysis)
                    session.add(db_analysis)
                    session.flush()

                    self._record_processed_file(
                        session,
                        analysis.metadata.file_path,
                        db_analysis.id
                    )

                    saved_ids.append(db_analysis.id)

        logger.info(f"Saved batch of {len(saved_ids)} analyses")
        return saved_ids

    def get_analysis(self, analysis_id: int) -> Optional[AnalysisResult]:
        """
        Retrieve a single analysis by ID.

        Args:
            analysis_id: Database ID

        Returns:
            AnalysisResult or None if not found
        """
        with self.session() as session:
            db_analysis = session.query(DBAnalysisResult).get(analysis_id)

            if db_analysis is None:
                return None

            return self._map_db_to_analysis(db_analysis)

    def get_historical_data(
        self,
        model: Optional[str] = None,
        days_back: int = 30,
        limit: int = 1000
    ) -> List[AnalysisResult]:
        """
        Get historical analysis data with optional filtering.

        Filters and sorts by trim date (file_date), not processing date (timestamp).
        This shows files based on when they were trimmed, not when processed.

        Args:
            model: Filter by model number (optional)
            days_back: How many days back to query (based on trim date)
            limit: Maximum number of results

        Returns:
            List of AnalysisResult objects sorted by trim date (newest first)
        """
        with self.session() as session:
            # Use joinedload to fetch tracks in single query (avoids N+1)
            query = session.query(DBAnalysisResult).options(
                joinedload(DBAnalysisResult.tracks)
            )

            # Filter by trim date (file_date), not processing date (timestamp)
            cutoff_date = datetime.now() - timedelta(days=days_back)
            query = query.filter(DBAnalysisResult.file_date >= cutoff_date)

            if model:
                query = query.filter(DBAnalysisResult.model == model)

            # Order by trim date (file_date), newest first
            query = query.order_by(desc(DBAnalysisResult.file_date)).limit(limit)

            results = []
            for db_analysis in query.all():
                mapped = self._map_db_to_analysis(db_analysis)
                if mapped is not None:  # Filter out failed mappings
                    results.append(mapped)

            return results

    def get_model_statistics(self, model: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Get aggregated statistics for a model.

        Filters by trim date (file_date), not processing date.

        Args:
            model: Model number
            days_back: Days to include in statistics (based on trim date)

        Returns:
            Dictionary with statistics
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Query track results for this model - filter by trim date
            results = (
                session.query(DBTrackResult)
                .join(DBAnalysisResult)
                .filter(
                    DBAnalysisResult.model == model,
                    DBAnalysisResult.file_date >= cutoff_date
                )
                .all()
            )

            if not results:
                return {
                    "model": model,
                    "count": 0,
                    "pass_rate": 0.0,
                    "avg_sigma_gradient": None,
                    "avg_failure_probability": None,
                }

            # Calculate statistics
            total_tracks = len(results)
            passed_tracks = sum(1 for r in results if r.sigma_pass)
            sigma_values = [r.sigma_gradient for r in results if r.sigma_gradient is not None]
            prob_values = [r.failure_probability for r in results if r.failure_probability is not None]

            return {
                "model": model,
                "count": total_tracks,
                "pass_rate": (passed_tracks / total_tracks * 100) if total_tracks > 0 else 0.0,
                "avg_sigma_gradient": sum(sigma_values) / len(sigma_values) if sigma_values else None,
                "avg_failure_probability": sum(prob_values) / len(prob_values) if prob_values else None,
            }

    # =========================================================================
    # Incremental Processing
    # =========================================================================

    def is_file_processed(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file has already been successfully processed.

        Uses SHA-256 hash for accurate duplicate detection even if file
        was moved or renamed. Only returns True for successful processing -
        errors will be retried. Checks both trim files and Final Test files.

        Args:
            file_path: Path to the file

        Returns:
            True if file was already successfully processed
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return False

        file_hash = self._calculate_file_hash(file_path)

        from laser_trim_analyzer.database.models import FinalTestResult as DBFinalTestResult

        with self.session() as session:
            # Check trim files (ProcessedFile) - only successful ones
            exists = (
                session.query(DBProcessedFile)
                .filter(
                    DBProcessedFile.file_hash == file_hash,
                    DBProcessedFile.success == True
                )
                .first()
            ) is not None

            if exists:
                return True

            # Also check Final Test files
            exists = (
                session.query(DBFinalTestResult)
                .filter(DBFinalTestResult.file_hash == file_hash)
                .first()
            ) is not None

            if exists:
                return True

            # Also check Output Smoothness files
            try:
                from laser_trim_analyzer.database.models import SmoothnessResult as DBSmoothnessResult
                exists = (
                    session.query(DBSmoothnessResult)
                    .filter(DBSmoothnessResult.file_hash == file_hash)
                    .first()
                ) is not None
                if exists:
                    return True
            except Exception:
                pass  # Table may not exist yet

            return False

    def get_unprocessed_files(self, file_paths: List[Path]) -> List[Path]:
        """
        Filter a list of files to only those not yet processed.

        Efficient batch operation for incremental processing.

        Args:
            file_paths: List of file paths to check

        Returns:
            List of paths that have not been processed
        """
        if not file_paths:
            return []

        # Calculate hashes for all files
        file_hashes = {}
        for path in file_paths:
            path = Path(path)
            if path.exists():
                file_hashes[self._calculate_file_hash(path)] = path

        if not file_hashes:
            return []

        # Query for existing hashes across all file type tables
        hash_list = list(file_hashes.keys())
        with self.session() as session:
            existing_hashes = set(
                row.file_hash for row in
                session.query(DBProcessedFile.file_hash)
                .filter(DBProcessedFile.file_hash.in_(hash_list))
                .all()
            )

            # Also check Final Test hashes
            from laser_trim_analyzer.database.models import FinalTestResult as DBFinalTestResult
            existing_hashes.update(
                row.file_hash for row in
                session.query(DBFinalTestResult.file_hash)
                .filter(DBFinalTestResult.file_hash.in_(hash_list))
                .all()
                if row.file_hash
            )

            # Also check Smoothness hashes
            try:
                from laser_trim_analyzer.database.models import SmoothnessResult as DBSmoothnessResult
                existing_hashes.update(
                    row.file_hash for row in
                    session.query(DBSmoothnessResult.file_hash)
                    .filter(DBSmoothnessResult.file_hash.in_(hash_list))
                    .all()
                    if row.file_hash
                )
            except Exception:
                pass  # Table may not exist yet

        # Return files whose hash is not in database
        return [
            path for hash_val, path in file_hashes.items()
            if hash_val not in existing_hashes
        ]

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _record_processed_file(
        self,
        session: Session,
        file_path: Path,
        analysis_id: int
    ) -> None:
        """Record a file as processed."""
        file_path = Path(file_path)

        if not file_path.exists():
            return

        try:
            processed_file = DBProcessedFile(
                filename=file_path.name,
                file_path=str(file_path),
                file_hash=self._calculate_file_hash(file_path),
                file_size=file_path.stat().st_size,
                file_modified_date=datetime.fromtimestamp(file_path.stat().st_mtime),
                analysis_id=analysis_id,
                success=True,
            )
            session.add(processed_file)
        except IntegrityError:
            # Already recorded, ignore
            pass

    # =========================================================================
    # QA Alerts
    # =========================================================================

    def create_alert(
        self,
        analysis_id: int,
        alert_type: str,
        severity: str,
        message: str,
        track_id: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
    ) -> int:
        """
        Create a QA alert.

        Args:
            analysis_id: Related analysis ID
            alert_type: Type of alert (from AlertType enum)
            severity: Severity level (Critical, High, Medium, Low)
            message: Alert message
            track_id: Related track (optional)
            metric_value: Value that triggered alert (optional)
            threshold_value: Threshold that was exceeded (optional)

        Returns:
            Database ID of created alert
        """
        with self.session() as session:
            alert = DBQAAlert(
                analysis_id=analysis_id,
                alert_type=DBAlertType[alert_type] if isinstance(alert_type, str) else alert_type,
                severity=severity,
                message=message,
                track_id=track_id,
                metric_value=metric_value,
                threshold_value=threshold_value,
            )
            session.add(alert)
            session.flush()

            logger.info(f"Created alert: {alert_type} - {message}")
            return alert.id

    def get_unresolved_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get unresolved QA alerts.

        Args:
            limit: Maximum number to return

        Returns:
            List of alert dictionaries
        """
        with self.session() as session:
            alerts = (
                session.query(DBQAAlert)
                .filter(DBQAAlert.resolved == False)
                .order_by(
                    # Critical first, then by date
                    desc(DBQAAlert.severity == "Critical"),
                    desc(DBQAAlert.created_date)
                )
                .limit(limit)
                .all()
            )

            return [
                {
                    "id": a.id,
                    "analysis_id": a.analysis_id,
                    "alert_type": a.alert_type.value if a.alert_type else None,
                    "severity": a.severity,
                    "message": a.message,
                    "created_date": a.created_date,
                    "acknowledged": a.acknowledged,
                }
                for a in alerts
            ]

    def acknowledge_alert(self, alert_id: int, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID
            acknowledged_by: User who acknowledged

        Returns:
            True if successful
        """
        with self.session() as session:
            alert = session.query(DBQAAlert).get(alert_id)
            if alert:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_date = datetime.now()
                return True
            return False

    def resolve_alert(
        self,
        alert_id: int,
        resolved_by: str,
        resolution_notes: Optional[str] = None
    ) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert ID
            resolved_by: User who resolved
            resolution_notes: Notes about resolution

        Returns:
            True if successful
        """
        with self.session() as session:
            alert = session.query(DBQAAlert).get(alert_id)
            if alert:
                if not alert.acknowledged:
                    # Auto-acknowledge when resolving
                    alert.acknowledged = True
                    alert.acknowledged_by = resolved_by
                    alert.acknowledged_date = datetime.now()

                alert.resolved = True
                alert.resolved_by = resolved_by
                alert.resolved_date = datetime.now()
                alert.resolution_notes = resolution_notes
                return True
            return False

    # =========================================================================
    # Dashboard Queries
    # =========================================================================

    def get_dashboard_stats(self, days_back: int = 7,
                            element_type: Optional[str] = None,
                            product_class: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for dashboard display.

        Filters by trim date (file_date), not processing date.
        Optionally filters by element type and/or product class via model_specs join.

        Args:
            days_back: Number of days to include (based on trim date)
            element_type: Filter to models with this element type
            product_class: Filter to models with this product class

        Returns:
            Dictionary with dashboard statistics
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Build model filter list from model_specs if filters are active
            filter_models = None
            if element_type or product_class:
                q = session.query(ModelSpec.model)
                if element_type:
                    q = q.filter(ModelSpec.element_type == element_type)
                if product_class:
                    q = q.filter(ModelSpec.product_class == product_class)
                filter_models = [r[0] for r in q.all()]
                if not filter_models:
                    # No models match — return empty stats
                    return {
                        "total_analyses": 0, "total_files": 0,
                        "passed": 0, "failed": 0, "pass_rate": 0.0,
                        "sigma_pass_rate": 0.0, "linearity_pass_rate": 0.0,
                        "total_tracks": 0, "unresolved_alerts": 0,
                        "high_risk_count": 0, "period_days": days_back,
                        "today_count": 0, "week_count": 0,
                        "daily_trend": [], "linearity_daily_trend": [],
                    }

            def _base_filter(query):
                """Apply date and optional model filters."""
                query = query.filter(DBAnalysisResult.file_date >= cutoff_date)
                if filter_models is not None:
                    query = query.filter(DBAnalysisResult.model.in_(filter_models))
                return query

            # Count analyses - filter by trim date
            total_analyses = (
                _base_filter(session.query(func.count(DBAnalysisResult.id)))
                .scalar()
            ) or 0

            # Count by status - filter by trim date
            status_q = session.query(
                DBAnalysisResult.overall_status,
                func.count(DBAnalysisResult.id)
            ).filter(DBAnalysisResult.file_date >= cutoff_date)
            if filter_models is not None:
                status_q = status_q.filter(DBAnalysisResult.model.in_(filter_models))
            status_counts = status_q.group_by(DBAnalysisResult.overall_status).all()

            passed = 0
            failed = 0
            for status, count in status_counts:
                if _status_matches(status, DBStatusType.PASS):
                    passed = count
                elif _status_matches(status, DBStatusType.FAIL, DBStatusType.ERROR):
                    failed = count

            # Count unresolved alerts
            unresolved_alerts = (
                session.query(func.count(DBQAAlert.id))
                .filter(DBQAAlert.resolved == False)
                .scalar()
            ) or 0

            # Get high-risk count - filter by trim date
            high_risk_q = (
                session.query(func.count(DBTrackResult.id))
                .join(DBAnalysisResult)
                .filter(
                    DBAnalysisResult.file_date >= cutoff_date,
                    DBTrackResult.risk_category == DBRiskCategory.HIGH
                )
            )
            if filter_models is not None:
                high_risk_q = high_risk_q.filter(DBAnalysisResult.model.in_(filter_models))
            high_risk = high_risk_q.scalar() or 0

            pass_rate = (passed / total_analyses * 100) if total_analyses > 0 else 0.0

            # Get track-level sigma and linearity pass rates - filter by trim date
            track_stats_q = (
                session.query(
                    func.count(DBTrackResult.id).label('total_tracks'),
                    func.sum(case((DBTrackResult.sigma_pass == True, 1), else_=0)).label('sigma_passed'),
                    func.sum(case((DBTrackResult.linearity_pass == True, 1), else_=0)).label('linearity_passed'),
                )
                .join(DBAnalysisResult)
                .filter(DBAnalysisResult.file_date >= cutoff_date)
            )
            if filter_models is not None:
                track_stats_q = track_stats_q.filter(DBAnalysisResult.model.in_(filter_models))
            track_stats = track_stats_q.first()

            total_tracks = track_stats.total_tracks or 0
            sigma_passed = track_stats.sigma_passed or 0
            linearity_passed = track_stats.linearity_passed or 0

            sigma_pass_rate = (sigma_passed / total_tracks * 100) if total_tracks > 0 else 0.0
            linearity_pass_rate = (linearity_passed / total_tracks * 100) if total_tracks > 0 else 0.0

            # Total files (all time)
            total_files = (
                session.query(func.count(DBAnalysisResult.id))
                .scalar()
            ) or 0

            # Today's count (by trim date)
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_q = session.query(func.count(DBAnalysisResult.id)).filter(DBAnalysisResult.file_date >= today)
            if filter_models is not None:
                today_q = today_q.filter(DBAnalysisResult.model.in_(filter_models))
            today_count = today_q.scalar() or 0

            # This week's count (by trim date)
            week_start = today - timedelta(days=today.weekday())
            week_q = session.query(func.count(DBAnalysisResult.id)).filter(DBAnalysisResult.file_date >= week_start)
            if filter_models is not None:
                week_q = week_q.filter(DBAnalysisResult.model.in_(filter_models))
            week_count = week_q.scalar() or 0

            # Daily trend for the past N days (by trim date) - optimized single query
            trend_start = today - timedelta(days=days_back - 1)

            # Single query with GROUP BY date to get all days at once
            daily_q = (
                session.query(
                    func.date(DBAnalysisResult.file_date).label('day'),
                    func.count(DBAnalysisResult.id).label('total'),
                    func.sum(
                        case(
                            (DBAnalysisResult.overall_status == DBStatusType.PASS, 1),
                            else_=0
                        )
                    ).label('passed')
                )
                .filter(DBAnalysisResult.file_date >= trend_start)
            )
            if filter_models is not None:
                daily_q = daily_q.filter(DBAnalysisResult.model.in_(filter_models))
            daily_data = (
                daily_q
                .group_by(func.date(DBAnalysisResult.file_date))
                .all()
            )

            # Convert to dict for quick lookup
            daily_dict = {str(row.day): {'total': row.total, 'passed': row.passed or 0} for row in daily_data}

            # Build daily trend list (fill in zeros for missing days)
            daily_trend = []
            for i in range(days_back):
                day = trend_start + timedelta(days=i)
                day_str = day.strftime("%Y-%m-%d")
                day_data = daily_dict.get(day_str, {'total': 0, 'passed': 0})
                day_total = day_data['total']
                day_passed = day_data['passed']
                day_pass_rate = (day_passed / day_total * 100) if day_total > 0 else 0.0
                daily_trend.append({
                    "date": day.strftime("%m/%d"),
                    "total": day_total,
                    "passed": day_passed,
                    "pass_rate": day_pass_rate,
                })

            # Linearity daily trend (track-level, independent of sigma)
            lin_daily_q = (
                session.query(
                    func.date(DBAnalysisResult.file_date).label('day'),
                    func.count(DBTrackResult.id).label('total'),
                    func.sum(case((DBTrackResult.linearity_pass == True, 1), else_=0)).label('passed')
                )
                .join(DBTrackResult, DBAnalysisResult.id == DBTrackResult.analysis_id)
                .filter(DBAnalysisResult.file_date >= trend_start)
            )
            if filter_models is not None:
                lin_daily_q = lin_daily_q.filter(DBAnalysisResult.model.in_(filter_models))
            linearity_daily_data = (
                lin_daily_q
                .group_by(func.date(DBAnalysisResult.file_date))
                .all()
            )

            lin_daily_dict = {str(row.day): {'total': row.total, 'passed': row.passed or 0}
                              for row in linearity_daily_data}

            linearity_daily_trend = []
            for i in range(days_back):
                day = trend_start + timedelta(days=i)
                day_str = day.strftime("%Y-%m-%d")
                day_data = lin_daily_dict.get(day_str, {'total': 0, 'passed': 0})
                day_total = day_data['total']
                day_passed = day_data['passed']
                day_pass_rate = (day_passed / day_total * 100) if day_total > 0 else 0.0
                linearity_daily_trend.append({
                    "date": day.strftime("%m/%d"),
                    "total": day_total,
                    "passed": day_passed,
                    "pass_rate": day_pass_rate,
                })

            return {
                "total_analyses": total_analyses,
                "total_files": total_files,
                "passed": passed,
                "failed": failed,
                "pass_rate": pass_rate,
                "sigma_pass_rate": sigma_pass_rate,
                "linearity_pass_rate": linearity_pass_rate,
                "total_tracks": total_tracks,
                "unresolved_alerts": unresolved_alerts,
                "high_risk_count": high_risk,
                "period_days": days_back,
                "today_count": today_count,
                "week_count": week_count,
                "daily_trend": daily_trend,
                "linearity_daily_trend": linearity_daily_trend,
            }

    def get_pass_rate_by_category(self, category: str = "element_type",
                                  days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Get pass rate grouped by element_type or product_class.

        Args:
            category: "element_type" or "product_class"
            days_back: Number of days to look back

        Returns:
            List of dicts with keys: category, total, passed, pass_rate
        """
        with self.session() as session:
            cutoff = datetime.now() - timedelta(days=days_back)
            spec_col = ModelSpec.element_type if category == "element_type" else ModelSpec.product_class

            results = session.query(
                spec_col.label("category"),
                func.count(DBAnalysisResult.id).label("total"),
                func.sum(
                    case(
                        (DBAnalysisResult.overall_status == DBStatusType.PASS, 1),
                        else_=0
                    )
                ).label("passed")
            ).join(
                ModelSpec,
                DBAnalysisResult.model == ModelSpec.model
            ).filter(
                DBAnalysisResult.file_date >= cutoff,
                spec_col.isnot(None)
            ).group_by(spec_col).all()

            return [
                {
                    "category": r.category,
                    "total": r.total,
                    "passed": r.passed or 0,
                    "pass_rate": ((r.passed or 0) / r.total * 100) if r.total > 0 else 0
                }
                for r in results
            ]

    def get_last_batch_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the most recent processing batch.

        A "batch" is defined as files processed within a 1-hour window.
        Uses the timestamp field (when file was added to DB) to identify batches.

        Returns:
            Dictionary with last batch statistics
        """
        with self.session() as session:
            # Find the most recent processing timestamp
            latest_timestamp = (
                session.query(func.max(DBAnalysisResult.timestamp))
                .scalar()
            )

            if not latest_timestamp:
                return {
                    "has_batch": False,
                    "files_processed": 0,
                    "passed": 0,
                    "warnings": 0,
                    "failed": 0,
                    "pass_rate": 0.0,
                    "batch_start": None,
                    "batch_end": None,
                }

            # Define batch as files processed within 1 hour of the latest
            batch_start = latest_timestamp - timedelta(hours=1)

            # Get files in this batch
            batch_files = (
                session.query(DBAnalysisResult)
                .filter(DBAnalysisResult.timestamp >= batch_start)
                .all()
            )

            if not batch_files:
                return {
                    "has_batch": False,
                    "files_processed": 0,
                    "passed": 0,
                    "warnings": 0,
                    "failed": 0,
                    "pass_rate": 0.0,
                    "batch_start": None,
                    "batch_end": None,
                }

            # Count by status
            passed = sum(1 for f in batch_files if f.overall_status == DBStatusType.PASS)
            warnings = sum(1 for f in batch_files if f.overall_status == DBStatusType.WARNING)
            failed = sum(1 for f in batch_files if f.overall_status in (DBStatusType.FAIL, DBStatusType.ERROR))
            total = len(batch_files)

            # Find actual batch time range
            actual_start = min(f.timestamp for f in batch_files)
            actual_end = max(f.timestamp for f in batch_files)

            return {
                "has_batch": True,
                "files_processed": total,
                "passed": passed,
                "warnings": warnings,
                "failed": failed,
                "pass_rate": (passed / total * 100) if total > 0 else 0.0,
                "batch_start": actual_start,
                "batch_end": actual_end,
            }

    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Get overall database statistics across all time.

        Returns:
            Dictionary with overall statistics including date range and model breakdown
        """
        with self.session() as session:
            # Total counts
            total_files = (
                session.query(func.count(DBAnalysisResult.id))
                .scalar()
            ) or 0

            if total_files == 0:
                return {
                    "total_files": 0,
                    "passed": 0,
                    "warnings": 0,
                    "failed": 0,
                    "pass_rate": 0.0,
                    "oldest_date": None,
                    "newest_date": None,
                    "unique_models": 0,
                }

            # Count by status
            status_counts = (
                session.query(
                    DBAnalysisResult.overall_status,
                    func.count(DBAnalysisResult.id)
                )
                .group_by(DBAnalysisResult.overall_status)
                .all()
            )

            passed = 0
            warnings = 0
            failed = 0
            for status, count in status_counts:
                if _status_matches(status, DBStatusType.PASS):
                    passed = count
                elif _status_matches(status, DBStatusType.WARNING):
                    warnings = count
                elif _status_matches(status, DBStatusType.FAIL, DBStatusType.ERROR):
                    failed += count

            # Date range (using file_date - when files were trimmed)
            date_range = (
                session.query(
                    func.min(DBAnalysisResult.file_date),
                    func.max(DBAnalysisResult.file_date)
                )
                .first()
            )
            oldest_date, newest_date = date_range if date_range else (None, None)

            # Unique models count
            unique_models = (
                session.query(func.count(func.distinct(DBAnalysisResult.model)))
                .scalar()
            ) or 0

            # Track-level pass rates (sigma and linearity)
            total_tracks = (
                session.query(func.count(DBTrackResult.id))
                .scalar()
            ) or 0

            sigma_passed = (
                session.query(func.count(DBTrackResult.id))
                .filter(DBTrackResult.sigma_pass == True)
                .scalar()
            ) or 0

            linearity_passed = (
                session.query(func.count(DBTrackResult.id))
                .filter(DBTrackResult.linearity_pass == True)
                .scalar()
            ) or 0

            return {
                "total_files": total_files,
                "passed": passed,
                "warnings": warnings,
                "failed": failed,
                "pass_rate": (passed / total_files * 100) if total_files > 0 else 0.0,
                "oldest_date": oldest_date,
                "newest_date": newest_date,
                "unique_models": unique_models,
                "total_tracks": total_tracks,
                "sigma_pass_rate": (sigma_passed / total_tracks * 100) if total_tracks > 0 else 0.0,
                "linearity_pass_rate": (linearity_passed / total_tracks * 100) if total_tracks > 0 else 0.0,
            }

    def get_system_comparison(self, days_back: int = 90) -> Dict[str, Any]:
        """Get System A vs System B comparison statistics."""
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            system_data = (
                session.query(
                    DBAnalysisResult.system,
                    func.count(func.distinct(DBAnalysisResult.id)).label('total_files'),
                    func.count(DBTrackResult.id).label('total_tracks'),
                    func.sum(case(
                        (DBTrackResult.linearity_pass == True, 1), else_=0
                    )).label('lin_passed'),
                    func.sum(case(
                        (DBTrackResult.sigma_pass == True, 1), else_=0
                    )).label('sigma_passed'),
                )
                .join(DBTrackResult, DBAnalysisResult.id == DBTrackResult.analysis_id)
                .filter(
                    DBAnalysisResult.file_date >= cutoff_date,
                    DBAnalysisResult.system.isnot(None),
                )
                .group_by(DBAnalysisResult.system)
                .all()
            )

            result = {"system_a": None, "system_b": None}
            for row in system_data:
                # Direct comparison — _status_matches is for DBStatusType only
                sys_val = row.system.value if isinstance(row.system, DBSystemType) else str(row.system)
                if sys_val == DBSystemType.A.value:
                    key = "system_a"
                elif sys_val == DBSystemType.B.value:
                    key = "system_b"
                else:
                    continue
                total_tracks = row.total_tracks or 0
                result[key] = {
                    "total_files": row.total_files or 0,
                    "total_tracks": total_tracks,
                    "linearity_pass_rate": (row.lin_passed or 0) / total_tracks * 100 if total_tracks > 0 else 0,
                    "sigma_pass_rate": (row.sigma_passed or 0) / total_tracks * 100 if total_tracks > 0 else 0,
                }
            return result

    def get_ft_dashboard_stats(self, days_back: int = 90) -> Dict[str, Any]:
        """Get Final Test statistics for dashboard display."""
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
            FinalTestTrack as DBFinalTestTrack,
        )

        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            total_ft = (
                session.query(func.count(DBFinalTestResult.id))
                .filter(DBFinalTestResult.file_date >= cutoff_date)
                .scalar()
            ) or 0

            if total_ft == 0:
                return {"total": 0, "pass_rate": 0, "linearity_pass_rate": 0,
                        "linked_count": 0, "link_rate": 0}

            ft_passed = (
                session.query(func.count(DBFinalTestResult.id))
                .filter(
                    DBFinalTestResult.file_date >= cutoff_date,
                    DBFinalTestResult.overall_status == DBStatusType.PASS,
                )
                .scalar()
            ) or 0

            linked_count = (
                session.query(func.count(DBFinalTestResult.id))
                .filter(
                    DBFinalTestResult.file_date >= cutoff_date,
                    DBFinalTestResult.linked_trim_id.isnot(None),
                )
                .scalar()
            ) or 0

            # Track-level linearity pass rate
            ft_track_stats = (
                session.query(
                    func.count(DBFinalTestTrack.id).label('total'),
                    func.sum(case(
                        (DBFinalTestTrack.linearity_pass == True, 1), else_=0
                    )).label('passed'),
                )
                .join(DBFinalTestResult)
                .filter(DBFinalTestResult.file_date >= cutoff_date)
                .first()
            )

            ft_total_tracks = (ft_track_stats.total or 0) if ft_track_stats else 0
            ft_lin_passed = (ft_track_stats.passed or 0) if ft_track_stats else 0

            return {
                "total": total_ft,
                "pass_rate": (ft_passed / total_ft * 100) if total_ft > 0 else 0,
                "linearity_pass_rate": (ft_lin_passed / ft_total_tracks * 100) if ft_total_tracks > 0 else 0,
                "linked_count": linked_count,
                "link_rate": (linked_count / total_ft * 100) if total_ft > 0 else 0,
            }

    def get_escape_overkill_analysis(self, days_back: int = 90, min_confidence: float = 0.70) -> Dict[str, Any]:
        """Analyze escapes and overkills by comparing trim vs FT linearity results.

        Escape = trim passed linearity but FT failed (bad unit shipped)
        Overkill = trim failed linearity but FT passed (unnecessarily rejected)
        """
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
        )

        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # File-level comparison: FT linearity_pass vs trim ALL-tracks-pass
            # For each linked FT record, check if trim linearity passed (all tracks)
            linked_data = (
                session.query(
                    DBFinalTestResult.model,
                    DBFinalTestResult.linearity_pass.label('ft_lin_pass'),
                    # Trim passed linearity if minimum track pass is 1 (all passed)
                    func.min(case(
                        (DBTrackResult.linearity_pass == True, 1), else_=0
                    )).label('trim_all_pass'),
                )
                .join(DBAnalysisResult, DBFinalTestResult.linked_trim_id == DBAnalysisResult.id)
                .join(DBTrackResult, DBAnalysisResult.id == DBTrackResult.analysis_id)
                .filter(
                    DBFinalTestResult.file_date >= cutoff_date,
                    DBFinalTestResult.linked_trim_id.isnot(None),
                    DBFinalTestResult.linearity_pass.isnot(None),
                    DBFinalTestResult.match_confidence >= min_confidence,
                )
                .group_by(DBFinalTestResult.id, DBFinalTestResult.model, DBFinalTestResult.linearity_pass)
                .all()
            )

            if not linked_data:
                return {
                    "total_linked": 0, "escapes": 0, "overkills": 0,
                    "agreements": 0, "escape_rate": 0, "overkill_rate": 0,
                    "agreement_rate": 0, "worst_escape_models": [],
                }

            total = len(linked_data)
            escapes = 0
            overkills = 0
            true_positives = 0  # Both pass
            true_negatives = 0  # Both fail
            model_escapes = {}

            for row in linked_data:
                trim_pass = bool(row.trim_all_pass)
                ft_pass = bool(row.ft_lin_pass)

                if trim_pass and not ft_pass:
                    escapes += 1
                    model_escapes[row.model] = model_escapes.get(row.model, 0) + 1
                elif not trim_pass and ft_pass:
                    overkills += 1
                elif trim_pass and ft_pass:
                    true_positives += 1
                else:
                    true_negatives += 1

            agreements = true_positives + true_negatives
            worst_models = sorted(model_escapes.items(), key=lambda x: x[1], reverse=True)[:5]

            return {
                "total_linked": total,
                "escapes": escapes,
                "overkills": overkills,
                "agreements": agreements,
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "escape_rate": (escapes / total * 100) if total > 0 else 0,
                "overkill_rate": (overkills / total * 100) if total > 0 else 0,
                "agreement_rate": (agreements / total * 100) if total > 0 else 0,
                "worst_escape_models": [{"model": m, "count": c} for m, c in worst_models],
            }

    def get_heatmap_data(
        self, days_back: int = 90, period: str = 'week', min_samples: int = 10
    ) -> Dict[str, Any]:
        """Get model x time period pass rate matrix for heat map visualization.

        Returns:
            {models: [str], periods: [str], values: [[float]]}
            values[i][j] = pass rate for model i in period j (NaN if no data)
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Group by model and week/month period
            if period == 'month':
                period_expr = func.strftime('%Y-%m', DBAnalysisResult.file_date)
            else:
                # Week: use Monday of the week
                period_expr = func.strftime('%Y-W%W', DBAnalysisResult.file_date)

            rows = (
                session.query(
                    DBAnalysisResult.model,
                    period_expr.label('period'),
                    func.count(DBTrackResult.id).label('total'),
                    func.sum(case(
                        (DBTrackResult.linearity_pass == True, 1), else_=0
                    )).label('passed'),
                )
                .join(DBTrackResult, DBAnalysisResult.id == DBTrackResult.analysis_id)
                .filter(DBAnalysisResult.file_date >= cutoff_date)
                .group_by(DBAnalysisResult.model, period_expr)
                .having(func.count(DBTrackResult.id) >= min_samples)
                .all()
            )

            if not rows:
                return {"models": [], "periods": [], "values": []}

            # Build sets of unique models and periods
            model_set = set()
            period_set = set()
            data_map = {}  # (model, period) -> pass_rate

            for row in rows:
                model_set.add(row.model)
                period_set.add(row.period)
                rate = (row.passed / row.total * 100) if row.total > 0 else float('nan')
                data_map[(row.model, row.period)] = rate

            # Sort models by worst average pass rate (worst first)
            model_avgs = {}
            for model in model_set:
                rates = [data_map[(model, p)] for p in period_set if (model, p) in data_map]
                model_avgs[model] = sum(rates) / len(rates) if rates else 100
            models = sorted(model_set, key=lambda m: model_avgs[m])[:25]  # Top 25 worst
            periods = sorted(period_set)

            # Build matrix
            values = []
            for model in models:
                row = []
                for p in periods:
                    val = data_map.get((model, p), float('nan'))
                    row.append(val)
                values.append(row)

            return {"models": models, "periods": periods, "values": values}

    def get_escape_scatter_data(
        self, days_back: int = 90, max_points: int = 500,
        min_confidence: float = 0.70
    ) -> Dict[str, Any]:
        """Get paired trim/FT linearity errors for scatter plot.

        Returns:
            {trim_errors: [float], ft_errors: [float], spec_limit: float}
        """
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
            FinalTestTrack as DBFinalTestTrack,
        )

        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Join trim tracks with FT tracks via linked_trim_id
            rows = (
                session.query(
                    DBTrackResult.final_linearity_error_shifted.label('trim_lin_error'),
                    DBFinalTestTrack.linearity_error.label('ft_lin_error'),
                )
                .join(DBAnalysisResult, DBTrackResult.analysis_id == DBAnalysisResult.id)
                .join(DBFinalTestResult, DBFinalTestResult.linked_trim_id == DBAnalysisResult.id)
                .join(DBFinalTestTrack, DBFinalTestTrack.final_test_id == DBFinalTestResult.id)
                .filter(
                    DBFinalTestResult.file_date >= cutoff_date,
                    DBFinalTestResult.match_confidence >= min_confidence,
                    DBTrackResult.final_linearity_error_shifted.isnot(None),
                    DBFinalTestTrack.linearity_error.isnot(None),
                    # Match track IDs: exact match (A=A, B=B) or FT single-track maps to trim track A
                    or_(
                        DBTrackResult.track_id == DBFinalTestTrack.track_id,
                        and_(
                            DBFinalTestTrack.track_id == "default",
                            DBTrackResult.track_id == "A",
                        ),
                    ),
                )
                .limit(max_points)
                .all()
            )

            if not rows:
                return {"trim_errors": [], "ft_errors": [], "spec_limit": 0.5}

            trim_errors = [float(r.trim_lin_error) for r in rows]
            ft_errors = [float(r.ft_lin_error) for r in rows]

            # Estimate spec limit from data (use 75th percentile as proxy)
            all_errors = sorted(trim_errors + ft_errors)
            if all_errors:
                idx = int(len(all_errors) * 0.75)
                spec_limit = float(all_errors[min(idx, len(all_errors) - 1)])
            else:
                spec_limit = 0.5

            return {
                "trim_errors": trim_errors,
                "ft_errors": ft_errors,
                "spec_limit": spec_limit,
            }

    def get_alerts(self, limit: int = 10, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """
        Get recent alerts.

        Args:
            limit: Maximum number of alerts to return
            include_resolved: Whether to include resolved alerts

        Returns:
            List of alert dictionaries
        """
        with self.session() as session:
            # Join with AnalysisResult to get model info
            query = (
                session.query(DBQAAlert, DBAnalysisResult.model)
                .outerjoin(DBAnalysisResult, DBQAAlert.analysis_id == DBAnalysisResult.id)
            )

            if not include_resolved:
                query = query.filter(DBQAAlert.resolved == False)

            results = (
                query
                .order_by(DBQAAlert.created_date.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "id": alert.id,
                    "analysis_id": alert.analysis_id,
                    "alert_type": alert.alert_type.value if alert.alert_type else "INFO",
                    "severity": alert.severity,
                    "message": alert.message,
                    "model": model or "Unknown",
                    "created_at": alert.created_date.strftime("%Y-%m-%d %H:%M") if alert.created_date else "",
                    "acknowledged": alert.acknowledged,
                    "resolved": alert.resolved,
                }
                for alert, model in results
            ]

    def get_model_stats(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get statistics by model number.

        Args:
            limit: Maximum number of models to return

        Returns:
            List of model statistics dictionaries
        """
        with self.session() as session:
            # Single query with conditional aggregation including track-level stats
            model_stats = (
                session.query(
                    DBAnalysisResult.model,
                    func.count(func.distinct(DBAnalysisResult.id)).label('count'),
                    func.sum(
                        case(
                            (DBAnalysisResult.overall_status == DBStatusType.PASS, 1),
                            else_=0
                        )
                    ).label('passed'),
                    func.count(DBTrackResult.id).label('total_tracks'),
                    func.sum(case((DBTrackResult.sigma_pass == True, 1), else_=0)).label('sigma_passed'),
                    func.sum(case((DBTrackResult.linearity_pass == True, 1), else_=0)).label('linearity_passed'),
                )
                .outerjoin(DBTrackResult, DBAnalysisResult.id == DBTrackResult.analysis_id)
                .filter(DBAnalysisResult.model.isnot(None))
                .group_by(DBAnalysisResult.model)
                .order_by(func.count(func.distinct(DBAnalysisResult.id)).desc())
                .limit(limit)
                .all()
            )

            result = []
            for model, count, passed, total_tracks, sigma_passed, linearity_passed in model_stats:
                passed = passed or 0
                total_tracks = total_tracks or 0
                sigma_passed = sigma_passed or 0
                linearity_passed = linearity_passed or 0

                pass_rate = (passed / count * 100) if count > 0 else 0.0
                sigma_pass_rate = (sigma_passed / total_tracks * 100) if total_tracks > 0 else 0.0
                linearity_pass_rate = (linearity_passed / total_tracks * 100) if total_tracks > 0 else 0.0

                result.append({
                    "model": model,
                    "count": count,
                    "passed": passed,
                    "failed": count - passed,
                    "pass_rate": pass_rate,
                    "sigma_pass_rate": sigma_pass_rate,
                    "linearity_pass_rate": linearity_pass_rate,
                })

            return result

    def get_trend_data(
        self,
        model: Optional[str] = None,
        days_back: int = 30,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get trend data for analysis.

        Filters by trim date (file_date), not processing date.

        Args:
            model: Filter by model number (None for all)
            days_back: Number of days to include (based on trim date)
            limit: Maximum number of records

        Returns:
            List of trend data dictionaries sorted by trim date
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            query = (
                session.query(
                    DBAnalysisResult.file_date,  # Trim date from file
                    DBAnalysisResult.model,
                    DBTrackResult.sigma_gradient,
                    DBTrackResult.sigma_threshold,
                    DBTrackResult.sigma_pass,
                    DBTrackResult.status,
                    DBTrackResult.unit_length,
                    DBTrackResult.linearity_spec,
                )
                .join(DBTrackResult)
                .filter(DBAnalysisResult.file_date >= cutoff_date)
            )

            if model:
                query = query.filter(DBAnalysisResult.model == model)

            results = (
                query
                .order_by(DBAnalysisResult.file_date.asc())  # Order by trim date
                .limit(limit)
                .all()
            )

            return [
                {
                    "date": r.file_date.strftime("%Y-%m-%d") if r.file_date else "",
                    "model": r.model,
                    "sigma_gradient": r.sigma_gradient,
                    "sigma_threshold": r.sigma_threshold,
                    "sigma_pass": r.sigma_pass,
                    "status": r.status.value if r.status else "UNKNOWN",
                    "unit_length": r.unit_length,
                    "linearity_spec": r.linearity_spec,
                }
                for r in results
            ]

    def search_for_export(
        self,
        model: Optional[str] = None,
        serial: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 500
    ) -> List[AnalysisResult]:
        """
        Search for units matching criteria for export selection.

        Supports partial matching on serial number (case-insensitive).
        Filters by trim date (file_date).

        Args:
            model: Filter by model number (exact match, None for all)
            serial: Filter by serial number (partial match, case-insensitive)
            date_from: Start of date range (inclusive)
            date_to: End of date range (inclusive)
            limit: Maximum number of results

        Returns:
            List of AnalysisResult objects sorted by trim date (newest first)
        """
        with self.session() as session:
            query = session.query(DBAnalysisResult)

            # Filter by model (exact match)
            if model and model != "All Models":
                query = query.filter(DBAnalysisResult.model == model)

            # Filter by serial (partial match, case-insensitive)
            if serial and serial.strip():
                serial_pattern = f"%{serial.strip()}%"
                query = query.filter(
                    func.lower(DBAnalysisResult.serial).like(func.lower(serial_pattern))
                )

            # Filter by date range (trim date)
            if date_from:
                query = query.filter(DBAnalysisResult.file_date >= date_from)
            if date_to:
                # Include the entire end date (until midnight)
                end_of_day = date_to.replace(hour=23, minute=59, second=59)
                query = query.filter(DBAnalysisResult.file_date <= end_of_day)

            # Order by trim date (newest first) and limit
            query = query.order_by(desc(DBAnalysisResult.file_date)).limit(limit)

            results = []
            for db_analysis in query.all():
                mapped = self._map_db_to_analysis(db_analysis)
                if mapped is not None:
                    results.append(mapped)

            return results

    # =========================================================================
    # Private Mapping Methods
    # =========================================================================

    def _map_analysis_to_db(self, analysis: AnalysisResult) -> DBAnalysisResult:
        """Map Pydantic AnalysisResult to SQLAlchemy model."""
        # Map system type
        system_type = DBSystemType.A if analysis.metadata.system == SystemType.A else DBSystemType.B

        # Map overall status
        status_map = {
            AnalysisStatus.PASS: DBStatusType.PASS,
            AnalysisStatus.FAIL: DBStatusType.FAIL,
            AnalysisStatus.WARNING: DBStatusType.WARNING,
            AnalysisStatus.ERROR: DBStatusType.ERROR,
        }
        overall_status = status_map.get(analysis.overall_status, DBStatusType.ERROR)

        db_analysis = DBAnalysisResult(
            filename=analysis.metadata.filename,
            file_path=str(analysis.metadata.file_path),
            file_date=analysis.metadata.file_date,
            model=analysis.metadata.model,
            serial=analysis.metadata.serial,
            system=system_type,
            has_multi_tracks=analysis.metadata.has_multi_tracks,
            overall_status=overall_status,
            processing_time=analysis.processing_time,
            timestamp=datetime.now(),
            data_quality=getattr(analysis, 'data_quality', 'good'),
            data_quality_issues=','.join(getattr(analysis, 'data_quality_issues', [])) or None,
        )

        # Add track results
        for track in analysis.tracks:
            db_track = self._map_track_to_db(track)
            db_analysis.tracks.append(db_track)

        return db_analysis

    def _map_track_to_db(self, track: TrackData) -> DBTrackResult:
        """Map Pydantic TrackData to SQLAlchemy model."""
        status_map = {
            AnalysisStatus.PASS: DBStatusType.PASS,
            AnalysisStatus.FAIL: DBStatusType.FAIL,
            AnalysisStatus.WARNING: DBStatusType.WARNING,
            AnalysisStatus.ERROR: DBStatusType.ERROR,
        }
        status = status_map.get(track.status, DBStatusType.ERROR)

        risk_map = {
            RiskCategory.HIGH: DBRiskCategory.HIGH,
            RiskCategory.MEDIUM: DBRiskCategory.MEDIUM,
            RiskCategory.LOW: DBRiskCategory.LOW,
            RiskCategory.UNKNOWN: DBRiskCategory.UNKNOWN,
        }
        risk_category = risk_map.get(track.risk_category, DBRiskCategory.UNKNOWN)

        return DBTrackResult(
            track_id=track.track_id,
            status=status,
            travel_length=track.travel_length,
            linearity_spec=track.linearity_spec,
            sigma_gradient=track.sigma_gradient,
            sigma_threshold=track.sigma_threshold,
            sigma_pass=track.sigma_pass,
            unit_length=track.unit_length,
            untrimmed_resistance=track.untrimmed_resistance,
            trimmed_resistance=track.trimmed_resistance,
            measured_electrical_angle=track.measured_electrical_angle,
            optimal_offset=track.optimal_offset,
            final_linearity_error_shifted=track.linearity_error,
            linearity_pass=track.linearity_pass,
            linearity_fail_points=track.linearity_fail_points,
            failure_probability=track.failure_probability,
            risk_category=risk_category,
            is_anomaly=track.is_anomaly,  # Anomaly detection flag
            anomaly_reason=track.anomaly_reason,  # Reason for anomaly flag
            position_data=track.position_data,
            error_data=track.error_data,
            upper_limits=track.upper_limits,  # Store position-dependent spec limits
            lower_limits=track.lower_limits,  # Store position-dependent spec limits
            untrimmed_positions=track.untrimmed_positions,  # Store untrimmed data for charts
            untrimmed_errors=track.untrimmed_errors,  # Store untrimmed data for charts
            # Trim effectiveness metrics
            resistance_change=track.resistance_change,
            resistance_change_percent=track.resistance_change_percent,
            trim_improvement_percent=track.trim_improvement_percent,
            untrimmed_rms_error=track.untrimmed_rms_error,
            trimmed_rms_error=track.trimmed_rms_error,
            max_error_reduction_percent=track.max_error_reduction_percent,
            # Max deviation metrics
            max_deviation=getattr(track, 'max_deviation', None),
            max_deviation_position=getattr(track, 'max_deviation_position', None),
            deviation_uniformity=getattr(track, 'deviation_uniformity', None),
            # Failure margin metrics
            max_violation=getattr(track, 'max_violation', None),
            avg_violation=getattr(track, 'avg_violation', None),
            margin_to_spec=getattr(track, 'margin_to_spec', None),
            # Spec-aware optimization fields (Phase 2)
            optimal_slope=getattr(track, 'optimal_slope', 1.0),
            station_compensation=getattr(track, 'station_compensation', None),
            linearity_type=getattr(track, 'linearity_type', None),
            raw_linearity_error=getattr(track, 'raw_linearity_error', None),
            optimized_linearity_error=getattr(track, 'optimized_linearity_error', None),
            raw_fail_points=getattr(track, 'raw_fail_points', None),
            # Computed metrics
            gradient_margin=track.gradient_margin,
            plot_path=str(track.plot_path) if track.plot_path else None,
        )

    @staticmethod
    def _map_status_enum(db_status, default=None):
        """Map DB status to AnalysisStatus, handling both enum and string values.

        SQLAlchemy Enum columns may return the actual enum, the enum name ('PASS'),
        or the enum value ('Pass') depending on the database and driver.
        """
        if default is None:
            default = AnalysisStatus.ERROR

        status_map = {
            DBStatusType.PASS: AnalysisStatus.PASS,
            DBStatusType.FAIL: AnalysisStatus.FAIL,
            DBStatusType.WARNING: AnalysisStatus.WARNING,
            DBStatusType.ERROR: AnalysisStatus.ERROR,
        }

        if isinstance(db_status, DBStatusType):
            return status_map.get(db_status, default)
        elif isinstance(db_status, str):
            # Check enum names (PASS, FAIL) and values (Pass, Fail)
            for db_enum, analysis_enum in status_map.items():
                if db_status == db_enum.name or db_status == db_enum.value:
                    return analysis_enum
        return default

    def _map_db_to_analysis(self, db_analysis: DBAnalysisResult) -> Optional[AnalysisResult]:
        """Map SQLAlchemy model back to Pydantic AnalysisResult.

        Returns None if the analysis has no valid tracks (corrupted data).
        """
        from laser_trim_analyzer.core.models import FileMetadata

        # Map system type
        system_type = SystemType.A if db_analysis.system == DBSystemType.A else SystemType.B

        # Map status - handle both enum and string values from DB
        overall_status = self._map_status_enum(db_analysis.overall_status)

        # Create metadata
        metadata = FileMetadata(
            filename=db_analysis.filename,
            file_path=Path(db_analysis.file_path) if db_analysis.file_path else Path("."),
            file_date=db_analysis.file_date or datetime.now(),
            model=db_analysis.model,
            serial=db_analysis.serial,
            system=system_type,
            has_multi_tracks=db_analysis.has_multi_tracks,
        )

        # Map tracks - filter out None results from failed mappings
        tracks = [t for t in (self._map_db_to_track(t) for t in db_analysis.tracks) if t is not None]

        # If no valid tracks could be mapped, return None or create with ERROR status
        if not tracks and overall_status != AnalysisStatus.ERROR:
            logger.warning(f"Analysis {db_analysis.filename} has no valid tracks, skipping")
            return None

        try:
            return AnalysisResult(
                metadata=metadata,
                overall_status=overall_status,
                processing_time=db_analysis.processing_time or 0.0,
                tracks=tracks,
            )
        except Exception as e:
            logger.error(f"Failed to create AnalysisResult for {db_analysis.filename}: {e}")
            return None

    def _map_db_to_track(self, db_track: DBTrackResult) -> Optional[TrackData]:
        """Map SQLAlchemy TrackResult back to Pydantic TrackData.

        Returns None if required fields are missing (corrupted/incomplete data).
        """
        # Use robust status mapping that handles both enum and string values
        status = self._map_status_enum(db_track.status)

        risk_map = {
            DBRiskCategory.HIGH: RiskCategory.HIGH,
            DBRiskCategory.MEDIUM: RiskCategory.MEDIUM,
            DBRiskCategory.LOW: RiskCategory.LOW,
            DBRiskCategory.UNKNOWN: RiskCategory.UNKNOWN,
        }
        risk_category = risk_map.get(db_track.risk_category, RiskCategory.UNKNOWN)

        # Handle missing required fields - these are required for TrackData
        sigma_gradient = db_track.sigma_gradient
        sigma_threshold = db_track.sigma_threshold
        sigma_pass = db_track.sigma_pass

        # If any required sigma fields are None, provide defaults or skip
        if sigma_gradient is None:
            logger.warning(f"Track {db_track.track_id} has None sigma_gradient, using 0.0")
            sigma_gradient = 0.0
        if sigma_threshold is None:
            logger.warning(f"Track {db_track.track_id} has None sigma_threshold, using 0.01")
            sigma_threshold = 0.01
        if sigma_pass is None:
            # Calculate from gradient and threshold
            sigma_pass = sigma_gradient <= sigma_threshold

        try:
            return TrackData(
                track_id=db_track.track_id or "default",
                status=status,
                travel_length=db_track.travel_length or 1.0,  # Default to 1.0 to avoid 0
                linearity_spec=db_track.linearity_spec or 0.01,
                sigma_gradient=sigma_gradient,
                sigma_threshold=sigma_threshold,
                sigma_pass=sigma_pass,
                optimal_offset=db_track.optimal_offset or 0.0,
                linearity_error=abs(db_track.final_linearity_error_shifted or 0.0),
                linearity_pass=db_track.linearity_pass if db_track.linearity_pass is not None else True,
                linearity_fail_points=db_track.linearity_fail_points or 0,
                unit_length=db_track.unit_length,
                untrimmed_resistance=db_track.untrimmed_resistance,
                trimmed_resistance=db_track.trimmed_resistance,
                measured_electrical_angle=getattr(db_track, 'measured_electrical_angle', None),
                failure_probability=db_track.failure_probability,
                risk_category=risk_category,
                is_anomaly=db_track.is_anomaly or False,  # Retrieve anomaly flag
                anomaly_reason=db_track.anomaly_reason,  # Retrieve anomaly reason
                position_data=db_track.position_data,
                error_data=db_track.error_data,
                upper_limits=db_track.upper_limits,  # Retrieve position-dependent spec limits
                lower_limits=db_track.lower_limits,  # Retrieve position-dependent spec limits
                untrimmed_positions=db_track.untrimmed_positions,  # Retrieve untrimmed data for charts
                untrimmed_errors=db_track.untrimmed_errors,  # Retrieve untrimmed data for charts
                # Trim effectiveness metrics
                resistance_change=db_track.resistance_change,
                trim_improvement_percent=db_track.trim_improvement_percent,
                untrimmed_rms_error=db_track.untrimmed_rms_error,
                trimmed_rms_error=db_track.trimmed_rms_error,
                max_error_reduction_percent=db_track.max_error_reduction_percent,
                # Failure margin metrics
                max_violation=getattr(db_track, 'max_violation', None),
                avg_violation=getattr(db_track, 'avg_violation', None),
                margin_to_spec=getattr(db_track, 'margin_to_spec', None),
            )
        except Exception as e:
            logger.error(f"Failed to map track {db_track.track_id}: {e}")
            return None

    def _update_existing_analysis(
        self,
        session: Session,
        analysis: AnalysisResult
    ) -> int:
        """Update an existing analysis record."""
        # Find existing record by filename only (model/serial may have changed due to parsing fixes)
        existing = (
            session.query(DBAnalysisResult)
            .filter(DBAnalysisResult.filename == analysis.metadata.filename)
            .first()
        )

        if existing:
            # Update ALL fields including model/serial (parsing may have changed)
            existing.model = analysis.metadata.model
            existing.serial = analysis.metadata.serial
            existing.system = DBSystemType.A if analysis.metadata.system == SystemType.A else DBSystemType.B
            existing.file_date = analysis.metadata.file_date
            existing.has_multi_tracks = analysis.metadata.has_multi_tracks
            existing.processing_time = analysis.processing_time
            existing.timestamp = datetime.now()

            # Map overall status
            status_map = {
                AnalysisStatus.PASS: DBStatusType.PASS,
                AnalysisStatus.FAIL: DBStatusType.FAIL,
                AnalysisStatus.WARNING: DBStatusType.WARNING,
                AnalysisStatus.ERROR: DBStatusType.ERROR,
            }
            existing.overall_status = status_map.get(analysis.overall_status, DBStatusType.ERROR)

            # Delete old tracks explicitly and flush before adding new ones
            # This avoids unique constraint violations
            session.query(DBTrackResult).filter(
                DBTrackResult.analysis_id == existing.id
            ).delete()
            session.flush()

            # Add new tracks
            for track in analysis.tracks:
                db_track = self._map_track_to_db(track)
                db_track.analysis_id = existing.id
                session.add(db_track)

            # Flush to ensure changes are written
            session.flush()
            logger.info(f"Updated analysis ID {existing.id}: status={analysis.overall_status.value}")
            return existing.id

        # If no existing record found, create new
        db_analysis = self._map_analysis_to_db(analysis)
        session.add(db_analysis)
        session.flush()
        return db_analysis.id

    # =========================================================================
    # Delete Operations
    # =========================================================================

    def delete_analysis(self, analysis_id: int) -> bool:
        """
        Delete an analysis record and all associated data.

        This removes:
        - The analysis record
        - All associated track results
        - The processed file record (to allow re-processing)
        - Any related alerts

        Args:
            analysis_id: Database ID of the analysis to delete

        Returns:
            True if deletion was successful, False if record not found
        """
        with self.session() as session:
            # Find the analysis record
            analysis = session.query(DBAnalysisResult).get(analysis_id)

            if not analysis:
                logger.warning(f"Analysis ID {analysis_id} not found for deletion")
                return False

            filename = analysis.filename

            # Delete associated tracks (cascade should handle this, but be explicit)
            session.query(DBTrackResult).filter(
                DBTrackResult.analysis_id == analysis_id
            ).delete()

            # Delete processed file record (allows re-processing of same file)
            session.query(DBProcessedFile).filter(
                DBProcessedFile.analysis_id == analysis_id
            ).delete()

            # Delete any related alerts
            session.query(DBQAAlert).filter(
                DBQAAlert.analysis_id == analysis_id
            ).delete()

            # Delete the analysis record itself
            session.delete(analysis)

            logger.info(f"Deleted analysis ID {analysis_id}: {filename}")
            return True

    def delete_analysis_by_filename(self, filename: str) -> bool:
        """
        Delete an analysis record by filename.

        Args:
            filename: The filename of the analysis to delete

        Returns:
            True if deletion was successful, False if record not found
        """
        with self.session() as session:
            analysis = session.query(DBAnalysisResult).filter(
                DBAnalysisResult.filename == filename
            ).first()

            if not analysis:
                logger.warning(f"Analysis with filename '{filename}' not found for deletion")
                return False

            analysis_id = analysis.id

            # Delete associated tracks
            session.query(DBTrackResult).filter(
                DBTrackResult.analysis_id == analysis_id
            ).delete()

            # Delete processed file record
            session.query(DBProcessedFile).filter(
                DBProcessedFile.analysis_id == analysis_id
            ).delete()

            # Delete any related alerts
            session.query(DBQAAlert).filter(
                DBQAAlert.analysis_id == analysis_id
            ).delete()

            # Delete the analysis record itself
            session.delete(analysis)

            logger.info(f"Deleted analysis by filename: {filename}")
            return True

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def close(self) -> None:
        """Close the database connection."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connection closed")

    def get_database_path(self) -> Path:
        """Get the path to the database file."""
        return self.database_path

    def get_record_count(self) -> Dict[str, int]:
        """Get count of records in main tables."""
        with self.session() as session:
            return {
                "analyses": session.query(func.count(DBAnalysisResult.id)).scalar() or 0,
                "tracks": session.query(func.count(DBTrackResult.id)).scalar() or 0,
                "processed_files": session.query(func.count(DBProcessedFile.id)).scalar() or 0,
                "alerts": session.query(func.count(DBQAAlert.id)).scalar() or 0,
            }

    def get_models_list(self) -> List[str]:
        """
        Get a list of all unique model numbers in the database.

        Returns:
            List of model strings, sorted numerically
        """
        with self.session() as session:
            models = (
                session.query(DBAnalysisResult.model)
                .filter(DBAnalysisResult.model.isnot(None))
                .distinct()
                .all()
            )
            model_list = [m[0] for m in models if m[0] and m[0] != "Unknown"]
            return sorted(model_list, key=_model_sort_key)

    def get_models_list_prioritized(
        self,
        mps_models: List[str] = None,
        recent_days: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Get models sorted by priority: MPS first, then Recently Active, then Inactive.

        Each group is sorted numerically using _model_sort_key.

        Args:
            mps_models: List of models on MPS schedule (user-managed)
            recent_days: Days threshold for "recently active" (default 90)

        Returns:
            List of dicts: [{'model': str, 'status': 'mps'|'active'|'inactive', 'count': int, 'last_date': datetime}]
        """
        mps_models = mps_models or []
        mps_set = set(mps_models)
        cutoff_date = datetime.now() - timedelta(days=recent_days)

        with self.session() as session:
            # Get all models with their latest activity date and count
            results = (
                session.query(
                    DBAnalysisResult.model,
                    func.max(DBAnalysisResult.file_date).label('last_date'),
                    func.count(DBAnalysisResult.id).label('count')
                )
                .filter(DBAnalysisResult.model.isnot(None))
                .group_by(DBAnalysisResult.model)
                .all()
            )

            # Categorize models
            mps_list = []
            active_list = []
            inactive_list = []

            for model, last_date, count in results:
                if not model or model == "Unknown":
                    continue

                entry = {
                    'model': model,
                    'count': count,
                    'last_date': last_date
                }

                if model in mps_set:
                    entry['status'] = 'mps'
                    mps_list.append(entry)
                elif last_date and last_date >= cutoff_date:
                    entry['status'] = 'active'
                    active_list.append(entry)
                else:
                    entry['status'] = 'inactive'
                    inactive_list.append(entry)

            # Sort each group numerically
            mps_list.sort(key=lambda x: _model_sort_key(x['model']))
            active_list.sort(key=lambda x: _model_sort_key(x['model']))
            inactive_list.sort(key=lambda x: _model_sort_key(x['model']))

            return mps_list + active_list + inactive_list

    # =========================================================================
    # Trends Page Methods (Active Models Summary)
    # =========================================================================

    def get_active_models_summary(
        self,
        days_back: int = 90,
        min_samples: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get summary statistics for all models with recent activity.

        Filters by trim date (file_date), not processing date.

        Args:
            days_back: Only include models with files trimmed in this period
            min_samples: Minimum samples required for inclusion

        Returns:
            List of model summaries sorted by sample count descending
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Single query with join to get all data at once - no N+1 problem
            # Filter by trim date (file_date)
            model_data = (
                session.query(
                    DBAnalysisResult.model,
                    func.count(func.distinct(DBAnalysisResult.id)).label('total'),
                    func.sum(
                        case(
                            (DBAnalysisResult.overall_status == DBStatusType.PASS, 1),
                            else_=0
                        )
                    ).label('passed'),
                    func.min(DBAnalysisResult.file_date).label('first_date'),
                    func.max(DBAnalysisResult.file_date).label('last_date'),
                    func.avg(DBTrackResult.sigma_gradient).label('avg_sigma'),
                    func.avg(DBTrackResult.sigma_threshold).label('avg_threshold'),
                    func.count(DBTrackResult.id).label('total_tracks'),
                    func.sum(case((DBTrackResult.sigma_pass == True, 1), else_=0)).label('sigma_passed'),
                    func.sum(case((DBTrackResult.linearity_pass == True, 1), else_=0)).label('linearity_passed'),
                )
                .outerjoin(DBTrackResult, DBAnalysisResult.id == DBTrackResult.analysis_id)
                .filter(
                    DBAnalysisResult.model.isnot(None),
                    DBAnalysisResult.file_date >= cutoff_date
                )
                .group_by(DBAnalysisResult.model)
                .having(func.count(func.distinct(DBAnalysisResult.id)) >= min_samples)
                .all()
            )

            results = []
            for row in model_data:
                model = row.model
                total = row.total
                passed = row.passed or 0
                total_tracks = row.total_tracks or 0
                sigma_passed = row.sigma_passed or 0
                linearity_passed = row.linearity_passed or 0

                if not model or total == 0 or model == "Unknown":
                    continue

                pass_rate = (passed / total * 100)
                sigma_pass_rate = (sigma_passed / total_tracks * 100) if total_tracks > 0 else 0.0
                linearity_pass_rate = (linearity_passed / total_tracks * 100) if total_tracks > 0 else 0.0

                results.append({
                    "model": model,
                    "total": total,
                    "passed": passed,
                    "failed": total - passed,
                    "pass_rate": pass_rate,
                    "sigma_pass_rate": sigma_pass_rate,
                    "linearity_pass_rate": linearity_pass_rate,
                    "avg_sigma": row.avg_sigma or 0,
                    "avg_threshold": row.avg_threshold or 0,
                    "first_date": row.first_date,
                    "last_date": row.last_date,
                })

            # Sort by sample count descending (for chart display)
            results.sort(key=lambda x: x["total"], reverse=True)
            return results

    def get_models_requiring_attention(
        self,
        days_back: int = 90,
        min_samples: int = 5,
        pass_rate_threshold: float = 80.0,
        trend_threshold: float = 10.0,
        rolling_days: int = 30,
        metric: str = "linearity",
    ) -> List[Dict[str, Any]]:
        """
        Get models that require attention based on alert criteria.

        Alert criteria:
        - Pass rate below threshold (default 80%)
        - Trending worse by threshold% over rolling period
        - High variance in recent samples

        Args:
            days_back: Period to analyze
            min_samples: Minimum samples for inclusion
            pass_rate_threshold: Alert if pass rate below this
            trend_threshold: Alert if trend worse by this %
            rolling_days: Rolling window for trend calculation
            metric: Which pass rate to use - "linearity", "sigma", or "overall"

        Returns:
            List of models requiring attention with alert details
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            rolling_cutoff = datetime.now() - timedelta(days=rolling_days)

            # Get active models
            active_models = self.get_active_models_summary(days_back, min_samples)

            # Select which rate to evaluate based on metric
            rate_key = {
                "linearity": "linearity_pass_rate",
                "sigma": "sigma_pass_rate",
                "overall": "pass_rate",
            }.get(metric, "linearity_pass_rate")
            rate_label = {
                "linearity": "Linearity",
                "sigma": "Sigma",
                "overall": "Overall",
            }.get(metric, "Linearity")

            alerts = []
            for model_data in active_models:
                model = model_data["model"]
                alert_reasons = []

                # Check 1: Low pass rate based on selected metric
                check_rate = model_data.get(rate_key, model_data["pass_rate"])
                if check_rate < pass_rate_threshold:
                    alert_reasons.append({
                        "type": "LOW_PASS_RATE",
                        "message": f"{rate_label} pass rate {check_rate:.1f}% is below {pass_rate_threshold}%",
                        "severity": "High" if check_rate < 70 else "Medium"
                    })

                # Check 2: Trending worse
                # Compare older period vs rolling period
                older_cutoff = cutoff_date
                older_end = rolling_cutoff

                # Use track-level linearity query when metric is "linearity"
                if metric == "linearity":
                    older_trend = (
                        session.query(
                            func.count(DBTrackResult.id),
                            func.sum(case((DBTrackResult.linearity_pass == True, 1), else_=0))
                        )
                        .join(DBAnalysisResult)
                        .filter(
                            DBAnalysisResult.model == model,
                            DBAnalysisResult.file_date >= older_cutoff,
                            DBAnalysisResult.file_date < older_end
                        )
                        .first()
                    )
                    recent_trend = (
                        session.query(
                            func.count(DBTrackResult.id),
                            func.sum(case((DBTrackResult.linearity_pass == True, 1), else_=0))
                        )
                        .join(DBAnalysisResult)
                        .filter(
                            DBAnalysisResult.model == model,
                            DBAnalysisResult.file_date >= rolling_cutoff
                        )
                        .first()
                    )
                else:
                    older_trend = (
                        session.query(
                            func.count(DBAnalysisResult.id),
                            func.sum(
                                case(
                                    (DBAnalysisResult.overall_status == DBStatusType.PASS, 1),
                                    else_=0
                                )
                            )
                        )
                        .filter(
                            DBAnalysisResult.model == model,
                            DBAnalysisResult.file_date >= older_cutoff,
                            DBAnalysisResult.file_date < older_end
                        )
                        .first()
                    )
                    recent_trend = (
                        session.query(
                            func.count(DBAnalysisResult.id),
                            func.sum(
                                case(
                                    (DBAnalysisResult.overall_status == DBStatusType.PASS, 1),
                                    else_=0
                                )
                            )
                        )
                        .filter(
                            DBAnalysisResult.model == model,
                            DBAnalysisResult.file_date >= rolling_cutoff
                        )
                        .first()
                    )

                older_count, older_passed = older_trend
                recent_count, recent_passed = recent_trend

                if older_count and older_count >= min_samples and recent_count and recent_count >= min_samples:
                    older_pct = (older_passed or 0) / older_count * 100
                    recent_pct = (recent_passed or 0) / recent_count * 100

                    if recent_pct < older_pct - trend_threshold:
                        alert_reasons.append({
                            "type": "TRENDING_WORSE",
                            "message": f"{rate_label} pass rate dropped from {older_pct:.1f}% to {recent_pct:.1f}% ({older_pct - recent_pct:.1f}% decline)",
                            "severity": "High" if (older_pct - recent_pct) > 20 else "Medium"
                        })

                # Check 3: High variance (use coefficient of variation)
                sigma_values = (
                    session.query(DBTrackResult.sigma_gradient)
                    .join(DBAnalysisResult)
                    .filter(
                        DBAnalysisResult.model == model,
                        DBAnalysisResult.file_date >= rolling_cutoff,
                        DBTrackResult.sigma_gradient.isnot(None),
                    )
                    .all()
                )

                if len(sigma_values) >= min_samples:
                    values = [v[0] for v in sigma_values if v[0] is not None]
                    if values:
                        import numpy as np
                        mean_val = np.mean(values)
                        std_val = np.std(values, ddof=1)
                        cv = (std_val / mean_val * 100) if mean_val > 0 else 0

                        # CV > 50% is high variance for sigma gradient
                        if cv > 50:
                            alert_reasons.append({
                                "type": "HIGH_VARIANCE",
                                "message": f"High variance in sigma gradient (CV={cv:.1f}%)",
                                "severity": "Medium"
                            })

                if alert_reasons:
                    alerts.append({
                        "model": model,
                        "pass_rate": model_data["pass_rate"],
                        "total_samples": model_data["total"],
                        "alerts": alert_reasons,
                        "severity": max(a["severity"] for a in alert_reasons)
                    })

            # Sort by severity (High first) then by pass rate (lowest first)
            severity_order = {"High": 0, "Medium": 1, "Low": 2}
            alerts.sort(key=lambda x: (severity_order.get(x["severity"], 3), x["pass_rate"]))

            return alerts

    def get_linearity_prioritization(
        self,
        days_back: int = 90,
        min_samples: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get models ranked by improvement impact for linearity.

        Combines failure volume, near-miss count (easy wins), and trend
        to help users prioritize where to spend time.

        Args:
            days_back: Period to analyze
            min_samples: Minimum track count for inclusion

        Returns:
            List of models sorted by impact_score descending
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            model_data = (
                session.query(
                    DBAnalysisResult.model,
                    func.count(DBTrackResult.id).label('total_tracks'),
                    func.sum(case((DBTrackResult.linearity_pass == True, 1), else_=0)).label('lin_passed'),
                    func.sum(case((DBTrackResult.linearity_pass == False, 1), else_=0)).label('lin_failed'),
                    func.sum(case((DBTrackResult.sigma_pass == True, 1), else_=0)).label('sigma_passed'),
                    # Near-miss: failed linearity with only 1-2 fail points (easy wins)
                    func.sum(
                        case(
                            (and_(
                                DBTrackResult.linearity_pass == False,
                                DBTrackResult.linearity_fail_points <= 2,
                                DBTrackResult.linearity_fail_points > 0,
                            ), 1),
                            else_=0
                        )
                    ).label('near_miss_count'),
                    # Average fail points on failing tracks
                    func.avg(
                        case(
                            (DBTrackResult.linearity_pass == False, DBTrackResult.linearity_fail_points),
                            else_=None
                        )
                    ).label('avg_fail_points'),
                    # Trim effectiveness (may be NULL if not yet calculated)
                    func.avg(DBTrackResult.trim_improvement_percent).label('avg_trim_improvement'),
                    func.avg(DBTrackResult.resistance_change_percent).label('avg_resistance_change'),
                )
                .join(DBTrackResult, DBAnalysisResult.id == DBTrackResult.analysis_id)
                .filter(
                    DBAnalysisResult.model.isnot(None),
                    DBAnalysisResult.model != "Unknown",
                    DBAnalysisResult.file_date >= cutoff_date,
                )
                .group_by(DBAnalysisResult.model)
                .having(func.count(DBTrackResult.id) >= min_samples)
                .all()
            )

            results = []
            for row in model_data:
                total = row.total_tracks or 0
                lin_passed = row.lin_passed or 0
                lin_failed = row.lin_failed or 0
                sigma_passed = row.sigma_passed or 0
                near_miss = row.near_miss_count or 0

                if total == 0:
                    continue

                lin_rate = lin_passed / total * 100
                sigma_rate = sigma_passed / total * 100
                avg_fps = row.avg_fail_points or 0

                # Impact score: volume of failures + near-miss opportunity
                impact_score = (
                    lin_failed * 0.5
                    + near_miss * 0.3
                    + (1 - lin_rate / 100) * total * 0.2
                )

                # Generate recommendation
                recommendation = self._generate_recommendation(
                    near_miss, lin_failed, avg_fps,
                    row.avg_trim_improvement, row.avg_resistance_change,
                    lin_rate, sigma_rate
                )

                results.append({
                    "model": row.model,
                    "total_tracks": total,
                    "linearity_pass_rate": round(lin_rate, 1),
                    "sigma_pass_rate": round(sigma_rate, 1),
                    "failed_units": lin_failed,
                    "near_miss_count": near_miss,
                    "avg_fail_points": round(avg_fps, 1),
                    "avg_trim_improvement": round(row.avg_trim_improvement, 1) if row.avg_trim_improvement else None,
                    "avg_resistance_change": round(row.avg_resistance_change, 2) if row.avg_resistance_change else None,
                    "impact_score": round(impact_score, 1),
                    "recommendation": recommendation,
                })

            # Compute percentile ranks
            if results:
                sorted_by_rate = sorted(results, key=lambda x: x["linearity_pass_rate"])
                for i, r in enumerate(sorted_by_rate):
                    r["percentile_rank"] = round(i / len(sorted_by_rate) * 100, 0)

            # Sort by impact score descending
            results.sort(key=lambda x: x["impact_score"], reverse=True)
            return results

    def _generate_recommendation(
        self,
        near_miss: int,
        lin_failed: int,
        avg_fail_points: float,
        avg_trim_improvement: Optional[float],
        avg_resistance_change: Optional[float],
        lin_rate: float,
        sigma_rate: float,
    ) -> str:
        """Generate actionable recommendation based on model data."""
        if lin_failed == 0:
            return "Passing — monitor sigma trends"

        # Check near-miss ratio (easy wins)
        near_miss_ratio = near_miss / lin_failed if lin_failed > 0 else 0
        if near_miss_ratio > 0.3:
            return f"Easy-win potential — {near_miss} units within 2 fail points of passing"

        # Check if trim is not effective
        if avg_trim_improvement is not None and avg_trim_improvement < 30:
            return "Trim not effective — investigate incoming material quality"

        # Check excessive trimming
        if avg_resistance_change is not None and abs(avg_resistance_change) > 15:
            return "Excessive trimming — process may be over-cutting"

        # Check severity
        if avg_fail_points > 10:
            return "Severe failures — units far from spec, may need process change"

        # Sigma OK but linearity bad
        if sigma_rate > 80 and lin_rate < 70:
            return "Sigma healthy but linearity failing — check spec limits"

        return f"Review needed — {lin_failed} linearity failures"

    def get_linearity_margin_analysis(
        self,
        model: str,
        days_back: int = 90,
    ) -> Dict[str, Any]:
        """
        Get fail-point distribution for a specific model.

        Shows how close failing tracks are to passing — highlights easy wins.

        Args:
            model: Model number
            days_back: Period to analyze

        Returns:
            Dict with fail point distribution and easy-win analysis
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Get all failing tracks for this model
            failing_tracks = (
                session.query(DBTrackResult.linearity_fail_points)
                .join(DBAnalysisResult)
                .filter(
                    DBAnalysisResult.model == model,
                    DBAnalysisResult.file_date >= cutoff_date,
                    DBTrackResult.linearity_pass == False,
                    DBTrackResult.linearity_fail_points > 0,
                )
                .all()
            )

            total_tracks = (
                session.query(func.count(DBTrackResult.id))
                .join(DBAnalysisResult)
                .filter(
                    DBAnalysisResult.model == model,
                    DBAnalysisResult.file_date >= cutoff_date,
                )
                .scalar()
            ) or 0

            passing_tracks = (
                session.query(func.count(DBTrackResult.id))
                .join(DBAnalysisResult)
                .filter(
                    DBAnalysisResult.model == model,
                    DBAnalysisResult.file_date >= cutoff_date,
                    DBTrackResult.linearity_pass == True,
                )
                .scalar()
            ) or 0

            # Build distribution
            fail_points = [row[0] for row in failing_tracks]
            distribution = {
                "1_point": sum(1 for fp in fail_points if fp == 1),
                "2_points": sum(1 for fp in fail_points if fp == 2),
                "3_to_5": sum(1 for fp in fail_points if 3 <= fp <= 5),
                "6_to_10": sum(1 for fp in fail_points if 6 <= fp <= 10),
                "over_10": sum(1 for fp in fail_points if fp > 10),
            }

            easy_wins = distribution["1_point"] + distribution["2_points"]
            failing_count = len(fail_points)

            return {
                "model": model,
                "total_tracks": total_tracks,
                "passing_tracks": passing_tracks,
                "failing_tracks": failing_count,
                "fail_point_distribution": distribution,
                "easy_win_count": easy_wins,
                "easy_win_percent": round(easy_wins / failing_count * 100, 1) if failing_count > 0 else 0,
            }

    def get_near_miss_summary(self, days_back: int = 90) -> Dict[str, Any]:
        """
        Get overall near-miss analysis across all models.

        Returns:
            Dict with fail-point distribution, near-miss percentage,
            and top models by near-miss count.
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Get all failing tracks with fail point counts
            failing = (
                session.query(
                    DBTrackResult.linearity_fail_points,
                    DBAnalysisResult.model,
                )
                .join(DBAnalysisResult)
                .filter(
                    DBAnalysisResult.file_date >= cutoff_date,
                    DBTrackResult.linearity_pass == False,
                    DBTrackResult.linearity_fail_points > 0,
                )
                .all()
            )

            total_failing = len(failing)
            if total_failing == 0:
                return {
                    "total_failing": 0,
                    "distribution": {},
                    "near_miss_count": 0,
                    "near_miss_percent": 0,
                    "hard_fail_count": 0,
                    "hard_fail_percent": 0,
                    "top_near_miss_models": [],
                }

            # Distribution buckets
            buckets = {"1-3 points": 0, "4-10 points": 0, "11-50 points": 0, "50+ points": 0}
            near_miss_by_model = {}

            for fp, model in failing:
                if fp <= 3:
                    buckets["1-3 points"] += 1
                    near_miss_by_model[model] = near_miss_by_model.get(model, 0) + 1
                elif fp <= 10:
                    buckets["4-10 points"] += 1
                elif fp <= 50:
                    buckets["11-50 points"] += 1
                else:
                    buckets["50+ points"] += 1

            near_miss = buckets["1-3 points"]
            hard_fail = buckets["11-50 points"] + buckets["50+ points"]

            # Top models by near-miss count
            top_models = sorted(
                near_miss_by_model.items(), key=lambda x: x[1], reverse=True
            )[:10]

            return {
                "total_failing": total_failing,
                "distribution": buckets,
                "near_miss_count": near_miss,
                "near_miss_percent": round(near_miss / total_failing * 100, 1),
                "hard_fail_count": hard_fail,
                "hard_fail_percent": round(hard_fail / total_failing * 100, 1),
                "top_near_miss_models": [
                    {"model": m, "near_miss_count": c} for m, c in top_models
                ],
            }

    def get_trending_worse_models(
        self,
        days_back: int = 90,
        min_samples: int = 20,
        trend_threshold: float = 10.0,
        rolling_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get models whose pass rate is declining.

        Compares older period vs recent rolling period pass rates.

        Args:
            days_back: Total period to analyze
            min_samples: Minimum samples required for both periods
            trend_threshold: Minimum decline % to be considered "trending worse"
            rolling_days: Recent period for comparison

        Returns:
            List of models with decline info, sorted by decline magnitude (biggest first)
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            rolling_cutoff = datetime.now() - timedelta(days=rolling_days)

            # Get active models with sufficient data
            active_models = self.get_active_models_summary(days_back, min_samples)

            trending_worse = []
            for model_data in active_models:
                model = model_data["model"]

                # Skip if not enough total samples
                if model_data["total"] < min_samples:
                    continue

                # Get older period pass rate (from cutoff to rolling_cutoff)
                older_result = (
                    session.query(
                        func.count(DBAnalysisResult.id),
                        func.sum(
                            case(
                                (DBAnalysisResult.overall_status == DBStatusType.PASS, 1),
                                else_=0
                            )
                        )
                    )
                    .filter(
                        DBAnalysisResult.model == model,
                        DBAnalysisResult.file_date >= cutoff_date,
                        DBAnalysisResult.file_date < rolling_cutoff
                    )
                    .first()
                )

                # Get recent period pass rate (from rolling_cutoff to now)
                recent_result = (
                    session.query(
                        func.count(DBAnalysisResult.id),
                        func.sum(
                            case(
                                (DBAnalysisResult.overall_status == DBStatusType.PASS, 1),
                                else_=0
                            )
                        )
                    )
                    .filter(
                        DBAnalysisResult.model == model,
                        DBAnalysisResult.file_date >= rolling_cutoff
                    )
                    .first()
                )

                older_count, older_passed = older_result
                recent_count, recent_passed = recent_result

                # Need sufficient samples in both periods
                if not older_count or older_count < 5:
                    continue
                if not recent_count or recent_count < 5:
                    continue

                older_pct = (older_passed or 0) / older_count * 100
                recent_pct = (recent_passed or 0) / recent_count * 100
                decline = older_pct - recent_pct

                # Only include if declining by threshold
                if decline >= trend_threshold:
                    trending_worse.append({
                        "model": model,
                        "pass_rate": model_data["pass_rate"],
                        "older_pass_rate": older_pct,
                        "recent_pass_rate": recent_pct,
                        "decline": decline,
                        "total_samples": model_data["total"],
                        "older_samples": older_count,
                        "recent_samples": recent_count,
                    })

            # Sort by decline magnitude (biggest drop first)
            trending_worse.sort(key=lambda x: x["decline"], reverse=True)

            return trending_worse

    def get_model_trend_data(
        self,
        model: str,
        days_back: int = 90,
        rolling_window: int = 30
    ) -> Dict[str, Any]:
        """
        Get detailed trend data for a specific model.

        Filters by trim date (file_date), not processing date.

        Args:
            model: Model number
            days_back: Total days to include (based on trim date)
            rolling_window: Days for rolling average

        Returns:
            Dict with trend data for charts
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Get all track results for this model (including anomaly flag)
            # Filter by trim date (file_date)
            results = (
                session.query(
                    DBAnalysisResult.file_date,
                    DBAnalysisResult.overall_status,
                    DBTrackResult.sigma_gradient,
                    DBTrackResult.sigma_threshold,
                    DBTrackResult.sigma_pass,
                    DBTrackResult.is_anomaly,
                    DBTrackResult.final_linearity_error_shifted,
                    DBTrackResult.linearity_spec,
                    DBTrackResult.linearity_pass,
                    DBTrackResult.linearity_fail_points,
                )
                .join(DBTrackResult)
                .filter(
                    DBAnalysisResult.model == model,
                    DBAnalysisResult.file_date >= cutoff_date,
                )
                .order_by(DBAnalysisResult.file_date.asc())
                .all()
            )

            if not results:
                return {
                    "model": model,
                    "data_points": [],
                    "rolling_averages": [],
                    "pass_rates_by_day": [],
                    "threshold": None,
                }

            # Extract data points
            data_points = []
            for file_date, status, sigma_gradient, sigma_threshold, sigma_pass, is_anomaly, \
                linearity_error, linearity_spec, linearity_pass, fail_points in results:
                if file_date and sigma_gradient is not None:
                    data_points.append({
                        "date": file_date,
                        "sigma_gradient": sigma_gradient,
                        "sigma_threshold": sigma_threshold,
                        "sigma_pass": sigma_pass,
                        "status": status.value if status else "UNKNOWN",
                        "is_anomaly": is_anomaly or False,
                        "linearity_error": linearity_error,
                        "linearity_spec": linearity_spec,
                        "linearity_pass": linearity_pass,
                        "fail_points": fail_points or 0,
                    })

            # Calculate threshold (use mode of thresholds)
            thresholds = [d["sigma_threshold"] for d in data_points if d["sigma_threshold"]]
            threshold = max(set(thresholds), key=thresholds.count) if thresholds else None

            # Calculate linearity spec (use mode of specs)
            linearity_specs = [d["linearity_spec"] for d in data_points if d["linearity_spec"]]
            linearity_spec = max(set(linearity_specs), key=linearity_specs.count) if linearity_specs else None

            # Calculate daily pass rates for rolling average
            from collections import defaultdict
            daily_data = defaultdict(lambda: {"passed": 0, "total": 0})

            for dp in data_points:
                day_key = dp["date"].strftime("%Y-%m-%d")
                daily_data[day_key]["total"] += 1
                if dp["sigma_pass"]:
                    daily_data[day_key]["passed"] += 1

            # Sort by date and calculate pass rates
            sorted_days = sorted(daily_data.keys())
            pass_rates_by_day = []
            for day in sorted_days:
                d = daily_data[day]
                pass_rates_by_day.append({
                    "date": day,
                    "pass_rate": (d["passed"] / d["total"] * 100) if d["total"] > 0 else 0,
                    "total": d["total"],
                })

            # Calculate daily LINEARITY pass rates
            linearity_daily_data = defaultdict(lambda: {"passed": 0, "total": 0})

            for dp in data_points:
                if dp["linearity_pass"] is not None:  # Only count units with linearity data
                    day_key = dp["date"].strftime("%Y-%m-%d")
                    linearity_daily_data[day_key]["total"] += 1
                    if dp["linearity_pass"]:
                        linearity_daily_data[day_key]["passed"] += 1

            # Sort by date and calculate linearity pass rates
            sorted_lin_days = sorted(linearity_daily_data.keys())
            linearity_pass_rates_by_day = []
            for day in sorted_lin_days:
                d = linearity_daily_data[day]
                linearity_pass_rates_by_day.append({
                    "date": day,
                    "pass_rate": (d["passed"] / d["total"] * 100) if d["total"] > 0 else 0,
                    "total": d["total"],
                })

            # Calculate rolling averages
            rolling_averages = []
            if len(pass_rates_by_day) >= 2:
                # Use the specified rolling window
                window_size = min(rolling_window, len(pass_rates_by_day))
                for i in range(len(pass_rates_by_day)):
                    start_idx = max(0, i - window_size + 1)
                    window = pass_rates_by_day[start_idx:i + 1]

                    total_passed = sum(d["pass_rate"] * d["total"] for d in window)
                    total_count = sum(d["total"] for d in window)
                    rolling_avg = total_passed / total_count if total_count > 0 else 0

                    rolling_averages.append({
                        "date": pass_rates_by_day[i]["date"],
                        "rolling_avg": rolling_avg,
                        "window_size": len(window),
                    })

            return {
                "model": model,
                "data_points": data_points,
                "rolling_averages": rolling_averages,
                "pass_rates_by_day": pass_rates_by_day,
                "linearity_pass_rates_by_day": linearity_pass_rates_by_day,
                "threshold": threshold,
                "linearity_spec": linearity_spec,
                "total_samples": len(data_points),
            }

    # =========================================================================
    # Final Test Methods - For post-assembly test data and comparison
    # =========================================================================

    def save_final_test(
        self,
        metadata: Dict[str, Any],
        tracks: List[Dict[str, Any]],
        test_results: Dict[str, Any],
        file_hash: str
    ) -> int:
        """
        Save a Final Test result to the database.

        Args:
            metadata: Dict with filename, model, serial, test_date, etc.
            tracks: List of track data dicts with positions, errors, etc.
            test_results: Dict with pass/fail for each test type
            file_hash: SHA256 hash of the file

        Returns:
            ID of saved FinalTestResult
        """
        from sqlalchemy.exc import IntegrityError
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
            FinalTestTrack as DBFinalTestTrack,
        )

        # Use lock to prevent race conditions with SQLite
        with self._write_lock:
            try:
                with self.session() as session:
                    # Check for duplicate by file_hash
                    existing = (
                        session.query(DBFinalTestResult)
                        .filter(DBFinalTestResult.file_hash == file_hash)
                        .first()
                    )
                    if existing:
                        logger.info(f"Final test already exists: {metadata.get('filename')}")
                        return existing.id

                    # Determine overall status from linearity
                    overall_status = DBStatusType.PASS
                    if test_results.get("linearity_pass") is False:
                        overall_status = DBStatusType.FAIL
                    elif test_results.get("linearity_pass") is None:
                        # Check track-level linearity
                        for track in tracks:
                            if track.get("linearity_pass") is False:
                                overall_status = DBStatusType.FAIL
                                break

                    # Find matching trim result
                    linked_trim_id, match_confidence, days_since_trim, match_method = self._find_matching_trim(
                        session,
                        metadata.get("model"),
                        metadata.get("serial"),
                        metadata.get("file_date") or metadata.get("test_date")
                    )

                    # Create FinalTestResult
                    db_result = DBFinalTestResult(
                        filename=metadata.get("filename", "unknown"),
                        file_path=str(metadata.get("file_path", "")),
                        file_hash=file_hash,
                        file_date=metadata.get("file_date"),
                        model=metadata.get("model", "unknown"),
                        serial=metadata.get("serial", "unknown"),
                        test_date=metadata.get("test_date"),
                        overall_status=overall_status,
                        linearity_pass=test_results.get("linearity_pass"),
                        linearity_error=tracks[0].get("linearity_error") if tracks else None,
                        resistance_pass=test_results.get("resistance_pass"),
                        resistance_value=test_results.get("resistance_value"),
                        resistance_tolerance=test_results.get("resistance_tolerance"),
                        electrical_angle_pass=test_results.get("electrical_angle_pass"),
                        hysteresis_pass=test_results.get("hysteresis_pass"),
                        phasing_pass=test_results.get("phasing_pass"),
                        linked_trim_id=linked_trim_id,
                        match_confidence=match_confidence,
                        days_since_trim=days_since_trim,
                        match_method=match_method,
                    )

                    session.add(db_result)
                    session.flush()
                    result_id = db_result.id

                    # Add tracks
                    for track_data in tracks:
                        # Use electrical_angles as position_data (X-axis for charts)
                        # electrical_angles contains: inches for linear pots, degrees for rotary
                        position_values = track_data.get("electrical_angles") or track_data.get("positions")

                        db_track = DBFinalTestTrack(
                            final_test_id=result_id,
                            track_id=track_data.get("track_id", "default"),
                            status=DBStatusType.PASS if track_data.get("linearity_pass", True) else DBStatusType.FAIL,
                            linearity_spec=track_data.get("linearity_spec"),
                            linearity_error=track_data.get("linearity_error"),
                            linearity_pass=track_data.get("linearity_pass"),
                            linearity_fail_points=track_data.get("linearity_fail_points", 0),
                            position_data=position_values,
                            error_data=track_data.get("errors"),
                            electrical_angle_data=track_data.get("electrical_angles"),
                            upper_limits=track_data.get("upper_limits"),
                            lower_limits=track_data.get("lower_limits"),
                            max_deviation=track_data.get("max_deviation"),
                            max_deviation_position=track_data.get("max_deviation_angle"),
                        )
                        session.add(db_track)

                    session.commit()
                    logger.info(f"Saved Final Test: {metadata.get('filename')} (ID: {result_id}, linked_trim: {linked_trim_id})")
                    return result_id

            except IntegrityError as e:
                # Handle race condition - another thread inserted first
                logger.warning(f"Final test duplicate detected (race condition): {metadata.get('filename')}")
                # Try to find the existing record
                try:
                    with self.session() as session:
                        existing = (
                            session.query(DBFinalTestResult)
                            .filter(DBFinalTestResult.file_hash == file_hash)
                            .first()
                        )
                        if existing:
                            return existing.id
                except Exception:
                    pass
                raise

    def get_ml_staleness(self) -> List[Dict[str, Any]]:
        """
        Get ML training staleness info for each trained model.

        Compares training_samples at training time vs current record count.
        Models with 50+ new records since training are flagged for retrain.

        Returns:
            List of dicts: model, training_date, training_samples,
            current_samples, new_since_training, needs_retrain, days_since_training
        """
        from laser_trim_analyzer.database.models import ModelMLState as DBModelMLState

        results = []
        with self.session() as session:
            # Get all trained models
            ml_states = session.query(DBModelMLState).filter(
                DBModelMLState.is_trained == True
            ).all()

            for state in ml_states:
                current_count = session.query(
                    func.count(DBAnalysisResult.id)
                ).filter(
                    DBAnalysisResult.model == state.model
                ).scalar() or 0

                training_samples = state.training_samples or 0
                new_records = max(0, current_count - training_samples)
                days_since = (datetime.now() - state.training_date).days if state.training_date else 999

                results.append({
                    "model": state.model,
                    "training_date": state.training_date,
                    "training_samples": training_samples,
                    "current_samples": current_count,
                    "new_since_training": new_records,
                    "needs_retrain": new_records >= 50,
                    "days_since_training": days_since,
                })

        # Sort by most stale first
        results.sort(key=lambda x: x["new_since_training"], reverse=True)
        return results

    def get_screening_recommendations(self, days_back: int = 90, min_samples: int = 20) -> List[Dict[str, Any]]:
        """
        Generate element screening recommendations based on failure patterns.

        Flags:
        - Near-miss rate >40%: candidate for in-process testing
        - Failure rate >50%: candidate for design review
        - High volume + high failure: candidate for incoming inspection

        Returns:
            List of recommendation dicts sorted by priority
        """
        recommendations = []

        try:
            priority_models = self.get_linearity_prioritization(
                days_back=days_back, min_samples=min_samples
            )
        except Exception:
            return recommendations

        for m in priority_models:
            model = m.get("model", "Unknown")
            lin_rate = m.get("linearity_pass_rate", 100)
            fail_rate = 100 - lin_rate
            failed = m.get("failed_units", 0)
            total = m.get("total_tracks", 0)
            near_miss = m.get("near_miss_count", 0)
            near_miss_ratio = near_miss / failed if failed > 0 else 0

            rec = {
                "model": model,
                "fail_rate": fail_rate,
                "total_units": total,
                "failed_units": failed,
                "near_miss_count": near_miss,
                "recommendations": [],
            }

            if fail_rate > 50:
                rec["recommendations"].append({
                    "type": "design_review",
                    "text": f"Design review — {fail_rate:.0f}% failure rate",
                    "priority": "high",
                })

            if near_miss_ratio > 0.4 and near_miss >= 3:
                rec["recommendations"].append({
                    "type": "in_process_testing",
                    "text": f"In-process testing candidate — {near_miss} near-miss ({near_miss_ratio:.0%} of failures)",
                    "priority": "medium",
                })

            if failed >= 10 and fail_rate > 30:
                rec["recommendations"].append({
                    "type": "incoming_inspection",
                    "text": f"Incoming element inspection — {failed} failures, high volume",
                    "priority": "medium",
                })

            if rec["recommendations"]:
                recommendations.append(rec)

        return recommendations

    def get_model_cpk(self, model: str, days_back: int = 90) -> Dict[str, Any]:
        """
        Calculate process capability (Cpk) for a model.

        Cpk = min(USL - mean, mean - LSL) / (3 * sigma)

        For sigma gradient: LSL=0, USL=threshold (one-sided: Cpk = (USL - mean) / (3*std))
        For linearity: based on pass rate as a capability proxy

        Args:
            model: Model number
            days_back: Period to analyze

        Returns:
            Dict with sigma_cpk, sigma_cpk_color, sample_count
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Get sigma gradient data for this model
            data = (
                session.query(
                    DBTrackResult.sigma_gradient,
                    DBTrackResult.sigma_threshold,
                )
                .join(DBAnalysisResult)
                .filter(
                    DBAnalysisResult.model == model,
                    DBAnalysisResult.file_date >= cutoff_date,
                    DBTrackResult.sigma_gradient.isnot(None),
                    DBTrackResult.sigma_threshold.isnot(None),
                )
                .all()
            )

            result = {
                "model": model,
                "sample_count": len(data),
                "sigma_cpk": None,
                "sigma_cpk_color": "gray",
            }

            if len(data) < 10:
                return result

            gradients = [float(d[0]) for d in data]
            threshold = float(data[0][1])  # Threshold is same for all tracks of a model

            import numpy as np
            mean = np.mean(gradients)
            std = np.std(gradients, ddof=1)  # Sample std dev

            if std > 0 and threshold > 0:
                # One-sided Cpk: process must stay BELOW threshold
                # Cpk = (USL - mean) / (3 * sigma)
                cpk = (threshold - mean) / (3 * std)
                result["sigma_cpk"] = round(cpk, 2)

                if cpk < 1.0:
                    result["sigma_cpk_color"] = "#e74c3c"  # Red — incapable
                elif cpk < 1.33:
                    result["sigma_cpk_color"] = "#f39c12"  # Yellow — marginal
                else:
                    result["sigma_cpk_color"] = "#27ae60"  # Green — capable

            return result

    @staticmethod
    def _normalize_serial(serial: str) -> str:
        """
        Normalize a serial number for fuzzy matching (selective).

        Handles common formatting differences between trim and FT files:
        - Strip leading zeros (007 -> 7)
        - Lowercase
        - Strip whitespace
        - Remove common prefixes (sn, s/n, #)
        - Strip known track-position suffixes only (A/B for dual-track,
          P/R for primary/redundant, T for test)
        - Do NOT strip other letters (25D, 31L stay as-is since they may
          be meaningful serial identifiers)
        """
        import re
        s = serial.lower().strip()
        s = re.sub(r'^(sn|s/n|s\.n\.|#)\s*', '', s)
        # Strip only known track-indicator suffixes
        s = re.sub(r'^(\d+)[abprt]$', r'\1', s)
        s = s.lstrip('0') or '0'
        return s

    @staticmethod
    def _normalize_serial_aggressive(serial: str) -> str:
        """
        Aggressively normalize a serial number — strips ALL trailing letters.

        Used as a fallback when selective normalization fails to find a match.
        May produce false matches (e.g. 25D matches 25E) but increases recall.
        """
        import re
        s = serial.lower().strip()
        s = re.sub(r'^(sn|s/n|s\.n\.|#)\s*', '', s)
        s = re.sub(r'^(\d+)[a-z]$', r'\1', s)
        s = s.lstrip('0') or '0'
        return s

    @staticmethod
    def _normalize_model(model: str) -> str:
        """
        Normalize a model number to its base form for variant matching.

        Strips trailing letter suffixes that indicate product variants:
        - 8275A, 8275B, 8275C → 8275
        - 8508-A, 8508-B → 8508
        - 7280-1-CT, 7280-1-AB → 7280-1

        Strips leading zeros in hyphenated suffixes:
        - 2475-08 → 2475-8
        - 8867-01 → 8867-1

        Does NOT strip numeric suffixes (8340-1 stays 8340-1) since
        those are distinct model configurations.
        """
        import re
        if not model:
            return model
        # Strip leading zeros in hyphenated numeric suffixes: "2475-08" → "2475-8"
        s = re.sub(r'-0+(\d)', r'-\1', model)
        # Strip trailing letter-only variant: "8275A" → "8275"
        s = re.sub(r'^(\d+)[A-Za-z]$', r'\1', s)
        # Strip trailing hyphen + letter(s) variant: "8508-A" → "8508", "7280-1-CT" → "7280-1"
        s = re.sub(r'^(\d+(?:-\d+)*)-[A-Za-z]+$', r'\1', s)
        return s

    def _find_matching_trim(
        self,
        session: Session,
        model: Optional[str],
        serial: Optional[str],
        test_date: Optional[datetime]
    ) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[str]]:
        """
        Find the matching trim result for a final test.

        Logic:
        1. Exact model + exact serial match (case-insensitive) — highest confidence
        2. Exact model + fuzzy serial match (strip zeros, prefixes, track suffixes)
        3. Normalized model + fuzzy serial match (8275A trim matches 8275 FT)

        Returns:
            Tuple of (trim_id, confidence, days_since_trim, match_method)
        """
        from laser_trim_analyzer.utils.constants import FINAL_TEST_MAX_DAYS_FROM_TRIM

        if not model or not serial or not test_date:
            return None, None, None, None

        serial_clean = serial.lower().strip()
        cutoff_date = test_date - timedelta(days=FINAL_TEST_MAX_DAYS_FROM_TRIM)

        # Attempt 1: Exact model + exact serial match (case-insensitive)
        candidates = (
            session.query(DBAnalysisResult)
            .filter(
                DBAnalysisResult.model == model,
                func.lower(DBAnalysisResult.serial) == serial_clean,
                DBAnalysisResult.file_date.isnot(None),
                DBAnalysisResult.file_date <= test_date,
                DBAnalysisResult.file_date >= cutoff_date,
            )
            .order_by(desc(DBAnalysisResult.file_date))
            .limit(5)
            .all()
        )

        if candidates:
            match = candidates[0]
            days_diff = (test_date - match.file_date).days
            confidence = self._calculate_match_confidence(days_diff, exact_serial=True)
            return match.id, confidence, days_diff, "exact"

        # Attempt 2: Exact model + fuzzy serial match
        ft_serial_norm = self._normalize_serial(serial)

        model_trims = (
            session.query(DBAnalysisResult.id, DBAnalysisResult.serial, DBAnalysisResult.file_date)
            .filter(
                DBAnalysisResult.model == model,
                DBAnalysisResult.file_date.isnot(None),
                DBAnalysisResult.file_date <= test_date,
                DBAnalysisResult.file_date >= cutoff_date,
            )
            .order_by(desc(DBAnalysisResult.file_date))
            .all()
        )

        for trim_id, trim_serial, trim_date in model_trims:
            if trim_serial and self._normalize_serial(trim_serial) == ft_serial_norm:
                days_diff = (test_date - trim_date).days
                confidence = self._calculate_match_confidence(days_diff, exact_serial=False)
                logger.debug(
                    f"Fuzzy match: FT serial '{serial}' → trim serial '{trim_serial}' "
                    f"(normalized: '{ft_serial_norm}'), {days_diff} days"
                )
                return trim_id, confidence, days_diff, "fuzzy_serial"

        # Attempt 2b: Exact model + aggressively normalized serial (strips all trailing letters)
        ft_serial_aggressive = self._normalize_serial_aggressive(serial)
        if ft_serial_aggressive != ft_serial_norm:
            for trim_id, trim_serial, trim_date in model_trims:
                if trim_serial and self._normalize_serial_aggressive(trim_serial) == ft_serial_aggressive:
                    days_diff = (test_date - trim_date).days
                    confidence = self._calculate_match_confidence(days_diff, exact_serial=False) * 0.90
                    logger.debug(
                        f"Aggressive fuzzy match: FT serial '{serial}' -> trim serial '{trim_serial}' "
                        f"(aggressive norm: '{ft_serial_aggressive}'), {days_diff} days"
                    )
                    return trim_id, confidence, days_diff, "fuzzy_serial_aggressive"

        # Attempt 3: Model variant matching — normalize model on both sides
        # This handles cases like FT model "8275" matching trim model "8275A"
        # or FT model "8508" matching trim model "8508-A"
        ft_model_norm = self._normalize_model(model)

        if ft_model_norm != model:
            # FT model itself has a suffix — try base model in trim
            variant_trims = (
                session.query(DBAnalysisResult.id, DBAnalysisResult.serial,
                              DBAnalysisResult.file_date, DBAnalysisResult.model)
                .filter(
                    DBAnalysisResult.model == ft_model_norm,
                    DBAnalysisResult.file_date.isnot(None),
                    DBAnalysisResult.file_date <= test_date,
                    DBAnalysisResult.file_date >= cutoff_date,
                )
                .order_by(desc(DBAnalysisResult.file_date))
                .all()
            )
            for trim_id, trim_serial, trim_date, trim_model in variant_trims:
                if trim_serial and self._normalize_serial(trim_serial) == ft_serial_norm:
                    days_diff = (test_date - trim_date).days
                    confidence = self._calculate_match_confidence(days_diff, exact_serial=False, model_variant=True)
                    logger.debug(
                        f"Model variant match: FT {model}/{serial} → trim {trim_model}/{trim_serial} "
                        f"(normalized model: '{ft_model_norm}'), {days_diff} days"
                    )
                    return trim_id, confidence, days_diff, "model_variant"

        # Try reverse: trim has variant suffixes, FT has base model
        # Find all trim models that normalize to our FT model
        # Use LIKE to find variants efficiently (e.g. "8275%" for FT model "8275")
        variant_trims = (
            session.query(DBAnalysisResult.id, DBAnalysisResult.serial,
                          DBAnalysisResult.file_date, DBAnalysisResult.model)
            .filter(
                DBAnalysisResult.model.like(f"{model}%"),
                DBAnalysisResult.model != model,  # Skip exact (already tried)
                DBAnalysisResult.file_date.isnot(None),
                DBAnalysisResult.file_date <= test_date,
                DBAnalysisResult.file_date >= cutoff_date,
            )
            .order_by(desc(DBAnalysisResult.file_date))
            .all()
        )

        for trim_id, trim_serial, trim_date, trim_model in variant_trims:
            # Verify this is actually a variant (normalizes to same base)
            if self._normalize_model(trim_model) != ft_model_norm:
                continue
            if trim_serial and self._normalize_serial(trim_serial) == ft_serial_norm:
                days_diff = (test_date - trim_date).days
                confidence = self._calculate_match_confidence(days_diff, exact_serial=False, model_variant=True)
                logger.debug(
                    f"Model variant match: FT {model}/{serial} → trim {trim_model}/{trim_serial} "
                    f"(base model: '{ft_model_norm}'), {days_diff} days"
                )
                return trim_id, confidence, days_diff, "model_variant"

        return None, None, None, None

    @staticmethod
    def _calculate_match_confidence(days_diff: int, exact_serial: bool = True,
                                     model_variant: bool = False) -> float:
        """Calculate match confidence based on time proximity and match quality.

        Confidence bands:
        - Exact model + exact serial, same week: 0.93-1.00
        - Exact model + fuzzy serial, same week: 0.84-0.90
        - Model variant + fuzzy serial, same week: 0.71-0.77
        - Any match beyond 30 days: drops significantly
        """
        # Time-based confidence
        if days_diff <= 7:
            time_conf = 1.0 - (days_diff * 0.01)
        elif days_diff <= 30:
            time_conf = 0.9 - ((days_diff - 7) * 0.01)
        else:
            time_conf = 0.7 - ((days_diff - 30) * 0.007)

        # Match quality penalties (applied multiplicatively)
        if not exact_serial:
            time_conf *= 0.90  # was 0.95 — fuzzy serial is less certain

        if model_variant:
            time_conf *= 0.85  # model variant adds uncertainty

        return max(0.40, time_conf)

    def get_unmatched_ft_diagnostics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Diagnose why Final Test records are unmatched.

        For each unmatched FT record, checks:
        - Does the model exist in trim data?
        - Does the serial exist for that model?
        - Are there trims but outside the date window?

        Returns:
            List of dicts with FT info and reason for no match
        """
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
        )
        from laser_trim_analyzer.utils.constants import FINAL_TEST_MAX_DAYS_FROM_TRIM

        results = []

        with self.session() as session:
            unmatched = (
                session.query(DBFinalTestResult)
                .filter(DBFinalTestResult.linked_trim_id.is_(None))
                .order_by(desc(DBFinalTestResult.file_date))
                .limit(limit)
                .all()
            )

            for ft in unmatched:
                diag = {
                    "ft_id": ft.id,
                    "filename": ft.filename,
                    "model": ft.model,
                    "serial": ft.serial,
                    "test_date": ft.file_date or ft.test_date,
                    "reason": "unknown",
                }

                # Check 1: Does model exist in trim data?
                model_exists = session.query(
                    func.count(DBAnalysisResult.id)
                ).filter(
                    DBAnalysisResult.model == ft.model
                ).scalar() or 0

                if model_exists == 0:
                    diag["reason"] = "no_model_in_trims"
                    diag["detail"] = f"Model '{ft.model}' has no trim records"
                    results.append(diag)
                    continue

                # Check 2: Does serial exist for this model?
                serial_clean = ft.serial.lower().strip() if ft.serial else ""
                serial_exists = session.query(
                    func.count(DBAnalysisResult.id)
                ).filter(
                    DBAnalysisResult.model == ft.model,
                    func.lower(DBAnalysisResult.serial) == serial_clean,
                ).scalar() or 0

                if serial_exists == 0:
                    # Check fuzzy match
                    ft_norm = self._normalize_serial(ft.serial) if ft.serial else ""
                    all_serials = session.query(
                        DBAnalysisResult.serial
                    ).filter(
                        DBAnalysisResult.model == ft.model
                    ).distinct().limit(5).all()
                    sample = [s[0] for s in all_serials]
                    diag["reason"] = "no_serial_match"
                    diag["detail"] = (
                        f"Serial '{ft.serial}' (norm: '{ft_norm}') not found. "
                        f"Sample trim serials: {sample}"
                    )
                    results.append(diag)
                    continue

                # Check 3: Serial exists but outside date window
                test_date = ft.file_date or ft.test_date
                if test_date:
                    cutoff = test_date - timedelta(days=FINAL_TEST_MAX_DAYS_FROM_TRIM)
                    in_window = session.query(
                        func.count(DBAnalysisResult.id)
                    ).filter(
                        DBAnalysisResult.model == ft.model,
                        func.lower(DBAnalysisResult.serial) == serial_clean,
                        DBAnalysisResult.file_date.isnot(None),
                        DBAnalysisResult.file_date <= test_date,
                        DBAnalysisResult.file_date >= cutoff,
                    ).scalar() or 0

                    if in_window == 0:
                        # Find nearest trim date
                        nearest = session.query(
                            DBAnalysisResult.file_date
                        ).filter(
                            DBAnalysisResult.model == ft.model,
                            func.lower(DBAnalysisResult.serial) == serial_clean,
                            DBAnalysisResult.file_date.isnot(None),
                        ).order_by(
                            func.abs(func.julianday(DBAnalysisResult.file_date) - func.julianday(test_date))
                        ).first()

                        if nearest and nearest[0]:
                            gap = abs((test_date - nearest[0]).days)
                            diag["reason"] = "outside_date_window"
                            diag["detail"] = (
                                f"Nearest trim is {gap} days away "
                                f"(max allowed: {FINAL_TEST_MAX_DAYS_FROM_TRIM})"
                            )
                        else:
                            diag["reason"] = "no_dated_trims"
                            diag["detail"] = "Matching trims have no file_date"
                    else:
                        diag["reason"] = "trim_after_test"
                        diag["detail"] = "Trims exist but all are after the FT date"
                else:
                    diag["reason"] = "no_test_date"
                    diag["detail"] = "FT record has no date"

                results.append(diag)

        return results

    def rematch_final_tests(self) -> Dict[str, int]:
        """
        Re-run matching for all Final Test records against current trim data.

        This is useful when trim files are imported after Final Test files,
        or when trim data has been updated.

        Returns:
            Dict with counts: new_matches, updated_matches, unchanged, total
        """
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
        )

        stats = {"new_matches": 0, "updated_matches": 0, "unchanged": 0, "total": 0}

        with self._write_lock:
            with self.session() as session:
                # Get all Final Test records
                final_tests = session.query(DBFinalTestResult).all()
                stats["total"] = len(final_tests)

                for ft in final_tests:
                    # Get test date (prefer file_date, fall back to test_date)
                    test_date = ft.file_date or ft.test_date

                    # Find matching trim
                    new_trim_id, new_confidence, new_days, new_method = self._find_matching_trim(
                        session, ft.model, ft.serial, test_date
                    )

                    # Check if match changed
                    if new_trim_id != ft.linked_trim_id:
                        if ft.linked_trim_id is None and new_trim_id is not None:
                            stats["new_matches"] += 1
                        elif ft.linked_trim_id is not None and new_trim_id is not None:
                            stats["updated_matches"] += 1

                        # Update the record
                        ft.linked_trim_id = new_trim_id
                        ft.match_confidence = new_confidence
                        ft.days_since_trim = new_days
                        ft.match_method = new_method
                    else:
                        stats["unchanged"] += 1

                session.commit()
                logger.info(
                    f"Rematch complete: {stats['new_matches']} new, "
                    f"{stats['updated_matches']} updated, {stats['unchanged']} unchanged"
                )

        return stats

    def get_final_test(self, final_test_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a Final Test result by ID.

        Returns:
            Dict with final test data, tracks, and linked trim info
        """
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
            FinalTestTrack as DBFinalTestTrack,
        )

        with self.session() as session:
            result = session.query(DBFinalTestResult).filter(
                DBFinalTestResult.id == final_test_id
            ).first()

            if not result:
                return None

            # Get tracks
            tracks = session.query(DBFinalTestTrack).filter(
                DBFinalTestTrack.final_test_id == final_test_id
            ).all()

            # Get linked trim if exists
            linked_trim = None
            if result.linked_trim_id:
                linked_trim = self._get_analysis_summary(session, result.linked_trim_id)

            return {
                "id": result.id,
                "filename": result.filename,
                "model": result.model,
                "serial": result.serial,
                "test_date": result.test_date,
                "file_date": result.file_date,
                "overall_status": result.overall_status.value if result.overall_status else "UNKNOWN",
                "linearity_pass": result.linearity_pass,
                "linearity_error": result.linearity_error,
                "resistance_pass": result.resistance_pass,
                "resistance_value": result.resistance_value,
                "linked_trim_id": result.linked_trim_id,
                "match_confidence": result.match_confidence,
                "days_since_trim": result.days_since_trim,
                "match_method": getattr(result, 'match_method', None),
                "linked_trim": linked_trim,
                "tracks": [
                    {
                        "track_id": t.track_id,
                        "status": t.status.value if t.status else "UNKNOWN",
                        "linearity_pass": t.linearity_pass,
                        "linearity_error": t.linearity_error,
                        "linearity_fail_points": t.linearity_fail_points,
                        # Use position_data if available, fall back to electrical_angle_data
                        "positions": t.position_data or t.electrical_angle_data or [],
                        "errors": t.error_data or [],
                        "electrical_angles": t.electrical_angle_data or [],
                        "upper_limits": t.upper_limits or [],
                        "lower_limits": t.lower_limits or [],
                    }
                    for t in tracks
                ],
            }

    def _get_analysis_summary(self, session: Session, analysis_id: int) -> Optional[Dict[str, Any]]:
        """Get a summary of an analysis result for linking."""
        result = session.query(DBAnalysisResult).filter(
            DBAnalysisResult.id == analysis_id
        ).first()

        if not result:
            return None

        return {
            "id": result.id,
            "filename": result.filename,
            "model": result.model,
            "serial": result.serial,
            "file_date": result.file_date,
            "overall_status": result.overall_status.value if result.overall_status else "UNKNOWN",
        }

    def search_final_tests(
        self,
        model: Optional[str] = None,
        serial: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        status: Optional[str] = None,
        linked_only: bool = False,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Search Final Test records with filters.

        Supports partial matching on serial number (case-insensitive).
        Filters by test date (file_date).

        Uses LEFT JOIN to fetch linked trim data in single query (avoids N+1).

        Args:
            model: Filter by model number (exact match, None for all)
            serial: Filter by serial number (partial match, case-insensitive)
            date_from: Start of date range (inclusive)
            date_to: End of date range (inclusive)
            status: Filter by status (Pass/Fail, None for all)
            linked_only: If True, only return records with linked trim
            limit: Maximum number of results

        Returns:
            List of Final Test result dicts sorted by test date (newest first)
        """
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
        )

        with self.session() as session:
            # Use joinedload to fetch linked_trim in single query (avoids N+1)
            query = session.query(DBFinalTestResult).options(
                joinedload(DBFinalTestResult.linked_trim)
            )

            # Filter by model (exact match)
            if model and model != "All Models":
                query = query.filter(DBFinalTestResult.model == model)

            # Filter by serial (partial match, case-insensitive)
            if serial and serial.strip():
                serial_pattern = f"%{serial.strip()}%"
                query = query.filter(
                    func.lower(DBFinalTestResult.serial).like(func.lower(serial_pattern))
                )

            # Filter by date range (test date)
            if date_from:
                query = query.filter(DBFinalTestResult.file_date >= date_from)
            if date_to:
                # Include the entire end date (until midnight)
                end_of_day = date_to.replace(hour=23, minute=59, second=59)
                query = query.filter(DBFinalTestResult.file_date <= end_of_day)

            # Filter by status
            if status and status != "All":
                from laser_trim_analyzer.database.models import FinalTestStatus
                if status == "Pass":
                    query = query.filter(DBFinalTestResult.overall_status == FinalTestStatus.PASS)
                elif status == "Fail":
                    query = query.filter(DBFinalTestResult.overall_status == FinalTestStatus.FAIL)

            # Filter by linked status
            if linked_only:
                query = query.filter(DBFinalTestResult.linked_trim_id.isnot(None))

            # Order by test date (newest first) and limit
            results = query.order_by(desc(DBFinalTestResult.file_date)).limit(limit).all()

            pairs = []
            for ft in results:
                # Get linked trim info from pre-loaded relationship (no extra query)
                linked_trim = None
                if ft.linked_trim:
                    linked_trim = {
                        "id": ft.linked_trim.id,
                        "filename": ft.linked_trim.filename,
                        "model": ft.linked_trim.model,
                        "serial": ft.linked_trim.serial,
                        "file_date": ft.linked_trim.file_date,
                        "overall_status": ft.linked_trim.overall_status.value if ft.linked_trim.overall_status else "UNKNOWN",
                    }

                pairs.append({
                    "final_test_id": ft.id,
                    "final_test_filename": ft.filename,
                    "model": ft.model,
                    "serial": ft.serial,
                    "final_test_date": ft.file_date or ft.test_date,
                    "final_test_status": ft.overall_status.value if ft.overall_status else "UNKNOWN",
                    "linearity_pass": ft.linearity_pass,
                    "linked_trim_id": ft.linked_trim_id,
                    "linked_trim": linked_trim,
                    "match_confidence": ft.match_confidence,
                    "days_since_trim": ft.days_since_trim,
                    "match_method": getattr(ft, 'match_method', None),
                    "is_linked": ft.linked_trim_id is not None,
                })

            return pairs

    def get_comparison_pairs(
        self,
        model: Optional[str] = None,
        days_back: int = 90,
        linked_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get Final Test + Trim comparison pairs.

        Args:
            model: Filter by model (None = all models)
            days_back: How far back to look
            linked_only: If True, only return pairs with linked trim

        Returns:
            List of comparison pair dicts
        """
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
        )

        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Filter by test date (file_date) not processing date (timestamp)
            query = session.query(DBFinalTestResult).filter(
                DBFinalTestResult.file_date >= cutoff_date
            )

            if model:
                query = query.filter(DBFinalTestResult.model == model)

            if linked_only:
                query = query.filter(DBFinalTestResult.linked_trim_id.isnot(None))

            # Sort by test date (file_date)
            results = query.order_by(desc(DBFinalTestResult.file_date)).limit(500).all()

            # Pre-fetch all linked trims in a single query (avoids N+1)
            linked_trim_ids = [ft.linked_trim_id for ft in results if ft.linked_trim_id]
            linked_trims = {}
            if linked_trim_ids:
                trim_results = session.query(DBAnalysisResult).filter(
                    DBAnalysisResult.id.in_(linked_trim_ids)
                ).all()
                for trim in trim_results:
                    linked_trims[trim.id] = {
                        "id": trim.id,
                        "filename": trim.filename,
                        "model": trim.model,
                        "serial": trim.serial,
                        "file_date": trim.file_date,
                        "overall_status": trim.overall_status.value if trim.overall_status else "UNKNOWN",
                    }

            pairs = []
            for ft in results:
                # Get linked trim info from pre-fetched dict
                linked_trim = linked_trims.get(ft.linked_trim_id) if ft.linked_trim_id else None

                pairs.append({
                    "final_test_id": ft.id,
                    "final_test_filename": ft.filename,
                    "model": ft.model,
                    "serial": ft.serial,
                    "final_test_date": ft.file_date or ft.test_date,
                    "final_test_status": ft.overall_status.value if ft.overall_status else "UNKNOWN",
                    "linearity_pass": ft.linearity_pass,
                    "linked_trim_id": ft.linked_trim_id,
                    "linked_trim": linked_trim,
                    "match_confidence": ft.match_confidence,
                    "days_since_trim": ft.days_since_trim,
                    "match_method": getattr(ft, 'match_method', None),
                    "is_linked": ft.linked_trim_id is not None,
                })

            return pairs

    def get_comparison_data(
        self,
        final_test_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get full comparison data for overlay chart.

        Returns both Final Test and linked Trim data with track details.
        """
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
            FinalTestTrack as DBFinalTestTrack,
        )

        with self.session() as session:
            # Get final test with tracks in single query (avoids N+1)
            ft = session.query(DBFinalTestResult).options(
                joinedload(DBFinalTestResult.tracks)
            ).filter(
                DBFinalTestResult.id == final_test_id
            ).first()

            if not ft:
                return None

            final_test_data = {
                "id": ft.id,
                "filename": ft.filename,
                "model": ft.model,
                "serial": ft.serial,
                "test_date": ft.file_date or ft.test_date,
                "status": ft.overall_status.value if ft.overall_status else "UNKNOWN",
                "linearity_pass": ft.linearity_pass,
                "linearity_error": ft.linearity_error,
                "tracks": [
                    {
                        "track_id": t.track_id,
                        # Use position_data if available, fall back to electrical_angle_data
                        "positions": t.position_data or t.electrical_angle_data or [],
                        "errors": t.error_data or [],
                        "electrical_angles": t.electrical_angle_data or [],
                        "upper_limits": t.upper_limits or [],
                        "lower_limits": t.lower_limits or [],
                        "linearity_error": t.linearity_error,
                    }
                    for t in ft.tracks  # Use pre-loaded tracks
                ]
            }

            # Get linked trim data if exists (with tracks in single query)
            trim_data = None
            if ft.linked_trim_id:
                trim = session.query(DBAnalysisResult).options(
                    joinedload(DBAnalysisResult.tracks)
                ).filter(
                    DBAnalysisResult.id == ft.linked_trim_id
                ).first()

                if trim:
                    trim_data = {
                        "id": trim.id,
                        "filename": trim.filename,
                        "model": trim.model,
                        "serial": trim.serial,
                        "file_date": trim.file_date,
                        "status": trim.overall_status.value if trim.overall_status else "UNKNOWN",
                        "tracks": [
                            {
                                "track_id": t.track_id,
                                "positions": t.position_data or [],
                                "errors": t.error_data or [],
                                "upper_limits": t.upper_limits or [],
                                "lower_limits": t.lower_limits or [],
                                "optimal_offset": t.optimal_offset or 0,
                                "linearity_error": t.final_linearity_error_shifted,
                                "linearity_pass": t.linearity_pass,
                                "sigma_gradient": t.sigma_gradient,
                                "sigma_pass": t.sigma_pass,
                            }
                            for t in trim.tracks  # Use pre-loaded tracks
                        ]
                    }

            return {
                "final_test": final_test_data,
                "trim": trim_data,
                "match_confidence": ft.match_confidence,
                "days_since_trim": ft.days_since_trim,
                "match_method": getattr(ft, 'match_method', None),
            }

    def get_final_test_models_list(self) -> List[str]:
        """
        Get list of unique models from Final Test results, sorted numerically.
        """
        from laser_trim_analyzer.database.models import FinalTestResult as DBFinalTestResult

        with self.session() as session:
            models = (
                session.query(DBFinalTestResult.model)
                .filter(DBFinalTestResult.model.isnot(None))
                .distinct()
                .all()
            )
            model_list = [m[0] for m in models if m[0]]
            return sorted(model_list, key=_model_sort_key)

    def get_final_tests_missing_tracks(self) -> List[Dict[str, Any]]:
        """
        Get Final Test records that have 0 tracks stored.

        Returns:
            List of dicts with id, filename, file_path, model
        """
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
            FinalTestTrack as DBFinalTestTrack,
        )

        with self.session() as session:
            # Subquery to count tracks per final test
            track_count_subq = (
                session.query(
                    DBFinalTestTrack.final_test_id,
                    func.count(DBFinalTestTrack.id).label('track_count')
                )
                .group_by(DBFinalTestTrack.final_test_id)
                .subquery()
            )

            # Get Final Tests with no tracks (LEFT JOIN where track_count is NULL)
            results = (
                session.query(DBFinalTestResult)
                .outerjoin(track_count_subq, DBFinalTestResult.id == track_count_subq.c.final_test_id)
                .filter(track_count_subq.c.track_count == None)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "filename": r.filename,
                    "file_path": r.file_path,
                    "model": r.model,
                    "serial": r.serial,
                }
                for r in results
            ]

    def update_final_test_tracks(
        self,
        final_test_id: int,
        tracks: List[Dict[str, Any]]
    ) -> bool:
        """
        Update track data for an existing Final Test record.

        Used to fix records that were created before parser improvements.

        Args:
            final_test_id: ID of the Final Test record
            tracks: List of track data dicts

        Returns:
            True if successful
        """
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
            FinalTestTrack as DBFinalTestTrack,
        )

        with self._write_lock:
            try:
                with self.session() as session:
                    # Get existing record
                    result = session.query(DBFinalTestResult).get(final_test_id)
                    if not result:
                        logger.warning(f"Final Test ID {final_test_id} not found")
                        return False

                    # Delete existing tracks (if any)
                    session.query(DBFinalTestTrack).filter(
                        DBFinalTestTrack.final_test_id == final_test_id
                    ).delete()

                    # Add new tracks
                    for track_data in tracks:
                        position_values = track_data.get("electrical_angles") or track_data.get("positions")

                        db_track = DBFinalTestTrack(
                            final_test_id=final_test_id,
                            track_id=track_data.get("track_id", "default"),
                            status=DBStatusType.PASS if track_data.get("linearity_pass", True) else DBStatusType.FAIL,
                            linearity_spec=track_data.get("linearity_spec"),
                            linearity_error=track_data.get("linearity_error"),
                            linearity_pass=track_data.get("linearity_pass"),
                            linearity_fail_points=track_data.get("linearity_fail_points", 0),
                            position_data=position_values,
                            error_data=track_data.get("errors"),
                            electrical_angle_data=track_data.get("electrical_angles"),
                            upper_limits=track_data.get("upper_limits"),
                            lower_limits=track_data.get("lower_limits"),
                            max_deviation=track_data.get("max_deviation"),
                            max_deviation_position=track_data.get("max_deviation_angle"),
                        )
                        session.add(db_track)

                    # Update linearity_error on main record if tracks have it
                    if tracks and tracks[0].get("linearity_error") is not None:
                        result.linearity_error = tracks[0].get("linearity_error")

                    session.commit()
                    logger.info(f"Updated Final Test {final_test_id} with {len(tracks)} tracks")
                    return True

            except Exception as e:
                logger.error(f"Error updating Final Test tracks: {e}")
                return False

    def get_trim_records_missing_tracks(self, linked_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get Trim (AnalysisResult) records that have no track data stored.

        Args:
            linked_only: If True, only return records that are linked to Final Tests

        Returns:
            List of record info dicts with id, filename, file_path, model, serial
        """
        with self.session() as session:
            # Subquery to count tracks per analysis
            track_count_subq = (
                session.query(
                    DBTrackResult.analysis_id,
                    func.count(DBTrackResult.id).label('track_count')
                )
                .group_by(DBTrackResult.analysis_id)
                .subquery()
            )

            # Base query for analyses with no tracks
            query = (
                session.query(DBAnalysisResult)
                .outerjoin(track_count_subq, DBAnalysisResult.id == track_count_subq.c.analysis_id)
                .filter(
                    (track_count_subq.c.track_count == None) |
                    (track_count_subq.c.track_count == 0)
                )
            )

            if linked_only:
                # Get IDs of analyses that are linked to Final Tests
                from laser_trim_analyzer.database.models import FinalTestResult as DBFinalTestResult
                linked_ids = (
                    session.query(DBFinalTestResult.linked_trim_id)
                    .filter(DBFinalTestResult.linked_trim_id != None)
                    .distinct()
                    .all()
                )
                linked_id_list = [lid[0] for lid in linked_ids]
                query = query.filter(DBAnalysisResult.id.in_(linked_id_list))

            results = query.all()

            return [
                {
                    "id": r.id,
                    "filename": r.filename,
                    "file_path": r.file_path,
                    "model": r.model,
                    "serial": r.serial,
                }
                for r in results
            ]

    def update_trim_tracks(
        self,
        analysis_id: int,
        tracks: List["TrackResult"]
    ) -> bool:
        """
        Update track data for an existing Trim (AnalysisResult) record.

        Used to fix records that were created before track data storage was added.

        Args:
            analysis_id: ID of the AnalysisResult record
            tracks: List of TrackResult objects from re-parsing

        Returns:
            True if successful
        """
        try:
            with self.session() as session:
                # Get existing record
                result = session.query(DBAnalysisResult).get(analysis_id)
                if not result:
                    logger.warning(f"Analysis ID {analysis_id} not found")
                    return False

                # Delete existing tracks (if any)
                session.query(DBTrackResult).filter(
                    DBTrackResult.analysis_id == analysis_id
                ).delete()

                # Add new tracks
                for track in tracks:
                    db_track = self._map_track_to_db(track)
                    db_track.analysis_id = analysis_id
                    session.add(db_track)

                session.commit()
                logger.info(f"Updated Analysis {analysis_id} with {len(tracks)} tracks")
                return True

        except Exception as e:
            logger.error(f"Error updating Trim tracks: {e}")
            return False

    def update_trim_tracks_from_final_test(
        self,
        analysis_id: int,
        ft_tracks: List[Dict[str, Any]]
    ) -> bool:
        """
        Update Trim (AnalysisResult) track data from Final Test format data.

        Used when "Trim" records actually point to Final Test files.
        Converts FT track format to TrackResult format.

        Args:
            analysis_id: ID of the AnalysisResult record
            ft_tracks: List of track dicts from Final Test parser

        Returns:
            True if successful
        """
        try:
            with self.session() as session:
                # Get existing record
                result = session.query(DBAnalysisResult).get(analysis_id)
                if not result:
                    logger.warning(f"Analysis ID {analysis_id} not found")
                    return False

                # Delete existing tracks (if any)
                session.query(DBTrackResult).filter(
                    DBTrackResult.analysis_id == analysis_id
                ).delete()

                # Add new tracks converted from FT format
                for ft_track in ft_tracks:
                    # Get position data (FT format uses electrical_angles)
                    positions = ft_track.get("electrical_angles") or ft_track.get("positions", [])
                    errors = ft_track.get("errors", [])
                    upper_limits = ft_track.get("upper_limits", [])
                    lower_limits = ft_track.get("lower_limits", [])

                    # Calculate linearity metrics
                    linearity_error = ft_track.get("linearity_error", 0.0)
                    linearity_spec = ft_track.get("linearity_spec", 0.02)
                    linearity_pass = ft_track.get("linearity_pass", True)

                    # Create TrackResult-compatible DB record
                    db_track = DBTrackResult(
                        analysis_id=analysis_id,
                        track_id=ft_track.get("track_id", "default"),
                        status=DBStatusType.PASS if linearity_pass else DBStatusType.FAIL,
                        # Sigma values - use defaults for FT data
                        sigma_gradient=0.0,
                        sigma_threshold=1.0,
                        sigma_pass=True,
                        # Linearity values
                        linearity_spec=linearity_spec,
                        final_linearity_error_shifted=linearity_error,
                        linearity_pass=linearity_pass,
                        linearity_fail_points=ft_track.get("linearity_fail_points", 0),
                        # Track data for charts
                        position_data=positions,
                        error_data=errors,
                        upper_limits=upper_limits,
                        lower_limits=lower_limits,
                        # Travel length from position range
                        travel_length=max(positions) - min(positions) if positions else 0,
                    )
                    session.add(db_track)

                session.commit()
                logger.info(f"Updated Analysis {analysis_id} with {len(ft_tracks)} tracks from FT format")
                return True

        except Exception as e:
            logger.error(f"Error updating Trim tracks from FT: {e}")
            return False

    # =========================================================================
    # Database Health & Cleanup
    # =========================================================================

    def scan_database_health(self) -> Dict[str, Any]:
        """
        Scan the entire database and return a health report.

        Identifies dirty/suspect records across multiple categories without
        modifying anything. Returns counts and record IDs for each issue.
        """
        health = {
            "total_analyses": 0,
            "total_tracks": 0,
            "issues": {},
            "total_dirty_records": 0,
        }

        with self.session() as session:
            health["total_analyses"] = (
                session.query(func.count(DBAnalysisResult.id)).scalar() or 0
            )
            health["total_tracks"] = (
                session.query(func.count(DBTrackResult.id)).scalar() or 0
            )

            dirty_ids = set()

            # 1. Unknown model
            unknown_model = session.query(DBAnalysisResult.id).filter(
                DBAnalysisResult.model == "Unknown"
            ).all()
            if unknown_model:
                ids = {r[0] for r in unknown_model}
                dirty_ids |= ids
                health["issues"]["unknown_model"] = {
                    "count": len(ids),
                    "label": "Unknown model (parser couldn't extract)",
                }

            # 2. Unknown serial
            unknown_serial = session.query(DBAnalysisResult.id).filter(
                DBAnalysisResult.serial == "Unknown"
            ).all()
            if unknown_serial:
                ids = {r[0] for r in unknown_serial}
                dirty_ids |= ids
                health["issues"]["unknown_serial"] = {
                    "count": len(ids),
                    "label": "Unknown serial number",
                }

            # 3. Missing file date
            null_date = session.query(DBAnalysisResult.id).filter(
                DBAnalysisResult.file_date.is_(None)
            ).all()
            if null_date:
                ids = {r[0] for r in null_date}
                dirty_ids |= ids
                health["issues"]["missing_file_date"] = {
                    "count": len(ids),
                    "label": "Missing file date",
                }

            # 4. ERROR status records
            error_records = session.query(DBAnalysisResult.id).filter(
                DBAnalysisResult.overall_status == DBStatusType.ERROR
            ).all()
            if error_records:
                ids = {r[0] for r in error_records}
                dirty_ids |= ids
                health["issues"]["error_status"] = {
                    "count": len(ids),
                    "label": "ERROR status (processing failed)",
                }

            # 5. Analyses with no tracks (orphaned)
            analyses_no_tracks = session.query(DBAnalysisResult.id).filter(
                ~exists().where(DBTrackResult.analysis_id == DBAnalysisResult.id)
            ).all()
            if analyses_no_tracks:
                ids = {r[0] for r in analyses_no_tracks}
                dirty_ids |= ids
                health["issues"]["no_tracks"] = {
                    "count": len(ids),
                    "label": "No track data (empty analyses)",
                }

            # 6. Track-level quality issues (negative sigma, all-zero data, etc.)
            bad_sigma = session.query(
                DBTrackResult.analysis_id
            ).filter(
                DBTrackResult.sigma_gradient < 0
            ).distinct().all()
            if bad_sigma:
                ids = {r[0] for r in bad_sigma}
                dirty_ids |= ids
                health["issues"]["negative_sigma"] = {
                    "count": len(ids),
                    "label": "Negative sigma gradient (impossible value)",
                }

            # 7. Tracks with no spec limits (can't determine pass/fail)
            no_limits = session.query(
                DBTrackResult.analysis_id
            ).filter(
                DBTrackResult.upper_limits.is_(None),
                DBTrackResult.lower_limits.is_(None),
                DBTrackResult.linearity_spec.is_(None),
            ).distinct().all()
            if no_limits:
                ids = {r[0] for r in no_limits}
                dirty_ids |= ids
                health["issues"]["no_spec_limits"] = {
                    "count": len(ids),
                    "label": "No spec limits (can't verify pass/fail)",
                }

            # 8. Already-flagged suspect quality
            suspect = session.query(DBAnalysisResult.id).filter(
                DBAnalysisResult.data_quality == "suspect"
            ).all()
            if suspect:
                ids = {r[0] for r in suspect}
                dirty_ids |= ids
                health["issues"]["suspect_quality"] = {
                    "count": len(ids),
                    "label": "Previously flagged as suspect",
                }

            health["total_dirty_records"] = len(dirty_ids)

        return health

    def retroactive_validate(self) -> Dict[str, Any]:
        """
        Retroactively validate ALL records in the database and update
        data_quality flags.

        Checks analysis-level and track-level quality issues, then
        updates the data_quality and data_quality_issues columns.

        Processes in batches to avoid loading all records into memory at once.

        Returns summary of what was found and updated.
        """
        summary = {"scanned": 0, "flagged": 0, "already_suspect": 0, "issues_by_type": {}}
        batch_size = 1000

        with self._write_lock:
            with self.session() as session:
                total = session.query(func.count(DBAnalysisResult.id)).scalar() or 0
                summary["scanned"] = total

                # Process in batches to avoid memory issues with large databases
                # Use subqueryload (not joinedload) because joinedload + limit
                # produces incorrect results due to JOIN row multiplication
                for offset in range(0, total, batch_size):
                    analyses = session.query(DBAnalysisResult).options(
                        subqueryload(DBAnalysisResult.tracks)
                    ).order_by(DBAnalysisResult.id).offset(offset).limit(batch_size).all()

                    for analysis in analyses:
                        issues = []

                        # Analysis-level checks
                        if analysis.model == "Unknown":
                            issues.append("Unknown model")
                        if analysis.serial == "Unknown":
                            issues.append("Unknown serial")
                        if analysis.file_date is None:
                            issues.append("Missing file date")
                        if not analysis.tracks:
                            issues.append("No track data")

                        # Track-level checks
                        for track in analysis.tracks:
                            tid = track.track_id or "?"

                            if track.sigma_gradient is not None and track.sigma_gradient < 0:
                                issues.append(f"{tid}: negative sigma_gradient ({track.sigma_gradient:.4f})")

                            if track.upper_limits is None and track.lower_limits is None and track.linearity_spec is None:
                                issues.append(f"{tid}: no spec limits")

                            # Check for all-zero error data
                            if track.error_data:
                                try:
                                    if all(v == 0 or v is None for v in track.error_data):
                                        issues.append(f"{tid}: all-zero error data")
                                except (TypeError, ValueError):
                                    issues.append(f"{tid}: corrupt error data")

                            # Check for very short position arrays
                            if track.position_data:
                                try:
                                    if len(track.position_data) < 10:
                                        issues.append(f"{tid}: too few data points ({len(track.position_data)})")
                                except TypeError:
                                    issues.append(f"{tid}: corrupt position data")

                            # Check position/error array length mismatch
                            if track.position_data and track.error_data:
                                try:
                                    if len(track.position_data) != len(track.error_data):
                                        issues.append(
                                            f"{tid}: array mismatch (pos={len(track.position_data)}, err={len(track.error_data)})"
                                        )
                                except TypeError:
                                    pass

                        # Update the record
                        if issues:
                            was_suspect = analysis.data_quality == "suspect"
                            analysis.data_quality = "suspect"
                            analysis.data_quality_issues = ", ".join(issues)
                            if was_suspect:
                                summary["already_suspect"] += 1
                            else:
                                summary["flagged"] += 1

                            # Count by issue type
                            for issue in issues:
                                # Normalize to category
                                category = issue.split(":")[0].strip() if ":" in issue else issue
                                summary["issues_by_type"][category] = summary["issues_by_type"].get(category, 0) + 1
                        else:
                            # Clear false positives from previous scans
                            if analysis.data_quality == "suspect":
                                analysis.data_quality = "good"
                                analysis.data_quality_issues = None

                    # Flush each batch to free memory
                    session.flush()

                logger.info(
                    f"Retroactive validation: scanned {summary['scanned']}, "
                    f"flagged {summary['flagged']} new, "
                    f"{summary['already_suspect']} already suspect"
                )

        return summary

    def _collect_cleanup_ids(
        self,
        session,
        delete_non_mps: bool = False,
        mps_models: Optional[List[str]] = None,
        delete_before_date: Optional[datetime] = None,
        delete_suspect_quality: bool = False,
        delete_unknown: bool = False,
        delete_error_status: bool = False,
        delete_no_tracks: bool = False,
        delete_misclassified_ft: bool = False,
    ) -> tuple:
        """
        Collect record IDs matching cleanup criteria. Shared by preview and execute.

        Returns:
            (ids_to_delete set, by_reason dict)
        """
        ids_to_delete = set()
        by_reason = {}

        if delete_non_mps and mps_models:
            mps_set = set(m.strip() for m in mps_models if m.strip())
            non_mps = session.query(
                DBAnalysisResult.id, DBAnalysisResult.model
            ).filter(
                DBAnalysisResult.model.notin_(mps_set)
            ).all()
            non_mps_ids = {r[0] for r in non_mps}
            non_mps_models = sorted(set(r[1] for r in non_mps))
            ids_to_delete |= non_mps_ids
            by_reason["non_mps_models"] = {
                "count": len(non_mps_ids),
                "models": non_mps_models,
            }

        if delete_before_date:
            old_records = session.query(
                DBAnalysisResult.id
            ).filter(
                DBAnalysisResult.file_date < delete_before_date
            ).all()
            old_ids = {r[0] for r in old_records}
            ids_to_delete |= old_ids
            by_reason["before_date"] = {
                "count": len(old_ids),
                "date": delete_before_date.strftime("%Y-%m-%d"),
            }

        if delete_suspect_quality:
            suspect = session.query(
                DBAnalysisResult.id
            ).filter(
                DBAnalysisResult.data_quality == "suspect"
            ).all()
            suspect_ids = {r[0] for r in suspect}
            ids_to_delete |= suspect_ids
            by_reason["suspect_quality"] = {
                "count": len(suspect_ids),
            }

        if delete_unknown:
            unknown = session.query(
                DBAnalysisResult.id
            ).filter(
                or_(
                    DBAnalysisResult.model == "Unknown",
                    DBAnalysisResult.serial == "Unknown",
                )
            ).all()
            unknown_ids = {r[0] for r in unknown}
            ids_to_delete |= unknown_ids
            by_reason["unknown_model_serial"] = {
                "count": len(unknown_ids),
            }

        if delete_error_status:
            errors = session.query(
                DBAnalysisResult.id
            ).filter(
                DBAnalysisResult.overall_status == DBStatusType.ERROR
            ).all()
            error_ids = {r[0] for r in errors}
            ids_to_delete |= error_ids
            by_reason["error_status"] = {
                "count": len(error_ids),
            }

        if delete_no_tracks:
            no_tracks = session.query(
                DBAnalysisResult.id
            ).filter(
                ~exists().where(DBTrackResult.analysis_id == DBAnalysisResult.id)
            ).all()
            no_track_ids = {r[0] for r in no_tracks}
            ids_to_delete |= no_track_ids
            by_reason["no_tracks"] = {
                "count": len(no_track_ids),
            }

        if delete_misclassified_ft:
            # Find trim records that are actually Final Test files:
            # 1. Files from "Test Station" paths
            # 2. Files with _Redundant_ or _Primary_ in filename
            # 3. Files with "final" followed by a number in filename
            ft_patterns = [
                DBAnalysisResult.filename.like("%Test Station%"),
                DBAnalysisResult.filename.like("%test station%"),
                DBAnalysisResult.filename.like("%_Redundant_%"),
                DBAnalysisResult.filename.like("%_redundant_%"),
                DBAnalysisResult.filename.like("%_Primary_%"),
                DBAnalysisResult.filename.like("%_primary_%"),
            ]
            misclassified = session.query(
                DBAnalysisResult.id
            ).filter(
                or_(*ft_patterns)
            ).all()
            misc_ids = {r[0] for r in misclassified}

            # Also find "model final NNN" pattern files in trim table
            final_pattern = session.query(
                DBAnalysisResult.id
            ).filter(
                DBAnalysisResult.filename.like("% final %")
            ).all()
            final_ids = {r[0] for r in final_pattern}
            misc_ids |= final_ids

            ids_to_delete |= misc_ids
            by_reason["misclassified_ft"] = {
                "count": len(misc_ids),
            }

        return ids_to_delete, by_reason

    def preview_cleanup(
        self,
        delete_non_mps: bool = False,
        mps_models: Optional[List[str]] = None,
        delete_before_date: Optional[datetime] = None,
        delete_suspect_quality: bool = False,
        delete_unknown: bool = False,
        delete_error_status: bool = False,
        delete_no_tracks: bool = False,
        delete_misclassified_ft: bool = False,
    ) -> Dict[str, Any]:
        """
        Preview what a cleanup operation would delete WITHOUT actually deleting.

        Returns:
            Dict with counts and model lists for what would be deleted
        """
        preview = {
            "total_records": 0,
            "records_to_delete": 0,
            "models_to_delete": [],
            "by_reason": {},
        }

        with self.session() as session:
            preview["total_records"] = (
                session.query(func.count(DBAnalysisResult.id)).scalar() or 0
            )

            ids_to_delete, by_reason = self._collect_cleanup_ids(
                session,
                delete_non_mps=delete_non_mps,
                mps_models=mps_models,
                delete_before_date=delete_before_date,
                delete_suspect_quality=delete_suspect_quality,
                delete_unknown=delete_unknown,
                delete_error_status=delete_error_status,
                delete_no_tracks=delete_no_tracks,
                delete_misclassified_ft=delete_misclassified_ft,
            )

            preview["by_reason"] = by_reason
            preview["records_to_delete"] = len(ids_to_delete)

            if ids_to_delete:
                models = session.query(
                    DBAnalysisResult.model
                ).filter(
                    DBAnalysisResult.id.in_(ids_to_delete)
                ).distinct().all()
                preview["models_to_delete"] = sorted(m[0] for m in models)

        return preview

    def execute_cleanup(
        self,
        delete_non_mps: bool = False,
        mps_models: Optional[List[str]] = None,
        delete_before_date: Optional[datetime] = None,
        delete_suspect_quality: bool = False,
        delete_unknown: bool = False,
        delete_error_status: bool = False,
        delete_no_tracks: bool = False,
        delete_misclassified_ft: bool = False,
    ) -> Dict[str, int]:
        """
        Execute database cleanup — permanently delete matching records.

        Uses the same filters as preview_cleanup(). Deletes analysis records
        and associated tracks and alerts. Keeps processed_files records so
        the same bad files won't be reprocessed next time (the FK has
        ondelete=SET NULL so the link is safely cleared).

        Returns:
            Dict with deletion counts
        """
        deleted = {"analyses": 0, "tracks": 0, "alerts": 0}

        with self._write_lock:
            with self.session() as session:
                ids_to_delete, _ = self._collect_cleanup_ids(
                    session,
                    delete_non_mps=delete_non_mps,
                    mps_models=mps_models,
                    delete_before_date=delete_before_date,
                    delete_suspect_quality=delete_suspect_quality,
                    delete_unknown=delete_unknown,
                    delete_error_status=delete_error_status,
                    delete_no_tracks=delete_no_tracks,
                    delete_misclassified_ft=delete_misclassified_ft,
                )

                if not ids_to_delete:
                    return deleted

                # Delete in batches to avoid SQLite variable limits
                id_list = list(ids_to_delete)
                batch_size = 500

                for i in range(0, len(id_list), batch_size):
                    batch = id_list[i:i + batch_size]

                    deleted["tracks"] += session.query(DBTrackResult).filter(
                        DBTrackResult.analysis_id.in_(batch)
                    ).delete(synchronize_session=False)

                    deleted["alerts"] += session.query(DBQAAlert).filter(
                        DBQAAlert.analysis_id.in_(batch)
                    ).delete(synchronize_session=False)

                    # Keep processed_files records — prevents reprocessing
                    # the same bad files. FK ondelete=SET NULL clears the link.

                    deleted["analyses"] += session.query(DBAnalysisResult).filter(
                        DBAnalysisResult.id.in_(batch)
                    ).delete(synchronize_session=False)

                logger.info(
                    f"Database cleanup: deleted {deleted['analyses']} analyses, "
                    f"{deleted['tracks']} tracks, {deleted['alerts']} alerts "
                    f"(processed_files kept to prevent reprocessing)"
                )

        return deleted

    def count_skipped_files(self) -> int:
        """Count non-trim/non-FT files that were skipped and recorded."""
        with self.session() as session:
            return session.query(func.count(DBProcessedFile.id)).filter(
                DBProcessedFile.analysis_id.is_(None),
                DBProcessedFile.success == True,
            ).scalar() or 0

    def reset_skipped_files(self) -> int:
        """
        Remove processed_files entries for skipped non-trim files so they
        get re-evaluated on the next processing run.

        Only clears entries with analysis_id=NULL (no analysis was created),
        which are files that were detected as non-trim and skipped.

        Returns:
            Number of entries cleared
        """
        with self._write_lock:
            with self.session() as session:
                count = session.query(DBProcessedFile).filter(
                    DBProcessedFile.analysis_id.is_(None),
                    DBProcessedFile.success == True,
                ).delete(synchronize_session=False)

                logger.info(f"Reset {count} skipped file entries for reprocessing")

        return count

    def mark_file_skipped(self, filename: str, file_path: str,
                          file_hash: str, file_size: int,
                          file_modified_date) -> None:
        """Record a non-trim file so it's skipped on future processing runs."""
        with self._write_lock:
            with self.session() as session:
                existing = session.query(DBProcessedFile).filter(
                    DBProcessedFile.file_hash == file_hash
                ).first()
                if existing:
                    return

                session.add(DBProcessedFile(
                    filename=filename,
                    file_path=file_path,
                    file_hash=file_hash,
                    file_size=file_size,
                    file_modified_date=file_modified_date,
                    analysis_id=None,
                    success=True,
                ))

    def backfill_max_deviation(self, batch_size: int = 1000) -> int:
        """
        Backfill max_deviation, max_deviation_position, and deviation_uniformity
        for existing tracks that have error_data but no max_deviation.

        Commits in batches so that partial progress is preserved if an error
        occurs mid-way.  This is intentional — a backfill that saves 900 of
        1000 rows is better than one that saves 0.

        Returns:
            Number of tracks updated (may be partial on error)
        """
        import json
        import statistics as stats_module

        updated = 0
        with self._write_lock:
            session = self._SessionFactory()
            try:
                # Get total count first
                total = session.execute(text(
                    "SELECT COUNT(*) FROM track_results "
                    "WHERE max_deviation IS NULL AND error_data IS NOT NULL"
                )).scalar()

                if total == 0:
                    logger.info("No tracks need max_deviation backfill")
                    return 0

                logger.info(f"Backfilling max_deviation for {total} tracks...")

                last_id = 0
                while True:
                    rows = session.execute(text(
                        "SELECT id, error_data, position_data, optimal_offset "
                        "FROM track_results "
                        "WHERE max_deviation IS NULL AND error_data IS NOT NULL "
                        "AND id > :last_id "
                        "ORDER BY id LIMIT :limit"
                    ), {"limit": batch_size, "last_id": last_id}).fetchall()

                    if not rows:
                        break
                    last_id = rows[-1].id

                    for row in rows:
                        try:
                            errors = json.loads(row.error_data) if isinstance(row.error_data, str) else row.error_data
                            positions = json.loads(row.position_data) if isinstance(row.position_data, str) else row.position_data
                            opt_offset = row.optimal_offset or 0.0

                            if not errors or not positions:
                                continue

                            shifted = [e + opt_offset for e in errors]
                            abs_errs = [abs(e) for e in shifted]
                            max_dev = max(abs_errs)
                            max_idx = abs_errs.index(max_dev)
                            max_dev_pos = positions[max_idx] if max_idx < len(positions) else None

                            dev_unif = None
                            if len(abs_errs) > 1:
                                mean_abs = stats_module.mean(abs_errs)
                                if mean_abs > 0:
                                    dev_unif = stats_module.stdev(abs_errs) / mean_abs

                            session.execute(text(
                                "UPDATE track_results SET "
                                "max_deviation = :max_dev, "
                                "max_deviation_position = :max_dev_pos, "
                                "deviation_uniformity = :dev_unif "
                                "WHERE id = :id"
                            ), {
                                "max_dev": max_dev,
                                "max_dev_pos": max_dev_pos,
                                "dev_unif": dev_unif,
                                "id": row.id
                            })
                            updated += 1
                        except Exception as e:
                            logger.warning(f"Failed to backfill track {row.id}: {e}")

                    session.commit()
                    logger.info(f"Backfilled {updated}/{total} tracks...")

            except Exception as e:
                session.rollback()
                logger.error(f"Backfill error after {updated} updates: {e}")
            finally:
                session.close()

        logger.info(f"Backfill complete: {updated} tracks updated")
        return updated

    # =========================================================================
    # Model Specifications
    # =========================================================================

    def get_all_model_specs(self) -> List[Dict[str, Any]]:
        """Get all model specs as dicts."""
        with self.session() as session:
            specs = session.query(ModelSpec).order_by(ModelSpec.model).all()
            return [
                {
                    "id": s.id,
                    "model": s.model,
                    "element_type": s.element_type,
                    "product_class": s.product_class,
                    "linearity_type": s.linearity_type,
                    "linearity_spec_text": s.linearity_spec_text,
                    "linearity_spec_pct": s.linearity_spec_pct,
                    "total_resistance_min": s.total_resistance_min,
                    "total_resistance_max": s.total_resistance_max,
                    "electrical_angle": s.electrical_angle,
                    "electrical_angle_tol": s.electrical_angle_tol,
                    "electrical_angle_unit": s.electrical_angle_unit,
                    "output_smoothness": s.output_smoothness,
                    "circuit_type": s.circuit_type,
                    "notes": s.notes,
                }
                for s in specs
            ]

    def get_model_spec(self, model: str) -> Optional[Dict[str, Any]]:
        """Get spec for a specific model."""
        with self.session() as session:
            spec = session.query(ModelSpec).filter(
                ModelSpec.model == model
            ).first()
            if not spec:
                return None
            return {
                "id": spec.id,
                "model": spec.model,
                "element_type": spec.element_type,
                "product_class": spec.product_class,
                "linearity_type": spec.linearity_type,
                "linearity_spec_text": spec.linearity_spec_text,
                "linearity_spec_pct": spec.linearity_spec_pct,
                "total_resistance_min": spec.total_resistance_min,
                "total_resistance_max": spec.total_resistance_max,
                "electrical_angle": spec.electrical_angle,
                "electrical_angle_tol": spec.electrical_angle_tol,
                "electrical_angle_unit": spec.electrical_angle_unit,
                "output_smoothness": spec.output_smoothness,
                "circuit_type": spec.circuit_type,
                "notes": spec.notes,
            }

    def save_model_spec(self, data: Dict[str, Any]) -> Tuple[int, bool]:
        """Create or update a model spec. Returns (spec_id, was_update)."""
        with self._write_lock:
            with self.session() as session:
                existing = session.query(ModelSpec).filter(
                    ModelSpec.model == data["model"]
                ).first()

                if existing:
                    for key, value in data.items():
                        if key not in ("id", "model", "created_at", "updated_at"):
                            setattr(existing, key, value)
                    # updated_at handled automatically by onupdate=utc_now
                    session.flush()
                    return existing.id, True
                else:
                    spec = ModelSpec(**{k: v for k, v in data.items() if k != "id"})
                    session.add(spec)
                    session.flush()
                    return spec.id, False

    def delete_model_spec(self, model: str) -> bool:
        """Delete a model spec. Returns True if found and deleted."""
        with self._write_lock:
            with self.session() as session:
                spec = session.query(ModelSpec).filter(
                    ModelSpec.model == model
                ).first()
                if spec:
                    session.delete(spec)
                    return True
                return False

    def get_distinct_element_types(self) -> List[str]:
        """Get all distinct element types from model_specs."""
        with self.session() as session:
            results = session.query(ModelSpec.element_type).filter(
                ModelSpec.element_type.isnot(None)
            ).distinct().order_by(ModelSpec.element_type).all()
            return [r[0] for r in results]

    def get_distinct_product_classes(self) -> List[str]:
        """Get all distinct product classes from model_specs."""
        with self.session() as session:
            results = session.query(ModelSpec.product_class).filter(
                ModelSpec.product_class.isnot(None)
            ).distinct().order_by(ModelSpec.product_class).all()
            return [r[0] for r in results]

    def import_model_specs_from_excel(self, file_path: str) -> Dict[str, int]:
        """
        Import model specs from the reference Excel file.
        Merges: updates existing, adds new, never deletes.

        Returns: {"updated": N, "added": N, "skipped": N}
        """
        import re
        import openpyxl

        wb = openpyxl.load_workbook(file_path, read_only=True)
        result = {"updated": 0, "added": 0, "skipped": 0}

        # Collect data from all three sheets
        model_data = {}  # model -> dict of fields

        # Sheet 1: Model Reference (primary, most complete)
        if "Model Reference" in wb.sheetnames:
            ws = wb["Model Reference"]
            for row in ws.iter_rows(min_row=2, values_only=True):
                if not row or len(row) < 9:
                    continue
                model = str(row[1]).strip() if row[1] else None
                if not model:
                    continue

                # Skip Customer (row[2]) and Operator (row[9])
                element_type = str(row[3]).strip() if row[3] else None
                linearity_text = str(row[4]).strip() if row[4] else None
                resistance_text = str(row[5]).strip() if row[5] else None
                angle_text = str(row[6]).strip() if row[6] else None
                smoothness = str(row[7]).strip() if row[7] else None
                circuit = str(row[8]).strip() if row[8] else None
                product_class = str(row[10]).strip() if len(row) > 10 and row[10] else None

                # Parse linearity type from text
                linearity_type = None
                linearity_pct = None
                if linearity_text:
                    # Extract type: look for (Absolute), (Independent), etc.
                    type_match = re.search(
                        r'\((Absolute|Independent|Term Base|Zero-Based|VR Max)\)',
                        linearity_text, re.IGNORECASE
                    )
                    if type_match:
                        linearity_type = type_match.group(1)
                    elif any(kw in linearity_text.lower() for kw in
                             ['see chart', 'see table', 'function', 'trim according']):
                        linearity_type = "Custom"

                    # Extract percentage: look for ± N.N% or +/- N.N%
                    pct_match = re.search(r'[±]\s*(\d+\.?\d*)\s*%', linearity_text)
                    if not pct_match:
                        pct_match = re.search(r'\+/?-?\s*\.?(\d+\.?\d*)\s*%', linearity_text)
                    if pct_match:
                        try:
                            linearity_pct = float(pct_match.group(1))
                        except ValueError:
                            pass

                # Parse resistance: "950 - 1,050 Ω" → min=950, max=1050
                r_min = None
                r_max = None
                if resistance_text:
                    r_match = re.search(
                        r'([\d,]+\.?\d*)\s*[-–]\s*([\d,]+\.?\d*)',
                        resistance_text
                    )
                    if r_match:
                        try:
                            r_min = float(r_match.group(1).replace(',', ''))
                            r_max = float(r_match.group(2).replace(',', ''))
                        except ValueError:
                            pass

                # Parse angle: '1.31" ± .005"' or '240° ± 2°'
                angle_val = None
                angle_tol = None
                angle_unit = None
                if angle_text:
                    # Try inches format: N.NN" ± .NNN"
                    a_match = re.search(r'([\d.]+)"?\s*[±]\s*\.?([\d.]+)', angle_text)
                    if a_match:
                        try:
                            angle_val = float(a_match.group(1))
                            tol_str = a_match.group(2)
                            angle_tol = float('0.' + tol_str) if '.' not in tol_str else float(tol_str)
                            angle_unit = "in"
                        except ValueError:
                            pass
                    if angle_val is None:
                        # Try just a number (degrees or inches)
                        num_match = re.search(r'([\d.]+)', angle_text)
                        if num_match:
                            try:
                                angle_val = float(num_match.group(1))
                                angle_unit = "in" if '"' in angle_text else "deg"
                            except ValueError:
                                pass

                model_data[model] = {
                    "model": model,
                    "element_type": element_type if element_type and element_type != 'None' else None,
                    "product_class": product_class if product_class and product_class != 'None' else None,
                    "linearity_type": linearity_type,
                    "linearity_spec_text": linearity_text if linearity_text and linearity_text != 'None' else None,
                    "linearity_spec_pct": linearity_pct,
                    "total_resistance_min": r_min,
                    "total_resistance_max": r_max,
                    "electrical_angle": angle_val,
                    "electrical_angle_tol": angle_tol,
                    "electrical_angle_unit": angle_unit,
                    "output_smoothness": smoothness if smoothness and smoothness != 'None' else None,
                    "circuit_type": circuit if circuit and circuit != 'None' else None,
                }

        # Sheet 2: Element Type (supplement — broader coverage)
        if "Element Type" in wb.sheetnames:
            ws = wb["Element Type"]
            for row in ws.iter_rows(min_row=2, values_only=True):
                model = str(row[0]).strip() if row[0] else None
                etype = str(row[1]).strip() if row[1] else None
                if model and etype and etype != 'None':
                    if model not in model_data:
                        model_data[model] = {"model": model, "element_type": etype}
                    elif not model_data[model].get("element_type"):
                        model_data[model]["element_type"] = etype

        # Sheet 3: Product Class (supplement — broadest coverage)
        if "Product Class" in wb.sheetnames:
            ws = wb["Product Class"]
            for row in ws.iter_rows(min_row=2, values_only=True):
                model = str(row[0]).strip() if row[0] else None
                pclass = str(row[1]).strip() if row[1] else None
                if model and pclass and pclass != 'None':
                    if model not in model_data:
                        model_data[model] = {"model": model, "product_class": pclass}
                    elif not model_data[model].get("product_class"):
                        model_data[model]["product_class"] = pclass

        wb.close()

        # Save to database (merge logic — save_model_spec handles upsert atomically)
        for model_name, data in model_data.items():
            try:
                _, was_update = self.save_model_spec(data)
                if was_update:
                    result["updated"] += 1
                else:
                    result["added"] += 1
            except Exception as e:
                logger.warning(f"Skipping model spec {model_name}: {e}")
                result["skipped"] += 1

        logger.info(
            f"Model specs import: {result['added']} added, "
            f"{result['updated']} updated, {result['skipped']} skipped"
        )
        return result

    # =========================================================================
    # Output Smoothness Methods
    # =========================================================================

    def save_smoothness_result(
        self, metadata: Dict[str, Any], tracks: List[Dict[str, Any]], file_hash: str
    ) -> int:
        """Save an Output Smoothness result. Returns ID."""
        from laser_trim_analyzer.database.models import (
            SmoothnessResult as DBSmoothnessResult,
            SmoothnessTrack as DBSmoothnessTrack,
        )

        with self._write_lock:
            try:
                with self.session() as session:
                    existing = session.query(DBSmoothnessResult).filter(
                        DBSmoothnessResult.file_hash == file_hash
                    ).first()
                    if existing:
                        return existing.id

                    overall_status = DBStatusType.PASS
                    for track in tracks:
                        if track.get("smoothness_pass") is False:
                            overall_status = DBStatusType.FAIL
                            break

                    max_smooth = max((t.get("max_smoothness", 0) or 0 for t in tracks), default=0)
                    avg_smooth = sum(t.get("avg_smoothness", 0) or 0 for t in tracks) / len(tracks) if tracks else 0
                    spec = metadata.get("smoothness_spec") or (tracks[0].get("smoothness_spec") if tracks else None)
                    passes = all(t.get("smoothness_pass", True) for t in tracks) if tracks else None

                    linked_trim_id, match_confidence, days_since_trim, match_method = self._find_matching_trim(
                        session, metadata.get("model"), metadata.get("serial"),
                        metadata.get("file_date") or metadata.get("test_date")
                    )

                    db_result = DBSmoothnessResult(
                        filename=metadata.get("filename", "unknown"),
                        file_path=str(metadata.get("file_path", "")),
                        file_hash=file_hash,
                        file_date=metadata.get("file_date"),
                        model=metadata.get("model", "unknown"),
                        serial=metadata.get("serial", "unknown"),
                        element_label=metadata.get("element_label"),
                        test_date=metadata.get("test_date"),
                        overall_status=overall_status,
                        smoothness_spec=spec,
                        max_smoothness_value=max_smooth,
                        avg_smoothness_value=avg_smooth,
                        smoothness_pass=passes,
                        linked_trim_id=linked_trim_id,
                        match_confidence=match_confidence,
                        match_method=match_method,
                        days_since_trim=days_since_trim,
                    )
                    session.add(db_result)
                    session.flush()
                    result_id = db_result.id

                    for track_data in tracks:
                        db_track = DBSmoothnessTrack(
                            smoothness_id=result_id,
                            track_id=track_data.get("track_id", "default"),
                            status=DBStatusType.PASS if track_data.get("smoothness_pass", True) else DBStatusType.FAIL,
                            smoothness_spec=track_data.get("smoothness_spec"),
                            max_smoothness=track_data.get("max_smoothness"),
                            avg_smoothness=track_data.get("avg_smoothness"),
                            smoothness_pass=track_data.get("smoothness_pass"),
                            position_data=track_data.get("positions"),
                            smoothness_data=track_data.get("smoothness_values"),
                        )
                        session.add(db_track)

                    logger.info(f"Saved Smoothness: {metadata.get('filename')} (ID: {result_id})")
                    return result_id

            except IntegrityError:
                logger.warning(f"Smoothness duplicate: {metadata.get('filename')}")
                with self.session() as session:
                    existing = session.query(DBSmoothnessResult).filter(
                        DBSmoothnessResult.file_hash == file_hash
                    ).first()
                    return existing.id if existing else -1

    def search_smoothness_results(
        self, model: Optional[str] = None, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Search Output Smoothness results."""
        from laser_trim_analyzer.database.models import SmoothnessResult as DBSmoothnessResult

        with self.session() as session:
            query = session.query(DBSmoothnessResult)
            if model and model != "All Models":
                query = query.filter(DBSmoothnessResult.model == model)
            results = query.order_by(desc(DBSmoothnessResult.file_date)).limit(limit).all()
            return [
                {
                    "id": r.id, "filename": r.filename, "model": r.model,
                    "serial": r.serial, "element_label": r.element_label,
                    "file_date": r.file_date, "test_date": r.test_date,
                    "overall_status": r.overall_status.value if r.overall_status else "UNKNOWN",
                    "smoothness_spec": r.smoothness_spec,
                    "max_smoothness_value": r.max_smoothness_value,
                    "smoothness_pass": r.smoothness_pass,
                    "linked_trim_id": r.linked_trim_id,
                    "match_confidence": r.match_confidence,
                    "match_method": r.match_method,
                }
                for r in results
            ]

    def get_smoothness_result(self, result_id: int) -> Optional[Dict[str, Any]]:
        """Get a single Output Smoothness result by ID with tracks."""
        from laser_trim_analyzer.database.models import (
            SmoothnessResult as DBSmoothnessResult,
            SmoothnessTrack as DBSmoothnessTrack,
        )
        with self.session() as session:
            result = session.query(DBSmoothnessResult).filter(
                DBSmoothnessResult.id == result_id
            ).first()
            if not result:
                return None
            tracks = session.query(DBSmoothnessTrack).filter(
                DBSmoothnessTrack.smoothness_id == result_id
            ).all()
            return {
                "id": result.id, "filename": result.filename,
                "model": result.model, "serial": result.serial,
                "element_label": result.element_label,
                "file_date": result.file_date, "test_date": result.test_date,
                "overall_status": result.overall_status.value if result.overall_status else "UNKNOWN",
                "smoothness_spec": result.smoothness_spec,
                "max_smoothness_value": result.max_smoothness_value,
                "smoothness_pass": result.smoothness_pass,
                "linked_trim_id": result.linked_trim_id,
                "match_method": result.match_method,
                "match_confidence": result.match_confidence,
                "tracks": [
                    {
                        "track_id": t.track_id,
                        "smoothness_spec": t.smoothness_spec,
                        "max_smoothness": t.max_smoothness,
                        "smoothness_pass": t.smoothness_pass,
                        "positions": t.position_data or [],
                        "smoothness_values": t.smoothness_data or [],
                    }
                    for t in tracks
                ],
            }

    def get_smoothness_stats(self, days_back: int = 90) -> Dict[str, Any]:
        """Get Output Smoothness dashboard statistics."""
        from laser_trim_analyzer.database.models import SmoothnessResult as DBSmoothnessResult
        with self.session() as session:
            cutoff = datetime.now() - timedelta(days=days_back)
            total = session.query(func.count(DBSmoothnessResult.id)).filter(
                DBSmoothnessResult.file_date >= cutoff
            ).scalar() or 0
            if total == 0:
                return {"total": 0, "pass_rate": 0, "linked_count": 0, "link_rate": 0}
            passed = session.query(func.count(DBSmoothnessResult.id)).filter(
                DBSmoothnessResult.file_date >= cutoff,
                DBSmoothnessResult.smoothness_pass == True,
            ).scalar() or 0
            linked = session.query(func.count(DBSmoothnessResult.id)).filter(
                DBSmoothnessResult.file_date >= cutoff,
                DBSmoothnessResult.linked_trim_id.isnot(None),
            ).scalar() or 0
            return {
                "total": total,
                "pass_rate": round(passed / total * 100, 1),
                "linked_count": linked,
                "link_rate": round(linked / total * 100, 1),
            }

    def get_spec_discrepancies(self, tolerance_pct: float = 5.0) -> List[Dict[str, Any]]:
        """
        Compare file-parsed linearity specs against model_specs reference.

        Flags models where the spec parsed from trim files differs from the
        engineering reference by more than tolerance_pct percent.

        Returns:
            List of dicts with model, file_spec, reference_spec, difference_pct
        """
        results = []

        with self.session() as session:
            file_specs = (
                session.query(
                    DBAnalysisResult.model,
                    func.avg(DBTrackResult.linearity_spec).label('avg_file_spec'),
                    func.min(DBTrackResult.linearity_spec).label('min_file_spec'),
                    func.max(DBTrackResult.linearity_spec).label('max_file_spec'),
                    func.count(DBTrackResult.id).label('sample_count'),
                )
                .join(DBTrackResult, DBAnalysisResult.id == DBTrackResult.analysis_id)
                .filter(
                    DBTrackResult.linearity_spec.isnot(None),
                    DBTrackResult.linearity_spec > 0,
                )
                .group_by(DBAnalysisResult.model)
                .all()
            )

            for row in file_specs:
                ref = session.query(ModelSpec).filter(
                    ModelSpec.model == row.model
                ).first()

                if not ref or not ref.linearity_spec_pct:
                    continue

                ref_spec = ref.linearity_spec_pct / 100.0  # Convert % to decimal
                file_spec = row.avg_file_spec

                if ref_spec > 0:
                    diff_pct = abs(file_spec - ref_spec) / ref_spec * 100
                else:
                    diff_pct = 0

                if diff_pct > tolerance_pct:
                    results.append({
                        "model": row.model,
                        "file_spec_avg": round(file_spec, 6),
                        "file_spec_min": round(row.min_file_spec, 6),
                        "file_spec_max": round(row.max_file_spec, 6),
                        "reference_spec_pct": ref.linearity_spec_pct,
                        "reference_spec_decimal": round(ref_spec, 6),
                        "difference_pct": round(diff_pct, 1),
                        "sample_count": row.sample_count,
                        "linearity_type": ref.linearity_type,
                    })

        results.sort(key=lambda x: x["difference_pct"], reverse=True)
        return results


# Global instance for convenience
_db_manager: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def reset_database() -> None:
    """Reset the global database manager (for testing)."""
    global _db_manager
    if _db_manager:
        _db_manager.close()
    _db_manager = None
