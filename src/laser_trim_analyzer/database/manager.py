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

from sqlalchemy import create_engine, func, and_, or_, desc, text, case
from sqlalchemy.orm import sessionmaker, Session, joinedload
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

    def _run_migrations(self) -> None:
        """Run database migrations for schema updates."""
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

            return exists

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

        # Query for existing hashes
        with self.session() as session:
            existing_hashes = set(
                row.file_hash for row in
                session.query(DBProcessedFile.file_hash)
                .filter(DBProcessedFile.file_hash.in_(list(file_hashes.keys())))
                .all()
            )

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

    def get_dashboard_stats(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get statistics for dashboard display.

        Filters by trim date (file_date), not processing date.

        Args:
            days_back: Number of days to include (based on trim date)

        Returns:
            Dictionary with dashboard statistics
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Count analyses - filter by trim date
            total_analyses = (
                session.query(func.count(DBAnalysisResult.id))
                .filter(DBAnalysisResult.file_date >= cutoff_date)
                .scalar()
            ) or 0

            # Count by status - filter by trim date
            status_counts = (
                session.query(
                    DBAnalysisResult.overall_status,
                    func.count(DBAnalysisResult.id)
                )
                .filter(DBAnalysisResult.file_date >= cutoff_date)
                .group_by(DBAnalysisResult.overall_status)
                .all()
            )

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
            high_risk = (
                session.query(func.count(DBTrackResult.id))
                .join(DBAnalysisResult)
                .filter(
                    DBAnalysisResult.file_date >= cutoff_date,
                    DBTrackResult.risk_category == DBRiskCategory.HIGH
                )
                .scalar()
            ) or 0

            pass_rate = (passed / total_analyses * 100) if total_analyses > 0 else 0.0

            # Get track-level sigma and linearity pass rates - filter by trim date
            track_stats = (
                session.query(
                    func.count(DBTrackResult.id).label('total_tracks'),
                    func.sum(case((DBTrackResult.sigma_pass == True, 1), else_=0)).label('sigma_passed'),
                    func.sum(case((DBTrackResult.linearity_pass == True, 1), else_=0)).label('linearity_passed'),
                )
                .join(DBAnalysisResult)
                .filter(DBAnalysisResult.file_date >= cutoff_date)
                .first()
            )

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
            today_count = (
                session.query(func.count(DBAnalysisResult.id))
                .filter(DBAnalysisResult.file_date >= today)
                .scalar()
            ) or 0

            # This week's count (by trim date)
            week_start = today - timedelta(days=today.weekday())
            week_count = (
                session.query(func.count(DBAnalysisResult.id))
                .filter(DBAnalysisResult.file_date >= week_start)
                .scalar()
            ) or 0

            # Daily trend for the past N days (by trim date) - optimized single query
            trend_start = today - timedelta(days=days_back - 1)

            # Single query with GROUP BY date to get all days at once
            daily_data = (
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
            }

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

    def get_models_list(self) -> List[str]:
        """Get list of all unique model numbers in database."""
        with self.session() as session:
            models = (
                session.query(DBAnalysisResult.model)
                .distinct()
                .order_by(DBAnalysisResult.model)
                .all()
            )
            return [m[0] for m in models if m[0]]

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
        )

    def _map_db_to_analysis(self, db_analysis: DBAnalysisResult) -> Optional[AnalysisResult]:
        """Map SQLAlchemy model back to Pydantic AnalysisResult.

        Returns None if the analysis has no valid tracks (corrupted data).
        """
        from laser_trim_analyzer.core.models import FileMetadata

        # Map system type
        system_type = SystemType.A if db_analysis.system == DBSystemType.A else SystemType.B

        # Map status
        status_map = {
            DBStatusType.PASS: AnalysisStatus.PASS,
            DBStatusType.FAIL: AnalysisStatus.FAIL,
            DBStatusType.WARNING: AnalysisStatus.WARNING,
            DBStatusType.ERROR: AnalysisStatus.ERROR,
        }
        overall_status = status_map.get(db_analysis.overall_status, AnalysisStatus.ERROR)

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
        status_map = {
            DBStatusType.PASS: AnalysisStatus.PASS,
            DBStatusType.FAIL: AnalysisStatus.FAIL,
            DBStatusType.WARNING: AnalysisStatus.WARNING,
            DBStatusType.ERROR: AnalysisStatus.ERROR,
        }
        status = status_map.get(db_track.status, AnalysisStatus.ERROR)

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
                linearity_error=db_track.final_linearity_error_shifted or 0.0,
                linearity_pass=db_track.linearity_pass if db_track.linearity_pass is not None else True,
                linearity_fail_points=db_track.linearity_fail_points or 0,
                unit_length=db_track.unit_length,
                untrimmed_resistance=db_track.untrimmed_resistance,
                trimmed_resistance=db_track.trimmed_resistance,
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
            List of model strings, sorted alphabetically
        """
        with self.session() as session:
            models = (
                session.query(DBAnalysisResult.model)
                .filter(DBAnalysisResult.model.isnot(None))
                .distinct()
                .order_by(DBAnalysisResult.model)
                .all()
            )
            return [m[0] for m in models if m[0]]

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

                if not model or total == 0:
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

            # Sort by sample count descending
            results.sort(key=lambda x: x["total"], reverse=True)
            return results

    def get_models_requiring_attention(
        self,
        days_back: int = 90,
        min_samples: int = 5,
        pass_rate_threshold: float = 80.0,
        trend_threshold: float = 10.0,
        rolling_days: int = 30
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

        Returns:
            List of models requiring attention with alert details
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            rolling_cutoff = datetime.now() - timedelta(days=rolling_days)

            # Get active models
            active_models = self.get_active_models_summary(days_back, min_samples)

            alerts = []
            for model_data in active_models:
                model = model_data["model"]
                alert_reasons = []

                # Check 1: Low pass rate (overall = both sigma and linearity must pass)
                if model_data["pass_rate"] < pass_rate_threshold:
                    alert_reasons.append({
                        "type": "LOW_PASS_RATE",
                        "message": f"Overall pass rate {model_data['pass_rate']:.1f}% is below {pass_rate_threshold}%",
                        "severity": "High" if model_data["pass_rate"] < 70 else "Medium"
                    })

                # Check 2: Trending worse
                # Compare older period vs rolling period
                older_cutoff = cutoff_date
                older_end = rolling_cutoff

                older_pass_rate = (
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

                recent_pass_rate = (
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

                older_count, older_passed = older_pass_rate
                recent_count, recent_passed = recent_pass_rate

                if older_count and older_count >= min_samples and recent_count and recent_count >= min_samples:
                    older_pct = (older_passed or 0) / older_count * 100
                    recent_pct = (recent_passed or 0) / recent_count * 100

                    if recent_pct < older_pct - trend_threshold:
                        alert_reasons.append({
                            "type": "TRENDING_WORSE",
                            "message": f"Pass rate dropped from {older_pct:.1f}% to {recent_pct:.1f}% ({older_pct - recent_pct:.1f}% decline)",
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
            for file_date, status, sigma_gradient, sigma_threshold, sigma_pass, is_anomaly in results:
                if file_date and sigma_gradient is not None:
                    data_points.append({
                        "date": file_date,
                        "sigma_gradient": sigma_gradient,
                        "sigma_threshold": sigma_threshold,
                        "sigma_pass": sigma_pass,
                        "status": status.value if status else "UNKNOWN",
                        "is_anomaly": is_anomaly or False,
                    })

            # Calculate threshold (use mode of thresholds)
            thresholds = [d["sigma_threshold"] for d in data_points if d["sigma_threshold"]]
            threshold = max(set(thresholds), key=thresholds.count) if thresholds else None

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
                "threshold": threshold,
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
                    linked_trim_id, match_confidence, days_since_trim = self._find_matching_trim(
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

    def _find_matching_trim(
        self,
        session: Session,
        model: Optional[str],
        serial: Optional[str],
        test_date: Optional[datetime]
    ) -> Tuple[Optional[int], Optional[float], Optional[int]]:
        """
        Find the matching trim result for a final test.

        Logic:
        - Same model and serial
        - Trim date < final test date
        - Closest trim date within 60 days

        Returns:
            Tuple of (trim_id, confidence, days_since_trim)
        """
        from laser_trim_analyzer.utils.constants import FINAL_TEST_MAX_DAYS_FROM_TRIM

        if not model or not serial or not test_date:
            return None, None, None

        # Normalize serial for matching (some have prefix variations)
        serial_clean = serial.lower().strip()

        # Query for matching trim results
        # Must be: same model, same serial, trim date <= test date (same day allowed), within 60 days
        cutoff_date = test_date - timedelta(days=FINAL_TEST_MAX_DAYS_FROM_TRIM)

        candidates = (
            session.query(DBAnalysisResult)
            .filter(
                DBAnalysisResult.model == model,
                func.lower(DBAnalysisResult.serial) == serial_clean,
                DBAnalysisResult.file_date.isnot(None),
                DBAnalysisResult.file_date <= test_date,  # Allow same-day matches
                DBAnalysisResult.file_date >= cutoff_date,
            )
            .order_by(desc(DBAnalysisResult.file_date))  # Most recent first
            .limit(5)
            .all()
        )

        if not candidates:
            # No match found - require both model AND serial to match
            return None, None, None

        # Found candidates with matching serial
        match = candidates[0]  # Most recent before test date
        days_diff = (test_date - match.file_date).days

        # Calculate confidence based on time proximity
        # 0-7 days: high confidence (0.9-1.0)
        # 8-30 days: medium confidence (0.7-0.9)
        # 31-60 days: low confidence (0.5-0.7)
        if days_diff <= 7:
            confidence = 1.0 - (days_diff * 0.01)  # 0.93-1.0
        elif days_diff <= 30:
            confidence = 0.9 - ((days_diff - 7) * 0.01)  # 0.67-0.9
        else:
            confidence = 0.7 - ((days_diff - 30) * 0.007)  # 0.5-0.7

        return match.id, max(0.5, confidence), days_diff

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
            }

    def get_final_test_models_list(self) -> List[str]:
        """
        Get list of unique models from Final Test results.
        """
        from laser_trim_analyzer.database.models import FinalTestResult as DBFinalTestResult

        with self.session() as session:
            models = (
                session.query(DBFinalTestResult.model)
                .filter(DBFinalTestResult.model.isnot(None))
                .distinct()
                .order_by(DBFinalTestResult.model)
                .all()
            )
            return [m[0] for m in models if m[0]]

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
