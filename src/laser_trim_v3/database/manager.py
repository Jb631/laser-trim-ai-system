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
from typing import Dict, List, Optional, Any, Union, Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, func, and_, or_, desc, text, case
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import IntegrityError, OperationalError

from laser_trim_v3.database.models import (
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
from laser_trim_v3.core.models import (
    AnalysisResult,
    TrackData,
    AnalysisStatus,
    SystemType,
    RiskCategory,
)
from laser_trim_v3.config import get_config

logger = logging.getLogger(__name__)


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
        self._engine = create_engine(
            self.database_url,
            echo=False,
            poolclass=StaticPool,  # Good for SQLite
            connect_args={"check_same_thread": False},
        )

        # Create session factory
        self._SessionFactory = sessionmaker(bind=self._engine)

        # Thread-local session storage
        self._thread_local = threading.local()

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Create all tables if they don't exist."""
        try:
            Base.metadata.create_all(self._engine, checkfirst=True)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")

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
            Database ID of the saved analysis
        """
        with self.session() as session:
            # Map Pydantic model to SQLAlchemy model
            db_analysis = self._map_analysis_to_db(analysis)

            try:
                session.add(db_analysis)
                session.flush()  # Get ID before commit

                # Record as processed file
                self._record_processed_file(
                    session,
                    analysis.metadata.file_path,
                    db_analysis.id
                )

                logger.info(f"Saved analysis: {analysis.metadata.filename} (ID: {db_analysis.id})")
                return db_analysis.id

            except IntegrityError as e:
                # Handle duplicate - update existing record
                logger.warning(f"Duplicate file, updating: {analysis.metadata.filename}")
                session.rollback()
                return self._update_existing_analysis(session, analysis)

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
                db_analysis = self._map_analysis_to_db(analysis)

                try:
                    session.add(db_analysis)
                    session.flush()

                    self._record_processed_file(
                        session,
                        analysis.metadata.file_path,
                        db_analysis.id
                    )

                    saved_ids.append(db_analysis.id)

                except IntegrityError:
                    session.rollback()
                    # Update existing
                    updated_id = self._update_existing_analysis(session, analysis)
                    saved_ids.append(updated_id)

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

        Args:
            model: Filter by model number (optional)
            days_back: How many days back to query
            limit: Maximum number of results

        Returns:
            List of AnalysisResult objects
        """
        with self.session() as session:
            query = session.query(DBAnalysisResult)

            # Apply filters
            cutoff_date = datetime.now() - timedelta(days=days_back)
            query = query.filter(DBAnalysisResult.timestamp >= cutoff_date)

            if model:
                query = query.filter(DBAnalysisResult.model == model)

            # Order by newest first and limit
            query = query.order_by(desc(DBAnalysisResult.timestamp)).limit(limit)

            results = []
            for db_analysis in query.all():
                mapped = self._map_db_to_analysis(db_analysis)
                if mapped is not None:  # Filter out failed mappings
                    results.append(mapped)

            return results

    def get_model_statistics(self, model: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Get aggregated statistics for a model.

        Args:
            model: Model number
            days_back: Days to include in statistics

        Returns:
            Dictionary with statistics
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Query track results for this model
            results = (
                session.query(DBTrackResult)
                .join(DBAnalysisResult)
                .filter(
                    DBAnalysisResult.model == model,
                    DBAnalysisResult.timestamp >= cutoff_date
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
        Check if a file has already been processed.

        Uses SHA-256 hash for accurate duplicate detection even if file
        was moved or renamed.

        Args:
            file_path: Path to the file

        Returns:
            True if file was already processed
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return False

        file_hash = self._calculate_file_hash(file_path)

        with self.session() as session:
            exists = (
                session.query(DBProcessedFile)
                .filter(DBProcessedFile.file_hash == file_hash)
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

        Args:
            days_back: Number of days to include

        Returns:
            Dictionary with dashboard statistics
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Count analyses
            total_analyses = (
                session.query(func.count(DBAnalysisResult.id))
                .filter(DBAnalysisResult.timestamp >= cutoff_date)
                .scalar()
            ) or 0

            # Count by status
            status_counts = (
                session.query(
                    DBAnalysisResult.overall_status,
                    func.count(DBAnalysisResult.id)
                )
                .filter(DBAnalysisResult.timestamp >= cutoff_date)
                .group_by(DBAnalysisResult.overall_status)
                .all()
            )

            passed = 0
            failed = 0
            for status, count in status_counts:
                if status == DBStatusType.PASS:
                    passed = count
                elif status in (DBStatusType.FAIL, DBStatusType.ERROR):
                    failed = count

            # Count unresolved alerts
            unresolved_alerts = (
                session.query(func.count(DBQAAlert.id))
                .filter(DBQAAlert.resolved == False)
                .scalar()
            ) or 0

            # Get high-risk count
            high_risk = (
                session.query(func.count(DBTrackResult.id))
                .join(DBAnalysisResult)
                .filter(
                    DBAnalysisResult.timestamp >= cutoff_date,
                    DBTrackResult.risk_category == DBRiskCategory.HIGH
                )
                .scalar()
            ) or 0

            pass_rate = (passed / total_analyses * 100) if total_analyses > 0 else 0.0

            # Total files (all time)
            total_files = (
                session.query(func.count(DBAnalysisResult.id))
                .scalar()
            ) or 0

            # Today's count
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_count = (
                session.query(func.count(DBAnalysisResult.id))
                .filter(DBAnalysisResult.timestamp >= today)
                .scalar()
            ) or 0

            # This week's count
            week_start = today - timedelta(days=today.weekday())
            week_count = (
                session.query(func.count(DBAnalysisResult.id))
                .filter(DBAnalysisResult.timestamp >= week_start)
                .scalar()
            ) or 0

            # Daily trend for the past N days
            daily_trend = []
            for i in range(days_back):
                day_start = (today - timedelta(days=days_back - 1 - i))
                day_end = day_start + timedelta(days=1)

                day_total = (
                    session.query(func.count(DBAnalysisResult.id))
                    .filter(
                        DBAnalysisResult.timestamp >= day_start,
                        DBAnalysisResult.timestamp < day_end
                    )
                    .scalar()
                ) or 0

                day_passed = (
                    session.query(func.count(DBAnalysisResult.id))
                    .filter(
                        DBAnalysisResult.timestamp >= day_start,
                        DBAnalysisResult.timestamp < day_end,
                        DBAnalysisResult.overall_status == DBStatusType.PASS
                    )
                    .scalar()
                ) or 0

                day_pass_rate = (day_passed / day_total * 100) if day_total > 0 else 0.0
                daily_trend.append({
                    "date": day_start.strftime("%m/%d"),
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
                "unresolved_alerts": unresolved_alerts,
                "high_risk_count": high_risk,
                "period_days": days_back,
                "today_count": today_count,
                "week_count": week_count,
                "daily_trend": daily_trend,
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
            query = session.query(DBQAAlert)

            if not include_resolved:
                query = query.filter(DBQAAlert.resolved == False)

            alerts = (
                query
                .order_by(DBQAAlert.created_date.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "id": a.id,
                    "analysis_id": a.analysis_id,
                    "alert_type": a.alert_type.value if a.alert_type else "INFO",
                    "severity": a.severity,
                    "message": a.message,
                    "model": a.model,
                    "created_at": a.created_date.strftime("%Y-%m-%d %H:%M") if a.created_date else "",
                    "acknowledged": a.acknowledged,
                    "resolved": a.resolved,
                }
                for a in alerts
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
            # Get counts by model
            model_counts = (
                session.query(
                    DBAnalysisResult.model,
                    func.count(DBAnalysisResult.id).label('count')
                )
                .group_by(DBAnalysisResult.model)
                .order_by(func.count(DBAnalysisResult.id).desc())
                .limit(limit)
                .all()
            )

            result = []
            for model, count in model_counts:
                if not model:
                    continue

                # Get pass rate for this model
                passed = (
                    session.query(func.count(DBAnalysisResult.id))
                    .filter(
                        DBAnalysisResult.model == model,
                        DBAnalysisResult.overall_status == DBStatusType.PASS
                    )
                    .scalar()
                ) or 0

                pass_rate = (passed / count * 100) if count > 0 else 0.0

                result.append({
                    "model": model,
                    "count": count,
                    "passed": passed,
                    "failed": count - passed,
                    "pass_rate": pass_rate,
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

        Args:
            model: Filter by model number (None for all)
            days_back: Number of days to include
            limit: Maximum number of records

        Returns:
            List of trend data dictionaries
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            query = (
                session.query(
                    DBAnalysisResult.timestamp,
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
                .filter(DBAnalysisResult.timestamp >= cutoff_date)
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
                    # Use file_date (trim date) if available, fallback to timestamp (processing date)
                    "date": (r.file_date or r.timestamp).strftime("%Y-%m-%d") if (r.file_date or r.timestamp) else "",
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
        from laser_trim_v3.core.models import FileMetadata

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

            # Clear old tracks and add new ones
            existing.tracks.clear()
            for track in analysis.tracks:
                db_track = self._map_track_to_db(track)
                existing.tracks.append(db_track)

            return existing.id

        # If no existing record found, create new
        db_analysis = self._map_analysis_to_db(analysis)
        session.add(db_analysis)
        session.flush()
        return db_analysis.id

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

        Args:
            days_back: Only include models with files in this period
            min_samples: Minimum samples required for inclusion

        Returns:
            List of model summaries sorted by sample count descending
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Get all models with recent activity
            model_data = (
                session.query(
                    DBAnalysisResult.model,
                    func.count(DBAnalysisResult.id).label('total'),
                    func.sum(
                        case(
                            (DBAnalysisResult.overall_status == DBStatusType.PASS, 1),
                            else_=0
                        )
                    ).label('passed'),
                    func.min(DBAnalysisResult.file_date).label('first_date'),
                    func.max(DBAnalysisResult.file_date).label('last_date'),
                )
                .filter(
                    DBAnalysisResult.model.isnot(None),
                    or_(
                        DBAnalysisResult.file_date >= cutoff_date,
                        DBAnalysisResult.timestamp >= cutoff_date
                    )
                )
                .group_by(DBAnalysisResult.model)
                .having(func.count(DBAnalysisResult.id) >= min_samples)
                .all()
            )

            results = []
            for model, total, passed, first_date, last_date in model_data:
                if not model or total == 0:
                    continue

                passed = passed or 0
                pass_rate = (passed / total * 100)

                # Get average sigma gradient for this model
                sigma_data = (
                    session.query(
                        func.avg(DBTrackResult.sigma_gradient),
                        func.avg(DBTrackResult.sigma_threshold),
                    )
                    .join(DBAnalysisResult)
                    .filter(
                        DBAnalysisResult.model == model,
                        or_(
                            DBAnalysisResult.file_date >= cutoff_date,
                            DBAnalysisResult.timestamp >= cutoff_date
                        ),
                        DBTrackResult.sigma_gradient.isnot(None),
                    )
                    .first()
                )

                avg_sigma = sigma_data[0] if sigma_data and sigma_data[0] else 0
                avg_threshold = sigma_data[1] if sigma_data and sigma_data[1] else 0

                results.append({
                    "model": model,
                    "total": total,
                    "passed": passed,
                    "failed": total - passed,
                    "pass_rate": pass_rate,
                    "avg_sigma": avg_sigma,
                    "avg_threshold": avg_threshold,
                    "first_date": first_date,
                    "last_date": last_date,
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

                # Check 1: Low pass rate
                if model_data["pass_rate"] < pass_rate_threshold:
                    alert_reasons.append({
                        "type": "LOW_PASS_RATE",
                        "message": f"Pass rate {model_data['pass_rate']:.1f}% is below {pass_rate_threshold}%",
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
                        or_(
                            and_(DBAnalysisResult.file_date >= older_cutoff, DBAnalysisResult.file_date < older_end),
                            and_(DBAnalysisResult.timestamp >= older_cutoff, DBAnalysisResult.timestamp < older_end)
                        )
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
                        or_(
                            DBAnalysisResult.file_date >= rolling_cutoff,
                            DBAnalysisResult.timestamp >= rolling_cutoff
                        )
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
                        or_(
                            DBAnalysisResult.file_date >= rolling_cutoff,
                            DBAnalysisResult.timestamp >= rolling_cutoff
                        ),
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

        Args:
            model: Model number
            days_back: Total days to include
            rolling_window: Days for rolling average

        Returns:
            Dict with trend data for charts
        """
        with self.session() as session:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Get all track results for this model
            results = (
                session.query(
                    DBAnalysisResult.file_date,
                    DBAnalysisResult.timestamp,
                    DBAnalysisResult.overall_status,
                    DBTrackResult.sigma_gradient,
                    DBTrackResult.sigma_threshold,
                    DBTrackResult.sigma_pass,
                )
                .join(DBTrackResult)
                .filter(
                    DBAnalysisResult.model == model,
                    or_(
                        DBAnalysisResult.file_date >= cutoff_date,
                        DBAnalysisResult.timestamp >= cutoff_date
                    ),
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
            for file_date, timestamp, status, sigma_gradient, sigma_threshold, sigma_pass in results:
                date = file_date or timestamp
                if date and sigma_gradient is not None:
                    data_points.append({
                        "date": date,
                        "sigma_gradient": sigma_gradient,
                        "sigma_threshold": sigma_threshold,
                        "sigma_pass": sigma_pass,
                        "status": status.value if status else "UNKNOWN",
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
