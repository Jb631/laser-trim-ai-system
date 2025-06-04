"""
Database Manager for Laser Trim Analyzer v2.

Handles all database operations with connection pooling, error handling,
and migration support. Designed for QA specialists to easily store and
retrieve potentiometer test results.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    Iterator
)
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, func, and_, or_, desc, inspect
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine

# Import our database models
from .models import (
    Base, AnalysisResult as DBAnalysisResult, TrackResult as DBTrackResult,
    MLPrediction as DBMLPrediction, QAAlert as DBQAAlert, BatchInfo as DBBatchInfo,
    AnalysisBatch as DBAnalysisBatch,
    SystemType as DBSystemType, StatusType as DBStatusType,
    RiskCategory as DBRiskCategory, AlertType as DBAlertType
)

# Import Pydantic models from core
from ..core.models import (
    AnalysisResult as PydanticAnalysisResult,
    TrackData, AnalysisStatus, SystemType, RiskCategory
)


class DatabaseManager:
    """
    Manages all database operations for the Laser Trim Analyzer.

    Features:
    - Connection pooling for better performance
    - Context managers for safe transaction handling
    - Comprehensive error handling and logging
    - Migration support ready
    - Optimized queries with proper indexing
    """

    def __init__(
            self,
            database_url_or_config: Optional[Union[str, Any]] = None,  # Can be string or Config object
            echo: bool = False,
            pool_size: int = 5,
            max_overflow: int = 10,
            pool_timeout: int = 30,
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the database manager.

        Args:
            database_url_or_config: SQLAlchemy database URL string or Config object. If None, uses SQLite default.
            echo: If True, log all SQL statements (useful for debugging)
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum overflow connections allowed
            pool_timeout: Timeout in seconds for getting connection from pool
            logger: Logger instance for database operations
        """
        self.logger = logger or logging.getLogger(__name__)

        # Handle Config object or string URL
        if database_url_or_config is None:
            # Default to SQLite in user's home directory
            db_dir = Path.home() / ".laser_trim_analyzer"
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / "analyzer_v2.db"
            database_url = f"sqlite:///{db_path}"
            self.logger.info(f"Using SQLite database at: {db_path}")
        elif hasattr(database_url_or_config, 'database'):
            # It's a Config object
            config = database_url_or_config
            if hasattr(config.database, 'url') and config.database.url:
                database_url = config.database.url
            else:
                # Use file path from config
                db_path = Path(config.database.path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                database_url = f"sqlite:///{db_path.absolute()}"
                self.logger.info(f"Using SQLite database from config at: {db_path}")
        else:
            # It's a string URL
            database_url = database_url_or_config

        self.database_url = database_url

        # Create engine with connection pooling
        engine_kwargs = {
            "echo": echo,
            "future": True,  # Use SQLAlchemy 2.0 style
        }

        # Configure pooling based on database type
        if database_url.startswith("sqlite"):
            # SQLite doesn't benefit from connection pooling
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        else:
            # Use connection pooling for other databases
            engine_kwargs["poolclass"] = QueuePool
            engine_kwargs["pool_size"] = pool_size
            engine_kwargs["max_overflow"] = max_overflow
            engine_kwargs["pool_timeout"] = pool_timeout
            engine_kwargs["pool_pre_ping"] = True  # Verify connections before use

        self.engine = create_engine(database_url, **engine_kwargs)

        # Create session factory
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autoflush=False
        )

        # Create scoped session for thread safety
        self.Session = scoped_session(self.session_factory)

        self.logger.info("Database manager initialized successfully")

    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """
        Context manager for database sessions.

        Ensures proper session cleanup and error handling.

        Usage:
            with db_manager.get_session() as session:
                # Perform database operations
                session.add(record)
                session.commit()
        """
        session = self.Session()
        try:
            yield session
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Database error: {str(e)}")
            raise
        except Exception as e:
            session.rollback()
            self.logger.error(f"Unexpected error: {str(e)}")
            raise
        finally:
            session.close()

    def init_db(self, drop_existing: bool = False) -> None:
        """
        Initialize database tables.

        Args:
            drop_existing: If True, drop all existing tables first (careful!)
        """
        try:
            if drop_existing:
                self.logger.warning("Dropping all existing tables...")
                Base.metadata.drop_all(self.engine)

            self.logger.info("Creating database tables...")
            Base.metadata.create_all(self.engine)

            # Verify tables were created
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            self.logger.info(f"Created {len(tables)} tables: {', '.join(tables)}")

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def initialize_schema(self) -> None:
        """
        Initialize database schema (alias for init_db).
        
        This method ensures all tables are created if they don't exist.
        """
        self.init_db(drop_existing=False)

    def check_duplicate_analysis(
            self,
            model: str,
            serial: str,
            file_date: datetime
    ) -> Optional[int]:
        """
        Check if this unit has already been analyzed.

        Args:
            model: Model number
            serial: Serial number
            file_date: Date from the file (not the analysis date)

        Returns:
            ID of existing analysis if found, None otherwise
        """
        with self.get_session() as session:
            # Check for exact match (model + serial + file date)
            existing = session.query(DBAnalysisResult).filter(
                and_(
                    DBAnalysisResult.model == model,
                    DBAnalysisResult.serial == serial,
                    DBAnalysisResult.file_date == file_date
                )
            ).first()
            
            if existing:
                self.logger.info(
                    f"Found duplicate analysis for {model}-{serial} from {file_date}: "
                    f"ID {existing.id}"
                )
                return existing.id
            
            return None

    def save_analysis(self, analysis_data: PydanticAnalysisResult) -> int:
        """
        Save a complete analysis result with all tracks.

        Args:
            analysis_data: Pydantic model containing analysis results

        Returns:
            ID of the saved analysis record
        """
        with self.get_session() as session:
            try:
                # Convert Pydantic SystemType to DB SystemType
                system_type = DBSystemType.A if analysis_data.metadata.system == SystemType.SYSTEM_A else DBSystemType.B

                # Convert Pydantic AnalysisStatus to DB StatusType
                status_map = {
                    AnalysisStatus.PASS: DBStatusType.PASS,
                    AnalysisStatus.FAIL: DBStatusType.FAIL,
                    AnalysisStatus.WARNING: DBStatusType.WARNING,
                    AnalysisStatus.ERROR: DBStatusType.ERROR,
                    AnalysisStatus.PENDING: DBStatusType.PROCESSING_FAILED
                }
                overall_status = status_map.get(analysis_data.overall_status, DBStatusType.ERROR)

                # Create main analysis record
                analysis = DBAnalysisResult(
                    filename=analysis_data.metadata.filename,
                    file_path=str(analysis_data.metadata.file_path),
                    file_date=analysis_data.metadata.file_date,
                    file_hash=None,  # Calculate if needed
                    model=analysis_data.metadata.model,
                    serial=analysis_data.metadata.serial,
                    system=system_type,
                    has_multi_tracks=analysis_data.metadata.has_multi_tracks,
                    overall_status=overall_status,
                    processing_time=analysis_data.processing_time,
                    output_dir=None,  # Set if available
                    software_version="2.0.0",  # Set from config
                    operator=None,  # Set if available
                    sigma_scaling_factor=None,  # Set from first track if available
                    filter_cutoff_frequency=None  # Set from config if available
                )

                # Add track results
                for track_id, track_data in analysis_data.tracks.items():
                    # Convert track status
                    track_status = status_map.get(track_data.status, DBStatusType.ERROR)

                    # Convert risk category if available
                    risk_category = None
                    if track_data.failure_prediction:
                        risk_map = {
                            RiskCategory.HIGH: DBRiskCategory.HIGH,
                            RiskCategory.MEDIUM: DBRiskCategory.MEDIUM,
                            RiskCategory.LOW: DBRiskCategory.LOW,
                            RiskCategory.UNKNOWN: DBRiskCategory.UNKNOWN
                        }
                        risk_category = risk_map.get(track_data.failure_prediction.risk_category)

                    track = DBTrackResult(
                        track_id=track_id,
                        status=track_status,
                        travel_length=track_data.travel_length,
                        linearity_spec=track_data.linearity_analysis.linearity_spec,
                        sigma_gradient=track_data.sigma_analysis.sigma_gradient,
                        sigma_threshold=track_data.sigma_analysis.sigma_threshold,
                        sigma_pass=track_data.sigma_analysis.sigma_pass,
                        unit_length=track_data.unit_properties.unit_length,
                        untrimmed_resistance=track_data.unit_properties.untrimmed_resistance,
                        trimmed_resistance=track_data.unit_properties.trimmed_resistance,
                        resistance_change=track_data.unit_properties.resistance_change,
                        resistance_change_percent=track_data.unit_properties.resistance_change_percent,
                        optimal_offset=track_data.linearity_analysis.optimal_offset,
                        final_linearity_error_raw=track_data.linearity_analysis.final_linearity_error_raw,
                        final_linearity_error_shifted=track_data.linearity_analysis.final_linearity_error_shifted,
                        linearity_pass=track_data.linearity_analysis.linearity_pass,
                        linearity_fail_points=track_data.linearity_analysis.linearity_fail_points,
                        max_deviation=track_data.linearity_analysis.max_deviation,
                        max_deviation_position=track_data.linearity_analysis.max_deviation_position,
                        deviation_uniformity=None,  # Calculate if needed
                        trim_improvement_percent=track_data.trim_effectiveness.improvement_percent if track_data.trim_effectiveness else None,
                        untrimmed_rms_error=track_data.trim_effectiveness.untrimmed_rms_error if track_data.trim_effectiveness else None,
                        trimmed_rms_error=track_data.trim_effectiveness.trimmed_rms_error if track_data.trim_effectiveness else None,
                        max_error_reduction_percent=track_data.trim_effectiveness.max_error_reduction_percent if track_data.trim_effectiveness else None,
                        worst_zone=track_data.zone_analysis.worst_zone if track_data.zone_analysis else None,
                        worst_zone_position=track_data.zone_analysis.worst_zone_position[
                            0] if track_data.zone_analysis and track_data.zone_analysis.worst_zone_position else None,
                        zone_details=track_data.zone_analysis.zone_results if track_data.zone_analysis else None,
                        failure_probability=track_data.failure_prediction.failure_probability if track_data.failure_prediction else None,
                        risk_category=risk_category,
                        gradient_margin=track_data.sigma_analysis.gradient_margin,
                        range_utilization_percent=track_data.dynamic_range.range_utilization_percent if track_data.dynamic_range else None,
                        minimum_margin=track_data.dynamic_range.minimum_margin if track_data.dynamic_range else None,
                        minimum_margin_position=track_data.dynamic_range.minimum_margin_position if track_data.dynamic_range else None,
                        margin_bias=track_data.dynamic_range.margin_bias if track_data.dynamic_range else None,
                        plot_path=str(track_data.plot_path) if track_data.plot_path else None
                    )
                    analysis.tracks.append(track)

                # Check for alerts
                self._generate_alerts(analysis, session)

                session.add(analysis)
                session.commit()

                self.logger.info(
                    f"Saved analysis for {analysis_data.metadata.filename} with {len(analysis.tracks)} tracks")
                return analysis.id

            except Exception as e:
                self.logger.error(f"Failed to save analysis: {str(e)}")
                raise

    def get_historical_data(
            self,
            model: Optional[str] = None,
            serial: Optional[str] = None,
            days_back: Optional[int] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            status: Optional[str] = None,
            risk_category: Optional[str] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            include_tracks: bool = True
    ) -> List[DBAnalysisResult]:
        """
        Retrieve historical analysis data with flexible filtering.

        Args:
            model: Filter by model number (supports wildcards with %)
            serial: Filter by serial number (supports wildcards with %)
            days_back: Number of days to look back from today
            start_date: Start date for date range filter
            end_date: End date for date range filter
            status: Filter by overall status
            risk_category: Filter by risk category in tracks
            limit: Maximum number of records to return
            offset: Number of records to skip (for pagination)
            include_tracks: Whether to include track details

        Returns:
            List of AnalysisResult objects
        """
        with self.get_session() as session:
            query = session.query(DBAnalysisResult)

            # Apply filters
            if model:
                query = query.filter(DBAnalysisResult.model.like(model))

            if serial:
                query = query.filter(DBAnalysisResult.serial.like(serial))

            # Date filtering
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(DBAnalysisResult.timestamp >= cutoff_date)
            elif start_date:
                query = query.filter(DBAnalysisResult.timestamp >= start_date)
                if end_date:
                    query = query.filter(DBAnalysisResult.timestamp <= end_date)

            if status:
                query = query.filter(DBAnalysisResult.overall_status == DBStatusType(status))

            if risk_category and include_tracks:
                # Join with tracks to filter by risk category
                query = query.join(DBTrackResult).filter(
                    DBTrackResult.risk_category == DBRiskCategory(risk_category)
                ).distinct()

            # Order by most recent first
            query = query.order_by(desc(DBAnalysisResult.timestamp))

            # Apply pagination
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)

            # Execute query
            results = query.all()

            # Optionally load tracks (eager loading)
            if include_tracks:
                for result in results:
                    # Access tracks to trigger loading
                    _ = result.tracks

            self.logger.info(f"Retrieved {len(results)} historical records")
            return results

    def get_model_statistics(self, model: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a specific model.

        Args:
            model: Model number to analyze

        Returns:
            Dictionary containing model statistics
        """
        with self.get_session() as session:
            # Base query for the model
            base_query = session.query(DBAnalysisResult).filter(
                DBAnalysisResult.model == model
            )

            # Get basic counts
            total_files = base_query.count()

            if total_files == 0:
                return {
                    "model": model,
                    "total_files": 0,
                    "total_tracks": 0,
                    "statistics": {}
                }

            # Get track statistics
            track_stats = session.query(
                func.count(DBTrackResult.id).label('total_tracks'),
                func.avg(DBTrackResult.sigma_gradient).label('avg_sigma'),
                func.min(DBTrackResult.sigma_gradient).label('min_sigma'),
                func.max(DBTrackResult.sigma_gradient).label('max_sigma'),
                func.sum(DBTrackResult.sigma_pass).label('sigma_passes'),
                func.sum(DBTrackResult.linearity_pass).label('linearity_passes'),
                func.avg(DBTrackResult.failure_probability).label('avg_failure_prob')
            ).join(
                DBAnalysisResult
            ).filter(
                DBAnalysisResult.model == model
            ).first()

            # Calculate pass rates
            total_tracks = track_stats.total_tracks or 0
            sigma_pass_rate = (track_stats.sigma_passes / total_tracks * 100) if total_tracks > 0 else 0
            linearity_pass_rate = (track_stats.linearity_passes / total_tracks * 100) if total_tracks > 0 else 0

            # Get risk distribution
            risk_dist = session.query(
                DBTrackResult.risk_category,
                func.count(DBTrackResult.id).label('count')
            ).join(
                DBAnalysisResult
            ).filter(
                DBAnalysisResult.model == model
            ).group_by(
                DBTrackResult.risk_category
            ).all()

            risk_distribution = {str(risk): count for risk, count in risk_dist if risk}

            # Get recent trend (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_trend = session.query(
                func.date(DBAnalysisResult.timestamp).label('date'),
                func.count(DBAnalysisResult.id).label('count'),
                func.avg(DBTrackResult.sigma_gradient).label('avg_sigma')
            ).join(
                DBTrackResult
            ).filter(
                and_(
                    DBAnalysisResult.model == model,
                    DBAnalysisResult.timestamp >= thirty_days_ago
                )
            ).group_by(
                func.date(DBAnalysisResult.timestamp)
            ).order_by(
                func.date(DBAnalysisResult.timestamp)
            ).all()

            return {
                "model": model,
                "total_files": total_files,
                "total_tracks": total_tracks,
                "statistics": {
                    "sigma_gradient": {
                        "average": float(track_stats.avg_sigma or 0),
                        "minimum": float(track_stats.min_sigma or 0),
                        "maximum": float(track_stats.max_sigma or 0)
                    },
                    "pass_rates": {
                        "sigma": sigma_pass_rate,
                        "linearity": linearity_pass_rate
                    },
                    "failure_probability": {
                        "average": float(track_stats.avg_failure_prob or 0)
                    },
                    "risk_distribution": risk_distribution
                },
                "recent_trend": [
                    {
                        "date": date.isoformat() if date and hasattr(date, 'isoformat') else str(date) if date else None,
                        "count": count,
                        "avg_sigma": float(avg_sigma or 0)
                    }
                    for date, count, avg_sigma in recent_trend
                ]
            }

    def get_risk_summary(self, days_back: Optional[int] = None) -> Dict[str, Any]:
        """
        Get summary of units by risk category.

        Args:
            days_back: Limit to records from last N days

        Returns:
            Dictionary with risk category counts and details
        """
        with self.get_session() as session:
            query = session.query(
                DBTrackResult.risk_category,
                func.count(DBTrackResult.id).label('count'),
                func.avg(DBTrackResult.failure_probability).label('avg_prob')
            )

            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.join(DBAnalysisResult).filter(
                    DBAnalysisResult.timestamp >= cutoff_date
                )

            results = query.group_by(DBTrackResult.risk_category).all()

            # Build summary
            summary = {
                "categories": {},
                "total": 0,
                "period_days": days_back
            }

            for risk_category, count, avg_prob in results:
                if risk_category:
                    category_name = risk_category.value
                    summary["categories"][category_name] = {
                        "count": count,
                        "percentage": 0,  # Will calculate after total
                        "avg_failure_probability": float(avg_prob or 0)
                    }
                    summary["total"] += count

            # Calculate percentages
            if summary["total"] > 0:
                for category in summary["categories"].values():
                    category["percentage"] = (category["count"] / summary["total"]) * 100

            # Get high-risk units details
            high_risk_units = session.query(
                DBAnalysisResult.filename,
                DBAnalysisResult.model,
                DBAnalysisResult.serial,
                DBTrackResult.track_id,
                DBTrackResult.failure_probability
            ).join(
                DBTrackResult
            ).filter(
                DBTrackResult.risk_category == DBRiskCategory.HIGH
            ).order_by(
                desc(DBTrackResult.failure_probability)
            ).limit(10).all()

            summary["high_risk_units"] = [
                {
                    "filename": filename,
                    "model": model,
                    "serial": serial,
                    "track_id": track_id,
                    "failure_probability": float(prob)
                }
                for filename, model, serial, track_id, prob in high_risk_units
            ]

            return summary

    def save_ml_prediction(self, prediction_data: Dict[str, Any]) -> int:
        """
        Save machine learning prediction results.

        Args:
            prediction_data: Dictionary containing ML prediction data

        Returns:
            ID of the saved prediction record
        """
        with self.get_session() as session:
            # Convert risk category if present
            risk_category = None
            if prediction_data.get('predicted_risk_category'):
                risk_category = DBRiskCategory(prediction_data['predicted_risk_category'])

            prediction = DBMLPrediction(
                analysis_id=prediction_data['analysis_id'],
                model_version=prediction_data.get('model_version'),
                prediction_type=prediction_data.get('prediction_type'),
                current_threshold=prediction_data.get('current_threshold'),
                recommended_threshold=prediction_data.get('recommended_threshold'),
                threshold_change_percent=prediction_data.get('threshold_change_percent'),
                false_positives=prediction_data.get('false_positives'),
                false_negatives=prediction_data.get('false_negatives'),
                predicted_failure_probability=prediction_data.get('predicted_failure_probability'),
                predicted_risk_category=risk_category,
                confidence_score=prediction_data.get('confidence_score'),
                feature_importance=prediction_data.get('feature_importance'),
                drift_detected=prediction_data.get('drift_detected', False),
                drift_percentage=prediction_data.get('drift_percentage'),
                drift_direction=prediction_data.get('drift_direction'),
                recommendations=prediction_data.get('recommendations')
            )

            session.add(prediction)
            session.commit()

            self.logger.info(f"Saved ML prediction: {prediction_data.get('prediction_type')}")
            return prediction.id

    def get_batch_statistics(self, batch_id: int) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific batch.

        Args:
            batch_id: ID of the batch

        Returns:
            Dictionary with batch statistics or None if not found
        """
        with self.get_session() as session:
            batch = session.query(DBBatchInfo).filter(DBBatchInfo.id == batch_id).first()

            if not batch:
                return None

            # Get linked analyses
            analyses = session.query(DBAnalysisResult).join(
                DBAnalysisBatch
            ).filter(
                DBAnalysisBatch.batch_id == batch_id
            ).all()

            # Calculate statistics from linked analyses
            total_tracks = 0
            passed_tracks = 0
            high_risk_tracks = 0
            sigma_values = []

            for analysis in analyses:
                for track in analysis.tracks:
                    total_tracks += 1
                    if track.sigma_pass and track.linearity_pass:
                        passed_tracks += 1
                    if track.risk_category == DBRiskCategory.HIGH:
                        high_risk_tracks += 1
                    if track.sigma_gradient:
                        sigma_values.append(track.sigma_gradient)

            # Update batch statistics
            batch.total_units = len(analyses)
            batch.passed_units = sum(1 for a in analyses if a.overall_status == DBStatusType.PASS)
            batch.failed_units = sum(1 for a in analyses if a.overall_status == DBStatusType.FAIL)

            if sigma_values:
                batch.average_sigma_gradient = sum(sigma_values) / len(sigma_values)

            if total_tracks > 0:
                batch.sigma_pass_rate = (passed_tracks / total_tracks) * 100

            batch.high_risk_count = high_risk_tracks

            session.commit()

            return {
                "batch_id": batch.id,
                "batch_name": batch.batch_name,
                "batch_type": batch.batch_type,
                "created_date": batch.created_date.isoformat() if batch.created_date and hasattr(batch.created_date, 'isoformat') else str(batch.created_date) if batch.created_date else None,
                "model": batch.model,
                "statistics": {
                    "total_units": batch.total_units,
                    "passed_units": batch.passed_units,
                    "failed_units": batch.failed_units,
                    "pass_rate": (batch.passed_units / batch.total_units * 100) if batch.total_units > 0 else 0,
                    "average_sigma_gradient": batch.average_sigma_gradient,
                    "sigma_pass_rate": batch.sigma_pass_rate,
                    "high_risk_count": batch.high_risk_count
                },
                "analyses": [
                    {
                        "filename": a.filename,
                        "model": a.model,
                        "serial": a.serial,
                        "status": a.overall_status.value
                    }
                    for a in analyses[:10]  # First 10 for preview
                ]
            }

    def _generate_alerts(self, analysis: DBAnalysisResult, session: Session) -> None:
        """
        Generate QA alerts based on analysis results.

        Args:
            analysis: Analysis result to check
            session: Database session
        """
        alerts = []

        for track in analysis.tracks:
            # Carbon screen check for 8340 models
            if "8340" in analysis.model and track.sigma_gradient > track.sigma_threshold:
                alerts.append(DBQAAlert(
                    alert_type=DBAlertType.CARBON_SCREEN,
                    severity="High",
                    track_id=track.track_id,
                    metric_name="sigma_gradient",
                    metric_value=track.sigma_gradient,
                    threshold_value=track.sigma_threshold,
                    message=f"Carbon screen check required for {analysis.model}-{analysis.serial} ({track.track_id})"
                ))

            # High risk alert
            if track.risk_category == DBRiskCategory.HIGH:
                alerts.append(DBQAAlert(
                    alert_type=DBAlertType.HIGH_RISK,
                    severity="Critical",
                    track_id=track.track_id,
                    metric_name="failure_probability",
                    metric_value=track.failure_probability,
                    message=f"High risk unit detected: {analysis.model}-{analysis.serial} ({track.track_id})"
                ))

            # Threshold exceeded alert
            if not track.sigma_pass:
                alerts.append(DBQAAlert(
                    alert_type=DBAlertType.THRESHOLD_EXCEEDED,
                    severity="High",
                    track_id=track.track_id,
                    metric_name="sigma_gradient",
                    metric_value=track.sigma_gradient,
                    threshold_value=track.sigma_threshold,
                    message=f"Sigma threshold exceeded for {analysis.model}-{analysis.serial} ({track.track_id})"
                ))

        # Add alerts to analysis
        for alert in alerts:
            analysis.qa_alerts.append(alert)
            self.logger.info(f"Generated alert: {alert.alert_type.value} for {analysis.filename}")

    def get_unresolved_alerts(
            self,
            severity: Optional[str] = None,
            alert_type: Optional[str] = None,
            days_back: Optional[int] = None
    ) -> List[DBQAAlert]:
        """
        Get unresolved QA alerts.

        Args:
            severity: Filter by severity level
            alert_type: Filter by alert type
            days_back: Limit to alerts from last N days

        Returns:
            List of unresolved QAAlert objects
        """
        with self.get_session() as session:
            query = session.query(DBQAAlert).filter(DBQAAlert.resolved == False)

            if severity:
                query = query.filter(DBQAAlert.severity == severity)

            if alert_type:
                query = query.filter(DBQAAlert.alert_type == DBAlertType(alert_type))

            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(DBQAAlert.created_date >= cutoff_date)

            # Order by severity and date
            query = query.order_by(
                desc(DBQAAlert.severity),
                desc(DBQAAlert.created_date)
            )

            return query.all()

    def create_batch(self, batch_name: str, model: str, batch_type: str = "production") -> int:
        """
        Create a new batch for grouping analyses.

        Args:
            batch_name: Name of the batch
            model: Model number for the batch
            batch_type: Type of batch (production, rework, test)

        Returns:
            ID of the created batch
        """
        with self.get_session() as session:
            batch = DBBatchInfo(
                batch_name=batch_name,
                batch_type=batch_type,
                model=model
            )
            session.add(batch)
            session.commit()

            self.logger.info(f"Created batch: {batch_name}")
            return batch.id

    def add_analysis_to_batch(self, analysis_id: int, batch_id: int) -> None:
        """
        Add an analysis to a batch.

        Args:
            analysis_id: ID of the analysis
            batch_id: ID of the batch
        """
        with self.get_session() as session:
            # Check if association already exists
            existing = session.query(DBAnalysisBatch).filter_by(
                analysis_id=analysis_id,
                batch_id=batch_id
            ).first()

            if not existing:
                association = DBAnalysisBatch(
                    analysis_id=analysis_id,
                    batch_id=batch_id
                )
                session.add(association)
                session.commit()
                self.logger.info(f"Added analysis {analysis_id} to batch {batch_id}")

    def close(self) -> None:
        """Close database connections and clean up resources."""
        try:
            self.Session.remove()
            self.engine.dispose()
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database: {str(e)}")

    def save_analysis_batch(self, analysis_list: List[PydanticAnalysisResult]) -> List[int]:
        """
        Save multiple analysis results in a batch operation for better performance.
        
        Args:
            analysis_list: List of analysis results to save
            
        Returns:
            List of analysis IDs
        """
        if not analysis_list:
            return []
            
        analysis_ids = []
        
        with self.get_session() as session:
            try:
                self.logger.info(f"Saving batch of {len(analysis_list)} analysis results")
                
                # Save each analysis individually for reliability
                for analysis_data in analysis_list:
                    try:
                        # Convert Pydantic SystemType to DB SystemType
                        system_type = DBSystemType.A if analysis_data.metadata.system == SystemType.SYSTEM_A else DBSystemType.B

                        # Convert Pydantic AnalysisStatus to DB StatusType
                        status_map = {
                            AnalysisStatus.PASS: DBStatusType.PASS,
                            AnalysisStatus.FAIL: DBStatusType.FAIL,
                            AnalysisStatus.WARNING: DBStatusType.WARNING,
                            AnalysisStatus.ERROR: DBStatusType.ERROR,
                            AnalysisStatus.PENDING: DBStatusType.PROCESSING_FAILED
                        }
                        overall_status = status_map.get(analysis_data.overall_status, DBStatusType.ERROR)

                        # Create main analysis record
                        analysis = DBAnalysisResult(
                            filename=analysis_data.metadata.filename,
                            file_path=str(analysis_data.metadata.file_path),
                            file_date=analysis_data.metadata.file_date,
                            file_hash=None,  # Calculate if needed
                            model=analysis_data.metadata.model,
                            serial=analysis_data.metadata.serial,
                            system=system_type,
                            has_multi_tracks=analysis_data.metadata.has_multi_tracks,
                            overall_status=overall_status,
                            processing_time=analysis_data.processing_time,
                            output_dir=None,  # Set if available
                            software_version="2.0.0",  # Set from config
                            operator=None,  # Set if available
                            sigma_scaling_factor=None,  # Set from first track if available
                            filter_cutoff_frequency=None  # Set from config if available
                        )

                        # Add track results
                        for track_id, track_data in analysis_data.tracks.items():
                            # Convert track status
                            track_status = status_map.get(track_data.status, DBStatusType.ERROR)

                            # Convert risk category if available
                            risk_category = None
                            if track_data.failure_prediction:
                                risk_map = {
                                    RiskCategory.HIGH: DBRiskCategory.HIGH,
                                    RiskCategory.MEDIUM: DBRiskCategory.MEDIUM,
                                    RiskCategory.LOW: DBRiskCategory.LOW,
                                    RiskCategory.UNKNOWN: DBRiskCategory.UNKNOWN
                                }
                                risk_category = risk_map.get(track_data.failure_prediction.risk_category)

                            track = DBTrackResult(
                                track_id=track_id,
                                status=track_status,
                                travel_length=track_data.travel_length,
                                linearity_spec=track_data.linearity_analysis.linearity_spec,
                                sigma_gradient=track_data.sigma_analysis.sigma_gradient,
                                sigma_threshold=track_data.sigma_analysis.sigma_threshold,
                                sigma_pass=track_data.sigma_analysis.sigma_pass,
                                unit_length=track_data.unit_properties.unit_length,
                                untrimmed_resistance=track_data.unit_properties.untrimmed_resistance,
                                trimmed_resistance=track_data.unit_properties.trimmed_resistance,
                                resistance_change=track_data.unit_properties.resistance_change,
                                resistance_change_percent=track_data.unit_properties.resistance_change_percent,
                                optimal_offset=track_data.linearity_analysis.optimal_offset,
                                final_linearity_error_raw=track_data.linearity_analysis.final_linearity_error_raw,
                                final_linearity_error_shifted=track_data.linearity_analysis.final_linearity_error_shifted,
                                linearity_pass=track_data.linearity_analysis.linearity_pass,
                                linearity_fail_points=track_data.linearity_analysis.linearity_fail_points,
                                max_deviation=track_data.linearity_analysis.max_deviation,
                                max_deviation_position=track_data.linearity_analysis.max_deviation_position,
                                deviation_uniformity=None,  # Calculate if needed
                                trim_improvement_percent=track_data.trim_effectiveness.improvement_percent if track_data.trim_effectiveness else None,
                                untrimmed_rms_error=track_data.trim_effectiveness.untrimmed_rms_error if track_data.trim_effectiveness else None,
                                trimmed_rms_error=track_data.trim_effectiveness.trimmed_rms_error if track_data.trim_effectiveness else None,
                                max_error_reduction_percent=track_data.trim_effectiveness.max_error_reduction_percent if track_data.trim_effectiveness else None,
                                worst_zone=track_data.zone_analysis.worst_zone if track_data.zone_analysis else None,
                                worst_zone_position=track_data.zone_analysis.worst_zone_position[
                                    0] if track_data.zone_analysis and track_data.zone_analysis.worst_zone_position else None,
                                zone_details=track_data.zone_analysis.zone_results if track_data.zone_analysis else None,
                                failure_probability=track_data.failure_prediction.failure_probability if track_data.failure_prediction else None,
                                risk_category=risk_category,
                                gradient_margin=track_data.sigma_analysis.gradient_margin,
                                range_utilization_percent=track_data.dynamic_range.range_utilization_percent if track_data.dynamic_range else None,
                                minimum_margin=track_data.dynamic_range.minimum_margin if track_data.dynamic_range else None,
                                minimum_margin_position=track_data.dynamic_range.minimum_margin_position if track_data.dynamic_range else None,
                                margin_bias=track_data.dynamic_range.margin_bias if track_data.dynamic_range else None,
                                plot_path=str(track_data.plot_path) if track_data.plot_path else None
                            )
                            analysis.tracks.append(track)

                        # Generate alerts for this analysis
                        self._generate_alerts(analysis, session)

                        # Add to session
                        session.add(analysis)
                        session.flush()  # Get the ID without committing
                        
                        analysis_ids.append(analysis.id)
                        
                        self.logger.debug(f"Prepared analysis for {analysis_data.metadata.filename}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to prepare analysis for {analysis_data.metadata.filename}: {e}")
                        # Continue with other analyses
                        continue
                
                # Single commit for the entire batch
                session.commit()
                
                self.logger.info(f"Successfully saved batch: {len(analysis_ids)} analyses with tracks")
                
                return analysis_ids
                
            except Exception as e:
                session.rollback()
                self.logger.error(f"Failed to save analysis batch: {str(e)}")
                raise

    def validate_saved_analysis(self, analysis_id: int) -> bool:
        """
        Validate that an analysis was properly saved to the database.
        
        Args:
            analysis_id: ID of the analysis to validate
            
        Returns:
            True if analysis is properly saved with all tracks
        """
        try:
            with self.get_session() as session:
                # Check analysis exists
                analysis = session.query(DBAnalysisResult).filter(
                    DBAnalysisResult.id == analysis_id
                ).first()
                
                if not analysis:
                    self.logger.error(f"Analysis ID {analysis_id} not found in database")
                    return False
                
                # Check tracks exist
                track_count = session.query(DBTrackResult).filter(
                    DBTrackResult.analysis_id == analysis_id
                ).count()
                
                if track_count == 0:
                    self.logger.error(f"No tracks found for analysis ID {analysis_id}")
                    return False
                
                self.logger.info(f"Analysis ID {analysis_id} validated: {track_count} tracks")
                return True
                
        except Exception as e:
            self.logger.error(f"Validation failed for analysis ID {analysis_id}: {e}")
            return False

    def force_save_analysis(self, analysis_data: PydanticAnalysisResult) -> int:
        """
        Force save analysis without duplicate checking - for critical saves.
        
        Args:
            analysis_data: Pydantic model containing analysis results
            
        Returns:
            ID of the saved analysis record
        """
        with self.get_session() as session:
            try:
                self.logger.info(f"Force saving analysis for {analysis_data.metadata.filename}")
                
                # Convert Pydantic SystemType to DB SystemType
                system_type = DBSystemType.A if analysis_data.metadata.system == SystemType.SYSTEM_A else DBSystemType.B

                # Convert Pydantic AnalysisStatus to DB StatusType
                status_map = {
                    AnalysisStatus.PASS: DBStatusType.PASS,
                    AnalysisStatus.FAIL: DBStatusType.FAIL,
                    AnalysisStatus.WARNING: DBStatusType.WARNING,
                    AnalysisStatus.ERROR: DBStatusType.ERROR,
                    AnalysisStatus.PENDING: DBStatusType.PROCESSING_FAILED
                }
                overall_status = status_map.get(analysis_data.overall_status, DBStatusType.ERROR)

                # Create main analysis record
                analysis = DBAnalysisResult(
                    filename=analysis_data.metadata.filename,
                    file_path=str(analysis_data.metadata.file_path),
                    file_date=analysis_data.metadata.file_date,
                    file_hash=None,  # Calculate if needed
                    model=analysis_data.metadata.model,
                    serial=analysis_data.metadata.serial,
                    system=system_type,
                    has_multi_tracks=analysis_data.metadata.has_multi_tracks,
                    overall_status=overall_status,
                    processing_time=analysis_data.processing_time,
                    output_dir=None,  # Set if available
                    software_version="2.0.0",  # Set from config
                    operator=None,  # Set if available
                    sigma_scaling_factor=None,  # Set from first track if available
                    filter_cutoff_frequency=None  # Set from config if available
                )

                # Add track results
                for track_id, track_data in analysis_data.tracks.items():
                    # Convert track status
                    track_status = status_map.get(track_data.status, DBStatusType.ERROR)

                    # Convert risk category if available
                    risk_category = None
                    if track_data.failure_prediction:
                        risk_map = {
                            RiskCategory.HIGH: DBRiskCategory.HIGH,
                            RiskCategory.MEDIUM: DBRiskCategory.MEDIUM,
                            RiskCategory.LOW: DBRiskCategory.LOW,
                            RiskCategory.UNKNOWN: DBRiskCategory.UNKNOWN
                        }
                        risk_category = risk_map.get(track_data.failure_prediction.risk_category)

                    track = DBTrackResult(
                        track_id=track_id,
                        status=track_status,
                        travel_length=track_data.travel_length,
                        linearity_spec=track_data.linearity_analysis.linearity_spec,
                        sigma_gradient=track_data.sigma_analysis.sigma_gradient,
                        sigma_threshold=track_data.sigma_analysis.sigma_threshold,
                        sigma_pass=track_data.sigma_analysis.sigma_pass,
                        unit_length=track_data.unit_properties.unit_length,
                        untrimmed_resistance=track_data.unit_properties.untrimmed_resistance,
                        trimmed_resistance=track_data.unit_properties.trimmed_resistance,
                        resistance_change=track_data.unit_properties.resistance_change,
                        resistance_change_percent=track_data.unit_properties.resistance_change_percent,
                        optimal_offset=track_data.linearity_analysis.optimal_offset,
                        final_linearity_error_raw=track_data.linearity_analysis.final_linearity_error_raw,
                        final_linearity_error_shifted=track_data.linearity_analysis.final_linearity_error_shifted,
                        linearity_pass=track_data.linearity_analysis.linearity_pass,
                        linearity_fail_points=track_data.linearity_analysis.linearity_fail_points,
                        max_deviation=track_data.linearity_analysis.max_deviation,
                        max_deviation_position=track_data.linearity_analysis.max_deviation_position,
                        deviation_uniformity=None,  # Calculate if needed
                        trim_improvement_percent=track_data.trim_effectiveness.improvement_percent if track_data.trim_effectiveness else None,
                        untrimmed_rms_error=track_data.trim_effectiveness.untrimmed_rms_error if track_data.trim_effectiveness else None,
                        trimmed_rms_error=track_data.trim_effectiveness.trimmed_rms_error if track_data.trim_effectiveness else None,
                        max_error_reduction_percent=track_data.trim_effectiveness.max_error_reduction_percent if track_data.trim_effectiveness else None,
                        worst_zone=track_data.zone_analysis.worst_zone if track_data.zone_analysis else None,
                        worst_zone_position=track_data.zone_analysis.worst_zone_position[
                            0] if track_data.zone_analysis and track_data.zone_analysis.worst_zone_position else None,
                        zone_details=track_data.zone_analysis.zone_results if track_data.zone_analysis else None,
                        failure_probability=track_data.failure_prediction.failure_probability if track_data.failure_prediction else None,
                        risk_category=risk_category,
                        gradient_margin=track_data.sigma_analysis.gradient_margin,
                        range_utilization_percent=track_data.dynamic_range.range_utilization_percent if track_data.dynamic_range else None,
                        minimum_margin=track_data.dynamic_range.minimum_margin if track_data.dynamic_range else None,
                        minimum_margin_position=track_data.dynamic_range.minimum_margin_position if track_data.dynamic_range else None,
                        margin_bias=track_data.dynamic_range.margin_bias if track_data.dynamic_range else None,
                        plot_path=str(track_data.plot_path) if track_data.plot_path else None
                    )
                    analysis.tracks.append(track)

                # Check for alerts
                self._generate_alerts(analysis, session)

                session.add(analysis)
                session.commit()

                # Validate the save was successful
                if not self.validate_saved_analysis(analysis.id):
                    raise RuntimeError(f"Analysis save validation failed for {analysis_data.metadata.filename}")

                self.logger.info(
                    f"Force saved analysis for {analysis_data.metadata.filename} with {len(analysis.tracks)} tracks (ID: {analysis.id})")
                return analysis.id

            except Exception as e:
                session.rollback()
                self.logger.error(f"Failed to force save analysis: {str(e)}")
                raise