"""
Database Manager for Laser Trim Analyzer v2.

Handles all database operations with connection pooling, error handling,
and migration support. Designed for QA specialists to easily store and
retrieve potentiometer test results.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, func, and_, or_, desc
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine

# Import our models
from .models import (
    Base, AnalysisResult, TrackResult, MLPrediction,
    QAAlert, BatchInfo, AnalysisBatch,
    SystemType, StatusType, RiskCategory, AlertType
)

# Import Pydantic models for type hints
# Assuming these exist in core/models.py
"@

# Save the file
Set-Content "src\laser_trim_analyzer\database\manager.py" -Value """
Database Manager for Laser Trim Analyzer v2.

Handles all database operations with connection pooling, error handling,
and migration support. Designed for QA specialists to easily store and
retrieve potentiometer test results.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, func, and_, or_, desc
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine

# Import our models
from .models import (
    Base, AnalysisResult, TrackResult, MLPrediction,
    QAAlert, BatchInfo, AnalysisBatch,
    SystemType, StatusType, RiskCategory, AlertType
)

# Import Pydantic models for type hints
# Assuming these exist in core/models.py



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
            database_url: Optional[str] = None,
            echo: bool = False,
            pool_size: int = 5,
            max_overflow: int = 10,
            pool_timeout: int = 30,
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the database manager.

        Args:
            database_url: SQLAlchemy database URL. If None, uses SQLite default.
            echo: If True, log all SQL statements (useful for debugging)
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum overflow connections allowed
            pool_timeout: Timeout in seconds for getting connection from pool
            logger: Logger instance for database operations
        """
        self.logger = logger or logging.getLogger(__name__)

        # Set up database URL
        if database_url is None:
            # Default to SQLite in user's home directory
            db_dir = Path.home() / ".laser_trim_analyzer"
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / "analyzer_v2.db"
            database_url = f"sqlite:///{db_path}"
            self.logger.info(f"Using SQLite database at: {db_path}")

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
    def get_session(self) -> Session:
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
            inspector = self.engine.inspect(self.engine)
            tables = inspector.get_table_names()
            self.logger.info(f"Created {len(tables)} tables: {', '.join(tables)}")

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise

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
                # Create main analysis record
                analysis = AnalysisResult(
                    filename=analysis_data.filename,
                    file_path=analysis_data.file_path,
                    file_date=analysis_data.file_date,
                    file_hash=analysis_data.file_hash,
                    model=analysis_data.model,
                    serial=analysis_data.serial,
                    system=SystemType(analysis_data.system),
                    has_multi_tracks=analysis_data.has_multi_tracks,
                    overall_status=StatusType(analysis_data.overall_status),
                    processing_time=analysis_data.processing_time,
                    output_dir=analysis_data.output_dir,
                    software_version=analysis_data.software_version,
                    operator=analysis_data.operator,
                    sigma_scaling_factor=analysis_data.sigma_scaling_factor,
                    filter_cutoff_frequency=analysis_data.filter_cutoff_frequency
                )

                # Add track results
                for track_data in analysis_data.tracks:
                    track = TrackResult(
                        track_id=track_data.track_id,
                        status=StatusType(track_data.status),
                        travel_length=track_data.travel_length,
                        linearity_spec=track_data.linearity_spec,
                        sigma_gradient=track_data.sigma_gradient,
                        sigma_threshold=track_data.sigma_threshold,
                        sigma_pass=track_data.sigma_pass,
                        unit_length=track_data.unit_length,
                        untrimmed_resistance=track_data.untrimmed_resistance,
                        trimmed_resistance=track_data.trimmed_resistance,
                        resistance_change=track_data.resistance_change,
                        resistance_change_percent=track_data.resistance_change_percent,
                        optimal_offset=track_data.optimal_offset,
                        final_linearity_error_raw=track_data.final_linearity_error_raw,
                        final_linearity_error_shifted=track_data.final_linearity_error_shifted,
                        linearity_pass=track_data.linearity_pass,
                        linearity_fail_points=track_data.linearity_fail_points,
                        max_deviation=track_data.max_deviation,
                        max_deviation_position=track_data.max_deviation_position,
                        deviation_uniformity=track_data.deviation_uniformity,
                        trim_improvement_percent=track_data.trim_improvement_percent,
                        untrimmed_rms_error=track_data.untrimmed_rms_error,
                        trimmed_rms_error=track_data.trimmed_rms_error,
                        max_error_reduction_percent=track_data.max_error_reduction_percent,
                        worst_zone=track_data.worst_zone,
                        worst_zone_position=track_data.worst_zone_position,
                        zone_details=track_data.zone_details,
                        failure_probability=track_data.failure_probability,
                        risk_category=RiskCategory(track_data.risk_category) if track_data.risk_category else None,
                        gradient_margin=track_data.gradient_margin,
                        range_utilization_percent=track_data.range_utilization_percent,
                        minimum_margin=track_data.minimum_margin,
                        minimum_margin_position=track_data.minimum_margin_position,
                        margin_bias=track_data.margin_bias,
                        plot_path=track_data.plot_path
                    )
                    analysis.tracks.append(track)

                # Check for alerts
                self._generate_alerts(analysis, session)

                session.add(analysis)
                session.commit()

                self.logger.info(f"Saved analysis for {analysis_data.filename} with {len(analysis.tracks)} tracks")
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
    ) -> List[AnalysisResult]:
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
            query = session.query(AnalysisResult)

            # Apply filters
            if model:
                query = query.filter(AnalysisResult.model.like(model))

            if serial:
                query = query.filter(AnalysisResult.serial.like(serial))

            # Date filtering
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(AnalysisResult.timestamp >= cutoff_date)
            elif start_date:
                query = query.filter(AnalysisResult.timestamp >= start_date)
                if end_date:
                    query = query.filter(AnalysisResult.timestamp <= end_date)

            if status:
                query = query.filter(AnalysisResult.overall_status == StatusType(status))

            if risk_category and include_tracks:
                # Join with tracks to filter by risk category
                query = query.join(TrackResult).filter(
                    TrackResult.risk_category == RiskCategory(risk_category)
                ).distinct()

            # Order by most recent first
            query = query.order_by(desc(AnalysisResult.timestamp))

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
            base_query = session.query(AnalysisResult).filter(
                AnalysisResult.model == model
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
                func.count(TrackResult.id).label('total_tracks'),
                func.avg(TrackResult.sigma_gradient).label('avg_sigma'),
                func.min(TrackResult.sigma_gradient).label('min_sigma'),
                func.max(TrackResult.sigma_gradient).label('max_sigma'),
                func.sum(TrackResult.sigma_pass).label('sigma_passes'),
                func.sum(TrackResult.linearity_pass).label('linearity_passes'),
                func.avg(TrackResult.failure_probability).label('avg_failure_prob')
            ).join(
                AnalysisResult
            ).filter(
                AnalysisResult.model == model
            ).first()

            # Calculate pass rates
            total_tracks = track_stats.total_tracks or 0
            sigma_pass_rate = (track_stats.sigma_passes / total_tracks * 100) if total_tracks > 0 else 0
            linearity_pass_rate = (track_stats.linearity_passes / total_tracks * 100) if total_tracks > 0 else 0

            # Get risk distribution
            risk_dist = session.query(
                TrackResult.risk_category,
                func.count(TrackResult.id).label('count')
            ).join(
                AnalysisResult
            ).filter(
                AnalysisResult.model == model
            ).group_by(
                TrackResult.risk_category
            ).all()

            risk_distribution = {str(risk): count for risk, count in risk_dist if risk}

            # Get recent trend (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_trend = session.query(
                func.date(AnalysisResult.timestamp).label('date'),
                func.count(AnalysisResult.id).label('count'),
                func.avg(TrackResult.sigma_gradient).label('avg_sigma')
            ).join(
                TrackResult
            ).filter(
                and_(
                    AnalysisResult.model == model,
                    AnalysisResult.timestamp >= thirty_days_ago
                )
            ).group_by(
                func.date(AnalysisResult.timestamp)
            ).order_by(
                func.date(AnalysisResult.timestamp)
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
                        "date": date.isoformat() if date else None,
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
                TrackResult.risk_category,
                func.count(TrackResult.id).label('count'),
                func.avg(TrackResult.failure_probability).label('avg_prob')
            )

            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.join(AnalysisResult).filter(
                    AnalysisResult.timestamp >= cutoff_date
                )

            results = query.group_by(TrackResult.risk_category).all()

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
                AnalysisResult.filename,
                AnalysisResult.model,
                AnalysisResult.serial,
                TrackResult.track_id,
                TrackResult.failure_probability
            ).join(
                TrackResult
            ).filter(
                TrackResult.risk_category == RiskCategory.HIGH
            ).order_by(
                desc(TrackResult.failure_probability)
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

    def save_ml_prediction(self, prediction_data: MLPrediction) -> int:
        """
        Save machine learning prediction results.

        Args:
            prediction_data: Pydantic model containing ML prediction

        Returns:
            ID of the saved prediction record
        """
        with self.get_session() as session:
            prediction = MLPrediction(
                analysis_id=prediction_data.analysis_id,
                model_version=prediction_data.model_version,
                prediction_type=prediction_data.prediction_type,
                current_threshold=prediction_data.current_threshold,
                recommended_threshold=prediction_data.recommended_threshold,
                threshold_change_percent=prediction_data.threshold_change_percent,
                false_positives=prediction_data.false_positives,
                false_negatives=prediction_data.false_negatives,
                predicted_failure_probability=prediction_data.predicted_failure_probability,
                predicted_risk_category=RiskCategory(
                    prediction_data.predicted_risk_category) if prediction_data.predicted_risk_category else None,
                confidence_score=prediction_data.confidence_score,
                feature_importance=prediction_data.feature_importance,
                drift_detected=prediction_data.drift_detected,
                drift_percentage=prediction_data.drift_percentage,
                drift_direction=prediction_data.drift_direction,
                recommendations=prediction_data.recommendations
            )

            session.add(prediction)
            session.commit()

            self.logger.info(f"Saved ML prediction: {prediction_data.prediction_type}")
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
            batch = session.query(BatchInfo).filter(BatchInfo.id == batch_id).first()

            if not batch:
                return None

            # Get linked analyses
            analyses = session.query(AnalysisResult).join(
                AnalysisBatch
            ).filter(
                AnalysisBatch.batch_id == batch_id
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
                    if track.risk_category == RiskCategory.HIGH:
                        high_risk_tracks += 1
                    if track.sigma_gradient:
                        sigma_values.append(track.sigma_gradient)

            # Update batch statistics
            batch.total_units = len(analyses)
            batch.passed_units = sum(1 for a in analyses if a.overall_status == StatusType.PASS)
            batch.failed_units = sum(1 for a in analyses if a.overall_status == StatusType.FAIL)

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
                "created_date": batch.created_date.isoformat() if batch.created_date else None,
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

    def _generate_alerts(self, analysis: AnalysisResult, session: Session) -> None:
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
                alerts.append(QAAlert(
                    alert_type=AlertType.CARBON_SCREEN,
                    severity="High",
                    track_id=track.track_id,
                    metric_name="sigma_gradient",
                    metric_value=track.sigma_gradient,
                    threshold_value=track.sigma_threshold,
                    message=f"Carbon screen check required for {analysis.model}-{analysis.serial} ({track.track_id})"
                ))

            # High risk alert
            if track.risk_category == RiskCategory.HIGH:
                alerts.append(QAAlert(
                    alert_type=AlertType.HIGH_RISK,
                    severity="Critical",
                    track_id=track.track_id,
                    metric_name="failure_probability",
                    metric_value=track.failure_probability,
                    message=f"High risk unit detected: {analysis.model}-{analysis.serial} ({track.track_id})"
                ))

            # Threshold exceeded alert
            if not track.sigma_pass:
                alerts.append(QAAlert(
                    alert_type=AlertType.THRESHOLD_EXCEEDED,
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
    ) -> List[QAAlert]:
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
            query = session.query(QAAlert).filter(QAAlert.resolved == False)

            if severity:
                query = query.filter(QAAlert.severity == severity)

            if alert_type:
                query = query.filter(QAAlert.alert_type == AlertType(alert_type))

            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(QAAlert.created_date >= cutoff_date)

            # Order by severity and date
            query = query.order_by(
                desc(QAAlert.severity),
                desc(QAAlert.created_date)
            )

            return query.all()

    def close(self) -> None:
        """Close database connections and clean up resources."""
        try:
            self.Session.remove()
            self.engine.dispose()
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database: {str(e)}")



# Now let's check if there are any references to the missing Pydantic models
Select-String -Path "src\laser_trim_analyzer\database\manager.py" -Pattern "MLPrediction|QAAlert|BatchInfo" | Select-Object Line, LineNumber
# Check for save_ml_prediction method
Select-String -Path "src\laser_trim_analyzer\database\manager.py" -Pattern "def save_ml_prediction" -Context 0,10

# Check for methods that might use QAAlert
Select-String -Path "src\laser_trim_analyzer\database\manager.py" -Pattern "QAAlert|alert" -Context 2,2 | Select-Object -First 5
# Create a backup first
Copy-Item "src\laser_trim_analyzer\database\manager.py" "src\laser_trim_analyzer\database\manager.py.bak"

# Read the content
"""
Database Manager for Laser Trim Analyzer v2.

Handles all database operations with connection pooling, error handling,
and migration support. Designed for QA specialists to easily store and
retrieve potentiometer test results.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, func, and_, or_, desc
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine

# Import our models
from .models import (
    Base, AnalysisResult, TrackResult, MLPrediction,
    QAAlert, BatchInfo, AnalysisBatch,
    SystemType, StatusType, RiskCategory, AlertType
)

# Import Pydantic models for type hints
# Assuming these exist in core/models.py



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
            database_url: Optional[str] = None,
            echo: bool = False,
            pool_size: int = 5,
            max_overflow: int = 10,
            pool_timeout: int = 30,
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the database manager.

        Args:
            database_url: SQLAlchemy database URL. If None, uses SQLite default.
            echo: If True, log all SQL statements (useful for debugging)
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum overflow connections allowed
            pool_timeout: Timeout in seconds for getting connection from pool
            logger: Logger instance for database operations
        """
        self.logger = logger or logging.getLogger(__name__)

        # Set up database URL
        if database_url is None:
            # Default to SQLite in user's home directory
            db_dir = Path.home() / ".laser_trim_analyzer"
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / "analyzer_v2.db"
            database_url = f"sqlite:///{db_path}"
            self.logger.info(f"Using SQLite database at: {db_path}")

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
    def get_session(self) -> Session:
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
            inspector = self.engine.inspect(self.engine)
            tables = inspector.get_table_names()
            self.logger.info(f"Created {len(tables)} tables: {', '.join(tables)}")

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise

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
                # Create main analysis record
                analysis = AnalysisResult(
                    filename=analysis_data.filename,
                    file_path=analysis_data.file_path,
                    file_date=analysis_data.file_date,
                    file_hash=analysis_data.file_hash,
                    model=analysis_data.model,
                    serial=analysis_data.serial,
                    system=SystemType(analysis_data.system),
                    has_multi_tracks=analysis_data.has_multi_tracks,
                    overall_status=StatusType(analysis_data.overall_status),
                    processing_time=analysis_data.processing_time,
                    output_dir=analysis_data.output_dir,
                    software_version=analysis_data.software_version,
                    operator=analysis_data.operator,
                    sigma_scaling_factor=analysis_data.sigma_scaling_factor,
                    filter_cutoff_frequency=analysis_data.filter_cutoff_frequency
                )

                # Add track results
                for track_data in analysis_data.tracks:
                    track = TrackResult(
                        track_id=track_data.track_id,
                        status=StatusType(track_data.status),
                        travel_length=track_data.travel_length,
                        linearity_spec=track_data.linearity_spec,
                        sigma_gradient=track_data.sigma_gradient,
                        sigma_threshold=track_data.sigma_threshold,
                        sigma_pass=track_data.sigma_pass,
                        unit_length=track_data.unit_length,
                        untrimmed_resistance=track_data.untrimmed_resistance,
                        trimmed_resistance=track_data.trimmed_resistance,
                        resistance_change=track_data.resistance_change,
                        resistance_change_percent=track_data.resistance_change_percent,
                        optimal_offset=track_data.optimal_offset,
                        final_linearity_error_raw=track_data.final_linearity_error_raw,
                        final_linearity_error_shifted=track_data.final_linearity_error_shifted,
                        linearity_pass=track_data.linearity_pass,
                        linearity_fail_points=track_data.linearity_fail_points,
                        max_deviation=track_data.max_deviation,
                        max_deviation_position=track_data.max_deviation_position,
                        deviation_uniformity=track_data.deviation_uniformity,
                        trim_improvement_percent=track_data.trim_improvement_percent,
                        untrimmed_rms_error=track_data.untrimmed_rms_error,
                        trimmed_rms_error=track_data.trimmed_rms_error,
                        max_error_reduction_percent=track_data.max_error_reduction_percent,
                        worst_zone=track_data.worst_zone,
                        worst_zone_position=track_data.worst_zone_position,
                        zone_details=track_data.zone_details,
                        failure_probability=track_data.failure_probability,
                        risk_category=RiskCategory(track_data.risk_category) if track_data.risk_category else None,
                        gradient_margin=track_data.gradient_margin,
                        range_utilization_percent=track_data.range_utilization_percent,
                        minimum_margin=track_data.minimum_margin,
                        minimum_margin_position=track_data.minimum_margin_position,
                        margin_bias=track_data.margin_bias,
                        plot_path=track_data.plot_path
                    )
                    analysis.tracks.append(track)

                # Check for alerts
                self._generate_alerts(analysis, session)

                session.add(analysis)
                session.commit()

                self.logger.info(f"Saved analysis for {analysis_data.filename} with {len(analysis.tracks)} tracks")
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
    ) -> List[AnalysisResult]:
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
            query = session.query(AnalysisResult)

            # Apply filters
            if model:
                query = query.filter(AnalysisResult.model.like(model))

            if serial:
                query = query.filter(AnalysisResult.serial.like(serial))

            # Date filtering
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(AnalysisResult.timestamp >= cutoff_date)
            elif start_date:
                query = query.filter(AnalysisResult.timestamp >= start_date)
                if end_date:
                    query = query.filter(AnalysisResult.timestamp <= end_date)

            if status:
                query = query.filter(AnalysisResult.overall_status == StatusType(status))

            if risk_category and include_tracks:
                # Join with tracks to filter by risk category
                query = query.join(TrackResult).filter(
                    TrackResult.risk_category == RiskCategory(risk_category)
                ).distinct()

            # Order by most recent first
            query = query.order_by(desc(AnalysisResult.timestamp))

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
            base_query = session.query(AnalysisResult).filter(
                AnalysisResult.model == model
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
                func.count(TrackResult.id).label('total_tracks'),
                func.avg(TrackResult.sigma_gradient).label('avg_sigma'),
                func.min(TrackResult.sigma_gradient).label('min_sigma'),
                func.max(TrackResult.sigma_gradient).label('max_sigma'),
                func.sum(TrackResult.sigma_pass).label('sigma_passes'),
                func.sum(TrackResult.linearity_pass).label('linearity_passes'),
                func.avg(TrackResult.failure_probability).label('avg_failure_prob')
            ).join(
                AnalysisResult
            ).filter(
                AnalysisResult.model == model
            ).first()

            # Calculate pass rates
            total_tracks = track_stats.total_tracks or 0
            sigma_pass_rate = (track_stats.sigma_passes / total_tracks * 100) if total_tracks > 0 else 0
            linearity_pass_rate = (track_stats.linearity_passes / total_tracks * 100) if total_tracks > 0 else 0

            # Get risk distribution
            risk_dist = session.query(
                TrackResult.risk_category,
                func.count(TrackResult.id).label('count')
            ).join(
                AnalysisResult
            ).filter(
                AnalysisResult.model == model
            ).group_by(
                TrackResult.risk_category
            ).all()

            risk_distribution = {str(risk): count for risk, count in risk_dist if risk}

            # Get recent trend (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_trend = session.query(
                func.date(AnalysisResult.timestamp).label('date'),
                func.count(AnalysisResult.id).label('count'),
                func.avg(TrackResult.sigma_gradient).label('avg_sigma')
            ).join(
                TrackResult
            ).filter(
                and_(
                    AnalysisResult.model == model,
                    AnalysisResult.timestamp >= thirty_days_ago
                )
            ).group_by(
                func.date(AnalysisResult.timestamp)
            ).order_by(
                func.date(AnalysisResult.timestamp)
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
                        "date": date.isoformat() if date else None,
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
                TrackResult.risk_category,
                func.count(TrackResult.id).label('count'),
                func.avg(TrackResult.failure_probability).label('avg_prob')
            )

            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.join(AnalysisResult).filter(
                    AnalysisResult.timestamp >= cutoff_date
                )

            results = query.group_by(TrackResult.risk_category).all()

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
                AnalysisResult.filename,
                AnalysisResult.model,
                AnalysisResult.serial,
                TrackResult.track_id,
                TrackResult.failure_probability
            ).join(
                TrackResult
            ).filter(
                TrackResult.risk_category == RiskCategory.HIGH
            ).order_by(
                desc(TrackResult.failure_probability)
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

    def save_ml_prediction(self, prediction_data: MLPrediction) -> int:
        """
        Save machine learning prediction results.

        Args:
            prediction_data: Pydantic model containing ML prediction

        Returns:
            ID of the saved prediction record
        """
        with self.get_session() as session:
            prediction = MLPrediction(
                analysis_id=prediction_data.analysis_id,
                model_version=prediction_data.model_version,
                prediction_type=prediction_data.prediction_type,
                current_threshold=prediction_data.current_threshold,
                recommended_threshold=prediction_data.recommended_threshold,
                threshold_change_percent=prediction_data.threshold_change_percent,
                false_positives=prediction_data.false_positives,
                false_negatives=prediction_data.false_negatives,
                predicted_failure_probability=prediction_data.predicted_failure_probability,
                predicted_risk_category=RiskCategory(
                    prediction_data.predicted_risk_category) if prediction_data.predicted_risk_category else None,
                confidence_score=prediction_data.confidence_score,
                feature_importance=prediction_data.feature_importance,
                drift_detected=prediction_data.drift_detected,
                drift_percentage=prediction_data.drift_percentage,
                drift_direction=prediction_data.drift_direction,
                recommendations=prediction_data.recommendations
            )

            session.add(prediction)
            session.commit()

            self.logger.info(f"Saved ML prediction: {prediction_data.prediction_type}")
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
            batch = session.query(BatchInfo).filter(BatchInfo.id == batch_id).first()

            if not batch:
                return None

            # Get linked analyses
            analyses = session.query(AnalysisResult).join(
                AnalysisBatch
            ).filter(
                AnalysisBatch.batch_id == batch_id
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
                    if track.risk_category == RiskCategory.HIGH:
                        high_risk_tracks += 1
                    if track.sigma_gradient:
                        sigma_values.append(track.sigma_gradient)

            # Update batch statistics
            batch.total_units = len(analyses)
            batch.passed_units = sum(1 for a in analyses if a.overall_status == StatusType.PASS)
            batch.failed_units = sum(1 for a in analyses if a.overall_status == StatusType.FAIL)

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
                "created_date": batch.created_date.isoformat() if batch.created_date else None,
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

    def _generate_alerts(self, analysis: AnalysisResult, session: Session) -> None:
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
                alerts.append(QAAlert(
                    alert_type=AlertType.CARBON_SCREEN,
                    severity="High",
                    track_id=track.track_id,
                    metric_name="sigma_gradient",
                    metric_value=track.sigma_gradient,
                    threshold_value=track.sigma_threshold,
                    message=f"Carbon screen check required for {analysis.model}-{analysis.serial} ({track.track_id})"
                ))

            # High risk alert
            if track.risk_category == RiskCategory.HIGH:
                alerts.append(QAAlert(
                    alert_type=AlertType.HIGH_RISK,
                    severity="Critical",
                    track_id=track.track_id,
                    metric_name="failure_probability",
                    metric_value=track.failure_probability,
                    message=f"High risk unit detected: {analysis.model}-{analysis.serial} ({track.track_id})"
                ))

            # Threshold exceeded alert
            if not track.sigma_pass:
                alerts.append(QAAlert(
                    alert_type=AlertType.THRESHOLD_EXCEEDED,
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
    ) -> List[QAAlert]:
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
            query = session.query(QAAlert).filter(QAAlert.resolved == False)

            if severity:
                query = query.filter(QAAlert.severity == severity)

            if alert_type:
                query = query.filter(QAAlert.alert_type == AlertType(alert_type))

            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(QAAlert.created_date >= cutoff_date)

            # Order by severity and date
            query = query.order_by(
                desc(QAAlert.severity),
                desc(QAAlert.created_date)
            )

            return query.all()

    def close(self) -> None:
        """Close database connections and clean up resources."""
        try:
            self.Session.remove()
            self.engine.dispose()
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database: {str(e)}")

 = Get-Content "src\laser_trim_analyzer\database\manager.py" -Raw

# Fix the imports - only import what actually exists in core.models
 = @"


# Note: MLPrediction, QAAlert, and BatchInfo are database-specific models
# and should be imported from the local models module, not core.models


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
            database_url: Optional[str] = None,
            echo: bool = False,
            pool_size: int = 5,
            max_overflow: int = 10,
            pool_timeout: int = 30,
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the database manager.

        Args:
            database_url: SQLAlchemy database URL. If None, uses SQLite default.
            echo: If True, log all SQL statements (useful for debugging)
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum overflow connections allowed
            pool_timeout: Timeout in seconds for getting connection from pool
            logger: Logger instance for database operations
        """
        self.logger = logger or logging.getLogger(__name__)

        # Set up database URL
        if database_url is None:
            # Default to SQLite in user's home directory
            db_dir = Path.home() / ".laser_trim_analyzer"
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / "analyzer_v2.db"
            database_url = f"sqlite:///{db_path}"
            self.logger.info(f"Using SQLite database at: {db_path}")

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
    def get_session(self) -> Session:
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
            inspector = self.engine.inspect(self.engine)
            tables = inspector.get_table_names()
            self.logger.info(f"Created {len(tables)} tables: {', '.join(tables)}")

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise

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
                # Create main analysis record
                analysis = AnalysisResult(
                    filename=analysis_data.filename,
                    file_path=analysis_data.file_path,
                    file_date=analysis_data.file_date,
                    file_hash=analysis_data.file_hash,
                    model=analysis_data.model,
                    serial=analysis_data.serial,
                    system=SystemType(analysis_data.system),
                    has_multi_tracks=analysis_data.has_multi_tracks,
                    overall_status=StatusType(analysis_data.overall_status),
                    processing_time=analysis_data.processing_time,
                    output_dir=analysis_data.output_dir,
                    software_version=analysis_data.software_version,
                    operator=analysis_data.operator,
                    sigma_scaling_factor=analysis_data.sigma_scaling_factor,
                    filter_cutoff_frequency=analysis_data.filter_cutoff_frequency
                )

                # Add track results
                for track_data in analysis_data.tracks:
                    track = TrackResult(
                        track_id=track_data.track_id,
                        status=StatusType(track_data.status),
                        travel_length=track_data.travel_length,
                        linearity_spec=track_data.linearity_spec,
                        sigma_gradient=track_data.sigma_gradient,
                        sigma_threshold=track_data.sigma_threshold,
                        sigma_pass=track_data.sigma_pass,
                        unit_length=track_data.unit_length,
                        untrimmed_resistance=track_data.untrimmed_resistance,
                        trimmed_resistance=track_data.trimmed_resistance,
                        resistance_change=track_data.resistance_change,
                        resistance_change_percent=track_data.resistance_change_percent,
                        optimal_offset=track_data.optimal_offset,
                        final_linearity_error_raw=track_data.final_linearity_error_raw,
                        final_linearity_error_shifted=track_data.final_linearity_error_shifted,
                        linearity_pass=track_data.linearity_pass,
                        linearity_fail_points=track_data.linearity_fail_points,
                        max_deviation=track_data.max_deviation,
                        max_deviation_position=track_data.max_deviation_position,
                        deviation_uniformity=track_data.deviation_uniformity,
                        trim_improvement_percent=track_data.trim_improvement_percent,
                        untrimmed_rms_error=track_data.untrimmed_rms_error,
                        trimmed_rms_error=track_data.trimmed_rms_error,
                        max_error_reduction_percent=track_data.max_error_reduction_percent,
                        worst_zone=track_data.worst_zone,
                        worst_zone_position=track_data.worst_zone_position,
                        zone_details=track_data.zone_details,
                        failure_probability=track_data.failure_probability,
                        risk_category=RiskCategory(track_data.risk_category) if track_data.risk_category else None,
                        gradient_margin=track_data.gradient_margin,
                        range_utilization_percent=track_data.range_utilization_percent,
                        minimum_margin=track_data.minimum_margin,
                        minimum_margin_position=track_data.minimum_margin_position,
                        margin_bias=track_data.margin_bias,
                        plot_path=track_data.plot_path
                    )
                    analysis.tracks.append(track)

                # Check for alerts
                self._generate_alerts(analysis, session)

                session.add(analysis)
                session.commit()

                self.logger.info(f"Saved analysis for {analysis_data.filename} with {len(analysis.tracks)} tracks")
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
    ) -> List[AnalysisResult]:
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
            query = session.query(AnalysisResult)

            # Apply filters
            if model:
                query = query.filter(AnalysisResult.model.like(model))

            if serial:
                query = query.filter(AnalysisResult.serial.like(serial))

            # Date filtering
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(AnalysisResult.timestamp >= cutoff_date)
            elif start_date:
                query = query.filter(AnalysisResult.timestamp >= start_date)
                if end_date:
                    query = query.filter(AnalysisResult.timestamp <= end_date)

            if status:
                query = query.filter(AnalysisResult.overall_status == StatusType(status))

            if risk_category and include_tracks:
                # Join with tracks to filter by risk category
                query = query.join(TrackResult).filter(
                    TrackResult.risk_category == RiskCategory(risk_category)
                ).distinct()

            # Order by most recent first
            query = query.order_by(desc(AnalysisResult.timestamp))

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
            base_query = session.query(AnalysisResult).filter(
                AnalysisResult.model == model
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
                func.count(TrackResult.id).label('total_tracks'),
                func.avg(TrackResult.sigma_gradient).label('avg_sigma'),
                func.min(TrackResult.sigma_gradient).label('min_sigma'),
                func.max(TrackResult.sigma_gradient).label('max_sigma'),
                func.sum(TrackResult.sigma_pass).label('sigma_passes'),
                func.sum(TrackResult.linearity_pass).label('linearity_passes'),
                func.avg(TrackResult.failure_probability).label('avg_failure_prob')
            ).join(
                AnalysisResult
            ).filter(
                AnalysisResult.model == model
            ).first()

            # Calculate pass rates
            total_tracks = track_stats.total_tracks or 0
            sigma_pass_rate = (track_stats.sigma_passes / total_tracks * 100) if total_tracks > 0 else 0
            linearity_pass_rate = (track_stats.linearity_passes / total_tracks * 100) if total_tracks > 0 else 0

            # Get risk distribution
            risk_dist = session.query(
                TrackResult.risk_category,
                func.count(TrackResult.id).label('count')
            ).join(
                AnalysisResult
            ).filter(
                AnalysisResult.model == model
            ).group_by(
                TrackResult.risk_category
            ).all()

            risk_distribution = {str(risk): count for risk, count in risk_dist if risk}

            # Get recent trend (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_trend = session.query(
                func.date(AnalysisResult.timestamp).label('date'),
                func.count(AnalysisResult.id).label('count'),
                func.avg(TrackResult.sigma_gradient).label('avg_sigma')
            ).join(
                TrackResult
            ).filter(
                and_(
                    AnalysisResult.model == model,
                    AnalysisResult.timestamp >= thirty_days_ago
                )
            ).group_by(
                func.date(AnalysisResult.timestamp)
            ).order_by(
                func.date(AnalysisResult.timestamp)
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
                        "date": date.isoformat() if date else None,
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
                TrackResult.risk_category,
                func.count(TrackResult.id).label('count'),
                func.avg(TrackResult.failure_probability).label('avg_prob')
            )

            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.join(AnalysisResult).filter(
                    AnalysisResult.timestamp >= cutoff_date
                )

            results = query.group_by(TrackResult.risk_category).all()

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
                AnalysisResult.filename,
                AnalysisResult.model,
                AnalysisResult.serial,
                TrackResult.track_id,
                TrackResult.failure_probability
            ).join(
                TrackResult
            ).filter(
                TrackResult.risk_category == RiskCategory.HIGH
            ).order_by(
                desc(TrackResult.failure_probability)
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

    def save_ml_prediction(self, prediction_data: MLPrediction) -> int:
        """
        Save machine learning prediction results.

        Args:
            prediction_data: Pydantic model containing ML prediction

        Returns:
            ID of the saved prediction record
        """
        with self.get_session() as session:
            prediction = MLPrediction(
                analysis_id=prediction_data.analysis_id,
                model_version=prediction_data.model_version,
                prediction_type=prediction_data.prediction_type,
                current_threshold=prediction_data.current_threshold,
                recommended_threshold=prediction_data.recommended_threshold,
                threshold_change_percent=prediction_data.threshold_change_percent,
                false_positives=prediction_data.false_positives,
                false_negatives=prediction_data.false_negatives,
                predicted_failure_probability=prediction_data.predicted_failure_probability,
                predicted_risk_category=RiskCategory(
                    prediction_data.predicted_risk_category) if prediction_data.predicted_risk_category else None,
                confidence_score=prediction_data.confidence_score,
                feature_importance=prediction_data.feature_importance,
                drift_detected=prediction_data.drift_detected,
                drift_percentage=prediction_data.drift_percentage,
                drift_direction=prediction_data.drift_direction,
                recommendations=prediction_data.recommendations
            )

            session.add(prediction)
            session.commit()

            self.logger.info(f"Saved ML prediction: {prediction_data.prediction_type}")
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
            batch = session.query(BatchInfo).filter(BatchInfo.id == batch_id).first()

            if not batch:
                return None

            # Get linked analyses
            analyses = session.query(AnalysisResult).join(
                AnalysisBatch
            ).filter(
                AnalysisBatch.batch_id == batch_id
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
                    if track.risk_category == RiskCategory.HIGH:
                        high_risk_tracks += 1
                    if track.sigma_gradient:
                        sigma_values.append(track.sigma_gradient)

            # Update batch statistics
            batch.total_units = len(analyses)
            batch.passed_units = sum(1 for a in analyses if a.overall_status == StatusType.PASS)
            batch.failed_units = sum(1 for a in analyses if a.overall_status == StatusType.FAIL)

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
                "created_date": batch.created_date.isoformat() if batch.created_date else None,
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

    def _generate_alerts(self, analysis: AnalysisResult, session: Session) -> None:
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
                alerts.append(QAAlert(
                    alert_type=AlertType.CARBON_SCREEN,
                    severity="High",
                    track_id=track.track_id,
                    metric_name="sigma_gradient",
                    metric_value=track.sigma_gradient,
                    threshold_value=track.sigma_threshold,
                    message=f"Carbon screen check required for {analysis.model}-{analysis.serial} ({track.track_id})"
                ))

            # High risk alert
            if track.risk_category == RiskCategory.HIGH:
                alerts.append(QAAlert(
                    alert_type=AlertType.HIGH_RISK,
                    severity="Critical",
                    track_id=track.track_id,
                    metric_name="failure_probability",
                    metric_value=track.failure_probability,
                    message=f"High risk unit detected: {analysis.model}-{analysis.serial} ({track.track_id})"
                ))

            # Threshold exceeded alert
            if not track.sigma_pass:
                alerts.append(QAAlert(
                    alert_type=AlertType.THRESHOLD_EXCEEDED,
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
    ) -> List[QAAlert]:
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
            query = session.query(QAAlert).filter(QAAlert.resolved == False)

            if severity:
                query = query.filter(QAAlert.severity == severity)

            if alert_type:
                query = query.filter(QAAlert.alert_type == AlertType(alert_type))

            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(QAAlert.created_date >= cutoff_date)

            # Order by severity and date
            query = query.order_by(
                desc(QAAlert.severity),
                desc(QAAlert.created_date)
            )

            return query.all()

    def close(self) -> None:
        """Close database connections and clean up resources."""
        try:
            self.Session.remove()
            self.engine.dispose()
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database: {str(e)}")
