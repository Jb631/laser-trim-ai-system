"""
Database Manager for Laser Trim Analyzer v2.

Handles all database operations with connection pooling, error handling,
and migration support. Designed for QA specialists to easily store and
retrieve potentiometer test results.

Production-ready implementation with proper error handling and real data operations only.
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

from sqlalchemy import create_engine, func, and_, or_, desc, inspect, text
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
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


class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Exception for database connection issues."""
    pass


class DatabaseIntegrityError(DatabaseError):
    """Exception for database integrity violations."""
    pass


class DatabaseManager:
    """
    Production-ready database manager for the Laser Trim Analyzer.

    Features:
    - Connection pooling for better performance
    - Context managers for safe transaction handling
    - Comprehensive error handling and logging
    - Migration support ready
    - Optimized queries with proper indexing
    - Production-ready CRUD operations
    - Proper empty database handling
    """

    def __init__(
            self,
            database_url_or_config: Optional[Union[str, Any]] = None,
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

        Raises:
            DatabaseConnectionError: If database connection cannot be established
        """
        self.logger = logger or logging.getLogger(__name__)
        self._engine = None
        self._session_factory = None
        self._Session = None

        try:
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
            self._initialize_engine(database_url, echo, pool_size, max_overflow, pool_timeout)
            self._test_connection()

        except Exception as e:
            self.logger.error(f"Failed to initialize database manager: {str(e)}")
            raise DatabaseConnectionError(f"Database initialization failed: {str(e)}") from e

    def _initialize_engine(self, database_url: str, echo: bool, pool_size: int, 
                          max_overflow: int, pool_timeout: int) -> None:
        """Initialize the database engine with proper configuration."""
        engine_kwargs = {
            "echo": echo,
            "future": True,  # Use SQLAlchemy 2.0 style
        }

        # Configure pooling based on database type
        if database_url.startswith("sqlite"):
            # SQLite doesn't benefit from connection pooling
            engine_kwargs["connect_args"] = {
                "check_same_thread": False,
                "timeout": 30  # 30 second timeout for SQLite operations
            }
        else:
            # Use connection pooling for other databases
            engine_kwargs["poolclass"] = QueuePool
            engine_kwargs["pool_size"] = pool_size
            engine_kwargs["max_overflow"] = max_overflow
            engine_kwargs["pool_timeout"] = pool_timeout
            engine_kwargs["pool_pre_ping"] = True  # Verify connections before use

        self._engine = create_engine(database_url, **engine_kwargs)

        # Create session factory
        self._session_factory = sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            autoflush=False
        )

        # Create scoped session for thread safety
        self._Session = scoped_session(self._session_factory)

    def _test_connection(self) -> None:
        """Test database connection and raise error if it fails."""
        try:
            with self._engine.connect() as conn:
                # Simple test query
                if self.database_url.startswith("sqlite"):
                    conn.execute(text("SELECT 1"))
                else:
                    conn.execute(text("SELECT 1"))
            self.logger.info("Database connection test successful")
        except Exception as e:
            raise DatabaseConnectionError(f"Database connection test failed: {str(e)}") from e

    @property
    def engine(self) -> Engine:
        """Get the database engine."""
        if self._engine is None:
            raise DatabaseConnectionError("Database engine not initialized")
        return self._engine

    @property
    def Session(self):
        """Get the scoped session class."""
        if self._Session is None:
            raise DatabaseConnectionError("Database session not initialized")
        return self._Session

    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """
        Context manager for database sessions with comprehensive error handling.

        Ensures proper session cleanup and error handling for production use.

        Usage:
            with db_manager.get_session() as session:
                # Perform database operations
                session.add(record)
                session.commit()

        Raises:
            DatabaseError: For database-related errors
            DatabaseIntegrityError: For constraint violations
        """
        if self._Session is None:
            raise DatabaseConnectionError("Database not initialized")

        session = self._Session()
        try:
            yield session
        except IntegrityError as e:
            session.rollback()
            self.logger.error(f"Database integrity error: {str(e)}")
            raise DatabaseIntegrityError(f"Data integrity violation: {str(e)}") from e
        except OperationalError as e:
            session.rollback()
            self.logger.error(f"Database operational error: {str(e)}")
            raise DatabaseConnectionError(f"Database operation failed: {str(e)}") from e
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Database error: {str(e)}")
            raise DatabaseError(f"Database operation failed: {str(e)}") from e
        except Exception as e:
            session.rollback()
            self.logger.error(f"Unexpected error in database session: {str(e)}")
            raise DatabaseError(f"Unexpected database error: {str(e)}") from e
        finally:
            try:
                session.close()
            except Exception as e:
                self.logger.warning(f"Error closing database session: {str(e)}")

    def init_db(self, drop_existing: bool = False) -> None:
        """
        Initialize database tables for production use.

        Args:
            drop_existing: If True, drop all existing tables first (USE WITH EXTREME CAUTION!)

        Raises:
            DatabaseError: If database initialization fails
        """
        try:
            if drop_existing:
                self.logger.warning("DROPPING ALL EXISTING TABLES - THIS WILL DELETE ALL DATA!")
                Base.metadata.drop_all(self.engine)
                self.logger.warning("All tables dropped")

            self.logger.info("Creating database tables...")
            Base.metadata.create_all(self.engine)

            # Verify tables were created
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            if not tables:
                raise DatabaseError("No tables were created")

            expected_tables = {
                'analysis_results', 'track_results', 'ml_predictions', 
                'qa_alerts', 'batch_info', 'analysis_batch'
            }
            
            missing_tables = expected_tables - set(tables)
            if missing_tables:
                raise DatabaseError(f"Missing expected tables: {missing_tables}")

            self.logger.info(f"Successfully created {len(tables)} tables: {', '.join(sorted(tables))}")

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise DatabaseError(f"Database initialization failed: {str(e)}") from e

    def initialize_schema(self) -> None:
        """
        Initialize database schema for production use.
        
        This method ensures all tables are created if they don't exist.
        Safe to call multiple times.

        Raises:
            DatabaseError: If schema initialization fails
        """
        try:
            self.init_db(drop_existing=False)
        except Exception as e:
            raise DatabaseError(f"Schema initialization failed: {str(e)}") from e

    def check_duplicate_analysis(
            self,
            model: str,
            serial: str,
            file_date: datetime
    ) -> Optional[int]:
        """
        Check if this unit has already been analyzed.

        Args:
            model: Model number (required, non-empty)
            serial: Serial number (required, non-empty)
            file_date: Date from the file (required)

        Returns:
            ID of existing analysis if found, None otherwise

        Raises:
            DatabaseError: If database operation fails
            ValueError: If required parameters are missing or invalid
        """
        if not model or not model.strip():
            raise ValueError("Model number is required and cannot be empty")
        if not serial or not serial.strip():
            raise ValueError("Serial number is required and cannot be empty")
        if not file_date:
            raise ValueError("File date is required")

        try:
            with self.get_session() as session:
                # Check for exact match (model + serial + file date)
                existing = session.query(DBAnalysisResult).filter(
                    and_(
                        DBAnalysisResult.model == model.strip(),
                        DBAnalysisResult.serial == serial.strip(),
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

        except Exception as e:
            self.logger.error(f"Failed to check for duplicate analysis: {str(e)}")
            raise DatabaseError(f"Duplicate check failed: {str(e)}") from e

    def save_analysis(self, analysis_data: PydanticAnalysisResult) -> int:
        """
        Save a complete analysis result with all tracks to production database.

        Args:
            analysis_data: Pydantic model containing analysis results (validated)

        Returns:
            ID of the saved analysis record

        Raises:
            DatabaseError: If save operation fails
            DatabaseIntegrityError: If data violates constraints
            ValueError: If analysis data is invalid
        """
        if not analysis_data:
            raise ValueError("Analysis data is required")
        
        if not analysis_data.metadata:
            raise ValueError("Analysis metadata is required")
            
        if not analysis_data.metadata.model or not analysis_data.metadata.model.strip():
            raise ValueError("Model number is required in analysis metadata")
            
        if not analysis_data.metadata.serial or not analysis_data.metadata.serial.strip():
            raise ValueError("Serial number is required in analysis metadata")

        try:
            with self.get_session() as session:
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
                    file_path=str(analysis_data.metadata.file_path) if analysis_data.metadata.file_path else None,
                    file_date=analysis_data.metadata.file_date,
                    file_hash=None,  # Can be calculated if needed
                    model=analysis_data.metadata.model.strip(),
                    serial=analysis_data.metadata.serial.strip(),
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

                # Validate and add track results
                if not analysis_data.tracks:
                    raise ValueError("Analysis must contain at least one track")

                for track_id, track_data in analysis_data.tracks.items():
                    if not track_id or not track_id.strip():
                        raise ValueError("Track ID cannot be empty")
                        
                    if track_data.sigma_analysis is None:
                        raise ValueError(f"Sigma analysis is required for track {track_id}")

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
                        track_id=track_id.strip(),
                        status=track_status,
                        travel_length=track_data.travel_length,
                        linearity_spec=track_data.linearity_analysis.linearity_spec if track_data.linearity_analysis else None,
                        sigma_gradient=track_data.sigma_analysis.sigma_gradient,
                        sigma_threshold=track_data.sigma_analysis.sigma_threshold,
                        sigma_pass=track_data.sigma_analysis.sigma_pass,
                        unit_length=track_data.unit_properties.unit_length if track_data.unit_properties else None,
                        untrimmed_resistance=track_data.unit_properties.untrimmed_resistance if track_data.unit_properties else None,
                        trimmed_resistance=track_data.unit_properties.trimmed_resistance if track_data.unit_properties else None,
                        resistance_change=track_data.unit_properties.resistance_change if track_data.unit_properties else None,
                        resistance_change_percent=track_data.unit_properties.resistance_change_percent if track_data.unit_properties else None,
                        optimal_offset=track_data.linearity_analysis.optimal_offset if track_data.linearity_analysis else None,
                        final_linearity_error_raw=track_data.linearity_analysis.final_linearity_error_raw if track_data.linearity_analysis else None,
                        final_linearity_error_shifted=track_data.linearity_analysis.final_linearity_error_shifted if track_data.linearity_analysis else None,
                        linearity_pass=track_data.linearity_analysis.linearity_pass if track_data.linearity_analysis else None,
                        linearity_fail_points=track_data.linearity_analysis.linearity_fail_points if track_data.linearity_analysis else None,
                        max_deviation=track_data.linearity_analysis.max_deviation if track_data.linearity_analysis else None,
                        max_deviation_position=track_data.linearity_analysis.max_deviation_position if track_data.linearity_analysis else None,
                        deviation_uniformity=None,  # Calculate if needed
                        trim_improvement_percent=track_data.trim_effectiveness.improvement_percent if track_data.trim_effectiveness else None,
                        untrimmed_rms_error=track_data.trim_effectiveness.untrimmed_rms_error if track_data.trim_effectiveness else None,
                        trimmed_rms_error=track_data.trim_effectiveness.trimmed_rms_error if track_data.trim_effectiveness else None,
                        max_error_reduction_percent=track_data.trim_effectiveness.max_error_reduction_percent if track_data.trim_effectiveness else None,
                        worst_zone=track_data.zone_analysis.worst_zone if track_data.zone_analysis else None,
                        worst_zone_position=track_data.zone_analysis.worst_zone_position[0] if (track_data.zone_analysis and track_data.zone_analysis.worst_zone_position) else None,
                        zone_details=track_data.zone_analysis.zone_results if track_data.zone_analysis else None,
                        failure_probability=track_data.failure_prediction.failure_probability if track_data.failure_prediction else None,
                        risk_category=risk_category,
                        gradient_margin=track_data.sigma_analysis.gradient_margin if hasattr(track_data.sigma_analysis, 'gradient_margin') else None,
                        range_utilization_percent=track_data.dynamic_range.range_utilization_percent if track_data.dynamic_range else None,
                        minimum_margin=track_data.dynamic_range.minimum_margin if track_data.dynamic_range else None,
                        minimum_margin_position=track_data.dynamic_range.minimum_margin_position if track_data.dynamic_range else None,
                        margin_bias=track_data.dynamic_range.margin_bias if track_data.dynamic_range else None,
                        plot_path=str(track_data.plot_path) if track_data.plot_path else None
                    )
                    analysis.tracks.append(track)

                # Generate alerts based on real analysis data
                self._generate_production_alerts(analysis, session)

                session.add(analysis)
                session.commit()

                self.logger.info(
                    f"Successfully saved analysis for {analysis_data.metadata.filename} "
                    f"with {len(analysis.tracks)} tracks (ID: {analysis.id})"
                )
                return analysis.id

        except ValueError as e:
            raise e  # Re-raise validation errors as-is
        except Exception as e:
            self.logger.error(f"Failed to save analysis: {str(e)}")
            raise DatabaseError(f"Analysis save failed: {str(e)}") from e

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
        Retrieve historical analysis data with flexible filtering for production use.

        Args:
            model: Filter by model number (supports wildcards with %)
            serial: Filter by serial number (supports wildcards with %)
            days_back: Number of days to look back from today (must be positive)
            start_date: Start date for date range filter
            end_date: End date for date range filter
            status: Filter by overall status
            risk_category: Filter by risk category in tracks
            limit: Maximum number of records to return (must be positive)
            offset: Number of records to skip for pagination (must be non-negative)
            include_tracks: Whether to include track details

        Returns:
            List of AnalysisResult objects (empty list if no data found)

        Raises:
            DatabaseError: If database operation fails
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if days_back is not None and days_back <= 0:
            raise ValueError("days_back must be positive")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        if offset is not None and offset < 0:
            raise ValueError("offset must be non-negative")
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date cannot be after end_date")

        try:
            with self.get_session() as session:
                query = session.query(DBAnalysisResult)

                # Apply filters
                if model:
                    query = query.filter(DBAnalysisResult.model.like(model.strip()))

                if serial:
                    query = query.filter(DBAnalysisResult.serial.like(serial.strip()))

                # Date filtering
                if days_back:
                    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                    query = query.filter(DBAnalysisResult.timestamp >= cutoff_date)
                elif start_date:
                    query = query.filter(DBAnalysisResult.timestamp >= start_date)
                    if end_date:
                        query = query.filter(DBAnalysisResult.timestamp <= end_date)

                if status:
                    try:
                        status_enum = DBStatusType(status)
                        query = query.filter(DBAnalysisResult.overall_status == status_enum)
                    except ValueError:
                        self.logger.warning(f"Invalid status filter: {status}")
                        return []

                if risk_category and include_tracks:
                    try:
                        risk_enum = DBRiskCategory(risk_category)
                        # Join with tracks to filter by risk category
                        query = query.join(DBTrackResult).filter(
                            DBTrackResult.risk_category == risk_enum
                        ).distinct()
                    except ValueError:
                        self.logger.warning(f"Invalid risk category filter: {risk_category}")
                        return []

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
                if include_tracks and results:
                    for result in results:
                        # Access tracks to trigger loading
                        _ = result.tracks

                self.logger.info(f"Retrieved {len(results)} historical records")
                return results

        except Exception as e:
            self.logger.error(f"Failed to retrieve historical data: {str(e)}")
            raise DatabaseError(f"Historical data retrieval failed: {str(e)}") from e

    def get_model_statistics(self, model: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a specific model from production data.

        Args:
            model: Model number to analyze (required, non-empty)

        Returns:
            Dictionary containing model statistics (empty stats if no data found)

        Raises:
            DatabaseError: If database operation fails
            ValueError: If model parameter is invalid
        """
        if not model or not model.strip():
            raise ValueError("Model number is required and cannot be empty")

        try:
            with self.get_session() as session:
                model = model.strip()
                
                # Base query for the model
                base_query = session.query(DBAnalysisResult).filter(
                    DBAnalysisResult.model == model
                )

                # Get basic counts
                total_files = base_query.count()

                if total_files == 0:
                    self.logger.info(f"No data found for model: {model}")
                    return {
                        "model": model,
                        "total_files": 0,
                        "total_tracks": 0,
                        "statistics": {},
                        "recent_trend": []
                    }

                # Get track statistics
                track_stats = session.query(
                    func.count(DBTrackResult.id).label('total_tracks'),
                    func.avg(DBTrackResult.sigma_gradient).label('avg_sigma'),
                    func.min(DBTrackResult.sigma_gradient).label('min_sigma'),
                    func.max(DBTrackResult.sigma_gradient).label('max_sigma'),
                    func.sum(func.cast(DBTrackResult.sigma_pass, Integer)).label('sigma_passes'),
                    func.sum(func.cast(DBTrackResult.linearity_pass, Integer)).label('linearity_passes'),
                    func.avg(DBTrackResult.failure_probability).label('avg_failure_prob')
                ).join(
                    DBAnalysisResult
                ).filter(
                    DBAnalysisResult.model == model
                ).first()

                # Calculate pass rates safely
                total_tracks = track_stats.total_tracks or 0
                sigma_passes = track_stats.sigma_passes or 0
                linearity_passes = track_stats.linearity_passes or 0
                
                sigma_pass_rate = (sigma_passes / total_tracks * 100) if total_tracks > 0 else 0
                linearity_pass_rate = (linearity_passes / total_tracks * 100) if total_tracks > 0 else 0

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

                risk_distribution = {}
                for risk, count in risk_dist:
                    if risk:
                        risk_distribution[str(risk.value)] = count

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

                trend_data = []
                for date, count, avg_sigma in recent_trend:
                    trend_data.append({
                        "date": date.isoformat() if date and hasattr(date, 'isoformat') else str(date) if date else None,
                        "count": count or 0,
                        "avg_sigma": float(avg_sigma or 0)
                    })

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
                            "sigma": round(sigma_pass_rate, 2),
                            "linearity": round(linearity_pass_rate, 2)
                        },
                        "failure_probability": {
                            "average": float(track_stats.avg_failure_prob or 0)
                        },
                        "risk_distribution": risk_distribution
                    },
                    "recent_trend": trend_data
                }

        except Exception as e:
            self.logger.error(f"Failed to get model statistics for {model}: {str(e)}")
            raise DatabaseError(f"Model statistics retrieval failed: {str(e)}") from e

    def get_risk_summary(self, days_back: Optional[int] = None) -> Dict[str, Any]:
        """
        Get summary of units by risk category from production data.

        Args:
            days_back: Limit to records from last N days (must be positive if provided)

        Returns:
            Dictionary with risk category counts and details

        Raises:
            DatabaseError: If database operation fails
            ValueError: If days_back is invalid
        """
        if days_back is not None and days_back <= 0:
            raise ValueError("days_back must be positive")

        try:
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
                    "period_days": days_back,
                    "high_risk_units": []
                }

                for risk_category, count, avg_prob in results:
                    if risk_category:
                        category_name = risk_category.value
                        summary["categories"][category_name] = {
                            "count": count or 0,
                            "percentage": 0,  # Will calculate after total
                            "avg_failure_probability": float(avg_prob or 0)
                        }
                        summary["total"] += (count or 0)

                # Calculate percentages
                if summary["total"] > 0:
                    for category in summary["categories"].values():
                        category["percentage"] = round((category["count"] / summary["total"]) * 100, 2)

                # Get high-risk units details (limit to prevent excessive data)
                high_risk_query = session.query(
                    DBAnalysisResult.filename,
                    DBAnalysisResult.model,
                    DBAnalysisResult.serial,
                    DBTrackResult.track_id,
                    DBTrackResult.failure_probability
                ).join(
                    DBTrackResult
                ).filter(
                    DBTrackResult.risk_category == DBRiskCategory.HIGH
                )

                if days_back:
                    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                    high_risk_query = high_risk_query.filter(
                        DBAnalysisResult.timestamp >= cutoff_date
                    )

                high_risk_units = high_risk_query.order_by(
                    desc(DBTrackResult.failure_probability)
                ).limit(10).all()

                summary["high_risk_units"] = [
                    {
                        "filename": filename,
                        "model": model,
                        "serial": serial,
                        "track_id": track_id,
                        "failure_probability": float(prob or 0)
                    }
                    for filename, model, serial, track_id, prob in high_risk_units
                ]

                return summary

        except Exception as e:
            self.logger.error(f"Failed to get risk summary: {str(e)}")
            raise DatabaseError(f"Risk summary retrieval failed: {str(e)}") from e

    def save_ml_prediction(self, prediction_data: Dict[str, Any]) -> int:
        """
        Save machine learning prediction results to production database.

        Args:
            prediction_data: Dictionary containing ML prediction data (validated)

        Returns:
            ID of the saved prediction record

        Raises:
            DatabaseError: If save operation fails
            ValueError: If prediction data is invalid
        """
        if not prediction_data:
            raise ValueError("Prediction data is required")
            
        if 'analysis_id' not in prediction_data:
            raise ValueError("analysis_id is required in prediction data")

        try:
            with self.get_session() as session:
                # Verify analysis exists
                analysis_exists = session.query(DBAnalysisResult).filter(
                    DBAnalysisResult.id == prediction_data['analysis_id']
                ).first()
                
                if not analysis_exists:
                    raise ValueError(f"Analysis ID {prediction_data['analysis_id']} does not exist")

                # Convert risk category if present
                risk_category = None
                if prediction_data.get('predicted_risk_category'):
                    try:
                        risk_category = DBRiskCategory(prediction_data['predicted_risk_category'])
                    except ValueError:
                        self.logger.warning(f"Invalid risk category: {prediction_data['predicted_risk_category']}")

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

                self.logger.info(f"Saved ML prediction: {prediction_data.get('prediction_type')} (ID: {prediction.id})")
                return prediction.id

        except ValueError as e:
            raise e  # Re-raise validation errors as-is
        except Exception as e:
            self.logger.error(f"Failed to save ML prediction: {str(e)}")
            raise DatabaseError(f"ML prediction save failed: {str(e)}") from e

    def get_batch_statistics(self, batch_id: int) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific batch from production data.

        Args:
            batch_id: ID of the batch (must be positive)

        Returns:
            Dictionary with batch statistics or None if not found

        Raises:
            DatabaseError: If database operation fails
            ValueError: If batch_id is invalid
        """
        if not isinstance(batch_id, int) or batch_id <= 0:
            raise ValueError("batch_id must be a positive integer")

        try:
            with self.get_session() as session:
                batch = session.query(DBBatchInfo).filter(DBBatchInfo.id == batch_id).first()

                if not batch:
                    self.logger.info(f"Batch ID {batch_id} not found")
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
                        if track.sigma_gradient is not None:
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
                        "total_units": batch.total_units or 0,
                        "passed_units": batch.passed_units or 0,
                        "failed_units": batch.failed_units or 0,
                        "pass_rate": round((batch.passed_units / batch.total_units * 100), 2) if batch.total_units and batch.total_units > 0 else 0,
                        "average_sigma_gradient": round(batch.average_sigma_gradient, 4) if batch.average_sigma_gradient else 0,
                        "sigma_pass_rate": round(batch.sigma_pass_rate, 2) if batch.sigma_pass_rate else 0,
                        "high_risk_count": batch.high_risk_count or 0
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

        except Exception as e:
            self.logger.error(f"Failed to get batch statistics for batch {batch_id}: {str(e)}")
            raise DatabaseError(f"Batch statistics retrieval failed: {str(e)}") from e

    def _generate_production_alerts(self, analysis: DBAnalysisResult, session: Session) -> None:
        """
        Generate QA alerts based on real analysis results for production use.

        Args:
            analysis: Analysis result to check (validated)
            session: Database session (active transaction)
        """
        if not analysis or not analysis.tracks:
            return

        alerts = []

        try:
            for track in analysis.tracks:
                if not track:
                    continue

                # Carbon screen check for 8340 models (production rule)
                if ("8340" in analysis.model and 
                    track.sigma_gradient is not None and 
                    track.sigma_threshold is not None and 
                    track.sigma_gradient > track.sigma_threshold):
                    
                    alerts.append(DBQAAlert(
                        alert_type=DBAlertType.CARBON_SCREEN,
                        severity="High",
                        track_id=track.track_id,
                        metric_name="sigma_gradient",
                        metric_value=track.sigma_gradient,
                        threshold_value=track.sigma_threshold,
                        message=f"Carbon screen check required for {analysis.model}-{analysis.serial} ({track.track_id})"
                    ))

                # High risk alert (production critical)
                if track.risk_category == DBRiskCategory.HIGH:
                    alerts.append(DBQAAlert(
                        alert_type=DBAlertType.HIGH_RISK,
                        severity="Critical",
                        track_id=track.track_id,
                        metric_name="failure_probability",
                        metric_value=track.failure_probability,
                        message=f"High risk unit detected: {analysis.model}-{analysis.serial} ({track.track_id})"
                    ))

                # Threshold exceeded alert (production quality)
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
                self.logger.info(f"Generated production alert: {alert.alert_type.value} for {analysis.filename}")

        except Exception as e:
            self.logger.error(f"Failed to generate alerts for analysis {analysis.id}: {str(e)}")
            # Don't raise - alerts are supplementary

    def get_unresolved_alerts(
            self,
            severity: Optional[str] = None,
            alert_type: Optional[str] = None,
            days_back: Optional[int] = None
    ) -> List[DBQAAlert]:
        """
        Get unresolved QA alerts from production data.

        Args:
            severity: Filter by severity level
            alert_type: Filter by alert type
            days_back: Limit to alerts from last N days (must be positive if provided)

        Returns:
            List of unresolved QAAlert objects (empty list if none found)

        Raises:
            DatabaseError: If database operation fails
            ValueError: If parameters are invalid
        """
        if days_back is not None and days_back <= 0:
            raise ValueError("days_back must be positive")

        try:
            with self.get_session() as session:
                query = session.query(DBQAAlert).filter(DBQAAlert.resolved == False)

                if severity:
                    query = query.filter(DBQAAlert.severity == severity.strip())

                if alert_type:
                    try:
                        alert_enum = DBAlertType(alert_type)
                        query = query.filter(DBQAAlert.alert_type == alert_enum)
                    except ValueError:
                        self.logger.warning(f"Invalid alert type filter: {alert_type}")
                        return []

                if days_back:
                    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                    query = query.filter(DBQAAlert.created_date >= cutoff_date)

                # Order by severity and date (most critical first)
                severity_order = func.case(
                    (DBQAAlert.severity == 'Critical', 1),
                    (DBQAAlert.severity == 'High', 2),
                    (DBQAAlert.severity == 'Medium', 3),
                    (DBQAAlert.severity == 'Low', 4),
                    else_=5
                )
                
                query = query.order_by(
                    severity_order,
                    desc(DBQAAlert.created_date)
                )

                results = query.all()
                self.logger.info(f"Retrieved {len(results)} unresolved alerts")
                return results

        except Exception as e:
            self.logger.error(f"Failed to get unresolved alerts: {str(e)}")
            raise DatabaseError(f"Unresolved alerts retrieval failed: {str(e)}") from e

    def create_batch(self, batch_name: str, model: str, batch_type: str = "production") -> int:
        """
        Create a new batch for grouping analyses in production.

        Args:
            batch_name: Name of the batch (required, non-empty, unique)
            model: Model number for the batch (required, non-empty)
            batch_type: Type of batch (production, rework, test)

        Returns:
            ID of the created batch

        Raises:
            DatabaseError: If batch creation fails
            DatabaseIntegrityError: If batch name already exists
            ValueError: If parameters are invalid
        """
        if not batch_name or not batch_name.strip():
            raise ValueError("Batch name is required and cannot be empty")
        if not model or not model.strip():
            raise ValueError("Model number is required and cannot be empty")
        if batch_type not in ["production", "rework", "test"]:
            raise ValueError("batch_type must be one of: production, rework, test")

        try:
            with self.get_session() as session:
                # Check if batch name already exists
                existing = session.query(DBBatchInfo).filter(
                    DBBatchInfo.batch_name == batch_name.strip()
                ).first()
                
                if existing:
                    raise DatabaseIntegrityError(f"Batch name '{batch_name}' already exists")

                batch = DBBatchInfo(
                    batch_name=batch_name.strip(),
                    batch_type=batch_type,
                    model=model.strip()
                )
                session.add(batch)
                session.commit()

                self.logger.info(f"Created batch: {batch_name} (ID: {batch.id})")
                return batch.id

        except DatabaseIntegrityError:
            raise  # Re-raise integrity errors as-is
        except Exception as e:
            self.logger.error(f"Failed to create batch {batch_name}: {str(e)}")
            raise DatabaseError(f"Batch creation failed: {str(e)}") from e

    def add_analysis_to_batch(self, analysis_id: int, batch_id: int) -> None:
        """
        Add an analysis to a batch in production database.

        Args:
            analysis_id: ID of the analysis (must exist and be positive)
            batch_id: ID of the batch (must exist and be positive)

        Raises:
            DatabaseError: If operation fails
            ValueError: If IDs are invalid or don't exist
        """
        if not isinstance(analysis_id, int) or analysis_id <= 0:
            raise ValueError("analysis_id must be a positive integer")
        if not isinstance(batch_id, int) or batch_id <= 0:
            raise ValueError("batch_id must be a positive integer")

        try:
            with self.get_session() as session:
                # Verify analysis exists
                analysis_exists = session.query(DBAnalysisResult).filter(
                    DBAnalysisResult.id == analysis_id
                ).first()
                if not analysis_exists:
                    raise ValueError(f"Analysis ID {analysis_id} does not exist")

                # Verify batch exists
                batch_exists = session.query(DBBatchInfo).filter(
                    DBBatchInfo.id == batch_id
                ).first()
                if not batch_exists:
                    raise ValueError(f"Batch ID {batch_id} does not exist")

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
                else:
                    self.logger.info(f"Analysis {analysis_id} already in batch {batch_id}")

        except ValueError as e:
            raise e  # Re-raise validation errors as-is
        except Exception as e:
            self.logger.error(f"Failed to add analysis {analysis_id} to batch {batch_id}: {str(e)}")
            raise DatabaseError(f"Add analysis to batch failed: {str(e)}") from e

    def close(self) -> None:
        """Close database connections and clean up resources safely."""
        try:
            if self._Session:
                self._Session.remove()
                self.logger.info("Database sessions closed")
            
            if self._engine:
                self._engine.dispose()
                self.logger.info("Database engine disposed")
                
        except Exception as e:
            self.logger.error(f"Error closing database: {str(e)}")

    def save_analysis_batch(self, analysis_list: List[PydanticAnalysisResult]) -> List[int]:
        """
        Save multiple analysis results in a batch operation for better performance in production.
        
        Args:
            analysis_list: List of analysis results to save (validated)
            
        Returns:
            List of analysis IDs (may be shorter than input if some saves failed)

        Raises:
            DatabaseError: If batch save operation fails completely
            ValueError: If analysis_list is invalid
        """
        if not analysis_list:
            raise ValueError("Analysis list cannot be empty")
            
        if not isinstance(analysis_list, list):
            raise ValueError("analysis_list must be a list")

        analysis_ids = []
        failed_count = 0
        
        try:
            with self.get_session() as session:
                self.logger.info(f"Starting batch save of {len(analysis_list)} analysis results")
                
                # Process each analysis individually for reliability
                for i, analysis_data in enumerate(analysis_list):
                    try:
                        if not analysis_data or not analysis_data.metadata:
                            self.logger.error(f"Invalid analysis data at index {i}")
                            failed_count += 1
                            continue

                        # Validate required fields
                        if not analysis_data.metadata.model or not analysis_data.metadata.model.strip():
                            self.logger.error(f"Missing model in analysis at index {i}")
                            failed_count += 1
                            continue
                            
                        if not analysis_data.metadata.serial or not analysis_data.metadata.serial.strip():
                            self.logger.error(f"Missing serial in analysis at index {i}")
                            failed_count += 1
                            continue

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
                            file_path=str(analysis_data.metadata.file_path) if analysis_data.metadata.file_path else None,
                            file_date=analysis_data.metadata.file_date,
                            file_hash=None,
                            model=analysis_data.metadata.model.strip(),
                            serial=analysis_data.metadata.serial.strip(),
                            system=system_type,
                            has_multi_tracks=analysis_data.metadata.has_multi_tracks,
                            overall_status=overall_status,
                            processing_time=analysis_data.processing_time,
                            output_dir=None,
                            software_version="2.0.0",
                            operator=None,
                            sigma_scaling_factor=None,
                            filter_cutoff_frequency=None
                        )

                        # Add track results
                        if not analysis_data.tracks:
                            self.logger.error(f"No tracks in analysis at index {i}")
                            failed_count += 1
                            continue

                        for track_id, track_data in analysis_data.tracks.items():
                            if not track_id or not track_id.strip():
                                self.logger.error(f"Empty track ID in analysis at index {i}")
                                continue
                                
                            if not track_data.sigma_analysis:
                                self.logger.error(f"Missing sigma analysis for track {track_id} at index {i}")
                                continue

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
                                track_id=track_id.strip(),
                                status=track_status,
                                travel_length=track_data.travel_length,
                                linearity_spec=track_data.linearity_analysis.linearity_spec if track_data.linearity_analysis else None,
                                sigma_gradient=track_data.sigma_analysis.sigma_gradient,
                                sigma_threshold=track_data.sigma_analysis.sigma_threshold,
                                sigma_pass=track_data.sigma_analysis.sigma_pass,
                                unit_length=track_data.unit_properties.unit_length if track_data.unit_properties else None,
                                untrimmed_resistance=track_data.unit_properties.untrimmed_resistance if track_data.unit_properties else None,
                                trimmed_resistance=track_data.unit_properties.trimmed_resistance if track_data.unit_properties else None,
                                resistance_change=track_data.unit_properties.resistance_change if track_data.unit_properties else None,
                                resistance_change_percent=track_data.unit_properties.resistance_change_percent if track_data.unit_properties else None,
                                optimal_offset=track_data.linearity_analysis.optimal_offset if track_data.linearity_analysis else None,
                                final_linearity_error_raw=track_data.linearity_analysis.final_linearity_error_raw if track_data.linearity_analysis else None,
                                final_linearity_error_shifted=track_data.linearity_analysis.final_linearity_error_shifted if track_data.linearity_analysis else None,
                                linearity_pass=track_data.linearity_analysis.linearity_pass if track_data.linearity_analysis else None,
                                linearity_fail_points=track_data.linearity_analysis.linearity_fail_points if track_data.linearity_analysis else None,
                                max_deviation=track_data.linearity_analysis.max_deviation if track_data.linearity_analysis else None,
                                max_deviation_position=track_data.linearity_analysis.max_deviation_position if track_data.linearity_analysis else None,
                                deviation_uniformity=None,
                                trim_improvement_percent=track_data.trim_effectiveness.improvement_percent if track_data.trim_effectiveness else None,
                                untrimmed_rms_error=track_data.trim_effectiveness.untrimmed_rms_error if track_data.trim_effectiveness else None,
                                trimmed_rms_error=track_data.trim_effectiveness.trimmed_rms_error if track_data.trim_effectiveness else None,
                                max_error_reduction_percent=track_data.trim_effectiveness.max_error_reduction_percent if track_data.trim_effectiveness else None,
                                worst_zone=track_data.zone_analysis.worst_zone if track_data.zone_analysis else None,
                                worst_zone_position=track_data.zone_analysis.worst_zone_position[0] if (track_data.zone_analysis and track_data.zone_analysis.worst_zone_position) else None,
                                zone_details=track_data.zone_analysis.zone_results if track_data.zone_analysis else None,
                                failure_probability=track_data.failure_prediction.failure_probability if track_data.failure_prediction else None,
                                risk_category=risk_category,
                                gradient_margin=track_data.sigma_analysis.gradient_margin if hasattr(track_data.sigma_analysis, 'gradient_margin') else None,
                                range_utilization_percent=track_data.dynamic_range.range_utilization_percent if track_data.dynamic_range else None,
                                minimum_margin=track_data.dynamic_range.minimum_margin if track_data.dynamic_range else None,
                                minimum_margin_position=track_data.dynamic_range.minimum_margin_position if track_data.dynamic_range else None,
                                margin_bias=track_data.dynamic_range.margin_bias if track_data.dynamic_range else None,
                                plot_path=str(track_data.plot_path) if track_data.plot_path else None
                            )
                            analysis.tracks.append(track)

                        # Generate production alerts
                        self._generate_production_alerts(analysis, session)

                        # Add to session
                        session.add(analysis)
                        session.flush()  # Get the ID without committing
                        
                        analysis_ids.append(analysis.id)
                        
                        self.logger.debug(f"Prepared analysis for {analysis_data.metadata.filename}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to prepare analysis at index {i}: {e}")
                        failed_count += 1
                        continue
                
                # Single commit for the entire batch
                session.commit()
                
                success_count = len(analysis_ids)
                self.logger.info(f"Batch save completed: {success_count} successful, {failed_count} failed")
                
                if failed_count > 0:
                    self.logger.warning(f"{failed_count} analyses failed to save in batch operation")
                
                return analysis_ids
                
        except Exception as e:
            self.logger.error(f"Batch save operation failed: {str(e)}")
            raise DatabaseError(f"Batch save failed: {str(e)}") from e

    def validate_saved_analysis(self, analysis_id: int) -> bool:
        """
        Validate that an analysis was properly saved to the production database.
        
        Args:
            analysis_id: ID of the analysis to validate (must be positive)
            
        Returns:
            True if analysis is properly saved with all tracks

        Raises:
            DatabaseError: If validation operation fails
            ValueError: If analysis_id is invalid
        """
        if not isinstance(analysis_id, int) or analysis_id <= 0:
            raise ValueError("analysis_id must be a positive integer")

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
                
                # Verify track data integrity
                tracks = session.query(DBTrackResult).filter(
                    DBTrackResult.analysis_id == analysis_id
                ).all()
                
                for track in tracks:
                    if not track.track_id or track.sigma_gradient is None:
                        self.logger.error(f"Track data incomplete for analysis ID {analysis_id}")
                        return False
                
                self.logger.info(f"Analysis ID {analysis_id} validated: {track_count} tracks")
                return True
                
        except Exception as e:
            self.logger.error(f"Validation failed for analysis ID {analysis_id}: {e}")
            raise DatabaseError(f"Analysis validation failed: {str(e)}") from e

    def force_save_analysis(self, analysis_data: PydanticAnalysisResult) -> int:
        """
        Force save analysis without duplicate checking - for critical production saves.
        
        Args:
            analysis_data: Pydantic model containing analysis results (validated)
            
        Returns:
            ID of the saved analysis record

        Raises:
            DatabaseError: If force save fails
            ValueError: If analysis data is invalid
        """
        if not analysis_data:
            raise ValueError("Analysis data is required")
            
        if not analysis_data.metadata:
            raise ValueError("Analysis metadata is required")

        try:
            with self.get_session() as session:
                self.logger.info(f"Force saving analysis for {analysis_data.metadata.filename}")
                
                # Use the same logic as regular save but without duplicate checking
                # Convert Pydantic SystemType to DB SystemType
                system_type = DBSystemType.A if analysis_data.metadata.system == SystemType.SYSTEM_A else DBSystemType.B

                # Convert Pydantic AnalysisStatus to DB StatusType
                status_map = {
                    AnalysisStatus.PASS: DBStatusType.PASS,
                    AnalysisStatus.FAIL: DBStatusType.FAIL,
                    AnalysisStatus.WARNING: DBStatusType.WARNING,
                    Anal
