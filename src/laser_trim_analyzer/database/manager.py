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
                    if risk_