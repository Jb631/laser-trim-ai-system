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
import time
import threading
from queue import Queue

from sqlalchemy import create_engine, func, and_, or_, desc, inspect, text, event, Integer
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError, DatabaseError as SQLDatabaseError
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

# Define a simple no-op decorator first
def no_op_decorator(*args, **kwargs):
    """No-op decorator for when modules are not available."""
    def decorator(func):
        return func
    return decorator

# Import error handling utilities
try:
    from ..core.error_handlers import (
        ErrorCode, ErrorCategory, ErrorSeverity,
        error_handler, handle_errors, ErrorContext
    )
    HAS_ERROR_HANDLERS = True
except ImportError:
    HAS_ERROR_HANDLERS = False
    handle_errors = no_op_decorator
    error_handler = None

# Import security utilities
try:
    from ..core.security import (
        SecurityValidator, SecurityLevel, ThreatType,
        validate_inputs, get_security_validator
    )
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False
    validate_inputs = no_op_decorator

# Import performance optimizer
try:
    from .performance_optimizer import (
        PerformanceOptimizer, cached_query
    )
    HAS_PERFORMANCE_OPTIMIZER = True
except ImportError:
    HAS_PERFORMANCE_OPTIMIZER = False
    cached_query = no_op_decorator  # Use no-op decorator when optimizer not available

# Import secure logging if available
try:
    from ..core.secure_logging import (
        get_logger, logged_function, LogLevel
    )
    HAS_SECURE_LOGGING = True
except ImportError:
    HAS_SECURE_LOGGING = False


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
    - Automatic retry mechanisms for transient errors
    - Connection health monitoring and recovery
    """
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 0.5  # seconds
    CONNECTION_CHECK_INTERVAL = 60  # seconds
    
    # Connection pool monitoring
    _connection_healthy = True
    _last_health_check = 0
    _health_check_lock = threading.Lock()

    def __init__(
            self,
            database_url_or_config: Optional[Union[str, Any]] = None,
            echo: bool = False,
            pool_size: int = 5,
            max_overflow: int = 10,
            pool_timeout: int = 30,
            logger: Optional[logging.Logger] = None,
            enable_performance_optimization: bool = True,
            cache_size: int = 1000,
            cache_ttl: int = 300
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
            enable_performance_optimization: Enable performance optimization features
            cache_size: Maximum number of cached queries
            cache_ttl: Default cache TTL in seconds

        Raises:
            DatabaseConnectionError: If database connection cannot be established
        """
        # Use secure logger if available
        if HAS_SECURE_LOGGING:
            self.logger = logger or get_logger(__name__)
        else:
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
            
            # CRITICAL: Initialize the database tables if they don't exist
            self.logger.info("Checking/creating database tables...")
            self.init_db(drop_existing=False)
            
            # Initialize performance optimizer if enabled
            self.performance_optimizer = None
            if enable_performance_optimization and HAS_PERFORMANCE_OPTIMIZER:
                try:
                    self.performance_optimizer = PerformanceOptimizer(
                        self._engine,
                        enable_profiling=True,
                        enable_caching=True,
                        cache_size=cache_size,
                        cache_ttl=cache_ttl
                    )
                    self.logger.info("Performance optimization enabled")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize performance optimizer: {e}")

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
            # Use StaticPool for SQLite to avoid connection issues
            engine_kwargs["poolclass"] = StaticPool
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
            
            # Update health status
            with self._health_check_lock:
                self._connection_healthy = True
                self._last_health_check = time.time()
                
        except Exception as e:
            with self._health_check_lock:
                self._connection_healthy = False
            raise DatabaseConnectionError(f"Database connection test failed: {str(e)}") from e
    
    def _is_connection_healthy(self) -> bool:
        """Check if database connection is healthy."""
        with self._health_check_lock:
            # Check if we need to perform a health check
            current_time = time.time()
            if current_time - self._last_health_check > self.CONNECTION_CHECK_INTERVAL:
                # Perform health check
                try:
                    with self._engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    self._connection_healthy = True
                except Exception:
                    self._connection_healthy = False
                self._last_health_check = current_time
            
            return self._connection_healthy
    
    def _reconnect(self) -> None:
        """Attempt to reconnect to the database."""
        self.logger.info("Attempting to reconnect to database...")
        
        try:
            # Dispose of the current engine
            if self._engine:
                self._engine.dispose()
            
            # Reinitialize the engine
            self._initialize_engine(
                self.database_url, 
                self._engine.echo if self._engine else False,
                5, 10, 30  # Use default pool settings
            )
            
            # Test the new connection
            self._test_connection()
            
            self.logger.info("Database reconnection successful")
            
        except Exception as e:
            self.logger.error(f"Failed to reconnect to database: {e}")
            raise DatabaseConnectionError(f"Database reconnection failed: {str(e)}") from e
    
    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic for transient errors."""
        last_error = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (OperationalError, SQLDatabaseError) as e:
                last_error = e
                
                # Check if error is retryable
                error_str = str(e).lower()
                retryable_errors = ['timeout', 'connection', 'locked', 'busy', 'deadlock']
                
                if any(err in error_str for err in retryable_errors):
                    if attempt < self.MAX_RETRIES - 1:
                        delay = self.RETRY_DELAY_BASE * (2 ** attempt)  # Exponential backoff
                        self.logger.warning(
                            f"Retryable database error on attempt {attempt + 1}/{self.MAX_RETRIES}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        
                        # For connection errors, try to reconnect
                        if 'connection' in error_str:
                            try:
                                self._reconnect()
                            except Exception:
                                pass
                        continue
                    else:
                        self.logger.error(f"Max retries ({self.MAX_RETRIES}) exceeded for database operation")
                
                # Non-retryable error or max retries exceeded
                raise
            except Exception as e:
                # Non-database errors are not retried
                raise
        
        # Should not reach here, but just in case
        if last_error:
            raise last_error
    
    def _on_connect(self, dbapi_conn, connection_record):
        """Event handler for new database connections."""
        self.logger.debug("New database connection established")
        with self._health_check_lock:
            self._connection_healthy = True
    
    def _on_checkout(self, dbapi_conn, connection_record, connection_proxy):
        """Event handler for connection checkout from pool."""
        # Could add connection validation here if needed
        pass
    
    def _on_checkin(self, dbapi_conn, connection_record):
        """Event handler for connection checkin to pool."""
        # Could add cleanup here if needed
        pass

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
            self.logger.error("DEBUG: Database session factory is None!")
            raise DatabaseConnectionError("Database not initialized")

        self.logger.debug("DEBUG: Creating new database session...")
        session = self._Session()
        self.logger.debug("DEBUG: Database session created successfully")
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

    @validate_inputs(
        model={'type': 'model_number'},
        serial={'type': 'serial_number'},
        file_date={'type': 'date'}
    )
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
                # Security: Use parameterized queries (SQLAlchemy does this automatically)
                # Model and serial are already sanitized by the decorator
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

    @handle_errors(
        category=ErrorCategory.DATABASE,
        severity=ErrorSeverity.ERROR,
        max_retries=2
    )
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
        
        # Log detailed information about what we're about to save
        self.logger.info(f"Attempting to save analysis: {analysis_data.metadata.filename}")
        self.logger.debug(f"Analysis details: model={analysis_data.metadata.model}, serial={analysis_data.metadata.serial}, tracks={len(analysis_data.tracks) if analysis_data.tracks else 0}")

        # Use retry logic for the entire save operation
        return self._execute_with_retry(self._save_analysis_impl, analysis_data)
    
    def _save_analysis_impl(self, analysis_data: PydanticAnalysisResult) -> int:
        """Internal implementation of save_analysis with transaction support."""
        # Log save operation for debugging
        self.logger.debug(f"DEBUG: _save_analysis_impl called for {analysis_data.metadata.filename}")
        
        if HAS_SECURE_LOGGING:
            self.logger.debug("Saving analysis to database", context={
                'model': analysis_data.metadata.model,
                'serial': analysis_data.metadata.serial,
                'file_date': analysis_data.metadata.file_date.isoformat() if analysis_data.metadata.file_date else None,
                'num_tracks': len(analysis_data.tracks),
                'overall_status': analysis_data.overall_status.value
            })
        else:
            self.logger.debug(f"Saving analysis: model={analysis_data.metadata.model}, serial={analysis_data.metadata.serial}, tracks={len(analysis_data.tracks)}")
        
        try:
            self.logger.debug("DEBUG: Getting database session...")
            with self.get_session() as session:
                self.logger.debug("DEBUG: Got database session successfully")
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
                        linearity_spec=getattr(track_data.linearity_analysis, 'linearity_spec', None) if track_data.linearity_analysis else None,
                        sigma_gradient=getattr(track_data.sigma_analysis, 'sigma_gradient', None) if track_data.sigma_analysis else None,
                        sigma_threshold=getattr(track_data.sigma_analysis, 'sigma_threshold', None) if track_data.sigma_analysis else None,
                        sigma_pass=getattr(track_data.sigma_analysis, 'sigma_pass', None) if track_data.sigma_analysis else None,
                        unit_length=getattr(track_data.unit_properties, 'unit_length', None) if track_data.unit_properties else None,
                        untrimmed_resistance=getattr(track_data.unit_properties, 'untrimmed_resistance', None) if track_data.unit_properties else None,
                        trimmed_resistance=getattr(track_data.unit_properties, 'trimmed_resistance', None) if track_data.unit_properties else None,
                        resistance_change=getattr(track_data.unit_properties, 'resistance_change', None) if track_data.unit_properties else None,
                        resistance_change_percent=getattr(track_data.unit_properties, 'resistance_change_percent', None) if track_data.unit_properties else None,
                        optimal_offset=getattr(track_data.linearity_analysis, 'optimal_offset', None) if track_data.linearity_analysis else None,
                        final_linearity_error_raw=getattr(track_data.linearity_analysis, 'final_linearity_error_raw', None) if track_data.linearity_analysis else None,
                        final_linearity_error_shifted=getattr(track_data.linearity_analysis, 'final_linearity_error_shifted', None) if track_data.linearity_analysis else None,
                        linearity_pass=getattr(track_data.linearity_analysis, 'linearity_pass', None) if track_data.linearity_analysis else None,
                        linearity_fail_points=getattr(track_data.linearity_analysis, 'linearity_fail_points', None) if track_data.linearity_analysis else None,
                        max_deviation=getattr(track_data.linearity_analysis, 'max_deviation', None) if track_data.linearity_analysis else None,
                        max_deviation_position=getattr(track_data.linearity_analysis, 'max_deviation_position', None) if track_data.linearity_analysis else None,
                        deviation_uniformity=None,  # Calculate if needed
                        trim_improvement_percent=getattr(track_data.trim_effectiveness, 'improvement_percent', None) if track_data.trim_effectiveness else None,
                        untrimmed_rms_error=getattr(track_data.trim_effectiveness, 'untrimmed_rms_error', None) if track_data.trim_effectiveness else None,
                        trimmed_rms_error=getattr(track_data.trim_effectiveness, 'trimmed_rms_error', None) if track_data.trim_effectiveness else None,
                        max_error_reduction_percent=getattr(track_data.trim_effectiveness, 'max_error_reduction_percent', None) if track_data.trim_effectiveness else None,
                        worst_zone=getattr(track_data.zone_analysis, 'worst_zone', None) if track_data.zone_analysis else None,
                        worst_zone_position=track_data.zone_analysis.worst_zone_position[0] if (track_data.zone_analysis and hasattr(track_data.zone_analysis, 'worst_zone_position') and track_data.zone_analysis.worst_zone_position) else None,
                        zone_details=getattr(track_data.zone_analysis, 'zone_results', None) if track_data.zone_analysis else None,
                        failure_probability=getattr(track_data.failure_prediction, 'failure_probability', None) if track_data.failure_prediction else None,
                        risk_category=risk_category,
                        gradient_margin=getattr(track_data.sigma_analysis, 'gradient_margin', None) if track_data.sigma_analysis else None,
                        range_utilization_percent=getattr(track_data.dynamic_range, 'range_utilization_percent', None) if track_data.dynamic_range else None,
                        minimum_margin=getattr(track_data.dynamic_range, 'minimum_margin', None) if track_data.dynamic_range else None,
                        minimum_margin_position=getattr(track_data.dynamic_range, 'minimum_margin_position', None) if track_data.dynamic_range else None,
                        margin_bias=getattr(track_data.dynamic_range, 'margin_bias', None) if track_data.dynamic_range else None,
                        plot_path=str(track_data.plot_path) if track_data.plot_path else None
                    )
                    analysis.tracks.append(track)

                # Generate alerts based on real analysis data
                self._generate_alerts(analysis, session)

                session.add(analysis)
                session.commit()

                self.logger.info(
                    f"Successfully saved analysis for {analysis_data.metadata.filename} "
                    f"with {len(analysis.tracks)} tracks (ID: {analysis.id})"
                )
                
                # Log save completion for debugging
                if HAS_SECURE_LOGGING:
                    self.logger.debug("Analysis save completed", context={
                        'analysis_id': analysis.id,
                        'model': analysis_data.metadata.model,
                        'serial': analysis_data.metadata.serial,
                        'tracks_saved': len(analysis.tracks),
                        'ml_predictions_saved': len(analysis.ml_predictions) if hasattr(analysis, 'ml_predictions') else 0,
                        'alerts_generated': len(analysis.qa_alerts) if hasattr(analysis, 'qa_alerts') else 0
                    })
                
                return analysis.id

        except IntegrityError as e:
            # Handle specific integrity errors
            error_str = str(e).lower()
            
            if "unique constraint" in error_str or "duplicate" in error_str:
                # Extract which field caused the duplicate
                duplicate_info = "duplicate entry"
                if "filename" in error_str:
                    duplicate_info = f"file '{analysis_data.metadata.filename}' already analyzed"
                elif "serial" in error_str:
                    duplicate_info = f"serial '{analysis_data.metadata.serial}' at this timestamp"
                
                error_handler.handle_error(
                    error=e,
                    category=ErrorCategory.DATABASE,
                    severity=ErrorSeverity.WARNING,
                    code=ErrorCode.DB_DUPLICATE_ENTRY,
                    user_message=f"Analysis already exists: {duplicate_info}",
                    recovery_suggestions=[
                        "Check if this file was already processed",
                        "Use 'Force Save' to overwrite existing data",
                        "Verify the serial number is correct"
                    ],
                    additional_data={
                        'filename': analysis_data.metadata.filename,
                        'model': analysis_data.metadata.model,
                        'serial': analysis_data.metadata.serial
                    }
                )
                raise DatabaseIntegrityError(f"Duplicate analysis: {duplicate_info}")
            else:
                raise DatabaseIntegrityError(f"Data integrity violation: {str(e)}")
                
        except ValueError as e:
            raise e  # Re-raise validation errors as-is
            
        except OperationalError as e:
            error_handler.handle_error(
                error=e,
                category=ErrorCategory.DATABASE,
                severity=ErrorSeverity.ERROR,
                code=ErrorCode.DB_QUERY_FAILED,
                user_message="Failed to save analysis to database. Please try again.",
                technical_details=str(e),
                recovery_suggestions=[
                    "Check database connection",
                    "Verify database is not full",
                    "Try again in a few moments"
                ]
            )
            raise DatabaseError(f"Database operation failed: {str(e)}") from e
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis: {str(e)}")
            
            error_handler.handle_error(
                error=e,
                category=ErrorCategory.DATABASE,
                severity=ErrorSeverity.ERROR,
                code=ErrorCode.UNKNOWN_ERROR,
                user_message="An unexpected error occurred while saving analysis.",
                technical_details=str(e),
                additional_data={
                    'filename': analysis_data.metadata.filename if analysis_data.metadata else 'unknown',
                    'error_type': type(e).__name__
                }
            )
            
            raise DatabaseError(f"Analysis save failed: {str(e)}") from e

    @validate_inputs(
        model={'type': 'model_number', 'required': False},
        serial={'type': 'serial_number', 'required': False},
        days_back={'type': 'number', 'min': 1, 'max': 3650, 'required': False},
        start_date={'type': 'date', 'required': False},
        end_date={'type': 'date', 'required': False},
        status={'type': 'string', 'max_length': 50, 'required': False},
        risk_category={'type': 'string', 'max_length': 50, 'required': False},
        limit={'type': 'number', 'min': 1, 'max': 10000, 'required': False},
        offset={'type': 'number', 'min': 0, 'max': 1000000, 'required': False}
    )
    @cached_query('historical_data', ttl=300)
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

                # Apply filters with SQL injection protection
                # Note: SQLAlchemy parameterizes these automatically
                if model:
                    # Sanitize wildcards to prevent DOS
                    safe_model = model.strip()
                    if safe_model.count('%') > 2:
                        self.logger.warning("Too many wildcards in model filter")
                        safe_model = safe_model.replace('%', '', safe_model.count('%') - 2)
                    query = query.filter(DBAnalysisResult.model.like(safe_model))

                if serial:
                    # Sanitize wildcards to prevent DOS
                    safe_serial = serial.strip()
                    if safe_serial.count('%') > 2:
                        self.logger.warning("Too many wildcards in serial filter")
                        safe_serial = safe_serial.replace('%', '', safe_serial.count('%') - 2)
                    query = query.filter(DBAnalysisResult.serial.like(safe_serial))

                # Date filtering
                if days_back:
                    # Use UTC to match database timestamps
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

    @validate_inputs(
        model={'type': 'model_number'}
    )
    @cached_query('model_stats', ttl=600)
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

    def _generate_alerts(self, analysis: DBAnalysisResult, session: Session) -> None:
        """Generate alerts for analysis based on thresholds and risk."""
        try:
            # Generate alerts for high-risk tracks
            for track in analysis.tracks:
                if track.risk_category == DBRiskCategory.HIGH:
                    alert = DBQAAlert(
                        analysis_id=analysis.id,
                        alert_type=DBAlertType.HIGH_RISK,
                        severity="High",  # Must be 'Critical', 'High', 'Medium', or 'Low'
                        message=f"High risk track {track.track_id}: failure probability {track.failure_probability:.2%}",
                        track_id=track.track_id,
                        threshold_value=0.7,  # High risk threshold
                        actual_value=track.failure_probability,
                        recommendation="Immediate inspection recommended"
                    )
                    session.add(alert)
                
                # Sigma gradient alerts
                if track.sigma_gradient and track.sigma_threshold and track.sigma_gradient > track.sigma_threshold:
                    alert = DBQAAlert(
                        analysis_id=analysis.id,
                        alert_type=DBAlertType.SIGMA_FAIL,
                        severity="Medium",  # Must be 'Critical', 'High', 'Medium', or 'Low'
                        message=f"Sigma gradient exceeds threshold on track {track.track_id}",
                        track_id=track.track_id,
                        threshold_value=track.sigma_threshold,
                        actual_value=track.sigma_gradient,
                        recommendation="Review trimming parameters"
                    )
                    session.add(alert)
            
            # Overall status alert
            if analysis.overall_status in [DBStatusType.FAIL, DBStatusType.ERROR]:
                alert = DBQAAlert(
                    analysis_id=analysis.id,
                    alert_type=DBAlertType.PROCESS_ERROR if analysis.overall_status == DBStatusType.ERROR else DBAlertType.HIGH_RISK,
                    severity="Critical" if analysis.overall_status == DBStatusType.ERROR else "High",  # Must be 'Critical', 'High', 'Medium', or 'Low'
                    message=f"Analysis failed with status: {analysis.overall_status.value}",
                    recommendation="Review all test parameters and retry"
                )
                session.add(alert)
                
        except Exception as e:
            self.logger.error(f"Failed to generate alerts: {e}")
            # Don't fail the entire save operation for alert generation
    
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
                            "count": count,
                            "percentage": 0,  # Will calculate after total
                            "avg_failure_probability": float(avg_prob or 0)
                        }
                        summary["total"] += count

                # Calculate percentages
                if summary["total"] > 0:
                    for category in summary["categories"].values():
                        category["percentage"] = (category["count"] / summary["total"]) * 100

                return summary

        except SQLAlchemyError as e:
            self.logger.error(f"Database error in get_risk_summary: {e}")
            raise DatabaseError(f"Failed to get risk summary: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in get_risk_summary: {e}")
            raise DatabaseError(f"Unexpected error getting risk summary: {e}")
    
    def save_analysis_batch(self, analyses: List[PydanticAnalysisResult]) -> List[int]:
        """
        Save multiple analyses in a batch for performance.
        
        Args:
            analyses: List of analysis results to save
            
        Returns:
            List of saved analysis IDs
            
        Raises:
            DatabaseError: If batch save fails
            ValueError: If input is invalid
        """
        if not analyses:
            raise ValueError("No analyses provided for batch save")
            
        if len(analyses) > 1000:
            raise ValueError("Batch size too large (max 1000)")
        
        self.logger.info(f"Starting batch save for {len(analyses)} analyses")
        
        # Validate all analyses first
        if HAS_SECURITY:
            validator = get_security_validator()
            for i, analysis in enumerate(analyses):
                if not analysis:
                    raise ValueError(f"Analysis at index {i} is None")
                if not analysis.metadata:
                    raise ValueError(f"Analysis at index {i} missing metadata")
                    
                # Validate model and serial
                model_result = validator.validate_input(
                    analysis.metadata.model, 
                    'model_number'
                )
                if not model_result.is_safe:
                    raise ValueError(f"Invalid model at index {i}: {model_result.validation_errors}")
                    
                serial_result = validator.validate_input(
                    analysis.metadata.serial,
                    'serial_number'
                )
                if not serial_result.is_safe:
                    raise ValueError(f"Invalid serial at index {i}: {serial_result.validation_errors}")
        else:
            # Basic validation without security module
            for i, analysis in enumerate(analyses):
                if not analysis:
                    raise ValueError(f"Analysis at index {i} is None")
                if not analysis.metadata:
                    raise ValueError(f"Analysis at index {i} missing metadata")
                if not analysis.metadata.model:
                    raise ValueError(f"Analysis at index {i} missing model")
                if not analysis.metadata.serial:
                    raise ValueError(f"Analysis at index {i} missing serial")
        
        saved_ids = []
        failed_saves = []
        
        # Use a single transaction for all saves with bulk operations
        try:
            with self.get_session() as session:
                # Prepare bulk data
                analyses_to_insert = []
                tracks_to_insert = []
                analysis_objects = []
                
                for i, analysis in enumerate(analyses):
                    try:
                        # Log each analysis being processed
                        self.logger.debug(f"Processing analysis {i+1}/{len(analyses)}: {analysis.metadata.filename}")
                        
                        # Check for duplicates
                        existing_id = self.check_duplicate_analysis(
                            analysis.metadata.model,
                            analysis.metadata.serial,
                            analysis.metadata.file_date
                        )
                        
                        if existing_id:
                            self.logger.info(f"Skipping duplicate: {analysis.metadata.model}-{analysis.metadata.serial}")
                            saved_ids.append(existing_id)
                            continue
                        
                        # Create database record but don't add to session yet
                        db_analysis = self._prepare_analysis_record(analysis)
                        analyses_to_insert.append(db_analysis)
                        analysis_objects.append((i, analysis))
                        
                    except Exception as e:
                        self.logger.error(f"Failed to prepare analysis {i+1} ({analysis.metadata.filename}): {str(e)}")
                        failed_saves.append((i, analysis.metadata.filename, str(e)))
                        saved_ids.append(None)  # Placeholder for failed save
                
                # Bulk insert all analyses at once
                if analyses_to_insert:
                    session.bulk_save_objects(analyses_to_insert, return_defaults=True)
                    session.flush()  # Ensure IDs are generated
                    
                    # Now prepare tracks with analysis IDs
                    for db_analysis, (orig_idx, orig_analysis) in zip(analyses_to_insert, analysis_objects):
                        saved_ids.insert(orig_idx, db_analysis.id)
                        
                        # Add tracks for this analysis
                        for track_id, track_data in orig_analysis.tracks.items():
                            db_track = self._prepare_track_record(track_data, db_analysis.id, track_id)
                            tracks_to_insert.append(db_track)
                    
                    # Bulk insert all tracks
                    if tracks_to_insert:
                        session.bulk_save_objects(tracks_to_insert)
                
                # Commit all at once
                session.commit()
                
            # Log summary
            successful_saves = len([id for id in saved_ids if id is not None])
            self.logger.info(f"Batch save completed: {successful_saves}/{len(analyses)} successful")
            
            if failed_saves:
                self.logger.error(f"Failed saves: {failed_saves}")
                
            return saved_ids
            
        except Exception as e:
            self.logger.error(f"Batch save failed: {e}")
            raise DatabaseError(f"Batch save failed: {str(e)}") from e
    
    def _prepare_analysis_record(
        self, 
        analysis_data: PydanticAnalysisResult
    ) -> DBAnalysisResult:
        """Prepare analysis record without adding to session (for bulk operations)."""
        self.logger.debug(f"Preparing analysis record for {analysis_data.metadata.filename}")
        
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
        
        return analysis
    
    def _prepare_track_record(
        self,
        track_data: TrackData,
        analysis_id: int,
        track_id: str
    ) -> DBTrackResult:
        """Prepare track record for bulk insert."""
        # Convert track status
        status_map = {
            AnalysisStatus.PASS: DBStatusType.PASS,
            AnalysisStatus.FAIL: DBStatusType.FAIL,
            AnalysisStatus.WARNING: DBStatusType.WARNING,
            AnalysisStatus.ERROR: DBStatusType.ERROR,
            AnalysisStatus.PENDING: DBStatusType.PROCESSING_FAILED
        }
        track_status = status_map.get(track_data.status, DBStatusType.ERROR)
        
        # Convert risk category
        risk_category = None
        if track_data.failure_prediction and track_data.failure_prediction.risk_category:
            risk_map = {
                RiskCategory.HIGH: "high",
                RiskCategory.MEDIUM: "medium", 
                RiskCategory.LOW: "low"
            }
            risk_category = risk_map.get(track_data.failure_prediction.risk_category)
        
        # Create track record
        track = DBTrackResult(
            analysis_id=analysis_id,
            track_id=track_id,
            status=track_status,
            sigma_gradient=track_data.sigma_analysis.sigma_gradient if track_data.sigma_analysis else None,
            sigma_threshold=track_data.sigma_analysis.sigma_threshold if track_data.sigma_analysis else None,
            sigma_pass=track_data.sigma_analysis.sigma_pass if track_data.sigma_analysis else None,
            linearity_spec=track_data.linearity_analysis.linearity_spec if track_data.linearity_analysis else None,
            linearity_error=track_data.linearity_analysis.final_linearity_error if track_data.linearity_analysis else None,
            linearity_error_shifted=track_data.linearity_analysis.final_linearity_error_shifted if track_data.linearity_analysis else None,
            optimal_offset=track_data.linearity_analysis.optimal_offset if track_data.linearity_analysis else None,
            linearity_pass=track_data.linearity_analysis.linearity_pass if track_data.linearity_analysis else None,
            trimmed_resistance=track_data.unit_properties.trimmed_resistance if track_data.unit_properties else None,
            untrimmed_resistance=track_data.unit_properties.untrimmed_resistance if track_data.unit_properties else None,
            trim_percentage=track_data.resistance_analysis.resistance_change_percent if track_data.resistance_analysis else None,
            abs_value_change=track_data.resistance_analysis.resistance_change if track_data.resistance_analysis else None,
            relative_change=track_data.resistance_analysis.resistance_change_percent if track_data.resistance_analysis else None,
            max_deviation=track_data.linearity_analysis.max_deviation if track_data.linearity_analysis else None,
            max_deviation_position=track_data.linearity_analysis.max_deviation_position if track_data.linearity_analysis else None,
            deviation_uniformity=None,  # Calculate if needed
            trim_improvement_percent=track_data.trim_effectiveness.improvement_percent if track_data.trim_effectiveness else None,
            untrimmed_rms_error=track_data.trim_effectiveness.untrimmed_rms_error if track_data.trim_effectiveness else None,
            trimmed_rms_error=track_data.trim_effectiveness.trimmed_rms_error if track_data.trim_effectiveness else None,
            max_error_reduction_percent=track_data.trim_effectiveness.max_error_reduction_percent if track_data.trim_effectiveness else None,
            worst_zone=track_data.zone_analysis.worst_zone if track_data.zone_analysis else None,
            worst_zone_position=track_data.zone_analysis.worst_zone_position[0] if (track_data.zone_analysis and track_data.zone_analysis.worst_zone_position) else None,
            zone_details=track_data.zone_analysis.zone_results if (track_data.zone_analysis and track_data.zone_analysis.zone_results) else None,
            failure_probability=track_data.failure_prediction.failure_probability if track_data.failure_prediction else None,
            risk_category=risk_category,
            gradient_margin=track_data.sigma_analysis.gradient_margin if hasattr(track_data.sigma_analysis, 'gradient_margin') else None,
            range_utilization_percent=track_data.dynamic_range.range_utilization_percent if track_data.dynamic_range else None,
            minimum_margin=track_data.dynamic_range.minimum_margin if track_data.dynamic_range else None,
            minimum_margin_position=track_data.dynamic_range.minimum_margin_position if track_data.dynamic_range else None,
            margin_bias=track_data.dynamic_range.margin_bias if track_data.dynamic_range else None,
            plot_path=str(track_data.plot_path) if track_data.plot_path else None
        )
        
        return track
    
    def _create_analysis_record(
        self, 
        analysis_data: PydanticAnalysisResult, 
        session: Session
    ) -> int:
        """Create analysis record in database (extracted for reuse)."""
        self.logger.debug(f"Creating analysis record for {analysis_data.metadata.filename}")
        # This is the core logic from _save_analysis_impl
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
                zone_details=track_data.zone_analysis.zone_results if (track_data.zone_analysis and track_data.zone_analysis.zone_results) else None,
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

        # Generate alerts
        self._generate_alerts(analysis, session)
        
        # Add to session
        session.add(analysis)
        
        try:
            session.flush()  # Get the ID without committing
            self.logger.debug(f"Successfully flushed analysis record, ID: {analysis.id}")
        except Exception as e:
            self.logger.error(f"Failed to flush analysis record: {str(e)}")
            self.logger.error(f"Analysis data: model={analysis_data.metadata.model}, serial={analysis_data.metadata.serial}, tracks={len(analysis_data.tracks)}")
            raise
        
        return analysis.id
    
    def validate_saved_analysis(self, analysis_id: int) -> bool:
        """
        Validate that an analysis was saved correctly.
        
        Args:
            analysis_id: ID of the analysis to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with self.get_session() as session:
                analysis = session.query(DBAnalysisResult).filter(
                    DBAnalysisResult.id == analysis_id
                ).first()
                
                if not analysis:
                    return False
                    
                # Check that it has tracks
                if not analysis.tracks:
                    self.logger.warning(f"Analysis {analysis_id} has no tracks")
                    return False
                    
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to validate analysis {analysis_id}: {e}")
            return False
    
    @handle_errors(
        category=ErrorCategory.DATABASE,
        severity=ErrorSeverity.WARNING
    )
    def force_save_analysis(self, analysis_data: PydanticAnalysisResult) -> int:
        """
        Force save an analysis, overwriting any existing data.
        
        Args:
            analysis_data: Analysis data to save
            
        Returns:
            ID of the saved analysis
            
        Raises:
            DatabaseError: If save fails
        """
        try:
            with self.get_session() as session:
                # Delete existing analysis if present
                existing = session.query(DBAnalysisResult).filter(
                    and_(
                        DBAnalysisResult.model == analysis_data.metadata.model,
                        DBAnalysisResult.serial == analysis_data.metadata.serial,
                        DBAnalysisResult.file_date == analysis_data.metadata.file_date
                    )
                ).first()
                
                if existing:
                    self.logger.warning(
                        f"Force overwriting analysis {existing.id} for "
                        f"{analysis_data.metadata.model}-{analysis_data.metadata.serial}"
                    )
                    session.delete(existing)
                    session.flush()
                
                # Create new analysis
                analysis_id = self._create_analysis_record(analysis_data, session)
                session.commit()
                
                return analysis_id
                
        except Exception as e:
            self.logger.error(f"Force save failed: {e}")
            raise DatabaseError(f"Force save failed: {str(e)}") from e
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get database performance report.
        
        Returns:
            Performance metrics and optimization suggestions
        """
        if not self.performance_optimizer:
            return {
                'enabled': False,
                'message': 'Performance optimization not enabled'
            }
            
        return self.performance_optimizer.get_performance_report()
    
    def optimize_performance(self) -> Dict[str, Any]:
        """
        Run automatic performance optimizations.
        
        Returns:
            Optimization results
        """
        if not self.performance_optimizer:
            return {
                'success': False,
                'message': 'Performance optimization not enabled'
            }
            
        return self.performance_optimizer.optimize()
    
    def create_suggested_indexes(self) -> List[Dict[str, Any]]:
        """
        Create indexes based on query pattern analysis.
        
        Returns:
            List of created indexes
        """
        if not self.performance_optimizer:
            return []
            
        # Get suggestions
        suggestions = self.performance_optimizer.index_optimizer.analyze_query_patterns(
            self.performance_optimizer._profiles
        )
        
        created = []
        for suggestion in suggestions[:5]:  # Limit to 5 at a time
            try:
                success = self.performance_optimizer.index_optimizer.create_index(
                    suggestion['table'],
                    suggestion['columns'],
                    suggestion['name']
                )
                if success:
                    created.append(suggestion)
                    self.logger.info(f"Created index: {suggestion['name']}")
            except Exception as e:
                self.logger.error(f"Failed to create index {suggestion['name']}: {e}")
                
        return created
    
    def clear_query_cache(self, pattern: Optional[str] = None):
        """
        Clear query result cache.
        
        Args:
            pattern: Optional pattern to match for selective clearing
        """
        if not self.performance_optimizer:
            return
            
        if pattern:
            self.performance_optimizer.cache.invalidate_pattern(pattern)
            self.logger.info(f"Cleared cache entries matching pattern: {pattern}")
        else:
            self.performance_optimizer.cache.clear()
            self.logger.info("Cleared entire query cache")
    
    def get_slow_queries(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Get queries slower than threshold.
        
        Args:
            threshold: Time threshold in seconds
            
        Returns:
            List of slow queries with statistics
        """
        if not self.performance_optimizer:
            return []
            
        return self.performance_optimizer.get_slow_queries(threshold)
    
    def batch_save_analyses(self, analyses: List[PydanticAnalysisResult], 
                          batch_size: int = 100) -> List[int]:
        """
        Save multiple analyses using batch optimization.
        
        Args:
            analyses: List of analyses to save
            batch_size: Batch size for processing
            
        Returns:
            List of saved analysis IDs
        """
        if not analyses:
            return []
            
        # Use performance optimizer if available
        if self.performance_optimizer:
            # Process in optimized batches
            saved_ids = []
            
            for i in range(0, len(analyses), batch_size):
                batch = analyses[i:i + batch_size]
                batch_ids = self.save_analysis_batch(batch)
                saved_ids.extend(batch_ids)
                
                # Clear cache for these models to ensure fresh data
                for analysis in batch:
                    if analysis.metadata:
                        self.clear_query_cache(f"model_stats:{analysis.metadata.model}")
                        
            return saved_ids
        else:
            # Fall back to regular batch save
            return self.save_analysis_batch(analyses)
    
    def prepare_common_queries(self):
        """Prepare commonly used queries for better performance."""
        if not self.performance_optimizer:
            return
            
        prepared = self.performance_optimizer.prepared_statements
        
        # Prepare common queries
        queries = {
            'get_by_model': '''
                SELECT * FROM analysis_results 
                WHERE model = :model 
                ORDER BY timestamp DESC
            ''',
            'get_recent_analyses': '''
                SELECT * FROM analysis_results 
                WHERE timestamp >= :cutoff_date 
                ORDER BY timestamp DESC 
                LIMIT :limit
            ''',
            'get_high_risk_tracks': '''
                SELECT t.*, a.model, a.serial 
                FROM track_results t 
                JOIN analysis_results a ON t.analysis_id = a.id 
                WHERE t.risk_category = :risk_category 
                ORDER BY t.failure_probability DESC
            ''',
            'count_by_status': '''
                SELECT overall_status, COUNT(*) as count 
                FROM analysis_results 
                WHERE timestamp >= :start_date 
                GROUP BY overall_status
            '''
        }
        
        for name, query in queries.items():
            try:
                prepared.prepare(name, query)
                self.logger.debug(f"Prepared query: {name}")
            except Exception as e:
                self.logger.error(f"Failed to prepare query {name}: {e}")
    
    def save_analysis_result(self, analysis_data: PydanticAnalysisResult) -> int:
        """
        Save analysis result without decorators for debugging.
        
        This method is a direct save without error handling decorators to help debug issues.
        
        Args:
            analysis_data: Analysis data to save
            
        Returns:
            ID of saved analysis
            
        Raises:
            Any exception that occurs
        """
        self.logger.info(f"DEBUG: save_analysis_result called for {analysis_data.metadata.filename if analysis_data is not None and analysis_data.metadata else 'unknown'}")
        
        # First check if tables exist
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            self.logger.debug(f"DEBUG: Available tables: {tables}")
            
            if 'analysis_results' not in tables:
                self.logger.error("DEBUG: analysis_results table does not exist! Creating tables...")
                self.init_db(drop_existing=False)
                tables = inspector.get_table_names()
                self.logger.debug(f"DEBUG: Tables after init: {tables}")
        except Exception as e:
            self.logger.error(f"DEBUG: Failed to check/create tables: {e}")
        
        try:
            # Validate input
            if not analysis_data:
                raise ValueError("Analysis data is None")
            if not analysis_data.metadata:
                raise ValueError("Analysis metadata is None")
            if not analysis_data.metadata.model:
                raise ValueError("Model is None")
            if not analysis_data.metadata.serial:
                raise ValueError("Serial is None")
            
            self.logger.debug(f"Validation passed, calling _save_analysis_impl")
            
            # Direct call to implementation
            result = self._save_analysis_impl(analysis_data)
            
            self.logger.info(f"DEBUG: Successfully saved analysis with ID {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"DEBUG: save_analysis_result failed with {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"DEBUG: Traceback:\n{traceback.format_exc()}")
            raise