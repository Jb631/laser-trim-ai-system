# src/laser_trim_analyzer/core/config.py
"""
Configuration management using Pydantic Settings.

Supports loading from environment variables, YAML files, and defaults.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
import logging
import tempfile

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    model_config = SettingsConfigDict(
        # Disable environment variable loading for this nested config
        # to prevent PATH corruption issues
        env_prefix="",
        env_nested_delimiter=""
    )
    
    enabled: bool = Field(default=True, description="Enable database storage")
    mode: str = Field(default="local", description="Database mode: local or shared")
    path: Path = Field(
        default=Path.home() / ".laser_trim_analyzer" / "analysis.db",
        description="Database file path"
    )
    shared_path: Optional[str] = Field(default=None, description="Network path for shared database")
    echo: bool = Field(default=False, description="Echo SQL statements")
    pool_size: int = Field(default=5, ge=1, description="Connection pool size")
    sqlite_timeout: int = Field(default=30, description="SQLite timeout for locked database")
    enable_wal_mode: bool = Field(default=True, description="Enable WAL mode for better concurrency")

    @field_validator('path')
    @classmethod
    def ensure_db_directory(cls, v: Path) -> Path:
        """
        Ensure database directory exists with enhanced error handling.
        
        FIXED: Better handling of environment variable corruption and path issues.
        """
        try:
            # Convert to Path object if it's a string
            if isinstance(v, str):
                # Check if this looks like a corrupted PATH environment variable
                if ';' in v and ('Program Files' in v or 'Windows' in v):
                    logger.warning(f"Detected corrupted PATH-like value in database path: {v[:100]}...")
                    # Use default path instead
                    v = Path.home() / ".laser_trim_analyzer" / "analyzer_v2.db"
                    logger.info(f"Using default database path: {v}")
                else:
                    # Expand environment variables first
                    expanded = os.path.expandvars(v)
                    v = Path(expanded)
            
            # Don't make relative paths absolute if they contain drive letters (Windows)
            # This avoids the issue of prepending CWD to an already absolute Windows path
            if not v.is_absolute() and not (len(str(v)) > 1 and str(v)[1] == ':'):
                v = Path.cwd() / v
            
            # Create parent directories with proper error handling
            try:
                v.parent.mkdir(parents=True, exist_ok=True)
                
                # Test write permissions
                test_file = v.parent / ".write_test"
                try:
                    test_file.touch()
                    test_file.unlink()
                except (PermissionError, OSError) as e:
                    logger.warning(f"No write permission to {v.parent}: {e}")
                    # Fall back to temp directory
                    temp_dir = Path(tempfile.gettempdir()) / "laser_trim_analyzer"
                    temp_dir.mkdir(exist_ok=True)
                    v = temp_dir / "analyzer_v2.db"
                    logger.info(f"Using temporary database path: {v}")
                    
            except (PermissionError, OSError) as e:
                logger.error(f"Cannot create database directory {v.parent}: {e}")
                # Use system temp directory as last resort
                temp_dir = Path(tempfile.gettempdir()) / "laser_trim_analyzer"
                temp_dir.mkdir(exist_ok=True)
                v = temp_dir / "analyzer_v2.db"
                logger.warning(f"Using system temp directory for database: {v}")
            
            return v
            
        except Exception as e:
            logger.error(f"Critical error in database path validation: {e}")
            # Absolute fallback to a known-good location
            fallback_path = Path.home() / ".laser_trim_analyzer" / "analyzer_v2.db"
            try:
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using fallback database path: {fallback_path}")
                return fallback_path
            except Exception as fallback_error:
                logger.critical(f"Even fallback path failed: {fallback_error}")
                # Last resort: use current directory
                emergency_path = Path.cwd() / "analyzer_v2.db"
                logger.critical(f"Emergency database path: {emergency_path}")
                return emergency_path


class ProcessingConfig(BaseSettings):
    """Processing configuration."""
    max_workers: int = Field(default=4, ge=1, le=16, description="Max parallel workers")
    generate_plots: bool = Field(default=True, description="Generate analysis plots")
    plot_dpi: int = Field(default=150, ge=72, le=300, description="Plot resolution")
    cache_enabled: bool = Field(default=True, description="Enable result caching")
    cache_ttl: int = Field(default=3600, ge=0, description="Cache TTL in seconds")

    # File processing
    file_extensions: List[str] = Field(
        default=[".xlsx", ".xls"],
        description="Supported file extensions"
    )
    skip_patterns: List[str] = Field(
        default=["~$*", ".*", "_backup_*"],
        description="File patterns to skip"
    )
    
    # Validation parameters (optimized for large-scale processing)
    max_file_size_mb: float = Field(
        default=50.0, 
        ge=0.1, 
        le=1000.0, 
        description="Maximum file size in MB"
    )
    max_batch_size: int = Field(
        default=1000,  # Increased from 100 to 1000
        ge=1,
        le=10000,      # Increased limit for very large batches
        description="Maximum number of files in a batch"
    )
    validation_level: str = Field(
        default="standard",
        description="Validation level: relaxed, standard, or strict"
    )
    
    # Performance tuning (optimized for large-scale)
    chunk_size: int = Field(
        default=2000,  # Increased from 1000
        ge=100,
        le=10000,
        description="Data processing chunk size"
    )
    memory_limit_mb: float = Field(
        default=2048.0,  # Increased from 512MB to 2GB
        ge=128.0,
        le=16384.0,      # Allow up to 16GB for very large operations
        description="Memory limit for processing in MB"
    )
    
    # System performance protection
    cpu_throttle_enabled: bool = Field(
        default=True,
        description="Enable CPU throttling to prevent system freezing"
    )
    memory_throttle_threshold: float = Field(
        default=1500.0,  # 1.5GB
        ge=512.0,
        le=8192.0,
        description="Memory usage threshold for throttling in MB"
    )
    ui_update_throttle_ms: int = Field(
        default=250,  # Update UI at most every 250ms
        ge=50,
        le=1000,
        description="Minimum interval between UI updates in milliseconds"
    )
    
    # Resource cleanup settings
    garbage_collection_interval: int = Field(
        default=50,  # Force GC every 50 files
        ge=10,
        le=500,
        description="Force garbage collection every N files processed"
    )
    matplotlib_cleanup_interval: int = Field(
        default=25,  # Close matplotlib figures every 25 files
        ge=5,
        le=100,
        description="Close matplotlib figures every N files to prevent memory leaks"
    )
    
    # Large-scale processing optimizations
    high_performance_mode: bool = Field(
        default=False,
        description="Enable high-performance mode for large batches"
    )
    disable_plots_large_batch: int = Field(
        default=500,
        ge=10,
        description="Disable plot generation for batches larger than this"
    )
    enable_streaming_processing: bool = Field(
        default=True,
        description="Enable streaming processing for memory efficiency"
    )
    max_concurrent_files: int = Field(
        default=50,  # Process up to 50 files concurrently
        ge=1,
        le=200,
        description="Maximum files to process concurrently"
    )
    concurrent_batch_size: int = Field(
        default=20,  # Process in smaller concurrent batches
        ge=5,
        le=100,
        description="Size of concurrent processing batches for memory management"
    )
    batch_commit_interval: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Commit database results every N files"
    )
    
    # Database optimizations for large batches
    enable_bulk_insert: bool = Field(
        default=True,
        description="Enable bulk database inserts for better performance"
    )
    database_batch_size: int = Field(
        default=200,
        ge=10,
        le=1000,
        description="Number of results to batch for database operations"
    )
    
    # File system optimizations
    enable_fast_metadata_scan: bool = Field(
        default=True,
        description="Use fast metadata scanning for large directories"
    )
    skip_duplicate_files: bool = Field(
        default=True,
        description="Skip files that have already been processed"
    )
    
    # Memory management
    gc_interval: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Force garbage collection every N files"
    )
    clear_cache_interval: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Clear file cache every N files to free memory"
    )

    # Phase 2: UnifiedProcessor Feature Flags (ADR-001, ADR-004)
    use_unified_processor: bool = Field(
        default=False,
        description="Use UnifiedProcessor instead of legacy processors (Phase 2)"
    )
    unified_processor_strategy: str = Field(
        default="auto",
        description="UnifiedProcessor strategy: 'auto', 'standard', 'turbo', 'memory_safe'"
    )

    # Phase 3: ML Integration Feature Flags (ADR-001, ADR-005)
    use_ml_failure_predictor: bool = Field(
        default=False,
        description="Use ML-based failure prediction instead of formula (Phase 3)"
    )
    use_ml_drift_detector: bool = Field(
        default=False,
        description="Use ML-based drift detection for historical analysis (Phase 3)"
    )


class AnalysisConfig(BaseSettings):
    """Analysis parameters configuration."""
    # Sigma analysis
    sigma_scaling_factor: float = Field(default=24.0, gt=0, description="Sigma threshold scaling")
    matlab_gradient_step: int = Field(default=3, ge=1, description="Gradient calculation step")

    # Filtering
    filter_sampling_frequency: int = Field(default=100, gt=0)
    filter_cutoff_frequency: int = Field(default=40, gt=0)
    
    # Lockheed Martin Compliance Mode
    lockheed_martin_compliance_mode: bool = Field(
        default=False, 
        description="Use original Lockheed Martin specifications (may violate technical limits)"
    )

    # Zone analysis
    default_num_zones: int = Field(default=5, ge=1, le=20)

    # Risk thresholds
    high_risk_threshold: float = Field(default=0.7, ge=0, le=1)
    low_risk_threshold: float = Field(default=0.3, ge=0, le=1)

    @field_validator('high_risk_threshold')
    @classmethod
    def validate_risk_thresholds(cls, v: float, info) -> float:
        """Ensure high risk threshold is greater than low."""
        if 'low_risk_threshold' in info.data and v <= info.data['low_risk_threshold']:
            raise ValueError("High risk threshold must be greater than low risk threshold")
        return v

    def get_effective_cutoff_frequency(self) -> int:
        """Get the effective cutoff frequency based on compliance mode."""
        from laser_trim_analyzer.core.constants import (
            FILTER_CUTOFF_FREQUENCY_LM_ORIGINAL, 
            FILTER_CUTOFF_FREQUENCY_CORRECTED
        )
        
        if self.lockheed_martin_compliance_mode:
            return FILTER_CUTOFF_FREQUENCY_LM_ORIGINAL  # 80 Hz - original LM spec
        else:
            return FILTER_CUTOFF_FREQUENCY_CORRECTED   # 40 Hz - technically correct


class MLConfig(BaseSettings):
    """Machine learning configuration."""
    enabled: bool = Field(default=True, description="Enable ML features")
    model_path: Path = Field(
        default=Path.home() / ".laser_trim_analyzer" / "models",
        description="ML models directory"
    )
    
    # Failure prediction
    failure_prediction_enabled: bool = Field(default=True)
    failure_prediction_confidence_threshold: float = Field(default=0.85, ge=0.5, le=1.0)  # Increased for production
    
    # Threshold optimization
    threshold_optimization_enabled: bool = Field(default=True)
    threshold_optimization_min_samples: int = Field(default=500, ge=50)  # Increased for production
    
    # Model retraining
    retrain_interval_days: int = Field(default=30, ge=1, description="Days between automatic retraining")
    min_training_samples: int = Field(default=1000, ge=100, description="Minimum samples required for training")  # Increased for production
    
    # Performance thresholds
    model_performance_threshold: float = Field(default=0.80, ge=0.5, le=1.0)
    drift_detection_threshold: float = Field(default=0.15, ge=0.01, le=1.0)
    
    @field_validator('model_path')
    @classmethod
    def ensure_model_directory(cls, v: Path) -> Path:
        """Ensure model directory exists."""
        try:
            # Handle string paths with environment variables
            if isinstance(v, str):
                expanded = os.path.expandvars(v)
                v = Path(expanded)
            
            # Don't make relative paths absolute if they contain drive letters
            if not v.is_absolute() and not (len(str(v)) > 1 and str(v)[1] == ':'):
                v = Path.cwd() / v
                
            v.mkdir(parents=True, exist_ok=True)
            return v
        except Exception as e:
            logger.warning(f"Could not create model directory {v}: {e}")
            # Fallback to temp directory
            fallback = Path(tempfile.gettempdir()) / "laser_trim_analyzer" / "models"
            fallback.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using fallback model directory: {fallback}")
            return fallback


class APIConfig(BaseSettings):
    """API configuration for AI services."""
    enabled: bool = Field(default=True, description="Enable AI API integration")
    base_url: str = Field(default="http://localhost:8000", description="API base URL")
    api_key: Optional[str] = Field(default=None, description="API authentication key")
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")

    # AI features - enabled by default
    enable_anomaly_detection: bool = Field(default=True)
    enable_quality_predictions: bool = Field(default=True)
    enable_maintenance_suggestions: bool = Field(default=True)


class GUIConfig(BaseSettings):
    """GUI configuration."""
    theme: str = Field(default="clam", description="TTK theme name")
    window_width: int = Field(default=1200, ge=800)
    window_height: int = Field(default=900, ge=600)

    # Feature flags
    show_historical_tab: bool = Field(default=True)
    show_ml_insights: bool = Field(default=True)
    show_batch_processing: bool = Field(default=True)

    # Auto-save
    autosave_enabled: bool = Field(default=True)
    autosave_interval: int = Field(default=300, ge=60)  # seconds


class Config(BaseSettings):
    """Main application configuration."""
    model_config = SettingsConfigDict(
        env_prefix="LTA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application info
    app_name: str = Field(default="Laser Trim Analyzer", description="Application name")
    version: str = Field(default="2.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    first_run: bool = Field(default=True, description="First run flag")
    initialized: bool = Field(default=False, description="Initialization complete flag")

    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    gui: GUIConfig = Field(default_factory=GUIConfig)

    # Paths
    data_directory: Path = Field(
        default=Path.home() / "LaserTrimResults",
        description="Default data directory"
    )
    log_directory: Path = Field(
        default=Path.home() / ".laser_trim_analyzer" / "logs",
        description="Log file directory"
    )

    @field_validator('data_directory', 'log_directory')
    @classmethod
    def ensure_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        try:
            # Ensure we have a proper Path object, not a corrupted one
            if isinstance(v, str):
                # Expand environment variables first
                expanded = os.path.expandvars(v)
                v = Path(expanded)
            
            # Validate that this is a reasonable path (not the Windows PATH variable)
            path_str = str(v)
            if ';' in path_str and len(path_str) > 1000:
                # This looks like the Windows PATH environment variable
                # Fall back to default based on field name
                if 'log' in str(v).lower():
                    v = Path.home() / ".laser_trim_analyzer" / "logs"
                else:
                    v = Path.home() / "LaserTrimResults"
            
            # Don't make relative paths absolute if they contain drive letters
            if not v.is_absolute() and not (len(str(v)) > 1 and str(v)[1] == ':'):
                v = Path.cwd() / v
            
            # Ensure the directory exists
            v.mkdir(parents=True, exist_ok=True)
            return v
        except (OSError, ValueError) as e:
            # If there's any issue creating the directory, fall back to safe defaults
            if 'log' in str(v).lower():
                default_path = Path.home() / ".laser_trim_analyzer" / "logs"
            else:
                default_path = Path.home() / "LaserTrimResults"
            default_path.mkdir(parents=True, exist_ok=True)
            return default_path

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Recursively expand environment variables in the config data
        def expand_env_vars(obj):
            if isinstance(obj, dict):
                return {k: expand_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [expand_env_vars(item) for item in obj]
            elif isinstance(obj, str):
                # Special handling to prevent PATH environment variable corruption
                expanded = os.path.expandvars(obj)
                # Check if this looks like the Windows PATH variable was expanded
                if ';' in expanded and ('Program Files' in expanded or 'Windows' in expanded) and len(expanded) > 500:
                    logger.warning(f"Detected corrupted PATH expansion in config value: {obj}")
                    # Return the original string without expansion
                    return obj
                return expanded
            else:
                return obj
        
        config_data = expand_env_vars(config_data)
        
        return cls(**config_data)

    def to_yaml(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        config_data = self.model_dump(exclude_none=True)

        # Convert Path objects to strings for YAML
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        config_data = convert_paths(config_data)

        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

    def get_model_config(self, model: str) -> Dict[str, Any]:
        """Get model-specific configuration overrides."""
        # This can be extended to load model-specific configs
        model_configs = {
            "8340": {
                "analysis": {
                    "sigma_scaling_factor": 24.0,
                }
            },
            "8555": {
                "analysis": {
                    "sigma_scaling_factor": 36.0,  # Example adjustment
                }
            }
        }
        return model_configs.get(model, {})


@lru_cache()
def get_config() -> Config:
    """Get application configuration (cached singleton)."""
    # Check for environment variable to determine which config to use
    env = os.environ.get('LTA_ENV', 'production').lower()
    
    # Try to load from environment-specific config first
    config_paths = []
    
    if env == 'development':
        config_paths.append(Path("config/development.yaml"))
    elif env == 'production':
        config_paths.append(Path("config/production.yaml"))
    
    # Add fallback paths
    config_paths.extend([
        Path("config/default.yaml"),
        Path.home() / ".laser_trim_analyzer" / "config.yaml",
    ])

    for config_path in config_paths:
        if config_path.exists():
            logger.info(f"Loading configuration from: {config_path}")
            config = Config.from_yaml(config_path)
            
            # Debug log the database path
            if hasattr(config, 'database') and hasattr(config.database, 'path'):
                logger.debug(f"Loaded database path: {config.database.path}")
                # Check if path looks corrupted
                path_str = str(config.database.path)
                if ';' in path_str and len(path_str) > 500:
                    logger.error(f"WARNING: Database path appears to be corrupted with PATH env var!")
                    logger.error(f"First 200 chars: {path_str[:200]}...")
            
            return config

    # Fall back to environment/defaults
    logger.warning("No configuration file found, using defaults")
    return Config()


# Create default configuration files
def create_default_configs():
    """Create default configuration files."""
    default_config = Config()

    # Create config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    # Save default config
    default_config.to_yaml(config_dir / "default.yaml")

    # Create production config with some overrides
    prod_config = Config(
        debug=False,
        processing=ProcessingConfig(
            max_workers=8,
            generate_plots=False  # Faster for production
        ),
        ml=MLConfig(
            enabled=True,
            failure_prediction_confidence_threshold=0.9
        )
    )
    prod_config.to_yaml(config_dir / "production.yaml")


if __name__ == "__main__":
    # Create default configs when run directly
    create_default_configs()
    print("Default configuration files created in config/")
