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


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    enabled: bool = Field(default=True, description="Enable database storage")
    path: Path = Field(
        default=Path.home() / ".laser_trim_analyzer" / "analysis.db",
        description="Database file path"
    )
    echo: bool = Field(default=False, description="Echo SQL statements")
    pool_size: int = Field(default=5, ge=1, description="Connection pool size")

    @field_validator('path')
    @classmethod
    def ensure_db_directory(cls, v: Path) -> Path:
        """Ensure database directory exists."""
        v.parent.mkdir(parents=True, exist_ok=True)
        return v


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


class AnalysisConfig(BaseSettings):
    """Analysis parameters configuration."""
    # Sigma analysis
    sigma_scaling_factor: float = Field(default=24.0, gt=0, description="Sigma threshold scaling")
    matlab_gradient_step: int = Field(default=3, ge=1, description="Gradient calculation step")

    # Filtering
    filter_sampling_frequency: int = Field(default=100, gt=0)
    filter_cutoff_frequency: int = Field(default=80, gt=0)

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


class MLConfig(BaseSettings):
    """Machine learning configuration."""
    enabled: bool = Field(default=True, description="Enable ML features")
    model_path: Path = Field(
        default=Path.home() / ".laser_trim_analyzer" / "models",
        description="ML model storage path"
    )

    # Failure prediction
    failure_prediction_enabled: bool = Field(default=True)
    failure_prediction_confidence_threshold: float = Field(default=0.8, ge=0, le=1)

    # Threshold optimization
    threshold_optimization_enabled: bool = Field(default=True)
    threshold_optimization_min_samples: int = Field(default=100, ge=10)

    # Model training
    retrain_interval_days: int = Field(default=30, ge=1)
    min_training_samples: int = Field(default=1000, ge=100)


class APIConfig(BaseSettings):
    """API configuration for AI services."""
    enabled: bool = Field(default=False, description="Enable AI API integration")
    base_url: str = Field(default="http://localhost:8000", description="API base URL")
    api_key: Optional[str] = Field(default=None, description="API authentication key")
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")

    # AI features
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
    version: str = Field(default="2.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")

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
        v.mkdir(parents=True, exist_ok=True)
        return v

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

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
    # Try to load from default locations
    config_paths = [
        Path("config/production.yaml"),
        Path("config/default.yaml"),
        Path.home() / ".laser_trim_analyzer" / "config.yaml",
    ]

    for config_path in config_paths:
        if config_path.exists():
            return Config.from_yaml(config_path)

    # Fall back to environment/defaults
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