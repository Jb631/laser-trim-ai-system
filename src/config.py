"""
Configuration management system.

Handles loading and managing configuration from files and environment.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from src.core.constants import *


@dataclass
class SystemConfig:
    """System-wide configuration settings."""

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    model_dir: Path = field(default_factory=lambda: Path("models"))

    # Processing settings
    batch_size: int = 100
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    generate_plots: bool = True

    # Analysis settings
    sigma_scaling_factor: float = DEFAULT_SIGMA_SCALING_FACTOR
    filter_cutoff: int = FILTER_CUTOFF_FREQUENCY
    filter_sampling: int = FILTER_SAMPLING_FREQUENCY
    gradient_step: int = MATLAB_GRADIENT_STEP

    # ML settings
    enable_ml: bool = True
    auto_threshold_optimization: bool = True
    failure_prediction_enabled: bool = True
    drift_detection_window_days: int = 30

    # Reporting settings
    excel_reports: bool = True
    html_reports: bool = True
    include_plots_in_reports: bool = True

    # Data validation
    strict_validation: bool = False
    min_data_points: int = MIN_DATA_POINTS

    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_attr in ["data_dir", "output_dir", "log_dir", "model_dir"]:
            dir_path = getattr(self, dir_attr)
            if isinstance(dir_path, str):
                dir_path = Path(dir_path)
                setattr(self, dir_attr, dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Manages system configuration."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to configuration file (YAML)
        """
        self.config = SystemConfig()
        self._config_file = config_file or self._find_config_file()

        if self._config_file and Path(self._config_file).exists():
            self.load_config(self._config_file)

        # Override with environment variables
        self._load_env_overrides()

    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations."""
        search_paths = [
            "config.yaml",
            "config/config.yaml",
            "config/default_config.yaml",
            "../config/config.yaml",
            os.path.expanduser("~/.laser_trim_ai/config.yaml")
        ]

        for path in search_paths:
            if Path(path).exists():
                return path

        return None

    def load_config(self, config_file: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)

            if config_data:
                self._update_config(config_data)

        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")

    def _update_config(self, config_data: Dict[str, Any]) -> None:
        """Update configuration with data from dict."""
        # Paths
        if "paths" in config_data:
            paths = config_data["paths"]
            for key in ["data_dir", "output_dir", "log_dir", "model_dir"]:
                if key in paths:
                    setattr(self.config, key, Path(paths[key]))

        # Processing settings
        if "processing" in config_data:
            proc = config_data["processing"]
            for key in ["batch_size", "parallel_processing", "max_workers", "generate_plots"]:
                if key in proc:
                    setattr(self.config, key, proc[key])

        # Analysis settings
        if "analysis" in config_data:
            analysis = config_data["analysis"]
            for key in ["sigma_scaling_factor", "filter_cutoff", "filter_sampling", "gradient_step"]:
                if key in analysis:
                    setattr(self.config, key, analysis[key])

        # ML settings
        if "ml" in config_data:
            ml = config_data["ml"]
            for key in ["enable_ml", "auto_threshold_optimization",
                        "failure_prediction_enabled", "drift_detection_window_days"]:
                if key in ml:
                    setattr(self.config, key, ml[key])

    def _load_env_overrides(self) -> None:
        """Load configuration overrides from environment variables."""
        # Example: LASER_TRIM_OUTPUT_DIR=/custom/path
        env_mappings = {
            "LASER_TRIM_OUTPUT_DIR": ("output_dir", Path),
            "LASER_TRIM_DATA_DIR": ("data_dir", Path),
            "LASER_TRIM_PARALLEL": ("parallel_processing", lambda x: x.lower() == "true"),
            "LASER_TRIM_MAX_WORKERS": ("max_workers", int),
            "LASER_TRIM_ENABLE_ML": ("enable_ml", lambda x: x.lower() == "true")
        }

        for env_var, (config_attr, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                try:
                    converted_value = converter(value)
                    setattr(self.config, config_attr, converted_value)
                except Exception as e:
                    print(f"Warning: Could not parse {env_var}={value}: {e}")

    def save_config(self, config_file: str) -> None:
        """Save current configuration to file."""
        config_data = {
            "paths": {
                "data_dir": str(self.config.data_dir),
                "output_dir": str(self.config.output_dir),
                "log_dir": str(self.config.log_dir),
                "model_dir": str(self.config.model_dir)
            },
            "processing": {
                "batch_size": self.config.batch_size,
                "parallel_processing": self.config.parallel_processing,
                "max_workers": self.config.max_workers,
                "generate_plots": self.config.generate_plots
            },
            "analysis": {
                "sigma_scaling_factor": self.config.sigma_scaling_factor,
                "filter_cutoff": self.config.filter_cutoff,
                "filter_sampling": self.config.filter_sampling,
                "gradient_step": self.config.gradient_step
            },
            "ml": {
                "enable_ml": self.config.enable_ml,
                "auto_threshold_optimization": self.config.auto_threshold_optimization,
                "failure_prediction_enabled": self.config.failure_prediction_enabled,
                "drift_detection_window_days": self.config.drift_detection_window_days
            }
        }

        Path(config_file).parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

    def get_output_path(self, filename: str, subfolder: Optional[str] = None) -> Path:
        """Get output path for a file, creating directories as needed."""
        if subfolder:
            output_path = self.config.output_dir / subfolder
        else:
            output_path = self.config.output_dir

        output_path.mkdir(parents=True, exist_ok=True)

        return output_path / filename

    def get_timestamped_output_dir(self, prefix: str = "run") -> Path:
        """Create and return a timestamped output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.config.output_dir / f"{prefix}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager()

    return _config_manager.config


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager()

    return _config_manager