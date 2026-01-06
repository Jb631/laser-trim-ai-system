"""
Configuration management for v3.

Simplified from v2's complex config system.
Single source of truth for all configuration.

DESIGN DECISION: Self-contained deployment
- Database lives in ./data/ relative to app
- All data in one folder for easy backup/migration
- No scattered config locations across system
"""

import os
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import yaml
import logging

logger = logging.getLogger(__name__)


def get_app_directory() -> Path:
    """
    Get the application directory.

    For deployed app: directory containing the executable
    For development: project root (laser-trim-ai-system/)

    Path from config.py:
    - config.py is in src/laser_trim_analyzer/
    - .parent = laser_trim_analyzer/
    - .parent.parent = src/
    - .parent.parent.parent = laser-trim-ai-system/ (project root)
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable (PyInstaller)
        return Path(sys.executable).parent
    else:
        # Running from source - go up to project root
        # config.py -> laser_trim_analyzer -> src -> project_root
        return Path(__file__).parent.parent.parent


@dataclass
class DatabaseConfig:
    """
    Database configuration.

    Default: ./data/analysis.db (relative to app directory)
    This keeps everything self-contained for easy deployment.
    """
    path: Path = field(default_factory=lambda: get_app_directory() / "data" / "analysis.db")
    echo: bool = False

    def ensure_directory(self) -> Path:
        """Ensure database directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self.path


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    batch_size: int = 100
    incremental: bool = True  # Default ON (90% use case)
    generate_plots: bool = True
    save_to_database: bool = True
    turbo_mode_threshold: int = 100  # Files before turbo mode kicks in


@dataclass
class MLConfig:
    """ML configuration."""
    enabled: bool = True
    use_threshold_optimizer: bool = True
    use_drift_detector: bool = True
    min_samples_for_training: int = 100


@dataclass
class GUIConfig:
    """GUI configuration."""
    theme: str = "dark"  # dark or light
    window_width: int = 1400
    window_height: int = 900
    remember_last_directory: bool = True
    last_directory: Optional[str] = None


@dataclass
class ModelsConfig:
    """ML models configuration."""
    path: Path = field(default_factory=lambda: get_app_directory() / "data" / "models")

    def ensure_directory(self) -> Path:
        """Ensure models directory exists."""
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path


@dataclass
class Config:
    """Main configuration container."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)

    # Version info
    version: str = "3.0.0"

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Load configuration from YAML file.

        Falls back to defaults if file doesn't exist.
        Config is stored in ./data/config.yaml for self-contained deployment.
        """
        config = cls()

        if config_path is None:
            # Config in data folder - truly self-contained deployment
            config_path = get_app_directory() / "data" / "config.yaml"

        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = yaml.safe_load(f) or {}

                # Apply loaded values
                if "database" in data:
                    for key, value in data["database"].items():
                        if hasattr(config.database, key):
                            if key == "path":
                                value = Path(os.path.expandvars(str(value)))
                            setattr(config.database, key, value)

                if "processing" in data:
                    for key, value in data["processing"].items():
                        if hasattr(config.processing, key):
                            setattr(config.processing, key, value)

                if "ml" in data:
                    for key, value in data["ml"].items():
                        if hasattr(config.ml, key):
                            setattr(config.ml, key, value)

                if "gui" in data:
                    for key, value in data["gui"].items():
                        if hasattr(config.gui, key):
                            setattr(config.gui, key, value)

                logger.info(f"Loaded config from {config_path}")

            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return config

    def save(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to YAML file in ./data/ folder."""
        if config_path is None:
            # Config in data folder - truly self-contained deployment
            config_path = get_app_directory() / "data" / "config.yaml"

        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "database": {
                "path": str(self.database.path),
                "echo": self.database.echo,
            },
            "processing": {
                "batch_size": self.processing.batch_size,
                "incremental": self.processing.incremental,
                "generate_plots": self.processing.generate_plots,
                "save_to_database": self.processing.save_to_database,
                "turbo_mode_threshold": self.processing.turbo_mode_threshold,
            },
            "ml": {
                "enabled": self.ml.enabled,
                "use_threshold_optimizer": self.ml.use_threshold_optimizer,
                "use_drift_detector": self.ml.use_drift_detector,
                "min_samples_for_training": self.ml.min_samples_for_training,
            },
            "gui": {
                "theme": self.gui.theme,
                "window_width": self.gui.window_width,
                "window_height": self.gui.window_height,
                "remember_last_directory": self.gui.remember_last_directory,
                "last_directory": self.gui.last_directory,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"Saved config to {config_path}")


# Singleton config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reload_config(config_path: Optional[Path] = None) -> Config:
    """Reload configuration from file."""
    global _config
    _config = Config.load(config_path)
    return _config
