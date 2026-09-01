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
from typing import Optional, List, Dict
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


# The drift presets the app accepts. Duplicated here rather than imported
# from ml.drift_types so config stays dependency-free (ml imports config);
# a test asserts the two lists stay identical.
DRIFT_SENSITIVITY_PRESETS = ("loose", "standard", "tight", "strict")


@dataclass
class MLConfig:
    """ML configuration."""
    enabled: bool = True
    use_threshold_optimizer: bool = True
    use_drift_detector: bool = True
    min_samples_for_training: int = 20
    drift_sensitivity: str = "standard"  # Spec 2: loose / standard / tight / strict


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
class ActiveModelsConfig:
    """Configuration for active model prioritization."""
    mps_models: List[str] = field(default_factory=list)  # User-managed MPS list
    recent_days: int = 90  # Days to consider "recently active"
    model_prices: Dict[str, float] = field(default_factory=dict)  # Model → unit price ($)
    cost_ratio: float = 0.50  # Fraction of unit price representing invested labor+material
    # Show the new unit-level yield surfaces (Trends single-model chart line
    # + Excel "Yield by Unit" sheet). Set to False to hide both instantly
    # without removing the underlying data. See
    # docs/superpowers/specs/2026-05-08-unit-level-yield-design.md.
    enable_unit_yield_view: bool = True


def normalize_ingest_path(path: str) -> str:
    """One canonical spelling of a folder, so duplicate detection can work.

    Cosmetic differences only: surrounding whitespace, environment variables,
    and a trailing separator. Nothing is resolved against the filesystem —
    these are network shares that are routinely offline when the app starts,
    and resolve()/realpath() on a dead SMB path blocks for the mount timeout.
    """
    p = os.path.expandvars(str(path or "").strip())
    while len(p) > 1 and p[-1] in ("/", "\\") and not p.endswith((":\\", ":/")):
        p = p[:-1]
    return p


def _ingest_key(path: str) -> str:
    """Comparison key: case-insensitive on Windows, exact everywhere else."""
    return os.path.normcase(normalize_ingest_path(path))


def ingest_folder_problem(path: str) -> Optional[str]:
    """None when the folder is usable; otherwise a sentence naming the problem.

    A folder that cannot be read must be REPORTED, never quietly skipped:
    these are shares like \\\\192.168.66.9\\... that do go offline, and a batch
    that silently processes 2 of 3 folders looks exactly like a batch that
    found nothing new.
    """
    p = normalize_ingest_path(path)
    if not p:
        return "empty path"
    try:
        target = Path(p)
        if not target.exists():
            return "not found — offline share, renamed, or unmapped drive?"
        if not target.is_dir():
            return "not a folder"
        os.listdir(target)
    except OSError as exc:
        return f"unreadable: {exc.strerror or exc}"
    return None


def missing_ingest_folders(folders: List[str]) -> List["tuple[str, str]"]:
    """[(folder, reason)] for every configured folder that cannot be used."""
    out = []
    for f in folders or []:
        problem = ingest_folder_problem(f)
        if problem:
            out.append((f, problem))
    return out


@dataclass
class IngestConfig:
    """The remembered ingest folders behind HOME's "Process everything new".

    ORDERED on purpose: the laser folders run first and the Final Test folder
    last, and the one-click batch walks the list top to bottom, so the order
    the user sets in Settings is the order the work happens in.

    Stored as plain strings, not Path: a UNC share is what the user typed and
    should round-trip through the config file unchanged.
    """
    folders: List[str] = field(default_factory=list)

    def index_of(self, path: str) -> Optional[int]:
        key = _ingest_key(path)
        for i, f in enumerate(self.folders):
            if _ingest_key(f) == key:
                return i
        return None

    def add(self, path: str) -> bool:
        """Append. False when the path is empty or already on the list."""
        p = normalize_ingest_path(path)
        if not p or self.index_of(p) is not None:
            return False
        self.folders.append(p)
        return True

    def remove(self, path: str) -> bool:
        i = self.index_of(path)
        if i is None:
            return False
        del self.folders[i]
        return True

    def move(self, index: int, delta: int) -> bool:
        """Move one entry by `delta` positions. False (no change) off either end."""
        j = index + delta
        if not (0 <= index < len(self.folders)) or not (0 <= j < len(self.folders)):
            return False
        self.folders.insert(j, self.folders.pop(index))
        return True


@dataclass
class Config:
    """Main configuration container."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    active_models: ActiveModelsConfig = field(default_factory=ActiveModelsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)

    # Export settings
    export_path: Optional[str] = None

    # Version info
    version: str = "5.0.0"

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
                    # An unknown preset is a KeyError deep in the drift
                    # threshold math (target_fp_for_tier), which would surface
                    # as a training crash rather than a bad setting. A config
                    # file is hand-editable, so validate here instead.
                    if config.ml.drift_sensitivity not in DRIFT_SENSITIVITY_PRESETS:
                        logger.warning(
                            "Unknown ml.drift_sensitivity %r in %s; using 'standard'",
                            config.ml.drift_sensitivity, config_path,
                        )
                        config.ml.drift_sensitivity = "standard"

                if "gui" in data:
                    for key, value in data["gui"].items():
                        if hasattr(config.gui, key):
                            setattr(config.gui, key, value)

                if "active_models" in data:
                    for key, value in data["active_models"].items():
                        if hasattr(config.active_models, key):
                            setattr(config.active_models, key, value)

                if "ingest" in data:
                    # Order is the contract; only str entries survive, so a
                    # hand-edited config that put a mapping here degrades to
                    # "no folders" (empty state) instead of crashing startup.
                    raw = (data["ingest"] or {}).get("folders") \
                        if isinstance(data["ingest"], dict) else None
                    if isinstance(raw, list):
                        config.ingest.folders = [
                            normalize_ingest_path(f) for f in raw
                            if isinstance(f, str) and normalize_ingest_path(f)
                        ]

                if "export_path" in data:
                    config.export_path = data["export_path"]

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
                # Omitting this dropped the user's drift preset on every save:
                # load() applies whatever keys it finds, so a key save() never
                # writes silently reverts to the dataclass default.
                "drift_sensitivity": self.ml.drift_sensitivity,
            },
            "gui": {
                "theme": self.gui.theme,
                "window_width": self.gui.window_width,
                "window_height": self.gui.window_height,
                "remember_last_directory": self.gui.remember_last_directory,
                "last_directory": self.gui.last_directory,
            },
            "active_models": {
                "mps_models": self.active_models.mps_models,
                "recent_days": self.active_models.recent_days,
                "model_prices": self.active_models.model_prices,
                "cost_ratio": self.active_models.cost_ratio,
                "enable_unit_yield_view": self.active_models.enable_unit_yield_view,
            },
            "ingest": {
                "folders": list(self.ingest.folders),
            },
            "export_path": self.export_path,
        }

        # Atomic save: write to temp file, then os.replace() so a crash
        # mid-write never corrupts the config file.
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=config_path.parent, suffix=".tmp", prefix=".config_"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
            os.replace(tmp_path, config_path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

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
