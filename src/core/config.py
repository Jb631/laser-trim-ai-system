"""
Configuration Module for AI-Powered Laser Trim Analysis System

This module provides centralized configuration management for the entire system,
including processing parameters, file paths, and calibration settings.

Author: QA Team
Date: 2024
Version: 1.0.0
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging


@dataclass
class ProcessingConfig:
    """Configuration for data processing parameters."""
    # Filter parameters (MATLAB-compatible)
    filter_sampling_freq: int = 100
    filter_cutoff_freq: int = 80

    # Gradient calculation
    gradient_step_size: int = 3

    # Threshold calculation
    default_scaling_factor: float = 24.0

    # Data cleaning
    endpoint_removal_count: int = 7

    # Validation
    min_data_points: int = 20
    max_position_range: float = 1000.0
    max_error_magnitude: float = 1.0


@dataclass
class SystemAConfig:
    """Configuration specific to System A files."""
    # Column indices (0-based)
    columns: Dict[str, int] = None

    # Cell references for unit properties
    unit_length_cell: str = "B26"
    resistance_cell: str = "B10"

    # Sheet name patterns
    untrimmed_pattern: str = "SEC1 TRK{} 0"
    trimmed_pattern: str = "SEC1 TRK{} TRM"

    # Track identifiers
    track_ids: list = None

    def __post_init__(self):
        if self.columns is None:
            self.columns = {
                'measured_volts': 3,
                'index': 4,
                'theory_volts': 5,
                'error': 6,
                'position': 7,
                'upper_limit': 8,
                'lower_limit': 9
            }
        if self.track_ids is None:
            self.track_ids = ['1', '2']


@dataclass
class SystemBConfig:
    """Configuration specific to System B files."""
    # Column indices (0-based)
    columns: Dict[str, int] = None

    # Cell references for unit properties
    unit_length_cell: str = "K1"
    resistance_cell: str = "R1"

    # Sheet names
    untrimmed_sheet: str = "test"
    final_sheet: str = "Lin Error"

    def __post_init__(self):
        if self.columns is None:
            self.columns = {
                'error': 3,
                'upper_limit': 5,
                'lower_limit': 6,
                'position': 8
            }


@dataclass
class CalibrationConfig:
    """Configuration for model-specific calibrations."""
    # Model-specific scaling factors
    model_scaling_factors: Dict[str, float] = None

    # Model patterns for system detection
    system_a_models: list = None
    system_b_models: list = None

    # Special handling flags
    apply_model_specific_thresholds: bool = True

    def __post_init__(self):
        if self.model_scaling_factors is None:
            self.model_scaling_factors = {
                '8340-1': 0.4,  # Fixed threshold for 8340-1
                'default': 24.0
            }
        if self.system_a_models is None:
            self.system_a_models = ['68', '78', '85']
        if self.system_b_models is None:
            self.system_b_models = ['8340', '834']


@dataclass
class OutputConfig:
    """Configuration for output and reporting."""
    # Output formats
    save_raw_data: bool = True
    save_filtered_data: bool = True
    save_gradients: bool = False

    # Report settings
    decimal_places: int = 6
    include_plots: bool = True

    # File naming
    output_filename_pattern: str = "{model}_{serial}_{timestamp}"
    timestamp_format: str = "%Y%m%d_%H%M%S"


@dataclass
class SystemConfig:
    """Main system configuration container."""
    processing: ProcessingConfig = None
    system_a: SystemAConfig = None
    system_b: SystemBConfig = None
    calibration: CalibrationConfig = None
    output: OutputConfig = None

    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.system_a is None:
            self.system_a = SystemAConfig()
        if self.system_b is None:
            self.system_b = SystemBConfig()
        if self.calibration is None:
            self.calibration = CalibrationConfig()
        if self.output is None:
            self.output = OutputConfig()


class ConfigManager:
    """
    Configuration manager for the laser trim analysis system.

    Handles loading, saving, and validation of configuration settings.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file (JSON format)
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = SystemConfig()
        self.logger = self._setup_logger()

        # Load configuration if path provided
        if self.config_path and self.config_path.exists():
            self.load_config()

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for configuration manager."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(self.config.log_format)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, self.config.log_level))
        return logger

    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to configuration file
        """
        if config_path:
            self.config_path = Path(config_path)

        if not self.config_path or not self.config_path.exists():
            raise ValueError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)

            # Load each configuration section
            if 'processing' in config_data:
                self.config.processing = ProcessingConfig(**config_data['processing'])
            if 'system_a' in config_data:
                self.config.system_a = SystemAConfig(**config_data['system_a'])
            if 'system_b' in config_data:
                self.config.system_b = SystemBConfig(**config_data['system_b'])
            if 'calibration' in config_data:
                self.config.calibration = CalibrationConfig(**config_data['calibration'])
            if 'output' in config_data:
                self.config.output = OutputConfig(**config_data['output'])

            # Load root-level settings
            if 'log_level' in config_data:
                self.config.log_level = config_data['log_level']
            if 'log_format' in config_data:
                self.config.log_format = config_data['log_format']

            self.logger.info(f"Configuration loaded from: {self.config_path}")

        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise

    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to JSON file.

        Args:
            config_path: Path to save configuration file
        """
        if config_path:
            self.config_path = Path(config_path)

        if not self.config_path:
            raise ValueError("No configuration path specified")

        try:
            # Convert configuration to dictionary
            config_dict = {
                'processing': asdict(self.config.processing),
                'system_a': asdict(self.config.system_a),
                'system_b': asdict(self.config.system_b),
                'calibration': asdict(self.config.calibration),
                'output': asdict(self.config.output),
                'log_level': self.config.log_level,
                'log_format': self.config.log_format
            }

            # Create directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)

            self.logger.info(f"Configuration saved to: {self.config_path}")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise

    def validate_config(self) -> bool:
        """
        Validate configuration settings.

        Returns:
            True if configuration is valid
        """
        is_valid = True

        # Validate processing config
        if self.config.processing.filter_sampling_freq <= 0:
            self.logger.error("Invalid filter sampling frequency")
            is_valid = False

        if self.config.processing.filter_cutoff_freq <= 0:
            self.logger.error("Invalid filter cutoff frequency")
            is_valid = False

        if self.config.processing.gradient_step_size <= 0:
            self.logger.error("Invalid gradient step size")
            is_valid = False

        # Validate column indices
        for system, config in [('A', self.config.system_a), ('B', self.config.system_b)]:
            for col_name, col_idx in config.columns.items():
                if col_idx < 0:
                    self.logger.error(f"Invalid column index for {col_name} in System {system}")
                    is_valid = False

        return is_valid

    def get_model_scaling_factor(self, model: str) -> float:
        """
        Get scaling factor for a specific model.

        Args:
            model: Model identifier

        Returns:
            Scaling factor for the model
        """
        # Check for exact match first
        if model in self.config.calibration.model_scaling_factors:
            return self.config.calibration.model_scaling_factors[model]

        # Check for pattern match (e.g., '8340-1' pattern)
        for pattern, factor in self.config.calibration.model_scaling_factors.items():
            if pattern in model:
                return factor

        # Return default
        return self.config.calibration.model_scaling_factors.get(
            'default',
            self.config.processing.default_scaling_factor
        )

    def update_setting(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific configuration setting.

        Args:
            section: Configuration section (e.g., 'processing', 'system_a')
            key: Setting key
            value: New value
        """
        if hasattr(self.config, section):
            section_config = getattr(self.config, section)
            if hasattr(section_config, key):
                setattr(section_config, key, value)
                self.logger.info(f"Updated {section}.{key} = {value}")
            else:
                self.logger.error(f"Unknown setting: {section}.{key}")
        else:
            self.logger.error(f"Unknown section: {section}")

    def reset_to_defaults(self) -> None:
        """Reset all configuration to default values."""
        self.config = SystemConfig()
        self.logger.info("Configuration reset to defaults")

    def create_default_config_file(self, path: Union[str, Path]) -> None:
        """
        Create a default configuration file.

        Args:
            path: Path where to create the configuration file
        """
        self.reset_to_defaults()
        self.save_config(path)


# Convenience function for creating processor with configuration
def create_configured_processor(config_path: Optional[Union[str, Path]] = None):
    """
    Create a DataProcessor instance with configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured DataProcessor instance
    """
    from data_processor import DataProcessor

    # Load configuration
    config_manager = ConfigManager(config_path)

    # Create processor
    processor = DataProcessor()

    # Apply configuration
    processor.FILTER_SAMPLING_FREQ = config_manager.config.processing.filter_sampling_freq
    processor.FILTER_CUTOFF_FREQ = config_manager.config.processing.filter_cutoff_freq
    processor.GRADIENT_STEP = config_manager.config.processing.gradient_step_size

    # Apply system-specific configurations
    processor.SYSTEM_A_COLUMNS = config_manager.config.system_a.columns
    processor.SYSTEM_B_COLUMNS = config_manager.config.system_b.columns

    # Store config manager reference for model-specific settings
    processor.config_manager = config_manager

    return processor


# Example usage
if __name__ == "__main__":
    # Create configuration manager
    config_mgr = ConfigManager()

    # Create default configuration file
    config_mgr.create_default_config_file("config/default_config.json")

    # Update a setting
    config_mgr.update_setting('processing', 'default_scaling_factor', 25.0)

    # Save updated configuration
    config_mgr.save_config("config/custom_config.json")

    # Validate configuration
    if config_mgr.validate_config():
        print("Configuration is valid")

    # Get model-specific scaling factor
    factor = config_mgr.get_model_scaling_factor('8340-1')
    print(f"Scaling factor for 8340-1: {factor}")