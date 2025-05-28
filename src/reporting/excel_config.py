"""
Excel Reporter Configuration

Configuration settings for the Excel Report Generator module.

Author: Laser Trim AI System
Date: 2024
Version: 1.0.0
"""

import os
from typing import Dict, Any, List, Optional


class ExcelReportConfig:
    """Configuration class for Excel Report generation."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with defaults and optional overrides.

        Args:
            config_dict: Optional dictionary to override default settings
        """
        # Default configuration
        self.config = self._get_default_config()

        # Apply any overrides
        if config_dict:
            self._update_config(config_dict)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            # API Configuration
            "api": {
                "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
                "openai_model": "gpt-4",
                "max_tokens": 500,
                "temperature": 0.7
            },

            # Report Settings
            "report": {
                "include_ai_insights": True,
                "include_charts": True,
                "include_raw_data": True,
                "max_rows_per_sheet": 1000000,
                "decimal_places": 4,
                "percentage_decimal_places": 2
            },

            # Sheet Configuration
            "sheets": {
                "executive_summary": {
                    "enabled": True,
                    "include_model_chart": True,
                    "include_trend_summary": True
                },
                "detailed_analysis": {
                    "enabled": True,
                    "include_filters": True,
                    "freeze_panes": True
                },
                "statistical_summary": {
                    "enabled": True,
                    "include_distribution_charts": True,
                    "metrics": [
                        "sigma_gradient",
                        "failure_probability",
                        "resistance_change_percent",
                        "trim_improvement_percent",
                        "unit_length"
                    ]
                },
                "trend_analysis": {
                    "enabled": True,
                    "days_to_analyze": 30,
                    "include_predictions": True,
                    "prediction_days": 7
                },
                "quality_metrics": {
                    "enabled": True,
                    "include_kpi_dashboard": True,
                    "target_cpk": 1.33,
                    "target_fpy": 0.95
                },
                "recommendations": {
                    "enabled": True,
                    "max_recommendations": 10,
                    "include_action_plan": True,
                    "action_plan_steps": 5
                },
                "raw_data": {
                    "enabled": True,
                    "include_all_columns": False,
                    "selected_columns": None  # None means auto-select
                }
            },

            # Formatting Settings
            "formatting": {
                "title_font_size": 16,
                "header_font_size": 12,
                "data_font_size": 10,
                "row_height": 15,
                "use_alternating_rows": True,
                "color_scheme": "blue"  # blue, green, orange, custom
            },

            # Chart Settings
            "charts": {
                "default_width": 600,
                "default_height": 400,
                "use_3d": False,
                "show_data_labels": True,
                "legend_position": "bottom"
            },

            # Performance Settings
            "performance": {
                "use_caching": True,
                "cache_dir": "cache/excel_reports",
                "parallel_processing": True,
                "chunk_size": 1000
            },

            # Export Settings
            "export": {
                "compress_images": True,
                "remove_personal_info": False,
                "add_watermark": False,
                "watermark_text": "CONFIDENTIAL"
            },

            # Thresholds for Status Classification
            "thresholds": {
                "pass_rate_critical": 0.80,
                "pass_rate_warning": 0.90,
                "sigma_gradient_warning": 0.003,
                "sigma_gradient_critical": 0.004,
                "high_risk_threshold": 0.70,
                "medium_risk_threshold": 0.30
            },

            # Custom Formulas
            "formulas": {
                "cpk_formula": "=(mean_threshold - mean_sigma) / (3 * std_sigma)",
                "dpm_formula": "=(1 - first_pass_yield) * 1000000",
                "oee_formula": "=availability * performance * quality"
            }
        }

    def _update_config(self, updates: Dict[str, Any]):
        """
        Recursively update configuration with new values.

        Args:
            updates: Dictionary of updates to apply
        """

        def deep_update(base_dict: dict, update_dict: dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(self.config, updates)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., "report.decimal_places")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config_dict = self.config

        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config_dict:
                config_dict[key] = {}
            config_dict = config_dict[key]

        # Set the value
        config_dict[keys[-1]] = value

    def get_enabled_sheets(self) -> List[str]:
        """Get list of enabled report sheets."""
        enabled = []
        sheets_config = self.config.get("sheets", {})

        for sheet_name, sheet_config in sheets_config.items():
            if isinstance(sheet_config, dict) and sheet_config.get("enabled", True):
                enabled.append(sheet_name)

        return enabled

    def get_color_scheme(self) -> Dict[str, str]:
        """Get color scheme for report formatting."""
        scheme_name = self.config["formatting"]["color_scheme"]

        color_schemes = {
            "blue": {
                "primary": "#4472C4",
                "secondary": "#D9E2F3",
                "accent": "#70AD47",
                "warning": "#FFA500",
                "error": "#FF0000",
                "success": "#00B050"
            },
            "green": {
                "primary": "#70AD47",
                "secondary": "#E2EFDA",
                "accent": "#4472C4",
                "warning": "#FFA500",
                "error": "#FF0000",
                "success": "#00B050"
            },
            "orange": {
                "primary": "#ED7D31",
                "secondary": "#FBE4D5",
                "accent": "#4472C4",
                "warning": "#FFA500",
                "error": "#FF0000",
                "success": "#00B050"
            },
            "custom": self.config.get("custom_colors", {})
        }

        return color_schemes.get(scheme_name, color_schemes["blue"])

    def validate(self) -> List[str]:
        """
        Validate configuration settings.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check API key if AI insights are enabled
        if self.get("report.include_ai_insights") and not self.get("api.openai_api_key"):
            errors.append("OpenAI API key required when AI insights are enabled")

        # Check thresholds
        if self.get("thresholds.pass_rate_warning") <= self.get("thresholds.pass_rate_critical"):
            errors.append("Warning pass rate threshold must be higher than critical threshold")

        # Check decimal places
        if self.get("report.decimal_places", 4) < 0 or self.get("report.decimal_places", 4) > 10:
            errors.append("Decimal places must be between 0 and 10")

        # Check enabled sheets
        if not self.get_enabled_sheets():
            errors.append("At least one report sheet must be enabled")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()

    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        import json

        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'ExcelReportConfig':
        """Load configuration from JSON file."""
        import json

        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        return cls(config_dict)


# Convenience functions
def get_default_config() -> ExcelReportConfig:
    """Get default Excel report configuration."""
    return ExcelReportConfig()


def create_custom_config(**kwargs) -> ExcelReportConfig:
    """
    Create custom configuration with specific overrides.

    Example:
        config = create_custom_config(
            report={'decimal_places': 6},
            formatting={'color_scheme': 'green'}
        )
    """
    return ExcelReportConfig(kwargs)


# Example configurations for different use cases
def get_minimal_config() -> ExcelReportConfig:
    """Get minimal configuration for fast report generation."""
    return ExcelReportConfig({
        "report": {
            "include_ai_insights": False,
            "include_charts": False,
            "include_raw_data": False
        },
        "sheets": {
            "executive_summary": {"enabled": True},
            "detailed_analysis": {"enabled": True},
            "statistical_summary": {"enabled": False},
            "trend_analysis": {"enabled": False},
            "quality_metrics": {"enabled": False},
            "recommendations": {"enabled": True},
            "raw_data": {"enabled": False}
        },
        "performance": {
            "parallel_processing": False
        }
    })


def get_full_config() -> ExcelReportConfig:
    """Get full configuration with all features enabled."""
    config = ExcelReportConfig()
    # All features are enabled by default
    return config


def get_production_config() -> ExcelReportConfig:
    """Get production-ready configuration."""
    return ExcelReportConfig({
        "report": {
            "include_ai_insights": True,
            "max_rows_per_sheet": 500000  # Limit for performance
        },
        "export": {
            "compress_images": True,
            "remove_personal_info": True,
            "add_watermark": True
        },
        "performance": {
            "use_caching": True,
            "parallel_processing": True,
            "chunk_size": 5000
        }
    })


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()

    # Validate
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration is valid")

    # Test getting values
    print(f"\nDecimal places: {config.get('report.decimal_places')}")
    print(f"Color scheme: {config.get('formatting.color_scheme')}")
    print(f"Enabled sheets: {config.get_enabled_sheets()}")

    # Test setting values
    config.set('report.decimal_places', 6)
    print(f"\nUpdated decimal places: {config.get('report.decimal_places')}")

    # Save example config
    config.save_to_file("excel_config_example.json")
    print("\nExample configuration saved to excel_config_example.json")