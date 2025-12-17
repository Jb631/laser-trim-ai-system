"""
Utility modules for the Laser Trim Analyzer.

Provides common functionality for data processing, filtering, plotting, and validation.
"""

from laser_trim_analyzer.utils.filter_utils import (
    apply_filter,
    smooth_data,
    remove_outliers,
    interpolate_missing_data
)
from laser_trim_analyzer.utils.plotting_utils import (
    create_analysis_plot,
    create_histogram,
    create_trend_chart,
    create_comparison_plot,
    save_plot
)
from laser_trim_analyzer.utils.validators import (
    validate_excel_file,
    validate_analysis_data,
    validate_model_number,
    validate_resistance_values,
    ValidationResult
)
from laser_trim_analyzer.utils.file_utils import (
    ensure_directory,
    get_unique_filename,
    calculate_file_hash,
    #cleanup_old_files
)
from laser_trim_analyzer.utils.excel_utils import (
    read_excel_sheet,
    extract_cell_value,
    find_data_columns,
    detect_system_type,
    #get_sheet_names
)

__all__ = [
    # Filter utilities
    "apply_filter",
    "smooth_data",
    "remove_outliers",
    "interpolate_missing_data",

    # Plotting utilities
    "create_analysis_plot",
    "create_histogram",
    "create_trend_chart",
    "create_comparison_plot",
    "save_plot",

    # Validators
    "validate_excel_file",
    "validate_analysis_data",
    "validate_model_number",
    "validate_resistance_values",
    "ValidationResult",

    # File utilities
    "ensure_directory",
    "get_unique_filename",
    "calculate_file_hash",
    #"cleanup_old_files",

    # Excel utilities
    "read_excel_sheet",
    "extract_cell_value",
    "find_data_columns",
    "detect_system_type",
    #"get_sheet_names",
]