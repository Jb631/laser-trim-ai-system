"""
Setup script for AI-Powered Laser Trim Analysis System

This script helps set up the project structure and initial configuration.

Author: QA Team
Date: 2024
Version: 1.0.0
"""

import os
import sys
import json
from pathlib import Path


def create_project_structure():
    """Create the recommended project directory structure."""

    directories = [
        "core",
        "examples",
        "config",
        "data",
        "data/samples",
        "data/output",
        "docs",
        "tests",
        "logs",
        "ml_models",  # For future ML models
        "reports",  # For generated reports
        "gui",  # For future GUI
        "database"  # For future database
    ]

    print("Creating project directory structure...")

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {directory}/")

    # Create .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
env/
.env

# Data files (usually large)
data/samples/*
data/output/*
!data/samples/.gitkeep
!data/output/.gitkeep

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Project specific
reports/*
!reports/.gitkeep
*.xlsx
*.xls
!data/samples/example_*.xlsx

# ML Models (can be large)
ml_models/*.pkl
ml_models/*.h5
ml_models/*.pt
!ml_models/.gitkeep

# Temporary files
*.tmp
*.temp
.cache/
"""

    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("  ✓ Created: .gitignore")

    # Create .gitkeep files for empty directories
    for directory in ["data/samples", "data/output", "reports", "ml_models"]:
        gitkeep_path = Path(directory) / ".gitkeep"
        gitkeep_path.touch()

    print("\nProject structure created successfully!")


def create_default_configuration():
    """Create default configuration file."""

    default_config = {
        "processing": {
            "filter_sampling_freq": 100,
            "filter_cutoff_freq": 80,
            "gradient_step_size": 3,
            "default_scaling_factor": 24.0,
            "endpoint_removal_count": 7,
            "min_data_points": 20,
            "max_position_range": 1000.0,
            "max_error_magnitude": 1.0
        },
        "system_a": {
            "columns": {
                "measured_volts": 3,
                "index": 4,
                "theory_volts": 5,
                "error": 6,
                "position": 7,
                "upper_limit": 8,
                "lower_limit": 9
            },
            "unit_length_cell": "B26",
            "resistance_cell": "B10",
            "untrimmed_pattern": "SEC1 TRK{} 0",
            "trimmed_pattern": "SEC1 TRK{} TRM",
            "track_ids": ["1", "2"]
        },
        "system_b": {
            "columns": {
                "error": 3,
                "upper_limit": 5,
                "lower_limit": 6,
                "position": 8
            },
            "unit_length_cell": "K1",
            "resistance_cell": "R1",
            "untrimmed_sheet": "test",
            "final_sheet": "Lin Error"
        },
        "calibration": {
            "model_scaling_factors": {
                "8340-1": 0.4,
                "default": 24.0
            },
            "system_a_models": ["68", "78", "85"],
            "system_b_models": ["8340", "834"],
            "apply_model_specific_thresholds": True
        },
        "output": {
            "save_raw_data": True,
            "save_filtered_data": True,
            "save_gradients": False,
            "decimal_places": 6,
            "include_plots": True,
            "output_filename_pattern": "{model}_{serial}_{timestamp}",
            "timestamp_format": "%Y%m%d_%H%M%S"
        },
        "log_level": "INFO",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }

    config_path = Path("config/default_config.json")

    print("\nCreating default configuration...")
    with open(config_path, "w") as f:
        json.dump(default_config, f, indent=4)

    print(f"  ✓ Created: {config_path}")
    print("\nDefault configuration created successfully!")


def move_files_to_structure():
    """Move the core files to appropriate directories."""

    file_moves = [
        ("data_processor.py", "core/data_processor.py"),
        ("test_data_processor.py", "tests/test_data_processor.py"),
        ("config.py", "core/config.py"),
        ("example_usage.py", "examples/example_usage.py")
    ]

    print("\nOrganizing files into project structure...")

    for src, dst in file_moves:
        if os.path.exists(src) and not os.path.exists(dst):
            os.rename(src, dst)
            print(f"  ✓ Moved: {src} → {dst}")
        elif os.path.exists(dst):
            print(f"  ℹ Already exists: {dst}")
        else:
            print(f"  ⚠ Not found: {src}")


def create_init_files():
    """Create __init__.py files for Python packages."""

    init_locations = [
        "core/__init__.py",
        "tests/__init__.py",
        "examples/__init__.py"
    ]

    print("\nCreating Python package files...")

    for init_file in init_locations:
        Path(init_file).touch()
        print(f"  ✓ Created: {init_file}")


def create_sample_data():
    """Create sample data file for testing."""

    print("\nCreating sample data file...")

    # This would create a small sample Excel file
    # For now, we'll just create a placeholder
    sample_info = """Sample Data Information

Place your laser trim Excel files in this directory for testing.

Expected formats:
- System A: Files with 'SEC1 TRK1 0' and 'SEC1 TRK2 0' sheets
- System B: Files with 'test' and 'Lin Error' sheets

File naming suggestions:
- 8340_A12345_20240115.xlsx
- 6845_B67890_20240115.xlsx
"""

    with open("data/samples/README.txt", "w") as f:
        f.write(sample_info)

    print("  ✓ Created: data/samples/README.txt")


def check_python_version():
    """Check if Python version is compatible."""

    print("Checking Python version...")
    version = sys.version_info

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ⚠ Warning: Python {version.major}.{version.minor} detected.")
        print("  Recommended: Python 3.8 or higher")
        return False
    else:
        print(f"  ✓ Python {version.major}.{version.minor} - Compatible")
        return True


def main():
    """Main setup function."""

    print("=" * 60)
    print("AI-Powered Laser Trim Analysis System - Setup")
    print("=" * 60)

    # Check Python version
    check_python_version()

    # Create project structure
    create_project_structure()

    # Create configuration
    create_default_configuration()

    # Move files if they exist
    move_files_to_structure()

    # Create init files
    create_init_files()

    # Create sample data info
    create_sample_data()

    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Create virtual environment: python -m venv venv")
    print("2. Activate it: source venv/bin/activate (or venv\\Scripts\\activate on Windows)")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Place your Excel files in data/samples/")
    print("5. Run example: python examples/example_usage.py")

    print("\nFor GitHub:")
    print("1. git add .")
    print("2. git commit -m 'Initial commit: Core data processing engine'")
    print("3. git push origin main")


if __name__ == "__main__":
    main()