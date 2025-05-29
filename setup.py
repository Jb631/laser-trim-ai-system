"""
Setup script for AI-Powered Laser Trim Analysis System

This script helps set up the project structure and initial configuration.
Updated for Python 3.12 compatibility.

Author: QA Team
Date: 2024
Version: 1.1.0
"""

import os
import sys
import json
from pathlib import Path
import subprocess


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"⚠ Warning: Python {version.major}.{version.minor} detected.")
        print("  Minimum required: Python 3.8")
        print("  Recommended: Python 3.11 or 3.12")
        return False
    elif version.major == 3 and version.minor >= 12:
        print(f"✓ Python {version.major}.{version.minor} - Compatible (Latest)")
        print("  Note: Using Python 3.12 compatible package versions")
        return True
    else:
        print(f"✓ Python {version.major}.{version.minor} - Compatible")
        return True


def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")

    # Upgrade pip first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install from requirements.txt
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    else:
        print("✗ requirements.txt not found!")
        print("  Creating requirements.txt...")
        create_requirements_file()
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def create_requirements_file():
    """Create requirements.txt with Python 3.12 compatible versions."""
    requirements = """# Laser Trim AI System - Python 3.12 Compatible Requirements
# Core Data Processing
pandas>=2.2.0
numpy>=1.26.0
scipy>=1.12.0

# Excel File Handling
openpyxl>=3.1.0
xlrd>=2.0.0
xlsxwriter>=3.1.0

# Machine Learning
scikit-learn>=1.4.0
joblib>=1.3.0
imbalanced-learn>=0.12.0

# GUI
tkinterdnd2>=0.3.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0

# Utilities
python-dateutil>=2.8.0
pyyaml>=6.0
click>=8.1.0
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✓ Created requirements.txt with Python 3.12 compatible versions")


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

    print("\nCreating project directory structure...")

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
laser_trim_env/
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


def create_launcher_script():
    """Create a launcher script for easy startup."""
    launcher_content = '''#!/usr/bin/env python
"""
Launcher for Laser Trim AI System
"""

import sys
import os
from pathlib import Path

def main():
    """Main launcher function."""
    print("=" * 60)
    print("LASER TRIM AI SYSTEM")
    print("=" * 60)
    print()

    # Add current directory to Python path
    sys.path.insert(0, str(Path(__file__).parent))

    print("Choose an option:")
    print("1. Launch GUI Application")
    print("2. Process Single File")
    print("3. Batch Process Folder")
    print("4. Run Examples")
    print("5. Exit")

    choice = input("\\nEnter your choice (1-5): ")

    if choice == "1":
        try:
            from gui_application import main as gui_main
            gui_main()
        except ImportError:
            print("GUI module not found. Running example instead...")
            from example_usage import main as example_main
            example_main()

    elif choice == "2":
        from data_processor import DataProcessor
        processor = DataProcessor()
        file_path = input("Enter Excel file path: ")
        try:
            result = processor.process_file(file_path)
            print(f"\\nProcessing complete!")
            print(f"Sigma gradient: {result}")
        except Exception as e:
            print(f"Error: {e}")

    elif choice == "3":
        from data_processor import DataProcessor
        processor = DataProcessor()
        folder_path = input("Enter folder path: ")
        try:
            results = processor.batch_process(folder_path)
            print(f"\\nProcessed {len(results)} files")
        except Exception as e:
            print(f"Error: {e}")

    elif choice == "4":
        from example_usage import main as example_main
        example_main()

    else:
        print("Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()
'''

    with open("launch_app.py", "w") as f:
        f.write(launcher_content)

    print("\n✓ Created launch_app.py")


def create_batch_launcher():
    """Create Windows batch file for easy launching."""
    batch_content = '''@echo off
echo ========================================
echo   LASER TRIM AI SYSTEM LAUNCHER
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "laser_trim_env\\Scripts\\python.exe" (
    echo Creating virtual environment...
    python -m venv laser_trim_env
    echo Virtual environment created.
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call laser_trim_env\\Scripts\\activate.bat

REM Check if packages are installed
python -c "import pandas" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    echo.
)

REM Launch the application
echo Starting Laser Trim AI System...
echo.
python launch_app.py

pause
'''

    with open("run_lasertrim.bat", "w") as f:
        f.write(batch_content)

    print("✓ Created run_lasertrim.bat (Windows launcher)")


def main():
    """Main setup function."""
    print("=" * 60)
    print("AI-Powered Laser Trim Analysis System - Setup")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        print("\n⚠ Warning: Python version may cause compatibility issues.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return

    # Create project structure
    create_project_structure()

    # Install dependencies
    try:
        install_dependencies()
    except Exception as e:
        print(f"\n✗ Error installing dependencies: {e}")
        print("  Please install manually using: pip install -r requirements.txt")

    # Create launcher scripts
    create_launcher_script()
    create_batch_launcher()

    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Place your Excel files in data/samples/")
    print("2. Run the application:")
    print("   - Windows: Double-click 'run_lasertrim.bat'")
    print("   - Or: python launch_app.py")
    print("\nFor help, see the documentation in docs/")

    print("\nFor GitHub:")
    print("1. git add .")
    print("2. git commit -m 'Initial commit: Core data processing engine'")
    print("3. git push origin main")


if __name__ == "__main__":
    main()