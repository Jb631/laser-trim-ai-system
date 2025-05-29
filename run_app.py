#!/usr/bin/env python
"""
Launcher for Laser Trim AI System

This script ensures all dependencies are available and launches the GUI application.
"""

import sys
import os
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'pandas',
        'numpy',
        'scipy',
        'openpyxl',
        'xlrd',
        'tkinterdnd2',
        'scikit-learn'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")

        response = input("\nWould you like to install missing packages? (y/n): ")
        if response.lower() == 'y':
            for pkg in missing_packages:
                print(f"Installing {pkg}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            print("\nPackages installed successfully!")
        else:
            print("\nPlease install missing packages manually:")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)


def create_directories():
    """Create required directories"""
    directories = [
        'output',
        'output/ml_models',
        'logs',
        'config',
        'data'
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    print("Required directories created.")


def main():
    """Main launcher function"""
    print("=" * 60)
    print("LASER TRIM AI SYSTEM")
    print("=" * 60)
    print()

    # Check Python version
    if sys.version_info < (3, 8):
        print(f"Error: Python {sys.version_info.major}.{sys.version_info.minor} detected.")
        print("This application requires Python 3.8 or higher.")
        sys.exit(1)

    print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected ✓")
    print()

    # Check dependencies
    print("Checking dependencies...")
    check_dependencies()
    print()

    # Create directories
    print("Setting up directories...")
    create_directories()
    print()

    # Launch the application
    print("Launching Laser Trim AI System...")
    print("-" * 60)

    try:
        # Import and run the GUI application
        from gui.gui_application import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"\nError importing GUI module: {e}")
        print("\nTrying alternative import...")

        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))

        try:
            from gui.gui_application import main as gui_main
            gui_main()
        except ImportError as e2:
            print(f"\nFailed to import GUI application: {e2}")
            print("\nPlease ensure the following file structure:")
            print("  laser-trim-ai-system/")
            print("  ├── run_app.py (this file)")
            print("  ├── gui/")
            print("  │   └── gui_application.py")
            print("  ├── core/")
            print("  │   ├── __init__.py")
            print("  │   ├── data_processor.py")
            print("  │   ├── data_processor_adapter.py")
            print("  │   └── config.py")
            print("  ├── ml_models/")
            print("  │   ├── __init__.py")
            print("  │   ├── ml_models.py")
            print("  │   └── ml_analyzer_adapter.py")
            print("  └── excel_reporter/")
            print("      ├── __init__.py")
            print("      ├── excel_reporter.py")
            print("      └── excel_report_adapter.py")
            sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()