"""
Create Launcher Files for Laser Trim AI System

This script creates all the launcher files in your project directory.
Run this once to set up the one-click launch system.
"""

import os
from pathlib import Path
import stat


def create_launchers():
    """Create all launcher files."""

    # Get the launch_app.py content from above
    launch_app_content = '''[Copy the launch_app.py content from the first artifact above]'''

    # Batch file content
    batch_content = '''@echo off
:: Laser Trim AI System - Windows Launcher
:: This batch file launches the Python launcher with proper error handling

setlocal enabledelayedexpansion

:: Set window title
title Laser Trim AI System Launcher

:: Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ========================================
echo  Laser Trim AI System - Launcher
echo ========================================
echo.

:: Check if Python is installed
echo Checking for Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    goto :error
)

:: Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

:: Check if launcher exists
if not exist "launch_app.py" (
    echo ERROR: launch_app.py not found
    echo Please ensure you're running this from the correct directory
    goto :error
)

:: Check if it's the first run (no venv yet)
if not exist "venv" (
    echo.
    echo First time setup detected...
    echo This may take a few minutes.
    echo.
)

:: Launch the Python launcher
echo.
echo Starting launcher...
python launch_app.py

:: Check if launch was successful
if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch application
    echo Check launcher.log for details
    goto :error
)

:: Success - close after 3 seconds
timeout /t 3 /nobreak >nul
exit /b 0

:error
echo.
echo Press any key to exit...
pause >nul
exit /b 1
'''

    # Unix shell script content
    shell_content = '''#!/bin/bash
# Laser Trim AI System - Unix/Linux/Mac Launcher

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo " Laser Trim AI System - Launcher"
echo "========================================"
echo

# Check if Python is installed
echo "Checking for Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo
    echo "Please install Python 3.8 or higher"
    echo "Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
    echo "macOS: brew install python3"
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Check if launcher exists
if [ ! -f "launch_app.py" ]; then
    echo "ERROR: launch_app.py not found"
    echo "Please ensure you're running this from the correct directory"
    exit 1
fi

# Check if it's the first run (no venv yet)
if [ ! -d "venv" ]; then
    echo
    echo "First time setup detected..."
    echo "This may take a few minutes."
    echo
fi

# Launch the Python launcher
echo
echo "Starting launcher..."
python3 launch_app.py

# Check if launch was successful
if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Failed to launch application"
    echo "Check launcher.log for details"
    exit 1
fi
'''

    # Create files
    print("Creating launcher files...")

    # Create launch_app.py
    with open("launch_app.py", "w", encoding="utf-8") as f:
        f.write(launch_app_content)
    print("✓ Created launch_app.py")

    # Create batch file for Windows
    with open("launch_laser_trim.bat", "w", encoding="utf-8") as f:
        f.write(batch_content)
    print("✓ Created launch_laser_trim.bat")

    # Create shell script for Unix/Linux/Mac
    with open("launch_laser_trim.sh", "w", encoding="utf-8") as f:
        f.write(shell_content)

    # Make shell script executable
    try:
        os.chmod("launch_laser_trim.sh", os.stat("launch_laser_trim.sh").st_mode | stat.S_IEXEC)
        print("✓ Created launch_laser_trim.sh (executable)")
    except:
        print("✓ Created launch_laser_trim.sh")

    # Create a simple README for the launchers
    readme_content = '''# Laser Trim AI System - Launcher Instructions

## Windows Users

### Option 1: Double-click Method
1. Double-click `launch_laser_trim.bat`
2. The launcher will start automatically

### Option 2: Python Method
1. Double-click `launch_app.py`
2. The launcher GUI will appear

### Option 3: PowerShell Method (Advanced)
1. Right-click `launch_laser_trim.ps1`
2. Select "Run with PowerShell"

## Mac/Linux Users

### Option 1: Terminal Method
1. Open Terminal in this directory
2. Run: `./launch_laser_trim.sh`

### Option 2: Python Method
1. Open Terminal in this directory
2. Run: `python3 launch_app.py`

## First Time Setup

The launcher will automatically:
- Check Python version (3.8+ required)
- Create virtual environment
- Install all dependencies
- Verify components
- Launch the application

This process takes 2-5 minutes on first run.

## Troubleshooting

- Check `launcher.log` for detailed error messages
- Ensure you have Python 3.8 or higher installed
- Windows: Run as Administrator if you get permission errors
- Mac/Linux: Use `chmod +x launch_laser_trim.sh` if not executable

## Requirements

- Python 3.8 or higher
- 500MB free disk space
- Internet connection (first run only)
'''

    with open("LAUNCHER_README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("✓ Created LAUNCHER_README.md")

    print("\n✅ All launcher files created successfully!")
    print("\nTo launch the application:")
    print("  Windows: Double-click launch_laser_trim.bat")
    print("  Mac/Linux: Run ./launch_laser_trim.sh")
    print("\nSee LAUNCHER_README.md for more details.")


if __name__ == "__main__":
    create_launchers()