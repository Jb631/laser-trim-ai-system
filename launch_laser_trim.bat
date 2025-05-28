@echo off
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