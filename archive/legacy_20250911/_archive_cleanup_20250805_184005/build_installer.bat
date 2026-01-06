@echo off
REM Build script for Laser Trim Analyzer Windows Installer
REM Requires: Python, PyInstaller, Inno Setup

echo ========================================
echo Building Laser Trim Analyzer v2.0.0
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo ERROR: Virtual environment not found!
    echo Please create a virtual environment first.
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install/upgrade PyInstaller
echo Installing PyInstaller...
pip install --upgrade pyinstaller

REM Clean previous builds
echo Cleaning previous builds...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

REM Create assets directory if it doesn't exist
if not exist "assets" mkdir "assets"

REM Check for app icon
if not exist "assets\app_icon.ico" (
    echo WARNING: App icon not found at assets\app_icon.ico
    echo Please add an icon file or the build will use default icon
    pause
)

REM Build the executable
echo Building executable with PyInstaller...
pyinstaller laser_trim_analyzer.spec --clean

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PyInstaller build failed!
    exit /b 1
)

echo.
echo PyInstaller build completed successfully!
echo.

REM Check if Inno Setup is installed
where /q iscc
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Inno Setup compiler (iscc) not found in PATH!
    echo Please install Inno Setup from: https://jrsoftware.org/isdownload.php
    echo Or add Inno Setup to your PATH
    echo.
    echo Typical installation paths:
    echo - C:\Program Files (x86)\Inno Setup 6\
    echo - C:\Program Files\Inno Setup 6\
    echo.
    echo Skipping installer creation...
) else (
    echo Building installer with Inno Setup...
    iscc installer.iss
    
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Inno Setup build failed!
        exit /b 1
    )
    
    echo.
    echo Installer created successfully!
    echo Check the 'dist' folder for the installer.
)

echo.
echo ========================================
echo Build completed!
echo ========================================
echo.
echo Next steps:
echo 1. Test the executable: dist\LaserTrimAnalyzer\LaserTrimAnalyzer.exe
echo 2. Test the installer: dist\LaserTrimAnalyzer_Setup_2.0.0.exe
echo 3. Deploy to your network for multi-user access
echo.
pause