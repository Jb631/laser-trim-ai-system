@echo off
REM Deployment script for Laser Trim Analyzer
REM Creates versioned deployment package

echo ================================
echo  Laser Trim Analyzer Deployment
echo ================================

REM Activate virtual environment
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found
)

REM Run the deployment script
echo Running deployment script...
python scripts\deploy.py

REM Check if deployment was successful
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================
    echo  🎉 Deployment Successful!
    echo ================================
    echo.
    echo Package is ready for distribution.
    echo Check the LaserTrimAnalyzer-v* folder.
    echo.
) else (
    echo.
    echo ================================
    echo  ❌ Deployment Failed
    echo ================================
    echo.
    echo Check the error messages above.
    echo.
)

pause