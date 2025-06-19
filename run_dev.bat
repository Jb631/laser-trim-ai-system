@echo off
REM Run Laser Trim Analyzer in Development Mode

echo Starting Laser Trim Analyzer in Development Mode...
echo.

REM Set development environment
set LTA_ENV=development

REM Check if virtual environment exists
if exist venv\ (
    echo Activating virtual environment...
    call venv\Scripts\activate
) else (
    echo No virtual environment found. Running with system Python...
)

REM Run the application
echo.
echo Running application with development configuration...
python src\__main__.py

REM Keep window open on error
if errorlevel 1 (
    echo.
    echo Application exited with error.
    pause
)