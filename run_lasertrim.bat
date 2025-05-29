@echo off
echo ========================================
echo   LASER TRIM AI SYSTEM LAUNCHER
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

REM Show Python version
echo Detected Python version:
python --version
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\python.exe" (
    echo Using existing virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo.

    echo Installing required packages...
    python -m pip install --upgrade pip
    python -m pip install pandas numpy scipy openpyxl xlrd xlsxwriter
    python -m pip install scikit-learn joblib matplotlib seaborn
    python -m pip install tkinterdnd2
    echo.
)

REM Launch the application
echo Starting Laser Trim AI System...
echo.
python run_app.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)