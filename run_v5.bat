@echo off
REM Laser Trim Analyzer V5 (classic UI) — Windows launcher.
REM Same self-installing venv as run_v6.bat; launches WITHOUT --v6.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating virtual environment...
    python -m venv .venv || goto :err
    echo Installing dependencies ^(one time, a few minutes^)...
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -e . || goto :err
)

echo Starting Laser Trim Analyzer V5 (classic UI)...
.venv\Scripts\python -m src
if errorlevel 1 goto :err
exit /b 0

:err
echo.
echo Something went wrong — read the message above. The app also logs to
echo data\laser_trim.log inside this folder.
pause
