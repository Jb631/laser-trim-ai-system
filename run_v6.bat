@echo off
REM Laser Trim Analyzer V6 — Windows launcher (work machine).
REM No hardcoded paths: runs from wherever this folder lives.
REM First run: creates .venv and installs dependencies. Needs Python 3.12 or newer on PATH:
REM the pinned numpy 2.5.2 and scipy 1.18.1 require 3.12. Tested 2026-09-26: the whole test
REM gate on 3.14 with this pinned set; the code on 3.11 with 3.11's own libraries: every module
REM imports, and the gate's only reds were a last-bit float from 3.11's older numpy/scipy and
REM a test that fakes Windows paths on a Mac. 3.12 and 3.13 were not run. Needs one-time
REM internet/proxy access for pip.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating virtual environment...
    python -m venv .venv || goto :err
    echo Installing dependencies ^(one time, a few minutes^)...
    .venv\Scripts\python -m pip install --upgrade pip
    REM Pinned = the exact library versions proven on the home machine.
    REM (Unpinned resolution broke work on 2026-07-10: newer pydantic.)
    .venv\Scripts\python -m pip install -r requirements-pinned.txt || goto :err
    .venv\Scripts\python -m pip install -e . --no-deps || goto :err
)

echo Starting Laser Trim Analyzer V6...
.venv\Scripts\python -m src --v6
if errorlevel 1 goto :err
exit /b 0

:err
echo.
echo Something went wrong — read the message above. The app also logs to
echo data\laser_trim.log inside this folder.
pause
