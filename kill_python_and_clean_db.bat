@echo off
REM Kill Python and Clean Databases Script
REM This script ensures all Python processes are closed before cleaning databases

echo ========================================
echo Kill Python Processes and Clean Databases
echo ========================================
echo.

REM Kill all Python processes
echo Killing all Python processes...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul

REM Wait a moment for processes to terminate
timeout /t 2 /nobreak >nul

REM Check if any Python processes are still running
tasklist | findstr /I "python" >nul
if %errorlevel% equ 0 (
    echo.
    echo WARNING: Some Python processes may still be running!
    echo Please close the Laser Trim Analyzer application manually.
    echo.
    pause
) else (
    echo All Python processes have been terminated.
)

echo.
echo Now cleaning databases...
echo.

REM Call the existing clean script
call clean_all_databases.bat

echo.
echo ========================================
echo Process Complete
echo ========================================
echo.
echo You can now run the Laser Trim Analyzer again with a clean database.
echo.
pause