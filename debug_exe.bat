@echo off
echo Testing laser-trim-analyzer-gui.exe...
echo.

REM Check if exe exists
if exist "laser-trim-analyzer-gui.exe" (
    echo Found laser-trim-analyzer-gui.exe
) else (
    echo ERROR: laser-trim-analyzer-gui.exe not found!
    echo Please build the executable first.
    pause
    exit /b 1
)

echo.
echo Running with console output...
laser-trim-analyzer-gui.exe

echo.
echo Exit code: %ERRORLEVEL%
pause