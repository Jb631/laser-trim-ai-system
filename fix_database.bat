@echo off
echo =====================================
echo Database Enum Fix Utility
echo =====================================
echo.
echo This utility will fix enum values in the database.
echo.

REM First run in dry-run mode to show what will be changed
echo Checking for issues (dry run)...
python scripts\fix_database_enums.py --dry-run

echo.
echo =====================================
set /p confirm="Do you want to apply these fixes? (Y/N): "

if /i "%confirm%" == "Y" (
    echo.
    echo Applying fixes...
    python scripts\fix_database_enums.py
    echo.
    echo Done! Database has been fixed.
) else (
    echo.
    echo Operation cancelled.
)

echo.
pause