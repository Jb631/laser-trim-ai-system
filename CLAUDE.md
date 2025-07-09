# Claude Code Configuration for Laser Trim Analyzer

## Project Rules and Guidelines

### PRODUCTION READINESS FOCUS
**The entire purpose of our sessions is to make ALL current features fully functional. No new features should be added. Focus exclusively on fixing, polishing, and ensuring production readiness.**

### Core Development Principles

1. **No Quick Fixes**
   - All fixes must be fully implemented and tested
   - Never leave partial implementations or TODOs without completing them
   - If a fix requires multiple steps, complete all steps before moving on

2. **Thoughtful Decision Making**
   - Analyze the existing codebase thoroughly before making changes
   - Consider the impact of changes on other parts of the system
   - Document reasoning for significant architectural decisions

3. **Code Modification Priority**
   - ALWAYS attempt to fix existing code before creating new scripts
   - Only create new files when absolutely necessary
   - Prefer modifying existing implementations over rewriting from scratch

4. **Parallel Tool Usage**
   - Use multiple tools concurrently when possible for better performance
   - Batch related operations (e.g., multiple file reads, grep searches)
   - Run independent bash commands in parallel

5. **Feature Completeness**
   - ALL features must be fully functional - nothing is optional
   - ML components must work completely, not be treated as optional
   - Every page, widget, and feature in the app must function properly
   - If a feature exists in the codebase, it must work

### Technical Guidelines

1. **Error Handling**
   - Implement comprehensive error handling in all code changes
   - Log errors appropriately using the existing logging system
   - Ensure the application remains stable even when components fail

2. **Testing**
   - Test all changes before considering them complete
   - Run existing tests when modifying core functionality
   - Verify UI changes by checking component rendering

3. **Code Style**
   - Follow existing code patterns and conventions
   - Maintain consistent indentation and formatting
   - Use type hints where the codebase already uses them

### Specific Project Context

- **ML Components**: ML features are REQUIRED and must work completely. They are NOT optional.
- **GUI Framework**: Using customtkinter (ctk) for the UI
- **Database**: SQLAlchemy-based with SQLite backend
- **Dependencies**: All defined in pyproject.toml, installed via `pip install -e .`
- **Drag and Drop**: Uses tkinterdnd2 and is a required feature, not optional

### Commands to Remember

- **Run Application (Production)**: `python src/__main__.py`
- **Run Application (Development)**: `run_dev.bat` or `set LTA_ENV=development && python src/__main__.py`
- **Run Application (from Linux/WSL)**: `.venv/Scripts/python.exe src/__main__.py` (Windows venv)
- **Run Application with Environment**: `export LTA_ENV=development && .venv/Scripts/python.exe src/__main__.py`
- **Initialize Development Database**: `python scripts/init_dev_database.py --clean --seed-data`
- **Install Dependencies**: `pip install -e .`
- **Run Tests**: `pytest tests/`
- **Check Linting**: `ruff check .` (if available)

**Note**: The application runs in GUI mode which may timeout in automated environments. The virtual environment is located at `.venv/` with Windows-style structure (Scripts/ instead of bin/).

### Database Configuration

The application uses different databases based on environment:

1. **Production** (`config/production.yaml`):
   - Database: `D:/LaserTrimData/production.db`
   - Models: `D:/LaserTrimData/models`
   - Data: `D:/LaserTrimData/Production`

2. **Development** (`config/development.yaml`):
   - Database: `%LOCALAPPDATA%/LaserTrimAnalyzer/dev/laser_trim_dev.db`
   - Models: `%LOCALAPPDATA%/LaserTrimAnalyzer/dev/models`
   - Data: `%USERPROFILE%/Documents/LaserTrimAnalyzer/dev/data`

3. **Deployment** (`config/deployment.yaml`):
   - Single User: `%LOCALAPPDATA%/LaserTrimAnalyzer/database/laser_trim_local.db`
   - Multi User: Network path configured by IT

To switch environments, set the `LTA_ENV` environment variable:
- Windows: `set LTA_ENV=development` or `set LTA_ENV=production`
- Linux/Mac: `export LTA_ENV=development` or `export LTA_ENV=production`

### Known Issues Tracking

**IMPORTANT**: Known issues must be tracked in both CLAUDE.md and CHANGELOG.md. When fixing issues:
1. Remove them from the Known Issues section in CLAUDE.md
2. Document the fix in CHANGELOG.md with root cause analysis
3. Update the Known Issues section in CHANGELOG.md

### Current Known Issues

None - All previously known issues have been fixed:
- ✓ Range utilization percent calculation now properly capped at 100%
- ✓ Final Test Comparison page datetime parsing fixed
- ✓ Final Test Comparison page now shows trim date from file
- ✓ Multi-track page correctly shows all tracks
- ✓ Home page enum value error for empty status fixed (2025-06-19)
- ✓ Database save error handling fixed (2025-06-19)
- ✓ QA alerts constraint error fixed (2025-06-19)
- ✓ Model Summary page TypeError with None values fixed (2025-07-08)
- ✓ Historical page drift_alert_card AttributeError fixed (2025-07-08)
- ✓ Multi-track analysis 0.0% variations issue fixed (2025-07-08)

### Change Tracking

**IMPORTANT**: All changes, fixes, and enhancements MUST be documented in `CHANGELOG.md`. This includes:
- Bug fixes with root cause analysis
- Feature enhancements
- Architecture changes
- Performance improvements
- Known issues and their workarounds

Before starting any session, review the CHANGELOG.md to understand recent changes and current state.

Remember: Always think through the full implementation before starting any fix.

### Session Start Checklist
**EVERY session MUST begin with these steps:**
1. Read CHANGELOG.md to understand recent changes and current state
2. Check the Known Issues sections in both CLAUDE.md and CHANGELOG.md
3. Use TodoWrite to plan the session tasks
4. Fix existing code - do NOT create new test files
5. Use parallel tool operations when possible
6. Document ALL changes in CHANGELOG.md immediately after implementation
7. Update Known Issues sections as issues are discovered or fixed