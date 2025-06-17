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

- **ML Components**: ML features are optional. Always handle cases where ML is not available.
- **GUI Framework**: Using customtkinter (ctk) for the UI
- **Database**: SQLAlchemy-based with SQLite backend
- **Dependencies**: All defined in pyproject.toml, installed via `pip install -e .`

### Commands to Remember

- **Run Application**: `python src/__main__.py` or `python run_dev.py`
- **Install Dependencies**: `pip install -e .`
- **Run Tests**: `pytest tests/`
- **Check Linting**: `ruff check .` (if available)

### Current Known Issues

- ML Tools page requires MetricCard widget (now implemented)
- Some ML features are optional and may not initialize without proper setup
- Drag-and-drop functionality depends on tkinterdnd2

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
2. Use TodoWrite to plan the session tasks
3. Fix existing code - do NOT create new test files
4. Use parallel tool operations when possible
5. Document ALL changes in CHANGELOG.md immediately after implementation