# Changelog

All notable changes to the Laser Trim Analyzer v2 project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Known Issues
- ML Tools page requires proper ML model initialization
- Some ML features are optional and may not initialize without proper setup
- Drag-and-drop functionality depends on tkinterdnd2 availability

## [2025-06-15]

### Fixed
- **Home page "Today's Performance" and "Recent Activity" not updating after analysis**
  - Root cause: Timezone mismatch between database storage (UTC) and queries (local time)
  - Updated `HomePage._get_today_stats()` to query using UTC time
  - Updated `HomePage._get_trend_data()` to use UTC for 7-day trend calculation
  - Updated `HomePage._get_recent_activity()` to use UTC for activity queries
  - Fixed `DatabaseManager.get_historical_data()` to use UTC with `days_back` parameter
  - Added UTC to local time conversion for user-friendly timestamp display
  - Added debug logging to help diagnose future issues

- **Timezone comparison error: "can't compare offset-naive and offset-aware datetimes"**
  - Root cause: Mixing timezone-aware and naive datetime objects
  - Changed from `datetime.now(timezone.utc)` to `datetime.utcnow()` to use naive UTC times
  - This matches the database storage format which uses naive UTC timestamps
  - Fixed timestamp display conversion using pytz for proper timezone handling

- **Excel export error: "At least one sheet must be visible"**
  - Root cause: Excel writer failing when sheet creation encounters errors
  - Added exception handling around sheet data collection in `report_generator.py`
  - Ensured Summary sheet is always written first, even if empty
  - Added fallback data for empty sheets to prevent Excel validation errors
  - Wrapped sheet creation in try-except blocks to handle partial failures gracefully

## [2025-06-14]

### Fixed
- **ML model persistence** - Models now save to correct location and persist between app restarts
- **Final Test Compare page blank display** - Fixed import errors preventing page from loading

### Enhanced
- GUI components and page layouts for improved user experience
- Plot viewer functionality
- Configuration management
- Excel export documentation

## Previous Sessions

### Implemented Core Features
- Complete GUI using customtkinter (ctk)
- SQLAlchemy-based database with SQLite backend
- Multi-track analysis support
- Batch processing capabilities
- Historical data viewing and filtering
- ML integration for failure prediction
- Excel export functionality
- Real-time analysis with progress tracking
- Dark mode support
- Responsive layouts

### Architecture Decisions
- Event-driven architecture for page updates
- Separation of concerns with interfaces and implementations
- Comprehensive error handling and logging
- Thread-safe database operations
- Memory-efficient processing for large files

### Testing Infrastructure
- Unit tests for core functionality
- Integration tests for workflows
- Performance validation tests
- UI integration tests