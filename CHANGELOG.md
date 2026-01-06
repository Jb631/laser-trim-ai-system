# Changelog - Laser Trim Analyzer V3

All notable changes documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [3.0.2] - 2025-12-26

### Fixed
- **xlrd dependency**: Added `xlrd>=2.0.1` for .xls (Excel 97-2003) file support
- **Incremental processing**: Now retries files that previously failed (checks `success == True`)
- **Final Test processing**: Handle None values for `file_date` and `linearity_pass`

### Added
- **ML Redesign Plan**: `docs/ML_REDESIGN_PLAN.md` - Design for per-model ML system
- **Progress Tracker**: `docs/ML_PROGRESS.md` - Track ML implementation progress

---

## [3.0.1] - 2025-12-22

### Added
- **Compare Page**: New page for Final Test vs Trim data comparison
- **Final Test Parser**: Parse post-assembly test files
- **Final Test Database**: Store and link Final Test results to trim data

### Fixed
- **Analyze Page**: Chart labels, SN parsing, delete feature
- **Final Test Processor**: SystemType validation

---

## [3.0.0] - 2025-12-16

### Changed
- **Complete V3 Rewrite**: Simplified from ~110 files to ~30 files
- **6 Pages**: Dashboard, Process, Analyze, Compare, Trends, Settings
- **Single Chart Widget**: Unified charting with multiple plot types
- **Excel-Only Export**: Removed CSV/HTML complexity
- **Self-Contained Config**: No external YAML files required

### Fixed
- **Trends Page**: SQLAlchemy 2.0 `case()` syntax (was stuck on "Loading")

---

## V2 Archive

V2 code and changelog archived in `archive/` directory. V2 is no longer maintained.
