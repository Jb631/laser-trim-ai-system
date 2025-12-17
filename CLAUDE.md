# Claude Code Configuration for Laser Trim Analyzer V3

## Project Overview

Laser Trim Analyzer v3 - A production-ready quality analysis platform for potentiometer laser trim data.

### V3 Architecture
- **~30 files** (down from 110 in v2)
- **5 pages**: Dashboard, Process, Analyze, Trends, Settings
- **1 chart widget** with multiple plot types
- **Self-contained configuration**
- **SQLite database** with SQLAlchemy
- **Excel-only export**

### Source Code Location
- **Main code**: `src/laser_trim_analyzer/`
- **Entry point**: `src/laser_trim_analyzer/__main__.py`
- **Archived V2**: `archive/`

---

## Commands

### Run Application
```bash
python src/__main__.py
```

### Build Deployment Package
```bash
deploy.bat
```

### Install Dependencies
```bash
pip install -e .
```

---

## Project Structure

```
src/laser_trim_analyzer/
├── __main__.py          # Entry point
├── app.py               # Main application class
├── config.py            # Self-contained configuration
├── core/
│   ├── parser.py        # Excel file parser
│   ├── processor.py     # Analysis processor
│   ├── analyzer.py      # Sigma/linearity analysis
│   └── models.py        # Data models
├── database/
│   ├── manager.py       # Database operations
│   └── models.py        # SQLAlchemy models
├── gui/
│   ├── app.py           # GUI application
│   ├── pages/           # Page components
│   │   ├── dashboard.py
│   │   ├── process.py
│   │   ├── analyze.py
│   │   ├── trends.py
│   │   └── settings.py
│   └── widgets/
│       └── chart.py     # Matplotlib chart widget
├── ml/
│   ├── threshold.py     # Threshold optimizer
│   └── drift.py         # Drift detector
└── export/
    └── excel.py         # Excel export
```

---

## Development Guidelines

### Core Principles
1. **Fix existing code** - Don't create new test files
2. **All features must work** - No optional features
3. **Self-contained deployment** - No external config files
4. **Excel-only export** - No CSV/HTML complexity

### Code Style
- Type hints where used
- Consistent logging with `logging` module
- SQLAlchemy 2.0 syntax (use `case()` not `func.case()`)

### Database
- SQLite with SQLAlchemy ORM
- Path: `./data/analysis.db` (relative to app)
- User settings: `~/.laser_trim_analyzer/config.yaml`

---

## Known Issues

Track issues here as they're discovered.

### Current Issues
(None tracked)

### Previously Fixed
- Trends page "Loading" stuck - SQLAlchemy 2.0 case() syntax (2025-12-16)

---

## Change Tracking

All changes should be documented in `CHANGELOG.md`.

---

## Archived Code

V2 code and documentation is archived in `archive/`:
- `archive/laser_trim_v2/` - V2 source code
- `archive/v2_docs/` - V2 documentation
- `archive/v2_tests/` - V2 test suite
- `archive/v2_config/` - V2 YAML config files

V2 is preserved for reference but no longer maintained.
