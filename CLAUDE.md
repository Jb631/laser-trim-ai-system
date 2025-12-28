# Claude Code Configuration for Laser Trim Analyzer V3

## Session Checklist

**Before starting work:**
1. Read `docs/ML_PROGRESS.md` - Check current phase and pending tasks
2. Continue from where we left off - don't start new work without checking progress

---

## Project Overview

**Laser Trim Analyzer v3** - Production quality analysis platform for potentiometer laser trim data.

### Key Features
- **6 pages**: Dashboard, Process, Analyze, Compare, Trends, Settings
- **Final Test support**: Parse and compare post-assembly test files
- **SQLite database** with SQLAlchemy ORM
- **Excel-only export**
- **Per-model ML** (in development) - threshold optimization and drift detection

### Source Code
- **Main code**: `src/laser_trim_analyzer/`
- **Entry point**: `src/laser_trim_analyzer/__main__.py`

---

## Commands

```bash
# Run application
python src/__main__.py

# Install dependencies
pip install -e .

# Build deployment (Windows)
deploy.bat
```

---

## Project Structure

```
src/laser_trim_analyzer/
├── __main__.py          # Entry point
├── app.py               # Main application
├── config.py            # Configuration
├── core/
│   ├── parser.py        # Trim file parser
│   ├── final_test_parser.py  # Final Test parser
│   ├── processor.py     # Analysis processor
│   ├── analyzer.py      # Sigma/linearity analysis
│   └── models.py        # Data models
├── database/
│   ├── manager.py       # Database operations
│   └── models.py        # SQLAlchemy models
├── gui/
│   ├── app.py           # GUI application
│   ├── pages/           # Dashboard, Process, Analyze, Compare, Trends, Settings
│   └── widgets/chart.py # Chart widget
├── ml/
│   ├── predictor.py           # Per-model failure prediction
│   ├── threshold_optimizer.py # Per-model threshold optimization
│   ├── drift_detector.py      # Per-model drift detection
│   ├── profiler.py            # Per-model statistical profiling
│   └── manager.py             # ML orchestration
└── export/excel.py      # Excel export
```

---

## Development Rules

### Core Principles
1. **Fix existing code** - Don't create unnecessary new files
2. **All features must work** - No partial implementations
3. **Self-contained deployment** - No external config files
4. **Keep it simple** - Avoid over-engineering

### Code Style
- Type hints where practical
- Logging with `logging` module
- SQLAlchemy 2.0 syntax (`case()` not `func.case()`)

### Database
- SQLite at `./data/analysis.db`
- User settings at `~/.laser_trim_analyzer/config.yaml`

---

## Active Development

### Per-Model ML System - **COMPLETE**
Per-model ML is fully implemented with threshold optimization and drift detection using Final Test data as ground truth. Train models in Settings page.

Design docs archived in `archive/completed_docs/`.

---

## Current Session Status (2025-12-27)

### Fixes Applied This Session

**Final Test Parser Overhaul - Success rate improved from 67% to 95.7%**

1. **Multi-format detection and parsing** - Added support for 4 distinct file formats
   - Format 1: Standard (`Sheet1` + `Data Table`) - 845 files
   - Format 2: Rout_ prefix (`Data` + `Charts` sheets) - 8 files
   - Format 3: Multi-track (`A`, `B`, `C` sheets) - 3 files
   - Format 4: Parameters sheet format - 3 files
   - File: `core/final_test_parser.py` - Added `_parse_format3_multitrack`, `_parse_format4_parameters`

2. **Format variation detection** - Auto-detect column layouts within Format 1
   - Some files have Col E = electrical_angle (standard)
   - Some files have Col E = error duplicate, position in Col F or Col 14
   - Added detection logic and fallback to index column

3. **numpy type handling** - Fixed `isinstance(val, (int, float))` to use `np.issubdtype`
   - Was causing parsing failures for numpy.float64 values

4. **Data sorting** - Added ascending sort by electrical_angle for proper chart display
   - Some files have descending X values (81° to -81°), now sorted correctly

5. **Database manager updates** - Use `electrical_angles` for position_data storage

**Database Thread Safety Fixes**

6. **Race condition fix** - Added thread locks for SQLite write operations
   - `save_analysis` and `save_final_test` now use locks to prevent concurrent write corruption
   - Prevents segfaults and UNIQUE constraint violations during batch processing

7. **Final Test double-save prevention** - `save_analysis` now skips Final Test files
   - Final Test files are saved in processor via `save_final_test`
   - Prevents incorrect storage in AnalysisResult table

8. **IntegrityError handling** - Added proper exception handling for duplicate key errors

### Parser Stats (Current)
- Total Final Test files: 859
- Successfully parsed: 822 (95.7%)
- Zero tracks (edge cases): 37 (4.3%)

---

## Previously Fixed Issues

- **Database thread safety** - Added locks to prevent SQLite race conditions (2025-12-27)
- **Final Test parser overhaul** - 95.7% success rate, 4 format types (2025-12-27)
- Final Test model suffix parsing (2025-12-27)
- Excel export missing anomaly data (2025-12-27)
- Same-day Final Test linking (2025-12-27)
- Processor crash on empty Final Test tracks (2025-12-27)
- Final Test None handling in processor (2025-12-26)
- Incremental processing not retrying errors (2025-12-26)
- Missing xlrd for .xls files (2025-12-26)
- Trends page stuck on "Loading" - SQLAlchemy 2.0 case() syntax (2025-12-16)
