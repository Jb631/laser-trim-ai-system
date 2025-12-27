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
1. **Final Test model parsing** - Now preserves suffixes like `8340-1` (was truncating to `8340`)
   - File: `core/final_test_parser.py:266` - Changed regex to `r'^(\d+(?:-\d+)?)'`

2. **Final Test column layout detection** - Added auto-detection for alternative file formats
   - File: `core/final_test_parser.py:288-350` - Detects if position is in col 0 vs col 4
   - **INCOMPLETE**: Needs user help to properly map all Final Test file variations

3. **Excel export anomaly detection** - Added `is_anomaly` and `anomaly_reason` to exports
   - File: `export/excel.py` - Added to Summary, Track Details, All Results sheets

4. **Final Test same-day linking** - Changed `<` to `<=` for date comparison
   - File: `database/manager.py:1802,1817`

5. **Processor crash fix** - Handle Final Test files with no track data gracefully
   - File: `core/processor.py:248-256`

6. **Added psutil dependency** - For memory monitoring on low-RAM systems
   - File: `pyproject.toml`

### Known Issues - TO FIX NEXT SESSION

1. **Final Test Column Mapping** (PRIORITY)
   - ~228 Final Test files (33%) have no track data due to column layout variations
   - Current auto-detection helps but doesn't cover all formats
   - Need user to review sample files and help map columns correctly
   - Sample problematic file: `test_files/Final Test files/2475/2475-sn0001_8-24-2015_11-23 AM.xls`

2. **Low Pass Rate (13.4%)**
   - 3940 Fail (56.6%), 1655 Warning (23.8%), 930 Pass (13.4%)
   - Linearity pass rate: 39%, Sigma pass rate: 64%
   - May improve after re-processing with fixed Final Test parsing

3. **Database needs re-processing**
   - After fixing Final Test model parsing, need to re-process files to:
     - Update model names (8340 -> 8340-1)
     - Extract track data from previously empty files
     - Re-link Final Test to Trim files

### Database Stats (Current)
- Analysis records: 6,959
- Track records: 6,567
- Final Test records: 696 (468 with tracks, 228 without)
- Distinct models: 336 (trim), 220 (final test)

---

## Previously Fixed Issues

- Trends page stuck on "Loading" - SQLAlchemy 2.0 case() syntax (2025-12-16)
- Missing xlrd for .xls files (2025-12-26)
- Incremental processing not retrying errors (2025-12-26)
- Final Test None handling in processor (2025-12-26)
- Final Test model suffix parsing (2025-12-27)
- Excel export missing anomaly data (2025-12-27)
- Same-day Final Test linking (2025-12-27)
- Processor crash on empty Final Test tracks (2025-12-27)
