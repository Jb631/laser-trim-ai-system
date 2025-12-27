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
│   ├── threshold.py     # Legacy global threshold optimizer
│   ├── drift.py         # Legacy global drift detector
│   └── (new per-model ML in development)
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

### Per-Model ML System
- **Design**: `docs/ML_REDESIGN_PLAN.md`
- **Progress**: `docs/ML_PROGRESS.md`

Goal: Replace global ML with per-model threshold optimization and drift detection using Final Test data as ground truth.

---

## Previously Fixed Issues

- Trends page stuck on "Loading" - SQLAlchemy 2.0 case() syntax (2025-12-16)
- Missing xlrd for .xls files (2025-12-26)
- Incremental processing not retrying errors (2025-12-26)
- Final Test None handling in processor (2025-12-26)
