# Claude Code Configuration for Laser Trim Analyzer V5

## Session Checklist

**Before starting work:**
1. **Set up Git credentials** - Run: `source .env 2>/dev/null && git remote set-url origin https://${GITHUB_TOKEN}@github.com/Jb631/laser-trim-ai-system.git 2>/dev/null` (silent if no token)
2. Read `docs/UPGRADE_TRACKER.md` - Check current phase and next pending task
3. Read the corresponding section in `docs/UPGRADE_PLAN_V4.md` for full details on the task
4. Continue from where we left off - don't start new work without checking progress
5. Explain code changes so James can learn and modify things himself

---

## Project Overview

**Laser Trim Analyzer v5** - Production quality analysis platform for potentiometer laser trim data.

### Key Features
- **10 pages**: Dashboard, Process, Analyze, Compare, Trends, Quality Health, Scorecard, Smoothness, Specs, Settings
- **Final Test support**: Parse and fuzzy-match post-assembly test files to trim data
- **Operational intelligence**: Near-miss detection, cost impact analysis, linearity prioritization
- **Cpk/Ppk** process capability per model
- **SQLite database** with SQLAlchemy 2.0 ORM
- **Excel-only export** (executive summary option)
- **Per-model ML**: threshold optimization, drift detection (CUSUM + EWMA), statistical profiling, failure prediction

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
│   ├── pages/           # Dashboard, Process, Analyze, Compare, Trends, Quality Health, Scorecard, Smoothness, Specs, Settings
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

### V5 — Released (tag `v5.0.0`, 2026-04-16)
`pyproject.toml` is at `version = "5.0.0"`. Main has continued past the tag with bugfixes, drift-tab redesign, and Trends consolidation work.

**Current focus:** `docs/BUGFIX_PLAN_V5.md` — 5-phase bugfix plan from a 6-agent code review swarm (20 issues: 5 critical, 7 high, 8 medium).

### V4 Upgrade — Operational Analytics & Data Quality — **COMPLETE**
**Plan:** `docs/UPGRADE_PLAN_V4.md`
**Tracker:** `docs/UPGRADE_TRACKER.md`

V4 transformed the app from a measurement recording tool into an operational root cause identification and cost impact analysis platform. All four phases complete:
- **Phase 1:** Data Foundation (parser filtering, cleanup, indexing, validation)
- **Phase 1.5:** Dashboard & Chart Fixes (Pareto, P-chart, layout, focus panel)
- **Phase 2:** Operational Analytics (pricing, near-miss, cost dashboard, trends filters)
- **Phase 3:** Predictive Improvements (FT fuzzy matching, Cpk, ML staleness)
- **Phase 4:** Operational Integration (executive export, screening recommendations)

### Per-Model ML System — **COMPLETE**
Per-model ML is fully implemented: threshold optimization, drift detection, statistical profiling, and failure prediction using Final Test data as ground truth. Train models in Settings page.

Design docs archived in `archive/completed_docs/`.

---

## Domain Context

**Product:** Potentiometers (variable resistors) for aerospace/defense customers
**Company:** AS9100 certified manufacturer, VC/PE owned
**Key process:** Carbon track elements are laser-trimmed to achieve linearity spec, then units go through final electrical testing
**Critical issue:** High failure rate at final linearity testing (~40% fail+warning). Most expensive place to catch defects because maximum labor/material already invested.
**Data note:** Same serial number can appear multiple times — this is VALID (unit trimmed multiple times). Do not treat as duplicates.
**Linearity spec:** Zero-tolerance — every single measurement point must be in-spec. This is a customer requirement, not configurable.

---

## Session Notes

Session logs are in `docs/SESSION_YYYY-MM-DD.md`
