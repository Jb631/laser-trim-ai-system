# Claude Code Configuration for Laser Trim Analyzer V5

## Session Checklist

**Before starting work:**
1. **Set up Git credentials** - Run: `source .env 2>/dev/null && git remote set-url origin https://${GITHUB_TOKEN}@github.com/Jb631/laser-trim-ai-system.git 2>/dev/null` (silent if no token)
2. Read `BRING_TO_WORK.md` for current state and next tasks. Session notes,
   plans, and specs were purged from the tree on 2026-08-28; they remain in git
   history — recover any with `git show 4c6ebd8:docs/SESSION_2026-07-13B.md`
3. Continue from where we left off - don't start new work without checking progress
4. Explain code changes so James can learn and modify things himself

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
# Run V6 UI (daily driver)      Windows: run_v6.bat   Mac: launch_v6.command
python -m src --v6

# Run V5 classic UI (fallback)  Windows: run_v5.bat   Mac: launch_app.command
python -m src

# Install dependencies
pip install -e .
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
5. **No packaged EXE** (James, 2026-07-16): deployment stays git pull +
   pinned venv (run_v6.bat). Single user; an EXE would need IT whitelisting.
   Do not re-suggest PyInstaller packaging.

### QA Sweeps (mandatory before calling any change done)
Two standing harnesses exercise the whole app against a COPY of the real DB —
run BOTH after any change to charts, queries, exports, or page data-loaders.
The DB argument is REQUIRED and both harnesses REFUSE `data/analysis.db`
itself (they open their target read-write; three sessions opened production
by accident in one night before the refusal existed):

    cp data/analysis.db /tmp/qa_copy.db
1. `python scripts/chart_qa_render_all.py qa_output /tmp/qa_copy.db` —
   renders every v6 chart across a real-data variant matrix (dense/sparse/
   single/stale models, fail-heavy and multi-track units resolved by query).
   INSPECT the output PNGs; don't just check it ran. Both harnesses write to
   `qa_output/` (gitignored) — a sweep must never leave modified tracked
   files behind.
2. `python scripts/app_qa_sweep.py /tmp/qa_copy.db` — 190+ feature/invariant
   checks (dashboard vs raw SQL, verdict-vs-failpoint consistency, export
   schemas/row counts, pipeline on test_files, ingest guards). Exit code =
   FAIL count. Delete the copy when done.
Weak assertions are forbidden in the sweeps: a check that can pass on an
ERROR result is a bug (that exact pattern let a missing dependency read as
green once). New features get a sweep entry in the same commit.
5. **UI thread discipline** - workers NEVER call Tk; post via gui/v6/ui_dispatch
6. **Domain rule** - linearity is the zero-tolerance customer disposition;
   sigma is an internal drift-watch signal, never a rejection

### Code Style
- Type hints where practical
- Logging with `logging` module
- SQLAlchemy 2.0 syntax (`case()` not `func.case()`)

### Database
- SQLite at `./data/analysis.db`
- User settings at `./data/config.yaml` (self-contained deployment — see
  `config.py`; it is NOT under `~/.laser_trim_analyzer/`)

---

## Active Development

### V5 — Released (tag `v5.0.0`, 2026-04-16)
`pyproject.toml` is at `version = "5.0.0"`. Main has continued past the tag with bugfixes, drift-tab redesign, and Trends consolidation work.

**Current focus:** V6 is feature-complete per the 2026-08-29 app-shape spec (FOCUS · INVESTIGATE stats table · HOME/shell, all shipped) — see `BRING_TO_WORK.md`. The trim-vs-FT overlay is in V6 (2026-08-31) and Fix Missing Tracks lives in V6 Settings, so V5 no longer holds anything V6 lacks; it stays launchable as fallback until James retires it explicitly. Remaining: first real LTS3 file validation (work machine), work-machine data scripts per BRING_TO_WORK. Older plans live in git history at `4c6ebd8` under `archive/completed_docs/`.

### V4 Upgrade — Operational Analytics & Data Quality — **COMPLETE**
**Plan/Tracker:** `git show 4c6ebd8:archive/completed_docs/UPGRADE_PLAN_V4.md` (and `UPGRADE_TRACKER.md`)

V4 transformed the app from a measurement recording tool into an operational root cause identification and cost impact analysis platform. All four phases complete:
- **Phase 1:** Data Foundation (parser filtering, cleanup, indexing, validation)
- **Phase 1.5:** Dashboard & Chart Fixes (Pareto, P-chart, layout, focus panel)
- **Phase 2:** Operational Analytics (pricing, near-miss, cost dashboard, trends filters)
- **Phase 3:** Predictive Improvements (FT fuzzy matching, Cpk, ML staleness)
- **Phase 4:** Operational Integration (executive export, screening recommendations)

### Per-Model ML System — **COMPLETE**
Per-model ML is fully implemented: threshold optimization, drift detection, statistical profiling, and failure prediction using Final Test data as ground truth. Train models in Settings page.

Design docs live in git history at `4c6ebd8` under `archive/completed_docs/`.

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

Session logs through 2026-07-13 are in git history at `4c6ebd8` under
`docs/SESSION_YYYY-MM-DD.md`. The tree no longer carries them.
