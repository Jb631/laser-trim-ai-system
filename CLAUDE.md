# Claude Code Configuration for Laser Trim Analyzer V5

## Session Checklist

**Before starting work:**
1. **Set up Git credentials** - Run: `source .env 2>/dev/null && git remote set-url origin https://${GITHUB_TOKEN}@github.com/Jb631/laser-trim-ai-system.git 2>/dev/null` (silent if no token)
2. Read `TRACKER.md` FIRST — what is open, in what order, and whose move it
   is. Update it in the same commit as any work that changes an item; a
   finished item is ticked and moved to "Done recently", a new one is added
   under its workstream. Then read `BRING_TO_WORK.md` for the step-by-step
   instructions at the work machine. Session notes,
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

### The test suite is the gate (2026-09-20)

    python scripts/run_test_gate.py          # every tests/test_*.py, one file per process, ~3-5 min

**The suite was never slow — it HUNG**, at >100% CPU, which looks identical to slow. The cause was a
session-scoped `tk_root` fixture leaving two or three live CTk roots in one process, so `update()`
never returned. Fixed 2026-09-20 (`7d8a91e`); `tk_root` is function-scoped and must stay that way.
The whole suite is now ~1,800 tests in about three minutes, so **run all of it** — a hand-picked file
list is what let two broken tests reach `main` on 2026-09-20. `run_test_gate.py` runs one file per
process with a hard time limit (a hang is killed and named), reads counts from junit, and treats a
file that collected nothing as a FAILURE.

Two more traps this repo has fallen into, both now fixed but worth knowing:
* `pytest -q` used to print NO pass count at all (`addopts` already carried `-q`, so `-q` again meant
  verbosity -2). Read counts from the junit `<testsuite>` element, never from a piped tail.
* `tests/test_parse_all_models.py` (645 real files) compared only whether each file parsed, not the
  numbers, while two documents described it as a full diff. It compares every frozen value now; if it
  fails, refresh only the entry that moved and say why in the commit.

### Customer data never reaches a commit

`Work Files/` holds the owner's real backlog export: customer names, PO numbers, unit prices. None of
it may appear in code, a test, a fixture or a commit message — and "just as an example" is how it got
in on 2026-09-20 (Claude used a real model/price pair in a brief; an implementer transcribed it).
Example data is INVENTED. Before any push:

    python scripts/check_no_customer_values.py            # origin/main..HEAD; prints WHERE, never the value

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
7. **Final-test grading** (James, 2026-09-13) - the disposition is the APP's
   corrected per-point grade, restricted to the cells the sheet grades (the
   column-I flag block, else the "# of elements to ignore" counts). The
   sheet's own PASSED/FAILED is stored as a reference only and must never
   overwrite it — "i dont want to just copy the excel, i want to grade the
   error and correct the offset but if cells should be ignored then we should
   ignore those cells." Blank cells are ungraded, never 0.0. One grading body,
   `core/ft_regrade.grade_ft_track`, shared by the processor and the re-grade
   repair pass.

### What must never be stored (hard-won, 2026-09-20)
- **A blank measurement is ungraded, never 0.0** — dead centre of the band is the most flattering
  value a zero-tolerance metric can hold.
- **A file with no cut is not a trimmed unit.** Laser 1 writes a `Lin Error` sheet even when no cut
  was made (the blank template: measured == theory). 1,182 tracks were stored as flawless linearity
  PASSes that way (1,180 laser 1, 2 laser 2, none laser 3). They take the UNTRIMMED path, and carry
  no laser pass.
- **A record that failed processing is not a measurement.** ERROR rows carry the analyser's `999.999`
  marker; `core/model_stats.failed_processing_statuses()` is the ONE definition — filter with it
  before averaging anything.
- **A failure must never look like a result.** An analyzer that crashes is named in `facts["errors"]`;
  a loader that fails is named in a banner, never rendered as "no data"; a long run that half-worked
  says how many models failed.
- **A pass rate is a verdict against a TEST.** 17 (model, track, laser) groups are graded against two
  or more limit tables inside one year, so never compare pass rates across a table change — the
  `limit_tables` analyzer reports them, and `ink_target` holds the table (and the station's
  final-resistance window) constant.

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

**Current focus:** V6 is feature-complete per the 2026-08-29 app-shape spec (FOCUS · INVESTIGATE stats table · HOME/shell, all shipped) — see `BRING_TO_WORK.md`. The trim-vs-FT overlay is in V6 (2026-08-31) and Fix Missing Tracks lives in V6 Settings, so V5 no longer holds anything V6 lacks; it stays launchable as fallback until James retires it explicitly. Remaining: work-machine data scripts per BRING_TO_WORK. (LTS3 validation is DONE — 547 real System C files processed through 2026-08-19 with 0 errors; verified 2026-09-18.) Older plans live in git history at `4c6ebd8` under `archive/completed_docs/`.

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
**Laser numbering (James, 2026-09-20):** the shop's numbers do NOT follow the code's
letters. **Laser 1 = `LTS` = System B. Laser 2 = `DLTS` = System A. Laser 3 = `LTS3` =
System C.** Laser 2 (DLTS) is the one whose files carry per-position cut data. When
writing to James, say "laser 2 (DLTS)" — never translate A/B/C to 1/2/3.

**And laser 3 writes laser 2's sheets, not laser 1's** (James, 2026-09-20: "laser 2 & 3
use the same style sheet not 1 & 3"). Several comments in this repo claimed the opposite.
The data settles it: all 547 LTS3 tracks carry `track_id = 'TRK1'`, which only the System A
reader produces, and LTS3 filenames follow the DLTS convention (`..._TEST DATA_...`), not
the LTS one (`..._TA_Test Data_...`). `System C` is only an IDENTITY label applied from the
folder name — `parse_file` passes the detected FORMAT to `_extract_tracks`, so an LTS3 file
is read by the System A reader. Never pass `SystemType.C` to an extractor: the dispatch
sends everything that is not A to the System B reader, and an A-format file read as B does
not raise, it produces wrong numbers. Pinned by `tests/test_laser3_is_read_like_laser2.py`.
**Cut length is TWO different quantities** (2026-09-20, James asked what "4100" was).
The `Trim Parameters` sheet labels it `Laser Cut Length (mm)` on laser 2 (DLTS) — real
millimetres, 0.55 / 0.75 / 0.88 — and `Laser Cut Length` with **no unit** on laser 1 (LTS),
where the values are raw machine counts in the thousands (2,950 / 4,000 / 4,100). Never pool
them and never call laser 1's number "longer" or "shorter": nothing in this codebase has
established its scale. `Trim Volts` is a SEPARATE field on the same sheet — the cut setting
is not a voltage. Laser 1 also carries `High Cut Length` / `Low Cut Length`, a different pair
again (constant at 2,950 on 8232-1 while `Laser Cut Length` moved 4100 → 4000), so name the
field you mean.

**Data note:** Same serial number can appear multiple times — this is VALID (unit trimmed multiple times). Do not treat as duplicates.
**Linearity spec:** Zero-tolerance — every single measurement point must be in-spec. This is a customer requirement, not configurable.

---

## Session Notes

Session logs through 2026-07-13 are in git history at `4c6ebd8` under
`docs/SESSION_YYYY-MM-DD.md`. The tree no longer carries them.
