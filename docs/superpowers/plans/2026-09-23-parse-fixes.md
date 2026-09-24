# Parse Upgrades and Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop failed records counting as measurements, stop V5's ML apply rewriting verdicts, make every ERROR say why, capture laser 1's `TrimVolts` sheets (with a back-fill for stored files), and close three small storage faults.

**Architecture:** Each fix lands where the fault is: the analyzer and the Findings loader (1), the ML manager's bulk update (2), the processor → manager → Model page path for the ERROR reason (3), the parser's per-pass reader plus the `trim_passes` table (4), a stand-alone back-fill script (5), and config/manager hygiene (6). No fix writes to the production database from the app; the back-fill is a script James runs.

**Tech Stack:** Python 3, pandas/xlrd (parsing), SQLAlchemy 2.0 on SQLite, CustomTkinter (one cell of the Model page).

**Spec:** `docs/superpowers/specs/2026-09-23-parse-fixes-design.md` (rulings under James's `/goal` of 2026-09-23).

## Global Constraints

- **Never open `data/analysis.db` read-write.** Read-only queries use `sqlite3.connect('file:data/analysis.db?mode=ro', uri=True)`. QA harnesses run on a COPY (`cp data/analysis.db /tmp/qa_copy.db`), deleted afterwards.
- Any test or script that builds a `Processor` or `DatabaseManager` injects BOTH `laser_trim_analyzer.database.manager._db_manager` AND `laser_trim_analyzer.database._db_manager`.
- `MLManager` defaults its storage to the RELATIVE `data/ml_models`; a test always passes `ml_storage_path=tmp_path`.
- The `tk_root` fixture stays function-scoped. App tests use `make_app`; never both in one test.
- Customer data (backlog names, PO numbers, prices) never appears in code, tests, fixtures or commit messages. Example data is INVENTED. `python scripts/check_no_customer_values.py` before any push.
- UI and log text names lasers the shop's way — "Laser 1 (LTS)", "Laser 2 (DLTS)", "Laser 3 (LTS3)" — never A/B/C.
- Linearity is the zero-tolerance disposition; sigma is a drift-watch signal and never decides FAIL.
- A blank measurement is ungraded, never 0.0. A record that failed processing is not a measurement: filter with `core/model_stats.failed_processing_statuses()` (or `_FAILED_PROCESSING` for raw SQL) — never re-type the list.
- `strftime("%-d")` is forbidden (raises on Windows).
- The gate is the whole suite: `.venv/bin/python scripts/run_test_gate.py` — read its summary line. After any parser change, also run `tests/test_parse_all_models.py` first (645 real files, ~40 s) and check its skip count. Every filter, refusal and guard a task adds is made to FAIL first (mutation-checked).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Stage by explicit path; never `git add -A`; never stage `.claude/settings.local.json`.

## File structure

| file | change | responsibility |
|---|---|---|
| `src/laser_trim_analyzer/core/analyzer.py` | modify | `_create_failed_track` carries no verdict |
| `src/laser_trim_analyzer/findings/data.py` | modify | loader drops failed-processing tracks |
| `src/laser_trim_analyzer/ml/manager.py` | modify | bulk apply keeps the ingest verdict rule |
| `src/laser_trim_analyzer/core/models.py` | modify | `AnalysisResult.error_reason` |
| `src/laser_trim_analyzer/core/processor.py` | modify | fill `error_reason`; guard records its reason |
| `src/laser_trim_analyzer/database/models.py` | modify | `analysis_results.error_reason`; three `trim_passes` columns |
| `src/laser_trim_analyzer/database/manager.py` | modify | migrations; write reason and increment volts; `get_known_models`; integer columns |
| `src/laser_trim_analyzer/gui/v6/pages/model_page.py` | modify | unit rows carry the reason |
| `src/laser_trim_analyzer/gui/v6/widgets/units_tab.py` | modify | an ERROR row's linearity cell shows the reason |
| `src/laser_trim_analyzer/core/trim_passes.py` | modify | pure `read_increment_volts()` |
| `src/laser_trim_analyzer/core/parser.py` | modify | laser-1 passes carry their `TrimVolts` sheet |
| `scripts/backfill_increment_volts.py` | create | fill stored laser-1 passes from their files |
| `src/laser_trim_analyzer/config.py` | modify | the database path travels between machines |
| tests | create/modify | per task |

---

### Task 1: A record that failed processing is never a measurement

**Files:**
- Modify: `src/laser_trim_analyzer/core/analyzer.py` (`_create_failed_track`, ~line 1209)
- Modify: `src/laser_trim_analyzer/findings/data.py` (`load_model_tracks`, ~line 131)
- Test: `tests/test_findings_data.py` (append), `tests/test_failed_track_has_no_verdict.py` (create)

**Interfaces:**
- Consumes: `core/model_stats._FAILED_PROCESSING` (tuple of status NAMES: `("ERROR", "PROCESSING_FAILED")`).
- Produces: `TrackData` from `_create_failed_track` has `linearity_pass is None` and `sigma_pass is None`; `load_model_tracks()` never returns a track whose stored status is a failed-processing status.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_failed_track_has_no_verdict.py`:

```python
"""A track too short to grade is an ERROR, and an ERROR carries no verdict.

2026-09-23: `_create_failed_track` stored linearity_pass=False, so four Findings analyzers
counted 94 'Insufficient data points' ERRORs as linearity FAILS -- 8856's laser-2 pass
rate read 27.7% when the graded tracks passed 49.0%.
"""
from laser_trim_analyzer.core.analyzer import Analyzer
from laser_trim_analyzer.core.models import AnalysisStatus


def _short_track(n=5):
    return {"track_id": "default", "positions": [float(i) for i in range(n)],
            "errors": [0.001] * n, "upper_limits": [0.01] * n, "lower_limits": [-0.01] * n,
            "travel_length": 1.0, "linearity_spec": 0.01}


def test_a_track_too_short_to_grade_is_an_error_with_no_verdict():
    t = Analyzer().analyze_track(_short_track())
    assert t.status == AnalysisStatus.ERROR
    assert t.anomaly_reason == "Insufficient data points"
    assert t.linearity_pass is None          # not False: nothing was graded
    assert t.sigma_pass is None
```

(If `analyze_track` needs more keys to reach the `< 10 points` check, add them from an existing analyzer test — the assertion block is the contract.)

Append to `tests/test_findings_data.py`:

```python
def test_a_failed_processing_track_is_not_a_measurement(fixture_db):
    """An ERROR track stored with linearity_pass=0 (every such row on the rebuilt work
    database) must not reach the analyzers, which count every non-None verdict."""
    from sqlalchemy import text
    from laser_trim_analyzer.findings.data import load_model_tracks
    before = load_model_tracks(fixture_db, "8232-1")
    victim = before[0].track_id
    with fixture_db.session() as s:
        s.execute(text("UPDATE track_results SET status='ERROR', linearity_pass=0 WHERE id=:i"),
                  {"i": victim})
        s.commit()
    after = load_model_tracks(fixture_db, "8232-1")
    assert victim not in {t.track_id for t in after}
    assert len(after) == len(before) - 1
```

- [ ] **Step 2: Run them and watch them fail.** `.venv/bin/python -m pytest tests/test_failed_track_has_no_verdict.py tests/test_findings_data.py -q` → the two new tests FAIL (`linearity_pass` is False; the victim is still loaded).

- [ ] **Step 3: Fix `_create_failed_track`.** In `src/laser_trim_analyzer/core/analyzer.py` change the two flags and say why:

```python
            sigma_threshold=0.001,
            # No verdict: nothing was graded. False here was counted as a linearity FAIL by
            # every consumer that reads `linearity_pass is not None` (2026-09-23: 8856 laser 2
            # read 27.7% instead of 49.0%). Same call enforce_measurement_backed_verdict makes.
            sigma_pass=None,
            optimal_offset=0.0,
            linearity_error=999.999,
            linearity_pass=None,
```

(`TrackData.linearity_pass` / `sigma_pass` must accept None — `enforce_measurement_backed_verdict` already assigns None to both; if the pydantic field rejects None at construction, make it `Optional[bool]` and note it in the commit.)

- [ ] **Step 4: Filter the loader.** In `src/laser_trim_analyzer/findings/data.py`, import the one definition and exclude failed-processing TRACKS from both queries (track status, not file status: a good track in a two-track file whose other track errored is still a measurement):

```python
from laser_trim_analyzer.core.model_stats import _FAILED_PROCESSING

# Status NAMES, as SQLAlchemy stores the enum. The one definition, never re-typed.
_FAILED_SQL = ", ".join(f"'{name}'" for name in _FAILED_PROCESSING)
```

and in `load_model_tracks`: the pass query gains `AND a.system IN ('A','B','C') AND t.status NOT IN ({_FAILED_SQL})`; the track query gains `AND t.status NOT IN ({_FAILED_SQL})`. (Build the SQL strings with an f-string around the constant; the values are code constants, not input.)

- [ ] **Step 5: Run the tests** → PASS. **Mutation-check:** remove the `NOT IN` from the track query → the findings test goes red; restore. Put `linearity_pass=False` back → the analyzer test goes red; restore.

- [ ] **Step 6: Find what else asserted the old flags.** `grep -rn "Insufficient data points\|_create_failed_track" tests/` and run those files; update any assertion of `linearity_pass is False` / `sigma_pass is False` on a failed track to `is None`, naming each in the commit.

- [ ] **Step 7: Gate** `.venv/bin/python scripts/run_test_gate.py` → GREEN.

- [ ] **Step 8: Commit** — `fix(findings): a track that failed processing carries no verdict and is never counted`.

---

### Task 2: V5's "Apply ML" keeps the ingest verdict rule

**Files:**
- Modify: `src/laser_trim_analyzer/ml/manager.py` (`apply_to_database`, the three bulk UPDATE blocks ~lines 527-670)
- Test: `tests/test_ml_apply_keeps_verdicts.py` (create)

**Interfaces:**
- Consumes: `MLManager(db_manager, ml_storage_path=...)`; attributes `trained_models`, `threshold_optimizers[model]` (`.is_calculated`, `.threshold`), `drift_detectors`, `predictors`; `apply_to_database(progress_callback=None, run_drift_detection=True)`.
- Produces: after an apply, every track keeps the ingest rule — FAIL whenever `linearity_pass` is False; ERROR, PROCESSING_FAILED and UNTRIMMED tracks untouched (status AND `sigma_pass`); a track with `linearity_pass` NULL untouched; every analysis's `overall_status` follows `processor._determine_overall_status`.

- [ ] **Step 1: Write the failing test** — `tests/test_ml_apply_keeps_verdicts.py`:

```python
"""V5 Settings -> Apply may move SIGMA, never the linearity verdict.

2026-09-23: the bulk update graded status as 'both pass -> PASS, both fail -> FAIL, else
WARNING'. On the rebuilt work database that would have turned 1,701 linearity FAILs into
WARNING, taken all 235 ERRORs out of ERROR and turned 6,973 UNTRIMMED sweeps into WARNING.
"""
from datetime import datetime
from types import SimpleNamespace


def _db(tmp_path, monkeypatch):
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    db = mgr.DatabaseManager(tmp_path / "apply.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)
    return db


def _add(db, serial, overall, tracks):
    """tracks: (status, linearity_pass, sigma_pass, sigma_gradient)."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SystemType, StatusType)
    when = datetime(2026, 5, 1)
    with db.session() as s:
        ar = DBAR(filename=f"M1-{serial}.xls", file_path=f"/f/M1/{serial}",
                  file_hash=f"M1{serial}".ljust(64, "0"), model="M1", serial=serial,
                  system=SystemType.A, file_date=when, timestamp=when,
                  overall_status=StatusType[overall], has_multi_tracks=len(tracks) > 1,
                  processing_time=0.1)
        s.add(ar); s.flush()
        for i, (st, lin, sig, grad) in enumerate(tracks):
            s.add(DBTR(analysis_id=ar.id, track_id=f"T{i+1}", status=StatusType[st],
                       linearity_pass=lin, sigma_pass=sig, sigma_gradient=grad))
        s.commit()
        return ar.id


def _apply(db, tmp_path, threshold):
    from laser_trim_analyzer.ml.manager import MLManager
    m = MLManager(db, ml_storage_path=tmp_path / "ml")
    m.trained_models = ["M1"]          # adapt to the real container type if it differs
    m.threshold_optimizers = {"M1": SimpleNamespace(is_calculated=True, threshold=threshold)}
    m.drift_detectors, m.predictors = {}, {}
    return m.apply_to_database(run_drift_detection=False)


def _state(db, aid):
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, TrackResult as DBTR
    with db.session() as s:
        a = s.get(DBAR, aid)
        tr = s.query(DBTR).filter(DBTR.analysis_id == aid).order_by(DBTR.track_id).all()
        return (a.overall_status.name,
                [(t.status.name, t.linearity_pass, t.sigma_pass) for t in tr])


def test_apply_moves_sigma_but_never_the_linearity_verdict(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    # threshold 0.5: sigma 0.1 passes, 0.9 fails
    lin_fail = _add(db, "s1", "FAIL", [("FAIL", False, True, 0.1)])
    watch = _add(db, "s2", "PASS", [("PASS", True, True, 0.9)])
    err = _add(db, "s3", "ERROR", [("ERROR", None, None, 999.999)])
    untrimmed = _add(db, "s4", "UNTRIMMED", [("UNTRIMMED", None, None, 0.1)])
    no_tracks = _add(db, "s5", "ERROR", [])
    mixed = _add(db, "s6", "WARNING", [("PASS", True, True, 0.1), ("UNTRIMMED", None, None, 0.9)])

    _apply(db, tmp_path, threshold=0.5)

    assert _state(db, lin_fail) == ("FAIL", [("FAIL", False, True)])          # never WARNING
    assert _state(db, watch) == ("WARNING", [("WARNING", True, False)])       # sigma moved
    assert _state(db, err) == ("ERROR", [("ERROR", None, None)])              # untouched
    assert _state(db, untrimmed) == ("UNTRIMMED", [("UNTRIMMED", None, None)])
    assert _state(db, no_tracks)[0] == "ERROR"                                # never PASS
    assert _state(db, mixed) == ("PASS", [("PASS", True, True), ("UNTRIMMED", None, None)])
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_ml_apply_keeps_verdicts.py -q` → FAIL (the FAIL track becomes WARNING; the no-track ERROR becomes PASS; the UNTRIMMED sweep changes). If the harness itself cannot drive `apply_to_database` (constructor or attribute shapes differ), fix the HARNESS to match the real class, never the assertions.

- [ ] **Step 3: Fix the three bulk steps** in `apply_to_database`. Build one skip list from the one definition, before the model loop:

```python
from laser_trim_analyzer.core.model_stats import failed_processing_statuses
# Never re-graded: failed records carry markers, not readings; an UNTRIMMED sweep has no
# trim verdict. Enum members, as the existing .name comparisons below expect names.
_UNGRADED = [s.name for s in failed_processing_statuses()] + [StatusType.UNTRIMMED.name]
```

1. The `sigma_pass` update gains `TrackResult.status.notin_(_UNGRADED)` in its `and_(...)`.
2. The track-status update becomes the ingest rule (`core/analyzer.py`, "Determine overall status"):

```python
session.execute(
    update(TrackResult)
    .where(TrackResult.analysis_id.in_(analysis_subquery))
    .where(TrackResult.status.notin_(_UNGRADED))
    .where(TrackResult.linearity_pass.isnot(None))     # ungraded: nothing to re-grade
    .values(status=case(
        # Linearity is zero-tolerance: a linearity failure is FAIL whatever sigma says.
        (TrackResult.linearity_pass == False, StatusType.FAIL.name),   # noqa: E712
        (TrackResult.sigma_pass == True, StatusType.PASS.name),        # noqa: E712
        else_=StatusType.WARNING.name))
)
```

3. The roll-up's final "all tracks PASS → PASS" update additionally requires that the analysis HAS a PASS track (`exists(select(TrackResult.id).where(TrackResult.analysis_id == AnalysisResult.id).where(TrackResult.status == StatusType.PASS.name))`). Without it an all-UNTRIMMED analysis, and an ERROR analysis with no tracks at all, satisfy "no ERROR/FAIL/WARNING track" and become PASS. Add a comment saying so.

- [ ] **Step 4: Run the test** → PASS. **Mutation-check each fix separately:** drop the `notin_` from the status update → red; restore. Swap the case order (sigma first) → red; restore. Drop the new `exists(PASS)` → red; restore. Record all three in the report.

- [ ] **Step 5: Gate** → GREEN.

- [ ] **Step 6: Commit** — `fix(ml): Apply moves sigma, never the linearity verdict, and leaves ERROR and UNTRIMMED alone`.

---

### Task 3: Every ERROR says why

**Files:**
- Modify: `src/laser_trim_analyzer/database/models.py` (`AnalysisResult`: `error_reason = Column(Text, nullable=True)` beside `overall_status`)
- Modify: `src/laser_trim_analyzer/core/models.py` (pydantic `AnalysisResult`: `error_reason: Optional[str] = Field(None, description="Why overall_status is ERROR")` beside `errors`)
- Modify: `src/laser_trim_analyzer/core/processor.py` (`process_file` after `_determine_overall_status`; `_create_error_result`; `enforce_measurement_backed_verdict`)
- Modify: `src/laser_trim_analyzer/database/manager.py` (start-up migration; `_map_analysis_to_db`; `_update_existing_analysis`; `_record_processed_file`)
- Modify: `src/laser_trim_analyzer/gui/v6/pages/model_page.py` (`_load_units`, `_search_units`)
- Modify: `src/laser_trim_analyzer/gui/v6/widgets/units_tab.py` (row render)
- Test: `tests/test_error_reason.py` (create)

**Interfaces:**
- Produces: `AnalysisResult.error_reason: Optional[str]` (pydantic and DB) — set for every ERROR result, None otherwise; unit dicts from `_load_units`/`_search_units` gain `"error_reason"`; `processed_files.error_message` on the row LINKED to an ERROR analysis holds the same text.

- [ ] **Step 1: Write the failing tests** — `tests/test_error_reason.py`:

```python
"""Every ERROR says why (2026-09-23: none of the 237 on the rebuilt database did)."""
from pathlib import Path

import pytest

from laser_trim_analyzer.core.models import AnalysisStatus


@pytest.fixture
def db(tmp_path, monkeypatch):
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    d = mgr.DatabaseManager(tmp_path / "reason.db")
    monkeypatch.setattr(mgr, "_db_manager", d, raising=False)
    monkeypatch.setattr(dbpkg, "_db_manager", d, raising=False)
    return d


def test_a_file_level_error_keeps_its_message():
    from laser_trim_analyzer.core.processor import Processor
    p = Processor(use_ml=False)
    r = p._create_error_result(p._create_minimal_metadata(Path("x.xls")), "No valid track data found", 0.0)
    assert r.overall_status == AnalysisStatus.ERROR
    assert r.error_reason == "No valid track data found"


def test_a_track_level_error_names_the_track_reason(monkeypatch):
    """An ERROR that comes from a track (bad limits, too few points) gets that track's
    words -- 234 of the 237 ERRORs on the rebuild are this kind."""
    from laser_trim_analyzer.core.processor import error_reason_of
    tracks = [type("T", (), {"status": AnalysisStatus.ERROR, "linearity_spec_warning": None,
                             "anomaly_reason": "Insufficient data points", "track_id": "TRK1"})(),
              type("T", (), {"status": AnalysisStatus.PASS, "linearity_spec_warning": None,
                             "anomaly_reason": None, "track_id": "TRK2"})()]
    assert error_reason_of(tracks, AnalysisStatus.ERROR) == "TRK1: Insufficient data points"
    assert error_reason_of(tracks, AnalysisStatus.PASS) is None


def test_the_reason_is_stored_and_linked(db, tmp_path):
    """On insert, on re-process, and on the processed_files row the analysis links to."""
    from sqlalchemy import text
    from laser_trim_analyzer.core.processor import Processor
    src = tmp_path / "8000_1_TEST DATA_1-2-2026_9-00 AM.xls"
    src.write_bytes(b"not a workbook")                   # invented name; unreadable on purpose
    result = Processor(use_ml=False).process_file(src)
    assert result is not None and result.overall_status == AnalysisStatus.ERROR
    aid = db.save_analysis(result)
    with db.session() as s:
        reason = s.execute(text("SELECT error_reason FROM analysis_results WHERE id=:i"), {"i": aid}).scalar()
        linked = s.execute(text("SELECT error_message FROM processed_files WHERE analysis_id=:i"), {"i": aid}).scalar()
    assert reason and linked == reason
```

(If an unreadable file is routed to "skipped" rather than an ERROR result, build the ERROR result with `_create_error_result` on a real temp file path and save that instead; the three assertions are the contract.)

Add a test for re-process: save the same ERROR result twice (the second save goes through `_update_existing_analysis`) and assert `error_reason` survives.

- [ ] **Step 2: Run** → FAIL (`error_reason` does not exist).

- [ ] **Step 3: The processor.** Add a module-level pure helper in `core/processor.py`, used by `process_file`:

```python
def error_reason_of(tracks, overall_status) -> Optional[str]:
    """Why an ERROR result is an ERROR, from the tracks that made it one. None otherwise.

    The words already exist on the track (the analyzer's linearity_spec_warning or
    anomaly_reason); this only brings them to the one place the app asks."""
    if overall_status != AnalysisStatus.ERROR:
        return None
    parts = []
    for t in tracks:
        if getattr(t, "status", None) != AnalysisStatus.ERROR:
            continue
        why = getattr(t, "linearity_spec_warning", None) or getattr(t, "anomaly_reason", None)
        if why:
            parts.append(f"{t.track_id}: {why}")
    return ("; ".join(parts)[:500]) or "ERROR with no recorded reason"
```

In `process_file`, pass `error_reason=error_reason_of(analyzed_tracks, overall_status)` to the `AnalysisResult(...)` constructor. In `_create_error_result`, pass `error_reason=error_msg[:500]`. In `enforce_measurement_backed_verdict`, when it fires, set `track.linearity_spec_warning = track.linearity_spec_warning or reason` so its reason flows the same way.

- [ ] **Step 4: The database.** In `database/models.py` add the column. In `manager.py`'s start-up migrations, add it the way `linearity_spec_warning` was added (check `PRAGMA table_info(analysis_results)`; `ALTER TABLE analysis_results ADD COLUMN error_reason TEXT` when missing). In `_map_analysis_to_db` set `error_reason=getattr(analysis, "error_reason", None)` beside `data_quality_issues`; in `_update_existing_analysis` set `existing.error_reason` the same way. In `_record_processed_file`, add `error_message` to BOTH the new-row constructor and the relink `.update({...})`: `None if success else reason`, where `save_analysis`/`save_batch` now pass `reason=getattr(analysis, "error_reason", None) or _error_reason(analysis)` — and `_write_failure_marker` keeps receiving `_error_reason(analysis)` exactly as today (which files are retried must not change; say so in a comment).

- [ ] **Step 5: The Model page.** `_load_units` and `_search_units` add one expression and one key:

```python
from sqlalchemy import func as _f
reason = _f.coalesce(DBAR.error_reason, DBTR.linearity_spec_warning, DBTR.anomaly_reason)
# ... .query(DBAR.id, DBAR.serial, DBAR.file_date, DBAR.overall_status,
#            DBTR.sigma_gradient, DBTR.final_linearity_error_shifted, reason)
# ... "error_reason": r[6]
```

(COALESCE is what explains the 234 rows stored before this column existed — no back-fill.) In `units_tab.py`, when a row's `overall_status` is ERROR and it has an `error_reason`, the *Linearity error* cell shows `"not graded: " + reason` (shortened to 60 characters with "…") in `TEXT_SECONDARY` instead of "—". Add a render test to `tests/test_error_reason.py` that builds the tab with one ERROR unit (use the existing units-tab tests' construction pattern) and finds the text.

- [ ] **Step 6: Run** → PASS. **Mutation-check:** remove the `error_reason=` from `_map_analysis_to_db` → the stored test goes red; remove it from `_update_existing_analysis` → the re-process test goes red; restore both.

- [ ] **Step 7: Sweep entry.** Add an `app_qa_sweep.py` check: every `analysis_results` row with `overall_status='ERROR'` has a non-empty `COALESCE(error_reason, <track reason>)` — make it FAIL first by pointing it at a row with none, then restore. Run the sweep on a COPY.

- [ ] **Step 8: Gate** → GREEN. **Commit** — `feat(ingest): every ERROR records why, and the Model page shows it`.

---

### Task 4: Laser 1's `TrimVolts` sheets are captured

**Files:**
- Modify: `src/laser_trim_analyzer/core/trim_passes.py` (new pure function)
- Modify: `src/laser_trim_analyzer/core/parser.py` (`_read_trim_passes`, System B passes only)
- Modify: `src/laser_trim_analyzer/database/models.py` (`TrimPass`: three columns)
- Modify: `src/laser_trim_analyzer/database/manager.py` (migration; `_write_trim_passes`)
- Test: `tests/test_increment_volts.py` (create); refresh `tests/fixtures/parse_baseline*.json` entries only if their frozen values move, naming each

**Interfaces:**
- Produces:
  - `read_increment_volts(df: pd.DataFrame, first_row: Optional[int], window: Optional[int]) -> Dict[str, Any]` returning `{"increment_volts": List[List[float]], "increment_volts_first_row": Optional[int], "increment_volts_truncated": bool}`.
  - pass dicts for laser-1 (System B format) `Trim N` passes gain those three keys; `trim_passes` rows gain columns `increment_volts` (SafeJSON), `increment_volts_first_row` (Integer), `increment_volts_truncated` (Boolean).
  - NEVER named `trim_volts`: that key already carries `Trim Parameters`' *Trim Volts* setting into `trim_voltage`.

Facts the code rests on (research 2026-09-23, 4,972 laser-1 files): `TrimVolts N` exists iff `Trim N` exists. No header row. Column *k* = the position at data row (`Initial Points Ignored` + *k*) of `Trim N`. Each row is one more laser increment; each cell the output voltage after it; row 0 non-zero in every engaged column; zeros are end padding only. Window = `Number of Readings (Lin)` − initial − ending ignored + 1 (83% exact). The .xls format caps a sheet at 256 columns: 11 sheets hit it and lose positions. The last reading is NOT `Trim N`'s measured value (live reading vs the verification sweep).

- [ ] **Step 1: Write the failing tests** — `tests/test_increment_volts.py`:

```python
"""Laser 1 (LTS) records how each position responds to each laser increment, in its
`TrimVolts N` sheets. The parser never read them (2026-09-23)."""
from pathlib import Path

import pandas as pd
import pytest

from laser_trim_analyzer.core.trim_passes import read_increment_volts

LTS = Path("tests/fixtures/trim/lts_8232-1_193.xls")


def test_zero_padding_is_dropped_and_each_position_keeps_its_readings():
    df = pd.DataFrame([[0.27, 0.40, 0.50],
                       [0.28, 0.41, 0.00],
                       [0.29, 0.00, 0.00]])
    out = read_increment_volts(df, first_row=2, window=3)
    assert out["increment_volts"] == [[0.27, 0.28, 0.29], [0.40, 0.41], [0.50]]
    assert out["increment_volts_first_row"] == 2
    assert out["increment_volts_truncated"] is False


def test_a_sheet_at_the_256_column_limit_is_marked_truncated():
    df = pd.DataFrame([[0.3] * 256])
    assert read_increment_volts(df, first_row=0, window=718)["increment_volts_truncated"] is True
    assert read_increment_volts(df, first_row=0, window=None)["increment_volts_truncated"] is True


def test_a_blank_cell_is_never_a_reading():
    df = pd.DataFrame([[0.27, float("nan")], [0.28, float("nan")]])
    assert read_increment_volts(df, first_row=0, window=2)["increment_volts"] == [[0.27, 0.28], []]


@pytest.mark.skipif(not LTS.exists(), reason="laser-1 fixture")
def test_the_real_laser_1_file_carries_its_response_curves():
    from laser_trim_analyzer.core.parser import ExcelParser
    parsed = ExcelParser().parse_file(LTS)
    passes = [p for t in parsed["tracks"] for p in t.get("trim_passes", [])
              if p["sheet"].lower().startswith("trim ")]
    assert passes, "the fixture has Trim 1 and Trim 2"
    for p in passes:
        curves = p["increment_volts"]
        assert len(curves) == 49                      # 57 readings - 2 - 7 + 1 (measured 2026-09-23)
        assert all(c and c[0] != 0.0 for c in curves)  # row 0 always read
        assert p["increment_volts_first_row"] == 2
        assert p["increment_volts_truncated"] is False
    lin = [p for t in parsed["tracks"] for p in t.get("trim_passes", [])
           if p["sheet"].lower().startswith("lin error")]
    assert all("increment_volts" not in p for p in lin)   # no TrimVolts for Lin Error
```

(The 49 / 2 figures were measured on `lts_8232-1_193.xls` `TrimVolts2`; confirm against `TrimVolts1` of the same file before trusting them for both passes, and correct the test — not the code — if pass 1 differs.)

Add a DB round-trip test: process the LTS fixture through `Processor(use_ml=False)` into a tmp DB (both globals injected) and assert the stored `trim_passes` rows for `Trim N` carry `increment_volts` equal to the parsed lists, and a DLTS fixture's rows carry NULL.

Add an alignment test: for the LTS fixture, the LAST reading of column *k* correlates with `Trim N`'s own measured-volts column at data row (`first_row` + *k*) at r > 0.999 (read the sheet with pandas in the test; this pins the mapping without claiming the two are equal).

- [ ] **Step 2: Run** → FAIL (no `read_increment_volts`).

- [ ] **Step 3: The pure reader** in `core/trim_passes.py`:

```python
XLS_MAX_COLUMNS = 256      # BIFF8 (.xls) stops at column IV


def read_increment_volts(df: pd.DataFrame, first_row: Optional[int],
                         window: Optional[int]) -> Dict[str, Any]:
    """Laser 1's `TrimVolts N` sheet: one column per engaged position, one row per laser
    increment, each cell the output voltage after it. Zeros are end padding (a position
    that converged early), never a reading; a blank is never a reading either.

    `first_row` is the position index of column 0 (the file's Initial Points Ignored);
    `window` is how many columns the file SHOULD have. A sheet at the .xls column limit,
    or shorter than its window, has lost positions and says so.
    """
    curves: List[List[float]] = []
    for col in range(df.shape[1]):
        readings: List[float] = []
        for v in df.iloc[:, col].tolist():
            if not isinstance(v, numbers.Real) or v != v or v == 0.0:
                break                 # end padding (or blank): this position's run is over
            readings.append(float(v))
        curves.append(readings)
    n = df.shape[1]
    truncated = n >= XLS_MAX_COLUMNS or (window is not None and n < window)
    return {"increment_volts": curves, "increment_volts_first_row": first_row,
            "increment_volts_truncated": bool(truncated)}
```

(`numbers` and `Optional` are already imported there; check. Stopping at the first zero is safe because the research found no zero between two readings in any of 107 passes.)

- [ ] **Step 4: The parser.** In `_read_trim_passes`, for `system_type == SystemType.B` and a sheet named `Trim N`, look up the sheet whose name matches `^trimvolts\s*N$` (case-insensitive) in `xl.sheet_names`; if present, read it with `header=None` and `entry.update(read_increment_volts(df, first_row, window))`. `first_row` and `window` come from the file's `Model Parameters` values the parser already reads for the trim setup (`Initial Points Ignored`, `Ending points Ignored`, `Number of Readings (Lin)`) — pass them into `_read_trim_passes` from the caller as optional ints rather than re-reading the sheet; when any is missing, `first_row=None`/`window=None`. A missing or malformed `TrimVolts` sheet logs at DEBUG (a touch-up file legitimately has none) and leaves the pass without the keys. Never raise: it sits inside the existing per-pass try.

- [ ] **Step 5: The database.** `TrimPass` gains `increment_volts = Column(SafeJSON, nullable=True)`, `increment_volts_first_row = Column(Integer)`, `increment_volts_truncated = Column(Boolean)`; the start-up migration adds the three columns when missing (same pattern as `track_results`'); `_write_trim_passes` writes them from the pass dict and excludes the three keys from the `recipe` JSON.

- [ ] **Step 6: Run** the new tests → PASS; then `tests/test_parse_all_models.py` and `tests/test_trim_capture_noop.py`: every value that existed before must be unchanged (new keys are allowed). If the 645-file gate reports moved VALUES, stop and investigate — this task must not change any existing value. **Mutation-check:** break the zero test (`v == 0.0` → never true) → the padding test goes red; restore.

- [ ] **Step 7: Measure the cost.** Time `tests/test_parse_all_models.py` before and after (it parses 645 real files) and put both numbers in the commit message — ingest speed is a known constraint.

- [ ] **Step 8: Sweep entry.** `app_qa_sweep.py`: on the copy, every laser-1 `Trim N` pass processed after this change carries `increment_volts` (skip rows whose analysis predates it: `increment_volts IS NULL AND created_date < <migration date>` is not a failure). Make it fail first.

- [ ] **Step 9: Gate** → GREEN. **Commit** — `feat(parser): laser 1's TrimVolts sheets -- each position's response to each laser increment`.

---

### Task 5: Back-fill the stored laser-1 passes

**Files:**
- Create: `scripts/backfill_increment_volts.py`
- Test: `tests/test_backfill_increment_volts.py`

**Interfaces:**
- Consumes: `read_increment_volts` (Task 4); the parser's pass-sheet naming; `analysis_results.file_path`; the three `trim_passes` columns.
- Produces: a CLI: `python scripts/backfill_increment_volts.py DB_PATH [--limit N] [--dry-run]`.

Behaviour (each line is a test):
1. Refuses to run without an explicit DB path; prints what it will touch and how to snapshot first (`python scripts/snapshot_db.py`).
2. Selects laser-1 analyses (`system='B'`) that have `Trim N` passes with `increment_volts IS NULL`, oldest first; opens each source file read-only; reads ONLY the `Model Parameters` values it needs and the `TrimVolts N` sheets; updates ONLY the three columns of the matching pass rows (matched by `track_result_id` + `sheet`).
3. Resumable: a second run touches nothing already filled; `--limit` bounds a run; progress every 200 files with files/second and an ETA.
4. A file that is missing or unreadable is counted and named in the summary, never fatal; nothing else in the database changes (test: hash every other column before and after).
5. `--dry-run` reads and reports without writing.
6. Commits in batches of 200 files so an interrupted run keeps what it did.

- [ ] **Step 1:** Write the tests against a tmp DB built by processing the LTS fixture with Task 4's capture DISABLED for the build (set the three columns to NULL after the save), then run the script and compare to a fresh parse. Include the "touches nothing else" hash test and the resume test.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS; mutation-check the resume guard (drop `IS NULL`) → the resume test goes red; restore.
- [ ] **Step 5:** Run it on a COPY of the work database with `--limit 50`; report files/second and the projected total for all laser-1 analyses (count them read-only). Delete the copy.
- [ ] **Step 6:** Gate → GREEN. **Commit** — `feat(scripts): back-fill laser 1's TrimVolts into stored passes, resumably`.

---

### Task 6: Three small ones

**Files:**
- Modify: `src/laser_trim_analyzer/config.py` (`Config.save`, `Config.load`)
- Modify: `src/laser_trim_analyzer/database/manager.py` (`get_known_models`, `_write_trim_setup`)
- Test: `tests/test_config_db_path_travels.py` (create), `tests/test_small_storage_fixes.py` (create)

- [ ] **Step 1: Failing tests.**

```python
# tests/test_config_db_path_travels.py
"""A data/ folder carried between Windows and the Mac must not carry a path that only
exists on one of them (2026-09-22: a junk `C:\\dev\\...\\analysis.db` appeared in the
repo root on the Mac)."""
from pathlib import Path

from laser_trim_analyzer import config as cfg


def test_a_path_inside_the_app_folder_is_saved_relative(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "get_app_directory", lambda: tmp_path)
    c = cfg.Config()
    c.database.path = tmp_path / "data" / "analysis.db"
    c.save(tmp_path / "data" / "config.yaml")
    text = (tmp_path / "data" / "config.yaml").read_text()
    assert str(tmp_path) not in text and "data/analysis.db" in text.replace("\\", "/")
    assert cfg.Config.load(tmp_path / "data" / "config.yaml").database.path == tmp_path / "data" / "analysis.db"


def test_a_windows_path_read_on_another_os_falls_back_to_the_default(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(cfg, "get_app_directory", lambda: tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "config.yaml").write_text(
        "database:\n  path: 'C:\\dev\\laser-trim-ai-system\\data\\analysis.db'\n")
    loaded = cfg.Config.load(tmp_path / "data" / "config.yaml")
    import os
    if os.name != "nt":
        assert loaded.database.path == tmp_path / "data" / "analysis.db"
        assert "another" in caplog.text.lower() or "windows" in caplog.text.lower()
```

(Also cover the mirror case on Windows — a POSIX absolute path read under `os.name == "nt"` — with the same shape, skipped off Windows. `get_app_directory` may be referenced through the dataclass default factory; patch whatever `DatabaseConfig`'s default actually calls.)

```python
# tests/test_small_storage_fixes.py
def test_unknown_is_not_a_known_model(tmp_path, monkeypatch):
    ...  # insert one analysis with model "Unknown" and one "8000"; get_known_models() == {"8000"}

def test_integer_columns_get_integers(tmp_path, monkeypatch):
    ...  # _write_trim_setup with {"initial_points_ignored": 2.0, "ending_points_ignored": 7.5}
         # -> points_ignored_start == 2 (typeof integer), points_ignored_end is None,
         #    and parameters still holds 7.5
```

(Write both bodies fully, with both globals injected; invented model names only.)

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.**
  - `Config.save`: if `database.path` is inside `get_app_directory()`, write it relative with forward slashes (`data/analysis.db`); otherwise write it as is.
  - `Config.load`: expand env vars; a relative path resolves against `get_app_directory()`; an absolute path of the OTHER family (`PureWindowsPath(p).drive` on a non-Windows OS; a path starting with `/` on Windows) logs a WARNING naming it and keeps the default.
  - `get_known_models`: drop `"Unknown"` from both sets (one named constant beside the function; the parser's sentinel).
  - `_write_trim_setup`: for columns whose type is `Integer` (`points_ignored_start`, `points_ignored_end`), store `int(v)` when `_as_float(v)` is integral, else None; `parameters` is untouched.
- [ ] **Step 4:** Run → PASS; mutation-check the fallback (make it keep the foreign path) → red; restore.
- [ ] **Step 5:** Gate → GREEN. **Commit** — `fix: the database path travels, Unknown is not a model, integer columns get integers`.

---

### Task 7: Close-out

- [ ] Run both QA sweeps on a COPY (`chart_qa_render_all.py`, `app_qa_sweep.py`); inspect; delete the copy.
- [ ] `TRACKER.md`: tick B1a (capture shipped; back-fill is James's to run), add the ERROR-reason, failed-track, ML-apply and small fixes to "Done recently" with their measured impact; close "laser 3 per-position" by measurement; record the `Response` ruling under B1c; update `BRING_TO_WORK.md` with the back-fill command (PowerShell: `.\.venv\Scripts\python scripts\backfill_increment_volts.py data\analysis.db --limit 2000`, snapshot first).
- [ ] `CLAUDE.md`: the TrimVolts paragraph gains what is now known (mapping, padding, 256-column limit, live-vs-verification) and that it is captured as `increment_volts`.
- [ ] `python scripts/check_no_customer_values.py` → 0 problems.
