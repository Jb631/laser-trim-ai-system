# Spec 1 — Upstream Signals Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new persisted `untrimmed_sigma_gradient` metric to `TrackResult`, calculated in `analyze_track` by running the existing `_calculate_sigma` against the untrimmed arrays, with an idempotent startup migration so existing databases upgrade in place. Pure additive change — no behavior change to existing flows.

**Architecture:** Single feature touching four layers (DB schema, Pydantic model, analyzer, DB save-mapping) plus one new test module. The change is additive at every layer: a new nullable column, a new optional Pydantic field, a parallel `_calculate_sigma` call gated on data availability, and a single field-assignment line in the `_track_to_db` mapper. The startup migration follows the established `try-probe-except-ALTER-TABLE` idiom already used at `manager.py:340` and surrounding lines.

**Tech Stack:** Python 3.x, SQLAlchemy 2.0, pandas, numpy, pytest. SQLite database.

**Target branch:** `main` (per Spec 1 design — V6 inherits via merge-forward). Verify you are on `main` before starting Task 1.

**Spec reference:** `docs/superpowers/specs/2026-05-30-spec1-upstream-signals-capture.md`

---

## File Structure

**Files modified (no new source files):**
- `src/laser_trim_analyzer/database/models.py` — add `untrimmed_sigma_gradient` Column + Index + CheckConstraint + validator
- `src/laser_trim_analyzer/database/manager.py` — add startup migration entry; add field assignment in `_track_to_db`
- `src/laser_trim_analyzer/core/models.py` — add `untrimmed_sigma_gradient` Pydantic field on `TrackData`
- `src/laser_trim_analyzer/core/analyzer.py` — add parallel `_calculate_sigma` call in `analyze_track`

**Files created:**
- `tests/test_spec1_untrimmed_sigma.py` — all unit + integration tests for this spec

---

## Task 1: DB schema — add the column to TrackResult

**Files:**
- Modify: `src/laser_trim_analyzer/database/models.py`
- Test: `tests/test_spec1_untrimmed_sigma.py` (create)

- [ ] **Step 1: Create the test file with the schema-level test**

Create `tests/test_spec1_untrimmed_sigma.py` with this initial content:

```python
"""Spec 1 — Upstream Signals Capture.

Each test maps to one element of the spec at
docs/superpowers/specs/2026-05-30-spec1-upstream-signals-capture.md.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Task 1: schema -- TrackResult.untrimmed_sigma_gradient column exists,
# is nullable, accepts non-negative values, rejects negative.
# ---------------------------------------------------------------------------


def test_track_result_has_untrimmed_sigma_gradient_column():
    """The ORM model must declare the new column as nullable Float."""
    from laser_trim_analyzer.database.models import TrackResult

    cols = {c.name: c for c in TrackResult.__table__.columns}
    assert "untrimmed_sigma_gradient" in cols, (
        f"TrackResult is missing 'untrimmed_sigma_gradient' column; "
        f"got columns: {sorted(cols)}"
    )
    col = cols["untrimmed_sigma_gradient"]
    assert col.nullable is True, "untrimmed_sigma_gradient must be nullable"
    # Float column on SQLite -> python type is float
    assert str(col.type).upper().startswith("FLOAT"), (
        f"untrimmed_sigma_gradient should be FLOAT type, got {col.type}"
    )


def test_track_result_validates_untrimmed_sigma_gradient_non_negative():
    """The @validates hook should reject a negative value."""
    from laser_trim_analyzer.database.models import TrackResult

    tr = TrackResult.__new__(TrackResult)  # skip __init__, just exercise validator
    # SQLAlchemy validators are bound to instances; the @validates decorator
    # wires them into setattr.  Easiest way to exercise: instantiate via the
    # constructor with the bad value and assert ValueError.
    with pytest.raises(ValueError, match="untrimmed_sigma_gradient cannot be negative"):
        TrackResult(
            analysis_id=1,
            track_id="TRK1",
            untrimmed_sigma_gradient=-0.5,
        )


def test_track_result_accepts_none_for_untrimmed_sigma_gradient():
    """None means 'not measured' and must pass validation."""
    from laser_trim_analyzer.database.models import TrackResult

    tr = TrackResult(
        analysis_id=1,
        track_id="TRK1",
        untrimmed_sigma_gradient=None,
    )
    assert tr.untrimmed_sigma_gradient is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spec1_untrimmed_sigma.py -v`

Expected: 3 FAILs — `AssertionError: TrackResult is missing 'untrimmed_sigma_gradient' column` or similar. The column doesn't exist yet.

- [ ] **Step 3: Add the column + index + constraint + validator to the ORM model**

In `src/laser_trim_analyzer/database/models.py`, find the `TrackResult` class (around line 310). Locate the existing `sigma_gradient` column at line 336:

```python
sigma_gradient = Column(Float, nullable=True)
```

Immediately after it, add:

```python
untrimmed_sigma_gradient = Column(Float, nullable=True)
```

Find the existing `Index('idx_track_sigma_gradient', 'sigma_gradient')` line (around line 422 inside the `__table_args__` tuple). Immediately after it, add:

```python
Index('idx_track_untrimmed_sigma_gradient', 'untrimmed_sigma_gradient'),
```

Find the existing `CheckConstraint('sigma_gradient >= 0', name='check_sigma_gradient_positive')` line (around line 432). Immediately after it, add:

```python
CheckConstraint(
    'untrimmed_sigma_gradient >= 0 OR untrimmed_sigma_gradient IS NULL',
    name='check_untrimmed_sigma_gradient_non_negative',
),
```

Find the existing `@validates('sigma_gradient')` block (around line 450). Immediately after the `return sigma_gradient` line of that validator, add a parallel validator:

```python
@validates('untrimmed_sigma_gradient')
def validate_untrimmed_sigma_gradient(self, key, value):
    """Validate untrimmed_sigma_gradient is non-negative when present."""
    if value is not None and value < 0:
        raise ValueError("untrimmed_sigma_gradient cannot be negative")
    return value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_spec1_untrimmed_sigma.py -v`

Expected: 3 PASS. No failures.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/database/models.py tests/test_spec1_untrimmed_sigma.py
git commit -m "feat(spec1): add untrimmed_sigma_gradient column to TrackResult

New nullable Float column + index + non-negative check constraint +
validator.  Mirrors the existing sigma_gradient schema entries.  Used
by the multi-metric drift detector (Spec 2) as an upstream element-
quality signal independent of post-trim trim quality.

Spec: docs/superpowers/specs/2026-05-30-spec1-upstream-signals-capture.md"
```

---

## Task 2: Startup migration — idempotent ALTER TABLE for existing DBs

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`
- Test: `tests/test_spec1_untrimmed_sigma.py` (append)

**Why:** Task 1's model change handles fresh databases (via `create_all`). Existing databases need the column added at startup. The codebase uses a `try-probe-except OperationalError-ALTER` pattern; we follow it.

- [ ] **Step 1: Add the migration test**

Append to `tests/test_spec1_untrimmed_sigma.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: startup migration -- existing DBs (pre-Spec-1) gain the column
# idempotently at app startup.
# ---------------------------------------------------------------------------


def test_migration_adds_column_to_preexisting_db(tmp_path):
    """A DB created BEFORE Spec 1 (column not in schema) must gain the column
    when DatabaseManager initializes against it.  Running DatabaseManager
    a second time on the same DB must be a no-op (idempotent).
    """
    import sqlite3

    db_path = tmp_path / "preexisting.db"

    # Build a track_results table WITHOUT the new column, simulating a pre-
    # Spec-1 database.  Minimum columns: id, analysis_id, track_id,
    # sigma_gradient.  Real schema has more but this is enough to verify
    # the migration probe + ALTER path.
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE track_results (
            id INTEGER PRIMARY KEY,
            analysis_id INTEGER NOT NULL,
            track_id TEXT NOT NULL,
            sigma_gradient FLOAT
        );
    """)
    conn.commit()
    conn.close()

    # Confirm pre-state: no untrimmed_sigma_gradient column.
    conn = sqlite3.connect(db_path)
    cols_before = {row[1] for row in conn.execute("PRAGMA table_info(track_results)")}
    conn.close()
    assert "untrimmed_sigma_gradient" not in cols_before

    # Run DatabaseManager init -- this is where startup migrations fire.
    from laser_trim_analyzer.database.manager import DatabaseManager

    mgr = DatabaseManager(db_path)

    # Verify post-state: column now exists.
    conn = sqlite3.connect(db_path)
    cols_after = {row[1] for row in conn.execute("PRAGMA table_info(track_results)")}
    conn.close()
    assert "untrimmed_sigma_gradient" in cols_after, (
        f"Migration failed to add column; got columns: {sorted(cols_after)}"
    )

    # Idempotency: build a second DatabaseManager against the same path; no
    # raise, no duplicate column.
    mgr2 = DatabaseManager(db_path)
    conn = sqlite3.connect(db_path)
    cols_repeat = {row[1] for row in conn.execute("PRAGMA table_info(track_results)")}
    conn.close()
    assert cols_repeat == cols_after, "Second init shouldn't change the schema"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_spec1_untrimmed_sigma.py::test_migration_adds_column_to_preexisting_db -v`

Expected: FAIL — column not added (migration not yet written).

- [ ] **Step 3: Add the migration entry in DatabaseManager startup**

In `src/laser_trim_analyzer/database/manager.py`, find the migration block at lines 340-360 (the `is_anomaly` migration). The pattern is: probe with a SELECT, catch `OperationalError`, rollback, then ALTER TABLE. Locate the end of one of the existing track_results migrations (a good landmark is the `max_deviation` migration around line 551-553). After that migration's closing `except` block, add a new migration entry following the same pattern:

```python
            # Migration: Add untrimmed_sigma_gradient column to track_results.
            # Spec 1 (2026-05-30): upstream element-quality signal independent
            # of post-trim sigma_gradient.  Backfilled by natural reprocess flow.
            try:
                session.execute(
                    text("SELECT untrimmed_sigma_gradient FROM track_results LIMIT 1")
                )
            except OperationalError:
                session.rollback()
                logger.info(
                    "Running migration: Adding untrimmed_sigma_gradient column"
                )
                try:
                    session.execute(text(
                        "ALTER TABLE track_results "
                        "ADD COLUMN untrimmed_sigma_gradient FLOAT"
                    ))
                    session.execute(text(
                        "CREATE INDEX IF NOT EXISTS "
                        "idx_track_untrimmed_sigma_gradient "
                        "ON track_results (untrimmed_sigma_gradient)"
                    ))
                    session.commit()
                    logger.info(
                        "Migration completed: Added untrimmed_sigma_gradient"
                    )
                except Exception as e:
                    logger.warning(
                        f"Migration warning (may already exist): {e}"
                    )
```

(Adjust indentation to match surrounding code — the migrations sit inside a method, typically four spaces deeper than the method header.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_spec1_untrimmed_sigma.py::test_migration_adds_column_to_preexisting_db -v`

Expected: PASS.

- [ ] **Step 5: Run the whole test file to confirm Task 1 tests still pass**

Run: `pytest tests/test_spec1_untrimmed_sigma.py -v`

Expected: 4 PASS (Task 1's 3 schema tests + Task 2's migration test).

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py tests/test_spec1_untrimmed_sigma.py
git commit -m "feat(spec1): startup migration for untrimmed_sigma_gradient

Adds idempotent ALTER TABLE + CREATE INDEX entries in
DatabaseManager startup so pre-existing databases gain the new column
without manual intervention.  Follows the existing try-probe-except
pattern used by other track_results column migrations."
```

---

## Task 3: Pydantic TrackData field

**Files:**
- Modify: `src/laser_trim_analyzer/core/models.py`
- Test: `tests/test_spec1_untrimmed_sigma.py` (append)

- [ ] **Step 1: Add the Pydantic test**

Append to `tests/test_spec1_untrimmed_sigma.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: Pydantic -- TrackData accepts untrimmed_sigma_gradient (optional).
# ---------------------------------------------------------------------------


def test_track_data_accepts_untrimmed_sigma_gradient_field():
    """TrackData must define the field as Optional[float], default None,
    with the same ge=0 constraint sigma_gradient uses.
    """
    from laser_trim_analyzer.core.models import TrackData

    # Field exists and defaults to None
    fields = TrackData.model_fields
    assert "untrimmed_sigma_gradient" in fields
    assert fields["untrimmed_sigma_gradient"].default is None

    # Round-trip a value
    td_kwargs = _minimal_track_data_kwargs()
    td_kwargs["untrimmed_sigma_gradient"] = 0.012
    td = TrackData(**td_kwargs)
    assert td.untrimmed_sigma_gradient == 0.012

    # Negative rejected by the ge=0 constraint
    td_kwargs["untrimmed_sigma_gradient"] = -0.001
    with pytest.raises(Exception):  # pydantic ValidationError
        TrackData(**td_kwargs)


def _minimal_track_data_kwargs():
    """Build the minimal valid kwargs for TrackData.

    Inspect TrackData's required fields in
    src/laser_trim_analyzer/core/models.py:86 and fill in sentinel values.
    The exact set may differ from what's shown here; if construction fails,
    add the missing fields with sentinel values.  Goal is to exercise the
    new field, not the rest of TrackData.
    """
    from laser_trim_analyzer.core.models import AnalysisStatus

    return dict(
        track_id="TRK1",
        status=AnalysisStatus.PASS,
        travel_length=1.0,
        linearity_spec=0.01,
        sigma_gradient=0.008,
        sigma_threshold=0.02,
        sigma_pass=True,
        optimal_offset=0.0,
        optimal_slope=0.0,
        linearity_error=0.005,
        linearity_pass=True,
        linearity_fail_points=0,
    )
```

> **Implementer note:** if `_minimal_track_data_kwargs` is missing required fields, Pydantic will tell you which ones. Add them with sensible sentinels; the goal is a TrackData instance, not a perfect one.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_spec1_untrimmed_sigma.py::test_track_data_accepts_untrimmed_sigma_gradient_field -v`

Expected: FAIL — `assert "untrimmed_sigma_gradient" in fields` fails because field doesn't exist yet.

- [ ] **Step 3: Add the field to TrackData**

In `src/laser_trim_analyzer/core/models.py`, find the `TrackData` class (around line 86). Locate the existing `sigma_gradient` field at line 101:

```python
sigma_gradient: Optional[float] = Field(None, ge=0, description="Sigma gradient value")
```

Immediately after it, add:

```python
untrimmed_sigma_gradient: Optional[float] = Field(
    None, ge=0,
    description="Sigma gradient calculated on untrimmed (pre-trim) arrays. "
                "Upstream element-quality signal; independent of post-trim "
                "process. NULL when untrimmed arrays absent or all-NaN.",
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_spec1_untrimmed_sigma.py::test_track_data_accepts_untrimmed_sigma_gradient_field -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/core/models.py tests/test_spec1_untrimmed_sigma.py
git commit -m "feat(spec1): add untrimmed_sigma_gradient to TrackData Pydantic model

Optional[float] with ge=0 constraint, mirroring sigma_gradient.
Default None means 'not measured' (untrimmed arrays absent or all-NaN).
The analyzer change in the next commit populates it."
```

---

## Task 4: Analyzer — compute untrimmed sigma in analyze_track

**Files:**
- Modify: `src/laser_trim_analyzer/core/analyzer.py`
- Test: `tests/test_spec1_untrimmed_sigma.py` (append)

This is the substantive logic change. Four edge cases must be covered:
1. Both untrimmed arrays present and well-formed → compute sigma
2. Arrays absent or empty → store None
3. Arrays present but all NaN → store None
4. Arrays present but too short (after NaN filter, fewer than `2 * END_POINT_FILTER_COUNT + 3` valid points) → store None

- [ ] **Step 1: Add the four analyzer tests**

Append to `tests/test_spec1_untrimmed_sigma.py`:

```python
# ---------------------------------------------------------------------------
# Task 4: analyzer -- analyze_track computes untrimmed_sigma_gradient
# correctly in four scenarios.
# ---------------------------------------------------------------------------


def _build_track_input(*, n=30, untrimmed_positions=None, untrimmed_errors=None,
                      include_untrimmed=True):
    """Build a synthetic track_data dict shaped like what the parser emits.

    Defaults to n=30 well-formed points so _calculate_sigma succeeds.
    Pass include_untrimmed=False to omit the untrimmed_* keys entirely.
    """
    import numpy as np

    positions = list(np.linspace(0.0, 1.0, n))
    # Errors with a small sinusoidal pattern -- non-zero gradients.
    errors = [0.001 * np.sin(2 * np.pi * i / n) for i in range(n)]

    out = dict(
        track_id="TRK1",
        positions=positions,
        errors=errors,
        upper_limits=[0.01] * n,
        lower_limits=[-0.01] * n,
        travel_length=1.0,
        linearity_spec=0.01,
        unit_length=1.0,
        untrimmed_resistance=1000.0,
        trimmed_resistance=950.0,
    )
    if include_untrimmed:
        out["untrimmed_positions"] = (
            untrimmed_positions if untrimmed_positions is not None else positions
        )
        out["untrimmed_errors"] = (
            untrimmed_errors
            if untrimmed_errors is not None
            else [0.002 * np.sin(2 * np.pi * i / n) for i in range(n)]
        )
    return out


def _run_analyze(track_input):
    """Invoke LaserTrimAnalyzer.analyze_track with sane wrapping context."""
    from laser_trim_analyzer.core.analyzer import LaserTrimAnalyzer
    from laser_trim_analyzer.config import Config

    analyzer = LaserTrimAnalyzer(config=Config())
    return analyzer.analyze_track(track_input, model="TEST-MODEL")


def test_untrimmed_sigma_populated_when_arrays_present():
    """Well-formed untrimmed arrays produce a finite positive sigma."""
    track_input = _build_track_input()
    result = _run_analyze(track_input)

    assert result.untrimmed_sigma_gradient is not None
    assert result.untrimmed_sigma_gradient >= 0
    # Spec says no Inf/NaN allowed
    import math
    assert math.isfinite(result.untrimmed_sigma_gradient)


def test_untrimmed_sigma_none_when_arrays_absent():
    """Missing the untrimmed_* keys entirely -> None, no error."""
    track_input = _build_track_input(include_untrimmed=False)
    result = _run_analyze(track_input)
    assert result.untrimmed_sigma_gradient is None


def test_untrimmed_sigma_none_when_arrays_empty():
    """Empty lists -> None."""
    track_input = _build_track_input(
        untrimmed_positions=[], untrimmed_errors=[]
    )
    result = _run_analyze(track_input)
    assert result.untrimmed_sigma_gradient is None


def test_untrimmed_sigma_none_when_arrays_all_nan():
    """All NaN -> filtered to empty -> None."""
    import math
    n = 30
    nans = [math.nan] * n
    track_input = _build_track_input(
        untrimmed_positions=nans, untrimmed_errors=nans
    )
    result = _run_analyze(track_input)
    assert result.untrimmed_sigma_gradient is None


def test_untrimmed_sigma_none_when_arrays_too_short():
    """Fewer than 2*END_POINT_FILTER_COUNT + 3 valid points -> None.

    Reads the threshold from constants so the test stays correct if
    END_POINT_FILTER_COUNT is tuned.
    """
    from laser_trim_analyzer.utils.constants import END_POINT_FILTER_COUNT

    too_short = 2 * END_POINT_FILTER_COUNT + 3 - 1  # one below the gate
    short_positions = [i * 0.01 for i in range(too_short)]
    short_errors = [i * 0.001 for i in range(too_short)]

    track_input = _build_track_input(
        untrimmed_positions=short_positions,
        untrimmed_errors=short_errors,
    )
    result = _run_analyze(track_input)
    assert result.untrimmed_sigma_gradient is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spec1_untrimmed_sigma.py -v -k untrimmed_sigma`

Expected: 5 FAILs — `result.untrimmed_sigma_gradient` is missing/None for all cases because the analyzer doesn't compute it yet. (Some may also fail with `AttributeError` if the field is just absent.)

- [ ] **Step 3: Add the calculation in `analyze_track`**

In `src/laser_trim_analyzer/core/analyzer.py`, find `analyze_track` (around line 165). The existing sigma calculation is at lines 213-217:

```python
        # Sigma analysis (with optional ML threshold)
        sigma_gradient, sigma_threshold = self._calculate_sigma(
            positions, errors, linearity_spec, travel_length, unit_length,
            model=model
        )
        sigma_pass = sigma_gradient <= sigma_threshold
```

Immediately after the `sigma_pass = sigma_gradient <= sigma_threshold` line (line 217), insert this block:

```python
        # Untrimmed (upstream) sigma -- independent signal for Spec 2 drift
        # detection.  Not used for pass/fail.  Gated on data availability;
        # exception-safe because the per-track save must still proceed even
        # if the untrimmed arrays are malformed.
        untrimmed_sigma_gradient: Optional[float] = None
        _untrimmed_positions = track_data.get("untrimmed_positions") or []
        _untrimmed_errors = track_data.get("untrimmed_errors") or []
        if _untrimmed_positions and _untrimmed_errors:
            try:
                # Filter NaN consistently with _calculate_trim_effectiveness.
                _valid_pairs = [
                    (p, e)
                    for p, e in zip(_untrimmed_positions, _untrimmed_errors)
                    if p is not None and e is not None
                    and not np.isnan(p) and not np.isnan(e)
                ]
                if len(_valid_pairs) > 2 * END_POINT_FILTER_COUNT + 3:
                    _up = [p for p, _ in _valid_pairs]
                    _ue = [e for _, e in _valid_pairs]
                    _sig, _ = self._calculate_sigma(
                        _up, _ue, linearity_spec, travel_length, unit_length, model
                    )
                    untrimmed_sigma_gradient = _sig
            except Exception as e:
                logger.warning(
                    f"Untrimmed sigma calculation failed for track "
                    f"{track_id!r}: {e}; storing NULL"
                )
                untrimmed_sigma_gradient = None
```

Now find the `return TrackData(...)` call at the end of `analyze_track` (around line 284-300). Locate the line `sigma_gradient=sigma_gradient,` and immediately after `sigma_pass=sigma_pass,` add:

```python
            untrimmed_sigma_gradient=untrimmed_sigma_gradient,
```

(Indentation must match the other field assignments inside that constructor.)

Verify the imports at the top of the file include `Optional` from `typing`. If not present, add it to the existing typing import. Verify `END_POINT_FILTER_COUNT` is already imported (line 21 confirms this).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_spec1_untrimmed_sigma.py -v -k untrimmed_sigma`

Expected: 5 PASS.

- [ ] **Step 5: Run the whole test file**

Run: `pytest tests/test_spec1_untrimmed_sigma.py -v`

Expected: 9 PASS total (Task 1: 3, Task 2: 1, Task 3: 1, Task 4: 5; some test names overlap — count is approximate but all should pass).

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/core/analyzer.py tests/test_spec1_untrimmed_sigma.py
git commit -m "feat(spec1): compute untrimmed_sigma_gradient in analyze_track

Parallel _calculate_sigma call on untrimmed arrays after the existing
post-trim sigma calculation.  Gated on data availability (returns None
when arrays absent, empty, all-NaN, or too short after NaN filter).
Exception-safe so the per-track save proceeds even if the untrimmed
arrays are malformed.

Reuses the same threshold-context inputs as the post-trim call but
consumes only the gradient half of the (gradient, threshold) tuple --
untrimmed sigma has no pass/fail threshold per Spec 1."
```

---

## Task 5: DB save mapping — persist the new field

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`
- Test: `tests/test_spec1_untrimmed_sigma.py` (append)

- [ ] **Step 1: Add the persistence round-trip test**

Append to `tests/test_spec1_untrimmed_sigma.py`:

```python
# ---------------------------------------------------------------------------
# Task 5: DB save mapping -- TrackData.untrimmed_sigma_gradient persists
# through save and reloads correctly.
# ---------------------------------------------------------------------------


def test_untrimmed_sigma_round_trips_through_db(tmp_path):
    """A TrackData with untrimmed_sigma_gradient set, saved via
    DatabaseManager, must reload with the same value.
    """
    from datetime import datetime
    from laser_trim_analyzer.core.models import (
        AnalysisResult, AnalysisStatus, FileMetadata, SystemType,
    )
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        TrackResult as DBTrackResult,
    )

    mgr = DatabaseManager(tmp_path / "roundtrip.db")

    # Build a minimal AnalysisResult that carries a TrackData with the new
    # field populated.  Use AnalysisStatus.PASS so the analyzer's track
    # validator (which permits empty tracks only for ERROR) is satisfied
    # by including one TrackData entry.
    td_kwargs = _minimal_track_data_kwargs()
    td_kwargs["untrimmed_sigma_gradient"] = 0.0123
    from laser_trim_analyzer.core.models import TrackData
    td = TrackData(**td_kwargs)

    ar = AnalysisResult(
        metadata=FileMetadata(
            filename="roundtrip.xls",
            file_path=Path("/fake/roundtrip.xls"),
            file_date=datetime(2026, 5, 30),
            model="TEST-MODEL",
            serial="0001",
            system=SystemType.A,
            has_multi_tracks=False,
        ),
        overall_status=AnalysisStatus.PASS,
        tracks=[td],
        processing_time=0.1,
    )

    analysis_id = mgr.save_analysis(ar)
    assert analysis_id > 0

    # Reload and verify.
    with mgr.session() as session:
        rows = session.query(DBTrackResult).filter(
            DBTrackResult.analysis_id == analysis_id
        ).all()
        assert len(rows) == 1
        assert rows[0].untrimmed_sigma_gradient == pytest.approx(0.0123)
```

> **Implementer note:** Adjust the `AnalysisResult` / `FileMetadata` / `TrackData` constructor calls to match the actual field names if Pydantic rejects this. The test from `tests/test_log_derived_bugfixes_2026_05_30.py::test_save_analysis_dedupes_on_metadata_not_file_path` (commit `2251c22` on `main`) is a good reference for the working constructor shape.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_spec1_untrimmed_sigma.py::test_untrimmed_sigma_round_trips_through_db -v`

Expected: FAIL — saved row's `untrimmed_sigma_gradient` is `None` because the `_track_to_db` mapping doesn't assign it yet.

- [ ] **Step 3: Add the field assignment in the track-to-DB mapper**

In `src/laser_trim_analyzer/database/manager.py`, find the `DBTrackResult(...)` constructor call (around line 2625-2660). Locate the existing `sigma_gradient=track.sigma_gradient,` line (around line 2630). Immediately after it, add:

```python
            untrimmed_sigma_gradient=track.untrimmed_sigma_gradient,
```

(Match the indentation of the surrounding field assignments.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_spec1_untrimmed_sigma.py::test_untrimmed_sigma_round_trips_through_db -v`

Expected: PASS.

- [ ] **Step 5: Run the whole test file**

Run: `pytest tests/test_spec1_untrimmed_sigma.py -v`

Expected: 10 PASS total. Zero failures.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py tests/test_spec1_untrimmed_sigma.py
git commit -m "feat(spec1): persist untrimmed_sigma_gradient via _track_to_db

Single field assignment in the DBTrackResult constructor call.  Mirrors
the existing sigma_gradient mapping.  Spec 1 column + analyzer +
persistence chain is now complete."
```

---

## Task 6: Regression guards + historical-test smoke

**Files:**
- Test: `tests/test_spec1_untrimmed_sigma.py` (append)
- No source changes — this task is verification.

- [ ] **Step 1: Add the regression test for post-trim sigma**

Append to `tests/test_spec1_untrimmed_sigma.py`:

```python
# ---------------------------------------------------------------------------
# Task 6: regression -- existing sigma_gradient unchanged on the same input,
# and the historical regression suite (test_5_8_2026_bugfixes.py) still
# passes with the new analyzer code.
# ---------------------------------------------------------------------------


def test_existing_sigma_gradient_unchanged_after_spec1():
    """Spec 1 must not change the value of sigma_gradient for any input.
    Compute sigma against the SAME post-trim arrays directly and confirm the
    analyze_track output matches.
    """
    from laser_trim_analyzer.core.analyzer import LaserTrimAnalyzer
    from laser_trim_analyzer.config import Config

    track_input = _build_track_input()
    analyzer = LaserTrimAnalyzer(config=Config())

    # Direct sigma calc via the same function the analyzer uses internally.
    direct_sigma, _direct_threshold = analyzer._calculate_sigma(
        track_input["positions"],
        track_input["errors"],
        track_input["linearity_spec"],
        track_input["travel_length"],
        track_input["unit_length"],
        model="TEST-MODEL",
    )

    # Now run the full analyze_track and confirm the returned sigma matches.
    result = analyzer.analyze_track(track_input, model="TEST-MODEL")
    assert result.sigma_gradient == pytest.approx(direct_sigma), (
        f"Regression: analyze_track returned sigma_gradient="
        f"{result.sigma_gradient} but direct _calculate_sigma returned "
        f"{direct_sigma}"
    )


def test_untrimmed_only_record_still_has_null_sigma_gradient(tmp_path):
    """Existing UNTRIMMED-status code path stores sigma_gradient=None; Spec 1
    must not regress that.  Mirrors tests/test_drift_dashboard_data.py:347
    but kept here so Spec 1's contract is self-contained.
    """
    from datetime import datetime, timedelta
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAnalysisResult,
        SystemType as DBSystemType,
        StatusType as DBStatusType,
    )

    db = DatabaseManager(tmp_path / "untrimmed_only.db")
    today = datetime.now()

    with db.session() as s:
        ar = DBAnalysisResult(
            filename="UNTRIMMED-MODEL_0001.xls",
            file_path="/fake/UNTRIMMED-MODEL_0001.xls",
            file_hash="untrimmed-only-spec1",
            model="UNTRIMMED-MODEL",
            serial="0001",
            system=DBSystemType.B,
            file_date=today - timedelta(days=1),
            timestamp=datetime.now(),
            overall_status=DBStatusType.UNTRIMMED,
            has_multi_tracks=False,
            processing_time=0.1,
        )
        s.add(ar)
        s.commit()

        # Confirm: no row -> no sigma_gradient.  This is the existing
        # behavior; Spec 1 doesn't add tracks to UNTRIMMED-only analyses.
        from laser_trim_analyzer.database.models import TrackResult as DBTR
        tracks = s.query(DBTR).filter(DBTR.analysis_id == ar.id).all()
        assert tracks == [], (
            "UNTRIMMED-only analyses should have no track rows per existing "
            "behavior; Spec 1 must not change this."
        )
```

- [ ] **Step 2: Run the regression tests**

Run: `pytest tests/test_spec1_untrimmed_sigma.py::test_existing_sigma_gradient_unchanged_after_spec1 tests/test_spec1_untrimmed_sigma.py::test_untrimmed_only_record_still_has_null_sigma_gradient -v`

Expected: 2 PASS.

- [ ] **Step 3: Run the full historical regression file**

Run: `pytest tests/test_log_derived_bugfixes_2026_05_30.py tests/test_5_8_2026_bugfixes.py tests/test_spec1_untrimmed_sigma.py -v`

Expected: all PASS (43 from log-derived + 8 from historical + 12 from Spec 1 ≈ 63 total). Zero failures.

If any regression appears in `test_log_derived_bugfixes_2026_05_30.py` or `test_5_8_2026_bugfixes.py`, stop and report — Spec 1 has accidentally regressed a prior fix.

- [ ] **Step 4: Commit**

```bash
git add tests/test_spec1_untrimmed_sigma.py
git commit -m "test(spec1): regression guards for sigma_gradient + UNTRIMMED path

Two final guards completing the Spec 1 test suite:

1. analyze_track's sigma_gradient value is unchanged for the same input
   after the untrimmed parallel calc was added.
2. UNTRIMMED-only analysis records still carry no track rows (existing
   behavior preserved).

Plus a full regression sweep of test_log_derived_bugfixes_2026_05_30.py
and test_5_8_2026_bugfixes.py to confirm Spec 1 hasn't disturbed prior
fixes."
```

---

## Post-implementation verification (one-shot)

After all 6 tasks land, run the full test suite to confirm health:

```
pytest tests/ -v
```

Expected: zero failures. Spec 1's 12 tests pass, all pre-existing tests pass.

Then verify on a real local run if convenient:
1. Process a real trim file (any with `untrimmed_positions` / `untrimmed_errors` in its data — that's most of them).
2. Open the SQLite DB: `sqlite3 ./data/analysis.db`
3. Query: `SELECT model, COUNT(*), COUNT(untrimmed_sigma_gradient) FROM track_results GROUP BY model LIMIT 10;`
4. Expect: post-Spec-1 files have non-NULL `untrimmed_sigma_gradient`; pre-Spec-1 files have NULL until reprocessed.

## Out-of-scope reminders (do NOT do these)

- **Do not** change the drift detector to consume the new column. That's Spec 2.
- **Do not** add a UI surface for the new column. That's Spec 3.
- **Do not** write a one-shot backfill script. Reprocess flow handles it naturally.
- **Do not** add a `sigma_threshold` companion for untrimmed sigma. It's a drift signal, not a manufacturing spec — no pass/fail.
- **Do not** export the new column in Excel exports yet. Add only when requested.
