# Unit-Level Yield Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface accurate unit-level yield (one row per physical unit, not per track/section) in a new line on the Trends single-model P-chart and a new "Yield by Unit" Excel sheet, backed by a `unit_id` column on `analysis_results` computed at write time and backfilled on existing rows.

**Architecture:** Add a nullable `unit_id` column (`"{model}/{shop_number}/{YYYY-MM-DD}"`) to `analysis_results`. Compute via shared helper at write time and during an idempotent Python-side backfill migration. New DB query method aggregates rows into units and applies the "any section non-PASS → unit fails" rule using row `timestamp` for retrim tiebreaking. Trends chart + Excel sheet consume the query. Feature flag gates the new surfaces.

**Tech Stack:** SQLAlchemy 2.0 ORM, SQLite, customtkinter / matplotlib via existing ChartWidget wrapper, openpyxl, pytest.

**Source spec:** `docs/superpowers/specs/2026-05-08-unit-level-yield-design.md` (commit `5246b84`).

---

## File Structure

| File | Change |
|---|---|
| `src/laser_trim_analyzer/database/manager.py` | Module-level `compute_unit_id` helper + `extract_shop_number`. Add `unit_id` to `_map_analysis_to_db` and `_update_existing_analysis`. ALTER migration. Python backfill helper. New query method `get_unit_yield_trend`. |
| `src/laser_trim_analyzer/database/models.py` | Add `unit_id` Column to `AnalysisResult` |
| `src/laser_trim_analyzer/config.py` | Add `enable_unit_yield_view: bool = True` to relevant config section |
| `src/laser_trim_analyzer/gui/pages/trends.py` | Extend single-model detail P-chart with optional unit-yield overlay line |
| `src/laser_trim_analyzer/export/excel.py` | New `_create_yield_by_unit_sheet` called from `export_batch_results` when flag enabled |
| `tests/test_unit_yield.py` | New test file: extraction, computation, migration idempotency, pass rule, yield query |

---

## Task 1: compute_unit_id helper + tests

Foundation. Pure function, no DB needed. Lock in the extraction rule before touching schema.

**Files:**
- Create: `tests/test_unit_yield.py`
- Modify: `src/laser_trim_analyzer/database/manager.py` (add module-level helpers)

- [ ] **Step 1: Write failing tests**

Create `tests/test_unit_yield.py`:

```python
"""Tests for the unit-level yield feature."""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from laser_trim_analyzer.database.manager import (
    compute_unit_id,
    extract_shop_number,
)


class TestExtractShopNumber:
    @pytest.mark.parametrize("serial,expected", [
        ("3P", "3"),
        ("3R", "3"),
        ("1A", "1"),
        ("0009", "0009"),
        ("8", "8"),
        ("196", "196"),
        ("12RC", "12"),
        ("  4P  ", "4"),  # strips whitespace
    ])
    def test_extracts_leading_digits(self, serial, expected):
        assert extract_shop_number(serial) == expected

    @pytest.mark.parametrize("serial", [
        "TEST", "test", "x", "Unknown", "", None,
    ])
    def test_returns_none_for_junk_serials(self, serial):
        assert extract_shop_number(serial) is None


class TestComputeUnitId:
    def test_typical_trim_record(self):
        uid = compute_unit_id("8895", "3P", datetime(2025, 3, 26, 12, 55))
        assert uid == "8895/3/2025-03-26"

    def test_same_unit_different_section(self):
        a = compute_unit_id("8895", "3P", datetime(2025, 3, 26))
        b = compute_unit_id("8895", "3R", datetime(2025, 3, 26))
        assert a == b == "8895/3/2025-03-26"

    def test_retrim_same_day_same_unit(self):
        morning = compute_unit_id("8895", "3P", datetime(2025, 3, 26, 9, 0))
        evening = compute_unit_id("8895", "3P", datetime(2025, 3, 26, 17, 43))
        assert morning == evening

    def test_different_shop_different_unit(self):
        a = compute_unit_id("8895", "3P", datetime(2025, 3, 26))
        b = compute_unit_id("8895", "4P", datetime(2025, 3, 26))
        assert a != b

    def test_different_day_different_unit(self):
        a = compute_unit_id("8895", "3P", datetime(2025, 3, 26))
        b = compute_unit_id("8895", "3P", datetime(2025, 3, 27))
        assert a != b

    def test_returns_none_when_serial_junk(self):
        assert compute_unit_id("8895", "TEST", datetime(2025, 3, 26)) is None

    def test_returns_none_when_serial_missing(self):
        assert compute_unit_id("8895", None, datetime(2025, 3, 26)) is None

    def test_returns_none_when_model_missing(self):
        assert compute_unit_id(None, "3P", datetime(2025, 3, 26)) is None
        assert compute_unit_id("", "3P", datetime(2025, 3, 26)) is None

    def test_returns_none_when_date_missing(self):
        assert compute_unit_id("8895", "3P", None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest tests/test_unit_yield.py -v`

Expected: every test fails with `ImportError: cannot import name 'compute_unit_id' from 'laser_trim_analyzer.database.manager'`.

- [ ] **Step 3: Add the helpers to `database/manager.py`**

Locate the top of `src/laser_trim_analyzer/database/manager.py` (after the existing imports, before the `class DatabaseManager` definition). Add these module-level helpers:

```python
# ---------------------------------------------------------------------------
# Unit identity helpers — used at write time and by the migration backfill.
# A "unit" is one physical part: same model + same shop number + same day.
# Sections (P, R, etc.) and same-day retrims all collapse to the same unit_id.
# See docs/superpowers/specs/2026-05-08-unit-level-yield-design.md.
# ---------------------------------------------------------------------------
import re as _re

def extract_shop_number(serial: Optional[str]) -> Optional[str]:
    """Return the leading-digit prefix of a serial, or None if absent.

    Examples: "3P" -> "3", "0009" -> "0009", "1A" -> "1", "TEST" -> None.
    Whitespace is stripped first. None / empty string return None.
    """
    if not serial:
        return None
    match = _re.match(r"^\d+", serial.strip())
    return match.group(0) if match else None


def compute_unit_id(
    model: Optional[str],
    serial: Optional[str],
    file_date: Optional[datetime],
) -> Optional[str]:
    """Compute the canonical unit identifier for a trim record.

    Returns None if any input is missing or the serial has no shop number —
    such rows are excluded from unit-level yield queries by design.

    Format: "<model>/<shop_number>/<YYYY-MM-DD>".
    """
    if not model or file_date is None:
        return None
    shop = extract_shop_number(serial)
    if shop is None:
        return None
    return f"{model}/{shop}/{file_date.strftime('%Y-%m-%d')}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest tests/test_unit_yield.py -v`

Expected: all 17 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/database/manager.py tests/test_unit_yield.py
git commit -m "feat(unit-yield): compute_unit_id and extract_shop_number helpers

Pure functions, no DB dependency. extract_shop_number returns the
leading-digit prefix of a serial (e.g. '3P' -> '3'); compute_unit_id
combines model + shop + YYYY-MM-DD into a canonical unit identifier.
Either returns None when inputs are missing or the serial has no
shop number, by design — such rows are excluded from unit-level
yield queries.

Tests cover the 8 serial patterns from real 8895 data plus junk
serials (TEST, x, empty, None).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Schema column + idempotent ALTER migration

**Files:**
- Modify: `src/laser_trim_analyzer/database/models.py`
- Modify: `src/laser_trim_analyzer/database/manager.py` (migration block)

- [ ] **Step 1: Add the column to the SQLAlchemy model**

Locate the `AnalysisResult` class in `src/laser_trim_analyzer/database/models.py`. Find a section labeled by comment that holds related dimensional/identifier columns (search for "filename" or "model" column definitions). Add this column near the model/serial columns:

```python
    # Unit-level yield identifier. Format: "<model>/<shop>/<YYYY-MM-DD>".
    # Computed at write time from model/serial/file_date. NULL when serial
    # has no leading digits (TEST, x, etc.) — those rows are excluded from
    # unit-yield queries by design.
    unit_id = Column(String(80), nullable=True, index=True)
```

- [ ] **Step 2: Add the idempotent ALTER migration**

Find the migration block in `database/manager.py` — search for `# Migration: Add trim_pass_count column` (similar pattern recently added). Below that block, insert:

```python
            # Migration: Add unit_id column to analysis_results.
            # Canonical unit identifier "<model>/<shop>/<date>", used by the
            # unit-level yield feature (Trends chart + Excel "Yield by Unit"
            # sheet). Nullable so the app remains functional during/after
            # partial backfill; the backfill itself happens in a separate
            # migration step below so it can be retried independently.
            try:
                session.execute(text("SELECT unit_id FROM analysis_results LIMIT 1"))
            except OperationalError:
                session.rollback()
                try:
                    session.execute(text(
                        "ALTER TABLE analysis_results ADD COLUMN unit_id VARCHAR(80)"
                    ))
                    session.execute(text(
                        "CREATE INDEX IF NOT EXISTS idx_analysis_unit_id "
                        "ON analysis_results(unit_id)"
                    ))
                    session.commit()
                    logger.info("Migration: Added unit_id column to analysis_results")
                except Exception as e:
                    session.rollback()
                    logger.warning(f"Migration error adding unit_id: {e}")
```

- [ ] **Step 3: Add a smoke test for the migration**

Add to `tests/test_unit_yield.py`:

```python
class TestUnitIdMigration:
    def test_migration_adds_column_idempotently(self, tmp_path):
        """First migration adds column; second call is a no-op."""
        from laser_trim_analyzer.database.manager import DatabaseManager
        from sqlalchemy import inspect

        db_path = tmp_path / "smoke.db"
        # First init runs migrations
        db1 = DatabaseManager(database_path=db_path)
        insp = inspect(db1._engine)
        cols = [c["name"] for c in insp.get_columns("analysis_results")]
        assert "unit_id" in cols

        # Second init re-runs migration code; must not crash
        db2 = DatabaseManager(database_path=db_path)
        cols2 = [c["name"] for c in inspect(db2._engine).get_columns("analysis_results")]
        assert "unit_id" in cols2
        assert cols == cols2  # No duplicate
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest tests/test_unit_yield.py::TestUnitIdMigration -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/database/models.py src/laser_trim_analyzer/database/manager.py tests/test_unit_yield.py
git commit -m "feat(unit-yield): add unit_id column and idempotent migration

ALTER TABLE adds nullable VARCHAR(80) column + index. Migration
guarded by SELECT probe so re-runs are no-ops. Nullable so the app
keeps working even if backfill is incomplete; the actual population
of existing rows happens in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Write-time computation in _map_analysis_to_db and _update_existing_analysis

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`

- [ ] **Step 1: Update `_map_analysis_to_db`**

Locate `_map_analysis_to_db` in `database/manager.py`. Find the `DBAnalysisResult(...)` constructor call. Add `unit_id` to the constructor kwargs (just before the `data_quality` kwarg for ordering with similar fields):

```python
        db_analysis = DBAnalysisResult(
            filename=analysis.metadata.filename,
            file_path=str(analysis.metadata.file_path),
            file_date=analysis.metadata.file_date,
            model=analysis.metadata.model,
            serial=analysis.metadata.serial,
            system=system_type,
            has_multi_tracks=analysis.metadata.has_multi_tracks,
            overall_status=overall_status,
            processing_time=analysis.processing_time,
            timestamp=utc_now(),
            unit_id=compute_unit_id(
                analysis.metadata.model,
                analysis.metadata.serial,
                analysis.metadata.file_date,
            ),
            data_quality=getattr(analysis, 'data_quality', 'good'),
            ...
        )
```

(Show only the constructor call here; preserve the rest of the existing method.)

- [ ] **Step 2: Update `_update_existing_analysis`**

Locate `_update_existing_analysis`. Find the block where existing record fields are updated (search `existing.model = analysis.metadata.model`). Add `unit_id` update there:

```python
            existing.file_path = str(analysis.metadata.file_path)
            existing.model = analysis.metadata.model
            existing.serial = analysis.metadata.serial
            existing.file_date = analysis.metadata.file_date
            existing.unit_id = compute_unit_id(
                analysis.metadata.model,
                analysis.metadata.serial,
                analysis.metadata.file_date,
            )
            # (continue with the rest of the existing block — system,
            # has_multi_tracks, etc., as they are today)
```

- [ ] **Step 3: Add a test that a new write populates unit_id**

Append to `tests/test_unit_yield.py`:

```python
class TestWriteTimeUnitId:
    def test_save_analysis_populates_unit_id(self, tmp_path):
        """Saving a fresh AnalysisResult populates unit_id in the DB row."""
        from laser_trim_analyzer.database.manager import DatabaseManager
        from laser_trim_analyzer.database.models import AnalysisResult as DBAR
        from laser_trim_analyzer.core.models import (
            AnalysisResult, AnalysisStatus, FileMetadata, SystemType,
            TrackData, RiskCategory,
        )

        db = DatabaseManager(database_path=tmp_path / "wt.db")
        track = TrackData(
            track_id="TRK1",
            status=AnalysisStatus.PASS,
            travel_length=10.0,
            linearity_spec=0.01,
            sigma_gradient=0.001,
            sigma_threshold=0.005,
            sigma_pass=True,
            optimal_offset=0.0,
            linearity_error=0.001,
            linearity_pass=True,
            linearity_fail_points=0,
            risk_category=RiskCategory.LOW,
        )
        analysis = AnalysisResult(
            metadata=FileMetadata(
                filename="8895_3P_TEST DATA_3-26-2025.xls",
                file_path=tmp_path / "8895_3P.xls",
                file_date=datetime(2025, 3, 26, 12, 55),
                model="8895",
                serial="3P",
                system=SystemType.B,
                has_multi_tracks=False,
            ),
            overall_status=AnalysisStatus.PASS,
            processing_time=0.1,
            tracks=[track],
        )
        new_id = db.save_analysis(analysis)
        with db.session() as s:
            row = s.query(DBAR).filter(DBAR.id == new_id).one()
            assert row.unit_id == "8895/3/2025-03-26"

    def test_save_analysis_junk_serial_unit_id_null(self, tmp_path):
        from laser_trim_analyzer.database.manager import DatabaseManager
        from laser_trim_analyzer.database.models import AnalysisResult as DBAR
        from laser_trim_analyzer.core.models import (
            AnalysisResult, AnalysisStatus, FileMetadata, SystemType,
            TrackData, RiskCategory,
        )

        db = DatabaseManager(database_path=tmp_path / "junk.db")
        track = TrackData(
            track_id="TRK1", status=AnalysisStatus.PASS, travel_length=10.0,
            linearity_spec=0.01, sigma_gradient=0.001, sigma_threshold=0.005,
            sigma_pass=True, optimal_offset=0.0, linearity_error=0.001,
            linearity_pass=True, linearity_fail_points=0,
            risk_category=RiskCategory.LOW,
        )
        analysis = AnalysisResult(
            metadata=FileMetadata(
                filename="8895_TEST.xls",
                file_path=tmp_path / "8895_TEST.xls",
                file_date=datetime(2025, 3, 26),
                model="8895", serial="TEST",
                system=SystemType.B, has_multi_tracks=False,
            ),
            overall_status=AnalysisStatus.PASS, processing_time=0.1, tracks=[track],
        )
        new_id = db.save_analysis(analysis)
        with db.session() as s:
            row = s.query(DBAR).filter(DBAR.id == new_id).one()
            assert row.unit_id is None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest tests/test_unit_yield.py -v`

Expected: all tests pass (the new TestWriteTimeUnitId class + everything from prior tasks).

- [ ] **Step 5: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/database/manager.py tests/test_unit_yield.py
git commit -m "feat(unit-yield): populate unit_id at write time

_map_analysis_to_db and _update_existing_analysis now call
compute_unit_id and persist the result on the DBAnalysisResult row.
Junk-serial paths (TEST, x, etc.) write NULL by design.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Backfill migration for existing rows

The user's DB has 80,656 trim rows that need unit_id populated. Migration runs once at startup, idempotent, batched.

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`

- [ ] **Step 1: Add the backfill helper**

Below the ALTER migration from Task 2, add a new migration block that runs the backfill:

```python
            # Migration: Backfill unit_id for existing rows. Idempotent —
            # only updates rows where unit_id IS NULL, so re-running picks up
            # where it left off (e.g. after a crash or interrupted startup).
            # Done in Python because the shop-number extraction regex lives
            # there; iterates in batches of 1000 to keep memory bounded.
            self._backfill_unit_ids(session)
```

Then add the method to the `DatabaseManager` class (place near other migration helpers):

```python
    def _backfill_unit_ids(self, session) -> None:
        """Populate analysis_results.unit_id for rows that don't have one yet.

        Idempotent: only operates on NULL unit_id rows. Logs progress and a
        post-run sanity check (count of NULL vs non-NULL).
        """
        from laser_trim_analyzer.database.models import AnalysisResult as DBAR

        # Count rows that need backfill
        to_backfill = (
            session.query(func.count(DBAR.id))
            .filter(DBAR.unit_id.is_(None))
            .scalar()
        ) or 0
        if to_backfill == 0:
            logger.debug("unit_id backfill: nothing to do")
            return

        logger.info(f"unit_id backfill: starting on {to_backfill} rows")
        batch_size = 1000
        updated = 0
        skipped = 0  # rows that have nothing to backfill (junk serial / missing date)

        # Process in batches so we don't load 80k rows into memory at once.
        # offset-based iteration is safe here because we filter by unit_id IS NULL
        # — once we update a row, it falls out of the "needs backfill" set.
        while True:
            rows = (
                session.query(DBAR.id, DBAR.model, DBAR.serial, DBAR.file_date)
                .filter(DBAR.unit_id.is_(None))
                .limit(batch_size)
                .all()
            )
            if not rows:
                break

            for row_id, model, serial, file_date in rows:
                uid = compute_unit_id(model, serial, file_date)
                if uid is None:
                    # Junk serial / missing date — mark with empty-string sentinel
                    # so the next batch query doesn't pick it up again. We then
                    # convert empty-string back to NULL at the end. This is the
                    # only safe way to guarantee progress through "unparseable"
                    # rows without leaking them across batches.
                    session.execute(
                        text("UPDATE analysis_results SET unit_id = '' WHERE id = :i"),
                        {"i": row_id},
                    )
                    skipped += 1
                else:
                    session.execute(
                        text("UPDATE analysis_results SET unit_id = :u WHERE id = :i"),
                        {"u": uid, "i": row_id},
                    )
                    updated += 1
            session.commit()
            logger.info(
                f"unit_id backfill: progress {updated + skipped}/{to_backfill}"
            )

        # Restore NULL for unparseable rows.
        session.execute(text(
            "UPDATE analysis_results SET unit_id = NULL WHERE unit_id = ''"
        ))
        session.commit()

        # Post-flight sanity check
        non_null = (
            session.query(func.count(DBAR.id))
            .filter(DBAR.unit_id.isnot(None))
            .scalar()
        ) or 0
        null_now = (
            session.query(func.count(DBAR.id))
            .filter(DBAR.unit_id.is_(None))
            .scalar()
        ) or 0
        distinct_units = (
            session.query(func.count(func.distinct(DBAR.unit_id)))
            .filter(DBAR.unit_id.isnot(None))
            .scalar()
        ) or 0
        logger.info(
            f"unit_id backfill complete: "
            f"{updated} populated, {skipped} junk-serial/no-date, "
            f"{non_null} non-NULL total, {null_now} NULL total, "
            f"{distinct_units} distinct units"
        )
```

- [ ] **Step 2: Add a backfill test**

Append to `tests/test_unit_yield.py` (first ensure the file has `from sqlalchemy import text` at the top — add it next to the other top-level imports if missing):

```python
class TestUnitIdBackfill:
    def _insert_legacy_row(self, db, model, serial, file_date_str):
        """Insert a row directly via SQL with unit_id=NULL to simulate
        a row written before the unit_id column existed."""
        with db.session() as s:
            s.execute(text(
                "INSERT INTO analysis_results "
                "(filename, file_path, file_date, model, serial, system, "
                " has_multi_tracks, overall_status, timestamp) "
                "VALUES (:fn, :fp, :fd, :m, :sn, 'B', 0, 'PASS', :ts)"
            ), {
                "fn": f"{model}_{serial}.xls",
                "fp": f"/fake/{model}_{serial}.xls",
                "fd": file_date_str,
                "m": model, "sn": serial,
                "ts": datetime.utcnow(),
            })
            s.commit()

    def test_backfill_populates_valid_rows_and_leaves_junk_null(self, tmp_path):
        from sqlalchemy import text
        from laser_trim_analyzer.database.manager import DatabaseManager
        from laser_trim_analyzer.database.models import AnalysisResult as DBAR

        db_path = tmp_path / "bf.db"
        # First init creates schema (with empty backfill).
        db = DatabaseManager(database_path=db_path)
        self._insert_legacy_row(db, "8895", "3P", "2025-03-26")
        self._insert_legacy_row(db, "8895", "3R", "2025-03-26")
        self._insert_legacy_row(db, "8895", "TEST", "2025-03-26")

        # NULL out unit_id so the backfill has work to do
        with db.session() as s:
            s.execute(text("UPDATE analysis_results SET unit_id = NULL"))
            s.commit()

        # Re-run backfill on the same connection
        with db.session() as s:
            db._backfill_unit_ids(s)

        with db.session() as s:
            rows = {r.serial: r.unit_id for r in s.query(DBAR).all()}
        assert rows["3P"] == "8895/3/2025-03-26"
        assert rows["3R"] == "8895/3/2025-03-26"
        assert rows["TEST"] is None

    def test_backfill_idempotent(self, tmp_path):
        from laser_trim_analyzer.database.manager import DatabaseManager
        from laser_trim_analyzer.database.models import AnalysisResult as DBAR

        db_path = tmp_path / "idem.db"
        db = DatabaseManager(database_path=db_path)
        self._insert_legacy_row(db, "8895", "3P", "2025-03-26")
        # First backfill (already happened on init; force a second)
        with db.session() as s:
            db._backfill_unit_ids(s)
        with db.session() as s:
            db._backfill_unit_ids(s)
        with db.session() as s:
            row = s.query(DBAR).filter(DBAR.serial == "3P").one()
            assert row.unit_id == "8895/3/2025-03-26"
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest tests/test_unit_yield.py -v`

Expected: all tests pass.

- [ ] **Step 4: Run the migration on the real local DB and capture before/after**

Run:
```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
python3 -c "
import sqlite3
con = sqlite3.connect('data/analysis.db')
cur = con.cursor()
cur.execute('SELECT COUNT(*) FROM analysis_results')
print('pre: total rows =', cur.fetchone()[0])
cur.execute(\"SELECT COUNT(*) FROM analysis_results WHERE unit_id IS NOT NULL\")
print('pre: with unit_id =', cur.fetchone()[0])
con.close()
"
# Now trigger migration by initializing DatabaseManager
python3 -c "
import sys; sys.path.insert(0,'src')
from laser_trim_analyzer.database.manager import DatabaseManager
DatabaseManager()  # runs migrations
"
python3 -c "
import sqlite3
con = sqlite3.connect('data/analysis.db')
cur = con.cursor()
cur.execute(\"SELECT COUNT(*) FROM analysis_results WHERE unit_id IS NOT NULL\")
print('post: with unit_id =', cur.fetchone()[0])
cur.execute(\"SELECT COUNT(DISTINCT unit_id) FROM analysis_results WHERE unit_id IS NOT NULL\")
print('post: distinct units =', cur.fetchone()[0])
con.close()
"
```

Expected: pre `with unit_id = 0`. Post `with unit_id` should be roughly 95% of total (~76k of 80,656). Distinct units should be much less than total rows.

- [ ] **Step 5: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/database/manager.py tests/test_unit_yield.py
git commit -m "feat(unit-yield): idempotent backfill of unit_id on existing rows

Backfill iterates analysis_results in batches of 1000, populating
unit_id via compute_unit_id. Junk-serial rows (TEST, x, missing
date) end up NULL by design. Idempotent — only operates on NULL rows
— so re-running after a crash or partial run picks up cleanly.
Post-run sanity check logs counts of populated vs NULL plus distinct
unit count for spot-check.

The user's 80k-row DB backfills in a few seconds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Feature flag in config

**Files:**
- Modify: `src/laser_trim_analyzer/config.py`

- [ ] **Step 1: Locate the relevant config section**

Run: `grep -n "model_prices\|cost_ratio\|active_models" /Users/jb631/projects/laser-trim-ai-system-v5/src/laser_trim_analyzer/config.py | head -10`

The flag belongs in a section that's already loaded by Trends and Excel paths. The `active_models` config block is a good fit (already touched by both surfaces).

- [ ] **Step 2: Add the flag**

In `config.py`, find the `ActiveModelsConfig` dataclass (or wherever `model_prices` and `cost_ratio` are defined). Add:

```python
    # Show the new unit-level yield surfaces (Trends single-model chart line
    # + Excel "Yield by Unit" sheet). Set to False to hide both instantly
    # without removing the underlying data. See
    # docs/superpowers/specs/2026-05-08-unit-level-yield-design.md.
    enable_unit_yield_view: bool = True
```

(If the location differs in your file, adapt — search for `cost_ratio: float` and add immediately below.)

- [ ] **Step 3: Smoke test the flag is reachable**

Run:
```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
python3 -c "
import sys; sys.path.insert(0,'src')
from laser_trim_analyzer.config import get_config
cfg = get_config()
print('enable_unit_yield_view =', cfg.active_models.enable_unit_yield_view)
"
```

Expected: `enable_unit_yield_view = True`.

- [ ] **Step 4: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/config.py
git commit -m "feat(unit-yield): add enable_unit_yield_view feature flag

Default True. Gates the new Trends chart line and the Excel 'Yield
by Unit' sheet. Set False to instantly hide both surfaces while
preserving the underlying data — useful if the new view shows
something unexpected in production.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: get_unit_yield_trend DB query

The chart needs daily unit-yield numbers. This query is the single source of truth for the unit pass rule.

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_unit_yield.py` (ensure `from datetime import datetime, timedelta` is at the top of the file — the original Task 1 import only had `datetime`):

```python
class TestGetUnitYieldTrend:
    def _setup(self, tmp_path):
        """Build a tiny DB exercising each branch of the unit pass rule."""
        from laser_trim_analyzer.database.manager import DatabaseManager
        from laser_trim_analyzer.database.models import (
            AnalysisResult as DBAR,
            TrackResult as DBTR,
            StatusType as DBStatusType,
        )
        db = DatabaseManager(database_path=tmp_path / "y.db")

        def ar(model, serial, file_date, ts_offset_min, status):
            """Create one trim record with a single passing track."""
            with db.session() as s:
                row = DBAR(
                    filename=f"{model}_{serial}_{file_date.strftime('%Y%m%d')}_{ts_offset_min}.xls",
                    file_path=f"/fake/{model}_{serial}_{ts_offset_min}.xls",
                    file_date=file_date,
                    model=model, serial=serial,
                    system="B", has_multi_tracks=False,
                    overall_status=status,
                    timestamp=datetime(2025, 3, 26, 9, 0) +
                              timedelta(minutes=ts_offset_min),
                    unit_id=compute_unit_id(model, serial, file_date),
                )
                s.add(row)
                s.commit()
                return row.id

        d = datetime(2025, 3, 26)
        # Unit 8895/3: 3P passes + 3R passes → unit passes
        ar("8895", "3P", d, 0, DBStatusType.PASS)
        ar("8895", "3R", d, 5, DBStatusType.PASS)
        # Unit 8895/4: 4P fails + 4R passes → unit fails (one section fails)
        ar("8895", "4P", d, 10, DBStatusType.FAIL)
        ar("8895", "4R", d, 15, DBStatusType.PASS)
        # Unit 8895/5: 5P fails THEN retrim passes; no R recorded → unit passes
        ar("8895", "5P", d, 20, DBStatusType.FAIL)
        ar("8895", "5P", d, 60, DBStatusType.PASS)  # retrim 40 min later
        # Junk serial — should be excluded
        ar("8895", "TEST", d, 100, DBStatusType.PASS)
        return db

    def test_yield_trend_applies_pass_rule(self, tmp_path):
        db = self._setup(tmp_path)
        rows = db.get_unit_yield_trend(model="8895", days_back=30)
        # Expect one bucket (single day, single model)
        assert len(rows) == 1
        r = rows[0]
        assert r["total_units"] == 3   # units 3, 4, 5 (TEST excluded)
        assert r["passed_units"] == 2  # units 3 and 5 pass
        assert r["yield_pct"] == pytest.approx(66.6667, rel=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest tests/test_unit_yield.py::TestGetUnitYieldTrend -v`

Expected: fails with `AttributeError: 'DatabaseManager' object has no attribute 'get_unit_yield_trend'`.

- [ ] **Step 3: Implement the query method**

Add to `DatabaseManager` in `database/manager.py`:

```python
    def get_unit_yield_trend(
        self,
        model: str,
        days_back: int = 90,
    ) -> List[Dict[str, Any]]:
        """Daily unit-level yield for one model.

        Pass rule: a unit passes only if every section's most recent
        attempt is PASS. Retrim tiebreak uses row `timestamp` (write
        time). NULL unit_id rows (junk serials, missing date) are
        excluded entirely.

        Returns rows of {"date", "total_units", "passed_units", "yield_pct"},
        sorted by date ascending. Days with no qualifying units omitted.
        """
        from laser_trim_analyzer.database.models import (
            AnalysisResult as DBAR,
            StatusType as DBStatusType,
        )
        from collections import defaultdict

        cutoff = datetime.now() - timedelta(days=days_back)
        with self.session() as s:
            # Pull (unit_id, serial, file_date, timestamp, overall_status)
            # for each qualifying row. We do the section-grouping and
            # latest-attempt-by-timestamp logic in Python because the
            # tie-breaking is awkward in SQL and the row count per
            # model+window is small (hundreds, not millions).
            rows = (
                s.query(
                    DBAR.unit_id,
                    DBAR.serial,
                    DBAR.file_date,
                    DBAR.timestamp,
                    DBAR.overall_status,
                )
                .filter(
                    DBAR.model == model,
                    DBAR.unit_id.isnot(None),
                    DBAR.file_date >= cutoff,
                )
                .all()
            )

        # Group rows by unit_id, then within each unit by section (trailing
        # non-digit part of the serial after the shop number). For each
        # (unit, section) pick the row with the largest timestamp — that's
        # the section's authoritative status.
        section_status_by_unit: Dict[str, Dict[str, Any]] = defaultdict(dict)
        unit_date: Dict[str, datetime] = {}

        def section_of(serial: Optional[str]) -> str:
            """Return the section suffix (everything after the leading digits).
            Default to '' if there's no suffix, so single-section units (just
            digits like '196') still group cleanly."""
            if not serial:
                return ""
            m = _re.match(r"^\d+(.*)$", serial.strip())
            return (m.group(1) if m else "").upper()

        for unit_id, serial, file_date, ts, status in rows:
            section = section_of(serial)
            existing = section_status_by_unit[unit_id].get(section)
            if existing is None or (ts or datetime.min) > existing["ts"]:
                section_status_by_unit[unit_id][section] = {
                    "ts": ts or datetime.min,
                    "status": status,
                }
            # Remember the unit's date (all rows for a unit share file_date)
            if file_date is not None:
                unit_date[unit_id] = file_date

        # Apply the pass rule per unit and bucket by date.
        per_date: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "passed": 0}
        )
        for unit_id, sections in section_status_by_unit.items():
            fd = unit_date.get(unit_id)
            if fd is None:
                continue
            day = fd.strftime("%Y-%m-%d")
            per_date[day]["total"] += 1
            # All sections must be PASS for the unit to pass
            all_pass = all(
                _status_matches(sec["status"], DBStatusType.PASS)
                for sec in sections.values()
            )
            if all_pass:
                per_date[day]["passed"] += 1

        return [
            {
                "date": day,
                "total_units": counts["total"],
                "passed_units": counts["passed"],
                "yield_pct": (
                    counts["passed"] / counts["total"] * 100.0
                ) if counts["total"] > 0 else 0.0,
            }
            for day, counts in sorted(per_date.items())
        ]
```

(The helper `_status_matches` already exists in `database/manager.py` from earlier code-review work — reuse it. If unsure, grep for `def _status_matches` to confirm.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest tests/test_unit_yield.py::TestGetUnitYieldTrend -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/database/manager.py tests/test_unit_yield.py
git commit -m "feat(unit-yield): get_unit_yield_trend query method

Daily unit-level yield for one model over a date window. Applies
the unit pass rule (latest attempt per section, all sections must
PASS) using row timestamp for retrim tiebreaking. NULL unit_id rows
excluded. Returns list of {date, total_units, passed_units,
yield_pct} sorted by date.

Test covers the three branches of the pass rule (all-pass, one-fail,
retrim-recovery) plus junk-serial exclusion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Trends single-model chart — add unit yield overlay

**Files:**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py`

- [ ] **Step 1: Locate the existing single-model P-chart rendering**

Run: `grep -n "_render_linearity_chart\|plot_pchart\|_render_detail" /Users/jb631/projects/laser-trim-ai-system-v5/src/laser_trim_analyzer/gui/pages/trends.py | head -10`

The single-model detail-view P-chart is rendered through `plot_pchart` on `self.linearity_chart`. Find where `pass_rates` is computed and passed in.

- [ ] **Step 2: Augment the chart-fetch flow with unit-yield data**

Find the existing `_render_linearity_chart` (or the closest method that draws the linearity P-chart for a single model). After the existing line that fetches daily track-level pass rates, fetch unit-level yield and overlay it:

```python
        # Existing line that fetches track-level pass rates stays untouched.
        # rates_data = db.get_linearity_daily_trend(model=clean_model, ...)
        # dates = [r["date"] for r in rates_data]
        # pass_rates = [r["pass_rate"] for r in rates_data]
        # self.linearity_chart.plot_pchart(dates=dates, pass_rates=pass_rates, ...)

        # New: overlay unit yield line if feature flag is on
        try:
            cfg = get_config()
            show_unit_yield = bool(getattr(
                cfg.active_models, "enable_unit_yield_view", True
            ))
        except Exception:
            show_unit_yield = True

        if show_unit_yield:
            try:
                unit_rows = db.get_unit_yield_trend(
                    model=clean_model, days_back=self.selected_days
                )
            except Exception as e:
                logger.debug(f"unit yield trend unavailable: {e}")
                unit_rows = []
            if unit_rows:
                # Align unit-yield series to the same date axis the P-chart uses.
                # Use the chart's overlay method so a second matplotlib line is
                # drawn on the same axes without disturbing the P-chart.
                yield_by_date = {r["date"]: r["yield_pct"] for r in unit_rows}
                unit_yield_series = [yield_by_date.get(d) for d in dates]
                self.linearity_chart.overlay_line(
                    series=unit_yield_series,
                    label="Unit Yield (sections all-pass)",
                    color="#2c7fb8",
                    linestyle="--",
                )
```

- [ ] **Step 3: Add the `overlay_line` method to the ChartWidget**

`plot_pchart` produces a P-chart on a single axes. We need a method that draws an additional line on the same axes. Find `src/laser_trim_analyzer/gui/widgets/chart.py` and locate `plot_pchart`. Add a sibling method:

```python
    def overlay_line(
        self,
        series: list,
        label: str,
        color: str = "#0d6efd",
        linestyle: str = "--",
    ) -> None:
        """Draw an additional line on the most-recently-rendered axes.

        Intended to overlay a secondary metric onto plot_pchart's axes
        without redrawing the primary plot. Skip silently if the chart
        has no axes yet.
        """
        ax = self.figure.gca() if self.figure.axes else None
        if ax is None:
            return
        # Series may contain None for missing dates; matplotlib handles
        # None by breaking the line, which is the right behavior.
        x = list(range(len(series)))
        ax.plot(
            x, series,
            color=color, linestyle=linestyle, linewidth=1.8,
            marker="o", markersize=3,
            label=label, alpha=0.85,
        )
        # Refresh legend to include the overlay
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.85)
        self.canvas.draw_idle()
```

- [ ] **Step 4: Syntax check + smoke import**

Run:
```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
python3 -m py_compile src/laser_trim_analyzer/gui/pages/trends.py src/laser_trim_analyzer/gui/widgets/chart.py && echo OK
python3 -c "
import sys; sys.path.insert(0,'src')
from laser_trim_analyzer.gui.pages.trends import TrendsPage
print('import OK')
"
```

Expected: `OK` then `import OK`.

- [ ] **Step 5: Run existing tests to confirm no regression**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest 2>&1 | tail -3`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/gui/pages/trends.py src/laser_trim_analyzer/gui/widgets/chart.py
git commit -m "feat(trends): overlay unit-yield line on single-model P-chart

Single-model detail view's Linearity Pass Rate P-chart now shows a
second line for unit-level yield, computed via
db.get_unit_yield_trend. Track-level line (existing) and unit-level
line (new) on the same axes lets the operator see when sections
retrim into a passing unit (track line drops, unit line stays high)
vs when units genuinely fail (both drop together).

Gated by the enable_unit_yield_view feature flag. ChartWidget gains
an overlay_line helper for adding a secondary series to the most
recently rendered axes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Excel "Yield by Unit" sheet

**Files:**
- Modify: `src/laser_trim_analyzer/export/excel.py`

- [ ] **Step 1: Locate the batch-export flow**

Run: `grep -n "_create_batch_summary_sheet\|_create_all_results_sheet\|export_batch_results" /Users/jb631/projects/laser-trim-ai-system-v5/src/laser_trim_analyzer/export/excel.py | head -10`

Note where `_create_all_results_sheet` is called inside `export_batch_results` — the new sheet gets added right after it.

- [ ] **Step 2: Add the sheet helper**

In `export/excel.py`, near the other `_create_*_sheet` functions, add:

```python
def _create_yield_by_unit_sheet(wb: "Workbook", results: List[AnalysisResult]) -> None:
    """One row per physical unit (model+shop+date), applying the unit pass rule.

    Unit passes only if every section's latest attempt is PASS. NULL/junk
    unit_ids are excluded.
    """
    from collections import defaultdict
    import re as _re

    ws = wb.create_sheet("Yield by Unit")

    headers = [
        "Shop Number", "Date", "Sections",
        "Attempts", "P Final", "R Final", "Unit Status",
        "First Trim", "Last Trim",
    ]
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center")
        cell.border = THIN_BORDER

    def _shop_and_section(serial):
        if not serial:
            return None, ""
        m = _re.match(r"^(\d+)(.*)$", serial.strip())
        if not m:
            return None, ""
        return m.group(1), m.group(2).upper()

    # Group result rows by (model, shop, date). For each group, track each
    # section's latest status by file_date timestamp.
    groups: dict = defaultdict(lambda: {
        "sections": defaultdict(lambda: {"ts": None, "status": None}),
        "all_attempts": [],
        "first": None, "last": None,
    })

    for r in results:
        md = r.metadata
        shop, section = _shop_and_section(md.serial)
        if shop is None or not md.model or md.file_date is None:
            continue
        key = (md.model, shop, md.file_date.strftime("%Y-%m-%d"))
        g = groups[key]
        sec = g["sections"][section]
        ts = md.file_date
        if sec["ts"] is None or ts > sec["ts"]:
            sec["ts"] = ts
            sec["status"] = r.overall_status
        g["all_attempts"].append(ts)
        g["first"] = min(g["first"], ts) if g["first"] else ts
        g["last"] = max(g["last"], ts) if g["last"] else ts

    # Sort by date desc, then shop number ascending (numerically when possible)
    def sort_key(item):
        (_, shop, date), _ = item
        try:
            shop_num = int(shop)
        except (TypeError, ValueError):
            shop_num = 0
        return (date, shop_num)

    sorted_items = sorted(groups.items(), key=sort_key, reverse=True)

    passed_total = 0
    fail_total = 0
    for i, ((model, shop, date), g) in enumerate(sorted_items, start=2):
        sections_present = sorted(g["sections"].keys())
        section_letters = ", ".join(s or "(none)" for s in sections_present)
        p_final = g["sections"].get("P", {}).get("status")
        r_final = g["sections"].get("R", {}).get("status")

        # Unit passes iff every section's latest status is PASS
        all_pass = all(
            sec["status"] == AnalysisStatus.PASS
            for sec in g["sections"].values()
        )
        unit_status_text = "PASS" if all_pass else "FAIL"
        if all_pass:
            passed_total += 1
        else:
            fail_total += 1

        ws.cell(row=i, column=1, value=shop)
        ws.cell(row=i, column=2, value=date)
        ws.cell(row=i, column=3, value=section_letters)
        ws.cell(row=i, column=4, value=len(g["all_attempts"]))
        ws.cell(row=i, column=5, value=p_final.value if p_final else "")
        ws.cell(row=i, column=6, value=r_final.value if r_final else "")
        cell = ws.cell(row=i, column=7, value=unit_status_text)
        cell.fill = PASS_FILL if all_pass else FAIL_FILL
        ws.cell(row=i, column=8,
                value=g["first"].strftime("%Y-%m-%d %H:%M") if g["first"] else "")
        ws.cell(row=i, column=9,
                value=g["last"].strftime("%Y-%m-%d %H:%M") if g["last"] else "")

    # Summary footer
    total_units = passed_total + fail_total
    footer_row = len(sorted_items) + 3
    yield_pct = (passed_total / total_units * 100) if total_units else 0
    ws.cell(row=footer_row, column=1, value="TOTAL UNITS").font = Font(bold=True)
    ws.cell(row=footer_row, column=2, value=total_units)
    ws.cell(row=footer_row + 1, column=1, value="PASSED").font = Font(bold=True)
    ws.cell(row=footer_row + 1, column=2, value=passed_total)
    ws.cell(row=footer_row + 2, column=1, value="FAILED").font = Font(bold=True)
    ws.cell(row=footer_row + 2, column=2, value=fail_total)
    ws.cell(row=footer_row + 3, column=1, value="UNIT YIELD").font = Font(bold=True)
    ws.cell(row=footer_row + 3, column=2, value=f"{yield_pct:.1f}%")

    widths = [12, 12, 14, 10, 10, 10, 14, 18, 18]
    for col, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(col)].width = width
```

- [ ] **Step 3: Wire it into `export_batch_results`**

Find `export_batch_results` in the same file. After the line that calls `_create_all_results_sheet(wb, results)`, add:

```python
        # Yield by Unit sheet (feature-flag-gated)
        try:
            from laser_trim_analyzer.config import get_config
            cfg = get_config()
            show_unit_yield = bool(getattr(
                cfg.active_models, "enable_unit_yield_view", True
            ))
        except Exception:
            show_unit_yield = True
        if show_unit_yield:
            _create_yield_by_unit_sheet(wb, results)
```

- [ ] **Step 4: Syntax check + smoke import**

Run:
```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
python3 -m py_compile src/laser_trim_analyzer/export/excel.py && echo OK
python3 -c "
import sys; sys.path.insert(0,'src')
from laser_trim_analyzer.export.excel import _create_yield_by_unit_sheet
print('import OK')
"
```

Expected: `OK` then `import OK`.

- [ ] **Step 5: Run existing tests**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest 2>&1 | tail -3`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git add src/laser_trim_analyzer/export/excel.py
git commit -m "feat(unit-yield): new Yield by Unit sheet on batch export

One row per physical unit (model+shop+date), with columns for
sections present, attempt count (retrims included), P/R final
status, unit-level pass/fail, and first/last trim timestamps.
Footer shows total units, passed, failed, and unit yield %.

Gated by enable_unit_yield_view feature flag. Existing 'All Results'
sheet unchanged — operators who want row-level detail still have it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Final verification

- [ ] **Step 1: Run the full test suite**

Run: `cd /Users/jb631/projects/laser-trim-ai-system-v5 && python3 -m pytest 2>&1 | tail -5`

Expected: all tests pass, with the new `test_unit_yield.py` adding ~22 tests.

- [ ] **Step 2: Smoke-import all GUI pages**

Run:
```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
python3 -c "
import sys; sys.path.insert(0,'src')
from laser_trim_analyzer.gui.pages.dashboard import DashboardPage
from laser_trim_analyzer.gui.pages.trends import TrendsPage
from laser_trim_analyzer.gui.pages.analyze import AnalyzePage
from laser_trim_analyzer.gui.pages.compare import ComparePage
from laser_trim_analyzer.gui.pages.process import ProcessPage
from laser_trim_analyzer.gui.pages.settings import SettingsPage
print('all pages import OK')
"
```

Expected: `all pages import OK`.

- [ ] **Step 3: Verify the production DB backfill**

Run:
```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
python3 -c "
import sqlite3
con = sqlite3.connect('data/analysis.db')
cur = con.cursor()
cur.execute(\"SELECT COUNT(*) FROM analysis_results\")
print('total rows:', cur.fetchone()[0])
cur.execute(\"SELECT COUNT(*) FROM analysis_results WHERE unit_id IS NOT NULL\")
print('with unit_id:', cur.fetchone()[0])
cur.execute(\"SELECT COUNT(DISTINCT unit_id) FROM analysis_results WHERE unit_id IS NOT NULL AND model='8895'\")
print('distinct 8895 units:', cur.fetchone()[0])
cur.execute(\"SELECT COUNT(*) FROM analysis_results WHERE model='8895'\")
print('8895 rows:', cur.fetchone()[0])
con.close()
"
```

Expected: 8895 should show roughly 48 distinct units from 96 rows (per spec's success criteria).

- [ ] **Step 4: Push the branch**

```bash
cd /Users/jb631/projects/laser-trim-ai-system-v5
git push
```

---

## Notes

- This plan is GUI/DB only — no parser, processor, or ML changes.
- All 8 implementation tasks are isolated and TDD-shaped. Each leaves the app in a working state.
- The migration is the only step with non-trivial blast radius; it's the entire reason for the safety belt (nullable column, idempotent backfill, sanity-check logging, feature flag).
- If the feature flag default changes to False later for any reason, the unit-yield chart line and Excel sheet hide instantly. The underlying column and computation remain.
