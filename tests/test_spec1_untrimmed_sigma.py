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
