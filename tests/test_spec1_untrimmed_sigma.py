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
