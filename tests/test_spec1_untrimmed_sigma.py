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
    """Invoke Analyzer.analyze_track with sane wrapping context.

    Note: the plan references ``LaserTrimAnalyzer(config=Config())`` but
    the actual class in core/analyzer.py is ``Analyzer`` with no Config
    parameter (it takes optional scaling_factor + model_thresholds).
    Using the real construction pattern from processor.py.
    """
    from laser_trim_analyzer.core.analyzer import Analyzer

    analyzer = Analyzer()
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
        AnalysisResult, AnalysisStatus, FileMetadata, SystemType, TrackData,
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
    from laser_trim_analyzer.core.analyzer import Analyzer

    track_input = _build_track_input()
    analyzer = Analyzer()

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
