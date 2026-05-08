"""Tests for the avg_error_reduction extension to get_trim_difficulty_by_model."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import (
    AnalysisResult,
    TrackResult,
    StatusType,
    SystemType,
    RiskCategory,
)


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(database_path=tmp_path / "trim_difficulty.db")


def _add_track(db, model, file_date, *, trim_passes, error_reduction, suffix=""):
    with db.session() as session:
        analysis = AnalysisResult(
            filename=f"{model}_{int(file_date.timestamp())}_{suffix}.xls",
            file_path=f"/fake/{model}_{suffix}.xls",
            file_date=file_date,
            model=model,
            serial=f"sn{int(file_date.timestamp())}_{suffix}",
            system=SystemType.A,
            has_multi_tracks=False,
            overall_status=StatusType.PASS,
        )
        session.add(analysis)
        session.flush()
        track = TrackResult(
            analysis_id=analysis.id,
            track_id="TRK1",
            status=StatusType.PASS,
            travel_length=10.0,
            linearity_spec=0.01,
            sigma_gradient=0.001,
            sigma_threshold=0.005,
            sigma_pass=True,
            optimal_offset=0.0,
            final_linearity_error_shifted=0.001,
            linearity_pass=True,
            linearity_fail_points=0,
            risk_category=RiskCategory.LOW,
            trim_pass_count=trim_passes,
            max_error_reduction_percent=error_reduction,
        )
        session.add(track)


def test_avg_error_reduction_present_when_data_exists(db):
    now = datetime.utcnow()
    for i, er in enumerate([40.0, 60.0, 50.0, 70.0, 80.0]):
        _add_track(db, "MODEL-X", now - timedelta(days=1),
                   trim_passes=2, error_reduction=er, suffix=f"x{i}")

    rows = db.get_trim_difficulty_by_model(
        days_back=30, min_units=5, limit=10
    )
    by_model = {r["model"]: r for r in rows}
    assert "MODEL-X" in by_model
    assert by_model["MODEL-X"]["avg_error_reduction"] == pytest.approx(60.0)


def test_avg_error_reduction_none_when_no_data(db):
    now = datetime.utcnow()
    for i in range(5):
        _add_track(db, "MODEL-Y", now - timedelta(days=1),
                   trim_passes=2, error_reduction=None, suffix=f"y{i}")

    rows = db.get_trim_difficulty_by_model(
        days_back=30, min_units=5, limit=10
    )
    by_model = {r["model"]: r for r in rows}
    assert by_model["MODEL-Y"]["avg_error_reduction"] is None
