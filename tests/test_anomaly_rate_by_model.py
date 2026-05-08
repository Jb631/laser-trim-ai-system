"""Tests for DatabaseManager.get_anomaly_rate_by_model."""
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
    return DatabaseManager(database_path=tmp_path / "anomaly.db")


def _add_track(db, model: str, file_date: datetime, *, is_anomaly: bool, suffix: str = ""):
    """Insert one analysis row with one track row."""
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
            is_anomaly=is_anomaly,
        )
        session.add(track)


def test_returns_empty_when_no_data(db):
    rows = db.get_anomaly_rate_by_model(days_back=90, min_samples=1)
    assert rows == []


def test_counts_anomalies_per_model(db):
    now = datetime.utcnow()
    # Model A: 8 normal + 2 anomaly -> 20% anomaly rate
    for i in range(8):
        _add_track(db, "MODEL-A", now - timedelta(days=1), is_anomaly=False, suffix=f"n{i}")
    for i in range(2):
        _add_track(db, "MODEL-A", now - timedelta(days=1), is_anomaly=True, suffix=f"a{i}")
    # Model B: 10 normal -> 0% rate
    for i in range(10):
        _add_track(db, "MODEL-B", now - timedelta(days=1), is_anomaly=False, suffix=f"b{i}")

    rows = db.get_anomaly_rate_by_model(days_back=30, min_samples=5)
    by_model = {r["model"]: r for r in rows}
    assert "MODEL-A" in by_model
    assert by_model["MODEL-A"]["anomaly_count"] == 2
    assert by_model["MODEL-A"]["total_tracks"] == 10
    assert by_model["MODEL-A"]["anomaly_rate"] == pytest.approx(20.0)
    assert by_model["MODEL-B"]["anomaly_count"] == 0
    assert by_model["MODEL-B"]["anomaly_rate"] == pytest.approx(0.0)


def test_filters_by_min_samples(db):
    now = datetime.utcnow()
    for i in range(3):
        _add_track(db, "TINY", now - timedelta(days=1), is_anomaly=True, suffix=f"t{i}")

    rows = db.get_anomaly_rate_by_model(days_back=30, min_samples=10)
    assert all(r["model"] != "TINY" for r in rows)


def test_filters_by_days_back(db):
    now = datetime.utcnow()
    # Older than the window
    for i in range(20):
        _add_track(db, "OLD", now - timedelta(days=120), is_anomaly=True, suffix=f"o{i}")

    rows = db.get_anomaly_rate_by_model(days_back=30, min_samples=5)
    assert all(r["model"] != "OLD" for r in rows)


def test_sorted_by_rate_descending(db):
    now = datetime.utcnow()
    # MODEL-LOW: 1/20 = 5%
    for i in range(19):
        _add_track(db, "MODEL-LOW", now - timedelta(days=1), is_anomaly=False, suffix=f"l{i}")
    _add_track(db, "MODEL-LOW", now - timedelta(days=1), is_anomaly=True, suffix="la")
    # MODEL-HIGH: 5/10 = 50%
    for i in range(5):
        _add_track(db, "MODEL-HIGH", now - timedelta(days=1), is_anomaly=False, suffix=f"hn{i}")
    for i in range(5):
        _add_track(db, "MODEL-HIGH", now - timedelta(days=1), is_anomaly=True, suffix=f"ha{i}")

    rows = db.get_anomaly_rate_by_model(days_back=30, min_samples=5)
    models_in_order = [r["model"] for r in rows]
    assert models_in_order.index("MODEL-HIGH") < models_in_order.index("MODEL-LOW")
