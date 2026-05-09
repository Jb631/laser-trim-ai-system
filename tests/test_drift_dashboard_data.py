"""Tests for the four DB queries that feed the redesigned Drift tab."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAnalysisResult,
    TrackResult as DBTrackResult,
    SystemType as DBSystemType,
    StatusType as DBStatusType,
    RiskCategory as DBRiskCategory,
)


@pytest.fixture
def db(tmp_path):
    """Fresh on-disk SQLite per test."""
    return DatabaseManager(tmp_path / "test.db")


def _add_analysis(db, model, file_date, sigma_values=(0.01,), serial="0001"):
    """Helper: insert one analysis row with N tracks, each with given sigma."""
    with db.session() as s:
        ar = DBAnalysisResult(
            filename=f"{model}_{serial}_{file_date.strftime('%Y%m%d')}.xls",
            file_path=f"/fake/{model}_{serial}.xls",
            file_hash=f"{model}{serial}{file_date.timestamp()}",
            model=model,
            serial=serial,
            system=DBSystemType.B,
            file_date=file_date,
            timestamp=datetime.now(),
            overall_status=DBStatusType.PASS,
            has_multi_tracks=False,
            processing_time=0.1,
        )
        s.add(ar)
        s.flush()
        for i, sigma in enumerate(sigma_values):
            tr = DBTrackResult(
                analysis_id=ar.id,
                track_id=f"TRK{i+1}",
                status=DBStatusType.PASS,
                sigma_gradient=sigma,
                sigma_threshold=0.02,
                sigma_pass=True,
                travel_length=1.0,
                linearity_spec=0.01,
                risk_category=DBRiskCategory.LOW,
            )
            s.add(tr)
        s.commit()


def test_get_models_with_sigma_data_includes_models_with_sigma(db):
    today = datetime.now()
    _add_analysis(db, "7965", today - timedelta(days=2), sigma_values=(0.012,))
    _add_analysis(db, "8492", today - timedelta(days=5), sigma_values=(0.015,))

    models = db.get_models_with_sigma_data(days_back=30)
    assert "7965" in models
    assert "8492" in models


def test_get_models_with_sigma_data_excludes_old_models(db):
    today = datetime.now()
    _add_analysis(db, "RECENT", today - timedelta(days=2), sigma_values=(0.01,))
    _add_analysis(db, "OLD", today - timedelta(days=400), sigma_values=(0.01,))

    models = db.get_models_with_sigma_data(days_back=30)
    assert "RECENT" in models
    assert "OLD" not in models


def test_get_models_with_sigma_data_excludes_null_sigma(db):
    today = datetime.now()
    # Insert model with valid sigma
    _add_analysis(db, "VALID", today - timedelta(days=2), sigma_values=(0.015,))
    # Insert model with no sigma data at all (it won't appear because no tracks match)
    with db.session() as s:
        ar = DBAnalysisResult(
            filename="NOSIGMA_0001.xls",
            file_path="/fake/NOSIGMA_0001.xls",
            file_hash="nosigma",
            model="NOSIGMA",
            serial="0001",
            system=DBSystemType.B,
            file_date=today - timedelta(days=2),
            timestamp=datetime.now(),
            overall_status=DBStatusType.PASS,
            has_multi_tracks=False,
            processing_time=0.1,
        )
        s.add(ar)
        s.commit()

    models = db.get_models_with_sigma_data(days_back=30)
    assert "VALID" in models
    # NOSIGMA has no tracks, so it won't appear in the result
    assert "NOSIGMA" not in models
