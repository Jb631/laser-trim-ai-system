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


def _add_analysis(db, model, file_date, sigma_values=(0.01,), serial="0001",
                  trim_pass_count=None):
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
                trim_pass_count=trim_pass_count,
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


def test_process_drift_table_returns_delta_pct_and_series(tmp_path):
    db = DatabaseManager(tmp_path / "process_drift.db")

    today = datetime.now()
    # 25 baseline samples for 7965 (60+ days ago) at resistance 3.0
    for i in range(25):
        _add_analysis(
            db, "7965",
            today - timedelta(days=60 + i),
            sigma_values=(0.01,),
            serial=f"{i:04d}",
        )
    # Patch their untrimmed_resistance to baseline value
    with db.session() as s:
        rows = (
            s.query(DBTrackResult)
            .join(DBAnalysisResult)
            .filter(DBAnalysisResult.model == "7965")
            .all()
        )
        for tr in rows:
            tr.untrimmed_resistance = 3.0
        s.commit()

    # Add 6 recent samples with higher resistance
    for i in range(6):
        _add_analysis(
            db, "7965",
            today - timedelta(days=i + 1),
            sigma_values=(0.01,),
            serial=f"R{i:03d}",
        )
    with db.session() as s:
        rows = (
            s.query(DBTrackResult)
            .join(DBAnalysisResult)
            .filter(
                DBAnalysisResult.model == "7965",
                DBAnalysisResult.serial.like("R%"),
            )
            .all()
        )
        for tr in rows:
            tr.untrimmed_resistance = 3.3
        s.commit()

    rows = db.get_process_drift_table(
        metric="untrimmed_resistance",
        baseline_days=90,
        recent_days=14,
    )
    assert rows, "expected at least one model row"
    r = next(x for x in rows if x["model"] == "7965")
    # Δ% should be +10% (3.0 → 3.3)
    assert 8.0 < r["delta_pct"] < 12.0
    # series is a list of (date_iso, value) tuples — at least one point
    assert isinstance(r["series"], list)
    assert len(r["series"]) >= 1
    pt = r["series"][0]
    assert isinstance(pt[0], str)
    assert isinstance(pt[1], float)


def test_get_model_drift_dashboard_returns_all_panels(tmp_path):
    db = DatabaseManager(tmp_path / "model_dashboard.db")

    today = datetime.now()
    for i in range(15):
        _add_analysis(
            db, "8492",
            today - timedelta(days=30 - i),
            sigma_values=(0.01 + i * 0.0001,),
            serial=f"{i:04d}",
        )

    data = db.get_model_drift_dashboard(model="8492", days_back=60)
    assert data["model"] == "8492"
    # Sigma panel
    assert "sigma_series" in data
    assert isinstance(data["sigma_series"], list)
    assert len(data["sigma_series"]) >= 10
    # Three process panels keyed by metric
    for metric in ("untrimmed_resistance",
                   "measured_electrical_angle",
                   "trim_pass_count"):
        assert metric in data["process"]
        panel = data["process"][metric]
        assert "series" in panel
        assert "baseline_mean" in panel
        assert "recent_mean" in panel


def test_get_model_drift_dashboard_returns_empty_for_unknown_model(tmp_path):
    db = DatabaseManager(tmp_path / "model_dashboard_empty.db")
    data = db.get_model_drift_dashboard(model="NOPE", days_back=60)
    assert data["model"] == "NOPE"
    assert data["sigma_series"] == []
    assert data["unit_count"] == 0


def test_get_drift_state_for_models_pulls_drift_start_date(tmp_path):
    db = DatabaseManager(tmp_path / "drift_state.db")

    today = datetime.now()
    _add_analysis(db, "7965", today - timedelta(days=2), sigma_values=(0.012,))

    # Insert ModelMLState row marking 7965 as drifting
    from laser_trim_analyzer.database.models import ModelMLState
    with db.session() as s:
        s.add(ModelMLState(
            model="7965",
            is_drifting=True,
            drift_direction="up",
            drift_start_date=today - timedelta(days=10),
            updated_date=today - timedelta(days=1),
        ))
        s.commit()

    data = db.get_drift_state_for_models(days_back=30)
    assert "7965" in data
    row = data["7965"]
    assert row["is_drifting"] is True
    assert row["direction"] == "up"
    assert row["drift_start_date"] is not None
    # Sigma trend present
    assert isinstance(row["sigma_series"], list)
    assert len(row["sigma_series"]) >= 1


def test_get_drift_state_for_models_includes_stable_models(tmp_path):
    db = DatabaseManager(tmp_path / "drift_state_stable.db")

    today = datetime.now()
    _add_analysis(db, "8275", today - timedelta(days=1), sigma_values=(0.01,))
    # No ModelMLState row → treated as no-baseline / stable.

    data = db.get_drift_state_for_models(days_back=30)
    assert "8275" in data
    row = data["8275"]
    assert row["is_drifting"] is False
    assert row["drift_start_date"] is None


def test_get_model_drift_dashboard_includes_baseline_cutoff_date(db):
    """baseline_cutoff_date must be `now - recent_days`, in ISO format,
    so all four panels can draw the same vertical reference line."""
    from datetime import datetime, timedelta

    _add_analysis(db, "TEST-A", datetime.now() - timedelta(days=5),
                  sigma_values=(0.01,))

    result = db.get_model_drift_dashboard(
        model="TEST-A", days_back=90, recent_days=14
    )

    assert "baseline_cutoff_date" in result, (
        "Top-level dict must include baseline_cutoff_date"
    )
    cutoff = datetime.fromisoformat(result["baseline_cutoff_date"])
    expected = datetime.now() - timedelta(days=14)
    # within a small wall-clock delta of expected
    assert abs((cutoff - expected).total_seconds()) < 5


def test_get_model_drift_dashboard_includes_retrim_rate_series(db):
    """retrim_rate_series is one (iso_date, 0_or_1) per non-NULL row,
    where 1 means trim_pass_count > 1."""
    from datetime import datetime, timedelta

    # Three rows: first two needed only 1 pass, third needed 2.
    base_date = datetime.now() - timedelta(days=10)
    for i, tpc in enumerate([1, 1, 2]):
        _add_analysis(
            db, "TEST-B",
            base_date + timedelta(hours=i),
            sigma_values=(0.01,),
            serial=f"{i:04d}",
            trim_pass_count=tpc,
        )

    result = db.get_model_drift_dashboard(
        model="TEST-B", days_back=90, recent_days=14
    )

    series = result["process"]["retrim_rate_series"]
    # 3 rows in, 3 entries out, values 0,0,1
    assert len(series) == 3
    assert [v for _, v in series] == [0, 0, 1]


def test_get_model_drift_dashboard_retrim_rate_skips_null_trim_pass_count(db):
    """Rows with NULL trim_pass_count (pre-feature data) must be excluded
    from retrim_rate_series so the panel can detect the all-NULL case."""
    from datetime import datetime, timedelta

    _add_analysis(
        db, "TEST-C",
        datetime.now() - timedelta(days=5),
        sigma_values=(0.01,),
        trim_pass_count=None,
    )

    result = db.get_model_drift_dashboard(
        model="TEST-C", days_back=90, recent_days=14
    )

    assert result["process"]["retrim_rate_series"] == []


def test_get_models_with_sigma_data_excludes_untrimmed_only(tmp_path):
    """Models whose only records are UNTRIMMED (test-sweep, no laser trim)
    must not appear in the sigma-data list — they have no sigma to drift
    over. Without this, the Drift filter dropdown would offer models that
    can't actually be analyzed."""
    db = DatabaseManager(tmp_path / "untrimmed.db")
    today = datetime.now()
    # One untrimmed-only model: track row written with sigma_gradient=None
    # and status=UNTRIMMED (the path the processor uses for test-sweep
    # files). Should NOT appear in the dropdown.
    with db.session() as s:
        ar = DBAnalysisResult(
            filename="UNTRIMMED-MODEL_0001.xls",
            file_path="/fake/UNTRIMMED-MODEL_0001.xls",
            file_hash="untrimmed-only-1",
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
        s.flush()
        tr = DBTrackResult(
            analysis_id=ar.id,
            track_id="TRK1",
            status=DBStatusType.UNTRIMMED,
            sigma_gradient=None,  # No trim ran → no sigma.
            sigma_threshold=None,
            sigma_pass=None,
            travel_length=1.0,
            linearity_spec=0.01,
            risk_category=DBRiskCategory.LOW,
        )
        s.add(tr)
        s.commit()
    # Add a normal model alongside so we know the function still returns
    # something — it just shouldn't return the untrimmed-only one.
    _add_analysis(db, "NORMAL", today - timedelta(days=1), sigma_values=(0.01,))

    models = db.get_models_with_sigma_data(days_back=30)
    assert "NORMAL" in models
    assert "UNTRIMMED-MODEL" not in models
