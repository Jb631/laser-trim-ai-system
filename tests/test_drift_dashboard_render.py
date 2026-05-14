"""End-to-end smoke for the redesigned single-model drift dashboard.

Builds a small temp DB with one model and exercises the helpers in the same
order the GUI renderer does. Verifies no crash, sigma violation count matches
the planted out-of-control rows, and retrim-rate bucketing produces the
expected percentages.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAnalysisResult,
    TrackResult as DBTrackResult,
    SystemType as DBSystemType,
    StatusType as DBStatusType,
    RiskCategory as DBRiskCategory,
)
from laser_trim_analyzer.gui.pages.trends import (
    _compute_buckets, _draw_smoothed_panel, _draw_sigma_panel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_analysis(db, model, file_date, sigma_values=(0.01,), serial="0001",
                  trim_pass_count=None):
    """Insert one analysis row with N tracks, each with the given sigma."""
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


class _StubDetector:
    has_baseline = True

    def get_control_limits(self):
        # UCL = 0.020 — planted values >0.020 below will count as violations
        return (0.005, 0.012, 0.020)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def db_with_drifting_model(tmp_path):
    """50 in-control sigma rows + 5 out-of-control, mixed retrim counts."""
    db = DatabaseManager(tmp_path / "test.db")
    base_date = datetime.now() - timedelta(days=20)

    # 50 in-control rows (sigma 0.010, trim_pass_count 1)
    for i in range(50):
        _add_analysis(
            db, "DR-MODEL",
            base_date + timedelta(hours=i),
            sigma_values=(0.010,),
            serial=f"{i:04d}",
            trim_pass_count=1,
        )

    # 5 out-of-control rows (sigma 0.030, trim_pass_count 2 — retrims)
    for i in range(5):
        _add_analysis(
            db, "DR-MODEL",
            base_date + timedelta(days=15, hours=i),
            sigma_values=(0.030,),
            serial=f"X{i:03d}",
            trim_pass_count=2,
        )

    return db


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_drift_dashboard_end_to_end(db_with_drifting_model):
    data = db_with_drifting_model.get_model_drift_dashboard(
        model="DR-MODEL", days_back=90, recent_days=14,
    )

    # Top-level shape (Task 1 contract)
    assert "baseline_cutoff_date" in data
    assert "retrim_rate_series" in data["process"]

    # Sigma panel: 5 dots above UCL = 5 violations
    fig, ax = plt.subplots()
    violations = _draw_sigma_panel(
        ax, sigma_series=data["sigma_series"],
        detector=_StubDetector(),
        baseline_cutoff_bucket_index=0,
        n_per_bucket=50,
    )
    assert violations == 5
    plt.close(fig)

    # Retrim rate: 5 out of 55 rows had trim_pass_count > 1 → 5/55 ≈ 9.1%
    buckets = _compute_buckets(
        data["process"]["retrim_rate_series"], n_per_bucket=50,
    )
    # With 55 rows and N=50, we get one bucket of 50 + a trailing 5 → 2 buckets
    # (5 ≥ 5 so it stands on its own).
    assert len(buckets) == 2
    # Overall mean across all rows should be 5/55:
    overall_mean = sum(b["mean"] * b["n"] for b in buckets) / sum(b["n"] for b in buckets)
    assert overall_mean == pytest.approx(5 / 55, rel=1e-6)

    # Smoothed-panel render does not crash:
    fig2, ax2 = plt.subplots()
    _draw_smoothed_panel(
        ax2, buckets=[{**b, "mean": b["mean"] * 100.0,
                       "stddev": b["stddev"] * 100.0,
                       "se": b["se"] * 100.0} for b in buckets],
        baseline_mean=0.0, baseline_cutoff_bucket_index=0,
        color="#ffb060",
    )
    plt.close(fig2)
