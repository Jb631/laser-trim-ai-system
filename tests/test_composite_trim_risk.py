"""Tests for the composite trim-risk early-warning feature (2026-06-01 plan)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_trackdata_has_untrimmed_error_max_field():
    from laser_trim_analyzer.core.models import TrackData
    fields = TrackData.model_fields  # pydantic v2
    assert "untrimmed_error_max" in fields, "TrackData must expose untrimmed_error_max"


def test_trackresult_has_untrimmed_error_max_column():
    from laser_trim_analyzer.database.models import TrackResult
    assert hasattr(TrackResult, "untrimmed_error_max"), \
        "TrackResult must have an untrimmed_error_max column"
