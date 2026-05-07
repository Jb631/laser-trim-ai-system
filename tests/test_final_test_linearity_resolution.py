import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laser_trim_analyzer.database.manager import DatabaseManager


def test_final_test_track_failure_overrides_passing_header():
    resolved = DatabaseManager._resolve_final_test_linearity_pass(
        {"linearity_pass": True},
        [
            {"track_id": "1", "linearity_pass": True},
            {"track_id": "2", "linearity_pass": False},
        ],
    )

    assert resolved is False


def test_final_test_missing_header_uses_all_track_results():
    resolved = DatabaseManager._resolve_final_test_linearity_pass(
        {"linearity_pass": None},
        [
            {"track_id": "1", "linearity_pass": True},
            {"track_id": "2", "linearity_pass": True},
        ],
    )

    assert resolved is True
