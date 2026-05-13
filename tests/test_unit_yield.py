"""Tests for the unit-level yield feature."""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from sqlalchemy import text
from laser_trim_analyzer.database.manager import (
    compute_unit_id,
    extract_shop_number,
)


class TestExtractShopNumber:
    @pytest.mark.parametrize("serial,expected", [
        ("3P", "3"),
        ("3R", "3"),
        ("1A", "1"),
        ("0009", "0009"),
        ("8", "8"),
        ("196", "196"),
        ("12RC", "12"),
        ("  4P  ", "4"),  # strips whitespace
    ])
    def test_extracts_leading_digits(self, serial, expected):
        assert extract_shop_number(serial) == expected

    @pytest.mark.parametrize("serial", [
        "TEST", "test", "x", "Unknown", "", None,
        "٣P",  # Persian digit 3 + P — must not match under re.ASCII
    ])
    def test_returns_none_for_junk_serials(self, serial):
        assert extract_shop_number(serial) is None


class TestComputeUnitId:
    def test_typical_trim_record(self):
        uid = compute_unit_id("8895", "3P", datetime(2025, 3, 26, 12, 55))
        assert uid == "8895/3/2025-03-26"

    def test_same_unit_different_section(self):
        a = compute_unit_id("8895", "3P", datetime(2025, 3, 26))
        b = compute_unit_id("8895", "3R", datetime(2025, 3, 26))
        assert a == b == "8895/3/2025-03-26"

    def test_retrim_same_day_same_unit(self):
        morning = compute_unit_id("8895", "3P", datetime(2025, 3, 26, 9, 0))
        evening = compute_unit_id("8895", "3P", datetime(2025, 3, 26, 17, 43))
        assert morning == evening

    def test_different_shop_different_unit(self):
        a = compute_unit_id("8895", "3P", datetime(2025, 3, 26))
        b = compute_unit_id("8895", "4P", datetime(2025, 3, 26))
        assert a != b

    def test_different_day_different_unit(self):
        a = compute_unit_id("8895", "3P", datetime(2025, 3, 26))
        b = compute_unit_id("8895", "3P", datetime(2025, 3, 27))
        assert a != b

    def test_returns_none_when_serial_junk(self):
        assert compute_unit_id("8895", "TEST", datetime(2025, 3, 26)) is None

    def test_returns_none_when_serial_missing(self):
        assert compute_unit_id("8895", None, datetime(2025, 3, 26)) is None

    def test_returns_none_when_model_missing(self):
        assert compute_unit_id(None, "3P", datetime(2025, 3, 26)) is None
        assert compute_unit_id("", "3P", datetime(2025, 3, 26)) is None

    def test_returns_none_when_date_missing(self):
        assert compute_unit_id("8895", "3P", None) is None


class TestUnitIdMigration:
    def test_migration_adds_column_idempotently(self, tmp_path):
        """First migration adds column; second call is a no-op."""
        from laser_trim_analyzer.database.manager import DatabaseManager
        from sqlalchemy import inspect

        db_path = tmp_path / "smoke.db"
        # First init runs migrations
        db1 = DatabaseManager(database_path=db_path)
        insp = inspect(db1._engine)
        cols = [c["name"] for c in insp.get_columns("analysis_results")]
        assert "unit_id" in cols

        # Second init re-runs migration code; must not crash
        db2 = DatabaseManager(database_path=db_path)
        cols2 = [c["name"] for c in inspect(db2._engine).get_columns("analysis_results")]
        assert "unit_id" in cols2
        assert cols == cols2  # No duplicate


class TestWriteTimeUnitId:
    def test_save_analysis_populates_unit_id(self, tmp_path):
        """Saving a fresh AnalysisResult populates unit_id in the DB row."""
        from laser_trim_analyzer.database.manager import DatabaseManager
        from laser_trim_analyzer.database.models import AnalysisResult as DBAR
        from laser_trim_analyzer.core.models import (
            AnalysisResult, AnalysisStatus, FileMetadata, SystemType,
            TrackData, RiskCategory,
        )

        db = DatabaseManager(database_path=tmp_path / "wt.db")
        track = TrackData(
            track_id="TRK1",
            status=AnalysisStatus.PASS,
            travel_length=10.0,
            linearity_spec=0.01,
            sigma_gradient=0.001,
            sigma_threshold=0.005,
            sigma_pass=True,
            optimal_offset=0.0,
            linearity_error=0.001,
            linearity_pass=True,
            linearity_fail_points=0,
            risk_category=RiskCategory.LOW,
        )
        analysis = AnalysisResult(
            metadata=FileMetadata(
                filename="8895_3P_TEST DATA_3-26-2025.xls",
                file_path=tmp_path / "8895_3P.xls",
                file_date=datetime(2025, 3, 26, 12, 55),
                model="8895",
                serial="3P",
                system=SystemType.B,
                has_multi_tracks=False,
            ),
            overall_status=AnalysisStatus.PASS,
            processing_time=0.1,
            tracks=[track],
        )
        new_id = db.save_analysis(analysis)
        with db.session() as s:
            row = s.query(DBAR).filter(DBAR.id == new_id).one()
            assert row.unit_id == "8895/3/2025-03-26"

    def test_save_analysis_junk_serial_unit_id_null(self, tmp_path):
        from laser_trim_analyzer.database.manager import DatabaseManager
        from laser_trim_analyzer.database.models import AnalysisResult as DBAR
        from laser_trim_analyzer.core.models import (
            AnalysisResult, AnalysisStatus, FileMetadata, SystemType,
            TrackData, RiskCategory,
        )

        db = DatabaseManager(database_path=tmp_path / "junk.db")
        track = TrackData(
            track_id="TRK1", status=AnalysisStatus.PASS, travel_length=10.0,
            linearity_spec=0.01, sigma_gradient=0.001, sigma_threshold=0.005,
            sigma_pass=True, optimal_offset=0.0, linearity_error=0.001,
            linearity_pass=True, linearity_fail_points=0,
            risk_category=RiskCategory.LOW,
        )
        analysis = AnalysisResult(
            metadata=FileMetadata(
                filename="8895_TEST.xls",
                file_path=tmp_path / "8895_TEST.xls",
                file_date=datetime(2025, 3, 26),
                model="8895", serial="TEST",
                system=SystemType.B, has_multi_tracks=False,
            ),
            overall_status=AnalysisStatus.PASS, processing_time=0.1, tracks=[track],
        )
        new_id = db.save_analysis(analysis)
        with db.session() as s:
            row = s.query(DBAR).filter(DBAR.id == new_id).one()
            assert row.unit_id is None


class TestUnitIdBackfill:
    def _insert_legacy_row(self, db, model, serial, file_date_str):
        """Insert a row directly via SQL with unit_id=NULL to simulate
        a row written before the unit_id column existed."""
        with db.session() as s:
            s.execute(text(
                "INSERT INTO analysis_results "
                "(filename, file_path, file_date, model, serial, system, "
                " has_multi_tracks, overall_status, timestamp) "
                "VALUES (:fn, :fp, :fd, :m, :sn, 'B', 0, 'PASS', :ts)"
            ), {
                "fn": f"{model}_{serial}.xls",
                "fp": f"/fake/{model}_{serial}.xls",
                "fd": file_date_str,
                "m": model, "sn": serial,
                "ts": datetime.utcnow(),
            })
            s.commit()

    def test_backfill_populates_valid_rows_and_leaves_junk_null(self, tmp_path):
        from laser_trim_analyzer.database.manager import DatabaseManager
        from laser_trim_analyzer.database.models import AnalysisResult as DBAR

        db_path = tmp_path / "bf.db"
        # First init creates schema (with empty backfill).
        db = DatabaseManager(database_path=db_path)
        self._insert_legacy_row(db, "8895", "3P", "2025-03-26")
        self._insert_legacy_row(db, "8895", "3R", "2025-03-26")
        self._insert_legacy_row(db, "8895", "TEST", "2025-03-26")

        # NULL out unit_id so the backfill has work to do
        with db.session() as s:
            s.execute(text("UPDATE analysis_results SET unit_id = NULL"))
            s.commit()

        # Re-run backfill on the same connection
        with db.session() as s:
            db._backfill_unit_ids(s)

        with db.session() as s:
            rows = {r.serial: r.unit_id for r in s.query(DBAR).all()}
        assert rows["3P"] == "8895/3/2025-03-26"
        assert rows["3R"] == "8895/3/2025-03-26"
        assert rows["TEST"] is None

    def test_backfill_idempotent(self, tmp_path):
        from laser_trim_analyzer.database.manager import DatabaseManager
        from laser_trim_analyzer.database.models import AnalysisResult as DBAR

        db_path = tmp_path / "idem.db"
        db = DatabaseManager(database_path=db_path)
        self._insert_legacy_row(db, "8895", "3P", "2025-03-26")
        # First backfill (already happened on init; force a second)
        with db.session() as s:
            db._backfill_unit_ids(s)
        with db.session() as s:
            db._backfill_unit_ids(s)
        with db.session() as s:
            row = s.query(DBAR).filter(DBAR.serial == "3P").one()
            assert row.unit_id == "8895/3/2025-03-26"
