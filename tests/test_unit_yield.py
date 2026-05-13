"""Tests for the unit-level yield feature."""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
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
