"""Bugfixes derived from work log analysis 2026-05-30.

Each test maps to one issue found in Work Files/5-30-26/laser_trim.log* and the
2026-05-28 terminal capture.  Filenames in fixtures are taken verbatim from the
production logs.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ---------------------------------------------------------------------------
# Issue: 18 'Model cannot be empty' + 2 'PermissionError' from filter gaps
# ---------------------------------------------------------------------------

NON_TRIM_FILENAMES = [
    "~$1844205_Final_Data.xlsx",                         # Excel lock file
    "master-sn108_5-30-2025_1-52 PM.xls",                # Template master
    "ChemCubed Ink Test 1.xlsx",                         # Lab experiment
    "ChemCubed Ink Test 17.xlsx",                        # (same pattern)
    "temp.xls",                                          # Scratch save
    "SN - Shop#.xlsx",                                   # Template with literal #
]


@pytest.mark.parametrize("filename", NON_TRIM_FILENAMES)
def test_non_trim_filter_catches_known_scratch_files(filename):
    """All filenames known to trigger 'Model cannot be empty' or PermissionError
    in the 2026-05-14..28 logs must be classified as non-trim by the regex
    list, so detect_file_type short-circuits before opening the file.
    """
    import re
    from laser_trim_analyzer.utils.constants import NON_TRIM_FILENAME_REGEXES

    matched = any(re.search(rx, filename.lower()) for rx in NON_TRIM_FILENAME_REGEXES)
    assert matched, (
        f"{filename!r} should be matched by NON_TRIM_FILENAME_REGEXES but isn't"
    )


# Sanity check: don't let new regexes false-positive on legitimate files.
LEGITIMATE_FILENAMES = [
    "8340-1 final SN 5140.xls",                          # Valid FT (handled in Task 3)
    "8340-1-sn470_5-30-2025_1-52 PM.xls",                # Valid FT
    "1081313-sn108_3-16-2011_12-17 PM.xls",              # Valid FT
    "8232-1_196_TA_Test Data_5-4-2026_10-48 AM_ResTrimmed Correct.xls",  # Valid trim
    "Rout_1091701_sn1695a_vo.xls",                       # Valid FT (Format 2)
    "1844205-shop1_5-18-2026 8-31-14 AM.xlsx",           # Valid FT (handled in Task 4)
]


@pytest.mark.parametrize("filename", LEGITIMATE_FILENAMES)
def test_non_trim_filter_does_not_match_real_files(filename):
    """Regexes added for scratch/template files must NOT match legitimate
    trim or FT filenames seen in production.
    """
    import re
    from laser_trim_analyzer.utils.constants import NON_TRIM_FILENAME_REGEXES

    matched_by = [rx for rx in NON_TRIM_FILENAME_REGEXES if re.search(rx, filename.lower())]
    assert not matched_by, (
        f"{filename!r} unexpectedly matched non-trim regex(es): {matched_by}"
    )


# ---------------------------------------------------------------------------
# Issue: 851 'Serial cannot be empty' from FT regex misses
# Filenames pulled verbatim from log.1, 2026-05-18 13:48-14:30 batch
# ---------------------------------------------------------------------------

FT_FILENAMES_AND_EXPECTED_SERIAL = [
    # (filename, expected_serial)
    # Pattern: "final SN <num>" (uppercase, space-separated)
    ("8340-1 final SN 5140.xls", "5140"),
    ("8340-1 final SN 5141.xls", "5141"),
    ("8340-1 final SN 5156.xls", "5156"),
    # Pattern: "final sn <num>xls" (lowercase, rename artifact)
    ("8340-1 final sn  116xls.xls", "116"),
    ("8340-1 final sn  119xls.xls", "119"),
    ("8340-1 final sn  140xls.xls", "140"),
    # Existing supported pattern: "final <num>" (no sn token) — must still work
    ("8340-1 final 215_6-4-2025_7-38 PM.xls", "215"),
]


@pytest.mark.parametrize(
    "filename,expected_serial", FT_FILENAMES_AND_EXPECTED_SERIAL
)
def test_final_test_serial_extracted_from_final_sn_pattern(filename, expected_serial):
    """Filenames with 'final [SN] <num>' must yield the numeric serial.
    Covers both new patterns from 2026-05-18 logs and existing 'final <num>'.
    """
    from laser_trim_analyzer.core.final_test_parser import FinalTestParser

    parser = FinalTestParser()
    metadata = parser._extract_metadata_from_filename(filename)
    assert metadata["serial"] == expected_serial, (
        f"{filename!r}: expected serial {expected_serial!r}, "
        f"got {metadata['serial']!r}"
    )


# Existing supported patterns — guard against regressions from the regex broadening.
EXISTING_FT_FILENAMES = [
    # (filename, expected_serial)
    ("1081313-sn108_3-16-2011_12-17 PM.xls", "108"),
    ("1844202-sn1004a_7-27-2022_1-26 PM.xls", "1004a"),
    ("8340-1-sn470_5-30-2025_1-52 PM.xls", "470"),
    ("Rout_1091701_sn1695a_vo.xls", "1695a"),
]


@pytest.mark.parametrize("filename,expected_serial", EXISTING_FT_FILENAMES)
def test_final_test_serial_existing_patterns_still_work(filename, expected_serial):
    """Existing -sn/_sn extraction must continue to work after broadening."""
    from laser_trim_analyzer.core.final_test_parser import FinalTestParser

    parser = FinalTestParser()
    metadata = parser._extract_metadata_from_filename(filename)
    assert metadata["serial"] == expected_serial, (
        f"{filename!r}: regression — expected {expected_serial!r}, "
        f"got {metadata['serial']!r}"
    )


# ---------------------------------------------------------------------------
# Issue: shop traveler ID used as serial at some stations (2026-05-18 batch)
# Decision recorded 2026-05-30: shop ID is the unit identifier, use verbatim.
# ---------------------------------------------------------------------------

FT_SHOP_FILENAMES = [
    ("1844205-shop1_5-18-2026 8-31-14 AM.xlsx", "shop1"),
    ("1844205-shop3_5-18-2026 9-46-24 AM.xlsx", "shop3"),
    ("1844205-shop4_5-18-2026 10-13-51 AM.xlsx", "shop4"),
    ("1844205-shop6_5-18-2026 10-22-42 AM.xlsx", "shop6"),
    ("1844205-shop9_5-18-2026 10-47-50 AM.xlsx", "shop9"),
]


@pytest.mark.parametrize("filename,expected_serial", FT_SHOP_FILENAMES)
def test_final_test_serial_extracted_from_shop_traveler_id(filename, expected_serial):
    """Files with '-shop<N>_<date>' use the shop ID as the unit serial.
    Production decision: this IS the unit identifier at those stations.
    Stored verbatim (lowercase) so it survives database lookups and matching.
    """
    from laser_trim_analyzer.core.final_test_parser import FinalTestParser

    parser = FinalTestParser()
    metadata = parser._extract_metadata_from_filename(filename)
    assert metadata["serial"] == expected_serial, (
        f"{filename!r}: expected serial {expected_serial!r}, "
        f"got {metadata['serial']!r}"
    )
    # Sanity: model should also be parsed.
    assert metadata["model"] == "1844205", (
        f"{filename!r}: expected model '1844205', got {metadata['model']!r}"
    )


# ---------------------------------------------------------------------------
# Issue: 40 'Worksheet named Sheet1 not found' errors -- hardcoded sheet name
# ---------------------------------------------------------------------------


def test_extract_format1_tracks_falls_back_to_first_sheet_when_sheet1_missing(
    tmp_path,
):
    """If the format detector mis-routes a file to _extract_format1_tracks
    but the workbook's main sheet isn't named 'Sheet1', the parser should
    fall back to the first available sheet instead of raising ValueError.
    """
    import pandas as pd
    from laser_trim_analyzer.core.final_test_parser import FinalTestParser

    # Build a workbook whose first sheet is named something else.  The data
    # layout matches Format 1 columns A..H so the tracks extractor can read
    # a valid track from it.  Columns:
    #   A=Measured, B=Index, C=Theory, D=Error, E=Angle, F=blank,
    #   G=Upper, H=Lower
    rows = [
        [0.10, 1, 0.10, 0.00, 0.00, None, 0.05, -0.05],
        [0.20, 2, 0.20, 0.00, 0.10, None, 0.05, -0.05],
        [0.30, 3, 0.30, 0.00, 0.20, None, 0.05, -0.05],
        [0.40, 4, 0.40, 0.00, 0.30, None, 0.05, -0.05],
        [0.50, 5, 0.50, 0.00, 0.40, None, 0.05, -0.05],
    ]
    df = pd.DataFrame(rows)

    fp = tmp_path / "format1_with_renamed_sheet.xlsx"
    # Write with a non-'Sheet1' sheet name -- this is the failure condition
    # we saw in production (some old stations exported with names like
    # 'Data', 'Test', or station-specific labels).
    with pd.ExcelWriter(fp, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Test", index=False, header=False)

    parser = FinalTestParser()
    with pd.ExcelFile(fp) as xl:
        # This call raised ValueError before the fix.
        tracks = parser._extract_format1_tracks(xl)

    assert tracks, "Fallback should produce at least one track from the first sheet"
    assert "positions" in tracks[0] or "errors" in tracks[0], (
        f"Track payload missing expected keys; got keys={list(tracks[0].keys())}"
    )
