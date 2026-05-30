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
