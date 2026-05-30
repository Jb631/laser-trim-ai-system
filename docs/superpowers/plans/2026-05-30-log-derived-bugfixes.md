# Log-Derived Bugfixes Implementation Plan (2026-05-30)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the ~2,300 error spam from production logs by fixing six discrete issues found in the 2026-05-14 → 2026-05-28 work logs (`Work Files/5-30-26/`): stale banner string, three non-trim filename gaps, Final Test serial extraction misses, hardcoded `Sheet1` reads, and the trim-save dedupe key mismatch that produces `UNIQUE constraint failed: analysis_results` IntegrityErrors.

**Architecture:** Six independent, additive changes across four files. No schema changes, no new dependencies. Each fix has a single deterministic test that uses real filenames from the logs as fixtures. The dedupe fix is the only structural change — it aligns the existence-check key in `save_analysis`/`save_batch` with the existing DB UNIQUE constraint key `(filename, file_date, model, serial)`.

**Tech Stack:** Python 3.x, pytest, SQLAlchemy 2.0, pandas, openpyxl.

**Out of scope (deferred):** Smoothness `_OS_Only_*` column-layout fix — needs a sample file the user will bring next time.

**Source evidence:** `Work Files/5-30-26/laser_trim.log` and `.log.1`–`.log.3`. Top failures by count:
- 851 × `Error processing Final Test ...: Serial cannot be empty` (FT regex misses)
- 270 × `UNIQUE constraint failed: analysis_results / final_test_results` (save dedupe key mismatch)
- 40 × `Worksheet named 'Sheet1' not found` (hardcoded sheet name)
- 18 × `Model cannot be empty` (template/scratch files not filtered)
- 2 × `PermissionError ... ~$...xlsx` (Excel lock files not filtered)
- 1 × stale `v3` banner

---

## File Structure

**Files modified (no new source files):**
- `src/laser_trim_analyzer/__main__.py` — fix banner string (1 line)
- `src/laser_trim_analyzer/utils/constants.py` — add 5 entries to `NON_TRIM_FILENAME_REGEXES`
- `src/laser_trim_analyzer/core/final_test_parser.py` — broaden serial regex (~10 lines), Sheet1 fallback (2 sites)
- `src/laser_trim_analyzer/database/manager.py` — change existence check key in `save_analysis` and `save_batch` (~8 lines total)

**Files created:**
- `tests/test_log_derived_bugfixes_2026_05_30.py` — all new test cases, following the convention of `tests/test_5_8_2026_bugfixes.py` (one test function per fix, real-log filenames as fixtures)

**No GUI changes required.** All fixes are at the parser / data-access layer. The GUI already calls `save_analysis()` and will silently benefit from the dedupe alignment.

---

## Task 1: Stale banner string

**Files:**
- Modify: `src/laser_trim_analyzer/__main__.py:52`
- Test: none (cosmetic; would require capturing logger output, low value)

- [ ] **Step 1: Edit the banner**

Change line 52 from:
```python
logger.info("Starting Laser Trim Analyzer v3...")
```
to:
```python
logger.info("Starting Laser Trim Analyzer v5...")
```

- [ ] **Step 2: Commit**

```bash
git add src/laser_trim_analyzer/__main__.py
git commit -m "chore: bump startup banner to v5"
```

---

## Task 2: Filter non-trim scratch/template/lock files

**Why:** 18 `Model cannot be empty` errors + 2 `PermissionError` come from files that the parser shouldn't have opened at all. Adding anchored regexes to `NON_TRIM_FILENAME_REGEXES` short-circuits them in `detect_file_type` (`parser.py:1276`) before any Excel I/O.

**Affected filenames from logs:**
- `~$1844205_Final_Data.xlsx` (Excel lock file — opened during PermissionError)
- `master-sn108_5-30-2025_1-52 PM.xls` (template master copy)
- `ChemCubed Ink Test 1.xlsx` (lab experiment, not production data)
- `temp.xls` (scratch save)
- `SN - Shop#.xlsx` (template with literal `#`)

**Files:**
- Modify: `src/laser_trim_analyzer/utils/constants.py:204-214` — extend `NON_TRIM_FILENAME_REGEXES`
- Test: `tests/test_log_derived_bugfixes_2026_05_30.py` — `test_non_trim_filter_*`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_log_derived_bugfixes_2026_05_30.py` with this content (we will append to it in later tasks):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_log_derived_bugfixes_2026_05_30.py::test_non_trim_filter_catches_known_scratch_files -v`

Expected: 6 FAILs (one per filename), each with assertion error like `'~$1844205_Final_Data.xlsx' should be matched by NON_TRIM_FILENAME_REGEXES but isn't`. The legitimate-files test should PASS already.

- [ ] **Step 3: Add the regex entries**

Edit `src/laser_trim_analyzer/utils/constants.py` — replace the existing `NON_TRIM_FILENAME_REGEXES` definition (lines 204-214) with:

```python
# Non-trim filename regex patterns (matched on lowercased filename).
# Use this when a substring check would false-positive on legitimate files
# (e.g. "vo_" would also match "servo_*"). Patterns are anchored or otherwise
# made specific.
NON_TRIM_FILENAME_REGEXES: Final[list] = [
    r"^vo_\d",            # Voltage Output files: VO_<model>_<sn>_<date>.xlsx
    # Experimental / shop-floor scratch files like:
    #   0022A.xls, 1050B_FAIL.xls, 73B_PASS.xls, 1054B_131713_FAIL.xls
    # Pattern: digits + single letter, optional _HHMMSS, optional _PASS|_FAIL.
    # These trip the FT sheet-pattern heuristic (Sheet1 + content) and
    # produce ~2000 'Serial cannot be empty' validation errors per batch.
    # Real trim and FT filenames carry hyphens / spaces / 'sn'/'final'/'Trimmed',
    # so this anchored pattern won't false-positive on production data.
    r"^\d+[a-z](_\d{6})?(_(pass|fail))?\.xlsx?$",
    # Excel lock files (e.g. "~$1844205_Final_Data.xlsx") appear briefly while
    # the file is open in Excel on the network share; opening one throws
    # PermissionError in the parser.
    r"^~\$",
    # Template / master copies (e.g. "master-sn108_*.xls",
    # "master-snN_*.xls").  These have no real serial — the "sn" token is
    # part of the template name itself.
    r"^master[-_]",
    # ChemCubed ink-test lab experiments (e.g. "ChemCubed Ink Test 1.xlsx").
    # Not production FT data; no model parseable from filename.
    r"^chemcubed\b",
    # Scratch files named exactly "temp.xls" / "temp.xlsx".
    r"^temp\.xlsx?$",
    # Template files with the literal "#" placeholder for shop number
    # (e.g. "SN - Shop#.xlsx").  Real shop traveler files use digits.
    r"shop#",
]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_log_derived_bugfixes_2026_05_30.py -v -k non_trim_filter`

Expected: 12 PASS (6 catch-known + 6 don't-match-real). No failures.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/utils/constants.py tests/test_log_derived_bugfixes_2026_05_30.py
git commit -m "fix(parser): filter Excel lock files and template/scratch files as non-trim

Adds 5 anchored regexes to NON_TRIM_FILENAME_REGEXES covering ~\$ lock
files, master- templates, ChemCubed lab tests, temp.xls scratch files,
and shop# templates.  Eliminates ~20 'Model cannot be empty' and 2
PermissionError errors per network-share batch.

Filenames verified verbatim against 2026-05-14..28 production logs."
```

---

## Task 3: Final Test serial extraction — `final [SN] <num>` pattern

**Why:** 851 `Serial cannot be empty` errors. The current fallback regex `\bfinal\s+(\d+)` at `final_test_parser.py:312` only matches when "final" is followed directly by digits. Real production filenames have `final SN 5140`, `final sn  116xls`, etc., with the word "SN" (case-insensitive, optional) between "final" and the number, often with trailing junk like `xls.xls` from rename artifacts.

**Files:**
- Modify: `src/laser_trim_analyzer/core/final_test_parser.py:309-314`
- Test: `tests/test_log_derived_bugfixes_2026_05_30.py` — append `test_final_test_serial_*`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_log_derived_bugfixes_2026_05_30.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_log_derived_bugfixes_2026_05_30.py::test_final_test_serial_extracted_from_final_sn_pattern -v`

Expected: 6 FAILs (the `final SN <num>` and `final sn  <num>xls` patterns return `None`). The "final 215" case PASSes already because the existing fallback handles it. The existing-patterns test should PASS entirely.

- [ ] **Step 3: Broaden the fallback regex**

In `src/laser_trim_analyzer/core/final_test_parser.py`, replace lines 309-314:

```python
        else:
            # Fallback: "model final serial" pattern (e.g., "8340-1 final 215_6-4-2025")
            # The word "final" separates model from serial number
            final_match = re.search(r'\bfinal\s+(\d+)', base, re.IGNORECASE)
            if final_match:
                metadata["serial"] = final_match.group(1)
```

with:

```python
        else:
            # Fallback: "model final [sn] serial" pattern.  Production stations
            # write filenames as any of:
            #   "8340-1 final 215_6-4-2025_..."     -> serial 215  (legacy)
            #   "8340-1 final SN 5140.xls"          -> serial 5140 (FY26 batches)
            #   "8340-1 final sn  116xls.xls"       -> serial 116  (rename artifact:
            #                                                       extra space + "xls"
            #                                                       embedded before ext)
            # The optional "sn " token and one-or-more whitespace handle all three.
            # \s+ is intentional (not \s) so a double-space doesn't break the match.
            final_match = re.search(
                r'\bfinal\s+(?:sn\s+)?(\d+)', base, re.IGNORECASE
            )
            if final_match:
                metadata["serial"] = final_match.group(1)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_log_derived_bugfixes_2026_05_30.py -v -k "final_test_serial"`

Expected: 11 PASS (7 new + 4 existing regressions). Zero failures.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/core/final_test_parser.py tests/test_log_derived_bugfixes_2026_05_30.py
git commit -m "fix(ft-parser): extract serial from 'final SN <num>' and rename-artifact patterns

Broadens the fallback regex in _extract_metadata_from_filename to accept
an optional 'sn ' token between 'final' and the digits, and uses \\\\s+
so double-space + 'xls' rename artifacts are tolerated.

Eliminates ~830 'Serial cannot be empty' errors per network-share batch,
matched against verbatim filenames from 2026-05-14..28 production logs."
```

---

## Task 4: Final Test serial extraction — `shop<N>` traveler ID pattern

**Why:** The 2026-05-18 batch contains files like `1844205-shop1_5-18-2026 8-31-14 AM.xlsx` where the post-`-` token is `shop1`/`shop3`/`shop4` etc. — the shop traveler ID, which is the unit identifier at that station. Per user decision (2026-05-30 brainstorming), treat the shop ID verbatim as the serial.

**Files:**
- Modify: `src/laser_trim_analyzer/core/final_test_parser.py` (extend the fallback chain added in Task 3)
- Test: `tests/test_log_derived_bugfixes_2026_05_30.py` — append `test_final_test_serial_shop_*`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_log_derived_bugfixes_2026_05_30.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_log_derived_bugfixes_2026_05_30.py::test_final_test_serial_extracted_from_shop_traveler_id -v`

Expected: 5 FAILs — current code returns `serial=None` because neither the `-sn` regex nor the `final` fallback matches `shop1`.

- [ ] **Step 3: Extend the fallback chain**

In `src/laser_trim_analyzer/core/final_test_parser.py`, locate the fallback block you edited in Task 3 (the `else:` after `sn_match` around lines 309-322 of the post-Task-3 file). Add a third fallback so the block reads:

```python
        else:
            # Fallback: "model final [sn] serial" pattern.  Production stations
            # write filenames as any of:
            #   "8340-1 final 215_6-4-2025_..."     -> serial 215  (legacy)
            #   "8340-1 final SN 5140.xls"          -> serial 5140 (FY26 batches)
            #   "8340-1 final sn  116xls.xls"       -> serial 116  (rename artifact:
            #                                                       extra space + "xls"
            #                                                       embedded before ext)
            # The optional "sn " token and one-or-more whitespace handle all three.
            # \s+ is intentional (not \s) so a double-space doesn't break the match.
            final_match = re.search(
                r'\bfinal\s+(?:sn\s+)?(\d+)', base, re.IGNORECASE
            )
            if final_match:
                metadata["serial"] = final_match.group(1)
            else:
                # Fallback B: "model-shop<N>" traveler-ID pattern.
                # Some FY26 stations identify the unit by shop traveler rather
                # than a serial number (e.g. "1844205-shop1_<date>.xlsx").
                # Per 2026-05-30 decision, the shop ID IS the unit identifier
                # at those stations, so we store it verbatim (lowercased) as
                # the serial.
                shop_match = re.search(r'[-_](shop\d+)\b', base, re.IGNORECASE)
                if shop_match:
                    metadata["serial"] = shop_match.group(1).lower()
```

- [ ] **Step 4: Run tests to verify pass**

Run all FT serial tests:
```
pytest tests/test_log_derived_bugfixes_2026_05_30.py -v -k "final_test_serial or shop_traveler"
```

Expected: 16 PASS total (7 final-pattern + 4 existing-regression + 5 shop). Zero failures.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/core/final_test_parser.py tests/test_log_derived_bugfixes_2026_05_30.py
git commit -m "fix(ft-parser): use shop traveler ID as serial when -shop<N> present

Some FY26 production stations identify the unit by shop traveler ID
('1844205-shop1_<date>.xlsx') instead of an explicit -sn<num>.  Adds a
third fallback that captures '-shop<digits>' verbatim (lowercased) into
metadata['serial'].

Eliminates the per-batch wave of Serial-cannot-be-empty errors from those
stations.  Verified against 2026-05-18 batch logs."
```

---

## Task 5: `Sheet1` not-found fallback to first sheet

**Why:** 40 errors from `pd.read_excel(xl, sheet_name="Sheet1", ...)` hardcoded at two sites in `final_test_parser.py` (lines 147, 370). The format detector at line 98 classified these as Format 1 because the sheet structure was close enough, but the file's main sheet is named differently. Falling back to `xl.sheet_names[0]` is safe because Format 1 has only one sheet of substance.

**Files:**
- Modify: `src/laser_trim_analyzer/core/final_test_parser.py` — two sites (line 147 in `_parse_format1`, line 370 in `_extract_format1_tracks`)
- Test: `tests/test_log_derived_bugfixes_2026_05_30.py` — append `test_sheet1_fallback_*`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_log_derived_bugfixes_2026_05_30.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_log_derived_bugfixes_2026_05_30.py::test_extract_format1_tracks_falls_back_to_first_sheet_when_sheet1_missing -v`

Expected: FAIL with `ValueError: Worksheet named 'Sheet1' not found`.

- [ ] **Step 3: Add fallback at `_extract_format1_tracks` (line 370)**

In `src/laser_trim_analyzer/core/final_test_parser.py`, find the line `df = pd.read_excel(xl, sheet_name="Sheet1", header=None)` inside `_extract_format1_tracks`. Replace that single line with:

```python
            try:
                df = pd.read_excel(xl, sheet_name="Sheet1", header=None)
            except ValueError:
                # Format detector routed this to Format 1 but the workbook's
                # main sheet isn't named 'Sheet1'.  Some FY26 station exports
                # use station-specific sheet names while keeping the same
                # column layout.  Fall back to the first sheet.
                if not xl.sheet_names:
                    raise
                fallback_sheet = xl.sheet_names[0]
                logger.warning(
                    f"'Sheet1' not found in {getattr(xl, 'io', '?')}, "
                    f"falling back to first sheet {fallback_sheet!r}"
                )
                df = pd.read_excel(xl, sheet_name=fallback_sheet, header=None)
```

(Keep the existing indentation — this is inside the `try:` block of `_extract_format1_tracks`.)

- [ ] **Step 4: Add the same fallback at `_parse_format1` (line 147)**

Find the earlier site in `_parse_format1`:

```python
        try:
            df = pd.read_excel(xl, sheet_name="Sheet1", header=None)
```

Replace the `pd.read_excel` line with the same fallback pattern (this site is also inside a `try:` but with broader exception handling — preserve that):

```python
        try:
            try:
                df = pd.read_excel(xl, sheet_name="Sheet1", header=None)
            except ValueError:
                # See _extract_format1_tracks for rationale.
                if not xl.sheet_names:
                    raise
                fallback_sheet = xl.sheet_names[0]
                logger.warning(
                    f"'Sheet1' not found while reading metadata, "
                    f"falling back to first sheet {fallback_sheet!r}"
                )
                df = pd.read_excel(xl, sheet_name=fallback_sheet, header=None)
```

- [ ] **Step 5: Run test to verify pass**

Run: `pytest tests/test_log_derived_bugfixes_2026_05_30.py::test_extract_format1_tracks_falls_back_to_first_sheet_when_sheet1_missing -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/core/final_test_parser.py tests/test_log_derived_bugfixes_2026_05_30.py
git commit -m "fix(ft-parser): fall back to first sheet when Sheet1 not found

The format detector at line 98 already routes some non-'Sheet1' workbooks
into Format 1 based on column structure, but the two read_excel calls
inside _parse_format1 and _extract_format1_tracks hardcode sheet_name=
'Sheet1' and raise ValueError on mismatch.  Catch that and try
xl.sheet_names[0], which is safe because Format 1 has a single sheet of
substance.

Eliminates 40 'Worksheet named Sheet1 not found' errors observed in the
2026-05-18 batch."
```

---

## Task 6: Align trim-save existence check with the DB UNIQUE constraint

**Why:** This is the bug that produced the DB errors you saw in the terminal. The DB UNIQUE constraint on `analysis_results` is `(filename, file_date, model, serial)`. The existence check in `save_analysis` (`database/manager.py:1058-1061`) and `save_batch` (lines 1108-1111) instead queries by `(filename, file_path)`. When the same logical record is re-presented with a different `file_path` string (UNC vs mapped drive, folder reorg, etc.), the lookup misses, the code attempts an INSERT, and SQLite raises `IntegrityError` because the constraint sees it as a duplicate. Aligning the lookup key with the constraint key turns those re-presentations into idempotent UPDATEs — same outcome as the existing `save_final_test` path (`manager.py:4293-4314` already documents this lesson).

**Verified evidence from `Work Files/5-30-26/2026-05-28 160741,148 - laser_trim_.txt`:**
- Cache loaded 146,415 file paths, batch incremental=True
- 18 files survived the cache filter in 40 seconds (cache leaks)
- 3 saved successfully, 12 IntegrityErrors on `analysis_results`, 3 misc failures
- All 12 failing inserts had params like `model='8877', serial='2', file_date='2025-12-07'` — exact duplicates of records that the cache missed because `file_path` differed

**Files:**
- Modify: `src/laser_trim_analyzer/database/manager.py:1054-1061` (save_analysis), `1097-1116` (save_batch)
- Test: `tests/test_log_derived_bugfixes_2026_05_30.py` — append `test_save_analysis_dedupe_*`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_log_derived_bugfixes_2026_05_30.py`:

```python
# ---------------------------------------------------------------------------
# Issue: 24+ IntegrityErrors per batch from save_analysis using the wrong
# existence-check key.  DB UNIQUE constraint is (filename, file_date, model,
# serial) but save_analysis checks (filename, file_path).
# Evidence: Work Files/5-30-26/2026-05-28 160741,148 - laser_trim_.txt
# ---------------------------------------------------------------------------


def _build_analysis_result(filename, file_path, model, serial, file_date):
    """Minimal AnalysisResult sufficient for save_analysis()."""
    from datetime import datetime as _dt
    from laser_trim_analyzer.core.models import (
        AnalysisResult,
        AnalysisStatus,
        FileMetadata,
        SystemType,
    )

    metadata = FileMetadata(
        filename=filename,
        file_path=Path(file_path),
        file_date=file_date or _dt(2025, 12, 7),
        model=model,
        serial=serial,
        system=SystemType.SYSTEM_A,
        has_multi_tracks=False,
    )
    return AnalysisResult(
        metadata=metadata,
        overall_status=AnalysisStatus.PASS,
        tracks={},
        processing_time=0.1,
        timestamp=_dt(2026, 5, 28, 16, 8, 20),
    )


def test_save_analysis_dedupes_on_metadata_not_file_path(tmp_path, monkeypatch):
    """Saving the same logical record (same filename+date+model+serial) via
    two DIFFERENT file_path strings should be idempotent -- the second save
    must UPDATE the existing row, not raise IntegrityError.
    """
    from datetime import datetime as _dt
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.database.manager import DatabaseManager

    # Build a fresh on-disk DB to exercise the real schema + constraint
    db_url = f"sqlite:///{tmp_path / 'dedupe.db'}"
    cfg = Config()
    cfg.database.path = str(tmp_path / "dedupe.db")
    mgr = DatabaseManager(database_url=db_url)
    mgr.init_db()

    common = dict(
        filename="8877_5_deg_2_TEST DATA_12-7-2025_4-25 AM.xls",
        model="8877",
        serial="2",
        file_date=_dt(2025, 12, 7),
    )

    # First save: insert.
    a1 = _build_analysis_result(
        file_path=r"\\192.168.66.9\BTXData\TEST_DATA\DLTS\8877-4\8877_5_deg\8877_5_deg_2_TEST DATA_12-7-2025_4-25 AM.xls",
        **common,
    )
    id1 = mgr.save_analysis(a1)
    assert id1 > 0, "First save should produce a valid row id"

    # Second save: same metadata, DIFFERENT file_path string (mapped drive
    # vs UNC).  Pre-fix, this raises IntegrityError.  Post-fix, it should
    # find the existing row and update it.
    a2 = _build_analysis_result(
        file_path=r"Z:\TEST_DATA\DLTS\8877-4\8877_5_deg\8877_5_deg_2_TEST DATA_12-7-2025_4-25 AM.xls",
        **common,
    )
    id2 = mgr.save_analysis(a2)  # must NOT raise
    assert id2 == id1, (
        f"Re-saving same logical record should hit the existing row "
        f"(id={id1}), but got id={id2}"
    )
```

> **Note:** The test imports `FileMetadata`/`AnalysisResult` from `core.models`. If the constructor signatures differ from what's shown here (the model has additional required fields), inspect `src/laser_trim_analyzer/core/models.py` and add the missing fields with sentinel values before failing this step. The test is verifying the dedupe BEHAVIOR; whatever constructor the codebase requires is fine.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_log_derived_bugfixes_2026_05_30.py::test_save_analysis_dedupes_on_metadata_not_file_path -v`

Expected: FAIL with `sqlalchemy.exc.IntegrityError: UNIQUE constraint failed: analysis_results.filename, analysis_results.file_date, analysis_results.model, analysis_results.serial` — exactly the error from the terminal capture.

- [ ] **Step 3: Fix `save_analysis` existence check**

In `src/laser_trim_analyzer/database/manager.py`, find the block at lines 1054-1066 (the `with self.session() as session:` inside `save_analysis`). Replace:

```python
        with self.session() as session:
            # Check for existing record by filename (stable identifier)
            # This ensures re-analysis updates the existing record even if
            # model/serial/date parsing changed
            existing = session.query(DBAnalysisResult).filter(
                DBAnalysisResult.filename == analysis.metadata.filename,
                DBAnalysisResult.file_path == str(analysis.metadata.file_path),
            ).first()

            if existing:
                logger.info(f"Updating existing analysis: {analysis.metadata.filename}")
                return self._update_existing_analysis(session, analysis)
```

with:

```python
        with self.session() as session:
            # Check for existing record by the DB's UNIQUE-constraint key
            # (filename, file_date, model, serial).  Pre-fix this filtered on
            # (filename, file_path), but the UNIQUE constraint is keyed on
            # the metadata tuple -- the same file on a different path string
            # (UNC vs mapped drive, folder reorg, network share migration)
            # missed this lookup and then raised IntegrityError at INSERT
            # time.  Aligning the lookup with the constraint turns these
            # re-presentations into idempotent UPDATEs.
            # See save_final_test for the parallel lesson.
            existing = session.query(DBAnalysisResult).filter(
                DBAnalysisResult.filename == analysis.metadata.filename,
                DBAnalysisResult.file_date == analysis.metadata.file_date,
                DBAnalysisResult.model == analysis.metadata.model,
                DBAnalysisResult.serial == analysis.metadata.serial,
            ).first()

            if existing:
                logger.info(f"Updating existing analysis: {analysis.metadata.filename}")
                return self._update_existing_analysis(session, analysis)
```

- [ ] **Step 4: Fix `save_batch` existence check (same change)**

Find the block at lines 1106-1116 inside `save_batch`. Replace:

```python
                        # Check for existing record by filename + file_path
                        # (must match save_analysis to avoid cross-directory overwrites)
                        existing = session.query(DBAnalysisResult).filter(
                            DBAnalysisResult.filename == analysis.metadata.filename,
                            DBAnalysisResult.file_path == str(analysis.metadata.file_path),
                        ).first()
```

with:

```python
                        # Check for existing record by the DB UNIQUE key
                        # (filename, file_date, model, serial).  Must match
                        # save_analysis; see comment there for rationale.
                        existing = session.query(DBAnalysisResult).filter(
                            DBAnalysisResult.filename == analysis.metadata.filename,
                            DBAnalysisResult.file_date == analysis.metadata.file_date,
                            DBAnalysisResult.model == analysis.metadata.model,
                            DBAnalysisResult.serial == analysis.metadata.serial,
                        ).first()
```

- [ ] **Step 5: Run test to verify pass**

Run: `pytest tests/test_log_derived_bugfixes_2026_05_30.py::test_save_analysis_dedupes_on_metadata_not_file_path -v`

Expected: PASS.

- [ ] **Step 6: Run the entire new test file plus the historical regression test**

```
pytest tests/test_log_derived_bugfixes_2026_05_30.py tests/test_5_8_2026_bugfixes.py -v
```

Expected: All tests in both files PASS. The older bugfix file is included as a smoke test to confirm we haven't regressed prior fixes (it touches some adjacent code paths).

- [ ] **Step 7: Commit**

```bash
git add src/laser_trim_analyzer/database/manager.py tests/test_log_derived_bugfixes_2026_05_30.py
git commit -m "fix(db): align trim-save existence check with UNIQUE constraint key

save_analysis and save_batch were filtering existing records by
(filename, file_path), but the DB UNIQUE constraint on analysis_results
is keyed on (filename, file_date, model, serial).  When the same logical
record was re-presented via a different file_path string (UNC vs mapped
drive, folder reorganization, network share migration), the lookup
missed and SQLite raised IntegrityError at INSERT time.

Switching the lookup to the constraint key turns these re-presentations
into idempotent UPDATEs -- matching the behavior already implemented for
final_test_results in save_final_test (see manager.py:4293-4314 for the
parallel lesson).

Evidence: 24 IntegrityErrors per batch in
Work Files/5-30-26/2026-05-28 160741,148 - laser_trim_.txt, all caused
by 146k cached file_path strings missing duplicates that the constraint
correctly identified."
```

---

## Task 7: End-to-end smoke test against real log filenames

**Why:** Confidence pass — wire the FT parser through `detect_file_type` and `_extract_metadata_from_filename` against every distinct filename pattern from the 2026-05-30 logs to confirm the pipeline routes them correctly and extracts non-empty `(model, serial)` tuples (or filters them as non_trim).

**Files:**
- Test: `tests/test_log_derived_bugfixes_2026_05_30.py` — append `test_log_corpus_pipeline_smoke`

- [ ] **Step 1: Append the smoke test**

Append to `tests/test_log_derived_bugfixes_2026_05_30.py`:

```python
# ---------------------------------------------------------------------------
# Smoke test: filenames-only run through the routing + metadata pipeline.
# Confirms every distinct filename pattern from the 2026-05-30 log corpus
# either routes to non_trim OR produces non-empty (model, serial).
# ---------------------------------------------------------------------------

LOG_CORPUS = [
    # (filename, expected_route, expected_model_or_None, expected_serial_or_None)
    # Non-trim
    ("~$1844205_Final_Data.xlsx",           "non_trim", None, None),
    ("master-sn108_5-30-2025_1-52 PM.xls",  "non_trim", None, None),
    ("ChemCubed Ink Test 1.xlsx",           "non_trim", None, None),
    ("temp.xls",                            "non_trim", None, None),
    ("SN - Shop#.xlsx",                     "non_trim", None, None),
    # FT -- final SN <num>
    ("8340-1 final SN 5140.xls",            "final_test", "8340-1", "5140"),
    ("8340-1 final sn  116xls.xls",         "final_test", "8340-1", "116"),
    ("8340-1 final 215_6-4-2025_7-38 PM.xls", "final_test", "8340-1", "215"),
    # FT -- shop traveler
    ("1844205-shop1_5-18-2026 8-31-14 AM.xlsx", "final_test", "1844205", "shop1"),
    # FT -- existing patterns (regression guard)
    ("8340-1-sn470_5-30-2025_1-52 PM.xls",  "final_test", "8340-1", "470"),
    ("1081313-sn108_3-16-2011_12-17 PM.xls", "final_test", "1081313", "108"),
]


@pytest.mark.parametrize(
    "filename,expected_route,expected_model,expected_serial", LOG_CORPUS
)
def test_log_corpus_pipeline_smoke(
    filename, expected_route, expected_model, expected_serial, tmp_path
):
    """For each distinct filename pattern from the 2026-05-30 log corpus,
    detect_file_type must produce the expected route, and (for final_test
    files) the FT metadata extractor must populate model+serial.

    Path setup mirrors production:
      - non_trim files: caught by NON_TRIM_FILENAME_REGEXES at the filename
        branch -- no folder context needed.
      - final_test files: caught by FINAL_TEST_FOLDER_INDICATORS because
        production stores them in \\\\share\\TEST_DATA\\Test Station\\... .
        The filename-only FT regex at parser.py:1289 (\\bfinal\\s+\\d) does
        NOT match 'final SN 5140' or 'shop1' on its own; the folder
        heuristic is what routes them in real batches.  We mirror that.
    """
    from laser_trim_analyzer.core.parser import detect_file_type
    from laser_trim_analyzer.core.final_test_parser import FinalTestParser

    # For FT files, place under a "Test Station" parent so the folder
    # heuristic routes them as final_test (same as production).
    # For non_trim files, the filename regex catches them at the top of
    # detect_file_type before any folder/sheet check -- tmp_path is fine.
    if expected_route == "final_test":
        station = tmp_path / "Test Station"
        station.mkdir()
        fp = station / filename
    else:
        fp = tmp_path / filename

    route = detect_file_type(fp)
    assert route == expected_route, (
        f"{filename!r}: expected route {expected_route!r}, got {route!r}"
    )

    if expected_route == "final_test":
        parser = FinalTestParser()
        meta = parser._extract_metadata_from_filename(filename)
        assert meta["model"] == expected_model, (
            f"{filename!r}: expected model {expected_model!r}, "
            f"got {meta['model']!r}"
        )
        assert meta["serial"] == expected_serial, (
            f"{filename!r}: expected serial {expected_serial!r}, "
            f"got {meta['serial']!r}"
        )
```

- [ ] **Step 2: Run the smoke test**

Run: `pytest tests/test_log_derived_bugfixes_2026_05_30.py::test_log_corpus_pipeline_smoke -v`

Expected: 11 PASS, zero failures. If any FAIL, it's a real gap in one of Tasks 2-4 — go back and inspect the failing row before continuing.

- [ ] **Step 3: Run the full new test file as the final check**

```
pytest tests/test_log_derived_bugfixes_2026_05_30.py -v
```

Expected: ~30+ tests PASS, zero failures.

- [ ] **Step 4: Commit**

```bash
git add tests/test_log_derived_bugfixes_2026_05_30.py
git commit -m "test(parser): end-to-end smoke test against 2026-05-30 log corpus

Wires every distinct filename pattern from Work Files/5-30-26/ logs
through detect_file_type + _extract_metadata_from_filename to confirm
the routing and serial extraction fixes from Tasks 2-4 hold together
as one pipeline."
```

---

## Deferred (not in this plan)

These were identified in the same log analysis but explicitly deferred per user decision (2026-05-30):

- **Smoothness `_OS_Only_*` format detection** — 36 files produced "Smoothness parser returned no tracks" because their column layout matches neither Betatronix nor the generic format. Fix requires inspecting one of these files, which the user will bring home next time. Open a follow-up issue once a sample is available; the fix will live in `src/laser_trim_analyzer/core/smoothness_parser.py` (or wherever `_OS_Only_` routes through).

---

## Verification beyond unit tests

After all 7 tasks land, the recommended hands-on check is to run the app against a small folder containing one of each failing pattern (~10 files total). The expected outcome:
- Banner says `v5`
- All filtered files appear in the log as `Skipping non-trim file (...)` with no Excel I/O
- All `final SN`, `final sn  <n>xls`, and `shop<N>` files save successfully and link to their trim records
- Re-running the same folder a second time produces zero IntegrityErrors and the existing rows get UPDATEd

The 2026-05-28 terminal-style log was 24 IntegrityErrors in 40 seconds; post-fix the same folder should produce zero.
