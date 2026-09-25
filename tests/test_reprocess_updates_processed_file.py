"""A reprocess records what THIS run found in `processed_files` (2026-09-25).

`_update_existing_analysis` (the path `save_analysis`/`save_batch` take when a
file's (filename, file_date, model, serial) identity already has a row) never
called `_record_processed_file` -- only the "create new" branch of
`save_analysis` did. So re-processing a file left `processed_files.success` /
`error_message` exactly as the EARLIER run wrote them: a file that failed once
and then processed cleanly still read as failed on the Model page, and a file
that processed cleanly and then started failing (a regressed parser, a file
that changed on disk) still read as clean.

The fix makes `_update_existing_analysis` call `_record_processed_file` itself,
the same way the "create new" branch of `save_analysis` already does, in the
SAME session/transaction as the analysis row it just wrote.

Trim rows have always updated the `analysis_results`/`track_results` rows in
place on a re-process (comment at `_update_existing_analysis`); this closes the
one row that did not follow -- `processed_files`, the one the Model page and
the retry-skip logic (`_load_processed_hashes`) actually read.
"""
from datetime import datetime
from pathlib import Path

import sqlalchemy as sa

from laser_trim_analyzer.core.models import (
    AnalysisResult, AnalysisStatus, FileMetadata, RiskCategory, SystemType, TrackData,
)
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import ProcessedFile

MODEL = "REPROC-1"


def _meta(file_path: Path, serial="SN1") -> FileMetadata:
    return FileMetadata(
        filename=file_path.name,
        file_path=file_path,
        file_date=datetime(2026, 1, 1),
        model=MODEL,
        serial=serial,
        system=SystemType.B,
    )


def _track(status: AnalysisStatus, linearity_pass) -> TrackData:
    return TrackData(
        track_id="default",
        status=status,
        travel_length=1.0,
        linearity_spec=0.01,
        optimal_offset=0.0,
        linearity_pass=linearity_pass,
        linearity_fail_points=0 if linearity_pass else 3,
    )


def _error_result(file_path: Path, serial="SN1") -> AnalysisResult:
    return AnalysisResult(
        metadata=_meta(file_path, serial),
        overall_status=AnalysisStatus.ERROR,
        processing_time=0.1,
        tracks=[_track(AnalysisStatus.ERROR, None)],
        errors=["Could not find data start"],
    )


def _clean_result(file_path: Path, serial="SN1") -> AnalysisResult:
    return AnalysisResult(
        metadata=_meta(file_path, serial),
        overall_status=AnalysisStatus.PASS,
        processing_time=0.1,
        tracks=[_track(AnalysisStatus.PASS, True)],
    )


def _processed_file_row(db, file_path: Path):
    """The one real `processed_files` row for this path (never the synthetic
    skip-marker row -- that one carries analysis_id=NULL)."""
    with db.session() as s:
        rows = (s.query(ProcessedFile)
                 .filter(ProcessedFile.file_path == str(file_path))
                 .filter(ProcessedFile.analysis_id.isnot(None))
                 .all())
        assert len(rows) == 1, (
            f"expected exactly one real processed_files row for {file_path}, "
            f"found {len(rows)}"
        )
        r = rows[0]
        return {"success": r.success, "error_message": r.error_message,
                "analysis_id": r.analysis_id}


def _all_processed_file_rows_for(db, file_path: Path):
    with db.session() as s:
        return (s.query(ProcessedFile)
                 .filter(ProcessedFile.file_path == str(file_path))
                 .filter(ProcessedFile.analysis_id.isnot(None))
                 .count())


# ---------------------------------------------------------------------------
# ERROR first, then a clean reprocess of the SAME file
# ---------------------------------------------------------------------------

def test_error_then_clean_reprocess_reads_success_with_no_error_message(tmp_path):
    f = tmp_path / "unit.xls"
    f.write_bytes(b"not a real workbook, content just needs to be stable")
    db = DatabaseManager(tmp_path / "t.db")

    first_id = db.save_analysis(_error_result(f))
    before = _processed_file_row(db, f)
    assert before["success"] is False
    assert before["error_message"] == "Could not find data start"

    second_id = db.save_analysis(_clean_result(f))
    assert second_id == first_id, "must update the same analysis row, not create another"

    after = _processed_file_row(db, f)
    assert after["success"] is True, (
        "a clean reprocess must flip processed_files.success to True -- it "
        f"still reads {after}"
    )
    assert after["error_message"] is None, (
        "a clean reprocess must clear the stale error_message -- it still "
        f"reads {after}"
    )
    assert _all_processed_file_rows_for(db, f) == 1, "reprocessing must not grow the row count"


# ---------------------------------------------------------------------------
# clean first, then a failed reprocess of the SAME file
# ---------------------------------------------------------------------------

def test_clean_then_failed_reprocess_reads_failed_with_the_new_message(tmp_path):
    f = tmp_path / "unit2.xls"
    f.write_bytes(b"also not a real workbook, but stable content")
    db = DatabaseManager(tmp_path / "t.db")

    first_id = db.save_analysis(_clean_result(f, serial="SN2"))
    before = _processed_file_row(db, f)
    assert before["success"] is True
    assert before["error_message"] is None

    bad = _error_result(f, serial="SN2")
    bad.errors = ["Bad limit columns on reprocess"]
    second_id = db.save_analysis(bad)
    assert second_id == first_id, "must update the same analysis row, not create another"

    after = _processed_file_row(db, f)
    assert after["success"] is False, (
        f"a failed reprocess must flip processed_files.success to False -- it still reads {after}"
    )
    assert after["error_message"] == "Bad limit columns on reprocess", (
        f"a failed reprocess must record THIS run's message, not the old one -- got {after}"
    )
    assert _all_processed_file_rows_for(db, f) == 1, "reprocessing must not grow the row count"


# ---------------------------------------------------------------------------
# save_batch's update branch goes through the same function -- pin it too
# ---------------------------------------------------------------------------

def test_save_batch_reprocess_also_updates_processed_file(tmp_path):
    f = tmp_path / "unit3.xls"
    f.write_bytes(b"batch path content")
    db = DatabaseManager(tmp_path / "t.db")

    db.save_analysis(_error_result(f, serial="SN3"))
    before = _processed_file_row(db, f)
    assert before["success"] is False

    db.save_batch([_clean_result(f, serial="SN3")])

    after = _processed_file_row(db, f)
    assert after["success"] is True, f"save_batch's reprocess must update processed_files too -- got {after}"
    assert after["error_message"] is None
    assert _all_processed_file_rows_for(db, f) == 1


def test_update_existing_analysis_fallback_with_no_existing_row_still_records_processed_file(tmp_path):
    """`_update_existing_analysis`'s own defensive fallback ("no existing
    record found, create new") mirrors `save_analysis`'s create-new branch --
    which calls `_record_processed_file`. Normally unreachable through
    `save_analysis`/`save_batch` (both already confirmed `existing` truthy
    before calling this), since it is called directly here -- but it must not
    be the one branch silently left different from the rest of this fix.
    """
    f = tmp_path / "unit4.xls"
    f.write_bytes(b"direct fallback content")
    db = DatabaseManager(tmp_path / "t.db")
    result = _clean_result(f, serial="SN4")

    with db.session() as s:
        new_id = db._update_existing_analysis(s, result)

    after = _processed_file_row(db, f)
    assert after["success"] is True
    assert after["analysis_id"] == new_id
