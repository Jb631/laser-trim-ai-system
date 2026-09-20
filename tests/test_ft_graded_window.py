"""Final-test grading is restricted to the rows the sheet itself grades.

THE DISPOSITION IS THE APP'S GRADE. Every assertion here is written from that
premise: `linearity_pass` is the analyzer's corrected per-point verdict, and
the station's own PASSED/FAILED is a reference column that must never replace
it. What this suite pins is which POINTS the app is allowed to grade, and that
the reference is read correctly and stored beside the verdict.

The six fixtures are real production files, each carrying a distinct template
problem found on 2026-09-13:

  8232-1 sn180   PASSED, flags on rows 5..49 of 57. The six-point lead-in reads
                 0 V against a theory of -0.045 V; grading it manufactured a
                 0.047 error against a +/-0.010 band and failed the file.
  8232-1 sn176   FAILED, flags on exactly eleven rows inside that window.
  7539-2 sn23    FAILED with 149 flags, and the app FAILS it too (11 points out
                 after offset correction). Until 2026-09-20 the app PASSED it,
                 and that was recorded here as "the offset correction rescues
                 it — a real and expected disagreement". It was not: column D is
                 88.95% populated, so the parser recomputed the errors and
                 divided them by full scale while the limits stayed in volts
                 (see tests/test_ft_recompute_units.py). The disagreement was a
                 units bug wearing the offset correction's clothes.
  7458 sn7       FAILED although column D sits inside G/H: this template grades
                 against limits in another unit, and writes a literal 0 flag on
                 rows it never measured.
  8639-30 sn88b  Format 3 — a verdict cell and a flag column PER TRACK SHEET.
  8434ct-1118D   Format 4 — one verdict cell, and a flag column that could not
                 be identified, so flags stay NULL rather than being guessed.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from laser_trim_analyzer.core.analyzer import Analyzer, parse_exclude_points
from laser_trim_analyzer.core.final_test_parser import FinalTestParser
from laser_trim_analyzer.core.ft_regrade import (
    analyzer_errors, grade_ft_track, merged_exclude_points,
    out_of_window_indices)

FIXTURES = Path(__file__).parent / "fixtures" / "final_test"
SN180 = FIXTURES / "8232-1-sn180_3-28-2026_9-37 AM.xls"
SN176 = FIXTURES / "8232-1-sn176_3-28-2026_7-59 AM.xls"
SN23 = FIXTURES / "7539-2-sn23_1-22-2026_2-44 PM.xls"
SN7 = FIXTURES / "7458-sn7_4-2-2026_3-19 PM.xls"
MULTITRACK = FIXTURES / "8639-30-sn88b_11-2-2015_9-19 AM.xls"
FORMAT4 = FIXTURES / "8434ct-1118D.xls"

NO_SPEC = {"linearity_type": None, "angle_spec": None, "angle_tol": None,
           "angle_tol_type": None, "exclude_points": None}


@pytest.fixture(scope="module")
def parser():
    return FinalTestParser()


def _track(parser, path, index=0):
    return parser.parse_file(path)["tracks"][index]


# ---- the window is read off the file --------------------------------------

def test_8232_1_sn180_window_is_the_station_s_forty_five_graded_rows(parser):
    parsed = parser.parse_file(SN180)
    track = parsed["tracks"][0]
    assert len(track["errors"]) == 57
    assert track["graded_window"] == (5, 49)
    assert track["graded_window_source"] == "flags"
    # The file DECLARES six and six, which would be 6..50 — off by one from
    # what it actually flagged. The flags are the evidence and they win; the
    # declaration is kept for disclosure, not for grading.
    assert (track["ignore_start"], track["ignore_end"]) == (6, 6)


def test_8232_1_sn180_station_passed_with_no_flag_set(parser):
    parsed = parser.parse_file(SN180)
    assert parsed["test_results"]["station_linearity_pass"] is True
    track = parsed["tracks"][0]
    assert track["station_linearity_pass"] is True
    assert track["station_fail_points"] == 0
    assert parsed["test_results"]["station_cell_flag_conflict"] is False


def test_8232_1_sn176_flags_are_exactly_the_eleven_rows_the_station_marked(parser):
    parsed = parser.parse_file(SN176)
    track = parsed["tracks"][0]
    flagged = [i for i, f in enumerate(track["station_flags"]) if f == 1]
    assert flagged == [23, 24, 27, 29, 30, 31, 32, 36, 37, 41, 42]
    assert track["station_fail_points"] == 11
    assert track["station_linearity_pass"] is False
    assert parsed["test_results"]["station_linearity_pass"] is False
    # Every flagged row is inside the window, which is what makes the window
    # credible: the station never flagged a row it claims not to grade.
    low, high = track["graded_window"]
    assert all(low <= i <= high for i in flagged)


def test_7458_sn7_window_excludes_the_rows_with_no_error_reading(parser):
    """This template writes a literal 0 flag on rows it never measured.

    Taking those at face value would put unmeasured points back inside the
    graded window, where a zero-tolerance grade counts each one as a fail —
    the exact defect this change exists to remove. So a row with no error
    reading cannot bound the window.
    """
    track = _track(parser, SN7)
    low, high = track["graded_window"]
    blank = [i for i, e in enumerate(track["errors"]) if e is None]
    assert blank, "fixture must carry blank error cells to be meaningful"
    assert all(i < low or i > high for i in blank)
    assert track["station_linearity_pass"] is False


def test_7458_sn7_station_says_failed_though_column_d_sits_inside_g_h(parser):
    """The station's limits are not this template's G/H columns.

    Its flags say 39 points are out; the raw error column is inside +/-0.1 at
    every one of them. Both are recorded; neither is "corrected" into the
    other, because the app cannot know which unit the station's limits are in.
    """
    parsed = parser.parse_file(SN7)
    track = parsed["tracks"][0]
    assert parsed["test_results"]["station_linearity_pass"] is False
    assert track["station_fail_points"] == 39
    low, high = track["graded_window"]
    inside = [e for i, e in enumerate(track["errors"])
              if low <= i <= high and e is not None]
    assert max(abs(e) for e in inside) < 0.1
    assert parsed["test_results"]["station_cell_flag_conflict"] is False


def test_7539_2_sn23_station_failed_with_149_flags(parser):
    parsed = parser.parse_file(SN23)
    track = parsed["tracks"][0]
    assert parsed["test_results"]["station_linearity_pass"] is False
    assert track["station_fail_points"] == 149
    assert track["graded_window"] == (10, 170)
    assert track["graded_window_source"] == "flags"


def test_7539_2_sn23_app_grade_is_computed_only_from_in_window_points(parser):
    """The app may legitimately disagree with the station — but not on this file.

    The owner's rule is "grade the error and correct the offset", so an app PASS
    against a station FAIL can be real. Here it is not: in volts, the best offset
    still leaves points outside the limits, and the app's old PASS came from
    errors that had been divided by full scale (fixed 2026-09-20). Pinned: the
    verdict's INPUT (no point outside the station's window may take part) AND,
    now, the verdict itself.
    """
    track = _track(parser, SN23)
    result = grade_ft_track(Analyzer(), track, NO_SPEC, model="7539-2")
    excluded = parse_exclude_points(
        merged_exclude_points(None, out_of_window_indices(
            len(track["errors"]), track["graded_window"])))
    assert excluded == {i for i in range(len(track["errors"]))
                        if i < 10 or i > 170}
    assert result.linearity_pass is False
    # Its fail points can only come from inside the window -- and there are some.
    assert 0 < track["linearity_fail_points"] <= (170 - 10 + 1)


# ---- format coverage -------------------------------------------------------

def test_format3_reads_a_verdict_cell_per_track_sheet(parser):
    parsed = parser.parse_file(MULTITRACK)
    assert parsed["format"] == "format3_multitrack"
    tracks = parsed["tracks"]
    assert [t["track_id"] for t in tracks] == ["A", "B", "C"]
    for track in tracks:
        assert track["station_cell_pass"] is True
        assert track["station_linearity_pass"] is True
        assert track["graded_window_source"] == "flags"
    # Zero-tolerance: the file passes only because every sheet passed.
    assert parsed["test_results"]["station_linearity_pass"] is True


def test_format4_stores_the_cell_and_refuses_to_guess_a_flag_column(parser):
    parsed = parser.parse_file(FORMAT4)
    assert parsed["format"] == "format4_parameters"
    assert parsed["test_results"]["station_linearity_pass"] is False
    track = parsed["tracks"][0]
    assert track["station_flags"] is None
    assert track["station_fail_points"] is None
    assert track["graded_window"] is None
    assert track["graded_window_source"] == "all_rows"


# ---- blanks are not zeros --------------------------------------------------

def test_blank_error_cells_are_none_never_zero(parser):
    """0.0 is dead centre of every band — the most flattering value a
    zero-tolerance metric can hold, and what used to be written here."""
    track = _track(parser, SN180)
    assert track["errors"][55] is None
    assert track["errors"][56] is None
    assert not any(e == 0.0 for e in track["errors"])


def test_linearity_error_skips_the_blanks(parser):
    track = _track(parser, SN180)
    measured = [abs(e) for e in track["errors"] if e is not None]
    assert track["linearity_error"] == pytest.approx(max(measured))


def test_analyzer_errors_turns_none_into_nan_and_leaves_the_rest_alone():
    """The analyzer builds `e + offset`, which None cannot survive and NaN
    can — and NaN already means "unmeasured" everywhere downstream."""
    out = analyzer_errors([0.1, None, -0.2])
    assert out[0] == 0.1 and out[2] == -0.2
    assert np.isnan(out[1])
    assert analyzer_errors([0.1, 0.2]) == [0.1, 0.2]


# ---- the exclusion channel -------------------------------------------------

def test_out_of_window_indices_names_both_ends():
    assert out_of_window_indices(10, (2, 7)) == {0, 1, 8, 9}
    assert out_of_window_indices(10, None) == set()
    assert out_of_window_indices(0, (2, 7)) == set()


def test_merged_exclude_points_keeps_the_spec_string_when_there_is_no_window():
    """A file with no window must hand the analyzer the IDENTICAL argument it
    received before this change — that is what makes the no-op provable."""
    spec = json.dumps({"exclude": [0, 1]})
    assert merged_exclude_points(spec, set()) is spec
    assert merged_exclude_points(None, set()) is None


def test_merged_exclude_points_unions_the_model_spec_with_the_window():
    spec = json.dumps({"exclude": [3, [10, 12]]})
    merged = parse_exclude_points(merged_exclude_points(spec, {0, 1, 98, 99}))
    assert merged == {0, 1, 3, 10, 11, 12, 98, 99}


# ---- the grade itself ------------------------------------------------------

def test_8232_1_sn180_passes_once_the_lead_in_is_out_of_the_grade(parser):
    """The headline case. Nine fail points before, none after, same analyzer."""
    track = _track(parser, SN180)
    result = grade_ft_track(Analyzer(), track, NO_SPEC, model="8232-1")
    assert result.linearity_pass is True
    assert track["linearity_pass"] is True
    assert track["linearity_fail_points"] == 0


def test_grading_without_the_window_still_fails_8232_1_sn180(parser):
    """The control for the test above: it is the WINDOW doing the work, not
    some other change. Grade the same track with the window removed and the
    old verdict comes back."""
    track = _track(parser, SN180)
    track["graded_window"] = None
    result = grade_ft_track(Analyzer(), track, NO_SPEC, model="8232-1")
    assert result.linearity_pass is False
    assert track["linearity_fail_points"] > 0


def test_the_processor_does_not_replace_the_app_grade_with_the_cell(parser):
    """8232-1 sn176: the station FAILED it and the app's corrected grade does
    not. The app's verdict is what lands on the track — the cell is reference
    only, and copying it would be exactly the change the owner refused."""
    parsed = parser.parse_file(SN176)
    track = parsed["tracks"][0]
    assert parsed["test_results"]["station_linearity_pass"] is False
    result = grade_ft_track(Analyzer(), track, NO_SPEC, model="8232-1")
    assert track["linearity_pass"] is result.linearity_pass
    assert track["station_linearity_pass"] is False
    # Same object, different question: the two must not be merged.
    assert track["linearity_pass"] is not track["station_linearity_pass"] \
        or track["linearity_pass"] is False


def test_a_track_with_nothing_gradeable_is_null_not_a_pass(parser):
    """Zero fail points over zero graded points is the right answer to "how
    many failed" and the wrong one to "did it pass"."""
    track = _track(parser, SN180)
    spec = dict(NO_SPEC)
    spec["exclude_points"] = json.dumps(
        {"exclude": [[0, len(track["errors"])]]})
    grade_ft_track(Analyzer(), track, spec, model="8232-1")
    assert track["linearity_pass"] is None
    assert track["linearity_fail_points"] == 0


# ---- the no-op control set -------------------------------------------------

def test_a_full_window_file_grades_exactly_as_it_did_before(parser):
    """Guardrail: on a file whose window is the whole sweep and whose error
    column has no blanks, the analyzer must receive byte-identical inputs, so
    the verdict cannot move. Proven at scale against the pre-change tree on
    372 local sample files; pinned here so it stays true.
    """
    track = _track(parser, MULTITRACK)  # 8639-30 sheet A: ignore 0 and 0
    track["graded_window"] = (0, len(track["errors"]) - 1)
    assert all(e is not None for e in track["errors"])
    assert analyzer_errors(track["errors"]) == list(track["errors"])
    assert merged_exclude_points(
        None, out_of_window_indices(len(track["errors"]),
                                    track["graded_window"])) is None

    windowed = dict(track)
    unwindowed = dict(track)
    unwindowed["graded_window"] = None
    a = grade_ft_track(Analyzer(), windowed, NO_SPEC, model="8639-30")
    b = grade_ft_track(Analyzer(), unwindowed, NO_SPEC, model="8639-30")
    assert a.linearity_pass == b.linearity_pass
    assert a.linearity_fail_points == b.linearity_fail_points
    assert a.linearity_error == b.linearity_error
    assert a.optimal_offset == b.optimal_offset
    assert a.optimal_slope == b.optimal_slope


# ---- database: migration, resolver, round trip -----------------------------

def _manager(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    return DatabaseManager(tmp_path / "ft.db")


def test_migration_adds_the_reference_columns_and_is_idempotent(tmp_path):
    import sqlite3
    path = tmp_path / "ft.db"
    from laser_trim_analyzer.database.manager import DatabaseManager
    DatabaseManager(path)
    DatabaseManager(path)          # second open must not raise
    conn = sqlite3.connect(path)
    try:
        result_cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(final_test_results)")}
        track_cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(final_test_tracks)")}
    finally:
        conn.close()
    assert {"station_linearity_pass", "station_cell_flag_conflict",
            "graded_window_source"} <= result_cols
    assert {"station_flags", "station_fail_points", "graded_start",
            "graded_end", "ignore_start", "ignore_end"} <= track_cols


def test_resolver_prefers_the_app_track_verdicts_not_the_station_cell(tmp_path):
    db = _manager(tmp_path)
    resolve = db.resolve_final_test_linearity_pass
    # The station cell says PASSED; one app-graded track says otherwise.
    assert resolve({"linearity_pass": True, "station_linearity_pass": True},
                   [{"linearity_pass": True}, {"linearity_pass": False}]) is False
    assert resolve({"linearity_pass": False},
                   [{"linearity_pass": True}]) is True
    # Nothing gradeable anywhere: NULL, not the header's opinion.
    assert resolve({"linearity_pass": True},
                   [{"linearity_pass": None}]) is None
    # No tracks at all is the one case the header still answers.
    assert resolve({"linearity_pass": True}, []) is True


def test_saving_stores_the_station_reference_beside_the_app_verdict(tmp_path, parser):
    from laser_trim_analyzer.database.models import (
        FinalTestResult, FinalTestTrack)
    db = _manager(tmp_path)
    parsed = parser.parse_file(SN176)
    track = parsed["tracks"][0]
    grade_ft_track(Analyzer(), track, NO_SPEC, model="8232-1")
    metadata = dict(parsed["metadata"])
    metadata["file_path"] = str(SN176)
    ft_id = db.save_final_test(
        metadata=metadata, tracks=parsed["tracks"],
        test_results=parsed["test_results"], file_hash=parsed["file_hash"])

    with db.session() as session:
        row = session.get(FinalTestResult, ft_id)
        stored = session.query(FinalTestTrack).filter(
            FinalTestTrack.final_test_id == ft_id).one()
        assert row.station_linearity_pass is False       # the sheet's cell
        assert row.graded_window_source == "flags"
        assert row.linearity_pass == track["linearity_pass"]   # the app's grade
        assert stored.station_fail_points == 11
        assert (stored.graded_start, stored.graded_end) == (5, 49)
        assert (stored.ignore_start, stored.ignore_end) == (6, 6)
        assert sum(1 for f in stored.station_flags if f == 1) == 11
    assert db.count_legacy_ft_verdicts() == 0


def test_a_row_saved_by_the_old_code_counts_as_legacy(tmp_path, parser):
    """`graded_window_source IS NULL` is the whole marker — it must not be
    given a default, or the re-grade pass loses its work list."""
    from sqlalchemy import text
    db = _manager(tmp_path)
    parsed = parser.parse_file(SN180)
    metadata = dict(parsed["metadata"])
    metadata["file_path"] = str(SN180)
    db.save_final_test(metadata=metadata, tracks=parsed["tracks"],
                       test_results=parsed["test_results"],
                       file_hash=parsed["file_hash"])
    assert db.count_legacy_ft_verdicts() == 0
    with db.session() as session:
        session.execute(text(
            "UPDATE final_test_results SET graded_window_source = NULL"))
        session.commit()
    assert db.count_legacy_ft_verdicts() == 1
    assert len(db.get_final_tests_for_regrade(only_legacy=True)) == 1


def test_regrade_dry_run_writes_nothing_and_apply_clears_the_legacy_mark(
        tmp_path, parser):
    from sqlalchemy import text
    from laser_trim_analyzer.core.ft_regrade import regrade_final_tests
    from laser_trim_analyzer.database.models import FinalTestResult

    db = _manager(tmp_path)
    parsed = parser.parse_file(SN180)
    metadata = dict(parsed["metadata"])
    metadata["file_path"] = str(SN180)
    ft_id = db.save_final_test(metadata=metadata, tracks=parsed["tracks"],
                               test_results=parsed["test_results"],
                               file_hash=parsed["file_hash"])
    # Rewind the row to what the pre-window code would have stored.
    with db.session() as session:
        session.execute(text(
            "UPDATE final_test_results SET graded_window_source = NULL, "
            "linearity_pass = 0 WHERE id = :i"), {"i": ft_id})
        session.commit()

    dry = regrade_final_tests(db, only_legacy=True, apply=False)
    assert dry.examined == 1
    assert dry.changed == 1
    assert dry.transitions()["fail_to_pass"] == 1
    with db.session() as session:
        assert session.get(FinalTestResult, ft_id).linearity_pass is False
    assert db.count_legacy_ft_verdicts() == 1, "a dry run must write nothing"

    applied = regrade_final_tests(db, only_legacy=True, apply=True)
    assert applied.changed == 1
    with db.session() as session:
        row = session.get(FinalTestResult, ft_id)
        assert row.linearity_pass is True
        assert row.graded_window_source == "flags"
        assert row.station_linearity_pass is True
    assert db.count_legacy_ft_verdicts() == 0


def test_regrade_counts_an_unreachable_source_instead_of_failing(tmp_path, parser):
    from sqlalchemy import text
    from laser_trim_analyzer.core.ft_regrade import regrade_final_tests
    db = _manager(tmp_path)
    parsed = parser.parse_file(SN180)
    metadata = dict(parsed["metadata"])
    metadata["file_path"] = r"\\192.168.66.9\BTXData\nowhere\x.xls"
    db.save_final_test(metadata=metadata, tracks=parsed["tracks"],
                       test_results=parsed["test_results"],
                       file_hash=parsed["file_hash"])
    with db.session() as session:
        session.execute(text(
            "UPDATE final_test_results SET graded_window_source = NULL"))
        session.commit()
    report = regrade_final_tests(db, only_legacy=True, apply=True)
    assert report.missing == 1
    assert report.changed == 0
    assert "unreachable" in report.summary()


# ---- what the reader is told -----------------------------------------------

def test_the_home_notice_says_nothing_when_there_is_nothing_to_say():
    from laser_trim_analyzer.core.ft_regrade import legacy_ft_notice
    assert legacy_ft_notice(0) == ""
    assert "150,202" in legacy_ft_notice(150202)
    assert "Re-grade final tests" in legacy_ft_notice(1)


def test_the_overlay_legend_names_both_verdicts_and_flags_a_difference():
    from laser_trim_analyzer.core.ft_overlay import ft_legend
    agree = ft_legend({"linearity_pass": False, "linearity_fail_points": 2,
                       "station_pass": False, "station_fail_points": 11})
    assert agree == "Final test: FAIL (2 points) · station: FAILED (11 flagged)"
    differ = ft_legend({"linearity_pass": True, "station_pass": False,
                        "station_fail_points": 149})
    assert "station: FAILED (149 flagged)" in differ
    assert "app grade differs" in differ
    assert ft_legend({"linearity_pass": None}) == "Final test: not graded"


def test_the_spec_cell_shows_the_bowtie_as_a_range_not_its_mean():
    """A mean half-width is a limit no station ever applied at any point."""
    from laser_trim_analyzer.export.unit_chart import _spec_band_text
    assert _spec_band_text({"upper_limits": [0.2, 0.05, 0.05],
                            "lower_limits": [-0.2, -0.05, -0.05]}) \
        == "±0.050–0.200"
    # No arrays: the scalar survives, but labelled so it cannot be misread.
    assert "(avg)" in _spec_band_text({"linearity_spec": 0.009})
