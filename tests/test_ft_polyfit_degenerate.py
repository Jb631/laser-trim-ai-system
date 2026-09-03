"""The Format-2 ideal-line fit must never hand LAPACK a bad design matrix.

Ingesting `Work Files` printed 12 lines of

    ** On entry to DLASCL, parameter number  4 had an illegal value

straight to the console — the only noise a full ingest produced. All 12 came
from the one `np.polyfit` in `_extract_format2_tracks`, six per call, on two
real files (`8213-1_sn21.xls` / `_sn22.xls`, model 8213-1 under the 8213-4
Test Station folder). Their Position column is 2,998 rows of literal 0.0.

Why zeros are fatal: `np.polyfit` conditions the Vandermonde matrix by
dividing each column by `sqrt((col**2).sum())`. A constant-zero x column has
that norm equal to 0, so the division is 0/0 and fills the column with NaN.
LAPACK's XERBLA then prints those six lines with C `printf` — bypassing
`sys.stdout` entirely — and numpy raises `LinAlgError: SVD did not converge`.
An infinity in either column reaches the same place by the other route: the
norm overflows to inf and inf/inf is NaN again.

James's first full ingest on the new laptop is ~150k final-test files. The
console has to stay clean, and the fit has to refuse a degenerate input
rather than crash inside a broad `except` that leaves no explanation.

The stdout assertions here capture FILE DESCRIPTOR 1, not `sys.stdout`:
`capsys` replaces the Python object and would report success while XERBLA
wrote past it.
"""
import ctypes
import contextlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from laser_trim_analyzer.core.final_test_parser import FinalTestParser

REPO = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def captured_fd_stdout():
    """Capture everything written to fd 1, C-level output included.

    C stdout is BLOCK-buffered when fd 1 is not a tty, so XERBLA's bytes would
    otherwise sit in libc's buffer until interpreter exit — long after the
    redirect was undone, making a broken parser look silent. `fflush(NULL)`
    flushes every C stream, which is the only way to force them out in time.
    """
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    libc.fflush(None)
    saved = os.dup(1)
    sink = tempfile.TemporaryFile()
    os.dup2(sink.fileno(), 1)
    box = {}
    try:
        yield box
    finally:
        sys.stdout.flush()
        libc.fflush(None)
        os.dup2(saved, 1)
        os.close(saved)
        sink.seek(0)
        box["text"] = sink.read().decode(errors="replace")
        sink.close()


def write_format2(path: Path, measured, positions) -> Path:
    """A minimal Format-2 workbook: 'Data' + 'Charts' is the format's signature.

    Column order matches FINAL_TEST_FORMAT2_COLUMNS — measured 0, position 1,
    index 2 — and there is no header row, which is what the real Rout_ files
    look like.
    """
    data = pd.DataFrame({
        0: list(measured),
        1: list(positions),
        2: list(range(1, len(measured) + 1)),
    })
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Data", header=False, index=False)
        pd.DataFrame({0: ["chart placeholder"]}).to_excel(
            writer, sheet_name="Charts", header=False, index=False)
    return path


def clean_series(n=40):
    """A well-behaved ramp with a deliberate bow, so the fit has real residuals."""
    positions = [i / (n - 1) for i in range(n)]
    measured = [5.0 * p + 0.01 * np.sin(np.pi * p) for p in positions]
    return measured, positions


def only_track(parsed):
    assert len(parsed["tracks"]) == 1, f"expected one track, got {parsed['tracks']}"
    return parsed["tracks"][0]


@pytest.mark.parametrize("column", ["position", "measured"])
def test_non_finite_row_leaves_the_fit_untouched(tmp_path, column):
    """An inf row must be dropped, not fitted — and drop nothing else with it.

    The reference is the SAME sheet with that row physically removed: if the
    parser filters correctly, the two runs are indistinguishable down to the
    per-point error series. Comparing against a hand-computed slope would only
    prove the fit ran; comparing against the row-removed sheet proves the fit
    ran on exactly the right rows.
    """
    measured, positions = clean_series()

    reference = write_format2(tmp_path / "reference.xlsx", measured, positions)

    poisoned_measured = list(measured)
    poisoned_positions = list(positions)
    if column == "position":
        poisoned_measured.insert(20, 2.5)
        poisoned_positions.insert(20, float("inf"))
    else:
        poisoned_measured.insert(20, float("inf"))
        poisoned_positions.insert(20, 0.5)
    poisoned = write_format2(tmp_path / "poisoned.xlsx",
                             poisoned_measured, poisoned_positions)

    parser = FinalTestParser()
    with captured_fd_stdout() as out:
        expected = parser.parse_file(reference)
        actual = parser.parse_file(poisoned)

    assert out["text"] == "", (
        f"the parser wrote to stdout: {out['text']!r}")

    want, got = only_track(expected), only_track(actual)
    assert got["positions"] == want["positions"]
    assert got["measured_values"] == want["measured_values"]
    assert got["errors"] == pytest.approx(want["errors"], abs=1e-12)
    assert got["linearity_error"] == pytest.approx(want["linearity_error"], abs=1e-12)
    assert got["max_deviation_position"] == pytest.approx(
        want["max_deviation_position"], abs=1e-12)
    # The fit itself: same slope and intercept through the surviving points.
    for track in (want, got):
        track["_fit"] = np.polyfit(np.array(track["positions"]),
                                   np.array(track["measured_values"]), 1)
    assert got["_fit"] == pytest.approx(want["_fit"], abs=1e-12)


def test_constant_position_column_is_refused_quietly(tmp_path):
    """The real 8213-1 shape: every position identical, so no line exists.

    Two things must hold. The console stays clean — no XERBLA, no traceback
    text. And the parser must NOT invent a track: a Format-2 track carries no
    spec limits, so a fabricated `errors` array of zeros would land in
    `final_test_tracks` as linearity_error 0.0 — a perfect score for a file
    that measured nothing, dragging down every FT average it joins. Four
    other row loops in this parser already refuse to fabricate a 0.0 for the
    same reason; a whole file deserves at least the same treatment.
    """
    measured, _ = clean_series()
    path = write_format2(tmp_path / "constant_position.xlsx",
                         measured, [0.0] * len(measured))

    parser = FinalTestParser()
    with captured_fd_stdout() as out:
        parsed = parser.parse_file(path)

    assert "DLASCL" not in out["text"] and out["text"] == "", (
        f"degenerate fit wrote to stdout: {out['text']!r}")
    assert parsed["format"] == "format2"
    assert parsed["tracks"] == [], (
        "a position column with no spread cannot yield a linearity result; "
        f"the parser invented {parsed['tracks']}")


def test_a_single_point_still_takes_the_no_fit_path(tmp_path):
    """One row is under-determined too, and was already handled — keep it so."""
    path = write_format2(tmp_path / "one_point.xlsx", [1.23], [0.5])
    parser = FinalTestParser()
    with captured_fd_stdout() as out:
        parsed = parser.parse_file(path)
    assert out["text"] == ""
    track = only_track(parsed)
    assert track["errors"] == [0.0]
    assert track["linearity_error"] == 0.0


REAL_FILES = [
    REPO / "Work Files/Sample_Base_2026-04-10/Test Station/8213-4/8213-1_sn21.xls",
    REPO / "Work Files/Sample_Base_2026-04-10/Test Station/8213-4/8213-1_sn22.xls",
]


@pytest.mark.parametrize("path", REAL_FILES, ids=lambda p: p.name)
def test_the_two_real_offenders_parse_silently(path):
    """The regression itself, against the files that produced all 12 lines.

    Skipped where the sample corpus is absent (the work machine keeps it
    elsewhere); on a checkout that has it, this is the check that would have
    caught the noise in the first place.
    """
    if not path.exists():
        pytest.skip(f"sample corpus not present: {path}")
    parser = FinalTestParser()
    with captured_fd_stdout() as out:
        parsed = parser.parse_file(path)
    assert out["text"] == "", f"{path.name} wrote to stdout: {out['text']!r}"
    assert parsed["tracks"] == []
    # Still identified, so the file is recorded and traceable even with no track.
    assert parsed["metadata"]["model"] == "8213-1"
