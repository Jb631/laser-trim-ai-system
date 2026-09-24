"""Every ERROR says why (2026-09-23: none of the 237 on the rebuilt database did).

`analysis_results` gained one nullable column, `error_reason` -- filled by the processor
for every ERROR result (from the ERROR tracks' own `linearity_spec_warning` /
`anomaly_reason`, or the file-level error message when there were no tracks at all) and
written on both insert and re-process. The Model page reads
`COALESCE(error_reason, linearity_spec_warning, anomaly_reason)` so the 234 rows written
before this column existed explain themselves too, with no back-fill (design doc §3,
ruling 3c).
"""
from pathlib import Path

import pytest

from laser_trim_analyzer.core.models import AnalysisStatus


@pytest.fixture
def db(tmp_path, monkeypatch):
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    d = mgr.DatabaseManager(tmp_path / "reason.db")
    monkeypatch.setattr(mgr, "_db_manager", d, raising=False)
    monkeypatch.setattr(dbpkg, "_db_manager", d, raising=False)
    return d


# ---------------------------------------------------------------------------
# error_reason_of() -- the pure aggregator
# ---------------------------------------------------------------------------

def test_a_file_level_error_keeps_its_message():
    from laser_trim_analyzer.core.processor import Processor
    p = Processor(use_ml=False)
    r = p._create_error_result(p._create_minimal_metadata(Path("x.xls")), "No valid track data found", 0.0)
    assert r.overall_status == AnalysisStatus.ERROR
    assert r.error_reason == "No valid track data found"


def test_a_track_level_error_names_the_track_reason(monkeypatch):
    """An ERROR that comes from a track (bad limits, too few points) gets that track's
    words -- 234 of the 237 ERRORs on the rebuild are this kind."""
    from laser_trim_analyzer.core.processor import error_reason_of
    tracks = [type("T", (), {"status": AnalysisStatus.ERROR, "linearity_spec_warning": None,
                             "anomaly_reason": "Insufficient data points", "track_id": "TRK1"})(),
              type("T", (), {"status": AnalysisStatus.PASS, "linearity_spec_warning": None,
                             "anomaly_reason": None, "track_id": "TRK2"})()]
    assert error_reason_of(tracks, AnalysisStatus.ERROR) == "TRK1: Insufficient data points"
    assert error_reason_of(tracks, AnalysisStatus.PASS) is None


def test_linearity_spec_warning_wins_over_anomaly_reason_when_both_are_set():
    """8097's two co-occurring tracks (Q3): a bad limit column AND a linear-slope anomaly
    on the same track. linearity_spec_warning is the more specific of the two -- it says
    the SPEC couldn't be trusted, not just that the sweep looked odd."""
    from laser_trim_analyzer.core.processor import error_reason_of
    t = type("T", (), {"status": AnalysisStatus.ERROR,
                       "linearity_spec_warning": "limit columns are not a +/- band",
                       "anomaly_reason": "Linear slope pattern (R²=1.000, range=10.00)",
                       "track_id": "TRK1"})()
    assert error_reason_of([t], AnalysisStatus.ERROR) == "TRK1: limit columns are not a +/- band"


def test_an_error_with_no_track_words_gets_a_generic_message_not_a_blank():
    """overall_status is ERROR but no ERROR track carries linearity_spec_warning or
    anomaly_reason -- must never come back as "" (falsy but still "explained")."""
    from laser_trim_analyzer.core.processor import error_reason_of
    t = type("T", (), {"status": AnalysisStatus.ERROR, "linearity_spec_warning": None,
                       "anomaly_reason": None, "track_id": "TRK1"})()
    assert error_reason_of([t], AnalysisStatus.ERROR) == "ERROR with no recorded reason"


def test_multiple_error_tracks_are_joined():
    from laser_trim_analyzer.core.processor import error_reason_of
    tracks = [type("T", (), {"status": AnalysisStatus.ERROR, "linearity_spec_warning": None,
                             "anomaly_reason": "Insufficient data points", "track_id": "TRK1"})(),
              type("T", (), {"status": AnalysisStatus.ERROR, "linearity_spec_warning": "bad limits",
                             "anomaly_reason": None, "track_id": "TRK2"})()]
    assert error_reason_of(tracks, AnalysisStatus.ERROR) == "TRK1: Insufficient data points; TRK2: bad limits"


# ---------------------------------------------------------------------------
# enforce_measurement_backed_verdict -- ruling 3b
# ---------------------------------------------------------------------------

def _measured_track(status, positions, errors, linearity_spec_warning=None):
    from laser_trim_analyzer.core.models import TrackData
    return TrackData(
        track_id="TRK1", status=AnalysisStatus[status], travel_length=1.0,
        linearity_spec=0.05, sigma_gradient=0.001, sigma_pass=True,
        linearity_error=0.01, linearity_pass=True,
        position_data=positions, error_data=errors,
        linearity_spec_warning=linearity_spec_warning)


def test_the_measurement_guard_records_its_reason_on_the_track():
    """Ruling 3b: the withdrawal reason must land on linearity_spec_warning -- the same
    field error_reason_of() already reads -- so this mechanism needs no field of its own."""
    from laser_trim_analyzer.core.processor import enforce_measurement_backed_verdict, error_reason_of

    t = _measured_track("FAIL", None, None)
    reason = enforce_measurement_backed_verdict(t)
    assert reason
    assert t.linearity_spec_warning == reason
    assert error_reason_of([t], AnalysisStatus.ERROR) == f"TRK1: {reason}"


def test_the_measurement_guard_never_overwrites_an_existing_warning():
    from laser_trim_analyzer.core.processor import enforce_measurement_backed_verdict

    t = _measured_track("FAIL", None, None, linearity_spec_warning="already flagged upstream")
    enforce_measurement_backed_verdict(t)
    assert t.linearity_spec_warning == "already flagged upstream"


def test_the_measurement_guard_leaves_a_backed_verdict_alone():
    from laser_trim_analyzer.core.processor import enforce_measurement_backed_verdict

    t = _measured_track("PASS", [0.0, 1.0], [0.1, 0.2])
    assert enforce_measurement_backed_verdict(t) is None
    assert t.linearity_spec_warning is None


# ---------------------------------------------------------------------------
# Storage: insert, re-process, and the linked processed_files row
# ---------------------------------------------------------------------------

def _unreadable_error_result(tmp_path, name="8000_1_TEST DATA_1-2-2026_9-00 AM.xls"):
    """An ERROR AnalysisResult for a file that exists on disk (so
    _record_processed_file's stat succeeds) but whose bytes are unreadable.

    process_file's own parse-failure route is what mechanism 1 (3 of the 237
    rows) actually goes through in production, but the exact exception a
    garbage byte string raises is a pandas/openpyxl implementation detail --
    it came back "non_trim" (silently skipped, no AnalysisResult at all) in
    this environment rather than reaching the parser's except-block. Per the
    brief's own escape hatch: build the ERROR result directly with
    _create_error_result on a real temp path instead. The three assertions
    each caller makes are the contract, not the route to an ERROR result.
    """
    from laser_trim_analyzer.core.processor import Processor
    src = tmp_path / name
    src.write_bytes(b"not a workbook")                   # invented name; unreadable on purpose
    proc = Processor(use_ml=False)
    return proc, proc._create_error_result(
        proc._create_minimal_metadata(src), "No valid track data found", 0.0)


def test_the_reason_is_stored_and_linked(db, tmp_path):
    """On insert, on re-process, and on the processed_files row the analysis links to."""
    from sqlalchemy import text
    _, result = _unreadable_error_result(tmp_path)
    assert result is not None and result.overall_status == AnalysisStatus.ERROR
    aid = db.save_analysis(result)
    with db.session() as s:
        reason = s.execute(text("SELECT error_reason FROM analysis_results WHERE id=:i"), {"i": aid}).scalar()
        linked = s.execute(text("SELECT error_message FROM processed_files WHERE analysis_id=:i"), {"i": aid}).scalar()
    assert reason and linked == reason


def test_the_reason_survives_reprocessing(db, tmp_path):
    """Save an ERROR result, then re-process the SAME file identity with a DIFFERENT
    reason (a parser upgrade changed what it found wrong): the second save goes through
    _update_existing_analysis (same filename/file_date/model/serial -> the DB's UNIQUE
    key). Deliberately a different reason each time, not the same object saved twice --
    saving an unchanged object would pass even if the UPDATE path never touched the
    column, because the row's first write already happened to hold the right text."""
    from sqlalchemy import text
    proc, result = _unreadable_error_result(tmp_path)
    assert result is not None and result.overall_status == AnalysisStatus.ERROR

    first_id = db.save_analysis(result)
    result2 = proc._create_error_result(result.metadata, "a different reason found on re-read", 0.0)
    second_id = db.save_analysis(result2)   # same (filename, file_date, model, serial)
    assert second_id == first_id, "same (filename, file_date, model, serial) must UPDATE, not insert"

    with db.session() as s:
        reason = s.execute(text("SELECT error_reason FROM analysis_results WHERE id=:i"),
                            {"i": second_id}).scalar()
    assert reason == "a different reason found on re-read"


def test_a_track_level_error_never_gets_a_retry_marker(db, tmp_path):
    """The failure-marker behaviour (which files are retried) must NOT change: a
    track-level ERROR (bad limits, too few points) has never populated the file-level
    `errors` list, so _write_failure_marker still sees nothing and still writes no
    marker -- even though the LINKED row now carries error_message for display.
    Reproduces the 234-row shape directly (bypasses the parser -- no real file needed
    to prove the manager-side wiring)."""
    from sqlalchemy import text
    from laser_trim_analyzer.core.models import AnalysisResult, FileMetadata, SystemType, TrackData
    from datetime import datetime

    meta = FileMetadata(filename="8856_1_TEST DATA_1-2-2026_9-00 AM.xls",
                         file_path=tmp_path / "8856_1_TEST DATA_1-2-2026_9-00 AM.xls",
                         file_date=datetime(2026, 1, 2), model="8856", serial="1",
                         system=SystemType.A)
    (tmp_path / "8856_1_TEST DATA_1-2-2026_9-00 AM.xls").write_bytes(b"x")  # _record_processed_file stats it
    track = TrackData(track_id="TRK1", status=AnalysisStatus.ERROR, travel_length=1.0,
                       linearity_spec=0.01, anomaly_reason="Insufficient data points")
    from laser_trim_analyzer.core.processor import error_reason_of
    result = AnalysisResult(metadata=meta, overall_status=AnalysisStatus.ERROR,
                             processing_time=0.01, tracks=[track],
                             error_reason=error_reason_of([track], AnalysisStatus.ERROR))
    assert result.errors == []  # the file-level list a track-level ERROR has always left empty

    aid = db.save_analysis(result)
    with db.session() as s:
        rows = s.execute(text(
            "SELECT error_message, analysis_id, success, file_hash FROM processed_files"
        )).fetchall()
    assert len(rows) == 1, f"a marker row must NOT appear for a track-level ERROR: {rows}"
    linked = rows[0]
    # raw sqlite3 (not the ORM) hands back SQLite's stored 0/1, not a Python
    # bool -- `0 is False` is False, so compare by truthiness, not identity.
    assert linked.analysis_id == aid and not linked.success
    assert not linked.file_hash.startswith("skip:")
    assert linked.error_message == "TRK1: Insufficient data points"


# ---------------------------------------------------------------------------
# Model page: _load_units / _search_units COALESCE onto the track (ruling 3c)
# ---------------------------------------------------------------------------

def test_load_units_explains_a_pre_column_row_via_coalesce(make_app):
    """A row saved before error_reason existed (NULL on analysis_results) still explains
    itself: COALESCE falls through to the track's own anomaly_reason."""
    from datetime import datetime
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType, SystemType)

    app = make_app()
    with app.db.session() as s:
        ar = DBAR(model="8000", serial="1", system=SystemType.A,
                  filename="8000_1_TEST DATA_1-2-2026_9-00 AM.xls",
                  file_date=datetime(2026, 1, 2), overall_status=StatusType.ERROR)
        s.add(ar)
        s.flush()
        s.add(DBTR(analysis_id=ar.id, track_id="TRK1", status=StatusType.ERROR,
                   travel_length=1.0, linearity_spec=0.01,
                   anomaly_reason="Insufficient data points"))

    page = app.page_container.get_page("model")
    # _load_units' window anchors to the MODEL's own latest file_date when
    # _current_model is set (see _window_cutoff) -- set directly rather than
    # through full page routing, which this test doesn't otherwise need.
    page._current_model = "8000"
    units = page._load_units("8000")
    assert len(units) == 1
    assert units[0]["error_reason"] == "Insufficient data points"

    found = page._search_units("8000", "1")
    assert len(found) == 1 and found[0]["error_reason"] == "Insufficient data points"


def test_load_units_prefers_error_reason_over_the_track_words(make_app):
    """Once error_reason is set (new rows, from here on) it wins the COALESCE -- it is
    the file's own aggregate, not just one track's."""
    from datetime import datetime
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType, SystemType)

    app = make_app()
    with app.db.session() as s:
        ar = DBAR(model="8001", serial="1", system=SystemType.A,
                  filename="8001_1_TEST DATA_1-2-2026_9-00 AM.xls",
                  file_date=datetime(2026, 1, 2), overall_status=StatusType.ERROR,
                  error_reason="TRK1: Insufficient data points")
        s.add(ar)
        s.flush()
        s.add(DBTR(analysis_id=ar.id, track_id="TRK1", status=StatusType.ERROR,
                   travel_length=1.0, linearity_spec=0.01,
                   anomaly_reason="Insufficient data points"))

    page = app.page_container.get_page("model")
    page._current_model = "8001"
    units = page._load_units("8001")
    assert units[0]["error_reason"] == "TRK1: Insufficient data points"


# ---------------------------------------------------------------------------
# units_tab.py: the render (ruling 3d)
# ---------------------------------------------------------------------------

def _all_labels(widget):
    """Every label in a widget tree."""
    out = []
    for child in widget.winfo_children():
        if hasattr(child, "cget"):
            try:
                child.cget("text")
                out.append(child)
            except Exception:
                pass
        out.extend(_all_labels(child))
    return out


def _cell_texts(row):
    """{column key: the text its cell shows} for one `_UnitRow`, by column POSITION --
    the row packs exactly one label per `_COLUMNS` entry, in that order."""
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.widgets.units_tab import _COLUMNS
    labels = [w for w in row.winfo_children() if isinstance(w, ctk.CTkLabel)]
    assert len(labels) == len(_COLUMNS), [w.cget("text") for w in labels]
    return {key: w.cget("text") for (key, _), w in zip(_COLUMNS, labels)}


def test_error_reason_cell_text_is_shortened_to_60_chars():
    from laser_trim_analyzer.gui.v6.widgets.units_tab import error_reason_cell_text

    assert error_reason_cell_text("Insufficient data points") == "not graded: Insufficient data points"

    long_reason = ("limit column is not a consistent band (median half-width "
                   "63.03 vs modal 0.03, 2101x apart)")
    shortened = error_reason_cell_text(long_reason)
    assert len(shortened) == 60
    assert shortened.endswith("…")
    assert shortened.startswith("not graded: limit column")


def test_units_tab_shows_the_reason_for_an_error_row(tk_root):
    from datetime import datetime
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab

    tab = UnitsTab(tk_root, theme=ThemeManager(), on_unit_click=lambda u: None, on_export=lambda: None)
    # The linearity cell is read by POSITION (the last column), so the dash the sigma
    # cell of a failed track now shows (2026-09-24) cannot stand in for it.
    units = [{"analysis_id": 1, "serial": "sn1", "file_date": datetime.now(),
              "overall_status": "Error", "sigma_gradient": 0.015, "linearity_error": None,
              "error_reason": "Insufficient data points", "track_status": "ERROR"}]
    tab.set_units(units)
    texts = _cell_texts(tab._rows[0])
    assert texts["linearity_error"] == "not graded: Insufficient data points", texts


def test_a_non_error_row_never_shows_not_graded(tk_root):
    from datetime import datetime
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab

    tab = UnitsTab(tk_root, theme=ThemeManager(), on_unit_click=lambda u: None, on_export=lambda: None)
    units = [{"analysis_id": 1, "serial": "sn1", "file_date": datetime.now(),
              "overall_status": "Pass", "sigma_gradient": 0.01, "linearity_error": 0.004,
              "error_reason": None, "track_status": "PASS"}]
    tab.set_units(units)
    texts = [w.cget("text") for w in _all_labels(tab._rows[0])]
    assert not any("not graded" in t for t in texts), texts
    assert "0.004" in texts


def test_an_error_row_with_no_reason_still_shows_the_dash(tk_root):
    """The 3 mechanism-1 rows, until reprocessed (ruling 3c): no reason anywhere yet, so
    the cell must fall back to the ordinary empty-value rendering, not crash or blank."""
    from datetime import datetime
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab

    tab = UnitsTab(tk_root, theme=ThemeManager(), on_unit_click=lambda u: None, on_export=lambda: None)
    units = [{"analysis_id": 1, "serial": "sn1", "file_date": datetime.now(),
              "overall_status": "Error", "sigma_gradient": None, "linearity_error": None,
              "error_reason": None, "track_status": "ERROR"}]
    tab.set_units(units)
    texts = _cell_texts(tab._rows[0])
    assert texts["linearity_error"] == "—", texts


# ---------------------------------------------------------------------------
# One row per TRACK: "not graded" and the sigma dash are the TRACK's (2026-09-24)
# ---------------------------------------------------------------------------

def _two_track_error_analysis(app, model, failed_status):
    """An ERROR analysis whose TRK1 was graded (WARNING) and whose TRK2 failed
    processing -- the shape of model 8530 on the work database today -- with the
    file's own error_reason set, as every ERROR saved from 2026-09-23 on has it.
    Invented values throughout."""
    from datetime import datetime
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType, SystemType)
    with app.db.session() as s:
        ar = DBAR(model=model, serial="21", system=SystemType.A,
                  filename=f"{model}_21_TEST DATA_3-4-2026_9-00 AM.xls",
                  file_date=datetime(2026, 3, 4), overall_status=StatusType.ERROR,
                  error_reason="TRK2: Insufficient data points")
        s.add(ar)
        s.flush()
        s.add(DBTR(analysis_id=ar.id, track_id="TRK1", status=StatusType.WARNING,
                   travel_length=1.0, linearity_spec=0.01, sigma_gradient=0.0123,
                   final_linearity_error_shifted=0.00417))
        s.add(DBTR(analysis_id=ar.id, track_id="TRK2", status=StatusType[failed_status],
                   travel_length=1.0, linearity_spec=0.01, sigma_gradient=999.999,
                   final_linearity_error_shifted=999.999,
                   anomaly_reason="Insufficient data points"))


@pytest.mark.parametrize("failed_status", ["ERROR", "PROCESSING_FAILED"])
def test_a_graded_track_in_an_error_analysis_keeps_its_own_numbers(make_app, failed_status):
    """Through the real loader and the real tab: TRK1 (graded) shows its own sigma and
    linearity error, never the file's "not graded: TRK2: ..."; TRK2 (failed processing)
    says why it was not graded and shows "—" for sigma -- never the 999.999 marker,
    which the cell's "{:.4g}" printed as "1000"."""
    app = make_app()
    _two_track_error_analysis(app, "9991", failed_status)
    page = app.page_container.get_page("model")
    page._current_model = "9991"
    for units in (page._load_units("9991"), page._search_units("9991", "21")):
        assert sorted(u["track_status"] for u in units) == sorted(["WARNING", failed_status])
        page._units_tab.set_units(units)
        app.update_idletasks()
        rows = {r.unit["track_status"]: r for r in page._units_tab._rows}
        assert all(r.winfo_manager() == "pack" for r in rows.values())

        graded = _cell_texts(rows["WARNING"])
        assert graded["linearity_error"] == "0.00417", graded
        assert graded["sigma_gradient"] == "0.0123", graded
        assert not any("not graded" in t for t in graded.values()), graded

        failed = _cell_texts(rows[failed_status])
        assert failed["linearity_error"] == "not graded: TRK2: Insufficient data points", failed
        assert failed["sigma_gradient"] == "—", failed
        assert not any(m in t for t in failed.values() for m in ("1000", "1e+03", "999")), failed


def test_a_failed_track_with_no_reason_shows_dashes_never_its_leftover_numbers(make_app):
    """No reason anywhere (the 3 mechanism-1 rows, until reprocessed): both measurement
    cells fall back to "—" -- a failed record's leftovers are not readings, even when
    the analyser left a number in the column (114 ERROR tracks on the work database
    carry a stored linearity error, some of them 999.999)."""
    from datetime import datetime
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, StatusType, SystemType)
    app = make_app()
    with app.db.session() as s:
        ar = DBAR(model="9992", serial="5", system=SystemType.B,
                  filename="9992_5_TA_Test Data_3-4-2026_9-00 AM.xls",
                  file_date=datetime(2026, 3, 4), overall_status=StatusType.ERROR)
        s.add(ar)
        s.flush()
        s.add(DBTR(analysis_id=ar.id, track_id="Track A", status=StatusType.ERROR,
                   travel_length=1.0, linearity_spec=0.01, sigma_gradient=999.999,
                   final_linearity_error_shifted=0.0321))
    page = app.page_container.get_page("model")
    page._current_model = "9992"
    units = page._load_units("9992")
    assert [u["error_reason"] for u in units] == [None]
    page._units_tab.set_units(units)
    row = page._units_tab._rows[0]
    assert row.winfo_manager() == "pack"
    cells = _cell_texts(row)
    assert (cells["sigma_gradient"], cells["linearity_error"]) == ("—", "—"), cells
