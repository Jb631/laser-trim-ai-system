"""The status bar at the foot of every page (option B, 2026-10-04).

James, with TMOG open: "this is an example of a finished peice of solftware". The bar says, left
to right: whether the database could be read, how many models are on file, the newest file, whether
the drift watch is current, how many files are being skipped -- and, on the right, the run in
flight with its live progress.

Its data comes from ONE loader, `gui/v6/status_data.load_status(db)`, on a worker; each part that
fails is NAMED, never shown as nothing or zero. The run line reuses the progress the Process page's
two runs already receive, reported app-wide through V6App's run observer.

Every database here is a tmp file; every model, serial and date is INVENTED.
"""
import threading
import time
from datetime import datetime, timedelta
from threading import Event

from laser_trim_analyzer.gui.v6 import status_data as sd

NOW = datetime(2026, 9, 30, 12, 0)


# ---- an invented database ------------------------------------------------------------------------

def _db(tmp_path, name="status.db"):
    from laser_trim_analyzer.database.manager import DatabaseManager
    return DatabaseManager(tmp_path / name)


_serial = iter(range(1, 10 ** 7))


def _trim(db, model, when, status="PASS", quality="good"):
    from laser_trim_analyzer.database.models import AnalysisResult, StatusType, SystemType
    n = next(_serial)
    with db.session() as s:
        s.add(AnalysisResult(model=model, serial=f"{model}-{n}", system=SystemType.A,
                             filename=f"{model}_{n}.xls", file_date=when,
                             overall_status=StatusType[status], data_quality=quality))


def _smoothness(db, model, when):
    from laser_trim_analyzer.database.models import SmoothnessResult, StatusType
    n = next(_serial)
    with db.session() as s:
        s.add(SmoothnessResult(model=model, serial=f"{model}-os-{n}",
                               filename=f"OS_{model}_{n}.xls", file_date=when,
                               overall_status=StatusType.PASS))


def _skip_marker(db, name, *, failed_read):
    db.mark_file_skipped(name, f"/invented/share/{name}", f"hash-{name}", 100, NOW,
                         error_message="invented reason", failed_read=failed_read)


# ---- the loader ----------------------------------------------------------------------------------

def test_the_model_count_is_every_model_the_models_picker_can_open(tmp_path):
    """A model with a laser file or a smoothness file -- the Models picker's own list
    (ml.manager.list_known_models) -- each once, whatever its files say."""
    from laser_trim_analyzer.ml.manager import list_known_models
    db = _db(tmp_path)
    _trim(db, "M1", NOW - timedelta(days=3))
    _trim(db, "M1", NOW - timedelta(days=2))
    _trim(db, "M2", NOW - timedelta(days=9), status="ERROR")
    _smoothness(db, "S1", NOW - timedelta(days=4))
    _smoothness(db, "M1", NOW - timedelta(days=4))
    status = sd.load_status(db, now=NOW)
    assert status.failed == {}
    assert status.models == 3
    assert status.models == len(list_known_models(db))


def test_the_newest_file_is_the_newest_trim_that_counts(tmp_path):
    """Graded, believed (not dated more than a day ahead of now) and not marked suspect -- the
    file the Overview's windows end on, so the bar and the Overview never name two dates."""
    db = _db(tmp_path)
    counts = NOW - timedelta(days=2)
    _trim(db, "M1", NOW - timedelta(days=20))
    _trim(db, "M1", counts, status="FAIL")
    _trim(db, "M1", NOW - timedelta(days=1), status="ERROR")            # failed processing
    _trim(db, "M1", NOW - timedelta(hours=20), status="UNTRIMMED")      # no cut
    _trim(db, "M1", NOW - timedelta(hours=10), status="FAIL", quality="suspect")
    _trim(db, "M1", NOW + timedelta(days=3))                            # a mistyped future date
    assert sd.load_status(db, now=NOW).newest == counts


def test_the_bar_and_the_overview_name_the_same_newest_file(tmp_path, monkeypatch):
    from laser_trim_analyzer.gui.v6 import overview_data as od
    from laser_trim_analyzer.ml.spc import FocusResult
    monkeypatch.setattr(od, "load_focus", lambda db, *a, **k: (
        FocusResult(focus=[], chronic=[], anchor=None), None))
    monkeypatch.setattr(od, "get_drifting_models", lambda db, *a, **k: [])
    db = _db(tmp_path)
    _trim(db, "M1", NOW - timedelta(days=5), status="WARNING")
    _trim(db, "M1", NOW - timedelta(days=1), status="ERROR")
    _trim(db, "M2", NOW - timedelta(hours=6), quality="suspect")
    _trim(db, "M2", NOW + timedelta(days=2))
    assert od.load_overview(db, now=NOW).anchor == sd.load_status(db, now=NOW).newest \
        == NOW - timedelta(days=5)


def test_with_no_trim_file_there_is_no_newest_file_and_that_is_not_a_failure(tmp_path):
    db = _db(tmp_path)
    _smoothness(db, "S1", NOW)
    status = sd.load_status(db, now=NOW)
    assert status.failed == {} and status.newest is None and status.models == 1


def test_skipped_counts_the_files_that_failed_to_read_and_nothing_else(tmp_path):
    """The files the ingest refuses to re-offer because they failed to read before -- the
    Overview's own notice (Settings → Retry unreadable files). A file skipped as not-test-data is
    not one of them."""
    db = _db(tmp_path)
    _skip_marker(db, "a.xls", failed_read=True)
    _skip_marker(db, "b.xls", failed_read=True)
    _skip_marker(db, "notes.xls", failed_read=False)
    assert sd.load_status(db, now=NOW).skipped == 2


class _Unreadable:
    """A database that cannot be read at all."""

    def session(self):
        from sqlalchemy.exc import OperationalError
        raise OperationalError("SELECT 1", {}, Exception("unable to open database file"))

    def count_failed_file_markers(self):
        raise AssertionError("not reached: the database could not be read")


def test_a_database_that_cannot_be_read_is_named_and_nothing_is_counted():
    status = sd.load_status(_Unreadable(), now=NOW)          # never raises
    assert status.failed == {sd.PART_DATABASE: "OperationalError"}
    assert not status.database_ok
    assert (status.models, status.newest, status.skipped) == (None, None, None)


def test_one_part_that_fails_is_named_and_the_others_still_load(tmp_path, monkeypatch):
    db = _db(tmp_path)
    _trim(db, "M1", NOW - timedelta(days=2))

    def locked():
        raise RuntimeError("database is locked")
    monkeypatch.setattr(db, "count_failed_file_markers", locked)
    status = sd.load_status(db, now=NOW)
    assert status.failed == {sd.PART_SKIPPED: "RuntimeError"}
    assert status.database_ok and status.models == 1 and status.newest == NOW - timedelta(days=2)
    assert status.skipped is None                            # unknown -- never 0


# ---- the words the bar prints (pure) -------------------------------------------------------------

def test_the_words_for_a_healthy_database():
    s = sd.Status(models=333, newest=datetime(2026, 9, 29, 17, 22), skipped=70)
    assert sd.database_words(s) == ("Database OK", "ok")
    assert sd.models_words(s) == ("333 models", "quiet")
    assert sd.newest_words(s) == ("Newest file 29 Sep 2026", "quiet")
    assert sd.skipped_words(s) == ("70 files skipped", "check")
    one = sd.Status(models=1, newest=None, skipped=1)
    assert sd.models_words(one) == ("1 model", "quiet")
    assert sd.newest_words(one) == ("No trim files yet", "quiet")
    assert sd.skipped_words(one) == ("1 file skipped", "check")
    assert sd.models_words(sd.Status(models=1234, skipped=0))[0] == "1,234 models"


def test_nothing_skipped_says_nothing():
    assert sd.skipped_words(sd.Status(models=3, skipped=0)) == ("", "quiet")


def test_before_the_first_load_the_bar_says_it_is_looking():
    assert sd.database_words(None) == ("Checking the database…", "quiet")
    for words in (sd.models_words, sd.newest_words, sd.skipped_words):
        assert words(None) == ("", "quiet")


def test_each_failed_part_is_named_in_the_check_colour_never_a_zero():
    dead = sd.Status(failed={sd.PART_DATABASE: "OperationalError"})
    assert sd.database_words(dead) == ("Database: could not read (OperationalError)", "check")
    # The database could not be read: the bar says that, once -- not three more times.
    for words in (sd.models_words, sd.newest_words, sd.skipped_words):
        assert words(dead) == ("", "quiet")
    part = sd.Status(newest=NOW, failed={sd.PART_MODELS: "OperationalError",
                                         sd.PART_SKIPPED: "RuntimeError"})
    assert sd.database_words(part) == ("Database OK", "ok")
    assert sd.models_words(part) == ("Models: could not count (OperationalError)", "check")
    assert sd.skipped_words(part) == ("Skipped files: could not count (RuntimeError)", "check")
    lost = sd.Status(models=2, skipped=0, failed={sd.PART_NEWEST: "DatabaseError"})
    assert sd.newest_words(lost) == ("Newest file: could not read (DatabaseError)", "check")


def test_the_drift_words():
    assert sd.drift_words(sd.DRIFT_CURRENT) == ("Drift watch current", "quiet")
    assert sd.drift_words(sd.DRIFT_UPDATING) == ("Drift watch updating…", "quiet")
    assert sd.drift_words(sd.DRIFT_FAILED, "OperationalError") == \
        ("Drift watch: could not update (OperationalError)", "check")


def test_the_run_words():
    assert sd.run_words(None) == ""
    run = sd.RunState(name="An ingest", label="Processing new files", done=34, total=120)
    assert sd.run_words(run) == "Processing new files · 34 of 120"
    big = sd.RunState(name="An ingest", label="Processing new files", done=1214, total=8900)
    assert sd.run_words(big) == "Processing new files · 1,214 of 8,900"
    # Before the pre-scan has a total: no "0 of 0" -- that reads as a stuck run.
    early = sd.RunState(name="An ingest", label="Processing a folder", done=0, total=0)
    assert sd.run_words(early) == "Processing a folder…"
    # A run that reports no progress of its own (a re-grade, a findings refresh) is named.
    assert sd.run_words(sd.RunState(name="A re-grade")) == "A re-grade is running"


# ---- the bar in the window -----------------------------------------------------------------------

def _walk(widget):
    import tkinter
    yield widget
    for child in tkinter.Misc.winfo_children(widget):
        yield from _walk(child)


def _pump(app, done, seconds=5.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not done():
        app.update()
        time.sleep(0.01)
    return done()


def _shown(label) -> bool:
    return label.winfo_manager() != ""


def test_the_bar_sits_at_the_foot_of_the_window_under_every_page(make_app):
    from laser_trim_analyzer.gui.v6.status_bar import HEIGHT, StatusBar
    app = make_app()
    bar = app.status_bar
    assert isinstance(bar, StatusBar)
    info = bar.grid_info()
    assert int(info["row"]) == int(app.page_container.grid_info()["row"]) + 1
    assert info["sticky"] == "ew" and int(app.grid_rowconfigure(int(info["row"]))["weight"]) == 0
    assert bar.cget("fg_color") == app.theme.SURFACE
    assert 24 <= HEIGHT <= 32 and bar.cget("height") == HEIGHT
    # A thin DIVIDER line along its top edge.
    first = bar.pack_slaves()[0]
    assert first.cget("fg_color") == app.theme.DIVIDER and first.cget("height") == 1
    t = app.theme
    for label in (bar._database, bar._models, bar._newest, bar._drift, bar._skipped, bar._run):
        assert label.cget("font") is t.font(t.SIZE_CAPTION)


def test_the_bar_says_what_the_database_holds(make_app):
    app = make_app()
    _trim(app.db, "M1", datetime(2026, 9, 29, 17, 22))
    _trim(app.db, "M2", datetime(2026, 9, 1, 8, 0), status="FAIL")
    _skip_marker(app.db, "a.xls", failed_read=True)
    bar = app.status_bar
    bar.refresh_now()
    t = app.theme
    assert bar._database.cget("text") == "Database OK" and bar._dot.cget("fg_color") == t.PASS_FG
    assert bar._models.cget("text") == "2 models"
    assert bar._newest.cget("text") == "Newest file 29 Sep 2026"
    assert bar._drift.cget("text") == "Drift watch current"
    assert bar._skipped.cget("text") == "1 file skipped"
    assert bar._skipped.cget("text_color") == t.CHECK
    words = (bar._database, bar._models, bar._newest, bar._drift, bar._skipped)
    assert all(_shown(w) for w in words)
    for label in (bar._database, bar._models, bar._newest, bar._drift):
        assert label.cget("text_color") == t.TEXT_SECONDARY
    # Left to right, in the order James reads them.
    order = [w for w in bar._left.pack_slaves() if w is not bar._dot]
    assert order == [bar._database, bar._models, bar._newest, bar._drift, bar._skipped]


def test_nothing_skipped_leaves_no_gap(make_app):
    app = make_app()
    _trim(app.db, "M1", NOW)
    app.status_bar.refresh_now()
    assert not _shown(app.status_bar._skipped)


def test_a_database_that_cannot_be_read_is_named_in_the_check_colour(make_app):
    app = make_app()
    bar = app.status_bar
    bar.show_status(sd.Status(failed={sd.PART_DATABASE: "OperationalError"}))
    t = app.theme
    assert bar._database.cget("text") == "Database: could not read (OperationalError)"
    assert bar._database.cget("text_color") == t.CHECK and bar._dot.cget("fg_color") == t.CHECK
    assert not any(_shown(w) for w in (bar._models, bar._newest, bar._skipped))
    assert _shown(bar._drift)                     # the drift watch is not the database's to say


def test_a_part_that_failed_is_named_where_its_number_would_be(make_app):
    app = make_app()
    bar = app.status_bar
    bar.show_status(sd.Status(newest=NOW, skipped=0, failed={sd.PART_MODELS: "OperationalError"}))
    assert bar._models.cget("text") == "Models: could not count (OperationalError)"
    assert bar._models.cget("text_color") == app.theme.CHECK and _shown(bar._models)
    assert bar._database.cget("text") == "Database OK"


def test_the_load_runs_on_a_worker_and_lands_on_the_bar(make_app):
    app = make_app()
    _trim(app.db, "M1", NOW)
    bar = app.status_bar
    bar.refresh()
    assert _pump(app, lambda: bar._models.cget("text") == "1 model")


class _Gate:
    """A loader that waits until it is let go, counting its calls -- for the refresh triggers."""

    def __init__(self):
        self.calls = 0
        self.go = Event()

    def __call__(self, db, now=None):
        self.calls += 1
        self.go.wait(5)
        return sd.Status(models=7, skipped=0)


def _gated(app, monkeypatch):
    gate = _Gate()
    _pump(app, lambda: not app.status_bar._loading)          # the start-up load lands first
    monkeypatch.setattr(app.status_bar, "_loader", gate)
    return gate


def test_refreshes_asked_for_during_a_load_become_one_more_load(make_app, monkeypatch):
    app = make_app()
    gate = _gated(app, monkeypatch)
    bar = app.status_bar
    for _ in range(4):
        bar.refresh()
    assert gate.calls <= 1
    gate.go.set()
    assert _pump(app, lambda: gate.calls == 2 and not bar._loading)
    _pump(app, lambda: False, seconds=0.3)
    assert gate.calls == 2


def test_the_bar_reloads_when_the_overview_is_shown(make_app, monkeypatch):
    app = make_app()
    gate = _gated(app, monkeypatch)
    gate.go.set()
    app.show_page("settings")
    assert gate.calls == 0
    app.show_page("home")
    assert _pump(app, lambda: gate.calls == 1)


# ---- the run line --------------------------------------------------------------------------------

class _Alive:
    """A run's thread: alive until stopped."""

    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive


def test_the_run_line_shows_the_run_in_flight_and_goes_when_it_ends(make_app, monkeypatch):
    app = make_app()
    gate = _gated(app, monkeypatch)
    gate.go.set()
    bar = app.status_bar
    assert not _shown(bar._run)                   # no run: nothing on the right
    cancel, thread = Event(), _Alive()
    app.register_ingest(cancel, thread, "An ingest")
    app.report_run_progress(cancel, "Processing new files", 34, 120)
    assert bar._run.cget("text") == "Processing new files · 34 of 120" and _shown(bar._run)
    app.report_run_progress(cancel, "Processing new files", 35, 120)
    assert bar._run.cget("text") == "Processing new files · 35 of 120"
    assert gate.calls == 0
    thread.alive = False
    app.unregister_ingest(cancel)
    assert not _shown(bar._run)
    # ...and a run that ended changed the database: the bar reads it again.
    assert _pump(app, lambda: gate.calls == 1)


def test_a_run_whose_thread_died_without_saying_so_still_leaves_the_bar(make_app, monkeypatch):
    """NewFilesRun's crash path posts no unregister; the bar checks the run is still alive."""
    app = make_app()
    gate = _gated(app, monkeypatch)
    gate.go.set()
    cancel, thread = Event(), _Alive()
    app.register_ingest(cancel, thread, "An ingest")
    app.report_run_progress(cancel, "Processing new files", 3, 9)
    assert _shown(app.status_bar._run)
    thread.alive = False
    assert _pump(app, lambda: not _shown(app.status_bar._run) and gate.calls == 1)


def test_a_run_that_reports_no_progress_is_named(make_app, monkeypatch):
    from laser_trim_analyzer.gui.v6 import status_bar as sb
    app = make_app()
    bar = app.status_bar
    cancel, thread = Event(), _Alive()
    app.register_ingest(cancel, thread, "A re-grade")
    # Not at once: the Process page's first progress lands within a quarter second, and a name
    # flashed in front of it would be a flicker.
    assert not _shown(bar._run)
    later = time.monotonic() + sb.QUIET_START + 0.1
    monkeypatch.setattr(sb, "_clock", lambda: later)
    bar._render_run()
    assert bar._run.cget("text") == "A re-grade is running" and _shown(bar._run)
    thread.alive = False
    app.unregister_ingest(cancel)


def test_a_paint_that_lands_after_its_run_ended_is_dropped(make_app):
    app = make_app()
    cancel, thread = Event(), _Alive()
    app.register_ingest(cancel, thread, "An ingest")
    thread.alive = False
    app.unregister_ingest(cancel)
    app.report_run_progress(cancel, "Processing new files", 120, 120)     # the ticker's last one
    assert app.run_progress() is None
    assert not _shown(app.status_bar._run)
    app.report_run_progress(Event(), "Processing new files", 1, 2)       # never registered
    assert app.run_progress() is None


def test_a_listener_that_breaks_never_breaks_a_run(make_app, caplog):
    import logging
    app = make_app()
    app.add_run_listener(lambda: 1 / 0)
    cancel, thread = Event(), _Alive()
    with caplog.at_level(logging.ERROR):
        app.register_ingest(cancel, thread, "An ingest")
        app.report_run_progress(cancel, "Processing new files", 1, 2)
    assert app.run_progress().done == 1
    assert any(r.exc_info and isinstance(r.exc_info[1], ZeroDivisionError) for r in caplog.records)
    thread.alive = False
    app.unregister_ingest(cancel)


def _paint_once(run, done, total):
    """What the Process page's ticker does four times a second: drain, then paint."""
    from laser_trim_analyzer.core.ingest_run import EtaEstimator, ProgressCoalescer
    from laser_trim_analyzer.core.models import ProcessingStatus
    coalescer = ProgressCoalescer()
    for i in range(done):
        coalescer.note(ProcessingStatus(filename=f"f{i}.xls", status="completed",
                                        progress_percent=0.0))
    state = {"n": total, "folder": 1, "folders": 2}
    run._paint(coalescer, state, EtaEstimator())


def test_the_process_pages_own_progress_reaches_the_bar(make_app):
    """The remembered-folder run: the numbers its own line just got, app-wide -- and the page's
    own line is what it always was."""
    app = make_app()
    run = app.page_container.get_page("process")._new_files
    run._cancel, thread = Event(), _Alive()
    app.register_ingest(run._cancel, thread, "An ingest")
    _paint_once(run, 34, 120)
    progress = app.run_progress()
    assert (progress.name, progress.label, progress.done, progress.total) == \
        ("An ingest", "Processing new files", 34, 120)
    assert app.status_bar._run.cget("text") == "Processing new files · 34 of 120"
    assert "34 done · 86 to go" in run._progress._status.cget("text")
    thread.alive = False
    app.unregister_ingest(run._cancel)


def test_the_specific_folder_run_reaches_the_bar_too(make_app):
    app = make_app()
    page = app.page_container.get_page("process")
    page._cancel, thread = Event(), _Alive()
    app.register_ingest(page._cancel, thread, "An ingest")
    _paint_once(page, 5, 40)
    assert app.status_bar._run.cget("text") == "Processing a folder · 5 of 40"
    thread.alive = False
    app.unregister_ingest(page._cancel)


def test_a_paint_with_no_run_started_changes_nothing(make_app):
    """The Process page's tests paint with no run started (its _cancel is None)."""
    app = make_app()
    _paint_once(app.page_container.get_page("process")._new_files, 1, 2)
    assert app.run_progress() is None and not _shown(app.status_bar._run)


# ---- the drift line ------------------------------------------------------------------------------

def _inline_threads(monkeypatch):
    class _Now:
        def __init__(self, target=None, daemon=None, args=(), kwargs=None):
            self._target = target

        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _Now)


def test_the_drift_line_says_updating_while_the_startup_catch_up_runs(make_app, monkeypatch):
    from laser_trim_analyzer.ml import drift_training
    app = make_app()
    bar = app.status_bar
    assert bar._drift.cget("text") == "Drift watch current"        # nothing is catching up
    seen = []
    monkeypatch.setattr(drift_training, "ensure_drift_rules", lambda db, preset: False)

    def advance(db):
        seen.append(bar._drift.cget("text"))
        return 0
    monkeypatch.setattr(drift_training, "advance_drift_state", advance)
    gate = _gated(app, monkeypatch)
    gate.go.set()
    _inline_threads(monkeypatch)
    app._start_drift_catchup()
    assert seen == ["Drift watch updating…"]
    assert _pump(app, lambda: bar._drift.cget("text") == "Drift watch current")
    assert app.drift_watch() == (sd.DRIFT_CURRENT, None)
    # ...and after the rebuild the bar reads the database again.
    assert _pump(app, lambda: gate.calls == 1)


def test_a_catch_up_that_failed_is_named_never_current(make_app, monkeypatch):
    from laser_trim_analyzer.ml import drift_training
    app = make_app()
    monkeypatch.setattr(drift_training, "ensure_drift_rules", lambda db, preset: False)

    def locked(db):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(drift_training, "advance_drift_state", locked)
    _inline_threads(monkeypatch)
    app._start_drift_catchup()
    bar = app.status_bar
    assert _pump(app, lambda: bar._drift.cget("text") ==
                 "Drift watch: could not update (RuntimeError)")
    assert bar._drift.cget("text_color") == app.theme.CHECK


def test_an_app_that_will_catch_up_says_updating_from_the_start(tmp_path):
    """Production: the catch-up is scheduled five seconds after start, and the flags on screen
    may be behind until it has run -- "current" before then would be a claim not yet checked."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.gui.v6.app import V6App
    cfg = Config()
    cfg.database.path = tmp_path / "v6.db"
    app = V6App(cfg, db=DatabaseManager(cfg.database.path), auto_train_on_first_run=True)
    try:
        app.withdraw()
        assert app.drift_watch() == (sd.DRIFT_UPDATING, None)
        assert app.status_bar._drift.cget("text") == "Drift watch updating…"
    finally:
        app.destroy()
