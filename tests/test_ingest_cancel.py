"""Stop: an ingest in flight finishes the batch it is on and quits cleanly.

James, mid-way through his first full ingest at work: "no way to stop
processing?" There was not. The only exits were killing the window — which
destroyed Tk while a batch was mid-write — and waiting hours.

What "cooperative" has to mean here, and what these tests pin:

  * NOTHING is killed. The 20-file batch already submitted to the thread pool
    runs to completion, its results are persisted exactly as an uncancelled
    batch's are, and only then does the loop stop. A half-written batch is how
    you get rows without their tracks.
  * The stop is honoured at a BATCH BOUNDARY, so "files processed" always
    lands on a multiple of the batch size — a promise a per-file check could
    not make and a thread kill would break outright.
  * The run says it stopped. A cancelled run that reports "3 folders · 214 new
    files" is indistinguishable from a finished one, and the difference is
    hours of unprocessed history.
  * Resuming is just pressing the button again: HOME is always incremental,
    so the second press skips everything the first press saved. The summary
    line says so, because otherwise the safe assumption is "I have to start
    over" and nobody presses Stop.
"""
import time
from pathlib import Path
from threading import Event

import pytest

from laser_trim_analyzer.config import Config
from laser_trim_analyzer.core import ingest_run
from laser_trim_analyzer.core.ingest_run import (
    FolderResult,
    IngestReport,
    format_ingest_summary,
    run_folder,
    run_folders,
)
from laser_trim_analyzer.core.processor import Outcome, Processor

BATCH = 20          # processor.py's batch_size — the boundary under test


# ---- scaffolding -----------------------------------------------------------

class _StubProcessor(Processor):
    """The REAL batch machinery (pool, batching, summary) with a stub parse.

    use_ml=False keeps the constructor off the database — Processor's ML
    threshold load goes through the global manager, which ignores an injected
    handle and would create one at the config default.

    The stub is the pool's unit of work, `analyse_path` (ingest-speed Task 9:
    the pool analyses, the consumer writes); it asks for no write.
    """

    def __init__(self, *a, **k):
        k.setdefault("use_ml", False)
        super().__init__(*a, **k)

    def analyse_path(self, file_path, disk_stat=None):
        return Outcome(path=str(file_path), result=self._create_error_result(
            self._create_minimal_metadata(Path(file_path)), "stub", time.time()))


class _FakeDb:
    def __init__(self):
        self.saved = []

    def save_analysis(self, result):
        self.saved.append(result)


def _paths(n):
    return [Path(f"/nowhere/f{i:05d}.xls") for i in range(n)]


def _make_files(folder, n):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"f{i:05d}.xls").write_bytes(b"")
    return folder


def _cancel_after(cancel, n, seen):
    """A progress callback that trips `cancel` once n files have finished."""
    def cb(status):
        if status.status in ("completed", "skipped", "failed"):
            seen.append(status.filename)
            if len(seen) >= n:
                cancel.set()
    return cb


# ---- the processor: the boundary itself ------------------------------------

def test_cancel_mid_batch_still_finishes_that_batch_and_stops_there():
    """Set during batch 1 of 6; exactly one batch's worth comes out."""
    cancel, seen = Event(), []
    proc = _StubProcessor(config=Config())
    got = list(proc.process_batch(_paths(6 * BATCH),
                                  progress_callback=_cancel_after(cancel, 3, seen),
                                  incremental=False, cancel=cancel))
    assert len(got) == BATCH, "stopped somewhere other than a batch boundary"
    assert len(seen) == BATCH


def test_a_cancel_already_set_processes_nothing_at_all():
    cancel = Event()
    cancel.set()
    proc = _StubProcessor(config=Config())
    assert list(proc.process_batch(_paths(3 * BATCH), incremental=False,
                                   cancel=cancel)) == []


def test_no_cancel_event_processes_everything_as_before():
    """The guard must not change the uncancelled path."""
    proc = _StubProcessor(config=Config())
    assert len(list(proc.process_batch(_paths(45), incremental=False))) == 45


def test_cancelled_batch_still_returns_its_summary():
    """A cancelled generator returns the same BatchSummary a finished one does
    — the page's final tally comes from it."""
    cancel, seen = Event(), []
    proc = _StubProcessor(config=Config())
    gen = proc.process_batch(_paths(5 * BATCH),
                             progress_callback=_cancel_after(cancel, 1, seen),
                             incremental=False, cancel=cancel)
    summary = None
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        summary = stop.value
    assert summary is not None
    assert summary.processed == BATCH
    assert summary.errors == BATCH


# ---- one folder ------------------------------------------------------------

def test_run_folder_persists_the_finished_batch_and_says_it_was_cancelled(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_run, "Processor", _StubProcessor)
    # >100 files, or the processor takes its sequential path (turbo_mode_
    # threshold) where one file is the batch and there is no 20-file boundary.
    _make_files(tmp_path / "laser", 6 * BATCH)
    cancel, seen, db = Event(), [], _FakeDb()

    class _Trip:
        """Trip the cancel from the coalescer's own worker-side hook."""
        def note(self, status):
            if status.status in ("completed", "skipped", "failed"):
                seen.append(status.filename)
                if len(seen) >= 2:
                    cancel.set()

        def bucket(self, name, reason=""):
            pass

        # The run sizes itself through the coalescer before the first file
        # (work total, and how much of this folder the database already has).
        # A progress object that cannot take those is not one run_folder can
        # drive, so the stub carries them rather than the code guessing.
        def set_total(self, total):
            pass

        def expect_known(self, n):
            pass

    res = run_folder(str(tmp_path / "laser"), db=db, config=Config(),
                     incremental=False, progress=_Trip(), cancel=cancel)
    assert res.cancelled is True
    assert res.ok is True                      # stopping is not failing
    assert res.error is None
    assert len(db.saved) == BATCH, "a partial batch was persisted"
    assert res.new_files == BATCH


def test_a_small_folder_stops_too_even_though_it_never_batches(tmp_path,
                                                               monkeypatch):
    """Under the turbo threshold the processor runs sequentially; Stop has to
    work there too, or a one-folder re-run is uninterruptible."""
    monkeypatch.setattr(ingest_run, "Processor", _StubProcessor)
    _make_files(tmp_path / "small", 40)
    cancel, seen, db = Event(), [], _FakeDb()

    class _Trip:
        def note(self, status):
            if status.status in ("completed", "skipped", "failed"):
                seen.append(status.filename)
                if len(seen) >= 5:
                    cancel.set()

        def bucket(self, name, reason=""):
            pass

        # The run sizes itself through the coalescer before the first file
        # (work total, and how much of this folder the database already has).
        # A progress object that cannot take those is not one run_folder can
        # drive, so the stub carries them rather than the code guessing.
        def set_total(self, total):
            pass

        def expect_known(self, n):
            pass

    res = run_folder(str(tmp_path / "small"), db=db, config=Config(),
                     incremental=False, progress=_Trip(), cancel=cancel)
    assert res.cancelled is True
    assert 5 <= len(db.saved) < 40


def test_run_folder_without_a_cancel_is_not_cancelled(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_run, "Processor", _StubProcessor)
    _make_files(tmp_path / "laser", 5)
    db = _FakeDb()
    res = run_folder(str(tmp_path / "laser"), db=db, config=Config(),
                     incremental=False)
    assert res.cancelled is False and len(db.saved) == 5


# ---- many folders ----------------------------------------------------------

def _stub_run_folder(monkeypatch, seen, *, cancel_on=None):
    def fake(folder, **kw):
        seen.append(folder)
        ev = kw.get("cancel")
        if folder == cancel_on and ev is not None:
            ev.set()
            return FolderResult(folder=folder, ok=True, files_found=10,
                                new_files=4, cancelled=True, seconds=1.0)
        return FolderResult(folder=folder, ok=True, files_found=10, new_files=4,
                            seconds=1.0)
    monkeypatch.setattr(ingest_run, "run_folder", fake)


def test_run_folders_stops_after_the_folder_it_is_on(monkeypatch):
    seen, cancel = [], Event()
    _stub_run_folder(monkeypatch, seen, cancel_on="/laser/b")
    report = run_folders(["/laser/a", "/laser/b", "/final_test"],
                         db=None, config=None, cancel=cancel)
    assert seen == ["/laser/a", "/laser/b"], "a third folder was started"
    assert report.cancelled is True
    assert report.folder_count == 2 and report.folders_requested == 3
    assert report.new_files == 8              # both folders' work is kept


def test_run_folders_threads_the_event_to_every_folder(monkeypatch):
    got, cancel = [], Event()
    def fake(folder, **kw):
        got.append(kw.get("cancel"))
        return FolderResult(folder=folder, ok=True)
    monkeypatch.setattr(ingest_run, "run_folder", fake)
    run_folders(["/a", "/b"], db=None, config=None, cancel=cancel)
    assert got == [cancel, cancel]


def test_run_folders_not_cancelled_reports_every_folder(monkeypatch):
    seen, cancel = [], Event()
    _stub_run_folder(monkeypatch, seen)
    report = run_folders(["/a", "/b"], db=None, config=None, cancel=cancel)
    assert seen == ["/a", "/b"] and report.cancelled is False


def test_a_cancel_set_before_the_run_starts_no_folder(monkeypatch):
    seen, cancel = [], Event()
    cancel.set()
    _stub_run_folder(monkeypatch, seen)
    report = run_folders(["/a", "/b"], db=None, config=None, cancel=cancel)
    assert seen == [] and report.cancelled is True


# ---- the summary line ------------------------------------------------------

def _cancelled(new, planned, folders_done, folders_total, seconds):
    return IngestReport(
        results=[FolderResult(folder=f"/f{i}", ok=True, new_files=n)
                 for i, n in enumerate(_split(new, folders_done))],
        seconds=seconds, cancelled=True, files_planned=planned,
        folders_requested=folders_total)


def _split(total, parts):
    each = [total // parts] * parts
    each[-1] += total - sum(each)
    return each


def test_cancelled_summary_says_how_far_it_got_and_how_to_continue():
    line = format_ingest_summary(_cancelled(1214, 8900, 2, 3, 724.0))
    assert line.startswith("Stopped after 1,214 of 8,900 new files "
                           "(2 of 3 folders) · 12 min 4 s")
    # Resuming must be spelled out: without it the safe reading is "I have to
    # start the whole thing again", and nobody ever presses Stop.
    assert "again" in line and "Process everything new" in line
    assert "skip" in line.lower() or "resume" in line.lower()


def test_cancelled_summary_omits_a_total_it_does_not_know():
    """files_planned is 0 for a run that never scanned ahead — say the count
    that is real, invent nothing."""
    line = format_ingest_summary(
        IngestReport(results=[FolderResult(folder="/a", ok=True, new_files=7)],
                     seconds=30.0, cancelled=True, folders_requested=2))
    assert line.startswith("Stopped after 7 new files (1 of 2 folders) · 30 s")
    assert " of 0 " not in line


def test_cancelled_summary_still_names_a_folder_that_failed():
    report = IngestReport(
        results=[FolderResult(folder="/a", ok=True, new_files=4),
                 FolderResult(folder="\\\\nas\\Laser", ok=False,
                              error="not found — offline share?")],
        seconds=61.0, cancelled=True, folders_requested=3, files_planned=99)
    line = format_ingest_summary(report)
    assert "Stopped after" in line
    assert "\\\\nas\\Laser" in line and "offline share" in line


def test_an_uncancelled_summary_is_word_for_word_what_it_was():
    """The Stop work must not have touched the normal line."""
    report = IngestReport(
        results=[FolderResult(folder=f"/f{i}", ok=True, files_found=100,
                              new_files=n, seconds=1.0)
                 for i, n in enumerate((100, 100, 14))],
        seconds=160.0, folders_requested=3, files_planned=214)
    assert format_ingest_summary(report) == \
        "3 folders · 214 new files · 2 min 40 s"


# ---- HOME ------------------------------------------------------------------

def _home(app):
    return app.page_container.get_page("home")


class _NoThread:
    started = []

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.target, self.args = target, args or ()

    def start(self):
        _NoThread.started.append((self.target, self.args))

    def is_alive(self):
        return True

    def join(self, timeout=None):
        pass


def _no_threads(monkeypatch, module):
    import types
    _NoThread.started = []
    monkeypatch.setattr(module, "threading",
                        types.SimpleNamespace(Thread=_NoThread))
    return _NoThread


def _packed(button):
    return bool(button.winfo_manager())


def test_home_has_no_stop_button_until_a_run_starts(make_app, tmp_path):
    app = make_app()
    page = _home(app)
    assert _packed(page._stop_button) is False


def test_home_stop_button_appears_while_running_and_leaves_after(
        make_app, monkeypatch, tmp_path):
    import laser_trim_analyzer.gui.v6.pages.home_page as home_mod
    _no_threads(monkeypatch, home_mod)
    app = make_app()
    app.config.ingest.add(str(tmp_path))
    page = _home(app)
    page.refresh_folders()
    page._start()
    assert _packed(page._stop_button) is True
    assert str(page._stop_button.cget("state")) == "normal"
    page._on_run_done(IngestReport(results=[], seconds=1.0))
    assert _packed(page._stop_button) is False


def test_home_stop_sets_the_event_and_relabels_itself(make_app, monkeypatch,
                                                      tmp_path):
    import laser_trim_analyzer.gui.v6.pages.home_page as home_mod
    _no_threads(monkeypatch, home_mod)
    app = make_app()
    app.config.ingest.add(str(tmp_path))
    page = _home(app)
    page.refresh_folders()
    page._start()
    page._stop()
    assert page._cancel.is_set()
    assert "Stopping" in page._stop_button.cget("text")
    assert str(page._stop_button.cget("state")) == "disabled"
    # ...and the label comes back for the next run.
    page._on_run_done(IngestReport(results=[], seconds=1.0))
    page._start()
    assert page._stop_button.cget("text") == "Stop"
    assert page._cancel.is_set() is False


def test_home_hands_the_same_event_to_the_runner(make_app, monkeypatch):
    seen = {}

    def fake(folders, **kw):
        seen["cancel"] = kw.get("cancel")
        return IngestReport(results=[], seconds=1.0)

    monkeypatch.setattr(ingest_run, "run_folders", fake)
    app = make_app()
    page = _home(app)
    page._cancel = Event()
    page._run(["/laser/a"], True, page._cancel)
    assert seen["cancel"] is page._cancel


def test_home_stop_before_a_run_is_a_no_op(make_app):
    page = _home(make_app())
    page._stop()                        # must not raise
    assert _packed(page._stop_button) is False


# ---- Process page ----------------------------------------------------------

def _process(app):
    return app.page_container.get_page("process")


def test_process_page_stop_button_appears_while_running(make_app, monkeypatch,
                                                        tmp_path):
    import laser_trim_analyzer.gui.v6.pages.process_page as proc_mod
    _no_threads(monkeypatch, proc_mod)
    app = make_app()
    page = _process(app)
    assert _packed(page._stop_button) is False
    page._folder_picker.set_value(str(tmp_path))
    page._start()
    assert _packed(page._stop_button) is True
    page._stop()
    assert page._cancel.is_set()
    assert "Stopping" in page._stop_button.cget("text")
    page._on_done()
    assert _packed(page._stop_button) is False


def test_process_page_hands_the_event_to_the_runner(make_app, monkeypatch,
                                                    tmp_path):
    seen = {}

    def fake(folder, **kw):
        seen["cancel"] = kw.get("cancel")
        return FolderResult(folder=folder, ok=True, files_found=2, new_files=2)

    monkeypatch.setattr(ingest_run, "run_folder", fake)
    app = make_app()
    page = _process(app)
    cancel = Event()
    page._run(str(tmp_path), incremental=False, cancel=cancel)
    assert seen["cancel"] is cancel


def test_process_page_says_it_stopped_early(make_app, monkeypatch, tmp_path):
    """A cancelled folder must not read as a completed one."""
    def fake(folder, **kw):
        return FolderResult(folder=folder, ok=True, files_found=100,
                            new_files=20, cancelled=True)

    monkeypatch.setattr(ingest_run, "run_folder", fake)
    app = make_app()
    page = _process(app)
    page._run(str(tmp_path), incremental=True, cancel=Event())
    app.ui._drain()
    assert "Stopped" in page._progress._status.cget("text")


# ---- closing the window ----------------------------------------------------

def test_closing_the_window_stops_an_ingest_before_destroying(make_app):
    """Killing Tk mid-batch was the only previous way out of a long run."""
    import threading
    app = make_app()
    cancel, ran = Event(), []

    def work():
        cancel.wait(10.0)
        ran.append("finished its batch")

    t = threading.Thread(target=work, daemon=True)
    t.start()
    app.register_ingest(cancel, t)
    app._on_closing()
    assert cancel.is_set()
    assert ran == ["finished its batch"], "the worker was not given its 2 s"
    assert t.is_alive() is False


def test_closing_with_no_run_in_flight_just_closes(make_app):
    app = make_app()
    app._on_closing()          # must not hang or raise


def test_closing_does_not_wait_forever_on_a_stuck_worker(make_app,
                                                         monkeypatch):
    """The grace period is a grace period, not a hostage situation."""
    import threading
    app = make_app()
    stuck = Event()
    t = threading.Thread(target=lambda: stuck.wait(30), daemon=True)
    t.start()
    cancel = Event()
    app.register_ingest(cancel, t)
    t0 = time.monotonic()
    app.stop_ingests(timeout=0.2)
    stuck.set()
    assert time.monotonic() - t0 < 5.0
