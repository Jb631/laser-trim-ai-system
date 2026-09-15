"""One long job at a time: a re-grade and an ingest must not overlap.

Work incident, 2026-09-14. The re-grade was started at 14:19. At 14:45, with
it still running, "Process everything new" was pressed. Nothing crashed and
nothing was corrupted — they simply shared one SQLite write lock and one SMB
link, and both got much worse:

    processed-file index load   2.6 s  ->  26 s, 32 s, 88 s, 101 s per folder
    final-test verify pass      140 s  ->  542 s
    the batch                            cancelled after 0 of 1,371 files

The re-grade kept going until 15:35. So the app now knows what is running and
says so, in ONE place — `V6App.active_run_name` — which both front ends ask.

These tests drive the registry and the page methods directly rather than
clicking widgets: the refusal is a decision, and a decision is testable
without a mainloop.
"""
import threading
import time
from threading import Event

import pytest

from laser_trim_analyzer.core.ingest_run import IngestReport


def _home(app):
    return app.page_container.get_page("home")


def _live_run(app, name):
    """Register a job that really is alive, and hand back its stop switch."""
    cancel = Event()
    thread = threading.Thread(target=lambda: cancel.wait(10.0), daemon=True)
    thread.start()
    app.register_ingest(cancel, thread, name)
    return cancel, thread


# ---- the one place that answers "is something running?" --------------------

def test_active_run_name_is_none_when_nothing_is_running(make_app):
    app = make_app()
    assert app.active_run_name() is None


def test_active_run_name_names_the_run_in_flight(make_app):
    app = make_app()
    cancel, thread = _live_run(app, "A re-grade")
    try:
        assert app.active_run_name() == "A re-grade"
    finally:
        cancel.set()
        thread.join(2.0)


def test_active_run_name_forgets_a_finished_run(make_app):
    app = make_app()
    cancel, thread = _live_run(app, "An ingest")
    cancel.set()
    thread.join(2.0)
    assert app.active_run_name() is None, (
        "a finished run must not block the next one forever")


def test_register_ingest_still_takes_two_arguments(make_app):
    """The Process page and two existing suites call it positionally."""
    app = make_app()
    cancel = Event()
    thread = threading.Thread(target=lambda: cancel.wait(5.0), daemon=True)
    thread.start()
    try:
        app.register_ingest(cancel, thread)
        assert app.active_run_name() == "An ingest"
    finally:
        cancel.set()
        thread.join(2.0)


def test_closing_the_window_still_stops_every_registered_run(make_app):
    """The registry gained a name; it must not have lost its original job."""
    app = make_app()
    cancel_a, thread_a = _live_run(app, "An ingest")
    cancel_b, thread_b = _live_run(app, "A re-grade")
    app._on_closing()
    assert cancel_a.is_set() and cancel_b.is_set()
    assert not thread_a.is_alive() and not thread_b.is_alive()


# ---- HOME refuses while anything else is running ---------------------------

def _home_that_records_instead_of_ingesting(app, tmp_path, gate=None):
    """HOME with its worker body replaced by a recorder.

    `_start` still builds and starts a REAL thread (that is the code under
    test), it just has nothing to do — so `threading.Thread` is left alone.
    Patching it module-wide also patched this file's own `_live_run`, which
    is how the first draft of these tests "passed" for the wrong reason.

    `gate`, when given, holds the worker open until it is set, so a test can
    look at the app while a run is genuinely in flight.
    """
    app.config.ingest.add(str(tmp_path))
    page = _home(app)
    page.refresh_folders()
    calls = []

    def record(*a, **k):
        calls.append(a)
        if gate is not None:
            gate.wait(5.0)

    page._run = record
    return page, calls


def test_home_refuses_to_start_while_a_regrade_is_running(make_app, tmp_path):
    app = make_app()
    page, calls = _home_that_records_instead_of_ingesting(app, tmp_path)
    cancel, thread = _live_run(app, "A re-grade")
    try:
        page._start()
        time.sleep(0.1)
        assert calls == [], "an ingest started on top of a re-grade"
        assert page._running is False
        said = page._summary.cget("text")
        assert "A re-grade is running" in said, said
        assert "stop it first" in said, said
    finally:
        cancel.set()
        thread.join(2.0)


def test_home_can_be_pressed_again_the_instant_its_own_run_reports_done(
        make_app, tmp_path):
    """The narrow window the liveness check alone would get wrong.

    A page learns its run finished because the worker POSTED the summary —
    and at that moment the worker thread is still alive for a few more
    instructions. Judged on liveness alone, the very next press would be
    refused by a run that had already delivered its result.
    """
    app = make_app()
    # The worker is held open on purpose: that IS the window under test.
    still_running = Event()
    page, calls = _home_that_records_instead_of_ingesting(
        app, tmp_path, gate=still_running)
    page._start()
    time.sleep(0.1)
    assert len(calls) == 1
    assert app.active_run_name() == "An ingest"
    page._on_run_done(IngestReport(results=[], seconds=1.0))
    assert app.active_run_name() is None, (
        "the app still thinks the finished run is running")
    page._start()
    time.sleep(0.1)
    assert len(calls) == 2
    still_running.set()


def test_home_starts_normally_once_the_other_run_has_finished(make_app,
                                                              tmp_path):
    """The refusal must be a state, not a latch."""
    app = make_app()
    page, calls = _home_that_records_instead_of_ingesting(app, tmp_path)
    cancel, thread = _live_run(app, "A re-grade")
    page._start()
    time.sleep(0.1)
    assert calls == []
    cancel.set()
    thread.join(2.0)
    page._start()
    time.sleep(0.2)
    assert len(calls) == 1, "HOME stayed blocked after the re-grade ended"
    assert page._running is True


# ---- Settings refuses while an ingest is running ---------------------------

def test_the_regrade_button_refuses_while_an_ingest_is_registered(make_app,
                                                                  monkeypatch):
    """Driven through the real section builder, with the count stubbed.

    The check has to happen BEFORE the count: counting is itself a query
    against the database the ingest is busy writing.
    """
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.sections import database_cleanup

    app = make_app()
    counted = []
    monkeypatch.setattr(type(app.db), "count_legacy_ft_verdicts",
                        lambda self: counted.append(1) or 5, raising=False)

    frame = ctk.CTkFrame(app)
    database_cleanup.build_database_cleanup_section(frame, app.theme, app)
    button = _find_button(frame, "Re-grade final tests")
    label = _find_status_label(frame)
    assert button is not None

    cancel, thread = _live_run(app, "An ingest")
    try:
        button.invoke()
        app.update_idletasks()
        said = label.cget("text")
        assert "An ingest is running" in said, said
        assert "stop it first" in said, said
        assert counted == [], "it queried the database anyway"
    finally:
        cancel.set()
        thread.join(2.0)
        frame.destroy()


def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)


def _find_button(root, text):
    import customtkinter as ctk
    for w in _walk(root):
        if isinstance(w, ctk.CTkButton) and w.cget("text") == text:
            return w
    return None


def _find_status_label(root):
    """The section's status label: the first empty-text CTkLabel it packs."""
    import customtkinter as ctk
    for w in _walk(root):
        if isinstance(w, ctk.CTkLabel) and w.cget("text") == "":
            return w
    return None
