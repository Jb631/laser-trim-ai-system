import threading
import time

import customtkinter as ctk


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _find_button(root, text):
    for w in _walk(root):
        if isinstance(w, ctk.CTkButton) and w.cget("text") == text:
            return w
    return None


def _find_status_label(root):
    """The section's status label: the first empty-text CTkLabel it packs."""
    for w in _walk(root):
        if isinstance(w, ctk.CTkLabel) and w.cget("text") == "":
            return w
    return None


def _section(app):
    from laser_trim_analyzer.gui.v6.sections import database_cleanup
    frame = ctk.CTkFrame(app)
    database_cleanup.build_database_cleanup_section(frame, app.theme, app)
    return frame, _find_button(frame, "Refresh process findings"), _find_status_label(frame)


def _pump(app, until, timeout=5.0):
    """Run the Tk event loop until `until()` is true (posted UI updates need it)."""
    end = time.time() + timeout
    while time.time() < end and not until():
        app.update()
        time.sleep(0.02)
    return until()


def test_the_button_refreshes_every_model_and_reports_the_count(make_app, monkeypatch):
    from laser_trim_analyzer.findings import engine
    app = make_app()
    calls = []
    monkeypatch.setattr(engine, "refresh_findings", lambda db, models=None: calls.append(models) or 7)
    frame, button, label = _section(app)
    try:
        assert button is not None
        button.invoke()
        assert _pump(app, lambda: "7 findings" in label.cget("text")), label.cget("text")
        assert calls == [None]                       # None = every model with trim data
        assert _pump(app, lambda: app.active_run_name() is None)   # unregistered when done
    finally:
        frame.destroy()


def test_the_button_refuses_while_another_run_is_in_flight(make_app, monkeypatch):
    from laser_trim_analyzer.findings import engine
    app = make_app()
    calls = []
    monkeypatch.setattr(engine, "refresh_findings", lambda db, models=None: calls.append(models) or 0)
    frame, button, label = _section(app)
    cancel = threading.Event()
    thread = threading.Thread(target=cancel.wait, daemon=True)
    thread.start()
    app.register_ingest(cancel, thread, "An ingest")
    try:
        button.invoke()
        app.update_idletasks()
        said = label.cget("text")
        assert "An ingest is running" in said and "let it finish first" in said, said
        assert calls == [], "it touched the database anyway"
    finally:
        cancel.set()
        thread.join(2.0)
        frame.destroy()


def test_while_it_runs_the_app_reports_a_findings_refresh_in_flight(make_app, monkeypatch):
    from laser_trim_analyzer.findings import engine
    app = make_app()
    release = threading.Event()
    monkeypatch.setattr(engine, "refresh_findings", lambda db, models=None: release.wait(5) and 0 or 0)
    frame, button, label = _section(app)
    try:
        button.invoke()
        assert _pump(app, lambda: app.active_run_name() == "A findings refresh"), app.active_run_name()
        release.set()
        assert _pump(app, lambda: app.active_run_name() is None)
    finally:
        release.set()
        frame.destroy()
