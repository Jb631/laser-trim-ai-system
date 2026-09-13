"""The run knows its own size: overall progress, a rate, and an ETA.

James, mid-ingest at work: "why the app is running it doesnt tell me how many
files it has to go?" It could not: the progress panel only ever knew the
CURRENT folder's total, reset to zero at every folder boundary, with no rate
and no ETA. A bar that restarts twice during a four-hour run and never names a
finishing time answers none of the questions someone standing over it has.

What is pinned here:

  * The arithmetic is PURE and lives in core/ingest_run.py. The widget gets a
    fraction and a finished sentence; it computes nothing. That is what makes
    the wording testable at all, and what stops two pages drifting into two
    different vocabularies for the same run.
  * The rate is a MOVING average over the last minute, so a share that goes
    slow is visible within a minute instead of being averaged away by an hour
    of earlier speed.
  * No invented precision. "about 7 min left" is a claim the estimator can
    support; "6 min 41 s left" is not, and reading it makes people wait for a
    number that was never real. Nothing is claimed at all for the first 30
    seconds, when the sample is one folder's worth of small files.
  * The bar counts WORK, not files on disk (2026-09-13). It used to count
    every file discovered and credit the already-known ones back a folder at a
    time, so on a re-run the thousands of known files in every folder the run
    had not reached yet sat inside "remaining" — and remaining divided by a
    real processing rate is where James's "it says like 24 hours" came from.
    The pre-scan decides new-vs-known from the in-memory index before the
    first file opens; the total is what is left to do, and the line says so.
"""
from threading import Event

import pytest

from laser_trim_analyzer.core import ingest_run
from laser_trim_analyzer.core.processor import Processor as _RealProcessor
from laser_trim_analyzer.core.ingest_run import (
    EtaEstimator,
    FolderResult,
    ProgressCoalescer,
    format_clock,
    format_progress_line,
    format_rate,
    run_folders,
)


def _status(**kw):
    from laser_trim_analyzer.core.models import ProcessingStatus
    kw.setdefault("progress_percent", 0.0)
    kw.setdefault("filename", "")
    return ProcessingStatus(**kw)


def _feed(est, pairs):
    """(time, processed) samples, oldest first."""
    for t, done in pairs:
        est.note(done, now=t)
    return est


# ---- the rate --------------------------------------------------------------

def test_rate_is_files_per_second_across_the_window():
    est = _feed(EtaEstimator(now=0.0),
                [(t, 17 * t) for t in range(0, 41, 5)])
    assert est.rate(now=40.0) == pytest.approx(17.0, abs=0.01)


def test_rate_forgets_samples_older_than_the_window():
    """A minute of 100/s followed by a minute of 5/s reads as ~5/s, not ~52/s
    — a share that went slow is the thing you need to see."""
    est = EtaEstimator(now=0.0)
    _feed(est, [(t, 100 * t) for t in range(0, 61, 5)])
    _feed(est, [(60 + t, 6000 + 5 * t) for t in range(5, 66, 5)])
    assert est.rate(now=125.0) == pytest.approx(5.0, abs=0.5)


def test_no_rate_from_a_sliver_of_time():
    """Two samples 0.25 s apart say nothing; publishing 4 files/s off one tick
    would make the first ETA wild."""
    est = _feed(EtaEstimator(now=0.0), [(0.0, 0), (0.25, 1)])
    assert est.rate(now=0.25) is None


def test_a_stall_drives_the_rate_down_instead_of_freezing_it():
    est = EtaEstimator(now=0.0)
    _feed(est, [(t, 20 * t) for t in range(0, 21)])
    fast = est.rate(now=20.0)
    _feed(est, [(20 + t, 400) for t in range(1, 41)])   # nothing moves
    slow = est.rate(now=60.0)
    assert fast == pytest.approx(20.0, abs=0.1)
    assert slow is not None and slow < fast / 2


def test_a_dead_stop_reports_no_rate_rather_than_zero_division():
    est = EtaEstimator(now=0.0)
    _feed(est, [(t, 0) for t in range(0, 121, 5)])
    assert est.rate(now=120.0) == 0.0
    assert est.eta_text(500, now=120.0) == "estimating…"


# ---- the ETA wording -------------------------------------------------------

def test_no_eta_at_all_for_the_first_thirty_seconds():
    est = _feed(EtaEstimator(now=0.0), [(t, 20 * t) for t in range(0, 30, 2)])
    assert est.eta_text(10_000, now=29.0) == "estimating…"
    assert est.eta_text(10_000, now=31.0) != "estimating…"


def test_under_a_minute_is_not_counted_in_seconds():
    est = _feed(EtaEstimator(now=0.0), [(t, 10 * t) for t in range(0, 41, 2)])
    assert est.eta_text(300, now=40.0) == "under a minute left"


def test_minutes_are_rounded_and_said_plainly():
    est = _feed(EtaEstimator(now=0.0), [(t, 10 * t) for t in range(0, 41, 2)])
    assert est.eta_text(4_000, now=40.0) == "about 7 min left"
    # ...and never with a fake second-level tail.
    assert "s left" not in est.eta_text(4_000, now=40.0)


def test_hours_read_as_hours_and_minutes():
    est = _feed(EtaEstimator(now=0.0), [(t, 10 * t) for t in range(0, 41, 2)])
    assert est.eta_text(100_000, now=40.0) == "about 2 h 47 min left"


def test_a_whole_number_of_hours_drops_the_minutes():
    est = _feed(EtaEstimator(now=0.0), [(t, 10 * t) for t in range(0, 41, 2)])
    assert est.eta_text(72_000, now=40.0) == "about 2 h left"


def test_one_minute_is_the_floor_above_a_minute():
    est = _feed(EtaEstimator(now=0.0), [(t, 10 * t) for t in range(0, 41, 2)])
    assert est.eta_text(650, now=40.0) == "about 1 min left"


def test_nothing_left_to_do_says_nothing_rather_than_under_a_minute():
    """The last paint of a finished run has remaining == 0."""
    est = _feed(EtaEstimator(now=0.0), [(t, 10 * t) for t in range(0, 41, 2)])
    assert est.eta_text(0, now=40.0) == ""


# ---- the line --------------------------------------------------------------

def test_the_progress_line_says_how_many_are_left():
    """James, mid-ingest: "it doesnt tell me how many files it has to go".
    "1,214 / 8,900" made him do the subtraction — and until the total became
    the run's WORK, the subtraction gave the wrong answer anyway."""
    assert format_progress_line(1214, 8900, folder=2, folders=3, rate=17.0,
                                eta="about 7 min left", elapsed=724.0) == \
        "Folder 2 of 3 · 1,214 done · 7,686 to go · 17 files/s · about 7 min " \
        "left · elapsed 12:04"


def test_the_line_drops_what_it_does_not_know_yet():
    """Before the scan finishes there is no total and no rate — say the one
    real number rather than "0 to go · 0 files/s"."""
    line = format_progress_line(12, 0, folder=1, folders=1, rate=None,
                                eta="estimating…", elapsed=4.0)
    assert line == "12 done · estimating… · elapsed 0:04"


def test_a_single_folder_run_does_not_say_folder_1_of_1():
    line = format_progress_line(5, 10, folder=1, folders=1, rate=2.0,
                                eta="under a minute left", elapsed=3.0)
    assert line.startswith("5 done · 5 to go")


def test_the_line_can_still_name_the_file_it_is_on():
    line = format_progress_line(5, 10, folder=1, folders=1, elapsed=3.0,
                                filename="8340-1_A12345.xls")
    assert line.endswith("· 8340-1_A12345.xls")


def test_rate_wording_keeps_one_decimal_only_where_it_means_something():
    assert format_rate(17.4) == "17 files/s"
    assert format_rate(3.26) == "3.3 files/s"
    assert format_rate(0.4) == "0.4 files/s"


def test_clock_wording():
    assert format_clock(4) == "0:04"
    assert format_clock(724) == "12:04"
    assert format_clock(3725) == "1:02:05"


# ---- already-processed files count as progress -----------------------------

def test_the_scan_credits_known_files_to_the_bar_in_one_step():
    """148k files the database already has must not read as 148k files still
    to do — the denominator counts every file discovered."""
    c = ProgressCoalescer()
    c.note(_status(status="known", count=1480))
    snap = c.drain()
    assert snap["done"] == 1480
    assert snap["moved"] is True


def test_credited_files_do_not_land_in_the_skipped_counter():
    """The five per-folder counters stay exactly what they were: Skipped is
    non-trim files, and the final BatchSummary reconciles the rest."""
    c = ProgressCoalescer()
    c.note(_status(status="known", count=900))
    assert c.drain()["counts"].get("skipped") is None


def test_the_rate_is_measured_on_real_work_not_on_the_credit():
    """A step of 148,000 in one tick is not 592,000 files/s."""
    c = ProgressCoalescer()
    c.note(_status(status="known", count=148_000))
    c.note(_status(status="completed", filename="a.xls"))
    snap = c.drain()
    assert snap["done"] == 148_001
    assert snap["processed"] == 1


def test_processed_and_done_agree_when_nothing_was_credited():
    c = ProgressCoalescer()
    for i in range(3):
        c.note(_status(status="completed", filename=f"{i}.xls"))
    snap = c.drain()
    assert snap["done"] == 3 and snap["processed"] == 3


def test_reset_clears_the_credit_too():
    c = ProgressCoalescer()
    c.note(_status(status="known", count=10))
    c.reset()
    assert c.drain()["done"] == 0 and c.drain()["processed"] == 0


# ---- the run scans itself first --------------------------------------------

def _fake_walk(monkeypatch, sizes, log=None):
    # The pre-scan asks whether each folder is usable before walking it (that
    # check is how an offline share stays run_folder's story); these paths do
    # not exist on disk, so say they are fine.
    monkeypatch.setattr(ingest_run, "ingest_folder_problem", lambda f: None)

    def fake(folder):
        if log is not None:
            log.append(("walk", folder))
        n = sizes[folder]
        files = [f"{folder}/f{i}.xls" for i in range(n)]
        return files, {f: (10, 1.0) for f in files}
    monkeypatch.setattr(ingest_run, "discover_excel_files", fake)


def _fake_run_folder(monkeypatch, log):
    def fake(folder, **kw):
        log.append(("process", folder))
        disc = kw.get("discovered")
        return FolderResult(folder=folder, ok=True,
                            files_found=len(disc[0]) if disc else 0,
                            new_files=1)
    monkeypatch.setattr(ingest_run, "run_folder", fake)


def test_every_folder_is_walked_before_any_file_is_processed(monkeypatch):
    log = []
    _fake_walk(monkeypatch, {"/a": 3, "/b": 5, "/c": 2}, log)
    _fake_run_folder(monkeypatch, log)
    run_folders(["/a", "/b", "/c"], db=None, config=None)
    assert log == [("walk", "/a"), ("walk", "/b"), ("walk", "/c"),
                   ("process", "/a"), ("process", "/b"), ("process", "/c")]


def test_the_walk_is_not_paid_for_twice(monkeypatch):
    """73 minutes for 170k files is what one extra walk of the share costs
    (2026-07-10). The pre-scan hands its result to the folder run."""
    log = []
    _fake_walk(monkeypatch, {"/a": 3, "/b": 5}, log)
    seen = {}

    def fake(folder, **kw):
        seen[folder] = kw.get("discovered")
        return FolderResult(folder=folder, ok=True)

    monkeypatch.setattr(ingest_run, "run_folder", fake)
    run_folders(["/a", "/b"], db=None, config=None)
    assert [f for kind, f in log if kind == "walk"] == ["/a", "/b"]
    assert len(seen["/a"][0]) == 3 and len(seen["/b"][0]) == 5


def test_the_run_reports_one_overall_total_before_processing(monkeypatch):
    log, totals = [], []
    _fake_walk(monkeypatch, {"/a": 3, "/b": 5, "/c": 2}, log)

    def fake(folder, **kw):
        log.append(("process", folder))
        return FolderResult(folder=folder, ok=True)

    monkeypatch.setattr(ingest_run, "run_folder", fake)
    report = run_folders(["/a", "/b", "/c"], db=None, config=None,
                         on_total=lambda n: (totals.append(n),
                                             log.append(("total", n))))
    assert totals == [10], "the total must be announced once, for the run"
    assert log[3] == ("total", 10)          # after the walks, before the work
    assert report.files_planned == 10


def test_the_scan_says_which_folder_it_is_on(monkeypatch):
    said = []
    _fake_walk(monkeypatch, {"/a": 1, "/b": 1})
    _fake_run_folder(monkeypatch, [])
    run_folders(["/a", "/b"], db=None, config=None, on_phase=said.append)
    assert any("1 of 2" in s for s in said)
    assert any("2 of 2" in s for s in said)
    assert all("Scanning" in s for s in said[:2])


def test_stop_during_the_scan_never_starts_processing(monkeypatch):
    log, cancel = [], Event()
    monkeypatch.setattr(ingest_run, "ingest_folder_problem", lambda f: None)

    def fake_walk(folder):
        log.append(("walk", folder))
        cancel.set()
        return [], {}

    monkeypatch.setattr(ingest_run, "discover_excel_files", fake_walk)
    _fake_run_folder(monkeypatch, log)
    report = run_folders(["/a", "/b"], db=None, config=None, cancel=cancel)
    assert log == [("walk", "/a")]
    assert report.cancelled is True
    # A half-finished scan is not a denominator.
    assert report.files_planned == 0


def test_an_unusable_folder_is_left_for_the_folder_run_to_report(monkeypatch):
    """The pre-scan must not swallow an offline share: run_folder owns the
    "not found — offline share?" wording and the ok=False result."""
    log = []
    _fake_walk(monkeypatch, {"/a": 4}, log)
    monkeypatch.setattr(ingest_run, "ingest_folder_problem",   # after the helper
                        lambda f: "not found — offline share?" if f == "/gone" else None)
    seen = {}

    def fake(folder, **kw):
        seen[folder] = kw.get("discovered")
        return FolderResult(folder=folder, ok=folder != "/gone",
                            error=None if folder != "/gone" else "not found")

    monkeypatch.setattr(ingest_run, "run_folder", fake)
    report = run_folders(["/a", "/gone"], db=None, config=None)
    assert log == [("walk", "/a")]           # the dead share was not walked
    assert seen["/gone"] is None             # ...and gets no pre-walk to reuse
    assert [r.folder for r in report.failed] == ["/gone"]


def test_a_pre_walked_folder_skips_its_own_walk(monkeypatch, tmp_path):
    """run_folder honours what it is handed instead of walking again."""
    called = []
    monkeypatch.setattr(ingest_run, "discover_excel_files",
                        lambda f: (called.append(f), ([], {}))[1])

    class _Nothing:
        last_scan_stats = {}

        def __init__(self, *a, **k):
            pass

        def process_batch(self, paths, **kw):
            self.seen = list(paths)
            return iter(())

    monkeypatch.setattr(ingest_run, "Processor", _Nothing)
    files = [str(tmp_path / "a.xls"), str(tmp_path / "b.xls")]
    res = ingest_run.run_folder(str(tmp_path), db=None, config=None,
                                incremental=False,
                                discovered=(files, {}), walk_seconds=4.5)
    assert called == []
    assert res.files_found == 2
    assert res.phases["walk"] == 4.5


# ---- the widget stays a view ----------------------------------------------

def test_set_overall_sets_the_fraction_and_the_sentence(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.process_progress_section import (
        ProcessProgressSection)
    s = ProcessProgressSection(tk_root, theme=ThemeManager())
    s.set_overall(25, 100, "Folder 1 of 2 · 25 / 100 files")
    assert s._bar.get() == pytest.approx(0.25)
    assert s._status.cget("text") == "Folder 1 of 2 · 25 / 100 files"


def test_set_phase_does_not_throw_the_bar_away(tk_root):
    """on_phase fires mid-run ("Checking 8,900 files against the database…").
    set_idle would zero the bar, which reads as progress lost."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.process_progress_section import (
        ProcessProgressSection)
    s = ProcessProgressSection(tk_root, theme=ThemeManager())
    s.set_overall(50, 100, "half")
    s.set_phase("Re-linked 12 final-test records…")
    assert s._bar.get() == pytest.approx(0.5)
    assert s._status.cget("text") == "Re-linked 12 final-test records…"


def test_the_single_event_path_credits_known_files_as_well(make_app):
    """The Process page keeps a one-event-at-a-time path beside the coalesced
    one; the two must not disagree about what a "known" event means."""
    app = make_app()
    page = app.page_container.get_page("process")
    page._done = 0
    page._apply_progress(_status(status="known", count=42), total=100)
    assert page._done == 42
    assert page._progress._counters["skipped"] == 0


# ---- HOME paints the whole run --------------------------------------------

def _home(app):
    return app.page_container.get_page("home")


def test_home_paints_the_run_total_not_the_folders(make_app):
    app = make_app()
    page = _home(app)
    coalescer = ProgressCoalescer()
    coalescer.note(_status(status="known", count=400))
    coalescer.note(_status(status="completed", filename="a.xls"))
    state = {"n": 8900, "folder": 2, "folders": 3}
    page._paint(coalescer, state, EtaEstimator())
    text = page._progress._status.cget("text")
    assert "401 done · 8,499 to go" in text
    assert "Folder 2 of 3" in text


def test_home_does_not_zero_the_count_at_a_folder_boundary(make_app,
                                                           monkeypatch):
    """The bar is the RUN's now. Resetting it per folder is what made "how
    many to go" unanswerable in the first place."""
    started = []

    def fake(folders, **kw):
        start = kw.get("on_folder_start")
        start(1, 2, "/a")
        kw["progress"].note(_status(status="completed", filename="a.xls"))
        start(2, 2, "/b")
        started.append(kw["progress"].drain()["done"])
        return ingest_run.IngestReport(results=[], seconds=1.0)

    monkeypatch.setattr(ingest_run, "run_folders", fake)
    app = make_app()
    _home(app)._run(["/a", "/b"], True, Event())
    assert started == [1], "the folder boundary threw away a file's progress"


# ---- the total is WORK, not files on disk ---------------------------------
#
# The bug this section exists for (James, 2026-09-12): "we recently tried to
# add to the processor to show how many files are left to process while its
# working. thats not working and the time remaining is not working its saying
# like 24 hours sometimes?" — on a re-run where the database already had
# nearly everything. The pre-scan counted every Excel file on the share, and a
# folder's known population only left "remaining" when that folder's own scan
# landed, so the denominator was ~150,000 files nobody was going to open.

def _known_db(tmp_path, paths):
    """A real SQLite DB whose processed_files rows match `paths` on disk.

    Real rows, real (size, mtime): `_classify_scan`'s stat fast-path is the
    thing under test, and a mocked index could not exercise it.
    """
    from datetime import datetime

    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import ProcessedFile

    db = DatabaseManager(tmp_path / "known.db")
    with db.session() as s:
        for i, p in enumerate(paths):
            st = p.stat()
            s.add(ProcessedFile(
                filename=p.name, file_path=str(p), file_hash=f"{i:064d}",   # the model validates 64 hex chars
                file_size=st.st_size,
                file_modified_date=datetime.fromtimestamp(st.st_mtime),
                success=True))
    return db


def _folder(tmp_path, name, n):
    d = tmp_path / name
    d.mkdir()
    out = []
    for i in range(n):
        f = d / f"{name}_{i:03d}.xls"
        f.write_bytes(b"x" * (10 + i))       # distinct sizes: real stat data
        out.append(f)
    return d, out


def _use_db(monkeypatch, db):
    """Point the module-global manager at this DB.

    Processor._load_processed_hashes reaches for `get_database()`, which
    IGNORES any manager handed to run_folders — the whole reason the QA sweep
    sets this too (memory: get_database() bypasses injection).
    """
    from laser_trim_analyzer.database import manager as _dbmod
    monkeypatch.setattr(_dbmod, "_db_manager", db)


def _totals_run(monkeypatch, folders, *, incremental=True):
    """run_folders with the real pre-scan and a stubbed folder run."""
    said, totals = [], []
    monkeypatch.setattr(ingest_run, "run_folder",
                        lambda folder, **kw: FolderResult(folder=folder, ok=True))
    run_folders([str(f) for f in folders], db=None, config=None,
                incremental=incremental, on_phase=said.append,
                on_total=totals.append)
    return totals, said


def test_the_total_counts_only_what_is_left_to_do(tmp_path, monkeypatch):
    """8 files on disk, 5 of them already in the database: the run has 3."""
    folder, files = _folder(tmp_path, "laser", 8)
    _use_db(monkeypatch, _known_db(tmp_path, files[:5]))
    totals, said = _totals_run(monkeypatch, [folder])
    assert totals == [3]
    assert any("5 files already in the database will be skipped." in s
               for s in said)


def test_a_folder_that_is_entirely_known_is_not_in_the_remainder(tmp_path,
                                                                 monkeypatch):
    """The shape of James's re-run: folder 2 is done, and until the run
    reached it, every one of its files counted as still to do."""
    a, a_files = _folder(tmp_path, "new_work", 4)
    b, b_files = _folder(tmp_path, "all_known", 60)
    _use_db(monkeypatch, _known_db(tmp_path, b_files))
    totals, said = _totals_run(monkeypatch, [a, b])
    assert totals == [4], "folder 2 was already in the database"
    assert any("60 files already in the database" in s for s in said)
    # ...and the line the user reads says the same thing.
    assert format_progress_line(0, totals[0], folder=1, folders=2,
                                elapsed=1.0) == \
        "Folder 1 of 2 · 0 done · 4 to go · elapsed 0:01"


def test_a_full_reprocess_counts_every_file(tmp_path, monkeypatch):
    """Incremental unchecked: nothing is skipped, so everything is work."""
    folder, files = _folder(tmp_path, "laser", 6)
    _use_db(monkeypatch, _known_db(tmp_path, files))
    totals, said = _totals_run(monkeypatch, [folder], incremental=False)
    assert totals == [6]
    assert not any("already in the database" in s for s in said)


def test_an_unreadable_index_falls_back_to_every_file_and_says_so(tmp_path,
                                                                  monkeypatch):
    """Wrong to the SAFE side: over-state the run rather than promise a finish
    that isn't coming — and never silently."""
    folder, files = _folder(tmp_path, "laser", 7)
    _use_db(monkeypatch, _known_db(tmp_path, files[:5]))

    class _NoIndex:
        def __init__(self, *a, **k):
            pass

        def _load_processed_hashes(self):
            raise RuntimeError("database is locked")

    monkeypatch.setattr(ingest_run, "Processor", _NoIndex)
    totals, said = _totals_run(monkeypatch, [folder])
    assert totals == [7]
    assert any("Could not read the processed-file index" in s for s in said)


def test_the_run_hands_the_coalescer_the_total_and_the_known_count(tmp_path,
                                                                   monkeypatch):
    """The two halves have to meet: the pre-scan's per-folder known count is
    what the coalescer subtracts from the processor's bulk credit."""
    folder, files = _folder(tmp_path, "laser", 9)
    _use_db(monkeypatch, _known_db(tmp_path, files[:6]))
    seen = {}

    class _Spy(ProgressCoalescer):
        def set_total(self, total):
            seen["total"] = total
            super().set_total(total)

        def expect_known(self, n):
            seen["known"] = n
            super().expect_known(n)

    spy = _Spy()
    monkeypatch.setattr(ingest_run, "Processor", _NothingProcessor)
    run_folders([str(folder)], db=None, config=None, progress=spy,
                incremental=True)
    assert seen == {"total": 3, "known": 6}


class _NothingProcessor(_RealProcessor):
    """The REAL incremental scan with the parsing taken out.

    It cannot be a bare stub: the pre-scan drives the same class through
    `_classify_scan`, so a processor with no scan would test the fallback
    instead of the plan.
    """

    def __init__(self, *a, **k):
        k.setdefault("use_ml", False)     # the ML load goes to the DB global
        super().__init__(*a, **k)

    def process_batch(self, paths, **kw):
        return iter(())


# ---- the coalescer counts work, not files ---------------------------------

def test_the_bulk_credit_drops_the_files_the_scan_already_knew():
    """summary.skipped counts BOTH the memory-settled files and the ones the
    scan had to hash. Only the hashed ones were work this run paid for."""
    c = ProgressCoalescer()
    c.set_total(200)
    c.expect_known(1480)                      # what the pre-scan settled free
    c.note(_status(status="known", count=1530))   # +50 verified by hash
    snap = c.drain()
    assert snap["done"] == 50
    assert snap["moved"] is True


def test_a_folder_the_scan_settled_whole_moves_nothing():
    """Every file known in memory: no work was done, so the bar must not move
    — and the panel keeps showing the scan's own sentence."""
    c = ProgressCoalescer()
    c.set_total(0)
    c.expect_known(150_938)
    c.note(_status(status="known", count=150_938))
    snap = c.drain()
    assert snap["done"] == 0 and snap["moved"] is False


def test_the_sequential_path_spends_the_same_budget_one_file_at_a_time():
    """Folders under the turbo threshold skip each known file individually —
    there is no bulk event to subtract from."""
    c = ProgressCoalescer()
    c.set_total(2)
    c.expect_known(3)
    for i in range(3):
        c.note(_status(status="skipped", filename=f"known{i}.xls"))
    assert c.drain()["done"] == 0
    for i in range(2):
        c.note(_status(status="completed", filename=f"new{i}.xls"))
    assert c.drain()["done"] == 2


def test_each_folder_gets_its_own_budget_not_the_last_one_s_leftovers():
    c = ProgressCoalescer()
    c.set_total(100)
    c.expect_known(40)
    c.note(_status(status="known", count=40))     # folder 1: all known
    c.expect_known(5)                             # folder 2 starts fresh
    c.note(_status(status="known", count=25))     # 5 settled, 20 verified
    assert c.drain()["done"] == 20


def test_the_bar_never_overruns_the_total_it_announced():
    """If the plan and the processor ever disagree, clamp and LOG it — a bar
    reading 104% hides the disagreement instead of reporting it."""
    c = ProgressCoalescer()
    c.set_total(10)
    for i in range(15):
        c.note(_status(status="completed", filename=f"{i}.xls"))
    assert c.drain()["done"] == 10


def test_without_an_announced_total_nothing_is_clamped():
    """A direct run_folder call with no plan still counts honestly."""
    c = ProgressCoalescer()
    for i in range(15):
        c.note(_status(status="completed", filename=f"{i}.xls"))
    assert c.drain()["done"] == 15
