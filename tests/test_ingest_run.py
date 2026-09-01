"""The shared ingest run — one pipeline, driven by both Process and Home.

Spec: docs/superpowers/specs/2026-08-29-app-shape-investigate-design.md §1
("same worker, no duplicate pipeline").

`core/ingest_run.py` owns the folder-processing core that used to live inside
ProcessPage._run. These tests pin the parts a second implementation would get
wrong: sequential order, one dead share not taking the batch down with it, the
combined summary's arithmetic, and the incremental flag actually reaching the
processor.
"""
import shutil
from pathlib import Path

import pytest

from laser_trim_analyzer.core import ingest_run
from laser_trim_analyzer.core.ingest_run import (
    FolderResult,
    IngestReport,
    ProgressCoalescer,
    discover_excel_files,
    format_ingest_summary,
    run_folder,
    run_folders,
)

SAMPLES = Path(__file__).resolve().parents[1] / "Work Files" / "Sample_Base_2026-04-10"


# ---- Task 1: the folder walk moved, unchanged -----------------------------

def test_discover_finds_every_excel_file_with_stats(tmp_path):
    made = []
    for rel in ("a.xls", "sub/b.xlsx", "sub/deep/c.XLS"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * (len(rel) + 10))
        made.append(str(p))
    (tmp_path / "notes.txt").write_text("ignored")
    files, stats = discover_excel_files(str(tmp_path))
    assert sorted(files) == sorted(made)
    assert set(stats) == set(made)
    for f in made:
        st = Path(f).stat()
        assert stats[f] == (st.st_size, st.st_mtime)


def test_discover_survives_unreadable_folder(tmp_path):
    """A permissions hiccup on one folder must not end the walk."""
    import os
    good = tmp_path / "good.xls"
    good.write_bytes(b"data")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "hidden.xls").write_bytes(b"data")
    os.chmod(blocked, 0o000)
    try:
        files, _ = discover_excel_files(str(tmp_path))
    finally:
        os.chmod(blocked, 0o755)
    assert str(good) in files


# ---- Task 2: progress coalescing (Tk-free, thread-safe) -------------------

def _status(**kw):
    from laser_trim_analyzer.core.models import ProcessingStatus
    kw.setdefault("progress_percent", 0.0)
    return ProcessingStatus(**kw)


def test_coalescer_counts_done_and_skips_then_resets_on_drain():
    c = ProgressCoalescer()
    c.note(_status(filename="a.xls", status="completed"))
    c.note(_status(filename="b.xls", status="skipped"))
    c.bucket("passed")
    snap = c.drain()
    assert snap["done"] == 2
    assert snap["file"] == "b.xls"
    assert snap["counts"]["skipped"] == 1
    assert snap["counts"]["passed"] == 1
    assert snap["moved"] is True
    # Drain resets the deltas but NOT the running total: a painted counter
    # that resets to zero mid-batch is a bug report waiting to happen.
    again = c.drain()
    assert again["done"] == 2
    assert again["counts"] == {}
    assert again["moved"] is False


def test_coalescer_keeps_scan_message_separate_from_progress():
    c = ProgressCoalescer()
    c.note(_status(filename="", status="scanning", message="Found 12 new files"))
    snap = c.drain()
    assert snap["scan_msg"] == "Found 12 new files"
    assert snap["done"] == 0 and snap["moved"] is False


def test_coalescer_keeps_only_the_last_reasons():
    c = ProgressCoalescer()
    for i in range(25):
        c.bucket("errors", f"file{i}.xls: boom")
    snap = c.drain()
    assert len(snap["reasons"]) <= 10
    assert snap["reasons"][-1] == "file24.xls: boom"


# ---- Task 3: one folder ---------------------------------------------------

def test_run_folder_reports_a_missing_folder_instead_of_calling_it_empty(tmp_path):
    """An offline share and an empty folder are different events. Conflating
    them is how a batch that processed nothing reads as 'nothing new'."""
    res = run_folder(str(tmp_path / "gone"), db=None, config=None)
    assert res.ok is False
    assert res.error and "not found" in res.error.lower()
    assert res.files_found == 0


def test_run_folder_on_an_empty_folder_is_a_success_with_no_files(tmp_path):
    res = run_folder(str(tmp_path), db=None, config=None)
    assert res.ok is True and res.error is None and res.files_found == 0
    assert res.new_files == 0


def test_run_folder_survives_a_processor_explosion_and_names_it(tmp_path,
                                                                monkeypatch):
    (tmp_path / "a.xls").write_bytes(b"junk")

    class _Boom:
        last_scan_stats = {}

        def __init__(self, *a, **k):
            pass

        def process_batch(self, *a, **k):
            raise RuntimeError("disk went away")
            yield  # pragma: no cover  (makes this a generator function)

    monkeypatch.setattr(ingest_run, "Processor", _Boom)
    res = run_folder(str(tmp_path), db=None, config=None)
    assert res.ok is False and "disk went away" in res.error


# ---- Task 4: many folders, in order, surviving a dead one -----------------

class _FakeDb:
    def __init__(self):
        self.saved = []

    def save_analysis(self, result):
        self.saved.append(result)


def _stub_run_folder(monkeypatch, seen, *, failing=()):
    def fake(folder, **kw):
        seen.append((folder, kw.get("incremental")))
        if folder in failing:
            return FolderResult(folder=folder, ok=False, error="share offline")
        return FolderResult(folder=folder, ok=True, files_found=10, new_files=4,
                            seconds=1.0)
    monkeypatch.setattr(ingest_run, "run_folder", fake)


def test_run_folders_visits_every_folder_in_configured_order(monkeypatch):
    seen = []
    _stub_run_folder(monkeypatch, seen)
    report = run_folders(["/laser/a", "/laser/b", "/final_test"],
                         db=None, config=None, incremental=True)
    assert [f for f, _ in seen] == ["/laser/a", "/laser/b", "/final_test"]
    assert [r.folder for r in report.results] == ["/laser/a", "/laser/b", "/final_test"]


def test_a_dead_share_does_not_abort_the_run_and_is_named(monkeypatch):
    seen = []
    _stub_run_folder(monkeypatch, seen, failing={"/laser/b"})
    report = run_folders(["/laser/a", "/laser/b", "/final_test"],
                         db=None, config=None)
    # Every folder was still attempted...
    assert [f for f, _ in seen] == ["/laser/a", "/laser/b", "/final_test"]
    # ...and the failure is reported, not swallowed.
    assert [r.folder for r in report.failed] == ["/laser/b"]
    assert report.ok is False
    line = format_ingest_summary(report)
    assert "/laser/b" in line and "share offline" in line


def test_combined_counts_and_elapsed_aggregate(monkeypatch):
    seen = []
    _stub_run_folder(monkeypatch, seen)
    report = run_folders(["/a", "/b", "/c"], db=None, config=None)
    assert report.new_files == 12          # 3 folders x 4
    assert report.files_found == 30
    assert report.folder_count == 3
    assert report.seconds >= 0


def test_incremental_flag_is_threaded_through_to_every_folder(monkeypatch):
    seen = []
    _stub_run_folder(monkeypatch, seen)
    run_folders(["/a", "/b"], db=None, config=None, incremental=False)
    assert [inc for _, inc in seen] == [False, False]
    seen.clear()
    run_folders(["/a", "/b"], db=None, config=None, incremental=True)
    assert [inc for _, inc in seen] == [True, True]


def test_empty_folder_list_is_a_clean_no_op(monkeypatch):
    seen = []
    _stub_run_folder(monkeypatch, seen)
    report = run_folders([], db=None, config=None)
    assert seen == [] and report.results == [] and report.ok is True
    assert "no folders" in format_ingest_summary(report).lower()


def test_folder_callbacks_fire_in_order(monkeypatch):
    seen, events = [], []
    _stub_run_folder(monkeypatch, seen)
    run_folders(["/a", "/b"], db=None, config=None,
                on_folder_start=lambda i, n, f: events.append(("start", i, n, f)),
                on_folder_done=lambda r: events.append(("done", r.folder)))
    assert events == [("start", 1, 2, "/a"), ("done", "/a"),
                      ("start", 2, 2, "/b"), ("done", "/b")]


# ---- Task 5: the combined summary line ------------------------------------

def _report(**kw):
    res = kw.pop("results", None)
    if res is None:
        res = [FolderResult(folder=f"/f{i}", ok=True, files_found=100,
                            new_files=n, seconds=1.0)
               for i, n in enumerate(kw.pop("per_folder", [214]))]
    return IngestReport(results=res, seconds=kw.pop("seconds", 160.0))


def test_summary_line_reads_like_the_spec():
    """Spec line 19: '3 folders · 214 new files · 2 min 40 s'."""
    r = _report(per_folder=[100, 100, 14], seconds=160.0)
    assert format_ingest_summary(r) == "3 folders · 214 new files · 2 min 40 s"


def test_summary_line_singulars():
    r = _report(per_folder=[1], seconds=12.0)
    assert format_ingest_summary(r) == "1 folder · 1 new file · 12 s"


def test_summary_line_says_nothing_new_rather_than_zero_files():
    r = _report(per_folder=[0, 0], seconds=5.0)
    line = format_ingest_summary(r)
    assert "2 folders" in line and "no new files" in line


def test_summary_line_names_the_failed_folder():
    good = FolderResult(folder="/a", ok=True, files_found=10, new_files=4,
                        seconds=1.0)
    bad = FolderResult(folder="\\\\192.168.66.9\\Laser", ok=False,
                       error="not found — offline share?")
    line = format_ingest_summary(IngestReport(results=[good, bad], seconds=61.0))
    assert "1 of 2 folders failed" in line
    assert "\\\\192.168.66.9\\Laser" in line


def test_elapsed_wording_covers_seconds_minutes_hours():
    from laser_trim_analyzer.core.ingest_run import format_elapsed
    assert format_elapsed(0.4) == "0 s"
    assert format_elapsed(45) == "45 s"
    assert format_elapsed(60) == "1 min 0 s"
    assert format_elapsed(160) == "2 min 40 s"
    assert format_elapsed(3600) == "1 h 0 min"
    assert format_elapsed(7325) == "2 h 2 min"


# ---- Task 6: end to end on the real corpus --------------------------------

@pytest.mark.skipif(not SAMPLES.exists(), reason="sample corpus not present")
def test_run_folder_end_to_end_saves_and_then_skips(tmp_path, monkeypatch):
    """The whole pipeline, for real: three trim files land in a fresh DB, and
    a second incremental pass over the same folder processes none of them."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.parser import detect_file_type
    from laser_trim_analyzer.database import manager as db_mod
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR

    picked = []
    for p in sorted(SAMPLES.rglob("*.xls*")):
        if not p.is_file():
            continue
        try:
            if detect_file_type(p) != "trim":
                continue
        except Exception:
            continue
        picked.append(p)
        if len(picked) == 3:
            break
    if len(picked) < 3:
        pytest.skip("fewer than 3 trim samples available")

    folder = tmp_path / "laser"
    folder.mkdir()
    for p in picked:
        shutil.copy2(p, folder / p.name)

    cfg = Config()
    cfg.database.path = tmp_path / "e2e.db"
    db = DatabaseManager(cfg.database.path)
    # The Processor resolves its incremental index through the GLOBAL manager
    # (database.get_database()), not the db handed to run_folder. Left alone
    # it opens data/analysis.db — the real 3.6 GB work database — so the pin
    # is what keeps this test isolated, and what makes the second-pass skip
    # assertion mean anything at all.
    monkeypatch.setattr(db_mod, "_db_manager", db)

    first = run_folder(str(folder), db=db, config=cfg, incremental=True)
    assert first.ok is True, first.error
    assert first.files_found == 3
    assert first.new_files == 3, first.summary
    assert first.summary is not None and first.summary.errors == 0
    with db.session() as s:
        assert s.query(DBAR.id).count() == 3

    second = run_folder(str(folder), db=db, config=cfg, incremental=True)
    assert second.ok is True, second.error
    assert second.files_found == 3
    assert second.new_files == 0                  # nothing re-processed
    assert second.summary.skipped == 3
    with db.session() as s:
        assert s.query(DBAR.id).count() == 3      # and nothing re-saved


# ---- Task 7: ProcessPage drives the SAME unit ------------------------------

def test_process_page_delegates_to_the_shared_runner(make_app, tmp_path,
                                                     monkeypatch):
    """The anti-duplication test: the page must not own a second pipeline."""
    calls = {}

    def fake(folder, **kw):
        calls["folder"] = folder
        calls["incremental"] = kw.get("incremental")
        return FolderResult(folder=folder, ok=True, files_found=2, new_files=2,
                            seconds=0.5)

    monkeypatch.setattr(ingest_run, "run_folder", fake)
    app = make_app()
    page = app.page_container.get_page("process")
    page._run(str(tmp_path), incremental=False)
    assert calls == {"folder": str(tmp_path), "incremental": False}


def test_process_page_has_no_private_pipeline():
    """_discover and the coalescing moved to core/ingest_run.py; the page must
    not have grown its own copies back."""
    from laser_trim_analyzer.gui.v6.pages import process_page
    src = Path(process_page.__file__).read_text()
    assert "os.scandir" not in src
    assert "process_batch" not in src
