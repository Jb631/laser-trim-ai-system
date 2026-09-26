"""run_folder saves in batches of 20 -- counted from what committed (ingest-speed Task 10; spec 3.1,
3.9; rulings 5, 20, 22; the review of Tasks 5-8, m-3 and m-5).

The ingest's BatchWriter makes every write of a folder -- trims, final tests, smoothness, skip
markers -- in `write_batch` transactions of 20 files, or 2 s after the first file of a batch, and
always at the generator's end, on Stop and at the folder's end. A file is COUNTED only once its
batch committed, from what was stored: a failed save is an error, never a pass; a malformed file
(its own rows broke a UNIQUE constraint) is failed; a smoothness file whose identity another
content hash holds is skipped. Two failed batch commits in a row -- or two batches in a row in
which every write failed -- stop the folder, the error named, counted per folder.
"""
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import save_rows
import v5_loop


@pytest.fixture
def db(tmp_path, monkeypatch):
    from laser_trim_analyzer.database.manager import DatabaseManager
    d = DatabaseManager(tmp_path / "run.db")
    save_rows.inject(d, monkeypatch)
    yield d
    d.close()


def _count(db, table, where=""):
    con = sqlite3.connect(f"file:{db.database_path}?mode=ro", uri=True)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table} {where}").fetchone()[0]
    finally:
        con.close()


# ---- the real thing: run_folder over every kind of file --------------------------------------

def test_run_folder_stores_exactly_what_v5s_loop_stored_and_counts_what_committed(db, tmp_path):
    """run_folder -- the batch writer, every write batched -- over the whole V5 scenario (every
    kind of file the ingest meets) stores exactly the rows V5's own loop stored at f232528 (the
    golden), in whatever order its batches wrote them. Its buckets are what COMMITTED, and they
    sum to the files it processed."""
    from laser_trim_analyzer.core.ingest_run import run_folder
    v5_loop.build_v5_scenario(tmp_path)
    res = run_folder(str(tmp_path / "in"), db=db, config=None, incremental=True)
    assert res.ok, res.error
    got = v5_loop.v5_snapshot(db.database_path, tmp_path, [])["tables"]
    want = save_rows.load_golden(v5_loop.GOLDEN)["tables"]
    diffs = save_rows.differences(v5_loop.order_free(got), v5_loop.order_free(want))
    assert not diffs, "\n".join(diffs)
    assert sum(res.buckets.values()) == res.new_files, (res.buckets, res.new_files)
    # The golden's own verdicts: 5 failed (3 trims, 2 final tests), 3 warnings, 6 passed (an
    # UNTRIMMED sweep, 4 final tests, 1 smoothness file), 8 errors (4 final tests, 1 smoothness
    # file, 3 trims) -- and not one of the errors counted as a pass.
    assert res.buckets == {"failed": 5, "warnings": 3, "passed": 6, "errors": 8}, res.buckets
    assert res.new_trims == 10                # every trim row committed, its ERROR rows included


def test_a_failed_save_counts_as_an_error_never_a_pass(db, tmp_path, monkeypatch):
    """A trim whose save fails inside the batch stores nothing -- no rows, no marker, so it is new
    again next run -- and is counted as an error with the reason; the rest of the batch commits
    and counts by its verdicts."""
    from laser_trim_analyzer.core.ingest_run import run_folder
    from laser_trim_analyzer.database.manager import DatabaseManager
    folder = tmp_path / "laser"
    names = ("dlts_8232-1_242.xls", "dlts_8232-1_243.xls", "lts_8232-1_193.xls")
    for n in names:
        save_rows._pinned_copy(save_rows.FIXTURES / "trim" / n, folder / n)
    real = DatabaseManager._save_analysis_in

    def fails_for_243(self, session, analysis, stat, file_hash):
        if analysis.metadata.filename == "dlts_8232-1_243.xls":
            raise RuntimeError("invented: the save failed")
        return real(self, session, analysis, stat, file_hash)

    monkeypatch.setattr(DatabaseManager, "_save_analysis_in", fails_for_243)
    progress = _Buckets()
    res = run_folder(str(folder), db=db, config=None, incremental=True, progress=progress)
    assert res.ok, res.error
    assert res.buckets.get("errors") == 1 and sum(res.buckets.values()) == 3, res.buckets
    assert any("dlts_8232-1_243.xls: not saved -- the database refused it, not the file "
               "(RuntimeError: invented: the save failed); new again next run" in r
               for r in progress.reasons), progress.reasons
    assert res.unsaved == 1
    assert res.new_trims == 2
    assert _count(db, "analysis_results", "WHERE filename = 'dlts_8232-1_243.xls'") == 0
    assert _count(db, "processed_files", "WHERE filename = 'dlts_8232-1_243.xls'") == 0


def test_a_malformed_file_counts_as_failed_and_the_batch_line_says_so(db, tmp_path, monkeypatch,
                                                                      caplog):
    """Review m-3: a `duplicate` inside a batch is a MALFORMED file -- its own rows broke a UNIQUE
    constraint (two tracks with one track_id). Nothing of it is stored; it counts as failed, not
    skipped, and the batch line says how many."""
    from sqlalchemy.exc import IntegrityError
    from laser_trim_analyzer.core.ingest_run import run_folder
    from laser_trim_analyzer.database.manager import DatabaseManager
    folder = tmp_path / "laser"
    for n in ("dlts_8232-1_242.xls", "lts_8232-1_193.xls"):
        save_rows._pinned_copy(save_rows.FIXTURES / "trim" / n, folder / n)
    real = DatabaseManager._save_analysis_in

    def malformed_193(self, session, analysis, stat, file_hash):
        if analysis.metadata.filename == "lts_8232-1_193.xls":
            raise IntegrityError("INSERT INTO track_results", {},
                                 Exception("UNIQUE constraint failed: track_results.analysis_id, "
                                           "track_results.track_id (invented)"))
        return real(self, session, analysis, stat, file_hash)

    monkeypatch.setattr(DatabaseManager, "_save_analysis_in", malformed_193)
    progress = _Buckets()
    with caplog.at_level("INFO"):
        res = run_folder(str(folder), db=db, config=None, incremental=True, progress=progress)
    assert res.ok, res.error
    assert res.buckets.get("failed", 0) >= 1 and "skipped" not in res.buckets, res.buckets
    assert any("lts_8232-1_193.xls: malformed, not saved" in r for r in progress.reasons)
    assert _count(db, "analysis_results", "WHERE filename = 'lts_8232-1_193.xls'") == 0
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Batch phases:")
                and "walk" in r.getMessage())
    assert "1 malformed file(s) not saved" in line, line


def test_a_smoothness_file_whose_identity_another_hash_holds_is_skipped(db, tmp_path):
    """Review m-3: a smoothness file whose identity (filename, file_date, model, serial) is held
    by a row with OTHER content stores nothing -- counted as skipped, not as a pass or a fail."""
    from laser_trim_analyzer.core.ingest_run import run_folder
    name = "9991-sn4_OS_2-3-2026_10-00-00 AM.xlsx"
    v5_loop._betatronix_os(tmp_path / "first" / name, max_dev=0.0031, spec=0.0050,
                           result="PASSED")
    again = v5_loop._betatronix_os(tmp_path / "second" / name, max_dev=0.0072, spec=0.0050,
                                   result="FAILED")           # same identity, other content --
    os.utime(again, (save_rows.MTIME + 3600, save_rows.MTIME + 3600))   # a re-export, later
    first = run_folder(str(tmp_path / "first"), db=db, config=None, incremental=True)
    second = run_folder(str(tmp_path / "second"), db=db, config=None, incremental=True)
    assert first.ok and second.ok
    assert first.buckets == {"passed": 1}, first.buckets
    assert second.buckets == {"skipped": 1}, second.buckets
    assert _count(db, "smoothness_results") == 1


# ---- the flush policy (ruling 5) --------------------------------------------------------------

class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class _RecordingDb:
    """A database whose write_batch saves everything, recording each batch it was handed."""

    def __init__(self):
        self.batches = []

    def write_batch(self, items):
        from laser_trim_analyzer.database.manager import WriteOutcome
        self.batches.append(list(items))
        return [WriteOutcome("saved", row_id=n + 1) for n, _ in enumerate(items)]


def _trim_outcome(i):
    from laser_trim_analyzer.core.models import AnalysisStatus
    from laser_trim_analyzer.core.processor import Outcome
    result = SimpleNamespace(file_type="trim", overall_status=AnalysisStatus.PASS,
                             metadata=SimpleNamespace(model="9990", filename=f"f{i}.xls"))
    return Outcome(path=f"/nowhere/f{i}.xls", result=result, stat=(1, 0.0),
                   file_hash=f"invented-{i}")


def _writer(db, clock=None, settled=None):
    from laser_trim_analyzer.core.ingest_run import BatchWriter
    settled = [] if settled is None else settled
    return BatchWriter(SimpleNamespace(), db, settled.append, clock=clock or _Clock()), settled


def test_a_batch_is_twenty_files():
    """Twenty files make one write_batch -- one transaction -- and the 21st waits for the next."""
    db = _RecordingDb()
    writer, settled = _writer(db)
    for i in range(21):
        writer.add(_trim_outcome(i))
    assert [len(b) for b in db.batches] == [20] and len(settled) == 20
    writer.flush()
    assert [len(b) for b in db.batches] == [20, 1] and len(settled) == 21
    writer.flush()                                            # nothing held: no empty batch
    assert len(db.batches) == 2


def test_a_batch_is_also_two_seconds_after_its_first_file():
    """A slow stream commits every 2 s: at the add that finds the batch due, or at the tick the
    consumer makes while the pool is quiet -- never before."""
    db, clock = _RecordingDb(), _Clock()
    writer, settled = _writer(db, clock)
    writer.add(_trim_outcome(0))
    clock.now += 1.9
    writer.tick()
    assert db.batches == []
    clock.now += 0.1                                          # 2.0 s after the first file
    writer.tick()
    assert [len(b) for b in db.batches] == [1]
    writer.add(_trim_outcome(1))
    clock.now += 2.5
    writer.add(_trim_outcome(2))                              # due: this add commits both
    assert [len(b) for b in db.batches] == [1, 2] and len(settled) == 3


def test_the_pool_ticks_the_writer_while_it_waits(tmp_path):
    """The consumer waits on the pool at most a second at a time (spec 3.1): a file that takes
    2.2 s to analyse still lets the writer commit what it holds on time."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.processor import Outcome, Processor

    class Slow(Processor):
        def analyse_path(self, file_path, disk_stat=None):
            if Path(file_path).name == "slow.xls":
                time.sleep(2.2)
            return Outcome(path=str(file_path))               # nothing to write

    class Ticks:
        def __init__(self):
            self.ticks, self.flushes = 0, 0

        def add(self, outcome):
            return outcome.result

        def tick(self):
            self.ticks += 1

        def flush(self):
            self.flushes += 1

    config = Config()
    config.processing.turbo_mode_threshold = 1                # the pool path
    proc, writer = Slow(config=config, use_ml=False), Ticks()
    list(proc.process_batch([Path("/nowhere/fast.xls"), Path("/nowhere/slow.xls")],
                            incremental=False, writer=writer))
    assert writer.ticks >= 2, writer.ticks
    assert writer.flushes == 1                                # the generator's end


def test_stop_lands_on_a_chunk_boundary_and_what_was_analysed_is_committed(db, tmp_path):
    """Stop keeps its contract (ruling 20): the in-flight chunk of 20 finishes, and everything it
    analysed is COMMITTED -- at the generator's end -- before the folder returns."""
    from threading import Event
    from laser_trim_analyzer.core.ingest_run import ProgressCoalescer, run_folder
    folder = tmp_path / "laser"
    for i in range(45):
        save_rows._pinned_copy(save_rows.FIXTURES / "trim" / "dlts_8232-1_242.xls",
                               folder / f"dlts_8232-1_{900 + i}.xls")
    cancel = Event()

    class StopAfterOne(ProgressCoalescer):
        def note(self, status):
            super().note(status)
            if status.status == "completed":
                cancel.set()

    from laser_trim_analyzer.config import Config
    config = Config()
    config.processing.turbo_mode_threshold = 1                # the chunked pool path
    res = run_folder(str(folder), db=db, config=config, incremental=True,
                     progress=StopAfterOne(), cancel=cancel)
    assert res.ok and res.cancelled
    assert res.new_trims == 20 == _count(db, "analysis_results"), res.new_trims
    assert sum(res.buckets.values()) == res.new_files == 20, (res.buckets, res.new_files)


def test_the_folders_end_commits_what_a_processor_left_unflushed(tmp_path, monkeypatch):
    """A processor that never flushes its writer (a stand-in, or a generator that ended early)
    loses nothing: run_folder commits what the writer holds at the folder's end."""
    from laser_trim_analyzer.core import ingest_run
    (tmp_path / "a.xls").write_bytes(b"junk")

    class Proc:
        last_scan_stats = {}

        def __init__(self, *a, **k):
            pass

        def process_batch(self, *a, **k):
            for i in range(3):
                yield k["writer"].add(_trim_outcome(i))
            return SimpleNamespace(processed=3)

    db = _RecordingDb()
    monkeypatch.setattr(ingest_run, "Processor", Proc)
    monkeypatch.setattr(ingest_run, "_post_batch", lambda *a, **k: None)
    res = ingest_run.run_folder(str(tmp_path), db=db, config=None)
    assert res.ok and [len(b) for b in db.batches] == [3] and res.new_trims == 3


# ---- the stop rules (ruling 22, m-5) -----------------------------------------------------------

class _FailingDb(_RecordingDb):
    """write_batch as `plan` says, one entry per call: "commit-fails" (BatchCommitError, the
    whole batch rolled back), "all-fail" (committed, every item failed), or "ok"."""

    def __init__(self, plan):
        super().__init__()
        self.plan = list(plan)
        self.manager_consecutive = 0

    def write_batch(self, items):
        from laser_trim_analyzer.database.manager import BatchCommitError, WriteOutcome
        self.batches.append(list(items))
        step = self.plan.pop(0) if self.plan else "ok"
        if step == "commit-fails":
            self.manager_consecutive += 1
            cause = RuntimeError("invented: database is locked")
            raise BatchCommitError(cause, [WriteOutcome("failed", reason="not committed")
                                           for _ in items], self.manager_consecutive)
        self.manager_consecutive = 0
        if step == "all-fail":
            return [WriteOutcome("failed", reason="OperationalError: invented: no such column")
                    for _ in items]
        return [WriteOutcome("saved", row_id=n + 1) for n, _ in enumerate(items)]


def test_two_failed_batch_commits_in_a_row_stop_the_folder_with_the_error_named():
    """Ruling 22. One failed commit is survivable (its files are new next run, counted as
    errors); the second in a row ends the folder, naming the cause. A commit between resets."""
    from laser_trim_analyzer.core.processor import WriterStop
    writer, settled = _writer(_FailingDb(["commit-fails", "ok", "commit-fails", "commit-fails"]))
    for i in range(3):
        writer.add(_trim_outcome(i))
        writer.flush() if i < 2 else None
    with pytest.raises(WriterStop) as stop:
        writer.flush()
        writer.add(_trim_outcome(9))
        writer.flush()
    assert "2 batch commits in a row failed" in str(stop.value)
    assert "invented: database is locked" in str(stop.value)
    assert [c.bucket for c in settled] == ["errors", "passed", "errors", "errors"], settled


def test_two_batches_in_a_row_that_stored_nothing_stop_the_folder():
    """Review m-5: batches that COMMIT with every write in them failed are a systemic per-file
    failure; two in a row stop the folder instead of parsing 60,000 files into nothing."""
    from laser_trim_analyzer.core.processor import WriterStop
    writer, settled = _writer(_FailingDb(["all-fail", "ok", "all-fail", "all-fail"]))
    writer.add(_trim_outcome(0))
    writer.flush()
    writer.add(_trim_outcome(1))
    writer.flush()                                            # a batch that stored: resets
    writer.add(_trim_outcome(2))
    writer.flush()
    writer.add(_trim_outcome(3))
    with pytest.raises(WriterStop) as stop:
        writer.flush()
    assert "2 batches in a row stored nothing" in str(stop.value)
    assert "invented: no such column" in str(stop.value)
    assert [c.bucket for c in settled] == ["errors", "passed", "errors", "errors"]


def test_the_stop_rules_count_per_folder_not_per_manager(tmp_path, monkeypatch):
    """Review m-5: a folder is stopped by ITS OWN failures. One failed commit in each of two
    folders stops neither -- although the manager's own count reaches two."""
    from laser_trim_analyzer.core import ingest_run
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "x.xls").write_bytes(b"junk")

    class Proc:
        last_scan_stats = {}

        def __init__(self, *a, **k):
            pass

        def process_batch(self, *a, **k):
            yield k["writer"].add(_trim_outcome(0))
            return SimpleNamespace(processed=1)

    db = _FailingDb(["commit-fails", "commit-fails"])
    monkeypatch.setattr(ingest_run, "Processor", Proc)
    monkeypatch.setattr(ingest_run, "_post_batch", lambda *a, **k: None)
    report = ingest_run.run_folders([str(tmp_path / "a"), str(tmp_path / "b")], db=db,
                                    config=None, incremental=False)
    assert db.manager_consecutive == 2
    assert [r.ok for r in report.results] == [True, True], [r.error for r in report.results]
    assert [r.buckets for r in report.results] == [{"errors": 1}, {"errors": 1}]


def test_run_folder_names_the_error_that_stopped_it(tmp_path, monkeypatch):
    """The stop reaches the folder's result and the screen, the error named; nothing it could not
    commit is counted as a result."""
    from laser_trim_analyzer.core import ingest_run
    (tmp_path / "a.xls").write_bytes(b"junk")

    class Proc:
        last_scan_stats = {}

        def __init__(self, *a, **k):
            pass

        def process_batch(self, *a, **k):
            for i in range(45):
                yield k["writer"].add(_trim_outcome(i))
            return SimpleNamespace(processed=45)

    said = []
    monkeypatch.setattr(ingest_run, "Processor", Proc)
    monkeypatch.setattr(ingest_run, "_post_batch", lambda *a, **k: None)
    res = ingest_run.run_folder(str(tmp_path), db=_FailingDb(["commit-fails", "commit-fails"]),
                                config=None, on_phase=said.append)
    assert res.ok is False and "2 batch commits in a row failed" in res.error, res.error
    assert res.buckets == {"errors": 40} and res.new_trims == 0, res.buckets
    assert any(s.startswith("Stopped: 2 batch commits in a row failed") for s in said), said


class _Buckets:
    """A progress sink that keeps the bucket reasons (ProgressCoalescer drains them away)."""

    def __init__(self):
        from laser_trim_analyzer.core.ingest_run import ProgressCoalescer
        self._inner = ProgressCoalescer()
        self.reasons = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def bucket(self, name, reason=""):
        if reason:
            self.reasons.append(reason)
        self._inner.bucket(name, reason)


def test_an_analysis_that_raises_in_the_pool_is_an_error_and_is_still_counted():
    """A file whose analysis itself raised in the pool has nothing to save -- and is still handed
    to the writer, as an error with its reason, so the buckets sum to what was processed."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.processor import Processor

    from laser_trim_analyzer.core.processor import Outcome

    class Explodes(Processor):
        def analyse_path(self, file_path, disk_stat=None):
            if Path(file_path).name == "boom.xls":
                raise RuntimeError("invented: the analysis exploded")
            result = self._create_error_result(self._create_minimal_metadata(Path(file_path)),
                                               "invented: an ordinary unreadable file", 0.0)
            return Outcome(path=str(file_path), result=result, stat=(1, 0.0), file_hash="h")

    config = Config()
    config.processing.turbo_mode_threshold = 1                # the pool path
    db = _RecordingDb()
    writer, settled = _writer(db)
    gen = Explodes(config=config, use_ml=False).process_batch(
        [Path("/nowhere/f0.xls"), Path("/nowhere/boom.xls"), Path("/nowhere/f1.xls")],
        incremental=False, writer=writer)
    yielded = []
    try:
        while True:
            yielded.append(next(gen))
    except StopIteration as stop:
        summary = stop.value
    assert len(yielded) == 2 and summary.processed == 3 and summary.errors == 3
    assert [c.bucket for c in settled] == ["errors"] * 3 == ["errors"] * summary.processed
    (boom,) = [c for c in settled if "invented: the analysis exploded" in c.reason]
    assert boom.path.endswith("boom.xls") and not boom.saved
    assert sum(1 for b in db.batches for _ in b) == 2          # nothing written for it


@pytest.mark.parametrize("pool", [True, False], ids=["the pool", "sequential"])
def test_a_writer_stop_ends_the_batch_in_either_path(pool):
    """The pool's per-file error handling must never swallow the writer's stop: two failed batch
    commits end process_batch, pool or sequential, and no file after them is analysed."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.core.processor import Outcome, Processor, WriterStop
    analysed = []

    class Stub(Processor):
        def analyse_path(self, file_path, disk_stat=None):
            analysed.append(Path(file_path).name)
            result = self._create_error_result(self._create_minimal_metadata(Path(file_path)),
                                               "invented: an ordinary unreadable file", 0.0)
            return Outcome(path=str(file_path), result=result, stat=(1, 0.0), file_hash="h")

    config = Config()
    config.processing.turbo_mode_threshold = 1 if pool else 10 ** 9
    writer, settled = _writer(_FailingDb(["commit-fails", "commit-fails"]))
    paths = [Path(f"/nowhere/f{i:03d}.xls") for i in range(100)]
    with pytest.raises(WriterStop):
        list(Stub(config=config, use_ml=False).process_batch(paths, incremental=False,
                                                              writer=writer))
    assert len(settled) == 40 and len(analysed) <= 60, (len(settled), len(analysed))


# ---- the review of Tasks 9-10: I-1 (the finished run's tally) --------------------------------

def test_the_process_pages_finished_tally_counts_what_committed(tk_root, db, tmp_path):
    """Review I-1 (spec 3.9). When a run finishes, the V6 Process page repaints its counters from
    `result.summary`. That tally now counts what COMMITTED, exactly as the live counters do: the
    refused save of 8434ct-1118D.xls ("Serial cannot be empty") is an error, not the fail its
    analysis reached. Driven through the real widget, over the V5 scenario."""
    from laser_trim_analyzer.core.ingest_run import ProgressCoalescer, run_folder
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.process_progress_section import (
        ProcessProgressSection)
    v5_loop.build_v5_scenario(tmp_path)
    progress = ProgressCoalescer()
    res = run_folder(str(tmp_path / "in"), db=db, config=None, incremental=True,
                     progress=progress)
    assert res.ok, res.error
    section = ProcessProgressSection(tk_root, theme=ThemeManager())
    snap = progress.drain()                                   # what the page's _paint hands it
    section.add_counts(snap["counts"], snap["reasons"])
    live = {k: lbl.cget("text") for k, lbl in section._labels.items()}
    section.set_final(res.summary)                            # the page's finished-run path
    final = {k: lbl.cget("text") for k, lbl in section._labels.items()}
    assert section._status.cget("text") == (
        "Complete: 5 passed, 3 warnings, 5 failed, 3 skipped, 8 errors."), section._status.cget(
        "text")
    assert (final["failed"], final["errors"]) == (live["failed"], live["errors"]) == (
        "Failed: 5", "Errors: 8"), (live, final)
    s = res.summary                                           # BatchSummary keeps UNTRIMMED apart
    assert (s.passed + s.untrimmed, s.warnings, s.failed, s.errors) == (
        res.buckets["passed"], res.buckets["warnings"], res.buckets["failed"],
        res.buckets["errors"]), (s, res.buckets)


# ---- the review of Tasks 9-10: I-2 (m-5 and the ruling of 2026-09-25) --------------------------

def _rout_files(folder: Path, n: int) -> None:
    """n synthetic Format 2 final tests (Rout_), each its own unit and its own bytes (invented)."""
    from test_ft_polyfit_degenerate import clean_series, write_format2
    folder.mkdir(parents=True, exist_ok=True)
    measured, positions = clean_series()
    for i in range(n):
        write_format2(folder / f"Rout_9990_sn{100 + i}_vo.xlsx",
                      [m + i * 1e-6 for m in measured], positions)


def _refused_by_the_database(*a, **k):
    from sqlalchemy.exc import OperationalError
    raise OperationalError("INSERT INTO invented", {}, Exception("no such column: invented_column"))


@pytest.fixture
def by_count(monkeypatch):
    """Batches by count alone: the 2-second rule pinned elsewhere cannot split them here."""
    from laser_trim_analyzer.core.ingest_run import BatchWriter
    monkeypatch.setattr(BatchWriter, "FLUSH_SECONDS", 10 ** 9)


def test_a_final_test_folder_the_database_refuses_stops_and_marks_nothing(db, tmp_path,
                                                                         monkeypatch, by_count):
    """Review I-2, the ruling of 2026-09-25: 60 final tests whose saves fail on a schema error.
    The folder STOPS after two batches, the error named; NOT ONE file is recorded as unreadable
    -- the database's error is not the files' -- Home says they were not saved, and the next
    healthy run processes all 60."""
    from laser_trim_analyzer.core.ingest_run import IngestReport, format_ingest_summary, run_folder
    from laser_trim_analyzer.database.manager import DatabaseManager
    folder = tmp_path / "Test Station"
    _rout_files(folder, 60)
    real = DatabaseManager._save_final_test_in
    monkeypatch.setattr(DatabaseManager, "_save_final_test_in", _refused_by_the_database)
    res = run_folder(str(folder), db=db, config=None, incremental=True)
    assert res.ok is False and "2 batches in a row stored nothing" in res.error, res.error
    assert "no such column: invented_column" in res.error
    assert _count(db, "processed_files") == 0, "a database error marked files unreadable"
    assert _count(db, "final_test_results") == 0
    assert res.buckets == {"errors": 40} and res.new_files == 40 == res.unsaved, res
    assert "40 files not saved — new again next run" in format_ingest_summary(
        IngestReport(results=[res]))
    monkeypatch.setattr(DatabaseManager, "_save_final_test_in", real)
    again = run_folder(str(folder), db=db, config=None, incremental=True)
    assert again.ok and again.new_files == 60 and _count(db, "final_test_results") == 60, again


def test_a_trim_folder_with_non_test_files_in_every_batch_stops_on_a_database_refusal(
        db, tmp_path, monkeypatch, by_count):
    """Review I-2: a marker is not a save. A trim folder in which one file in three is not test
    data (their markers commit fine) and whose trim saves the database refuses still stops after
    two batches -- the markers neither count toward the rule nor reset it."""
    from laser_trim_analyzer.core.ingest_run import run_folder
    from laser_trim_analyzer.database.manager import DatabaseManager
    folder = tmp_path / "laser"
    for i in range(40):
        save_rows._pinned_copy(save_rows.FIXTURES / "trim" / "dlts_8232-1_242.xls",
                               folder / f"dlts_8232-1_{900 + i}.xls")
    for i in range(20):
        (folder / f"9993_noise_capture_{i}.xls").write_bytes(
            f"invented: an oscilloscope capture {i}".encode())
    monkeypatch.setattr(DatabaseManager, "_save_analysis_in", _refused_by_the_database)
    res = run_folder(str(folder), db=db, config=None, incremental=True)
    assert res.ok is False and "2 batches in a row stored nothing" in res.error, res.error
    assert _count(db, "analysis_results") == 0
    assert res.new_files == sum(res.buckets.values()) == res.buckets["errors"] == res.unsaved


def test_a_trim_folder_the_database_refuses_stops_as_before(db, tmp_path, monkeypatch, by_count):
    """The control: trims alone, refused by the database -- stopped after two batches, nothing
    stored, and the next healthy run takes every file."""
    from laser_trim_analyzer.core.ingest_run import run_folder
    from laser_trim_analyzer.database.manager import DatabaseManager
    folder = tmp_path / "laser"
    for i in range(60):
        save_rows._pinned_copy(save_rows.FIXTURES / "trim" / "dlts_8232-1_242.xls",
                               folder / f"dlts_8232-1_{900 + i}.xls")
    real = DatabaseManager._save_analysis_in
    monkeypatch.setattr(DatabaseManager, "_save_analysis_in", _refused_by_the_database)
    res = run_folder(str(folder), db=db, config=None, incremental=True)
    assert res.ok is False and "2 batches in a row stored nothing" in res.error, res.error
    assert _count(db, "analysis_results") == 0 and _count(db, "processed_files") == 0
    assert res.buckets == {"errors": 40} and res.new_files == 40 == res.unsaved
    monkeypatch.setattr(DatabaseManager, "_save_analysis_in", real)
    again = run_folder(str(folder), db=db, config=None, incremental=True)
    assert again.ok and again.new_files == 60 and _count(db, "analysis_results") == 60


def test_one_content_refusal_is_marked_with_its_reason(db, tmp_path, monkeypatch):
    """The ruling's other half: a save the file's OWN content refuses records it as unreadable,
    with its reason; the files around it are saved as ever."""
    from laser_trim_analyzer.core.ingest_run import run_folder
    from laser_trim_analyzer.database.manager import DatabaseManager
    folder = tmp_path / "Test Station"
    _rout_files(folder, 5)
    real = DatabaseManager._save_final_test_in

    def refuses_sn102(self, session, metadata, *a, **k):
        if metadata.get("serial") == "102":
            raise ValueError("invented: the tracks disagree")
        return real(self, session, metadata, *a, **k)

    monkeypatch.setattr(DatabaseManager, "_save_final_test_in", refuses_sn102)
    res = run_folder(str(folder), db=db, config=None, incremental=True)
    assert res.ok and res.buckets.get("errors") == 1 and res.unsaved == 0, res
    con = sqlite3.connect(f"file:{db.database_path}?mode=ro", uri=True)
    try:
        markers = con.execute("SELECT filename, error_message FROM processed_files").fetchall()
    finally:
        con.close()
    assert len(markers) == 1 and markers[0][0] == "Rout_9990_sn102_vo.xlsx", markers
    assert markers[0][1].startswith("unreadable: ValueError: invented: the tracks disagree")
    assert _count(db, "final_test_results") == 4


def test_content_refusals_are_the_files_own_and_never_stop_the_folder(db, tmp_path, monkeypatch,
                                                                     by_count):
    """A folder whose every save its files' content refuses (a folder of old files with no
    serial, say) is not a failing database: each file is recorded with its reason, and the folder
    runs to its end instead of stopping every 40 files, run after run."""
    from laser_trim_analyzer.core.ingest_run import run_folder
    from laser_trim_analyzer.database.manager import DatabaseManager
    folder = tmp_path / "Test Station"
    _rout_files(folder, 45)

    def refuses_all(self, *a, **k):
        raise ValueError("invented: the tracks disagree")

    monkeypatch.setattr(DatabaseManager, "_save_final_test_in", refuses_all)
    res = run_folder(str(folder), db=db, config=None, incremental=True)
    assert res.ok, res.error
    assert res.buckets == {"errors": 45} and res.unsaved == 0
    assert _count(db, "processed_files", "WHERE error_message LIKE 'unreadable: ValueError%'") == 45


# ---- trained models present (2026-09-25: the path no golden had covered) -----------------------

def _invented_trained_models_for_8232_1(db) -> None:
    """A deployed composite trim-risk model, a trained failure predictor and a sigma threshold for
    8232-1 -- all invented, trained on invented data -- where the app keeps them: the app
    directory's data/ml_models (tests/conftest.py redirects the app directory to tmp)."""
    import random
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from laser_trim_analyzer.config import ml_models_directory
    from laser_trim_analyzer.database.models import ModelMLState
    from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel, CompositeTrainingResult
    from laser_trim_analyzer.ml.predictor import FEATURE_COLUMNS, ModelPredictor
    folder = ml_models_directory()
    random.seed(11)
    crm = CompositeRiskModel("8232-1")
    crm.features_used = ["untrimmed_error_max", "trim_pass_count"]
    X = np.array([[random.random(), random.randint(1, 6)] for _ in range(80)])
    y = np.array([1 if a + 0.1 * b > 0.9 else 0 for a, b in X])
    crm._pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=1000)).fit(X, y)
    crm._feat_median = {"untrimmed_error_max": 0.5, "trim_pass_count": 3.0}
    crm.is_trained = True
    crm.result = CompositeTrainingResult("8232-1", 80, int(y.sum()), crm.features_used, 0.8, 0.7,
                                         0.5, deployed=True, reason="invented")
    crm.save(folder / "composite_risk" / "8232-1.pkl")
    predictor = ModelPredictor("8232-1")
    Xp = pd.DataFrame([{c: random.random() + (0.4 if i % 3 == 0 else 0.0) for c in FEATURE_COLUMNS}
                       for i in range(60)])
    predictor.train(Xp, pd.Series([1 if i % 3 == 0 else 0 for i in range(60)]))
    predictor.classifier.n_jobs = 1      # one thread (the trainer's own default since Task 11)
    assert predictor.save(folder / "predictors" / "8232-1.pkl")
    with db.session() as s:
        s.add(ModelMLState(model="8232-1", is_trained=True, sigma_threshold=0.0042))


def test_with_trained_models_present_the_batch_stores_what_the_per_file_way_stores(
        tmp_path, monkeypatch):
    """Trained models PRESENT: a composite trim-risk model, a failure predictor and a sigma
    threshold. run_folder -- the snapshot's ML state, every write batched -- stores exactly what
    the old per-file way stores (process_file + save_analysis, the ML state from the database), the
    composite_trim_risk_score, failure_probability and sigma_threshold of every track included, and
    the models were in force."""
    from laser_trim_analyzer.core.ingest_run import run_folder
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml import invalidate_shared_ml_manager
    from laser_trim_analyzer.ml.predictor import ModelPredictor
    predicted = {"batched": 0, "per_file": 0}
    side = {"now": None}
    real_predict = ModelPredictor.predict_failure_probability

    def counted(self, features):          # the formula also fills failure_probability, so COUNT
        predicted[side["now"]] += 1
        return real_predict(self, features)

    monkeypatch.setattr(ModelPredictor, "predict_failure_probability", counted)
    folder = tmp_path / "laser"
    for n in ("dlts_8232-1_242.xls", "dlts_8232-1_243.xls", "lts_8232-1_193.xls",
              "lts_8232-1_194.xls", "dlts_8074_18.xls"):
        save_rows._pinned_copy(save_rows.FIXTURES / "trim" / n, folder / n)
    batched, per_file = (DatabaseManager(tmp_path / f"{n}.db") for n in ("batched", "per_file"))
    try:
        _invented_trained_models_for_8232_1(batched)
        with per_file.session() as s:
            from laser_trim_analyzer.database.models import ModelMLState
            s.add(ModelMLState(model="8232-1", is_trained=True, sigma_threshold=0.0042))
        invalidate_shared_ml_manager()
        save_rows.inject(batched, monkeypatch)
        side["now"] = "batched"
        res = run_folder(str(folder), db=batched, config=None, incremental=True)
        assert res.ok, res.error
        invalidate_shared_ml_manager()
        save_rows.inject(per_file, monkeypatch)
        side["now"] = "per_file"
        proc = Processor(use_ml=True)
        for f in sorted(folder.iterdir()):
            per_file.save_analysis(proc.process_file(f))
    finally:
        invalidate_shared_ml_manager()
        batched.close()
        per_file.close()
    got = save_rows.dump_rows(tmp_path / "batched.db", tmp_path)
    want = save_rows.dump_rows(tmp_path / "per_file.db", tmp_path)
    diffs = save_rows.differences(v5_loop.order_free(got), v5_loop.order_free(want), exact=True)
    assert not diffs, "\n".join(diffs)
    tracks = [t for t in got["track_results"]]
    scored = [t for t in tracks if t["composite_trim_risk_score"] is not None]
    assert len(scored) == 4, "the composite model scored every 8232-1 track, and no other"
    assert predicted == {"batched": 4, "per_file": 4}, predicted   # the predictor ran, on both
    assert sum(t["sigma_threshold"] == 0.0042 for t in tracks) == 4, "the ML threshold ruled"
