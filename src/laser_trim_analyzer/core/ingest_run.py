"""The ingest run — ONE folder-processing pipeline, driven by every page.

This is ProcessPage._run, lifted out whole. Home's "Process everything new"
and the Process page's folder picker are two front ends onto the same worker
(spec 2026-08-29-app-shape-investigate-design.md §1: "same worker, no
duplicate pipeline"), and the only way to keep them honest is for there to be
exactly one implementation to call.

Lives in `core/` because everything here is domain work — walk a tree, run
`Processor.process_batch`, persist the trims, re-link final tests, advance the
drift detectors — with no widget in sight. Nothing in this module imports
tkinter or customtkinter, so the QA sweep and the tests exercise the real
pipeline headlessly, and a worker thread running it cannot violate the
"workers never call Tk" rule by accident. Pages communicate with it through
plain callables (`on_phase`, `on_folder_done`) and the `ProgressCoalescer`,
and are responsible for marshalling anything they receive back onto the Tk
thread.

The hard-won behaviours preserved from the Process page, each of which cost a
work incident to learn:

  * The walk is a parallel BFS over `scandir`, capturing (size, mtime) from
    the listing itself — one network round trip instead of one stat() per file
    (2026-07-10: 73 minutes for 170k files).
  * Per-file progress is COALESCED. One UI post per file made the whole app
    sluggish for the length of a 170k-file batch (2026-07-13); workers only
    bump in-memory counters, and the page paints a snapshot a few times a
    second.
  * A batch that raises must SAY SO. An exception here once killed the worker
    thread silently, leaving Start disabled and the app looking hung
    (2026-07-09).
  * A folder that cannot be read is an ERROR with a reason, not an empty
    folder. Offline shares are routine, and "0 new files" is the same thing a
    healthy, already-ingested folder says.
"""
import functools
import logging
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from laser_trim_analyzer.config import ingest_folder_problem
from laser_trim_analyzer.core.models import AnalysisStatus, ProcessingStatus
from laser_trim_analyzer.core.processor import Processor

logger = logging.getLogger(__name__)

BUCKETS = ("passed", "warnings", "failed", "skipped", "errors")
MAX_REASONS = 10          # what one repaint can usefully show
TICK_SECONDS = 0.25       # 4 Hz: responsive, and nowhere near saturating Tk

# CPython hands the GIL to a waiting thread only every `switchinterval`
# seconds, and 5 ms (the default) is an eternity to a Tk repaint that needs
# the lock dozens of times per frame. This is why the app "kept freezing" on
# the new laptop's first full ingest: the ingest was never stuck, the window
# just could not get the lock. Measured with the real V6 window open,
# run_folder over 1,308 files into a fresh database, a 20 ms heartbeat on the
# Tk thread — median of three runs each:
#
#   default (5 ms):  283 UI stalls > 60 ms, 48% of the run frozen, worst 439 ms
#   0.5 ms:           77 UI stalls > 60 ms, 12% frozen,            worst 142 ms
#
# with ingest wall clock 66 s -> 68 s. Sampled stacks put the blame on
# Processor._process_parallel's Excel parsers: up to four CPU-bound threads,
# and the UI lost every race to them by about one switch interval. Capping
# that pool to 2 was measured too and bought nothing, while costing read
# parallelism against the network share — so the pool is left alone and only
# the interval moves.
#
# It is a trade, not a free win: more switching means a LONG main-thread job
# that overlaps a batch gets preempted more and finishes slower in wall clock.
# A chart paint forced to overlap the start of a batch went 555 ms -> 1030 ms
# in the same probe. That is the right way round — a window that answers every
# 140 ms beats one that vanishes for 400 ms at a time — but it is the cost.
INGEST_SWITCH_INTERVAL = 0.0005

_switch_lock = threading.Lock()
_switch_depth = 0                        # ingests currently inside the guard
_switch_saved: Optional[float] = None    # the caller's interval, to give back


def _with_ingest_switch_interval(fn):
    """Hold a UI-friendly switch interval for as long as `fn` runs.

    `sys.setswitchinterval` is PROCESS-WIDE, which is the entire risk here, so
    the scope is deliberately the ingest and nothing else: an idle app keeps
    the interpreter default, and whatever the caller had set comes back in
    `finally` — on a failure result, and on an exception, not just the happy
    path. The cost of the fast interval is more context switches, which is a
    fine trade while a batch is running and a pointless one while it is not.

    Runs are COUNTED rather than each minding its own, because they nest and
    overlap: `run_folders` calls `run_folder`, and Home's "process everything
    new" and the Process page's folder run each disable only their own button,
    so two ingests can be in flight on two worker threads. A run that saved
    the interval for itself would save the other run's already-fast 0.5 ms as
    "the caller's" and hand that back at the end — leaving the whole process
    fast forever — and would restore mid-flight while the other run is still
    parsing. Only the outermost run in flight touches the interval, and the
    lock is what keeps that count honest across threads.
    """
    @functools.wraps(fn)
    def guarded(*args, **kwargs):
        global _switch_depth, _switch_saved
        with _switch_lock:
            if _switch_depth == 0:
                _switch_saved = sys.getswitchinterval()
                sys.setswitchinterval(INGEST_SWITCH_INTERVAL)
            _switch_depth += 1
        try:
            return fn(*args, **kwargs)
        finally:
            with _switch_lock:
                _switch_depth -= 1
                if _switch_depth == 0:
                    sys.setswitchinterval(_switch_saved)
                    _switch_saved = None
    return guarded


def bucket_for_status(status: AnalysisStatus) -> str:
    """Which progress counter a finished file lands in.

    C1: there is no AnalysisStatus.SKIPPED — a skip is a ProcessingStatus, not
    a result — and UNTRIMMED is a valid outcome (a test sweep with no trim
    run), so it counts as processed, never as a failure.
    """
    return {AnalysisStatus.PASS: "passed", AnalysisStatus.WARNING: "warnings",
            AnalysisStatus.FAIL: "failed", AnalysisStatus.ERROR: "errors",
            AnalysisStatus.UNTRIMMED: "passed"}.get(status, "passed")


def discover_excel_files(folder: str) -> Tuple[List[str], Dict[str, tuple]]:
    """Walk the tree AND capture (size, mtime) from the directory listings.

    On Windows/SMB, scandir returns each entry's stat data with the listing
    itself — the same network round trip. Passing it to the processor makes
    the incremental check pure in-memory comparison (V4-era seconds) instead
    of one stat() round trip per file.

    Parallel BFS, 8 workers: over SMB the cost of a listing is round-trip
    LATENCY, not local work, so overlapping 8 of them is ~8x on a deep tree.
    Each worker only READS one directory and RETURNS what it found; the
    results are merged here, on one thread, so no lock is needed and no shared
    structure can be torn. Unreadable entries/folders are skipped — a
    permissions hiccup must not end the walk.
    """
    def scan_one(d):
        subdirs, found = [], []
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            subdirs.append(entry.path)
                        elif entry.name.lower().endswith((".xls", ".xlsx")):
                            st = entry.stat()
                            found.append((entry.path, (st.st_size, st.st_mtime)))
                    except OSError:
                        continue   # unreadable entry: skip, don't die
        except OSError:
            logger.warning("Could not list folder: %s", d)
        return subdirs, found

    out: List[str] = []
    stats: dict = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = {pool.submit(scan_one, folder)}
        while pending:      # ends when no directory is left in flight
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                subdirs, found = fut.result()
                for path, st in found:
                    out.append(path)
                    stats[path] = st
                pending |= {pool.submit(scan_one, d) for d in subdirs}
    return out, stats


class ProgressCoalescer:
    """Thread-safe in-memory progress accumulator. No Tk, no callbacks.

    Workers `note()`/`bucket()` freely; the UI `drain()`s a few times a second
    and paints ONE snapshot. `done` and `file` are running values (a counter
    that resets between paints reads as a bug); counts and reasons are DELTAS
    since the last drain, because the widgets they feed accumulate their own.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._done = 0
        self._file = ""
        self._scan_msg: Optional[str] = None
        self._moved = False
        self._counts = {k: 0 for k in BUCKETS}
        self._reasons: deque = deque(maxlen=MAX_REASONS)

    # ---- worker side ----
    def note(self, status: ProcessingStatus) -> None:
        with self._lock:
            if status.status == "scanning":
                # "Found N new files (M already in database)" — the headline
                # the user waits for. Not progress: it precedes it.
                self._scan_msg = status.message or "Scanning…"
                return
            if status.status in ("completed", "skipped", "failed"):
                self._done += 1
                self._file = status.filename or self._file
            if status.status == "skipped":
                self._counts["skipped"] += 1

    def bucket(self, name: str, reason: str = "") -> None:
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + 1
            self._moved = True
            if reason and name in ("failed", "errors"):
                self._reasons.append(reason)

    # ---- UI side ----
    def drain(self) -> dict:
        with self._lock:
            scan_msg, self._scan_msg = self._scan_msg, None
            counts = {k: v for k, v in self._counts.items() if v}
            reasons = list(self._reasons)
            moved = self._moved or bool(counts) or bool(reasons)
            self._counts = {k: 0 for k in BUCKETS}
            self._reasons.clear()
            self._moved = False
            return {"scan_msg": scan_msg, "done": self._done, "file": self._file,
                    "counts": counts, "reasons": reasons, "moved": moved}

    def reset(self) -> None:
        with self._lock:
            self._done = 0
            self._file = ""
            self._scan_msg = None
            self._moved = False
            self._counts = {k: 0 for k in BUCKETS}
            self._reasons.clear()


class ProgressTicker:
    """Call `fn` every `interval` seconds on a daemon thread until stopped.

    Tk-free on purpose: pages pass a function that marshals onto the UI thread
    (`lambda: self.safe_after(self._paint)`), so both front ends share one
    repaint cadence without either of them owning a timer loop.
    """

    def __init__(self, fn: Callable[[], None], interval: float = TICK_SECONDS):
        self._fn = fn
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ProgressTicker":
        def loop():
            while not self._stop.wait(self._interval):
                try:
                    self._fn()
                except Exception:
                    logger.exception("Progress ticker callback failed")
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()


@dataclass
class FolderResult:
    """What one folder's pass produced. `ok=False` always carries an `error`.

    `cancelled` is NOT a failure: the folder did real work and saved it, the
    user just asked it to stop. Conflating the two would put a red "folder
    failed" line under a perfectly healthy partial run.
    """
    folder: str
    ok: bool
    error: Optional[str] = None
    cancelled: bool = False
    files_found: int = 0          # Excel files on disk
    new_files: int = 0            # actually processed (not skipped as known)
    new_trims: int = 0            # trim analyses this pass saved
    models: Set[str] = field(default_factory=set)
    summary: object = None        # BatchSummary | None
    seconds: float = 0.0
    phases: Dict[str, float] = field(default_factory=dict)


@dataclass
class IngestReport:
    """Every folder's result plus the wall-clock the whole run took."""
    results: List[FolderResult] = field(default_factory=list)
    seconds: float = 0.0
    cancelled: bool = False
    # What the run SET OUT to do, as opposed to what it got through. Both are
    # 0 when nobody told us (a direct run_folders call in a test); the summary
    # line prints only the halves it actually knows.
    folders_requested: int = 0
    files_planned: int = 0

    @property
    def folder_count(self) -> int:
        return len(self.results)

    @property
    def files_found(self) -> int:
        return sum(r.files_found for r in self.results)

    @property
    def new_files(self) -> int:
        return sum(r.new_files for r in self.results)

    @property
    def new_trims(self) -> int:
        return sum(r.new_trims for r in self.results)

    @property
    def failed(self) -> List[FolderResult]:
        return [r for r in self.results if not r.ok]

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def models(self) -> Set[str]:
        out: Set[str] = set()
        for r in self.results:
            out |= r.models
        return out


def format_elapsed(seconds: float) -> str:
    s = int(round(max(seconds, 0)))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60} s"
    return f"{s // 3600} h {(s % 3600) // 60} min"


def format_ingest_summary(report: IngestReport) -> str:
    """The one line Home shows after a run. Spec: "3 folders · 214 new files ·
    2 min 40 s" — and, when a share was down, which one and why.

    A STOPPED run gets a different sentence, not the same one with a smaller
    number: "3 folders · 214 new files" reads as a completed history import,
    and being wrong about that costs hours. It says how far it got, out of
    what, and — the part that decides whether Stop ever gets pressed — that
    pressing the button again continues rather than starts over.
    """
    if not report.results and not report.cancelled:
        return "No folders configured — add them in Settings."
    n = report.folder_count
    new = report.new_files
    if report.cancelled:
        total_folders = report.folders_requested or n
        # "of N" only when N was actually measured (the pre-run scan). An
        # invented denominator is worse than none.
        of_planned = f" of {report.files_planned:,}" if report.files_planned else ""
        line = (f"Stopped after {new:,}{of_planned} new file"
                + ("" if new == 1 and not of_planned else "s")
                + f" ({n} of {total_folders} folder"
                + ("" if total_folders == 1 else "s") + ")"
                f" · {format_elapsed(report.seconds)}"
                " — press Process everything new again to continue; it resumes "
                "where this left off (everything already saved is skipped).")
        bad = report.failed
        if bad:
            detail = "; ".join(f"{r.folder} ({r.error})" for r in bad)
            line += f"  ⚠ {len(bad)} folder" + ("" if len(bad) == 1 else "s")
            line += f" failed: {detail}"
        return line
    files = ("no new files" if new == 0 else
             f"{new:,} new file" + ("" if new == 1 else "s"))
    line = (f"{n} folder" + ("" if n == 1 else "s") + f" · {files}"
            f" · {format_elapsed(report.seconds)}")
    bad = report.failed
    if bad:
        # Named, not counted: "1 folder failed" sends someone hunting through
        # a log for which one.
        detail = "; ".join(f"{r.folder} ({r.error})" for r in bad)
        line += f"  ⚠ {len(bad)} of {n} folders failed: {detail}"
    return line


def log_phases(phases: dict, total: int, processor, summary) -> None:
    """One INFO line per batch naming every phase and what it cost.

    When a batch is slow, this line says WHICH phase — the 2026-07/08 work
    investigations each cost hours of log archaeology because the only timings
    were per-file DEBUG noise that had already rotated away.
    """
    s = getattr(processor, "last_scan_stats", {}) or {}
    parts = [f"walk {phases.get('walk', 0):.1f}s ({total:,} files)",
             f"load {s.get('load_seconds', 0):.1f}s",
             f"check {s.get('check_seconds', 0):.1f}s "
             f"({s.get('memory_hits', 0):,} known in memory)",
             f"verify {s.get('needs_hash', 0):,} files "
             f"{s.get('verify_seconds', 0):.1f}s",
             f"process {getattr(summary, 'processed', 0):,} files "
             f"{phases.get('process', 0):.1f}s"]
    parts.append(f"rematch {phases['rematch']:.1f}s" if "rematch" in phases
                 else "rematch skipped (no new trims)")
    if "retrain" in phases:
        parts.append(f"retrains {phases['retrain']:.1f}s")
    if "advance" in phases:
        parts.append(f"advance {phases['advance']:.1f}s")
    logger.info("Batch phases: %s", " | ".join(parts))


def _say(on_phase: Optional[Callable[[str], None]], text: str) -> None:
    if on_phase is None:
        return
    try:
        on_phase(text)
    except Exception:
        logger.exception("on_phase callback failed")


def _post_batch(db, models_in_batch: Set[str], new_trims: int, phases: dict,
                on_phase) -> None:
    """Re-link final tests, retrain what the links changed, advance drift.

    Order matters. FT files that arrived before their trim files matched
    nothing at save time (2026-07-13: a third of recent unmatched FT records
    had an in-window trim that simply wasn't in the DB yet), so the rematch
    runs BEFORE the drift advance — otherwise escape/FT metrics advance over
    links that don't exist yet.

    The rematch only runs when this batch actually SAVED trims, and only for
    those models. FT records are matched at their own save time, so a rematch
    can only ever help trims that arrived after their FT record. An FT-only
    batch used to re-attempt all ~100k unmatchable records ("0 of 101,605
    linked", after every batch).
    """
    if new_trims:
        t = time.monotonic()
        try:
            rl = db.rematch_unlinked_final_tests(models=models_in_batch)
            if rl.get("new_matches"):
                _say(on_phase, f"Re-linked {rl['new_matches']:,} final-test "
                               "records to their trim data…")
                # Late links carry OLD test dates — often behind the escape/FT
                # watermark, so advance would never feed them. Retrain the
                # affected models so their baselines rebuild WITH the links.
                t_r = time.monotonic()
                try:
                    from laser_trim_analyzer.ml.drift_training import train_drift_detector
                    for m in rl.get("models", []):
                        train_drift_detector(db, model=m)
                    logger.info("Retrained %d models after FT relink",
                                len(rl.get("models", [])))
                except Exception:
                    logger.exception("Post-relink retrain failed")
                phases["retrain"] = time.monotonic() - t_r
        except Exception:
            logger.exception("Post-batch FT rematch failed")
        phases["rematch"] = time.monotonic() - t

    t = time.monotonic()
    try:
        from laser_trim_analyzer.ml.drift_training import advance_drift_state
        advanced = 0
        for m in sorted(models_in_batch):
            advanced += advance_drift_state(db, model=m)
        logger.info("Drift state advanced for %d (model, metric) rows across "
                    "%d models", advanced, len(models_in_batch))
    except Exception:
        logger.exception("Drift advance after batch failed")
    phases["advance"] = time.monotonic() - t


@_with_ingest_switch_interval
def run_folder(folder: str, *, db, config, incremental: bool = True,
               progress: Optional[ProgressCoalescer] = None,
               on_phase: Optional[Callable[[str], None]] = None,
               on_total: Optional[Callable[[int], None]] = None,
               cancel: Optional[threading.Event] = None) -> FolderResult:
    """Process ONE folder end to end. Never raises; failures come back as data.

    Blocking and thread-safe to call from a worker. Every callback is invoked
    on THIS thread, so a caller that touches widgets must marshal them itself.

    `cancel` is cooperative and lands at a batch boundary inside the processor
    (see Processor.process_batch). Everything the run had already saved stays
    saved and still gets its post-batch consistency work — the FT re-link and
    the drift advance — because a stopped run's rows are as real as a finished
    run's, and leaving them unlinked would quietly skew the next FOCUS list.
    """
    started = time.monotonic()
    phases: dict = {}
    _say(on_phase, "Scanning folder for Excel files… "
                   "(network folders can take a minute)")

    problem = ingest_folder_problem(folder)
    if problem:
        logger.warning("Ingest folder unusable: %s — %s", folder, problem)
        _say(on_phase, f"{folder}: {problem}")
        return FolderResult(folder=folder, ok=False, error=problem,
                            seconds=time.monotonic() - started)

    t = time.monotonic()
    files, disk_stats = discover_excel_files(folder)
    phases["walk"] = time.monotonic() - t
    if not files:
        _say(on_phase, "No .xls/.xlsx files found.")
        return FolderResult(folder=folder, ok=True, files_found=0, phases=phases,
                            seconds=time.monotonic() - started)

    processor = Processor(config=config)          # I7: no db= param
    total = len(files)
    if on_total is not None:
        on_total(total)
    _say(on_phase, f"Checking {total:,} files against the database…")

    def progress_callback(status: ProcessingStatus) -> None:
        if progress is not None:
            progress.note(status)             # worker-side, no Tk

    def note_bucket(bucket: str, reason: str = "") -> None:
        if progress is not None:
            progress.bucket(bucket, reason)

    gen = processor.process_batch([Path(p) for p in files],
                                  progress_callback=progress_callback,
                                  incremental=incremental,
                                  disk_stats=disk_stats,
                                  cancel=cancel)
    summary = None
    models_in_batch: Set[str] = set()
    new_trims = 0                # trim analyses actually saved by THIS batch
    t = time.monotonic()
    try:
        while True:
            result = next(gen)
            # Persist trim results (the caller owns the trim save; FT and
            # smoothness are already saved inside the processor).
            if getattr(result, "file_type", "trim") == "trim":
                try:
                    db.save_analysis(result)
                    new_trims += 1
                except Exception as exc:
                    # A duplicate hitting the unique constraint means the unit
                    # is ALREADY in the database (e.g. the same file under a
                    # second path form) — that's a skip, not an error.
                    if "UNIQUE constraint" in str(exc) or "IntegrityError" in type(exc).__name__:
                        note_bucket("skipped")
                    else:
                        note_bucket("errors",
                                    f"{result.metadata.filename}: save failed: {exc}")
            model = getattr(result.metadata, "model", None)
            if model and model != "Unknown":
                models_in_batch.add(model)
            bucket = bucket_for_status(result.overall_status)
            reason = (f"{result.metadata.filename}: {result.overall_status.value}"
                      if bucket in ("failed", "errors") else "")
            note_bucket(bucket, reason)
    except StopIteration as stop:
        summary = stop.value
        phases["process"] = time.monotonic() - t
    except Exception as exc:
        # 2026-07-09: an exception here previously killed the worker thread
        # silently — Start stayed disabled, the app looked locked, and the
        # reason never reached the screen. It reaches the caller now.
        logger.exception("Batch processing aborted for %s", folder)
        _say(on_phase, f"Stopped: {exc}")
        return FolderResult(folder=folder, ok=False, error=str(exc),
                            files_found=total, new_trims=new_trims,
                            models=models_in_batch, phases=phases,
                            seconds=time.monotonic() - started)

    if models_in_batch:
        _post_batch(db, models_in_batch, new_trims, phases, on_phase)
    log_phases(phases, total, processor, summary)
    stopped = bool(cancel is not None and cancel.is_set())
    return FolderResult(folder=folder, ok=True, cancelled=stopped,
                        files_found=total,
                        new_files=int(getattr(summary, "processed", 0) or 0),
                        new_trims=new_trims, models=models_in_batch,
                        summary=summary, phases=phases,
                        seconds=time.monotonic() - started)


@_with_ingest_switch_interval
def run_folders(folders: Sequence[str], *, db, config, incremental: bool = True,
                progress: Optional[ProgressCoalescer] = None,
                on_phase: Optional[Callable[[str], None]] = None,
                on_total: Optional[Callable[[int], None]] = None,
                on_folder_start: Optional[Callable[[int, int, str], None]] = None,
                on_folder_done: Optional[Callable[[FolderResult], None]] = None,
                cancel: Optional[threading.Event] = None,
                ) -> IngestReport:
    """Run the configured folders SEQUENTIALLY, in order, to the end.

    Sequential because they share one database and one disk: two folders
    racing each other only trade wall-clock for lock contention. In order
    because the order is the user's — laser folders first, Final Test last, so
    the FT records find their trims.

    One folder failing does NOT stop the run. A share being down is a Tuesday,
    and losing the other two folders' work over it is a much worse day; the
    failure is carried in the report and named in the summary line.

    A CANCEL, by contrast, does stop the run — that is the whole point of the
    button — but only between folders and only after the folder in flight has
    finished its current batch and saved it. Every folder that ran is still in
    the report, with everything it counted.
    """
    started = time.monotonic()
    report = IngestReport(folders_requested=len(folders))
    total_folders = len(folders)
    for i, folder in enumerate(folders, start=1):
        if cancel is not None and cancel.is_set():
            report.cancelled = True
            break
        if on_folder_start is not None:
            on_folder_start(i, total_folders, folder)
        result = run_folder(folder, db=db, config=config,
                            incremental=incremental, progress=progress,
                            on_phase=on_phase, on_total=on_total, cancel=cancel)
        report.results.append(result)
        if on_folder_done is not None:
            on_folder_done(result)
        if result.cancelled or (cancel is not None and cancel.is_set()):
            report.cancelled = True
            break
    report.seconds = time.monotonic() - started
    return report
