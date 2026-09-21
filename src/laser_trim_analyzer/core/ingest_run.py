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

# Rate/ETA shape. The window is short on purpose: a share that goes slow
# half-way through a four-hour run has to become visible within a minute, not
# be averaged away by the hour of fast files before it.
RATE_WINDOW_SECONDS = 60.0
# Nothing is claimed for the first half minute. That early sample is one
# folder's worth of small files and predicts nothing about 106k of them; a
# confident wrong ETA is worse than no ETA.
ETA_WARMUP_SECONDS = 30.0
# Two samples 0.25 s apart are noise, not a rate.
MIN_RATE_SPAN_SECONDS = 2.0

# How often the run says where its time is going, in files saved. A folder-end
# summary is no use for diagnosis: a run that is stopped because it is too slow
# never reaches the end, which is exactly what happened on 2026-09-21 -- the
# batch line existed and the log had none of it. This reports while it runs.
#
# Small on purpose. The whole point is to answer "where is the time going" from
# a SHORT look at a slow run: at the 1 file/sec that prompted this, 500 would
# have meant eight minutes before the first line, which is the same failure in
# a smaller size. 200 gives an answer inside four minutes at that rate, and
# every 55 seconds at the 3.6 files/sec the same run later reached.
SAVE_REPORT_EVERY = 200

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

    EVERY path leaves here in `str(Path(...))` form — the returned list, the
    stats keys, and the roots the walk recurses into. That is not tidiness: it
    is the key the consumer looks the dict up by. `Processor._classify_scan`
    asks `self._disk_stats[str(file_path)]`, and the paths stored in
    `analysis_results.file_path` / `final_test_results.file_path` are written
    the same way, so any other spelling is a guaranteed miss.

    The work incident (2026-09-14): the configured folders come from Tk's
    `askdirectory`, which returns FORWARD slashes on Windows, so `entry.path`
    came out mixed — `//192.168.66.9/…/Test Station\\6607\\file.xls` — while
    the lookup used the all-backslash `str(Path(...))` form. Nothing matched.
    Every scan logged "check … (0 known in memory)" and then stat()ed all
    171,006 final-test files over the share: 542 seconds, every run, on a
    folder where nothing had changed. Normalising here (~0.3 s for 172k paths,
    against nine minutes of network round trips) is the whole fix, and it
    belongs here rather than in the processor because this is where the key
    and the value are created together.
    """
    def scan_one(d):
        subdirs, found = [], []
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            subdirs.append(str(Path(entry.path)))
                        elif entry.name.lower().endswith((".xls", ".xlsx")):
                            st = entry.stat()
                            found.append((str(Path(entry.path)),
                                          (st.st_size, st.st_mtime)))
                    except OSError:
                        continue   # unreadable entry: skip, don't die
        except OSError:
            logger.warning("Could not list folder: %s", d)
        return subdirs, found

    out: List[str] = []
    stats: dict = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        # The root too: a root carrying a trailing or doubled separator would
        # otherwise push that spelling into every child path scandir joins.
        pending = {pool.submit(scan_one, str(Path(folder)))}
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

    `done` counts WORK, against the work total the pre-scan announced (see
    `_plan_work`). Files the database already had are not work and are not
    counted: crediting them made "1,214 / 8,900 files" mean "1,214 of the
    8,900 files that happen to be on the share", and the several thousand
    already-known files of every folder the run had not reached yet sat inside
    "remaining" until that folder's own scan landed. Divided by a real
    processing rate, that is where "about 24 h left" came from on a re-run
    that had about ten minutes of work in it (James, 2026-09-12).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._done = 0
        # Of `_done`, how many were CREDITED in bulk by the incremental scan
        # rather than processed one at a time. The bar wants both together;
        # the rate wants only the steady stream, or one 3,000-file credit in a
        # single 0.25 s tick reads as 12,000 files/s and the ETA says "under a
        # minute" to a run with three hours left.
        self._known = 0
        self._file = ""
        self._scan_msg: Optional[str] = None
        self._moved = False
        self._counts = {k: 0 for k in BUCKETS}
        self._reasons: deque = deque(maxlen=MAX_REASONS)
        # The announced denominator, and the current folder's budget of files
        # the pre-scan already settled from memory. See expect_known().
        self._total: Optional[int] = None
        self._expect_known = 0
        self._clamped = False

    # ---- run setup (called by run_folder / run_folders, not by the UI) ----
    def set_total(self, total: int) -> None:
        """The work total that was announced through `on_total`.

        Held only to CLAMP: `done` overrunning the denominator means the
        pre-scan and the processor disagreed about what counts as work, and a
        bar that reads 104% hides that instead of reporting it.
        """
        with self._lock:
            self._total = max(0, int(total))

    def expect_known(self, n: int) -> None:
        """How many of the next folder's files the pre-scan settled in memory.

        The processor reports its whole already-in-database population in one
        "known" event whose count also includes the files it had to verify by
        hash — those WERE work and are in the total; the memory-settled ones
        never were. This budget is what separates the two.

        It REPLACES any remainder rather than adding to it: a folder that
        failed or stopped half-way leaves its budget unspent, and carrying
        that into the next folder would silently swallow that folder's credit.
        """
        with self._lock:
            self._expect_known = max(0, int(n))

    # ---- worker side ----
    def _credit(self, n: int) -> None:
        """Add `n` finished units of work to the bar. Caller holds the lock."""
        if n <= 0:
            return
        self._done += n
        if self._total is not None and self._done > self._total:
            if not self._clamped:
                self._clamped = True
                logger.warning(
                    "Ingest progress overran its plan: %d done against a "
                    "planned %d — the pre-scan and the processor disagreed "
                    "about what counts as work. Clamping the bar.",
                    self._done, self._total)
            self._done = self._total

    def note(self, status: ProcessingStatus) -> None:
        with self._lock:
            if status.status == "scanning":
                # "Found N new files (M already in database)" — the headline
                # the user waits for. Not progress: it precedes it.
                self._scan_msg = status.message or "Scanning…"
                return
            if status.status == "known":
                # The whole already-in-database population of this folder, in
                # one event. Only the part the pre-scan could NOT settle from
                # memory is work this run did (it cost a stat or a full hash
                # over the share); the rest was never in the denominator.
                n = int(getattr(status, "count", 0) or 0)
                settled = min(self._expect_known, n)
                self._expect_known -= settled
                credit = n - settled
                self._credit(credit)
                self._known += credit
                # `moved` stays False for a folder that was entirely known:
                # nothing moved, so the page keeps showing the scan message
                # ("…already in database") instead of repainting the same bar.
                self._moved = self._moved or bool(credit)
                return
            if status.status in ("completed", "skipped", "failed"):
                n = int(getattr(status, "count", 1) or 1)
                if status.status == "skipped" and self._expect_known > 0:
                    # The sequential path (folders under the turbo threshold)
                    # has no bulk "known" event — it skips each already-known
                    # file individually. Same budget, spent one at a time.
                    settled = min(self._expect_known, n)
                    self._expect_known -= settled
                    n -= settled
                self._credit(n)
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
            return {"scan_msg": scan_msg, "done": self._done,
                    "processed": self._done - self._known, "file": self._file,
                    "counts": counts, "reasons": reasons, "moved": moved}

    def reset(self) -> None:
        with self._lock:
            self._done = 0
            self._known = 0
            self._file = ""
            self._scan_msg = None
            self._moved = False
            self._counts = {k: 0 for k in BUCKETS}
            self._reasons.clear()
            self._total = None
            self._expect_known = 0
            self._clamped = False


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


class EtaEstimator:
    """How fast the run is going and how much longer it has. Pure, Tk-free.

    Lives here rather than in the page for one reason: the wording is the part
    that can be wrong. "about 7 min left" is a claim a moving average can
    support; "6 min 41 s left" is a claim nothing here can support, and
    printing it makes someone stand and wait for a number that was never real.
    Keeping the arithmetic in `core/` means the thresholds are unit-tested once
    and both front ends say the same words.

    Fed with the PROCESSED count — files this run actually parsed — not the
    bar's numerator. They differ by the incremental scan's bulk credit for
    files the database already had, and folding a 148,000-file step change
    into a moving average computed over 0.25 s produces a rate off by five
    orders of magnitude.

    Every method takes an optional `now` so the tests can drive time instead
    of sleeping through a minute of it.
    """

    def __init__(self, *, window: float = RATE_WINDOW_SECONDS,
                 warmup: float = ETA_WARMUP_SECONDS,
                 min_span: float = MIN_RATE_SPAN_SECONDS,
                 now: Optional[float] = None) -> None:
        self._window = window
        self._warmup = warmup
        self._min_span = min_span
        self._started = time.monotonic() if now is None else now
        self._samples: deque = deque()      # (timestamp, processed count)

    def note(self, processed: int, now: Optional[float] = None) -> None:
        """Record where the run is. Called on the Tk thread, 4 Hz, arithmetic
        only — cheap enough to run on every repaint, and running on EVERY
        repaint (not only the ones where the count moved) is what makes a
        stall show up as a falling rate instead of a frozen one."""
        t = time.monotonic() if now is None else now
        self._samples.append((t, processed))
        cutoff = t - self._window
        while len(self._samples) > 2 and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def elapsed(self, now: Optional[float] = None) -> float:
        return max(0.0, (time.monotonic() if now is None else now) - self._started)

    def rate(self, now: Optional[float] = None) -> Optional[float]:
        """Files per second over the retained window, or None when the sample
        is too short to mean anything."""
        if len(self._samples) < 2:
            return None
        (t0, d0), (t1, d1) = self._samples[0], self._samples[-1]
        span = t1 - t0
        if span < self._min_span:
            return None
        return max(0.0, (d1 - d0) / span)

    def eta_seconds(self, remaining: int,
                    now: Optional[float] = None) -> Optional[float]:
        if remaining <= 0:
            return 0.0
        if self.elapsed(now) < self._warmup:
            return None
        r = self.rate(now)
        if not r:                       # None, or a dead stop: no honest answer
            return None
        return remaining / r

    def eta_text(self, remaining: int, now: Optional[float] = None) -> str:
        if remaining <= 0:
            # Nothing left to predict. Saying "under a minute left" about a
            # run that has finished counting is worse than saying nothing.
            return ""
        eta = self.eta_seconds(remaining, now)
        if eta is None:
            return "estimating…"
        return format_eta(eta)


def format_eta(seconds: float) -> str:
    """Words for a duration, at the precision the estimate deserves.

    Deliberately coarse. Rounding 401 seconds to "about 7 min" is honest about
    a moving average; rendering it "6 min 41 s" invents four significant
    figures out of a number that moves every tick.
    """
    if seconds < 60:
        return "under a minute left"
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"about {max(1, minutes)} min left"
    hours, minutes = divmod(minutes, 60)
    if minutes == 0:
        return f"about {hours} h left"
    return f"about {hours} h {minutes} min left"


def format_rate(rate: float) -> str:
    """"17 files/s" — one decimal only where it carries information."""
    return (f"{rate:.0f} files/s" if rate >= 10 else f"{rate:.1f} files/s")


def format_clock(seconds: float) -> str:
    """Elapsed as a clock: 0:04, 12:04, 1:02:05."""
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def format_progress_line(done: int, total: int, *, folder: int = 0,
                         folders: int = 0, rate: Optional[float] = None,
                         eta: str = "", elapsed: Optional[float] = None,
                         filename: str = "") -> str:
    """The one line above the bar, for the WHOLE run.

    Spec: "Folder 2 of 3 · 1,214 done · 7,686 to go · 17 files/s · about 7 min
    left · elapsed 12:04". Every part is dropped when it is not known yet
    rather than rendered as a zero — "0 / 0 files · 0 files/s" during the
    folder walk is worse than saying nothing, because it looks like a stuck
    run.

    "N to go" rather than "done / total" because that is the question being
    asked of it (James, mid-ingest: "how many files is left to process?"). A
    fraction makes the reader do the subtraction, and the old numerator and
    denominator were both files-on-disk, so the subtraction gave the wrong
    answer anyway — see ProgressCoalescer. `total` is the run's WORK, so the
    remainder here is work left, which is the number the ETA is computed from.
    """
    parts = []
    if folders > 1 and folder:
        parts.append(f"Folder {folder} of {folders}")
    parts.append(f"{done:,} done")
    if total > 0:
        parts.append(f"{max(0, total - done):,} to go")
    if rate:
        parts.append(format_rate(rate))
    if eta:
        parts.append(eta)
    if elapsed is not None:
        parts.append(f"elapsed {format_clock(elapsed)}")
    if filename:
        parts.append(filename)
    return " · ".join(parts)


@dataclass
class FolderPlan:
    """What one folder will COST this run, decided before a file is opened.

    `work` is what the bar counts down: files the processor will actually
    touch — the new ones, plus the few whose identity the in-memory index
    cannot settle and which therefore need a stat or a full hash over the
    share. Those are work even when the hash says "already known", because the
    run paid for them. `known` is the rest: settled from memory, free, and
    never part of the denominator.

    `from_index=False` marks the fallback — the processed-file index could not
    be read, so every file found is counted as work. That OVER-states the run,
    which is the right way to be wrong about a denominator: the ETA comes in
    long and shortens, instead of promising a finish that isn't coming.
    """
    files: int = 0
    work: int = 0
    known: int = 0
    from_index: bool = True


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
    # Files this run recorded as permanently unreadable (a skip marker was
    # written for their path). NOT the "skipped" bucket, which counts files
    # already in the database — conflating the two would report ~170,000
    # known-good final tests as junk. Measured as the growth in marker rows
    # across the run, so it needs no per-file plumbing through the processor.
    marked_unreadable: int = 0

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


def _unreadable_clause(report: IngestReport) -> str:
    """The " · N files could not be read…" half-sentence, or nothing.

    Silent at zero, which is what every run after the first one should be.
    It exists because the count is otherwise invisible: a file the parser
    refuses produces no record to look at, and before 2026-09-14 it produced
    no marker either, so ~930 of them were re-offered as "new" every day
    forever. Now they are recorded once, and this says so the one time it
    happens.
    """
    n = getattr(report, "marked_unreadable", 0) or 0
    if n <= 0:
        return ""
    return (f" · {n:,} file" + ("" if n == 1 else "s")
            + " could not be read and "
            + ("was" if n == 1 else "were")
            + " recorded as unreadable (skipped from now on)")


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
        line += _unreadable_clause(report)
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
    line += _unreadable_clause(report)
    bad = report.failed
    if bad:
        # Named, not counted: "1 folder failed" sends someone hunting through
        # a log for which one.
        detail = "; ".join(f"{r.folder} ({r.error})" for r in bad)
        line += f"  ⚠ {len(bad)} of {n} folders failed: {detail}"
    return line


def unreadable_count(db) -> int:
    """How many files the app is currently refusing to re-offer.

    Wrapped rather than called directly so HOME can ask a database that
    predates the marker without a try/except of its own.
    """
    try:
        return int(db.count_failed_file_markers())
    except Exception:  # noqa: BLE001 - a count must never break the page
        logger.debug("unreadable-file count unavailable", exc_info=True)
        return 0


def unreadable_notice(count: int) -> str:
    """The one line HOME shows, or "" when there is nothing to say.

    Never silent about a decision the app made on its own: "process everything
    new" quietly skipping 178 files it failed on last week is the same class
    of surprise as re-reading them forever, and the way back has to be named
    where the number is. Pure, so the wording is testable without a Tk root.
    """
    if not count or count < 0:
        return ""
    noun = "file" if count == 1 else "files"
    verb = "is" if count == 1 else "are"
    return (f"{count:,} {noun} {verb} being skipped because they failed to "
            f"read before — Settings → Retry unreadable files")


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
    # Saving is inside `process` and is the part a worker pool cannot speed up,
    # so name it separately and as a share -- "process 520s" alone sent one
    # investigation looking at the parser when most of it was the database.
    done = max(1, getattr(summary, "processed", 0) or 1)
    if "save" in phases:
        proc_s = phases.get("process", 0) or 1
        parts.append(f"of which save {phases['save']:.1f}s"
                     f" ({phases['save'] / proc_s * 100:.0f}%,"
                     f" {phases['save'] / done * 1000:.0f} ms/file)")
        parts.append(f"rest {phases.get('process', 0) - phases['save']:.1f}s"
                     f" ({(phases.get('process', 0) - phases['save']) / done * 1000:.0f} ms/file)")
    parts.append(f"rematch {phases['rematch']:.1f}s" if "rematch" in phases
                 else "rematch skipped (no new trims)")
    if "retrain" in phases:
        parts.append(f"retrains {phases['retrain']:.1f}s")
    if "advance" in phases:
        parts.append(f"advance {phases['advance']:.1f}s")
    if "findings" in phases:
        parts.append(f"findings {phases['findings']:.1f}s")
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

    # Process findings read trim verdicts and captured passes only, so a batch
    # that saved no trims changes nothing they depend on -- the same gate the
    # rematch above uses. Guarded like every other phase here: findings are an
    # aid, and an aid must never be able to fail an ingest.
    if new_trims:
        t = time.monotonic()
        try:
            from laser_trim_analyzer.findings import engine as _findings
            _say(on_phase, f"Working out process findings for {len(models_in_batch):,} models… "
                           f"(a few minutes; the ingest itself is finished)")
            report: dict = {}
            stored = _findings.refresh_findings(db, sorted(models_in_batch), report)
            # refresh_findings catches per-model failures itself and returns a count, so the guard
            # below would almost never fire: without this, 446 models ALL failing logs as
            # "refreshed for 446 models (0 findings)" -- indistinguishable from nothing to report.
            failed, partial = report.get("failed_models") or {}, report.get("analyzer_errors") or {}
            if failed or partial:
                names = sorted(set(failed) | set(partial))
                logger.error("Process findings: %d of %d models could not be worked out at all and %d "
                             "had an analyzer fail (%s%s). Each model's Findings tab names what failed.",
                             len(failed), len(models_in_batch), len(partial), ", ".join(names[:8]),
                             " …" if len(names) > 8 else "")
                _say(on_phase, f"Process findings: {len(failed):,} model(s) could not be worked out, "
                               f"{len(partial):,} had an analyzer fail — see the log")
            else:
                logger.info("Process findings refreshed for %d models (%d findings)",
                            len(models_in_batch), stored)
        except Exception:
            logger.exception("Process findings refresh after batch failed")
        phases["findings"] = time.monotonic() - t


@_with_ingest_switch_interval
def run_folder(folder: str, *, db, config, incremental: bool = True,
               progress: Optional[ProgressCoalescer] = None,
               on_phase: Optional[Callable[[str], None]] = None,
               on_total: Optional[Callable[[int], None]] = None,
               cancel: Optional[threading.Event] = None,
               discovered: Optional[Tuple[List[str], Dict[str, tuple]]] = None,
               walk_seconds: float = 0.0,
               plan: Optional[FolderPlan] = None) -> FolderResult:
    """Process ONE folder end to end. Never raises; failures come back as data.

    Blocking and thread-safe to call from a worker. Every callback is invoked
    on THIS thread, so a caller that touches widgets must marshal them itself.

    `plan` is the multi-folder run's pre-scan verdict on this folder (how much
    of it is work, how much the database already has). Given one, this folder
    is a part of a larger run: the total was announced for the whole run and
    must not be re-announced here. Without one — the Process page's single
    folder — the same sizing is done here, for this folder alone.

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

    if discovered is not None:
        files, disk_stats = discovered
        phases["walk"] = walk_seconds
    else:
        t = time.monotonic()
        files, disk_stats = discover_excel_files(folder)
        phases["walk"] = time.monotonic() - t
    if not files:
        _say(on_phase, "No .xls/.xlsx files found.")
        return FolderResult(folder=folder, ok=True, files_found=0, phases=phases,
                            seconds=time.monotonic() - started)

    total = len(files)
    own_plan = plan is None
    if own_plan:
        # A folder run of its own (the Process page). Size it the way the
        # multi-folder pre-scan sizes a whole run, so both front ends count
        # down the same thing: work, not files-on-disk.
        if not incremental:
            plan = FolderPlan(files=total, work=total)
        else:
            planned = _plan_work({folder: files}, disk_stats, config=config)
            plan = (planned[folder] if planned else
                    FolderPlan(files=total, work=total, from_index=False))
            if not planned:
                _say(on_phase, "Could not read the processed-file index — "
                               "counting every file found as still to do.")
    if progress is not None:
        progress.expect_known(plan.known)
        if own_plan:
            progress.set_total(plan.work)
    if own_plan and on_total is not None:
        on_total(plan.work)
    if plan.known:
        _say(on_phase, f"Checking {total:,} files against the database… "
                       f"({plan.known:,} already in it, {plan.work:,} to do)")
    else:
        _say(on_phase, f"Checking {total:,} files against the database…")

    processor = Processor(config=config)          # I7: no db= param

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
    save_seconds = 0.0           # the serial half: every save, one after another
    t = time.monotonic()
    try:
        while True:
            result = next(gen)
            # Persist trim results (the caller owns the trim save; FT and
            # smoothness are already saved inside the processor).
            if getattr(result, "file_type", "trim") == "trim":
                try:
                    # Timed because it is the SERIAL half of the loop: the pool
                    # parses four at a time, every save happens here, one after
                    # another, on this thread. Measured 2026-09-21 at work: a
                    # batch spent 771.7 ms per file where the same pool parsing
                    # alone costs 339.2, so 56% of the ingest is outside parse
                    # -- and a process pool would not touch any of it.
                    _t_save = time.monotonic()
                    db.save_analysis(result)
                    save_seconds += time.monotonic() - _t_save
                    new_trims += 1
                    if new_trims and new_trims % SAVE_REPORT_EVERY == 0:
                        _since = time.monotonic() - t
                        logger.info(
                            "Ingest so far: %s saved | %.0f ms/file overall | "
                            "save %.0f ms/file (%.0f%%) | everything else %.0f ms/file. "
                            "Saving is SERIAL -- one at a time on this thread -- so it "
                            "is the part more workers cannot help.",
                            f"{new_trims:,}", _since / new_trims * 1000,
                            save_seconds / new_trims * 1000,
                            (save_seconds / _since * 100) if _since else 0.0,
                            (_since - save_seconds) / new_trims * 1000)
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
        phases["save"] = save_seconds
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


def _plan_work(files_by_folder: Dict[str, Sequence[str]],
               disk_stats: Dict[str, tuple], *, config
               ) -> Optional[Dict[str, FolderPlan]]:
    """How much of what the walk found is WORK, decided from memory alone.

    This is what makes "how many to go" a real number. The walk knows how many
    Excel files are on the share; on any re-run almost all of them are already
    in the database, and counting them as remaining is what produced "about
    24 h left" on a run with minutes of work in it.

    It is affordable because the processor's own scan already splits in two:
    `_classify_scan` answers from the in-memory index (no file I/O at all,
    only the (size, mtime) the walk captured with each directory listing), and
    only the leftovers pay a stat or a hash over the share. This calls the
    memory half — once, for every folder, on a THROWAWAY processor whose
    caches nobody else sees — and leaves the I/O half where it belongs, inside
    the run. Those decisions are deterministic given the same database state,
    so the per-folder run reaches the same counts when it gets there.

    The price is ONE extra load of the processed-file index (the per-folder
    run loads its own; a few seconds against the 3.8 GB work database) — paid
    once for the whole run, against the 70-minute re-hash that lifting the I/O
    half in here would have cost.

    `use_ml=False`: the ML threshold load reaches for the global database
    manager and is no part of counting files.

    Returns None when the index cannot be read at all — including when the
    module's `Processor` has been replaced by a stub that has no scan. The
    caller then counts every file found as work and says so.
    """
    if not any(files_by_folder.values()):
        return {f: FolderPlan() for f in files_by_folder}
    try:
        proc = Processor(config=config, use_ml=False)
        proc._load_processed_hashes()
        if getattr(proc, "_processed_filenames", None) is None:
            return None
        # The union, so a file reached under one folder's path form is still
        # recognised while another folder's files are being classified.
        proc._disk_stats = disk_stats
        plans: Dict[str, FolderPlan] = {}
        for folder, files in files_by_folder.items():
            new = needs = known = 0
            for f in files:
                decision = proc._classify_scan(Path(f))
                if decision == "processed":
                    known += 1
                elif decision == "needs_hash":
                    needs += 1
                else:
                    new += 1
            plans[folder] = FolderPlan(files=len(files), work=new + needs,
                                       known=known)
        return plans
    except Exception:
        logger.exception("Could not size the run from the processed-file "
                         "index; counting every file found as work")
        return None


def _prescan(folders: Sequence[str], on_phase, cancel, *, config,
             incremental: bool
             ) -> Tuple[Dict[str, tuple], Dict[str, FolderPlan], bool]:
    """Walk every folder BEFORE processing any of it, so the run knows its size.

    This is the whole of "how many files has it got to go?". Until it existed
    the panel knew only the folder in flight, and the bar restarted at every
    folder boundary — during a four-hour first ingest that is three separate
    progress bars and no answer.

    The walk finds every Excel file; `_plan_work` then says how many of them
    are actually work. Before that split the denominator was files-on-disk,
    and each folder's already-known population only left "remaining" when that
    folder's own scan landed — so a re-run of a mostly-ingested share spent
    most of its life claiming hours of work it did not have.

    Folders that are unusable are NOT walked and NOT reported here — run_folder
    owns the "offline share?" wording and the ok=False result, and having two
    places that decide a share is dead is how they end up disagreeing.
    """
    walked: Dict[str, tuple] = {}
    found: Dict[str, Sequence[str]] = {}
    stats: Dict[str, tuple] = {}
    n = len(folders)
    for i, folder in enumerate(folders, start=1):
        if cancel is not None and cancel.is_set():
            return walked, {}, False
        _say(on_phase, f"Scanning folder {i} of {n} for Excel files… "
                       "(network folders can take a minute)")
        if ingest_folder_problem(folder):
            continue
        t = time.monotonic()
        files, folder_stats = discover_excel_files(folder)
        walked[folder] = (files, folder_stats, time.monotonic() - t)
        found[folder] = files
        stats.update(folder_stats)

    if not incremental:
        # Nothing is skipped, so every file found is work — no index needed.
        return walked, {f: FolderPlan(files=len(v), work=len(v))
                        for f, v in found.items()}, True

    plans = _plan_work(found, stats, config=config)
    if plans is None:
        _say(on_phase, "Could not read the processed-file index — counting "
                       "every file found as still to do.")
        plans = {f: FolderPlan(files=len(v), work=len(v), from_index=False)
                 for f, v in found.items()}
    return walked, plans, True


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

    `on_total` is called ONCE, with the whole run's WORK, after the pre-scan
    and before the first file is processed. It used to fire per folder, which
    is why the bar restarted twice during a run; it used to carry every file
    found on disk, which is why a re-run of an ingested share said it had
    hours left when it had minutes.
    """
    started = time.monotonic()
    report = IngestReport(folders_requested=len(folders))
    total_folders = len(folders)
    try:
        markers_before = db.count_skipped_files()
    except Exception:
        markers_before = None
    walked, plans, complete = _prescan(folders, on_phase, cancel,
                                       config=config, incremental=incremental)
    if not complete:
        # Stopped during the scan: nothing was processed, and half a scan is
        # not a denominator to quote afterwards.
        report.cancelled = True
        report.seconds = time.monotonic() - started
        return report
    planned = sum(p.work for p in plans.values())
    known = sum(p.known for p in plans.values())
    report.files_planned = planned
    if progress is not None:
        progress.set_total(planned)
    if on_total is not None:
        on_total(planned)
    if known:
        # Said once, for the whole run, so the size of the job is on screen
        # before the first file opens: the number NOT being done is most of
        # what an incremental re-run is doing.
        _say(on_phase, f"{known:,} file" + ("" if known == 1 else "s")
             + " already in the database will be skipped.")
    for i, folder in enumerate(folders, start=1):
        if cancel is not None and cancel.is_set():
            report.cancelled = True
            break
        if on_folder_start is not None:
            on_folder_start(i, total_folders, folder)
        pre = walked.get(folder)
        result = run_folder(folder, db=db, config=config,
                            incremental=incremental, progress=progress,
                            on_phase=on_phase, cancel=cancel,
                            discovered=None if pre is None else (pre[0], pre[1]),
                            walk_seconds=0.0 if pre is None else pre[2],
                            plan=plans.get(folder))
        report.results.append(result)
        if on_folder_done is not None:
            on_folder_done(result)
        if result.cancelled or (cancel is not None and cancel.is_set()):
            report.cancelled = True
            break
    if markers_before is not None:
        try:
            report.marked_unreadable = max(
                0, db.count_skipped_files() - markers_before)
        except Exception:
            report.marked_unreadable = 0
    report.seconds = time.monotonic() - started
    return report
