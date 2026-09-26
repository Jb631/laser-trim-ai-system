"""A dropped worker comes back when memory recovers (ingest-speed Task 12, A5; spec section 5,
ruling 21).

Before A5 the ingest's memory throttle went one way only: above 90% memory it took a worker away
for the rest of the folder, and nothing ever gave it back -- a folder that met one bad minute two
hours in ran its remaining hours on fewer workers. Throttling is now an in-flight CAP over one
fixed pool (threads or worker processes), asked at every 20-file chunk:

  * above 90% memory, one fewer file in flight (never below one);
  * above 95%, one file at a time -- today's sequential fallback;
  * below 80% on two checks running, one more -- up to the pool's size.

The 10-point gap and the two-check wait keep one reading from making it oscillate, and every
change is logged once ("workers 7 -> 8: memory back to 78%"). What it cannot do: an idle worker
process keeps its ~0.27 GB, so when our own workers are the pressure the answer is fewer at start
(ruling 14), not the cap.
"""
import logging
import threading
import time

import pytest

import worker_stubs


def _parallel_config():
    from laser_trim_analyzer.config import Config
    cfg = Config()
    cfg.processing.turbo_mode_threshold = 1
    return cfg


class _Collect:
    def __init__(self):
        self.outcomes = []

    def add(self, outcome):
        self.outcomes.append(outcome)
        return outcome.result

    def tick(self):
        pass

    def flush(self):
        pass


# (memory %, the cap after it, the calm checks after it), from a cap of 8 on a pool of 8.
SCRIPT = [
    (50.0, 8, 1),     # calm, and already the pool's size: it holds
    (91.0, 7, 0),     # above 90: one fewer
    (92.0, 6, 0),
    (85.0, 6, 0),     # between 80 and 90: it holds, and the calm count starts again
    (79.0, 6, 1),     # one calm check is not enough...
    (85.0, 6, 0),     # ...and an interruption starts it again
    (79.0, 6, 1),
    (79.0, 7, 0),     # two calm checks running: one more
    (79.0, 7, 1),
    (79.0, 8, 0),
    (79.0, 8, 1),     # the pool's size: never above it, however long memory stays calm
    (79.0, 8, 2),
    (79.0, 8, 2),
    (96.0, 1, 0),     # above 95: one file at a time
    (93.0, 1, 0),     # never below one
    (70.0, 1, 1),
    (70.0, 2, 0),
    (90.0, 2, 0),     # exactly 90 is not ABOVE 90: it holds
    (95.0, 1, 0),     # exactly 95 is above 90, not above 95: one fewer
    (80.0, 1, 0),     # exactly 80 is not BELOW 80: it holds
    (79.9, 1, 1),
    (79.9, 2, 0),
]


def test_the_cap_follows_a_scripted_memory_sequence():
    from laser_trim_analyzer.core.processor import next_cap
    k, calm = 8, 0
    for i, (percent, want_k, want_calm) in enumerate(SCRIPT):
        k, calm = next_cap(k, percent, calm, 8)
        assert (k, calm) == (want_k, want_calm), f"step {i}: {percent}% -> {k}, {calm}"


def _scripted(monkeypatch, readings):
    """The memory probe the dispatch reads, scripted: one reading per call, the last repeated."""
    from laser_trim_analyzer.core import processor
    values = list(readings)
    calls = []

    def probe():
        calls.append(1)
        return values[min(len(calls), len(values)) - 1]

    monkeypatch.setattr(processor, "memory_percent", probe)
    return calls


def _in_flight_per_chunk(monkeypatch, pool_class, proc, files):
    """Most files in flight while each 20-file chunk was being handed out."""
    in_flight, out, most = [0], [0], {}
    real_submit = pool_class.submit

    def submit(self, path, disk_stat=None):
        chunk = out[0] // 20
        out[0] += 1
        in_flight[0] += 1
        most[chunk] = max(most.get(chunk, 0), in_flight[0])
        return real_submit(self, path, disk_stat)

    monkeypatch.setattr(pool_class, "submit", submit)

    class _Count(_Collect):
        def add(self, outcome):
            in_flight[0] -= 1
            return super().add(outcome)

    got = list(proc.process_batch(files, incremental=False, writer=_Count()))
    assert len(got) == len(files)
    return most


def test_a_run_whose_memory_recovers_gets_its_workers_back_and_says_so(
        tmp_path, monkeypatch, caplog):
    """On threads (a pool of 4), over a scripted memory sequence -- the start check, then one
    reading per chunk boundary: the cap falls under pressure, comes back once memory has been
    calm for two checks running, and the in-flight count follows it; each change is logged once."""
    from laser_trim_analyzer.core import ingest_worker
    from laser_trim_analyzer.core.processor import _ThreadPool
    from laser_trim_analyzer.database.specs import SpecSnapshot
    monkeypatch.setattr(ingest_worker, "PROCESS_MIN_FILES", 10 ** 9)      # threads, whatever
    _scripted(monkeypatch, [50.0, 91.0, 92.0, 79.0, 79.0, 79.0, 79.0, 50.0])
    proc = worker_stubs.StubProcessor(config=_parallel_config(), snapshot=SpecSnapshot())
    monkeypatch.setattr(proc, "_get_safe_worker_count", lambda n: 4)
    files = worker_stubs.make_files(tmp_path, [f"slow_{i:03d}.xls" for i in range(200)])
    with caplog.at_level(logging.INFO):
        most = _in_flight_per_chunk(monkeypatch, _ThreadPool, proc, files)
    changes = [r.getMessage() for r in caplog.records if r.getMessage().startswith("workers ")]
    assert changes == ["workers 4 → 3: memory at 91%", "workers 3 → 2: memory at 92%",
                       "workers 2 → 3: memory back to 79%", "workers 3 → 4: memory back to 79%"], changes
    # chunk 0 at the full window (20); 1 at 3; 2 and 3 at 2; 4 and 5 at 3; 6 onwards the full 20
    assert proc.last_workers.startswith("4 threads"), proc.last_workers
    assert [most[c] for c in range(10)] == [20, 3, 2, 2, 3, 3, 20, 20, 20, 20], most


def test_worker_processes_are_capped_the_same_way(tmp_path, monkeypatch, caplog):
    """The same cap over a pool of worker processes: under pressure the files of the next chunk
    go out one at a time, and the cap comes back."""
    from laser_trim_analyzer.core import ingest_worker
    from laser_trim_analyzer.database.specs import SpecSnapshot
    monkeypatch.setattr(ingest_worker, "PROCESS_MIN_FILES", 1)
    monkeypatch.setattr(ingest_worker, "MAX_WORKERS", 2)
    _scripted(monkeypatch, [50.0, 91.0, 79.0, 79.0, 50.0])
    proc = worker_stubs.StubProcessor(config=_parallel_config(), snapshot=SpecSnapshot())
    files = worker_stubs.make_files(tmp_path, [f"slow_{i:03d}.xls" for i in range(100)])
    with caplog.at_level(logging.INFO):
        most = _in_flight_per_chunk(monkeypatch, ingest_worker.WorkerPool, proc, files)
    assert worker_stubs.ran_on_processes(proc.last_workers, 2), proc.last_workers
    changes = [r.getMessage() for r in caplog.records if r.getMessage().startswith("workers ")]
    assert changes == ["workers 2 → 1: memory at 91%", "workers 1 → 2: memory back to 79%"], changes
    # chunk 0 out whole; chunk 1 (taken at 91%) and chunk 2 one file at a time, once chunk 0 was
    # back; chunk 3 (taken as memory came back) whole again, beside chunk 2's last file
    assert (most[0], most[1], most[2], most[3]) == (20, 1, 1, 21), most


def test_stop_works_on_the_memory_critical_path_too(tmp_path, monkeypatch):
    """Memory critical when the folder starts: one file at a time, in this process -- and that
    path used to ignore Stop (review of Tasks 9-10)."""
    from laser_trim_analyzer.database.specs import SpecSnapshot
    _scripted(monkeypatch, [97.0])
    proc = worker_stubs.StubProcessor(config=_parallel_config(), snapshot=SpecSnapshot())
    files = worker_stubs.make_files(tmp_path, [f"f{i:03d}.xls" for i in range(40)])
    cancel, seen = threading.Event(), []

    def trip(status):
        if status.status in ("completed", "skipped", "failed"):
            seen.append(status.filename)
            if len(seen) >= 3:
                cancel.set()

    got = list(proc.process_batch(files, progress_callback=trip, incremental=False,
                                  cancel=cancel, writer=_Collect()))
    assert proc.last_workers.startswith("sequential (memory was critical"), proc.last_workers
    assert len(got) == 3, len(got)
