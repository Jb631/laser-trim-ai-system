"""Would PROCESSES actually beat threads on this machine? Nothing real is written.

    python scripts/pool_probe.py "\\\\192.168.66.9\\...\\DLTS" 200 [--specs-from DB]

Runs the same files four ways -- one at a time, a thread pool, and a process pool at
two sizes -- and prints files/sec for each. It does what the ingest does to a file
(`Processor.process_file`: parse AND analyse). That is not database-free: the processor
looks up each model's spec, SAVES final-test and smoothness files and marks refused ones,
all through `get_database()`. So every process -- this one and each pool worker -- gets
its own throwaway database in a temporary folder, deleted at the end. Nothing touches the
app's database, and it is safe to run while an ingest is going.

Two mistakes the first version of this script made, both corrected here after the
2026-09-21 run at work:

* **It charged worker startup to throughput.** Windows SPAWNS workers instead of
  forking, so each one re-imports pandas from scratch with the endpoint scanner
  reading every module. Over 48 files, 8 workers parsed ~6 files each and paid
  seconds of import -- which made 8 processes look SLOWER than 4 (10.2 vs 17.7
  files/sec) and produced a "processes help less than hoped" verdict that was an
  artefact of the sample size. The pool is now warmed first and startup is timed and
  reported on its own line.
* **It measured bare parsing, not the job.** `ExcelParser.parse_file` came out at
  98.7 ms/file on the work laptop while the ingest's own `process_file` cost
  173-293 ms. Roughly half the real per-file cost is the ANALYSIS, and a probe that
  leaves it out is measuring the wrong thing.

A third, found 2026-09-25 (spec 2026-09-25-ingest-speed-design.md F8, ruling 3):

* **It timed the analysis WITHOUT model specs.** A fresh throwaway database has an empty
  model_specs table, and the ingest's analysis is about half again as expensive with the
  specs it really has (DLTS on the Mac: 67 -> 99 ms/file, same files). So the 339 ms/file
  measured at work was the cheap analysis, not the ingest's. Every throwaway database now
  gets the model specs of a NAMED one (`--specs-from`, default this checkout's
  data/analysis.db, opened read-only).

Use enough files that startup stops mattering: 200+ is a fair test, 48 is not.
"""
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _probe_specs import copy_model_specs, describe, specs_from_argv  # noqa: E402

_PROC = None
_SCRATCH_DIR = None         # this process's throwaway-database folder; see _use_scratch_db
_SPECS_FROM = None          # the named database whose model specs every throwaway one gets
_SPECS_COPIED = 0


def _use_scratch_db(folder: str, specs_from=None) -> None:
    """Where this process's throwaway database goes, and whose model specs it gets. main()
    calls it for itself and hands it to every pool worker as the initializer, so each worker
    writes only there and analyses with the same specs as the ingest."""
    global _SCRATCH_DIR, _SPECS_FROM
    _SCRATCH_DIR = folder
    _SPECS_FROM = specs_from


def _processor():
    """One Processor per worker, built on first use and kept.

    Built lazily rather than at import: a spawned worker imports this module before
    it is given any work, and constructing it there would be paid even by a pool
    that never receives a file.
    """
    global _PROC, _SPECS_COPIED
    if _PROC is None:
        import logging
        logging.disable(logging.WARNING)
        from laser_trim_analyzer.core.processor import Processor
        from laser_trim_analyzer.database import manager as mgr
        import laser_trim_analyzer.database as dbpkg
        # Before the Processor: everything it does with a database goes through
        # get_database(), and with nothing injected that is the app's default -- refused
        # outside the app. One file per process: pool workers are separate processes.
        path = Path(_SCRATCH_DIR) / f"probe_{os.getpid()}.db"
        db = mgr.DatabaseManager(path)
        if _SPECS_FROM:
            _SPECS_COPIED = copy_model_specs(_SPECS_FROM, path)
        mgr._db_manager = db            # BOTH globals, or get_database() builds one at
        dbpkg._db_manager = db          # the CONFIGURED path.
        _PROC = Processor(use_ml=False)
    return _PROC


def warm(_ignored=None) -> bool:
    """Force this worker to import its modules and build its Processor."""
    _processor()
    return True


def work(path_str: str) -> bool:
    """Exactly what the ingest does to one file, on this process's throwaway database."""
    try:
        return _processor().process_file(Path(path_str)) is not None
    except Exception:
        return False                    # a file that refuses still costs its time


def _time_pool(label, make_pool, files, workers):
    """(startup seconds, throughput files/sec). Startup is NOT counted as throughput."""
    if make_pool is None:                                  # serial
        t = time.perf_counter()
        warm()
        start = time.perf_counter() - t
        t = time.perf_counter()
        ok = sum(1 for f in files if work(f))
        el = time.perf_counter() - t
    else:
        with make_pool() as ex:
            t = time.perf_counter()
            list(ex.map(warm, range(workers)))             # every worker up and imported
            start = time.perf_counter() - t
            t = time.perf_counter()
            ok = sum(1 for r in ex.map(work, files, chunksize=4) if r)
            el = time.perf_counter() - t
    print(f"  {label:<26} {el:6.2f}s  {el/len(files)*1000:7.1f} ms/file  "
          f"= {len(files)/el:5.2f} files/sec   (startup {start:4.1f}s, {ok}/{len(files)} ok)")
    return len(files) / el


def main() -> int:
    specs_from, argv = specs_from_argv(sys.argv, REPO)
    if len(argv) < 2:
        print(__doc__)
        return 2
    src = Path(argv[1])
    n = int(argv[2]) if len(argv) > 2 else 200
    if not src.is_dir():
        print(f"not a directory: {src}")
        return 2
    files = [str(p) for p in sorted(src.rglob("*.xls*")) if not p.name.startswith("~$")][:n]
    if len(files) < 32:
        print(f"need at least 32 workbooks under {src}, found {len(files)}")
        return 2

    cpus = os.cpu_count() or 1
    big = min(8, cpus)
    print(f"{len(files)} files from {src}")
    print(f"{cpus} logical CPUs")
    if len(files) < 150:
        print("  NOTE: fewer than 150 files -- worker startup may still colour the result.")
    print("\nparse AND analyse, on throwaway databases (startup excluded from the rate):")

    with tempfile.TemporaryDirectory(prefix="pool_probe_", ignore_cleanup_errors=True) as scratch:
        _use_scratch_db(scratch, str(specs_from))

        def processes(workers):
            return lambda: ProcessPoolExecutor(max_workers=workers, initializer=_use_scratch_db,
                                               initargs=(scratch, str(specs_from)))
        try:
            serial = _time_pool("1 at a time", None, files, 1)
            print(f"  {describe(_SPECS_COPIED, specs_from)}")
            _time_pool("4 threads (what runs now)", lambda: ThreadPoolExecutor(max_workers=4),
                       files, 4)
            p4 = _time_pool("4 processes", processes(4), files, 4)
            p8 = _time_pool(f"{big} processes", processes(big), files, big)
        finally:
            from laser_trim_analyzer.database import manager as mgr
            mgr.reset_database()        # close this process's file before its folder goes

    best = max(p4, p8)
    remaining = 244_000
    print(f"\n  best process pool is {best/serial:.1f}x the single loop ({best:.1f} files/sec)")
    print(f"  {remaining:,} files: {remaining/best/3600:.1f} hours,"
          f" against {remaining/serial/3600:.1f} at one at a time")
    if best / serial >= 3.0:
        print("  >> PROCESSES ARE THE FIX. Worth rebuilding the ingest pool around them.")
    elif best / serial >= 1.8:
        print("  >> Processes are a real win. Worth building, but they will not alone")
        print("     turn an overnight job into a coffee break.")
    else:
        print("  >> Processes do NOT help enough here. The target is the per-file cost,")
        print("     not the pool -- do not rewrite the ingest on this evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
