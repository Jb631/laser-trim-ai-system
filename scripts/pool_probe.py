"""Would PROCESSES actually beat threads on this machine? Parse only, nothing written.

    python scripts/pool_probe.py "\\\\192.168.66.9\\...\\DLTS" 48

Parses the same files four ways -- one at a time, then a thread pool, then a process
pool at two sizes -- and prints files/sec for each. Nothing is written: no database is
opened, no file is modified, so it is safe to run while an ingest is going.

Why (2026-09-21): the work laptop runs the real ingest at ~1 file/sec while a single
parse loop on the same machine manages ~2.9. Parsing there is CPU-bound -- proven by
`parse_locality_probe`, where local copies were no faster than the share -- and four
Python threads cannot run bytecode at the same time, so the thread pool cannot help
however many workers it is given. Processes can. This measures whether that is true
HERE before anyone rewrites the ingest around it.

Read the last block: if processes win by 3x or more, rebuilding the pool is worth the
work. If they do not, the per-file cost is the target instead.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def parse_one(path_str: str) -> bool:
    """Parse one workbook and throw the result away.

    Top level and importable by name: a Windows worker process re-imports this
    module rather than forking, so a closure or a local function cannot be sent.
    Imports live INSIDE the function so the parent does not pay for them twice
    and a worker does not inherit a half-built module.
    """
    import logging
    logging.disable(logging.WARNING)
    from laser_trim_analyzer.core.parser import ExcelParser
    try:
        ExcelParser().parse_file(Path(path_str))
        return True
    except Exception:
        return False


def _run(label, fn, files):
    t = time.perf_counter()
    ok = sum(1 for r in fn(files) if r)
    el = time.perf_counter() - t
    print(f"  {label:<26} {el:6.2f}s  {el/len(files)*1000:7.1f} ms/file  "
          f"= {len(files)/el:5.2f} files/sec   ({ok}/{len(files)} parsed)")
    return len(files) / el


def _serial(files):
    return [parse_one(f) for f in files]


def _threads(n):
    def go(files):
        with ThreadPoolExecutor(max_workers=n) as ex:
            return list(ex.map(parse_one, files))
    return go


def _procs(n):
    def go(files):
        with ProcessPoolExecutor(max_workers=n) as ex:
            return list(ex.map(parse_one, files, chunksize=1))
    return go


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    if not src.is_dir():
        print(f"not a directory: {src}")
        return 2
    files = [str(p) for p in sorted(src.rglob("*.xls*")) if not p.name.startswith("~$")][:n]
    if len(files) < 8:
        print(f"need at least 8 workbooks under {src}, found {len(files)}")
        return 2

    import os
    cpus = os.cpu_count() or 1
    print(f"{len(files)} files from {src}")
    print(f"{cpus} logical CPUs\n")

    parse_one(files[0])                       # warm the imports in the parent

    print("parse only, nothing written:")
    serial = _run("1 at a time", _serial, files)
    _run("4 threads (what runs now)", _threads(4), files)
    p4 = _run("4 processes", _procs(4), files)
    p8 = _run(f"{min(8, cpus)} processes", _procs(min(8, cpus)), files)

    best = max(p4, p8)
    print(f"\n  best process pool is {best/serial:.1f}x the single loop"
          f" ({best:.1f} files/sec)")
    remaining = 244_000
    print(f"  at that rate, {remaining:,} files would take {remaining/best/3600:.1f} hours"
          f" (against {remaining/serial/3600:.1f} at one at a time)")
    if best / serial >= 3.0:
        print("  >> PROCESSES ARE THE FIX. Worth rebuilding the ingest pool around them.")
    elif best / serial >= 1.5:
        print("  >> Processes help, but less than hoped. Weigh the rebuild against")
        print("     simply running the ingest in chunks overnight.")
    else:
        print("  >> Processes do NOT help here. The target is the per-file cost,")
        print("     not the pool -- do not rewrite the ingest on this evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
