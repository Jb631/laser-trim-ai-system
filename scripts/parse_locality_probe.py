"""Is the share (or the scanner sitting in front of it) making PARSING slow?

    python scripts/parse_locality_probe.py "\\\\192.168.66.9\\...\\DLTS" 25

Takes N files, times `process_file` on them WHERE THEY LIVE, copies the very same
files to local disk, and times them again. Same files, same code, same machine --
the only thing that changes is which disk the workbook is opened from.

Why this exists (2026-09-21): the work laptop parses at 231-293 ms/file where a Mac
manages 67.8 ms on comparable files, and a 4x CPU gap on a 48 GB laptop is not
credible. `raw read` prices ONE read of the bytes, but parsing opens the workbook
again through pandas and walks a dozen sheets, so anything that charges per OPEN --
SMB round trips, on-access virus scanning -- lands in the parse number instead of
the read number, where nobody thinks to look for it.

Reads the source files and writes only to a throwaway database, so it is safe to
run while an ingest is going.

    local much faster  -> mirror the share first (tracker A2) and rebuild from the
                          copy; the network or the scanner is charging per open
    no difference      -> the machine really is this slow at parsing; the lever is
                          processes instead of threads (tracker A3)
"""
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _time_parse(proc, files):
    """(total seconds, per-file ms list, files that produced a result)."""
    per, ok = [], 0
    t0 = time.perf_counter()
    for f in files:
        t = time.perf_counter()
        try:
            r = proc.process_file(f)
            if r is not None:
                ok += 1
        except Exception:
            pass                      # a file that refuses to parse still costs time
        per.append((time.perf_counter() - t) * 1000.0)
    return time.perf_counter() - t0, per, ok


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    if not src.is_dir():
        print(f"not a directory: {src}")
        return 2

    files = [p for p in sorted(src.rglob("*.xls*")) if not p.name.startswith("~$")][:n]
    if not files:
        print(f"no workbooks under {src}")
        return 2
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"{len(files)} files, {total_mb:.1f} MB, from {src}")

    import logging
    logging.disable(logging.WARNING)       # per-file skip notices would bury the numbers
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg

    tmp = Path(tempfile.mkdtemp(prefix="locality_"))
    try:
        db = mgr.DatabaseManager(tmp / "probe.db")
        mgr._db_manager = db               # BOTH globals, or get_database() builds
        dbpkg._db_manager = db             # one at the CONFIGURED path.
        proc = Processor(use_ml=False)
        proc.process_file(files[0])        # warm the imports, not the measurement

        print("\n1. parsing them WHERE THEY LIVE ...")
        remote_s, remote_per, remote_ok = _time_parse(proc, files)

        print("2. copying the same files to local disk ...")
        local_dir = tmp / "local"
        local_dir.mkdir()
        t = time.perf_counter()
        local_files = []
        for f in files:
            dst = local_dir / f.name
            shutil.copy2(f, dst)
            local_files.append(dst)
        copy_s = time.perf_counter() - t

        print("3. parsing the LOCAL copies ...")
        local_s, local_per, local_ok = _time_parse(proc, local_files)

        print(f"\n   where they live : {remote_s:6.2f}s  {remote_s/len(files)*1000:7.1f} ms/file"
              f"  (median {statistics.median(remote_per):6.1f})  {remote_ok} parsed")
        print(f"   local copies    : {local_s:6.2f}s  {local_s/len(files)*1000:7.1f} ms/file"
              f"  (median {statistics.median(local_per):6.1f})  {local_ok} parsed")
        print(f"   copying cost    : {copy_s:6.2f}s  {copy_s/len(files)*1000:7.1f} ms/file"
              f"  ({total_mb/max(copy_s,1e-9):.1f} MB/s)")

        if local_ok != remote_ok:
            print(f"\n   NOTE: {remote_ok} parsed remotely vs {local_ok} locally -- not the same "
                  "work both times, so read the medians with that in mind.")
        speedup = remote_s / local_s if local_s else 0.0
        print(f"\n   local is {speedup:.1f}x faster")
        if speedup >= 1.5:
            per_file_saved = (remote_s - local_s) / len(files)
            print("   >> THE FILE'S LOCATION IS THE COST, not the parser. Mirror the share")
            print("      and rebuild from the copy (tracker A2). Even including the copy,")
            print(f"      that saves ~{per_file_saved*1000:.0f} ms/file"
                  f" = ~{per_file_saved*250000/3600:.1f} h over 250,000 files.")
        elif speedup <= 1.15:
            print("   >> Location is NOT the cost: the machine really is this slow at")
            print("      parsing. The lever is processes instead of threads (tracker A3),")
            print("      not moving the files.")
        else:
            print("   >> Some of the cost is location, but not most of it. Worth mirroring")
            print("      only if a rebuild is otherwise too slow to finish.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
