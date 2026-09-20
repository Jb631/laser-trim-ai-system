"""Find out WHICH layer is slow when ingest crawls.

Times four layers on the same real files, so the slow one is obvious:

    1. raw read      -- open the file, read the bytes, close. No parsing.
                        Slow here = disk, network share, or endpoint security
                        scanning every file open. Nothing in this app can fix it.
    2. parse         -- the Excel parse and analysis, in memory.
    3. parse + save  -- the above, written to a THROWAWAY database.
    4. worker choice -- what the pool would pick on this machine right now.

Reference, measured 2026-09-20 on the developer's Mac against real files:

    raw read ~1 ms | parse ~50 ms | parse+save ~56 ms | 18 files/sec

If this machine's parse figure is close to that but the owner sees ~0.5
files/sec in the app, the cost is NOT in parsing -- look at layer 1 and at
the worker count.

SAFETY: never opens the real database. Writes only to a temp file, removed
on exit. Pass a folder of .xls files:

    python scripts/ingest_speed_probe.py "//192.168.66.9/BTXData/.../DLTS" 25
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _files(root: Path, n: int):
    out = []
    for p in root.rglob("*.xls"):
        if p.name.startswith("~$"):
            continue
        out.append(p)
        if len(out) >= n:
            break
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    if not root.is_dir():
        print(f"not a folder: {root}")
        return 2

    print(f"scanning {root} ...", flush=True)
    t = time.perf_counter()
    files = _files(root, n)
    kind = "NETWORK PATH" if str(root)[:2] in ("//", chr(92) * 2) else "local path"
    print(f"  found {len(files)} files in {time.perf_counter()-t:.1f}s  ({kind})")
    if not files:
        print("no .xls files found")
        return 1

    # ---- layer 1: raw read -------------------------------------------------
    t = time.perf_counter()
    total_bytes = 0
    for f in files:
        with open(f, "rb") as fh:
            total_bytes += len(fh.read())
    raw = time.perf_counter() - t
    mb = total_bytes / 1024 / 1024
    print(f"\n1. raw read      {raw:7.2f}s  {raw/len(files)*1000:8.1f} ms/file"
          f"   ({mb:.0f} MB, {mb/raw:.1f} MB/s)")

    # ---- layers 2 and 3 ----------------------------------------------------
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg

    tmp = Path(tempfile.mkdtemp(prefix="speedprobe_"))
    try:
        db = mgr.DatabaseManager(tmp / "probe.db")
        mgr._db_manager = db          # BOTH globals, or get_database() would
        dbpkg._db_manager = db        # build one at the CONFIGURED path.
        proc = Processor(use_ml=False)
        proc.process_file(files[0])   # warm up imports

        parse = save = 0.0
        for f in files:
            t = time.perf_counter()
            r = proc.process_file(f)
            parse += time.perf_counter() - t
            if r is not None:
                t = time.perf_counter()
                db.save_analysis(r)
                save += time.perf_counter() - t
        print(f"2. parse         {parse:7.2f}s  {parse/len(files)*1000:8.1f} ms/file")
        print(f"3. save          {save:7.2f}s  {save/len(files)*1000:8.1f} ms/file")
        tot = parse + save
        print(f"   parse+save    {tot:7.2f}s  {tot/len(files)*1000:8.1f} ms/file"
              f"  = {len(files)/tot:.1f} files/sec")

        # ---- layer 4: worker choice ---------------------------------------
        print()
        try:
            import psutil
            avail = psutil.virtual_memory().available / 1024**3
            total_gb = psutil.virtual_memory().total / 1024**3
            print(f"4. memory        {avail:.1f} GB free of {total_gb:.1f} GB")
        except Exception:
            print("4. memory        psutil unavailable -> pool would use 2 workers")
        print(f"   workers the pool would pick RIGHT NOW: "
              f"{proc._get_safe_worker_count(len(files))}  (hard cap is 4)")

        # ---- the verdict ---------------------------------------------------
        print("\n--- where the time goes ---")
        whole = raw + tot
        for name, v in (("raw read", raw), ("parse", parse), ("save", save)):
            print(f"  {name:<10} {v/whole*100:5.1f}%")
        print(f"\n  one worker would manage {len(files)/whole:.1f} files/sec here.")
        if raw / len(files) > 0.3:
            print("  >> RAW READ IS THE BOTTLENECK. Not the app: that is the")
            print("     disk, the network share, or endpoint security scanning")
            print("     every file open.")
        elif tot / len(files) > 0.3:
            print("  >> PARSING IS THE BOTTLENECK -- this machine is much slower")
            print("     than the reference. Send this output back.")
        else:
            print("  >> Neither layer is slow. If the app still crawls, the cost")
            print("     is in the GUI/worker layer, not in reading or parsing.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
