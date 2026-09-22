"""Make ONE self-contained, verified copy of the database -- the safe way to carry it home.

    python scripts/snapshot_db.py data\\analysis.db C:\\Users\\you\\OneDrive\\analysis_2026-09-23.db

Close the app first. Then run this, let OneDrive show the new file as synced, and copy
that ONE file down at the other end.

Why it exists (2026-09-22): a finished 21.8-hour rebuild was carried home through
OneDrive by syncing the `data` folder. The small log files arrived straight away; the
3.75 GB database had not finished uploading, so what came down was the previous
version of the database next to today's logs -- and its `-wal` journal came from a
different moment again, so SQLite refused the pair as "malformed". Nothing was
corrupt. It was three files that belong together arriving at three different times.

A live SQLite database is up to three files: `analysis.db`, plus `analysis.db-wal`
(committed changes not yet folded into the main file) and `analysis.db-shm`. Copy them
separately and you can get a mismatched set. This uses SQLite's own backup API, which
reads everything the database contains -- journal included -- and writes it into ONE
new file with no journal at all. Then it proves the copy: an integrity check, and the
row counts of the tables that matter, compared against the original.

The original is opened READ-ONLY and is never written.
"""
import shutil
import sqlite3
import sys
import time
from pathlib import Path

# The tables worth counting on both sides. A copy that matches on these has carried
# the capture, the final tests and the findings across.
KEY_TABLES = ("analysis_results", "track_results", "trim_passes", "trim_setup",
              "final_test_results", "process_findings", "processed_files")


def _counts(conn):
    have = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    return {t: conn.execute(f'select count(*) from "{t}"').fetchone()[0]
            for t in KEY_TABLES if t in have}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])

    if not src.exists():
        print(f"no such database: {src}")
        return 2
    # Checked BEFORE "already exists": the source always exists, so in the other order
    # this message could never be reached and the refusal would name the wrong reason.
    if dst.resolve() == src.resolve():
        print("REFUSED: the snapshot cannot be the database itself.")
        return 2
    if dst.exists():
        print(f"REFUSED: {dst} already exists. Pick a new name -- this never overwrites.")
        return 2
    if not dst.parent.is_dir():
        print(f"no such folder: {dst.parent}")
        return 2
    need = src.stat().st_size * 1.1
    free = shutil.disk_usage(dst.parent).free
    if free < need:
        print(f"REFUSED: {free/1e9:.1f} GB free at {dst.parent}, need ~{need/1e9:.1f} GB.")
        return 2

    wal = src.with_name(src.name + "-wal")
    print(f"source:   {src}  {src.stat().st_size/1e9:.2f} GB"
          + (f"  (+ journal {wal.stat().st_size/1e6:.1f} MB, which WILL be included)"
             if wal.exists() and wal.stat().st_size else ""))
    print(f"snapshot: {dst}\n")

    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)     # read-only, always
    try:
        before = _counts(source)
    except sqlite3.DatabaseError as e:
        print(f"SQLite refuses the source: {e}")
        print("If it has a -wal beside it that came from another copy or sync, the pair")
        print("may not belong together -- scripts/verify_rebuild.py explains how to test.")
        return 2

    snap = sqlite3.connect(str(dst))
    t0 = time.monotonic()
    last = [0.0]

    def progress(status, remaining, total):
        done = (total - remaining) / total if total else 1.0
        if time.monotonic() - last[0] > 5 or remaining == 0:
            last[0] = time.monotonic()
            print(f"   copying ... {done*100:5.1f}%  ({time.monotonic()-t0:5.0f}s)", flush=True)

    try:
        source.backup(snap, pages=16384, progress=progress)
        # The copy carries the source's header, which marks it as a WAL database; opened
        # like that it would grow its own -wal and -shm on the other machine. DELETE
        # mode makes it one plain file that means the same thing wherever it lands.
        snap.execute("PRAGMA journal_mode=DELETE")
        snap.commit()
    finally:
        snap.close()
        source.close()

    # ---- prove it -----------------------------------------------------------------
    check = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
    try:
        ok = check.execute("PRAGMA quick_check").fetchone()[0]
        mode = check.execute("PRAGMA journal_mode").fetchone()[0]
        after = _counts(check)
    finally:
        check.close()

    print(f"\nintegrity: {ok}    journal mode: {mode}")
    print(f"{'table':20s} {'original':>12s} {'snapshot':>12s}")
    mismatch = []
    for t in KEY_TABLES:
        a, b = before.get(t), after.get(t)
        if a is None and b is None:
            continue
        flag = "" if a == b else "   <-- DIFFERENT"
        if a != b:
            mismatch.append(t)
        print(f"{t:20s} {a if a is not None else '-':>12,} {b if b is not None else '-':>12,}{flag}"
              if isinstance(a, int) and isinstance(b, int)
              else f"{t:20s} {str(a):>12s} {str(b):>12s}{flag}")

    stray = [p for p in (dst.with_name(dst.name + "-wal"), dst.with_name(dst.name + "-shm"))
             if p.exists()]
    if ok != "ok" or mismatch or mode != "delete" or stray:
        print("\nTHE SNAPSHOT IS NOT GOOD -- do not carry it home:")
        if ok != "ok":
            print(f"  - integrity check said {ok!r}")
        if mismatch:
            print(f"  - row counts differ on {', '.join(mismatch)}")
        if mode != "delete":
            print(f"  - journal mode is {mode!r}, not a single self-contained file")
        if stray:
            print(f"  - it left {', '.join(p.name for p in stray)} beside it")
        return 1

    size = dst.stat().st_size
    print(f"\nGOOD: one file, {size/1e9:.2f} GB ({size:,} bytes), took {time.monotonic()-t0:.0f}s.")
    print("Carry THAT file only. At the other end, before using it, check its size is")
    print(f"exactly {size:,} bytes -- OneDrive shows a file before it has finished arriving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
