"""Did the fresh rebuild land what it was supposed to? READ-ONLY.

    python scripts/verify_rebuild.py data/analysis.db

Opens the database read-only (`mode=ro`) so it is safe to point at the real one --
it cannot write, and it does not need a copy.

The 2026-09-20 rebuild was pushed with four fixes to what gets STORED, and each made
a prediction. This checks them, so "the rebuild finished" can become "the rebuild did
what it said". Anything it cannot check it says so, rather than passing quietly.
"""
import sqlite3
import sys
from pathlib import Path


def q1(c, sql, default=None):
    try:
        r = c.execute(sql).fetchone()
        return default if r is None else r[0]
    except sqlite3.Error:
        return default


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"no such database: {p}")
        return 2
    size = p.stat().st_size
    print(f"{p}  {size/1e9:.2f} GB")
    wal, shm = p.with_suffix(p.suffix + "-wal"), p.with_suffix(p.suffix + "-shm")
    if wal.exists():
        print(f"   beside it: {wal.name} {wal.stat().st_size/1e6:.1f} MB")
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    except sqlite3.DatabaseError as e:
        # "database disk image is malformed" here usually is NOT corruption. A -wal
        # belongs to ONE version of its database; copy the two at different moments --
        # a cloud-sync folder is the classic way -- and SQLite correctly refuses the
        # mismatched pair. The main file is often perfectly good on its own.
        print(f"\n   SQLite refuses this database: {e}")
        if wal.exists():
            print("   There IS a -wal beside it. That pair may simply not belong together:")
            print("   a journal only matches the exact database it was written for, and a")
            print("   sync or a copy made while the app was running can mix two versions.")
            print("   To test, on a COPY (never the original):")
            print(f"      move {wal.name} and {shm.name} aside, then run this again.")
            print("   If it reads fine without them, the database is intact and you have")
            print("   lost only whatever had not been checkpointed out of that journal.")
        else:
            print("   No -wal beside it, so this looks like real damage. Do not write to it.")
        return 2
    print()
    problems = []

    # --- 1. the capture tables exist and are filled -------------------------
    print("1. The capture (why the rebuild was run at all)")
    for t, what in (("trim_passes", "one row per cut, with that pass's sweep and recipe"),
                    ("trim_setup", "one row per file, 46 setup parameters")):
        if t not in tables:
            print(f"   MISSING  {t:12s} -- {what}")
            problems.append(f"{t} table absent: this is NOT a post-capture rebuild")
        else:
            n = q1(c, f"select count(*) from {t}", 0)
            print(f"   {n:>9,}  {t:12s} -- {what}")
            if n == 0:
                problems.append(f"{t} exists but is EMPTY")
    if "trim_passes" in tables:
        pos = q1(c, "select count(*) from trim_passes where cut_lengths is not null "
                    "and cut_lengths != 'null'", 0)
        print(f"   {pos:>9,}  passes carrying per-POSITION data (laser 2 and 3 only)")

    # --- 2. what the four fixes predicted ----------------------------------
    print("\n2. The four storage fixes, against what they predicted")
    untrimmed = q1(c, "select count(*) from track_results where status='UNTRIMMED'", 0)
    print(f"   UNTRIMMED tracks: {untrimmed:,}   (predicted ~1,100 MORE than the old"
          f" database's 145; no-cut files stop being flawless passes)")
    if untrimmed < 500:
        problems.append(f"only {untrimmed:,} UNTRIMMED tracks -- the no-cut fix may not be in")

    errs = q1(c, "select count(*) from analysis_results where overall_status like '%ERROR%'", 0)
    print(f"   ERROR rows:       {errs:,}   (old database carried 112 junk files)")

    ft = q1(c, "select count(*) from final_test_results", 0)
    print(f"   final-test rows:  {ft:,}")

    # --- 3. yield per laser, which the no-cut fix moves ---------------------
    print("\n3. Trim yield by laser (the no-cut fix lowers it where fake passes were)")
    lab = {"A": "laser 2 (DLTS)", "B": "laser 1 (LTS)", "C": "laser 3 (LTS3)"}
    try:
        rows = list(c.execute("""
            select a.system,
                   count(*) n,
                   sum(case when t.linearity_pass then 1 else 0 end) ok
            from track_results t join analysis_results a on a.id = t.analysis_id
            where t.linearity_pass is not null group by a.system order by n desc"""))
        for s, n, ok in rows:
            print(f"   {lab.get(s, s):16s} {ok:>8,} / {n:>8,} = {ok/n*100:5.1f}% left the laser in spec")
    except sqlite3.Error as e:
        print(f"   could not compute: {e}")

    # --- 4. the findings the app should now be able to show -----------------
    print("\n4. Process findings")
    if "process_findings" not in tables:
        print("   MISSING  process_findings -- the findings engine has never run here")
        problems.append("process_findings table absent")
    else:
        nf = q1(c, "select count(*) from process_findings", 0)
        print(f"   {nf:,} findings stored")
        try:
            for an, n in c.execute("select analyzer, count(*) from process_findings "
                                   "group by analyzer order by 2 desc"):
                print(f"      {an:16s} {n}")
        except sqlite3.Error:
            pass
        if nf == 0:
            print("      none yet -- Settings > Database > Refresh process findings")

    print("\n" + "=" * 66)
    if problems:
        print(f"{len(problems)} thing(s) to look at:")
        for x in problems:
            print(f"  - {x}")
        return len(problems)
    print("Everything this script can check looks right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
