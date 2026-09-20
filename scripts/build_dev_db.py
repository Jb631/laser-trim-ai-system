"""Build a throwaway DEVELOPMENT database from a folder of real files.

For working on analyzers without the full rebuild: point it at a slice of the
share (see scripts/pull_model_slice.ps1) or at the sample base, and it runs
the real pipeline -- parse, grade, save, final-test matching -- into a fresh
database that carries the new trim_passes / trim_setup tables.

    python scripts/build_dev_db.py "Work Files/home_slice" "Work Files/dev_db/slice.db"

SAFETY: refuses data/analysis.db by resolved path, refuses to overwrite an
existing file, and injects BOTH database globals so nothing on the pipeline
can open the configured (production) database.
"""
import sys
import time
import logging
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    if not src.is_dir():
        print(f"not a folder: {src}")
        return 2
    if out.resolve() == (REPO / "data" / "analysis.db").resolve() or out.name == "analysis.db":
        print("REFUSED: that is the production database. Pick another output path.")
        return 2
    if out.exists():
        print(f"REFUSED: {out} already exists -- delete it yourself if you mean to rebuild.")
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)

    logging.disable(logging.WARNING)          # keep the per-file chatter out of the way
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg

    db = mgr.DatabaseManager(out)
    mgr._db_manager = db                      # BOTH globals, or get_database()
    dbpkg._db_manager = db                    # would open the configured path.
    proc = Processor(use_ml=False)

    files = [p for p in src.rglob("*.xls*") if not p.name.startswith("~$")]
    # Trim files first, final tests last: a final test links to its trim record
    # at save time, so the trim record has to exist already.
    files.sort(key=lambda p: ("test station" in str(p).lower(), str(p)))
    print(f"{len(files):,} files under {src}")

    t0 = time.perf_counter()
    saved = skipped = errors = 0
    for i, f in enumerate(files, 1):
        try:
            r = proc.process_file(f)
            if r is None:
                skipped += 1
            else:
                db.save_analysis(r)
                saved += 1
        except Exception as e:                # one bad file must not end the build
            errors += 1
            if errors <= 5:
                print(f"  ERROR {f.name}: {type(e).__name__}: {e}")
        if i % 250 == 0:
            el = time.perf_counter() - t0
            print(f"  {i:,}/{len(files):,}  {i/el:.1f} files/sec", flush=True)
    el = time.perf_counter() - t0
    print(f"\nsaved {saved:,} | skipped {skipped:,} | errors {errors:,} | "
          f"{el:.0f}s ({len(files)/el:.1f} files/sec)")

    import sqlite3
    c = sqlite3.connect(f"file:{out}?mode=ro", uri=True)
    for table in ("analysis_results", "track_results", "trim_passes", "trim_setup",
                  "final_test_results", "final_test_tracks"):
        try:
            print(f"  {table:<20} {c.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:>8,}")
        except sqlite3.OperationalError as e:
            print(f"  {table:<20} {e}")
    print(f"\n{out}  ({out.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
