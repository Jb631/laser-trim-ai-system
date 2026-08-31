"""One-time repair: recover trim clock times, then re-point every FT link.

Why this exists (2026-08-30)
----------------------------
A unit that fails linearity is re-trimmed until it passes. Each attempt writes
its own file carrying the same calendar date, and the only record of their
ORDER is the clock time in the filename ("..._10-01 AM", "..._11-17 AM").
`_extract_date_from_filename` used to parse the date and throw that time away,
so every same-day attempt tied on `file_date`; `_find_matching_trim` ordered
candidates by `file_date` alone and linked an arbitrary one — in practice the
earliest-ingested.

When the arbitrary pick was a failing early attempt, the pair scored as an
"overkill": the trim station blamed for rejecting a unit that final test then
passed. On the work database that is 124 phantom overkills and 32 hidden
escapes over 12 months (6607: 415 -> 338, 8340-1: 145 -> 132).

The parser now keeps the time, so newly-ingested files are correct. Rows
already in the database still sit at midnight — this script fixes them:

  1. backfill_trim_file_times()  re-reads the time out of `filename`
  2. rematch_final_tests()       re-points linked_trim_id at the day's LAST
                                 attempt, now that the times order them

Both steps are idempotent; running twice is harmless.

Usage
-----
    python scripts/repair_trim_ft_links.py path/to/analysis.db
    python scripts/repair_trim_ft_links.py path/to/analysis.db --yes

This REWRITES `analysis_results.file_date` and `final_test_results`' link
columns in place. Take a backup first. Without --yes it asks before writing.
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    assume_yes = "--yes" in sys.argv or "-y" in sys.argv
    if not args:
        print(__doc__)
        return 2

    db_path = Path(args[0])
    if not db_path.exists():
        # Checked BEFORE DatabaseManager: constructing it CREATES an empty
        # database, and "repairing" a fresh empty file reports success for
        # nothing at all.
        print(f"FATAL | no database at {db_path}")
        return 1

    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(db_path)
    with db.session() as s:
        from sqlalchemy import func, text
        n_mid = s.execute(text(
            "SELECT COUNT(*) FROM analysis_results WHERE file_date IS NOT NULL "
            "AND strftime('%H:%M:%S', file_date) = '00:00:00'")).scalar()
        n_ft = s.execute(text("SELECT COUNT(*) FROM final_test_results")).scalar()

    print(f"database : {db_path}  ({db_path.stat().st_size / 1e9:.1f} GB)")
    print(f"trim rows still at midnight : {n_mid:,}")
    print(f"final-test rows to rematch  : {n_ft:,}")
    print("this rewrites file_date and linked_trim_id IN PLACE — back up first.")
    if not assume_yes:
        try:
            if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("aborted, nothing written")
                return 1
        except EOFError:
            print("aborted (no tty); re-run with --yes to skip this prompt")
            return 1

    t0 = time.time()
    stats = db.backfill_trim_file_times()
    print(f"[1/2] times recovered: {stats['updated']:,} updated, "
          f"{stats['no_time_in_filename']:,} had no time in the filename "
          f"({time.time() - t0:.1f}s)")

    t1 = time.time()
    rm = db.rematch_final_tests()
    print(f"[2/2] links repaired: {rm['updated_matches']:,} re-pointed, "
          f"{rm['new_matches']:,} newly linked, {rm['unchanged']:,} unchanged "
          f"({time.time() - t1:.1f}s)")

    print(f"done in {time.time() - t0:.1f}s — verify with:\n"
          f"    python scripts/app_qa_sweep.py {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
