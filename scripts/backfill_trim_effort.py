"""One-time, idempotent backfill of derivable trim-effort metrics.

Fills NULL rows ONLY, from data already in the row:
  - untrimmed_error_max     = max(|untrimmed_errors|)
  - untrimmed_rms_error     = rms(untrimmed_errors)
  - resistance_change       = trimmed_resistance - untrimmed_resistance
  - resistance_change_percent = change / untrimmed_resistance * 100

Does NOT touch trim_pass_count (not derivable -- needs a reprocess).
Safe to run repeatedly. It WRITES, with no dry run -- back up first.

    python scripts/backfill_trim_effort.py path/to/analysis.db

The database path is REQUIRED: this used to fall back to data/analysis.db, the
production database, which only the app may open without being named.
"""
import argparse
import json
import math
import os
import sqlite3
import sys


def _stats(raw):
    try:
        arr = [float(x) for x in json.loads(raw) if x is not None]
    except Exception:
        return None, None
    arr = [x for x in arr if not math.isnan(x)]
    if not arr:
        return None, None
    emax = max(abs(x) for x in arr)
    rms = math.sqrt(sum(x * x for x in arr) / len(arr))
    return emax, rms


def backfill_trim_effort(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    updated = 0

    # --- error_max / rms from the untrimmed_errors array ---
    rows = cur.execute(
        "SELECT id, untrimmed_errors FROM track_results "
        "WHERE untrimmed_errors IS NOT NULL "
        "AND (untrimmed_error_max IS NULL OR untrimmed_rms_error IS NULL)"
    ).fetchall()
    for rid, raw in rows:
        emax, rms = _stats(raw)
        if emax is None:
            continue
        cur.execute(
            "UPDATE track_results SET "
            "untrimmed_error_max = COALESCE(untrimmed_error_max, ?), "
            "untrimmed_rms_error = COALESCE(untrimmed_rms_error, ?) "
            "WHERE id = ?",
            (emax, rms, rid),
        )
        updated += cur.rowcount

    # --- resistance_change / percent from the two resistance columns ---
    cur.execute(
        "UPDATE track_results SET "
        "resistance_change = COALESCE(resistance_change, trimmed_resistance - untrimmed_resistance) "
        "WHERE resistance_change IS NULL "
        "AND untrimmed_resistance IS NOT NULL AND trimmed_resistance IS NOT NULL"
    )
    cur.execute(
        "UPDATE track_results SET "
        "resistance_change_percent = COALESCE(resistance_change_percent, "
        "(trimmed_resistance - untrimmed_resistance) / untrimmed_resistance * 100.0) "
        "WHERE resistance_change_percent IS NULL "
        "AND untrimmed_resistance IS NOT NULL AND untrimmed_resistance != 0 "
        "AND trimmed_resistance IS NOT NULL"
    )

    con.commit()
    con.close()
    return updated


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="backfill_trim_effort.py",
                                 description="Fill derivable trim-effort metrics (writes).")
    ap.add_argument("db_path", help="the database to update (required), e.g. data/analysis.db")
    path = ap.parse_args(sys.argv[1:] if argv is None else argv).db_path
    if not os.path.exists(path):
        # sqlite3.connect CREATES a missing file; refuse instead of leaving one behind.
        print(f"FATAL | no database at {path}")
        return 1
    n = backfill_trim_effort(path)
    print(f"Backfilled trim-effort metrics in {path}: {n} array-derived row-updates "
          f"(resistance columns updated in bulk).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
