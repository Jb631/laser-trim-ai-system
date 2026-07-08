"""One-time re-summarize of SmoothnessTrack max/avg/pass under Batch C's magnitude rule.

Background (audit AUDIT_FIX_PLAN_2026-06-01 Batch C / C2):
  smoothness_parser.py was fixed so the GENERIC layout computes
    max_smoothness = max(|v|)   avg_smoothness = mean(|v|)
  matching the Betatronix path. Records stored before that fix used signed max()
  and could pass a failing unit. The Betatronix path itself trusts the file's
  pre-computed Max Deviation cell, which is broader than the charted array.

Strategy:
  For each smoothness_tracks row whose smoothness_data array is present:
    real_max = max(|v|);  real_avg = mean(|v|)
    if stored max < real_max - eps:
        # signed-bug victim: overwrite with magnitude
        max := real_max;  avg := real_avg
        smoothness_pass := (max <= smoothness_spec) when spec set
    else:
        leave alone  (stored >= |max|  -> likely a Betatronix file's authoritative
                      Max Dev that's broader than the array, or already-magnitude)

Does NOT touch status (track-level status reflects more than smoothness; let the
app/analyst re-derive when reviewed).

Safe to run repeatedly. Usage:
    python scripts/resummarize_smoothness.py            # default DB
    python scripts/resummarize_smoothness.py --dry-run
    python scripts/resummarize_smoothness.py --db /path/to/analysis.db
"""
import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "analysis.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    rows = cur.execute(
        "SELECT id, smoothness_data, max_smoothness, avg_smoothness, "
        "smoothness_spec, smoothness_pass "
        "FROM smoothness_tracks "
        "WHERE smoothness_data IS NOT NULL"
    ).fetchall()

    eps = 1e-9
    updated = unchanged = bad_array = flipped_p2f = flipped_f2p = 0

    for rid, raw, stored_max, _stored_avg, spec, stored_pass in rows:
        try:
            arr = json.loads(raw)
            arr = [float(x) for x in arr if x is not None]
            arr = [x for x in arr if not math.isnan(x)]
        except Exception:
            bad_array += 1
            continue
        if not arr:
            bad_array += 1
            continue

        real_max = max(abs(x) for x in arr)
        real_avg = sum(abs(x) for x in arr) / len(arr)

        if stored_max is not None and stored_max + eps >= real_max:
            unchanged += 1
            continue

        new_pass = None
        if spec is not None and spec > 0:
            new_pass = 1 if real_max <= spec else 0
            old_pass = 1 if (stored_max is not None and stored_max <= spec) else 0
            if old_pass and not new_pass:
                flipped_p2f += 1
            elif not old_pass and new_pass:
                flipped_f2p += 1

        if not args.dry_run:
            cur.execute(
                "UPDATE smoothness_tracks SET "
                "max_smoothness = ?, avg_smoothness = ?, smoothness_pass = ? "
                "WHERE id = ?",
                (real_max, real_avg, new_pass, rid),
            )
        updated += 1

    if not args.dry_run:
        con.commit()
    con.close()

    print(f"Scanned {len(rows)} smoothness_tracks rows.")
    print(f"  updated:    {updated}   (stored max < |max(arr)|; signed-bug victims)")
    print(f"  unchanged:  {unchanged} (stored >= |max(arr)|; Betatronix or already-magnitude)")
    print(f"  bad arrays: {bad_array}")
    print(f"  pass -> fail flips: {flipped_p2f}")
    print(f"  fail -> pass flips: {flipped_f2p}")
    if args.dry_run:
        print("  (DRY RUN — no writes.)")


if __name__ == "__main__":
    main()
