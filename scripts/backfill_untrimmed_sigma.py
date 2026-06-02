#!/usr/bin/env python3
"""One-time backfill of TrackResult.untrimmed_sigma_gradient.

Computes the untrimmed-sweep sigma from the untrimmed arrays already stored in
the DB (untrimmed_positions / untrimmed_errors), using the SAME core.analyzer
logic that processing uses -- so backfilled values match freshly-processed ones.

SAFE: idempotent; only fills rows where untrimmed_sigma_gradient IS NULL and the
untrimmed arrays are present. Never touches any other column. Back up the DB
first anyway.

Usage:
    python3 scripts/backfill_untrimmed_sigma.py            # backfill default DB
    python3 scripts/backfill_untrimmed_sigma.py --dry-run  # count only, no writes
    python3 scripts/backfill_untrimmed_sigma.py --db /path/to/analysis.db
    python3 scripts/backfill_untrimmed_sigma.py --limit 2000   # first N (rehearsal)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from laser_trim_analyzer.core.analyzer import Analyzer, END_POINT_FILTER_COUNT
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import AnalysisResult, TrackResult


def compute_untrimmed_sigma(analyzer, positions, errors, linearity_spec,
                            travel_length, unit_length, model):
    """Replicates core/analyzer.py:219-247 exactly."""
    valid = [
        (p, e) for p, e in zip(positions or [], errors or [])
        if p is not None and e is not None
        and not np.isnan(p) and not np.isnan(e)
    ]
    if len(valid) <= 2 * END_POINT_FILTER_COUNT + 3:
        return None
    up = [p for p, _ in valid]
    ue = [e for _, e in valid]
    try:
        sig, _ = analyzer._calculate_sigma(
            up, ue, linearity_spec or 0.01, travel_length or 0.0, unit_length, model
        )
    except Exception:
        return None
    if sig is None or np.isnan(sig) or sig < 0:
        return None  # respects the CHECK(untrimmed_sigma_gradient >= 0 OR NULL)
    return float(sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="DB path (default: config)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    db = DatabaseManager(Path(args.db)) if args.db else DatabaseManager()
    analyzer = Analyzer(model_thresholds={})

    with db.session() as s:
        q = (s.query(TrackResult.id)
             .filter(TrackResult.untrimmed_sigma_gradient.is_(None))
             .filter(TrackResult.untrimmed_errors.isnot(None)))
        ids = [r[0] for r in q.all()]
    if args.limit:
        ids = ids[:args.limit]
    print(f"Candidate tracks (NULL sigma + arrays present): {len(ids)}"
          f"{' [DRY RUN]' if args.dry_run else ''}")

    updated = skipped = 0
    t0 = time.time()
    for i in range(0, len(ids), args.batch):
        chunk = ids[i:i + args.batch]
        with db.session() as s:
            rows = (s.query(TrackResult, AnalysisResult.model)
                    .join(AnalysisResult, TrackResult.analysis_id == AnalysisResult.id)
                    .filter(TrackResult.id.in_(chunk)).all())
            for tr, model in rows:
                val = compute_untrimmed_sigma(
                    analyzer, tr.untrimmed_positions, tr.untrimmed_errors,
                    tr.linearity_spec, tr.travel_length, tr.unit_length, model)
                if val is None:
                    skipped += 1
                    continue
                if not args.dry_run:
                    tr.untrimmed_sigma_gradient = val
                updated += 1
            if not args.dry_run:
                s.commit()
        done = i + len(chunk)
        if (i // args.batch) % 20 == 0 or done == len(ids):
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(ids)}  computed={updated} skipped={skipped}  "
                  f"({rate:.0f}/s)")

    print(f"\nDone in {time.time() - t0:.0f}s: {updated} backfilled, "
          f"{skipped} skipped (too few points / bad arrays)."
          f"{' NO WRITES (dry run).' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
