#!/usr/bin/env python3
"""One-time backfill of the linearity error MAGNITUDE columns.

Recovers the number that analyzer._calculate_linearity used to throw away.
Until 2026-08-31 the magnitude was taken with `max(abs(e) for e in errors)`,
and Python's max() returns NaN whenever the FIRST element is NaN. Models
whose files open with an unmeasured lead-in — 8232-1 above all — therefore
produced a NaN magnitude on every track, which BaseAnalysisModel's
NaN->None coercion turned into None and SQLite stored as NULL. The verdict
columns (linearity_pass, fail_points, spec, offset) were unaffected, so
nothing looked broken while the zero-tolerance customer metric went blind.

The measurements themselves were never lost: position_data / error_data /
optimal_offset are all still in the DB, so the magnitude is recomputable
exactly. This fills it in using the SAME analyzer helper that processing
now uses, so backfilled values are identical to freshly-processed ones.

SAFE: idempotent; only fills rows where the magnitude IS NULL and the
arrays plus the stored offset are present, so it reconstructs rather than
recomputes — the offset the analyzer chose at the time is reused, never
re-derived. Never touches a verdict column, never touches UNTRIMMED rows.
Back up the DB first anyway:
    cp data/analysis.db data/analysis.db.bak-$(date +%F)-pre-linerror-fix

Usage:
    python3 scripts/backfill_linearity_error.py --dry-run     # count only
    python3 scripts/backfill_linearity_error.py               # default DB
    python3 scripts/backfill_linearity_error.py --db /path/to/analysis.db
    python3 scripts/backfill_linearity_error.py --limit 500   # rehearsal
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laser_trim_analyzer.core.analyzer import max_abs_measured
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import AnalysisResult, TrackResult


def recompute(track):
    """Return (shifted_magnitude, raw_magnitude) or (None, None) if unusable.

    Mirrors core/analyzer.py _calculate_linearity exactly: the shift is
    `error + theory*k + offset` when a theory rotation was applied, and
    `error + offset` otherwise.
    """
    errors = track.error_data
    offset = track.optimal_offset
    if not errors or offset is None:
        return None, None

    k = track.optimal_slope or 0.0
    theory = track.theory_data
    if k and theory and len(theory) >= len(errors):
        shifted = []
        for i, e in enumerate(errors):
            if e is None:
                shifted.append(None)
                continue
            t = theory[i]
            shifted.append(e + (t * k if t is not None else 0.0) + offset)
    else:
        shifted = [None if e is None else e + offset for e in errors]

    return max_abs_measured(shifted), max_abs_measured(errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="DB path (default: config)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    db = DatabaseManager(Path(args.db)) if args.db else DatabaseManager()

    with db.session() as s:
        ids = [r[0] for r in (
            s.query(TrackResult.id)
            .filter(TrackResult.final_linearity_error_shifted.is_(None))
            .filter(TrackResult.status != "UNTRIMMED")
            .filter(TrackResult.error_data.isnot(None))
            .filter(TrackResult.optimal_offset.isnot(None))
            .all())]
    if args.limit:
        ids = ids[:args.limit]
    print(f"Candidate tracks (NULL magnitude + arrays + offset present): {len(ids)}"
          f"{' [DRY RUN]' if args.dry_run else ''}")

    updated = skipped = 0
    per_model = Counter()
    t0 = time.time()
    for i in range(0, len(ids), args.batch):
        chunk = ids[i:i + args.batch]
        with db.session() as s:
            rows = (s.query(TrackResult, AnalysisResult.model)
                    .join(AnalysisResult, TrackResult.analysis_id == AnalysisResult.id)
                    .filter(TrackResult.id.in_(chunk)).all())
            for tr, model in rows:
                shifted, raw = recompute(tr)
                if shifted is None:
                    # Nothing measured on the whole sweep — leave it NULL.
                    # Manufacturing a 0.0 here would report a flawless part.
                    skipped += 1
                    continue
                if not args.dry_run:
                    tr.final_linearity_error_shifted = shifted
                    tr.optimized_linearity_error = shifted
                    tr.max_deviation = shifted
                    if raw is not None and tr.raw_linearity_error is None:
                        tr.raw_linearity_error = raw
                updated += 1
                per_model[model] += 1
            if not args.dry_run:
                s.commit()
        done = i + len(chunk)
        if (i // args.batch) % 5 == 0 or done == len(ids):
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(ids)}  recovered={updated} skipped={skipped}  "
                  f"({rate:.0f}/s)")

    print(f"\nDone in {time.time() - t0:.0f}s: {updated} magnitudes recovered, "
          f"{skipped} left NULL (no measured point on the sweep)."
          f"{' NO WRITES (dry run).' if args.dry_run else ''}")
    if per_model:
        print("\nBy model:")
        for model, n in per_model.most_common():
            print(f"  {str(model):14s} {n:6d}")

    # Say out loud what this script CANNOT reach, so the remaining NULLs are
    # a known quantity rather than a silent residue. Rows whose
    # optimal_offset is itself NULL were hit by the same NaN leak one layer
    # down (a NaN error at a limited index poisoned the offset search).
    # Reconstructing those would mean re-deriving the offset, which needs the
    # per-model exclude_points spec — that is processing's job, not a
    # backfill's, so they are reported instead of guessed at.
    with db.session() as s:
        stranded = (s.query(TrackResult, AnalysisResult.model)
                    .join(AnalysisResult, TrackResult.analysis_id == AnalysisResult.id)
                    .filter(TrackResult.final_linearity_error_shifted.is_(None))
                    .filter(TrackResult.status != "UNTRIMMED")
                    .filter(TrackResult.error_data.isnot(None))
                    .filter(TrackResult.optimal_offset.is_(None))
                    .all())
    if stranded:
        by_model = Counter(m for _tr, m in stranded)
        print(f"\nStill NULL after this backfill: {len(stranded)} track(s) whose "
              f"optimal_offset is also missing.\n"
              f"  These need REPROCESSING (the offset must be re-derived with the "
              f"model's exclude_points), not a backfill.\n"
              f"  Affected models: "
              + ", ".join(f"{m}={n}" for m, n in by_model.most_common(8)))


if __name__ == "__main__":
    main()
