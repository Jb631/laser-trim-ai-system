#!/usr/bin/env python3
"""
Verify DB consistency for single-file vs non-turbo batch runs.

Usage examples:
  python scripts/verify_consistency.py --model 8340 --serial A12345 --limit 5
  python scripts/verify_consistency.py --model 8340 --limit 10

Prints recent analyses and key fields to check that single and batch
produce identical risk fields (failure_probability, risk_category).
"""

from pathlib import Path
from typing import Optional
import argparse
from datetime import datetime

from laser_trim_analyzer.core.config import get_config
from laser_trim_analyzer.database.manager import DatabaseManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Filter by model')
    parser.add_argument('--serial', type=str, help='Filter by serial')
    parser.add_argument('--limit', type=int, default=10, help='Max rows to show')
    args = parser.parse_args()

    config = get_config()
    db = DatabaseManager(config)

    results = db.get_historical_data(
        model=args.model,
        serial=args.serial,
        limit=args.limit,
        include_tracks=True
    )

    if not results:
        print('No results found.')
        return

    print(f"Showing up to {args.limit} recent analyses")
    for r in results:
        ts = r.timestamp if hasattr(r, 'timestamp') else None
        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(ts, datetime) else str(ts)
        print(f"\nFile: {r.filename}\n  Model: {r.model}  Serial: {r.serial}  Timestamp: {ts_str}  Status: {r.overall_status.value}")
        for t in r.tracks:
            fp = getattr(t, 'failure_probability', None)
            rc = t.risk_category.value if getattr(t, 'risk_category', None) else None
            sg = getattr(t, 'sigma_gradient', None)
            sp = getattr(t, 'sigma_pass', None)
            print(f"   - Track {t.track_id}: sigma={sg} pass={sp}  failure_prob={fp}  risk={rc}")

    print("\nTip: Process the same file singly and in a non-turbo batch, then re-run this script to compare rows.")

if __name__ == '__main__':
    main()

