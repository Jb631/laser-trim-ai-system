#!/usr/bin/env python3
"""
Move archival candidates into archive/legacy_YYYYMMDD/.

Usage:
  python scripts/archive_move.py --dry-run  # default, prints planned moves
  python scripts/archive_move.py --confirm  # performs move

Only archives folders whose names start with "_archive" or match "_archive_cleanup_*".
"""

import argparse
import datetime as dt
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def find_candidates():
    candidates = []
    for p in ROOT.iterdir():
        if p.is_dir() and p.name.startswith("_archive"):
            candidates.append(p)
    for p in ROOT.rglob("_archive_cleanup_*"):
        if p.is_dir():
            candidates.append(p)
    # Deduplicate
    unique = []
    seen = set()
    for p in candidates:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm', action='store_true', help='Perform move (not a dry run)')
    args = parser.parse_args()

    candidates = find_candidates()
    if not candidates:
        print("No archival candidates found.")
        return

    target_root = ROOT / 'archive' / f"legacy_{dt.datetime.now():%Y%m%d}"
    target_root.mkdir(parents=True, exist_ok=True)

    for src in candidates:
        rel = src.relative_to(ROOT)
        dest = target_root / rel
        print(f"Archive: {rel} -> {dest.relative_to(ROOT)}")
        if args.confirm:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))

    if not args.confirm:
        print("\nDry run complete. Re-run with --confirm to perform moves.")

if __name__ == '__main__':
    main()

