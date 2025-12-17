#!/usr/bin/env python3
"""
Inventory script for repository cleanup.

Identifies candidate folders/files for archival based on simple heuristics:
- Names starting with "_archive" (historical backups)
- Deep nested "_archive_cleanup_*" folders

This script only prints candidates; it does not perform moves.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    candidates = []
    # Top-level archive-like folders
    for p in ROOT.iterdir():
        if p.is_dir() and p.name.startswith("_archive"):
            candidates.append(p)

    # Recursively find nested archive cleanup folders
    for p in ROOT.rglob("_archive_cleanup_*"):
        if p.is_dir():
            candidates.append(p)

    # Deduplicate and sort
    unique = sorted(set(candidates))
    if not unique:
        print("No archival candidates found.")
        return

    print("Archival candidates:\n")
    for p in unique:
        try:
            size_mb = sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / (1024 * 1024)
        except Exception:
            size_mb = 0.0
        print(f"- {p.relative_to(ROOT)}  (~{size_mb:.1f} MB)")

    print("\nUse scripts/archive_move.py to move candidates into archive/legacy_YYYYMMDD (dry run by default).")

if __name__ == "__main__":
    main()

