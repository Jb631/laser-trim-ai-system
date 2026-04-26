"""Re-run the snapshot and diff against the baseline.

Reports which files changed and how. Use after each parser fix to confirm
that only the targeted files moved, in the expected direction.

Usage:
    python scripts/parser_audit/diff.py [baseline_path]
"""
import sys
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.parser_audit.snapshot import build_snapshot


# Fields that are sensitive to fix changes
TRACK_FIELDS = [
    "n_points",
    "linearity_error",
    "linearity_spec",
    "sigma_gradient",
    "linearity_fail_points",
    "linearity_pass",
    "status",
]


def diff_track(a: dict, b: dict) -> dict:
    """Return dict of {field: (old, new)} for fields that differ."""
    diffs = {}
    for f in TRACK_FIELDS:
        if a.get(f) != b.get(f):
            diffs[f] = (a.get(f), b.get(f))
    return diffs


def diff_file(rel: str, base: dict, curr: dict) -> dict | None:
    """Return summary of differences for one file, or None if identical."""
    # Top-level changes: skipped, exception, result_none, file_type, status, quality
    top_diffs = {}
    for k in ["file_type", "skipped", "exception", "result_none",
              "overall_status", "data_quality"]:
        if base.get(k) != curr.get(k):
            top_diffs[k] = (base.get(k), curr.get(k))

    # Track-level diffs
    base_tracks = {t.get("track_id"): t for t in (base.get("tracks") or [])}
    curr_tracks = {t.get("track_id"): t for t in (curr.get("tracks") or [])}
    if set(base_tracks) != set(curr_tracks):
        top_diffs["track_ids"] = (sorted(base_tracks), sorted(curr_tracks))

    track_diffs = {}
    for tid in set(base_tracks) | set(curr_tracks):
        a = base_tracks.get(tid, {})
        b = curr_tracks.get(tid, {})
        d = diff_track(a, b)
        if d:
            track_diffs[tid] = d

    if not top_diffs and not track_diffs:
        return None
    return {"top": top_diffs, "tracks": track_diffs}


def main():
    base_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else ROOT / "scripts" / "parser_audit" / "baseline.json"
    )
    if not base_path.exists():
        print(f"ERROR: baseline not found at {base_path}", file=sys.stderr)
        print("Run snapshot.py first.", file=sys.stderr)
        sys.exit(1)

    base = json.loads(base_path.read_text())
    print(f"Baseline loaded: {base['total_files']} files from {base_path.name}\n")

    print("Running current snapshot ...")
    work_root = ROOT / "Work Files"
    curr = build_snapshot(work_root)

    # Diff
    base_files = base["files"]
    curr_files = curr["files"]

    only_in_base = sorted(set(base_files) - set(curr_files))
    only_in_curr = sorted(set(curr_files) - set(base_files))
    common = sorted(set(base_files) & set(curr_files))

    changed = []
    for rel in common:
        d = diff_file(rel, base_files[rel], curr_files[rel])
        if d:
            changed.append((rel, d))

    print(f"\n=== Diff summary ===")
    print(f"Total files (baseline / current): "
          f"{base['total_files']} / {curr['total_files']}")
    print(f"Files only in baseline: {len(only_in_base)}")
    print(f"Files only in current:  {len(only_in_curr)}")
    print(f"Files in both:          {len(common)}")
    print(f"Files with differences: {len(changed)}\n")

    if only_in_base:
        print("Only in baseline (sample of 5):")
        for r in only_in_base[:5]:
            print(f"  {r}")
        print()
    if only_in_curr:
        print("Only in current (sample of 5):")
        for r in only_in_curr[:5]:
            print(f"  {r}")
        print()

    if not changed:
        print("✅ No differences from baseline.")
        return

    print(f"\n=== {len(changed)} changed files ===\n")
    for rel, d in changed:
        print(f"--- {rel}")
        for k, (old, new) in d.get("top", {}).items():
            print(f"    {k}: {old!r} -> {new!r}")
        for tid, fields in d.get("tracks", {}).items():
            print(f"    track[{tid}]:")
            for f, (old, new) in fields.items():
                print(f"      {f}: {old!r} -> {new!r}")


if __name__ == "__main__":
    main()
