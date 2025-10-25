# Archive Index

This folder contains legacy or unused files moved from the repository root to keep the project clean and organized.

Move process
- Candidates are identified by `scripts/archive_inventory.py`.
- Use `scripts/archive_move.py` to move them into `archive/legacy_YYYYMMDD/`.
- This preserves history and avoids clutter in the active codebase.

Selection criteria
- Folder names starting with `_archive*` or `_archive_cleanup_*`.
- Obvious backups or unused legacy artifacts.
- Not referenced by the current build (spec), packaging, or docs.

Notes
- Moves are non-destructive for code; no runtime paths should point into archived locations.
- If a moved item is still needed, it can be restored from `archive/legacy_YYYYMMDD/` or Git history.

