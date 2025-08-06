#!/usr/bin/env python3
"""
Archive Old Files Script

This script archives old files that are no longer needed for active development.
It creates a timestamped archive directory and moves specified files there.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import argparse


def create_archive_directory():
    """Create a timestamped archive directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(f"_archive_unused_{timestamp}")
    archive_dir.mkdir(exist_ok=True)
    return archive_dir


def archive_files(dry_run=False):
    """Archive old files that are no longer needed."""
    
    archive_dir = create_archive_directory() if not dry_run else None
    
    # List of files and directories to archive
    files_to_archive = [
        # Configuration files that might cause confusion
        # Keep only development.yaml, production.yaml, and deployment.yaml
        ("config/default.yaml", "config/"),  # Generated default, not used
        
        # Build artifacts
        ("build/", "build_artifacts/"),
        ("dist/", "build_artifacts/"),
        ("laser_trim_analyzer.spec", "build_artifacts/"),
        
        # Note: The following were already cleaned up:
        # - check_phantom_ids.py
        # - verify_database_state.py 
        # - check_production_db.py
        # - clean_production_db.py
        # - debug_exe.bat
        # - Old example files
    ]
    
    archived_count = 0
    
    print("Archive Old Files")
    print("=" * 80)
    
    if dry_run:
        print("DRY RUN MODE - No files will be moved\n")
    else:
        print(f"Archive directory: {archive_dir}\n")
    
    for file_path, target_subdir in files_to_archive:
        source = Path(file_path)
        
        if source.exists():
            if dry_run:
                print(f"Would archive: {source}")
            else:
                # Create subdirectory in archive
                target_dir = archive_dir / target_subdir
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Move the file or directory
                target = target_dir / source.name
                try:
                    shutil.move(str(source), str(target))
                    print(f"✓ Archived: {source} -> {target}")
                    archived_count += 1
                except Exception as e:
                    print(f"✗ Failed to archive {source}: {e}")
        else:
            print(f"- Skip (not found): {source}")
    
    # Create archive summary
    if not dry_run and archived_count > 0:
        summary_path = archive_dir / "ARCHIVE_SUMMARY.md"
        summary_content = f"""# Archive Summary

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Purpose

This archive contains old files that are no longer needed for active development.
These files were archived to keep the main directory clean and organized.

## Contents

- **examples/**: Old example scripts that demonstrated various features
- **debug_scripts/**: Debug batch files and scripts
- **old_test_data/**: Test data files that should be in test_files directory
- **diagnostic_scripts/**: Scripts used for troubleshooting specific issues

## Notes

These files can be safely deleted if no longer needed. They are archived here
for reference in case they contain useful code snippets or patterns.
"""
        
        try:
            summary_path.write_text(summary_content)
            print(f"\n✓ Created archive summary: {summary_path}")
        except Exception as e:
            print(f"\n✗ Failed to create summary: {e}")
    
    print(f"\n{'Would archive' if dry_run else 'Archived'} {archived_count} items")
    
    if dry_run:
        print("\nRun without --dry-run to actually archive files")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Archive old files that are no longer needed"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be archived without actually moving files"
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    archive_files(dry_run=args.dry_run)


if __name__ == "__main__":
    main()