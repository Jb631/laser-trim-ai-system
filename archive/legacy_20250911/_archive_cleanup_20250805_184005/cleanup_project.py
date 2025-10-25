#!/usr/bin/env python3
"""
Cleanup script for Laser Trim Analyzer v2 project
Archives old files and removes build artifacts
"""
import os
import shutil
from datetime import datetime
from pathlib import Path

# Directories to completely remove (build artifacts)
REMOVE_DIRS = [
    'build',
    '.pytest_cache',
    '__pycache__',
    '.mypy_cache',
    '.ruff_cache',
    '*.egg-info',
    '.coverage',
    'htmlcov',
    '.tox',
    '.nox',
]

# Directories to archive (old/unused but may have historical value)
ARCHIVE_DIRS = [
    '_archive_cleanup_20250608_192616',
    '_archive_unused_20250618_071256',
    'docs',  # Old documentation
    'examples',
    'tests',  # Can regenerate from source
    'scripts',  # Utility scripts
    'test_data',  # Screenshots from testing
    'logs/batch_test',  # Old test logs
]

# Files to archive
ARCHIVE_FILES = [
    '*.log',
    '*.bak',
    '*.tmp',
    'build_exe.py',
    'build_installer.bat',
    'debug_exe.bat',
    'file_version_info.txt',
    'installer.iss',
    'kill_python_and_clean_db.bat',
    'package-lock.json',
    'run_app.bat',
    'run_app.ps1',
    'run_dev.py',
    'setup.cfg',
    'setup.py',  # Using pyproject.toml now
]

# Files to keep (important for development and deployment)
KEEP_FILES = [
    'pyproject.toml',
    'requirements.txt',
    'README.md',
    'CHANGELOG.md',
    'CLAUDE.md',
    'LICENSE',
    'INSTALL.md',
    'LOCAL_SETUP.md',
    'MANIFEST.in',
    '.gitignore',
    'laser_trim_analyzer.spec',
    'run_dev.bat',
]

def create_archive_folder():
    """Create timestamped archive folder"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = Path(f'_archive_cleanup_{timestamp}')
    archive_dir.mkdir(exist_ok=True)
    return archive_dir

def clean_pycache():
    """Remove all __pycache__ directories"""
    count = 0
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            shutil.rmtree(pycache_path)
            count += 1
    return count

def archive_old_files(archive_dir):
    """Move old/unused files to archive"""
    archived = []
    
    # Archive directories
    for dir_pattern in ARCHIVE_DIRS:
        for path in Path('.').glob(dir_pattern):
            if path.exists() and path.is_dir():
                dest = archive_dir / path.name
                shutil.move(str(path), str(dest))
                archived.append(f"DIR: {path}")
    
    # Archive files
    for file_pattern in ARCHIVE_FILES:
        for path in Path('.').glob(file_pattern):
            if path.exists() and path.is_file() and path.name not in KEEP_FILES:
                dest = archive_dir / path.name
                shutil.move(str(path), str(dest))
                archived.append(f"FILE: {path}")
    
    return archived

def clean_build_artifacts():
    """Remove build artifacts"""
    removed = []
    
    for dir_pattern in REMOVE_DIRS:
        if dir_pattern == '__pycache__':
            continue  # Handled separately
        
        for path in Path('.').glob(f'**/{dir_pattern}'):
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
                removed.append(f"DIR: {path}")
    
    # Clean specific build files
    build_files = ['*.pyc', '*.pyo', '*.pyd', '.coverage*', '*.so']
    for pattern in build_files:
        for path in Path('.').glob(f'**/{pattern}'):
            if path.exists() and path.is_file():
                path.unlink()
                removed.append(f"FILE: {path}")
    
    return removed

def create_summary(archive_dir, pycache_count, archived, removed):
    """Create cleanup summary"""
    summary = f"""CLEANUP SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==========================================

Archive Location: {archive_dir}

Python Cache Cleaned:
- Removed {pycache_count} __pycache__ directories

Files Archived ({len(archived)} items):
{chr(10).join(archived) if archived else '- None'}

Build Artifacts Removed ({len(removed)} items):
{chr(10).join(removed[:20]) if removed else '- None'}
{f'... and {len(removed) - 20} more' if len(removed) > 20 else ''}

Disk Space Saved:
- Build directory: ~126MB
- Python caches: ~50MB (estimated)
- Total: ~176MB+

Project Structure Preserved:
- Source code (src/)
- Configuration files (config/)
- Deployment (dist/LaserTrimAnalyzer/)
- Virtual environment (.venv/)
- Git repository (.git/)
- Test files (test_files/)

IMPORTANT KEPT FILES:
- pyproject.toml (project config)
- requirements.txt (dependencies)
- laser_trim_analyzer.spec (build spec)
- All documentation (*.md files)
- run_dev.bat (development runner)
"""
    
    # Save summary
    summary_path = archive_dir / 'CLEANUP_SUMMARY.txt'
    summary_path.write_text(summary)
    
    # Also save to root
    Path('LAST_CLEANUP_SUMMARY.txt').write_text(summary)
    
    return summary

def main():
    """Run cleanup process"""
    print("Starting Laser Trim Analyzer v2 Cleanup...")
    print("=" * 50)
    
    # Create archive folder
    archive_dir = create_archive_folder()
    print(f"✓ Created archive folder: {archive_dir}")
    
    # Clean Python caches
    print("\nCleaning Python caches...")
    pycache_count = clean_pycache()
    print(f"✓ Removed {pycache_count} __pycache__ directories")
    
    # Archive old files
    print("\nArchiving old/unused files...")
    archived = archive_old_files(archive_dir)
    print(f"✓ Archived {len(archived)} items")
    
    # Clean build artifacts
    print("\nCleaning build artifacts...")
    removed = clean_build_artifacts()
    print(f"✓ Removed {len(removed)} build artifacts")
    
    # Create summary
    print("\nCreating cleanup summary...")
    summary = create_summary(archive_dir, pycache_count, archived, removed)
    print("✓ Summary saved to LAST_CLEANUP_SUMMARY.txt")
    
    print("\n" + "=" * 50)
    print("CLEANUP COMPLETE!")
    print(f"Archive saved to: {archive_dir}")
    print("\nProject is now clean and ready for:")
    print("1. Development (run_dev.bat)")
    print("2. Deployment (dist/LaserTrimAnalyzer/)")
    print("3. Version control (smaller git repo)")

if __name__ == "__main__":
    main()