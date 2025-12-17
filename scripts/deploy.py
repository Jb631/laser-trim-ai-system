#!/usr/bin/env python3
"""
Deployment Script for Laser Trim Analyzer V3

Creates a versioned deployment package with proper folder structure.
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def get_version():
    """Get version from pyproject.toml"""
    import tomllib

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    return data["project"]["version"]

def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        import PyInstaller
        return True
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        return True

def run_pyinstaller():
    """Run PyInstaller to build the application"""
    root_dir = Path(__file__).parent.parent
    spec_file = root_dir / "laser_trim_analyzer.spec"

    if not spec_file.exists():
        print(f"Error: Spec file not found at {spec_file}")
        return False

    check_pyinstaller()

    print("Building application with PyInstaller...")
    print(f"Using spec file: {spec_file}")

    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(spec_file), "--clean"],
        cwd=str(root_dir),
        check=False
    )

    return result.returncode == 0

def create_deployment_package():
    """Create the final deployment package"""
    version = get_version()
    timestamp = datetime.now().strftime("%Y%m%d")

    root_dir = Path(__file__).parent.parent
    dist_dir = root_dir / "dist"

    # Source folder created by PyInstaller
    source_folder = dist_dir / f"LaserTrimAnalyzer-v{version}"

    # Target deployment folder
    deploy_folder = root_dir / f"LaserTrimAnalyzer-v{version}-{timestamp}"

    if deploy_folder.exists():
        shutil.rmtree(deploy_folder)

    print(f"Creating deployment package: {deploy_folder.name}")

    # Copy the built application
    if source_folder.exists():
        shutil.copytree(source_folder, deploy_folder / "LaserTrimAnalyzer")
    else:
        print(f"Error: Built application not found at {source_folder}")
        return False

    # Copy additional deployment files (V3 doesn't need config/*.yaml)
    deployment_files = [
        ("README.md", "README.md"),
        ("CHANGELOG.md", "CHANGELOG.md"),
    ]

    for src, dst in deployment_files:
        src_path = root_dir / src
        dst_path = deploy_folder / dst

        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if src_path.is_file():
                shutil.copy2(src_path, dst_path)
            else:
                shutil.copytree(src_path, dst_path)

    # Create deployment instructions
    instructions = f"""# Laser Trim Analyzer v{version} - Deployment Instructions

## Quick Start
1. Extract this folder to your desired location (e.g., C:\\LaserTrimAnalyzer\\)
2. Run LaserTrimAnalyzer\\LaserTrimAnalyzer.exe
3. No installation required!

## Version Information
- Version: {version}
- Build Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Package: LaserTrimAnalyzer-v{version}-{timestamp}

## V3 Features
- Streamlined 5-page interface (Dashboard, Process, Analyze, Trends, Settings)
- Self-contained configuration (no external config files needed)
- SQLite database stored in ./data/ relative to app
- Excel-only export for clean reports
- ML-integrated threshold optimization

## System Requirements
- Windows 10/11 (64-bit)
- No Python installation required
- 4GB RAM minimum, 8GB recommended for large batch processing
- 500MB disk space

## Database Location
The database is stored in ./data/analysis.db relative to the application.
User settings are stored in ~/.laser_trim_analyzer/config.yaml

## Troubleshooting
### Application won't start
- Check Windows Event Viewer for errors
- Ensure no antivirus blocking the executable
- Try running as administrator (not normally required)

### Database errors
- Check file permissions on data folder
- The app will create the data folder automatically

## File Structure
LaserTrimAnalyzer-v{version}-{timestamp}/
+-- LaserTrimAnalyzer/           # Main application folder
|   +-- LaserTrimAnalyzer.exe    # Main executable
|   +-- data/                    # Database folder (created on first run)
|   +-- ...                      # Supporting files
+-- README.md                    # User documentation
+-- CHANGELOG.md                 # Version history
+-- DEPLOYMENT.txt               # This file

## IT Deployment Notes
- No registry modifications required
- No services or drivers installed
- Can be deployed via network share or USB
- All settings in user profile, no admin rights needed

## Updates
To update:
1. Keep backup of old folder
2. Replace entire LaserTrimAnalyzer folder with new version
3. Database in ./data/ is preserved if kept in place
"""

    # Write deployment instructions
    with open(deploy_folder / "DEPLOYMENT.txt", "w", encoding="utf-8") as f:
        f.write(instructions)

    print(f"[OK] Deployment package created: {deploy_folder}")
    print(f"[INFO] Package size: {get_folder_size(deploy_folder):.1f} MB")

    return True

def get_folder_size(folder_path):
    """Calculate folder size in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB

def main():
    """Main deployment function"""
    print("Starting Laser Trim Analyzer V3 Deployment")
    print("=" * 50)

    version = get_version()
    print(f"Version: {version}")

    try:
        # Step 1: Build with PyInstaller
        print("\n[STEP 1] Building application...")
        if not run_pyinstaller():
            print("[ERROR] Build failed!")
            return 1

        # Step 2: Create deployment package
        print("\n[STEP 2] Creating deployment package...")
        if not create_deployment_package():
            print("[ERROR] Package creation failed!")
            return 1

        print("\n[SUCCESS] Deployment completed!")
        print("Ready for distribution")

        return 0

    except Exception as e:
        print(f"[ERROR] Deployment failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
