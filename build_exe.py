#!/usr/bin/env python3
"""
Build script for creating Laser Trim Analyzer executable
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess

def clean_build_dirs():
    """Clean previous build artifacts"""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            print(f"Cleaning {dir_name}/...")
            shutil.rmtree(dir_name)

def build_exe():
    """Build the executable using PyInstaller"""
    print("Building Laser Trim Analyzer executable...")
    
    # Clean previous builds
    clean_build_dirs()
    
    # Build command
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "laser-trim-analyzer.spec",
        "--clean",
        "--noconfirm"
    ]
    
    # Run build
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n✅ Build successful!")
        print(f"Executable location: dist/laser-trim-analyzer-gui/laser-trim-analyzer-gui.exe")
        
        # Create a shortcut batch file
        shortcut_content = "@echo off\ncd /d %~dp0\nstart dist\\laser-trim-analyzer-gui\\laser-trim-analyzer-gui.exe"
        with open("LaserTrimAnalyzer.bat", "w") as f:
            f.write(shortcut_content)
        print("Created LaserTrimAnalyzer.bat shortcut")
        
    else:
        print("\n❌ Build failed!")
        print("Error output:")
        print(result.stderr)
        return 1
    
    return 0

def quick_build():
    """Quick build without cleaning (faster)"""
    print("Quick building (no clean)...")
    cmd = [sys.executable, "-m", "PyInstaller", "laser-trim-analyzer.spec"]
    subprocess.run(cmd)

if __name__ == "__main__":
    if "--quick" in sys.argv:
        quick_build()
    else:
        build_exe()