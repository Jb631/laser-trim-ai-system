#!/usr/bin/env python
"""Validate Laser Trim Analyzer Installation

This script checks that all required dependencies are installed and
that the package can be imported successfully.
"""

import sys
import importlib
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.10 or higher."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_package_installed(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def check_core_dependencies():
    """Check if core dependencies are installed."""
    print("\nChecking core dependencies...")
    
    core_packages = [
        ("pydantic", "Pydantic"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("sqlalchemy", "SQLAlchemy"),
        ("sklearn", "Scikit-learn"),
        ("customtkinter", "CustomTkinter"),
        ("click", "Click"),
        ("yaml", "PyYAML"),
        ("psutil", "PSUtil"),
        ("httpx", "HTTPX"),
        ("openpyxl", "OpenPyXL"),
        ("PIL", "Pillow"),
        ("cryptography", "Cryptography"),
        ("diskcache", "DiskCache"),
        ("h5py", "H5Py"),
        ("zarr", "Zarr"),
        ("memory_profiler", "Memory Profiler"),
        ("filelock", "FileLock"),
        ("dateutil", "Python-DateUtil"),
        ("pyarrow", "PyArrow"),
    ]
    
    all_installed = True
    for package, display_name in core_packages:
        if check_package_installed(package):
            print(f"  ✅ {display_name}")
        else:
            print(f"  ❌ {display_name} - Run: pip install {package}")
            all_installed = False
    
    return all_installed


def check_laser_trim_analyzer():
    """Check if laser_trim_analyzer package is installed."""
    print("\nChecking laser_trim_analyzer installation...")
    
    try:
        import laser_trim_analyzer
        print("✅ laser_trim_analyzer package found")
        
        # Check version
        if hasattr(laser_trim_analyzer, '__version__'):
            print(f"  Version: {laser_trim_analyzer.__version__}")
        
        # Check submodules
        submodules = [
            "laser_trim_analyzer.core",
            "laser_trim_analyzer.analysis",
            "laser_trim_analyzer.database",
            "laser_trim_analyzer.gui",
            "laser_trim_analyzer.ml",
            "laser_trim_analyzer.utils",
            "laser_trim_analyzer.cli",
            "laser_trim_analyzer.api",
        ]
        
        print("\n  Checking submodules:")
        for submodule in submodules:
            try:
                importlib.import_module(submodule)
                print(f"    ✅ {submodule}")
            except ImportError as e:
                print(f"    ❌ {submodule} - Error: {e}")
        
        return True
        
    except ImportError:
        print("❌ laser_trim_analyzer not found")
        print("  Run: pip install -e . (from the project root)")
        return False


def check_cli_commands():
    """Check if CLI commands are available."""
    print("\nChecking CLI commands...")
    
    commands = [
        ("lta", "--version"),
        ("laser-trim-analyzer", "--version"),
    ]
    
    all_working = True
    for cmd, arg in commands:
        try:
            result = subprocess.run([cmd, arg], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ {cmd} {arg}")
            else:
                print(f"  ❌ {cmd} {arg} - Error: {result.stderr}")
                all_working = False
        except FileNotFoundError:
            print(f"  ❌ {cmd} - Command not found")
            all_working = False
    
    return all_working


def check_optional_dependencies():
    """Check optional dependencies."""
    print("\nChecking optional dependencies...")
    
    optional_packages = [
        ("pytest", "Pytest (testing)"),
        ("tensorflow", "TensorFlow (advanced ML)"),
        ("torch", "PyTorch (advanced ML)"),
        ("transformers", "Transformers (advanced ML)"),
        ("line_profiler", "Line Profiler (performance)"),
        ("notebook", "Jupyter Notebook (development)"),
    ]
    
    for package, display_name in optional_packages:
        if check_package_installed(package):
            print(f"  ✅ {display_name}")
        else:
            print(f"  ⚠️  {display_name} - Optional, install if needed")


def check_system_requirements():
    """Check system requirements."""
    print("\nChecking system requirements...")
    
    # Check tkinter
    try:
        import tkinter
        print("  ✅ Tkinter (GUI support)")
    except ImportError:
        print("  ❌ Tkinter - Required for GUI")
        print("     Install: python3-tk (Linux) or included with Python (Windows/macOS)")
    
    # Check disk space
    try:
        import psutil
        disk_usage = psutil.disk_usage('/')
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 1:
            print(f"  ⚠️  Low disk space: {free_gb:.1f} GB free")
        else:
            print(f"  ✅ Disk space: {free_gb:.1f} GB free")
    except:
        print("  ⚠️  Could not check disk space")
    
    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        print(f"  ✅ Memory: {available_gb:.1f}/{total_gb:.1f} GB available")
    except:
        print("  ⚠️  Could not check memory")


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("Laser Trim Analyzer v2 - Installation Validation")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_core_dependencies(),
        check_laser_trim_analyzer(),
        check_cli_commands(),
    ]
    
    check_optional_dependencies()
    check_system_requirements()
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ All core components are properly installed!")
        print("\nYou can now run:")
        print("  - lta --help          # For CLI usage")
        print("  - lta gui             # To launch the GUI")
        print("  - lta process <file>  # To process a file")
    else:
        print("❌ Some components are missing or not properly installed.")
        print("\nPlease run:")
        print("  pip install -e .")
        print("\nFor full installation with all features:")
        print("  pip install -e '.[all]'")
    print("=" * 60)


if __name__ == "__main__":
    main()