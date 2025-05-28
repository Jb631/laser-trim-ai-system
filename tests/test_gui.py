"""
Test script for GUI application
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test if all imports work"""
    print("Testing imports...")

    try:
        from gui_application.gui_application import LaserTrimAIApp
        print("✓ GUI application import successful")
    except ImportError as e:
        print(f"✗ GUI application import failed: {e}")
        return False

    try:
        from core_engine.data_processor import LaserTrimDataProcessor
        print("✓ Data processor import successful")
    except ImportError as e:
        print(f"✗ Data processor import failed: {e}")
        return False

    try:
        from ml_models.ml_analyzer import MLAnalyzer
        print("✓ ML analyzer import successful")
    except ImportError as e:
        print(f"✗ ML analyzer import failed: {e}")
        return False

    try:
        from excel_reporter.report_generator import ExcelReportGenerator
        print("✓ Report generator import successful")
    except ImportError as e:
        print(f"✗ Report generator import failed: {e}")
        return False

    try:
        import tkinterdnd2
        print("✓ tkinterdnd2 import successful")
    except ImportError as e:
        print(f"✗ tkinterdnd2 import failed: {e}")
        print("  Install with: pip install tkinterdnd2")
        return False

    return True


def test_gui_creation():
    """Test if GUI can be created"""
    print("\nTesting GUI creation...")

    try:
        from gui_application.gui_application import LaserTrimAIApp

        # Create app instance
        app = LaserTrimAIApp()
        print("✓ GUI application created successfully")

        # Test window properties
        assert app.root.title() == "Laser Trim AI System - QA Analysis"
        print("✓ Window title set correctly")

        # Test main components exist
        assert hasattr(app, 'notebook')
        assert hasattr(app, 'file_listbox')
        assert hasattr(app, 'analyze_button')
        print("✓ Main components created")

        # Close the window
        app.root.destroy()
        print("✓ GUI closed successfully")

        return True

    except Exception as e:
        print(f"✗ GUI creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_analysis():
    """Test with mock data"""
    print("\nTesting with mock data...")

    try:
        from gui_application.gui_application import LaserTrimAIApp
        from unittest.mock import MagicMock

        # Create app with mocked components
        app = LaserTrimAIApp()

        # Mock the processor
        app.processor.process_file = MagicMock(return_value={
            'filename': 'test.xlsx',
            'overall_status': 'PASS',
            'analysis_results': {
                'sigma_gradient': 0.0234,
                'sigma_threshold': 0.0400
            }
        })

        # Add a test file
        app.loaded_files = ['test.xlsx']
        app.file_listbox.insert(tk.END, 'test.xlsx')

        print("✓ Mock setup complete")

        # Close
        app.root.destroy()
        return True

    except Exception as e:
        print(f"✗ Mock test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("GUI Application Test Suite")
    print("=" * 50)

    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 7):
        print("WARNING: Python 3.7+ recommended")

    # Run tests
    tests_passed = 0
    tests_total = 0

    # Test 1: Imports
    tests_total += 1
    if test_imports():
        tests_passed += 1

    # Test 2: GUI Creation
    tests_total += 1
    if test_gui_creation():
        tests_passed += 1

    # Test 3: Mock Analysis
    tests_total += 1
    if test_mock_analysis():
        tests_passed += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        print("✓ All tests passed!")

        # Ask to run the actual GUI
        root = tk.Tk()
        root.withdraw()
        if messagebox.askyesno("Test Complete", "All tests passed! Would you like to launch the GUI?"):
            from gui_application.gui_application import main as run_gui
            run_gui()
    else:
        print("✗ Some tests failed")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure all modules are in the correct directories")
        print("3. Check Python version (3.7+ required)")


if __name__ == "__main__":
    main()