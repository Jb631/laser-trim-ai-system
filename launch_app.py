#!/usr/bin/env python
"""
Fixed Launcher for Laser Trim AI System
This version looks for files in the correct subdirectories
"""

import sys
import os
from pathlib import Path


def main():
    """Main launcher function."""
    print("=" * 60)
    print("LASER TRIM AI SYSTEM")
    print("=" * 60)
    print()

    # Add subdirectories to Python path
    base_dir = Path(__file__).parent
    sys.path.insert(0, str(base_dir))
    sys.path.insert(0, str(base_dir / "core"))
    sys.path.insert(0, str(base_dir / "examples"))
    sys.path.insert(0, str(base_dir / "gui"))
    sys.path.insert(0, str(base_dir / "excel_reporter"))
    sys.path.insert(0, str(base_dir / "ml_models"))
    sys.path.insert(0, str(base_dir / "database"))

    print("Choose an option:")
    print("1. Launch GUI Application")
    print("2. Process Single File")
    print("3. Batch Process Folder")
    print("4. Run Examples")
    print("5. Exit")

    choice = input("\nEnter your choice (1-5): ")

    if choice == "1":
        try:
            # Try different possible locations for GUI
            try:
                from gui_application import main as gui_main
                gui_main()
            except ImportError:
                try:
                    from gui.gui_application import main as gui_main
                    gui_main()
                except ImportError:
                    try:
                        from gui_application.gui_application import main as gui_main
                        gui_main()
                    except ImportError:
                        print("GUI module not found in any expected location.")
                        print("Let's try the simple GUI instead...")
                        from simple_gui import main as simple_gui_main
                        simple_gui_main()
        except Exception as e:
            print(f"Error launching GUI: {e}")
            input("\nPress Enter to continue...")

    elif choice == "2":
        try:
            # Try to import from different possible locations
            try:
                from data_processor import DataProcessor
            except ImportError:
                try:
                    from core.data_processor import DataProcessor
                except ImportError:
                    from data_processor_minimal import DataProcessor

            processor = DataProcessor()
            file_path = input("Enter Excel file path: ")
            try:
                result = processor.process_file(file_path)
                print(f"\nProcessing complete!")
                print(f"Results: {result}")
            except Exception as e:
                print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")
        input("\nPress Enter to continue...")

    elif choice == "3":
        try:
            # Try to import from different possible locations
            try:
                from data_processor import DataProcessor
            except ImportError:
                try:
                    from core.data_processor import DataProcessor
                except ImportError:
                    from data_processor_minimal import DataProcessor

            processor = DataProcessor()
            folder_path = input("Enter folder path: ")
            try:
                results = processor.batch_process(folder_path)
                print(f"\nProcessed {results.get('total_files', 0)} files")
                print(f"Successful: {results.get('processed', 0)}")
                print(f"Failed: {results.get('failed', 0)}")
            except Exception as e:
                print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")
        input("\nPress Enter to continue...")

    elif choice == "4":
        try:
            # Try different locations for examples
            try:
                from example_usage import main as example_main
                example_main()
            except ImportError:
                try:
                    from examples.example_usage import main as example_main
                    example_main()
                except ImportError:
                    print("Example module not found.")
                    print("Creating a simple example...")
                    print("\nExample: Processing a file")
                    print("result = processor.process_file('data.xlsx')")
                    print("This would analyze your laser trim data and return sigma calculations.")
        except Exception as e:
            print(f"Error: {e}")
        input("\nPress Enter to continue...")

    else:
        print("Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        input("\nPress Enter to exit...")