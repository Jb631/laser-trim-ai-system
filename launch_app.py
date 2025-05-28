"""
Laser Trim AI System - One-Click Launcher
=========================================

This script handles complete setup and launch of the application:
- Python version check
- Virtual environment setup
- Dependency installation
- Component verification
- GUI launch

Author: QA Team
Date: 2024
"""

import os
import sys
import subprocess
import platform
import time
import json
import logging
from pathlib import Path
import shutil
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import webbrowser


class LaserTrimLauncher:
    """One-click launcher for Laser Trim AI System."""

    PYTHON_MIN_VERSION = (3, 8)
    VENV_NAME = "venv"
    REQUIREMENTS_FILE = "requirements.txt"
    MAIN_GUI_PATH = Path("src/gui/gui_application.py")

    def __init__(self):
        """Initialize the launcher."""
        self.root_dir = Path(__file__).parent
        self.venv_path = self.root_dir / self.VENV_NAME
        self.log_file = self.root_dir / "launcher.log"
        self.setup_logging()

        # GUI elements
        self.window = None
        self.progress_var = None
        self.status_label = None
        self.progress_bar = None
        self.log_text = None

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log(self, message, level="INFO"):
        """Log message and update GUI if available."""
        getattr(self.logger, level.lower())(message)

        if self.log_text and self.window:
            self.window.after(0, lambda: self._update_log_text(message, level))

    def _update_log_text(self, message, level):
        """Update GUI log text widget."""
        if self.log_text:
            timestamp = time.strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"{timestamp} [{level}] {message}\n")
            self.log_text.see(tk.END)
            self.log_text.update()

    def create_gui(self):
        """Create launcher GUI."""
        self.window = tk.Tk()
        self.window.title("Laser Trim AI System Launcher")
        self.window.geometry("600x500")
        self.window.resizable(False, False)

        # Set icon if available
        icon_path = self.root_dir / "assets" / "icon.ico"
        if icon_path.exists():
            self.window.iconbitmap(str(icon_path))

        # Main frame
        main_frame = tk.Frame(self.window, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = tk.Label(
            main_frame,
            text="Laser Trim AI System",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=(0, 10))

        # Subtitle
        subtitle_label = tk.Label(
            main_frame,
            text="One-Click Launcher",
            font=("Arial", 12)
        )
        subtitle_label.pack(pady=(0, 20))

        # Status label
        self.status_label = tk.Label(
            main_frame,
            text="Ready to launch...",
            font=("Arial", 10)
        )
        self.status_label.pack(pady=(0, 10))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame,
            length=400,
            variable=self.progress_var,
            mode='determinate'
        )
        self.progress_bar.pack(pady=(0, 20))

        # Log text area
        log_frame = tk.Frame(main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            log_frame,
            height=12,
            width=70,
            font=("Consolas", 9),
            bg="#f0f0f0"
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=(20, 0))

        # Launch button
        self.launch_button = tk.Button(
            button_frame,
            text="Launch Application",
            command=self.launch_async,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.launch_button.pack(side=tk.LEFT, padx=(0, 10))

        # Help button
        help_button = tk.Button(
            button_frame,
            text="Help",
            command=self.show_help,
            font=("Arial", 10),
            padx=15,
            pady=8
        )
        help_button.pack(side=tk.LEFT)

        # Center window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() - self.window.winfo_width()) // 2
        y = (self.window.winfo_screenheight() - self.window.winfo_height()) // 2
        self.window.geometry(f"+{x}+{y}")

    def update_progress(self, value, status_text=""):
        """Update progress bar and status."""
        if self.window:
            self.window.after(0, lambda: self._update_progress_gui(value, status_text))

    def _update_progress_gui(self, value, status_text):
        """Update GUI progress elements."""
        if self.progress_var:
            self.progress_var.set(value)
        if self.status_label and status_text:
            self.status_label.config(text=status_text)
        if self.window:
            self.window.update()

    def check_python_version(self):
        """Check if Python version meets requirements."""
        self.log("Checking Python version...")
        self.update_progress(10, "Checking Python version...")

        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        self.log(f"Python version: {version_str}")

        if version < self.PYTHON_MIN_VERSION:
            error_msg = (f"Python {self.PYTHON_MIN_VERSION[0]}.{self.PYTHON_MIN_VERSION[1]}+ required, "
                         f"but {version_str} found.")
            self.log(error_msg, "ERROR")
            raise RuntimeError(error_msg)

        self.log("✓ Python version OK")
        return True

    def check_virtual_env(self):
        """Check and create virtual environment if needed."""
        self.log("Checking virtual environment...")
        self.update_progress(20, "Setting up virtual environment...")

        # Check if we're already in a virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.log("Already running in a virtual environment")
            return True

        # Check if venv exists
        if not self.venv_path.exists():
            self.log(f"Creating virtual environment: {self.VENV_NAME}")
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", str(self.venv_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                self.log("✓ Virtual environment created")
            except subprocess.CalledProcessError as e:
                self.log(f"Failed to create virtual environment: {e}", "ERROR")
                raise
        else:
            self.log("✓ Virtual environment exists")

        return True

    def get_venv_python(self):
        """Get path to Python executable in virtual environment."""
        if platform.system() == "Windows":
            python_path = self.venv_path / "Scripts" / "python.exe"
        else:
            python_path = self.venv_path / "bin" / "python"

        if not python_path.exists():
            raise FileNotFoundError(f"Python executable not found: {python_path}")

        return str(python_path)

    def check_and_install_dependencies(self):
        """Check and install required dependencies."""
        self.log("Checking dependencies...")
        self.update_progress(40, "Installing dependencies...")

        if not Path(self.REQUIREMENTS_FILE).exists():
            self.log(f"Requirements file not found: {self.REQUIREMENTS_FILE}", "WARNING")
            return True

        venv_python = self.get_venv_python()

        # First, upgrade pip
        self.log("Upgrading pip...")
        try:
            subprocess.run(
                [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            self.log(f"Warning: Failed to upgrade pip: {e.stderr}", "WARNING")

        # Check if dependencies are already installed
        self.log("Checking installed packages...")
        result = subprocess.run(
            [venv_python, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            installed_packages = {pkg['name'].lower(): pkg['version']
                                  for pkg in json.loads(result.stdout)}

            # Read requirements
            with open(self.REQUIREMENTS_FILE, 'r') as f:
                requirements = [line.strip() for line in f
                                if line.strip() and not line.startswith('#')]

            # Check if all required packages are installed
            missing_packages = []
            for req in requirements:
                # Parse package name (handle ==, >=, etc.)
                pkg_name = req.split('==')[0].split('>=')[0].split('<=')[0].strip().lower()
                if pkg_name not in installed_packages:
                    missing_packages.append(req)

            if not missing_packages:
                self.log("✓ All dependencies already installed")
                return True
            else:
                self.log(f"Missing packages: {len(missing_packages)}")

        # Install dependencies
        self.log("Installing dependencies (this may take a few minutes)...")
        try:
            process = subprocess.Popen(
                [venv_python, "-m", "pip", "install", "-r", self.REQUIREMENTS_FILE],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )

            # Monitor installation progress
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    if "Successfully installed" in output:
                        self.log(output.strip())
                    elif "Requirement already satisfied" in output:
                        # Don't log these to reduce clutter
                        pass

            return_code = process.poll()
            if return_code != 0:
                error_output = process.stderr.read()
                self.log(f"Dependency installation failed: {error_output}", "ERROR")
                raise RuntimeError("Failed to install dependencies")

        except Exception as e:
            self.log(f"Error installing dependencies: {e}", "ERROR")
            raise

        self.log("✓ Dependencies installed successfully")
        return True

    def verify_components(self):
        """Verify all required components are present."""
        self.log("Verifying components...")
        self.update_progress(70, "Verifying components...")

        # Check main directories
        required_dirs = [
            "src/core",
            "src/ml",
            "src/reporting",
            "src/gui",
            "src/database",
            "config",
            "output",
            "models"
        ]

        for dir_path in required_dirs:
            path = self.root_dir / dir_path
            if not path.exists():
                self.log(f"Creating missing directory: {dir_path}")
                path.mkdir(parents=True, exist_ok=True)

        # Check main files
        required_files = [
            ("src/core/data_processor.py", "Data Processor"),
            ("src/ml/ml_models.py", "ML Models"),
            ("src/reporting/excel_reporter.py", "Excel Reporter"),
            ("src/gui/gui_application.py", "GUI Application"),
            ("src/database/database_manager.py", "Database Manager"),
        ]

        missing_components = []
        for file_path, component_name in required_files:
            if not (self.root_dir / file_path).exists():
                missing_components.append(component_name)
                self.log(f"✗ Missing: {component_name} ({file_path})", "WARNING")
            else:
                self.log(f"✓ Found: {component_name}")

        if missing_components:
            self.log(f"Warning: {len(missing_components)} components missing", "WARNING")
            # Continue anyway - some components might be optional

        # Check/create default config
        config_file = self.root_dir / "config" / "default_config.json"
        if not config_file.exists():
            self.log("Creating default configuration...")
            self.create_default_config(config_file)

        self.log("✓ Component verification complete")
        return True

    def create_default_config(self, config_path):
        """Create default configuration file."""
        default_config = {
            "processing": {
                "filter_sampling_freq": 100,
                "filter_cutoff_freq": 80,
                "gradient_step_size": 3,
                "default_scaling_factor": 24.0
            },
            "output": {
                "save_raw_data": True,
                "save_filtered_data": True,
                "decimal_places": 6
            },
            "ml": {
                "enable_ml_analysis": True,
                "auto_retrain": False,
                "model_path": "models"
            },
            "system": {
                "log_level": "INFO",
                "parallel_processing": True
            }
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)

    def launch_application(self):
        """Launch the main GUI application."""
        self.log("Launching Laser Trim AI System...")
        self.update_progress(90, "Launching application...")

        venv_python = self.get_venv_python()

        # Add project root to PYTHONPATH
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{self.root_dir}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = str(self.root_dir)

        # Launch the GUI
        try:
            if platform.system() == "Windows":
                # On Windows, use CREATE_NEW_CONSOLE to detach from launcher
                subprocess.Popen(
                    [venv_python, str(self.MAIN_GUI_PATH)],
                    env=env,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # On Unix-like systems
                subprocess.Popen(
                    [venv_python, str(self.MAIN_GUI_PATH)],
                    env=env,
                    start_new_session=True
                )

            self.log("✓ Application launched successfully!")
            self.update_progress(100, "Launch complete!")

            # Close launcher after a delay
            if self.window:
                self.window.after(2000, self.window.quit)

        except Exception as e:
            self.log(f"Failed to launch application: {e}", "ERROR")
            raise

    def create_desktop_shortcut(self):
        """Create desktop shortcut (Windows only)."""
        if platform.system() != "Windows":
            return

        try:
            import win32com.client

            desktop = Path.home() / "Desktop"
            shortcut_path = desktop / "Laser Trim AI System.lnk"

            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(str(shortcut_path))

            # Point to the launcher
            shortcut.TargetPath = str(self.root_dir / "launch_app.py")
            shortcut.WorkingDirectory = str(self.root_dir)
            shortcut.Description = "Launch Laser Trim AI System"

            # Set icon if available
            icon_path = self.root_dir / "assets" / "icon.ico"
            if icon_path.exists():
                shortcut.IconLocation = str(icon_path)

            shortcut.save()
            self.log("✓ Desktop shortcut created")

        except ImportError:
            self.log("pywin32 not installed - skipping shortcut creation", "WARNING")
        except Exception as e:
            self.log(f"Failed to create shortcut: {e}", "WARNING")

    def launch_async(self):
        """Launch application in a separate thread."""
        self.launch_button.config(state=tk.DISABLED, text="Launching...")
        thread = threading.Thread(target=self.run_launch_sequence)
        thread.start()

    def run_launch_sequence(self):
        """Run the complete launch sequence."""
        try:
            self.check_python_version()
            self.check_virtual_env()
            self.check_and_install_dependencies()
            self.verify_components()
            self.create_desktop_shortcut()
            self.launch_application()

        except Exception as e:
            self.log(f"Launch failed: {e}", "ERROR")
            if self.window:
                self.window.after(0, lambda: messagebox.showerror(
                    "Launch Failed",
                    f"Failed to launch application:\n\n{str(e)}\n\nCheck launcher.log for details."
                ))
                self.window.after(0, lambda: self.launch_button.config(
                    state=tk.NORMAL,
                    text="Retry Launch"
                ))

    def show_help(self):
        """Show help dialog."""
        help_text = """
Laser Trim AI System Launcher

This launcher automatically:
• Checks Python version (3.8+ required)
• Creates/activates virtual environment
• Installs all dependencies
• Verifies components
• Launches the main application

Requirements:
• Python 3.8 or higher
• 500MB free disk space
• Windows 7 or higher

Troubleshooting:
• Check launcher.log for detailed errors
• Ensure antivirus isn't blocking
• Run as administrator if needed

For more help, see the documentation.
        """
        messagebox.showinfo("Launcher Help", help_text.strip())

    def run(self):
        """Run the launcher."""
        self.create_gui()
        self.log("Laser Trim AI System Launcher started")
        self.log(f"Working directory: {self.root_dir}")
        self.window.mainloop()


def main():
    """Main entry point."""
    # Check if running as script
    if not __name__ == "__main__":
        return

    # Create and run launcher
    launcher = LaserTrimLauncher()

    # If command line arguments provided, run without GUI
    if len(sys.argv) > 1 and sys.argv[1] == "--no-gui":
        try:
            launcher.check_python_version()
            launcher.check_virtual_env()
            launcher.check_and_install_dependencies()
            launcher.verify_components()
            launcher.launch_application()
        except Exception as e:
            print(f"Launch failed: {e}")
            sys.exit(1)
    else:
        # Run with GUI
        launcher.run()


if __name__ == "__main__":
    main()